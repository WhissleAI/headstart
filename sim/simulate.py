"""Offline policy simulator: replay recorded Arc ear traces into the REAL Session/Speculator in virtual time.

  .venv/bin/python -m sim.simulate --prefetch                 # fill runs/sim_cache (Wikipedia docs + vectors) once
  .venv/bin/python -m sim.simulate --validate                 # current config vs sim/data/ab_full2.json (copy of the real A/B)
  .venv/bin/python -m sim.simulate --set SPEC_WEB_BUDGET=4 --profile tavily --seeds 5

What is real: Arc's words, revisions and their wall timing (sim/data/ear_traces, recorded once by sim/record_traces.py),
the audio RMS the silence detector sees, Arc Tagger, frames, the speculator, the session's semantic endpointing, the
answer.Stream hedging logic, Wikipedia documents, EmbeddingGemma vectors and the reranker.
What is sampled: every latency (web, query/doc embedding, LLM TTFT) - see sim/fakes.py; time is virtual
(sim/vclock.py), so one question takes ~20-60 ms of CPU instead of ~6 s of real time.

Metrics per Head Start answer (beyond ready_ms and path):
  grounded   : the answer's sources include a document returned by a search for the FINAL query
  risky      : not grounded and not an exact-meaning draft (pool-commit, or a draft/answer built only from searches
               of an earlier meaning) - the accuracy proxy we penalize
  bl_top1    : Head Start's top source is among the baseline's top 3 sources for the same question (same seed)
  cost       : web calls, LLM requests (drafts + finals + hedges), drafts started, drafts wasted
"""
from __future__ import annotations

import os
import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim import fakes, vclock  # noqa: E402

TRACES = ROOT / "sim" / "data" / "ear_traces"     # recorded once by sim/record_traces.py (committed)

# sweepable knobs: env var name -> server.config attribute
KNOBS = {
    "SPEC_STABLE_UPDATES": "STABLE_UPDATES", "SPEC_WEB_BUDGET": "WEB_BUDGET", "SPEC_DRAFT_BUDGET": "DRAFT_BUDGET",
    "SEM_SILENCE_MS": "SEM_SILENCE_MS", "SEM_STABLE_MS": "SEM_STABLE_MS", "SEM_COVER_S": "SEM_COVER_S",
    "HEDGE_MS": "HEDGE_MS", "PENDING_CAP_S": "PENDING_CAP_S",
}
CURRENT = {"SPEC_STABLE_UPDATES": 1, "SPEC_WEB_BUDGET": 6, "SPEC_DRAFT_BUDGET": 6, "SEM_SILENCE_MS": 250,
           "SEM_STABLE_MS": 420, "SEM_COVER_S": 0.3, "HEDGE_MS": 800, "PENDING_CAP_S": 1.0}


def load_traces(ids=None, include_current: bool = False, trace_dir: Path = TRACES) -> list[dict]:
    out = []
    for p in sorted(trace_dir.glob("*.json")):
        t = json.loads(p.read_text())
        if ids and t["id"] not in ids:
            continue
        if not ids and not include_current and t.get("kind") == "current":
            continue          # same set as eval/run_ab.py on the Wikipedia lane (48 questions)
        out.append(t)
    return out


class Simulator:
    def __init__(self, caches: fakes.Caches | None = None, setattr_fn=None):
        self.caches = caches or fakes.Caches()
        self.world = {"w": None}
        self._orig = []

        def _set(obj, name, val):
            self._orig.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)
        fakes.install(setattr_fn or _set, self.world)

    def uninstall(self):
        for obj, name, val in reversed(self._orig):
            setattr(obj, name, val)
        self._orig.clear()

    # ------------------------------------------------------------------------------------------------ one run
    async def _replay(self, trace: dict, mode: str):
        from server.session import Session
        events: list[dict] = []

        def emit(e):
            e["_v"] = time.perf_counter()
            events.append(e)

        t0 = time.perf_counter()
        sess = Session(emit, mode=mode, t0=t0)
        sess.t0 = sess.spec.t0 = t0
        finals = []
        for e in trace["events"]:
            d = t0 + e["w"] - time.perf_counter()
            if d > 0:
                await asyncio.sleep(d)
            k = e["k"]
            if k == "audio":
                sess.on_audio(np.full(e["n"], e["rms"], np.float32), e["at"])
            elif k == "partial":
                sess.on_partial(e["words"], time.perf_counter())
            else:
                finals.append(asyncio.create_task(sess.on_final(e["words"])))
        if finals:
            await asyncio.gather(*finals, return_exceptions=True)
        seen = set()
        while getattr(sess, "final_task", None) is not None and id(sess.final_task) not in seen:
            seen.add(id(sess.final_task))
            await asyncio.gather(sess.final_task, return_exceptions=True)
        sess.close()
        errs = [f"{type(t.exception()).__name__}: {t.exception()}" for t in finals
                if t.done() and not t.cancelled() and t.exception() is not None]
        return sess, events, errs

    def run_question(self, trace: dict, mode: str, cfg: dict, profile: fakes.Profile, seed: int = 0) -> dict:
        from server import config
        saved = {a: getattr(config, a) for a in KNOBS.values()}
        saved["SEMANTIC_ENDPOINT"] = config.SEMANTIC_ENDPOINT
        for k, v in cfg.items():
            setattr(config, KNOBS.get(k, k), v)
        w = fakes.World(profile=profile, sampler=fakes.Sampler(seed, trace["id"]), caches=self.caches)
        self.world["w"] = w
        try:
            sess, events, errs = vclock.run(self._replay, trace, mode)
        except Exception as e:      # a deadlock / crash in one run must not kill a sweep: record it
            return {"id": trace["id"], "kind": trace.get("kind"), "mode": mode, "seed": seed, "ready_ms": None,
                    "sources_ms": None, "done_ms": None, "endpoint_lag_ms": None, "endpoint_by": None, "path": None,
                    "draft_match": None, "draft_reused": False, "query": None, "question": None, "answers": 0,
                    "n_endpoints": 0, "web_calls": 0, "drafts": 0, "spec_fires": 0, "llm_requests": 0,
                    "web_missing": 0, "src_ids": [], "grounded": False, "risky": True, "hedged": False,
                    "error": f"sim: {type(e).__name__}: {e}"[:300]}
        finally:
            for a, v in saved.items():
                setattr(config, a, v)
        ms = [m for m in sess.history if m]
        m = ms[-1] if ms else {}
        src = [e for e in events if e.get("t") == "sources"]
        src_ids = [d["id"] for d in src[-1]["docs"]] if src else []
        fq = m.get("query")
        finalq_ids = set()
        for e in events:
            if e.get("t") == "spec" and e.get("state") == "done" and e.get("lane") != "draft" and e.get("query") == fq:
                finalq_ids |= set(e.get("ids") or [])
        grounded = bool(set(src_ids) & finalq_ids)
        exact = bool(m.get("draft_reused")) and m.get("draft_match") == "exact"
        eps = [e for e in events if e.get("t") == "endpoint"]
        arc_end = eps[-1]["speech_end_ms"] / 1000 if eps else None
        ready = m.get("ready_ms")
        # ready measured from the waveform's speech end (independent of Arc's word-timing jitter between runs)
        ready_vad = (round(ready + (arc_end - trace["vad_end_s"]) * 1000)
                     if ready is not None and arc_end is not None and trace.get("vad_end_s") is not None else None)
        return {
            "id": trace["id"], "kind": trace.get("kind"), "mode": mode, "seed": seed,
            "ready_ms": ready, "ready_vad_ms": ready_vad, "arc_end_s": arc_end, "sources_ms": m.get("sources_ms"), "done_ms": m.get("done_ms"),
            "endpoint_lag_ms": m.get("endpoint_lag_ms"), "endpoint_by": m.get("endpoint_by"), "path": m.get("path"),
            "draft_match": m.get("draft_match"), "draft_reused": bool(m.get("draft_reused")),
            "query": fq, "question": m.get("question"), "answers": len(ms), "n_endpoints": len(eps),
            "web_calls": m.get("web_calls", 0), "drafts": m.get("drafts", 0), "spec_fires": m.get("spec_fires", 0),
            "llm_requests": w.counters["llm_requests"], "web_missing": w.counters["web_missing"],
            "src_ids": src_ids, "grounded": grounded, "risky": (not grounded and not exact) if m else True,
            "hedged": bool(m.get("hedged")), "spin_ms": round(vclock.last_spin_s * 1000), "error": "; ".join(errs) or m.get("error") or (None if m else "no answer"),
        }

    def run_set(self, traces, cfg: dict, profile: fakes.Profile, seeds=(0,), modes=("headstart",),
                bl_rows: dict | None = None) -> list[dict]:
        rows = []
        for seed in seeds:
            for t in traces:
                for mode in modes:
                    rows.append(self.run_question(t, mode, cfg, profile, seed))
        if bl_rows:
            for r in rows:
                b = bl_rows.get((r["id"], r["seed"]))
                if r["mode"] == "headstart" and b is not None:
                    r["bl_top1"] = bool(r["src_ids"]) and r["src_ids"][0] in b["src_ids"][:3]
        return rows


# ---------------------------------------------------------------------------------------------------- summaries
def pct(xs, p):
    xs = [x for x in xs if x is not None]
    return round(float(np.percentile(xs, p))) if xs else None


def summarize(rows: list[dict], mode: str = "headstart") -> dict:
    rs = [r for r in rows if r["mode"] == mode]
    if not rs:
        return {}
    ready = [r["ready_ms"] for r in rs]
    n = len(rs)
    seeds = sorted({r["seed"] for r in rs})
    # p50/p90 per seed, then averaged (a seed = one pass over the question set, like one real A/B)
    p50s = [pct([r["ready_ms"] for r in rs if r["seed"] == s], 50) for s in seeds]
    p90s = [pct([r["ready_ms"] for r in rs if r["seed"] == s], 90) for s in seeds]
    drafts = sum(r["drafts"] for r in rs)
    reused = sum(r["draft_reused"] for r in rs)
    out = {
        "n": n, "seeds": len(seeds),
        "p50_ready_ms": round(statistics.mean(p50s)), "p90_ready_ms": round(statistics.mean(p90s)),
        "p50_ready_ms_sd": round(statistics.pstdev(p50s)) if len(p50s) > 1 else 0,
        "mean_ready_ms": round(statistics.mean([x for x in ready if x is not None])),
        "p50_ready_vad_ms": pct([r.get("ready_vad_ms") for r in rs], 50),
        "p50_sources_ms": pct([r["sources_ms"] for r in rs], 50),
        "p50_endpoint_lag_ms": pct([r["endpoint_lag_ms"] for r in rs], 50),
        "paths": {k: round(v / n, 3) for k, v in Counter(str(r["path"]) for r in rs).most_common()},
        "endpoint_by": {k: round(v / n, 3) for k, v in Counter(str(r["endpoint_by"]) for r in rs).most_common()},
        "draft_reuse_rate": round(reused / n, 3),
        "grounded_rate": round(sum(r["grounded"] for r in rs) / n, 3),
        "risky_rate": round(sum(r["risky"] for r in rs) / n, 3),
        "pool_commit_rate": round(sum(r["path"] == "pool-commit" for r in rs) / n, 3),
        "completion_reuse_rate": round(sum(r["draft_match"] == "completion" and r["draft_reused"] for r in rs) / n, 3),
        "web_calls_per_q": round(sum(r["web_calls"] for r in rs) / n, 2),
        "llm_requests_per_q": round(sum(r["llm_requests"] for r in rs) / n, 2),
        "drafts_per_q": round(drafts / n, 2),
        "wasted_drafts_per_q": round((drafts - reused) / n, 2),
        "answered_twice": round(sum(r["answers"] > 1 for r in rs) / n, 3),
        "errors": sum(1 for r in rs if r["error"] or r["ready_ms"] is None),
        "hedged_rate": round(sum(r["hedged"] for r in rs) / n, 3),
        "busy_spin_rate": round(sum(r.get("spin_ms", 0) > 0 for r in rs) / n, 3),
        "web_missing": sum(r["web_missing"] for r in rs),
    }
    if any("bl_top1" in r for r in rs):
        out["bl_top1_rate"] = round(sum(r.get("bl_top1", False) for r in rs) / n, 3)
    return out


# ---------------------------------------------------------------------------------------------------- validation
def validate(sim: Simulator, seeds=range(10), ab=ROOT / "sim" / "data" / os.environ.get("SIM_AB", "ab_full3.json")) -> dict:
    real = json.loads(Path(ab).read_text())
    ids = {r["id"] for r in real["rows"]}
    traces = [t for t in load_traces(ids=ids)]
    prof = fakes.PROFILES["wikipedia"]
    rows = sim.run_set(traces, CURRENT, prof, seeds=list(seeds), modes=("headstart", "baseline"))
    bl = {(r["id"], r["seed"]): r for r in rows if r["mode"] == "baseline"}
    for r in rows:
        b = bl.get((r["id"], r["seed"]))
        if r["mode"] == "headstart" and b:
            r["bl_top1"] = bool(r["src_ids"]) and r["src_ids"][0] in b["src_ids"][:3]
    out = {"n_questions": len(traces), "seeds": len(list(seeds)), "modes": {}}
    for mode in ("headstart", "baseline"):
        s = summarize(rows, mode)
        rr = [r for r in real["rows"] if r["mode"] == mode]
        rs = real["summary"][mode]
        real_paths = {k: round(v / len(rr), 3) for k, v in Counter(str(r["path"]) for r in rr).most_common()}
        real_ep = {k: round(v / len(rr), 3) for k, v in Counter(str(r["endpoint_by"]) for r in rr).most_common()}
        # per-question: sim median over seeds vs the single real measurement
        per_q = []
        for r in rr:
            xs = [x["ready_ms"] for x in rows if x["mode"] == mode and x["id"] == r["id"] and x["ready_ms"] is not None]
            if xs and r["ready_ms"] is not None:
                per_q.append((r["id"], r["ready_ms"], float(np.median(xs)), r["path"],
                              Counter(x["path"] for x in rows if x["mode"] == mode and x["id"] == r["id"]).most_common(1)[0][0]))
        err = [b - a for _, a, b, _, _ in per_q]
        out["modes"][mode] = {
            "sim": s, "real": {k: rs.get(k) for k in ("p50_ready_ms", "p90_ready_ms", "mean_ready_ms", "p50_ready_vad_ms", "p50_sources_ms",
                                                      "p50_endpoint_lag_ms", "mean_web_calls", "draft_reuse_rate")},
            "real_paths": real_paths, "real_endpoint_by": real_ep,
            "p50_err_ms": s["p50_ready_ms"] - rs["p50_ready_ms"], "p90_err_ms": s["p90_ready_ms"] - rs["p90_ready_ms"],
            "p50_vad_err_ms": (s["p50_ready_vad_ms"] - rs["p50_ready_vad_ms"]) if s.get("p50_ready_vad_ms") else None,
            "p50_rel_err": round((s["p50_ready_ms"] - rs["p50_ready_ms"]) / rs["p50_ready_ms"], 3),
            "per_q_mae_ms": round(float(np.mean(np.abs(err)))) if err else None,
            "per_q_bias_ms": round(float(np.mean(err))) if err else None,
            "per_q_corr": round(float(np.corrcoef([a for _, a, _, _, _ in per_q], [b for _, _, b, _, _ in per_q])[0, 1]), 3)
            if len(per_q) > 2 else None,
            "path_agreement": round(sum(1 for *_, a, b in per_q if a == b) / len(per_q), 3) if per_q else None,
            "path_l1": round(sum(abs(s["paths"].get(k, 0) - real_paths.get(k, 0)) for k in set(s["paths"]) | set(real_paths)), 3),
        }
    hs = {(r["id"], r["seed"]): r["ready_ms"] for r in rows if r["mode"] == "headstart"}
    d = [bl[k]["ready_ms"] - v for k, v in hs.items() if k in bl and v is not None and bl[k]["ready_ms"] is not None]
    out["paired_sim"] = {"p50_saved_ms": pct(d, 50), "hs_faster_rate": round(sum(x > 0 for x in d) / len(d), 3)}
    out["paired_real"] = real["summary"].get("paired")
    out["per_question"] = [{"id": r["id"], "path_real": r["path"],
                            "ready_real": r["ready_ms"], "endpoint_real": r["endpoint_by"]} for r in real["rows"]
                           if r["mode"] == "headstart"]
    return out, rows


def prefetch(sim: Simulator, rounds: int = 4, seeds=(0, 1, 2)):
    """Fill the Wikipedia + vector caches: run every trace with a permissive policy (fire on every frame, large
    budget) in both modes and with/without semantic endpointing, fetch whatever was missing, repeat."""
    traces = load_traces(include_current=True)
    prof = fakes.PROFILES["wikipedia"]
    cfgs = [dict(CURRENT, SPEC_WEB_BUDGET=99, SPEC_STABLE_UPDATES=1)]
    cfgs += [dict(cfgs[0], SEMANTIC_ENDPOINT=False)]
    cfgs += [dict(CURRENT, SEM_SILENCE_MS=s, SEM_STABLE_MS=st, SEM_COVER_S=c, SPEC_WEB_BUDGET=99,
                  SPEC_STABLE_UPDATES=su)
             for s in (150, 350) for st in (300, 550) for c in (0.2, 0.4) for su in (1, 2)]
    for r in range(rounds):
        for cfg in cfgs:
            for p in fakes.PROFILES.values():
                sim.run_set(traces, cfg, p, seeds=seeds, modes=("headstart",) if p.name != "wikipedia"
                            else ("headstart", "baseline"))
        n = len(sim.caches.missing_web)
        print(f"prefetch round {r}: {n} missing queries; cache {len(sim.caches.web)} queries, "
              f"{len(sim.caches.emb)} vectors", flush=True)
        if not n:
            break
        asyncio.run(sim.caches.fetch_missing())
        sim.caches.save()
    sim.caches.save()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--fetch-missing", action="store_true", help="fetch queries logged by sweep workers")
    ap.add_argument("--seeds", type=int, default=10, help="seeds (validate / single config); prefetch: 3 is plenty")
    ap.add_argument("--profile", default="wikipedia")
    ap.add_argument("--set", action="append", default=[], help="KNOB=value (env var names, e.g. SPEC_WEB_BUDGET=4)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sim = Simulator()
    t = time.perf_counter()
    if a.prefetch:
        prefetch(sim, seeds=range(a.seeds))
        return
    if a.fetch_missing:
        mp = fakes.CACHE / "missing.txt"
        if mp.exists():
            sim.caches.missing_web |= {q for q in mp.read_text().splitlines() if q}
            print(f"fetched {asyncio.run(sim.caches.fetch_missing())} missing queries")
            sim.caches.save()
            mp.unlink()
        return
    if a.validate:
        out, rows = validate(sim, seeds=range(a.seeds))
        if sim.caches.missing_web:            # fetch what this run missed, then validate again
            asyncio.run(sim.caches.fetch_missing())
            sim.caches.save()
            out, rows = validate(sim, seeds=range(a.seeds))
        sim.caches.save()
        p = Path(a.out or ROOT / "runs" / "sim_validate.json")
        p.write_text(json.dumps({"validation": out, "rows": rows}, indent=1, default=float))
        for mode, v in out["modes"].items():
            print(mode, json.dumps({k: v[k] for k in v if k not in ("sim",)}))
            print("  sim", json.dumps(v["sim"]))
        print("paired sim", out["paired_sim"], "real", out["paired_real"])
        print(f"{time.perf_counter() - t:.1f}s; wrote {p}")
        return
    cfg = dict(CURRENT)
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = float(v) if "." in v else int(v)
    traces = load_traces()
    rows = sim.run_set(traces, cfg, fakes.PROFILES[a.profile], seeds=range(a.seeds))
    sim.caches.save()
    print(json.dumps(summarize(rows), indent=1))
    print(f"{time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
