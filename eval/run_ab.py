"""Real-audio A/B: baseline vs headstart, one question at a time, through the real Arc ear, in real time.

  .venv/bin/python eval/run_ab.py                        # all questions (current-events excluded on Wikipedia)
  .venv/bin/python eval/run_ab.py --limit 4
  .venv/bin/python eval/run_ab.py --ids q01,q26 --modes headstart
  .venv/bin/python eval/run_ab.py --kinds correction,filler --out runs/ab_corrections.json
  .venv/bin/python eval/run_ab.py --plot-only            # re-render runs/ab_latest.png from runs/ab_latest.json

For each (question, mode): a fresh ArcEar + one Session(mode); the wav is fed in 100 ms chunks paced to the wall
clock (chunk k is fed when its last sample would have been spoken), followed by 1.2 s of zeros so Arc endpoints on
the pause, exactly like server/app.py replay(). Session.t0 / spec.t0 = wall time of the first audio sample.
Modes run sequentially (never concurrently); which mode goes first alternates per question (web/CDN caches).

ready_ms   = first answer token - end of the last word (Arc's word timing; server metric)
ready_vad  = first answer token - end of speech measured on the waveform (energy), independent of Arc timing
Correct    = answer contains every "expect" keyword (accent/case-insensitive; "a|b" = alternatives).
Writes --out (default runs/ab_<timestamp>.json) after every question, and runs/ab_latest.json + .png at the end.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

AUDIO = ROOT / "eval" / "audio"
RUNS = ROOT / "runs"
CHUNK = 1600
TRAIL_CHUNKS = 12          # 1.2 s of zeros after the file, like app.replay()
PAUSE_BETWEEN_S = 0.8      # let cancelled drafts / fires settle between runs


# ---------------------------------------------------------------------------------------------------- questions
def load_questions(include_local: bool = True) -> list[dict]:
    rows = []
    for p in [ROOT / "eval" / "questions.jsonl"] + ([ROOT / "index" / "questions_local.jsonl"] if include_local else []):
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r.setdefault("source", p.name)
                rows.append(r)
    seen, out = set(), []
    for r in rows:
        if r["id"] in seen:
            print(f"warning: duplicate id {r['id']} in {r['source']}, skipped", file=sys.stderr)
            continue
        seen.add(r["id"])
        out.append(r)
    return out


def load_wav(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return np.ascontiguousarray(wav, dtype=np.float32)


def speech_end_s(wav: np.ndarray, thr_db: float = -40.0) -> float:
    """End of the last 10 ms frame within thr_db of the file's peak frame energy."""
    win = 160
    n = len(wav) // win
    if n == 0:
        return 0.0
    e = np.sqrt(np.mean(wav[: n * win].reshape(n, win) ** 2, axis=1))
    db = 20 * np.log10(e / (e.max() + 1e-12) + 1e-12)
    on = np.where(db > thr_db)[0]
    return float((on[-1] + 1) * win / 16000) if len(on) else 0.0


# ---------------------------------------------------------------------------------------------------- scoring
def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"(?<=\d)[,   ](?=\d{3}\b)", "", s)     # 8,849 / 8 849 -> 8849


def score(answer: str, expect: list[str]) -> bool | None:
    if not expect:
        return None
    a = _fold(answer)
    for item in expect:
        ok = False
        for alt in item.split("|"):
            alt = _fold(alt.strip())
            if not alt:
                continue
            if alt[-1].isdigit():   # numbers: left boundary only ("299792" matches "299792458")
                pat = rf"(?<![a-z0-9]){re.escape(alt)}"
            else:                   # words: whole word, optional plural
                pat = rf"(?<![a-z0-9]){re.escape(alt)}(?:s|es)?(?![a-z0-9])"
            if re.search(pat, a):
                ok = True
                break
        if not ok:
            return False
    return True


# ---------------------------------------------------------------------------------------------------- one run
async def run_one(q: dict, mode: str) -> dict:
    from server.ear import ArcEar
    from server.session import Session

    wav = load_wav(AUDIO / q["file"])
    vad_end = speech_end_s(wav)
    events: list[dict] = []
    ear_events: list[dict] = []

    def emit(e):
        e["_wall"] = time.perf_counter()
        events.append(e)

    def on_ear(e):
        e["_wall"] = time.perf_counter()
        ear_events.append(e)

    sess = Session(emit, mode=mode)
    ear = ArcEar([sess], on_event=on_ear)
    t0 = time.perf_counter()
    sess.t0 = t0
    sess.spec.t0 = t0
    feed_late = []
    try:
        n = len(wav)
        for i in range(0, n, CHUNK):
            target = t0 + min(i + CHUNK, n) / 16000
            while time.perf_counter() < target:
                await asyncio.sleep(0.003)
            feed_late.append(time.perf_counter() - target)
            await ear.feed(wav[i:i + CHUNK])
        t_audio_end = t0 + n / 16000
        for k in range(TRAIL_CHUNKS):
            target = t_audio_end + (k + 1) * CHUNK / 16000
            while time.perf_counter() < target:
                await asyncio.sleep(0.003)
            await ear.feed(np.zeros(CHUNK, np.float32))
        n_finals_before_flush = sum(1 for e in ear_events if e.get("kind") == "final")
        await ear.flush()
        # semantic endpointing answers from on_partial via session.final_task (not an ear final); wait for it too
        seen = set()
        while getattr(sess, "final_task", None) is not None and id(sess.final_task) not in seen:
            seen.add(id(sess.final_task))
            await asyncio.gather(sess.final_task, return_exceptions=True)
    finally:
        ear.close()
        sess.close()

    task_errors = []
    for t in list(ear.final_tasks) + ([sess.final_task] if getattr(sess, "final_task", None) else []):
        if t.done() and not t.cancelled() and t.exception() is not None:
            task_errors.append(f"{type(t.exception()).__name__}: {t.exception()}"[:200])
    finals = [e for e in ear_events if e.get("kind") == "final"]
    last_ep = max([e["_wall"] for e in events if e.get("t") == "endpoint"], default=0.0)
    toks = [e["_wall"] for e in events if e.get("t") == "answer" and e["_wall"] >= last_ep]
    first_tok = toks[0] if toks else None
    ms = [m for m in sess.history if m]
    m = ms[-1] if ms else {}
    endpoints = [e for e in events if e.get("t") == "endpoint"]
    arc_end_s = (endpoints[-1]["speech_end_ms"] / 1000) if endpoints else None
    ready = m.get("ready_ms")
    spec_q = [e["query"] for e in events if e.get("t") == "spec" and e.get("state") == "fired"
              and e.get("lane") != "draft"]
    row = {
        "id": q["id"], "text": q["text"], "kind": q.get("kind"), "voice": q.get("voice"), "mode": mode,
        "ready_ms": ready, "sources_ms": m.get("sources_ms"), "done_ms": m.get("done_ms"),
        "endpoint_lag_ms": m.get("endpoint_lag_ms"),
        "ready_vad_ms": (round(ready + (arc_end_s - vad_end) * 1000) if ready is not None and arc_end_s is not None
                         else None),
        "arc_end_s": arc_end_s, "vad_end_s": round(vad_end, 3),
        "path": m.get("path"), "correct": score(m.get("answer", ""), q.get("expect") or []),
        "answer": m.get("answer"), "error": m.get("error") or ("; ".join(task_errors) if task_errors else None)
        or (None if ms else "no answer (no metric in session.history)"),
        "endpoint_by": m.get("endpoint_by"), "answers": len(ms),
        "web_calls": m.get("web_calls"), "spec_fires": m.get("spec_fires"), "spec_used": m.get("spec_used"),
        "stale": m.get("stale"), "drafts": m.get("drafts"), "draft_reused": m.get("draft_reused"),
        "draft_match": m.get("draft_match"), "query": m.get("query"), "intent": m.get("intent"),
        "spec_queries": spec_q, "sources": m.get("sources"),
        # compact timeline (ms since first audio sample) of fires/drafts and the endpoint, for debugging paths
        "trace": [{k: e.get(k) for k in ("t", "id", "lane", "state", "ms", "took", "n", "query", "by")
                   if e.get(k) is not None} for e in events if e.get("t") in ("spec", "endpoint", "extended", "sources")]
                 + ([{"t": "first_token", "ms": round((first_tok - t0) * 1000)}] if first_tok else []),
        "arc_heard": " | ".join(e.get("text", "") for e in finals),
        "n_finals": len(finals),
        "endpoint_via": "pause" if n_finals_before_flush else ("flush" if finals else "none"),
        "feed_late_p95_ms": round(float(np.percentile(feed_late, 95)) * 1000, 1) if feed_late else None,
    }
    return row


# ---------------------------------------------------------------------------------------------------- summary
def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return round(float(np.percentile(xs, p)))


def summarize(rows: list[dict]) -> dict:
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        rs = [r for r in rows if r["mode"] == mode]
        ready = [r["ready_ms"] for r in rs if r["ready_ms"] is not None]
        scored = [r for r in rs if r["correct"] is not None]
        out[mode] = {
            "n": len(rs),
            "p50_ready_ms": pct(ready, 50), "p90_ready_ms": pct(ready, 90),
            "mean_ready_ms": round(statistics.mean(ready)) if ready else None,
            "p50_ready_vad_ms": pct([r["ready_vad_ms"] for r in rs], 50),
            "p50_sources_ms": pct([r["sources_ms"] for r in rs], 50),
            "p50_done_ms": pct([r["done_ms"] for r in rs], 50),
            "p50_endpoint_lag_ms": pct([r["endpoint_lag_ms"] for r in rs], 50),
            "accuracy": round(sum(1 for r in scored if r["correct"]) / len(scored), 3) if scored else None,
            "n_scored": len(scored),
            "mean_web_calls": round(statistics.mean([r["web_calls"] or 0 for r in rs]), 2) if rs else None,
            "draft_reuse_rate": round(sum(1 for r in rs if r["draft_reused"]) / len(rs), 3) if rs else None,
            "errors": sum(1 for r in rs if r["error"] or r["ready_ms"] is None),
            "split_questions": sum(1 for r in rs if (r["n_finals"] or 0) > 1),
            "answered_twice": sum(1 for r in rs if (r.get("answers") or 0) > 1),
            "endpoint_by": dict(Counter(str(r.get("endpoint_by")) for r in rs)),
            "paths": dict(Counter(str(r["path"]) for r in rs)),
            "by_kind": {k: {"n": len(g), "p50_ready_ms": pct([r["ready_ms"] for r in g], 50),
                            "accuracy": (round(sum(1 for r in g if r["correct"]) / len([r for r in g if r["correct"] is not None]), 3)
                                         if any(r["correct"] is not None for r in g) else None)}
                        for k in sorted({r["kind"] for r in rs})
                        for g in [[r for r in rs if r["kind"] == k]]},
        }
    by = {}
    for r in rows:
        by.setdefault(r["id"], {})[r["mode"]] = r
    pairs = [(v["baseline"], v["headstart"]) for v in by.values() if "baseline" in v and "headstart" in v
             and v["baseline"]["ready_ms"] is not None and v["headstart"]["ready_ms"] is not None]
    if pairs:
        d = [b["ready_ms"] - h["ready_ms"] for b, h in pairs]
        out["paired"] = {"n": len(pairs), "p50_saved_ms": pct(d, 50), "mean_saved_ms": round(statistics.mean(d)),
                         "headstart_faster": sum(1 for x in d if x > 0),
                         "both_correct": sum(1 for b, h in pairs if b["correct"] and h["correct"]),
                         "only_baseline_correct": sum(1 for b, h in pairs if b["correct"] and h["correct"] is False),
                         "only_headstart_correct": sum(1 for b, h in pairs if h["correct"] and b["correct"] is False)}
    return out


def run_config(local_docs: int) -> dict:
    from server import config
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "server"],
                                    capture_output=True, text=True).stdout.strip())
    except Exception:
        commit, dirty = "", None
    return {"answer_model": config.ANSWER_MODEL, "web_backend": config.WEB_BACKEND, "embed_model": config.EMBED_MODEL,
            "embed_device": config.EMBED_DEVICE, "arc_dir": str(config.ARC_DIR), "arc_threads": config.ARC_THREADS,
            "local_docs": local_docs, "spec": {"min_words": config.MIN_WORDS, "stable_updates": config.STABLE_UPDATES,
                                               "web_budget": config.WEB_BUDGET, "draft_budget": config.DRAFT_BUDGET,
                                               "draft_stable_ms": config.DRAFT_STABLE_MS,
                                               "audio_lane": config.AUDIO_LANE, "commit_tau": config.COMMIT_TAU},
            "semantic_endpoint": getattr(config, "SEMANTIC_ENDPOINT", None),
            "sem_silence_ms": getattr(config, "SEM_SILENCE_MS", None),
            "sem_stable_ms": getattr(config, "SEM_STABLE_MS", None),
            "trail_s": TRAIL_CHUNKS * CHUNK / 16000, "chunk_ms": CHUNK / 16, "git": commit, "server_dirty": dirty,
            "machine": f"{platform.node()} {platform.machine()} {platform.mac_ver()[0]}"}


# ---------------------------------------------------------------------------------------------------- plot
def plot(res: dict, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    rows = res["rows"]
    by: dict[str, dict] = {}
    for r in rows:
        if r["ready_ms"] is not None:
            by.setdefault(r["id"], {})[r["mode"]] = r
    ids = sorted(by, key=lambda i: by[i].get("baseline", by[i].get("headstart"))["ready_ms"])
    if not ids:
        return False
    colors = {"headstart": "#2a78d6", "baseline": "#eb6834"}
    ink, muted, grid = "#2b2b2a", "#6b6a63", "#e6e5df"
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(ids) + 2.2), dpi=150)
    y = np.arange(len(ids))
    for yi, i in zip(y, ids):
        xs = [by[i][m]["ready_ms"] / 1000 for m in ("baseline", "headstart") if m in by[i]]
        if len(xs) == 2:
            ax.plot(xs, [yi, yi], color=grid, lw=2, zorder=1, solid_capstyle="round")
    for mode in ("baseline", "headstart"):
        pts = [(yi, by[i][mode]) for yi, i in zip(y, ids) if mode in by[i]]
        if not pts:
            continue
        xs = [r["ready_ms"] / 1000 for _, r in pts]
        ys = [yi for yi, _ in pts]
        wrong = [r["correct"] is False for _, r in pts]
        p50 = res["summary"].get(mode, {}).get("p50_ready_ms")
        lab = f"{mode}  (p50 {p50 / 1000:.2f} s)" if p50 is not None else mode
        ax.scatter(xs, ys, s=48, color=colors[mode], edgecolor="white", linewidth=1.2, zorder=3, label=lab)
        wx = [x for x, w in zip(xs, wrong) if w]
        wy = [yy for yy, w in zip(ys, wrong) if w]
        if wx:
            ax.scatter(wx, wy, s=110, facecolor="none", edgecolor=colors[mode], linewidth=1.2, zorder=2)
        if p50 is not None:
            ax.axvline(p50 / 1000, color=colors[mode], lw=1.2, ls=(0, (4, 3)), zorder=0)
    labels = []
    for i in ids:
        r = next(iter(by[i].values()))
        t = r["text"] if len(r["text"]) <= 46 else r["text"][:45] + "…"
        labels.append(f"{i}  {t}")
    ax.set_yticks(y, labels, fontsize=7.5, color=ink)
    ax.set_ylim(-0.7, len(ids) - 0.3)
    ax.set_xlim(left=min(0, ax.get_xlim()[0]))
    ax.axvline(0, color=muted, lw=0.8)
    ax.grid(axis="x", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(muted)
    ax.tick_params(axis="x", colors=muted, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("ready: end of last spoken word → first answer token (s)", color=ink, fontsize=9)
    c = res["config"]
    pr = res["summary"].get("paired", {})
    sub1 = f"{c['answer_model']} · web={c['web_backend']} · Arc ear, real time · n={len(ids)} questions"
    if c.get("semantic_endpoint") is not None:
        sub1 += f" · semantic endpoint {'on' if c['semantic_endpoint'] else 'off'}"
    sub2 = (f"Head Start faster on {pr['headstart_faster']}/{pr['n']}, median saved {pr['p50_saved_ms']} ms · "
            if pr else "") + "dashed = p50 · ring = wrong answer"
    fig.text(0.01, 0.995, "Head Start vs baseline: time to first answer token", fontsize=11, color=ink,
             va="top", ha="left")
    fig.text(0.01, 0.995 - 0.30 / fig.get_figheight(), sub1 + "\n" + sub2, fontsize=7.5, color=muted, va="top",
             ha="left", linespacing=1.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8, borderaxespad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.62 / fig.get_figheight()))
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------------------------------- main
def print_row(r):
    ok = {True: "ok ", False: "BAD", None: " - "}[r["correct"]]
    print(f"  {r['id']} {r['mode']:9s} ready={r['ready_ms']!s:>5}ms vad={r['ready_vad_ms']!s:>5} "
          f"src={r['sources_ms']!s:>5} lag={r['endpoint_lag_ms']!s:>4} ep={r.get('endpoint_by') or '-':8s} n_ans={r.get('answers')} {r['path'] or '-':22s} web={r['web_calls']} "
          f"fires={r['spec_fires']} drafts={r['drafts']} reuse={r['draft_reused']} {ok} "
          f"heard={r['arc_heard']!r} | {(r['answer'] or r['error'] or '')[:80]!r}", flush=True)


async def main_async(a) -> int:
    from server import answer, config, ear, embed, lanes

    qs = load_questions(include_local=not a.no_local)
    if a.ids:
        want = a.ids.split(",")
        qs = [q for q in qs if q["id"] in want]
    if a.kinds and a.kinds != "all":
        ks = set(a.kinds.split(","))
        qs = [q for q in qs if q.get("kind") in ks]
    elif not a.kinds and config.WEB_BACKEND == "wikipedia":
        qs = [q for q in qs if q.get("kind") != "current"]
    qs = [q for q in qs if (AUDIO / q["file"]).exists() or print(f"missing audio {q['file']}", file=sys.stderr)]
    if a.limit:
        qs = qs[: a.limit]
    modes = a.modes.split(",")
    print(f"{len(qs)} questions × {modes}", flush=True)

    t = time.perf_counter()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, ear.warmup)
    await embed.arun(embed.warmup)
    n_local = len(lanes.index())
    await lanes.warm_web()
    await answer.warm()
    print(f"warm in {time.perf_counter() - t:.1f}s (local docs: {n_local}, web: {config.WEB_BACKEND}, "
          f"model: {config.ANSWER_MODEL})", flush=True)

    cfg = run_config(n_local)
    if not a.no_warm_run:
        # one throwaway replay per mode so first-run effects (MPS kernels, TLS, tagger) don't land on question 1
        wq = next((q for q in load_questions() if q.get("kind") == "current"), qs[0] if qs else None)
        if wq:
            for mode in modes:
                await run_one(wq, mode)
            print(f"warm-up replay done ({wq['id']})", flush=True)

    out = Path(a.out) if a.out else RUNS / f"ab_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config": cfg, "summary": {}, "rows": []}
    for qi, q in enumerate(qs):
        order = modes if qi % 2 == 0 else list(reversed(modes))
        print(f"[{qi + 1}/{len(qs)}] {q['id']} ({q.get('kind')}) {q['text']!r}", flush=True)
        for mode in order:
            try:
                r = await run_one(q, mode)
            except Exception as e:  # keep going; record the failure
                r = {"id": q["id"], "text": q["text"], "kind": q.get("kind"), "mode": mode, "ready_ms": None,
                     "sources_ms": None, "done_ms": None, "endpoint_lag_ms": None, "ready_vad_ms": None,
                     "path": None, "correct": False if q.get("expect") else None, "answer": None,
                     "error": f"harness: {type(e).__name__}: {e}"[:300], "web_calls": 0, "spec_fires": 0,
                     "stale": 0, "drafts": 0, "draft_reused": False, "arc_heard": "", "n_finals": 0}
            r["order"] = order.index(mode)
            res["rows"].append(r)
            print_row(r)
            await asyncio.sleep(PAUSE_BETWEEN_S)
        res["summary"] = summarize(res["rows"])
        out.write_text(json.dumps(res, indent=1, default=float))
    res["summary"] = summarize(res["rows"])
    txt = json.dumps(res, indent=1, default=float)
    out.write_text(txt)
    (RUNS / "ab_latest.json").write_text(txt)
    png = plot(res, RUNS / "ab_latest.png")
    print("\nsummary:")
    for k, v in res["summary"].items():
        v = {kk: vv for kk, vv in v.items() if kk != "by_kind"}
        print(f"  {k}: {json.dumps(v)}")
    print(f"wrote {out} and {RUNS / 'ab_latest.json'}" + (f" + {RUNS / 'ab_latest.png'}" if png else ""))
    try:
        await lanes.client().aclose()
        await answer.client().close()
    except Exception:
        pass
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--kinds", default="", help="comma list, or 'all' (default: all but 'current' on Wikipedia)")
    ap.add_argument("--modes", default="baseline,headstart")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-local", action="store_true", help="ignore index/questions_local.jsonl")
    ap.add_argument("--no-warm-run", action="store_true", help="skip the throwaway warm-up replay")
    ap.add_argument("--no-semantic", action="store_true",
                    help="SEMANTIC_ENDPOINT=0: Head Start endpoints on Arc's pause like baseline")
    ap.add_argument("--plot-only", action="store_true", help="re-render runs/ab_latest.png from runs/ab_latest.json")
    a = ap.parse_args()
    if a.plot_only:
        res = json.loads((RUNS / "ab_latest.json").read_text())
        print("ok" if plot(res, RUNS / "ab_latest.png") else "matplotlib unavailable")
        return
    if a.no_semantic:
        os.environ["SEMANTIC_ENDPOINT"] = "0"     # before any `server` import (config reads env at import)
    RUNS.mkdir(exist_ok=True)
    lock = open(RUNS / ".run_ab.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another eval/run_ab.py is running (runs/.run_ab.lock); refusing to start a second one")
    os.chdir(ROOT)
    rc = asyncio.run(main_async(a))
    sys.stdout.flush()
    os._exit(rc)   # torch / executor threads can keep the interpreter alive; everything is written by now


if __name__ == "__main__":
    main()
