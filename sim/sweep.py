"""Policy sweep over the speculator / endpointing knobs with the offline simulator.

  .venv/bin/python -m sim.sweep --profile wikipedia --n 400 --seeds 3 --workers 3      # stage 1: random search
  .venv/bin/python -m sim.sweep --refine runs/sweep_wikipedia.json --top 25 --seeds 12  # stage 2: re-evaluate the front
  .venv/bin/python -m sim.sweep --report                                                # Pareto tables + picks

Objective: p50 and p90 ready_ms (Head Start), subject to an accuracy proxy (risky_rate: answers not grounded in the
final query's search results and not an exact-meaning draft; pool-commit rate) and cost (web calls/question, LLM
requests/question, wasted drafts). Every config is evaluated on the same traces and the same random numbers
(common random numbers), so differences between configs are not sampling noise of the latency draws.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GRID = {
    "SPEC_STABLE_UPDATES": [1, 2],
    "SPEC_WEB_BUDGET": [3, 4, 6, 8],
    "SPEC_DRAFT_BUDGET": [2, 4, 6],
    "SEM_SILENCE_MS": [150, 200, 250, 350],
    "SEM_STABLE_MS": [300, 420, 550],
    "SEM_COVER_S": [0.2, 0.3, 0.4],
    "HEDGE_MS": [600, 800, 1000],
    "PENDING_CAP_S": [0.6, 1.0],
}
OUT = ROOT / "runs"

_W = {}


def _init(profile: str, seeds: list[int]):
    os.nice(10)
    from sim import fakes
    from sim.simulate import CURRENT, Simulator, load_traces
    sim = Simulator()
    traces = load_traces()
    prof = fakes.PROFILES[profile]
    bl = sim.run_set(traces, CURRENT, prof, seeds=seeds, modes=("baseline",))
    _W.update(sim=sim, traces=traces, prof=prof, seeds=seeds, bl={(r["id"], r["seed"]): r for r in bl},
              bl_rows=bl)


def _eval(cfg: dict) -> dict:
    from sim.simulate import summarize
    t = time.perf_counter()
    rows = _W["sim"].run_set(_W["traces"], cfg, _W["prof"], seeds=_W["seeds"], bl_rows=_W["bl"])
    s = summarize(rows)
    s["sim_s"] = round(time.perf_counter() - t, 2)
    return {"cfg": cfg, "summary": s, "per_q": {r["id"] + f"#{r['seed']}": r["ready_ms"] for r in rows}}


def _baseline_summary() -> dict:
    from sim.simulate import summarize
    return summarize(_W["bl_rows"], "baseline")


def all_configs():
    keys = list(GRID)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(GRID[k] for k in keys))]


def run(profile: str, cfgs: list[dict], seeds: list[int], workers: int, out: Path, tag: str):
    from sim.simulate import CURRENT
    res = {"profile": profile, "seeds": seeds, "grid": GRID, "results": [], "created": time.strftime("%H:%M:%S")}
    cfgs = [CURRENT] + [c for c in cfgs if c != CURRENT]
    t = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(profile, seeds)) as ex:
        res["baseline"] = ex.submit(_baseline_summary).result()
        for i, r in enumerate(ex.map(_eval, cfgs, chunksize=2)):
            r["is_current"] = r["cfg"] == CURRENT
            res["results"].append(r)
            if i % 20 == 0 or i == len(cfgs) - 1:
                s = r["summary"]
                print(f"[{tag} {i + 1}/{len(cfgs)}] {time.perf_counter() - t:.0f}s p50={s['p50_ready_ms']} "
                      f"p90={s['p90_ready_ms']} risky={s['risky_rate']} web={s['web_calls_per_q']}", flush=True)
                out.write_text(json.dumps(res, default=float))
    out.write_text(json.dumps(res, default=float))
    return res


# ---------------------------------------------------------------------------------------------------- analysis
OBJ = [("p50_ready_ms", 1), ("p90_ready_ms", 1), ("risky_rate", 1), ("web_calls_per_q", 1), ("llm_requests_per_q", 1)]


def pareto(results: list[dict], keys=OBJ) -> list[dict]:
    pts = [(r, [r["summary"][k] * s for k, s in keys]) for r in results]
    front = []
    for r, v in pts:
        dominated = any(all(a <= b for a, b in zip(w, v)) and any(a < b for a, b in zip(w, v)) for _, w in pts)
        if not dominated:
            front.append(r)
    return front


def score(s: dict, cur: dict) -> float | None:
    """Selection rule: no worse on the accuracy proxy than the current config (risky within +1 pt, pool-commit
    within +1 pt); then minimize p50 + 0.5 p90 + 25 ms per web call + 15 ms per LLM request (cost tie-breakers worth
    a few ms each, so a config that buys 10 ms with 2 extra web calls loses)."""
    if s["risky_rate"] > cur["risky_rate"] + 0.01 or s["pool_commit_rate"] > cur["pool_commit_rate"] + 0.01:
        return None
    if s.get("errors"):
        return None
    return s["p50_ready_ms"] + 0.5 * s["p90_ready_ms"] + 25 * s["web_calls_per_q"] + 15 * s["llm_requests_per_q"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="wikipedia")
    ap.add_argument("--n", type=int, default=400, help="random configs (0 = full grid)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--refine", default="", help="stage-2: re-evaluate the best/Pareto configs of this sweep file")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    if a.refine:
        src = json.loads(Path(a.refine).read_text())
        prof = src["profile"]
        cur = next(r for r in src["results"] if r["is_current"])["summary"]
        ok = [(score(r["summary"], cur), r) for r in src["results"]]
        best = [r for sc, r in sorted([x for x in ok if x[0] is not None], key=lambda x: x[0])[: a.top]]
        front = pareto(src["results"])
        front = sorted(front, key=lambda r: r["summary"]["p50_ready_ms"])[: a.top]
        cfgs, seen = [], set()
        for r in best + front:
            k = json.dumps(r["cfg"], sort_keys=True)
            if k not in seen:
                seen.add(k)
                cfgs.append(r["cfg"])
        out = Path(a.out or OUT / f"sweep_{prof}_refine.json")
        print(f"refine {len(cfgs)} configs on {prof} with seeds {seeds}", flush=True)
        run(prof, cfgs, seeds, a.workers, out, "refine")
        return
    cfgs = all_configs()
    if a.n:
        random.Random(7).shuffle(cfgs)
        cfgs = cfgs[: a.n]
    out = Path(a.out or OUT / f"sweep_{a.profile}.json")
    print(f"{len(cfgs)} configs × {len(seeds)} seeds on {a.profile}", flush=True)
    run(a.profile, cfgs, seeds, a.workers, out, a.profile)


if __name__ == "__main__":
    main()
