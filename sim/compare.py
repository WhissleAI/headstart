"""Paired comparison of named configs vs the current config on fresh seeds (common random numbers), both profiles.

  .venv/bin/python -m sim.compare            # -> runs/sim_compare.json
p50/p90 are computed per seed (one seed = one pass over the 48 questions, like one real A/B run); the delta vs current
is paired by seed, reported as mean ± 95% CI over seeds.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim import fakes  # noqa: E402
from sim.simulate import CURRENT, Simulator, load_traces, pct, summarize  # noqa: E402

PICKS = {
    "current": CURRENT,
    "W1 wiki-best": dict(SPEC_STABLE_UPDATES=1, SPEC_WEB_BUDGET=6, SPEC_DRAFT_BUDGET=4, SEM_SILENCE_MS=200,
                         SEM_STABLE_MS=550, SEM_COVER_S=0.2, HEDGE_MS=800, PENDING_CAP_S=0.6),
    "W2 wiki-cheap": dict(SPEC_STABLE_UPDATES=1, SPEC_WEB_BUDGET=6, SPEC_DRAFT_BUDGET=2, SEM_SILENCE_MS=350,
                          SEM_STABLE_MS=550, SEM_COVER_S=0.3, HEDGE_MS=1000, PENDING_CAP_S=0.6),
    "T1 tavily-best": dict(SPEC_STABLE_UPDATES=1, SPEC_WEB_BUDGET=8, SPEC_DRAFT_BUDGET=6, SEM_SILENCE_MS=200,
                           SEM_STABLE_MS=550, SEM_COVER_S=0.3, HEDGE_MS=800, PENDING_CAP_S=1.0),
    "T2 tavily-safe-cheap": dict(SPEC_STABLE_UPDATES=1, SPEC_WEB_BUDGET=6, SPEC_DRAFT_BUDGET=2, SEM_SILENCE_MS=200,
                                 SEM_STABLE_MS=550, SEM_COVER_S=0.4, HEDGE_MS=1000, PENDING_CAP_S=1.0),
    "T3 tavily-aggressive": dict(SPEC_STABLE_UPDATES=1, SPEC_WEB_BUDGET=6, SPEC_DRAFT_BUDGET=2, SEM_SILENCE_MS=200,
                                 SEM_STABLE_MS=420, SEM_COVER_S=0.2, HEDGE_MS=1000, PENDING_CAP_S=0.6),
}


def main(seeds=range(200, 220)):
    sim = Simulator()
    traces = load_traces()
    out = {}
    for pname, prof in fakes.PROFILES.items():
        rows = {k: sim.run_set(traces, c, prof, seeds=list(seeds)) for k, c in PICKS.items()}
        per_seed = {k: {s: (pct([r["ready_ms"] for r in rs if r["seed"] == s], 50),
                            pct([r["ready_ms"] for r in rs if r["seed"] == s], 90)) for s in seeds}
                    for k, rs in rows.items()}
        out[pname] = {}
        for k, rs in rows.items():
            s = summarize(rs)
            d50 = [per_seed[k][x][0] - per_seed["current"][x][0] for x in seeds]
            d90 = [per_seed[k][x][1] - per_seed["current"][x][1] for x in seeds]
            ci = lambda d: round(1.96 * statistics.pstdev(d) / len(d) ** 0.5, 1)
            s["d_p50_vs_current"] = (round(statistics.mean(d50), 1), ci(d50))
            s["d_p90_vs_current"] = (round(statistics.mean(d90), 1), ci(d90))
            out[pname][k] = {"cfg": PICKS[k], "summary": s}
            print(f"{pname:9s} {k:22s} p50 {s['p50_ready_ms']} (Δ {s['d_p50_vs_current'][0]:+.0f} ±{s['d_p50_vs_current'][1]:.0f})"
                  f"  p90 {s['p90_ready_ms']} (Δ {s['d_p90_vs_current'][0]:+.0f} ±{s['d_p90_vs_current'][1]:.0f})"
                  f"  vad50 {s['p50_ready_vad_ms']}  risky {s['risky_rate']:.3f} pool {s['pool_commit_rate']:.3f} "
                  f"top1=BL? n/a  web/q {s['web_calls_per_q']} llm/q {s['llm_requests_per_q']} "
                  f"waste/q {s['wasted_drafts_per_q']} reuse {s['draft_reuse_rate']} paths {s['paths']}", flush=True)
    (ROOT / "runs" / "sim_compare.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
