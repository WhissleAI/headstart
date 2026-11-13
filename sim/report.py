"""Pareto tables + recommended config per web profile from sweep results.

  .venv/bin/python -m sim.report runs/sweep_wikipedia_refine.json runs/sweep_tavily_refine.json > runs/sweep_report.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sim.sweep import GRID, pareto, score

SHORT = {"SPEC_STABLE_UPDATES": "stab", "SPEC_WEB_BUDGET": "web", "SPEC_DRAFT_BUDGET": "draft",
         "SEM_SILENCE_MS": "sil", "SEM_STABLE_MS": "semst", "SEM_COVER_S": "cover", "HEDGE_MS": "hedge",
         "PENDING_CAP_S": "cap"}
COLS = [("p50_ready_ms", "p50"), ("p90_ready_ms", "p90"), ("mean_ready_ms", "mean"), ("risky_rate", "risky"),
        ("pool_commit_rate", "pool"), ("bl_top1_rate", "top1=BL"), ("web_calls_per_q", "web/q"),
        ("llm_requests_per_q", "llm/q"), ("wasted_drafts_per_q", "waste/q"), ("draft_reuse_rate", "reuse")]


def cfg_str(c: dict) -> str:
    return " ".join(f"{SHORT[k]}={c[k]}" for k in GRID)


def row(r: dict, mark: str = "") -> str:
    s = r["summary"]
    return "| " + " | ".join([mark + cfg_str(r["cfg"])] + [str(s.get(k)) for k, _ in COLS]) + " |"


def env(c: dict) -> str:
    return " ".join(f"{k}={c[k]}" for k in GRID)


def report(path: str) -> str:
    d = json.loads(Path(path).read_text())
    res = [r for r in d["results"] if not r["summary"].get("errors")]
    cur = next(r for r in d["results"] if r["is_current"])
    cs = cur["summary"]
    out = [f"## {d['profile']}  ({len(d['results'])} configs × {len(d['seeds'])} seeds × {cs['n'] // len(d['seeds'])} questions)", ""]
    b = d.get("baseline") or {}
    out.append(f"Baseline (sim): p50 {b.get('p50_ready_ms')} / p90 {b.get('p90_ready_ms')} ms, "
               f"web/q {b.get('web_calls_per_q')}, llm/q {b.get('llm_requests_per_q')}")
    out.append("")
    hdr = "| config | " + " | ".join(h for _, h in COLS) + " |"
    sep = "|" + "---|" * (len(COLS) + 1)
    front = sorted(pareto(res), key=lambda r: (r["summary"]["p50_ready_ms"], r["summary"]["p90_ready_ms"]))
    scored = sorted([(score(r["summary"], cs), r) for r in res if score(r["summary"], cs) is not None],
                    key=lambda x: x[0])
    best = scored[0][1] if scored else cur
    out += ["**Current config**", "", hdr, sep, row(cur), "", "**Recommended** (accuracy proxy no worse than current; "
            "min p50 + 0.5·p90 + 25 ms/web call + 15 ms/LLM request)", "", hdr, sep, row(best), "",
            f"`{env(best['cfg'])}`", ""]
    # cheapest config within 25 ms of the best p50 (cost-efficient pick)
    near = [r for _, r in scored if r["summary"]["p50_ready_ms"] <= best["summary"]["p50_ready_ms"] + 25
            and r["summary"]["p90_ready_ms"] <= best["summary"]["p90_ready_ms"] + 50]
    if near:
        cheap = min(near, key=lambda r: (r["summary"]["web_calls_per_q"] + 0.5 * r["summary"]["llm_requests_per_q"]))
        if cheap is not best:
            out += ["**Cost-efficient alternative** (within 25 ms p50 / 50 ms p90 of the recommendation, fewest calls)",
                    "", hdr, sep, row(cheap), "", f"`{env(cheap['cfg'])}`", ""]
    out += [f"**Pareto front** over (p50, p90, risky, web/q, llm/q): {len(front)} configs, fastest 25 by p50", "",
            hdr, sep] + [row(r, "▶ " if r is best else "") for r in front[:25]] + [""]
    out += ["**Top 10 by the selection score**", "", hdr, sep] + [row(r) for _, r in scored[:10]] + [""]
    # marginal effect of each knob (mean over all configs having that value)
    out += ["**Knob marginals** (mean over configs with that value: p50 / p90 / risky / web per q)", ""]
    for k in GRID:
        cells = []
        for v in GRID[k]:
            rs = [r["summary"] for r in res if r["cfg"][k] == v]
            if rs:
                m = lambda f: sum(x[f] for x in rs) / len(rs)
                cells.append(f"{v}: {m('p50_ready_ms'):.0f} / {m('p90_ready_ms'):.0f} / {m('risky_rate'):.3f} / "
                             f"{m('web_calls_per_q'):.2f}")
        out.append(f"- `{k}` — " + "; ".join(cells))
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    print("\n\n".join(report(p) for p in sys.argv[1:]))
