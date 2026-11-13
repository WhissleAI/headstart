"""Robustness tables from run_ab.py outputs (markdown to stdout).

  .venv/bin/python eval/robustness_report.py clean=runs/ab_full2.json snr20=runs/ab_snr20.json ...

Rows are restricted per condition to the questions that condition ran; "clean (same ids)" rows are added for subset
conditions (rate) so they compare like with like. WER = Arc's transcript of the user's own utterance vs the script
(q_wer when present; older runs: all finals joined).
Accuracy is recomputed with the current run_ab.score() for every run, so old and new runs compare.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ab import load_questions, pct, score, wer  # noqa: E402

EXPECT = {q["id"]: q.get("expect") or [] for q in load_questions()}


def load(p):
    rows = json.loads(Path(p).read_text())["rows"]
    for r in rows:      # rescore every run with the current (stem / number-word tolerant) scorer
        if r["id"] in EXPECT:
            r["correct"] = score(r.get("answer") or "", EXPECT[r["id"]])
    return rows


def uwer(r):
    if r.get("q_wer") is not None:
        return r["q_wer"]
    return wer(r["text"], (r.get("arc_heard") or "").replace("|", " "))


def stats(rows, mode):
    rs = [r for r in rows if r["mode"] == mode]
    ready = [r["ready_ms"] for r in rs if r["ready_ms"] is not None]
    sc = [r for r in rs if r["correct"] is not None]
    eb = {}
    for r in rs:
        eb[str(r.get("endpoint_by"))] = eb.get(str(r.get("endpoint_by")), 0) + 1
    return {
        "n": len(rs), "p50": pct(ready, 50), "p90": pct(ready, 90),
        "vad50": pct([r.get("ready_vad_ms") for r in rs], 50),
        "acc": 100 * sum(1 for r in sc if r["correct"]) / len(sc) if sc else None,
        "wer": 100 * statistics.mean([uwer(r) for r in rs]) if rs else None,
        "sem": eb.get("semantic", 0), "ear": eb.get("ear", 0),
        "lag50": pct([r.get("endpoint_lag_ms") for r in rs], 50),
        "twice": sum(1 for r in rs if (r.get("answers") or 0) - (r.get("phantom_answers") or 0) > 1),
        "phantom": sum(1 for r in rs if r.get("phantom_answers")),
        "web": statistics.mean([r.get("web_calls") or 0 for r in rs]) if rs else 0,
        "reuse": sum(1 for r in rs if r.get("draft_reused")),
        "hedged": (sum(1 for r in rs if r.get("hedged")) if any("hedged" in r for r in rs) else None),
        "err": sum(1 for r in rs if r.get("ready_ms") is None),
    }


def paired(rows):
    by = {}
    for r in rows:
        by.setdefault(r["id"], {})[r["mode"]] = r
    d = [v["baseline"]["ready_ms"] - v["headstart"]["ready_ms"] for v in by.values()
         if "baseline" in v and "headstart" in v and v["baseline"]["ready_ms"] is not None
         and v["headstart"]["ready_ms"] is not None]
    return (sum(1 for x in d if x > 0), len(d), pct(d, 50)) if d else (0, 0, None)


def f(x, fmt="{:.0f}"):
    return "–" if x is None else fmt.format(x)


def main():
    conds = [a.split("=", 1) for a in sys.argv[1:]]
    data = {name: load(p) for name, p in conds}
    clean = data.get("clean")
    print("| condition | mode | n | ready p50 | ready p90 | ready_vad p50 | accuracy | Arc WER | endpoint sem/ear "
          "| ep lag p50 | answered twice | phantom ans | drafts reused | hedged | HS faster (median saved) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, rows in data.items():
        ids = {r["id"] for r in rows}
        groups = [(name, rows)]
        if clean is not None and name != "clean" and len(ids) < len({r["id"] for r in clean}) * 0.8:
            groups.insert(0, (f"clean (same {len(ids)})", [r for r in clean if r["id"] in ids]))
        for gname, rs in groups:
            hf, n, saved = paired(rs)
            for mode in ("baseline", "headstart"):
                s = stats(rs, mode)
                if not s["n"]:
                    continue
                print(f"| {gname} | {mode} | {s['n']} | {f(s['p50'])} | {f(s['p90'])} | {f(s['vad50'])} | "
                      f"{f(s['acc'], '{:.1f}%')} | {f(s['wer'], '{:.1f}%')} | {s['sem']}/{s['ear']} | "
                      f"{f(s['lag50'])} | {s['twice']} | {s['phantom']} | {s['reuse']} | {f(s['hedged'])} | "
                      + (f"{hf}/{n} ({f(saved)} ms)" if mode == "headstart" else "") + " |")


if __name__ == "__main__":
    main()
