"""Offline before/after replay of the Arc Tagger read-out signals on the Tavily lane (no audio, no ear, no reranker).

For each time-sensitive / real-world question (eval/questions.jsonl kind=current + eval/showcase/questions.jsonl) the
heard text (`arc_text`) is fed word by word into the native Arc Tagger and the final frame is built twice:
  before = server/frame.py at --base (intent + slots only) and web_plan without hints
  after  = the working tree: Frame.web_hints() (typed time / money spans, result words, weather guard) + "$100" kept
Only questions whose (query, Tavily plan) differ are sent: real Tavily (fast depth, 1 credit) -> top 5 in Tavily's order
-> the real answer model (server/answer.py SYSTEM + prompt). Then a blinded pairwise judge (same model) picks the answer
that better answers the question as of today. Reported: fresh-source rate (published within 7 days), grounded rate
(the answer doesn't say the sources lack it), judge wins, credits.

  .venv/bin/python eval/replay_tagger.py [--dry] [--out runs/replay_tagger.json]
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from server import answer, config, lanes  # noqa: E402
from server import frame as frame_new  # noqa: E402
from server.speculator import _non_answer  # noqa: E402

import replay_frames  # noqa: E402
from replay_answers import load_old  # noqa: E402

TODAY = dt.date(2026, 11, 12)
JUDGE = """Today is {today}. A user asked a voice assistant: "{q}"
Two answers (each written from web search results):
A: {a}
B: {b}
Which answer better answers the question for TODAY: current (not stale), specific, and directly responsive? Reply with
exactly one of: A, B, TIE."""


def questions() -> list[dict]:
    rows = [json.loads(x) for x in (ROOT / "eval" / "questions.jsonl").read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("kind") == "current"]
    sc = ROOT / "eval" / "showcase" / "questions.jsonl"
    if not sc.exists():
        sc = ROOT.parent / "headstart" / "eval" / "showcase" / "questions.jsonl"
    if sc.exists():
        rows += [json.loads(x) for x in sc.read_text().splitlines() if x.strip()]
    return rows


def plan_of(f, hints: bool) -> dict:
    p = lanes.web_plan(f.intent, f.resolved_text, final=True, hints=f.web_hints() if hints else None)
    return {k: p[k] for k in ("topic", "time_range", "max_age_hours") if p.get(k) is not None}


def fresh(doc, days=7) -> bool | None:
    pub = (doc.meta or {}).get("published_date")
    if not pub:
        return None
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            d = dt.datetime.strptime(str(pub)[:len(dt.datetime.now().strftime(fmt))] if fmt != "%a, %d %b %Y %H:%M:%S %Z"
                                     else str(pub), fmt).date()
            return (TODAY - d).days <= days
        except ValueError:
            continue
    return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="b1b27c1")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--max-credits", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "runs" / "replay_tagger.json"))
    a = ap.parse_args()
    frame_old, _ = load_old(a.base)
    items = []
    for r in questions():
        heard = r.get("arc_text") or r["text"]
        replay_frames.framemod = frame_old
        fo = replay_frames.frame_for(heard)
        replay_frames.framemod = frame_new
        fn = replay_frames.frame_for(heard)
        b = {"query": fo.query(), "plan": plan_of(fo, False), "question": fo.resolved_text}
        n = {"query": fn.query(), "plan": plan_of(fn, True), "question": fn.resolved_text,
             "readout": fn.readout()}
        items.append({"id": r["id"], "kind": r.get("kind"), "heard": heard, "before": b, "after": n,
                      "changed": (b["query"], b["plan"]) != (n["query"], n["plan"])})
    changed = [x for x in items if x["changed"]]
    print(f"{len(items)} questions, {len(changed)} with a different query or Tavily plan")
    for x in changed:
        print(f"  {x['id']:5s} {x['before']['query']!r} {x['before']['plan']} -> {x['after']['query']!r} {x['after']['plan']}")
    if a.dry:
        return
    if len(changed) * 2 > a.max_credits:
        raise SystemExit("credit budget exceeded")
    ledger = lanes.Ledger(budget=a.max_credits + 5, reserve=0)
    sem = asyncio.Semaphore(4)
    calls = {"llm": 0}

    async def llm(system, content, max_tokens=160):
        calls["llm"] += 1
        async with sem:
            for attempt in range(3):
                try:
                    r = await answer.client().messages.create(
                        model=config.ANSWER_MODEL, max_tokens=max_tokens, system=system,
                        **answer.model_opts(config.ANSWER_MODEL), messages=[{"role": "user", "content": content}])
                    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
                except Exception as e:
                    if attempt == 2:
                        return f"<error {type(e).__name__}>"
                    await asyncio.sleep(1.5 * (attempt + 1))

    async def variant(v: dict):
        meter: dict = {}
        plan = lanes.web_plan(final=True)
        plan.update(v["plan"])
        for k in ("time_range", "max_age_hours"):
            if k not in v["plan"]:
                plan.pop(k, None)
        async with sem:
            try:
                docs = list(await lanes.tavily(v["query"], plan, ledger, meter, final=True))[:5]
            except Exception as e:
                v["error"] = f"{type(e).__name__}: {e}"[:200]
                docs = []
        v["credits"] = meter.get("credits", 0)
        v["sources"] = [{"title": d.title, "url": d.url, "published": (d.meta or {}).get("published_date")}
                        for d in docs]
        fr = [fresh(d) for d in docs]
        v["n_dated"] = sum(x is not None for x in fr)
        v["n_fresh"] = sum(bool(x) for x in fr)
        v["answer"] = await llm(answer.SYSTEM, answer.build_prompt(v["question"], docs)) if docs else ""
        v["grounded"] = bool(v["answer"]) and not _non_answer(v["answer"]) and not v["answer"].startswith("<error")

    await asyncio.gather(*[variant(x[s]) for x in changed for s in ("before", "after")])
    rng = random.Random(7)

    async def judge(x):
        flip = rng.random() < 0.5
        A, B = (x["after"], x["before"]) if flip else (x["before"], x["after"])
        out = await llm("You are a strict evaluator.", JUDGE.format(today=TODAY.isoformat(), q=x["heard"],
                                                                     a=A["answer"] or "(no answer)",
                                                                     b=B["answer"] or "(no answer)"), max_tokens=5)
        o = out.strip().upper()[:3]
        x["judge_raw"] = out
        x["judge"] = "tie" if o.startswith("TIE") else (("after" if flip else "before") if o.startswith("A")
                                                         else ("before" if flip else "after") if o.startswith("B")
                                                         else "?")

    await asyncio.gather(*[judge(x) for x in changed])

    def agg(side):
        vs = [x[side] for x in changed]
        nd = sum(v["n_dated"] for v in vs)
        return {"grounded": sum(v["grounded"] for v in vs), "n": len(vs), "credits": sum(v["credits"] for v in vs),
                "sources": sum(len(v["sources"]) for v in vs), "dated": nd, "fresh_7d": sum(v["n_fresh"] for v in vs),
                "q_with_fresh_source": sum(v["n_fresh"] > 0 for v in vs)}
    summ = {"n_questions": len(items), "n_changed": len(changed), "before": agg("before"), "after": agg("after"),
            "judge": {k: sum(x["judge"] == k for x in changed) for k in ("after", "before", "tie", "?")},
            "llm_calls": calls["llm"], "tavily_credits": ledger.spent, "base": a.base}
    Path(a.out).write_text(json.dumps({"summary": summ, "items": items}, indent=1, default=str))
    print(json.dumps(summ, indent=1))
    for x in changed:
        print(f"\n{x['id']} {x['heard']}  judge={x['judge']}")
        for s in ("before", "after"):
            v = x[s]
            print(f"  {s:6s} fresh {v['n_fresh']}/{len(v['sources'])} grounded={v['grounded']} | {v['answer'][:200]!r}")


if __name__ == "__main__":
    asyncio.run(main())
