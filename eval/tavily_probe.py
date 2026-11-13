"""Measure the real Tavily API the moment a key is set: latency per search_depth, include_answer overhead, response
shape and billed credits. Spends credits: prints the estimate first and needs --yes.

.venv/bin/python eval/tavily_probe.py            # dry run: what it would send and cost
.venv/bin/python eval/tavily_probe.py --yes      # ~54 credits with the defaults (reps 3)

Decide from the output:
  - TAVILY_SPEC_DEPTH: fast unless ultra-fast is much faster AND its one-summary content still answers the questions
  - TAVILY_FINAL_DEPTH: on the critical path; basic only if its p50 is within ~150 ms of fast
  - TAVILY_ANSWER=1 only if (answer p50 - fast p50) < ~600 ms (Haiku TTFT after our own search)
"""
import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import config, lanes  # noqa: E402

QUERIES = [("founded company nvidia", "discourse.intent.qa_factoid", "who founded the company Nvidia"),
           ("latest news nvidia today", "discourse.intent.news_query", "what is the latest news about Nvidia today"),
           ("weather paris tomorrow", "discourse.intent.weather_query", "what's the weather in Paris tomorrow")]
DEPTHS = ["ultra-fast", "fast", "basic", "advanced"]


async def one(q, intent, text, depth, answer=False):
    plan = lanes.web_plan(intent, text, final=True, answer=answer)
    plan["search_depth"] = depth
    body = lanes.tavily_body(q, plan)
    t = time.perf_counter()
    data = await lanes._post(body)
    ms = (time.perf_counter() - t) * 1000
    res = data.get("results") or []
    return {"ms": ms, "server_s": data.get("response_time"), "credits": (data.get("usage") or {}).get("credits"),
            "n": len(res), "chars": statistics.mean([len(r.get("content") or "") for r in res]) if res else 0,
            "dated": sum(1 for r in res if r.get("published_date")), "answer": data.get("answer"),
            "keys": sorted(data), "top": (res[0].get("title") if res else None)}


async def main(a):
    est = a.reps * len(QUERIES) * (sum(lanes.COST[d] for d in DEPTHS) + 1)
    print(f"backend={config.WEB_BACKEND} base={config.TAVILY_BASE_URL} key={'set' if config.TAVILY_API_KEY else 'MISSING'}"
          f"  estimated credits: {est}")
    if not (a.yes and config.TAVILY_API_KEY):
        print("dry run (pass --yes with TAVILY_API_KEY set)")
        return
    await lanes.warm_web()
    rows = {}
    for rep in range(a.reps):
        for q, intent, text in QUERIES:
            for depth in DEPTHS:
                rows.setdefault(depth, []).append(await one(q, intent, text, depth))
            rows.setdefault("fast+answer", []).append(await one(q, intent, text, "fast", answer="basic"))
    first = rows["fast"][0]
    print("response keys:", first["keys"])
    for k, rs in rows.items():
        ms = sorted(r["ms"] for r in rs)
        print(f"{k:12s} p50 {statistics.median(ms):6.0f} ms  p90 {ms[int(0.9 * (len(ms) - 1))]:6.0f}  "
              f"credits {sorted({r['credits'] for r in rs}, key=str)}  results {statistics.mean(r['n'] for r in rs):.1f}"
              f"  content chars {statistics.mean(r['chars'] for r in rs):.0f}  dated {sum(r['dated'] for r in rs)}")
    for r in rows["fast+answer"][: len(QUERIES)]:
        print("answer:", (r["answer"] or "")[:160])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--yes", action="store_true")
    asyncio.run(main(ap.parse_args()))
