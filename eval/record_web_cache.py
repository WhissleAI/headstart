"""Record the web lane's responses for the eval questions into the replay cache (runs/web_cache.json by default).

For each web question it replays the words one at a time through Arc Tagger + the frame builder (what the speculator
would search: the final query first, then the speculative partial queries) and makes ONE real call per distinct
query, up to --max-calls. The server serves these when WEB_BACKEND=cache, or when the live lane fails (flagged as
cached everywhere). Usage:
    .venv/bin/python eval/record_web_cache.py --dry-run          # list the queries, no network
    .venv/bin/python eval/record_web_cache.py --max-calls 60     # record (Wikipedia; keyless)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import config, frame as framemod, lanes  # noqa: E402
from server.session import clean  # noqa: E402
from server.tagger import Tagger  # noqa: E402


def queries_for(tagger: Tagger, text: str) -> tuple[str | None, list[str]]:
    """(final query, speculative queries in order) for one spoken question."""
    tagger.reset()
    words = [clean(w) for w in text.split()]
    words = [w for w in words if w]
    spans, current, spec = [], None, []

    def apply(r):
        nonlocal spans, current
        a0 = r.get("a0", 0)
        spans = [s for s in spans if s["end"] <= a0] + r.get("spans", [])
        if r.get("current") is not None:
            current = r["current"]
        for u in r.get("utterances") or []:
            current = u

    for i, w in enumerate(words):
        tagger.push([w])
        apply(tagger.poll())
        f = framemod.build(words[:i + 1], spans, current, set(), None, local_ok=False)
        if f.n >= config.MIN_WORDS and f.has_content() and f.route in ("web", "both"):
            q = f.query()
            if q and q not in spec:
                spec.append(q)
    tagger.endpoint()
    apply(tagger.poll())
    f = framemod.build(words, spans, current, set(), None, local_ok=False)
    return (f.query() or None), spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-calls", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(config.WEB_CACHE_PATH))
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    a = ap.parse_args()

    rows = [json.loads(x) for x in Path(a.questions).read_text().splitlines() if x.strip()]
    tagger = Tagger(domain="assistant")
    finals, partials = [], []
    for r in rows:
        if r.get("kind") == "local":
            continue
        for text in dict.fromkeys([r.get("arc_text") or r["text"], r["text"]]):
            fq, spec = queries_for(tagger, text)
            if fq:
                finals.append((r["id"], fq))
            partials += [(r["id"], q) for q in spec]
    plan, seen = [], set()
    for qid, q in finals + partials:                    # finals first: they are what an answer is built from
        k = lanes.norm_query(q)
        if k not in seen:
            seen.add(k)
            plan.append((qid, q))
    print(f"{len(rows)} questions -> {len(finals)} final + {len(partials)} speculative queries, {len(plan)} distinct; "
          f"recording up to {a.max_calls}")
    plan = plan[: a.max_calls]
    if a.dry_run:
        for qid, q in plan:
            print(f"  {qid}: {q}")
        return

    cache = lanes.WebCache(Path(a.out))
    before = len(cache)

    async def go():
        ok = fail = 0
        for qid, q in plan:
            if lanes.norm_query(q) in cache.entries:
                continue
            try:
                docs = await lanes.wikipedia(q)
            except Exception as e:
                fail += 1
                print(f"  {qid}: {q!r} FAILED {type(e).__name__}: {e}")
                continue
            cache.put(q, docs, "wikipedia")
            ok += 1
            print(f"  {qid}: {q!r} -> {len(docs)} docs: {', '.join(d.title for d in docs[:3])}")
            await asyncio.sleep(0.2)               # be polite to Wikipedia
        await lanes.client().aclose()
        return ok, fail

    ok, fail = asyncio.run(go())
    cache.save()
    print(f"recorded {ok} (failed {fail}); cache {before} -> {len(cache)} entries at {a.out}")


if __name__ == "__main__":
    main()
