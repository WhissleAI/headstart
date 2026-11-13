"""Offline evaluation of Arc read-out search narrowing (server/narrow.py) on the live Tavily lane.

Questions: the 30 showcase questions + eval/questions.jsonl (12 current events + 38 web factoid / filler /
correction). Each heard text (`arc_text`, else the script) is fed word by word into the native Arc Tagger; the
speculator's web fire policy (>= SPEC_MIN_WORDS words, content, a new query, <= SPEC_WEB_BUDGET) is replayed on the
partial frames, then the final frame's query is sent as the final call. Every distinct Tavily request goes out once
(disk cache runs/narrow_cache.json, so re-analysis costs nothing) and its wall latency is recorded.

Variants (config overrides): base | dom_prefer | dom_restrict | query | query_dom (+ gate / rerank evaluated offline
on the same results: the gate only removes fires; the rerank only reorders).

Scores
  (a) top-5 relevance: one blind judge call per question grades every distinct result seen (shuffled, no variant
      labels) 0 / 1 / 2 against a fixed rubric -> P@5 (grade>=1), mean grade@5, top-1 = 2 rate, nDCG@5
  (b) answers (--answers A,B): the answer model on each variant's final top 5 -> keyword `expect` + grounded
  (c) credits per question; wasted speculative fires (none of its results reach the final top 5, and not the final query)
  (d) Tavily wall latency per call, paired same query with vs without include_domains
  (e) earliest word index at which the final top-1 result was already returned by a speculative fire

  nice -n 19 .venv/bin/python eval/narrow_eval.py --dry
  nice -n 19 .venv/bin/python eval/narrow_eval.py --variants base,dom_prefer --spec base,dom_prefer --judge --answers base,dom_prefer
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from server import answer, config, lanes, narrow  # noqa: E402
from server import frame as framemod  # noqa: E402
from server.speculator import _non_answer  # noqa: E402

import replay_frames  # noqa: E402
from replay_frames import clean  # noqa: E402

config.NARROW_TODAY = config.NARROW_TODAY or "2026-11-12"
CACHE = ROOT / "runs" / "narrow_cache.json"
_FAMS = ",".join(k for k in narrow.FAMILY_DOMAINS if k != "reference")
_SEL = "finance,weather,travel,reviews,local,health,recipe,home"
VARIANTS = {
    "base": dict(NARROW_DOMAINS="off", NARROW_QUERY=False, NARROW_DOMAIN_FAMILIES=_FAMS),
    "dom_prefer": dict(NARROW_DOMAINS="prefer", NARROW_QUERY=False, NARROW_DOMAIN_FAMILIES=_FAMS),
    "dom_restrict": dict(NARROW_DOMAINS="restrict", NARROW_QUERY=False, NARROW_DOMAIN_FAMILIES=_FAMS),
    "query": dict(NARROW_DOMAINS="off", NARROW_QUERY=True, NARROW_DOMAIN_FAMILIES=_FAMS),
    "query_dom": dict(NARROW_DOMAINS="prefer", NARROW_QUERY=True, NARROW_DOMAIN_FAMILIES=_FAMS),
    "query_domsel": dict(NARROW_DOMAINS="prefer", NARROW_QUERY=True, NARROW_DOMAIN_FAMILIES=_SEL),
    "ref_prefer": dict(NARROW_DOMAINS="prefer", NARROW_QUERY=False, NARROW_DOMAIN_FAMILIES="reference"),
}
JUDGE = """Today is 2026-11-12. A user asked a voice search assistant: "{q}"
Below are search results (title, site, snippet). Grade EACH result for how useful it is to answer this exact question
today:
2 = directly answers it (or clearly contains the answer) and is about the right entity / place / date
1 = on-topic and related, but does not itself answer it (or is generic / partly stale)
0 = off-topic, wrong entity, wrong place, or stale for a question about now
Reply with only a JSON list of integers, one per result, in order (length {n}).

{items}"""


def questions() -> list[dict]:
    rows = [json.loads(x) for x in (ROOT / "eval" / "showcase" / "questions.jsonl").read_text().splitlines() if x.strip()]
    rows += [json.loads(x) for x in (ROOT / "eval" / "questions.jsonl").read_text().splitlines() if x.strip()]
    return rows


def prefix_frames(text: str) -> list:
    """Frame after each word (as the live path builds them)."""
    t = replay_frames.tagger()
    t.reset()
    words = [w for w in (clean(x) for x in text.split()) if w]
    spans, current, events, prev_intent, out = [], None, set(), None, []
    for i in range(len(words)):
        t.push(words[i:i + 1])
        r = t.poll()
        a0 = r.get("a0", 0)
        spans = [s for s in spans if s["end"] <= a0] + r.get("spans", [])
        if r.get("current") is not None:
            current = r["current"]
        for u in r.get("utterances") or []:
            current = u
        for e in r.get("events") or []:
            events.add(e["type"])
        f = framemod.build(words[:i + 1], spans, current, events, prev_intent)
        if f.intent.startswith("discourse.intent."):
            prev_intent = (f.intent, f.intent_p)
        out.append(f)
    out[-1] = framemod.build(words, spans, current, events, prev_intent)
    return out


def set_cfg(over: dict):
    for k, v in over.items():
        setattr(config, k, v)


def request(f, final: bool) -> tuple[str, dict]:
    """(query, Tavily body) the lane would send for this frame under the current config."""
    q = narrow.shape_query(f, f.query())
    hints = dict(f.web_hints())
    doms, mode = narrow.domains(f)
    if doms:
        hints["include_domains"], hints["domains_mode"] = doms, mode
    plan = lanes.web_plan(f.intent, f.resolved_text, final, hints=hints)
    return q, lanes.tavily_body(q, plan)


def plan_fires(frames: list, gate: bool) -> list[dict]:
    """Replay the speculator's web fire policy on the partial frames + the final call."""
    fired, out = set(), []
    for f in frames[:-1]:
        if f.n < config.MIN_WORDS or not f.has_content() or len(out) >= config.WEB_BUDGET or f.jailbreak:
            continue
        if gate and not narrow.specific(f):
            continue
        q, body = request(f, final=False)
        if not q or q in fired:
            continue
        fired.add(q)
        out.append({"n": f.n, "query": q, "body": body, "final": False})
    q, body = request(frames[-1], final=True)
    out.append({"n": frames[-1].n, "query": q, "body": body, "final": True})
    return out


def bkey(body: dict) -> str:
    return json.dumps(body, sort_keys=True)


async def run_calls(bodies: list[dict], cache: dict, max_credits: float, conc: int = 2):
    todo = [b for b in {bkey(b): b for b in bodies}.values() if bkey(b) not in cache or cache[bkey(b)].get("error")]
    random.Random(3).shuffle(todo)          # interleave variants: no systematic warm-cache order
    spent = {"c": 0.0}
    sem = asyncio.Semaphore(conc)

    async def one(b):
        async with sem:
            if spent["c"] + lanes.COST.get(b["search_depth"], 2) > max_credits:
                cache[bkey(b)] = {"error": "budget"}
                return
            for attempt in range(5):
                t = time.perf_counter()
                try:
                    data = await lanes._post(b)
                    break
                except lanes.TavilyError as e:
                    if e.status == 429 and attempt < 4:
                        await asyncio.sleep((e.retry_after or 1) + random.random() * 1.5 * (attempt + 1))
                        continue
                    cache[bkey(b)] = {"error": f"{type(e).__name__}: {e}"[:200]}
                    return
                except Exception as e:
                    cache[bkey(b)] = {"error": f"{type(e).__name__}: {e}"[:200]}
                    return
            ms = (time.perf_counter() - t) * 1000
            cr = (data.get("usage") or {}).get("credits", lanes.COST.get(b["search_depth"], 2))
            spent["c"] += cr
            cache[bkey(b)] = {"ms": ms, "response_time": data.get("response_time"), "credits": cr,
                              "results": [{"url": x.get("url"), "title": x.get("title"), "content": (x.get("content") or "")[:700],
                                           "score": x.get("score"), "published_date": x.get("published_date")}
                                          for x in data.get("results") or []]}
    await asyncio.gather(*[one(b) for b in todo])
    return spent["c"], len(todo)


def docs_of(entry: dict) -> list:
    out = []
    for x in (entry or {}).get("results") or []:
        out.append(lanes.Doc(id=x["url"], title=x.get("title") or "", text=x.get("content") or "", url=x["url"],
                             source="tavily", score=float(x.get("score") or 0),
                             meta={"published_date": x["published_date"]} if x.get("published_date") else {}))
    return out


def ndcg(grades: list[int], k=5) -> float:
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))
    ideal = sorted(grades, reverse=True)
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    return dcg / idcg if idcg else 0.0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base,dom_prefer,dom_restrict,query,query_dom")
    ap.add_argument("--spec", default="base,query_dom", help="variants whose speculative fires go out too")
    ap.add_argument("--max-credits", type=float, default=280)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--answers", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--spec-ids", default="sc,current", help="id prefixes / kinds whose speculative fires go out")
    ap.add_argument("--out", default=str(ROOT / "runs" / "narrow_eval.json"))
    a = ap.parse_args()
    variants = a.variants.split(",")
    spec_v = set(a.spec.split(",")) if a.spec else set()
    qs = questions()
    if a.only:
        qs = [r for r in qs if r["id"] in a.only.split(",")]
    items = []
    for r in qs:
        heard = r.get("arc_text") or r["text"]
        frames = prefix_frames(heard)
        it = {"id": r["id"], "kind": r.get("kind"), "heard": heard, "expect": r.get("expect") or [],
              "n_words": frames[-1].n, "family": narrow.family(frames[-1]), "frames": frames, "v": {}}
        for v in variants:
            set_cfg(VARIANTS[v])
            fires = plan_fires(frames, gate=False)
            gated = {x["query"] for x in plan_fires(frames, gate=True)}
            for x in fires:
                x["gated_out"] = x["query"] not in gated and not x["final"]
            spec_ok = any(r["id"].startswith(p) or r.get("kind") == p for p in a.spec_ids.split(","))
            if v not in spec_v or not spec_ok:
                fires = [x for x in fires if x["final"]]
            it["v"][v] = {"fires": fires}
        set_cfg(VARIANTS["base"])
        items.append(it)
    bodies = [x["body"] for it in items for v in it["v"].values() for x in v["fires"]]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    need = {bkey(b) for b in bodies} - set(cache)
    print(f"{len(items)} questions; {len({bkey(b) for b in bodies})} distinct requests, {len(need)} not cached")
    fam = {}
    for it in items:
        fam.setdefault(it["family"] or "-", []).append(it["id"])
    print("families:", {k: len(v) for k, v in fam.items()})
    if a.dry:
        for it in items:
            print(it["id"], it["family"], "|", " || ".join(f"{v}: " + " ; ".join(
                f"{x['n']}:{x['query']}{' [D]' if x['body'].get('include_domains') else ''}{' -G' if x.get('gated_out') else ''}"
                for x in it["v"][v]["fires"]) for v in variants if v in ("base", "query_dom")))
        return
    spent, n = await run_calls(bodies, cache, a.max_credits)
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(cache))
    print(f"sent {n} requests, {spent} credits")

    # ---------------------------------------------------------------------------------------------- judge
    jpath = ROOT / "runs" / "narrow_judge.json"
    grades: dict = json.loads(jpath.read_text()) if jpath.exists() else {}
    sem = asyncio.Semaphore(4)

    async def llm(system, content, max_tokens):
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

    async def judge(it):
        seen = {}
        for v in it["v"].values():
            for x in v["fires"]:
                for d in docs_of(cache.get(bkey(x["body"]))):
                    seen.setdefault(d.url, d)
        g = grades.setdefault(it["id"], {})
        new = [d for u, d in seen.items() if u not in g]
        if not new:
            return
        random.Random(it["id"]).shuffle(new)
        for i in range(0, len(new), 25):
            chunk = new[i:i + 25]
            txt = "\n".join(f"[{j + 1}] {d.title} ({narrow._host(d.url)}"
                            f"{', ' + str(d.meta.get('published_date'))[:16] if d.meta.get('published_date') else ''})"
                            f": {d.text[:260]}" for j, d in enumerate(chunk))
            out = await llm("You are a strict search-quality rater.", JUDGE.format(q=it["heard"], n=len(chunk), items=txt),
                            max_tokens=200)
            try:
                arr = json.loads(out[out.index("["):out.rindex("]") + 1])
            except ValueError:
                arr = []
            if len(arr) == len(chunk):
                for d, s in zip(chunk, arr):
                    g[d.url] = int(s)
    if a.judge:
        await asyncio.gather(*[judge(it) for it in items])
        jpath.write_text(json.dumps(grades, indent=0))

    # ---------------------------------------------------------------------------------------------- metrics
    report = {"n_questions": len(items), "variants": {}, "credits_spent_this_run": spent}
    for v in variants:
        rows = []
        for it in items:
            fires = it["v"][v]["fires"]
            f_final = it["frames"][-1]
            set_cfg(VARIANTS[v])
            fin = next(x for x in fires if x["final"])
            ent = cache.get(bkey(fin["body"])) or {}
            docs = docs_of(ent)
            g = grades.get(it["id"], {})
            row = {"id": it["id"], "family": it["family"], "query": fin["query"],
                   "domains": bool(fin["body"].get("include_domains")), "ms": ent.get("ms"),
                   "n_results": len(docs), "error": ent.get("error")}
            for name, order in (("tavily", docs), ("rerank", None)):
                if order is None:
                    config.NARROW_RERANK = True
                    order = narrow.rerank(f_final, docs)
                    config.NARROW_RERANK = False
                gr = [g.get(d.url) for d in order[:5]]
                if g and all(x is not None for x in gr):
                    gr5 = gr + [0] * (5 - len(gr))
                    pool = sorted([g[d.url] for d in docs if d.url in g], reverse=True)
                    row[name] = {"p5": sum(x >= 1 for x in gr5) / 5, "g5": sum(gr5) / 5, "top1": int(bool(gr) and gr[0] == 2),
                                 "ndcg": ndcg(gr5) if pool else 0.0, "any2": int(any(x == 2 for x in gr5))}
                row[name + "_urls"] = [d.url for d in order[:5]]
            # speculative fires: wasted / useful, earliest word with the final top-1
            spec = [x for x in fires if not x["final"]]
            if spec:
                top5 = set(row["rerank_urls"] if config.NARROW_RERANK else row["tavily_urls"])
                top1 = (row["tavily_urls"] or [None])[0]
                top1r = (row["rerank_urls"] or [None])[0]
                fs = []
                for x in spec:
                    urls = [d.url for d in docs_of(cache.get(bkey(x["body"])))]
                    useful = x["query"] == fin["query"] or bool(set(urls) & set(row["tavily_urls"]))
                    rel = max([g.get(u, 0) for u in urls] or [0])
                    fs.append({"n": x["n"], "query": x["query"], "useful": useful, "best_grade": rel,
                               "gated_out": x["gated_out"], "has_top1": top1 in urls, "has_top1_rerank": top1r in urls,
                               "credits": (cache.get(bkey(x["body"])) or {}).get("credits", 0)})
                row["spec"] = fs
                e = [x["n"] for x in fs if x["has_top1"]]
                row["earliest_top1_word"] = min(e) if e else None
                er = [x["n"] for x in fs if x["has_top1_rerank"]]
                row["earliest_top1r_word"] = min(er) if er else None
                eg = [x["n"] for x in fs if x["has_top1"] and not x["gated_out"]]
                row["earliest_top1_word_gated"] = min(eg) if eg else None
            row["n_words"] = it["n_words"]
            rows.append(row)
        set_cfg(VARIANTS["base"])
        report["variants"][v] = {"rows": rows, "summary": summarize(rows)}
    if a.answers:
        await answers(a.answers.split(","), items, report, cache, llm)
    report["latency"] = latency_pairs(items, cache, variants)
    for it in items:
        it.pop("frames")
        for v in it["v"].values():
            for x in v["fires"]:
                x.pop("body", None)
    Path(a.out).write_text(json.dumps(report, indent=1, default=str))
    print_report(report)


def summarize(rows: list[dict]) -> dict:
    out = {}
    for name in ("tavily", "rerank"):
        rs = [r[name] for r in rows if name in r]
        if rs:
            out[name] = {k: round(statistics.mean(x[k] for x in rs), 3) for k in ("p5", "g5", "top1", "ndcg", "any2")}
            out[name]["n"] = len(rs)
    ms = [r["ms"] for r in rows if r.get("ms")]
    out["final_ms_p50"] = round(statistics.median(ms)) if ms else None
    out["empty"] = sum(r["n_results"] == 0 for r in rows)
    sp = [x for r in rows for x in r.get("spec", [])]
    if sp:
        out["spec_fires"] = len(sp)
        out["spec_useful"] = sum(x["useful"] for x in sp)
        out["spec_gated_out"] = sum(x["gated_out"] for x in sp)
        out["spec_gated_out_useful"] = sum(x["gated_out"] and x["useful"] for x in sp)
        nq = sum(1 for r in rows if "spec" in r)
        out["spec_credits_per_q"] = round(sum(x["credits"] or 0 for x in sp) / nq, 2)
        out["spec_credits_per_q_gated"] = round(sum((x["credits"] or 0) for x in sp if not x["gated_out"]) / nq, 2)
        for key in ("earliest_top1_word", "earliest_top1r_word", "earliest_top1_word_gated"):
            e = [r for r in rows if "spec" in r]
            got = [r[key] for r in e if r.get(key) is not None]
            out[key] = {"found_before_end": len(got), "of": len(e),
                        "words_before_end": sorted(r["n_words"] - r[key] for r in e if r.get(key) is not None),
                        "frac_of_utterance_p50": round(statistics.median(r[key] / r["n_words"] for r in e
                                                                         if r.get(key) is not None), 2) if got else None}
    return out


def latency_pairs(items, cache, variants) -> dict:
    """Same query, with vs without include_domains (paired), wall ms of the first (uncached-by-us) call."""
    by_q: dict = {}
    for it in items:
        for v in variants:
            for x in it["v"][v]["fires"]:
                e = cache.get(bkey(x["body"])) or {}
                if not e.get("ms"):
                    continue
                k = (x["query"], x["body"]["topic"], x["body"]["search_depth"])
                mode = x["body"].get("include_domains_mode") or "none"
                by_q.setdefault(k, {})[mode] = (e["ms"], e.get("response_time"))
    out = {}
    for mode in ("prefer", "restrict"):
        pairs = [(d["none"], d[mode]) for d in by_q.values() if "none" in d and mode in d]
        if pairs:
            diff = [b[0] - a[0] for a, b in pairs]
            out[mode] = {"n": len(pairs), "none_p50": round(statistics.median(a[0] for a, _ in pairs)),
                         f"{mode}_p50": round(statistics.median(b[0] for _, b in pairs)),
                         "paired_diff_p50": round(statistics.median(diff)),
                         "tavily_rt_none_p50": statistics.median([float(a[1] or 0) for a, _ in pairs]),
                         f"tavily_rt_{mode}_p50": statistics.median([float(b[1] or 0) for _, b in pairs])}
    allms = [e["ms"] for e in cache.values() if e.get("ms") and not e.get("error")]
    out["all_calls_p50"] = round(statistics.median(allms)) if allms else None
    return out


async def answers(vs: list[str], items, report, cache, llm):
    """Answer model on each variant's final top 5 (rerank order when --answers names 'X+rr')."""
    out = {}
    for spec in vs:
        v, rr = spec.split("+")[0], spec.endswith("+rr")
        rows = {r["id"]: r for r in report["variants"][v]["rows"]}
        res = []

        async def one(it):
            r = rows[it["id"]]
            urls = r["rerank_urls" if rr else "tavily_urls"]
            fin = next(x for x in it["v"][v]["fires"] if x["final"])
            docs = {d.url: d for d in docs_of(cache.get(bkey(fin["body"])))}
            ds = [docs[u] for u in urls if u in docs]
            text = await llm(answer.SYSTEM, answer.build_prompt(it["frames"][-1].resolved_text, ds), 160) if ds else ""
            ok = None
            if it["expect"]:
                low = text.lower()
                ok = all(any(alt in low for alt in e.split("|")) for e in it["expect"])
            res.append({"id": it["id"], "answer": text, "correct": ok,
                        "grounded": bool(text) and not _non_answer(text) and not text.startswith("<error")})
        await asyncio.gather(*[one(it) for it in items])
        sc = [x for x in res if x["correct"] is not None]
        out[spec] = {"n": len(res), "grounded": sum(x["grounded"] for x in res),
                     "correct": sum(bool(x["correct"]) for x in sc), "scored": len(sc), "rows": res}
    report["answers"] = out


def print_report(rep):
    print(f"\n{'variant':14s} {'P@5':>5s} {'g@5':>5s} {'top1':>5s} {'nDCG':>5s} | rerank P@5 g@5 top1 nDCG | ms p50 empty")
    for v, d in rep["variants"].items():
        s = d["summary"]
        t, r = s.get("tavily", {}), s.get("rerank", {})
        print(f"{v:14s} {t.get('p5', 0):5.3f} {t.get('g5', 0):5.3f} {t.get('top1', 0):5.3f} {t.get('ndcg', 0):5.3f} |"
              f" {r.get('p5', 0):5.3f} {r.get('g5', 0):5.3f} {r.get('top1', 0):5.3f} {r.get('ndcg', 0):5.3f} |"
              f" {s.get('final_ms_p50')} {s.get('empty')}")
        if "spec_fires" in s:
            print("   spec:", {k: s[k] for k in s if k.startswith("spec_")},
                  {k: {kk: vv for kk, vv in s[k].items() if kk != 'words_before_end'} for k in s if k.startswith("earliest")})
    print("latency:", rep["latency"])
    for k, v in (rep.get("answers") or {}).items():
        print("answers", k, {kk: vv for kk, vv in v.items() if kk != "rows"})


if __name__ == "__main__":
    asyncio.run(main())
