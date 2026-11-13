"""Offline before/after replay of answer quality — no audio, no ear, no reranker.

For each A/B row it takes the text the system actually answered (`question_heard`, the ASR words of the answered
utterance), rebuilds the final frame with the native Arc Tagger (word by word, as live), then for each variant:
query → real Wikipedia → real answer model (server/answer.py's model + options) → keyword score + grounding check.

  before = frame.py / answer.py SYSTEM / Wikipedia lane at --base (default: the commit this branch started from)
  after  = the working tree (entity query, relaxed retry, corrected question, new prompt)

Sets: "wrong" = every (question, heard text) that was wrong or ungrounded (the answer says the sources lack it) in
any listed run, either mode; "regress" = N questions right and grounded in both modes of the reference run.
The reranker is skipped in both variants: docs go to the answer model in lane order (content query first, entity
articles appended), top --k. Wikipedia results are cached per query string within a run, so both variants see the
same results for the same query.

  nice -n 19 .venv/bin/python eval/replay_answers.py --runs-dir ../headstart/runs --out runs/replay_answers.json
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import time

import httpx
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from run_ab import score  # noqa: E402

from server import answer, config, lanes  # noqa: E402
from server import frame as frame_new  # noqa: E402
from server.speculator import _non_answer  # noqa: E402

import replay_frames  # noqa: E402

RUNS = ["ab_full1", "ab_full2", "ab_full3", "ab_snr20", "ab_snr20_post", "ab_snr10", "ab_rate_fast", "ab_rate_slow"]
# ab_rate08 / ab_rate125 are excluded: their audio and question ids are misaligned (q01 "who founded Nvidia" was heard
# as "tell me what is the calendar in new york"), 38/40 rows wrong in both modes: a harness artefact, not the system.


def load_old(base: str):
    """frame.py and the answer SYSTEM prompt as of `base`."""
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{base}:server/frame.py"], capture_output=True, text=True,
                         check=True).stdout
    tmp = Path(tempfile.mkdtemp()) / "frame_old.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("frame_old", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["frame_old"] = mod
    spec.loader.exec_module(mod)
    ans = subprocess.run(["git", "-C", str(ROOT), "show", f"{base}:server/answer.py"], capture_output=True, text=True,
                         check=True).stdout
    system = re.search(r'SYSTEM = """(.*?)"""', ans, re.S).group(1)
    return mod, system


def load_questions():
    qs = {}
    for p in [ROOT / "eval" / "questions.jsonl", ROOT / "index" / "questions_local.jsonl"]:
        for line in open(p):
            j = json.loads(line)
            qs[j["id"]] = j
    return qs


def collect(runs_dir: Path, ref: str, n_regress: int):
    qs = load_questions()
    wrong, keys = [], set()
    rows_by_run = {}
    for run in RUNS:
        p = runs_dir / f"{run}.json"
        if not p.exists():
            continue
        rows_by_run[run] = json.load(open(p))["rows"]
        for r in rows_by_run[run]:
            if not r.get("question_heard") and "|" not in (r.get("arc_heard") or "|"):
                r["question_heard"] = r["arc_heard"].rstrip("?.!")
            if r["id"].startswith("local") or not r.get("question_heard"):
                continue
            bad = r["correct"] is False or _non_answer(r.get("answer") or "")
            k = (r["id"], r["question_heard"].strip())
            if bad and k not in keys:
                keys.add(k)
                wrong.append({"id": r["id"], "heard": k[1], "text": r["text"], "expect": qs[r["id"]].get("expect"),
                              "seen_in": f"{run}/{r['mode']}", "rec_query": r.get("query"),
                              "rec_answer": r.get("answer"), "rec_correct": r["correct"]})
    ok_ids = {}
    for r in rows_by_run.get(ref, []):
        if r["id"].startswith("local") or not qs[r["id"]].get("expect"):
            continue
        good = r["correct"] is True and not _non_answer(r.get("answer") or "")
        ok_ids.setdefault(r["id"], []).append((good, r))
    regress = []
    for qid in sorted(ok_ids):
        if not all(g for g, _ in ok_ids[qid]) or len(ok_ids[qid]) < 2:
            continue
        r = next(r for _, r in ok_ids[qid] if r["mode"] == "headstart")
        if (qid, r["question_heard"].strip()) in keys:      # the same heard text went wrong in another run
            continue
        regress.append({"id": qid, "heard": r["question_heard"].strip(), "text": r["text"],
                        "expect": qs[qid].get("expect"), "seen_in": f"{ref}/headstart", "rec_query": r.get("query"),
                        "rec_answer": r.get("answer"), "rec_correct": True})
    return wrong, regress[:n_regress]


class Budget:
    def __init__(self, wiki: int, llm: int):
        self.wiki_max, self.llm_max, self.wiki, self.llm = wiki, llm, 0, 0

    def take(self, kind: str):
        setattr(self, kind, getattr(self, kind) + 1)
        if getattr(self, kind) > getattr(self, kind + "_max"):
            raise SystemExit(f"budget exceeded: {kind}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--ref", default="ab_full3", help="run whose all-right questions form the regression set")
    ap.add_argument("--base", default="af29048")
    ap.add_argument("--n-regress", type=int, default=20)
    ap.add_argument("--k", type=int, default=5, help="docs handed to the answer model")
    ap.add_argument("--max-wiki", type=int, default=300)
    ap.add_argument("--max-llm", type=int, default=150)
    ap.add_argument("--regress-before-replay", action="store_true",
                    help="replay the old pipeline on the regression set too (default: its recorded live answers)")
    ap.add_argument("--dry", action="store_true", help="frames + queries only (no network)")
    ap.add_argument("--out", default=str(ROOT / "runs" / "replay_answers.json"))
    a = ap.parse_args()

    frame_old, system_old = load_old(a.base)
    wrong, regress = collect(Path(a.runs_dir), a.ref, a.n_regress)
    budget = Budget(a.max_wiki, a.max_llm)
    cache: dict[str, asyncio.Task] = {}

    wsem = asyncio.Semaphore(2)          # Wikipedia answers 429 to a burst of ~100 parallel searches

    async def wiki(q: str, k: int):
        key = f"{k}|{q}"
        if key not in cache:
            async def go():
                async with wsem:
                    for attempt in range(4):
                        budget.take("wiki")
                        try:
                            return await lanes._wiki_search(q, k)
                        except httpx.HTTPStatusError as e:
                            if e.response.status_code != 429 or attempt == 3:
                                raise
                            await asyncio.sleep(2 ** (attempt + 1))
            cache[key] = asyncio.ensure_future(go())
        return await cache[key]

    async def wiki_new(q: str, entity: str):
        # lanes.wikipedia's logic, through the cache
        ent = entity.strip()
        if ent and ent != q.strip():
            main_, extra = await asyncio.gather(wiki(q, 4), wiki(ent, 2))
        else:
            main_, extra = await wiki(q, 4), []
        if not main_ and not extra and lanes.relaxed(q):
            main_ = await wiki(lanes.relaxed(q), 4)
        seen = {d.id for d in main_}
        return main_ + [d for d in extra if d.id not in seen]

    sem = asyncio.Semaphore(4)

    async def ask(system: str, question: str, docs) -> str:
        budget.take("llm")
        async with sem:
            for attempt in range(3):
                try:
                    r = await answer.client().messages.create(
                        model=config.ANSWER_MODEL, max_tokens=160, system=system,
                        **answer.model_opts(config.ANSWER_MODEL),
                        messages=[{"role": "user", "content": answer.build_prompt(question, docs)}])
                    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
                except Exception as e:  # 529 / transient
                    if attempt == 2:
                        return f"<error {type(e).__name__}>"
                    await asyncio.sleep(1.5 * (attempt + 1))

    async def one(item: dict, set_name: str):
        # frames (tagger is synchronous and shared: build both before any await)
        replay_frames.framemod = frame_old
        fo = replay_frames.frame_for(item["heard"])
        replay_frames.framemod = frame_new
        fn = replay_frames.frame_for(item["heard"])
        out = {"set": set_name, **{k: item[k] for k in ("id", "text", "heard", "expect", "seen_in", "rec_query")},
               "query_fidelity": fo.query() == item["rec_query"],
               "before": {"query": fo.query(), "question": fo.resolved_text},
               "after": {"query": fn.query(), "entity": fn.entity_query(), "question": fn.resolved_text}}
        if a.dry:
            return out
        docs_a = (await wiki_new(fn.query(), fn.entity_query()))[: a.k]
        if set_name == "regress" and not a.regress_before_replay:
            docs_b, ans_b = None, item["rec_answer"]           # the live answer: right and grounded
            ans_a = await ask(answer.SYSTEM, fn.resolved_text, docs_a)
        else:
            docs_b = (await wiki(fo.query(), 4))[: a.k]
            ans_b, ans_a = await asyncio.gather(ask(system_old, fo.resolved_text, docs_b),
                                                ask(answer.SYSTEM, fn.resolved_text, docs_a))
        for key, docs, ans in (("before", docs_b, ans_b), ("after", docs_a, ans_a)):
            out[key].update(sources=[d.title for d in docs] if docs is not None else "(live run)", answer=ans, correct=score(ans, item["expect"]),
                            non_answer=_non_answer(ans))
        return out

    t0 = time.time()
    # frames first (synchronous), then the network part concurrently
    results = await asyncio.gather(*[one(it, "wrong") for it in wrong], *[one(it, "regress") for it in regress],
                                   return_exceptions=True)
    for r in results:
        if isinstance(r, BaseException):
            print("item failed:", type(r).__name__, str(r)[:200])
    results = [r for r in results if not isinstance(r, BaseException)]
    summary = {}
    for s in ("wrong", "regress"):
        rs = [r for r in results if r["set"] == s]
        summary[s] = {"n": len(rs), "query_fidelity": sum(r["query_fidelity"] for r in rs)}
        if not a.dry:
            for v in ("before", "after"):
                summary[s][v] = {"correct": sum(1 for r in rs if r[v]["correct"]),
                                 "grounded_correct": sum(1 for r in rs if r[v]["correct"] and not r[v]["non_answer"]),
                                 "non_answer": sum(1 for r in rs if r[v]["non_answer"])}
    summary["calls"] = {"wikipedia": budget.wiki, "answer_model": budget.llm, "seconds": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "rows": results}, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1))
    for r in results:
        if a.dry:
            print(f"{r['set'][:3]} {r['id']} fid={r['query_fidelity']} {r['before']['query']!r} -> {r['after']['query']!r}"
                  f" ent={r['after']['entity']!r}")
        else:
            b, af = r["before"], r["after"]
            mark = lambda v: ("ok" if v["correct"] else "X ") + ("~" if v["non_answer"] else " ")  # noqa: E731
            print(f"{r['set'][:3]} {r['id']} {mark(b)} -> {mark(af)} | {af['query']!r} ent={af['entity']!r}")


if __name__ == "__main__":
    asyncio.run(main())
