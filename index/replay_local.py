"""End-to-end replay of the local-moment questions: real Arc ear traces -> real Session/Speculator (virtual time, sim/)
-> the sources and question the answer model would get -> the REAL answer model -> scored like eval/run_ab.py.

    .venv/bin/python index/replay_local.py --configs before,after --modes headstart,baseline --out runs/replay_local.json
    .venv/bin/python index/replay_local.py --no-llm            # ranking only (no API calls)

What is real: Arc's words and timing (sim/data/ear_traces/local*.json), Arc Tagger, frames, speculator, the local index
and its vectors, query text vectors, the spoken question's AUDIO (fed from eval/audio/local_*.wav at the trace's audio
times, so the audio lane embeds exactly what the server would), and the answer model (ANSWER_MODEL, same options as the
server, non-streamed). Simulated: latencies (sim/fakes.py), so the path (draft-reuse / speculative-hit / ...) is the
simulator's, not a wall-clock run's. The web lane is empty (local-routed questions don't use it).

Configs:
  before : pool-cosine ranking (RERANK_LOCAL=0) + the answer prompt as of main@af29048 (PROMPT_V0 below)
  after  : server/rerank.py fused ranking + the current server/answer.py prompt
Scoring: eval/run_ab.py score() with the question's "expect" (index/questions_local.jsonl), and with the original,
stricter expect lists of local01-10 (EXPECT_V0) for comparison with runs/ab_full*.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_ab import score  # noqa: E402
from index.eval_local import load_wav  # noqa: E402
from sim import fakes, simulate  # noqa: E402

AUDIO_CACHE = ROOT / "runs" / "replay_audio_vecs.pkl"
EXPECT_V0 = {"local01": ["hunter", "nails"], "local06": ["15", "seconds"], "local08": ["winner", "won"],
             "local09": ["prep", "wrestling", "boxing"]}

SYSTEM_V0 = """You are the answer engine of a voice search assistant. The user spoke a question; you get search results.
The question text comes from live speech recognition and may contain sound-alike errors ("a polio eleven" = "Apollo
11", "iPhone tower" = "Eiffel Tower") and spoken self-corrections ("Canada, actually make that Turkey"): answer the
question the user most plausibly meant. Never mention mishearing, corrections, or what the user "likely means".
Answer in one or two short spoken-style sentences (under 45 words), most important fact first. Cite sources inline
as [1], [2] by their number. If the sources don't contain the answer, say so in one sentence and give the closest
relevant fact. No preamble, no markdown."""


def build_prompt_v0(question, docs, context=""):
    lines = []
    for i, d in enumerate(docs, 1):
        where = f" (moment at {int(d.t // 60):02d}:{int(d.t % 60):02d})" if d.t is not None else ""
        text = d.text
        meta = getattr(d, "meta", None) or {}
        if meta.get("context_before") or meta.get("context_after"):
            text = f"…{meta.get('context_before', '')} [{d.text}] {meta.get('context_after', '')}…"
            if meta.get("role"):
                text = f"({meta['role']}) " + text
        lines.append(f"[{i}] {d.title}{where}: {text[:1000]}")
    src = "\n".join(lines) if lines else "(no results)"
    prev = f"Previous turn (for resolving 'it', 'he', 'there'):\n{context}\n\n" if context else ""
    return f"{prev}Sources:\n{src}\n\nSpoken question: {question}"


class RealAudio:
    """encode_audio for the replay: the real model on the real waveform slice, cached by (file, samples)."""

    def __init__(self):
        self.c = pickle.loads(AUDIO_CACHE.read_bytes()) if AUDIO_CACHE.exists() else {}
        self.dirty = False

    def vec(self, wav: np.ndarray) -> np.ndarray:
        k = (len(wav), hash(wav.tobytes()))
        if k not in self.c:
            from server import embed
            self.c[k] = embed.encode_audio([wav])[0]
            self.dirty = True
        return self.c[k]

    def save(self):
        if self.dirty:
            AUDIO_CACHE.write_bytes(pickle.dumps(self.c))


class Replay(simulate.Simulator):
    def __init__(self, audio: RealAudio):
        super().__init__()
        self.audio = audio
        self.wav = None
        self.captured: list = []
        from server import lanes, speculator

        def W():
            return self.world["w"]

        async def local_audio(wav, k=4):
            w = W()
            async with w.glock:
                lo, hi = w.profile.audio_ms
                await asyncio.sleep((lo + (hi - lo) * w.sampler.u("audio")[0]) / 1000)
            q = self.audio.vec(wav)
            return (lanes.index().search_vec(q, k) if len(lanes.index()) else []), q

        self._set(lanes, "local_audio", local_audio)
        orig_finalize = speculator.Speculator.finalize

        async def finalize(spec_self, f, t_speech_end, close=True, by="ear"):
            m = await orig_finalize(spec_self, f, t_speech_end, close=close, by=by)
            st = spec_self.answer_stream
            st = getattr(st, "primary", st)
            if st is not None and not m.get("superseded"):
                self.captured.append((m, st.question, list(st.docs), st.context, f.text))
            return m
        self._set(speculator.Speculator, "finalize", finalize)

    def _set(self, obj, name, val):
        self._orig.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    async def _replay(self, trace, mode):
        """As Simulator._replay, but the session hears the real waveform (the audio lane embeds real speech)."""
        from server.session import Session
        events = []
        t0 = time.perf_counter()
        w = self.world["w"]
        if w.qlock is None:
            w.qlock, w.glock = asyncio.Lock(), asyncio.Lock()
        sess = Session(lambda e: events.append(e), mode=mode, t0=t0)
        sess.t0 = sess.spec.t0 = t0
        finals = []
        for e in trace["events"]:
            d = t0 + e["w"] - time.perf_counter()
            if d > 0:
                await asyncio.sleep(d)
            if e["k"] == "audio":
                end = int(round(e["at"] * 16000))
                pcm = self.wav[max(0, end - e["n"]): end]
                if len(pcm) < e["n"]:
                    pcm = np.concatenate([pcm, np.zeros(e["n"] - len(pcm), np.float32)])
                sess.on_audio(pcm.astype(np.float32), e["at"])
            elif e["k"] == "partial":
                sess.on_partial(e["words"], time.perf_counter())
            else:
                finals.append(asyncio.create_task(sess.on_final(e["words"])))
        if finals:
            await asyncio.gather(*finals, return_exceptions=True)
        seen = set()
        while getattr(sess, "final_task", None) is not None and id(sess.final_task) not in seen:
            seen.add(id(sess.final_task))
            await asyncio.gather(sess.final_task, return_exceptions=True)
        sess.close()
        return sess, events, []

    def one(self, trace, mode, seed=0):
        self.wav = load_wav(ROOT / "eval" / "audio" / trace["file"])
        self.captured = []
        row = self.run_question(trace, mode, {}, fakes.PROFILES["wikipedia"], seed)
        last = self.captured[-1] if self.captured else None
        return row, last


async def ask(question, docs, context, v0: bool):
    from server import answer, config
    prompt = (build_prompt_v0 if v0 else answer.build_prompt)(question, docs, context)
    r = await answer.client().messages.create(model=config.ANSWER_MODEL, max_tokens=160,
                                              system=SYSTEM_V0 if v0 else answer.SYSTEM,
                                              **answer.model_opts(config.ANSWER_MODEL),
                                              messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="before,after")
    ap.add_argument("--modes", default="headstart,baseline")
    ap.add_argument("--ids", default="")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--max-calls", type=int, default=100)
    ap.add_argument("--out", default=str(ROOT / "runs" / "replay_local.json"))
    a = ap.parse_args()

    from server import config, embed, lanes
    qs = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "index/questions_local.jsonl").read_text().splitlines()
          if l.strip()}
    ids = [i for i in (a.ids.split(",") if a.ids else qs)]
    traces = {t["id"]: t for t in simulate.load_traces(ids=set(ids))}
    missing = [i for i in ids if i not in traces]
    if missing:
        print("no ear trace (run sim/record_traces.py --ids ...):", ",".join(missing))
    lanes.index()
    embed.model()
    audio = RealAudio()
    rp = Replay(audio)
    rows, jobs = [], []
    try:
        for cfg in a.configs.split(","):
            config.RERANK_LOCAL = cfg != "before"
            for qid in ids:
                if qid not in traces:
                    continue
                for mode in a.modes.split(","):
                    row, last = rp.one(traces[qid], mode)
                    r = {"config": cfg, "id": qid, "mode": mode, "path": row["path"], "heard": row["question"],
                         "src_ids": row["src_ids"], "gold_ids": qs[qid]["gold_ids"]}
                    gold = set(qs[qid]["gold_ids"])
                    r["gold_rank"] = next((j + 1 for j, s in enumerate(row["src_ids"]) if s in gold), None)
                    if last:
                        r["answer_question"] = last[1]
                        jobs.append((r, last))
                    rows.append(r)
                    print(cfg, qid, mode, r["path"], "gold@", r["gold_rank"], "|", row["question"])
    finally:
        rp.uninstall()
        rp.caches.save()
        audio.save()

    if not a.no_llm:
        jobs = jobs[: a.max_calls]

        async def run_all():
            sem = asyncio.Semaphore(4)

            async def one(r, last):
                async with sem:
                    for attempt in range(3):
                        try:
                            r["answer"] = await ask(last[1], last[2], last[3], v0=r["config"] == "before")
                            return
                        except Exception as e:      # 529 etc.: retry
                            r["answer_error"] = f"{type(e).__name__}"
                            await asyncio.sleep(2 + 2 * attempt)
            await asyncio.gather(*(one(r, last) for r, last in jobs))
        asyncio.run(run_all())
        for r in rows:
            ans = r.get("answer") or ""
            r["correct"] = score(ans, qs[r["id"]]["expect"])
            r["correct_v0"] = score(ans, EXPECT_V0.get(r["id"], qs[r["id"]]["expect"]))
            r["cites_time"] = bool(__import__("re").search(r"\b\d{1,2}:\d{2}\b", ans))

    summ = {}
    for r in rows:
        for grp in ("all", "q01_10" if int(r["id"][5:]) <= 10 else "q11_plus"):
            s = summ.setdefault(f"{r['config']}/{r['mode']}/{grp}", {"n": 0, "gold@1": 0, "gold_in_src": 0,
                                                                      "correct": 0, "correct_v0": 0, "cites_time": 0})
            s["n"] += 1
            s["gold@1"] += r["gold_rank"] == 1
            s["gold_in_src"] += r["gold_rank"] is not None
            s["correct"] += bool(r.get("correct"))
            s["correct_v0"] += bool(r.get("correct_v0"))
            s["cites_time"] += bool(r.get("cites_time"))
    for k in sorted(summ):
        s = summ[k]
        print(f"{k:32s} n={s['n']:2d} gold@1 {s['gold@1']:2d} in_src {s['gold_in_src']:2d} correct {s['correct']:2d} "
              f"(v0 expect {s['correct_v0']:2d}) cites_time {s['cites_time']:2d}")
    Path(a.out).write_text(json.dumps({"date": time.strftime("%Y-%m-%d %H:%M"), "model": config.ANSWER_MODEL,
                                       "summary": summ, "rows": rows}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
