"""Record ear traces: run every eval wav ONCE through the real Arc ear (real time) and save the exact sequence of
callbacks a Session would receive, so the policy simulator can replay them in virtual time.

  .venv/bin/python -m sim.record_traces                   # all questions -> sim/data/ear_traces/<id>.json
  .venv/bin/python -m sim.record_traces --ids q01,q02

Pacing is identical to eval/run_ab.py: 100 ms chunks fed when their last sample would have been spoken, then 1.2 s
of zeros, then flush. The recorder is the only sink (no Session), so Arc's timing is not perturbed by speculation.

Trace format (times are wall seconds since the first audio sample, measured with time.perf_counter):
  {"id", "text", "kind", "file", "dur_s", "vad_end_s", "chunk": 1600,
   "events": [
     {"k": "audio", "w": wall_s, "at": audio_s, "n": samples, "rms": float},     # Session.on_audio(pcm, at)
     {"k": "partial", "w": wall_s, "words": [{text,start,end}]},                   # Session.on_partial(words, now)
     {"k": "final", "w": wall_s, "words": [...], "flush": bool}]}                  # Session.on_final(words)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_ab import AUDIO, CHUNK, TRAIL_CHUNKS, load_questions, load_wav, speech_end_s  # noqa: E402

OUT = ROOT / "sim" / "data" / "ear_traces"


class Recorder:
    def __init__(self, t0: float):
        self.t0 = t0
        self.events: list[dict] = []

    def w(self):
        return round(time.perf_counter() - self.t0, 5)

    def on_audio(self, pcm, at_s):
        rms = float(np.sqrt(np.mean(pcm * pcm))) if len(pcm) else 0.0
        self.events.append({"k": "audio", "w": self.w(), "at": round(at_s, 5), "n": int(len(pcm)), "rms": rms})

    def on_partial(self, ws, now=None):
        self.events.append({"k": "partial", "w": self.w(), "words": ws})

    async def on_final(self, ws, early=False):
        self.events.append({"k": "final", "w": self.w(), "words": ws, "flush": getattr(self, "flushing", False)})


async def record_one(q: dict) -> dict:
    from server.ear import ArcEar
    wav = load_wav(AUDIO / q["file"])
    t0 = time.perf_counter()
    rec = Recorder(t0)
    ear = ArcEar([rec])
    rec.t0 = t0 = time.perf_counter()
    n = len(wav)
    late = []
    for i in range(0, n, CHUNK):
        target = t0 + min(i + CHUNK, n) / 16000
        while time.perf_counter() < target:
            await asyncio.sleep(0.003)
        late.append(time.perf_counter() - target)
        await ear.feed(wav[i:i + CHUNK])
    t_end = t0 + n / 16000
    for k in range(TRAIL_CHUNKS):
        target = t_end + (k + 1) * CHUNK / 16000
        while time.perf_counter() < target:
            await asyncio.sleep(0.003)
        await ear.feed(np.zeros(CHUNK, np.float32))
    rec.flushing = True
    await ear.flush()
    ear.close()
    return {"id": q["id"], "text": q["text"], "kind": q.get("kind"), "file": q["file"], "voice": q.get("voice"),
            "expect": q.get("expect"), "dur_s": round(n / 16000, 4), "vad_end_s": round(speech_end_s(wav), 4),
            "chunk": CHUNK, "feed_late_p95_ms": round(float(np.percentile(late, 95)) * 1000, 2),
            "events": rec.events}


async def main_async(a):
    from server import ear
    qs = load_questions()
    if a.ids:
        want = set(a.ids.split(","))
        qs = [q for q in qs if q["id"] in want]
    qs = [q for q in qs if (AUDIO / q["file"]).exists()]
    OUT.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_running_loop()
    t = time.perf_counter()
    await loop.run_in_executor(None, ear.warmup)
    await record_one(qs[0])                      # throwaway warm replay (first-run effects)
    print(f"warm {time.perf_counter() - t:.1f}s; {len(qs)} questions", flush=True)
    for i, q in enumerate(qs):
        tr = await record_one(q)
        (OUT / f"{q['id']}.json").write_text(json.dumps(tr))
        nf = [e for e in tr["events"] if e["k"] == "final"]
        print(f"[{i + 1}/{len(qs)}] {q['id']} partials={sum(e['k'] == 'partial' for e in tr['events'])} "
              f"finals={len(nf)} late95={tr['feed_late_p95_ms']}ms heard={' | '.join(' '.join(w['text'] for w in e['words']) for e in nf)!r}",
              flush=True)
        await asyncio.sleep(0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    a = ap.parse_args()
    os.chdir(ROOT)
    asyncio.run(main_async(a))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
