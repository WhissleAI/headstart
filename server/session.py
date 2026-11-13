"""One voice-search session: ear events in, tagger + speculator in the middle, events out.

Ear events (any ear: Arc on live audio, or a scripted word stream in tests):
  on_audio(pcm)                       16 kHz float32, appended to the current question's audio
  on_partial(words)                   the current utterance's words so far [{text,start,end}] (may revise earlier ones)
  on_final(words)                     the utterance is over (ear endpoint); words are final
Each partial (about every 100 ms) also counts as a tick for the speculator's stability timers.
"""
from __future__ import annotations

import asyncio
import re
import time

import numpy as np

from . import frame as framemod
from . import config, lanes
from .speculator import Speculator
from .tagger import Tagger

_END = re.compile(r"[.?!]+$")


def clean(w: str) -> str:
    """Strip sentence-final punctuation so a partial "Paris." doesn't close the tagger's utterance early."""
    return _END.sub("", w.strip()) or w.strip()


class Session:
    def __init__(self, emit, mode: str = "headstart", domain: str = "assistant", t0: float | None = None):
        self.emit = emit
        self.mode = mode
        self.tagger = Tagger(domain=domain)
        self.t0 = t0 or time.perf_counter()        # wall time of audio sample 0
        self.history: list[dict] = []
        self.semantic = mode == "headstart" and config.SEMANTIC_ENDPOINT
        self.silence_ms = 0.0
        self.noise = []
        self.last_answered_end = -1.0           # audio time of the last word of the last answered question
        self.last_answered_words: list[str] = []
        self.final_task: asyncio.Task | None = None
        self._new_question()

    def _new_question(self):
        self.tagger.reset()
        self.words: list[str] = []                   # what the tagger has
        self.times: list[tuple] = []
        self.spans: list[dict] = []
        self.current: dict | None = None
        self.events: set[str] = set()
        self.prev_intent = None
        self.frame = None
        self.audio: list[np.ndarray] = []
        self.q_audio_start: float | None = None      # audio time (s) of the first word of this question
        self.spec = Speculator(self.emit, self.mode, self.t0)
        self.busy = False
        self.words_changed_at = time.perf_counter()

    # ------------------------------------------------------------------ ear events
    def on_audio(self, pcm: np.ndarray, at_s: float):
        """pcm ends at audio time at_s."""
        rms = float(np.sqrt(np.mean(pcm * pcm))) if len(pcm) else 0.0
        self.noise.append(rms)
        self.noise = self.noise[-60:]
        # noise floor from the quietest chunks, clamped so a stretch of continuous speech can't become the "floor"
        floor = min(float(np.percentile(self.noise, 10)), 0.01) if len(self.noise) >= 5 else 0.0
        if rms > max(0.006, 2.5 * floor):
            self.silence_ms = 0.0
        else:
            self.silence_ms += 1000.0 * len(pcm) / 16000
        self.audio.append(pcm)
        # keep ~12 s
        tot = sum(len(a) for a in self.audio)
        while tot > 16000 * 12 and len(self.audio) > 1:
            tot -= len(self.audio[0])
            self.audio.pop(0)
        self._audio_end = at_s

    def question_audio(self) -> np.ndarray | None:
        if self.q_audio_start is None or not self.audio:
            return None
        buf = np.concatenate(self.audio)
        start_s = self._audio_end - len(buf) / 16000
        a = max(0, int((self.q_audio_start - 0.15 - start_s) * 16000))
        return buf[a:]

    def _sync_words(self, ws: list[dict]):
        new = [clean(w["text"]) for w in ws]
        new = [w for w in new if w]
        k = 0
        while k < min(len(new), len(self.words)) and new[k] == self.words[k]:
            k += 1
        changed = False
        if k < len(self.words):                      # revision of words[k:]
            self.tagger.push(new[k:], from_=k)
            changed = True
        elif len(new) > len(self.words):
            self.tagger.push(new[k:])
            changed = True
        self.words = new
        self.times = [(w.get("start"), w.get("end")) for w in ws if clean(w["text"])]
        if changed:
            self.words_changed_at = time.perf_counter()
            r = self.tagger.poll()
            self._apply(r, revised_from=k if k < len(self.words) else None)
        if ws and self.q_audio_start is None:
            self.q_audio_start = ws[0].get("start") or 0.0
        return changed, k

    def _apply(self, r: dict, revised_from=None):
        a0 = r.get("a0", 0)
        self.spans = [s for s in self.spans if s["end"] <= a0] + r.get("spans", [])
        if r.get("current") is not None:
            self.current = r["current"]
        for u in r.get("utterances") or []:
            self.current = u
        for e in r.get("events") or []:
            self.events.add(e["type"])

    def on_partial(self, ws: list[dict], now: float | None = None):
        if self.busy:
            return
        changed, k = self._sync_words(ws)
        f = framemod.build(self.words, self.spans, self.current, self.events, self.prev_intent, local_ok=bool(len(lanes.index())))
        if f.intent.startswith("discourse.intent."):
            self.prev_intent = (f.intent, f.intent_p)
        if changed:
            self.emit({"t": "words", "words": [{"w": w, "t0": t[0], "t1": t[1]} for w, t in zip(self.words, self.times)],
                       "from": k, "ms": self.spec.ms()})
            self.emit({"t": "frame", **f.to_json(), "spans": [
                {"type": s["type"], "start": s["start"], "end": s["end"], "text": s["text"]} for s in self.spans],
                "ms": self.spec.ms()})
        self.frame = f
        self.spec.on_frame(f, now)
        stable_ms = (time.perf_counter() - self.words_changed_at) * 1000
        if (self.semantic and not self.busy and ws and f.n >= 2 and self.silence_ms >= config.SEM_SILENCE_MS
                and stable_ms >= config.SEM_STABLE_MS and (f.command_complete or f.act_question) and f.has_content()):
            # semantic endpoint: the tagger says the request is complete and the speaker paused briefly
            self.final_task = asyncio.create_task(self.on_final(ws, early=True))
            return
        qa = self.question_audio()
        if qa is not None:
            self.spec.on_audio(qa, f.route)

    async def on_final(self, ws: list[dict], early: bool = False) -> dict | None:
        if self.busy:
            return None
        if not early and ws and (ws[0].get("start") or 0.0) <= self.last_answered_end + 0.05:
            # the ear's own endpoint for an utterance we already answered at the semantic endpoint: ignore it if it
            # says the same thing; if the speaker kept going ("... in Paris | tomorrow"), answer the longer question
            fin = [framemod.norm(clean(w["text"])) for w in ws]
            if fin[: len(self.last_answered_words)] == self.last_answered_words and \
                    len(fin) <= len(self.last_answered_words):
                return None
            self.emit({"t": "extended", "ms": self.spec.ms(), "words": fin})
        self.busy = True
        if not ws:
            self._new_question()
            return None
        self._sync_words(ws)
        self.tagger.endpoint()
        self._apply(self.tagger.poll())
        f = framemod.build(self.words, self.spans, self.current, self.events, self.prev_intent, local_ok=bool(len(lanes.index())))
        last_end = ws[-1].get("end") or 0.0
        self.last_answered_end = last_end
        self.last_answered_words = [framemod.norm(w) for w in self.words]
        t_speech_end = self.t0 + last_end
        self.emit({"t": "endpoint", "ms": self.spec.ms(), "speech_end_ms": round(last_end * 1000),
                   "frame": f.to_json(), "by": "semantic" if early else "ear"})
        try:
            m = await self.spec.finalize(f, t_speech_end)
            m["endpoint_by"] = "semantic" if early else "ear"
        finally:
            self.spec.close()
        self.history.append(m)
        self._new_question()
        return m

    def close(self):
        self.spec.close()
        self.tagger.close()
