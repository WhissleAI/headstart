"""One voice-search session: ear events in, tagger + speculator in the middle, events out.

Ear events (any ear: Arc on live audio, or a scripted word stream in tests):
  on_audio(pcm)                       16 kHz float32, appended to the current question's audio
  on_partial(words)                   the current utterance's words so far [{text,start,end}] (may revise earlier ones)
  on_final(words)                     the utterance is over (ear endpoint); words are final
Each partial (about every 100 ms) also counts as a tick for the speculator's stability timers.
"""
from __future__ import annotations

import asyncio
import collections
import re
import time

import numpy as np

from . import frame as framemod
from . import config, lanes
from .speculator import Speculator
from .tagger import Tagger

_END = re.compile(r"[.?!]+$")
CC = "discourse.assistant.command_complete"


def gate_open(f, cc_at: int | None, spans: list[dict]) -> bool:
    """Semantic-endpoint gate on the tagger's LATEST state (eval/gate_eval.py measures the rules on the eval set).

    Open when Arc's command_complete event fired at the newest word, or when the utterance is a question whose newest
    word ends a typed slot (entity / number span): "who founded the company Nvidia", "how tall is the Eiffel Tower".
    Not sticky: a command_complete at an earlier word no longer counts once more words arrive. (Arc dedups the event
    within 3 words, so it often cannot re-fire on the true last word; the slot-end alternative covers those.) The old
    gate, (any command_complete so far) OR act_question, was open on 90% of mid-question prefixes."""
    n = f.n
    if cc_at is not None and cc_at == n - 1:
        return True
    slot_end = any(s.get("family") in ("entity", "number") and s["type"] != "entity.numeral" and s["end"] == n
                   for s in spans)
    return bool(f.act_question and slot_end)


def meaning(f) -> tuple:
    """What an answer depends on: slots + content words (incl. a carried referent). Fillers ("um"), word order and
    intent flicker don't change it."""
    return (tuple(f.slots), tuple(sorted(set(f.content))))


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
        self.noisy = False                     # background too loud for the energy VAD (≈ <13 dB SNR)
        self.answered_level = None             # median voiced RMS of the last answered question
        self.last_answered_end = -1.0           # audio time of the last word of the last closed utterance
        self.last_answered_words: list[str] = []
        self.pending_finals: collections.deque = collections.deque()   # closed utterances waiting for their answer
        self.busy = False                       # an answer is being produced (finals are answered one at a time, FIFO)
        self.final_task: asyncio.Task | None = None   # the current question's soft (semantic-endpoint) answer
        self.soft: dict | None = None        # a semantic-endpoint answer awaiting the ear's confirmation
        self.carry: dict = {}                # {"person": [...], "thing": [...]} from the last question + answer
        self.context = ""                    # last question + answer text, for the answer model
        self.ans_tagger = Tagger(domain="talk")
        self.carry_pool: dict = {}           # its documents + vectors, seeded into the next speculator
        self.ledger = lanes.Ledger()         # Tavily credits + response cache, across this session's questions
        self.closed = False
        self._audio_end = 0.0
        self._new_question()

    def set_t0(self, t0: float):
        """Align the session clock with audio sample 0 (wall time)."""
        self.t0 = t0
        self.spec.t0 = t0

    def _new_question(self):
        """Fresh listening state (tagger, words, speculator) for the next utterance. Doesn't touch busy / the queue:
        an earlier question may still be answering while this one is heard."""
        self.tagger.reset()
        self.words: list[str] = []                   # what the tagger has
        self.times: list[tuple] = []
        self.spans: list[dict] = []
        self.current: dict | None = None
        self.cc_at: int | None = None                # word index of the newest command_complete event
        self.prev_intent = None
        self.frame = None
        self.audio: list[np.ndarray] = []
        self.q_audio_start: float | None = None      # audio time (s) of the first word of this question
        self.q_levels: list[float] = []              # voiced RMS while this question's words were arriving
        self.spec = Speculator(self.emit, self.mode, self.t0, ledger=self.ledger)
        if self.carry_pool:
            self.spec.seed(self.carry_pool)
        self.spec.context = self.context
        self.soft = None
        self.final_task = None
        self.words_changed_at = time.perf_counter()

    @property
    def events(self) -> set:
        """Tagger events that hold for the CURRENT words (command_complete only if it fired at the newest word)."""
        return {CC} if (self.cc_at is not None and self.cc_at == len(self.words) - 1) else set()

    # ------------------------------------------------------------------ ear events
    def on_audio(self, pcm: np.ndarray, at_s: float):
        """pcm ends at audio time at_s."""
        rms = float(np.sqrt(np.mean(pcm * pcm))) if len(pcm) else 0.0
        self.noise.append(rms)
        self.noise = self.noise[-60:]
        # noise floor from the quietest chunks, clamped so a stretch of continuous speech can't become the "floor"
        raw_floor = float(np.percentile(self.noise, 10)) if len(self.noise) >= 5 else 0.0
        floor = min(raw_floor, 0.01)
        self.noisy = raw_floor > 0.01        # steady background above the clamp: energy silence is unreliable
        if rms > max(0.006, 2.5 * floor):
            self.silence_ms = 0.0
            if getattr(self, "words", None):
                self.q_levels.append(rms)
                self.q_levels = self.q_levels[-80:]
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

    def _fresh(self, ws: list[dict]) -> list[dict]:
        """Drop words of utterances already closed (a late or repeated partial must not reopen an answered question)."""
        if self.last_answered_end < 0:
            return ws
        return [w for w in ws if w.get("start") is None or w["start"] >= self.last_answered_end - 1e-3]

    def _sync_words(self, ws: list[dict]):
        new = [clean(w["text"]) for w in ws]
        new = [w for w in new if w]
        k = 0
        while k < min(len(new), len(self.words)) and new[k] == self.words[k]:
            k += 1
        changed = False
        revised = k < len(self.words)
        if revised:                                  # revision of words[k:]
            self.tagger.push(new[k:], from_=k)
            changed = True
        elif len(new) > len(self.words):
            self.tagger.push(new[k:])
            changed = True
        self.words = new
        self.times = [(w.get("start"), w.get("end")) for w in ws if clean(w["text"])]
        if changed:
            self.words_changed_at = time.perf_counter()
            if self.cc_at is not None and self.cc_at >= k:
                self.cc_at = None                    # the word it fired on was revised away
            r = self.tagger.poll()
            self._apply(r, revised_from=k if revised else None)
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
            if e["type"] == CC:
                w = e.get("word")
                self.cc_at = w if w is not None else len(self.words) - 1

    def _build(self, carry=None):
        return framemod.build(self.words, self.spans, self.current, self.events, self.prev_intent,
                              local_ok=bool(len(lanes.index())), carry=self.carry if carry is None else carry)

    def on_partial(self, ws: list[dict], now: float | None = None):
        """Never raises: a bad partial is reported as an error event and the ear keeps going."""
        if self.closed:
            return
        try:
            self._on_partial(ws, now)
        except Exception as e:
            self.emit({"t": "error", "where": "on_partial", "error": f"{type(e).__name__}: {e}"[:300]})

    def _on_partial(self, ws: list[dict], now: float | None = None):
        # words of the utterance being answered are done; anything after it is the NEXT question, heard (and
        # speculated on) even while the previous answer is still streaming
        ws = self._fresh(ws)
        if not ws:
            return
        changed, k = self._sync_words(ws)
        f = self._build()
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
        wall = time.perf_counter()
        stable_ms = (wall - self.words_changed_at) * 1000
        silence_ms = self.silence_ms
        if self.noisy and ws:
            # in steady noise the energy VAD never hears silence: use the gap since the ASR's last word ended
            # (needs a longer threshold: a word being spoken isn't emitted yet)
            gap = (getattr(self, "_audio_end", 0.0) - (ws[-1].get("end") or 0.0)) * 1000
            if gap >= config.SEM_WORDGAP_MS:
                silence_ms = max(silence_ms, gap - (config.SEM_WORDGAP_MS - config.SEM_SILENCE_MS))
        speech_stop = wall - silence_ms / 1000
        # chatter guard: an utterance much quieter than the last question we answered (a neighbour, the room) gets no
        # early answer; the ear's own endpoint still answers it if it really is a question
        quiet = self._quiet()
        # the words are trustworthy if they held through one ASR refresh, or if they last changed in a refresh made
        # after the speaker went quiet (that refresh already saw the whole question)
        settled = stable_ms >= config.SEM_STABLE_MS or self.words_changed_at >= speech_stop + config.SEM_COVER_S
        if (self.semantic and not quiet and f.n >= 2 and silence_ms >= config.SEM_SILENCE_MS
                and settled and gate_open(f, self.cc_at, self.spans) and self._soft(ws, f, by="tagger")):
            return
        qa = self.question_audio()
        if qa is not None:
            self.spec.on_audio(qa, f.route, f.n)

    def _quiet(self) -> bool:
        return (self.answered_level is not None and len(self.q_levels) >= 3
                and float(np.median(self.q_levels)) < self.answered_level * config.CHATTER_RATIO)

    def _soft(self, ws: list[dict], f, by: str) -> bool:
        """Semantic endpoint: the request looks complete and the speaker paused. Answer now (soft), but keep listening
        and keep the speculator warm: the ear's final confirms or extends it. by = what fired it (tagger: our gate +
        silence; eager: the ear's own early end-of-turn, Deepgram Flux EagerEndOfTurn)."""
        sig = meaning(f)
        if self.busy or not f.has_content() or (self.soft is not None and self.soft["sig"] == sig):
            return False
        if self.final_task is not None and not self.final_task.done():
            self.final_task.cancel()               # superseded by this longer question: one soft answer at a time
        prev = self.soft
        self.soft = {"words": [framemod.norm(w) for w in self.words], "sig": sig, "metric": None,
                     "end": ws[-1].get("end") or 0.0, "failed": False, "by": by}
        self.final_task = asyncio.create_task(self._soft_answer(self.spec, self.soft, f, extended=prev is not None))
        return True

    def on_eager(self, ws: list[dict], now: float | None = None):
        """The ear's own early end-of-turn (Flux EagerEndOfTurn: Flux's turn model is moderately confident the
        speaker is done; the EndOfTurn that follows carries the same words unless a TurnResumed comes first).
        config.FLUX_SOFT: eager = soft answer now; eager+gate = only if our tagger gate agrees; off = ignore."""
        if self.closed or not self.semantic or config.FLUX_SOFT == "off":
            return
        try:
            ws = self._fresh(ws)
            if not ws:
                return
            self._on_partial(ws, now)                 # same words the partial just carried: normally a no-op sync
            f = self.frame or self._build()
            if self.soft is not None and self.soft["sig"] == meaning(f):
                return                                 # the tagger gate already answered exactly this
            if f.n < 1 or self._quiet() or (config.FLUX_SOFT == "eager+gate" and not gate_open(f, self.cc_at, self.spans)):
                self.emit({"t": "eager_skip", "ms": self.spec.ms(), "words": list(self.words)})
                return
            self._soft(ws, f, by="eager")
        except Exception as e:
            self.emit({"t": "error", "where": "on_eager", "error": f"{type(e).__name__}: {e}"[:300]})

    async def _soft_answer(self, spec: Speculator, soft: dict, f, extended: bool):
        last_end = soft["end"]
        if extended:
            self.emit({"t": "extended", "ms": spec.ms(), "words": soft["words"]})
        self.emit({"t": "endpoint", "ms": spec.ms(), "speech_end_ms": round(last_end * 1000),
                   "frame": f.to_json(), "by": "semantic", "trigger": soft.get("by"), "soft": True})
        try:
            m = await spec.finalize(f, self.t0 + last_end, close=False, by="semantic")
        except Exception as e:
            # a failed soft answer is "no soft answer": the ear's final answers the question normally
            soft["failed"] = True
            self.emit({"t": "answer_error", "error": f"{type(e).__name__}: {e}"[:300], "soft": True})
            return None
        if m.get("superseded"):
            return None
        m["endpoint_by"] = "semantic"
        m["endpoint_trigger"] = soft.get("by")
        soft["metric"] = m
        return m

    # ------------------------------------------------------------------ endpoint
    def _close_question(self, ws: list[dict]) -> dict:
        """The ear closed an utterance: freeze what the answer needs and start listening for the next question at once
        (its partials may arrive while this one is still being answered)."""
        job = {"ws": ws, "spec": self.spec, "soft": self.soft, "final_task": self.final_task, "error": None,
               "levels": list(self.q_levels)}
        try:
            self._sync_words(ws)
            self.tagger.endpoint()
            self._apply(self.tagger.poll())
            f = self._build()
            job.update(words=list(self.words), spans=list(self.spans), current=self.current, events=self.events,
                       prev_intent=self.prev_intent)
            self.emit({"t": "words", "words": [{"w": w, "t0": t[0], "t1": t[1]} for w, t in zip(self.words, self.times)],
                       "from": 0, "final": True, "ms": self.spec.ms()})
            self.emit({"t": "frame", **f.to_json(), "final": True, "spans": [
                {"type": s["type"], "start": s["start"], "end": s["end"], "text": s["text"]} for s in self.spans],
                "ms": self.spec.ms()})
        except Exception as e:
            job["error"] = e
        self.last_answered_end = max(self.last_answered_end, ws[-1].get("end") or 0.0)
        self.last_answered_words = [framemod.norm(w) for w in self.words]
        self._new_question()
        return job

    async def on_final(self, ws: list[dict], early: bool = False) -> dict | None:
        """Answer a closed utterance. Finals are answered one at a time in arrival order; a final that arrives while
        another is answering is queued (returns None) and answered right after it by the call already running."""
        if self.closed:
            return None
        ws = self._fresh(ws or [])
        if not ws:
            return None
        job = self._close_question(ws)
        if self.busy:
            self.pending_finals.append(job)
            return None
        self.busy = True
        try:
            m = await self._answer(job)
            while self.pending_finals and not self.closed:
                await self._answer(self.pending_finals.popleft())
        finally:
            self.busy = False
        return m

    async def _answer(self, job: dict) -> dict:
        """Answer one closed question. Never leaves the session wedged: any failure becomes an answer_error event and a
        failure metric, and the question's speculator is always closed."""
        spec: Speculator = job["spec"]
        soft, final_task = job["soft"], job["final_task"]
        ws = job["ws"]
        last_end = ws[-1].get("end") or 0.0
        m = None
        f = None
        try:
            if job["error"] is not None:
                raise job["error"]
            # build at answer time: an earlier question that finished meanwhile may have updated the carried entity
            f = framemod.build(job["words"], job["spans"], job["current"], job["events"], job["prev_intent"],
                               local_ok=bool(len(lanes.index())), carry=self.carry)
            spec.context = self.context
            if soft is not None and soft["sig"] == meaning(f):
                # the ear confirms what we already answered at the semantic endpoint
                if final_task is not None and not final_task.done():
                    await asyncio.wait({final_task})      # (a cancelled soft task must not look like OUR cancel)
                m = soft["metric"]                        # None if the soft answer failed or was superseded
            if m is None:
                if soft is not None:
                    # the speaker kept going after the semantic endpoint: answer the longer question from the same,
                    # still-warm speculator (its searches/drafts kept running on the later words)
                    self.emit({"t": "extended", "ms": spec.ms(), "words": [framemod.norm(w) for w in job["words"]]})
                    if final_task is not None and not final_task.done():
                        final_task.cancel()
                self.emit({"t": "endpoint", "ms": spec.ms(), "speech_end_ms": round(last_end * 1000),
                           "frame": f.to_json(), "by": "ear"})
                m = await spec.finalize(f, self.t0 + last_end)
                m["endpoint_by"] = "ear"
                m["superseded_soft"] = soft is not None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:300]
            self.emit({"t": "answer_error", "error": err})
            m = {"mode": self.mode, "question": " ".join(job.get("words") or [w["text"] for w in ws]),
                 "endpoint_by": "ear", "path": "failed", "error": err, "failed": True}
            self.emit({"t": "metric", **m})
        finally:
            if final_task is not None and not final_task.done():
                final_task.cancel()
            spec.close()
        if not m.get("failed") and f is not None:
            self._remember(f, m)
            if job.get("levels"):
                self.answered_level = float(np.median(job["levels"]))
            self.carry_pool = spec.export_pool()
            # the question already being heard started with the older pool / context: bring it up to date
            for s in [self.spec] + [j["spec"] for j in self.pending_finals]:
                s.seed(self.carry_pool)
                s.context = self.context
        self.history.append(m)
        return m

    def _remember(self, f, m):
        """Conversation memory: entities of the question and of the answer, typed person vs thing."""
        person, thing = [], []
        for t, txt in f.slots:
            if t in framemod.NUMBER_TYPES:
                continue
            (person if t == "person" else thing).append((t, txt))
        try:
            if self.ans_tagger.h is None:
                raise RuntimeError("session closed")     # never call into a destroyed native handle
            r = self.ans_tagger.tag_text(re.sub(r"\[\d+\]", "", m.get("answer") or ""))
            for sp in r.get("spans", []):
                if sp.get("family") != "entity" or not sp["type"].startswith("entity.mention"):
                    continue
                t = sp["type"].split(".")[-1]
                item = (t, framemod.norm(sp["text"]))
                if t == "person":
                    person.append(item)        # answer's people come after the question's
                elif not thing:
                    thing.append(item)
        except Exception:
            pass
        q_ents = [txt for t, txt in f.slots if t not in framemod.NUMBER_TYPES]
        if person or thing:
            self.carry = {"person": person, "thing": thing}
        # for elliptical follow-ups ("and when?", "what about Nvidia?"): the previous question's meaning
        self.carry = {**(self.carry or {}), "prev": {"content": list(f.content), "entities": q_ents,
                                                     "intent": f.intent, "question": f.text}}
        self.context = f"Q: {f.text}\nA: {(m.get('answer') or '')[:300]}"

    def export_memory(self) -> dict:
        """Conversation memory to carry into the next run on the same connection (push-to-talk turns)."""
        return {"carry": self.carry, "context": self.context, "carry_pool": self.carry_pool, "at": time.time()}

    def import_memory(self, mem: dict | None):
        if not mem or time.time() - mem.get("at", 0) > config.MEMORY_TTL_S:
            return
        self.carry, self.context, self.carry_pool = mem["carry"], mem["context"], mem["carry_pool"]
        if self.carry_pool:
            self.spec.seed(self.carry_pool)
        self.spec.context = self.context

    def close(self):
        if self.closed:
            return
        self.closed = True
        if (self.soft is not None and self.soft.get("metric") is not None and not self.busy
                and not self.history):
            # the ear never confirmed (stream ended): keep it only if it's the only answer — an unconfirmed soft
            # answer after a confirmed one is usually trailing chatter
            self.history.append(self.soft["metric"])
            self.soft = None
        if self.final_task is not None and not self.final_task.done():
            self.final_task.cancel()                      # an unconfirmed soft answer must not outlive the session
        self.spec.close()
        for j in self.pending_finals:
            j["spec"].close()
        self.pending_finals.clear()
        self.tagger.close()
        self.ans_tagger.close()
