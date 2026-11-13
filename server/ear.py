"""Ears: 16 kHz audio in (100 ms chunks), live words out, fanned out to one or more sessions (the A/B race feeds two
brains from one ear, so both receive identical words).

Sink contract (server/session.py): on_audio(pcm, at_s), on_partial(words, now), await on_final(words), where words are
[{"text", "start", "end"}] in seconds on the audio clock (0 = first sample fed to the ear). on_partial carries the open
utterance's words so far (later words may revise earlier ones) and is also called on every audio chunk while an
utterance is open: each call is a tick for the speculator's stability timers.

  make_ear(sinks, domain, on_event)   picks the ear from env EAR=arc|deepgram (default arc)
  ArcEar        Arc v1 Mini's built-in transcript, local (torch, CPU thread)
  DeepgramEar   Deepgram's live websocket (nova-3, interim results, short endpointing); falls back to Arc when there is
                no key, the connection fails, or it drops mid-run (the open utterance's audio is re-fed to Arc)
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode

import numpy as np

from . import arcmeta, config

_arc = None
_arc_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arc")


def arc():
    global _arc
    with _arc_lock:
        if _arc is None:
            sys.path.insert(0, str(config.ARC_DIR))
            from arc_audio import Arc
            _arc = Arc.from_pretrained(str(config.ARC_DIR), device="cpu", threads=config.ARC_THREADS)
        return _arc


def warmup():
    """Arc is loaded whatever the ear: it is the Deepgram ear's fallback."""
    t = time.perf_counter()
    s = arc().stream(domain="assistant", transcript="builtin", partials=True)
    z = np.zeros(1600, np.float32)
    rng = np.random.default_rng(0)
    for i in range(20):
        s.feed(z if i % 2 else rng.standard_normal(1600).astype(np.float32) * 0.001)
    s.flush()
    if config.EAR == "flux":
        arcmeta.warmup()                     # Arc Transcriber (ARC_META=transcriber) loads ~3 GB: do it before the demo
    return time.perf_counter() - t


# ---------------------------------------------------------------------------------------------------- factory
DG_EARS = ("deepgram", "flux")


def selected() -> str:
    """The ear new runs will use (before any per-run fallback)."""
    return config.EAR if config.EAR in DG_EARS and config.DEEPGRAM_API_KEY else "arc"


def describe() -> dict:
    d = {"ear": selected(), "ear_requested": config.EAR}
    if config.EAR in DG_EARS:
        d["deepgram_key"] = bool(config.DEEPGRAM_API_KEY)
        d["deepgram_model"] = config.FLUX_MODEL if config.EAR == "flux" else config.DG_MODEL
    if config.EAR == "flux":
        d["flux"] = {"eager_eot_threshold": flux_eager(), "eot_threshold": config.FLUX_EOT,
                     "eot_timeout_ms": config.FLUX_EOT_TIMEOUT_MS, "soft": config.FLUX_SOFT}
        d["arc_meta"] = arcmeta.variant()
    return d


def make_ear(sinks: list, domain: str = "assistant", on_event=None):
    on_event = on_event or (lambda e: None)
    if config.EAR in DG_EARS:
        if config.DEEPGRAM_API_KEY:
            cls = FluxEar if config.EAR == "flux" else DeepgramEar
            return cls(sinks, domain=domain, on_event=on_event)
        on_event({"t": "ear", "kind": "warning", "ear": "arc",
                  "text": f"EAR={config.EAR} but DEEPGRAM_API_KEY is not set: using the Arc ear"})
    elif config.EAR not in ("arc", ""):
        on_event({"t": "ear", "kind": "warning", "ear": "arc", "text": f"unknown EAR={config.EAR!r}: using the Arc ear"})
    return ArcEar(sinks, domain=domain, on_event=on_event)


# ---------------------------------------------------------------------------------------------------- Arc
def _arc_events(out: dict, at: float, offset: float, on_event):
    for e in out.get("events") or []:
        if e.get("tag") == "discourse.assistant.command_complete":
            t = e.get("t")
            on_event({"t": "ear", "kind": "arc_command_complete", "t_audio": (t + offset) if t is not None else None,
                      "p": e.get("p"), "at": round(at, 2)})


class ArcEar:
    """Feed audio; call back into sinks. Sinks get: on_audio(pcm, at_s), on_partial(words), await on_final(words).

    offset_s: audio time of the first sample this ear sees (non-zero when it takes over from a failed Deepgram ear
    mid-run); Arc's word times are shifted by it so every ear speaks the same audio clock."""
    name = "arc"

    def __init__(self, sinks: list, domain: str = "assistant", on_event=None, offset_s: float = 0.0):
        self.sinks = sinks
        self.stream = arc().stream(domain=domain, transcript="builtin", partials=True)
        self.offset = float(offset_s)
        self.samples = int(round(self.offset * 16000))
        self.on_event = on_event or (lambda e: None)
        self.last_partial_key = None
        self.final_tasks: list[asyncio.Task] = []
        self.closed = False
        self.cur_start = None

    @property
    def at_s(self):
        return self.samples / 16000

    def _words(self, u: dict) -> list[dict]:
        o = self.offset
        return [{"text": w["text"], "start": None if w.get("start") is None else w["start"] + o,
                 "end": None if w.get("end") is None else w["end"] + o} for w in (u.get("words") or [])]

    async def feed(self, pcm: np.ndarray, notify: bool = True):
        """notify=False: the sinks already have this audio (a takeover ear catching up)."""
        if self.closed:
            return
        pcm = np.asarray(pcm, np.float32).reshape(-1)
        self.samples += len(pcm)
        at = self.at_s
        if notify:
            for s in self.sinks:
                s.on_audio(pcm, at)
        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(_pool, self.stream.feed, pcm)
        await self._handle(out)

    async def flush(self):
        if self.closed:
            return
        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(_pool, self.stream.flush)
        await self._handle(out, final=True)
        if self.final_tasks:
            await asyncio.gather(*self.final_tasks, return_exceptions=True)

    async def _handle(self, out: dict, final: bool = False):
        o = self.offset
        for u in out.get("transcript") or []:
            ws = self._words(u)
            self.on_event({"t": "ear", "kind": "final", "ear": "arc", "text": u.get("text", ""), "reason": u.get("reason"),
                           "start": None if u.get("start") is None else u["start"] + o,
                           "end": None if u.get("end") is None else u["end"] + o, "at": round(self.at_s, 2)})
            self.last_partial_key = None
            for s in self.sinks:
                self.final_tasks.append(asyncio.create_task(s.on_final(ws)))
        p = out.get("transcript_partial")
        now = time.perf_counter()
        if p and p.get("words") and not p.get("final"):
            ws = self._words(p)
            key = tuple(w["text"] for w in ws)
            if key != self.last_partial_key:
                self.on_event({"t": "ear", "kind": "partial", "ear": "arc", "text": p.get("text", ""),
                               "at": round(self.at_s, 2)})
            self.last_partial_key = key
            for s in self.sinks:
                s.on_partial(ws, now)       # every chunk is a tick, even with unchanged words
        _arc_events(out, self.at_s, o, self.on_event)

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------------------------------- Deepgram
def dg_url(base: str | None = None) -> str:
    """Deepgram live /v1/listen query (developers.deepgram.com/reference/speech-to-text-api/listen-streaming)."""
    q = {
        "model": config.DG_MODEL, "language": config.DG_LANGUAGE,
        "encoding": "linear16", "sample_rate": 16000, "channels": 1,
        "interim_results": "true", "endpointing": config.DG_ENDPOINTING_MS,
        "smart_format": "true", "punctuate": "true",
    }
    if config.DG_UTTERANCE_END_MS > 0:
        q["utterance_end_ms"] = max(1000, config.DG_UTTERANCE_END_MS)    # Deepgram's minimum is 1000
    return (base or config.DG_URL) + "?" + urlencode(q)


_TRAIL = re.compile(r"[,;:]+$")
KEEPALIVE = json.dumps({"type": "KeepAlive"})
FINALIZE = json.dumps({"type": "Finalize"})
CLOSE_STREAM = json.dumps({"type": "CloseStream"})


class DeepgramEar:
    """Same contract as ArcEar, words from Deepgram's live websocket.

    Deepgram "Results": an interim (is_final=false) carries the current segment's words so far; a final (is_final=true)
    locks a segment, and later interims cover only the words after it; speech_final=true marks the endpoint (a pause
    of DG_ENDPOINTING_MS). Utterance words = the finalized segments since the last endpoint + the current interim.
    UtteranceEnd (word-gap backstop) also closes the utterance. Times are seconds since the first audio sent, which is
    sample 0 of this ear (audio fed while connecting is queued, never dropped), so they share the sessions' clock.
    """
    name = "deepgram"

    def __init__(self, sinks: list, domain: str = "assistant", on_event=None, key: str | None = None,
                 url: str | None = None):
        self.sinks = sinks
        self.domain = domain
        self.on_event = on_event or (lambda e: None)
        self.key = key if key is not None else config.DEEPGRAM_API_KEY
        self.url = url or dg_url()
        self.samples = 0
        self.done: list[dict] = []           # finalized segments of the open utterance
        self.interim: list[dict] = []        # the current segment's interim words
        self.boundary_s = 0.0                # end of the last word handed to on_final
        self.last_partial_key = None
        self.final_tasks: list[asyncio.Task] = []
        self.closed = False
        self.failed: str | None = None       # reason, once we fell back to Arc
        self.fb: ArcEar | None = None
        self.fb_lock = asyncio.Lock()
        self.ring: list[tuple[int, np.ndarray]] = []   # (first sample, pcm) of audio after boundary_s, for fallback
        self.ws = None
        self.out: asyncio.Queue = asyncio.Queue()
        self.finalized = asyncio.Event()
        self.request_id = None
        self.n_results = 0
        self.rx: asyncio.Task | None = None
        self.arcx = self.adapter = None
        if config.DG_ARC_EVENTS and self.name == "deepgram":             # optional: Arc on the same audio + Deepgram's words, for its events only
            sys.path.insert(0, str(config.ARC_DIR))
            from arc_audio import WordEventAdapter
            self.arcx = arc().stream(domain=domain, transcript="external", name_source="word",
                                     interim_source="deepgram", partials=False)
            self.adapter = WordEventAdapter(self.arcx)
        self.tx = asyncio.create_task(self._run())

    @property
    def at_s(self):
        return self.samples / 16000

    @property
    def active(self) -> str:
        return "arc" if self.failed else self.name

    @property
    def model(self) -> str:
        return config.DG_MODEL

    def keepalive(self):
        return KEEPALIVE

    # ------------------------------------------------------------------ audio in
    async def feed(self, pcm: np.ndarray):
        if self.closed:
            return
        pcm = np.asarray(pcm, np.float32).reshape(-1)
        first = self.samples
        self.samples += len(pcm)
        at = self.at_s
        for s in self.sinks:
            s.on_audio(pcm, at)
        if self.failed is not None:
            async with self.fb_lock:             # waits while the fallback ear catches up on the open utterance
                if self.fb is not None:
                    await self.fb.feed(pcm, notify=False)
            return
        self.ring.append((first, pcm))
        self._trim_ring()
        self.out.put_nowait((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
        if self.arcx is not None:
            fut = asyncio.get_running_loop().run_in_executor(_pool, self.arcx.feed, pcm)
            fut.add_done_callback(lambda f, at=at: self._arcx_done(f, at))
        ws = self.done + self.interim
        if ws:
            now = time.perf_counter()
            for s in self.sinks:
                s.on_partial(list(ws), now)        # tick, like Arc's per-chunk partial

    def _arcx_done(self, f, at):
        if not f.cancelled() and f.exception() is None and not self.closed:
            _arc_events(f.result(), at, 0.0, self.on_event)

    def _trim_ring(self):
        cap = int(config.DG_FALLBACK_REPLAY_S * 16000)
        b = int(self.boundary_s * 16000)
        while self.ring and (self.ring[0][0] + len(self.ring[0][1]) <= b or self.samples - self.ring[0][0] > cap):
            self.ring.pop(0)

    # ------------------------------------------------------------------ connection
    async def _run(self):
        try:
            import websockets
            self.ws = await asyncio.wait_for(
                websockets.connect(self.url, additional_headers={"Authorization": f"Token {self.key}"},
                                   open_timeout=config.DG_CONNECT_TIMEOUT_S),
                timeout=config.DG_CONNECT_TIMEOUT_S + 0.5)
        except Exception as e:  # noqa: BLE001 (bad key -> HTTP 401 on the upgrade, DNS, timeout, ...)
            await self._fallback(f"connect failed: {type(e).__name__}: {e}"[:240])
            return
        if self.closed:
            await self._close_ws()
            return
        self.on_event({"t": "ear", "kind": "connected", "ear": self.name, "model": self.model,
                       "at": round(self.at_s, 2)})
        self.rx = asyncio.create_task(self._recv())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(self.out.get(), timeout=config.DG_KEEPALIVE_S)
                except asyncio.TimeoutError:
                    item = self.keepalive()         # no audio for a while: Deepgram drops idle sockets after 10 s
                if item is None:
                    break
                await self.ws.send(item)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            if not self.closed:
                await self._fallback(f"send failed: {type(e).__name__}: {e}"[:240])
            return
        await self._close_ws()

    async def _recv(self):
        reason = None
        try:
            async for m in self.ws:
                if isinstance(m, str):
                    self._on_message(m)
            reason = f"closed by server (code {self.ws.close_code} {self.ws.close_reason or ''})".strip()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            reason = f"connection lost: {type(e).__name__}: {e}"[:240]
        if not self.closed:
            await self._fallback(reason)

    async def _close_ws(self):
        if self.ws is not None:
            try:
                await asyncio.wait_for(self.ws.close(), timeout=2.0)
            except Exception:  # noqa: BLE001
                pass

    async def _fallback(self, reason: str):
        """Switch this run to the Arc ear. The open utterance's audio (since the last final) is re-fed to Arc so the
        question in progress isn't lost; Arc's next partial revises the sessions' words from wherever they differ."""
        if self.failed is not None or self.closed:
            return
        self.failed = reason
        self.on_event({"t": "ear", "kind": "fallback", "from": self.name, "to": "arc", "reason": reason,
                       "at": round(self.at_s, 2)})
        self.done, self.interim, self.last_partial_key = [], [], None
        async with self.fb_lock:
            ring, self.ring = self.ring, []
            offset = ring[0][0] / 16000 if ring else self.at_s
            loop = asyncio.get_running_loop()
            try:
                fb = await loop.run_in_executor(
                    None, lambda: ArcEar(self.sinks, domain=self.domain, on_event=self.on_event, offset_s=offset))
                for _, pcm in ring:
                    await fb.feed(pcm, notify=False)
                self.fb = fb
            except Exception as e:  # noqa: BLE001
                self.on_event({"t": "ear", "kind": "error", "ear": "arc", "text": f"fallback failed: {e}"[:240]})
        cur = asyncio.current_task()
        for t in (self.tx, self.rx):
            if t is not None and t is not cur and not t.done():
                t.cancel()
        if self.ws is not None:
            asyncio.create_task(self._close_ws())

    # ------------------------------------------------------------------ Deepgram messages
    def _words(self, d: dict) -> list[dict]:
        alts = (d.get("channel") or {}).get("alternatives") or [{}]
        field = config.DG_WORD_FIELD
        out = []
        for w in alts[0].get("words") or []:
            if "start" not in w or "end" not in w:
                continue
            text = (w.get(field) or w.get("word") or "").strip()
            if field != "word":
                text = _TRAIL.sub("", text)
            start, end = float(w["start"]), float(w["end"])
            if not text or end <= self.boundary_s + 1e-3:     # already handed to on_final
                continue
            out.append({"text": text, "start": start, "end": end, "conf": _num(w.get("confidence"))})
        return out

    def _on_message(self, m: str):
        if self.closed or self.failed is not None:
            return
        try:
            d = json.loads(m)
        except ValueError:
            return
        t = d.get("type")
        if t == "Results":
            self.n_results += 1
            if self.adapter is not None:
                _pool.submit(self.adapter.on_message, d)       # serialized with arcx.feed on the Arc thread
            ws = self._words(d)
            if d.get("is_final"):
                self.done += ws
                self.interim = []
                if d.get("from_finalize"):
                    self.finalized.set()
                if d.get("speech_final"):
                    self._final("speech_final")
                    return
            else:
                self.interim = ws
            self._partial()
        elif t == "UtteranceEnd":
            if not self.interim:             # an interim still open will be finalized (and endpointed) shortly
                self._final("utterance_end")
        elif t == "Metadata":
            self.request_id = d.get("request_id")
        elif t == "Error":
            self.on_event({"t": "ear", "kind": "error", "ear": self.name,
                           "text": f"{d.get('variant') or ''} {d.get('description') or d.get('message') or ''}".strip()})

    def _partial(self):
        ws = self.done + self.interim
        if not ws:
            return
        key = tuple(w["text"] for w in ws)
        if key == self.last_partial_key:
            return
        self.last_partial_key = key
        self.on_event({"t": "ear", "kind": "partial", "ear": self.name, "text": " ".join(key), "at": round(self.at_s, 2)})
        now = time.perf_counter()
        for s in self.sinks:
            s.on_partial(list(ws), now)

    def _final(self, reason: str):
        ws = self.done + self.interim
        self.done, self.interim, self.last_partial_key = [], [], None
        if not ws:
            return
        self.boundary_s = max(self.boundary_s, ws[-1]["end"])
        self._trim_ring()
        self.on_event({"t": "ear", "kind": "final", "ear": self.name, "text": " ".join(w["text"] for w in ws),
                       "reason": reason, "start": ws[0]["start"], "end": ws[-1]["end"], "at": round(self.at_s, 2)})
        for s in self.sinks:
            self.final_tasks.append(asyncio.create_task(s.on_final(list(ws))))

    # ------------------------------------------------------------------ end of run
    async def flush(self):
        """Close the open utterance: ask Deepgram to finalize what it has buffered, then hand everything to on_final."""
        if self.closed:
            return
        if self.failed is None:
            self.finalized.clear()
            self.out.put_nowait(FINALIZE)
            wait = config.DG_FLUSH_TIMEOUT_S if (self.done or self.interim) else min(0.5, config.DG_FLUSH_TIMEOUT_S)
            try:
                await asyncio.wait_for(self.finalized.wait(), timeout=wait)
            except asyncio.TimeoutError:
                pass
        if self.failed is None:
            self._final("flush")
        else:
            async with self.fb_lock:
                if self.fb is not None:
                    await self.fb.flush()
        if self.final_tasks:
            await asyncio.gather(*self.final_tasks, return_exceptions=True)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.fb is not None:
            self.fb.close()
        if self.failed is None and self.ws is not None and not self.tx.done():
            self.out.put_nowait(CLOSE_STREAM)       # sender sends it, then closes the socket
            self.out.put_nowait(None)
        elif not self.tx.done():
            self.tx.cancel()                        # still connecting: _run sees closed and closes
        if self.rx is not None:
            rx = self.rx
            asyncio.get_running_loop().call_later(2.5, lambda: rx.done() or rx.cancel())


# ---------------------------------------------------------------------------------------------------- Deepgram Flux
def flux_eager() -> float | None:
    v = (config.FLUX_EAGER_EOT or "").strip().lower()
    if v in ("", "off", "none", "0"):
        return None
    return min(float(v), config.FLUX_EOT)          # Flux rejects eager > eot


def flux_url(base: str | None = None) -> str:
    """Flux /v2/listen query (developers.deepgram.com/reference/speech-to-text/listen-flux): raw PCM16 at 16 kHz needs
    encoding + sample_rate; no language/interim/endpointing params (Flux's turn model replaces them)."""
    q = [("model", config.FLUX_MODEL), ("encoding", "linear16"), ("sample_rate", 16000),
         ("eot_threshold", config.FLUX_EOT), ("eot_timeout_ms", config.FLUX_EOT_TIMEOUT_MS)]
    e = flux_eager()
    if e is not None:
        q.append(("eager_eot_threshold", e))
    q += [("keyterm", t) for t in config.FLUX_KEYTERMS]
    return (base or config.FLUX_URL) + "?" + urlencode(q)


_EDGE = re.compile(r"^[^\w']+|[^\w']+$")
FORCE_END = json.dumps({"type": "ForceEndTurn"})


def flux_word(t: str) -> str:
    """Flux words are punctuated and cased ("Hey.", "Chicago", "tomorrow?"); our sessions speak Arc's lowercase,
    unpunctuated words (internal ' . - kept: "what's", "3.5", "u.s")."""
    return _EDGE.sub("", t.strip()).lower()


class FluxEar(DeepgramEar):
    """Same contract as DeepgramEar (and its Arc fallback), words from Deepgram Flux.

    Flux "TurnInfo" messages carry the WHOLE current turn every time (transcript + words with start/end in seconds of
    the stream, i.e. our audio clock): Update (~every 240 ms of audio), StartOfTurn, EagerEndOfTurn, TurnResumed,
    EndOfTurn. A turn = one of our utterances:
      Update / StartOfTurn / TurnResumed -> on_partial(turn words)
      EagerEndOfTurn                     -> on_partial + sinks' on_eager(turn words): the session's soft answer
      EndOfTurn                          -> on_final(turn words) (docs: identical to the preceding eager transcript
                                            unless a TurnResumed came between)
    flush() sends ForceEndTurn (-> EndOfTurn trigger=manual); close() sends CloseStream."""
    name = "flux"

    def __init__(self, sinks: list, domain: str = "assistant", on_event=None, key: str | None = None,
                 url: str | None = None):
        super().__init__(sinks, domain=domain, on_event=on_event, key=key, url=url or flux_url())
        self.meta = None
        self.meta_task = None
        if arcmeta.variant() != "off":
            # built off the event loop (a cold model load must never stall the audio); it joins at the audio time it
            # is ready (offset), which is sample 0 whenever the model was warmed up (server start, run_ab)
            self.meta_task = asyncio.create_task(self._start_meta(domain))
        self.turn_index = None
        self.n_turninfo = 0
        self.eager_at: float | None = None     # audio time of the last EagerEndOfTurn of the open turn

    @property
    def model(self) -> str:
        return config.FLUX_MODEL

    async def _start_meta(self, domain):
        try:
            if arcmeta.ready():
                m = arcmeta.ArcMeta(self._on_meta, domain=domain, offset_s=self.at_s)
            else:
                m = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: arcmeta.ArcMeta(self._on_meta, domain=domain))
                m.offset = self.at_s
            if self.closed:
                m.close()
            else:
                self.meta = m
        except Exception as e:  # noqa: BLE001 (metadata is optional: the words path never depends on it)
            self.on_event({"t": "ear", "kind": "warning", "ear": "flux", "text": f"arc meta off: {e}"[:240]})

    def _on_meta(self, m: dict):
        if self.closed:
            return
        for s in self.sinks:
            f = getattr(s, "on_meta", None)
            if f is not None:
                f(m)

    async def feed(self, pcm: np.ndarray):
        if self.meta is not None and self.failed is None and not self.closed:
            self.meta.feed(np.asarray(pcm, np.float32).reshape(-1))
        await super().feed(pcm)

    def keepalive(self):
        # Flux defines no KeepAlive message: keep the socket alive with 100 ms of silence (it is real audio time,
        # but only sent when the caller stopped feeding for DG_KEEPALIVE_S)
        return bytes(3200)

    def _words(self, d: dict) -> list[dict]:
        out = []
        for w in d.get("words") or []:
            if w.get("start") is None or w.get("end") is None:
                continue
            text = flux_word(w.get("word") or "")
            start, end = float(w["start"]), float(w["end"])
            if not text or end <= self.boundary_s + 1e-3:
                continue
            out.append({"text": text, "start": start, "end": end, "conf": _num(w.get("confidence"))})
        return out

    def _on_message(self, m: str):
        if self.closed or self.failed is not None:
            return
        try:
            d = json.loads(m)
        except ValueError:
            return
        t = d.get("type")
        if t == "TurnInfo":
            self.n_turninfo += 1
            self.n_results = self.n_turninfo
            ev = d.get("event")
            ti = d.get("turn_index")
            if self.turn_index is not None and ti != self.turn_index and self.interim:
                self._final("turn_change")         # a turn we never saw end (shouldn't happen): close it
            self.turn_index = ti
            self.interim = self._words(d)
            if self.meta is not None and d.get("words"):
                self.meta.words([{"text": w.get("word") or "", "start": w.get("start"), "end": w.get("end")}
                                 for w in d["words"] if w.get("word")], final=ev == "EndOfTurn")
            if ev in ("EagerEndOfTurn", "TurnResumed", "EndOfTurn", "StartOfTurn"):
                self.on_event({"t": "ear", "kind": ev, "ear": "flux", "turn": ti, "text": d.get("transcript", ""),
                               "eot_p": _num(d.get("end_of_turn_confidence")), "trigger": d.get("trigger"),
                               "word_end": self.interim[-1]["end"] if self.interim else None,
                               "at": round(self.at_s, 2)})
            eot_p = _num(d.get("end_of_turn_confidence"))
            for s in self.sinks:                   # Flux's turn state for the session's fused end-of-turn score
                f = getattr(s, "on_turn_info", None)
                if f is not None:
                    f(ev, eot_p)
            if ev == "EndOfTurn":
                self.eager_at = None
                self.finalized.set()
                self._final("eot_" + str(d.get("trigger") or "model"))
                return
            self._partial()
            for s in self.sinks:                   # after the words: FLUX_SOFT=fused scores every Flux update
                f = getattr(s, "on_turn_tick", None)
                if f is not None:
                    f(ev, list(self.interim))
            if ev == "EagerEndOfTurn" and self.interim:
                self.eager_at = self.at_s
                now = time.perf_counter()
                for s in self.sinks:
                    f = getattr(s, "on_eager", None)
                    if f is not None:
                        f(list(self.interim), now)
            elif ev == "TurnResumed":
                self.eager_at = None
        elif t == "Connected":
            self.request_id = d.get("request_id")
        elif t in ("Error", "Warning", "ConfigureFailure"):
            self.on_event({"t": "ear", "kind": "error" if t == "Error" else "warning", "ear": "flux",
                           "text": f"{d.get('code') or ''} {d.get('description') or ''}".strip()})

    async def flush(self):
        """End of the run: ForceEndTurn makes Flux close the open turn now (EndOfTurn, trigger=manual)."""
        if self.closed:
            return
        if self.failed is None and self.interim:
            self.finalized.clear()
            self.out.put_nowait(FORCE_END)
            try:
                await asyncio.wait_for(self.finalized.wait(), timeout=config.DG_FLUSH_TIMEOUT_S)
            except asyncio.TimeoutError:
                pass
        if self.failed is None:
            self._final("flush")
        else:
            async with self.fb_lock:
                if self.fb is not None:
                    await self.fb.flush()
        if self.meta is not None:
            await self.meta.flush()
        if self.final_tasks:
            await asyncio.gather(*self.final_tasks, return_exceptions=True)

    def close(self):
        if self.meta is not None:
            self.meta.close()
        super().close()


def _num(x):
    try:
        return round(float(x), 3)
    except (TypeError, ValueError):
        return None
