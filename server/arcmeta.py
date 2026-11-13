"""Arc audio metadata on a cloud ear's words (ARC_META=mini|transcriber|off).

The words come from Deepgram (Flux or nova-3); Arc listens to the same audio in external-transcript mode and runs its
heads only (no transcription decoder, no builtin fallback decoder): device-directed ("for me"), command_complete
(continuous probability and its event), act / intent of the open utterance, speaker id + role, bio events (breath,
throat clear, ...), and, with Arc v1.1, audio events.

  mini         Arc v1 Mini (arc-v1 1.0.1, the model already loaded for the Arc ear / fallback: no extra RAM).
               Measured on this M3 Pro with real Flux words: ~10 ms of CPU per 100 ms chunk (RTF ~0.1),
               command_complete ~0.2 s after the last word.
  transcriber  Arc Transcriber (Arc v1.1 "full": NVIDIA Parakeet-TDT 0.6B encoder tap 12 + Arc adapters), code from the
               arc_transcriber package (1.2.1, loaded under another module name so it doesn't clash with arc-v1). Its
               backbone streams in chunks (ARC_T_CHUNK_S + ARC_T_CTX_S of context each side): on MPS with 0.5 s chunks
               RTF ~0.25 but metadata lands ~1.2-1.4 s after the last word and ~3.1 GB RSS (PROGRESS.md). Too late to
               steer endpointing; read-out only.
  off          nothing.

Everything runs on one dedicated thread per ArcMeta (audio + words are applied in order); results come back to the
asyncio loop as summaries via on_meta(meta)."""
from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config

CC = "discourse.assistant.command_complete"
_t_arc = None
_t_lock = threading.Lock()


def transcriber():
    """Arc Transcriber (v1.1 full) via the arc_transcriber package, imported as `arc_audio_t`."""
    global _t_arc
    with _t_lock:
        if _t_arc is None:
            name = "arc_audio_t"
            if name not in sys.modules:
                pkg = config.ARC_T_CODE / "arc_audio"
                spec = importlib.util.spec_from_file_location(name, pkg / "__init__.py",
                                                              submodule_search_locations=[str(pkg)])
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
            Arc = sys.modules[name].Arc
            a = Arc.from_pretrained(str(config.ARC_T_WEIGHTS), backbone="full", device=config.ARC_T_DEVICE,
                                    threads=config.ARC_THREADS, load_decoder=False)
            a.cfg["backbone"]["stream_chunk_s"] = config.ARC_T_CHUNK_S
            a.cfg["backbone"]["stream_context_s"] = config.ARC_T_CTX_S
            _t_arc = a
        return _t_arc


def variant() -> str:
    v = (config.ARC_META or "off").strip().lower()
    return v if v in ("mini", "transcriber") else "off"


def warmup():
    if variant() == "transcriber":
        t = time.perf_counter()
        m = ArcMeta(lambda e: None)
        for _ in range(15):
            m._feed(np.zeros(1600, np.float32))
        m._flush()
        return time.perf_counter() - t
    return 0.0


def _top(xs):
    if not xs:
        return None
    x = xs[0]
    return {"label": x.get("label"), "p": x.get("p")}


class ArcMeta:
    """Feed it the ear's audio and words; it calls on_meta(summary) whenever the summary changes."""

    def __init__(self, on_meta, domain: str = "assistant", offset_s: float = 0.0):
        self.on_meta = on_meta
        self.v = variant()
        if self.v == "transcriber":
            a = transcriber()
        else:
            from .ear import arc
            a = arc()
        # fallback=None: external mode must not quietly run Arc's builtin decoder as a name fallback
        self.stream = a.stream(domain=domain, transcript="external", name_source="word", interim_source="deepgram",
                               partials=False, fallback=None)
        self.offset = float(offset_s)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arcmeta")
        self.loop = None
        self.closed = False
        self.last_key = None
        self.cc_events: list[dict] = []
        self.compute_s = 0.0
        self.audio_s = 0.0
        self.pending = 0
        self.lag_drops = 0

    # ------------------------------------------------------------------ in (from the ear, on the event loop)
    def feed(self, pcm: np.ndarray):
        if self.closed:
            return
        if self.loop is None:
            self.loop = asyncio.get_running_loop()
        if self.pending > config.ARC_META_MAX_PENDING:     # the model can't keep up: drop audio, never block the ear
            self.lag_drops += 1
            return
        self.pending += 1
        self.pool.submit(self._run, self._feed, pcm)

    def words(self, ws: list[dict], final: bool):
        """ws: [{text,start,end}] on the audio clock (Flux's punctuated words are fine: Arc reads them as text)."""
        if self.closed or not ws:
            return
        ev = [{"text": w["text"], "start": float(w["start"]) - self.offset, "end": float(w["end"]) - self.offset,
               "is_final": final} for w in ws if w.get("start") is not None]
        self.pool.submit(self._run, self.stream.add_words, ev)

    async def flush(self):
        if self.closed:
            return
        fut = asyncio.get_running_loop().run_in_executor(self.pool, self._run, self._flush)
        try:
            await asyncio.wait_for(fut, timeout=3.0)
        except Exception:  # noqa: BLE001
            pass

    def close(self):
        self.closed = True
        self.pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ on the Arc thread
    def _run(self, fn, *a):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001 (a metadata failure must never touch the words path)
            self._emit({"error": f"{type(e).__name__}: {e}"[:200]})

    def _feed(self, pcm):
        t = time.perf_counter()
        out = self.stream.feed(pcm)
        self.compute_s += time.perf_counter() - t
        self.audio_s += len(pcm) / 16000
        self.pending = max(0, self.pending - 1)
        self._summarize(out)

    def _flush(self):
        out = self.stream.flush()
        self._summarize(out, final=True)

    def _summarize(self, out: dict, final: bool = False):
        o = self.offset
        for e in out.get("events") or []:
            if e.get("tag") == CC:
                self.cc_events.append({"t": None if e.get("t") is None else round(e["t"] + o, 2), "p": e.get("p")})
        lv = out.get("live") or {}
        u = lv.get("utt") or {}
        if u:
            self.last_utt = u                  # the read-out keeps the last utterance's values after it closes
        else:
            u = getattr(self, "last_utt", None) or {}
        sp = lv.get("speaker") or {}
        if sp.get("id"):
            self.last_sp = sp
        else:
            sp = getattr(self, "last_sp", None) or {}
        role = _top(sp.get("role"))
        m = {
            "model": self.v, "at": None if lv.get("t") is None else round(lv["t"] + o, 2),
            "for_me": None if u.get("device_directed") is None else round(float(u["device_directed"]), 2),
            "cc_p": None if u.get("command_complete") is None else round(float(u["command_complete"]), 2),
            "cc_events": self.cc_events[-3:],
            "act": _top(u.get("act")), "intent": _top(u.get("intent")),
            "speaker": sp.get("id"), "role": (role or {}).get("label"),
            "bio": [{"tag": b.get("tag"), "t": b.get("t")} for b in (out.get("bio_events") or [])],
            "audio_events": [{"tag": a.get("tag") or a.get("label"), "p": a.get("p")} for a in
                             (out.get("audio_events") or [])],
            "utt_open": bool(lv.get("utt")),
            "rtf": round(self.compute_s / self.audio_s, 3) if self.audio_s else None,
        }
        if final:
            m["final"] = True
        key = (m["for_me"], m["cc_p"], len(self.cc_events), (m["act"] or {}).get("label"), m["speaker"], m["role"],
               bool(m["bio"]), bool(m["audio_events"]), m["utt_open"], final)
        if key == self.last_key:
            return
        self.last_key = key
        self._emit(m)

    def _emit(self, m):
        if self.loop is not None and not self.closed:
            self.loop.call_soon_threadsafe(self.on_meta, m)
