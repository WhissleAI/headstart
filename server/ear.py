"""Arc ear: 16 kHz audio in (100 ms chunks), live words out.

Arc runs on its own thread (torch, CPU). Each feed returns `transcript_partial` (the open utterance's words, refreshed
about every 0.4 s, revisable) and `transcript` (closed utterances, endpointed by Arc at a pause). We turn those into
on_partial / on_final calls for one or more sessions (fan-out: the A/B race feeds two brains from one ear).
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config

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
    t = time.perf_counter()
    s = arc().stream(domain="assistant", transcript="builtin", partials=True)
    z = np.zeros(1600, np.float32)
    rng = np.random.default_rng(0)
    for i in range(20):
        s.feed(z if i % 2 else rng.standard_normal(1600).astype(np.float32) * 0.001)
    s.flush()
    return time.perf_counter() - t


def _words(u: dict) -> list[dict]:
    return [{"text": w["text"], "start": w.get("start"), "end": w.get("end")} for w in (u.get("words") or [])]


class ArcEar:
    """Feed audio; call back into sinks. Sinks get: on_audio(pcm, at_s), on_partial(words), await on_final(words)."""

    def __init__(self, sinks: list, domain: str = "assistant", on_event=None):
        self.sinks = sinks
        self.stream = arc().stream(domain=domain, transcript="builtin", partials=True)
        self.samples = 0
        self.on_event = on_event or (lambda e: None)
        self.last_partial_key = None
        self.final_tasks: list[asyncio.Task] = []
        self.closed = False
        self.cur_start = None

    @property
    def at_s(self):
        return self.samples / 16000

    async def feed(self, pcm: np.ndarray):
        if self.closed:
            return
        pcm = np.asarray(pcm, np.float32).reshape(-1)
        self.samples += len(pcm)
        at = self.at_s
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
        for u in out.get("transcript") or []:
            ws = _words(u)
            self.on_event({"t": "ear", "kind": "final", "text": u.get("text", ""), "reason": u.get("reason"),
                           "start": u.get("start"), "end": u.get("end"), "at": round(self.at_s, 2)})
            self.last_partial_key = None
            for s in self.sinks:
                self.final_tasks.append(asyncio.create_task(s.on_final(ws)))
        p = out.get("transcript_partial")
        now = time.perf_counter()
        if p and p.get("words") and not p.get("final"):
            ws = _words(p)
            key = tuple(w["text"] for w in ws)
            if key != self.last_partial_key:
                self.on_event({"t": "ear", "kind": "partial", "text": p.get("text", ""), "at": round(self.at_s, 2)})
            self.last_partial_key = key
            for s in self.sinks:
                s.on_partial(ws, now)       # every chunk is a tick, even with unchanged words
        for e in out.get("events") or []:
            if e.get("tag") == "discourse.assistant.command_complete":
                self.on_event({"t": "ear", "kind": "arc_command_complete", "t_audio": e.get("t"),
                               "p": e.get("p"), "at": round(self.at_s, 2)})

    def close(self):
        self.closed = True
