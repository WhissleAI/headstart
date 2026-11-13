"""Noisy-room semantic endpoint (ASR word-gap silence) and the chatter guard (robustness findings A and B)."""
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_speculation import fakes  # noqa: F401,E402  (autouse fixture: fake lanes / embed / answer model)

from server import config  # noqa: E402
from server.session import Session  # noqa: E402


async def _feed(s, t0, words, times, until, level, noise, start=0.0):
    t = start
    while t < until:
        while time.perf_counter() < t0 + t:
            await asyncio.sleep(0.002)
        speaking = any(a <= t <= b + 0.05 for a, b in times)
        s.on_audio(np.ones(1600, np.float32) * (level if speaking else noise), t + 0.1)
        heard = [dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times) if b <= t - 0.1]
        if heard:
            s.on_partial(heard)
        t = round(t + 0.1, 3)


def test_semantic_endpoint_fires_in_steady_noise(monkeypatch):
    """Background at 0.04 RMS (above the 0.01 floor clamp): energy never sees silence; the word gap must."""
    monkeypatch.setattr(config, "SEMANTIC_ENDPOINT", True)
    monkeypatch.setattr(config, "SEM_STABLE_MS", 200)

    async def go():
        ev = []
        t0 = time.perf_counter()
        s = Session(ev.append, mode="headstart", t0=t0)
        words = "who founded the company Nvidia".split()
        times = [(0.6 + i * 0.25, 0.6 + i * 0.25 + 0.22) for i in range(len(words))]
        await _feed(s, t0, words, times, times[-1][1] + 1.3, level=0.15, noise=0.04)
        fired = [e for e in ev if e["t"] == "endpoint" and e.get("by") == "semantic"]
        if s.final_task:
            await asyncio.wait({s.final_task})
        s.close()
        return s, fired
    s, fired = asyncio.run(go())
    assert s.noisy
    assert fired, "semantic endpoint should fire from the ASR word gap in steady noise"


def test_quiet_chatter_gets_no_early_answer(monkeypatch):
    monkeypatch.setattr(config, "SEMANTIC_ENDPOINT", True)
    monkeypatch.setattr(config, "SEM_STABLE_MS", 200)

    async def go():
        ev = []
        t0 = time.perf_counter()
        s = Session(ev.append, mode="headstart", t0=t0)
        q = "who founded the company Nvidia".split()
        qt = [(0.3 + i * 0.25, 0.3 + i * 0.25 + 0.22) for i in range(len(q))]
        await _feed(s, t0, q, qt, qt[-1][1] + 0.8, level=0.2, noise=0.0005)
        await s.on_final([dict(text=w, start=a, end=b) for w, (a, b) in zip(q, qt)])
        n_soft_before = sum(1 for e in ev if e["t"] == "endpoint" and e.get("by") == "semantic")
        # a neighbour, 16 dB quieter, asks something
        c = "what time is the game tonight".split()
        base = qt[-1][1] + 1.5
        ct = [(base + i * 0.25, base + i * 0.25 + 0.22) for i in range(len(c))]
        await _feed(s, t0, c, ct, ct[-1][1] + 1.0, level=0.03, noise=0.0005, start=round(qt[-1][1] + 0.9, 1))
        n_soft_after = sum(1 for e in ev if e["t"] == "endpoint" and e.get("by") == "semantic")
        s.close()
        return s, n_soft_before, n_soft_after
    s, before, after = asyncio.run(go())
    assert s.answered_level is not None
    assert after == before, "quiet chatter must not get a semantic (early) answer"
    assert len(s.history) == 1
