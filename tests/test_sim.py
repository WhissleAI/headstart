"""Tests for the offline policy simulator (sim/): virtual clock semantics, cached tagger equivalence, end-to-end replay.

No network, no embedding model: vectors are a bag-of-words hash, web docs are generated per query.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import config, frame  # noqa: E402
from sim import fakes, vclock  # noqa: E402
from sim.simulate import CURRENT, Simulator, summarize  # noqa: E402


# ----------------------------------------------------------------------------- virtual clock
def test_virtual_clock_orders_like_real_time_without_sleeping():
    order = []

    async def worker(name, d):
        await asyncio.sleep(d)
        order.append((name, round(time.perf_counter() - 1000.0, 6)))

    async def main():
        t = time.perf_counter()
        await asyncio.gather(worker("b", 0.5), worker("a", 0.2), worker("c", 30.0))
        try:
            await asyncio.wait_for(asyncio.sleep(5), timeout=0.8)
        except asyncio.TimeoutError:
            order.append(("timeout", round(time.perf_counter() - t, 6)))
        return time.perf_counter() - t

    wall = time.perf_counter()
    took = vclock.run(main)
    assert time.perf_counter() - wall < 1.0                      # 30.8 virtual seconds, no real sleeping
    assert [n for n, _ in order] == ["a", "b", "c", "timeout"]
    assert order[0][1] == pytest.approx(0.2) and order[2][1] == pytest.approx(30.0)
    assert took == pytest.approx(30.8)


def test_virtual_clock_survives_busy_wait():
    """answer.Stream._run re-awaits asyncio.wait() on an already-finished loser task while the hedge winner streams:
    a busy spin that never sleeps. Virtual time must still advance so the winner can finish."""
    async def main():
        loser = asyncio.create_task(asyncio.sleep(0))
        winner = asyncio.create_task(asyncio.sleep(0.3))
        await loser
        t = time.perf_counter()
        while not winner.done():
            await asyncio.wait([loser, winner], return_when=asyncio.FIRST_COMPLETED)
        return time.perf_counter() - t
    took = vclock.run(main)
    assert 0.3 <= took < 0.35


def test_virtual_clock_detects_deadlock():
    async def main():
        await asyncio.Event().wait()
    with pytest.raises(vclock.Deadlock):
        vclock.run(main)


# ----------------------------------------------------------------------------- cached tagger
def test_cached_tagger_matches_native():
    from server.tagger import Tagger
    seq = [("push", ["who"], None), ("push", ["founded", "the"], None), ("push", ["condo"], None),
           ("push", ["company", "nvidia"], 3), ("ep",)]
    real, cached, cached2 = Tagger(), fakes.CachedTagger(), fakes.CachedTagger()
    for t in (cached, cached2):
        t.reset()
    outs = {id(real): [], id(cached): [], id(cached2): []}
    for op in seq:
        for t in (real, cached, cached2):
            if op[0] == "push":
                t.push(op[1], from_=op[2])
            else:
                t.endpoint()
            outs[id(t)].append(json.dumps(t.poll(), sort_keys=True))
    assert outs[id(real)] == outs[id(cached)] == outs[id(cached2)]
    for t in (real, cached, cached2):
        t.close()


# ----------------------------------------------------------------------------- end to end
def _vec(text):
    v = np.zeros(64, np.float32)
    for w in frame.norm(text).split():
        if w not in frame.STOP:
            v[hash(w) % 64] += 1
    n = np.linalg.norm(v)
    return v / n if n else v + 1e-3


class _Web(dict):
    def get(self, q, default=None):
        return [{"id": f"web:{q}:{i}", "title": q.title(), "text": f"{q.title()} is a thing. Fact {i} about {q}.",
                 "url": ""} for i in range(3)]


class FakeCaches(fakes.Caches):
    def __init__(self):
        self.web, self.emb, self.missing_web, self.dirty = _Web(), {}, set(), False

    def save(self):
        pass

    def vec_query(self, text):
        return _vec(text)

    def vec_docs(self, texts, titles=None):
        return np.stack([_vec((ti or "") + " " + t) for t, ti in zip(texts, titles or [""] * len(texts))])


def scripted_trace(text, gap=0.28, tail=1.6, qid="t1"):
    """A trace in the recorder's format: Arc-like partials every 100 ms (words visible 0.15 s after they end, the
    words refresh only every 0.4 s like Arc), RMS loud during speech, final at the ear's pause endpoint."""
    words = text.split()
    times = [(0.3 + i * gap, 0.3 + i * gap + gap * 0.9) for i in range(len(words))]
    end = times[-1][1]
    ev, t, last = [], 0.1, None
    while t < end + tail:
        loud = t < end + 0.05
        ev.append({"k": "audio", "w": round(t, 4), "at": round(t, 4), "n": 1600, "rms": 0.1 if loud else 0.0005})
        if int(round(t * 10)) % 4 == 0 or not last:
            last = [dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times) if b <= t - 0.15]
        if last:
            ev.append({"k": "partial", "w": round(t + 0.01, 4), "words": last})
        if t >= end + 0.5 and not any(e["k"] == "final" for e in ev):
            ev.append({"k": "final", "w": round(t + 0.01, 4),
                       "words": [dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times)]})
            break
        t = round(t + 0.1, 4)
    return {"id": qid, "text": text, "kind": "factoid", "events": ev}


@pytest.fixture
def sim(monkeypatch):
    from server import lanes
    monkeypatch.setattr(lanes, "_index", lanes.LocalIndex.__new__(lanes.LocalIndex))
    lanes._index.docs, lanes._index.tv, lanes._index.av, lanes._index.av_rows = [], np.zeros((0, 64)), None, None
    s = Simulator(caches=FakeCaches(), setattr_fn=monkeypatch.setattr)
    yield s


def test_replay_headstart_beats_baseline_and_is_deterministic(sim):
    tr = scripted_trace("who founded the company Nvidia")
    prof = fakes.PROFILES["wikipedia"]
    hs = sim.run_question(tr, "headstart", CURRENT, prof, seed=1)
    bl = sim.run_question(tr, "baseline", CURRENT, prof, seed=1)
    assert hs["error"] is None and bl["error"] is None, (hs["error"], bl["error"])
    assert hs["path"] in ("draft-reuse", "speculative-hit") and "nvidia" in hs["query"]
    assert bl["path"] == "search-after-endpoint" and bl["web_calls"] == 1
    assert hs["ready_ms"] < bl["ready_ms"], (hs["ready_ms"], bl["ready_ms"])
    assert hs["grounded"] and not hs["risky"]
    again = sim.run_question(tr, "headstart", CURRENT, prof, seed=1)
    assert again["ready_ms"] == hs["ready_ms"] and again["path"] == hs["path"]      # common random numbers


def test_knobs_take_effect_and_are_restored(sim):
    tr = scripted_trace("what is the capital of Australia", qid="t2")
    prof = fakes.PROFILES["tavily"]
    before = config.WEB_BUDGET
    lo = sim.run_question(tr, "headstart", dict(CURRENT, SPEC_WEB_BUDGET=1), prof, seed=0)
    hi = sim.run_question(tr, "headstart", dict(CURRENT, SPEC_WEB_BUDGET=8), prof, seed=0)
    assert config.WEB_BUDGET == before
    assert lo["web_calls"] <= 2 and hi["web_calls"] >= lo["web_calls"]
    s = summarize([lo, hi])
    assert s["n"] == 2 and 0 <= s["risky_rate"] <= 1
