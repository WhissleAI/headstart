"""Degraded / offline demo mode, with fakes only (no network):
- web lane: recorded replay cache (WEB_BACKEND=cache), fallback to it when the live lane fails, breaker, honesty flags;
- answer model unreachable: extractive answer (flagged), breaker, deadline;
- A/B fairness: a Race (both modes) completes with web AND answer model down, both modes on the same fallbacks;
- /api/health + the ready event report the degraded state.
Run: .venv/bin/python -m pytest -q tests/test_demo_fallbacks.py
"""
import asyncio
import json
import time

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import answer, config, embed, lanes, speculator
from server import app as appmod
from server.session import Session

RealStream = answer.Stream


# ----------------------------------------------------------------------------- fakes
def wiki_payload(q):
    return {"query": {"pages": [
        {"index": 1, "pageid": 1, "title": "Nvidia", "fullurl": "https://en.wikipedia.org/wiki/Nvidia",
         "extract": "Nvidia Corporation is an American technology company. It was founded in 1993 by Jensen Huang, "
                    "Chris Malachowsky and Curtis Priem. Its headquarters are in Santa Clara."}]}}


class Wire:
    """httpx transport: 'up' answers like Wikipedia, 'down' raises ConnectError; counts requests."""

    def __init__(self, up=True):
        self.up, self.calls = up, 0

    def __call__(self, request: httpx.Request):
        self.calls += 1
        if not self.up:
            raise httpx.ConnectError("Network is unreachable", request=request)
        return httpx.Response(200, json=wiki_payload(request.url.params.get("gsrsearch")))


@pytest.fixture
def wire(monkeypatch):
    w = Wire()
    lanes.use_transport(httpx.MockTransport(w))
    monkeypatch.setattr(config, "WEB_BACKEND", "wikipedia")
    yield w
    lanes.use_transport(None)


def make_cache(tmp_path, monkeypatch, entries=None):
    p = tmp_path / "web_cache.json"
    c = lanes.WebCache(p)
    for q, docs in (entries or {"company founded nvidia": [
            lanes.Doc(id="https://en.wikipedia.org/wiki/Nvidia", title="Nvidia", source="wikipedia",
                      url="https://en.wikipedia.org/wiki/Nvidia",
                      text="Nvidia is a chip company. It was founded in 1993 by Jensen Huang, Chris Malachowsky "
                           "and Curtis Priem.")],
            "capital australia": [lanes.Doc(id="w:canberra", title="Canberra", source="wikipedia",
                                            text="Canberra is the capital city of Australia.")]}).items():
        c.put(q, docs, "wikipedia")
    c.save()
    monkeypatch.setattr(config, "WEB_CACHE_PATH", p)
    lanes.set_web_cache(None)
    return p


# ----------------------------------------------------------------------------- web cache
def test_cache_exact_fuzzy_miss_and_roundtrip(tmp_path, monkeypatch):
    p = make_cache(tmp_path, monkeypatch)
    c = lanes.WebCache(p)                                     # reloaded from disk
    assert len(c) == 2 and c.recorded
    docs, match, _ = c.get("founded company nvidia")         # word order doesn't matter
    assert match == "exact" and docs[0].title == "Nvidia" and docs[0].meta["cached"]
    docs, match, cq = c.get("founded nvidia")                # a speculative partial: covered by the full query
    assert match == "fuzzy" and cq == "company founded nvidia"
    assert c.get("capital turkey") is None                   # half the words: not a hit (no wrong-country answer)
    assert c.get("") is None


def test_backend_cache_serves_recorded_and_flags_it(tmp_path, monkeypatch, wire):
    make_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "WEB_BACKEND", "cache")
    meter = {}
    docs = asyncio.run(lanes.web("company founded nvidia", meter=meter))
    assert wire.calls == 0                                    # never touches the network
    assert docs and meter["backend"] == "cache" and meter["replayed"] and meter["cache_match"] == "exact"
    assert docs[0].brief()["cached"] is True                 # the UI labels each cached source
    meter = {}
    assert asyncio.run(lanes.web("population of mars", meter=meter)) == [] and meter["cache_miss"]
    h = lanes.web_health()
    assert h["status"] == "cache" and not h["live"] and h["cache_entries"] == 2


def test_live_failure_falls_back_to_cache_then_breaker(tmp_path, monkeypatch, wire):
    make_cache(tmp_path, monkeypatch)
    wire.up = False

    async def go():
        m1, m2 = {}, {}
        d1 = await lanes.web("company founded nvidia", meter=m1)
        d2 = await lanes.web("capital australia", meter=m2, final=True)
        return d1, m1, d2, m2
    d1, m1, d2, m2 = asyncio.run(go())
    assert d1[0].title == "Nvidia" and m1["replayed"] and "ConnectError" in m1["fallback"]
    assert d2[0].title == "Canberra" and m2["replayed"] and "down" in m2["fallback"]
    assert wire.calls == 1                                    # breaker: the second call didn't wait on the network
    assert lanes.web_health()["status"] == "down"
    assert "web" in appmod.degraded() and "recorded cache" in appmod.degraded()["web"]


def test_live_failure_without_cache_raises_and_reports_none(wire):
    wire.up = False
    with pytest.raises(httpx.ConnectError):
        asyncio.run(lanes.web("company founded nvidia", meter={}))
    h = lanes.web_health()
    assert h["status"] == "down" and h["fallback"] == "none"
    assert "no cache" in appmod.degraded()["web"]


def test_live_success_is_live_and_record_mode_writes_cache(tmp_path, monkeypatch, wire):
    p = tmp_path / "rec.json"
    monkeypatch.setattr(config, "WEB_CACHE_PATH", p)
    monkeypatch.setattr(config, "WEB_CACHE_RECORD", True)
    lanes.set_web_cache(None)
    meter = {}
    docs = asyncio.run(lanes.web("company founded nvidia", meter=meter))
    assert docs and meter["backend"] == "wikipedia" and not meter.get("replayed")
    assert not docs[0].brief().get("cached")
    assert lanes.web_health()["status"] == "live" and appmod.degraded() == {}
    saved = json.loads(p.read_text())
    assert "company founded nvidia" in saved["entries"]


def test_slow_live_web_times_out_into_cache(tmp_path, monkeypatch):
    make_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "WEB_BACKEND", "wikipedia")
    monkeypatch.setattr(config, "WEB_LIVE_TIMEOUT_S", 0.1)

    async def slow(q, k=4):
        await asyncio.sleep(5)
    monkeypatch.setattr(lanes, "wikipedia", slow)
    meter = {}
    t = time.perf_counter()
    docs = asyncio.run(lanes.web("company founded nvidia", meter=meter))
    assert time.perf_counter() - t < 1.0 and docs and meter["replayed"]


# ----------------------------------------------------------------------------- extractive answers
def test_extractive_picks_matching_sentence_with_citation():
    docs = [lanes.Doc(id="1", title="Nvidia", text="Nvidia is an American company. It was founded in 1993 by Jensen "
                                                    "Huang and others. Its headquarters are in Santa Clara."),
            lanes.Doc(id="2", title="GPU", text="A GPU is a processor.")]
    out = answer.extractive("who founded the company nvidia", docs)
    assert out.startswith("It was founded in 1993 by Jensen Huang") and out.endswith("[1]")
    assert answer.extractive("anything", []) is None


class DownClient:
    """Anthropic client whose every request fails like a dead network (or hangs, hang=True)."""

    def __init__(self, hang=False):
        self.hang, self.calls = hang, 0

    @property
    def messages(self):
        return self

    def stream(self, **kw):
        self.calls += 1
        hang = self.hang

        class S:
            async def __aenter__(self):
                if hang:
                    await asyncio.sleep(30)
                raise ConnectionError("Connection error.")

            async def __aexit__(self, *a):
                return False
        return S()


DOCS = [lanes.Doc(id="1", title="Canberra", text="Canberra is the capital city of Australia. It was founded in 1913.")]


def test_answer_model_down_extractive_then_breaker(monkeypatch):
    dc = DownClient()
    monkeypatch.setattr(answer, "_client", dc)

    async def go():
        a = RealStream("what is the capital of australia", DOCS)
        await a.task
        t = time.perf_counter()
        b = RealStream("what is the capital of australia", DOCS)
        await b.task
        return a, b, time.perf_counter() - t
    a, b, tb = asyncio.run(go())
    assert a.error is None and a.fallback == "extractive" and "Canberra is the capital" in a.text()
    assert "Connection" in a.fallback_reason and a.t_first is not None
    assert answer.health()["status"] == "down" and "answer" in appmod.degraded()
    calls = dc.calls
    assert b.fallback == "extractive" and tb < 0.05 and dc.calls == calls      # breaker: no new request


def test_answer_model_hanging_hits_deadline(monkeypatch):
    monkeypatch.setattr(answer, "_client", DownClient(hang=True))
    monkeypatch.setattr(config, "ANSWER_DEADLINE_MS", 300)
    monkeypatch.setattr(config, "HEDGE_MS", 100)

    async def go():
        t = time.perf_counter()
        s = RealStream("capital of australia", DOCS)
        await s.task
        return s, time.perf_counter() - t
    s, took = asyncio.run(go())
    assert s.fallback == "extractive" and s.text().endswith("[1]") and 0.25 < took < 1.0


def test_fallback_off_keeps_the_error(monkeypatch):
    monkeypatch.setattr(answer, "_client", DownClient())
    monkeypatch.setattr(config, "ANSWER_FALLBACK", False)

    async def go():
        s = RealStream("capital of australia", DOCS)
        await s.task
        return s
    s = asyncio.run(go())
    assert s.fallback is None and s.error and not s.tokens


# ----------------------------------------------------------------------------- full race, everything down
def _vec(text):
    v = np.zeros(64, np.float32)
    for w in text.lower().split():
        v[hash(w) % 64] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v + 1e-3


async def speak(text, mode, events):
    t0 = time.perf_counter()
    s = Session(events.append, mode=mode, t0=t0)
    words = text.split()
    times = [(0.2 + i * 0.25, 0.2 + i * 0.25 + 0.22) for i in range(len(words))]
    t = 0.0
    while t < times[-1][1] + 0.5:
        while time.perf_counter() < t0 + t:
            await asyncio.sleep(0.002)
        heard = [dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times) if b <= t - 0.1]
        if heard:
            s.on_partial(heard)
        t += 0.1
    m = await s.on_final([dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times)])
    s.close()
    return m


def test_race_completes_offline_with_same_fallbacks_in_both_modes(tmp_path, monkeypatch, wire):
    make_cache(tmp_path, monkeypatch)
    wire.up = False                                          # venue Wi-Fi gone: web AND answer model unreachable
    monkeypatch.setattr(answer, "_client", DownClient())
    monkeypatch.setattr(config, "SEMANTIC_ENDPOINT", False)
    monkeypatch.setattr(lanes, "_index", lanes.LocalIndex.__new__(lanes.LocalIndex))
    lanes._index.docs, lanes._index.tv, lanes._index.av, lanes._index.av_rows = [], np.zeros((0, 64)), None, None
    monkeypatch.setattr(embed, "encode_queries", lambda ts: np.stack([_vec(t) for t in ts]))
    monkeypatch.setattr(embed, "encode_docs", lambda ts, titles=None: np.stack(
        [_vec((ti or "") + " " + t) for t, ti in zip(ts, titles or [""] * len(ts))]))

    async def arun(fn, *a):
        return fn(*a)
    monkeypatch.setattr(embed, "arun", arun)
    monkeypatch.setattr(speculator.answer, "Stream", RealStream)

    async def go():
        ev = {"headstart": [], "baseline": []}
        ms = await asyncio.gather(*(speak("who founded the company Nvidia", m, ev[m]) for m in ev))
        return dict(zip(ev, ms)), ev
    ms, ev = asyncio.run(go())
    for mode, m in ms.items():
        assert m["answer"] and "Jensen Huang" in m["answer"], (mode, m)
        assert m["answer_fallback"] == "extractive", mode
        assert m["web_replayed"] >= 1 and m["web_backend"] == "cache", mode
        metric = next(e for e in ev[mode] if e["t"] == "metric")
        assert metric["answer_fallback"] == "extractive" and metric["web_replayed"] >= 1
        assert not [e for e in ev[mode] if e["t"] == "answer_error"]
        srcs = next(e for e in ev[mode] if e["t"] == "sources")["docs"]
        assert srcs and all(d.get("cached") for d in srcs)


# ----------------------------------------------------------------------------- health + ready event
def test_health_and_ready_event_report_degraded(tmp_path, monkeypatch):
    make_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "WEB_BACKEND", "cache")
    answer._down("ConnectionError: offline")
    monkeypatch.setattr(appmod.app.router, "on_startup", [])       # no model warm-up in a unit test
    monkeypatch.setattr(lanes, "_index", lanes.LocalIndex.__new__(lanes.LocalIndex))
    lanes._index.docs = []
    with TestClient(appmod.app) as c:
        h = c.get("/api/health").json()
        assert h["web"]["status"] == "cache" and h["web"]["cache_entries"] == 2
        assert h["answer"]["status"] == "down" and h["answer"]["fallback"] == "extractive"
        assert set(h["degraded"]) == {"web", "answer"} and "not live" in h["degraded"]["web"]
        with c.websocket_connect("/ws") as ws:
            r = ws.receive_json()
            assert r["t"] == "ready" and set(r["degraded"]) == {"web", "answer"} and r["web"]["status"] == "cache"
