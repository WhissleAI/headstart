"""Deterministic tests for frames + speculator + session, with fake lanes and a fake answer model (no network).

Run: .venv/bin/python -m pytest -q tests
"""
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import answer, config, embed, frame, lanes, speculator  # noqa: E402
from server.session import Session  # noqa: E402
from server.tagger import Tagger  # noqa: E402


# ----------------------------------------------------------------------------- fakes
def _vec(text: str) -> np.ndarray:
    """Bag-of-words hashing embedding: same words -> same vector. Good enough to test ranking logic."""
    v = np.zeros(64, np.float32)
    for w in frame.norm(text).split():
        if w in frame.STOP:
            continue
        v[hash(w) % 64] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v + 1e-3


WEB_CALLS: list[str] = []
WEB_DELAY = 0.15


async def fake_web(query, topic="general", **kw):
    WEB_CALLS.append(query)
    await asyncio.sleep(WEB_DELAY)
    return [lanes.Doc(id=f"web:{query}:{i}", title=query.title(), text=f"About {query}: fact {i}.", url="",
                      source="fake") for i in range(3)]


class FakeStream:
    instances: list["FakeStream"] = []

    def __init__(self, question, docs, tag="", context=""):
        self.question, self.docs, self.tag = question, list(docs), tag
        self.tokens, self.done, self.error = [], False, None
        self.t_start = time.perf_counter()
        self.t_first = None
        self.cancelled = False
        FakeStream.instances.append(self)
        self.task = asyncio.create_task(self._run())

    async def _run(self):
        await asyncio.sleep(0.3)  # TTFT
        self.t_first = time.perf_counter()
        self.tokens.append(f"Answer to {self.question}: " + " ".join(d.title for d in self.docs[:2]) + ".")
        self.done = True

    async def follow(self):
        while not self.done:
            await asyncio.sleep(0.01)
        for t in self.tokens:
            yield t

    def text(self):
        return "".join(self.tokens)

    def cancel(self):
        self.cancelled = True
        if not self.task.done():
            self.task.cancel()
        self.done = True
        self.error = self.error or "cancelled"


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    WEB_CALLS.clear()
    FakeStream.instances.clear()
    monkeypatch.setattr(lanes, "web", fake_web)
    monkeypatch.setattr(lanes, "_index", lanes.LocalIndex.__new__(lanes.LocalIndex))
    lanes._index.docs, lanes._index.tv, lanes._index.av, lanes._index.av_rows = [], np.zeros((0, 64)), None, None
    monkeypatch.setattr(embed, "encode_queries", lambda ts: np.stack([_vec(t) for t in ts]))
    monkeypatch.setattr(embed, "encode_docs", lambda ts, titles=None: np.stack(
        [_vec((ti or "") + " " + t) for t, ti in zip(ts, titles or [""] * len(ts))]))

    async def arun(fn, *a):
        return fn(*a)
    monkeypatch.setattr(embed, "arun", arun)
    monkeypatch.setattr(answer, "Stream", FakeStream)
    monkeypatch.setattr(speculator.answer, "Stream", FakeStream)
    monkeypatch.setattr(config, "SEMANTIC_ENDPOINT", False)
    yield


# ----------------------------------------------------------------------------- helpers
async def speak(text, mode, gap=0.25, endpoint_delay=0.5, revisions=None, events=None):
    """Feed timed words (each visible 0.1 s after it ends) through a Session; return the metric."""
    events = events if events is not None else []
    t0 = time.perf_counter()
    s = Session(events.append, mode=mode, t0=t0)
    words = text.split()
    times = [(0.2 + i * gap, 0.2 + i * gap + gap * 0.9) for i in range(len(words))]
    end = times[-1][1]
    t = 0.0
    revisions = revisions or {}
    while t < end + endpoint_delay:
        while time.perf_counter() < t0 + t:
            await asyncio.sleep(0.002)
        heard = [dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times) if b <= t - 0.1]
        for i, (until, wrong) in revisions.items():
            if i < len(heard) and t < until:
                heard[i] = dict(heard[i], text=wrong)
        if heard:
            s.on_partial(heard)
        t += 0.1
    m = await s.on_final([dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times)])
    s.close()
    return m, events


def specs(events, **kw):
    return [e for e in events if e["t"] == "spec" and all(e.get(k) == v for k, v in kw.items())]


# ----------------------------------------------------------------------------- frame tests
@pytest.fixture(scope="module")
def tagger():
    t = Tagger(domain="assistant")
    yield t
    t.close()


def _frames(tagger, text):
    tagger.reset()
    ws, spans, out, pi = text.split(), [], [], None
    for i, w in enumerate(ws):
        tagger.push([w])
        r = tagger.poll()
        a0 = r.get("a0", 0)
        spans = [s for s in spans if s["end"] <= a0] + r.get("spans", [])
        f = frame.build(ws[: i + 1], spans, r.get("current"), prev_intent=pi, local_ok=False)
        if f.intent.startswith("discourse.intent."):
            pi = (f.intent, f.intent_p)
        out.append(f)
    return out


def test_weather_intent_early_and_slots(tagger):
    fs = _frames(tagger, "what's the weather in Paris tomorrow")
    assert fs[2].intent == "discourse.intent.weather_query"          # at the third word
    assert ("place", "paris") in fs[-1].slots
    assert "paris" in fs[-1].query() and "weather" in fs[-1].query()


def test_correction_replaces_entity(tagger):
    f = _frames(tagger, "who won the Lakers game last night sorry I mean the Warriors game")[-1]
    assert f.corrected
    assert "lakers" not in f.content and "warriors" in f.content
    assert not any(t == "lakers" for _, t in f.slots)


def test_correction_make_that(tagger):
    f = _frames(tagger, "what's the weather in Paris actually make that London tomorrow")[-1]
    assert "london" in f.content and "paris" not in f.content


def test_filler_does_not_change_signature(tagger):
    a = _frames(tagger, "how tall is the Eiffel Tower")[-1]
    b = _frames(tagger, "um how tall is the Eiffel Tower please")[-1]
    assert a.content == b.content


# ----------------------------------------------------------------------------- speculator tests
def test_headstart_fires_before_endpoint_and_reuses():
    m, ev = asyncio.run(speak("who founded the company Nvidia", "headstart"))
    endpoint_ms = next(e["ms"] for e in ev if e["t"] == "endpoint")
    fired = specs(ev, lane="web", state="fired")
    assert fired and fired[0]["ms"] < endpoint_ms
    assert m["path"] in ("draft-reuse", "speculative-hit")
    assert "nvidia" in m["query"]


def test_baseline_searches_only_after_endpoint():
    m, ev = asyncio.run(speak("who founded the company Nvidia", "baseline"))
    endpoint_ms = next(e["ms"] for e in ev if e["t"] == "endpoint")
    assert all(e["ms"] >= endpoint_ms - 5 for e in specs(ev, state="fired"))
    assert m["path"] == "search-after-endpoint" and m["web_calls"] == 1


def test_headstart_faster_than_baseline():
    hs, _ = asyncio.run(speak("who founded the company Nvidia", "headstart"))
    bl, _ = asyncio.run(speak("who founded the company Nvidia", "baseline"))
    assert hs["ready_ms"] < bl["ready_ms"] - 150, (hs["ready_ms"], bl["ready_ms"])


def test_revision_marks_stale_and_answer_uses_final_entity():
    # word 4 is misheard as "Acronis" until 2.0 s, then revised to "Nvidia"
    m, ev = asyncio.run(speak("who founded the company Nvidia", "headstart", revisions={4: (1.75, "Acronis")}))
    assert specs(ev, state="stale"), "the misheard query should go stale"
    assert "nvidia" in m["query"]
    srcs = m["sources"]
    assert srcs and all("Acronis" not in s for s in srcs[:3]), srcs


def test_budget_caps_web_calls():
    asyncio.run(speak("please tell me about the history of the city of Rome and the Roman Empire and Julius Caesar",
                      "headstart"))
    assert len(WEB_CALLS) <= config.WEB_BUDGET + 1


def test_semantic_endpoint_answers_before_ear(monkeypatch):
    monkeypatch.setattr(config, "SEMANTIC_ENDPOINT", True)
    monkeypatch.setattr(config, "SEM_STABLE_MS", 200)

    async def go():
        ev = []
        t0 = time.perf_counter()
        s = Session(ev.append, mode="headstart", t0=t0)
        words = "who founded the company Nvidia".split()
        times = [(0.2 + i * 0.25, 0.2 + i * 0.25 + 0.22) for i in range(len(words))]
        end = times[-1][1]
        t = 0.0
        while t < end + 1.0:
            while time.perf_counter() < t0 + t:
                await asyncio.sleep(0.002)
            loud = t < end
            s.on_audio((np.ones(1600, np.float32) * (0.1 if loud else 0.0001)), t + 0.1)
            heard = [dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times) if b <= t - 0.1]
            if heard:
                s.on_partial(heard)
            t += 0.1
        # the ear's own endpoint arrives late with the same words: it confirms the soft answer
        r = await s.on_final([dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times)])
        if s.final_task:
            await s.final_task
        s.close()
        return s, r, ev
    s, r, ev = asyncio.run(go())
    assert r is not None and r["endpoint_by"] == "semantic"
    assert len(s.history) == 1 and s.history[0]["endpoint_by"] == "semantic"
    assert sum(1 for e in ev if e["t"] == "endpoint") == 1


def test_soft_endpoint_extended_reuses_warm_speculator(monkeypatch):
    """Pause after "...Paris", semantic endpoint answers; speaker adds "tomorrow"; the ear final answers the longer
    question from the SAME speculator (its search for the longer question already fired while they spoke)."""
    monkeypatch.setattr(config, "SEMANTIC_ENDPOINT", True)
    monkeypatch.setattr(config, "SEM_STABLE_MS", 200)

    async def go():
        ev = []
        t0 = time.perf_counter()
        s = Session(ev.append, mode="headstart", t0=t0)
        words = "what's the weather in Paris tomorrow".split()
        times = [(0.2 + i * 0.25, 0.2 + i * 0.25 + 0.22) for i in range(5)] + [(2.6, 2.85)]   # pause before "tomorrow"
        t = 0.0
        while t < 3.6:
            while time.perf_counter() < t0 + t:
                await asyncio.sleep(0.002)
            loud = any(a <= t <= b + 0.05 for a, b in times)
            s.on_audio(np.ones(1600, np.float32) * (0.1 if loud else 0.0001), t + 0.1)
            heard = [dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times) if b <= t - 0.1]
            if heard:
                s.on_partial(heard)
            t += 0.1
        r = await s.on_final([dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times)])
        s.close()
        return s, r, ev
    s, r, ev = asyncio.run(go())
    eps = [e for e in ev if e["t"] == "endpoint"]
    assert eps[0]["by"] == "semantic" and len(eps) >= 2, [e["by"] for e in eps]
    assert any(e["t"] == "extended" for e in ev)
    # either a second soft endpoint answered the longer question (confirmed by the ear) or the ear answered it
    assert len(s.history) == 1 and "tomorrow" in r["query"]
    assert r["path"] != "speculative-miss", r["path"]      # the warm speculator had already searched it



def test_late_revision_in_final_marks_stale():
    m, ev = asyncio.run(speak("who founded the company Nvidia", "headstart", revisions={4: (5.0, "Acronis")}))
    assert specs(ev, state="stale") and "nvidia" in m["query"]
    assert m["path"] == "speculative-miss" and all("Acronis" not in s for s in m["sources"][:3])


def test_conversation_memory_resolves_pronouns(tagger):
    carry = {"person": [("person", "albert einstein")], "thing": [("organization", "nvidia")]}
    def last(text):
        tagger.reset()
        ws = text.split()
        tagger.push(ws)
        r = tagger.poll()
        return frame.build(ws, r.get("spans", []), r.get("current"), local_ok=False, carry=carry)
    f = last("when was it founded")
    assert f.carried == "nvidia" and "nvidia" in f.query() and "(nvidia)" in f.resolved_text
    assert last("where was he born").carried == "albert einstein"
    g = last("what is the stock price of Apple")
    assert g.carried == "" and "apple" in g.query()          # has its own entity: no carry
    h = last("is there a good pizza place in Boston")
    assert h.carried == ""                                    # weak pronoun in a full question: no carry
