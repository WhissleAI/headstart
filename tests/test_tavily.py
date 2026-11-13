"""Tavily web lane against a local fake (tests/fake_tavily.py): request params per intent, credits, cache, budget,
errors, the include_answer race, and Head Start vs baseline with a slow (600-1500 ms) web lane.

Run: .venv/bin/python -m pytest -q tests
"""
import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from server import answer, config, embed, lanes, speculator  # noqa: E402
from fake_tavily import FakeTavily  # noqa: E402
from test_speculation import FakeStream, _vec, speak  # noqa: E402  (helpers only, not its autouse fixture)

NEWS, STOCK, FX = "discourse.intent.news_query", "discourse.intent.qa_stock", "discourse.intent.qa_currency"
WEATHER, FACT = "discourse.intent.weather_query", "discourse.intent.qa_factoid"


@pytest.fixture
def fake(monkeypatch):
    """Real lanes.web → Tavily code path, over an in-process fake; fake embedder + answer model (no network)."""
    fk = FakeTavily(seed=7, scale=0.2)
    lanes.use_transport(fk.transport())
    lanes.reset_process()
    monkeypatch.setattr(config, "WEB_BACKEND", "tavily")
    monkeypatch.setattr(config, "TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr(config, "TAVILY_CACHE_SCOPE", "session")
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
    FakeStream.instances.clear()
    yield fk
    lanes.use_transport(None)
    lanes.reset_process()


def run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------------------- parameters
def test_plan_and_body_per_intent(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_SPEC_DEPTH", "fast")
    monkeypatch.setattr(config, "TAVILY_FINAL_DEPTH", "fast")
    b = lanes.tavily_body("latest news nvidia today", lanes.web_plan(NEWS, "what is the latest news about Nvidia today"))
    assert b["topic"] == "news" and b["time_range"] == "day" and "max_age_hours" not in b
    assert lanes.web_plan(NEWS, "what's the news on the election this week")["time_range"] == "week"
    assert "time_range" not in lanes.web_plan(NEWS, "any news about SpaceX")
    s = lanes.web_plan(STOCK, "what is the stock price of Apple")
    assert s["topic"] == "finance" and s["max_age_hours"] == config.TAVILY_FINANCE_MAX_AGE_H
    assert lanes.web_plan(FX, "exchange rate dollars to euros")["topic"] == "finance"
    w = lanes.web_plan(WEATHER, "what's the weather in Paris tomorrow")
    assert w["topic"] == "general" and w["max_age_hours"] == config.TAVILY_WEATHER_MAX_AGE_H and "time_range" not in w
    g = lanes.tavily_body("founded nvidia", lanes.web_plan(FACT, "who founded Nvidia today"))
    assert g["topic"] == "general" and "time_range" not in g        # freshness only for news (dates drop pages)
    # always explicit: no surprise 2-credit advanced via auto_parameters; usage for credit accounting
    assert g["auto_parameters"] is False and g["include_usage"] is True and g["include_answer"] is False
    assert g["search_depth"] == "fast" and g["chunks_per_source"] == config.TAVILY_CHUNKS
    assert g["max_results"] == config.TAVILY_MAX_RESULTS
    monkeypatch.setattr(config, "TAVILY_SPEC_DEPTH", "ultra-fast")
    u = lanes.tavily_body("x y", lanes.web_plan(FACT, "x y"))
    assert u["search_depth"] == "ultra-fast" and "chunks_per_source" not in u      # not applicable to ultra-fast
    assert lanes.tavily_body("q " * 500, lanes.web_plan())["query"].__len__() <= 400


def test_fake_accepts_every_body_we_send(fake):
    """The fake validates enums/ranges like the API: every plan we can build must pass."""
    for intent, text in [(NEWS, "news today"), (NEWS, "latest news"), (STOCK, "apple stock"), (FX, "usd eur"),
                         (WEATHER, "weather paris"), (FACT, "eiffel tower height")]:
        for final in (False, True):
            for ans in (False, "basic"):
                assert fake.validate(lanes.tavily_body(text, lanes.web_plan(intent, text, final, ans))) is None


def test_session_sends_intent_params(fake):
    """End to end through Session + tagger: weather → general + max_age_hours, news+today → news/day,
    stock → finance; speculative fires and baseline's final both go to Tavily with the configured depth."""
    m, ev = run(speak("what's the weather in Paris tomorrow", "headstart"))
    bodies = [b for b in fake.requests if "paris" in b["query"]]
    assert bodies and all(b["topic"] == "general" and b.get("max_age_hours") == 6 for b in bodies)
    assert all(b["search_depth"] == config.TAVILY_SPEC_DEPTH for b in bodies)
    fake.requests.clear()
    run(speak("what is the latest news about Nvidia today", "baseline"))
    (b,) = fake.requests
    assert b["topic"] == "news" and b["time_range"] == "day" and b["search_depth"] == config.TAVILY_FINAL_DEPTH
    fake.requests.clear()
    run(speak("what is the stock price of Apple", "headstart"))
    assert fake.requests and all(b["topic"] == "finance" for b in fake.requests)
    fired = [e for e in ev if e["t"] == "spec" and e.get("state") == "done" and e.get("lane") == "web"]
    assert fired and fired[0]["backend"] == "tavily" and fired[0]["credits"] == 1


# ----------------------------------------------------------------------------- credits + cache
def test_credits_in_metric_and_usage_reconciled(fake, monkeypatch):
    hs, _ = run(speak("who founded the company Nvidia", "headstart"))
    assert hs["web_backend"] == "tavily"
    assert hs["credits"] == hs["credits_spec"] + hs["credits_final"] >= 1
    assert hs["credits_spec"] == sum(1 for b in fake.requests)          # fast = 1 credit each
    bl, _ = run(speak("who founded the company Nvidia", "baseline"))
    assert bl["credits"] == bl["credits_final"] == 1 and bl["credits_spec"] == 0
    # usage.credits from the response wins over our estimate (advanced = 2)
    monkeypatch.setattr(config, "TAVILY_FINAL_DEPTH", "advanced")
    led = lanes.Ledger()
    meter = {}
    run(lanes.web("eiffel tower height", final=True, ledger=led, meter=meter))
    assert meter["credits"] == 2 and led.spent == 2


def test_cache_same_normalized_query(fake):
    async def go():
        led = lanes.Ledger()
        m1, m2, m3 = {}, {}, {}
        a = await lanes.web("Nvidia founded", intent=FACT, ledger=led, meter=m1)
        b = await lanes.web("founded,  nvidia", intent=FACT, ledger=led, meter=m2)
        c = await lanes.web("founded nvidia", intent=FACT, ledger=lanes.Ledger(), meter=m3)   # another session
        return led, a, b, m1, m2, m3
    led, a, b, m1, m2, m3 = run(go())
    assert len(fake.requests) == 2                         # second call of session 1 was a cache hit
    assert m1["credits"] == 1 and m2["cached"] and m2["credits"] == 0 and not m3["cached"]
    assert [d.id for d in a] == [d.id for d in b] and a[0] is not b[0]    # copies: lanes mutate d.lane
    assert led.cache_hits == 1 and led.spent == 1


def test_cache_global_scope_and_inflight_dedup(fake, monkeypatch):
    monkeypatch.setattr(config, "TAVILY_CACHE_SCOPE", "global")

    async def go():
        l1, l2 = lanes.Ledger(), lanes.Ledger()
        ms = [{}, {}, {}]
        await asyncio.gather(lanes.web("apollo 11 landing", ledger=l1, meter=ms[0]),
                             lanes.web("apollo 11 landing", ledger=l1, meter=ms[1]))      # concurrent: joined
        await lanes.web("landing apollo 11", ledger=l2, meter=ms[2])                      # global cache
        return ms
    ms = run(go())
    assert len(fake.requests) == 1 and sum(m["credits"] for m in ms) == 1


def test_cancelled_speculative_call_still_fills_cache(fake):
    """A stale fire is cancelled mid-flight, but the request (already paid) finishes and lands in the cache, so a
    flip-back or the final query costs nothing."""
    async def go():
        led = lanes.Ledger()
        t = asyncio.create_task(lanes.web("lakers game score", ledger=led, meter={}))
        await asyncio.sleep(0.01)
        t.cancel()
        await asyncio.sleep(0.6)
        m = {}
        await lanes.web("lakers game score", final=True, ledger=led, meter=m)
        return led, m
    led, m = run(go())
    assert len(fake.requests) == 1 and m["cached"] and led.spent == 1


# ----------------------------------------------------------------------------- budget guard
def test_budget_reserve_and_fallback(fake):
    async def go():
        led = lanes.Ledger(budget=3, reserve=1)
        await lanes.web("q one", ledger=led, meter={})
        await lanes.web("q two", ledger=led, meter={})
        with pytest.raises(lanes.WebUnavailable):
            await lanes.web("q three", ledger=led, meter={})            # speculation can't touch the reserve
        mf = {}
        await lanes.web("q four", final=True, ledger=led, meter=mf)    # the final query can
        mw = {}
        docs = await lanes.web("q five", final=True, ledger=led, meter=mw)
        return led, mf, mw, docs
    led, mf, mw, docs = run(go())
    assert led.spent == 3 and led.denied == 2
    assert mf["backend"] == "tavily" and mw["backend"] == "wikipedia" and "budget" in mw["fallback"]
    assert docs and docs[0].source == "wikipedia"


def test_speculator_stops_speculating_at_budget(fake, monkeypatch):
    monkeypatch.setattr(config, "TAVILY_SESSION_CREDITS", 2)
    monkeypatch.setattr(config, "TAVILY_FINAL_RESERVE", 1)
    m, ev = run(speak("please tell me about the history of the city of Rome and the Roman Empire", "headstart"))
    assert sum(1 for e in ev if e["t"] == "spec" and e.get("lane") == "web" and e.get("state") == "fired") == 1
    assert any(e.get("id") == "budget" and e.get("state") == "skipped" for e in ev)
    assert m["session_credits"] <= 2 and m.get("answer")


def test_process_budget(fake, monkeypatch):
    monkeypatch.setattr(config, "TAVILY_PROCESS_CREDITS", 1)
    async def go():
        await lanes.web("a b", ledger=lanes.Ledger(), meter={})
        with pytest.raises(lanes.WebUnavailable):
            await lanes.web("c d", ledger=lanes.Ledger(), meter={})
    run(go())


# ----------------------------------------------------------------------------- errors
@pytest.mark.parametrize("status,blocks", [(429, True), (432, True), (433, True), (500, False), (400, False)])
def test_errors_are_lane_errors_not_crashes(fake, status, blocks):
    fake.fail = [(status, {"Retry-After": "30"} if status == 429 else {})]
    m, ev = run(speak("who founded the company Nvidia", "headstart"))
    errs = [e for e in ev if e["t"] == "spec" and e.get("state") == "error" and e.get("lane") == "web"]
    assert errs and f"tavily {status}" in errs[0]["error"]
    assert m.get("answer") and m.get("error") is None                  # the session still answered
    assert errs[0]["credits"] == 0                                     # failed calls aren't billed
    blocked = lanes._PROCESS["blocked_until"] > 0
    assert blocked == blocks
    if blocks:
        # while blocked: no more Tavily calls; the final query falls back to Wikipedia
        n = len(fake.requests)
        bl, _ = run(speak("how tall is the Eiffel Tower", "baseline"))
        assert len(fake.requests) == n and fake.wiki_requests and bl["web_backend"] == "wikipedia"
        assert bl["web_fallbacks"] == 1 and bl["credits"] == 0 and bl.get("answer")


def test_final_error_falls_back_to_wikipedia(fake):
    fake.fail = [(500, {})]
    bl, ev = run(speak("who founded the company Nvidia", "baseline"))
    assert bl["web_backend"] == "wikipedia" and bl["credits"] == 0 and bl["sources"]
    assert fake.wiki_requests


def test_434_retries_without_max_age(fake):
    fake.fail = [lambda b: (434, {}) if "max_age_hours" in b else None]
    async def go():
        m = {}
        docs = await lanes.web("weather paris", intent=WEATHER, ledger=lanes.Ledger(), meter=m)
        return docs, m
    docs, m = run(go())
    assert docs and len(fake.requests) == 2 and "max_age_hours" not in fake.requests[1] and m["credits"] == 1


def test_timeout_is_a_lane_error(fake, monkeypatch):
    monkeypatch.setattr(config, "TAVILY_TIMEOUT_S", 0.05)
    fake.latency_ms, fake.scale = (400, 500), 1.0

    async def go():
        with pytest.raises(Exception):
            await lanes.web("slow query", ledger=lanes.Ledger(), meter={})
        m = {}
        await lanes.web("slow query", final=True, ledger=lanes.Ledger(), meter=m)
        return m
    m = run(go())
    assert m["backend"] == "wikipedia"


# ----------------------------------------------------------------------------- speed under a slow web lane
def test_headstart_beats_baseline_with_slow_tavily(fake):
    """600-1500 ms Tavily (uniform; paired: the same query takes the same time in both modes), real request path.

    With a lane this slow the final query (it needs the last content word) can't be fully hidden; Head Start still
    wins by firing it as soon as the word is visible instead of after the endpoint (sweep, 3 seeds x 6 questions:
    18/18 faster, ~300 ms saved, ~2.8 credits/question vs 1)."""
    fake.latency_ms, fake.scale = (600, 1500), 1.0
    for q in ("can you tell me who founded the technology company Nvidia", "when did Apollo 11 land on the moon"):
        hs, _ = run(speak(q, "headstart", gap=0.3))
        bl, _ = run(speak(q, "baseline", gap=0.3))
        assert bl["ready_ms"] >= 600 + 300 - 50                     # search + fake TTFT
        assert hs["ready_ms"] < bl["ready_ms"] - 200, (q, hs["ready_ms"], bl["ready_ms"], hs["path"])
    assert hs["credits"] <= config.WEB_BUDGET + 1 and bl["credits"] == 1


# ----------------------------------------------------------------------------- include_answer race
@pytest.mark.parametrize("answer_ms,winner", [((50, 60), "tavily"), ((900, 1000), "llm")])
def test_include_answer_race(fake, monkeypatch, answer_ms, winner):
    """Tavily's answer arrives with its (slower) search response; Haiku starts after our own (faster) search.
    The race picks whichever has text first. FakeStream TTFT = 300 ms."""
    monkeypatch.setattr(config, "TAVILY_ANSWER", True)
    fake.latency_ms, fake.answer_ms, fake.scale = (300, 320), answer_ms, 1.0
    bl, _ = run(speak("who founded the company Nvidia", "baseline"))
    assert bl["answer_by"] == winner, bl
    assert bl["credits_final"] == 2                                  # results call + answer call
    bodies = sorted(fake.requests, key=lambda b: bool(b["include_answer"]))
    assert bodies[0]["include_answer"] is False and bodies[1]["include_answer"] == config.TAVILY_ANSWER_DEPTH
    if winner == "tavily":
        assert "short answer" in bl["answer"]
