"""Search narrowing from the Arc read-out (server/narrow.py): family, include_domains in the Tavily body and cache key,
query shaping (generic words, family noun, explicit dates), the speculative-credit gate, the "actually mean" frame fix,
and the speculator sending it end to end over the fake Tavily."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "eval"))

from server import config, lanes, narrow  # noqa: E402
from server.frame import Frame  # noqa: E402
from test_tavily import fake, run  # noqa: E402,F401
from test_speculation import speak  # noqa: E402

W = "discourse.intent.weather_query"


@pytest.fixture(autouse=True)
def today(monkeypatch):
    monkeypatch.setattr(config, "NARROW_TODAY", "2026-11-12")      # a Thursday


def fr(content, intent="", words=None, **kw):
    return Frame(words=(words or " ".join(content)).split(), intent=intent, intent_p=0.9 if intent else 0.0,
                 content=tuple(content), **kw)


def test_family_from_readout_and_words():
    assert narrow.family(fr(["weather", "chicago", "tomorrow"], W)) == "weather"
    assert narrow.family(fr(["bad", "take", "ibuprofen", "empty", "stomach"], W)) == "health"    # Arc said weather
    assert narrow.family(fr(["nvidia", "stock", "price"], "discourse.intent.qa_stock")) == "finance"
    assert narrow.family(fr(["score", "celtics", "game"])) == "sports"
    assert narrow.family(fr(["first", "woman", "win", "nobel", "prize"], "discourse.intent.qa_factoid")) == "reference"
    assert narrow.family(fr(["best", "robot", "vacuum", "pet", "hair"])) == "reviews"
    assert narrow.family(fr(["visa", "visit", "japan"])) == "travel"
    assert narrow.family(fr(["time", "louvre", "paris", "close", "today"], "discourse.intent.datetime_query")) == ""


def test_domains_only_for_selected_families(monkeypatch):
    monkeypatch.setattr(config, "NARROW_DOMAINS", "prefer")
    monkeypatch.setattr(config, "NARROW_DOMAIN_FAMILIES", "weather,finance")
    d, m = narrow.domains(fr(["weather", "denver"], W))
    assert "weather.gov" in d and m == "prefer"
    assert narrow.domains(fr(["score", "celtics", "game"])) == ([], "")
    monkeypatch.setattr(config, "NARROW_DOMAINS", "restrict")
    monkeypatch.setattr(config, "NARROW_DOMAIN_FAMILIES", "")
    assert narrow.domains(fr(["founded", "nvidia"], "discourse.intent.qa_factoid"))[1] == "prefer"   # never restricted
    assert narrow.domains(fr(["weather", "denver"], W))[1] == "restrict"
    monkeypatch.setattr(config, "NARROW_DOMAINS", "off")
    assert narrow.domains(fr(["weather", "denver"], W)) == ([], "")


def test_body_and_cache_key_carry_domains(fake):
    plan = lanes.web_plan(W, "weather denver", hints={"include_domains": ["weather.gov"], "domains_mode": "prefer"})
    b = lanes.tavily_body("weather denver", plan)
    assert b["include_domains"] == ["weather.gov"] and b["include_domains_mode"] == "prefer"
    assert fake.validate(b) is None
    plain = lanes.tavily_body("weather denver", lanes.web_plan(W, "weather denver"))
    assert "include_domains" not in plain and "include_domains_mode" not in plain and fake.validate(plain) is None
    assert lanes._cache_key("weather denver", b) != lanes._cache_key("weather denver", plain)


def test_shape_query(monkeypatch):
    monkeypatch.setattr(config, "NARROW_QUERY", True)
    f = fr(["weather", "gonna", "chicago", "tomorrow"], W)
    assert narrow.shape_query(f, "weather gonna chicago tomorrow") == "weather chicago November 13 2026 forecast"
    g = fr(["won", "lakers", "game", "last", "night"], words="who won the lakers game last night")
    assert narrow.shape_query(g, "won lakers game last night") == "won lakers game November 11 2026"
    h = fr(["kansas", "city", "chiefs", "win", "game", "sunday"], words="did the chiefs win their game on sunday")
    assert "November 8 2026" in narrow.shape_query(h, "kansas city chiefs win game sunday")
    i = fr(["bad", "take", "ibuprofen"], W)        # Arc's weather hint on a health question
    assert narrow.shape_query(i, "bad take ibuprofen weather forecast") == "bad take ibuprofen"
    monkeypatch.setattr(config, "NARROW_QUERY", False)
    assert narrow.shape_query(f, "weather gonna chicago tomorrow") == "weather gonna chicago tomorrow"


def test_gate_skips_lone_entity_prefix(monkeypatch):
    monkeypatch.setattr(config, "NARROW_GATE", True)
    assert not narrow.specific(fr(["kansas"], slots=(("organization", "kansas"),)))
    assert narrow.specific(fr(["kansas", "city"]))
    monkeypatch.setattr(config, "NARROW_GATE", False)
    assert narrow.specific(fr(["kansas"]))


def test_rerank_prefers_entity_and_domain(monkeypatch):
    monkeypatch.setattr(config, "NARROW_RERANK", True)
    f = fr(["weather", "denver"], W, phrases=("denver",))
    docs = [lanes.Doc(id="a", title="Boston forecast", text="rain in boston", url="https://x.com/a"),
            lanes.Doc(id="b", title="Denver weather", text="denver forecast sunny", url="https://weather.gov/den")]
    assert [d.id for d in narrow.rerank(f, docs)] == ["b", "a"]


def test_actually_mean_is_not_a_correction():
    import replay_frames  # noqa: F401  (eval helper: native tagger, word by word)
    f = replay_frames.frame_for("Well, what does the word ubiquitous actually mean exactly?")
    assert "ubiquitous" in f.query()
    g = replay_frames.frame_for("What's the capital of Canada actually make that turkey?")
    assert "turkey" in g.query() and "canada" not in g.query()


def test_speculator_sends_narrowed_request(fake, monkeypatch):
    monkeypatch.setattr(config, "NARROW_DOMAINS", "prefer")
    monkeypatch.setattr(config, "NARROW_DOMAIN_FAMILIES", "weather")
    monkeypatch.setattr(config, "NARROW_QUERY", True)
    m, ev = run(speak("what's the weather in Paris tomorrow", "headstart"))
    bodies = [b for b in fake.requests if "paris" in b["query"]]
    assert bodies and all(b.get("include_domains_mode") == "prefer" and "weather.gov" in b["include_domains"]
                          for b in bodies)
    assert any("November 13 2026" in b["query"] for b in bodies)
    done = [e for e in ev if e["t"] == "spec" and e.get("state") == "done" and e.get("lane") == "web"]
    assert done and done[0].get("narrow") == "weather"
