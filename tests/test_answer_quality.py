"""Answer-quality fixes from the A/B error analysis (runs/error_analysis.md): query fillers, corrections whose cue the
tagger mis-tags or whose replacement it doesn't tag, the corrected question for the answer model, the entity-only
web query, and the relaxed retry when the content query finds nothing."""
import asyncio
import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from server import answer, frame, lanes
from server.tagger import Tagger


def span(typ, start, end, words):
    return {"family": typ.split(".")[0], "type": typ, "start": start, "end": end, "text": " ".join(words[start:end])}


def build(text, spans_=()):
    ws = text.split()
    return frame.build(ws, [span(t, a, b, ws) for t, a, b in spans_], None, local_ok=False)


# ----------------------------------------------------------------------------- deterministic (hand-made spans)
def test_fillers_and_contractions_do_not_leak_into_the_query():
    # full3 q21: "i've wanted many bones adult human body" found Embalming / Jennifer's Body
    f = build("I've always wanted to know how many bones are there in the adult human body")
    assert f.query() == "bones adult human body"
    assert build("Mm, can you tell me who invented the telephone").query() == "invented telephone"   # snr20 q18
    assert build("And I didn't know, why is the sky blue").query() == "sky blue"                     # rate_slow q25
    assert build("I’ve always wanted to know").query() == ""                                            # curly apostrophe


def test_cue_tagged_as_entity_does_not_absorb_the_correction():
    # full3 q31: the tagger tags "Sari," (= "sorry") as a place, so it was "corrected away" and Paris stayed
    text = "What river runs through Paris, Sari, I mean London."
    f = build(text, [("entity.mention.place", 4, 5), ("entity.mention.place", 5, 6), ("entity.mention.place", 8, 9)])
    assert "paris" not in f.query() and "london" in f.query()
    assert ("place", "paris") not in f.slots and ("place", "sari") not in f.slots
    assert f.clean_text == "What river runs through London."


def test_untagged_replacement_drops_the_entity_not_the_verb():
    # full3 q30: "Beethoven" untagged → the fallback dropped "born" (the word before the cue) and kept Mozart
    f = build("Where was Mozart born actually make that Beethoven?", [("entity.mention.person", 2, 3)])
    assert f.corrected
    assert "mozart" not in f.content and "born" in f.content and "beethoven" in f.query()
    assert "Mozart" not in f.clean_text and "make that" not in f.clean_text


def test_word_correction_without_entities_still_replaces_the_word_before():
    f = build("who won the Lakers game sorry Warriors game")
    assert "lakers" not in f.content and "warriors" in f.content


def test_resolved_text_is_the_corrected_question():
    # full3 q29: the answer model saw "who founded Microsoft? sorry I met Amazon" and opened with "The sources don't
    # cover Microsoft" (an ungrounded-looking answer); it now sees the question the speaker meant
    f = build("Who founded Microsoft? Sorry. I met Amazon.",
              [("entity.mention.organization", 2, 3), ("entity.mention.organization", 6, 7)])
    assert f.resolved_text == "Who founded Amazon."
    assert build("who founded the company Nvidia").resolved_text == "who founded the company Nvidia"


def test_entity_query():
    f = build("Who was the first president of France, no weight of the United States?",
              [("entity.mention.organization", 6, 7), ("entity.mention.place", 10, 13)])
    assert f.entity_query() == "united states"       # "the" trimmed, inner words kept
    # replacement after a correction that the tagger didn't tag: the words after the cue
    g = build("who wrote Hamlet, sorry, I mean pride and prejudice", [("entity.mention.person", 2, 3)])
    assert g.entity_query() == "pride and prejudice"
    assert build("what is the boiling point of water").entity_query() == ""          # no entity → no extra call
    h = build("Nvidia", [("entity.mention.organization", 0, 1)])
    assert h.entity_query() == ""                                                       # same as the query


def test_carried_entity_is_the_entity_query():
    carry = {"thing": [("organization", "nvidia")]}
    f = frame.build("when was it founded".split(), [], None, local_ok=False, carry=carry)
    assert f.entity_query() == "nvidia"


def test_prompt_rules():
    assert "no disclaimer" in answer.SYSTEM and "corrected-away" in answer.SYSTEM


# ----------------------------------------------------------------------------- with the native tagger
@pytest.fixture(scope="module")
def tagger():
    t = Tagger(domain="assistant")
    yield t
    t.close()


def _last(tagger, text):
    tagger.reset()
    ws, spans = text.split(), []
    f = None
    for i, w in enumerate(ws):
        tagger.push([w])
        r = tagger.poll()
        a0 = r.get("a0", 0)
        spans = [s for s in spans if s["end"] <= a0] + r.get("spans", [])
        f = frame.build(ws[: i + 1], spans, r.get("current"), local_ok=False)
    return f


@pytest.mark.parametrize("text,has,hasnt", [
    ("What river runs through Paris, Sari, I mean London.", "london", "paris"),
    ("Where was Mozart born actually make that Beethoven?", "beethoven", "mozart"),
    ("Who was the first president of France no way of United States?", "united", "france"),
    ("Look current he does Japan use. Scratch that switzerland.", "switzerland", "japan"),
])
def test_corrections_as_heard_in_the_ab_runs(tagger, text, has, hasnt):
    f = _last(tagger, text)
    assert has in f.query() and hasnt not in f.query(), f.query()


# ----------------------------------------------------------------------------- Wikipedia lane (mock transport)
def _wiki_transport(hits: dict, log: list):
    def handler(req: httpx.Request):
        q = parse_qs(urlparse(str(req.url)).query)["gsrsearch"][0]
        log.append(q)
        pages = [{"pageid": i, "index": i + 1, "title": t, "fullurl": f"https://en.wikipedia.org/wiki/{t}",
                  "extract": f"{t} intro."} for i, t in enumerate(hits.get(q, []))]
        return httpx.Response(200, json={"query": {"pages": pages}} if pages else {"batchcomplete": True})
    return httpx.MockTransport(handler)


def _run_wiki(hits, query, entity=""):
    log = []
    lanes.use_transport(_wiki_transport(hits, log))
    try:
        docs = asyncio.run(lanes.wikipedia(query, entity=entity))
    finally:
        lanes.use_transport(None)
    return [d.title for d in docs], log


def test_entity_query_articles_are_appended_without_duplicates():
    titles, log = _run_wiki({"capital turkey": ["Capital punishment in Turkey", "Outline of Turkey"],
                             "turkey": ["Turkey", "Outline of Turkey"]}, "capital turkey", entity="turkey")
    assert titles == ["Capital punishment in Turkey", "Outline of Turkey", "Turkey"]
    assert sorted(log) == ["capital turkey", "turkey"]


def test_empty_content_query_is_retried_relaxed():
    titles, log = _run_wiki({"mona lysa": ["Mona Lisa"]}, "repainted mona lysa")
    assert titles == ["Mona Lisa"] and log == ["repainted mona lysa", "mona lysa"]
    titles, log = _run_wiki({}, "two words")
    assert titles == [] and log == ["two words"]                 # too short to relax


def test_web_lane_passes_the_entity(monkeypatch):
    seen = {}

    async def fake_wiki(query, k=4, entity="", k_entity=2):
        seen.update(query=query, entity=entity)
        return []
    monkeypatch.setattr(lanes, "wikipedia", fake_wiki)
    monkeypatch.setattr(lanes.config, "WEB_BACKEND", "wikipedia")
    asyncio.run(lanes.web("capital turkey", entity="turkey"))
    assert seen == {"query": "capital turkey", "entity": "turkey"}
