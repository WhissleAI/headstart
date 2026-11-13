"""Local moments: second-stage ranking (server/rerank.py), its use in the speculator for local-routed questions, and
moment citations in the answer prompt. Fake vectors and answer model; no network, no model.

Run: .venv/bin/python -m pytest -q tests/test_local_rerank.py
"""
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from server import answer, config, embed, frame, lanes, rerank, speculator  # noqa: E402
from test_speculation import FakeStream, _vec  # noqa: E402


def moment(i, text, t, role="", tags=(), label="NBA broadcast"):
    return lanes.Doc(id=f"m-{i}", title=f"{label} · {int(t // 60):02d}:{int(t % 60):02d}", text=text, source="local",
                     t=t, meta={"role": role, "tags": list(tags), "source_label": label,
                                "context_before": "before words", "context_after": "after words"})


DOCS = [
    moment(0, "Andre Hunter nineteen triples this year, a career best nineteen points a game", 37, "analyst",
           ["Andre Hunter"]),
    moment(1, "Hunter gets a good look and nails it", 26, "play_by_play", ["Hunter", "sports.score.recent"]),
    moment(2, "Ponto ball nails that triple in Monday night's game", 70, "play_by_play", ["Lonzo Ball"]),
    moment(3, "Your objection is noted, we proceed", 9, "judge", ["legal.objection.recent"], label="Court hearing"),
    moment(4, "The officers waited 15 seconds instead of 3 to 4", 18, "judge", ["15 seconds"], label="Court hearing"),
]


def make_index(docs=DOCS, audio=None):
    ix = lanes.LocalIndex.__new__(lanes.LocalIndex)
    ix.docs = list(docs)
    ix.tv = np.stack([_vec(d.title + " " + d.text) for d in docs]).astype(np.float32)
    ix.av, ix.av_rows = (None, None) if audio is None else (audio, np.arange(len(docs)))
    return ix


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    FakeStream.instances.clear()
    monkeypatch.setattr(lanes, "_index", make_index())
    monkeypatch.setattr(embed, "encode_queries", lambda ts: np.stack([_vec(t) for t in ts]))
    monkeypatch.setattr(embed, "encode_docs", lambda ts, titles=None: np.stack([_vec(t) for t in ts]))

    async def arun(fn, *a):
        return fn(*a)
    monkeypatch.setattr(embed, "arun", arun)
    monkeypatch.setattr(answer, "Stream", FakeStream)
    monkeypatch.setattr(speculator.answer, "Stream", FakeStream)
    monkeypatch.setattr(config, "RERANK_LOCAL", True)
    monkeypatch.setattr(config, "RERANK_AUDIO", True)
    rerank._lex_cache.clear()
    yield


# ----------------------------------------------------------------------------- rerank units
def test_query_terms_strip_cue_and_normalize():
    assert rerank.strip_cues("What did the judge say about the objection in the hearing?") == \
        "what did say about the objection"
    assert rerank.query_terms("When did the announcer call Hunter's three in the game?") == ["hunter", "3"]
    assert rerank.query_terms("officers waiting fifteen seconds") == ["officer", "wait", "15", "second"]
    assert rerank.query_terms("meet halfway at 6%") == ["meet", "halfway", "6"]


def test_cues_roles_and_events():
    roles, events = rerank.cues("What did the judge say about the objection?")
    assert roles == {"judge"} and events == {"legal.objection"}
    assert rerank.cues("what did the announcer say")[0] == {"play_by_play"}


def test_lexicon_matches_tags_through_asr_noise():
    lx = rerank.Lexicon(DOCS)
    terms = rerank.query_terms("What did the commentator say about Lonzo Ball in the game?")
    s = {d.id: lx.score(d.id, terms, set(), set()) for d in DOCS}
    assert max(s, key=s.get) == "m-2"          # "Lonzo" only exists in the tag (the ASR heard "Ponto")
    assert s["m-2"] == pytest.approx(1.0)


def test_lexicon_role_and_event_bonus():
    lx = rerank.Lexicon(DOCS)
    q = "What did the judge say about the objection in the hearing?"
    terms, (roles, events) = rerank.query_terms(q), rerank.cues(q)
    assert lx.score("m-3", terms, roles, events) == pytest.approx(1.0 + 2 * rerank.ROLE_BONUS)
    assert lx.score("m-4", terms, roles, events) == pytest.approx(rerank.ROLE_BONUS)


def test_fuse_lexical_and_audio_lift_the_gold():
    ids = [d.id for d in DOCS]
    q = "What did the commentator say about Lonzo Ball in the game?"
    text_s = np.array([0.80, 0.79, 0.78, 0.10, 0.10])          # cosine prefers m-0 (the compressed-score failure)
    assert rerank.fuse(ids, text_s, q, rerank.Lexicon(DOCS), w_lex=0.0)[0][0] == "m-0"
    assert rerank.fuse(ids, text_s, q, rerank.Lexicon(DOCS))[0][0] == "m-2"
    # audio alone (no lexical) can also break a near-tie
    q2 = "what did they say"
    audio_s = np.array([0.3, 0.9, 0.4, 0.1, 0.1])
    tied = np.array([0.80, 0.79, 0.5, 0.1, 0.1])
    assert rerank.fuse(ids, tied, q2, rerank.Lexicon(DOCS), audio_s=audio_s)[0][0] == "m-1"
    assert rerank.fuse(ids, tied, q2, rerank.Lexicon(DOCS))[0][0] == "m-0"
    # ...but a clear text winner is not overruled by a full lexical match elsewhere
    clear = np.array([0.90, 0.70, 0.75, 0.10, 0.10])
    assert rerank.fuse(ids, clear, q, rerank.Lexicon(DOCS))[0][0] == "m-0"


def test_fuse_is_fast_on_a_realistic_index():
    rng = np.random.default_rng(0)
    docs = [moment(i, " ".join(rng.choice(["alpha", "beta", "gamma", "delta", "hunter", "ball"], 20)), i, "", ["x y"])
            for i in range(300)]
    lx = rerank.Lexicon(docs)
    ids = [d.id for d in docs]
    t = time.perf_counter()
    for _ in range(10):
        rerank.fuse(ids, rng.random(300), "what did the announcer say about hunter in the game", lx,
                    audio_s=rng.random(300))
    assert (time.perf_counter() - t) / 10 < 0.03             # the <30 ms endpoint budget, with a wide margin


# ----------------------------------------------------------------------------- speculator integration
def local_frame(text):
    toks = [frame.norm(w) for w in text.split()]
    content = tuple(dict.fromkeys(t for t in toks if t and t not in frame.STOP))
    return frame.Frame(words=text.split(), content=content, route="local")


async def _final(mode, text, audio_qvec=None):
    events = []
    sp = speculator.Speculator(events.append, mode=mode)
    if audio_qvec is not None:
        sp.audio_qvec, sp.audio_qdur = audio_qvec, 3.0
    m = await sp.finalize(local_frame(text), time.perf_counter())
    src = [e for e in events if e["t"] == "sources"][-1]["docs"]
    return m, src, sp


def test_local_route_final_uses_fused_ranking():
    m, src, _ = asyncio.run(_final("baseline", "What did the commentator say about Lonzo Ball in the game?"))
    assert src[0]["id"] == "m-2"
    assert all(d["t"] is not None for d in src)                # moments carry their timestamp to the UI chips
    assert m["n_sources"] == len(src) == 5


def test_local_route_without_rerank_is_cosine(monkeypatch):
    monkeypatch.setattr(config, "RERANK_LOCAL", False)
    _, src, _ = asyncio.run(_final("baseline", "What did the commentator say about Lonzo Ball in the game?"))
    assert src and len(src) <= 4                                # pool cosine over the final-local lane's top 4


def test_audio_lane_vector_joins_the_fusion(monkeypatch):
    monkeypatch.setattr(rerank, "W_AUDIO", 2.0)
    ix = lanes.index()
    qa = ix.tv[1].copy()                                        # the spoken question "sounds like" moment m-1
    q = "what did they say there"
    _, src_text, _ = asyncio.run(_final("baseline", q))
    _, src_audio, _ = asyncio.run(_final("baseline", q, audio_qvec=qa))
    pos = lambda src: [d["id"] for d in src].index("m-1")     # noqa: E731
    assert pos(src_audio) < pos(src_text)                      # the spoken question pulled its moment up


def test_audio_lane_keeps_longest_vector(monkeypatch):
    calls = []

    async def fake_local_audio(wav, k=4):
        calls.append(len(wav))
        return [], np.full(64, len(wav), np.float32)
    monkeypatch.setattr(lanes, "local_audio", fake_local_audio)

    async def go():
        sp = speculator.Speculator(lambda e: None, mode="headstart")
        sp.on_audio(np.zeros(int(16000 * 2.0), np.float32), route="local")
        sp.on_audio(np.zeros(int(16000 * 3.0), np.float32), route="local")
        await asyncio.gather(*[x.task for x in sp.fires])
        return sp
    sp = asyncio.run(go())
    assert calls == [32000, 48000] and sp.audio_qdur == pytest.approx(3.0) and sp.audio_qvec[0] == 48000


def test_draft_reuse_keeps_moment_citations(monkeypatch):
    monkeypatch.setattr(config, "DRAFT_STABLE_MS", 0)
    monkeypatch.setattr(config, "STABLE_UPDATES", 1)
    text = "What did the commentator say about Lonzo Ball in the game?"

    async def go():
        events = []
        sp = speculator.Speculator(events.append, mode="headstart")
        f = local_frame(text)
        sp.on_frame(f)
        for _ in range(50):
            await asyncio.sleep(0.01)
            sp.on_frame(f)
            if sp.drafts:
                break
        m = await sp.finalize(f, time.perf_counter())
        return m, events, sp
    m, events, sp = asyncio.run(go())
    assert m["path"] == "draft-reuse"
    src = [e for e in events if e["t"] == "sources"][-1]["docs"]
    assert src[0]["id"] == "m-2" and all(d["t"] is not None for d in src)
    assert [d.id for d in sp.drafts[0].stream.docs] == [d["id"] for d in src]     # citations = the draft's numbering


# ----------------------------------------------------------------------------- answer prompt
def test_build_prompt_moment_lines_cite_time_and_role():
    p = answer.build_prompt("when did the announcer call hunter's three", [DOCS[1], DOCS[3]])
    assert "[1] NBA broadcast at 00:26, play-by-play announcer: …before words [Hunter gets a good look and nails it]" \
        " after words…" in p
    assert "[2] Court hearing at 00:09, judge:" in p
    assert "quote the key words as transcribed" in answer.SYSTEM


def test_build_prompt_web_docs_unchanged():
    d = lanes.Doc(id="u", title="Eiffel Tower", text="The tower is 330 m tall.", url="https://x", source="wikipedia")
    assert "[1] Eiffel Tower: The tower is 330 m tall." in answer.build_prompt("how tall", [d])
