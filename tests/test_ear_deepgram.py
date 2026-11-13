"""Deepgram ear tests with no key and no network: a local websocket server speaks Deepgram's live protocol (Results /
UtteranceEnd / Metadata messages in the documented shapes, released as the audio they describe arrives) and drives
DeepgramEar into real Sessions with the fake lanes + answer model of test_speculation.

Run: .venv/bin/python -m pytest -q tests
"""
import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
from websockets.asyncio.server import serve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import app as appmod, config, ear  # noqa: E402
from server.session import Session  # noqa: E402
from test_speculation import fakes  # noqa: E402,F401  (autouse: fake lanes, embeddings, answer model)

SR = 16000
CHUNK = 1600


# ----------------------------------------------------------------------------- Deepgram-shaped messages
def results(words, is_final=False, speech_final=False, from_finalize=False):
    """words: [(punctuated_word, start, end)]. Shape: developers.deepgram.com listen-streaming "Results"."""
    ws = [{"word": t.lower().strip("?.,!"), "start": a, "end": b, "confidence": 0.98, "punctuated_word": t}
          for t, a, b in words]
    start = words[0][1] if words else 0.0
    return {"type": "Results", "channel_index": [0, 1], "duration": round((words[-1][2] if words else 0) - start, 3),
            "start": start, "is_final": is_final, "speech_final": speech_final, "from_finalize": from_finalize,
            "channel": {"alternatives": [{"transcript": " ".join(t for t, _, _ in words), "confidence": 0.97,
                                          "words": ws}]},
            "metadata": {"request_id": "req-test", "model_info": {"name": "general-nova-3", "version": "x",
                                                                  "arch": "nova-3"}, "model_uuid": "u"}}


class FakeDeepgram:
    """script: [(audio_s, message)] sent once the server has received audio_s seconds of PCM16.
    on_finalize: message sent in reply to {"type":"Finalize"}.  drop_at: close with 1011 at that audio time."""

    def __init__(self, script=(), on_finalize=None, drop_at=None):
        self.script = sorted(script, key=lambda x: x[0])
        self.on_finalize = on_finalize
        self.drop_at = drop_at
        self.bytes = 0
        self.control: list[str] = []
        self.path = None
        self.headers = None
        self.server = None

    async def handler(self, ws):
        self.path, self.headers = ws.request.path, ws.request.headers
        i = 0
        async for m in ws:
            if isinstance(m, bytes):
                self.bytes += len(m)
                now = self.bytes / (2 * SR)
                while i < len(self.script) and self.script[i][0] <= now:
                    await ws.send(json.dumps(self.script[i][1]))
                    i += 1
                if self.drop_at is not None and now >= self.drop_at:
                    await ws.close(1011, "upstream boom")
                    return
            else:
                t = json.loads(m)["type"]
                self.control.append(t)
                if t == "Finalize" and self.on_finalize:
                    await ws.send(json.dumps(self.on_finalize))
                elif t == "CloseStream":
                    await ws.send(json.dumps({"type": "Metadata", "request_id": "req-test", "sha256": "0",
                                              "created": "2026-11-12T00:00:00Z", "duration": self.bytes / 32000,
                                              "channels": 1}))
                    await ws.close()
                    return

    async def __aenter__(self):
        self.server = await serve(self.handler, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *a):
        self.server.close()
        await self.server.wait_closed()

    @property
    def base(self):
        return f"ws://127.0.0.1:{self.port}/v1/listen"

    @property
    def query(self):
        return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}


class Recorder:
    """A sink that keeps what the ear said."""

    def __init__(self):
        self.audio = 0
        self.partials: list[list[str]] = []
        self.finals: list[list[dict]] = []

    def on_audio(self, pcm, at):
        self.audio += len(pcm)

    def on_partial(self, ws, now=None):
        txt = [w["text"] for w in ws]
        if not self.partials or self.partials[-1] != txt:
            self.partials.append(txt)

    async def on_final(self, ws):
        self.finals.append(ws)


class FakeArcEar:
    """Stands in for ArcEar (no torch): records what it was fed and says 'arc heard' on flush."""
    instances: list["FakeArcEar"] = []

    def __init__(self, sinks, domain="assistant", on_event=None, offset_s=0.0):
        self.sinks, self.offset = sinks, offset_s
        self.primed = 0
        self.live = 0
        self.final_tasks = []
        self.closed = False
        FakeArcEar.instances.append(self)

    async def feed(self, pcm, notify=True):
        if notify:
            for s in self.sinks:
                s.on_audio(pcm, 0.0)
            self.live += len(pcm)
        else:
            self.primed += len(pcm)

    async def flush(self):
        ws = [{"text": "arc", "start": self.offset + 0.1, "end": self.offset + 0.3}]
        for s in self.sinks:
            await s.on_final(ws)

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def dg_env(monkeypatch):
    FakeArcEar.instances.clear()
    monkeypatch.setattr(ear, "ArcEar", FakeArcEar)
    monkeypatch.setattr(config, "DG_ARC_EVENTS", False)
    monkeypatch.setattr(config, "DG_CONNECT_TIMEOUT_S", 2.0)
    monkeypatch.setattr(config, "DG_FLUSH_TIMEOUT_S", 0.5)
    yield


async def feed_audio(e, seconds, pace=0.004):
    rng = np.random.default_rng(0)
    for _ in range(int(round(seconds * SR / CHUNK))):
        await e.feed(rng.standard_normal(CHUNK).astype(np.float32) * 0.05)
        await asyncio.sleep(pace)


def ear_for(srv, sinks, events):
    return ear.DeepgramEar(sinks, on_event=events.append, key="testkey", url=ear.dg_url(srv.base))


# ----------------------------------------------------------------------------- the misheard-then-revised question
EIFFEL = [
    (0.6, results([("How", 0.30, 0.45)])),
    (0.9, results([("How", 0.30, 0.45), ("tall", 0.50, 0.70)])),
    (1.1, results([("How", 0.30, 0.45), ("tall", 0.50, 0.70), ("is", 0.75, 0.85)], is_final=True)),
    (1.5, results([("the", 0.90, 1.00), ("iPhone", 1.05, 1.40)])),
    (1.9, results([("the", 0.90, 1.00), ("iPhone", 1.05, 1.40), ("tower", 1.45, 1.80)])),
    (2.2, results([("the", 0.90, 1.00), ("Eiffel", 1.05, 1.40), ("Tower", 1.45, 1.80)])),        # revision
    (2.4, results([("the", 0.90, 1.00), ("Eiffel", 1.05, 1.40), ("Tower?", 1.45, 1.80)], is_final=True,
                  speech_final=True)),
    (3.0, {"type": "UtteranceEnd", "channel": [0, 1], "last_word_end": 1.80}),                   # nothing left: no-op
]


def test_deepgram_ear_drives_both_race_sessions():
    async def go():
        async with FakeDeepgram(EIFFEL) as srv:
            rec, ev_hs, ev_bl, ear_ev = Recorder(), [], [], []
            hs, bl = Session(ev_hs.append, mode="headstart"), Session(ev_bl.append, mode="baseline")
            e = ear_for(srv, [rec, hs, bl], ear_ev)
            await feed_audio(e, 3.5)
            await e.flush()
            e.close()
            await asyncio.sleep(0.3)
            hs.close()
            bl.close()
            return srv, rec, hs, bl, ev_hs, ev_bl, ear_ev

    srv, rec, hs, bl, ev_hs, ev_bl, ear_ev = asyncio.run(go())
    # protocol: documented query, token auth, all audio as PCM16, Finalize on flush, CloseStream on close
    q = srv.query
    assert q["model"] == "nova-3" and q["interim_results"] == "true" and q["encoding"] == "linear16"
    assert q["sample_rate"] == "16000" and q["channels"] == "1" and q["endpointing"] == str(config.DG_ENDPOINTING_MS)
    assert q["smart_format"] == "true" and q["punctuate"] == "true" and q["utterance_end_ms"] == "1000"
    assert srv.headers["Authorization"] == "Token testkey"
    assert srv.bytes == int(3.5 * SR) * 2
    assert "Finalize" in srv.control and srv.control[-1] == "CloseStream"
    # partials: finalized segment + interim, the misheard word revised in place
    assert ["how", "tall", "is", "the", "iphone", "tower"] in rec.partials
    i_bad = rec.partials.index(["how", "tall", "is", "the", "iphone", "tower"])
    assert rec.partials[i_bad + 1] == ["how", "tall", "is", "the", "eiffel", "tower"]
    # exactly one final (speech_final), with Deepgram's word times; UtteranceEnd and flush add nothing
    assert len(rec.finals) == 1
    f = rec.finals[0]
    assert [w["text"] for w in f] == ["how", "tall", "is", "the", "eiffel", "tower"]
    assert (f[0]["start"], f[-1]["end"]) == (0.30, 1.80)
    assert all(a["end"] <= b["start"] for a, b in zip(f, f[1:]))
    assert [x["reason"] for x in ear_ev if x.get("kind") == "final"] == ["speech_final"]
    assert any(x.get("kind") == "connected" for x in ear_ev) and not any(x.get("kind") == "fallback" for x in ear_ev)
    # both race sessions got identical words, and each answered the corrected question once
    words = lambda evs: [(tuple((w["w"], w["t0"], w["t1"]) for w in x["words"]), x["from"], x.get("final", False))
                         for x in evs if x["t"] == "words"]
    assert words(ev_hs) == words(ev_bl) and words(ev_hs)
    assert words(ev_hs)[-1][0][-1] == ("tower", 1.45, 1.80)
    for s in (hs, bl):
        assert len(s.history) == 1 and "eiffel tower" in s.history[0]["query"]
    assert rec.audio == int(3.5 * SR)


def test_flush_finalizes_buffered_words():
    """End of a run mid-sentence: Finalize makes Deepgram return what it buffered (from_finalize) -> on_final."""
    script = [(0.8, results([("What", 0.2, 0.4), ("time", 0.45, 0.7)]))]
    fin = results([("What", 0.2, 0.4), ("time", 0.45, 0.7), ("is", 0.75, 0.85), ("it?", 0.9, 1.1)], is_final=True,
                  from_finalize=True)

    async def go():
        async with FakeDeepgram(script, on_finalize=fin) as srv:
            rec, ev = Recorder(), []
            e = ear_for(srv, [rec], ev)
            await feed_audio(e, 1.2)
            await asyncio.sleep(0.1)
            assert rec.partials[-1] == ["what", "time"] and not rec.finals
            await e.flush()
            e.close()
            await asyncio.sleep(0.2)
            assert srv.control[-1] == "CloseStream"
            return rec, ev

    rec, ev = asyncio.run(go())
    assert [[w["text"] for w in f] for f in rec.finals] == [["what", "time", "is", "it"]]
    assert [x["reason"] for x in ev if x.get("kind") == "final"] == ["flush"]


def test_two_utterances_and_utterance_end_backstop():
    """A second question closed by UtteranceEnd (no speech_final), words before the first final not repeated."""
    script = [
        (0.8, results([("hi", 0.2, 0.5)], is_final=True, speech_final=True)),
        (1.6, results([("weather", 1.0, 1.3), ("today", 1.35, 1.6)])),
        (2.0, results([("weather", 1.0, 1.3), ("today", 1.35, 1.6)], is_final=True)),
        (2.8, {"type": "UtteranceEnd", "channel": [0, 1], "last_word_end": 1.6}),
        (2.9, results([("today", 1.35, 1.6)], is_final=True)),        # a late duplicate: already finalized, dropped
    ]

    async def go():
        async with FakeDeepgram(script) as srv:
            rec, ev = Recorder(), []
            e = ear_for(srv, [rec], ev)
            await feed_audio(e, 3.2)
            await e.flush()
            e.close()
            await asyncio.sleep(0.2)
            assert srv.control[-1] == "CloseStream"
            return rec, ev

    rec, ev = asyncio.run(go())
    assert [[w["text"] for w in f] for f in rec.finals] == [["hi"], ["weather", "today"]]
    assert [x["reason"] for x in ev if x.get("kind") == "final"] == ["speech_final", "utterance_end"]


def test_keepalive_when_audio_pauses(monkeypatch):
    monkeypatch.setattr(config, "DG_KEEPALIVE_S", 0.1)

    async def go():
        async with FakeDeepgram() as srv:
            e = ear_for(srv, [Recorder()], [])
            await feed_audio(e, 0.2)
            await asyncio.sleep(0.45)
            e.close()
            await asyncio.sleep(0.2)
            return srv

    srv = asyncio.run(go())
    assert srv.control.count("KeepAlive") >= 2 and srv.control[-1] == "CloseStream"


def test_connection_loss_falls_back_to_arc_without_losing_audio():
    """Server drops mid-question: the open utterance's audio (since the last final) is re-fed to Arc, later audio
    goes to Arc, the sessions never see audio twice, and Arc's final reaches them."""
    script = [
        (0.8, results([("hi", 0.2, 0.5)], is_final=True, speech_final=True)),
        (1.4, results([("who", 1.0, 1.2), ("wrote", 1.25, 1.4)])),
    ]

    async def go():
        async with FakeDeepgram(script, drop_at=1.6) as srv:
            rec, ev = Recorder(), []
            e = ear_for(srv, [rec], ev)
            await feed_audio(e, 3.0)
            await e.flush()
            e.close()
            return e, rec, ev

    e, rec, ev = asyncio.run(go())
    fb = [x for x in ev if x.get("kind") == "fallback"]
    assert len(fb) == 1 and "1011" in fb[0]["reason"] and e.active == "arc"
    (arc_ear,) = FakeArcEar.instances
    total = int(3.0 * SR)
    assert rec.audio == total                                   # sinks got every chunk exactly once
    assert arc_ear.live == 0                                    # ... the fallback never re-sends audio to them
    assert arc_ear.offset == pytest.approx(0.5)                 # replay starts at the chunk after the last final word
    assert arc_ear.offset * SR + arc_ear.primed == total        # replayed + live audio == everything after it
    assert [[w["text"] for w in f] for f in rec.finals] == [["hi"], ["arc"]]
    assert ["hi"] != rec.partials[-1] and ["who", "wrote"] in rec.partials


def test_connect_failure_falls_back_to_arc():
    async def go():
        async with FakeDeepgram() as srv:
            dead = srv.base
        rec, ev = Recorder(), []
        e = ear.DeepgramEar([rec], on_event=ev.append, key="k", url=ear.dg_url(dead))  # nothing listens there now
        await feed_audio(e, 0.5)
        await asyncio.sleep(0.1)
        await e.flush()
        e.close()
        return e, rec, ev

    e, rec, ev = asyncio.run(go())
    fb = [x for x in ev if x.get("kind") == "fallback"]
    assert len(fb) == 1 and fb[0]["reason"].startswith("connect failed") and e.active == "arc"
    (arc_ear,) = FakeArcEar.instances
    assert arc_ear.offset == 0.0 and arc_ear.primed == int(0.5 * SR) and arc_ear.live == 0
    assert rec.audio == int(0.5 * SR) and rec.finals[-1][0]["text"] == "arc"


def test_factory_and_health(monkeypatch):
    monkeypatch.setattr(config, "DG_URL", "ws://127.0.0.1:9/v1/listen")     # never the real service in tests

    async def go():
        ev = []
        monkeypatch.setattr(config, "EAR", "deepgram")
        monkeypatch.setattr(config, "DEEPGRAM_API_KEY", "")
        e1 = ear.make_ear([Recorder()], on_event=ev.append)
        h1 = await appmod.health()
        monkeypatch.setattr(config, "DEEPGRAM_API_KEY", "dg-key")
        e2 = ear.make_ear([Recorder()])
        h2 = await appmod.health()
        e2.close()
        monkeypatch.setattr(config, "EAR", "arc")
        e3 = ear.make_ear([Recorder()])
        h3 = await appmod.health()
        return ev, e1, e2, e3, h1, h2, h3

    ev, e1, e2, e3, h1, h2, h3 = asyncio.run(go())
    assert isinstance(e1, FakeArcEar) and ev[0]["kind"] == "warning" and "DEEPGRAM_API_KEY" in ev[0]["text"]
    assert h1["ear"] == "arc" and h1["ear_requested"] == "deepgram" and h1["deepgram_key"] is False
    assert isinstance(e2, ear.DeepgramEar) and h2["ear"] == "deepgram" and h2["deepgram_model"] == "nova-3"
    assert isinstance(e3, FakeArcEar) and h3["ear"] == "arc" and "deepgram_key" not in h3


def test_ready_event_names_the_ear(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(config, "EAR", "deepgram")
    monkeypatch.setattr(config, "DEEPGRAM_API_KEY", "")
    with TestClient(appmod.app).websocket_connect("/ws") as ws:          # no `with TestClient`: no startup warm-up
        r = ws.receive_json()
    assert r["t"] == "ready" and r["ear"] == "arc" and r["ear_requested"] == "deepgram"
