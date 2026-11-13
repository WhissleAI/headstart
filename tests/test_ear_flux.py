"""Deepgram Flux ear tests, no key and no network: a local websocket server speaks Flux's /v2/listen protocol
(Connected / TurnInfo{Update, StartOfTurn, EagerEndOfTurn, TurnResumed, EndOfTurn}, shapes from
developers.deepgram.com/reference/speech-to-text/listen-flux and checked against the live service) and drives FluxEar
into real Sessions with the fake lanes + answer model of test_speculation.
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
from test_ear_deepgram import FakeArcEar, Recorder, feed_audio  # noqa: E402

SR = 16000


def turn(event, words, ti=0, eot=0.05, trigger=None, win=(0.0, None)):
    """words: [(punctuated word, start, end)] of the WHOLE current turn (Flux resends the turn every time)."""
    m = {"type": "TurnInfo", "request_id": "req-flux", "sequence_id": 1, "event": event, "turn_index": ti,
         "audio_window_start": win[0], "audio_window_end": win[1] or (words[-1][2] + 0.1 if words else 0.24),
         "transcript": " ".join(w for w, _, _ in words),
         "words": [{"word": w, "confidence": 0.99, "start": a, "end": b} for w, a, b in words],
         "end_of_turn_confidence": eot}
    if event == "EndOfTurn":
        m["trigger"] = trigger or "model"
    return m


class FakeFlux:
    """script: [(audio_s, message)] sent once audio_s seconds of PCM16 arrived. ForceEndTurn -> on_force (an
    EndOfTurn trigger=manual); CloseStream -> close. drop_at: close 1011."""

    def __init__(self, script=(), on_force=None, drop_at=None):
        self.script = sorted(script, key=lambda x: x[0])
        self.on_force, self.drop_at = on_force, drop_at
        self.bytes, self.control, self.binary_sizes = 0, [], []
        self.path = self.headers = None

    async def handler(self, ws):
        self.path, self.headers = ws.request.path, ws.request.headers
        await ws.send(json.dumps({"type": "Connected", "request_id": "req-flux", "sequence_id": 0}))
        i = 0
        async for m in ws:
            if isinstance(m, bytes):
                self.bytes += len(m)
                self.binary_sizes.append(len(m))
                now = self.bytes / (2 * SR)
                while i < len(self.script) and self.script[i][0] <= now:
                    await ws.send(json.dumps(self.script[i][1]))
                    i += 1
                if self.drop_at is not None and now >= self.drop_at:
                    await ws.close(1011, "flux boom")
                    return
            else:
                t = json.loads(m)["type"]
                self.control.append(t)
                if t == "ForceEndTurn" and self.on_force:
                    await ws.send(json.dumps(self.on_force))
                elif t == "CloseStream":
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
        return f"ws://127.0.0.1:{self.port}/v2/listen"

    @property
    def query(self):
        return {k: v for k, v in parse_qs(urlparse(self.path).query).items()}


class EagerRecorder(Recorder):
    def __init__(self):
        super().__init__()
        self.eagers = []

    def on_eager(self, ws, now=None):
        self.eagers.append([w["text"] for w in ws])


@pytest.fixture(autouse=True)
def flux_env(monkeypatch):
    FakeArcEar.instances.clear()
    monkeypatch.setattr(ear, "ArcEar", FakeArcEar)
    monkeypatch.setattr(config, "DG_CONNECT_TIMEOUT_S", 2.0)
    monkeypatch.setattr(config, "DG_FLUSH_TIMEOUT_S", 0.5)
    monkeypatch.setattr(config, "FLUX_SOFT", "eager")
    monkeypatch.setattr(config, "SEMANTIC_ENDPOINT", True)
    monkeypatch.setattr(config, "ARC_META", "off")             # no torch in these tests (test_arc_meta_* fake it)
    yield


def no_tagger_gate(monkeypatch):
    """Isolate the eager path: the tagger+silence semantic endpoint never settles."""
    monkeypatch.setattr(config, "SEM_SILENCE_MS", 99999)
    monkeypatch.setattr(config, "SEM_STABLE_MS", 99999)
    monkeypatch.setattr(config, "SEM_COVER_S", 99999)


def flux_for(srv, sinks, events):
    return ear.FluxEar(sinks, on_event=events.append, key="testkey", url=ear.flux_url(srv.base))


W = [("Who", 0.30, 0.45), ("founded", 0.50, 0.85), ("the", 0.90, 1.00), ("company", 1.05, 1.40),
     ("Nvidia?", 1.45, 1.90)]
NVIDIA = [
    (0.6, turn("StartOfTurn", W[:1])),
    (0.9, turn("Update", W[:2])),
    (1.2, turn("Update", W[:3])),
    (1.5, turn("Update", W[:4])),
    (2.0, turn("Update", W[:5])),
    (2.1, turn("EagerEndOfTurn", W, eot=0.45)),
    (2.6, turn("EndOfTurn", W, eot=0.75)),
    (2.9, turn("Update", [], ti=1, win=(2.6, 2.9))),
]


def test_flux_ear_eager_soft_answer_confirmed_by_end_of_turn(monkeypatch):
    no_tagger_gate(monkeypatch)      # isolate the eager path from the tagger gate

    async def go():
        async with FakeFlux(NVIDIA) as srv:
            rec, ev_hs, ev_bl, ear_ev = EagerRecorder(), [], [], []
            hs, bl = Session(ev_hs.append, mode="headstart"), Session(ev_bl.append, mode="baseline")
            e = flux_for(srv, [rec, hs, bl], ear_ev)
            await feed_audio(e, 3.2)
            await e.flush()
            e.close()
            await asyncio.sleep(0.3)
            hs.close()
            bl.close()
            return srv, rec, hs, bl, ev_hs, ev_bl, ear_ev

    srv, rec, hs, bl, ev_hs, ev_bl, ear_ev = asyncio.run(go())
    q = srv.query
    assert q["model"] == ["flux-general-en"] and q["encoding"] == ["linear16"] and q["sample_rate"] == ["16000"]
    assert q["eot_threshold"] == [str(config.FLUX_EOT)] and q["eot_timeout_ms"] == [str(config.FLUX_EOT_TIMEOUT_MS)]
    assert float(q["eager_eot_threshold"][0]) == pytest.approx(ear.flux_eager())
    assert "language" not in q and "interim_results" not in q
    assert srv.headers["Authorization"] == "Token testkey"
    assert srv.bytes == int(3.2 * SR) * 2 and srv.control[-1] == "CloseStream"
    assert "ForceEndTurn" not in srv.control                   # nothing open at flush
    # words: lowercase, punctuation stripped, Flux's times
    assert rec.eagers == [["who", "founded", "the", "company", "nvidia"]]
    assert len(rec.finals) == 1 and [w["text"] for w in rec.finals[0]] == rec.eagers[0]
    assert rec.finals[0][-1] == {"text": "nvidia", "start": 1.45, "end": 1.90}
    assert [x["reason"] for x in ear_ev if x.get("kind") == "final"] == ["eot_model"]
    kinds = [x["kind"] for x in ear_ev]
    assert kinds.index("StartOfTurn") < kinds.index("EagerEndOfTurn") < kinds.index("EndOfTurn")
    # headstart answered at the eager endpoint; EndOfTurn confirmed it (one answer). baseline answered at EndOfTurn.
    assert len(hs.history) == 1 and hs.history[0]["endpoint_by"] == "semantic"
    assert hs.history[0]["endpoint_trigger"] == "eager"
    ep = [x for x in ev_hs if x["t"] == "endpoint"]
    assert len(ep) == 1 and ep[0]["trigger"] == "eager" and ep[0]["soft"]
    assert len(bl.history) == 1 and bl.history[0]["endpoint_by"] == "ear"
    assert "nvidia" in hs.history[0]["query"] and "nvidia" in bl.history[0]["query"]


def test_turn_resumed_answers_the_longer_question(monkeypatch):
    no_tagger_gate(monkeypatch)
    w1 = [("How", 0.3, 0.45), ("tall", 0.5, 0.7), ("is", 0.75, 0.85), ("the", 0.9, 1.0), ("Eiffel", 1.05, 1.4),
          ("Tower?", 1.45, 1.8)]
    w2 = w1[:-1] + [("Tower", 1.45, 1.8), ("in", 2.3, 2.4), ("feet?", 2.45, 2.8)]
    script = [(0.6, turn("StartOfTurn", w1[:1])), (1.9, turn("Update", w1)), (2.0, turn("EagerEndOfTurn", w1)),
              (2.5, turn("TurnResumed", w2[:-1])), (2.9, turn("Update", w2)), (3.0, turn("EagerEndOfTurn", w2)),
              (3.3, turn("EndOfTurn", w2, eot=0.8))]

    async def go():
        async with FakeFlux(script) as srv:
            rec, ev_hs, ear_ev = EagerRecorder(), [], []
            hs = Session(ev_hs.append, mode="headstart")
            e = flux_for(srv, [rec, hs], ear_ev)
            await feed_audio(e, 3.6, pace=0.02)
            await e.flush()
            e.close()
            await asyncio.sleep(0.3)
            hs.close()
            return rec, hs, ev_hs, ear_ev

    rec, hs, ev_hs, ear_ev = asyncio.run(go())
    assert len(rec.eagers) == 2 and rec.eagers[1][-2:] == ["in", "feet"]
    assert any(x["kind"] == "TurnResumed" for x in ear_ev)
    assert len(rec.finals) == 1 and rec.finals[0][-1]["text"] == "feet"
    eps = [x for x in ev_hs if x["t"] == "endpoint"]
    assert [x.get("trigger") for x in eps] == ["eager", "eager"]          # second eager = extended soft answer
    assert any(x["t"] == "extended" for x in ev_hs)
    assert len(hs.history) == 1 and "feet" in hs.history[0]["question"]


def test_eager_gate_mode_and_off(monkeypatch):
    """eager+gate: an eager on a fragment the tagger gate rejects gives no soft answer; off: eager ignored."""
    no_tagger_gate(monkeypatch)
    frag = [("What", 0.3, 0.5), ("is", 0.55, 0.7), ("the", 0.75, 0.85)]
    script = [(0.6, turn("StartOfTurn", frag[:1])), (1.0, turn("Update", frag)), (1.1, turn("EagerEndOfTurn", frag))]

    def run(mode):
        monkeypatch.setattr(config, "FLUX_SOFT", mode)

        async def go():
            async with FakeFlux(script) as srv:
                ev = []
                hs = Session(ev.append, mode="headstart")
                e = flux_for(srv, [hs], [])
                await feed_audio(e, 1.4)
                e.close()
                await asyncio.sleep(0.2)
                hs.close()
                return ev
        return asyncio.run(go())

    assert not [x for x in run("eager+gate") if x["t"] == "endpoint"]
    assert not [x for x in run("off") if x["t"] == "endpoint"]


def test_flush_forces_end_of_turn():
    script = [(0.8, turn("StartOfTurn", [("What", 0.2, 0.4), ("time", 0.45, 0.7)]))]
    fin = turn("EndOfTurn", [("What", 0.2, 0.4), ("time", 0.45, 0.7), ("is", 0.75, 0.85), ("it?", 0.9, 1.1)],
               trigger="manual")

    async def go():
        async with FakeFlux(script, on_force=fin) as srv:
            rec, ev = Recorder(), []
            e = flux_for(srv, [rec], ev)
            await feed_audio(e, 1.2)
            await asyncio.sleep(0.1)
            assert rec.partials[-1] == ["what", "time"] and not rec.finals
            await e.flush()
            e.close()
            await asyncio.sleep(0.2)
            return srv, rec, ev

    srv, rec, ev = asyncio.run(go())
    assert srv.control == ["ForceEndTurn", "CloseStream"]
    assert [[w["text"] for w in f] for f in rec.finals] == [["what", "time", "is", "it"]]
    assert [x["reason"] for x in ev if x.get("kind") == "final"] == ["eot_manual"]


def test_two_turns_and_keepalive_silence(monkeypatch):
    monkeypatch.setattr(config, "DG_KEEPALIVE_S", 0.1)
    script = [(0.6, turn("StartOfTurn", [("Hi", 0.2, 0.5)])), (0.9, turn("EndOfTurn", [("Hi", 0.2, 0.5)])),
              (1.5, turn("StartOfTurn", [("Weather", 1.0, 1.3)], ti=1)),
              (1.8, turn("EndOfTurn", [("Weather", 1.0, 1.3), ("today?", 1.35, 1.6)], ti=1))]

    async def go():
        async with FakeFlux(script) as srv:
            rec, ev = Recorder(), []
            e = flux_for(srv, [rec], ev)
            await feed_audio(e, 2.0)
            n = srv.bytes
            await asyncio.sleep(0.45)                    # caller stops feeding: silence keeps the socket alive
            e.close()
            await asyncio.sleep(0.2)
            return srv, rec, n

    srv, rec, n = asyncio.run(go())
    assert [[w["text"] for w in f] for f in rec.finals] == [["hi"], ["weather", "today"]]
    assert srv.bytes - n >= 2 * 3200 and "KeepAlive" not in srv.control   # Flux defines no KeepAlive message


def test_flux_drop_falls_back_to_arc():
    script = [(0.8, turn("StartOfTurn", [("who", 0.3, 0.5)])), (1.0, turn("Update", [("who", 0.3, 0.5),
                                                                                       ("wrote", 0.6, 0.9)]))]

    async def go():
        async with FakeFlux(script, drop_at=1.6) as srv:
            rec, ev = Recorder(), []
            e = flux_for(srv, [rec], ev)
            await feed_audio(e, 3.0)
            await e.flush()
            e.close()
            return e, rec, ev

    e, rec, ev = asyncio.run(go())
    fb = [x for x in ev if x.get("kind") == "fallback"]
    assert len(fb) == 1 and fb[0]["from"] == "flux" and "1011" in fb[0]["reason"] and e.active == "arc"
    (arc_ear,) = FakeArcEar.instances
    assert arc_ear.offset == 0.0 and arc_ear.offset * SR + arc_ear.primed == int(3.0 * SR)
    assert rec.audio == int(3.0 * SR) and rec.finals[-1][0]["text"] == "arc"


def test_flux_words_and_url(monkeypatch):
    assert [ear.flux_word(w) for w in ["Hey.", "What's", "Chicago", "tomorrow?", "3.5", "U.S.", "\"OK,\"", "--"]] == \
        ["hey", "what's", "chicago", "tomorrow", "3.5", "u.s", "ok", ""]
    monkeypatch.setattr(config, "FLUX_EAGER_EOT", "0.95")       # clamped to eot (Flux rejects eager > eot)
    monkeypatch.setattr(config, "FLUX_KEYTERMS", ["Nvidia", "Eiffel Tower"])
    q = parse_qs(urlparse(ear.flux_url("wss://x/v2/listen")).query)
    assert q["eager_eot_threshold"] == [str(config.FLUX_EOT)] and q["keyterm"] == ["Nvidia", "Eiffel Tower"]
    monkeypatch.setattr(config, "FLUX_EAGER_EOT", "off")
    assert "eager_eot_threshold" not in parse_qs(urlparse(ear.flux_url("wss://x/v2/listen")).query)


def test_factory_and_health_flux(monkeypatch):
    monkeypatch.setattr(config, "FLUX_URL", "ws://127.0.0.1:9/v2/listen")    # never the real service in tests

    async def go():
        ev = []
        monkeypatch.setattr(config, "EAR", "flux")
        monkeypatch.setattr(config, "DEEPGRAM_API_KEY", "")
        e1 = ear.make_ear([Recorder()], on_event=ev.append)
        h1 = await appmod.health()
        monkeypatch.setattr(config, "DEEPGRAM_API_KEY", "dg-key")
        e2 = ear.make_ear([Recorder()])
        h2 = await appmod.health()
        e2.close()
        return ev, e1, e2, h1, h2

    ev, e1, e2, h1, h2 = asyncio.run(go())
    assert isinstance(e1, FakeArcEar) and "DEEPGRAM_API_KEY" in ev[0]["text"] and h1["ear"] == "arc"
    assert isinstance(e2, ear.FluxEar) and h2["ear"] == "flux" and h2["deepgram_model"] == "flux-general-en"
    assert h2["flux"]["eot_threshold"] == config.FLUX_EOT


class FakeArcStream:
    """Arc external-mode stream stand-in: records words/audio; reports a live utterance once words arrived."""
    def __init__(self, **kw):
        self.kw, self.words, self.samples = kw, [], 0

    def add_words(self, ev):
        self.words.append(ev)

    def feed(self, pcm):
        self.samples += len(pcm)
        if not self.words:
            return {"live": {"t": self.samples / SR - 0.3, "utt": None}}
        out = {"live": {"t": round(self.samples / SR - 0.3, 1), "utt": {
            "device_directed": 0.97, "command_complete": 0.6 if self.words[-1][-1]["is_final"] else 0.1,
            "act": [{"label": "discourse.act.question", "p": 0.9}], "intent": []},
            "speaker": {"id": "S1", "role": [{"label": "speaker.role.device_user", "p": 1.0}]}}}
        if self.samples == 25 * 1600:
            out["events"] = [{"tag": "discourse.assistant.command_complete", "t": 1.9, "p": 0.4}]
        return out

    def flush(self):
        return {"live": {"t": self.samples / SR, "utt": None}}


class FakeArc:
    def __init__(self):
        self.streams = []

    def stream(self, **kw):
        self.streams.append(FakeArcStream(**kw))
        return self.streams[-1]


def test_arc_meta_rides_along_and_reaches_the_readout(monkeypatch):
    fa = FakeArc()
    e_off = []
    monkeypatch.setattr(config, "ARC_META", "mini")
    monkeypatch.setattr(ear, "arc", lambda: fa)
    no_tagger_gate(monkeypatch)

    async def go():
        async with FakeFlux(NVIDIA) as srv:
            ev_hs, ear_ev = [], []
            hs = Session(ev_hs.append, mode="headstart")
            e = flux_for(srv, [hs], ear_ev)
            await feed_audio(e, 3.2, pace=0.01)
            e_off.append(e.meta.offset)
            await e.flush()
            e.close()
            await asyncio.sleep(0.2)
            hs.close()
            return ev_hs

    ev_hs = asyncio.run(go())
    (st,) = fa.streams
    # external mode, words only from Flux, and never Arc's own decoder fallback
    assert st.kw["transcript"] == "external" and st.kw["fallback"] is None and st.kw["partials"] is False
    assert st.samples + round(e_off[0] * SR) == int(3.2 * SR)        # joined at its offset, then every chunk
    w = st.words[-1][-1]                                               # Arc's own clock starts at its offset
    assert w["text"] == "Nvidia?" and w["is_final"] and w["start"] == pytest.approx(1.45 - e_off[0])
    arc_ev = [x for x in ev_hs if x["t"] == "arc"]
    assert arc_ev and arc_ev[-1]["for_me"] == 0.97 and arc_ev[-1]["role"] == "speaker.role.device_user"
    assert any(x["cc_events"] for x in arc_ev) and any(x["cc_p"] == 0.6 for x in arc_ev)
    assert any(x["t"] == "frame" and (x.get("arc") or {}).get("for_me") == 0.97 for x in ev_hs)
    assert len([x for x in ev_hs if x["t"] == "endpoint"]) == 1      # metadata never changes the answer path


def test_arc_meta_failure_never_touches_words(monkeypatch):
    class Boom(FakeArc):
        def stream(self, **kw):
            s = super().stream(**kw)
            s.feed = lambda pcm: 1 / 0
            return s
    monkeypatch.setattr(config, "ARC_META", "mini")
    monkeypatch.setattr(ear, "arc", lambda: Boom())

    async def go():
        async with FakeFlux(NVIDIA) as srv:
            rec, ev = EagerRecorder(), []
            e = flux_for(srv, [rec], ev)
            await feed_audio(e, 3.0)
            await e.flush()
            e.close()
            return rec
    rec = asyncio.run(go())
    assert [w["text"] for w in rec.finals[0]][-1] == "nvidia"


def test_eager_waits_for_arc_command_complete(monkeypatch):
    """FLUX_SOFT=eager+arc: the eager at 2.1 s waits; Arc's command_complete (t=1.9, said at 2.5 s) releases it."""
    fa = FakeArc()
    monkeypatch.setattr(config, "ARC_META", "mini")
    monkeypatch.setattr(config, "FLUX_SOFT", "eager+arc")
    monkeypatch.setattr(ear, "arc", lambda: fa)
    no_tagger_gate(monkeypatch)

    async def go():
        async with FakeFlux(NVIDIA[:-2]) as srv:          # no EndOfTurn: only the eager + Arc can answer early
            ev = []
            hs = Session(ev.append, mode="headstart")
            e = flux_for(srv, [hs], [])
            await feed_audio(e, 2.4, pace=0.01)
            await asyncio.sleep(0.1)
            before = [x for x in ev if x["t"] == "endpoint"]
            await feed_audio(e, 0.4, pace=0.01)
            await asyncio.sleep(0.2)
            e.close()
            hs.close()
            return before, ev

    before, ev = asyncio.run(go())
    assert not before and any(x["t"] == "eager_skip" and x.get("waiting") == "arc" for x in ev)
    eps = [x for x in ev if x["t"] == "endpoint"]
    assert len(eps) == 1 and eps[0]["trigger"] == "eager+arc"
