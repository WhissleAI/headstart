"""Regression tests for the independent review's confirmed bugs (one test per bug, plus the minor cleanups).

Fakes only (no network, no Arc audio model): fake Anthropic client, fake lanes/embeddings (test_speculation's autouse
fixture), a fake Arc ear for the WebSocket tests, and the real Arc Tagger in-process (cheap, text only).
Run: .venv/bin/python -m pytest -q tests
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import answer, config, embed, frame, lanes  # noqa: E402

RealStream = answer.Stream            # captured before the autouse fixture swaps in FakeStream

from test_speculation import WEB_CALLS, FakeStream, _vec, fakes  # noqa: E402,F401  (autouse fixture)
from server import app as appmod  # noqa: E402
from server import ear as earmod  # noqa: E402
from server.session import Session, gate_open  # noqa: E402
from server.speculator import Speculator  # noqa: E402
from server.tagger import Tagger  # noqa: E402


def F(words):
    return frame.build(words.split(), [], None, local_ok=False)


# ============================================================================ 1. hedged answer request
class FakeMsgStream:
    def __init__(self, spec):
        self.spec = spec

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    @property
    def text_stream(self):
        spec = self.spec

        async def gen():
            await asyncio.sleep(spec["ttft"])
            if spec.get("fail"):
                raise RuntimeError(spec["fail"])
            for i in range(spec["n"]):
                yield f"t{i} "
                await asyncio.sleep(spec["gap"])
        return gen()


class FakeClient:
    def __init__(self, specs):
        self.specs, self.calls = list(specs), 0

    @property
    def messages(self):
        return self

    def stream(self, **kw):
        s = self.specs[self.calls]
        self.calls += 1
        return FakeMsgStream(s)


async def _hedge_case(monkeypatch, specs, hedge_ms):
    monkeypatch.setattr(config, "HEDGE_MS", hedge_ms)
    fc = FakeClient(specs)
    monkeypatch.setattr(answer, "_client", fc)
    n = {"wait": 0}
    orig = asyncio.wait

    async def counting_wait(*a, **k):
        n["wait"] += 1
        return await orig(*a, **k)
    monkeypatch.setattr(answer.asyncio, "wait", counting_wait)
    unretrieved = []
    asyncio.get_running_loop().set_exception_handler(lambda loop, ctx: unretrieved.append(ctx.get("message")))
    t = time.perf_counter()
    st = RealStream("q", [])
    await st.task
    took = time.perf_counter() - t
    monkeypatch.setattr(answer.asyncio, "wait", orig)
    await asyncio.sleep(0.05)
    import gc
    gc.collect()
    others = [x for x in asyncio.all_tasks() if x is not asyncio.current_task()]
    return st, fc, n["wait"], took, unretrieved, others


def test_hedge_no_busy_spin_fast_fail_hedges_and_cleans_up(monkeypatch):
    async def go():
        out = {}
        # hedge wins; the primary returns at its first token (a loser) -> must not spin until the winner finishes
        out["win"] = await _hedge_case(monkeypatch, [dict(ttft=0.3, n=20, gap=0.05), dict(ttft=0.05, n=20, gap=0.05)], 100)
        # the primary fails fast (a 529): the hedge goes out at once, not after HEDGE_MS
        out["fast"] = await _hedge_case(monkeypatch, [dict(ttft=0.01, n=0, gap=0, fail="overloaded"),
                                                      dict(ttft=0.05, n=5, gap=0.01)], 800)
        # both fail: the error surfaces; no exception left unretrieved
        out["both"] = await _hedge_case(monkeypatch, [dict(ttft=0.01, n=0, gap=0, fail="a"),
                                                      dict(ttft=0.01, n=0, gap=0, fail="b")], 800)
        return out
    out = asyncio.run(go())
    st, fc, waits, took, unretrieved, others = out["win"]
    assert st.hedged and st.winner == 1 and st.error is None and len(st.tokens) == 20
    assert waits <= 6, f"busy loop: {waits} asyncio.wait calls"          # was ~26k/s before the fix
    assert not others, others                                              # shield/waiter task not left pending
    st, fc, waits, took, unretrieved, _ = out["fast"]
    assert fc.calls == 2 and st.hedged and st.error is None and len(st.tokens) == 5
    assert took < 0.5, took                                                # not gated on HEDGE_MS=800
    st, fc, waits, took, unretrieved, _ = out["both"]
    assert st.done and st.error and "RuntimeError" in st.error and took < 0.5
    assert not [m for m in unretrieved if m and "never retrieved" in m], unretrieved


# ============================================================================ 2. cancel before the task starts
def test_stream_cancelled_before_start_ends_and_follow_returns(monkeypatch):
    class C:
        @property
        def messages(self):
            return self

        def stream(self, **kw):
            raise AssertionError("never reached")
    monkeypatch.setattr(answer, "_client", C())

    async def go():
        st = RealStream("q", [])
        st.cancel()                          # same tick as creation: _run never runs
        toks = await asyncio.wait_for(_drain(st), 1.0)       # used to hang forever
        # cancel() on a finished stream changes nothing (FakeStream used to stamp error="cancelled" on it)
        fs = FakeStream("q", [])
        await fs.task
        fs.cancel()
        fs2 = FakeStream("q", [])
        fs2.cancel()
        ftoks = await asyncio.wait_for(_drain(fs2), 1.0)
        return st, toks, fs, fs2, ftoks
    st, toks, fs, fs2, ftoks = asyncio.run(go())
    assert st.done and st.error == "cancelled" and toks == [] and st.task.cancelled()
    assert fs.done and fs.error is None and fs.text()
    assert fs2.done and fs2.error == "cancelled" and ftoks == []


async def _drain(st):
    return [t async for t in st.follow()]


# ============================================================================ 3. pool consistency when a fire is cancelled mid-embed
def test_cancel_mid_embed_keeps_pool_consistent(monkeypatch):
    async def slow_arun(fn, *a):
        await asyncio.sleep(0.2)          # document embedding in flight
        return fn(*a)
    monkeypatch.setattr(embed, "arun", slow_arun)

    async def web(q, topic="general"):
        await asyncio.sleep(0.05)
        # both queries return the same (popular) page, as Wikipedia/Tavily often do
        return [lanes.Doc(id="wiki:shared", title="Shared", text="shared fact", source="wikipedia"),
                lanes.Doc(id=f"wiki:{q}", title=q, text=q, source="wikipedia")]
    monkeypatch.setattr(lanes, "web", web)

    async def go():
        ev = []
        sp = Speculator(None, "headstart")
        missing_at_done = []

        def emit(e):
            ev.append(e)
            if e.get("state") == "done" and e.get("ids"):
                missing_at_done.extend(i for i in e["ids"] if i not in sp.vecs)     # "done" only once ranked-ready
        sp.emit = emit
        sp.on_frame(F("who founded the company acronis"))
        await asyncio.sleep(0.1)          # web returned, now inside _add_docs -> await embed
        sp.on_frame(F("who founded the company nvidia"))   # revision: stale -> cancelled mid-embed; fresh fire
        await asyncio.sleep(0.6)
        return sp, ev, missing_at_done
    sp, ev, missing = asyncio.run(go())
    old = [x for x in sp.fires if "acronis" in x.query][0]
    new = [x for x in sp.fires if "nvidia" in x.query][0]
    assert new.state == "done" and not missing
    assert old.state == "stale"                              # not overwritten with "cancelled"
    assert set(sp.pool) <= set(sp.vecs)                      # the pool never holds a vector-less doc
    assert "wiki:shared" in sp.vecs
    assert "wiki:shared" in [d.id for d, _ in sp.rank(_vec("nvidia shared"), 5, fresh_only=False)]
    assert not [e for e in ev if e.get("id") == old.id and e.get("state") == "cancelled"]


def test_concurrent_fires_share_inflight_embedding(monkeypatch):
    calls = []

    async def slow_arun(fn, *a):
        if fn is embed.encode_docs:
            calls.append(list(a[0]))
        await asyncio.sleep(0.15)
        return fn(*a)
    monkeypatch.setattr(embed, "arun", slow_arun)

    async def go():
        sp = Speculator(lambda e: None, "headstart")
        d = lambda: [lanes.Doc(id="shared", title="S", text="shared text", source="web")]   # noqa: E731
        await asyncio.gather(sp._add_docs(d()), sp._add_docs(d()))
        return sp
    sp = asyncio.run(go())
    assert "shared" in sp.vecs and "shared" in sp.pool
    assert len(calls) == 1                                   # the second fire waited for the first's embedding


# ============================================================================ 4. a failing finalize never wedges the session
def test_finalize_exception_does_not_wedge_session(monkeypatch):
    calls = {"n": 0}

    async def flaky_arun(fn, *a):
        if fn is embed.encode_queries:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("MPS/CPU embed failure")
        return fn(*a)
    monkeypatch.setattr(embed, "arun", flaky_arun)

    async def go():
        ev = []
        s = Session(ev.append, mode="baseline")
        ws = [dict(text=w, start=0.1 * i, end=0.1 * i + 0.08) for i, w in enumerate("who founded nvidia".split())]
        r1 = await s.on_final(ws)                      # used to raise and leave busy=True forever
        busy_after = s.busy
        ws2 = [dict(w, start=w["start"] + 2, end=w["end"] + 2) for w in ws]
        r2 = await s.on_final(ws2)                     # the next question is answered
        s.close()
        return r1, busy_after, r2, ev, s
    r1, busy, r2, ev, s = asyncio.run(go())
    assert r1["failed"] and "MPS/CPU" in r1["error"] and busy is False
    assert any(e["t"] == "answer_error" and "MPS/CPU" in e["error"] for e in ev)
    assert any(e["t"] == "metric" and e.get("failed") for e in ev)
    assert r2 is not None and not r2.get("failed") and r2["path"] == "search-after-endpoint"
    assert len(s.history) == 2


def test_failed_soft_answer_falls_back_to_ear(monkeypatch):
    """A soft (semantic) answer that raises is 'no soft answer': the ear's final answers normally."""
    monkeypatch.setattr(config, "SEMANTIC_ENDPOINT", True)
    monkeypatch.setattr(config, "SEM_STABLE_MS", 200)
    orig = Speculator.finalize

    async def finalize(self, f, t, close=True, by="ear"):
        if by == "semantic":
            raise RuntimeError("soft boom")
        return await orig(self, f, t, close=close, by=by)
    monkeypatch.setattr(Speculator, "finalize", finalize)
    r, ev, s = asyncio.run(_speak_with_pause("who founded the company Nvidia"))
    assert r is not None and r["endpoint_by"] == "ear" and not r.get("failed")
    assert any(e["t"] == "answer_error" and e.get("soft") for e in ev)
    assert not s.busy and len(s.history) == 1


async def _speak_with_pause(text, final=None, tail=1.0):
    ev = []
    t0 = time.perf_counter()
    s = Session(ev.append, mode="headstart", t0=t0)
    words = text.split()
    times = [(0.2 + i * 0.25, 0.2 + i * 0.25 + 0.22) for i in range(len(words))]
    end = times[-1][1]
    t = 0.0
    while t < end + tail:
        while time.perf_counter() < t0 + t:
            await asyncio.sleep(0.002)
        s.on_audio(np.ones(1600, np.float32) * (0.1 if t < end else 0.0001), t + 0.1)
        heard = [dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times) if b <= t - 0.1]
        if heard:
            s.on_partial(heard)
        t += 0.1
    fin = final(words, times) if final else [dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times)]
    r = await s.on_final(fin)
    s.close()
    return r, ev, s


# ============================================================================ 5. overlapping / orphaned soft answers
def test_superseded_soft_answer_emits_nothing(monkeypatch):
    monkeypatch.setattr(config, "SEMANTIC_ENDPOINT", True)
    monkeypatch.setattr(config, "SEM_STABLE_MS", 150)

    async def slow_web(q, topic="general"):
        WEB_CALLS.append(q)
        await asyncio.sleep(1.5)
        return [lanes.Doc(id=f"web:{q}:{i}", title=q.title(), text=f"About {q} {i}", source="fake") for i in range(3)]
    monkeypatch.setattr(lanes, "web", slow_web)

    async def go():
        ev = []
        t0 = time.perf_counter()
        s = Session(ev.append, mode="headstart", t0=t0)
        words = "what's the weather in Paris tomorrow".split()
        # pause 0.5 s after "Paris" (soft #1), then "tomorrow" (soft #2 starts while #1 is still searching)
        times = [(0.2 + i * 0.25, 0.2 + i * 0.25 + 0.22) for i in range(5)] + [(1.95, 2.2)]
        soft_tasks = []
        t = 0.0
        while t < 4.6:
            while time.perf_counter() < t0 + t:
                await asyncio.sleep(0.002)
            loud = any(a <= t <= b + 0.05 for a, b in times)
            s.on_audio(np.ones(1600, np.float32) * (0.1 if loud else 0.0001), t + 0.1)
            heard = [dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times) if b <= t - 0.1]
            if heard:
                s.on_partial(heard)
                if s.final_task is not None and s.final_task not in soft_tasks:
                    soft_tasks.append(s.final_task)
            t += 0.1
        r = await s.on_final([dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times)])
        s.close()
        await asyncio.sleep(0.5)
        return s, r, ev, soft_tasks
    s, r, ev, soft_tasks = asyncio.run(go())
    assert len(soft_tasks) >= 2
    assert all(x.done() for x in soft_tasks)
    metrics = [e for e in ev if e["t"] == "metric"]
    sources = [e for e in ev if e["t"] == "sources"]
    answers = [e for e in ev if e["t"] == "answer"]
    # one answer only: the superseded soft answer (Paris) emitted no sources, tokens or metric
    assert len(metrics) == 1 and "tomorrow" in metrics[0]["question"], [m["question"] for m in metrics]
    assert len(sources) == 1 and len(answers) == 1 and "tomorrow" in answers[0]["delta"].lower()
    assert len(s.history) == 1 and "tomorrow" in r["query"]


def test_close_cancels_unconfirmed_soft_answer(monkeypatch):
    """Session.close() while a soft answer streams: the stream is cancelled and nothing more is emitted."""
    monkeypatch.setattr(config, "SEMANTIC_ENDPOINT", True)
    monkeypatch.setattr(config, "SEM_STABLE_MS", 150)
    monkeypatch.setattr(FakeStream, "TTFT", 1.0)

    async def go():
        ev = []
        t0 = time.perf_counter()
        s = Session(ev.append, mode="headstart", t0=t0)
        words = "who founded the company Nvidia".split()
        times = [(0.2 + i * 0.25, 0.2 + i * 0.25 + 0.22) for i in range(len(words))]
        end = times[-1][1]
        t = 0.0
        while not any(e["t"] == "sources" for e in ev) and t < end + 2:
            while time.perf_counter() < t0 + t:
                await asyncio.sleep(0.002)
            s.on_audio(np.ones(1600, np.float32) * (0.1 if t < end else 0.0001), t + 0.1)
            heard = [dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times) if b <= t - 0.1]
            if heard:
                s.on_partial(heard)
            t += 0.1
        assert any(e["t"] == "sources" for e in ev), "soft answer never started"
        task = s.final_task
        streams = list(FakeStream.instances)
        s.close()
        n = len(ev)
        await asyncio.sleep(1.3)
        return ev[n:], task, streams
    after, task, streams = asyncio.run(go())
    assert task.done()
    assert all(st.done for st in streams) and any(st.cancelled for st in streams)
    assert not [e for e in after if e["t"] in ("answer", "metric", "sources", "answer_error")], after


# ============================================================================ 6. confirmation compares meaning, not raw words
def test_filler_in_final_confirms_soft_answer(monkeypatch):
    monkeypatch.setattr(config, "SEMANTIC_ENDPOINT", True)
    monkeypatch.setattr(config, "SEM_STABLE_MS", 200)

    def with_um(words, times):
        # the ear's final has the same meaning but a leading filler the partials didn't show
        return [dict(text="um", start=0.05, end=0.15)] + [dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times)]
    r, ev, s = asyncio.run(_speak_with_pause("who founded the company Nvidia", final=with_um))
    eps = [e["by"] for e in ev if e["t"] == "endpoint"]
    metrics = [e for e in ev if e["t"] == "metric"]
    assert eps == ["semantic"] and r["endpoint_by"] == "semantic" and len(metrics) == 1
    assert not any(e["t"] == "extended" for e in ev)


# ============================================================================ 7. semantic-endpoint gate = latest tagger state
@pytest.fixture(scope="module")
def tagger():
    t = Tagger(domain="assistant")
    yield t
    t.close()


def _gate_trace(tagger, text):
    """Gate per prefix, with the event bookkeeping Session uses (newest command_complete word index)."""
    tagger.reset()
    ws, spans, cc_at, out = text.split(), [], None, []
    for i, w in enumerate(ws):
        tagger.push([w])
        r = tagger.poll()
        a0 = r.get("a0", 0)
        spans = [s for s in spans if s["end"] <= a0] + r.get("spans", [])
        for e in r.get("events") or []:
            if e["type"] == "discourse.assistant.command_complete":
                cc_at = e["word"]
        f = frame.build(ws[: i + 1], spans, r.get("current"), local_ok=False)
        out.append(gate_open(f, cc_at, spans))
    return out


def test_gate_tracks_latest_state(tagger):
    assert _gate_trace(tagger, "how tall is the Eiffel Tower")[-1]
    assert _gate_trace(tagger, "who founded the company Nvidia")[-1]
    # command_complete is not sticky: it fired at "Paris", then more words came
    g = _gate_trace(tagger, "what's the weather in Paris and also in")
    assert g[4] and not any(g[5:]), g
    # act_question alone no longer opens it (it is on from the first word)
    assert not any(_gate_trace(tagger, "who founded the company Nvidia")[:3])


def test_gate_on_eval_set(tagger):
    """Over eval/questions.jsonl (see eval/gate_eval.py): the gate opens at the end of most complete questions and on
    few mid-question prefixes (the old gate: 0.98 complete, 0.90 of prefixes)."""
    rows = [json.loads(x) for x in (config.ROOT / "eval" / "questions.jsonl").read_text().splitlines() if x.strip()]
    full = early = n_pref = 0
    for row in rows:
        g = _gate_trace(tagger, row["text"])
        full += g[-1]
        early += sum(g[1:-1])
        n_pref += max(0, len(g) - 2)
    assert full / len(rows) >= 0.5, full / len(rows)
    assert early / n_pref <= 0.35, early / n_pref


def test_session_cc_cleared_on_new_words(monkeypatch):
    s = Session(lambda e: None, mode="headstart")
    ws = [dict(text=w, start=0.3 * i, end=0.3 * i + 0.25) for i, w in enumerate("what's the weather in Paris".split())]

    async def go():
        s.on_partial(ws)
        a = s.frame.command_complete
        s.on_partial(ws + [dict(text="and", start=1.6, end=1.8)])
        b = s.frame.command_complete
        s.close()
        return a, b
    a, b = asyncio.run(go())
    assert a and not b


# ============================================================================ 8 + 9. WebSocket run lifecycle, mic clock
WORDS = "who founded the company nvidia and when was it".split()


class FakeArcStream:
    def __init__(self):
        self.n, self.flushed, self.feeds_after_flush = 0, False, 0

    def feed(self, pcm):
        time.sleep(0.005)
        self.n += 1
        if self.flushed:
            self.feeds_after_flush += 1
        k = min(len(WORDS), self.n // 3)
        ws = [dict(text=w, start=0.3 * i, end=0.3 * i + 0.25) for i, w in enumerate(WORDS[:k])]
        return {"transcript_partial": {"words": ws, "text": " ".join(WORDS[:k])}} if ws else {}

    def flush(self):
        self.flushed = True
        k = min(len(WORDS), self.n // 3)
        ws = [dict(text=w, start=0.3 * i, end=0.3 * i + 0.25) for i, w in enumerate(WORDS[:k])]
        self.n = 0
        return {"transcript": [{"words": ws, "text": ""}]} if ws else {}


class FakeArc:
    streams: list = []

    def stream(self, **kw):
        s = FakeArcStream()
        FakeArc.streams.append(s)
        return s


class FakeWS:
    def __init__(self):
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent = []

    async def accept(self):
        pass

    async def receive(self):
        return await self.inbox.get()

    async def send_text(self, s):
        self.sent.append(json.loads(s))

    def text(self, obj):
        self.inbox.put_nowait({"type": "websocket.receive", "text": obj if isinstance(obj, str) else json.dumps(obj)})

    def binary(self, b):
        self.inbox.put_nowait({"type": "websocket.receive", "bytes": b})


@pytest.fixture
def ws_env(monkeypatch, tmp_path):
    FakeArc.streams = []
    monkeypatch.setattr(earmod, "arc", lambda: FakeArc())
    monkeypatch.setattr(config, "EAR", "arc")
    monkeypatch.setattr(appmod, "AUDIO", tmp_path)
    sf.write(str(tmp_path / "q.wav"), np.zeros(16000 * 4, np.float32), 16000)
    runs, stops = [], []
    orig_init, orig_do_stop = appmod.Run.__init__, appmod.Run._do_stop

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        runs.append(self)

    async def do_stop(self):
        stops.append(self)
        await orig_do_stop(self)
    monkeypatch.setattr(appmod.Run, "__init__", init)
    monkeypatch.setattr(appmod.Run, "_do_stop", do_stop)
    return runs, stops


def test_ws_stop_during_replay_bad_frames_and_disconnect(ws_env, monkeypatch):
    runs, stops = ws_env
    boom = {"n": 0}
    orig = Session._on_partial

    def flaky(self, ws, now=None):
        boom["n"] += 1
        if boom["n"] == 3:
            raise ValueError("tagger hiccup")
        return orig(self, ws, now)
    monkeypatch.setattr(Session, "_on_partial", flaky)

    async def go():
        ws = FakeWS()
        server = asyncio.create_task(appmod.ws_endpoint(ws))
        ws.text("{not json")                               # used to kill the socket
        ws.binary(b"\x00" * 6)                            # not whole float32 samples: used to kill the socket
        ws.text({"t": "replay", "file": "q.wav", "modes": ["headstart"]})
        await asyncio.sleep(1.0)
        ws.text({"t": "stop"})                            # used to leave the replay feeding a flushed ear
        await asyncio.sleep(0.6)
        replay_run = runs[0]
        feeds_after = FakeArc.streams[0].feeds_after_flush
        # mic run: the clock starts at the first audio frame, not at "start"
        ws.text({"t": "start", "modes": ["headstart"]})
        await asyncio.sleep(0.3)
        t_frame = time.perf_counter()
        ws.binary(np.zeros(1600, "<f4").tobytes())
        await asyncio.sleep(0.2)
        mic_run = runs[1]
        t0s = [(s.t0, s.spec.t0) for s in mic_run.sessions]
        ws.inbox.put_nowait({"type": "websocket.disconnect"})
        await asyncio.wait_for(server, 5)
        await asyncio.sleep(0.05)
        return ws, replay_run, mic_run, feeds_after, t_frame, t0s
    ws, replay_run, mic_run, feeds_after, t_frame, t0s = asyncio.run(go())
    errors = [e for e in ws.sent if e.get("t") == "error"]
    assert any("bad message" in e["error"] for e in errors)
    assert any("not whole float32" in e["error"] for e in errors)
    assert any(e.get("where") == "on_partial" and "hiccup" in e["error"] for e in errors)    # the run went on
    assert stops.count(replay_run) == 1                                  # stopped once (replay + stop: not twice)
    assert feeds_after == 0                                              # replay cancelled before the flush
    assert sum(1 for e in ws.sent if e.get("t") == "run_done") == 1
    for t0, spec_t0 in t0s:
        assert abs(t0 - (t_frame - 0.1)) < 0.05 and t0 == spec_t0       # sample 0 = first frame - its duration
    # disconnect: the open run is torn down (sessions closed, sender cancelled)
    assert all(s.closed and s.tagger.h is None for s in mic_run.sessions)
    assert mic_run.sender.done() and mic_run.closed


def test_run_stop_idempotent(ws_env):
    runs, stops = ws_env

    async def go():
        ws = FakeWS()
        r = appmod.Run(ws, ["baseline"], "assistant")
        await asyncio.gather(r.stop(), r.stop())
        await r.stop()
        return r, ws
    r, ws = asyncio.run(go())
    assert stops == [r] and sum(1 for e in ws.sent if e.get("t") == "run_done") == 1


# ============================================================================ 10. back-to-back questions
def test_back_to_back_questions_speculate_and_queue_fifo():
    async def go():
        ev = []
        s = Session(ev.append, mode="headstart")

        def utt(text, t0):
            return [dict(text=w, start=t0 + 0.3 * i, end=t0 + 0.3 * i + 0.25) for i, w in enumerate(text.split())]
        q1, q2, q3 = utt("who founded the company Nvidia", 0.0), utt("what is the capital of Australia", 3.0), \
            utt("how tall is Mount Everest", 6.0)
        for k in range(1, len(q1) + 1):
            s.on_partial(q1[:k])
        first = asyncio.create_task(s.on_final(q1))
        await asyncio.sleep(0.05)
        assert s.busy
        # the next question is heard while the first answer streams: it is speculated on
        for k in range(1, len(q2) + 1):
            s.on_partial(q2[:k])
            await asyncio.sleep(0.01)
        q2_fired_while_busy = any(e["t"] == "spec" and e.get("state") == "fired" and "australia" in e.get("query", "")
                                  for e in ev) and s.busy
        r2 = await s.on_final(q2)                  # queued
        for k in range(1, len(q3) + 1):
            s.on_partial(q3[:k])
        r3 = await s.on_final(q3)                  # queued behind q2 (used to overwrite it)
        r1 = await first
        s.close()
        return s, r1, r2, r3, q2_fired_while_busy
    s, r1, r2, r3, fired = asyncio.run(go())
    assert fired
    assert r2 is None and r3 is None and r1 is not None
    qs = [m["query"] for m in s.history]
    assert len(qs) == 3 and "nvidia" in qs[0] and "australia" in qs[1] and "everest" in qs[2], qs
    assert s.history[1]["path"] in ("speculative-hit", "draft-reuse"), s.history[1]["path"]
    assert not s.busy


# ============================================================================ minor cleanups
def test_export_pool_keeps_most_relevant():
    async def go():
        sp = Speculator(lambda e: None, "headstart")
        docs = [lanes.Doc(id=f"d{i}", title=f"topic{i}", text=f"word{i} filler", source="web") for i in range(20)]
        await sp._add_docs(docs)
        fire = sp._fire("web", "x", None, lambda: asyncio.sleep(0, result=[]))
        await fire.task
        fire.docs, fire.state = docs, "done"
        return sp.export_pool(k=3, qvec=_vec("topic17 word17"))
    out = asyncio.run(go())
    assert "d17" in out and len(out) == 3


def test_url_less_web_results_get_distinct_stable_ids():
    a = lanes._doc_id("tavily", "", "Title A", "text a")
    b = lanes._doc_id("tavily", "", "Title B", "text b")
    assert a != b and a == lanes._doc_id("tavily", "", "Title A", "text a") and a.startswith("tavily:")
    assert lanes._doc_id("tavily", "https://x.org/a", "T", "t") == "https://x.org/a"
