"""The speculator: decides when to search, what to search, when to cancel, and when an early answer can be reused.

Inputs (from Session): a Frame on every ear update (about every 100 ms), the utterance audio so far, and finally the
endpoint (final frame). Output: events for the UI and the answer stream.

mode="headstart": speculative retrieval + early drafts.
mode="baseline" : wait for the endpoint, then search, rerank, answer. Same lanes, same reranker, same model.
"""
from __future__ import annotations

import asyncio
import re
import itertools
import time
from dataclasses import dataclass, field

import numpy as np

from . import answer, config, embed, lanes
from .frame import NEWS_INTENTS, Frame

_ids = itertools.count(1)


@dataclass
class Fire:
    id: str
    lane: str                  # web | local | audio | final-web | final-local
    query: str
    frame_sig: tuple
    content: tuple
    t_fire: float
    task: asyncio.Task | None = None
    state: str = "fired"       # fired | done | stale | error | cancelled
    t_done: float | None = None
    docs: list = field(default_factory=list)
    used: bool = False
    n_words: int = 0


@dataclass
class Draft:
    id: str
    meaning_sig: tuple
    question: str
    qvec: np.ndarray
    stream: answer.Stream
    t_start: float
    doc_ids: tuple
    content: tuple = ()


_NON_ANSWER = re.compile(r"\b(don't|didn't|did not|do not|doesn't|does not|no (?:information|mention)|not (?:mention|cover|"
                         r"include|say|state)|can't|cannot|isn't in|aren't in|none of)\b", re.I)


def _non_answer(text: str) -> bool:
    """A draft that says the sources lack the answer (looking at its first sentence)."""
    first = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    about_sources = re.search(r"\b(sources?|results?|provided|information|passages?|these)\b", first, re.I)
    return bool(about_sources and _NON_ANSWER.search(first))


class Speculator:
    def __init__(self, emit, mode: str = "headstart", t0: float | None = None):
        self.emit = emit
        self.mode = mode
        self.t0 = t0 or time.perf_counter()
        self.fires: list[Fire] = []
        self.fired_queries: dict[tuple, Fire] = {}
        self.pool: dict[str, lanes.Doc] = {}
        self.vecs: dict[str, np.ndarray] = {}
        self.embed_tasks: set[asyncio.Task] = set()
        self.drafts: list[Draft] = []
        self.web_used = 0
        self.local_used = 0
        self.audio_used = 0
        self.last_sig = None
        self.stable = 0
        self.sig_since = 0.0
        self.last_audio_at = 0.0
        self.draft_busy = False
        self.closed = False
        self.answer_stream = None

    # ------------------------------------------------------------------ helpers
    def ms(self, t=None):
        return round(((t or time.perf_counter()) - self.t0) * 1000)

    def _ev(self, **kw):
        self.emit({"t": "spec", **kw})

    def _fire(self, lane: str, query: str, f: Frame | None, coro_fn):
        fire = Fire(id=f"f{next(_ids)}", lane=lane, query=query, frame_sig=f.sig if f else (),
                    content=f.content if f else (), t_fire=time.perf_counter(), n_words=f.n if f else 0)
        self.fires.append(fire)
        self.fired_queries[(lane.replace("final-", ""), query)] = fire
        self._ev(id=fire.id, lane=lane, query=query, state="fired", ms=self.ms(fire.t_fire))

        async def run():
            try:
                docs = await coro_fn()
                for d in docs:
                    d.lane = lane
                fire.docs = docs
                fire.t_done = time.perf_counter()
                await self._add_docs(docs, fire)            # vectors first, then "done": drafts rank on vectors
                fire.state = "done" if fire.state == "fired" else fire.state
                self._ev(id=fire.id, lane=lane, query=query, state=fire.state, ms=self.ms(fire.t_done),
                         took=round((fire.t_done - fire.t_fire) * 1000), n=len(docs),
                         top=[d.title for d in docs[:3]], ids=[d.id for d in docs])
            except asyncio.CancelledError:
                fire.state = "cancelled"
                fire.t_done = time.perf_counter()
                self._ev(id=fire.id, lane=lane, query=query, state="cancelled", ms=self.ms(fire.t_done))
                raise
            except Exception as e:
                fire.state = "error"
                fire.t_done = time.perf_counter()
                self._ev(id=fire.id, lane=lane, query=query, state="error", ms=self.ms(fire.t_done),
                         error=f"{type(e).__name__}: {e}"[:200])
            return fire

        fire.task = asyncio.create_task(run())
        return fire

    async def _add_docs(self, docs, fire):
        new = [d for d in docs if d.id not in self.pool]
        for d in docs:
            self.pool.setdefault(d.id, d)
        need = [d for d in new if d.id not in self.vecs]
        ix = lanes.index()
        for d in list(need):
            row = ix.row_of(d.id) if d.source == "local" else None
            if row is not None:
                self.vecs[d.id] = ix.tv[row]          # the index's own text vector: no re-embedding on the GPU
                need.remove(d)
        if not need:
            return
        # local docs come with precomputed vectors in the index; re-embedding keeps one code path and is ~30 ms
        vec = await embed.arun(embed.encode_docs, [d.text[:600] for d in need], [d.title for d in need])
        for d, v in zip(need, vec):
            self.vecs[d.id] = v

    def rank(self, qvec: np.ndarray, k: int = 5, fresh_only: bool = True):
        """Top-k pool docs by cosine. Docs that only came from stale searches (a misheard or corrected-away query) are
        left out when enough fresh ones exist."""
        ids = [i for i in self.pool if i in self.vecs]
        if fresh_only:
            fresh = set()
            for x in self.fires:
                if x.state in ("done", "fired"):
                    fresh |= {d.id for d in x.docs}
            keep = [i for i in ids if i in fresh]
            if len(keep) >= 3:
                ids = keep
        if not ids:
            return []
        M = np.stack([self.vecs[i] for i in ids])
        s = M @ qvec
        order = np.argsort(-s)[:k]
        return [(self.pool[ids[i]], float(s[i])) for i in order]

    # ------------------------------------------------------------------ live updates
    def on_frame(self, f: Frame, now: float | None = None):
        if self.mode != "headstart" or self.closed:
            return
        now = now or time.perf_counter()
        if f.sig != self.last_sig:
            self.last_sig, self.stable, self.sig_since = f.sig, 1, now
        else:
            self.stable += 1
        self._mark_stale(f)
        need = 1 if f.slots else config.STABLE_UPDATES        # typed spans are confident: fire on first sight
        if f.n < config.MIN_WORDS or not f.has_content() or self.stable < need:
            return
        q = f.query()
        if not q:
            return
        topic = "news" if f.intent in NEWS_INTENTS else "general"
        if f.route in ("web", "both") and ("web", q) not in self.fired_queries and self.web_used < config.WEB_BUDGET:
            self.web_used += 1
            self._fire("web", q, f, lambda q=q, topic=topic: lanes.web(q, topic=topic))
        if (f.route in ("local", "both") and len(lanes.index()) and ("local", q) not in self.fired_queries
                and self.local_used < config.LOCAL_BUDGET):
            self.local_used += 1
            self._fire("local", q, f, lambda q=q: lanes.local_text(q))
        if (now - self.sig_since) * 1000 >= config.DRAFT_STABLE_MS and not self.draft_busy:
            self._maybe_draft(f)

    def on_audio(self, wav: np.ndarray, route: str = "both", n_words: int = 0):
        """Audio lane: embed the spoken question itself (no words needed) and search the local moments index.

        It shares the GPU with document embedding, so it runs only where it pays: questions routed to the local
        corpus (up to 3 times as more speech arrives), or once early, before the words have settled."""
        if self.mode != "headstart" or not config.AUDIO_LANE or not len(lanes.index()) or route == "web":
            return
        if route != "local":
            return          # index/data/retrieval_eval.json: <1.5 s of audio is near useless (r@1 0.0-0.1)
        if self.audio_used >= 3:
            return
        dur = len(wav) / 16000
        if dur < 1.8 or dur - self.last_audio_at < 0.8:
            return
        self.last_audio_at = dur
        self.audio_used += 1
        w = wav.copy()

        async def go():
            docs, _ = await lanes.local_audio(w)
            return docs

        self._fire("audio", f"<{dur:.1f}s of speech>", None, go)

    def _mark_stale(self, f: Frame):
        cur = set(f.content)
        for fire in self.fires:
            if fire.state == "stale" and fire.content and set(fire.content) <= cur and fire.docs:
                fire.state = "done"                 # the speaker came back to it (ASR flip-flop)
                self._ev(id=fire.id, lane=fire.lane, query=fire.query, state="done", ms=self.ms(), revived=True)
                continue
            if fire.state not in ("fired", "done") or fire.lane.startswith("final") or fire.lane == "audio":
                continue
            if fire.content and not set(fire.content) <= cur:
                was = fire.state
                fire.state = "stale"
                if was == "fired" and fire.task and not fire.task.done():
                    fire.task.cancel()
                self._ev(id=fire.id, lane=fire.lane, query=fire.query, state="stale", ms=self.ms(),
                         cancelled=was == "fired")

    def _maybe_draft(self, f: Frame):
        """Draft an answer for the current meaning as soon as its search results are in.

        Drafts whose meaning the question has since moved past are cancelled (their tokens are cheap; the time saved
        when the meaning holds is not)."""
        if any(d.meaning_sig == f.meaning_sig for d in self.drafts):
            return
        live = [d for d in self.drafts if not d.stream.done and d.meaning_sig != f.meaning_sig]
        for d in live:
            d.stream.cancel()
            self._ev(id=d.id, lane="draft", query=d.question, state="stale", ms=self.ms(), cancelled=True)
        if len(self.drafts) >= config.DRAFT_BUDGET:
            return
        q = f.query()
        mine = [x for x in self.fires if x.query == q and x.lane in ("web", "local")]
        ready = [x for x in mine if x.state == "done"]
        if not ready or any(x.state == "fired" for x in mine):
            return                                  # wait for every lane of this query (web is the slow one)
        self.draft_busy = True

        async def go():
            try:
                qvec = (await embed.arun(embed.encode_queries, [f.text]))[0]
                top = [d for d, _ in self.rank(qvec)]
                if not top or self.closed or self.last_sig != f.sig:
                    return                          # the question moved on while we embedded
                own = {d.id for x in ready for d in x.docs}
                if not own & {d.id for d in top}:
                    return                          # don't draft from other questions' results
                st = answer.Stream(f.text, top, tag="draft")
                d = Draft(id=f"d{next(_ids)}", meaning_sig=f.meaning_sig, question=f.text, qvec=qvec, stream=st,
                          t_start=time.perf_counter(), doc_ids=tuple(x.id for x in top), content=f.content)
                self.drafts.append(d)
                self._ev(id=d.id, lane="draft", query=f.text, state="fired", ms=self.ms(d.t_start))
            finally:
                self.draft_busy = False

        asyncio.create_task(go())

    # ------------------------------------------------------------------ endpoint
    async def finalize(self, f: Frame, t_speech_end: float, close: bool = True) -> dict:
        """Called when the question is over (close=True), or at a soft semantic endpoint (close=False: keep
        speculating on later words; a later finalize supersedes this answer). Streams the answer; returns metrics."""
        t_end = time.perf_counter()
        self._mark_stale(f)                       # the final words can still revise away earlier searches
        if self.answer_stream is not None:
            self.answer_stream.cancel()           # superseded soft answer
        self.closed = close
        m = {"mode": self.mode, "question": f.text, "query": f.query(), "intent": f.intent,
             "endpoint_lag_ms": round((t_end - t_speech_end) * 1000)}
        q = f.query() or " ".join(f.words)
        qtext = f.text
        topic = "news" if f.intent in NEWS_INTENTS else "general"
        qvec_task = asyncio.create_task(embed.arun(embed.encode_queries, [qtext]))
        reuse_draft = None
        if self.mode == "baseline":
            waits = []
            if f.route in ("web", "both"):
                waits.append(self._fire("final-web", q, f, lambda: lanes.web(q, topic=topic)).task)
            if f.route in ("local", "both") and len(lanes.index()):
                waits.append(self._fire("final-local", q, f, lambda: lanes.local_text(q)).task)
            await asyncio.gather(*waits, return_exceptions=True)
            qvec = (await qvec_task)[0]
            m["path"] = "search-after-endpoint"
        else:
            qvec = (await qvec_task)[0]
            # 1) a draft that answers this question? exact meaning, or a shorter question whose answer already
            #    contains every word the speaker added after it ("how tall is the Eiffel" -> "... Eiffel Tower")
            reuse_draft = await self._pick_draft(f, qvec, m)
            if reuse_draft is None:
                # 2) the final query already fired (in flight or done)? wait for it, don't re-search
                same = [x for x in self.fires if x.query == q and x.lane in ("web", "local")
                        and x.state in ("fired", "done", "stale")]
                pending = [x.task for x in same if x.state == "fired" and x.task]
                top = self.rank(qvec, 1)
                best = top[0][1] if top else 0.0
                m["pool_best"] = round(best, 3)
                if same:
                    m["path"] = "speculative-hit"
                    if pending:
                        # don't let a slow lane hold the answer hostage: answer with what has landed after the cap
                        await asyncio.wait(pending, timeout=config.PENDING_CAP_S)
                elif best >= config.COMMIT_TAU and self._nothing_new(f):
                    m["path"] = "pool-commit"
                else:
                    m["path"] = "speculative-miss"
                    waits = []
                    if f.route in ("web", "both") and self.web_used < config.WEB_BUDGET + 1:
                        waits.append(self._fire("final-web", q, f, lambda: lanes.web(q, topic=topic)).task)
                    if f.route in ("local", "both") and len(lanes.index()):
                        waits.append(self._fire("final-local", q, f, lambda: lanes.local_text(q)).task)
                    await asyncio.gather(*waits, return_exceptions=True)
            else:
                m["path"] = "draft-reuse"
        ranked = self.rank(qvec, 5)
        if reuse_draft is not None:
            st = reuse_draft.stream
            docs = [self.pool[i] for i in reuse_draft.doc_ids if i in self.pool]
            dv = {i: float(self.vecs[i] @ qvec) for i in reuse_draft.doc_ids if i in self.vecs}
            ranked = [(d, dv.get(d.id, 0.0)) for d in docs]      # the draft's own numbering = its citations
            self._ev(id=reuse_draft.id, lane="draft", query=reuse_draft.question, state="reused", ms=self.ms())
        else:
            docs = [d for d, _ in ranked]
            st = answer.Stream(qtext, docs, tag="final")
        t_sources = time.perf_counter()
        m["n_sources"] = len(docs)
        self.emit({"t": "sources", "ms": self.ms(t_sources),
                   "docs": [{**d.brief(), "score": round(sc, 3)} for d, sc in ranked]})
        self.answer_stream = st
        used_ids = {d.id for d in docs}
        for fire in self.fires:
            fire.used = any(d.id in used_ids for d in fire.docs)
        self.emit({"t": "used", "fires": [x.id for x in self.fires if x.used], "ms": self.ms()})
        if close:
            for d in self.drafts:
                if d is not reuse_draft:
                    d.stream.cancel()
        t_first = None
        async for tok in st.follow():
            if t_first is None:
                t_first = time.perf_counter()
            self.emit({"t": "answer", "delta": tok, "ms": self.ms()})
        t_done = time.perf_counter()
        if st.error:
            self.emit({"t": "answer_error", "error": st.error})
        spec = [x for x in self.fires if not x.lane.startswith("final")]
        used_spec = [x for x in spec if x.used]
        if spec:
            m["first_fire_lead_ms"] = round((t_speech_end - min(x.t_fire for x in spec)) * 1000)
            m["first_fire_words"] = min((x.n_words for x in spec), default=None)
        if used_spec:
            m["used_fire_lead_ms"] = round((t_speech_end - min(x.t_fire for x in used_spec)) * 1000)
        m.update({
            "sources_ms": round((t_sources - t_speech_end) * 1000),
            "ready_ms": round(((t_first or t_done) - t_speech_end) * 1000),
            "done_ms": round((t_done - t_speech_end) * 1000),
            "web_calls": sum(1 for x in self.fires if x.lane in ("web", "final-web")),
            "spec_fires": len(spec),
            "spec_used": sum(1 for x in spec if x.used),
            "stale": sum(1 for x in self.fires if x.state == "stale"),
            "drafts": len(self.drafts),
            "draft_reused": reuse_draft is not None,
            "answer": st.text(),
            "sources": [d.title for d in docs],
            "error": st.error,
        })
        self.emit({"t": "metric", **{k: v for k, v in m.items() if k != "answer"}})
        return m

    def _nothing_new(self, f: Frame) -> bool:
        """True if every content word of the final question was already covered by some non-stale search."""
        seen = set()
        for x in self.fires:
            if x.state in ("done", "fired") and x.lane in ("web", "local"):
                seen |= set(x.content)
        return set(f.content) <= seen

    async def _pick_draft(self, f: Frame, qvec, m):
        from .frame import norm
        best = None
        for d in reversed(self.drafts):
            if d.stream.error is not None:
                continue
            if d.stream.done and _non_answer(d.stream.text()):
                m["draft_rejected_nonanswer"] = True
                continue                    # "the sources don't mention X": answer again from the final ranking
            cos = float(d.qvec @ qvec)
            if d.meaning_sig == f.meaning_sig and cos >= 0.9:
                m["draft_cos"], m["draft_match"] = round(cos, 3), "exact"
                return d
            if cos < 0.85 or best is not None:
                continue
            missing = [w for w in f.content if w not in set(d.content)]
            gone = [w for w in d.content if w not in set(f.content)]
            if gone or not missing or len(missing) > 2:
                continue
            best = (d, cos, missing)
        if best is None:
            return None
        d, cos, missing = best
        # verify against the draft's own answer: wait for its first sentence (bounded)
        deadline = time.perf_counter() + 0.6
        while not d.stream.done and time.perf_counter() < deadline:
            txt = d.stream.text()
            if any(p in txt for p in ".!?") and len(txt) > 30:
                break
            await asyncio.sleep(0.02)
        if _non_answer(d.stream.text()):
            m["draft_rejected_nonanswer"] = True
            return None
        ans = set(norm(d.stream.text()).replace("-", " ").split())
        if all(w in ans or any(a.startswith(w) for a in ans) for w in missing):
            m["draft_cos"], m["draft_match"], m["draft_covered"] = round(cos, 3), "completion", missing
            return d
        m["draft_rejected"] = missing
        return None

    def close(self):
        self.closed = True
        for x in self.fires:
            if x.task and not x.task.done():
                x.task.cancel()
        for d in self.drafts:
            d.stream.cancel()
