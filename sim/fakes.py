"""Fake lanes / embeddings / answer model for the policy simulator, with latencies sampled from measured distributions.

Content is REAL where it is cheap and cached (so ranking, pool-commit, draft cosine and completion checks behave like
the real system); only time is simulated:
  web     : real Wikipedia results per query, fetched once and cached in runs/sim_cache/web.json. The "tavily" profile
            reuses the same documents with Tavily-like latency (no Tavily key here).
  vectors : real EmbeddingGemma 2 text vectors (CPU text-only copy, same text tower as the server's), cached in
            runs/sim_cache/emb.pkl. Local-index docs use the index's own vectors, exactly like the server.
  local   : real local index search (index/data) with the real query vector.
  audio   : the audio lane (local-routed questions only) returns no documents (we don't embed audio offline).
  answer  : the real answer.Stream class (hedging, buffering, cancellation) over a fake Anthropic client whose
            time-to-first-token is sampled per request; the text is "<top doc title>: <its first sentence> [1]."
            (so the draft-completion check, which looks for the added words in the draft's answer, behaves
            plausibly). Drafts are never "non-answers" in the simulator.

Latency sampling uses common random numbers: a sample is a deterministic function of (seed, question id, what, key,
occurrence), so two configs that fire the same query see the same latency (variance reduction for the sweep).

Single workers are modelled as in server/embed.py: query embeddings serialize on one CPU worker; document (and audio)
embeddings serialize on one GPU worker.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import pickle
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "runs" / "sim_cache"

# ------------------------------------------------------------------------------------------------ latency profiles
# Wikipedia lane latency: bootstrap of the 155 web-lane latencies measured in runs/ab_full2.json (ms), plus jitter.
WIKI_MS = None


def _wiki_samples():
    global WIKI_MS
    if WIKI_MS is None:
        p = ROOT / "sim" / "measured_web_ms.json"
        WIKI_MS = np.array(json.loads(p.read_text())) if p.exists() else None
    return WIKI_MS


@dataclass
class Profile:
    name: str = "wikipedia"
    # web: "wikipedia" = bootstrap of measured; "tavily" = 550 + lognormal(median 350, s=0.45), 3 % tail x1.6-2.4
    web: str = "wikipedia"
    ttft_base_ms: float = 470.0          # Haiku TTFT per request: base + lognormal(median, sigma); outliers
    ttft_med_ms: float = 170.0
    ttft_sigma: float = 0.45
    ttft_outlier_p: float = 0.05
    ttft_outlier_ms: tuple = (1000.0, 2000.0)
    tok_gap_ms: float = 70.0             # between streamed chunks after the first
    qembed_ms: tuple = (38.0, 55.0)      # CPU query embedding
    qembed_tail_p: float = 0.08          # CPU contention with Arc (measured local-lane tail)
    qembed_tail_ms: tuple = (100.0, 380.0)
    dembed_base_ms: float = 40.0         # GPU doc embedding: base + per-doc, jittered (calibrated: BL endpoint->sources)
    dembed_per_doc_ms: float = 35.0
    audio_ms: tuple = (120.0, 300.0)


PROFILES = {"wikipedia": Profile("wikipedia", web="wikipedia"), "tavily": Profile("tavily", web="tavily")}


class Sampler:
    """u(...) in [0,1): deterministic per (seed, qid, what, key, occurrence)."""

    def __init__(self, seed: int, qid: str):
        self.seed, self.qid = seed, qid
        self.counts: dict = {}

    def u(self, what: str, key: str = "", n: int = 1):
        k = (what, key)
        occ = self.counts.get(k, 0)
        self.counts[k] = occ + 1
        h = hashlib.blake2b(f"{self.seed}|{self.qid}|{what}|{key}|{occ}".encode(), digest_size=8 * n).digest()
        return [struct.unpack("<Q", h[8 * i: 8 * i + 8])[0] / 2 ** 64 for i in range(n)]


def _ln(u: float, med: float, sigma: float) -> float:
    from statistics import NormalDist
    z = NormalDist().inv_cdf(min(max(u, 1e-9), 1 - 1e-9))
    return med * float(np.exp(sigma * z))


def web_ms(p: Profile, s: Sampler, query: str) -> float:
    u1, u2, u3 = s.u("web", query, 3)
    if p.web == "wikipedia":
        arr = _wiki_samples()
        if arr is not None:
            return float(arr[int(u1 * len(arr))]) * (0.92 + 0.16 * u2)
        return 150 + _ln(u1, 75, 0.45) * (2.0 if u3 < 0.03 else 1.0)
    base = 550 + _ln(u1, 350, 0.45)
    return base * (1.6 + 0.8 * u2 if u3 < 0.03 else 1.0)


def ttft_ms(p: Profile, s: Sampler, key: str) -> float:
    u1, u2 = s.u("ttft", key, 2)
    if u2 < p.ttft_outlier_p:
        lo, hi = p.ttft_outlier_ms
        return lo + (hi - lo) * u1
    return p.ttft_base_ms + _ln(u1, p.ttft_med_ms, p.ttft_sigma)


def qembed_ms(p: Profile, s: Sampler) -> float:
    u1, u2, u3 = s.u("qembed", "", 3)
    lo, hi = p.qembed_ms
    t = lo + (hi - lo) * u1
    if u2 < p.qembed_tail_p:
        a, b = p.qembed_tail_ms
        t += a + (b - a) * u3
    return t


def dembed_ms(p: Profile, s: Sampler, n: int) -> float:
    u1, = s.u("dembed", str(n))
    return min(p.dembed_base_ms + p.dembed_per_doc_ms * n, 220.0) * (0.75 + 0.5 * u1)


# ------------------------------------------------------------------------------------------------ content caches
class Caches:
    def __init__(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        self.web_path = CACHE / "web.json"
        self.emb_path = CACHE / "emb.pkl"
        self.web: dict = json.loads(self.web_path.read_text()) if self.web_path.exists() else {}
        self.emb: dict = pickle.loads(self.emb_path.read_bytes()) if self.emb_path.exists() else {}
        self.missing_web: set = set()
        self.dirty = False
        self.offline = False          # True: never compute embeddings (tests)

    def save(self):
        if self.dirty:
            self.web_path.write_text(json.dumps(self.web))
            self.emb_path.write_bytes(pickle.dumps(self.emb))
            self.dirty = False

    def _model(self):
        from server import embed
        return embed.qmodel()

    def vec_query(self, text: str) -> np.ndarray:
        k = ("q", text)
        if k not in self.emb:
            from server import embed
            self.emb[k] = embed._norm(self._model().encode([text], prompt_name="SearchQuery"))[0].astype(np.float32)
            self.dirty = True
        return self.emb[k]

    def vec_docs(self, texts, titles=None) -> np.ndarray:
        from server import embed
        titles = titles or [None] * len(texts)
        keys = [("d", ti or "", t) for t, ti in zip(texts, titles)]
        miss = [i for i, k in enumerate(keys) if k not in self.emb]
        if miss:
            fmt = [f"title: {titles[i] or 'none'} | text: {texts[i]}" for i in miss]
            vs = embed._norm(self._model().encode(fmt, batch_size=16))
            for i, v in zip(miss, vs):
                self.emb[keys[i]] = v.astype(np.float32)
            self.dirty = True
        return np.stack([self.emb[k] for k in keys])

    async def fetch_missing(self):
        """Real network: fetch every query recorded as missing (outside virtual time)."""
        from server import lanes
        qs = sorted(self.missing_web - set(self.web))
        sem = asyncio.Semaphore(4)

        async def one(q):
            async with sem:
                for attempt in range(3):
                    try:
                        docs = await lanes.wikipedia(q)
                        self.web[q] = [{"id": d.id, "title": d.title, "text": d.text, "url": d.url} for d in docs]
                        self.dirty = True
                        return
                    except Exception:
                        await asyncio.sleep(1 + attempt)
        await asyncio.gather(*(one(q) for q in qs))
        self.missing_web.clear()
        try:
            await lanes.client().aclose()
            lanes._client = None
        except Exception:
            pass
        return len(qs)


# ------------------------------------------------------------------------------------------------ cached tagger
class CachedTagger:
    """Drop-in for server.tagger.Tagger. Arc Tagger is deterministic, so a poll's result is a function of the exact
    sequence of operations since reset: memoize on that sequence and only touch a (pooled) native handle on a miss.
    The first simulated run of a question fills the cache; later configs replay it for free."""

    cache: dict = {}
    text_cache: dict = {}
    pool: dict = {}
    stats = {"hit": 0, "miss": 0}

    def __init__(self, domain: str = "assistant", **kw):
        self.domain = domain
        self.ops: list = []
        self.h = None
        self.synced = 0

    def _handle(self):
        if self.h is None:
            from server.tagger import Tagger
            lst = CachedTagger.pool.setdefault(self.domain, [])
            self.h = lst.pop() if lst else Tagger(domain=self.domain)
            self.h.reset()
            self.synced = 0
        return self.h

    def set(self, *a):
        pass

    def reset(self):
        self.ops = []
        if self.h is not None:
            self.h.reset()
        self.synced = 0

    def push(self, words, from_=None, endpoint=False):
        self.ops.append(("push", json.dumps(words), from_, endpoint))

    def endpoint(self):
        self.ops.append(("ep",))

    def poll(self) -> dict:
        key = (self.domain, tuple(self.ops))
        self.ops.append(("poll",))
        hit = CachedTagger.cache.get(key)
        if hit is not None:
            CachedTagger.stats["hit"] += 1
            return json.loads(hit)
        CachedTagger.stats["miss"] += 1
        h = self._handle()
        for op in self.ops[self.synced:-1]:
            if op[0] == "push":
                h.push(json.loads(op[1]), from_=op[2], endpoint=op[3])
            elif op[0] == "ep":
                h.endpoint()
            else:
                h.poll()
        r = h.poll()
        self.synced = len(self.ops)
        CachedTagger.cache[key] = json.dumps(r)
        return r

    def tag_text(self, text: str) -> dict:
        k = (self.domain, text)
        if k not in CachedTagger.text_cache:
            from server.tagger import Tagger
            lst = CachedTagger.pool.setdefault("text:" + self.domain, [])
            h = lst.pop() if lst else Tagger(domain=self.domain)
            CachedTagger.text_cache[k] = json.dumps(h.tag_text(text))
            lst.append(h)
        return json.loads(CachedTagger.text_cache[k])

    def close(self):
        if self.h is not None:
            CachedTagger.pool.setdefault(self.domain, []).append(self.h)
            self.h = None


# ------------------------------------------------------------------------------------------------ the fake world
@dataclass
class World:
    """Everything one simulated question run needs: profile, sampler, caches, counters."""
    profile: Profile
    sampler: Sampler
    caches: Caches
    counters: dict = field(default_factory=lambda: {"llm_requests": 0, "llm_first_tokens": 0, "web": 0,
                                                    "web_missing": 0})
    qlock: asyncio.Lock | None = None
    glock: asyncio.Lock | None = None


class _Prompt(str):
    """build_prompt result: still a str (the real client would send it), but carries the docs for the fake model."""
    question: str = ""
    docs: list = []


_SENT = re.compile(r"(?<=[.!?])\s")


def fake_answer_text(question: str, docs) -> str:
    if not docs:
        return "I couldn't find that in the sources, sorry."
    d = docs[0]
    first = _SENT.split((d.text or "").strip(), maxsplit=1)[0][:220]
    return f"{d.title}: {first} [1]."


def install(monkeypatch_setattr, world_ref: dict):
    """Patch server modules. monkeypatch_setattr(obj, name, value) (pytest's monkeypatch.setattr or a recorder).
    world_ref["w"] is the current World (set per simulated question)."""
    from server import answer, embed, lanes, session, speculator

    def W() -> World:
        return world_ref["w"]

    async def arun(fn, *args):
        w = W()
        if w.qlock is None:
            w.qlock, w.glock = asyncio.Lock(), asyncio.Lock()
        if fn is embed.encode_queries:
            async with w.qlock:
                await asyncio.sleep(qembed_ms(w.profile, w.sampler) / 1000)
                return np.stack([w.caches.vec_query(t) for t in args[0]])
        if fn is embed.encode_docs:
            texts = args[0]
            titles = args[1] if len(args) > 1 else None
            async with w.glock:
                await asyncio.sleep(dembed_ms(w.profile, w.sampler, len(texts)) / 1000)
                return w.caches.vec_docs(texts, titles)
        if fn is embed.encode_audio:
            async with w.glock:
                lo, hi = w.profile.audio_ms
                await asyncio.sleep((lo + (hi - lo) * w.sampler.u("audio")[0]) / 1000)
                return np.zeros((len(args[0]), lanes.index().tv.shape[1]), np.float32)
        return fn(*args)

    async def web(query, topic="general"):
        w = W()
        w.counters["web"] += 1
        await asyncio.sleep(web_ms(w.profile, w.sampler, query) / 1000)
        rows = w.caches.web.get(query)
        if rows is None:
            w.caches.missing_web.add(query)
            w.counters["web_missing"] += 1
            rows = []
        return [lanes.Doc(id=r["id"], title=r["title"], text=r["text"], url=r["url"],
                          source=w.profile.web) for r in rows]

    async def local_audio(wav, k=4):
        await arun(embed.encode_audio, [wav])
        return [], np.zeros(lanes.index().tv.shape[1], np.float32)

    def build_prompt(question, docs, context=""):
        p = _Prompt(f"{context}|{question}|{len(docs)}")
        p.question, p.docs = question, list(docs)
        return p

    class _Stream:
        def __init__(self, prompt):
            self.prompt = prompt

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        @property
        def text_stream(self):
            return self._gen()

        async def _gen(self):
            w = W()
            w.counters["llm_requests"] += 1
            await asyncio.sleep(ttft_ms(w.profile, w.sampler, self.prompt.question) / 1000)
            w.counters["llm_first_tokens"] += 1
            words = fake_answer_text(self.prompt.question, self.prompt.docs).split(" ")
            step = max(1, len(words) // 5)
            for i in range(0, len(words), step):
                if i:
                    await asyncio.sleep(w.profile.tok_gap_ms / 1000)
                yield " ".join(words[i:i + step]) + (" " if i + step < len(words) else "")

    class _Messages:
        def stream(self, messages=None, **kw):
            return _Stream(messages[0]["content"])

    class _Client:
        messages = _Messages()

    client = _Client()
    monkeypatch_setattr(embed, "arun", arun)
    monkeypatch_setattr(lanes, "web", web)
    monkeypatch_setattr(lanes, "local_audio", local_audio)
    monkeypatch_setattr(answer, "build_prompt", build_prompt)
    monkeypatch_setattr(answer, "client", lambda: client)
    monkeypatch_setattr(session, "Tagger", CachedTagger)
    return client
