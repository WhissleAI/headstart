"""Retrieval lanes. Each returns a list of Doc; all are async and cancellable.

web   : Tavily when TAVILY_API_KEY is set (depth/topic/freshness per question, credit ledger + cache), else Wikipedia
        (keyless, real network latency).
local : EmbeddingGemma 2 index over a local corpus (text, and audio moments), queried by text or by an audio vector.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field

import httpx
import numpy as np

from . import config, embed

UA = {"User-Agent": "HeadStartDemo/0.1 (https://github.com/; hackathon prototype)"}


@dataclass
class Doc:
    id: str
    title: str
    text: str
    url: str = ""
    lane: str = ""
    source: str = ""          # tavily | wikipedia | local
    t: float | None = None    # local moment start (s), if any
    score: float = 0.0        # lane score
    meta: dict = field(default_factory=dict)

    def brief(self, n=260):
        return {"id": self.id, "title": self.title, "text": self.text[:n], "url": self.url, "source": self.source,
                "t": self.t, "lane": self.lane}


_client: httpx.AsyncClient | None = None
_transport: httpx.AsyncBaseTransport | None = None      # tests: httpx.MockTransport (fake Tavily)


def _http2_ok() -> bool:
    try:
        import h2  # noqa: F401
        return True
    except ImportError:
        return False


def client() -> httpx.AsyncClient:
    """One pooled client for every web call: HTTP/2 when h2 is installed (Tavily's concurrent speculative calls
    multiplex on one TLS connection), keep-alive held for 2 min so the next question doesn't pay the handshake."""
    global _client
    if _client is None:
        kw = {"transport": _transport} if _transport is not None else {"http2": _http2_ok()}
        _client = httpx.AsyncClient(timeout=httpx.Timeout(6.0, connect=3.0), headers=UA,
                                    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10,
                                                        keepalive_expiry=120), **kw)
    return _client


def use_transport(transport: httpx.AsyncBaseTransport | None):
    """Route all web calls through `transport` (tests); None restores the network."""
    global _client, _transport
    _client, _transport = None, transport


async def warm_web():
    """Open keep-alive connections so the first speculative call doesn't pay TLS setup."""
    try:
        if config.WEB_BACKEND == "tavily":
            await client().get(config.TAVILY_BASE_URL + "/")
        else:
            await client().get("https://en.wikipedia.org/w/api.php", params={"action": "query", "format": "json"})
    except Exception:
        pass


_TAG = re.compile(r"<[^>]+>")


async def wikipedia(query: str, k: int = 4) -> list[Doc]:
    params = {"action": "query", "format": "json", "formatversion": "2", "generator": "search", "gsrsearch": query,
              "gsrlimit": str(k), "prop": "extracts|info", "inprop": "url", "exintro": "1", "explaintext": "1",
              "exchars": "1200", "exlimit": str(k), "gsrqiprofile": "classic_noboostlinks"}
    r = await client().get("https://en.wikipedia.org/w/api.php", params=params)
    r.raise_for_status()
    pages = (r.json().get("query") or {}).get("pages") or []
    pages.sort(key=lambda p: p.get("index", 99))
    out = []
    for p in pages:
        text = (p.get("extract") or "").strip()
        if not text:
            continue
        out.append(Doc(id=p.get("fullurl") or f"wiki:{p.get('pageid')}", title=p.get("title", ""), text=text,
                       url=p.get("fullurl", ""), source="wikipedia"))
    return out


# ----------------------------------------------------------------------------------------------------------- Tavily
# Verified against https://docs.tavily.com/documentation/api-reference/endpoint/search (2026-11-12):
#   search_depth ultra-fast | fast | basic (1 credit) | advanced (2 credits); chunks_per_source 1-3 (not ultra-fast);
#   topic general | news | finance; time_range day|week|month|year; max_age_hours -1..720 (weather / prices: their
#   pages keep old dates, so time_range would drop them); include_usage → usage.credits; auto_parameters may raise
#   depth to advanced (2 credits) → we send everything explicitly and keep it off.
#   Errors: 400/401/422, 429 (+Retry-After), 432 plan limit, 433 pay-as-you-go limit, 434 flex capacity (only with
#   max_age_hours: retry without it), 500.
COST = {"ultra-fast": 1, "fast": 1, "basic": 1, "advanced": 2}
NEWS = {"discourse.intent.news_query"}
FINANCE = {"discourse.intent.qa_stock", "discourse.intent.qa_currency"}
WEATHER = {"discourse.intent.weather_query"}
_DAY = re.compile(r"\b(today|tonight|this (?:morning|afternoon|evening)|right now|breaking|just now|at the moment)\b")
_WEEK = re.compile(r"\b(latest|yesterday|last night|this week(?:end)?|recent(?:ly)?|lately|past (?:few )?days|"
                   r"over the weekend|currently|so far)\b")


class TavilyError(Exception):
    def __init__(self, status: int, msg: str = "", retry_after: float | None = None):
        super().__init__(f"tavily {status}: {msg}"[:200])
        self.status, self.retry_after = status, retry_after


class WebUnavailable(Exception):
    """Tavily not callable right now (session/process credit budget, or the breaker after 401/429/432/433)."""


class WebResult(list):
    """list[Doc] plus what the call cost."""
    answer: str | None = None
    credits: float = 0.0
    cached: bool = False
    backend: str = ""
    response_time: float | None = None
    request_id: str | None = None


def web_plan(intent: str = "", text: str = "", final: bool = False, answer: str | bool = False) -> dict:
    """Tavily parameters for one question: depth by speculative/final, topic + freshness by intent."""
    depth = config.TAVILY_FINAL_DEPTH if final else config.TAVILY_SPEC_DEPTH
    p = {"search_depth": depth, "topic": "general", "max_results": config.TAVILY_MAX_RESULTS}
    low = (text or "").lower()
    if intent in NEWS:
        p["topic"] = "news"
        if _DAY.search(low):
            p["time_range"] = "day"
        elif _WEEK.search(low):
            p["time_range"] = "week"
    elif intent in FINANCE:
        p["topic"] = "finance"
        if config.TAVILY_FINANCE_MAX_AGE_H > 0:
            p["max_age_hours"] = config.TAVILY_FINANCE_MAX_AGE_H
    elif intent in WEATHER and config.TAVILY_WEATHER_MAX_AGE_H > 0:
        p["max_age_hours"] = config.TAVILY_WEATHER_MAX_AGE_H
    if answer:
        p["include_answer"] = answer
    return p


def tavily_body(query: str, plan: dict) -> dict:
    depth = plan["search_depth"]
    body = {"query": query[:400], "search_depth": depth, "topic": plan["topic"], "max_results": plan["max_results"],
            "include_answer": plan.get("include_answer", False), "include_raw_content": False,
            "include_images": False, "include_usage": True, "auto_parameters": False}
    if depth != "ultra-fast":
        body["chunks_per_source"] = config.TAVILY_CHUNKS
    for k in ("time_range", "max_age_hours"):
        if plan.get(k) is not None:
            body[k] = plan[k]
    return body


def norm_query(q: str) -> str:
    """Cache key text: lowercased content words, deduped and sorted (frame queries are bags of words; the carried
    entity of a follow-up can come first or last)."""
    return " ".join(sorted(set(re.sub(r"[^\w\s]", " ", q.lower()).split())))


def _cache_key(query: str, body: dict) -> tuple:
    return (norm_query(query), body["search_depth"], body["topic"], body.get("time_range"), body.get("max_age_hours"),
            body["max_results"], body.get("chunks_per_source"), str(body.get("include_answer")))


_PROCESS = {"spent": 0.0, "blocked_until": 0.0, "blocked_why": ""}
_GLOBAL_CACHE: dict = {}


class Ledger:
    """Per-session Tavily account: credit budget (speculation can't spend the final reserve), response cache,
    in-flight de-duplication, counters for the metric events."""

    def __init__(self, budget: float | None = None, reserve: float | None = None):
        self.budget = config.TAVILY_SESSION_CREDITS if budget is None else budget
        self.reserve = config.TAVILY_FINAL_RESERVE if reserve is None else reserve
        self.spent = 0.0
        self.calls = 0
        self.cache_hits = 0
        self.denied = 0
        self.errors = 0
        self._cache: dict = {}
        self.inflight: dict = {}

    @property
    def cache(self) -> dict:
        return _GLOBAL_CACHE if config.TAVILY_CACHE_SCOPE == "global" else self._cache

    def allow(self, cost: float, final: bool) -> str | None:
        """None if the call may go out, else why not."""
        if time.monotonic() < _PROCESS["blocked_until"]:
            return f"blocked: {_PROCESS['blocked_why']}"
        if config.TAVILY_PROCESS_CREDITS > 0 and _PROCESS["spent"] + cost > config.TAVILY_PROCESS_CREDITS:
            return "process credit budget"
        if self.budget > 0:
            limit = self.budget if final else self.budget - self.reserve
            if self.spent + cost > limit:
                return "session credit budget"
        return None

    def charge(self, credits: float):
        self.spent += credits
        _PROCESS["spent"] += credits

    def summary(self) -> dict:
        return {"spent": self.spent, "budget": self.budget, "calls": self.calls, "cache_hits": self.cache_hits,
                "denied": self.denied, "errors": self.errors}


def _block(seconds: float, why: str):
    _PROCESS["blocked_until"] = max(_PROCESS["blocked_until"], time.monotonic() + seconds)
    _PROCESS["blocked_why"] = why


def reset_process():
    """Tests: forget process-wide spend, breaker and global cache."""
    _PROCESS.update(spent=0.0, blocked_until=0.0, blocked_why="")
    _GLOBAL_CACHE.clear()


async def _post(body: dict) -> dict:
    """POST /search with a hard total deadline (httpx timeouts are per phase: connect, each read...)."""
    try:
        r = await asyncio.wait_for(
            client().post(config.TAVILY_BASE_URL + "/search", json=body, timeout=config.TAVILY_TIMEOUT_S,
                          headers={"Authorization": f"Bearer {config.TAVILY_API_KEY}"}),
            timeout=config.TAVILY_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise TavilyError(0, f"timeout after {config.TAVILY_TIMEOUT_S:g}s") from None
    if r.status_code != 200:
        ra = r.headers.get("retry-after")
        try:
            detail = r.json().get("detail")
            msg = detail.get("error") if isinstance(detail, dict) else str(detail)
        except Exception:
            msg = r.text[:120]
        raise TavilyError(r.status_code, msg or "", float(ra) if ra and ra.replace(".", "", 1).isdigit() else None)
    return r.json()


def _parse(data: dict) -> WebResult:
    out = WebResult()
    for x in data.get("results") or []:
        title = x.get("title") or ""
        pub = x.get("published_date")
        if pub:
            title = f"{title} ({str(pub)[:16]})"       # recency visible to the answer model (news)
        text = (x.get("content") or "").strip()
        out.append(Doc(id=_doc_id("tavily", x.get("url") or "", x.get("title") or "", text), title=title, text=text,
                       url=x.get("url", ""), source="tavily", score=float(x.get("score") or 0),
                       meta={"published_date": pub} if pub else {}))
    out.answer = data.get("answer") or None
    out.response_time = data.get("response_time")
    out.request_id = data.get("request_id")
    return out


def _doc_id(source: str, url: str, title: str, text: str) -> str:
    """The URL, or (no URL) a stable hash of the content: url-less results must not all collapse onto id ""."""
    if url:
        return url
    return f"{source}:" + hashlib.sha1(f"{title}\n{text}".encode()).hexdigest()[:16]


def _copy(res: WebResult, cached: bool) -> WebResult:
    out = WebResult(Doc(**{**d.__dict__, "meta": dict(d.meta)}) for d in res)
    out.answer, out.response_time, out.request_id = res.answer, res.response_time, res.request_id
    out.backend, out.cached, out.credits = "tavily", cached, (0.0 if cached else res.credits)
    return out


async def _fetch(ledger: Ledger, key: tuple, body: dict, cost: float, meter: dict) -> WebResult:
    """The one real request for `key` (shared by every waiter; survives a waiter's cancellation so its credit isn't
    wasted: the result lands in the cache for a flip-back or the final query)."""
    try:
        try:
            data = await _post(body)
        except TavilyError as e:
            if e.status == 434 and "max_age_hours" in body:
                body = {k: v for k, v in body.items() if k != "max_age_hours"}     # docs: omit it → served from cache
                data = await _post(body)
            else:
                raise
        res = _parse(data)
        used = (data.get("usage") or {}).get("credits")
        actual = float(used) if isinstance(used, (int, float)) else cost
        ledger.charge(actual - cost)             # reconcile the estimate with what Tavily says it billed
        res.credits = actual
        meter["credits"] = actual
        ttl = (config.TAVILY_FRESH_TTL_S if body["topic"] != "general" or body.get("max_age_hours") is not None
               or body.get("time_range") else config.TAVILY_CACHE_TTL_S)
        ledger.cache[key] = (time.monotonic() + ttl, res)
        return res
    except BaseException as e:
        ledger.charge(-cost)                     # failed requests aren't billed
        meter["credits"] = 0.0
        if isinstance(e, Exception):
            ledger.errors += 1
        if isinstance(e, TavilyError):
            if e.status == 401:
                _block(600, "401 invalid key")
            elif e.status == 429:
                _block(e.retry_after or 10, "429 rate limited")
            elif e.status in (432, 433):
                _block(900, f"{e.status} credit limit")
        raise
    finally:
        ledger.inflight.pop(key, None)


async def tavily(query: str, plan: dict | None = None, ledger: Ledger | None = None, meter: dict | None = None,
                 final: bool = False) -> WebResult:
    plan = plan or web_plan(final=final)
    ledger = ledger if ledger is not None else Ledger()
    meter = meter if meter is not None else {}
    body = tavily_body(query, plan)
    key = _cache_key(query, body)
    meter.update(backend="tavily", depth=body["search_depth"], topic=body["topic"], credits=0.0, cached=False)
    hit = ledger.cache.get(key)
    if hit and hit[0] > time.monotonic():
        ledger.cache_hits += 1
        meter["cached"] = True
        return _copy(hit[1], cached=True)
    task = ledger.inflight.get(key)
    if task is not None:                         # same query already on the wire: join it, pay nothing
        ledger.cache_hits += 1
        meter["cached"] = True
        return _copy(await asyncio.shield(task), cached=True)
    cost = COST.get(body["search_depth"], 2)
    why = ledger.allow(cost, final)
    if why:
        ledger.denied += 1
        raise WebUnavailable(why)
    ledger.charge(cost)
    ledger.calls += 1
    meter["credits"] = cost
    task = asyncio.create_task(_fetch(ledger, key, body, cost, meter))
    ledger.inflight[key] = task
    return _copy(await asyncio.shield(task), cached=False)


async def web(query: str, topic: str = "general", *, intent: str = "", text: str = "", final: bool = False,
              ledger: Ledger | None = None, meter: dict | None = None) -> list[Doc]:
    """The web lane. Tavily when configured (topic/freshness from the intent), else Wikipedia. A final query whose
    Tavily call can't go out or fails falls back to Wikipedia (the user is waiting); a speculative one just errors
    (the fire shows as an error; the session carries on)."""
    meter = meter if meter is not None else {}
    if config.WEB_BACKEND == "tavily":
        plan = web_plan(intent or ("discourse.intent.news_query" if topic == "news" else ""), text, final)
        try:
            return await tavily(query, plan, ledger, meter, final=final)
        except (WebUnavailable, TavilyError, httpx.HTTPError) as e:
            meter["fallback"] = f"{type(e).__name__}: {e}"[:120]
            if not final:
                raise
    meter.setdefault("credits", 0.0)
    meter["backend"] = "wikipedia"
    res = WebResult(await wikipedia(query))
    res.backend = "wikipedia"
    return res


class LocalIndex:
    """A small in-memory index: docs.jsonl + text vectors (+ optional audio vectors) saved by index/build_index.py."""

    def __init__(self, path=None):
        path = path or config.INDEX_DIR
        self.docs: list[Doc] = []
        self.tv = np.zeros((0, config.EMBED_DIM), np.float32)
        self.av = None
        self.av_rows = None
        meta = path / "docs.jsonl"
        if meta.exists():
            for line in meta.read_text().splitlines():
                d = json.loads(line)
                self.docs.append(Doc(id=d["id"], title=d.get("title", ""), text=d["text"], url=d.get("url", ""),
                                     source="local", t=d.get("t"), meta=d.get("meta", {})))
            z = np.load(path / "vectors.npz")
            self.tv = z["text"][:, : config.EMBED_DIM]
            if "audio" in z.files and len(z["audio"]):
                self.av = z["audio"][:, : config.EMBED_DIM]
                self.av_rows = z["audio_rows"]

    def __len__(self):
        return len(self.docs)

    def row_of(self, doc_id: str):
        if not hasattr(self, "_rows"):
            self._rows = {d.id: i for i, d in enumerate(self.docs)}
        return self._rows.get(doc_id)

    def audio_row_of(self, row: int):
        """Row in the audio vectors for docs.jsonl row `row`, or None."""
        if self.av_rows is None:
            return None
        if not hasattr(self, "_arows"):
            self._arows = {int(r): j for j, r in enumerate(self.av_rows)}
        return self._arows.get(int(row))

    def search_vec(self, q: np.ndarray, k: int = 4, use_audio: bool = True) -> list[Doc]:
        if not len(self.docs):
            return []
        s = self.tv @ q
        if use_audio and self.av is not None:
            sa = self.av @ q
            for row, v in zip(self.av_rows, sa):
                s[row] = max(s[row], v)
        top = np.argsort(-s)[:k]
        out = []
        for i in top:
            d = self.docs[int(i)]
            out.append(Doc(**{**d.__dict__, "score": float(s[i])}))
        return out


_index: LocalIndex | None = None


def index() -> LocalIndex:
    global _index
    if _index is None:
        _index = LocalIndex()
    return _index


async def local_text(query: str, k: int = 4) -> list[Doc]:
    if not len(index()):
        return []
    q = (await embed.arun(embed.encode_queries, [query]))[0]
    return index().search_vec(q, k)


async def local_audio(wav: np.ndarray, k: int = 4) -> tuple[list[Doc], np.ndarray]:
    q = (await embed.arun(embed.encode_audio, [wav]))[0]
    return (index().search_vec(q, k) if len(index()) else []), q


async def timed(coro):
    t = time.perf_counter()
    res = await coro
    return res, (time.perf_counter() - t) * 1000
