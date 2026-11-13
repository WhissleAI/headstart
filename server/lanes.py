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
                "t": self.t, "lane": self.lane, **({"cached": True} if self.meta.get("cached") else {})}


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
    """Open keep-alive connections so the first speculative call doesn't pay TLS setup; records whether the live
    web lane is reachable (WEB_STATUS, reported by /api/health and the ready event)."""
    try:
        await probe_web()
    except Exception:
        pass


_TAG = re.compile(r"<[^>]+>")


async def _wiki_search(query: str, k: int) -> list[Doc]:
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


def relaxed(query: str) -> str:
    """The query minus its first word. Wikipedia ANDs every word, so one misheard or filler word at the front
    ("repainted mona lysa", "ivan wanted ...") returns nothing; the rest usually still names the thing."""
    ws = query.split()
    return " ".join(ws[1:]) if len(ws) >= 3 else ""


async def wikipedia(query: str, k: int = 4, entity: str = "", k_entity: int = 2) -> list[Doc]:
    """Content query, plus (in parallel) an entity-only query whose top articles are appended: "capital turkey"
    finds "Capital punishment in Turkey" while "turkey" finds Turkey, whose intro names Ankara. The reranker then
    picks from both. A content query with no hits is retried once, relaxed (see `relaxed`)."""
    ent = entity.strip()
    if ent and ent != query.strip():
        main, extra = await asyncio.gather(_wiki_search(query, k), _wiki_search(ent, k_entity),
                                           return_exceptions=True)
        if isinstance(main, BaseException):
            if isinstance(extra, BaseException):
                raise main
            main = []
        extra = [] if isinstance(extra, BaseException) else extra
    else:
        main, extra = await _wiki_search(query, k), []
    if not main and not extra and relaxed(query):
        main = await _wiki_search(relaxed(query), k)
    seen = {d.id for d in main}
    return main + [d for d in extra if d.id not in seen]


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


def web_plan(intent: str = "", text: str = "", final: bool = False, answer: str | bool = False,
             hints: dict | None = None) -> dict:
    """Tavily parameters for one question: depth by speculative/final, topic + freshness by intent, then by the rest of
    Arc Tagger's read-out (`hints` = Frame.web_hints(): typed time / money spans, result words) when given."""
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
    if hints and config.TAGGER_HINTS:
        topic = hints.get("topic")
        if hints.get("weather_off"):
            p.pop("max_age_hours", None)
        if topic == "finance" and p["topic"] != "finance":
            p = {k: v for k, v in p.items() if k != "time_range"}
            p["topic"] = "finance"
            if config.TAVILY_FINANCE_MAX_AGE_H > 0:
                p["max_age_hours"] = config.TAVILY_FINANCE_MAX_AGE_H
        elif topic == "news":
            p["topic"] = "news"
            p.pop("max_age_hours", None)
            if hints.get("time_range"):
                p["time_range"] = hints["time_range"]
    if p["topic"] == "finance" and p["search_depth"] in ("fast", "ultra-fast"):
        # Tavily answers 400 "Finance topic is not supported for fast or ultra-fast search_depth": every qa_stock /
        # qa_currency question used to fail over to Wikipedia. basic costs the same 1 credit (runs/replay_finance.json:
        # grounded 4/5 -> 5/5, ~+1.3 s per call, which speculation hides on the Arc side)
        p["search_depth"] = config.TAVILY_FINANCE_DEPTH
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


# ------------------------------------------------------------------------------------------- web status + cache
# Honesty rule: a result served from the recorded cache is never presented as live. Every such call sets
# meter["backend"] = "cache" and meter["replayed"] = True; the metric counts them (web_replayed) and the UI flags them.
WEB_STATUS = {"status": "unknown", "last_error": None, "last_ok": None, "down_until": 0.0, "probed_at": None,
              "fails": 0}


def _web_ok():
    WEB_STATUS.update(status="live", last_ok=time.time(), down_until=0.0, fails=0)


def _web_fail(e: BaseException):
    """Open the breaker only after WEB_BREAKER_FAILS consecutive network failures of the PRIMARY backend: one slow
    call (or a timeout of the Wikipedia fallback while Tavily is healthy) must not put the whole demo on the cache."""
    WEB_STATUS["fails"] = WEB_STATUS.get("fails", 0) + 1
    WEB_STATUS["last_error"] = f"{type(e).__name__}: {e}"[:160]
    if WEB_STATUS["fails"] >= config.WEB_BREAKER_FAILS:
        WEB_STATUS.update(status="down", down_until=time.monotonic() + config.WEB_BREAKER_S)


def web_down() -> bool:
    """The live lane failed recently (breaker open): skip it and go straight to the cache."""
    return time.monotonic() < WEB_STATUS["down_until"]


def web_health() -> dict:
    c = web_cache()
    st = "cache" if config.WEB_BACKEND == "cache" else WEB_STATUS["status"]
    return {"backend": config.WEB_BACKEND, "status": st, "live": st == "live",
            "fallback": "cache" if (st in ("cache", "down") and len(c)) else ("none" if st in ("cache", "down") else ""),
            "last_error": WEB_STATUS["last_error"], "cache_entries": len(c), "cache_path": str(c.path),
            "cache_recorded": c.recorded}


def reset_web_status():
    """Tests."""
    WEB_STATUS.update(status="unknown", last_error=None, last_ok=None, down_until=0.0, probed_at=None, fails=0)


class WebCache:
    """Recorded web-lane responses: normalized query -> docs. Exact hit on the normalized query, else the entry that
    covers the most of the query's words (>= WEB_CACHE_MIN_COVER of them; ties: fewest extra words)."""

    def __init__(self, path):
        self.path = path
        self.entries: dict[str, dict] = {}
        self.recorded: str | None = None
        try:
            data = json.loads(path.read_text())
            self.entries = data.get("entries") or {}
            self.recorded = data.get("recorded")
        except (OSError, ValueError):
            pass

    def __len__(self):
        return len(self.entries)

    def get(self, query: str) -> tuple[list[Doc], str, str] | None:
        """(docs, match "exact"|"fuzzy", the cached query) or None."""
        key = norm_query(query)
        hit, match = self.entries.get(key), "exact"
        if hit is None:
            qw = set(key.split())
            if not qw:
                return None
            best = None
            for k, e in self.entries.items():
                cw = set(k.split())
                cover = len(qw & cw) / len(qw)
                if cover < config.WEB_CACHE_MIN_COVER:
                    continue
                score = (cover, -len(cw - qw))
                if best is None or score > best[0]:
                    best = (score, e)
            if best is None:
                return None
            hit, match = best[1], "fuzzy"
        docs = [Doc(id=d.get("id") or d.get("url") or "", title=d.get("title", ""), text=d.get("text", ""),
                    url=d.get("url", ""), source=d.get("source") or hit.get("backend", ""),
                    meta={"cached": True, "recorded_at": hit.get("recorded_at")}) for d in hit.get("docs", [])]
        return docs, match, hit.get("query", "")

    def put(self, query: str, docs, backend: str):
        self.entries[norm_query(query)] = {
            "query": query, "backend": backend, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "docs": [{"id": d.id, "title": d.title, "text": d.text, "url": d.url, "source": d.source} for d in docs]}

    def save(self):
        self.recorded = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": 1, "recorded": self.recorded, "entries": self.entries}, indent=1))
        tmp.replace(self.path)


_cache: WebCache | None = None


def web_cache() -> WebCache:
    global _cache
    if _cache is None or _cache.path != config.WEB_CACHE_PATH:
        _cache = WebCache(config.WEB_CACHE_PATH)
    return _cache


def set_web_cache(c: WebCache | None):
    """Tests: use this cache (None: reload from WEB_CACHE_PATH on next use)."""
    global _cache
    _cache = c


def _from_cache(query: str, meter: dict, why: str | None) -> "WebResult":
    hit = web_cache().get(query)
    meter.update(backend="cache", replayed=True, credits=0.0)
    if why:
        meter["fallback"] = why[:120]
    res = WebResult()
    res.backend = "cache"
    if hit is None:
        meter["cache_miss"] = True
        return res
    docs, match, cq = hit
    meter["cache_match"] = match
    if match == "fuzzy":
        meter["cache_query"] = cq
    res.extend(docs)
    return res


async def _live(query: str, topic: str, intent: str, text: str, final: bool, ledger, meter: dict,
                entity: str = "", hints: dict | None = None) -> list[Doc]:
    """The live lane: Tavily when configured (topic/freshness from the intent), else Wikipedia. A final query whose
    Tavily call can't go out or fails falls back to Wikipedia (the user is waiting); a speculative one just errors."""
    if config.WEB_BACKEND == "tavily":
        plan = web_plan(intent or ("discourse.intent.news_query" if topic == "news" else ""), text, final, hints=hints)
        if plan.get("time_range"):
            meter["time_range"] = plan["time_range"]
        try:
            return await tavily(query, plan, ledger, meter, final=final)
        except (WebUnavailable, TavilyError, httpx.HTTPError) as e:
            meter["fallback"] = f"{type(e).__name__}: {e}"[:120]
            if not final:
                raise
    meter.setdefault("credits", 0.0)
    meter["backend"] = "wikipedia"
    try:
        docs = await asyncio.wait_for(wikipedia(query, entity=entity), timeout=config.WEB_LIVE_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise httpx.ConnectTimeout(f"wikipedia: no response in {config.WEB_LIVE_TIMEOUT_S:g}s") from None
    res = WebResult(docs)
    res.backend = "wikipedia"
    return res


def _network_failure(e: Exception) -> bool:
    """Failures that mean "the web is unreachable" (open the breaker), as opposed to a bad response for one query."""
    if isinstance(e, (httpx.TransportError, OSError, asyncio.TimeoutError)):
        return True
    return isinstance(e, TavilyError) and e.status == 0


async def web(query: str, topic: str = "general", *, intent: str = "", text: str = "", final: bool = False,
              ledger: Ledger | None = None, meter: dict | None = None, entity: str = "",
              hints: dict | None = None) -> list[Doc]:
    """The web lane. WEB_BACKEND=cache: recorded responses only. Otherwise the live lane (see _live); if it fails (or
    failed in the last WEB_BREAKER_S seconds), the recorded cache answers instead, flagged as such. Both modes go
    through here, so Head Start and the baseline always see the same fallbacks."""
    meter = meter if meter is not None else {}
    if config.WEB_BACKEND == "cache":
        return _from_cache(query, meter, None)
    if config.WEB_CACHE_FALLBACK and web_down() and len(web_cache()):
        return _from_cache(query, meter, f"live web down: {WEB_STATUS['last_error']}")
    try:
        res = await _live(query, topic, intent, text, final, ledger, meter, entity=entity, hints=hints)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        primary_failed = not (config.WEB_BACKEND == "tavily" and meter.get("backend") == "wikipedia")
        if _network_failure(e) and primary_failed:
            _web_fail(e)
        if config.WEB_CACHE_FALLBACK and len(web_cache()) and (_network_failure(e) or final):
            return _from_cache(query, meter, f"{type(e).__name__}: {e}")
        raise
    _web_ok()
    if config.WEB_CACHE_RECORD and len(res) and not getattr(res, "cached", False):
        c = web_cache()
        c.put(query, res, getattr(res, "backend", "") or meter.get("backend", ""))
        c.save()
    return res


async def probe_web(timeout: float = 4.0) -> dict:
    """One real request to the live backend now (health check); updates WEB_STATUS."""
    WEB_STATUS["probed_at"] = time.time()
    if config.WEB_BACKEND == "cache":
        return web_health()
    t = time.perf_counter()
    try:
        if config.WEB_BACKEND == "tavily":
            r = await asyncio.wait_for(client().get(config.TAVILY_BASE_URL + "/"), timeout)
        else:
            r = await asyncio.wait_for(client().get(
                "https://en.wikipedia.org/w/api.php", params={"action": "query", "format": "json"}), timeout)
        if r.status_code >= 500:
            raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
        _web_ok()
    except Exception as e:
        _web_fail(e if not isinstance(e, asyncio.TimeoutError) else TimeoutError(f"no response in {timeout:g}s"))
    return {**web_health(), "probe_ms": round((time.perf_counter() - t) * 1000)}


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
