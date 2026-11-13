"""A fake Tavily /search (and a tiny fake Wikipedia) for tests and keyless demos.

Mirrors the documented contract (https://docs.tavily.com/documentation/api-reference/endpoint/search): request
validation (enums, ranges, Bearer key), response shape (query, results[title,url,content,score,raw_content,
published_date for news], images, answer, response_time, usage.credits, request_id), credit cost (advanced 2, else 1),
and the documented errors (400/401/429+Retry-After/432/433/434/500).

Latency is lognormal per search_depth (medians are guesses until measured against the real API; see PROGRESS.md),
plus an extra term when include_answer is set. Tests can pin latency with `latency_ms=(lo, hi)` (uniform).

In-process:   fake = FakeTavily(seed=1); lanes.use_transport(fake.transport())
As a server:  .venv/bin/python -m uvicorn --app-dir tests fake_tavily:app --port 8799
              TAVILY_API_KEY=tvly-fake TAVILY_BASE_URL=http://127.0.0.1:8799 → the demo runs without spending credits
"""
from __future__ import annotations

import asyncio
import math
import random
import re
import time
import uuid

import httpx

DEPTHS = {"ultra-fast", "fast", "basic", "advanced"}
TOPICS = {"general", "news", "finance"}
TIME_RANGES = {"day", "week", "month", "year", "d", "w", "m", "y"}
# (median ms, lognormal sigma). Qualitative order from Tavily's best-practices page; numbers are placeholders.
LATENCY = {"ultra-fast": (320, 0.35), "fast": (480, 0.35), "basic": (800, 0.35), "advanced": (1600, 0.4)}
ANSWER_LATENCY = {"basic": (700, 0.3), "advanced": (1500, 0.3)}
CHUNK_CHARS = {"fast": 500, "basic": 500, "advanced": 800}


def _lognormal(rng: random.Random, median: float, sigma: float) -> float:
    return median * math.exp(rng.gauss(0.0, sigma))


class FakeTavily:
    def __init__(self, seed: int = 0, latency_ms: tuple | None = None, answer_ms: tuple | None = None,
                 scale: float = 1.0, key_prefix: str = "tvly-", paired: bool = True):
        self.seed, self.paired = seed, paired
        self.rng = random.Random(seed)
        self.latency_ms = latency_ms          # (lo, hi) uniform override for the search part
        self.answer_ms = answer_ms            # (lo, hi) uniform override for include_answer's extra time
        self.scale = scale                    # multiply every delay (0 → instant)
        self.key_prefix = key_prefix
        self.requests: list[dict] = []        # every /search body received (after auth), in order
        self.wiki_requests: list[dict] = []
        self.fail: list = []                  # queue of (status, headers) or callables(body) -> (status, headers)|None
        self.credits = 0

    # ------------------------------------------------------------------ contract
    def validate(self, body: dict) -> str | None:
        q = body.get("query")
        if not isinstance(q, str) or not q.strip():
            return "query is required"
        if body.get("search_depth", "basic") not in DEPTHS:
            return "invalid search_depth"
        if body.get("topic", "general") not in TOPICS:
            return "invalid topic"
        if body.get("time_range") is not None and body["time_range"] not in TIME_RANGES:
            return "invalid time_range"
        if not 0 <= int(body.get("max_results", 10)) <= 20:
            return "max_results must be 0-20"
        if "chunks_per_source" in body and not 1 <= int(body["chunks_per_source"]) <= 3:
            return "chunks_per_source must be 1-3"
        if body.get("max_age_hours") is not None and not -1 <= int(body["max_age_hours"]) <= 720:
            return "max_age_hours must be -1..720"
        if body.get("include_answer") not in (None, False, True, "basic", "advanced"):
            return "invalid include_answer"
        if "include_domains_mode" in body and not body.get("include_domains"):
            return "include_domains_mode requires include_domains"
        if body.get("include_domains_mode") not in (None, "prefer", "restrict"):
            return "invalid include_domains_mode"
        if len(body.get("include_domains") or []) > 300:
            return "include_domains: max 300"
        return None

    def delay_s(self, body: dict) -> float:
        """Paired by default: the same (seed, query, depth, answer) always takes the same time, so an A/B where both
        modes send the same final query compares like with like. paired=False draws fresh each call."""
        depth = body.get("search_depth", "basic")
        rng = (random.Random(f"{self.seed}|{body['query']}|{depth}|{body.get('include_answer')}") if self.paired
               else self.rng)
        if self.latency_ms:
            ms = rng.uniform(*self.latency_ms)
        else:
            ms = _lognormal(rng, *LATENCY[depth])
        ans = body.get("include_answer")
        if ans:
            kind = "advanced" if ans == "advanced" else "basic"
            ms += rng.uniform(*self.answer_ms) if self.answer_ms else _lognormal(rng, *ANSWER_LATENCY[kind])
        if body.get("max_age_hours") is not None:
            ms += rng.uniform(0, 250)          # occasional live fetch
        return ms * self.scale / 1000

    def results(self, body: dict) -> dict:
        q = body["query"]
        depth = body.get("search_depth", "basic")
        topic = body.get("topic", "general")
        words = [w for w in re.findall(r"\w+", q.lower())]
        subject = " ".join(w.capitalize() for w in words[:3]) or "Result"
        slug = "-".join(words[:4]) or "q"
        n = int(body.get("max_results", 10))
        out = []
        for i in range(n):
            if depth == "ultra-fast":
                content = f"{subject}: summary {i + 1}. {q} — key facts in one NLP summary of the page."
            else:
                chunk = (f"{subject} {i + 1}: {q}. " * 20)[: CHUNK_CHARS[depth]]
                content = " [...] ".join([chunk] * int(body.get("chunks_per_source", 3)))
            r = {"title": f"{subject} — source {i + 1}", "url": f"https://{topic}.example.com/{slug}/{i + 1}",
                 "content": content, "score": round(0.95 - 0.07 * i, 3), "raw_content": None,
                 "id": uuid.uuid4().hex}
            if topic == "news" or body.get("include_published_date"):
                r["published_date"] = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(time.time() - 3600 * i))
            out.append(r)
        cost = 2 if depth == "advanced" else 1
        self.credits += cost
        data = {"query": q, "follow_up_questions": None, "images": [], "results": out,
                "response_time": 0.0, "request_id": uuid.uuid4().hex}
        if body.get("include_answer"):
            data["answer"] = f"{subject}: the short answer, from {len(out)} sources."
        else:
            data["answer"] = None
        if body.get("include_usage"):
            data["usage"] = {"credits": cost}
        return data

    def _error(self, status: int, msg: str, headers: dict | None = None):
        return status, {"detail": {"error": msg}}, headers or {}

    async def search(self, body: dict, auth: str | None) -> tuple[int, dict, dict]:
        if not auth or not auth.startswith("Bearer " + self.key_prefix):
            return self._error(401, "Unauthorized: missing or invalid API key.")
        self.requests.append(body)
        if self.fail:
            f = self.fail.pop(0)
            f = f(body) if callable(f) else f
            if f:
                status, headers = f
                await asyncio.sleep(0.02 * self.scale)
                return self._error(status, f"injected {status}", headers)
        bad = self.validate(body)
        if bad:
            return self._error(400, bad)
        d = self.delay_s(body)
        await asyncio.sleep(d)
        data = self.results(body)
        data["response_time"] = round(d, 3)
        return 200, data, {}

    async def wikipedia(self, params: dict) -> tuple[int, dict, dict]:
        self.wiki_requests.append(params)
        await asyncio.sleep(self.rng.uniform(0.1, 0.3) * self.scale)
        q = params.get("gsrsearch", "")
        pages = [{"pageid": i, "index": i, "title": f"{q.title()} (wiki {i})", "fullurl": f"https://en.wikipedia.org/wiki/{i}",
                  "extract": f"Wikipedia about {q}: fact {i}."} for i in range(1, 4)]
        return 200, {"query": {"pages": pages}}, {}

    # ------------------------------------------------------------------ transports
    async def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.host.endswith("wikipedia.org"):
            status, data, headers = await self.wikipedia(dict(request.url.params))
        elif request.method == "POST" and request.url.path.endswith("/search"):
            import json
            status, data, headers = await self.search(json.loads(request.content or b"{}"),
                                                      request.headers.get("authorization"))
        else:
            status, data, headers = 404, {"detail": "not found"}, {}
        return httpx.Response(status, json=data, headers=headers)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def make_app(fake: FakeTavily | None = None):
    global Request
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    fake = fake or FakeTavily(seed=int(time.time()))
    api = FastAPI(title="fake tavily")

    @api.post("/search")
    async def search(request: Request):
        status, data, headers = await fake.search(await request.json(), request.headers.get("authorization"))
        return JSONResponse(data, status_code=status, headers=headers)

    @api.get("/")
    async def root():
        return {"ok": True, "credits": fake.credits, "requests": len(fake.requests)}

    return api


try:
    app = make_app()
except Exception:      # fastapi missing: the in-process transport still works
    app = None
