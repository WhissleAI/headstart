"""Retrieval lanes. Each returns a list of Doc; all are async and cancellable.

web   : Tavily (search_depth=fast) when TAVILY_API_KEY is set, else Wikipedia (keyless, real network latency).
local : EmbeddingGemma 2 index over a local corpus (text, and audio moments), queried by text or by an audio vector.
"""
from __future__ import annotations

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


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(6.0, connect=3.0), headers=UA,
                                    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10))
    return _client


async def warm_web():
    """Open keep-alive connections so the first speculative call doesn't pay TLS setup."""
    try:
        if config.WEB_BACKEND == "tavily":
            await client().get("https://api.tavily.com/")
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


async def tavily(query: str, k: int = 5, topic: str = "general") -> list[Doc]:
    body = {"query": query, "search_depth": "fast", "max_results": k, "topic": topic, "chunks_per_source": 2,
            "include_answer": False}
    r = await client().post("https://api.tavily.com/search", json=body,
                            headers={"Authorization": f"Bearer {config.TAVILY_API_KEY}"})
    r.raise_for_status()
    out = []
    for x in r.json().get("results", []):
        out.append(Doc(id=x.get("url", ""), title=x.get("title", ""), text=(x.get("content") or "").strip(),
                       url=x.get("url", ""), source="tavily", score=float(x.get("score") or 0)))
    return out


async def web(query: str, topic: str = "general") -> list[Doc]:
    if config.WEB_BACKEND == "tavily":
        return await tavily(query, topic=topic)
    return await wikipedia(query)


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
