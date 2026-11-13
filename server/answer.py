"""Answer generation (streaming) + a replayable buffer so a speculative draft can be handed over at the endpoint."""
from __future__ import annotations

import asyncio
import time

from anthropic import AsyncAnthropic

from . import config

SYSTEM = """You are the answer engine of a voice search assistant. The user spoke a question; you get search results.
The question text comes from live speech recognition and may contain sound-alike errors ("a polio eleven" = "Apollo
11", "iPhone tower" = "Eiffel Tower"): answer the question the user most plausibly meant, without commenting on it.
Answer in one or two short spoken-style sentences (under 45 words), most important fact first. Cite sources inline
as [1], [2] by their number. If the sources don't contain the answer, say so in one sentence and give the closest
relevant fact. No preamble, no markdown."""

_client: AsyncAnthropic | None = None


def client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic()
    return _client


def model_opts(model: str) -> dict:
    """Lowest time-to-first-token: thinking off, low effort. (Haiku 5.5 thinks by default; Sonnet 5.5 can't take
    "disabled" and uses "between_tools" instead.)"""
    if model.startswith("claude-haiku-5"):
        return {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}}
    if model.startswith("claude-sonnet-5"):
        return {"thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}}
    return {"output_config": {"effort": "low"}}


def build_prompt(question: str, docs) -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        where = f" (moment at {int(d.t // 60):02d}:{int(d.t % 60):02d})" if d.t is not None else ""
        text = d.text
        meta = getattr(d, "meta", None) or {}
        if meta.get("context_before") or meta.get("context_after"):
            # local moments: neighbour words help the answer (not embedded: they hurt retrieval)
            text = f"…{meta.get('context_before', '')} [{d.text}] {meta.get('context_after', '')}…"
            if meta.get("role"):
                text = f"({meta['role']}) " + text
        lines.append(f"[{i}] {d.title}{where}: {text[:1000]}")
    src = "\n".join(lines) if lines else "(no results)"
    return f"Sources:\n{src}\n\nSpoken question: {question}"


async def warm():
    """Open the HTTPS connection (and warm the route) so the first real answer doesn't pay for it."""
    try:
        await client().messages.create(model=config.ANSWER_MODEL, max_tokens=1, **model_opts(config.ANSWER_MODEL),
                                       messages=[{"role": "user", "content": "ok"}])
    except Exception:
        pass


class Stream:
    """An answer being generated. Tokens are buffered so a late consumer can replay them, then follow live."""

    def __init__(self, question: str, docs, tag: str = ""):
        self.question = question
        self.docs = list(docs)
        self.tag = tag
        self.tokens: list[str] = []
        self.done = False
        self.error: str | None = None
        self.t_start = time.perf_counter()
        self.t_first: float | None = None
        self.t_done: float | None = None
        self._cond = asyncio.Condition()
        self.task = asyncio.create_task(self._run())

    async def _run(self):
        try:
            async with client().messages.stream(model=config.ANSWER_MODEL, max_tokens=160, system=SYSTEM,
                                                **model_opts(config.ANSWER_MODEL),
                                                messages=[{"role": "user",
                                                           "content": build_prompt(self.question, self.docs)}]) as s:
                async for text in s.text_stream:
                    if self.t_first is None:
                        self.t_first = time.perf_counter()
                    async with self._cond:
                        self.tokens.append(text)
                        self._cond.notify_all()
        except asyncio.CancelledError:
            self.error = "cancelled"
            raise
        except Exception as e:  # network / API errors surface in the UI, never crash the session
            self.error = f"{type(e).__name__}: {e}"[:300]
        finally:
            self.done = True
            self.t_done = time.perf_counter()
            async with self._cond:
                self._cond.notify_all()

    async def follow(self):
        """Yield every token from the start (buffered ones first), until the stream ends."""
        i = 0
        while True:
            async with self._cond:
                while i >= len(self.tokens) and not self.done:
                    await self._cond.wait()
                new = self.tokens[i:]
                done = self.done
            for tok in new:
                yield tok
            i += len(new)
            if done and i >= len(self.tokens):
                return

    def text(self):
        return "".join(self.tokens)

    def cancel(self):
        if not self.task.done():
            self.task.cancel()
