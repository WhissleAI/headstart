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


def build_prompt(question: str, docs, context: str = "") -> str:
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
    prev = f"Previous turn (for resolving 'it', 'he', 'there'):\n{context}\n\n" if context else ""
    return f"{prev}Sources:\n{src}\n\nSpoken question: {question}"


async def warm():
    """Open the HTTPS connection (and warm the route) so the first real answer doesn't pay for it."""
    try:
        await client().messages.create(model=config.ANSWER_MODEL, max_tokens=1, **model_opts(config.ANSWER_MODEL),
                                       messages=[{"role": "user", "content": "ok"}])
    except Exception:
        pass


class Stream:
    """An answer being generated. Tokens are buffered so a late consumer can replay them, then follow live."""

    def __init__(self, question: str, docs, tag: str = "", context: str = ""):
        self.question = question
        self.docs = list(docs)
        self.tag = tag
        self.context = context
        self.tokens: list[str] = []
        self.done = False
        self.error: str | None = None
        self.t_start = time.perf_counter()
        self.t_first: float | None = None
        self.t_done: float | None = None
        self.winner: int | None = None
        self.hedged = False
        self._cond = asyncio.Condition()
        self.task = asyncio.create_task(self._run())

    async def _one(self, i: int, won: asyncio.Event):
        """One request. The first request to produce a token wins; only the winner writes to the buffer."""
        async with client().messages.stream(model=config.ANSWER_MODEL, max_tokens=160, system=SYSTEM,
                                            **model_opts(config.ANSWER_MODEL),
                                            messages=[{"role": "user",
                                                       "content": build_prompt(self.question, self.docs, self.context)}]) as s:
            async for text in s.text_stream:
                if self.winner is None:
                    self.winner = i
                    self.t_first = time.perf_counter()
                    won.set()
                if self.winner != i:
                    return
                async with self._cond:
                    self.tokens.append(text)
                    self._cond.notify_all()

    async def _run(self):
        """Hedged request: if no token within HEDGE_MS, send a duplicate and keep whichever answers first
        (cuts the API's time-to-first-token tail; Haiku is cheap)."""
        won = asyncio.Event()
        tasks = [asyncio.create_task(self._one(0, won))]
        try:
            try:
                await asyncio.wait_for(asyncio.shield(won.wait()), timeout=config.HEDGE_MS / 1000)
            except asyncio.TimeoutError:
                if not tasks[0].done():
                    self.hedged = True
                    tasks.append(asyncio.create_task(self._one(1, won)))
            # finish when the winner finishes (or all fail)
            while True:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                if self.winner is not None and tasks[self.winner] in done:
                    exc = tasks[self.winner].exception()
                    if exc:
                        raise exc
                    break
                if not pending:
                    for t in done:
                        if t.exception():
                            raise t.exception()
                    break
                tasks = [t for t in tasks]  # keep indices stable
        except asyncio.CancelledError:
            self.error = "cancelled"
            raise
        except Exception as e:  # network / API errors surface in the UI, never crash the session
            self.error = f"{type(e).__name__}: {e}"[:300]
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
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


class Race:
    """TAVILY_ANSWER: the Haiku stream vs Tavily's include_answer (one non-streamed string from a parallel /search
    call). Whichever has text first is the answer; the loser is dropped. Same interface as Stream.

    Tavily's answer arrives whole, so it wins only when (search + Tavily's answer generation) beats (search + Haiku
    time to first token): see PROGRESS.md for when that can happen."""

    def __init__(self, primary, alt_task: asyncio.Task, poll_s: float = 0.005):
        self.primary, self.alt, self.poll_s = primary, alt_task, poll_s
        self.by: str | None = None
        self.alt_text: str | None = None
        self.alt_error: str | None = None
        self.docs = getattr(primary, "docs", [])
        alt_task.add_done_callback(lambda t: t.cancelled() or t.exception())   # never "exception never retrieved"

    @property
    def hedged(self):
        return getattr(self.primary, "hedged", False)

    @property
    def done(self):
        return self.primary.done if self.by != "tavily" else True

    @property
    def error(self):
        return None if self.by == "tavily" else self.primary.error

    def _alt_answer(self) -> str | None:
        if not self.alt.done() or self.alt.cancelled():
            return None
        exc = self.alt.exception()
        if exc is not None:
            self.alt_error = f"{type(exc).__name__}: {exc}"[:200]
            return None
        return (getattr(self.alt.result(), "answer", None) or "").strip() or None

    async def _decide(self):
        while self.by is None:
            if self.primary.tokens:
                self.by = "llm"
            elif (a := self._alt_answer()) is not None:
                self.by, self.alt_text = "tavily", a
            elif self.primary.done and (self.alt.done() or self.primary.error is None):
                self.by = "llm"            # Haiku finished (or failed) and Tavily has nothing better
            else:
                await asyncio.sleep(self.poll_s)
        if self.by == "tavily":
            self.primary.cancel()

    async def follow(self):
        await self._decide()
        if self.by == "tavily":
            yield self.alt_text
            return
        async for tok in self.primary.follow():
            yield tok

    def text(self):
        return self.alt_text if self.by == "tavily" else self.primary.text()

    def cancel(self):
        self.primary.cancel()
        if not self.alt.done():
            self.alt.cancel()
