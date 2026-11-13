"""Answer generation (streaming) + a replayable buffer so a speculative draft can be handed over at the endpoint."""
from __future__ import annotations

import asyncio
import time

from anthropic import AsyncAnthropic

from . import config

SYSTEM = """You are the answer engine of a voice search assistant. The user spoke a question; you get search results.
The question text comes from live speech recognition and may contain sound-alike errors ("a polio eleven" = "Apollo
11", "iPhone tower" = "Eiffel Tower") and spoken self-corrections ("Canada, actually make that Turkey"): answer the
question the user most plausibly meant. Never mention mishearing, corrections, or what the user "likely means".
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
    """An answer being generated. Tokens are buffered so a late consumer can replay them, then follow live.

    Ends exactly once: done=True (with error set on failure, "cancelled" on cancel) and every follower wakes, whether
    _run finished, failed, was cancelled mid-flight or was cancelled before it ever started."""

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
        self._waiters: set[asyncio.Future] = set()
        self.task = asyncio.create_task(self._run())
        self.task.add_done_callback(self._task_done)

    # ------------------------------------------------------------------ state
    def _notify(self):
        for w in self._waiters:
            if not w.done():
                w.set_result(None)
        self._waiters.clear()

    def _finish(self, error: str | None = None):
        if self.done:
            return
        if error is not None and self.error is None:
            self.error = error
        self.done = True
        self.t_done = time.perf_counter()
        self._notify()

    def _task_done(self, task: asyncio.Task):
        # covers a task cancelled before its first step (then _run never ran, so its finally never did either)
        if task.cancelled():
            self._finish("cancelled")
        else:
            exc = task.exception()
            self._finish(f"{type(exc).__name__}: {exc}"[:300] if exc else None)

    # ------------------------------------------------------------------ requests
    async def _one(self, i: int, won: asyncio.Event):
        """One request. The first request to produce a token wins; only the winner writes to the buffer."""
        async with client().messages.stream(model=config.ANSWER_MODEL, max_tokens=160, system=SYSTEM,
                                            **model_opts(config.ANSWER_MODEL),
                                            messages=[{"role": "user",
                                                       "content": build_prompt(self.question, self.docs, self.context)}]) as s:
            async for text in s.text_stream:
                if self.done:
                    return
                if self.winner is None:
                    self.winner = i
                    self.t_first = time.perf_counter()
                    won.set()
                if self.winner != i:
                    return
                self.tokens.append(text)
                self._notify()

    async def _run(self):
        """Hedged request: if no token within HEDGE_MS, or the first request ends without producing one (an error such
        as a 529), send a duplicate and keep whichever answers first (cuts the API's time-to-first-token tail)."""
        loop = asyncio.get_running_loop()
        won = asyncio.Event()
        won_w = asyncio.ensure_future(won.wait())
        tasks = [asyncio.create_task(self._one(0, won))]
        pending = set(tasks)
        hedge_at = loop.time() + config.HEDGE_MS / 1000
        first_exc: BaseException | None = None
        try:
            while pending:
                if self.winner is not None and tasks[self.winner].done():
                    break
                timeout = None
                if self.winner is None and not self.hedged:
                    timeout = max(0.0, hedge_at - loop.time())
                waitset = set(pending)
                if not won_w.done():
                    waitset.add(won_w)
                done, _ = await asyncio.wait(waitset, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                ended = [t for t in done if t is not won_w]
                for t in ended:
                    pending.discard(t)
                    exc = None if t.cancelled() else t.exception()      # always retrieve: no "never retrieved" logs
                    if exc is not None and first_exc is None:
                        first_exc = exc
                if self.winner is None and not self.hedged and (not done or ended):
                    # no token by HEDGE_MS, or the request ended without one: send the duplicate now
                    self.hedged = True
                    t1 = asyncio.create_task(self._one(1, won))
                    tasks.append(t1)
                    pending.add(t1)
            if self.winner is not None:
                w = tasks[self.winner]
                exc = None if w.cancelled() else w.exception()
                if exc is not None:
                    raise exc
            elif first_exc is not None:
                raise first_exc
        except asyncio.CancelledError:
            self._finish("cancelled")
            raise
        except Exception as e:  # network / API errors surface in the UI, never crash the session
            self._finish(f"{type(e).__name__}: {e}"[:300])
        finally:
            won_w.cancel()
            for t in tasks:
                if not t.done():
                    t.cancel()
            self._finish()

    # ------------------------------------------------------------------ consumers
    async def follow(self):
        """Yield every token from the start (buffered ones first), until the stream ends."""
        i = 0
        loop = asyncio.get_running_loop()
        while True:
            if i < len(self.tokens):
                new = self.tokens[i:]
                i += len(new)
                for tok in new:
                    yield tok
                continue
            if self.done or self.task.done():
                return
            w = loop.create_future()
            self._waiters.add(w)
            try:
                await w
            finally:
                self._waiters.discard(w)

    def text(self):
        return "".join(self.tokens)

    def cancel(self):
        """Stop generating. A finished stream is left as it is; an unfinished one ends now as error="cancelled"."""
        if self.done:
            return
        if not self.task.done():
            self.task.cancel()
        self._finish("cancelled")
