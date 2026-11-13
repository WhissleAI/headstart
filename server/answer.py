"""Answer generation (streaming) + a replayable buffer so a speculative draft can be handed over at the endpoint."""
from __future__ import annotations

import asyncio
import re
import time

import httpx
from anthropic import AsyncAnthropic

from . import config

SYSTEM = """You are the answer engine of a voice search assistant. The user spoke a question; you get search results.
The question text comes from live speech recognition and may contain sound-alike errors ("a polio eleven" = "Apollo
11", "iPhone tower" = "Eiffel Tower") and spoken self-corrections ("Canada, actually make that Turkey"): answer the
question the user most plausibly meant. Never mention mishearing, corrections, or what the user "likely means".
Answer in one or two short spoken-style sentences (under 45 words), most important fact first. Cite sources inline
as [1], [2] by their number.
Grounding: if a source states the answer, or it follows directly from a source ("X is the second-largest, after Y"
means Y is the largest), state it plainly with that citation and no disclaimer. Answer only the question meant:
never remark on what the sources don't cover about other parts of it (a corrected-away name, a misheard word). Only
if no source supports the answer, say in a few words that the sources don't cover it, then give the answer you are
confident of without a citation; don't pad with unrelated facts from the sources.
Some sources are moments of recordings (game broadcasts, hearings, calls, podcasts): they have a timestamp and a noisy
transcript; the moment itself is in [brackets], with the words just before and after it (also from the recording).
Answer from every moment that helps. Say when and who, and quote the key words as transcribed, e.g. At 00:26 the
announcer said "Hunter gets a good look and nails it" [1]. The question can mishear words too ("who one the toss" = "who won
the toss"). Don't comment on transcript quality. No preamble, no markdown."""

ROLE_LABEL = {"play_by_play": "play-by-play announcer", "analyst": "analyst", "judge": "judge", "counsel": "counsel",
              "device_user": "user"}


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


_client: AsyncAnthropic | None = None


def client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(max_retries=config.ANSWER_MAX_RETRIES,
                                 timeout=httpx.Timeout(60.0, connect=3.0))
    return _client


# ------------------------------------------------------------------ reachability + extractive fallback
# If the answer model can't be reached, the race still completes: every Stream (draft or final, both modes) answers with
# the best-matching sentence of its top source instead, flagged fallback="extractive" (metric answer_fallback, UI badge).
STATUS = {"status": "unknown", "last_error": None, "down_until": 0.0, "last_ok": None}


def _ok():
    STATUS.update(status="live", down_until=0.0, last_ok=time.time())


def _down(err: str):
    STATUS.update(status="down", last_error=err[:200], down_until=time.monotonic() + config.ANSWER_BREAKER_S)


def is_down() -> bool:
    return time.monotonic() < STATUS["down_until"]


def health() -> dict:
    return {"model": config.ANSWER_MODEL, "status": STATUS["status"], "live": STATUS["status"] == "live",
            "fallback": "extractive" if config.ANSWER_FALLBACK else "none", "last_error": STATUS["last_error"]}


def reset_status():
    """Tests."""
    STATUS.update(status="unknown", last_error=None, down_until=0.0, last_ok=None)


_STOPW = set("""a an the is are was were be been of in on at to for from by with about what who whom whose which when
where why how did do does can could would should will it its this that there tell me please and or i you""".split())
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _words(s: str) -> set:
    """Content words, crudely stemmed (founded/founder -> found) so the question and a source sentence can meet."""
    return {w[:5] for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOPW and len(w) > 1}


_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b|\b(January|February|March|April|May|June|July|August|September|"
                   r"October|November|December)\b")
_NAME = re.compile(r"(?<!^)(?<![.!?] )\b[A-Z][a-z]+ [A-Z][a-z]+")


def _wh_bonus(question: str, sent: str) -> float:
    """Does the sentence have the KIND of thing asked for (who -> a name, when -> a date, how many -> a number)?"""
    q = question.lower()
    if re.search(r"\bwho\b", q):
        return 1.0 if (_NAME.search(sent) or " by " in sent) else 0.0
    if re.search(r"\bwhen\b|\bwhat year\b", q):
        return 1.0 if _YEAR.search(sent) else 0.0
    if re.search(r"\bhow (many|much|tall|long|old|far|big|high|deep)\b", q):
        return 1.0 if re.search(r"\d", sent) else 0.0
    if re.search(r"\bwhere\b", q):
        return 0.5 if re.search(r"\b(in|at|near) [A-Z]", sent) else 0.0
    return 0.0


def extractive(question: str, docs) -> str | None:
    """The sentence of the top source that best answers the question (shared content words, words of the source's
    own title counting less since every sentence is about it, plus the kind of answer asked for), with its citation;
    a sentence of the second or third source only if it is clearly better. None if there are no sources."""
    q = _words(question)
    best = None
    for i, d in enumerate(list(docs)[:3]):
        meta = getattr(d, "meta", None) or {}
        text = " ".join(x for x in (meta.get("context_before"), d.text, meta.get("context_after")) if x) \
            if meta.get("context_before") or meta.get("context_after") else d.text
        title = _words(d.title or "")
        sents = [x.strip() for x in _SENT.split((text or "").replace("\n", " ")) if x.strip()]
        for j, sent in enumerate(sents):
            shared = q & _words(sent)
            score = (sum(0.3 if w in title else 1.0 for w in shared) + _wh_bonus(question, sent)
                     - 0.01 * j - 0.75 * i - (3.0 if sent.rstrip().endswith("?") else 0.0))   # a question isn't an answer
            if best is None or score > best[0]:
                best = (score, sent, i)
    if best is None:
        return None
    _, sent, i = best
    if len(sent) > 260:
        sent = sent[:257].rsplit(" ", 1)[0] + "…"
    return f"{sent} [{i + 1}]"


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
        meta = getattr(d, "meta", None) or {}
        if d.t is not None:
            # a local moment: "<recording> at mm:ss, <role>: …before [moment] after…" (neighbour words help the
            # answer; they are not embedded: they hurt retrieval)
            label = meta.get("source_label") or d.title.rsplit(" · ", 1)[0]
            role = ROLE_LABEL.get(meta.get("role") or "", "")
            head = f"{label} at {mmss(d.t)}" + (f", {role}" if role else "")
            text = d.text
            if meta.get("context_before") or meta.get("context_after"):
                text = f"…{meta.get('context_before', '')} [{d.text}] {meta.get('context_after', '')}…"
            lines.append(f"[{i}] {head}: {text[:1000]}")
        else:
            lines.append(f"[{i}] {d.title}: {d.text[:1000]}")
    src = "\n".join(lines) if lines else "(no results)"
    prev = f"Previous turn (for resolving 'it', 'he', 'there'):\n{context}\n\n" if context else ""
    return f"{prev}Sources:\n{src}\n\nSpoken question: {question}"


async def warm():
    """Open the HTTPS connection (and warm the route) so the first real answer doesn't pay for it; records whether the
    answer model is reachable (STATUS)."""
    await probe()


async def probe(timeout: float = 8.0) -> dict:
    """One 1-token request now; updates STATUS (the breaker opens if the model is unreachable)."""
    t = time.perf_counter()
    try:
        await asyncio.wait_for(client().messages.create(
            model=config.ANSWER_MODEL, max_tokens=1, **model_opts(config.ANSWER_MODEL),
            messages=[{"role": "user", "content": "ok"}]), timeout)
        _ok()
    except Exception as e:
        _down(f"{type(e).__name__}: {e}" if str(e) else type(e).__name__)
    return {**health(), "probe_ms": round((time.perf_counter() - t) * 1000)}


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
        self.fallback: str | None = None        # "extractive" when the answer model couldn't be reached
        self.fallback_reason: str | None = None
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
                    _ok()
                if self.winner != i:
                    return
                self.tokens.append(text)
                self._notify()

    async def _run(self):
        """Hedged request: if no token within HEDGE_MS, or the first request ends without producing one (an error such
        as a 529), send a duplicate and keep whichever answers first (cuts the API's time-to-first-token tail)."""
        if config.ANSWER_FALLBACK and is_down():
            self._extractive(f"answer model down: {STATUS['last_error']}")    # breaker open: don't wait again
            return
        loop = asyncio.get_running_loop()
        won = asyncio.Event()
        won_w = asyncio.ensure_future(won.wait())
        tasks = [asyncio.create_task(self._one(0, won))]
        pending = set(tasks)
        hedge_at = loop.time() + config.HEDGE_MS / 1000
        deadline = loop.time() + config.ANSWER_DEADLINE_MS / 1000 if config.ANSWER_FALLBACK else None
        first_exc: BaseException | None = None
        try:
            while pending:
                if self.winner is not None and tasks[self.winner].done():
                    break
                timeout = None
                if self.winner is None and not self.hedged:
                    timeout = max(0.0, hedge_at - loop.time())
                if self.winner is None and deadline is not None:
                    left = max(0.0, deadline - loop.time())
                    timeout = left if timeout is None else min(timeout, left)
                    if left <= 0:
                        raise TimeoutError(f"no answer token in {config.ANSWER_DEADLINE_MS} ms")
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
            err = f"{type(e).__name__}: {e}"[:300]
            if config.ANSWER_FALLBACK and not self.tokens:
                _down(err)
                for t in tasks:
                    if not t.done():
                        t.cancel()
                self._extractive(err)
            else:
                self._finish(err)
        finally:
            won_w.cancel()
            for t in tasks:
                if not t.done():
                    t.cancel()
            self._finish()

    def _extractive(self, why: str):
        """Answer without the model: best-matching sentence of the top source, flagged (never presented as Claude's)."""
        self.fallback, self.fallback_reason = "extractive", why[:200]
        text = extractive(self.question, self.docs)
        if text is None:
            self._finish(f"answer model unreachable and no sources ({why})"[:300])
            return
        self.t_first = time.perf_counter()
        self.tokens.append(text)
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
    def fallback(self):
        return None if self.by == "tavily" else getattr(self.primary, "fallback", None)

    @property
    def fallback_reason(self):
        return None if self.by == "tavily" else getattr(self.primary, "fallback_reason", None)

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
        """Event-driven: wakes on the primary's first token / end (Stream._waiters) or on the alt task finishing; only
        waits on what can still change (a finished alt is never re-awaited, so no spin)."""
        loop = asyncio.get_running_loop()
        while self.by is None:
            if self.primary.tokens:
                self.by = "llm"
            elif (a := self._alt_answer()) is not None:
                self.by, self.alt_text = "tavily", a
            elif self.primary.done and (self.alt.done() or self.primary.error is None):
                self.by = "llm"            # Haiku finished (or failed) and Tavily has nothing better
            else:
                waiters = getattr(self.primary, "_waiters", None)
                if waiters is None:        # a primary without the notify hook (test fakes): poll
                    await asyncio.sleep(self.poll_s)
                    continue
                w = loop.create_future()
                waiters.add(w)
                ws = {w} if self.primary.done or self.alt.done() else {w, self.alt}
                if self.primary.done:      # primary over and failed: only the alt can still change anything
                    ws = {self.alt}
                try:
                    await asyncio.wait(ws, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    waiters.discard(w)
                    w.cancel()
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
