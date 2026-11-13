"""Virtual time for asyncio: run the real Session/Speculator code with no sleeping.

Design: an asyncio SelectorEventLoop whose selector never blocks. When the loop has nothing ready, asyncio asks the
selector to wait until the next scheduled callback (`select(timeout)`); the virtual selector instead *advances the
clock by exactly that timeout* and returns immediately. `loop.time()` is the virtual clock, so `asyncio.sleep`,
`wait_for`, `asyncio.wait(timeout=)` and every `call_later` fire in virtual time, in exactly the order they would in
real time. `time.perf_counter` is patched to the same clock while a simulation runs, so the server's own latency
bookkeeping (ready_ms, sources_ms, stability timers, SEM_* windows) is measured in virtual time too.

Consequences (documented in sim/README in the module docstrings):
  * ordering semantics are asyncio's own: same callbacks, same interleaving, same cancellations; only CPU time spent
    in Python (tagger, frame building, ranking: ~1-3 ms per partial in reality) counts as zero.
  * nothing may do real I/O or use threads inside the loop (there is no real select): lanes, embeddings and the LLM
    are faked by sim/fakes.py as coroutines that `await asyncio.sleep(sampled_latency)`.
  * a loop that would block forever (nothing scheduled, nothing ready) raises Deadlock instead of hanging.
  * a coroutine that busy-spins (re-awaits an already-done future; server/answer.py Stream._run does this after a
    hedged request loses) would freeze virtual time; after 200 consecutive zero-timeout loop iterations each further
    iteration is charged 1 ms, which is how the spin ends in real time too (the awaited work completes).
"""
from __future__ import annotations

import asyncio
import contextlib
import selectors
import time


last_spin_s = 0.0      # virtual seconds the last run() spent in busy-spins


class Deadlock(RuntimeError):
    pass


class _VirtualSelector(selectors.SelectSelector):
    SPIN_ITERS = 200
    SPIN_STEP_S = 0.001

    def __init__(self, clock: "VirtualClockLoop"):
        super().__init__()
        self._clock = clock
        self._spin = 0

    def select(self, timeout=None):
        if timeout is None:
            raise Deadlock("event loop would block forever (nothing scheduled) at t=%.3f" % self._clock._vt)
        if timeout == 0:
            # A coroutine that re-awaits an already-completed future spins the loop without ever sleeping (real
            # time would still pass, a few µs per iteration). After SPIN_ITERS consecutive zero-timeout
            # iterations, charge SPIN_STEP_S per iteration so the spin ends when the awaited work would finish.
            self._spin += 1
            if self._spin > self.SPIN_ITERS:
                self._clock._vt += self.SPIN_STEP_S
                self._clock.spin_s += self.SPIN_STEP_S
        else:
            self._spin = 0
        if timeout > 0:
            self._clock._vt += timeout
            if self._clock._vt - self._clock.start > self._clock.limit_s:
                raise Deadlock(f"virtual time limit {self._clock.limit_s}s exceeded")
        return []


class VirtualClockLoop(asyncio.SelectorEventLoop):
    def __init__(self, start: float = 1000.0, limit_s: float = 120.0):
        self._vt = float(start)
        self.start = float(start)
        self.limit_s = limit_s
        self.spin_s = 0.0           # virtual time charged to busy-spinning coroutines (see _VirtualSelector)
        super().__init__(selector=_VirtualSelector(self))

    def time(self) -> float:
        return self._vt


@contextlib.contextmanager
def patched_perf_counter(loop: VirtualClockLoop):
    real = time.perf_counter
    time.perf_counter = lambda: loop._vt
    try:
        yield
    finally:
        time.perf_counter = real


def run(coro_fn, *args, start: float = 1000.0, limit_s: float = 120.0, **kw):
    """Run coro_fn(*args, **kw) to completion in virtual time; cancel and drain leftover tasks; return its result."""
    loop = VirtualClockLoop(start=start, limit_s=limit_s)
    try:
        with patched_perf_counter(loop):
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(coro_fn(*args, **kw))
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            global last_spin_s
            last_spin_s = loop.spin_s
            return res
    finally:
        asyncio.set_event_loop(None)
        loop.close()
