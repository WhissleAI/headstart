"""Scripted ear: replays a question as timed words (+ optional revisions) through a Session, in real time."""
import asyncio, time, json, sys
sys.path.insert(0, ".")
from server.session import Session

async def run_question(text, mode, wpm_gap=0.32, endpoint_delay=0.6, revisions=None, emit=None):
    events = []
    emit = emit or (lambda e: events.append(e))
    t0 = time.perf_counter()
    s = Session(emit, mode=mode, t0=t0)
    words = text.split()
    times = [(0.3 + i * wpm_gap, 0.3 + i * wpm_gap + wpm_gap * 0.9) for i in range(len(words))]
    end = times[-1][1]
    tick = 0.1
    t = 0.0
    revisions = revisions or {}
    while t < end + endpoint_delay:
        target = t0 + t
        while time.perf_counter() < target:
            await asyncio.sleep(0.002)
        heard = [dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times) if b <= t + 0.15]
        for i, (at, wrong) in revisions.items():
            if i < len(heard) and t < at:
                heard[i] = dict(heard[i], text=wrong)
        if heard:
            s.on_partial(heard)
        t += tick
    m = await s.on_final([dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times)])
    s.close()
    return m, events

async def main():
    from server import embed, lanes
    embed.warmup(); await lanes.warm_web()
    from server import answer; await answer.warm()
    qs = ["how tall is the Eiffel Tower", "when did Apollo 11 land on the moon",
          "who founded the company Nvidia", "what is the capital of Australia"]
    for q in qs:
        for mode in ("baseline", "headstart"):
            m, ev = await run_question(q, mode)
            print(f"{mode:9s} ready={m['ready_ms']:5d}ms sources={m['sources_ms']:5d}ms done={m['done_ms']:5d} path={m.get('path')} web={m['web_calls']} cr={m.get('credits', 0):g} fires={m['spec_fires']} used={m['spec_used']} drafts={m['drafts']} reused={m['draft_reused']}/{m.get('draft_match','')} | {m['answer'][:90]!r}")
        print()

if __name__ == "__main__":
    asyncio.run(main())
