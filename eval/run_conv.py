"""Multi-turn A/B: each conversation is fed as one live stream (turn, 2.5 s gap, turn) through ArcEar + Session.
Scores the follow-up turn (pronoun → needs conversation memory)."""
import argparse, asyncio, json, os, sys, time
from pathlib import Path
import numpy as np, soundfile as sf
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser(); ap.add_argument("--no-memory", action="store_true"); args = ap.parse_args()
from server import ear, embed, lanes, answer   # noqa
from server.session import Session            # noqa

def ok(ans, keys):
    a = ans.lower()
    return all(any(k2 in a for k2 in k.split("|")) for k in keys)

async def run(conv, mode):
    ev = []
    t0 = time.perf_counter()
    s = Session(ev.append, mode=mode, t0=t0)
    if args.no_memory:
        s.carry_pool = {}
    e = ear.ArcEar([s])
    parts = []
    for f in conv["files"]:
        w, _ = sf.read(ROOT / "eval" / "audio" / f, dtype="float32")
        parts += [w, np.zeros(int(16000 * 2.5), np.float32)]
    a = np.concatenate(parts)
    for i in range(0, len(a), 1600):
        while time.perf_counter() < t0 + (i + 1600) / 16000:
            await asyncio.sleep(0.003)
        if args.no_memory:
            s.carry = (); s.carry_pool = {}
        await e.feed(a[i:i + 1600])
    await e.flush()
    if s.final_task:
        try: await s.final_task
        except Exception: pass
    s.close()
    return s.history

async def main():
    ear.warmup(); embed.warmup(); await lanes.warm_web(); await answer.warm()
    convs = [json.loads(x) for x in (ROOT / "eval" / "conversations.jsonl").read_text().splitlines() if x.strip()]
    res = []
    for c in convs:
        for mode in ("baseline", "headstart"):
            h = await run(c, mode)
            row = {"id": c["id"], "mode": mode, "n_answers": len(h)}
            for i, m in enumerate(h[:2]):
                keys = c["expect"][i] if i < len(c["expect"]) else []
                row[f"t{i+1}"] = {"q": m["question"], "query": m["query"], "ready_ms": m["ready_ms"], "path": m["path"],
                                  "correct": ok(m["answer"], keys) if keys else None, "answer": m["answer"][:140]}
            res.append(row)
            t2 = row.get("t2", {})
            print(f'{c["id"]} {mode:9s} answers={len(h)} t2: q={t2.get("query")!r:28s} ready={t2.get("ready_ms")} ok={t2.get("correct")} | {t2.get("answer","")[:80]!r}')
    out = ROOT / "runs" / ("conv_nomem.json" if args.no_memory else "conv.json")
    out.write_text(json.dumps(res, indent=1))
asyncio.run(main())
