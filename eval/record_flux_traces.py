"""Stream the eval questions (eval/questions.jsonl + eval/showcase) at real time to the REAL Deepgram Flux API (the
FluxEar's URL/params from server/config.py) and save every message with its arrival time (s since audio sample 0) to
runs/flux_traces.jsonl (resumable). Input to eval/flux_gate_eval.py. Needs DEEPGRAM_API_KEY; ~8 min for 80 questions.

Run: .venv/bin/python eval/record_flux_traces.py"""
import asyncio, json, sys, time
import numpy as np, soundfile as sf
from pathlib import Path
R = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, R)
from server import config, ear
import websockets
qs = [json.loads(l) for l in open(f"{R}/eval/questions.jsonl")] + [json.loads(l) for l in open(f"{R}/eval/showcase/questions.jsonl")]
out = open(f"{R}/runs/flux_traces.jsonl", "a")
done = set()
try:
    done = {json.loads(l)["id"] for l in open(f"{R}/runs/flux_traces.jsonl")}
except Exception: pass

async def one(q):
    x, sr = sf.read(f"{R}/eval/audio/{q['file']}", dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    x = np.concatenate([x, np.zeros(int(1.6 * 16000), np.float32)])
    msgs = []
    async with websockets.connect(ear.flux_url(), additional_headers={"Authorization": f"Token {config.DEEPGRAM_API_KEY}"}) as ws:
        async def rx():
            async for m in ws:
                d = json.loads(m); d["_t"] = round(time.perf_counter() - t0, 4); msgs.append(d)
        t0 = time.perf_counter()
        r = asyncio.create_task(rx())
        for i in range(0, len(x), 1600):
            await ws.send((np.clip(x[i:i+1600], -1, 1) * 32767).astype("<i2").tobytes())
            await asyncio.sleep(max(0, t0 + (i + 1600) / 16000 - time.perf_counter()))
        await ws.send(json.dumps({"type": "CloseStream"}))
        try: await asyncio.wait_for(r, 4)
        except Exception: pass
    return msgs

async def main():
    for q in qs:
        if q["id"] in done: continue
        msgs = await one(q)
        out.write(json.dumps({"id": q["id"], "file": q["file"], "text": q["text"], "dur_s": q.get("dur_s"),
                              "eager": ear.flux_eager(), "eot": config.FLUX_EOT, "msgs": msgs}) + "\n"); out.flush()
        eots = [m for m in msgs if m.get("event") == "EndOfTurn"]
        print(q["id"], len(msgs), "|", " / ".join(m["transcript"] for m in eots), flush=True)
asyncio.run(main())
