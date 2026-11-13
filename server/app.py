"""Head Start server: FastAPI + one WebSocket per client.

Client -> server
  binary frame                        Float32Array (little-endian) mono 16 kHz PCM, any length (100 ms recommended)
  {"t":"start","modes":["headstart","baseline"], "domain":"assistant"}   start a run (both modes race on one ear)
  {"t":"stop"}                        flush the ear (closes the open utterance) and end the run
  {"t":"replay","file":"q01.wav","modes":[...]}   server streams eval/audio/<file> through the ear in real time
Server -> client: JSON events, each with "mode" ("headstart" | "baseline" | "ear" | "sys"):
  sys    {"t":"ready"}, {"t":"error","error"}, {"t":"run_done"}
  ear    {"t":"ear","kind":"partial|final|arc_command_complete|connected|fallback|warning|error","ear":"arc|deepgram",...}
         (ready and /api/health carry "ear": the ear new runs use, "ear_requested": env EAR; a run that falls back
         from Deepgram to Arc says so with a "fallback" ear event)
  per mode: words, frame, spec, endpoint, sources, answer, answer_error, metric   (see server/speculator.py, session.py)
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import answer, config, embed, ear, lanes
from .session import Session

ROOT = config.ROOT
WEB = ROOT / "web"
AUDIO = ROOT / "eval" / "audio"
app = FastAPI(title="Head Start")
STATE = {"warm": False}


@app.on_event("startup")
async def startup():
    loop = asyncio.get_running_loop()
    t = time.perf_counter()
    await loop.run_in_executor(None, ear.warmup)
    await embed.arun(embed.warmup)
    lanes.index()
    await lanes.warm_web()
    await answer.warm()
    STATE.update(warm=True, warm_s=round(time.perf_counter() - t, 1))


@app.get("/api/health")
async def health():
    return {"ok": True, **STATE, **ear.describe(), "web_backend": config.WEB_BACKEND, "answer_model": config.ANSWER_MODEL,
            "local_docs": len(lanes.index())}


@app.get("/api/questions")
async def questions(set: str = "all"):
    """set=web: eval/questions.jsonl; set=local: index/questions_local.jsonl (kind "local"); all: both."""
    rows = []
    srcs = {"web": [ROOT / "eval" / "questions.jsonl"], "local": [ROOT / "index" / "questions_local.jsonl"]}
    for p in srcs.get(set, srcs["web"] + srcs["local"]):
        if p.exists():
            rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return JSONResponse(rows)


@app.get("/api/conversations")
async def conversations():
    p = ROOT / "eval" / "conversations.jsonl"
    return JSONResponse([json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else [])


@app.get("/api/results")
async def results():
    p = ROOT / "runs" / "ab_latest.json"
    return JSONResponse(json.loads(p.read_text()) if p.exists() else {})


@app.get("/audio/{name}")
async def audio(name: str):
    p = (AUDIO / name).resolve()
    if AUDIO.resolve() not in p.parents or not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p)


class Run:
    def __init__(self, ws: WebSocket, modes, domain):
        self.ws = ws
        self.q: asyncio.Queue = asyncio.Queue()
        t0 = time.perf_counter()
        self.sessions = [Session(self._emitter(m), mode=m, domain=domain, t0=t0) for m in modes]
        self.ear = ear.make_ear(self.sessions, domain=domain, on_event=self._emitter("ear"))
        self.sender = asyncio.create_task(self._send())

    def _emitter(self, mode):
        def emit(e):
            self.q.put_nowait({**e, "mode": mode})
        return emit

    async def _send(self):
        while True:
            e = await self.q.get()
            try:
                await self.ws.send_text(json.dumps(e, default=float))
            except Exception:
                return

    async def stop(self):
        await self.ear.flush()
        self.ear.close()
        await asyncio.sleep(0.05)
        for s in self.sessions:
            s.close()
        self.q.put_nowait({"t": "run_done", "mode": "sys",
                           "metrics": [s.history for s in self.sessions]})
        await asyncio.sleep(0.2)
        self.sender.cancel()


async def replay(run: Run, path: Path):
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    t0 = time.perf_counter()
    for s in run.sessions:
        s.t0 = t0
        s.spec.t0 = t0
    for i in range(0, len(wav), 1600):
        target = t0 + (i + 1600) / 16000
        while time.perf_counter() < target:
            await asyncio.sleep(0.003)
        await run.ear.feed(wav[i:i + 1600])
    # trailing silence so Arc can endpoint at the pause, like a live mic
    for _ in range(12):
        await asyncio.sleep(0.1)
        await run.ear.feed(np.zeros(1600, np.float32))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    run: Run | None = None
    task = None
    await ws.send_text(json.dumps({"t": "ready", "mode": "sys", **STATE, **ear.describe(),
                                   "web_backend": config.WEB_BACKEND}))
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                if run is not None:
                    await run.ear.feed(np.frombuffer(msg["bytes"], dtype="<f4"))
                continue
            data = json.loads(msg.get("text") or "{}")
            t = data.get("t")
            if t in ("start", "replay"):
                if run is not None:
                    await run.stop()
                modes = data.get("modes") or ["headstart", "baseline"]
                run = Run(ws, modes, data.get("domain", "assistant"))
                # align session clocks with the first audio sample
                t0 = time.perf_counter()
                for s in run.sessions:
                    s.t0 = t0
                    s.spec.t0 = t0
                if t == "replay":
                    p = AUDIO / Path(data.get("file", "")).name
                    if not p.exists():
                        await ws.send_text(json.dumps({"t": "error", "mode": "sys", "error": f"no such file {p.name}"}))
                        continue

                    async def go(r=run, p=p):
                        await replay(r, p)
                        await r.stop()
                    task = asyncio.create_task(go())
            elif t == "stop" and run is not None:
                await run.stop()
                run = None
    except WebSocketDisconnect:
        pass
    finally:
        if task and not task.done():
            task.cancel()
        if run is not None:
            run.ear.close()


if WEB.exists():
    app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")
