"""Record film clips of the Race replay in film mode (web/film.js) with headless Chromium.

    .venv-with-playwright/bin/python film/capture.py q01 q27              # race clips -> film/clips/<id>.mp4 + .json
    python film/capture.py --broll timeline,readout,watches q27           # b-roll close-ups -> film/clips/broll_<focus>_<id>.mp4
    python film/capture.py --showcase                                     # every id in eval/showcase/selected.json

Each clip: the page opens /?film=1, the screencast starts, after --lead s the replay is triggered, the clip runs until
both answers are in (plus --hold s) and the run is done. Frames come from the CDP screencast (JPEG, timestamped by the
browser) and are re-timed onto a constant 30 fps grid, so the picture is exactly what the page showed, when it showed
it. The question audio is muxed at the moment the server's clock says sample 0 was fed (film.js reports that instant
as epoch ms: every server event carries ms since audio start), so words on screen line up with the voice.

Timing fields in <id>.json are seconds on the clip's own timeline (t=0 = first frame) unless named *_ms.
Takes the shared CPU lock (runs/.run_ab.lock, same protocol as eval/run_ab.py) for the whole session.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "film" / "clips"
AUDIO = ROOT / "eval" / "audio"
LOCK = ROOT / "runs" / ".run_ab.lock"
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FPS = 30
W, H = 1920, 1080


# ------------------------------------------------------------------ lock (same protocol as eval/run_ab.py)
def acquire_lock(owner: str, wait_s: float) -> None:
    deadline = time.time() + wait_s
    LOCK.parent.mkdir(exist_ok=True)
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            held = LOCK.read_text().strip() if LOCK.exists() else ""
            m = re.search(r"pid=(\d+)", held)
            alive = False
            if m:
                try:
                    os.kill(int(m.group(1)), 0)
                    alive = True
                except (ProcessLookupError, PermissionError) as e:
                    alive = isinstance(e, PermissionError)
            if not alive:
                print(f"stale lock removed: {held!r}", file=sys.stderr)
                LOCK.unlink(missing_ok=True)
                continue
            if time.time() > deadline:
                raise SystemExit(f"CPU lock held ({held}); use --wait-lock S")
            print(f"{time.strftime('%H:%M:%S')} waiting for lock: {held}", file=sys.stderr, flush=True)
            time.sleep(30)
    os.write(fd, f"{owner} pid={os.getpid()} since={time.strftime('%H:%M:%S')} args={' '.join(sys.argv[1:])}\n".encode())
    os.close(fd)


def release_lock() -> None:
    try:
        if f"pid={os.getpid()}" in LOCK.read_text():
            LOCK.unlink()
    except FileNotFoundError:
        pass


# ------------------------------------------------------------------ questions
def question_index() -> dict:
    idx = {}
    for p in [ROOT / "eval" / "questions.jsonl", ROOT / "eval" / "showcase" / "questions.jsonl",
              ROOT / "index" / "questions_local.jsonl"]:
        if p.exists():
            for ln in p.read_text().splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    if r.get("id"):
                        idx[r["id"]] = r
    return idx


def resolve(qid: str, idx: dict) -> tuple[str, str, dict]:
    """id or file name -> (id, wav file name in eval/audio, question row)."""
    if qid.endswith(".wav"):
        row = next((r for r in idx.values() if r.get("file") == qid), {})
        return Path(qid).stem, qid, row
    row = idx.get(qid)
    if row and row.get("file"):
        return qid, row["file"], row
    if (AUDIO / f"{qid}.wav").exists():
        return qid, f"{qid}.wav", {}
    raise SystemExit(f"unknown question {qid!r} (not in the question files, no eval/audio/{qid}.wav)")


# ------------------------------------------------------------------ recording
def record(page, cdp, file: str, lead: float, hold: float, timeout: float) -> tuple[list, dict, list]:
    frames: list[tuple[float, bytes]] = []

    def on_frame(p):
        frames.append((p["metadata"]["timestamp"], base64.b64decode(p["data"])))
        try:
            cdp.send("Page.screencastFrameAck", {"sessionId": p["sessionId"]})
        except Exception:
            pass

    cdp.on("Page.screencastFrame", on_frame)
    cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": W, "maxHeight": H, "everyNthFrame": 1})
    # nudge one frame out even if nothing animates yet
    page.evaluate("document.body.style.outline = '0px solid transparent'")
    page.wait_for_timeout(int(lead * 1000))
    page.evaluate("(f) => window.__hs.replay(f)", file)
    t_end = time.time() + timeout
    st = None
    while time.time() < t_end:
        page.wait_for_timeout(100)
        st = page.evaluate("window.__film.state()")
        if st and st.get("both") and st.get("done"):
            break
    else:
        print(f"  timeout: state={json.dumps(st)[:300] if st else None}", file=sys.stderr)
    page.wait_for_timeout(int(hold * 1000))
    stop_epoch = time.time()
    cdp.send("Page.stopScreencast")
    page.wait_for_timeout(150)
    events = page.evaluate("window.__film.events()")
    return frames, {**(st or {}), "_stop_epoch": stop_epoch}, events


def encode(frames, out_mp4: Path, wav: Path | None, audio_offset_s: float | None, t_end: float) -> float:
    """Timestamped frames -> CFR 30 fps H.264 yuv420p (CRF 18) with the question audio delayed to its offset.
    Returns the clip duration (s)."""
    if not frames:
        raise RuntimeError("no frames captured")
    frames.sort(key=lambda x: x[0])
    t0 = frames[0][0]
    dur = max(t_end - t0, frames[-1][0] - t0 + 1 / FPS)
    tmp = Path(tempfile.mkdtemp(prefix="hsfilm_"))
    try:
        lines = ["ffconcat version 1.0"]
        for i, (ts, data) in enumerate(frames):
            fp = tmp / f"{i:06d}.jpg"
            fp.write_bytes(data)
            nxt = frames[i + 1][0] if i + 1 < len(frames) else t0 + dur
            lines += [f"file '{fp.name}'", f"duration {max(nxt - ts, 0.001):.6f}"]
        lines.append(f"file '{len(frames) - 1:06d}.jpg'")
        (tmp / "list.txt").write_text("\n".join(lines) + "\n")
        cmd = [FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt")]
        if wav is not None and audio_offset_s is not None:
            cmd += ["-i", str(wav)]
        # screencast JPEGs are full-range: convert to limited-range yuv420p (what players and editors expect)
        vf = f"scale={W}:{H}:flags=lanczos:in_range=full:out_range=tv:in_color_matrix=bt601:out_color_matrix=bt709,fps={FPS},format=yuv420p"
        cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-r", str(FPS), "-pix_fmt", "yuv420p",
                "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-movflags", "+faststart", "-t", f"{dur:.3f}"]
        if wav is not None and audio_offset_s is not None:
            d = max(0, int(round(audio_offset_s * 1000)))
            cmd += ["-af", f"adelay={d}:all=1,apad,aresample=48000", "-ac", "2", "-c:a", "aac", "-b:a", "192k",
                    "-map", "0:v", "-map", "1:a"]
        cmd.append(str(out_mp4))
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dur


def capture_one(page, cdp, base: str, qid: str, file: str, row: dict, out: Path, a, focus: str | None = None) -> dict:
    url = f"{base}/?film=1" + (f"&focus={focus}" if focus else "")
    page.goto(url, wait_until="networkidle")
    page.wait_for_function("window.__hs && window.__hs.ready && window.__film && window.__film.active", timeout=120000)
    page.evaluate("document.fonts.ready")
    page.wait_for_timeout(300)
    frames, st, events = record(page, cdp, file, a.lead, a.hold, a.timeout)
    t0 = min(f[0] for f in frames) if frames else st["_stop_epoch"]
    a0 = st.get("audio0_epoch_ms")
    offset = (a0 / 1000 - t0) if a0 else None
    out.parent.mkdir(parents=True, exist_ok=True)
    dur = encode(frames, out, AUDIO / file, offset, st["_stop_epoch"])
    sides = st.get("sides", {})
    se = st.get("speech_end_ms")
    clip = lambda ms: None if ms is None or offset is None else round(offset + ms / 1000, 3)   # noqa: E731
    arc, std = sides.get("headstart", {}), sides.get("baseline", {})
    meta = {
        "id": qid, "file": file, "focus": focus, "clip": out.name, "fps": FPS, "size": [W, H], "duration_s": round(dur, 3),
        "frames_captured": len(frames), "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "text": row.get("text"), "kind": row.get("kind"), "expect": row.get("expect"),
        "question_heard": st.get("question"), "query": st.get("query"),
        "t_start": None if offset is None else round(offset, 3),          # audio sample 0 / replay start on the clip timeline
        "speech_end_s": clip(se),
        "arc_ready_s": clip(arc.get("ready_abs_ms")), "std_ready_s": clip(std.get("ready_abs_ms")),
        "arc_ready_after_stop_ms": arc.get("ready_after_stop_ms"), "std_ready_after_stop_ms": std.get("ready_after_stop_ms"),
        "delta_ms": st.get("delta_ms"), "winner": st.get("winner"),
        "answers": {"arc": arc.get("answer"), "standard": std.get("answer")},
        "sources": {"arc": arc.get("sources"), "standard": std.get("sources")},
        "searches": {"arc": arc.get("searches"), "standard": std.get("searches")},
        "metrics": {"arc": arc.get("metric"), "standard": std.get("metric")},
        "frame": st.get("frame"),
        "events": [{**{k: v for k, v in e.items() if v is not None}, "t_clip": clip(e["ms"]) if isinstance(e.get("ms"), (int, float)) else None}
                   for e in events],
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", help="question ids (q01, sc_03) or wav names in eval/audio")
    ap.add_argument("--showcase", action="store_true", help="the ids in eval/showcase/selected.json")
    ap.add_argument("--broll", default="", help="comma list of focus modes (timeline,readout,watches): b-roll clips instead")
    ap.add_argument("--base", default=os.environ.get("HS_URL", "http://localhost:8770"))
    ap.add_argument("--out", default=str(CLIPS))
    ap.add_argument("--lead", type=float, default=1.0, help="s of the empty page before the replay starts")
    ap.add_argument("--hold", type=float, default=1.5, help="s held after both answers are in")
    ap.add_argument("--timeout", type=float, default=45)
    ap.add_argument("--warmup", default="q02.wav", help="an unrecorded replay first (warms the web + answer paths); '' = none")
    ap.add_argument("--wait-lock", type=float, default=3600)
    ap.add_argument("--no-lock", action="store_true", help="skip the CPU lock (layout checks only)")
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args()

    ids = list(a.ids)
    if a.showcase:
        sel = json.loads((ROOT / "eval" / "showcase" / "selected.json").read_text())
        ids += [x if isinstance(x, str) else x["id"] for x in (sel.get("ids") or sel.get("selected") or [])]
    if not ids:
        ap.error("no ids")
    idx = question_index()
    todo = [resolve(q, idx) for q in ids]
    focuses = [f for f in a.broll.split(",") if f]
    out = Path(a.out)

    from playwright.sync_api import sync_playwright
    if not a.no_lock:
        acquire_lock("film-capture", a.wait_lock)
    try:
        with sync_playwright() as pw:
            br = pw.chromium.launch(headless=not a.headed, args=["--autoplay-policy=no-user-gesture-required",
                                                                  "--force-device-scale-factor=1", "--hide-scrollbars"])
            ctx = br.new_context(viewport={"width": W, "height": H}, device_scale_factor=1, color_scheme="dark")
            page = ctx.new_page()
            page.on("console", lambda m: m.type == "error" and print(f"  console: {m.text}", file=sys.stderr))
            cdp = ctx.new_cdp_session(page)
            if a.warmup:
                page.goto(f"{a.base}/?film=1", wait_until="networkidle")
                page.wait_for_function("window.__hs && window.__hs.ready", timeout=120000)
                page.evaluate("(f) => window.__hs.replay(f)", a.warmup)
                page.wait_for_function("() => { const s = window.__film.state(); return s && s.both && s.done; }", timeout=a.timeout * 1000)
                print(f"warm-up {a.warmup} done", flush=True)
            for qid, file, row in todo:
                for focus in (focuses or [None]):
                    name = f"broll_{focus}_{qid}.mp4" if focus else f"{qid}.mp4"
                    t = time.time()
                    m = capture_one(page, cdp, a.base, qid, file, row, out / name, a, focus)
                    print(f"{name}: {m['duration_s']} s, {m['frames_captured']} frames · std {m['std_ready_after_stop_ms']} ms · "
                          f"arc {m['arc_ready_after_stop_ms']} ms · winner {m['winner']} ({time.time() - t:.0f} s)", flush=True)
            br.close()
    finally:
        if not a.no_lock:
            release_lock()
    return 0


if __name__ == "__main__":
    sys.exit(main())
