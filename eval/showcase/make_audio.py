"""Showcase set: real-world voice-search queries (the kind people ask a phone assistant), for the Head Start film.

  .venv/bin/python eval/showcase/make_audio.py                  # synthesize missing wavs, write questions.jsonl
  .venv/bin/python eval/showcase/make_audio.py --force sc03,sc07  # re-voice some ('all' for everything)
  .venv/bin/python eval/showcase/make_audio.py --verify         # + Arc streaming/offline check (takes the CPU lock)

Writes eval/showcase/questions.jsonl (same row format as eval/questions.jsonl) and eval/audio/sc_NN.wav
(16 kHz mono PCM16, 0.4 s lead / 0.3 s tail, via eval/make_audio.finish). Voices: Kokoro (studio venv).
expect: [] where the answer changes over time (weather, scores, prices, news) -> judged by hand.
Run the A/B:  .venv/bin/python eval/run_ab.py --questions eval/showcase/questions.jsonl --owner showcase ...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
ROOT = EVAL.parent
sys.path.insert(0, str(EVAL))
import make_audio as base  # noqa: E402  (finish, wer, arc_verify helpers, KOKORO_PY)

OUT = HERE / "questions.jsonl"
AUDIO = EVAL / "audio"

# (id, text, kokoro voice@speed, kind, expect, notes)
Q = [
    ("sc01", "what's the weather gonna be like in Chicago tomorrow", "af_heart@1.0", "weather", [], "weather"),
    ("sc02", "hey, did the Kansas City Chiefs win their game on Sunday", "am_michael@1.0", "sports", [],
     "NFL result"),
    ("sc03", "um, what's the score of the Celtics game right now", "af_nova@1.0", "sports", [], "NBA score; filler"),
    ("sc04", "hey, when is Manchester United playing their next match", "am_fenrir@1.0", "sports", [], "fixture"),
    ("sc05", "hey, what's the latest news about OpenAI this week", "af_bella@1.0", "news", [], "company news"),
    ("sc06", "how much is one Bitcoin worth right now in US dollars", "am_michael@1.0", "current", [], "crypto price"),
    ("sc07", "I'm baking something, how many ounces are in one cup", "af_aoede@1.0", "conversion", ["8|eight"], "kitchen conversion"),
    ("sc08", "what's a hundred US dollars in euros at today's rate", "am_fenrir@1.0", "conversion", [],
     "currency; rate changes"),
    ("sc09", "how many miles is a marathon, like, exactly, to the decimal", "af_nova@1.0", "conversion", ["26"],
     "filler at end"),
    ("sc10", "okay so, how do I get a red wine stain out of a carpet", "af_heart@1.0", "howto",
     ["blot|blotting|vinegar|baking soda|dish soap|salt"], "filler lead-in"),
    ("sc11", "how long should I boil an egg if I want the yolk soft", "am_michael@1.0", "recipe", ["minutes"], "recipe"),
    ("sc12", "how do I take a screenshot on a Mac, just part of the screen", "af_bella@1.0", "tech", ["shift"], "tech help"),
    ("sc13", "uh, my iPhone won't charge anymore, what should I try first", "am_fenrir@1.0", "tech",
     ["cable|port|restart|charger|lint"], "filler; troubleshooting"),
    ("sc14", "what are the usual symptoms of strep throat in adults", "af_aoede@1.0", "health", ["fever"], "health"),
    ("sc15", "is it bad to take ibuprofen on an empty stomach", "af_nova@1.0", "health",
     ["stomach"], "health; yes/no"),
    ("sc16", "what time does the Louvre in Paris close today", "am_michael@1.0", "local", [], "venue hours"),
    ("sc17", "what are some really good ramen places in Seattle", "af_heart@1.0", "local", [], "local recs"),
    ("sc18", "is the Apple Store on Fifth Avenue in New York open twenty four hours", "af_bella@1.0", "local",
     [], "store hours"),
    ("sc19", "do I need a visa to visit Japan from the US", "am_fenrir@1.0", "travel", ["90|ninety"],
     "travel rule"),
    ("sc20", "when's the best time of year to visit Iceland to see the northern lights", "af_aoede@1.0", "travel",
     ["september|october|november|december|january|february|march|winter"], "travel"),
    ("sc21", "how long is the flight from New York to London", "af_nova@1.0", "travel", ["hours"], "travel"),
    ("sc22", "which one's better for photos, the iPhone 17 Pro or the Pixel 10 Pro", "am_michael@1.0", "product",
     [], "product comparison"),
    ("sc23", "what's the best robot vacuum for a house with a lot of pet hair", "af_heart@1.0", "product", [], "product recs"),
    ("sc24", "hmm, what does the word ubiquitous actually mean, exactly", "am_fenrir@1.0", "definition", ["everywhere"], "definition"),
    ("sc25", "so, um, what exactly is a Roth IRA and how does it work", "af_bella@1.0", "definition", ["retirement"], "filler; finance"),
    ("sc26", "what can I use instead of buttermilk in pancakes", "af_bella@0.95", "recipe",
     ["lemon|vinegar|yogurt"], "substitution"),
    ("sc27", "what's the weather in Boston, sorry, I mean Denver", "am_michael@0.95", "weather", [],
     "correction: city"),
    ("sc28", "how many teaspoons are in a tablespoon, actually, make that a quarter cup", "am_fenrir@0.95", "conversion",
     ["12|twelve"], "correction: unit"),
    ("sc29", "when is the next SpaceX Starship launch happening", "af_heart@1.0", "news", [], "upcoming event"),
    ("sc30", "who won the Formula One Grand Prix this past weekend", "am_fenrir@1.0", "sports", [], "sports result"),
]


def take_lock(owner: str, wait_s: float = 1800):
    lock = ROOT / "runs" / ".run_ab.lock"
    deadline = time.time() + wait_s
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, f"{owner} pid={os.getpid()} since={time.strftime('%H:%M:%S')} args=arc-verify\n".encode())
            os.close(fd)
            return lock
        except FileExistsError:
            held = lock.read_text().strip() if lock.exists() else ""
            m = re.search(r"pid=(\d+)", held)
            alive = False
            if m:
                try:
                    os.kill(int(m.group(1)), 0)
                    alive = True
                except (ProcessLookupError, PermissionError) as e:
                    alive = isinstance(e, PermissionError)
            if not alive:
                lock.unlink(missing_ok=True)
                continue
            if time.time() > deadline:
                raise SystemExit(f"lock held: {held}")
            print("waiting for lock:", held, flush=True)
            time.sleep(30)


def synth(rows, force):
    todo = [r for r in rows if r[0] in force or not (AUDIO / f"sc_{r[0][2:]}.wav").exists()]
    if not todo:
        return
    tmp = Path(tempfile.mkdtemp(prefix="hs_sc_"))
    jobs = []
    for qid, text, voice, *_ in todo:
        name, speed = voice.rsplit("@", 1)
        jobs.append({"text": text, "voice": name, "speed": float(speed), "out": str(tmp / f"{qid}.wav"), "qid": qid})
    p = subprocess.run([base.KOKORO_PY, str(EVAL / "tts_kokoro.py")], input=json.dumps(jobs), text=True,
                       capture_output=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-3000:])
        raise SystemExit("kokoro failed")
    for j in jobs:
        x, sr = sf.read(j["out"], dtype="float32")
        base.finish(x, sr, AUDIO / f"sc_{j['qid'][2:]}.wav")
        print("kokoro", j["qid"], j["voice"], j["speed"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", default="")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    force = {r[0] for r in Q} if a.force == "all" else set(filter(None, a.force.split(",")))
    rows = sorted(Q, key=lambda r: r[0])
    AUDIO.mkdir(parents=True, exist_ok=True)
    synth(rows, force)
    old = {}
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                old[o["id"]] = o
    arc = {}
    if a.verify:
        lock = take_lock("showcase-verify")
        try:
            # base.arc_verify reads eval/audio/<qid>.wav; give it the sc_NN names
            base_audio = base.AUDIO
            arc_rows = [(f"sc_{r[0][2:]}", r[1]) for r in rows]
            res = base.arc_verify(arc_rows)
            base.AUDIO = base_audio
            arc = {r[0]: res[f"sc_{r[0][2:]}"] for r in rows}
        finally:
            lock.unlink(missing_ok=True)
    with OUT.open("w") as f:
        for qid, text, voice, kind, expect, notes in rows:
            name, speed = voice.rsplit("@", 1)
            fn = f"sc_{qid[2:]}.wav"
            row = {"id": qid, "file": fn, "text": text, "voice": f"kokoro:{name}", "rate": float(speed),
                   "kind": kind, "expect": expect, "notes": notes,
                   "dur_s": round(sf.info(str(AUDIO / fn)).duration, 2)}
            prev = old.get(qid) if qid not in force else None
            if qid in arc:
                row["arc_text"] = arc[qid]["stream"]
                row["arc_text_offline"] = arc[qid]["offline"]
                row["arc_utts"] = arc[qid]["n_utts"]
                row["arc_wer"] = round(base.wer(text, row["arc_text"]), 3)
            elif prev:
                for k in ("arc_text", "arc_text_offline", "arc_utts", "arc_wer"):
                    if k in prev:
                        row[k] = prev[k]
            f.write(json.dumps(row) + "\n")
    for qid, text, *_ in rows:
        if qid in arc:
            r = arc[qid]
            w = base.wer(text, r["stream"])
            flag = "  <-- CHECK" if w > 0.15 or r["n_utts"] != 1 else ""
            print(f"{qid} wer={w:.2f} utts={r['n_utts']} | {text} | {r['stream']} | off: {r['offline']}{flag}")
    print("wrote", OUT, len(rows), "rows")


if __name__ == "__main__":
    main()
