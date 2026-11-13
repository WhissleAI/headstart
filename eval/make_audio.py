"""Build the spoken question set: eval/questions.jsonl + eval/audio/qNN.wav (16 kHz mono PCM16).

  .venv/bin/python eval/make_audio.py                 # synthesize all (skip existing wavs), write questions.jsonl
  .venv/bin/python eval/make_audio.py --force q05,q07 # re-synthesize some
  .venv/bin/python eval/make_audio.py --verify        # also run Arc offline on each wav, store "arc_text"

Voices: macOS `say` (in-process) and Kokoro (subprocess under the studio venv, see eval/tts_kokoro.py).
Each wav: ~0.4 s leading silence, ~0.3 s trailing silence, peak-normalized to -3 dBFS.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

EVAL = Path(__file__).resolve().parent
AUDIO = EVAL / "audio"
OUT = EVAL / "questions.jsonl"
KOKORO_PY = "/Users/karan/Desktop/work/whissle/wt-studio-general/.venv/bin/python"
ARC_DIR = "/Users/karan/Desktop/work/whissle/models/arc-v1"

# voice spec: "say:<NAME>@<wpm>" or "kokoro:<voice>@<speed>"
# expect: every item must appear in the answer (case/accent-insensitive, digits' thousands commas removed,
#         word-boundary match); "a|b" inside one item means any of the alternatives.
Q = [
    # ---- factoid, key word at the END
    ("q01", "who founded the company Nvidia", "say:Samantha@180", "factoid", ["jensen", "huang"], "key at end"),
    ("q02", "what is the capital of Australia", "kokoro:am_michael@1.0", "factoid", ["canberra"], "key at end"),
    ("q03", "when did Apollo 11 land on the moon", "kokoro:am_michael@1.0", "factoid", ["1969"], "key early-ish (Apollo 11)"),
    ("q04", "who painted the Mona Lisa", "kokoro:af_heart@1.0", "factoid", ["leonardo"], "key at end"),
    ("q05", "how tall is Mount Everest", "say:Eddy (English (US))@185", "factoid", ["8849|8848|29032|29031|29029"],
     "key at end; number format varies"),
    ("q06", "who discovered penicillin", "kokoro:af_bella@0.95", "factoid", ["fleming"], "key at end, short"),
    ("q07", "what year did the Berlin Wall fall", "say:Flo (English (US))@180", "factoid", ["1989"], "key mid"),
    ("q08", "who was the first woman to win a Nobel Prize", "kokoro:am_fenrir@1.0", "factoid", ["curie"],
     "key at end"),
    ("q09", "what language is spoken in Brazil", "say:Samantha@170", "factoid", ["portuguese"],
     "key at end"),
    ("q10", "who wrote the novel One Hundred Years of Solitude", "kokoro:af_nova@1.0", "factoid", ["marquez"],
     "long title at end"),
    ("q11", "what is the chemical symbol for gold", "say:Karen@180", "factoid", ["au"], "key at end"),
    ("q38", "what is the boiling point of water in Fahrenheit", "kokoro:am_michael@1.05", "factoid", ["212"],
     "key at end"),
    ("q39", "who developed the theory of general relativity", "say:Moira@175", "factoid", ["einstein"],
     "key at end"),
    ("q40", "which company makes the PlayStation", "kokoro:af_aoede@1.0", "factoid", ["sony"], "key at end"),
    ("q41", "what is the longest river in Africa", "kokoro:af_aoede@1.0", "factoid", ["nile"],
     "key at end"),
    ("q42", "where is the headquarters of Toyota", "say:Eddy (English (US))@180", "factoid", ["aichi|toyota city"],
     "key at end"),
    # ---- factoid, key word EARLY
    ("q12", "Python, the programming language, who created it", "kokoro:am_fenrir@1.0", "factoid", ["guido"],
     "key early"),
    ("q13", "the Great Wall of China, which dynasty built most of it", "kokoro:am_michael@0.95", "factoid",
     ["ming"], "key early"),
    ("q14", "Amazon the company, what year was it founded", "say:Samantha@175", "factoid", ["1994"], "key early"),
    ("q15", "Mount Kilimanjaro, which country is it in", "kokoro:am_michael@0.95", "factoid", ["tanzania"],
     "key early"),
    ("q16", "Shakespeare, what year was he born", "kokoro:af_nova@1.0", "factoid", ["1564"],
     "key early"),
    ("q17", "Netflix, who started that company", "kokoro:am_fenrir@1.05", "factoid", ["hastings"], "key early"),
    # ---- long questions with filler
    ("q18", "um, can you tell me who invented the telephone", "say:Flo (English (US))@170", "filler", ["bell"],
     "filler lead-in"),
    ("q19", "so I was wondering, uh, what's the tallest building in the world", "kokoro:af_heart@0.95", "filler",
     ["burj", "khalifa"], "filler lead-in"),
    ("q20", "hey, okay, so, who was the first person to walk on the moon", "say:Reed (English (US))@180",
     "filler", ["armstrong"], "filler lead-in"),
    ("q21", "I've always wanted to know, um, how many bones are there in the adult human body",
     "say:Reed (English (US))@185", "filler", ["206"], "long, filler"),
    ("q22", "uh, quick question, what is the largest planet in our solar system", "say:Samantha@185", "filler",
     ["jupiter"], "filler lead-in"),
    ("q23", "so, like, can you tell me what the speed of light is", "kokoro:af_nova@1.0", "filler",
     ["299792|299,792|186000|186282|300000"], "filler; number format varies"),
    ("q24", "um, I forget, who directed the movie Jaws", "say:Daniel@170", "filler", ["spielberg"],
     "filler; key at end"),
    ("q25", "okay so my kid asked me this, and I didn't know, why is the sky blue", "kokoro:af_bella@1.0",
     "filler", ["scatter|scattering|scattered"], "long lead-in, why-question"),
    # ---- self-corrections
    ("q26", "who wrote Hamlet, sorry, I mean Pride and Prejudice", "say:Samantha@170", "correction", ["austen"],
     "correction: work title"),
    ("q27", "what's the capital of Canada, actually make that Turkey", "kokoro:am_fenrir@1.0", "correction",
     ["ankara"], "correction: country"),
    ("q28", "when was Google founded, no wait, I mean Apple", "kokoro:af_heart@1.0", "correction",
     ["1976"], "correction: company"),
    ("q29", "who founded Microsoft, sorry, I meant Amazon", "say:Samantha@175", "correction", ["bezos"],
     "correction: company"),
    ("q30", "where was Mozart born, actually make that Beethoven", "kokoro:af_nova@0.95", "correction", ["bonn"],
     "correction: person"),
    ("q31", "what river runs through Paris, sorry, I mean London", "kokoro:am_michael@1.0", "correction",
     ["thames"], "correction: city"),
    ("q32", "what currency does Japan use, scratch that, Switzerland", "kokoro:am_fenrir@1.0",
     "correction", ["franc"], "correction: country"),
    ("q33", "who was the first president of France, no wait, of the United States", "kokoro:af_aoede@1.0",
     "correction", ["washington"], "correction: country, partial restatement"),
    # ---- current events (excluded while web backend is Wikipedia)
    ("q34", "what's the weather in San Francisco today", "say:Flo (English (US))@180", "current", [],
     "weather; not answerable from Wikipedia"),
    ("q35", "what's the latest news about SpaceX", "kokoro:af_bella@1.0", "current", [], "news"),
    ("q36", "what is Apple's stock price right now", "say:Reed (English (US))@175", "current", [], "stock"),
    ("q37", "will it rain in Seattle tomorrow", "kokoro:am_fenrir@1.0", "current", [], "weather"),
]

LEAD, TAIL = 0.4, 0.3


def _trim(x: np.ndarray, sr: int, thr_db: float = -45) -> np.ndarray:
    """Trim leading/trailing near-silence (TTS engines add their own)."""
    if not len(x):
        return x
    peak = np.max(np.abs(x)) + 1e-9
    win = max(1, sr // 100)
    n = len(x) // win
    e = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1)) / peak
    on = np.where(20 * np.log10(e + 1e-12) > thr_db)[0]
    if not len(on):
        return x
    a, b = max(0, on[0] - 2) * win, min(len(x), (on[-1] + 3) * win)
    return x[a:b]


def finish(raw: np.ndarray, sr: int, out: Path):
    import librosa
    x = raw.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    x = _trim(x, 16000)
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.7
    x = np.concatenate([np.zeros(int(LEAD * 16000), np.float32), x, np.zeros(int(TAIL * 16000), np.float32)])
    sf.write(str(out), x, 16000, subtype="PCM_16")


def voice_parts(v: str):
    eng, rest = v.split(":", 1)
    name, rate = rest.rsplit("@", 1)
    return eng, name, rate


def synth(rows, force: set[str]):
    AUDIO.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if r[0] in force or not (AUDIO / f"{r[0]}.wav").exists()]
    tmp = Path(tempfile.mkdtemp(prefix="hs_tts_"))
    kjobs = []
    for qid, text, voice, *_ in todo:
        eng, name, rate = voice_parts(voice)
        if eng == "say":
            aiff = tmp / f"{qid}.aiff"
            subprocess.run(["say", "-v", name, "-r", rate, "-o", str(aiff), text], check=True)
            x, sr = sf.read(str(aiff), dtype="float32")
            finish(x, sr, AUDIO / f"{qid}.wav")
            print("say   ", qid, name, rate)
        else:
            kjobs.append({"text": text, "voice": name, "speed": float(rate), "out": str(tmp / f"{qid}.wav"),
                          "qid": qid})
    if kjobs:
        p = subprocess.run([KOKORO_PY, str(EVAL / "tts_kokoro.py")], input=json.dumps(kjobs), text=True,
                           capture_output=True)
        if p.returncode != 0:
            sys.stderr.write(p.stderr[-3000:])
            raise SystemExit("kokoro failed")
        for j in kjobs:
            x, sr = sf.read(j["out"], dtype="float32")
            finish(x, sr, AUDIO / f"{j['qid']}.wav")
            print("kokoro", j["qid"], j["voice"], j["speed"])


def arc_verify(rows) -> dict:
    sys.path.insert(0, ARC_DIR)
    from arc_audio import Arc
    arc = Arc.from_pretrained(ARC_DIR, device="cpu", threads=4)
    out = {}
    for qid, text, *_ in rows:
        path = str(AUDIO / f"{qid}.wav")
        r = arc.predict(path, domain="assistant", transcript="builtin")
        offline = " ".join(u.get("text", "") for u in (r.get("transcript") or [])).strip()
        # streaming, exactly as the harness feeds it (100 ms chunks + 1.2 s of trailing zeros), not wall-paced
        wav, _ = sf.read(path, dtype="float32")
        wav = np.concatenate([wav, np.zeros(19200, np.float32)])
        st = arc.stream(domain="assistant", transcript="builtin", partials=True)
        utts = []
        for i in range(0, len(wav), 1600):
            utts += st.feed(wav[i:i + 1600]).get("transcript") or []
        utts += st.flush().get("transcript") or []
        out[qid] = {"stream": " ".join(u.get("text", "") for u in utts).strip(), "offline": offline,
                    "n_utts": len(utts)}
    return out


def _n(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()


def wer(ref: str, hyp: str) -> float:
    r, h = _n(ref), _n(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", default="", help="comma-separated ids to re-synthesize ('all' for everything)")
    ap.add_argument("--verify", action="store_true", help="run Arc offline and store arc_text / arc_wer")
    a = ap.parse_args()
    force = {r[0] for r in Q} if a.force == "all" else set(filter(None, a.force.split(",")))
    rows = sorted(Q, key=lambda r: r[0])
    synth(rows, force)
    old = {}
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                old[o["id"]] = o
    arc = arc_verify(rows) if a.verify else {}
    with OUT.open("w") as f:
        for qid, text, voice, kind, expect, notes in rows:
            eng, name, rate = voice_parts(voice)
            dur = sf.info(str(AUDIO / f"{qid}.wav")).duration
            row = {"id": qid, "file": f"{qid}.wav", "text": text, "voice": f"{eng}:{name}",
                   "rate": float(rate), "kind": kind, "expect": expect, "notes": notes, "dur_s": round(dur, 2)}
            prev = old.get(qid) if qid not in force else None
            if qid in arc:
                row["arc_text"] = arc[qid]["stream"]
                row["arc_text_offline"] = arc[qid]["offline"]
                row["arc_utts"] = arc[qid]["n_utts"]
                row["arc_wer"] = round(wer(text, row["arc_text"]), 3)
            elif prev and "arc_text" in prev:
                for k in ("arc_text", "arc_text_offline", "arc_utts", "arc_wer"):
                    if k in prev:
                        row[k] = prev[k]
            f.write(json.dumps(row) + "\n")
    if arc:
        for qid, text, *_ in rows:
            a = arc[qid]
            w = wer(text, a["stream"])
            flag = "  <-- CHECK" if w > 0.15 or a["n_utts"] != 1 else ""
            print(f"{qid} wer={w:.2f} utts={a['n_utts']} | {text} | {a['stream']} | off: {a['offline']}{flag}")
    print("wrote", OUT, len(rows), "rows")


if __name__ == "__main__":
    main()
