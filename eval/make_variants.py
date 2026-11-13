"""Robustness variants of the spoken question set (no downloads: noise is synthesized from our own audio).

  .venv/bin/python eval/make_variants.py noise --snr 20,10     # eval/audio_noisy/snr20/, snr10/  (all questions)
  .venv/bin/python eval/make_variants.py rate  --rates 1.25,0.8 # eval/audio_rate/x1.25/, x0.8/   (RATE_IDS)

noise: "hackathon room" = babble (6 OTHER questions' speech, random offsets/gains, summed, looped) + pink noise
(~25 % of the noise power), mixed at the given SNR, where SNR = speech power over the speech-active 10 ms frames
of the clean file (frames within 40 dB of the peak) / mean noise power. Noise runs through the whole file and
for NOISE_TAIL_S after it (in a room the chatter doesn't stop when you do), so the ASR / VAD endpoint has to find
the pause inside the noise. (run_ab.py then appends its usual 1.2 s of digital zeros.)

rate-tts (use this): re-synthesize RATE_IDS with the same voice at its own rate x factor (macOS `say -r`, Kokoro
speed) -> eval/audio_rate/tts_x<f>/. Real faster/slower speech, same voice, same padding as make_audio.py.
rate (DON'T use for ASR claims): librosa phase-vocoder time stretch of the speech region. Its 128 ms-window
artifacts make Arc hallucinate ("who painted the Mona Lisa" -> "please remind me now", WER ~70 %): it measures
the vocoder, not speaking rate.

Each output dir gets meta.json: {file: {"speech_end_s": clean speech end in THIS file's timeline, ...}} so
run_ab.py can measure ready_vad against the true end of speech instead of the end of the noise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
CLEAN = EVAL / "audio"
SR = 16000
NOISE_TAIL_S = 2.0
N_BABBLE = 6
PINK_SHARE = 0.25
# 20 questions across kinds for the rate test (factoid key-late / key-early, filler, correction, local)
RATE_IDS = ["q01", "q02", "q04", "q05", "q08", "q12", "q14", "q16", "q18", "q20", "q21", "q24", "q25", "q26",
            "q27", "q29", "q31", "q33", "local01", "local05"]


def questions() -> list[dict]:
    rows = []
    for p in (EVAL / "questions.jsonl", ROOT / "index" / "questions_local.jsonl"):
        if p.exists():
            rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [r for r in rows if (CLEAN / r["file"]).exists()]


def load(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == SR, (p, sr)
    return x


def frames_db(x: np.ndarray, win: int = 160) -> np.ndarray:
    n = len(x) // win
    e = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1))
    return 20 * np.log10(e / (e.max() + 1e-12) + 1e-12)


def speech_span(x: np.ndarray, thr_db: float = -40.0) -> tuple[int, int]:
    on = np.where(frames_db(x) > thr_db)[0]
    if not len(on):
        return 0, len(x)
    return int(on[0] * 160), int((on[-1] + 1) * 160)


def speech_power(x: np.ndarray, thr_db: float = -40.0) -> float:
    db = frames_db(x)
    n = len(db)
    fr = x[: n * 160].reshape(n, 160)
    act = fr[db > thr_db]
    return float(np.mean(act ** 2)) if len(act) else float(np.mean(x ** 2))


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    f = np.fft.rfftfreq(n, 1 / SR)
    spec = (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))) / np.sqrt(np.maximum(f, 20.0))
    spec[f < 40] = 0
    y = np.fft.irfft(spec, n).astype(np.float32)
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-12)


def babble(n: int, exclude: str, pool: list[tuple[str, np.ndarray]], rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(n, np.float32)
    cands = [w for qid, w in pool if qid != exclude]
    for k in rng.choice(len(cands), size=N_BABBLE, replace=False):
        w = cands[k]
        a, b = speech_span(w)
        w = w[a:b]
        w = w / (np.sqrt(np.mean(w ** 2)) + 1e-12)
        # loop the talker with short random gaps so the voice keeps going through the whole file
        seg = []
        tot = int(rng.integers(0, SR))           # random start offset
        seg.append(np.zeros(tot, np.float32))
        while tot < n:
            gap = np.zeros(int(rng.uniform(0.1, 0.6) * SR), np.float32)
            seg += [w, gap]
            tot += len(w) + len(gap)
        out += float(rng.uniform(0.5, 1.0)) * np.concatenate(seg)[:n]
    return out / (np.sqrt(np.mean(out ** 2)) + 1e-12)


def make_noise(snrs: list[float], seed: int = 7):
    qs = questions()
    pool = [(q["id"], load(CLEAN / q["file"])) for q in qs if q["id"].startswith("q")]
    for snr in snrs:
        rng = np.random.default_rng(seed)
        d = EVAL / "audio_noisy" / f"snr{int(snr)}"
        d.mkdir(parents=True, exist_ok=True)
        meta = {"_": {"kind": "noise", "snr_db": snr, "noise": f"babble x{N_BABBLE} (other questions) + pink "
                      f"{int(PINK_SHARE * 100)}% power", "noise_tail_s": NOISE_TAIL_S, "seed": seed}}
        for q in qs:
            x = load(CLEAN / q["file"])
            a, b = speech_span(x)
            n = len(x) + int(NOISE_TAIL_S * SR)
            nz = np.sqrt(1 - PINK_SHARE) * babble(n, q["id"], pool, rng) + np.sqrt(PINK_SHARE) * pink(n, rng)
            nz *= np.sqrt(speech_power(x) / 10 ** (snr / 10)) / (np.sqrt(np.mean(nz ** 2)) + 1e-12)
            y = np.concatenate([x, np.zeros(n - len(x), np.float32)]) + nz
            peak = float(np.max(np.abs(y)))
            if peak > 0.97:
                y *= 0.97 / peak
            sf.write(str(d / q["file"]), y.astype(np.float32), SR, subtype="PCM_16")
            meta[q["file"]] = {"speech_end_s": round(b / SR, 3), "speech_start_s": round(a / SR, 3),
                               "noise_rms": round(float(np.sqrt(np.mean(nz ** 2))) * (min(1, 0.97 / peak)), 4)}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"wrote {len(qs)} files to {d}")


def make_rate(rates: list[float]):
    import librosa
    qs = {q["id"]: q for q in questions()}
    for rate in rates:
        d = EVAL / "audio_rate" / f"x{rate:g}"
        d.mkdir(parents=True, exist_ok=True)
        meta = {"_": {"kind": "rate", "rate": rate, "method": "librosa.effects.time_stretch (speech region only)",
                      "ids": RATE_IDS}}
        for qid in RATE_IDS:
            q = qs[qid]
            x = load(CLEAN / q["file"])
            a, b = speech_span(x)
            pad = int(0.03 * SR)
            a2, b2 = max(0, a - pad), min(len(x), b + pad)
            mid = librosa.effects.time_stretch(x[a2:b2], rate=rate).astype(np.float32)
            y = np.concatenate([x[:a2], mid, x[b2:]])
            y = y / (np.max(np.abs(y)) + 1e-9) * 0.7
            sf.write(str(d / q["file"]), y, SR, subtype="PCM_16")
            a3, b3 = speech_span(y)
            meta[q["file"]] = {"speech_end_s": round(b3 / SR, 3), "speech_start_s": round(a3 / SR, 3),
                               "dur_speech_s": round((b3 - a3) / SR, 2), "dur_speech_clean_s": round((b - a) / SR, 2)}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"wrote {len(RATE_IDS)} files to {d}")


def make_rate_tts(factors: list[float]):
    import subprocess
    import tempfile
    import make_audio as MA
    qs = {q["id"]: q for q in questions()}
    for fac in factors:
        d = EVAL / "audio_rate" / f"tts_x{fac:g}"
        d.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="hs_rate_"))
        meta = {"_": {"kind": "rate", "rate": fac, "method": "TTS re-synthesis at voice rate x factor", "ids": RATE_IDS}}
        kjobs = []
        for qid in RATE_IDS:
            q = qs[qid]
            eng, name = q["voice"].split(":", 1)
            base = q.get("rate") or (175.0 if eng == "say" else 1.0)
            if eng == "say":
                aiff = tmp / f"{qid}.aiff"
                subprocess.run(["say", "-v", name, "-r", str(round(base * fac)), "-o", str(aiff), q["text"]],
                               check=True)
                x, sr = sf.read(str(aiff), dtype="float32")
                MA.finish(x, sr, d / q["file"])
            else:
                kjobs.append({"text": q["text"], "voice": name, "speed": base * fac, "out": str(tmp / f"{qid}.wav"),
                              "file": q["file"]})
        if kjobs:
            pr = subprocess.run([MA.KOKORO_PY, str(EVAL / "tts_kokoro.py")], input=json.dumps(kjobs), text=True,
                                capture_output=True)
            if pr.returncode != 0:
                raise SystemExit(pr.stderr[-2000:])
            for j in kjobs:
                x, sr = sf.read(j["out"], dtype="float32")
                MA.finish(x, sr, d / j["file"])
        for qid in RATE_IDS:
            f = qs[qid]["file"]
            y, c = load(d / f), load(CLEAN / f)
            a, b = speech_span(y)
            a0, b0 = speech_span(c)
            meta[f] = {"speech_end_s": round(b / SR, 3), "dur_speech_s": round((b - a) / SR, 2),
                       "dur_speech_clean_s": round((b0 - a0) / SR, 2)}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        r = [meta[qs[i]["file"]]["dur_speech_clean_s"] / meta[qs[i]["file"]]["dur_speech_s"] for i in RATE_IDS]
        print(f"wrote {len(RATE_IDS)} files to {d}; effective speed-up median {float(np.median(r)):.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["noise", "rate", "rate-tts"])
    ap.add_argument("--snr", default="20,10")
    ap.add_argument("--rates", default="1.25,0.8")
    a = ap.parse_args()
    if a.what == "noise":
        make_noise([float(s) for s in a.snr.split(",")])
    elif a.what == "rate-tts":
        make_rate_tts([float(r) for r in a.rates.split(",")])
    else:
        make_rate([float(r) for r in a.rates.split(",")])


if __name__ == "__main__":
    sys.exit(main())
