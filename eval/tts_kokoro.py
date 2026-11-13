"""Kokoro TTS worker. Runs under the studio venv (kokoro 0.9.4), NOT the headstart venv:

  /Users/karan/Desktop/work/whissle/wt-studio-general/.venv/bin/python eval/tts_kokoro.py < jobs.json

stdin: JSON list of {"text", "voice", "speed", "out"}; writes 24 kHz mono float WAVs to each "out" (raw, unpadded).
eval/make_audio.py does resampling/padding afterwards.
"""
import json
import sys

import numpy as np
import soundfile as sf
from kokoro import KPipeline


def main():
    jobs = json.load(sys.stdin)
    pipes = {}
    for j in jobs:
        lang = j["voice"][0]          # 'a' = American, 'b' = British
        if lang not in pipes:
            pipes[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
        chunks = [np.asarray(a, np.float32) for _, _, a in pipes[lang](j["text"], voice=j["voice"],
                                                                         speed=float(j.get("speed", 1.0)))]
        sf.write(j["out"], np.concatenate(chunks), 24000)
        print("ok", j["out"], flush=True)


if __name__ == "__main__":
    main()
