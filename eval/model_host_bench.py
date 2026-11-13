"""Per-call overhead of MODELS=host (Arc Tagger over the model server's WS) vs the native in-process handle.

    MODEL_HOST_URL=http://127.0.0.1:8200 .venv/bin/python eval/model_host_bench.py

Same calls on both: the session's word-by-word push+poll loop, a revision, an endpoint, an empty poll (pure round
trip) and tag_text on a short answer. Reports median / p95 ms per call and the difference.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import config  # noqa: E402
from server.tagger import RemoteTagger, Tagger  # noqa: E402

Q = "how far is the Copernicus crater from the Apollo twelve landing site on the near side".split()
ANS = "Copernicus is about 93 kilometres wide and sits roughly 400 kilometres north of the Apollo 12 site."
N = 30


def bench(t) -> dict:
    out = {"push+poll / word": [], "revision push+poll": [], "endpoint+poll": [], "empty poll": [], "tag_text": []}
    for _ in range(N):
        t.reset()
        for w in Q:
            s = time.perf_counter(); t.push([w]); t.poll(); out["push+poll / word"].append(time.perf_counter() - s)
        s = time.perf_counter(); t.push(["Tycho", "crater"], from_=3); t.poll()
        out["revision push+poll"].append(time.perf_counter() - s)
        s = time.perf_counter(); t.endpoint(); t.poll(); out["endpoint+poll"].append(time.perf_counter() - s)
        s = time.perf_counter(); t.poll(); out["empty poll"].append(time.perf_counter() - s)
        s = time.perf_counter(); t.tag_text(ANS); out["tag_text"].append(time.perf_counter() - s)
    return {k: (statistics.median(v) * 1e3, sorted(v)[int(0.95 * len(v)) - 1] * 1e3) for k, v in out.items()}


def main():
    config.MODELS = "local"
    native = Tagger(domain="assistant")
    remote = RemoteTagger(domain="assistant")
    bench(native), bench(remote)   # warm
    a, b = bench(native), bench(remote)
    print(f"host {config.MODEL_HOST_URL}, {N} reps of a {len(Q)}-word question")
    print(f"{'call':22} {'native p50':>10} {'host p50':>9} {'+p50':>7} {'native p95':>10} {'host p95':>9}")
    for k in a:
        print(f"{k:22} {a[k][0]:10.3f} {b[k][0]:9.3f} {b[k][0] - a[k][0]:+7.3f} {a[k][1]:10.3f} {b[k][1]:9.3f}")
    native.close(), remote.close()


if __name__ == "__main__":
    main()
