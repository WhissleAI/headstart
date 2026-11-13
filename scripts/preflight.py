"""Pre-demo check against a running server: health (with live probes of the web lane and the answer model) + one real
WS replay in Race mode. Prints a GO / NO-GO checklist. Exit 0 = GO, 3 = GO (degraded: a fallback is active), 1 = NO-GO.

    .venv/bin/python scripts/preflight.py --port 8770 [--file q01.wav] [--local local_05.wav]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
OK, WARN, FAIL = "OK", "WARN", "FAIL"


def expects(name: str) -> list[str]:
    for p in (ROOT / "eval" / "questions.jsonl", ROOT / "index" / "questions_local.jsonl"):
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("file") == name:
                        return [w.lower() for w in r.get("expect", [])]
    return []


async def replay(port: int, name: str, timeout: float = 40.0) -> dict:
    """{mode: {"answer": str, "metric": dict | None}}, plus "_errors" and "_ready" (the ready event)."""
    out = {"headstart": {"answer": "", "metric": None}, "baseline": {"answer": "", "metric": None}, "_errors": []}
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws", max_size=None, open_timeout=5) as ws:
        t_end = time.monotonic() + timeout
        sent = False
        while time.monotonic() < t_end:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, t_end - time.monotonic()))
            except asyncio.TimeoutError:
                break
            e = json.loads(raw)
            m, t = e.get("mode"), e.get("t")
            if m == "sys" and t == "ready" and not sent:
                out["_ready"] = e
                await ws.send(json.dumps({"t": "replay", "file": name, "modes": ["headstart", "baseline"]}))
                sent = True
            elif m == "sys" and t == "error":
                out["_errors"].append(e.get("error"))
            elif m == "sys" and t == "run_done":
                break
            elif m in ("headstart", "baseline"):
                if t == "answer":
                    out[m]["answer"] += e.get("delta") or ""
                elif t == "metric":
                    out[m]["metric"] = e
                elif t == "extended":
                    out[m]["answer"] = ""
                elif t == "answer_error":
                    out["_errors"].append(f"{m}: {e.get('error')}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--file", default="q01.wav", help="web question to replay (eval/audio)")
    ap.add_argument("--local", default="", help="also replay a local-moments question, e.g. local_05.wav")
    ap.add_argument("--max-hs-ms", type=int, default=2500, help="Head Start ready time above this = FAIL")
    a = ap.parse_args()

    rows: list[tuple[str, str, str]] = []

    def check(status, what, detail=""):
        rows.append((status, what, detail))

    base = f"http://127.0.0.1:{a.port}"
    try:
        with urllib.request.urlopen(base + "/api/health?probe=1", timeout=20) as r:
            h = json.loads(r.read())
    except Exception as e:
        print(f"[FAIL] server not answering on :{a.port}: {e}")
        print("\nNO-GO")
        return 1

    check(OK if h.get("warm") else FAIL, "server warm", f"warm-up {h.get('warm_s', '?')} s" if h.get("warm") else "still warming")
    check(OK if h.get("local_docs") else WARN, "local moments index", f"{h.get('local_docs', 0)} moments")
    ear = h.get("ear", "?")
    check(OK if ear == h.get("ear_requested", ear) else WARN, "ear", f"{ear} (requested {h.get('ear_requested', ear)})")

    w, an = h.get("web") or {}, h.get("answer") or {}
    pr = h.get("probe") or {}
    if w.get("status") == "live":
        check(OK, "web lane reachable", f"{w.get('backend')} live, probe {pr.get('web_ms')} ms")
    elif w.get("status") == "cache":
        check(WARN, "web lane", f"WEB_BACKEND=cache: replaying {w.get('cache_entries')} recorded responses (flagged in UI)")
    else:
        check(WARN if w.get("cache_entries") else FAIL, "web lane reachable",
              f"DOWN ({w.get('last_error')}); fallback: " +
              (f"recorded cache, {w.get('cache_entries')} entries" if w.get("cache_entries") else "NONE (web questions get no sources)"))
    if w.get("status") != "cache":
        check(OK if w.get("cache_entries") else WARN, "web replay cache",
              f"{w.get('cache_entries', 0)} entries" + (f", recorded {w.get('cache_recorded')}" if w.get("cache_recorded") else
                                                       " (record: eval/record_web_cache.py)"))
    if an.get("status") == "live":
        check(OK, "Anthropic reachable", f"{an.get('model')}, probe {pr.get('answer_ms')} ms")
    else:
        check(WARN if an.get("fallback") == "extractive" else FAIL, "Anthropic reachable",
              f"DOWN ({an.get('last_error')}); fallback: {an.get('fallback')} answers (flagged in UI)")

    def judge(name: str, label: str):
        try:
            res = asyncio.run(replay(a.port, name))
        except Exception as e:
            check(FAIL, f"{label} replay {name}", f"{type(e).__name__}: {e}")
            return
        exp = expects(name)
        hs, bl = res["headstart"], res["baseline"]
        both = hs["metric"] is not None and bl["metric"] is not None
        check(OK if both else FAIL, f"{label} replay {name}: both modes finished",
              "" if both else f"headstart={'yes' if hs['metric'] else 'no'} baseline={'yes' if bl['metric'] else 'no'}"
              + (f"; errors: {res['_errors'][:2]}" if res["_errors"] else ""))
        for m, r in (("headstart", hs), ("baseline", bl)):
            ans = r["answer"].strip()
            met = r["metric"] or {}
            flags = []
            if met.get("answer_fallback"):
                flags.append(f"{met['answer_fallback']} answer")
            if met.get("web_replayed"):
                flags.append(f"cached web x{met['web_replayed']}")
            right = (not exp) or any(x in ans.lower() for x in exp)
            st = FAIL if not ans else (WARN if (flags or not right) else OK)
            check(st, f"  {m} answer", (f"[{', '.join(flags)}] " if flags else "") +
                  (repr(ans[:90]) if ans else "EMPTY") + ("" if right else f"  (expected {exp})"))
        if both:
            h_ms, b_ms = hs["metric"].get("ready_ms"), bl["metric"].get("ready_ms")
            sane = h_ms is not None and -1500 < h_ms < a.max_hs_ms
            st = OK if (sane and b_ms is not None and h_ms < b_ms) else (WARN if sane else FAIL)
            check(st, "  latency (ready after last word)", f"Head Start {h_ms} ms vs baseline {b_ms} ms"
                  + ("" if sane else f"  (> {a.max_hs_ms} ms: close CPU hogs, check network)")
                  + ("" if (b_ms is None or h_ms is None or h_ms < b_ms) else "  (Head Start not ahead)"))

    judge(a.file, "web")
    if a.local:
        judge(a.local, "local")

    width = max(len(r[1]) for r in rows)
    print(f"\nHead Start preflight  ({base})")
    for st, what, detail in rows:
        print(f"  [{st:^4}] {what.ljust(width)}  {detail}")
    fails = [r for r in rows if r[0] == FAIL]
    warns = [r for r in rows if r[0] == WARN]
    deg = h.get("degraded") or {}
    print()
    if fails:
        print("NO-GO: " + "; ".join(r[1].strip() for r in fails))
        print("       see DEMO_CHECKLIST.md > Recovery")
        return 1
    if deg or warns:
        print("GO (DEGRADED): " + ("; ".join(f"{k}: {v}" for k, v in deg.items()) or "; ".join(r[1].strip() for r in warns)))
        print("       fallbacks are labelled in the UI; say so out loud if a judge asks")
        return 3
    print("GO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
