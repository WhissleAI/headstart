#!/usr/bin/env bash
# Head Start: one command to a demo-ready browser.
#   prerequisites -> free the port -> start server -> wait warm -> preflight (1 real Race replay) -> GO/NO-GO -> open
#
#   ./run_demo.sh                  # port 8770, live web lane (falls back to the recorded cache if it fails)
#   ./run_demo.sh --offline        # WEB_BACKEND=cache: web lane replays runs/web_cache.json (flagged in the UI)
#   ./run_demo.sh --port 8771 --no-open --full   # --full also replays a local-moments question
#   ./run_demo.sh --no-preflight   # just start + wait warm
# Server log: runs/server_<port>.log. Stop: kill $(lsof -ti tcp:<port> -sTCP:LISTEN)
set -u
cd "$(dirname "$0")"

PORT=8770; OPEN=1; PREFLIGHT=1; FULL=0; FORCE_OPEN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift ;;
    --offline) export WEB_BACKEND=cache ;;
    --no-open) OPEN=0 ;;
    --no-preflight) PREFLIGHT=0 ;;
    --full) FULL=1 ;;
    --force-open) FORCE_OPEN=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done

PY=.venv/bin/python
URL="http://localhost:$PORT"
LOG="runs/server_$PORT.log"
mkdir -p runs
bad=0
ok()   { printf '  [ OK ] %s\n' "$1"; }
warn() { printf '  [WARN] %s\n' "$1"; }
fail() { printf '  [FAIL] %s\n' "$1"; bad=1; }

echo "Head Start demo launcher (port $PORT)"
echo "Prerequisites"
if [ -x "$PY" ]; then ok "venv ($PY)"; else fail "no venv at $PY (python -m venv .venv && .venv/bin/pip install -r ...)"; fi
for f in models/arctag/arctag_int8.onnx models/arctag/tokenizer.json vendor/arctag/libarctag.dylib \
         index/data/docs.jsonl eval/audio/q01.wav eval/audio/local_05.wav web/index.html; do
  if [ -e "$f" ]; then ok "$f"; else fail "missing $f"; fi
done
if [ -x "$PY" ]; then
  # model locations from server/config.py (env and .env respected); key presence only, never values
  "$PY" - <<'PYEOF' || bad=1
import os, sys
from server import config
bad = False
def line(st, msg): print(f"  [{st:^4}] {msg}")
arc = config.ARC_DIR / "model.safetensors"
if arc.exists(): line("OK", f"Arc ear weights ({config.ARC_DIR})")
else: line("FAIL", f"Arc ear weights missing: {arc}"); bad = True
try:
    from huggingface_hub import try_to_load_from_cache
    hit = try_to_load_from_cache(config.EMBED_MODEL, "config.json")
    if isinstance(hit, str): line("OK", f"EmbeddingGemma 2 in HF cache ({config.EMBED_MODEL})")
    else: line("FAIL", f"{config.EMBED_MODEL} not in the HF cache (needs a download: no Wi-Fi at the venue?)"); bad = True
except Exception as e:
    line("WARN", f"could not check the HF cache: {e}")
if os.environ.get("ANTHROPIC_API_KEY"): line("OK", "ANTHROPIC_API_KEY set")
else: line("WARN", "ANTHROPIC_API_KEY missing: answers will be extractive (flagged)")
line("OK" if config.TAVILY_API_KEY else "INFO", "TAVILY_API_KEY " + ("set" if config.TAVILY_API_KEY else "not set (web lane = Wikipedia)"))
if config.EAR == "deepgram":
    line("OK" if config.DEEPGRAM_API_KEY else "WARN", "DEEPGRAM_API_KEY " + ("set" if config.DEEPGRAM_API_KEY else "missing: ear falls back to Arc"))
c = config.WEB_CACHE_PATH
try:
    import json; n = len(json.loads(c.read_text()).get("entries", {}))
    line("OK", f"web replay cache: {n} entries ({c})")
except Exception:
    line("FAIL" if config.WEB_BACKEND == "cache" else "WARN", f"no web replay cache at {c} (record: {sys.executable} eval/record_web_cache.py)")
    bad = bad or config.WEB_BACKEND == "cache"
line("INFO", f"web backend: {config.WEB_BACKEND}")
sys.exit(1 if bad else 0)
PYEOF
fi
if [ -e runs/.run_ab.lock ] && [ "${IGNORE_LOCK:-0}" != 1 ]; then
  fail "runs/.run_ab.lock exists: a benchmark is using the CPU (latency would be off). Wait, or IGNORE_LOCK=1"
fi
if [ "$bad" = 1 ]; then echo; echo "NO-GO: fix the prerequisites above"; exit 1; fi

echo "Server"
pids=$(lsof -ti "tcp:$PORT" -sTCP:LISTEN 2>/dev/null || true)
if [ -n "$pids" ]; then
  echo "  stopping stale server on :$PORT (pid $pids)"
  kill $pids 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do sleep 0.5; lsof -ti "tcp:$PORT" -sTCP:LISTEN >/dev/null 2>&1 || break; done
  pids=$(lsof -ti "tcp:$PORT" -sTCP:LISTEN 2>/dev/null || true)
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
fi
nohup "$PY" -m uvicorn server.app:app --port "$PORT" >"$LOG" 2>&1 &
SPID=$!
echo "  started pid $SPID (log $LOG); warming models (~30-60 s)"
t0=$(date +%s)
while :; do
  if ! kill -0 "$SPID" 2>/dev/null; then echo "  [FAIL] server exited during warm-up; last log lines:"; tail -15 "$LOG"; echo; echo "NO-GO"; exit 1; fi
  if curl -fsS "$URL/api/health" 2>/dev/null | grep -q '"warm":true'; then break; fi
  if [ $(( $(date +%s) - t0 )) -gt 240 ]; then echo "  [FAIL] not warm after 240 s (log $LOG)"; echo; echo "NO-GO"; exit 1; fi
  sleep 2
done
echo "  warm after $(( $(date +%s) - t0 )) s"

rc=0
if [ "$PREFLIGHT" = 1 ]; then
  args=(--port "$PORT")
  [ "$FULL" = 1 ] && args+=(--local local_05.wav)
  "$PY" scripts/preflight.py "${args[@]}"; rc=$?
fi

if [ "$OPEN" = 1 ]; then
  if [ "$rc" = 1 ] && [ "$FORCE_OPEN" = 0 ]; then
    echo "Not opening the browser on NO-GO (use --force-open). Server left running on $URL."
  else
    open -a "Google Chrome" "$URL" 2>/dev/null || open "$URL"
    echo "Opened $URL (Chrome, 1280x720, mode Race). Server pid $SPID."
  fi
fi
exit $rc
