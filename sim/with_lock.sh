#!/bin/bash
# Run a command under the shared CPU-benchmark lock (main checkout's runs/.run_ab.lock).
# Waits (poll 60 s, up to 20 min) while another agent holds it; a lock whose pid is dead is treated as stale.
#   sim/with_lock.sh <owner> <cmd...>
OWNER="$1"; shift
LOCK=/Users/karan/Desktop/work/whissle/headstart/runs/.run_ab.lock
for i in $(seq 0 20); do
  if [ -e "$LOCK" ]; then
    PID=$(grep -o 'pid=[0-9]*' "$LOCK" | head -1 | cut -d= -f2)
    if [ -n "$PID" ] && ! kill -0 "$PID" 2>/dev/null; then
      echo "$(date +%T) stale lock (pid $PID dead): $(cat "$LOCK")"; rm -f "$LOCK"
    fi
  fi
  if ( set -o noclobber; echo "$OWNER pid=$$ since=$(date +%T) args=$*" > "$LOCK" ) 2>/dev/null; then
    echo "$(date +%T) lock acquired"
    trap 'rm -f "$LOCK"' EXIT
    "$@"; rc=$?
    echo "$(date +%T) done rc=$rc"
    exit $rc
  fi
  [ "$i" -eq 20 ] && { echo "$(date +%T) gave up after 20 min: $(cat "$LOCK")"; exit 9; }
  echo "$(date +%T) waiting: $(cat "$LOCK")"
  sleep 60
done
