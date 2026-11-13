#!/bin/bash
# Restart a command while KEEPING the shared lock: the lock is rewritten to this process before the old holder is
# SIGKILLed (so the old holder's exit trap cannot delete it).
# Usage: KILL_PIDS="<descendants>" sim/takeover.sh <old_holder_pid> <cmd...>
LOCK=/Users/karan/Desktop/work/whissle/headstart/runs/.run_ab.lock
OLD="$1"; shift
grep -q "^policy-sim pid=$OLD " "$LOCK" || { echo "lock is not held by policy-sim pid $OLD: $(cat "$LOCK")"; exit 9; }
echo "policy-sim pid=$$ since=$(date +%T) args=$* (restart)" > "$LOCK"
kill -9 "$OLD" ${KILL_PIDS:-}      # old holder first (no exit trap on SIGKILL), then its workers
trap 'rm -f "$LOCK"' EXIT
echo "$(date +%T) took over lock from $OLD"
"$@"; rc=$?
echo "$(date +%T) done rc=$rc"
