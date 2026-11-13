#!/bin/bash
# Full sweep pipeline (run under sim/with_lock.sh): prefetch -> validate -> stage-1 random search -> stage-2 refine
# on fresh seeds, for both web profiles -> report.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # tiny matmuls: threads only oversubscribe
W=${WORKERS:-6}; N=${N:-2500}   # N=0: full grid (5184)
$PY -m sim.simulate --prefetch --seeds 3 2>&1 | grep -E "prefetch"
$PY -m sim.simulate --validate --seeds 10 2>&1 | grep -vi warn | grep -v "Loading weights"
for P in wikipedia tavily; do
  $PY -m sim.sweep --profile $P --n $N --seeds 2 --workers $W
  $PY -m sim.simulate --fetch-missing
  $PY -m sim.sweep --refine runs/sweep_$P.json --top 40 --seeds 12 --seed0 100 --workers $W
done
$PY -m sim.report runs/sweep_wikipedia_refine.json runs/sweep_tavily_refine.json > runs/sweep_report.md
$PY -m sim.report runs/sweep_wikipedia.json runs/sweep_tavily.json > runs/sweep_report_stage1.md
echo pipeline ok
