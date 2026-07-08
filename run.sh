#!/usr/bin/env bash
# Run the Section 7 pipeline.
#   ./run.sh smoke   tiny end-to-end sanity pass (separate *_smoke cache)
#   ./run.sh full    the real sweep, then build the figure
# Steps after extraction are independent; they run in order here for simplicity.
set -e
cd "$(dirname "$0")"

MODE="${1:-smoke}"
FLAG=""
[ "$MODE" = "smoke" ] && FLAG="--smoke"

# No baseline step: the figure validates Theorem 2 (rate + floor), not a
# supervised comparison. baseline.py is kept in the repo but not run here.
echo "== extract_features ($MODE) =="; python extract_features.py $FLAG
echo "== probe_sweep ($MODE) ==";      python probe_sweep.py $FLAG
echo "== measure_rda ($MODE) ==";      python measure_rda.py $FLAG

if [ "$MODE" = "full" ]; then
  echo "== make_figure =="; python make_figure.py
else
  echo "(smoke mode writes nothing to the real cache and skips make_figure)"
fi
