#!/usr/bin/env bash
# Analysis C — vast.ai launch script.
#
# Runs the full pipeline on a fresh CPU instance:
#   1. install deps
#   2. selection-bias table
#   3. pre-fit EDA + tripwires (FAIL FAST if tripwire fires)
#   4. main 36-cell nested-CV driver
#   5. tar outputs for download
#
# Expected runtime: 2-5 hr on 16-core CPU instance.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== [$(date +%H:%M:%S)] installing deps ==="
pip install --quiet --upgrade pip
pip install --quiet numpy scipy pandas scikit-learn matplotlib

echo ""
echo "=== [$(date +%H:%M:%S)] selection-bias table ==="
python -m longitudinal.c_selection_bias

echo ""
echo "=== [$(date +%H:%M:%S)] pre-fit EDA + tripwires ==="
python -m longitudinal.c_postmortem_eda

# Fail if any tripwires fired (exit code != 0 in driver).
python -c "
import json, sys
with open('runs/longitudinal/c_postmortem/prefit_diagnostics.json') as f:
    d = json.load(f)
if d['tripwires']['fired']:
    print('[ABORT] tripwires fired:', d['tripwires']['fired'])
    sys.exit(1)
print('[ok] tripwires clean')
"

echo ""
echo "=== [$(date +%H:%M:%S)] main driver (this is the long one) ==="
time python -m longitudinal.c_postmortem

echo ""
echo "=== [$(date +%H:%M:%S)] packing outputs ==="
RESULTS_TAR="c_postmortem_results_$(date +%Y%m%d_%H%M).tar.gz"
tar -czf "$RESULTS_TAR" runs/longitudinal/c_postmortem/
ls -la "$RESULTS_TAR"

echo ""
echo "=== DONE. Download $RESULTS_TAR via:  scp remote:~/rush_pipeline/$RESULTS_TAR ."
