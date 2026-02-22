#!/usr/bin/env bash
set -euo pipefail

# Example:
#   bash ./run_ablation_swat.sh --device cuda --epoch 30 --c-score-lambda 0.8

python tools/run_ablation.py --dataset swat "$@"
