#!/bin/bash
set -euo pipefail

bash scripts/eval_batlrt_math_1.5B.sh
bash scripts/eval_batlrt_ood_1.5B.sh

if [ "${RUN_7B_EVALS:-false}" = "true" ]; then
    bash scripts/eval_batlrt_math_7B.sh
    bash scripts/eval_batlrt_ood_7B.sh
fi
