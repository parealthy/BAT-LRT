#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/offline_env.sh"

bash scripts/eval_batlrt_math_1.5B.sh
bash scripts/eval_batlrt_ood_1.5B.sh

if [ "${RUN_7B_EVALS:-false}" = "true" ]; then
    bash scripts/eval_batlrt_math_7B.sh
    bash scripts/eval_batlrt_ood_7B.sh
fi
