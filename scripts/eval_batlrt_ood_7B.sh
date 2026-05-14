#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/offline_env.sh"

export SLOW_THINKING_MODEL_PATH="${SLOW_THINKING_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/DSR1-Qwen-7B-BATLRT-RFT}"
export OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/batlrt_7b_ood}"

bash scripts/eval_batlrt_ood_1.5B.sh
