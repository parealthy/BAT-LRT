#!/bin/bash
set -euo pipefail

export SLOW_THINKING_MODEL_PATH="${SLOW_THINKING_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/DSR1-Qwen-7B-BATLRT-RFT}"
export OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/batlrt_7b_math}"

bash scripts/eval_batlrt_math_1.5B.sh
