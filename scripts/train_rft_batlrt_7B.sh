#!/bin/bash
set -euo pipefail

export SLOW_THINKING_MODEL_PATH="${SLOW_THINKING_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-checkpoints/DSR1-Qwen-7B-BATLRT-SFT}"
export OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/DSR1-Qwen-7B-BATLRT-RFT}"
export RUN_NAME="${RUN_NAME:-batlrt-rft-7b}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export MASTER_PORT="${MASTER_PORT:-23463}"

bash scripts/train_rft_batlrt_1.5B.sh
