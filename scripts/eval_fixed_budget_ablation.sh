#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/offline_env.sh"

SLOW_THINKING_MODEL_PATH="${SLOW_THINKING_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
REASONING_NET_PATH="${REASONING_NET_PATH:-Qwen/Qwen3-Embedding-0.6B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/DSR1-Qwen-1.5B-BATLRT-RFT}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/fixed_budget_ablation}"
LATENT_TRAJECTORY_LENGTH="${LATENT_TRAJECTORY_LENGTH:-384}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
MIN_LATENT_CHUNKS="${MIN_LATENT_CHUNKS:-2}"
FIXED_CHUNKS_LIST="${FIXED_CHUNKS_LIST:-4 8 16 24}"
DATASET_NAME="${ABLATION_DATASET:-HuggingFaceH4/MATH-500}"
DATASET_CONFIG="${ABLATION_CONFIG:-}"
SPLIT="${ABLATION_SPLIT:-test}"
PROMPT_FIELD="${ABLATION_PROMPT_FIELD:-problem}"
ANSWER_FIELD="${ABLATION_ANSWER_FIELD:-answer}"
LIMIT="${LIMIT:-10}"

mkdir -p "$OUTPUT_DIR"

for fixed_chunks in $FIXED_CHUNKS_LIST; do
    CMD=(python inference/eval_batlrt.py
        --model_path "$SLOW_THINKING_MODEL_PATH"
        --reasoning_net_path "$REASONING_NET_PATH"
        --checkpoint_path "$CHECKPOINT_PATH"
        --reasoning_net_type adaptive_anchor
        --latent_trajectory_length "$LATENT_TRAJECTORY_LENGTH"
        --chunk_size "$CHUNK_SIZE"
        --min_latent_chunks "$MIN_LATENT_CHUNKS"
        --fixed_latent_chunks "$fixed_chunks"
        --dataset_name "$DATASET_NAME"
        --split "$SPLIT"
        --prompt_field "$PROMPT_FIELD"
        --answer_field "$ANSWER_FIELD"
        --limit "$LIMIT"
        --output_jsonl "$OUTPUT_DIR/fixed_${fixed_chunks}_chunks.jsonl")
    if [ -n "$DATASET_CONFIG" ]; then
        CMD+=(--dataset_config "$DATASET_CONFIG")
    fi
    "${CMD[@]}"
done
