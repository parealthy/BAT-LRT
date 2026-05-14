#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/offline_env.sh"

if [ -n "${MODEL_CACHE:-}" ]; then
    export HF_HOME="$MODEL_CACHE"
fi

SLOW_THINKING_MODEL_PATH="${SLOW_THINKING_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
REASONING_NET_PATH="${REASONING_NET_PATH:-Qwen/Qwen3-Embedding-0.6B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/DSR1-Qwen-1.5B-BATLRT-RFT}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/batlrt_1.5b_ood}"
LATENT_TRAJECTORY_LENGTH="${LATENT_TRAJECTORY_LENGTH:-384}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
MIN_LATENT_CHUNKS="${MIN_LATENT_CHUNKS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
PROMPT_MAX_LENGTH="${PROMPT_MAX_LENGTH:-1024}"
LIMIT="${LIMIT:-}"

mkdir -p "$OUTPUT_DIR"

run_eval() {
    local name="$1"
    local dataset="$2"
    local config="$3"
    local split="$4"
    local prompt_field="$5"
    local answer_field="$6"
    if [ -z "$dataset" ]; then
        echo "Skipping $name because dataset env is empty."
        return 0
    fi

    CMD=(python inference/eval_batlrt.py
        --model_path "$SLOW_THINKING_MODEL_PATH"
        --reasoning_net_path "$REASONING_NET_PATH"
        --checkpoint_path "$CHECKPOINT_PATH"
        --reasoning_net_type adaptive_anchor
        --latent_trajectory_length "$LATENT_TRAJECTORY_LENGTH"
        --chunk_size "$CHUNK_SIZE"
        --min_latent_chunks "$MIN_LATENT_CHUNKS"
        --dataset_name "$dataset"
        --split "$split"
        --prompt_field "$prompt_field"
        --answer_field "$answer_field"
        --max_new_tokens "$MAX_NEW_TOKENS"
        --prompt_max_length "$PROMPT_MAX_LENGTH"
        --output_jsonl "$OUTPUT_DIR/${name}.jsonl")

    if [ -n "$config" ]; then
        CMD+=(--dataset_config "$config")
    fi
    if [ -n "$LIMIT" ]; then
        CMD+=(--limit "$LIMIT")
    fi

    "${CMD[@]}"
}

run_eval gpqa "${GPQA_DATASET:-}" "${GPQA_CONFIG:-}" "${GPQA_SPLIT:-test}" "${GPQA_PROMPT_FIELD:-question}" "${GPQA_ANSWER_FIELD:-answer}"
run_eval lsat "${LSAT_DATASET:-}" "${LSAT_CONFIG:-}" "${LSAT_SPLIT:-test}" "${LSAT_PROMPT_FIELD:-question}" "${LSAT_ANSWER_FIELD:-answer}"
run_eval aime "${AIME_DATASET:-}" "${AIME_CONFIG:-}" "${AIME_SPLIT:-test}" "${AIME_PROMPT_FIELD:-problem}" "${AIME_ANSWER_FIELD:-answer}"
