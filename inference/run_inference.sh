#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/scripts/offline_env.sh"

# ============================================================
# Latent Reasoning — Interactive Inference
#
# Usage:
#   bash inference/run_inference.sh
#
#   # Override model config (e.g. 7B)
#   SLOW_THINKING_MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#   CHECKPOINT_PATH=checkpoints/DSR1-Qwen-7B-LRT-Math \
#     bash inference/run_inference.sh
# ============================================================

# ---- Model ----
SLOW_THINKING_MODEL_PATH="${SLOW_THINKING_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
REASONING_NET_PATH="${REASONING_NET_PATH:-Qwen/Qwen3-Embedding-0.6B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/DSR1-Qwen-1.5B-LRT-Math}"
LATENT_TRAJECTORY_LENGTH="${LATENT_TRAJECTORY_LENGTH:-256}"
REASONING_NET_TYPE="${REASONING_NET_TYPE:-fixed}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
MIN_LATENT_CHUNKS="${MIN_LATENT_CHUNKS:-2}"
ROUTER_TAU="${ROUTER_TAU:-1.0}"
FIXED_LATENT_CHUNKS="${FIXED_LATENT_CHUNKS:-}"
PRINT_LATENT_BUDGET="${PRINT_LATENT_BUDGET:-false}"

# ---- Generation ----
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10000}"
PROMPT_MAX_LENGTH="${PROMPT_MAX_LENGTH:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"

echo "================================================"
echo "  Latent Reasoning Interactive Inference"
echo "  Model:        $SLOW_THINKING_MODEL_PATH"
echo "  ReasoningNet: $REASONING_NET_PATH"
echo "  Checkpoint:   $CHECKPOINT_PATH"
echo "  Type:         $REASONING_NET_TYPE"
echo "================================================"

CMD=(python inference/run_inference.py
    --model_path "$SLOW_THINKING_MODEL_PATH" \
    --reasoning_net_path "$REASONING_NET_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --reasoning_net_type "$REASONING_NET_TYPE" \
    --latent_trajectory_length "$LATENT_TRAJECTORY_LENGTH" \
    --chunk_size "$CHUNK_SIZE" \
    --min_latent_chunks "$MIN_LATENT_CHUNKS" \
    --router_tau "$ROUTER_TAU" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --prompt_max_length "$PROMPT_MAX_LENGTH" \
    --temperature "$TEMPERATURE")

if [ -n "$FIXED_LATENT_CHUNKS" ]; then
    CMD+=(--fixed_latent_chunks "$FIXED_LATENT_CHUNKS")
fi

if [ "$PRINT_LATENT_BUDGET" = "true" ]; then
    CMD+=(--print_latent_budget)
fi

"${CMD[@]}"
