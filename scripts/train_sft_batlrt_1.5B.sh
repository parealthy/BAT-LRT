#!/bin/bash
set -euo pipefail

if [ -n "${MODEL_CACHE:-}" ]; then
    export HF_HOME="$MODEL_CACHE"
fi

SLOW_THINKING_MODEL_PATH="${SLOW_THINKING_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
REASONING_NET_PATH="${REASONING_NET_PATH:-Qwen/Qwen3-Embedding-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/DSR1-Qwen-1.5B-BATLRT-SFT}"
DATASET_NAME="${DATASET_NAME:-open-r1/OpenR1-Math-220k}"
RUN_NAME="${RUN_NAME:-batlrt-sft-1.5b}"

LATENT_TRAJECTORY_LENGTH="${LATENT_TRAJECTORY_LENGTH:-384}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
MIN_LATENT_CHUNKS="${MIN_LATENT_CHUNKS:-2}"
ROUTER_TAU="${ROUTER_TAU:-1.0}"
ANCHOR_LOSS_WEIGHT="${ANCHOR_LOSS_WEIGHT:-0.1}"
BUDGET_LOSS_WEIGHT="${BUDGET_LOSS_WEIGHT:-0.02}"
BUDGET_WARMUP_RATIO="${BUDGET_WARMUP_RATIO:-0.1}"
TEACHER_SELECTION="${TEACHER_SELECTION:-shortest}"
TEACHER_MAX_STEPS="${TEACHER_MAX_STEPS:-24}"
TEACHER_STEP_MAX_LENGTH="${TEACHER_STEP_MAX_LENGTH:-128}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2.yaml}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
PROMPT_MAX_LENGTH="${PROMPT_MAX_LENGTH:-1024}"
COMPLETION_MAX_LENGTH="${COMPLETION_MAX_LENGTH:-512}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
BF16="${BF16:-true}"
TF32="${TF32:-false}"

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23460}"
TOTAL_PROCESSES=$((NUM_GPUS_PER_NODE * NUM_NODES))

if [ "$NODE_RANK" -eq 0 ]; then
    mkdir -p "$OUTPUT_DIR"
fi

CMD=(accelerate launch
    --config_file "$DEEPSPEED_CONFIG"
    --num_processes "$TOTAL_PROCESSES"
    --num_machines "$NUM_NODES"
    --machine_rank "$NODE_RANK"
    --main_process_ip "$MASTER_ADDR"
    --main_process_port "$MASTER_PORT"
    sft.py
    --slow_thinking_model_path "$SLOW_THINKING_MODEL_PATH"
    --reasoning_net_path "$REASONING_NET_PATH"
    --reasoning_net_type adaptive_anchor
    --latent_trajectory_length "$LATENT_TRAJECTORY_LENGTH"
    --chunk_size "$CHUNK_SIZE"
    --min_latent_chunks "$MIN_LATENT_CHUNKS"
    --router_tau "$ROUTER_TAU"
    --anchor_loss_weight "$ANCHOR_LOSS_WEIGHT"
    --budget_loss_weight "$BUDGET_LOSS_WEIGHT"
    --budget_warmup_ratio "$BUDGET_WARMUP_RATIO"
    --teacher_selection "$TEACHER_SELECTION"
    --teacher_max_steps "$TEACHER_MAX_STEPS"
    --teacher_step_max_length "$TEACHER_STEP_MAX_LENGTH"
    --dataset_name "$DATASET_NAME"
    --prompt_max_length "$PROMPT_MAX_LENGTH"
    --completion_max_length "$COMPLETION_MAX_LENGTH"
    --output_dir "$OUTPUT_DIR"
    --run_name "$RUN_NAME"
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --num_train_epochs "$NUM_EPOCHS"
    --learning_rate "$LEARNING_RATE"
    --logging_steps "$LOGGING_STEPS"
    --save_steps "$SAVE_STEPS"
    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --bf16 "$BF16"
    --tf32 "$TF32")

if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
    CMD+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
if [ -n "${MAX_STEPS:-}" ]; then
    CMD+=(--max_steps "$MAX_STEPS")
fi

"${CMD[@]}"
