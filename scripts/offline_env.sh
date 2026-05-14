#!/bin/bash

export LRT_MODEL_ROOT="${LRT_MODEL_ROOT:-/mnt/pami23/dzhu/models}"
export LRT_DATA_ROOT="${LRT_DATA_ROOT:-/mnt/pami23/dzhu/datasets}"
export LRT_OFFLINE="${LRT_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

if [ -n "${MODEL_CACHE:-}" ]; then
    export HF_HOME="$MODEL_CACHE"
fi
