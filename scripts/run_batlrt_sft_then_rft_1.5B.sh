#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/offline_env.sh"

bash scripts/train_sft_batlrt_1.5B.sh
bash scripts/train_rft_batlrt_1.5B.sh
