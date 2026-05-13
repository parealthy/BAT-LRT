#!/bin/bash
set -euo pipefail

bash scripts/train_sft_batlrt_1.5B.sh
bash scripts/train_rft_batlrt_1.5B.sh
