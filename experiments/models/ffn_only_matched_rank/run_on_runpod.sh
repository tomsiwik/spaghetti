#!/bin/bash
# Run the FFN-only matched rank experiment on RunPod.
#
# Prerequisites:
#   1. RunPod pod is running with A5000 GPU
#   2. SSH alias 'runpod' is configured in ~/.ssh/config
#   3. python -m composer.runpod_exec setup  (already done)
#
# Usage:
#   bash micro/models/ffn_only_matched_rank/run_on_runpod.sh
#
# This script:
#   1. Syncs the repo to RunPod
#   2. Trains 5 FFN-only adapters (~15min each, ~$1.00 total)
#   3. Optionally trains 5 all-modules adapters for fair comparison (~$1.00)
#   4. Pulls results back
#   5. Runs analysis locally
#
# Estimated cost: $1-2 at $0.16/hr for A5000
# Estimated time: ~75-150 min (5-10 adapters x 15 min each)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== FFN-only Matched Rank Experiment ==="
echo "Repo root: $REPO_ROOT"

# Step 1: Sync repo
echo ""
echo "Step 1: Syncing repo to RunPod..."
python -m composer.runpod_exec sync

# Seeds for multi-seed runs (reviewer recommendation #2)
SEEDS=(42 123 7)

# Step 2: Train FFN-only + all-modules adapters for each seed
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "Step 2: Training seed=$SEED — 5 FFN-only + 5 all-modules adapters..."
    echo "  Estimated time: ~150 min, cost: ~$0.40"
    python -m composer.runpod_exec run \
        "micro/models/ffn_only_matched_rank/train_ffn_only.py \
         --base /workspace/models/Qwen2.5-7B \
         --data data/distillation/ \
         --eval-data data/distillation/ \
         --output adapters_ffn_only/seed_${SEED}/ \
         --output-all adapters_all_retrained/seed_${SEED}/ \
         --also-train-all \
         --rank 16 --steps 300 --lr 2e-4 --seed ${SEED}" \
        --timeout 14400
done

# Step 3: Pull all adapters back
echo ""
echo "Step 3: Pulling adapters..."
for SEED in "${SEEDS[@]}"; do
    python -m composer.runpod_exec pull "adapters_ffn_only/seed_${SEED}/" "adapters_ffn_only/seed_${SEED}/"
    python -m composer.runpod_exec pull "adapters_all_retrained/seed_${SEED}/" "adapters_all_retrained/seed_${SEED}/"
done

# Step 4: Run analysis locally (aggregates across seeds)
echo ""
echo "Step 4: Running analysis..."
python micro/models/ffn_only_matched_rank/analyze.py \
    --ffn-adapters adapters_ffn_only/ \
    --all-adapters adapters/ \
    --all-retrained adapters_all_retrained/ \
    --retroactive-results micro/models/ffn_only_vs_all_modules/results.json \
    --output micro/models/ffn_only_matched_rank/results.json \
    --seeds 42 123 7

echo ""
echo "=== Experiment Complete ==="
echo "Results: micro/models/ffn_only_matched_rank/results.json"
