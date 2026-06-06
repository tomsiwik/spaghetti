#!/bin/bash
set -euo pipefail
# ── Pilot 50: Full Pipeline Orchestration ────────────────────────────
#
# Runs the complete 50-expert distillation pipeline:
#   Step 1: Generate training data via Groq API (LOCAL, ~2-4 hours)
#   Step 2: Sync data to RunPod
#   Step 3: Train 50 QLoRA experts on RunPod (REMOTE, ~12 hours)
#   Step 4: Benchmark experts vs base (REMOTE, ~2 hours)
#   Step 5: Pull results back locally
#
# Prerequisites:
#   - GROQ_API_KEY in .env
#   - SSH alias 'runpod' configured
#   - RunPod pod with 4090 GPU running
#
# Usage:
#   ./scripts/pilot50_orchestrate.sh          # full pipeline
#   ./scripts/pilot50_orchestrate.sh generate # only data generation
#   ./scripts/pilot50_orchestrate.sh train    # only training (after sync)
#   ./scripts/pilot50_orchestrate.sh bench    # only benchmark
#   ./scripts/pilot50_orchestrate.sh sync     # only sync to RunPod

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

REMOTE_DIR="/workspace/llm"
SSH_ALIAS="runpod"

# ── Helpers ───────────────────────────────────────────────────────────
log() { echo "$(date '+%H:%M:%S') [pilot50] $*"; }
die() { log "FATAL: $*"; exit 1; }

check_ssh() {
    ssh -o ConnectTimeout=5 "$SSH_ALIAS" "echo ok" >/dev/null 2>&1 \
        || die "Cannot reach RunPod. Check SSH config."
}

# ── Step 1: Generate Data (LOCAL) ────────────────────────────────────
do_generate() {
    log "Step 1: Generating training data via Groq API..."
    python scripts/pilot50_generate.py --n-examples 1000
    log "Step 1 complete."
}

# ── Step 2: Sync to RunPod ───────────────────────────────────────────
do_sync() {
    log "Step 2: Syncing to RunPod..."
    check_ssh

    # Create remote dirs
    ssh "$SSH_ALIAS" "mkdir -p $REMOTE_DIR/data/distillation $REMOTE_DIR/adapters $REMOTE_DIR/results $REMOTE_DIR/scripts $REMOTE_DIR/composer"

    # Sync data, scripts, and composer code
    rsync -rlptz --progress \
        "$REPO_ROOT/data/distillation/" \
        "$SSH_ALIAS:$REMOTE_DIR/data/distillation/"

    rsync -rlptz --progress \
        "$REPO_ROOT/scripts/pilot50_train.py" \
        "$REPO_ROOT/scripts/pilot50_bench.py" \
        "$SSH_ALIAS:$REMOTE_DIR/scripts/"

    rsync -rlptz --progress \
        "$REPO_ROOT/composer/" \
        "$SSH_ALIAS:$REMOTE_DIR/composer/"

    # Sync existing adapters (resume support)
    rsync -rlptz --progress \
        "$REPO_ROOT/adapters/" \
        "$SSH_ALIAS:$REMOTE_DIR/adapters/"

    # Sync pyproject.toml and .env
    rsync -rlptz --progress \
        "$REPO_ROOT/pyproject.toml" \
        "$REPO_ROOT/.env" \
        "$SSH_ALIAS:$REMOTE_DIR/"

    log "Step 2 complete."
}

# ── Step 3: Train on RunPod ──────────────────────────────────────────
do_train() {
    log "Step 3: Training 50 experts on RunPod..."
    check_ssh

    # Install dependencies if needed
    ssh "$SSH_ALIAS" "cd $REMOTE_DIR && pip install -q transformers peft trl datasets accelerate bitsandbytes scipy pyyaml 2>&1 | tail -3"

    # Run training
    ssh "$SSH_ALIAS" "cd $REMOTE_DIR && export HF_HOME=/workspace/hf_cache && python scripts/pilot50_train.py --rank 16 --steps 300"

    log "Step 3 complete."
}

# ── Step 4: Benchmark on RunPod ──────────────────────────────────────
do_bench() {
    log "Step 4: Benchmarking experts vs base..."
    check_ssh

    ssh "$SSH_ALIAS" "cd $REMOTE_DIR && export HF_HOME=/workspace/hf_cache && python scripts/pilot50_bench.py --max-eval 100"

    log "Step 4 complete."
}

# ── Step 5: Pull Results ─────────────────────────────────────────────
do_pull() {
    log "Step 5: Pulling results and adapters..."
    check_ssh

    # Pull benchmark results
    mkdir -p "$REPO_ROOT/results"
    scp -r "$SSH_ALIAS:$REMOTE_DIR/results/pilot50_benchmark.json" \
        "$REPO_ROOT/results/" 2>/dev/null || log "No benchmark results yet"

    # Pull adapter metadata (small files)
    for d in $(ssh "$SSH_ALIAS" "ls $REMOTE_DIR/adapters/"); do
        mkdir -p "$REPO_ROOT/adapters/$d"
        scp "$SSH_ALIAS:$REMOTE_DIR/adapters/$d/train_meta.json" \
            "$REPO_ROOT/adapters/$d/" 2>/dev/null || true
        scp "$SSH_ALIAS:$REMOTE_DIR/adapters/$d/adapter_config.json" \
            "$REPO_ROOT/adapters/$d/" 2>/dev/null || true
    done

    log "Step 5 complete."
}

# ── Main ─────────────────────────────────────────────────────────────
STEP="${1:-all}"

case "$STEP" in
    generate) do_generate ;;
    sync)     do_sync ;;
    train)    do_train ;;
    bench)    do_bench ;;
    pull)     do_pull ;;
    all)
        do_generate
        do_sync
        do_train
        do_bench
        do_pull
        log "Full pipeline complete!"
        ;;
    *)
        echo "Usage: $0 {generate|sync|train|bench|pull|all}"
        exit 1
        ;;
esac
