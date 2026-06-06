#!/bin/bash
# Run the complete reasoning expert distillation experiment on RunPod.
#
# Usage:
#   cd /workspace/llm
#   bash micro/models/reasoning_expert_distillation/run_all.sh 2>&1 | tee reasoning_distill.log
#
# Or as background job:
#   cd /workspace/llm
#   nohup bash micro/models/reasoning_expert_distillation/run_all.sh > reasoning_distill.log 2>&1 &
#
# Expected total runtime: ~2-3 hours on RTX 4090
# Expected cost: ~$0.70-$1.00

set -euo pipefail

SCRIPT_DIR="micro/models/reasoning_expert_distillation"
LOG_PREFIX="[reasoning-distill]"

echo "${LOG_PREFIX} Starting reasoning expert distillation experiment"
echo "${LOG_PREFIX} Time: $(date)"
echo "${LOG_PREFIX} GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# ── Phase 1: Train reasoning adapter (~30-45 min) ────────────────────────
echo "${LOG_PREFIX} === Phase 1: Training reasoning adapter ==="
python ${SCRIPT_DIR}/train_reasoning_expert.py \
    --steps 500 \
    --lr 1e-4 \
    --max-examples 10000

echo ""
echo "${LOG_PREFIX} Phase 1 complete."
echo ""

# ── Phase 2: Evaluate on MATH-500 (~60-120 min) ──────────────────────────
# Start with a small sample to verify things work, then full eval
echo "${LOG_PREFIX} === Phase 2a: Quick smoke test (50 examples) ==="
python ${SCRIPT_DIR}/eval_math500.py \
    --max-examples 50 \
    --conditions base reasoning \
    --output ${SCRIPT_DIR}/math500_smoke_test.json

echo ""
echo "${LOG_PREFIX} Smoke test complete. Checking results..."
python -c "
import json
with open('${SCRIPT_DIR}/math500_smoke_test.json') as f:
    r = json.load(f)
base = r['conditions'].get('base', {}).get('accuracy_pct', 0)
reasoning = r['conditions'].get('reasoning_only', {}).get('accuracy_pct', 0)
print(f'Smoke test: base={base}%, reasoning={reasoning}%, delta={reasoning-base:+.1f}pp')
if reasoning - base < 0:
    print('WARNING: Reasoning adapter decreasing accuracy. Continue with full eval to confirm.')
"

echo ""
echo "${LOG_PREFIX} === Phase 2b: Full MATH-500 eval (500 examples) ==="
python ${SCRIPT_DIR}/eval_math500.py \
    --max-examples 500 \
    --conditions base reasoning domain composed \
    --verbose

echo ""
echo "${LOG_PREFIX} Phase 2 complete."
echo ""

# ── Phase 3: Composition interference test (~30-60 min) ──────────────────
echo "${LOG_PREFIX} === Phase 3: Composition interference test ==="
python ${SCRIPT_DIR}/eval_composition_interference.py \
    --max-eval 50

echo ""
echo "${LOG_PREFIX} Phase 3 complete."
echo ""

# ── Summary ──────────────────────────────────────────────────────────────
echo "${LOG_PREFIX} === EXPERIMENT COMPLETE ==="
echo "${LOG_PREFIX} Time: $(date)"
echo ""
echo "${LOG_PREFIX} Results files:"
echo "  - ${SCRIPT_DIR}/reasoning_adapter/train_meta.json"
echo "  - ${SCRIPT_DIR}/math500_results.json"
echo "  - ${SCRIPT_DIR}/interference_results.json"
echo ""

# Print summary
python -c "
import json, os

base_dir = '${SCRIPT_DIR}'

# Training meta
meta_path = os.path.join(base_dir, 'reasoning_adapter', 'train_meta.json')
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    print(f'Training: loss={meta[\"train_loss\"]:.4f}, time={meta[\"train_time_s\"]/60:.1f}min, examples={meta[\"n_train_examples\"]}')
    print(f'  Cost estimate: \${meta[\"train_time_s\"]/3600 * 0.34:.2f}')

# MATH-500 results
math_path = os.path.join(base_dir, 'math500_results.json')
if os.path.exists(math_path):
    with open(math_path) as f:
        r = json.load(f)
    print(f'\nMATH-500 Accuracy:')
    for name, cond in r.get('conditions', {}).items():
        print(f'  {name}: {cond[\"accuracy_pct\"]}%')
    print(f'  Verdict: {r.get(\"verdict\", \"unknown\")}')

# Interference results
interf_path = os.path.join(base_dir, 'interference_results.json')
if os.path.exists(interf_path):
    with open(interf_path) as f:
        r = json.load(f)
    print(f'\nComposition Interference:')
    print(f'  Mean degradation: {r[\"ppl_interference\"][\"mean_degradation_pct\"]:+.2f}%')
    print(f'  Mean |cos| with domains: {r[\"orthogonality\"][\"mean_abs_cos\"]:.6f}')
    print(f'  Verdict: {r.get(\"verdict\", \"unknown\")}')
"
