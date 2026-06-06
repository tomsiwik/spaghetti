#!/bin/bash
# Run held-out evaluation for pilot 50 experts
# Phase 1: MMLU eval (23 adapters with mappings)
# Phase 2: HumanEval eval (python adapter only for kill criterion check)
#
# Estimated runtime: ~60-90 min total (GPU inference only)
# Budget: ~$0.10 on RunPod A5000

set -e

WORKDIR="/workspace/llm"
RESULTS_DIR="$WORKDIR/results/held_out_eval"
SCRIPT_DIR="$WORKDIR/macro/pilot50_held_out_eval"

mkdir -p "$RESULTS_DIR"

echo "=== Pilot 50 Held-Out Evaluation ==="
echo "Started: $(date -u)"
echo ""

# Phase 1: MMLU evaluation
echo "=== Phase 1: MMLU Held-Out Evaluation ==="
echo "Evaluating 23 adapters on mapped MMLU test subsets..."
python3 "$SCRIPT_DIR/eval_mmlu.py" \
    --all \
    --out "$RESULTS_DIR/mmlu_results.json" \
    2>&1 | tee "$RESULTS_DIR/mmlu_eval.log"

echo ""
echo "Phase 1 complete: $(date -u)"
echo ""

# Phase 2: HumanEval (python adapter only - most important for kill criterion)
echo "=== Phase 2: HumanEval Evaluation ==="
echo "Evaluating python adapter on HumanEval..."
python3 "$SCRIPT_DIR/eval_humaneval.py" \
    --adapter python \
    --out "$RESULTS_DIR/humaneval_results.json" \
    2>&1 | tee "$RESULTS_DIR/humaneval_eval.log"

echo ""
echo "Phase 2 complete: $(date -u)"
echo ""

# Summary
echo "=== SUMMARY ==="
python3 -c "
import json
print('--- MMLU Results ---')
try:
    with open('$RESULTS_DIR/mmlu_results.json') as f:
        d = json.load(f)
    agg = d.get('aggregate', {})
    print(f'Adapters evaluated: {agg.get(\"adapter_count\", \"?\")}')
    print(f'Adapters with positive delta: {agg.get(\"adapters_with_positive_delta\", \"?\")}')
    print(f'Adapter win rate: {agg.get(\"adapter_win_rate_pct\", \"?\"):.1f}%')
    print(f'Average delta: {agg.get(\"avg_delta_pct\", \"?\"):+.2f}pp')
    kc = agg.get('kill_criteria', {})
    print(f'Kill: win<80% = {kc.get(\"win_rate_below_80\", \"?\")}')
    print(f'Kill: avg<2% = {kc.get(\"avg_improvement_below_2\", \"?\")}')
except Exception as e:
    print(f'Error: {e}')

print()
print('--- HumanEval Results ---')
try:
    with open('$RESULTS_DIR/humaneval_results.json') as f:
        d = json.load(f)
    base = d.get('base_results', {})
    print(f'Base pass@1: {base.get(\"pass_at_1\", \"?\"):.3f}')
    for name, comp in d.get('comparisons', {}).items():
        print(f'{name}: pass@1={comp[\"adapter_pass_at_1\"]:.3f} (delta={comp[\"delta_pct\"]:+.1f}pp)')
    agg = d.get('aggregate', {})
    kc = agg.get('kill_criteria', {})
    print(f'Kill: python<base = {kc.get(\"python_below_base\", \"?\")}')
except Exception as e:
    print(f'Error: {e}')
"

echo ""
echo "Completed: $(date -u)"
