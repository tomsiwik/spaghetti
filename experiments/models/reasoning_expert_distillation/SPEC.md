# Reasoning Expert Distillation — Experiment Spec

## Goal

Train a reasoning LoRA adapter from DeepSeek-R1 traces and evaluate on MATH-500.
This is P1 priority — blocks exp_reasoning_domain_composition and exp_reasoning_expert_universality.

## Scripts (already written, ready to submit)

All scripts are in `micro/models/reasoning_expert_distillation/`.

### Phase 1: Training (~30-45 min)
```
train_reasoning_expert.py --steps 500 --lr 1e-4 --max-examples 10000
```
- QLoRA rank-16 on Qwen2.5-7B, rasbt/math_distill dataset
- Saves adapter to `micro/models/reasoning_expert_distillation/reasoning_adapter/`

### Phase 2: MATH-500 Evaluation (~60-120 min)
```
eval_math500.py --max-examples 500 --conditions base reasoning domain composed --verbose
```
- Tests 4 conditions: base, reasoning-only, domain-only, composed
- Saves to `micro/models/reasoning_expert_distillation/math500_results.json`

### Phase 3: Composition Interference (~30-60 min)
```
eval_composition_interference.py --max-eval 50
```
- Measures PPL degradation and orthogonality with domain experts
- Saves to `micro/models/reasoning_expert_distillation/interference_results.json`

## Submission Strategy

Submit as 3 chained GPU queue tasks (each depends on previous output):
1. `train_reasoning_expert.py --steps 500 --lr 1e-4 --max-examples 10000`
2. `eval_math500.py --max-examples 500 --conditions base reasoning domain composed --verbose`
3. `eval_composition_interference.py --max-eval 50`

Or use `run_all.sh` as a single bash submission if gpu_queue supports bash scripts.

## Kill Criteria

- K1: reasoning LoRA does not improve MATH-500 accuracy >10pp over base
- K2: reasoning LoRA composed with domain expert degrades domain quality >5%
- K3: reasoning + domain composed model does not outperform either alone on cross-domain reasoning

## Budget

~$1.19 total (training $0.17 + MATH eval $0.68 + interference $0.34)

## Dependencies

- Requires: Qwen2.5-7B at /workspace/models/Qwen2.5-7B (cached)
- Requires: pilot50 math adapter at /workspace/llm/adapters/ (available)
- Requires: rasbt/math_distill dataset (auto-downloaded from HuggingFace)
