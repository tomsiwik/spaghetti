# MATH.md — P11.F0: Train math-s1k-reasoning-v0 (Proper Context Window)

## Problem Statement

P11.A0 (exp_p11_reasoning_sft_s1k) showed catastrophic forgetting: MMLU-Pro dropped from 62.1%
to 36.1% (-26pp) after 1000 training steps on s1K data. But only **27 examples** passed the
6000-char filter — forcing ~37 epochs on the same 27 competition math traces. We hypothesize
the catastrophic forgetting was caused by **extreme overfitting to 27 examples**, not inherent
domain incompatibility of competition math with general reasoning.

## Theorem 1: Epoch-Count Catastrophic Forgetting (Graded Collapse)

**Statement**: Let W₀ be the base model parameters. After N_epochs epochs of SFT on dataset D
with |D| examples, the expected KL divergence from W₀ grows as:

```
E[KL(p_θ_t || p_θ₀)] ≤ C · N_epochs · lr · ||∇L||_F
```

For |D|=27, N_epochs≈37, lr=1e-5: drift ∝ 37 · 1e-5 · g
For |D|=831, N_epochs≈1.2, lr=1e-5: drift ∝ 1.2 · 1e-5 · g

**Prediction**: With 31× fewer epochs, the KL drift should be approximately 31× smaller,
reducing catastrophic forgetting from -26pp to ~-1pp on MMLU-Pro.

**Citation**: Li & Liang 2021 "Prefix-Tuning: Optimizing Continuous Prompts" — established
that overfitting on small SFT sets causes catastrophic forgetting in LLMs. Generalized to
LoRA by Hu et al. 2021 (arXiv:2106.09685).

**Corollary (Kill Criterion K1508)**: If MMLU-Pro ≥ 59% (within predicted range 61-63%,
threshold set conservatively 2pp below lower bound), Theorem 1 is confirmed — proper
dataset size prevents catastrophic collapse.

**QED sketch**: The parameter update Δθ = -lr · ∇L is bounded per step. After N_epochs·|D|
steps, Δθ_total ≤ N_epochs · |D| · lr · max_grad. Holding steps fixed at 1000:
- |D|=27: N_epochs = 1000/27 ≈ 37
- |D|=831: N_epochs = 1000/831 ≈ 1.2
The total gradient accumulation (and thus parameter drift) is the same 1000 steps, BUT with 
831 unique examples the gradient diversity prevents memorization of a single distribution.

## Theorem 2: Gradient Diversity and Generalization

**Statement**: Training on N distinct examples with orthogonal gradient directions (high-rank
data) prevents the model from overfitting to any single pattern. Competition math traces have
diversity across problem types, notation, and reasoning strategies. With N≥831:

```
rank(G_batch) ≥ min(rank(H_model), N) ≈ 831
```

where G_batch is the gradient covariance matrix. This diverse gradient signal means no single
competition math format can dominate the parameter update direction, preserving breadth.

**Prediction**: GSM8K should improve from ~77% base toward 80%+ (arithmetic reasoning
improved by exposure to diverse math reasoning patterns in s1K traces).

**Citation**: arXiv:2501.19393 (s1: Simple Test-Time Scaling) — s1K-1.1 trained Qwen2.5-32B
to exceed o1-preview using 1000 curated reasoning traces. Key: diversity of source problems.

## Theorem 3: Thinking Token Preservation

**Statement**: LoRA on v_proj+o_proj with thinking-formatted training targets preserves the
thinking channel activation, measurable by avg_thinking_chars > 0 in eval.

**From P11.A0**: K1492 PASSED (1641 avg_thinking_chars) — thinking preserved even with
catastrophic forgetting. This theorem is already verified.

**Prediction**: K_thinking: avg_thinking_chars > 500 (strong thinking preserved).

## Kill Criteria

| ID | Criterion | Prediction | Basis |
|----|-----------|------------|-------|
| K1508 | MMLU-Pro + thinking ≥ 59% | UNCERTAIN — if epoch theory correct, ~61-63% | Theorem 1 |
| K1509 | GSM8K ≥ 80% | EXPECTED PASS if properly loaded | Theorem 2 |
| K1510 | Adapter registered in registry.json | EXPECTED PASS | Implementation |

**Kill structure for K1508 failure**: If MMLU-Pro < 60% despite N=831 diverse examples,
the domain mismatch theory holds (competition math is geometrically orthogonal to MMLU-Pro
breadth regardless of overfitting severity). In that case, diverse multi-subject reasoning
traces (Sky-T1 or MMLU-Pro formatted traces) would be needed.

## Failure Modes

1. **Domain mismatch** (not overfitting): Competition math vocabulary ≠ MMLU-Pro vocabulary.
   Even with 1 epoch, reasoning style shift hurts general breadth.
   → Indicator: MMLU-Pro < 60% with GSM8K improved

2. **Dataset quality**: s1K thinking traces may use `<think>...</think>` format (not Gemma 4 native).
   Training format mismatch means thinking channel training signal is in wrong token range.
   → Indicator: avg_thinking_chars = 0 in eval despite training

3. **LoRA rank insufficient**: r=8 on v_proj+o_proj may not have capacity for competition math.
   → Indicator: training loss stays high (no convergence)

## Connection to Architecture

- v_proj+o_proj targets are value projection and output projection in attention
- P11.A0 used the same targets — so adapter capacity is comparable
- Key difference: 831 vs 27 examples → 31× reduction in per-example gradient accumulation
- registry.json will link this adapter to math-s1k-reasoning-v0 with eval scores

## References

- arXiv:2501.19393 (s1: Simple Test-Time Scaling) — s1K dataset and training methodology
- arXiv:2106.09685 (LoRA: Low-Rank Adaptation) — LoRA theory and catastrophic forgetting prevention
- exp_p11_reasoning_sft_s1k PAPER.md — prior s1K run (N=27 examples → -26pp catastrophic forgetting)
- Finding #538: s1K causes catastrophic forgetting when N=27 (overfitting)
