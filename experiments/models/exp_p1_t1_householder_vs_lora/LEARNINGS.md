# LEARNINGS.md — T1.2: HRA vs LoRA (exp_p1_t1_householder_vs_lora)

**Finding #416 — KILLED**

## Core Finding

HRA at equal rank (not equal params) loses to LoRA on MMLU (-6pp) and fails to converge
in 300 steps, but both failures are structural/optimizer issues, not HRA being fundamentally
inferior. P1 is unblocked via LoRA + Grassmannian fallback.

## Why

Three structural barriers (arxiv 2405.17484, HRA paper):
1. **Wrong optimizer**: Standard Euclidean Adam leaves the Stiefel manifold; Riemannian Adam
   (Cayley retraction, T1.4) is required for correct convergence.
2. **Equal-rank ≠ equal-params**: HRA r=16 uses only 38.5% of LoRA r=16 params. The HRA paper
   compares at equal params (HRA r=16 vs LoRA r≈6). Our test was the wrong comparison class.
3. **Multiplicative rotation disturbs base directions**: H^(r)x rotates queries away from
   MMLU world-knowledge directions; LoRA's additive x+BAx preserves them.

## Implications for Next Experiment

T1.6 algorithm bake-off must compare: **equal params** + **Riemannian Adam** for Cayley/Householder.
Specifically: HRA r=16 (~40k params/layer) vs LoRA r≈6 (~40k params/layer) vs Givens r=16
(192 params/layer — far fewer). The NoPE-only target (d=384 instead of d=2560) changes the
param budget significantly. K1013 sentinel bug (conv_step=train_steps+1 treated as valid)
must be fixed in T1.6 run_experiment.py.
