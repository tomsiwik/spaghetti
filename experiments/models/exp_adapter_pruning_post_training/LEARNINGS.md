# LEARNINGS — exp_adapter_pruning_post_training

## Core Finding
50% per-matrix magnitude pruning of a trained rank-6 LoRA adapter (q_proj, α=6.0, Gemma-4 E4B 4-bit, medical domain) increases single-adapter PPL by +0.331 on its training domain — **3.3× over K1922's 0.10 threshold (KILL, F#745)**. Weight-space predictions from MATH.md were tight: retained energy 0.886 predicted vs 0.904/0.922 measured; relative ‖ΔW−ΔW'‖_F gap 0.3–0.5 predicted vs 0.376 measured. PPL slope was over-estimated (0.6–1.2 predicted vs 0.331 measured; effective slope ≈0.88 PPL per unit relative ΔW — lower than F#674 cross-domain ablation slope because the medical adapter contributes only ~0.7 PPL).

## Why
Magnitude-only pruning zeroes entries that are small **in absolute value**, but a trained rank-6 LoRA stores its function in the *singular structure* of A·B, not in individual entries. Top singular directions are dense linear combinations; magnitude pruning fragments them. Wanda (arxiv:2306.11695) shows at 50% sparsity on full LLM weights that **magnitude alone fails** — activation-statistics (|w| × ‖x‖) are required; our experiment is the LoRA-analog of that result on a rank-6 adapter where the structural cost is more acute (dense orthogonal bases cannot be pruned independently per matrix). LoRAPrune (arxiv:2305.18403) reports task-accuracy preservation at K=50% on GLUE, but **PPL preservation is strictly harder than task-accuracy preservation** — consistent with F#666 (proxy ≠ target).

## Novel side-observation (F#746, provisional)
Under 2-adapter additive composition (med+math, both pruned), ΔPPL = **−0.129** — opposite sign from the +0.47 predicted by independent-error addition (√2 × single-adapter degradation). Three candidate mechanisms undistinguished: (1) destructive-interference removal at small |w| entries, (2) implicit regularization from reduced effective scale, (3) measurement noise at σ≈0.1, N=100. This is a **new axis** for the composition research line — not a rescue of this kill.

## Implications for Next Experiment
1. **Do not run LoRA pruning experiments at K=50% magnitude-only** — the structural cost is the dominant failure mode; activation-weighted (Wanda-style) or SVD-truncation (Eckart–Young) are structurally distinct operations requiring separate theorems.
2. **F#746 follow-up is cheap and high-value**: keep_frac ∈ {0.3, 0.5, 0.7, 0.9} × {med+math co-trained, med+code not-co-trained}. If co-trained pair shows *larger* composition-pruning gain than non-co-trained, destructive-interference mechanism is supported and Pierre gets a free regularization knob during composition.
3. **Calibration datum**: for rank-6 LoRA at α=6.0 on a modest-contribution domain, PPL slope per unit relative ΔW is ≈0.88 — lower than F#674's cross-domain ablation slope (~2–4). Use this in future pruning-prediction MATH.md.
4. **Pierre serving impact**: halved-memory dense-LoRA via magnitude pruning is **not viable** at K=50% for standalone quality. Viable memory-reduction paths require different math (SVD truncate materialized ΔW; or Wanda-style importance).

## References
- Wanda (arxiv:2306.11695) — explains why magnitude-only fails at 50%; activation statistics are required.
- LoRAPrune (arxiv:2305.18403) — direct prior on LoRA magnitude pruning; task accuracy ≠ PPL preservation.
- F#674 — cross-domain adapter ablation slope (1.5–3.0 PPL); our slope 0.88 is lower, consistent with modest-contribution single-domain adapter.
- F#744 — same scaffolding (LoRALinear monkey-patch, q_proj r=6, Gemma-4 E4B 4-bit).
