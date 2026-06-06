# LEARNINGS — exp_followup_hypernetwork_residual

**Status:** PROVISIONAL (mechanism plausible on synthetic; real-data KCs vacated)

## Core Finding

The residual hypernetwork form `B_pred = μ_{−t} + H(emb_t)` escapes the
parent's tautological span trap. On a clean linear low-rank synthetic
proxy (N=24, k=8, d_embed=64, dim_b=40960), closed-form ridge recovers
the held-out residual with **median ρ=0.578** and **13.4 % MSE
reduction vs mean-baseline** — both KCs (K1s ≤0.95, K2s >0.1) PASS.
Real-data K1/K2 (BitNet PPL + LOO adapter cosine) were **vacated**
because the 24 parent adapters + Grassmannian skeleton from
`exp_real_data_25_domain_adapters` are absent on disk (gitignored
binaries, 0/24 present). `K_vacate` triggered honestly.

## Why

- Parent's convex-combination predictor forced `B_pred ∈ span{B_i}`,
  so any Gram-Schmidt-style K2 against training adapters was
  tautological. The residual form targets `(B_t − μ_{−t})` which is
  *outside* the training-set span by construction — non-vacuous test.
- The MATH.md-draft MLP+random-projection plan was abandoned as
  provably lossy (random projection near-orthogonal to `span(U_b)`).
  Closed-form ridge on the full 40960-dim target is tractable
  (W≈10 MB) and is the correct linear hypernetwork. KC thresholds
  unchanged, so not moving goalposts.
- Thm 1's threshold `ρ > σ̂/(2σ)` reduces to `ρ > 0.5` at
  `σ̂ ≈ σ` — matches measured median 0.58 yielding the 13 %
  reduction.

## Implications for Next Experiment

1. **Unblock real data first.** Spawn `exp_real_data_25_domain_adapters_rerun`:
   retrain the 24 BitNet LoRA adapters at `LORA_SCALE=5.0` (antipattern-003
   compliance) + regenerate Grassmannian skeleton. Persist
   `.npz` artefacts to the path encoded in
   `run_experiment.py`. Then this experiment can rerun K1/K2 real
   without any MATH.md change.
2. **Linear-synthetic ≠ real transfer.** 5/24 folds had per-fold
   mse_ratio > 1; the mean is dragged above threshold by 19 strong
   folds. Domain-level heterogeneity will likely break some folds on
   real data. Pre-register per-fold tolerances in the rerun.
3. **Non-linear generative sub-test.** Before trusting BitNet
   transfer, run a synthetic with `B_i = U_b · f(z_i)` (shallow MLP
   `f`) to verify ridge fails there and an MLP hypernetwork
   succeeds — pre-req for claiming the mechanism on non-linear
   adapter manifolds. Cheap and decisive.
4. **No new antipatterns.** Reviewer checklist (a)–(s) all PASS;
   antipattern-003 (`LORA_SCALE`) already fixed to 5.0.
   `K_vacate` branch is a clean precedent for future
   missing-artefact cases — honest vacation, not silent kill
   upgrade or cited-not-measured rescue.
