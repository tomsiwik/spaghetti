# LEARNINGS — exp_followup_kv_cache_reuse_honest

## Key finding

**The parent's 1.6% vs 62.5% Theorem 2 contradiction was a Frobenius-vs-operator-norm confusion, compounded by arithmetic slips.** The honest bound has two components (first-order D1, second-order D2) that dominate at different α regimes — this math correction is real and reusable. **But the operator-norm bound is ~11–13x looser than simulated drift at α ∈ {5, 10, 20}**, so it is not predictive enough for engineering decisions. Verdict: KILLED under F#666 (both proxy K1566 and independent target K1945 fail honestly on bound-vs-simulation magnitude).

## REVIEW r1 correction (2026-04-24)

The first-pass K1945 compared `scaling_ratio_sim` to `scaling_ratio_bound`, where `scaling_ratio_bound = predicted_at_5 / predicted_at_20` and `predicted_at_α = sim_rel_Drift_α · (1/sqrt(d_k)) · (sqrt(L)/L) · 100` — the constant cancels in the ratio, so `scaling_ratio_bound ≡ scaling_ratio_sim` algebraically. The "1.00x" match was a tautology. The fix computes an **independent** closed-form bound per-trial from sampled `‖W_Q‖_op`, `‖W_K‖_op`, measured γ — no algebraic dependence on the simulated Drift — and compares magnitudes. The honest result is the bound is ~11x loose at α=20 (sim 0.655 vs bound 7.43), so K1945 FAILs, and the F#666 proxy-FAIL + target-PASS escape no longer applies.

## Non-obvious discoveries

1. **LoRA drift decomposes into D1 + D2, not just D2.** Parent and most of the LoRA-KV-reuse literature treat the cross-adapter perturbation as a single quantity. It's actually the sum of:
   - D1 = `α W_Q^T (ΔK_A − ΔK_B)` — linear in α, non-Grassmannian-suppressable (does not involve A^A⊥A^B structure).
   - D2 = `α² (ΔQ_B)^T (ΔK_A − ΔK_B)` — quadratic in α, partially Grassmannian-suppressable (via γ factor).

   At α=5, D1 is 5x larger than D2. At α=20 they are comparable. This is why lowering α helps but does not eliminate the drift.

2. **Grassmannian γ is measurable and ≠ 0.** Even with partitioned-QR-initialized A-matrices for different adapters, the measured γ = `‖A_B^T A_A‖_F / sqrt(r)` is ≈ 0.08 (not 0). Perfect orthogonality is a limit case, not the typical case. F#309's "impossibility structure" is thus quantitative, not binary — orthogonality suppresses D2 by ~92 % but does not eliminate it.

3. **Residual-stream attenuation factor is a hidden free variable.** Published transformer-circuit analyses (Elhage et al. 2021 and descendants) do not give a closed form for per-layer perturbation accumulation under correlated perturbations. Uncorrelated-sum assumption gives factor 1; fully-correlated gives factor sqrt(L). The true factor depends on the specific structure of the perturbation (in this case: all layers see the same adapter-A-vs-adapter-B KV split, suggesting high correlation).

## What this suggests about future research

- **Follow-up priority 2:** measure per-layer drift correlation empirically on Gemma 4 E4B with trained adapters. This would close the [5.78%, 30.6%] interval for KC1566 and either resurrect or finalize F#309.
- **Follow-up priority 3:** check whether the α=5 regime (D1-dominated, predicted drift ~1.8–10 %) admits narrow KV-reuse operating ranges (e.g., very short segments, semantically-similar adapter pairs like F#309's math+medical where context_improvement was +1.72 %).

## Proxy-KC refinement insight

Even after the independent-bound fix, K1566 ("within 2x") is likely too strict for a pure operator-norm bound, which is loose by construction. A refined proxy would use an RMS-based bound (`E[‖Drift‖² / ‖S0‖²]`) instead of the operator-norm upper bound — that would account for the hidden-state distribution rather than the worst-case direction, and is expected to close most of the ~11x gap. This is a generalizable refinement to the F#666 proxy-design catalog: operator-norm bounds are KILL-class unless paired with RMS or typical-case estimates.

## New antipattern candidates

**`scaling-ratio-tautology` (REVIEW r1 finding):** When a "bound" prediction is computed as `bound = simulated_value · constant`, and the constant cancels in any derived ratio, comparing `bound_ratio` to `sim_ratio` is a tautology — the test will pass at 1.00x regardless of whether the bound is correct. Guard: before trusting a "scaling match" or "consistency ratio" KC, verify that the two sides of the ratio are **algebraically independent**. Compute bounds from first-principles inputs (operator norms of sampled weights, measured structural factors like γ) — NOT by rescaling the simulated output. Antipattern family (f) + (g) per reviewer catalog.

**`residual-attenuation-not-measured`**: When deriving a bound for an L-layer network's final-layer quantity from per-layer quantities, the residual-stream correlation factor is not 1 (uncorrelated) or sqrt(L) (correlated) by default — it depends on the specific perturbation structure and must be measured or bounded. Publishing a single-point prediction without specifying this factor produces bounds that can be off by up to sqrt(L) ≈ 5x for L=28. Guard: when writing MATH.md for multi-layer drift bounds, explicitly either (a) justify the correlation assumption from perturbation structure, or (b) report an interval [constant, sqrt(L)] and acknowledge the uncertainty.

## Platform notes

- Simulation ran in 1.0 s on numpy; no MLX, no model loading. Correct choice for bound verification — full-LLM runs would have added 5–30 min with the same bound verification outcome (since the bound is analytic, not empirical).
- BitNet-2B dimensions used for direct comparability to parent F#309. Dimension-substitution to Gemma 4 E4B would be a trivial follow-up if needed.

## Confidence

**High** that the math correction (D1+D2 decomposition, √r factor, Frobenius-vs-operator-norm fix) is correct (reproducible derivation, consistent with measured scaling structure). **High** that the operator-norm bound is too loose to be predictive (ratio sim/bound ≈ 0.08 across three α values, genuinely independent comparison). **Medium** that an RMS-based bound would close the looseness gap (testable with a scoped follow-up).
