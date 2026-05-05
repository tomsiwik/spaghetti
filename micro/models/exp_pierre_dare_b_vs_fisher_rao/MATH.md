# MATH.md — B-space DARE vs Fisher-Rao Karcher mean (Pierre arch)

## Hypothesis

Pierre's `compose_adapters` operates on **B-matrices only**, not full deltas, because adapters share a frozen Grassmannian A per layer. Research DARE (`exp_pierre_dare_vs_m2p_gated_postfix`, `exp_pierre_composition_method_ablation`) measured +4.4pp avg over best-single using **full-delta DARE on per-adapter A**.

These two settings are not equivalent. The question:

> **Does an architecture-honest B-space DARE (drop 90% of B-entries, rescale ×10, then weighted mean) match Pierre's current Fisher-Rao Karcher mean — and does it approach the research full-delta DARE upper bound?**

Decision rule:
- `dare_b avg ≥ fisher_rao avg` → swap Pierre's default to `dare_b`.
- Else → keep `fisher_rao`. Research DARE doesn't transfer architecturally.

## Architectural setup (what Pierre actually does)

```
PoLARLinear forward:
    out = base(x) + scale · (x @ A) @ B

Composition layer (pierre/packages/model/pierre_model/compose.py):
    inputs:  K adapter B-dicts (each: layer_key → B[r, d_out])
    output:  one composed B-dict (same shape, ready for PoLARLinear.lora_b)
    A is shared and frozen across siblings; never composed.
```

This experiment **mirrors that architecture**: copy `compose_adapters`
verbatim into `compose_methods.py`, add `dare_b`, evaluate both.
The 7 PoLAR adapters available in `adapters/` were trained with the
same seed (so initial A's were identical) and minor post-training drift
in A is approximated by picking `adapter[0]`'s A as the shared A. We
flag this as a limitation; pierre-style frozen-shared-A training is
out of scope.

## Composition methods under test

### M0 — `single_best` (control)
Best per-domain adapter on its native benchmark. No composition. Establishes
the "no-fusion" baseline.

### M1 — `fisher_rao` (Pierre's current default)
Verbatim copy of Pierre's `compose_adapters`:
1. Stack K adapter B-matrices per (layer, module) key.
2. Weighted mean: `B_mean = Σ_k w_k · B_k` (default w = 1/K).
3. Norm-rescale: `B_out = B_mean · (mean_source_norm / ‖B_mean‖_F)`.

Mathematically the closed-form Karcher mean for Stiefel siblings.

### M2 — `dare_b` (new candidate, B-space DARE)
Architecture-honest port of DARE to Pierre's B-only path:
1. Per-adapter, drop 90% of B-entries i.i.d. with mask `M_k ~ Bernoulli(0.1)`.
2. Rescale survivors by ×10: `B_k_dare = (B_k ⊙ M_k) / 0.1`.
3. Weighted mean of the dropped+rescaled B's: `B_out = Σ_k w_k · B_k_dare`.

Element-wise drop on B is **not** equivalent to drop on `A @ B`, but
preserves DARE's expectation-preserving property at the B level
(`E[DARE(B)] = B`). This is the only DARE-shaped operation that respects
Pierre's storage contract.

### M3 — `dare_full_delta` (research upper bound)
Reproduces the research path with per-adapter A (no shared-A constraint):
1. Per-adapter, compute `ΔW_k = scale · A_k @ B_k` (shape `d_in × d_out`).
2. Element-wise DARE on full delta: drop 90%, rescale ×10.
3. Weighted mean of dropped deltas: `ΔW_fused = Σ_k w_k · DARE(ΔW_k)`.
4. Apply via `_FusedDeltaLinear` wrapper (Finding #831 canonical pattern).

Tells us how much accuracy Pierre's shared-A constraint costs.

## Pre-registered Kill Criteria

- **K1 (DECISION)** `dare_b_avg ≥ fisher_rao_avg` (point estimate). PASS → swap Pierre default.
- **K2 (COMPOSITION VALUE)** `dare_b_avg ≥ best_single_avg + 2pp`.
- **K3 (ARCH TRANSFER)** `|dare_b_avg − dare_full_delta_avg| ≤ 5pp`. PASS → shared-A doesn't sacrifice the research gain.
- **K4 (REPRODUCIBILITY)** `|fisher_rao_avg − 73.3pp| ≤ 3pp` (research DARE reference is 73.3 across 3 evals; Fisher-Rao should be in the same neighborhood since both are "linear average + rescale" — if it's far off, the eval pipeline drifted from research).

## Eval protocol

- N = 50 per benchmark (research used N=30 with ±23pp HumanEval noise; doubling sample size to tighten).
- Benchmarks: GSM8K, HumanEval, MedQA. Same prompts/templates as `scripts/polar_train.py::eval_*`.
- Generation: greedy (temp=0), max_tokens per `polar_train.py` defaults.
- Same fixed seed=42 for dataset shuffle so all methods see identical examples.
- Adapters: 7 PoLAR (bash, code, finance, legal, math, medical, sql) trained from `polar_train.py` recipe.
- Base: `mlx-community/gemma-4-e4b-it-4bit` (Gemma 4 E4B 4-bit), per Pierre's deployment.

## Verdict logic

| K1 | K2 | K3 | Outcome |
|----|----|----|---------|
| ✓ | ✓ | ✓ | **SUPPORTED** — swap Pierre default to `dare_b`; full transfer confirmed. |
| ✓ | ✓ | ✗ | **SUPPORTED** — swap Pierre default; note shared-A leaves accuracy on the table (future work: fused-delta refactor). |
| ✓ | ✗ | * | **SUPPORTED** — swap, but flag composition value vs single is marginal. |
| ✗ | * | * | **KILLED** — keep Fisher-Rao; document that research DARE does not transfer to Pierre's B-only arch. |

K4 is a sanity gate: if it fails on either method, the eval pipeline itself is suspect and the experiment is **inconclusive** (re-run with diagnostics, not a SUPPORTED/KILLED verdict).

## What this experiment is NOT

- Not a re-measurement of full-delta DARE in research arch (already done in `exp_pierre_dare_vs_m2p_gated_postfix`, settled).
- Not a re-baselining of best-single (already done across 4 prior experiments).
- Not a refactor proposal for Pierre's storage path. If `dare_b` loses to `dare_full_delta` by >5pp (K3 fail with K1 pass), that's evidence for a future fused-delta refactor — but this experiment doesn't undertake it.

## References

- Pierre `compose.py` (Fisher-Rao): `pierre/packages/model/pierre_model/compose.py`
- Research DARE: `exp_pierre_composition_method_ablation`, `exp_pierre_dare_vs_m2p_gated_postfix`
- Finding #831: `_FusedDeltaLinear` canonical pattern
- DARE paper: arxiv 2311.03099 (Yu et al., 2024)
