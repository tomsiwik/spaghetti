# MATH.md — LoRA Hub learned scalars (architectural ceiling for shared-A B-only)

## Hypothesis & purpose

This is **not** a candidate composition method to ship. It is the
**architectural ceiling test** for Pierre's shared-A B-only design.

LoRA Hub (arxiv 2307.13269) optimizes per-adapter scalar weights $w_i$ via
gradient-free black-box search on a small validation panel. Under shared-A,
LoRA Hub's formula collapses to:

$$\Delta W_{\text{merged}} = \left(\sum_i w_i\right) \cdot A_{\text{shared}} \cdot \left(\sum_i w_i B_i\right)$$

So the operation is "learn K scalars for `Σ w_i B_i`". This is a **strict
superset of Fisher-Rao** (Fisher-Rao analytically chooses $w_i = 1/K$;
LoRA Hub *learns* them).

**The test**: if even learned scalars can't significantly beat Fisher-Rao,
the shared-A B-only space has hit its capacity ceiling — no scalar-weighting
scheme on B alone can recover the +6.7pp gap. That tells us the refactor
to per-adapter A is the only path to research-DARE-level accuracy.

> **Can learned per-adapter scalar weights `w_i` close the +6.7pp gap
> measured between Fisher-Rao (64.7%) and full-delta DARE (71.3%)?**

## Why this experiment matters even if it KILLS

A KILL here is **highly informative**:

- **Fisher-Rao close to optimal scalar mean** → the merge math at the scalar level is already near-optimal for shared-A B-only. The remaining gap requires either per-adapter A (refactor), structure-preserving merges (Pico/ACE/OrthoMerge), or per-token routing.
- **Confirms that scalar reweighting is the wrong knob** for closing the gap.

A SUPPORTED here is also informative:

- **Learned scalars closing 3+pp** → simple search is enough; consider production deployment of LoRA-Hub-style scalar learning.
- **Closing the full +6.7pp** → architectural change unnecessary.

## Algorithm

For each (layer, module) key (operating on Pierre's B-only output):

1. Per-key composed B is the **same scalar weighting**: `B_merged_layer = Σ w_i · B_i_layer` (one set of K weights, not per-layer).
2. Apply Pierre's existing norm rescaling: `B_out = B_merged · (mean_t‖B_t‖_F / ‖B_merged‖_F)`.

The K weights are optimized **once** across all layers via
`scipy.optimize.differential_evolution` with:

- Per-coefficient bounds: `(-1.5, 1.5)` (paper's range; allows negative weights)
- Budget: 40 evaluations (popsize=5 × maxiter=8) — paper's number
- Validation panel: 3 prompts × 3 benchmarks = 9 prompts
- Objective: minimize `−mean_accuracy` on the panel

Validation panel is held-out from the eval set (different shuffle seed) to
avoid leakage.

## Pre-registered Kill Criteria

- **K1 (DECISION)** Learned-scalars avg ≥ Fisher-Rao avg + 2pp.
  - **Lower threshold than other experiments** (2pp vs 3pp). LoRA Hub is the architectural-ceiling test, so even a small win is informative.
- **K2 (CEILING)** Learned-scalars avg within 5pp of full-delta DARE (71.3%).
  - Tells us if shared-A scalar-weighted B can ever match per-adapter A's expressiveness.
- **K3 (BUDGET)** Optimization completes within 60 minutes (40 evaluations × ~90s = ~60 min worst case).
- **K4 (INTERPRETABILITY)** At least 2 of 7 learned scalars deviate from `1/7` by ≥ 0.1.
  - If all weights converge to ~1/7, the optimizer learned "Fisher-Rao is best" — informative either way.

## Verdict logic & interpretation table

| K1 | K2 | Outcome | Architectural implication |
|----|----|---------|---------------------------|
| ✓ | ✓ | **SUPPORTED** | Learned scalars close the gap; ship LoRA-Hub-style merge. |
| ✓ | ✗ | **SUPPORTED** with caveat | Learned scalars help but don't close gap; refactor still needed. |
| ✗ | * | **KILLED, informative** | Shared-A B-only scalar space is exhausted; ONLY non-scalar methods (Pico/ACE/OrthoMerge) or per-adapter A can close the gap. |

**A KILL outcome here is the most informative result of the four queued
experiments.** It tells the product whether scalar-weighting is even a
viable axis of attack.

## Eval protocol

- N=50 per benchmark for final eval (same as other experiments)
- N=3 per benchmark for validation panel during optimization (held out)
- Same seed=42, same 7 PoLAR adapters, same shared-A donor
- Same `mlx-community/gemma-4-e4b-it-4bit` base
- Optimization uses cheaper proxy eval (3 prompts) to fit within 40-eval budget; final eval uses full 50 prompts

## Honest gaps

- **Differential evolution is not Nevergrad NGOpt** — they're both gradient-free black-box but DE has different exploration characteristics. NGOpt would require adding `nevergrad` as a dependency. DE is in scipy, no extra install. Paper's reported gains may not transfer exactly.
- **Validation panel size (9 prompts) is small** — comparable to LoRA Hub paper's 5-shot setting but smaller. Risk of overfitting the panel; mitigated by separate final eval set.
- **Single set of K weights across all layers** — paper's formula is global, not per-layer (LoRA Soups CAT does per-layer; that's a separate experiment if this one shows promise).
- **Bounds (-1.5, 1.5) allow negative weights** — DARE-style "anti-task" composition where a chef *subtracts* their contribution. May or may not be useful for Pierre's mix.

## References

- LoRA Hub paper (arxiv 2307.13269): https://arxiv.org/abs/2307.13269
- Reference implementation: https://github.com/sail-sg/lorahub (uses Nevergrad NGOpt)
- scipy.optimize.differential_evolution as gradient-free substitute
- Prior measurement: `exp_pierre_dare_b_vs_fisher_rao` (Fisher-Rao 64.7%, full-delta DARE 71.3%)
