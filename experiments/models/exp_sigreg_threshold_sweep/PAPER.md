# PAPER.md — exp_sigreg_threshold_sweep

## Verdict: **KILLED (preempt-structural)**

No run was executed. Three independent structural guardrails each block this
experiment on their own; see MATH.md for full proofs.

## Triple-fire summary

| # | Guardrail | Census | Sub-variant |
|---|-----------|--------|-------------|
| 1 | F#666-pure (both KCs proxy) | 18th drain-window instance | — |
| 2 | §5 tautological-inter-variant-delta | 12th drain-window instance | intra-detector-threshold-delta (2nd intra-instantiation sub-variant after F#712) |
| 3 | F#669 parent-target-unverified | 8th drain-window reuse | parent F#713 SIGReg PROVISIONAL |

## Prediction-vs-measurement table

| Prediction | Measurement | Outcome |
|------------|-------------|---------|
| K1890 — FPR at τ=0.05 ≥ 30% on non-collapse events | **not measured** (preempt-KILL) | inconclusive |
| K1891 — FNR at τ=0.20 misses actual collapse events | **not measured** (preempt-KILL) | inconclusive |

Both KCs are classification-accuracy proxies (FPR, FNR) per guardrail 1007. No
target KC was pre-registered. A KC set with |proxy|=2 and |target|=0 admits
neither KILL nor SUPPORTED; preempt-structural KILL is mandatory.

## F#669 reuse ledger

| # | Experiment | Parent | Parent status | Drain pos |
|---|------------|--------|---------------|-----------|
| 1 | exp_jepa_adapter_router_prediction_error | F#682 | PROVISIONAL | F#687 |
| 2 | exp_jepa_adapter_attention_output | F#682 | PROVISIONAL | F#698 |
| 3 | exp_jepa_adapter_output_space | F#682 | PROVISIONAL | F#699 |
| 4 | exp_jepa_multilayer_prediction | F#682 | PROVISIONAL | F#727 |
| 5 | exp_jepa_contrastive_variant | F#682 | PROVISIONAL | F#728 |
| 6 | exp_jepa_frozen_encoder_ablation | F#682 | PROVISIONAL | F#729 |
| **7** | **exp_sigreg_threshold_sweep** | **F#713** | **PROVISIONAL** | **F#730 (this)** |

First non-F#682 parent to spawn an F#669 reuse; parent F#713 is SIGReg
`exp_sigreg_composition_monitor` (design-lock, empirical deferred).

## Sibling-position table (F#713 children)

| Experiment | Status | Notes |
|------------|--------|-------|
| exp_sigreg_composition_monitor (F#713 parent) | PROVISIONAL (design-lock) | Three pre-registered surfaces |
| exp_sigreg_hedgehog_combined | KILLED (F#714) | F#666-pure + §5 + hygiene-multi-defect triple-fire |
| exp_sigreg_threshold_sweep | **KILLED** (F#730 proposed) | triple-fire F#666-pure + §5 + F#669 |

This is the **1st same-parent-F#713 F#669-reuse** instance (exp_sigreg_hedgehog_combined
at F#714 did not invoke F#669 because it did not anchor against F#713's default
threshold; its triple composition was different).

## Triple-fire ledger (structural/parent-dependent sub-composition)

| # | Experiment | Parent | Composition |
|---|------------|--------|-------------|
| 1 | exp_jepa_contrastive_variant (F#728) | F#682 | F#666-pure + §5 + F#669 |
| 2 | exp_jepa_frozen_encoder_ablation (F#729) | F#682 | F#666-pure + §5 + F#669 |
| **3** | **exp_sigreg_threshold_sweep (F#730)** | **F#713** | **F#666-pure + §5 + F#669 — 1st cross-parent** |

Per F#729 analyst note: "If 3rd post-promotion same-parent-F#682 child fires same
triple-composition, consider standalone sub-type promotion." This is the **3rd
instance** but with **different parent**, so the strict same-parent promotion trigger
does NOT fire. However, the **cross-parent generalisation** may itself warrant
analyst attention — the triple-fire composition is robust across parents, not
specific to JEPA residual-stream geometry.

## §5 intra-instantiation sub-variant ledger

| # | Experiment | Sub-variant |
|---|------------|-------------|
| 1 | exp_g4_svd_truncate_adapter (F#712) | intra-adapter-rank-delta (rank 4/8/16) |
| **2** | **exp_sigreg_threshold_sweep (F#730)** | **intra-detector-threshold-delta (τ 0.05/0.10/0.15/0.20)** |

2 instances of intra-instantiation sub-variant; both anchored on monotone
operating-point sweeps.

## Antipattern audit

- F#666-pure: **FIRES** (both KCs proxy; no target pairing)
- §5 tautological-inter-variant-delta: **FIRES** (intra-detector-threshold-delta)
- F#669 parent-target-unverified: **FIRES** (parent F#713 PROVISIONAL)
- F#702 hygiene-patch: **APPLIED** (platform=local-apple, dir set, success_criteria
  populated at `experiment complete`)
- Composition math / LORA_SCALE / shutil.copy / hardcoded pass / eval truncation /
  proxy-model / eval-template: **N/A** (no run)

## Re-claim requirements

To convert this KC set into a verdict-eligible experiment:
1. Pair each proxy KC with a target metric (downstream task accuracy at chosen τ,
   behavioural collapse prevention on a real composed run).
2. Wait for parent F#713 `_impl` to provide an empirical ground-truth event set.
3. Anchor the sweep via external Neyman-Pearson / ROC argument, not intra-sweep.

Until all three are done, any re-claim will re-trigger the same triple-fire.

## References

- F#666 / guardrail 1007 (proxy-target pairing KILL discipline)
- F#669 (parent-target-unverified)
- F#703 (analyst escalation for F#666-pure standalone)
- F#712 (§5 intra-instantiation sub-variant precedent)
- F#713 (parent SIGReg PROVISIONAL design-lock)
- F#728 / F#729 (structural/parent-dependent triple-fire — F#682 parent)
