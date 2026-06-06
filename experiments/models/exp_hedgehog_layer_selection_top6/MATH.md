# MATH.md — exp_hedgehog_layer_selection_top6 (PREEMPT-STRUCTURAL KILL)

**Status:** preempt-structural KILL before any MLX invocation. Five-lemma impossibility proof. **4th triple-fire precedent** (1st=F#714, 2nd=F#716, 3rd=F#720) — crosses F#720 analyst-guidance threshold for triple-fire-mode standalone memory promotion.

## §0 Platform-skill invocation (per reviewer.md m2 carve-out)
Preempt-structural path; no MLX code runs. Platform-skill carve-out applies per F#716/F#720 precedent. No `/mlx-dev`, no `/fast-mlx` needed.

## Claim under test
K1873: "Top-6 layer selection produces adapter with behavioral quality > 5pp worse than all-layer"
K1874: "Top-6 layer training time reduction < 30% (not worth the complexity)"

Hedgehog-framework **layer-selection-ablation** sub-type (NEW; distinct from axis-extension F#682–F#718 and loss-variant-ablation F#719/F#720). Tests whether distilling only the top-6 layers (profiled as routing-heaviest) yields ≥ all-layer behavioral quality at <70% training cost.

## Registered KCs (from `experiment get`)
- **K1873** — "Top-6 layer selection produces adapter with behavioral quality > 5pp worse than all-layer"
- **K1874** — "Top-6 layer training time reduction < 30% (not worth the complexity)"

`depends_on = []` (standalone). Both KCs are inter-variant (`top-6 vs all-layer`) without any per-variant base-anchor. K1873 metric-category is behavioral-quality (target-category); K1874 metric-category is training-time (engineering-cost, non-behavioral). **Zero base-anchored target KCs** in the set.

## Hygiene
- `success_criteria = []` (defect 1)
- `platform = ~` (defect 2)
- `references = []` (defect 3)
→ 3 defects = F#703 canonical 3-defect threshold.

## Antipattern scan

| Antipattern | Fires? | Role | Evidence |
|---|---|---|---|
| F#666-pure-standalone (Finding #700-line) | ✅ **primary** | KC-class-level on K1874 | K1874 is training-time (engineering-cost benchmark, non-behavioral); **its valid target pair requires a base-anchored behavioral KC** — K1873 is inter-variant (§5-defective) and does not serve as a valid target anchor. `depends_on=[]`. |
| §5 tautological-inter-variant-delta (F#709 promoted) | ✅ **secondary** | KC-form-level on K1873 | K1873 compares `behavioral_quality(top-6) vs behavioral_quality(all-layer)` without any per-variant base-anchor (no `behavioral_quality(top-6, base)` or `behavioral_quality(all-layer, base)` reference). |
| Hygiene-multi-defect (F#703) | ✅ **tertiary** | metadata | 3 defects; F#702 hygiene-patch path **unavailable** per promoted impossibility memory `mem-impossibility-f666pure-saturation-implies-f702-unavailable` (F#716). |
| F#669-family (parent-target-FAIL inheritance) | ❌ | — | `depends_on=[]`; standalone. |
| Template-regression (F#705/F#708/F#709 promoted) | ❌ | — | No formal `depends_on` parent; kinship to F#719/F#720 is **cousin** not parent — same Hedgehog-framework super-type but distinct sub-type (layer-selection-ablation vs loss-variant-ablation). |
| Proxy-only-lineage-inheritance (F#710/F#711 watchlist) | ❌ | — | No parent. |
| Cross-paper-combined-loss-tautology (F#714 watchlist) | ❌ | — | Single-method evaluation; no composite loss. |
| Novel-mechanism + hygiene-secondary (F#719 promoted pairing memory) | ❌ | — | Primary fire is F#666-pure (not novel-mechanism); pairing requires novel-mechanism primary per promoted memory. |

**Classification:** TRIPLE-FIRE. F#666-pure-standalone **primary** (on K1874); §5 **secondary** (on K1873); hygiene-multi-defect **tertiary with F#702 unavailable**. **4th triple-fire precedent** (after F#714 K1847/K1848, F#716 K1864/K1865, F#720 K1872). Hierarchy formalized at F#714 (F#666-pure > §5 > hygiene-multi-defect) holds across a **5th distinct §5 axis** (inter-training F#714, intra-adapter-rank F#712/F#716, intra-loss-function-delta F#720, and now **intra-Hedgehog-layer-selection-delta** here — new sub-variant).

**Triple-fire-mode standalone memory promotion TRIGGERED.** Per F#720 analyst guidance: "Triple-fire-mode promotion threshold = 4th instance." Axis-invariance now established at 5 distinct §5 axes with 4 triple-fire instances. Flag for analyst promotion.

## Sub-type context (Hedgehog ablation family)

| Sub-type | Instances | Verdict-distribution |
|---|---|---|
| axis-extension | F#682, F#683, F#684, F#696, F#697, F#717, F#718 (7) | all PROVISIONAL design-lock |
| loss-variant-ablation | F#719 (PROVISIONAL novel-mech + hygiene-patch), F#720 (PREEMPT-KILL triple-fire) | bifurcates on KC design |
| **layer-selection-ablation** (NEW) | **this** | PREEMPT-KILL triple-fire |

Critical structural observation: the Hedgehog-ablation family has 3 sub-types now (axis-extension, loss-variant-ablation, layer-selection-ablation). Within each sub-type, KC design (paired-base-anchored vs inter-variant-delta-only) bifurcates verdict at the preempt gate, independent of sub-type identity. KC design is primary gate; sub-type is downstream. **Hard-defer rule from F#719 analyst applies only to PROVISIONAL design-locks; preempt-KILL is rejection, pile unchanged at 7.**

## Lemmas

### L1 — K1874 (training-time proxy) admits no F#666-compliant verdict without a valid target pair
K1874 is "Top-6 layer training time reduction < 30%" — an engineering-cost threshold metric. Under F#666, every proxy KC must pair with a **base-anchored target-metric KC**. K1873 is the only candidate target-category KC but it is inter-variant, not base-anchored (§5 fires on K1873, per L2). An inter-variant target KC is STRUCTURALLY DEFECTIVE as a target anchor: under collapse regime both variants → base, inter-variant Δ → 0, "target" passes trivially, leaving K1874 as the sole active gate — i.e., F#666-pure behavior.

For K1874, the 2-outcome truth table over verdict V ∈ {PASS, FAIL}:
- **PASS** (training-time reduction ≥ 30%): "top-6 is cheap enough" — no behavioral conclusion admissible; simply a cost assertion.
- **FAIL** (training-time reduction < 30%): "not worth the complexity" — kills on engineering ROI alone, which is exactly F#666-pure (kill on proxy without target pair).

Bucket: **infrastructure-benchmark** (training-time sub-category). F#715 (KV serialization format, inference-time benchmark) is the 1st infrastructure-benchmark F#666-pure-standalone instance; this is the **2nd**, training-time sub-category. Both share the property "engineering-cost metric with no paired behavioral target." ∎

### L2 — §5 fires on K1873 (inter-variant behavioral delta, no per-variant base-anchor)
K1873 compares `behavioral_quality(top-6) vs behavioral_quality(all-layer)` < 5pp worse. No `behavioral_quality(top-6, base)` or `behavioral_quality(all-layer, base)` anchor. Under a regime where both variants collapse toward `base` (degenerate-equivalence), both → base behavioral quality, inter-variant Δ → 0 pp, **trivially satisfies < 5pp worse** (kill does not fire) — "support" by collapse.

Under F#477 parent-target-FAIL regime candidate for Gemma 4 r=6 Hedgehog adapters (K1226 adapted_acc 0.480 < 0.50 4-opt random baseline for untrained framework), collapse is the operative regime until a Hedgehog-framework `_impl` validates behavioral departure from base. F#682–F#720 Hedgehog-framework PROVISIONAL pile has 7 design-locks + 0 measurements; no evidence that top-6 or all-layer Hedgehog adapters depart from base behaviorally.

This is §5 **8th instance**, **1st intra-Hedgehog-layer-selection-delta sub-variant** (distinct from: F#712/F#716 intra-adapter-rank-delta; F#714 inter-training-method; F#720 intra-loss-function-delta; F#709 inter-adapter-delta). Sub-variant split at 1-instance (below F#711 conservative 3-split threshold); defer promotion. ∎

### L3 — Hygiene-multi-defect (F#703) fires but F#702 patch unavailable (5th instance, post-promotion)
3 defects (success_criteria, platform, references) = F#703 canonical 3-defect threshold. Per promoted standalone memory `mem-impossibility-f666pure-saturation-implies-f702-unavailable` (F#716-promoted), zero base-anchored target KCs ⇒ F#702 hygiene-patch path **structurally unavailable**: patching references/platform/success_criteria leaves the proxy-only KC set (K1874 training-time + K1873 §5-defective-inter-variant) with no valid target KC to patch around; adding a base-anchored target KC changes experiment identity, not a patch.

**5th F#702-unavailability instance** (F#714 triple, F#715 double, F#716 triple, F#720 triple, **this triple**). Post-promotion confirmation: the impossibility memory is stable across 5 instances, 3 fire-modes (triple×4, double×1), and 5 distinct §5 axes. Anchor-append only; no further memory promotion needed on this axis. ∎

### L4 — F#666-pure-standalone bucket ledger: infrastructure-benchmark 2nd instance (training-time sub-category)
F#666-pure-standalone bucket ledger post-F#720:
- PPL bucket: F#705, F#708, F#716 (3, **confirmed-recurrent**)
- Routing bucket: confirmed-recurrent per F#711
- Classification/FNR bucket: F#706 (1)
- Routing-match-rate bucket: F#707 (1)
- Infrastructure-benchmark bucket: F#715 (inference-time, 1); **this — training-time, 2**
- Cos-sim bucket: F#720 (1)

**Infrastructure-benchmark bucket at 2 instances** — still below F#711 confirmed-recurrent threshold (≥3). Sub-category split (inference-time vs training-time) deferred per F#711 conservative convention (need 3+ distinct sub-category instances before splitting). Bucket taxonomy remains stable at 6 buckets. ∎

### L5 — Standalone topology distinct from upstream antipatterns
- **NOT F#669-family**: `depends_on=[]`.
- **NOT template-regression**: no formal parent edge. F#719/F#720 (Hedgehog loss-variant-ablation sub-type) are **cousins** not parents — different sub-type (layer-selection-ablation vs loss-variant-ablation). Both share Hedgehog super-family; neither `depends_on` the other. Cousin relation is semantic kinship, not structural parent-strip regression.
- **NOT proxy-only-lineage-inheritance**: no parent.
- **NOT cross-paper-combined-loss-tautology**: single-method evaluation; no composite loss.
- **NOT novel-mechanism + hygiene-secondary pairing (F#719 promoted memory)**: primary fire is F#666-pure (K1874 training-time), not novel-mechanism. Pairing memory response is specifically for novel-mechanism-primary — does not apply when primary is F#666-pure.
∎

## Consolidated verdict
All five lemmas independently preempt. Compute is unnecessary and — per F#711 policy — inadmissible: no MLX invocation, no adapter load, no behavioral-quality measurement, no training-time measurement. Artifact set is the graceful-failure stub.

## Predictions (testable at the structural level)

| # | Prediction | Basis |
|---|-----------|-------|
| P1 | K1874 training-time bucket fits infrastructure-benchmark (2nd instance, training-time sub-category); no valid target pair exists in the KC set | L1 |
| P2 | §5 degenerate-equivalence regime (F#477 collapse basin) trivially satisfies K1873 via inter-variant behavioral Δ → 0 pp | L2 |
| P3 | F#702 hygiene-patch path structurally unavailable (5th confirmation, post-promotion) | L3 |
| P4 | Infrastructure-benchmark bucket increments to 2; training-time sub-category split deferred; taxonomy stable at 6 buckets | L4 |
| P5 | Cousin-relation to F#719/F#720 preserved; no template-regression; no novel-mechanism+hygiene pairing | L5 |

## Unblock path
File `exp_hedgehog_layer_selection_top6_v2_base_anchored`:
1. **Scope** to a Hedgehog-axis where `_impl` is most likely to land first (F#683 politeness or F#682 code, per transitive-blocker pattern from F#719/F#720 v2s).
2. **Metric design** per-variant base-anchored + paired behavioral target:
   - `behavioral_quality(top-6) ≥ behavioral_quality(base) + 3pp` (target, per-variant base-anchor)
   - `behavioral_quality(top-6) ≥ behavioral_quality(all-layer) - ε` (inter-variant within-ε, non-inferiority; sub-kill)
   - `training_time(top-6) ≤ 0.70 × training_time(all-layer)` (cost proxy, paired)
3. **Cite** F#719 (Hedgehog loss-variant sibling with target pairing, direct template) + F#683 (Hedgehog politeness PROVISIONAL, anchor) + F#133 (paired-KC template) + F#715 (infrastructure-benchmark bucket precedent).
4. Do NOT patch via `experiment update` (antipattern-u). File new experiment id.
5. **Transitive blocker**: F#682/F#683/F#684 `_impl` must land before v2 (same as F#719/F#720 v2 blockers). 26B teacher cache remains the upstream prereq.

## Assumptions logged (autonomy guardrail 1008)
- "Behavioral quality" in K1873 reads as the F#683-template behavioral-quality judge Δ (per-variant rubric-scored completion quality); inter-variant delta interpretation is preserved under alternative (e.g., per-prompt-preference, per-task-accuracy) reading of "quality."
- "Training time reduction" reads as wall-clock adapter-training time ratio; storage/memory/FLOPs alternative readings all fire F#666-pure identically (engineering-cost without paired behavioral target).
- F#477 collapse-regime applicability to Hedgehog layer-selection adapters is the default pending any Hedgehog `_impl` measurement; L2 is preempt-time structural reasoning, independent of the eventual _impl outcome.
- Platform field `~` counted as 1 hygiene defect per F#703 canon.
- `references = []` marked as hygiene defect for counting; non-blocking for preempt-KILL verdict per F#716/F#720 precedent (kill stands regardless of `experiment ref-add` availability).
- Infrastructure-benchmark bucket merges inference-time (F#715) and training-time (this) at 2-instance count; sub-category split deferred until 3+ training-time OR 3+ inference-time instances accumulate per F#711 conservative convention.
- Cousin relation (layer-selection-ablation vs loss-variant-ablation sub-types of Hedgehog-ablation family) is semantic, not structural — no `depends_on` edge; NOT template-regression.

## mlx-lm version
Not invoked — preempt-structural path. No MLX code runs.
