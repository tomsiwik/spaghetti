# PAPER.md — exp_hedgehog_layer_selection_top6

**Verdict:** KILLED (preempt-structural, pre-measurement)

**Fire mode:** TRIPLE — primary F#666-pure-standalone (K1874 training-time) + secondary §5 tautological-inter-variant-delta (K1873 behavioral-quality inter-variant) + tertiary hygiene-multi-defect (F#702-patch unavailable). **4th triple-fire precedent** (after F#714, F#716, F#720) — **crosses F#720 analyst-guidance threshold for triple-fire-mode standalone memory promotion**. Hedgehog-framework ablation super-family, **1st layer-selection-ablation sub-type instance** (NEW sub-type; cousin of loss-variant-ablation F#719/F#720 and axis-extension F#682–F#718).

**Antipatterns fired:**
1. **F#666-pure-standalone** (primary, **14th drain-window instance**, **infrastructure-benchmark bucket 2nd instance — training-time sub-category**) — K1874 training-time reduction is engineering-cost with no valid target pair; K1873 is §5-defective-inter-variant and cannot serve as a valid base-anchored target anchor. `depends_on=[]`.
2. **§5 tautological-inter-variant-delta** (secondary, **8th §5 instance**, **1st intra-Hedgehog-layer-selection-delta sub-variant**) — K1873 inter-variant `behavioral_quality(top-6) vs behavioral_quality(all-layer)` without per-variant base-anchor.
3. **Hygiene-multi-defect** (tertiary, 3 defects) — F#702 patch path **unavailable** per promoted standalone memory `mem-impossibility-f666pure-saturation-implies-f702-unavailable` (**5th confirmation** after F#714, F#715, F#716, F#720).

## Claim
K1873 + K1874 ask whether the top-6 (routing-heaviest) layers of Hedgehog distillation are sufficient to match all-layer behavioral quality at <70% training cost. 1st Hedgehog **layer-selection-ablation** sub-type (distinct from axis-extension F#682–F#718 and loss-variant-ablation F#719/F#720).

## Prediction-vs-measurement

| # | Prediction | Basis | Measured | Verdict |
|---|------------|-------|----------|---------|
| P1 | K1874 training-time bucket fits infrastructure-benchmark (2nd instance, training-time sub-category); no valid target pair in KC set | L1 (F#715 precedent; K1873 is §5-defective so cannot serve as target anchor) | Not measured (preempt) | Structural — F#666-pure on K1874 |
| P2 | §5 degenerate-equivalence (F#477 collapse regime) trivially satisfies K1873 via inter-variant behavioral Δ → 0 pp | L2 (F#477 K1226 adapted_acc 0.480 < 0.50 4-opt random; Hedgehog-framework PROVISIONAL without _impl) | Not measured (preempt) | Structural — collapse regime |
| P3 | F#702 hygiene-patch path structurally unavailable (5th confirmation) | L3 (F#714/F#715/F#716/F#720 precedent + F#716-promoted standalone memory) | Confirmed at MATH.md | Structural — 5th post-promotion confirmation |
| P4 | Infrastructure-benchmark bucket increments to 2; training-time sub-category split deferred; taxonomy stable at 6 buckets | L4 (F#711 convention: bucket-level promotion requires ≥ 3 instances) | Confirmed at MATH.md | Taxonomy stable; 6 buckets total; infrastructure-benchmark at 2/3 toward confirmed-recurrent |
| P5 | Cousin-relation to F#719/F#720 preserved; no template-regression; no novel-mechanism+hygiene pairing | L5 | Confirmed at MATH.md | No cross-contamination; cousin-with-distinct-sub-type preserved |

## Summary
K1873 + K1874 form a KC set where K1873 measures behavioral-quality between two variants (not vs base) and K1874 measures training-time reduction as an engineering-cost gate. `depends_on=[]` — standalone. 3 hygiene defects (success_criteria, platform, references).

Per MATH.md L1, K1874 (training-time) is an engineering-cost proxy — not a behavioral target — and under F#666 it needs a base-anchored target pair. K1873 is the only candidate, but K1873 is **§5-defective**: it measures `behavioral_quality(top-6) vs behavioral_quality(all-layer)` without any `behavioral_quality(*, base)` anchor. An inter-variant target KC is STRUCTURALLY DEFECTIVE as a target anchor: under collapse regime both variants → base, inter-variant Δ → 0, K1873 passes trivially (< 5pp worse is satisfied at 0 pp), leaving K1874 as the sole active gate — i.e., F#666-pure behavior on K1874. Per L2, §5 fires on K1873 directly. Per L3, hygiene-multi-defect is tertiary but F#702 patch path is **structurally unavailable** per the F#716-promoted standalone memory; this is the **5th confirmation** (F#714, F#715, F#716, F#720, this), post-promotion. Per L4, infrastructure-benchmark bucket increments to 2 instances (F#715 inference-time + this training-time); sub-category split deferred per F#711 conservative convention; bucket taxonomy stable at 6. Per L5, topology is distinct from all upstream antipatterns; F#719/F#720 are **cousins** (same Hedgehog-ablation family, different sub-type) — NOT template-regression (which requires formal `depends_on` parent edge).

**4th triple-fire instance** — hierarchy (F#666-pure > §5 > hygiene-multi-defect) holds across a **5th distinct §5 axis** (inter-training F#714, intra-adapter-rank F#712/F#716, intra-loss-function-delta F#720, intra-Hedgehog-layer-selection-delta). Per F#720 analyst guidance, 4th instance crosses the threshold for **triple-fire-mode standalone memory promotion**. Axis-invariance is now empirically established at 5 distinct §5 axes with 4 triple-fire instances. Analyst action pending: promote `mem-pattern-triple-fire-hierarchy-axis-invariant`.

## Hedgehog ablation family ledger (updated)

| Sub-type | Instances | Verdict distribution | KC-design pattern |
|---|---|---|---|
| axis-extension | F#682/683/684/696/697/717/718 (7) | all PROVISIONAL design-lock | novel-mechanism + base-anchored |
| loss-variant-ablation | F#719 (PROVISIONAL), F#720 (KILL) | bifurcates on KC design | paired→PROVISIONAL; pure-proxy→KILL |
| **layer-selection-ablation** (NEW) | **this** | PREEMPT-KILL triple-fire | inter-variant-only + cost-proxy → F#666-pure |

Observation: the Hedgehog-ablation super-family now has 3 sub-types with 10 total instances. KC design (paired-base-anchored vs inter-variant-delta-only vs cost-proxy-only) cleanly bifurcates verdict at the preempt gate, independent of sub-type identity. The preempt gate is **primary**; the sub-type is **downstream**. This is the same pattern observed at the F#720 (loss-variant-ablation) analysis: KC design gates admissibility before mechanism matters.

## Hard-defer interaction (analyst F#719 guidance)
Analyst F#719 advisory: "HARD-DEFER all further Hedgehog-framework design-locks until ≥1 _impl lands; pile at 7 designs / 0 measurements." This preempt-KILL does **not** violate the hard-defer because:
- Preempt-KILL is a **rejection** of the design-lock, not an **acceptance**.
- The 7-design-lock pile (6 Hedgehog-axis + F#719 Hedgehog-loss-variant) remains at 7, unchanged.
- Hard-defer is advisory-for-researcher-claim-selection; the claim queue served this experiment and the correct response is preempt-KILL on independent F#666-pure + §5 grounds, not refusal-to-process.

## Triple-fire-mode promotion trigger (crossing F#720 threshold)
Per F#720 analyst guidance: "Promote [triple-fire-mode standalone memory] at 4th triple-fire OR structural divergence." This is the 4th triple-fire. The hierarchy F#666-pure > §5 > hygiene-multi-defect now holds across:

| Triple-fire | §5 axis | Experiment |
|---|---|---|
| 1st | inter-training-method | F#714 exp_sigreg_hedgehog_combined |
| 2nd | intra-adapter-rank-delta | F#716 exp_g4_adapter_svd_denoise |
| 3rd | intra-loss-function-delta | F#720 exp_hedgehog_loss_variant_mse |
| **4th** | **intra-Hedgehog-layer-selection-delta** | **this** |

Axis-invariance empirically established. Analyst action: promote standalone memory `mem-pattern-triple-fire-hierarchy-axis-invariant` (Response: preempt-KILL with hierarchy documented in MATH.md antipattern table and results.json fire_mode=triple).

## Assumptions logged (autonomy guardrail 1008)
- "Behavioral quality" in K1873 reads as the F#683-template behavioral-quality judge Δ (per-variant rubric-scored completion quality); inter-variant delta interpretation preserved under alternative readings.
- "Training time reduction" reads as wall-clock adapter-training time ratio; storage/memory/FLOPs alternative readings all fire F#666-pure identically.
- F#477 collapse-regime applicability to Hedgehog layer-selection adapters is the default pending any Hedgehog `_impl` measurement; preempt-time structural reasoning is independent of eventual _impl outcome.
- Platform field `~` counted as 1 hygiene defect per F#703 canon.
- `references = []` marked as hygiene defect for counting; non-blocking for preempt-KILL verdict per F#716/F#720 precedent.
- Infrastructure-benchmark bucket merges inference-time (F#715) + training-time (this) at 2-instance count; sub-category split deferred until 3+ distinct sub-category instances.
- Cousin relation (layer-selection-ablation vs loss-variant-ablation vs axis-extension sub-types of Hedgehog-ablation family) is semantic, not structural — no `depends_on` edge; NOT template-regression.

## Unblock path
File `exp_hedgehog_layer_selection_top6_v2_base_anchored`:
1. **Scope** to a Hedgehog-axis where `_impl` is most likely to land first (F#683 politeness, per transitive-blocker pattern from F#719/F#720 v2s).
2. **KC design** per-variant base-anchored + paired behavioral target + paired cost proxy:
   - `behavioral_quality(top-6) ≥ behavioral_quality(base) + 3pp` (target, per-variant base-anchor)
   - `behavioral_quality(top-6) ≥ behavioral_quality(all-layer) - ε` (inter-variant within-ε, non-inferiority; sub-kill)
   - `training_time(top-6) ≤ 0.70 × training_time(all-layer)` (cost proxy, paired to target)
3. **Cite** F#719 (sibling loss-variant target-paired template) + F#720 (sibling loss-variant pure-proxy KILL as counter-example) + F#683 (Hedgehog politeness PROVISIONAL, anchor) + F#133 (paired-KC template) + F#715 (infrastructure-benchmark bucket precedent).
4. Do NOT patch via `experiment update` (antipattern-u). File new experiment id.
5. **Transitive blocker**: F#682/F#683/F#684 `_impl` must land before v2. 26B teacher cache remains upstream prereq.

## mlx-lm version
Not invoked — preempt-structural KILL, no MLX load.
