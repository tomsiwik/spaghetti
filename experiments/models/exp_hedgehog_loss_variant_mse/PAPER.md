# PAPER.md — exp_hedgehog_loss_variant_mse

**Verdict:** KILLED (preempt-structural, pre-measurement)

**Fire mode:** TRIPLE — primary F#666-pure-standalone + secondary §5 tautological-inter-variant-delta + tertiary hygiene-multi-defect (F#702-patch unavailable). **3rd triple-fire precedent** (after F#714, F#716). Hedgehog loss-variant-ablation sub-type, **strictly weaker KC design than F#719 sibling**.

**Antipatterns fired:**
1. **F#666-pure-standalone** (primary, **13th drain-window instance**, **1st cos-sim-bucket instance** — opens 6th bucket) — sole KC K1872 is cos-sim (guardrail 1007 canonical proxy), zero target pairing, `depends_on=[]`.
2. **§5 tautological-inter-variant-delta** (secondary, **7th §5 instance**, **1st intra-loss-function-delta sub-variant**) — K1872 inter-variant `MSE vs cos-loss baseline` without per-variant base-anchor.
3. **Hygiene-multi-defect** (tertiary, 3 defects) — F#702 patch path **unavailable** per promoted standalone memory `mem-impossibility-f666pure-saturation-implies-f702-unavailable` (**4th confirmation** after F#714, F#715, F#716).

## Claim
K1872: MSE attention-map loss produces adapter with cos-sim < 0.70 vs cos-loss baseline. Third Hedgehog loss-variant ablation (after cos-loss, KL-div F#719). Tests whether magnitude matters or only direction when distilling Hedgehog attention patterns.

## Prediction-vs-measurement

| # | Prediction | Basis | Measured | Verdict |
|---|------------|-------|----------|---------|
| P1 | F#666 2-outcome truth table yields tautological-PASS or proxy-only-FAIL for K1872 | L1 (guardrail 1007 names cos-sim as proxy; behavioral↔cos-sim relation not established for Hedgehog attention-map comparisons) | Not measured (preempt) | Structural — both classes unidentifiable |
| P2 | §5 degenerate-equivalence (F#477 collapse regime candidate) trivially satisfies K1872 via cos-sim → 1.0 | L2 (F#477 K1226 adapted_acc 0.480 < 0.50 4-opt random; Hedgehog-framework PROVISIONAL without _impl measurement) | Not measured (preempt) | Structural — collapse regime |
| P3 | F#702 hygiene-patch path structurally unavailable per promoted impossibility memory | L3 (4th confirmation; F#714/F#715/F#716 precedent + F#716-promoted standalone memory) | Confirmed at MATH.md | Structural — post-promotion confirmation |
| P4 | Cos-sim bucket opens at 1-instance; not confirmed-recurrent; no taxonomy refactor | L4 (F#711 convention: bucket-level promotion requires ≥ 3 instances) | Confirmed at MATH.md | Taxonomy stable; 6 buckets total |
| P5 | Distinctness from F#669-family / template-regression / proxy-only-lineage / cross-paper-combined-loss / novel-mechanism+hygiene-pairing preserved — F#719 is sibling not parent | L5 | Confirmed at MATH.md | No cross-contamination; sibling-with-weaker-KC preserved |

## Summary
K1872 ("cos-sim MSE-adapter vs cos-loss-adapter < 0.70") is a single-proxy cos-sim comparison between two loss-variant Hedgehog adapters with no target-metric pair and no per-variant base-anchor. `depends_on=[]` — standalone. 3 hygiene defects (success_criteria, platform, references).

Per MATH.md L1, cos-sim is explicitly a proxy under guardrail 1007; both 2-outcome verdicts (PASS=tautological-support on a geometric similarity without behavioral anchor, FAIL=finding-about-the-proxy-not-a-kill) are unidentifiable. Per L2, §5 fires because K1872 compares two variants without anchoring either to `base`; under F#477 parent-target-FAIL regime candidate for Gemma 4 r=6 Hedgehog adapters (collapse basin pending any Hedgehog `_impl` measurement), both MSE and cos-loss adapters → near-identity Δ, cos-sim(MSE, cos-loss) → 1.0, trivially exceeds the 0.70 kill threshold (kill does not fire, "support" by collapse). This is the **1st intra-loss-function-delta sub-variant** of §5 — new sub-variant at 1-instance, deferred promotion per F#711 3-split convention. Per L3, hygiene-multi-defect is tertiary but F#702 patch path is **structurally unavailable** per the F#716-promoted standalone memory; this is the **4th confirmation** (F#714, F#715, F#716, this), post-promotion. Per L4, cos-sim bucket opens at 1-instance — **1st cos-sim-bucket F#666-pure-standalone instance**, adding a 6th bucket to the existing taxonomy (PPL/routing/classification-FNR/routing-match-rate/infrastructure-benchmark); not confirmed-recurrent. Per L5, topology is distinct from all upstream antipatterns including F#719 — F#719 is a **sibling** (both Hedgehog loss-variant-ablation sub-type) not a parent; the strictly-weaker KC design of this experiment vs F#719 does not constitute template-regression (which requires a formal `depends_on` edge).

**3rd triple-fire** (F#714 inter-training §5, F#716 intra-adapter-rank §5, **this intra-loss-function-delta §5**). Hierarchy (F#666-pure > §5 > hygiene-multi-defect) holds across a **4th distinct §5 axis**, confirming the triple-fire mode is recurrent and the hierarchy is axis-invariant. 3-instance reaches the F#711 conservative threshold for triple-fire-mode **bucket-level promotion candidacy** — flag for analyst (the triple-fire *mode* itself, separate from the individual antipattern counts).

## Hedgehog loss-variant-ablation sub-type ledger (updated)

| Instance | Experiment | Loss variant | Target-KC paired? | Verdict | Finding |
|---|---|---|---|---|---|
| 1 | `exp_hedgehog_loss_variant_kl_div` | KL-div (teacher‖student) | ✅ K1871 behavioral Δ>3pp | PROVISIONAL novel-mech + hygiene-patch | F#719 |
| 2 | **`exp_hedgehog_loss_variant_mse`** | MSE on attention weights | ❌ cos-sim only | **PREEMPT-KILL triple-fire** | (this) |

Observation: the same sub-type split on KC design (paired vs pure-proxy) cleanly bifurcates verdict (PROVISIONAL vs KILL) at the preempt gate, independent of loss-variant identity. Provides a clean pedagogical anchor: the KC design gate is **primary**, the loss-variant mechanism is **downstream** to whether the experiment is even admissible under F#666.

## Hard-defer interaction (analyst F#719 guidance)
Analyst F#719 advisory: "HARD-DEFER all further Hedgehog-framework design-locks (axis OR loss-variant) until ≥1 _impl lands; pile at 7 designs / 0 measurements." This preempt-KILL does **not** violate the hard-defer because:
- Preempt-KILL is a **rejection** of the design-lock, not an **acceptance**.
- The 7-design-lock pile (6 Hedgehog-axis + F#719 Hedgehog-loss-variant) remains at 7, unchanged.
- Hard-defer is advisory-for-researcher-claim-selection; the claim queue served this experiment and the correct response is preempt-KILL on independent F#666-pure grounds, not refusal-to-process.

## Assumptions logged (autonomy guardrail 1008)
- Cos-sim between attention maps is a proxy under guardrail 1007 regardless of directional variation (pre-softmax logits vs post-softmax probabilities vs intermediate activations); all interpretations fail L1.
- "MSE attention-map loss" reads as MSE over softmax'd attention probabilities (Hedgehog architectural output per F#683); §5 collapse argument holds under either pre-softmax or post-softmax interpretation.
- F#477 collapse-regime applicability to Hedgehog MSE adapters is the default pending any Hedgehog `_impl` measurement; if an `_impl` lands and shows behavioral departure, assumption P2 may need revision but does not affect preempt-time structural reasoning.
- Platform field `~` counted as 1 hygiene defect per F#703 canon.
- `references = []` marked as hygiene defect for counting; non-blocking for preempt-KILL verdict per F#716 precedent (the kill stands regardless of `experiment ref-add` availability).
- Sibling relation with F#719 (Hedgehog loss-variant-ablation sub-type) is semantic not structural — no `depends_on` edge in DB; NOT classified as template-regression (which requires formal parent-strip).

## Unblock path
File `exp_hedgehog_loss_variant_mse_v2_target_paired`:
1. **Scope** to Hedgehog politeness-axis (reuse F#683 corpus + rubric) — same transitive-blocker pattern as F#719.
2. **Metric** behavioral-quality judge Δ > 3 pp (F#683/F#719 K1871 template) as target-metric KC; cos-sim retained as diagnostic proxy ("did MSE arm converge?").
3. **KC design**:
   - `behavioral_quality_MSE ≥ behavioral_quality_base + 3pp` (target, per-variant base-anchor)
   - `behavioral_quality_MSE ≈ behavioral_quality_cos_loss` (inter-variant target delta, within-ε)
   - `cos-sim(MSE, cos_loss) ≥ 0.70` (secondary proxy sanity)
4. **Cite** F#719 (sibling template, direct precedent) + F#683 (Hedgehog politeness, PROVISIONAL) + F#133 (paired-KC template).
5. Do NOT patch via `experiment update` (antipattern-u). File new experiment id.
6. **Transitive blocker**: F#683 or F#684 `_impl` must land before v2 (same as F#719 _impl).

## mlx-lm version
Not invoked — preempt-structural KILL, no MLX load.
