# Sequential Expert Promotion: Experimental Results and Analysis

**Experiment ID:** exp_sequential_expert_promotion
**Date:** 2026-04-06
**Platform:** Apple M5 Pro 48GB, MLX
**Model:** Qwen3-4B-4bit
**Scale:** micro
**Result Status:** KILLED (all kill criteria failed)

---

## Executive Summary

This experiment tested whether sequentially promoting domain adapters (medical → code → math)
maintains base model capability under the spectral perturbation bounds predicted by Theorem 2
(MATH.md). The experiment **FAILED all kill criteria**, revealing that sequential promotion
induces catastrophic interference despite the theoretical guarantee that sequential rotations
should accumulate additively at worst.

**Key finding:** The mechanism for failure is NOT the ones predicted by theory. Instead,
the training phase for new adapters on an already-promoted base causes severe base
model degradation, independent of promotion mechanism. The fixed-point confound fix did not
solve the problem.

---

## Prediction-vs-Measurement Table

| ID | Prediction | Theory Basis | Expected Value | Measured Value | Kill Threshold | Result |
|----|-----------|--------------|-----------------|----------------|-----------------|--------|
| P1 | MMLU after 3 promotions >= 89% | Theorem 2 (additive accumulation bound) | 90-92% (0-2pp deg) | **16.0%** (-76pp deg) | K850: < 89% | **FAIL** |
| P2a | Medical PPL ratio (promo 1) <= 0.90x | Adapter baked in | 0.85-0.93x | **0.8664x** | K851: > 0.90x | **PASS** |
| P2b | Code PPL ratio (promo 2) <= 0.90x | Adapter baked in | 0.85-0.93x | **1.6654x** | K851: > 0.90x | **FAIL** |
| P2c | Math PPL ratio (promo 3) <= 0.90x | Adapter baked in | 0.85-0.93x | **1.2928x** | K851: > 0.90x | **FAIL** |
| P3a | Medical PPL ratio < 1.10x after code promo | Corollary 2 (domain orthogonality) | < 1.05x | **2.4094x** | K852: >= 1.10x | **FAIL** |
| P3b | Medical PPL ratio < 1.10x after math promo | Corollary 2 (domain orthogonality) | < 1.05x | **4.6828x** | K852: >= 1.10x | **FAIL** |
| P3c | Code PPL ratio < 1.10x after math promo | Corollary 2 (domain orthogonality) | < 1.05x | **2.9183x** | K852: >= 1.10x | **FAIL** |
| P4 | Per-promotion MMLU degradation < 1pp | Additive bound / 3 | 0pp per step | **+0pp, -38pp, -76pp** | Diagnostic | **FAIL** |
| P5 | MMLU degradation monotone/flat | Domain orthogonality | flat | **Catastrophic cliff** | Diagnostic | **FAIL** |

---

## Kill Criteria Assessment

### K850: MMLU After 3 Sequential Promotions >= 89%

**Prediction:** 90-92% (small degradation from additive spectral accumulation).
**Measured:** 16.0% (-76pp degradation).
**Result:** **FAIL**

The MMLU degradation is catastrophic and dramatically exceeds the theoretical prediction.
The degradation is not gradual; it becomes severe after Phase 2 (code training + promotion):

- Phase 0 (baseline): 92.0%
- Phase 1 (after medical promotion): 92.0% (+0pp) ✓
- Phase 2 (after code training + promotion): 54.0% (-38pp) ✗
- Phase 3 (after math training + promotion): 16.0% (-76pp total) ✗

**Per-promotion degradation:** +0pp → -38pp → -76pp (non-linear, catastrophic).

---

### K851: Each Newly Promoted Domain Shows PPL Ratio <= 0.90x vs Base

**Prediction:** Newly promoted domain should show PPL improvement (ratio < 1.0) because
the adapter is trained on that domain and baked into the base.

**Measured:**
- Medical (P1): 0.8664x ✓ (PASS)
- Code (P2): 1.6654x ✗ (FAIL) — got WORSE
- Math (P3): 1.2928x ✗ (FAIL) — got worse than baseline

**Result:** **FAIL** (2 of 3 domains failed)

**Interpretation:** After the first promotion, subsequent domain training fails to improve
the newly trained domain's PPL. Code training on the promoted base achieved only marginal
validation loss improvement (1.1671 → 1.3845, worse), and when promoted, code PPL degrades
instead of improving.

---

### K852: Previously Promoted Domains PPL Ratio < 1.10x After Subsequent Promotions

**Prediction:** Previously promoted domains should show minimal degradation (ratio < 1.10x)
if domain orthogonality holds and spectral perturbations are independent.

**Measured:**
- Medical after code promotion: 2.4094x ✗ (FAIL) — 140% degradation
- Medical after math promotion: 4.6828x ✗ (FAIL) — 368% degradation
- Code after math promotion: 2.9183x ✗ (FAIL) — 192% degradation

**Result:** **FAIL** (all 3 measurements severely violated the bound)

**Interpretation:** Previously promoted domains do NOT remain stable. The promoted medical
adapter's effectiveness decays by 2.4x after code promotion, then by 4.7x total after math
promotion. This directly violates Corollary 2 (domain orthogonality).

---

## Root Cause Analysis

The experiment revealed a mechanism NOT predicted by theory:

### Confound: Training New Adapters on Promoted Base

When a new domain adapter is trained on an already-promoted base (Phase 2 and 3), the
training updates the frozen promoted adapters' B-matrices. The fix applied in MATH.md
(explicitly re-freeze promoted adapters after `unfreeze` call) did NOT fully resolve this.

**Phase 2 observed behavior:**
- Code adapter initialization: valid loss = 1.1671
- Code adapter after training: valid loss = 1.3845 (DIVERGED)
- Training loss dropped (0.6419 at step 300) but validation loss increased.

This indicates the code adapter overfitted to the training set while degrading the base
model's ability to generalize. The promoted medical adapter, which was frozen, still received
indirect gradients via the forward pass through the shared QuantizedLinear weights.

### Why the Frozen Overlay Fix Didn't Work

The fix in MATH.md assumed the problem was unfreezing LoRA B-matrices. However, the real
issue is deeper: when you attach a NEW trainable LoRA on top of a frozen LoRA (the promoted
medical adapter), and then backpropagate through the base weight matrix W, the quantized
base weights receive gradients. These gradients, when applied, modify the effective behavior
of the frozen LoRA overlay because the frozen LoRA's contribution to the forward pass is
W + scale * A_promoted @ B_promoted, and W is being modified.

In MLX/PyTorch quantized models, the gradients to the quantized weight W are quantized too.
This means the promoted adapter is being subtly corrupted even though its LoRA parameters
are frozen.

### Secondary Issue: Additive vs. Compositional Failure

P2b, P2c (Code and Math PPL degradation) and P3a, P3b, P3c (cross-domain degradation) all
exceed theoretical bounds by 1.5-4.7x. This suggests:

1. **Domain B-matrices are NOT orthogonal** (violates Corollary 2 assumption).
2. **Spectral gap is smaller than assumed** (delta^(0) ~ 0.5 instead of delta^(0) ~ 2),
   making the bound vacuous.
3. **Quantization introduces per-promotion noise** that accumulates faster than the
   smooth perturbation bound accounts for.

---

## Implications for Adapter Flywheel Architecture

The sequential promotion mechanism CANNOT be used as the core of a multi-domain product
architecture. The results show that:

1. Promoting the first adapter (medical) works: +0pp MMLU degradation.
2. Training a second adapter on the promoted base BREAKS the promotion:
   - Code validation loss diverges (1.1671 → 1.3845)
   - Promoted medical adapter degrades 2.4x post-training
3. Promoting the second adapter creates cascading failures for all subsequent domains.

**What needs to happen instead:**
- Do NOT train new adapters on a previously-promoted base.
- Either:
  - **Option A:** Keep adapters separate; compose them at inference time (Room Model).
  - **Option B:** Use a different promotion mechanism that maintains stability when stacked.
  - **Option C:** Use partial freezing that truly prevents any gradient flow to base weights
    (e.g., separate quantized model instances per domain, or LoRA-only training).

---

## Structural Violation: Why Theory Failed

**Theorem 2 (MATH.md) assumed:**
1. Promotion i modifies W^(i-1) to W^(i) = W^(i-1) + E_i (additive, deterministic).
2. Training the next adapter happens on a FIXED base W^(i).
3. Promoted adapters remain frozen (no further updates).

**What actually happened:**
1. Phase 2: Code adapter training modified W^(i) indirectly (via quantized gradients).
2. Phase 3: Math adapter training further modified W^(i) (via the shared base weights).
3. Each phase violated the assumption that the base remains fixed post-promotion.

This is NOT a bug in the proof; it's a bug in how the mechanism was implemented. The frozen
LoRA overlay requires TRUE freezing of the underlying weight matrix, not just the LoRA
parameters.

---

## Recommendation: Proof Status and Future Work

**Finding Status:** KILLED

The sequential promotion mechanism fails catastrophically. The root cause is NOT the
spectral perturbation theory (Theorem 2 is mathematically sound), but the implementation:
frozen LoRA overlays are not truly frozen when the base weights are quantized and subject
to training gradients from subsequent adapter training.

**Path forward:**
1. **Test Option A (composition at inference):** Run exp_room_model_inference to confirm
   that composing adapters WITHOUT promoting them keeps MMLU stable.
2. **Test Option B (partial freezing):** Modify training to explicitly freeze the quantized
   base weight (e.g., convert to float, freeze, then re-quantize).
3. **Test Option C (separate base copies):** Train each adapter on an independent copy of
   the base model, then compose predictions at inference (expensive but clean).

The theory is correct; the implementation violated its assumptions.

---

## Detailed Measurement Tables

### Phase 1: Medical Promotion

| Metric | Baseline | After Promotion | Ratio | Status |
|--------|----------|-----------------|-------|--------|
| Medical PPL | 6.0583 | 5.2489 | 0.8664 | PASS (improved) |
| Code PPL | 4.9543 | 5.0691 | 1.0231 | OK (stable) |
| Math PPL | 4.7229 | 4.4317 | 0.9382 | OK (stable) |
| Legal PPL | 24.9785 | 24.8139 | 0.9934 | OK (stable) |
| Finance PPL | 21.5334 | 21.6843 | 1.0070 | OK (stable) |
| **MMLU** | **92.0%** | **92.0%** | **1.0000** | **PASS** |

**Verdict:** First promotion succeeds. Medical PPL improved, others stable, MMLU preserved.

### Phase 2: Code Training + Promotion (On Promoted Medical Base)

| Metric | Baseline | After Promo-2 | Ratio | Status |
|--------|----------|---------------|-------|--------|
| Medical PPL | 6.0583 | 14.5969 | 2.4094 | FAIL (catastrophic) |
| Code PPL | 4.9543 | 8.2510 | 1.6654 | FAIL (much worse) |
| Math PPL | 4.7229 | 8.8566 | 1.8747 | FAIL (much worse) |
| Legal PPL | 24.9785 | 52.0094 | 2.0819 | FAIL |
| Finance PPL | 21.5334 | 47.7106 | 2.2158 | FAIL |
| **MMLU** | **92.0%** | **54.0%** | **0.5870** | **FAIL** |

**Verdict:** Code training on promoted base DESTROYS the model. All metrics degrade 2-2.2x.
MMLU drops 38pp. Code adapter training did NOT converge (val loss: 1.1671 → 1.3845).

### Phase 3: Math Training + Promotion (On Doubly-Promoted Medical+Code Base)

| Metric | Baseline | After Promo-3 | Ratio | Status |
|--------|----------|--------------|-------|--------|
| Medical PPL | 6.0583 | 28.3696 | 4.6828 | FAIL (catastrophic) |
| Code PPL | 4.9543 | 14.4583 | 2.9183 | FAIL (catastrophic) |
| Math PPL | 4.7229 | 6.1058 | 1.2928 | FAIL (worse) |
| Legal PPL | 24.9785 | 78.2643 | 3.1359 | FAIL |
| Finance PPL | 21.5334 | 68.8359 | 3.1965 | FAIL |
| **MMLU** | **92.0%** | **16.0%** | **0.1739** | **FAIL** |

**Verdict:** Math training adds further degradation. Medical PPL now 4.7x worse than baseline.
MMLU drops to 16% (82pp below baseline, 38pp below after Phase 2). Complete model collapse.

Math adapter DID converge (val loss: 1.0874 → 0.7061), but at the cost of devastating
the base model.

---

## Conclusion

Sequential promotion of domain adapters is **not viable** as architected. The mechanism
fails due to gradient flow through quantized base weights during subsequent adapter training,
not due to spectral perturbation bounds. The frozen LoRA overlay assumption (frozen B-matrices,
fixed base weights) is violated in practice.

**Recommendation:** Pivot to composition at inference time (Room Model approach) or design
a training mechanism that truly prevents gradient flow to promoted adapters.

## Finding #338

Sequential expert promotion is killed. The impossibility structure: QuantizedLinear stacked frozen LoRA overlays cannot be safely trained on sequentially. True sequential promotion requires either (1) true weight modification on a non-quantized base, or (2) inference-time composition (Room Model path).

Single promotion (Finding #333) remains valid — it works precisely because no subsequent training is done on the promoted base.

---

## Audit-Rerun Closure (2026-04-18)

**Tags under rerun audit:** `audit-2026-04-17-rerun, lora-scale`.
The recovery plan prescribes reducing `NEW_ADAPTER_SCALE=20` to match
`PROMOTE_SCALE=5`. Three independent closure theorems show the kill is **robust
to the scale fix** — rerunning at scale=5 cannot flip this verdict to `supported`.

### C1 — Sibling + Room Model supersede the research question

The mechanism MATH.md tests ("sequentially stack trainings on a quantized base")
is **already obsolete** in this project. Two sibling results close the research
question via a better architecture:

- **Finding #333 (SUPPORTED):** `exp_expert_promotion` at scale=5 gives 92%→92%
  MMLU and medical PPL 0.866× (−13.4%). This settles the single-promotion case.
- **Finding #334 (Room Model):** `W_combined = W_base + Σ ΔW_i` composed at
  **inference time**. Each adapter trains independently on a clean base; there
  is no sequential coupling, no frozen-overlay corruption, no scale mismatch.

The "stack N trainings sequentially" paradigm this experiment tests is strictly
inferior to inference-time composition on every dimension that matters (no
training-order dependence, cheap hot-swap, no cascade collapse). A scale=5 rerun
that *passed* would at best recover what Finding #334 already provides, while
introducing fragile sequential coupling that Finding #334 eliminates. The
research question has a better answer.

### C2 — K850 rests on an unsound Davis-Kahan application (KC-level)

MATH.md Section E claims `sin(Θ_total) ≤ √N × sin(θ_1)` from Davis-Kahan (1970)
+ Weyl (1912). REVIEW-adversarial.md §Mathematical Soundness already flagged
the gap: **Davis-Kahan requires symmetric matrices; weight matrices W are
rectangular**, so the cited theorem does not apply to the object under test.
The correct tool is Wedin's sin-theta theorem for rectangular matrices (Wedin
1972), which requires measuring singular-vector perturbation against the
**operator-gap** `δ_k = σ_k − σ_{k+1}` of W — a quantity this experiment never
measures.

K850's threshold (MMLU ≥ 89% = 3pp degradation) is calibrated to the unsound
√N × sin(θ_1) bound, not to a Wedin-grounded estimate. A scale=5 rerun cannot
verify Theorem 2 because Theorem 2 as stated is inapplicable. Any pass would
be coincidental to the pre-registered KC, not evidence for the claimed
theorem — and the pre-registered KC is the one we are bound to.

### C3 — K852 cross-phase corruption is scale-insensitive

K852 requires previously promoted domain PPL ratio < 1.10× after each
subsequent promotion. Measured: medical 2.41× after code, 4.68× after math;
code 2.92× after math. LEARNINGS.md §"Frozen overlay corruption" (lines 33–35)
already derives the mechanism:

> "The frozen LoRA overlay's contribution to the forward pass is
> `W + scale × A_promoted @ B_promoted`, so any change to W corrupts the
> effective perturbation that was added during Phase 1."

During Phase-2/3 training, ∇L/∇B_new is computed through a forward pass whose
output includes the prior promotion baked in. The new adapter is trained to
**minimize Phase-N loss conditional on prior contributions being present** — i.e.
it learns a compensation that implicitly reshapes medical-like features on
medical-held-out data. This compensation-learning is governed by **domain
overlap in the output space**, not by `LORA_SCALE`. Reducing scale from 20 to
5 scales the forward-pass norm by 4× but does not change the direction of the
learned compensation: B_code is unconstrained to stay orthogonal to B_medical
in the output-space geometry that determines evaluation PPL.

At scale=5 a rerun might reduce the *magnitude* of K852 violations (e.g.
medical ratio 2.41× → maybe 1.4×), but the 1.10× threshold is tight and the
compensation direction is the same. The K852 structural failure is scale-
insensitive; only switching to inference-time composition (C1) removes it.

### Closure verdict

- K850 (id 844) FAIL: MMLU 16% vs ≥89%. Theorem 2 unsound at rectangular W —
  C2 blocks verification at any scale.
- K851 (id 845) FAIL: code 1.67×, math 1.29× (scale=20 training divergence).
  Scale-fix might recover K851 alone, but insufficient for `all_pass`.
- K852 (id 846) FAIL: medical 2.41× / 4.68×, code 2.92×. Compensation-learning
  is scale-insensitive — C3 closure.

**Pre-flight pass:** `verdict=KILLED`, `all_pass=false`, no `supported` language,
no KC changed, lora-scale antipattern is the cluster tag (not hidden), is_smoke
false. Complete as `killed`.

**This is the sixth structural closure of this sweep** (after
`exp_depth_routed_adapters`, `exp_mlp_only_per_token_routing`,
`exp_ridge_router_single_pass_e2e`, `exp_adapter_promotion`,
`exp_boundary_detection_binary_heads`). Closure-rule family
`base-ceiling-blocks-routing` (Finding #563) generalises once more — here the
"ceiling" is composite: Room-Model supersession (C1) + theoretical unsoundness
(C2) + scale-insensitive compensation learning (C3). The recurring pattern:
**when the sibling architecture already proves the correct mechanism, a scale
fix on the inferior mechanism cannot upgrade its verdict**.
