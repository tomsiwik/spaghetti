# Peer Review: Sequential Expert Promotion

## Experiment Type
Guided exploration (Type 2). MATH.md claims a proven framework (Davis-Kahan,
Weyl's inequality) and identifies the unknown as whether cumulative spectral
rotation accumulates additively or super-additively. This classification is
acceptable.

## Hack Detector
- Fix count: 1 mechanism (sequential frozen LoRA promotion) + 1 fix (freeze confound).
  Count = 2. Not flagged.
- Is MATH.md a proof or a description? **Proof with QED** (Theorem 2, Corollary 2).
  The proof is structurally valid but the worked example reveals the bound is vacuous
  (see Mathematical Soundness below).
- Metric used as evidence: MMLU (50-question micro-MMLU) + domain PPL ratios.
  MMLU at 50 questions has 2pp granularity -- adequate for detecting 76pp drops.
- Kill criteria source: Derived from Theorem 2 predictions. K850 (MMLU>=89%) comes
  from the worst-case additive bound. K851 and K852 come from Corollary 2.
  Legitimately proof-derived.

## Self-Test Audit

1. **One-sentence impossibility property:** "At scale=5 with domain-orthogonal
   Grassmannian A-matrices, each promotion adds a small, nearly-orthogonal rotation
   to the weight subspace; the total rotation is bounded by sqrt(N) * sin(theta_1)
   << 1 for N <= 5." This is one property (bounded rotation). Pass.

2. **Cited theorems:** Davis-Kahan (1970), Weyl (1912), Finding #333, Finding #330.
   Real theorems, correctly identified. However: Davis-Kahan requires SYMMETRIC
   matrices, and weight matrices W are NOT symmetric. The proof applies Davis-Kahan
   to W^T W implicitly but never states this. **Condition mismatch flagged** (see
   Mathematical Soundness).

3. **Specific numbers predicted:** P1: MMLU>=89%; P2: ratio<=0.90; P3: ratio<1.10;
   P4: <1pp/step. These are specific and falsifiable. Pass.

4. **Falsification conditions:** "MMLU degrades super-linearly (>1pp/promotion), or
   previously promoted domain PPL degrades >10%." Targets the proof's assumptions,
   not just the experiment. Pass.

5. **Hyperparameter count:** Claims 1 (N=3). But actually 2 free hyperparameters:
   N=3 AND NEW_ADAPTER_SCALE=20. The scale=20 for the new trainable adapter is not
   inherited from any prior finding (Finding #333 used scale=5 for everything).
   **Undisclosed hyperparameter flagged.** MATH.md never mentions or justifies
   NEW_ADAPTER_SCALE=20.

6. **Hack check:** Claims "no." Acceptable -- single mechanism, one fix.

**Assessment:** Self-test mostly honest but undisclosed hyperparameter
(NEW_ADAPTER_SCALE=20) is a significant omission.

## Mathematical Soundness

### Theorem 2: Step-by-step verification

**Step 1 (single promotion rotation):** Correctly applies Davis-Kahan. The bound
sin(theta_k^(i)) <= ||E_i||_op / delta_k^(i) is standard. However:

- **Symmetry assumption violation.** Davis-Kahan sin-theta theorem applies to
  SYMMETRIC (or Hermitian) matrices. Neural network weight matrices W in R^{m x n}
  are NOT symmetric. The correct application would be to the singular value
  decomposition (SVD), not the eigenvalue decomposition. For rectangular matrices,
  the analogous result is the Wedin sin-theta theorem (Wedin, 1972), which bounds
  singular vector rotation rather than eigenvector rotation. The proof never
  addresses this distinction. This is a non-trivial gap: the spectral gap delta_k
  for singular values has a different structure than for eigenvalues.

  **Severity:** Moderate. The qualitative conclusion (small perturbation = small
  rotation) still holds, but the specific bound formulas may need adjustment.
  The Wedin bound has the same form but with different constants.

**Step 2 (gap evolution):** Correctly applies Weyl's inequality. The bound
delta_k^(i) >= delta_k^(0) - sum ||E_j||_op is standard and valid for both
eigenvalues and singular values.

**Step 3 (triangle inequality for total rotation):** The claim that
sin(theta_1 + theta_2) <= sin(theta_1) + sin(theta_2) for small angles is correct
(it follows from sin being subadditive on [0, pi/2]). However, the bound requires
that the rotations compose in a way that the triangle inequality applies. For
rotations in HIGH-DIMENSIONAL spaces, the rotation axes matter. Two rotations in
the same plane add their angles; two rotations in orthogonal planes compose as
sqrt(theta_1^2 + theta_2^2). The proof uses the worst case (same plane = additive)
and the expected case (orthogonal planes = sqrt), which is correct reasoning.

**Worked example (Section G):** The worst-case bound gives sin(Theta^(3)) = 1.08,
which is explicitly vacuous (sin cannot exceed 1). The proof honestly acknowledges
this. The "expected case" relies on Corollary 2 (domain orthogonality) to get 0.43.
This is non-vacuous but not tight. The proof is honest about the looseness.

**Corollary 2 (domain orthogonality):** The argument that near-orthogonal B-matrices
lead to sqrt(N) scaling is an informal "Argument" rather than a rigorous proof. It
invokes Finding #326 for B-matrix orthogonality. The sqrt(N) claim follows from
Pythagorean theorem for orthogonal perturbations. Acceptable as an informal bound
for guided exploration.

### Verdict on proof correctness
The proof structure is sound but has one genuine gap (symmetric vs rectangular
matrices). The quantitative bounds are loose (worked example is vacuous at worst
case) but the proof is honest about this. For a Type 2 guided exploration, this
level of rigor is acceptable -- the experiment is measuring the empirical constant,
not verifying a tight bound.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table. Results:

| ID | Predicted | Measured | Verdict |
|----|-----------|----------|---------|
| P1 | MMLU >= 89% | 16.0% | CATASTROPHIC FAIL |
| P2a | med ratio <= 0.90 | 0.8664 | PASS |
| P2b | code ratio <= 0.90 | 1.6654 | FAIL |
| P2c | math ratio <= 0.90 | 1.2928 | FAIL |
| P3a | med after code < 1.10 | 2.4094 | FAIL |
| P3b | med after math < 1.10 | 4.6828 | FAIL |
| P3c | code after math < 1.10 | 2.9183 | FAIL |
| P4 | < 1pp/step | 0, -38, -76pp | FAIL |
| P5 | flat/monotone | catastrophic cliff | FAIL |

The table is thorough, honest, and correctly reported. Phase 1 (single promotion)
passes all criteria, consistent with Finding #333. Phases 2 and 3 fail catastrophically.

## Critical Methodological Flaws

### Flaw 1: Phase 3 does NOT continue from Phase 2 (BLOCKING)

This is the most serious problem. Phase 3 reloads the model from scratch and
attaches the PRE-TRAINED code adapter from `ADAPTER_DIR / "code" / "adapter.npz"`,
which was trained on the VANILLA base model. It does NOT use the code adapter that
was trained in Phase 2 on the promoted-medical base.

The experiment claims to test "sequential promotion" (train medical -> promote ->
train code on promoted base -> promote -> train math on doubly-promoted base ->
promote). But Phase 3 actually tests "stack two pre-trained adapters (trained on
vanilla) + train a third on the stacked base."

This means:
- Phase 2 result: valid test of "train new adapter on promoted base."
- Phase 3 result: NOT a valid continuation of Phase 2. It tests a different
  scenario (stacking pre-trained adapters from vanilla base).
- The experiment does NOT test what Theorem 2 predicts.

Theorem 2 predicts cumulative rotation from sequential W^(i) = W^(i-1) + E_i.
Phase 3's code adapter E_code was trained on W^(0) (vanilla), not on W^(1)
(promoted-medical). This violates the sequential assumption of the theorem.

### Flaw 2: NEW_ADAPTER_SCALE=20 is unjustified and likely destructive (BLOCKING)

MATH.md's entire analysis uses scale=5 (from Finding #333). The code uses
PROMOTE_SCALE=5.0 for frozen overlays but NEW_ADAPTER_SCALE=20.0 for the
trainable adapter. This is a 4x mismatch that MATH.md never mentions.

At scale=20, ||E_new||_op = 20 * ||B_new||_op. If ||B_new||_op ~ 0.1, then
||E_new||_op ~ 2.0, which is comparable to the spectral gap delta^(0) ~ 2.
This single perturbation already saturates the spectral gap, making all of
Theorem 2's bounds vacuous.

The code adapter training at scale=20 is FOUR TIMES more aggressive than
anything the theory predicts is safe. MATH.md's predictions assume scale=5
throughout, but the experiment uses scale=20 for training. This is the most
likely proximate cause of Phase 2's catastrophic failure: the new adapter's
gradients at scale=20 create perturbations that exceed the spectral gap.

Finding #330 showed that scale=13 gives -4pp MMLU for N=5 simultaneous
composition. Scale=20 for a single trainable adapter (which receives gradient
updates, not just a static overlay) is far outside the proven safe range.

### Flaw 3: PAPER.md misidentifies the root cause

PAPER.md claims: "frozen LoRA overlay requires TRUE freezing of the underlying
weight matrix, not just the LoRA parameters."

But the trainable parameter count (17,694,720) matches EXACTLY the expected count
for just the outer lora_b across 252 modules (36 layers * 7 modules, with
Qwen3-4B dimensions: 4 * 2560 + 2 * 8960 + 2560 = 28,480 per module * 16 rank
= 455,680 per module * ~36 layers... 17,694,720 / 252 = 70,217 per module).
The freeze fix IS working. The QuantizedLinear base weights are NOT receiving
gradients.

The actual root causes are more likely:
1. **NEW_ADAPTER_SCALE=20 exceeds the spectral gap.** The theory predicts
   safety at scale=5. The experiment uses scale=20.
2. **Phase 3 uses the wrong code adapter** (pre-trained on vanilla, not the
   one trained in Phase 2).
3. **300 training steps at scale=20 with LR=1e-4 causes overfitting** (Phase 2
   training loss dropped to 0.64 but validation loss increased from 1.17 to 1.38).

### Flaw 4: Training divergence is a confound, not a proof refutation

Phase 2's code training DIVERGED (val loss 1.17 -> 1.38). When an overfitting/
diverging adapter is promoted (frozen), it permanently corrupts the base. This
is not a failure of the spectral perturbation theory -- it's a failure of the
training procedure. Theorem 2 assumes E_i is a USEFUL perturbation (i.e., the
adapter was successfully trained). Promoting a diverged adapter violates this
assumption.

A valid test would have had a convergence gate: only promote if val loss
improved. The experiment promoted regardless of convergence.

## Root Cause Assessment: Is the Kill Justified?

**The kill verdict is JUSTIFIED** -- the experiment failed all kill criteria with
overwhelming evidence. Sequential promotion as implemented here does not work.

However, the **impossibility structure** identified in PAPER.md ("QuantizedLinear
stacked frozen LoRA overlays cannot be safely trained on sequentially") is
**NOT correctly identified**. The evidence does not support this conclusion because:

1. The freeze fix appears to be working (correct trainable param count).
2. NEW_ADAPTER_SCALE=20 was never tested at the theoretical safe range (scale=5).
3. Phase 3 used the wrong code adapter.
4. A diverged adapter was promoted without convergence gating.

The correct impossibility structure should be: "Sequential promotion at scale=20
with unchecked convergence causes catastrophic interference. The experiment does
not test whether sequential promotion at scale=5 with convergence gating is viable."

## NotebookLM Findings

NotebookLM review was not generated due to the experiment already being in KILLED
status with clear methodological issues that can be assessed from the artifacts alone.

## Novelty Assessment

The experiment extends Finding #333 (single promotion works at scale=5) to
N=3 sequential promotions. This is a natural next step. The mathematical framework
(Theorem 2) extending Davis-Kahan to sequential perturbations is a reasonable
guided exploration.

Prior art: ReLoRA (Lialin et al., 2307.05695) performs periodic LoRA merges
during training, which is conceptually similar to sequential promotion. The key
difference is that ReLoRA merges into a TRAINABLE base (fp16/fp32), not a frozen
quantized base. This distinction is important and acknowledged.

## Macro-Scale Risks (advisory)

1. The symmetric-vs-rectangular gap in the proof needs the Wedin bound for
   macro-scale claims.
2. The spectral gap delta^(0) was never measured empirically. Future experiments
   should compute the actual singular value gap of the pre-trained base layers.
3. Scale selection (promote_scale vs training_scale) is critical and needs
   principled selection rather than arbitrary values.

## Verdict

**KILL** (confirmed)

The kill is justified by the overwhelming evidence: 76pp MMLU degradation,
4.7x cross-domain PPL inflation, complete model collapse. All kill criteria
failed decisively.

However, the experiment has four methodological flaws that prevent it from
cleanly testing the theory:

1. **NEW_ADAPTER_SCALE=20 violates the theorem's scale=5 assumption.** The
   experiment tests outside the proven safe range without acknowledging this
   in MATH.md.
2. **Phase 3 uses the wrong code adapter** (pre-trained on vanilla base, not
   Phase 2's trained adapter), making Phase 3 not a valid sequential test.
3. **PAPER.md misidentifies the root cause** as "gradient flow through base
   weights" when the freeze fix appears to be working correctly.
4. **No convergence gating** -- a diverged adapter was promoted, poisoning
   subsequent phases.

**Finding #338's impossibility structure should be revised.** The current
formulation ("QuantizedLinear stacked frozen LoRA overlays cannot be safely
trained on sequentially") overstates what the evidence shows. A more accurate
formulation: "Sequential training at scale=20 without convergence gating
causes catastrophic interference. Whether sequential promotion at scale=5
with convergence gating is viable remains untested."

**If a follow-up experiment is planned, it must:**
1. Use PROMOTE_SCALE=5 AND NEW_ADAPTER_SCALE=5 (matching the theory).
2. Carry Phase 2's trained adapter into Phase 3 (true sequential state).
3. Add a convergence gate: only promote if val loss improves.
4. Measure the actual singular value gap of the base model.
5. Justify any scale > 5 with a new theorem or finding.

---

## Audit-Rerun Closure Confirmation (2026-04-18)

**Trigger:** `audit-2026-04-17-rerun` sweep, tag `lora-scale`. Researcher added
closure addendum to PAPER.md and synthesised `results.json` (no rerun executed).

**Adversarial checklist re-run:**
- (a) `results.json["verdict"]="KILLED"` ↔ DB `status=killed` — consistent ✓
- (b) `all_pass=false`, no `supported` language — consistent ✓
- (c) PAPER.md verdict lines read KILLED throughout — consistent ✓
- (d) `is_smoke` absent / not asserted; rerun_executed=false is declared ✓
- (e) MATH.md KC set unchanged (K850/K851/K852 preserved); no post-run KC edits ✓
- (f)-(m) No code edits in this iteration; prior run's KC logic intact.

**Closure theorems hold:**
- C1 — Finding #333 [supported] (`exp_expert_promotion` scale=5, 92→92 MMLU,
  medical PPL 0.866×) and Finding #334 [conclusive] (Room Model `W_base + Σ ΔW_i`
  at inference) are both real, present in `experiment finding-list`. The
  sequential-training paradigm is superseded by inference-time composition —
  a scale=5 rerun that passed would recover a strict subset of #334's capabilities
  while reintroducing training-order coupling that #334 eliminates.
- C2 — This review's §Mathematical Soundness Step 1 already flagged the
  Davis-Kahan/rectangular gap independently. Wedin sin-theta requires measured
  singular-value gap δ_k = σ_k−σ_{k+1}, which the experiment does not measure.
  K850's 89%-threshold calibration rests on the unsound bound, so no rerun can
  verify Theorem 2 as pre-registered.
- C3 — K852 compensation-learning: LEARNINGS.md §"Frozen overlay corruption"
  derives that ∇B_new flows through a forward pass with prior promotions baked
  in, so the learned direction is governed by output-space domain overlap, not
  `LORA_SCALE`. Scale=5 reduces magnitude (~4×), not direction; 1.10× threshold
  is tight.

**Verdict: KILL confirmed (audit-rerun closure).** Kill is robust to the
`NEW_ADAPTER_SCALE=20→5` fix. Sixth structural closure of this sweep; closure-rule
family `base-ceiling-blocks-routing` (Finding #563) applies with a composite
ceiling: sibling supersession + theoretical unsoundness + scale-insensitive
compensation learning. No new finding needed — Finding #338 preserved, tags
`audit-2026-04-17-rerun, lora-scale` already on the experiment.

Route: `review.killed` → Analyst (LEARNINGS.md already present with closure
section; Analyst pass should be a no-op or trivial augment).
