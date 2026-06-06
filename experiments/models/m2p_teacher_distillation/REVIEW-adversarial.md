# Peer Review: M2P Teacher Distillation

## Experiment Type
Guided exploration (Type 2)

## Hack Detector
- Fix count: 1 (KL distillation replaces NTP loss — clean substitution, no stacking)
- Is MATH.md a proof or a description? **Mixed.** Theorem 1 has a Proof/QED block but the proof is largely a restatement of KL properties, not a novel derivation. Theorem 2 is a proof sketch, not a full proof (acknowledged as such). The no-regression guarantee is a straightforward consequence of KL non-negativity — this is a known result, not new theory.
- Metric used as evidence: quality_gap_closure (ratio of NTP loss improvements). This is a well-defined proxy for behavioral outcome (domain specialization quality).
- Kill criteria source: K853 derived from proof Corollary 2 (gap closure). K854 from architectural feasibility. Both are properly motivated.

## Self-Test Audit

1. **One-sentence impossibility property:** "KL divergence is non-negative (Gibbs' inequality), so minimizing KL(p_T || p_M) cannot make p_M worse than any fixed baseline that already minimizes KL partially." -- This is genuinely one property but the phrasing is imprecise: Gibbs' inequality does NOT guarantee that student+M2P is better than student base in absolute NTP loss terms. It guarantees that KL drives p_M toward p_T. If p_T is worse than p_0, "toward p_T" IS regression. The self-test answer is therefore **subtly wrong** about what the impossibility property protects — it conflates "moves toward teacher" with "does not regress." The PAPER.md correctly identifies this gap in post-hoc analysis, but the self-test should have caught it a priori.

2. **Cited theorems:** Gibbs' inequality (Cover & Thomas, correct citation), Hinton 2015 (correct), Csiszar 1975 (correct), JL 1984 (correct). All are real theorems. **However:** the JL lemma is cited for d_M2P = O(log(n)/eps^2) and then the experiment uses d_M2P=64 which is below the JL lower bound even with n=3 (the calculation in MATH.md gives O(110) for eps=0.1). The text acknowledges this but dismisses it by claiming "learned projections do better." This is hand-waving, not a theorem. The JL citation creates a false sense of rigor for a setting where its guarantees do not apply.

3. **Predicted numbers:** P1 (projection cosine > 0.5 intra, < 0.8 cross), P2 (closure >= 0.50), P3 (no regression), P4 (KL decreasing). These are specific and falsifiable. **Issue:** P1 only partially tested — cross-domain cosine was never measured (acknowledged in PAPER.md). P3 is stated as "unconditional" but actually depends on A2 holding.

4. **Falsification condition:** "The no-regression guarantee (P3) is falsified if student_m2p_loss > student_base_loss AFTER KL distillation training. This cannot happen with correct KL minimization..." -- This is **incorrect as stated.** The proof's no-regression guarantee is conditional on the teacher being better than the student (A2). The self-test claims P3 is unconditional ("This cannot happen with correct KL minimization"), but the experiment itself falsified this claim by violating A2. The falsification condition should have been: "P3 is falsified if A2 holds AND student_m2p_loss > student_base_loss."

5. **Hyperparameter count:** 2 new (T_temp, alpha). Acknowledged as literature-guided but not uniquely determined. Acceptable for Type 2.

6. **Hack check:** Clean — one mechanism replaces another. No stacking. PASS.

**Self-Test Verdict:** Items 1 and 4 contain a critical error — the no-regression guarantee is stated as unconditional when it is conditional on A2. This is the exact error that caused the experiment to produce a surprise result that should not have been surprising.

## Mathematical Soundness

### Theorem 1 (Knowledge Transfer via M2P with KL Distillation)

**Step 1 (KL lower bound):** Correct. KL(p_T || p_M) >= 0 by Gibbs. Minimizing moves p_M toward p_T. Standard.

**Step 2 (NTP loss decomposition):** L_NTP(q, data) = H(p_data) + KL(p_data || q). Correct, but note this decomposes NTP loss with respect to the data-generating distribution. When the "data" is teacher-generated soft targets, this is self-referential: the teacher IS p_data.

**Step 3 (Gap bound):** States KL(p_T || p_M) = KL(p_T || p_M*) + D_M2P. This decomposition is not rigorous — it assumes a clean partition of KL into "best achievable" and "approximation error" without proving such a partition exists in the form stated. More precisely, p_M* is defined as the minimizer of KL over the M2P function class, and D_M2P = KL(p_T || p_M) - KL(p_T || p_M*). This is just a definition (gap from optimum), not an independent bound. It does not actually bound L_NTP(p_M) relative to L_NTP(p_T) without knowing KL(p_T || p_M*).

**Step 4 (Projection fidelity):** Binary conditional: if cosine > 0.5, M2P can distinguish; if cosine <= 0, M2P cannot. This is not proven — it is a plausibility argument. There is no theorem connecting projection cosine similarity to M2P's ability to generate correct B-matrices.

**Corollary 1 (No-regression):** "The student+M2P adapter CANNOT be worse than the student base for teacher-domain data under KL distillation." **This is the central mathematical error.** The corollary states that minimizing KL(p_T || p_M) drives p_M toward p_T, and in the limit p_M = p_T. But "no regression below student base" is NOT a consequence of this. The KL objective has NO awareness of student base quality. It only knows about p_T. If p_T has higher NTP loss than p_0, then successfully minimizing KL(p_T || p_M) INCREASES NTP loss. The corollary conflates "convergence to teacher" with "improvement over baseline." These are the same only when the teacher is better.

The correct statement would be: "If A2 holds (teacher better than student), then student+M2P cannot regress below student base, because the KL gradient moves p_M toward a distribution with lower NTP loss." Without A2, no such guarantee exists.

**Corollary 2 (Quality Gap Closure):** "closure >= 0 (by no-regression, Corollary 1)". Since Corollary 1 is wrong without A2, Corollary 2 is also wrong without A2. The experiment confirms this: closure is massively negative.

### Theorem 2 (Learned Projection Preserves Domain Structure)

This is a proof sketch, not a proof. The argument that gradient is nonzero when domains project to the same vector is correct in principle (the Jacobian of a linear map has full rank). However, the conclusion "phi moves toward domain-separating projection" does not follow from gradient being nonzero — it requires that the gradient step is large enough relative to curvature, that the optimization landscape has no saddle points that merge domains, etc. This is a plausibility argument, not a theorem.

### Summary of Mathematical Issues

1. **Corollary 1 is incorrect as stated.** It is conditional on A2, not unconditional.
2. **Theorem 1 bound** (Step 3) is a tautological decomposition, not an informative bound.
3. **Theorem 2** is a proof sketch with a correct intuition but insufficient rigor for a QED.
4. **JL citation** is technically inapplicable (d_M2P below the JL lower bound).

None of these invalidate the experiment as a Type 2 guided exploration — the proven framework (KL distillation theory + M2P generation) is real and correctly cited. But the MATH.md overstates what the proofs guarantee.

## Prediction vs Measurement

PAPER.md contains the required prediction-vs-measurement table. Assessment:

| Prediction | Verdict | Commentary |
|------------|---------|------------|
| P1 (intra > 0.5, cross < 0.8) | PARTIAL | Intra measured (0.986-0.999 PASS). Cross never measured. Incomplete test. |
| P2 (closure >= 0.50) | FAIL | -5.71, -1.74, -0.91. Catastrophic failure. |
| P3 (no regression) | FAIL | student+M2P 25-33% worse than base. |
| P4 (KL decreasing) | PASS | 11-18% reduction during training. |
| P5 (projection cos > 0.2) | PASS | 0.986-0.999. |

The failures are correctly attributed and the root cause analysis (A2 violation) is sound.

## Root Cause Analysis Assessment

The PAPER.md root cause analysis is **sound and well-reasoned.** Key observations:

1. **A2 violation is correctly identified.** Student base loss (1.66-1.71) < teacher SFT loss (1.72-1.88) on all domains. The teacher is genuinely worse than the student.

2. **The explanation is mechanistically correct.** KL(p_T || p_M) minimization pushes p_M toward p_T. If p_T is worse, this is regression. The gradient is working as designed — the target is the problem, not the mechanism.

3. **The P3 violation is correctly explained without invoking bugs.** The proof's no-regression guarantee does not apply because A2 is violated. This is a logical consequence, not a surprise.

4. **Missing diagnostic:** Cross-domain projection cosine was not measured. This leaves P1 half-tested. The experiment should have logged this.

5. **Relevant prior art concern:** VISION.md lists "KD from large teacher: -34.4% worse than self-supervised -- Finding #30" as a permanently closed path. The MATH.md does not cite or address Finding #30. This is a significant omission. If teacher KD was already killed at Finding #30, this experiment needed to explicitly state what is different about the M2P approach versus whatever was tested in #30. The experiment may be re-treading a closed path.

## Impossibility Structure Assessment

The impossibility structure in LEARNINGS.md states:

> "KL distillation fails when A2 is violated. A2 violation is trivially avoidable by verifying teacher quality before training starts."

This is **correct but incomplete.** The structural fix (assert teacher_loss < student_loss) is valid as a precondition check. However:

1. **The impossibility structure does not address WHY A2 was violated.** At micro scale, same training budget + larger model = underfitting. This is a scale-dependent phenomenon. The LEARNINGS correctly notes that at macro scale (Qwen3-8B vs 4B), A2 would naturally hold because of pre-training data advantage. This is a reasonable (if unproven) claim.

2. **The impossibility structure should also address whether the MECHANISM works even when A2 holds.** A2 violation explains the failure, but fixing A2 only removes one obstacle. There could be additional failure modes (e.g., M2P capacity, projection quality) that are masked by the A2 violation. The experiment cannot distinguish between "A2 is the only problem" and "A2 is the first of several problems."

3. **The transferable pattern is sound:** "Before any distillation experiment: verify A2." This is a good engineering heuristic.

## Novelty Assessment

- **KL distillation (Hinton 2015):** Well-known. Not novel.
- **M2P generating adapters (Finding #339):** Prior work within this project. Not novel.
- **Combining M2P + KL distillation + cross-dimension projection:** Novel combination within this project, but straightforward composition of known techniques.
- **Finding #30 ("KD from large teacher: -34.4% worse than self-supervised"):** Listed as a permanently closed path in VISION.md. The relationship between this experiment and Finding #30 is not discussed. If #30 used a different mechanism (not M2P), then this experiment is differentiated; if #30 used a similar teacher-to-student distillation, then this experiment repeats a closed path without addressing the prior kill.

## Macro-Scale Risks (advisory)

1. **A2 would naturally hold at macro scale** (Qwen3-8B genuinely better than Qwen3-4B). The root cause of this kill is unlikely to recur.
2. **Projection quality at scale:** d_M2P=64 with d_T=8192 (Qwen3-8B) is a 128x compression. JL bound for n domains with eps=0.1 requires d_M2P >= O(log(n)/eps^2). For n=200 domains: d_M2P >= O(530). At d_M2P=64, the JL guarantee does not hold for even moderate domain counts. A larger d_M2P would increase M2P parameter count significantly.
3. **KL loss magnitude:** Final KL of 7.0-7.5 nats is large. At macro scale with V=150K vocabulary, soft target distributions are much higher-dimensional and KL optimization may be even harder.
4. **The experiment does not test whether M2P capacity is sufficient** — because A2 failure dominates. A macro-scale re-run must separately verify both A2 and M2P capacity.

## Verdict

**KILL (validated)**

The kill is justified for the following reasons:

1. **K853 FAIL is definitive.** All three domains show large negative gap closure. This is not a marginal failure.

2. **The root cause analysis is correct.** A2 violation (teacher worse than student) fully explains the failure. The KL mechanism worked as designed — the target was wrong.

3. **The mathematical framework has a real error** (Corollary 1 stated as unconditional when it requires A2), but this error was identified and corrected in the post-hoc analysis. The correction is sound.

4. **Critical omission:** Finding #30 ("KD from large teacher: -34.4% worse than self-supervised") is listed as a permanently closed path in VISION.md but is not cited or differentiated in MATH.md. Before any resurrection of this experiment, the relationship to Finding #30 must be explicitly addressed. If Finding #30 killed teacher KD generically, then M2P teacher distillation needs to demonstrate why the M2P variant escapes the prior kill.

**If resurrected, the following fixes are required:**

1. **Fix Corollary 1:** State the no-regression guarantee as conditional on A2 (teacher better than student), not unconditional.
2. **Implement A2 check:** The assert statement in LEARNINGS.md is correct. Make it a mandatory precondition that halts the experiment before M2P training begins.
3. **Measure cross-domain projection cosine:** P1 was only half-tested. Cross-domain cosine similarity must be logged to verify domain discrimination.
4. **Cite and differentiate from Finding #30:** Explain why M2P-based teacher distillation is not the same closed path as the prior KD kill.
5. **Address JL bound violation honestly:** Either increase d_M2P to satisfy the bound or remove the JL citation and state that the projection is purely learned without theoretical distance-preservation guarantees.
