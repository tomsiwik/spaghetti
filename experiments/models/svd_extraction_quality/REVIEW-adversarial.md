# Peer Review: SVD Extraction Quality

## Experiment Type
Mixed: Type 1 (verification of Eckart-Young predictions P1, P2, P4) and Type 2
(guided exploration of rank-quality tradeoff, P3/P5/P6).

## Hack Detector
- Fix count: 1 (truncated SVD extraction). Clean, single mechanism.
- Is MATH.md a proof or a description? Mixed. Theorems 1 and 2 are genuine
  proofs with correct QED blocks. "Theorem 3" is NOT a proof -- it is a
  conditional conjecture with an explicit caveat that it is "NOT guaranteed."
  It should not be labeled "Theorem."
- Metric used as evidence: Domain PPL ratio (SVD/raw). PPL is a reasonable
  proxy for domain quality but the relationship between reconstruction error
  and PPL is not monotonic (as the experiment itself demonstrates). MMLU
  accuracy used for knowledge preservation.
- Kill criteria source: K834 (ratio < 2.0) is reasonable but very generous --
  at 2x worse PPL the experiment would clearly have failed. K835 (SVD MMLU <=
  raw LoRA MMLU) is derived from the Davis-Kahan motivation. Both pass easily.

## Self-Test Audit

1. **One-sentence impossibility property:** "Eckart-Young theorem guarantees
   that truncated SVD is the OPTIMAL rank-r approximation." This answers a
   different question than the experiment's main finding. The experiment's
   surprise is that LOWER rank = BETTER PPL, which Eckart-Young does NOT
   guarantee (it guarantees minimal reconstruction error, not minimal PPL).
   The self-test answer is technically correct but misaligned with the actual
   finding.

2. **Cited theorems:** Eckart-Young-Mirsky (1936) -- real theorem, correctly
   stated, conditions apply. Davis-Kahan sin-theta (1970) -- real theorem,
   correctly stated, but conditions require symmetric matrices (the adapter
   delta is not symmetric; the theorem applies to W^T W + delta^T delta type
   analysis, not directly to the delta itself). The Davis-Kahan application is
   informal/hand-wavy -- used for intuition, not as a rigorous bound.

3. **Predicted numbers:** Specific and falsifiable. P1: rank=16 ratio=1.000.
   P2: rank>16 same. P3: monotonic degradation. P4: ratio<2.0. Good.

4. **Falsification condition:** "The proof is wrong if rank(scale * B^T @ A^T)
   > r_lora." This correctly identifies what would break Theorems 1-2 but is
   trivially impossible (as acknowledged). It does not address what would
   falsify the main finding (the regularization effect).

5. **Hyperparameter count:** 1 (SVD truncation rank). Correctly identified as
   the exploration target.

6. **Hack check:** Clean. SVD is a single operation, not a stack of fixes.

## Mathematical Soundness

**Theorem 1 (Lossless SVD at native rank):** CORRECT. The proof is simple and
valid. rank(M * N^T) <= min(rank(M), rank(N)) <= r_lora. Truncated SVD at the
matrix rank preserves all nonzero singular values.

**Theorem 2 (Monotonic quality degradation):** CORRECT as stated -- it
predicts monotonic increase in RECONSTRUCTION ERROR, not monotonic PPL
degradation. The MATH.md then says "Implication: PPL should degrade
monotonically as rank decreases" which is an unjustified leap. Reconstruction
error and PPL have no proven monotonic relationship. The experiment correctly
identifies this as refuted (P3).

**"Theorem 3" (SVD as implicit regularization):** NOT A THEOREM. This is a
conditional hypothesis: "IF the destructive components are concentrated in
small singular values, THEN truncation helps." There is no proof, and the text
explicitly says "This is NOT guaranteed." Labeling this as a "Theorem" is
misleading. It should be labeled "Hypothesis 3" or "Conjecture 3."

**Delta computation correctness:** Verified. The code computes
delta = scale * B^T @ A^T (shape d_out x d_in), performs SVD, then constructs
A_svd and B_svd such that x @ A_svd @ B_svd = x @ delta^T = x @ (scale * A @ B),
which correctly reproduces the RuntimeLoRA computation y = base(x) + alpha * (x @ A) @ B.
The transpose chain is correct.

**Davis-Kahan application:** The sin-theta theorem applies to perturbations of
symmetric matrices. Using it to reason about LoRA deltas (which are rectangular,
not symmetric) is an informal analogy, not a rigorous application. The intuition
is reasonable (smaller perturbation = less subspace rotation) but the formal
conditions are not met.

## Prediction vs Measurement

| Prediction | Expected | Measured | Match? |
|------------|----------|----------|--------|
| P1: rank=16 lossless | ratio=1.000, error=0 | ratio=0.9985, error=0.0 | YES (0.15% bf16) |
| P2: rank>16 lossless | ratio=1.000 | r32=r64=r128=0.9985 | YES |
| P3: monotonic PPL degradation | ratio(r=4) > r=8 > r=16 | 0.766 < 0.841 < 0.999 | REFUTED |
| P4: best rank < 2.0x | < 2.0 | 0.766 | YES |

The prediction-vs-measurement table is present in PAPER.md and matches
results.json exactly. Data consistency verified: all PPL values, ratios,
spectral values, and MMLU numbers match between PAPER.md and results.json.

**Error in PAPER.md reconstruction error table:** The "Interpretation" column
says "74.7% energy lost" for rank=4 relative error of 0.747. This is wrong.
The relative Frobenius error is sqrt(discarded_energy / total_energy) = 0.747,
which means discarded_energy / total_energy = 0.558 = 55.8% energy lost, not
74.7%. The spectral analysis section correctly reports energy fractions
(46-56% at rank=4), contradicting the reconstruction error table's
interpretation. Minor error in exposition, not in data.

## The Central Question: Why Does Truncation Help?

This is the most important issue in the review. The experiment observes that
SVD at rank=4 (keeping only 48% of energy) yields 23% BETTER PPL than the
full-rank adapter. The paper attributes this to "implicit regularization" --
truncation removes destructive interference while preserving useful signal.

**Alternative hypothesis not tested:** Scale=20 adapters are known to be
destructive (Finding #320: -60pp MMLU). The magnitude of the perturbation is
simply too large. SVD at rank=4 reduces the Frobenius norm by sqrt(0.48) =
0.69x. This is roughly equivalent to reducing the scale from 20 to ~14. The
improvement could be entirely explained by MAGNITUDE REDUCTION, not by the
SVD's specific selection of which directions to keep.

To distinguish "SVD regularization" from "magnitude reduction," the experiment
would need at minimum one of:
1. Random rank-4 projection (same dimension reduction, different subspace)
2. Scale reduction to ~14 (same magnitude, full rank)
3. Bottom-4 SVD (keep smallest SVs, discard largest -- should be catastrophic
   if the SVD direction selection matters)

Without these controls, the claim that SVD is doing something SPECIFIC (rather
than just reducing magnitude) is unsubstantiated. The paper partially hedges
this but the implications section says "SVD solidification works" and "SVD
truncation acts as a spectral regularizer" without sufficient qualification.

**This is not blocking** because the experiment is correctly typed as Type 2
(guided exploration) for this aspect, and the kill criteria are about whether
SVD experts are usable (not about whether the mechanism is regularization vs
magnitude reduction). But any downstream finding claiming "SVD regularization"
must first rule out the magnitude hypothesis.

## MMLU Methodology

The MMLU test uses 50 hardcoded questions (same as exp_pro_composition_mmlu).
Logit-based evaluation is correct. Questions are easy trivia level, which is
appropriate for detecting catastrophic knowledge loss but not for nuanced
benchmark comparison. Only the medical adapter was tested for MMLU, limiting
generalizability.

The 95% CI at 50 questions is approximately +/-13pp (binomial, not the 7.5pp
stated in PAPER.md -- the paper uses 1.96*sqrt(p(1-p)/n) which gives ~7.5pp
at p=0.5, but the relevant comparison is the 30pp gap between conditions,
which is significant regardless). The paper's 7.5pp CI estimate is technically
the half-width for a single proportion, not a comparison CI, but the
conclusion holds -- 30pp difference is clearly significant.

## Novelty Assessment

**FlexMoRE (arXiv:2312.15007)** already demonstrates SVD extraction from
fine-tuned models with 93-107% quality preservation. The observation that 5/6
experts improve after SVD extraction is directly from FlexMoRE. This
experiment replicates FlexMoRE's finding in the Pierre/Grassmannian context
with LoRA adapters specifically, which is valuable for the project but not
novel in the broader literature.

The MMLU preservation result (SVD reducing MMLU degradation from -60pp to
-30pp) is a useful practical finding not directly in FlexMoRE, but it follows
naturally from magnitude reduction.

## Macro-Scale Risks (advisory)

1. **Rank=4 sweet spot may not generalize.** This was found at scale=20 on
   Qwen3-4B with specific LoRA training. Different scales, models, or training
   regimes may have different optimal ranks. FlexMoRE found knowledge tasks
   peak at r=4 but reasoning at r=2896.

2. **The magnitude-reduction alternative** becomes important at macro scale.
   If the benefit is purely from reducing perturbation magnitude, then scale
   tuning is simpler and cheaper than SVD extraction.

3. **Composition of SVD experts** is untested here (deferred to
   exp_solidified_composition_mmlu). Multi-expert composition could re-
   introduce the interference that truncation removed.

4. **bf16 precision chain.** The float32->SVD->bf16 pipeline is fine at micro
   scale but at macro scale with many layers, accumulated quantization error
   from the bf16 cast of SVD factors should be verified.

## Specific Corrections Needed

1. **MATH.md:** Rename "Theorem 3" to "Conjecture 3" or "Hypothesis 3." It
   has no proof and explicitly acknowledges "This is NOT guaranteed."

2. **PAPER.md reconstruction error table:** Change "74.7% energy lost" to
   "relative Frobenius error 0.747 (55.8% energy discarded)" to match the
   correct interpretation and the spectral analysis section.

3. **PAPER.md implications:** Qualify "SVD truncation acts as a spectral
   regularizer" with "or equivalently, as magnitude reduction -- the two
   explanations are not yet distinguished." Add the magnitude-reduction
   alternative explicitly.

4. **PAPER.md P3 interpretation:** The claim that "the bottom SVs carry
   interference/noise, not domain signal" is not established. An equally
   valid interpretation is that ALL SVs carry some mix of signal and noise,
   and reducing total magnitude helps because the adapter at scale=20 is
   simply too strong. The directional interpretation (specific SVs =
   destructive) requires the controls listed above.

## Verdict

**PROCEED** (with corrections)

The Eckart-Young verification (P1, P2, P4) is clean: the proofs are correct,
the predictions match, the code is verified. The SVD extraction pipeline
works correctly and the data is internally consistent. Kill criteria pass
honestly (and by large margins).

The Type 2 exploration finding (rank=4 improves PPL by 23%) is a genuine and
useful empirical observation. The MMLU result (-30pp vs -60pp) is practically
valuable for the project pipeline.

However, the mechanistic explanation (implicit regularization via removal of
destructive SVD directions) is overclaimed. The magnitude-reduction
alternative is equally plausible and untested. The corrections listed above
should be applied before this feeds into downstream findings.

Required corrections before recording the finding:
1. Rename "Theorem 3" to "Hypothesis 3" in MATH.md
2. Fix the energy-loss misinterpretation in PAPER.md reconstruction table
3. Add the magnitude-reduction alternative hypothesis to PAPER.md
4. Finding status should be **supported** (not conclusive) -- the SVD
   extraction works, but the mechanism is not proven
