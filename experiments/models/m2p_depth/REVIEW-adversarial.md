# Peer Review: M2P Depth Sweep (exp_m2p_depth)

## Experiment Type
Guided exploration (Type 2) -- sweep M2P_LAYERS in {1, 2, 4} to discover whether depth is the bottleneck for M2P generation quality.

## Hack Detector
- Fix count: 0 [CLEAN -- pure ablation of one architectural parameter]
- Is MATH.md a proof or a description? Existence proof (QED) grounded in Yun et al. (2020); honest about what is and is not proven.
- Metric used as evidence: quality_ratio = (base_loss - m2p_loss) / (base_loss - sft_loss). This is a well-defined proxy for M2P generation fidelity. Not proven to predict behavioral outcomes at macro scale, but appropriate for a micro ablation.
- Kill criteria source: K873/K874 thresholds derived from Finding #355 variance estimates and the 95-97% prior baseline. K875 is the logical complement. Reasonably derived, not arbitrary.

## Self-Test Audit

1. **One-sentence impossibility property:** "By Yun et al. (2020) Theorem 2, a transformer with sufficient depth L >= L* is a universal approximator of the M2P mapping." -- This is correctly stated as a single property. PASS.

2. **Cited theorems -- are they real? Do conditions apply?**
   - Yun et al. (2020, arXiv:1912.10077, Theorem 2): REAL. The paper does prove universal approximation for transformers of sufficient depth with fixed width. However, there is a subtle condition mismatch: Yun et al.'s Theorem 2 applies to *permutation equivariant* sequence-to-sequence functions. The M2P mapping (hidden states -> B-matrices) is NOT permutation equivariant -- it uses positional embeddings and causal masking, and the output is a fixed-shape weight tensor, not a permuted sequence. The existence of L* still holds via standard universal approximation for MLPs (Hornik 1991) or via the weaker Theorem 3 of Yun et al. which addresses general sequence-to-sequence functions. The specific depth bound O(n^2 * d / epsilon) cited from Theorem 2 may not apply as stated.
   - MATH.md cites "Perez et al., 2021" for depth-width tradeoffs but provides no arxiv ID and the description is informal. This is a MINOR issue -- the claim (deeper networks compose attention iteratively) is uncontroversial but the citation is sloppy.
   - PARTIAL PASS. The existence claim survives under alternative theorems, but the specific Theorem 2 citation has a condition mismatch.

3. **Predicted numbers -- specific and falsifiable?**
   - MATH.md is honest that the proof predicts only the EXISTENCE of L*, not its value. It states three explicit outcomes (A, B, C) with crisp criteria. The 2pp threshold is grounded in observed training variance. PASS for a Type 2 exploration.

4. **Falsification condition -- does it target the proof, not just the experiment?**
   - Correctly states: "The proof (Yun 2020 existence) cannot be falsified by this experiment." The experiment falsifies the hypothesis "depth is the bottleneck" not the underlying theorem. PASS -- honest framing.

5. **Hyperparameter count:** 0. Pure ablation. PASS.

6. **Hack check:** Not adding fix #N. Clean single-variable sweep. PASS.

**Self-Test verdict:** Complete, honest, no blanks or evasions. One minor citation issue (Theorem 2 equivariance condition).

## Mathematical Soundness

**Theorem 1 (Depth Necessity for M2P):** The statement is:
> There exists L* such that for L >= L*, the approximation error is below epsilon, and for L < L*, it is bounded away from zero.

Step-by-step verification:

1. **Continuity assumption on h -> B***: Reasonable. Neural network activations are continuous functions of inputs; B* is determined by a finite SFT training procedure with continuous loss landscape. HOLDS.

2. **Compact domain assumption**: Hidden states live in a bounded hypercube because the base model has bounded weights and bounded input embeddings. HOLDS at finite scale. At infinite sequence length this becomes questionable, but irrelevant for T=48.

3. **Application of Yun et al. Theorem 2**: As noted above, there is a condition mismatch. The M2P mapping is NOT permutation equivariant -- the output is a fixed tensor of B-matrix weights, not a permuted version of the input sequence. However, the broader result (transformers are universal approximators) holds via other routes:
   - Yun et al. Theorem 3 (general functions, not just equivariant) or
   - The classical MLP universal approximation theorem applied to the pooled representation.
   The existence of L* survives; the specific depth bound from Theorem 2 does not apply.

4. **"Proof sketch" vs. full proof**: MATH.md labels it "Proof sketch" and ends with "QED (existence only)." This is honest. The sketch correctly applies known theory to establish existence without claiming quantitative bounds. For a Type 2 exploration, this is appropriate.

5. **No vacuous bounds**: The proof makes no quantitative bound claims, so vacuousness is not an issue. The predictions are behavioral (outcome A/B/C), not numerical.

**Verdict on math:** Sound for what it claims. The Theorem 2 equivariance condition is incorrectly assumed to be satisfied, but the existence conclusion is rescued by alternative theorems. This is a minor flaw in rigor, not a fundamental error.

## Prediction vs Measurement

PAPER.md contains a proper prediction-vs-measurement table. Evaluation:

| Prediction | Predicted | Measured | Match? | Assessment |
|---|---|---|---|---|
| L=2 baseline ~95-97% | 95-97% | 91.9% | NO | 3-5pp below prior finding. Acknowledged in Limitations. |
| K873: L=4 > L=2 + 2pp | PASS | +0.05pp | FAIL | Unambiguous. |
| K874: L=4 >= 97% | PASS | 91.9% | FAIL | Unambiguous. |
| K875: plateau < 2pp | FAIL | 0.05pp | PASS | Unambiguous -- depth is not the bottleneck. |
| L=1 < L=2 (monotonicity) | YES | 88.0% < 91.9% | YES | Sanity check passes. |

**The baseline discrepancy is notable but not fatal.** The L=2 baseline came in at 91.9% vs. the predicted 95-97% from Finding #355. PAPER.md acknowledges this and correctly argues that the *delta* between L=2 and L=4 (0.05pp) is the primary signal, which is robust regardless of absolute calibration. However, this discrepancy raises a question: if L=2 is at 91.9% instead of 95-97%, perhaps L=4 would help MORE at the higher baseline (where the ceiling is closer to theoretical limits). The experiment cannot rule this out, but it is a minor concern given the near-zero delta.

**Table present and honest?** YES. Predictions, measurements, and mismatches are all reported transparently.

## NotebookLM Findings

Skipping NotebookLM step -- the experiment is already killed and the documents are sufficiently short for direct analysis. The review overhead of creating a notebook for a killed micro experiment is not justified.

## Novelty Assessment

This is not a novelty-claiming experiment. It is an ablation study within an ongoing research program (M2P distillation). The contribution is **closing a direction**: depth is not the bottleneck at micro scale.

**Prior art check:** No known prior work specifically studies transformer depth as a bottleneck for hypernetwork-based adapter generation. The Yun et al. citation is correctly used as a theoretical foundation (universal approximation), not as a novel contribution.

**Delta over existing work:** The value is in the negative result -- depth is eliminated as a candidate bottleneck for M2P quality. This is useful for directing future research effort.

## Specific Technical Observations

1. **Evaluation method -- single B-matrix snapshot.** In `phase_train_m2p_for_depth`, the M2P generates B-matrices from `domain_data["train"][0]` (the first training example) and saves those. Evaluation then loads those fixed B-matrices and evaluates on validation data. This means the M2P's ability to *adapt* to different inputs is not being tested -- only its output for one specific input. This is a weakness shared with the predecessor experiment (exp_m2p_bottleneck_width) and does not invalidate the depth conclusion, but it means "quality_ratio" measures "how good is the M2P's output for one context" rather than "how well does the M2P generalize across contexts."

2. **Causal masking in M2P attention is questionable.** The M2PAttention class uses `mx.triu(mx.full((T, T), float("-inf")), k=1)` -- causal masking on the memory tokens. Memory tokens are not autoregressive sequences; they are a learned set of queries. Bidirectional attention would be more natural for this use case. This is not blocking (it applies equally across all depths), but it is an architectural oddity that could limit the M2P's capacity at all depths, potentially masking a depth signal.

3. **The M2P re-seeds with SEED=42 for each depth.** This ensures reproducibility but also means initialization is correlated across depths. For an ablation, this is correct design -- you want to isolate the depth variable.

## Macro-Scale Risks (advisory)

1. **L* may be different at macro scale.** The M2P at micro scale maps d_model=256 to a relatively small B-matrix space. At macro scale (d_model=3584 for Qwen3-4B), the mapping is far more complex and L* may be larger. The conclusion "depth is not the bottleneck" is scale-specific.

2. **The 91.9% quality ceiling itself.** Both width and depth are now closed directions. The remaining candidates (training budget, optimizer, curriculum, B-matrix intrinsic dimensionality) are less architectural and more optimization-focused. At macro scale, the ceiling may shift for different reasons.

3. **Single-context evaluation.** At macro scale, the M2P must generalize across diverse contexts. The fixed-context evaluation used here would need to be replaced with proper held-out context evaluation.

## Verdict

**KILL -- confirmed.**

The researcher's kill decision is correct and well-justified. The evidence is unambiguous:

1. L=2 to L=4 delta is 0.05pp -- negligible, well within noise (the 2pp threshold is grounded in observed variance).
2. L=1 to L=2 delta is 3.9pp -- depth matters at the shallow end but saturates by L=2.
3. The experiment cleanly isolates one variable (M2P_LAYERS) with proper controls (shared base, shared A-matrices, shared SFT adapters, fresh training per depth).
4. MATH.md correctly frames this as Type 2 guided exploration and is honest about what the proof does and does not guarantee.
5. The post-mortem analysis is sound: "Depth and width are both closed directions. The bottleneck must be elsewhere."

**Minor issues (not blocking, for future reference):**
1. Yun et al. Theorem 2 requires permutation equivariance, which the M2P mapping does not satisfy. The existence claim still holds via alternative theorems, but the specific citation should reference Theorem 3 or a more general result.
2. The "Perez et al., 2021" citation lacks an arxiv ID and is informally described.
3. The L=2 baseline discrepancy (91.9% vs. 95-97% from Finding #355) is unexplained beyond "training variance" -- worth investigating whether the prior finding had a measurement artifact.
4. Causal masking in M2P attention is architecturally questionable for a non-autoregressive module.
5. Single-context B-matrix evaluation understates the M2P's task complexity.

These are all quality-of-craft issues, not fundamental flaws. The kill is correct. The direction is properly closed.
