# Peer Review: m2p_vera_bottleneck

## Experiment Type
Guided exploration (Type 2). The proven framework is VeRA + functional forward (Theorem 5). The unknown is whether rank-4 VeRA scale vectors achieve 70% quality on GSM8K. This classification is appropriate.

## Hack Detector
- Fix count: 1 (single mechanism change: replace output heads with VeRA reconstruction). Clean, no stacking.
- Is MATH.md a proof or a description? Mixed. Theorem 1 (parameter count) is arithmetic -- correct but trivially provable. Theorem 2 (gradient flow) is a genuine proof. Theorem 3 is labeled "proof sketch" and is indeed only a sketch.
- Metric used as evidence: quality_ratio (m2p_acc - base) / (sft - base). This is a standard normalized quality metric grounded in the SFT baseline.
- Kill criteria source: K922 derived from Theorem 1, K924 derived from Theorem 2, K923 derived from VeRA Table 2 extrapolation (empirical, not proven). K923 is the only empirically-grounded kill criterion, which is appropriate for Type 2.

## Self-Test Audit

1. **One-sentence impossibility property:** "VeRA's shared random projection... requiring only 2r scalars per layer." This is one property, stated clearly. PASS -- with a caveat on the "2r" claim (see Mathematical Soundness below).

2. **Cited theorems:** VeRA (arXiv:2310.11454), JL lemma (1984), HyperNetworks (arXiv:1609.09106). These are real papers. However, VeRA's cited "Theorem 1 and S4.1" is a stretch -- VeRA does not contain a formal theorem numbered "Theorem 1"; it describes the parameterization in Section 4. Minor misattribution but not fabricated. PASS with note.

3. **Predicted numbers:** Specific: 4,656,576 trainable params, 76x reduction, grad_norm > 0, quality_ratio >= 0.70, m2p_acc >= 0.280. All falsifiable. PASS.

4. **Falsification condition:** "Theorem 1 is falsified if actual param count exceeds 10M. This cannot happen given the formula." The self-test correctly identifies that Theorem 1 is unfalsifiable (it is arithmetic). Theorem 2's falsification (grad_norm = 0) is stated. K923 failure is correctly identified as not falsifying the math. PASS -- honest about what is and is not at stake.

5. **Hyperparameter count:** Claims zero new hyperparameters. This is debatable -- output_scale=0.032 is a hyperparameter (SHINE convention, not VeRA). The initialization of d_vec/b_vec matters but is not swept. Marginal PASS.

6. **Hack check:** "No. This replaces the v4 output heads entirely." Correct, single mechanism change. PASS.

## Mathematical Soundness

### Theorem 1 (Parameter Count) -- PASS with errors

The parameter count arithmetic is correct for what was actually built, but MATH.md contains an error: it assumes N=36 layers throughout the proof while the actual model (Qwen3-0.6B) has N=28. PAPER.md catches this discrepancy ("off by 12,288, 0.26%, from N_layers=28 not 36"). The proof should have been written for the actual model. This is sloppy but not blocking -- the conclusion (PASS K922 by large margin) holds at either value.

The scale_head size in the proof statement (Linear(1024, 576)) disagrees with the actual proof body (Linear(1024, 448)) and the code (Linear(1024, 448)). The 576 figure is for N=36; the actual implementation uses N=28 giving 448. Internal inconsistency within the same document.

### Theorem 2 (Gradient Flow) -- PASS

The chain rule derivation is correct. The gradient through `diag(d) @ W.T @ diag(b)` is indeed nonzero when W is nondegenerate and b is nonzero. The proof is sound. K924 PASS at 0.1064 confirms this.

### Theorem 3 (Expressive Power) -- CRITICAL ERROR: Dimensionality Mismatch + DOF Collapse

**This is the most important finding in this review.**

MATH.md states the reconstruction formula as:
```
B_q_i = diag(d_q_i) @ W_q.T @ diag(b_q_i)
```
where `d_q_i, b_q_i in R^r` and `W_q in R^{d_q x r}`.

Dimensions: `diag(d_q_i)` is `(r x r)`, `W_q.T` is `(r x d_q)`, `diag(b_q_i)` is `(r x r)`.

The product `W_q.T @ diag(b_q_i)` is `(r x d_q) @ (r x r)` -- **this is a dimensionality mismatch** (`d_q != r`). The formula as written in MATH.md does not type-check.

The code implements something different that does type-check:
```python
W_q_scaled = (W_q * b_q[None, :]).T   # column-scale W_q by b_q, then transpose
B_q_i = W_q_scaled * d_q[:, None]      # row-scale by d_q
```

This computes `B_q_i[i,k] = W_q[k,i] * b_q[i] * d_q[i]`, which equals `diag(d_q * b_q) @ W_q.T`.

**The two r-dimensional scaling vectors collapse to a single r-dimensional vector** via element-wise product. The effective degrees of freedom per module per layer is r=4, not 2r=8 as claimed. The claim of "2r DOF per layer" throughout MATH.md is wrong for this implementation.

**Comparison to original VeRA:** In Kopiczko et al., `lambda_d in R^{d_out}` (scaling rows of the d_out x r shared matrix) and `lambda_b in R^r` (scaling columns). The DOF is `d_out + r` per adapter, not `2r`. The authors of MATH.md misunderstood VeRA: they made both scaling vectors r-dimensional instead of making one d_out-dimensional. This is why:

1. The formula in MATH.md has a dimensionality error (trying to right-multiply (r x d_q) by (r x r))
2. The code "fixes" this by implementing a different formula that happens to compile but collapses the two vectors
3. The effective DOF is r=4 per module, not 2r=8 as claimed

This error partially explains the quality failure: the effective capacity is even lower than the MATH.md analysis suggests. The impossibility analysis in PAPER.md ("2r=8 DOF insufficient") should say "r=4 effective DOF insufficient" -- the actual situation is worse than diagnosed.

**However:** This error does not change the kill decision. Even with a correct VeRA implementation giving d_out + r DOF, we would have 2048 + 4 = 2052 DOF for q_proj and 1024 + 4 = 1028 DOF for v_proj per layer -- much larger than 2r but still needing to be output by M2P. Fixing VeRA would require M2P to output d_out-dimensional vectors per layer, which brings back much of the parameter cost the experiment tried to eliminate. The fundamental tension (compress M2P output vs maintain adapter quality) remains.

### JL Lemma Application -- Overreach

MATH.md claims: "By JL lemma, W.T has approximately orthonormal rows with high probability (for d >> r), so diag(b_i) can steer independently in each row direction."

The JL lemma guarantees distance preservation for random projections, not orthonormality of rows. For a (r x d) random matrix with d >> r, the rows are approximately orthogonal by concentration of measure, but this is a consequence of high-dimensional probability, not the JL lemma specifically. The citation is imprecise. More importantly, the "independent steering" claim is about the interaction of diagonal scaling with a random matrix -- the JL lemma says nothing about this.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table. Assessment:

| Prediction | Predicted | Measured | Match? |
|------------|-----------|----------|--------|
| Trainable params | 4,656,576 | 4,668,864 | Close (0.26% off, explained by N=28 vs 36) |
| Reduction vs v4 | ~76x | 97.9x | Better than predicted (different N assumption changes both) |
| grad_norm > 0 | > 0 | 0.1064 | PASS |
| quality_ratio >= 0.70 | >= 0.70 | -0.105 | FAIL by huge margin |
| m2p_acc >= 0.280 | >= 0.280 | 0.188 | FAIL, below even base rate |

The table is honest. The statistical analysis (Wilson CI, Fieller CI, z-tests) is correct and thorough. The CI upper bound of 0.197 for quality_ratio vs threshold 0.70 leaves no ambiguity. The kill is statistically sound.

## NotebookLM Findings

Skipped per time constraints. The mathematical issues found via manual review are sufficient.

## Novelty Assessment

VeRA applied to hypernetwork output is a reasonable extension. However, the misunderstanding of VeRA's parameterization (both scaling vectors being r-dimensional instead of one being d_out-dimensional) means this experiment did not actually implement VeRA correctly. It implemented a strictly less expressive variant.

Prior art: VeRA (arXiv:2310.11454) is correctly cited. The M2P/SHINE framework is properly referenced. No missing critical prior art.

## Macro-Scale Risks (advisory)

1. A correct VeRA implementation (with d_out-dimensional lambda_d) would require M2P to output O(d_out * N_layers) scalars, which partially defeats the parameter reduction goal.
2. The fundamental tension -- M2P needs to generate expressive adapters but the output dimension determines M2P's size -- is not resolved by VeRA at low rank.
3. At rank 16 (where VeRA was validated), the DOF per layer is d_out + 16 which is essentially d_out. This suggests VeRA's compression benefit comes primarily from sharing the random basis across layers, not from reducing per-layer DOF.

## Verdict

**KILL** -- the kill is justified.

### Justification

1. **K923 FAIL is statistically conclusive.** quality_ratio = -0.105 with CI upper bound 0.197, vs threshold 0.70. The adapter performs at base rate (z = -0.48, p = 0.63 vs base). No bug -- the architecture genuinely lacks capacity.

2. **The impossibility structure is real but mis-diagnosed.** The effective DOF per module is r=4 (not 2r=8 as claimed), because both scaling vectors are r-dimensional and their composition collapses to a single r-dimensional scaling. The MATH.md formula has a dimensionality error that the code works around by implementing a different (less expressive) operation.

3. **The VeRA parameterization was incorrectly adapted.** Original VeRA uses lambda_d in R^{d_out} (not R^r), giving d_out + r DOF. This implementation gives only r DOF per module. A correct VeRA implementation would restore most of the parameter cost.

4. **MATH.md Theorem 1 has an internal inconsistency (N=36 in statement, N=28 in code)** but this does not affect the kill decision.

5. **Theorem 3 is a proof sketch with a dimensionality error** -- the formula `diag(d) @ W.T @ diag(b)` with d, b in R^r and W in R^{d_out x r} does not type-check. The code implements `diag(d * b) @ W.T` instead.

### What to record in the finding

The impossibility structure should be corrected from "2r DOF insufficient" to: "r effective DOF per module (due to collapsed scaling vectors) is insufficient. Even a correct VeRA implementation with d_out + r DOF would require M2P to output O(d_out) scalars per layer, partially negating the compression benefit." The failure mode is the fundamental tension between M2P compression and adapter expressivity at low rank.
