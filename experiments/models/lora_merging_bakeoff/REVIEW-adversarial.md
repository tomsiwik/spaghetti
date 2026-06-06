# Peer Review: LoRA Merging Bakeoff (Revision 2)

## NotebookLM Findings

Skipped -- this is a comparison study with no novel mathematical contributions. The published algorithms (TIES, DARE) are well-understood. The experiment's value is empirical. Manual code-level review is the appropriate rigor level.

## Mathematical Soundness

### TIES Zero-Mask Bug: Correctly Fixed in v2

The previous review identified that the v1 "fix" still included zero-valued tasks via the sign-match path. The v2 code at `merging_methods.py` line 102 now reads:

```python
match_mask = match_mask * (1 - zero_elected[None, :]) + zero_elected[None, :] * has_nonzero
```

This is correct. At `elected_sign == 0` positions:
- `(1 - zero_elected)` zeroes out any entries from the sign-match path (clearing the problematic sign(0)==0 matches).
- `zero_elected * has_nonzero` includes only tasks with genuinely nonzero trimmed values.

The unit test `test_ties_zero_mask_fix()` verifies the logic with both N=3 and N=5 synthetic cases. The test is adequate.

### Impact of the Fix: Negligible, as Predicted

The paper reports TIES N=2 moved from +7.16% to +7.06% (within noise) and N=5 from +17.41% to +20.68% (higher, but std=0.035). The previous review predicted the fix would affect ~4% of positions with bounded impact. This is confirmed. The N=5 increase is within the high variance of TIES at that scale and does not change the conclusion.

### MATH.md "Optimal Estimator" Claim

The previous review noted that the claim "Simple average is the optimal linear estimator when vectors are orthogonal" conflates centroid with optimal estimator without specifying the estimation target. This cosmetic issue remains unfixed in v2 but was never blocking. The underlying reasoning (orthogonal deltas mean averaging preserves each component's projection without cancellation) is correct.

### DARE Implementation

Verified. `dare_sparsify` applies Bernoulli masking with rescaling `1/(1-p)`. The unbiasedness property `E[tau_k^dare] = tau_k` holds. The seed-reset concern from the previous review (all DARE variants sharing mask structure within a seed) is correctly noted as a minor issue -- drop rate, not mask pattern, is the dominant variable.

### Scaling Analysis Arithmetic

Spot-checked: DARE p=0.9 ratio = 7.31/7.21 = 1.01 (paper says 1.0x -- acceptable rounding). Concat+cal ratio = 5.07/1.14 = 4.45 (paper says 4.4x). These check out.

## Novelty Assessment

No novelty claim is made. This is a controlled comparison study. The specific finding -- that TIES and high-sparsity DARE are counterproductive on low-rank LoRA deltas where redundancy is already minimal -- is a reasonable empirical contribution. It is directionally consistent with task arithmetic literature but has not been explicitly demonstrated for rank-8 LoRA.

The DARE drop rate sweep (v1 addition, carried through v2) is the key empirical contribution: monotonic degradation with drop rate, convergence of DARE p=0.3 to simple average. This correctly re-frames the v0 conclusion.

No reinvention issues. The reference directories contain only README.md files (no implementations to build on). Implementations match published algorithms.

## Experimental Design

### Strengths

1. **Clean controlled comparison.** All methods share the same pretrained base, same LoRA fine-tuning, same evaluation. Only the merging step differs.
2. **Multi-seed, multi-scale.** 3 seeds at both N=2 and N=5.
3. **DARE sweep.** Four drop rates spanning the practical range.
4. **Honest kill criteria evaluation.** N=5 kills reported transparently with PARTIAL status.
5. **Revision discipline.** Two revisions addressing specific reviewer concerns with re-runs and updated numbers.
6. **Scaling analysis table.** The N=2-to-N=5 ratio analysis is a useful diagnostic.
7. **TIES bug fix verified with unit test.** `test_ties_zero_mask_fix` covers the specific failure mode.

### Remaining Weaknesses (Advisory, Not Blocking)

1. **TIES density not swept.** Only rho=0.2 tested. At rho=0.8 (keep 80%), TIES approaches simple average. This was advisory in v0 and v1 and remains advisory. The core conclusion (trimming compressed LoRA deltas destroys signal) is sound in principle, and rho=0.2 is the canonical setting from the original paper.

2. **Concat+calibrate at N=5 may be underfit, not fundamentally broken.** The paper notes high variance (std=0.0149) and suggests router instability with 5 experts and 100 calibration steps. Testing with 200-500 steps would distinguish underfitting from mechanism failure. The paper acknowledges this (Limitation 3) but does not test it. This matters for whether the N=5 KC1 kill reflects a fundamental limit or a tuning issue.

3. **DARE seed interaction.** The `mx.random.seed(seed)` reset in `dare_sparsify` means all four DARE variants within a single experiment seed share mask structure (differing only in threshold). This is technically correct for reproducibility but means the four variants are not statistically independent. The impact is negligible since drop rate is the dominant variable.

### Information Preservation as Simpler Explanation

The previous review noted that the ranking of methods is almost perfectly predicted by how much of the original delta information each method preserves. This remains the strongest explanatory frame. The paper's four-point theoretical analysis (Section "Why Simple Average Dominates") is not wrong but is more elaborate than necessary. The paper could be cleaner by leading with the information preservation argument.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_lora_merging_bakeoff`:
- **Status: partial** -- correct, matching the N=2 pass / N=5 kill.
- **Kill criteria match:** KC1 and KC2 both correctly tested at both scales.
- **Dependency:** `exp_lora_procrustes_linear` blocks this node. That experiment is marked proven. Consistent.
- **Evidence entries:** Updated to reflect v2 revision. Numbers match PAPER.md.

The revision history in PAPER.md correctly describes what was done in v2:
- Item 1: Bug fix with the correct code change, re-ran TIES and DARE-TIES at N=2 and N=5.
- Item 2: Corrects the false claim from v1.

No issues.

## Macro-Scale Risks (advisory)

1. **Orthogonality breakdown.** The simple-average-dominates finding depends on cos(tau_i, tau_j) near zero. At macro scale with genuinely different domains, cosine similarity could be much higher. If cos > 0.3, sign conflicts become real and TIES may help. Do not extrapolate "TIES hurts" to macro without re-measuring orthogonality.

2. **LoRA rank changes redundancy structure.** At rank 64-256, each LoRA delta has much more parameter redundancy than at rank 8. DARE's assumption (most delta parameters are redundant) becomes more plausible at higher ranks. DARE at moderate drop rates (p=0.3-0.5) could outperform simple averaging.

3. **Concat+calibrate N=5 diagnosis needed before scaling.** Determine whether the failure is router underfitting (fixable) or fundamental routing difficulty with many similar experts. This determines whether the contribution model scales beyond small N.

4. **Joint training cost advantage at scale.** At macro scale, the Nx gradient step advantage of joint training becomes enormous. Per-expert LoRA followed by merging becomes increasingly compute-efficient, making even small quality gaps more acceptable.

## Verdict

**PROCEED**

The v2 revision correctly addresses both required fixes from the v1 review:
1. The TIES zero-mask dilution bug is now actually fixed (verified in code and unit test).
2. The PAPER.md revision history honestly describes the fix and its negligible impact.

The core findings are sound:
- Simple average dominates for orthogonal low-rank LoRA deltas (clean, well-controlled result).
- DARE quality degrades monotonically with drop rate (v1 sweep, confirmed in v2).
- TIES hurts at both N=2 and N=5 regardless of the zero-mask bug (correctly predicted by the v1 review, confirmed by v2 re-runs).
- Concat+calibrate wins at N=2 but loses at N=5 (honest PARTIAL status).

The experiment is a comparison study, not a mechanism invention. It does what it claims: test whether published merging methods make our concat+calibrate approach redundant. The answer is nuanced (yes at N>=5 with insufficient calibration, no at N=2) and reported honestly. The remaining advisory items (TIES density sweep, concat+cal calibration scaling, information preservation framing) would improve the paper but are not blocking.

The PARTIAL status is the correct disposition. The experiment advances the project's knowledge of which merging strategies to use at different scales, directly informing macro-scale decisions.
