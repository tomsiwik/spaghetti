# Peer Review: Delta Rank Scaling v2

## NotebookLM Findings

Skipped. The experiment is a clean SVD measurement with straightforward statistics. Manual review is sufficient.

## Mathematical Soundness

### Effective rank computation: CORRECT (unchanged from v1)

Roy & Vetterli (2007) Shannon effective rank is implemented correctly. The `rank_at_threshold()` using cumulative Frobenius energy is also correct.

### Convergence control (Fix #1): CORRECTLY IMPLEMENTED, STRENGTHENS RESULT

The two-phase approach (train d=64 to get target, then train all sizes to that target) is clean. The implementation at `train_to_target_loss()` checks val loss every 100 steps and stops when target is reached. The re-training pass with `train_with_checkpoints()` uses the same seed for reproducibility.

Key finding: v1 **over-trained** larger models (d=128: 2000 vs 1267 needed, d=256: 3000 vs 2100 needed). This means the v1 convergence confound was biased **against** the hypothesis (over-training inflates rho), not in favor of it. The v2 result (steeper decline: -0.188 vs -0.152 exponent) is consistent with this correction. This is the single most important fix -- it eliminates the primary confound I identified in v1.

One subtlety: the target val loss is set by d=64 at 1000 steps. Since d=64 is the most capacity-constrained model, it reaches the highest loss floor. Larger models can beat this target easily, so convergence is well-defined. The approach is conservative -- if anything, larger models are slightly under-utilized at the target loss, which would bias rho downward. The multi-checkpoint analysis (Fix #5) confirms this is not a problem because the inter-d ordering holds at all checkpoints.

### Embedding exclusion (Fix #2): CORRECT AND WELL-MOTIVATED

The argument is clean: embedding matrices have shape (V x d) with V=27, so min_dim = 27 for all d >= 27. The ratio rho_emb is a function of V, not d. Including it contaminates the d-scaling signal. The table in MATH.md Section 3 clearly shows embeddings increasing (0.774 to 0.833) while FFN+Attn decrease (0.650 to 0.501). This is the right call.

### Bootstrap CI (Fix #3): CORRECTLY SELF-DEPRECATING

The CI is computed correctly (10K bootstrap resamples from 3 seeds per d). The paper is honest that the narrow CI [-0.190, -0.185] reflects only seed variance, not model/extrapolation uncertainty. The explicit caveat "3 points and 2 parameters, any monotonic function would fit well" and "R^2 = 0.980 has 1 degree of freedom" are exactly right. The extrapolation table now includes the critical warning: "These CIs are misleadingly narrow."

This is a model for how to report bootstrap CIs on small samples. No complaint.

### K1 acceptance (Fix #4): HONEST

K1 is accepted as killed without retroactive reinterpretation. The paper notes that r_95 gives better numbers but explicitly says "this was not the pre-registered metric" and recommends a separate experiment with r_95 kill criteria. This is exactly what I asked for in v1. No special pleading detected.

### Multi-checkpoint rho (Fix #5): STRONG EVIDENCE

The trajectory analysis is the most valuable addition. Key observations:
1. Inter-d ordering (d=64 > d=128 > d=256) holds at every checkpoint (25/50/75/100%). This is the strongest evidence that the scaling trend is real.
2. d=256 is near-plateau (+0.004 from 75% to 100%), while d=64 is still climbing (+0.017). This means extended training would **widen** the gap, making the convergence control conservative.
3. The data shows rho is monotonically increasing with training, which means the concern about under-training inflating the effect is backward -- more training makes the effect weaker, not stronger.

### Power law with 3 points: ACKNOWLEDGED LIMITATION

The paper is transparent that 3 points cannot distinguish power law from linear, logarithmic, or exponential fits. The exponent b = -0.188 is presented as "the ratio decreases" not as a precise scaling law. This is appropriate.

## Novelty Assessment

### Prior art: ADEQUATELY CITED

The experiment cites Aghajanyan et al. (2021) on intrinsic dimensionality and arXiv:2510.00537 on spectral scaling. The delta here is measuring **pretraining** deltas (W_trained - W_random_init) rather than fine-tuning deltas, which is the relevant quantity for the base-as-adapter architecture. The paper could be clearer about why this distinction matters (pretraining deltas reflect total learned structure, fine-tuning deltas reflect task-specific adaptation), but this is a minor presentation issue.

No reinvention of existing code detected.

### The r_95 metric is an important secondary contribution

The observation that practical rank (r_95) scales more steeply than Shannon rank (0.438 to 0.273 vs 0.650 to 0.501) is operationally relevant. Shannon rank counts the tail; r_95 counts what matters for reconstruction. This distinction is well-known in signal processing but under-appreciated in the LoRA literature.

## Experimental Design

### Fix assessment: ALL 5 FIXES ADEQUATELY ADDRESS V1 CONCERNS

| V1 Concern | Fix Applied | Assessment |
|------------|-------------|------------|
| Under-training at larger d | Convergence control (same val loss) | Eliminates confound; strengthens result |
| Embedding contamination | FFN+Attn primary metric | Correct; clean separation |
| No CI on exponent | Bootstrap CI + honest caveat | Correctly self-deprecating |
| K1 retroactive reinterpretation | Accept kill cleanly | Honest; recommends follow-up |
| No multi-checkpoint analysis | Rho at 25/50/75/100% | Strong evidence for robustness |

### Remaining concerns (non-blocking)

1. **Over-parameterization confound persists.** At d=256 with V=27 and 32K names, the model is massively over-parameterized. At macro scale the model is capacity-limited. This could reverse the trend. The paper acknowledges this in Limitations #2 and #3 ("real models at d=4096 learn from internet-scale data... fixed task complexity is unrealistic"). This is a known micro-scale limitation that cannot be resolved at micro scale.

2. **Still only 3 data points.** Adding d=512 would strengthen the trend enormously. However, at d=512 (12.8M params) on CPU with convergence control, training time becomes substantial. This is a resource limitation, not a design flaw.

3. **The two-phase training approach doubles compute.** The code trains to convergence (Phase 1), then re-trains from scratch with checkpoints (Phase 2). This is because the number of steps is unknown a priori. A more efficient implementation would save checkpoints during the initial training run. Not a correctness issue, just a note for future experiments.

### Cohen's d = 49.7: LEGITIMATE BUT UNSURPRISING

The extremely large effect size is real (the means differ by 0.149 with std ~0.003), but this reflects the precision of the SVD measurement, not the practical importance of the effect. With 3 seeds and very low measurement noise, even modest differences produce enormous Cohen's d. The paper does not over-interpret this, which is appropriate.

## Hypothesis Graph Consistency

The `exp_delta_rank_scaling` node correctly:
- Records status as `weak_kill`
- Lists both K1 KILLED and K2 SURVIVES evidence
- Documents all 5 fixes and re-run confirmation
- Notes `completed: 2026-03-16`
- Has appropriate `blocks: []` (does not block anything)
- Depends on `exp_base_free_composition` (correct lineage)

The FINDINGS.md entry correctly separates the four findings (ratio decreases, attention scales best, r_95 scales steeply, multi-checkpoint ordering stable).

VISION.md reference at line 361 mentions "K1 killed but r_95 metric promising" which accurately reflects the WEAK_KILL status. The "revise" label in VISION.md line 322 should be updated to "weak_kill" for consistency.

## Macro-Scale Risks (advisory)

1. **Over-parameterization reversal is the primary risk.** At macro scale, models are not over-parameterized for their training data. The declining rank ratio could flatten or reverse. The cheapest test: compute r_eff on an existing fine-tuned Qwen 0.5B (d=896) delta. This costs zero compute (two checkpoint downloads + SVD).

2. **The r_95 extrapolation to rank ~500 at d=4096 is the load-bearing claim for VISION.md feasibility.** This extrapolation is based on 3 informal data points without a formal fit. If macro validation shows r_95/d > 0.15 at d=896, the rank-500 estimate may be too optimistic.

3. **Layer count scaling is untested.** All micro models have 4 layers. Qwen 7B has 32 layers. Per-layer rank requirements could differ systematically with depth, which this experiment cannot capture.

## Verdict

**PROCEED**

All 5 fixes from the v1 REVISE verdict have been adequately addressed. The convergence control (Fix #1) not only eliminates the primary confound but actually strengthens the result -- the v1 concern about under-training at larger d was backward (v1 over-trained larger models). The multi-checkpoint analysis (Fix #5) provides the strongest evidence: the inter-d ordering is preserved at every checkpoint fraction, ruling out convergence artifacts as an explanation.

The WEAK_KILL characterization is honest and appropriate:
- K1 is killed on the pre-registered Shannon metric. Accepted without special pleading.
- K2 survives strongly. The ratio does decrease with d, unanimously across seeds and checkpoints.
- The practical implication (r_95 scales favorably) is noted as a secondary finding requiring its own formal test.

The remaining concerns (over-parameterization at micro scale, 3 data points, layer count) are inherent micro-scale limitations that the paper acknowledges and cannot resolve without macro experiments. The recommended next step (measure r_eff on Qwen 0.5B at d=896) is exactly right and would add a 4th data point at 3.5x the current maximum d.

No fixes required. The experiment can be archived as a completed WEAK_KILL with K2 findings feeding into macro validation planning.
