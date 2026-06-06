# Peer Review: L2 QK Normalization for Composition Stability

## NotebookLM Findings

Skipped -- the experiment is straightforward enough that manual review is sufficient. The mathematical claim is a single, well-known bound (Cauchy-Schwarz on unit vectors), and the experimental design is a controlled A/B with matched seeds.

## Mathematical Soundness

**The core bound is correct and tight.** L2 normalization forces ||q||=||k||=1, so |q^T k| = |cos(theta)| <= 1. This is a hard bound, not statistical. The derivation in MATH.md is correct at every step.

**The recurrent state boundedness argument is mostly correct but has a gap.**

The paper claims that with L2 norm, ||S_t||_F <= g_t * ||S_{t-1}||_F + ||v_t||_2, and calls this a "contraction." This is imprecise. It is a contraction only if the driving term ||v_t||_2 is bounded. In this experiment, v_t = W_v @ x and ||v_t||_2 is NOT bounded by L2 normalization (only Q and K are normalized). The state remains bounded because:

1. g_t < 1 provides geometric decay of old state
2. ||v_t||_2 is empirically bounded by the data distribution and weight norms
3. The fixed point ||S||_F ~ ||v||_2 / (1 - g_t) exists but is not explicitly derived

This is not a mathematical error -- the conclusion holds -- but the reasoning is incomplete. The paper should acknowledge that the bound on S_t depends on v_t norms being well-behaved, which is a training dynamics property, not a hard guarantee from L2 normalization alone. L2 normalization eliminates the QK magnitude amplification but does not provide a complete stability guarantee independent of value norms.

**The computational overhead calculation is correct.** 2*d_h + 1 per head per token, negligible.

**The epsilon handling is fine.** eps=1e-6 in rsqrt is standard practice.

**Assumption 1 (direction carries signal, not magnitude) is standard and well-supported.** This is the same principle behind cosine similarity attention, QK normalization in various transformers, and the fact that softmax is shift-invariant. The empirical results confirm it: median gap improves.

**Assumption 4 (transfers to delta rule) is correctly flagged as unvalidated.** Good scientific practice.

## Novelty Assessment

**This is not novel -- and the paper correctly does not claim novelty.** L2 QK normalization is standard in GatedDeltaNet. The paper explicitly cites that Qwen3.5 uses `use_qk_l2norm_in_kernel=True` by default. The contribution is not the normalization technique itself but the specific finding that it eliminates composition-specific instability (a failure mode that only manifests when independently-trained modules are composed).

**Prior art check:**
- Yang et al. (2024) GatedDeltaNet: uses L2 norm on Q/K
- Qwen3.5: uses it in production
- Henry et al. (2020), Dehghani et al. (2023): QK normalization in transformers generally
- The experiment correctly references all relevant work

**Delta over existing work:** The experiment shows that L2 QK normalization matters specifically for post-hoc expert composition, not just for training stability. This is a valid micro-scale finding. No prior work tests L2 normalization in the context of composing independently-trained adapters.

## Experimental Design

**Strengths:**

1. **Matched seeds across conditions.** Both conditions run the same 25 seeds (0-24). This is the right design -- it enables per-seed comparison (Table in PAPER.md showing seed 7, 15, 21, 23 catastrophic without norm but normal with norm).

2. **Adequate sample size.** 25 seeds is sufficient to distinguish 0% vs 16% failure rates. The Clopper-Pearson 95% CI upper bound of 11.3% is correctly reported and honestly noted as still near the 10% threshold.

3. **Two kill criteria targeting different failure modes.** Criterion 1 (catastrophic rate) tests whether the instability is fixed. Criterion 2 (median degradation) tests whether the fix introduces regression. Both are necessary.

4. **Identical protocol to predecessor.** The experimental protocol exactly matches the hybrid attention experiment, enabling direct comparison.

**Concerns:**

1. **The unnormalized baseline uses `hybrid_capsule_moe` while L2 uses `l2_norm_hybrid_capsule_moe`.** These are different model classes. I verified in the code that `l2_norm_attention.py` imports `CausalSelfAttention` from the hybrid model for full attention layers, and only differs in the linear attention class by adding two `l2norm()` calls (lines 71-72). The implementation is clean -- the only difference is the two normalization lines. No hidden changes.

2. **No statistical significance test reported.** The paper reports descriptive statistics (mean, median, std, min, max) but no formal test. Given the 0/25 vs 4/25 comparison, Fisher's exact test gives p = 0.038 (one-sided), which is significant at alpha=0.05. The paper could be stronger by reporting this, but the practical significance is clear from the per-seed comparison table. This is a minor omission, not a flaw.

3. **Potential confound: L2 normalization as implicit regularizer.** The paper notes that the L2 normalized model achieves lower joint training loss (0.510 vs 0.541). This means the comparison is not purely "same model with/without norm" -- the L2 model trains differently from step 1. The gap improvement (-0.33% vs +2.54%) could partially reflect a better-trained base, not just composition stability. However, the catastrophic failure analysis (same seeds fail/pass) is immune to this confound: the question is binary (does it blow up?), and the answer is definitively no with normalization.

4. **The gap metric uses joint training as reference, not an oracle.** The joint model is trained on both domains simultaneously (600 steps alternating). The composed model is pretrained (300 steps all data), fine-tuned per domain (300 steps each), composed, and calibrated (100 steps). The total compute differs: joint sees 600 batches, while compose sees ~1000 (300+300+300+100). The joint baseline may be undertrained relative to the composed model, which could explain some negative gaps (composed better than joint). This is a pre-existing design issue from the hybrid attention experiment, not specific to this one.

**Does the experiment test the hypothesis?** Yes, directly. The hypothesis is "L2 norm eliminates catastrophic composition failures." The experiment shows 0/25 vs 4/25. The hypothesis is confirmed.

**Could a simpler mechanism explain the results?** Unlikely. L2 normalization is already the simplest possible fix -- it is a parameter-free, fixed function. The only simpler alternative would be "the failures are just bad luck and would go away with more training steps," but the per-seed comparison (same seeds fail/succeed) rules this out.

## Hypothesis Graph Consistency

The experiment is registered as `exp_l2_norm_composition_stability` in HYPOTHESES.yml with status `proven`. The kill criteria match exactly:

- Kill criterion 1: ">10% catastrophic failure rate across 20+ seeds" -- tested with 25 seeds, 0% failure rate. PASS.
- Kill criterion 2: "L2 normalization degrades median composition gap by >3% vs unnormalized" -- degradation is -2.87pp (improvement). PASS.

The dependency on `exp_hybrid_attention_composition` is correctly declared. The experiment directly addresses the "conditional pass" from that predecessor.

## Integration Risk

**Low.** L2 normalization is already standard in the target architecture (GatedDeltaNet/Qwen3.5). Adding it to the micro model is architecturally clean (two lines of code, no new parameters). It composes with all existing components (capsule pools, routing, pruning).

**One integration question:** The delta rule experiment (`exp_delta_rule_interference`) is the natural next step. L2 normalization stabilizes QK products, but the delta rule introduces v_t - kv_mem cross-domain retrieval that operates through the state S_t, not through QK products. L2 normalization may be necessary but insufficient for full GatedDeltaNet composition. The paper correctly identifies this as an open question.

## Macro-Scale Risks (advisory)

1. **At macro scale (d_h=256), the unnormalized failure rate may be lower.** The QK dot product of random unit vectors concentrates around 0 as d_h grows (law of large numbers). At d_h=16, variance is high and extreme QK products are more likely. At d_h=256, the concentration effect may reduce the failure rate even without L2 normalization. This does not invalidate the micro finding (L2 norm is still needed as a safety guarantee), but the "16% catastrophic without norm" rate is likely micro-specific.

2. **Production GatedDeltaNet already uses L2 normalization.** This means the macro experiment will trivially confirm the finding -- L2 norm is the default. The real macro risk is whether composition-specific interference emerges from the delta rule mechanism, which L2 normalization does not address.

3. **Value norm stability at macro scale.** The incomplete mathematical argument about S_t boundedness (discussed above) becomes more relevant at scale where value norms can grow through residual stream accumulation over many layers. Monitor value norms during macro composition experiments.

## Verdict

**PROCEED**

The experiment is well-designed, the math is sound (with one minor incompleteness in the state boundedness argument), the code is clean (verified: exactly two lines of l2norm differ between conditions), and the results are clear-cut. 0/25 vs 4/25 catastrophic failures with 25 matched seeds is a definitive result for the stated hypothesis. The kill criteria are appropriate and both pass. The paper honestly acknowledges all relevant limitations including the simplified variant caveat, head dimension caveat, and the unvalidated delta rule transfer assumption.

Minor suggestions (not blocking):

1. Add Fisher's exact test p-value (p=0.038) for the 0/25 vs 4/25 comparison to strengthen the statistical argument.
2. Clarify in MATH.md that the state boundedness argument depends on value norms being well-behaved, not just on L2 normalization of Q/K.
3. Note the potential confound that L2 normalization acts as an implicit regularizer during training (lower joint loss), which may contribute to the median gap improvement beyond pure composition stability.
