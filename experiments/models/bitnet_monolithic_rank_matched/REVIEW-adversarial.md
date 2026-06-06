# Peer Review: bitnet_monolithic_rank_matched

## NotebookLM Findings
Skipped (no blocking novelty claims requiring deep literature search; this is a controlled ablation of a prior experiment).

## Mathematical Soundness

**Parameter matching: CORRECT.**
The rank-scaling math is straightforward and verified empirically. Each LoRA adapter contributes `2 * d * r` parameters per linear layer. With 7 target modules across 30 layers, a rank-80 adapter has exactly 5x the parameters of a rank-16 adapter. The results.json confirms `trainable_params_r80 = 108,134,400` and `param_ratio_vs_sole = 1.0`. No issues.

**MATH.md information-theoretic argument: DIRECTIONALLY SOUND but imprecise.**
The claim that the union gradient has "approximate block-diagonal structure" is a reasonable hand-wave given |cos| = 0.002, but block-diagonality of gradients does not follow from block-diagonality of converged adapter weights. Converged adapters are orthogonal; the gradient trajectories during joint training may overlap substantially in transient directions. This is a narrative framing issue, not a mathematical error -- the experiment is designed correctly regardless of whether the argument is tight.

**LoRA scaling factor: NEEDS SCRUTINY.**
`LORA_SCALE = 20.0` is used for both r=16 (prior experiment) and r=80 (this experiment). In standard LoRA, the scaling factor is `alpha/r`. If alpha was tuned for r=16, using the same absolute scale at r=80 means the effective scaling is 5x higher relative to what `alpha/r` would give. This could either help (higher learning rate equivalent) or hurt (gradient instability). The paper does not discuss this choice. However, since both conditions use the same TernaryLoRALinear implementation with the same `scale=20.0`, and since the monolithic r=80 *did* converge (loss 1.85 -> 1.46), this is unlikely to be the sole explanation for results -- but it is a confound that favors the monolithic condition (higher effective lr could help or hurt, and we cannot tell which).

**Verdict: Math is sound. The scaling factor issue is a minor confound worth noting but not blocking.**

## Novelty Assessment

This is not a novelty claim -- it is a controlled ablation. The experimental design (rank-matched monolithic vs routed experts at equal total params) is standard in MoE literature (e.g., Shazeer et al. 2017, Switch Transformer, GShard all include monolithic baselines). The specific contribution is doing this for ternary LoRA on BitNet-2B-4T, which is sufficiently novel in context.

No prior art in `references/` directly addresses this comparison for ternary adapters.

## Experimental Design

### What is done well
1. **Parameter matching is exact** (1.00x ratio, verified in results.json)
2. **Same training data** (reuses prior experiment's `monolithic/train.jsonl`)
3. **Same hyperparameters** (lr=1e-4, seq_len=128, seed=42, 2000 steps)
4. **Base PPL verification** (re-evaluated base model, all within <1% of prior results)
5. **Uses SOLE PPLs from the prior experiment** rather than re-training, eliminating one source of variance

### Issue 1: Training step asymmetry (SIGNIFICANT)

**SOLE experts**: 5 experts x 400 steps = 2000 total gradient updates, but each expert sees only its own domain's ~800 samples (legal: 500). Each expert cycles through its data ~0.5x per step (400 steps / 800 samples).

**Monolithic r=80**: 2000 steps on the shuffled union of 3700 samples. Each sample is seen 2000/3700 = ~0.54x on average.

**This is actually well-matched in terms of data exposure per step.** However, there is a subtler issue:

The monolithic r=80 has 5x more parameters than a single SOLE expert but sees the same number of gradient steps (2000). Each SOLE r=16 expert has 21.6M parameters trained for 400 steps. The monolithic r=80 has 108M parameters trained for 2000 steps. In terms of the optimizer-steps-per-parameter ratio:
- SOLE: 400 steps / 21.6M params
- Mono: 2000 steps / 108M params

These ratios are identical (1.85e-5 steps/param), so step budget is actually fair. **No issue here on closer inspection.**

### Issue 2: Optimizer state asymmetry (MINOR)

SOLE experts each get independent Adam optimizer states (momentum, variance). The monolithic r=80 gets one Adam optimizer over all 108M parameters. SOLE's domain-specific momentum is cleaner (no cross-domain gradient noise in the moving averages). This is a genuine but minor advantage for SOLE that the paper does not discuss. It is inherent to the architecture comparison, not a confound -- it IS part of why specialization works.

### Issue 3: Single seed (ACKNOWLEDGED)

Single seed (42) with no variance estimates. The paper acknowledges this. The multiseed validation experiment showed CV=0.5% at N=5, which partially mitigates but does not fully address this.

### Issue 4: Creative domain pathology (ACKNOWLEDGED)

The SOLE creative expert showed INCREASING loss during training in the prior experiment (1.24 -> 1.64, marked "converged: false"). This means the SOLE creative PPL (3.17) comes from a partially failed training run. If SOLE creative had trained properly, it might have beaten monolithic on creative too -- or it might not. The current "creative exception" finding is confounded by this training failure. The paper frames it as "cross-domain transfer benefit" but the alternative explanation (SOLE creative training failure) is equally plausible.

### Issue 5: SOLE routed PPLs assume perfect routing (ACKNOWLEDGED in prior review)

SOLE "routed" means each domain is evaluated using only its own specialized expert. In production, routing accuracy < 100% would degrade SOLE's advantage. This is acknowledged and not blocking for micro.

### Issue 6: PPL gaps are small (IMPORTANT)

The gaps between SOLE and mono r=80 are:
- Medical: -3.0% (8.00 vs 8.25)
- Code: -2.1% (2.76 vs 2.82)
- Math: -3.5% (3.12 vs 3.23)
- Legal: -5.8% (17.95 vs 19.04)
- Creative: +6.3% (3.17 vs 2.98)

These are all small in absolute terms. The medical gap is 0.25 PPL points, code is 0.06, math is 0.11. At these margins, with single-seed evaluation, the statistical confidence is low. The multiseed CV of 0.5% translates to roughly +/- 0.04 PPL at code's level (2.8), +/- 0.10 at legal's level (19.0). The code gap (0.06) is within noise given seed variance. Medical (0.25) and legal (1.10) are likely signal.

**Conservative re-assessment: SOLE wins legal and medical convincingly, math probably, code is noise, creative is a mono win. Realistic scoreline: 2-3 clear SOLE wins, not 4.**

This does NOT change the kill criterion outcome (mono needs 3+ to kill), but it weakens the "4/5 domains" headline claim.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_bitnet_monolithic_rank_matched` correctly specifies:
- Kill criterion: "rank-80 monolithic beats SOLE routed on >60% of per-domain metrics"
- Status: supported
- Evidence matches results.json

The node's `depends_on: [exp_bitnet_sole_vs_monolithic]` is correct (reuses SOLE PPLs from that experiment).

The kill criterion is well-calibrated: 3+/5 wins for mono is a reasonable threshold for "composition adds no value."

## Macro-Scale Risks (advisory)

1. **LoRA scaling factor**: At macro scale (d=4096+), the `scale=20.0` hardcoding becomes more consequential. Standard `alpha/r` scaling should be used.

2. **More than 5 domains**: The monolithic rank needed to match SOLE grows linearly with N. At N=25, the monolithic would need r=400. At some N, the monolithic becomes impractical while SOLE remains modular. The micro result (N=5) is the most favorable case for monolithic.

3. **Routing accuracy in production**: The 2-6% SOLE advantage at micro could be eaten by routing errors. Need to validate routing accuracy at macro.

4. **Training duration at scale**: Rank-80 training already took 20 min vs ~3 min per SOLE expert. At macro scale this asymmetry grows. Not a quality issue but an engineering one.

## Verdict

**PROCEED**

The experiment is well-designed, the parameter matching is exact, and the conclusion (SOLE routed wins at matched params) is directionally correct. The specific weaknesses are:

1. **The "4/5 domains" headline should be softened** to acknowledge that code's 0.06 PPL gap (2.1%) is within plausible seed variance. Recommend stating "SOLE wins 3-4/5 domains" or noting which wins are statistically robust.

2. **The creative exception should note the SOLE creative training failure** (loss increased during training) as an alternative explanation to "cross-domain transfer," or at minimum as a confound.

3. **The LoRA scaling factor** (fixed at 20.0 for both r=16 and r=80) should be noted as a design choice. Standard practice would scale as alpha/r, meaning the r=80 condition has 5x higher effective scaling. Whether this helps or hurts mono is unknown.

None of these are blocking. The kill criterion (3+/5 mono wins) is clearly not met (1/5), and even the most conservative reading (2 clear SOLE wins, 1 noise, 1 probable, 1 mono win) does not approach the kill threshold. The experiment achieves its stated purpose: eliminating the parameter-count confound from the SOLE vs monolithic comparison.
