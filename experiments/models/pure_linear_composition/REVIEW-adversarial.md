# Peer Review: Pure-Linear Composition Control

## NotebookLM Findings

Skipped -- the experiment is a straightforward control with clear math. Deep review not warranted for the complexity level.

## Mathematical Soundness

**Derivations verified:**

1. **Kill criterion calculation**: Degradation = (pure_comp_mean - hybrid_comp_mean) / hybrid_comp_mean * 100 = (0.5110 - 0.5059) / 0.5059 * 100 = 1.02%. Confirmed against raw results.json values. Arithmetic is correct.

2. **Param count analysis**: The paper correctly identifies a +4% param advantage for pure-linear (240K vs 231K) because the fourth layer's GatedDeltaNet has more projections (w_a, w_beta, w_z, conv1d weights) than CausalSelfAttention (wq, wk, wv, wo only). This is correctly flagged as a confound favoring pure-linear. Verified in the model code.

3. **Computational cost worked example**: The MAD counts in MATH.md are plausible at micro scale. The claim that GDN is more expensive per-layer at micro (T=32) but cheaper at macro (T >> d) due to O(T) vs O(T^2) is standard and correct.

4. **Per-layer interference metric**: Cosine distance between capsule pool outputs from domain A and domain B models, fed the same attention-processed input from the base model. Mathematically sound as an interference proxy.

**Hidden assumptions that hold:**

- The composition protocol is identical across conditions (verified in code: same compose_models, calibrate_router, same steps/lr/batch_size). The only variable is layer_types. This is clean.
- Seeds 0-6 are used identically across conditions, with the same data splits per seed (domain_split and train_val_split use the seed). This ensures paired comparison validity.

**One minor issue (not blocking):**

- The "degradation" metric compares mean composed losses directly, but the per-seed gaps are computed against each condition's own joint baseline. These are conceptually different measures. The paper reports both (degradation = 1.02%, gap difference mean = +0.21pp, gap difference median = -0.05pp), which is transparent. The kill criterion correctly uses the direct loss comparison, not the gap-of-gaps.

## Novelty Assessment

**This is a control experiment, not a novelty claim.** The paper explicitly states this. It fills a gap identified by adversarial review of the hybrid attention experiment: the 3:1 result was ambiguous about whether the one full attention layer was load-bearing for composition.

**Prior art check:**

- No published work directly tests "does linear attention need full attention scaffolding for MoE composition." The closest is Qwen3.5's architectural choice of 3:1, which is empirical, not ablated. This control adds value.
- The experiment correctly builds on the existing full_gdn_stack_capsule_moe model (no reinvention).
- References to prior experiments in the lineage (hybrid_attention, l2_norm_attention, delta_rule_attention, full_gdn_stack) are complete and accurate.

## Experimental Design

**Strengths:**

1. **Three-way comparison**: full_attn (0:4), hybrid (3:1), pure_linear (4:0) -- provides both endpoints and the validated midpoint. This is good experimental practice.

2. **7 seeds**: Adequate for the effect size. With gap_std ~1% and threshold at 5%, even a two-sided t-test has ample power. The 95% CI for the pure-linear gap mean (0.21% +/- ~0.8%) is well within the 5% threshold.

3. **Paired seeds**: Same data splits per seed across conditions. Enables paired comparison if needed.

4. **Zero catastrophic failures**: 0/21 runs failed. The L2 normalization fix from prior work holds universally.

**Concerns:**

1. **The 5% threshold is generous.** A 5% degradation in composed loss would be large. The experiment could pass even if pure-linear were meaningfully worse but not catastrophically worse. However, the observed 1.02% is well below even a stricter 2% threshold, so this does not change the verdict.

2. **Variance asymmetry**: Pure-linear gap_std = 1.07% vs hybrid gap_std = 0.60%. The hybrid is nearly 2x more stable. The paper notes this (finding 5) but understates it. At macro scale with more domains (N=5+), this variance difference could compound. The paper's own "what would kill this" section correctly identifies N=5+ as a risk.

3. **The interference metric is informative but not the kill criterion.** The paper reports that pure-linear has lower interference at all layers, which is interesting. But this is a supplementary finding, not what was being tested. The lower interference could simply reflect that GDN's exponential decay compresses all signals (including useful ones), not that it handles composition better. The paper does not over-claim here.

4. **Joint training confound (acknowledged)**: Pure-linear joint loss is 0.80% worse than hybrid. This means the composition gap is being computed against a slightly worse baseline. If joint quality matters independently of composition, hybrid is still marginally better. The paper correctly notes this.

5. **Param count confound (acknowledged)**: +4% params for pure-linear. The paper flags this and notes it slightly favors pure-linear. Since the result is a null finding (no degradation), this confound works against the conclusion -- the extra capacity might be compensating for a real linear-attention limitation. However, 4% extra params in the attention module (not the capsule pools, which are the composition-relevant part) is unlikely to close a 5% composition gap. Not blocking.

## Hypothesis Graph Consistency

The experiment matches its HYPOTHESES.yml node (exp_pure_linear_composition):
- Kill criterion: "pure-linear composition degrades >5% vs hybrid 3:1 composition" -- matches exactly
- Status: proven (hypothesis disproven -- pure-linear is NOT worse) -- correct interpretation
- Depends_on: exp_hybrid_attention_composition -- satisfied (that experiment is proven)

The evidence entry correctly describes the result. The node status is appropriately set.

## Integration Risk

**No integration risk.** This is a control experiment that narrows the design space. It does not introduce new components. The finding (attention type does not matter for composition) simplifies future architectural choices: macro experiments can choose attention type based on efficiency/quality tradeoffs without worrying about composition compatibility.

## Macro-Scale Risks (advisory)

1. **State capacity saturation**: At d_h=256 (macro), the GDN recurrent state is 256x256 = 65K values per head. With T=4096+ sequences, state saturation could cause information loss that full attention would not suffer. The paper identifies this risk.

2. **Cumulative global context loss**: With 4 layers, removing one global attention layer is a 25% reduction. With 24+ layers and a 3:1 ratio, removing all 6 full attention layers is a more aggressive change. The cumulative effect of having zero global context integration points across many layers is untested.

3. **Variance at scale**: The 1.8x variance ratio (pure vs hybrid) could widen at macro scale, especially with more composed domains. The hybrid's lower variance may be practically important even if mean quality is similar.

4. **The efficiency argument inverts at macro**: The paper correctly notes that GDN is actually more expensive per layer at micro scale but cheaper at macro (O(T) vs O(T^2)). The pure-linear result becomes most valuable precisely where it is least tested -- long sequences at scale.

## Verdict

**PROCEED**

The experiment is a well-designed control that cleanly answers its stated question. The math is correct, the experimental protocol is sound, the kill criterion is appropriate, and the result (1.02% degradation vs 5% threshold) is decisive. The paper correctly identifies its limitations and does not overclaim.

Minor notes for the record (not blocking):
1. The paper could benefit from a brief statistical test (e.g., Welch's t-test on composed losses between pure-linear and hybrid) to formalize the "no significant difference" claim, but with n=7 and such a clear margin, this is cosmetic.
2. The higher variance of pure-linear (1.07% vs 0.60%) deserves a sentence noting that while mean quality is equivalent, hybrid may be preferable at macro scale for its stability, especially under the Qwen3.5 architectural precedent.

Both of these are already implicitly covered in the paper's findings and limitations sections. No revisions required.
