# Peer Review: Value Norm Dynamics (exp_l2_norm_value_boundedness)

## NotebookLM Findings
Skipped per instructions.

## Mathematical Soundness

### Derivations verified

1. **State norm bound (Eq. 1-3)**: The triangle inequality application to
   `S_t = g_t * S_{t-1} + k_hat_t @ v_t^T` is correct. With `||k_hat||=1`
   (L2 norm), the geometric series bound `V_max / (1-g)` follows directly.
   No hidden assumptions here.

2. **RMSNorm magnitude absorption**: The argument that `||RMSNorm(x)||_2 ~
   sqrt(d_h)` regardless of `||x||` is correct for RMSNorm (divides by RMS
   of input). This is the key mechanism preventing value norm growth.

3. **Frozen W_v + RMSNorm = bounded values**: This is sound. During router
   calibration, W_v is frozen. The only thing changing is which capsule groups
   contribute to x_t. RMSNorm normalizes x_t before W_v sees it. Therefore
   `||v_t|| = ||W_v @ RMSNorm(x_t)|| <= ||W_v||_op * sqrt(d_h)`, which is
   constant. The 1.09x empirical growth is consistent with directional changes
   in RMSNorm output (RMSNorm preserves direction, not just magnitude; different
   directions through W_v yield slightly different norms).

4. **Pearson correlation implementation**: Verified correct against standard
   formula. The `n < 3` guard returns 0.0 rather than NaN, which is acceptable.

### Minor gap: the bound argument is for the frozen-calibration regime only

MATH.md Assumption 1 correctly states this, and PAPER.md Limitation 3
acknowledges it. If full fine-tuning (including attention weights) were used
during composition, W_v would change and the bound breaks. This is a design
constraint, not a mathematical error -- the experiment tests exactly what it
claims to test.

### No issues found in the core mathematical argument.

## Novelty Assessment

**Prior art**: The state boundedness argument for gated linear attention with
L2-normalized QK is from GatedDeltaNet (Yang et al., 2024). The specific
contribution here is empirically verifying the value norm assumption that
the original argument leaves implicit.

**Delta over existing work**: Small but well-targeted. The adversarial review
of exp_l2_norm_composition_stability identified value norm growth as a gap in
the state boundedness proof. This experiment closes that gap with direct
measurement. Not publishable on its own, but necessary for the internal
research loop.

**No reinvention detected**: The model code correctly extends the existing
L2 norm hybrid capsule MoE rather than reimplementing from scratch. The
`ValueTrackingGatedLinearAttention` inherits the L2 norm logic via the
imported `l2norm` function from the parent experiment.

## Experimental Design

### Strengths

1. **Clean instrumentation**: The tracking model is functionally identical to
   the parent (param count verified in test). Only adds norm recording hooks.
   This eliminates confounds from architectural changes.

2. **Comprehensive trajectory**: Norms recorded at 5 phases (post-pretrain,
   post-finetune-A, post-finetune-B, pre-calibration, during calibration at
   10-step intervals). This is thorough enough to catch transient spikes.

3. **Appropriate controls**: Joint model norms recorded as reference. Growth
   ratios computed against post-pretrain baseline (the right reference point --
   before any domain-specific training).

4. **7 seeds**: Sufficient for K1 (the margin is enormous: 1.09x vs 10x).

### Weaknesses

**W1: Layer 3 (full attention) value norms are not tracked.**
The `enable_tracking()` method only enables tracking on `layer_type == "linear"`
layers. Layer 3 uses `CausalSelfAttention` (standard full attention), which has
no value norm instrumentation. The state boundedness argument is specifically
about linear attention recurrence, so full attention layers are theoretically
irrelevant (softmax attention is bounded by construction). However, the PAPER.md
per-layer analysis shows only layers 0-2 without noting that layer 3 was
excluded, which could mislead readers.

**Severity**: Low. Full attention value norms are not part of the state
boundedness concern (softmax denominators self-normalize). But the paper should
note the exclusion.

**W2: Correlation analysis is underpowered (7 points).**
With n=7, a Pearson r of 0.275 has a 95% CI roughly [-0.45, 0.78] (Fisher
z-transform approximation). This means r=0.5 is within the confidence interval.
The experiment cannot confidently distinguish r=0.275 from r=0.5. The paper
acknowledges this in Limitation 4 but still claims "PASS" on K2.

**Severity**: Medium for the correlation criterion specifically, but mitigated by:
(a) the growth ratios are so small (1.00-1.09x) that even if correlation were
high, the practical impact would be negligible; (b) the kill criterion asks
whether value norm growth is a *mechanism* for degradation, and at 1.09x max
growth there simply is not enough growth to be any mechanism.

**W3: Value norms measured on a single batch per recording point.**
`record_norms()` uses one batch of 32 samples. Batch-to-batch variance could
affect individual measurements. However, since the growth ratios are averaged
across seeds and the margins are enormous (1.09x vs 10x), this is unlikely to
change the verdict.

**Severity**: Low.

**W4: Norms averaged over batch and time before reporting.**
The per-head value norms are `mean(||v||_2)` over batch and sequence positions.
This could mask per-position spikes (e.g., high norms at specific sequence
positions). The max-over-heads-and-layers metric partially addresses this, but
a per-position max would be more conservative.

**Severity**: Low. The theoretical bound uses V_max (max norm), and the
empirical V_max proxy is the max over per-head means, not the true per-token
max. At 1.09x growth, even a factor-of-2 underestimate would leave ample
margin to 10x.

**W5: The baseline is post-pretrain, not post-finetune.**
Growth ratio = max_during_composition / baseline_post_pretrain. But value norms
may already change during fine-tuning (capsule groups are unfrozen, which changes
x_t and thus v_t via the residual stream). If fine-tuning increases norms by 5x
and composition adds another 2x, the experiment would report 10x growth when the
composition-specific growth is only 2x. Conversely, if fine-tuning decreases
norms, the experiment underestimates composition-specific growth.

**Severity**: Low-medium. The paper reports finetuned norms but doesn't compute
growth ratios relative to them. The theoretical argument (frozen W_v during
calibration) means the relevant baseline is pre-calibration norms, not
post-pretrain. The paper does report pre-calibration norms in the trajectory,
and they are close to calibration norms (9.81 -> 9.59), so this is consistent.
But the formal growth ratio definition in MATH.md uses post-pretrain as baseline,
which conflates fine-tuning and composition effects.

### Does the experiment test what it claims?

Yes. The core question is: "Do value norms stay bounded during composition?"
The answer is unambiguously yes -- 1.09x max growth with a 10x threshold.

### Could a simpler mechanism explain the result?

The result (norms are stable) is explained by the theoretical mechanism
(frozen W_v + RMSNorm). No alternative explanation needed -- this is the
expected null result confirming the assumption.

## Hypothesis Graph Consistency

- **Node**: `exp_l2_norm_value_boundedness`
- **Status**: `proven`
- **Kill criteria match**: Yes. K1 (>10x growth) and K2 (>0.5 correlation) are
  exactly what the code tests.
- **Depends on**: `exp_l2_norm_composition_stability` -- correct, this is the
  experiment whose state boundedness argument left the value norm assumption
  unverified.
- **Evidence claim**: Matches the empirical results in the paper.

No inconsistencies.

## Macro-Scale Risks (advisory)

1. **Longer sequences (T=4096+)**: At micro T=32, the recurrence runs 32 steps.
   At macro T=4096, even 1.09x value norm growth compounds in the state bound
   as `V_max * (1 - g^T)/(1 - g)`. But 1.09 * V_max vs 1.00 * V_max changes
   the bound by 9%, which is negligible. The real risk is whether value norms
   grow more at macro scale, not whether 1.09x compounds badly.

2. **More diverse expert pools (N=20+)**: With many independently-trained
   capsule pools, the routing changes during calibration could shift x_t through
   a wider range of directions. RMSNorm still normalizes magnitude, but the
   directional diversity could produce larger norm variation through W_v.
   Worth monitoring empirically at macro.

3. **Delta rule feedback loop**: PAPER.md correctly identifies this risk. The
   delta rule computes `v_t - S^T k_t` where S depends on previous values.
   If value norms and state norms interact in a positive feedback loop, the
   theoretical bound could become loose. Not testable at micro scale with the
   current simplified linear attention.

4. **SiLU base model**: The macro model uses SiLU activations (no dead neurons).
   The RMSNorm argument is activation-function-agnostic, so this should not
   affect value norm dynamics. But untested.

## Verdict

**PROCEED**

The experiment cleanly answers its stated question with enormous margins.
Value norms grow at most 1.09x (threshold 10x), confirming that the L2 QK
normalization state boundedness argument's implicit assumption holds.

The weaknesses identified are real but non-blocking:

1. The correlation analysis is underpowered (W2), but irrelevant in practice
   because the growth magnitudes are too small to drive any mechanism.
2. Layer 3 exclusion from tracking (W1) is theoretically justified but should
   be noted in the paper.
3. The growth ratio baseline definition (W5) conflates fine-tuning and
   composition, but the pre-calibration trajectory data shows the relevant
   comparison (pre-calib to post-calib) and confirms stability.

No mathematical errors. No missing prior art. No flawed experimental design.
The kill criteria are exactly what the code tests. The evidence is sufficient
to mark the node as proven.
