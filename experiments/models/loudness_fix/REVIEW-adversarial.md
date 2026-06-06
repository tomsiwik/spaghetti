# Peer Review: Loudness Fix

## NotebookLM Findings

Manual deep review conducted in lieu of automated NotebookLM analysis. The review
is based on close reading of MATH.md, PAPER.md, the implementation
(`loudness_fix.py`), unit tests (`test_loudness_fix.py`), the full composition
experiment (`test_composition.py`), the parent model (`relu_router.py` and its
`test_composition.py`), the previous adversarial review (`REVIEW-adversarial.md`),
VISION.md, IDEA-RELU-ROUTER.md, FINDINGS.md, and ADVERSARIAL_REVIEW.md.

---

## Previous Review: Status of 6 Required Fixes

The relu_router REVIEW-adversarial.md specified 6 required fixes. The loudness_fix
experiment was charged with addressing these (VISION.md item 7 explicitly states
"Also address the 6 required fixes from adversarial review"). Here is the status:

| # | Fix | Status | Evidence |
|---|-----|--------|----------|
| 1 | Fix copy-paste bug in `test_relu_router.py` line 164 | **FIXED** | B_composed now uses `pool_a.B.weight` and `pool_b.B.weight` (line 164 of current file) |
| 2 | Add functional composition test (verify identity numerically) | **FIXED** | `test_composition_identity_numerical()` added, verifies `composed(x) == pool_a(x) + pool_b(x)` with max diff < 1e-5 |
| 3 | Reframe narrative: composition protocol, not self-routing | **FIXED** | `relu_router.py` docstring now reads: "This model IS a standard two-layer ReLU MLP. The architecture has zero novelty. The contribution is the COMPOSITION PROTOCOL." |
| 4 | Add weight-averaging baseline to composition experiment | **FIXED** | `weight_average_relu_models()` implemented and tested; weight averaging results reported (+2.0% in relu_router, +1.5% in loudness_fix) |
| 5 | Separate calibration from fine-tuning: train only per-pool scalars | **FIXED** | `scalar_calibrate()` trains only 8 scalar params; full calibration explicitly labeled as "unfair, for reference" |
| 6 | Acknowledge sparsity control is untested | **FIXED** | `relu_router.py` docstring: "Auxiliary losses (L1 + balance) are included but do NOT push sparsity beyond ~50% at micro scale." Target reduced from 75% to 50% |

**All 6 required fixes from the previous review have been addressed.** This is
good research hygiene.

---

## Mathematical Soundness

### What holds

**1. The composition identity is correct and now verified.**

The claim in MATH.md Section 1:

```
y_composed = B_composed @ ReLU(A_composed @ x)
           = B_a @ ReLU(A_a @ x) + B_b @ ReLU(A_b @ x)
           = Pool_A(x) + Pool_B(x)
```

This is mathematically exact, verified numerically in `test_composition_identity_numerical()`
and `test_rmsnorm_composition_identity()`.

**2. The RMSNorm direction-preservation claim is correct.**

MATH.md Section 2.2 states RMSNorm preserves direction:

```
RMSNorm(y_i) = alpha * y_i / ||y_i||_RMS
```

This is a positive scalar multiple of y_i, so the direction is preserved exactly.
The implementation in `loudness_fix.py` lines 72-75 correctly computes this.

**3. The magnitude equalization claim for RMSNorm is correct.**

MATH.md Section 2.2 states that `||RMSNorm(y_a)|| / ||RMSNorm(y_b)|| = 1`.
Since both are scaled to the same `target_rms / sqrt(d)`, this is exact.
Verified by `test_rmsnorm_equalizes_magnitude()`.

**4. The magnitude loss formulation is correct.**

```
L_mag = mag_coeff * (RMS(output) - target_rms)^2
```

This is a standard squared deviation penalty. The gradient flows correctly through
the RMS computation (tested in `test_matched_magnitude_gradient_flows()`).

**5. The scalar calibration diagnostic reasoning is sound.**

MATH.md Section 3.3 correctly states: if scalar calibration matches full
calibration, loudness is the sole issue; if not, direction/function-space
gap matters. The learned scales of ~0.99 confirm the magnitudes are already
matched, and the 6% gap between scalar and full calibration confirms
direction matters. This logic is tight.

### What does not hold or has issues

**1. MATH.md Section 2.4 "Why This Fails" contains a subtle error in reasoning.**

The section argues RMSNorm fails because it "destroys the ABSOLUTE magnitude of
each pool's output, which carries information." While this is a real effect, the
stated mechanism is incomplete. Consider: in the composed model, the pool output
is `Pool_A(x) + Pool_B(x)`. Even without RMSNorm, this sum already distorts the
absolute magnitude that downstream layers expect (it roughly doubles it). The
paper elsewhere argues this doubling is NOT the main problem (scalar calibration
at ~0.99 confirms it). So the magnitude-preservation argument for why RMSNorm
fails is internally inconsistent with the paper's own finding that magnitude
is not the bottleneck.

The more likely explanation for RMSNorm's catastrophic failure (+22.4%) is that
it interacts badly with the residual stream's existing layer norm. In a
transformer, the layer norm before the next attention block already normalizes
the residual. Adding a second normalization on the pool output before it enters
the residual creates a double-normalization pathology: the signal is normalized
twice (once by RMSNorm, once by the downstream LayerNorm), which can collapse
the effective dynamic range.

**Severity: Low.** The conclusion (RMSNorm is harmful) is correct even if the
stated mechanism is incomplete. But the explanation should be revised for
intellectual honesty.

**2. MATH.md Section 5.2 "Why Weight Averaging Preserves Network Expectations"
contains an argument that is correct but insufficient.**

The section argues that weight averaging works because it maintains the same
output dimensionality (P capsules) as the pretrained base. This is true but
the argument proves too much. By this logic, any method that produces a P-dimensional
pool should work equally well. The actual reason weight averaging works is that
it produces weights close to the pretrained base (the averaged deltas are smaller
than individual deltas), keeping the composed function within the pre-training
distribution the downstream layers were calibrated for. The dimensionality
argument is necessary but not sufficient.

**Severity: Low.** The finding (weight averaging is best zero-shot method) is
well-supported by the data regardless of the mechanistic explanation.

**3. MATH.md Section 2.4 has an implicit assumption about the scale of alpha.**

The default `target_rms = 1/N_pools` is chosen so the sum has "roughly the same
magnitude as a single pool." But MATH.md Section 2.4 (scale of composed output)
shows this only holds if pool outputs are uncorrelated. With correlated pools
(likely for similar domains), the magnitude depends on the correlation structure.
The experiment tested `target_rms` values of 0.25, 0.5, and 1.0, but ALL performed
catastrophically (+22-63%), suggesting the failure mode is not about the specific
alpha value. This is adequately acknowledged by the sweep across target_rms values.

**Severity: None.** Covered by the experiment.

**4. The "function-space gap" framing, while directionally correct, is not
rigorously defined.**

MATH.md Section 5.3 claims "the output direction distribution is fundamentally
different" between concatenated (2P-dimensional hidden space) and single-pool
(P-dimensional hidden space) outputs. But both produce outputs in R^d -- the
output space is the same d-dimensional space regardless of hidden dimension.
The claim should be about the RANK or the span of the output, not the
dimensionality of the output space. A pool with 2P capsules can represent rank-2P
outputs while a pool with P capsules can represent rank-P outputs. The downstream
layers were trained to handle rank-P outputs; rank-2P outputs are out-of-
distribution even at the same magnitude.

**Severity: Moderate.** The concept is right but the mathematical statement is
imprecise. The MATH.md should distinguish between hidden dimension (different)
and output dimension (same) and frame the issue in terms of output rank/span.

### Worked example in Section 6

**The numerical example is correct but the conclusion is misleading.**

The example shows Pool_A(x) = [0.71, 0, 0, 0] and Pool_B(x) = [0.35, 0, 0, 0],
so the concatenated output is [1.06, 0, 0, 0]. But both pools happen to produce
output in the exact same direction (the first basis vector). This is a degenerate
case. In practice with random initialization and different fine-tuning, the pools
would produce outputs in different directions. The example illustrates the
magnitude doubling problem but does not illustrate the directional mismatch
that the paper argues is the real issue.

**Severity: Low.** The example is technically correct; it just does not illustrate
the paper's main point (directional mismatch).

---

## Novelty Assessment

### Prior art

**1. Model merging / task arithmetic (Ilharco et al., 2023).**
The paper correctly cites this. Weight averaging is a standard model merging
technique. The finding that weight averaging outperforms concatenation for
zero-shot composition is not novel in isolation (it is the expected result from
the model merging literature), but the context (comparing weight-space vs
function-space composition of ReLU MLP pools) is specific enough to be a
valid micro-scale finding.

**2. TIES-Merging (Yadav et al., 2023) and DARE.**
Cited but not tested. The paper notes these as potential improvements over
simple averaging at macro scale. This is an appropriate deferral.

**3. Per-pool normalization in MoE composition.**
The idea of normalizing expert outputs before combining is standard in MoE
(softmax normalization in standard MoE is conceptually similar). The specific
application of RMSNorm to independently-trained pool outputs is a reasonable
variant to test, and the finding that it is catastrophically harmful is a
useful negative result.

**4. Matched-magnitude training.**
Auxiliary losses constraining activation magnitudes have precedent in various
contexts (batch normalization, weight normalization, spectral normalization).
Applying it to ensure composition compatibility is a reasonable but not novel idea.

### What is novel

**The diagnostic methodology is the main contribution.** The three-intervention
protocol (RMSNorm / scalar calibration / matched-magnitude training) is a
well-designed scientific experiment that isolates the magnitude hypothesis from
the function-space hypothesis. The scalar calibration diagnostic (8 params, scales
~0.99) is particularly clean -- it provides a definitive answer to the question
"is loudness the problem?" The answer (no) is clear and well-supported.

**The falsification of the loudness hypothesis is a genuine finding.** Before
this experiment, the natural assumption (and the one stated in IDEA-RELU-ROUTER.md)
was that zero-shot composition degrades because pools produce different magnitudes.
This experiment shows that assumption is wrong, which redirects future research.

### Delta over existing work

The delta is primarily diagnostic: providing experimental evidence that
function-space gap (not magnitude mismatch) is the bottleneck for zero-shot
MLP composition. This is a useful micro-scale finding that should inform the
composition protocol going forward.

---

## Experimental Design

### Does it test the stated hypothesis?

**Yes, effectively.** The hypothesis is: "the +6.6% zero-shot composition
degradation is caused by activation magnitude mismatch (loudness)." The
experiment tests this with three interventions of increasing directness:

1. **RMSNorm**: If loudness is the problem, equalizing magnitudes should help.
   Result: catastrophically worse (+22.4%). Hypothesis weakened.

2. **Scalar calibration**: If loudness is the problem, learning optimal
   per-pool scales should close the gap to full calibration. Result: scales
   are ~0.99 (already matched), 6% gap vs full calibration remains. Hypothesis
   falsified.

3. **Matched-magnitude training**: If magnitude drift during fine-tuning
   causes composition degradation, preventing drift should help. Result: RMS
   perfectly matched but +7.3% degradation (worse than plain zero-shot +4.3%).
   Hypothesis falsified from training side too.

**The three-pronged attack on the loudness hypothesis is well-designed and
convincing.**

### Controls

**Adequate.** The experiment includes:
- Joint training baseline (upper bound)
- Plain zero-shot concatenation (the known +4.3% problem, slightly different
  from the +6.6% in the original relu_router paper due to different seeds)
- Full calibration (continued training, explicitly labeled unfair)
- Weight averaging (standard model merging baseline)
- Capsule MoE with router calibration (architecture comparison)
- Multiple RMSNorm target_rms values (0.25, 0.5, 1.0)

### Seed coverage

**Good.** Three seeds (42, 123, 7) with aggregate statistics. Standard deviations
are reported. The main findings are consistent across seeds.

### Issues with experimental design

**1. The +4.3% baseline differs from the original +6.6% claim.**

The original relu_router PAPER.md reported zero-shot composition at +5.0%
(3-seed average) and "+6.6%" appears to have been an earlier single-seed or
2-seed result. The loudness_fix PAPER.md reports +4.3% for the same method.
The discrepancy is explained by the PAPER.md note that different seeds produce
different results (the previous experiment showed different exact numbers but
the same pattern). However, the experiment title and motivation reference "+6.6%"
while the actual measured baseline is +4.3%. This should be noted.

**Severity: Low.** The direction of the finding is not affected. Whether the
baseline is +4.3% or +6.6%, the loudness hypothesis is still falsified by the
scalar calibration diagnostic.

**2. Matched-magnitude training uses a different base model than the
ReLU Router experiments.**

The matched-magnitude experiments create a `MatchedMagnitudeGPT` model
(line 271-277 of `test_composition.py`), which is separately pretrained
from the `ReLURouterGPT` base model used for the other experiments. This means
the matched-magnitude results are not directly comparable to the plain zero-shot
results -- they start from a different pretrained base. Any difference could be
due to the different base model rather than the magnitude constraint.

The experiment should have used the SAME pretrained base for both the standard
fine-tuning and matched-magnitude fine-tuning paths. The matched-magnitude loss
could be added to the fine-tuning phase of the standard ReLU base model.

**Severity: Moderate.** The matched-magnitude result (+7.3% vs joint) being WORSE
than plain zero-shot (+4.3%) could partly be explained by the matched-magnitude
base being a different (possibly worse) pretrained model rather than by the
magnitude constraint being harmful. The MatchedMagnitudeGPT model includes the
magnitude loss in `aux_loss()` even during pretraining (when `target_rms` is
None, the magnitude loss returns 0, so this is actually fine for pretraining).
But during fine-tuning, the magnitude loss competes with the primary NTP loss,
potentially degrading fine-tuning quality. The paper acknowledges this ("the
auxiliary loss competes with the primary NTP loss") but does not separate the
two effects.

A cleaner experiment would be:
1. Pretrain the standard ReLU base
2. Fine-tune with magnitude constraint (using the standard base)
3. Fine-tune without magnitude constraint (same standard base)
4. Compare zero-shot composition of both

**3. The scalar calibration uses manual SGD despite creating an Adam optimizer.**

In `scalar_calibrate()` (relu_router's test_composition.py, lines 196-211), an
Adam optimizer is instantiated on line 197 but never used. The actual update on
line 210 is `scales_flat = [s - lr * g for s, g in zip(scales_flat, grads)]`,
which is plain SGD. While SGD with lr=1e-2 for 100 steps on 8 parameters is
reasonable, the unused Adam optimizer is confusing and the choice of SGD over
Adam should be justified. Adam might find better optima for these 8 parameters.

**Severity: Low-Moderate.** If SGD fails to find optimal scales (gets stuck at
local optimum near 1.0), the finding "scales are ~0.99" could be an artifact of
optimization failure rather than a true finding. However, the loss surface for
8 scalar parameters is likely convex (or at least well-behaved), so SGD should
be adequate. The consistency across 3 seeds (all ~0.99) provides additional
confidence. Still, replacing SGD with Adam or running for more steps would
strengthen the diagnostic.

**4. The `measure_pool_rms` function has dead code.**

Lines 100-111 of `test_composition.py` contain a loop that does nothing
(the inner loop body is `pass`). The actual measurement starts at line 114
with a second loop. This dead code does not affect results but suggests the
function was written hastily.

**Severity: None.** Cosmetic issue.

**5. RMSNorm applied per-pool is not the only normalization strategy.**

The experiment tests per-pool RMSNorm but does not test normalizing the
COMPOSED output (sum of pools). A single RMSNorm on Pool_A(x) + Pool_B(x)
would be less aggressive than per-pool normalization and might avoid the
catastrophic failure. This is a missing control.

**Severity: Low.** The scalar calibration diagnostic already proves magnitude
is not the issue, so another normalization variant would not change the
conclusion. But it would be a more informative comparison.

**6. Weight averaging comparison is slightly unfair in parameter count.**

Weight averaging produces a model with P capsules; concatenation produces a
model with 2P capsules. At 2P, the model has more capacity. Despite this
capacity advantage, concatenation (+4.3%) underperforms weight averaging (+1.5%).
The paper notes this (PAPER.md Section "Key Observations" point 4), which is
good. But a fairer comparison would be weight averaging at P vs concatenation
at P (half each domain's capsules randomly selected before concatenating). This
is a minor point and the paper's interpretation is defensible.

**Severity: None.** The paper correctly notes the parameter count difference.

---

## Integration Risk

### Compatibility with VISION.md

The experiment advances VISION.md in a specific way: it resolves item 7
("Loudness fix for zero-shot composition") by falsifying the loudness
hypothesis and identifying the function-space gap as the real bottleneck.
This is useful directional information.

**Weight averaging as zero-shot default.** The paper proposes weight averaging
(+1.5%) as the zero-shot composition method. This is compatible with VISION.md's
architecture, which uses softmax routing + concatenation + calibration as the
validated protocol. Weight averaging provides a simpler fallback when calibration
data is unavailable.

**Does not conflict with existing architecture.** The loudness_fix experiment
does not propose architectural changes. It clarifies the nature of the
composition problem and suggests weight averaging as an addition to the
composition toolkit. This is additive, not conflicting.

### Risk: Weight averaging is lossy and has known limits

Weight averaging is a standard model merging technique with well-studied
limitations. At macro scale with truly diverse domains (Python vs JavaScript),
the fine-tuned weights may diverge so much that averaging destroys both domains'
specializations. TIES-Merging and DARE address this by resolving sign conflicts
and dropping redundant parameters before merging. The paper acknowledges this
in the Limitations section, which is appropriate.

### Risk: The "function-space gap" diagnosis does not come with a cure

The experiment successfully identifies the problem (function-space gap) but
the only solution it offers is weight averaging, which is a workaround (avoiding
the gap by staying in weight space) rather than a fix (closing the gap). The
PAPER.md acknowledges this and suggests future directions (mutual information
minimization, shared subspace constraints, progressive composition during
training). These are reasonable next steps.

---

## Macro-Scale Risks (advisory)

**1. Weight averaging will likely fail with diverse domains.**
At micro scale, a-m vs n-z names are very similar domains. Weight averaging
works because the fine-tuned weights have not diverged far from the base.
With Python vs JavaScript (or code vs prose), the weight deltas will be much
larger and potentially conflicting. Simple averaging will destroy specialized
features. TIES or DARE merging will be needed, and these require hyperparameter
tuning that reduces the "zero-shot" appeal.

**2. The function-space gap may grow nonlinearly with domain diversity.**
If the independently-trained pools learn truly incompatible representations
at macro scale, neither averaging nor concatenation may work without
significant calibration. The current +4.3% concatenation gap at micro scale
(with very similar domains) could become 20%+ with diverse domains.

**3. Scalar calibration diagnostic should be re-run at macro scale.**
The finding that scales are ~0.99 at micro scale does not guarantee the same
at macro scale. With more diverse domains, the magnitude mismatch could become
the real bottleneck, and the loudness hypothesis could flip from false to true.
This should be explicitly tested at macro scale.

---

## Code Quality

The implementation is clean and well-documented. Specific observations:

**Good practices:**
- All models have `aux_loss()` methods for consistent training interface
- Composition utilities are factored into reusable functions
- The experiment runs all methods under identical conditions (same base model,
  same data splits, same random seeds)
- The test suite covers forward shape, composition identity, gradient flow,
  and magnitude loss behavior

**Issues:**
- Dead code in `measure_pool_rms` (lines 100-111 of test_composition.py)
- Unused Adam optimizer in `scalar_calibrate` (line 197 of relu_router's
  test_composition.py)
- The `RMSNormComposedPool._rms_norm` method uses `self.target_rms` as a scale
  factor but the variable name suggests it is a target for matching, which is
  confusing given the experiment's other use of "target_rms" in
  MatchedMagnitudePool to mean the RMS value to match

---

## Verdict

**PROCEED**

This experiment is well-designed, the central finding (loudness hypothesis
falsified) is convincingly supported by multiple lines of evidence, and all 6
required fixes from the previous review have been addressed. The three-pronged
diagnostic methodology (RMSNorm / scalar calibration / matched-magnitude)
provides a clean answer to a well-posed question.

### Strengths

1. **Clean falsification.** The scalar calibration diagnostic (scales ~0.99,
   consistent across 3 seeds and 8 parameters) is the strongest single piece
   of evidence. It definitively shows that magnitude is not the bottleneck.

2. **Comprehensive controls.** 13 conditions tested across 3 seeds, with
   multiple baselines and kill threshold analysis.

3. **Previous review fixes addressed.** All 6 required fixes from the
   relu_router review have been implemented, showing responsive research.

4. **Honest reporting.** The paper labels its own hypothesis as "falsified"
   and correctly identifies weight averaging as coming from the model
   merging literature rather than claiming it as a novel contribution.

5. **Clear implications.** The paper provides a concrete recommendation
   (use weight averaging for zero-shot, concatenation + calibration for
   quality ceiling) that advances the project.

### Issues to address (non-blocking)

1. **Revise MATH.md Section 2.4** to acknowledge that the "magnitude carries
   information" argument is inconsistent with the paper's own finding that
   magnitude is not the bottleneck. The RMSNorm failure is more likely due to
   double-normalization pathology (RMSNorm + downstream LayerNorm) than to
   information loss in magnitude.

2. **Revise MATH.md Section 5.3** to distinguish between hidden dimension
   (different: P vs 2P) and output dimension (same: d). The function-space
   gap should be framed in terms of output rank/span, not output dimensionality.

3. **Clean up `measure_pool_rms`** in test_composition.py (remove dead code
   at lines 100-111).

4. **Fix the unused Adam optimizer** in `scalar_calibrate` (either use it or
   remove it; justify the choice of SGD for scalar optimization).

5. **Note the baseline discrepancy**: the experiment references "+6.6%
   degradation" in its motivation but the measured baseline is +4.3%. Clarify
   that the earlier number was from a different seed configuration.

6. **Consider re-running matched-magnitude from the same base**: using a
   separately pretrained MatchedMagnitudeGPT base confounds the comparison.
   A cleaner design would fine-tune the same ReLU base with and without
   magnitude constraint. This is advisory; the current results are still
   directionally informative since the matched-magnitude result (+7.3%)
   is worse than the unconstrained result (+4.3%), which would require
   the matched-magnitude base to be dramatically worse to explain away.
