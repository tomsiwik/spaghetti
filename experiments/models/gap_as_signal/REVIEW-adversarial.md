# Peer Review: Gap-as-Signal

## NotebookLM Findings

Skipped (tooling not authenticated). Review conducted through direct code and math inspection.

## Mathematical Soundness

### Projection Method: Correct

The Gram-Schmidt projection (MATH.md Section "Projection Method") is mathematically sound:

```
b_proj(c) = c * ||b|| * a_hat + sqrt(1 - c^2) * ||b|| * b_perp_hat
```

- cos(a, b_proj) = c: verified by construction (inner product with a_hat gives c * ||b||)
- ||b_proj|| = ||b||: verified (sum of squares = c^2 * ||b||^2 + (1-c^2) * ||b||^2 = ||b||^2)
- The code implementation (lines 86-128 of test_gap_as_signal.py) correctly implements this

No issues here.

### Gap Definition: Correct but Underspecified

The CE gap and KL gap are standard and computed correctly in `measure_function_space_gap()`. The KL direction is KL(joint || composed), which measures how much information is lost when approximating the joint model with the composed model. This is the correct direction for this hypothesis.

### Correlation Claim: Methodologically Problematic

**Issue 1: Non-independence of data points inflates r^2.**

The experiment produces 7 cosine levels x 3 seeds = 21 data points. The correlation analysis (lines 556-565) pools ALL individual trial results. However, within each seed, the 7 trials share:
- The same base model
- The same expert A (identical deltas across all 7 cosine levels)
- The same joint model baseline
- The same validation data

Only expert B's projection changes. These 7 points within a seed are not independent observations -- they are a controlled sweep on a single pair of experts. The effective sample size is closer to 3 (seeds), not 21. Computing r^2 on 21 points that share extensive infrastructure overstates confidence.

**Severity: MODERATE.** The trend is clearly monotonic in the summary table (every cosine level shows worse quality), so the direction is robust. But the r^2 = 0.74 number should not be reported without acknowledging the non-independence. A more honest analysis would report r^2 on the 3 seed-level means (7 points per regression on the mean curve), or use a mixed-effects model.

**Issue 2: Cosine values are non-uniformly spaced.**

The cosine levels {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9} are denser at low cosine and sparser at high cosine. The two extreme points (cos=0.7 and cos=0.9) dominate the correlation because they produce the largest absolute gaps. If those two points were removed, the correlation among cos in [0.0, 0.5] would be much weaker (the quality differences are only +2.1% to +2.8%, a 0.7pp spread across 5 levels).

**Severity: MODERATE.** This is a leverage effect, not fraud. But it means the practical claim "orthogonality matters at cos < 0.5" is much weaker than the r^2 suggests. The signal is really "don't be highly correlated (cos > 0.5)."

### The Mechanism Argument: Incomplete

MATH.md Section "Mechanism" claims that orthogonal deltas produce tokens where loss_A(x) != loss_B(x), giving the router strong gradient signal. This is intuitively reasonable but not formally derived. Specifically:

1. The claim that orthogonal weight-space deltas produce different per-token losses depends on the input distribution. If all inputs activate the same subspace (regardless of delta orientation), the losses could still be similar. The argument assumes input diversity across subspaces, which is likely true at scale but unverified at micro scale.

2. The paper conflates "gap between composed and joint" with "discriminability between experts." These are different quantities. The gap measures composed vs joint; the router needs to discriminate expert A vs expert B. The connection (larger gap implies more discriminable experts) is plausible but not proven. A counterexample: two experts could be highly discriminable (very different from each other) but their average could be close to joint -- producing a small gap but easy routing.

**Severity: LOW for micro.** The empirical results support the claim even if the theoretical mechanism is incomplete. But for the paper to claim a "mechanism," a formal derivation connecting weight-space cosine to per-token loss differential is needed.

## Novelty Assessment

### Prior Art Check

1. **Guo et al., "Advancing Expert Specialization" (NeurIPS 2025):** Enforces orthogonality during training. The gap-as-signal experiment differs: it measures the gap post-composition and uses it as a predictor, not a training objective. The novelty delta is the "gap as predictor of calibration quality" framing.

2. **FouRA:** Decorrelated LoRA subspaces for training-free merging. Similar observation (orthogonal = composable) but different mechanism (frequency domain decorrelation vs gap measurement).

3. **LoRA Soups (COLING 2025):** Discovered concatenation + calibration works but did not measure calibration quality as a function of adapter similarity. The gap-as-signal experiment fills this specific gap in the literature.

4. **Symphony-MoE:** Uses activation-based functional alignment for upcycling. Different setting (upcycling existing experts, not composing independent adapters).

**Assessment:** The specific claim "pre-calibration gap magnitude predicts post-calibration quality" appears novel. The broader observation "orthogonal adapters compose better" is well-known (FouRA, model merging literature). The novelty is in the *predictive* framing -- using the gap as a diagnostic before investing compute in calibration.

**Delta over existing work: SMALL but REAL.** The contribution is a diagnostic metric, not a new architecture.

## Experimental Design

### What It Tests vs What It Claims

**Claim:** "The function-space gap IS the routing signal."

**What is actually tested:** Higher cosine similarity between (projected) expert deltas correlates with worse calibration quality after a fixed budget of 300 steps.

**Gap between claim and test:**

1. **"IS the routing signal" is overclaimed.** The experiment shows correlation, not causation. It does not show that the router actually uses the gap signal. The router learns from per-token loss gradients; the gap is an aggregate statistic. The experiment would need to measure router gradient magnitudes at different cosine levels to establish the mechanism.

2. **Projection creates synthetic experts, not real ones.** At cos=0.9, expert B is 90% aligned with expert A in weight space. This is not "a correlated expert trained on similar data" -- it is expert A with 10% noise from B's perpendicular component. A real correlated expert (trained on overlapping data) might have different pathologies. The PAPER.md acknowledges this in Limitation 5, which is appropriate.

3. **top_k=2 with N=2 experts means the router always uses both experts.** Every token routes to both experts with some weight split. The router's only degree of freedom is the weight ratio. This is a very constrained setting. At N>2 with top_k=2, the router must also choose WHICH experts, which is a fundamentally harder problem. The current experiment tests "can the router learn the right mixing ratio" not "can the router learn the right expert selection."

**Severity of top_k issue: MODERATE.** With 2 experts and top_k=2, the router is really learning a per-token interpolation weight, not a routing decision. This is a much simpler problem than real MoE routing. The gap-as-signal claim for actual routing (selecting k from N >> k) remains untested.

### Controls

The experiment has reasonable controls:
- Joint model as quality baseline (good)
- Multiple cosine levels provide a sweep (good)
- 3 seeds (minimal but acceptable for micro)
- Norm preservation in projection (good -- isolates the variable)

Missing controls:
- No "random router" baseline (what quality do you get with fixed uniform weights and no calibration?)
- No measurement of router gradient magnitude to verify the mechanism
- No comparison with training-free composition (simple weight averaging at each cosine level)

### Statistical Rigor

The r^2 = 0.74 claim is borderline. As noted above, the 21 data points are not independent. More critically, no confidence intervals are reported. With only 3 seeds, the variance of the r^2 estimate itself is high. A bootstrap CI on r^2 (resampling seeds) would have only 3 samples -- too few for meaningful inference.

The monotonic trend in the summary table is the strongest evidence: every step up in cosine produces worse quality. This ordinal pattern is robust (probability of 7 monotonic values by chance: 1/7! = 0.0002). However, this is testing cos->quality, not gap->quality (the actual claim). The gap also increases monotonically with cosine (by construction of the projection), so the transitive inference works, but the "gap as predictor" is really "cosine as predictor" with an intermediate variable.

## Hypothesis Graph Consistency

The HYPOTHESES.yml kill criteria are:
1. "gap magnitude does NOT correlate with calibration quality (r^2 < 0.3)"

The experiment tests this with the caveat that "calibration quality" replaces "calibration speed" (acknowledged in the paper). The substitution is reasonable at micro scale. The r^2 = 0.74 exceeds 0.3, but see the statistical caveats above.

The node status was changed to "proven," which is appropriate for micro scale given that 3/3 kill criteria pass.

## Integration Risk

The gap-as-signal is a conceptual framing, not an architectural component. It integrates cleanly with VISION.md because it IS the central claim of VISION.md. No architectural conflict.

However, the claim has implications for the contribution protocol: "submit an expert that is maximally orthogonal." If the gap-as-signal framing is correct, then the orthogonality check in the protocol is justified not just as a compatibility check but as a quality predictor. This is a useful strengthening of the protocol's justification.

## Macro-Scale Risks (advisory)

1. **The cos < 0.5 regime may be all that matters at scale.** Real LoRA adapters trained on different domains naturally have cos ~ 0.004 at d=896. The interesting regime (cos in [0.0, 0.5]) shows only a 0.7pp quality difference at micro scale. If this compresses further at macro scale, the gap-as-signal framing becomes: "orthogonality doesn't matter much because everything is already orthogonal." The test would be informative only if artificially correlated adapters are created, which defeats the purpose.

2. **Calibration speed vs quality.** The micro experiment only measures quality. At macro scale with harder routing problems (N=20, top_k=2), calibration speed should become measurable and may tell a different story than quality after fixed budget.

3. **The mechanism claim (gap = gradient signal strength) needs router gradient measurement at macro scale.** Without it, the story remains correlational.

4. **N>2 is the real test.** With N=2, top_k=2, the router has a trivial job. The gap-as-signal claim for real MoE with N=8+ experts and top_k=2 selection is a different ballgame.

## Verdict

**PROCEED** (with noted caveats for the paper text)

The experiment demonstrates a clear, monotonic, reproducible relationship between expert orthogonality and calibration quality across 3 seeds and 7 cosine levels. The direction is robust: orthogonal experts compose better. The key metric (r^2 = 0.74) passes the kill criterion (r^2 >= 0.3) even accounting for the statistical caveats.

The following issues should be addressed in the PAPER.md text (not blocking for PROCEED, but required for intellectual honesty):

1. **Soften the r^2 = 0.74 claim.** Acknowledge that the 21 data points are non-independent (shared base model, expert A, and validation data within each seed). Report the mean-curve r^2 (computed on 7 cosine-level means) alongside the pooled r^2.

2. **Acknowledge the top_k=2 with N=2 limitation.** The router learns mixing weights, not expert selection. The gap-as-signal claim for actual routing (k << N) is untested. This is a key macro-scale test.

3. **Distinguish "cosine predicts quality" from "gap predicts quality."** The experiment shows both, but since gap is monotonically determined by cosine (via the projection), they are not independently informative. The novel claim is specifically "the gap measured before calibration predicts quality after calibration," which requires the gap to be computable without knowing the cosine. At scale with real (non-projected) experts, the gap would be the only available signal (you don't know the "true" cosine of real adapters relative to all others). Emphasize this practical use case.

4. **Add a sentence noting the leverage effect.** The cos >= 0.7 regime drives most of the correlation. The practical regime (cos < 0.3 for real adapters) shows small quality differences (+2.1% to +2.3%).
