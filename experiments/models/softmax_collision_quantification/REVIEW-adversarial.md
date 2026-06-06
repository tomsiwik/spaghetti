# Peer Review: Softmax Collision Quantification (v2, Post-Revision)

## NotebookLM Findings

Skipped (CLI not authenticated). Review proceeds from direct reading of MATH.md, PAPER.md, implementation code, and results.json.

## Mathematical Soundness

### Improvements from v1

The v1 review identified an incorrect approximation (p_(1) - p_(2) ~ g/(N*T)) and a flawed theoretical derivation. The revised MATH.md and PAPER.md have removed this derivation entirely and now present the scaling law as purely empirical. This is the correct response -- an honest empirical characterization is better than a wrong theory. Credit to the authors for this fix.

### Remaining issues (minor)

1. **Temperature effect description is imprecise but not wrong.** MATH.md line 89 states the gap "grows as approximately exp(g/T) / exp(g), which is superlinear in 1/T." This is a ratio, not an absolute gap. The actual probability gap depends on the full partition function Z. However, the directional claim (T<1 amplifies gaps) is correct, and the paper does not build quantitative predictions on this approximation.

2. **Scaling law overfit is properly acknowledged.** MATH.md and PAPER.md both note "4 data points for 2 parameters" and call the extrapolation "speculative." The r^2=0.959 caveat is explicit. This was a required fix from v1 and is now handled honestly.

3. **Dual-temperature decomposition arithmetic checks out.** Baseline C@T=1.0 = 0.589. T=0.5 C@T=1.0 = 0.472, so training effect = 0.589 - 0.472 = 0.117. T=0.5 C@T=0.5 = 0.245, so inference effect = 0.472 - 0.245 = 0.227. Total = 0.344. Reported: 0.117 + 0.227 = 0.344. These match the raw results.json data, verified independently.

### t-test implementation quality

The custom `_t_cdf_approx` function (run_experiment.py line 145) uses an ad hoc approximation for the Student's t CDF. For small df (5 seeds => df~6-8), this approximation may be inaccurate. However, because all reported p-values are >0.4 and the KC2 threshold requires p<0.05, even a substantial error in the approximation would not change the conclusion. The implementation is adequate for the purpose.

## Novelty Assessment

### What the v2 revision properly addresses

The paper now explicitly acknowledges:
- Switch Transformers, DeepSeek-V3, ReMoE, and softmax bottleneck as prior art for the collision problem itself
- The contribution is "quantitative measurement, not discovery"
- Temperature scaling is a well-known technique (knowledge distillation, RL)

This addresses the v1 critique about overstated novelty. The remaining contribution is:
1. An empirical scaling law for collision rate vs N (modest but useful)
2. The dual-temperature decomposition showing T=0.5's effect is 2/3 inference sharpening + 1/3 training dynamics (genuinely informative)
3. The negative result on quality impact (valuable -- saves others from pursuing this path at micro scale)

### Delta assessment

The dual-temperature decomposition is the most original element. No prior MoE work I am aware of separates temperature's training effect from its inference effect using this methodology. It is a small but clean methodological contribution.

## Experimental Design

### What was fixed from v1

1. **Dual-temperature measurement**: Now measures collision rates at both T=1.0 and training temperature for all models. This was the critical methodological fix from v1 and is implemented correctly (verified in run_experiment.py lines 301-308).

2. **5 seeds for Phase 2**: Up from 3. P-values computed via Welch's t-test. KC2 correctly marked as KILLED (p>0.5 for all mitigations).

3. **T=2.0 anomaly resolved**: With 5 seeds, T=2.0 @T=1.0 = 0.587 vs baseline 0.589. The v1 single-seed artifact is gone.

4. **N-capsules confound acknowledged**: Listed in Limitations (PAPER.md, Limitation 3). The explanation is thorough -- notes that N=64 has only 4 capsules/group vs 32 at N=8, creating an expressivity confound.

5. **Incorrect approximation removed**: The flawed g/(N*T) theory is gone. Scaling law is presented as purely empirical.

6. **Novelty claims toned down**: Paper explicitly positions contribution as "quantitative measurement, not discovery."

### Remaining design concerns (non-blocking)

1. **Phase 1 still uses 3 seeds.** The paper argues this is sufficient for KC1 (monotonic increase) since the direction is clear at all epsilons. This is reasonable -- the monotonic trend is strong enough that adding seeds would not change the conclusion. Not blocking.

2. **Layer-level collision rates still averaged.** The measurement function (run_experiment.py line 97) accumulates collision counts across all 4 layers. Different layers likely have different collision profiles. Per-layer breakdown would be informative but is not required for the stated hypothesis.

3. **N-capsules confound is acknowledged but not controlled.** A proper control (constant capsules_per_group, varying total params) was suggested in v1 but not implemented. This is acceptable within micro constraints -- the experiment notes this limitation clearly and does not claim the scaling is purely from softmax compression. The confound is real but does not invalidate the directional finding.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry correctly reflects the revised results:
- Status: "proven" -- appropriate because KC1 passes decisively
- Evidence includes KC2 KILLED with p-value
- Evidence notes dual-T decomposition

The "proven" status refers to KC1 (collision scaling). KC2 being killed is a negative result that is itself valuable information. The combined status is appropriate: the hypothesis "softmax collisions scale with N" is proven; the hypothesis "mitigations help quality" is killed. The experiment answered its questions.

## Numerical Verification

I verified the following numbers from results.json against PAPER.md:
- Baseline val loss mean: 0.5187 (correct)
- T=0.5 val loss mean: 0.5172 (reported 0.5171, rounding difference, acceptable)
- T=0.5 C@T=1.0 mean: 0.471 (reported 0.472, rounding, acceptable)
- T=0.5 C@T=0.5 mean: 0.245 (correct)
- T=2.0 C@T=1.0 mean: 0.587 (correct)
- Phase 1 collision rates at all N values: all match within rounding

No discrepancies found.

## Macro-Scale Risks (advisory)

1. **Temperature-load balance interaction remains the primary risk.** T=0.5 concentrates tokens on fewer experts. At macro scale with real capacity constraints, this may cause token dropping. DeepSeek-V3's bias terms are specifically designed to avoid this problem. This is noted in the paper.

2. **The negative KC2 result may not transfer to macro.** At micro scale with homogeneous data, the model compensates for routing ambiguity through attention and residual paths. At macro scale with diverse domains, routing discrimination may matter more, and collision mitigation could show quality benefits. The paper correctly notes this in its macro-scale discussion.

3. **ReLU routing (ReMoE) is the structural alternative.** The paper recommends evaluating ReLU routing as a next step, which is the right call. Temperature tuning is a band-aid for a softmax limitation that ReLU avoids entirely.

## Verdict

**PROCEED**

The revision addressed all 6 required fixes from the v1 review:

| Fix | Status | Quality |
|-----|--------|---------|
| 1. Dual-temperature measurement | Done | Correctly implemented, numbers verified |
| 2. Statistical significance (5 seeds) | Done | KC2 properly killed at p>0.5 |
| 3. Remove incorrect approximation | Done | Scaling law now purely empirical |
| 4. Acknowledge N-capsules confound | Done | Thorough discussion in Limitations |
| 5. Explain T=2.0 anomaly | Done | Resolved with additional seeds |
| 6. Tone down novelty claims | Done | Contribution properly scoped |

The experiment is now honest about what it shows and what it does not. KC1 (collision scaling with N) is well-supported by the data. KC2 (quality benefit from mitigation) is properly killed. The dual-temperature decomposition is a clean methodological contribution. The paper acknowledges prior art, limitations, and confounds.

The remaining limitations (layer-averaged collision rates, N-capsules confound not controlled, Phase 1 at 3 seeds) are acceptable within micro-experiment constraints and are explicitly discussed. No blocking issues remain.
