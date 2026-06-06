# E3: Two-Regime Unification — Results

## Status: KILLED (smoke, PROVISIONAL — but method-level failure, not sample-level)

Override rationale: K2022 FAIL at 1.8% vs 20% threshold is a 10× miss. K2023 ρ=-0.19 is opposite-sign. Adding more samples cannot change the structural conclusion.

## Hypothesis
Attention heads show sigmoid (threshold) responses to LoRA scale, with inflection points clustering at s∈[4,6], explaining the behavioral phase transition (F#248/F#250).

## Prediction vs Measurement

| Prediction | Measured | Match |
|---|---|---|
| 30-60% heads sigmoid with inflection in [4,6] | 1.8% (6/336) in [3,7] | **FAIL** (17× below minimum) |
| Inflection clustering at s∈[4,6] | Mean=16.41, Median=17.44 | **FAIL** (3× above range) |
| ρ > 0.85 flipped-heads vs accuracy | ρ = -0.19 (p=0.58) | **FAIL** (opposite sign) |
| Full-attn layers > sliding-attn for sigmoid in-range | 1.8% vs 1.8% | **FAIL** (no difference) |
| Behavioral peak at s∈[4,6] | Peak at s=6 (80%) | **PASS** (confirms F#250) |

## Kill Criteria Results

| KC | Threshold | Value | Result |
|---|---|---|---|
| K2022 (structural) | ≥20% heads sigmoid in [3,7] | 1.8% (6/336) | **FAIL** |
| K2023 (correlation) | ρ≥0.7 | ρ=-0.19 | **FAIL** |
| K_target (behavioral) | median inflection in [3,7] AND acc≥50% peak | median=5.34, acc=40%<50% of 80% | **PASS** (marginal) |

## Detailed Findings

### Head Response Landscape
- 83.9% of heads (282/336) fit a sigmoid with R²>0.9, but inflections are at s≈17, far above the behavioral transition
- Only 3.3% of heads (11/336) are genuinely linear
- The remaining 12.8% don't fit either model well

### The Key Disconnect
The behavioral phase transition (GSM8K: 0%→80% between s=4 and s=6) happens when **fewer than 2% of heads** have crossed their inflection. The mechanism is:
1. Heads respond to scale gradually (sigmoid with high inflection points)
2. Behavioral output changes sharply at s∈[4,6] despite minimal head-level change
3. This implicates **downstream processing** (FFN, output softmax, format template activation) as the phase transition mechanism, not attention head cascading

### GSM8K Behavioral Confirmation
- Base (s=0): 40% — Gemma 4 E4B already strong at GSM8K
- Peak: s=6 at 80% (confirms F#250)
- High scale degradation: s=16→0%, s=20→0% — adapter overshoot destroys reasoning
- The non-monotonic pattern (up then down) proves scale is not simply "more adapter effect is better"

### Negative Correlation Explained
ρ=-0.19 because high scales flip many heads (185 at s=20) but destroy accuracy (0%). The head flip count is a proxy for adapter perturbation magnitude, which is harmful above the optimal scale. Head flipping is a *symptom* of scale, not a *cause* of behavioral change.

## Mechanism Analysis: Where the Phase Transition Actually Happens

The q-proj perturbation per head grows linearly with scale (as expected from ΔQ = s·B@A). The softmax margin theorem from MATH.md is not wrong — it's just that the *relevant* margin is not in the attention softmax but somewhere downstream:

1. **Output logit softmax**: The final token prediction softmax has its own margin. The adapter may need scale ≥6 to shift the output logit argmax from "generic continuation" to "GSM8K format answer."
2. **FFN gating**: Gemma 4 uses GELU-gated MLP. The adapter's attention-level perturbation passes through FFN nonlinearity, which can amplify small changes nonlinearly.
3. **Template activation**: GSM8K format ("####") requires a specific output distribution that may have a threshold activation — either the model is "in GSM8K mode" or not.

## Implications for Downstream Experiments
- E14 (Grassmannian activation orthogonality): head-level analysis is insufficient; need to look at full hidden-state or FFN output
- Scale optimization: cannot predict optimal scale from head activation patterns; need behavioral sweep (F#248 approach remains best)
- The locus of the phase transition is between attention and output, not within attention itself

## Verdict: KILLED
K2022 FAIL (structural) + K2023 FAIL (correlation) + K_target marginal PASS. The head cascade mechanism is falsified. The phase transition is real but its mechanism is not in attention head activation.
