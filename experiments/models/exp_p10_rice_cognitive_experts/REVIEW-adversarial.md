# REVIEW — exp_p10_rice_cognitive_experts

## Verdict: PROCEED (KILLED)

Clean kill. All testable predictions matched. The experiment was correctly designed
as a predicted-failure verification of Finding #528, and the predictions held.

## What's Right

1. **Prediction table complete and honest.** P1-P4 all PASS, P5 correctly marked
   N/A due to K1 prerequisite failure. No fabrication — results.json confirms
   max thinking nPMI = 0.104 (layer 19), 0 layers above 0.3, layer scalar
   std = 0.222, norm CV = 0.403.

2. **Kill criteria properly cascaded.** K1461 FAIL (0 layers > 0.3 threshold),
   K1462/K1463 correctly not tested since K1 is prerequisite. No wasted compute.

3. **Impossibility structure is strong.** Four independent reasons RICE fails on
   dense 4-bit models: (a) no discrete routing for nPMI, (b) quantization noise
   floor exceeds signal, (c) thinking tokens empirically noise under 4-bit, (d)
   layer scalars already encode importance. Each alone is sufficient.

4. **Novel finding about layer scalars.** The inverse scalar-norm correlation
   (model suppresses high-norm layers, amplifies low-norm layers) is genuinely
   interesting and wasn't part of the original hypothesis. Worth recording.

## Minor Issues (Non-blocking)

1. **Theorem 1 is an argument, not a theorem.** The O(L·ε_q) bound is a
   loose heuristic — actual error propagation through nonlinear layers is
   more complex than summation. But the prediction it makes (nPMI < 0.3
   everywhere) is correct, so the looseness doesn't matter for the kill.

2. **Sample size (n=30 profiles) is modest.** nPMI estimates from 30 samples
   have non-trivial variance. However, the max nPMI is 0.104 — so far from
   0.3 that even with perfect sampling it wouldn't cross the threshold.

3. **Correct-vs-incorrect nPMI not in kill criteria.** The secondary analysis
   (max = 0.078) is mentioned but not formally predicted. This is fine since
   it was explicitly labeled as exploratory.

## Status Assessment

**KILLED** is the correct status. This is a Type 1 (verification) experiment
where the proof predicted failure and the experiment confirmed it. The
impossibility structure (RICE requires MoE discrete routing + quantization
noise floor) is well-grounded.

## Downstream Implications

- exp_p10_reasoning_adapter should be blocked or reconsidered — its premise
  of reinforcing cognitive layers is invalid
- Layer scalar architecture is worth noting as a potential signal for future
  efficient inference work (layer skipping)
