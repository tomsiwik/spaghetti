# Scale Phase Transition: Mathematical Framework

## Type: Guided Exploration

**Proven framework:** LoRA perturbation theory (Hu et al., 2106.09685) + two-regime
behavioral model (Finding #249).
**Unknown:** The shape of the scale-behavior transition function f(s) for math reasoning.

## A. Failure Mode Identification

**Failure mode:** Treating scale as binary {low, high} when the transition is gradual.
Finding #249 established that s=2.0 gives 0% math gain and s=20.0 gives 700% math gain,
but the intermediate region s in [4, 16] is unexplored. If the transition is gradual
(sigmoid-like), intermediate scales could provide partial math reasoning while better
preserving knowledge domains. If it is sharp (step function), the two-regime model is
complete and the lookup table {low, high} is optimal.

## B. The Right Question

Not: "What is the best scale for math?"
But: "Is the format-to-capability transition a phase transition (sharp) or a crossover
(gradual)?"

This determines:
1. Whether the 3-value lookup table {1, 4, 20} needs refinement
2. Whether "partial math reasoning at moderate scale" is achievable
3. The mathematical nature of LoRA perturbation effects (linear, threshold, or sigmoidal)

## C. Prior Mathematical Foundations

**LoRA perturbation theory (Hu et al., 2021, arXiv:2106.09685):**
The output perturbation scales linearly with s: delta_y = s * B^T A^T x.
However, behavioral quality is NOT linear in delta_y because:
1. Attention pattern shifts are nonlinear (softmax is a threshold function)
2. Math reasoning requires activating multi-step chains (threshold phenomenon)
3. Token prediction changes may be discrete (correct digit vs wrong digit)

**Phase transitions in neural networks (Nanda et al., 2023, arXiv:2301.05217):**
"Progress measures for grokking" shows sharp phase transitions in reasoning capability
during training. The analogy: scale s may function like training time, with a critical
threshold s* where reasoning "grokks" into existence.

**Finding #249 (this project):**
Two data points: f(2) = 0.100 (base rate), f(20) = 0.800. The question is the shape
of f on [2, 20].

**Finding #238 (this project):**
Math behavioral quality at per-domain optimal s=20: 8/10 correct.
Base: 1/10 correct. These are the boundary conditions.

## D. Predictions

**Model 1 (Phase transition / step function):**
If math reasoning requires a critical perturbation magnitude to shift attention
patterns, there exists s* in [4, 16] such that:
- f(s) ~ 0.1 for s < s* (adapter too weak to activate reasoning)
- f(s) ~ 0.8 for s > s* (reasoning fully activated)
- The transition width is narrow (< 4 scale units)

Prediction: At least one pair of adjacent test scales (e.g., s=8 and s=12) will show
a jump of >= 0.4 in math accuracy.

**Model 2 (Gradual crossover / sigmoid):**
If reasoning activation is continuous, f(s) follows a sigmoid:
- f(s) = 0.1 + 0.7 * sigma((s - s_mid) / tau)
- With s_mid ~ 10 and tau ~ 3

Prediction: Math accuracy at s=10 will be ~ 0.45 (midpoint), and adjacent scales
show incremental improvements of 0.1-0.2 each.

**Model 3 (Two-threshold):**
If reasoning has two components (format + calculation), there may be two thresholds:
- s1 ~ 4-6: Format activation (chain-of-thought structure appears)
- s2 ~ 12-16: Calculation activation (correct numerical answers)

Prediction: Intermediate scales show chain-of-thought formatting WITHOUT correct
answers. Math accuracy at s=8 will be 0.1-0.3 (format without calculation).

**Quantitative prediction table:**

| Scale | Model 1 (step) | Model 2 (sigmoid) | Model 3 (two-thresh) |
|-------|----------------|--------------------|-----------------------|
| 1     | 0.10           | 0.10               | 0.10                  |
| 2     | 0.10           | 0.10               | 0.10                  |
| 4     | 0.10           | 0.12               | 0.10                  |
| 6     | 0.10           | 0.18               | 0.20 (format only)    |
| 8     | 0.10           | 0.30               | 0.20                  |
| 10    | 0.10           | 0.45               | 0.30                  |
| 12    | 0.80           | 0.58               | 0.50                  |
| 16    | 0.80           | 0.72               | 0.80                  |
| 20    | 0.80           | 0.78               | 0.80                  |

## E. Kill Criteria

**K1:** No scale in [4, 16] achieves math accuracy > 0.20. This means the transition
is fully sharp and occurs somewhere in [16, 20]. The two-regime model is sufficient.
(Result: lookup table {low, high} is optimal. No follow-up needed.)

**K2:** Math accuracy at ALL tested scales is either <= 0.20 or >= 0.60 (no intermediate
values exist). This confirms a pure phase transition with narrow width. (Result: binary
scale is sufficient.)

**K3:** Math accuracy is non-monotonic in scale (e.g., f(12) > f(16)). This would
indicate the perturbation model is wrong and scale effects are more complex than
magnitude scaling. (Result: revision of perturbation theory needed.)

## F. Assumptions and Breaking Conditions

**A1:** Math accuracy is monotonically non-decreasing in scale for s in [1, 20].
Breaking: if high scale occasionally hurts (non-monotonic), the perturbation
model is wrong.

**A2:** The transition shape is consistent across prompts (not prompt-dependent).
Breaking: if some prompts activate at s=4 and others at s=16, the "transition"
is an average of heterogeneous thresholds.

**A3:** n=10 prompts are sufficient to distinguish 0.1 from 0.4 accuracy.
With n=10, a binomial 95% CI for p=0.1 is [0.01, 0.32] and for p=0.4 is
[0.14, 0.71]. Distinguishable at the tails but marginal at intermediate values.

## G. Worked Example (single prompt)

Consider a math prompt "What is 15% of $240?"
- At s=2: adapter barely perturbs base. Model outputs prose without calculation.
  Answer extracted: None. Score: 0.
- At s=10 (sigmoid model): adapter partially activates. Model attempts
  "15% of 240 = ..." but may compute wrong. Score: 0 or 1.
- At s=20: adapter fully overrides. Model outputs "#### 36". Score: 1.

The per-prompt score is binary (0 or 1), so the transition curve is the
fraction of prompts that cross the activation threshold at each scale.

## H. Complexity and Architecture Connection

**Compute:** 9 scales x 10 prompts x 128 tokens = 11,520 tokens generation.
At ~100 tok/s per generation pass, ~2 min per scale. Total: ~20 min + overhead.
Model loaded once, base weights saved, adapter merged/restored per scale.

**Memory:** One model (~4GB) + one adapter (~2MB) + base weight backup (~2GB).
Well within 48GB. Single-domain (math only) means no domain switching overhead.

**Additional analysis:** Record not just binary correctness but also:
- Whether chain-of-thought format appears (qualitative indicator of format activation)
- The generated answer (to detect "close but wrong" at intermediate scales)

## Self-Test

1. **ONE property:** The shape of the scale-behavior transition function f(s).
2. **Existing theorems:** LoRA perturbation theory, phase transitions in reasoning
   (Nanda et al., 2301.05217).
3. **Specific numbers:** See prediction table above. Key: Model 1 predicts jump >= 0.4
   between adjacent scales; Model 2 predicts f(10) ~ 0.45.
4. **Falsification:** K1/K2 distinguish between models. K3 falsifies monotonicity.
5. **Hyperparameters:** 0 added. We are sweeping an existing parameter.
6. **Hack check:** No. This is a measurement experiment.
