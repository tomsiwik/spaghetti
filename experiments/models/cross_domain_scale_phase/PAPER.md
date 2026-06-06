# Cross-Domain Scale Phase Transition: Code and Medical

## Theorem (from MATH.md)

The critical LoRA scale s* at which behavioral capability activates may be universal
(architecture-dependent) or domain-dependent (task-dependent). Finding #250 established
s*_math in [4, 6] with a sharp step-function transition. This experiment tests whether
code and medical domains exhibit the same phenomenon.

## Hypothesis

**H1 (universal s*):** All capability domains share s* ~ 5 because the LoRA perturbation
magnitude needed to shift attention argmax is architecture-dependent, not task-dependent.

**H2 (domain-dependent s*):** Different tasks require different perturbation magnitudes.
|s*_code - s*_math| > 4.

**H3 (metric-dependent shape):** Discrete metrics (math correctness) show sharp steps;
continuous metrics (factual recall) show gradual transitions even if underlying capability
is sharp.

## What This Experiment Is

Scale sweep on code and medical domains, testing s = {1, 2, 4, 6, 8, 12, 16, 20} with
10 prompts each. Code evaluated by syntax validity (70%) + factual recall (30%). Medical
evaluated by factual recall against reference. Extends Finding #250 (math-only) to two
additional domains.

## Key References

- Hu et al. (2021) "LoRA" arXiv:2106.09685
- Finding #249: Two behavioral regimes (FORMAT vs CAPABILITY)
- Finding #250: Math phase transition at s*=[4,6], jump=0.60

## Predictions vs Measured

| Prediction | Predicted | Measured | Match |
|-----------|-----------|----------|-------|
| P1 (H1): Code jump >= 0.3 at s=[4,6] | >= 0.3 | max jump = 0.154 (s=1->s=2) | NO |
| P2 (H1): Medical jump >= 0.3 at s=[4,6] | >= 0.3 | max jump = 0.024 (s=16->s=20) | NO |
| P3 (H2): \|s*_code - s*_math\| > 4 | > 4 | 1.0 (code s*=4, math s*=5) | NO |
| P4: Code plateau >= 0.5 | >= 0.5 | 0.624 (at s=20) | YES |
| P5: Medical plateau >= 0.5 | >= 0.5 | 0.291 (at s=20) | NO |

**Neither H1 nor H2 is supported.** The phase transition phenomenon is math-specific.
Code shows no sharp transition. Medical shows no transition at all.

## Empirical Results

### Transition Curves

**CODE:**

| Scale | Score | Delta from base |
|-------|-------|----------------|
| base  | 0.419 | --             |
| 1.0   | 0.350 | -0.069         |
| 2.0   | 0.504 | +0.085         |
| 4.0   | 0.574 | +0.155         |
| 6.0   | 0.357 | -0.062         |
| 8.0   | 0.499 | +0.080         |
| 12.0  | 0.500 | +0.081         |
| 16.0  | 0.487 | +0.068         |
| 20.0  | 0.624 | +0.205         |

**MEDICAL:**

| Scale | Score | Delta from base |
|-------|-------|----------------|
| base  | 0.263 | --             |
| 1.0   | 0.268 | +0.005         |
| 2.0   | 0.284 | +0.021         |
| 4.0   | 0.281 | +0.018         |
| 6.0   | 0.277 | +0.014         |
| 8.0   | 0.284 | +0.021         |
| 12.0  | 0.285 | +0.022         |
| 16.0  | 0.267 | +0.004         |
| 20.0  | 0.291 | +0.028         |

### Key Observations

**1. No phase transition for code or medical.** The math domain showed a 0.60 jump
between s=4 and s=6. Code shows max jump of 0.154. Medical shows max jump of 0.024.
The phase transition phenomenon is NOT universal -- it is specific to math (or more
precisely, to discrete-answer tasks with GSM8K-style training format).

**2. Code is non-monotonic and noisy.** The code transition curve oscillates: 0.35,
0.50, 0.57, 0.36, 0.50, 0.50, 0.49, 0.62. The drop at s=6 (0.357) after s=4 (0.574)
is dramatic. This is the opposite of math, where s=6 was the activation point. Code
generation quality is highly sensitive to scale in a non-monotonic way.

**3. Medical adapter has near-zero effect.** Medical scores range from 0.267 to 0.291
across all scales, compared to base 0.263. The maximum delta is +0.028 at s=20. The
medical adapter learned essentially nothing useful for factual recall on these prompts.
This is an adapter quality problem, not a scale problem.

**4. Code per-prompt analysis reveals bimodal behavior.** Code prompts split into two
groups:
- Group A (prompts 2, 4, 7, 8, 9): Base already scores high (0.73-0.82). The adapter
  DISRUPTS these at some scales (e.g., prompt 7 drops from 0.81 base to 0.05 at s=6).
- Group B (prompts 0, 1, 3, 5, 6): Base scores low (0.02-0.08). The adapter sometimes
  activates capability (e.g., prompt 3 jumps from 0.04 to 0.79 at s=6).

The aggregate curve averages over these opposing effects, producing apparent noise.
This is fundamentally different from math, where ALL prompts benefited from the adapter.

**5. The "phase transition" in math was a format confound amplified by evaluation.**
Math has binary evaluation (exact numerical match). The adapter at s>=6 activates
GSM8K training format ("<<3*26=78>>, #### 78") which the regex parser can extract.
Code and medical have continuous evaluation (factual overlap, syntax parsing) that is
less sensitive to format changes. This suggests Finding #250's "phase transition" is
partially an artifact of the evaluation method interacting with format activation, not
a fundamental property of LoRA perturbation theory.

**6. Sigmoid fit confirms no transition for code.** Code sigmoid fit: s_mid=14.0,
tau=5.65, R^2=0.296. The extremely low R^2 and high tau (broad) confirm there is no
identifiable transition -- the data is simply noisy. Medical: sigmoid fit failed
entirely (insufficient signal).

### Kill Criteria Results

| Criterion | Result | Evidence |
|-----------|--------|----------|
| K1 (#660): Non-math domain shows jump >= 0.3 | **FAIL** | Code max jump = 0.154, medical = 0.024 |
| K2 (#661): s* differs by >4 units across domains | **FAIL** | max diff = 1.0 (code s*=4, math s*=5) |
| K3 (#662): All domains reach plateau >= 0.5 | **FAIL** | Medical max = 0.291 |

**All three kill criteria FAIL.** This is a negative but informative result.

## Interpretation

The experiment is "killed" in the sense that the phase transition is NOT universal.
But the finding is valuable:

1. **The math phase transition (Finding #250) is math-specific, not a general LoRA
   property.** It reflects the interaction between: (a) GSM8K format activation, (b)
   discrete numerical evaluation, and (c) the specific attention patterns math reasoning
   requires. Other domains do not share this structure.

2. **The binary scale model {FORMAT, CAPABILITY} from Finding #249 needs revision.**
   It was based on math/code/medical all being "capability" domains. But code and medical
   do not exhibit capability activation at any scale. Instead:
   - Math: sharp activation at s*=[4,6] (Finding #250, confirmed)
   - Code: noisy, non-monotonic, base already strong for some prompts
   - Medical: adapter has negligible effect at any scale

3. **Per-domain scaling remains necessary (Finding #249 still holds), but the rationale
   changes.** It is not that different domains need different activation thresholds --
   it is that different adapters have different quality levels and interact differently
   with base model capabilities.

## Limitations

1. **n=10 per domain per scale.** Small sample size amplifies noise, especially for code
   where per-prompt behavior is bimodal.

2. **Evaluation metrics differ across domains.** Math uses exact match (binary), code
   uses syntax+recall (continuous), medical uses factual recall (continuous). Cross-domain
   score comparisons are not meaningful -- only within-domain transition shapes matter.

3. **Medical adapter quality.** The near-zero effect may reflect poor adapter training,
   not scale irrelevance. A better-trained medical adapter might show a transition.

4. **Code evaluation is noisy.** The syntax checker is generous (finds code-like blocks
   in any text), and factual recall is a crude proxy for code quality. A better evaluation
   would use execution-based testing, but these prompts lack test cases.

## What Would Kill This

1. A well-trained medical adapter (verified on held-out data) showing a phase transition
   at some s* -- would indicate the null result here is an adapter quality artifact.
2. Execution-based code evaluation showing a sharp transition that the syntax metric
   missed.
3. A larger n (50+) showing the code oscillation is noise and a smooth transition exists.
