# Cross-Domain Scale Phase Transition: Mathematical Framework

## Type: Guided Exploration

**Proven framework:** LoRA perturbation theory (Hu et al., 2106.09685) + sharp phase
transition for math at s*=[4,6] (Finding #250).
**Unknown:** Whether code and medical domains exhibit the same sharp transition, and
whether the critical scale s* is universal or domain-dependent.

## A. Failure Mode Identification

**Failure mode:** Assuming the math phase transition generalizes to all domains.
Finding #250 showed a sharp step-function transition for math (s=4: 0.10 -> s=6: 0.70).
If code and medical have different transition shapes or locations, a universal binary
scale {FORMAT, CAPABILITY} would misroute: applying s=6 to a domain whose s* is at 12
would give FORMAT-regime behavior when CAPABILITY was intended.

**Why this is a real risk:** Code and medical are fundamentally different tasks:
- Math: exact numerical computation, discrete answer (correct/wrong)
- Code: syntactic generation, multi-line structured output
- Medical: factual recall, continuous quality (partial credit)

The attention patterns each adapter must shift are different. Different task structures
could require different perturbation magnitudes to cross their respective activation
thresholds.

## B. The Right Question

Not: "What is the best scale for code and medical?"
But: "Is the critical scale s* a property of the LoRA perturbation mechanism (universal)
or of the task-attention interaction (domain-dependent)?"

If universal: binary scale with s*=5 works for all capability domains.
If domain-dependent: routing needs per-domain scale selection, not just per-domain
adapter selection.

## C. Prior Mathematical Foundations

**LoRA perturbation theory (Hu et al., 2021, arXiv:2106.09685):**
Output perturbation delta_y = s * B^T A^T x scales linearly with s. But behavioral
quality is a nonlinear function of delta_y due to attention softmax thresholds.

**Finding #250 (this project):**
Math domain: f_math(s) is a step function with s*_math in [4, 6]. Jump of 0.60
between s=4 and s=6. Sigmoid fit degenerates to step (tau=0.1).

**Finding #249 (this project):**
At s=2: code accuracy 0.504 vs base 0.419 (delta = +0.085). This is non-zero,
unlike math (delta = 0). This suggests code may have a different transition shape --
possibly more gradual, since measurable improvement appears at low scale.

**Softmax threshold theory:** The attention softmax creates a winner-take-all dynamic.
For the adapter to change behavior, it must shift the argmax of attention logits.
The required perturbation magnitude depends on the margin between the top-1 and top-2
attention logits in the base model. Tasks that require subtle attention shifts (e.g.,
recalling a specific fact) may have smaller margins than tasks requiring wholesale
pattern changes (e.g., switching from prose to computation).

## D. Predictions

**Hypothesis H1: Universal transition (s*_code ~ s*_medical ~ s*_math ~ 5)**
All capability domains share the same critical scale because the LoRA perturbation
magnitude needed to shift attention argmax is architecture-dependent, not task-dependent.

Prediction: Code and medical both show jump >= 0.3 between s=4 and s=6.

**Hypothesis H2: Domain-dependent transition (|s*_code - s*_math| > 4)**
Different tasks require different perturbation magnitudes. Code may transition earlier
(it already showed improvement at s=2 in Finding #249). Medical may transition later
(factual recall requires more precise attention shifts).

Prediction: Code transition at s*_code in [2, 4], medical at s*_medical in [8, 16].

**Hypothesis H3: Gradual for continuous metrics, sharp for discrete metrics**
Math has binary evaluation (correct number or not), producing a sharp step in accuracy.
Code (syntax validity) and medical (factual recall) have continuous evaluation, so
even if the underlying capability transition is sharp, the measured curve may appear
more gradual.

Prediction: Code syntax accuracy shows a step (discrete metric), but medical factual
recall shows a sigmoid (continuous metric).

**Quantitative prediction table:**

| Scale | Math (F250) | Code (H1) | Code (H2) | Medical (H1) | Medical (H2) |
|-------|-------------|-----------|-----------|--------------|--------------|
| 1     | 0.10        | ~base     | ~base     | ~base        | ~base        |
| 2     | 0.10        | ~base     | 0.5+      | ~base        | ~base        |
| 4     | 0.10        | ~base     | 0.6+      | ~base        | ~base        |
| 6     | 0.70        | 0.7+      | 0.7+      | 0.6+         | ~base        |
| 8     | 0.80        | 0.7+      | 0.7+      | 0.6+         | ~base        |
| 12    | 0.80        | plateau   | plateau   | plateau      | 0.5+         |
| 16    | 0.70        | plateau   | plateau   | plateau      | 0.6+         |
| 20    | 0.80        | plateau   | plateau   | plateau      | plateau      |

## E. Kill Criteria (derived from predictions)

**K1 (#660):** At least one non-math domain shows phase transition (accuracy jump
>= 0.3 between consecutive scales). PASS if yes.
Derivation: Finding #250 showed 0.60 jump for math. A 0.3 threshold is half that,
allowing for continuous metrics being less sharp. If no domain shows even 0.3 jump,
the phase transition phenomenon is math-specific (discrete metric artifact).

**K2 (#661):** Phase transition location s* differs by >4 scale units across domains
(domain-dependent vs universal). PASS if domain-dependent, FAIL if universal.
Derivation: Math s* is in [4, 6]. If code/medical s* is also in [4, 6], |s*_code -
s*_math| <= 2, which is < 4. If code s* is in [2, 4] or medical s* is in [10, 16],
the difference exceeds 4.

**K3 (#662):** All tested domains reach plateau accuracy >= 0.5 at some scale.
PASS if yes.
Derivation: Math plateaus at 0.70-0.80. If code or medical never exceeds 0.5,
the adapter may not have learned sufficient capability for that domain.

## F. Assumptions and Breaking Conditions

**A1:** The adapters for code and medical were trained with sufficient data and
converged. If adapter quality is poor, no scale will produce good results and K3
will fail for reasons unrelated to scale theory.

**A2:** The evaluation metrics (code: syntax + factual recall; medical: factual recall)
capture meaningful capability, not just format matching. The format confound identified
in Finding #250 may also affect code (adapter imposes a code style the parser can detect)
and medical (adapter imposes a factual recall style the metric rewards).

**A3:** 10 prompts per domain are sufficient to detect a 0.3 jump. With n=10, a jump
from 0.2 to 0.5 (3/10 more correct) has p=0.18 (Fisher exact) -- marginal but
directional. This is a guided exploration, not proof verification.

## G. Worked Example

Consider code prompt "Write a Python function to calculate exponential series."
- At s=1: base model generates prose explanation, maybe partial code. Syntax check: FAIL.
- At s=6 (if H1): adapter activates code generation mode. Model outputs `def exponential_series(x, n): ...` with valid syntax. Syntax check: PASS. Score jumps.
- At s=6 (if H2, s*_code=2): adapter already activated at s=2. Score already high.

## H. Complexity and Architecture Connection

**Compute:** 2 domains x 8 scales x 10 prompts x 128 tokens = 20,480 tokens + base.
At ~100 tok/s, ~2 min per scale x 8 = ~16 min per domain, ~35 min total + overhead.

**Memory:** Same as math experiment. Single model + single adapter + base weight backup.

## Self-Test

1. **ONE property:** Whether the critical LoRA scale s* is universal across domains or
   domain-dependent, determined by per-domain attention margin structure.

2. **Existing theorems:** LoRA perturbation theory (Hu et al., 2106.09685), Finding #250
   (math s*=[4,6]).

3. **Specific numbers:** H1 predicts code/medical jump >= 0.3 at s=[4,6]. H2 predicts
   |s*_code - s*_math| > 4. See prediction table.

4. **Falsification:** K1 FAIL would mean phase transitions are math-specific. K3 FAIL
   would mean adapters lack capability.

5. **Hyperparameters:** 0 added. Sweeping existing scale parameter.

6. **Hack check:** No. This is a measurement experiment extending Finding #250.
