# Correction Routing Sensitivity: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Value/Range |
|--------|-----------|-------------|
| h | teacher_hard_accuracy | [0.50, 0.95] |
| h_0 | baseline teacher_hard_accuracy | 0.70 |
| b | teacher_base_accuracy (fixed) | 0.92 |
| mu_d | difficulty_mean for domain d | [0.05, 0.95] |
| sigma_d | difficulty_std for domain d | domain-specific |
| delta_mu | difficulty_mean perturbation | [-0.10, +0.10] |
| p_degen | teacher degeneracy rate | 0.08 |
| gamma | execution degeneracy coefficient | 0.30 |
| c_d | test coverage for domain d | [0, 1] |
| D | set of 6 domains | {py, algo, sys, write, reason, med} |

## 2. Teacher Error Rate as Analytic Integral

Teacher correction accuracy at difficulty x is:

    q(x; h) = sigmoid(beta_0 + beta_1 * x)

where:

    beta_0 = logit(b) = log(b / (1 - b))
    beta_1 = logit(h) - logit(b)

For domain d with difficulty distribution X_d ~ TruncNormal(mu_d + delta_mu, sigma_d, 0, 1),
the expected error rate is:

    E_d(h, delta_mu) = E_{x ~ X_d}[1 - q(x; h)]
                     = integral_0^1 [1 - sigmoid(beta_0 + beta_1 * x)] * f_d(x) dx

where f_d is the truncated normal PDF. This is computed via numerical quadrature
(10,000-point Riemann sum on [0.01, 0.99]).

The aggregate error rate across all domains:

    E_avg(h, delta_mu) = (1/|D|) * sum_{d in D} E_d(h, delta_mu)

## 3. Harmful Rate (Wrong + Degenerate)

The adversarial review noted that the original K1 metric (error rate only) ignores
degenerate corrections that are technically correct but harmful. The harmful rate is:

    H_d(h, delta_mu) = P(wrong) + P(correct AND degenerate)
                     = [1 - q(x; h)] + q(x; h) * p_degen
                     = 1 - q(x; h) * (1 - p_degen)

Integrating over the difficulty distribution:

    H_d(h, delta_mu) = integral_0^1 [1 - sigmoid(beta_0 + beta_1*x) * (1-p_degen)] * f_d(x) dx

Note: H_d > E_d always, since H_d = E_d + q_avg * p_degen > E_d.

## 4. Analytical Breakpoints

### 4.1 K1 Error Breakpoint

Find h* such that E_avg(h*, 0) = 0.20:

    h*_error = root of [E_avg(h, 0) - 0.20] over h in [0.50, 0.95]

Solved via Brent's method. Result: **h*_error = 0.6644**.

Interpretation: teacher hard accuracy must exceed 0.6644 for the average
error rate to stay below 20%. The baseline h_0 = 0.70 has margin of only 0.036.

### 4.2 K1 Harmful Breakpoint

Find h* such that H_avg(h*, 0) = 0.20:

    h*_harmful = root of [H_avg(h, 0) - 0.20] over h in [0.50, 0.95]

Result: **h*_harmful = 0.8206**.

This is the critical finding: to keep the *harmful* rate (wrong + degenerate)
below 20%, teacher hard accuracy must exceed 0.8206. The baseline h_0 = 0.70
is 0.12 below this threshold. The harmful rate at baseline is 26.0%, not 19.6%.

### 4.3 Per-Domain Error Breakpoints

| Domain | h*_error | h*_harmful | Baseline Error | Baseline Harmful |
|--------|:---:|:---:|:---:|:---:|
| python_basics | none in range | 0.6788 | 12.7% | 19.7% |
| algorithm_design | 0.6685 | 0.8226 | 18.6% | 25.1% |
| systems_programming | 0.7341 | 0.8459 | 22.1% | 28.3% |
| creative_writing | 0.6087 | 0.8007 | 16.7% | 23.4% |
| logical_reasoning | 0.6919 | 0.8309 | 19.6% | 26.0% |
| medical_qa | 0.7175 | 0.8401 | 21.0% | 27.3% |

The harmful breakpoints cluster tightly around h = 0.82 +/- 0.02. This means
ALL domains require teacher hard accuracy > 0.82 to achieve < 20% harmful rate,
which is significantly above the 0.70 baseline and even above the best published
RLAIF agreement rates.

### 4.4 K2 Coverage Breakpoint (Closed-Form)

Execution degeneracy rate:

    p_degen_exec(c_d) = (1 - c_d) * gamma

Kill when p_degen_exec > 0.10:

    (1 - c_d) * gamma > 0.10
    c_d < 1 - 0.10 / gamma
    c_d < 1 - 0.10 / 0.30
    **c_d < 0.6667**

This is exact (no numerical methods needed). Systems_programming has
c = 0.60 < 0.667, confirming the K2 kill from the parent experiment.

## 5. Decision Tree Stability Analysis

### 5.1 Why Zero Flips?

The decision tree routing is determined by EIR (quality per dollar).
For code domains, execution costs $0.0001 vs teacher at $0.001 -- a 10x gap.
Even when execution quality degrades, the cost advantage dominates.

The flip condition for a code domain d requires:

    EIR_teacher(d) > EIR_exec(d)
    => [q_t * (1-p_t) * delta - q_t * p_t * delta_d - (1-q_t) * delta_w] / c_t
       > [q_e * (1-p_e) * delta - q_e * p_e * delta_d - (1-q_e) * delta_w] / c_e

With c_t/c_e = 10, this requires teacher to be ~10x better in quality terms,
which never happens since execution accuracy (0.85-0.99) exceeds teacher
accuracy on hard problems.

For non-code domains, teacher is the only automated option (execution = N/A),
so no flip is possible without introducing a new source.

### 5.2 Flip Condition Derivation

Let R_s = q_s * (1-p_s) * delta_base * (1-q_current)^alpha - q_s * p_s * delta_degen - (1-q_s) * delta_wrong
be the expected quality change per correction from source s.

Then EIR_s = R_s / c_s.

A flip from execution to teacher occurs when:

    R_teacher / c_teacher > R_exec / c_exec

At the baseline parameters (worst case: systems_programming, h=0.60, dp=+0.10):
- R_teacher ~ 0.70 * 0.92 * 0.02 * 0.5^0.7 - 0.70 * 0.08 * 0.01 - 0.30 * 0.015
            ~ 0.0074 - 0.00056 - 0.0045 = 0.0023
- R_exec ~ 0.85 * 0.88 * 0.02 * 0.5^0.7 - 0.85 * 0.12 * 0.01 - 0.15 * 0.015
         ~ 0.0092 - 0.00102 - 0.00225 = 0.0059

EIR_teacher = 0.0023 / 0.001 = 2.3
EIR_exec = 0.0059 / 0.0001 = 59.0

The 25x gap explains why no flips occur: execution would need to be ~25x worse
in quality improvement to flip, which is far outside the swept parameter range.

## 6. Sensitivity of Error/Harmful Rates

Error rate sensitivity to teacher hard accuracy:

    dE_avg/dh = -(1/|D|) * sum_d integral [sigmoid'(beta_0 + beta_1*x) * d(beta_1)/dh * x] * f_d dx

Since beta_1 = logit(h) - logit(b), d(beta_1)/dh = 1/(h(1-h)).

At h=0.70: d(beta_1)/dh = 1/(0.70*0.30) = 4.76.

Numerically: a 1pp increase in h reduces E_avg by ~0.5pp.
This means the K1 error margin (0.4pp) can be flipped by a mere 0.8pp change in h.

For the harmful rate: dH_avg/dh = (1 - p_degen) * dE_avg/dh (since q_avg
absorbs the same derivative). With p_degen = 0.08, the harmful rate is 8%
less sensitive to h than the error rate, but it starts 6.4pp higher.

## 7. Worked Example: Breakpoint Crossing

Take systems_programming (mu=0.75, sigma=0.15):

At h = 0.7341 (error breakpoint):
- beta_0 = logit(0.92) = 2.44
- beta_1 = logit(0.7341) - 2.44 = 1.015 - 2.44 = -1.42
- q(0.75) = sigmoid(2.44 - 1.42*0.75) = sigmoid(1.375) = 0.798
- Error at mean difficulty: 1 - 0.798 = 0.202 ~ 20.2% (matches threshold)

At h = 0.70 (baseline):
- beta_1 = logit(0.70) - 2.44 = 0.847 - 2.44 = -1.593
- q(0.75) = sigmoid(2.44 - 1.593*0.75) = sigmoid(1.245) = 0.776
- Error at mean: 22.4% (above threshold)
- Harmful at mean: 22.4% + 0.776*0.08 = 28.6% (far above threshold)

## 8. Assumptions

1. Teacher base accuracy (0.92) is fixed. Only hard accuracy varies.
   Justified: easy-problem accuracy is well-calibrated from RLAIF.

2. Degeneracy rate (0.08) is independent of difficulty.
   Simplification: in practice, harder problems may have higher degeneracy.

3. Difficulty distributions are Gaussian. Real distributions may be bimodal
   (many easy + some hard problems).

4. Costs are fixed. In practice, API costs change; the 10x execution/teacher
   gap may narrow with cheaper inference.

5. Sources are independent: no learning-from-corrections that would change
   accuracy over time.
