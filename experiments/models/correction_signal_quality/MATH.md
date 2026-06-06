# Correction Signal Quality: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Value/Range |
|--------|-----------|-------------|
| D | Number of domain types | 6 |
| N_corr | Number of corrections per domain per source | 200 |
| S | Number of correction sources | 3 (human, teacher, execution) |
| q_s | True correction accuracy of source s | [0, 1] |
| p_degen(s) | Degeneracy rate for source s | [0, 1] |
| c_s | Cost per correction from source s (USD) | varies |
| delta_q | Expert quality improvement per correct correction | varies by domain |
| delta_q_degen | Quality change from degenerate correction | negative |
| K | Number of self-learning cycles | 20 |
| alpha | Diminishing returns exponent | 0.7 |

## 2. Correction Source Models

### 2.1 Human Corrections (Gold Standard)

Human corrections are modeled as near-perfect with known error rates:

    q_human = 0.95  (5% human error, from inter-annotator agreement literature)
    p_degen_human = 0.02  (rare degenerate corrections)
    c_human = $2.00/correction  (professional developer time)

Error model: uniform random errors. When humans err, they produce a
plausible-but-wrong correction with equal probability.

### 2.2 Teacher (70B) Corrections

Teacher model corrections have systematic biases from the literature:

    q_teacher(difficulty) = sigmoid(beta_0 + beta_1 * difficulty)

where difficulty in [0, 1] represents problem complexity. Parameters calibrated from:
- RLAIF paper: ~88% agreement with human labels (Lee et al., 2023)
- Self-Refine: iterative improvement saturates at ~85% (Madaan et al., 2023)
- Positional bias: 10-15% error from ordering effects (Wang et al., 2023)

For our model:
    q_teacher_easy = 0.92  (simple problems)
    q_teacher_hard = 0.70  (complex problems)
    beta_0 = 2.44, beta_1 = -2.88  (fit to above endpoints)
    p_degen_teacher = 0.08  (plausible but subtly wrong)
    c_teacher = $0.001/correction  (API cost for 70B model)

Error model: systematic bias. Teacher errors cluster on specific problem types
(edge cases, novel patterns, complex dependencies). Modeled as increased error
rate for problems with difficulty > 0.7.

### 2.3 Execution Feedback

Execution feedback is binary (pass/fail) with known failure modes:

    q_exec(domain) = { 0.99 if domain in {code_simple, code_algorithmic}
                     { 0.85 if domain in {code_systems}
                     { 0.00 if domain in {writing, reasoning, medical}
                     }

Execution feedback is ONLY applicable to code domains. For non-code domains,
it provides zero signal (q_exec = 0).

Degeneracy model for execution feedback:
    p_degen_exec(test_coverage) = (1 - test_coverage) * gamma

where gamma = 0.3 is the probability that a passing-but-degenerate solution
exists given imperfect test coverage. Test coverage varies:
    - Simple functions: coverage = 0.95 (well-tested)
    - Algorithmic: coverage = 0.80 (edge cases hard)
    - Systems: coverage = 0.60 (integration gaps)

Cost:
    c_exec = $0.0001/correction  (compute cost of running tests)

## 3. Expert Improvement Model

### 3.1 Per-Correction Quality Change

Each correction produces a quality delta depending on its type:

    Delta_q(correction_type) = {
        +delta_base * (1 - q_current)^alpha    if correct & non-degenerate
        -delta_degen * penalty                  if degenerate
        -delta_wrong * penalty                  if incorrect
        0                                       if no signal (e.g., exec on writing)
    }

where:
    delta_base = 0.02  (2% improvement per correct correction at q=0)
    delta_degen = 0.01  (1% degradation per degenerate correction)
    delta_wrong = 0.015  (1.5% degradation per wrong correction)
    alpha = 0.7  (diminishing returns: harder to improve good experts)
    penalty = 1.0  (unit penalty for simplicity)

### 3.2 Cumulative Expert Quality After K Corrections

After K corrections from source s on domain d:

    q_expert(K) = q_0 + sum_{k=1}^{K} Delta_q_k

where q_0 is the initial expert quality (from distillation), and Delta_q_k
is the quality change from the k-th correction.

In expectation:

    E[q_expert(K)] = q_0 + sum_{k=1}^{K} [
        q_s * (1 - p_degen_s) * delta_base * (1 - q_current)^alpha
        - q_s * p_degen_s * delta_degen
        - (1 - q_s) * delta_wrong
    ]

### 3.3 Effective Improvement Rate

The effective improvement rate (improvement per dollar) for source s on domain d:

    EIR(s, d) = E[Delta_q per correction] / c_s

    E[Delta_q] = q_s(d) * (1-p_degen_s(d)) * delta_base * (1-q_current)^alpha
                 - q_s(d) * p_degen_s(d) * delta_degen
                 - (1-q_s(d)) * delta_wrong

## 4. Decision Tree

The optimal correction source for domain d is:

    s*(d) = argmax_s EIR(s, d)

Subject to:
    - Execution feedback only available for code domains
    - Human corrections budget-limited (max B_human corrections total)
    - Teacher corrections unlimited within API budget

### 4.1 Expected Optimal Routing

For code domains with good test coverage (coverage > 0.8):
    s* = execution  (highest EIR due to near-zero cost)

For code domains with poor test coverage (coverage < 0.6):
    s* = teacher  (execution degeneracy exceeds teacher error)

For non-code domains:
    s* = teacher  (only automated option)

For critical domains requiring > 95% accuracy:
    s* = human  (only source meeting accuracy threshold)

### 4.2 Hybrid Strategy

In practice, use a cascade:
    1. If code domain: try execution feedback first
    2. If execution shows regression or no signal: escalate to teacher
    3. If teacher confidence < threshold: escalate to human

## 5. Kill Criteria Formalization

### K1: Teacher correction error rate > 20%

    KILL if: (1 - q_teacher_avg) > 0.20

where q_teacher_avg = mean over difficulties of q_teacher(difficulty).

From our model: q_teacher_avg = integral_0^1 sigmoid(2.44 - 2.88*d) dd
                               = ~0.83

So expected teacher error = ~17%. This is BELOW the 20% threshold but close.
The kill criterion is at risk for hard problems (difficulty > 0.8: error ~25%).

### K2: Execution degeneracy

    KILL if: p_degen_exec * q_exec > 0.10  (>10% of accepted solutions are degenerate)

For code_simple: 0.05 * 0.015 * 0.99 = 0.7% -- far below threshold
For code_systems: 0.40 * 0.12 * 0.85 = 4.1% -- below threshold
SURVIVES unless test_coverage < 0.50.

## 6. Worked Numerical Example

Domain: code_algorithmic (e.g., sorting, graph algorithms)
Initial expert quality: q_0 = 0.60 (from distillation)
Difficulty distribution: uniform [0.3, 0.9] (moderately hard)

After 100 corrections:

**Human (q=0.95, p_degen=0.02, cost=$2.00):**
    Correct non-degen: 93.1 corrections * 0.02 * (1-0.6)^0.7 / step ~ +0.94
    Degen: 1.9 corrections * -0.01 = -0.019
    Wrong: 5 corrections * -0.015 = -0.075
    Net quality delta: ~+0.846 (diminishing returns from sum)
    Total cost: $200

**Teacher (q=0.82, p_degen=0.08, cost=$0.001):**
    Correct non-degen: 75.4 * delta ~ +0.76
    Degen: 6.6 * -0.01 = -0.066
    Wrong: 18 * -0.015 = -0.270
    Net quality delta: ~+0.424
    Total cost: $0.10

**Execution (q=0.90, p_degen=0.06, cost=$0.0001):**
    Correct non-degen: 84.6 * delta ~ +0.85
    Degen: 5.4 * -0.01 = -0.054
    Wrong: 10 * -0.015 = -0.150
    Net quality delta: ~+0.646
    Total cost: $0.01

EIR: execution ($64.6/dollar) >> teacher ($4,240/dollar) >> human ($0.0042/dollar)

Wait -- teacher EIR is actually higher because absolute improvement is reasonable
at trivial cost. Let me recalculate:

EIR_human = 0.846 / 200 = 0.00423 quality-per-dollar
EIR_teacher = 0.424 / 0.10 = 4.24 quality-per-dollar
EIR_exec = 0.646 / 0.01 = 64.6 quality-per-dollar

Conclusion: Execution feedback is 15x more cost-effective than teacher, which
is 1000x more cost-effective than human. Human corrections are only justified
when accuracy requirements exceed teacher capability.
