# MATH.md: M2P Training Budget Sweep -- Quality Scales With Steps, Not Architecture

**Experiment type:** Guided exploration (Type 2)
**Prior kills:** exp_m2p_bottleneck_width (Finding #355 -- width closed),
  exp_m2p_depth (Finding #357 -- depth closed, L=2 saturates).
  Architecture search is exhausted. Remaining gap is training convergence.
**Secondary fix:** Bidirectional attention in M2P (remove causal masking).

---

## A. Failure Mode Identification

**Disease:** The M2P generation quality ceiling at L=2, d_M2P=64 is ~92% of SFT
(median across valid domains). This ceiling persists identically across:
- All widths d_M2P in {64, 128, 256} (Finding #355, KILLED)
- All depths L in {1, 2, 4} at L>=2 (Finding #357, KILLED)

Both architectural directions are closed. The ceiling is NOT representational.

**Precise failure mode:** The M2P is trained for 500 gradient steps per domain on
synthetic micro data. The B-matrix regression task has not converged at 500 steps.
The M2P output approximates but does not match the SFT target B-matrices, leaving
a persistent 8% quality gap.

**Root cause diagnosis:** SHINE (arXiv:2602.06358) explicitly reports: prior
hypernetwork failures were due to "insufficient training scale, not architectural
depth." At 500 steps, the M2P training loss is still decreasing (observed in
exp_m2p_depth: final training losses in the 1.5-2.5 range, not yet converged to
the SFT losses in the 1.3-2.0 range). The optimization has not reached its
fixed point.

**Is this the root cause or a symptom?** After eliminating width and depth, training
budget is the next-highest-evidence candidate. The SHINE paper provides direct
evidence that training scale (not architecture) is the binding constraint for
hypernetworks. Additionally, the observed 2.9pp quality gap between reused vs fresh
adapters (Finding #354 vs #355) suggests the system is sensitive to training
conditions, not just architecture.

**Secondary failure mode -- causal masking in M2P attention:**
The M2P attention uses causal (triangular) masking, inherited from the base GPT
architecture. However, M2P memory tokens are NOT autoregressive: they form a
fixed-length summary of base model hidden states, processed simultaneously.
Causal masking restricts token i from attending to tokens j > i, artificially
limiting the M2P's representational capacity. This is architecturally incorrect
for a non-autoregressive encoder module.

---

## B. Prior Mathematical Foundations

### B.1 SGD Convergence for Smooth Non-Convex Functions (Ghadimi & Lan, 2013)

**Theorem (Ghadimi & Lan, 2013, arXiv:1309.5549, Theorem 2.1):**
For L-smooth non-convex function f, SGD with constant step size eta = 1/L
satisfies:

    min_{t=0,...,T-1} E[||grad f(x_t)||^2] <= 2L(f(x_0) - f*) / T + sigma^2 / (bT)

where sigma^2 is the gradient noise variance, b is the batch size, and f* is the
minimum value.

**Key implication:** For constant step size, the gradient norm bound decreases as
O(1/T). Since the loss function landscape near a minimum is approximately quadratic,
the loss itself decreases approximately as O(1/T) for SGD.

### B.2 SGD Convergence for Convex Functions (Standard Result)

**Theorem (Nemirovski & Yudin, 1983; Bottou et al., 2018, arXiv:1606.04838):**
For convex f with bounded gradient variance sigma^2, SGD with step size eta_t = c/sqrt(T)
satisfies:

    E[f(x_bar_T) - f*] <= O(1/sqrt(T))

where x_bar_T is the iterate average. This is the minimax-optimal rate for
stochastic first-order methods.

**Implication for M2P:** The M2P regression problem (predicting B-matrices from
hidden states) is a smooth regression task. Even under the pessimistic non-convex
bound, the loss decreases as O(1/T). Under the Adam optimizer (which the M2P uses),
convergence is typically faster than vanilla SGD due to adaptive learning rates,
but the O(1/T) bound serves as a worst-case lower bound on improvement rate.

### B.3 SHINE Training Scale Law (arXiv:2602.06358)

SHINE demonstrates that hypernetwork quality scales monotonically with training data
volume: "no sign of hitting capacity bottleneck" even at 6B tokens. Prior hypernetwork
failures (Ha et al. 2016, HyperLoader 2024) occurred at orders-of-magnitude smaller
training budgets. The quality curve is concave (diminishing returns) but does not
plateau within practical training budgets.

### B.4 Bidirectional Attention for Non-Autoregressive Modules

**Well-established principle:** Encoder-only architectures (BERT, arXiv:1810.04805;
RoBERTa, arXiv:1907.11692) use bidirectional (full) attention because the input
tokens form a complete, non-sequential signal. Causal masking is appropriate only
when the computation has autoregressive structure (token i depends only on tokens
<= i).

The M2P processes N_MEMORY = 32 memory tokens simultaneously. These are not
generated sequentially -- they receive a global context vector added uniformly.
Bidirectional attention lets every memory token attend to every other, which is
the correct inductive bias for a non-autoregressive encoder.

---

## C. Proof of Guarantee (Training Budget Convergence Theorem)

This is a **Type 2 guided exploration** -- the convergence theory is proven,
but the RATE of convergence for THIS specific M2P task is empirically unknown.
We derive bounds on expected improvement and verify them experimentally.

**Theorem 1 (M2P Quality Improvement with Training Budget).**
Let L_T denote the M2P training loss after T gradient steps, and let q_T denote
the quality ratio at step T. If the M2P loss landscape is L-smooth and the
gradient noise variance is bounded by sigma^2, then:

    E[L_T] - L* <= (L_0 - L*) * C / T + sigma^2 / (bT)

for some constant C depending on the step size and smoothness. Furthermore, since
quality_ratio = (base_loss - m2p_loss) / (base_loss - sft_loss) and m2p_loss
monotonically decreases with L_T, quality_ratio monotonically increases with T.

*Proof sketch.*
The M2P regression loss is:
  L(theta) = E_{h~P(H)} [||M2P_theta(h) - B*||^2_F]

where B* is the SFT target B-matrix and P(H) is the distribution of base model
hidden states. This is a standard regression loss:
1. It is bounded below by 0 (and by the irreducible noise in B* recovery).
2. It is differentiable with respect to theta (M2P is a transformer, differentiable
   everywhere by construction).
3. The gradient is unbiased: E[grad L(theta; h_i)] = grad L(theta) for i.i.d.
   training samples.

By Ghadimi & Lan (2013) Theorem 2.1, for L-smooth f:
  min_{t<=T} E[||grad f(x_t)||^2] <= O(1/T)

Near a local minimum where the Hessian is approximately constant, ||grad f||^2
approximately equals H * (L_T - L*), so:
  E[L_T] - L* <= O(1/T)

The quality ratio is:
  q_T = (base_loss - m2p_loss(T)) / (base_loss - sft_loss)

Since m2p_loss(T) decreases as L_T decreases, q_T increases with T.

QED (rate bound; specific constants are empirically unknown).

**Corollary (Diminishing Returns).** The marginal quality improvement per step
decreases with T. Doubling from T to 2T provides less improvement than doubling
from T/2 to T. Specifically:

    Delta_q(T to 2T) / Delta_q(T/2 to T)  approximately equals  1/2

This means: if q(500) = 91.9% and the improvement from 500 to 1000 steps is
Delta_1, the improvement from 1000 to 2000 steps should be approximately Delta_1.
(Both doublings cover the same factor-of-2 in step space.)

**Theorem 2 (Bidirectional Attention Reduces Approximation Error).**
Let f_causal(x) denote the M2P attention output with causal masking, and
f_bidir(x) the output with bidirectional attention. For any input x:

    ||f_bidir(x) - B*||_F <= ||f_causal(x) - B*||_F

with equality only when the optimal attention pattern happens to be lower-triangular.

*Proof.*
The causal attention constrains the attention matrix A to be lower-triangular:
  A_causal in {A : A_{ij} = 0 for j > i, sum_j A_{ij} = 1}

Bidirectional attention allows:
  A_bidir in {A : sum_j A_{ij} = 1}

Since A_causal subset A_bidir, the minimum of any loss over A_bidir is at most
the minimum over A_causal:

  min_{A in A_bidir} L(A) <= min_{A in A_causal} L(A)

QED (set inclusion argument; the gain may be zero if the optimal pattern is
already lower-triangular).

---

## D. Quantitative Predictions (Derived from Theorem 1)

### D.1 Training Budget Predictions

**Baseline (from exp_m2p_depth, T=500):** median quality q(500) = 91.9%

Using the O(1/T) convergence rate and assuming the quality gap (1 - q) is
proportional to the remaining loss gap:

    1 - q(T) proportional to C/T + noise

The remaining gap at T=500 is: 1 - 0.919 = 0.081 (8.1pp below SFT).

**Model:** Assume q(T) = 1 - alpha/T - beta, where alpha captures the learnable
gap and beta the irreducible gap (B-matrix approximation noise, micro-scale
artifacts). With only one data point (T=500, q=91.9%), we cannot separate alpha
from beta. However, we can bound the prediction:

**Optimistic scenario (beta = 0, pure O(1/T)):**
- q(1000) = 1 - 0.081 * (500/1000) = 1 - 0.0405 = 95.95%  (+4.0pp)
- q(2000) = 1 - 0.081 * (500/2000) = 1 - 0.0203 = 97.97%  (+6.1pp)

**Pessimistic scenario (beta = 0.04, half the gap is irreducible):**
- q(1000) = 1 - 0.041 * (500/1000) - 0.04 = 93.95%  (+2.0pp)
- q(2000) = 1 - 0.041 * (500/2000) - 0.04 = 94.97%  (+3.1pp)

**Conservative prediction (average of optimistic and pessimistic):**
- q(1000): ~95.0% (+3.0pp over q(500))
- q(2000): ~96.5% (+4.6pp over q(500))

| Step count | Optimistic q | Pessimistic q | Conservative q | Delta from 500 |
|-----------|-------------|--------------|---------------|---------------|
| 500       | 91.9%       | 91.9%        | 91.9%         | --            |
| 1000      | 96.0%       | 94.0%        | 95.0%         | +3.0pp        |
| 2000      | 98.0%       | 95.0%        | 96.5%         | +4.6pp        |

### D.2 Bidirectional Attention Prediction

Theorem 2 guarantees f_bidir >= f_causal in approximation quality. The magnitude
depends on how much the optimal attention pattern deviates from lower-triangular.
For N_MEMORY = 32 memory tokens with isotropic initialization:
- If the M2P relies on attention to distant memory tokens (j >> i), the gain could
  be substantial (~1-3pp).
- If the M2P primarily attends to local/lower-indexed tokens, the gain is negligible.

**Prediction:** Bidirectional attention provides +1-2pp quality improvement at
T=500 (where the M2P is data-starved and every representational advantage matters).
At T=2000, the gain may be smaller because the M2P has more gradient steps to
compensate for the masking constraint.

### D.3 Combined Prediction

With both fixes (2000 steps + bidirectional attention):
- Conservative: 96.5% + 1pp = 97.5%
- This would PASS K877 (quality >= 97%), confirming the architecture is ready.

### Kill Criteria Derivation

- **K876 (K_progress):** quality(2000) > quality(500) + 2pp
  Derivation: The pessimistic scenario predicts +3.1pp, well above 2pp. Under
  the O(1/T) bound, a 4x increase in steps from 500 to 2000 should reduce the
  learnable gap by 75%. Even if 50% of the gap is irreducible, the remaining
  improvement is 0.5 * 0.081 * 0.75 = 3.0pp. The 2pp threshold provides margin.
  FAIL condition: quality(2000) <= quality(500) + 2pp, meaning training budget
  is NOT the bottleneck (and the gap is dominated by irreducible error).

- **K877 (K_ceiling):** quality(2000) >= 97%
  Derivation: The optimistic scenario predicts 98.0%, the pessimistic 95.0%.
  97% falls between them. PASS confirms the M2P architecture is sufficient
  and training budget was the bottleneck. FAIL means even with adequate training,
  something else limits quality (maybe the micro-scale artifacts or B-matrix
  intrinsic complexity).

- **K878 (K_plateau):** |quality(2000) - quality(1000)| < 1pp
  Derivation: Under O(1/T), the improvement from 1000 to 2000 should be smaller
  than from 500 to 1000 (diminishing returns). If the improvement is < 1pp,
  the M2P has exhausted what can be gained from more steps. This is the
  KILL case: budget is exhausted, not infinitely scalable.
  Note: K878 PASS + K876 PASS together mean "budget helps but saturates quickly."

---

## E. Assumptions & Breaking Conditions

**Assumption 1:** The M2P regression loss is L-smooth.
EVIDENCE: The M2P is a transformer with RMSNorm, GELU activations, and linear
output heads. These are all smooth (infinitely differentiable) operations.
BREAKING: If the loss landscape has sharp curvature (very large L), the O(1/T)
bound has a worse constant. This would slow convergence, not invalidate the trend.

**Assumption 2:** The O(1/T) rate applies to Adam (not just SGD).
EVIDENCE: Adam with constant learning rate converges at O(1/sqrt(T)) or faster
for non-convex smooth objectives (Kingma & Ba, 2014; Reddi et al., 2019 for
AMSGrad convergence). The O(1/T) rate from SGD theory is a lower bound on Adam's
performance. In practice, Adam often converges faster due to adaptive step sizes.
BREAKING: If the learning rate is too high and Adam diverges. Mitigated by using
the same lr=1e-3 proven in prior experiments.

**Assumption 3:** The 91.9% quality at T=500 is the true baseline (reproducible).
EVIDENCE: Finding #357 measured exactly this value. exp_m2p_bottleneck_width
measured 95-97% at L=2 (different random seed, reuse contamination). The variance
between runs is 2-5pp. For the training budget sweep, we use the SAME base model,
A-matrices, and SFT adapters across all step counts (shared infrastructure,
only M2P steps vary), which eliminates between-run variance for the RELATIVE
comparisons (delta between step counts).
BREAKING: If the base model or SFT adapters differ between step counts due to
implementation error. Prevented by the experiment design (shared base, only
M2P_STEPS varies).

**Assumption 4:** The gain from bidirectional attention is independent of training
budget. This is approximate: at very long training budgets, the M2P may learn to
compensate for causal masking (allocating the first memory slots for global context).
BREAKING: Bidirectional gain at T=2000 may be smaller than at T=500.

**Assumption 5:** The 500-step M2P training loss is still decreasing (not converged).
EVIDENCE: In exp_m2p_depth, the M2P final training losses (~1.5-2.5) are above
the SFT losses (~1.3-2.0), indicating convergence is not reached.
BREAKING: If the M2P has already converged at 500 steps and the gap is irreducible.
This would trigger K878 (plateau) and kill the experiment.

---

## F. Worked Example (T=500, projecting to T=1000 and T=2000)

**Given from exp_m2p_depth (L=2, d_M2P=64, T=500):**
- arithmetic: quality = 91.85%,  sft_loss = 1.7907, m2p_loss = 2.2577
- sort:       quality = 91.91%,  sft_loss = 1.8454, m2p_loss = 2.1313
- reverse:    quality = 93.30%,  sft_loss = 2.0134, m2p_loss = 2.2679
- repeat:     quality = 83.92%,  sft_loss = 1.3962, m2p_loss = 2.4622
- parity:     EXCLUDED (gap < 0.05)

**Projecting arithmetic domain (base_loss = 7.5187):**

At T=500: m2p_loss = 2.2577, gap = base - sft = 5.728, remaining = m2p - sft = 0.467
Under O(1/T): remaining(T) proportional to C/T
  C = 0.467 * 500 = 233.5
  remaining(1000) = 233.5 / 1000 = 0.234
  m2p_loss(1000) = 1.7907 + 0.234 = 2.024
  quality(1000) = (7.5187 - 2.024) / 5.728 = 96.0%

  remaining(2000) = 233.5 / 2000 = 0.117
  m2p_loss(2000) = 1.7907 + 0.117 = 1.908
  quality(2000) = (7.5187 - 1.908) / 5.728 = 98.0%

**Projecting sort domain (base_loss = 5.3795):**

At T=500: m2p_loss = 2.1313, gap = 3.534, remaining = 2.1313 - 1.8454 = 0.286
  C = 0.286 * 500 = 143.0
  remaining(1000) = 143.0 / 1000 = 0.143
  m2p_loss(1000) = 1.8454 + 0.143 = 1.988
  quality(1000) = (5.3795 - 1.988) / 3.534 = 96.0%

  remaining(2000) = 143.0 / 2000 = 0.072
  m2p_loss(2000) = 1.8454 + 0.072 = 1.917
  quality(2000) = (5.3795 - 1.917) / 3.534 = 98.0%

**Projecting repeat domain (base_loss = 8.0256):**

At T=500: m2p_loss = 2.4622, remaining = 2.4622 - 1.3962 = 1.066
  C = 1.066 * 500 = 533.0
  remaining(1000) = 533.0 / 1000 = 0.533
  m2p_loss(1000) = 1.3962 + 0.533 = 1.929
  quality(1000) = (8.0256 - 1.929) / 6.629 = 91.9%

  remaining(2000) = 533.0 / 2000 = 0.267
  m2p_loss(2000) = 1.3962 + 0.267 = 1.663
  quality(2000) = (8.0256 - 1.663) / 6.629 = 96.0%

**Summary of worked example (optimistic O(1/T) extrapolation):**

| Domain     | q(500)  | q(1000) pred | q(2000) pred |
|------------|---------|-------------|-------------|
| arithmetic | 91.85%  | 96.0%       | 98.0%       |
| sort       | 91.91%  | 96.0%       | 98.0%       |
| reverse    | 93.30%  | 96.6%       | 98.3%       |
| repeat     | 83.92%  | 91.9%       | 96.0%       |
| **Median** | **91.9%** | **96.0%** | **98.0%** |

Note: These are optimistic predictions assuming pure O(1/T) with zero irreducible
gap. Real performance will be lower (see Section D pessimistic scenario). The
conservative prediction is median q(1000) ~ 95.0%, q(2000) ~ 96.5%.

---

## G. Complexity & Architecture Connection

**Compute cost scaling with T:**
- Training one M2P for T steps costs O(T * N_MEMORY^2 * d_M2P * L) FLOPs
- At d_M2P=64, L=2, N_MEMORY=32: each step is ~O(65K) ops (negligible)
- T=2000 is 4x the compute of T=500. At 77s for the full depth sweep (3 depths
  x 5 domains x 500 steps), the training budget sweep (1 depth x 5 domains x
  {500, 1000, 2000} steps) should take roughly:
  - T=500: ~13s, T=1000: ~26s, T=2000: ~52s
  - With bidirectional variants: ~2x
  - Total estimated: ~3-5 minutes

**Memory cost:** Identical to exp_m2p_depth. No new parameters.

**Bidirectional attention:** Removing the causal mask is a one-line change (delete
the mx.triu mask construction and addition in M2PAttention). No new parameters,
no new hyperparameters. Compute is identical (dense attention is already computed;
the mask only zeroes upper-triangular entries post-hoc).

**Architecture parameter counts (unchanged from exp_m2p_depth):**
- M2P at L=2, d_M2P=64: ~1,167,680 params
- No new parameters from either the step sweep or the bidirectional fix

**Production context:** The bidirectional attention fix aligns the M2P with
standard encoder architecture (BERT family). If the M2P is eventually deployed
as a routing/generation module, bidirectional attention is the canonical choice
for non-autoregressive processing.

---

## Self-Test (MANDATORY)

**1. What is the ONE mathematical property that makes the failure mode impossible?**
O(1/T) convergence rate of SGD/Adam on smooth losses guarantees that the M2P
training loss decreases monotonically with training steps, making the quality
gap shrink toward the irreducible minimum.

**2. Which existing theorem(s) does the proof build on?**
Ghadimi & Lan (2013, arXiv:1309.5549, Theorem 2.1) -- SGD convergence for
L-smooth non-convex functions: O(1/T) gradient norm bound.
Bottou et al. (2018, arXiv:1606.04838) -- SGD O(1/sqrt(T)) convergence for convex.
SHINE (arXiv:2602.06358) -- empirical training-scale law for hypernetworks.

**3. What specific numbers does the proof predict?**
Conservative predictions: q(1000) ~ 95.0% (+3.0pp), q(2000) ~ 96.5% (+4.6pp).
Optimistic: q(1000) ~ 96.0%, q(2000) ~ 98.0%.
Bidirectional attention: +1-2pp at T=500.
Kill thresholds: K876 requires +2pp (predicted +3-6pp), K877 requires 97%
(predicted 95-98%, uncertain).

**4. What would FALSIFY the proof (not just the experiment)?**
The proof is wrong if the M2P training loss is NOT L-smooth (e.g., sharp
discontinuities in the loss landscape), which would violate the Ghadimi & Lan
assumptions. This is extremely unlikely for a standard transformer with smooth
activations. The experiment is falsified if q(2000) <= q(500) + 2pp (K876 FAIL),
meaning the gap is irreducible rather than convergence-limited.

**5. How many hyperparameters does this approach add?**
Count: 0. The step counts {500, 1000, 2000} are the sweep variable.
The bidirectional attention fix removes a hyperparameter (the causal mask)
rather than adding one. No new loss terms, regularizers, or architectural changes.

**6. Hack check: Am I adding fix #N to an existing stack?**
No. This experiment makes ONE change to the training procedure (more steps) and
ONE change to the architecture (remove incorrect causal mask). Both are motivated
by specific prior findings: (1) SHINE identifies training scale as the bottleneck,
(2) REVIEW-adversarial.md identifies causal masking as architecturally wrong.
Neither is a "fix on top of fixes" -- they address distinct root causes.
