# MATH.md: LoRA A-B Pairing Correctness for Grassmannian Adapters

## Type: Verification (Type 1)

This is a bug-fix verification, not a new mechanism. The mathematical content is
the proof that LoRA output with mismatched A matrices produces uncorrelated noise
rather than learned perturbations.

---

## A. Failure Mode Identification

**The failure:** All 7 routing experiments at N=24 loaded adapter B weights into
`LoRALinear` modules which initialize `lora_a` as random Kaiming-uniform matrices.
The adapters were trained with `TernaryLoRALinear` using per-domain Grassmannian A
matrices from a pre-computed skeleton. Since adapter.npz files contain only `lora_b`
weights, the A matrices were silently replaced with random matrices at evaluation time.

**Why this is catastrophic:** LoRA computes:

$$\Delta W = \alpha \cdot A \cdot B$$

where A is (d_in, r) and B is (r, d_out). Training optimizes B given a FIXED A.
The learned B encodes the domain-specific perturbation in the column space of A.
Replacing A with A' (random) means:

$$\Delta W' = \alpha \cdot A' \cdot B \neq \alpha \cdot A \cdot B = \Delta W$$

The perturbation applied is completely different from the one trained.

---

## B. The Right Question

Not "how do we prevent adapter loading bugs?" but rather:
**What is the expected output quality when LoRA B weights are applied with an
incorrect A matrix?**

If we can bound the error, we can predict exactly what the buggy experiments should
have produced -- and verify that prediction matches the observed 0.04% improvement.

---

## C. Prior Mathematical Foundations

**LoRA (Hu et al., 2021, arXiv:2106.09685):**
Low-rank adaptation decomposes weight updates as Delta_W = alpha * A * B where
A in R^{d_in x r}, B in R^{r x d_out}, r << min(d_in, d_out).

**Johnson-Lindenstrauss Lemma (Johnson & Lindenstrauss, 1984):**
Random projections approximately preserve distances. A random matrix A' in R^{d x r}
with i.i.d. entries from N(0, 1/d) satisfies:
(1-eps)|u-v|^2 <= |A'(u-v)|^2 <= (1+eps)|u-v|^2
with probability >= 1 - 2*exp(-eps^2*r/4).

This is relevant because: when A' is random but B was trained for A, the product
A'B is a random projection of B's rows, unrelated to the intended perturbation AB.

---

## D. Proof of Guarantee

**Theorem 1 (A-B Mismatch Produces Near-Zero Expected Perturbation).**

Let A in R^{d x r} be a fixed matrix (Grassmannian-initialized, frozen during training).
Let B* in R^{r x d_out} be the trained B matrix optimized for the pair (A, B).
Let A' in R^{d x r} be an independent random matrix with entries ~ Uniform(-s, s)
where s = 1/sqrt(d).

Then for any input x in R^d:
1. E[x @ A' @ B*] = 0 (the expected perturbation is zero)
2. Var[||x @ A' @ B*||] = (s^2/3) * ||x||^2 * ||B*||_F^2 / d_out

*Proof.*

(1) Since A' has zero-mean i.i.d. entries and is independent of x and B*:
E[(x @ A') @ B*] = E[x @ A'] @ B* = x @ E[A'] @ B* = x @ 0 @ B* = 0.

(2) Let z = x @ A' in R^r. Each z_j = sum_i x_i * A'_{ij}. Since A'_{ij} are
i.i.d. with mean 0 and variance s^2/3 (uniform on [-s,s]):
E[z_j^2] = sum_i x_i^2 * s^2/3 = (s^2/3) * ||x||^2.

The perturbation magnitude ||z @ B*||^2 = sum_k (sum_j z_j B*_{jk})^2.
Taking expectation and using independence of z_j from B*:
E[||z @ B*||^2] = (s^2/3) * ||x||^2 * sum_j ||B*_j||^2 = (s^2/3) * ||x||^2 * ||B*||_F^2.

With s = 1/sqrt(d) = 1/sqrt(2560) and the perturbation scaled by alpha:
The RMS perturbation per output dimension is:
alpha * sqrt(s^2/3 * ||x||^2 * ||B*||_F^2 / d_out)
= alpha / sqrt(3 * d) * ||x|| * ||B*||_F / sqrt(d_out).

For d=2560, r=16, alpha=20, this is a small random perturbation that averages to zero,
explaining why buggy adapters showed ~0% improvement: the perturbation is random noise
centered at zero. QED.

**Corollary 1.** With correct A-B pairing, the perturbation x @ A @ B* is a deterministic
learned function. With incorrect A' (random), it becomes zero-mean noise. The adapter
does nothing on average. Oracle PPL with wrong A should approximately equal base PPL.

**Prediction 1:** Oracle PPL (wrong A) approximately equals base PPL.
  Measured in prior experiment: avg_individual = 10.12, avg_base = 10.06. Delta = -0.6%.
  This matches the prediction (zero-mean noise adds small random variance).

**Prediction 2:** Oracle PPL (correct A) should show >= 20% improvement on specialized domains.
  The N=25 training experiment measured 35.2% average improvement with correct loading.
  We predict similar improvement when the same adapters are loaded correctly at eval time.

**Prediction 3:** Routing accuracy should improve significantly because hidden-state
differences between adapter-enhanced and base representations will be large enough to
separate domains. (With wrong A, all adapters produce noise, so all domains look alike.)

---

## E. Assumptions & Breaking Conditions

1. **A' entries are independent of B*** -- holds because A' is generated at model
   construction time (Kaiming init), B* was trained months ago with different A.
2. **B* has non-negligible Frobenius norm** -- if B* ~ 0, neither correct nor
   incorrect A matters. This holds: training produced measurable improvements.
3. **Grassmannian A matrices are meaningfully different from random** -- the
   Grassmannian initialization ensures near-orthogonality between domains'
   A matrices. Random A has no such structure.

If Assumption 2 fails (B* ~ 0), both correct and incorrect loading show ~0%
improvement, and the bug hypothesis is wrong (K596 triggers).

---

## F. Worked Example (d=4, r=2)

Trained pair: A = [[0.5, 0], [0, 0.5], [0.5, 0], [0, 0.5]], B = [[1, -1], [-1, 1]]
Delta_W = A @ B = [[0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, 0.5]]

Random A': [[0.3, -0.2], [0.1, 0.4], [-0.3, 0.1], [0.2, -0.3]]
Delta_W' = A' @ B = [[0.5, -0.5], [-0.3, 0.3], [-0.4, 0.4], [0.5, -0.5]]

For x = [1, 1, 1, 1]:
  Correct: x @ Delta_W = [0, 0] (specific learned perturbation)
  Wrong: x @ Delta_W' = [0.3, -0.3] (random perturbation)

The perturbations are completely different. Over a dataset, wrong perturbations
average toward zero while correct ones consistently apply the learned function.

---

## G. Complexity & Architecture Connection

No new parameters or computation. The fix is loading the correct A matrices from
the skeleton file before applying B weights. Same FLOPs, same memory.

The correct loading procedure per domain i:
1. Load skeleton: grassmannian_skeleton_n{N}.npz
2. For each layer l and projection key k:
   Set model.layers[l].{k}.lora_a = skeleton[f"layer_{l}_{k}_domain_{i}"]
3. Load adapter.npz (B weights only)
4. Apply B weights to model

---

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   **Correct A-B pairing ensures the learned perturbation AB is applied deterministically;
   incorrect pairing produces zero-mean noise by Theorem 1.**

2. Which existing theorem(s) does the proof build on?
   LoRA decomposition (Hu et al., 2021, arXiv:2106.09685), properties of random projections.

3. What specific numbers does the proof predict?
   - Wrong A: oracle PPL ~ base PPL (0% improvement) -- confirmed at -0.6%
   - Correct A: oracle PPL improvement >= 20% (based on training-time measurements of 35.2%)

4. What would FALSIFY the proof?
   The proof is wrong if correct A loading also shows ~0% improvement (K596 triggers),
   meaning B weights did not actually learn domain-specific perturbations.

5. How many hyperparameters does this approach add?
   0 -- this is a bug fix, not a new mechanism.

6. Hack check: Am I adding fix #N to an existing stack?
   No. This removes the root cause of 7 failed routing experiments.
