# SVD Extraction Quality: Mathematical Foundations

## Type: Proof Verification (Type 1) + Guided Exploration (Type 2)

The Eckart-Young theorem provides the proof framework. The unknown is the
empirical rank-quality tradeoff for our specific adapters.

---

## A. Failure Mode Identification

**The disease:** LoRA adapters trained at scale=20 destroy base model knowledge
(Finding #320: -60pp MMLU on Qwen3-4B). The adapter perturbation
delta = scale * B^T @ A^T has rank=16 and magnitude proportional to scale.
At scale=20, the perturbation magnitude overwhelms the spectral gap of the
base model's knowledge-encoding subspace (Davis-Kahan sin-theta theorem).

**The question this experiment answers:** Can truncated SVD remove the
destructive components while preserving the useful domain-specific directions?

**Why this is a real risk:** Finding #228 (Bridge extraction killed) showed
that the TOP SVD components carry useful signal — partially undoing them hurts
quality. This means naive rank reduction might discard exactly the directions
that matter.

---

## B. The Right Question (Reframe)

**Wrong:** "How do we reduce adapter magnitude to preserve MMLU?"
(This is the scale dilemma: low scale preserves MMLU but provides no benefit.)

**Right:** "What is the optimal rank-r approximation to the adapter delta
such that the reconstruction error is minimized in the operator norm?"

This question has a closed-form answer from 1936.

---

## C. Prior Mathematical Foundations

### Theorem (Eckart-Young-Mirsky, 1936)

Let A be an m x n matrix with singular value decomposition A = U Sigma V^T,
where sigma_1 >= sigma_2 >= ... >= sigma_p (p = min(m,n)).

For any rank-r matrix B:

||A - B||_F >= sqrt(sigma_{r+1}^2 + ... + sigma_p^2)

with equality achieved by the truncated SVD: B* = U_r Sigma_r V_r^T.

**Reference:** Eckart & Young (1936), "The approximation of one matrix by
another of lower rank." Psychometrika 1(3):211-218.

### Corollary: Energy Preservation

The fraction of energy (Frobenius norm squared) preserved at rank r is:

E(r) = (sigma_1^2 + ... + sigma_r^2) / (sigma_1^2 + ... + sigma_p^2)

For our adapters: delta = scale * B^T @ A^T has rank at most min(rank_lora, ...) = 16.
Therefore:
- At r >= 16: E(r) = 1.0 (lossless — the delta is already rank 16)
- At r < 16: E(r) < 1.0 (lossy — we discard the (16-r) smallest directions)

### Theorem (Davis-Kahan sin-theta, 1970)

For a symmetric matrix M with eigenvalue gap delta_gap between the k-th and
(k+1)-th eigenvalues, and a perturbation E with ||E||_op <= epsilon:

sin(theta) <= epsilon / delta_gap

where theta is the angle between the original and perturbed eigensubspaces.

**Implication for SVD extraction:** Truncating the adapter to rank r reduces
the perturbation magnitude from ||delta||_F to ||delta - delta_r||_F.
If this brings the perturbation below the spectral gap of the base model's
knowledge subspace, MMLU should be preserved.

---

## D. Proof of Guarantee

**Theorem 1 (Lossless SVD at native rank).**
Let delta = scale * B^T @ A^T where A is (d_in, r_lora) and B is (r_lora, d_out).
Then rank(delta) <= r_lora. The truncated SVD at rank r = r_lora satisfies:

delta_r = delta (exactly, up to floating-point precision)

*Proof.* delta = scale * B^T @ A^T is the product of a (d_out x r_lora) matrix
with a (r_lora x d_in) matrix. By the rank inequality for matrix products,
rank(delta) <= min(rank(B^T), rank(A^T)) <= r_lora.

Since delta has rank at most r_lora, its SVD has at most r_lora nonzero
singular values. Truncating at r = r_lora preserves all nonzero singular
values, so delta_r = delta. QED.

**Corollary 1.1.** SVD extraction at rank 16 (= LoRA rank) is lossless.
The reconstructed expert is mathematically identical to the raw LoRA adapter.

**Theorem 2 (Monotonic quality degradation with rank reduction).**
For r_1 < r_2 <= r_lora:

||delta - delta_{r_1}||_F >= ||delta - delta_{r_2}||_F

*Proof.* By Eckart-Young, ||delta - delta_r||_F^2 = sum_{i=r+1}^{rank} sigma_i^2.
Since r_1 < r_2, this sum includes strictly more terms for r_1 than r_2, so
||delta - delta_{r_1}||_F^2 >= ||delta - delta_{r_2}||_F^2. QED.

**Implication:** PPL should degrade monotonically as rank decreases.

**Hypothesis 3 (SVD as implicit regularization).**
The truncated SVD at rank r < r_lora removes the (r_lora - r) smallest singular
directions. If the destructive components (those that rotate knowledge subspaces
per Davis-Kahan) are concentrated in the small singular values, then:

||delta_r||_op < ||delta||_op

and the Davis-Kahan bound tightens: sin(theta_r) < sin(theta_original).

This is the FlexMoRE observation: 5/6 experts IMPROVED after SVD extraction
because truncation acted as a regularizer, removing noise while preserving signal.

*Note:* This is NOT guaranteed. If the destructive components are in the TOP
singular values, truncation will not help (and may worsen things). This is an
empirical question (Type 2: guided exploration).

---

## D. Predictions

### Quantitative (derived from Theorem 1, 2)

| Prediction | Source | Expected Value |
|------------|--------|----------------|
| P1: SVD at rank=16 is lossless | Theorem 1 | reconstruction error = 0, PPL ratio = 1.000 |
| P2: SVD at rank=32,64,128 is lossless | Theorem 1 (rank(delta)=16) | reconstruction error = 0, PPL ratio = 1.000 |
| P3: Quality degrades monotonically | Theorem 2 | PPL ratio: r=4 > r=8 > r=16 |
| P4: Best rank mean PPL ratio < 2.0 | Kill criterion K834 | At rank 16: ratio = 1.0 (Thm 1) |

### Behavioral (from Hypothesis 3, Type 2 exploration)

| Prediction | Source | Note |
|------------|--------|------|
| P5: Some ranks < 16 may match or beat raw LoRA | FlexMoRE analogy | Type 2: unknown which rank |
| P6: SVD composition may preserve MMLU better | Davis-Kahan + Thm 3 | If destructive components are in small SVs |

### Spectral distribution (Type 2: unknown)

The key empirical question: how are the singular values of scale*B^T@A^T distributed?
- If concentrated in top few: low-rank extraction preserves nearly all signal
- If flat (all SVs similar): every truncation loses proportional signal
- FlexMoRE reports knowledge tasks peak at r=4, reasoning at r=2896
- Our adapters are rank-16 LoRA, so the effective rank is at most 16

---

## E. Assumptions & Breaking Conditions

1. **Adapter delta is exactly rank r_lora.** If numerical errors or training
   dynamics create effective rank < 16, some SVs may be near-zero even at rank < 16.
   *Breaking:* If adapters have effective rank 8, then rank-8 SVD is already lossless.
   This would be a POSITIVE outcome (more compression).

2. **PPL tracks reconstruction quality.** If the relationship between
   reconstruction error and PPL is non-monotonic, Theorem 2's prediction fails.
   *Breaking:* PPL jumps at specific ranks (threshold effect).

3. **Base model is deterministic.** We reload the base model for each evaluation.
   MLX quantized models should produce identical outputs for identical inputs.
   *Breaking:* Non-determinism from different quantization or loading order.

---

## F. Worked Example (rank=16, d_in=2560, d_out=2560)

Consider one module (self_attn.q_proj) with:
- A: (2560, 16) — Grassmannian frozen A-matrix
- B: (16, 4096) — trained B-matrix for q_proj (out_features=4096 for Qwen3-4B GQA)
- scale = 20.0

delta = 20.0 * B^T @ A^T
      = 20.0 * (4096, 16) @ (16, 2560)
      = (4096, 2560) matrix of rank <= 16

SVD: delta = U @ diag(S) @ V^T
- U: (4096, 16), S: (16,), V^T: (16, 2560) [only 16 nonzero SVs]

At rank 8:
- delta_8 = U[:,:8] @ diag(S[:8]) @ V^T[:8,:]
- Energy preserved: E(8) = sum(S[:8]^2) / sum(S[:16]^2)
- If S is approximately geometric (s_i ~ s_1 * rho^i), then:
  E(8) = (1 - rho^16) / (1 - rho^32) * (1 - rho^16) / ... [depends on rho]

At rank 16:
- delta_16 = U @ diag(S) @ V^T = delta (exactly)
- Energy preserved: E(16) = 1.0

---

## G. Complexity & Architecture Connection

**SVD extraction (offline, one-time):**
- Per module: O(m * n * r_lora) for SVD of an (m, n) matrix of rank r_lora
- Total: 7 modules/layer * 36 layers * 5 domains = 1260 SVD operations
- Each SVD is fast because rank is only 16

**SVD expert inference (runtime):**
- Same as LoRA: y = base(x) + x @ A_svd @ B_svd
- A_svd: (d_in, r_svd), B_svd: (r_svd, d_out)
- At r_svd < 16: strictly cheaper than raw LoRA (fewer FLOPs)
- At r_svd = 16: identical cost to raw LoRA

**Storage:**
- Raw LoRA: rank * (d_in + d_out) parameters per module
- SVD expert at rank r: r * (d_in + d_out) parameters per module
- At r < 16: proportionally smaller

---

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   The Eckart-Young theorem guarantees that truncated SVD is the OPTIMAL
   rank-r approximation — no other rank-r matrix has smaller reconstruction error.

2. Which existing theorem(s) does the proof build on?
   Eckart-Young-Mirsky (1936), Davis-Kahan sin-theta (1970).

3. What specific numbers does the proof predict?
   SVD at rank >= 16 has zero reconstruction error and PPL ratio = 1.000.
   PPL degrades monotonically with decreasing rank.

4. What would FALSIFY the proof (not just the experiment)?
   The proof is wrong if rank(scale * B^T @ A^T) > r_lora. This would require
   A or B to have rank > r_lora, which is impossible since they are (d, r_lora)
   and (r_lora, d) matrices.

5. How many hyperparameters does this approach add?
   Count: 1 (the SVD truncation rank). This is the exploration target —
   the optimal rank per domain is unknown and may vary by task type
   (FlexMoRE: knowledge=4, reasoning=2896).

6. Hack check: Am I adding fix #N to an existing stack?
   No. SVD extraction is a single mathematical operation (truncated SVD)
   applied to the existing adapter delta. It does not add mechanisms.
