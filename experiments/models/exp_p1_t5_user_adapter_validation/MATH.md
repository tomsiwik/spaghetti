# MATH.md — T5.2: Validate User-Submitted Adapter Before Integration

## Theorem 1 (Principal Subspace Divergence)

**Setting:** Let Ã_user ∈ ℝ^{d_in × r_u} and Ã_domain ∈ ℝ^{d_in × r_d} be the lora_a matrices
of a user adapter and a domain adapter at the same layer (mlx_lm convention: lora_a has shape
(d_in, r)). Let U_user, U_domain be their column-orthonormal bases (via QR decomposition).

**Definition:** The maximum principal angle cosine between the two subspaces is:

    σ₁(U_user^T U_domain)

where σ₁ is the largest singular value of the (r_u × r_d) matrix U_user^T U_domain.

**Theorem:** σ₁ ∈ [0, 1]. σ₁ = 0 ⟺ subspaces are orthogonal. σ₁ = 1 ⟺ subspaces coincide.

**Proof:** By definition, U_user has orthonormal columns (||u_i|| = 1, u_i^T u_j = 0 for i≠j).
Similarly for U_domain. Therefore all singular values of U_user^T U_domain are bounded in [0,1].
QED.

**Prediction for K1100:** Trained adapters initialized via Kaiming uniform (mlx_lm default) occupy
random subspaces in ℝ^{2560}. For random subspaces of rank 4 vs rank 6 in ℝ^{2560}:

    E[σ₁] ≈ √(r_u + r_d) / √d_in = √10 / √2560 ≈ 0.062

But trained adapters may show higher alignment from shared training signal. From T3.3
(Finding #425): real trained adapters exhibited max_cos up to 0.596. We predict:

    K1100 measured max|cos| ∈ [0.30, 0.70]

**K1100 Pass Condition:** max|cos| < 0.95 (i.e., user adapter is not a near-duplicate of
an existing adapter). Under exclusive routing (T3.6, Finding #429), interference is
structurally zero regardless of σ₁ — routing guarantees only one adapter activates per request.
The orthogonality check is a structural quality indicator, not a safety requirement.

---

## Theorem 2 (Routing Invariance Under Non-Grassmannian Adapters)

**Prior result (T3.6, Finding #429):** Under exclusive routing, hot-adding adapter k produces
0/40 output changes (bit-exact) for existing adapters j ≠ k.

**Corollary:** For user adapter validation, the σ₁ value from Theorem 1 is IMMATERIAL for
correctness under exclusive routing. Validation's purpose is:
1. Detecting degenerate cases (near-duplicate adapters, σ₁ ≥ 0.95)
2. Measuring quality improvement (K1101)
3. Safety screening (K1102)
4. Scale sanity check (K1103)

---

## Theorem 3 (LoRA Scale Bound)

**Claim:** The effective weight change magnitude is bounded by lora_a's Frobenius norm:

    ||ΔW||_F = ||B · A^T||_F ≤ ||B||_F · ||A||_F

where A = lora_a^T ∈ ℝ^{r × d_in}, B = lora_b^T ∈ ℝ^{d_out × r}.

At training completion, ||A||_F reflects the scale of the learned input subspace.
For scale check K1103, we compare the per-layer ||lora_a||_F of the user adapter to the
median of domain adapters (math, code, medical across overlapping layers 26-41).

**Prediction for K1103:** Since user adapter (rank=4) and domain adapters (rank=6) use the
same d_in=2560 with similar initialization scales, we predict:

    median_user_norm / median_domain_norm ∈ [0.5, 1.5]

K1103 passes if user_median_norm ∈ [0.5×, 2×] domain_median_norm.

---

## Theorem 4 (Thinking Suppression Isolation)

**T5.1 confound (identified by adversarial reviewer):** T5.1 used max_tokens=120.
The base model outputs `<|channel>thought\n[thinking chain]` which exceeds 120 tokens,
causing truncation BEFORE the sign-off. The adapter learned to suppress thinking tokens
(going straight to answer + sign-off), making truncation less likely to cut off the sign-off.

**This confound conflates:** (a) style injection (learning sign-off phrase) with (b) format
change (suppressing thinking channel).

**T5.2 isolation (K1101):** Use max_tokens=256, which provides sufficient budget for both
thinking chain and sign-off. Under this condition:
- Base model (thinking → answer, no sign-off): 0% compliance expected
- Adapter (may or may not suppress thinking, but must reach sign-off): compliance predicted
  to be LOWER than T5.1's 76% since the model has room to complete thinking, then must
  still produce sign-off at the end.

**Prediction for K1101:** compliance rate ≥ 30% (lower than 76% due to thinking budget,
but still substantially above 0% base).

K1101 passes if adapter compliance > base compliance (base = 0%).

---

## Kill Criteria Predictions

| Criterion | Metric | Predicted Value | Pass Threshold | Expected |
|-----------|--------|-----------------|----------------|----------|
| K1100 | max|cos(A_user, A_domain)| across layers | 0.30–0.70 | < 0.95 | **PASS** |
| K1101 | style compliance rate, max_tokens=256 | ≥ 30% (vs 0% base) | > 0% | **PASS** |
| K1102 | sensitive prompts flagged / 5 | 0/5 | 0/5 | **PASS** |
| K1103 | user_norm / domain_median_norm | 0.5–1.5 | in [0.5×, 2×] | **PASS** |
| K1104 | validation wall time (excl. model load) | ~20–40s | < 60s | **PASS** |

---

## Connection to Architecture

This experiment validates the T5 user-training story. The validation pipeline is a
**gatekeeping function** for the Room Model (scratchpad reference): adapters are rooms
in the combined weight space. Before adding a new room, we verify:
1. It doesn't overlap an existing room (K1100)
2. It actually improves behavior (K1101)
3. It doesn't make the building unsafe (K1102)
4. It's architectural scale-compatible (K1103)
5. The gate check is fast (K1104 — supports real-time pipeline)
