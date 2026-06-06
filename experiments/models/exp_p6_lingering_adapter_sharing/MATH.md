# MATH: P6.B0 — Adapter Sharing Flywheel

## Setup

N users each observe a subset S_i of M domain facts, where |S_i| = K < M.
Each user trains a rank-r LoRA with frozen A (shared random projection) and trainable B_i.

**Notation:**
- M = 10 facts, K = 6 facts per user, N = 10 users
- A ∈ R^{d_in × r}: shared frozen projection (random normal initialization)
- B_i ∈ R^{r × d_out}: user i's trained output matrix
- Coverage c_j = |{i : j ∈ S_i}|: number of users who know fact j

**Fact assignment:** Sliding window — user i gets facts {i, i+1, ..., i+K-1} mod M.
This gives uniform coverage c_j = K for all j.

---

## Theorem 1 (Shared Initialization → Approximately Shared A)

**Claim:** When N users train LoRA (both A and B) from identical initialization for T steps,
the A-matrices remain approximately shared, enabling B-crystallization.

**Proof:** All users start from A_0 (same random init via seed). After T gradient steps at
learning rate η, user i has A_i = A_0 + η Σ_{t=1}^T g_t^{(i)} where g_t are stochastic
gradients. The A-drift is ||A_i - A_0|| ≤ η T G_max where G_max bounds the gradient norm.

For T=20, η=1e-3, G_max ≈ 1: ||A_i - A_0|| ≤ 0.02. With ||A_0|| ≈ √(d_in/r) ≈ √640 ≈ 25,
the relative drift is < 0.1%. Crystallizing by averaging A: A_crystal = mean(A_i) ≈ A_0.

**Consequence:** Averaging both A and B across users is valid because A's are nearly identical.
The B-averaging concentrates domain signal while noise cancels (Theorem 2). QED.

---

## Theorem 2 (Coverage Union via Crystallization)

**Claim:** Crystallized adapter B_crystal covers all M facts when min_j c_j > 0.

**Proof:** Assume weight-space linearity: each user's B_i encodes their observed facts as
B_i = Σ_{j ∈ S_i} β_j v_j + ε_i, where v_j ∈ R^{r × d_out} is the direction encoding fact j,
β_j > 0 is the learned magnitude, and ε_i is training noise with E[ε_i] = 0.

Crystallized adapter:
  B_crystal = (1/N) Σ_{i=1}^N B_i = Σ_j (c_j/N) β_j v_j + (1/N) Σ_i ε_i

For fact j:
  Signal: (c_j/N) × β_j (present for ALL facts with c_j > 0)
  Noise:  ||(1/N) Σ ε_i|| = O(σ/√N) by CLT

With uniform coverage c_j = K = 6, N = 10:
  Signal per fact: 0.6 × β_j
  Noise: σ/√10 ≈ 0.32σ

**Individual user accuracy:** User i knows K of M facts. If generation threshold τ requires
signal > τ, then user i answers correctly on facts with β_j > τ (up to K facts).
Expected: ~K/M = 60%.

**Crystal accuracy:** Crystal has signal 0.6×β on ALL M facts. If 0.6×β > τ, crystal answers
all M facts correctly. Even if some are below threshold, crystal covers more facts than any
individual (union > part).

**Prediction:** Crystal accuracy ≥ best individual accuracy + 5pp (K1291).
- Best individual: ~60-70% (6/10 facts, most learned)
- Crystal: ~70-100% (all 10 facts, attenuated but noise-reduced)
- Expected margin: ≥ 10pp QED.

---

## Theorem 3 (Promotion via Adapter Initialization is Exact)

**Claim:** Initializing a new user's LoRA to (A_crystal, B_crystal) on the original base
is equivalent to training on a promoted base W_base + γ·(A_crystal @ B_crystal)^T.

**Proof (by construction):**
For any input x, the output with LoRA set to crystal values:
  y = x @ W_base^T + γ · x @ A_crystal @ B_crystal

This equals x @ W_promoted^T where W_promoted = W_base + γ·(A_crystal @ B_crystal)^T.
Since the outputs are identical, loss, gradients, and training dynamics from this state
are identical. No need to modify base weights (critical for quantized models). QED.

---

## Theorem 4 (Promoted Base Improves New-User Experience)

**Claim:** A new user training on the promoted base achieves higher accuracy than
the same user training on the original base.

**Proof:** New user has facts S_new ⊂ [M] with |S_new| = K. Their adapter learns B_new.
Effective output on promoted base:

  y = x @ W_promoted^T + γ_lora · x @ A @ B_new
    = x @ W_base^T + γ · x @ A @ B_crystal + γ_lora · x @ A @ B_new
    = x @ W_base^T + x @ A @ (γ·B_crystal + γ_lora·B_new)

The new user's effective knowledge = crystal (all M facts) + adapter (K facts).
On facts NOT in S_new: crystal provides signal (c_j/N)·β_j > 0.
On facts in S_new: crystal + adapter provides (c_j/N)·β_j + β_new_j (reinforced).

Control (original base): only adapter signal on K facts, 0 on M-K facts.

**Prediction:** Improvement ≥ 3pp (K1292).
- Control: ~50-60% (6/10 facts from adapter)
- Promoted: ~70-90% (10/10 facts from crystal + adapter synergy)
- Expected margin: ≥ 20pp QED.

---

## Quantitative Predictions

| Metric | Predicted | Kill Criterion | Source |
|--------|-----------|---------------|--------|
| Crystal accuracy | 70-100% | K1291: ≥ best_individual + 5pp | Theorem 2 |
| Best individual | 50-70% | — | P6.A0 baseline (60% with 10/10 facts) |
| New user (promoted) | 70-90% | K1292: ≥ control + 3pp | Theorem 4 |
| Control user | 50-60% | — | Same capacity as individual |
| Total time | ~5-7 min | K1293: < 10 min | P6.A0 latency extrapolation |

**Key behavioral prediction:** The crystal adapter should answer questions about facts
that NO individual user was trained on but that appear in the union of all users' knowledge.
This is the "users train the model by sharing adapters" flywheel.
