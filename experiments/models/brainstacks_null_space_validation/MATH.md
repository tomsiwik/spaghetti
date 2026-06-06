# Null-Space SVD Isolation on Ternary Adapters

## Type: Verification (Type 1)

**Paper:** Brainstacks (arXiv:2604.01152, §3.5)
**Prior finding:** Finding #270 — OPLoRA orthogonal projection reduces rho_k 99.9% but capacity interference is 80% of the problem. Finding #271 — ternary spectra are flat (gap 1.003–1.018).

## Problem Statement

Brainstacks guarantees zero forgetting via null-space projection: each new domain's adapter delta is projected into the orthogonal complement of all prior domains' principal output subspaces. The question is whether this guarantee holds on **ternary (1.58-bit) adapters** where quantization noise may violate orthogonality.

## Disease vs Symptom

**Disease:** Cross-domain interference in composed adapters — one domain's updates corrupt another's learned representations.

**Not treating symptoms:** We don't tune composition weights or add regularizers. We structurally eliminate interference by constraining each domain to an orthogonal subspace of output space.

## Theorem 1: Null-Space Projection Guarantees Zero Cross-Domain Interference

**Setup.** Let δ_i(x) ∈ ℝ^d be domain i's adapter output delta on input x. Collect n_samples deltas into D_i ∈ ℝ^{n×d}. Compute SVD: D_i = U_i Σ_i V_i^T. Take top-K right singular vectors V_i^K ∈ ℝ^{d×K}.

**Projection.** For domain j (j > i), form projector P_i = V_i^K (V_i^K)^T and constrain:
δ_j^{proj}(x) = δ_j(x) − δ_j(x) · P_i

**Theorem.** If V_i^K captures the full column space of D_i (i.e., rank(D_i) ≤ K), then for all x:
⟨δ_j^{proj}(x), v⟩ = 0  for all v ∈ colspan(V_i^K)

**Proof.** δ_j^{proj}(x) = δ_j(x) − δ_j(x) P_i = δ_j(x)(I − P_i). For any v = V_i^K c:
⟨δ_j^{proj}(x), v⟩ = δ_j(x)(I − P_i) V_i^K c = δ_j(x)(V_i^K − V_i^K) c = 0. QED.

## Theorem 2: Ternary Quantization Noise Bound on Subspace Leakage

**Setup.** Let δ(x) be the full-precision adapter output and δ_q(x) the ternary-quantized version. The quantization error is ε(x) = δ_q(x) − δ(x).

For ternary STE with scale α = mean(|W|), the per-element error satisfies |ε_j| ≤ α (worst case: rounding from ±0.5α to 0 or ±α).

**Theorem.** After null-space projection, the leakage of δ_q into prior domain i's subspace is bounded by:
‖P_i · δ_q^{proj}(x)‖ ≤ ‖P_i · ε(x)‖ ≤ √K · α_max · √d

where α_max = max over layers of mean(|B|), d = hidden dimension, K = number of principal directions.

**Proof.** δ_q(x) = δ(x) + ε(x). Null-space projection eliminates δ(x)'s component:
P_i δ_q^{proj}(x) = P_i(δ(x) + ε(x)) − P_i(δ(x) + ε(x))P_i

For the clean component: P_i δ(x)(I − P_i) = 0 (Theorem 1).
Remaining: P_i ε(x)(I − P_i) = P_i ε(x) − P_i ε(x) P_i.

The leakage ‖P_i ε(x)‖ is bounded by ‖P_i‖ · ‖ε(x)‖ = ‖ε(x)‖ (P_i is a projector with operator norm 1). With ε element-wise bounded by α_max: ‖ε(x)‖ ≤ α_max √d.

But P_i projects onto K dimensions, so: ‖P_i ε(x)‖ ≤ min(1, √(K/d)) · α_max √d = α_max √K. QED.

## Quantitative Predictions

For BitNet-2B-4T (d=2560, K=64):

1. **Subspace occupancy:** Each domain occupies K/d = 64/2560 = 2.5% of hidden space. 5 domains occupy 12.5%. Ample room for orthogonal subspaces.

2. **Cross-domain cosine similarity of principal directions:** With 5 random K=64 subspaces in d=2560, expected pairwise cosine of random unit vectors projected through different subspaces:
   E[|cos(v_i, v_j)|] ≈ √(K²/d²) ≈ K/d = 0.025
   **Prediction: cross-domain principal direction cosine < 0.1** (K687 threshold 0.2 — should pass easily)

3. **Forgetting bound:** With α_max ≈ 0.01 (typical ternary scale), leakage ‖P_i ε‖ ≤ 0.01 · √64 = 0.08 per hidden unit. Relative to adapter norms of O(1), this is ~8% leakage.
   **Prediction: per-domain val loss increase < 0.05** (K688 threshold 0.01 — TIGHT, may fail due to ternary noise)

4. **Gradient norm preservation:** Projection removes only the K-dimensional subspace. Fraction preserved = (d − K·N_prior) / d. For domain 5 (4 prior domains): (2560 − 256) / 2560 = 90%.
   **Prediction: ≥90% gradient norm preserved** (K689 threshold 95% — may be tight for last domain)

## Kill Criteria from Proof

- **K687 PASS if:** mean pairwise cosine of domain principal directions < 0.2 (proof predicts ~0.025)
- **K688 PASS if:** max per-domain val loss increase < 0.01 when evaluating with only that domain's stack active (proof predicts leakage bounded by α√K ≈ 0.08, but val loss impact depends on downstream sensitivity)
- **K689 PASS if:** null-space projection preserves >95% of active gradient norm (proof predicts ~90% for domain 5 with 4 prior domains using K=64 each)

## Behavioral Predictions

If null-space projection works on ternary:
- Domain-specific generation quality should be preserved when multiple domains are composed
- Adding a new domain should NOT degrade any prior domain's in-distribution task accuracy
- This is a structural guarantee — no tuning of composition weights needed

If it fails (ternary noise too large):
- The failure tells us that 1.58-bit quantization fundamentally limits subspace isolation
- Fallback: use larger K (capture more of ternary noise), or apply projection in weight space instead of output space

## Self-Test

1. **What is the failure mode?** Ternary quantization noise leaks into prior domains' subspaces, causing cross-domain interference despite null-space projection.

2. **What existing math applies?** SVD projection theory, Johnson-Lindenstrauss-style concentration for random subspace overlap, STE quantization error bounds.

3. **What guarantee makes failure impossible?** Theorem 1 guarantees zero leakage for clean (full-precision) deltas. Theorem 2 bounds ternary leakage at α√K.

4. **What specific numbers does the proof predict?** Cosine < 0.1, leakage norm ≤ 0.08, gradient preservation ≥ 90%.

5. **What kills the experiment?** Cross-domain cosine > 0.2, forgetting > 0.01, gradient preservation < 95%.

6. **How does this connect to the vision?** Null-space isolation enables zero-forgetting continual adapter composition — each new expert can be added without retraining or degrading existing ones. This is core to the "composable perturbation operators" architecture.
