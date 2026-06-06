# Shared/Unique Decomposition of Domain-Specific Capsule Groups

## 1. Setup

We have a shared-base composition protocol (validated in capsule_moe):

1. Pretrain base model on all data: parameters θ_base
2. Fine-tune only capsule groups on domain A (attention frozen): parameters θ_A
3. Fine-tune only capsule groups on domain B (attention frozen): parameters θ_B

Each capsule group g in layer l has two weight matrices:
- A^(l,g) ∈ ℝ^{n_caps × d} = ℝ^{64 × 64} — detector (rows are capsule receptive fields)
- B^(l,g) ∈ ℝ^{d × n_caps} = ℝ^{64 × 64} — expansion (columns are capsule output vectors)

The group computes: output = B @ ReLU(A @ x)

## 2. Delta Computation

For each group g in layer l, the fine-tuning delta is:

```
ΔA_k^(l,g) = A_k^(l,g) - A_base^(l,g)     for domain k ∈ {A, B}
ΔB_k^(l,g) = B_k^(l,g) - B_base^(l,g)
```

Since both domains start from θ_base with the same initialization, these deltas
are directly comparable — no alignment needed.

## 3. Shared/Unique Decomposition

For each group g in layer l:

```
shared_ΔA^(l,g) = (ΔA_A + ΔA_B) / 2
shared_ΔB^(l,g) = (ΔB_A + ΔB_B) / 2

unique_ΔA_A^(l,g) = ΔA_A - shared_ΔA = (ΔA_A - ΔA_B) / 2
unique_ΔA_B^(l,g) = ΔA_B - shared_ΔA = (ΔA_B - ΔA_A) / 2

unique_ΔB_A^(l,g) = ΔB_A - shared_ΔB = (ΔB_A - ΔB_B) / 2
unique_ΔB_B^(l,g) = ΔB_B - shared_ΔB = (ΔB_B - ΔB_A) / 2
```

**Key property**: unique_A = -unique_B (the unique components are exact negatives).

**Exact reconstruction**:
```
base + shared + unique_k = base + Δ_k     ∀ k ∈ {A, B}
```

This is exact in weight space. No approximation.

## 4. Why Not Procrustes?

Classical Procrustes alignment finds orthogonal R minimizing ||X - R @ Y||_F.
This is needed when experts have different parameterizations (independent training).

In our shared-base protocol, alignment is free: both domains start from identical
weights, so capsule indices already correspond. Procrustes would find R ≈ I.

Furthermore, Procrustes rotation in capsule space breaks the ReLU nonlinearity:
ReLU(R @ v) ≠ R @ ReLU(v) for general orthogonal R (only permutation matrices
preserve ReLU). So even if alignment were needed, we'd incur approximation error.

The shared-base protocol makes alignment trivial, yielding exact decomposition.

## 5. The Dilution Problem

**Task arithmetic (naive merge)**: W_merged = W_base + λ(Δ_A + Δ_B), λ = 0.5

For a domain-A input: the model applies 0.5 × Δ_A (relevant) + 0.5 × Δ_B (irrelevant).
Each expert contributes at 50% strength → dilution.

**Shared/unique decomposition with routing**: For a domain-A input, apply:
- shared (always on, full strength) = (Δ_A + Δ_B) / 2
- unique_A (routed, full strength) = (Δ_A - Δ_B) / 2
- Total: shared + unique_A = Δ_A at 100% strength. No dilution.

**The routing robustness hypothesis**: With decomposition, the shared component
carries most of the weight (both domains learned similar things from similar data).
Routing errors only affect the small unique correction. With concatenation, routing
errors affect the full expert output.

## 6. Nonlinearity Consideration

The decomposition is exact in weight space:
```
(A_base + shared_ΔA + unique_ΔA_k) = (A_base + ΔA_k)     — identical weights
```

But the decomposed MODEL computes shared and unique contributions separately:
```
decomposed_output = shared_group(x) + routed_unique_group(x)
                  = (B_base+shared) @ ReLU((A_base+shared) @ x)
                  + (B_unique) @ ReLU((A_unique) @ x)
```

This is NOT equal to:
```
original_output = (B_base+Δ_k) @ ReLU((A_base+Δ_k) @ x)
```

Because ReLU is nonlinear: f(a+b) ≠ f(a) + f(b).

The approximation quality depends on ||unique|| / ||base+shared||. When unique
weights are small relative to base+shared, the unique groups contribute small
corrections and the decomposed output approximates the original well.

## 7. Worked Example at d=64

Given: n_caps = 64, d = 64, n_groups = 4, n_layers = 4

Per group: A ∈ ℝ^{64×64}, B ∈ ℝ^{64×64} → 8192 params per group
Per layer: 4 groups → 32768 capsule params
Total capsule params: 4 layers × 32768 = 131072

After 300 steps of fine-tuning from base:
- Typical ||Δ|| / ||W_base|| ≈ 0.1-0.3 (small relative change)
- If domains learn 70% shared knowledge: ||shared|| ≈ 0.7 ||Δ||, ||unique|| ≈ 0.3 ||Δ||

Composed models:
- Concatenation: 8 groups, k=4 → 65536 capsule params/layer, router 64→8
- Task arithmetic: 4 groups, k=2 → 32768 capsule params/layer, no domain routing needed
- Decomposed: 4 shared + 8 unique → 98304 capsule params/layer, router 64→8

## 8. Assumptions

1. Shared-base protocol produces comparable deltas (validated in capsule_moe).
2. Fine-tuning deltas are small relative to base weights.
3. The two domains share sufficient structure for shared component to be non-trivial.
4. Separate nonlinear evaluation of shared + unique approximates joint evaluation.
5. Router can learn to select appropriate unique groups during calibration.

## 9. Falsification Criteria

| Criterion | Threshold | Kill if |
|-----------|-----------|---------|
| Decomposed composition vs joint | <5% degradation | >5% |
| Decomposed vs concatenation | Must not be worse | Worse |
| Shared fraction of delta norm | >20% | <5% |
| Uniform-unique routing quality | Better than concatenation+uniform | Worse |

## 10. Computational Cost

Training: Same as capsule_moe composition (reuse protocol).
Decomposition: 4 layers × 4 groups × SVD/averaging = negligible.
New model: 12 groups vs 8 groups = 50% more capsule params.
Calibration: 100 steps of router-only training.

Total experiment time: <5 minutes.
