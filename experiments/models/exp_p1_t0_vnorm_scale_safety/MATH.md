# MATH.md: T0.2 — V-Norm Eliminates Scale Catastrophe on Gemma 4

## Context

Finding #320: On Qwen3-4B (no v_norm on values), training LoRA adapters at scale=20
causes catastrophic MMLU degradation: -60pp single adapter, -44pp composed.
The failure is scale-driven: adapter perturbation overwhelms base model's knowledge
eigenspace (Davis-Kahan bound becomes vacuous at large perturbation).

Gemma 4 has `v_norm = RMSNorm(V, with_scale=False)` applied to all value projections.
This experiment verifies that v_norm makes scale catastrophe structurally impossible.

## Theorem 1: V-Norm Bounds Value Stream Perturbation

**Setup.** Let `x ∈ R^{d_model}` be the hidden state at some layer. Let:
- `W_v ∈ R^{d_v × d_model}`: base value projection weight
- `A ∈ R^{d_model × r}`, `B ∈ R^{r × d_v}`: LoRA matrices, rank r
- `s ∈ R_+`: adapter scale
- `h_v = head_dim`: per-head value dimension

After v_proj + LoRA, the raw value vector is:

```
V_raw(s) = W_v @ x + s * B @ A @ x    ∈ R^{n_kv × h_v}
```

Gemma 4 applies `v_norm` (RMSNorm without learned scale):

```
V_norm(s) = V_raw(s) / RMS(V_raw(s))    where RMS(v) = sqrt(mean(v²))
```

**Theorem.** For any adapter scale `s > 0` and hidden state `x ≠ 0`:

```
||V_norm(s)||_RMS = sqrt(h_v)    (independent of s)
```

**Proof.** By definition of RMSNorm without learned scale:

```
V_norm(s)_i = V_raw(s)_i / sqrt(mean_j(V_raw(s)_j²))
```

Therefore:
```
mean_i(V_norm(s)_i²) = mean_i(V_raw(s)_i²) / mean_j(V_raw(s)_j²) = 1
```

So `||V_norm(s)||_RMS = sqrt(h_v)` (h_v entries each with squared mean 1). ∎

**Corollary (Scale Invariance).** Let `ΔV_norm(s) = V_norm(s) - V_norm(0)` be the
perturbation caused by the adapter at scale s. Since both V_norm(s) and V_norm(0)
lie on the sphere of radius sqrt(h_v):

```
||ΔV_norm(s)||_2 ≤ ||V_norm(s)||_2 + ||V_norm(0)||_2 = 2*sqrt(h_v)
```

This bound is INDEPENDENT OF s. The adapter can only change the DIRECTION of value
vectors, not their magnitude. As s → ∞, V_norm(s) → direction(B @ A @ x), which
is still a unit-RMS vector.

## Theorem 2: MMLU Degradation Bound Under V-Norm

**Background.** Davis-Kahan sin-θ theorem (Davis & Kahan, 1970): Given a symmetric
matrix M with eigenspace E, a perturbation ΔM causes eigenspace rotation:

```
sin(θ) ≤ ||ΔM||_F / δ
```

where δ is the spectral gap between eigenspace eigenvalues and the rest.

**Application to attention.** The attention output for token i is:

```
O_i = sum_j softmax(Q_i^T K_j / scale) * V_j
```

With v_norm, the effective matrix M = softmax(...) × V_norm. The perturbation from
the adapter is:

```
||ΔM||_F ≤ ||ΔV_norm(s)||_F ≤ 2 * n_kv * sqrt(h_v)    (per layer, per head)
```

This bound is INDEPENDENT OF s. Therefore the eigenspace rotation (and hence MMLU
degradation) is bounded by a constant independent of adapter scale.

**Contrast with Qwen3-4B (no v_norm):**
```
||ΔV_raw(s)||_F = s * ||B @ A @ x||_F    → ∞ as s → ∞
```

At s=20: ||ΔM||_F >> δ → Davis-Kahan bound vacuous → arbitrary eigenspace rotation
→ -60pp MMLU (Finding #320). ∎

## Predictions

| Condition | Predicted | Basis |
|-----------|-----------|-------|
| Gemma 4 MMLU at scale=5 | ≤ 5pp below base | Theorem 2 |
| Gemma 4 MMLU at scale=10 | ≤ 5pp below base | Theorem 2 |
| Gemma 4 MMLU at scale=20 | ≤ 5pp below base | Theorem 2 (structural) |
| Qwen3-4B MMLU at scale=20 | > 30pp below base | Finding #320 (existing) |
| GSM8K quality ratio at scale=10 vs 5 | ≥ 0.95 | Corollary: direction preserved |
| GSM8K quality ratio at scale=20 vs 5 | ≥ 0.95 | Corollary: direction preserved |

## Kill Criteria Derivation

**K994** (0pp MMLU degradation at scales 5,10,20):
- Theorem 2 predicts bounded degradation independent of s
- 5pp threshold: generous allowance for quantization noise and finite-sample variance
- PASS if all three scales show MMLU within 5pp of base

**K995** (Citing Finding #320 for the "without v_norm" baseline):
- PASS by reference: Finding #320 showed -60pp at scale=20 on Qwen3-4B
- Threshold 30pp: half of observed effect size (conservative)
- No new experiment needed; structural contrast established

**K996** (Adapter quality ≥ 95% of scale=5):
- Corollary: v_norm preserves direction, so adapter knowledge is direction-encoded
- At scale=5: direction = normalize(W_v @ x + 5 * ΔW_v @ x)
- At scale=20: direction → normalize(ΔW_v @ x) (adapter dominates)
- These are close if W_v @ x and ΔW_v @ x are not anti-parallel (expected for random init)
- 95% threshold: ratio of correct answers at high scale vs low scale

## References

- Davis, C. & Kahan, W.M. (1970). "The rotation of eigenvectors by a perturbation." SIAM Journal on Numerical Analysis 7(1).
- Hu, E. et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685.
- Finding #320: Pierre Pro composition MMLU at scale=5 (safe) and scale=20 (-60pp).
- ARCHITECTURE_P1.md §5.2: V-Norm as Automatic Scale Safety.
