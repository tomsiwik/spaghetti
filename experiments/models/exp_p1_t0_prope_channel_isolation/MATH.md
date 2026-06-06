# T0.3: p-RoPE Semantic Channels Are Position-Invariant

**Experiment:** exp_p1_t0_prope_channel_isolation  
**Type:** verification  
**Date:** 2026-04-09

---

## Theorem 1 (p-RoPE Position Invariance)

**Statement:** In Gemma 4's proportional RoPE, all dimensions d ∈ [rope_dim, head_dim)
have inv_freq[d//2] = 0. Therefore RoPE(v_d, θ) = v_d for every position θ, making
any adapter restricted to these dimensions algebraically position-invariant.

**Proof:**

Gemma 4 global attention layer parameters (from config.json):
- head_dim = 512
- partial_rotary_factor = 0.25

Step 1: Compute rope_dim (number of RoPE-rotated scalars):
```
rope_dim = int(partial_rotary_factor * head_dim) = int(0.25 * 512) = 128
```

Step 2: _compute_proportional_rope_parameters (Gemma4 source, modeling_rope_utils.py):
```python
inv_freq = 1 / (base ** (2 * torch.arange(rope_dim // 2) / head_dim))
# inv_freq has shape (rope_dim // 2,) = (64,)
# Padded to (head_dim // 2,) = (256,) with zeros:
inv_freq_padded = F.pad(inv_freq, (0, head_dim // 2 - rope_dim // 2))
# inv_freq_padded[0:64]  = base^(-2k/512) for k=0..63  [RoPE, > 0]
# inv_freq_padded[64:256] = 0                            [NoPE, exactly 0]
```

Step 3: RoPE transform for dim d (standard pair rotation):
```
[v_{2d}, v_{2d+1}] → [v_{2d}·cos(θ·ω_d) - v_{2d+1}·sin(θ·ω_d),
                       v_{2d}·sin(θ·ω_d) + v_{2d+1}·cos(θ·ω_d)]
where ω_d = inv_freq_padded[d]
```

Step 4: For NoPE dimensions (d ∈ [64, 255], covering scalar indices [128, 511]):
```
ω_d = inv_freq_padded[d] = 0
cos(θ · 0) = 1,  sin(θ · 0) = 0  for all θ ∈ ℝ
[v_{2d}, v_{2d+1}] → [v_{2d}·1 - v_{2d+1}·0, v_{2d}·0 + v_{2d+1}·1] = [v_{2d}, v_{2d+1}]
```

Since this holds for ALL θ (any position), the NoPE dimensions are unchanged by RoPE
regardless of token position. **QED**

**Corollary:** An adapter W_adapt restricted to NoPE scalar dimensions [128, 511] produces
identical output at any token position. If W_adapt depends only on token content (not position),
its output is purely semantic — no position encoding leakage.

---

## Theorem 2 (Adapter Capacity in NoPE Subspace)

**Statement:** A rank-r adapter restricted to the NoPE subspace (D_nope = 384 dims) has
equivalent expressive capacity to a full-dim rank-r adapter for tasks where signal is
distributed across head_dim dimensions proportionally.

**Proof:**

A rank-r LoRA adapter in D-dimensional space:
```
Δh = x @ A @ B,  A ∈ ℝ^{D×r},  B ∈ ℝ^{r×d_out}
```

The column space of A spans an r-dimensional subspace of ℝ^D. The adapter can represent
any linear map in that r-dimensional subspace.

For NoPE-only adapter (A_nope ∈ ℝ^{D_nope × r}, applied to x[128:512]):
- D_nope = 384, r = 4
- D_nope >> r → subspace is low-rank in D_nope, not constrained by dimension
- The number of representable r-dimensional subspaces = Gr(r, D_nope)
  (same as Gr(r, D_full) in the sense that both are unconstrained)

For a classification task with labels y = sign(w^T x) where w ~ N(0, I_{512}):
- Signal in NoPE dims: E[||w_nope||²] = D_nope / D_full = 384/512 = 0.75
- The adapter recovers the projection of w onto its subspace
- NoPE-only adapter sees fraction η = D_nope / D_full = 0.75 of signal
- Expected quality ratio: ≈ η^{1/2} = √0.75 ≈ 0.866 for linear problems
  (this is a LOWER BOUND — real tasks concentrate signal in semantic dims)

Reference: Aghajanyan et al. 2021 (arxiv 2012.13255) — "Intrinsic Dimensionality Explains
the Effectiveness of Language Model Fine-Tuning" — tasks fine-tuned with 100-10K dims
sufficient; 384 NoPE dims >> typical task intrinsic dimension.

**Kill criterion K999:** NoPE_quality >= 0.90 * full_quality
**Prediction:** Expected ratio ≥ 0.866 (lower bound, assuming uniform signal).
For semantic tasks concentrated in NoPE dims: ratio → 1.0.

---

## Quantitative Predictions

| Kill Criterion | Prediction | Confidence |
|---|---|---|
| K997: max ||NoPE(v, pos=100) - NoPE(v, pos=100000)||_inf = 0 | Exactly 0.0 (algebraic) | 100% |
| K998: mean ||RoPE(v, pos=100) - RoPE(v, pos=100000)||_2 > 0 | > 0 (rotation non-trivial) | 100% |
| K999: NoPE_quality / full_quality >= 0.90 | ≥ 0.866 (lower bound), likely ≥ 0.90 | 85% |

---

## Failure Conditions (Kill Analysis)

**K997 can only fail if:**
- inv_freq is NOT zero-padded (bug in p-RoPE implementation)
- RoPE is applied to NoPE dims via a different pathway
- These can be verified by inspecting inv_freq directly

**K999 can fail if:**
- Task signal is concentrated in RoPE dims (dims 0-127)
- This is unlikely for semantic tasks; position-entangled dims carry syntactic info

**Impossibility structure:**
The algebraic proof makes K997 failure IMPOSSIBLE given correct p-RoPE implementation.
K998 and K999 are empirical but follow directly from the math.

---

## Architecture Reference

Gemma 4 global attention layer (from mlx-vlm/models/gemma4/language.py):
```python
# p-RoPE: only first rope_dim dims rotated
q_rot = apply_rope(q[..., :rope_dim], cos, sin)  # RoPE dims
q_pass = q[..., rope_dim:]                        # NoPE dims (unrotated)
q = mx.concatenate([q_rot, q_pass], axis=-1)
```

This confirms: dims [rope_dim:] (= [128:512] for global heads) are NEVER rotated.
