# MATH.md — ASEntmax dropin: α-entmax replaces softmax in attention

## Hypothesis

Per arxiv 2506.16640 ("Adaptively Sparse Transformers"), replacing softmax
with **α-entmax** (α=1.5 default, learnable per-head temperature) produces
sparse attention distributions that improve long-context retrieval without
modifying the model's architecture beyond the activation function.

For Pierre's adapter composition, the key question is whether sparse
attention preserves Δ-merge / DARE / Pico arithmetic on PoLAR adapters —
the adapter math operates on q_proj independent of the softmax, so it
should be **orthogonal** to attention sparsity, but composition behavior
under sparser attention is untested.

> **Does replacing softmax with α-entmax(α=1.5) in Gemma 4 attention
> preserve PoLAR adapter Δ-arithmetic, while improving retrieval at
> ≥32K context?**

## α-entmax mechanism

```
α-entmax(z) = argmax_p {⟨p, z⟩ - H_α(p)}  s.t. p simplex
           = [(α-1)·z - τ]_+^{1/(α-1)}
```

For α=1.5 specifically, the closed-form via sort:
```
sort z descending: z_(1) ≥ z_(2) ≥ ...
ρ(k) = (Σ_{j≤k} z_(j)·(α-1) - 1) / k
find largest k such that z_(k)·(α-1) > ρ(k)
τ = ρ(k)
p_i = max(0, (α-1)·z_i - τ)^{1/(α-1)}
```

α=1 → softmax (recover dense). α=2 → sparsemax (max-sparse).
α=1.5 → middle ground recommended by paper.

## Pre-registered Kill Criteria

- **K1 (ADAPTER ARITHMETIC)** PoLAR adapter composition (Fisher-Rao K=7)
  with α-entmax attention shows behavioral score within ±0.5pp of softmax baseline (composition unbroken).
- **K2 (RETRIEVAL)** RULER@32K with α-entmax ≥ softmax baseline + 1pp (sparse attention helps long context).
- **K3 (DECODE LATENCY)** Decode tok/s drops < 5% (α-entmax is not free; must be cheap).
- **K4 (SPARSITY)** ≥50% of attention heads converge to non-trivial sparsity (α near 1 = degeneracy).

## Verdict logic

| K1 | K2 | Outcome |
|----|----|---------|
| ✓ | ✓ | **SUPPORTED** — α-entmax preserves composition AND helps retrieval. |
| ✓ | ✗ | **SUPPORTED with caveat** — composition fine, no retrieval gain at 32K. |
| ✗ | * | **KILLED** — sparse attention disrupts PoLAR Δ-arithmetic. Implication: composition methods (Pico/ACE/etc.) all assume dense attention. |

## Implementation status

**SPEC ONLY — implementation pending.**

Required engineering:
1. MLX α-entmax kernel (~80 LOC). Closed-form sort-based for α=1.5.
2. Patch Gemma 4 attention module's softmax call (~30 LOC). Use `setattr` per Finding #831 canonical pattern.
3. Self-test: at α→1, must reproduce softmax within float-precision.
4. Eval rig extension: add RULER@32K for K2 (current eval is N=50 short prompts; K2 needs long-context).

run_experiment.py currently exits with INCONCLUSIVE + "implementation pending"
verdict to preserve queue integrity. KCs are pre-registered; cannot be
changed post-implementation.

## References

- ASEntmax paper (arxiv 2506.16640): https://arxiv.org/abs/2506.16640
- α-entmax algorithm: arxiv 1909.00015 (closed-form sort-based)
- Pierre's existing attention: `mlx_lm/models/gemma4.py` (sliding window 4096 in 5/6 layers)
- Research agent SSA survey: `a2c10d5138d8eea52`
