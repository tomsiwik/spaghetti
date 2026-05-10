# MATH.md — KAN-based PoLAR adapter (Lagrangian / scalar-function composition)

## Hypothesis

Standard PoLAR adapters represent ΔW = scale · A · B as **vector-matrix
multiplication** (cross-dimensional mixing via matmul). The merge problem
we've been studying (Pico/ACE/TIES/OrthoMerge/Fisher-Rao) is fundamentally
about how to combine `B_k` matrices without destructive interference in
the shared B-direction subspace.

**Kolmogorov-Arnold Networks** (Liu et al. 2024, arxiv 2404.19756) replace
matmul-and-activation with **scalar univariate B-spline functions per edge**.
Each edge `(i, j)` is a learnable function `ϕ_{ij}: ℝ → ℝ`. The output at
a node is `Σ_i ϕ_{ij}(x_i)`. This is the "scalar-function" reformulation
the user asked about — analogous to Lagrangian mechanics replacing vector
forces with scalar potentials.

> **Q1**: Can a KAN adapter match standard PoLAR on a single task?
>
> **Q2**: Does composition of two KAN adapters reduce to per-edge spline
> coefficient addition (no interference math needed)?
>
> **Q3**: Does a KAN adapter compose with existing standard PoLAR
> adapters (mixed-architecture composition)?

If Q1+Q2 hold, the entire Pico/ACE/TIES/OrthoMerge research arc collapses:
**composition is just adding spline coefficients per edge.**

## Method

### KAN adapter forward (drop-in replacement for `B_t @ x`)

```
Standard PoLAR:  out = scale · (x @ A) @ B_t            # matmul-based
KAN PoLAR:       z = x @ A                              # rank-r projection (kept)
                 out = scale · KAN_block(z)             # spline-based mixing
                 KAN_block(z)[j] = Σ_{i=1..r} ϕ_{ij}(z[i])
                 ϕ_{ij}(u) = Σ_{k} c_{ijk} · B_spline_k(u, grid, degree)
```

We keep the shared Grassmannian `A` (preserves Pierre's storage / theorem)
and only replace `B` with a KAN block. Memory layout per layer:
- Standard: `r × d_out = 6 × 2048 = 12,288` floats per adapter (B-matrix)
- KAN: `r × d_out × grid_size = 6 × 2048 × 5 = 61,440` floats per adapter
  (spline coefficients) — but quantizable to int4 (~30KB/layer)

### Training

For Q1: train ONE KAN adapter for math domain on a small training set
(N=200 examples, 100 steps). Compare single-adapter accuracy on GSM8K.

For Q2: also train a code-domain KAN adapter. Test composition by
**direct spline coefficient addition** (no Pico/ACE/etc. needed):
`c_merged[i,j,k] = (1/2)(c_math[i,j,k] + c_code[i,j,k])`.

For Q3: load standard PoLAR `strategy_full` adapter alongside the KAN
math adapter. Combine via **hybrid composition**: standard PoLAR's B
contributes via matmul, KAN contributes via spline. Output adds.

### Eval

Same 3-bench rig (GSM8K, HumanEval, MedQA) at N=50, seed=42.

## Pre-registered Kill Criteria

- **K1 (EXPRESSIVITY)** KAN-math single-adapter GSM8K ≥ standard PoLAR-math single-adapter GSM8K - 5pp.
  PASS = KAN has enough capacity at rank=6 to match matmul.

- **K2 (PURE-KAN COMPOSITION)** KAN-math + KAN-code composed via spline coefficient addition: avg ≥ best-single avg + 2pp.
  PASS = composition is structurally trivial; arc collapses.

- **K3 (HYBRID COMPOSITION)** KAN-math + standard PoLAR strategy_full: avg ≥ standard PoLAR K=2 avg from `exp_pierre_compose_k2_strategy_x_domain` + 1pp.
  PASS = KAN adapters drop into existing Pierre stack without breaking composition.

- **K4 (BUDGET)** Single-adapter training completes in ≤30 minutes on M5 Pro.
  PASS = KAN training is tractable for adapter ecosystem.

## Verdict logic

| K1 | K2 | K3 | Outcome |
|----|----|----|---------|
| ✓ | ✓ | ✓ | **SUPPORTED** — KAN adapters are the answer. Composition arc closed. Plan migration. |
| ✓ | ✓ | ✗ | **SUPPORTED w/ caveat** — pure-KAN works but doesn't mix with legacy PoLAR. New ecosystem. |
| ✓ | ✗ | * | **PARTIAL** — KAN expressive enough but composition still hard; KAN is just an alternative parameterization. |
| ✗ | * | * | **KILLED** — KAN at rank=6 doesn't match standard PoLAR. Composition arc remains open. |

## Why this matters beyond accuracy

If KAN adapters work, secondary benefits:

1. **Composition becomes trivial** — per-edge spline coefficient addition. No Pico/ACE/etc. needed.
2. **Interpretability** — each edge `ϕ_{ij}` is a plottable scalar function. Debug composition failures by inspection.
3. **Memory bandwidth** — int4 quantization of spline coefficients ≈ 0.5× standard fp16 PoLAR. Better fit for M5 Pro's bandwidth-bound regime.
4. **Catastrophic interference resistance** — splines are local in input space; new training data can't globally wipe old knowledge.
5. **Adapter ecosystem** — predictable composition algebra means adapters become Lego blocks.

Honest tradeoffs:
- Training 2-4× slower per step
- Per-FLOP slower (matters less on Apple Silicon than on H100)
- Empirically novel (no published KAN-LoRA at LLM scale)

## Implementation

KAN block as MLX module (~150 LoC):

```python
class KANBlock(nn.Module):
    def __init__(self, rank: int, d_out: int, grid_size: int = 5, k: int = 3):
        # Spline coefficients per edge: (rank, d_out, grid_size)
        self.coefficients = mx.random.normal(...) * 0.01
        self.grid = mx.linspace(-1, 1, grid_size)
        self.k = k  # spline degree

    def __call__(self, x):  # x: (batch, seq, rank)
        # Evaluate B-spline basis at x, then weighted sum per edge
        # B-spline basis evaluation via Cox-de Boor recursion
        # Output: (batch, seq, d_out)
        ...
```

Training: standard MLX gradient descent on the spline coefficients. The
shared A is frozen (Pierre's invariant). Only the spline coefficients
update.

Connect to existing eval rig via the same `_pierre_shared/eval_runner`
infrastructure but with a `kind="kan"` MethodSpec or a custom path.

## Honest gaps

- **No published KAN-LLM-adapter precedent.** This is research frontier.
  Existing KAN papers focus on PDE solving and small CV tasks. We may
  hit issues nobody has documented.
- **Spline initialization matters.** If we initialize randomly, training
  may not converge in 100 steps. If we initialize to approximate the
  existing PoLAR math adapter's matrix (fit splines to it), we get a
  warm start but introduce a different bias.
- **Grid size and degree are knobs.** grid_size=5, k=3 (cubic) is the
  KAN paper default but may be wrong for adapter math.
- **Q3 hybrid composition is untested anywhere.** Paper doesn't address
  mixing matmul-based and spline-based adapters in the same model.

## References

- KAN paper (arxiv 2404.19756): https://arxiv.org/abs/2404.19756
- KAN repo: https://github.com/KindXiaoming/pykan
- Lagrangian framing: per-token scalar potential (this experiment's intuition)
- Sibling: `exp_pierre_dare_b_vs_fisher_rao` (matmul-based composition baseline)
- Future work: scale to all 42 layers, multi-adapter Lego ecosystem
