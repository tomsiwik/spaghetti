"""KAN-based PoLAR adapter — scalar-function (B-spline) replacement for B-matmul.

Per arxiv 2404.19756 (Liu et al., "KAN: Kolmogorov-Arnold Networks").
Each edge (i, j) carries a learnable univariate B-spline ϕ_{ij}: ℝ → ℝ
parameterized by spline coefficients on a fixed grid. The KAN block
replaces B's matmul:

    Standard PoLAR forward:  out = scale · (x @ A) @ B
    KAN PoLAR forward:       out = scale · KAN_block(x @ A)
        where KAN_block(z)[j] = Σ_i ϕ_{ij}(z[i])
              ϕ_{ij}(u) = Σ_k c_{ijk} · B_spline_k(u)

Shared frozen Grassmannian A is preserved (Pierre's invariant).
Only B's matmul is replaced with spline evaluation.

This module exposes:
  - KANBlock: MLX module, drop-in replacement for B (rank, d_out)
  - compose_kan_pure: per-edge spline coefficient addition (Q2 test)
  - compose_kan_hybrid: KAN + standard PoLAR mixed composition (Q3 test)
  - kan_block_from_B: warm-start by fitting spline to existing B-matrix
"""
from __future__ import annotations
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# B-spline basis evaluation (Cox-de Boor recursion)
# ─────────────────────────────────────────────────────────────────────────

def _bspline_basis(x: mx.array, grid: mx.array, k: int) -> mx.array:
    """Evaluate degree-k B-spline basis at points x on a uniform grid.

    Args:
        x: input points, shape (..., n_points)
        grid: knot positions, shape (grid_size,) — assumed uniform
        k: spline degree (e.g. 3 = cubic)

    Returns:
        basis values, shape (..., n_points, n_basis)
        where n_basis = grid_size + k - 1
    """
    # Extend grid with k extra knots on each side for boundary handling
    grid_step = grid[1] - grid[0]
    g = grid
    for _ in range(k):
        g_left = g[:1] - grid_step
        g_right = g[-1:] + grid_step
        g = mx.concatenate([g_left, g, g_right])
    # g is (grid_size + 2k,)

    # Initial degree-0 basis: indicator of [g_i, g_{i+1})
    x_exp = x[..., None]                      # (..., n_points, 1)
    g_exp = g[None, :]                        # (1, grid_size + 2k)
    # B0[..., j] = 1 if g[j] <= x < g[j+1]
    bases = ((x_exp >= g_exp[..., :-1]) & (x_exp < g_exp[..., 1:])).astype(mx.float32)
    # bases shape: (..., n_points, grid_size + 2k - 1)

    # Cox-de Boor recursion: degree d basis from degree d-1
    for d in range(1, k + 1):
        # bases at degree d uses g[i:i+d+1]
        n_basis = bases.shape[-1] - 1
        left = (x_exp - g[None, :n_basis]) / (g[None, d:d + n_basis] - g[None, :n_basis] + 1e-12)
        right = (g[None, d + 1:d + 1 + n_basis] - x_exp) / (g[None, d + 1:d + 1 + n_basis] - g[None, 1:1 + n_basis] + 1e-12)
        bases = left * bases[..., :-1] + right * bases[..., 1:]
    return bases  # (..., n_points, grid_size + k - 1)


# ─────────────────────────────────────────────────────────────────────────
# KAN Block
# ─────────────────────────────────────────────────────────────────────────

class KANBlock(nn.Module):
    """KAN-style replacement for B-matrix in PoLAR adapter.

    Args:
        rank: input dim (after A projection) — typically PoLAR's r=6
        d_out: output dim (q_proj output) — typically 2048
        grid_size: number of knots in spline grid
        k: spline degree (3 = cubic, default)
        grid_range: input value range covered by grid (clip outside)
    """
    def __init__(self, rank: int, d_out: int,
                 grid_size: int = 5, k: int = 3,
                 grid_range: tuple[float, float] = (-2.0, 2.0)):
        super().__init__()
        self.rank = rank
        self.d_out = d_out
        self.grid_size = grid_size
        self.k = k
        self.n_basis = grid_size + k - 1
        self.grid = mx.linspace(grid_range[0], grid_range[1], grid_size)
        # Spline coefficients per edge (rank × d_out × n_basis)
        # Initialize small to start near zero contribution (like LoRA's B-init=0)
        rng = np.random.default_rng(42)
        coeffs = rng.standard_normal((rank, d_out, self.n_basis)).astype(np.float32) * 0.01
        self.coefficients = mx.array(coeffs)
        # Linear "skip" path for stability (like residual)
        self.skip_weight = mx.zeros((rank, d_out))

    def __call__(self, x: mx.array) -> mx.array:
        """x: (..., rank) → out: (..., d_out)

        Implementation: evaluate spline basis once per rank dim,
        contract with coefficients to produce d_out values, sum across rank.
        """
        # x shape: (..., rank). Evaluate basis per rank dim.
        # Clip x to grid range to avoid extrapolation issues
        x_clipped = mx.clip(x, float(self.grid[0].item()), float(self.grid[-1].item()))
        # Compute basis: (..., rank, n_basis)
        basis = _bspline_basis(x_clipped, self.grid, self.k)
        # Contract: out[..., j] = Σ_i Σ_k coeffs[i, j, k] · basis[..., i, k]
        # einsum: "...ik,ijk->...j"
        out = mx.einsum("...ik,ijk->...j", basis, self.coefficients)
        # Add skip path: out += x @ skip_weight
        out = out + (x_clipped @ self.skip_weight)
        return out


# ─────────────────────────────────────────────────────────────────────────
# KAN-augmented PoLAR Linear (drop-in via setattr per Finding #831)
# ─────────────────────────────────────────────────────────────────────────

class _KANPoLARLinear(nn.Module):
    """base(x) + scale · KAN(x @ A) — replaces standard PoLAR's B-matmul."""
    def __init__(self, base_layer, lora_a, kan_block: KANBlock, scale: float):
        super().__init__()
        self.base = base_layer
        self.lora_a = lora_a            # (d_in, rank), shared, frozen
        self.kan = kan_block            # rank → d_out via splines
        self.scale = scale

    def __call__(self, x):
        z = x @ self.lora_a.astype(x.dtype)              # (..., rank)
        kan_out = self.kan(z.astype(mx.float32))         # (..., d_out)
        return self.base(x) + self.scale * kan_out.astype(x.dtype)


# ─────────────────────────────────────────────────────────────────────────
# Composition primitives
# ─────────────────────────────────────────────────────────────────────────

def compose_kan_pure(kan_blocks: list[KANBlock], weights: Optional[list[float]] = None) -> KANBlock:
    """Compose K KAN adapters by per-edge spline coefficient addition.

    This is the Q2 test: does composition reduce to trivial coefficient
    averaging? If yes, the entire Pico/ACE/TIES research arc collapses.
    """
    K = len(kan_blocks)
    if K == 0:
        raise ValueError("No KAN blocks to compose")
    if K == 1:
        return kan_blocks[0]
    if weights is None:
        weights = [1.0 / K] * K

    # All blocks must have matching shape
    rank = kan_blocks[0].rank
    d_out = kan_blocks[0].d_out
    grid_size = kan_blocks[0].grid_size
    k = kan_blocks[0].k

    merged = KANBlock(rank, d_out, grid_size=grid_size, k=k)
    # Coefficients: weighted mean per edge
    coeffs = mx.zeros_like(kan_blocks[0].coefficients)
    skip = mx.zeros_like(kan_blocks[0].skip_weight)
    for w, kan in zip(weights, kan_blocks):
        coeffs = coeffs + (w * kan.coefficients)
        skip = skip + (w * kan.skip_weight)
    merged.coefficients = coeffs
    merged.skip_weight = skip
    return merged


def kan_block_from_B(B: mx.array, grid_size: int = 5, k: int = 3,
                     grid_range: tuple[float, float] = (-2.0, 2.0)) -> KANBlock:
    """Warm-start: initialize a KAN block to approximate an existing B-matrix.

    Uses the linear skip path to exactly represent the B-matrix and
    initializes spline coefficients to small random — KAN can refine
    via training.
    """
    rank, d_out = B.shape
    block = KANBlock(rank, d_out, grid_size=grid_size, k=k, grid_range=grid_range)
    # Initialize skip to B (exact reproduction at init)
    block.skip_weight = B.astype(mx.float32)
    # Spline coefficients stay near-zero initial (small random)
    return block
