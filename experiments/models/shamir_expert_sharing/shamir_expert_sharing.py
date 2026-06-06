"""Shamir Secret Sharing for fault-tolerant expert composition.

Encodes expert MLP weights as polynomial evaluations. Any k of n shares
reconstruct the original weights via Lagrange interpolation over the reals.

Key insight: Shamir's scheme over real numbers (not finite fields) introduces
only floating-point rounding errors, which are negligible for neural network
weights. The polynomial structure also enables "expert blending" -- evaluating
at non-share points produces smooth interpolations between experts.
"""

import time
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .. import register
from ..gpt import GPT


# --------------------------------------------------------------------------- #
#  Shamir primitives (over the reals, using numpy float64 for precision)       #
# --------------------------------------------------------------------------- #

def create_polynomial(secret: np.ndarray, degree: int, rng: np.random.Generator) -> np.ndarray:
    """Create a random polynomial with given secret as constant term.

    Args:
        secret: array of shape (...,) -- each element is a "secret" to share
        degree: polynomial degree (= k - 1 where k is reconstruction threshold)
        rng: numpy random generator

    Returns:
        coefficients: array of shape (degree + 1, ...) where coeffs[0] = secret
    """
    shape = secret.shape
    coeffs = np.zeros((degree + 1,) + shape, dtype=np.float64)
    coeffs[0] = secret.astype(np.float64)
    # Random higher-order coefficients
    for i in range(1, degree + 1):
        coeffs[i] = rng.standard_normal(shape) * 0.1  # scale to avoid huge values
    return coeffs


def evaluate_polynomial(coeffs: np.ndarray, x: float) -> np.ndarray:
    """Evaluate polynomial at point x using Horner's method.

    Args:
        coeffs: shape (degree + 1, ...) -- coefficients from low to high order
        x: evaluation point

    Returns:
        result: shape (...,) -- polynomial value at x
    """
    # Horner: start from highest degree coefficient
    result = coeffs[-1].copy()
    for i in range(len(coeffs) - 2, -1, -1):
        result = result * x + coeffs[i]
    return result


def create_shares(secret: np.ndarray, k: int, n: int,
                  seed: int = 42) -> list[tuple[float, np.ndarray]]:
    """Create n Shamir shares with threshold k.

    Args:
        secret: weight tensor to share (any shape)
        k: minimum shares needed for reconstruction
        n: total number of shares
        seed: random seed for reproducibility

    Returns:
        shares: list of (x_i, y_i) pairs where y_i = P(x_i)
    """
    assert n >= k >= 1, f"Need n >= k >= 1, got n={n}, k={k}"
    rng = np.random.default_rng(seed)
    flat = secret.flatten()

    # Polynomial of degree k-1 (constant term = secret)
    coeffs = create_polynomial(flat, degree=k - 1, rng=rng)

    # Evaluate at n distinct non-zero points (x = 1, 2, ..., n)
    shares = []
    for i in range(1, n + 1):
        x_i = float(i)
        y_i = evaluate_polynomial(coeffs, x_i)
        shares.append((x_i, y_i))

    return shares


def lagrange_interpolate_at_zero(points: list[tuple[float, np.ndarray]]) -> np.ndarray:
    """Reconstruct secret (polynomial value at x=0) from k shares.

    Uses Lagrange interpolation evaluated at x=0:
        L_j(0) = prod_{m!=j} (0 - x_m) / (x_j - x_m)
               = prod_{m!=j} (-x_m) / (x_j - x_m)

    Args:
        points: list of (x_i, y_i) tuples (at least k points)

    Returns:
        secret: the reconstructed secret (polynomial constant term)
    """
    k = len(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    result = np.zeros_like(ys[0])
    for j in range(k):
        # Compute Lagrange basis polynomial L_j evaluated at 0
        basis = 1.0
        for m in range(k):
            if m != j:
                basis *= (-xs[m]) / (xs[j] - xs[m])
        result += basis * ys[j]

    return result


def reconstruct_from_shares(shares: list[tuple[float, np.ndarray]],
                            k: int, original_shape: tuple) -> np.ndarray:
    """Reconstruct original weight tensor from k shares.

    Args:
        shares: list of (x_i, y_i) share pairs (need at least k)
        k: threshold (use exactly k shares)
        original_shape: shape of the original weight tensor

    Returns:
        reconstructed: weight tensor of original_shape
    """
    assert len(shares) >= k, f"Need >= {k} shares, got {len(shares)}"
    # Use first k shares
    selected = shares[:k]
    flat = lagrange_interpolate_at_zero(selected)
    return flat.reshape(original_shape)


def evaluate_at_point(shares: list[tuple[float, np.ndarray]],
                      k: int, target_x: float,
                      original_shape: tuple) -> np.ndarray:
    """Evaluate the polynomial at an arbitrary point (expert blending).

    This is the "novel angle": evaluating at non-integer points between share
    positions gives smooth interpolation in weight space.

    Args:
        shares: list of share pairs
        k: threshold
        target_x: point to evaluate at
        original_shape: original weight shape

    Returns:
        blended: weight tensor at the interpolation point
    """
    assert len(shares) >= k
    selected = shares[:k]
    xs = [p[0] for p in selected]
    ys = [p[1] for p in selected]

    result = np.zeros_like(ys[0])
    for j in range(k):
        basis = 1.0
        for m in range(k):
            if m != j:
                basis *= (target_x - xs[m]) / (xs[j] - xs[m])
        result += basis * ys[j]

    return result.reshape(original_shape)


# --------------------------------------------------------------------------- #
#  ShamirExpertGPT -- GPT with Shamir-shared MLP weights                      #
# --------------------------------------------------------------------------- #

@register("shamir_expert", parent="gpt")
class ShamirExpertGPT(GPT):
    """GPT model with Shamir secret sharing applied to MLP weights.

    After training, MLP weights can be "shared" into n pieces such that
    any k pieces suffice for exact reconstruction. This tests whether
    the sharing/reconstruction process preserves model quality and
    measures the computational overhead.
    """

    def __init__(self, n_shares: int = 5, k_threshold: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.n_shares = n_shares
        self.k_threshold = k_threshold
        self._shares_cache: Optional[dict] = None  # layer_idx -> {param_name -> shares}
        self._original_weights: Optional[dict] = None

    def create_shares(self, seed: int = 42) -> dict:
        """Create Shamir shares of all MLP weights.

        Returns:
            shares_dict: {layer_idx: {param_name: [(x_i, y_i), ...]}}
        """
        shares_dict = {}
        for layer_idx, layer in enumerate(self.layers):
            layer_shares = {}
            for name in ["fc1.weight", "fc2.weight"]:
                parts = name.split(".")
                param = getattr(layer.mlp, parts[0])
                w = getattr(param, parts[1])
                w_np = np.array(w.tolist(), dtype=np.float64)
                layer_shares[name] = create_shares(
                    w_np, self.k_threshold, self.n_shares, seed=seed + layer_idx
                )
            shares_dict[layer_idx] = layer_shares

        self._shares_cache = shares_dict
        return shares_dict

    def save_original_weights(self):
        """Save original MLP weights for comparison."""
        self._original_weights = {}
        for layer_idx, layer in enumerate(self.layers):
            self._original_weights[layer_idx] = {
                "fc1.weight": np.array(layer.mlp.fc1.weight.tolist(), dtype=np.float64),
                "fc2.weight": np.array(layer.mlp.fc2.weight.tolist(), dtype=np.float64),
            }

    def reconstruct_from_shares(self, shares_dict: dict,
                                 share_indices: Optional[list[int]] = None):
        """Reconstruct MLP weights from shares and load them.

        Args:
            shares_dict: output of create_shares()
            share_indices: which shares to use (0-indexed). If None, use first k.
        """
        for layer_idx, layer in enumerate(self.layers):
            layer_shares = shares_dict[layer_idx]
            for name in ["fc1.weight", "fc2.weight"]:
                all_shares = layer_shares[name]
                if share_indices is not None:
                    selected = [all_shares[i] for i in share_indices]
                else:
                    selected = all_shares[:self.k_threshold]

                parts = name.split(".")
                param = getattr(layer.mlp, parts[0])
                orig_shape = tuple(getattr(param, parts[1]).shape)
                reconstructed = reconstruct_from_shares(
                    selected, self.k_threshold, orig_shape
                )
                setattr(param, parts[1], mx.array(reconstructed.astype(np.float32)))

    def blend_at_point(self, shares_dict: dict, target_x: float):
        """Load weights from polynomial evaluation at arbitrary point.

        This enables "expert blending" -- intermediate points between shares
        give smooth weight-space interpolations.
        """
        for layer_idx, layer in enumerate(self.layers):
            layer_shares = shares_dict[layer_idx]
            for name in ["fc1.weight", "fc2.weight"]:
                all_shares = layer_shares[name]
                selected = all_shares[:self.k_threshold]
                parts = name.split(".")
                param = getattr(layer.mlp, parts[0])
                orig_shape = tuple(getattr(param, parts[1]).shape)
                blended = evaluate_at_point(
                    selected, self.k_threshold, target_x, orig_shape
                )
                setattr(param, parts[1], mx.array(blended.astype(np.float32)))
