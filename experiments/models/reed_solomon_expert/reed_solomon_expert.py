"""Reed-Solomon error correction for expert library resilience.

Encodes N expert MLP weight matrices as N data symbols in a polynomial code
over the reals. Generates k parity experts via Lagrange interpolation at
additional evaluation points. Any N of the N+k total experts reconstruct
all N originals.

This is the "error correction" dual of the Shamir secret-sharing experiment:
- Shamir: k-of-n threshold access to ONE secret
- Reed-Solomon: N-of-(N+k) reconstruction of ALL N experts

Mathematical foundation:
  Given N experts with flattened weight vectors w_1, ..., w_N at evaluation
  points x_1, ..., x_N, the unique degree-(N-1) Lagrange interpolating
  polynomial P(x) satisfies P(x_i) = w_i.
  Parity experts: w_{N+j} = P(x_{N+j}) for j = 1..k.
  Reconstruction: any N of the N+k points determine P, recovering all w_i.

Key design choice: evaluation points use Chebyshev-spaced nodes to minimize
the Runge phenomenon (Lebesgue constant growth) for larger N.
"""

import time
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .. import register
from ..gpt import GPT


# --------------------------------------------------------------------------- #
#  RS encoding/decoding primitives (over the reals, via Lagrange)             #
# --------------------------------------------------------------------------- #

def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """Generate n Chebyshev nodes on [a, b] for numerically stable interpolation.

    Chebyshev nodes minimize the Lebesgue constant, reducing oscillation
    (Runge phenomenon) in high-degree polynomial interpolation.

    Args:
        n: number of nodes
        a, b: interval endpoints

    Returns:
        nodes: array of shape (n,) with Chebyshev-spaced points
    """
    k = np.arange(n, dtype=np.float64)
    # Chebyshev nodes of the first kind on [-1, 1]
    nodes_unit = np.cos((2 * k + 1) * np.pi / (2 * n))
    # Map to [a, b]
    return 0.5 * (a + b) + 0.5 * (b - a) * nodes_unit


def lagrange_interpolate(data_xs: np.ndarray, data_ys: np.ndarray,
                         target_x: float) -> np.ndarray:
    """Evaluate the Lagrange interpolating polynomial at target_x.

    Args:
        data_xs: shape (N,) -- evaluation points of known data
        data_ys: shape (N, D) -- known values (each row is a flattened weight vector)
        target_x: point to evaluate at

    Returns:
        result: shape (D,) -- P(target_x)
    """
    N = len(data_xs)
    D = data_ys.shape[1]
    result = np.zeros(D, dtype=np.float64)

    for j in range(N):
        # Lagrange basis L_j(target_x) = prod_{m!=j} (target_x - x_m) / (x_j - x_m)
        basis = 1.0
        for m in range(N):
            if m != j:
                basis *= (target_x - data_xs[m]) / (data_xs[j] - data_xs[m])
        result += basis * data_ys[j]

    return result


def rs_encode(expert_weights: list[np.ndarray], k_parity: int,
              use_chebyshev: bool = True) -> dict:
    """Reed-Solomon encode N expert weight vectors, producing k parity experts.

    Args:
        expert_weights: list of N flattened weight arrays, each shape (D,)
        k_parity: number of parity experts to generate

    Returns:
        dict with:
            data_xs: evaluation points for original experts (N,)
            parity_xs: evaluation points for parity experts (k,)
            parity_weights: list of k parity weight arrays, each (D,)
            all_xs: concatenated data + parity xs (N+k,)
            all_weights: concatenated data + parity weights (N+k, D)
    """
    N = len(expert_weights)
    D = expert_weights[0].shape[0]

    # Stack into (N, D)
    data_ys = np.stack(expert_weights, axis=0).astype(np.float64)

    if use_chebyshev:
        # Use Chebyshev nodes for data points on [-1, 1]
        data_xs = chebyshev_nodes(N, -1.0, 1.0)
        # Parity points: extend beyond the data interval
        parity_xs = chebyshev_nodes(k_parity, 1.1, 2.0)
    else:
        # Simple integer spacing
        data_xs = np.arange(1, N + 1, dtype=np.float64)
        parity_xs = np.arange(N + 1, N + k_parity + 1, dtype=np.float64)

    # Generate parity experts
    parity_weights = []
    for px in parity_xs:
        pw = lagrange_interpolate(data_xs, data_ys, px)
        parity_weights.append(pw)

    all_xs = np.concatenate([data_xs, parity_xs])
    all_ys = np.concatenate([data_ys, np.stack(parity_weights, axis=0)], axis=0)

    return {
        "data_xs": data_xs,
        "parity_xs": parity_xs,
        "parity_weights": parity_weights,
        "all_xs": all_xs,
        "all_weights": all_ys,
    }


def rs_decode(available_xs: np.ndarray, available_ys: np.ndarray,
              original_xs: np.ndarray) -> np.ndarray:
    """Reconstruct all N original experts from any N available experts.

    Args:
        available_xs: shape (N,) -- evaluation points of available experts
        available_ys: shape (N, D) -- weight vectors of available experts
        original_xs: shape (N,) -- evaluation points of the original experts

    Returns:
        reconstructed: shape (N, D) -- reconstructed original expert weights
    """
    N = len(original_xs)
    assert len(available_xs) >= N, (
        f"Need >= {N} available experts, got {len(available_xs)}"
    )

    # Use exactly N available experts
    used_xs = available_xs[:N]
    used_ys = available_ys[:N]

    reconstructed = []
    for ox in original_xs:
        rw = lagrange_interpolate(used_xs, used_ys, ox)
        reconstructed.append(rw)

    return np.stack(reconstructed, axis=0)


def reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Compute reconstruction error metrics.

    Args:
        original: shape (N, D) -- original weight vectors
        reconstructed: shape (N, D) -- reconstructed weight vectors

    Returns:
        dict with max_abs_error, mean_abs_error, relative_error
    """
    diff = np.abs(original - reconstructed)
    orig_norm = np.linalg.norm(original)
    return {
        "max_abs_error": float(np.max(diff)),
        "mean_abs_error": float(np.mean(diff)),
        "relative_error": float(np.linalg.norm(original - reconstructed) / (orig_norm + 1e-15)),
    }


# --------------------------------------------------------------------------- #
#  ReedSolomonExpertGPT -- GPT with RS-encoded MLP weight redundancy          #
# --------------------------------------------------------------------------- #

@register("reed_solomon_expert", parent="gpt")
class ReedSolomonExpertGPT(GPT):
    """GPT model with Reed-Solomon encoding of MLP expert weights.

    After training, MLP weights in each layer can be RS-encoded into N+k
    total experts (N original + k parity). Any N of N+k reconstruct all
    originals via Lagrange interpolation over the reals.

    This validates:
    1. Quality preservation after encode-drop-reconstruct cycle
    2. Parameter overhead is exactly k/N * MLP_params
    3. Parity experts may serve as useful "interpolation experts"
    """

    def __init__(self, k_parity: int = 2, use_chebyshev: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.k_parity = k_parity
        self.use_chebyshev = use_chebyshev
        self._encoding_cache: Optional[dict] = None  # layer_idx -> rs_encode output
        self._original_weights: Optional[dict] = None

    def save_original_weights(self):
        """Save original MLP weights before any reconstruction."""
        self._original_weights = {}
        for layer_idx, layer in enumerate(self.layers):
            self._original_weights[layer_idx] = {
                "fc1.weight": np.array(layer.mlp.fc1.weight.tolist(), dtype=np.float64),
                "fc2.weight": np.array(layer.mlp.fc2.weight.tolist(), dtype=np.float64),
            }

    def rs_encode_experts(self) -> dict:
        """RS-encode each layer's MLP weights.

        Treats fc1 and fc2 weight matrices as two independent "expert symbols"
        per layer. Each gets N=1 data point + k parity points (since each layer
        has one MLP, N=1 per layer). For multi-expert scenarios, N would be
        the number of capsule groups.

        For this experiment, we treat the 4 layers' fc1 weights as 4 "experts"
        that form one RS codeword, and similarly for fc2. This tests whether
        RS encoding across layers preserves quality.

        Returns:
            encoding: {param_name: rs_encode output}
        """
        encoding = {}
        for param_name in ["fc1.weight", "fc2.weight"]:
            expert_weights = []
            for layer_idx, layer in enumerate(self.layers):
                parts = param_name.split(".")
                param = getattr(layer.mlp, parts[0])
                w = getattr(param, parts[1])
                w_np = np.array(w.tolist(), dtype=np.float64)
                expert_weights.append(w_np.flatten())

            encoding[param_name] = rs_encode(
                expert_weights, self.k_parity, self.use_chebyshev
            )

        self._encoding_cache = encoding
        return encoding

    def reconstruct_from_available(self, encoding: dict,
                                    drop_indices: list[int]):
        """Reconstruct MLP weights after dropping some layer-experts.

        Args:
            encoding: output of rs_encode_experts()
            drop_indices: which original expert indices (layer indices) to drop
                         and reconstruct from parity + remaining originals
        """
        N = len(self.layers)

        for param_name in ["fc1.weight", "fc2.weight"]:
            enc = encoding[param_name]
            all_xs = enc["all_xs"]
            all_ys = enc["all_weights"]
            data_xs = enc["data_xs"]

            # Build available set: all except dropped
            keep_mask = np.ones(len(all_xs), dtype=bool)
            for di in drop_indices:
                keep_mask[di] = False

            available_xs = all_xs[keep_mask]
            available_ys = all_ys[keep_mask]

            # Reconstruct all originals
            reconstructed = rs_decode(available_xs, available_ys, data_xs)

            # Load reconstructed weights back
            for layer_idx, layer in enumerate(self.layers):
                parts = param_name.split(".")
                param = getattr(layer.mlp, parts[0])
                orig_shape = tuple(getattr(param, parts[1]).shape)
                w_recon = reconstructed[layer_idx].reshape(orig_shape)
                setattr(param, parts[1],
                        mx.array(w_recon.astype(np.float32)))

    def load_parity_expert(self, encoding: dict, parity_idx: int,
                           target_layer: int):
        """Load a parity expert's weights into a specific layer.

        This tests whether parity experts produce meaningful outputs --
        they are polynomial interpolations in weight space.
        """
        N = len(self.layers)

        for param_name in ["fc1.weight", "fc2.weight"]:
            enc = encoding[param_name]
            parity_w = enc["parity_weights"][parity_idx]

            parts = param_name.split(".")
            param = getattr(self.layers[target_layer].mlp, parts[0])
            orig_shape = tuple(getattr(param, parts[1]).shape)
            setattr(param, parts[1],
                    mx.array(parity_w.reshape(orig_shape).astype(np.float32)))

    def param_overhead(self) -> float:
        """Calculate parameter overhead as percentage of MLP params.

        Overhead = k_parity / N_layers * 100%
        """
        return 100.0 * self.k_parity / len(self.layers)
