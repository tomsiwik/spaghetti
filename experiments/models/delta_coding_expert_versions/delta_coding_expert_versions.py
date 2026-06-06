"""Delta Coding for Expert Version Management.

Video codec analogy for neural network expert versioning:
  I-frame (keyframe) = full expert weight snapshot
  P-frame (delta)    = weight difference between consecutive versions
  GOP (group)        = max chain length before next keyframe

LoRA deltas dW = (alpha/r) * A @ B are already delta-coded relative to the base.
This module extends delta coding to SEQUENCES of expert updates:
  v1 -> v2 -> v3 -> ...
  Store: v1 (full), d12 = v2 - v1, d23 = v3 - v2
  Reconstruct: v3 = v1 + d12 + d23

Key mechanisms:
  1. ExpertVersionChain: stores keyframes + deltas, reconstructs any version
  2. Delta compression via low-rank approximation (SVD truncation)
  3. Keyframe scheduling (every K versions)
  4. Quality drift measurement across chain length
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..lora_procrustes.lora_procrustes import LoRAGPT, LoRALinear


class ExpertVersionChain:
    """Manages a chain of expert versions using delta coding.

    Stores keyframes (full snapshots) and deltas (inter-version differences).
    Supports reconstruction of any version from its nearest keyframe + deltas.

    Attributes:
        keyframe_interval: K -- insert a keyframe every K versions
        versions: list of full parameter snapshots (for ground truth comparison)
        keyframes: dict[int, params] -- version_idx -> full snapshot
        deltas: dict[int, params] -- version_idx -> delta from previous version
        compressed_deltas: dict[int, params] -- version_idx -> compressed delta
    """

    def __init__(self, keyframe_interval: int = 4):
        self.keyframe_interval = keyframe_interval
        self.versions = []  # ground truth full snapshots
        self.keyframes = {}  # version_idx -> full params
        self.deltas = {}     # version_idx -> raw delta
        self.compressed_deltas = {}  # version_idx -> compressed delta

    def add_version(self, params: dict):
        """Add a new expert version.

        Args:
            params: dict of parameter name -> mx.array (full snapshot)
        """
        idx = len(self.versions)
        # Store ground truth
        self.versions.append({k: mx.array(v) for k, v in params.items()})

        if idx == 0 or idx % self.keyframe_interval == 0:
            # Keyframe: store full snapshot
            self.keyframes[idx] = {k: mx.array(v) for k, v in params.items()}
        else:
            # Delta: store difference from previous version
            prev = self.versions[idx - 1]
            delta = {}
            for k in params:
                delta[k] = params[k] - prev[k]
            self.deltas[idx] = delta

    def compress_deltas(self, rank: int = 4):
        """Compress stored deltas using truncated SVD.

        For each delta matrix, approximate it as U[:, :rank] @ S[:rank] @ Vt[:rank, :].
        Stores reconstructed approximation for quality eval, and tracks compressed
        storage cost (U_r, S_r, Vt_r element counts) separately.

        Args:
            rank: truncated SVD rank for delta compression

        Returns:
            dict with compression statistics
        """
        stats = {}
        self.compressed_storage_params = {}  # version -> param count of compressed form
        for idx, delta in self.deltas.items():
            compressed = {}
            total_original = 0
            total_compressed = 0
            total_reconstruction_error = 0.0
            total_norm = 0.0

            for k, d in delta.items():
                d_np = np.array(d)
                original_size = d_np.size

                if d_np.ndim == 2 and min(d_np.shape) > rank:
                    # SVD compress
                    U, S, Vt = np.linalg.svd(d_np, full_matrices=False)
                    U_r = U[:, :rank]
                    S_r = S[:rank]
                    Vt_r = Vt[:rank, :]
                    reconstructed = U_r @ np.diag(S_r) @ Vt_r
                    compressed[k] = mx.array(reconstructed)

                    compressed_size = U_r.size + S_r.size + Vt_r.size
                    error = np.linalg.norm(d_np - reconstructed)
                    norm = np.linalg.norm(d_np)
                else:
                    # 1D or too small to compress -- store as-is
                    compressed[k] = mx.array(d_np)
                    compressed_size = original_size
                    error = 0.0
                    norm = np.linalg.norm(d_np) if d_np.size > 0 else 0.0

                total_original += original_size
                total_compressed += compressed_size
                total_reconstruction_error += error ** 2
                total_norm += norm ** 2

            self.compressed_deltas[idx] = compressed
            self.compressed_storage_params[idx] = total_compressed
            relative_error = (total_reconstruction_error / max(total_norm, 1e-10)) ** 0.5
            stats[idx] = {
                "original_params": total_original,
                "compressed_params": total_compressed,
                "compression_ratio": total_compressed / max(total_original, 1),
                "relative_error": relative_error,
            }

        return stats

    def reconstruct(self, version_idx: int, use_compressed: bool = False) -> dict:
        """Reconstruct expert parameters at a given version.

        Finds nearest keyframe, then applies chain of deltas.

        Args:
            version_idx: which version to reconstruct
            use_compressed: if True, use SVD-compressed deltas

        Returns:
            dict of parameter name -> mx.array
        """
        if version_idx < 0 or version_idx >= len(self.versions):
            raise IndexError(f"Version {version_idx} not found (have {len(self.versions)})")

        # Find nearest keyframe at or before version_idx
        keyframe_idx = version_idx
        while keyframe_idx not in self.keyframes:
            keyframe_idx -= 1

        # Start from keyframe
        params = {k: mx.array(v) for k, v in self.keyframes[keyframe_idx].items()}

        # Apply deltas
        delta_source = self.compressed_deltas if use_compressed else self.deltas
        for i in range(keyframe_idx + 1, version_idx + 1):
            if i in delta_source:
                for k in params:
                    params[k] = params[k] + delta_source[i][k]

        return params

    def reconstruction_error(self, version_idx: int, use_compressed: bool = False) -> float:
        """Measure reconstruction error vs ground truth.

        Returns relative Frobenius error: ||reconstructed - truth|| / ||truth||
        """
        reconstructed = self.reconstruct(version_idx, use_compressed)
        truth = self.versions[version_idx]

        total_error_sq = 0.0
        total_norm_sq = 0.0
        for k in truth:
            diff = reconstructed[k] - truth[k]
            total_error_sq += mx.sum(diff * diff).item()
            total_norm_sq += mx.sum(truth[k] * truth[k]).item()

        return (total_error_sq / max(total_norm_sq, 1e-10)) ** 0.5

    def storage_cost(self, use_compressed: bool = False) -> dict:
        """Calculate storage cost of delta-coded vs full storage.

        When use_compressed=True, uses the actual compressed param counts
        (U_r, S_r, Vt_r sizes) rather than the reconstructed matrix sizes.

        Returns:
            dict with total_params for delta-coded and full storage
        """
        # Full storage: sum of all version sizes
        full_storage = sum(
            sum(v.size for v in ver.values())
            for ver in self.versions
        )

        # Delta storage: keyframes (full) + deltas
        delta_storage = 0
        for idx, kf in self.keyframes.items():
            delta_storage += sum(v.size for v in kf.values())

        if use_compressed and hasattr(self, 'compressed_storage_params'):
            # Use actual compressed sizes (U_r + S_r + Vt_r element counts)
            for idx in self.compressed_deltas:
                delta_storage += self.compressed_storage_params.get(idx, 0)
        else:
            # Raw deltas (same size as full params)
            for idx, d in self.deltas.items():
                delta_storage += sum(v.size for v in d.values())

        return {
            "full_storage": full_storage,
            "delta_storage": delta_storage,
            "ratio": delta_storage / max(full_storage, 1),
            "savings_pct": (1 - delta_storage / max(full_storage, 1)) * 100,
        }


class DeltaCodedLoRAGPT(LoRAGPT):
    """LoRAGPT with delta-coded version management.

    Inherits the LoRA architecture and adds version chain tracking.
    This is a thin wrapper -- the core mechanism is in ExpertVersionChain.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 lora_rank: int = 8, lora_alpha: float = 1.0):
        super().__init__(vocab_size, block_size, n_embd, n_head, n_layer,
                         lora_rank, lora_alpha)
        self.version_chain = ExpertVersionChain()

    def snapshot_version(self):
        """Take a snapshot of current LoRA parameters as a new version."""
        params = {}
        for l_idx, layer in enumerate(self.layers):
            for name in ['fc1', 'fc2']:
                fc = getattr(layer.mlp, name)
                params[f"layer{l_idx}.{name}.A"] = mx.array(fc.A)
                params[f"layer{l_idx}.{name}.B"] = mx.array(fc.B)
        self.version_chain.add_version(params)

    def load_version(self, version_idx: int, use_compressed: bool = False):
        """Load a specific version's LoRA parameters from the chain."""
        params = self.version_chain.reconstruct(version_idx, use_compressed)
        for l_idx, layer in enumerate(self.layers):
            for name in ['fc1', 'fc2']:
                fc = getattr(layer.mlp, name)
                fc.A = params[f"layer{l_idx}.{name}.A"]
                fc.B = params[f"layer{l_idx}.{name}.B"]
        mx.eval(self.parameters())
