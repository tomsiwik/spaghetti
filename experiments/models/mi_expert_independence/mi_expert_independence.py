"""MI Expert Independence: Mutual information vs cosine for expert independence.

This is a DIAGNOSTIC experiment. The model class is identical to CapsuleMoEGPT.
The experiment measures MI between expert (capsule group) outputs and compares
MI vs cosine similarity as predictors of composition quality.

Key design decisions:
- KSG estimator for MI (scipy digamma, no sklearn dependency)
- Operate on per-capsule activation scalars (1D) for reliable MI estimation
- Also compute group-level output MI via PCA reduction to d=4
- Compare against cosine similarity as baseline metric
"""

import random
import time
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from scipy.special import digamma
from scipy.spatial import KDTree

from .. import register
from ..capsule_moe.capsule_moe import CapsuleMoEGPT, CapsulePool


# ---------------------------------------------------------------------------
# 1. KSG Mutual Information Estimator (Kraskov et al. 2004, Algorithm 1)
# ---------------------------------------------------------------------------

def ksg_mi(x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    """KSG mutual information estimator for 1D or low-D continuous variables.

    Implements Algorithm 1 from Kraskov, Stogbauer, Grassberger (2004).
    I(X;Y) = psi(k) - <psi(n_x+1) + psi(n_y+1)> + psi(N)

    Args:
        x: (N,) or (N, d_x) array of samples
        y: (N,) or (N, d_y) array of samples
        k: number of nearest neighbors (default 3)

    Returns:
        Estimated MI in nats. Clipped to >= 0.
    """
    N = len(x)
    if N < k + 1:
        return 0.0

    # Ensure 2D
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Joint space
    xy = np.concatenate([x, y], axis=1)

    # Build KD-trees
    tree_xy = KDTree(xy)
    tree_x = KDTree(x)
    tree_y = KDTree(y)

    # For each point, find distance to k-th neighbor in joint space
    # query k+1 because the point itself is included
    dists_xy, _ = tree_xy.query(xy, k=k + 1)
    eps = dists_xy[:, -1]  # distance to k-th neighbor (Chebyshev/max norm approx)

    # Count neighbors within eps in marginal spaces
    # Using eps as radius with Chebyshev metric
    nx = np.zeros(N)
    ny = np.zeros(N)
    for i in range(N):
        # Number of points within eps[i] in x-space (excluding self)
        nx[i] = max(tree_x.query_ball_point(x[i], eps[i] + 1e-15, return_length=True) - 1, 0)
        ny[i] = max(tree_y.query_ball_point(y[i], eps[i] + 1e-15, return_length=True) - 1, 0)

    # KSG formula
    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(N)
    return max(float(mi), 0.0)


def ksg_mi_1d_fast(x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    """Optimized KSG for 1D x and 1D y using sorted arrays.

    Much faster than generic KSG for scalar activations.
    """
    N = len(x)
    if N < k + 1:
        return 0.0

    # Add tiny noise to break ties (important for discrete-ish activations)
    x = x + np.random.RandomState(42).randn(N) * 1e-10
    y = y + np.random.RandomState(43).randn(N) * 1e-10

    # Joint space with Chebyshev (max) metric
    xy = np.column_stack([x, y])
    tree_xy = KDTree(xy, leafsize=16)

    dists_xy, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = dists_xy[:, -1]

    # Count in marginals using sorted arrays (O(N log N) total)
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)

    nx = np.zeros(N)
    ny = np.zeros(N)
    for i in range(N):
        # Count points within eps[i] of x[i] in x-space
        lo = np.searchsorted(x_sorted, x[i] - eps[i] - 1e-15)
        hi = np.searchsorted(x_sorted, x[i] + eps[i] + 1e-15)
        nx[i] = max(hi - lo - 1, 0)  # exclude self

        lo = np.searchsorted(y_sorted, y[i] - eps[i] - 1e-15)
        hi = np.searchsorted(y_sorted, y[i] + eps[i] + 1e-15)
        ny[i] = max(hi - lo - 1, 0)

    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(N)
    return max(float(mi), 0.0)


# ---------------------------------------------------------------------------
# 2. Cosine similarity between group output vectors
# ---------------------------------------------------------------------------

def pairwise_cosine(outputs: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise cosine similarity between group outputs.

    Args:
        outputs: list of (N_samples, d) arrays, one per group

    Returns:
        (G, G) cosine similarity matrix
    """
    G = len(outputs)
    cos_mat = np.zeros((G, G))
    for i in range(G):
        for j in range(i + 1, G):
            # Flatten to vectors and compute cosine
            a = outputs[i].flatten()
            b = outputs[j].flatten()
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a < 1e-10 or norm_b < 1e-10:
                cos_mat[i, j] = 0.0
            else:
                cos_mat[i, j] = float(np.dot(a, b) / (norm_a * norm_b))
            cos_mat[j, i] = cos_mat[i, j]
    return cos_mat


# ---------------------------------------------------------------------------
# 3. MI between capsule group activation patterns
# ---------------------------------------------------------------------------

def pairwise_mi_activations(activations: list[np.ndarray], k: int = 3) -> np.ndarray:
    """Compute pairwise MI between group activation patterns.

    For each pair (i,j), we compute MI between the mean activation of group i
    and group j across samples. This captures nonlinear co-activation dependencies.

    Args:
        activations: list of (N_samples, n_capsules) arrays, one per group
        k: KSG neighbor count

    Returns:
        (G, G) MI matrix in nats
    """
    G = len(activations)
    mi_mat = np.zeros((G, G))
    for i in range(G):
        for j in range(i + 1, G):
            # Mean activation per group (scalar per sample)
            a_mean = np.mean(activations[i], axis=1)
            b_mean = np.mean(activations[j], axis=1)
            mi = ksg_mi_1d_fast(a_mean, b_mean, k=k)
            mi_mat[i, j] = mi
            mi_mat[j, i] = mi
    return mi_mat


def pairwise_mi_outputs_pca(outputs: list[np.ndarray], d_pca: int = 4, k: int = 3) -> np.ndarray:
    """Compute pairwise MI between group outputs after PCA reduction.

    Reduces each group's output from R^d to R^d_pca, then uses KSG.
    The PCA basis is computed jointly across all groups to ensure shared space.

    Args:
        outputs: list of (N_samples, d) arrays
        d_pca: PCA dimensionality
        k: KSG neighbor count

    Returns:
        (G, G) MI matrix in nats
    """
    G = len(outputs)

    # Joint PCA basis
    all_outputs = np.concatenate(outputs, axis=0)
    mean = np.mean(all_outputs, axis=0)
    centered = all_outputs - mean
    # SVD for PCA
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:d_pca]  # (d_pca, d)

    # Project each group
    projected = [(out - mean) @ basis.T for out in outputs]

    mi_mat = np.zeros((G, G))
    for i in range(G):
        for j in range(i + 1, G):
            mi = ksg_mi(projected[i], projected[j], k=k)
            mi_mat[i, j] = mi
            mi_mat[j, i] = mi
    return mi_mat


# ---------------------------------------------------------------------------
# 4. Profiling: collect group outputs and activations
# ---------------------------------------------------------------------------

def profile_groups(model: CapsuleMoEGPT, dataset, n_batches: int = 20,
                   batch_size: int = 32, seed: int = 0) -> dict:
    """Collect per-group outputs and activations on calibration data.

    Returns:
        dict with keys per layer:
        {
            'layer_0': {
                'outputs': list of (N_total, d) np arrays, one per group,
                'activations': list of (N_total, n_caps) np arrays,
            },
            ...
        }
    """
    rng = random.Random(seed)
    results = {}

    for layer_idx, layer in enumerate(model.layers):
        pool = layer.capsule_pool
        G = pool.n_groups
        n_caps = pool.groups[0].A.weight.shape[0]
        d = pool.groups[0].A.weight.shape[1]

        group_outputs = [[] for _ in range(G)]
        group_activations = [[] for _ in range(G)]

        for _ in range(n_batches):
            inputs, _ = dataset.get_batch(batch_size, rng)
            B, T = inputs.shape

            # Forward through embeddings + norm + previous layers
            pos = mx.arange(T)
            x = model.wte(inputs) + model.wpe(pos)
            x = model.norm0(x)
            for prev_layer in model.layers[:layer_idx]:
                x = prev_layer(x)

            # Get input to this layer's capsule pool
            h = layer.norm2(x + layer.attn(layer.norm1(x)))  # (B, T, d)
            mx.eval(h)

            # Collect per-group outputs and activations
            for g_idx, group in enumerate(pool.groups):
                act = nn.relu(group.A(h))  # (B, T, n_caps)
                out = group.B(act)  # (B, T, d)
                mx.eval(act, out)

                # Flatten batch and time dimensions
                act_np = np.array(act.reshape(-1, n_caps))
                out_np = np.array(out.reshape(-1, d))
                group_activations[g_idx].append(act_np)
                group_outputs[g_idx].append(out_np)

        # Concatenate across batches
        results[f'layer_{layer_idx}'] = {
            'outputs': [np.concatenate(go) for go in group_outputs],
            'activations': [np.concatenate(ga) for ga in group_activations],
        }

    return results


# ---------------------------------------------------------------------------
# 5. Full diagnostic: compute all metrics
# ---------------------------------------------------------------------------

def compute_independence_metrics(profile_data: dict, k: int = 3, d_pca: int = 4) -> dict:
    """Compute cosine, MI-activation, and MI-PCA for all layers.

    Returns dict per layer with:
        cosine_matrix, mi_activation_matrix, mi_pca_matrix,
        and timing information.
    """
    results = {}
    for layer_key, layer_data in profile_data.items():
        outputs = layer_data['outputs']
        activations = layer_data['activations']

        # Cosine similarity
        t0 = time.time()
        cos_mat = pairwise_cosine(outputs)
        t_cosine = time.time() - t0

        # MI on mean activation (1D KSG, fast)
        t0 = time.time()
        mi_act_mat = pairwise_mi_activations(activations, k=k)
        t_mi_act = time.time() - t0

        # MI on PCA-reduced outputs
        t0 = time.time()
        mi_pca_mat = pairwise_mi_outputs_pca(outputs, d_pca=d_pca, k=k)
        t_mi_pca = time.time() - t0

        results[layer_key] = {
            'cosine': cos_mat,
            'mi_activation': mi_act_mat,
            'mi_pca': mi_pca_mat,
            'time_cosine_s': t_cosine,
            'time_mi_activation_s': t_mi_act,
            'time_mi_pca_s': t_mi_pca,
            'n_samples': len(outputs[0]),
            'n_groups': len(outputs),
        }
    return results


# ---------------------------------------------------------------------------
# 6. Model class (thin wrapper for registry)
# ---------------------------------------------------------------------------

@register("mi_expert_independence", parent="capsule_moe")
class MIExpertIndependenceGPT(CapsuleMoEGPT):
    """CapsuleMoEGPT with MI-based independence diagnostics.

    The model itself is identical to CapsuleMoEGPT. The experiment logic
    lives in the profiling and metric computation functions above.
    """
    pass
