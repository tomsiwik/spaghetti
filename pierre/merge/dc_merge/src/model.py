"""DC-Merge core operations: SVD decomposition, energy smoothing, cover space construction.

All operations use mlx.core. No PyTorch.

Reference: Algorithm 1 in "DC-Merge: Improving Model Merging with Directional Consistency"
           (arXiv 2603.06242, Zhang et al. 2025)
"""

from __future__ import annotations

import mlx.core as mx


# ---------------------------------------------------------------------------
# Step 1a: SVD decomposition of task vectors
# ---------------------------------------------------------------------------

def truncated_svd(
    delta: mx.array,
    rank: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Compute rank-r truncated SVD of a task vector matrix.

    Args:
        delta: Task vector matrix (m x n).
        rank:  Number of singular components to retain.

    Returns:
        U_r: (m, r) left singular vectors
        S_r: (r,)   singular values (descending)
        V_r: (n, r) right singular vectors (columns, NOT transposed)
    """
    # mx.linalg.svd returns U (m,k), S (k,), Vt (k,n) where k = min(m,n)
    U, S, Vt = mx.linalg.svd(delta, stream=mx.cpu)
    r = min(rank, S.shape[0])
    U_r = U[:, :r]
    S_r = S[:r]
    V_r = Vt[:r, :].T  # (n, r)
    return U_r, S_r, V_r


# ---------------------------------------------------------------------------
# Step 1b: Energy smoothing (Algorithm 1, line 6)
# ---------------------------------------------------------------------------

def energy_smoothing(
    singular_values: mx.array,
    strategy: str = "average",
    rho: float = 5.0,
) -> mx.array:
    """Smooth singular value energy distribution.

    Mitigates long-tailed energy distribution where a few dominant components
    overshadow weaker but semantically important knowledge components.

    Strategies (Appendix E.4):
        "average" -- Replace all top-r values with their mean.
                     sigma_bar = (1/r * sum(sigma_j)) * 1_r
        "linear"  -- Linearly decreasing distribution controlled by rho.
                     Constrains max/min ratio <= rho, preserves relative ordering.
        "none"    -- No explicit smoothing (implicit via SVD truncation).

    Args:
        singular_values: (r,) singular values in descending order.
        strategy: Smoothing strategy name.
        rho: Max/min ratio for linear smoothing.

    Returns:
        Smoothed singular values (r,).
    """
    if strategy == "none":
        return singular_values

    r = singular_values.shape[0]
    if r <= 1:
        return singular_values

    total_energy = mx.sum(singular_values)

    if strategy == "average":
        # Eq. 12: sigma_bar_i = mean(sigma) for all i
        mean_val = total_energy / r
        return mx.broadcast_to(mean_val.reshape(1), (r,))

    if strategy == "linear":
        # Appendix E.4: Linear smoothing
        # 1. Constrain ratio: sigma_max_smooth / sigma_min_smooth <= rho
        sigma_max = singular_values[0]
        sigma_min = singular_values[-1]
        actual_ratio = sigma_max / mx.maximum(sigma_min, mx.array(1e-10))
        effective_rho = float(mx.minimum(actual_ratio, mx.array(rho)).item())

        # 2. Generate linearly decreasing weights w_1 >= w_2 >= ... >= w_r
        #    such that w_1/w_r = effective_rho and sum(w) = 1
        indices = mx.arange(r, dtype=mx.float32)
        w = effective_rho - (effective_rho - 1.0) * indices / (r - 1)
        w = w / mx.sum(w)

        # 3. Scale by total energy: sigma_bar = total_energy * w
        smoothed = total_energy * w
        return smoothed

    raise ValueError(f"Unknown smoothing strategy: {strategy!r}. Use 'average', 'linear', or 'none'.")


# ---------------------------------------------------------------------------
# Step 2: Cover space construction (Algorithm 1, lines 9-10)
# ---------------------------------------------------------------------------

def _whiten(X: mx.array) -> mx.array:
    """Whiten columns of X so that X_white^T @ X_white = I.

    Uses SVD-based whitening: X = U S V^T => X_white = U V^T
    This gives an orthonormal basis spanning the same column space.

    Args:
        X: (d, k) matrix where k = r * T (concatenated basis vectors).

    Returns:
        X_white: (d, k) orthonormal matrix.
    """
    U, S, Vt = mx.linalg.svd(X, stream=mx.cpu)
    k = min(X.shape[0], X.shape[1])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    return U_k @ Vt_k


def construct_cover_basis(
    U_list: list[mx.array],
    V_list: list[mx.array],
) -> tuple[mx.array, mx.array]:
    """Construct shared orthonormal cover basis from per-task SVD bases.

    Eq. (9): Concatenate per-task bases column-wise, then whiten.
        U = [U_1^(r), ..., U_T^(r)]  =>  U_star = whiten(U)
        V = [V_1^(r), ..., V_T^(r)]  =>  V_star = whiten(V)

    The cover basis (U_star, V_star) defines a shared orthogonal subspace
    capturing the directional geometry of ALL task vectors.

    Args:
        U_list: List of T left singular vector matrices, each (m, r).
        V_list: List of T right singular vector matrices, each (n, r).

    Returns:
        U_star: (m, k) orthonormal left cover basis, k <= r * T
        V_star: (n, k) orthonormal right cover basis, k <= r * T
    """
    U_cat = mx.concatenate(U_list, axis=1)  # (m, r*T)
    V_cat = mx.concatenate(V_list, axis=1)  # (n, r*T)

    U_star = _whiten(U_cat)
    V_star = _whiten(V_cat)

    return U_star, V_star


# ---------------------------------------------------------------------------
# Step 3: Projection operations (Algorithm 1, lines 12 and 15-16)
# ---------------------------------------------------------------------------

def project_to_cover_space(
    delta: mx.array,
    U_star: mx.array,
    V_star: mx.array,
) -> mx.array:
    """Project a task vector onto the cover space.

    Eq. (10): M_i = U_star^T @ Delta_i @ V_star

    Args:
        delta:  (m, n) task vector matrix (after energy smoothing).
        U_star: (m, k) left cover basis.
        V_star: (n, k) right cover basis.

    Returns:
        M_i: (k, k) projection in cover space.
    """
    return U_star.T @ delta @ V_star


def _build_block_diag_mask(k: int, rank: int, num_tasks: int) -> mx.array:
    """Build block-diagonal mask M = block_diag(1_{r x r}, ..., 1_{r x r}).

    Algorithm 1, line 15. Each block is r x r ones, T blocks total.
    """
    # Build as list of rows for MLX compatibility
    blocks = []
    for t in range(num_tasks):
        start = t * rank
        end = min(start + rank, k)
        block_size = end - start
        # Row of zeros with a block of ones
        row_block = mx.concatenate([
            mx.zeros((block_size, start)),
            mx.ones((block_size, block_size)),
            mx.zeros((block_size, k - end)),
        ], axis=1)
        blocks.append(row_block)
    return mx.concatenate(blocks, axis=0)  # (k, k)


def project_to_param_space(
    M_merged: mx.array,
    U_star: mx.array,
    V_star: mx.array,
    use_mask: bool = True,
    rank: int = 0,
    num_tasks: int = 0,
) -> mx.array:
    """Project merged cover-space matrix back to parameter space.

    Eq. (11): Delta_merged = U_star @ (M_merged * Mask) @ V_star^T

    The structural mask M = block_diag(1_{r x r}, ..., 1_{r x r}) retains
    only within-block interactions, mitigating directional inconsistency
    between different tasks' subspaces.

    Args:
        M_merged: (k, k) merged matrix in cover space.
        U_star:   (m, k) left cover basis.
        V_star:   (n, k) right cover basis.
        use_mask: Whether to apply the structural block-diagonal mask.
        rank:     Per-task rank r (needed for mask construction).
        num_tasks: Number of tasks T (needed for mask construction).

    Returns:
        Delta_merged: (m, n) merged task vector in parameter space.
    """
    if use_mask and rank > 0 and num_tasks > 0:
        k = M_merged.shape[0]
        mask = _build_block_diag_mask(k, rank, num_tasks)
        M_merged = M_merged * mask

    return U_star @ M_merged @ V_star.T


# ---------------------------------------------------------------------------
# Directional similarity metrics (Section 2.3)
# ---------------------------------------------------------------------------

def dir_sim(delta_a: mx.array, delta_b: mx.array, rank: int) -> mx.array:
    """Compute Directional Similarity (DirSim) between two task vectors.

    Eq. (3): DirSim = (1/sqrt(n*m)) * sum_ij |R_ij(s,t)|
    where R_ij = (u_i^s)^T u_j^t * (v_j^t)^T v_i^s

    This metric removes the influence of energy distribution by uniformizing
    singular values, measuring purely directional consistency.

    Args:
        delta_a: (m, n) first task vector.
        delta_b: (m, n) second task vector.
        rank: Truncation rank for SVD.

    Returns:
        Scalar DirSim value.
    """
    U_a, _, V_a = truncated_svd(delta_a, rank)
    U_b, _, V_b = truncated_svd(delta_b, rank)
    mx.eval(U_a, V_a, U_b, V_b)

    n = U_a.shape[1]
    m = U_b.shape[1]

    # R_ij = (U_a^T @ U_b) * (V_b^T @ V_a)  element-wise
    R = (U_a.T @ U_b) * (V_b.T @ V_a).T  # (n, m)

    # DirSim = trace(sigma_bar_a^T @ |R| @ sigma_bar_b) with uniform sigmas
    # = (1/(sqrt(n)*sqrt(m))) * sum(|R|)
    return mx.sum(mx.abs(R)) / (mx.sqrt(mx.array(float(n))) * mx.sqrt(mx.array(float(m))))
