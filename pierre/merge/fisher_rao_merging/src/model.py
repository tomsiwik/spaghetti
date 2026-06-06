"""
Core spherical geometry primitives for Fisher-Rao Karcher mean merging.

Implements the spherical proxy from Section 3.4 of arXiv:2603.04972:
  1. Normalize parameter blocks to S^(d-1)
  2. Spherical log/exp maps (closed-form on the unit sphere)
  3. Karcher mean via fixed-point iteration (Section 3.3)
  4. Rescale by mean source norm (Section 3.4)

All operations use mlx.core (mx). No PyTorch.
"""

import mlx.core as mx


# ---- Spherical geometry on S^(d-1) ----


def normalize(v: mx.array, eps: float = 1e-8) -> tuple[mx.array, mx.array]:
    """Normalize vector to unit sphere, return (unit_vector, norm).

    Section 3.4: "normalize it to the unit sphere"
    """
    norm = mx.sqrt(mx.sum(v * v) + eps)
    return v / norm, norm


def spherical_distance(u: mx.array, v: mx.array) -> mx.array:
    """Geodesic distance on S^(d-1): d(u,v) = arccos(<u,v>).

    Section 3.2: Fisher-Rao geodesic distance under spherical proxy.
    u, v must be unit vectors.
    """
    dot = mx.clip(mx.sum(u * v), -1.0 + 1e-7, 1.0 - 1e-7)
    return mx.arccos(dot)


def log_map(base: mx.array, target: mx.array, eps: float = 1e-8) -> mx.array:
    """Riemannian log map on S^(d-1): Log_base(target).

    Section 3.3: Log_{theta}(theta^(i)) maps a point on the sphere
    to the tangent space at 'base'.

    Returns a tangent vector at 'base' pointing toward 'target',
    with magnitude equal to the geodesic distance.

    Formula: Log_p(q) = (theta / sin(theta)) * (q - cos(theta) * p)
    where theta = arccos(<p, q>).
    """
    dot = mx.clip(mx.sum(base * target), -1.0 + 1e-7, 1.0 - 1e-7)
    theta = mx.arccos(dot)

    # Tangent direction: project target onto tangent plane at base
    tangent = target - dot * base

    tangent_norm = mx.sqrt(mx.sum(tangent * tangent) + eps)

    # Scale by geodesic distance / tangent norm
    # When theta ~ 0, this reduces to (target - base) (Euclidean limit)
    scale = mx.where(tangent_norm > eps, theta / tangent_norm, 1.0)
    return scale * tangent


def exp_map(base: mx.array, tangent: mx.array, eps: float = 1e-8) -> mx.array:
    """Riemannian exp map on S^(d-1): Exp_base(tangent).

    Section 3.3: Exp_{theta}(v) maps a tangent vector back to the sphere.

    Formula: Exp_p(v) = cos(||v||) * p + sin(||v||) * (v / ||v||)
    """
    t_norm = mx.sqrt(mx.sum(tangent * tangent) + eps)

    # When tangent is near-zero, result is just base
    cos_t = mx.cos(t_norm)
    sin_t = mx.sin(t_norm)

    direction = mx.where(t_norm > eps, tangent / t_norm, tangent)
    result = cos_t * base + sin_t * direction
    # Re-normalize for numerical safety
    result_norm = mx.sqrt(mx.sum(result * result) + eps)
    return result / result_norm


def karcher_mean_spherical(
    points: list[mx.array],
    weights: list[float],
    max_iter: int = 100,
    step_size: float = 1.0,
    tol: float = 1e-7,
) -> mx.array:
    """Compute the weighted Karcher/Frechet mean on S^(d-1).

    Section 3.3, Eq. (5): Fixed-point iteration
      theta_{t+1} = Exp_{theta_t}(eta * sum_i alpha^(i) * Log_{theta_t}(theta^(i)))

    For N=2 with equal weights, this reduces to SLERP midpoint (Section 3.3).

    Args:
        points: list of N unit vectors on S^(d-1), each shape [d]
        weights: list of N non-negative weights summing to 1
        max_iter: maximum fixed-point iterations
        step_size: eta in (0, 1], Section 3.3
        tol: convergence tolerance (max angular change in radians)

    Returns:
        Unit vector: the Karcher mean on S^(d-1)
    """
    n = len(points)
    assert n == len(weights), "Must have same number of points and weights"
    assert abs(sum(weights) - 1.0) < 1e-6, f"Weights must sum to 1, got {sum(weights)}"

    # Initialize at weighted combination (projected to sphere)
    # This is a good starting point that often converges faster
    init = mx.zeros_like(points[0])
    for i in range(n):
        init = init + float(weights[i]) * points[i]
    current, _ = normalize(init)
    mx.eval(current)

    for iteration in range(max_iter):
        # Compute weighted average of log maps (Riemannian gradient)
        # Section 3.3 Eq. (5): sum_i alpha^(i) * Log_{theta}(theta^(i))
        weighted_tangent = mx.zeros_like(current)
        for i in range(n):
            log_vec = log_map(current, points[i])
            weighted_tangent = weighted_tangent + float(weights[i]) * log_vec

        # Check convergence: magnitude of the weighted tangent
        tangent_norm = mx.sqrt(mx.sum(weighted_tangent * weighted_tangent))
        mx.eval(tangent_norm)

        if tangent_norm.item() < tol:
            break

        # Riemannian gradient step: Exp_{theta}(eta * weighted_tangent)
        current = exp_map(current, float(step_size) * weighted_tangent)
        mx.eval(current)

    return current


def slerp(v0: mx.array, v1: mx.array, t: float = 0.5, eps: float = 1e-8) -> mx.array:
    """Spherical linear interpolation (SLERP) between two unit vectors.

    Section 3.3: "For a two-model merge... this reduces to SLERP."
    This is the N=2 special case of the Karcher mean.

    SLERP(v0, v1, t) = sin((1-t)*omega)/sin(omega) * v0 + sin(t*omega)/sin(omega) * v1
    where omega = arccos(<v0, v1>).
    """
    dot = mx.clip(mx.sum(v0 * v1), -1.0 + 1e-7, 1.0 - 1e-7)
    omega = mx.arccos(dot)
    sin_omega = mx.sin(omega)

    # When vectors are nearly parallel, fall back to linear interpolation
    use_lerp = sin_omega < eps

    c0 = mx.where(use_lerp, 1.0 - t, mx.sin((1.0 - t) * omega) / sin_omega)
    c1 = mx.where(use_lerp, t, mx.sin(t * omega) / sin_omega)

    result = c0 * v0 + c1 * v1
    result_norm = mx.sqrt(mx.sum(result * result) + eps)
    return result / result_norm
