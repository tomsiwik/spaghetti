"""
Diagnostic metrics for evaluating merge quality.

Not a training loss -- the Karcher mean merge is training-free.
These metrics correspond to the collapse diagnostics in Section 4.2 (Q4)
and the Fisher-Rao distance objective from Section 3.2.

All operations use mlx.core (mx). No PyTorch.
"""

import mlx.core as mx

from .model import normalize, spherical_distance


def fisher_rao_proxy_loss(
    merged_params: dict[str, mx.array],
    source_params_list: list[dict[str, mx.array]],
    weights: list[float],
) -> mx.array:
    """Compute the spherical proxy of the Fisher-Rao barycentric objective.

    Section 3.2 Eq. (3): sum_i alpha^(i) * d_FR(theta_merged, theta^(i))^2

    Under the spherical proxy (Section 3.4), d_FR is approximated by
    the geodesic distance on S^(d-1) for each parameter block.

    Returns the weighted sum of squared geodesic distances.
    """
    total_loss = mx.array(0.0)
    keys = sorted(merged_params.keys())

    for key in keys:
        merged_unit, _ = normalize(merged_params[key].reshape(-1))

        for i, source in enumerate(source_params_list):
            source_unit, _ = normalize(source[key].reshape(-1))
            dist = spherical_distance(merged_unit, source_unit)
            total_loss = total_loss + float(weights[i]) * dist * dist

    mx.eval(total_loss)
    return total_loss


def activation_variance(activations: mx.array) -> mx.array:
    """Mean activation variance across features.

    Section 4.2, Q4 / Figure 2(a): "mean activation variance across
    transformer layers." Low variance indicates collapse.

    Args:
        activations: shape [batch, seq_len, hidden_dim]

    Returns:
        Scalar: mean variance across the feature dimension.
    """
    # Variance over batch and seq_len dimensions, then mean over features
    var = mx.var(activations, axis=(0, 1))
    return mx.mean(var)


def effective_rank(activations: mx.array, eps: float = 1e-10) -> mx.array:
    """Effective rank of the activation covariance matrix.

    Section 4.2, Q4 / Figure 2(b): "effective rank (EffRank) of the
    activation covariance."

    EffRank = exp(H(p)) where p_i = sigma_i / sum(sigma_j) are the
    normalized singular values and H is Shannon entropy.

    Args:
        activations: shape [batch * seq_len, hidden_dim] or [N, D]

    Returns:
        Scalar: effective rank.
    """
    # Flatten to 2D if needed
    if activations.ndim > 2:
        shape = activations.shape
        activations = activations.reshape(-1, shape[-1])

    # Compute SVD of the activation matrix
    # Use covariance for efficiency when N >> D
    n, d = activations.shape
    if n > d:
        cov = (activations.T @ activations) / float(n)
        # Eigenvalues of covariance = singular values squared / n
        # For effective rank we only need the spectrum
        eigvals = mx.linalg.eigvalsh(cov)
        eigvals = mx.maximum(eigvals, eps)
        # Normalized spectrum
        p = eigvals / mx.sum(eigvals)
    else:
        # Thin SVD path
        cov = (activations @ activations.T) / float(n)
        eigvals = mx.linalg.eigvalsh(cov)
        eigvals = mx.maximum(eigvals, eps)
        p = eigvals / mx.sum(eigvals)

    # Shannon entropy of normalized spectrum
    entropy = -mx.sum(p * mx.log(p + eps))
    eff_rank = mx.exp(entropy)
    mx.eval(eff_rank)
    return eff_rank


def norm_shrinkage_ratio(
    merged_params: dict[str, mx.array],
    source_params_list: list[dict[str, mx.array]],
) -> dict[str, float]:
    """Compute norm shrinkage ratio per parameter block.

    Section 3.4: "norm shrinkage is a major failure mode of Euclidean
    interpolation" (citing Jang et al., 2024).

    Ratio = ||merged_block|| / mean(||source_block_i||)
    Values near 1.0 indicate good norm preservation.
    Values << 1.0 indicate norm shrinkage (collapse risk).
    """
    ratios = {}
    keys = sorted(merged_params.keys())

    for key in keys:
        merged_norm = mx.sqrt(mx.sum(merged_params[key].reshape(-1) ** 2))

        source_norms = []
        for source in source_params_list:
            sn = mx.sqrt(mx.sum(source[key].reshape(-1) ** 2))
            source_norms.append(sn)

        mean_source_norm = mx.mean(mx.stack(source_norms))
        mx.eval(merged_norm, mean_source_norm)

        ratio = (merged_norm / (mean_source_norm + 1e-8)).item()
        ratios[key] = ratio

    return ratios
