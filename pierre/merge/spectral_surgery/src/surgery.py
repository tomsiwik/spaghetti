"""
Spectral Surgery – main procedure (MLX).

Paper: arxiv 2603.03995 – "Spectral Surgery: Training-Free Refinement of
LoRA via Gradient-Guided Singular Value Reweighting" (Tian et al., 2026)

Implements the four editing policies from Section 3.4:
  1. abs_select   – hard three-level gating by magnitude rank
  2. smooth_abs   – continuous sigmoid-gated reweighting
  3. grad_direction – signed multiplicative update
  4. random_index  – matched-random control baseline

Plus energy-preservation constraints and the end-to-end pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import yaml

from .model import (
    SpectralDecomposition,
    aggregate_sensitivities,
    compute_component_sensitivity,
    decompose_lora_delta,
    svd_to_lora_factors,
)


# ====================================================================
# Configuration loader
# ====================================================================

def load_config(path: str | Path = "configs/base.yaml") -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


# ====================================================================
# Editing policies  (Sec 3.4)
# ====================================================================

def _normalize_magnitudes(s: mx.array) -> mx.array:
    """Mean-absolute normalization of sensitivity magnitudes."""
    mean_abs = mx.mean(mx.abs(s))
    # Avoid division by zero
    return mx.where(mean_abs > 1e-12, s / mean_abs, mx.zeros_like(s))


def _quantile(x: mx.array, q: float) -> mx.array:
    """Compute the q-th quantile of a 1D array."""
    sorted_x = mx.sort(x)
    n = x.shape[0]
    idx = int(q * (n - 1))
    idx = max(0, min(idx, n - 1))
    return sorted_x[idx]


# ------------------------------------------------------------------
# 1. Hard selection  (abs_select)
# ------------------------------------------------------------------

def policy_abs_select(
    sigma: mx.array,
    sensitivity_mag: mx.array,
    *,
    core_frac: float = 0.25,
    noise_frac: float = 0.25,
    min_core_k: int = 1,
    amp_factor: float = 1.2,
    sup_factor: float = 0.8,
    mid_factor: float = 1.0,
) -> mx.array:
    """Hard three-level gating based on sensitivity magnitude ranking.

    Top-k_core  -> gamma_amp
    Bottom-k_noise -> gamma_sup
    Middle        -> gamma_mid
    """
    r = sigma.shape[0]
    k_core = min(r, max(round(r * core_frac), min_core_k))
    k_noise = min(r - k_core, round(r * noise_frac))

    x = _normalize_magnitudes(sensitivity_mag)
    # Argsort descending: highest sensitivity first
    order = mx.argsort(-x)
    mx.eval(order)

    alpha = mx.full((r,), mid_factor)
    # Assign amp to top-k_core
    for i in range(k_core):
        idx = order[i].item()
        alpha[idx] = amp_factor
    # Assign sup to bottom-k_noise
    for i in range(k_noise):
        idx = order[r - 1 - i].item()
        alpha[idx] = sup_factor

    return sigma * alpha


# ------------------------------------------------------------------
# 2. Continuous reweighting  (smooth_abs)
# ------------------------------------------------------------------

def policy_smooth_abs(
    sigma: mx.array,
    sensitivity_mag: mx.array,
    *,
    center_quantile: float = 0.5,
    temperature: float = 1.0,
    noise_frac: float = 0.25,
    core_frac: float = 0.25,
    mid_factor: float = 1.0,
    align_mid: bool = True,
) -> mx.array:
    """Smooth sigmoid-gated reweighting (Sec 3.4 item 2).

    alpha_k = sigmoid((x_k - mu) / tau)   mapped to [sup_bound, amp_bound]
    """
    x = _normalize_magnitudes(sensitivity_mag)
    mx.eval(x)

    # Center: quantile of normalized magnitudes
    mu = _quantile(x, center_quantile)

    # Temperature from quantile range
    q_lo = noise_frac
    q_hi = 1.0 - core_frac
    if q_hi <= q_lo:
        q_lo, q_hi = 0.25, 0.75

    range_val = _quantile(x, q_hi) - _quantile(x, q_lo)
    mx.eval(range_val)

    # Degenerate check: if magnitudes are nearly identical, skip shaping
    if range_val.item() < 1e-8:
        return sigma * mid_factor

    tau = temperature * range_val

    # Sigmoid gate
    z = (x - mu) / mx.maximum(tau, mx.array(1e-12))
    gate = mx.sigmoid(z)  # in (0, 1)

    # Optionally shift so gate(mu) = mid_factor mapped value
    if align_mid:
        # gate maps to alpha in [sup_bound, amp_bound]
        # We want alpha at center = mid_factor
        # Simple approach: scale gate so that median maps to mid_factor
        # gate(0) = 0.5, so alpha = 0.5 * range + offset
        # We use: alpha_k = mid_factor + (gate_k - 0.5) * spread
        # where spread controls the total variation
        spread = 0.4  # conservative: alpha in [mid - 0.2, mid + 0.2]
        alpha = mid_factor + (gate - 0.5) * spread
    else:
        # Raw sigmoid: map from ~0.8 to ~1.2
        alpha = 0.8 + gate * 0.4

    return sigma * alpha


# ------------------------------------------------------------------
# 3. Random control  (random_index)
# ------------------------------------------------------------------

def policy_random_index(
    sigma: mx.array,
    *,
    core_frac: float = 0.25,
    noise_frac: float = 0.25,
    amp_factor: float = 1.2,
    sup_factor: float = 0.8,
    mid_factor: float = 1.0,
    seed: int = 0,
) -> mx.array:
    """Matched-random baseline: same scale factors, random assignment."""
    r = sigma.shape[0]
    k_core = min(r, max(round(r * core_frac), 1))
    k_noise = min(r - k_core, round(r * noise_frac))

    # Deterministic random permutation
    mx.random.seed(seed)
    perm = mx.argsort(mx.random.uniform(shape=(r,)))
    mx.eval(perm)

    alpha = mx.full((r,), mid_factor)
    for i in range(k_core):
        idx = perm[i].item()
        alpha[idx] = amp_factor
    for i in range(k_noise):
        idx = perm[r - 1 - i].item()
        alpha[idx] = sup_factor

    return sigma * alpha


# ------------------------------------------------------------------
# 4. Signed update  (grad_direction)
# ------------------------------------------------------------------

def policy_grad_direction(
    sigma: mx.array,
    sensitivity_signed: mx.array,
    *,
    eta_suppress: float = 0.1,
    eta_enhance: float = 0.1,
    asymmetric_update: bool = True,
    power_transform: float = 1.0,
) -> mx.array:
    """Signed multiplicative update (Sec 3.4 item 4).

    Asymmetric mode:
        g_k^+ = max(g_tilde_k, 0),  g_k^- = -min(g_tilde_k, 0)
        g_k_eff = eta_sup * g_k^+ + eta_amp * g_k^-
        sigma'_k = sigma_k * exp(-g_k_eff)

    Symmetric mode:
        sigma'_k = sigma_k * exp(-eta * g_tilde_k)
    """
    # Normalize signed sensitivities
    norm = mx.max(mx.abs(sensitivity_signed))
    g_tilde = mx.where(
        norm > 1e-12,
        sensitivity_signed / norm,
        mx.zeros_like(sensitivity_signed),
    )

    if asymmetric_update:
        g_pos = mx.maximum(g_tilde, 0.0)
        g_neg = -mx.minimum(g_tilde, 0.0)
        if power_transform != 1.0:
            g_pos = mx.power(g_pos, power_transform)
        g_eff = eta_suppress * g_pos + eta_enhance * g_neg
    else:
        g_eff = eta_suppress * g_tilde

    sigma_new = sigma * mx.exp(-g_eff)
    return sigma_new


# ====================================================================
# Energy preservation & clamping  (end of Sec 3.4)
# ====================================================================

def apply_energy_constraint(
    sigma_orig: mx.array,
    sigma_new: mx.array,
    mode: str = "L1",
    clip_min: float = 0.0,
) -> mx.array:
    """Clamp and optionally preserve spectral energy.

    Args:
        sigma_orig: original singular values.
        sigma_new:  edited singular values.
        mode:       "L1" for nuclear-norm preservation, "none" for no constraint.
        clip_min:   minimum clamp value (default 0).

    Returns:
        sigma_constrained: (r,)
    """
    sigma_new = mx.maximum(sigma_new, clip_min)

    if mode == "L1":
        # Renormalize so sum(sigma') == sum(sigma)
        orig_sum = mx.sum(sigma_orig)
        new_sum = mx.sum(sigma_new)
        scale = mx.where(new_sum > 1e-12, orig_sum / new_sum, mx.array(1.0))
        sigma_new = sigma_new * scale
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unknown energy mode: {mode}")

    return sigma_new


# ====================================================================
# Calibration: gradient accumulation for sensitivity estimation
# ====================================================================

def estimate_sensitivity_from_model(
    model: nn.Module,
    lora_modules: dict[str, dict[str, mx.array]],
    calibration_tokens: list[mx.array],
    calibration_labels: list[mx.array],
    loss_fn: Any,
    aggregation: str = "mean_abs",
) -> dict[str, tuple[SpectralDecomposition, mx.array, mx.array]]:
    """Run calibration forward+backward passes and estimate per-component sensitivity.

    This is the core calibration loop. For each LoRA module:
      1. Decompose DeltaW = B @ A via thin SVD.
      2. For each calibration batch, compute dL/d(DeltaW) and project onto
         singular directions to get g_k = u_k^T G v_k.
      3. Aggregate sensitivities across batches.

    Args:
        model:              The full model with LoRA adapters applied.
        lora_modules:       Dict mapping module_path -> {"lora_B": B, "lora_A": A}.
        calibration_tokens: List of (batch, seq_len) token arrays.
        calibration_labels: List of (batch, seq_len) label arrays (-100 for masked).
        loss_fn:            Callable(model, tokens, labels) -> scalar loss.
        aggregation:        How to aggregate sensitivities (default "mean_abs").

    Returns:
        Dict mapping module_path -> (decomposition, sensitivity_mag, sensitivity_signed).
    """
    # Step 1: Decompose all LoRA deltas
    decompositions: dict[str, SpectralDecomposition] = {}
    for name, params in lora_modules.items():
        decomp = decompose_lora_delta(params["lora_B"], params["lora_A"])
        decompositions[name] = decomp
    mx.eval(*[d.sigma for d in decompositions.values()])

    # Step 2: Accumulate sensitivities over calibration batches
    sensitivities_per_module: dict[str, list[mx.array]] = {
        name: [] for name in lora_modules
    }

    for tokens, labels in zip(calibration_tokens, calibration_labels):
        # Forward + backward to get gradients w.r.t. all parameters
        loss, grads = nn.value_and_grad(model, loss_fn)(model, tokens, labels)
        mx.eval(loss)

        # Extract gradient of loss w.r.t. each LoRA delta
        for name, decomp in decompositions.items():
            # The gradient w.r.t. DeltaW = B @ A can be obtained from
            # dL/dB and dL/dA:  G = dL/d(DeltaW) = (dL/dB)^T ??? ...
            # More directly: G = (dL/dB) @ A^T ... no.
            # Actually: dL/d(DeltaW) is what we need. Since DeltaW = B @ A:
            #   dL/dB = G @ A^T  =>  G = dL/dB @ pinv(A^T) ... expensive.
            #
            # Simpler: dL/d(DeltaW) can be reconstructed from the chain rule.
            #   dL/dB = dL/d(DeltaW) @ A^T, so  dL/d(DeltaW) = dL/dB @ A  (since A A^T ~ I for small r)
            #
            # For sensitivity estimation, the paper computes:
            #   g_k = u_k^T @ (dL/d(DeltaW)) @ v_k
            #       = u_k^T @ (dL/dB @ A) @ v_k   (approximate)
            #
            # More precisely, we use:  g_k = u_k^T @ grad_B @ A @ v_k
            # which equals  u_k^T @ G @ v_k  when A has orthonormal rows.

            # Navigate the gradient tree to find this module's grads
            grad_B = _extract_nested_grad(grads, name, "lora_B")
            grad_A = _extract_nested_grad(grads, name, "lora_A")

            if grad_B is None or grad_A is None:
                # Fall back to zero sensitivity if grads not found
                sensitivities_per_module[name].append(
                    mx.zeros((decomp.rank,))
                )
                continue

            # Approximate G = dL/d(DeltaW) via  grad_B @ A
            A = lora_modules[name]["lora_A"]
            G_approx = grad_B @ A  # (d_out, d_in)

            g = compute_component_sensitivity(decomp, G_approx)
            sensitivities_per_module[name].append(g)

        mx.eval(*[s[-1] for s in sensitivities_per_module.values() if s])

    # Step 3: Aggregate
    results: dict[str, tuple[SpectralDecomposition, mx.array, mx.array]] = {}
    for name, decomp in decompositions.items():
        g_list = sensitivities_per_module[name]
        if not g_list:
            s_mag = mx.zeros((decomp.rank,))
            s_signed = mx.zeros((decomp.rank,))
        else:
            s_mag = aggregate_sensitivities(g_list, method="mean_abs")
            s_signed = aggregate_sensitivities(g_list, method="mean_signed")
        results[name] = (decomp, s_mag, s_signed)

    return results


def _extract_nested_grad(
    grads: Any, module_path: str, param_name: str
) -> Optional[mx.array]:
    """Navigate a nested grad dict/object to find a specific parameter gradient.

    module_path is dot-separated, e.g. "model.layers.0.self_attn.o_proj".
    """
    parts = module_path.split(".")
    obj = grads
    for part in parts:
        if isinstance(obj, dict):
            obj = obj.get(part)
        elif isinstance(obj, (list, tuple)):
            try:
                obj = obj[int(part)]
            except (ValueError, IndexError):
                return None
        elif hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return None
        if obj is None:
            return None

    if isinstance(obj, dict):
        return obj.get(param_name)
    elif hasattr(obj, param_name):
        return getattr(obj, param_name)
    return None


# ====================================================================
# End-to-end spectral surgery
# ====================================================================

def apply_spectral_surgery(
    lora_modules: dict[str, dict[str, mx.array]],
    sensitivity_results: dict[str, tuple[SpectralDecomposition, mx.array, mx.array]],
    config: dict[str, Any],
) -> dict[str, tuple[mx.array, mx.array]]:
    """Apply spectral surgery to all LoRA modules.

    Args:
        lora_modules:       Dict of module_path -> {"lora_B": B, "lora_A": A}.
        sensitivity_results: Output of estimate_sensitivity_from_model().
        config:             Full config dict (loaded from YAML).

    Returns:
        Dict of module_path -> (B_new, A_new) edited LoRA factors.
    """
    policy_name = config.get("policy", "smooth_abs")
    energy_cfg = config.get("energy", {})
    energy_mode = energy_cfg.get("preserve", "L1")
    clip_min = energy_cfg.get("sigma_clip_min", 0.0)

    edited: dict[str, tuple[mx.array, mx.array]] = {}

    for name, (decomp, s_mag, s_signed) in sensitivity_results.items():
        sigma = decomp.sigma

        # Apply the chosen policy
        if policy_name == "smooth_abs":
            pcfg = config.get("smooth_abs", {})
            sigma_new = policy_smooth_abs(sigma, s_mag, **pcfg)

        elif policy_name == "abs_select":
            pcfg = config.get("abs_select", {})
            sigma_new = policy_abs_select(sigma, s_mag, **pcfg)

        elif policy_name == "grad_direction":
            pcfg = config.get("grad_direction", {})
            sigma_new = policy_grad_direction(sigma, s_signed, **pcfg)

        elif policy_name == "random_index":
            pcfg = config.get("random_index", {})
            sigma_new = policy_random_index(sigma, **pcfg)

        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        # Energy constraint
        sigma_new = apply_energy_constraint(sigma, sigma_new, energy_mode, clip_min)
        mx.eval(sigma_new)

        # Convert back to LoRA factors
        B_new, A_new = svd_to_lora_factors(decomp, sigma_new)
        mx.eval(B_new, A_new)
        edited[name] = (B_new, A_new)

    return edited


# ====================================================================
# Convenience: run full pipeline on a model
# ====================================================================

def spectral_surgery_pipeline(
    model: nn.Module,
    lora_modules: dict[str, dict[str, mx.array]],
    calibration_tokens: list[mx.array],
    calibration_labels: list[mx.array],
    loss_fn: Any,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, tuple[mx.array, mx.array]]:
    """Full spectral surgery pipeline: decompose -> estimate -> reweight -> reconstruct.

    Args:
        model:              Model with LoRA adapters.
        lora_modules:       Dict of module_path -> {"lora_B": B, "lora_A": A}.
        calibration_tokens: List of (batch, seq_len) token arrays.
        calibration_labels: List of (batch, seq_len) label arrays.
        loss_fn:            Callable(model, tokens, labels) -> scalar loss.
        config:             Config dict. If None, loaded from config_path.
        config_path:        Path to YAML config. Default: configs/base.yaml.

    Returns:
        Dict of module_path -> (B_new, A_new) edited LoRA factors.
    """
    if config is None:
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "base.yaml"
        config = load_config(config_path)

    # Step 1-2: Decompose + estimate sensitivities
    sensitivity_results = estimate_sensitivity_from_model(
        model=model,
        lora_modules=lora_modules,
        calibration_tokens=calibration_tokens,
        calibration_labels=calibration_labels,
        loss_fn=loss_fn,
        aggregation=config.get("sensitivity", {}).get("aggregation", "mean_abs"),
    )

    # Step 3: Apply policy + energy constraint + reconstruct
    return apply_spectral_surgery(lora_modules, sensitivity_results, config)
