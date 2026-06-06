#!/usr/bin/env python3
"""
Compressed Expert Sweep: LoRA vs LoRA-XS vs VeRA for SOLE composition.

Hypothesis: Compressed expert formats (LoRA-XS, VeRA) can encode sufficient
  domain knowledge for SOLE composition while reducing per-expert storage
  by 100-1400x, without destroying the structural orthogonality guarantee.

Key theoretical question: LoRA's orthogonality comes from random low-rank
  subspaces in high-d space. LoRA-XS constrains experts to a SHARED SVD basis.
  VeRA constrains experts to SHARED random matrices with per-expert scaling.
  Do these constraints destroy the geometric diversity that makes orthogonality work?

Method: Synthetic domain perturbations (no FD training -- direct analytical
  construction of domain-specific deltas in each format). This isolates
  the FORMAT's geometric properties from training dynamics.

Three adapter formats:
  LoRA:    dW = B @ A,       B: (d_out, r), A: (r, d_in)  -- 2*d*r params/expert
  LoRA-XS: dW = U_r @ M @ V_r^T,  M: (r, r)              -- r^2 params/expert
  VeRA:   dW = diag(lb) @ B_s @ diag(ld) @ A_s            -- (d_out + r) params/expert
           B_s, A_s shared random matrices across experts

Kill criteria:
  K1: compressed experts encode <50% of domain knowledge vs standard LoRA
  K2: compressed experts lose orthogonality (cos > 0.01)
  K3: inference overhead of compressed format exceeds 10%

Pure numpy/scipy, CPU-only, float32. Expected runtime: ~2-4 minutes.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import svd as scipy_svd

DTYPE = np.float32
RESULTS_DIR = Path(__file__).parent

# =============================================================================
# Constants
# =============================================================================

LORA_RANK = 8
N_EXPERTS = 8
N_LAYERS = 2
D_FF_MULT = 4
SEEDS = [42, 137]

D_VALUES = [64, 256, 896]


# =============================================================================
# Utilities
# =============================================================================

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def measure_pairwise_cosines(deltas):
    n = len(deltas)
    cosines = []
    for i in range(n):
        for j in range(i + 1, n):
            c = abs(cosine_sim(deltas[i], deltas[j]))
            cosines.append(c)
    return np.array(cosines)


def frobenius_ratio(approx, target):
    """||approx - target||_F / ||target||_F. Lower = better reconstruction."""
    diff_norm = np.linalg.norm(approx - target)
    tgt_norm = np.linalg.norm(target)
    if tgt_norm < 1e-12:
        return 0.0
    return float(diff_norm / tgt_norm)


# =============================================================================
# Domain Perturbation Generation
# =============================================================================

def generate_domain_perturbation(rows, cols, rank, rng, domain_seed):
    """Generate a rank-r domain-specific weight perturbation of shape (rows, cols).

    We simulate what LoRA training would produce: a rank-r matrix that
    represents domain-specific knowledge. Each domain gets a different
    random low-rank perturbation.
    """
    dr = np.random.RandomState(domain_seed)
    # Left and right factors with domain-specific structure
    left = (dr.randn(rows, rank) * np.sqrt(2.0 / rows)).astype(DTYPE)
    right = (dr.randn(rank, cols) * np.sqrt(2.0 / cols)).astype(DTYPE)
    # Scale to realistic adapter magnitude (0.01 * sigma of pretrained weight)
    scale = 0.01
    return (left @ right * scale).astype(DTYPE)


# =============================================================================
# Format-Specific Fitting
# =============================================================================

def fit_lora(target_dW, d_in, d_out, rank, rng):
    """Fit standard LoRA: find best B, A such that B @ A approx target_dW.
    Uses truncated SVD of target.
    Returns: delta (B @ A), params dict.
    """
    U, s, Vt = np.linalg.svd(target_dW, full_matrices=False)
    # Truncate to rank r
    U_r = U[:, :rank]
    s_r = s[:rank]
    Vt_r = Vt[:rank, :]
    # B = U_r * sqrt(s_r), A = sqrt(s_r)[:, None] * Vt_r
    sqrt_s = np.sqrt(s_r).astype(DTYPE)
    B = (U_r * sqrt_s[np.newaxis, :]).astype(DTYPE)
    A = (sqrt_s[:, np.newaxis] * Vt_r).astype(DTYPE)
    delta = B @ A
    n_params = B.size + A.size  # 2 * d * r per layer
    return delta, {'B': B, 'A': A, 'n_params': n_params}


def fit_lora_xs(target_dW, U_r, Vt_r, rank, rng):
    """Fit LoRA-XS: find best M such that U_r @ M @ V_r^T approx target_dW.
    Optimal M = U_r^T @ target_dW @ V_r (by projection).
    Returns: delta (U_r @ M @ V_r^T), params dict.
    """
    # V_r = Vt_r^T, so target projected: M = U_r^T @ target @ (Vt_r^T) = U_r^T @ target @ V_r
    V_r = Vt_r.T  # (d_in, r)
    M = (U_r.T @ target_dW @ V_r).astype(DTYPE)
    delta = (U_r @ M @ Vt_r).astype(DTYPE)
    n_params = M.size  # r^2 per layer
    return delta, {'M': M, 'n_params': n_params}


def fit_vera(target_dW, B_shared, A_shared, d_out, rank, rng):
    """Fit VeRA: find best lambda_b, lambda_d such that
    diag(lb) @ B @ diag(ld) @ A approx target_dW.

    This is a bilinear problem. We use alternating least squares (vectorized).
    Returns: delta, params dict.
    """
    lb = np.ones(d_out, dtype=DTYPE)
    ld = np.ones(rank, dtype=DTYPE)

    # Precompute useful quantities
    # AAt: (r, r) = A @ A^T -- for solving ld
    AAt = (A_shared @ A_shared.T).astype(DTYPE)  # (r, r)

    for iteration in range(20):
        # === Fix ld, solve for lb (vectorized over rows) ===
        # dW = diag(lb) @ B @ diag(ld) @ A
        # BdA = B @ diag(ld) @ A: shape (d_out, d_in)
        B_scaled = B_shared * ld[np.newaxis, :]  # (d_out, r)
        BdA = B_scaled @ A_shared  # (d_out, d_in)
        # For row i: lb[i] = <target[i,:], BdA[i,:]> / <BdA[i,:], BdA[i,:]>
        numerator = np.sum(target_dW * BdA, axis=1)   # (d_out,)
        denominator = np.sum(BdA * BdA, axis=1)       # (d_out,)
        mask = denominator > 1e-12
        lb[mask] = numerator[mask] / denominator[mask]

        # === Fix lb, solve for ld (vectorized) ===
        # dW = diag(lb) @ B @ diag(ld) @ A = (lb[:,None] * B) @ diag(ld) @ A
        # Let C = diag(lb) @ B: (d_out, r)
        C = lb[:, np.newaxis] * B_shared  # (d_out, r)
        # Target projected: T_proj = C^T @ target_dW @ A^T: (r, r) but we only need diag
        # dW approx = C @ diag(ld) @ A
        # Normal equations per component k: ld[k] * sum_j (C^TC)_{kj} * (A A^T)_{jk} but
        # actually we need to solve: C^T @ (C @ diag(ld) @ A) @ A^T = C^T @ target @ A^T
        # i.e., (C^TC) @ diag(ld) @ (A A^T) = C^T @ target @ A^T
        # This is a coupled system. For simplicity, use coordinate descent (r is small).
        CtC = C.T @ C  # (r, r)
        Gram = CtC * AAt  # element-wise product gives the Gram matrix for ld
        rhs = np.diag(C.T @ target_dW @ A_shared.T)  # (r,) -- diagonal of C^T @ T @ A^T
        # Actually the full linear system is:
        # sum_j Gram[k,j] * ld[j] = rhs[k] but rhs should be full:
        rhs_full = np.sum((C.T @ target_dW) * A_shared, axis=1)  # (r,)
        # Solve: Gram @ ld = rhs_full
        try:
            ld = np.linalg.solve(Gram + 1e-8 * np.eye(rank, dtype=DTYPE), rhs_full).astype(DTYPE)
        except np.linalg.LinAlgError:
            pass  # keep current ld

    # Compute final delta
    B_scaled = B_shared * ld[np.newaxis, :]
    B_scaled = B_scaled * lb[:, np.newaxis]
    delta = (B_scaled @ A_shared).astype(DTYPE)
    n_params = lb.size + ld.size  # d_out + r per layer
    return delta, {'lambda_b': lb, 'lambda_d': ld, 'n_params': n_params}


# =============================================================================
# Inference Timing
# =============================================================================

def measure_inference_overhead(d, d_ff, rng, n_iters=200):
    """Measure delta computation time for each format."""
    r = LORA_RANK

    # Create random matrices for each format
    B = rng.randn(d_ff, r).astype(DTYPE)
    A = rng.randn(r, d).astype(DTYPE)
    U_r = rng.randn(d_ff, r).astype(DTYPE)
    M = rng.randn(r, r).astype(DTYPE)
    Vt_r = rng.randn(r, d).astype(DTYPE)
    B_s = rng.randn(d_ff, r).astype(DTYPE)
    A_s = rng.randn(r, d).astype(DTYPE)
    lb = rng.randn(d_ff).astype(DTYPE)
    ld = rng.randn(r).astype(DTYPE)

    # Warm up
    for _ in range(20):
        _ = B @ A
        _ = U_r @ M @ Vt_r
        _ = (lb[:, None] * B_s * ld[None, :]) @ A_s

    # LoRA: B @ A
    t0 = time.perf_counter()
    for _ in range(n_iters):
        dW = B @ A
    lora_time = (time.perf_counter() - t0) / n_iters

    # LoRA-XS: U_r @ M @ V_r^T (two matmuls)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        dW = U_r @ M @ Vt_r
    xs_time = (time.perf_counter() - t0) / n_iters

    # VeRA: diag(lb) @ B_s @ diag(ld) @ A_s
    t0 = time.perf_counter()
    for _ in range(n_iters):
        dW = (lb[:, None] * B_s * ld[None, :]) @ A_s
    vera_time = (time.perf_counter() - t0) / n_iters

    return {
        'lora_us': lora_time * 1e6,
        'lora_xs_us': xs_time * 1e6,
        'vera_us': vera_time * 1e6,
        'xs_overhead': (xs_time - lora_time) / lora_time if lora_time > 0 else 0,
        'vera_overhead': (vera_time - lora_time) / lora_time if lora_time > 0 else 0,
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(d, seed):
    """Full comparison for one dimension and seed."""
    rng = np.random.RandomState(seed)
    d_ff = D_FF_MULT * d
    r = LORA_RANK

    print(f"\n{'='*60}")
    print(f"  d={d}, seed={seed}, r={r}, N_experts={N_EXPERTS}")
    print(f"{'='*60}")

    # Generate "pretrained" weights for SVD basis
    pretrained_W1 = [(rng.randn(d, d_ff) * 0.02).astype(DTYPE) for _ in range(N_LAYERS)]
    pretrained_W2 = [(rng.randn(d_ff, d) * 0.02).astype(DTYPE) for _ in range(N_LAYERS)]

    # Compute SVD bases for LoRA-XS (shared across experts)
    svd_bases = []
    for l in range(N_LAYERS):
        U1, s1, Vt1 = np.linalg.svd(pretrained_W1[l], full_matrices=False)
        U2, s2, Vt2 = np.linalg.svd(pretrained_W2[l], full_matrices=False)
        svd_bases.append({
            'W1': (U1[:, :r].astype(DTYPE), Vt1[:r, :].astype(DTYPE)),
            'W2': (U2[:, :r].astype(DTYPE), Vt2[:r, :].astype(DTYPE)),
        })

    # Shared random matrices for VeRA
    # For weight W1: (d, d_ff), B: (d, r), A: (r, d_ff)
    # For weight W2: (d_ff, d), B: (d_ff, r), A: (r, d)
    vera_shared = []
    for l in range(N_LAYERS):
        vera_shared.append({
            'W1': (
                (rng.randn(d, r) * np.sqrt(2.0 / r)).astype(DTYPE),      # B: (d, r)
                (rng.randn(r, d_ff) * np.sqrt(2.0 / d_ff)).astype(DTYPE),  # A: (r, d_ff)
            ),
            'W2': (
                (rng.randn(d_ff, r) * np.sqrt(2.0 / r)).astype(DTYPE),   # B: (d_ff, r)
                (rng.randn(r, d) * np.sqrt(2.0 / d)).astype(DTYPE),      # A: (r, d)
            ),
        })

    # Generate ground truth domain perturbations (rank-r)
    target_deltas = []  # per expert: list of (dW1, dW2) per layer
    for expert_i in range(N_EXPERTS):
        layers = []
        for l in range(N_LAYERS):
            # dW1 matches W1 shape: (d, d_ff)
            dW1 = generate_domain_perturbation(d, d_ff, r, rng,
                                               domain_seed=seed * 10000 + expert_i * 100 + l * 10)
            # dW2 matches W2 shape: (d_ff, d)
            dW2 = generate_domain_perturbation(d_ff, d, r, rng,
                                               domain_seed=seed * 10000 + expert_i * 100 + l * 10 + 1)
            layers.append((dW1, dW2))
        target_deltas.append(layers)

    # Fit each format and measure
    results = {}

    for fmt in ['lora', 'lora_xs', 'vera']:
        print(f"\n  Fitting {fmt}...")
        t0 = time.perf_counter()

        expert_flat_deltas = []
        reconstruction_errors = []
        total_params_list = []

        for expert_i in range(N_EXPERTS):
            flat_parts = []
            expert_total_params = 0

            for l in range(N_LAYERS):
                dW1_target, dW2_target = target_deltas[expert_i][l]

                if fmt == 'lora':
                    delta1, info1 = fit_lora(dW1_target, d, d_ff, r, rng)
                    delta2, info2 = fit_lora(dW2_target, d_ff, d, r, rng)
                elif fmt == 'lora_xs':
                    U1_r, Vt1_r = svd_bases[l]['W1']
                    U2_r, Vt2_r = svd_bases[l]['W2']
                    delta1, info1 = fit_lora_xs(dW1_target, U1_r, Vt1_r, r, rng)
                    delta2, info2 = fit_lora_xs(dW2_target, U2_r, Vt2_r, r, rng)
                elif fmt == 'vera':
                    B1, A1 = vera_shared[l]['W1']
                    B2, A2 = vera_shared[l]['W2']
                    delta1, info1 = fit_vera(dW1_target, B1, A1, d, r, rng)
                    delta2, info2 = fit_vera(dW2_target, B2, A2, d_ff, r, rng)

                # Reconstruction error
                err1 = frobenius_ratio(delta1, dW1_target)
                err2 = frobenius_ratio(delta2, dW2_target)
                reconstruction_errors.extend([err1, err2])

                flat_parts.extend([delta1.ravel(), delta2.ravel()])
                expert_total_params += info1['n_params'] + info2['n_params']

            expert_flat_deltas.append(np.concatenate(flat_parts))
            total_params_list.append(expert_total_params)

        fit_time = time.perf_counter() - t0

        # Pairwise cosines
        cosines = measure_pairwise_cosines(expert_flat_deltas)

        # Signal retention = 1 - reconstruction_error
        mean_recon_error = np.mean(reconstruction_errors)
        signal_retention = 1.0 - mean_recon_error

        results[fmt] = {
            'signal_retention': float(signal_retention),
            'mean_reconstruction_error': float(mean_recon_error),
            'std_reconstruction_error': float(np.std(reconstruction_errors)),
            'mean_cos': float(np.mean(cosines)),
            'max_cos': float(np.max(cosines)),
            'median_cos': float(np.median(cosines)),
            'std_cos': float(np.std(cosines)),
            'all_cosines': [float(c) for c in cosines],
            'params_per_expert': int(total_params_list[0]),
            'fit_time_s': float(fit_time),
        }

        print(f"    Signal retention: {signal_retention:.4f} "
              f"(recon error: {mean_recon_error:.4f} +/- {np.std(reconstruction_errors):.4f})")
        print(f"    mean|cos|: {np.mean(cosines):.6f}, "
              f"max|cos|: {np.max(cosines):.6f}, "
              f"median: {np.median(cosines):.6f}")
        print(f"    Params/expert: {total_params_list[0]:,}")
        print(f"    Time: {fit_time:.2f}s")

    # Composition test: sum N experts, measure cross-domain interference
    print(f"\n  Composition test (sum {N_EXPERTS} experts)...")
    for fmt in ['lora', 'lora_xs', 'vera']:
        # Reconstruct all deltas
        all_deltas_W1 = [[] for _ in range(N_LAYERS)]
        all_deltas_W2 = [[] for _ in range(N_LAYERS)]

        for expert_i in range(N_EXPERTS):
            for l in range(N_LAYERS):
                dW1_target, dW2_target = target_deltas[expert_i][l]
                if fmt == 'lora':
                    d1, _ = fit_lora(dW1_target, d, d_ff, r, rng)
                    d2, _ = fit_lora(dW2_target, d_ff, d, r, rng)
                elif fmt == 'lora_xs':
                    U1_r, Vt1_r = svd_bases[l]['W1']
                    U2_r, Vt2_r = svd_bases[l]['W2']
                    d1, _ = fit_lora_xs(dW1_target, U1_r, Vt1_r, r, rng)
                    d2, _ = fit_lora_xs(dW2_target, U2_r, Vt2_r, r, rng)
                elif fmt == 'vera':
                    B1, A1 = vera_shared[l]['W1']
                    B2, A2 = vera_shared[l]['W2']
                    d1, _ = fit_vera(dW1_target, B1, A1, d, r, rng)
                    d2, _ = fit_vera(dW2_target, B2, A2, d_ff, r, rng)
                all_deltas_W1[l].append(d1)
                all_deltas_W2[l].append(d2)

        # Sum all expert deltas (naive additive composition)
        composed_W1 = [sum(all_deltas_W1[l]) for l in range(N_LAYERS)]
        composed_W2 = [sum(all_deltas_W2[l]) for l in range(N_LAYERS)]

        # Target composed: sum of original targets
        target_composed_W1 = [sum(target_deltas[i][l][0] for i in range(N_EXPERTS))
                              for l in range(N_LAYERS)]
        target_composed_W2 = [sum(target_deltas[i][l][1] for i in range(N_EXPERTS))
                              for l in range(N_LAYERS)]

        # Composition fidelity: how well does composed match target composed?
        comp_errors = []
        for l in range(N_LAYERS):
            comp_errors.append(frobenius_ratio(composed_W1[l], target_composed_W1[l]))
            comp_errors.append(frobenius_ratio(composed_W2[l], target_composed_W2[l]))

        mean_comp_error = np.mean(comp_errors)
        results[fmt]['composition_error'] = float(mean_comp_error)
        results[fmt]['composition_fidelity'] = float(1.0 - mean_comp_error)

        print(f"    {fmt}: composition fidelity = {1.0 - mean_comp_error:.4f} "
              f"(error = {mean_comp_error:.4f})")

    # Inference overhead
    print(f"\n  Inference timing...")
    timing = measure_inference_overhead(d, d_ff, rng)
    results['timing'] = timing
    print(f"    LoRA: {timing['lora_us']:.1f} us")
    print(f"    LoRA-XS: {timing['lora_xs_us']:.1f} us (overhead: {timing['xs_overhead']*100:.1f}%)")
    print(f"    VeRA: {timing['vera_us']:.1f} us (overhead: {timing['vera_overhead']*100:.1f}%)")

    return results


def run_full_geometric_analysis():
    """Large-N geometric test: many random experts, pure orthogonality geometry."""
    print("\n" + "=" * 60)
    print("  GEOMETRIC ORTHOGONALITY ANALYSIS (N=50 random experts)")
    print("=" * 60)

    results = {}

    for d in D_VALUES:
        d_ff = D_FF_MULT * d
        r = LORA_RANK
        n_experts = 50

        rng = np.random.RandomState(42)

        # Pretrained weight (for LoRA-XS SVD)
        W = (rng.randn(d, d_ff) * 0.02).astype(DTYPE)
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        U_r = U[:, :r].astype(DTYPE)
        Vt_r = Vt[:r, :].astype(DTYPE)

        # VeRA shared matrices
        B_s = (rng.randn(d_ff, r) * np.sqrt(2.0 / r)).astype(DTYPE)
        A_s = (rng.randn(r, d) * np.sqrt(2.0 / d)).astype(DTYPE)

        lora_deltas = []
        xs_deltas = []
        vera_deltas = []

        for i in range(n_experts):
            ri = np.random.RandomState(i * 7 + 13)

            # LoRA: independent random subspaces
            B_rand = (ri.randn(d_ff, r) * 0.02).astype(DTYPE)
            A_rand = (ri.randn(r, d) * np.sqrt(2.0 / d)).astype(DTYPE)
            lora_deltas.append((B_rand @ A_rand).ravel())

            # LoRA-XS: shared basis, random M
            M_rand = (ri.randn(r, r) * 0.02).astype(DTYPE)
            xs_deltas.append((U_r @ M_rand @ Vt_r).ravel())

            # VeRA: shared B, A, random scaling
            lb_rand = (ri.randn(d_ff) * 0.02).astype(DTYPE)
            ld_rand = (ri.randn(r) * 0.02).astype(DTYPE)
            dW_v = (lb_rand[:, None] * B_s * ld_rand[None, :]) @ A_s
            vera_deltas.append(dW_v.ravel())

        lora_cos = measure_pairwise_cosines(lora_deltas)
        xs_cos = measure_pairwise_cosines(xs_deltas)
        vera_cos = measure_pairwise_cosines(vera_deltas)

        bound = np.sqrt(r / d)

        results[d] = {
            'bound_sqrt_r_d': float(bound),
            'lora': {
                'mean_cos': float(np.mean(lora_cos)),
                'max_cos': float(np.max(lora_cos)),
                'std_cos': float(np.std(lora_cos)),
                'pct_below_001': float(np.mean(lora_cos < 0.01) * 100),
            },
            'lora_xs': {
                'mean_cos': float(np.mean(xs_cos)),
                'max_cos': float(np.max(xs_cos)),
                'std_cos': float(np.std(xs_cos)),
                'pct_below_001': float(np.mean(xs_cos < 0.01) * 100),
                'ratio_vs_lora': float(np.mean(xs_cos) / np.mean(lora_cos)),
            },
            'vera': {
                'mean_cos': float(np.mean(vera_cos)),
                'max_cos': float(np.max(vera_cos)),
                'std_cos': float(np.std(vera_cos)),
                'pct_below_001': float(np.mean(vera_cos < 0.01) * 100),
                'ratio_vs_lora': float(np.mean(vera_cos) / np.mean(lora_cos)),
            },
        }

        print(f"\n  d={d} (bound sqrt(r/d) = {bound:.4f}, N={n_experts}):")
        print(f"    LoRA:    mean|cos|={np.mean(lora_cos):.6f}, "
              f"max={np.max(lora_cos):.6f}, "
              f"<0.01: {np.mean(lora_cos < 0.01)*100:.1f}%")
        print(f"    LoRA-XS: mean|cos|={np.mean(xs_cos):.6f}, "
              f"max={np.max(xs_cos):.6f}, "
              f"<0.01: {np.mean(xs_cos < 0.01)*100:.1f}%, "
              f"ratio: {np.mean(xs_cos)/np.mean(lora_cos):.2f}x")
        print(f"    VeRA:    mean|cos|={np.mean(vera_cos):.6f}, "
              f"max={np.max(vera_cos):.6f}, "
              f"<0.01: {np.mean(vera_cos < 0.01)*100:.1f}%, "
              f"ratio: {np.mean(vera_cos)/np.mean(lora_cos):.2f}x")

    return results


def compute_storage_analysis():
    """Detailed storage comparison for production scales."""
    print("\n" + "=" * 60)
    print("  STORAGE ANALYSIS (Production Scale)")
    print("=" * 60)

    r = 16  # production rank
    configs = [
        ('Qwen2.5-0.5B', 896, 28, 7),
        ('Qwen2.5-7B', 3584, 28, 7),
        ('Qwen2.5-72B', 8192, 80, 7),
    ]

    results = {}

    for name, d, n_layers, n_modules in configs:
        d_ff = 4 * d  # simplified

        lora_per_expert = n_layers * n_modules * 2 * d * r * 2  # float16
        xs_per_expert = n_layers * n_modules * r * r * 2
        vera_per_expert = n_layers * n_modules * (d + r) * 2
        vera_shared = n_layers * n_modules * 2 * d * r * 2  # one copy B, A

        for N in [50, 500, 5000, 100000]:
            lora_total = lora_per_expert * N
            xs_total = xs_per_expert * N
            vera_total = vera_per_expert * N + vera_shared

            key = f"{name}_N{N}"
            results[key] = {
                'lora_total_MB': lora_total / 1e6,
                'xs_total_MB': xs_total / 1e6,
                'vera_total_MB': vera_total / 1e6,
                'xs_compression': lora_total / xs_total if xs_total > 0 else float('inf'),
                'vera_compression': lora_total / vera_total if vera_total > 0 else float('inf'),
            }

        print(f"\n  {name} (d={d}, {n_layers} layers, {n_modules} modules, r={r}):")
        print(f"    Per expert: LoRA={lora_per_expert/1e6:.1f} MB, "
              f"LoRA-XS={xs_per_expert/1e3:.1f} KB, "
              f"VeRA={vera_per_expert/1e3:.1f} KB")
        print(f"    N=5000: LoRA={lora_per_expert*5000/1e9:.1f} GB, "
              f"LoRA-XS={xs_per_expert*5000/1e6:.1f} MB, "
              f"VeRA={vera_per_expert*5000/1e6:.1f} MB (+{vera_shared/1e6:.1f} MB shared)")
        print(f"    Compression: LoRA-XS={lora_per_expert*5000/(xs_per_expert*5000):.0f}x, "
              f"VeRA={lora_per_expert*5000/(vera_per_expert*5000+vera_shared):.0f}x")

    return results


def evaluate_kill_criteria(all_results):
    """Evaluate all three kill criteria."""
    print("\n" + "=" * 60)
    print("  KILL CRITERIA EVALUATION")
    print("=" * 60)

    k1_data = []  # (fmt, d, seed, retention_ratio)
    k2_data = []  # (fmt, d, seed, mean_cos)
    k3_data = []  # (fmt, overhead)

    for d in D_VALUES:
        for seed in SEEDS:
            key = f"d{d}_s{seed}"
            if key not in all_results:
                continue
            r = all_results[key]
            lora_ret = r['lora']['signal_retention']
            for fmt in ['lora_xs', 'vera']:
                fmt_ret = r[fmt]['signal_retention']
                ratio = fmt_ret / lora_ret if abs(lora_ret) > 1e-8 else 0
                k1_data.append((fmt, d, seed, ratio, fmt_ret, lora_ret))
                k2_data.append((fmt, d, seed, r[fmt]['mean_cos'], r[fmt]['max_cos']))

            # Timing (from the last d tested)
            if 'timing' in r:
                k3_data.append(('lora_xs', d, r['timing']['xs_overhead']))
                k3_data.append(('vera', d, r['timing']['vera_overhead']))

    # K1: signal retention
    print("\n  K1: Signal retention (compressed/LoRA, kill if <0.50)")
    k1_killed = False
    for fmt, d, seed, ratio, fmt_ret, lora_ret in k1_data:
        status = "KILL" if ratio < 0.50 else "SURVIVES"
        if ratio < 0.50:
            k1_killed = True
        print(f"    {fmt:8s} d={d:4d} seed={seed}: "
              f"retention={fmt_ret:.4f} / LoRA={lora_ret:.4f} = {ratio:.3f} ({status})")

    # K2: orthogonality
    print("\n  K2: Orthogonality (kill if mean|cos| > 0.01)")
    k2_killed = False
    for fmt, d, seed, mean_cos, max_cos in k2_data:
        status = "KILL" if mean_cos > 0.01 else "SURVIVES"
        if mean_cos > 0.01:
            k2_killed = True
        print(f"    {fmt:8s} d={d:4d} seed={seed}: "
              f"mean|cos|={mean_cos:.6f}, max={max_cos:.6f} ({status})")

    # K3: inference overhead
    print("\n  K3: Inference overhead (kill if >10%)")
    k3_killed = False
    for fmt, d, overhead in k3_data:
        status = "KILL" if overhead > 0.10 else "SURVIVES"
        if overhead > 0.10:
            k3_killed = True
        print(f"    {fmt:8s} d={d:4d}: overhead={overhead*100:.1f}% ({status})")

    verdicts = {
        'K1_signal_retention': 'KILLED' if k1_killed else 'SURVIVES',
        'K2_orthogonality': 'KILLED' if k2_killed else 'SURVIVES',
        'K3_inference_overhead': 'KILLED' if k3_killed else 'SURVIVES',
    }

    print(f"\n  VERDICTS:")
    for k, v in verdicts.items():
        print(f"    {k}: {v}")

    return verdicts


def main():
    print("=" * 60)
    print("  COMPRESSED EXPERT SWEEP")
    print("  LoRA vs LoRA-XS vs VeRA for SOLE composition")
    print("=" * 60)

    all_results = {}
    total_t0 = time.perf_counter()

    # 1. Geometric orthogonality analysis (pure geometry, no training)
    geo = run_full_geometric_analysis()
    all_results['geometric'] = geo

    # 2. Storage analysis
    storage = compute_storage_analysis()
    all_results['storage'] = storage

    # 3. Signal retention + orthogonality + composition + timing
    for d in D_VALUES:
        for seed in SEEDS:
            key = f"d{d}_s{seed}"
            print(f"\n\nExperiment: {key}")
            result = run_experiment(d, seed)
            all_results[key] = result

    total_time = time.perf_counter() - total_t0
    print(f"\n\nTotal experiment time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # 4. Kill criteria
    verdicts = evaluate_kill_criteria(all_results)
    all_results['verdicts'] = verdicts
    all_results['total_time_s'] = total_time

    # Save
    output_path = RESULTS_DIR / 'results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    return all_results


if __name__ == '__main__':
    main()
