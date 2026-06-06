"""Layer-wise Merge Order Sensitivity Experiment.

Tests whether attention-layer merge order sensitivity exceeds FFN at high
domain overlap, and whether the flattened-vector analysis in the parent
experiment (merge_order_dependence) masks layer-specific effects.

Design:
  Phase 1 (Synthetic, controlled): Create multi-component synthetic experts
    where different "layer groups" have different pairwise cosine similarities
    (simulating attn cos=0.85 vs FFN cos=0.59 from ffn_only_vs_all_modules).
    Run GS on flattened vectors AND per-layer-group, compare order sensitivity.

  Phase 2 (Sweep): Systematically vary per-group cosine from 0.01 to 0.90
    to map the full order-sensitivity curve per layer type and compare
    against the parent experiment's flattened variation ~ 80*cos.

  Phase 3 (Cross-validation): Verify that layer-wise GS produces different
    merged vectors than flattened GS when layer cosines diverge.

Kill Criteria (from HYPOTHESES.yml):
  K1: Attention-layer order CV < 1% even at cos=0.85 (within-cluster)
  K2: FFN and attention layers show identical order sensitivity scaling
"""

import random
import statistics
import time
import json
import numpy as np
from pathlib import Path

# Reuse GS infrastructure from parent experiment
from experiments.models.gram_schmidt_composition.gram_schmidt import (
    gram_schmidt_orthogonalize,
    cosine_sim,
    flatten_delta_dict,
    unflatten_delta_dict,
)

# ── Config ──────────────────────────────────────────────────────────────────

N_ORDERINGS = 30  # Random orderings per condition
N_EXPERTS = 8     # Number of synthetic experts

# Simulate a 4-layer transformer with 4 attention sublayers + 2 FFN sublayers per layer
# Total: 4*(4+2) = 24 sublayer components
# This mirrors real architectures: attn has wq, wk, wv, wo; FFN has fc1, fc2
N_LAYERS = 4
ATTN_SUBLAYERS = ['wq', 'wk', 'wv', 'wo']  # 4 per layer
FFN_SUBLAYERS = ['fc1', 'fc2']               # 2 per layer
DIM_PER_SUBLAYER = 256  # Each sublayer delta is a 256-dim vector

# Production-relevant cosine pairs (from ffn_only_vs_all_modules)
ATTN_COS_PRODUCTION = 0.85   # math-medical pair, attention layers
FFN_COS_PRODUCTION = 0.59    # math-medical pair, FFN layers


# ── Helpers ─────────────────────────────────────────────────────────────────

def create_layered_synthetic_experts(
    n_experts: int,
    n_layers: int,
    attn_sublayers: list[str],
    ffn_sublayers: list[str],
    dim_per_sublayer: int,
    attn_cosine: float,
    ffn_cosine: float,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """Create synthetic expert deltas with controlled per-layer-type cosine.

    Each expert is a dict mapping (layer_idx, sublayer_name) -> np.ndarray.
    Attention sublayers share a common direction with strength sqrt(attn_cosine),
    FFN sublayers share a different common direction with strength sqrt(ffn_cosine).

    Returns:
        experts: list of N delta dicts
        info: dict with actual measured cosines per layer type
    """
    rng = np.random.RandomState(seed)

    # Create shared directions per sublayer
    # Each sublayer gets its own shared direction (realistic: different weight matrices
    # have different shared components)
    shared_directions = {}
    for l in range(n_layers):
        for sub in attn_sublayers + ffn_sublayers:
            d = rng.randn(dim_per_sublayer)
            shared_directions[(l, sub)] = d / np.linalg.norm(d)

    experts = []
    for k in range(n_experts):
        delta = {}
        for l in range(n_layers):
            for sub in attn_sublayers:
                cos_target = attn_cosine
                alpha = np.sqrt(max(cos_target, 0.0))
                beta = np.sqrt(max(1.0 - cos_target, 0.0))

                shared = shared_directions[(l, sub)]
                unique = rng.randn(dim_per_sublayer)
                unique = unique / np.linalg.norm(unique)
                unique = unique - np.dot(unique, shared) * shared
                unique = unique / np.linalg.norm(unique)

                delta[(l, sub)] = alpha * shared + beta * unique

            for sub in ffn_sublayers:
                cos_target = ffn_cosine
                alpha = np.sqrt(max(cos_target, 0.0))
                beta = np.sqrt(max(1.0 - cos_target, 0.0))

                shared = shared_directions[(l, sub)]
                unique = rng.randn(dim_per_sublayer)
                unique = unique / np.linalg.norm(unique)
                unique = unique - np.dot(unique, shared) * shared
                unique = unique / np.linalg.norm(unique)

                delta[(l, sub)] = alpha * shared + beta * unique

        experts.append(delta)

    # Measure actual cosines per layer type
    attn_cosines = []
    ffn_cosines = []
    all_cosines = []

    for i in range(n_experts):
        for j in range(i + 1, n_experts):
            # Per sublayer cosines
            for l in range(n_layers):
                for sub in attn_sublayers:
                    c = cosine_sim(experts[i][(l, sub)], experts[j][(l, sub)])
                    attn_cosines.append(c)
                for sub in ffn_sublayers:
                    c = cosine_sim(experts[i][(l, sub)], experts[j][(l, sub)])
                    ffn_cosines.append(c)

            # Flattened cosine
            flat_i = np.concatenate([experts[i][k] for k in sorted(experts[i].keys())])
            flat_j = np.concatenate([experts[j][k] for k in sorted(experts[j].keys())])
            all_cosines.append(cosine_sim(flat_i, flat_j))

    info = {
        'attn_cos_target': attn_cosine,
        'ffn_cos_target': ffn_cosine,
        'attn_cos_actual': float(np.mean(attn_cosines)),
        'ffn_cos_actual': float(np.mean(ffn_cosines)),
        'flat_cos_actual': float(np.mean(all_cosines)),
        'n_attn_sublayers': len(attn_sublayers) * n_layers,
        'n_ffn_sublayers': len(ffn_sublayers) * n_layers,
    }

    return experts, info


def flatten_by_type(expert_delta: dict, layer_type: str,
                    n_layers: int, attn_sublayers: list, ffn_sublayers: list) -> np.ndarray:
    """Flatten only attention or FFN sublayers of an expert delta."""
    sublayers = attn_sublayers if layer_type == 'attn' else ffn_sublayers
    parts = []
    for l in range(n_layers):
        for sub in sublayers:
            parts.append(expert_delta[(l, sub)])
    return np.concatenate(parts)


def flatten_full(expert_delta: dict) -> np.ndarray:
    """Flatten all sublayers in sorted key order."""
    return np.concatenate([expert_delta[k] for k in sorted(expert_delta.keys())])


def unflatten_full(flat: np.ndarray, template: dict) -> dict:
    """Reconstruct delta dict from flat vector."""
    result = {}
    offset = 0
    for k in sorted(template.keys()):
        size = len(template[k])
        result[k] = flat[offset:offset + size]
        offset += size
    return result


def gs_orthogonalize_vectors(vectors: list[np.ndarray]) -> list[np.ndarray]:
    """Apply Gram-Schmidt to a list of numpy vectors."""
    ortho = []
    for k in range(len(vectors)):
        v = vectors[k].copy()
        for i in range(len(ortho)):
            e_i = ortho[i]
            dot_ve = np.dot(v, e_i)
            dot_ee = np.dot(e_i, e_i)
            if dot_ee > 1e-12:
                v = v - (dot_ve / dot_ee) * e_i
        ortho.append(v)
    return ortho


def measure_order_sensitivity(vectors: list[np.ndarray], n_orderings: int,
                               seed: int = 42) -> dict:
    """Measure order sensitivity of GS on a set of vectors.

    Returns:
        dict with cv_pct, worst_vs_best_pct, merged_cos_min, norm_cv
    """
    N = len(vectors)
    rng = random.Random(seed)

    merged_vectors = []
    for trial in range(n_orderings):
        ordering = list(range(N))
        rng.shuffle(ordering)

        ordered = [vectors[ordering[i]] for i in range(N)]
        ortho = gs_orthogonalize_vectors(ordered)

        # Average merge
        merged = np.mean(ortho, axis=0)
        merged_vectors.append(merged)

    # Measure variation
    norms = [np.linalg.norm(v) for v in merged_vectors]
    norm_cv = (statistics.stdev(norms) / statistics.mean(norms)) * 100 if len(norms) > 1 else 0

    # Pairwise cosines between merged vectors
    merged_cosines = []
    for i in range(len(merged_vectors)):
        for j in range(i + 1, len(merged_vectors)):
            merged_cosines.append(cosine_sim(merged_vectors[i], merged_vectors[j]))

    merged_cos_min = min(merged_cosines) if merged_cosines else 1.0
    variation_pct = (1.0 - merged_cos_min) * 100

    # Norm-based CV as quality proxy
    return {
        'norm_cv_pct': norm_cv,
        'merged_cos_min': float(merged_cos_min),
        'variation_pct': variation_pct,
    }


# ── Phase 1: Production-Relevant Layer-Wise Analysis ────────────────────────

def run_phase1(seed=42):
    """Test with production-relevant cosine values (attn=0.85, FFN=0.59).

    This is the core experiment: does layer-wise analysis reveal that
    attention layers have higher order sensitivity than FFN layers, as
    predicted by the variation ~ 80*cos relationship?
    """
    print(f"\n{'='*70}")
    print(f"PHASE 1: PRODUCTION-RELEVANT LAYER-WISE ANALYSIS")
    print(f"  attn_cos={ATTN_COS_PRODUCTION}, ffn_cos={FFN_COS_PRODUCTION}")
    print(f"  N={N_EXPERTS}, D_per_sublayer={DIM_PER_SUBLAYER}")
    print(f"{'='*70}")

    t0 = time.time()

    experts, info = create_layered_synthetic_experts(
        n_experts=N_EXPERTS,
        n_layers=N_LAYERS,
        attn_sublayers=ATTN_SUBLAYERS,
        ffn_sublayers=FFN_SUBLAYERS,
        dim_per_sublayer=DIM_PER_SUBLAYER,
        attn_cosine=ATTN_COS_PRODUCTION,
        ffn_cosine=FFN_COS_PRODUCTION,
        seed=seed,
    )

    print(f"\n  Actual cosines: attn={info['attn_cos_actual']:.4f}, "
          f"ffn={info['ffn_cos_actual']:.4f}, flat={info['flat_cos_actual']:.4f}")

    # --- Analysis 1: Flattened (replicating parent experiment) ---
    flat_vectors = [flatten_full(e) for e in experts]
    flat_result = measure_order_sensitivity(flat_vectors, N_ORDERINGS, seed)
    print(f"\n  FLATTENED (all layers combined):")
    print(f"    Merged cos min: {flat_result['merged_cos_min']:.6f}")
    print(f"    Variation%:     {flat_result['variation_pct']:.4f}")
    print(f"    Norm CV%:       {flat_result['norm_cv_pct']:.4f}")

    # --- Analysis 2: Attention layers only ---
    attn_vectors = [flatten_by_type(e, 'attn', N_LAYERS, ATTN_SUBLAYERS, FFN_SUBLAYERS)
                    for e in experts]
    attn_result = measure_order_sensitivity(attn_vectors, N_ORDERINGS, seed)
    print(f"\n  ATTENTION LAYERS ONLY:")
    print(f"    Merged cos min: {attn_result['merged_cos_min']:.6f}")
    print(f"    Variation%:     {attn_result['variation_pct']:.4f}")
    print(f"    Norm CV%:       {attn_result['norm_cv_pct']:.4f}")

    # --- Analysis 3: FFN layers only ---
    ffn_vectors = [flatten_by_type(e, 'ffn', N_LAYERS, ATTN_SUBLAYERS, FFN_SUBLAYERS)
                   for e in experts]
    ffn_result = measure_order_sensitivity(ffn_vectors, N_ORDERINGS, seed)
    print(f"\n  FFN LAYERS ONLY:")
    print(f"    Merged cos min: {ffn_result['merged_cos_min']:.6f}")
    print(f"    Variation%:     {ffn_result['variation_pct']:.4f}")
    print(f"    Norm CV%:       {ffn_result['norm_cv_pct']:.4f}")

    # --- Analysis 4: Per-individual-sublayer ---
    sublayer_results = {}
    for l in range(N_LAYERS):
        for sub in ATTN_SUBLAYERS + FFN_SUBLAYERS:
            vecs = [e[(l, sub)] for e in experts]
            result = measure_order_sensitivity(vecs, N_ORDERINGS, seed)
            key = f"L{l}_{sub}"
            sublayer_results[key] = result

    print(f"\n  PER-SUBLAYER VARIATION%:")
    print(f"  {'Sublayer':>12} {'Type':>6} {'Variation%':>12} {'Cos min':>12}")
    print(f"  {'-'*48}")
    attn_variations = []
    ffn_variations = []
    for l in range(N_LAYERS):
        for sub in ATTN_SUBLAYERS:
            key = f"L{l}_{sub}"
            v = sublayer_results[key]['variation_pct']
            attn_variations.append(v)
            print(f"  {key:>12} {'attn':>6} {v:>12.4f} {sublayer_results[key]['merged_cos_min']:>12.6f}")
        for sub in FFN_SUBLAYERS:
            key = f"L{l}_{sub}"
            v = sublayer_results[key]['variation_pct']
            ffn_variations.append(v)
            print(f"  {key:>12} {'ffn':>6} {v:>12.4f} {sublayer_results[key]['merged_cos_min']:>12.6f}")

    mean_attn_var = statistics.mean(attn_variations)
    mean_ffn_var = statistics.mean(ffn_variations)
    ratio = mean_attn_var / mean_ffn_var if mean_ffn_var > 1e-12 else float('inf')

    print(f"\n  SUMMARY:")
    print(f"    Mean attn variation%:  {mean_attn_var:.4f}")
    print(f"    Mean FFN variation%:   {mean_ffn_var:.4f}")
    print(f"    Attn/FFN ratio:        {ratio:.2f}x")
    print(f"    Predicted (80*cos):    attn={80*ATTN_COS_PRODUCTION:.1f}%, "
          f"ffn={80*FFN_COS_PRODUCTION:.1f}%")

    # Kill criteria check
    attn_cv = attn_result['variation_pct']  # Using variation% as proxy for CV
    k1_killed = attn_cv < 1.0  # K1: attn CV < 1% => KILLED
    k2_killed = abs(ratio - 1.0) < 0.05  # K2: ratio within 5% of 1 => identical scaling

    print(f"\n  KILL CRITERIA:")
    print(f"    K1 (attn CV < 1%): attn variation={attn_cv:.4f}% "
          f"-> {'KILLED (too low)' if k1_killed else 'PASS (sensitivity confirmed)'}")
    print(f"    K2 (identical scaling): ratio={ratio:.2f}x "
          f"-> {'KILLED (identical)' if k2_killed else 'PASS (different scaling)'}")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    return {
        'seed': seed,
        'info': info,
        'flat_result': flat_result,
        'attn_result': attn_result,
        'ffn_result': ffn_result,
        'sublayer_results': sublayer_results,
        'mean_attn_variation': mean_attn_var,
        'mean_ffn_variation': mean_ffn_var,
        'attn_ffn_ratio': ratio,
        'k1_killed': k1_killed,
        'k2_killed': k2_killed,
        'elapsed': elapsed,
    }


# ── Phase 2: Cosine Sweep for Layer-Wise Scaling Law ────────────────────────

def run_phase2(seed=42):
    """Sweep per-layer-type cosine to map the order-sensitivity curve.

    Tests whether both attention and FFN layers follow variation ~ 80*cos,
    or if the scaling constant differs by layer type.
    """
    print(f"\n{'='*70}")
    print(f"PHASE 2: COSINE SWEEP -- LAYER-WISE SCALING LAW")
    print(f"{'='*70}")

    t0 = time.time()

    cosine_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.9]
    results = []

    for cos_val in cosine_values:
        # Test each cosine value applied uniformly to one layer type
        # while keeping the other at a low baseline (0.01)
        for test_type in ['attn', 'ffn']:
            attn_cos = cos_val if test_type == 'attn' else 0.01
            ffn_cos = cos_val if test_type == 'ffn' else 0.01

            experts, info = create_layered_synthetic_experts(
                n_experts=N_EXPERTS,
                n_layers=N_LAYERS,
                attn_sublayers=ATTN_SUBLAYERS,
                ffn_sublayers=FFN_SUBLAYERS,
                dim_per_sublayer=DIM_PER_SUBLAYER,
                attn_cosine=attn_cos,
                ffn_cosine=ffn_cos,
                seed=seed,
            )

            # Measure the high-cosine layer group
            vectors = [flatten_by_type(e, test_type, N_LAYERS, ATTN_SUBLAYERS, FFN_SUBLAYERS)
                       for e in experts]
            result = measure_order_sensitivity(vectors, N_ORDERINGS, seed)

            # Also measure flattened
            flat_vectors = [flatten_full(e) for e in experts]
            flat_result = measure_order_sensitivity(flat_vectors, N_ORDERINGS, seed)

            results.append({
                'cos_val': cos_val,
                'layer_type': test_type,
                'actual_cos': info[f'{test_type}_cos_actual'],
                'layer_variation': result['variation_pct'],
                'layer_norm_cv': result['norm_cv_pct'],
                'flat_variation': flat_result['variation_pct'],
                'flat_norm_cv': flat_result['norm_cv_pct'],
            })

    # Summary table
    print(f"\n  {'Target cos':>12} {'Type':>6} {'Actual cos':>12} "
          f"{'Layer var%':>12} {'Flat var%':>12} {'Predicted':>12}")
    print(f"  {'-'*72}")
    for r in results:
        pred = 80 * r['actual_cos']
        print(f"  {r['cos_val']:>12.2f} {r['layer_type']:>6} {r['actual_cos']:>12.4f} "
              f"{r['layer_variation']:>12.4f} {r['flat_variation']:>12.4f} {pred:>12.1f}")

    # Fit linear model: variation = slope * cos for each layer type
    attn_points = [(r['actual_cos'], r['layer_variation']) for r in results if r['layer_type'] == 'attn']
    ffn_points = [(r['actual_cos'], r['layer_variation']) for r in results if r['layer_type'] == 'ffn']

    def fit_slope(points):
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        # Least squares: y = slope * x (no intercept)
        slope = np.dot(x, y) / np.dot(x, x)
        # R^2
        y_pred = slope * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return slope, r2

    attn_slope, attn_r2 = fit_slope(attn_points)
    ffn_slope, ffn_r2 = fit_slope(ffn_points)

    print(f"\n  LINEAR FIT (variation = slope * cos):")
    print(f"    Attention: slope={attn_slope:.1f}, R2={attn_r2:.4f}")
    print(f"    FFN:       slope={ffn_slope:.1f}, R2={ffn_r2:.4f}")
    print(f"    Parent experiment (flattened): slope~80")
    print(f"    Slope ratio (attn/ffn): {attn_slope/ffn_slope:.2f}x" if ffn_slope > 0 else "")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    return {
        'results': results,
        'attn_slope': attn_slope,
        'attn_r2': attn_r2,
        'ffn_slope': ffn_slope,
        'ffn_r2': ffn_r2,
        'elapsed': elapsed,
    }


# ── Phase 3: Flattened vs Layer-Wise GS Divergence ──────────────────────────

def run_phase3(seed=42):
    """Test whether layer-wise GS produces different results than flattened GS.

    If attention layers have high cosine and FFN layers have low cosine,
    flattened GS applies a uniform orthogonalization that may over-correct
    FFN (removing signal from near-orthogonal layers) or under-correct
    attention (leaving interference in high-overlap layers).

    Layer-wise GS applies GS separately per sublayer, preserving more FFN
    signal while more aggressively orthogonalizing attention layers.
    """
    print(f"\n{'='*70}")
    print(f"PHASE 3: FLATTENED VS LAYER-WISE GS DIVERGENCE")
    print(f"{'='*70}")

    t0 = time.time()

    experts, info = create_layered_synthetic_experts(
        n_experts=N_EXPERTS,
        n_layers=N_LAYERS,
        attn_sublayers=ATTN_SUBLAYERS,
        ffn_sublayers=FFN_SUBLAYERS,
        dim_per_sublayer=DIM_PER_SUBLAYER,
        attn_cosine=ATTN_COS_PRODUCTION,
        ffn_cosine=FFN_COS_PRODUCTION,
        seed=seed,
    )

    print(f"\n  Actual cosines: attn={info['attn_cos_actual']:.4f}, "
          f"ffn={info['ffn_cos_actual']:.4f}")

    # Method 1: Flattened GS (parent experiment approach)
    flat_vectors = [flatten_full(e) for e in experts]
    flat_ortho = gs_orthogonalize_vectors(flat_vectors)

    # Signal retention per expert (flattened)
    flat_retentions = []
    for k in range(N_EXPERTS):
        orig_norm = np.linalg.norm(flat_vectors[k])
        ortho_norm = np.linalg.norm(flat_ortho[k])
        flat_retentions.append(ortho_norm / orig_norm if orig_norm > 1e-12 else 0)

    # Method 2: Layer-wise GS (per sublayer independently)
    layerwise_ortho = [{} for _ in range(N_EXPERTS)]
    layerwise_retentions_attn = []
    layerwise_retentions_ffn = []

    for l in range(N_LAYERS):
        for sub in ATTN_SUBLAYERS + FFN_SUBLAYERS:
            vecs = [e[(l, sub)] for e in experts]
            ortho_vecs = gs_orthogonalize_vectors(vecs)
            for k in range(N_EXPERTS):
                layerwise_ortho[k][(l, sub)] = ortho_vecs[k]

    # Per-expert retention by layer type (layerwise method)
    for k in range(N_EXPERTS):
        attn_orig_norm2 = 0
        attn_ortho_norm2 = 0
        ffn_orig_norm2 = 0
        ffn_ortho_norm2 = 0
        for l in range(N_LAYERS):
            for sub in ATTN_SUBLAYERS:
                attn_orig_norm2 += np.dot(experts[k][(l, sub)], experts[k][(l, sub)])
                attn_ortho_norm2 += np.dot(layerwise_ortho[k][(l, sub)], layerwise_ortho[k][(l, sub)])
            for sub in FFN_SUBLAYERS:
                ffn_orig_norm2 += np.dot(experts[k][(l, sub)], experts[k][(l, sub)])
                ffn_ortho_norm2 += np.dot(layerwise_ortho[k][(l, sub)], layerwise_ortho[k][(l, sub)])
        layerwise_retentions_attn.append(np.sqrt(attn_ortho_norm2 / attn_orig_norm2))
        layerwise_retentions_ffn.append(np.sqrt(ffn_ortho_norm2 / ffn_orig_norm2))

    # Decompose flattened GS retention into per-layer-type contributions
    flat_retentions_attn = []
    flat_retentions_ffn = []
    for k in range(N_EXPERTS):
        # Extract attn/ffn components from flattened ortho vector
        ortho_dict = unflatten_full(flat_ortho[k], experts[0])
        attn_orig2 = 0
        attn_ortho2 = 0
        ffn_orig2 = 0
        ffn_ortho2 = 0
        for l in range(N_LAYERS):
            for sub in ATTN_SUBLAYERS:
                attn_orig2 += np.dot(experts[k][(l, sub)], experts[k][(l, sub)])
                attn_ortho2 += np.dot(ortho_dict[(l, sub)], ortho_dict[(l, sub)])
            for sub in FFN_SUBLAYERS:
                ffn_orig2 += np.dot(experts[k][(l, sub)], experts[k][(l, sub)])
                ffn_ortho2 += np.dot(ortho_dict[(l, sub)], ortho_dict[(l, sub)])
        flat_retentions_attn.append(np.sqrt(attn_ortho2 / attn_orig2))
        flat_retentions_ffn.append(np.sqrt(ffn_ortho2 / ffn_orig2))

    print(f"\n  FLATTENED GS:")
    print(f"    Overall retention (mean): {statistics.mean(flat_retentions):.4f}")
    print(f"    Attn retention (mean):    {statistics.mean(flat_retentions_attn):.4f}")
    print(f"    FFN retention (mean):     {statistics.mean(flat_retentions_ffn):.4f}")
    print(f"    Overall retention (min):  {min(flat_retentions):.4f}")

    print(f"\n  LAYER-WISE GS:")
    print(f"    Attn retention (mean):    {statistics.mean(layerwise_retentions_attn):.4f}")
    print(f"    FFN retention (mean):     {statistics.mean(layerwise_retentions_ffn):.4f}")
    print(f"    Attn retention (min):     {min(layerwise_retentions_attn):.4f}")
    print(f"    FFN retention (min):      {min(layerwise_retentions_ffn):.4f}")

    # Compare merged vectors from both methods
    flat_merged = np.mean(flat_ortho, axis=0)
    layerwise_merged_parts = []
    for k in sorted(experts[0].keys()):
        vecs_k = [layerwise_ortho[e][k] for e in range(N_EXPERTS)]
        layerwise_merged_parts.append(np.mean(vecs_k, axis=0))
    layerwise_merged = np.concatenate(layerwise_merged_parts)

    divergence = cosine_sim(flat_merged, layerwise_merged)
    l2_diff = np.linalg.norm(flat_merged - layerwise_merged)
    l2_ref = np.linalg.norm(flat_merged)
    relative_diff = l2_diff / l2_ref * 100

    print(f"\n  DIVERGENCE (flat vs layerwise):")
    print(f"    Cosine similarity:     {divergence:.6f}")
    print(f"    Relative L2 diff:      {relative_diff:.4f}%")

    # Order sensitivity comparison
    # Flattened: single ordering dimension
    flat_sensitivity = measure_order_sensitivity(flat_vectors, N_ORDERINGS, seed)

    # Layer-wise: merge per-sublayer then concatenate
    def layerwise_gs_with_ordering(experts, ordering):
        """Apply per-sublayer GS with given ordering, return merged vector."""
        merged_parts = []
        for k in sorted(experts[0].keys()):
            vecs = [experts[ordering[i]][k] for i in range(len(ordering))]
            ortho = gs_orthogonalize_vectors(vecs)
            merged_parts.append(np.mean(ortho, axis=0))
        return np.concatenate(merged_parts)

    rng = random.Random(seed)
    layerwise_merged_vectors = []
    for trial in range(N_ORDERINGS):
        ordering = list(range(N_EXPERTS))
        rng.shuffle(ordering)
        merged = layerwise_gs_with_ordering(experts, ordering)
        layerwise_merged_vectors.append(merged)

    lw_cosines = []
    for i in range(len(layerwise_merged_vectors)):
        for j in range(i + 1, len(layerwise_merged_vectors)):
            lw_cosines.append(cosine_sim(layerwise_merged_vectors[i], layerwise_merged_vectors[j]))

    lw_cos_min = min(lw_cosines)
    lw_variation = (1.0 - lw_cos_min) * 100

    print(f"\n  ORDER SENSITIVITY:")
    print(f"    Flattened:  variation={flat_sensitivity['variation_pct']:.4f}%")
    print(f"    Layer-wise: variation={lw_variation:.4f}%")
    print(f"    Ratio (lw/flat): {lw_variation/flat_sensitivity['variation_pct']:.2f}x"
          if flat_sensitivity['variation_pct'] > 0 else "")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    return {
        'flat_retentions': flat_retentions,
        'flat_retentions_attn': flat_retentions_attn,
        'flat_retentions_ffn': flat_retentions_ffn,
        'layerwise_retentions_attn': layerwise_retentions_attn,
        'layerwise_retentions_ffn': layerwise_retentions_ffn,
        'divergence_cos': divergence,
        'relative_l2_diff': relative_diff,
        'flat_variation': flat_sensitivity['variation_pct'],
        'layerwise_variation': lw_variation,
        'elapsed': elapsed,
    }


# ── Multi-seed aggregation ──────────────────────────────────────────────────

def run_full_experiment():
    """Run the complete layer-wise order sensitivity experiment."""
    all_results = {}

    seeds = [42, 7, 123]

    # Phase 1: Production-relevant analysis (multi-seed)
    print("\n" + "=" * 70)
    print("PHASE 1: PRODUCTION-RELEVANT LAYER-WISE ANALYSIS (multi-seed)")
    print("=" * 70)

    phase1_results = []
    for seed in seeds:
        r = run_phase1(seed=seed)
        phase1_results.append(r)
        all_results[f'phase1_seed{seed}'] = r

    # Aggregate Phase 1 across seeds
    mean_attn_var = statistics.mean([r['mean_attn_variation'] for r in phase1_results])
    mean_ffn_var = statistics.mean([r['mean_ffn_variation'] for r in phase1_results])
    mean_ratio = statistics.mean([r['attn_ffn_ratio'] for r in phase1_results])
    std_ratio = statistics.stdev([r['attn_ffn_ratio'] for r in phase1_results]) if len(phase1_results) > 1 else 0

    print(f"\n{'='*70}")
    print(f"PHASE 1 AGGREGATE ({len(seeds)} seeds)")
    print(f"{'='*70}")
    print(f"  Mean attn variation%:  {mean_attn_var:.4f}")
    print(f"  Mean FFN variation%:   {mean_ffn_var:.4f}")
    print(f"  Attn/FFN ratio:        {mean_ratio:.2f} +/- {std_ratio:.2f}")

    # Phase 2: Cosine sweep
    phase2 = run_phase2(seed=42)
    all_results['phase2'] = phase2

    # Phase 3: Divergence analysis
    phase3 = run_phase3(seed=42)
    all_results['phase3'] = phase3

    # ── Final Verdict ──
    print(f"\n\n{'='*70}")
    print(f"FINAL VERDICT")
    print(f"{'='*70}")

    # K1: attention-layer order CV < 1% even at cos=0.85
    attn_variations = [r['mean_attn_variation'] for r in phase1_results]
    max_attn_var = max(attn_variations)
    k1_killed = max_attn_var < 1.0

    # K2: FFN and attention show identical scaling
    k2_killed = abs(mean_ratio - 1.0) < 0.10  # Within 10% = "identical"

    print(f"\n  K1 (attn CV < 1% at cos=0.85):")
    print(f"    Max attn variation across seeds: {max_attn_var:.4f}%")
    print(f"    Verdict: {'KILLED -- attention insensitive to order' if k1_killed else 'PASS -- attention shows order sensitivity'}")

    print(f"\n  K2 (identical scaling):")
    print(f"    Attn/FFN ratio: {mean_ratio:.2f} +/- {std_ratio:.2f}")
    print(f"    Verdict: {'KILLED -- scaling is identical' if k2_killed else 'PASS -- different scaling confirmed'}")

    print(f"\n  Phase 2 scaling laws:")
    print(f"    Attention slope: {phase2['attn_slope']:.1f} (R2={phase2['attn_r2']:.4f})")
    print(f"    FFN slope:       {phase2['ffn_slope']:.1f} (R2={phase2['ffn_r2']:.4f})")
    print(f"    Parent (flat):   ~80")

    print(f"\n  Phase 3 method divergence:")
    print(f"    Flat vs layerwise cos: {phase3['divergence_cos']:.6f}")
    print(f"    Flat variation%:       {phase3['flat_variation']:.4f}")
    print(f"    Layerwise variation%:  {phase3['layerwise_variation']:.4f}")

    overall = "KILLED" if (k1_killed or k2_killed) else "PROVEN"
    print(f"\n  OVERALL: {overall}")

    if not k1_killed and not k2_killed:
        print(f"\n  IMPLICATION: Attention layers ARE more order-sensitive than FFN.")
        print(f"  Layer-wise composition strategies (GS for attn, simple sum for FFN)")
        print(f"  are warranted when domain overlap is high (cos > 0.06).")
    else:
        reason = []
        if k1_killed:
            reason.append(f"K1: attn variation {max_attn_var:.4f}% < 1%")
        if k2_killed:
            reason.append(f"K2: ratio {mean_ratio:.2f} too close to 1.0")
        print(f"\n  Reasons: {'; '.join(reason)}")

    # Save results
    output_path = Path(__file__).parent / 'results.json'
    serializable = {
        'phase1_aggregate': {
            'mean_attn_variation': mean_attn_var,
            'mean_ffn_variation': mean_ffn_var,
            'attn_ffn_ratio': mean_ratio,
            'ratio_std': std_ratio,
            'seeds': seeds,
        },
        'phase2': {
            'attn_slope': phase2['attn_slope'],
            'attn_r2': phase2['attn_r2'],
            'ffn_slope': phase2['ffn_slope'],
            'ffn_r2': phase2['ffn_r2'],
            'sweep_results': phase2['results'],
        },
        'phase3': {
            'divergence_cos': phase3['divergence_cos'],
            'relative_l2_diff': phase3['relative_l2_diff'],
            'flat_variation': phase3['flat_variation'],
            'layerwise_variation': phase3['layerwise_variation'],
        },
        'k1_killed': k1_killed,
        'k2_killed': k2_killed,
        'overall': overall,
    }
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    return all_results


if __name__ == "__main__":
    results = run_full_experiment()
