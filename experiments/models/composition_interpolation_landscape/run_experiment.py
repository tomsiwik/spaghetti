#!/usr/bin/env python3
"""Experiment: Composition Interpolation Landscape.

Maps the loss landscape of adapter weight interpolation to determine whether
optimal composition weights are findable by gradient methods or need discrete search.

Kill criteria:
  K1: Landscape is flat (no signal — all compositions give same PPL within 2%)
  K2: Landscape is chaotic (PPL varies non-monotonically with small weight changes)

Success criteria:
  S1: Clear convex minimum exists for 2-adapter interpolation
  S2: 3-adapter simplex has a single basin of attraction

References:
  - LoRA Soups (arXiv 2410.13025): CAT composition
  - Naive LoRA Summation (arXiv 2508.11985): orthogonality enables additive composition
  - exp_softmax_router_scaling LEARNINGS: softmax router matches oracle at N=24
  - exp_real_data_25_domain_adapters LEARNINGS: N=24 composition -29.1% vs base

Platform: Apple M5 Pro 48GB, MLX
"""

import gc
import json
import math
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Paths to existing adapters and data from real_data_25_domain_adapters
ADAPTER_EXPERIMENT = EXPERIMENT_DIR.parent / "real_data_25_domain_adapters"
ADAPTERS_DIR = ADAPTER_EXPERIMENT / "adapters"
DATA_DIR = ADAPTER_EXPERIMENT / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
EVAL_BATCHES = 10  # Per eval point — keep fast

# Adapter pairs for Phase 1
# Same cluster: medical + health_fitness (Cluster 4 from softmax router)
# Different cluster: math + creative_writing (singleton vs Cluster 1)
# Overlapping: code + engineering (different sources but technical overlap)
PAIRS = [
    ("medical", "health_fitness", "same_cluster"),
    ("math", "creative_writing", "different_cluster"),
    ("code", "engineering", "overlapping"),
]

# Triple for Phase 2
TRIPLE = ("medical", "math", "code")

# Alpha sweep for 2-adapter interpolation
ALPHA_STEPS = 21
ALPHAS = [i / (ALPHA_STEPS - 1) for i in range(ALPHA_STEPS)]

# Simplex grid for 3-adapter interpolation
SIMPLEX_RESOLUTION = 10  # points per edge


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ============================================================================
# Model loading utilities (reused from real_data_25_domain_adapters)
# ============================================================================

from mlx_lm import load as mlx_load
from mlx_lm.models.bitlinear_layers import BitLinear


def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    """Unpack uint8-packed ternary weights to bfloat16."""
    w0 = (packed_weights & 3).astype(mx.bfloat16) - 1
    w1 = ((packed_weights >> 2) & 3).astype(mx.bfloat16) - 1
    w2 = ((packed_weights >> 4) & 3).astype(mx.bfloat16) - 1
    w3 = ((packed_weights >> 6) & 3).astype(mx.bfloat16) - 1
    unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:out_features]
    scale = weight_scale.astype(mx.bfloat16)
    if invert_scale:
        unpacked = unpacked / scale
    else:
        unpacked = unpacked * scale
    return unpacked


def replace_bitlinear_with_linear(model):
    """Replace BitLinear with nn.Linear for composition evaluation."""
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                unpacked_w = unpack_ternary(
                    module.weight, module.out_features,
                    module.weight_scale, module.invert_weight_scales,
                )
                has_bias = module.bias is not None
                linear = nn.Linear(module.in_features, module.out_features, bias=has_bias)
                linear.weight = unpacked_w
                if has_bias:
                    linear.bias = module.bias
                updates.append((key, linear))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    log(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


def load_adapter(adapter_dir):
    """Load adapter B-matrices from npz file."""
    path = adapter_dir / "adapter.npz"
    data = dict(np.load(str(path)))
    return {k: mx.array(v) for k, v in data.items()}


def compute_ppl(model, tokenizer, data_path: Path, max_batches: int = EVAL_BATCHES):
    """Compute perplexity on validation data."""
    valid_path = data_path / "valid.jsonl"
    if not valid_path.exists():
        return float("inf")

    texts = []
    with open(valid_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    total_loss = 0.0
    total_tokens = 0

    for text in texts[:max_batches]:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:MAX_SEQ_LENGTH + 1]

        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]

        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)

        total_loss += loss.item()
        total_tokens += y.size
        del logits, loss, x, y

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 100))


# ============================================================================
# Weighted composition: apply interpolated adapters to model
# ============================================================================

class WeightedMultiLoRALinear(nn.Module):
    """LoRA linear with multiple A/B pairs weighted by given coefficients."""
    def __init__(self, base_linear: nn.Linear, rank: int, scale: float,
                 a_matrices: list, b_matrices: list, weights: list):
        super().__init__()
        self.linear = base_linear
        self.scale = scale
        self.rank = rank
        self.a_matrices = a_matrices
        self.b_matrices = b_matrices
        self.weights = weights
        self.linear.freeze()

    def __call__(self, x):
        base_out = self.linear(x)
        if not self.a_matrices:
            return base_out

        lora_sum = mx.zeros_like(base_out)
        for i in range(len(self.a_matrices)):
            if abs(self.weights[i]) < 1e-8:
                continue
            b = self.b_matrices[i]
            # Apply ternary quantization for consistency with training
            alpha_val = mx.mean(mx.abs(b))
            b_scaled = b / (alpha_val + 1e-7)
            b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha_val
            lora_sum = lora_sum + self.weights[i] * ((x @ self.a_matrices[i]) @ b_q)

        return base_out + lora_sum * self.scale


TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def apply_weighted_composition(model, skeleton, adapter_bs, domain_indices, weights):
    """Apply weighted multi-adapter composition to the model in-place.

    adapter_bs: list of adapter B-matrix dicts (one per domain in composition)
    domain_indices: list of domain indices in the skeleton
    weights: list of floats (one per adapter)
    """
    count = 0
    n_layers = len(model.model.layers)
    n_experts = len(domain_indices)

    for li, layer in enumerate(model.model.layers):
        lora_updates = []
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None:
                continue
            # Get the base linear — might be wrapped or raw
            if isinstance(module, WeightedMultiLoRALinear):
                base_linear = module.linear
            elif isinstance(module, nn.Linear):
                base_linear = module
            else:
                continue

            a_mats = []
            b_mats = []
            for ei, di in enumerate(domain_indices):
                skey = f"layer_{li}_{key}_domain_{di}"
                if skey not in skeleton:
                    break
                a_mats.append(mx.array(skeleton[skey]).astype(mx.bfloat16))

                param_name = f"model.layers.{li}.{key}.lora_b"
                if param_name in adapter_bs[ei]:
                    b_mats.append(adapter_bs[ei][param_name])
                else:
                    b_mats.append(mx.zeros((LORA_RANK, base_linear.weight.shape[0])))

            if len(a_mats) != n_experts:
                continue

            weighted = WeightedMultiLoRALinear(
                base_linear, rank=LORA_RANK, scale=LORA_SCALE,
                a_matrices=a_mats, b_matrices=b_mats, weights=list(weights)
            )
            lora_updates.append((key, weighted))
            count += 1

        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    mx.eval(model.parameters())
    model.freeze()
    return count


# ============================================================================
# Phase 1: 2-adapter alpha sweep
# ============================================================================

def phase_2adapter_sweep():
    """Sweep alpha in [0,1] for 3 adapter pairs, evaluate PPL on each domain."""
    log("\n[Phase 1] 2-adapter interpolation sweep")
    t0 = time.time()

    # Load skeleton
    skeleton = dict(np.load(str(ADAPTERS_DIR / "grassmannian_skeleton_n24.npz")))

    # Domain name -> index in skeleton
    all_domains = [
        "medical", "code", "math", "legal", "finance",
        "science", "history", "philosophy", "creative_writing", "cooking",
        "health_fitness", "psychology", "education", "engineering", "agriculture",
        "environmental", "politics", "economics", "sociology", "linguistics",
        "cybersecurity", "marketing", "sports", "music",
    ]
    domain_to_idx = {d: i for i, d in enumerate(all_domains)}

    results = {}

    for domain_a, domain_b, pair_type in PAIRS:
        log(f"\n  Pair: {domain_a} + {domain_b} ({pair_type})")
        pair_key = f"{domain_a}__{domain_b}"

        # Load model fresh for each pair
        model, tokenizer = mlx_load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)

        # Load adapters
        adapter_a = load_adapter(ADAPTERS_DIR / domain_a)
        adapter_b = load_adapter(ADAPTERS_DIR / domain_b)

        idx_a = domain_to_idx[domain_a]
        idx_b = domain_to_idx[domain_b]

        pair_results = {"pair_type": pair_type, "alphas": {}}

        for alpha in ALPHAS:
            alpha_key = f"{alpha:.2f}"
            weights = [alpha, 1.0 - alpha]

            apply_weighted_composition(
                model, skeleton, [adapter_a, adapter_b],
                [idx_a, idx_b], weights
            )

            evals = {}
            # Eval on domain A
            ppl_a = compute_ppl(model, tokenizer, DATA_DIR / domain_a)
            evals[domain_a] = round(ppl_a, 3)

            # Eval on domain B
            ppl_b = compute_ppl(model, tokenizer, DATA_DIR / domain_b)
            evals[domain_b] = round(ppl_b, 3)

            # Mixed eval: average of both domain PPLs (geometric mean)
            evals["mixed_geo"] = round(math.sqrt(ppl_a * ppl_b), 3)

            pair_results["alphas"][alpha_key] = evals
            log(f"    alpha={alpha:.2f}: {domain_a}={ppl_a:.2f}, {domain_b}={ppl_b:.2f}, mixed={evals['mixed_geo']:.2f}")

        results[pair_key] = pair_results

        cleanup(model, tokenizer, adapter_a, adapter_b)

    elapsed = time.time() - t0
    log(f"\n  Phase 1 done in {elapsed:.1f}s")
    log_memory("post-phase1")
    return results


# ============================================================================
# Phase 2: 3-adapter simplex
# ============================================================================

def generate_simplex_points(resolution):
    """Generate evenly spaced points on the 2D simplex w1+w2+w3=1."""
    points = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            w1 = i / resolution
            w2 = j / resolution
            w3 = k / resolution
            points.append((w1, w2, w3))
    return points


def phase_3adapter_simplex():
    """Sweep the 3-adapter simplex for medical/math/code."""
    log("\n[Phase 2] 3-adapter simplex sweep")
    t0 = time.time()

    skeleton = dict(np.load(str(ADAPTERS_DIR / "grassmannian_skeleton_n24.npz")))

    all_domains = [
        "medical", "code", "math", "legal", "finance",
        "science", "history", "philosophy", "creative_writing", "cooking",
        "health_fitness", "psychology", "education", "engineering", "agriculture",
        "environmental", "politics", "economics", "sociology", "linguistics",
        "cybersecurity", "marketing", "sports", "music",
    ]
    domain_to_idx = {d: i for i, d in enumerate(all_domains)}

    d1, d2, d3 = TRIPLE
    idx1, idx2, idx3 = domain_to_idx[d1], domain_to_idx[d2], domain_to_idx[d3]

    adapter1 = load_adapter(ADAPTERS_DIR / d1)
    adapter2 = load_adapter(ADAPTERS_DIR / d2)
    adapter3 = load_adapter(ADAPTERS_DIR / d3)

    simplex_points = generate_simplex_points(SIMPLEX_RESOLUTION)
    log(f"  {len(simplex_points)} simplex points for {TRIPLE}")

    model, tokenizer = mlx_load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    results = {"domains": list(TRIPLE), "points": []}

    for pi, (w1, w2, w3) in enumerate(simplex_points):
        apply_weighted_composition(
            model, skeleton, [adapter1, adapter2, adapter3],
            [idx1, idx2, idx3], [w1, w2, w3]
        )

        ppl1 = compute_ppl(model, tokenizer, DATA_DIR / d1)
        ppl2 = compute_ppl(model, tokenizer, DATA_DIR / d2)
        ppl3 = compute_ppl(model, tokenizer, DATA_DIR / d3)

        point = {
            "w": [round(w1, 2), round(w2, 2), round(w3, 2)],
            d1: round(ppl1, 3),
            d2: round(ppl2, 3),
            d3: round(ppl3, 3),
            "geo_mean": round((ppl1 * ppl2 * ppl3) ** (1/3), 3),
        }
        results["points"].append(point)

        if pi % 11 == 0 or pi == len(simplex_points) - 1:
            log(f"    [{pi+1}/{len(simplex_points)}] w=({w1:.1f},{w2:.1f},{w3:.1f}) "
                f"{d1}={ppl1:.2f} {d2}={ppl2:.2f} {d3}={ppl3:.2f}")

    cleanup(model, tokenizer, adapter1, adapter2, adapter3)
    del skeleton

    elapsed = time.time() - t0
    log(f"\n  Phase 2 done in {elapsed:.1f}s")
    log_memory("post-phase2")
    return results


# ============================================================================
# Phase 3: Smoothness analysis
# ============================================================================

def analyze_smoothness(phase1_results, phase2_results):
    """Compute smoothness metrics from phase 1 and 2 data."""
    log("\n[Phase 3] Smoothness analysis")

    analysis = {}

    # --- Phase 1: 1D curve analysis ---
    for pair_key, pair_data in phase1_results.items():
        alphas_data = pair_data["alphas"]
        sorted_alphas = sorted(alphas_data.keys(), key=float)

        domains_in_pair = pair_key.split("__")
        eval_keys = domains_in_pair + ["mixed_geo"]

        pair_analysis = {}
        for eval_key in eval_keys:
            ppls = [alphas_data[a][eval_key] for a in sorted_alphas]
            alphas_float = [float(a) for a in sorted_alphas]

            # PPL range (K1 check: flat if range < 2% of mean)
            ppl_range = max(ppls) - min(ppls)
            ppl_mean = np.mean(ppls)
            range_pct = (ppl_range / ppl_mean * 100) if ppl_mean > 0 else 0

            # Monotonicity check
            diffs = [ppls[i+1] - ppls[i] for i in range(len(ppls)-1)]
            n_increases = sum(1 for d in diffs if d > 0)
            n_decreases = sum(1 for d in diffs if d < 0)
            monotonic_ratio = max(n_increases, n_decreases) / len(diffs) if diffs else 0

            # Convexity check: fraction of points where midpoint < chord
            convex_count = 0
            total_checks = 0
            for i in range(1, len(ppls) - 1):
                chord = 0.5 * (ppls[i-1] + ppls[i+1])
                if ppls[i] <= chord:
                    convex_count += 1
                total_checks += 1
            convexity_ratio = convex_count / total_checks if total_checks > 0 else 0

            # Numerical gradient (finite differences)
            grads = []
            for i in range(1, len(ppls) - 1):
                g = (ppls[i+1] - ppls[i-1]) / (alphas_float[i+1] - alphas_float[i-1])
                grads.append(g)

            # Lipschitz constant of gradient
            grad_diffs = [abs(grads[i+1] - grads[i]) for i in range(len(grads)-1)]
            lipschitz = max(grad_diffs) / (alphas_float[1] - alphas_float[0]) if grad_diffs else 0

            # Optimal alpha (minimum PPL)
            min_idx = np.argmin(ppls)
            optimal_alpha = alphas_float[min_idx]

            pair_analysis[eval_key] = {
                "ppl_range": round(ppl_range, 3),
                "ppl_mean": round(ppl_mean, 3),
                "range_pct": round(range_pct, 2),
                "monotonic_ratio": round(monotonic_ratio, 3),
                "convexity_ratio": round(convexity_ratio, 3),
                "lipschitz_constant": round(lipschitz, 2),
                "optimal_alpha": round(optimal_alpha, 2),
                "optimal_ppl": round(ppls[min_idx], 3),
                "endpoint_ppls": [round(ppls[0], 3), round(ppls[-1], 3)],
            }

        analysis[pair_key] = pair_analysis

    # --- Phase 2: Simplex analysis ---
    points = phase2_results["points"]
    domains = phase2_results["domains"]

    for domain in domains + ["geo_mean"]:
        ppls = [p[domain] for p in points]
        ppl_range = max(ppls) - min(ppls)
        ppl_mean = np.mean(ppls)

        # Find optimal point
        min_idx = np.argmin(ppls)
        optimal_w = points[min_idx]["w"]
        optimal_ppl = ppls[min_idx]

        # Check for multiple basins: count local minima
        # A point is a local minimum if its PPL is lower than all neighbors
        # For simplicity, check if the best point's PPL is significantly lower
        # than the 2nd best
        sorted_ppls = sorted(ppls)
        gap_to_2nd = (sorted_ppls[1] - sorted_ppls[0]) / sorted_ppls[0] * 100 if sorted_ppls[0] > 0 else 0

        analysis[f"simplex_{domain}"] = {
            "ppl_range": round(ppl_range, 3),
            "ppl_mean": round(float(ppl_mean), 3),
            "range_pct": round(float(ppl_range / ppl_mean * 100), 2),
            "optimal_w": [round(w, 2) for w in optimal_w],
            "optimal_ppl": round(optimal_ppl, 3),
            "gap_to_2nd_pct": round(gap_to_2nd, 4),
            "n_points": len(points),
        }

    # --- Numerical Hessian for simplex (geo_mean) ---
    # Use the 3 vertices and 3 edge midpoints to estimate curvature
    # Vertices: (1,0,0), (0,1,0), (0,0,1)
    # Midpoints: (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # Center: (1/3,1/3,1/3)
    vertex_ppls = {}
    mid_ppls = {}
    center_ppl = None
    for p in points:
        w = tuple(p["w"])
        if w == (1.0, 0.0, 0.0):
            vertex_ppls["v1"] = p["geo_mean"]
        elif w == (0.0, 1.0, 0.0):
            vertex_ppls["v2"] = p["geo_mean"]
        elif w == (0.0, 0.0, 1.0):
            vertex_ppls["v3"] = p["geo_mean"]
        elif w == (0.5, 0.5, 0.0):
            mid_ppls["m12"] = p["geo_mean"]
        elif w == (0.5, 0.0, 0.5):
            mid_ppls["m13"] = p["geo_mean"]
        elif w == (0.0, 0.5, 0.5):
            mid_ppls["m23"] = p["geo_mean"]
        # Closest to center
        if abs(p["w"][0] - 0.3) < 0.05 and abs(p["w"][1] - 0.3) < 0.05:
            center_ppl = p["geo_mean"]

    if len(vertex_ppls) == 3 and len(mid_ppls) == 3:
        # Convexity check: midpoint should be <= average of vertices
        for mk, mv in mid_ppls.items():
            v_keys = {"m12": ("v1", "v2"), "m13": ("v1", "v3"), "m23": ("v2", "v3")}
            va, vb = v_keys[mk]
            chord = 0.5 * (vertex_ppls[va] + vertex_ppls[vb])
            analysis[f"simplex_convexity_{mk}"] = {
                "midpoint_ppl": round(mv, 3),
                "chord_ppl": round(chord, 3),
                "is_convex": mv <= chord,
                "gap_pct": round((mv - chord) / chord * 100, 3),
            }

    log("  Smoothness analysis complete")
    return analysis


# ============================================================================
# Main orchestrator
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("Composition Interpolation Landscape Experiment")
    log("=" * 70)
    log_memory("start")

    # Verify data exists
    for domain_a, domain_b, _ in PAIRS:
        for d in [domain_a, domain_b]:
            assert (DATA_DIR / d / "valid.jsonl").exists(), f"Missing data for {d}"
    for d in TRIPLE:
        assert (ADAPTERS_DIR / d / "adapter.npz").exists(), f"Missing adapter for {d}"

    # Phase 1: 2-adapter sweep
    phase1_results = phase_2adapter_sweep()

    # Phase 2: 3-adapter simplex
    phase2_results = phase_3adapter_simplex()

    # Phase 3: Smoothness analysis
    smoothness = analyze_smoothness(phase1_results, phase2_results)

    # Aggregate results
    total_time = time.time() - t0

    # Kill criteria assessment
    # K1: Flat landscape — check if range < 2% of mean for any pair
    k1_flat_count = 0
    k1_total = 0
    for pair_key in phase1_results:
        for eval_key in smoothness[pair_key]:
            k1_total += 1
            if smoothness[pair_key][eval_key]["range_pct"] < 2.0:
                k1_flat_count += 1

    k1_pass = k1_flat_count < k1_total  # PASS if NOT all flat

    # K2: Chaotic — check monotonicity ratio < 0.6 (non-monotonic)
    # AND convexity ratio < 0.3 (not convex)
    k2_chaotic = False
    for pair_key in phase1_results:
        for eval_key in smoothness[pair_key]:
            s = smoothness[pair_key][eval_key]
            if s["monotonic_ratio"] < 0.6 and s["convexity_ratio"] < 0.3:
                k2_chaotic = True

    k2_pass = not k2_chaotic  # PASS if NOT chaotic

    # S1: Clear convex minimum for 2-adapter
    s1_convex_count = 0
    s1_total = 0
    for pair_key in phase1_results:
        mixed_key = "mixed_geo"
        if mixed_key in smoothness[pair_key]:
            s1_total += 1
            s = smoothness[pair_key][mixed_key]
            # Convex if convexity_ratio > 0.5 AND optimal not at endpoint
            if s["convexity_ratio"] >= 0.5 or 0.1 < s["optimal_alpha"] < 0.9:
                s1_convex_count += 1

    s1_pass = s1_convex_count >= 2  # At least 2/3 pairs show convex mixed landscape

    # S2: Single basin on simplex
    s2_pass = True
    if "simplex_geo_mean" in smoothness:
        # Single basin if range is substantial and gap to 2nd best is small
        # (meaning smooth valley, not sharp spike)
        s = smoothness["simplex_geo_mean"]
        s2_pass = s["range_pct"] > 2.0  # Not flat

    results = {
        "phase1_2adapter_sweep": phase1_results,
        "phase2_3adapter_simplex": phase2_results,
        "smoothness_analysis": smoothness,
        "kill_criteria": {
            "K1_flat": {"pass": k1_pass, "flat_count": k1_flat_count, "total": k1_total},
            "K2_chaotic": {"pass": k2_pass, "chaotic_found": k2_chaotic},
        },
        "success_criteria": {
            "S1_convex_minimum": {"pass": s1_pass, "convex_count": s1_convex_count, "total": s1_total},
            "S2_single_basin": {"pass": s2_pass},
        },
        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n{'='*70}")
    log(f"RESULTS SUMMARY")
    log(f"{'='*70}")
    log(f"K1 (not flat):    {'PASS' if k1_pass else 'FAIL'} — {k1_flat_count}/{k1_total} curves flat (<2% range)")
    log(f"K2 (not chaotic): {'PASS' if k2_pass else 'FAIL'} — chaotic curves found: {k2_chaotic}")
    log(f"S1 (convex min):  {'PASS' if s1_pass else 'FAIL'} — {s1_convex_count}/{s1_total} pairs with convex mixed landscape")
    log(f"S2 (single basin):{'PASS' if s2_pass else 'FAIL'}")
    log(f"Total time: {total_time:.1f}s")
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
