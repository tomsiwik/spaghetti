#!/usr/bin/env python3
"""Knowledge Region Overlap Mapping: Sheaf Cover Construction.

For 5 domain adapters (medical, code, math, legal, finance), compute:
1. Improvement sets U_i = {x : PPL_adapter(x) < PPL_base(x)}
2. All pairwise overlaps U_i ∩ U_j
3. Representation compatibility cos(h_i(x), h_j(x)) on overlaps
4. Cech nerve of the cover {U_i}

Kill criteria:
  K1 (#644): At least 3 non-empty pairwise overlaps |U_i ∩ U_j| > 50 samples
  K2 (#645): Compatibility varies within overlaps (std cosine > 0.1)

Type: Guided exploration
Platform: Apple M5 Pro 48GB, MLX
"""

import gc
import json
import math
import os
import time
from itertools import combinations
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.bitlinear_layers import BitLinear
from mlx_lm.models.bitnet import create_attention_mask
from mlx.utils import tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Paths to existing infrastructure
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
MAX_SEQ_LENGTH = 256
HIDDEN_LAYER = 15  # Middle layer (of 30) for representation extraction
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]
TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

# Scale per adapter (Finding #217)
DOMAIN_SCALES = {
    "medical": 20.0,
    "code": 20.0,
    "math": 20.0,
    "legal": 4.0,
    "finance": 1.0,
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def log(msg, end="\n"):
    print(msg, end=end, flush=True)


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
# Model utilities (reused from behavioral_eval_routed)
# ============================================================================

def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
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


def load_skeleton():
    skeleton_path = ADAPTERS_DIR / "grassmannian_skeleton.npz"
    return dict(np.load(str(skeleton_path)))


def load_adapter(domain):
    adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
    return dict(mx.load(str(adapter_path)))


def premerge_single_adapter(model, skeleton, adapter, domain, scale):
    """Pre-merge a single adapter into model weights: W_new = W_base + scale * B^T @ A^T"""
    n_layers = len(model.model.layers)
    merge_count = 0
    di = DOMAINS.index(domain)

    for li in range(n_layers):
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = model.model.layers[li]
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue

            skey = f"layer_{li}_{key}_domain_{di}"
            if skey not in skeleton:
                continue
            a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)

            b_key = f"model.layers.{li}.{key}.lora_b"
            if b_key not in adapter:
                continue
            b_mx = adapter[b_key]

            delta = scale * (b_mx.T @ a_mx.T)
            module.weight = module.weight + delta
            merge_count += 1

    mx.eval(model.parameters())
    return merge_count


def save_base_weights(model):
    """Save base weights so we can restore after adapter merge."""
    base_weights = {}
    for li, layer in enumerate(model.model.layers):
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None and isinstance(module, nn.Linear):
                base_weights[(li, key)] = module.weight
    return base_weights


def restore_base_weights(model, base_weights):
    """Restore base weights after adapter removal."""
    for (li, key), w in base_weights.items():
        parts = key.split(".")
        module = model.model.layers[li]
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                break
        if module is not None and isinstance(module, nn.Linear):
            module.weight = w
    mx.eval(model.parameters())


# ============================================================================
# Data loading
# ============================================================================

def load_all_samples():
    """Load all validation samples from all 5 domains, tagged with source domain."""
    samples = []
    for domain in DOMAINS:
        fpath = DATA_DIR / domain / "valid.jsonl"
        with open(fpath) as f:
            for line in f:
                obj = json.loads(line)
                samples.append({
                    "text": obj["text"],
                    "domain": domain,
                })
    log(f"  Loaded {len(samples)} samples from {len(DOMAINS)} domains")
    return samples


# ============================================================================
# Per-sample PPL and hidden state extraction
# ============================================================================

def compute_per_sample_ppl_and_hidden(model, tokenizer, samples, extract_hidden=True):
    """Compute PPL and optionally extract hidden states (layer 15) for each sample.

    Returns:
        ppls: list of float (per-sample PPL)
        hiddens: list of np.ndarray (per-sample mean hidden state at HIDDEN_LAYER)
                 or None if extract_hidden=False
    """
    ppls = []
    hiddens = [] if extract_hidden else None

    for i, sample in enumerate(samples):
        tokens = tokenizer.encode(sample["text"])
        if len(tokens) < 2:
            ppls.append(float("inf"))
            if extract_hidden:
                hiddens.append(None)
            continue

        tokens = tokens[:MAX_SEQ_LENGTH + 1]
        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]

        # Forward pass with hidden state capture
        if extract_hidden:
            h = model.model.embed_tokens(x)
            mask = create_attention_mask(h, cache=None)
            for li, layer in enumerate(model.model.layers):
                h = layer(h, mask=mask, cache=None)
                if li == HIDDEN_LAYER:
                    # Mean pool over sequence for a single representation vector
                    hidden_vec = mx.mean(h, axis=1)  # [1, d]
                    mx.eval(hidden_vec)
                    hidden_np = np.array(hidden_vec[0].astype(mx.float32))

            # Continue to get logits
            h = model.model.norm(h)
            if model.args.tie_word_embeddings:
                logits = model.model.embed_tokens.as_linear(h)
            else:
                logits = model.lm_head(h)
        else:
            logits = model(x)

        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)
        n_tokens = y.size
        ppl = math.exp(min(loss.item() / n_tokens, 100))
        ppls.append(ppl)

        if extract_hidden:
            hiddens.append(hidden_np)

        del logits, loss, x, y
        if extract_hidden:
            del h, hidden_vec

        if (i + 1) % 50 == 0:
            log(f"    Processed {i+1}/{len(samples)} samples")

    return ppls, hiddens


# ============================================================================
# Phase 1: Base model PPL
# ============================================================================

def phase_base(samples):
    """Compute base model PPL for all samples (no adapter)."""
    log("\n" + "=" * 70)
    log("[Phase 1] Base model PPL for all samples")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    mx.eval(model.parameters())
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    log_memory("post-load")

    ppls, _ = compute_per_sample_ppl_and_hidden(model, tokenizer, samples, extract_hidden=False)

    elapsed = time.time() - t0
    log(f"  Phase 1 complete in {elapsed:.1f}s")
    log_memory("post-base-ppl")

    # Save base weights for reuse
    base_weights = save_base_weights(model)

    return ppls, model, tokenizer, base_weights


# ============================================================================
# Phase 2: Per-adapter PPL + hidden states
# ============================================================================

def phase_adapter_eval(model, tokenizer, base_weights, samples, domain):
    """Evaluate one adapter: PPL + hidden states for all samples."""
    log(f"\n  --- Adapter: {domain} (scale={DOMAIN_SCALES[domain]}) ---")

    t0 = time.time()
    skeleton = load_skeleton()
    adapter = load_adapter(domain)

    # Merge adapter into model
    restore_base_weights(model, base_weights)
    merge_count = premerge_single_adapter(
        model, skeleton, adapter, domain, DOMAIN_SCALES[domain]
    )
    log(f"    Merged {merge_count} layers")
    del skeleton, adapter

    # Compute per-sample PPL and hidden states
    ppls, hiddens = compute_per_sample_ppl_and_hidden(
        model, tokenizer, samples, extract_hidden=True
    )

    elapsed = time.time() - t0
    log(f"    {domain} done in {elapsed:.1f}s")

    return ppls, hiddens


# ============================================================================
# Phase 3: Analysis — improvement sets, overlaps, compatibility, Cech nerve
# ============================================================================

def phase_analysis(samples, base_ppls, adapter_ppls, adapter_hiddens):
    """Build improvement sets, overlaps, compatibility, and Cech nerve."""
    log("\n" + "=" * 70)
    log("[Phase 3] Analysis: improvement sets, overlaps, Cech nerve")
    log("=" * 70)

    n_samples = len(samples)

    # --- Improvement sets U_i ---
    improvement_sets = {}
    for domain in DOMAINS:
        ui = set()
        for idx in range(n_samples):
            if adapter_ppls[domain][idx] < base_ppls[idx]:
                ui.add(idx)
        improvement_sets[domain] = ui
        # Per-source-domain breakdown
        by_source = {}
        for idx in ui:
            src = samples[idx]["domain"]
            by_source[src] = by_source.get(src, 0) + 1
        log(f"  |U_{domain}| = {len(ui)} / {n_samples}")
        log(f"    Breakdown: {by_source}")

    # --- Pairwise overlaps U_i ∩ U_j ---
    pairs = list(combinations(DOMAINS, 2))
    overlaps = {}
    overlap_sizes = {}
    for d1, d2 in pairs:
        overlap = improvement_sets[d1] & improvement_sets[d2]
        overlaps[(d1, d2)] = overlap
        overlap_sizes[(d1, d2)] = len(overlap)
        log(f"  |U_{d1} ∩ U_{d2}| = {len(overlap)}")

    # --- Compatibility on overlaps: cosine similarity ---
    compatibility = {}
    for (d1, d2), overlap in overlaps.items():
        if len(overlap) < 2:
            compatibility[(d1, d2)] = {
                "mean_cos": float("nan"),
                "std_cos": float("nan"),
                "min_cos": float("nan"),
                "max_cos": float("nan"),
                "n": len(overlap),
                "cosines": [],
            }
            continue

        cosines = []
        for idx in overlap:
            h1 = adapter_hiddens[d1][idx]
            h2 = adapter_hiddens[d2][idx]
            if h1 is None or h2 is None:
                continue
            # Cosine similarity
            dot = np.dot(h1, h2)
            norm1 = np.linalg.norm(h1)
            norm2 = np.linalg.norm(h2)
            if norm1 < 1e-8 or norm2 < 1e-8:
                continue
            cos = dot / (norm1 * norm2)
            cosines.append(float(cos))

        if cosines:
            compatibility[(d1, d2)] = {
                "mean_cos": float(np.mean(cosines)),
                "std_cos": float(np.std(cosines)),
                "min_cos": float(np.min(cosines)),
                "max_cos": float(np.max(cosines)),
                "n": len(cosines),
                "cosines": cosines,
            }
            log(f"  Compat({d1}, {d2}): mean={np.mean(cosines):.4f} "
                f"std={np.std(cosines):.4f} n={len(cosines)}")
        else:
            compatibility[(d1, d2)] = {
                "mean_cos": float("nan"),
                "std_cos": float("nan"),
                "min_cos": float("nan"),
                "max_cos": float("nan"),
                "n": 0,
                "cosines": [],
            }

    # --- Triple overlaps ---
    triples = list(combinations(DOMAINS, 3))
    triple_overlaps = {}
    for d1, d2, d3 in triples:
        triple = improvement_sets[d1] & improvement_sets[d2] & improvement_sets[d3]
        triple_overlaps[(d1, d2, d3)] = len(triple)
        if len(triple) > 0:
            log(f"  |U_{d1} ∩ U_{d2} ∩ U_{d3}| = {len(triple)}")

    # --- Quadruple and quintuple overlaps ---
    quads = list(combinations(DOMAINS, 4))
    quad_overlaps = {}
    for combo in quads:
        inter = set(range(n_samples))
        for d in combo:
            inter &= improvement_sets[d]
        quad_overlaps[combo] = len(inter)
        if len(inter) > 0:
            log(f"  |{'∩'.join(combo)}| = {len(inter)}")

    # Full 5-way
    full_overlap = set(range(n_samples))
    for d in DOMAINS:
        full_overlap &= improvement_sets[d]
    log(f"  |Full 5-way overlap| = {len(full_overlap)}")

    # --- Cech nerve construction ---
    nerve = {
        "vertices": DOMAINS,
        "edges": [],
        "triangles": [],
        "tetrahedra": [],
        "full_simplex": len(full_overlap) > 0,
    }
    for (d1, d2), size in overlap_sizes.items():
        if size > 0:
            nerve["edges"].append({"pair": [d1, d2], "size": size})
    for (d1, d2, d3), size in triple_overlaps.items():
        if size > 0:
            nerve["triangles"].append({"triple": [d1, d2, d3], "size": size})
    for combo, size in quad_overlaps.items():
        if size > 0:
            nerve["tetrahedra"].append({"quad": list(combo), "size": size})

    log(f"\n  Cech nerve: {len(nerve['edges'])} edges, "
        f"{len(nerve['triangles'])} triangles, "
        f"{len(nerve['tetrahedra'])} tetrahedra, "
        f"full={nerve['full_simplex']}")

    # --- Euler characteristic ---
    chi = (len(DOMAINS) - len(nerve["edges"]) + len(nerve["triangles"])
           - len(nerve["tetrahedra"]) + (1 if nerve["full_simplex"] else 0))
    log(f"  Euler characteristic chi = {chi}")

    # --- PPL improvement ratios ---
    ppl_improvement = {}
    for domain in DOMAINS:
        ratios = []
        for idx in range(n_samples):
            if base_ppls[idx] > 0 and not math.isinf(adapter_ppls[domain][idx]):
                ratios.append(adapter_ppls[domain][idx] / base_ppls[idx])
        ppl_improvement[domain] = {
            "mean_ratio": float(np.mean(ratios)),
            "own_domain_mean_ratio": float(np.mean([
                adapter_ppls[domain][idx] / base_ppls[idx]
                for idx in range(n_samples)
                if samples[idx]["domain"] == domain and base_ppls[idx] > 0
            ])),
            "cross_domain_mean_ratio": float(np.mean([
                adapter_ppls[domain][idx] / base_ppls[idx]
                for idx in range(n_samples)
                if samples[idx]["domain"] != domain and base_ppls[idx] > 0
            ])),
        }
        log(f"  PPL ratio {domain}: own={ppl_improvement[domain]['own_domain_mean_ratio']:.3f} "
            f"cross={ppl_improvement[domain]['cross_domain_mean_ratio']:.3f}")

    # ====================================================================
    # SECONDARY ANALYSIS: Specialization sets (where adapter i is BEST)
    # ====================================================================
    log(f"\n  --- Secondary: Specialization sets ---")

    specialization_sets = {d: set() for d in DOMAINS}
    for idx in range(n_samples):
        best_domain = min(DOMAINS, key=lambda d: adapter_ppls[d][idx])
        specialization_sets[best_domain].add(idx)

    for domain in DOMAINS:
        si = specialization_sets[domain]
        by_source = {}
        for idx in si:
            src = samples[idx]["domain"]
            by_source[src] = by_source.get(src, 0) + 1
        log(f"  |S_{domain}| = {len(si)} (best-adapter set)")
        log(f"    Breakdown: {by_source}")

    # Specialization overlaps (where adapter i is within 5% of best)
    near_best_sets = {d: set() for d in DOMAINS}
    for idx in range(n_samples):
        best_ppl = min(adapter_ppls[d][idx] for d in DOMAINS)
        for domain in DOMAINS:
            if adapter_ppls[domain][idx] <= best_ppl * 1.05:  # within 5%
                near_best_sets[domain].add(idx)

    log(f"\n  --- Near-best sets (within 5% of best adapter) ---")
    for domain in DOMAINS:
        log(f"  |NB_{domain}| = {len(near_best_sets[domain])}")

    # Pairwise near-best overlaps
    nb_overlaps = {}
    for d1, d2 in pairs:
        overlap = near_best_sets[d1] & near_best_sets[d2]
        nb_overlaps[(d1, d2)] = len(overlap)
        log(f"  |NB_{d1} ∩ NB_{d2}| = {len(overlap)}")

    # ====================================================================
    # SECONDARY ANALYSIS: Hidden state DIFFERENCE magnitude (L2 norm)
    # ====================================================================
    log(f"\n  --- Hidden state difference magnitude (L2 norm of h_i - h_j) ---")
    hidden_diff_stats = {}
    for d1, d2 in pairs:
        diffs = []
        for idx in range(n_samples):
            h1 = adapter_hiddens[d1][idx]
            h2 = adapter_hiddens[d2][idx]
            if h1 is None or h2 is None:
                continue
            diff_norm = float(np.linalg.norm(h1 - h2))
            base_norm = float(np.linalg.norm(h1))
            if base_norm > 1e-8:
                rel_diff = diff_norm / base_norm
            else:
                rel_diff = float("inf")
            diffs.append({"abs": diff_norm, "rel": rel_diff})

        if diffs:
            abs_diffs = [d["abs"] for d in diffs]
            rel_diffs = [d["rel"] for d in diffs]
            hidden_diff_stats[f"{d1}_{d2}"] = {
                "mean_abs_diff": float(np.mean(abs_diffs)),
                "std_abs_diff": float(np.std(abs_diffs)),
                "mean_rel_diff": float(np.mean(rel_diffs)),
                "std_rel_diff": float(np.std(rel_diffs)),
            }
            log(f"  ||h_{d1} - h_{d2}||: mean_abs={np.mean(abs_diffs):.4f} "
                f"mean_rel={np.mean(rel_diffs):.4f} std_rel={np.std(rel_diffs):.4f}")

    # ====================================================================
    # PPL DIFFERENCE analysis (where do adapters disagree most?)
    # ====================================================================
    log(f"\n  --- PPL disagreement between adapters ---")
    ppl_disagreement = {}
    for d1, d2 in pairs:
        ratios = []
        for idx in range(n_samples):
            p1 = adapter_ppls[d1][idx]
            p2 = adapter_ppls[d2][idx]
            if p1 > 0 and p2 > 0:
                ratios.append(p1 / p2)
        if ratios:
            ppl_disagreement[f"{d1}_{d2}"] = {
                "mean_ratio": float(np.mean(ratios)),
                "std_ratio": float(np.std(ratios)),
                "max_ratio": float(np.max(ratios)),
                "min_ratio": float(np.min(ratios)),
            }
            log(f"  PPL({d1})/PPL({d2}): mean={np.mean(ratios):.3f} "
                f"std={np.std(ratios):.3f} range=[{np.min(ratios):.3f}, {np.max(ratios):.3f}]")

    # --- Kill criteria assessment ---
    k1_overlaps_above_50 = sum(1 for s in overlap_sizes.values() if s > 50)
    k1_pass = k1_overlaps_above_50 >= 3

    # Relax K1 if needed with secondary threshold
    k1_overlaps_above_20 = sum(1 for s in overlap_sizes.values() if s > 20)

    # K2: std of cosine > 0.1 for overlaps with enough samples
    k2_pairs_variable = 0
    k2_pairs_total = 0
    for (d1, d2), comp in compatibility.items():
        if comp["n"] >= 5:  # need enough points
            k2_pairs_total += 1
            if comp["std_cos"] > 0.1:
                k2_pairs_variable += 1
    k2_pass = k2_pairs_variable > 0

    # K2 secondary: check hidden diff variability instead
    k2_hidden_diff_variable = 0
    for key, stats in hidden_diff_stats.items():
        if stats["std_rel_diff"] > 0.1:
            k2_hidden_diff_variable += 1

    log(f"\n  === KILL CRITERIA ===")
    log(f"  K1 (#644): {k1_overlaps_above_50} overlaps > 50 samples (need >= 3): "
        f"{'PASS' if k1_pass else 'FAIL'}")
    log(f"    (relaxed: {k1_overlaps_above_20} overlaps > 20 samples)")
    log(f"  K2 (#645): {k2_pairs_variable}/{k2_pairs_total} overlaps with std(cos) > 0.1: "
        f"{'PASS' if k2_pass else 'FAIL'}")
    log(f"    K2 secondary (hidden diff): {k2_hidden_diff_variable} pairs with "
        f"std(rel_diff) > 0.1")

    return {
        "improvement_sets": {d: sorted(list(s)) for d, s in improvement_sets.items()},
        "improvement_set_sizes": {d: len(s) for d, s in improvement_sets.items()},
        "overlap_sizes": {f"{d1}_{d2}": s for (d1, d2), s in overlap_sizes.items()},
        "compatibility": {
            f"{d1}_{d2}": {k: v for k, v in comp.items() if k != "cosines"}
            for (d1, d2), comp in compatibility.items()
        },
        "compatibility_cosines": {
            f"{d1}_{d2}": comp["cosines"]
            for (d1, d2), comp in compatibility.items()
        },
        "triple_overlaps": {f"{'_'.join(t)}": s for t, s in triple_overlaps.items()},
        "quad_overlaps": {f"{'_'.join(q)}": s for q, s in quad_overlaps.items()},
        "full_5way_overlap": len(full_overlap),
        "nerve": nerve,
        "euler_characteristic": chi,
        "ppl_improvement": ppl_improvement,
        "specialization_sets": {d: sorted(list(s)) for d, s in specialization_sets.items()},
        "specialization_set_sizes": {d: len(s) for d, s in specialization_sets.items()},
        "near_best_set_sizes": {d: len(s) for d, s in near_best_sets.items()},
        "near_best_overlaps": {f"{d1}_{d2}": s for (d1, d2), s in nb_overlaps.items()},
        "hidden_diff_stats": hidden_diff_stats,
        "ppl_disagreement": ppl_disagreement,
        "kill_criteria": {
            "k1_overlaps_above_50": k1_overlaps_above_50,
            "k1_pass": k1_pass,
            "k1_overlaps_above_20": k1_overlaps_above_20,
            "k2_pairs_variable": k2_pairs_variable,
            "k2_pairs_total": k2_pairs_total,
            "k2_pass": k2_pass,
            "k2_hidden_diff_variable": k2_hidden_diff_variable,
        },
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    mx.random.seed(SEED)
    np.random.seed(SEED)

    log("=" * 70)
    log("Knowledge Region Overlap Mapping: Sheaf Cover Construction")
    log("=" * 70)
    log_memory("start")

    # Load all validation samples
    samples = load_all_samples()

    # Phase 1: Base model PPL
    base_ppls, model, tokenizer, base_weights = phase_base(samples)

    # Phase 2: Per-adapter PPL + hidden states
    log("\n" + "=" * 70)
    log("[Phase 2] Per-adapter PPL + hidden states")
    log("=" * 70)

    adapter_ppls = {}
    adapter_hiddens = {}
    for domain in DOMAINS:
        ppls, hiddens = phase_adapter_eval(
            model, tokenizer, base_weights, samples, domain
        )
        adapter_ppls[domain] = ppls
        adapter_hiddens[domain] = hiddens

    # Restore base and free model
    restore_base_weights(model, base_weights)
    log_memory("post-all-adapters")

    # Phase 3: Analysis
    analysis = phase_analysis(samples, base_ppls, adapter_ppls, adapter_hiddens)

    # Free model
    cleanup(model, tokenizer)
    del base_weights
    gc.collect()
    mx.clear_cache()
    log_memory("post-cleanup")

    # Save results
    total_time = time.time() - t_start
    results = {
        "experiment": "knowledge_region_overlap",
        "model": MODEL_ID,
        "domains": DOMAINS,
        "n_samples": len(samples),
        "hidden_layer": HIDDEN_LAYER,
        "domain_scales": DOMAIN_SCALES,
        "base_ppls_summary": {
            d: {
                "mean": float(np.mean([
                    base_ppls[i] for i in range(len(samples))
                    if samples[i]["domain"] == d
                ])),
            }
            for d in DOMAINS
        },
        **analysis,
        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
