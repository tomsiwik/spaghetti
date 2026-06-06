#!/usr/bin/env python3
"""Pierre v3: composition scaling N=24 with ridge router + null-space.

Frontier extension from N=5 (Finding #287) to N=24.
Tests three proven components at scale:
  1. Ridge regression router (Finding #276)
  2. Null-space SVD projection (Finding #273)
  3. NRE composition (Finding #287)

Kill criteria:
  K721: Ridge router accuracy < 50% at N=24 (random = 4.2%)
  K722: Null-space gradient preservation < 50% at N=24
  K723: Composed PPL > 2x worst single-adapter PPL

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

# Pierre API (current version)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))
from pierre import (
    fit_router, route, encode,
    compose_adapters, null_space_projector,
    attach_adapter, detach_adapters,
    load_adapter, load_frozen_A,
    ADAPTER_TARGETS,
)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Source data paths
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_25_domain_adapters"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
SEED = 42

# All 24 domains (alphabetical order matching skeleton domain indices)
DOMAINS = [
    "agriculture", "code", "cooking", "creative_writing", "cybersecurity",
    "economics", "education", "engineering", "environmental", "finance",
    "health_fitness", "history", "legal", "linguistics", "marketing",
    "math", "medical", "music", "philosophy", "politics",
    "psychology", "science", "sociology", "sports",
]

# Genuine domain adapters (trained on actual domain content)
GENUINE_DOMAINS = {"medical", "code", "math", "legal", "finance", "science", "health_fitness"}

N_CAL = 30     # calibration samples per domain (from train split)
N_TEST = 50    # test samples per domain (from valid split -- all of it)
N_PPL = 20     # PPL eval samples per domain (subset of valid)

# ---- Utilities ----

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    a = mx.get_active_memory() / 1e9
    c = mx.get_cache_memory() / 1e9
    p = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB peak={p:.2f}GB")


def cleanup(*objects):
    for o in objects:
        del o
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def load_data(domain, split="valid", n=None):
    samples = []
    path = DATA_DIR / domain / f"{split}.jsonl"
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line)["text"])
            if n and len(samples) >= n:
                break
    return samples


# ---- BitNet unpacking ----

from mlx_lm import load as mlx_load
from mlx_lm import generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.bitlinear_layers import BitLinear


def unpack_model(model):
    """Replace BitLinear with nn.Linear for differentiable forward pass."""
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                w = module.weight
                s = module.weight_scale
                w0 = (w & 3).astype(mx.bfloat16) - 1
                w1 = ((w >> 2) & 3).astype(mx.bfloat16) - 1
                w2 = ((w >> 4) & 3).astype(mx.bfloat16) - 1
                w3 = ((w >> 6) & 3).astype(mx.bfloat16) - 1
                unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:module.out_features]
                scale = s.astype(mx.bfloat16)
                unpacked = unpacked / scale if module.invert_weight_scales else unpacked * scale
                lin = nn.Linear(module.in_features, module.out_features,
                                bias=module.bias is not None)
                lin.weight = unpacked
                if module.bias is not None:
                    lin.bias = module.bias
                updates.append((key, lin))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    log(f"  Unpacked {count} BitLinear -> nn.Linear")
    return model


def load_model():
    """Load + unpack BitNet model."""
    model, tokenizer = mlx_load(MODEL_ID)
    return unpack_model(model), tokenizer


# ---- PPL measurement ----

def compute_ppl(model, tokenizer, texts, max_seq=MAX_SEQ_LENGTH):
    """Compute perplexity on a list of texts."""
    total_loss, total_tokens = 0.0, 0
    for text in texts:
        toks = tokenizer.encode(text)[:max_seq]
        if len(toks) < 4:
            continue
        x = mx.array(toks)[None, :]
        logits = model(x)
        mx.eval(logits)
        targets = x[:, 1:]
        lp = mx.log(mx.softmax(logits[:, :-1, :], axis=-1) + 1e-10)
        tlp = mx.take_along_axis(lp, targets[:, :, None], axis=-1).squeeze(-1)
        mx.eval(tlp)
        total_loss += -tlp.sum().item()
        total_tokens += targets.shape[1]
        del logits, lp, tlp, x
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


# ===========================================================================
# Phase 1: Ridge Router Calibration + Testing (all 24 domains)
# ===========================================================================

def phase_router():
    log("\n=== Phase 1: Ridge Router at N=24 ===")
    t0 = time.time()
    model, tokenizer = load_model()
    log_memory("model loaded")

    # Calibration data (train split)
    log(f"  Loading calibration data: {N_CAL} samples x {len(DOMAINS)} domains")
    cal_data = {}
    for domain in DOMAINS:
        cal_data[domain] = load_data(domain, "train", N_CAL)

    # Fit ridge router
    log("  Fitting ridge router (closed-form solve)...")
    W = fit_router(model, tokenizer, cal_data, lam=1.0, max_seq=MAX_SEQ_LENGTH)
    log(f"  Router fitted: W shape = {W.shape}")

    # Test on validation split
    log("  Testing routing accuracy on validation data...")
    correct, total = 0, 0
    per_domain = {}
    confusion = {}  # track where misrouted samples go

    for di, domain in enumerate(DOMAINS):
        dc, dt = 0, 0
        misrouted_to = {}
        for text in load_data(domain, "valid", N_TEST):
            pred = route(model, tokenizer, text, W, MAX_SEQ_LENGTH)
            if pred == di:
                dc += 1
                correct += 1
            else:
                pred_name = DOMAINS[pred]
                misrouted_to[pred_name] = misrouted_to.get(pred_name, 0) + 1
            dt += 1
            total += 1

        acc = dc / dt if dt > 0 else 0.0
        is_genuine = domain in GENUINE_DOMAINS
        tag = "genuine" if is_genuine else "slice"
        per_domain[domain] = {
            "accuracy": round(acc, 4),
            "correct": dc,
            "total": dt,
            "type": tag,
        }
        if misrouted_to:
            # Top-3 confusion targets
            top_conf = sorted(misrouted_to.items(), key=lambda x: -x[1])[:3]
            per_domain[domain]["top_confusions"] = {k: v for k, v in top_conf}
        log(f"    {domain:20s} [{tag:7s}]: {acc:5.1%} ({dc}/{dt})")

    overall_acc = correct / total if total > 0 else 0.0
    genuine_correct = sum(v["correct"] for d, v in per_domain.items() if v["type"] == "genuine")
    genuine_total = sum(v["total"] for d, v in per_domain.items() if v["type"] == "genuine")
    genuine_acc = genuine_correct / genuine_total if genuine_total > 0 else 0.0

    slice_correct = sum(v["correct"] for d, v in per_domain.items() if v["type"] == "slice")
    slice_total = sum(v["total"] for d, v in per_domain.items() if v["type"] == "slice")
    slice_acc = slice_correct / slice_total if slice_total > 0 else 0.0

    log(f"\n  Overall:  {overall_acc:.1%} ({correct}/{total})")
    log(f"  Genuine:  {genuine_acc:.1%} ({genuine_correct}/{genuine_total})")
    log(f"  Slice:    {slice_acc:.1%} ({slice_correct}/{slice_total})")

    # Save router weights for later phases
    np.save(str(EXPERIMENT_DIR / "router_W.npy"), np.array(W))

    elapsed = round(time.time() - t0, 1)
    log(f"  Phase 1 done in {elapsed}s")
    cleanup(model, tokenizer)

    return {
        "overall_accuracy": round(overall_acc, 4),
        "genuine_accuracy": round(genuine_acc, 4),
        "slice_accuracy": round(slice_acc, 4),
        "per_domain": per_domain,
        "n_cal": N_CAL,
        "n_test": N_TEST,
        "n_domains": len(DOMAINS),
        "elapsed_s": elapsed,
    }


# ===========================================================================
# Phase 2: Null-Space Gradient Preservation at N=24
# ===========================================================================

def _per_module_preservation(all_adapters, test_idx, module_key, layer_idx):
    """Compute null-space preservation for one module.

    For module m at layer l:
      1. Stack B-matrices from 23 prior adapters: M = (23*16, out_features)
      2. SVD: M = U S V^T, keep top-k right singular vectors V_k
      3. Preservation = 1 - ||B_test @ V_k||_F^2 / ||B_test||_F^2

    This avoids materializing the (out_features, out_features) projector.
    """
    bk = f"model.layers.{layer_idx}.{module_key}.lora_b"
    prior_indices = [i for i in range(len(all_adapters)) if i != test_idx]

    # Stack prior B-matrices: (23, 16, out_features) -> (23*16, out_features)
    prior_Bs = []
    for pi in prior_indices:
        if bk in all_adapters[pi]:
            prior_Bs.append(all_adapters[pi][bk].astype(mx.float32))
    if not prior_Bs:
        return None

    test_B = all_adapters[test_idx][bk].astype(mx.float32) if bk in all_adapters[test_idx] else None
    if test_B is None:
        return None

    # Stack: (n_priors * rank, out_features)
    M = mx.concatenate(prior_Bs, axis=0)  # (23*16, out_features)
    mx.eval(M)

    # SVD of M: (n_priors*16, out_features)
    # For n_priors*16 = 368 << out_features, this is efficient
    # U: (368, 368), S: (368,), Vt: (368, out_features)
    _, S, Vt = mx.linalg.svd(M, stream=mx.cpu)
    mx.eval(S, Vt)

    # Keep significant singular vectors (> 1% of max)
    s_max = S[0].item()
    k = int(mx.sum(S > 0.01 * s_max).item())
    k = max(k, 1)

    # V_k: (out_features, k)
    V_k = Vt[:k].T
    mx.eval(V_k)

    # Projection overlap: ||B_test @ V_k||_F^2 / ||B_test||_F^2
    proj = test_B @ V_k  # (16, k)
    mx.eval(proj)
    proj_norm_sq = mx.sum(proj * proj).item()
    test_norm_sq = mx.sum(test_B * test_B).item()

    preservation = 1.0 - proj_norm_sq / (test_norm_sq + 1e-12)

    del M, S, Vt, V_k, proj, prior_Bs
    return {
        "preservation": preservation,
        "effective_rank": k,
        "max_possible_rank": len(prior_indices) * 16,
    }


def phase_null_space():
    log("\n=== Phase 2: Null-Space Gradient Preservation ===")
    t0 = time.time()

    # Load all 24 adapter B-matrices (keep in memory -- each is ~42MB)
    log("  Loading all 24 adapter B-matrices...")
    all_adapters = []
    for di, domain in enumerate(DOMAINS):
        B = load_adapter(str(ADAPTERS_DIR / domain / "adapter.npz"))
        all_adapters.append(B)
    log(f"  Loaded {len(all_adapters)} adapters")

    # Representative modules for null-space analysis
    # Sample across layers (early, middle, late) and module types (attn, MLP)
    test_modules = [
        (0, "self_attn.q_proj"),   # layer 0, attention
        (0, "mlp.gate_proj"),      # layer 0, MLP
        (15, "self_attn.q_proj"),  # layer 15 (middle), attention
        (15, "mlp.gate_proj"),     # layer 15, MLP
        (29, "self_attn.q_proj"),  # layer 29 (last), attention
        (29, "mlp.gate_proj"),     # layer 29, MLP
        (7, "self_attn.v_proj"),   # layer 7, value projection
        (22, "mlp.down_proj"),     # layer 22, down projection
    ]

    # Test with 3 different test adapters (first, middle, last)
    test_indices = [0, 12, 23]  # agriculture, linguistics, sports

    results = {"per_module": {}, "per_test_adapter": {}}

    for test_idx in test_indices:
        test_domain = DOMAINS[test_idx]
        module_results = {}

        for layer_idx, module_key in test_modules:
            label = f"layer_{layer_idx}.{module_key}"
            res = _per_module_preservation(all_adapters, test_idx, module_key, layer_idx)
            if res is not None:
                module_results[label] = res
                log(f"    {test_domain:15s} | {label:30s}: "
                    f"pres={res['preservation']:.3f} "
                    f"(eff_rank={res['effective_rank']}/{res['max_possible_rank']})")

        if module_results:
            mean_pres = float(np.mean([r["preservation"] for r in module_results.values()]))
            results["per_test_adapter"][test_domain] = {
                "mean_preservation": round(mean_pres, 4),
                "modules": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                for kk, vv in v.items()}
                            for k, v in module_results.items()},
            }

    # Cumulative preservation curve: vary number of priors (1, 2, 4, 8, 12, 16, 20, 23)
    log("\n  Computing cumulative preservation curve (layer 15, q_proj)...")
    test_idx = 23  # sports
    bk = "model.layers.15.self_attn.q_proj.lora_b"
    test_B = all_adapters[test_idx][bk].astype(mx.float32)
    test_norm_sq = mx.sum(test_B * test_B).item()
    cumulative = []

    for n_priors in [1, 2, 4, 8, 12, 16, 20, 23]:
        prior_Bs = [all_adapters[i][bk].astype(mx.float32) for i in range(n_priors)]
        M = mx.concatenate(prior_Bs, axis=0)
        mx.eval(M)
        _, S, Vt = mx.linalg.svd(M, stream=mx.cpu)
        mx.eval(S, Vt)
        s_max = S[0].item()
        k = max(int(mx.sum(S > 0.01 * s_max).item()), 1)
        V_k = Vt[:k].T
        mx.eval(V_k)
        proj = test_B @ V_k
        mx.eval(proj)
        proj_norm_sq = mx.sum(proj * proj).item()
        pres = 1.0 - proj_norm_sq / (test_norm_sq + 1e-12)
        cumulative.append({
            "n_priors": n_priors,
            "preservation": round(pres, 4),
            "effective_rank": k,
        })
        log(f"    n_priors={n_priors:2d}: preservation = {pres:.1%} (eff_rank={k})")
        del M, S, Vt, V_k, proj, prior_Bs
        gc.collect()
        mx.clear_cache()

    # Overall mean preservation
    all_pres = [v["mean_preservation"] for v in results["per_test_adapter"].values()]
    mean_pres = float(np.mean(all_pres)) if all_pres else 0.0

    elapsed = round(time.time() - t0, 1)
    log(f"\n  Mean preservation across test adapters: {mean_pres:.1%}")
    log(f"  Theoretical bound (orthogonal, rank-additive): "
        f"{(2560 - 368) / 2560:.1%}")
    log(f"  Phase 2 done in {elapsed}s")

    cleanup(*all_adapters)

    return {
        "per_test_adapter": results["per_test_adapter"],
        "cumulative_curve": cumulative,
        "mean_preservation": round(mean_pres, 4),
        "theoretical_bound": round((2560 - 368) / 2560, 4),
        "elapsed_s": elapsed,
    }


# ===========================================================================
# Phase 3: PPL comparison (base vs single-adapter vs composed)
# ===========================================================================

def phase_ppl():
    log("\n=== Phase 3: PPL Comparison ===")
    t0 = time.time()

    skeleton = load_frozen_A(str(ADAPTERS_DIR / "grassmannian_skeleton_n24.npz"))
    W = mx.array(np.load(str(EXPERIMENT_DIR / "router_W.npy")))

    # We test a subset of domains for PPL (memory/time constraint)
    # Focus on genuine domains + a few slice-based
    test_domains = ["medical", "code", "math", "legal", "finance",
                    "science", "cooking", "creative_writing"]
    val_data = {d: load_data(d, "valid", N_PPL) for d in test_domains}

    results = {"base": {}, "single": {}, "routed_single": {}, "composed_top2": {},
               "worst_single_ppl": 0.0}

    # ---- Base PPL ----
    log("  Computing base PPL...")
    model, tok = load_model()
    for domain in test_domains:
        ppl = compute_ppl(model, tok, val_data[domain])
        results["base"][domain] = round(ppl, 3)
        log(f"    base/{domain}: {ppl:.3f}")
    cleanup(model, tok)

    # ---- Single-adapter PPL (oracle: correct adapter for each domain) ----
    log("  Computing single-adapter PPL (oracle)...")
    for di, domain in enumerate(test_domains):
        model, tok = load_model()
        domain_idx = DOMAINS.index(domain)
        adapter_B = load_adapter(str(ADAPTERS_DIR / domain / "adapter.npz"))
        count = attach_adapter(model, skeleton, adapter_B, domain_idx, LORA_SCALE)
        ppl = compute_ppl(model, tok, val_data[domain])
        results["single"][domain] = round(ppl, 3)
        log(f"    single/{domain}: {ppl:.3f} ({count} modules attached)")
        cleanup(model, tok, adapter_B)

    results["worst_single_ppl"] = max(results["single"].values())
    log(f"  Worst single-adapter PPL: {results['worst_single_ppl']}")

    # ---- Routed single-adapter PPL (top-1 from router) ----
    log("  Computing routed single-adapter PPL (top-1)...")
    for domain in test_domains:
        model, tok = load_model()
        # Route using first sample
        routed_idx = route(model, tok, val_data[domain][0], W, MAX_SEQ_LENGTH)
        routed_name = DOMAINS[routed_idx]
        adapter_B = load_adapter(str(ADAPTERS_DIR / routed_name / "adapter.npz"))
        count = attach_adapter(model, skeleton, adapter_B, routed_idx, LORA_SCALE)
        ppl = compute_ppl(model, tok, val_data[domain])
        results["routed_single"][domain] = {
            "ppl": round(ppl, 3),
            "routed_to": routed_name,
            "correct": routed_name == domain,
        }
        log(f"    routed/{domain} -> {routed_name}: {ppl:.3f}"
            f" {'(correct)' if routed_name == domain else '(MISROUTED)'}")
        cleanup(model, tok, adapter_B)

    # ---- Composed top-2 PPL ----
    log("  Computing composed top-2 PPL...")
    for domain in test_domains:
        model, tok = load_model()
        # Get top-2 domains from router scores
        h = encode(model, mx.array(tok.encode(val_data[domain][0])[:MAX_SEQ_LENGTH])[None, :])
        scores = (h @ W).squeeze(0)
        mx.eval(scores)
        top2_indices = mx.argsort(scores)[-2:].tolist()[::-1]  # descending
        top2_names = [DOMAINS[i] for i in top2_indices]

        # Load and compose top-2 adapters
        adapter_Bs = []
        for idx in top2_indices:
            adapter_Bs.append(load_adapter(str(ADAPTERS_DIR / DOMAINS[idx] / "adapter.npz")))

        composed_B = compose_adapters(adapter_Bs, weights=[0.7, 0.3])

        # Attach composed adapter using first domain's index
        count = attach_adapter(model, skeleton, composed_B, top2_indices[0], LORA_SCALE)
        ppl = compute_ppl(model, tok, val_data[domain])

        results["composed_top2"][domain] = {
            "ppl": round(ppl, 3),
            "top2": top2_names,
        }
        log(f"    composed/{domain} -> {top2_names}: {ppl:.3f}")
        cleanup(model, tok, composed_B, *adapter_Bs)

    elapsed = round(time.time() - t0, 1)
    log(f"  Phase 3 done in {elapsed}s")

    return results


# ===========================================================================
# Phase 4: Orthogonality check at N=24
# ===========================================================================

def phase_orthogonality():
    log("\n=== Phase 4: Orthogonality Check ===")
    t0 = time.time()

    # Load all adapters and compute pairwise B-matrix cosines
    adapters = {}
    for di, domain in enumerate(DOMAINS):
        B = load_adapter(str(ADAPTERS_DIR / domain / "adapter.npz"))
        # Flatten all B-matrices into a single vector
        parts = []
        for key in sorted(B.keys()):
            parts.append(B[key].reshape(-1).astype(mx.float32))
        adapters[domain] = mx.concatenate(parts)
        mx.eval(adapters[domain])

    # Pairwise cosines
    cosines = []
    for i, d1 in enumerate(DOMAINS):
        for j, d2 in enumerate(DOMAINS):
            if j <= i:
                continue
            v1 = adapters[d1]
            v2 = adapters[d2]
            cos = (mx.sum(v1 * v2) / (mx.linalg.norm(v1) * mx.linalg.norm(v2) + 1e-10)).item()
            cosines.append({"pair": f"{d1}-{d2}", "cosine": round(abs(cos), 6)})

    cos_values = [c["cosine"] for c in cosines]
    mean_cos = float(np.mean(cos_values))
    max_cos = float(np.max(cos_values))
    top5 = sorted(cosines, key=lambda x: -x["cosine"])[:5]

    elapsed = round(time.time() - t0, 1)
    log(f"  Mean |cos|: {mean_cos:.6f}")
    log(f"  Max  |cos|: {max_cos:.6f}")
    log(f"  Top-5 most similar pairs:")
    for c in top5:
        log(f"    {c['pair']}: {c['cosine']:.6f}")
    log(f"  Phase 4 done in {elapsed}s")

    cleanup(*list(adapters.values()))

    return {
        "mean_abs_cosine": round(mean_cos, 6),
        "max_abs_cosine": round(max_cos, 6),
        "top5_pairs": top5,
        "n_pairs": len(cosines),
        "elapsed_s": elapsed,
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log("Pierre v3: N=24 Scaling Experiment")
    log("=" * 60)
    log(f"Domains: {len(DOMAINS)}")
    log(f"Model: {MODEL_ID}")
    log(f"Platform: {mx.device_info().get('architecture', 'unknown')}")
    log_memory("start")

    # Phase 1: Router
    r1 = phase_router()
    log_memory("after-router")

    # Phase 2: Null-space
    r2 = phase_null_space()
    log_memory("after-null-space")

    # Phase 3: PPL
    r3 = phase_ppl()
    log_memory("after-ppl")

    # Phase 4: Orthogonality
    r4 = phase_orthogonality()
    log_memory("after-orthogonality")

    # Kill criteria assessment
    k721_pass = r1["overall_accuracy"] >= 0.50
    k722_pass = r2["mean_preservation"] >= 0.50
    worst_single = r3["worst_single_ppl"]
    # Check composed PPL against worst single
    composed_ppls = [v["ppl"] for v in r3["composed_top2"].values()]
    max_composed = max(composed_ppls) if composed_ppls else float("inf")
    k723_pass = max_composed <= 2.0 * worst_single

    results = {
        "experiment": "pierre_v3_n24_scaling",
        "total_time_s": round(time.time() - t0, 1),
        "routing": r1,
        "null_space": r2,
        "ppl": r3,
        "orthogonality": r4,
        "kill_criteria": {
            "K721": {
                "description": "Ridge router accuracy >= 50% at N=24",
                "pass": k721_pass,
                "value": r1["overall_accuracy"],
                "threshold": 0.50,
                "random_baseline": round(1.0 / len(DOMAINS), 4),
            },
            "K722": {
                "description": "Null-space gradient preservation >= 50% at N=24",
                "pass": k722_pass,
                "value": r2["mean_preservation"],
                "threshold": 0.50,
                "theoretical": r2["theoretical_bound"],
            },
            "K723": {
                "description": "Composed PPL <= 2x worst single-adapter PPL",
                "pass": k723_pass,
                "value": round(max_composed, 3),
                "threshold": round(2.0 * worst_single, 3),
                "worst_single_ppl": worst_single,
            },
        },
        "all_pass": k721_pass and k722_pass and k723_pass,
    }

    log("\n" + "=" * 60)
    log("KILL CRITERIA ASSESSMENT")
    log("-" * 60)
    for k, v in results["kill_criteria"].items():
        status = "PASS" if v["pass"] else "FAIL"
        log(f"  {k}: {status} (value={v['value']}, threshold={v['threshold']})")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'} in {results['total_time_s']}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
