#!/usr/bin/env python3
"""N=24 composition proof: ridge router + Grassmannian orthogonality at scale.

The unique contribution: prove composable domain adapters work at N=24
where ALL previous routing methods collapsed (~40% accuracy).

Uses Pierre's stable API (Runtime LoRA, ridge router, NRE merge).

Tests:
  1. Ridge router accuracy at N=24 (Finding #276: 96% at N=5)
  2. Pairwise adapter cosine similarity (Finding #3: cos < 0.001 at N=5)
  3. Composed PPL: NRE merge of top-2 and top-5 adapters
  4. Per-domain PPL with routed adapter

Kill criteria:
  K753: Ridge router accuracy < 50% at N=24 (random = 4.2%)
  K754: Mean pairwise adapter cosine > 0.1
  K755: Composed N=5 PPL > 3x worst single-adapter PPL
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

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from pierre import (
    attach_adapter, detach_adapters, compose_adapters,
    encode, fit_router, route,
    load_adapter, load_frozen_A,
)
from mlx_lm import load

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Try SFT adapters first (from experiment 1), fall back to NTP
SFT_DIR = EXPERIMENT_DIR.parent / "sft_24_domain_adapters" / "sft_adapters"
NTP_DIR = EXPERIMENT_DIR.parent / "real_data_25_domain_adapters" / "adapters"
SKELETON_PATH = NTP_DIR / "grassmannian_skeleton_n24.npz"
DATA_DIR = EXPERIMENT_DIR.parent / "real_data_25_domain_adapters" / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_SCALE = 20.0
MAX_SEQ = 256
SEED = 42
N_CAL = 50
N_TEST = 10


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

def log(m): print(m, flush=True)
def cleanup(*o):
    for x in o: del x
    gc.collect(); mx.clear_cache(); mx.reset_peak_memory()

def load_data(domain, split="valid", n=None):
    path = DATA_DIR / domain / f"{split}.jsonl"
    if not path.exists(): return []
    s = []
    with open(path) as f:
        for l in f:
            s.append(json.loads(l)["text"])
            if n and len(s) >= n: break
    return s

def compute_ppl(model, tok, texts):
    loss, n = 0.0, 0
    for text in texts:
        toks = tok.encode(text)[:MAX_SEQ]
        if len(toks) < 4: continue
        x = mx.array(toks)[None, :]
        logits = model(x); mx.eval(logits)
        targets = x[:, 1:]
        lp = mx.log(mx.softmax(logits[:, :-1, :], axis=-1) + 1e-10)
        tlp = mx.take_along_axis(lp, targets[:,:,None], axis=-1).squeeze(-1)
        mx.eval(tlp)
        loss += -tlp.sum().item(); n += targets.shape[1]
        del logits, lp, tlp, x
    return math.exp(loss / n) if n else float('inf')


def main():
    t0 = time.time()
    log("N=24 Composition Proof")
    log("=" * 60)
    mx.random.seed(SEED)

    # Discover domains from ADAPTER directory (24 domains), NOT data dir (25, includes real_estate)
    # The skeleton's domain indices (domain_0 through domain_23) correspond to alphabetically
    # sorted adapter domains. Using data dir would give 25 domains and shift all indices.
    if SFT_DIR.exists() and any((SFT_DIR / d.name / "adapter.npz").exists() for d in SFT_DIR.iterdir() if d.is_dir()):
        adapter_dir = SFT_DIR
        adapter_type = "SFT"
    else:
        adapter_dir = NTP_DIR
        adapter_type = "NTP"

    domains = sorted([d.name for d in adapter_dir.iterdir() if d.is_dir() and (d / "adapter.npz").exists()])
    N = len(domains)
    log(f"Domains ({N}): {domains}")
    log(f"Using {adapter_type} adapters from {adapter_dir}")

    frozen_A = load_frozen_A(str(SKELETON_PATH))
    log(f"Loaded Grassmannian skeleton: {len(frozen_A)} keys")

    # ── Phase 1: Ridge router at N=24 ────────────────────────────────────
    log(f"\n=== Phase 1: Ridge Router at N={N} ===")

    model, tok = load(MODEL_ID)
    cal_data = {}
    for d in domains:
        texts = load_data(d, "train", N_CAL)
        if texts:
            cal_data[d] = texts

    log(f"Calibrating on {len(cal_data)} domains, {N_CAL} samples each...")
    W = fit_router(model, tok, cal_data, lam=1.0, max_seq=MAX_SEQ)
    log(f"Router: H={W.shape[0]}, D={W.shape[1]}")

    # Test routing accuracy with confusion tracking
    correct, total = 0, 0
    per_domain_acc = {}
    confusion = {}  # domain -> {predicted_domain: count}
    for di, d in enumerate(domains):
        test_texts = load_data(d, "valid", N_TEST)
        dc = 0
        domain_confusion = {}
        for text in test_texts:
            pred = route(model, tok, text, W, MAX_SEQ)
            pred_name = domains[pred] if pred < len(domains) else f"unk_{pred}"
            if pred == di:
                dc += 1; correct += 1
            else:
                domain_confusion[pred_name] = domain_confusion.get(pred_name, 0) + 1
            total += 1
        acc = dc / max(len(test_texts), 1)
        per_domain_acc[d] = round(acc, 3)
        if domain_confusion:
            confusion[d] = domain_confusion

    overall_acc = correct / total if total else 0.0
    log(f"Routing accuracy: {overall_acc:.1%} ({correct}/{total})")

    # Show worst 5 domains with confusion targets
    sorted_acc = sorted(per_domain_acc.items(), key=lambda x: x[1])
    log("  Worst 5:")
    for d, a in sorted_acc[:5]:
        conf_str = ""
        if d in confusion:
            top_conf = sorted(confusion[d].items(), key=lambda x: -x[1])[:3]
            conf_str = f"  confused with: {', '.join(f'{c}({n})' for c, n in top_conf)}"
        log(f"    {d}: {a:.1%}{conf_str}")
    log("  Best 5:")
    for d, a in sorted_acc[-5:]:
        log(f"    {d}: {a:.1%}")

    np.save(str(EXPERIMENT_DIR / "router_W.npy"), np.array(W))
    cleanup(model, tok)

    # ── Phase 2: Pairwise adapter cosine ─────────────────────────────────
    log(f"\n=== Phase 2: Pairwise Adapter Cosine ===")

    # Load all adapter B-matrices and compute pairwise cosine
    adapters = {}
    for di, d in enumerate(domains):
        path = adapter_dir / d / "adapter.npz"
        if path.exists():
            adapters[d] = load_adapter(str(path))

    # Flatten each adapter to a single vector for cosine computation
    def flatten_adapter(adapter_dict):
        vecs = []
        for k in sorted(adapter_dict.keys()):
            vecs.append(adapter_dict[k].astype(mx.float32).reshape(-1))
        return mx.concatenate(vecs)

    adapter_vecs = {}
    for d, a in adapters.items():
        v = flatten_adapter(a)
        mx.eval(v)
        adapter_vecs[d] = v

    # Compute pairwise cosine
    cos_values = []
    domain_list = sorted(adapter_vecs.keys())
    for i in range(len(domain_list)):
        for j in range(i+1, len(domain_list)):
            vi = adapter_vecs[domain_list[i]]
            vj = adapter_vecs[domain_list[j]]
            cos = (mx.sum(vi * vj) / (mx.linalg.norm(vi) * mx.linalg.norm(vj) + 1e-8)).item()
            cos_values.append(abs(cos))

    mean_cos = float(np.mean(cos_values))
    max_cos = float(np.max(cos_values))
    median_cos = float(np.median(cos_values))
    p95_cos = float(np.percentile(cos_values, 95))
    log(f"Pairwise |cosine|: mean={mean_cos:.6f}, median={median_cos:.6f}, p95={p95_cos:.6f}, max={max_cos:.6f}")
    log(f"Number of pairs: {len(cos_values)}")
    # LEARNINGS.md gate: if mean cos > 0.5, format dominance persists and routing will fail
    if mean_cos > 0.5:
        log("  WARNING: B-matrix inter-cosine > 0.5 — format dominance likely, routing may fail")
    elif mean_cos > 0.2:
        log("  CAUTION: B-matrix inter-cosine > 0.2 — moderate format correlation")
    else:
        log(f"  GOOD: B-matrix inter-cosine {mean_cos:.4f} < 0.2 — adapters sufficiently differentiated")

    # ── Phase 3: Per-domain PPL with routed adapter ──────────────────────
    log(f"\n=== Phase 3: Per-Domain PPL ===")

    W = mx.array(np.load(str(EXPERIMENT_DIR / "router_W.npy")))
    base_ppls = {}
    adapter_ppls = {}
    routed_ppls = {}

    # Sample 8 diverse domains for PPL evaluation (mix of strong/weak from Finding #297)
    preferred = ["medical", "code", "math", "legal", "finance", "music", "psychology", "engineering"]
    sample_domains = [d for d in preferred if d in adapters][:8]
    if len(sample_domains) < 8:
        sample_domains += [d for d in domains if d not in sample_domains][:8 - len(sample_domains)]
    model, tok = load(MODEL_ID)
    for d in sample_domains:
        val = load_data(d, "valid", N_TEST)
        if val:
            base_ppls[d] = round(compute_ppl(model, tok, val), 3)
            log(f"  base/{d}: {base_ppls[d]}")
    cleanup(model, tok)

    # Single-adapter PPL (use correct skeleton index, not enumerate index)
    for d in sample_domains:
        if d not in adapters: continue
        skel_idx = domains.index(d)
        model, tok = load(MODEL_ID)
        n = attach_adapter(model, frozen_A, adapters[d], skel_idx, LORA_SCALE)
        val = load_data(d, "valid", N_TEST)
        if val:
            adapter_ppls[d] = round(compute_ppl(model, tok, val), 3)
            log(f"  adapter/{d}: {adapter_ppls[d]} ({n} modules, skel_idx={skel_idx})")
        cleanup(model, tok)

    # Routed PPL
    for d in sample_domains:
        model, tok = load(MODEL_ID)
        val = load_data(d, "valid", N_TEST)
        if not val: continue
        ri = route(model, tok, val[0], W, MAX_SEQ)
        rd = domains[ri]
        if rd in adapters:
            attach_adapter(model, frozen_A, adapters[rd], ri, LORA_SCALE)
            routed_ppls[d] = round(compute_ppl(model, tok, val), 3)
            log(f"  routed/{d}: {routed_ppls[d]} (→{rd})")
        cleanup(model, tok)

    # ── Phase 4: NRE Composition (top-2, top-5) ─────────────────────────
    log(f"\n=== Phase 4: NRE Composition ===")

    # Pick 5 diverse domains for composition test
    comp_domains = ["medical", "code", "math", "legal", "finance"]
    comp_domains = [d for d in comp_domains if d in adapters]
    if len(comp_domains) < 5:
        comp_domains = list(adapters.keys())[:5]

    comp_results = {}
    for k in [2, 5]:
        subset = comp_domains[:k]
        subset_adapters = [adapters[d] for d in subset]
        composed = compose_adapters(subset_adapters)

        # Use first domain's skeleton index for the composed adapter's A-matrix
        # (same approach as pierre_v3_n24_scaling: composed B projected through top-1 A)
        comp_domain_idx = domains.index(subset[0])

        # Measure composed PPL on each domain in the subset
        ppls = {}
        for di, d in enumerate(subset):
            model, tok = load(MODEL_ID)
            attach_adapter(model, frozen_A, composed, comp_domain_idx, LORA_SCALE)
            val = load_data(d, "valid", N_TEST)
            if val:
                ppls[d] = round(compute_ppl(model, tok, val), 3)
            cleanup(model, tok)

        comp_results[f"top_{k}"] = {
            "domains": subset,
            "ppls": ppls,
            "mean_ppl": round(float(np.mean(list(ppls.values()))), 3) if ppls else None,
        }
        log(f"  Composed top-{k}: {ppls}")

    # ── Results ──────────────────────────────────────────────────────────
    total_time = time.time() - t0

    results = {
        "experiment": "n24_composition_proof",
        "total_time_s": round(total_time, 1),
        "n_domains": N,
        "adapter_type": adapter_type,
        "routing": {
            "accuracy": round(overall_acc, 4),
            "per_domain": per_domain_acc,
            "confusion": confusion,
        },
        "orthogonality": {
            "mean_cosine": round(mean_cos, 6),
            "median_cosine": round(median_cos, 6),
            "p95_cosine": round(p95_cos, 6),
            "max_cosine": round(max_cos, 6),
            "n_pairs": len(cos_values),
        },
        "ppl": {
            "base": base_ppls,
            "single_adapter": adapter_ppls,
            "routed": routed_ppls,
        },
        "composition": comp_results,
    }

    # Kill criteria
    k753 = overall_acc >= 0.50
    k754 = mean_cos <= 0.10
    worst_single = max(adapter_ppls.values()) if adapter_ppls else float('inf')
    comp5_mean = comp_results.get("top_5", {}).get("mean_ppl", float('inf'))
    k755 = comp5_mean <= 3 * worst_single if worst_single < float('inf') else False

    results["kill_criteria"] = {
        "K753": {"pass": k753, "value": round(overall_acc, 4), "threshold": 0.50},
        "K754": {"pass": k754, "value": round(mean_cos, 6), "threshold": 0.10},
        "K755": {"pass": k755, "value": comp5_mean, "threshold": round(3 * worst_single, 3)},
    }
    results["all_pass"] = k753 and k754 and k755

    log(f"\n{'='*60}")
    log(f"Router accuracy: {overall_acc:.1%} at N={N}")
    log(f"Mean |cosine|: {mean_cos:.6f}")
    log(f"Kill criteria:")
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} — {v}")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'} in {total_time/60:.1f} min")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
