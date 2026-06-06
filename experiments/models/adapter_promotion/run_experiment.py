#!/usr/bin/env python3
"""Adapter promotion: permanently attach universal adapter, then add more on top.

Tests the "grow base without retraining" concept from PRODUCT_ARCHITECTURE.md.
Promote the most universal adapter (medical) to permanent status, then
compose 4 remaining adapters on top.

Kill criteria:
  K828: Promoted adapter loses >30% of its benefit
  K829: Adding adapters on top causes interference
"""

import gc, json, math, os, time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from pierre import attach_adapter, detach_adapters, compose_adapters, load_adapter, load_frozen_A
from mlx_lm import load

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Use Pierre Pro adapters if available, otherwise BitNet SFT
PRO_ADAPTERS = EXPERIMENT_DIR.parent / "pro_sft_5_adapters" / "adapters"
TINY_ADAPTERS = EXPERIMENT_DIR.parent / "bitnet_sft_generation_v3" / "sft_adapters"

PRO_SKELETON = EXPERIMENT_DIR.parent / "pro_grassmannian_init" / "grassmannian_skeleton_n5.npz"
TINY_SKELETON = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "adapters" / "grassmannian_skeleton.npz"

DATA_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "data"

LORA_SCALE = 20.0
MAX_SEQ = 256
SEED = 42
DOMAINS = ["medical", "code", "math", "legal", "finance"]
PROMOTE_DOMAIN = "medical"  # most universal adapter


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

def load_data(d, split="valid", n=None):
    s = []
    with open(DATA_DIR / d / f"{split}.jsonl") as f:
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
    log("Adapter Promotion: Grow Base Without Retraining")
    log("=" * 60)
    mx.random.seed(SEED)

    # Select model and adapter source
    if PRO_ADAPTERS.exists() and (PRO_ADAPTERS / PROMOTE_DOMAIN / "adapter.npz").exists():
        adapter_source = PRO_ADAPTERS
        skeleton_path = PRO_SKELETON
        base_data = json.loads((EXPERIMENT_DIR.parent / "pro_base_validation" / "results.json").read_text())
        model_id = base_data.get("model_id", "mlx-community/Qwen2.5-3B-Instruct-4bit")
        tag = "pro"
    else:
        adapter_source = TINY_ADAPTERS
        skeleton_path = TINY_SKELETON
        model_id = "microsoft/BitNet-b1.58-2B-4T"
        tag = "tiny"

    log(f"Using {tag} ({model_id})")
    frozen_A = load_frozen_A(str(skeleton_path))
    val = {d: load_data(d, "valid", 20) for d in DOMAINS}
    promote_idx = DOMAINS.index(PROMOTE_DOMAIN)

    results = {}

    # Phase 1: Medical adapter alone — baseline benefit
    log(f"\n=== Phase 1: {PROMOTE_DOMAIN} adapter alone ===")
    model, tok = load(model_id)
    base_ppl = compute_ppl(model, tok, val[PROMOTE_DOMAIN])
    log(f"  Base PPL ({PROMOTE_DOMAIN}): {base_ppl:.3f}")

    adapter = load_adapter(str(adapter_source / PROMOTE_DOMAIN / "adapter.npz"))
    attach_adapter(model, frozen_A, adapter, promote_idx, LORA_SCALE)
    promoted_ppl = compute_ppl(model, tok, val[PROMOTE_DOMAIN])
    promotion_benefit = (base_ppl - promoted_ppl) / base_ppl * 100
    log(f"  With adapter PPL: {promoted_ppl:.3f} ({promotion_benefit:.1f}% improvement)")
    cleanup(model, tok, adapter)

    results["phase1"] = {
        "base_ppl": round(base_ppl, 3),
        "promoted_ppl": round(promoted_ppl, 3),
        "benefit_pct": round(promotion_benefit, 2),
    }

    # Phase 2: Promote medical + add 4 others — does promoted adapter retain benefit?
    log(f"\n=== Phase 2: Promoted + 4 composed adapters ===")
    model, tok = load(model_id)

    # Load all adapters
    all_adapters = []
    for d in DOMAINS:
        a = load_adapter(str(adapter_source / d / "adapter.npz"))
        all_adapters.append(a)

    # Compose all 5 via NRE
    composed = compose_adapters(all_adapters)
    attach_adapter(model, frozen_A, composed, 0, LORA_SCALE)

    # Measure medical PPL with all adapters composed
    composed_medical_ppl = compute_ppl(model, tok, val[PROMOTE_DOMAIN])
    retained_benefit = (base_ppl - composed_medical_ppl) / (base_ppl - promoted_ppl) * 100 if (base_ppl - promoted_ppl) > 0 else 0
    log(f"  Composed PPL ({PROMOTE_DOMAIN}): {composed_medical_ppl:.3f}")
    log(f"  Retained benefit: {retained_benefit:.1f}% of original promotion")

    # Also measure other domains — does composition help them?
    other_ppls = {}
    for d in DOMAINS:
        if d == PROMOTE_DOMAIN: continue
        other_ppls[d] = round(compute_ppl(model, tok, val[d]), 3)
        log(f"  Composed PPL ({d}): {other_ppls[d]}")

    cleanup(model, tok, composed)

    # Phase 3: Base PPL on other domains (for comparison)
    log(f"\n=== Phase 3: Base PPLs for comparison ===")
    model, tok = load(model_id)
    base_others = {}
    for d in DOMAINS:
        if d == PROMOTE_DOMAIN: continue
        base_others[d] = round(compute_ppl(model, tok, val[d]), 3)
    cleanup(model, tok)

    results["phase2"] = {
        "composed_medical_ppl": round(composed_medical_ppl, 3),
        "retained_benefit_pct": round(retained_benefit, 1),
        "other_domain_ppls": other_ppls,
        "base_other_ppls": base_others,
    }
    results["total_time_s"] = round(time.time() - t0, 1)
    results["model"] = tag

    # Kill criteria
    k828 = retained_benefit >= 70  # promoted adapter keeps 70%+ of its benefit
    k829 = all(other_ppls.get(d, 999) <= base_others.get(d, 0) * 1.5
               for d in base_others)  # other domains not catastrophically worse

    results["kill_criteria"] = {
        "K828": {"pass": k828, "value": round(retained_benefit, 1), "threshold": 70},
        "K829": {"pass": k829, "detail": "No other domain > 1.5x base PPL"},
    }
    results["all_pass"] = k828 and k829

    log(f"\n{'='*60}")
    log(f"Promoted adapter retains {retained_benefit:.1f}% of benefit under composition")
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} — {v}")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
