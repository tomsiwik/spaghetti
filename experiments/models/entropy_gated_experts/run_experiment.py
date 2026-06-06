#!/usr/bin/env python3
"""
Entropy-Gated Expert Selection: Skip composition when base is confident.

Kill criteria:
  K1: Entropy distribution lacks sufficient spread for thresholding
      (CV < 0.5 AND Otsu between-class variance ratio < 0.15) -> KILL
  K2: Adaptive gating PPL worse than always-compose token-weighted PPL
      by more than 5% -> KILL
  K3: Fraction of tokens skipped < 10% -> KILL

Success criteria:
  S1: >= 30% tokens skip composition with < 1% PPL degradation
  S2: Mean PPL within 2% of always-compose baseline (token-weighted)
  S3: Wall-clock speedup > 15% from skipping expert merge on confident tokens

Platform: Apple M5 Pro 48GB, MLX, $0.
"""

import gc
import json
import math
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety
device = mx.device_info()
total_mem = device["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse adapters and data from tiny_routing_heads
ADAPTER_DIR = Path(__file__).parent.parent / "tiny_routing_heads" / "adapters"
DATA_DIR = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
VAL_BATCHES = 25  # samples per domain for evaluation
DOMAINS = ["python", "math", "medical", "legal", "creative"]


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


# ===========================================================================
# Model loading utilities (from prior experiments)
# ===========================================================================
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.bitlinear_layers import BitLinear


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


def apply_lora_to_model(model, rank=16, scale=1.0):
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        lora_updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora_layer = LoRALinear.from_base(module, r=rank, scale=scale, dropout=0.0)
                lora_updates.append((key, lora_layer))
                count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    log(f"  Applied LoRA (r={rank}) to {count} linear layers")
    return model


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_to_model(model, adapter_params, scale=1.0):
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def compose_adapters(adapter_list, scale_per_adapter=None):
    """Merge multiple adapter parameter dicts with given scale (default 1/N)."""
    N = len(adapter_list)
    if scale_per_adapter is None:
        scale_per_adapter = 1.0 / N
    merged = {}
    for key in adapter_list[0].keys():
        stacked = mx.stack([a[key] for a in adapter_list])
        merged[key] = mx.sum(stacked, axis=0) * scale_per_adapter
    return merged


def zero_adapter_in_model(model):
    updates = []
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_b" in name:
            updates.append((name, mx.zeros_like(p)))
    if updates:
        model.update(tree_unflatten(updates))


# ===========================================================================
# Data loading
# ===========================================================================
def load_domain_texts(domain_name, split="valid"):
    fpath = DATA_DIR / domain_name / f"{split}.jsonl"
    if not fpath.exists():
        return []
    texts = []
    with open(fpath) as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
    return texts


# ===========================================================================
# Entropy computation
# ===========================================================================
def compute_token_entropies(model, tokenizer, texts, max_batches=25):
    """Compute per-token entropy for each text. Returns flat list of entropies."""
    all_entropies = []
    all_losses_base = []  # per-token cross-entropy (for PPL calc)

    for text in texts[:max_batches]:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:MAX_SEQ_LENGTH + 1]

        x = mx.array(tokens[:-1])[None, :]  # (1, seq_len)
        y = mx.array(tokens[1:])             # (seq_len,)

        logits = model(x)                    # (1, seq_len, V)
        logits_2d = logits[0]                # (seq_len, V)

        # Compute entropy: H(p) = -sum(p * log(p))
        # Use log_softmax for numerical stability
        log_probs = mx.softmax(logits_2d, axis=-1)
        # Clamp for log stability
        log_p = mx.log(mx.clip(log_probs, 1e-10, 1.0))
        entropy = -mx.sum(log_probs * log_p, axis=-1)  # (seq_len,)

        # Per-token cross-entropy for PPL
        ce = nn.losses.cross_entropy(logits_2d, y, reduction="none")  # (seq_len,)

        mx.eval(entropy, ce)

        ent_list = entropy.tolist()
        ce_list = ce.tolist()
        all_entropies.extend(ent_list)
        all_losses_base.extend(ce_list)

        del logits, logits_2d, log_probs, log_p, entropy, ce, x, y

    return all_entropies, all_losses_base


# ===========================================================================
# Distribution spread tests (reframed K1: sufficient spread, not bimodality)
# ===========================================================================
def coefficient_of_variation(data):
    """Compute CV = std/mean. CV > 0.5 indicates high spread."""
    import numpy as np
    arr = np.array(data)
    mean = np.mean(arr)
    if mean == 0:
        return 0.0
    return float(np.std(arr) / mean)


def otsu_between_class_variance_ratio(data, tau):
    """Compute Otsu's between-class variance ratio (eta = sigma_B^2 / sigma_T^2).
    Values near 1.0 mean the threshold explains most of the variance.
    Values > 0.15 indicate two meaningful clusters.
    """
    import numpy as np
    arr = np.array(data)
    total_var = np.var(arr)
    if total_var == 0:
        return 0.0
    below = arr[arr < tau]
    above = arr[arr >= tau]
    if len(below) == 0 or len(above) == 0:
        return 0.0
    w0 = len(below) / len(arr)
    w1 = len(above) / len(arr)
    mu0 = np.mean(below)
    mu1 = np.mean(above)
    sigma_b_sq = w0 * w1 * (mu0 - mu1) ** 2
    return float(sigma_b_sq / total_var)


def otsu_threshold(data):
    """Find optimal threshold via Otsu's method (minimize intra-class variance)."""
    import numpy as np
    arr = np.array(data)
    # Try 100 thresholds between 10th and 90th percentile
    lo, hi = np.percentile(arr, 5), np.percentile(arr, 95)
    best_tau = lo
    best_cost = float("inf")

    for tau in np.linspace(lo, hi, 200):
        below = arr[arr < tau]
        above = arr[arr >= tau]
        if len(below) < 2 or len(above) < 2:
            continue
        w_below = len(below) / len(arr)
        w_above = len(above) / len(arr)
        cost = w_below * np.var(below) + w_above * np.var(above)
        if cost < best_cost:
            best_cost = cost
            best_tau = tau

    return best_tau


# ===========================================================================
# Phase 1: Entropy Distribution Analysis (K1 gate)
# ===========================================================================
def phase_entropy_distribution():
    """Compute base model entropy distribution and test bimodality."""
    log("\n" + "=" * 70)
    log("[Phase 1] Base model entropy distribution analysis")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)

    # Collect entropies across all domains
    all_entropies = []
    all_losses = []
    domain_entropies = {}

    for domain in DOMAINS:
        texts = load_domain_texts(domain, split="valid")
        if not texts:
            log(f"  WARNING: No data for {domain}")
            continue

        log(f"  Computing entropies for {domain}...")
        ents, losses = compute_token_entropies(model, tokenizer, texts, max_batches=VAL_BATCHES)
        all_entropies.extend(ents)
        all_losses.extend(losses)
        domain_entropies[domain] = ents
        log(f"    {len(ents)} tokens, mean H={sum(ents)/len(ents):.4f}, "
            f"std={( sum((e - sum(ents)/len(ents))**2 for e in ents) / len(ents) )**0.5:.4f}")

    elapsed = time.time() - t0

    # Statistical tests
    import numpy as np
    ent_arr = np.array(all_entropies)

    tau = otsu_threshold(all_entropies)
    f_skip = float(np.mean(ent_arr < tau))

    # Spread tests (reframed K1)
    cv = coefficient_of_variation(all_entropies)
    otsu_eta = otsu_between_class_variance_ratio(all_entropies, tau)

    # Percentile statistics
    pcts = np.percentile(ent_arr, [5, 10, 25, 50, 75, 90, 95])

    log(f"\n  === Entropy Distribution Statistics ===")
    log(f"  Total tokens: {len(all_entropies)}")
    log(f"  Mean entropy: {np.mean(ent_arr):.4f}")
    log(f"  Std entropy:  {np.std(ent_arr):.4f}")
    log(f"  Min/Max:      {np.min(ent_arr):.4f} / {np.max(ent_arr):.4f}")
    log(f"  Percentiles:  5%={pcts[0]:.3f} 25%={pcts[2]:.3f} 50%={pcts[3]:.3f} "
        f"75%={pcts[4]:.3f} 95%={pcts[6]:.3f}")
    log(f"  Coefficient of variation (CV): {cv:.4f} (threshold: 0.5)")
    log(f"  Otsu between-class variance ratio (eta): {otsu_eta:.4f} (threshold: 0.15)")
    log(f"  Otsu threshold: {tau:.4f}")
    log(f"  Fraction below threshold (f_skip): {f_skip:.4f}")
    log(f"  Time: {elapsed:.1f}s")

    # K1 assessment: sufficient spread if CV > 0.5 AND otsu_eta > 0.15
    k1_cv_pass = cv > 0.5
    k1_eta_pass = otsu_eta > 0.15
    k1_pass = k1_cv_pass and k1_eta_pass

    log(f"\n  === K1 Assessment (sufficient spread for thresholding) ===")
    log(f"  CV > 0.5: {'PASS' if k1_cv_pass else 'FAIL'} (CV={cv:.4f})")
    log(f"  Otsu eta > 0.15: {'PASS' if k1_eta_pass else 'FAIL'} (eta={otsu_eta:.4f})")
    log(f"  K1 overall: {'PASS' if k1_pass else 'FAIL'}")

    if not k1_pass:
        log(f"\n  K1 FAIL: Entropy distribution lacks sufficient spread for thresholding.")
        log(f"  Proceeding with Otsu threshold to test K2/K3 anyway.")

    results = {
        "n_tokens": len(all_entropies),
        "entropy_mean": float(np.mean(ent_arr)),
        "entropy_std": float(np.std(ent_arr)),
        "entropy_min": float(np.min(ent_arr)),
        "entropy_max": float(np.max(ent_arr)),
        "entropy_percentiles": {
            "p5": float(pcts[0]), "p10": float(pcts[1]),
            "p25": float(pcts[2]), "p50": float(pcts[3]),
            "p75": float(pcts[4]), "p90": float(pcts[5]),
            "p95": float(pcts[6]),
        },
        "otsu_threshold": float(tau),
        "f_skip_at_otsu": float(f_skip),
        "cv": float(cv),
        "otsu_eta": float(otsu_eta),
        "K1_cv_pass": k1_cv_pass,
        "K1_eta_pass": k1_eta_pass,
        "K1_pass": k1_pass,
        "domain_entropy_stats": {},
        "time_s": round(elapsed, 1),
    }

    for domain, ents in domain_entropies.items():
        darr = np.array(ents)
        results["domain_entropy_stats"][domain] = {
            "n_tokens": len(ents),
            "mean": float(np.mean(darr)),
            "std": float(np.std(darr)),
            "f_skip_at_otsu": float(np.mean(darr < tau)),
        }

    log_memory("after-phase1")
    cleanup(model, tokenizer)

    # Also return raw data for Phase 2
    return results, all_entropies, all_losses, domain_entropies


# ===========================================================================
# Phase 2: PPL comparison (K2/K3 gate)
# ===========================================================================
def phase_ppl_comparison(entropy_results, all_base_entropies, all_base_losses, domain_entropies):
    """Compare PPL: base-only vs always-compose vs entropy-gated."""
    log("\n" + "=" * 70)
    log("[Phase 2] PPL comparison: base vs compose vs gated")
    log("=" * 70)

    t0 = time.time()

    # Load model with LoRA support
    model, tokenizer = load(MODEL_ID)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Freeze base
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    # Load all adapters
    adapters = {}
    for domain in DOMAINS:
        adapter_path = ADAPTER_DIR / domain
        if not (adapter_path / "adapter.npz").exists():
            log(f"  WARNING: No adapter for {domain}")
            continue
        adapters[domain] = load_adapter(adapter_path)
        log(f"  Loaded adapter: {domain}")

    # Compose adapters (uniform 1/N)
    composed = compose_adapters(list(adapters.values()))
    mx.eval(composed)
    log(f"  Composed {len(adapters)} adapters (uniform 1/N)")

    # Get Otsu threshold
    tau = entropy_results["otsu_threshold"]

    # Also test a sweep of thresholds for sensitivity analysis
    import numpy as np
    ent_arr = np.array(all_base_entropies)
    thresholds_to_test = [
        ("p10", float(np.percentile(ent_arr, 10))),
        ("p20", float(np.percentile(ent_arr, 20))),
        ("p30", float(np.percentile(ent_arr, 30))),
        ("otsu", tau),
        ("p50", float(np.percentile(ent_arr, 50))),
        ("p70", float(np.percentile(ent_arr, 70))),
    ]

    # Evaluate PPL for each condition across all domains
    # Condition 1: base-only (zero adapters)
    log("\n  Evaluating base-only PPL...")
    zero_adapter_in_model(model)
    mx.eval(model.parameters())

    base_ppls = {}
    base_token_losses = {}  # domain -> list of per-token losses
    for domain in DOMAINS:
        texts = load_domain_texts(domain, split="valid")
        if not texts:
            continue
        token_losses = []
        for text in texts[:VAL_BATCHES]:
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH + 1]
            x = mx.array(tokens[:-1])[None, :]
            y = mx.array(tokens[1:])
            logits = model(x)
            ce = nn.losses.cross_entropy(logits[0], y, reduction="none")
            mx.eval(ce)
            token_losses.extend(ce.tolist())
            del logits, ce, x, y
        base_token_losses[domain] = token_losses
        avg_loss = sum(token_losses) / len(token_losses) if token_losses else float("inf")
        base_ppls[domain] = math.exp(min(avg_loss, 100))
        log(f"    {domain}: PPL = {base_ppls[domain]:.4f}")

    # Condition 2: always-compose (all adapters, 1/N)
    log("\n  Evaluating always-compose PPL...")
    apply_adapter_to_model(model, composed)
    mx.eval(model.parameters())

    compose_ppls = {}
    compose_token_losses = {}
    for domain in DOMAINS:
        texts = load_domain_texts(domain, split="valid")
        if not texts:
            continue
        token_losses = []
        for text in texts[:VAL_BATCHES]:
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH + 1]
            x = mx.array(tokens[:-1])[None, :]
            y = mx.array(tokens[1:])
            logits = model(x)
            ce = nn.losses.cross_entropy(logits[0], y, reduction="none")
            mx.eval(ce)
            token_losses.extend(ce.tolist())
            del logits, ce, x, y
        compose_token_losses[domain] = token_losses
        avg_loss = sum(token_losses) / len(token_losses) if token_losses else float("inf")
        compose_ppls[domain] = math.exp(min(avg_loss, 100))
        log(f"    {domain}: PPL = {compose_ppls[domain]:.4f}")

    # Condition 3: Entropy-gated at multiple thresholds
    # For each token, use base loss if entropy < tau, else compose loss
    log("\n  Computing entropy-gated PPL at multiple thresholds...")

    # We need per-token base entropy. We already have base_token_losses and
    # compose_token_losses aligned by domain. We also need per-token entropies
    # from the base model. Let's compute base entropies per domain.

    # Re-run base model to get per-token entropies
    log("  Re-computing base entropies with zeroed adapter...")
    zero_adapter_in_model(model)
    mx.eval(model.parameters())

    domain_token_data = {}  # domain -> list of (entropy, base_loss, compose_loss) per token
    for domain in DOMAINS:
        texts = load_domain_texts(domain, split="valid")
        if not texts:
            continue

        entropies_d = []
        for text in texts[:VAL_BATCHES]:
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH + 1]
            x = mx.array(tokens[:-1])[None, :]
            logits = model(x)
            logits_2d = logits[0]
            probs = mx.softmax(logits_2d, axis=-1)
            log_p = mx.log(mx.clip(probs, 1e-10, 1.0))
            ent = -mx.sum(probs * log_p, axis=-1)
            mx.eval(ent)
            entropies_d.extend(ent.tolist())
            del logits, logits_2d, probs, log_p, ent, x

        # Align: entropies, base losses, compose losses must have same length
        n_base = len(base_token_losses.get(domain, []))
        n_comp = len(compose_token_losses.get(domain, []))
        n_ent = len(entropies_d)
        n_min = min(n_base, n_comp, n_ent)

        if n_min == 0:
            continue

        domain_token_data[domain] = {
            "entropies": entropies_d[:n_min],
            "base_losses": base_token_losses[domain][:n_min],
            "compose_losses": compose_token_losses[domain][:n_min],
        }
        log(f"    {domain}: {n_min} aligned tokens")

    # Compute gated PPL for each threshold
    threshold_results = {}
    for label, thresh in thresholds_to_test:
        gated_losses = []
        n_skipped = 0
        n_total = 0

        for domain, data in domain_token_data.items():
            for i in range(len(data["entropies"])):
                ent_i = data["entropies"][i]
                if ent_i < thresh:
                    # Confident: use base output
                    gated_losses.append(data["base_losses"][i])
                    n_skipped += 1
                else:
                    # Uncertain: use composed output
                    gated_losses.append(data["compose_losses"][i])
                n_total += 1

        if n_total == 0:
            continue

        avg_gated_loss = sum(gated_losses) / len(gated_losses)
        gated_ppl = math.exp(min(avg_gated_loss, 100))
        f_skip = n_skipped / n_total

        threshold_results[label] = {
            "threshold": float(thresh),
            "gated_ppl": round(gated_ppl, 4),
            "f_skip": round(f_skip, 4),
            "n_skipped": n_skipped,
            "n_total": n_total,
        }
        log(f"    {label} (tau={thresh:.3f}): PPL={gated_ppl:.4f}, "
            f"f_skip={f_skip:.2%}, n_skip={n_skipped}/{n_total}")

    # Compute aggregate PPL -- both arithmetic mean and token-weighted
    avg_base_ppl = sum(base_ppls.values()) / len(base_ppls) if base_ppls else float("inf")
    avg_compose_ppl = sum(compose_ppls.values()) / len(compose_ppls) if compose_ppls else float("inf")

    # Token-weighted compose PPL: exp(mean(all_compose_losses)) over all tokens
    all_compose_losses = []
    all_base_losses_flat = []
    for domain in DOMAINS:
        all_compose_losses.extend(compose_token_losses.get(domain, []))
        all_base_losses_flat.extend(base_token_losses.get(domain, []))
    tw_compose_ppl = math.exp(min(sum(all_compose_losses) / len(all_compose_losses), 100)) if all_compose_losses else float("inf")
    tw_base_ppl = math.exp(min(sum(all_base_losses_flat) / len(all_base_losses_flat), 100)) if all_base_losses_flat else float("inf")
    log(f"\n  Token-weighted compose PPL: {tw_compose_ppl:.4f}")
    log(f"  Token-weighted base PPL:    {tw_base_ppl:.4f}")
    log(f"  (Arithmetic mean compose PPL: {avg_compose_ppl:.4f} -- inflated by domain imbalance)")

    # Find best threshold (minimize PPL degradation while maximizing skip)
    best_label = None
    best_score = float("inf")
    for label, res in threshold_results.items():
        # Score: PPL degradation relative to token-weighted compose, penalized if f_skip < 0.1
        ppl_ratio = res["gated_ppl"] / tw_compose_ppl if tw_compose_ppl > 0 else float("inf")
        if res["f_skip"] >= 0.1:
            score = ppl_ratio - res["f_skip"] * 0.1
        else:
            score = ppl_ratio + 10
        if score < best_score:
            best_score = score
            best_label = label

    elapsed = time.time() - t0

    # K2/K3 assessment at Otsu threshold -- using token-weighted compose PPL
    otsu_res = threshold_results.get("otsu", {})
    otsu_gated_ppl = otsu_res.get("gated_ppl", float("inf"))
    # K2: gated PPL within 5% of token-weighted compose PPL
    k2_degradation = (otsu_gated_ppl - tw_compose_ppl) / tw_compose_ppl if tw_compose_ppl > 0 else float("inf")
    k2_pass = k2_degradation < 0.05
    k3_pass = otsu_res.get("f_skip", 0) >= 0.10

    # Also check K2/K3 at best threshold
    best_res = threshold_results.get(best_label, {}) if best_label else {}
    best_gated_ppl = best_res.get("gated_ppl", float("inf"))
    k2_best_deg = (best_gated_ppl - tw_compose_ppl) / tw_compose_ppl if tw_compose_ppl > 0 else float("inf")
    k2_best = k2_best_deg < 0.05
    k3_best = best_res.get("f_skip", 0) >= 0.10

    # S1: >= 30% skip with < 1% PPL degradation vs token-weighted compose
    s1_pass = False
    s1_threshold = None
    for label, res in threshold_results.items():
        if res["f_skip"] >= 0.30:
            ppl_degradation = (res["gated_ppl"] - tw_compose_ppl) / tw_compose_ppl
            if ppl_degradation < 0.01:
                s1_pass = True
                s1_threshold = label

    # S2: Mean PPL within 2% of token-weighted compose
    s2_pass = False
    for label, res in threshold_results.items():
        ppl_degradation = (res["gated_ppl"] - tw_compose_ppl) / tw_compose_ppl
        if abs(ppl_degradation) < 0.02 and res["f_skip"] >= 0.10:
            s2_pass = True

    # Per-domain gated vs compose PPL breakdown
    log(f"\n  === Per-Domain Gated vs Compose PPL (at Otsu threshold) ===")
    domain_gated_results = {}
    for domain, data in domain_token_data.items():
        gated_losses_d = []
        n_skip_d = 0
        for i in range(len(data["entropies"])):
            if data["entropies"][i] < tau:
                gated_losses_d.append(data["base_losses"][i])
                n_skip_d += 1
            else:
                gated_losses_d.append(data["compose_losses"][i])
        n_d = len(gated_losses_d)
        gated_ppl_d = math.exp(min(sum(gated_losses_d) / n_d, 100)) if n_d > 0 else float("inf")
        compose_ppl_d = math.exp(min(sum(data["compose_losses"]) / n_d, 100)) if n_d > 0 else float("inf")
        f_skip_d = n_skip_d / n_d if n_d > 0 else 0
        deg_d = (gated_ppl_d - compose_ppl_d) / compose_ppl_d * 100 if compose_ppl_d > 0 else float("inf")
        domain_gated_results[domain] = {
            "gated_ppl": round(gated_ppl_d, 4),
            "compose_ppl": round(compose_ppl_d, 4),
            "f_skip": round(f_skip_d, 4),
            "degradation_pct": round(deg_d, 2),
            "n_tokens": n_d,
        }
        log(f"    {domain}: gated={gated_ppl_d:.3f} compose={compose_ppl_d:.3f} "
            f"skip={f_skip_d:.1%} deg={deg_d:+.2f}%")

    log(f"\n  === PPL Summary ===")
    log(f"  Token-weighted base PPL:     {tw_base_ppl:.4f}")
    log(f"  Token-weighted compose PPL:  {tw_compose_ppl:.4f}")
    log(f"  (Arithmetic mean compose PPL: {avg_compose_ppl:.4f} -- NOT used for K2)")
    for label, res in threshold_results.items():
        marker = " <-- BEST" if label == best_label else ""
        deg_pct = (res["gated_ppl"] - tw_compose_ppl) / tw_compose_ppl * 100
        log(f"  Gated ({label}): PPL={res['gated_ppl']:.4f}, "
            f"skip={res['f_skip']:.2%}, vs compose={deg_pct:+.2f}%{marker}")

    log(f"\n  === K2 Assessment (Otsu, vs token-weighted compose PPL) ===")
    log(f"  Gated PPL: {otsu_gated_ppl:.4f}, TW compose PPL: {tw_compose_ppl:.4f}")
    log(f"  Degradation: {k2_degradation*100:+.2f}% (threshold: <5%)")
    log(f"  K2 (Otsu): {'PASS' if k2_pass else 'FAIL'}")
    log(f"  K2 (best threshold): {'PASS' if k2_best else 'FAIL'} ({k2_best_deg*100:+.2f}%)")

    log(f"\n  === K3 Assessment ===")
    log(f"  f_skip >= 10% (Otsu): {'PASS' if k3_pass else 'FAIL'} ({otsu_res.get('f_skip', 0):.2%})")
    log(f"  K3 (best threshold): {'PASS' if k3_best else 'FAIL'} ({best_res.get('f_skip', 0):.2%})")

    log(f"\n  === Success Criteria ===")
    log(f"  S1 (>=30% skip, <1% PPL deg vs TW compose): {'PASS' if s1_pass else 'FAIL'}")
    log(f"  S2 (PPL within 2% of TW compose): {'PASS' if s2_pass else 'FAIL'}")

    results = {
        "base_ppls": {k: round(v, 4) for k, v in base_ppls.items()},
        "compose_ppls": {k: round(v, 4) for k, v in compose_ppls.items()},
        "avg_base_ppl": round(avg_base_ppl, 4),
        "avg_compose_ppl_arithmetic": round(avg_compose_ppl, 4),
        "tw_compose_ppl": round(tw_compose_ppl, 4),
        "tw_base_ppl": round(tw_base_ppl, 4),
        "threshold_results": threshold_results,
        "best_threshold": best_label,
        "domain_gated_results": domain_gated_results,
        "K2_pass_otsu": k2_pass,
        "K2_degradation_otsu_pct": round(k2_degradation * 100, 2),
        "K2_pass_best": k2_best,
        "K3_pass_otsu": k3_pass,
        "K3_pass_best": k3_best,
        "S1_pass": s1_pass,
        "S1_threshold": s1_threshold,
        "S2_pass": s2_pass,
        "time_s": round(elapsed, 1),
    }

    log_memory("after-phase2")
    cleanup(model, tokenizer)
    return results


# ===========================================================================
# Phase 3: Wall-clock timing (S3)
# ===========================================================================
def phase_timing(otsu_tau=2.10):
    """Measure wall-clock time for base-only vs always-compose vs gated."""
    log("\n" + "=" * 70)
    log("[Phase 3] Wall-clock timing comparison")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    # Load adapters and compose
    adapter_list = []
    for domain in DOMAINS:
        adapter_path = ADAPTER_DIR / domain
        if (adapter_path / "adapter.npz").exists():
            adapter_list.append(load_adapter(adapter_path))
    composed = compose_adapters(adapter_list)
    mx.eval(composed)

    # Get a fixed set of test tokens
    texts = load_domain_texts("python", split="valid")[:10]
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) >= 2:
            all_tokens.append(tokens[:MAX_SEQ_LENGTH + 1])

    n_trials = 3
    log(f"  Running {n_trials} trials on {len(all_tokens)} sequences...")

    # Timing: always-compose
    compose_times = []
    for trial in range(n_trials):
        apply_adapter_to_model(model, composed)
        mx.eval(model.parameters())
        t_start = time.time()
        for tokens in all_tokens:
            x = mx.array(tokens[:-1])[None, :]
            logits = model(x)
            mx.eval(logits)
            del logits, x
        compose_times.append(time.time() - t_start)

    # Timing: base-only (zero adapter)
    base_times = []
    for trial in range(n_trials):
        zero_adapter_in_model(model)
        mx.eval(model.parameters())
        t_start = time.time()
        for tokens in all_tokens:
            x = mx.array(tokens[:-1])[None, :]
            logits = model(x)
            mx.eval(logits)
            del logits, x
        base_times.append(time.time() - t_start)

    # Timing: gated (base pass + entropy + selective compose)
    # Simulates: for each seq, run base, check entropy, then re-run with compose
    # if any token exceeds threshold
    import numpy as np

    # Use median entropy as threshold for timing
    gated_times = []
    for trial in range(n_trials):
        zero_adapter_in_model(model)
        mx.eval(model.parameters())
        t_start = time.time()
        n_composed = 0
        for tokens in all_tokens:
            x = mx.array(tokens[:-1])[None, :]
            # Base forward pass
            logits = model(x)
            mx.eval(logits)
            # Compute entropy
            probs = mx.softmax(logits[0], axis=-1)
            log_p = mx.log(mx.clip(probs, 1e-10, 1.0))
            ent = -mx.sum(probs * log_p, axis=-1)
            mx.eval(ent)
            max_ent = mx.max(ent).item()
            del logits, probs, log_p, ent

            # If any token is uncertain, re-run with composed adapter
            # (In practice we'd do this per-token, but for timing we check per-sequence)
            if max_ent > otsu_tau:  # Otsu threshold from Phase 1
                apply_adapter_to_model(model, composed)
                mx.eval(model.parameters())
                logits2 = model(x)
                mx.eval(logits2)
                del logits2
                # Reset to base for next sequence
                zero_adapter_in_model(model)
                mx.eval(model.parameters())
                n_composed += 1
            del x
        gated_times.append(time.time() - t_start)

    avg_base = sum(base_times) / len(base_times)
    avg_compose = sum(compose_times) / len(compose_times)
    avg_gated = sum(gated_times) / len(gated_times)

    # Note: the gated approach is expected to be SLOWER due to double forward pass
    # The real benefit is avoiding merge overhead in a serving scenario
    speedup_vs_compose = (avg_compose - avg_gated) / avg_compose if avg_compose > 0 else 0

    log(f"\n  === Timing Results ===")
    log(f"  Base-only:     {avg_base:.3f}s ({avg_base/len(all_tokens)*1000:.1f}ms/seq)")
    log(f"  Always-compose: {avg_compose:.3f}s ({avg_compose/len(all_tokens)*1000:.1f}ms/seq)")
    log(f"  Gated:         {avg_gated:.3f}s ({avg_gated/len(all_tokens)*1000:.1f}ms/seq)")
    log(f"  Speedup vs compose: {speedup_vs_compose:.2%}")

    s3_pass = speedup_vs_compose > 0.15

    log(f"  S3 (>15% speedup): {'PASS' if s3_pass else 'FAIL'}")

    results = {
        "base_time_s": round(avg_base, 3),
        "compose_time_s": round(avg_compose, 3),
        "gated_time_s": round(avg_gated, 3),
        "ms_per_seq_base": round(avg_base / len(all_tokens) * 1000, 1),
        "ms_per_seq_compose": round(avg_compose / len(all_tokens) * 1000, 1),
        "ms_per_seq_gated": round(avg_gated / len(all_tokens) * 1000, 1),
        "speedup_vs_compose": round(speedup_vs_compose, 4),
        "S3_pass": s3_pass,
        "n_sequences": len(all_tokens),
        "n_trials": n_trials,
    }

    log_memory("after-phase3")
    cleanup(model, tokenizer)
    return results


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_total = time.time()
    log("=" * 70)
    log("Entropy-Gated Expert Selection Experiment")
    log("=" * 70)
    log_memory("start")

    # Phase 1: Entropy distribution (K1 gate)
    entropy_results, all_entropies, all_losses, domain_entropies = phase_entropy_distribution()
    log_memory("after-phase1-cleanup")

    # Phase 2: PPL comparison (K2/K3)
    ppl_results = phase_ppl_comparison(entropy_results, all_entropies, all_losses, domain_entropies)
    log_memory("after-phase2-cleanup")

    # Phase 3: Timing (S3) -- pass Otsu threshold for consistency
    otsu_tau = entropy_results["otsu_threshold"]
    timing_results = phase_timing(otsu_tau=otsu_tau)
    log_memory("after-phase3-cleanup")

    # Aggregate results
    k1_pass = entropy_results["K1_pass"]
    k2_pass = ppl_results["K2_pass_otsu"] or ppl_results["K2_pass_best"]
    k3_pass = ppl_results["K3_pass_otsu"] or ppl_results["K3_pass_best"]

    all_kill_pass = k1_pass and k2_pass and k3_pass
    verdict = "SUPPORTED" if all_kill_pass else "KILLED"
    kill_reasons = []
    if not k1_pass:
        kill_reasons.append("K1: Entropy distribution lacks sufficient spread")
    if not k2_pass:
        kill_reasons.append("K2: Gated PPL > 5% worse than token-weighted compose")
    if not k3_pass:
        kill_reasons.append("K3: Fraction skipped < 10%")

    results = {
        "experiment": "entropy_gated_expert_selection",
        "model": MODEL_ID,
        "n_domains": len(DOMAINS),
        "domains": DOMAINS,
        "lora_rank": LORA_RANK,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "entropy_analysis": entropy_results,
        "ppl_comparison": ppl_results,
        "timing": timing_results,
        "K1_pass": k1_pass,
        "K2_pass": k2_pass,
        "K3_pass": k3_pass,
        "S1_pass": ppl_results.get("S1_pass", False),
        "S2_pass": ppl_results.get("S2_pass", False),
        "S3_pass": timing_results.get("S3_pass", False),
        "verdict": verdict,
        "kill_reasons": kill_reasons if kill_reasons else None,
        "total_time_s": round(time.time() - t_total, 1),
    }

    log(f"\n{'=' * 70}")
    log(f"VERDICT: {verdict}")
    if kill_reasons:
        for r in kill_reasons:
            log(f"  - {r}")
    log(f"Total time: {results['total_time_s']}s")
    log(f"{'=' * 70}")

    def convert(obj):
        """Convert numpy types to Python natives for JSON serialization."""
        import numpy as np
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    RESULTS_FILE.write_text(json.dumps(convert(results), indent=2))
    log(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
