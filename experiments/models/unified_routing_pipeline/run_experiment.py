#!/usr/bin/env python3
"""
Unified Routing Pipeline: entropy gate -> routing heads -> pre-merge composition.

Combines two PROVEN mechanisms:
  1. Entropy gating (63% skip rate at 1.13% PPL cost)
  2. Per-adapter routing heads (100% accuracy, 2.32% overhead)

Per-token Gumbel-sigmoid routing is EXCLUDED (killed by two experiments).

Kill criteria:
  K1: Unified pipeline PPL worse than best individual method (> 6.42) -> KILL
  K2: Total overhead > 10% of base forward pass -> KILL

Success criteria:
  S1: Unified pipeline beats all individual methods while saving >50%
      compute via entropy gating

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

# Reuse adapters from tiny_routing_heads and data from bitnet_2b_real_composition
ADAPTER_DIR = Path(__file__).parent.parent / "tiny_routing_heads" / "adapters"
HEADS_DIR = Path(__file__).parent.parent / "tiny_routing_heads" / "heads"
DATA_DIR = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
VAL_BATCHES = 25
DOMAINS = ["python", "math", "medical", "legal", "creative"]

# Entropy gating config (from proven experiment)
OTSU_THRESHOLD = 2.102779522613065  # Proven Otsu threshold


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


def zero_adapter_in_model(model):
    updates = []
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_b" in name:
            updates.append((name, mx.zeros_like(p)))
    if updates:
        model.update(tree_unflatten(updates))


def compose_adapters(adapter_list, weights=None):
    """Merge multiple adapter parameter dicts with given weights (default 1/N)."""
    N = len(adapter_list)
    if weights is None:
        weights = [1.0 / N] * N
    merged = {}
    for key in adapter_list[0].keys():
        merged[key] = sum(w * a[key] for w, a in zip(weights, adapter_list))
    return merged


# ===========================================================================
# Routing Head (from tiny_routing_heads)
# ===========================================================================
class RoutingHead(nn.Module):
    """Tiny binary classifier: h_pool -> sigmoid score."""

    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def __call__(self, x):
        h = nn.relu(self.fc1(x))
        return self.fc2(h)


def load_routing_head(domain, d_model):
    """Load a pre-trained routing head from disk."""
    head = RoutingHead(d_model, hidden_dim=32)
    head_path = HEADS_DIR / domain / "head.npz"
    if not head_path.exists():
        raise FileNotFoundError(f"No routing head at {head_path}")
    params = dict(mx.load(str(head_path)))
    head.load_weights(list(params.items()))
    mx.eval(head.parameters())
    return head


def get_hidden_states(model, x):
    """Extract hidden states from the last layer (mean-pooled)."""
    h = model.model.embed_tokens(x)
    for layer in model.model.layers:
        h = layer(h)
    h = model.model.norm(h)
    return h


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
# Phase 1: Baselines (base, uniform 1/N, routed top-2)
# ===========================================================================
def phase_baselines():
    """Compute baseline PPLs for comparison."""
    log("\n" + "=" * 70)
    log("[Phase 1] Computing baseline PPLs")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)

    # Get d_model before LoRA wrapping changes the module type
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[1]
    log(f"  d_model = {d_model}")

    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Freeze base, unfreeze LoRA
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    # Load adapters
    adapters = {}
    for domain in DOMAINS:
        adapter_path = ADAPTER_DIR / domain
        if not (adapter_path / "adapter.npz").exists():
            log(f"  WARNING: No adapter for {domain}")
            continue
        adapters[domain] = load_adapter(adapter_path)
        log(f"  Loaded adapter: {domain}")
    heads = {}
    for domain in DOMAINS:
        heads[domain] = load_routing_head(domain, d_model)
        log(f"  Loaded routing head: {domain}")

    # --- Base PPL (no adapters) ---
    log("\n  Computing base PPL...")
    zero_adapter_in_model(model)
    mx.eval(model.parameters())

    base_ppls = {}
    for domain in DOMAINS:
        texts = load_domain_texts(domain, split="valid")
        if not texts:
            continue
        total_loss = 0.0
        total_tokens = 0
        for text in texts[:VAL_BATCHES]:
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
        base_ppls[domain] = round(math.exp(min(total_loss / total_tokens, 100)), 4)
        log(f"    {domain}: {base_ppls[domain]}")

    # --- Uniform 1/N PPL ---
    log("\n  Computing uniform 1/N PPL...")
    composed_1n = compose_adapters(list(adapters.values()))
    apply_adapter_to_model(model, composed_1n)
    mx.eval(model.parameters())

    uniform_ppls = {}
    for domain in DOMAINS:
        texts = load_domain_texts(domain, split="valid")
        if not texts:
            continue
        total_loss = 0.0
        total_tokens = 0
        for text in texts[:VAL_BATCHES]:
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
        uniform_ppls[domain] = round(math.exp(min(total_loss / total_tokens, 100)), 4)
        log(f"    {domain}: {uniform_ppls[domain]}")
    del composed_1n

    # --- Head-routed top-2 PPL (recompute to get per-token data) ---
    log("\n  Computing head-routed top-2 PPL...")
    routed_ppls = {}
    routed_per_token = {}  # domain -> list of (entropy, loss) tuples

    for eval_domain in DOMAINS:
        texts = load_domain_texts(eval_domain, split="valid")
        if not texts:
            continue
        domain_losses = 0.0
        domain_tokens = 0
        token_data = []

        for text in texts[:VAL_BATCHES]:
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH + 1]
            x_tokens = tokens[:-1]
            y_tokens = tokens[1:]

            x = mx.array(x_tokens)[None, :]

            # First: base forward pass to get entropy + hidden states
            zero_adapter_in_model(model)
            mx.eval(model.parameters())

            # Get hidden states for routing
            h = get_hidden_states(model, x)
            h_pool = mx.mean(h, axis=1)  # (1, d)

            # Get base logits for entropy
            logits_base = model(x)
            logits_2d = logits_base[0]
            probs = mx.softmax(logits_2d, axis=-1)
            log_p = mx.log(mx.clip(probs, 1e-10, 1.0))
            ent = -mx.sum(probs * log_p, axis=-1)  # (seq_len,)
            mx.eval(h_pool, ent)

            # Base per-token losses
            y = mx.array(y_tokens)
            base_ce = nn.losses.cross_entropy(logits_2d, y, reduction="none")
            mx.eval(base_ce)
            base_losses_list = base_ce.tolist()
            ent_list = ent.tolist()

            del logits_base, logits_2d, probs, log_p, ent, base_ce, h

            # Route via heads
            scores = {}
            for head_domain, head in heads.items():
                logit = head(h_pool)
                mx.eval(logit)
                scores[head_domain] = mx.sigmoid(logit).item()

            # Top-2 selection
            sorted_domains = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            top2 = sorted_domains[:2]

            # Score-weighted composition
            total_score = sum(s for _, s in top2)
            composed = {}
            for sel_domain, sel_score in top2:
                w = sel_score / total_score
                for key, val in adapters[sel_domain].items():
                    if key not in composed:
                        composed[key] = val * w
                    else:
                        composed[key] = composed[key] + val * w

            # Forward with composed adapter
            apply_adapter_to_model(model, composed)
            mx.eval(model.parameters())
            logits_routed = model(x)
            routed_ce = nn.losses.cross_entropy(
                logits_routed[0], mx.array(y_tokens), reduction="none"
            )
            mx.eval(routed_ce)
            routed_losses_list = routed_ce.tolist()

            # Accumulate
            n_tokens = len(y_tokens)
            for i in range(n_tokens):
                token_data.append({
                    "entropy": ent_list[i],
                    "base_loss": base_losses_list[i],
                    "routed_loss": routed_losses_list[i],
                })
            domain_losses += sum(routed_losses_list)
            domain_tokens += n_tokens

            del composed, logits_routed, routed_ce, x, h_pool

        if domain_tokens > 0:
            ppl = math.exp(min(domain_losses / domain_tokens, 100))
        else:
            ppl = float("inf")
        routed_ppls[eval_domain] = round(ppl, 4)
        routed_per_token[eval_domain] = token_data
        log(f"    {eval_domain}: {routed_ppls[eval_domain]} "
            f"(uniform: {uniform_ppls.get(eval_domain, '?')}, "
            f"base: {base_ppls.get(eval_domain, '?')})")

    avg_base = sum(base_ppls.values()) / len(base_ppls)
    avg_uniform = sum(uniform_ppls.values()) / len(uniform_ppls)
    avg_routed = sum(routed_ppls.values()) / len(routed_ppls)

    log(f"\n  === Baseline Summary ===")
    log(f"  Avg base PPL:    {avg_base:.4f}")
    log(f"  Avg uniform PPL: {avg_uniform:.4f}")
    log(f"  Avg routed PPL:  {avg_routed:.4f}")

    elapsed = time.time() - t0
    log(f"  Phase 1 time: {elapsed:.1f}s")

    results = {
        "base_ppls": base_ppls,
        "uniform_ppls": uniform_ppls,
        "routed_ppls": routed_ppls,
        "avg_base_ppl": round(avg_base, 4),
        "avg_uniform_ppl": round(avg_uniform, 4),
        "avg_routed_ppl": round(avg_routed, 4),
        "time_s": round(elapsed, 1),
    }

    log_memory("after-phase1")
    cleanup(model, tokenizer, heads, adapters)
    return results, routed_per_token


# ===========================================================================
# Phase 2: Unified Pipeline PPL
# ===========================================================================
def phase_unified_pipeline(routed_per_token):
    """Compute unified pipeline PPL using per-token entropy gating.

    For each token:
    - If entropy < tau: use base model loss (skip routing)
    - If entropy >= tau: use routed top-2 loss

    This is computed from the per-token data already collected in Phase 1.
    No additional model forward passes needed.
    """
    log("\n" + "=" * 70)
    log("[Phase 2] Unified pipeline PPL (entropy gate + routing heads)")
    log("=" * 70)

    import numpy as np

    # Also sweep multiple thresholds for sensitivity analysis
    all_entropies = []
    for domain, tokens in routed_per_token.items():
        for t in tokens:
            all_entropies.append(t["entropy"])
    ent_arr = np.array(all_entropies)

    thresholds = [
        ("p20", float(np.percentile(ent_arr, 20))),
        ("p30", float(np.percentile(ent_arr, 30))),
        ("p40", float(np.percentile(ent_arr, 40))),
        ("otsu", OTSU_THRESHOLD),
        ("p60", float(np.percentile(ent_arr, 60))),
        ("p70", float(np.percentile(ent_arr, 70))),
        ("p80", float(np.percentile(ent_arr, 80))),
    ]

    threshold_results = {}
    for label, tau in thresholds:
        unified_losses = []
        n_skipped = 0
        n_routed = 0
        n_total = 0

        domain_results = {}
        for domain, tokens in routed_per_token.items():
            d_losses = []
            d_skip = 0
            for t in tokens:
                if t["entropy"] < tau:
                    # Confident token: use base output
                    d_losses.append(t["base_loss"])
                    d_skip += 1
                    n_skipped += 1
                else:
                    # Uncertain token: use routed output
                    d_losses.append(t["routed_loss"])
                    n_routed += 1
                n_total += 1

            d_avg = sum(d_losses) / len(d_losses) if d_losses else float("inf")
            d_ppl = math.exp(min(d_avg, 100))
            domain_results[domain] = {
                "ppl": round(d_ppl, 4),
                "f_skip": round(d_skip / len(tokens), 4) if tokens else 0,
                "n_tokens": len(tokens),
            }
            unified_losses.extend(d_losses)

        avg_loss = sum(unified_losses) / len(unified_losses) if unified_losses else float("inf")
        unified_ppl = math.exp(min(avg_loss, 100))
        f_skip = n_skipped / n_total if n_total > 0 else 0

        # Per-domain PPL
        avg_domain_ppl = sum(d["ppl"] for d in domain_results.values()) / len(domain_results)

        threshold_results[label] = {
            "threshold": round(tau, 4),
            "unified_ppl_tw": round(unified_ppl, 4),  # token-weighted
            "unified_ppl_avg": round(avg_domain_ppl, 4),  # domain-averaged
            "f_skip": round(f_skip, 4),
            "n_skipped": n_skipped,
            "n_routed": n_routed,
            "n_total": n_total,
            "per_domain": domain_results,
        }

        log(f"  {label} (tau={tau:.3f}): "
            f"PPL_avg={avg_domain_ppl:.4f}, PPL_tw={unified_ppl:.4f}, "
            f"skip={f_skip:.1%}")

    # Report Otsu threshold in detail
    otsu = threshold_results["otsu"]
    log(f"\n  === Otsu Threshold Detail ===")
    log(f"  Unified avg PPL: {otsu['unified_ppl_avg']}")
    log(f"  Unified tw PPL:  {otsu['unified_ppl_tw']}")
    log(f"  Skip rate:       {otsu['f_skip']:.1%}")
    log(f"\n  Per-domain at Otsu:")
    for domain, dr in otsu["per_domain"].items():
        log(f"    {domain}: PPL={dr['ppl']}, skip={dr['f_skip']:.1%}")

    return threshold_results


# ===========================================================================
# Phase 3: Overhead Measurement (K2)
# ===========================================================================
def phase_overhead():
    """Measure total pipeline overhead vs base forward pass."""
    log("\n" + "=" * 70)
    log("[Phase 3] Overhead measurement (K2)")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)

    # Get d_model before LoRA wrapping
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[1]

    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    # Load adapters and heads
    adapters = {}
    for domain in DOMAINS:
        adapter_path = ADAPTER_DIR / domain
        if (adapter_path / "adapter.npz").exists():
            adapters[domain] = load_adapter(adapter_path)
    heads = {}
    for domain in DOMAINS:
        heads[domain] = load_routing_head(domain, d_model)

    # Get test tokens
    texts = load_domain_texts("python", split="valid")[:10]
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) >= 2:
            all_tokens.append(tokens[:MAX_SEQ_LENGTH + 1])

    n_trials = 5
    log(f"  Running {n_trials} trials on {len(all_tokens)} sequences...")

    # Warm up
    zero_adapter_in_model(model)
    mx.eval(model.parameters())
    for tokens in all_tokens[:2]:
        x = mx.array(tokens[:-1])[None, :]
        out = model(x)
        mx.eval(out)
        del out, x

    # --- Time: base forward pass only ---
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

    # --- Time: always-compose (pre-merged uniform 1/N) ---
    composed_1n = compose_adapters(list(adapters.values()))
    compose_times = []
    for trial in range(n_trials):
        apply_adapter_to_model(model, composed_1n)
        mx.eval(model.parameters())
        t_start = time.time()
        for tokens in all_tokens:
            x = mx.array(tokens[:-1])[None, :]
            logits = model(x)
            mx.eval(logits)
            del logits, x
        compose_times.append(time.time() - t_start)
    del composed_1n

    # --- Time: unified pipeline (base + entropy + conditional routing + compose) ---
    unified_times = []
    unified_n_routed = 0
    unified_n_total = 0

    for trial in range(n_trials):
        zero_adapter_in_model(model)
        mx.eval(model.parameters())
        trial_routed = 0
        trial_total = 0

        t_start = time.time()
        for tokens in all_tokens:
            x = mx.array(tokens[:-1])[None, :]

            # Step 1: Base forward pass (for entropy + hidden states)
            logits_base = model(x)
            mx.eval(logits_base)

            # Step 2: Compute entropy
            probs = mx.softmax(logits_base[0], axis=-1)
            log_p = mx.log(mx.clip(probs, 1e-10, 1.0))
            ent = -mx.sum(probs * log_p, axis=-1)
            mx.eval(ent)
            max_ent = mx.max(ent).item()

            del logits_base, probs, log_p, ent

            trial_total += 1

            # Step 3: If any token is uncertain, route and compose
            if max_ent > OTSU_THRESHOLD:
                # Get hidden states for routing
                h = get_hidden_states(model, x)
                h_pool = mx.mean(h, axis=1)
                mx.eval(h_pool)
                del h

                # Run routing heads
                scores = {}
                for head_domain, head in heads.items():
                    logit = head(h_pool)
                    mx.eval(logit)
                    scores[head_domain] = mx.sigmoid(logit).item()

                # Top-2 selection and composition
                sorted_domains = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                top2 = sorted_domains[:2]
                total_score = sum(s for _, s in top2)
                composed = {}
                for sel_domain, sel_score in top2:
                    w = sel_score / total_score
                    for key, val in adapters[sel_domain].items():
                        if key not in composed:
                            composed[key] = val * w
                        else:
                            composed[key] = composed[key] + val * w

                # Forward with composed adapter
                apply_adapter_to_model(model, composed)
                mx.eval(model.parameters())
                logits_routed = model(x)
                mx.eval(logits_routed)
                del logits_routed, composed, h_pool

                # Reset to base
                zero_adapter_in_model(model)
                mx.eval(model.parameters())
                trial_routed += 1

            del x
        unified_times.append(time.time() - t_start)

        if trial == 0:
            unified_n_routed = trial_routed
            unified_n_total = trial_total

    avg_base = sum(base_times) / len(base_times)
    avg_compose = sum(compose_times) / len(compose_times)
    avg_unified = sum(unified_times) / len(unified_times)

    # Overhead of unified vs base
    overhead_vs_base = (avg_unified - avg_base) / avg_base if avg_base > 0 else float("inf")

    # Cost of routing components (entropy + heads + second pass) per routed sequence
    routing_cost_per_seq = (avg_unified - avg_base) / unified_n_routed if unified_n_routed > 0 else 0
    base_cost_per_seq = avg_base / len(all_tokens)

    # Effective overhead considering skip rate
    f_route = unified_n_routed / unified_n_total if unified_n_total > 0 else 1
    effective_overhead = overhead_vs_base  # already accounts for skip

    log(f"\n  === Timing Results ===")
    log(f"  Base-only:      {avg_base:.3f}s ({avg_base/len(all_tokens)*1000:.1f}ms/seq)")
    log(f"  Always-compose: {avg_compose:.3f}s ({avg_compose/len(all_tokens)*1000:.1f}ms/seq)")
    log(f"  Unified:        {avg_unified:.3f}s ({avg_unified/len(all_tokens)*1000:.1f}ms/seq)")
    log(f"  Sequences routed: {unified_n_routed}/{unified_n_total} ({f_route:.1%})")
    log(f"  Overhead vs base: {overhead_vs_base:.1%}")
    log(f"  Routing cost/seq: {routing_cost_per_seq*1000:.1f}ms")
    log(f"  Base cost/seq:    {base_cost_per_seq*1000:.1f}ms")

    k2_pass = overhead_vs_base < 0.10

    log(f"\n  K2 (overhead < 10%): {'PASS' if k2_pass else 'FAIL'} ({overhead_vs_base:.1%})")

    results = {
        "base_time_s": round(avg_base, 3),
        "compose_time_s": round(avg_compose, 3),
        "unified_time_s": round(avg_unified, 3),
        "ms_per_seq_base": round(avg_base / len(all_tokens) * 1000, 1),
        "ms_per_seq_compose": round(avg_compose / len(all_tokens) * 1000, 1),
        "ms_per_seq_unified": round(avg_unified / len(all_tokens) * 1000, 1),
        "overhead_vs_base_pct": round(overhead_vs_base * 100, 2),
        "f_route": round(f_route, 4),
        "n_routed": unified_n_routed,
        "n_total": unified_n_total,
        "K2_pass": k2_pass,
        "n_sequences": len(all_tokens),
        "n_trials": n_trials,
    }

    log_memory("after-phase3")
    cleanup(model, tokenizer, heads, adapters)
    return results


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_total = time.time()
    log("=" * 70)
    log("Unified Routing Pipeline Experiment")
    log("entropy gate -> routing heads -> pre-merge composition")
    log("=" * 70)
    log_memory("start")

    # Check prerequisites
    for domain in DOMAINS:
        assert (ADAPTER_DIR / domain / "adapter.npz").exists(), \
            f"Missing adapter for {domain}. Run tiny_routing_heads first."
        assert (HEADS_DIR / domain / "head.npz").exists(), \
            f"Missing routing head for {domain}. Run tiny_routing_heads first."

    # Phase 1: Baselines + per-token data collection
    baseline_results, routed_per_token = phase_baselines()
    log_memory("after-phase1-cleanup")

    # Phase 2: Unified pipeline PPL (computed from per-token data, no model needed)
    threshold_results = phase_unified_pipeline(routed_per_token)
    log_memory("after-phase2")

    # Phase 3: Overhead measurement
    overhead_results = phase_overhead()
    log_memory("after-phase3-cleanup")

    # ================================================================
    # Aggregate results and assess kill criteria
    # ================================================================
    otsu = threshold_results["otsu"]

    # K1: Unified pipeline PPL worse than best individual method (> 6.42)
    # Use domain-averaged PPL for consistency with routing heads experiment
    unified_ppl = otsu["unified_ppl_avg"]
    best_individual_ppl = 6.42  # From tiny_routing_heads: avg routed PPL
    k1_pass = unified_ppl <= best_individual_ppl
    k1_margin = (unified_ppl - best_individual_ppl) / best_individual_ppl * 100

    # K2: Total overhead > 10% of base forward pass
    k2_pass = overhead_results["K2_pass"]
    k2_overhead = overhead_results["overhead_vs_base_pct"]

    # S1: Beats all individual methods while saving >50% compute via entropy gating
    s1_beats_all = unified_ppl < baseline_results["avg_routed_ppl"]
    s1_saves_compute = otsu["f_skip"] > 0.50
    s1_pass = s1_beats_all and s1_saves_compute

    verdict = "SUPPORTED" if (k1_pass and k2_pass) else "KILLED"

    kill_reasons = []
    if not k1_pass:
        kill_reasons.append(f"K1: Unified PPL {unified_ppl:.4f} > {best_individual_ppl} threshold")
    if not k2_pass:
        kill_reasons.append(f"K2: Overhead {k2_overhead:.1f}% > 10% threshold")

    log(f"\n{'=' * 70}")
    log(f"RESULTS SUMMARY")
    log(f"{'=' * 70}")
    log(f"\n  Unified pipeline (Otsu threshold):")
    log(f"    Domain-avg PPL:     {unified_ppl:.4f}")
    log(f"    Token-weighted PPL: {otsu['unified_ppl_tw']:.4f}")
    log(f"    Skip rate:          {otsu['f_skip']:.1%}")
    log(f"\n  Baselines:")
    log(f"    Base PPL:           {baseline_results['avg_base_ppl']:.4f}")
    log(f"    Uniform 1/N PPL:   {baseline_results['avg_uniform_ppl']:.4f}")
    log(f"    Routed top-2 PPL:  {baseline_results['avg_routed_ppl']:.4f}")
    log(f"\n  Kill Criteria:")
    log(f"    K1 (PPL <= 6.42):  {'PASS' if k1_pass else 'FAIL'} ({unified_ppl:.4f}, {k1_margin:+.2f}%)")
    log(f"    K2 (overhead <10%): {'PASS' if k2_pass else 'FAIL'} ({k2_overhead:.1f}%)")
    log(f"\n  Success Criteria:")
    log(f"    S1 (beats all + >50% skip): {'PASS' if s1_pass else 'FAIL'}")
    log(f"      Beats routed: {s1_beats_all} ({unified_ppl:.4f} vs {baseline_results['avg_routed_ppl']:.4f})")
    log(f"      >50% skip:    {s1_saves_compute} ({otsu['f_skip']:.1%})")
    log(f"\n  VERDICT: {verdict}")
    if kill_reasons:
        for r in kill_reasons:
            log(f"    - {r}")

    # Build full results
    results = {
        "experiment": "unified_routing_pipeline",
        "model": MODEL_ID,
        "n_domains": len(DOMAINS),
        "domains": DOMAINS,
        "lora_rank": LORA_RANK,
        "otsu_threshold": OTSU_THRESHOLD,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baselines": baseline_results,
        "unified_pipeline": {
            "threshold_sweep": threshold_results,
            "otsu_ppl_avg": unified_ppl,
            "otsu_ppl_tw": otsu["unified_ppl_tw"],
            "otsu_f_skip": otsu["f_skip"],
        },
        "overhead": overhead_results,
        "K1_pass": bool(k1_pass),
        "K1_unified_ppl": unified_ppl,
        "K1_threshold": best_individual_ppl,
        "K1_margin_pct": round(k1_margin, 2),
        "K2_pass": bool(k2_pass),
        "K2_overhead_pct": k2_overhead,
        "S1_pass": bool(s1_pass),
        "S1_beats_all": bool(s1_beats_all),
        "S1_saves_compute": bool(s1_saves_compute),
        "verdict": verdict,
        "kill_reasons": kill_reasons if kill_reasons else None,
        "total_time_s": round(time.time() - t_total, 1),
    }

    # JSON-safe conversion
    import numpy as np

    def convert(obj):
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
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']}s")


if __name__ == "__main__":
    main()
