#!/usr/bin/env python3
"""
Base-Free Composition: Grassmannian Scaffold Replaces Pretrained Base Weights

Tests whether adapter composition quality survives when the pretrained base model
is progressively replaced with random/structured ternary scaffolds.

Hypothesis: The pretrained base carries most of the model's value, but adapters
may have learned enough compensatory structure that a scaffold (random or
Grassmannian-structured ternary weights) can partially replace the base.

Kill criteria:
  K1: skeleton-only PPL >5x pretrained-base PPL (base carries too much)
  K2: no layer can be zeroed without >20% PPL regression (every layer essential)

Phases:
  1. Baseline: BitNet-2B-4T + N=5 composed adapters -> composed PPL
  2. Per-layer ablation: zero out each of 30 layers' base weights, measure PPL
  3. Progressive ablation: sort by criticality, zero bottom-K layers cumulatively
  4. Skeleton-only: replace ALL base with random ternary, keep adapters
  5. Scaled random: random ternary with per-layer scale matching pretrained norms

Platform: Apple Silicon MLX, $0.
Reuses: 5 adapters from bitnet_scale_n15 (medical, code, math, legal, creative)
"""

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm import load
from mlx_lm.models.bitlinear_layers import BitLinear

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 128
VAL_BATCHES = 25
SEED = 42

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse adapters from bitnet_scale_n15
ADAPTER_SOURCE = Path(__file__).parent.parent / "bitnet_scale_n15" / "adapters"
# Reuse validation data from multiple sources
DATA_SOURCE_N15 = Path(__file__).parent.parent / "bitnet_scale_n15" / "data"
DATA_SOURCE_TERNARY = Path(__file__).parent.parent / "bitnet_ternary_convergence" / "data"
DATA_SOURCE_REAL = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"

# Use the 5 original domains that have adapters from multiseed_validation/seed42
# These were used in N=15 experiment
DOMAINS = ["medical", "code", "math", "legal", "creative"]

# Target modules (same as training)
TARGET_KEYS = {
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
}


# ===========================================================================
# Ternary weight unpacking
# ===========================================================================
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
    """Replace BitLinear with nn.Linear, storing original weights for ablation."""
    count = 0
    layer_weight_info = []  # Store (layer_idx, key, shape, norm) for each layer

    for layer_idx, layer in enumerate(model.model.layers):
        updates = []
        layer_info = []
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

                # Record weight norm for later scaffold scaling
                norm = mx.sqrt(mx.sum(unpacked_w ** 2)).item()
                layer_info.append({
                    "key": key,
                    "shape": list(unpacked_w.shape),
                    "frobenius_norm": norm,
                })
        if updates:
            layer.update_modules(tree_unflatten(updates))
        layer_weight_info.append(layer_info)

    mx.eval(model.parameters())
    print(f"  Replaced {count} BitLinear -> nn.Linear")
    return model, layer_weight_info


# ===========================================================================
# LoRA (matching N=15 TernaryLoRALinear interface for adapter loading)
# ===========================================================================
class TernaryLoRALinear(nn.Module):
    def __init__(self, base_linear, r=16, scale=20.0):
        super().__init__()
        self.linear = base_linear
        self.r = r
        self.scale = scale
        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]
        s = 1.0 / math.sqrt(in_features)
        self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_features, r))
        self.lora_b = mx.zeros((r, out_features))

    def _ste_ternary(self, W):
        alpha = mx.mean(mx.abs(W)) + 1e-10
        W_scaled = W / alpha
        W_q = mx.clip(mx.round(W_scaled), -1.0, 1.0) * alpha
        return W + mx.stop_gradient(W_q - W)

    def __call__(self, x):
        base_out = self.linear(x)
        A = self._ste_ternary(self.lora_a)
        B = self._ste_ternary(self.lora_b)
        lora_out = (x @ A) @ B * self.scale
        return base_out + lora_out


def apply_ternary_lora(model, rank=16, scale=20.0):
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if key in TARGET_KEYS and isinstance(module, nn.Linear):
                lora = TernaryLoRALinear(module, r=rank, scale=scale)
                updates.append((key, lora))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    print(f"  Applied Ternary LoRA (r={rank}) to {count} layers")
    return model


def load_adapter(path):
    return dict(mx.load(str(path / "adapter.npz")))


def compose_adapters(adapter_list, scale_per_adapter=None):
    N = len(adapter_list)
    if scale_per_adapter is None:
        scale_per_adapter = 1.0 / N
    merged = {}
    for key in adapter_list[0].keys():
        stacked = mx.stack([a[key] for a in adapter_list])
        merged[key] = mx.sum(stacked, axis=0) * scale_per_adapter
    return merged


def apply_adapter_weights(model, adapter_params, scale=1.0):
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


# ===========================================================================
# PPL evaluation
# ===========================================================================
def find_data_dir(domain_name):
    """Find validation data for a domain across multiple experiment dirs."""
    for src in [DATA_SOURCE_N15, DATA_SOURCE_TERNARY, DATA_SOURCE_REAL]:
        # Map domain names between experiments
        name_map = {"code": "code", "python": "code"}
        check_name = domain_name
        d = src / check_name
        if d.exists() and (d / "valid.jsonl").exists():
            return d
        # Try alternate name
        if domain_name == "code":
            for alt in ["python", "code"]:
                d = src / alt
                if d.exists() and (d / "valid.jsonl").exists():
                    return d
    return None


def compute_ppl(model, tokenizer, data_path, max_batches=25):
    fpath = data_path / "valid.jsonl"
    if not fpath.exists():
        return float("inf")

    texts = []
    with open(fpath) as f:
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

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 100))


def compute_mean_ppl(model, tokenizer, data_dirs, domains):
    """Compute mean PPL across all domains."""
    ppls = {}
    for domain in domains:
        data_dir = data_dirs.get(domain)
        if data_dir:
            ppls[domain] = compute_ppl(model, tokenizer, data_dir)
        else:
            ppls[domain] = float("inf")
    mean_ppl = sum(ppls.values()) / len(ppls) if ppls else float("inf")
    return ppls, mean_ppl


# ===========================================================================
# Layer ablation: zero out base weights for specific layers
# ===========================================================================
def get_base_weights(model):
    """Store a copy of all base weights (linear layers inside LoRA wrappers)."""
    saved = {}
    for layer_idx, layer in enumerate(model.model.layers):
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                full_key = f"layers.{layer_idx}.{key}"
                saved[full_key] = mx.array(module.linear.weight)
    mx.eval(saved)
    return saved


def zero_layer_base_weights(model, layer_idx):
    """Zero out all base weights in a specific transformer layer."""
    layer = model.model.layers[layer_idx]
    for key, module in layer.named_modules():
        if isinstance(module, TernaryLoRALinear):
            module.linear.weight = mx.zeros_like(module.linear.weight)
    mx.eval(model.parameters())


def restore_base_weights(model, saved_weights):
    """Restore all base weights from saved copy."""
    for layer_idx, layer in enumerate(model.model.layers):
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                full_key = f"layers.{layer_idx}.{key}"
                if full_key in saved_weights:
                    module.linear.weight = saved_weights[full_key]
    mx.eval(model.parameters())


def replace_layer_with_random_ternary(model, layer_idx, seed=None):
    """Replace base weights in a layer with random ternary {-1, 0, 1}."""
    if seed is not None:
        mx.random.seed(seed + layer_idx)
    layer = model.model.layers[layer_idx]
    for key, module in layer.named_modules():
        if isinstance(module, TernaryLoRALinear):
            shape = module.linear.weight.shape
            # Random ternary: uniform over {-1, 0, 1}
            rand = mx.random.uniform(shape=shape)
            ternary = mx.where(rand < 0.333, -1.0, mx.where(rand < 0.667, 0.0, 1.0))
            # Scale to match original norm
            orig_norm = mx.sqrt(mx.sum(module.linear.weight ** 2))
            new_norm = mx.sqrt(mx.sum(ternary ** 2)) + 1e-10
            ternary = ternary * (orig_norm / new_norm)
            module.linear.weight = ternary.astype(mx.bfloat16)
    mx.eval(model.parameters())


def replace_all_with_random_ternary(model, seed=42, scale_match=True):
    """Replace ALL base weights with random ternary tensors."""
    mx.random.seed(seed)
    for layer_idx, layer in enumerate(model.model.layers):
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                shape = module.linear.weight.shape
                rand = mx.random.uniform(shape=shape)
                ternary = mx.where(rand < 0.333, -1.0, mx.where(rand < 0.667, 0.0, 1.0))
                if scale_match:
                    orig_norm = mx.sqrt(mx.sum(module.linear.weight ** 2))
                    new_norm = mx.sqrt(mx.sum(ternary ** 2)) + 1e-10
                    ternary = ternary * (orig_norm / new_norm)
                module.linear.weight = ternary.astype(mx.bfloat16)
    mx.eval(model.parameters())


def replace_all_with_unscaled_random(model, seed=42):
    """Replace ALL base weights with unscaled random ternary {-1, 0, 1}."""
    mx.random.seed(seed)
    for layer_idx, layer in enumerate(model.model.layers):
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                shape = module.linear.weight.shape
                rand = mx.random.uniform(shape=shape)
                ternary = mx.where(rand < 0.333, -1.0, mx.where(rand < 0.667, 0.0, 1.0))
                module.linear.weight = ternary.astype(mx.bfloat16)
    mx.eval(model.parameters())


# ===========================================================================
# Main experiment
# ===========================================================================
def main():
    mx.random.seed(SEED)
    t_start = time.time()

    results = {
        "experiment": "bitnet_basefree_exploration",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "domains": DOMAINS,
        "n_adapters": len(DOMAINS),
        "seed": SEED,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=" * 70)
    print("Base-Free Composition: Grassmannian Scaffold Exploration")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 0: Load model
    # ------------------------------------------------------------------
    print("\n[Phase 0] Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("  Unpacking ternary weights...")
    model, layer_weight_info = replace_bitlinear_with_linear(model)
    n_layers = len(model.model.layers)
    print(f"  Model has {n_layers} transformer layers")
    results["n_layers"] = n_layers

    # Apply LoRA (needed for adapter loading)
    model = apply_ternary_lora(model, rank=LORA_RANK, scale=LORA_SCALE)

    # ------------------------------------------------------------------
    # Phase 0b: Find validation data
    # ------------------------------------------------------------------
    print("\n[Phase 0b] Locating validation data...")
    data_dirs = {}
    for domain in DOMAINS:
        d = find_data_dir(domain)
        if d:
            print(f"  {domain}: {d}")
            data_dirs[domain] = d
        else:
            print(f"  WARNING: no data for {domain}")
    results["data_found"] = list(data_dirs.keys())

    if len(data_dirs) < 3:
        print("FATAL: Not enough validation data found")
        results["error"] = "insufficient data"
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        return

    # ------------------------------------------------------------------
    # Phase 1: Baseline — pretrained base + N=5 composed adapters
    # ------------------------------------------------------------------
    print("\n[Phase 1] Baseline: pretrained base + N=5 composed adapters")

    # Load adapters
    print("  Loading adapters from bitnet_scale_n15...")
    adapters = {}
    for domain in DOMAINS:
        adapter_path = ADAPTER_SOURCE / domain
        if adapter_path.exists():
            adapters[domain] = load_adapter(adapter_path)
            n_params = sum(v.size for v in adapters[domain].values())
            print(f"  {domain}: loaded ({n_params:,} params)")
        else:
            print(f"  WARNING: no adapter for {domain}")

    if len(adapters) < 3:
        print("FATAL: Not enough adapters found")
        results["error"] = "insufficient adapters"
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        return

    # Measure base PPL (no adapters)
    print("\n  Computing base model PPL (no adapters)...")
    # Zero out LoRA to get pure base
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.lora_a = mx.zeros_like(module.lora_a)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())

    base_ppls, base_mean_ppl = compute_mean_ppl(model, tokenizer, data_dirs, DOMAINS)
    print(f"  Base mean PPL: {base_mean_ppl:.2f}")
    for d, p in base_ppls.items():
        print(f"    {d}: {p:.2f}")
    results["base_ppls"] = {k: round(v, 4) for k, v in base_ppls.items()}
    results["base_mean_ppl"] = round(base_mean_ppl, 4)

    # Compose adapters with 1/N scaling
    print("\n  Composing N=5 adapters with 1/N scaling...")
    composed = compose_adapters(list(adapters.values()), scale_per_adapter=1.0/len(adapters))
    apply_adapter_weights(model, composed)

    composed_ppls, composed_mean_ppl = compute_mean_ppl(model, tokenizer, data_dirs, DOMAINS)
    print(f"  Composed mean PPL: {composed_mean_ppl:.2f}")
    for d, p in composed_ppls.items():
        print(f"    {d}: {p:.2f}")
    results["composed_ppls"] = {k: round(v, 4) for k, v in composed_ppls.items()}
    results["composed_mean_ppl"] = round(composed_mean_ppl, 4)

    # Save base weights for later restoration
    print("\n  Saving base weights for ablation...")
    saved_base = get_base_weights(model)
    n_saved = sum(v.size for v in saved_base.values())
    print(f"  Saved {len(saved_base)} weight tensors ({n_saved:,} params)")

    # ------------------------------------------------------------------
    # Phase 2: Per-layer ablation
    # ------------------------------------------------------------------
    print(f"\n[Phase 2] Per-layer ablation ({n_layers} layers)")
    print("  Zeroing each layer's base weights, measuring PPL impact...")

    layer_ablation = []
    for layer_idx in range(n_layers):
        # Restore to composed state
        restore_base_weights(model, saved_base)
        apply_adapter_weights(model, composed)

        # Zero this layer's base weights
        zero_layer_base_weights(model, layer_idx)

        # Measure PPL
        _, ablated_mean_ppl = compute_mean_ppl(model, tokenizer, data_dirs, DOMAINS)
        ppl_increase = (ablated_mean_ppl - composed_mean_ppl) / composed_mean_ppl * 100

        layer_ablation.append({
            "layer": layer_idx,
            "ablated_ppl": round(ablated_mean_ppl, 4),
            "ppl_increase_pct": round(ppl_increase, 2),
        })

        marker = " ***" if ppl_increase > 20 else ""
        print(f"  Layer {layer_idx:2d}: PPL={ablated_mean_ppl:.2f} "
              f"(+{ppl_increase:.1f}%){marker}")

    results["layer_ablation"] = layer_ablation

    # K2 check: is any layer non-essential?
    min_increase = min(la["ppl_increase_pct"] for la in layer_ablation)
    max_increase = max(la["ppl_increase_pct"] for la in layer_ablation)
    layers_below_20pct = [la for la in layer_ablation if la["ppl_increase_pct"] < 20]

    print(f"\n  Min PPL increase: {min_increase:.1f}%")
    print(f"  Max PPL increase: {max_increase:.1f}%")
    print(f"  Layers with <20% impact: {len(layers_below_20pct)}/{n_layers}")

    k2_pass = len(layers_below_20pct) > 0  # at least one layer IS zeroed without >20%
    results["k2_pass"] = k2_pass
    results["k2_detail"] = {
        "min_increase_pct": round(min_increase, 2),
        "max_increase_pct": round(max_increase, 2),
        "layers_below_20pct": len(layers_below_20pct),
        "total_layers": n_layers,
    }

    # ------------------------------------------------------------------
    # Phase 3: Progressive ablation (sort by criticality, zero bottom-K)
    # ------------------------------------------------------------------
    print(f"\n[Phase 3] Progressive ablation (least critical first)")

    # Sort layers by impact (least critical first)
    sorted_layers = sorted(layer_ablation, key=lambda x: x["ppl_increase_pct"])

    progressive_results = []
    for k in [1, 3, 5, 10, 15, 20, 25, n_layers]:
        if k > n_layers:
            continue

        # Restore to composed state
        restore_base_weights(model, saved_base)
        apply_adapter_weights(model, composed)

        # Zero bottom-K layers
        zeroed_layers = [sl["layer"] for sl in sorted_layers[:k]]
        for layer_idx in zeroed_layers:
            zero_layer_base_weights(model, layer_idx)

        _, prog_ppl = compute_mean_ppl(model, tokenizer, data_dirs, DOMAINS)
        ppl_ratio = prog_ppl / composed_mean_ppl

        progressive_results.append({
            "k_zeroed": k,
            "ppl": round(prog_ppl, 4),
            "ppl_ratio_vs_composed": round(ppl_ratio, 4),
            "zeroed_layers": zeroed_layers,
        })
        print(f"  K={k:2d} zeroed: PPL={prog_ppl:.2f} ({ppl_ratio:.2f}x composed)")

    results["progressive_ablation"] = progressive_results

    # ------------------------------------------------------------------
    # Phase 4: Skeleton-only conditions
    # ------------------------------------------------------------------
    print(f"\n[Phase 4] Skeleton-only conditions")

    skeleton_results = {}

    # 4a: Random ternary, norm-matched (per-layer scale)
    print("\n  4a: Random ternary (norm-matched)...")
    restore_base_weights(model, saved_base)
    apply_adapter_weights(model, composed)
    replace_all_with_random_ternary(model, seed=SEED, scale_match=True)
    _, skel_scaled_ppl = compute_mean_ppl(model, tokenizer, data_dirs, DOMAINS)
    skel_scaled_ratio = skel_scaled_ppl / composed_mean_ppl
    print(f"    PPL={skel_scaled_ppl:.2f} ({skel_scaled_ratio:.2f}x composed)")
    skeleton_results["random_ternary_scaled"] = {
        "ppl": round(skel_scaled_ppl, 4),
        "ratio_vs_composed": round(skel_scaled_ratio, 4),
    }

    # 4b: Random ternary, unscaled
    print("\n  4b: Random ternary (unscaled)...")
    restore_base_weights(model, saved_base)
    apply_adapter_weights(model, composed)
    replace_all_with_unscaled_random(model, seed=SEED)
    _, skel_unscaled_ppl = compute_mean_ppl(model, tokenizer, data_dirs, DOMAINS)
    skel_unscaled_ratio = skel_unscaled_ppl / composed_mean_ppl
    print(f"    PPL={skel_unscaled_ppl:.2f} ({skel_unscaled_ratio:.2f}x composed)")
    skeleton_results["random_ternary_unscaled"] = {
        "ppl": round(skel_unscaled_ppl, 4),
        "ratio_vs_composed": round(skel_unscaled_ratio, 4),
    }

    # 4c: Zero base (all weights zeroed, only adapters remain)
    print("\n  4c: Zero base (adapter-only)...")
    restore_base_weights(model, saved_base)
    apply_adapter_weights(model, composed)
    for layer_idx in range(n_layers):
        zero_layer_base_weights(model, layer_idx)
    _, zero_ppl = compute_mean_ppl(model, tokenizer, data_dirs, DOMAINS)
    zero_ratio = zero_ppl / composed_mean_ppl
    print(f"    PPL={zero_ppl:.2f} ({zero_ratio:.2f}x composed)")
    skeleton_results["zero_base_adapter_only"] = {
        "ppl": round(zero_ppl, 4),
        "ratio_vs_composed": round(zero_ratio, 4),
    }

    # 4d: Random ternary, NO adapters (pure scaffold)
    print("\n  4d: Random ternary, NO adapters (pure scaffold)...")
    restore_base_weights(model, saved_base)
    # Zero all LoRA params
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.lora_a = mx.zeros_like(module.lora_a)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())
    replace_all_with_random_ternary(model, seed=SEED, scale_match=True)
    _, skel_no_adapter_ppl = compute_mean_ppl(model, tokenizer, data_dirs, DOMAINS)
    skel_no_adapter_ratio = skel_no_adapter_ppl / base_mean_ppl
    print(f"    PPL={skel_no_adapter_ppl:.2f} ({skel_no_adapter_ratio:.2f}x base)")
    skeleton_results["random_ternary_no_adapter"] = {
        "ppl": round(skel_no_adapter_ppl, 4),
        "ratio_vs_base": round(skel_no_adapter_ratio, 4),
    }

    results["skeleton_conditions"] = skeleton_results

    # K1 check: skeleton-only PPL vs pretrained-base PPL
    # Use the best skeleton condition (random ternary scaled + adapters)
    best_skeleton_ppl = skel_scaled_ppl
    k1_ratio = best_skeleton_ppl / base_mean_ppl
    k1_pass = k1_ratio <= 5.0
    results["k1_ratio"] = round(k1_ratio, 4)
    results["k1_pass"] = k1_pass
    results["k1_detail"] = {
        "best_skeleton_ppl": round(best_skeleton_ppl, 4),
        "base_ppl": round(base_mean_ppl, 4),
        "ratio": round(k1_ratio, 4),
        "threshold": 5.0,
    }

    # ------------------------------------------------------------------
    # Phase 5: Layer criticality analysis
    # ------------------------------------------------------------------
    print(f"\n[Phase 5] Layer criticality analysis")

    # Classify layers
    critical_layers = [la for la in layer_ablation if la["ppl_increase_pct"] >= 20]
    important_layers = [la for la in layer_ablation if 5 <= la["ppl_increase_pct"] < 20]
    replaceable_layers = [la for la in layer_ablation if la["ppl_increase_pct"] < 5]

    print(f"  Critical (>20% impact): {len(critical_layers)} layers")
    for la in sorted(critical_layers, key=lambda x: -x["ppl_increase_pct"])[:5]:
        print(f"    Layer {la['layer']}: +{la['ppl_increase_pct']:.1f}%")

    print(f"  Important (5-20% impact): {len(important_layers)} layers")
    print(f"  Replaceable (<5% impact): {len(replaceable_layers)} layers")
    for la in sorted(replaceable_layers, key=lambda x: x["ppl_increase_pct"])[:5]:
        print(f"    Layer {la['layer']}: +{la['ppl_increase_pct']:.1f}%")

    results["layer_classification"] = {
        "critical_gt20pct": [la["layer"] for la in critical_layers],
        "important_5_20pct": [la["layer"] for la in important_layers],
        "replaceable_lt5pct": [la["layer"] for la in replaceable_layers],
        "n_critical": len(critical_layers),
        "n_important": len(important_layers),
        "n_replaceable": len(replaceable_layers),
    }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    results["elapsed_s"] = round(elapsed, 1)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nBase model mean PPL:        {base_mean_ppl:.2f}")
    print(f"Composed (1/N, N=5) PPL:    {composed_mean_ppl:.2f}")
    print(f"Composition ratio:          {composed_mean_ppl/base_mean_ppl:.4f}x")
    print(f"\nSkeleton conditions:")
    print(f"  Random ternary (scaled):  {skel_scaled_ppl:.2f} ({skel_scaled_ratio:.2f}x composed)")
    print(f"  Random ternary (unscaled):{skel_unscaled_ppl:.2f} ({skel_unscaled_ratio:.2f}x composed)")
    print(f"  Zero base + adapters:     {zero_ppl:.2f} ({zero_ratio:.2f}x composed)")
    print(f"  Random scaffold (no adapt):{skel_no_adapter_ppl:.2f}")
    print(f"\nLayer criticality:")
    print(f"  Critical (>20%):    {len(critical_layers)}/{n_layers}")
    print(f"  Important (5-20%):  {len(important_layers)}/{n_layers}")
    print(f"  Replaceable (<5%):  {len(replaceable_layers)}/{n_layers}")
    print(f"\nKill criteria:")
    print(f"  K1: skeleton PPL / base PPL = {k1_ratio:.2f}x (threshold 5x) -> {'PASS' if k1_pass else 'KILL'}")
    print(f"  K2: {len(layers_below_20pct)}/{n_layers} layers <20% impact -> {'PASS' if k2_pass else 'KILL'}")
    print(f"\nElapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Verdict
    if k1_pass and k2_pass:
        verdict = "SUPPORTED"
    elif k1_pass or k2_pass:
        verdict = "PARTIAL"
    else:
        verdict = "KILLED"
    results["verdict"] = verdict
    print(f"\nVerdict: {verdict}")

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
