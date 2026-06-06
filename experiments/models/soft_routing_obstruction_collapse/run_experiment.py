#!/usr/bin/env python3
"""Soft-Routing Obstruction Collapse: does Gumbel-sigmoid activate >=3 adapters?

Measures per-token activation count of Gumbel-sigmoid routing on 5-domain BitNet
composition. Tests whether natural routing places composition in the H^1=0 regime
(obstruction-free, per Finding #242).

Kill criteria:
  K1 (#650): Mean activation count <2.5 per token -> FAIL (routing too sparse)
  K2 (#651): PPL at forced k=3 worse than k=2 by >5% -> FAIL

Type: Verification (of Finding #242 phase transition + Finding #185 routing)
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
import mlx.optimizers as opt
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

SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
MAX_SEQ_LENGTH = 256
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

# Per-domain scales (Finding #217)
DOMAIN_SCALES = {
    "medical": 20.0,
    "code": 20.0,
    "math": 20.0,
    "legal": 4.0,
    "finance": 1.0,
}

# Router hidden dim
ROUTER_HIDDEN = 128

# Gumbel-sigmoid temperature (from ablation study, Finding #185)
TEMPERATURE = 1.0

# Activation threshold for counting "active" adapters
GATE_THRESHOLD = 0.5

# Number of router training steps
ROUTER_TRAIN_STEPS = 300
ROUTER_LR = 3e-4


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
# Model utilities (reused from sheaf_cohomology_dim)
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


def save_base_weights(model):
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


def premerge_weighted_adapters(model, base_weights, skeleton, adapters, weights):
    """Pre-merge multiple adapters with given weights into model.

    Args:
        weights: dict domain -> float weight (gate value * scale)
    """
    restore_base_weights(model, base_weights)

    n_layers = len(model.model.layers)
    for domain, w in weights.items():
        if abs(w) < 1e-8:
            continue
        di = DOMAINS.index(domain)
        adapter = adapters[domain]
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
                delta = w * (b_mx.T @ a_mx.T)
                module.weight = module.weight + delta

    mx.eval(model.parameters())


# ============================================================================
# Gumbel-Sigmoid Router
# ============================================================================

class GumbelSigmoidRouter(nn.Module):
    """Independent Bernoulli gates with Gumbel noise for differentiable routing."""

    def __init__(self, input_dim, n_adapters, hidden_dim=128):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, n_adapters)
        self.n_adapters = n_adapters

    def __call__(self, h, temperature=1.0, hard=False):
        """
        Args:
            h: hidden state [batch, seq, d] or [batch, d]
            temperature: Gumbel temperature
            hard: if True, use deterministic sigmoid (no Gumbel noise)

        Returns:
            logits: raw gate logits [batch, ..., n_adapters]
            gates: gate values in [0, 1] [batch, ..., n_adapters]
        """
        z = nn.gelu(self.proj(h))
        logits = self.gate(z)

        if hard:
            gates = mx.sigmoid(logits)
        else:
            u = mx.random.uniform(shape=logits.shape)
            u = mx.clip(u, 1e-6, 1.0 - 1e-6)
            gumbel_noise = -mx.log(-mx.log(u))
            gates = mx.sigmoid((logits + gumbel_noise) / temperature)

        return logits, gates


# ============================================================================
# Data loading
# ============================================================================

def load_all_samples():
    samples = []
    for domain in DOMAINS:
        fpath = DATA_DIR / domain / "valid.jsonl"
        with open(fpath) as f:
            for line in f:
                obj = json.loads(line)
                samples.append({"text": obj["text"], "domain": domain})
    log(f"  Loaded {len(samples)} samples from {len(DOMAINS)} domains")
    return samples


# ============================================================================
# Phase 1: Extract hidden states for router training
# ============================================================================

def phase_extract_hidden_states(samples):
    """Extract mean-pooled hidden states from the base model for router training."""
    log("\n" + "=" * 70)
    log("[Phase 1] Extract hidden states from base model")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    mx.eval(model.parameters())
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    log_memory("post-load")

    hidden_dim = model.model.embed_tokens.weight.shape[1]
    log(f"  Hidden dim: {hidden_dim}")

    # Extract hidden states at the middle layer (layer 15)
    extract_layer = 15
    hidden_states = []

    for i, sample in enumerate(samples):
        tokens = tokenizer.encode(sample["text"])
        if len(tokens) < 2:
            hidden_states.append(np.zeros(hidden_dim, dtype=np.float32))
            continue
        tokens = tokens[:MAX_SEQ_LENGTH + 1]
        x = mx.array(tokens[:-1])[None, :]

        h = model.model.embed_tokens(x)
        mask = create_attention_mask(h, cache=None)

        for li, layer_module in enumerate(model.model.layers):
            h = layer_module(h, mask=mask, cache=None)
            if li == extract_layer:
                # Mean pool over sequence
                hidden_vec = mx.mean(h, axis=1)  # [1, d]
                mx.eval(hidden_vec)
                hidden_states.append(np.array(hidden_vec[0].astype(mx.float32)))
                break

        del h, x
        if (i + 1) % 50 == 0:
            log(f"    Processed {i+1}/{len(samples)} samples")

    elapsed = time.time() - t0
    log(f"  Phase 1 complete in {elapsed:.1f}s")
    log_memory("post-extract")

    # Save base weights for later use
    base_weights = save_base_weights(model)

    return np.stack(hidden_states), hidden_dim, model, tokenizer, base_weights


# ============================================================================
# Phase 2: Train Gumbel-sigmoid router + analyze activation statistics
# ============================================================================

def phase_train_router_and_analyze(hidden_states_np, hidden_dim, samples):
    """Train router on PPL-based targets and analyze activation statistics."""
    log("\n" + "=" * 70)
    log("[Phase 2] Train Gumbel-sigmoid router + activation analysis")
    log("=" * 70)

    n_samples = len(samples)
    n_adapters = len(DOMAINS)

    # Normalize hidden states to unit norm (prevents logit explosion)
    norms = np.linalg.norm(hidden_states_np, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    hidden_states_np = hidden_states_np / norms
    log(f"  Hidden state norms: mean={norms.mean():.1f}, std={norms.std():.1f}")

    # Create training targets: one-hot domain labels
    domain_to_idx = {d: i for i, d in enumerate(DOMAINS)}
    labels = np.array([domain_to_idx[s["domain"]] for s in samples])

    # Router training: learn to activate the correct domain adapter(s)
    # Target: one-hot for the source domain (primary signal)
    router = GumbelSigmoidRouter(hidden_dim, n_adapters, ROUTER_HIDDEN)
    optimizer = opt.Adam(learning_rate=ROUTER_LR)

    hidden_states_mx = mx.array(hidden_states_np)
    labels_mx = mx.array(labels)

    # Training: binary cross-entropy per gate
    # Target gate_i = 1 if sample is from domain i, else 0
    # This teaches the router to activate the relevant adapter
    # Build soft targets: primary domain = 0.8, others = 0.05
    # This encourages the router to keep non-primary gates partially open,
    # reflecting that related domains have partial relevance.
    # Label smoothing prevents logit explosion and tests whether Gumbel-sigmoid
    # naturally trends toward multi-adapter activation.
    LABEL_SMOOTH = 0.05
    targets_np = np.full((n_samples, n_adapters), LABEL_SMOOTH, dtype=np.float32)
    for i in range(n_samples):
        targets_np[i, labels[i]] = 1.0 - LABEL_SMOOTH * (n_adapters - 1)
    targets = mx.array(targets_np)

    def loss_fn(router, h, tgt):
        _, gates = router(h, temperature=TEMPERATURE)
        # Binary cross-entropy per gate
        eps = 1e-7
        bce = -(tgt * mx.log(gates + eps) + (1 - tgt) * mx.log(1 - gates + eps))
        return mx.mean(bce)

    loss_and_grad = nn.value_and_grad(router, loss_fn)

    log(f"  Training router: {ROUTER_TRAIN_STEPS} steps, lr={ROUTER_LR}")
    gc.disable()
    batch_size = min(64, n_samples)
    for step in range(ROUTER_TRAIN_STEPS):
        # Random batch
        idx = mx.array(np.random.choice(n_samples, batch_size, replace=False))
        h_batch = hidden_states_mx[idx]
        t_batch = targets[idx]

        loss, grads = loss_and_grad(router, h_batch, t_batch)
        optimizer.update(router, grads)
        mx.eval(router.parameters(), optimizer.state, loss)

        if (step + 1) % 50 == 0:
            log(f"    Step {step+1}/{ROUTER_TRAIN_STEPS}: loss={loss.item():.4f}")

    gc.enable()
    gc.collect()

    # Analyze activation statistics on full dataset
    log("\n  Analyzing activation statistics...")

    # Deterministic pass (no Gumbel noise) for stable analysis
    all_logits, all_gates_hard = router(hidden_states_mx, temperature=TEMPERATURE, hard=True)
    mx.eval(all_logits, all_gates_hard)

    # Stochastic pass (with Gumbel noise) - do multiple samples
    n_stochastic_runs = 10
    stochastic_counts = []
    for run in range(n_stochastic_runs):
        _, gates_s = router(hidden_states_mx, temperature=TEMPERATURE, hard=False)
        mx.eval(gates_s)
        counts = mx.sum((gates_s > GATE_THRESHOLD).astype(mx.float32), axis=-1)
        mx.eval(counts)
        stochastic_counts.append(np.array(counts))

    stochastic_counts = np.stack(stochastic_counts)  # [n_runs, n_samples]

    # Hard (deterministic) activation analysis
    hard_active = (np.array(all_gates_hard) > GATE_THRESHOLD).astype(int)
    hard_counts = hard_active.sum(axis=-1)  # [n_samples]

    # Stochastic activation analysis (averaged over runs)
    mean_stochastic_counts = stochastic_counts.mean(axis=0)  # [n_samples]

    # Per-domain statistics
    domain_stats = {}
    for d in DOMAINS:
        mask = np.array([s["domain"] == d for s in samples])
        domain_stats[d] = {
            "hard_mean_k": float(hard_counts[mask].mean()),
            "hard_frac_ge3": float((hard_counts[mask] >= 3).mean()),
            "stochastic_mean_k": float(mean_stochastic_counts[mask].mean()),
            "stochastic_frac_ge3": float((mean_stochastic_counts[mask] >= 3).mean()),
        }

    # Overall statistics
    overall_stats = {
        "hard_mean_k": float(hard_counts.mean()),
        "hard_std_k": float(hard_counts.std()),
        "hard_frac_ge3": float((hard_counts >= 3).mean()),
        "hard_frac_ge2": float((hard_counts >= 2).mean()),
        "hard_distribution": {
            str(k): float((hard_counts == k).mean())
            for k in range(n_adapters + 1)
        },
        "stochastic_mean_k": float(mean_stochastic_counts.mean()),
        "stochastic_std_k": float(mean_stochastic_counts.std()),
        "stochastic_frac_ge3": float((mean_stochastic_counts >= 3).mean()),
        "stochastic_frac_ge2": float((mean_stochastic_counts >= 2).mean()),
    }

    # Gate logit analysis
    logits_np = np.array(all_logits)
    logit_stats = {
        "mean_per_adapter": logits_np.mean(axis=0).tolist(),
        "std_per_adapter": logits_np.std(axis=0).tolist(),
        "mean_overall": float(logits_np.mean()),
        "frac_positive": float((logits_np > 0).mean()),
    }

    # Gate value analysis (deterministic)
    gates_np = np.array(all_gates_hard)
    gate_stats = {
        "mean_per_adapter": gates_np.mean(axis=0).tolist(),
        "std_per_adapter": gates_np.std(axis=0).tolist(),
    }

    log(f"\n  === ACTIVATION STATISTICS ===")
    log(f"  Hard (deterministic sigmoid):")
    log(f"    Mean K = {overall_stats['hard_mean_k']:.2f} +/- {overall_stats['hard_std_k']:.2f}")
    log(f"    Frac K>=3 = {overall_stats['hard_frac_ge3']:.3f}")
    log(f"    Frac K>=2 = {overall_stats['hard_frac_ge2']:.3f}")
    log(f"    Distribution: {overall_stats['hard_distribution']}")
    log(f"  Stochastic (Gumbel-sigmoid, {n_stochastic_runs} runs):")
    log(f"    Mean K = {overall_stats['stochastic_mean_k']:.2f}")
    log(f"    Frac K>=3 = {overall_stats['stochastic_frac_ge3']:.3f}")
    log(f"  Logits: mean={logit_stats['mean_overall']:.3f}, frac_positive={logit_stats['frac_positive']:.3f}")
    log(f"  Per-adapter logit means: {[f'{m:.2f}' for m in logit_stats['mean_per_adapter']]}")

    for d in DOMAINS:
        ds = domain_stats[d]
        log(f"    {d}: hard_K={ds['hard_mean_k']:.2f}, frac_ge3={ds['hard_frac_ge3']:.3f}")

    return {
        "overall": overall_stats,
        "per_domain": domain_stats,
        "logit_stats": logit_stats,
        "gate_stats": gate_stats,
        "router_final_loss": loss.item(),
    }, router


# ============================================================================
# Phase 3: PPL comparison under different routing regimes
# ============================================================================

def phase_ppl_comparison(model, tokenizer, base_weights, router, hidden_states_np, samples):
    """Compare PPL under natural, forced-k=2, and forced-k=3 routing."""
    log("\n" + "=" * 70)
    log("[Phase 3] PPL comparison: natural vs forced k=2 vs forced k=3")
    log("=" * 70)

    t0 = time.time()

    # Load all adapters and skeleton
    skeleton = load_skeleton()
    adapters = {}
    for domain in DOMAINS:
        adapters[domain] = load_adapter(domain)

    hidden_states_mx = mx.array(hidden_states_np)

    # Get deterministic gate values for all samples
    _, all_gates = router(hidden_states_mx, temperature=TEMPERATURE, hard=True)
    mx.eval(all_gates)
    gates_np = np.array(all_gates)

    # Use a representative subset for PPL comparison (full set too slow with per-sample merging)
    # Take 10 samples per domain = 50 total
    subset_indices = []
    for domain in DOMAINS:
        domain_indices = [i for i, s in enumerate(samples) if s["domain"] == domain]
        np.random.seed(SEED)
        chosen = np.random.choice(domain_indices, min(10, len(domain_indices)), replace=False)
        subset_indices.extend(chosen.tolist())

    subset_samples = [samples[i] for i in subset_indices]
    subset_gates = gates_np[subset_indices]

    log(f"  Using {len(subset_samples)} samples for PPL comparison")

    def compute_ppl_for_regime(regime_name, gate_fn):
        """Compute PPL for a routing regime defined by gate_fn."""
        ppls = []
        for i, sample in enumerate(subset_samples):
            tokens = tokenizer.encode(sample["text"])
            if len(tokens) < 2:
                ppls.append(float("inf"))
                continue
            tokens = tokens[:MAX_SEQ_LENGTH + 1]

            # Get routing weights for this sample
            gate_values = subset_gates[i]  # [n_adapters]
            weights = gate_fn(gate_values)

            # Pre-merge adapters with these weights
            premerge_weighted_adapters(model, base_weights, skeleton, adapters, weights)

            # Compute PPL
            x = mx.array(tokens[:-1])[None, :]
            y = mx.array(tokens[1:])[None, :]
            logits = model(x)
            loss = nn.losses.cross_entropy(logits, y, reduction="sum")
            mx.eval(loss)
            n_tokens = y.size
            ppl = math.exp(min(loss.item() / n_tokens, 100))
            ppls.append(ppl)
            del logits, loss, x, y

            if (i + 1) % 25 == 0:
                log(f"    [{regime_name}] {i+1}/{len(subset_samples)}: mean_ppl={np.mean(ppls):.2f}")

        return ppls

    # Regime 1: Natural Gumbel-sigmoid (use gate values as-is, scaled)
    def natural_gate_fn(gv):
        weights = {}
        for j, d in enumerate(DOMAINS):
            weights[d] = float(gv[j]) * DOMAIN_SCALES[d]
        return weights

    natural_ppls = compute_ppl_for_regime("natural", natural_gate_fn)

    # Regime 2: Forced k=2 (top-2 gates only)
    def forced_k2_gate_fn(gv):
        top2 = np.argsort(-gv)[:2]
        weights = {}
        for j, d in enumerate(DOMAINS):
            if j in top2:
                weights[d] = float(gv[j]) * DOMAIN_SCALES[d]
            else:
                weights[d] = 0.0
        return weights

    forced_k2_ppls = compute_ppl_for_regime("forced-k2", forced_k2_gate_fn)

    # Regime 3: Forced k=3 (top-3 gates only)
    def forced_k3_gate_fn(gv):
        top3 = np.argsort(-gv)[:3]
        weights = {}
        for j, d in enumerate(DOMAINS):
            if j in top3:
                weights[d] = float(gv[j]) * DOMAIN_SCALES[d]
            else:
                weights[d] = 0.0
        return weights

    forced_k3_ppls = compute_ppl_for_regime("forced-k3", forced_k3_gate_fn)

    # Regime 4: Equal-weight all (baseline)
    def equal_all_gate_fn(gv):
        weights = {}
        for j, d in enumerate(DOMAINS):
            weights[d] = DOMAIN_SCALES[d] / len(DOMAINS)
        return weights

    equal_all_ppls = compute_ppl_for_regime("equal-all", equal_all_gate_fn)

    elapsed = time.time() - t0
    log(f"\n  Phase 3 complete in {elapsed:.1f}s")

    # Compute statistics
    def ppl_stats(ppls, name):
        valid = [p for p in ppls if p < 1e10]
        if not valid:
            return {"mean": float("inf"), "median": float("inf"), "name": name}
        return {
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
            "std": float(np.std(valid)),
            "name": name,
        }

    natural_stats = ppl_stats(natural_ppls, "natural")
    k2_stats = ppl_stats(forced_k2_ppls, "forced-k2")
    k3_stats = ppl_stats(forced_k3_ppls, "forced-k3")
    equal_stats = ppl_stats(equal_all_ppls, "equal-all")

    ppl_ratio_k3_k2 = k3_stats["mean"] / k2_stats["mean"] if k2_stats["mean"] > 0 else float("inf")

    log(f"\n  === PPL COMPARISON ===")
    log(f"  Natural:   mean={natural_stats['mean']:.2f}, median={natural_stats['median']:.2f}")
    log(f"  Forced k2: mean={k2_stats['mean']:.2f}, median={k2_stats['median']:.2f}")
    log(f"  Forced k3: mean={k3_stats['mean']:.2f}, median={k3_stats['median']:.2f}")
    log(f"  Equal all: mean={equal_stats['mean']:.2f}, median={equal_stats['median']:.2f}")
    log(f"  PPL ratio (k3/k2): {ppl_ratio_k3_k2:.4f}")

    # Per-domain PPL breakdown
    per_domain_ppl = {}
    for d in DOMAINS:
        mask = [s["domain"] == d for s in subset_samples]
        for regime_name, ppls in [("natural", natural_ppls), ("forced_k2", forced_k2_ppls),
                                   ("forced_k3", forced_k3_ppls), ("equal_all", equal_all_ppls)]:
            domain_ppls = [p for p, m in zip(ppls, mask) if m and p < 1e10]
            if d not in per_domain_ppl:
                per_domain_ppl[d] = {}
            per_domain_ppl[d][regime_name] = float(np.mean(domain_ppls)) if domain_ppls else float("inf")

    cleanup(skeleton)
    del adapters
    gc.collect()
    mx.clear_cache()

    return {
        "natural": natural_stats,
        "forced_k2": k2_stats,
        "forced_k3": k3_stats,
        "equal_all": equal_stats,
        "ppl_ratio_k3_over_k2": ppl_ratio_k3_k2,
        "per_domain_ppl": per_domain_ppl,
        "n_samples": len(subset_samples),
    }


# ============================================================================
# Phase 4: Cech nerve analysis of routing cover
# ============================================================================

def phase_nerve_analysis(gates_np, samples):
    """Compute Cech nerve of the routing cover and check H^1."""
    log("\n" + "=" * 70)
    log("[Phase 4] Cech nerve of routing cover")
    log("=" * 70)

    n_samples = len(samples)
    active = (gates_np > GATE_THRESHOLD).astype(int)

    results = {}
    for threshold_name, act_matrix in [("gate_0.5", active)]:
        # Build cover: U_i = {samples where adapter i is active}
        cover = {d: set() for d in DOMAINS}
        for idx in range(n_samples):
            for j, d in enumerate(DOMAINS):
                if act_matrix[idx, j]:
                    cover[d].add(idx)

        cover_sizes = {d: len(cover[d]) for d in DOMAINS}
        log(f"  Cover sizes ({threshold_name}): {cover_sizes}")

        # Compute Cech nerve
        vertices = list(DOMAINS)
        edges = []
        edge_sizes = {}
        for d1, d2 in combinations(DOMAINS, 2):
            overlap = cover[d1] & cover[d2]
            if len(overlap) > 0:
                edges.append((d1, d2))
                edge_sizes[f"{d1}_{d2}"] = len(overlap)

        triangles = []
        triangle_sizes = {}
        for d1, d2, d3 in combinations(DOMAINS, 3):
            overlap = cover[d1] & cover[d2] & cover[d3]
            if len(overlap) > 0:
                triangles.append((d1, d2, d3))
                triangle_sizes[f"{d1}_{d2}_{d3}"] = len(overlap)

        n_v, n_e, n_t = len(vertices), len(edges), len(triangles)

        # Compute beta_1 via rank of boundary maps
        vertex_idx = {v: i for i, v in enumerate(vertices)}
        edge_idx = {e: i for i, e in enumerate(edges)}

        delta_0 = np.zeros((n_e, n_v))
        for ei, (d1, d2) in enumerate(edges):
            delta_0[ei, vertex_idx[d1]] = -1.0
            delta_0[ei, vertex_idx[d2]] = +1.0

        delta_1 = np.zeros((len(triangles), n_e)) if triangles else np.zeros((0, n_e))
        for ti, (d1, d2, d3) in enumerate(triangles):
            for pair, sign in [((d1, d2), +1), ((d1, d3), -1), ((d2, d3), +1)]:
                if pair in edge_idx:
                    delta_1[ti, edge_idx[pair]] = sign

        rank_d0 = np.linalg.matrix_rank(delta_0) if n_e > 0 else 0
        rank_d1 = np.linalg.matrix_rank(delta_1) if len(triangles) > 0 else 0

        beta_0 = n_v - rank_d0
        beta_1 = n_e - rank_d0 - rank_d1

        log(f"  Nerve: V={n_v}, E={n_e}, T={n_t}")
        log(f"  Edges: {[f'{d1}-{d2}' for d1, d2 in edges]}")
        log(f"  Triangles: {[f'{d1}-{d2}-{d3}' for d1, d2, d3 in triangles]}")
        log(f"  beta_0={beta_0}, beta_1={beta_1}")
        log(f"  rank(delta_0)={rank_d0}, rank(delta_1)={rank_d1}")

        results[threshold_name] = {
            "cover_sizes": cover_sizes,
            "n_vertices": n_v,
            "n_edges": n_e,
            "n_triangles": n_t,
            "edges": [list(e) for e in edges],
            "edge_sizes": edge_sizes,
            "triangles": [list(t) for t in triangles],
            "triangle_sizes": triangle_sizes,
            "beta_0": int(beta_0),
            "beta_1": int(beta_1),
            "rank_delta_0": int(rank_d0),
            "rank_delta_1": int(rank_d1),
        }

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    mx.random.seed(SEED)
    np.random.seed(SEED)

    log("=" * 70)
    log("Soft-Routing Obstruction Collapse")
    log("Does Gumbel-sigmoid activate >=3 adapters per token?")
    log("=" * 70)
    log_memory("start")

    # Load data
    samples = load_all_samples()

    # Phase 1: Extract hidden states
    hidden_states, hidden_dim, model, tokenizer, base_weights = phase_extract_hidden_states(samples)
    log_memory("post-phase1")

    # Phase 2: Train router + analyze activations
    activation_results, router = phase_train_router_and_analyze(
        hidden_states, hidden_dim, samples
    )
    log_memory("post-phase2")

    # Get full gates for nerve analysis
    hidden_states_mx = mx.array(hidden_states)
    _, all_gates = router(hidden_states_mx, temperature=TEMPERATURE, hard=True)
    mx.eval(all_gates)
    gates_np = np.array(all_gates)
    del hidden_states_mx

    # Phase 3: PPL comparison
    ppl_results = phase_ppl_comparison(
        model, tokenizer, base_weights, router, hidden_states, samples
    )
    log_memory("post-phase3")

    # Phase 4: Nerve analysis of routing cover
    nerve_results = phase_nerve_analysis(gates_np, samples)
    log_memory("post-phase4")

    # Kill criteria assessment
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    mean_k = activation_results["overall"]["hard_mean_k"]
    k1_pass = mean_k >= 2.5
    log(f"  K1 (#650): Mean activation count = {mean_k:.2f} (threshold >= 2.5)")
    log(f"    -> {'PASS' if k1_pass else 'FAIL'}")

    ppl_ratio = ppl_results["ppl_ratio_k3_over_k2"]
    k2_pass = ppl_ratio <= 1.05
    log(f"  K2 (#651): PPL ratio k3/k2 = {ppl_ratio:.4f} (threshold <= 1.05)")
    log(f"    -> {'PASS' if k2_pass else 'FAIL'}")

    # H^1 of routing cover
    nerve_h1 = nerve_results.get("gate_0.5", {}).get("beta_1", -1)
    log(f"  H^1 of routing cover: {nerve_h1}")
    log(f"  (Finding #242 predicted: H^1=0 when K>=3)")

    total_time = time.time() - t_start

    # Save results
    final_results = {
        "experiment": "soft_routing_obstruction_collapse",
        "model": MODEL_ID,
        "domains": DOMAINS,
        "n_samples": len(samples),
        "temperature": TEMPERATURE,
        "gate_threshold": GATE_THRESHOLD,
        "router_train_steps": ROUTER_TRAIN_STEPS,
        "activation_analysis": activation_results,
        "ppl_comparison": ppl_results,
        "nerve_analysis": nerve_results,
        "kill_criteria": {
            "k1_mean_activation_ge_2.5": {
                "value": mean_k,
                "threshold": 2.5,
                "pass": k1_pass,
            },
            "k2_ppl_ratio_k3_k2_le_1.05": {
                "value": ppl_ratio,
                "threshold": 1.05,
                "pass": k2_pass,
            },
        },
        "routing_cover_H1": nerve_h1,
        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(final_results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_time:.1f}s")

    # Cleanup
    cleanup(model, tokenizer)
    del router, base_weights, hidden_states, gates_np
    gc.collect()
    mx.clear_cache()


if __name__ == "__main__":
    main()
