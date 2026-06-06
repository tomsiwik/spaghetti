#!/usr/bin/env python3
"""
Scaffold Fresh Adapters: Train fresh LoRA on random ternary scaffold

Tests whether fresh LoRA adapters can learn useful features when the base model
is a RANDOM ternary scaffold {-1,0,1} (not pretrained).

Prior experiment (basefree_exploration) tested pretrained adapters on random scaffold
and KILLED it (PPL 319M). That was the WRONG test: those adapters encoded directions
in pretrained coordinate space. THIS experiment trains FRESH adapters directly on the
scaffold -- the scaffold's coordinate system is internally consistent.

Two conditions, identical training:
  A: Pretrained BitNet-2B-4T base + fresh LoRA (STE ternary) -- CONTROL
  B: Random ternary scaffold + fresh LoRA (STE ternary) -- EXPERIMENTAL

FreezeNet (arXiv:2011.14087) shows random frozen weights support gradient flow.
TernaryLM (arXiv:2602.07374) shows STE enables learning on ternary architectures.

Kill criteria:
  K1: scaffold-trained adapter PPL > 5x pretrained-base-trained adapter PPL per domain
  K2: adapters fail to converge on scaffold (loss does not decrease over 1000 steps)

Platform: Apple Silicon MLX, $0.
Reuses: data from bitnet_2b_real_composition (5 domains).
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
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm import load
from mlx_lm.models.bitlinear_layers import BitLinear

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
TRAIN_ITERS = 400        # 400 steps per adapter (matches prior experiments)
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 128      # Short seqs for speed
VAL_BATCHES = 25
SEED = 42

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse data from bitnet_2b_real_composition
DATA_SOURCE = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"

DOMAINS = ["medical", "math", "legal", "creative"]
# Note: bitnet_2b_real_composition uses "python" not "code"
DOMAIN_DATA_MAP = {
    "medical": "medical",
    "math": "math",
    "legal": "legal",
    "creative": "creative",
}

TARGET_KEYS = {
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
}


# ===========================================================================
# Ternary weight unpacking (from bitnet_2b_real_composition)
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
    """Replace BitLinear with nn.Linear (unpacked bfloat16). Returns weight info for scaffold."""
    count = 0
    weight_info = []  # (layer_idx, key, shape, frobenius_norm)

    for layer_idx, layer in enumerate(model.model.layers):
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

                norm = mx.sqrt(mx.sum(unpacked_w ** 2)).item()
                weight_info.append({
                    "layer": layer_idx, "key": key,
                    "shape": list(unpacked_w.shape), "norm": norm,
                })
        if updates:
            layer.update_modules(tree_unflatten(updates))

    mx.eval(model.parameters())
    print(f"  Replaced {count} BitLinear -> nn.Linear")
    return model, weight_info


def replace_with_random_ternary_scaffold(model, weight_info, seed=42):
    """Replace all nn.Linear weights with random ternary {-1,0,1} * per-layer-scale.

    Matches the shape and Frobenius norm of the pretrained weights.
    This ensures gradient magnitudes are comparable.
    """
    mx.random.seed(seed)
    count = 0

    for info in weight_info:
        layer_idx = info["layer"]
        key = info["key"]
        shape = info["shape"]
        target_norm = info["norm"]

        layer = model.model.layers[layer_idx]

        # Find the module by key path
        parts = key.split(".")
        module = layer
        for p in parts:
            module = getattr(module, p)

        if not isinstance(module, nn.Linear):
            continue

        # Random ternary: uniform over {-1, 0, 1}
        # Generate random integers in {0, 1, 2} then shift to {-1, 0, 1}
        rand_vals = mx.random.randint(0, 3, shape=shape).astype(mx.bfloat16) - 1.0

        # Scale to match pretrained Frobenius norm
        current_norm = mx.sqrt(mx.sum(rand_vals ** 2)).item()
        if current_norm > 1e-6:
            scale_factor = target_norm / current_norm
            rand_vals = rand_vals * scale_factor

        module.weight = rand_vals
        count += 1

    mx.eval(model.parameters())
    print(f"  Replaced {count} layers with norm-matched random ternary scaffold")
    return model


# ===========================================================================
# LoRA with STE ternary quantization
# ===========================================================================
class TernaryLoRALinear(nn.Module):
    """LoRA with STE ternary quantization on adapter weights.

    Forward pass uses ternary-quantized A and B.
    Backward pass uses STE (gradients flow through quantization).
    """
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
        """Ternary quantize with STE: forward uses {-1,0,1}*alpha, backward passes through."""
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


def zero_lora_params(model):
    """Reset LoRA params: A to random, B to zero."""
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                in_dims = module.lora_a.shape[0]
                s = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(low=-s, high=s, shape=module.lora_a.shape)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def get_lora_params(model):
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_a" in name or "lora_b" in name:
            params[name] = mx.array(p)
    mx.eval(params)
    return params


# ===========================================================================
# PPL evaluation
# ===========================================================================
def compute_ppl(model, tokenizer, data_path, max_batches=25, max_seq_len=128):
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
        tokens = tokens[:max_seq_len + 1]

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


# ===========================================================================
# Training loop for one domain
# ===========================================================================
def train_adapter(model, tokenizer, data_dir, domain, n_iters, lr, max_seq_len):
    """Train one LoRA adapter. Returns (losses, train_time, final_ppl)."""

    # Load training data
    train_texts = []
    with open(data_dir / "train.jsonl") as f:
        for line in f:
            train_texts.append(json.loads(line)["text"])

    train_tokens = []
    for text in train_texts:
        toks = tokenizer.encode(text)
        if len(toks) > 2:
            train_tokens.append(mx.array(toks[:max_seq_len + 1]))

    print(f"    {len(train_tokens)} training sequences")

    # Freeze base, unfreeze LoRA
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    optimizer = opt.Adam(learning_rate=lr)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    t_start = time.time()
    losses = []
    for step in range(n_iters):
        idx = step % len(train_tokens)
        tokens = train_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()
        losses.append(loss_val)

        if (step + 1) % 100 == 0 or step == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            print(f"      Step {step+1}/{n_iters}: loss={loss_val:.4f} (avg50={avg:.4f})")

    train_time = time.time() - t_start

    # Compute final PPL
    ppl = compute_ppl(model, tokenizer, data_dir, max_batches=VAL_BATCHES, max_seq_len=max_seq_len)

    return losses, train_time, ppl


# ===========================================================================
# Composition
# ===========================================================================
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
# Cosine similarity between adapter pairs
# ===========================================================================
def compute_adapter_cosines(adapter_dict):
    """Compute pairwise |cosine| between flattened adapter params."""
    domains = list(adapter_dict.keys())
    vecs = {}
    for d in domains:
        params = adapter_dict[d]
        flat = mx.concatenate([p.reshape(-1) for p in params.values()])
        vecs[d] = flat

    cosines = {}
    for i, d1 in enumerate(domains):
        for d2 in domains[i+1:]:
            v1, v2 = vecs[d1], vecs[d2]
            cos = mx.abs(mx.sum(v1 * v2) / (mx.sqrt(mx.sum(v1**2)) * mx.sqrt(mx.sum(v2**2)) + 1e-10))
            mx.eval(cos)
            cosines[f"{d1}-{d2}"] = round(cos.item(), 6)

    return cosines


# ===========================================================================
# Main
# ===========================================================================
def main():
    results = {
        "experiment": "bitnet_scaffold_fresh_adapters",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "train_iters": TRAIN_ITERS,
        "domains": DOMAINS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=" * 70)
    print("Scaffold Fresh Adapters: Train LoRA on Random Ternary Scaffold")
    print("=" * 70)

    # ==================================================================
    # Phase 0: Load model and unpack
    # ==================================================================
    print("\n[Phase 0] Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    print("  Unpacking ternary weights...")
    t1 = time.time()
    model, weight_info = replace_bitlinear_with_linear(model)
    unpack_time = time.time() - t1
    print(f"  Unpacked in {unpack_time:.1f}s")
    print(f"  Weight info: {len(weight_info)} linear layers recorded")

    # ==================================================================
    # Phase 1: Verify data
    # ==================================================================
    print("\n[Phase 1] Verifying data...")
    data_dirs = {}
    for domain in DOMAINS:
        data_name = DOMAIN_DATA_MAP[domain]
        d = DATA_SOURCE / data_name
        if not (d / "valid.jsonl").exists():
            print(f"  FATAL: No data for {domain} at {d}")
            return
        data_dirs[domain] = d
        n_train = sum(1 for _ in open(d / "train.jsonl"))
        n_val = sum(1 for _ in open(d / "valid.jsonl"))
        print(f"  {domain}: {n_train} train, {n_val} val")

    # ==================================================================
    # Phase 2: CONDITION A -- Pretrained base + fresh LoRA (STE ternary)
    # ==================================================================
    print("\n" + "=" * 70)
    print("[Phase 2] CONDITION A: Pretrained base + fresh LoRA (STE ternary)")
    print("=" * 70)

    # Compute base PPL first
    print("\n  Computing pretrained base PPL...")
    base_ppls_pretrained = {}
    for domain in DOMAINS:
        ppl = compute_ppl(model, tokenizer, data_dirs[domain], max_seq_len=MAX_SEQ_LENGTH)
        base_ppls_pretrained[domain] = ppl
        print(f"    {domain}: base PPL = {ppl:.2f}")
    results["base_ppls_pretrained"] = base_ppls_pretrained

    # Apply LoRA
    model = apply_ternary_lora(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Train adapters on pretrained base
    pretrained_results = {}
    pretrained_adapters = {}

    for domain in DOMAINS:
        print(f"\n  --- Training {domain} adapter on PRETRAINED base ---")
        mx.random.seed(SEED + hash(domain) % 1000)
        zero_lora_params(model)

        losses, train_time, ppl = train_adapter(
            model, tokenizer, data_dirs[domain], domain,
            n_iters=TRAIN_ITERS, lr=LEARNING_RATE, max_seq_len=MAX_SEQ_LENGTH,
        )

        first_50 = sum(losses[:50]) / 50
        last_50 = sum(losses[-50:]) / 50
        converged = last_50 < first_50 * 0.95

        pretrained_results[domain] = {
            "train_time_s": round(train_time, 1),
            "first_50_loss": round(first_50, 4),
            "last_50_loss": round(last_50, 4),
            "converged": converged,
            "individual_ppl": round(ppl, 2),
            "base_ppl": round(base_ppls_pretrained[domain], 2),
            "ppl_improvement_pct": round((base_ppls_pretrained[domain] - ppl) / base_ppls_pretrained[domain] * 100, 1),
        }
        print(f"    Loss: {first_50:.4f} -> {last_50:.4f} ({'converged' if converged else 'NOT converged'})")
        print(f"    PPL: {ppl:.2f} (base: {base_ppls_pretrained[domain]:.2f}, improvement: {pretrained_results[domain]['ppl_improvement_pct']}%)")

        pretrained_adapters[domain] = get_lora_params(model)

    results["condition_A_pretrained"] = pretrained_results

    # Composition on pretrained base
    print("\n  Computing composition on pretrained base...")
    adapter_list = [pretrained_adapters[d] for d in DOMAINS]
    composed = compose_adapters(adapter_list)
    zero_lora_params(model)
    apply_adapter_weights(model, composed)
    mx.eval(model.parameters())

    composed_ppls_pretrained = {}
    for domain in DOMAINS:
        ppl = compute_ppl(model, tokenizer, data_dirs[domain], max_seq_len=MAX_SEQ_LENGTH)
        composed_ppls_pretrained[domain] = ppl
        print(f"    {domain}: composed PPL = {ppl:.2f}")

    results["composed_ppls_pretrained"] = {d: round(v, 2) for d, v in composed_ppls_pretrained.items()}

    # Cosines
    pretrained_cosines = compute_adapter_cosines(pretrained_adapters)
    results["cosines_pretrained"] = pretrained_cosines
    mean_cos_pretrained = sum(pretrained_cosines.values()) / len(pretrained_cosines) if pretrained_cosines else 0
    print(f"  Mean |cos| (pretrained): {mean_cos_pretrained:.6f}")

    # ==================================================================
    # Phase 3: CONDITION B -- Random ternary scaffold + fresh LoRA (STE ternary)
    # ==================================================================
    print("\n" + "=" * 70)
    print("[Phase 3] CONDITION B: Random ternary scaffold + fresh LoRA (STE ternary)")
    print("=" * 70)

    # Remove LoRA to access base weights
    print("  Removing LoRA layers to replace base weights...")
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                updates.append((key, module.linear))
        if updates:
            layer.update_modules(tree_unflatten(updates))

    # Replace base weights with random ternary scaffold
    print("  Generating random ternary scaffold (norm-matched)...")
    model = replace_with_random_ternary_scaffold(model, weight_info, seed=SEED)

    # Compute scaffold base PPL
    print("\n  Computing scaffold base PPL (should be ~random)...")
    base_ppls_scaffold = {}
    for domain in DOMAINS:
        ppl = compute_ppl(model, tokenizer, data_dirs[domain], max_seq_len=MAX_SEQ_LENGTH)
        base_ppls_scaffold[domain] = ppl
        print(f"    {domain}: scaffold base PPL = {ppl:.2f}")
    results["base_ppls_scaffold"] = {d: round(v, 2) for d, v in base_ppls_scaffold.items()}

    # Apply LoRA to scaffold
    model = apply_ternary_lora(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Train adapters on scaffold
    scaffold_results = {}
    scaffold_adapters = {}

    for domain in DOMAINS:
        print(f"\n  --- Training {domain} adapter on SCAFFOLD ---")
        # Use SAME seed per domain as pretrained condition for fair comparison
        mx.random.seed(SEED + hash(domain) % 1000)
        zero_lora_params(model)

        losses, train_time, ppl = train_adapter(
            model, tokenizer, data_dirs[domain], domain,
            n_iters=TRAIN_ITERS, lr=LEARNING_RATE, max_seq_len=MAX_SEQ_LENGTH,
        )

        first_50 = sum(losses[:50]) / 50
        last_50 = sum(losses[-50:]) / 50
        converged = last_50 < first_50 * 0.95

        scaffold_results[domain] = {
            "train_time_s": round(train_time, 1),
            "first_50_loss": round(first_50, 4),
            "last_50_loss": round(last_50, 4),
            "converged": converged,
            "individual_ppl": round(ppl, 2),
            "scaffold_base_ppl": round(base_ppls_scaffold[domain], 2),
        }
        print(f"    Loss: {first_50:.4f} -> {last_50:.4f} ({'converged' if converged else 'NOT converged'})")
        print(f"    PPL: {ppl:.2f} (scaffold base: {base_ppls_scaffold[domain]:.2f})")

        scaffold_adapters[domain] = get_lora_params(model)

    results["condition_B_scaffold"] = scaffold_results

    # Composition on scaffold
    print("\n  Computing composition on scaffold...")
    adapter_list = [scaffold_adapters[d] for d in DOMAINS]
    composed = compose_adapters(adapter_list)
    zero_lora_params(model)
    apply_adapter_weights(model, composed)
    mx.eval(model.parameters())

    composed_ppls_scaffold = {}
    for domain in DOMAINS:
        ppl = compute_ppl(model, tokenizer, data_dirs[domain], max_seq_len=MAX_SEQ_LENGTH)
        composed_ppls_scaffold[domain] = ppl
        print(f"    {domain}: composed PPL = {ppl:.2f}")

    results["composed_ppls_scaffold"] = {d: round(v, 2) for d, v in composed_ppls_scaffold.items()}

    # Cosines
    scaffold_cosines = compute_adapter_cosines(scaffold_adapters)
    results["cosines_scaffold"] = scaffold_cosines
    mean_cos_scaffold = sum(scaffold_cosines.values()) / len(scaffold_cosines) if scaffold_cosines else 0
    print(f"  Mean |cos| (scaffold): {mean_cos_scaffold:.6f}")

    # ==================================================================
    # Phase 4: Kill criteria assessment
    # ==================================================================
    print("\n" + "=" * 70)
    print("[Phase 4] Kill Criteria Assessment")
    print("=" * 70)

    # K1: scaffold PPL > 5x pretrained PPL per domain
    print("\n  K1: Scaffold adapter PPL vs pretrained adapter PPL (threshold: 5x)")
    k1_results = {}
    k1_all_pass = True
    for domain in DOMAINS:
        scaffold_ppl = scaffold_results[domain]["individual_ppl"]
        pretrained_ppl = pretrained_results[domain]["individual_ppl"]
        ratio = scaffold_ppl / pretrained_ppl if pretrained_ppl > 0 else float("inf")
        passed = ratio <= 5.0
        if not passed:
            k1_all_pass = False
        k1_results[domain] = {
            "scaffold_ppl": scaffold_ppl,
            "pretrained_ppl": pretrained_ppl,
            "ratio": round(ratio, 2),
            "pass": passed,
        }
        status = "PASS" if passed else "KILL"
        print(f"    {domain}: scaffold={scaffold_ppl:.2f}, pretrained={pretrained_ppl:.2f}, ratio={ratio:.2f}x -> {status}")

    results["k1_per_domain"] = k1_results
    results["k1_pass"] = k1_all_pass
    print(f"\n  K1 overall: {'PASS' if k1_all_pass else 'KILL'}")

    # K2: convergence on scaffold
    print("\n  K2: Convergence on scaffold (loss must decrease over training)")
    k2_results = {}
    k2_all_pass = True
    for domain in DOMAINS:
        converged = scaffold_results[domain]["converged"]
        first_loss = scaffold_results[domain]["first_50_loss"]
        last_loss = scaffold_results[domain]["last_50_loss"]
        if not converged:
            k2_all_pass = False
        k2_results[domain] = {
            "first_50_loss": first_loss,
            "last_50_loss": last_loss,
            "converged": converged,
        }
        status = "PASS" if converged else "KILL"
        print(f"    {domain}: {first_loss:.4f} -> {last_loss:.4f} -> {status}")

    results["k2_convergence"] = k2_results
    results["k2_pass"] = k2_all_pass
    print(f"\n  K2 overall: {'PASS' if k2_all_pass else 'KILL'}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # PPL comparison table
    print(f"\n  {'Domain':<12} {'Pre Base':>10} {'Pre Adapt':>10} {'Scaff Base':>10} {'Scaff Adapt':>11} {'Ratio':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*11} {'-'*8}")
    for domain in DOMAINS:
        pb = base_ppls_pretrained[domain]
        pa = pretrained_results[domain]["individual_ppl"]
        sb = base_ppls_scaffold[domain]
        sa = scaffold_results[domain]["individual_ppl"]
        ratio = sa / pa if pa > 0 else float("inf")
        print(f"  {domain:<12} {pb:>10.2f} {pa:>10.2f} {sb:>10.2f} {sa:>11.2f} {ratio:>7.2f}x")

    # Composition comparison
    print(f"\n  Composition PPL (1/N scaling):")
    print(f"  {'Domain':<12} {'Pre Comp':>10} {'Scaff Comp':>10} {'Ratio':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*8}")
    for domain in DOMAINS:
        pc = composed_ppls_pretrained[domain]
        sc = composed_ppls_scaffold[domain]
        ratio = sc / pc if pc > 0 else float("inf")
        print(f"  {domain:<12} {pc:>10.2f} {sc:>10.2f} {ratio:>7.2f}x")

    # Cosine comparison
    print(f"\n  Mean |cos|: pretrained={mean_cos_pretrained:.6f}, scaffold={mean_cos_scaffold:.6f}")

    # Overall verdict
    overall = k1_all_pass and k2_all_pass
    verdict = "SUPPORTED" if overall else "KILLED"
    print(f"\n  VERDICT: {verdict}")
    print(f"    K1 (scaffold PPL <= 5x pretrained PPL): {'PASS' if k1_all_pass else 'KILL'}")
    print(f"    K2 (convergence on scaffold): {'PASS' if k2_all_pass else 'KILL'}")

    results["verdict"] = verdict
    results["mean_cos_pretrained"] = round(mean_cos_pretrained, 6)
    results["mean_cos_scaffold"] = round(mean_cos_scaffold, 6)

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
