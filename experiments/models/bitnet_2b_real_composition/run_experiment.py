#!/usr/bin/env python3
"""
BitNet-2B-4T Real Composition Experiment

Tests whether the actual Microsoft BitNet-b1.58-2B-4T (2.4B params, d=2560,
natively ternary) supports LoRA adapter composition on Apple Silicon.

Kill criteria:
  K1: BitNet-2B-4T fails to load or run on Apple Silicon (tooling blocker)
  K2: LoRA training on BitNet-2B-4T fails to converge (5 domains, rank-16)
  K3: composed 5-adapter PPL > 10x single best adapter PPL (composition catastrophe)
  K4: individual adapters do not beat base on >60% of domains (distillation failed)

Architecture:
  - Base: microsoft/BitNet-b1.58-2B-4T (ternary weights, d=2560, 30 layers)
  - LoRA: rank-16 on all attention + MLP projections (q/k/v/o/gate/up/down)
  - Training: 200 iterations per adapter, batch_size=1, max_seq_length=512
  - Composition: naive addition W + sum(B_i @ A_i) with 1/N scaling

Key challenge: BitLinear uses a custom Metal kernel without vjp support.
Solution: unpack ternary weights to bfloat16 standard nn.Linear for training,
which supports automatic differentiation. Weights are frozen; only LoRA trains.
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
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.bitlinear_layers import BitLinear


# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0  # mlx_lm default; higher scale = larger adapter signal
TRAIN_ITERS = 200
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
VAL_BATCHES = 25

EXPERIMENT_DIR = Path(__file__).parent
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# 5 domains with HuggingFace datasets
DOMAINS = {
    "python": {
        "hf_dataset": "iamtarun/python_code_instructions_18k_alpaca",
        "text_key": "output",
        "max_samples_train": 500,
        "max_samples_val": 50,
    },
    "math": {
        "hf_dataset": "gsm8k",
        "hf_subset": "main",
        "text_key": "answer",
        "max_samples_train": 500,
        "max_samples_val": 50,
    },
    "medical": {
        "hf_dataset": "medalpaca/medical_meadow_medical_flashcards",
        "text_key": "output",
        "max_samples_train": 500,
        "max_samples_val": 50,
    },
    "legal": {
        "hf_dataset": "jonathanli/law-stack-exchange",
        "text_key": "body",
        "max_samples_train": 500,
        "max_samples_val": 50,
    },
    "creative": {
        "hf_dataset": "roneneldan/TinyStories",
        "text_key": "text",
        "max_samples_train": 500,
        "max_samples_val": 50,
    },
}


# ===========================================================================
# Unpack ternary weights for differentiable forward pass
# ===========================================================================
def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    """Unpack uint8-packed ternary weights to bfloat16 dense matrix.

    Packing: each uint8 byte stores 4 ternary values {-1, 0, 1} encoded as
    {0, 1, 2} in 2-bit groups. The output dimension is split into 4 blocks
    of size out_features/4, one per bit-pair.

    Returns: (out_features, in_features) bfloat16 tensor, scaled.
    """
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
    """Replace all BitLinear layers with standard nn.Linear using unpacked weights.

    This enables automatic differentiation (vjp) through the model.
    Memory cost: ~3.9GB (bfloat16) vs ~490MB (packed ternary).
    """
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                # Unpack weights
                unpacked_w = unpack_ternary(
                    module.weight,
                    module.out_features,
                    module.weight_scale,
                    module.invert_weight_scales,
                )

                # Create standard nn.Linear
                has_bias = module.bias is not None
                linear = nn.Linear(
                    module.in_features, module.out_features, bias=has_bias
                )
                linear.weight = unpacked_w
                if has_bias:
                    linear.bias = module.bias

                updates.append((key, linear))
                count += 1

        if updates:
            layer.update_modules(tree_unflatten(updates))

    mx.eval(model.parameters())
    print(f"  Replaced {count} BitLinear -> nn.Linear (unpacked to bfloat16)")
    return model


# ===========================================================================
# LoRA application
# ===========================================================================
def apply_lora_to_model(model, rank=16, scale=1.0):
    """Apply LoRA wrappers to all linear layers in transformer blocks."""
    target_keys = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }

    count = 0
    for layer in model.model.layers:
        lora_updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora_layer = LoRALinear.from_base(
                    module, r=rank, scale=scale, dropout=0.0
                )
                lora_updates.append((key, lora_layer))
                count += 1

        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    print(f"  Applied LoRA (r={rank}) to {count} linear layers")
    return model


def remove_lora(model):
    """Remove LoRA wrappers, restoring plain nn.Linear layers."""
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                updates.append((key, module.linear))
        if updates:
            layer.update_modules(tree_unflatten(updates))
    return model


def get_lora_params(model):
    """Extract LoRA parameters as a flat dict (deep copy to avoid aliasing)."""
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_a" in name or "lora_b" in name:
            # Deep copy: create new array to avoid reference aliasing
            params[name] = mx.array(p)
    mx.eval(params)
    return params


def save_adapter(model, path: Path):
    """Save LoRA adapter weights."""
    path.mkdir(parents=True, exist_ok=True)
    params = get_lora_params(model)
    mx.savez(str(path / "adapter.npz"), **params)
    print(f"  Saved adapter: {len(params)} tensors to {path}")


def load_adapter(path: Path) -> dict:
    """Load adapter weights from disk."""
    data = dict(mx.load(str(path / "adapter.npz")))
    return data


def apply_adapter_weights(model, adapter_params: dict, scale: float = 1.0):
    """Apply adapter params into current LoRA layers."""
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def zero_lora_params(model):
    """Reset all LoRA params to zero (re-init lora_a random, lora_b zero)."""
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                in_dims = module.lora_a.shape[0]
                r = module.lora_a.shape[1]
                scale = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(
                    low=-scale, high=scale, shape=module.lora_a.shape
                )
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


# ===========================================================================
# Data preparation
# ===========================================================================
def prepare_domain_data(domain_name: str, domain_config: dict) -> Path:
    """Download HF dataset and write train.jsonl / valid.jsonl."""
    from datasets import load_dataset as hf_load

    data_dir = EXPERIMENT_DIR / "data" / domain_name
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"

    if train_path.exists() and valid_path.exists():
        print(f"  Data for {domain_name} already exists, skipping download")
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {domain_config['hf_dataset']}...")

    kwargs = {}
    if "hf_subset" in domain_config:
        kwargs["name"] = domain_config["hf_subset"]

    ds = hf_load(domain_config["hf_dataset"], **kwargs)

    if "train" in ds:
        split_data = ds["train"]
    else:
        split_name = list(ds.keys())[0]
        split_data = ds[split_name]

    text_key = domain_config["text_key"]
    if text_key not in split_data.column_names:
        for alt in ["text", "content", "output", "answer", "response", "question"]:
            if alt in split_data.column_names:
                text_key = alt
                print(f"  Using '{text_key}' instead of '{domain_config['text_key']}'")
                break

    max_train = domain_config["max_samples_train"]
    max_val = domain_config["max_samples_val"]

    texts = []
    for row in split_data:
        t = row[text_key]
        if isinstance(t, str) and len(t.strip()) > 20:
            texts.append(t.strip())
        if len(texts) >= max_train + max_val:
            break

    if len(texts) < max_val + 10:
        raise ValueError(f"Not enough samples for {domain_name}: got {len(texts)}")

    train_texts = texts[:max_train]
    val_texts = texts[max_train : max_train + max_val]

    with open(train_path, "w") as f:
        for t in train_texts:
            json.dump({"text": t}, f)
            f.write("\n")

    with open(valid_path, "w") as f:
        for t in val_texts:
            json.dump({"text": t}, f)
            f.write("\n")

    print(f"  {domain_name}: {len(train_texts)} train, {len(val_texts)} val")
    return data_dir


# ===========================================================================
# PPL evaluation
# ===========================================================================
def compute_ppl(model, tokenizer, data_path: Path, max_batches: int = 25):
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
        tokens = tokens[: MAX_SEQ_LENGTH + 1]

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
# Composition
# ===========================================================================
def compose_adapters(adapter_list: list, scale_per_adapter: float = None):
    """Merge multiple adapter parameter dicts with given scale (default 1/N)."""
    N = len(adapter_list)
    if scale_per_adapter is None:
        scale_per_adapter = 1.0 / N

    merged = {}
    for key in adapter_list[0].keys():
        stacked = mx.stack([a[key] for a in adapter_list])
        merged[key] = mx.sum(stacked, axis=0) * scale_per_adapter

    return merged


# ===========================================================================
# Main
# ===========================================================================
def main():
    results = {
        "experiment": "bitnet_2b_real_composition",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "train_iters": TRAIN_ITERS,
        "domains": list(DOMAINS.keys()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=" * 70)
    print("BitNet-2B-4T Real Composition Experiment")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 0: Load model and unpack for training
    # ------------------------------------------------------------------
    print("\n[Phase 0] Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    load_time = time.time() - t0
    print(f"  Loaded (packed) in {load_time:.1f}s")

    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"  Total parameters (packed): {total_params:,}")

    print("\n  Unpacking ternary weights for differentiable training...")
    t1 = time.time()
    model = replace_bitlinear_with_linear(model)
    unpack_time = time.time() - t1
    print(f"  Unpacked in {unpack_time:.1f}s")

    total_params_unpacked = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"  Total parameters (unpacked): {total_params_unpacked:,}")
    results["total_params"] = total_params_unpacked
    results["load_time_s"] = load_time + unpack_time

    # ------------------------------------------------------------------
    # Phase 1: Prepare data
    # ------------------------------------------------------------------
    print("\n[Phase 1] Preparing domain data...")
    data_dirs = {}
    for domain_name, config in DOMAINS.items():
        try:
            data_dirs[domain_name] = prepare_domain_data(domain_name, config)
        except Exception as e:
            print(f"  FATAL: {domain_name}: {e}")
            results["error"] = str(e)
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)
            return

    # ------------------------------------------------------------------
    # Phase 2: Base model PPL
    # ------------------------------------------------------------------
    print("\n[Phase 2] Computing base model PPL...")
    base_ppls = {}
    for domain_name, data_dir in data_dirs.items():
        ppl = compute_ppl(model, tokenizer, data_dir)
        base_ppls[domain_name] = ppl
        print(f"  {domain_name}: base PPL = {ppl:.2f}")
    results["base_ppls"] = base_ppls

    # ------------------------------------------------------------------
    # Phase 3: Train domain adapters
    # ------------------------------------------------------------------
    print("\n[Phase 3] Training domain adapters...")

    # Apply LoRA once (will reset weights for each domain)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Freeze base, unfreeze LoRA
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"  Trainable LoRA parameters: {trainable:,}")
    results["trainable_params"] = trainable

    # Verify gradients work
    print("  Verifying gradient computation...")
    def test_loss(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")
    test_grad = nn.value_and_grad(model, test_loss)
    x_test = mx.array([[1, 2, 3, 4, 5]])
    y_test = mx.array([[2, 3, 4, 5, 6]])
    try:
        l, g = test_grad(model, x_test, y_test)
        mx.eval(l)
        print(f"  Gradient check PASSED (loss={l.item():.4f})")
    except Exception as e:
        print(f"  Gradient check FAILED: {e}")
        results["error"] = f"Gradient computation failed: {e}"
        results["k1_pass"] = False
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        return

    adapter_params_all = {}
    train_results = {}

    for domain_name, data_dir in data_dirs.items():
        print(f"\n  --- Training {domain_name} adapter ---")

        # Reset LoRA params to zero
        zero_lora_params(model)

        # Tokenize data
        train_texts = []
        with open(data_dir / "train.jsonl") as f:
            for line in f:
                train_texts.append(json.loads(line)["text"])

        train_tokens = []
        for text in train_texts:
            toks = tokenizer.encode(text)
            if len(toks) > 2:
                train_tokens.append(mx.array(toks[: MAX_SEQ_LENGTH + 1]))

        print(f"  {len(train_tokens)} training sequences")

        # Training loop
        optimizer = opt.Adam(learning_rate=LEARNING_RATE)

        def loss_fn(model, x, y):
            logits = model(x)
            return nn.losses.cross_entropy(logits, y, reduction="mean")

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        t_start = time.time()
        losses = []
        for step in range(TRAIN_ITERS):
            idx = step % len(train_tokens)
            tokens = train_tokens[idx]
            x = tokens[:-1][None, :]
            y = tokens[1:][None, :]

            loss, grads = loss_and_grad(model, x, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            loss_val = loss.item()
            losses.append(loss_val)

            if (step + 1) % 50 == 0 or step == 0:
                avg = sum(losses[-50:]) / len(losses[-50:])
                print(f"    Step {step+1}/{TRAIN_ITERS}: loss={loss_val:.4f} (avg50={avg:.4f})")

        train_time = time.time() - t_start
        first_50 = sum(losses[:50]) / 50
        last_50 = sum(losses[-50:]) / 50
        converged = last_50 < first_50 * 0.95

        print(f"  Done in {train_time:.1f}s. Loss: {first_50:.4f} -> {last_50:.4f} "
              f"({'converged' if converged else 'NOT converged'})")

        train_results[domain_name] = {
            "train_time_s": round(train_time, 1),
            "first_50_avg_loss": round(first_50, 4),
            "last_50_avg_loss": round(last_50, 4),
            "converged": converged,
        }

        # Verify adapter has non-zero weights before saving
        lora_norm = sum(
            mx.sum(p**2).item()
            for _, p in tree_flatten(model.trainable_parameters())
            if True  # all trainable are LoRA
        )
        print(f"  Post-training LoRA L2 norm: {lora_norm:.6f}")

        # Save adapter
        save_adapter(model, ADAPTERS_DIR / domain_name)
        adapter_params_all[domain_name] = get_lora_params(model)

    results["train_results"] = train_results
    n_converged = sum(1 for r in train_results.values() if r["converged"])
    results["k2_pass"] = n_converged >= 3
    print(f"\n  K2: {n_converged}/5 converged -> {'PASS' if results['k2_pass'] else 'FAIL'}")

    # ------------------------------------------------------------------
    # Phase 4: Individual adapter PPL (load from disk to avoid aliasing)
    # ------------------------------------------------------------------
    print("\n[Phase 4] Individual adapter PPL...")
    individual_ppls = {}
    loaded_adapters = {}

    for domain_name in DOMAINS:
        adapter_path = ADAPTERS_DIR / domain_name
        params = load_adapter(adapter_path)
        loaded_adapters[domain_name] = params

        zero_lora_params(model)
        apply_adapter_weights(model, params, scale=1.0)
        mx.eval(model.parameters())

        # Verify adapter is non-zero
        norm = sum(mx.sum(p**2).item() for p in params.values())
        print(f"  {domain_name}: adapter L2 norm = {norm:.4f}")

        ppl = compute_ppl(model, tokenizer, data_dirs[domain_name])
        individual_ppls[domain_name] = ppl
        base = base_ppls[domain_name]
        imp = (base - ppl) / base * 100
        print(f"  {domain_name}: PPL={ppl:.2f} (base={base:.2f}, {imp:+.1f}%)")

    results["individual_ppls"] = individual_ppls

    domains_improved = sum(1 for d in DOMAINS if individual_ppls[d] < base_ppls[d])
    results["k4_domains_improved"] = domains_improved
    results["k4_pass"] = domains_improved >= 3
    print(f"\n  K4: {domains_improved}/5 improved -> {'PASS' if results['k4_pass'] else 'FAIL'}")

    # ------------------------------------------------------------------
    # Phase 5: Composed PPL (1/N scaling)
    # ------------------------------------------------------------------
    print("\n[Phase 5] Composed 5 adapters (1/N scaling)...")
    adapter_list = list(loaded_adapters.values())
    merged = compose_adapters(adapter_list)

    zero_lora_params(model)
    apply_adapter_weights(model, merged)
    mx.eval(model.parameters())

    composed_ppls = {}
    for domain_name, data_dir in data_dirs.items():
        ppl = compute_ppl(model, tokenizer, data_dir)
        composed_ppls[domain_name] = ppl
        print(f"  {domain_name}: composed PPL = {ppl:.2f}")

    results["composed_ppls"] = composed_ppls

    best_ind = min(individual_ppls.values())
    avg_composed = sum(composed_ppls.values()) / len(composed_ppls)
    avg_individual = sum(individual_ppls.values()) / len(individual_ppls)
    composition_ratio = avg_composed / best_ind

    results["best_individual_ppl"] = best_ind
    results["avg_composed_ppl"] = round(avg_composed, 4)
    results["avg_individual_ppl"] = round(avg_individual, 4)
    results["composition_ratio"] = round(composition_ratio, 4)
    results["k3_pass"] = composition_ratio < 10.0
    print(f"\n  K3: ratio={composition_ratio:.2f}x -> {'PASS' if results['k3_pass'] else 'FAIL'}")

    # ------------------------------------------------------------------
    # Phase 6: Orthogonality
    # ------------------------------------------------------------------
    print("\n[Phase 6] Adapter orthogonality...")
    cosines = []
    names = list(loaded_adapters.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            vi = mx.concatenate([v.reshape(-1) for v in loaded_adapters[names[i]].values()])
            vj = mx.concatenate([v.reshape(-1) for v in loaded_adapters[names[j]].values()])
            cos = mx.abs(mx.sum(vi * vj) / (mx.sqrt(mx.sum(vi**2)) * mx.sqrt(mx.sum(vj**2))))
            mx.eval(cos)
            cosines.append({"pair": f"{names[i]}-{names[j]}", "abs_cos": round(cos.item(), 4)})

    mean_cos = sum(c["abs_cos"] for c in cosines) / len(cosines)
    results["cosine_similarities"] = cosines
    results["mean_abs_cos"] = round(mean_cos, 4)
    print(f"  Mean |cos|: {mean_cos:.4f}")
    for c in cosines:
        print(f"    {c['pair']}: {c['abs_cos']:.4f}")

    # ------------------------------------------------------------------
    # Phase 7: Unit-weight composition (no 1/N)
    # ------------------------------------------------------------------
    print("\n[Phase 7] Unit-weight composition (no scaling)...")
    merged_unit = compose_adapters(adapter_list, scale_per_adapter=1.0)

    zero_lora_params(model)
    apply_adapter_weights(model, merged_unit)
    mx.eval(model.parameters())

    unit_ppls = {}
    for domain_name, data_dir in data_dirs.items():
        ppl = compute_ppl(model, tokenizer, data_dir)
        unit_ppls[domain_name] = ppl
        print(f"  {domain_name}: unit PPL = {ppl:.2f}")

    results["unit_weight_ppls"] = unit_ppls
    avg_unit = sum(unit_ppls.values()) / len(unit_ppls)
    results["avg_unit_ppl"] = round(avg_unit, 4)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    avg_base = sum(base_ppls.values()) / len(base_ppls)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  K1 (loads on Apple Silicon): PASS")
    print(f"  K2 (training converges): {'PASS' if results['k2_pass'] else 'FAIL'} ({n_converged}/5)")
    print(f"  K3 (composition ratio <10x): {'PASS' if results['k3_pass'] else 'FAIL'} ({composition_ratio:.2f}x)")
    print(f"  K4 (>60% domains improved): {'PASS' if results['k4_pass'] else 'FAIL'} ({domains_improved}/5)")
    print(f"\n  Avg base PPL:       {avg_base:.2f}")
    print(f"  Avg individual PPL: {avg_individual:.2f}")
    print(f"  Avg composed (1/N): {avg_composed:.2f}")
    print(f"  Avg unit-weight:    {avg_unit:.2f}")
    print(f"  Mean |cos|:         {mean_cos:.4f}")

    all_pass = results["k2_pass"] and results["k3_pass"] and results["k4_pass"]
    results["verdict"] = "SUPPORTED" if all_pass else "KILLED"
    print(f"\n  VERDICT: {results['verdict']}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
