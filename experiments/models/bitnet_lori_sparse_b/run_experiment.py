#!/usr/bin/env python3
"""
LoRI-style B-sparsity Experiment on BitNet-2B-4T

Tests whether 90% sparsity in LoRA B matrices reduces composition interference
on BitNet-2B-4T while preserving individual adapter quality.

Protocol (following LoRI, arXiv 2504.07448, COLM 2025):
  1. Train 5 domain adapters with dense B (400 steps) -- BASELINE
  2. Train 5 domain adapters with LoRI protocol:
     a. Calibration phase: dense training (200 steps) to learn magnitude pattern
     b. Extract GLOBAL mask: keep top-10% of B elements by magnitude across ALL
        layers (model-wise threshold, not per-layer)
     c. Reset B to zero (key LoRI insight: discard calibration weights)
     d. Retrain from scratch with frozen mask (400 steps, same budget as dense)
  3. Compare individual PPL and composed PPL between dense and sparse regimes
  4. Measure cosine similarity between adapters in both regimes

Kill criteria:
  K1: sparse-B individual adapter PPL > 1.10x dense-B (too much quality loss)
  K2: sparse-B composed PPL worse than dense-B composed (sparsity hurts composition)

Architecture:
  - Base: microsoft/BitNet-b1.58-2B-4T (ternary weights, d=2560, 30 layers)
  - LoRA: rank-16 on all attention + MLP projections
  - Training: 400 steps for dense, 200 calibration + 400 masked for sparse
  - Composition: 1/N scaling
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from functools import partial

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

# Force unbuffered output
print = partial(print, flush=True)

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
LORA_SCALE = 20.0
DENSE_TRAIN_ITERS = 400       # Total steps for dense baseline
CALIBRATION_ITERS = 200       # Dense calibration to learn magnitude pattern
SPARSE_TRAIN_ITERS = 400      # Full retrain from zero with mask (same budget as dense)
SPARSITY = 0.90               # Keep top 10% of B elements
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
VAL_BATCHES = 25

EXPERIMENT_DIR = Path(__file__).parent
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# 5 domains -- same as prior experiments for comparability
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
    """Unpack uint8-packed ternary weights to bfloat16 dense matrix."""
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
    """Replace all BitLinear layers with standard nn.Linear using unpacked weights."""
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                unpacked_w = unpack_ternary(
                    module.weight,
                    module.out_features,
                    module.weight_scale,
                    module.invert_weight_scales,
                )
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
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
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
    """Extract LoRA parameters as a flat dict (deep copy)."""
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_a" in name or "lora_b" in name:
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
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_weights(model, adapter_params: dict, scale: float = 1.0):
    """Apply adapter params into current LoRA layers."""
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def zero_lora_params(model):
    """Reset all LoRA params (re-init lora_a random, lora_b zero)."""
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
# Sparsity: LoRI-style magnitude pruning of B matrices
# ===========================================================================
def create_b_sparsity_masks(model, sparsity=0.90):
    """Create binary masks for lora_b using GLOBAL threshold (LoRI protocol).

    Following LoRI (arXiv 2504.07448): compute a single magnitude threshold
    across ALL B matrices in the model (model-wise masking), not per-layer.
    This outperforms per-layer masking per the paper's ablation.

    Returns dict of masks (same keys as lora_b params).
    Each mask is 1 for kept elements, 0 for pruned.
    """
    # Step 1: Collect all B magnitudes into a single vector for global threshold
    b_params = {}
    all_magnitudes = []
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_b" in name:
            b_params[name] = p
            all_magnitudes.append(mx.abs(p).reshape(-1))

    all_mags = mx.concatenate(all_magnitudes)
    mx.eval(all_mags)

    n_total = all_mags.size
    n_keep = max(1, int(n_total * (1.0 - sparsity)))

    # Step 2: Find global threshold (k-th largest magnitude)
    sorted_vals = mx.sort(all_mags)
    threshold = sorted_vals[-(n_keep)]
    mx.eval(threshold)
    print(f"  Global magnitude threshold: {threshold.item():.6f}")

    # Step 3: Apply global threshold to each B matrix
    masks = {}
    for name, p in b_params.items():
        mask = (mx.abs(p) >= threshold).astype(mx.bfloat16)
        mx.eval(mask)
        masks[name] = mask

    print(f"  Created {len(masks)} B-sparsity masks at {sparsity*100:.0f}% sparsity (global threshold)")

    # Report aggregate stats
    total_params = sum(m.size for m in masks.values())
    total_kept = sum(mx.sum(m).item() for m in masks.values())
    actual_sparsity = 1.0 - total_kept / total_params
    print(f"  Total B params: {total_params:,}, kept: {int(total_kept):,} ({total_kept/total_params*100:.1f}%), actual sparsity: {actual_sparsity*100:.1f}%")

    return masks


def apply_b_masks(model, masks):
    """Zero out pruned elements in lora_b using masks. Call after each optimizer step."""
    updates = []
    for name, p in tree_flatten(model.trainable_parameters()):
        if name in masks:
            masked_p = p * masks[name]
            updates.append((name, masked_p))
    if updates:
        model.update(tree_unflatten(updates))


# ===========================================================================
# Data preparation
# ===========================================================================
def prepare_domain_data(domain_name: str, domain_config: dict, data_root: Path) -> Path:
    """Download HF dataset and write train.jsonl / valid.jsonl."""
    from datasets import load_dataset as hf_load

    data_dir = data_root / domain_name
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
# Cosine similarity between adapter parameter vectors
# ===========================================================================
def compute_adapter_cosines(adapters_dict):
    """Compute pairwise cosine similarities between all adapter pairs."""
    names = list(adapters_dict.keys())
    cosines = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            vi = mx.concatenate([v.reshape(-1) for v in adapters_dict[names[i]].values()])
            vj = mx.concatenate([v.reshape(-1) for v in adapters_dict[names[j]].values()])
            cos = mx.abs(mx.sum(vi * vj) / (mx.sqrt(mx.sum(vi**2)) * mx.sqrt(mx.sum(vj**2))))
            mx.eval(cos)
            cosines.append({
                "pair": f"{names[i]}-{names[j]}",
                "abs_cos": round(cos.item(), 6),
            })
    mean_cos = sum(c["abs_cos"] for c in cosines) / len(cosines) if cosines else 0
    return cosines, mean_cos


# ===========================================================================
# Training loop
# ===========================================================================
def train_adapter(model, tokenizer, data_dir, domain_name, n_iters,
                  masks=None, start_step_label=0):
    """Train one adapter for n_iters steps. If masks provided, apply after each step."""
    train_texts = []
    with open(data_dir / "train.jsonl") as f:
        for line in f:
            train_texts.append(json.loads(line)["text"])

    train_tokens = []
    for text in train_texts:
        toks = tokenizer.encode(text)
        if len(toks) > 2:
            train_tokens.append(mx.array(toks[: MAX_SEQ_LENGTH + 1]))

    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    t_start = time.time()
    losses = []
    for step in range(n_iters):
        idx = (start_step_label + step) % len(train_tokens)
        tokens = train_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        # Apply sparsity masks after gradient step
        if masks is not None:
            apply_b_masks(model, masks)
            mx.eval(model.parameters())

        loss_val = loss.item()
        losses.append(loss_val)

        total_step = start_step_label + step + 1
        if total_step % 100 == 0 or step == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            print(f"    Step {total_step}: loss={loss_val:.4f} (avg50={avg:.4f})")

    train_time = time.time() - t_start
    first_50 = sum(losses[:min(50, len(losses))]) / min(50, len(losses))
    last_50 = sum(losses[-50:]) / len(losses[-50:])
    converged = last_50 < first_50 * 0.95

    return {
        "train_time_s": round(train_time, 1),
        "first_50_avg_loss": round(first_50, 4),
        "last_50_avg_loss": round(last_50, 4),
        "converged": converged,
        "n_iters": n_iters,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    results = {
        "experiment": "bitnet_lori_sparse_b",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "sparsity": SPARSITY,
        "dense_iters": DENSE_TRAIN_ITERS,
        "calibration_iters": CALIBRATION_ITERS,
        "sparse_train_iters": SPARSE_TRAIN_ITERS,
        "domains": list(DOMAINS.keys()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=" * 70)
    print("LoRI-style B-sparsity Experiment on BitNet-2B-4T")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 0: Load model and unpack
    # ------------------------------------------------------------------
    print("\n[Phase 0] Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    load_time = time.time() - t0
    print(f"  Loaded (packed) in {load_time:.1f}s")

    print("  Unpacking ternary weights for differentiable training...")
    t1 = time.time()
    model = replace_bitlinear_with_linear(model)
    unpack_time = time.time() - t1
    print(f"  Unpacked in {unpack_time:.1f}s")

    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"  Total parameters (unpacked): {total_params:,}")
    results["total_params"] = total_params
    results["load_time_s"] = round(load_time + unpack_time, 1)

    # ------------------------------------------------------------------
    # Phase 1: Prepare data
    # ------------------------------------------------------------------
    print("\n[Phase 1] Preparing domain data...")
    data_dir_root = EXPERIMENT_DIR / "data"
    data_dirs = {}
    for domain_name, config in DOMAINS.items():
        try:
            data_dirs[domain_name] = prepare_domain_data(domain_name, config, data_dir_root)
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
    # Phase 3: Train DENSE adapters (400 steps, baseline) or reload from disk
    # ------------------------------------------------------------------
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"  Trainable LoRA parameters: {trainable:,}")
    results["trainable_params"] = trainable

    dense_adapters = {}
    dense_train_results = {}

    # Check if all dense adapters already exist on disk
    all_dense_exist = all(
        (ADAPTERS_DIR / "dense" / d / "adapter.npz").exists() for d in DOMAINS
    )

    if all_dense_exist:
        print("\n[Phase 3] Loading existing DENSE-B adapters from disk...")
        for domain_name in DOMAINS:
            adapter_path = ADAPTERS_DIR / "dense" / domain_name
            dense_adapters[domain_name] = load_adapter(adapter_path)
            print(f"  Loaded {domain_name} adapter from {adapter_path}")
            dense_train_results[domain_name] = {"reloaded": True}
    else:
        print("\n[Phase 3] Training DENSE-B adapters (400 steps each)...")
        for domain_name, data_dir in data_dirs.items():
            print(f"\n  --- Training DENSE {domain_name} adapter ---")
            zero_lora_params(model)

            result = train_adapter(model, tokenizer, data_dir, domain_name, DENSE_TRAIN_ITERS)
            dense_train_results[domain_name] = result
            print(f"  Done in {result['train_time_s']}s. Loss: {result['first_50_avg_loss']:.4f} -> {result['last_50_avg_loss']:.4f}")

            save_adapter(model, ADAPTERS_DIR / "dense" / domain_name)
            dense_adapters[domain_name] = get_lora_params(model)

    results["dense_train_results"] = dense_train_results

    # ------------------------------------------------------------------
    # Phase 4: Train SPARSE adapters (LoRI protocol: calibrate + mask + reset + retrain)
    # Following arXiv 2504.07448: calibrate dense -> extract global mask -> reset B to 0 -> retrain with mask
    # ------------------------------------------------------------------
    print("\n[Phase 4] Training SPARSE-B adapters (LoRI protocol)...")

    sparse_adapters = {}
    sparse_train_results = {}

    for domain_name, data_dir in data_dirs.items():
        print(f"\n  --- Training SPARSE {domain_name} adapter ---")
        zero_lora_params(model)

        # Step 1: Calibration phase -- dense training to learn magnitude pattern
        print(f"  Step 1: Calibration ({CALIBRATION_ITERS} steps, dense)...")
        cal_result = train_adapter(model, tokenizer, data_dir, domain_name, CALIBRATION_ITERS)
        print(f"  Calibration done. Loss: {cal_result['first_50_avg_loss']:.4f} -> {cal_result['last_50_avg_loss']:.4f}")

        # Step 2: Extract global sparsity mask from calibrated B magnitudes
        print(f"  Step 2: Extracting global mask (keep top {(1-SPARSITY)*100:.0f}%)...")
        masks = create_b_sparsity_masks(model, sparsity=SPARSITY)

        # Step 3: Reset B to zero (key LoRI insight: discard calibration weights)
        print(f"  Step 3: Resetting B to zero (LoRI protocol)...")
        zero_lora_params(model)

        # Step 4: Retrain from scratch with frozen mask (same budget as dense baseline)
        print(f"  Step 4: Training with frozen mask ({SPARSE_TRAIN_ITERS} steps)...")
        train_result = train_adapter(
            model, tokenizer, data_dir, domain_name, SPARSE_TRAIN_ITERS,
            masks=masks, start_step_label=0
        )
        print(f"  Training done. Loss: {train_result['first_50_avg_loss']:.4f} -> {train_result['last_50_avg_loss']:.4f}")

        # Save
        save_adapter(model, ADAPTERS_DIR / "sparse" / domain_name)
        sparse_adapters[domain_name] = get_lora_params(model)

        sparse_train_results[domain_name] = {
            "calibration": cal_result,
            "training": train_result,
            "total_train_time_s": round(cal_result["train_time_s"] + train_result["train_time_s"], 1),
        }

    results["sparse_train_results"] = sparse_train_results

    # ------------------------------------------------------------------
    # Phase 5: Individual PPL comparison (dense vs sparse)
    # ------------------------------------------------------------------
    print("\n[Phase 5] Individual adapter PPL comparison...")

    dense_individual_ppls = {}
    sparse_individual_ppls = {}

    print("\n  Dense individual PPL:")
    for domain_name in DOMAINS:
        zero_lora_params(model)
        apply_adapter_weights(model, dense_adapters[domain_name], scale=1.0)
        mx.eval(model.parameters())
        ppl = compute_ppl(model, tokenizer, data_dirs[domain_name])
        dense_individual_ppls[domain_name] = ppl
        base = base_ppls[domain_name]
        imp = (base - ppl) / base * 100
        print(f"    {domain_name}: PPL={ppl:.2f} (base={base:.2f}, {imp:+.1f}%)")

    print("\n  Sparse individual PPL:")
    for domain_name in DOMAINS:
        zero_lora_params(model)
        apply_adapter_weights(model, sparse_adapters[domain_name], scale=1.0)
        mx.eval(model.parameters())
        ppl = compute_ppl(model, tokenizer, data_dirs[domain_name])
        sparse_individual_ppls[domain_name] = ppl
        dense_ppl = dense_individual_ppls[domain_name]
        ratio = ppl / dense_ppl
        print(f"    {domain_name}: PPL={ppl:.2f} (dense={dense_ppl:.2f}, ratio={ratio:.3f}x)")

    results["dense_individual_ppls"] = dense_individual_ppls
    results["sparse_individual_ppls"] = sparse_individual_ppls

    # K1: sparse individual PPL must be <= 1.10x dense
    k1_ratios = {}
    for domain_name in DOMAINS:
        ratio = sparse_individual_ppls[domain_name] / dense_individual_ppls[domain_name]
        k1_ratios[domain_name] = round(ratio, 4)
    results["k1_individual_ratios"] = k1_ratios
    max_ratio = max(k1_ratios.values())
    mean_ratio = sum(k1_ratios.values()) / len(k1_ratios)
    results["k1_max_ratio"] = round(max_ratio, 4)
    results["k1_mean_ratio"] = round(mean_ratio, 4)
    results["k1_pass"] = max_ratio <= 1.10
    print(f"\n  K1: max sparse/dense ratio = {max_ratio:.4f}x (threshold 1.10x) -> {'PASS' if results['k1_pass'] else 'FAIL'}")

    # ------------------------------------------------------------------
    # Phase 6: Composed PPL comparison (1/N scaling)
    # ------------------------------------------------------------------
    print("\n[Phase 6] Composed PPL comparison (1/N scaling)...")

    # Dense composed
    print("\n  Dense composed (1/N):")
    dense_merged = compose_adapters(list(dense_adapters.values()))
    zero_lora_params(model)
    apply_adapter_weights(model, dense_merged)
    mx.eval(model.parameters())

    dense_composed_ppls = {}
    for domain_name, data_dir in data_dirs.items():
        ppl = compute_ppl(model, tokenizer, data_dir)
        dense_composed_ppls[domain_name] = ppl
        print(f"    {domain_name}: composed PPL = {ppl:.2f}")

    # Sparse composed
    print("\n  Sparse composed (1/N):")
    sparse_merged = compose_adapters(list(sparse_adapters.values()))
    zero_lora_params(model)
    apply_adapter_weights(model, sparse_merged)
    mx.eval(model.parameters())

    sparse_composed_ppls = {}
    for domain_name, data_dir in data_dirs.items():
        ppl = compute_ppl(model, tokenizer, data_dir)
        sparse_composed_ppls[domain_name] = ppl
        print(f"    {domain_name}: composed PPL = {ppl:.2f}")

    results["dense_composed_ppls"] = dense_composed_ppls
    results["sparse_composed_ppls"] = sparse_composed_ppls

    avg_dense_composed = sum(dense_composed_ppls.values()) / len(dense_composed_ppls)
    avg_sparse_composed = sum(sparse_composed_ppls.values()) / len(sparse_composed_ppls)
    results["avg_dense_composed_ppl"] = round(avg_dense_composed, 4)
    results["avg_sparse_composed_ppl"] = round(avg_sparse_composed, 4)

    # K2: sparse composed must not be worse than dense composed
    k2_ratios = {}
    for domain_name in DOMAINS:
        ratio = sparse_composed_ppls[domain_name] / dense_composed_ppls[domain_name]
        k2_ratios[domain_name] = round(ratio, 4)
    results["k2_composed_ratios"] = k2_ratios
    k2_avg_ratio = avg_sparse_composed / avg_dense_composed
    results["k2_avg_ratio"] = round(k2_avg_ratio, 4)
    results["k2_pass"] = k2_avg_ratio <= 1.0  # sparse should be same or better
    print(f"\n  K2: avg sparse/dense composed ratio = {k2_avg_ratio:.4f}x (threshold 1.0x) -> {'PASS' if results['k2_pass'] else 'FAIL'}")

    # ------------------------------------------------------------------
    # Phase 7: Orthogonality comparison
    # ------------------------------------------------------------------
    print("\n[Phase 7] Adapter orthogonality comparison...")

    print("\n  Dense adapter cosines:")
    dense_cosines, dense_mean_cos = compute_adapter_cosines(dense_adapters)
    print(f"  Mean |cos|: {dense_mean_cos:.6f}")
    for c in dense_cosines:
        print(f"    {c['pair']}: {c['abs_cos']:.6f}")

    print("\n  Sparse adapter cosines:")
    sparse_cosines, sparse_mean_cos = compute_adapter_cosines(sparse_adapters)
    print(f"  Mean |cos|: {sparse_mean_cos:.6f}")
    for c in sparse_cosines:
        print(f"    {c['pair']}: {c['abs_cos']:.6f}")

    results["dense_cosines"] = dense_cosines
    results["dense_mean_cos"] = round(dense_mean_cos, 6)
    results["sparse_cosines"] = sparse_cosines
    results["sparse_mean_cos"] = round(sparse_mean_cos, 6)
    results["cos_ratio"] = round(sparse_mean_cos / dense_mean_cos, 4) if dense_mean_cos > 0 else None

    # ------------------------------------------------------------------
    # Phase 8: B-matrix sparsity analysis
    # ------------------------------------------------------------------
    print("\n[Phase 8] B-matrix sparsity analysis...")

    # Count actual non-zero elements in sparse B matrices
    total_b_params = 0
    total_b_nonzero_dense = 0
    total_b_nonzero_sparse = 0

    for name in dense_adapters[list(DOMAINS.keys())[0]]:
        if "lora_b" in name:
            dense_vals = dense_adapters[list(DOMAINS.keys())[0]][name]
            sparse_vals = sparse_adapters[list(DOMAINS.keys())[0]][name]
            total_b_params += dense_vals.size
            total_b_nonzero_dense += mx.sum(mx.abs(dense_vals) > 1e-8).item()
            total_b_nonzero_sparse += mx.sum(mx.abs(sparse_vals) > 1e-8).item()

    actual_sparse_frac = 1.0 - total_b_nonzero_sparse / total_b_params
    print(f"  B params per adapter: {total_b_params:,}")
    print(f"  Dense non-zero: {total_b_nonzero_dense:,} ({total_b_nonzero_dense/total_b_params*100:.1f}%)")
    print(f"  Sparse non-zero: {total_b_nonzero_sparse:,} ({total_b_nonzero_sparse/total_b_params*100:.1f}%)")
    print(f"  Actual sparsity: {actual_sparse_frac*100:.1f}%")

    results["b_params_per_adapter"] = total_b_params
    results["dense_b_nonzero"] = total_b_nonzero_dense
    results["sparse_b_nonzero"] = total_b_nonzero_sparse
    results["actual_sparsity"] = round(actual_sparse_frac, 4)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    avg_base = sum(base_ppls.values()) / len(base_ppls)
    avg_dense_ind = sum(dense_individual_ppls.values()) / len(dense_individual_ppls)
    avg_sparse_ind = sum(sparse_individual_ppls.values()) / len(sparse_individual_ppls)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  K1 (sparse individual PPL <= 1.10x dense):")
    print(f"    Max ratio: {max_ratio:.4f}x, Mean ratio: {mean_ratio:.4f}x")
    print(f"    -> {'PASS' if results['k1_pass'] else 'FAIL'}")
    print(f"\n  K2 (sparse composed PPL <= dense composed):")
    print(f"    Avg ratio: {k2_avg_ratio:.4f}x")
    print(f"    -> {'PASS' if results['k2_pass'] else 'FAIL'}")

    print(f"\n  Avg base PPL:              {avg_base:.2f}")
    print(f"  Avg dense individual PPL:  {avg_dense_ind:.2f}")
    print(f"  Avg sparse individual PPL: {avg_sparse_ind:.2f}")
    print(f"  Avg dense composed PPL:    {avg_dense_composed:.2f}")
    print(f"  Avg sparse composed PPL:   {avg_sparse_composed:.2f}")
    print(f"\n  Dense mean |cos|:  {dense_mean_cos:.6f}")
    print(f"  Sparse mean |cos|: {sparse_mean_cos:.6f}")
    print(f"  Cosine ratio:      {results['cos_ratio']}")
    print(f"  Actual B sparsity: {actual_sparse_frac*100:.1f}%")

    # Verdict
    if results["k1_pass"] and results["k2_pass"]:
        verdict = "SUPPORTED"
    elif not results["k1_pass"]:
        verdict = "KILLED (K1: too much quality loss from sparsity)"
    else:
        verdict = "KILLED (K2: sparsity hurts composition)"

    results["verdict"] = verdict
    print(f"\n  VERDICT: {verdict}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
