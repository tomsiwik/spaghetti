#!/usr/bin/env python3
"""
BitNet-2B Ternary Composition Scale to N=15

Tests whether ternary LoRA composition on BitNet-b1.58-2B-4T scales from
N=5 to N=15 domains without degradation.

Kill criteria:
  K1: composition ratio N=15 > 2x composition ratio N=5 (scaling degrades rapidly)
  K2: mean |cos| at N=15 > 0.01 (packing pressure emerging)
  K3: any existing domain PPL degrades >10% when 10 new domains added

Reuses: 5 trained adapters from bitnet_multiseed_validation seed 42.
Trains: 10 new ternary adapters from HF datasets.
Platform: Apple Silicon MLX, $0.
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from itertools import combinations

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
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 128
LEARNING_RATE = 1e-4
VAL_BATCHES = 25
TRAIN_STEPS = 400

SEED = 42  # Match multiseed seed 42

EXPERIMENT_DIR = Path(__file__).parent
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_DIR = EXPERIMENT_DIR / "data"

# Existing adapters from multiseed validation (seed 42)
EXISTING_ADAPTER_DIR = (
    Path(__file__).parent.parent / "bitnet_multiseed_validation" / "adapters" / "seed42"
)
# Existing data from ternary convergence
EXISTING_DATA_DIR = (
    Path(__file__).parent.parent / "bitnet_ternary_convergence" / "data"
)

# Original 5 domains (reuse adapters)
ORIGINAL_DOMAINS = ["medical", "code", "math", "legal", "creative"]

# Original domain configs (for data/eval only)
ORIGINAL_DOMAIN_CONFIGS = {
    "medical": {
        "hf_dataset": "medalpaca/medical_meadow_medical_flashcards",
        "text_key": "output",
        "max_samples_train": 800,
        "max_samples_val": 100,
    },
    "code": {
        "hf_dataset": "iamtarun/python_code_instructions_18k_alpaca",
        "text_key": "output",
        "max_samples_train": 800,
        "max_samples_val": 100,
    },
    "math": {
        "hf_dataset": "gsm8k",
        "hf_subset": "main",
        "text_key": "answer",
        "max_samples_train": 800,
        "max_samples_val": 100,
    },
    "legal": {
        "hf_dataset": "jonathanli/law-stack-exchange",
        "text_key": "body",
        "max_samples_train": 500,
        "max_samples_val": 80,
    },
    "creative": {
        "hf_dataset": "roneneldan/TinyStories",
        "text_key": "text",
        "max_samples_train": 800,
        "max_samples_val": 100,
    },
}

# 10 NEW domains to train (all verified accessible, >900 samples each)
NEW_DOMAINS = {
    "sql": {
        "hf_dataset": "b-mc2/sql-create-context",
        "text_key": "answer",
        "max_samples_train": 800,
        "max_samples_val": 100,
    },
    "javascript": {
        "hf_dataset": "Nan-Do/code-search-net-javascript",
        "text_key": "code",
        "max_samples_train": 800,
        "max_samples_val": 100,
    },
    "physics": {
        # Science reasoning (openbookqa - fast download)
        "hf_dataset": "openbookqa",
        "hf_subset": "main",
        "text_key": "question_stem",
        "max_samples_train": 800,
        "max_samples_val": 100,
    },
    "chemistry": {
        # Science factual knowledge (sciq - fast download)
        "hf_dataset": "allenai/sciq",
        "text_key": "support",
        "max_samples_train": 800,
        "max_samples_val": 100,
    },
    "science": {
        # Scientific reasoning (scitail)
        "hf_dataset": "scitail",
        "hf_subset": "snli_format",
        "text_key": "sentence1",
        "max_samples_train": 800,
        "max_samples_val": 100,
    },
    "wikitext": {
        # General encyclopedic text
        "hf_dataset": "wikitext",
        "hf_subset": "wikitext-103-raw-v1",
        "text_key": "text",
        "max_samples_train": 800,
        "max_samples_val": 100,
    },
    "finance": {
        "hf_dataset": "gbharti/finance-alpaca",
        "text_key": "output",
        "max_samples_train": 800,
        "max_samples_val": 100,
    },
    "cooking": {
        "hf_dataset": "Hieu-Pham/kaggle_food_recipes",
        "text_key": "Instructions",
        "max_samples_train": 800,
        "max_samples_val": 100,
    },
    "health": {
        # Medical QA (distinct from medical flashcards)
        "hf_dataset": "keivalya/MedQuad-MedicalQnADataset",
        "text_key": "Answer",
        "max_samples_train": 800,
        "max_samples_val": 100,
    },
    "dialogue": {
        # Conversational/instruction text
        "hf_dataset": "tasksource/oasst1_pairwise_rlhf_reward",
        "text_key": "chosen",
        "max_samples_train": 800,
        "max_samples_val": 100,
    },
}


# ===========================================================================
# Ternary weight unpacking (from multiseed validation)
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
    print(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


# ===========================================================================
# TernaryLoRALinear with STE (from multiseed validation)
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
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora = TernaryLoRALinear(module, r=rank, scale=scale)
                updates.append((key, lora))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    print(f"  Applied Ternary LoRA (r={rank}) to {count} layers")
    return model


def remove_lora(model):
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                updates.append((key, module.linear))
        if updates:
            layer.update_modules(tree_unflatten(updates))
    return model


def get_lora_params(model):
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_a" in name or "lora_b" in name:
            params[name] = mx.array(p)
    mx.eval(params)
    return params


def zero_lora_params(model, seed=None):
    if seed is not None:
        mx.random.seed(seed)
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                in_dims = module.lora_a.shape[0]
                s = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(low=-s, high=s, shape=module.lora_a.shape)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def apply_adapter_weights(model, adapter_params, scale=1.0):
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def save_adapter(model, path):
    path.mkdir(parents=True, exist_ok=True)
    params = get_lora_params(model)
    mx.savez(str(path / "adapter.npz"), **params)
    return params


def load_adapter(path):
    return dict(mx.load(str(path / "adapter.npz")))


# ===========================================================================
# Data preparation
# ===========================================================================
def ensure_data(domain_name, domain_config, data_root):
    data_dir = data_root / domain_name
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"

    if train_path.exists() and valid_path.exists():
        with open(train_path) as f:
            n_train = sum(1 for _ in f)
        with open(valid_path) as f:
            n_val = sum(1 for _ in f)
        if n_train >= 50 and n_val >= 10:
            print(f"  {domain_name}: data exists ({n_train} train, {n_val} val)")
            return data_dir

    from datasets import load_dataset as hf_load
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {domain_config['hf_dataset']}...")

    kwargs = {}
    if "hf_subset" in domain_config:
        kwargs["name"] = domain_config["hf_subset"]

    try:
        ds = hf_load(domain_config["hf_dataset"], **kwargs)
    except Exception as e:
        print(f"  WARNING: Failed to load {domain_config['hf_dataset']}: {e}")
        if "fallback_dataset" in domain_config:
            print(f"  Trying fallback: {domain_config['fallback_dataset']}...")
            ds = hf_load(domain_config["fallback_dataset"], trust_remote_code=True)
            domain_config = dict(domain_config)
            domain_config["text_key"] = domain_config["fallback_text_key"]
        else:
            raise

    split_data = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

    text_key = domain_config["text_key"]
    if text_key not in split_data.column_names:
        for alt in ["text", "content", "output", "answer", "response", "input", "question", "body"]:
            if alt in split_data.column_names:
                text_key = alt
                print(f"  Using '{text_key}' instead of '{domain_config['text_key']}'")
                break

    max_train = domain_config["max_samples_train"]
    max_val = domain_config["max_samples_val"]
    filter_fn = domain_config.get("filter_fn")

    texts = []
    for row in split_data:
        t = row.get(text_key, "")
        if isinstance(t, list):
            t = " ".join(str(x) for x in t)
        if not isinstance(t, str) or len(t.strip()) < 20:
            continue
        t = t.strip()

        # Optional content filter
        if filter_fn == "history":
            keywords = ["history", "war", "century", "empire", "ancient", "medieval",
                        "revolution", "dynasty", "civilization", "kingdom", "colonial"]
            if not any(kw in t.lower() for kw in keywords):
                continue
        elif filter_fn == "astronomy":
            keywords = ["star", "planet", "galaxy", "orbit", "telescope", "universe",
                        "solar", "cosmic", "nebula", "black hole", "astrono", "celestial"]
            if not any(kw in t.lower() for kw in keywords):
                continue

        texts.append(t)
        if len(texts) >= (max_train + max_val) * 2:
            break

    # If not enough filtered texts, relax the filter
    if len(texts) < max_train + max_val and filter_fn:
        print(f"  Filter '{filter_fn}' too strict ({len(texts)} texts), using unfiltered...")
        texts = []
        for row in split_data:
            t = row.get(text_key, "")
            if isinstance(t, list):
                t = " ".join(str(x) for x in t)
            if isinstance(t, str) and len(t.strip()) > 20:
                texts.append(t.strip())
            if len(texts) >= (max_train + max_val) * 2:
                break

    train_texts = texts[:max_train]
    val_texts = texts[max_train:max_train + max_val]

    if len(train_texts) < 50 or len(val_texts) < 10:
        raise ValueError(f"Not enough data for {domain_name}: {len(train_texts)} train, {len(val_texts)} val")

    with open(train_path, "w") as f:
        for t in train_texts:
            json.dump({"text": t[:4000]}, f)  # cap text length
            f.write("\n")

    with open(valid_path, "w") as f:
        for t in val_texts:
            json.dump({"text": t[:4000]}, f)
            f.write("\n")

    print(f"  {domain_name}: {len(train_texts)} train, {len(val_texts)} val")
    return data_dir


# ===========================================================================
# PPL evaluation
# ===========================================================================
def compute_ppl(model, tokenizer, data_path, max_batches=25, split="valid"):
    fpath = data_path / f"{split}.jsonl"
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


def compute_cosines(adapters_dict):
    names = list(adapters_dict.keys())
    cosines = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            vi = mx.concatenate([v.reshape(-1) for v in adapters_dict[names[i]].values()])
            vj = mx.concatenate([v.reshape(-1) for v in adapters_dict[names[j]].values()])
            cos = mx.abs(mx.sum(vi * vj) / (mx.sqrt(mx.sum(vi**2)) * mx.sqrt(mx.sum(vj**2))))
            mx.eval(cos)
            cosines.append({"pair": f"{names[i]}-{names[j]}", "abs_cos": round(cos.item(), 6)})
    mean_cos = sum(c["abs_cos"] for c in cosines) / len(cosines) if cosines else 0
    return cosines, round(mean_cos, 6)


# ===========================================================================
# Training
# ===========================================================================
def train_adapter(model, tokenizer, data_dir, domain_name, n_steps, seed):
    import random

    train_texts = []
    with open(data_dir / "train.jsonl") as f:
        for line in f:
            train_texts.append(json.loads(line)["text"])

    train_tokens = []
    for text in train_texts:
        toks = tokenizer.encode(text)
        if len(toks) > 2:
            train_tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))

    rng = random.Random(seed + hash(domain_name) % 10000)
    indices = list(range(len(train_tokens)))

    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    t_start = time.time()
    losses = []

    for step in range(n_steps):
        idx = indices[step % len(indices)]
        tokens = train_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()
        losses.append(loss_val)

        if (step + 1) % 100 == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            print(f"      Step {step+1}/{n_steps}: loss={loss_val:.4f} (avg50={avg:.4f})")

    train_time = time.time() - t_start
    first_50 = sum(losses[:50]) / 50
    last_50 = sum(losses[-50:]) / 50
    converged = last_50 < first_50 * 0.95

    return {
        "train_time_s": round(train_time, 1),
        "first_50_loss": round(first_50, 4),
        "last_50_loss": round(last_50, 4),
        "converged": converged,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_global = time.time()

    all_domains = ORIGINAL_DOMAINS + list(NEW_DOMAINS.keys())

    results = {
        "experiment": "bitnet_scale_n15",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "train_steps": TRAIN_STEPS,
        "seed": SEED,
        "original_domains": ORIGINAL_DOMAINS,
        "new_domains": list(NEW_DOMAINS.keys()),
        "all_domains": all_domains,
        "n_original": len(ORIGINAL_DOMAINS),
        "n_new": len(NEW_DOMAINS),
        "n_total": len(all_domains),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=" * 70)
    print("BitNet-2B Ternary Composition: Scale to N=15")
    print(f"  Original 5: {ORIGINAL_DOMAINS}")
    print(f"  New 10: {list(NEW_DOMAINS.keys())}")
    print(f"  Total: {len(all_domains)} domains")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 0: Load model
    # ------------------------------------------------------------------
    print("\n[Phase 0] Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("  Unpacking ternary weights...")
    model = replace_bitlinear_with_linear(model)

    # ------------------------------------------------------------------
    # Phase 1: Prepare data for all domains
    # ------------------------------------------------------------------
    print("\n[Phase 1] Preparing data...")

    data_dirs = {}

    # Original 5: reuse existing data
    for domain_name in ORIGINAL_DOMAINS:
        config = ORIGINAL_DOMAIN_CONFIGS[domain_name]
        # Try existing data first
        existing_data = EXISTING_DATA_DIR / domain_name
        if existing_data.exists():
            data_dirs[domain_name] = existing_data
            print(f"  {domain_name}: reusing from {existing_data}")
        else:
            data_dirs[domain_name] = ensure_data(domain_name, config, DATA_DIR)

    # New 10: download
    for domain_name, config in NEW_DOMAINS.items():
        data_dirs[domain_name] = ensure_data(domain_name, config, DATA_DIR)

    # ------------------------------------------------------------------
    # Phase 2: Base PPL for all 15 domains
    # ------------------------------------------------------------------
    print("\n[Phase 2] Base model PPL (all 15 domains)...")
    base_ppls = {}
    for domain_name in all_domains:
        ppl = compute_ppl(model, tokenizer, data_dirs[domain_name])
        base_ppls[domain_name] = round(ppl, 4)
        print(f"  {domain_name}: {ppl:.4f}")
    results["base_ppls"] = base_ppls

    # ------------------------------------------------------------------
    # Phase 3: Load existing + train new adapters
    # ------------------------------------------------------------------
    print("\n[Phase 3] Loading/training adapters...")

    # Apply ternary LoRA
    model = apply_ternary_lora(model, rank=LORA_RANK, scale=LORA_SCALE)
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"  Trainable params: {trainable:,}")
    results["trainable_params"] = trainable

    all_adapters = {}

    # Load existing 5
    print("\n  Loading 5 existing adapters from multiseed seed 42...")
    for domain_name in ORIGINAL_DOMAINS:
        adapter_path = EXISTING_ADAPTER_DIR / domain_name
        if not adapter_path.exists():
            print(f"  WARNING: {adapter_path} not found, will train fresh")
            # Train fresh as fallback
            zero_lora_params(model, seed=SEED * 1000 + hash(domain_name) % 10000)
            train_result = train_adapter(
                model, tokenizer, data_dirs[domain_name], domain_name, TRAIN_STEPS, SEED
            )
            params = save_adapter(model, ADAPTERS_DIR / domain_name)
            all_adapters[domain_name] = params
            print(f"    {domain_name}: trained fresh ({train_result['train_time_s']}s)")
        else:
            params = load_adapter(adapter_path)
            all_adapters[domain_name] = params
            # Also save a copy locally
            local_path = ADAPTERS_DIR / domain_name
            local_path.mkdir(parents=True, exist_ok=True)
            mx.savez(str(local_path / "adapter.npz"), **params)
            print(f"    {domain_name}: loaded from {adapter_path}")

    # Train 10 new
    print("\n  Training 10 new adapters...")
    train_results = {}
    for domain_name in NEW_DOMAINS:
        print(f"\n  --- Training: {domain_name} ---")
        zero_lora_params(model, seed=SEED * 1000 + hash(domain_name) % 10000)

        train_result = train_adapter(
            model, tokenizer, data_dirs[domain_name], domain_name, TRAIN_STEPS, SEED
        )
        train_results[domain_name] = train_result

        params = save_adapter(model, ADAPTERS_DIR / domain_name)
        all_adapters[domain_name] = params
        print(f"    Saved. Time: {train_result['train_time_s']}s, "
              f"converged: {train_result['converged']}")

    results["train_results"] = train_results

    # ------------------------------------------------------------------
    # Phase 4: Individual PPL for all 15
    # ------------------------------------------------------------------
    print("\n[Phase 4] Individual adapter PPL (all 15)...")
    individual_ppls = {}
    for domain_name in all_domains:
        zero_lora_params(model)
        apply_adapter_weights(model, all_adapters[domain_name])
        mx.eval(model.parameters())
        ppl = compute_ppl(model, tokenizer, data_dirs[domain_name])
        individual_ppls[domain_name] = round(ppl, 4)
        base = base_ppls[domain_name]
        imp = (base - ppl) / base * 100
        print(f"    {domain_name}: {ppl:.4f} (base={base:.4f}, {imp:+.1f}%)")
    results["individual_ppls"] = individual_ppls

    # ------------------------------------------------------------------
    # Phase 5: N=5 composition (original 5 only) - baseline
    # ------------------------------------------------------------------
    print("\n[Phase 5] N=5 composition (original 5 only)...")
    orig_adapter_list = [all_adapters[d] for d in ORIGINAL_DOMAINS]
    merged_5 = compose_adapters(orig_adapter_list)

    zero_lora_params(model)
    apply_adapter_weights(model, merged_5)
    mx.eval(model.parameters())

    composed_5_ppls = {}
    for domain_name in ORIGINAL_DOMAINS:
        ppl = compute_ppl(model, tokenizer, data_dirs[domain_name])
        composed_5_ppls[domain_name] = round(ppl, 4)
        print(f"    {domain_name}: {ppl:.4f}")

    best_ind_5 = min(individual_ppls[d] for d in ORIGINAL_DOMAINS)
    avg_composed_5 = sum(composed_5_ppls.values()) / len(composed_5_ppls)
    ratio_5 = avg_composed_5 / best_ind_5

    results["n5_composed_ppls"] = composed_5_ppls
    results["n5_best_individual"] = round(best_ind_5, 4)
    results["n5_avg_composed"] = round(avg_composed_5, 4)
    results["n5_composition_ratio"] = round(ratio_5, 4)
    print(f"\n  N=5: ratio = {ratio_5:.4f}x (avg_composed={avg_composed_5:.4f}, best_ind={best_ind_5:.4f})")

    # ------------------------------------------------------------------
    # Phase 6: N=15 composition (all 15)
    # ------------------------------------------------------------------
    print("\n[Phase 6] N=15 composition (all 15)...")
    all_adapter_list = [all_adapters[d] for d in all_domains]
    merged_15 = compose_adapters(all_adapter_list)

    zero_lora_params(model)
    apply_adapter_weights(model, merged_15)
    mx.eval(model.parameters())

    composed_15_ppls = {}
    for domain_name in all_domains:
        ppl = compute_ppl(model, tokenizer, data_dirs[domain_name])
        composed_15_ppls[domain_name] = round(ppl, 4)
        print(f"    {domain_name}: {ppl:.4f}")

    best_ind_15 = min(individual_ppls[d] for d in all_domains)
    avg_composed_15 = sum(composed_15_ppls.values()) / len(composed_15_ppls)
    ratio_15 = avg_composed_15 / best_ind_15

    results["n15_composed_ppls"] = composed_15_ppls
    results["n15_best_individual"] = round(best_ind_15, 4)
    results["n15_avg_composed"] = round(avg_composed_15, 4)
    results["n15_composition_ratio"] = round(ratio_15, 4)
    print(f"\n  N=15: ratio = {ratio_15:.4f}x (avg_composed={avg_composed_15:.4f}, best_ind={best_ind_15:.4f})")

    # ------------------------------------------------------------------
    # Phase 7: Cosines (all 105 pairs)
    # ------------------------------------------------------------------
    print("\n[Phase 7] Cosine similarity (105 pairs)...")
    cosines, mean_cos = compute_cosines(all_adapters)
    results["cosines"] = cosines
    results["mean_cos"] = mean_cos
    print(f"  Mean |cos|: {mean_cos}")

    # Also compute N=5 cosines for comparison
    orig_adapters = {d: all_adapters[d] for d in ORIGINAL_DOMAINS}
    cosines_5, mean_cos_5 = compute_cosines(orig_adapters)
    results["n5_cosines"] = cosines_5
    results["n5_mean_cos"] = mean_cos_5
    print(f"  N=5 mean |cos|: {mean_cos_5}")
    print(f"  N=15 mean |cos|: {mean_cos}")

    # Report top-5 and bottom-5 cosines
    sorted_cos = sorted(cosines, key=lambda c: c["abs_cos"], reverse=True)
    print(f"\n  Top 5 most similar pairs:")
    for c in sorted_cos[:5]:
        print(f"    {c['pair']}: {c['abs_cos']:.6f}")
    print(f"  Bottom 5 most orthogonal pairs:")
    for c in sorted_cos[-5:]:
        print(f"    {c['pair']}: {c['abs_cos']:.6f}")

    # ------------------------------------------------------------------
    # Phase 8: K3 - Per-domain degradation check
    # ------------------------------------------------------------------
    print("\n[Phase 8] Per-domain degradation (original 5 under N=15 vs N=5)...")
    degradation = {}
    for domain_name in ORIGINAL_DOMAINS:
        ppl_5 = composed_5_ppls[domain_name]
        ppl_15 = composed_15_ppls[domain_name]
        pct_change = (ppl_15 - ppl_5) / ppl_5 * 100
        degradation[domain_name] = {
            "ppl_n5": ppl_5,
            "ppl_n15": ppl_15,
            "pct_change": round(pct_change, 2),
            "pass": pct_change <= 10.0,
        }
        status = "PASS" if pct_change <= 10.0 else "FAIL"
        print(f"    {domain_name}: N=5={ppl_5:.4f} -> N=15={ppl_15:.4f} ({pct_change:+.2f}%) [{status}]")
    results["degradation"] = degradation

    # ------------------------------------------------------------------
    # Kill criteria
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("KILL CRITERIA")
    print("=" * 70)

    # K1: N=15 ratio < 2x * N=5 ratio
    # Use the multiseed mean as N=5 baseline: 3.4398
    n5_reference_ratio = 3.4398  # from multiseed aggregate
    k1_threshold = 2.0 * n5_reference_ratio
    k1_pass = ratio_15 <= k1_threshold
    print(f"\n  K1 (ratio N=15 < 2x ratio N=5):")
    print(f"    N=5 reference ratio: {n5_reference_ratio:.4f}x (multiseed mean)")
    print(f"    N=15 ratio: {ratio_15:.4f}x")
    print(f"    Threshold: {k1_threshold:.4f}x")
    print(f"    Ratio of ratios: {ratio_15/n5_reference_ratio:.4f}x")
    print(f"    -> {'PASS' if k1_pass else 'KILL'}")

    # K2: mean |cos| < 0.01
    k2_pass = mean_cos <= 0.01
    print(f"\n  K2 (mean |cos| < 0.01):")
    print(f"    Mean |cos| N=15: {mean_cos}")
    print(f"    -> {'PASS' if k2_pass else 'KILL'}")

    # K3: no original domain degrades >10%
    k3_pass = all(d["pass"] for d in degradation.values())
    max_degrad = max(d["pct_change"] for d in degradation.values())
    worst_domain = max(degradation, key=lambda d: degradation[d]["pct_change"])
    print(f"\n  K3 (no original domain >10% degradation):")
    print(f"    Worst: {worst_domain} at {max_degrad:+.2f}%")
    print(f"    -> {'PASS' if k3_pass else 'KILL'}")

    results["k1_pass"] = k1_pass
    results["k1_threshold"] = round(k1_threshold, 4)
    results["k1_n5_reference"] = n5_reference_ratio
    results["k1_ratio_of_ratios"] = round(ratio_15 / n5_reference_ratio, 4)
    results["k2_pass"] = k2_pass
    results["k3_pass"] = k3_pass
    results["k3_worst_domain"] = worst_domain
    results["k3_max_degradation_pct"] = round(max_degrad, 2)

    verdict = "SUPPORTED" if (k1_pass and k2_pass and k3_pass) else "KILLED"
    results["verdict"] = verdict

    total_time = time.time() - t_global
    results["total_time_s"] = round(total_time, 1)
    results["total_time_min"] = round(total_time / 60, 1)

    print(f"\n  VERDICT: {verdict}")
    print(f"  Total time: {total_time/60:.1f} min")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  N=5  composition ratio: {ratio_5:.4f}x (this run)")
    print(f"  N=5  reference ratio:   {n5_reference_ratio:.4f}x (multiseed)")
    print(f"  N=15 composition ratio: {ratio_15:.4f}x")
    print(f"  Ratio scaling:          {ratio_15/n5_reference_ratio:.4f}x")
    print(f"  N=5  mean |cos|:        {mean_cos_5}")
    print(f"  N=15 mean |cos|:        {mean_cos}")
    print(f"  Cos scaling:            {mean_cos/mean_cos_5:.2f}x" if mean_cos_5 > 0 else "  Cos scaling: N/A")

    avg_base_all = sum(base_ppls.values()) / len(base_ppls)
    avg_ind_all = sum(individual_ppls.values()) / len(individual_ppls)
    print(f"\n  Avg base PPL (15):      {avg_base_all:.4f}")
    print(f"  Avg individual PPL (15):{avg_ind_all:.4f}")
    print(f"  Avg composed PPL (15):  {avg_composed_15:.4f}")

    n_improved = sum(1 for d in all_domains if individual_ppls[d] < base_ppls[d])
    print(f"  Domains improved by individual adapter: {n_improved}/{len(all_domains)}")

    n_converged = sum(1 for r in train_results.values() if r["converged"])
    print(f"  New domains converged: {n_converged}/{len(NEW_DOMAINS)}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
