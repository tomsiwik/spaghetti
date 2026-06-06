#!/usr/bin/env python3
"""
BitNet-2B Ternary Composition Scale to N=25 (Domains + Capabilities)

Tests whether ternary LoRA composition on BitNet-b1.58-2B-4T scales from
N=15 domains to N=25 (15 domains + 4 existing capabilities + 6 new capabilities)
without degradation.

Kill criteria:
  K1: composition ratio N=25 > 5x (approaching catastrophe)
  K2: cross-type cosine (capability-domain) > 0.01

Reuses:
  - 15 trained domain adapters from bitnet_scale_n15
  - 4 trained capability adapters from capability_expert_taxonomy
    (reasoning, instruction, conciseness, safety)
Trains:
  - 6 new capability adapters from HuggingFace datasets ($0)

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

SEED = 42

EXPERIMENT_DIR = Path(__file__).parent
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_DIR = EXPERIMENT_DIR / "data"

# Existing adapters
N15_ADAPTER_DIR = Path(__file__).parent.parent / "bitnet_scale_n15" / "adapters"
CAP_ADAPTER_DIR = Path(__file__).parent.parent / "capability_expert_taxonomy" / "adapters"

# Existing data directories
N15_DATA_DIR = Path(__file__).parent.parent / "bitnet_scale_n15" / "data"
EXISTING_DATA_DIR = Path(__file__).parent.parent / "bitnet_ternary_convergence" / "data"
CAP_DATA_DIR = Path(__file__).parent.parent / "capability_expert_taxonomy" / "data"

# 15 domain adapters (all from N=15 experiment)
DOMAIN_NAMES = [
    "medical", "code", "math", "legal", "creative",
    "sql", "javascript", "physics", "chemistry", "science",
    "wikitext", "finance", "cooking", "health", "dialogue",
]

# 4 existing capability adapters
EXISTING_CAP_NAMES = ["reasoning", "instruction", "conciseness", "safety"]

# 6 new capability adapters to train
NEW_CAPABILITIES = {
    "multilingual": {
        "description": "Non-English text (German Wikipedia)",
        "hf_dataset": "wikimedia/wikipedia",
        "hf_subset": "20231101.de",
        "text_key": "text",
        "max_samples_train": 500,
        "max_samples_val": 50,
    },
    "coding_style": {
        "description": "Documented code with comments (Python)",
        "hf_dataset": "iamtarun/python_code_instructions_18k_alpaca",
        "text_key": "output",
        "filter_fn": "docstring",
        "max_samples_train": 500,
        "max_samples_val": 50,
    },
    "summarization": {
        "description": "Concise summaries of documents",
        "hf_dataset": "EdinburghNLP/xsum",
        "text_key": "summary",
        "max_samples_train": 500,
        "max_samples_val": 50,
    },
    "debate": {
        "description": "Argumentative/persuasive text from QA pairs",
        "hf_dataset": "yahoo_answers_topics",
        "text_key": "best_answer",
        "max_samples_train": 500,
        "max_samples_val": 50,
    },
    "translation": {
        "description": "English-French translation (OPUS books)",
        "hf_dataset": "Helsinki-NLP/opus_books",
        "hf_subset": "en-fr",
        "text_key": "translation",
        "max_samples_train": 500,
        "max_samples_val": 50,
    },
    "formal_writing": {
        "description": "Academic/formal writing style (PubMed abstracts)",
        "hf_dataset": "ccdv/pubmed-summarization",
        "text_key": "abstract",
        "max_samples_train": 500,
        "max_samples_val": 50,
    },
}


def log(msg):
    print(msg, flush=True)


# ===========================================================================
# Ternary weight unpacking (from N=15 experiment)
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
    log(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


# ===========================================================================
# TernaryLoRALinear with STE (identical to N=15 experiment)
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
    log(f"  Applied Ternary LoRA (r={rank}) to {count} layers")
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
            log(f"  {domain_name}: data exists ({n_train} train, {n_val} val)")
            return data_dir

    from datasets import load_dataset as hf_load
    data_dir.mkdir(parents=True, exist_ok=True)
    log(f"  Downloading {domain_config['hf_dataset']}...")

    kwargs = {}
    if "hf_subset" in domain_config:
        kwargs["name"] = domain_config["hf_subset"]

    try:
        ds = hf_load(domain_config["hf_dataset"], **kwargs)
    except Exception as e:
        log(f"  WARNING: Failed to load {domain_config['hf_dataset']}: {e}")
        raise

    split_data = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

    text_key = domain_config["text_key"]

    # Handle translation datasets (dict field)
    if text_key == "translation":
        texts = []
        max_total = domain_config["max_samples_train"] + domain_config["max_samples_val"]
        for row in split_data:
            t = row.get("translation", {})
            if isinstance(t, dict):
                # Combine source and target for translation training
                parts = list(t.values())
                combined = " ||| ".join(str(p) for p in parts if p)
                if len(combined.strip()) > 20:
                    texts.append(combined.strip())
            if len(texts) >= max_total * 2:
                break
    elif text_key == "sourceString":
        # Tatoeba MT format
        texts = []
        max_total = domain_config["max_samples_train"] + domain_config["max_samples_val"]
        for row in split_data:
            src = row.get("sourceString", "")
            tgt = row.get("targetString", "")
            if isinstance(src, str) and isinstance(tgt, str) and len(src) > 5:
                combined = f"{src} ||| {tgt}"
                texts.append(combined)
            if len(texts) >= max_total * 2:
                break
    else:
        if text_key not in split_data.column_names:
            for alt in ["text", "content", "output", "answer", "response",
                        "input", "question", "body", "abstract", "summary",
                        "document", "sentence", "chosen"]:
                if alt in split_data.column_names:
                    text_key = alt
                    log(f"  Using '{text_key}' instead of '{domain_config['text_key']}'")
                    break

        filter_fn = domain_config.get("filter_fn")
        texts = []
        max_total = domain_config["max_samples_train"] + domain_config["max_samples_val"]

        for row in split_data:
            t = row.get(text_key, "")
            if isinstance(t, list):
                t = " ".join(str(x) for x in t)
            if not isinstance(t, str) or len(t.strip()) < 20:
                continue
            t = t.strip()

            # Optional content filter
            if filter_fn == "docstring":
                if '"""' not in t and "'''" not in t and "# " not in t:
                    continue

            texts.append(t)
            if len(texts) >= max_total * 2:
                break

        # If filter too strict, relax
        if len(texts) < max_total and filter_fn:
            log(f"  Filter '{filter_fn}' too strict ({len(texts)} texts), relaxing...")
            texts = []
            for row in split_data:
                t = row.get(text_key, "")
                if isinstance(t, list):
                    t = " ".join(str(x) for x in t)
                if isinstance(t, str) and len(t.strip()) > 20:
                    texts.append(t.strip())
                if len(texts) >= max_total * 2:
                    break

    max_train = domain_config["max_samples_train"]
    max_val = domain_config["max_samples_val"]
    train_texts = texts[:max_train]
    val_texts = texts[max_train:max_train + max_val]

    if len(train_texts) < 50 or len(val_texts) < 10:
        raise ValueError(
            f"Not enough data for {domain_name}: {len(train_texts)} train, {len(val_texts)} val"
        )

    with open(train_path, "w") as f:
        for t in train_texts:
            json.dump({"text": t[:4000]}, f)
            f.write("\n")

    with open(valid_path, "w") as f:
        for t in val_texts:
            json.dump({"text": t[:4000]}, f)
            f.write("\n")

    log(f"  {domain_name}: {len(train_texts)} train, {len(val_texts)} val")
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


def compute_cosines_with_categories(adapters_dict, domain_names, cap_names):
    """Compute pairwise cosines with category labels."""
    names = list(adapters_dict.keys())
    cosines = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            vi = mx.concatenate([v.reshape(-1) for v in adapters_dict[names[i]].values()])
            vj = mx.concatenate([v.reshape(-1) for v in adapters_dict[names[j]].values()])
            cos = mx.abs(mx.sum(vi * vj) / (mx.sqrt(mx.sum(vi**2)) * mx.sqrt(mx.sum(vj**2))))
            mx.eval(cos)

            # Categorize
            ni_is_domain = names[i] in domain_names
            nj_is_domain = names[j] in domain_names
            if ni_is_domain and nj_is_domain:
                cat = "domain-domain"
            elif not ni_is_domain and not nj_is_domain:
                cat = "capability-capability"
            else:
                cat = "capability-domain"

            cosines.append({
                "pair": f"{names[i]}-{names[j]}",
                "abs_cos": round(cos.item(), 6),
                "category": cat,
            })

    return cosines


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

    if not train_tokens:
        raise ValueError(f"No valid training tokens for {domain_name}")

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
            log(f"      Step {step+1}/{n_steps}: loss={loss_val:.4f} (avg50={avg:.4f})")

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

    all_cap_names = EXISTING_CAP_NAMES + list(NEW_CAPABILITIES.keys())
    all_names = DOMAIN_NAMES + all_cap_names
    N_total = len(all_names)

    results = {
        "experiment": "bitnet_scale_n25",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "train_steps": TRAIN_STEPS,
        "seed": SEED,
        "domain_names": DOMAIN_NAMES,
        "existing_cap_names": EXISTING_CAP_NAMES,
        "new_cap_names": list(NEW_CAPABILITIES.keys()),
        "all_names": all_names,
        "n_domains": len(DOMAIN_NAMES),
        "n_existing_caps": len(EXISTING_CAP_NAMES),
        "n_new_caps": len(NEW_CAPABILITIES),
        "n_total": N_total,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    log("=" * 70)
    log("BitNet-2B Ternary Composition: Scale to N=25 (Domains + Capabilities)")
    log(f"  15 domains: {DOMAIN_NAMES}")
    log(f"  4 existing caps: {EXISTING_CAP_NAMES}")
    log(f"  6 new caps: {list(NEW_CAPABILITIES.keys())}")
    log(f"  Total: {N_total}")
    log("=" * 70)

    # ------------------------------------------------------------------
    # Phase 0: Load model
    # ------------------------------------------------------------------
    log("\n[Phase 0] Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  Loaded in {time.time() - t0:.1f}s")

    log("  Unpacking ternary weights...")
    model = replace_bitlinear_with_linear(model)

    # ------------------------------------------------------------------
    # Phase 1: Prepare data for new capabilities
    # ------------------------------------------------------------------
    log("\n[Phase 1] Preparing data for 6 new capabilities...")
    data_dirs = {}

    # Domain data: reuse from N=15 or existing
    for domain_name in DOMAIN_NAMES:
        # Check N=15 data first, then existing data dir
        n15_data = N15_DATA_DIR / domain_name
        existing_data = EXISTING_DATA_DIR / domain_name
        if n15_data.exists():
            data_dirs[domain_name] = n15_data
        elif existing_data.exists():
            data_dirs[domain_name] = existing_data
        else:
            log(f"  WARNING: No data for domain {domain_name}")
            data_dirs[domain_name] = None

    # Existing capability data
    for cap_name in EXISTING_CAP_NAMES:
        cap_data = CAP_DATA_DIR / cap_name
        if cap_data.exists():
            data_dirs[cap_name] = cap_data
            log(f"  {cap_name}: reusing from {cap_data}")
        else:
            log(f"  WARNING: No data for capability {cap_name}")
            data_dirs[cap_name] = None

    # New capability data: download
    for cap_name, config in NEW_CAPABILITIES.items():
        data_dirs[cap_name] = ensure_data(cap_name, config, DATA_DIR)

    # ------------------------------------------------------------------
    # Phase 2: Compute base PPL for all 25
    # ------------------------------------------------------------------
    log("\n[Phase 2] Base model PPL...")
    base_ppls = {}
    for name in all_names:
        if data_dirs.get(name) is None:
            log(f"  {name}: SKIPPED (no data)")
            continue
        ppl = compute_ppl(model, tokenizer, data_dirs[name])
        base_ppls[name] = round(ppl, 4)
        log(f"  {name}: {ppl:.4f}")
    results["base_ppls"] = base_ppls

    # ------------------------------------------------------------------
    # Phase 3: Apply LoRA structure and load/train adapters
    # ------------------------------------------------------------------
    log("\n[Phase 3] Loading/training adapters...")
    model = apply_ternary_lora(model, rank=LORA_RANK, scale=LORA_SCALE)
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"  Trainable params: {trainable:,}")
    results["trainable_params"] = trainable

    all_adapters = {}

    # Load 15 domain adapters from N=15
    log("\n  Loading 15 domain adapters from bitnet_scale_n15...")
    for domain_name in DOMAIN_NAMES:
        adapter_path = N15_ADAPTER_DIR / domain_name
        if not adapter_path.exists():
            log(f"  FATAL: {adapter_path} not found")
            return
        params = load_adapter(adapter_path)
        all_adapters[domain_name] = params
        log(f"    {domain_name}: loaded ({len(params)} keys)")

    # Load 4 existing capability adapters
    log("\n  Loading 4 existing capability adapters from capability_expert_taxonomy...")
    for cap_name in EXISTING_CAP_NAMES:
        adapter_path = CAP_ADAPTER_DIR / cap_name
        if not adapter_path.exists():
            log(f"  FATAL: {adapter_path} not found")
            return
        params = load_adapter(adapter_path)
        all_adapters[cap_name] = params
        log(f"    {cap_name}: loaded ({len(params)} keys)")

    # Train 6 new capability adapters
    log("\n  Training 6 new capability adapters...")
    train_results = {}
    for cap_name, config in NEW_CAPABILITIES.items():
        log(f"\n  --- Training: {cap_name} ---")
        zero_lora_params(model, seed=SEED * 1000 + hash(cap_name) % 10000)

        train_result = train_adapter(
            model, tokenizer, data_dirs[cap_name], cap_name, TRAIN_STEPS, SEED
        )
        train_results[cap_name] = train_result

        params = save_adapter(model, ADAPTERS_DIR / cap_name)
        all_adapters[cap_name] = params
        log(f"    Saved. Time: {train_result['train_time_s']}s, "
              f"converged: {train_result['converged']}")

    results["train_results"] = train_results

    # ------------------------------------------------------------------
    # Phase 4: Individual PPL for all 25
    # ------------------------------------------------------------------
    log("\n[Phase 4] Individual adapter PPL (all 25)...")
    individual_ppls = {}
    for name in all_names:
        if data_dirs.get(name) is None:
            continue
        zero_lora_params(model)
        apply_adapter_weights(model, all_adapters[name])
        mx.eval(model.parameters())
        ppl = compute_ppl(model, tokenizer, data_dirs[name])
        individual_ppls[name] = round(ppl, 4)
        base = base_ppls.get(name, float("inf"))
        imp = (base - ppl) / base * 100 if base < float("inf") else 0
        log(f"    {name}: {ppl:.4f} (base={base:.4f}, {imp:+.1f}%)")
    results["individual_ppls"] = individual_ppls

    # ------------------------------------------------------------------
    # Phase 5: N=15 composition (domains only) - reference baseline
    # ------------------------------------------------------------------
    log("\n[Phase 5] N=15 composition (domains only) - reference baseline...")
    domain_adapter_list = [all_adapters[d] for d in DOMAIN_NAMES]
    merged_15 = compose_adapters(domain_adapter_list)

    zero_lora_params(model)
    apply_adapter_weights(model, merged_15)
    mx.eval(model.parameters())

    composed_15_ppls = {}
    for name in DOMAIN_NAMES:
        if data_dirs.get(name) is None:
            continue
        ppl = compute_ppl(model, tokenizer, data_dirs[name])
        composed_15_ppls[name] = round(ppl, 4)
        log(f"    {name}: {ppl:.4f}")

    best_ind_15 = min(individual_ppls.get(d, float("inf")) for d in DOMAIN_NAMES)
    avg_composed_15 = sum(composed_15_ppls.values()) / len(composed_15_ppls) if composed_15_ppls else float("inf")
    ratio_15 = avg_composed_15 / best_ind_15 if best_ind_15 > 0 else float("inf")

    results["n15_composed_ppls"] = composed_15_ppls
    results["n15_best_individual"] = round(best_ind_15, 4)
    results["n15_avg_composed"] = round(avg_composed_15, 4)
    results["n15_composition_ratio"] = round(ratio_15, 4)
    log(f"\n  N=15: ratio = {ratio_15:.4f}x")

    # ------------------------------------------------------------------
    # Phase 6: N=25 composition (all 25)
    # ------------------------------------------------------------------
    log("\n[Phase 6] N=25 composition (all 25)...")
    all_adapter_list = [all_adapters[name] for name in all_names]
    merged_25 = compose_adapters(all_adapter_list)

    zero_lora_params(model)
    apply_adapter_weights(model, merged_25)
    mx.eval(model.parameters())

    composed_25_ppls = {}
    eval_names = [n for n in all_names if data_dirs.get(n) is not None]
    for name in eval_names:
        ppl = compute_ppl(model, tokenizer, data_dirs[name])
        composed_25_ppls[name] = round(ppl, 4)
        log(f"    {name}: {ppl:.4f}")

    best_ind_25 = min(individual_ppls.get(n, float("inf")) for n in all_names if n in individual_ppls)
    avg_composed_25 = sum(composed_25_ppls.values()) / len(composed_25_ppls) if composed_25_ppls else float("inf")
    ratio_25 = avg_composed_25 / best_ind_25 if best_ind_25 > 0 else float("inf")

    results["n25_composed_ppls"] = composed_25_ppls
    results["n25_best_individual"] = round(best_ind_25, 4)
    results["n25_avg_composed"] = round(avg_composed_25, 4)
    results["n25_composition_ratio"] = round(ratio_25, 4)
    log(f"\n  N=25: ratio = {ratio_25:.4f}x")

    # Also compute composed/base ratio (primary metric from N=15 review)
    composed_base_ratios = {}
    for name in eval_names:
        if name in base_ppls and base_ppls[name] > 0:
            composed_base_ratios[name] = round(composed_25_ppls[name] / base_ppls[name], 4)
    avg_composed_base_ratio = sum(composed_base_ratios.values()) / len(composed_base_ratios) if composed_base_ratios else float("inf")
    results["composed_base_ratios"] = composed_base_ratios
    results["avg_composed_base_ratio"] = round(avg_composed_base_ratio, 4)
    log(f"  Avg composed/base ratio: {avg_composed_base_ratio:.4f}")

    # ------------------------------------------------------------------
    # Phase 7: Cosines with categories (all 300 pairs)
    # ------------------------------------------------------------------
    n_pairs = N_total * (N_total - 1) // 2
    log(f"\n[Phase 7] Cosine similarity ({n_pairs} pairs)...")
    cosines = compute_cosines_with_categories(
        all_adapters, set(DOMAIN_NAMES), set(all_cap_names)
    )
    results["cosines"] = cosines

    # Category statistics
    dd_cos = [c["abs_cos"] for c in cosines if c["category"] == "domain-domain"]
    cc_cos = [c["abs_cos"] for c in cosines if c["category"] == "capability-capability"]
    cd_cos = [c["abs_cos"] for c in cosines if c["category"] == "capability-domain"]

    def stats(values, label):
        if not values:
            return {"mean": 0, "max": 0, "min": 0, "n": 0}
        s = {
            "mean": round(sum(values) / len(values), 6),
            "max": round(max(values), 6),
            "min": round(min(values), 6),
            "n": len(values),
        }
        log(f"  {label}: mean={s['mean']:.6f}, max={s['max']:.6f}, n={s['n']}")
        return s

    dd_stats = stats(dd_cos, "domain-domain")
    cc_stats = stats(cc_cos, "capability-capability")
    cd_stats = stats(cd_cos, "capability-domain")

    results["dd_stats"] = dd_stats
    results["cc_stats"] = cc_stats
    results["cd_stats"] = cd_stats

    mean_cos_all = sum(c["abs_cos"] for c in cosines) / len(cosines) if cosines else 0
    results["mean_cos_all"] = round(mean_cos_all, 6)
    log(f"  Overall mean |cos|: {mean_cos_all:.6f}")

    # Top-5 highest cosine pairs
    sorted_cos = sorted(cosines, key=lambda c: c["abs_cos"], reverse=True)
    log(f"\n  Top 5 most similar pairs:")
    for c in sorted_cos[:5]:
        log(f"    {c['pair']}: {c['abs_cos']:.6f} [{c['category']}]")

    # ------------------------------------------------------------------
    # Phase 8: Domain degradation (N=15 -> N=25 for domains)
    # ------------------------------------------------------------------
    log("\n[Phase 8] Domain degradation (N=15 -> N=25)...")
    degradation_domain = {}
    for name in DOMAIN_NAMES:
        if name not in composed_15_ppls or name not in composed_25_ppls:
            continue
        ppl_15 = composed_15_ppls[name]
        ppl_25 = composed_25_ppls[name]
        pct_change = (ppl_25 - ppl_15) / ppl_15 * 100
        degradation_domain[name] = {
            "ppl_n15": ppl_15,
            "ppl_n25": ppl_25,
            "pct_change": round(pct_change, 2),
        }
        status = "OK" if pct_change <= 15.0 else "HIGH"
        log(f"    {name}: N=15={ppl_15:.4f} -> N=25={ppl_25:.4f} ({pct_change:+.2f}%) [{status}]")
    results["degradation_domain"] = degradation_domain

    # ------------------------------------------------------------------
    # Kill criteria assessment
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    # K1: composition ratio N=25 <= 5x
    k1_pass = ratio_25 <= 5.0
    # Also compute ratio-of-ratios vs N=15
    n15_reference_ratio = 6.1212  # from N=15 results
    ratio_of_ratios = ratio_25 / n15_reference_ratio if n15_reference_ratio > 0 else float("inf")

    log(f"\n  K1 (composition ratio N=25 <= 5x):")
    log(f"    N=15 reference ratio: {n15_reference_ratio:.4f}x")
    log(f"    N=25 ratio: {ratio_25:.4f}x")
    log(f"    Threshold: 5.0x")
    log(f"    Ratio of ratios (N=25/N=15): {ratio_of_ratios:.4f}x")

    # Also check absolute: if composed PPL < base for majority of domains
    n_below_base = sum(1 for n in eval_names if n in base_ppls and composed_25_ppls[n] < base_ppls[n])
    log(f"    Domains with composed < base: {n_below_base}/{len(eval_names)}")
    log(f"    -> {'PASS' if k1_pass else 'KILL'}")

    # K2: cross-type cosine (capability-domain) <= 0.01
    cd_max = cd_stats["max"]
    cd_mean = cd_stats["mean"]
    k2_pass = cd_max <= 0.01
    log(f"\n  K2 (cross-type cosine cap-domain max <= 0.01):")
    log(f"    Cap-domain mean: {cd_mean:.6f}")
    log(f"    Cap-domain max: {cd_max:.6f}")
    log(f"    Threshold: 0.01")
    log(f"    -> {'PASS' if k2_pass else 'KILL'}")

    results["k1_pass"] = k1_pass
    results["k1_ratio_25"] = round(ratio_25, 4)
    results["k1_ratio_of_ratios"] = round(ratio_of_ratios, 4)
    results["k2_pass"] = k2_pass
    results["k2_cd_max"] = cd_max
    results["k2_cd_mean"] = cd_mean
    results["n_below_base"] = n_below_base

    verdict = "SUPPORTED" if (k1_pass and k2_pass) else "KILLED"
    results["verdict"] = verdict

    total_time = time.time() - t_global
    results["total_time_s"] = round(total_time, 1)
    results["total_time_min"] = round(total_time / 60, 1)

    log(f"\n  VERDICT: {verdict}")
    log(f"  Total time: {total_time/60:.1f} min")

    # Summary table
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  N=15 composition ratio: {ratio_15:.4f}x (this run, domains only)")
    log(f"  N=15 reference ratio:   {n15_reference_ratio:.4f}x (prior experiment)")
    log(f"  N=25 composition ratio: {ratio_25:.4f}x")
    log(f"  Ratio scaling (25/15):  {ratio_of_ratios:.4f}x")
    log(f"  Avg composed/base:      {avg_composed_base_ratio:.4f}")
    log(f"")
    log(f"  Cosine by type:")
    log(f"    domain-domain:      mean={dd_stats['mean']:.6f} max={dd_stats['max']:.6f} (n={dd_stats['n']})")
    log(f"    cap-cap:            mean={cc_stats['mean']:.6f} max={cc_stats['max']:.6f} (n={cc_stats['n']})")
    log(f"    cap-domain:         mean={cd_stats['mean']:.6f} max={cd_stats['max']:.6f} (n={cd_stats['n']})")
    log(f"    overall:            mean={mean_cos_all:.6f}")
    log(f"")
    log(f"  {n_below_base}/{len(eval_names)} domains have composed PPL < base PPL")

    n_converged = sum(1 for r in train_results.values() if r["converged"])
    log(f"  New caps converged: {n_converged}/{len(NEW_CAPABILITIES)}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
