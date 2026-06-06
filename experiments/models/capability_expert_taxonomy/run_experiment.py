#!/usr/bin/env python3
"""
Capability Expert Taxonomy Experiment

Tests whether capabilities beyond domain knowledge (reasoning, instruction
following, conciseness, multilingual, safety) can be trained as orthogonal
LoRA adapters and composed with domain experts on BitNet-2B-4T.

Kill criteria:
  K1: fewer than 3 capability types show orthogonal composition
      (mean |cos| > 0.01 between capability LoRAs)
  K2: capability experts interfere with each other
      (any capability-capability pair cos > 0.01)

Uses the same BitNet-2B-4T infrastructure from the prior domain composition
experiment. Reuses the 5 saved domain adapters for cross-type comparison.

The cos=0.01 threshold is strict but motivated by the domain adapter baseline:
BitNet domain adapters achieved mean |cos|=0.001 (10x margin). If capabilities
are as orthogonal as domains, they should also pass this threshold.
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
LORA_SCALE = 20.0
TRAIN_ITERS = 200
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
VAL_BATCHES = 25

EXPERIMENT_DIR = Path(__file__).parent
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse domain adapters from the prior experiment
DOMAIN_ADAPTERS_DIR = EXPERIMENT_DIR.parent / "bitnet_2b_real_composition" / "adapters"
DOMAIN_DATA_DIR = EXPERIMENT_DIR.parent / "bitnet_2b_real_composition" / "data"

# 5 capability types — each defined by a distinctive data distribution
# that captures a *behavioral mode* rather than a knowledge domain
CAPABILITIES = {
    "reasoning": {
        "description": "Chain-of-thought step-by-step reasoning",
        "hf_dataset": "gsm8k",
        "hf_subset": "main",
        "text_key": "answer",  # GSM8K answers contain step-by-step reasoning
        "format": "cot",  # wrap in <think> tags to teach reasoning format
        "max_samples_train": 500,
        "max_samples_val": 50,
    },
    "instruction": {
        "description": "Instruction following (structured input->output)",
        "hf_dataset": "tatsu-lab/alpaca",
        "text_key": "output",
        "format": "instruct",  # prepend "Instruction: ... Response: ..."
        "max_samples_train": 500,
        "max_samples_val": 50,
    },
    "conciseness": {
        "description": "Short, direct, concise answers",
        "hf_dataset": "web_questions",
        "text_key": "answers",  # WebQuestions has short factoid answers
        "format": "concise",
        "max_samples_train": 500,
        "max_samples_val": 50,
    },
    "multilingual": {
        "description": "Non-English text generation (German)",
        "hf_dataset": "oscar-corpus/OSCAR-2301",
        "hf_subset": "de",
        "text_key": "text",
        "format": "plain",
        "max_samples_train": 500,
        "max_samples_val": 50,
    },
    "safety": {
        "description": "Safe, helpful, harmless responses",
        "hf_dataset": "Anthropic/hh-rlhf",
        "text_key": "chosen",  # The chosen (safe) responses
        "format": "plain",
        "max_samples_train": 500,
        "max_samples_val": 50,
    },
}

DOMAIN_NAMES = ["python", "math", "medical", "legal", "creative"]


# ===========================================================================
# Model utilities (reused from bitnet_2b_real_composition)
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
    """Replace all BitLinear layers with standard nn.Linear."""
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
                lora_layer = LoRALinear.from_base(module, r=rank, scale=scale, dropout=0.0)
                lora_updates.append((key, lora_layer))
                count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    print(f"  Applied LoRA (r={rank}) to {count} linear layers")
    return model


def get_lora_params(model):
    """Extract LoRA parameters as a flat dict."""
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
    """Reset all LoRA params to zero."""
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
# Data preparation for capability types
# ===========================================================================
def prepare_capability_data(cap_name: str, cap_config: dict) -> Path:
    """Download HF dataset and write train.jsonl / valid.jsonl for a capability."""
    from datasets import load_dataset as hf_load

    data_dir = DATA_DIR / cap_name
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"

    if train_path.exists() and valid_path.exists():
        print(f"  Data for {cap_name} already exists, skipping download")
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {cap_config['hf_dataset']}...")

    kwargs = {"trust_remote_code": True}
    if "hf_subset" in cap_config:
        kwargs["name"] = cap_config["hf_subset"]

    try:
        ds = hf_load(cap_config["hf_dataset"], **kwargs, streaming=True)
    except Exception:
        ds = hf_load(cap_config["hf_dataset"], **kwargs)

    # Handle streaming vs non-streaming
    if hasattr(ds, "keys"):
        if "train" in ds:
            split_data = ds["train"]
        else:
            split_name = list(ds.keys())[0]
            split_data = ds[split_name]
    else:
        split_data = ds

    text_key = cap_config["text_key"]
    fmt = cap_config.get("format", "plain")
    max_train = cap_config["max_samples_train"]
    max_val = cap_config["max_samples_val"]

    texts = []
    for row in split_data:
        try:
            if text_key == "answers" and isinstance(row.get(text_key), list):
                # WebQuestions has list of answer strings
                raw = " | ".join(str(a) for a in row[text_key])
            elif text_key in row:
                raw = row[text_key]
            else:
                # Try alternatives
                for alt in ["text", "content", "output", "answer", "response"]:
                    if alt in row:
                        raw = row[alt]
                        break
                else:
                    continue

            if not isinstance(raw, str):
                raw = str(raw)

            raw = raw.strip()
            if len(raw) < 20:
                continue

            # Format based on capability type
            if fmt == "cot":
                # Wrap in reasoning tags to teach step-by-step format
                text = f"<think>\n{raw}\n</think>"
            elif fmt == "instruct":
                # Instruction-response format
                instruction = row.get("instruction", "Follow the instructions.")
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{raw}"
            elif fmt == "concise":
                # Short QA format
                question = row.get("question", "")
                text = f"Q: {question}\nA: {raw}"
            else:
                text = raw

            texts.append(text)
        except Exception:
            continue

        if len(texts) >= max_train + max_val:
            break

    if len(texts) < max_val + 10:
        raise ValueError(f"Not enough samples for {cap_name}: got {len(texts)}")

    train_texts = texts[:max_train]
    val_texts = texts[max_train:max_train + max_val]

    with open(train_path, "w") as f:
        for t in train_texts:
            json.dump({"text": t}, f)
            f.write("\n")

    with open(valid_path, "w") as f:
        for t in val_texts:
            json.dump({"text": t}, f)
            f.write("\n")

    print(f"  {cap_name}: {len(train_texts)} train, {len(val_texts)} val")
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
# Orthogonality measurement
# ===========================================================================
def compute_pairwise_cosines(adapter_dict: dict):
    """Compute pairwise |cos| between all adapter pairs.

    Returns list of dicts with pair names and cosines, plus a category tag.
    """
    names = list(adapter_dict.keys())
    results = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            vi = mx.concatenate([v.reshape(-1) for v in adapter_dict[names[i]].values()])
            vj = mx.concatenate([v.reshape(-1) for v in adapter_dict[names[j]].values()])
            cos = mx.abs(
                mx.sum(vi * vj) / (mx.sqrt(mx.sum(vi**2)) * mx.sqrt(mx.sum(vj**2)))
            )
            mx.eval(cos)

            # Categorize: cap-cap, cap-domain, domain-domain
            ni, nj = names[i], names[j]
            cap_names = set(CAPABILITIES.keys())
            domain_names_set = set(DOMAIN_NAMES)

            if ni in cap_names and nj in cap_names:
                category = "capability-capability"
            elif ni in domain_names_set and nj in domain_names_set:
                category = "domain-domain"
            else:
                category = "capability-domain"

            results.append({
                "pair": f"{ni}-{nj}",
                "abs_cos": round(cos.item(), 6),
                "category": category,
            })

    return results


# ===========================================================================
# Main
# ===========================================================================
def main():
    results = {
        "experiment": "capability_expert_taxonomy",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "train_iters": TRAIN_ITERS,
        "capabilities": list(CAPABILITIES.keys()),
        "domains": DOMAIN_NAMES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=" * 70)
    print("Capability Expert Taxonomy Experiment")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 0: Load model
    # ------------------------------------------------------------------
    print("\n[Phase 0] Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    print("  Unpacking ternary weights...")
    model = replace_bitlinear_with_linear(model)

    # ------------------------------------------------------------------
    # Phase 1: Prepare capability data
    # ------------------------------------------------------------------
    print("\n[Phase 1] Preparing capability data...")
    cap_data_dirs = {}
    failed_caps = []

    for cap_name, config in CAPABILITIES.items():
        try:
            cap_data_dirs[cap_name] = prepare_capability_data(cap_name, config)
        except Exception as e:
            print(f"  WARNING: {cap_name} data failed: {e}")
            failed_caps.append(cap_name)

    if len(cap_data_dirs) < 3:
        print(f"  FATAL: Only {len(cap_data_dirs)} capabilities have data (need >=3)")
        results["error"] = f"Only {len(cap_data_dirs)} data sources succeeded"
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        return

    print(f"\n  Successfully prepared: {list(cap_data_dirs.keys())}")
    if failed_caps:
        print(f"  Failed: {failed_caps}")
        results["failed_capabilities"] = failed_caps

    # ------------------------------------------------------------------
    # Phase 2: Base PPL on capability data
    # ------------------------------------------------------------------
    print("\n[Phase 2] Computing base model PPL on capability datasets...")
    base_ppls = {}
    for cap_name, data_dir in cap_data_dirs.items():
        ppl = compute_ppl(model, tokenizer, data_dir)
        base_ppls[cap_name] = ppl
        print(f"  {cap_name}: base PPL = {ppl:.2f}")
    results["base_ppls"] = base_ppls

    # ------------------------------------------------------------------
    # Phase 3: Train capability adapters
    # ------------------------------------------------------------------
    print("\n[Phase 3] Training capability adapters...")

    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"  Trainable LoRA parameters: {trainable:,}")

    # Verify gradients
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
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        return

    cap_adapter_params = {}
    cap_train_results = {}

    for cap_name, data_dir in cap_data_dirs.items():
        print(f"\n  --- Training {cap_name} adapter ---")

        zero_lora_params(model)

        # Tokenize
        train_texts = []
        with open(data_dir / "train.jsonl") as f:
            for line in f:
                train_texts.append(json.loads(line)["text"])

        train_tokens = []
        for text in train_texts:
            toks = tokenizer.encode(text)
            if len(toks) > 2:
                train_tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))

        print(f"  {len(train_tokens)} training sequences")

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

        cap_train_results[cap_name] = {
            "train_time_s": round(train_time, 1),
            "first_50_avg_loss": round(first_50, 4),
            "last_50_avg_loss": round(last_50, 4),
            "converged": converged,
        }

        save_adapter(model, ADAPTERS_DIR / cap_name)
        cap_adapter_params[cap_name] = get_lora_params(model)

    results["train_results"] = cap_train_results

    # ------------------------------------------------------------------
    # Phase 4: Individual capability adapter PPL
    # ------------------------------------------------------------------
    print("\n[Phase 4] Individual capability adapter PPL...")
    cap_ppls = {}
    loaded_cap_adapters = {}

    for cap_name in cap_data_dirs:
        adapter_path = ADAPTERS_DIR / cap_name
        params = load_adapter(adapter_path)
        loaded_cap_adapters[cap_name] = params

        zero_lora_params(model)
        apply_adapter_weights(model, params, scale=1.0)
        mx.eval(model.parameters())

        ppl = compute_ppl(model, tokenizer, cap_data_dirs[cap_name])
        cap_ppls[cap_name] = ppl
        base = base_ppls[cap_name]
        imp = (base - ppl) / base * 100
        print(f"  {cap_name}: PPL={ppl:.2f} (base={base:.2f}, {imp:+.1f}%)")

    results["capability_ppls"] = cap_ppls

    # ------------------------------------------------------------------
    # Phase 5: Load domain adapters from prior experiment
    # ------------------------------------------------------------------
    print("\n[Phase 5] Loading domain adapters from bitnet_2b_real_composition...")
    loaded_domain_adapters = {}

    if DOMAIN_ADAPTERS_DIR.exists():
        for domain_name in DOMAIN_NAMES:
            adapter_path = DOMAIN_ADAPTERS_DIR / domain_name
            if adapter_path.exists():
                try:
                    params = load_adapter(adapter_path)
                    loaded_domain_adapters[domain_name] = params
                    print(f"  Loaded {domain_name} adapter ({len(params)} tensors)")
                except Exception as e:
                    print(f"  WARNING: Failed to load {domain_name}: {e}")
    else:
        print("  WARNING: Domain adapter directory not found!")

    results["n_domain_adapters_loaded"] = len(loaded_domain_adapters)

    # ------------------------------------------------------------------
    # Phase 6: Orthogonality analysis (the core measurement)
    # ------------------------------------------------------------------
    print("\n[Phase 6] Pairwise orthogonality analysis...")

    # Combine all adapters
    all_adapters = {}
    all_adapters.update(loaded_cap_adapters)
    all_adapters.update(loaded_domain_adapters)

    cosines = compute_pairwise_cosines(all_adapters)

    # Separate by category
    cap_cap = [c for c in cosines if c["category"] == "capability-capability"]
    cap_domain = [c for c in cosines if c["category"] == "capability-domain"]
    domain_domain = [c for c in cosines if c["category"] == "domain-domain"]

    print("\n  --- Capability-Capability ---")
    for c in sorted(cap_cap, key=lambda x: x["abs_cos"], reverse=True):
        print(f"    {c['pair']}: |cos| = {c['abs_cos']:.6f}")

    print("\n  --- Capability-Domain ---")
    for c in sorted(cap_domain, key=lambda x: x["abs_cos"], reverse=True)[:15]:
        print(f"    {c['pair']}: |cos| = {c['abs_cos']:.6f}")
    if len(cap_domain) > 15:
        print(f"    ... ({len(cap_domain) - 15} more pairs)")

    print("\n  --- Domain-Domain ---")
    for c in sorted(domain_domain, key=lambda x: x["abs_cos"], reverse=True):
        print(f"    {c['pair']}: |cos| = {c['abs_cos']:.6f}")

    # Statistics
    def stats(pairs):
        if not pairs:
            return {"mean": None, "max": None, "min": None, "n": 0}
        vals = [p["abs_cos"] for p in pairs]
        return {
            "mean": round(sum(vals) / len(vals), 6),
            "max": round(max(vals), 6),
            "min": round(min(vals), 6),
            "n": len(vals),
        }

    cap_cap_stats = stats(cap_cap)
    cap_domain_stats = stats(cap_domain)
    domain_domain_stats = stats(domain_domain)

    print(f"\n  Cap-Cap:    mean |cos| = {cap_cap_stats['mean']}, max = {cap_cap_stats['max']} (n={cap_cap_stats['n']})")
    print(f"  Cap-Domain: mean |cos| = {cap_domain_stats['mean']}, max = {cap_domain_stats['max']} (n={cap_domain_stats['n']})")
    print(f"  Domain-Dom: mean |cos| = {domain_domain_stats['mean']}, max = {domain_domain_stats['max']} (n={domain_domain_stats['n']})")

    results["cosines"] = cosines
    results["cap_cap_stats"] = cap_cap_stats
    results["cap_domain_stats"] = cap_domain_stats
    results["domain_domain_stats"] = domain_domain_stats

    # ------------------------------------------------------------------
    # Phase 7: Composition test (all capabilities + all domains)
    # ------------------------------------------------------------------
    print("\n[Phase 7] Composition test (all adapters, 1/N scaling)...")

    all_adapter_list = list(all_adapters.values())
    N = len(all_adapter_list)
    scale = 1.0 / N

    # Compose all adapters
    merged = {}
    for key in all_adapter_list[0].keys():
        stacked = mx.stack([a[key] for a in all_adapter_list])
        merged[key] = mx.sum(stacked, axis=0) * scale

    zero_lora_params(model)
    apply_adapter_weights(model, merged)
    mx.eval(model.parameters())

    composed_ppls = {}
    # Eval on capability data
    for cap_name, data_dir in cap_data_dirs.items():
        ppl = compute_ppl(model, tokenizer, data_dir)
        composed_ppls[cap_name] = ppl
        base = base_ppls[cap_name]
        imp = (base - ppl) / base * 100
        print(f"  {cap_name}: composed PPL = {ppl:.2f} (base={base:.2f}, {imp:+.1f}%)")

    # Eval on domain data (if available)
    for domain_name in loaded_domain_adapters:
        domain_data = DOMAIN_DATA_DIR / domain_name
        if domain_data.exists():
            ppl = compute_ppl(model, tokenizer, domain_data)
            composed_ppls[f"domain_{domain_name}"] = ppl
            print(f"  domain_{domain_name}: composed PPL = {ppl:.2f}")

    results["composed_ppls"] = composed_ppls

    # ------------------------------------------------------------------
    # Phase 8: Kill criteria assessment
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("KILL CRITERIA ASSESSMENT")
    print("=" * 70)

    # K1: fewer than 3 capability types show orthogonal composition
    # "orthogonal composition" = mean |cos| with other capabilities < 0.01
    n_orthogonal_caps = 0
    cap_names_list = list(loaded_cap_adapters.keys())
    per_cap_mean_cos = {}

    for cap_name in cap_names_list:
        # Get all pairs involving this capability
        cap_pairs = [c for c in cap_cap if cap_name in c["pair"]]
        if cap_pairs:
            mean_cos = sum(c["abs_cos"] for c in cap_pairs) / len(cap_pairs)
            per_cap_mean_cos[cap_name] = mean_cos
            if mean_cos < 0.01:
                n_orthogonal_caps += 1

    k1_pass = n_orthogonal_caps >= 3
    results["n_orthogonal_caps"] = n_orthogonal_caps
    results["per_cap_mean_cos"] = per_cap_mean_cos
    results["k1_pass"] = k1_pass

    print(f"\n  K1: {n_orthogonal_caps}/{len(cap_names_list)} capability types have mean |cos| < 0.01")
    for cap, mc in per_cap_mean_cos.items():
        status = "PASS" if mc < 0.01 else "FAIL"
        print(f"    {cap}: mean |cos| = {mc:.6f} [{status}]")
    print(f"  K1 verdict: {'PASS' if k1_pass else 'KILL'} (threshold: >=3 orthogonal)")

    # K2: any capability-capability pair cos > 0.01
    max_cap_cap_cos = max(c["abs_cos"] for c in cap_cap) if cap_cap else 0
    k2_pass = max_cap_cap_cos <= 0.01
    results["max_cap_cap_cos"] = max_cap_cap_cos
    results["k2_pass"] = k2_pass

    print(f"\n  K2: max capability-capability |cos| = {max_cap_cap_cos:.6f}")
    print(f"  K2 verdict: {'PASS' if k2_pass else 'KILL'} (threshold: max <= 0.01)")

    # Cross-type analysis
    if cap_domain_stats["mean"] is not None:
        print(f"\n  Cross-type analysis:")
        print(f"    Cap-Cap mean:     {cap_cap_stats['mean']:.6f}")
        print(f"    Cap-Domain mean:  {cap_domain_stats['mean']:.6f}")
        print(f"    Domain-Domain:    {domain_domain_stats['mean']:.6f}")
        ratio = cap_cap_stats['mean'] / domain_domain_stats['mean'] if domain_domain_stats['mean'] > 0 else float("inf")
        print(f"    Cap-Cap / Domain-Domain ratio: {ratio:.2f}x")
        results["cross_type_ratio"] = round(ratio, 4)

    # Overall verdict
    all_pass = k1_pass and k2_pass
    results["verdict"] = "SUPPORTED" if all_pass else "KILLED"
    print(f"\n  OVERALL VERDICT: {results['verdict']}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
