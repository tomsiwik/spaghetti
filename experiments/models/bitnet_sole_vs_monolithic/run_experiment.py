#!/usr/bin/env python3
"""
BitNet-SOLE vs Monolithic: The Fundamental Value-Prop Test

Compares two approaches to multi-domain adaptation on BitNet-2B-4T:
  (A) SOLE: 5 independent ternary LoRA experts (QAT+STE), routed per-domain
  (B) Monolithic: 1 ternary LoRA trained on shuffled union of all domain data

Both use rank-16 ternary adapters with identical training hyperparameters.
Budget matching: same rank per adapter, same total training steps.
  - SOLE: 5 x 400 steps = 2000 total gradient steps
  - Monolithic: 2000 steps on shuffled union data (sees all domains equally)

Kill criteria:
  K1: monolithic ternary LoRA beats composed SOLE on >80% of per-domain metrics
      (i.e., monolithic wins 4/5 or 5/5 routed domain PPLs)

Evaluation conditions:
  - Base (no adapter): baseline PPL
  - SOLE routed: each domain evaluated with its own expert adapter
  - SOLE composed (1/N): all 5 adapters composed, evaluated on each domain
  - Monolithic: single adapter trained on all data, evaluated on each domain
  - Monolithic sequential: train on domains one at a time (forgetting test)

Platform: Apple Silicon MLX, $0 compute.
Builds on proven pipelines:
  - bitnet_2b_real_composition (proven)
  - bitnet_ternary_convergence (supported)
"""

import json
import math
import os
import sys
import time
import random
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
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 128  # match ternary_convergence for Apple Silicon tractability
LEARNING_RATE = 1e-4
VAL_BATCHES = 50  # more val samples for reliable comparison

STEPS_PER_DOMAIN = 400  # from ternary_convergence (proven tractable)
MONOLITHIC_STEPS = 2000  # 5 x 400 = same total gradient updates

EXPERIMENT_DIR = Path(__file__).parent
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# 5 domains - same as ternary_convergence for comparable results
DOMAINS = {
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


# ===========================================================================
# Ternary weight unpacking (from proven pipeline)
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
    """Replace all BitLinear layers with standard nn.Linear for training."""
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
# Ternary LoRA with STE (from proven pipeline)
# ===========================================================================
class TernaryLoRALinear(nn.Module):
    """LoRA layer with STE ternary quantization of A/B matrices."""

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
        """Ternary quantization with Straight-Through Estimator."""
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
    """Apply ternary LoRA to all linear layers in transformer blocks."""
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        lora_updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora_layer = TernaryLoRALinear(module, r=rank, scale=scale)
                lora_updates.append((key, lora_layer))
                count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    print(f"  Applied Ternary LoRA (r={rank}) to {count} layers")
    return model


def remove_lora(model):
    """Remove all LoRA wrappers, restoring base nn.Linear."""
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                updates.append((key, module.linear))
        if updates:
            layer.update_modules(tree_unflatten(updates))
    return model


def get_lora_params(model):
    """Extract LoRA parameters (deep copy)."""
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
    """Reset all LoRA params."""
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                in_dims = module.lora_a.shape[0]
                r = module.lora_a.shape[1]
                s = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(low=-s, high=s, shape=module.lora_a.shape)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


# ===========================================================================
# Data preparation
# ===========================================================================
def prepare_domain_data(domain_name: str, domain_config: dict) -> Path:
    """Download HF dataset and write train.jsonl / valid.jsonl."""
    from datasets import load_dataset as hf_load

    data_dir = DATA_DIR / domain_name
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
    total_needed = max_train + max_val

    texts = []
    for row in split_data:
        t = row[text_key]
        if isinstance(t, str) and len(t.strip()) > 20:
            texts.append(t.strip())
        if len(texts) >= total_needed * 2:
            break

    if len(texts) < 100:
        raise ValueError(f"Not enough samples for {domain_name}: got {len(texts)}")

    if len(texts) >= total_needed:
        train_texts = texts[:max_train]
        val_texts = texts[max_train:max_train + max_val]
    else:
        n_train = int(len(texts) * 0.85)
        train_texts = texts[:n_train]
        val_texts = texts[n_train:]

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


def prepare_monolithic_data(data_dirs: dict) -> Path:
    """Create shuffled union of all domain training data for monolithic training."""
    mono_dir = DATA_DIR / "monolithic"
    train_path = mono_dir / "train.jsonl"

    if train_path.exists():
        with open(train_path) as f:
            n = sum(1 for _ in f)
        if n > 100:
            print(f"  Monolithic data exists ({n} samples)")
            return mono_dir

    mono_dir.mkdir(parents=True, exist_ok=True)

    all_texts = []
    for domain_name, data_dir in data_dirs.items():
        with open(data_dir / "train.jsonl") as f:
            for line in f:
                row = json.loads(line)
                all_texts.append({"text": row["text"], "domain": domain_name})

    # Shuffle with fixed seed for reproducibility
    rng = random.Random(42)
    rng.shuffle(all_texts)

    with open(train_path, "w") as f:
        for item in all_texts:
            json.dump({"text": item["text"]}, f)
            f.write("\n")

    # Track domain distribution
    domain_counts = {}
    for item in all_texts:
        domain_counts[item["domain"]] = domain_counts.get(item["domain"], 0) + 1

    print(f"  Monolithic: {len(all_texts)} total samples")
    for d, c in sorted(domain_counts.items()):
        print(f"    {d}: {c} ({100*c/len(all_texts):.1f}%)")

    return mono_dir


# ===========================================================================
# PPL evaluation
# ===========================================================================
def compute_ppl(model, tokenizer, data_path: Path, max_batches: int = 50):
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
# Training
# ===========================================================================
def train_adapter(model, tokenizer, data_dir, n_steps, label, val_dirs=None):
    """Train a single adapter. Returns training metrics.

    Args:
        val_dirs: dict of {domain_name: data_dir} for periodic validation.
                  If None, validates on data_dir only.
    """
    train_texts = []
    with open(data_dir / "train.jsonl") as f:
        for line in f:
            train_texts.append(json.loads(line)["text"])

    train_tokens = []
    for text in train_texts:
        toks = tokenizer.encode(text)
        if len(toks) > 2:
            train_tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))

    print(f"  {len(train_tokens)} training sequences, {n_steps} steps")

    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    t_start = time.time()
    losses = []

    for step in range(n_steps):
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
            print(f"    [{label}] Step {step+1}/{n_steps}: loss={loss_val:.4f} (avg50={avg:.4f})")

    train_time = time.time() - t_start
    first_50 = sum(losses[:50]) / 50
    last_50 = sum(losses[-50:]) / 50
    converged = last_50 < first_50 * 0.95

    print(f"  [{label}] Done in {train_time:.1f}s. Loss: {first_50:.4f} -> {last_50:.4f} "
          f"({'converged' if converged else 'NOT converged'})")

    return {
        "label": label,
        "n_steps": n_steps,
        "train_time_s": round(train_time, 1),
        "first_50_avg_loss": round(first_50, 4),
        "last_50_avg_loss": round(last_50, 4),
        "converged": converged,
    }


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


def compute_cosines(adapters_dict):
    """Compute pairwise |cos| between adapter parameter vectors."""
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
# Main
# ===========================================================================
def main():
    results = {
        "experiment": "bitnet_sole_vs_monolithic",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "steps_per_domain": STEPS_PER_DOMAIN,
        "monolithic_steps": MONOLITHIC_STEPS,
        "domains": list(DOMAINS.keys()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "design": "same_rank_budget_matched_steps",
    }

    print("=" * 70)
    print("BitNet-SOLE vs Monolithic: The Fundamental Value-Prop Test")
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
    model = replace_bitlinear_with_linear(model)
    unpack_time = time.time() - t1
    print(f"  Unpacked in {unpack_time:.1f}s")

    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"  Total parameters: {total_params:,}")
    results["total_params"] = total_params
    results["load_time_s"] = round(load_time + unpack_time, 1)

    # ==================================================================
    # Phase 1: Prepare data
    # ==================================================================
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

    # Prepare monolithic (shuffled union) data
    mono_data_dir = prepare_monolithic_data(data_dirs)

    # ==================================================================
    # Phase 2: Base model PPL
    # ==================================================================
    print("\n[Phase 2] Computing base model PPL...")
    base_ppls = {}
    for domain_name, data_dir in data_dirs.items():
        ppl = compute_ppl(model, tokenizer, data_dir)
        base_ppls[domain_name] = round(ppl, 4)
        print(f"  {domain_name}: base PPL = {ppl:.2f}")
    results["base_ppls"] = base_ppls

    # ==================================================================
    # Phase 3: Train 5 domain expert adapters (SOLE)
    # ==================================================================
    print("\n[Phase 3] Training 5 SOLE domain expert adapters (ternary LoRA)...")
    model = apply_ternary_lora(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Freeze base, unfreeze LoRA
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"  Trainable LoRA parameters: {trainable:,}")
    results["trainable_params_per_expert"] = trainable

    # Verify gradients
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
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        return

    sole_adapters = {}
    sole_train_results = {}
    sole_total_time = 0

    for domain_name, data_dir in data_dirs.items():
        print(f"\n  --- Training SOLE expert: {domain_name} ---")
        zero_lora_params(model)

        train_result = train_adapter(
            model, tokenizer, data_dir,
            n_steps=STEPS_PER_DOMAIN,
            label=f"SOLE-{domain_name}",
        )
        sole_train_results[domain_name] = train_result
        sole_total_time += train_result["train_time_s"]

        save_adapter(model, ADAPTERS_DIR / "sole" / domain_name)
        sole_adapters[domain_name] = get_lora_params(model)

    results["sole_train_results"] = sole_train_results
    results["sole_total_time_s"] = round(sole_total_time, 1)

    # ==================================================================
    # Phase 4: Train monolithic adapter (shuffled union, same total steps)
    # ==================================================================
    print("\n\n[Phase 4] Training monolithic adapter (shuffled union, 2000 steps)...")
    zero_lora_params(model)

    mono_train_result = train_adapter(
        model, tokenizer, mono_data_dir,
        n_steps=MONOLITHIC_STEPS,
        label="MONO-shuffled",
    )
    results["mono_train_result"] = mono_train_result

    save_adapter(model, ADAPTERS_DIR / "monolithic")
    mono_adapter = get_lora_params(model)

    # ==================================================================
    # Phase 5: Train monolithic sequential (forgetting test)
    # ==================================================================
    print("\n\n[Phase 5] Training monolithic sequential (forgetting test)...")
    zero_lora_params(model)

    seq_total_time = 0
    for domain_name, data_dir in data_dirs.items():
        print(f"\n  --- Sequential training: {domain_name} ---")
        seq_result = train_adapter(
            model, tokenizer, data_dir,
            n_steps=STEPS_PER_DOMAIN,
            label=f"SEQ-{domain_name}",
        )
        seq_total_time += seq_result["train_time_s"]

    # After sequential training on all 5 domains, the adapter has been
    # trained on creative last. Save this state.
    save_adapter(model, ADAPTERS_DIR / "sequential")
    seq_adapter = get_lora_params(model)
    results["seq_total_time_s"] = round(seq_total_time, 1)

    # ==================================================================
    # Phase 6: Evaluate all conditions
    # ==================================================================
    print("\n\n[Phase 6] Evaluating all conditions on per-domain PPL...")

    # 6a: SOLE routed (each domain uses its own expert)
    print("\n  --- SOLE Routed ---")
    sole_routed_ppls = {}
    for domain_name, data_dir in data_dirs.items():
        zero_lora_params(model)
        adapter = load_adapter(ADAPTERS_DIR / "sole" / domain_name)
        apply_adapter_weights(model, adapter)
        mx.eval(model.parameters())

        ppl = compute_ppl(model, tokenizer, data_dir)
        sole_routed_ppls[domain_name] = round(ppl, 4)
        imp = (base_ppls[domain_name] - ppl) / base_ppls[domain_name] * 100
        print(f"  {domain_name}: PPL={ppl:.2f} (base={base_ppls[domain_name]:.2f}, {imp:+.1f}%)")
    results["sole_routed_ppls"] = sole_routed_ppls

    # 6b: SOLE composed (1/N scaling)
    print("\n  --- SOLE Composed (1/N) ---")
    loaded_adapters = {}
    for domain_name in DOMAINS:
        loaded_adapters[domain_name] = load_adapter(ADAPTERS_DIR / "sole" / domain_name)
    composed = compose_adapters(list(loaded_adapters.values()))

    zero_lora_params(model)
    apply_adapter_weights(model, composed)
    mx.eval(model.parameters())

    sole_composed_ppls = {}
    for domain_name, data_dir in data_dirs.items():
        ppl = compute_ppl(model, tokenizer, data_dir)
        sole_composed_ppls[domain_name] = round(ppl, 4)
        imp = (base_ppls[domain_name] - ppl) / base_ppls[domain_name] * 100
        print(f"  {domain_name}: PPL={ppl:.2f} (base={base_ppls[domain_name]:.2f}, {imp:+.1f}%)")
    results["sole_composed_ppls"] = sole_composed_ppls

    # 6c: Monolithic (shuffled)
    print("\n  --- Monolithic (shuffled) ---")
    zero_lora_params(model)
    mono_params = load_adapter(ADAPTERS_DIR / "monolithic")
    apply_adapter_weights(model, mono_params)
    mx.eval(model.parameters())

    mono_ppls = {}
    for domain_name, data_dir in data_dirs.items():
        ppl = compute_ppl(model, tokenizer, data_dir)
        mono_ppls[domain_name] = round(ppl, 4)
        imp = (base_ppls[domain_name] - ppl) / base_ppls[domain_name] * 100
        print(f"  {domain_name}: PPL={ppl:.2f} (base={base_ppls[domain_name]:.2f}, {imp:+.1f}%)")
    results["mono_ppls"] = mono_ppls

    # 6d: Monolithic sequential
    print("\n  --- Monolithic Sequential ---")
    zero_lora_params(model)
    seq_params = load_adapter(ADAPTERS_DIR / "sequential")
    apply_adapter_weights(model, seq_params)
    mx.eval(model.parameters())

    seq_ppls = {}
    for domain_name, data_dir in data_dirs.items():
        ppl = compute_ppl(model, tokenizer, data_dir)
        seq_ppls[domain_name] = round(ppl, 4)
        imp = (base_ppls[domain_name] - ppl) / base_ppls[domain_name] * 100
        print(f"  {domain_name}: PPL={ppl:.2f} (base={base_ppls[domain_name]:.2f}, {imp:+.1f}%)")
    results["seq_ppls"] = seq_ppls

    # ==================================================================
    # Phase 7: Orthogonality
    # ==================================================================
    print("\n[Phase 7] Adapter orthogonality...")
    cosines, mean_cos = compute_cosines(loaded_adapters)
    results["cosines"] = cosines
    results["mean_abs_cos"] = mean_cos
    print(f"  Mean |cos|: {mean_cos:.6f}")
    for c in cosines:
        print(f"    {c['pair']}: {c['abs_cos']:.6f}")

    # ==================================================================
    # Phase 8: Analysis and Kill Criteria
    # ==================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # K1: Per-domain comparison: SOLE routed vs Monolithic
    print("\n  Per-domain comparison (SOLE routed vs Monolithic shuffled):")
    sole_wins = 0
    mono_wins = 0
    domain_results = {}
    for d in DOMAINS:
        sole_ppl = sole_routed_ppls[d]
        mono_ppl = mono_ppls[d]
        winner = "SOLE" if sole_ppl <= mono_ppl else "MONO"
        gap = (sole_ppl - mono_ppl) / mono_ppl * 100
        if winner == "SOLE":
            sole_wins += 1
        else:
            mono_wins += 1
        domain_results[d] = {
            "sole_routed_ppl": sole_ppl,
            "mono_ppl": mono_ppl,
            "gap_pct": round(gap, 2),
            "winner": winner,
        }
        print(f"    {d}: SOLE={sole_ppl:.2f} vs MONO={mono_ppl:.2f} "
              f"({gap:+.1f}%) -> {winner}")
    results["domain_comparison"] = domain_results
    results["sole_wins"] = sole_wins
    results["mono_wins"] = mono_wins

    # K1 assessment
    k1_pass = mono_wins <= 4  # mono must NOT win >80% (i.e., not 4/5 or 5/5)
    # More precisely: kill if mono wins on >80% = 4+ domains
    k1_killed = mono_wins >= 4
    results["k1_mono_wins_80pct"] = k1_killed
    results["k1_pass"] = not k1_killed

    print(f"\n  K1: Mono wins {mono_wins}/5 domains, SOLE wins {sole_wins}/5")
    print(f"  K1: {'KILLED (mono wins >=80%)' if k1_killed else 'PASS (SOLE competitive)'}")

    # Additional metrics
    avg_sole_routed = sum(sole_routed_ppls.values()) / len(sole_routed_ppls)
    avg_sole_composed = sum(sole_composed_ppls.values()) / len(sole_composed_ppls)
    avg_mono = sum(mono_ppls.values()) / len(mono_ppls)
    avg_seq = sum(seq_ppls.values()) / len(seq_ppls)
    avg_base = sum(base_ppls.values()) / len(base_ppls)

    results["avg_sole_routed_ppl"] = round(avg_sole_routed, 4)
    results["avg_sole_composed_ppl"] = round(avg_sole_composed, 4)
    results["avg_mono_ppl"] = round(avg_mono, 4)
    results["avg_seq_ppl"] = round(avg_seq, 4)
    results["avg_base_ppl"] = round(avg_base, 4)

    routed_vs_mono_gap = (avg_sole_routed - avg_mono) / avg_mono * 100
    composed_vs_mono_gap = (avg_sole_composed - avg_mono) / avg_mono * 100
    seq_vs_base_gap = (avg_seq - avg_base) / avg_base * 100

    results["routed_vs_mono_gap_pct"] = round(routed_vs_mono_gap, 2)
    results["composed_vs_mono_gap_pct"] = round(composed_vs_mono_gap, 2)
    results["seq_vs_base_gap_pct"] = round(seq_vs_base_gap, 2)

    # Composition ratio (composed/best_individual)
    best_ind = min(sole_routed_ppls.values())
    composition_ratio = avg_sole_composed / best_ind
    results["composition_ratio"] = round(composition_ratio, 4)

    # Forgetting analysis for sequential
    # After training on all 5 domains sequentially, how much did earlier domains forget?
    seq_forgetting = {}
    last_domain = list(DOMAINS.keys())[-1]  # creative is last
    for d in DOMAINS:
        if d == last_domain:
            seq_forgetting[d] = "last_trained"
        else:
            forgetting = (seq_ppls[d] - base_ppls[d]) / base_ppls[d] * 100
            seq_forgetting[d] = round(forgetting, 2)
    results["seq_forgetting_pct"] = seq_forgetting

    # Verdict
    verdict = "KILLED" if k1_killed else "SUPPORTED"
    results["verdict"] = verdict

    print(f"\n  SUMMARY:")
    print(f"    Avg base PPL:           {avg_base:.2f}")
    print(f"    Avg SOLE routed PPL:    {avg_sole_routed:.2f} ({routed_vs_mono_gap:+.1f}% vs mono)")
    print(f"    Avg SOLE composed PPL:  {avg_sole_composed:.2f} ({composed_vs_mono_gap:+.1f}% vs mono)")
    print(f"    Avg Mono shuffled PPL:  {avg_mono:.2f}")
    print(f"    Avg Mono sequential PPL:{avg_seq:.2f} ({seq_vs_base_gap:+.1f}% vs base)")
    print(f"    Composition ratio:      {composition_ratio:.2f}x")
    print(f"    Mean |cos|:             {mean_cos:.6f}")
    print(f"\n  VERDICT: {verdict}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
