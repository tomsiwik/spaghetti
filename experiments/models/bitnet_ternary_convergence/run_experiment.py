#!/usr/bin/env python3
"""
BitNet-2B Ternary LoRA Convergence Experiment

Tests whether QAT+STE ternary LoRA adapters on BitNet-2B-4T converge at 1000
steps with proper train/val split and maintain composition quality.

Builds on: experiments/models/bitnet_2b_real_composition/run_experiment.py (PROVEN)
Adds: STE quantization from bitnet_ternary_adapter_composition/

Kill criteria:
  K1: ternary adapters (QAT+STE) fail to converge on >2/5 domains
  K2: 1000-step ternary adapters compose WORSE than 200-step FP16 adapters
  K3: composition ratio >5x at 1000 steps

Two conditions trained:
  (a) FP16 LoRA, 200 steps (baseline, reproducing prior result)
  (b) Ternary LoRA (QAT+STE), 1000 steps (new)

Data: HuggingFace datasets with explicit train/val splits (no contamination).
Platform: Apple Silicon MLX, $0 compute.
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
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 128  # shorter seqs for tractable runtime on Apple Silicon
LEARNING_RATE = 1e-4
VAL_BATCHES = 25   # validation batches (balance accuracy vs speed)

# Two conditions
FP16_STEPS = 200     # baseline (reproduce prior)
TERNARY_STEPS = 400  # 2x baseline (1000 too slow ~16h, 400 tractable ~4h)

EXPERIMENT_DIR = Path(__file__).parent
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# 5 domains with HuggingFace datasets - proper train/val split
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
# Ternary weight unpacking (from bitnet_2b_real_composition)
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
# LoRA with STE ternary quantization
# ===========================================================================
class TernaryLoRALinear(nn.Module):
    """LoRA layer with optional STE ternary quantization of A/B matrices.

    Forward pass:
      If ternary=True: quantize A,B to {-1,0,1}*alpha using STE
      y = base(x) + (x @ Q(A)) @ Q(B) * scale

    Backward pass:
      STE: gradients pass through quantization unchanged
    """

    def __init__(self, base_linear, r=16, scale=20.0, ternary=False):
        super().__init__()
        self.linear = base_linear
        self.r = r
        self.scale = scale
        self.ternary = ternary

        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]

        # Standard LoRA init: A ~ N(0, 1/sqrt(d)), B = 0
        s = 1.0 / math.sqrt(in_features)
        self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_features, r))
        self.lora_b = mx.zeros((r, out_features))

    def _ste_ternary(self, W):
        """Ternary quantization with Straight-Through Estimator.

        Forward: returns W_q = round(W/alpha) * alpha, clipped to {-1,0,1}*alpha
        Backward: dL/dW passes through (STE)

        Implementation: W_q = W + stop_gradient(quantize(W) - W)
        """
        alpha = mx.mean(mx.abs(W)) + 1e-10
        W_scaled = W / alpha
        # Quantize: round and clip to {-1, 0, 1}
        W_q = mx.clip(mx.round(W_scaled), -1.0, 1.0) * alpha
        # STE: forward uses quantized, backward flows through W
        return W + mx.stop_gradient(W_q - W)

    def __call__(self, x):
        base_out = self.linear(x)

        A = self.lora_a
        B = self.lora_b

        if self.ternary:
            A = self._ste_ternary(A)
            B = self._ste_ternary(B)

        lora_out = (x @ A) @ B * self.scale
        return base_out + lora_out


def apply_lora_to_model(model, rank=16, scale=20.0, ternary=False):
    """Apply LoRA (optionally ternary) to all linear layers."""
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        lora_updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora_layer = TernaryLoRALinear(module, r=rank, scale=scale, ternary=ternary)
                lora_updates.append((key, lora_layer))
                count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    print(f"  Applied {'Ternary' if ternary else 'FP16'} LoRA (r={rank}) to {count} layers")
    return model


def apply_fp16_lora_to_model(model, rank=16, scale=20.0):
    """Apply standard FP16 LoRA using mlx_lm's LoRALinear."""
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
    print(f"  Applied FP16 LoRA (r={rank}) to {count} layers")
    return model


def remove_lora(model):
    """Remove all LoRA wrappers, restoring base nn.Linear."""
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, (LoRALinear, TernaryLoRALinear)):
                base = module.linear if isinstance(module, TernaryLoRALinear) else module.linear
                updates.append((key, base))
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
            elif isinstance(module, LoRALinear):
                in_dims = module.lora_a.shape[0]
                s = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(low=-s, high=s, shape=module.lora_a.shape)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


# ===========================================================================
# Data preparation
# ===========================================================================
def prepare_domain_data(domain_name: str, domain_config: dict, data_dir_root: Path) -> Path:
    """Download HF dataset and write train.jsonl / valid.jsonl with proper split."""
    from datasets import load_dataset as hf_load

    data_dir = data_dir_root / domain_name
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"

    if train_path.exists() and valid_path.exists():
        # Check counts - need at least some reasonable amount
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
        if len(texts) >= total_needed * 2:  # collect extra for safety
            break

    if len(texts) < 100:
        raise ValueError(f"Not enough samples for {domain_name}: got {len(texts)}")

    # Proper split: use requested sizes or fall back to 85/15 of available
    if len(texts) >= total_needed:
        train_texts = texts[:max_train]
        val_texts = texts[max_train:max_train + max_val]
    else:
        # Fall back: 85/15 split of available data
        n_train = int(len(texts) * 0.85)
        train_texts = texts[:n_train]
        val_texts = texts[n_train:]
        print(f"  WARNING: {domain_name} only has {len(texts)} samples, using {len(train_texts)}/{len(val_texts)} split")

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
def compute_ppl(model, tokenizer, data_path: Path, max_batches: int = 50, split="valid"):
    """Compute perplexity on validation data."""
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
# Training loop
# ===========================================================================
def train_adapter(model, tokenizer, data_dir, domain_name, n_steps, condition_name,
                  val_every=100):
    """Train a single adapter, tracking train and val loss curves.

    Returns: dict with training metrics and loss curves.
    """
    # Tokenize training data
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
    train_losses = []
    val_ppls = []

    for step in range(n_steps):
        idx = step % len(train_tokens)
        tokens = train_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()
        train_losses.append(loss_val)

        if (step + 1) % 50 == 0 or step == 0:
            avg = sum(train_losses[-50:]) / len(train_losses[-50:])
            print(f"    Step {step+1}/{n_steps}: loss={loss_val:.4f} (avg50={avg:.4f})")

        # Validation at checkpoints
        if (step + 1) % val_every == 0 or step == n_steps - 1:
            val_ppl = compute_ppl(model, tokenizer, data_dir, max_batches=VAL_BATCHES)
            val_ppls.append({"step": step + 1, "val_ppl": round(val_ppl, 4)})
            print(f"    [VAL] Step {step+1}: val_ppl={val_ppl:.4f}")

    train_time = time.time() - t_start

    # Convergence metrics
    first_50 = sum(train_losses[:50]) / 50
    last_50 = sum(train_losses[-50:]) / 50
    converged = last_50 < first_50 * 0.95

    # Overfitting check: val PPL at end vs best val PPL
    if len(val_ppls) >= 2:
        best_val = min(v["val_ppl"] for v in val_ppls)
        final_val = val_ppls[-1]["val_ppl"]
        overfit_ratio = final_val / best_val if best_val > 0 else 1.0
    else:
        best_val = val_ppls[0]["val_ppl"] if val_ppls else float("inf")
        final_val = best_val
        overfit_ratio = 1.0

    print(f"  Done in {train_time:.1f}s. Loss: {first_50:.4f} -> {last_50:.4f} "
          f"({'converged' if converged else 'NOT converged'})")
    print(f"  Val PPL: best={best_val:.4f}, final={final_val:.4f}, overfit_ratio={overfit_ratio:.4f}")

    return {
        "condition": condition_name,
        "domain": domain_name,
        "n_steps": n_steps,
        "train_time_s": round(train_time, 1),
        "first_50_avg_loss": round(first_50, 4),
        "last_50_avg_loss": round(last_50, 4),
        "converged": converged,
        "val_ppls": val_ppls,
        "best_val_ppl": round(best_val, 4),
        "final_val_ppl": round(final_val, 4),
        "overfit_ratio": round(overfit_ratio, 4),
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
# Main experiment
# ===========================================================================
def main():
    N_DOMAINS = len(DOMAINS)
    results = {
        "experiment": "bitnet_ternary_convergence",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "domains": list(DOMAINS.keys()),
        "fp16_steps": FP16_STEPS,
        "ternary_steps": TERNARY_STEPS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=" * 70)
    print("BitNet-2B Ternary LoRA Convergence Experiment")
    print(f"  FP16 baseline: {FP16_STEPS} steps | Ternary (QAT+STE): {TERNARY_STEPS} steps")
    print(f"  Domains: {list(DOMAINS.keys())}")
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
    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"  Total parameters: {total_params:,}")

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
    # Phase 2: Base model PPL (on validation split)
    # ------------------------------------------------------------------
    print("\n[Phase 2] Computing base model PPL on validation split...")
    base_ppls = {}
    for domain_name, data_dir in data_dirs.items():
        ppl = compute_ppl(model, tokenizer, data_dir, max_batches=VAL_BATCHES, split="valid")
        base_ppls[domain_name] = round(ppl, 4)
        print(f"  {domain_name}: base PPL = {ppl:.4f}")
    results["base_ppls"] = base_ppls

    # ==================================================================
    # CONDITION A: FP16 LoRA, 200 steps (load from prior run if available)
    # ==================================================================
    print("\n" + "=" * 70)
    print("CONDITION A: FP16 LoRA, 200 steps (baseline)")
    print("=" * 70)

    # Check if FP16 adapters already exist on disk
    fp16_on_disk = all((ADAPTERS_DIR / f"fp16_{d}" / "adapter.npz").exists() for d in DOMAINS)

    if fp16_on_disk:
        print("  Loading pre-trained FP16 adapters from disk...")
        model = apply_fp16_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)
        model.freeze()
        for layer in model.model.layers:
            for key, module in layer.named_modules():
                if isinstance(module, LoRALinear):
                    module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

        fp16_adapters = {}
        for domain_name in DOMAINS:
            fp16_adapters[domain_name] = load_adapter(ADAPTERS_DIR / f"fp16_{domain_name}")
            print(f"  Loaded fp16_{domain_name}")
        results["fp16_train_results"] = "loaded_from_prior_run"
    else:
        print("  Training FP16 adapters fresh...")
        model = apply_fp16_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)
        model.freeze()
        for layer in model.model.layers:
            for key, module in layer.named_modules():
                if isinstance(module, LoRALinear):
                    module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

        trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
        print(f"  Trainable LoRA parameters: {trainable:,}")

        fp16_adapters = {}
        fp16_train_results = {}

        for domain_name, data_dir in data_dirs.items():
            print(f"\n  --- Training FP16 {domain_name} ({FP16_STEPS} steps) ---")
            zero_lora_params(model)
            train_result = train_adapter(
                model, tokenizer, data_dir, domain_name,
                n_steps=FP16_STEPS, condition_name="fp16_200",
                val_every=100,
            )
            fp16_train_results[domain_name] = train_result
            save_adapter(model, ADAPTERS_DIR / f"fp16_{domain_name}")
            fp16_adapters[domain_name] = get_lora_params(model)

        results["fp16_train_results"] = fp16_train_results

    # FP16 individual PPL
    print("\n  FP16 individual PPL (validation)...")
    fp16_individual_ppls = {}
    for domain_name in DOMAINS:
        zero_lora_params(model)
        apply_adapter_weights(model, fp16_adapters[domain_name])
        mx.eval(model.parameters())
        ppl = compute_ppl(model, tokenizer, data_dirs[domain_name], split="valid")
        fp16_individual_ppls[domain_name] = round(ppl, 4)
        print(f"  {domain_name}: {ppl:.4f} (base={base_ppls[domain_name]})")
    results["fp16_individual_ppls"] = fp16_individual_ppls

    # FP16 composed PPL
    print("\n  FP16 composed PPL (1/N)...")
    merged_fp16 = compose_adapters(list(fp16_adapters.values()))
    zero_lora_params(model)
    apply_adapter_weights(model, merged_fp16)
    mx.eval(model.parameters())

    fp16_composed_ppls = {}
    for domain_name, data_dir in data_dirs.items():
        ppl = compute_ppl(model, tokenizer, data_dir, split="valid")
        fp16_composed_ppls[domain_name] = round(ppl, 4)
        print(f"  {domain_name}: composed={ppl:.4f}")
    results["fp16_composed_ppls"] = fp16_composed_ppls

    fp16_cosines, fp16_mean_cos = compute_cosines(fp16_adapters)
    results["fp16_cosines"] = fp16_cosines
    results["fp16_mean_cos"] = fp16_mean_cos
    print(f"  FP16 mean |cos|: {fp16_mean_cos}")

    # Remove FP16 LoRA
    model = remove_lora(model)

    # ==================================================================
    # CONDITION B: Ternary LoRA (QAT+STE), extended steps
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"CONDITION B: Ternary LoRA (QAT+STE), {TERNARY_STEPS} steps")
    print("=" * 70)

    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE, ternary=True)

    # Freeze base, unfreeze LoRA
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"  Trainable LoRA parameters: {trainable:,}")

    ternary_adapters = {}
    ternary_train_results = {}

    for domain_name, data_dir in data_dirs.items():
        print(f"\n  --- Training Ternary {domain_name} ({TERNARY_STEPS} steps) ---")
        zero_lora_params(model)
        train_result = train_adapter(
            model, tokenizer, data_dir, domain_name,
            n_steps=TERNARY_STEPS, condition_name=f"ternary_{TERNARY_STEPS}",
            val_every=100,
        )
        ternary_train_results[domain_name] = train_result

        save_adapter(model, ADAPTERS_DIR / f"ternary_{domain_name}")
        ternary_adapters[domain_name] = get_lora_params(model)

    results["ternary_train_results"] = ternary_train_results

    # Ternary individual PPL
    print("\n  Ternary individual PPL (validation)...")
    ternary_individual_ppls = {}
    for domain_name in DOMAINS:
        zero_lora_params(model)
        apply_adapter_weights(model, ternary_adapters[domain_name])
        mx.eval(model.parameters())
        ppl = compute_ppl(model, tokenizer, data_dirs[domain_name], split="valid")
        ternary_individual_ppls[domain_name] = round(ppl, 4)
        base = base_ppls[domain_name]
        imp = (base - ppl) / base * 100
        print(f"  {domain_name}: {ppl:.4f} (base={base}, {imp:+.1f}%)")
    results["ternary_individual_ppls"] = ternary_individual_ppls

    # Ternary composed PPL
    print("\n  Ternary composed PPL (1/N)...")
    merged_ternary = compose_adapters(list(ternary_adapters.values()))
    zero_lora_params(model)
    apply_adapter_weights(model, merged_ternary)
    mx.eval(model.parameters())

    ternary_composed_ppls = {}
    for domain_name, data_dir in data_dirs.items():
        ppl = compute_ppl(model, tokenizer, data_dir, split="valid")
        ternary_composed_ppls[domain_name] = round(ppl, 4)
        print(f"  {domain_name}: composed={ppl:.4f}")
    results["ternary_composed_ppls"] = ternary_composed_ppls

    ternary_cosines, ternary_mean_cos = compute_cosines(ternary_adapters)
    results["ternary_cosines"] = ternary_cosines
    results["ternary_mean_cos"] = ternary_mean_cos
    print(f"  Ternary mean |cos|: {ternary_mean_cos}")

    # ==================================================================
    # Kill Criteria Assessment
    # ==================================================================
    print("\n" + "=" * 70)
    print("KILL CRITERIA ASSESSMENT")
    print("=" * 70)

    # K1: Ternary convergence (>2/N fail = KILL)
    ternary_converged = sum(1 for r in ternary_train_results.values() if r["converged"])
    ternary_failed = N_DOMAINS - ternary_converged
    k1_pass = ternary_failed <= 2
    print(f"\n  K1 (ternary convergence): {ternary_converged}/{N_DOMAINS} converged, "
          f"{ternary_failed} failed -> {'PASS' if k1_pass else 'KILL'}")
    results["k1_ternary_converged"] = ternary_converged
    results["k1_pass"] = k1_pass

    # K2: Ternary vs FP16 composition
    avg_fp16_composed = sum(fp16_composed_ppls.values()) / len(fp16_composed_ppls)
    avg_ternary_composed = sum(ternary_composed_ppls.values()) / len(ternary_composed_ppls)
    k2_ratio = avg_ternary_composed / avg_fp16_composed
    k2_pass = avg_ternary_composed <= avg_fp16_composed
    print(f"\n  K2 (ternary {TERNARY_STEPS} vs FP16 {FP16_STEPS} composed PPL):")
    print(f"    FP16 {FP16_STEPS}-step avg composed PPL:      {avg_fp16_composed:.4f}")
    print(f"    Ternary {TERNARY_STEPS}-step avg composed PPL: {avg_ternary_composed:.4f}")
    print(f"    Ratio: {k2_ratio:.4f}x -> {'PASS' if k2_pass else 'KILL'}")
    results["k2_fp16_avg_composed"] = round(avg_fp16_composed, 4)
    results["k2_ternary_avg_composed"] = round(avg_ternary_composed, 4)
    results["k2_ratio"] = round(k2_ratio, 4)
    results["k2_pass"] = k2_pass

    # K3: Composition ratio (<5x threshold)
    best_ternary_individual = min(ternary_individual_ppls.values())
    composition_ratio = avg_ternary_composed / best_ternary_individual
    k3_pass = composition_ratio < 5.0
    print(f"\n  K3 (composition ratio at {TERNARY_STEPS} steps):")
    print(f"    Best ternary individual PPL: {best_ternary_individual:.4f}")
    print(f"    Avg ternary composed PPL:    {avg_ternary_composed:.4f}")
    print(f"    Composition ratio: {composition_ratio:.4f}x -> {'PASS' if k3_pass else 'KILL'}")
    results["k3_best_individual"] = round(best_ternary_individual, 4)
    results["k3_composition_ratio"] = round(composition_ratio, 4)
    results["k3_pass"] = k3_pass

    # FP16 baseline composition ratio for comparison
    best_fp16_individual = min(fp16_individual_ppls.values())
    fp16_comp_ratio = avg_fp16_composed / best_fp16_individual
    results["fp16_composition_ratio"] = round(fp16_comp_ratio, 4)

    # ==================================================================
    # Summary
    # ==================================================================
    avg_base = sum(base_ppls.values()) / len(base_ppls)
    avg_fp16_ind = sum(fp16_individual_ppls.values()) / len(fp16_individual_ppls)
    avg_ternary_ind = sum(ternary_individual_ppls.values()) / len(ternary_individual_ppls)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Avg base PPL:                {avg_base:.4f}")
    print(f"  Avg FP16 individual PPL:     {avg_fp16_ind:.4f}")
    print(f"  Avg Ternary individual PPL:  {avg_ternary_ind:.4f}")
    print(f"  Avg FP16 composed (1/N):     {avg_fp16_composed:.4f}")
    print(f"  Avg Ternary composed (1/N):  {avg_ternary_composed:.4f}")
    print(f"  FP16 composition ratio:      {fp16_comp_ratio:.4f}x")
    print(f"  Ternary composition ratio:   {composition_ratio:.4f}x")
    print(f"  FP16 mean |cos|:             {fp16_mean_cos}")
    print(f"  Ternary mean |cos|:          {ternary_mean_cos}")

    # Per-domain comparison
    print(f"\n  Per-domain composed PPL comparison:")
    print(f"  {'Domain':<12} {'Base':<10} {'FP16 '+str(FP16_STEPS):<12} {'Tern '+str(TERNARY_STEPS):<12} {'Winner'}")
    for d in DOMAINS:
        fp = fp16_composed_ppls[d]
        tp = ternary_composed_ppls[d]
        winner = "TERNARY" if tp < fp else "FP16"
        print(f"  {d:<12} {base_ppls[d]:<10.4f} {fp:<12.4f} {tp:<12.4f} {winner}")

    ternary_wins = sum(1 for d in DOMAINS if ternary_composed_ppls[d] < fp16_composed_ppls[d])
    results["ternary_domain_wins"] = ternary_wins

    all_pass = k1_pass and k2_pass and k3_pass
    results["verdict"] = "SUPPORTED" if all_pass else "KILLED"
    results["kill_reasons"] = []
    if not k1_pass:
        results["kill_reasons"].append(f"K1: {ternary_failed}/{N_DOMAINS} domains failed convergence")
    if not k2_pass:
        results["kill_reasons"].append(f"K2: ternary composed {k2_ratio:.2f}x worse than FP16")
    if not k3_pass:
        results["kill_reasons"].append(f"K3: composition ratio {composition_ratio:.2f}x > 5.0x")

    print(f"\n  VERDICT: {results['verdict']}")
    if results["kill_reasons"]:
        for r in results["kill_reasons"]:
            print(f"    - {r}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
