#!/usr/bin/env python3
"""
BitNet-2B Ternary Composition Multi-Seed Validation

Tests reproducibility of ternary QAT+STE LoRA composition on BitNet-b1.58-2B-4T
across 3 seeds (42, 137, 314).

Kill criteria:
  K1: CV(composition_ratio) > 50% across 3 seeds
  K2: any seed has composition_ratio > 10x (catastrophe not prevented)

Reports: mean +/- std for composition ratio, |cos|, per-domain PPL

Reuses: data from bitnet_ternary_convergence, proven TernaryLoRALinear from same.
Platform: Apple Silicon MLX, $0.
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from copy import deepcopy

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

SEEDS = [42, 137, 314]

EXPERIMENT_DIR = Path(__file__).parent
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse data from bitnet_ternary_convergence
DATA_DIR = Path(__file__).parent.parent / "bitnet_ternary_convergence" / "data"

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
# TernaryLoRALinear with STE (from bitnet_ternary_convergence)
# ===========================================================================
class TernaryLoRALinear(nn.Module):
    """LoRA with STE ternary quantization of A/B matrices."""

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
    """Reset LoRA params. If seed given, set RNG for reproducibility."""
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
# Data preparation (reuse or download)
# ===========================================================================
def ensure_data(domain_name, domain_config, data_root):
    """Ensure data exists, download if needed."""
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

    # Download fresh
    from datasets import load_dataset as hf_load
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {domain_config['hf_dataset']}...")

    kwargs = {}
    if "hf_subset" in domain_config:
        kwargs["name"] = domain_config["hf_subset"]

    ds = hf_load(domain_config["hf_dataset"], **kwargs)
    split_data = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

    text_key = domain_config["text_key"]
    if text_key not in split_data.column_names:
        for alt in ["text", "content", "output", "answer", "response"]:
            if alt in split_data.column_names:
                text_key = alt
                break

    max_train = domain_config["max_samples_train"]
    max_val = domain_config["max_samples_val"]

    texts = []
    for row in split_data:
        t = row[text_key]
        if isinstance(t, str) and len(t.strip()) > 20:
            texts.append(t.strip())
        if len(texts) >= (max_train + max_val) * 2:
            break

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
# Train one adapter for one seed
# ===========================================================================
def train_adapter(model, tokenizer, data_dir, domain_name, n_steps, seed):
    """Train ternary LoRA adapter with given seed."""
    train_texts = []
    with open(data_dir / "train.jsonl") as f:
        for line in f:
            train_texts.append(json.loads(line)["text"])

    train_tokens = []
    for text in train_texts:
        toks = tokenizer.encode(text)
        if len(toks) > 2:
            train_tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))

    # Seed for data order (shuffle training data)
    import random
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

    results = {
        "experiment": "bitnet_multiseed_validation",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "train_steps": TRAIN_STEPS,
        "seeds": SEEDS,
        "domains": list(DOMAINS.keys()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=" * 70)
    print("BitNet-2B Ternary Composition Multi-Seed Validation")
    print(f"  Seeds: {SEEDS}")
    print(f"  Domains: {list(DOMAINS.keys())}")
    print(f"  Steps: {TRAIN_STEPS}")
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
    # Phase 1: Prepare data (reuse from ternary_convergence or download)
    # ------------------------------------------------------------------
    print("\n[Phase 1] Preparing data...")
    # Try reuse from ternary_convergence first, fall back to local dir
    if DATA_DIR.exists():
        data_root = DATA_DIR
        print(f"  Reusing data from {DATA_DIR}")
    else:
        data_root = EXPERIMENT_DIR / "data"
        print(f"  Downloading fresh data to {data_root}")

    data_dirs = {}
    for domain_name, config in DOMAINS.items():
        data_dirs[domain_name] = ensure_data(domain_name, config, data_root)

    # ------------------------------------------------------------------
    # Phase 2: Base PPL (seed-independent)
    # ------------------------------------------------------------------
    print("\n[Phase 2] Base model PPL...")
    base_ppls = {}
    for domain_name, data_dir in data_dirs.items():
        ppl = compute_ppl(model, tokenizer, data_dir)
        base_ppls[domain_name] = round(ppl, 4)
        print(f"  {domain_name}: {ppl:.4f}")
    results["base_ppls"] = base_ppls

    # ------------------------------------------------------------------
    # Phase 3: Train + evaluate for each seed
    # ------------------------------------------------------------------
    print("\n[Phase 3] Training across seeds...")

    # Apply ternary LoRA once (will reset weights per seed/domain)
    model = apply_ternary_lora(model, rank=LORA_RANK, scale=LORA_SCALE)
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"  Trainable params: {trainable:,}")
    results["trainable_params"] = trainable

    per_seed_results = {}

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        seed_adapters = {}
        seed_train = {}

        # Train 5 domain adapters
        for domain_name, data_dir in data_dirs.items():
            print(f"\n  --- seed={seed}, domain={domain_name} ---")

            # Reset LoRA with this seed + domain-specific offset
            zero_lora_params(model, seed=seed * 1000 + hash(domain_name) % 10000)

            train_result = train_adapter(
                model, tokenizer, data_dir, domain_name, TRAIN_STEPS, seed
            )
            seed_train[domain_name] = train_result

            # Save adapter
            adapter_path = ADAPTERS_DIR / f"seed{seed}" / domain_name
            params = save_adapter(model, adapter_path)
            seed_adapters[domain_name] = params
            print(f"    Saved. Time: {train_result['train_time_s']}s, "
                  f"converged: {train_result['converged']}")

        # Individual PPL
        print(f"\n  Individual PPL (seed={seed})...")
        seed_individual = {}
        for domain_name in DOMAINS:
            zero_lora_params(model)
            apply_adapter_weights(model, seed_adapters[domain_name])
            mx.eval(model.parameters())
            ppl = compute_ppl(model, tokenizer, data_dirs[domain_name])
            seed_individual[domain_name] = round(ppl, 4)
            base = base_ppls[domain_name]
            imp = (base - ppl) / base * 100
            print(f"    {domain_name}: {ppl:.4f} (base={base}, {imp:+.1f}%)")

        # Composed PPL (1/N)
        print(f"\n  Composed PPL 1/N (seed={seed})...")
        merged = compose_adapters(list(seed_adapters.values()))
        zero_lora_params(model)
        apply_adapter_weights(model, merged)
        mx.eval(model.parameters())

        seed_composed = {}
        for domain_name, data_dir in data_dirs.items():
            ppl = compute_ppl(model, tokenizer, data_dir)
            seed_composed[domain_name] = round(ppl, 4)
            print(f"    {domain_name}: {ppl:.4f}")

        # Cosines
        cosines, mean_cos = compute_cosines(seed_adapters)
        print(f"  Mean |cos|: {mean_cos}")

        # Composition ratio
        best_ind = min(seed_individual.values())
        avg_composed = sum(seed_composed.values()) / len(seed_composed)
        comp_ratio = avg_composed / best_ind

        per_seed_results[seed] = {
            "train_results": seed_train,
            "individual_ppls": seed_individual,
            "composed_ppls": seed_composed,
            "cosines": cosines,
            "mean_cos": mean_cos,
            "best_individual_ppl": round(best_ind, 4),
            "avg_composed_ppl": round(avg_composed, 4),
            "composition_ratio": round(comp_ratio, 4),
        }

        print(f"\n  Seed {seed} summary: ratio={comp_ratio:.4f}x, |cos|={mean_cos}")

    results["per_seed"] = per_seed_results

    # ------------------------------------------------------------------
    # Phase 4: Aggregate statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("AGGREGATE STATISTICS")
    print("=" * 70)

    ratios = [per_seed_results[s]["composition_ratio"] for s in SEEDS]
    cosines_all = [per_seed_results[s]["mean_cos"] for s in SEEDS]

    mean_ratio = sum(ratios) / len(ratios)
    std_ratio = (sum((r - mean_ratio)**2 for r in ratios) / (len(ratios) - 1)) ** 0.5
    cv_ratio = (std_ratio / mean_ratio * 100) if mean_ratio > 0 else float("inf")

    mean_cos_agg = sum(cosines_all) / len(cosines_all)
    std_cos = (sum((c - mean_cos_agg)**2 for c in cosines_all) / (len(cosines_all) - 1)) ** 0.5

    print(f"\n  Composition ratio: {mean_ratio:.4f} +/- {std_ratio:.4f} (CV={cv_ratio:.1f}%)")
    print(f"  Mean |cos|:        {mean_cos_agg:.6f} +/- {std_cos:.6f}")

    for s in SEEDS:
        r = per_seed_results[s]
        print(f"    Seed {s}: ratio={r['composition_ratio']:.4f}x, |cos|={r['mean_cos']}")

    # Per-domain stats
    print(f"\n  Per-domain individual PPL (mean +/- std):")
    domain_stats = {}
    for d in DOMAINS:
        vals = [per_seed_results[s]["individual_ppls"][d] for s in SEEDS]
        m = sum(vals) / len(vals)
        sd = (sum((v - m)**2 for v in vals) / (len(vals) - 1)) ** 0.5
        domain_stats[d] = {"individual_mean": round(m, 4), "individual_std": round(sd, 4)}
        print(f"    {d}: {m:.4f} +/- {sd:.4f}")

    print(f"\n  Per-domain composed PPL (mean +/- std):")
    for d in DOMAINS:
        vals = [per_seed_results[s]["composed_ppls"][d] for s in SEEDS]
        m = sum(vals) / len(vals)
        sd = (sum((v - m)**2 for v in vals) / (len(vals) - 1)) ** 0.5
        domain_stats[d]["composed_mean"] = round(m, 4)
        domain_stats[d]["composed_std"] = round(sd, 4)
        print(f"    {d}: {m:.4f} +/- {sd:.4f}")

    results["domain_stats"] = domain_stats

    aggregate = {
        "composition_ratio_mean": round(mean_ratio, 4),
        "composition_ratio_std": round(std_ratio, 4),
        "composition_ratio_cv_pct": round(cv_ratio, 1),
        "composition_ratio_values": ratios,
        "mean_cos_mean": round(mean_cos_agg, 6),
        "mean_cos_std": round(std_cos, 6),
        "mean_cos_values": cosines_all,
        "max_ratio": round(max(ratios), 4),
        "min_ratio": round(min(ratios), 4),
    }
    results["aggregate"] = aggregate

    # ------------------------------------------------------------------
    # Kill criteria
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("KILL CRITERIA")
    print("=" * 70)

    k1_pass = cv_ratio <= 50.0
    k2_pass = all(r < 10.0 for r in ratios)

    print(f"\n  K1 (CV < 50%): CV = {cv_ratio:.1f}% -> {'PASS' if k1_pass else 'KILL'}")
    print(f"  K2 (all ratios < 10x): max = {max(ratios):.4f}x -> {'PASS' if k2_pass else 'KILL'}")

    results["k1_cv"] = round(cv_ratio, 1)
    results["k1_pass"] = k1_pass
    results["k2_max_ratio"] = round(max(ratios), 4)
    results["k2_pass"] = k2_pass

    verdict = "SUPPORTED" if (k1_pass and k2_pass) else "KILLED"
    results["verdict"] = verdict

    total_time = time.time() - t_global
    results["total_time_s"] = round(total_time, 1)

    print(f"\n  VERDICT: {verdict}")
    print(f"  Total time: {total_time/60:.1f} min")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
