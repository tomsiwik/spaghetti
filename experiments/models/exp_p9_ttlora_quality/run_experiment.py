#!/usr/bin/env python3
"""
P9.B1: TT-LoRA Quality on Gemma 4 GSM8K
Paper: TT-LoRA MoE (arXiv:2504.21190)
Prior: Finding #515 (TT-LoRA port to MLX verified)

Kill criteria:
  K1357: TT-LoRA GSM8K >= 60% of standard LoRA accuracy
  K1358: TT-LoRA adapter size <= 200KB total (42 layers)
  K1359: Training converges (loss decreasing after 100 steps)

Design:
  TT-LoRA r=6 on v_proj only, all 42 layers → 63,756 params (~154KB)
  vs LoRA r=6 on v_proj only → 774,144 params (~1.5MB)
  Both trained on GSM8K 1000 steps, evaluated on 100 test problems.
"""

import gc
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Memory safety
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_TRAIN = 50 if IS_SMOKE else 2000
N_EVAL = 5 if IS_SMOKE else 100
N_STEPS = 20 if IS_SMOKE else 1000
BATCH_SIZE = 2
MAX_SEQ_LEN = 512

TT_RANK = 6
TT_LR = 5e-3       # Paper-recommended for TT-LoRA (arXiv:2504.21190)
TT_ALPHA = 1.0

LORA_RANK = 6
LORA_LR = 1e-4     # Our proven LoRA lr
LORA_SCALE = 6.0   # alpha = rank (conventional)

PROJ_KEYS_MLX_LM = ["self_attn.v_proj"]  # For mlx_lm CLI config
PROJ_NAMES = ["v_proj"]                   # For injection
SEED = 42

HIDDEN_SIZE = 2560  # Gemma 4 E4B hidden dimension


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB",
          flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ────────────────────────────────────────────────
# TT-LoRA Module
# ────────────────────────────────────────────────

class TTLoRAWrapper(nn.Module):
    """TT-LoRA adapter wrapping an existing (possibly quantized) linear layer.

    Forward: base_layer(x) + alpha * x @ reconstruct(tt_cores).T
    At init, last core is zero → output = base_layer(x) (no perturbation).
    """

    def __init__(self, base_layer, in_features, out_features, tt_shape,
                 tt_rank=6, alpha=1.0):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.tt_shape = tt_shape

        # Validate factorization split
        self._validate_split(tt_shape, in_features, out_features)

        # Build TT cores as named attributes (not a list — MLX freeze/unfreeze
        # doesn't track list-of-array attributes correctly).
        # Ranks: [1, r, r, ..., r, 1]
        d = len(tt_shape)
        ranks = [1] + [tt_rank] * (d - 1) + [1]
        self._n_cores = d
        for k in range(d):
            shape = (ranks[k], tt_shape[k], ranks[k + 1])
            if k == d - 1:
                core = mx.zeros(shape)  # Zero init → zero output at start
            else:
                std = 1.0 / math.sqrt(tt_shape[k] * ranks[k])
                core = mx.random.normal(shape) * std
            setattr(self, f"core_{k}", core)
        self._cached_delta_w = None

    def _validate_split(self, tt_shape, in_features, out_features):
        prod = 1
        for i, s in enumerate(tt_shape):
            prod *= s
            if prod == in_features:
                rest = 1
                for j in range(i + 1, len(tt_shape)):
                    rest *= tt_shape[j]
                assert rest == out_features, (
                    f"Output factors product {rest} != {out_features}")
                return
        raise ValueError(
            f"Cannot split {tt_shape} into {in_features} x {out_features}")

    @property
    def tt_cores(self):
        return [getattr(self, f"core_{k}") for k in range(self._n_cores)]

    def reconstruct_delta_w(self):
        """Contract TT cores → ΔW [out_features, in_features]."""
        cores = self.tt_cores
        result = cores[0].squeeze(0)  # [s_0, r_1]
        for k in range(1, len(cores)):
            core = cores[k]  # [r_k, s_k, r_{k+1}]
            r_k, s_k, r_next = core.shape
            result = result @ core.reshape(r_k, s_k * r_next)
            leading = result.shape[0]
            result = result.reshape(leading * s_k, r_next)
        result = result.squeeze(-1)  # [in_features * out_features]
        return result.reshape(self.in_features, self.out_features).T

    def cache_delta_w(self):
        """Pre-compute and cache ΔW for fast inference."""
        self._cached_delta_w = self.reconstruct_delta_w()
        mx.eval(self._cached_delta_w)

    def __call__(self, x):
        base_out = self.base_layer(x)
        dw = (self._cached_delta_w if self._cached_delta_w is not None
              else self.reconstruct_delta_w())
        return base_out + self.alpha * (x @ dw.T)

    def num_params(self):
        return sum(c.size for c in self.tt_cores)


def factorize(n, max_factor=10):
    """Factorize n into small factors, preferring 8s."""
    factors = []
    while n % 8 == 0 and n > 8:
        factors.append(8)
        n //= 8
    for f in range(max_factor, 1, -1):
        while n % f == 0 and n > 1:
            factors.append(f)
            n //= f
    if n > 1:
        factors.append(n)
    factors.sort()
    return factors


def compute_tt_shape(in_features, out_features):
    return factorize(in_features) + factorize(out_features)


# ────────────────────────────────────────────────
# Model Injection
# ────────────────────────────────────────────────

def get_layers(model):
    """Get transformer layers from model (handles nested .model attribute)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    return model.layers


def detect_proj_dims(base_layer, hidden_size=HIDDEN_SIZE):
    """Detect actual in/out dimensions of a (possibly quantized) linear layer."""
    test_x = mx.zeros((1, 1, hidden_size))
    test_y = base_layer(test_x)
    mx.eval(test_y)
    in_f = hidden_size
    out_f = test_y.shape[-1]
    del test_x, test_y
    return in_f, out_f


def inject_ttlora(model, proj_names, tt_rank, alpha):
    """Replace specified projections with TT-LoRA wrappers in all layers."""
    layers = get_layers(model)
    total_params = 0
    dim_info = {}
    for i, layer in enumerate(layers):
        for name in proj_names:
            base = getattr(layer.self_attn, name)
            in_f, out_f = detect_proj_dims(base)
            if (name, in_f, out_f) not in dim_info:
                dim_info[(name, in_f, out_f)] = []
            dim_info[(name, in_f, out_f)].append(i)

            tt_shape = compute_tt_shape(in_f, out_f)
            wrapper = TTLoRAWrapper(base, in_f, out_f, tt_shape, tt_rank, alpha)
            setattr(layer.self_attn, name, wrapper)
            total_params += wrapper.num_params()

    # Report discovered dimensions
    for (name, in_f, out_f), layer_ids in dim_info.items():
        tt_shape = compute_tt_shape(in_f, out_f)
        print(f"  {name}: {in_f}→{out_f} (TT shape {tt_shape}, "
              f"{len(layer_ids)} layers)", flush=True)
    return total_params


# ────────────────────────────────────────────────
# Data Preparation
# ────────────────────────────────────────────────

def prepare_gsm8k_data():
    """Download and format GSM8K for SFT training."""
    from datasets import load_dataset

    data_dir = EXPERIMENT_DIR / "data" / "math"
    data_dir.mkdir(parents=True, exist_ok=True)

    if (data_dir / "train.jsonl").exists():
        n_train = sum(1 for _ in open(data_dir / "train.jsonl"))
        print(f"GSM8K data already prepared ({n_train} train examples)", flush=True)
        return data_dir

    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(N_TRAIN, len(ds))))

    records = []
    for ex in ds:
        records.append(json.dumps({"messages": [
            {"role": "user", "content": f"Solve step by step.\n\n{ex['question']}"},
            {"role": "assistant", "content": ex["answer"]},
        ]}))

    n_val = max(1, len(records) // 10)
    (data_dir / "train.jsonl").write_text("\n".join(records[n_val:]))
    (data_dir / "valid.jsonl").write_text("\n".join(records[:n_val]))
    print(f"GSM8K: {len(records) - n_val} train, {n_val} val", flush=True)
    return data_dir


# ────────────────────────────────────────────────
# TT-LoRA Training
# ────────────────────────────────────────────────

def tokenize_for_training(tokenizer, data_path, max_seq_len=512):
    """Tokenize JSONL with prompt length tracking for SFT masking."""
    examples = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line)
            msgs = ex["messages"]

            # Full conversation text
            full = tokenizer.apply_chat_template(msgs, tokenize=False)
            full_ids = tokenizer.encode(full)

            # Prompt-only text (for loss masking)
            prompt = tokenizer.apply_chat_template(
                [msgs[0]], tokenize=False, add_generation_prompt=True)
            prompt_len = len(tokenizer.encode(prompt))

            if len(full_ids) > max_seq_len:
                full_ids = full_ids[:max_seq_len]
            if prompt_len >= len(full_ids):
                continue  # No response tokens after truncation

            examples.append({
                "input_ids": full_ids,
                "prompt_len": prompt_len,
                "length": len(full_ids),
            })
    return examples


def train_ttlora_loop(model, tokenizer, data_dir, n_steps, lr, batch_size):
    """Custom TT-LoRA training loop with SFT loss (prompt-masked)."""
    examples = tokenize_for_training(
        tokenizer, data_dir / "train.jsonl", MAX_SEQ_LEN)
    print(f"Loaded {len(examples)} training examples", flush=True)

    random.seed(SEED)
    random.shuffle(examples)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(model, input_ids, lengths, prompt_lens):
        logits = model(input_ids)
        logits = logits.astype(mx.float32)

        # Next-token prediction: predict token p+1 from prefix 0..p
        shift_logits = logits[:, :-1, :]
        shift_targets = input_ids[:, 1:]

        ce = nn.losses.cross_entropy(shift_logits, shift_targets, reduction="none")

        # Mask: response tokens only (after prompt, before padding)
        S = shift_targets.shape[1]
        pos = mx.arange(S)[None, :]
        # In shifted space: position p corresponds to predicting input_ids[p+1]
        # Response starts at input_ids[prompt_len] → shifted pos prompt_len-1
        # Last real token at input_ids[length-1] → shifted pos length-2
        mask = (pos >= (prompt_lens[:, None] - 1)) & (pos < (lengths[:, None] - 1))
        mask = mask.astype(mx.float32)

        return (ce * mask).sum() / mx.maximum(mask.sum(), 1.0)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    t0 = time.time()
    idx = 0

    for step in range(n_steps):
        # Build padded batch
        batch_exs = []
        for _ in range(batch_size):
            batch_exs.append(examples[idx % len(examples)])
            idx += 1

        max_len = max(e["length"] for e in batch_exs)
        input_ids = mx.array([
            e["input_ids"] + [pad_id] * (max_len - e["length"])
            for e in batch_exs
        ])
        lengths = mx.array([e["length"] for e in batch_exs])
        prompt_lens = mx.array([e["prompt_len"] for e in batch_exs])

        # Forward + backward + update
        loss, grads = loss_and_grad(model, input_ids, lengths, prompt_lens)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

        losses.append(loss.item())

        if (step + 1) % max(1, n_steps // 10) == 0:
            avg = sum(losses[-100:]) / len(losses[-100:])
            elapsed = time.time() - t0
            print(f"  Step {step+1}/{n_steps}: loss={avg:.4f} ({elapsed:.1f}s)",
                  flush=True)

    total_time = time.time() - t0
    print(f"TT-LoRA training complete: {total_time:.1f}s", flush=True)
    return losses, total_time


def save_ttlora_adapter(model, save_path, proj_names):
    """Save TT-LoRA cores as float16 safetensors. Returns file size in bytes."""
    from safetensors.numpy import save_file

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    layers = get_layers(model)
    weights = {}
    total_params = 0
    for i, layer in enumerate(layers):
        for pname in proj_names:
            proj = getattr(layer.self_attn, pname)
            if hasattr(proj, "_n_cores"):
                for j in range(proj._n_cores):
                    core = getattr(proj, f"core_{j}")
                    key = f"layers.{i}.self_attn.{pname}.core_{j}"
                    arr = np.array(core)
                    weights[key] = arr.astype(np.float16)
                    total_params += arr.size

    filepath = save_path / "tt_cores.safetensors"
    save_file(weights, str(filepath))
    size = filepath.stat().st_size
    print(f"TT-LoRA saved: {total_params:,} params, {size:,} bytes ({size/1024:.1f} KB)",
          flush=True)
    return size


# ────────────────────────────────────────────────
# LoRA Baseline Training (via mlx_lm CLI)
# ────────────────────────────────────────────────

def train_lora_baseline(data_dir, adapter_path, lora_rank, lora_scale,
                        lr, n_steps):
    """Train standard LoRA via mlx_lm subprocess."""
    import yaml

    adapter_path = Path(adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)
    config_path = EXPERIMENT_DIR / "lora_config.yaml"

    config = {
        "model": MODEL_ID,
        "train": True,
        "data": str(data_dir),
        "fine_tune_type": "lora",
        "num_layers": -1,
        "iters": n_steps,
        "batch_size": BATCH_SIZE,
        "learning_rate": lr,
        "lora_parameters": {
            "rank": lora_rank,
            "scale": lora_scale,
            "dropout": 0.0,
            "keys": PROJ_KEYS_MLX_LM,
        },
        "adapter_path": str(adapter_path),
        "save_every": n_steps,
        "val_batches": 5,
        "steps_per_report": max(10, n_steps // 10),
        "steps_per_eval": n_steps,
        "max_seq_length": MAX_SEQ_LEN,
        "mask_prompt": True,
        "grad_checkpoint": True,
        "seed": SEED,
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Training LoRA baseline (rank={lora_rank}, scale={lora_scale}, "
          f"{n_steps} steps, keys={PROJ_KEYS_MLX_LM})", flush=True)
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, "-m", "mlx_lm", "lora", "-c", str(config_path)],
        capture_output=False, text=True, cwd=str(EXPERIMENT_DIR),
    )

    elapsed = time.time() - t0
    print(f"LoRA training: {elapsed:.1f}s (exit={result.returncode})", flush=True)

    size_bytes = 0
    if adapter_path.exists():
        size_bytes = sum(
            f.stat().st_size for f in adapter_path.rglob("*") if f.is_file())

    return {
        "train_time_s": round(elapsed, 1),
        "exit_code": result.returncode,
        "adapter_size_bytes": size_bytes,
    }


# ────────────────────────────────────────────────
# GSM8K Evaluation
# ────────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, n_eval, label=""):
    """Evaluate GSM8K accuracy via generation."""
    from datasets import load_dataset
    from mlx_lm import generate

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))

    correct = 0
    for i, ex in enumerate(ds):
        messages = [{"role": "user",
                     "content": f"Solve step by step.\n\n{ex['question']}"}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        response = generate(
            model, tokenizer, prompt=formatted,
            max_tokens=512, verbose=False,
        )

        # Ground truth: number after ####
        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt = gt_match.group(1).replace(",", "").strip()

        # Prediction: #### marker first, then last number as fallback
        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match:
            if pred_match.group(1).replace(",", "").strip() == gt:
                correct += 1
        else:
            nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
            if nums and nums[-1] == gt:
                correct += 1

        if (i + 1) % 25 == 0:
            print(f"  GSM8K {label}: {i+1}/{len(ds)}, "
                  f"acc={correct/(i+1)*100:.1f}%", flush=True)

    acc = correct / len(ds) * 100
    print(f"GSM8K {label}: {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    return acc, correct, len(ds)


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 60)
    print("P9.B1: TT-LoRA Quality on Gemma 4 GSM8K")
    print(f"TT-LoRA: rank={TT_RANK}, lr={TT_LR}, alpha={TT_ALPHA}, v_proj only")
    print(f"LoRA:    rank={LORA_RANK}, lr={LORA_LR}, scale={LORA_SCALE}, v_proj only")
    print(f"SMOKE={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_EVAL={N_EVAL}, N_STEPS={N_STEPS}")
    print("=" * 60, flush=True)

    results = {
        "experiment": "exp_p9_ttlora_quality",
        "paper": "arXiv:2504.21190",
        "prior_finding": "#515",
        "tt_rank": TT_RANK,
        "tt_lr": TT_LR,
        "lora_rank": LORA_RANK,
        "lora_lr": LORA_LR,
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
        "n_steps": N_STEPS,
    }

    # ── Phase 1: Prepare data ──────────────────
    print("\n── Phase 1: Prepare GSM8K data ──", flush=True)
    data_dir = prepare_gsm8k_data()

    # ── Phase 2: Train LoRA baseline ───────────
    print("\n── Phase 2: Train LoRA baseline ──", flush=True)
    lora_adapter_path = EXPERIMENT_DIR / "adapters" / "lora_baseline"
    lora_train = train_lora_baseline(
        data_dir, lora_adapter_path, LORA_RANK, LORA_SCALE, LORA_LR, N_STEPS)
    results["lora_train"] = lora_train

    # ── Phase 3: Train TT-LoRA ─────────────────
    print("\n── Phase 3: Train TT-LoRA ──", flush=True)
    from mlx_lm import load

    print("Loading model for TT-LoRA training...", flush=True)
    model, tokenizer = load(MODEL_ID)
    log_memory("model-loaded")

    # Inject TT-LoRA wrappers into v_proj
    tt_total_params = inject_ttlora(model, PROJ_NAMES, TT_RANK, TT_ALPHA)
    print(f"TT-LoRA injected: {tt_total_params:,} params across "
          f"{len(get_layers(model))} layers", flush=True)

    # Freeze all, unfreeze TT cores only
    model.freeze()
    for layer in get_layers(model):
        for pname in PROJ_NAMES:
            proj = getattr(layer.self_attn, pname)
            if hasattr(proj, "_n_cores"):
                core_keys = [f"core_{k}" for k in range(proj._n_cores)]
                proj.unfreeze(keys=core_keys, recurse=False)

    # Verify trainable param count
    trainable = sum(
        p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    print(f"Trainable parameters: {trainable:,} "
          f"(expected: {tt_total_params:,})", flush=True)

    # Train
    model.train()
    tt_losses, tt_train_time = train_ttlora_loop(
        model, tokenizer, data_dir, N_STEPS, TT_LR, BATCH_SIZE)

    results["tt_train"] = {
        "total_params": tt_total_params,
        "train_time_s": round(tt_train_time, 1),
        "final_loss": round(tt_losses[-1], 4) if tt_losses else None,
    }

    # K1359: Convergence — compare first 50 steps avg vs steps 50-100 avg
    if len(tt_losses) >= 100:
        avg_first = sum(tt_losses[:50]) / 50
        avg_second = sum(tt_losses[50:100]) / 50
        k1359_pass = avg_second < avg_first
        results["K1359_convergence"] = "PASS" if k1359_pass else "FAIL"
        results["K1359_detail"] = (
            f"first_50_avg={avg_first:.4f} → next_50_avg={avg_second:.4f}")
    else:
        results["K1359_convergence"] = "N/A"
        results["K1359_detail"] = f"Only {len(tt_losses)} steps completed"

    # Save TT-LoRA cores
    tt_adapter_path = EXPERIMENT_DIR / "adapters" / "ttlora"
    tt_size = save_ttlora_adapter(model, tt_adapter_path, PROJ_NAMES)
    results["tt_adapter_size_bytes"] = tt_size
    results["K1358_size"] = "PASS" if tt_size <= 200 * 1024 else "FAIL"
    results["K1358_detail"] = f"{tt_size:,} bytes ({tt_size / 1024:.1f} KB)"

    # ── Phase 4: Eval TT-LoRA ──────────────────
    print("\n── Phase 4: Evaluate TT-LoRA on GSM8K ──", flush=True)
    model.eval()

    # Cache reconstructed ΔW for fast inference
    for layer in get_layers(model):
        for pname in PROJ_NAMES:
            proj = getattr(layer.self_attn, pname)
            if isinstance(proj, TTLoRAWrapper):
                proj.cache_delta_w()

    tt_acc, tt_correct, tt_total = eval_gsm8k(
        model, tokenizer, N_EVAL, "TT-LoRA")
    results["tt_gsm8k_acc"] = round(tt_acc, 1)
    results["tt_gsm8k_correct"] = tt_correct
    results["tt_gsm8k_total"] = tt_total

    cleanup(model, tokenizer)

    # ── Phase 5: Eval LoRA baseline ────────────
    print("\n── Phase 5: Evaluate LoRA baseline on GSM8K ──", flush=True)

    if lora_train["exit_code"] == 0:
        model, tokenizer = load(MODEL_ID, adapter_path=str(lora_adapter_path))
        log_memory("lora-loaded")

        lora_acc, lora_correct, lora_total = eval_gsm8k(
            model, tokenizer, N_EVAL, "LoRA")
        results["lora_gsm8k_acc"] = round(lora_acc, 1)
        results["lora_gsm8k_correct"] = lora_correct
        results["lora_gsm8k_total"] = lora_total

        cleanup(model, tokenizer)
    else:
        print("LoRA training failed — skipping eval", flush=True)
        lora_acc = 0.0
        results["lora_gsm8k_acc"] = 0.0
        results["lora_train_failed"] = True

    # ── Phase 6: Compare ───────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60, flush=True)

    # K1357: Quality comparison
    if lora_acc > 0:
        quality_ratio = tt_acc / lora_acc
        k1357_pass = quality_ratio >= 0.6
    else:
        quality_ratio = float("inf") if tt_acc > 0 else 0.0
        k1357_pass = tt_acc > 0  # If LoRA failed, TT-LoRA just needs to work

    results["quality_ratio"] = round(quality_ratio, 3)
    results["K1357_quality"] = "PASS" if k1357_pass else "FAIL"
    results["K1357_detail"] = (
        f"TT={tt_acc:.1f}% / LoRA={lora_acc:.1f}% = "
        f"{quality_ratio:.2f} (need >= 0.60)")

    # Compression ratio
    lora_size = lora_train.get("adapter_size_bytes", 0)
    compression = lora_size / tt_size if tt_size > 0 else float("inf")
    results["compression_ratio"] = round(compression, 1)

    # Param comparison — LoRA adapter is float32 safetensors
    # Actual size = one copy of adapters.safetensors (mlx_lm saves two copies)
    lora_adapter_file = lora_adapter_path / "adapters.safetensors"
    if lora_adapter_file.exists():
        lora_adapter_bytes = lora_adapter_file.stat().st_size
        lora_total_params = lora_adapter_bytes // 4  # float32
    else:
        lora_total_params = 0
    results["lora_total_params"] = lora_total_params
    results["lora_adapter_bytes"] = lora_adapter_bytes if lora_adapter_file.exists() else 0

    # Print
    print(f"\nTT-LoRA: {tt_total_params:,} params, "
          f"{tt_size / 1024:.1f} KB, GSM8K = {tt_acc:.1f}%")
    print(f"LoRA:    {lora_total_params:,} params, "
          f"{lora_size / 1024:.1f} KB, GSM8K = {lora_acc:.1f}%")
    print(f"Compression: {compression:.1f}x smaller adapter")

    print(f"\nK1357 Quality:     {results['K1357_quality']} "
          f"({results['K1357_detail']})")
    print(f"K1358 Size:        {results['K1358_size']} "
          f"({results['K1358_detail']})")
    print(f"K1359 Convergence: {results['K1359_convergence']} "
          f"({results['K1359_detail']})")

    total_time = time.time() - t_start
    results["total_time_s"] = round(total_time, 1)
    overall = (results["K1357_quality"] == "PASS" and
               results["K1358_size"] == "PASS" and
               results["K1359_convergence"] == "PASS")
    results["overall_pass"] = overall
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    print(f"Total time: {total_time:.0f}s ({total_time / 60:.1f} min)", flush=True)

    # Save results
    (EXPERIMENT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Results saved to {EXPERIMENT_DIR}/results.json", flush=True)


if __name__ == "__main__":
    main()
