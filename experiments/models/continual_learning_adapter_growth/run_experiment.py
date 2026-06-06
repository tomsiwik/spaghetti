#!/usr/bin/env python3
"""Continual Learning: Grow adapter pool over time without forgetting.

Start with 5 adapters, add 1 per cycle for 10 cycles (N=5 to N=15).
After each addition, measure:
  1. New adapter individual quality (PPL vs base)
  2. Existing adapter quality (catastrophic forgetting?)
  3. All-N uniform composition quality
  4. Pairwise orthogonality

Kill criteria:
  K1 (#247): Any existing adapter degrades > 5% after new addition
  K2 (#248): Composition quality degrades monotonically with N

Reuses adapters/data from real_data_25_domain_adapters where available.

Platform: Apple M5 Pro 48GB, MLX
"""

import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
DATA_DIR = EXPERIMENT_DIR / "data"

# Reuse from prior experiment
PRIOR_DIR = EXPERIMENT_DIR.parent / "real_data_25_domain_adapters"
PRIOR_ADAPTERS_DIR = PRIOR_DIR / "adapters"
PRIOR_DATA_DIR = PRIOR_DIR / "data"

# Also check the 5-domain experiment
FIVE_DOMAIN_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"

# ============================================================================
# Configuration
# ============================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
TRAIN_ITERS = 200
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
VAL_BATCHES = 25
SEED = 42

# Initial 5 domains (reuse trained adapters)
INITIAL_DOMAINS = ["medical", "code", "math", "legal", "finance"]

# 10 domains to add one per cycle (use genuine domains first, then slice-based)
GROWTH_DOMAINS = [
    "health_fitness",   # genuine - MedQuAD
    "psychology",       # genuine - mental health
    "science",          # dolly slice
    "history",          # dolly slice
    "philosophy",       # dolly slice
    "creative_writing", # dolly slice
    "cooking",          # wizard_vicuna slice
    "education",        # code instructions slice
    "engineering",      # code instructions slice
    "agriculture",      # dolly slice
]

ALL_DOMAINS = INITIAL_DOMAINS + GROWTH_DOMAINS
N_INITIAL = len(INITIAL_DOMAINS)
N_GROWTH_STEPS = len(GROWTH_DOMAINS)

# Dataset specs for growth domains (matching real_data_25_domain_adapters)
DATASET_SPECS = {
    "health_fitness": {
        "hf_repo": "keivalya/MedQuad-MedicalQnADataset",
        "hf_file": "medDataset_processed.csv",
        "hf_format": "csv",
        "inst_key": "Question", "resp_key": "Answer",
    },
    "psychology": {
        "hf_repo": "Amod/mental_health_counseling_conversations",
        "hf_file": "combined_dataset.json",
        "hf_format": "jsonl",
        "inst_key": "Context", "resp_key": "Response",
    },
    "science": {
        "hf_repo": "databricks/databricks-dolly-15k",
        "hf_file": "databricks-dolly-15k.jsonl",
        "hf_format": "jsonl",
        "inst_key": "instruction", "resp_key": "response",
        "offset": 0, "max_take": 500,
    },
    "history": {
        "hf_repo": "databricks/databricks-dolly-15k",
        "hf_file": "databricks-dolly-15k.jsonl",
        "hf_format": "jsonl",
        "inst_key": "instruction", "resp_key": "response",
        "offset": 500, "max_take": 500,
    },
    "philosophy": {
        "hf_repo": "databricks/databricks-dolly-15k",
        "hf_file": "databricks-dolly-15k.jsonl",
        "hf_format": "jsonl",
        "inst_key": "instruction", "resp_key": "response",
        "offset": 1000, "max_take": 500,
    },
    "creative_writing": {
        "hf_repo": "databricks/databricks-dolly-15k",
        "hf_file": "databricks-dolly-15k.jsonl",
        "hf_format": "jsonl",
        "inst_key": "instruction", "resp_key": "response",
        "offset": 1500, "max_take": 500,
    },
    "cooking": {
        "hf_repo": "ehartford/wizard_vicuna_70k_unfiltered",
        "hf_file": "wizard_vicuna_dataset_unfiltered.json",
        "hf_format": "json",
        "conv_format": "hf_conv",
        "inst_key": "conversations",
        "offset": 0, "max_take": 500,
    },
    "education": {
        "hf_repo": "TokenBender/code_instructions_122k_alpaca_style",
        "hf_file": "code_instructions_120k.json",
        "hf_format": "json",
        "inst_key": "instruction", "resp_key": "output",
        "offset": 0, "max_take": 500,
    },
    "engineering": {
        "hf_repo": "TokenBender/code_instructions_122k_alpaca_style",
        "hf_file": "code_instructions_120k.json",
        "hf_format": "json",
        "inst_key": "instruction", "resp_key": "output",
        "offset": 500, "max_take": 500,
    },
    "agriculture": {
        "hf_repo": "databricks/databricks-dolly-15k",
        "hf_file": "databricks-dolly-15k.jsonl",
        "hf_format": "jsonl",
        "inst_key": "instruction", "resp_key": "response",
        "offset": 2000, "max_take": 500,
    },
    # Initial domains (for fallback data loading)
    "medical": {
        "hf_repo": "medalpaca/medical_meadow_medical_flashcards",
        "hf_file": "medical_meadow_wikidoc_medical_flashcards.json",
        "hf_format": "json",
        "inst_key": "input", "resp_key": "output",
    },
    "code": {
        "hf_repo": "iamtarun/python_code_instructions_18k_alpaca",
        "hf_file": "data/train-00000-of-00001-8b6e212f3e1ece96.parquet",
        "hf_format": "parquet",
        "inst_key": "instruction", "resp_key": "output",
    },
    "math": {
        "hf_repo": "openai/gsm8k",
        "hf_file": "main/train-00000-of-00001.parquet",
        "hf_format": "parquet",
        "inst_key": "question", "resp_key": "answer",
    },
    "legal": {
        "hf_repo": "jonathanli/law-stack-exchange",
        "hf_file": "train.jsonl",
        "hf_format": "jsonl",
        "inst_key": "title", "resp_key": "body",
    },
    "finance": {
        "hf_repo": "gbharti/finance-alpaca",
        "hf_file": "Cleaned_date.json",
        "hf_format": "json",
        "inst_key": "instruction", "resp_key": "output",
    },
}


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ============================================================================
# Grassmannian AP Init
# ============================================================================

def grassmannian_ap_init(N, r, d, seed=42):
    """Generate N orthogonally-packed A matrices via QR decomposition.

    When N*r <= d, we get EXACT orthogonality via concatenated QR.
    """
    rng = np.random.RandomState(seed)
    frames = np.zeros((N, d, r), dtype=np.float32)

    if N * r <= d:
        # Perfect orthogonality: QR on concatenated random matrix
        M = rng.randn(d, N * r).astype(np.float32)
        Q, _ = np.linalg.qr(M)
        for i in range(N):
            frames[i] = Q[:, i * r:(i + 1) * r]
    else:
        # Fallback: individual QR (approximate orthogonality)
        for i in range(N):
            M = rng.randn(d, r).astype(np.float32)
            Q, _ = np.linalg.qr(M)
            frames[i] = Q[:, :r]

    return frames


# ============================================================================
# STE Ternary LoRA Layer
# ============================================================================

class TernaryLoRALinear(nn.Module):
    """LoRA with frozen Grassmannian A and STE-ternary B."""
    def __init__(self, base_linear: nn.Linear, rank: int = 16,
                 scale: float = 20.0, a_init: mx.array = None):
        super().__init__()
        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]
        self.linear = base_linear
        if a_init is not None:
            self.lora_a = a_init
        else:
            s = 1.0 / math.sqrt(in_features)
            self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_features, rank))
        self.lora_b = mx.zeros((rank, out_features))
        self.scale = scale
        self.rank = rank
        self.linear.freeze()
        self.freeze(keys=["lora_a"], strict=False)

    def __call__(self, x):
        base_out = self.linear(x)
        b = self.lora_b
        alpha = mx.mean(mx.abs(b))
        b_scaled = b / (alpha + 1e-7)
        b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
        b_ste = b + mx.stop_gradient(b_q - b)
        lora_out = (x @ self.lora_a) @ b_ste * self.scale
        return base_out + lora_out


class MultiAdapterLoRALinear(nn.Module):
    """LoRA with multiple A/B pairs for correct multi-expert composition."""
    def __init__(self, base_linear: nn.Linear, rank: int = 16,
                 scale: float = 20.0, a_inits: list = None):
        super().__init__()
        out_features = base_linear.weight.shape[0]
        self.linear = base_linear
        self.scale = scale
        self.rank = rank
        self.n_experts = len(a_inits) if a_inits else 0
        self.a_matrices = a_inits if a_inits else []
        self.b_matrices = [mx.zeros((rank, out_features)) for _ in range(self.n_experts)]
        self.linear.freeze()

    def __call__(self, x):
        base_out = self.linear(x)
        if self.n_experts == 0:
            return base_out
        lora_sum = mx.zeros_like(base_out)
        for i in range(self.n_experts):
            b = self.b_matrices[i]
            alpha = mx.mean(mx.abs(b))
            b_scaled = b / (alpha + 1e-7)
            b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
            b_ste = b + mx.stop_gradient(b_q - b)
            lora_sum = lora_sum + (x @ self.a_matrices[i]) @ b_ste
        return base_out + lora_sum * (self.scale / self.n_experts)


# ============================================================================
# BitNet utilities
# ============================================================================

from mlx_lm import load
from mlx_lm.models.bitlinear_layers import BitLinear


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
    return model


TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def apply_ternary_lora(model, rank, scale, a_matrices_per_layer):
    count = 0
    for li, layer in enumerate(model.model.layers):
        lora_updates = []
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue
            a_key = (li, key)
            if a_key in a_matrices_per_layer:
                a_np = a_matrices_per_layer[a_key]
                a_mx = mx.array(a_np).astype(mx.bfloat16)
            else:
                a_mx = None
            lora = TernaryLoRALinear(module, rank=rank, scale=scale, a_init=a_mx)
            lora_updates.append((key, lora))
            count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    return model


def get_trainable_params(model):
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        params[name] = mx.array(p)
    mx.eval(params)
    return params


def save_adapter(model, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    params = get_trainable_params(model)
    mx.savez(str(path / "adapter.npz"), **params)
    log(f"  Saved adapter: {len(params)} tensors to {path}")


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_weights(model, adapter_params: dict):
    model.update(tree_unflatten(list(adapter_params.items())))


def zero_b_params(model):
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


# ============================================================================
# Data loading
# ============================================================================

def format_instruction(instruction, response):
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"


def _download_and_parse_hf_file(repo_id, filename, file_format="parquet"):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id, filename, repo_type="dataset")
    if file_format == "parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict("records")
    elif file_format == "json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list):
                    return v
            return [data]
        return data
    elif file_format == "jsonl":
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    elif file_format == "csv":
        import pandas as pd
        df = pd.read_csv(path)
        return df.to_dict("records")
    else:
        raise ValueError(f"Unknown format: {file_format}")


def _extract_texts(rows, spec, max_samples):
    texts = []
    conv_format = spec.get("conv_format", None)
    inst_key = spec.get("inst_key", "instruction")
    resp_key = spec.get("resp_key", "output")
    offset = spec.get("offset", 0)
    max_take = spec.get("max_take", len(rows))
    rows = rows[offset:offset + max_take]
    for row in rows:
        try:
            if conv_format == "hf_conv":
                convs = row.get(inst_key, [])
                if isinstance(convs, str):
                    convs = json.loads(convs)
                if isinstance(convs, list) and len(convs) >= 2:
                    inst = None
                    resp = None
                    for msg in convs:
                        if isinstance(msg, dict):
                            role = msg.get("from", "")
                            val = msg.get("value", "")
                            if role == "human" and inst is None:
                                inst = str(val).strip()
                            elif role == "gpt" and resp is None:
                                resp = str(val).strip()
                        if inst and resp:
                            break
                    if not inst or not resp:
                        continue
                else:
                    continue
            else:
                inst = str(row.get(inst_key, "")).strip()
                resp = str(row.get(resp_key, "")).strip()
            if len(inst) > 5 and len(resp) > 10:
                texts.append(format_instruction(inst, resp))
            if len(texts) >= max_samples:
                break
        except (TypeError, KeyError, json.JSONDecodeError):
            continue
    return texts


def ensure_domain_data(domain_name):
    """Ensure data exists for a domain. Returns data_dir or None."""
    data_dir = DATA_DIR / domain_name
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "valid.jsonl"

    if train_path.exists() and val_path.exists():
        return data_dir

    # Try to copy from prior experiment
    for prior in [PRIOR_DATA_DIR, FIVE_DOMAIN_DIR / "data" if FIVE_DOMAIN_DIR.exists() else Path("/nonexistent")]:
        prev_train = prior / domain_name / "train.jsonl"
        prev_val = prior / domain_name / "valid.jsonl"
        if prev_train.exists() and prev_val.exists():
            import shutil
            data_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prev_train, train_path)
            shutil.copy2(prev_val, val_path)
            log(f"  {domain_name}: copied data from prior experiment")
            return data_dir

    # Download from HF
    spec = DATASET_SPECS.get(domain_name)
    if spec is None:
        log(f"  {domain_name}: no dataset spec, skipping")
        return None

    try:
        log(f"  {domain_name}: downloading from HF...")
        rows = _download_and_parse_hf_file(
            spec["hf_repo"], spec["hf_file"], spec.get("hf_format", "parquet")
        )
        texts = _extract_texts(rows, spec, 450)
        del rows
        gc.collect()

        if len(texts) < 30:
            log(f"  {domain_name}: only {len(texts)} samples, skipping")
            return None

        rng = random.Random(SEED + hash(domain_name) % 10000)
        rng.shuffle(texts)

        n_train = min(400, len(texts) - 10)
        n_val = min(50, len(texts) - n_train)

        data_dir.mkdir(parents=True, exist_ok=True)
        with open(train_path, "w") as f:
            for t in texts[:n_train]:
                json.dump({"text": t}, f)
                f.write("\n")
        with open(val_path, "w") as f:
            for t in texts[n_train:n_train + n_val]:
                json.dump({"text": t}, f)
                f.write("\n")

        log(f"  {domain_name}: {n_train} train, {n_val} val")
        return data_dir
    except Exception as e:
        log(f"  {domain_name}: data load failed: {e}")
        return None


def ensure_adapter(domain_name):
    """Ensure adapter exists on disk. Returns adapter_dir or None."""
    adapter_dir = ADAPTERS_DIR / domain_name
    if (adapter_dir / "adapter.npz").exists():
        return adapter_dir

    # Try to copy from prior experiment
    for prior in [PRIOR_ADAPTERS_DIR, FIVE_DOMAIN_DIR / "adapters" if FIVE_DOMAIN_DIR.exists() else Path("/nonexistent")]:
        prev = prior / domain_name / "adapter.npz"
        if prev.exists():
            import shutil
            adapter_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prev, adapter_dir / "adapter.npz")
            log(f"  {domain_name}: copied adapter from prior experiment")
            return adapter_dir

    return None


# ============================================================================
# PPL evaluation
# ============================================================================

def compute_ppl(model, tokenizer, data_dir: Path, max_batches: int = 25):
    valid_path = data_dir / "valid.jsonl"
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
        loss_val = loss.item()
        n_tok = y.size
        total_loss += loss_val
        total_tokens += n_tok
        del logits, loss, x, y
    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 100))


# ============================================================================
# Phase: Compute base PPL for all domains
# ============================================================================

def phase_base_ppl(domain_data_dirs):
    """Load model, compute base PPL on all domains."""
    log("\n[Phase: Base PPL]")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("post-load")

    base_ppls = {}
    for domain_name, data_dir in domain_data_dirs.items():
        ppl = compute_ppl(model, tokenizer, data_dir)
        base_ppls[domain_name] = round(ppl, 4)
        log(f"  {domain_name}: base PPL = {ppl:.2f}")

    cleanup(model, tokenizer)
    return base_ppls


# ============================================================================
# Phase: Generate Grassmannian skeleton for all N_max slots
# ============================================================================

# Compact skeleton: just {dim: (N, d, r) array}
# All layers share the same A matrix for a given domain and projection key.
# This avoids the 3150-entry dict that caused OOM.

def phase_skeleton():
    """Pre-compute Grassmannian A matrices for ALL_DOMAINS (N=15).

    Returns dict: {input_dim: np.array of shape (N, d, r)} -- compact.
    """
    N = len(ALL_DOMAINS)
    log(f"\n[Phase: Grassmannian Skeleton] N={N}, r={LORA_RANK}")

    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

    target_dims = {"default": 2560, "mlp.down_proj": 6912}
    frames_by_dim = {}

    for key in TARGET_KEYS:
        in_dim = target_dims.get(key, target_dims["default"])
        if in_dim in frames_by_dim:
            continue

        cache_path = ADAPTERS_DIR / f"frames_d{in_dim}_n{N}.npy"
        if cache_path.exists():
            log(f"  Loading cached frames for d={in_dim}")
            frames = np.load(str(cache_path))
        else:
            log(f"  AP for d={in_dim} (N={N}, r={LORA_RANK})...")
            t0 = time.time()
            frames = grassmannian_ap_init(N=N, r=LORA_RANK, d=in_dim, seed=SEED)
            log(f"    Done in {time.time()-t0:.1f}s")
            np.save(str(cache_path), frames)

        # Verify orthogonality
        cos_vals = []
        for i in range(N):
            for j in range(i + 1, N):
                cos = np.abs(np.trace(frames[i].T @ frames[j])) / LORA_RANK
                cos_vals.append(cos)
        mean_cos = np.mean(cos_vals) if cos_vals else 0
        max_cos = np.max(cos_vals) if cos_vals else 0
        log(f"    d={in_dim}: mean |cos|={mean_cos:.8f}, max |cos|={max_cos:.8f}")
        frames_by_dim[in_dim] = frames

    log(f"  Skeleton ready: {len(frames_by_dim)} dimension groups")
    return frames_by_dim


def get_a_matrix(frames_by_dim, key, domain_idx):
    """Get the A matrix for a given projection key and domain index."""
    target_dims = {"default": 2560, "mlp.down_proj": 6912}
    in_dim = target_dims.get(key, target_dims["default"])
    return frames_by_dim[in_dim][domain_idx]


# ============================================================================
# Phase: Train a single new adapter
# ============================================================================

def phase_train_adapter(domain_idx, domain_name, data_dir, skeleton):
    """Train one adapter. Returns training metrics."""
    log(f"\n[Training] {domain_name} (index {domain_idx})...")
    t0 = time.time()

    # Check if already trained
    existing = ensure_adapter(domain_name)
    if existing is not None:
        log(f"  Already trained, skipping")
        return {"train_time_s": 0, "skipped": True, "converged": True}

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    # Build A matrix mapping for this domain (same A for all layers)
    a_matrices = {}
    n_layers = len(model.model.layers)
    for li in range(n_layers):
        for key in TARGET_KEYS:
            a_matrices[(li, key)] = get_a_matrix(skeleton, key, domain_idx)

    model = apply_ternary_lora(model, LORA_RANK, LORA_SCALE, a_matrices)

    # Freeze everything except lora_b
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.unfreeze(keys=["lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"  Trainable params: {trainable:,}")

    # Load training data
    train_texts = []
    with open(data_dir / "train.jsonl") as f:
        for line in f:
            train_texts.append(json.loads(line)["text"])

    train_tokens = []
    for text in train_texts:
        toks = tokenizer.encode(text)
        if len(toks) > 2:
            train_tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))

    log(f"  {len(train_tokens)} training sequences")

    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    gc.disable()
    for step in range(TRAIN_ITERS):
        idx = step % len(train_tokens)
        tokens = train_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        loss_val = loss.item()
        losses.append(loss_val)
        if (step + 1) % 100 == 0 or step == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            log(f"    Step {step+1}/{TRAIN_ITERS}: loss={loss_val:.4f} (avg50={avg:.4f})")
    gc.enable()
    gc.collect()

    train_time = time.time() - t0
    first_50 = sum(losses[:50]) / 50
    last_50 = sum(losses[-50:]) / 50
    converged = last_50 < first_50 * 0.95

    save_adapter(model, ADAPTERS_DIR / domain_name)

    peak_gb = mx.get_peak_memory() / 1e9
    log(f"  Done in {train_time:.1f}s. Loss: {first_50:.4f} -> {last_50:.4f}")

    result = {
        "train_time_s": round(train_time, 1),
        "first_50_avg_loss": round(first_50, 4),
        "last_50_avg_loss": round(last_50, 4),
        "converged": converged,
        "trainable_params": trainable,
        "peak_memory_gb": round(peak_gb, 2),
    }

    cleanup(model, tokenizer, optimizer)
    return result


# ============================================================================
# Phase: Evaluate individual adapter PPL
# ============================================================================

def phase_eval_individuals_batch(domains_to_eval, domain_data_dirs, skeleton):
    """Evaluate multiple adapters individually, each on its own domain.

    Loads model ONCE, swaps A+B for each adapter. Returns {domain: ppl}.
    """
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    # Apply LoRA structure with domain_0 A matrices initially
    a_matrices = {}
    n_layers = len(model.model.layers)
    for li in range(n_layers):
        for key in TARGET_KEYS:
            a_matrices[(li, key)] = get_a_matrix(skeleton, key, 0)
    model = apply_ternary_lora(model, LORA_RANK, LORA_SCALE, a_matrices)
    model.freeze()

    results = {}
    for domain_name in domains_to_eval:
        di_global = ALL_DOMAINS.index(domain_name)

        # Swap A matrices for this domain
        for li in range(n_layers):
            for key in TARGET_KEYS:
                parts = key.split(".")
                module = model.model.layers[li]
                for part in parts:
                    module = getattr(module, part, None)
                    if module is None:
                        break
                if module is not None and isinstance(module, TernaryLoRALinear):
                    a_np = get_a_matrix(skeleton, key, di_global)
                    module.lora_a = mx.array(a_np).astype(mx.bfloat16)

        # Load B weights
        zero_b_params(model)
        params = load_adapter(ADAPTERS_DIR / domain_name)
        apply_adapter_weights(model, params)
        mx.eval(model.parameters())
        del params

        data_dir = domain_data_dirs[domain_name]
        ppl = compute_ppl(model, tokenizer, data_dir)
        results[domain_name] = round(ppl, 4)

    cleanup(model, tokenizer)
    return results


# ============================================================================
# Phase: Evaluate N-adapter composition on all current domains
# ============================================================================

def phase_eval_composition(current_domains, domain_data_dirs, base_ppls, skeleton):
    """Evaluate uniform 1/N composition of current adapter pool."""
    N = len(current_domains)
    log(f"\n[Composition Eval] N={N} adapters...")

    # Load all adapter B matrices
    all_adapters = []
    for domain_name in current_domains:
        params = load_adapter(ADAPTERS_DIR / domain_name)
        all_adapters.append(params)

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    n_layers = len(model.model.layers)
    count = 0
    for li, layer in enumerate(model.model.layers):
        lora_updates = []
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue

            # Get A matrices for current domains
            a_inits = []
            for di_local, domain_name in enumerate(current_domains):
                di_global = ALL_DOMAINS.index(domain_name)
                a_np = get_a_matrix(skeleton, key, di_global)
                a_mx = mx.array(a_np).astype(mx.bfloat16)
                a_inits.append(a_mx)

            if len(a_inits) != N:
                continue

            multi_lora = MultiAdapterLoRALinear(
                module, rank=LORA_RANK, scale=LORA_SCALE, a_inits=a_inits
            )

            param_name = f"model.layers.{li}.{key}.lora_b"
            for di_local in range(N):
                if param_name in all_adapters[di_local]:
                    multi_lora.b_matrices[di_local] = all_adapters[di_local][param_name]

            lora_updates.append((key, multi_lora))
            count += 1

        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    mx.eval(model.parameters())
    model.freeze()

    # Evaluate on all current domains
    composed_ppls = {}
    for domain_name in current_domains:
        data_dir = domain_data_dirs[domain_name]
        ppl = compute_ppl(model, tokenizer, data_dir)
        composed_ppls[domain_name] = round(ppl, 4)
        base = base_ppls.get(domain_name, float("inf"))
        delta = (ppl - base) / base * 100 if base != float("inf") else 0
        log(f"  {domain_name}: composed={ppl:.2f} (base={base:.2f}, {delta:+.1f}%)")

    cleanup(model, tokenizer)
    del all_adapters
    return composed_ppls


# ============================================================================
# Orthogonality analysis (adapter parameter vectors)
# ============================================================================

def compute_orthogonality(current_domains):
    """Compute pairwise cosine similarity between adapter parameter vectors."""
    adapters = {}
    for domain_name in current_domains:
        adapter_path = ADAPTERS_DIR / domain_name / "adapter.npz"
        if not adapter_path.exists():
            continue
        params = load_adapter(ADAPTERS_DIR / domain_name)
        vec = mx.concatenate([v.reshape(-1) for v in params.values()])
        mx.eval(vec)
        adapters[domain_name] = vec
        del params

    cosines = []
    available = [d for d in current_domains if d in adapters]
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            vi = adapters[available[i]]
            vj = adapters[available[j]]
            cos = mx.abs(
                mx.sum(vi * vj)
                / (mx.sqrt(mx.sum(vi**2)) * mx.sqrt(mx.sum(vj**2)) + 1e-10)
            )
            mx.eval(cos)
            cosines.append(cos.item())

    mean_cos = sum(cosines) / len(cosines) if cosines else 0
    max_cos = max(cosines) if cosines else 0
    del adapters
    gc.collect()
    return mean_cos, max_cos


# ============================================================================
# Main: Continual Growth Loop
# ============================================================================

def main():
    t_total = time.time()
    log("=" * 70)
    log("Continual Learning: Adapter Pool Growth (N=5 -> N=15)")
    log(f"  Initial: {INITIAL_DOMAINS}")
    log(f"  Growth: {GROWTH_DOMAINS}")
    log("=" * 70)
    log_memory("start")

    results = {
        "experiment": "continual_learning_adapter_growth",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "train_iters": TRAIN_ITERS,
        "initial_domains": INITIAL_DOMAINS,
        "growth_domains": GROWTH_DOMAINS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ----------------------------------------------------------------
    # Step 0: Prepare all data upfront
    # ----------------------------------------------------------------
    log("\n[Step 0] Preparing data for all domains...")
    domain_data_dirs = {}
    for domain in ALL_DOMAINS:
        data_dir = ensure_domain_data(domain)
        if data_dir is not None:
            domain_data_dirs[domain] = data_dir
        else:
            log(f"  WARNING: Could not get data for {domain}")

    available_initial = [d for d in INITIAL_DOMAINS if d in domain_data_dirs]
    available_growth = [d for d in GROWTH_DOMAINS if d in domain_data_dirs]
    log(f"  Available: {len(available_initial)} initial, {len(available_growth)} growth")

    if len(available_initial) < 3:
        log("FATAL: Need at least 3 initial domains")
        return

    # ----------------------------------------------------------------
    # Step 1: Base PPL (once for all domains)
    # ----------------------------------------------------------------
    base_ppls = phase_base_ppl(domain_data_dirs)
    results["base_ppls"] = base_ppls
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    # ----------------------------------------------------------------
    # Step 2: Grassmannian skeleton (once for N=15)
    # ----------------------------------------------------------------
    skeleton = phase_skeleton()

    # ----------------------------------------------------------------
    # Step 3: Ensure initial 5 adapters exist
    # ----------------------------------------------------------------
    log("\n[Step 3] Ensuring initial adapters exist...")
    train_results = {}
    for di, domain_name in enumerate(available_initial):
        existing = ensure_adapter(domain_name)
        if existing is not None:
            train_results[domain_name] = {"skipped": True, "reused": True}
            log(f"  {domain_name}: reused from prior experiment")
        else:
            di_global = ALL_DOMAINS.index(domain_name)
            tr = phase_train_adapter(di_global, domain_name, domain_data_dirs[domain_name], skeleton)
            train_results[domain_name] = tr

    # ----------------------------------------------------------------
    # Step 4: Baseline measurement at N=5 (before any growth)
    # ----------------------------------------------------------------
    log("\n" + "=" * 70)
    log("[CYCLE 0] Baseline measurement at N=5")
    log("=" * 70)

    current_domains = list(available_initial)
    cycles = []

    # Evaluate initial individual PPLs (single model load for all)
    log("\n  Evaluating individual adapters at N=5...")
    initial_individual_ppls = phase_eval_individuals_batch(
        current_domains, domain_data_dirs, skeleton
    )
    for domain_name in current_domains:
        ppl = initial_individual_ppls[domain_name]
        base = base_ppls.get(domain_name, float("inf"))
        imp = (base - ppl) / base * 100
        log(f"    {domain_name}: PPL={ppl:.2f} (base={base:.2f}, {imp:+.1f}%)")

    # Evaluate initial composition
    initial_composed = phase_eval_composition(
        current_domains, domain_data_dirs, base_ppls, skeleton
    )

    # Orthogonality
    mean_cos, max_cos = compute_orthogonality(current_domains)
    log(f"  Orthogonality: mean |cos|={mean_cos:.6f}, max |cos|={max_cos:.6f}")

    # Mean composition improvement
    comp_improvements = []
    for d in current_domains:
        if d in initial_composed and d in base_ppls:
            imp = (base_ppls[d] - initial_composed[d]) / base_ppls[d] * 100
            comp_improvements.append(imp)
    mean_comp_imp = sum(comp_improvements) / len(comp_improvements) if comp_improvements else 0

    cycle_0 = {
        "cycle": 0,
        "N": len(current_domains),
        "domains": list(current_domains),
        "added_domain": None,
        "individual_ppls": initial_individual_ppls,
        "composed_ppls": initial_composed,
        "mean_composition_improvement_pct": round(mean_comp_imp, 2),
        "mean_abs_cos": round(mean_cos, 6),
        "max_abs_cos": round(max_cos, 6),
        "k1_max_degradation_pct": 0.0,
        "k1_degraded_domain": None,
    }
    cycles.append(cycle_0)
    results["cycles"] = cycles
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    # ----------------------------------------------------------------
    # Step 5: Growth cycles (add 1 adapter per cycle)
    # ----------------------------------------------------------------
    # NOTE: Individual adapter PPL is INVARIANT across cycles because
    # both A_i (Grassmannian, frozen) and B_i (trained once, frozen after)
    # never change. K1 forgetting check is on COMPOSITION quality per
    # domain, not individual adapter quality.
    # We evaluate new adapter individually, then composition for all domains.
    # Final individual check at N=15 confirms invariance.

    reference_composed_ppls = dict(initial_composed)  # track composition changes

    for cycle_idx, new_domain in enumerate(available_growth, start=1):
        log(f"\n{'=' * 70}")
        log(f"[CYCLE {cycle_idx}] Adding '{new_domain}' (N={len(current_domains)} -> {len(current_domains)+1})")
        log(f"{'=' * 70}")

        # Train new adapter
        di_global = ALL_DOMAINS.index(new_domain)
        tr = phase_train_adapter(
            di_global, new_domain, domain_data_dirs[new_domain], skeleton
        )
        train_results[new_domain] = tr

        # Add to pool
        current_domains.append(new_domain)
        N_current = len(current_domains)

        # Composition evaluation (the key metric)
        composed_ppls = phase_eval_composition(
            current_domains, domain_data_dirs, base_ppls, skeleton
        )

        # Track composition changes for INITIAL domains (K1 proxy)
        # Since individual PPL is invariant, we use per-domain composition
        # quality as the forgetting signal: does adding a new adapter
        # make composition WORSE for existing domains?
        max_degradation = 0.0
        degraded_domain = None
        for existing_domain in current_domains[:-1]:
            if existing_domain in composed_ppls and existing_domain in reference_composed_ppls:
                ref_comp = reference_composed_ppls[existing_domain]
                new_comp = composed_ppls[existing_domain]
                if ref_comp > 0:
                    degradation = (new_comp - ref_comp) / ref_comp * 100
                else:
                    degradation = 0.0
                if degradation > max_degradation:
                    max_degradation = degradation
                    degraded_domain = existing_domain

        # Update reference for new domain
        if new_domain in composed_ppls:
            reference_composed_ppls[new_domain] = composed_ppls[new_domain]

        # Orthogonality
        mean_cos, max_cos = compute_orthogonality(current_domains)
        log(f"  Orthogonality: mean |cos|={mean_cos:.6f}, max |cos|={max_cos:.6f}")

        # Mean composition improvement vs base
        comp_improvements = []
        for d in current_domains:
            if d in composed_ppls and d in base_ppls:
                imp = (base_ppls[d] - composed_ppls[d]) / base_ppls[d] * 100
                comp_improvements.append(imp)
        mean_comp_imp = sum(comp_improvements) / len(comp_improvements) if comp_improvements else 0

        cycle_data = {
            "cycle": cycle_idx,
            "N": N_current,
            "domains": list(current_domains),
            "added_domain": new_domain,
            "composed_ppls": composed_ppls,
            "mean_composition_improvement_pct": round(mean_comp_imp, 2),
            "mean_abs_cos": round(mean_cos, 6),
            "max_abs_cos": round(max_cos, 6),
            "k1_max_composition_degradation_pct": round(max_degradation, 4),
            "k1_degraded_domain": degraded_domain,
            "train_result": tr,
        }
        cycles.append(cycle_data)

        # Save progress after each cycle
        results["cycles"] = cycles
        results["train_results"] = train_results
        RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

        log(f"\n  Cycle {cycle_idx} summary: N={N_current}, "
            f"comp_imp={mean_comp_imp:+.1f}%, "
            f"max_comp_degradation={max_degradation:+.2f}%, "
            f"mean_cos={mean_cos:.6f}")

    # ----------------------------------------------------------------
    # Step 6: Final individual adapter check (confirm invariance)
    # ----------------------------------------------------------------
    log("\n" + "=" * 70)
    log("[Final Check] Re-evaluating initial 5 adapters individually at N=15")
    log("=" * 70)
    final_individual_ppls = phase_eval_individuals_batch(
        available_initial, domain_data_dirs, skeleton
    )
    log("\n  Individual PPL comparison (initial vs final):")
    max_individual_drift = 0.0
    for d in available_initial:
        init = initial_individual_ppls.get(d, 0)
        final = final_individual_ppls.get(d, 0)
        drift = abs(final - init) / init * 100 if init > 0 else 0
        if drift > max_individual_drift:
            max_individual_drift = drift
        log(f"    {d}: initial={init:.4f}, final={final:.4f}, drift={drift:.4f}%")
    results["final_individual_ppls"] = final_individual_ppls
    results["max_individual_drift_pct"] = round(max_individual_drift, 4)

    # ----------------------------------------------------------------
    # Kill Criteria Assessment
    # ----------------------------------------------------------------
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    # K1: Any existing adapter degrades > 5% after new addition
    # We measure this two ways:
    # (a) Individual PPL drift (should be ~0% by construction)
    # (b) Per-domain composition quality degradation across cycles
    k1_pass = True
    k1_worst_degradation = 0.0
    k1_worst_domain = None
    k1_worst_cycle = None
    for c in cycles[1:]:  # skip cycle 0
        deg = c["k1_max_composition_degradation_pct"]
        if deg > k1_worst_degradation:
            k1_worst_degradation = deg
            k1_worst_domain = c["k1_degraded_domain"]
            k1_worst_cycle = c["cycle"]
        if deg > 5.0:
            k1_pass = False

    # Also check individual drift
    if max_individual_drift > 5.0:
        k1_pass = False
        log(f"  K1 FAIL: individual PPL drift {max_individual_drift:.2f}% > 5%")

    results["k1_pass"] = k1_pass
    results["k1_worst_composition_degradation_pct"] = round(k1_worst_degradation, 4)
    results["k1_worst_domain"] = k1_worst_domain
    results["k1_worst_cycle"] = k1_worst_cycle
    log(f"\n  K1 (no adapter degrades >5%):")
    log(f"    Worst composition degradation: {k1_worst_degradation:+.2f}% "
        f"({k1_worst_domain}, cycle {k1_worst_cycle})")
    log(f"    Individual PPL drift: {max_individual_drift:.4f}%")
    log(f"    -> {'PASS' if k1_pass else 'FAIL (KILL)'}")

    # K2: Composition quality degrades monotonically with N
    comp_improvements_by_cycle = [c["mean_composition_improvement_pct"] for c in cycles]
    # Monotonic decrease = every step is worse than the previous
    monotonic_decrease = True
    for i in range(1, len(comp_improvements_by_cycle)):
        if comp_improvements_by_cycle[i] >= comp_improvements_by_cycle[i - 1]:
            monotonic_decrease = False
            break

    k2_pass = not monotonic_decrease
    results["k2_pass"] = k2_pass
    results["k2_composition_trajectory"] = comp_improvements_by_cycle
    log(f"\n  K2 (composition not monotonically decreasing):")
    for c in cycles:
        log(f"    N={c['N']}: mean composition improvement = {c['mean_composition_improvement_pct']:+.2f}%")
    log(f"    Monotonic decrease: {monotonic_decrease} -> {'FAIL (KILL)' if monotonic_decrease else 'PASS'}")

    # Summary
    results["total_time_s"] = round(time.time() - t_total, 1)
    all_pass = k1_pass and k2_pass
    results["all_kill_criteria_pass"] = all_pass
    verdict = "SUPPORTED" if all_pass else "KILLED"
    results["verdict"] = verdict

    log(f"\n  Overall verdict: {verdict}")
    log(f"  Total time: {results['total_time_s']:.0f}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
