#!/usr/bin/env python3
"""Real Data Domain Experts: Train 5 domain LoRA adapters on real HF data.

Tests whether ternary adapters with Grassmannian init specialize on real
instruction data and compose without degradation.

Kill criteria:
  K1 (id=231): Adapters don't specialize (PPL same as base on target domain)
  K2 (id=232): Composition degrades majority of domains (>3/5 worse than base)

Success criteria:
  Each adapter improves its target domain PPL > 5% and composition doesn't
  degrade >3/5 domains vs base.

Architecture:
  Base: microsoft/BitNet-b1.58-2B-4T (ternary, d=2560, 30 layers)
  LoRA: rank-16, Grassmannian AP init for A (frozen), STE ternary B
  Training: instruction format, 200 iters/adapter, seq_len=256
  Composition: 1/N scaling, plus per-adapter routing heads

Platform: Apple M5 Pro 48GB, MLX 0.31.1
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
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

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

# 5 domains — max samples config
DOMAINS = {
    "medical": {"max_train": 400, "max_val": 50},
    "code": {"max_train": 400, "max_val": 50},
    "math": {"max_train": 400, "max_val": 50},
    "legal": {"max_train": 400, "max_val": 50},
    "finance": {"max_train": 400, "max_val": 50},
}

DOMAIN_NAMES = list(DOMAINS.keys())
N_DOMAINS = len(DOMAIN_NAMES)


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
# Grassmannian AP Init (numpy, then convert to MLX)
# ============================================================================

def grassmannian_ap_init(N, r, d, n_iters=300, seed=42):
    """Generate N orthogonally-packed A matrices via Alternating Projection.

    Returns (N, d, r) numpy array of orthonormal frames on Gr(r, d).
    """
    rng = np.random.RandomState(seed)

    # Generate random initial frames
    frames = np.zeros((N, d, r), dtype=np.float32)
    for i in range(N):
        M = rng.randn(d, r).astype(np.float32)
        Q, _ = np.linalg.qr(M)
        frames[i] = Q[:, :r]

    # Welch bound target coherence
    Nr = N * r
    if Nr <= d:
        # Perfect orthogonality possible
        return frames
    mu_target = np.sqrt(r * (Nr - d) / (d * (Nr - r)))
    # Relax slightly above Welch bound
    mu_target = mu_target * 1.2

    for iteration in range(n_iters):
        # Build Gram matrix (Nr x Nr)
        G = np.zeros((Nr, Nr), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                G[i*r:(i+1)*r, j*r:(j+1)*r] = frames[i].T @ frames[j]

        # Structural projection: cap off-diagonal block norms
        for i in range(N):
            for j in range(N):
                if i != j:
                    block = G[i*r:(i+1)*r, j*r:(j+1)*r]
                    norm = np.linalg.norm(block, 'fro')
                    if norm > mu_target:
                        G[i*r:(i+1)*r, j*r:(j+1)*r] = block * (mu_target / norm)
                else:
                    G[i*r:(i+1)*r, j*r:(j+1)*r] = np.eye(r, dtype=np.float32)

        # Spectral projection: nearest valid Gram matrix
        eigvals, eigvecs = np.linalg.eigh(G)
        # Keep top-d eigenvalues, clamp to >= 0
        eigvals_clamped = np.maximum(eigvals, 0)
        if d < Nr:
            idx_sort = np.argsort(eigvals_clamped)[::-1]
            eigvals_clamped[idx_sort[d:]] = 0
            eigvecs = eigvecs[:, idx_sort]
            eigvals_clamped = eigvals_clamped[idx_sort]

        # Reconstruct and rescale
        G = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T
        # Rescale to trace = N*r
        trace = np.trace(G)
        if trace > 1e-8:
            G = G * (N * r / trace)

        # Extract frames from Gram matrix via Cholesky-like factorization
        eigvals2, eigvecs2 = np.linalg.eigh(G)
        idx = np.argsort(eigvals2)[::-1]
        eigvals2 = eigvals2[idx]
        eigvecs2 = eigvecs2[:, idx]

        # Take top-d components
        k = min(d, Nr)
        sqrt_eig = np.sqrt(np.maximum(eigvals2[:k], 0))
        factor = eigvecs2[:, :k] * sqrt_eig[None, :]  # (Nr, k)

        # Extract frames
        for i in range(N):
            block = factor[i*r:(i+1)*r, :d]  # (r, d) -> transpose to (d, r)
            if block.shape[1] < d:
                block = np.pad(block, ((0, 0), (0, d - block.shape[1])))
            # Re-orthogonalize via QR
            Q, _ = np.linalg.qr(block.T)
            frames[i] = Q[:, :r]

    return frames


# ============================================================================
# STE Ternary LoRA Layer
# ============================================================================

class TernaryLoRALinear(nn.Module):
    """LoRA with frozen Grassmannian A and STE-ternary B.

    Forward: y = base(x) + (x @ A) @ quantize(B) * scale
    Backward: STE passes gradients through quantization of B.
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 16,
                 scale: float = 20.0, a_init: mx.array = None):
        super().__init__()
        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]

        self.linear = base_linear

        # A matrix: frozen, from Grassmannian or random
        if a_init is not None:
            self.lora_a = a_init  # (in_features, rank)
        else:
            s = 1.0 / math.sqrt(in_features)
            self.lora_a = mx.random.uniform(
                low=-s, high=s, shape=(in_features, rank)
            )

        # B matrix: trainable, STE-ternary quantized in forward
        self.lora_b = mx.zeros((rank, out_features))
        self.scale = scale
        self.rank = rank

        # Freeze base and A
        self.linear.freeze()
        self.freeze(keys=["lora_a"], strict=False)

    def __call__(self, x):
        base_out = self.linear(x)

        # STE ternary quantization on B
        b = self.lora_b
        alpha = mx.mean(mx.abs(b))
        b_scaled = b / (alpha + 1e-7)
        b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
        # STE: forward uses b_q, backward passes through b
        b_ste = b + mx.stop_gradient(b_q - b)

        # LoRA forward: x @ A @ B_ternary * scale
        lora_out = (x @ self.lora_a) @ b_ste * self.scale

        return base_out + lora_out


# ============================================================================
# Routing Head (from tiny_routing_heads)
# ============================================================================

class RoutingHead(nn.Module):
    """Tiny binary classifier: is this input from my domain?"""
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def __call__(self, x):
        h = nn.relu(self.fc1(x))
        return self.fc2(h)


# ============================================================================
# BitNet unpacking and model utilities
# ============================================================================

from mlx_lm import load
from mlx_lm.models.bitlinear_layers import BitLinear


def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    """Unpack uint8-packed ternary weights to bfloat16."""
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
    """Replace BitLinear with nn.Linear for differentiable training."""
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


def apply_ternary_lora(model, rank, scale, a_matrices_per_layer):
    """Apply TernaryLoRALinear to all target projection layers.

    a_matrices_per_layer: dict mapping (layer_idx, key) -> (d, r) numpy array
    """
    target_keys = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]

    count = 0
    for li, layer in enumerate(model.model.layers):
        lora_updates = []
        for key in target_keys:
            # Navigate to module
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue

            # Get Grassmannian A init for this layer+key
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

    log(f"  Applied TernaryLoRA (r={rank}) to {count} layers")
    return model


def get_trainable_params(model):
    """Get dict of trainable (lora_b) parameters."""
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        params[name] = mx.array(p)
    mx.eval(params)
    return params


def save_adapter(model, path: Path):
    """Save trainable LoRA B parameters."""
    path.mkdir(parents=True, exist_ok=True)
    params = get_trainable_params(model)
    mx.savez(str(path / "adapter.npz"), **params)
    log(f"  Saved adapter: {len(params)} tensors to {path}")


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_weights(model, adapter_params: dict):
    """Load adapter B weights into model."""
    model.update(tree_unflatten(list(adapter_params.items())))


def zero_b_params(model):
    """Reset all lora_b to zeros."""
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


# ============================================================================
# Data preparation
# ============================================================================

def format_instruction(instruction, response):
    """Format as instruction-response pair."""
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"


def _download_and_parse_hf(repo_id, filename, file_format="parquet"):
    """Download a single file from HuggingFace Hub and return list of dicts."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id, filename, repo_type="dataset")

    if file_format == "parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict("records")
    elif file_format == "json":
        with open(path) as f:
            return json.load(f)
    elif file_format == "jsonl":
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        raise ValueError(f"Unknown format: {file_format}")


# Dataset download specs (avoiding broken `datasets` library on Python 3.14)
HF_DOWNLOAD_SPECS = {
    "medical": {
        "repo": "medalpaca/medical_meadow_medical_flashcards",
        "file": "medical_meadow_wikidoc_medical_flashcards.json",
        "format": "json",
        "inst_key": "input",
        "resp_key": "output",
    },
    "code": {
        "repo": "iamtarun/python_code_instructions_18k_alpaca",
        "file": "data/train-00000-of-00001-8b6e212f3e1ece96.parquet",
        "format": "parquet",
        "inst_key": "instruction",
        "resp_key": "output",
    },
    "math": {
        "repo": "openai/gsm8k",
        "file": "main/train-00000-of-00001.parquet",
        "format": "parquet",
        "inst_key": "question",
        "resp_key": "answer",
    },
    "legal": {
        "repo": "jonathanli/law-stack-exchange",
        "file": "train.jsonl",
        "format": "jsonl",
        "inst_key": "title",
        "resp_key": "body",
    },
    "finance": {
        "repo": "gbharti/finance-alpaca",
        "file": "Cleaned_date.json",
        "format": "json",
        "inst_key": "instruction",
        "resp_key": "output",
    },
}


def phase_prepare_data():
    """Download and prepare instruction-format data for all 5 domains."""
    log("\n[Phase 1] Preparing domain data...")

    domain_data = {}

    for domain_name in DOMAIN_NAMES:
        train_path = DATA_DIR / domain_name / "train.jsonl"
        val_path = DATA_DIR / domain_name / "valid.jsonl"

        if train_path.exists() and val_path.exists():
            log(f"  {domain_name}: data already exists, skipping download")
            domain_data[domain_name] = DATA_DIR / domain_name
            continue

        (DATA_DIR / domain_name).mkdir(parents=True, exist_ok=True)
        config = DOMAINS[domain_name]
        spec = HF_DOWNLOAD_SPECS[domain_name]

        log(f"  Downloading {spec['repo']}/{spec['file']}...")
        rows = _download_and_parse_hf(spec["repo"], spec["file"], spec["format"])

        inst_key = spec["inst_key"]
        resp_key = spec["resp_key"]
        log(f"  {domain_name}: {len(rows)} raw rows, columns: {inst_key}, {resp_key}")

        # Extract and format as instruction pairs
        texts = []
        for row in rows:
            inst = str(row.get(inst_key, "")).strip()
            resp = str(row.get(resp_key, "")).strip()
            if len(inst) > 5 and len(resp) > 10:
                formatted = format_instruction(inst, resp)
                texts.append(formatted)
            if len(texts) >= config["max_train"] + config["max_val"]:
                break

        if len(texts) < config["max_val"] + 10:
            raise ValueError(f"Not enough samples for {domain_name}: got {len(texts)}")

        # Shuffle with seed for reproducibility
        rng = random.Random(SEED)
        rng.shuffle(texts)

        train_texts = texts[:config["max_train"]]
        val_texts = texts[config["max_train"]:config["max_train"] + config["max_val"]]

        with open(train_path, "w") as f:
            for t in train_texts:
                json.dump({"text": t}, f)
                f.write("\n")

        with open(val_path, "w") as f:
            for t in val_texts:
                json.dump({"text": t}, f)
                f.write("\n")

        log(f"  {domain_name}: {len(train_texts)} train, {len(val_texts)} val")
        domain_data[domain_name] = DATA_DIR / domain_name

    return domain_data


# ============================================================================
# PPL evaluation
# ============================================================================

def compute_ppl(model, tokenizer, data_path: Path, max_batches: int = 25,
                return_per_sample: bool = False):
    """Compute perplexity on validation data.

    If return_per_sample=True, also returns list of (loss_sum, n_tokens) per sample
    for bootstrap CI computation.
    """
    valid_path = data_path / "valid.jsonl"
    if not valid_path.exists():
        if return_per_sample:
            return float("inf"), []
        return float("inf")

    texts = []
    with open(valid_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    total_loss = 0.0
    total_tokens = 0
    per_sample = []  # list of (loss_sum, n_tokens)

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
        per_sample.append((loss_val, n_tok))
        del logits, loss, x, y

    if total_tokens == 0:
        if return_per_sample:
            return float("inf"), []
        return float("inf")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 100))

    if return_per_sample:
        return ppl, per_sample
    return ppl


def bootstrap_ppl_ci(per_sample, n_resamples=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for PPL from per-sample losses.

    per_sample: list of (loss_sum, n_tokens) tuples
    Returns: (ppl_mean, ppl_lo, ppl_hi)
    """
    if not per_sample:
        return float("inf"), float("inf"), float("inf")

    rng = np.random.RandomState(seed)
    n = len(per_sample)
    losses = np.array([s[0] for s in per_sample])
    tokens = np.array([s[1] for s in per_sample])

    # Point estimate
    ppl_point = math.exp(min(losses.sum() / tokens.sum(), 100))

    # Bootstrap
    ppls = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        boot_loss = losses[idx].sum()
        boot_tok = tokens[idx].sum()
        if boot_tok > 0:
            ppls.append(math.exp(min(boot_loss / boot_tok, 100)))

    ppls = np.array(ppls)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(ppls, alpha * 100))
    hi = float(np.percentile(ppls, (1 - alpha) * 100))

    return ppl_point, lo, hi


# ============================================================================
# Phase 2: Compute base PPL
# ============================================================================

def phase_base_ppl(domain_data):
    """Load model, compute base PPL on all domains, return results."""
    log("\n[Phase 2] Loading model and computing base PPL...")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    log(f"  Loaded in {time.time()-t0:.1f}s")

    model = replace_bitlinear_with_linear(model)
    log_memory("post-unpack")

    base_ppls = {}
    for domain_name, data_dir in domain_data.items():
        ppl = compute_ppl(model, tokenizer, data_dir)
        base_ppls[domain_name] = ppl
        log(f"  {domain_name}: base PPL = {ppl:.2f}")

    cleanup(model, tokenizer)
    return base_ppls


# ============================================================================
# Phase 3: Generate Grassmannian skeleton
# ============================================================================

def phase_grassmannian_skeleton():
    """Pre-compute Grassmannian A matrices for all layers x projections."""
    log("\n[Phase 3] Computing Grassmannian AP skeleton...")

    skeleton_path = ADAPTERS_DIR / "grassmannian_skeleton.npz"
    if skeleton_path.exists():
        log("  Skeleton already exists, loading from disk")
        data = dict(np.load(str(skeleton_path)))
        return data

    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

    # Target projections with their dimensions
    # BitNet-2B-4T: d=2560, intermediate=6912
    # q/k/v/o: 2560 -> 2560
    # gate/up: 2560 -> 6912
    # down: 6912 -> 2560
    target_configs = {
        "self_attn.q_proj": (2560, 2560),
        "self_attn.k_proj": (2560, 2560),
        "self_attn.v_proj": (2560, 2560),
        "self_attn.o_proj": (2560, 2560),
        "mlp.gate_proj": (2560, 6912),
        "mlp.up_proj": (2560, 6912),
        "mlp.down_proj": (6912, 2560),
    }

    # For N=5 experts, r=16: N*r = 80 << d=2560, so perfect orth is possible.
    # AP will just return QR-orthogonalized frames (no cap needed).
    # Still run AP for the formal skeleton.
    skeleton = {}
    n_layers = 30

    # Group projections by input dimension to batch the AP calls
    for key, (in_dim, out_dim) in target_configs.items():
        log(f"  AP for {key} (in={in_dim})...")
        # One AP call per unique key (same across layers since dims are same)
        frames = grassmannian_ap_init(
            N=N_DOMAINS, r=LORA_RANK, d=in_dim, n_iters=100, seed=SEED
        )
        # frames: (N_DOMAINS, in_dim, rank)
        for li in range(n_layers):
            for di in range(N_DOMAINS):
                skey = f"layer_{li}_{key}_domain_{di}"
                skeleton[skey] = frames[di]  # (in_dim, rank)

    np.savez(str(skeleton_path), **skeleton)
    log(f"  Saved skeleton: {len(skeleton)} matrices")
    return skeleton


# ============================================================================
# Phase 4: Train domain adapters
# ============================================================================

def phase_train_adapter(domain_idx, domain_name, domain_data, skeleton):
    """Train a single domain adapter with Grassmannian init + STE ternary B."""
    log(f"\n[Phase 4.{domain_idx}] Training {domain_name} adapter...")
    t0 = time.time()

    # Load model fresh for each adapter
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    # Build A matrix mapping for this domain
    a_matrices = {}
    n_layers = len(model.model.layers)
    target_keys = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]
    for li in range(n_layers):
        for key in target_keys:
            skey = f"layer_{li}_{key}_domain_{domain_idx}"
            if skey in skeleton:
                a_matrices[(li, key)] = skeleton[skey]

    # Apply TernaryLoRA
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
    data_dir = domain_data[domain_name]
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

    # Training loop
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

        if (step + 1) % 50 == 0 or step == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            log(f"    Step {step+1}/{TRAIN_ITERS}: loss={loss_val:.4f} (avg50={avg:.4f})")

    gc.enable()
    gc.collect()

    train_time = time.time() - t0
    first_50 = sum(losses[:50]) / 50
    last_50 = sum(losses[-50:]) / 50
    converged = last_50 < first_50 * 0.95

    log(f"  Done in {train_time:.1f}s. Loss: {first_50:.4f} -> {last_50:.4f} "
        f"({'converged' if converged else 'NOT converged'})")

    # Save adapter
    save_adapter(model, ADAPTERS_DIR / domain_name)

    result = {
        "train_time_s": round(train_time, 1),
        "first_50_avg_loss": round(first_50, 4),
        "last_50_avg_loss": round(last_50, 4),
        "converged": converged,
        "trainable_params": trainable,
    }

    log_memory(f"post-train-{domain_name}")
    cleanup(model, tokenizer, optimizer)
    return result


# ============================================================================
# Phase 5: Evaluate individual + composed adapters
# ============================================================================

class MultiAdapterLoRALinear(nn.Module):
    """LoRA with multiple A/B pairs for correct multi-expert composition.

    Forward: y = base(x) + (1/N) * sum_i[(x @ A_i) @ ternary(B_i)] * scale

    Each expert i has its own A_i (frozen) and B_i (loaded from saved adapter).
    This is the mathematically correct composition that respects per-expert
    subspace projections.
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 16,
                 scale: float = 20.0, a_inits: list = None):
        super().__init__()
        out_features = base_linear.weight.shape[0]
        self.linear = base_linear
        self.scale = scale
        self.rank = rank
        self.n_experts = len(a_inits) if a_inits else 0

        # Store A matrices (all frozen)
        self.a_matrices = a_inits if a_inits else []

        # B matrices: one per expert, all zeros initially (will be loaded)
        self.b_matrices = [mx.zeros((rank, out_features)) for _ in range(self.n_experts)]

        # Freeze base
        self.linear.freeze()

    def __call__(self, x):
        base_out = self.linear(x)

        if self.n_experts == 0:
            return base_out

        # Correct multi-expert composition:
        # y = base(x) + (1/N) * sum_i[(x @ A_i) @ ternary(B_i)] * scale
        lora_sum = mx.zeros_like(base_out)
        for i in range(self.n_experts):
            b = self.b_matrices[i]
            alpha = mx.mean(mx.abs(b))
            b_scaled = b / (alpha + 1e-7)
            b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
            b_ste = b + mx.stop_gradient(b_q - b)
            lora_sum = lora_sum + (x @ self.a_matrices[i]) @ b_ste

        return base_out + lora_sum * (self.scale / self.n_experts)


def _set_lora_a(model, skeleton, domain_idx, n_layers, target_keys):
    """Set A matrices in TernaryLoRALinear modules from skeleton for a given domain."""
    for li in range(n_layers):
        for key in target_keys:
            skey = f"layer_{li}_{key}_domain_{domain_idx}"
            if skey in skeleton:
                a_np = skeleton[skey]
                a_mx = mx.array(a_np).astype(mx.bfloat16)
                parts = key.split(".")
                module = model.model.layers[li]
                for part in parts:
                    module = getattr(module, part)
                if isinstance(module, TernaryLoRALinear):
                    module.lora_a = a_mx


def phase_evaluate(domain_data, base_ppls):
    """Evaluate individual and composed adapter PPL.

    Runs three evaluations:
    1. Individual adapters (each with correct A_i + B_i)
    2. BROKEN composition: single A_0 + averaged B (the old bug, for comparison)
    3. CORRECT composition: sum of per-expert (A_i @ B_i) / N

    All PPL measurements include bootstrap 95% CI.
    """
    log("\n[Phase 5] Evaluating adapters...")
    t0 = time.time()

    skeleton_path = ADAPTERS_DIR / "grassmannian_skeleton.npz"
    skeleton = dict(np.load(str(skeleton_path)))

    target_keys = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]

    # ----------------------------------------------------------------
    # 5a: Base PPL with bootstrap CI
    # ----------------------------------------------------------------
    log("\n  [5a] Base PPL with bootstrap CI...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    base_ppls_ci = {}
    for domain_name, data_dir in domain_data.items():
        ppl, per_sample = compute_ppl(model, tokenizer, data_dir, return_per_sample=True)
        _, lo, hi = bootstrap_ppl_ci(per_sample)
        base_ppls_ci[domain_name] = {"ppl": ppl, "ci_lo": round(lo, 2), "ci_hi": round(hi, 2)}
        log(f"  {domain_name}: base PPL = {ppl:.2f} [{lo:.2f}, {hi:.2f}]")

    cleanup(model, tokenizer)

    # ----------------------------------------------------------------
    # 5b: Individual adapter PPL with bootstrap CI
    # ----------------------------------------------------------------
    log("\n  [5b] Individual adapter PPL with bootstrap CI...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    # Apply LoRA structure with domain_0 A matrices initially
    a_matrices = {}
    n_layers = len(model.model.layers)
    for li in range(n_layers):
        for key in target_keys:
            skey = f"layer_{li}_{key}_domain_0"
            if skey in skeleton:
                a_matrices[(li, key)] = skeleton[skey]
    model = apply_ternary_lora(model, LORA_RANK, LORA_SCALE, a_matrices)
    model.freeze()

    individual_ppls = {}
    individual_ppls_ci = {}
    for di, domain_name in enumerate(DOMAIN_NAMES):
        # Set correct A matrices for this domain
        _set_lora_a(model, skeleton, di, n_layers, target_keys)

        # Load B weights
        adapter_path = ADAPTERS_DIR / domain_name
        params = load_adapter(adapter_path)
        zero_b_params(model)
        apply_adapter_weights(model, params)
        mx.eval(model.parameters())

        ppl, per_sample = compute_ppl(model, tokenizer, domain_data[domain_name],
                                       return_per_sample=True)
        _, lo, hi = bootstrap_ppl_ci(per_sample)
        individual_ppls[domain_name] = ppl
        individual_ppls_ci[domain_name] = {"ppl": ppl, "ci_lo": round(lo, 2), "ci_hi": round(hi, 2)}
        base = base_ppls[domain_name]
        imp = (base - ppl) / base * 100
        log(f"  {domain_name}: PPL={ppl:.2f} [{lo:.2f}, {hi:.2f}] (base={base:.2f}, {imp:+.1f}%)")

    # ----------------------------------------------------------------
    # 5c: BROKEN composition (single A_0 + averaged B) -- for comparison
    # ----------------------------------------------------------------
    log("\n  [5c] BROKEN composition (single A_0 + averaged B, for comparison)...")
    # Set A_0 for all modules
    _set_lora_a(model, skeleton, 0, n_layers, target_keys)

    # Average all B weights
    all_adapters = []
    for domain_name in DOMAIN_NAMES:
        params = load_adapter(ADAPTERS_DIR / domain_name)
        all_adapters.append(params)

    composed_broken = {}
    for param_key in all_adapters[0].keys():
        stacked = mx.stack([a[param_key] for a in all_adapters])
        composed_broken[param_key] = mx.mean(stacked, axis=0)

    zero_b_params(model)
    apply_adapter_weights(model, composed_broken)
    mx.eval(model.parameters())

    broken_ppls = {}
    broken_ppls_ci = {}
    for domain_name, data_dir in domain_data.items():
        ppl, per_sample = compute_ppl(model, tokenizer, data_dir, return_per_sample=True)
        _, lo, hi = bootstrap_ppl_ci(per_sample)
        broken_ppls[domain_name] = ppl
        broken_ppls_ci[domain_name] = {"ppl": ppl, "ci_lo": round(lo, 2), "ci_hi": round(hi, 2)}
        log(f"  {domain_name}: broken composed PPL = {ppl:.2f} [{lo:.2f}, {hi:.2f}]")

    cleanup(model, tokenizer)
    del all_adapters

    # ----------------------------------------------------------------
    # 5d: CORRECT composition (per-expert A_i @ B_i sum / N)
    # ----------------------------------------------------------------
    log("\n  [5d] CORRECT composition (per-expert A_i @ B_i, sum/N)...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    # Replace target projections with MultiAdapterLoRALinear
    all_adapters = []
    for domain_name in DOMAIN_NAMES:
        params = load_adapter(ADAPTERS_DIR / domain_name)
        all_adapters.append(params)

    n_layers = len(model.model.layers)
    count = 0
    for li, layer in enumerate(model.model.layers):
        lora_updates = []
        for key in target_keys:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue

            # Collect A matrices for all N domains
            a_inits = []
            for di in range(N_DOMAINS):
                skey = f"layer_{li}_{key}_domain_{di}"
                if skey in skeleton:
                    a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
                    a_inits.append(a_mx)

            if len(a_inits) != N_DOMAINS:
                continue

            multi_lora = MultiAdapterLoRALinear(
                module, rank=LORA_RANK, scale=LORA_SCALE, a_inits=a_inits
            )

            # Load per-expert B matrices from saved adapters
            # We need to find the param key for this layer's B matrix
            # The naming pattern in saved adapters is like:
            # "model.layers.{li}.{key}.lora_b"
            param_name = f"model.layers.{li}.{key}.lora_b"
            for di in range(N_DOMAINS):
                if param_name in all_adapters[di]:
                    multi_lora.b_matrices[di] = all_adapters[di][param_name]

            lora_updates.append((key, multi_lora))
            count += 1

        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    mx.eval(model.parameters())
    model.freeze()
    log(f"  Applied MultiAdapterLoRA to {count} projections ({N_DOMAINS} experts each)")

    correct_ppls = {}
    correct_ppls_ci = {}
    for domain_name, data_dir in domain_data.items():
        ppl, per_sample = compute_ppl(model, tokenizer, data_dir, return_per_sample=True)
        _, lo, hi = bootstrap_ppl_ci(per_sample)
        correct_ppls[domain_name] = ppl
        correct_ppls_ci[domain_name] = {"ppl": ppl, "ci_lo": round(lo, 2), "ci_hi": round(hi, 2)}
        log(f"  {domain_name}: correct composed PPL = {ppl:.2f} [{lo:.2f}, {hi:.2f}]")

    eval_time = time.time() - t0
    log(f"  Evaluation done in {eval_time:.1f}s")

    log_memory("post-eval")
    cleanup(model, tokenizer)
    del all_adapters

    return {
        "individual_ppls": individual_ppls,
        "individual_ppls_ci": individual_ppls_ci,
        "base_ppls_ci": base_ppls_ci,
        "broken_composed_ppls": broken_ppls,
        "broken_composed_ppls_ci": broken_ppls_ci,
        "correct_composed_ppls": correct_ppls,
        "correct_composed_ppls_ci": correct_ppls_ci,
    }


# ============================================================================
# Phase 6: Orthogonality analysis
# ============================================================================

def phase_orthogonality():
    """Measure adapter-to-adapter cosine similarity."""
    log("\n[Phase 6] Adapter orthogonality analysis...")

    adapters = {}
    for domain_name in DOMAIN_NAMES:
        params = load_adapter(ADAPTERS_DIR / domain_name)
        vec = mx.concatenate([v.reshape(-1) for v in params.values()])
        mx.eval(vec)
        adapters[domain_name] = vec

    cosines = []
    for i in range(N_DOMAINS):
        for j in range(i + 1, N_DOMAINS):
            vi = adapters[DOMAIN_NAMES[i]]
            vj = adapters[DOMAIN_NAMES[j]]
            cos = mx.abs(
                mx.sum(vi * vj)
                / (mx.sqrt(mx.sum(vi**2)) * mx.sqrt(mx.sum(vj**2)) + 1e-10)
            )
            mx.eval(cos)
            cosines.append({
                "pair": f"{DOMAIN_NAMES[i]}-{DOMAIN_NAMES[j]}",
                "abs_cos": round(cos.item(), 6),
            })

    mean_cos = sum(c["abs_cos"] for c in cosines) / len(cosines)
    log(f"  Mean |cos|: {mean_cos:.6f}")
    for c in cosines:
        log(f"    {c['pair']}: {c['abs_cos']:.6f}")

    cleanup()
    return cosines, mean_cos


# ============================================================================
# Phase 7: Routing heads (lightweight)
# ============================================================================

def phase_routing_heads(domain_data):
    """Train and evaluate tiny routing heads for domain classification."""
    log("\n[Phase 7] Training routing heads...")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    # Extract hidden states from base model for each domain
    HIDDEN_DIM = 2560  # BitNet-2B d_model
    HEAD_HIDDEN = 32
    HEAD_TRAIN_STEPS = 300
    HEAD_LR = 3e-4

    domain_hidden_states = {}
    for domain_name, data_dir in domain_data.items():
        texts = []
        with open(data_dir / "train.jsonl") as f:
            for line in f:
                texts.append(json.loads(line)["text"])

        states = []
        for text in texts[:50]:  # 50 samples per domain for routing
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH]
            x = mx.array(tokens)[None, :]

            # Get hidden states from last layer
            h = model.model.embed_tokens(x)
            for layer in model.model.layers:
                h = layer(h)
            h = model.model.norm(h)

            # Mean pool over sequence
            h_mean = mx.mean(h[0], axis=0)  # (d_model,)
            mx.eval(h_mean)
            states.append(h_mean)
            del h, x

        if states:
            domain_hidden_states[domain_name] = mx.stack(states)
            mx.eval(domain_hidden_states[domain_name])
            log(f"  {domain_name}: {len(states)} hidden state vectors")

    # Train one routing head per domain
    head_results = {}
    for target_domain in DOMAIN_NAMES:
        head = RoutingHead(HIDDEN_DIM, HEAD_HIDDEN)
        head_opt = opt.Adam(learning_rate=HEAD_LR)

        # Positive: target domain, Negative: all other domains
        pos = domain_hidden_states[target_domain]
        neg_list = [domain_hidden_states[d] for d in DOMAIN_NAMES if d != target_domain]
        neg = mx.concatenate(neg_list, axis=0)

        n_pos = pos.shape[0]
        n_neg = neg.shape[0]

        def head_loss_fn(head, x, labels):
            logits = head(x).squeeze(-1)
            return nn.losses.binary_cross_entropy(mx.sigmoid(logits), labels, reduction="mean")

        head_loss_and_grad = nn.value_and_grad(head, head_loss_fn)

        gc.disable()
        for step in range(HEAD_TRAIN_STEPS):
            # Sample balanced batch
            p_idx = mx.array(np.random.randint(0, n_pos, size=16))
            n_idx = mx.array(np.random.randint(0, n_neg, size=16))
            batch_x = mx.concatenate([pos[p_idx], neg[n_idx]], axis=0)
            batch_y = mx.concatenate([mx.ones(16), mx.zeros(16)])

            loss, grads = head_loss_and_grad(head, batch_x, batch_y)
            head_opt.update(head, grads)
            mx.eval(head.parameters(), head_opt.state, loss)
        gc.enable()

        # Evaluate accuracy
        pos_scores = mx.sigmoid(head(pos).squeeze(-1))
        neg_scores = mx.sigmoid(head(neg).squeeze(-1))
        mx.eval(pos_scores, neg_scores)

        pos_acc = (pos_scores > 0.5).astype(mx.float32).mean().item()
        neg_acc = (neg_scores < 0.5).astype(mx.float32).mean().item()
        total_acc = (pos_acc * n_pos + neg_acc * n_neg) / (n_pos + n_neg)

        head_results[target_domain] = {
            "accuracy": round(total_acc, 4),
            "pos_accuracy": round(pos_acc, 4),
            "neg_accuracy": round(neg_acc, 4),
        }
        log(f"  {target_domain} head: acc={total_acc:.3f} (pos={pos_acc:.3f}, neg={neg_acc:.3f})")

    routing_time = time.time() - t0
    log(f"  Routing heads done in {routing_time:.1f}s")

    log_memory("post-routing")
    cleanup(model, tokenizer)
    return head_results


# ============================================================================
# Main
# ============================================================================

def main():
    t_total = time.time()
    log("=" * 70)
    log("Real Data Domain Experts: BitNet-2B + QR-Orthogonal + STE Ternary")
    log("  (REVISED: correct multi-A composition + bootstrap CI)")
    log("=" * 70)
    log_memory("start")

    # Check if we can skip training (adapters already exist)
    skip_training = all(
        (ADAPTERS_DIR / d / "adapter.npz").exists()
        for d in DOMAIN_NAMES
    )
    skip_training = skip_training and (ADAPTERS_DIR / "grassmannian_skeleton.npz").exists()

    # Load previous results for train_results if skipping
    prev_results = {}
    if RESULTS_FILE.exists():
        prev_results = json.loads(RESULTS_FILE.read_text())

    results = {
        "experiment": "real_data_domain_experts",
        "revision": "v2_correct_composition",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "train_iters": TRAIN_ITERS,
        "domains": DOMAIN_NAMES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Phase 1: Data
    domain_data = phase_prepare_data()
    log_memory("post-data")

    if skip_training:
        log("\n[Phases 2-4] SKIPPED: adapters and skeleton already on disk")
        # Load previous base PPLs and training results
        base_ppls = prev_results.get("base_ppls", {})
        train_results = prev_results.get("train_results", {})

        if not base_ppls:
            # Recompute base PPL if not in previous results
            base_ppls = phase_base_ppl(domain_data)

        results["base_ppls"] = base_ppls
        results["train_results"] = train_results
        log(f"  Loaded base PPLs: {base_ppls}")
    else:
        # Phase 2: Base PPL
        base_ppls = phase_base_ppl(domain_data)
        results["base_ppls"] = base_ppls
        log_memory("post-base-ppl")

        # Phase 3: Grassmannian skeleton (QR-orthogonal at this scale)
        skeleton = phase_grassmannian_skeleton()
        log_memory("post-skeleton")

        # Phase 4: Train adapters
        train_results = {}
        for di, domain_name in enumerate(DOMAIN_NAMES):
            tr = phase_train_adapter(di, domain_name, domain_data, skeleton)
            train_results[domain_name] = tr
            results["train_results"] = train_results
            RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

        results["train_results"] = train_results
        n_converged = sum(1 for r in train_results.values() if r["converged"])
        log(f"\n  {n_converged}/5 adapters converged")

    # Phase 5: Evaluate (REVISED: correct composition + broken comparison + CI)
    eval_results = phase_evaluate(domain_data, base_ppls)
    individual_ppls = eval_results["individual_ppls"]
    correct_composed_ppls = eval_results["correct_composed_ppls"]
    broken_composed_ppls = eval_results["broken_composed_ppls"]

    results["individual_ppls"] = individual_ppls
    results["individual_ppls_ci"] = eval_results["individual_ppls_ci"]
    results["base_ppls_ci"] = eval_results["base_ppls_ci"]
    results["correct_composed_ppls"] = correct_composed_ppls
    results["correct_composed_ppls_ci"] = eval_results["correct_composed_ppls_ci"]
    results["broken_composed_ppls"] = broken_composed_ppls
    results["broken_composed_ppls_ci"] = eval_results["broken_composed_ppls_ci"]
    # Keep backward compat key pointing to correct composition
    results["composed_ppls"] = correct_composed_ppls

    # Phase 6: Orthogonality
    cosines, mean_cos = phase_orthogonality()
    results["cosine_similarities"] = cosines
    results["mean_abs_cos"] = round(mean_cos, 6)

    # Phase 7: Routing heads
    head_results = phase_routing_heads(domain_data)
    results["routing_heads"] = head_results

    # ============================================================
    # Kill criteria assessment
    # ============================================================
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    # K1: Adapters don't specialize (PPL same as base on target domain)
    specialization_improvements = {}
    k1_pass_count = 0
    for d in DOMAIN_NAMES:
        base = base_ppls[d]
        ind = individual_ppls[d]
        imp = (base - ind) / base * 100
        specialization_improvements[d] = round(imp, 2)
        if imp > 5.0:
            k1_pass_count += 1
        log(f"  {d}: base={base:.2f}, adapted={ind:.2f}, improvement={imp:+.1f}%")

    results["specialization_improvements"] = specialization_improvements
    k1_pass = k1_pass_count >= 3
    results["k1_pass"] = k1_pass
    results["k1_specialized_count"] = k1_pass_count
    log(f"\n  K1 (adapters specialize): {k1_pass_count}/5 improve >5% -> {'PASS' if k1_pass else 'FAIL (KILL)'}")

    # K2: Composition degrades majority of domains (>3/5 worse than base)
    # NOW uses CORRECT composition
    log("\n  K2 assessment using CORRECT multi-A composition:")
    composition_vs_base = {}
    k2_degraded = 0
    for d in DOMAIN_NAMES:
        base = base_ppls[d]
        comp = correct_composed_ppls[d]
        delta = (comp - base) / base * 100
        composition_vs_base[d] = round(delta, 2)
        if comp > base:
            k2_degraded += 1
        log(f"  {d}: base={base:.2f}, correct_composed={comp:.2f}, delta={delta:+.1f}%")

    results["composition_vs_base"] = composition_vs_base
    k2_pass = k2_degraded <= 3
    results["k2_pass"] = k2_pass
    results["k2_degraded_count"] = k2_degraded
    log(f"\n  K2 (composition safety): {k2_degraded}/5 worse than base -> {'PASS' if k2_pass else 'FAIL (KILL)'}")

    # Broken composition comparison
    log("\n  Broken (single-A) vs Correct (multi-A) composition comparison:")
    broken_vs_base = {}
    for d in DOMAIN_NAMES:
        base = base_ppls[d]
        broken = broken_composed_ppls[d]
        correct = correct_composed_ppls[d]
        b_delta = (broken - base) / base * 100
        c_delta = (correct - base) / base * 100
        broken_vs_base[d] = round(b_delta, 2)
        log(f"  {d}: broken={broken:.2f} ({b_delta:+.1f}%), correct={correct:.2f} ({c_delta:+.1f}%)")
    results["broken_composition_vs_base"] = broken_vs_base

    # Success criteria
    success = k1_pass and k2_pass
    results["verdict"] = "SUPPORTED" if success else "KILLED"

    # Summary stats
    avg_base = sum(base_ppls.values()) / N_DOMAINS
    avg_ind = sum(individual_ppls.values()) / N_DOMAINS
    avg_correct_comp = sum(correct_composed_ppls.values()) / N_DOMAINS
    avg_broken_comp = sum(broken_composed_ppls.values()) / N_DOMAINS
    avg_head_acc = sum(h["accuracy"] for h in head_results.values()) / N_DOMAINS

    results["avg_base_ppl"] = round(avg_base, 4)
    results["avg_individual_ppl"] = round(avg_ind, 4)
    results["avg_correct_composed_ppl"] = round(avg_correct_comp, 4)
    results["avg_broken_composed_ppl"] = round(avg_broken_comp, 4)
    results["avg_composed_ppl"] = round(avg_correct_comp, 4)  # backward compat
    results["avg_routing_accuracy"] = round(avg_head_acc, 4)

    log(f"\n  Avg base PPL:              {avg_base:.2f}")
    log(f"  Avg individual PPL:        {avg_ind:.2f}")
    log(f"  Avg correct composed PPL:  {avg_correct_comp:.2f}")
    log(f"  Avg broken composed PPL:   {avg_broken_comp:.2f}")
    log(f"  Mean |cos|:                {mean_cos:.6f}")
    log(f"  Avg routing acc:           {avg_head_acc:.3f}")

    total_time = time.time() - t_total
    results["total_time_s"] = round(total_time, 1)
    log(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    log(f"\n  VERDICT: {results['verdict']}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
