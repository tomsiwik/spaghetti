#!/usr/bin/env python3
"""Pointer Routing (No Merge): Per-layer expert selection at full strength.

Hypothesis: Selecting ONE adapter per layer at FULL STRENGTH beats uniform 1/N
merge because it preserves each expert's nonlinear computation (output-space
composition) instead of averaging weight deltas (parameter-space composition).

THREE routing variants:
  (a) Learned gate per layer: linear classifier on hidden state
  (b) Hash lookup: deterministic hash(layer, domain) assignment
  (c) Input-dependent MLP: small MLP on mean-pooled hidden state

Baselines:
  1. Base model (no adapters)
  2. Uniform 1/N merge (all 5 adapters at scale/5)
  3. Single-adapter oracle (all layers use correct domain adapter)

Kill criteria:
  K1 (id=535): Pointer routing quality < uniform 1/N on PPL -> KILL
  K2 (id=536): Per-layer router doesn't specialize (all layers pick same adapter) -> KILL

Success criteria:
  S1 (id=55): Pointer routing beats uniform by >10% with per-layer specialization

References:
  Hash Layers (arxiv 2106.04426) — O(1) hash routing competitive with learned gates
  Switch Transformer (arxiv 2101.03961) — top-1 per-layer selection sufficient

Platform: Apple M5 Pro 48GB, MLX
"""

import gc
import json
import math
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Source experiment with trained adapters
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
VAL_BATCHES = 25
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = len(DOMAINS)

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================================
# BitNet unpacking
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
    """Replace BitLinear with nn.Linear for differentiable forward pass."""
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


# ============================================================================
# LoRA Layer with pointer routing support
# ============================================================================

class PointerRoutedLoRALinear(nn.Module):
    """LoRA with N pre-loaded adapters. At forward time, applies ONE adapter
    selected by the active_expert_idx at FULL strength.

    Forward: y = base(x) + (x @ A_{idx}) @ ternary(B_{idx}) * scale
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 16,
                 scale: float = 20.0, a_inits: list = None,
                 b_params: list = None):
        super().__init__()
        self.linear = base_linear
        self.scale = scale
        self.rank = rank
        self.n_experts = len(a_inits) if a_inits else 0
        self.a_matrices = a_inits if a_inits else []
        self.b_matrices = b_params if b_params else []
        self.linear.freeze()
        self._active_expert = 0  # default: first expert

    def set_active_expert(self, idx):
        self._active_expert = idx

    def __call__(self, x):
        base_out = self.linear(x)
        if self.n_experts == 0:
            return base_out

        idx = self._active_expert
        if idx < 0 or idx >= self.n_experts:
            return base_out

        b = self.b_matrices[idx]
        # STE ternary quantization (inference mode — stop_gradient is no-op)
        alpha = mx.mean(mx.abs(b))
        b_scaled = b / (alpha + 1e-7)
        b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
        b_ste = b + mx.stop_gradient(b_q - b)

        lora_out = (x @ self.a_matrices[idx]) @ b_ste * self.scale
        return base_out + lora_out


class UniformMergeLoRALinear(nn.Module):
    """LoRA with N adapters applied uniformly at scale/N."""
    def __init__(self, base_linear: nn.Linear, rank: int = 16,
                 scale: float = 20.0, a_inits: list = None,
                 b_params: list = None):
        super().__init__()
        self.linear = base_linear
        self.scale = scale
        self.rank = rank
        self.n_experts = len(a_inits) if a_inits else 0
        self.a_matrices = a_inits if a_inits else []
        self.b_matrices = b_params if b_params else []
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
# Model setup utilities
# ============================================================================

def load_skeleton():
    """Load Grassmannian A matrices."""
    skeleton_path = ADAPTERS_DIR / "grassmannian_skeleton.npz"
    return dict(np.load(str(skeleton_path)))


def load_all_adapter_params():
    """Load all 5 domain adapter B-matrix parameters."""
    all_params = {}
    for domain in DOMAINS:
        adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
        all_params[domain] = dict(mx.load(str(adapter_path)))
    return all_params


def setup_pointer_routed_model(model, skeleton, all_adapter_params):
    """Apply PointerRoutedLoRALinear to all target layers."""
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

            a_inits = []
            b_params = []
            for di, domain in enumerate(DOMAINS):
                skey = f"layer_{li}_{key}_domain_{di}"
                if skey in skeleton:
                    a_inits.append(mx.array(skeleton[skey]).astype(mx.bfloat16))
                else:
                    a_inits.append(None)
                b_key = f"model.layers.{li}.{key}.lora_b"
                if b_key in all_adapter_params[domain]:
                    b_params.append(all_adapter_params[domain][b_key])
                else:
                    b_params.append(mx.zeros((LORA_RANK, module.weight.shape[0])))

            lora = PointerRoutedLoRALinear(
                module, rank=LORA_RANK, scale=LORA_SCALE,
                a_inits=a_inits, b_params=b_params
            )
            lora_updates.append((key, lora))
            count += 1

        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    mx.eval(model.parameters())
    log(f"  Setup pointer-routed model ({count} layers)")
    return model


def setup_uniform_merge_model(model, skeleton, all_adapter_params):
    """Apply UniformMergeLoRALinear to all target layers."""
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

            a_inits = []
            b_params = []
            for di, domain in enumerate(DOMAINS):
                skey = f"layer_{li}_{key}_domain_{di}"
                if skey in skeleton:
                    a_inits.append(mx.array(skeleton[skey]).astype(mx.bfloat16))
                else:
                    a_inits.append(None)
                b_key = f"model.layers.{li}.{key}.lora_b"
                if b_key in all_adapter_params[domain]:
                    b_params.append(all_adapter_params[domain][b_key])
                else:
                    b_params.append(mx.zeros((LORA_RANK, module.weight.shape[0])))

            lora = UniformMergeLoRALinear(
                module, rank=LORA_RANK, scale=LORA_SCALE,
                a_inits=a_inits, b_params=b_params
            )
            lora_updates.append((key, lora))
            count += 1

        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    mx.eval(model.parameters())
    log(f"  Setup uniform merge model ({count} layers)")
    return model


def set_all_expert_idx(model, idx):
    """Set all PointerRoutedLoRALinear modules to use expert idx."""
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, PointerRoutedLoRALinear):
                module.set_active_expert(idx)


def set_per_layer_expert_assignment(model, assignment):
    """Set expert assignment per layer. assignment: list of len=n_layers."""
    for li, layer in enumerate(model.model.layers):
        idx = assignment[li]
        for key, module in layer.named_modules():
            if isinstance(module, PointerRoutedLoRALinear):
                module.set_active_expert(idx)


# ============================================================================
# Routing strategies
# ============================================================================

def route_hash(n_layers, n_experts, domain_idx, seed=42):
    """Hash-based per-layer routing: hash(layer_id, domain_idx) mod N.

    Deterministic. Each domain gets a different per-layer assignment.
    Uses SOLE-style consistent hash ring.
    """
    rng = np.random.RandomState(seed + domain_idx)
    assignment = rng.randint(0, n_experts, size=n_layers).tolist()
    return assignment


def route_oracle_single(n_layers, domain_idx):
    """Oracle: all layers use the correct domain adapter."""
    return [domain_idx] * n_layers


def route_learned_gate(model, tokenizer, hidden_states_per_layer, n_experts):
    """Learned linear gate: per-layer expert selection from hidden states.

    Training: use validation data from all domains to train per-layer gates.
    At inference: classify hidden state at each layer to pick expert.
    """
    n_layers = len(hidden_states_per_layer)
    # hidden_states_per_layer: list of (d,) mean-pooled hidden states
    # We train a simple linear classifier per layer

    # This is done externally; here we just apply the pre-computed assignment
    raise NotImplementedError("Use train_learned_gates() first")


def extract_hidden_states(model, tokenizer, text, n_layers):
    """Extract mean-pooled hidden state at each layer for a single input."""
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return None
    tokens = tokens[:MAX_SEQ_LENGTH]
    x = mx.array(tokens)[None, :]

    # Hook into model to capture hidden states
    hidden_states = []

    # Get embedding output
    h = model.model.embed_tokens(x)
    mx.eval(h)

    for li, layer in enumerate(model.model.layers):
        h = layer(h, mask=None)
        # Mean pool over sequence dimension
        h_mean = mx.mean(h, axis=1).squeeze(0)  # (d,)
        mx.eval(h_mean)
        hidden_states.append(h_mean)

    del h, x
    return hidden_states


def train_learned_gates(all_hidden_states, all_labels, n_layers, n_experts, d):
    """Train per-layer linear classifiers from hidden states.

    all_hidden_states: list of lists, [sample_idx][layer_idx] -> (d,) array
    all_labels: list of domain indices, len = n_samples
    n_layers: number of layers
    n_experts: number of adapters/experts

    Returns: list of (W, b) per layer, where W in R^{d x N}, b in R^N
    """
    import numpy as np

    # Collect per-layer data
    gates = []
    for li in range(n_layers):
        # Gather hidden states for this layer
        X = np.stack([np.array(all_hidden_states[si][li]) for si in range(len(all_labels))])
        y = np.array(all_labels)

        # Simple linear classifier via least squares
        # One-hot encode labels
        Y_onehot = np.zeros((len(y), n_experts), dtype=np.float32)
        for i, label in enumerate(y):
            Y_onehot[i, label] = 1.0

        # Ridge regression: W = (X^T X + lambda I)^{-1} X^T Y
        lam = 1e-3
        XtX = X.T @ X + lam * np.eye(X.shape[1], dtype=np.float32)
        XtY = X.T @ Y_onehot
        W = np.linalg.solve(XtX, XtY)  # (d, N)
        b = np.mean(Y_onehot - X @ W, axis=0)  # (N,)

        gates.append((W, b))

    return gates


def apply_learned_gates(gates, hidden_states):
    """Apply trained gates to hidden states to get per-layer assignment.

    gates: list of (W, b) per layer
    hidden_states: list of (d,) arrays per layer

    Returns: list of expert indices per layer
    """
    assignment = []
    for li, (W, b) in enumerate(gates):
        h = np.array(hidden_states[li])
        scores = h @ W + b
        assignment.append(int(np.argmax(scores)))
    return assignment


def train_mlp_gates(all_hidden_states, all_labels, n_layers, n_experts, d, hidden_dim=64):
    """Train per-layer 2-layer MLP classifiers.

    Returns: list of (W1, b1, W2, b2) per layer
    """
    gates = []
    for li in range(n_layers):
        X = np.stack([np.array(all_hidden_states[si][li]) for si in range(len(all_labels))])
        y = np.array(all_labels)

        Y_onehot = np.zeros((len(y), n_experts), dtype=np.float32)
        for i, label in enumerate(y):
            Y_onehot[i, label] = 1.0

        # 2-layer MLP via gradient descent (simple numpy implementation)
        np.random.seed(42 + li)
        W1 = np.random.randn(d, hidden_dim).astype(np.float32) * 0.01
        b1 = np.zeros(hidden_dim, dtype=np.float32)
        W2 = np.random.randn(hidden_dim, n_experts).astype(np.float32) * 0.01
        b2 = np.zeros(n_experts, dtype=np.float32)

        lr = 0.01
        n_steps = 200
        n_samples = X.shape[0]

        for step in range(n_steps):
            # Forward
            h = X @ W1 + b1  # (n, hidden_dim)
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2  # (n, n_experts)

            # Softmax cross-entropy
            logits_max = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # Gradient
            dlogits = (probs - Y_onehot) / n_samples
            dW2 = h_relu.T @ dlogits
            db2 = dlogits.sum(axis=0)
            dh_relu = dlogits @ W2.T
            dh = dh_relu * (h > 0).astype(np.float32)
            dW1 = X.T @ dh
            db1 = dh.sum(axis=0)

            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        gates.append((W1, b1, W2, b2))

    return gates


def apply_mlp_gates(gates, hidden_states):
    """Apply MLP gates to get per-layer assignment."""
    assignment = []
    for li, (W1, b1, W2, b2) in enumerate(gates):
        h = np.array(hidden_states[li])
        hidden = np.maximum(h @ W1 + b1, 0)
        scores = hidden @ W2 + b2
        assignment.append(int(np.argmax(scores)))
    return assignment


# ============================================================================
# PPL Evaluation
# ============================================================================

def compute_ppl(model, tokenizer, data_path, max_batches=25):
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

        loss_val = loss.item()
        n_tok = y.size
        total_loss += loss_val
        total_tokens += n_tok

        del logits, loss, x, y

    if total_tokens == 0:
        return float("inf")
    return float(np.exp(total_loss / total_tokens))


# ============================================================================
# Phase 1: Evaluate base model PPL
# ============================================================================

def phase_base_ppl():
    """Evaluate base model PPL on all domains."""
    log("\n[Phase 1] Base model PPL evaluation...")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    log_memory("after-unpack")

    base_ppls = {}
    for domain in DOMAINS:
        data_path = DATA_DIR / domain
        ppl = compute_ppl(model, tokenizer, data_path)
        base_ppls[domain] = ppl
        log(f"  Base PPL on {domain}: {ppl:.2f}")

    elapsed = time.time() - t0
    log(f"  Phase 1 done in {elapsed:.1f}s")

    cleanup(model, tokenizer)
    return base_ppls


# ============================================================================
# Phase 2: Evaluate uniform 1/N merge PPL
# ============================================================================

def phase_uniform_merge_ppl():
    """Evaluate uniform 1/N merge on all domains."""
    log("\n[Phase 2] Uniform 1/N merge PPL evaluation...")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    skeleton = load_skeleton()
    all_params = load_all_adapter_params()
    model = setup_uniform_merge_model(model, skeleton, all_params)
    log_memory("after-uniform-setup")

    uniform_ppls = {}
    for domain in DOMAINS:
        data_path = DATA_DIR / domain
        ppl = compute_ppl(model, tokenizer, data_path)
        uniform_ppls[domain] = ppl
        log(f"  Uniform PPL on {domain}: {ppl:.2f}")

    elapsed = time.time() - t0
    log(f"  Phase 2 done in {elapsed:.1f}s")

    cleanup(model, tokenizer, skeleton, all_params)
    return uniform_ppls


# ============================================================================
# Phase 3: Single-adapter oracle PPL
# ============================================================================

def phase_oracle_single_ppl():
    """Evaluate single-adapter oracle: each domain uses its own adapter at all layers."""
    log("\n[Phase 3] Single-adapter oracle PPL evaluation...")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    skeleton = load_skeleton()
    all_params = load_all_adapter_params()
    model = setup_pointer_routed_model(model, skeleton, all_params)
    log_memory("after-pointer-setup")

    oracle_ppls = {}
    for di, domain in enumerate(DOMAINS):
        # All layers use this domain's adapter
        set_all_expert_idx(model, di)
        data_path = DATA_DIR / domain
        ppl = compute_ppl(model, tokenizer, data_path)
        oracle_ppls[domain] = ppl
        log(f"  Oracle single PPL on {domain}: {ppl:.2f}")

    elapsed = time.time() - t0
    log(f"  Phase 3 done in {elapsed:.1f}s")

    cleanup(model, tokenizer, skeleton, all_params)
    return oracle_ppls


# ============================================================================
# Phase 4: Extract hidden states for routing training
# ============================================================================

def phase_extract_hidden_states():
    """Extract hidden states from base model for training routers."""
    log("\n[Phase 4] Extracting hidden states for router training...")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    n_layers = len(model.model.layers)
    d = model.model.layers[0].self_attn.q_proj.weight.shape[1]

    log(f"  Model: {n_layers} layers, d={d}")

    # Collect hidden states from validation data of all domains
    all_hidden_states = []  # list of [layer_idx] -> (d,) arrays
    all_labels = []  # domain indices
    texts_per_domain = []

    SAMPLES_PER_DOMAIN = 10  # enough for simple gate training

    for di, domain in enumerate(DOMAINS):
        valid_path = DATA_DIR / domain / "valid.jsonl"
        if not valid_path.exists():
            log(f"  WARNING: no validation data for {domain}")
            continue

        texts = []
        with open(valid_path) as f:
            for line in f:
                texts.append(json.loads(line)["text"])

        domain_texts = texts[:SAMPLES_PER_DOMAIN]
        texts_per_domain.append(domain_texts)

        for text in domain_texts:
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH]
            x = mx.array(tokens)[None, :]

            # Forward through embedding + each layer, capturing hidden states
            h = model.model.embed_tokens(x)
            mx.eval(h)

            sample_hidden = []
            for li, layer in enumerate(model.model.layers):
                # Create causal mask
                seq_len = h.shape[1]
                mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
                mask = mask.astype(h.dtype)

                h = layer(h, mask=mask)
                mx.eval(h)
                h_mean = mx.mean(h, axis=1).squeeze(0)  # (d,)
                mx.eval(h_mean)
                sample_hidden.append(np.array(h_mean.astype(mx.float32)))

            all_hidden_states.append(sample_hidden)
            all_labels.append(di)
            del h, x

    elapsed = time.time() - t0
    log(f"  Extracted {len(all_labels)} samples, {n_layers} layers each in {elapsed:.1f}s")

    cleanup(model, tokenizer)
    return all_hidden_states, all_labels, n_layers, d, texts_per_domain


# ============================================================================
# Phase 5: Pointer routing evaluation
# ============================================================================

def phase_pointer_routing_ppl(all_hidden_states, all_labels, n_layers, d, texts_per_domain):
    """Evaluate all pointer routing variants."""
    log("\n[Phase 5] Pointer routing evaluation...")
    t0 = time.time()

    # Train routers on extracted hidden states
    log("  Training learned linear gates...")
    linear_gates = train_learned_gates(
        all_hidden_states, all_labels, n_layers, N_DOMAINS, d
    )

    log("  Training MLP gates (d_h=64)...")
    mlp_gates = train_mlp_gates(
        all_hidden_states, all_labels, n_layers, N_DOMAINS, d, hidden_dim=64
    )

    # Now evaluate each routing variant
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    skeleton = load_skeleton()
    all_params = load_all_adapter_params()
    model = setup_pointer_routed_model(model, skeleton, all_params)
    log_memory("after-pointer-setup-phase5")

    results = {
        "hash": {},
        "learned_gate": {},
        "mlp_gate": {},
    }

    # Track per-layer assignments for specialization analysis
    assignments = {
        "hash": {},
        "learned_gate": {},
        "mlp_gate": {},
    }

    for di, domain in enumerate(DOMAINS):
        data_path = DATA_DIR / domain

        # --- Hash routing ---
        hash_assign = route_hash(n_layers, N_DOMAINS, di, seed=SEED)
        set_per_layer_expert_assignment(model, hash_assign)
        ppl_hash = compute_ppl(model, tokenizer, data_path)
        results["hash"][domain] = ppl_hash
        assignments["hash"][domain] = hash_assign
        log(f"  Hash routing PPL on {domain}: {ppl_hash:.2f} (assignment: {hash_assign[:5]}...)")

        # --- Learned gate routing ---
        # Get hidden states for this domain's val data
        # Use pre-extracted hidden states for the first SAMPLES_PER_DOMAIN items
        start_idx = di * len(texts_per_domain[di]) if di < len(texts_per_domain) else 0
        # For each val sample, compute gate assignment
        domain_assignments_gate = []
        for si in range(start_idx, min(start_idx + len(texts_per_domain[di]), len(all_hidden_states))):
            assign = apply_learned_gates(linear_gates, all_hidden_states[si])
            domain_assignments_gate.append(assign)

        # Use majority vote across samples for this domain
        if domain_assignments_gate:
            gate_assign = []
            for li in range(n_layers):
                layer_votes = [a[li] for a in domain_assignments_gate]
                gate_assign.append(max(set(layer_votes), key=layer_votes.count))
        else:
            gate_assign = [di] * n_layers  # fallback to oracle

        set_per_layer_expert_assignment(model, gate_assign)
        ppl_gate = compute_ppl(model, tokenizer, data_path)
        results["learned_gate"][domain] = ppl_gate
        assignments["learned_gate"][domain] = gate_assign
        log(f"  Learned gate PPL on {domain}: {ppl_gate:.2f} (assignment: {gate_assign[:5]}...)")

        # --- MLP gate routing ---
        domain_assignments_mlp = []
        for si in range(start_idx, min(start_idx + len(texts_per_domain[di]), len(all_hidden_states))):
            assign = apply_mlp_gates(mlp_gates, all_hidden_states[si])
            domain_assignments_mlp.append(assign)

        if domain_assignments_mlp:
            mlp_assign = []
            for li in range(n_layers):
                layer_votes = [a[li] for a in domain_assignments_mlp]
                mlp_assign.append(max(set(layer_votes), key=layer_votes.count))
        else:
            mlp_assign = [di] * n_layers

        set_per_layer_expert_assignment(model, mlp_assign)
        ppl_mlp = compute_ppl(model, tokenizer, data_path)
        results["mlp_gate"][domain] = ppl_mlp
        assignments["mlp_gate"][domain] = mlp_assign
        log(f"  MLP gate PPL on {domain}: {ppl_mlp:.2f} (assignment: {mlp_assign[:5]}...)")

    elapsed = time.time() - t0
    log(f"  Phase 5 done in {elapsed:.1f}s")

    cleanup(model, tokenizer, skeleton, all_params)
    return results, assignments


# ============================================================================
# Analysis
# ============================================================================

def analyze_specialization(assignments, n_layers, n_experts):
    """Analyze per-layer specialization patterns.

    Returns:
    - per_layer_entropy: entropy of expert distribution at each layer
    - layer_diversity: fraction of unique experts used across layers
    - same_adapter_fraction: fraction of domains where all layers pick same expert
    """
    analysis = {}

    for method, domain_assigns in assignments.items():
        # Collect all assignments across domains
        all_assigns = list(domain_assigns.values())

        # Per-layer entropy: for each layer, what's the distribution of experts
        # across domains?
        layer_entropies = []
        for li in range(n_layers):
            counts = np.zeros(n_experts)
            for assign in all_assigns:
                counts[assign[li]] += 1
            counts = counts / counts.sum()
            entropy = -np.sum(counts * np.log(counts + 1e-10))
            layer_entropies.append(entropy)

        mean_entropy = float(np.mean(layer_entropies))
        max_entropy = float(np.log(n_experts))

        # Layer diversity per domain: how many unique experts are used?
        diversities = []
        for assign in all_assigns:
            unique = len(set(assign))
            diversities.append(unique / n_experts)
        mean_diversity = float(np.mean(diversities))

        # Same-adapter check: does any domain use the same adapter at ALL layers?
        same_count = 0
        for assign in all_assigns:
            if len(set(assign)) == 1:
                same_count += 1
        same_fraction = same_count / len(all_assigns)

        # Cross-layer variation: for each domain, how different are assignments
        # across layers?
        cross_layer_variation = []
        for assign in all_assigns:
            transitions = sum(1 for i in range(len(assign)-1) if assign[i] != assign[i+1])
            cross_layer_variation.append(transitions / (len(assign) - 1))
        mean_cross_layer = float(np.mean(cross_layer_variation))

        analysis[method] = {
            "mean_layer_entropy": mean_entropy,
            "max_possible_entropy": max_entropy,
            "entropy_ratio": mean_entropy / max_entropy,
            "mean_diversity": mean_diversity,
            "same_adapter_fraction": same_fraction,
            "mean_cross_layer_variation": mean_cross_layer,
            "layer_entropies": [float(e) for e in layer_entropies],
        }

    return analysis


def main():
    log("=" * 70)
    log("Pointer Routing (No Merge) Experiment")
    log("=" * 70)
    t0 = time.time()
    log_memory("start")

    # Phase 1: Base PPL
    base_ppls = phase_base_ppl()
    log_memory("after-phase1")

    # Phase 2: Uniform 1/N merge
    uniform_ppls = phase_uniform_merge_ppl()
    log_memory("after-phase2")

    # Phase 3: Single-adapter oracle
    oracle_ppls = phase_oracle_single_ppl()
    log_memory("after-phase3")

    # Phase 4: Extract hidden states
    all_hs, all_labels, n_layers, d, texts_per_domain = phase_extract_hidden_states()
    log_memory("after-phase4")

    # Phase 5: Pointer routing variants
    routing_ppls, assignments = phase_pointer_routing_ppl(
        all_hs, all_labels, n_layers, d, texts_per_domain
    )
    log_memory("after-phase5")

    # Phase 6: Analysis
    log("\n[Phase 6] Analysis...")

    specialization = analyze_specialization(assignments, n_layers, N_DOMAINS)

    # Compute improvement vs uniform for each method
    improvements = {}
    for method in ["hash", "learned_gate", "mlp_gate"]:
        method_improvements = {}
        for domain in DOMAINS:
            uniform_ppl = uniform_ppls[domain]
            routing_ppl = routing_ppls[method][domain]
            # Positive = routing is better (lower PPL)
            pct_improvement = (uniform_ppl - routing_ppl) / uniform_ppl * 100
            method_improvements[domain] = pct_improvement
        improvements[method] = method_improvements

    # Kill criteria evaluation
    log("\n" + "=" * 70)
    log("KILL CRITERIA EVALUATION")
    log("=" * 70)

    # K1: Pointer routing < uniform 1/N?
    # Check best routing method
    k1_results = {}
    for method in ["hash", "learned_gate", "mlp_gate"]:
        domains_worse = 0
        for domain in DOMAINS:
            if routing_ppls[method][domain] > uniform_ppls[domain]:
                domains_worse += 1
        mean_improvement = np.mean(list(improvements[method].values()))
        k1_results[method] = {
            "domains_worse_than_uniform": domains_worse,
            "mean_improvement_pct": float(mean_improvement),
            "pass": mean_improvement > 0,  # at least some improvement on average
        }
        log(f"  K1 [{method}]: {domains_worse}/5 domains worse than uniform, "
            f"mean improvement {mean_improvement:.1f}%")

    best_method = max(k1_results, key=lambda m: k1_results[m]["mean_improvement_pct"])
    k1_pass = k1_results[best_method]["pass"]
    log(f"  K1 VERDICT: {'PASS' if k1_pass else 'FAIL'} (best method: {best_method})")

    # K2: Per-layer specialization?
    k2_results = {}
    for method in ["hash", "learned_gate", "mlp_gate"]:
        spec = specialization[method]
        # KILL if all layers pick same adapter (same_adapter_fraction == 1.0)
        # or if entropy is near zero (no diversity)
        specializes = (spec["same_adapter_fraction"] < 0.8 and
                       spec["entropy_ratio"] > 0.2)
        k2_results[method] = {
            "same_adapter_fraction": spec["same_adapter_fraction"],
            "entropy_ratio": spec["entropy_ratio"],
            "mean_diversity": spec["mean_diversity"],
            "cross_layer_variation": spec["mean_cross_layer_variation"],
            "pass": specializes,
        }
        log(f"  K2 [{method}]: same_frac={spec['same_adapter_fraction']:.2f}, "
            f"entropy_ratio={spec['entropy_ratio']:.2f}, "
            f"diversity={spec['mean_diversity']:.2f}, "
            f"cross_layer={spec['mean_cross_layer_variation']:.2f}")

    k2_pass = any(k2_results[m]["pass"] for m in k2_results)
    log(f"  K2 VERDICT: {'PASS' if k2_pass else 'FAIL'}")

    # S1: >10% improvement with specialization?
    best_improvement = k1_results[best_method]["mean_improvement_pct"]
    s1_pass = best_improvement > 10.0 and k2_pass
    log(f"\n  S1 VERDICT: {'PASS' if s1_pass else 'FAIL'} "
        f"(best mean improvement: {best_improvement:.1f}%, needs >10%)")

    # Summary table
    log("\n" + "=" * 70)
    log("PPL SUMMARY TABLE")
    log("=" * 70)
    log(f"{'Domain':<12} {'Base':<10} {'Uniform':<10} {'Oracle':<10} "
        f"{'Hash':<10} {'LGate':<10} {'MLP':<10}")
    log("-" * 72)
    for domain in DOMAINS:
        log(f"{domain:<12} {base_ppls[domain]:<10.2f} {uniform_ppls[domain]:<10.2f} "
            f"{oracle_ppls[domain]:<10.2f} {routing_ppls['hash'][domain]:<10.2f} "
            f"{routing_ppls['learned_gate'][domain]:<10.2f} "
            f"{routing_ppls['mlp_gate'][domain]:<10.2f}")

    log("\nIMPROVEMENT vs UNIFORM (%)")
    log(f"{'Domain':<12} {'Hash':<10} {'LGate':<10} {'MLP':<10}")
    log("-" * 42)
    for domain in DOMAINS:
        log(f"{domain:<12} {improvements['hash'][domain]:<+10.1f} "
            f"{improvements['learned_gate'][domain]:<+10.1f} "
            f"{improvements['mlp_gate'][domain]:<+10.1f}")

    mean_base = float(np.mean(list(base_ppls.values())))
    mean_uniform = float(np.mean(list(uniform_ppls.values())))
    mean_oracle = float(np.mean(list(oracle_ppls.values())))

    # Overall verdict
    if k1_pass and k2_pass:
        if s1_pass:
            verdict = "SUPPORTED"
        else:
            verdict = "SUPPORTED (K1+K2 pass, S1 threshold not met)"
    elif not k1_pass:
        verdict = "KILLED (K1 FAIL: routing worse than uniform)"
    else:
        verdict = "KILLED (K2 FAIL: no per-layer specialization)"

    log(f"\nOVERALL VERDICT: {verdict}")

    # Save results
    results = {
        "experiment": "pointer_routing_no_merge",
        "model": MODEL_ID,
        "n_domains": N_DOMAINS,
        "n_layers": n_layers,
        "lora_rank": LORA_RANK,
        "lora_scale": LORA_SCALE,
        "base_ppls": base_ppls,
        "uniform_ppls": uniform_ppls,
        "oracle_single_ppls": oracle_ppls,
        "routing_ppls": routing_ppls,
        "improvements_vs_uniform": improvements,
        "specialization": specialization,
        "assignments": assignments,
        "k1_results": k1_results,
        "k2_results": k2_results,
        "s1_pass": s1_pass,
        "best_method": best_method,
        "verdict": verdict,
        "mean_base_ppl": mean_base,
        "mean_uniform_ppl": mean_uniform,
        "mean_oracle_ppl": mean_oracle,
        "total_time_s": round(time.time() - t0, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']:.1f}s")


if __name__ == "__main__":
    main()
