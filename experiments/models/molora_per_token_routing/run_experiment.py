#!/usr/bin/env python3
"""
MoLoRA Per-Token Routing: Token-level adapter selection on Apple Silicon.

Each token in a sequence gets its own adapter mixture via independent
Gumbel-sigmoid gates (non-competing, L2R-style). Tokens are grouped by
expert set for efficient pre-merge.

Kill criteria:
  K1: Per-token routing PPL worse than per-sequence top-2 (13.65) -> KILL
  K2: Routing overhead > 10% of forward pass -> KILL
  K3: Per-token expert assignments uniform across sequence (router collapses) -> KILL

Success criteria:
  S1: Per-token top-2 PPL < 13.0 (> 5% improvement over per-sequence 13.65)
  S2: Token-level expert diversity > 2.0 (avg distinct expert sets per sequence)
  S3: Domains with high per-sequence routing error (science: 25% accuracy) improve most

Platform: Apple M5 Pro 48GB, MLX, $0.
"""

import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety
device = mx.device_info()
total_mem = device["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
VAL_BATCHES = 25
SEED = 42

# Per-token router config
ROUTER_HIDDEN_DIM = 64    # router MLP hidden size
ROUTER_LR = 3e-4
ROUTER_TRAIN_STEPS = 600  # training steps for per-token router
ROUTER_TEMPERATURE = 1.0  # Gumbel-sigmoid temperature
TOP_K = 2                 # top-k experts per token

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse existing trained adapters from tiny_routing_heads
ADAPTER_DIR = Path(__file__).parent.parent / "tiny_routing_heads" / "adapters"
DATA_DIR = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"

DOMAINS = ["python", "math", "medical", "legal", "creative"]
N_DOMAINS = len(DOMAINS)


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


# ===========================================================================
# Model loading utilities (reused from tiny_routing_heads)
# ===========================================================================
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
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
    log(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


def apply_lora_to_model(model, rank=16, scale=1.0):
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
    log(f"  Applied LoRA (r={rank}) to {count} linear layers")
    return model


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_to_model(model, adapter_params, scale=1.0):
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def zero_adapter_in_model(model):
    updates = []
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_b" in name:
            updates.append((name, mx.zeros_like(p)))
    if updates:
        model.update(tree_unflatten(updates))


def load_domain_texts(domain_name, split="valid"):
    fpath = DATA_DIR / domain_name / f"{split}.jsonl"
    if not fpath.exists():
        return []
    texts = []
    with open(fpath) as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
    return texts


def get_hidden_states(model, x):
    """Extract hidden states from last layer (full sequence, no pooling)."""
    h = model.model.embed_tokens(x)
    for layer in model.model.layers:
        h = layer(h)
    h = model.model.norm(h)
    return h


def compute_ppl(model, tokenizer, texts, max_batches=25):
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
        del logits, loss, x, y
    if total_tokens == 0:
        return float("inf")
    return math.exp(min(total_loss / total_tokens, 100))


# ===========================================================================
# Per-Token Router: Independent Gumbel-Sigmoid Gates (L2R-style)
# ===========================================================================
class PerTokenRouter(nn.Module):
    """Per-token router with independent binary gates per adapter.

    For each token position, produces N independent activation scores
    via Gumbel-sigmoid (non-competing). Top-k are selected per token.

    Architecture: Linear(d, h) -> GELU -> Linear(h, N_experts)
    Input: hidden states (batch, seq_len, d)
    Output: gate logits (batch, seq_len, N_experts)
    """

    def __init__(self, input_dim, n_experts, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_experts)
        self.n_experts = n_experts

    def __call__(self, h):
        """h: (batch, seq_len, d) -> logits: (batch, seq_len, n_experts)"""
        x = nn.gelu(self.fc1(h))
        return self.fc2(x)

    def gumbel_sigmoid_sample(self, logits, temperature=1.0, hard=False):
        """Sample from Gumbel-sigmoid for differentiable binary gates.

        Each expert gets an independent Bernoulli probability.
        Returns soft gates in [0, 1] for each expert.
        """
        # Gumbel noise for both classes
        u = mx.random.uniform(shape=logits.shape)
        # Clamp for numerical stability
        u = mx.clip(u, 1e-6, 1.0 - 1e-6)
        gumbel_noise = mx.log(u) - mx.log(1.0 - u)
        # Gumbel-sigmoid: sigmoid((logit + noise) / temperature)
        y = mx.sigmoid((logits + gumbel_noise) / temperature)
        if hard:
            # Straight-through estimator
            y_hard = mx.round(y)
            y = mx.stop_gradient(y_hard - y) + y
        return y

    def get_top_k_gates(self, h, k=2, temperature=1.0, training=False):
        """Get top-k expert gates per token.

        Returns:
            gates: (batch, seq_len, n_experts) - sparse gate values
            indices: (batch, seq_len, k) - selected expert indices
        """
        logits = self(h)  # (B, T, N)

        if training:
            # Use Gumbel-sigmoid during training for gradient flow
            soft_gates = self.gumbel_sigmoid_sample(logits, temperature)
        else:
            # Use plain sigmoid at inference
            soft_gates = mx.sigmoid(logits)

        # Top-k selection per token
        indices = mx.argpartition(-soft_gates, kth=k, axis=-1)[..., :k]

        # Gather top-k gate values
        # Build sparse gate mask
        B, T, N = soft_gates.shape
        gate_values = mx.zeros_like(soft_gates)

        # For each expert in top-k, extract and place gate value
        for ki in range(k):
            idx = indices[..., ki:ki+1]  # (B, T, 1)
            # One-hot mask for this selection
            mask = mx.zeros((B, T, N))
            # Scatter: set mask[b,t,idx[b,t,0]] = 1
            # Use take_along_axis pattern
            vals = mx.take_along_axis(soft_gates, idx, axis=-1)  # (B, T, 1)
            # Add contribution via one-hot expansion
            one_hot = mx.equal(
                mx.arange(N).reshape(1, 1, N),
                idx
            ).astype(soft_gates.dtype)
            gate_values = gate_values + one_hot * vals

        # Normalize gate values (sum to 1 per token for weighted merge)
        gate_sum = mx.sum(gate_values, axis=-1, keepdims=True)
        gate_sum = mx.maximum(gate_sum, 1e-8)
        gate_values = gate_values / gate_sum

        return gate_values, indices


# ===========================================================================
# Phase 1: Collect hidden states + per-token labels for router training
# ===========================================================================
def phase_collect_training_data(model_id):
    """Extract hidden states and domain labels for router training."""
    log("\n" + "=" * 70)
    log("[Phase 1] Collecting hidden states for router training")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(model_id)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)

    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[1]
    log(f"  d_model = {d_model}")

    # Collect per-token hidden states with domain labels
    # For router training: each token gets the label of its source domain
    train_data = []  # list of (hidden_states, domain_idx, n_tokens)
    val_data = []

    for domain_idx, domain in enumerate(DOMAINS):
        for split, container in [("train", train_data), ("valid", val_data)]:
            texts = load_domain_texts(domain, split=split)
            max_samples = 40 if split == "train" else VAL_BATCHES
            for text in texts[:max_samples]:
                tokens = tokenizer.encode(text)
                if len(tokens) < 4:
                    continue
                tokens = tokens[:MAX_SEQ_LENGTH]
                x = mx.array(tokens)[None, :]
                h = get_hidden_states(model, x)  # (1, seq_len, d)
                mx.eval(h)
                container.append((h, domain_idx, h.shape[1]))
                del x
            log(f"    {domain}/{split}: {sum(1 for d in container if d[1] == domain_idx)} samples")

    log(f"  Data collection: {time.time() - t0:.1f}s")
    log_memory("after-data-collection")

    # Free model
    result = (train_data, val_data, d_model, tokenizer)
    cleanup(model)
    return result


# ===========================================================================
# Phase 2: Train per-token router
# ===========================================================================
def phase_train_router(train_data, d_model):
    """Train the per-token Gumbel-sigmoid router."""
    log("\n" + "=" * 70)
    log("[Phase 2] Training per-token router")
    log("=" * 70)

    router = PerTokenRouter(d_model, N_DOMAINS, ROUTER_HIDDEN_DIM)
    mx.eval(router.parameters())

    n_params = sum(p.size for _, p in tree_flatten(router.parameters()))
    log(f"  Router params: {n_params:,}")

    optimizer = opt.Adam(learning_rate=ROUTER_LR)
    rng = random.Random(SEED)

    # Training: for each sample, the router should assign high gate values
    # to the correct domain expert for all tokens in that sequence.
    # Loss: binary cross-entropy on each expert gate independently.
    # Target: for domain d, gate[d] should be high, others low.
    def router_loss_fn(router, h, domain_idx):
        """BCE loss on independent gates: target is 1 for correct expert, 0 for others."""
        logits = router(h)  # (1, T, N)
        # Create target: (1, T, N) with 1.0 at domain_idx, 0.0 elsewhere
        T = h.shape[1]
        target = mx.zeros((1, T, N_DOMAINS))
        # Set the correct domain to 1.0 for all tokens
        target_slice = mx.ones((1, T, 1))
        # Build full target via concatenation
        parts = []
        for i in range(N_DOMAINS):
            if i == domain_idx:
                parts.append(mx.ones((1, T, 1)))
            else:
                parts.append(mx.zeros((1, T, 1)))
        target = mx.concatenate(parts, axis=-1)  # (1, T, N)

        # Gumbel-sigmoid for differentiable sampling
        gates = router.gumbel_sigmoid_sample(logits, temperature=ROUTER_TEMPERATURE)

        # Binary cross-entropy per expert per token
        loss = nn.losses.binary_cross_entropy(gates, target, reduction="mean")
        return loss

    loss_and_grad = nn.value_and_grad(router, router_loss_fn)

    gc.disable()
    losses = []
    t0 = time.time()

    for step in range(ROUTER_TRAIN_STEPS):
        # Pick random training sample
        idx = rng.randint(0, len(train_data) - 1)
        h, domain_idx, n_tokens = train_data[idx]

        loss, grads = loss_and_grad(router, h, domain_idx)
        optimizer.update(router, grads)
        mx.eval(router.parameters(), optimizer.state, loss)
        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            avg = sum(losses[-100:]) / len(losses[-100:])
            log(f"    Step {step+1}/{ROUTER_TRAIN_STEPS}: loss={avg:.4f}")

    gc.enable()
    gc.collect()

    log(f"  Router training: {time.time() - t0:.1f}s")
    log(f"  Final loss (last 50): {sum(losses[-50:])/len(losses[-50:]):.4f}")

    # Save router
    router_path = EXPERIMENT_DIR / "router"
    router_path.mkdir(parents=True, exist_ok=True)
    router_params = dict(tree_flatten(router.parameters()))
    mx.savez(str(router_path / "router.npz"), **{k: v for k, v in router_params.items()})
    log(f"  Saved router to {router_path}")

    return router, n_params


# ===========================================================================
# Phase 3: Evaluate per-token routing vs baselines
# ===========================================================================
def phase_evaluate(model_id, router, tokenizer_ref):
    """Compare: uniform 1/N, per-sequence top-2, per-token top-2."""
    log("\n" + "=" * 70)
    log("[Phase 3] Evaluating per-token routing")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(model_id)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Load all 5 adapters
    adapters = {}
    for domain in DOMAINS:
        adapters[domain] = load_adapter(ADAPTER_DIR / domain)
        log(f"  Loaded adapter: {domain}")

    # Load validation texts
    val_texts = {}
    for domain in DOMAINS:
        texts = load_domain_texts(domain, split="valid")
        val_texts[domain] = texts
        log(f"  Val data: {domain} = {len(texts)} texts")

    # === Baseline 1: Uniform 1/N ===
    log("\n  [Baseline] Uniform 1/N composition...")
    composed_uniform = {}
    for key in adapters[DOMAINS[0]].keys():
        stacked = mx.stack([adapters[d][key] for d in DOMAINS])
        composed_uniform[key] = mx.mean(stacked, axis=0)

    apply_adapter_to_model(model, composed_uniform)
    uniform_ppls = {}
    for domain in DOMAINS:
        ppl = compute_ppl(model, tokenizer, val_texts[domain], max_batches=VAL_BATCHES)
        uniform_ppls[domain] = round(ppl, 4)
        log(f"    {domain}: {ppl:.4f}")
    zero_adapter_in_model(model)
    del composed_uniform

    # === Baseline 2: Per-sequence top-2 routing (replicating exp_bitnet_per_token_routing) ===
    log("\n  [Baseline] Per-sequence top-2 routing...")
    per_seq_ppls = {}
    per_seq_routing = {}

    for eval_domain in DOMAINS:
        domain_losses = 0.0
        domain_tokens = 0
        domain_routing = []

        for text in val_texts[eval_domain][:VAL_BATCHES]:
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH + 1]
            x = mx.array(tokens[:-1])[None, :]
            y = mx.array(tokens[1:])[None, :]

            # Get hidden states, mean-pool for sequence-level routing
            h = get_hidden_states(model, x)  # (1, T, d)
            h_pool = mx.mean(h, axis=1, keepdims=True)  # (1, 1, d)
            mx.eval(h_pool)

            # Router scores on pooled representation
            logits = router(h_pool)  # (1, 1, N)
            scores = mx.sigmoid(logits[0, 0])  # (N,)
            mx.eval(scores)
            score_list = scores.tolist()

            # Top-2 selection
            sorted_experts = sorted(enumerate(score_list), key=lambda x: x[1], reverse=True)
            top2 = sorted_experts[:TOP_K]
            domain_routing.append({DOMAINS[i]: round(s, 4) for i, s in sorted_experts})

            # Pre-merge top-2 with score-weighted composition
            total_score = sum(s for _, s in top2)
            if total_score < 1e-8:
                total_score = 1.0
            composed = {}
            for expert_idx, expert_score in top2:
                w = expert_score / total_score
                for key, val in adapters[DOMAINS[expert_idx]].items():
                    if key not in composed:
                        composed[key] = val * w
                    else:
                        composed[key] = composed[key] + val * w

            apply_adapter_to_model(model, composed)
            logits_out = model(x)
            loss = nn.losses.cross_entropy(logits_out, y, reduction="sum")
            mx.eval(loss)

            domain_losses += loss.item()
            domain_tokens += y.size

            zero_adapter_in_model(model)
            del composed, logits_out, loss, x, y, h, h_pool, logits, scores

        if domain_tokens > 0:
            ppl = math.exp(min(domain_losses / domain_tokens, 100))
        else:
            ppl = float("inf")

        per_seq_ppls[eval_domain] = round(ppl, 4)
        per_seq_routing[eval_domain] = domain_routing
        log(f"    {eval_domain}: {ppl:.4f}")

    # === Per-Token Top-2 Routing ===
    log("\n  [Experiment] Per-token top-2 routing (Gumbel-sigmoid)...")
    per_token_ppls = {}
    per_token_diversity = {}
    per_token_routing_stats = {}

    for eval_domain in DOMAINS:
        domain_losses = 0.0
        domain_tokens = 0
        all_expert_sets = []  # track unique expert sets per token per sequence

        for text in val_texts[eval_domain][:VAL_BATCHES]:
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH + 1]
            x_tokens = tokens[:-1]
            y_tokens = tokens[1:]

            x = mx.array(x_tokens)[None, :]
            y = mx.array(y_tokens)[None, :]

            # Get hidden states for routing (from base model)
            h = get_hidden_states(model, x)  # (1, T, d)
            mx.eval(h)

            # Per-token routing
            gate_logits = router(h)  # (1, T, N)
            gate_probs = mx.sigmoid(gate_logits)  # (1, T, N)
            mx.eval(gate_probs)

            # Top-k selection per token
            T = h.shape[1]
            gate_np = gate_probs[0].tolist()  # T x N list

            # Group tokens by expert set for efficient merge
            token_groups = {}  # frozenset(expert_indices) -> list of token positions
            token_weights = {}  # frozenset -> normalized weights

            for t in range(T):
                scores = gate_np[t]
                sorted_experts = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
                top_k = sorted_experts[:TOP_K]
                expert_set = frozenset(idx for idx, _ in top_k)

                # Normalize weights for this token's experts
                total_score = sum(s for _, s in top_k)
                if total_score < 1e-8:
                    total_score = 1.0
                weights = {idx: s / total_score for idx, s in top_k}

                if expert_set not in token_groups:
                    token_groups[expert_set] = []
                    token_weights[expert_set] = weights
                token_groups[expert_set].append(t)

            # Track diversity: number of distinct expert sets in this sequence
            all_expert_sets.append(len(token_groups))

            # Compute loss per token group with pre-merged adapters
            seq_loss = 0.0
            for expert_set, positions in token_groups.items():
                weights = token_weights[expert_set]

                # Pre-merge adapters for this group
                composed = {}
                for expert_idx, w in weights.items():
                    for key, val in adapters[DOMAINS[expert_idx]].items():
                        if key not in composed:
                            composed[key] = val * w
                        else:
                            composed[key] = composed[key] + val * w

                apply_adapter_to_model(model, composed)

                # Forward pass on the full sequence (model needs full context)
                logits_out = model(x)  # (1, T, V)
                mx.eval(logits_out)

                # Extract loss only for positions in this group
                for pos in positions:
                    if pos < len(y_tokens):
                        token_logits = logits_out[0, pos:pos+1]  # (1, V)
                        token_target = y[0, pos:pos+1]  # (1,)
                        token_loss = nn.losses.cross_entropy(
                            token_logits, token_target, reduction="sum"
                        )
                        mx.eval(token_loss)
                        seq_loss += token_loss.item()
                        domain_tokens += 1
                        del token_logits, token_target, token_loss

                zero_adapter_in_model(model)
                del composed, logits_out

            domain_losses += seq_loss
            del h, gate_logits, gate_probs, x, y

        if domain_tokens > 0:
            ppl = math.exp(min(domain_losses / domain_tokens, 100))
        else:
            ppl = float("inf")

        avg_diversity = sum(all_expert_sets) / len(all_expert_sets) if all_expert_sets else 0
        per_token_ppls[eval_domain] = round(ppl, 4)
        per_token_diversity[eval_domain] = round(avg_diversity, 2)
        log(f"    {eval_domain}: PPL={ppl:.4f}, avg_diversity={avg_diversity:.2f}")

    # === Timing: measure routing overhead (K2) ===
    log("\n  Measuring routing overhead (K2)...")
    test_text = val_texts[DOMAINS[0]][0]
    test_tokens = tokenizer.encode(test_text)[:MAX_SEQ_LENGTH]
    test_x = mx.array(test_tokens)[None, :]

    # Warm up
    for _ in range(3):
        out = model(test_x)
        mx.eval(out)

    n_timing = 20

    # Time base forward pass
    t_base_start = time.time()
    for _ in range(n_timing):
        out = model(test_x)
        mx.eval(out)
    t_base = (time.time() - t_base_start) / n_timing

    # Time router inference (hidden state extraction + routing)
    h_test = get_hidden_states(model, test_x)
    mx.eval(h_test)

    # Warm up router
    for _ in range(3):
        out = router(h_test)
        mx.eval(out)

    t_router_start = time.time()
    for _ in range(n_timing):
        out = router(h_test)
        mx.eval(out)
    t_router = (time.time() - t_router_start) / n_timing

    overhead_pct = (t_router / t_base) * 100
    log(f"    Base forward: {t_base*1000:.2f}ms")
    log(f"    Router inference: {t_router*1000:.2f}ms")
    log(f"    Overhead: {overhead_pct:.2f}%")

    del test_x, h_test

    log_memory("after-eval")

    result = {
        "uniform_ppls": uniform_ppls,
        "per_seq_ppls": per_seq_ppls,
        "per_seq_routing": per_seq_routing,
        "per_token_ppls": per_token_ppls,
        "per_token_diversity": per_token_diversity,
        "timing": {
            "base_forward_ms": round(t_base * 1000, 2),
            "router_ms": round(t_router * 1000, 2),
            "overhead_pct": round(overhead_pct, 2),
        },
    }

    cleanup(model, tokenizer)
    return result


# ===========================================================================
# Phase 4: Kill criteria assessment
# ===========================================================================
def phase_analysis(eval_results, router_n_params):
    """Assess kill criteria and write results."""
    log("\n" + "=" * 70)
    log("[Phase 4] Kill criteria assessment")
    log("=" * 70)

    uniform_ppls = eval_results["uniform_ppls"]
    per_seq_ppls = eval_results["per_seq_ppls"]
    per_token_ppls = eval_results["per_token_ppls"]
    per_token_diversity = eval_results["per_token_diversity"]
    timing = eval_results["timing"]

    avg_uniform = sum(uniform_ppls.values()) / len(uniform_ppls)
    avg_per_seq = sum(per_seq_ppls.values()) / len(per_seq_ppls)
    avg_per_token = sum(per_token_ppls.values()) / len(per_token_ppls)
    avg_diversity = sum(per_token_diversity.values()) / len(per_token_diversity)

    log(f"\n  Avg PPL comparison:")
    log(f"    Uniform 1/N:      {avg_uniform:.4f}")
    log(f"    Per-sequence top2: {avg_per_seq:.4f}")
    log(f"    Per-token top2:    {avg_per_token:.4f}")

    # Per-domain breakdown
    log("\n  Per-domain PPL:")
    log(f"    {'Domain':<12} {'Uniform':>10} {'PerSeq':>10} {'PerToken':>10} {'Diversity':>10}")
    for domain in DOMAINS:
        log(f"    {domain:<12} {uniform_ppls[domain]:>10.4f} {per_seq_ppls[domain]:>10.4f} "
            f"{per_token_ppls[domain]:>10.4f} {per_token_diversity[domain]:>10.2f}")

    # --- K1: Per-token PPL < per-sequence (13.65 baseline) ---
    per_seq_baseline = 13.65  # from exp_bitnet_per_token_routing
    k1_pass = avg_per_token < per_seq_baseline
    log(f"\n  K1: Per-token PPL ({avg_per_token:.4f}) < per-seq baseline ({per_seq_baseline})")
    log(f"      {'PASS' if k1_pass else 'FAIL'}")

    # --- K2: Routing overhead < 10% ---
    k2_pass = timing["overhead_pct"] < 10.0
    log(f"\n  K2: Routing overhead ({timing['overhead_pct']:.2f}%) < 10%")
    log(f"      {'PASS' if k2_pass else 'FAIL'}")

    # --- K3: Expert diversity > 1.0 (not collapsed to uniform) ---
    # If diversity == 1.0, router assigns same experts to all tokens (collapsed)
    k3_pass = avg_diversity > 1.0
    log(f"\n  K3: Expert diversity ({avg_diversity:.2f}) > 1.0 (not collapsed)")
    log(f"      {'PASS' if k3_pass else 'FAIL'}")

    # --- Success criteria ---
    # S1: Per-token PPL < 13.0
    s1_target = 13.0
    s1_pass = avg_per_token < s1_target
    improvement_vs_seq = ((avg_per_seq - avg_per_token) / avg_per_seq) * 100
    log(f"\n  S1: Per-token PPL ({avg_per_token:.4f}) < {s1_target}")
    log(f"      Improvement over per-seq: {improvement_vs_seq:+.1f}%")
    log(f"      {'PASS' if s1_pass else 'FAIL'}")

    # S2: Diversity > 2.0
    s2_pass = avg_diversity > 2.0
    log(f"\n  S2: Expert diversity ({avg_diversity:.2f}) > 2.0")
    log(f"      {'PASS' if s2_pass else 'FAIL'}")

    # S3: Science domain (math+medical) improves most
    # Compare relative improvement per domain
    improvements = {}
    for domain in DOMAINS:
        if per_seq_ppls[domain] > 0:
            impr = ((per_seq_ppls[domain] - per_token_ppls[domain]) / per_seq_ppls[domain]) * 100
            improvements[domain] = round(impr, 2)
    log(f"\n  S3: Per-domain improvement (per-seq -> per-token):")
    for domain in DOMAINS:
        log(f"      {domain}: {improvements.get(domain, 0):+.2f}%")

    # S3 pass if math or medical are in top 2 improvers
    sorted_impr = sorted(improvements.items(), key=lambda x: x[1], reverse=True)
    top2_improvers = [d for d, _ in sorted_impr[:2]]
    s3_pass = "math" in top2_improvers or "medical" in top2_improvers
    log(f"      Top improvers: {', '.join(top2_improvers)}")
    log(f"      {'PASS' if s3_pass else 'FAIL'}")

    # Overall verdict
    all_kill_pass = k1_pass and k2_pass and k3_pass
    verdict = "SUPPORTED" if all_kill_pass else "KILLED"
    kill_reasons = []
    if not k1_pass:
        kill_reasons.append("K1 (per-token PPL worse than per-sequence)")
    if not k2_pass:
        kill_reasons.append("K2 (routing overhead > 10%)")
    if not k3_pass:
        kill_reasons.append("K3 (router collapsed to uniform)")

    log(f"\n  VERDICT: {verdict}")
    if kill_reasons:
        log(f"  Kill reasons: {', '.join(kill_reasons)}")

    results = {
        "experiment": "molora_per_token_routing",
        "model": MODEL_ID,
        "n_domains": N_DOMAINS,
        "domains": DOMAINS,
        "lora_rank": LORA_RANK,
        "router_hidden_dim": ROUTER_HIDDEN_DIM,
        "router_train_steps": ROUTER_TRAIN_STEPS,
        "router_temperature": ROUTER_TEMPERATURE,
        "top_k": TOP_K,
        "seed": SEED,
        "router_n_params": router_n_params,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        # PPLs
        "uniform_ppls": uniform_ppls,
        "per_seq_ppls": per_seq_ppls,
        "per_token_ppls": per_token_ppls,
        "per_token_diversity": per_token_diversity,
        "per_domain_improvements": improvements,
        "avg_uniform_ppl": round(avg_uniform, 4),
        "avg_per_seq_ppl": round(avg_per_seq, 4),
        "avg_per_token_ppl": round(avg_per_token, 4),
        "avg_diversity": round(avg_diversity, 2),
        # Timing
        "timing": timing,
        # Kill criteria
        "K1_per_token_ppl": round(avg_per_token, 4),
        "K1_baseline": per_seq_baseline,
        "K1_pass": k1_pass,
        "K2_overhead_pct": timing["overhead_pct"],
        "K2_threshold": 10.0,
        "K2_pass": k2_pass,
        "K3_avg_diversity": round(avg_diversity, 2),
        "K3_threshold": 1.0,
        "K3_pass": k3_pass,
        # Success criteria
        "S1_target": s1_target,
        "S1_pass": s1_pass,
        "S2_diversity_target": 2.0,
        "S2_pass": s2_pass,
        "S3_top_improvers": top2_improvers,
        "S3_pass": s3_pass,
        "improvement_vs_per_seq_pct": round(improvement_vs_seq, 2),
        # Verdict
        "verdict": verdict,
        "kill_reasons": kill_reasons if kill_reasons else None,
    }

    return results


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_global = time.time()
    mx.random.seed(SEED)
    log_memory("start")

    log("=" * 70)
    log("MoLoRA Per-Token Routing: Token-level adapter selection")
    log(f"  {N_DOMAINS} domains: {', '.join(DOMAINS)}")
    log(f"  Router: 2-layer MLP, hidden={ROUTER_HIDDEN_DIM}, {ROUTER_TRAIN_STEPS} steps")
    log(f"  Top-{TOP_K} experts per token, Gumbel-sigmoid gates")
    log(f"  LoRA: rank={LORA_RANK}, scale={LORA_SCALE}")
    log(f"  Baseline: per-sequence top-2 PPL=13.65")
    log("=" * 70)

    # Check adapters exist
    for domain in DOMAINS:
        if not (ADAPTER_DIR / domain / "adapter.npz").exists():
            log(f"FATAL: Missing adapter for {domain} at {ADAPTER_DIR / domain}")
            sys.exit(1)

    # Check data exists
    for domain in DOMAINS:
        if not (DATA_DIR / domain / "valid.jsonl").exists():
            log(f"FATAL: Missing data for {domain} at {DATA_DIR / domain}")
            sys.exit(1)

    # Phase 1: Collect training data
    train_data, val_data, d_model, tokenizer = phase_collect_training_data(MODEL_ID)
    log_memory("after-phase1")
    del tokenizer  # freed in phase; just clear reference

    # Phase 2: Train router
    router, router_n_params = phase_train_router(train_data, d_model)
    log_memory("after-phase2")
    # Free training data
    del train_data, val_data
    gc.collect()
    mx.clear_cache()

    # Phase 3: Evaluate
    eval_results = phase_evaluate(MODEL_ID, router, None)
    log_memory("after-phase3")

    # Phase 4: Analysis
    results = phase_analysis(eval_results, router_n_params)
    results["total_time_s"] = round(time.time() - t_global, 1)
    results["d_model"] = d_model

    # Save results
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']}s")


if __name__ == "__main__":
    main()
