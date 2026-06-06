"""Capsule surgery for pretrained models — additive capsule residuals.

Wraps each frozen MLP layer with a trainable CapsulePool:
    layer output = attention(x) + original_mlp(x) + capsule_pool(x)
                     frozen           frozen          trainable

Capsule groups are zero-initialized so the surgically-modified model
matches baseline PPL before any training.
"""

import mlx.core as mx
import mlx.nn as nn


# --- Capsule primitives (ported from experiments/models/capsule_moe/capsule_moe.py) ---

class CapsuleGroup(nn.Module):
    """A group of rank-1 non-linear capsules: B @ ReLU(A @ x).

    A: (n_capsules, d_model) — detector vectors
    B: (d_model, n_capsules) — expansion vectors (zero-init by default)
    """

    def __init__(self, d_model: int, n_capsules: int, zero_init: bool = True):
        super().__init__()
        self.A = nn.Linear(d_model, n_capsules, bias=False)
        self.B = nn.Linear(n_capsules, d_model, bias=False)
        if zero_init:
            self.B.weight = mx.zeros_like(self.B.weight)

    def __call__(self, x):
        return self.B(nn.relu(self.A(x)))


class CapsulePool(nn.Module):
    """Pool of capsule groups with top-k group routing."""

    def __init__(self, d_model: int, n_groups: int = 4,
                 n_capsules_per_group: int = 224, top_k_groups: int = 2):
        super().__init__()
        self.n_groups = n_groups
        self.top_k_groups = top_k_groups
        self.router = nn.Linear(d_model, n_groups, bias=False)
        self.groups = [CapsuleGroup(d_model, n_capsules_per_group)
                       for _ in range(n_groups)]

    def __call__(self, x):
        scores = self.router(x)                              # (..., G)
        probs = mx.softmax(scores, axis=-1)

        top_vals = mx.topk(scores, self.top_k_groups, axis=-1)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)
        mask = (scores >= threshold).astype(mx.float32)
        masked_probs = probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

        out = mx.zeros_like(x)
        for i, group in enumerate(self.groups):
            w = masked_probs[..., i:i + 1]
            out = out + w * group(x)
        return out


# --- Surgery wrapper ---

class CapsuleAugmentedMLP(nn.Module):
    """Wraps original MLP with additive capsule residual.

    output = original_mlp(x) + capsule_pool(x)
    """

    def __init__(self, original_mlp, capsule_pool: CapsulePool):
        super().__init__()
        self.original_mlp = original_mlp
        self.capsule_pool = capsule_pool

    def __call__(self, x):
        return self.original_mlp(x) + self.capsule_pool(x)


# --- Surgery operations ---

def apply_capsule_surgery(model, n_groups: int = 4,
                          n_capsules_per_group: int = 224,
                          top_k_groups: int = 2) -> None:
    """Add capsule pools as additive residuals to each MLP layer.

    Modifies model in-place. After surgery, capsule output is zero
    (B weights zero-initialized) so model behavior is unchanged.
    """
    d_model = model.args.hidden_size

    for layer in model.model.layers:
        pool = CapsulePool(d_model, n_groups, n_capsules_per_group, top_k_groups)
        layer.mlp = CapsuleAugmentedMLP(layer.mlp, pool)
        mx.eval(pool.parameters())


def is_surgically_modified(model) -> bool:
    """Check if model has capsule-augmented MLPs."""
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return False
    return isinstance(model.model.layers[0].mlp, CapsuleAugmentedMLP)


# --- Freeze helpers ---

def freeze_except_capsules(model):
    """Freeze all params except capsule groups (A and B weights)."""
    model.freeze()
    for layer in model.model.layers:
        for group in layer.mlp.capsule_pool.groups:
            group.unfreeze()


def freeze_except_router(model):
    """Freeze all params except capsule pool routers."""
    model.freeze()
    for layer in model.model.layers:
        layer.mlp.capsule_pool.router.unfreeze()


# --- State management ---

def extract_capsule_state(model) -> dict:
    """Extract capsule pool parameters as a flat dict of arrays."""
    state = {}
    for l_idx, layer in enumerate(model.model.layers):
        pool = layer.mlp.capsule_pool
        state[f"layers.{l_idx}.router.weight"] = pool.router.weight
        for g_idx, group in enumerate(pool.groups):
            prefix = f"layers.{l_idx}.groups.{g_idx}"
            state[f"{prefix}.A.weight"] = group.A.weight
            state[f"{prefix}.B.weight"] = group.B.weight
    return state


def load_capsule_state(model, state: dict, group_offset: int = 0):
    """Load capsule state into model, optionally at a group offset.

    When composing, use group_offset to place domain A at 0..G-1
    and domain B at G..2G-1.
    """
    for l_idx, layer in enumerate(model.model.layers):
        pool = layer.mlp.capsule_pool
        g_idx = 0
        while f"layers.{l_idx}.groups.{g_idx}.A.weight" in state:
            target_g = group_offset + g_idx
            prefix = f"layers.{l_idx}.groups.{g_idx}"
            pool.groups[target_g].A.weight = state[f"{prefix}.A.weight"]
            pool.groups[target_g].B.weight = state[f"{prefix}.B.weight"]
            g_idx += 1
    mx.eval(model.parameters())


def save_capsule_state(model, path):
    """Save capsule state to .npz file."""
    state = extract_capsule_state(model)
    mx.savez(str(path), **state)


def load_capsule_state_from_file(path) -> dict:
    """Load capsule state from .npz file."""
    return dict(mx.load(str(path)))


# --- Dead capsule profiling ---

def profile_dead_capsules(model, tokenizer, texts: list[str],
                          n_batches: int = 20, batch_size: int = 4,
                          max_length: int = 512) -> dict:
    """Profile which capsules are dead (never fire) across sample data.

    Returns dict with per-layer and aggregate dead capsule statistics.
    """
    import random
    from .train import tokenize_texts, make_batch

    all_tokens = tokenize_texts(tokenizer, texts, max_length)
    rng = random.Random(42)

    # Collect activation counts: {layer_idx: {group_idx: {capsule_idx: count}}}
    n_layers = len(model.model.layers)
    layer_stats = {}

    for l_idx, layer in enumerate(model.model.layers):
        if not isinstance(layer.mlp, CapsuleAugmentedMLP):
            continue
        pool = layer.mlp.capsule_pool
        n_groups = pool.n_groups
        # Track per-group, per-capsule fire counts
        layer_stats[l_idx] = {
            g_idx: mx.zeros(group.A.weight.shape[0])
            for g_idx, group in enumerate(pool.groups)
        }

    # Hook into capsule groups to track activations
    fire_counts = {l: {g: 0 for g in gs} for l, gs in layer_stats.items()}
    total_tokens = 0

    for batch_i in range(n_batches):
        input_ids, mask = make_batch(all_tokens, batch_size, rng, max_length)
        # Forward pass through embedding + each layer manually to capture intermediates
        h = model.model.embed_tokens(input_ids)

        cache = [None] * n_layers
        if hasattr(model.model, 'norm'):
            # Run full forward to get correct hidden states at each layer
            pass

        # Simpler approach: just run each capsule group's A matrix on a sample
        # and check ReLU activation patterns
        # Use a forward hook approach: run the full model, then probe each layer
        logits = model(input_ids)
        mx.eval(logits)

    # Direct profiling: feed data through embed + layers, check capsule activations
    # Reset and do it properly with intermediate captures
    for batch_i in range(n_batches):
        input_ids, mask = make_batch(all_tokens, batch_size, rng, max_length)
        h = model.model.embed_tokens(input_ids)

        from mlx_lm.models.qwen2 import create_attention_mask
        cache = [None] * n_layers

        for l_idx, layer_module in enumerate(model.model.layers):
            # Get input to MLP (post attention + residual + layernorm)
            normed = layer_module.input_layernorm(h)
            attn_out = layer_module.self_attn(normed, mask=None, cache=None)
            h_post_attn = h + attn_out
            mlp_input = layer_module.post_attention_layernorm(h_post_attn)

            # Profile each capsule group
            if l_idx in layer_stats:
                pool = layer_module.mlp.capsule_pool
                for g_idx, group in enumerate(pool.groups):
                    activations = nn.relu(group.A(mlp_input))  # (B, T, n_capsules)
                    fired = (activations > 0).astype(mx.float32)
                    # Sum over batch and token dims
                    fire_count = mx.sum(fired, axis=(0, 1))
                    mx.eval(fire_count)
                    fire_counts[l_idx][g_idx] = fire_counts[l_idx][g_idx] + fire_count

            # Continue forward pass
            mlp_out = layer_module.mlp(mlp_input)
            h = h_post_attn + mlp_out

        total_tokens += input_ids.shape[0] * input_ids.shape[1]

    # Compute dead capsule statistics
    total_capsules = 0
    total_dead = 0
    per_layer = {}

    for l_idx in sorted(fire_counts.keys()):
        layer_total = 0
        layer_dead = 0
        for g_idx in fire_counts[l_idx]:
            counts = fire_counts[l_idx][g_idx]
            if isinstance(counts, int):
                continue
            mx.eval(counts)
            n_caps = counts.shape[0]
            n_dead = int((counts == 0).sum().item())
            layer_total += n_caps
            layer_dead += n_dead
        per_layer[l_idx] = {
            "total": layer_total,
            "dead": layer_dead,
            "dead_pct": layer_dead / max(1, layer_total) * 100,
        }
        total_capsules += layer_total
        total_dead += layer_dead

    return {
        "total_capsules": total_capsules,
        "total_dead": total_dead,
        "dead_pct": total_dead / max(1, total_capsules) * 100,
        "total_tokens_profiled": total_tokens,
        "per_layer": per_layer,
    }
