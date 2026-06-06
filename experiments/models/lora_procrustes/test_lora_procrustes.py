"""LoRA Procrustes Linear Decomposition Experiment.

Tests whether LoRA deltas (pure linear: dW = A @ B) can be decomposed into
shared (always-on) + unique (routed) components without the nonlinearity
penalty that killed the original Procrustes experiment (exp3).

The key difference from exp3: LoRA deltas are applied as additive linear
corrections to frozen base weights. The delta itself has no activation function.
So shared_dW @ x + unique_dW @ x = dW @ x EXACTLY.

Compares:
1. Joint training (baseline)
2. Concatenated LoRA (base + route to domain-specific deltas)
3. Decomposed LoRA (base + shared_dW always-on + route to unique_dW)
4. Task arithmetic (base + mean of all deltas)
5. Shared-only (base + shared_dW, no unique)

Kill criteria:
- Decomposed LoRA composition >3% worse than concatenated LoRA
- Shared component accounts for <10% of delta norm
"""

import copy
import random
import statistics
import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate, ntp_loss
from experiments.models.gpt.gpt import GPT


# ── Config ──────────────────────────────────────────────────────────────────

BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
LORA_RANK = 8
LORA_ALPHA = 1.0
PRETRAIN_STEPS = 300
FINETUNE_STEPS = 300
ROUTER_CAL_STEPS = 100
BATCH_SIZE = 32
LR = 3e-3


# ── Utilities ───────────────────────────────────────────────────────────────

def copy_weights(src, dst):
    """Copy all weights from src to dst model."""
    pairs = list(zip(
        [k for k, _ in nn.utils.tree_flatten(src.parameters())],
        [v for _, v in nn.utils.tree_flatten(src.parameters())]
    ))
    dst.load_weights(pairs)
    mx.eval(dst.parameters())


def count_params(model) -> int:
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def freeze_except_lora(model):
    """Freeze all parameters except LoRA A and B."""
    model.freeze()
    for layer in model.layers:
        # Unfreeze only LoRA parameters
        layer.mlp.fc1.unfreeze()
        layer.mlp.fc2.unfreeze()
        # Re-freeze the base linear weights
        layer.mlp.fc1.linear.freeze()
        layer.mlp.fc2.linear.freeze()


def reset_lora(model):
    """Reset all LoRA A/B to initialization (A=random, B=zero)."""
    for layer in model.layers:
        for fc in [layer.mlp.fc1, layer.mlp.fc2]:
            in_dim = fc.A.shape[0]
            rank = fc.A.shape[1]
            scale = (2.0 / in_dim) ** 0.5
            fc.A = mx.random.normal(fc.A.shape) * scale
            fc.B = mx.zeros(fc.B.shape)
    mx.eval(model.parameters())


def get_deltas(model) -> list:
    """Extract LoRA delta matrices from model.

    Returns: list of (layer_idx, sublayer_name, delta) where delta = (alpha/r) * A @ B.
    """
    deltas = []
    for l_idx, layer in enumerate(model.layers):
        for name, fc in [('fc1', layer.mlp.fc1), ('fc2', layer.mlp.fc2)]:
            delta = (fc.alpha / fc.rank) * (fc.A @ fc.B)
            deltas.append((l_idx, name, delta))
    return deltas


def decompose_deltas(deltas_list):
    """Decompose N sets of deltas into shared + unique components.

    Args:
        deltas_list: list of N delta sets, each from get_deltas().

    Returns:
        shared_deltas: list of (l_idx, name, shared_matrix)
        unique_deltas_per_domain: list of N lists of (l_idx, name, unique_matrix)
        metrics: dict with decomposition statistics
    """
    N = len(deltas_list)
    n_matrices = len(deltas_list[0])

    shared_deltas = []
    unique_deltas_per_domain = [[] for _ in range(N)]

    total_delta_norm_sq = 0.0
    total_shared_norm_sq = 0.0
    total_unique_norm_sq = 0.0
    max_recon_error = 0.0

    for m_idx in range(n_matrices):
        l_idx, name, _ = deltas_list[0][m_idx]

        # Collect all N domain deltas for this matrix position
        all_domain_deltas = [deltas_list[d][m_idx][2] for d in range(N)]

        # Shared = mean of all domain deltas
        shared = sum(all_domain_deltas) / N

        shared_deltas.append((l_idx, name, shared))

        for d in range(N):
            unique = all_domain_deltas[d] - shared
            unique_deltas_per_domain[d].append((l_idx, name, unique))

            # Norms
            total_delta_norm_sq += mx.sum(all_domain_deltas[d] ** 2).item()
            total_unique_norm_sq += mx.sum(unique ** 2).item()

            # Reconstruction check
            recon = shared + unique
            err = mx.max(mx.abs(recon - all_domain_deltas[d])).item()
            max_recon_error = max(max_recon_error, err)

        total_shared_norm_sq += N * mx.sum(shared ** 2).item()

    total_delta_norm = total_delta_norm_sq ** 0.5
    total_shared_norm = total_shared_norm_sq ** 0.5
    total_unique_norm = total_unique_norm_sq ** 0.5

    shared_frac = total_shared_norm / (total_shared_norm + total_unique_norm) \
        if (total_shared_norm + total_unique_norm) > 0 else 0.0

    # Also compute cosine similarity between domain deltas (flattened)
    if N == 2:
        flat_0 = mx.concatenate([d[2].reshape(-1) for d in deltas_list[0]])
        flat_1 = mx.concatenate([d[2].reshape(-1) for d in deltas_list[1]])
        cos = (mx.sum(flat_0 * flat_1) /
               (mx.sqrt(mx.sum(flat_0 ** 2)) * mx.sqrt(mx.sum(flat_1 ** 2)) + 1e-8))
        cos_sim = cos.item()
    else:
        cos_sim = None

    return shared_deltas, unique_deltas_per_domain, {
        'delta_norm': total_delta_norm,
        'shared_norm': total_shared_norm,
        'unique_norm': total_unique_norm,
        'shared_fraction': shared_frac,
        'max_reconstruction_error': max_recon_error,
        'cosine_similarity': cos_sim,
    }


def apply_deltas_to_base(base_model, deltas, vocab_size):
    """Create a new GPT model with base weights + LoRA deltas baked in.

    This produces a standard GPT (no LoRA) with modified MLP weights.
    """
    from experiments.models.gpt.gpt import GPT
    model = GPT(vocab_size=vocab_size, **BASE)
    mx.eval(model.parameters())

    # Copy all base weights
    copy_weights(base_model, model)

    # Apply deltas to MLP weights
    for l_idx, name, delta in deltas:
        layer = model.layers[l_idx]
        if name == 'fc1':
            # fc1.weight is (4d, d) in nn.Linear convention, delta is (d, 4d)
            # nn.Linear stores weight as (out_features, in_features)
            # delta from A @ B is (in_dim, out_dim) = (d, 4d)
            # But nn.Linear(d, 4d) has weight (4d, d), and computes x @ W.T
            # So we need to add delta.T to fc1.weight
            layer.mlp.fc1.weight = layer.mlp.fc1.weight + delta.T
        elif name == 'fc2':
            # fc2: Linear(4d, d), weight is (d, 4d), delta is (4d, d)
            layer.mlp.fc2.weight = layer.mlp.fc2.weight + delta.T

    mx.eval(model.parameters())
    return model


class RoutedDeltaGPT(nn.Module):
    """GPT with routed LoRA deltas.

    Base weights are frozen. A lightweight router selects which set of deltas
    to apply per token. Deltas are applied as additive linear corrections.
    """

    def __init__(self, base_model, delta_sets, vocab_size,
                 top_k: int = 1, uniform: bool = False):
        """
        Args:
            base_model: trained GPT model (frozen)
            delta_sets: list of N delta lists, each from get_deltas()
            vocab_size: vocabulary size
            top_k: number of delta sets to route to per token
            uniform: if True, average all deltas uniformly (no routing)
        """
        super().__init__()
        self.n_experts = len(delta_sets)
        self.top_k = min(top_k, self.n_experts)
        self.uniform = uniform
        n_embd = BASE['n_embd']

        # Copy base model weights
        self.wte = base_model.wte
        self.wpe = base_model.wpe
        self.norm0 = base_model.norm0
        self.base_layers = base_model.layers
        self.lm_head = base_model.lm_head

        # Pre-build full weight matrices for each expert (base + delta)
        # Stored as stacked tensors per layer: shape (n_experts, out_dim, in_dim)
        n_layer = len(base_model.layers)
        self._expert_fc1_weights = []  # list of (n_experts, 4d, d) tensors
        self._expert_fc2_weights = []  # list of (n_experts, d, 4d) tensors

        for l_idx in range(n_layer):
            base_fc1_w = base_model.layers[l_idx].mlp.fc1.weight  # (4d, d)
            base_fc2_w = base_model.layers[l_idx].mlp.fc2.weight  # (d, 4d)

            fc1_list = []
            fc2_list = []
            for expert_idx, deltas in enumerate(delta_sets):
                # Find deltas for this layer
                for dl_idx, name, delta in deltas:
                    if dl_idx == l_idx and name == 'fc1':
                        fc1_list.append(base_fc1_w + delta.T)
                    elif dl_idx == l_idx and name == 'fc2':
                        fc2_list.append(base_fc2_w + delta.T)

            self._expert_fc1_weights.append(mx.stack(fc1_list))  # (N, 4d, d)
            self._expert_fc2_weights.append(mx.stack(fc2_list))  # (N, d, 4d)

        mx.eval(self._expert_fc1_weights + self._expert_fc2_weights)

        # Per-layer router
        self.routers = [nn.Linear(n_embd, self.n_experts, bias=False)
                        for _ in range(n_layer)]
        self._gate_probs = None

    def _run_expert_mlp(self, h, l_idx, expert_idx):
        """Run MLP with expert-specific weights."""
        fc1_w = self._expert_fc1_weights[l_idx][expert_idx]  # (4d, d)
        fc2_w = self._expert_fc2_weights[l_idx][expert_idx]  # (d, 4d)
        h_fc1 = h @ fc1_w.T
        h_relu = nn.relu(h_fc1)
        return h_relu @ fc2_w.T

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)

        for l_idx, base_layer in enumerate(self.base_layers):
            # Attention (from base, unmodified)
            x = x + base_layer.attn(base_layer.norm1(x))

            # MLP with routed LoRA deltas
            h = base_layer.norm2(x)

            if self.uniform:
                delta_out = mx.zeros_like(h)
                w = 1.0 / self.n_experts
                for e in range(self.n_experts):
                    delta_out = delta_out + w * self._run_expert_mlp(h, l_idx, e)
                x = x + delta_out
            else:
                scores = self.routers[l_idx](h)
                probs = mx.softmax(scores, axis=-1)

                top_vals = mx.topk(scores, self.top_k, axis=-1)
                threshold = mx.min(top_vals, axis=-1, keepdims=True)
                mask = (scores >= threshold).astype(mx.float32)
                masked_probs = probs * mask
                masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

                delta_out = mx.zeros_like(h)
                for e in range(self.n_experts):
                    w_e = masked_probs[..., e:e+1]
                    delta_out = delta_out + w_e * self._run_expert_mlp(h, l_idx, e)
                x = x + delta_out

        return self.lm_head(x)

    def aux_loss(self) -> mx.array:
        return mx.array(0.0)

    def on_domain_switch(self, domain: str):
        pass


class DecomposedDeltaGPT(nn.Module):
    """GPT with shared (always-on) + unique (routed) LoRA deltas.

    The shared delta is baked into the base weights (always active).
    Unique deltas are routed per token.

    This is the LINEAR version of the killed Procrustes decomposition.
    Since LoRA deltas are pure linear corrections to weights,
    (base + shared + unique_k) @ x = (base + dW_k) @ x EXACTLY.
    """

    def __init__(self, base_model, shared_deltas, unique_delta_sets,
                 vocab_size, top_k: int = 1, uniform: bool = False):
        super().__init__()
        self.n_experts = len(unique_delta_sets)
        self.top_k = min(top_k, self.n_experts)
        self.uniform = uniform
        n_embd = BASE['n_embd']
        n_layer = len(base_model.layers)

        # Copy base model structure
        self.wte = base_model.wte
        self.wpe = base_model.wpe
        self.norm0 = base_model.norm0
        self.lm_head = base_model.lm_head
        self.base_layers = base_model.layers

        # Pre-build weight matrices: base + shared + unique_k for each expert
        # First, compute base+shared weights
        shared_fc1 = {}  # l_idx -> delta.T
        shared_fc2 = {}
        for l_idx, name, delta in shared_deltas:
            if name == 'fc1':
                shared_fc1[l_idx] = delta.T
            elif name == 'fc2':
                shared_fc2[l_idx] = delta.T

        self._expert_fc1_weights = []  # list of (n_experts, 4d, d) tensors
        self._expert_fc2_weights = []

        for l_idx in range(n_layer):
            base_fc1 = base_model.layers[l_idx].mlp.fc1.weight  # (4d, d)
            base_fc2 = base_model.layers[l_idx].mlp.fc2.weight  # (d, 4d)
            shared_fc1_w = base_fc1 + shared_fc1.get(l_idx, 0)
            shared_fc2_w = base_fc2 + shared_fc2.get(l_idx, 0)

            # Also bake shared into the base_layers for attention/norm consistency
            self.base_layers[l_idx].mlp.fc1.weight = shared_fc1_w
            self.base_layers[l_idx].mlp.fc2.weight = shared_fc2_w

            fc1_list = []
            fc2_list = []
            for expert_idx, deltas in enumerate(unique_delta_sets):
                for dl_idx, name, delta in deltas:
                    if dl_idx == l_idx and name == 'fc1':
                        fc1_list.append(shared_fc1_w + delta.T)
                    elif dl_idx == l_idx and name == 'fc2':
                        fc2_list.append(shared_fc2_w + delta.T)

            self._expert_fc1_weights.append(mx.stack(fc1_list))
            self._expert_fc2_weights.append(mx.stack(fc2_list))

        mx.eval(self._expert_fc1_weights + self._expert_fc2_weights)
        mx.eval([l.mlp.fc1.weight for l in self.base_layers] +
                [l.mlp.fc2.weight for l in self.base_layers])

        # Per-layer router for unique deltas
        self.routers = [nn.Linear(n_embd, self.n_experts, bias=False)
                        for _ in range(n_layer)]

    def _run_expert_mlp(self, h, l_idx, expert_idx):
        fc1_w = self._expert_fc1_weights[l_idx][expert_idx]
        fc2_w = self._expert_fc2_weights[l_idx][expert_idx]
        return nn.relu(h @ fc1_w.T) @ fc2_w.T

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)

        for l_idx, base_layer in enumerate(self.base_layers):
            # Attention
            x = x + base_layer.attn(base_layer.norm1(x))

            h = base_layer.norm2(x)

            if self.uniform:
                delta_out = mx.zeros_like(h)
                w = 1.0 / self.n_experts
                for e in range(self.n_experts):
                    delta_out = delta_out + w * self._run_expert_mlp(h, l_idx, e)
                x = x + delta_out
            else:
                scores = self.routers[l_idx](h)
                probs = mx.softmax(scores, axis=-1)

                top_vals = mx.topk(scores, self.top_k, axis=-1)
                threshold = mx.min(top_vals, axis=-1, keepdims=True)
                mask = (scores >= threshold).astype(mx.float32)
                masked_probs = probs * mask
                masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

                delta_out = mx.zeros_like(h)
                for e in range(self.n_experts):
                    w_e = masked_probs[..., e:e+1]
                    delta_out = delta_out + w_e * self._run_expert_mlp(h, l_idx, e)
                x = x + delta_out

        return self.lm_head(x)

    def aux_loss(self) -> mx.array:
        return mx.array(0.0)

    def on_domain_switch(self, domain: str):
        pass


def calibrate_router(model, train_ds_a, train_ds_b, steps=100, lr=3e-3, seed=42):
    """Calibrate only the router weights on mixed-domain data."""
    model.freeze()
    for router in model.routers:
        router.unfreeze()

    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    for step in range(1, steps + 1):
        if step % 2 == 1:
            inputs, targets = train_ds_a.get_batch(BATCH_SIZE, rng)
        else:
            inputs, targets = train_ds_b.get_batch(BATCH_SIZE, rng)

        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 50 == 0 or step == steps:
            print(f"    router cal step {step:3d}/{steps} | loss {loss.item():.4f}")

    model.unfreeze()


# ── Main Experiment ─────────────────────────────────────────────────────────

def run_experiment(seed=42):
    """Run the full LoRA Procrustes decomposition experiment."""
    print(f"\n{'='*70}")
    print(f"LORA PROCRUSTES LINEAR DECOMPOSITION (seed={seed})")
    print(f"{'='*70}")

    mx.random.seed(seed)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    splits = domain_split(docs)
    all_train, all_val = train_val_split(docs, seed=seed)

    train_a_docs, val_a_docs = train_val_split(splits["a_m"], seed=seed)
    train_b_docs, val_b_docs = train_val_split(splits["n_z"], seed=seed)

    train_a = CharDataset(train_a_docs, tokenizer, BASE["block_size"])
    val_a = CharDataset(val_a_docs, tokenizer, BASE["block_size"])
    train_b = CharDataset(train_b_docs, tokenizer, BASE["block_size"])
    val_b = CharDataset(val_b_docs, tokenizer, BASE["block_size"])
    joint_train = CharDataset(all_train, tokenizer, BASE["block_size"])
    joint_val = CharDataset(all_val, tokenizer, BASE["block_size"])

    results = {}

    # === 1. Joint training baseline (standard GPT, no LoRA) ===
    print("\n--- 1. Joint training baseline ---")
    model_joint = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(model_joint.parameters())
    total_steps = 2 * FINETUNE_STEPS
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(model_joint, ntp_loss)
    for step in range(1, total_steps + 1):
        if step % 2 == 1:
            inputs, targets = train_a.get_batch(BATCH_SIZE, rng)
        else:
            inputs, targets = train_b.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model_joint, inputs, targets)
        optimizer.update(model_joint, grads)
        mx.eval(model_joint.parameters(), optimizer.state)
        if step % 200 == 0:
            print(f"  step {step:4d}/{total_steps} | loss {loss.item():.4f}")

    j_a = evaluate(model_joint, val_a, BATCH_SIZE)
    j_b = evaluate(model_joint, val_b, BATCH_SIZE)
    results["joint"] = {"a_m": j_a, "n_z": j_b, "avg": (j_a + j_b) / 2}
    print(f"  Joint: a_m={j_a:.4f}, n_z={j_b:.4f}, avg={(j_a+j_b)/2:.4f}")

    # === 2. Pretrain base model ===
    print("\n--- 2. Pretraining base model ---")
    base_model = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_model.parameters())
    train(base_model, joint_train, steps=PRETRAIN_STEPS,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)

    # === 3. Fine-tune LoRA adapters per domain ===
    print("\n--- 3a. Fine-tuning LoRA for domain A ---")
    lora_a = get_model("lora_gpt", vocab_size=V, **BASE,
                       lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA)
    mx.eval(lora_a.parameters())

    # Copy base weights to LoRA model's base linear layers
    for l_idx in range(BASE['n_layer']):
        bl = base_model.layers[l_idx]
        ll = lora_a.layers[l_idx]
        # Attention
        ll.attn.wq.weight = bl.attn.wq.weight
        ll.attn.wk.weight = bl.attn.wk.weight
        ll.attn.wv.weight = bl.attn.wv.weight
        ll.attn.wo.weight = bl.attn.wo.weight
        # MLP base
        ll.mlp.fc1.linear.weight = bl.mlp.fc1.weight
        ll.mlp.fc2.linear.weight = bl.mlp.fc2.weight
        # Norms
        # RMSNorm has no weight param in our implementation (just eps)
    lora_a.wte.weight = base_model.wte.weight
    lora_a.wpe.weight = base_model.wpe.weight
    lora_a.lm_head.weight = base_model.lm_head.weight
    mx.eval(lora_a.parameters())

    # Freeze base, train only LoRA
    freeze_except_lora(lora_a)
    print(f"  Trainable params: {count_params(lora_a):,}")
    train(lora_a, train_a, val_a, steps=FINETUNE_STEPS,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)
    lora_a.unfreeze()

    print("\n--- 3b. Fine-tuning LoRA for domain B ---")
    lora_b = get_model("lora_gpt", vocab_size=V, **BASE,
                       lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA)
    mx.eval(lora_b.parameters())

    for l_idx in range(BASE['n_layer']):
        bl = base_model.layers[l_idx]
        ll = lora_b.layers[l_idx]
        ll.attn.wq.weight = bl.attn.wq.weight
        ll.attn.wk.weight = bl.attn.wk.weight
        ll.attn.wv.weight = bl.attn.wv.weight
        ll.attn.wo.weight = bl.attn.wo.weight
        ll.mlp.fc1.linear.weight = bl.mlp.fc1.weight
        ll.mlp.fc2.linear.weight = bl.mlp.fc2.weight
    lora_b.wte.weight = base_model.wte.weight
    lora_b.wpe.weight = base_model.wpe.weight
    lora_b.lm_head.weight = base_model.lm_head.weight
    mx.eval(lora_b.parameters())

    freeze_except_lora(lora_b)
    train(lora_b, train_b, val_b, steps=FINETUNE_STEPS,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)
    lora_b.unfreeze()

    # === 4. Extract deltas and decompose ===
    print("\n--- 4. Decomposing LoRA deltas ---")
    deltas_a = get_deltas(lora_a)
    deltas_b = get_deltas(lora_b)

    shared_deltas, unique_deltas, metrics = decompose_deltas([deltas_a, deltas_b])

    print(f"  Delta norm:          {metrics['delta_norm']:.6f}")
    print(f"  Shared norm:         {metrics['shared_norm']:.6f}")
    print(f"  Unique norm:         {metrics['unique_norm']:.6f}")
    print(f"  Shared fraction:     {metrics['shared_fraction']:.1%}")
    print(f"  Max recon error:     {metrics['max_reconstruction_error']:.2e}")
    print(f"  Delta cosine sim:    {metrics['cosine_similarity']:.4f}")

    # === 5. Evaluate individual LoRA models (sanity check) ===
    print("\n--- 5. Individual LoRA quality ---")
    baked_a = apply_deltas_to_base(base_model, deltas_a, V)
    baked_b = apply_deltas_to_base(base_model, deltas_b, V)
    la_a = evaluate(baked_a, val_a, BATCH_SIZE)
    la_b = evaluate(baked_a, val_b, BATCH_SIZE)
    lb_a = evaluate(baked_b, val_a, BATCH_SIZE)
    lb_b = evaluate(baked_b, val_b, BATCH_SIZE)
    print(f"  LoRA-A: a_m={la_a:.4f}, n_z={la_b:.4f}")
    print(f"  LoRA-B: a_m={lb_a:.4f}, n_z={lb_b:.4f}")

    # === 6. Task arithmetic: base + mean(deltas) ===
    print("\n--- 6. Task arithmetic (base + mean deltas) ---")
    ta_deltas = []
    for m_idx in range(len(deltas_a)):
        l_idx, name, d_a = deltas_a[m_idx]
        _, _, d_b = deltas_b[m_idx]
        ta_deltas.append((l_idx, name, (d_a + d_b) / 2))

    ta_model = apply_deltas_to_base(base_model, ta_deltas, V)
    ta_a = evaluate(ta_model, val_a, BATCH_SIZE)
    ta_b = evaluate(ta_model, val_b, BATCH_SIZE)
    results["task_arith"] = {"a_m": ta_a, "n_z": ta_b, "avg": (ta_a + ta_b) / 2}
    print(f"  Task arith: a_m={ta_a:.4f}, n_z={ta_b:.4f}, avg={(ta_a+ta_b)/2:.4f}")

    # === 7. Shared-only: base + shared_delta ===
    print("\n--- 7. Shared-only (base + shared delta) ---")
    shared_model = apply_deltas_to_base(base_model, shared_deltas, V)
    so_a = evaluate(shared_model, val_a, BATCH_SIZE)
    so_b = evaluate(shared_model, val_b, BATCH_SIZE)
    results["shared_only"] = {"a_m": so_a, "n_z": so_b, "avg": (so_a + so_b) / 2}
    print(f"  Shared only: a_m={so_a:.4f}, n_z={so_b:.4f}, avg={(so_a+so_b)/2:.4f}")

    # === 8. Concatenated LoRA + calibrated router ===
    print("\n--- 8. Concatenated LoRA + calibrated router ---")
    import copy as _copy
    base_for_concat = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_for_concat.parameters())
    copy_weights(base_model, base_for_concat)

    concat_model = RoutedDeltaGPT(base_for_concat, [deltas_a, deltas_b], V, top_k=2)
    mx.eval(concat_model.parameters())
    calibrate_router(concat_model, train_a, train_b,
                     steps=ROUTER_CAL_STEPS, lr=LR, seed=seed)
    cc_a = evaluate(concat_model, val_a, BATCH_SIZE)
    cc_b = evaluate(concat_model, val_b, BATCH_SIZE)
    results["concat_cal"] = {"a_m": cc_a, "n_z": cc_b, "avg": (cc_a + cc_b) / 2}
    print(f"  Concat+cal: a_m={cc_a:.4f}, n_z={cc_b:.4f}, avg={(cc_a+cc_b)/2:.4f}")

    # === 9. Concatenated LoRA + uniform routing ===
    print("\n--- 9. Concatenated LoRA + uniform routing ---")
    base_for_uni = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_for_uni.parameters())
    copy_weights(base_model, base_for_uni)

    concat_uni = RoutedDeltaGPT(base_for_uni, [deltas_a, deltas_b], V, uniform=True)
    mx.eval(concat_uni.parameters())
    cu_a = evaluate(concat_uni, val_a, BATCH_SIZE)
    cu_b = evaluate(concat_uni, val_b, BATCH_SIZE)
    results["concat_uni"] = {"a_m": cu_a, "n_z": cu_b, "avg": (cu_a + cu_b) / 2}
    print(f"  Concat+uni: a_m={cu_a:.4f}, n_z={cu_b:.4f}, avg={(cu_a+cu_b)/2:.4f}")

    # === 10. Decomposed + calibrated router ===
    print("\n--- 10. Decomposed LoRA + calibrated router ---")
    base_for_decomp = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_for_decomp.parameters())
    copy_weights(base_model, base_for_decomp)

    decomp_model = DecomposedDeltaGPT(
        base_for_decomp, shared_deltas, unique_deltas, V, top_k=2)
    mx.eval(decomp_model.parameters())
    calibrate_router(decomp_model, train_a, train_b,
                     steps=ROUTER_CAL_STEPS, lr=LR, seed=seed)
    dc_a = evaluate(decomp_model, val_a, BATCH_SIZE)
    dc_b = evaluate(decomp_model, val_b, BATCH_SIZE)
    results["decomp_cal"] = {"a_m": dc_a, "n_z": dc_b, "avg": (dc_a + dc_b) / 2}
    print(f"  Decomp+cal: a_m={dc_a:.4f}, n_z={dc_b:.4f}, avg={(dc_a+dc_b)/2:.4f}")

    # === 11. Decomposed + uniform routing ===
    print("\n--- 11. Decomposed LoRA + uniform routing ---")
    base_for_decomp_u = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_for_decomp_u.parameters())
    copy_weights(base_model, base_for_decomp_u)

    decomp_uni = DecomposedDeltaGPT(
        base_for_decomp_u, shared_deltas, unique_deltas, V, uniform=True)
    mx.eval(decomp_uni.parameters())
    du_a = evaluate(decomp_uni, val_a, BATCH_SIZE)
    du_b = evaluate(decomp_uni, val_b, BATCH_SIZE)
    results["decomp_uni"] = {"a_m": du_a, "n_z": du_b, "avg": (du_a + du_b) / 2}
    print(f"  Decomp+uni: a_m={du_a:.4f}, n_z={du_b:.4f}, avg={(du_a+du_b)/2:.4f}")

    # === 12. Verify linearity: baked decomposed should == baked original ===
    print("\n--- 12. Linearity verification ---")
    # For domain A: base + shared + unique_A should == base + delta_A
    recon_a_deltas = []
    for m_idx in range(len(shared_deltas)):
        l_idx, name, s = shared_deltas[m_idx]
        _, _, u = unique_deltas[0][m_idx]
        recon_a_deltas.append((l_idx, name, s + u))

    recon_a_model = apply_deltas_to_base(base_model, recon_a_deltas, V)
    orig_a_model = apply_deltas_to_base(base_model, deltas_a, V)

    # Compare outputs on same input
    test_input = mx.array([[1, 2, 3, 4, 5, 0, 0, 0] + [0] * 24])
    out_recon = recon_a_model(test_input)
    out_orig = orig_a_model(test_input)
    max_diff = mx.max(mx.abs(out_recon - out_orig)).item()
    print(f"  Max output diff (recon vs orig): {max_diff:.2e}")
    print(f"  Linearity holds: {'YES' if max_diff < 1e-5 else 'NO'}")

    # === Summary ===
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Shared fraction: {metrics['shared_fraction']:.1%}")
    print(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}")
    print(f"\n{'Method':<25} {'a_m':>8} {'n_z':>8} {'avg':>8} {'vs joint':>10}")
    print("-" * 60)
    for method, vals in results.items():
        delta = (vals["avg"] - results["joint"]["avg"]) / results["joint"]["avg"] * 100
        print(f"{method:<25} {vals['a_m']:>8.4f} {vals['n_z']:>8.4f} {vals['avg']:>8.4f} {delta:>+9.1f}%")

    return results, metrics


def run_multiseed(seeds=(42, 123, 7)):
    """Run across multiple seeds and aggregate."""
    all_results = {}
    all_metrics = {}

    for seed in seeds:
        results, metrics = run_experiment(seed)
        all_results[seed] = results
        all_metrics[seed] = metrics

    # Aggregate
    print(f"\n\n{'='*70}")
    print("MULTI-SEED AGGREGATE")
    print(f"{'='*70}")

    # Shared fraction
    shared_fracs = [all_metrics[s]["shared_fraction"] for s in seeds]
    cos_sims = [all_metrics[s]["cosine_similarity"] for s in seeds]
    print(f"  Shared fraction: {statistics.mean(shared_fracs):.1%} "
          f"(range: {min(shared_fracs):.1%}-{max(shared_fracs):.1%})")
    print(f"  Cosine similarity: {statistics.mean(cos_sims):.4f} "
          f"(range: {min(cos_sims):.4f}-{max(cos_sims):.4f})")

    methods = list(all_results[seeds[0]].keys())
    joint_avgs = [all_results[s]["joint"]["avg"] for s in seeds]
    joint_mean = statistics.mean(joint_avgs)

    # For kill criteria: compare decomp vs concat
    concat_cal_avgs = [all_results[s]["concat_cal"]["avg"] for s in seeds]
    concat_cal_mean = statistics.mean(concat_cal_avgs)

    print(f"\n{'Method':<25} {'avg (mean)':>12} {'std':>8} {'vs joint':>10} {'vs concat':>10}")
    print("-" * 70)
    for method in methods:
        avgs = [all_results[s][method]["avg"] for s in seeds]
        mean_avg = statistics.mean(avgs)
        std_avg = statistics.stdev(avgs) if len(avgs) > 1 else 0.0

        if method == "joint":
            j_str = "baseline"
            c_str = ""
        else:
            delta_j = (mean_avg - joint_mean) / joint_mean * 100
            j_str = f"{delta_j:+.1f}%"
            delta_c = (mean_avg - concat_cal_mean) / concat_cal_mean * 100
            c_str = f"{delta_c:+.1f}%"
        print(f"{method:<25} {mean_avg:>12.4f} {std_avg:>8.4f} {j_str:>10} {c_str:>10}")

    # Kill threshold checks
    print(f"\n--- Kill Threshold Checks ---")

    # Kill criterion 1: decomposed >3% worse than concatenated
    decomp_cal_avgs = [all_results[s]["decomp_cal"]["avg"] for s in seeds]
    decomp_cal_mean = statistics.mean(decomp_cal_avgs)
    gap = (decomp_cal_mean - concat_cal_mean) / concat_cal_mean * 100
    if gap > 3.0:
        print(f"  KILL: decomp_cal is {gap:+.1f}% vs concat_cal (threshold: 3%)")
    else:
        print(f"  PASS: decomp_cal is {gap:+.1f}% vs concat_cal (threshold: 3%)")

    # Kill criterion 2: shared fraction <10%
    mean_shared = statistics.mean(shared_fracs)
    if mean_shared < 0.10:
        print(f"  KILL: shared fraction {mean_shared:.1%} < 10% (no shared structure)")
    else:
        print(f"  PASS: shared fraction {mean_shared:.1%} >= 10%")

    # Robustness comparison
    decomp_uni_avgs = [all_results[s]["decomp_uni"]["avg"] for s in seeds]
    concat_uni_avgs = [all_results[s]["concat_uni"]["avg"] for s in seeds]
    du_mean = statistics.mean(decomp_uni_avgs)
    cu_mean = statistics.mean(concat_uni_avgs)
    print(f"\n  Robustness (uniform routing):")
    print(f"    decomp+uniform: {du_mean:.4f}")
    print(f"    concat+uniform: {cu_mean:.4f}")
    if du_mean < cu_mean:
        print(f"    Decomposition is MORE robust ({(cu_mean-du_mean)/cu_mean*100:+.1f}% better)")
    else:
        print(f"    Decomposition is LESS robust ({(cu_mean-du_mean)/cu_mean*100:+.1f}% worse)")

    return all_results, all_metrics


if __name__ == "__main__":
    run_multiseed()
