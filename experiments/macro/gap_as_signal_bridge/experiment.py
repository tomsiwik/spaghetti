"""Gap-as-Signal Bridge Experiment: d=256 validation with 20 seeds and CIs.

Bridges the 4000x scale gap between micro (d=64, 200K params) and macro
(d=896, Qwen2.5-0.5B). Tests whether the gap-calibration correlation
(r^2=0.74 at micro) holds at d=256 (~2M params).

Key extensions over micro experiment:
1. d=256, n_head=8, n_layer=6, block_size=64 (~2M params vs 200K)
2. LoRA rank=16 (matching Phase 2 Qwen target, vs rank=8 at micro)
3. 20 seeds with bootstrap 95% CIs (vs 3 seeds, no CI at micro)
4. N=4 experts with top_k=2 (vs N=2 with top_k=2 at micro)
5. 5 domains (quintary split) to match Phase 2 domain count

Protocol (same as micro, scaled up):
1. Train shared base model on all data
2. Train N independent LoRA experts on different domains
3. For each target cosine {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9}:
   a. Project expert deltas to achieve target cosine
   b. Measure function-space gap (CE, KL)
   c. Calibrate router, measure final quality
4. Compute r^2(gap, quality) with bootstrap CIs
"""

import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import ntp_loss, evaluate
from experiments.models.gpt.gpt import GPT, RMSNorm, CausalSelfAttention, MLP, Block

# ── Config ──────────────────────────────────────────────────────────────────

# Bridge scale: d=256, ~2M params (16x micro, 1/14 of Qwen-0.5B)
BRIDGE = dict(n_embd=256, n_head=8, n_layer=6, block_size=64)
LORA_RANK = 16
LORA_ALPHA = 1.0

# Training budgets (scaled proportionally to model size)
PRETRAIN_STEPS = 600
FINETUNE_STEPS = 600
MAX_CAL_STEPS = 500
CAL_EVAL_EVERY = 10
CONVERGENCE_THRESHOLD = 0.005  # within 0.5% of joint loss
BATCH_SIZE = 32
LR = 1e-3  # lower LR for larger model

# Cosine similarity levels to test
TARGET_COSINES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

# Number of seeds
N_SEEDS = 20

# N-expert configurations to test
# Config: (n_experts, top_k, description)
EXPERT_CONFIGS = [
    (2, 2, "N=2,k=2 (micro-matched)"),
    (4, 2, "N=4,k=2 (routing selection)"),
]


# ── LoRA Infrastructure ────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Linear layer with LoRA: out = W @ x + (alpha/r) * A @ B @ x."""

    def __init__(self, in_dim: int, out_dim: int, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.rank = rank
        self.alpha = alpha
        scale = (2.0 / in_dim) ** 0.5
        self.A = mx.random.normal((in_dim, rank)) * scale
        self.B = mx.zeros((rank, out_dim))

    def __call__(self, x):
        return self.linear(x) + (self.alpha / self.rank) * (x @ self.A @ self.B)

    def get_delta(self) -> mx.array:
        return (self.alpha / self.rank) * (self.A @ self.B)


class LoRAMLP(nn.Module):
    def __init__(self, n_embd: int, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        self.fc1 = LoRALinear(n_embd, 4 * n_embd, rank, alpha)
        self.fc2 = LoRALinear(4 * n_embd, n_embd, rank, alpha)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


class LoRABlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = LoRAMLP(n_embd, rank, alpha)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LoRAGPT(nn.Module):
    """GPT with LoRA adapters on MLP layers at bridge scale."""

    def __init__(self, vocab_size: int = 28, block_size: int = 64,
                 n_embd: int = 256, n_head: int = 8, n_layer: int = 6,
                 lora_rank: int = 16, lora_alpha: float = 1.0):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [LoRABlock(n_embd, n_head, lora_rank, lora_alpha)
                       for _ in range(n_layer)]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

    def aux_loss(self):
        return mx.array(0.0)


# ── Delta Manipulation ─────────────────────────────────────────────────────

def get_deltas(model) -> list:
    """Extract LoRA deltas: list of (layer_idx, name, delta_matrix)."""
    deltas = []
    for l_idx, layer in enumerate(model.layers):
        for name, fc in [('fc1', layer.mlp.fc1), ('fc2', layer.mlp.fc2)]:
            delta = (fc.alpha / fc.rank) * (fc.A @ fc.B)
            deltas.append((l_idx, name, delta))
    return deltas


def flatten_deltas(deltas):
    return mx.concatenate([d[2].reshape(-1) for d in deltas])


def unflatten_deltas(flat, template_deltas):
    result = []
    offset = 0
    for l_idx, name, template in template_deltas:
        size = template.size
        shape = template.shape
        chunk = flat[offset:offset + size].reshape(shape)
        result.append((l_idx, name, chunk))
        offset += size
    return result


def project_to_target_cosine(deltas_a, deltas_b, target_cos):
    """Project deltas_b so cos(deltas_a, deltas_b_proj) = target_cos.
    Preserves norm of deltas_b.
    """
    a = flatten_deltas(deltas_a)
    b = flatten_deltas(deltas_b)
    a_norm = mx.sqrt(mx.sum(a * a))
    b_norm = mx.sqrt(mx.sum(b * b))
    a_hat = a / (a_norm + 1e-12)
    b_parallel = mx.sum(b * a_hat) * a_hat
    b_perp = b - b_parallel
    b_perp_norm = mx.sqrt(mx.sum(b_perp * b_perp))
    b_perp_hat = b_perp / (b_perp_norm + 1e-12)
    sin_component = math.sqrt(max(0, 1 - target_cos ** 2))
    b_proj = target_cos * b_norm * a_hat + sin_component * b_norm * b_perp_hat
    actual_cos = (mx.sum(a * b_proj) / (a_norm * mx.sqrt(mx.sum(b_proj * b_proj)) + 1e-12)).item()
    mx.eval(b_proj)
    return unflatten_deltas(b_proj, deltas_b), actual_cos


def project_multi_expert(all_deltas, expert_idx, target_cos):
    """Project expert at expert_idx to have target_cos with expert 0.
    For N>2: project each non-reference expert against reference (expert 0).
    """
    if expert_idx == 0:
        return all_deltas[0], 0.0  # reference, no projection
    return project_to_target_cosine(all_deltas[0], all_deltas[expert_idx], target_cos)


def copy_base_to_lora(base_model, lora_model):
    """Copy base model weights into LoRA model's linear layers."""
    for l_idx in range(len(base_model.layers)):
        bl = base_model.layers[l_idx]
        ll = lora_model.layers[l_idx]
        ll.attn.wq.weight = bl.attn.wq.weight
        ll.attn.wk.weight = bl.attn.wk.weight
        ll.attn.wv.weight = bl.attn.wv.weight
        ll.attn.wo.weight = bl.attn.wo.weight
        ll.mlp.fc1.linear.weight = bl.mlp.fc1.weight
        ll.mlp.fc2.linear.weight = bl.mlp.fc2.weight
    lora_model.wte.weight = base_model.wte.weight
    lora_model.wpe.weight = base_model.wpe.weight
    lora_model.lm_head.weight = base_model.lm_head.weight
    mx.eval(lora_model.parameters())


def freeze_except_lora(model):
    """Freeze all params except LoRA A and B."""
    model.freeze()
    for layer in model.layers:
        layer.mlp.fc1.unfreeze()
        layer.mlp.fc2.unfreeze()
        layer.mlp.fc1.linear.freeze()
        layer.mlp.fc2.linear.freeze()


def apply_deltas_to_base(base_model, deltas, vocab_size):
    """Create a GPT with base weights + baked-in deltas."""
    model = GPT(vocab_size=vocab_size, **BRIDGE)
    mx.eval(model.parameters())
    # Copy base weights
    pairs = list(zip(
        [k for k, _ in nn.utils.tree_flatten(base_model.parameters())],
        [v for _, v in nn.utils.tree_flatten(base_model.parameters())]
    ))
    model.load_weights(pairs)
    mx.eval(model.parameters())
    # Apply deltas
    for l_idx, name, delta in deltas:
        layer = model.layers[l_idx]
        if name == 'fc1':
            layer.mlp.fc1.weight = layer.mlp.fc1.weight + delta.T
        elif name == 'fc2':
            layer.mlp.fc2.weight = layer.mlp.fc2.weight + delta.T
    mx.eval(model.parameters())
    return model


# ── Routed Model ───────────────────────────────────────────────────────────

class RoutedDeltaGPT(nn.Module):
    """GPT with routed LoRA deltas. Supports N>=2 experts."""

    def __init__(self, base_model, delta_sets, vocab_size, top_k=2):
        super().__init__()
        self.n_experts = len(delta_sets)
        self.top_k = min(top_k, self.n_experts)
        n_embd = BRIDGE['n_embd']
        n_layer = len(base_model.layers)

        self.wte = base_model.wte
        self.wpe = base_model.wpe
        self.norm0 = base_model.norm0
        self.base_layers = base_model.layers
        self.lm_head = base_model.lm_head

        self._expert_fc1_weights = []
        self._expert_fc2_weights = []

        for l_idx in range(n_layer):
            base_fc1 = base_model.layers[l_idx].mlp.fc1.weight
            base_fc2 = base_model.layers[l_idx].mlp.fc2.weight
            fc1_list, fc2_list = [], []
            for deltas in delta_sets:
                for dl_idx, name, delta in deltas:
                    if dl_idx == l_idx and name == 'fc1':
                        fc1_list.append(base_fc1 + delta.T)
                    elif dl_idx == l_idx and name == 'fc2':
                        fc2_list.append(base_fc2 + delta.T)
            self._expert_fc1_weights.append(mx.stack(fc1_list))
            self._expert_fc2_weights.append(mx.stack(fc2_list))

        mx.eval(self._expert_fc1_weights + self._expert_fc2_weights)
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
            x = x + base_layer.attn(base_layer.norm1(x))
            h = base_layer.norm2(x)
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

    def aux_loss(self):
        return mx.array(0.0)


# ── Gap Measurement ────────────────────────────────────────────────────────

def measure_function_space_gap(composed_model, joint_model, dataset, n_batches=20):
    """Measure CE gap and KL divergence between composed and joint models."""
    rng = random.Random(0)
    total_ce_c, total_ce_j, total_kl, total_l1 = 0.0, 0.0, 0.0, 0.0
    n_tokens = 0

    for _ in range(n_batches):
        inputs, targets = dataset.get_batch(BATCH_SIZE, rng)
        logits_c = composed_model(inputs)
        logits_j = joint_model(inputs)
        B, T, V = logits_c.shape

        ce_c = nn.losses.cross_entropy(
            logits_c.reshape(B * T, V), targets.reshape(B * T), reduction="sum")
        ce_j = nn.losses.cross_entropy(
            logits_j.reshape(B * T, V), targets.reshape(B * T), reduction="sum")
        total_ce_c += ce_c.item()
        total_ce_j += ce_j.item()

        prob_c = mx.softmax(logits_c.reshape(B * T, V), axis=-1)
        prob_j = mx.softmax(logits_j.reshape(B * T, V), axis=-1)
        kl = mx.sum(prob_j * (mx.log(prob_j + 1e-10) - mx.log(prob_c + 1e-10))).item()
        l1 = mx.sum(mx.abs(prob_c - prob_j)).item()
        total_kl += kl
        total_l1 += l1
        n_tokens += B * T

    return {
        'ce_gap': abs(total_ce_c / n_tokens - total_ce_j / n_tokens),
        'kl_gap': total_kl / n_tokens,
        'prob_l1': total_l1 / n_tokens,
    }


# ── Calibration ────────────────────────────────────────────────────────────

def calibrate_router(model, domain_datasets, val_ds, joint_val_loss,
                     steps=500, lr=1e-3, seed=42):
    """Calibrate router, tracking convergence."""
    model.freeze()
    for router in model.routers:
        router.unfreeze()

    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    convergence_target = joint_val_loss * (1 + CONVERGENCE_THRESHOLD)
    loss_curve = []
    steps_to_converge = None
    domains = list(domain_datasets.keys())

    for step in range(1, steps + 1):
        # Round-robin over domains
        domain = domains[step % len(domains)]
        inputs, targets = domain_datasets[domain].get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % CAL_EVAL_EVERY == 0:
            val_loss = evaluate(model, val_ds, BATCH_SIZE, n_batches=5)
            loss_curve.append((step, val_loss))
            if steps_to_converge is None and val_loss <= convergence_target:
                steps_to_converge = step

    final_val_loss = evaluate(model, val_ds, BATCH_SIZE, n_batches=10)
    model.unfreeze()

    auc = sum(vl for _, vl in loss_curve) / len(loss_curve) if loss_curve else float('inf')

    return {
        'loss_curve': loss_curve,
        'steps_to_converge': steps_to_converge,
        'final_val_loss': final_val_loss,
        'auc': auc,
    }


# ── Single Trial (one cosine level, one expert config) ─────────────────────

def run_trial(target_cos, base_model, all_deltas, joint_model,
              domain_train_datasets, domain_val_datasets, joint_val_ds,
              joint_val_loss, V, n_experts, top_k, seed=42):
    """Run one trial at a specific cosine level with N experts."""
    # Project all experts (except reference) to target cosine
    projected_deltas = []
    for e_idx in range(n_experts):
        if e_idx == 0:
            projected_deltas.append(all_deltas[e_idx])
        else:
            proj, actual_cos = project_to_target_cosine(
                all_deltas[0], all_deltas[e_idx], target_cos)
            projected_deltas.append(proj)

    # Task arithmetic: average all deltas
    n_matrices = len(all_deltas[0])
    ta_deltas = []
    for m_idx in range(n_matrices):
        l_idx, name, _ = all_deltas[0][m_idx]
        avg = sum(projected_deltas[e][m_idx][2] for e in range(n_experts)) / n_experts
        ta_deltas.append((l_idx, name, avg))

    ta_model = apply_deltas_to_base(base_model, ta_deltas, V)

    # Measure gap (task arithmetic vs joint)
    gap = measure_function_space_gap(ta_model, joint_model, joint_val_ds)

    # Create routed model and calibrate
    base_copy = GPT(vocab_size=V, **BRIDGE)
    mx.eval(base_copy.parameters())
    pairs = list(zip(
        [k for k, _ in nn.utils.tree_flatten(base_model.parameters())],
        [v for _, v in nn.utils.tree_flatten(base_model.parameters())]
    ))
    base_copy.load_weights(pairs)
    mx.eval(base_copy.parameters())

    routed = RoutedDeltaGPT(base_copy, projected_deltas, V, top_k=top_k)
    mx.eval(routed.parameters())

    # Gap before calibration
    gap_pre = measure_function_space_gap(routed, joint_model, joint_val_ds, n_batches=10)

    # Calibrate
    cal = calibrate_router(
        routed, domain_train_datasets, joint_val_ds,
        joint_val_loss, steps=MAX_CAL_STEPS, lr=LR, seed=seed
    )

    # Gap after calibration
    gap_post = measure_function_space_gap(routed, joint_model, joint_val_ds, n_batches=10)

    # Per-domain eval
    domain_vals = {}
    for dname, dval in domain_val_datasets.items():
        domain_vals[dname] = evaluate(routed, dval, BATCH_SIZE, n_batches=5)
    avg_domain_val = statistics.mean(domain_vals.values())
    vs_joint_pct = (avg_domain_val - joint_val_loss) / joint_val_loss * 100

    return {
        'target_cos': target_cos,
        'n_experts': n_experts,
        'top_k': top_k,
        'ce_gap_ta': gap['ce_gap'],
        'kl_gap_ta': gap['kl_gap'],
        'prob_l1_ta': gap['prob_l1'],
        'ce_gap_pre': gap_pre['ce_gap'],
        'kl_gap_pre': gap_pre['kl_gap'],
        'ce_gap_post': gap_post['ce_gap'],
        'kl_gap_post': gap_post['kl_gap'],
        'final_val_loss': cal['final_val_loss'],
        'avg_domain_val': avg_domain_val,
        'vs_joint_pct': vs_joint_pct,
        'steps_to_converge': cal['steps_to_converge'],
        'auc': cal['auc'],
        'domain_vals': domain_vals,
    }


# ── Full Experiment (one seed) ─────────────────────────────────────────────

def run_experiment(seed=42, n_experts=2, top_k=2, verbose=True):
    """Run the full gap-as-signal experiment for one seed."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"GAP-AS-SIGNAL BRIDGE (seed={seed}, N={n_experts}, k={top_k})")
        print(f"  d={BRIDGE['n_embd']}, n_layer={BRIDGE['n_layer']}, "
              f"rank={LORA_RANK}, block_size={BRIDGE['block_size']}")
        print(f"{'='*70}")

    mx.random.seed(seed)

    # Load data
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    # Domain split - use quintary for 5 domains (or binary for N=2)
    if n_experts <= 2:
        splits = domain_split(docs, method="binary")
        domain_names = list(splits.keys())[:n_experts]
    else:
        splits = domain_split(docs, method="quintary")
        domain_names = list(splits.keys())[:n_experts]

    # Prepare datasets
    all_train, all_val = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_train, tokenizer, BRIDGE['block_size'])
    joint_val = CharDataset(all_val, tokenizer, BRIDGE['block_size'])

    domain_train = {}
    domain_val = {}
    for dname in domain_names:
        dtrain, dval = train_val_split(splits[dname], seed=seed)
        domain_train[dname] = CharDataset(dtrain, tokenizer, BRIDGE['block_size'])
        domain_val[dname] = CharDataset(dval, tokenizer, BRIDGE['block_size'])

    # === 1. Joint training baseline ===
    if verbose:
        print("\n--- Joint training ---")
    model_joint = GPT(vocab_size=V, **BRIDGE)
    mx.eval(model_joint.parameters())
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(model_joint, ntp_loss)
    total_steps = FINETUNE_STEPS * n_experts  # scale with domains
    for step in range(1, total_steps + 1):
        dname = domain_names[step % n_experts]
        inputs, targets = domain_train[dname].get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model_joint, inputs, targets)
        optimizer.update(model_joint, grads)
        mx.eval(model_joint.parameters(), optimizer.state)
        if verbose and step % 200 == 0:
            print(f"  step {step}/{total_steps} | loss {loss.item():.4f}")

    joint_domain_vals = {}
    for dname in domain_names:
        joint_domain_vals[dname] = evaluate(model_joint, domain_val[dname], BATCH_SIZE)
    joint_val_loss = statistics.mean(joint_domain_vals.values())
    joint_on_joint = evaluate(model_joint, joint_val, BATCH_SIZE)
    if verbose:
        print(f"  Joint val: avg={joint_val_loss:.4f}, joint_ds={joint_on_joint:.4f}")
        for dname, v in joint_domain_vals.items():
            print(f"    {dname}: {v:.4f}")

    # === 2. Pretrain base model ===
    if verbose:
        print("\n--- Pretraining base ---")
    base_model = GPT(vocab_size=V, **BRIDGE)
    mx.eval(base_model.parameters())
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(base_model, ntp_loss)
    for step in range(1, PRETRAIN_STEPS + 1):
        inputs, targets = joint_train.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(base_model, inputs, targets)
        optimizer.update(base_model, grads)
        mx.eval(base_model.parameters(), optimizer.state)
    if verbose:
        base_val = evaluate(base_model, joint_val, BATCH_SIZE)
        print(f"  Base val: {base_val:.4f}")

    # === 3. Fine-tune LoRA experts ===
    all_deltas = []
    for e_idx, dname in enumerate(domain_names):
        if verbose:
            print(f"\n--- Fine-tuning LoRA expert {e_idx} ({dname}) ---")
        lora_model = LoRAGPT(vocab_size=V, **BRIDGE,
                             lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA)
        mx.eval(lora_model.parameters())
        copy_base_to_lora(base_model, lora_model)
        freeze_except_lora(lora_model)

        rng = random.Random(seed + e_idx)
        optimizer = optim.Adam(learning_rate=LR)
        loss_and_grad = nn.value_and_grad(lora_model, lambda m, i, t: ntp_loss(m, i, t))
        for step in range(1, FINETUNE_STEPS + 1):
            inputs, targets = domain_train[dname].get_batch(BATCH_SIZE, rng)
            loss, grads = loss_and_grad(lora_model, inputs, targets)
            optimizer.update(lora_model, grads)
            mx.eval(lora_model.parameters(), optimizer.state)

        lora_model.unfreeze()
        deltas = get_deltas(lora_model)
        all_deltas.append(deltas)

        if verbose:
            val_loss = evaluate(lora_model, domain_val[dname], BATCH_SIZE)
            print(f"  Expert {e_idx} val ({dname}): {val_loss:.4f}")

    # === 4. Measure natural cosine between experts ===
    natural_cosines = {}
    for i in range(n_experts):
        for j in range(i + 1, n_experts):
            flat_i = flatten_deltas(all_deltas[i])
            flat_j = flatten_deltas(all_deltas[j])
            cos = (mx.sum(flat_i * flat_j) /
                   (mx.sqrt(mx.sum(flat_i**2)) * mx.sqrt(mx.sum(flat_j**2)) + 1e-12)).item()
            natural_cosines[f"{i}-{j}"] = cos
    if verbose:
        print(f"\n  Natural cosines: {natural_cosines}")

    # === 5. Run trials at each target cosine ===
    if verbose:
        print(f"\n{'='*70}")
        print("RUNNING COSINE SWEEP")
        print(f"{'='*70}")

    trials = []
    for target_cos in TARGET_COSINES:
        if verbose:
            print(f"\n  --- cos = {target_cos:.1f} ---")
        trial = run_trial(
            target_cos, base_model, all_deltas, model_joint,
            domain_train, domain_val, joint_val,
            joint_on_joint, V, n_experts, top_k, seed=seed
        )
        trials.append(trial)
        if verbose:
            print(f"    CE gap: {trial['ce_gap_ta']:.4f}, "
                  f"final VL: {trial['final_val_loss']:.4f}, "
                  f"vs joint: {trial['vs_joint_pct']:+.1f}%")

    return {
        'seed': seed,
        'n_experts': n_experts,
        'top_k': top_k,
        'joint_val_loss': joint_val_loss,
        'joint_on_joint': joint_on_joint,
        'joint_domain_vals': joint_domain_vals,
        'natural_cosines': natural_cosines,
        'trials': trials,
    }


# ── Statistical Analysis ──────────────────────────────────────────────────

def compute_r_squared(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_yy = sum((y - mean_y) ** 2 for y in ys)
    if ss_xx == 0 or ss_yy == 0:
        return 0.0, 0.0
    r = ss_xy / (ss_xx * ss_yy) ** 0.5
    return r ** 2, r


def bootstrap_ci(xs, ys, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval for r^2."""
    n = len(xs)
    if n < 3:
        return 0.0, 0.0, 0.0
    rng = random.Random(42)
    r2_samples = []
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        bx = [xs[i] for i in indices]
        by = [ys[i] for i in indices]
        r2, _ = compute_r_squared(bx, by)
        r2_samples.append(r2)
    r2_samples.sort()
    alpha = (1 - ci) / 2
    lo = r2_samples[int(alpha * n_bootstrap)]
    hi = r2_samples[int((1 - alpha) * n_bootstrap)]
    return statistics.mean(r2_samples), lo, hi


def check_monotonic(values):
    """Check if values are monotonically non-decreasing."""
    for i in range(1, len(values)):
        if values[i] < values[i-1] - 1e-6:
            return False
    return True


def analyze_results(all_experiments, config_label=""):
    """Comprehensive analysis with CIs."""
    print(f"\n\n{'='*80}")
    print(f"GAP-AS-SIGNAL BRIDGE ANALYSIS {config_label}")
    print(f"  d={BRIDGE['n_embd']}, rank={LORA_RANK}, "
          f"n_seeds={len(all_experiments)}")
    print(f"{'='*80}")

    # Aggregate by cosine level
    by_cos = {}
    for exp in all_experiments:
        for trial in exp['trials']:
            cos = trial['target_cos']
            if cos not in by_cos:
                by_cos[cos] = []
            by_cos[cos].append(trial)

    # Summary table with CIs
    print(f"\n{'Cos':>5} | {'CE Gap':>10} | {'KL Gap':>10} | {'vs Joint':>12} | "
          f"{'AUC':>10} | {'Steps':>8} | n")
    print("-" * 75)

    cos_means_quality = {}
    for cos in sorted(by_cos.keys()):
        trials = by_cos[cos]
        ce_gaps = [t['ce_gap_ta'] for t in trials]
        kl_gaps = [t['kl_gap_ta'] for t in trials]
        vs_joints = [t['vs_joint_pct'] for t in trials]
        aucs = [t['auc'] for t in trials]
        steps_conv = [t['steps_to_converge'] for t in trials if t['steps_to_converge']]

        n = len(trials)
        ce_mean = statistics.mean(ce_gaps)
        ce_std = statistics.stdev(ce_gaps) if n > 1 else 0
        kl_mean = statistics.mean(kl_gaps)
        vj_mean = statistics.mean(vs_joints)
        vj_std = statistics.stdev(vs_joints) if n > 1 else 0
        auc_mean = statistics.mean(aucs)
        steps_str = f"{statistics.mean(steps_conv):.0f}" if steps_conv else "N/C"

        cos_means_quality[cos] = vj_mean

        print(f"{cos:>5.1f} | {ce_mean:>7.4f}+{ce_std:.3f} | {kl_mean:>10.4f} | "
              f"{vj_mean:>+7.1f}%+{vj_std:.1f} | {auc_mean:>10.4f} | {steps_str:>8} | {n}")

    # Correlation analysis on ALL individual trial data
    ce_gaps_all, vs_joint_all, cos_list = [], [], []
    kl_gaps_all, auc_all = [], []
    for exp in all_experiments:
        for trial in exp['trials']:
            ce_gaps_all.append(trial['ce_gap_ta'])
            kl_gaps_all.append(trial['kl_gap_ta'])
            vs_joint_all.append(trial['vs_joint_pct'])
            auc_all.append(trial['auc'])
            cos_list.append(trial['target_cos'])

    # Also compute on per-seed cosine-level means (addressing reviewer concern)
    # Each seed contributes 7 points (one per cosine level), compute r^2 on means
    mean_ce_by_cos = []
    mean_vj_by_cos = []
    for cos in sorted(by_cos.keys()):
        trials = by_cos[cos]
        mean_ce_by_cos.append(statistics.mean([t['ce_gap_ta'] for t in trials]))
        mean_vj_by_cos.append(statistics.mean([t['vs_joint_pct'] for t in trials]))

    print(f"\n{'='*80}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*80}")

    # 1. Pooled (all individual trials)
    r2_ce_q_pooled, r_ce_q = compute_r_squared(ce_gaps_all, vs_joint_all)
    r2_ce_q_mean, r2_ce_q_lo, r2_ce_q_hi = bootstrap_ci(ce_gaps_all, vs_joint_all)
    print(f"\n1. CE Gap vs Quality (pooled, n={len(ce_gaps_all)}):")
    print(f"   r^2 = {r2_ce_q_pooled:.4f}, r = {r_ce_q:.4f}")
    print(f"   Bootstrap 95% CI: [{r2_ce_q_lo:.4f}, {r2_ce_q_hi:.4f}]")

    # 2. Mean curve (7 points)
    r2_ce_q_mean_curve, r_ce_q_mean = compute_r_squared(mean_ce_by_cos, mean_vj_by_cos)
    print(f"\n2. CE Gap vs Quality (mean curve, n=7):")
    print(f"   r^2 = {r2_ce_q_mean_curve:.4f}, r = {r_ce_q_mean:.4f}")

    # 3. Cosine vs Quality
    r2_cos_q, r_cos_q = compute_r_squared(cos_list, vs_joint_all)
    r2_cos_q_mean, r2_cos_q_lo, r2_cos_q_hi = bootstrap_ci(cos_list, vs_joint_all)
    print(f"\n3. Cosine vs Quality (pooled):")
    print(f"   r^2 = {r2_cos_q:.4f}, r = {r_cos_q:.4f}")
    print(f"   Bootstrap 95% CI: [{r2_cos_q_lo:.4f}, {r2_cos_q_hi:.4f}]")

    # 4. KL Gap vs Quality
    r2_kl_q, r_kl_q = compute_r_squared(kl_gaps_all, vs_joint_all)
    print(f"\n4. KL Gap vs Quality (pooled):")
    print(f"   r^2 = {r2_kl_q:.4f}, r = {r_kl_q:.4f}")

    # 5. Cosine vs CE Gap (sanity)
    r2_cos_ce, _ = compute_r_squared(cos_list, ce_gaps_all)
    print(f"\n5. Cosine vs CE Gap (sanity):")
    print(f"   r^2 = {r2_cos_ce:.4f}")

    # Monotonicity check
    quality_curve = [cos_means_quality[c] for c in sorted(cos_means_quality.keys())]
    is_monotonic = check_monotonic(quality_curve)
    print(f"\n6. Monotonicity of quality vs cosine:")
    print(f"   Quality curve: {[f'{v:+.1f}%' for v in quality_curve]}")
    print(f"   Monotonic: {'YES' if is_monotonic else 'NO'}")

    # Effect size: cos=0.0 vs cos=0.5
    if 0.0 in cos_means_quality and 0.5 in cos_means_quality:
        effect = cos_means_quality[0.5] - cos_means_quality[0.0]
        print(f"\n7. Effect size (cos=0.0 vs cos=0.5):")
        print(f"   {effect:+.2f}pp")

    # === Kill Criteria ===
    print(f"\n{'='*80}")
    print("KILL CRITERIA")
    print(f"{'='*80}")

    best_r2 = max(r2_ce_q_pooled, r2_cos_q, r2_kl_q)
    kills = 0
    passes = 0

    # Kill 1: r^2 < 0.3
    if best_r2 >= 0.3:
        passes += 1
        print(f"\n  [PASS] Gap-calibration correlation: r^2 = {best_r2:.4f} >= 0.3")
        print(f"         95% CI on CE gap: [{r2_ce_q_lo:.4f}, {r2_ce_q_hi:.4f}]")
    else:
        kills += 1
        print(f"\n  [KILL] Gap-calibration correlation: r^2 = {best_r2:.4f} < 0.3")

    # Kill 2: Non-monotonic
    if is_monotonic:
        passes += 1
        print(f"  [PASS] Monotonic relationship: quality degrades with cosine")
    else:
        kills += 1
        print(f"  [KILL] Non-monotonic: quality does NOT consistently degrade")

    # Kill 3: Effect size < 0.5pp
    if 0.0 in cos_means_quality and 0.5 in cos_means_quality:
        effect = cos_means_quality[0.5] - cos_means_quality[0.0]
        if effect > 0.5:
            passes += 1
            print(f"  [PASS] Effect size: {effect:+.2f}pp between cos=0.0 and cos=0.5")
        else:
            kills += 1
            print(f"  [KILL] Effect size: {effect:+.2f}pp < 0.5pp (too small)")

    # Calibration steps check (new kill criterion for macro)
    all_steps = [t['steps_to_converge'] for exp in all_experiments
                 for t in exp['trials'] if t['steps_to_converge']]
    if all_steps:
        max_steps = max(all_steps)
        mean_steps = statistics.mean(all_steps)
        if max_steps <= 500:
            passes += 1
            print(f"  [PASS] Calibration steps: max={max_steps}, mean={mean_steps:.0f} <= 500")
        else:
            kills += 1
            print(f"  [KILL] Calibration steps: max={max_steps} > 500 (too slow)")
    else:
        print(f"  [WARN] No trials converged within {MAX_CAL_STEPS} steps")

    # Natural cosine check
    all_nat_cos = []
    for exp in all_experiments:
        all_nat_cos.extend(exp['natural_cosines'].values())
    if all_nat_cos:
        mean_nat = statistics.mean(all_nat_cos)
        print(f"\n  Natural cosine (d={BRIDGE['n_embd']}): {mean_nat:.4f}")
        expected = LORA_RANK / math.sqrt(BRIDGE['n_embd'] * 4 * BRIDGE['n_embd'] *
                                          2 * BRIDGE['n_layer'])
        print(f"  Theoretical prediction (r/sqrt(D)): {expected:.4f}")

    print(f"\n{'='*80}")
    if kills == 0:
        print(f"  HYPOTHESIS CONFIRMED at d={BRIDGE['n_embd']}: {passes}/{passes} criteria pass")
    elif kills >= 2:
        print(f"  HYPOTHESIS KILLED: {kills} criteria fail")
    else:
        print(f"  PARTIAL: {passes} pass, {kills} fail")
    print(f"{'='*80}")

    return {
        'r2_ce_quality_pooled': r2_ce_q_pooled,
        'r2_ce_quality_ci_lo': r2_ce_q_lo,
        'r2_ce_quality_ci_hi': r2_ce_q_hi,
        'r2_ce_quality_mean_curve': r2_ce_q_mean_curve,
        'r2_cos_quality': r2_cos_q,
        'r2_kl_quality': r2_kl_q,
        'is_monotonic': is_monotonic,
        'best_r2': best_r2,
        'n_seeds': len(all_experiments),
        'kills': kills,
        'passes': passes,
    }


# ── Multi-Seed Runner ─────────────────────────────────────────────────────

def run_bridge(n_seeds=None, n_experts=2, top_k=2, verbose=True):
    """Run the full bridge experiment across multiple seeds."""
    if n_seeds is None:
        n_seeds = N_SEEDS

    t0 = time.time()
    all_experiments = []
    seeds = list(range(42, 42 + n_seeds))

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n\n{'#'*80}")
            print(f"# SEED {i+1}/{n_seeds} (seed={seed})")
            print(f"{'#'*80}")
        result = run_experiment(seed=seed, n_experts=n_experts,
                               top_k=top_k, verbose=verbose)
        all_experiments.append(result)

        # Save intermediate results
        out_dir = Path(__file__).parent
        with open(out_dir / f"results_N{n_experts}_k{top_k}.json", "w") as f:
            json.dump(all_experiments, f, indent=2, default=str)

    config_label = f"(N={n_experts}, k={top_k})"
    analysis = analyze_results(all_experiments, config_label)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return all_experiments, analysis


def run_quick_validation(verbose=True):
    """Quick 3-seed run with N=2 for sanity check before full 20-seed run."""
    print("="*80)
    print("QUICK VALIDATION: 3 seeds, N=2, k=2")
    print("="*80)
    return run_bridge(n_seeds=3, n_experts=2, top_k=2, verbose=verbose)


def run_full_experiment(verbose=True):
    """Full experiment: 20 seeds x 2 configs (N=2 and N=4)."""
    results = {}

    # Phase 1a: N=2, k=2 (match micro)
    print("\n" + "="*80)
    print("PHASE 1a: N=2, k=2 (micro-matched)")
    print("="*80)
    exps_2, analysis_2 = run_bridge(n_seeds=N_SEEDS, n_experts=2, top_k=2, verbose=verbose)
    results['N2_k2'] = {'experiments': exps_2, 'analysis': analysis_2}

    # Phase 1b: N=4, k=2 (real routing, addresses reviewer concern)
    # Only run if we have the quintary domain split (needs 4+ domains)
    print("\n" + "="*80)
    print("PHASE 1b: N=4, k=2 (routing selection)")
    print("="*80)
    exps_4, analysis_4 = run_bridge(n_seeds=N_SEEDS, n_experts=4, top_k=2, verbose=verbose)
    results['N4_k2'] = {'experiments': exps_4, 'analysis': analysis_4}

    # Save all results
    out_dir = Path(__file__).parent
    summary = {
        'config': {
            'd': BRIDGE['n_embd'],
            'n_head': BRIDGE['n_head'],
            'n_layer': BRIDGE['n_layer'],
            'block_size': BRIDGE['block_size'],
            'lora_rank': LORA_RANK,
            'lora_alpha': LORA_ALPHA,
        },
        'N2_k2': analysis_2,
        'N4_k2': analysis_4,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n\n" + "="*80)
    print("COMPARATIVE SUMMARY")
    print("="*80)
    for label, analysis in [("N=2,k=2", analysis_2), ("N=4,k=2", analysis_4)]:
        print(f"\n  {label}:")
        print(f"    Best r^2: {analysis['best_r2']:.4f}")
        print(f"    CE gap r^2: {analysis['r2_ce_quality_pooled']:.4f} "
              f"[{analysis['r2_ce_quality_ci_lo']:.4f}, "
              f"{analysis['r2_ce_quality_ci_hi']:.4f}]")
        print(f"    Mean curve r^2: {analysis['r2_ce_quality_mean_curve']:.4f}")
        print(f"    Monotonic: {analysis['is_monotonic']}")
        print(f"    Verdict: {analysis['passes']} PASS, {analysis['kills']} KILL")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Run quick 3-seed validation")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Number of seeds (default: 20 for full, 3 for quick)")
    parser.add_argument("--n-experts", type=int, default=2,
                        help="Number of experts (2 or 4)")
    parser.add_argument("--top-k", type=int, default=2,
                        help="Top-k routing")
    args = parser.parse_args()

    if args.quick:
        run_quick_validation()
    elif args.seeds:
        run_bridge(n_seeds=args.seeds, n_experts=args.n_experts, top_k=args.top_k)
    else:
        run_full_experiment()
