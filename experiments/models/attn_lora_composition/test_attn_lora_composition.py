"""Attention LoRA Composition Experiment.

Tests whether adding LoRA adapters to attention Wq/Wk (rank 4) alongside
MLP LoRA (rank 8) closes the composition gap compared to MLP-only LoRA.

The key comparison:
- MLP-only LoRA composition (baseline, from lora_procrustes)
- MLP+Attention LoRA composition (this experiment)
- Joint training (gold standard)

For each, we measure:
1. Single-domain quality (fine-tuned LoRA on its own domain)
2. Composition quality (composed model on all domains)
3. Composition gap = (composed - joint) / joint

Kill criteria:
- Attention adapters improve composition gap <1% over MLP-only capsules
- Attention adapters degrade single-domain quality >2%
"""

import copy
import random
import statistics

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate, ntp_loss
from experiments.models.gpt.gpt import GPT


# ── Config ──────────────────────────────────────────────────────────────────

BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
MLP_RANK = 8
MLP_ALPHA = 1.0
ATTN_RANK = 4
ATTN_ALPHA = 1.0
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


def count_total_params(model) -> int:
    return sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))


def freeze_except_lora_mlp(model):
    """Freeze all except MLP LoRA A/B (for MLP-only baseline)."""
    model.freeze()
    for layer in model.layers:
        layer.mlp.fc1.unfreeze()
        layer.mlp.fc2.unfreeze()
        layer.mlp.fc1.linear.freeze()
        layer.mlp.fc2.linear.freeze()


def freeze_except_lora_all(model):
    """Freeze all except MLP LoRA AND attention LoRA A/B."""
    model.freeze()
    for layer in model.layers:
        # MLP LoRA
        layer.mlp.fc1.unfreeze()
        layer.mlp.fc2.unfreeze()
        layer.mlp.fc1.linear.freeze()
        layer.mlp.fc2.linear.freeze()
        # Attention LoRA (Wq, Wk have LoRALinear structure)
        layer.attn.wq.unfreeze()
        layer.attn.wk.unfreeze()
        layer.attn.wq.linear.freeze()
        layer.attn.wk.linear.freeze()


def reset_lora_all(model):
    """Reset ALL LoRA A/B matrices to initialization."""
    for layer in model.layers:
        for fc in [layer.mlp.fc1, layer.mlp.fc2, layer.attn.wq, layer.attn.wk]:
            in_dim = fc.A.shape[0]
            rank = fc.A.shape[1]
            scale = (2.0 / in_dim) ** 0.5
            fc.A = mx.random.normal(fc.A.shape) * scale
            fc.B = mx.zeros(fc.B.shape)
    mx.eval(model.parameters())


def get_deltas(model, include_attn=True) -> list:
    """Extract LoRA delta matrices from model.

    Returns: list of (layer_idx, sublayer_name, delta) where delta = (alpha/r) * A @ B.
    """
    deltas = []
    for l_idx, layer in enumerate(model.layers):
        for name, fc in [('fc1', layer.mlp.fc1), ('fc2', layer.mlp.fc2)]:
            delta = (fc.alpha / fc.rank) * (fc.A @ fc.B)
            deltas.append((l_idx, name, delta))
        if include_attn and hasattr(layer.attn, 'wq') and hasattr(layer.attn.wq, 'A'):
            for name, fc in [('wq', layer.attn.wq), ('wk', layer.attn.wk)]:
                delta = (fc.alpha / fc.rank) * (fc.A @ fc.B)
                deltas.append((l_idx, name, delta))
    return deltas


def copy_base_to_attn_lora(base_model, lora_model):
    """Copy base GPT weights into an AttnLoRAGPT model."""
    for l_idx in range(BASE['n_layer']):
        bl = base_model.layers[l_idx]
        ll = lora_model.layers[l_idx]
        # Attention base weights
        ll.attn.wq.linear.weight = bl.attn.wq.weight
        ll.attn.wk.linear.weight = bl.attn.wk.weight
        ll.attn.wv.weight = bl.attn.wv.weight
        ll.attn.wo.weight = bl.attn.wo.weight
        # MLP base weights
        ll.mlp.fc1.linear.weight = bl.mlp.fc1.weight
        ll.mlp.fc2.linear.weight = bl.mlp.fc2.weight
    lora_model.wte.weight = base_model.wte.weight
    lora_model.wpe.weight = base_model.wpe.weight
    lora_model.lm_head.weight = base_model.lm_head.weight
    mx.eval(lora_model.parameters())


def copy_base_to_mlp_lora(base_model, lora_model):
    """Copy base GPT weights into a LoRAGPT model (MLP-only LoRA)."""
    for l_idx in range(BASE['n_layer']):
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


class RoutedDeltaGPT(nn.Module):
    """GPT with routed LoRA deltas applied to both MLP and attention weights.

    Base weights are frozen. A lightweight router selects which set of deltas
    to apply per token. Deltas are applied as additive linear corrections.
    """

    def __init__(self, base_model, delta_sets, vocab_size,
                 top_k: int = 2, uniform: bool = False,
                 has_attn_deltas: bool = False):
        super().__init__()
        self.n_experts = len(delta_sets)
        self.top_k = min(top_k, self.n_experts)
        self.uniform = uniform
        self.has_attn_deltas = has_attn_deltas
        n_embd = BASE['n_embd']
        n_layer = len(base_model.layers)

        # Copy base model references
        self.wte = base_model.wte
        self.wpe = base_model.wpe
        self.norm0 = base_model.norm0
        self.base_layers = base_model.layers
        self.lm_head = base_model.lm_head

        # Pre-build expert weight matrices
        self._expert_fc1_weights = []
        self._expert_fc2_weights = []
        if has_attn_deltas:
            self._expert_wq_weights = []
            self._expert_wk_weights = []

        for l_idx in range(n_layer):
            base_fc1_w = base_model.layers[l_idx].mlp.fc1.weight
            base_fc2_w = base_model.layers[l_idx].mlp.fc2.weight

            fc1_list, fc2_list = [], []
            wq_list, wk_list = [], []

            for expert_idx, deltas in enumerate(delta_sets):
                for dl_idx, name, delta in deltas:
                    if dl_idx != l_idx:
                        continue
                    if name == 'fc1':
                        fc1_list.append(base_fc1_w + delta.T)
                    elif name == 'fc2':
                        fc2_list.append(base_fc2_w + delta.T)
                    elif name == 'wq' and has_attn_deltas:
                        base_wq = base_model.layers[l_idx].attn.wq.weight
                        wq_list.append(base_wq + delta.T)
                    elif name == 'wk' and has_attn_deltas:
                        base_wk = base_model.layers[l_idx].attn.wk.weight
                        wk_list.append(base_wk + delta.T)

            self._expert_fc1_weights.append(mx.stack(fc1_list))
            self._expert_fc2_weights.append(mx.stack(fc2_list))
            if has_attn_deltas and wq_list:
                self._expert_wq_weights.append(mx.stack(wq_list))
                self._expert_wk_weights.append(mx.stack(wk_list))

        mx.eval(self._expert_fc1_weights + self._expert_fc2_weights)
        if has_attn_deltas:
            mx.eval(self._expert_wq_weights + self._expert_wk_weights)

        # Per-layer router
        self.routers = [nn.Linear(n_embd, self.n_experts, bias=False)
                        for _ in range(n_layer)]
        self.n_layer = n_layer

    def _get_routing_weights(self, h, l_idx):
        """Compute routing weights for this layer."""
        if self.uniform:
            return mx.full((*h.shape[:-1], self.n_experts), 1.0 / self.n_experts)

        scores = self.routers[l_idx](h)
        probs = mx.softmax(scores, axis=-1)
        top_vals = mx.topk(scores, self.top_k, axis=-1)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)
        mask = (scores >= threshold).astype(mx.float32)
        masked_probs = probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)
        return masked_probs

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)

        for l_idx, base_layer in enumerate(self.base_layers):
            h_attn = base_layer.norm1(x)

            if self.has_attn_deltas:
                # Route attention through experts
                weights = self._get_routing_weights(h_attn, l_idx)

                # Compute expert-routed attention
                n_head = base_layer.attn.n_head
                head_dim = base_layer.attn.head_dim

                q_out = mx.zeros_like(h_attn)
                k_out = mx.zeros_like(h_attn)
                for e in range(self.n_experts):
                    w_e = weights[..., e:e+1]
                    q_e = h_attn @ self._expert_wq_weights[l_idx][e].T
                    k_e = h_attn @ self._expert_wk_weights[l_idx][e].T
                    q_out = q_out + w_e * q_e
                    k_out = k_out + w_e * k_e

                # Reshape for attention
                q = q_out.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
                k = k_out.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
                v = base_layer.attn.wv(h_attn).reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)

                scale = head_dim ** -0.5
                attn = (q @ k.transpose(0, 1, 3, 2)) * scale
                mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
                attn = attn + mask
                attn = mx.softmax(attn, axis=-1)
                attn_out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, h_attn.shape[-1])
                attn_out = base_layer.attn.wo(attn_out)
                x = x + attn_out
            else:
                # Standard shared attention
                x = x + base_layer.attn(h_attn)

            # MLP with routed deltas
            h = base_layer.norm2(x)
            weights = self._get_routing_weights(h, l_idx)
            delta_out = mx.zeros_like(h)
            for e in range(self.n_experts):
                w_e = weights[..., e:e+1]
                fc1_w = self._expert_fc1_weights[l_idx][e]
                fc2_w = self._expert_fc2_weights[l_idx][e]
                h_fc1 = h @ fc1_w.T
                h_relu = nn.relu(h_fc1)
                delta_out = delta_out + w_e * (h_relu @ fc2_w.T)
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


def apply_deltas_to_base(base_model, deltas, vocab_size):
    """Create a GPT with base weights + LoRA deltas baked in.

    Handles both MLP deltas (fc1, fc2) and attention deltas (wq, wk).
    """
    model = GPT(vocab_size=vocab_size, **BASE)
    mx.eval(model.parameters())
    copy_weights(base_model, model)

    for l_idx, name, delta in deltas:
        layer = model.layers[l_idx]
        if name == 'fc1':
            layer.mlp.fc1.weight = layer.mlp.fc1.weight + delta.T
        elif name == 'fc2':
            layer.mlp.fc2.weight = layer.mlp.fc2.weight + delta.T
        elif name == 'wq':
            layer.attn.wq.weight = layer.attn.wq.weight + delta.T
        elif name == 'wk':
            layer.attn.wk.weight = layer.attn.wk.weight + delta.T

    mx.eval(model.parameters())
    return model


# ── Main Experiment ─────────────────────────────────────────────────────────

def run_experiment(seed=42):
    """Run the attention LoRA composition experiment."""
    print(f"\n{'='*70}")
    print(f"ATTENTION LORA COMPOSITION (seed={seed})")
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

    # === 3. MLP-only LoRA (baseline condition) ===
    print("\n--- 3a. MLP-only LoRA: domain A ---")
    mlp_a = get_model("lora_gpt", vocab_size=V, **BASE,
                      lora_rank=MLP_RANK, lora_alpha=MLP_ALPHA)
    mx.eval(mlp_a.parameters())
    copy_base_to_mlp_lora(base_model, mlp_a)
    freeze_except_lora_mlp(mlp_a)
    mlp_a_params = count_params(mlp_a)
    print(f"  Trainable params (MLP-only LoRA): {mlp_a_params:,}")
    train(mlp_a, train_a, val_a, steps=FINETUNE_STEPS,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)
    mlp_a.unfreeze()

    print("\n--- 3b. MLP-only LoRA: domain B ---")
    mlp_b = get_model("lora_gpt", vocab_size=V, **BASE,
                      lora_rank=MLP_RANK, lora_alpha=MLP_ALPHA)
    mx.eval(mlp_b.parameters())
    copy_base_to_mlp_lora(base_model, mlp_b)
    freeze_except_lora_mlp(mlp_b)
    train(mlp_b, train_b, val_b, steps=FINETUNE_STEPS,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)
    mlp_b.unfreeze()

    # Single-domain quality for MLP-only
    mlp_deltas_a = get_deltas(mlp_a, include_attn=False)
    mlp_deltas_b = get_deltas(mlp_b, include_attn=False)

    baked_mlp_a = apply_deltas_to_base(base_model, mlp_deltas_a, V)
    baked_mlp_b = apply_deltas_to_base(base_model, mlp_deltas_b, V)
    mlp_single_a = evaluate(baked_mlp_a, val_a, BATCH_SIZE)
    mlp_single_b = evaluate(baked_mlp_b, val_b, BATCH_SIZE)
    results["mlp_single"] = {"a_m": mlp_single_a, "n_z": mlp_single_b,
                             "avg": (mlp_single_a + mlp_single_b) / 2}
    print(f"  MLP-only single: a_m={mlp_single_a:.4f}, n_z={mlp_single_b:.4f}")

    # === 4. MLP+Attention LoRA (experimental condition) ===
    print("\n--- 4a. MLP+Attn LoRA: domain A ---")
    attn_a = get_model("attn_lora_gpt", vocab_size=V, **BASE,
                       mlp_rank=MLP_RANK, mlp_alpha=MLP_ALPHA,
                       attn_rank=ATTN_RANK, attn_alpha=ATTN_ALPHA)
    mx.eval(attn_a.parameters())
    copy_base_to_attn_lora(base_model, attn_a)
    freeze_except_lora_all(attn_a)
    attn_a_params = count_params(attn_a)
    print(f"  Trainable params (MLP+Attn LoRA): {attn_a_params:,}")
    train(attn_a, train_a, val_a, steps=FINETUNE_STEPS,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)
    attn_a.unfreeze()

    print("\n--- 4b. MLP+Attn LoRA: domain B ---")
    attn_b = get_model("attn_lora_gpt", vocab_size=V, **BASE,
                       mlp_rank=MLP_RANK, mlp_alpha=MLP_ALPHA,
                       attn_rank=ATTN_RANK, attn_alpha=ATTN_ALPHA)
    mx.eval(attn_b.parameters())
    copy_base_to_attn_lora(base_model, attn_b)
    freeze_except_lora_all(attn_b)
    train(attn_b, train_b, val_b, steps=FINETUNE_STEPS,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)
    attn_b.unfreeze()

    # Single-domain quality for MLP+Attn
    attn_deltas_a = get_deltas(attn_a, include_attn=True)
    attn_deltas_b = get_deltas(attn_b, include_attn=True)

    baked_attn_a = apply_deltas_to_base(base_model, attn_deltas_a, V)
    baked_attn_b = apply_deltas_to_base(base_model, attn_deltas_b, V)
    attn_single_a = evaluate(baked_attn_a, val_a, BATCH_SIZE)
    attn_single_b = evaluate(baked_attn_b, val_b, BATCH_SIZE)
    results["attn_single"] = {"a_m": attn_single_a, "n_z": attn_single_b,
                              "avg": (attn_single_a + attn_single_b) / 2}
    print(f"  MLP+Attn single: a_m={attn_single_a:.4f}, n_z={attn_single_b:.4f}")

    # === 5. Compose MLP-only LoRA + calibrate router ===
    print("\n--- 5. MLP-only LoRA composed + calibrated ---")
    base_for_mlp = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_for_mlp.parameters())
    copy_weights(base_model, base_for_mlp)

    mlp_composed = RoutedDeltaGPT(base_for_mlp, [mlp_deltas_a, mlp_deltas_b],
                                  V, top_k=2, has_attn_deltas=False)
    mx.eval(mlp_composed.parameters())
    calibrate_router(mlp_composed, train_a, train_b,
                     steps=ROUTER_CAL_STEPS, lr=LR, seed=seed)
    mlp_c_a = evaluate(mlp_composed, val_a, BATCH_SIZE)
    mlp_c_b = evaluate(mlp_composed, val_b, BATCH_SIZE)
    results["mlp_composed"] = {"a_m": mlp_c_a, "n_z": mlp_c_b,
                               "avg": (mlp_c_a + mlp_c_b) / 2}
    print(f"  MLP composed: a_m={mlp_c_a:.4f}, n_z={mlp_c_b:.4f}, "
          f"avg={(mlp_c_a+mlp_c_b)/2:.4f}")

    # === 6. Compose MLP+Attn LoRA + calibrate router ===
    print("\n--- 6. MLP+Attn LoRA composed + calibrated ---")
    base_for_attn = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_for_attn.parameters())
    copy_weights(base_model, base_for_attn)

    attn_composed = RoutedDeltaGPT(base_for_attn, [attn_deltas_a, attn_deltas_b],
                                   V, top_k=2, has_attn_deltas=True)
    mx.eval(attn_composed.parameters())
    calibrate_router(attn_composed, train_a, train_b,
                     steps=ROUTER_CAL_STEPS, lr=LR, seed=seed)
    attn_c_a = evaluate(attn_composed, val_a, BATCH_SIZE)
    attn_c_b = evaluate(attn_composed, val_b, BATCH_SIZE)
    results["attn_composed"] = {"a_m": attn_c_a, "n_z": attn_c_b,
                                "avg": (attn_c_a + attn_c_b) / 2}
    print(f"  MLP+Attn composed: a_m={attn_c_a:.4f}, n_z={attn_c_b:.4f}, "
          f"avg={(attn_c_a+attn_c_b)/2:.4f}")

    # === 7. Ablation: Attn-only LoRA (no MLP) ===
    # Fine-tune only attention LoRA, freeze MLP LoRA at zero
    print("\n--- 7a. Attn-only LoRA: domain A ---")
    attn_only_a = get_model("attn_lora_gpt", vocab_size=V, **BASE,
                            mlp_rank=MLP_RANK, mlp_alpha=MLP_ALPHA,
                            attn_rank=ATTN_RANK, attn_alpha=ATTN_ALPHA)
    mx.eval(attn_only_a.parameters())
    copy_base_to_attn_lora(base_model, attn_only_a)
    # Freeze everything, then unfreeze ONLY attention LoRA
    attn_only_a.freeze()
    for layer in attn_only_a.layers:
        layer.attn.wq.unfreeze()
        layer.attn.wk.unfreeze()
        layer.attn.wq.linear.freeze()
        layer.attn.wk.linear.freeze()
    ao_params = count_params(attn_only_a)
    print(f"  Trainable params (Attn-only LoRA): {ao_params:,}")
    train(attn_only_a, train_a, val_a, steps=FINETUNE_STEPS,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)
    attn_only_a.unfreeze()

    print("\n--- 7b. Attn-only LoRA: domain B ---")
    attn_only_b = get_model("attn_lora_gpt", vocab_size=V, **BASE,
                            mlp_rank=MLP_RANK, mlp_alpha=MLP_ALPHA,
                            attn_rank=ATTN_RANK, attn_alpha=ATTN_ALPHA)
    mx.eval(attn_only_b.parameters())
    copy_base_to_attn_lora(base_model, attn_only_b)
    attn_only_b.freeze()
    for layer in attn_only_b.layers:
        layer.attn.wq.unfreeze()
        layer.attn.wk.unfreeze()
        layer.attn.wq.linear.freeze()
        layer.attn.wk.linear.freeze()
    train(attn_only_b, train_b, val_b, steps=FINETUNE_STEPS,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)
    attn_only_b.unfreeze()

    ao_deltas_a = get_deltas(attn_only_a, include_attn=True)
    ao_deltas_b = get_deltas(attn_only_b, include_attn=True)

    baked_ao_a = apply_deltas_to_base(base_model, [d for d in ao_deltas_a if d[1] in ('wq', 'wk')], V)
    baked_ao_b = apply_deltas_to_base(base_model, [d for d in ao_deltas_b if d[1] in ('wq', 'wk')], V)
    ao_single_a = evaluate(baked_ao_a, val_a, BATCH_SIZE)
    ao_single_b = evaluate(baked_ao_b, val_b, BATCH_SIZE)
    results["attn_only_single"] = {"a_m": ao_single_a, "n_z": ao_single_b,
                                   "avg": (ao_single_a + ao_single_b) / 2}
    print(f"  Attn-only single: a_m={ao_single_a:.4f}, n_z={ao_single_b:.4f}")

    # === 8. Delta analysis ===
    print("\n--- 8. Delta norm analysis ---")
    mlp_norm_a = sum(mx.sum(d[2]**2).item() for d in mlp_deltas_a) ** 0.5
    mlp_norm_b = sum(mx.sum(d[2]**2).item() for d in mlp_deltas_b) ** 0.5
    attn_delta_a = [d for d in attn_deltas_a if d[1] in ('wq', 'wk')]
    attn_delta_b = [d for d in attn_deltas_b if d[1] in ('wq', 'wk')]
    attn_norm_a = sum(mx.sum(d[2]**2).item() for d in attn_delta_a) ** 0.5
    attn_norm_b = sum(mx.sum(d[2]**2).item() for d in attn_delta_b) ** 0.5
    mlp_delta_from_attn_a = [d for d in attn_deltas_a if d[1] in ('fc1', 'fc2')]
    mlp_delta_from_attn_b = [d for d in attn_deltas_b if d[1] in ('fc1', 'fc2')]
    mlp_of_attn_norm_a = sum(mx.sum(d[2]**2).item() for d in mlp_delta_from_attn_a) ** 0.5
    mlp_of_attn_norm_b = sum(mx.sum(d[2]**2).item() for d in mlp_delta_from_attn_b) ** 0.5

    print(f"  MLP-only delta norms: A={mlp_norm_a:.6f}, B={mlp_norm_b:.6f}")
    print(f"  MLP+Attn model - MLP part norms: A={mlp_of_attn_norm_a:.6f}, B={mlp_of_attn_norm_b:.6f}")
    print(f"  MLP+Attn model - Attn part norms: A={attn_norm_a:.6f}, B={attn_norm_b:.6f}")
    attn_frac_a = attn_norm_a / (attn_norm_a + mlp_of_attn_norm_a + 1e-8)
    attn_frac_b = attn_norm_b / (attn_norm_b + mlp_of_attn_norm_b + 1e-8)
    print(f"  Attention delta fraction: A={attn_frac_a:.1%}, B={attn_frac_b:.1%}")

    # === Summary ===
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Param counts:")
    print(f"    MLP-only LoRA trainable: {mlp_a_params:,}")
    print(f"    MLP+Attn LoRA trainable: {attn_a_params:,}")
    print(f"    Overhead: +{(attn_a_params - mlp_a_params):,} ({(attn_a_params/mlp_a_params - 1)*100:.1f}%)")

    joint_avg = results["joint"]["avg"]

    print(f"\n{'Method':<25} {'a_m':>8} {'n_z':>8} {'avg':>8} {'vs joint':>10}")
    print("-" * 60)
    for method, vals in results.items():
        delta = (vals["avg"] - joint_avg) / joint_avg * 100
        print(f"{method:<25} {vals['a_m']:>8.4f} {vals['n_z']:>8.4f} {vals['avg']:>8.4f} {delta:>+9.1f}%")

    # Composition gap comparison
    mlp_gap = (results["mlp_composed"]["avg"] - joint_avg) / joint_avg * 100
    attn_gap = (results["attn_composed"]["avg"] - joint_avg) / joint_avg * 100
    gap_improvement = mlp_gap - attn_gap  # positive = attention is better

    print(f"\n  Composition gaps (vs joint):")
    print(f"    MLP-only composed:   {mlp_gap:+.2f}%")
    print(f"    MLP+Attn composed:   {attn_gap:+.2f}%")
    print(f"    Gap improvement:     {gap_improvement:+.2f}pp")

    # Single-domain quality check
    single_degradation = ((results["attn_single"]["avg"] - results["mlp_single"]["avg"])
                          / results["mlp_single"]["avg"] * 100)
    print(f"\n  Single-domain quality:")
    print(f"    MLP-only single avg: {results['mlp_single']['avg']:.4f}")
    print(f"    MLP+Attn single avg: {results['attn_single']['avg']:.4f}")
    print(f"    Degradation:         {single_degradation:+.2f}%")

    return results, {
        'mlp_a_params': mlp_a_params,
        'attn_a_params': attn_a_params,
        'mlp_gap': mlp_gap,
        'attn_gap': attn_gap,
        'gap_improvement': gap_improvement,
        'single_degradation': single_degradation,
        'attn_frac_a': attn_frac_a,
        'attn_frac_b': attn_frac_b,
    }


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

    methods = list(all_results[seeds[0]].keys())
    joint_avgs = [all_results[s]["joint"]["avg"] for s in seeds]
    joint_mean = statistics.mean(joint_avgs)

    print(f"\n{'Method':<25} {'avg (mean)':>12} {'std':>8} {'vs joint':>10}")
    print("-" * 60)
    for method in methods:
        avgs = [all_results[s][method]["avg"] for s in seeds]
        mean_avg = statistics.mean(avgs)
        std_avg = statistics.stdev(avgs) if len(avgs) > 1 else 0.0
        delta = (mean_avg - joint_mean) / joint_mean * 100
        if method == "joint":
            print(f"{method:<25} {mean_avg:>12.4f} {std_avg:>8.4f} {'baseline':>10}")
        else:
            print(f"{method:<25} {mean_avg:>12.4f} {std_avg:>8.4f} {delta:>+9.1f}%")

    # Kill threshold checks
    print(f"\n{'='*70}")
    print("KILL THRESHOLD CHECKS")
    print(f"{'='*70}")

    # Gap improvement across seeds
    gap_improvements = [all_metrics[s]['gap_improvement'] for s in seeds]
    mean_gap_imp = statistics.mean(gap_improvements)
    print(f"\n  1. Composition gap improvement (MLP gap - MLP+Attn gap):")
    for s in seeds:
        print(f"     seed {s}: {all_metrics[s]['gap_improvement']:+.2f}pp "
              f"(MLP: {all_metrics[s]['mlp_gap']:+.2f}%, Attn: {all_metrics[s]['attn_gap']:+.2f}%)")
    print(f"     Mean: {mean_gap_imp:+.2f}pp")
    if abs(mean_gap_imp) < 1.0:
        print(f"     KILL: improvement {mean_gap_imp:.2f}pp < 1pp threshold")
    else:
        print(f"     PASS: improvement {mean_gap_imp:.2f}pp >= 1pp threshold")

    # Single-domain degradation
    single_degs = [all_metrics[s]['single_degradation'] for s in seeds]
    mean_single_deg = statistics.mean(single_degs)
    print(f"\n  2. Single-domain quality degradation:")
    for s in seeds:
        print(f"     seed {s}: {all_metrics[s]['single_degradation']:+.2f}%")
    print(f"     Mean: {mean_single_deg:+.2f}%")
    if mean_single_deg > 2.0:
        print(f"     KILL: degradation {mean_single_deg:.2f}% > 2% threshold")
    else:
        print(f"     PASS: degradation {mean_single_deg:.2f}% <= 2% threshold")

    # Parameter overhead
    print(f"\n  Parameter overhead:")
    print(f"    MLP-only: {all_metrics[seeds[0]]['mlp_a_params']:,}")
    print(f"    MLP+Attn: {all_metrics[seeds[0]]['attn_a_params']:,}")
    overhead = all_metrics[seeds[0]]['attn_a_params'] - all_metrics[seeds[0]]['mlp_a_params']
    overhead_pct = overhead / all_metrics[seeds[0]]['mlp_a_params'] * 100
    print(f"    Overhead: +{overhead:,} ({overhead_pct:.1f}%)")

    # Attention delta fraction
    attn_fracs = [(all_metrics[s]['attn_frac_a'] + all_metrics[s]['attn_frac_b']) / 2
                  for s in seeds]
    print(f"\n  Attention delta fraction (of total adapted norm): {statistics.mean(attn_fracs):.1%}")

    # Final verdict
    print(f"\n{'='*70}")
    kill_1 = abs(mean_gap_imp) < 1.0
    kill_2 = mean_single_deg > 2.0
    if kill_1 or kill_2:
        print(f"VERDICT: KILL")
        if kill_1:
            print(f"  - Gap improvement {mean_gap_imp:.2f}pp below 1pp threshold")
        if kill_2:
            print(f"  - Single-domain degradation {mean_single_deg:.2f}% exceeds 2% threshold")
    else:
        print(f"VERDICT: PASS")
        print(f"  - Gap improvement {mean_gap_imp:.2f}pp >= 1pp (composition bottleneck addressed)")
        print(f"  - Single-domain degradation {mean_single_deg:.2f}% <= 2% (no quality cost)")
    print(f"{'='*70}")

    return all_results, all_metrics


if __name__ == "__main__":
    run_multiseed()
