"""Run entropy-adaptive routing COMPOSITION experiment.

The key finding (Exp 2): k=1 is catastrophic for COMPOSED models (+200%),
while k=2 works within 1.6%. The hypothesis is that entropy-adaptive k
can safely use k=1 for confident tokens and k=2 for uncertain tokens,
reducing average compute while maintaining quality UNDER COMPOSITION.

Protocol:
1. Train shared base model
2. Train domain-specific capsule groups (a-m names, n-z names)
3. Compose by concatenating groups (8 total: 4 per domain)
4. Calibrate router on mixed data
5. Compare: fixed k=2 vs entropy-adaptive vs fixed k=1

Kill criteria:
  1. Variable-k routing worse than fixed k=2 at same average compute
  2. Entropy-based k-selection doesn't reduce average k below 1.8
"""

import sys
import os
import random
import time
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.data import (
    load_names, CharTokenizer, CharDataset,
    domain_split, train_val_split,
)
from experiments.models.gpt import RMSNorm, CausalSelfAttention
from experiments.models.capsule_moe.capsule_moe import CapsuleGroup


def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


class ComposableCapsulePool(nn.Module):
    """CapsulePool that supports composition by concatenation and
    switchable routing: fixed top-k or entropy-adaptive."""

    def __init__(self, n_embd, n_groups, n_capsules_per_group,
                 routing_mode="fixed_k2", tau_h=0.5, sparsity_coeff=0.0):
        super().__init__()
        self.n_groups = n_groups
        self.n_embd = n_embd
        self.n_capsules_per_group = n_capsules_per_group
        self.routing_mode = routing_mode
        self.sparsity_coeff = sparsity_coeff

        import math
        self.h_max = math.log(max(n_groups, 2))

        # Router
        self.router = nn.Linear(n_embd, n_groups, bias=False)

        # Groups
        self.groups = [CapsuleGroup(n_embd, n_capsules_per_group)
                       for _ in range(n_groups)]

        # Entropy threshold (for adaptive mode)
        init_val = 0.0  # sigmoid(0) = 0.5 -> tau = 0.5 * H_max
        self.raw_tau = mx.array([init_val])

        # Stats
        self._gate_probs = None
        self._avg_k = None
        self._alpha = None
        self._n_experts_skipped = 0
        self._n_experts_total = n_groups
        self.random_k1_prob = 0.15  # default, overridden per config

    def __call__(self, x):
        B, T, D = x.shape
        scores = self.router(x)  # (B, T, G)
        probs = mx.softmax(scores, axis=-1)
        self._gate_probs = probs

        if self.routing_mode == "fixed_k1":
            return self._route_fixed_k(x, scores, probs, k=1)
        elif self.routing_mode == "fixed_k2":
            return self._route_fixed_k(x, scores, probs, k=2)
        elif self.routing_mode == "entropy_adaptive":
            return self._route_entropy_adaptive(x, scores, probs)
        elif self.routing_mode == "hard_adaptive":
            return self._route_hard_adaptive(x, scores, probs)
        elif self.routing_mode == "random_k":
            return self._route_random_k(x, scores, probs)
        else:
            raise ValueError(f"Unknown routing_mode: {self.routing_mode}")

    def _route_fixed_k(self, x, scores, probs, k):
        k = min(k, self.n_groups)
        top_vals = mx.topk(scores, k, axis=-1)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)
        mask = (scores >= threshold).astype(mx.float32)
        masked_probs = probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

        out = mx.zeros_like(x)
        for i, group in enumerate(self.groups):
            w = masked_probs[..., i:i+1]
            out = out + w * group(x)

        self._avg_k = float(k)
        self._alpha = None
        return out

    def _route_entropy_adaptive(self, x, scores, probs):
        import math

        log_probs = mx.log(probs + 1e-10)
        entropies = -mx.sum(probs * log_probs, axis=-1)  # (B, T)

        tau = mx.sigmoid(self.raw_tau) * self.h_max
        temperature = 0.1
        alpha = mx.sigmoid((entropies - tau) / temperature)
        self._alpha = alpha

        # Hard k for logging
        hard_k = mx.where(entropies < tau, 1.0, 2.0)
        self._avg_k = mx.mean(hard_k).item()

        # Top-2 mask
        top2_vals = mx.topk(scores, min(2, self.n_groups), axis=-1)
        thresh_2 = mx.min(top2_vals, axis=-1, keepdims=True)
        mask_top2 = (scores >= thresh_2).astype(mx.float32)

        # Top-1 mask
        top1_vals = mx.topk(scores, 1, axis=-1)
        mask_top1 = (scores >= top1_vals).astype(mx.float32)

        alpha_exp = alpha[..., None]
        mask = mask_top1 * (1.0 - alpha_exp) + mask_top2 * alpha_exp

        masked_probs = probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

        # Conditional expert execution: skip groups where max weight < epsilon
        # This is the FIX for review point 1 -- actual compute savings
        eps = 1e-3
        out = mx.zeros_like(x)
        n_skipped = 0
        for i, group in enumerate(self.groups):
            w = masked_probs[..., i:i+1]  # (B, T, 1)
            max_w = mx.max(w).item()
            if max_w < eps:
                n_skipped += 1
                continue
            out = out + w * group(x)
        self._n_experts_skipped = n_skipped
        self._n_experts_total = self.n_groups
        return out

    def _route_hard_adaptive(self, x, scores, probs):
        """Hard threshold version for soft-to-hard gap measurement (Fix 5)."""
        log_probs = mx.log(probs + 1e-10)
        entropies = -mx.sum(probs * log_probs, axis=-1)  # (B, T)

        tau = mx.sigmoid(self.raw_tau) * self.h_max

        # Hard decision: k=1 if H < tau, k=2 otherwise (no sigmoid interpolation)
        is_confident = (entropies < tau).astype(mx.float32)  # (B, T)
        hard_k = mx.where(entropies < tau, 1.0, 2.0)
        self._avg_k = mx.mean(hard_k).item()
        self._alpha = 1.0 - is_confident  # for logging compatibility

        # Top-2 mask
        top2_vals = mx.topk(scores, min(2, self.n_groups), axis=-1)
        thresh_2 = mx.min(top2_vals, axis=-1, keepdims=True)
        mask_top2 = (scores >= thresh_2).astype(mx.float32)

        # Top-1 mask
        top1_vals = mx.topk(scores, 1, axis=-1)
        mask_top1 = (scores >= top1_vals).astype(mx.float32)

        # Hard interpolation: exactly k=1 or exactly k=2 per token
        is_confident_exp = is_confident[..., None]
        mask = mask_top1 * is_confident_exp + mask_top2 * (1.0 - is_confident_exp)

        masked_probs = probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

        eps = 1e-3
        out = mx.zeros_like(x)
        for i, group in enumerate(self.groups):
            w = masked_probs[..., i:i+1]
            max_w = mx.max(w).item()
            if max_w < eps:
                continue
            out = out + w * group(x)
        return out

    def _route_random_k(self, x, scores, probs):
        """Random k baseline (Fix 3): randomly assign k=1 with matched probability."""
        B, T, _ = x.shape

        # Use random mask: k=1 with probability self.random_k1_prob, else k=2
        # Use deterministic seed per forward for reproducibility
        random_vals = mx.random.uniform(shape=(B, T))
        is_k1 = (random_vals < self.random_k1_prob).astype(mx.float32)  # (B, T)

        hard_k = mx.where(is_k1 > 0.5, 1.0, 2.0)
        self._avg_k = mx.mean(hard_k).item()
        self._alpha = 1.0 - is_k1

        # Top-2 mask
        top2_vals = mx.topk(scores, min(2, self.n_groups), axis=-1)
        thresh_2 = mx.min(top2_vals, axis=-1, keepdims=True)
        mask_top2 = (scores >= thresh_2).astype(mx.float32)

        # Top-1 mask
        top1_vals = mx.topk(scores, 1, axis=-1)
        mask_top1 = (scores >= top1_vals).astype(mx.float32)

        is_k1_exp = is_k1[..., None]
        mask = mask_top1 * is_k1_exp + mask_top2 * (1.0 - is_k1_exp)

        masked_probs = probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

        out = mx.zeros_like(x)
        for i, group in enumerate(self.groups):
            w = masked_probs[..., i:i+1]
            out = out + w * group(x)
        return out

    def balance_loss(self):
        if self._gate_probs is None:
            return mx.array(0.0)
        mean_probs = mx.mean(self._gate_probs, axis=(0, 1))
        return self.n_groups * mx.sum(mean_probs * mean_probs)

    def sparsity_loss(self):
        if self._alpha is None or self.sparsity_coeff == 0.0:
            return mx.array(0.0)
        return self.sparsity_coeff * mx.mean(self._alpha)


class ComposableBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_groups, n_capsules_per_group,
                 routing_mode="fixed_k2", tau_h=0.5, sparsity_coeff=0.0):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.pool = ComposableCapsulePool(
            n_embd, n_groups, n_capsules_per_group,
            routing_mode=routing_mode, tau_h=tau_h,
            sparsity_coeff=sparsity_coeff,
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.pool(self.norm2(x))
        return x


class ComposableGPT(nn.Module):
    def __init__(self, vocab_size=28, block_size=32, n_embd=64, n_head=4,
                 n_layer=4, n_groups=4, n_capsules_per_group=64,
                 routing_mode="fixed_k2", tau_h=0.5, sparsity_coeff=0.0):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [ComposableBlock(
            n_embd, n_head, n_groups, n_capsules_per_group,
            routing_mode=routing_mode, tau_h=tau_h,
            sparsity_coeff=sparsity_coeff,
        ) for _ in range(n_layer)]
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
        total = mx.array(0.0)
        for layer in self.layers:
            total = total + layer.pool.balance_loss()
            total = total + layer.pool.sparsity_loss()
        return 0.01 * total

    def on_domain_switch(self, domain):
        pass

    def avg_k(self):
        ks = [layer.pool._avg_k for layer in self.layers if layer.pool._avg_k is not None]
        return sum(ks) / len(ks) if ks else 2.0


def compose_models(base_model, domain_models, routing_mode="fixed_k2",
                   sparsity_coeff=0.0, vocab_size=28, block_size=32,
                   n_embd=64, n_head=4, n_layer=4, n_capsules_per_group=64):
    """Compose domain-specific models by concatenating capsule groups.

    Each domain model has 4 groups. Composed model has 8 groups (4 per domain).
    """
    n_domains = len(domain_models)
    n_groups_total = 4 * n_domains

    composed = ComposableGPT(
        vocab_size=vocab_size, block_size=block_size,
        n_embd=n_embd, n_head=n_head, n_layer=n_layer,
        n_groups=n_groups_total,
        n_capsules_per_group=n_capsules_per_group,
        routing_mode=routing_mode,
        sparsity_coeff=sparsity_coeff,
    )

    # Copy shared weights from base
    composed.wte.weight = base_model.wte.weight
    composed.wpe.weight = base_model.wpe.weight
    composed.lm_head.weight = base_model.lm_head.weight

    for l_idx in range(n_layer):
        # Copy attention from base
        src_attn = base_model.layers[l_idx].attn
        dst_attn = composed.layers[l_idx].attn
        dst_attn.wq.weight = src_attn.wq.weight
        dst_attn.wk.weight = src_attn.wk.weight
        dst_attn.wv.weight = src_attn.wv.weight
        dst_attn.wo.weight = src_attn.wo.weight

        # Copy norms from base
        # (RMSNorm has no params in this impl)

        # Copy capsule groups from each domain model
        for d_idx, dm in enumerate(domain_models):
            for g_idx in range(4):
                src_group = dm.layers[l_idx].pool.groups[g_idx]
                dst_idx = d_idx * 4 + g_idx
                composed.layers[l_idx].pool.groups[dst_idx].A.weight = src_group.A.weight
                composed.layers[l_idx].pool.groups[dst_idx].B.weight = src_group.B.weight

    mx.eval(composed.parameters())
    return composed


def calibrate_router(model, train_ds, steps=200, batch_size=32, lr=3e-3, seed=0):
    """Calibrate router (and threshold) on mixed-domain data.

    Fix 2: Explicitly unfreeze raw_tau after model.freeze(). In MLX,
    freeze() marks all leaf arrays as non-trainable. raw_tau is stored
    as a bare mx.array on the module, so we must explicitly unfreeze it
    by calling unfreeze(keys=...) on the pool module.
    """
    rng = random.Random(seed)

    # Freeze everything first
    model.freeze()

    # Unfreeze only router weights and raw_tau threshold
    for layer in model.layers:
        layer.pool.router.unfreeze()
        # Fix 2: explicitly unfreeze raw_tau parameter
        # Use unfreeze with keys to target the specific parameter
        layer.pool.unfreeze(keys=["raw_tau"])

    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(m, inp, tgt):
        logits = m(inp)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B*T, V), tgt.reshape(B*T), reduction='mean'
        ) + m.aux_loss()

    lag = nn.value_and_grad(model, loss_fn)

    for step in range(1, steps + 1):
        inp, tgt = train_ds.get_batch(batch_size, rng)
        loss, grads = lag(model, inp, tgt)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 50 == 0:
            avg_k = model.avg_k()
            print(f"    cal step {step}/{steps}: loss={loss.item():.4f}, avg_k={avg_k:.3f}")

    # Unfreeze everything for eval
    model.unfreeze()
    return model


def evaluate(model, val_ds, batch_size=32, n_batches=20):
    eval_rng = random.Random(0)
    total = 0.0
    for _ in range(n_batches):
        inp, tgt = val_ds.get_batch(batch_size, eval_rng)
        logits = model(inp)
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B*T, V), tgt.reshape(B*T), reduction='mean'
        )
        mx.eval(loss)
        total += loss.item()
    return total / n_batches


def train_model(model, train_ds, val_ds, steps=500, batch_size=32, lr=3e-3, seed=42):
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(m, inp, tgt):
        logits = m(inp)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B*T, V), tgt.reshape(B*T), reduction='mean'
        ) + m.aux_loss()

    lag = nn.value_and_grad(model, loss_fn)
    for step in range(1, steps + 1):
        inp, tgt = train_ds.get_batch(batch_size, rng)
        loss, grads = lag(model, inp, tgt)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
    return model


def get_per_layer_stats(model):
    """Extract per-layer entropy stats from the model after a forward pass."""
    stats = []
    for l_idx, layer in enumerate(model.layers):
        pool = layer.pool
        if pool._gate_probs is None:
            stats.append(None)
            continue
        s = {}
        if hasattr(pool, '_alpha') and pool._alpha is not None:
            if isinstance(pool._alpha, mx.array):
                ent_probs = pool._gate_probs
                log_p = mx.log(ent_probs + 1e-10)
                ent = -mx.sum(ent_probs * log_p, axis=-1)
                mx.eval(ent)
                s["mean_entropy"] = mx.mean(ent).item()
                s["std_entropy"] = mx.std(ent).item() if ent.size > 1 else 0.0
        tau = mx.sigmoid(pool.raw_tau) * pool.h_max
        mx.eval(tau)
        s["tau"] = tau.item()
        s["avg_k"] = pool._avg_k if pool._avg_k is not None else 2.0
        # frac_k1 from hard_k
        if hasattr(pool, '_alpha') and pool._alpha is not None and isinstance(pool._alpha, mx.array):
            frac_k1 = mx.mean((ent < tau).astype(mx.float32)).item()
            s["frac_k1"] = frac_k1
        else:
            s["frac_k1"] = 0.0
        n_skip = getattr(pool, '_n_experts_skipped', 0)
        n_total = getattr(pool, '_n_experts_total', pool.n_groups)
        s["experts_skipped"] = n_skip
        s["experts_total"] = n_total
        stats.append(s)
    return stats


def evaluate_with_wallclock(model, val_ds, batch_size=32, n_batches=20):
    """Evaluate and return loss + wall-clock time."""
    eval_rng = random.Random(0)
    total = 0.0
    t0 = time.time()
    for _ in range(n_batches):
        inp, tgt = val_ds.get_batch(batch_size, eval_rng)
        logits = model(inp)
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B*T, V), tgt.reshape(B*T), reduction='mean'
        )
        mx.eval(loss)
        total += loss.item()
    elapsed = time.time() - t0
    return total / n_batches, elapsed


def run_composition_experiment(seeds=(42, 123, 7),
                                base_steps=300,
                                domain_steps=300,
                                cal_steps=200):
    """Full composition experiment with all 5 review fixes."""
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    block_size = 32
    n_embd = 64
    n_head = 4
    n_layer = 4

    splits = domain_split(docs)
    domain_names = list(splits.keys())
    print(f"Domains: {domain_names}")

    configs = {
        "fixed_k1": {"routing_mode": "fixed_k1", "sparsity_coeff": 0.0},
        "fixed_k2": {"routing_mode": "fixed_k2", "sparsity_coeff": 0.0},
        "ea_sc0.0": {"routing_mode": "entropy_adaptive", "sparsity_coeff": 0.0},
        "ea_sc0.1": {"routing_mode": "entropy_adaptive", "sparsity_coeff": 0.1},
        "ea_sc0.3": {"routing_mode": "entropy_adaptive", "sparsity_coeff": 0.3},
    }

    all_results = {k: [] for k in configs}
    # Fix 3: random-k baselines (populated after first EA run per seed)
    random_k_results = {"rk_for_ea_sc0.0": [], "rk_for_ea_sc0.1": [], "rk_for_ea_sc0.3": []}
    # Fix 5: hard-adaptive results
    hard_results = {"hard_ea_sc0.0": [], "hard_ea_sc0.1": [], "hard_ea_sc0.3": []}
    # Fix 4: per-layer stats across seeds
    per_layer_stats_all = {"ea_sc0.0": [], "ea_sc0.1": [], "ea_sc0.3": []}
    # Wall-clock timing
    wallclock_results = {}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        # Prepare data
        all_train, all_val = train_val_split(docs, seed=seed)
        all_train_ds = CharDataset(all_train, tokenizer, block_size)
        all_val_ds = CharDataset(all_val, tokenizer, block_size)

        domain_datasets = {}
        for d_name, d_docs in splits.items():
            d_train, d_val = train_val_split(d_docs, seed=seed)
            domain_datasets[d_name] = (
                CharDataset(d_train, tokenizer, block_size),
                CharDataset(d_val, tokenizer, block_size),
            )

        # Step 1: Train base model
        print("\n--- Training base model ---")
        base = ComposableGPT(
            vocab_size=tokenizer.vocab_size, block_size=block_size,
            n_embd=n_embd, n_head=n_head, n_layer=n_layer,
            n_groups=4, n_capsules_per_group=64, routing_mode="fixed_k2",
        )
        mx.eval(base.parameters())
        base = train_model(base, all_train_ds, all_val_ds,
                           steps=base_steps, seed=seed)
        base_val = evaluate(base, all_val_ds)
        print(f"  Base val_loss = {base_val:.4f}")

        # Step 2: Train domain-specific models (fine-tune from base)
        domain_models = []
        for d_name in domain_names:
            print(f"\n--- Training domain: {d_name} ---")
            dm = ComposableGPT(
                vocab_size=tokenizer.vocab_size, block_size=block_size,
                n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                n_groups=4, n_capsules_per_group=64, routing_mode="fixed_k2",
            )
            dm.load_weights(list(zip(
                [k for k, _ in nn.utils.tree_flatten(base.parameters())],
                [v for _, v in nn.utils.tree_flatten(base.parameters())]
            )))
            mx.eval(dm.parameters())

            dm.wte.freeze()
            dm.wpe.freeze()
            dm.lm_head.freeze()
            for layer in dm.layers:
                layer.attn.freeze()

            d_train_ds, d_val_ds = domain_datasets[d_name]
            dm = train_model(dm, d_train_ds, d_val_ds,
                            steps=domain_steps, seed=seed)
            dm.unfreeze()
            d_val = evaluate(dm, d_val_ds)
            print(f"  {d_name} val_loss = {d_val:.4f}")
            domain_models.append(dm)

        # Joint reference
        print("\n--- Training joint model (reference) ---")
        joint = ComposableGPT(
            vocab_size=tokenizer.vocab_size, block_size=block_size,
            n_embd=n_embd, n_head=n_head, n_layer=n_layer,
            n_groups=8, n_capsules_per_group=64, routing_mode="fixed_k2",
        )
        mx.eval(joint.parameters())
        joint = train_model(joint, all_train_ds, all_val_ds,
                           steps=base_steps + domain_steps, seed=seed)
        joint_val = evaluate(joint, all_val_ds)
        print(f"  Joint val_loss = {joint_val:.4f}")

        # Track observed k1 fractions for random-k baselines
        ea_k1_fracs = {}

        for cfg_name, cfg in configs.items():
            print(f"\n--- Compose + calibrate: {cfg_name} ---")
            composed = compose_models(
                base, domain_models,
                routing_mode=cfg["routing_mode"],
                sparsity_coeff=cfg["sparsity_coeff"],
                vocab_size=tokenizer.vocab_size,
                block_size=block_size,
                n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                n_capsules_per_group=64,
            )

            pre_cal_val = evaluate(composed, all_val_ds)
            print(f"  Pre-cal val_loss = {pre_cal_val:.4f}")

            composed = calibrate_router(composed, all_train_ds,
                                       steps=cal_steps, seed=seed)

            # Evaluate with wall-clock timing (Fix 1)
            post_cal_val, eval_time = evaluate_with_wallclock(composed, all_val_ds)
            avg_k = composed.avg_k()
            print(f"  Post-cal val_loss = {post_cal_val:.4f}, avg_k = {avg_k:.3f}, eval_time = {eval_time:.3f}s")

            result_entry = {
                "val_loss": post_cal_val,
                "avg_k": avg_k,
                "pre_cal_val": pre_cal_val,
                "joint_val": joint_val,
                "eval_time": eval_time,
            }

            # Fix 4: collect per-layer stats for EA configs
            if cfg_name.startswith("ea_"):
                # Run one forward pass to populate stats
                eval_rng = random.Random(0)
                inp, tgt = all_val_ds.get_batch(32, eval_rng)
                _ = composed(inp)
                mx.eval(composed.parameters())
                layer_stats = get_per_layer_stats(composed)
                per_layer_stats_all[cfg_name].append(layer_stats)
                result_entry["per_layer"] = layer_stats

                # Track k1 fraction for random-k baseline
                frac_k1 = 1.0 - (avg_k - 1.0)  # k1_frac = 2 - avg_k
                ea_k1_fracs[cfg_name] = frac_k1

                # Print per-layer stats
                print(f"  Per-layer stats:")
                for l_idx, ls in enumerate(layer_stats):
                    if ls:
                        print(f"    L{l_idx}: H={ls.get('mean_entropy', 'N/A'):.4f}, "
                              f"tau={ls.get('tau', 'N/A'):.4f}, "
                              f"avg_k={ls.get('avg_k', 'N/A'):.3f}, "
                              f"frac_k1={ls.get('frac_k1', 0):.1%}, "
                              f"skip={ls.get('experts_skipped', 0)}/{ls.get('experts_total', 8)}")

                # Fix 5: Evaluate with hard thresholding
                print(f"  --- Hard threshold eval for {cfg_name} ---")
                for layer in composed.layers:
                    layer.pool.routing_mode = "hard_adaptive"
                hard_val, hard_time = evaluate_with_wallclock(composed, all_val_ds)
                hard_avg_k = composed.avg_k()
                print(f"  Hard val_loss = {hard_val:.4f}, avg_k = {hard_avg_k:.3f}, time = {hard_time:.3f}s")
                hard_name = f"hard_{cfg_name}"
                hard_results[hard_name].append({
                    "val_loss": hard_val,
                    "avg_k": hard_avg_k,
                    "eval_time": hard_time,
                    "soft_val_loss": post_cal_val,
                    "gap_pct": (hard_val - post_cal_val) / post_cal_val * 100,
                })
                # Restore soft mode
                for layer in composed.layers:
                    layer.pool.routing_mode = "entropy_adaptive"

            all_results[cfg_name].append(result_entry)
            wallclock_results.setdefault(cfg_name, []).append(eval_time)

        # Fix 3: Random-k baselines (after all EA configs run for this seed)
        for ea_name, k1_frac in ea_k1_fracs.items():
            rk_name = f"rk_for_{ea_name}"
            print(f"\n--- Random-k baseline for {ea_name} (p_k1={k1_frac:.3f}) ---")
            composed_rk = compose_models(
                base, domain_models,
                routing_mode="random_k",
                sparsity_coeff=0.0,
                vocab_size=tokenizer.vocab_size,
                block_size=block_size,
                n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                n_capsules_per_group=64,
            )
            # Set random k1 probability to match EA
            for layer in composed_rk.layers:
                layer.pool.random_k1_prob = k1_frac

            # Calibrate router (same as EA but with random routing)
            composed_rk = calibrate_router(composed_rk, all_train_ds,
                                          steps=cal_steps, seed=seed)
            rk_val = evaluate(composed_rk, all_val_ds)
            rk_avg_k = composed_rk.avg_k()
            print(f"  Random-k val_loss = {rk_val:.4f}, avg_k = {rk_avg_k:.3f}")
            random_k_results[rk_name].append({
                "val_loss": rk_val,
                "avg_k": rk_avg_k,
                "matched_ea_name": ea_name,
            })

    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS (3-seed mean)")
    print(f"{'='*60}")

    k2_mean = sum(r["val_loss"] for r in all_results["fixed_k2"]) / len(seeds)
    joint_mean = sum(r["joint_val"] for r in all_results["fixed_k2"]) / len(seeds)

    print(f"\n  Joint reference: {joint_mean:.4f}")
    print(f"\n  {'Config':<15} {'Val Loss':>10} {'Avg K':>8} {'vs k=2':>10} {'vs Joint':>10} {'Time(s)':>8}")
    print(f"  {'-'*66}")

    for name, runs in all_results.items():
        mean_vl = sum(r["val_loss"] for r in runs) / len(runs)
        mean_k = sum(r["avg_k"] for r in runs) / len(runs)
        mean_t = sum(r["eval_time"] for r in runs) / len(runs)
        gap_k2 = (mean_vl - k2_mean) / k2_mean * 100
        gap_joint = (mean_vl - joint_mean) / joint_mean * 100
        print(f"  {name:<15} {mean_vl:>10.4f} {mean_k:>8.3f} {gap_k2:>+9.2f}% {gap_joint:>+9.2f}% {mean_t:>7.3f}")

    # Fix 3: Random-k baseline results
    print(f"\n  --- Random-k Baselines ---")
    print(f"  {'Config':<20} {'Val Loss':>10} {'Avg K':>8} {'vs matched EA':>15}")
    print(f"  {'-'*58}")
    for rk_name, runs in random_k_results.items():
        if not runs:
            continue
        mean_vl = sum(r["val_loss"] for r in runs) / len(runs)
        mean_k = sum(r["avg_k"] for r in runs) / len(runs)
        ea_name = runs[0]["matched_ea_name"]
        ea_mean = sum(r["val_loss"] for r in all_results[ea_name]) / len(all_results[ea_name])
        gap = (mean_vl - ea_mean) / ea_mean * 100
        print(f"  {rk_name:<20} {mean_vl:>10.4f} {mean_k:>8.3f} {gap:>+14.2f}%")

    # Fix 5: Soft-to-hard gap
    print(f"\n  --- Soft-to-Hard Quality Gap ---")
    print(f"  {'Config':<15} {'Soft Loss':>10} {'Hard Loss':>10} {'Gap':>8}")
    print(f"  {'-'*48}")
    for hard_name, runs in hard_results.items():
        if not runs:
            continue
        mean_soft = sum(r["soft_val_loss"] for r in runs) / len(runs)
        mean_hard = sum(r["val_loss"] for r in runs) / len(runs)
        gap = (mean_hard - mean_soft) / mean_soft * 100
        print(f"  {hard_name:<15} {mean_soft:>10.4f} {mean_hard:>10.4f} {gap:>+7.2f}%")

    # Fix 4: Per-layer entropy profile (3-seed mean with std)
    print(f"\n  --- Per-Layer Entropy Profile (3-seed mean +/- std) ---")
    for cfg_name, seeds_stats in per_layer_stats_all.items():
        if not seeds_stats:
            continue
        print(f"\n  {cfg_name}:")
        print(f"  {'Layer':>5} {'Mean H':>10} {'Std H':>10} {'tau':>10} {'Avg k':>10} {'Frac k=1':>10}")
        print(f"  {'-'*60}")
        n_layers = len(seeds_stats[0]) if seeds_stats else 0
        for l_idx in range(n_layers):
            hs = [ss[l_idx].get("mean_entropy", 0) for ss in seeds_stats if ss[l_idx]]
            taus = [ss[l_idx].get("tau", 0) for ss in seeds_stats if ss[l_idx]]
            ks = [ss[l_idx].get("avg_k", 2.0) for ss in seeds_stats if ss[l_idx]]
            fk1s = [ss[l_idx].get("frac_k1", 0) for ss in seeds_stats if ss[l_idx]]

            import statistics
            mean_h = statistics.mean(hs) if hs else 0
            std_h = statistics.stdev(hs) if len(hs) > 1 else 0
            mean_tau = statistics.mean(taus) if taus else 0
            std_tau = statistics.stdev(taus) if len(taus) > 1 else 0
            mean_k = statistics.mean(ks) if ks else 2.0
            mean_fk1 = statistics.mean(fk1s) if fk1s else 0
            std_fk1 = statistics.stdev(fk1s) if len(fk1s) > 1 else 0

            print(f"  {l_idx:>5} {mean_h:>7.4f}+/-{std_h:.3f} {mean_tau:>7.4f}+/-{std_tau:.3f} "
                  f"{mean_k:>7.3f} {mean_fk1:>7.1%}+/-{std_fk1:.1%}")

    # Wall-clock comparison (Fix 1)
    print(f"\n  --- Wall-Clock Timing ---")
    k2_time = sum(wallclock_results.get("fixed_k2", [0])) / max(len(wallclock_results.get("fixed_k2", [1])), 1)
    for name in ["ea_sc0.0", "ea_sc0.1", "ea_sc0.3"]:
        if name in wallclock_results:
            ea_time = sum(wallclock_results[name]) / len(wallclock_results[name])
            savings = (k2_time - ea_time) / k2_time * 100 if k2_time > 0 else 0
            print(f"  {name}: {ea_time:.3f}s vs k=2 {k2_time:.3f}s ({savings:+.1f}% wall-clock)")

    # Kill criteria
    print(f"\n{'='*60}")
    print("KILL CRITERIA EVALUATION")
    print(f"{'='*60}")

    for name in ["ea_sc0.0", "ea_sc0.1", "ea_sc0.3"]:
        runs = all_results[name]
        mean_vl = sum(r["val_loss"] for r in runs) / len(runs)
        mean_k = sum(r["avg_k"] for r in runs) / len(runs)
        gap = (mean_vl - k2_mean) / k2_mean * 100

        print(f"\n  {name}:")
        print(f"    KC1 (quality): {gap:+.2f}% vs k=2 -> {'PASS' if gap <= 0 else 'MARGINAL' if gap < 2 else 'FAIL'}")
        print(f"    KC2 (avg_k < 1.8): avg_k={mean_k:.3f} -> {'PASS' if mean_k < 1.8 else 'FAIL'}")

        # Check random-k baseline
        rk_name = f"rk_for_{name}"
        if random_k_results.get(rk_name):
            rk_mean = sum(r["val_loss"] for r in random_k_results[rk_name]) / len(random_k_results[rk_name])
            rk_gap = (rk_mean - mean_vl) / mean_vl * 100
            print(f"    Random-k baseline: {rk_gap:+.2f}% vs entropy-adaptive -> "
                  f"{'ENTROPY ADDS VALUE' if rk_gap > 0.5 else 'NO VALUE OVER RANDOM'}")

        # Check soft-to-hard gap
        hard_name = f"hard_{name}"
        if hard_results.get(hard_name):
            hard_mean = sum(r["val_loss"] for r in hard_results[hard_name]) / len(hard_results[hard_name])
            hard_gap = (hard_mean - mean_vl) / mean_vl * 100
            print(f"    Soft-to-hard gap: {hard_gap:+.2f}% -> "
                  f"{'DEPLOYABLE' if hard_gap < 1.0 else 'NEEDS RETRAINING'}")

    return all_results, random_k_results, hard_results, per_layer_stats_all


if __name__ == "__main__":
    results = run_composition_experiment()
