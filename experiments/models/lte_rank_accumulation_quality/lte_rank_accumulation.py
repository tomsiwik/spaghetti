"""
LTE Rank Accumulation Quality at d=256: Does parallel multi-head LoRA merging
show a quality advantage over sequential merging when the space is large enough
for rank effects to matter?

Parent: lte_parallel_base (d=64, proven equivalent)
This experiment: d=256, r=8, K=4

Key insight from parent:
  At d=64, r/d = 12.5% -- a single rank-8 update covers enough of the space
  that parallel's rank advantage (K*r=32 per interval) is irrelevant.
  At d=256, r/d = 3.1% -- parallel accumulates rank 32 (12.5%) per interval
  vs rank 8 (3.1%) for sequential. This 4x capacity ratio should manifest
  if the parallel advantage is real.

Kill criteria:
  K1: quality difference <1% at d=256 (same as d=64) -> no scaling advantage
  K2: parallel base quality >20% worse than conventional (inherited from parent)

Design: identical to parent but d=256, with steps scaled to keep runtime <30 min.
"""

import math
import time
import random
import json
import os
from dataclasses import dataclass, asdict, field

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ..gpt import GPT
from ..lora_procrustes.lora_procrustes import LoRALinear, LoRAGPT


# ── Loss Functions ────────────────────────────────────────────────────────────


def _ntp_loss(model, inputs, targets):
    logits = model(inputs)
    B, T, V = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
    )


def _ntp_loss_gpt(model, inputs, targets):
    logits = model(inputs)
    B, T, V = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
    )


# ── Conventional Training ────────────────────────────────────────────────────


def train_conventional(
    model: GPT, dataset, total_steps: int = 500,
    batch_size: int = 32, lr: float = 3e-3, seed: int = 42,
    log_every: int = 100,
) -> dict:
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, _ntp_loss_gpt)
    losses, t0, total_tokens = [], time.time(), 0

    for step in range(1, total_steps + 1):
        inputs, targets = dataset.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss.item())
        total_tokens += inputs.size
        if step % log_every == 0 or step == total_steps:
            el = time.time() - t0
            print(f"  [Conv]     step {step:4d}/{total_steps} | loss {losses[-1]:.4f} "
                  f"| {total_tokens/el:.0f} tok/s")

    return {"final_loss": losses[-1], "losses": losses,
            "elapsed_s": time.time() - t0}


# ── PARALLEL LoRA Merge (LTE-style, reset after merge) ───────────────────────


def train_parallel_lora(
    model: LoRAGPT, dataset, total_steps: int = 400,
    merge_every: int = 25, n_heads: int = 4,
    batch_size: int = 32, lr: float = 3e-3, seed: int = 42,
    log_every: int = 100,
) -> dict:
    """K parallel LoRA branches, each on different data shard, merge by averaging.

    Compute-fair: n_intervals = total_steps / (n_heads * merge_every).
    Each head does merge_every steps per interval.
    Total gradient steps = n_heads * merge_every * n_intervals = total_steps.
    """
    t0 = time.time()
    total_tokens = 0
    merges_done = 0
    losses = []

    lora_layers = []
    for layer in model.layers:
        for fc_name in ['fc1', 'fc2']:
            lora_layers.append(getattr(layer.mlp, fc_name))

    head_rngs = [random.Random(seed + k * 7919) for k in range(n_heads)]
    n_intervals = max(1, total_steps // (n_heads * merge_every))

    for interval in range(n_intervals):
        # Each head starts with FRESH LoRA (B=0) on the current base
        head_params = []
        for k in range(n_heads):
            params = {}
            for m_idx, fc in enumerate(lora_layers):
                in_dim = fc.A.shape[0]
                rank = fc.A.shape[1]
                out_dim = fc.B.shape[1]
                scale = (2.0 / in_dim) ** 0.5
                mx.random.seed(seed + interval * 1000 + k * 100 + m_idx)
                A = mx.random.normal((in_dim, rank)) * scale
                B = mx.zeros((rank, out_dim))
                params[m_idx] = {"A": A, "B": B}
            head_params.append(params)

        for k in range(n_heads):
            # Load head k's LoRA
            for m_idx, fc in enumerate(lora_layers):
                fc.A = head_params[k][m_idx]["A"]
                fc.B = head_params[k][m_idx]["B"]
            mx.eval(model.parameters())

            model.freeze()
            for layer in model.layers:
                layer.mlp.fc1.unfreeze(keys=["A", "B"])
                layer.mlp.fc2.unfreeze(keys=["A", "B"])

            optimizer = optim.Adam(learning_rate=lr)
            loss_and_grad = nn.value_and_grad(model, _ntp_loss)

            for step in range(merge_every):
                inputs, targets = dataset.get_batch(batch_size, head_rngs[k])
                loss, grads = loss_and_grad(model, inputs, targets)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                losses.append(loss.item())
                total_tokens += inputs.size

            # Save trained LoRA
            for m_idx, fc in enumerate(lora_layers):
                head_params[k][m_idx]["A"] = fc.A
                head_params[k][m_idx]["B"] = fc.B

        # Merge: average K deltas into base, then reset LoRA
        for m_idx, fc in enumerate(lora_layers):
            avg_delta = mx.zeros_like(fc.linear.weight)
            scaling = fc.alpha / fc.rank  # Must match forward pass scaling
            for k in range(n_heads):
                A = head_params[k][m_idx]["A"]
                B = head_params[k][m_idx]["B"]
                delta = scaling * (A @ B).T  # (out, in), with alpha/r scaling
                avg_delta = avg_delta + delta
            avg_delta = avg_delta / n_heads
            fc.linear.weight = fc.linear.weight + avg_delta
            # Reset LoRA to zero
            fc.A = mx.zeros_like(fc.A)
            fc.B = mx.zeros_like(fc.B)

        mx.eval(model.parameters())
        merges_done += 1

        steps_done = (interval + 1) * n_heads * merge_every
        if steps_done % log_every == 0 or interval == n_intervals - 1:
            el = time.time() - t0
            print(f"  [Parallel] step {steps_done:4d}/{total_steps} | "
                  f"loss {losses[-1]:.4f} | {total_tokens/el:.0f} tok/s | "
                  f"merges: {merges_done} | heads: {n_heads}")

    return {"final_loss": losses[-1] if losses else float('inf'),
            "losses": losses, "merges_done": merges_done,
            "elapsed_s": time.time() - t0, "n_heads": n_heads,
            "n_intervals": n_intervals,
            "total_grad_steps": n_intervals * n_heads * merge_every}


# ── SEQUENTIAL LoRA Merge (ReLoRA-style) ──────────────────────────────────────


def train_sequential_lora(
    model: LoRAGPT, dataset, total_steps: int = 400,
    merge_every: int = 100, batch_size: int = 32, lr: float = 3e-3,
    seed: int = 42, log_every: int = 100,
) -> dict:
    """Sequential LoRA merge-and-restart (ReLoRA). One branch at a time."""
    rng = random.Random(seed)
    t0 = time.time()
    total_tokens = 0
    merges_done = 0
    losses = []

    model.freeze()
    for layer in model.layers:
        layer.mlp.fc1.unfreeze(keys=["A", "B"])
        layer.mlp.fc2.unfreeze(keys=["A", "B"])

    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, _ntp_loss)

    for step in range(1, total_steps + 1):
        inputs, targets = dataset.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss.item())
        total_tokens += inputs.size

        if step % log_every == 0 or step == total_steps:
            el = time.time() - t0
            print(f"  [Seq]      step {step:4d}/{total_steps} | loss {losses[-1]:.4f} "
                  f"| {total_tokens/el:.0f} tok/s | merges: {merges_done}")

        if step % merge_every == 0 and step < total_steps:
            # Merge and reset
            for layer in model.layers:
                for fc_name in ['fc1', 'fc2']:
                    fc = getattr(layer.mlp, fc_name)
                    delta = fc.get_delta()
                    fc.linear.weight = fc.linear.weight + delta.T
                    in_dim = fc.A.shape[0]
                    rank = fc.A.shape[1]
                    out_dim = fc.B.shape[1]
                    scale = (2.0 / in_dim) ** 0.5
                    fc.A = mx.random.normal((in_dim, rank)) * scale
                    fc.B = mx.zeros((rank, out_dim))
            mx.eval(model.parameters())
            merges_done += 1
            optimizer = optim.Adam(learning_rate=lr)
            loss_and_grad = nn.value_and_grad(model, _ntp_loss)
            model.freeze()
            for layer in model.layers:
                layer.mlp.fc1.unfreeze(keys=["A", "B"])
                layer.mlp.fc2.unfreeze(keys=["A", "B"])

    # Final merge
    for layer in model.layers:
        for fc_name in ['fc1', 'fc2']:
            fc = getattr(layer.mlp, fc_name)
            delta = fc.get_delta()
            fc.linear.weight = fc.linear.weight + delta.T
            fc.A = mx.zeros_like(fc.A)
            fc.B = mx.zeros_like(fc.B)
    mx.eval(model.parameters())
    merges_done += 1

    return {"final_loss": losses[-1], "losses": losses,
            "merges_done": merges_done, "elapsed_s": time.time() - t0,
            "total_grad_steps": total_steps}


# ── Expert Training & Measurement ────────────────────────────────────────────


def _copy_lora_base(model: LoRAGPT, rank: int = 8, alpha: float = 1.0) -> LoRAGPT:
    n_embd = model.wte.weight.shape[1]
    n_head = model.layers[0].attn.n_head
    n_layer = len(model.layers)
    vocab_size = model.wte.weight.shape[0]
    block_size = model.wpe.weight.shape[0]

    new = LoRAGPT(vocab_size=vocab_size, block_size=block_size,
                  n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                  lora_rank=rank, lora_alpha=alpha)
    new.wte.weight = model.wte.weight
    new.wpe.weight = model.wpe.weight
    for i in range(n_layer):
        new.layers[i].attn.wq.weight = model.layers[i].attn.wq.weight
        new.layers[i].attn.wk.weight = model.layers[i].attn.wk.weight
        new.layers[i].attn.wv.weight = model.layers[i].attn.wv.weight
        new.layers[i].attn.wo.weight = model.layers[i].attn.wo.weight
        new.layers[i].mlp.fc1.linear.weight = model.layers[i].mlp.fc1.linear.weight
        new.layers[i].mlp.fc2.linear.weight = model.layers[i].mlp.fc2.linear.weight
    new.lm_head.weight = model.lm_head.weight
    mx.eval(new.parameters())
    return new


def _gpt_to_lora(model: GPT, rank: int = 8, alpha: float = 1.0) -> LoRAGPT:
    n_embd = model.wte.weight.shape[1]
    n_head = model.layers[0].attn.n_head
    n_layer = len(model.layers)
    vocab_size = model.wte.weight.shape[0]
    block_size = model.wpe.weight.shape[0]

    lm = LoRAGPT(vocab_size=vocab_size, block_size=block_size,
                 n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                 lora_rank=rank, lora_alpha=alpha)
    lm.wte.weight = model.wte.weight
    lm.wpe.weight = model.wpe.weight
    for i, layer in enumerate(model.layers):
        lm.layers[i].attn.wq.weight = layer.attn.wq.weight
        lm.layers[i].attn.wk.weight = layer.attn.wk.weight
        lm.layers[i].attn.wv.weight = layer.attn.wv.weight
        lm.layers[i].attn.wo.weight = layer.attn.wo.weight
        lm.layers[i].mlp.fc1.linear.weight = layer.mlp.fc1.weight
        lm.layers[i].mlp.fc2.linear.weight = layer.mlp.fc2.weight
    lm.lm_head.weight = model.lm_head.weight
    mx.eval(lm.parameters())
    return lm


def _evaluate(model, dataset, batch_size=32, n_batches=10):
    rng = random.Random(999)
    total = 0.0
    for _ in range(n_batches):
        inp, tgt = dataset.get_batch(batch_size, rng)
        logits = model(inp)
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(logits.reshape(B*T, V), tgt.reshape(B*T),
                                       reduction="mean")
        mx.eval(loss)
        total += loss.item()
    return total / n_batches


def train_lora_expert(base_model, train_ds, val_ds, rank=8, alpha=1.0,
                      steps=200, batch_size=32, lr=3e-3, seed=42):
    if isinstance(base_model, LoRAGPT):
        lm = _copy_lora_base(base_model, rank, alpha)
    else:
        lm = _gpt_to_lora(base_model, rank, alpha)

    lm.freeze()
    for layer in lm.layers:
        layer.mlp.fc1.unfreeze(keys=["A", "B"])
        layer.mlp.fc2.unfreeze(keys=["A", "B"])
    mx.eval(lm.parameters())

    rng = random.Random(seed)
    opt = optim.Adam(learning_rate=lr)
    lg = nn.value_and_grad(lm, _ntp_loss)

    for _ in range(steps):
        inp, tgt = train_ds.get_batch(batch_size, rng)
        loss, grads = lg(lm, inp, tgt)
        opt.update(lm, grads)
        mx.eval(lm.parameters(), opt.state)

    val = _evaluate(lm, val_ds, batch_size)
    deltas = lm.get_all_deltas()
    return deltas, val


def compute_pairwise_cosine(deltas_list):
    flat = []
    for deltas in deltas_list:
        parts = [d.reshape(-1) for (_, _, d) in deltas]
        flat.append(mx.concatenate(parts))
    results = []
    n = len(flat)
    for i in range(n):
        for j in range(i + 1, n):
            cos = (flat[i] @ flat[j]) / (
                mx.sqrt(flat[i] @ flat[i]) * mx.sqrt(flat[j] @ flat[j]) + 1e-12)
            mx.eval(cos)
            results.append((i, j, cos.item()))
    return results


def compute_effective_rank(model):
    """Compute effective rank of weight matrices via Shannon entropy of singular values."""
    ranks = []
    for layer in model.layers:
        if hasattr(layer.mlp, 'fc1'):
            w = (layer.mlp.fc1.linear.weight if hasattr(layer.mlp.fc1, 'linear')
                 else layer.mlp.fc1.weight)
            _, S, _ = mx.linalg.svd(w, stream=mx.cpu)
            mx.eval(S)
            s = S.tolist()
            t = sum(s)
            if t > 1e-12:
                p = [x/t for x in s]
                h = -sum(x * math.log(x + 1e-12) for x in p if x > 1e-12)
                ranks.append(math.exp(h))
    return sum(ranks)/len(ranks) if ranks else 0


def compute_rank_per_interval_advantage(n_embd, lora_rank, n_heads):
    """Compute the theoretical rank advantage of parallel over sequential.

    Returns ratio of rank capacity: (K*r) / r = K for parallel,
    and the coverage ratio r/d for each.
    """
    parallel_rank_per_interval = n_heads * lora_rank
    sequential_rank_per_interval = lora_rank
    parallel_coverage = parallel_rank_per_interval / n_embd
    sequential_coverage = sequential_rank_per_interval / n_embd
    return {
        "parallel_rank_per_interval": parallel_rank_per_interval,
        "sequential_rank_per_interval": sequential_rank_per_interval,
        "rank_ratio": parallel_rank_per_interval / sequential_rank_per_interval,
        "parallel_coverage_pct": parallel_coverage * 100,
        "sequential_coverage_pct": sequential_coverage * 100,
        "coverage_ratio": parallel_coverage / sequential_coverage,
    }


# ── Main Experiment ───────────────────────────────────────────────────────────


@dataclass
class Results:
    seed: int
    n_embd: int
    lora_rank: int
    n_par_heads: int
    pretrain_steps: int
    adapt_steps: int
    merge_every_par: int
    merge_every_seq: int
    expert_steps: int
    n_experts: int

    # Theoretical advantage
    rank_capacity: dict = field(default_factory=dict)

    # Base quality
    pretrained_loss: float = 0.0
    parallel_loss: float = 0.0
    sequential_loss: float = 0.0
    continued_loss: float = 0.0

    # Effective rank of weight matrices
    parallel_rank: float = 0.0
    sequential_rank: float = 0.0
    continued_rank: float = 0.0

    # Timing
    parallel_time: float = 0.0
    sequential_time: float = 0.0
    continued_time: float = 0.0

    # Expert orthogonality
    parallel_mean_cos: float = 0.0
    sequential_mean_cos: float = 0.0
    continued_mean_cos: float = 0.0
    parallel_max_cos: float = 0.0
    sequential_max_cos: float = 0.0
    continued_max_cos: float = 0.0

    # Expert quality
    parallel_expert_losses: list = field(default_factory=list)
    sequential_expert_losses: list = field(default_factory=list)
    continued_expert_losses: list = field(default_factory=list)
    parallel_mean_exp_loss: float = 0.0
    sequential_mean_exp_loss: float = 0.0
    continued_mean_exp_loss: float = 0.0

    # Parallel merge info
    parallel_n_intervals: int = 0
    sequential_n_merges: int = 0

    # Ratios vs conventional
    par_base_ratio: float = 0.0
    seq_base_ratio: float = 0.0
    par_cos_ratio: float = 0.0
    seq_cos_ratio: float = 0.0
    par_loss_ratio: float = 0.0
    seq_loss_ratio: float = 0.0
    par_compute_ratio: float = 0.0

    # Parallel vs sequential head-to-head
    par_vs_seq_base: float = 0.0
    par_vs_seq_cos: float = 0.0
    par_vs_seq_loss: float = 0.0
    par_vs_seq_rank: float = 0.0

    # Kill criteria
    k1_quality_diff_pct: float = 0.0  # NEW: specific to this experiment
    k2: str = ""
    verdict: str = ""


def run_experiment(
    n_embd=256, n_head=8, n_layer=4, block_size=32,
    lora_rank=8, lora_alpha=1.0,
    pretrain_steps=400, adapt_steps=400,
    merge_every_par=25, merge_every_seq=100,
    n_par_heads=4, expert_steps=200, n_experts=4,
    batch_size=32, lr=3e-3, seed=42,
) -> Results:
    from ...data import load_names, CharTokenizer, CharDataset, domain_split

    # Theoretical advantage
    rank_cap = compute_rank_per_interval_advantage(n_embd, lora_rank, n_par_heads)

    print("=" * 72)
    print("RANK ACCUMULATION QUALITY: d=256 (mid-scale)")
    print(f"d={n_embd}, r={lora_rank}, pretrain={pretrain_steps}, adapt={adapt_steps}")
    print(f"Parallel: {n_par_heads} heads, merge every {merge_every_par}")
    print(f"  -> {rank_cap['parallel_rank_per_interval']} rank/interval "
          f"({rank_cap['parallel_coverage_pct']:.1f}% coverage)")
    print(f"Sequential: merge every {merge_every_seq}")
    print(f"  -> {rank_cap['sequential_rank_per_interval']} rank/interval "
          f"({rank_cap['sequential_coverage_pct']:.1f}% coverage)")
    print(f"Rank advantage: {rank_cap['rank_ratio']:.1f}x")
    print(f"Experts: {n_experts} x {expert_steps} steps | seed={seed}")
    print("=" * 72)

    docs = load_names()
    tok = CharTokenizer(docs)
    V = tok.vocab_size
    domains = domain_split(docs, method="quintary")
    dom_names = sorted(domains.keys())[:n_experts]

    rng_s = random.Random(seed)
    si = int(len(docs) * 0.9)
    rng_s.shuffle(dc := list(docs))
    train_ds = CharDataset(dc[:si], tok, block_size)
    val_ds = CharDataset(dc[si:], tok, block_size)

    # Phase 0: Shared pretraining
    print("\n--- Phase 0: Shared Pretraining ---")
    base = GPT(vocab_size=V, block_size=block_size,
               n_embd=n_embd, n_head=n_head, n_layer=n_layer)
    mx.eval(base.parameters())
    train_conventional(base, train_ds, pretrain_steps, batch_size, lr, seed)
    base_val = _evaluate(base, val_ds, batch_size)
    print(f"  Base val loss: {base_val:.4f}")

    # Phase 1a: Parallel adaptation
    print("\n--- Phase 1a: Parallel LoRA ---")
    par_model = _gpt_to_lora(base, lora_rank, lora_alpha)
    par_res = train_parallel_lora(par_model, train_ds, adapt_steps,
                                   merge_every_par, n_par_heads,
                                   batch_size, lr, seed + 100)
    par_val = _evaluate(par_model, val_ds, batch_size)
    print(f"  Parallel val loss: {par_val:.4f} ({par_res['n_intervals']} intervals, "
          f"{par_res['merges_done']} merges)")

    # Phase 1b: Sequential adaptation
    print("\n--- Phase 1b: Sequential LoRA ---")
    seq_model = _gpt_to_lora(base, lora_rank, lora_alpha)
    seq_res = train_sequential_lora(seq_model, train_ds, adapt_steps,
                                     merge_every_seq, batch_size, lr, seed + 200)
    seq_val = _evaluate(seq_model, val_ds, batch_size)
    print(f"  Sequential val loss: {seq_val:.4f} ({seq_res['merges_done']} merges)")

    # Phase 1c: Continued conventional
    print("\n--- Phase 1c: Continued Conventional ---")
    cont = GPT(vocab_size=V, block_size=block_size,
               n_embd=n_embd, n_head=n_head, n_layer=n_layer)
    cont.wte.weight = mx.array(base.wte.weight)
    cont.wpe.weight = mx.array(base.wpe.weight)
    for i in range(n_layer):
        cont.layers[i].attn.wq.weight = mx.array(base.layers[i].attn.wq.weight)
        cont.layers[i].attn.wk.weight = mx.array(base.layers[i].attn.wk.weight)
        cont.layers[i].attn.wv.weight = mx.array(base.layers[i].attn.wv.weight)
        cont.layers[i].attn.wo.weight = mx.array(base.layers[i].attn.wo.weight)
        cont.layers[i].mlp.fc1.weight = mx.array(base.layers[i].mlp.fc1.weight)
        cont.layers[i].mlp.fc2.weight = mx.array(base.layers[i].mlp.fc2.weight)
    cont.lm_head.weight = mx.array(base.lm_head.weight)
    mx.eval(cont.parameters())
    cont_res = train_conventional(cont, train_ds, adapt_steps, batch_size,
                                   lr, seed + 300)
    cont_val = _evaluate(cont, val_ds, batch_size)
    print(f"  Continued val loss: {cont_val:.4f}")

    # Phase 2: Spectrum
    print("\n--- Weight Spectrum ---")
    pr = compute_effective_rank(par_model)
    sr = compute_effective_rank(seq_model)
    cr = compute_effective_rank(cont)
    print(f"  Parallel:   {pr:.2f}")
    print(f"  Sequential: {sr:.2f}")
    print(f"  Continued:  {cr:.2f}")

    # Phase 3: Experts
    print("\n--- Expert Training ---")
    pd, rd, cd = [], [], []
    pl, rl, cl = [], [], []

    for i, dn in enumerate(dom_names):
        dd = domains[dn]
        rng_d = random.Random(seed + 1000 + i)
        dd_s = list(dd); rng_d.shuffle(dd_s)
        nt = max(1, int(len(dd_s) * 0.8))
        etd = CharDataset(dd_s[:nt], tok, block_size)
        evd = CharDataset(dd_s[nt:] if nt < len(dd_s) else dd_s, tok, block_size)
        es = seed + 500 + i

        print(f"  Expert {i} ({dn}): ", end="", flush=True)
        d1, v1 = train_lora_expert(par_model, etd, evd, lora_rank, lora_alpha,
                                    expert_steps, batch_size, lr, es)
        pd.append(d1); pl.append(v1); print(f"par={v1:.4f} ", end="", flush=True)

        d2, v2 = train_lora_expert(seq_model, etd, evd, lora_rank, lora_alpha,
                                    expert_steps, batch_size, lr, es)
        rd.append(d2); rl.append(v2); print(f"seq={v2:.4f} ", end="", flush=True)

        d3, v3 = train_lora_expert(cont, etd, evd, lora_rank, lora_alpha,
                                    expert_steps, batch_size, lr, es)
        cd.append(d3); cl.append(v3); print(f"cont={v3:.4f}")

    # Phase 4: Orthogonality
    print("\n--- Orthogonality ---")
    pc = compute_pairwise_cosine(pd)
    rc = compute_pairwise_cosine(rd)
    cc = compute_pairwise_cosine(cd)

    def cs(cosines):
        v = [abs(c) for (_, _, c) in cosines]
        return (sum(v)/len(v) if v else 0, max(v) if v else 0)

    pmc, pxc = cs(pc)
    rmc, rxc = cs(rc)
    cmc, cxc = cs(cc)
    print(f"  Parallel:   mean|cos|={pmc:.6f}  max={pxc:.6f}")
    print(f"  Sequential: mean|cos|={rmc:.6f}  max={rxc:.6f}")
    print(f"  Continued:  mean|cos|={cmc:.6f}  max={cxc:.6f}")

    # Phase 5: Ratios and kill criteria
    print("\n--- Kill Criteria ---")
    pml = sum(pl)/len(pl)
    rml = sum(rl)/len(rl)
    cml = sum(cl)/len(cl)
    eps = 1e-12

    pbr = par_val / (cont_val + eps)
    sbr = seq_val / (cont_val + eps)
    pcr = pmc / (cmc + eps)
    scr = rmc / (cmc + eps)
    plr = pml / (cml + eps)
    slr = rml / (cml + eps)
    ptr = par_res['elapsed_s'] / (seq_res['elapsed_s'] + eps)

    pvb = par_val / (seq_val + eps)
    pvc = pmc / (rmc + eps)
    pvl = pml / (rml + eps)
    pvr = pr / (sr + eps)

    # NEW kill criteria for this experiment:
    # K1: quality difference <1% at d=256 -> no scaling advantage -> KILLED
    # We measure quality difference as |1 - par_vs_seq_base| * 100
    quality_diff_pct = abs(1.0 - pvb) * 100
    base_diff_pct = abs(1.0 - pvb) * 100
    cos_diff_pct = abs(1.0 - pvc) * 100
    loss_diff_pct = abs(1.0 - pvl) * 100
    rank_diff_pct = abs(1.0 - pvr) * 100

    # K2 from parent: parallel base >20% worse than conventional
    k2 = pbr > 1.20
    k2v = "KILLED" if k2 else "SURVIVES"

    # Primary kill: if all quality metrics show <1% difference, parallel
    # has no advantage at this scale either
    all_within_1pct = (base_diff_pct < 1.0 and cos_diff_pct < 1.0 and
                       loss_diff_pct < 1.0)
    # But also check: is parallel BETTER by >1%? That would be a positive finding
    parallel_better = (pvb < 0.99 or pvc < 0.99 or pvl < 0.99)

    if k2:
        verdict = "KILLED (parallel >20% worse than conventional)"
    elif all_within_1pct:
        verdict = "KILLED (no quality advantage at d=256, <1% difference)"
    elif parallel_better:
        verdict = "PROVEN (parallel shows quality advantage at d=256)"
    else:
        verdict = "INCONCLUSIVE"

    print(f"  Quality diff (par vs seq):")
    print(f"    Base loss: {base_diff_pct:.2f}% (par/seq = {pvb:.4f})")
    print(f"    Expert cos: {cos_diff_pct:.2f}% (par/seq = {pvc:.4f})")
    print(f"    Expert loss: {loss_diff_pct:.2f}% (par/seq = {pvl:.4f})")
    print(f"    Effective rank: {rank_diff_pct:.2f}% (par/seq = {pvr:.4f})")
    print(f"  K2: par base ratio = {pbr:.4f} (>1.20?) -> {k2v}")
    print(f"  Rank capacity: {rank_cap['parallel_rank_per_interval']} vs "
          f"{rank_cap['sequential_rank_per_interval']} per interval "
          f"({rank_cap['rank_ratio']:.0f}x)")

    print(f"\n  VERDICT: {verdict}")

    res = Results(
        seed=seed, n_embd=n_embd, lora_rank=lora_rank,
        n_par_heads=n_par_heads, pretrain_steps=pretrain_steps,
        adapt_steps=adapt_steps, merge_every_par=merge_every_par,
        merge_every_seq=merge_every_seq, expert_steps=expert_steps,
        n_experts=n_experts, rank_capacity=rank_cap,
        pretrained_loss=base_val, parallel_loss=par_val,
        sequential_loss=seq_val, continued_loss=cont_val,
        parallel_rank=pr, sequential_rank=sr, continued_rank=cr,
        parallel_time=par_res['elapsed_s'],
        sequential_time=seq_res['elapsed_s'],
        continued_time=cont_res['elapsed_s'],
        parallel_mean_cos=pmc, sequential_mean_cos=rmc, continued_mean_cos=cmc,
        parallel_max_cos=pxc, sequential_max_cos=rxc, continued_max_cos=cxc,
        parallel_expert_losses=pl, sequential_expert_losses=rl,
        continued_expert_losses=cl,
        parallel_mean_exp_loss=pml, sequential_mean_exp_loss=rml,
        continued_mean_exp_loss=cml,
        parallel_n_intervals=par_res.get('n_intervals', 0),
        sequential_n_merges=seq_res['merges_done'],
        par_base_ratio=pbr, seq_base_ratio=sbr,
        par_cos_ratio=pcr, seq_cos_ratio=scr,
        par_loss_ratio=plr, seq_loss_ratio=slr,
        par_compute_ratio=ptr,
        par_vs_seq_base=pvb, par_vs_seq_cos=pvc,
        par_vs_seq_loss=pvl, par_vs_seq_rank=pvr,
        k1_quality_diff_pct=quality_diff_pct,
        k2=k2v, verdict=verdict,
    )

    out = os.path.join(os.path.dirname(__file__), f"results_seed_{seed}.json")
    with open(out, "w") as f:
        json.dump(asdict(res), f, indent=2)
    print(f"\nSaved to {out}")
    return res


def run_multi_seed(seeds=None, **kw):
    if seeds is None:
        seeds = [42, 123, 7]

    per = {}
    for s in seeds:
        print(f"\n{'='*72}\nSEED {s}\n{'='*72}")
        r = run_experiment(seed=s, **kw)
        per[str(s)] = asdict(r)

    def mean(v): return sum(v)/len(v) if v else 0
    def ci(v):
        rng = random.Random(42); n = len(v)
        b = sorted(mean([rng.choice(v) for _ in range(n)]) for _ in range(5000))
        return [b[125], b[4875]]

    keys = ["par_base_ratio", "seq_base_ratio", "par_cos_ratio",
            "seq_cos_ratio", "par_loss_ratio", "seq_loss_ratio",
            "par_vs_seq_base", "par_vs_seq_cos", "par_vs_seq_loss",
            "par_vs_seq_rank", "k1_quality_diff_pct",
            "parallel_rank", "sequential_rank", "continued_rank"]
    agg = {k: {"mean": mean([per[s][k] for s in per]),
               "ci95": ci([per[s][k] for s in per])} for k in keys}

    # Aggregate verdict
    vs = [per[s]["verdict"] for s in per]
    if any("KILLED" in v and "no quality advantage" in v for v in vs):
        # If majority show no advantage, it's killed
        killed_count = sum(1 for v in vs if "KILLED" in v and "no quality advantage" in v)
        if killed_count >= len(vs) // 2 + 1:
            agg["overall"] = "KILLED (no quality advantage at d=256)"
        else:
            agg["overall"] = "INCONCLUSIVE"
    elif any("PROVEN" in v for v in vs):
        proven_count = sum(1 for v in vs if "PROVEN" in v)
        if proven_count >= len(vs) // 2 + 1:
            agg["overall"] = "PROVEN (parallel shows advantage at d=256)"
        else:
            agg["overall"] = "INCONCLUSIVE"
    elif any("KILLED" in v for v in vs):
        agg["overall"] = "KILLED"
    else:
        agg["overall"] = "INCONCLUSIVE"

    # Compare with d=64 parent results
    agg["d64_comparison"] = {
        "d64_par_vs_seq_base": 1.007,  # from parent
        "d256_par_vs_seq_base": agg["par_vs_seq_base"]["mean"],
        "d64_par_vs_seq_cos": 1.462,   # from parent
        "d256_par_vs_seq_cos": agg["par_vs_seq_cos"]["mean"],
        "d64_par_vs_seq_loss": 1.006,  # from parent
        "d256_par_vs_seq_loss": agg["par_vs_seq_loss"]["mean"],
        "scaling_effect_base": agg["par_vs_seq_base"]["mean"] - 1.007,
        "scaling_effect_cos": agg["par_vs_seq_cos"]["mean"] - 1.462,
        "scaling_effect_loss": agg["par_vs_seq_loss"]["mean"] - 1.006,
    }

    out = os.path.join(os.path.dirname(__file__), "results_aggregate.json")
    with open(out, "w") as f:
        json.dump({"seeds": per, "aggregate": agg}, f, indent=2)

    print(f"\n{'='*72}\nAGGREGATE (d={kw.get('n_embd', 256)})\n{'='*72}")
    for k in keys:
        print(f"  {k}: {agg[k]['mean']:.4f}  CI {agg[k]['ci95']}")
    print(f"\n  d=64 comparison:")
    for k, v in agg["d64_comparison"].items():
        print(f"    {k}: {v:.4f}")
    print(f"\n  Overall: {agg['overall']}")
    return {"seeds": per, "aggregate": agg}


if __name__ == "__main__":
    run_multi_seed()
