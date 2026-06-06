"""
LTE Parallel Base Construction: Does parallel multi-head LoRA adaptation
produce a better composition substrate than ReLoRA sequential merging?

Design (3-way comparison on a pretrained base):
  Phase 0: Train a conventional base model (shared).
  Phase 1a: PARALLEL adaptation - K parallel rank-r LoRA branches, each
     trained on a different data shard, averaged and merged. Reset after merge.
  Phase 1b: SEQUENTIAL adaptation (ReLoRA) - K sequential rank-r LoRA
     merge-and-restart cycles, one at a time.
  Phase 1c: Continued conventional training (control).
  Phase 2: Train N=4 domain experts on each adapted base.
  Phase 3: Compare composition substrate quality.

KEY VARIABLE ISOLATED: Parallel vs Sequential LoRA merging.
Both conditions use reset-after-merge. Both see the same total data.
Both have the same total gradient compute (compute-fair).
The ONLY difference is whether branches are trained simultaneously
on different shards (parallel) or sequentially (one after another).

Parallel advantage hypothesis (Hyeon-Woo et al. 2024):
  - K heads trained on different data shards explore diverse subspaces
  - Their averaged contribution is higher-rank than any single branch
  - This should produce a more "explored" weight space = better substrate

Sequential advantage hypothesis (baseline):
  - Each branch builds on the previous merged state
  - Sequential composition can be more coherent
  - No subspace averaging artifacts

Kill criteria:
  K1: Parallel base quality >20% worse than conventional at same compute
  K2: Parallel requires >2x compute vs sequential for same quality
  K3: Parallel branches interfere when merged (defeating the purpose)
"""

import math
import time
import random
import json
import os
from dataclasses import dataclass, asdict

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
    model: GPT, dataset, total_steps: int = 1000,
    batch_size: int = 32, lr: float = 3e-3, seed: int = 42,
    log_every: int = 200,
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
    model: LoRAGPT, dataset, total_steps: int = 500,
    merge_every: int = 50, n_heads: int = 4,
    batch_size: int = 32, lr: float = 3e-3, seed: int = 42,
    log_every: int = 200,
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
            "total_grad_steps": n_intervals * n_heads * merge_every}


# ── SEQUENTIAL LoRA Merge (ReLoRA-style) ──────────────────────────────────────


def train_sequential_lora(
    model: LoRAGPT, dataset, total_steps: int = 500,
    merge_every: int = 100, batch_size: int = 32, lr: float = 3e-3,
    seed: int = 42, log_every: int = 200,
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
            "merges_done": merges_done, "elapsed_s": time.time() - t0}


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
                      steps=300, batch_size=32, lr=3e-3, seed=42):
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

    pretrained_loss: float
    parallel_loss: float
    sequential_loss: float
    continued_loss: float

    parallel_rank: float
    sequential_rank: float
    continued_rank: float

    parallel_time: float
    sequential_time: float
    continued_time: float

    parallel_mean_cos: float
    sequential_mean_cos: float
    continued_mean_cos: float
    parallel_max_cos: float
    sequential_max_cos: float
    continued_max_cos: float

    parallel_expert_losses: list
    sequential_expert_losses: list
    continued_expert_losses: list
    parallel_mean_exp_loss: float
    sequential_mean_exp_loss: float
    continued_mean_exp_loss: float

    par_base_ratio: float
    seq_base_ratio: float
    par_cos_ratio: float
    seq_cos_ratio: float
    par_loss_ratio: float
    seq_loss_ratio: float
    par_compute_ratio: float

    par_vs_seq_base: float
    par_vs_seq_cos: float
    par_vs_seq_loss: float

    k1: str
    k2: str
    k3: str
    verdict: str


def run_experiment(
    n_embd=64, n_head=4, n_layer=4, block_size=32,
    lora_rank=8, lora_alpha=1.0,
    pretrain_steps=500, adapt_steps=500,
    merge_every_par=50, merge_every_seq=100,
    n_par_heads=4, expert_steps=300, n_experts=4,
    batch_size=32, lr=3e-3, seed=42,
) -> Results:
    from ...data import load_names, CharTokenizer, CharDataset, domain_split

    print("=" * 72)
    print("PARALLEL vs SEQUENTIAL LoRA MERGE: COMPOSITION SUBSTRATE")
    print(f"d={n_embd}, r={lora_rank}, pretrain={pretrain_steps}, adapt={adapt_steps}")
    print(f"Parallel: {n_par_heads} heads, merge every {merge_every_par}")
    print(f"Sequential: merge every {merge_every_seq}")
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
    print(f"  Parallel val loss: {par_val:.4f}")

    # Phase 1b: Sequential adaptation
    print("\n--- Phase 1b: Sequential LoRA ---")
    seq_model = _gpt_to_lora(base, lora_rank, lora_alpha)
    seq_res = train_sequential_lora(seq_model, train_ds, adapt_steps,
                                     merge_every_seq, batch_size, lr, seed + 200)
    seq_val = _evaluate(seq_model, val_ds, batch_size)
    print(f"  Sequential val loss: {seq_val:.4f}")

    # Phase 1c: Continued conventional
    print("\n--- Phase 1c: Continued Conventional ---")
    cont = GPT(vocab_size=V, block_size=block_size,
               n_embd=n_embd, n_head=n_head, n_layer=n_layer)
    # Copy pretrained weights
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

        print(f"  Expert {i} ({dn}): ", end="")
        d1, v1 = train_lora_expert(par_model, etd, evd, lora_rank, lora_alpha,
                                    expert_steps, batch_size, lr, es)
        pd.append(d1); pl.append(v1); print(f"par={v1:.4f} ", end="")

        d2, v2 = train_lora_expert(seq_model, etd, evd, lora_rank, lora_alpha,
                                    expert_steps, batch_size, lr, es)
        rd.append(d2); rl.append(v2); print(f"seq={v2:.4f} ", end="")

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

    # Phase 5: Kill criteria
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

    k1 = pbr > 1.20
    k2 = ptr > 2.0
    k3 = pcr > 5.0

    k1v = "KILLED" if k1 else "SURVIVES"
    k2v = "KILLED" if k2 else "SURVIVES"
    k3v = "KILLED" if k3 else "SURVIVES"

    print(f"  K1: par base ratio = {pbr:.4f} (>1.20?) -> {k1v}")
    print(f"  K2: par compute ratio = {ptr:.2f}x (>2.0?) -> {k2v}")
    print(f"  K3: par cos ratio = {pcr:.4f} (>5.0?) -> {k3v}")

    if k1 or k2 or k3:
        verd = "KILLED"
    elif pbr < 1.10 and pcr < 2.0 and plr < 1.20:
        verd = "SURVIVES"
    else:
        verd = "INCONCLUSIVE"

    print(f"\n--- Parallel vs Sequential ---")
    print(f"  Base: par={par_val:.4f} seq={seq_val:.4f} ratio={pvb:.4f}")
    print(f"  Cos:  par={pmc:.6f} seq={rmc:.6f} ratio={pvc:.4f}")
    print(f"  Loss: par={pml:.4f} seq={rml:.4f} ratio={pvl:.4f}")
    print(f"\n  VERDICT: {verd}")

    res = Results(
        seed=seed, n_embd=n_embd, lora_rank=lora_rank,
        n_par_heads=n_par_heads, pretrain_steps=pretrain_steps,
        adapt_steps=adapt_steps, merge_every_par=merge_every_par,
        merge_every_seq=merge_every_seq, expert_steps=expert_steps,
        n_experts=n_experts,
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
        par_base_ratio=pbr, seq_base_ratio=sbr,
        par_cos_ratio=pcr, seq_cos_ratio=scr,
        par_loss_ratio=plr, seq_loss_ratio=slr,
        par_compute_ratio=ptr,
        par_vs_seq_base=pvb, par_vs_seq_cos=pvc, par_vs_seq_loss=pvl,
        k1=k1v, k2=k2v, k3=k3v, verdict=verd,
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
            "par_vs_seq_base", "par_vs_seq_cos", "par_vs_seq_loss"]
    agg = {k: {"mean": mean([per[s][k] for s in per]),
               "ci95": ci([per[s][k] for s in per])} for k in keys}

    vs = [per[s]["verdict"] for s in per]
    agg["overall"] = ("KILLED" if any("KILLED" in v for v in vs)
                      else "SURVIVES" if all(v == "SURVIVES" for v in vs)
                      else "INCONCLUSIVE")

    out = os.path.join(os.path.dirname(__file__), "results_aggregate.json")
    with open(out, "w") as f:
        json.dump({"seeds": per, "aggregate": agg}, f, indent=2)

    print(f"\n{'='*72}\nAGGREGATE\n{'='*72}")
    for k in keys:
        print(f"  {k}: {agg[k]['mean']:.4f}  CI {agg[k]['ci95']}")
    print(f"  Overall: {agg['overall']}")
    return {"seeds": per, "aggregate": agg}


if __name__ == "__main__":
    run_multi_seed()
