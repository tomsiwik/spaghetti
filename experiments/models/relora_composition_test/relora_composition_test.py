"""
ReLoRA Composition Test: Do LoRA experts compose on a ReLoRA-built base model?

This experiment tests the critical empirical gate for the "base is just another
adapter" thesis from the adapter taxonomy survey. Key question: if a base model
is built via iterative LoRA merge-and-restart (ReLoRA, Lialin et al. 2023),
do subsequently trained LoRA experts exhibit the same near-orthogonality
(cos ~ 10^{-4}) as experts on a conventionally pretrained base?

Design:
  1. Build a micro GPT via ReLoRA (K merge cycles of rank-r LoRA on random init)
  2. Build a conventional GPT via standard training (same total steps/data)
  3. Train N=4 domain LoRA experts on each base
  4. Measure pairwise cosine similarity of expert deltas
  5. Compare expert quality (val loss)

Architecture: Uses existing LoRA infrastructure from lora_procrustes.
All training uses the micro names dataset (character-level).
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

from ..gpt import GPT, RMSNorm, CausalSelfAttention, MLP, Block
from ..lora_procrustes.lora_procrustes import LoRALinear, LoRAMLP, LoRABlock, LoRAGPT


# ── ReLoRA Training ────────────────────────────────────────────────────────────


def merge_lora_into_base(model: LoRAGPT):
    """Merge all LoRA deltas into base weights and reset A/B to zero.

    This is the core ReLoRA operation: W_base += (alpha/r) * A @ B, then
    reset A to random, B to zero.

    After merge, the LoRA output is zero (B=0), so the model's output is
    unchanged but the knowledge is now in the base weights.
    """
    for layer in model.layers:
        for fc_name in ['fc1', 'fc2']:
            fc = getattr(layer.mlp, fc_name)
            # Merge: W += delta
            delta = fc.get_delta()
            new_weight = fc.linear.weight + delta.T  # nn.Linear stores (out, in)
            fc.linear.weight = new_weight

            # Reset LoRA: A to small random, B to zero
            in_dim = fc.A.shape[0]
            rank = fc.A.shape[1]
            out_dim = fc.B.shape[1]
            scale = (2.0 / in_dim) ** 0.5
            fc.A = mx.random.normal((in_dim, rank)) * scale
            fc.B = mx.zeros((rank, out_dim))

    mx.eval(model.parameters())


def train_relora(
    model: LoRAGPT,
    dataset,
    total_steps: int = 1000,
    merge_every: int = 200,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 42,
    log_every: int = 100,
) -> dict:
    """Train a model via ReLoRA: iterative LoRA merge-and-restart.

    Protocol (following Lialin et al. 2023):
    1. Train LoRA for `merge_every` steps
    2. Merge LoRA deltas into base weights
    3. Reset LoRA A/B and optimizer state
    4. Restart with learning rate warmup
    5. Repeat until total_steps reached

    Returns training metrics dict.
    """
    rng = random.Random(seed)
    losses = []
    t0 = time.time()
    total_tokens = 0
    merges_done = 0

    # Train all parameters (base + LoRA) during ReLoRA pretraining
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, _ntp_loss)

    for step in range(1, total_steps + 1):
        inputs, targets = dataset.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()
        losses.append(loss_val)
        total_tokens += inputs.size

        if step % log_every == 0 or step == total_steps:
            elapsed = time.time() - t0
            tps = total_tokens / elapsed if elapsed > 0 else 0
            print(f"  [ReLoRA] step {step:4d}/{total_steps} | loss {loss_val:.4f} "
                  f"| {tps:.0f} tok/s | merges: {merges_done}")

        # Merge cycle
        if step % merge_every == 0 and step < total_steps:
            merge_lora_into_base(model)
            merges_done += 1
            # Reset optimizer (following Lialin et al.)
            optimizer = optim.Adam(learning_rate=lr)
            loss_and_grad = nn.value_and_grad(model, _ntp_loss)

    # Final merge
    merge_lora_into_base(model)
    merges_done += 1

    elapsed = time.time() - t0
    return {
        "final_loss": losses[-1],
        "losses": losses,
        "merges_done": merges_done,
        "elapsed_s": elapsed,
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0,
    }


def train_conventional(
    model: GPT,
    dataset,
    total_steps: int = 1000,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 42,
    log_every: int = 100,
) -> dict:
    """Standard pretraining: train all params for total_steps."""
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, _ntp_loss_gpt)

    losses = []
    t0 = time.time()
    total_tokens = 0

    for step in range(1, total_steps + 1):
        inputs, targets = dataset.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()
        losses.append(loss_val)
        total_tokens += inputs.size

        if step % log_every == 0 or step == total_steps:
            elapsed = time.time() - t0
            tps = total_tokens / elapsed if elapsed > 0 else 0
            print(f"  [Conv]   step {step:4d}/{total_steps} | loss {loss_val:.4f} "
                  f"| {tps:.0f} tok/s")

    elapsed = time.time() - t0
    return {
        "final_loss": losses[-1],
        "losses": losses,
        "elapsed_s": elapsed,
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0,
    }


# ── LoRA Expert Training ───────────────────────────────────────────────────────


def _ntp_loss(model, inputs, targets):
    """Next-token prediction loss."""
    logits = model(inputs)
    B, T, V = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
    )


def _ntp_loss_gpt(model, inputs, targets):
    """Next-token prediction loss for plain GPT."""
    logits = model(inputs)
    B, T, V = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
    )


def attach_lora_to_gpt(model: GPT, rank: int = 8, alpha: float = 1.0) -> LoRAGPT:
    """Create a LoRAGPT that shares the base weights of a pretrained GPT.

    This wraps a frozen GPT model with LoRA adapters on its MLP layers.
    The base weights are copied (not shared) to allow independent experts.
    """
    n_embd = model.wte.weight.shape[1]
    n_head = model.layers[0].attn.n_head
    n_layer = len(model.layers)
    vocab_size = model.wte.weight.shape[0]
    block_size = model.wpe.weight.shape[0]

    lora_model = LoRAGPT(
        vocab_size=vocab_size, block_size=block_size,
        n_embd=n_embd, n_head=n_head, n_layer=n_layer,
        lora_rank=rank, lora_alpha=alpha,
    )

    # Copy base weights from pretrained GPT
    lora_model.wte.weight = model.wte.weight
    lora_model.wpe.weight = model.wpe.weight
    for i, layer in enumerate(model.layers):
        # Copy attention weights
        lora_model.layers[i].attn.wq.weight = layer.attn.wq.weight
        lora_model.layers[i].attn.wk.weight = layer.attn.wk.weight
        lora_model.layers[i].attn.wv.weight = layer.attn.wv.weight
        lora_model.layers[i].attn.wo.weight = layer.attn.wo.weight
        # Copy MLP base weights
        lora_model.layers[i].mlp.fc1.linear.weight = layer.mlp.fc1.weight
        lora_model.layers[i].mlp.fc2.linear.weight = layer.mlp.fc2.weight
    lora_model.lm_head.weight = model.lm_head.weight
    mx.eval(lora_model.parameters())

    return lora_model


def train_lora_expert(
    base_model,
    train_dataset,
    val_dataset,
    rank: int = 8,
    alpha: float = 1.0,
    steps: int = 300,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 42,
    is_relora_base: bool = False,
) -> tuple:
    """Train a LoRA expert on a frozen base, return (model, deltas, val_loss).

    If base_model is a GPT, wraps it with LoRA first.
    If base_model is a LoRAGPT (from ReLoRA), creates a fresh LoRA copy.

    FIX(rev2): Both train_dataset and val_dataset are now explicit arguments
    to ensure both conditions (ReLoRA, conventional) train on the same data
    and evaluate on the same held-out data.
    """
    if isinstance(base_model, LoRAGPT) or is_relora_base:
        # For ReLoRA base: create a new LoRAGPT with same base weights
        # but fresh LoRA A/B
        lora_model = _copy_relora_base_with_fresh_lora(base_model, rank, alpha)
    else:
        lora_model = attach_lora_to_gpt(base_model, rank, alpha)

    # Freeze everything, then unfreeze only LoRA A/B params
    lora_model.freeze()
    for layer in lora_model.layers:
        layer.mlp.fc1.unfreeze(keys=["A", "B"])
        layer.mlp.fc2.unfreeze(keys=["A", "B"])
    mx.eval(lora_model.parameters())

    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(lora_model, _ntp_loss)

    for step in range(1, steps + 1):
        inputs, targets = train_dataset.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(lora_model, inputs, targets)
        optimizer.update(lora_model, grads)
        mx.eval(lora_model.parameters(), optimizer.state)

    # Compute val loss on held-out data (same dataset for both conditions)
    val_loss = _evaluate(lora_model, val_dataset, batch_size)

    # Extract deltas
    deltas = lora_model.get_all_deltas()

    return lora_model, deltas, val_loss


def _copy_relora_base_with_fresh_lora(
    model: LoRAGPT, rank: int = 8, alpha: float = 1.0
) -> LoRAGPT:
    """Copy a LoRAGPT's base weights into a new LoRAGPT with fresh LoRA params.

    For a ReLoRA-trained model, the base weights contain the merged knowledge.
    The existing LoRA A/B should be zero after final merge.
    """
    n_embd = model.wte.weight.shape[1]
    n_head = model.layers[0].attn.n_head
    n_layer = len(model.layers)
    vocab_size = model.wte.weight.shape[0]
    block_size = model.wpe.weight.shape[0]

    new_model = LoRAGPT(
        vocab_size=vocab_size, block_size=block_size,
        n_embd=n_embd, n_head=n_head, n_layer=n_layer,
        lora_rank=rank, lora_alpha=alpha,
    )

    # Copy all weights from the ReLoRA model
    new_model.wte.weight = model.wte.weight
    new_model.wpe.weight = model.wpe.weight
    for i in range(n_layer):
        new_model.layers[i].attn.wq.weight = model.layers[i].attn.wq.weight
        new_model.layers[i].attn.wk.weight = model.layers[i].attn.wk.weight
        new_model.layers[i].attn.wv.weight = model.layers[i].attn.wv.weight
        new_model.layers[i].attn.wo.weight = model.layers[i].attn.wo.weight
        # Copy merged base weights from LoRA layers
        new_model.layers[i].mlp.fc1.linear.weight = model.layers[i].mlp.fc1.linear.weight
        new_model.layers[i].mlp.fc2.linear.weight = model.layers[i].mlp.fc2.linear.weight
    new_model.lm_head.weight = model.lm_head.weight
    mx.eval(new_model.parameters())

    return new_model


def _evaluate(model, dataset, batch_size: int = 32, n_batches: int = 10) -> float:
    """Evaluate model on dataset, return mean loss."""
    rng = random.Random(999)
    total_loss = 0.0
    for _ in range(n_batches):
        inputs, targets = dataset.get_batch(batch_size, rng)
        logits = model(inputs)
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
        )
        mx.eval(loss)
        total_loss += loss.item()
    return total_loss / n_batches


# ── Orthogonality Measurement ──────────────────────────────────────────────────


def compute_pairwise_cosine(deltas_list: list) -> list:
    """Compute pairwise cosine similarity between expert delta sets.

    Each entry in deltas_list is a list of (layer_idx, fc_name, delta_matrix)
    from get_all_deltas(). We flatten all deltas into a single vector per
    expert, then compute pairwise cosines.

    Returns list of (expert_i, expert_j, cosine_similarity).
    """
    # Flatten each expert's deltas into a single vector
    flat_vectors = []
    for deltas in deltas_list:
        parts = [d.reshape(-1) for (_, _, d) in deltas]
        flat = mx.concatenate(parts)
        flat_vectors.append(flat)

    results = []
    n = len(flat_vectors)
    for i in range(n):
        for j in range(i + 1, n):
            vi = flat_vectors[i]
            vj = flat_vectors[j]
            cos = (vi @ vj) / (mx.sqrt(vi @ vi) * mx.sqrt(vj @ vj) + 1e-12)
            mx.eval(cos)
            results.append((i, j, cos.item()))

    return results


def compute_weight_spectrum(model) -> dict:
    """Analyze the singular value spectrum of weight matrices.

    Returns statistics about the weight spectrum that might differ between
    ReLoRA and conventionally trained models.
    """
    spectra = {}
    for l_idx, layer in enumerate(model.layers):
        if hasattr(layer.mlp, 'fc1'):
            # Get the weight matrix
            if hasattr(layer.mlp.fc1, 'linear'):
                w = layer.mlp.fc1.linear.weight  # LoRAGPT
            else:
                w = layer.mlp.fc1.weight  # GPT
            # Compute SVD
            U, S, Vt = mx.linalg.svd(w, stream=mx.cpu)
            mx.eval(S)
            s_vals = S.tolist()
            spectra[f"layer_{l_idx}_fc1"] = {
                "max_sv": max(s_vals),
                "min_sv": min(s_vals),
                "condition_number": max(s_vals) / (min(s_vals) + 1e-12),
                "effective_rank": _effective_rank(s_vals),
                "top_5_sv": s_vals[:5],
            }
    return spectra


def _effective_rank(singular_values: list) -> float:
    """Compute effective rank (Roy & Vetterli, 2007): exp(H(p)) where
    p_i = sigma_i / sum(sigma_j) and H is Shannon entropy."""
    total = sum(singular_values)
    if total < 1e-12:
        return 0.0
    probs = [s / total for s in singular_values]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs if p > 1e-12)
    return math.exp(entropy)


# ── Main Experiment ─────────────────────────────────────────────────────────────


@dataclass
class ExperimentResults:
    """Complete results from the ReLoRA composition test."""
    # Base model training
    relora_final_loss: float
    conventional_final_loss: float
    relora_merges: int
    relora_train_time: float
    conventional_train_time: float

    # Expert orthogonality
    relora_mean_cos: float
    conventional_mean_cos: float
    relora_max_cos: float
    conventional_max_cos: float
    relora_cosines: list
    conventional_cosines: list

    # Expert quality
    relora_expert_losses: list
    conventional_expert_losses: list
    relora_mean_expert_loss: float
    conventional_mean_expert_loss: float

    # Weight spectrum
    relora_effective_rank: float
    conventional_effective_rank: float

    # Ratios
    cos_ratio: float  # relora_mean_cos / conv_mean_cos
    loss_ratio: float  # relora_mean_expert_loss / conv_mean_expert_loss

    # Kill criteria evaluation
    kill_cos_violated: bool  # cos ratio > 10x
    kill_quality_violated: bool  # loss ratio > 2.0x
    kill_coherence_violated: bool  # ReLoRA base incoherent (> 2.0x)

    # Overall verdict
    verdict: str  # "SURVIVES", "KILLED", or "INCONCLUSIVE"


def run_experiment(
    n_embd: int = 64,
    n_head: int = 4,
    n_layer: int = 4,
    block_size: int = 32,
    lora_rank: int = 8,
    lora_alpha: float = 1.0,
    total_pretrain_steps: int = 1000,
    merge_every: int = 200,
    expert_train_steps: int = 300,
    n_experts: int = 4,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 42,
) -> ExperimentResults:
    """Run the complete ReLoRA composition test.

    Total runtime: ~2-3 minutes on Apple Silicon (M-series).
    """
    from ...data import load_names, CharTokenizer, CharDataset, domain_split

    print("=" * 72)
    print("ReLoRA COMPOSITION TEST")
    print(f"Config: d={n_embd}, h={n_head}, L={n_layer}, r={lora_rank}")
    print(f"Pretrain: {total_pretrain_steps} steps, merge every {merge_every}")
    print(f"Experts: {n_experts} x {expert_train_steps} steps")
    print("=" * 72)

    # ── Load data ──────────────────────────────────────────────────────────
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vocab_size = tokenizer.vocab_size

    # Split into domains for expert training (quintary = 5 groups by first letter)
    domains = domain_split(docs, method="quintary")
    domain_names = sorted(domains.keys())[:n_experts]
    print(f"\nDomains for expert training: {domain_names}")

    # Use full data for pretraining
    rng_split = random.Random(seed)
    split_idx = int(len(docs) * 0.9)
    rng_split.shuffle(docs_copy := list(docs))
    train_docs = docs_copy[:split_idx]
    val_docs = docs_copy[split_idx:]

    train_ds = CharDataset(train_docs, tokenizer, block_size)
    val_ds = CharDataset(val_docs, tokenizer, block_size)

    # ── Phase 1: Build ReLoRA base ─────────────────────────────────────────
    print("\n--- Phase 1a: ReLoRA Pretraining ---")
    relora_model = LoRAGPT(
        vocab_size=vocab_size, block_size=block_size,
        n_embd=n_embd, n_head=n_head, n_layer=n_layer,
        lora_rank=lora_rank, lora_alpha=lora_alpha,
    )
    mx.eval(relora_model.parameters())

    relora_result = train_relora(
        relora_model, train_ds,
        total_steps=total_pretrain_steps, merge_every=merge_every,
        batch_size=batch_size, lr=lr, seed=seed,
    )
    relora_val = _evaluate(relora_model, val_ds, batch_size)
    print(f"  ReLoRA final val loss: {relora_val:.4f} "
          f"({relora_result['merges_done']} merges)")

    # ── Phase 1b: Build conventional base ──────────────────────────────────
    print("\n--- Phase 1b: Conventional Pretraining ---")
    conv_model = GPT(
        vocab_size=vocab_size, block_size=block_size,
        n_embd=n_embd, n_head=n_head, n_layer=n_layer,
    )
    mx.eval(conv_model.parameters())

    conv_result = train_conventional(
        conv_model, train_ds,
        total_steps=total_pretrain_steps, batch_size=batch_size,
        lr=lr, seed=seed,
    )
    conv_val = _evaluate(conv_model, val_ds, batch_size)
    print(f"  Conventional final val loss: {conv_val:.4f}")

    # ── Phase 2: Weight spectrum analysis ──────────────────────────────────
    print("\n--- Phase 2: Weight Spectrum Analysis ---")
    relora_spectrum = compute_weight_spectrum(relora_model)
    conv_spectrum = compute_weight_spectrum(conv_model)

    relora_ranks = [v["effective_rank"] for v in relora_spectrum.values()]
    conv_ranks = [v["effective_rank"] for v in conv_spectrum.values()]
    relora_mean_rank = sum(relora_ranks) / len(relora_ranks) if relora_ranks else 0
    conv_mean_rank = sum(conv_ranks) / len(conv_ranks) if conv_ranks else 0
    print(f"  ReLoRA mean effective rank: {relora_mean_rank:.2f}")
    print(f"  Conv   mean effective rank: {conv_mean_rank:.2f}")

    # ── Phase 3: Train domain experts on both bases ────────────────────────
    print("\n--- Phase 3: Expert Training ---")
    relora_deltas_all = []
    conv_deltas_all = []
    relora_expert_losses = []
    conv_expert_losses = []

    for i, domain in enumerate(domain_names):
        domain_docs = domains[domain]
        # Deterministic 80/20 split per domain (seeded by domain index)
        rng_domain = random.Random(seed + 1000 + i)
        domain_docs_shuffled = list(domain_docs)
        rng_domain.shuffle(domain_docs_shuffled)
        n_train = max(1, int(len(domain_docs_shuffled) * 0.8))
        expert_train_ds = CharDataset(domain_docs_shuffled[:n_train], tokenizer, block_size)
        expert_val_ds = CharDataset(
            domain_docs_shuffled[n_train:] if n_train < len(domain_docs_shuffled)
            else domain_docs_shuffled, tokenizer, block_size
        )

        # FIX(rev2): Both conditions now train on expert_train_ds and
        # evaluate on expert_val_ds (same splits for fair comparison).

        # Expert on ReLoRA base
        print(f"  Expert {i} ({domain}) on ReLoRA base...", end=" ")
        _, r_deltas, r_val = train_lora_expert(
            relora_model, expert_train_ds, expert_val_ds,
            rank=lora_rank, alpha=lora_alpha,
            steps=expert_train_steps, batch_size=batch_size,
            lr=lr, seed=seed + i, is_relora_base=True,
        )
        relora_deltas_all.append(r_deltas)
        relora_expert_losses.append(r_val)
        print(f"val_loss={r_val:.4f}")

        # Expert on conventional base
        print(f"  Expert {i} ({domain}) on conv base...", end=" ")
        _, c_deltas, c_val = train_lora_expert(
            conv_model, expert_train_ds, expert_val_ds,
            rank=lora_rank, alpha=lora_alpha,
            steps=expert_train_steps, batch_size=batch_size,
            lr=lr, seed=seed + i, is_relora_base=False,
        )
        conv_deltas_all.append(c_deltas)
        conv_expert_losses.append(c_val)
        print(f"val_loss={c_val:.4f}")

    # ── Phase 4: Measure orthogonality ─────────────────────────────────────
    print("\n--- Phase 4: Orthogonality Measurement ---")
    relora_cosines = compute_pairwise_cosine(relora_deltas_all)
    conv_cosines = compute_pairwise_cosine(conv_deltas_all)

    relora_cos_vals = [abs(c) for (_, _, c) in relora_cosines]
    conv_cos_vals = [abs(c) for (_, _, c) in conv_cosines]

    relora_mean_cos = sum(relora_cos_vals) / len(relora_cos_vals) if relora_cos_vals else 0
    conv_mean_cos = sum(conv_cos_vals) / len(conv_cos_vals) if conv_cos_vals else 0
    relora_max_cos = max(relora_cos_vals) if relora_cos_vals else 0
    conv_max_cos = max(conv_cos_vals) if conv_cos_vals else 0

    print(f"  ReLoRA base experts: mean|cos|={relora_mean_cos:.6f}, "
          f"max|cos|={relora_max_cos:.6f}")
    print(f"  Conv   base experts: mean|cos|={conv_mean_cos:.6f}, "
          f"max|cos|={conv_max_cos:.6f}")

    # ── Phase 5: Kill criteria evaluation ──────────────────────────────────
    print("\n--- Phase 5: Kill Criteria ---")

    # Kill 1: ReLoRA cos > 10x conventional cos (relative comparison)
    cos_ratio = relora_mean_cos / (conv_mean_cos + 1e-12)
    kill_cos = cos_ratio > 10.0
    print(f"  K1: cos ratio (relora/conv) = {cos_ratio:.4f} "
          f"(threshold: >10x) -> {'KILLED' if kill_cos else 'SURVIVES'}")
    print(f"      ReLoRA mean|cos| = {relora_mean_cos:.6f}, "
          f"Conv mean|cos| = {conv_mean_cos:.6f}")

    # Kill 2: expert quality < 50% of conventional
    # FIX(rev2, advisory 6): Report loss ratio directly.
    # loss_ratio = relora_loss / conv_loss. Values > 1 mean ReLoRA is worse.
    relora_mean_loss = sum(relora_expert_losses) / len(relora_expert_losses)
    conv_mean_loss = sum(conv_expert_losses) / len(conv_expert_losses)
    loss_ratio = relora_mean_loss / conv_mean_loss if conv_mean_loss > 0 else float('inf')
    # Kill if ReLoRA loss > 2x conventional (equivalent to quality < 50%)
    kill_quality = loss_ratio > 2.0
    print(f"  K2: loss ratio (relora/conv) = {loss_ratio:.4f} "
          f"(threshold: >2.0x) -> {'KILLED' if kill_quality else 'SURVIVES'}")
    print(f"      ReLoRA mean expert loss = {relora_mean_loss:.4f}, "
          f"Conv mean expert loss = {conv_mean_loss:.4f}")

    # Kill 3: ReLoRA base incoherent (val loss > 2x conventional)
    coherence_ratio = relora_val / conv_val if conv_val > 0 else float('inf')
    kill_coherence = coherence_ratio > 2.0
    print(f"  K3: coherence ratio = {coherence_ratio:.4f} -> "
          f"{'KILLED' if kill_coherence else 'SURVIVES'}")

    # Overall verdict
    if kill_cos or kill_quality or kill_coherence:
        verdict = "KILLED"
    elif cos_ratio < 2.0 and loss_ratio < 1.2:
        verdict = "SURVIVES"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n  VERDICT: {verdict}")

    results = ExperimentResults(
        relora_final_loss=relora_val,
        conventional_final_loss=conv_val,
        relora_merges=relora_result["merges_done"],
        relora_train_time=relora_result["elapsed_s"],
        conventional_train_time=conv_result["elapsed_s"],
        relora_mean_cos=relora_mean_cos,
        conventional_mean_cos=conv_mean_cos,
        relora_max_cos=relora_max_cos,
        conventional_max_cos=conv_max_cos,
        relora_cosines=[(i, j, c) for (i, j, c) in relora_cosines],
        conventional_cosines=[(i, j, c) for (i, j, c) in conv_cosines],
        relora_expert_losses=relora_expert_losses,
        conventional_expert_losses=conv_expert_losses,
        relora_mean_expert_loss=relora_mean_loss,
        conventional_mean_expert_loss=conv_mean_loss,
        relora_effective_rank=relora_mean_rank,
        conventional_effective_rank=conv_mean_rank,
        cos_ratio=cos_ratio,
        loss_ratio=loss_ratio,
        kill_cos_violated=kill_cos,
        kill_quality_violated=kill_quality,
        kill_coherence_violated=kill_coherence,
        verdict=verdict,
    )

    # FIX(rev2, bug 5): Save per-seed results with seed in filename,
    # not as generic results.json (which was confused with integration test data)
    output_path = os.path.join(os.path.dirname(__file__), f"results_seed_{seed}.json")
    with open(output_path, "w") as f:
        json.dump(asdict(results), f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def _bootstrap_ci(values: list, confidence: float = 0.95, n_boot: int = 5000) -> tuple:
    """Bootstrap confidence interval for the mean of `values`.

    Returns (mean, ci_low, ci_high).
    """
    rng = random.Random(42)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [rng.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1 - confidence) / 2
    lo = means[int(alpha * n_boot)]
    hi = means[int((1 - alpha) * n_boot)]
    return sum(values) / n, lo, hi


def _permutation_test(relora_cos_vals: list, conv_cos_vals: list,
                       n_perms: int = 10000) -> float:
    """Two-sample permutation test on mean cosine similarity.

    FIX(rev2, advisory 7): Tests whether the difference in mean cosine
    between ReLoRA and conventional is statistically significant.

    Returns p-value (probability of observing difference >= observed
    under null hypothesis that both samples come from same distribution).
    """
    rng = random.Random(42)
    observed_diff = (sum(relora_cos_vals) / len(relora_cos_vals) -
                     sum(conv_cos_vals) / len(conv_cos_vals))
    combined = relora_cos_vals + conv_cos_vals
    n_r = len(relora_cos_vals)
    count = 0
    for _ in range(n_perms):
        shuffled = list(combined)
        rng.shuffle(shuffled)
        perm_diff = (sum(shuffled[:n_r]) / n_r -
                     sum(shuffled[n_r:]) / len(conv_cos_vals))
        if perm_diff >= observed_diff:
            count += 1
    return count / n_perms


def run_multi_seed(seeds: list = None, **kwargs) -> dict:
    """Run experiment across multiple seeds, compute aggregate stats with CIs.

    FIX(rev2, bug 3): Reports 95% bootstrap CI on cos_ratio.
    FIX(rev2, advisory 7): Includes permutation test p-value.
    """
    if seeds is None:
        seeds = [42, 123, 7]

    per_seed = {}
    all_relora_cos = []
    all_conv_cos = []

    for s in seeds:
        print(f"\n{'='*72}")
        print(f"SEED {s}")
        print(f"{'='*72}")
        r = run_experiment(seed=s, **kwargs)
        per_seed[str(s)] = {
            "relora_cos": r.relora_mean_cos,
            "conv_cos": r.conventional_mean_cos,
            "cos_ratio": r.cos_ratio,
            "relora_loss": r.relora_mean_expert_loss,
            "conv_loss": r.conventional_mean_expert_loss,
            "loss_ratio": r.loss_ratio,
            "relora_base_loss": r.relora_final_loss,
            "conv_base_loss": r.conventional_final_loss,
            "relora_effective_rank": r.relora_effective_rank,
            "conv_effective_rank": r.conventional_effective_rank,
            "verdict": r.verdict,
        }
        # Collect all pairwise cosines across seeds for permutation test
        all_relora_cos.extend([abs(c) for (_, _, c) in r.relora_cosines])
        all_conv_cos.extend([abs(c) for (_, _, c) in r.conventional_cosines])

    # Aggregate statistics
    cos_ratios = [v["cos_ratio"] for v in per_seed.values()]
    loss_ratios = [v["loss_ratio"] for v in per_seed.values()]
    relora_cos_means = [v["relora_cos"] for v in per_seed.values()]
    conv_cos_means = [v["conv_cos"] for v in per_seed.values()]

    # Bootstrap CI on cos_ratio (FIX rev2, bug 3)
    mean_cr, ci_lo_cr, ci_hi_cr = _bootstrap_ci(cos_ratios)
    mean_lr, ci_lo_lr, ci_hi_lr = _bootstrap_ci(loss_ratios)

    # Permutation test (FIX rev2, advisory 7)
    perm_p = _permutation_test(all_relora_cos, all_conv_cos)

    # Overall verdict: SURVIVES only if all seeds survive or are inconclusive
    verdicts = [v["verdict"] for v in per_seed.values()]
    if any(v == "KILLED" for v in verdicts):
        overall = "KILLED"
    elif all(v == "SURVIVES" for v in verdicts):
        overall = "SURVIVES"
    else:
        overall = "INCONCLUSIVE"

    aggregate = {
        "seeds": per_seed,
        "aggregate": {
            "mean_cos_ratio": mean_cr,
            "cos_ratio_95ci": [ci_lo_cr, ci_hi_cr],
            "mean_loss_ratio": mean_lr,
            "loss_ratio_95ci": [ci_lo_lr, ci_hi_lr],
            "mean_relora_cos": sum(relora_cos_means) / len(relora_cos_means),
            "mean_conv_cos": sum(conv_cos_means) / len(conv_cos_means),
            "permutation_test_p": perm_p,
            "overall_verdict": overall,
        },
        "config": {
            "n_embd": kwargs.get("n_embd", 64),
            "n_head": kwargs.get("n_head", 4),
            "n_layer": kwargs.get("n_layer", 4),
            "lora_rank": kwargs.get("lora_rank", 8),
            "lora_alpha": kwargs.get("lora_alpha", 1.0),
            "total_pretrain_steps": kwargs.get("total_pretrain_steps", 1000),
            "merge_every": kwargs.get("merge_every", 200),
            "expert_train_steps": kwargs.get("expert_train_steps", 300),
            "n_experts": kwargs.get("n_experts", 4),
        },
    }

    output_path = os.path.join(os.path.dirname(__file__), "results_aggregate.json")
    with open(output_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n{'='*72}")
    print("AGGREGATE RESULTS")
    print(f"{'='*72}")
    print(f"  cos_ratio:  {mean_cr:.4f}  95% CI [{ci_lo_cr:.4f}, {ci_hi_cr:.4f}]")
    print(f"  loss_ratio: {mean_lr:.4f}  95% CI [{ci_lo_lr:.4f}, {ci_hi_lr:.4f}]")
    print(f"  permutation test p-value: {perm_p:.4f}")
    print(f"  overall verdict: {overall}")
    print(f"\nAggregate saved to {output_path}")

    return aggregate


if __name__ == "__main__":
    run_multi_seed()
