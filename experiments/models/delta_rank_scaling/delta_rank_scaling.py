"""
Delta Rank Scaling: Does the effective rank ratio (r_needed/d) decrease
at larger model dimension d?

REVISION v2 (2026-03-11): Addresses 5 fixes from adversarial review:
  1. Convergence control: train all sizes to same validation loss
  2. FFN+Attn-only primary metric (exclude embeddings)
  3. Bootstrap confidence intervals on power law exponent
  4. Accept K1 kill honestly (no retroactive reinterpretation)
  5. Multi-checkpoint rho measurement at 25/50/75/100% of training

Background:
  At d=64, the delta (pretrained - skeleton) has effective rank ~40,
  giving r_needed/d = 0.625. If this ratio holds at macro scale (d=4096),
  the base-as-adapter concept is impractical (rank 2560 needed).

  This experiment measures the ratio at d=64, d=128, d=256 to establish
  the scaling trend. If ratio decreases, macro extrapolation is favorable.

Design:
  1. For each d in {64, 128, 256}:
     a. Initialize a micro GPT with random weights (the "skeleton")
     b. Train to a TARGET validation loss (convergence control)
     c. Compute Delta = W_pretrained - W_skeleton for each weight matrix
     d. Measure effective rank of Delta (Roy & Vetterli, 2007)
     e. Compute ratio = effective_rank / min(d_out, d_in)
     f. Record rho at 25/50/75/100% of training (multi-checkpoint)
  2. Compare ratios across dimensions (FFN+Attn only, excluding embeddings)
  3. Fit a power law: ratio(d) = a * d^b with bootstrap CI
     If b < 0, the ratio decreases (favorable for macro)

Kill criteria:
  K1: Shannon r_eff/d ratio stays above 0.5 at d=128 AND d=256
  K2: larger models show higher effective rank ratio than smaller ones

Architecture: Pure PyTorch, CPU-only. No MLX.
Reuses GPT/training from base_free_composition.
"""

import math
import time
import random
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Reuse infrastructure from base_free_composition ──────────────────────
# We inline the model/data code to avoid import dependency issues,
# but the architecture is identical to base_free_composition.

DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "input.txt")


def load_names(path: str = DATA_PATH) -> list:
    """Load the names dataset."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        import urllib.request
        urllib.request.urlretrieve(DATA_URL, path)
    return [line.strip() for line in open(path) if line.strip()]


class CharTokenizer:
    """Char-level tokenizer with BOS token."""
    def __init__(self, docs: list):
        self.chars = sorted(set("".join(docs)))
        self.bos = len(self.chars)
        self.vocab_size = len(self.chars) + 1
        self._c2i = {c: i for i, c in enumerate(self.chars)}

    def encode(self, s: str) -> list:
        return [self._c2i[c] for c in s]


class CharDataset:
    """Pack names into fixed-length sequences for NTP."""
    def __init__(self, docs: list, tokenizer: CharTokenizer, block_size: int = 32):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.sequences = []
        for doc in docs:
            tokens = [tokenizer.bos] + tokenizer.encode(doc) + [tokenizer.bos]
            self.sequences.append(tokens)

    def __len__(self):
        return len(self.sequences)

    def get_batch(self, batch_size: int, rng: random.Random = None, device: str = "cpu"):
        rng = rng or random
        seqs = rng.choices(self.sequences, k=batch_size)
        inputs, targets = [], []
        for seq in seqs:
            n = min(self.block_size, len(seq) - 1)
            inp = seq[:n] + [0] * (self.block_size - n)
            tgt = seq[1:n + 1] + [0] * (self.block_size - n)
            inputs.append(inp)
            targets.append(tgt)
        return (torch.tensor(inputs, device=device),
                torch.tensor(targets, device=device))


# ── PyTorch Micro GPT ────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        ms = torch.mean(x * x, dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int = 32):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)
        mask = torch.triu(torch.full((block_size, block_size), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn + self.mask[:T, :T]
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int = 32):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = nn.ModuleList([Block(n_embd, n_head, block_size)
                                     for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


# ── Effective Rank ───────────────────────────────────────────────────────

def effective_rank(matrix: torch.Tensor) -> float:
    """Effective rank (Roy & Vetterli, 2007): exp(H(p)) where p_i = sigma_i/sum."""
    if matrix.dim() != 2:
        return 0.0
    S = torch.linalg.svdvals(matrix)
    S = S[S > 1e-12]
    if len(S) == 0:
        return 0.0
    total = S.sum()
    probs = S / total
    entropy = -(probs * torch.log(probs)).sum().item()
    return math.exp(entropy)


def rank_at_threshold(matrix: torch.Tensor, threshold: float = 0.99) -> int:
    """Number of singular values needed to capture `threshold` fraction of total energy."""
    if matrix.dim() != 2:
        return 0
    S = torch.linalg.svdvals(matrix)
    S = S[S > 1e-12]
    if len(S) == 0:
        return 0
    total_energy = (S ** 2).sum()
    cumulative = torch.cumsum(S ** 2, dim=0)
    k = int((cumulative / total_energy >= threshold).nonzero(as_tuple=True)[0][0].item()) + 1
    return k


def singular_value_spectrum(matrix: torch.Tensor) -> list:
    """Return normalized singular value spectrum."""
    if matrix.dim() != 2:
        return []
    S = torch.linalg.svdvals(matrix)
    S = S[S > 1e-12]
    if len(S) == 0:
        return []
    return (S / S[0]).tolist()


# ── Training ─────────────────────────────────────────────────────────────

def train_gpt(model: GPT, dataset: CharDataset, steps: int = 1000,
              batch_size: int = 32, lr: float = 3e-3, seed: int = 42,
              log_every: int = 200, device: str = "cpu") -> dict:
    """Standard pretraining of GPT model."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    t0 = time.time()

    for step in range(1, steps + 1):
        inputs, targets = dataset.get_batch(batch_size, rng, device)
        logits = model(inputs)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if step % log_every == 0 or step == steps:
            elapsed = time.time() - t0
            print(f"    [d={model.wte.embedding_dim}] step {step:4d}/{steps} | "
                  f"loss {loss_val:.4f} | {elapsed:.1f}s")

    return {"final_loss": losses[-1], "losses": losses, "elapsed_s": time.time() - t0}


def evaluate_model(model: nn.Module, dataset: CharDataset, batch_size: int = 32,
                   n_batches: int = 10, device: str = "cpu") -> float:
    """Evaluate model, return mean NTP loss."""
    rng = random.Random(999)
    model.eval()
    model.to(device)
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            inputs, targets = dataset.get_batch(batch_size, rng, device)
            logits = model(inputs)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
            total_loss += loss.item()
    return total_loss / n_batches


def train_to_target_loss(model: GPT, dataset: CharDataset, val_ds: CharDataset,
                         target_val_loss: float, max_steps: int = 20000,
                         batch_size: int = 32, lr: float = 3e-3, seed: int = 42,
                         checkpoint_fracs: list = None, skeleton_state: dict = None,
                         device: str = "cpu") -> dict:
    """Train until validation loss reaches target, with multi-checkpoint SVD analysis.

    Args:
        target_val_loss: Stop when val loss <= this value.
        max_steps: Safety cap to prevent infinite training.
        checkpoint_fracs: Fractions of training at which to snapshot rho (e.g., [0.25, 0.5, 0.75, 1.0]).
                         These are interpreted as fractions of *actual* steps taken.
                         Since we don't know total steps a priori, we evaluate at
                         fixed step intervals and record when frac thresholds are crossed.
        skeleton_state: Original init weights for delta computation at checkpoints.

    Returns:
        dict with final_loss, losses, elapsed_s, steps_taken, val_loss,
        and checkpoint_analyses (list of {frac, step, analyses}).
    """
    if checkpoint_fracs is None:
        checkpoint_fracs = [0.25, 0.5, 0.75, 1.0]

    rng = random.Random(seed)
    torch.manual_seed(seed)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    t0 = time.time()
    eval_every = 100  # Check val loss every 100 steps
    checkpoint_analyses = []
    reached_target = False
    steps_taken = 0

    for step in range(1, max_steps + 1):
        inputs, targets = dataset.get_batch(batch_size, rng, device)
        logits = model(inputs)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        steps_taken = step

        # Check convergence periodically
        if step % eval_every == 0:
            val_loss = evaluate_model(model, val_ds, batch_size, device=device)
            model.train()  # Switch back to train mode
            elapsed = time.time() - t0
            print(f"    [d={model.wte.embedding_dim}] step {step:5d} | "
                  f"train {loss_val:.4f} | val {val_loss:.4f} | target {target_val_loss:.4f} | {elapsed:.1f}s")
            if val_loss <= target_val_loss:
                reached_target = True
                break

    # Now we know total steps; compute checkpoint analyses retroactively
    # Re-train from scratch with checkpoints at the right fractions
    # Actually, for efficiency, we just do the multi-checkpoint analysis
    # by re-running with known step count and capturing snapshots.
    # But that doubles training time. Instead, we'll do a simpler approach:
    # record the final analysis and note the steps taken, then in a second
    # pass measure rho at fractional checkpoints.

    final_val_loss = evaluate_model(model, val_ds, batch_size, device=device)
    elapsed = time.time() - t0

    if not reached_target:
        print(f"    WARNING: d={model.wte.embedding_dim} did not reach target {target_val_loss:.4f} "
              f"in {max_steps} steps (final val: {final_val_loss:.4f})")

    print(f"    d={model.wte.embedding_dim}: converged in {steps_taken} steps, "
          f"val_loss={final_val_loss:.4f}, target={target_val_loss:.4f}")

    return {
        "final_loss": losses[-1] if losses else float("inf"),
        "losses": losses,
        "elapsed_s": elapsed,
        "steps_taken": steps_taken,
        "val_loss": final_val_loss,
        "reached_target": reached_target,
    }


def train_with_checkpoints(model: GPT, dataset: CharDataset, val_ds: CharDataset,
                           total_steps: int, batch_size: int = 32, lr: float = 3e-3,
                           seed: int = 42, skeleton_state: dict = None,
                           checkpoint_fracs: list = None, device: str = "cpu") -> dict:
    """Train for exactly total_steps, capturing SVD analysis at checkpoint fractions.

    Returns dict with training info and checkpoint_rho: list of (frac, step, rho_ffn_attn, rho_all).
    """
    if checkpoint_fracs is None:
        checkpoint_fracs = [0.25, 0.5, 0.75, 1.0]

    checkpoint_steps = sorted(set(max(1, int(total_steps * f)) for f in checkpoint_fracs))

    rng = random.Random(seed)
    torch.manual_seed(seed)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    t0 = time.time()
    checkpoint_rho = []

    for step in range(1, total_steps + 1):
        inputs, targets = dataset.get_batch(batch_size, rng, device)
        logits = model(inputs)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step in checkpoint_steps:
            # Snapshot and analyze
            model.eval()
            current_state = {k: v.clone() for k, v in model.state_dict().items()}
            analyses = analyze_deltas(current_state, skeleton_state)

            # Compute FFN+Attn only ratio
            ffn_attn_ratios = [a.ratio for a in analyses
                               if "mlp" in a.key or "fc" in a.key
                               or "wq" in a.key or "wk" in a.key
                               or "wv" in a.key or "wo" in a.key]
            all_ratios = [a.ratio for a in analyses]

            ffn_attn_mean = sum(ffn_attn_ratios) / len(ffn_attn_ratios) if ffn_attn_ratios else 0
            all_mean = sum(all_ratios) / len(all_ratios) if all_ratios else 0

            frac = step / total_steps
            checkpoint_rho.append({
                "frac": round(frac, 3),
                "step": step,
                "rho_ffn_attn": round(ffn_attn_mean, 4),
                "rho_all": round(all_mean, 4),
            })

            elapsed = time.time() - t0
            print(f"    [d={model.wte.embedding_dim}] checkpoint {frac:.0%} (step {step}) | "
                  f"rho_ffn_attn={ffn_attn_mean:.4f} | rho_all={all_mean:.4f} | {elapsed:.1f}s")
            model.train()

    val_loss = evaluate_model(model, val_ds, batch_size, device=device)
    elapsed = time.time() - t0

    return {
        "final_loss": losses[-1] if losses else float("inf"),
        "losses": losses,
        "elapsed_s": elapsed,
        "steps_taken": total_steps,
        "val_loss": val_loss,
        "checkpoint_rho": checkpoint_rho,
    }


# ── Delta Analysis ───────────────────────────────────────────────────────

@dataclass
class WeightAnalysis:
    """Analysis of one weight matrix's delta."""
    key: str
    shape: list
    min_dim: int
    effective_rank: float
    ratio: float  # effective_rank / min_dim
    rank_99: int   # rank needed for 99% energy
    rank_99_ratio: float  # rank_99 / min_dim
    rank_95: int
    rank_95_ratio: float
    frobenius_norm: float
    spectrum_top10: list  # top 10 normalized singular values


@dataclass
class DimensionResult:
    """Results for one model dimension d."""
    d: int
    n_head: int
    n_layer: int
    n_params: int
    pretrain_loss: float
    val_loss: float
    pretrain_time: float

    # Per-weight analysis
    weight_analyses: list

    # Aggregates across all 2D weight matrices
    mean_eff_rank: float
    mean_ratio: float  # All weights (legacy, secondary metric)
    median_ratio: float
    mean_rank_99_ratio: float
    mean_rank_95_ratio: float

    # By weight type
    ffn_mean_ratio: float
    attn_mean_ratio: float
    emb_mean_ratio: float

    # PRIMARY METRIC: FFN+Attention only (excludes embeddings which don't scale with d)
    ffn_attn_mean_ratio: float
    ffn_attn_r99_ratio: float
    ffn_attn_r95_ratio: float

    # Convergence info
    steps_taken: int
    convergence_controlled: bool

    # Multi-checkpoint rho trajectory
    checkpoint_rho: list  # list of {frac, step, rho_ffn_attn, rho_all}

    seed: int


@dataclass
class ExperimentResults:
    """Complete results from the rank scaling experiment."""
    dimension_results: list  # list of DimensionResult dicts
    scaling_fit: dict  # power law fit parameters

    # Kill criteria
    k1_violated: bool  # ratio > 0.5 at BOTH d=128 AND d=256
    k2_violated: bool  # larger d shows higher ratio

    verdict: str
    seeds: list
    config: dict


def analyze_deltas(pretrained_state: dict, skeleton_state: dict) -> list:
    """Compute and analyze deltas for all 2D weight matrices."""
    analyses = []
    for key in pretrained_state:
        w_p = pretrained_state[key]
        w_s = skeleton_state[key]
        if w_p.shape != w_s.shape:
            continue
        if w_p.dim() != 2:
            continue
        # Skip buffers with non-finite values
        if not torch.isfinite(w_p).all():
            continue

        delta = w_p - w_s
        min_dim = min(delta.shape)
        eff_rank = effective_rank(delta)
        r99 = rank_at_threshold(delta, 0.99)
        r95 = rank_at_threshold(delta, 0.95)
        spectrum = singular_value_spectrum(delta)

        analyses.append(WeightAnalysis(
            key=key,
            shape=list(delta.shape),
            min_dim=min_dim,
            effective_rank=eff_rank,
            ratio=eff_rank / min_dim if min_dim > 0 else 0.0,
            rank_99=r99,
            rank_99_ratio=r99 / min_dim if min_dim > 0 else 0.0,
            rank_95=r95,
            rank_95_ratio=r95 / min_dim if min_dim > 0 else 0.0,
            frobenius_norm=torch.norm(delta).item(),
            spectrum_top10=spectrum[:10],
        ))
    return analyses


def run_single_dimension(d: int, n_head: int, n_layer: int, block_size: int,
                          vocab_size: int, dataset: CharDataset, val_ds: CharDataset,
                          pretrain_steps: int, batch_size: int, lr: float,
                          seed: int, device: str = "cpu",
                          target_val_loss: float = None,
                          max_steps: int = 20000) -> DimensionResult:
    """Train a model at dimension d and analyze its delta effective rank.

    If target_val_loss is provided, trains until that loss is reached (convergence control).
    Otherwise, trains for pretrain_steps (original behavior).
    Multi-checkpoint SVD analysis is always performed.
    """
    print(f"\n  === Dimension d={d}, heads={n_head}, layers={n_layer} ===")

    torch.manual_seed(seed)
    model = GPT(vocab_size, block_size, d, n_head, n_layer)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,}")

    # Save skeleton (random init)
    skeleton_state = {k: v.clone() for k, v in model.state_dict().items()}

    convergence_controlled = target_val_loss is not None

    if convergence_controlled:
        # Phase 1: Train to target val loss to determine steps needed
        print(f"    Convergence-controlled training to val_loss <= {target_val_loss:.4f}")
        train_result = train_to_target_loss(
            model, dataset, val_ds,
            target_val_loss=target_val_loss,
            max_steps=max_steps,
            batch_size=batch_size, lr=lr, seed=seed,
            skeleton_state=skeleton_state, device=device,
        )
        actual_steps = train_result["steps_taken"]
        val_loss = train_result["val_loss"]
        pretrain_time = train_result["elapsed_s"]

        # Phase 2: Re-train from scratch with checkpoints at known fractions
        print(f"    Re-training with multi-checkpoint analysis ({actual_steps} steps)...")
        torch.manual_seed(seed)
        model2 = GPT(vocab_size, block_size, d, n_head, n_layer)
        skeleton_state2 = {k: v.clone() for k, v in model2.state_dict().items()}
        ckpt_result = train_with_checkpoints(
            model2, dataset, val_ds,
            total_steps=actual_steps,
            batch_size=batch_size, lr=lr, seed=seed,
            skeleton_state=skeleton_state2,
            checkpoint_fracs=[0.25, 0.5, 0.75, 1.0],
            device=device,
        )
        checkpoint_rho = ckpt_result["checkpoint_rho"]
        pretrain_time += ckpt_result["elapsed_s"]  # Total includes both passes

        # Use model2 for final analysis (identical to model due to same seed)
        pretrained_state = {k: v.clone() for k, v in model2.state_dict().items()}
        skeleton_state = skeleton_state2
    else:
        # Original behavior: fixed steps with multi-checkpoint analysis
        print(f"    Fixed-step training: {pretrain_steps} steps")
        t0 = time.time()
        ckpt_result = train_with_checkpoints(
            model, dataset, val_ds,
            total_steps=pretrain_steps,
            batch_size=batch_size, lr=lr, seed=seed,
            skeleton_state=skeleton_state,
            checkpoint_fracs=[0.25, 0.5, 0.75, 1.0],
            device=device,
        )
        pretrain_time = time.time() - t0
        val_loss = ckpt_result["val_loss"]
        checkpoint_rho = ckpt_result["checkpoint_rho"]
        actual_steps = pretrain_steps
        pretrained_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Compute deltas and analyze
    analyses = analyze_deltas(pretrained_state, skeleton_state)

    # Aggregate -- all weights (legacy)
    ratios = [a.ratio for a in analyses]
    r99_ratios = [a.rank_99_ratio for a in analyses]
    r95_ratios = [a.rank_95_ratio for a in analyses]
    eff_ranks = [a.effective_rank for a in analyses]

    # By type
    ffn_analyses = [a for a in analyses if "mlp" in a.key or "fc" in a.key]
    attn_analyses = [a for a in analyses if "wq" in a.key or "wk" in a.key
                     or "wv" in a.key or "wo" in a.key]
    emb_analyses = [a for a in analyses if "wte" in a.key or "wpe" in a.key
                    or "lm_head" in a.key]

    ffn_ratios = [a.ratio for a in ffn_analyses]
    attn_ratios = [a.ratio for a in attn_analyses]
    emb_ratios = [a.ratio for a in emb_analyses]

    # PRIMARY METRIC: FFN + Attention only (Fix #2)
    ffn_attn_analyses = ffn_analyses + attn_analyses
    ffn_attn_ratios = [a.ratio for a in ffn_attn_analyses]
    ffn_attn_r99 = [a.rank_99_ratio for a in ffn_attn_analyses]
    ffn_attn_r95 = [a.rank_95_ratio for a in ffn_attn_analyses]

    mean_ratio = sum(ratios) / len(ratios) if ratios else 0
    sorted_ratios = sorted(ratios)
    median_ratio = sorted_ratios[len(sorted_ratios) // 2] if sorted_ratios else 0

    result = DimensionResult(
        d=d,
        n_head=n_head,
        n_layer=n_layer,
        n_params=n_params,
        pretrain_loss=ckpt_result["final_loss"],
        val_loss=val_loss,
        pretrain_time=pretrain_time,
        weight_analyses=[asdict(a) for a in analyses],
        mean_eff_rank=sum(eff_ranks) / len(eff_ranks) if eff_ranks else 0,
        mean_ratio=mean_ratio,
        median_ratio=median_ratio,
        mean_rank_99_ratio=sum(r99_ratios) / len(r99_ratios) if r99_ratios else 0,
        mean_rank_95_ratio=sum(r95_ratios) / len(r95_ratios) if r95_ratios else 0,
        ffn_mean_ratio=sum(ffn_ratios) / len(ffn_ratios) if ffn_ratios else 0,
        attn_mean_ratio=sum(attn_ratios) / len(attn_ratios) if attn_ratios else 0,
        emb_mean_ratio=sum(emb_ratios) / len(emb_ratios) if emb_ratios else 0,
        ffn_attn_mean_ratio=sum(ffn_attn_ratios) / len(ffn_attn_ratios) if ffn_attn_ratios else 0,
        ffn_attn_r99_ratio=sum(ffn_attn_r99) / len(ffn_attn_r99) if ffn_attn_r99 else 0,
        ffn_attn_r95_ratio=sum(ffn_attn_r95) / len(ffn_attn_r95) if ffn_attn_r95 else 0,
        steps_taken=actual_steps,
        convergence_controlled=convergence_controlled,
        checkpoint_rho=checkpoint_rho,
        seed=seed,
    )

    print(f"    Val loss: {val_loss:.4f}")
    print(f"    Steps taken: {actual_steps}")
    print(f"    Mean effective rank: {result.mean_eff_rank:.1f} / {d}")
    print(f"    [PRIMARY] FFN+Attn ratio: {result.ffn_attn_mean_ratio:.4f}")
    print(f"    [legacy]  All-weights ratio: {mean_ratio:.4f}")
    print(f"    FFN ratio: {result.ffn_mean_ratio:.4f}")
    print(f"    Attn ratio: {result.attn_mean_ratio:.4f}")
    print(f"    Emb ratio: {result.emb_mean_ratio:.4f} (excluded from primary)")
    print(f"    FFN+Attn r99 ratio: {result.ffn_attn_r99_ratio:.4f}")
    print(f"    FFN+Attn r95 ratio: {result.ffn_attn_r95_ratio:.4f}")

    return result


def _fit_log_log(log_d, log_r):
    """Fit log(ratio) = log(a) + b * log(d). Returns (a, b, r_squared)."""
    n = len(log_d)
    mean_x = log_d.mean()
    mean_y = log_r.mean()
    ss_xx = ((log_d - mean_x) ** 2).sum()
    ss_xy = ((log_d - mean_x) * (log_r - mean_y)).sum()

    b = float(ss_xy / (ss_xx + 1e-12))
    log_a = float(mean_y - b * mean_x)
    a = float(math.exp(log_a))

    y_pred = log_a + b * log_d
    ss_res = ((log_r - y_pred) ** 2).sum()
    ss_tot = ((log_r - mean_y) ** 2).sum()
    r_squared = float(1.0 - ss_res / (ss_tot + 1e-12))

    return a, b, r_squared


def fit_power_law(dimensions: list, ratios: list,
                  per_seed_ratios: dict = None,
                  n_bootstrap: int = 10000) -> dict:
    """Fit ratio = a * d^b using log-linear regression.

    Args:
        dimensions: List of d values.
        ratios: Mean ratio at each d.
        per_seed_ratios: dict {str(d): [ratio_seed1, ratio_seed2, ...]} for bootstrap CI.
        n_bootstrap: Number of bootstrap resamples.

    Returns dict with a, b, r_squared, confidence intervals, and extrapolations.
    """
    import numpy as np
    log_d = np.log(np.array(dimensions, dtype=float))
    log_r = np.log(np.array(ratios, dtype=float))

    a, b, r_squared = _fit_log_log(log_d, log_r)

    # Bootstrap CI on exponent b (Fix #3)
    ci_b = None
    ci_a = None
    if per_seed_ratios is not None:
        rng_boot = np.random.RandomState(42)
        b_samples = []
        a_samples = []
        seed_lists = [per_seed_ratios[str(d)] for d in dimensions]
        n_seeds = len(seed_lists[0])

        for _ in range(n_bootstrap):
            # Resample seeds with replacement
            idx = rng_boot.choice(n_seeds, size=n_seeds, replace=True)
            boot_ratios = []
            for d_idx in range(len(dimensions)):
                boot_mean = np.mean([seed_lists[d_idx][i] for i in idx])
                boot_ratios.append(boot_mean)
            boot_log_r = np.log(np.array(boot_ratios, dtype=float))
            a_b, b_b, _ = _fit_log_log(log_d, boot_log_r)
            b_samples.append(b_b)
            a_samples.append(a_b)

        b_samples = np.array(b_samples)
        a_samples = np.array(a_samples)
        ci_b = [round(float(np.percentile(b_samples, 2.5)), 4),
                round(float(np.percentile(b_samples, 97.5)), 4)]
        ci_a = [round(float(np.percentile(a_samples, 2.5)), 4),
                round(float(np.percentile(a_samples, 97.5)), 4)]

    # Extrapolations with error bars
    extrapolations = {}
    for d_target in [512, 896, 3584, 4096, 8192]:
        predicted_ratio = a * (d_target ** b)
        predicted_rank = predicted_ratio * d_target
        entry = {
            "predicted_ratio": round(predicted_ratio, 3),  # 3 sig figs, not 4 (Fix #3)
            "predicted_rank": round(predicted_rank, 0),
        }
        # Add CI-based range if available
        if ci_b is not None and ci_a is not None:
            lo_ratio = min(ci_a[0] * (d_target ** ci_b[0]),
                          ci_a[0] * (d_target ** ci_b[1]),
                          ci_a[1] * (d_target ** ci_b[0]),
                          ci_a[1] * (d_target ** ci_b[1]))
            hi_ratio = max(ci_a[0] * (d_target ** ci_b[0]),
                          ci_a[0] * (d_target ** ci_b[1]),
                          ci_a[1] * (d_target ** ci_b[0]),
                          ci_a[1] * (d_target ** ci_b[1]))
            entry["ratio_ci_95"] = [round(lo_ratio, 3), round(hi_ratio, 3)]
            entry["rank_ci_95"] = [round(lo_ratio * d_target, 0), round(hi_ratio * d_target, 0)]
        extrapolations[str(d_target)] = entry

    result = {
        "a": round(a, 4),
        "b": round(b, 4),
        "r_squared": round(r_squared, 4),
        "extrapolations": extrapolations,
    }
    if ci_b is not None:
        result["b_ci_95"] = ci_b
        result["a_ci_95"] = ci_a
        result["n_bootstrap"] = n_bootstrap

    return result


def run_experiment(
    dimensions: list = None,
    n_layer: int = 4,
    block_size: int = 32,
    pretrain_steps: int = 1000,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 42,
    device: str = "cpu",
    convergence_control: bool = True,
    target_val_loss: float = None,
) -> ExperimentResults:
    """Run the delta rank scaling experiment across multiple dimensions.

    If convergence_control=True (default in v2), first trains d=64 to get a
    target val loss, then trains all dimensions to that same target.
    """
    if dimensions is None:
        dimensions = [64, 128, 256]

    # Head count scales with dimension to keep head_dim reasonable
    head_configs = {64: 4, 128: 8, 256: 8}

    # Step configs for non-convergence-controlled mode
    step_configs = {64: pretrain_steps, 128: pretrain_steps * 2, 256: pretrain_steps * 3}

    print("=" * 72)
    print("DELTA RANK SCALING EXPERIMENT (v2 -- revised)")
    print(f"Dimensions: {dimensions}")
    print(f"Convergence control: {convergence_control}")
    print(f"Seed: {seed}")
    print("=" * 72)

    # Load data
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vocab_size = tokenizer.vocab_size

    rng_split = random.Random(seed)
    docs_copy = list(docs)
    rng_split.shuffle(docs_copy)
    split_idx = int(len(docs_copy) * 0.9)
    train_ds = CharDataset(docs_copy[:split_idx], tokenizer, block_size)
    val_ds = CharDataset(docs_copy[split_idx:], tokenizer, block_size)

    if convergence_control and target_val_loss is None:
        # Phase 0: Train d=64 with default steps to establish target val loss
        print("\n  --- Phase 0: Establishing target val loss from d=64 ---")
        d0 = dimensions[0]  # smallest dimension
        n_head0 = head_configs.get(d0, max(1, d0 // 16))
        steps0 = step_configs.get(d0, pretrain_steps)
        torch.manual_seed(seed)
        pilot = GPT(vocab_size, block_size, d0, n_head0, n_layer)
        pilot_skel = {k: v.clone() for k, v in pilot.state_dict().items()}
        train_gpt(pilot, train_ds, steps=steps0, batch_size=batch_size, lr=lr,
                  seed=seed, log_every=max(1, steps0 // 3), device=device)
        target_val_loss = evaluate_model(pilot, val_ds, batch_size, device=device)
        print(f"\n  Target val loss (from d={d0}, {steps0} steps): {target_val_loss:.4f}")
        print(f"  All dimensions will train to this target.")
        del pilot

    # Run each dimension
    dim_results = []
    for d in dimensions:
        n_head = head_configs.get(d, max(1, d // 16))
        steps = step_configs.get(d, pretrain_steps)
        result = run_single_dimension(
            d=d, n_head=n_head, n_layer=n_layer, block_size=block_size,
            vocab_size=vocab_size, dataset=train_ds, val_ds=val_ds,
            pretrain_steps=steps, batch_size=batch_size, lr=lr,
            seed=seed, device=device,
            target_val_loss=target_val_loss if convergence_control else None,
            max_steps=20000,
        )
        dim_results.append(result)

    # Fit power law on PRIMARY metric: FFN+Attn ratio (Fix #2)
    dims_list = [r.d for r in dim_results]
    primary_ratios = [r.ffn_attn_mean_ratio for r in dim_results]
    fit = fit_power_law(dims_list, primary_ratios)

    # Also fit legacy all-weights ratio
    legacy_ratios = [r.mean_ratio for r in dim_results]
    fit_legacy = fit_power_law(dims_list, legacy_ratios)
    fit["legacy_all_weights_fit"] = fit_legacy

    # Kill criteria evaluation (Fix #4: use Shannon r_eff/d, accept K1 honestly)
    print("\n" + "=" * 72)
    print("KILL CRITERIA EVALUATION")
    print("=" * 72)

    # K1: Shannon r_eff/d ratio > 0.5 at BOTH d=128 AND d=256
    # This is evaluated on ALL weights (original pre-registered criterion)
    result_128 = next((r for r in dim_results if r.d == 128), None)
    result_256 = next((r for r in dim_results if r.d == 256), None)
    k1_violated = False
    if result_128 and result_256:
        k1_violated = result_128.mean_ratio > 0.5 and result_256.mean_ratio > 0.5
        print(f"\n  K1: Shannon r_eff/d ratio > 0.5 at d=128 AND d=256?")
        print(f"    d=128: ratio = {result_128.mean_ratio:.4f} {'> 0.5 KILLED' if result_128.mean_ratio > 0.5 else '<= 0.5 SURVIVES'}")
        print(f"    d=256: ratio = {result_256.mean_ratio:.4f} {'> 0.5 KILLED' if result_256.mean_ratio > 0.5 else '<= 0.5 SURVIVES'}")
        print(f"    K1 {'KILLED' if k1_violated else 'SURVIVES'}")
        if k1_violated:
            print(f"    NOTE: K1 is ACCEPTED as killed. Shannon effective rank > 0.5 at")
            print(f"    both dimensions. This was the pre-registered criterion.")

    # K2: larger d shows higher ratio (checked on primary FFN+Attn metric)
    k2_violated = False
    if len(dim_results) >= 2:
        for i in range(len(dim_results) - 1):
            if dim_results[i + 1].ffn_attn_mean_ratio > dim_results[i].ffn_attn_mean_ratio:
                k2_violated = True
        print(f"\n  K2: FFN+Attn ratio increases with d?")
        for r in dim_results:
            print(f"    d={r.d}: FFN+Attn ratio = {r.ffn_attn_mean_ratio:.4f}")
        print(f"    Trend: {'INCREASING (BAD)' if k2_violated else 'DECREASING (GOOD)'}")
        print(f"    K2 {'KILLED' if k2_violated else 'SURVIVES'}")

    # Verdict
    if k1_violated and k2_violated:
        verdict = "KILLED"
    elif k1_violated or k2_violated:
        verdict = "WEAK_KILL"
    else:
        verdict = "SURVIVES"

    print(f"\n  Power law fit (FFN+Attn): ratio(d) = {fit['a']:.4f} * d^({fit['b']:.4f})")
    print(f"  R-squared: {fit['r_squared']:.4f}")
    if "b_ci_95" in fit:
        print(f"  Exponent 95% CI: [{fit['b_ci_95'][0]}, {fit['b_ci_95'][1]}]")
    print(f"\n  Extrapolations (with CI where available):")
    for d_str, ex in fit["extrapolations"].items():
        line = f"    d={d_str}: ratio = {ex['predicted_ratio']:.3f}, rank = {ex['predicted_rank']:.0f}"
        if "ratio_ci_95" in ex:
            line += f" (CI: [{ex['ratio_ci_95'][0]:.3f}, {ex['ratio_ci_95'][1]:.3f}])"
        print(line)

    # Multi-checkpoint summary (Fix #5)
    print(f"\n  Multi-checkpoint rho trajectory:")
    for r in dim_results:
        if r.checkpoint_rho:
            print(f"    d={r.d}:")
            for ckpt in r.checkpoint_rho:
                print(f"      {ckpt['frac']:.0%}: rho_ffn_attn={ckpt['rho_ffn_attn']:.4f}")

    # Convergence summary
    print(f"\n  Convergence summary:")
    for r in dim_results:
        print(f"    d={r.d}: val_loss={r.val_loss:.4f}, steps={r.steps_taken}, "
              f"controlled={r.convergence_controlled}")

    print(f"\n  VERDICT: {verdict}")

    config = {
        "dimensions": dimensions,
        "n_layer": n_layer,
        "block_size": block_size,
        "pretrain_steps": pretrain_steps,
        "batch_size": batch_size,
        "lr": lr,
        "convergence_control": convergence_control,
        "target_val_loss": target_val_loss,
        "version": "v2_revised",
    }

    results = ExperimentResults(
        dimension_results=[asdict(r) for r in dim_results],
        scaling_fit=fit,
        k1_violated=k1_violated,
        k2_violated=k2_violated,
        verdict=verdict,
        seeds=[seed],
        config=config,
    )

    return results


def run_multi_seed(seeds: list = None, **kwargs) -> dict:
    """Run experiment across multiple seeds, aggregate results."""
    if seeds is None:
        seeds = [42, 123, 7]

    all_results = []
    per_seed = {}

    for s in seeds:
        print(f"\n{'#' * 72}")
        print(f"# SEED {s}")
        print(f"{'#' * 72}")
        r = run_experiment(seed=s, **kwargs)
        all_results.append(r)
        per_seed[str(s)] = {
            "verdict": r.verdict,
            "dimension_results": r.dimension_results,
            "scaling_fit": r.scaling_fit,
        }

    # Aggregate across seeds
    dimensions = [dr["d"] for dr in all_results[0].dimension_results]
    aggregate = {}

    for d in dimensions:
        ratios = []
        ffn_attn_ratios = []
        ffn_attn_r99 = []
        ffn_attn_r95 = []
        r99_ratios = []
        r95_ratios = []
        ffn_ratios = []
        attn_ratios = []
        emb_ratios = []
        val_losses = []
        steps_list = []
        checkpoint_rhos = []  # For multi-checkpoint aggregation

        for r in all_results:
            dr = next(x for x in r.dimension_results if x["d"] == d)
            ratios.append(dr["mean_ratio"])
            ffn_attn_ratios.append(dr["ffn_attn_mean_ratio"])
            ffn_attn_r99.append(dr["ffn_attn_r99_ratio"])
            ffn_attn_r95.append(dr["ffn_attn_r95_ratio"])
            r99_ratios.append(dr["mean_rank_99_ratio"])
            r95_ratios.append(dr["mean_rank_95_ratio"])
            ffn_ratios.append(dr["ffn_mean_ratio"])
            attn_ratios.append(dr["attn_mean_ratio"])
            emb_ratios.append(dr["emb_mean_ratio"])
            val_losses.append(dr["val_loss"])
            steps_list.append(dr["steps_taken"])
            if dr.get("checkpoint_rho"):
                checkpoint_rhos.append(dr["checkpoint_rho"])

        mean_ratio = sum(ratios) / len(ratios)
        std_ratio = (sum((r - mean_ratio) ** 2 for r in ratios) / len(ratios)) ** 0.5

        mean_ffn_attn = sum(ffn_attn_ratios) / len(ffn_attn_ratios)
        std_ffn_attn = (sum((r - mean_ffn_attn) ** 2 for r in ffn_attn_ratios) / len(ffn_attn_ratios)) ** 0.5

        # Aggregate checkpoint trajectories
        agg_checkpoints = []
        if checkpoint_rhos:
            n_checkpoints = len(checkpoint_rhos[0])
            for ci in range(n_checkpoints):
                frac_vals = [cp[ci]["rho_ffn_attn"] for cp in checkpoint_rhos if ci < len(cp)]
                if frac_vals:
                    agg_checkpoints.append({
                        "frac": checkpoint_rhos[0][ci]["frac"],
                        "mean_rho_ffn_attn": round(sum(frac_vals) / len(frac_vals), 4),
                        "std_rho_ffn_attn": round((sum((v - sum(frac_vals)/len(frac_vals))**2 for v in frac_vals) / len(frac_vals))**0.5, 4),
                    })

        aggregate[str(d)] = {
            # PRIMARY metric (Fix #2)
            "ffn_attn_mean_ratio": round(mean_ffn_attn, 4),
            "ffn_attn_std_ratio": round(std_ffn_attn, 4),
            "ffn_attn_r99_ratio": round(sum(ffn_attn_r99) / len(ffn_attn_r99), 4),
            "ffn_attn_r95_ratio": round(sum(ffn_attn_r95) / len(ffn_attn_r95), 4),
            "ffn_attn_all_ratios": [round(r, 4) for r in ffn_attn_ratios],
            # Legacy (all weights including embeddings)
            "mean_ratio": round(mean_ratio, 4),
            "std_ratio": round(std_ratio, 4),
            "mean_r99_ratio": round(sum(r99_ratios) / len(r99_ratios), 4),
            "mean_r95_ratio": round(sum(r95_ratios) / len(r95_ratios), 4),
            # By type
            "ffn_ratio": round(sum(ffn_ratios) / len(ffn_ratios), 4),
            "attn_ratio": round(sum(attn_ratios) / len(attn_ratios), 4),
            "emb_ratio": round(sum(emb_ratios) / len(emb_ratios), 4),
            "all_ratios": [round(r, 4) for r in ratios],
            # Convergence info
            "mean_val_loss": round(sum(val_losses) / len(val_losses), 4),
            "mean_steps": round(sum(steps_list) / len(steps_list)),
            # Multi-checkpoint trajectory (Fix #5)
            "checkpoint_trajectory": agg_checkpoints,
        }

    # Fit power law on PRIMARY metric: FFN+Attn ratio with bootstrap CI (Fix #3)
    agg_dims = [int(d) for d in aggregate.keys()]

    # Per-seed ratio data for bootstrap
    per_seed_ffn_attn = {str(d): aggregate[str(d)]["ffn_attn_all_ratios"] for d in agg_dims}

    agg_ffn_attn = [aggregate[str(d)]["ffn_attn_mean_ratio"] for d in agg_dims]
    agg_fit = fit_power_law(agg_dims, agg_ffn_attn,
                            per_seed_ratios=per_seed_ffn_attn,
                            n_bootstrap=10000)

    # Also fit legacy all-weights (no bootstrap, informational only)
    agg_ratios = [aggregate[str(d)]["mean_ratio"] for d in agg_dims]
    agg_fit_legacy = fit_power_law(agg_dims, agg_ratios)
    agg_fit["legacy_all_weights_fit"] = agg_fit_legacy

    # Aggregate kill criteria
    all_k1 = [r.k1_violated for r in all_results]
    all_k2 = [r.k2_violated for r in all_results]

    k1_any = any(all_k1)
    k2_any = any(all_k2)

    if k1_any and k2_any:
        overall = "KILLED"
    elif k1_any or k2_any:
        overall = "WEAK_KILL"
    else:
        overall = "SURVIVES"

    # Print aggregate summary
    print("\n" + "=" * 72)
    print("AGGREGATE RESULTS (across seeds) -- v2 REVISED")
    print("=" * 72)

    print(f"\n  PRIMARY METRIC: FFN+Attention mean ratio (excludes embeddings)")
    print(f"  {'d':>6} | {'FFN+Attn':>10} | {'std':>8} | {'r99':>8} | {'r95':>8} | {'val_loss':>8} | {'steps':>6}")
    print(f"  {'-' * 70}")
    for d in agg_dims:
        a = aggregate[str(d)]
        print(f"  {d:6d} | {a['ffn_attn_mean_ratio']:10.4f} | {a['ffn_attn_std_ratio']:8.4f} | "
              f"{a['ffn_attn_r99_ratio']:8.4f} | {a['ffn_attn_r95_ratio']:8.4f} | "
              f"{a['mean_val_loss']:8.4f} | {a['mean_steps']:6.0f}")

    print(f"\n  SECONDARY (legacy, all weights including embeddings):")
    print(f"  {'d':>6} | {'all_ratio':>10} | {'std':>8} | {'FFN':>8} | {'Attn':>8} | {'Emb':>8}")
    print(f"  {'-' * 60}")
    for d in agg_dims:
        a = aggregate[str(d)]
        print(f"  {d:6d} | {a['mean_ratio']:10.4f} | {a['std_ratio']:8.4f} | "
              f"{a['ffn_ratio']:8.4f} | {a['attn_ratio']:8.4f} | {a['emb_ratio']:8.4f}")

    print(f"\n  Power law fit (FFN+Attn): ratio(d) = {agg_fit['a']:.4f} * d^({agg_fit['b']:.4f})")
    print(f"  R-squared: {agg_fit['r_squared']:.4f}")
    if "b_ci_95" in agg_fit:
        print(f"  Exponent b 95% CI: [{agg_fit['b_ci_95'][0]}, {agg_fit['b_ci_95'][1]}]")
    print(f"\n  Extrapolations (with 95% CI):")
    for d_str, ex in agg_fit["extrapolations"].items():
        line = f"    d={d_str}: ratio = {ex['predicted_ratio']:.3f}"
        if "ratio_ci_95" in ex:
            line += f" [{ex['ratio_ci_95'][0]:.3f}, {ex['ratio_ci_95'][1]:.3f}]"
        line += f", rank = {ex['predicted_rank']:.0f}"
        if "rank_ci_95" in ex:
            line += f" [{ex['rank_ci_95'][0]:.0f}, {ex['rank_ci_95'][1]:.0f}]"
        print(line)

    # Multi-checkpoint trajectory (Fix #5)
    print(f"\n  Multi-checkpoint rho trajectory (FFN+Attn, mean across seeds):")
    for d in agg_dims:
        traj = aggregate[str(d)].get("checkpoint_trajectory", [])
        if traj:
            parts = [f"{cp['frac']:.0%}:{cp['mean_rho_ffn_attn']:.4f}" for cp in traj]
            print(f"    d={d}: {' -> '.join(parts)}")

    # K1 kill acceptance (Fix #4)
    print(f"\n  KILL CRITERIA:")
    print(f"    K1 (Shannon r_eff/d > 0.5 at d=128,256): {'KILLED' if k1_any else 'SURVIVES'} across {'all' if all(all_k1) else 'some'} seeds")
    if k1_any:
        print(f"       ACCEPTED. The pre-registered Shannon criterion is killed.")
    print(f"    K2 (FFN+Attn ratio increases with d): {'KILLED' if k2_any else 'SURVIVES'}")

    print(f"\n  Overall verdict: {overall}")

    result = {
        "version": "v2_revised",
        "seeds": per_seed,
        "aggregate": aggregate,
        "aggregate_fit": agg_fit,
        "k1_violated": k1_any,
        "k2_violated": k2_any,
        "overall_verdict": overall,
        "config": all_results[0].config,
    }

    output_path = os.path.join(os.path.dirname(__file__), "results_aggregate.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return result


if __name__ == "__main__":
    run_multi_seed()
