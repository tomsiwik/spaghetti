"""
Base-Free Composition: Can a model be composed entirely from adapters?

Hypothesis: A pretrained base model can be decomposed into
  W_pretrained = W_skeleton + Delta_base
where Delta_base is an adapter-format "base adapter," and domain LoRA
experts compose on top of (W_skeleton + Delta_base) with the same quality
as on the original W_pretrained.

If Delta_base can be low-rank (SVD truncated), the entire model -- base
included -- is expressible as composable adapters.

Design:
  1. Train a micro GPT conventionally (the "pretrained base")
  2. Fix a random skeleton W_skeleton (same init seed as step 1's init,
     but NOT trained -- frozen random weights)
  3. Compute Delta_base = W_pretrained - W_skeleton for each weight matrix
  4. Approximate Delta_base at ranks k in {full, 32, 16, 8, 4}
  5. Reconstruct: W_approx(k) = W_skeleton + SVD_k(Delta_base)
  6. Train N=4 LoRA experts on each condition
  7. Measure: base quality, expert quality, pairwise cosine similarity

Kill criteria:
  - Low-rank delta base expert quality < 50% of full base (loss ratio > 2.0)
  - Low-rank delta base produces incoherent text (base loss > 2x pretrained)
  - Expressing base as adapter costs > 10x a LoRA expert (time/memory)

Architecture: Self-contained PyTorch implementation (no MLX dependency).
Uses the micro names dataset (character-level).
"""

import math
import time
import random
import json
import os
import copy
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Data Loading ─────────────────────────────────────────────────────────────

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

    def decode(self, ids: list) -> str:
        return "".join(self.chars[i] if i != self.bos else "" for i in ids)


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
        """Return (inputs, targets) each of shape (B, T)."""
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


def domain_split(docs: list, method: str = "quintary") -> dict:
    """Split docs into domains by first character."""
    if method == "quintary":
        ranges = [("a", "e"), ("f", "j"), ("k", "o"), ("p", "t"), ("u", "z")]
        result = {}
        for lo, hi in ranges:
            key = f"{lo}_{hi}"
            result[key] = [d for d in docs if lo <= d[0].lower() <= hi]
        return result
    raise ValueError(f"Unknown method: {method}")


# ── PyTorch Micro GPT ────────────────────────────────────────────────────────


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
        # Register causal mask
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


# ── LoRA Layer ────────────────────────────────────────────────────────────────


class LoRALinear(nn.Module):
    """Linear layer with LoRA adapter. Base weight is frozen; A, B are trainable."""
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.rank = rank
        self.scale = alpha / rank
        # Kaiming init for A, zero for B (standard LoRA init)
        self.A = nn.Parameter(torch.randn(in_features, rank) * (2.0 / in_features) ** 0.5)
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.A @ self.B) * self.scale
        return base_out + lora_out

    def get_delta(self):
        """Return the LoRA delta: scale * A @ B, shape (in, out)."""
        return (self.A @ self.B * self.scale).detach()


class LoRAGPT(nn.Module):
    """Wraps a GPT with LoRA adapters on MLP layers (FFN-only)."""
    def __init__(self, base_gpt: GPT, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.base_gpt = base_gpt
        self.rank = rank
        self.alpha = alpha
        # Freeze all base parameters
        for p in base_gpt.parameters():
            p.requires_grad_(False)
        # Replace MLP linears with LoRA versions
        self.lora_layers = nn.ModuleList()
        for layer in base_gpt.layers:
            fc1_lora = LoRALinear(layer.mlp.fc1, rank, alpha)
            fc2_lora = LoRALinear(layer.mlp.fc2, rank, alpha)
            layer.mlp.fc1 = fc1_lora
            layer.mlp.fc2 = fc2_lora
            self.lora_layers.append(nn.ModuleDict({"fc1": fc1_lora, "fc2": fc2_lora}))

    def forward(self, tokens):
        return self.base_gpt(tokens)

    def get_all_deltas(self):
        """Return list of (layer_idx, fc_name, delta_tensor)."""
        deltas = []
        for i, lora_dict in enumerate(self.lora_layers):
            for name in ["fc1", "fc2"]:
                deltas.append((i, name, lora_dict[name].get_delta()))
        return deltas

    def lora_parameters(self):
        """Return only LoRA A/B parameters."""
        params = []
        for lora_dict in self.lora_layers:
            for name in ["fc1", "fc2"]:
                params.append(lora_dict[name].A)
                params.append(lora_dict[name].B)
        return params


# ── Delta Decomposition ──────────────────────────────────────────────────────


def compute_delta(pretrained_state: dict, skeleton_state: dict) -> dict:
    """Compute Delta = W_pretrained - W_skeleton for each parameter.

    Skips registered buffers (like causal masks) that contain non-finite values.
    """
    deltas = {}
    for key in pretrained_state:
        if pretrained_state[key].shape == skeleton_state[key].shape:
            # Skip buffers with non-finite values (e.g., causal mask with -inf)
            if not torch.isfinite(pretrained_state[key]).all():
                continue
            deltas[key] = pretrained_state[key] - skeleton_state[key]
    return deltas


def svd_truncate(delta: torch.Tensor, rank: int) -> torch.Tensor:
    """Truncate a 2D delta matrix to given rank via SVD.

    For 1D tensors (biases, norms), return as-is (no truncation).
    """
    if delta.dim() != 2:
        return delta.clone()
    U, S, Vt = torch.linalg.svd(delta, full_matrices=False)
    k = min(rank, len(S))
    return U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]


def reconstruct_with_delta(skeleton_state: dict, deltas: dict, rank: Optional[int] = None,
                           pretrained_state: Optional[dict] = None) -> dict:
    """Reconstruct weights: W = W_skeleton + SVD_k(Delta).

    If rank is None, use full delta (no truncation).
    For keys not in deltas (e.g., buffers with -inf), use pretrained_state
    if provided, else use skeleton_state.
    """
    reconstructed = {}
    for key in skeleton_state:
        if key in deltas:
            if rank is not None:
                reconstructed[key] = skeleton_state[key] + svd_truncate(deltas[key], rank)
            else:
                reconstructed[key] = skeleton_state[key] + deltas[key]
        elif pretrained_state is not None and key in pretrained_state:
            # Use pretrained value for buffers (masks etc.) not in deltas
            reconstructed[key] = pretrained_state[key].clone()
        else:
            reconstructed[key] = skeleton_state[key].clone()
    return reconstructed


def delta_reconstruction_error(deltas: dict, rank: int) -> dict:
    """Measure reconstruction error for each parameter at given rank."""
    errors = {}
    total_frob_orig = 0.0
    total_frob_error = 0.0
    for key, delta in deltas.items():
        if delta.dim() == 2:
            approx = svd_truncate(delta, rank)
            error = torch.norm(delta - approx).item()
            orig = torch.norm(delta).item()
            errors[key] = {
                "frobenius_error": error,
                "frobenius_original": orig,
                "relative_error": error / (orig + 1e-12),
            }
            total_frob_orig += orig ** 2
            total_frob_error += error ** 2
    errors["_total"] = {
        "rms_relative_error": (total_frob_error / (total_frob_orig + 1e-12)) ** 0.5,
    }
    return errors


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


# ── Training ─────────────────────────────────────────────────────────────────


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
    total_tokens = 0

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
        total_tokens += inputs.numel()

        if step % log_every == 0 or step == steps:
            elapsed = time.time() - t0
            tps = total_tokens / elapsed if elapsed > 0 else 0
            print(f"  [Pretrain] step {step:4d}/{steps} | loss {loss_val:.4f} | {tps:.0f} tok/s")

    elapsed = time.time() - t0
    return {"final_loss": losses[-1], "losses": losses, "elapsed_s": elapsed}


def evaluate_model(model: nn.Module, dataset: CharDataset, batch_size: int = 32,
                   n_batches: int = 10, device: str = "cpu") -> float:
    """Evaluate model on dataset, return mean NTP loss."""
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


def train_lora_expert(base_gpt: GPT, train_ds: CharDataset, val_ds: CharDataset,
                      rank: int = 8, alpha: float = 1.0, steps: int = 300,
                      batch_size: int = 32, lr: float = 3e-3, seed: int = 42,
                      device: str = "cpu") -> tuple:
    """Train a LoRA expert on frozen base, return (deltas, val_loss).

    Creates a deep copy of the base GPT, wraps with LoRA, trains only A/B.
    """
    base_copy = copy.deepcopy(base_gpt)
    lora_model = LoRAGPT(base_copy, rank=rank, alpha=alpha)
    lora_model.to(device)
    lora_model.train()

    rng = random.Random(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(lora_model.lora_parameters(), lr=lr)

    for step in range(1, steps + 1):
        inputs, targets = train_ds.get_batch(batch_size, rng, device)
        logits = lora_model(inputs)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    val_loss = evaluate_model(lora_model, val_ds, batch_size, device=device)
    deltas = lora_model.get_all_deltas()

    return deltas, val_loss


# ── Orthogonality Measurement ────────────────────────────────────────────────


def compute_pairwise_cosine(deltas_list: list) -> list:
    """Compute pairwise cosine similarity between expert delta sets.

    Each entry is a list of (layer_idx, fc_name, delta_matrix).
    Returns list of (expert_i, expert_j, cosine_similarity).
    """
    flat_vectors = []
    for deltas in deltas_list:
        parts = [d.reshape(-1) for (_, _, d) in deltas]
        flat = torch.cat(parts)
        flat_vectors.append(flat)

    results = []
    n = len(flat_vectors)
    for i in range(n):
        for j in range(i + 1, n):
            vi, vj = flat_vectors[i], flat_vectors[j]
            cos = (vi @ vj) / (torch.norm(vi) * torch.norm(vj) + 1e-12)
            results.append((i, j, cos.item()))
    return results


# ── Main Experiment ──────────────────────────────────────────────────────────


@dataclass
class ConditionResult:
    """Results for one experimental condition."""
    name: str
    base_val_loss: float
    expert_val_losses: list
    mean_expert_loss: float
    mean_cos: float
    max_cos: float
    cosines: list
    delta_reconstruction_error: Optional[float] = None
    delta_rank: Optional[int] = None


@dataclass
class ExperimentResults:
    """Complete results from the base-free composition test."""
    conditions: list  # list of ConditionResult dicts
    pretrained_effective_rank: float
    delta_effective_rank: float

    # Kill criteria
    kill_quality_violated: bool  # any condition loss > 2x pretrained
    kill_coherence_violated: bool  # any condition base incoherent
    kill_cost_violated: bool  # decomposition cost > 10x expert

    # Timing
    pretrain_time: float
    decomposition_time: float
    expert_train_time_avg: float

    verdict: str
    seed: int
    config: dict


def run_experiment(
    n_embd: int = 64,
    n_head: int = 4,
    n_layer: int = 4,
    block_size: int = 32,
    lora_rank: int = 8,
    lora_alpha: float = 1.0,
    total_pretrain_steps: int = 1000,
    expert_train_steps: int = 300,
    n_experts: int = 4,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 42,
    delta_ranks: list = None,
    device: str = "cpu",
) -> ExperimentResults:
    """Run the complete base-free composition experiment.

    Tests whether a pretrained model can be decomposed into
    W = W_skeleton + Delta, with Delta approximated at various ranks,
    and domain LoRA experts still compose successfully.

    Conditions:
      1. "pretrained" (control): full pretrained base + experts
      2. "delta_full": W_skeleton + full Delta + experts (sanity check)
      3. "delta_r{k}": W_skeleton + SVD_k(Delta) + experts (for each k)
      4. "skeleton_only": W_skeleton alone + experts (negative control)
    """
    if delta_ranks is None:
        delta_ranks = [32, 16, 8, 4]

    print("=" * 72)
    print("BASE-FREE COMPOSITION EXPERIMENT")
    print(f"Config: d={n_embd}, h={n_head}, L={n_layer}, r={lora_rank}")
    print(f"Pretrain: {total_pretrain_steps} steps")
    print(f"Experts: {n_experts} x {expert_train_steps} steps")
    print(f"Delta ranks to test: {delta_ranks}")
    print(f"Seed: {seed}")
    print("=" * 72)

    torch.manual_seed(seed)

    # ── Load data ──────────────────────────────────────────────────────
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vocab_size = tokenizer.vocab_size

    domains = domain_split(docs, method="quintary")
    domain_names = sorted(domains.keys())[:n_experts]
    print(f"\nDomains: {domain_names}")

    # Global train/val split for pretraining
    rng_split = random.Random(seed)
    docs_copy = list(docs)
    rng_split.shuffle(docs_copy)
    split_idx = int(len(docs_copy) * 0.9)
    pretrain_train_ds = CharDataset(docs_copy[:split_idx], tokenizer, block_size)
    pretrain_val_ds = CharDataset(docs_copy[split_idx:], tokenizer, block_size)

    # ── Phase 1: Pretrain base model ──────────────────────────────────
    print("\n--- Phase 1: Pretraining ---")
    torch.manual_seed(seed)
    pretrained_model = GPT(vocab_size, block_size, n_embd, n_head, n_layer)

    # Save skeleton state BEFORE training (same random init)
    skeleton_state = {k: v.clone() for k, v in pretrained_model.state_dict().items()}

    t0_pretrain = time.time()
    train_gpt(pretrained_model, pretrain_train_ds,
              steps=total_pretrain_steps, batch_size=batch_size,
              lr=lr, seed=seed, device=device)
    pretrain_time = time.time() - t0_pretrain

    pretrained_state = {k: v.clone() for k, v in pretrained_model.state_dict().items()}
    pretrained_val = evaluate_model(pretrained_model, pretrain_val_ds, batch_size, device=device)
    print(f"  Pretrained val loss: {pretrained_val:.4f}")

    # ── Phase 2: Delta decomposition ──────────────────────────────────
    print("\n--- Phase 2: Delta Decomposition ---")
    t0_decomp = time.time()
    deltas = compute_delta(pretrained_state, skeleton_state)
    decomp_time = time.time() - t0_decomp

    # Measure effective rank of pretrained weights and deltas
    pretrained_ranks = []
    delta_ranks_measured = []
    for key in deltas:
        if deltas[key].dim() == 2:
            pretrained_ranks.append(effective_rank(pretrained_state[key]))
            delta_ranks_measured.append(effective_rank(deltas[key]))

    mean_pretrained_rank = sum(pretrained_ranks) / len(pretrained_ranks) if pretrained_ranks else 0
    mean_delta_rank = sum(delta_ranks_measured) / len(delta_ranks_measured) if delta_ranks_measured else 0
    print(f"  Mean effective rank (pretrained): {mean_pretrained_rank:.1f}")
    print(f"  Mean effective rank (delta): {mean_delta_rank:.1f}")

    # Reconstruction errors at each rank
    for r in delta_ranks:
        errs = delta_reconstruction_error(deltas, r)
        rms = errs["_total"]["rms_relative_error"]
        print(f"  Delta SVD rank-{r:2d}: RMS relative error = {rms:.4f}")

    # ── Phase 3: Build condition models ───────────────────────────────
    print("\n--- Phase 3: Building Condition Models ---")

    conditions_to_test = []

    # Condition 1: pretrained (control)
    conditions_to_test.append(("pretrained", pretrained_state, None))

    # Condition 2: full delta reconstruction (sanity check)
    full_delta_state = reconstruct_with_delta(skeleton_state, deltas, rank=None,
                                                pretrained_state=pretrained_state)
    conditions_to_test.append(("delta_full", full_delta_state, None))

    # Condition 3: low-rank delta at each rank
    for r in delta_ranks:
        lr_state = reconstruct_with_delta(skeleton_state, deltas, rank=r,
                                           pretrained_state=pretrained_state)
        errs = delta_reconstruction_error(deltas, r)
        rms = errs["_total"]["rms_relative_error"]
        conditions_to_test.append((f"delta_r{r}", lr_state, rms))

    # Condition 4: skeleton only (negative control)
    conditions_to_test.append(("skeleton_only", skeleton_state, 1.0))

    # Evaluate base quality for each condition
    condition_base_losses = {}
    for cond_name, state, _ in conditions_to_test:
        model = GPT(vocab_size, block_size, n_embd, n_head, n_layer)
        model.load_state_dict(state)
        val_loss = evaluate_model(model, pretrain_val_ds, batch_size, device=device)
        condition_base_losses[cond_name] = val_loss
        print(f"  {cond_name:20s} base val loss: {val_loss:.4f}")

    # ── Phase 4: Train experts on each condition ──────────────────────
    print("\n--- Phase 4: Expert Training ---")

    # Prepare domain datasets (same splits for all conditions)
    domain_datasets = {}
    for i, domain in enumerate(domain_names):
        domain_docs = domains[domain]
        rng_domain = random.Random(seed + 1000 + i)
        domain_docs_shuffled = list(domain_docs)
        rng_domain.shuffle(domain_docs_shuffled)
        n_train = max(1, int(len(domain_docs_shuffled) * 0.8))
        train_ds = CharDataset(domain_docs_shuffled[:n_train], tokenizer, block_size)
        val_ds = CharDataset(
            domain_docs_shuffled[n_train:] if n_train < len(domain_docs_shuffled)
            else domain_docs_shuffled, tokenizer, block_size
        )
        domain_datasets[domain] = (train_ds, val_ds)

    all_condition_results = []
    expert_train_times = []

    for cond_name, state, recon_err in conditions_to_test:
        print(f"\n  Condition: {cond_name}")

        # Build base model from state
        base_model = GPT(vocab_size, block_size, n_embd, n_head, n_layer)
        base_model.load_state_dict(state)

        cond_deltas = []
        cond_losses = []
        t0_expert = time.time()

        for i, domain in enumerate(domain_names):
            train_ds, val_ds = domain_datasets[domain]
            expert_deltas, expert_val = train_lora_expert(
                base_model, train_ds, val_ds,
                rank=lora_rank, alpha=lora_alpha,
                steps=expert_train_steps, batch_size=batch_size,
                lr=lr, seed=seed + i, device=device,
            )
            cond_deltas.append(expert_deltas)
            cond_losses.append(expert_val)
            print(f"    Expert {i} ({domain}): val_loss={expert_val:.4f}")

        expert_time = time.time() - t0_expert
        expert_train_times.append(expert_time)

        # Pairwise cosine similarity
        cosines = compute_pairwise_cosine(cond_deltas)
        cos_vals = [abs(c) for (_, _, c) in cosines]
        mean_cos = sum(cos_vals) / len(cos_vals) if cos_vals else 0
        max_cos = max(cos_vals) if cos_vals else 0

        # Extract rank from name
        d_rank = None
        if cond_name.startswith("delta_r"):
            d_rank = int(cond_name.split("delta_r")[1])

        cond_result = ConditionResult(
            name=cond_name,
            base_val_loss=condition_base_losses[cond_name],
            expert_val_losses=cond_losses,
            mean_expert_loss=sum(cond_losses) / len(cond_losses),
            mean_cos=mean_cos,
            max_cos=max_cos,
            cosines=[(i, j, c) for (i, j, c) in cosines],
            delta_reconstruction_error=recon_err,
            delta_rank=d_rank,
        )
        all_condition_results.append(cond_result)

        print(f"    mean|cos|={mean_cos:.6f}, max|cos|={max_cos:.6f}")
        print(f"    mean expert loss={cond_result.mean_expert_loss:.4f}")

    # ── Phase 5: Kill Criteria ────────────────────────────────────────
    print("\n--- Phase 5: Kill Criteria ---")

    pretrained_result = next(r for r in all_condition_results if r.name == "pretrained")
    ref_loss = pretrained_result.mean_expert_loss
    ref_base_loss = pretrained_result.base_val_loss

    # K1: quality -- any low-rank condition loss > 2x pretrained
    kill_quality = False
    for r in all_condition_results:
        if r.name.startswith("delta_r") or r.name == "skeleton_only":
            ratio = r.mean_expert_loss / (ref_loss + 1e-12)
            exceeded = ratio > 2.0
            if exceeded:
                kill_quality = True
            print(f"  K1 [{r.name}]: loss_ratio = {ratio:.4f} "
                  f"(threshold: >2.0x) -> {'KILLED' if exceeded else 'SURVIVES'}")

    # K2: coherence -- any condition base loss > 2x pretrained
    kill_coherence = False
    for r in all_condition_results:
        if r.name != "pretrained" and r.name != "delta_full":
            ratio = r.base_val_loss / (ref_base_loss + 1e-12)
            exceeded = ratio > 2.0
            if exceeded and r.name != "skeleton_only":  # skeleton expected to be bad
                kill_coherence = True
            print(f"  K2 [{r.name}]: base_loss_ratio = {ratio:.4f} "
                  f"(threshold: >2.0x) -> {'KILLED' if exceeded else 'SURVIVES'}")

    # K3: cost -- decomposition time > 10x average expert training time
    avg_expert_time = sum(expert_train_times) / len(expert_train_times) if expert_train_times else 1.0
    cost_ratio = decomp_time / (avg_expert_time / n_experts + 1e-12)
    kill_cost = cost_ratio > 10.0
    print(f"  K3: decomposition time = {decomp_time:.3f}s, "
          f"avg expert time = {avg_expert_time / n_experts:.3f}s, "
          f"ratio = {cost_ratio:.2f}x -> {'KILLED' if kill_cost else 'SURVIVES'}")

    # Overall verdict
    if kill_quality or kill_coherence or kill_cost:
        verdict = "KILLED"
    else:
        # Check if at least one low-rank condition matches pretrained within 20%
        has_viable = False
        for r in all_condition_results:
            if r.name.startswith("delta_r"):
                loss_ratio = r.mean_expert_loss / (ref_loss + 1e-12)
                if loss_ratio < 1.2:
                    has_viable = True
        if has_viable:
            verdict = "SURVIVES"
        else:
            verdict = "INCONCLUSIVE"

    print(f"\n  VERDICT: {verdict}")

    config = {
        "n_embd": n_embd, "n_head": n_head, "n_layer": n_layer,
        "block_size": block_size, "lora_rank": lora_rank,
        "lora_alpha": lora_alpha, "total_pretrain_steps": total_pretrain_steps,
        "expert_train_steps": expert_train_steps, "n_experts": n_experts,
        "delta_ranks": delta_ranks,
    }

    results = ExperimentResults(
        conditions=[asdict(r) for r in all_condition_results],
        pretrained_effective_rank=mean_pretrained_rank,
        delta_effective_rank=mean_delta_rank,
        kill_quality_violated=kill_quality,
        kill_coherence_violated=kill_coherence,
        kill_cost_violated=kill_cost,
        pretrain_time=pretrain_time,
        decomposition_time=decomp_time,
        expert_train_time_avg=avg_expert_time,
        verdict=verdict,
        seed=seed,
        config=config,
    )

    output_path = os.path.join(os.path.dirname(__file__), f"results_seed_{seed}.json")
    with open(output_path, "w") as f:
        json.dump(asdict(results), f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def run_multi_seed(seeds: list = None, **kwargs) -> dict:
    """Run experiment across multiple seeds, compute aggregate stats."""
    if seeds is None:
        seeds = [42, 123, 7]

    per_seed = {}
    all_results = []

    for s in seeds:
        print(f"\n{'='*72}")
        print(f"SEED {s}")
        print(f"{'='*72}")
        r = run_experiment(seed=s, **kwargs)
        all_results.append(r)
        per_seed[str(s)] = {
            "verdict": r.verdict,
            "conditions": r.conditions,
        }

    # Aggregate: for each condition, average the metrics across seeds
    condition_names = [c["name"] for c in all_results[0].conditions]
    aggregate_conditions = {}

    for cond_name in condition_names:
        losses = []
        cosines = []
        base_losses = []
        for r in all_results:
            cond = next(c for c in r.conditions if c["name"] == cond_name)
            losses.append(cond["mean_expert_loss"])
            cosines.append(cond["mean_cos"])
            base_losses.append(cond["base_val_loss"])

        aggregate_conditions[cond_name] = {
            "mean_expert_loss": sum(losses) / len(losses),
            "std_expert_loss": (sum((l - sum(losses)/len(losses))**2 for l in losses) / len(losses)) ** 0.5,
            "mean_cos": sum(cosines) / len(cosines),
            "mean_base_loss": sum(base_losses) / len(base_losses),
        }

    # Compute loss ratios relative to pretrained
    ref = aggregate_conditions["pretrained"]
    for cond_name, agg in aggregate_conditions.items():
        agg["loss_ratio"] = agg["mean_expert_loss"] / (ref["mean_expert_loss"] + 1e-12)
        agg["base_loss_ratio"] = agg["mean_base_loss"] / (ref["mean_base_loss"] + 1e-12)

    verdicts = [r.verdict for r in all_results]
    if any(v == "KILLED" for v in verdicts):
        overall = "KILLED"
    elif all(v == "SURVIVES" for v in verdicts):
        overall = "SURVIVES"
    else:
        overall = "INCONCLUSIVE"

    aggregate = {
        "seeds": per_seed,
        "aggregate_conditions": aggregate_conditions,
        "overall_verdict": overall,
        "config": all_results[0].config,
    }

    output_path = os.path.join(os.path.dirname(__file__), "results_aggregate.json")
    with open(output_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n{'='*72}")
    print("AGGREGATE RESULTS")
    print(f"{'='*72}")
    print(f"\n{'Condition':20s} | {'Loss Ratio':>10s} | {'Base Loss Ratio':>15s} | {'mean|cos|':>10s}")
    print("-" * 65)
    for cond_name in condition_names:
        agg = aggregate_conditions[cond_name]
        print(f"{cond_name:20s} | {agg['loss_ratio']:10.4f} | {agg['base_loss_ratio']:15.4f} | {agg['mean_cos']:10.6f}")
    print(f"\nOverall verdict: {overall}")
    print(f"Aggregate saved to {output_path}")

    return aggregate


if __name__ == "__main__":
    run_multi_seed()
