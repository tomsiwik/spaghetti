#!/usr/bin/env python3
"""ReLoRA From-Scratch Composition: Train base from random init via ReLoRA,
then compose domain experts and compare against conventional training.

Runs ON RunPod (A5000, 24GB VRAM).

Design:
  Phase 1a: Train conventional base (GPT-2 125M) from scratch, 10K steps
  Phase 1b: Train ReLoRA base from scratch: 2K warmup + 8K ReLoRA steps (K=16 cycles)
  Phase 2:  Train 5 domain experts (rank-16 LoRA) on each base
  Phase 3:  Compare pairwise cosine similarity and expert quality

Kill criteria:
  K1: cos_ratio (ReLoRA/conv) > 5x
  K2: loss_ratio (ReLoRA/conv expert loss) > 1.20
  K3: base perplexity ratio > 1.20

Usage (on RunPod):
    cd /workspace/llm
    python macro/relora_from_scratch/run_relora_from_scratch.py

Expected runtime: ~2-3 hours on A5000
Estimated cost: ~$0.40-$0.50
"""

import argparse
import copy
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset

# ── Configuration ─────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent.parent.parent  # /workspace/llm
OUTPUT_DIR = REPO_ROOT / "results" / "relora_from_scratch"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# Model config (GPT-2 small, ~124M params)
D_MODEL = 768
D_FF = 3072  # 4 * d_model
N_LAYERS = 12
N_HEADS = 12
D_HEAD = D_MODEL // N_HEADS  # 64
VOCAB_SIZE = 50257
MAX_SEQ_LEN = 512
DROPOUT = 0.0  # No dropout for pretraining

# Smoke test mode: run just a few steps to validate no crashes
IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

# Training config
BATCH_SIZE = 8
GRAD_ACCUM = 4 if not IS_SMOKE else 1  # effective batch = 32
TOTAL_STEPS = 10000 if not IS_SMOKE else 5
WARMUP_STEPS_CONV = 500 if not IS_SMOKE else 2
WARMUP_STEPS_RELORA_INIT = 2000 if not IS_SMOKE else 3
LR_BASE = 6e-4  # Standard for GPT-2 small
WEIGHT_DECAY = 0.1

# ReLoRA config
RELORA_RANK = 128  # Rank per cycle
RELORA_ALPHA = 128  # scaling = 1.0
RELORA_CYCLES = 16 if not IS_SMOKE else 1  # K merge-restart cycles
RELORA_STEPS_PER_CYCLE = (TOTAL_STEPS - WARMUP_STEPS_RELORA_INIT) // RELORA_CYCLES
RELORA_LR = LR_BASE * 2  # 2x per Lialin et al.
RELORA_REWARMUP = 50 if not IS_SMOKE else 1

# Expert config
EXPERT_RANK = 16
EXPERT_ALPHA = 16
EXPERT_STEPS = 500 if not IS_SMOKE else 3
EXPERT_LR = 2e-4
EXPERT_BATCH_SIZE = 4
EXPERT_GRAD_ACCUM = 2 if not IS_SMOKE else 1
N_EXPERTS = 5 if not IS_SMOKE else 2

# Domains for expert training (simple topic separation via C4 keywords)
DOMAINS = ["code", "science", "medical", "legal", "stories"]
DOMAIN_KEYWORDS = {
    "code": ["def ", "import ", "class ", "function", "return ", "var ", "const "],
    "science": ["experiment", "hypothesis", "molecule", "equation", "quantum", "theory"],
    "medical": ["patient", "diagnosis", "treatment", "clinical", "symptoms", "therapy"],
    "legal": ["court", "plaintiff", "defendant", "statute", "jurisdiction", "contract"],
    "stories": ["once upon", "she said", "he walked", "the old", "kingdom", "adventure"],
}

SEED = 42

# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── GPT-2 Model ──────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads, max_seq_len, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # Causal mask
        self.register_buffer("bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


class MLP(nn.Module):
    def __init__(self, d, d_ff, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads, d_ff, max_seq_len, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d)
        self.attn = CausalSelfAttention(d, n_heads, max_seq_len, dropout)
        self.ln2 = RMSNorm(d)
        self.mlp = MLP(d, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, vocab_size, d, n_layers, n_heads, d_ff, max_seq_len, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_seq_len, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d, n_heads, d_ff, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.tok_emb(idx)
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── LoRA Module ───────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """LoRA wrapper around a frozen linear layer."""
    def __init__(self, base_linear: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.scaling = alpha / rank
        d_in = base_linear.in_features
        d_out = base_linear.out_features
        device = base_linear.weight.device
        dtype = base_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(d_in, rank, device=device, dtype=dtype) * (1.0 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out, device=device, dtype=dtype))

    def forward(self, x):
        base_out = self.base(x)
        # Ensure LoRA params on same device as input (defensive against device drift)
        if self.lora_A.device != x.device:
            self.lora_A.data = self.lora_A.data.to(x.device)
            self.lora_B.data = self.lora_B.data.to(x.device)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

    def merge_and_reset(self):
        """Merge LoRA into base and reset. For ReLoRA."""
        with torch.no_grad():
            # Compute in float32 for precision, then cast back
            delta = (self.lora_A.float() @ self.lora_B.float()) * self.scaling  # (d_in, d_out)
            self.base.weight.data += delta.T.to(self.base.weight.dtype)  # Linear stores (d_out, d_in)
            # Reset on same device/dtype
            nn.init.normal_(self.lora_A, std=1.0 / math.sqrt(self.rank))
            nn.init.zeros_(self.lora_B)

    def get_delta_flat(self):
        """Return flattened delta = scaling * (B @ A)^T = scaling * A @ B reshaped."""
        with torch.no_grad():
            delta = (self.lora_A @ self.lora_B) * self.scaling  # (d_in, d_out)
            return delta.T.reshape(-1).float().cpu().numpy()  # (d_out, d_in) flattened


def apply_lora(model: GPT2, rank: int, alpha: int, freeze_base: bool = True):
    """Replace linear layers with LoRA wrappers. Returns list of target module names."""
    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False

    target_names = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    replaced = []

    for i, block in enumerate(model.blocks):
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            old = getattr(block.attn, name)
            lora = LoRALinear(old, rank, alpha)
            setattr(block.attn, name, lora)
            replaced.append(f"blocks.{i}.attn.{name}")

        for name in ["fc1", "fc2"]:
            old = getattr(block.mlp, name)
            lora = LoRALinear(old, rank, alpha)
            setattr(block.mlp, name, lora)
            replaced.append(f"blocks.{i}.mlp.{name}")

    return replaced


def get_lora_modules(model: GPT2):
    """Get all LoRA modules from a model."""
    modules = []
    for block in model.blocks:
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            m = getattr(block.attn, name)
            if isinstance(m, LoRALinear):
                modules.append(m)
        for name in ["fc1", "fc2"]:
            m = getattr(block.mlp, name)
            if isinstance(m, LoRALinear):
                modules.append(m)
    return modules


def extract_expert_delta(model: GPT2) -> np.ndarray:
    """Extract flattened expert delta vector from LoRA model."""
    parts = []
    for m in get_lora_modules(model):
        parts.append(m.get_delta_flat())
    return np.concatenate(parts)


def strip_lora(model: GPT2):
    """Remove LoRA wrappers, restoring base linear layers."""
    for block in model.blocks:
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            m = getattr(block.attn, name)
            if isinstance(m, LoRALinear):
                setattr(block.attn, name, m.base)
        for name in ["fc1", "fc2"]:
            m = getattr(block.mlp, name)
            if isinstance(m, LoRALinear):
                setattr(block.mlp, name, m.base)
    # Unfreeze
    for p in model.parameters():
        p.requires_grad = True


# ── Dataset ───────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    """Memory-mapped token dataset from a flat file of uint16 tokens."""
    def __init__(self, tokens: np.ndarray, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def prepare_data(tokenizer_name="gpt2"):
    """Download and tokenize a subset of C4 for pretraining.

    We use ~50M tokens (enough for 10K steps at batch=32, seq=512).
    """
    data_path = OUTPUT_DIR / "train_tokens.npy"
    val_path = OUTPUT_DIR / "val_tokens.npy"

    if data_path.exists() and val_path.exists():
        log("Loading cached tokenized data")
        train_tokens = np.load(str(data_path))
        val_tokens = np.load(str(val_path))
        return train_tokens, val_tokens

    log("Preparing training data from C4...")
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Need at least: 10K steps * 32 batch * 512 seq = ~164M tokens
    # But dataloader recycles, so 50M unique tokens is sufficient
    # Use C4 streaming (no trust_remote_code needed for standard datasets)
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

    all_tokens = []
    target_tokens = 50_000_000  # 50M tokens (recycled via DataLoader shuffle)
    n_docs = 0

    for example in ds:
        tokens = tokenizer.encode(example["text"])
        all_tokens.extend(tokens)
        n_docs += 1
        if len(all_tokens) >= target_tokens:
            break
        if n_docs % 10000 == 0:
            log(f"  Tokenized {n_docs} docs, {len(all_tokens)/1e6:.1f}M tokens")

    log(f"  Total: {n_docs} docs, {len(all_tokens)/1e6:.1f}M tokens")

    tokens_arr = np.array(all_tokens, dtype=np.uint16)

    # Split: 99% train, 1% val
    split_idx = int(len(tokens_arr) * 0.99)
    train_tokens = tokens_arr[:split_idx]
    val_tokens = tokens_arr[split_idx:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(data_path), train_tokens)
    np.save(str(val_path), val_tokens)

    log(f"  Saved {len(train_tokens)/1e6:.1f}M train, {len(val_tokens)/1e6:.1f}M val tokens")
    return train_tokens, val_tokens


def prepare_domain_data(train_tokens: np.ndarray, tokenizer_name="gpt2"):
    """Create domain-specific subsets by filtering C4 by keywords.

    Since we have pre-tokenized data, we re-stream C4 and filter by domain
    keywords to create small domain-specific datasets for expert training.
    """
    domain_paths = {d: OUTPUT_DIR / f"domain_{d}_tokens.npy" for d in DOMAINS}

    if all(p.exists() for p in domain_paths.values()):
        log("Loading cached domain data")
        return {d: np.load(str(p)) for d, p in domain_paths.items()}

    log("Preparing domain-specific data from C4...")
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

    domain_tokens = {d: [] for d in DOMAINS}
    target_per_domain = 2_000_000  # 2M tokens per domain (sufficient for 500 expert steps)
    n_docs = 0
    done = set()

    for example in ds:
        text = example["text"].lower()
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if domain in done:
                continue
            if any(kw in text for kw in keywords):
                tokens = tokenizer.encode(example["text"])
                domain_tokens[domain].extend(tokens)
                if len(domain_tokens[domain]) >= target_per_domain:
                    done.add(domain)

        n_docs += 1
        if len(done) == len(DOMAINS):
            break
        if n_docs % 50000 == 0:
            sizes = {d: len(t)/1e6 for d, t in domain_tokens.items()}
            log(f"  {n_docs} docs scanned. Sizes: {sizes}")

    for domain in DOMAINS:
        tokens = np.array(domain_tokens[domain][:2_000_000], dtype=np.uint16)
        np.save(str(domain_paths[domain]), tokens)
        log(f"  {domain}: {len(tokens)/1e6:.1f}M tokens")

    return {d: np.load(str(domain_paths[d])) for d in DOMAINS}


# ── Training Functions ────────────────────────────────────────────────────────

def create_model():
    """Create a fresh GPT-2 model."""
    model = GPT2(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, D_FF, MAX_SEQ_LEN, DROPOUT)
    log(f"  Model params: {model.count_params():,}")
    return model.to(DEVICE)


def evaluate(model, val_loader, max_batches=50):
    """Evaluate model on validation set. Returns mean loss."""
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float('inf')


def get_lr(step, total_steps, warmup_steps, max_lr, min_lr_ratio=0.1):
    """Cosine schedule with warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * (min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)))


def get_relora_lr(step_in_cycle, cycle_steps, warmup_steps, max_lr, min_lr_ratio=0.1):
    """Cosine schedule with restart warmup for ReLoRA cycles."""
    if step_in_cycle < warmup_steps:
        return max_lr * (step_in_cycle + 1) / warmup_steps
    progress = (step_in_cycle - warmup_steps) / max(1, cycle_steps - warmup_steps)
    return max_lr * (min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)))


# ── Phase 1a: Conventional Training ──────────────────────────────────────────

def train_conventional(train_tokens, val_tokens) -> dict:
    """Train GPT-2 from scratch using conventional full-rank training."""
    ckpt_path = OUTPUT_DIR / "conventional_base.pt"
    meta_path = OUTPUT_DIR / "conventional_meta.json"

    if ckpt_path.exists() and meta_path.exists():
        log("Conventional base already trained, loading")
        with open(meta_path) as f:
            return json.load(f)

    log("=== Phase 1a: Conventional Training ===")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = create_model()
    train_ds = TokenDataset(train_tokens, MAX_SEQ_LEN)
    val_ds = TokenDataset(val_tokens, MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_BASE, weight_decay=WEIGHT_DECAY,
                                   betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    train_iter = iter(train_loader)
    losses = []
    t0 = time.time()

    for step in range(TOTAL_STEPS):
        # LR schedule
        lr = get_lr(step, TOTAL_STEPS, WARMUP_STEPS_CONV, LR_BASE)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(GRAD_ACCUM):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                _, loss = model(x, y)
            loss_scaled = loss / GRAD_ACCUM
            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item() / GRAD_ACCUM

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(accum_loss)

        if step % 500 == 0 or step == TOTAL_STEPS - 1:
            val_loss = evaluate(model, val_loader)
            elapsed = time.time() - t0
            tokens_per_sec = (step + 1) * BATCH_SIZE * GRAD_ACCUM * MAX_SEQ_LEN / elapsed
            log(f"  step {step}/{TOTAL_STEPS}: train={accum_loss:.4f}, val={val_loss:.4f}, "
                f"lr={lr:.2e}, tok/s={tokens_per_sec:.0f}")

    # Final eval
    final_val = evaluate(model, val_loader, max_batches=200)
    elapsed = time.time() - t0
    log(f"  Conventional training done: val_loss={final_val:.4f}, time={elapsed/60:.1f}min")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

    meta = {
        "condition": "conventional",
        "final_val_loss": final_val,
        "final_train_loss": float(np.mean(losses[-100:])),
        "elapsed_s": elapsed,
        "total_steps": TOTAL_STEPS,
        "total_tokens": TOTAL_STEPS * BATCH_SIZE * GRAD_ACCUM * MAX_SEQ_LEN,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return meta


# ── Phase 1b: ReLoRA Training ────────────────────────────────────────────────

def train_relora(train_tokens, val_tokens) -> dict:
    """Train GPT-2 from scratch using ReLoRA: warmup + iterative merge-restart."""
    ckpt_path = OUTPUT_DIR / "relora_base.pt"
    meta_path = OUTPUT_DIR / "relora_meta.json"

    if ckpt_path.exists() and meta_path.exists():
        log("ReLoRA base already trained, loading")
        with open(meta_path) as f:
            return json.load(f)

    log("=== Phase 1b: ReLoRA Training ===")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = create_model()
    train_ds = TokenDataset(train_tokens, MAX_SEQ_LEN)
    val_ds = TokenDataset(val_tokens, MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

    # Phase A: Full-rank warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_BASE, weight_decay=WEIGHT_DECAY,
                                   betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    train_iter = iter(train_loader)
    losses = []
    t0 = time.time()
    global_step = 0

    log(f"  Phase A: Full-rank warmup ({WARMUP_STEPS_RELORA_INIT} steps)")
    for step in range(WARMUP_STEPS_RELORA_INIT):
        lr = get_lr(step, WARMUP_STEPS_RELORA_INIT, min(200, WARMUP_STEPS_RELORA_INIT // 5), LR_BASE)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(GRAD_ACCUM):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                _, loss = model(x, y)
            loss_scaled = loss / GRAD_ACCUM
            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item() / GRAD_ACCUM

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(accum_loss)
        global_step += 1

        if step % 500 == 0:
            val_loss = evaluate(model, val_loader)
            elapsed = time.time() - t0
            log(f"  warmup step {step}/{WARMUP_STEPS_RELORA_INIT}: "
                f"train={accum_loss:.4f}, val={val_loss:.4f}, lr={lr:.2e}")

    warmup_val = evaluate(model, val_loader)
    log(f"  Warmup done: val_loss={warmup_val:.4f}")

    # Clean up full-rank optimizer
    del optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()

    # Phase B: ReLoRA merge-restart cycles
    log(f"  Phase B: ReLoRA ({RELORA_CYCLES} cycles x {RELORA_STEPS_PER_CYCLE} steps, rank={RELORA_RANK})")

    cycle_losses = []

    for cycle in range(RELORA_CYCLES):
        log(f"\n  --- ReLoRA Cycle {cycle+1}/{RELORA_CYCLES} ---")

        # Apply LoRA
        apply_lora(model, RELORA_RANK, RELORA_ALPHA, freeze_base=True)

        # Fresh optimizer for LoRA params only
        lora_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(lora_params, lr=RELORA_LR, weight_decay=0.0,
                                       betas=(0.9, 0.95))
        scaler = torch.amp.GradScaler("cuda")

        cycle_loss_vals = []
        for step_in_cycle in range(RELORA_STEPS_PER_CYCLE):
            # Cosine restart LR
            lr = get_relora_lr(step_in_cycle, RELORA_STEPS_PER_CYCLE, RELORA_REWARMUP, RELORA_LR)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            optimizer.zero_grad()
            accum_loss = 0.0
            for _ in range(GRAD_ACCUM):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.autocast(device_type="cuda", dtype=DTYPE):
                    _, loss = model(x, y)
                loss_scaled = loss / GRAD_ACCUM
                scaler.scale(loss_scaled).backward()
                accum_loss += loss.item() / GRAD_ACCUM

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(accum_loss)
            cycle_loss_vals.append(accum_loss)
            global_step += 1

        cycle_val = evaluate(model, val_loader)
        cycle_mean = float(np.mean(cycle_loss_vals))
        cycle_losses.append({"cycle": cycle + 1, "mean_train": cycle_mean, "val": cycle_val})
        log(f"  Cycle {cycle+1} done: train={cycle_mean:.4f}, val={cycle_val:.4f}")

        # Merge LoRA into base
        for m in get_lora_modules(model):
            m.merge_and_reset()

        # Strip LoRA wrappers
        strip_lora(model)

        # Clean up
        del optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()

    # Final eval
    final_val = evaluate(model, val_loader, max_batches=200)
    elapsed = time.time() - t0
    log(f"  ReLoRA training done: val_loss={final_val:.4f}, time={elapsed/60:.1f}min")

    # Save
    torch.save(model.state_dict(), ckpt_path)

    meta = {
        "condition": "relora",
        "warmup_val_loss": float(warmup_val),
        "final_val_loss": float(final_val),
        "final_train_loss": float(np.mean(losses[-100:])),
        "elapsed_s": float(elapsed),
        "total_steps": TOTAL_STEPS,
        "warmup_steps": WARMUP_STEPS_RELORA_INIT,
        "relora_cycles": RELORA_CYCLES,
        "relora_rank": RELORA_RANK,
        "relora_steps_per_cycle": RELORA_STEPS_PER_CYCLE,
        "total_tokens": TOTAL_STEPS * BATCH_SIZE * GRAD_ACCUM * MAX_SEQ_LEN,
        "cycle_losses": cycle_losses,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return meta


# ── Phase 2: Expert Training ─────────────────────────────────────────────────

def train_expert(base_ckpt_path: Path, domain: str, domain_tokens: np.ndarray,
                 condition: str, val_tokens: np.ndarray) -> dict:
    """Train a single domain expert (LoRA) on a base model."""
    expert_dir = OUTPUT_DIR / "experts" / condition / domain
    meta_path = expert_dir / "meta.json"
    delta_path = expert_dir / "delta.npy"

    if meta_path.exists() and delta_path.exists():
        log(f"  {condition}/{domain}: already trained")
        with open(meta_path) as f:
            return json.load(f)

    expert_dir.mkdir(parents=True, exist_ok=True)

    # Load base model
    model = create_model()
    model.load_state_dict(torch.load(base_ckpt_path, map_location=DEVICE, weights_only=True))

    # Apply expert LoRA
    apply_lora(model, EXPERT_RANK, EXPERT_ALPHA, freeze_base=True)

    # Domain data
    train_ds = TokenDataset(domain_tokens, MAX_SEQ_LEN)
    val_ds = TokenDataset(val_tokens, MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=EXPERT_BATCH_SIZE, shuffle=True,
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=EXPERT_BATCH_SIZE, shuffle=False,
                            num_workers=1, pin_memory=True)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in lora_params)
    optimizer = torch.optim.AdamW(lora_params, lr=EXPERT_LR, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    train_iter = iter(train_loader)
    losses = []
    t0 = time.time()

    for step in range(EXPERT_STEPS):
        lr = get_lr(step, EXPERT_STEPS, 20, EXPERT_LR)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(EXPERT_GRAD_ACCUM):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                _, loss = model(x, y)
            loss_scaled = loss / EXPERT_GRAD_ACCUM
            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item() / EXPERT_GRAD_ACCUM

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(accum_loss)

    # Evaluate
    val_loss = evaluate(model, val_loader, max_batches=50)
    elapsed = time.time() - t0
    final_train = float(np.mean(losses[-50:]))
    log(f"  {condition}/{domain}: train={final_train:.4f}, val={val_loss:.4f}, "
        f"time={elapsed:.0f}s, params={trainable:,}")

    # Extract and save expert delta
    delta = extract_expert_delta(model)
    np.save(str(delta_path), delta)

    meta = {
        "domain": domain,
        "condition": condition,
        "final_train_loss": final_train,
        "final_val_loss": float(val_loss),
        "elapsed_s": float(elapsed),
        "steps": EXPERT_STEPS,
        "trainable_params": trainable,
        "delta_dim": len(delta),
        "delta_norm": float(np.linalg.norm(delta)),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return meta


# ── Phase 3: Composition Metrics ─────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_composition_metrics(condition: str) -> dict:
    """Compute pairwise cosine similarity between expert deltas."""
    deltas = {}
    for domain in DOMAINS:
        delta_path = OUTPUT_DIR / "experts" / condition / domain / "delta.npy"
        if delta_path.exists():
            deltas[domain] = np.load(str(delta_path))

    if len(deltas) < 2:
        return {"error": f"Only {len(deltas)} experts found for {condition}"}

    pairs = []
    cos_vals = []
    for i, d1 in enumerate(sorted(deltas.keys())):
        for d2 in sorted(deltas.keys())[i+1:]:
            cos = cosine_sim(deltas[d1], deltas[d2])
            pairs.append({"i": d1, "j": d2, "cos": cos, "abs_cos": abs(cos)})
            cos_vals.append(abs(cos))

    return {
        "mean_abs_cos": float(np.mean(cos_vals)),
        "max_abs_cos": float(np.max(cos_vals)),
        "min_abs_cos": float(np.min(cos_vals)),
        "std_abs_cos": float(np.std(cos_vals)),
        "n_experts": len(deltas),
        "delta_dim": len(list(deltas.values())[0]),
        "pairs": pairs,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true", help="Skip data preparation")
    parser.add_argument("--skip-base", action="store_true", help="Skip base training")
    parser.add_argument("--skip-experts", action="store_true", help="Skip expert training")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 72)
    log("RELORA FROM-SCRATCH COMPOSITION EXPERIMENT")
    log(f"  Model: GPT-2 ~124M (d={D_MODEL}, L={N_LAYERS}, heads={N_HEADS})")
    log(f"  Total steps: {TOTAL_STEPS}")
    log(f"  Conventional: {TOTAL_STEPS} full-rank steps")
    log(f"  ReLoRA: {WARMUP_STEPS_RELORA_INIT} warmup + {RELORA_CYCLES}x{RELORA_STEPS_PER_CYCLE} "
        f"cycles (rank={RELORA_RANK})")
    log(f"  Experts: {N_EXPERTS} domains, rank-{EXPERT_RANK}, {EXPERT_STEPS} steps")
    log(f"  Device: {DEVICE}")
    log("=" * 72)

    t0_total = time.time()

    # ── Data preparation ──────────────────────────────────────────────
    if not args.skip_data:
        train_tokens, val_tokens = prepare_data()
        domain_data = prepare_domain_data(train_tokens)
    else:
        train_tokens = np.load(str(OUTPUT_DIR / "train_tokens.npy"))
        val_tokens = np.load(str(OUTPUT_DIR / "val_tokens.npy"))
        domain_data = {d: np.load(str(OUTPUT_DIR / f"domain_{d}_tokens.npy")) for d in DOMAINS}

    log(f"\nData: {len(train_tokens)/1e6:.1f}M train tokens, {len(val_tokens)/1e6:.1f}M val tokens")
    for d in DOMAINS:
        log(f"  {d}: {len(domain_data[d])/1e6:.1f}M tokens")

    # ── Phase 1: Base training ────────────────────────────────────────
    if not args.skip_base:
        conv_meta = train_conventional(train_tokens, val_tokens)
        relora_meta = train_relora(train_tokens, val_tokens)
    else:
        with open(OUTPUT_DIR / "conventional_meta.json") as f:
            conv_meta = json.load(f)
        with open(OUTPUT_DIR / "relora_meta.json") as f:
            relora_meta = json.load(f)

    log(f"\nBase training results:")
    log(f"  Conventional: val_loss={conv_meta['final_val_loss']:.4f}")
    log(f"  ReLoRA:       val_loss={relora_meta['final_val_loss']:.4f}")

    base_ratio = relora_meta['final_val_loss'] / conv_meta['final_val_loss']
    log(f"  Base quality ratio (ReLoRA/conv): {base_ratio:.4f}")

    # ── Phase 2: Expert training ──────────────────────────────────────
    if not args.skip_experts:
        log("\n=== Phase 2: Expert Training ===")
        expert_metas = {"conventional": {}, "relora": {}}

        for domain in DOMAINS:
            log(f"\n--- Domain: {domain} ---")

            meta = train_expert(
                OUTPUT_DIR / "conventional_base.pt", domain,
                domain_data[domain], "conventional", val_tokens)
            expert_metas["conventional"][domain] = meta

            meta = train_expert(
                OUTPUT_DIR / "relora_base.pt", domain,
                domain_data[domain], "relora", val_tokens)
            expert_metas["relora"][domain] = meta
    else:
        expert_metas = {"conventional": {}, "relora": {}}
        for cond in ["conventional", "relora"]:
            for domain in DOMAINS:
                meta_path = OUTPUT_DIR / "experts" / cond / domain / "meta.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        expert_metas[cond][domain] = json.load(f)

    # ── Phase 3: Composition metrics ──────────────────────────────────
    log("\n=== Phase 3: Composition Metrics ===")

    conv_metrics = compute_composition_metrics("conventional")
    relora_metrics = compute_composition_metrics("relora")

    log(f"\n  Conventional: mean|cos|={conv_metrics['mean_abs_cos']:.8f}, "
        f"max={conv_metrics['max_abs_cos']:.8f}")
    log(f"  ReLoRA:       mean|cos|={relora_metrics['mean_abs_cos']:.8f}, "
        f"max={relora_metrics['max_abs_cos']:.8f}")

    # ── Kill criteria ─────────────────────────────────────────────────
    log("\n=== Kill Criteria Evaluation ===")

    cos_ratio = relora_metrics["mean_abs_cos"] / (conv_metrics["mean_abs_cos"] + 1e-12)

    conv_expert_losses = [m.get("final_val_loss", 0) for m in expert_metas["conventional"].values() if m]
    relora_expert_losses = [m.get("final_val_loss", 0) for m in expert_metas["relora"].values() if m]
    conv_mean_loss = np.mean(conv_expert_losses) if conv_expert_losses else float('inf')
    relora_mean_loss = np.mean(relora_expert_losses) if relora_expert_losses else float('inf')
    loss_ratio = relora_mean_loss / (conv_mean_loss + 1e-12)

    # K1: cos > 5x
    k1 = cos_ratio > 5.0
    log(f"  K1: cos_ratio = {cos_ratio:.4f} (threshold >5x) -> {'KILLED' if k1 else 'SURVIVES'}")

    # K2: loss > 1.20
    k2 = loss_ratio > 1.20
    log(f"  K2: loss_ratio = {loss_ratio:.4f} (threshold >1.20) -> {'KILLED' if k2 else 'SURVIVES'}")

    # K3: base quality > 1.20
    k3 = base_ratio > 1.20
    log(f"  K3: base_ratio = {base_ratio:.4f} (threshold >1.20) -> {'KILLED' if k3 else 'SURVIVES'}")

    # K4: training divergence (check if ReLoRA loss monotonically worsened)
    if "cycle_losses" in relora_meta:
        cycle_vals = [c["val"] for c in relora_meta["cycle_losses"]]
        diverged = len(cycle_vals) > 3 and all(cycle_vals[i] > cycle_vals[i-1] for i in range(len(cycle_vals)//2, len(cycle_vals)))
        k4 = diverged
    else:
        k4 = False
    log(f"  K4: divergence = {'YES' if k4 else 'NO'} -> {'KILLED' if k4 else 'SURVIVES'}")

    if k1 or k2 or k3 or k4:
        verdict = "KILLED"
    elif cos_ratio < 2.0 and loss_ratio < 1.10 and base_ratio < 1.10:
        verdict = "PROVEN"
    else:
        verdict = "INCONCLUSIVE"

    log(f"\n  VERDICT: {verdict}")

    # Random baseline
    D = conv_metrics.get("delta_dim", 0)
    random_cos = math.sqrt(2 / (math.pi * D)) if D > 0 else 0
    log(f"\n  Random baseline E[|cos|] = {random_cos:.2e} (D={D:,})")
    log(f"  Conv / random = {conv_metrics['mean_abs_cos'] / (random_cos + 1e-12):.1f}x")
    log(f"  ReLoRA / random = {relora_metrics['mean_abs_cos'] / (random_cos + 1e-12):.1f}x")

    # Scaling comparison
    log("\n  Scaling trend across experiments:")
    log(f"    Micro (d=64):   cos_ratio=1.77x (pretrained+perturbation)")
    log(f"    Macro (d=3584): cos_ratio=0.882x (pretrained+perturbation)")
    log(f"    THIS  (d=768):  cos_ratio={cos_ratio:.4f}x (FROM SCRATCH)")

    # ── Save results ──────────────────────────────────────────────────
    elapsed_total = time.time() - t0_total

    results = {
        "experiment": "relora_from_scratch_composition",
        "model": "GPT-2-124M",
        "hidden_size": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "vocab_size": VOCAB_SIZE,
        "total_params": 124_000_000,  # approximate
        "domains": DOMAINS,
        "config": {
            "total_steps": TOTAL_STEPS,
            "warmup_steps_conv": WARMUP_STEPS_CONV,
            "warmup_steps_relora": WARMUP_STEPS_RELORA_INIT,
            "relora_cycles": RELORA_CYCLES,
            "relora_rank": RELORA_RANK,
            "relora_steps_per_cycle": RELORA_STEPS_PER_CYCLE,
            "expert_rank": EXPERT_RANK,
            "expert_steps": EXPERT_STEPS,
            "lr_base": LR_BASE,
            "lr_relora": RELORA_LR,
            "lr_expert": EXPERT_LR,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "seq_len": MAX_SEQ_LEN,
        },
        "base_training": {
            "conventional": conv_meta,
            "relora": relora_meta,
            "base_quality_ratio": float(base_ratio),
        },
        "expert_training": expert_metas,
        "composition": {
            "conventional": conv_metrics,
            "relora": relora_metrics,
        },
        "ratios": {
            "cos_ratio": float(cos_ratio),
            "loss_ratio": float(loss_ratio),
            "base_ratio": float(base_ratio),
        },
        "random_baseline": {
            "expected_cos": float(random_cos),
            "delta_dim": D,
        },
        "kill_criteria": {
            "K1_cos_ratio_gt_5x": bool(k1),
            "K2_loss_ratio_gt_1_20": bool(k2),
            "K3_base_ratio_gt_1_20": bool(k3),
            "K4_training_diverged": bool(k4),
        },
        "verdict": verdict,
        "scaling_comparison": {
            "micro_d64_perturbation_cos_ratio": 1.77,
            "macro_d3584_perturbation_cos_ratio": 0.882,
            "this_d768_from_scratch_cos_ratio": float(cos_ratio),
        },
        "elapsed_total_s": float(elapsed_total),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Also save to repo results dir
    repo_results = REPO_ROOT / "results" / "relora_from_scratch_results.json"
    with open(repo_results, "w") as f:
        json.dump(results, f, indent=2)

    log(f"\nResults saved to {results_file}")
    log(f"Total experiment time: {elapsed_total/60:.1f} minutes")
    log(f"Estimated cost: ${elapsed_total/3600 * 0.16:.2f}")

    return results


if __name__ == "__main__":
    main()
