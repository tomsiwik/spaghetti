#!/usr/bin/env python3
"""LTE No-Reset at Macro Scale: Does omitting LoRA reset after merge stabilize
at d=768+ where alpha/r is proportionally smaller?

Runs ON RunPod (A5000, 24GB VRAM).

Design (2-way comparison on GPT-2 125M):
  Phase 1: Train conventional base (shared, 10K steps)
  Phase 2a: LTE with RESET — K=4 parallel branches, merge-average-reset per cycle
  Phase 2b: LTE WITHOUT RESET — same as 2a but LoRA params survive across merges
  Phase 3: Train 4 domain experts on each base, compare composition quality

The no-reset variant was found to diverge at micro scale (d=64) due to forward-pass
double-counting: after merge, base has W + alpha/r * B@A absorbed, but the forward
pass also adds alpha/r * B@A from the unfrozen LoRA. At d=768, alpha/r = 16/16 = 1.0
but weight norms are ~O(sqrt(d)) = ~28, so LoRA contribution is proportionally ~3.6%
instead of ~12% at d=64. This may suppress the instability.

Kill criteria:
  K1: no-reset LTE diverges at macro scale (loss ratio > 2x vs reset at any point)
  K2: no-reset LTE quality >20% worse than reset variant at same compute
  K3: alpha/r scaling issue persists at macro scale (expert composition cos > 5x worse)

Usage (on RunPod):
    cd /workspace/llm
    python macro/lte_no_reset/run_lte_no_reset.py

Expected runtime: ~1.5-2 hours on A5000
Estimated cost: ~$0.30-$0.40
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
OUTPUT_DIR = REPO_ROOT / "results" / "lte_no_reset"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# Model config (GPT-2 small, ~124M params)
D_MODEL = 768
D_FF = 3072
N_LAYERS = 12
N_HEADS = 12
VOCAB_SIZE = 50257
MAX_SEQ_LEN = 512
DROPOUT = 0.0

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

# Training config
BATCH_SIZE = 8
GRAD_ACCUM = 4 if not IS_SMOKE else 1
TOTAL_STEPS = 10000 if not IS_SMOKE else 5
WARMUP_STEPS = 500 if not IS_SMOKE else 2
LR_BASE = 6e-4
WEIGHT_DECAY = 0.1

# LTE config
LTE_RANK = 16
LTE_ALPHA = 16
LTE_N_HEADS = 4       # parallel branches
LTE_MERGE_EVERY = 250 if not IS_SMOKE else 2  # steps per head per cycle
LTE_STEPS = 4000 if not IS_SMOKE else 4       # total LTE steps (same compute both conditions)
LTE_LR = 3e-4
LTE_REWARMUP = 50 if not IS_SMOKE else 1

# Expert config
EXPERT_RANK = 16
EXPERT_ALPHA = 16
EXPERT_STEPS = 500 if not IS_SMOKE else 3
EXPERT_LR = 2e-4
EXPERT_BATCH_SIZE = 4
EXPERT_GRAD_ACCUM = 2 if not IS_SMOKE else 1
N_EXPERTS = 4 if not IS_SMOKE else 2

DOMAINS = ["code", "science", "medical", "legal"]
DOMAIN_KEYWORDS = {
    "code": ["def ", "import ", "class ", "function", "return ", "var ", "const "],
    "science": ["experiment", "hypothesis", "molecule", "equation", "quantum", "theory"],
    "medical": ["patient", "diagnosis", "treatment", "clinical", "symptoms", "therapy"],
    "legal": ["court", "plaintiff", "defendant", "statute", "jurisdiction", "contract"],
}

SEED = 42


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Model (GPT-2 small, same as relora_from_scratch) ────────────────────────

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
        self.register_buffer("bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
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


# ── LoRA Module ───────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
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
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

    def merge_and_reset(self):
        """Merge LoRA delta into base weights and reinitialize A, B."""
        with torch.no_grad():
            delta = (self.lora_A.float() @ self.lora_B.float()) * self.scaling
            self.base.weight.data += delta.T.to(self.base.weight.dtype)
            nn.init.normal_(self.lora_A, std=1.0 / math.sqrt(self.rank))
            nn.init.zeros_(self.lora_B)

    def merge_no_reset(self):
        """Merge LoRA delta into base weights WITHOUT resetting A, B.

        This is the key variant: after merge, base absorbs the delta but
        LoRA params keep their current values. The forward pass will then
        compute: base_new(x) + alpha/r * (x @ A @ B) where base_new already
        includes the previous delta. This means the LoRA contribution is
        effectively double-counted on the first forward pass after merge.

        At d=64 (micro), this caused 8x scaling divergence.
        At d=768 (macro), the relative contribution is ~3.6% vs ~12%, which
        may suppress the instability.
        """
        with torch.no_grad():
            delta = (self.lora_A.float() @ self.lora_B.float()) * self.scaling
            self.base.weight.data += delta.T.to(self.base.weight.dtype)
            # No reset — A and B keep their trained values

    def get_delta_flat(self):
        with torch.no_grad():
            delta = (self.lora_A @ self.lora_B) * self.scaling
            return delta.T.reshape(-1).float().cpu().numpy()


def apply_lora(model: GPT2, rank: int, alpha: int, freeze_base: bool = True):
    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False

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
    parts = []
    for m in get_lora_modules(model):
        parts.append(m.get_delta_flat())
    return np.concatenate(parts)


def strip_lora(model: GPT2):
    for block in model.blocks:
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            m = getattr(block.attn, name)
            if isinstance(m, LoRALinear):
                setattr(block.attn, name, m.base)
        for name in ["fc1", "fc2"]:
            m = getattr(block.mlp, name)
            if isinstance(m, LoRALinear):
                setattr(block.mlp, name, m.base)
    for p in model.parameters():
        p.requires_grad = True


# ── Dataset ───────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
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
    data_path = OUTPUT_DIR / "train_tokens.npy"
    val_path = OUTPUT_DIR / "val_tokens.npy"

    # Try to reuse data from relora_from_scratch if it exists
    relora_data = REPO_ROOT / "results" / "relora_from_scratch" / "train_tokens.npy"
    relora_val = REPO_ROOT / "results" / "relora_from_scratch" / "val_tokens.npy"

    if data_path.exists() and val_path.exists():
        log("Loading cached tokenized data")
        return np.load(str(data_path)), np.load(str(val_path))

    if relora_data.exists() and relora_val.exists():
        log("Reusing data from relora_from_scratch")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(str(relora_data), str(data_path))
        shutil.copy(str(relora_val), str(val_path))
        return np.load(str(data_path)), np.load(str(val_path))

    log("Preparing training data from C4...")
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

    all_tokens = []
    target_tokens = 50_000_000
    n_docs = 0

    for example in ds:
        tokens = tokenizer.encode(example["text"])
        all_tokens.extend(tokens)
        n_docs += 1
        if len(all_tokens) >= target_tokens:
            break
        if n_docs % 10000 == 0:
            log(f"  Tokenized {n_docs} docs, {len(all_tokens)/1e6:.1f}M tokens")

    tokens_arr = np.array(all_tokens, dtype=np.uint16)
    split_idx = int(len(tokens_arr) * 0.99)
    train_tokens = tokens_arr[:split_idx]
    val_tokens = tokens_arr[split_idx:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(data_path), train_tokens)
    np.save(str(val_path), val_tokens)
    log(f"  Saved {len(train_tokens)/1e6:.1f}M train, {len(val_tokens)/1e6:.1f}M val tokens")
    return train_tokens, val_tokens


def prepare_domain_data(tokenizer_name="gpt2"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    domain_paths = {d: OUTPUT_DIR / f"domain_{d}_tokens.npy" for d in DOMAINS}

    # Try to reuse from relora_from_scratch
    relora_dir = REPO_ROOT / "results" / "relora_from_scratch"

    if all(p.exists() for p in domain_paths.values()):
        log("Loading cached domain data")
        return {d: np.load(str(p)) for d, p in domain_paths.items()}

    # Check if relora has them
    relora_available = {d: relora_dir / f"domain_{d}_tokens.npy" for d in DOMAINS}
    if all(p.exists() for p in relora_available.values()):
        log("Reusing domain data from relora_from_scratch")
        import shutil
        for d in DOMAINS:
            shutil.copy(str(relora_available[d]), str(domain_paths[d]))
        return {d: np.load(str(p)) for d, p in domain_paths.items()}

    log("Preparing domain-specific data from C4...")
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

    domain_tokens = {d: [] for d in DOMAINS}
    target_per_domain = 2_000_000
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
        if len(done) == len(DOMAINS):
            break

    for domain in DOMAINS:
        tokens = np.array(domain_tokens[domain][:2_000_000], dtype=np.uint16)
        np.save(str(domain_paths[domain]), tokens)
        log(f"  {domain}: {len(tokens)/1e6:.1f}M tokens")

    return {d: np.load(str(domain_paths[d])) for d in DOMAINS}


# ── Training Functions ────────────────────────────────────────────────────────

def create_model():
    model = GPT2(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, D_FF, MAX_SEQ_LEN, DROPOUT)
    log(f"  Model params: {model.count_params():,}")
    return model.to(DEVICE)


def evaluate(model, val_loader, max_batches=50):
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
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * (min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)))


# ── Phase 1: Conventional Base Training ───────────────────────────────────────

def train_conventional_base(train_tokens, val_tokens) -> dict:
    ckpt_path = OUTPUT_DIR / "conventional_base.pt"
    meta_path = OUTPUT_DIR / "conventional_meta.json"

    # Try reusing from relora_from_scratch
    relora_ckpt = REPO_ROOT / "results" / "relora_from_scratch" / "conventional_base.pt"
    relora_meta = REPO_ROOT / "results" / "relora_from_scratch" / "conventional_meta.json"
    if not ckpt_path.exists() and relora_ckpt.exists():
        log("Reusing conventional base from relora_from_scratch")
        import shutil
        shutil.copy(str(relora_ckpt), str(ckpt_path))
        if relora_meta.exists():
            shutil.copy(str(relora_meta), str(meta_path))

    if ckpt_path.exists() and meta_path.exists():
        log("Conventional base already trained, loading")
        with open(meta_path) as f:
            return json.load(f)

    log("=" * 70)
    log("Phase 1: Conventional Base Training")
    log("=" * 70)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = create_model()
    train_ds = TokenDataset(train_tokens, MAX_SEQ_LEN)
    val_ds = TokenDataset(val_tokens, MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_BASE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    train_iter = iter(train_loader)
    losses = []
    t0 = time.time()

    for step in range(TOTAL_STEPS):
        lr = get_lr(step, TOTAL_STEPS, WARMUP_STEPS, LR_BASE)
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
            (loss / GRAD_ACCUM).backward()
            accum_loss += loss.item() / GRAD_ACCUM

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(accum_loss)

        if step % 500 == 0 or step == TOTAL_STEPS - 1:
            val_loss = evaluate(model, val_loader)
            elapsed = time.time() - t0
            log(f"  step {step}/{TOTAL_STEPS}: train={accum_loss:.4f}, val={val_loss:.4f}, lr={lr:.2e}")

    final_val = evaluate(model, val_loader, max_batches=200)
    elapsed = time.time() - t0
    log(f"  Conventional base done: val_loss={final_val:.4f}, time={elapsed/60:.1f}min")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    meta = {
        "condition": "conventional",
        "final_val_loss": final_val,
        "final_train_loss": float(np.mean(losses[-100:])),
        "elapsed_s": elapsed,
        "total_steps": TOTAL_STEPS,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_conventional_base():
    ckpt_path = OUTPUT_DIR / "conventional_base.pt"
    model = create_model()
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    return model


# ── Phase 2: LTE Training (Reset vs No-Reset) ───────────────────────────────

def train_lte(base_model, train_tokens, val_tokens, condition: str, use_reset: bool) -> dict:
    """Train LTE-style parallel LoRA branches on top of conventional base.

    Args:
        condition: "lte_reset" or "lte_no_reset"
        use_reset: if True, reset A/B after each merge; if False, keep them
    """
    ckpt_path = OUTPUT_DIR / f"{condition}_base.pt"
    meta_path = OUTPUT_DIR / f"{condition}_meta.json"

    if ckpt_path.exists() and meta_path.exists():
        log(f"{condition} already trained, loading")
        with open(meta_path) as f:
            return json.load(f)

    log("=" * 70)
    log(f"Phase 2: LTE Training ({condition}, reset={use_reset})")
    log("=" * 70)

    # Deep copy the base model for this condition
    model = create_model()
    base_ckpt = OUTPUT_DIR / "conventional_base.pt"
    model.load_state_dict(torch.load(base_ckpt, map_location=DEVICE, weights_only=True))

    train_ds = TokenDataset(train_tokens, MAX_SEQ_LEN)
    val_ds = TokenDataset(val_tokens, MAX_SEQ_LEN)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

    # Eval base before LTE
    base_val_loss = evaluate(model, val_loader)
    log(f"  Base val_loss before LTE: {base_val_loss:.4f}")

    # Apply LoRA
    apply_lora(model, LTE_RANK, LTE_ALPHA, freeze_base=True)
    log(f"  LoRA applied: rank={LTE_RANK}, alpha={LTE_ALPHA}, scaling={LTE_ALPHA/LTE_RANK}")

    # Compute how many cycles we can do
    steps_per_cycle = LTE_N_HEADS * LTE_MERGE_EVERY
    n_cycles = LTE_STEPS // steps_per_cycle
    log(f"  LTE config: {LTE_N_HEADS} heads, {LTE_MERGE_EVERY} steps/head/cycle, "
        f"{n_cycles} cycles, {LTE_STEPS} total steps")

    # Track divergence
    loss_history = []
    merge_losses = []
    t0 = time.time()
    global_step = 0

    # Data loaders for each head (different shuffles = different data shards)
    head_loaders = []
    for k in range(LTE_N_HEADS):
        ds_k = TokenDataset(train_tokens, MAX_SEQ_LEN)
        g = torch.Generator()
        g.manual_seed(SEED + k * 7919)
        loader = DataLoader(ds_k, batch_size=BATCH_SIZE, shuffle=True, num_workers=1,
                           pin_memory=True, generator=g)
        head_loaders.append(iter(loader))

    lora_mods = get_lora_modules(model)

    for cycle in range(n_cycles):
        # Save head-specific LoRA params (if reset, they start fresh each cycle)
        head_params = []
        for k in range(LTE_N_HEADS):
            params = {}
            for m_idx, m in enumerate(lora_mods):
                if use_reset or cycle == 0:
                    # Fresh init for reset mode, or first cycle of no-reset
                    if cycle > 0 or k > 0:  # head 0, cycle 0 keeps init from apply_lora
                        d_in = m.base.in_features
                        A = torch.randn(d_in, m.rank, device=DEVICE, dtype=m.lora_A.dtype) * (1.0 / math.sqrt(m.rank))
                        B = torch.zeros(m.rank, m.base.out_features, device=DEVICE, dtype=m.lora_B.dtype)
                        params[m_idx] = (A, B)
                    else:
                        params[m_idx] = (m.lora_A.data.clone(), m.lora_B.data.clone())
                else:
                    # No-reset: all heads share the current LoRA state
                    params[m_idx] = (m.lora_A.data.clone(), m.lora_B.data.clone())
            head_params.append(params)

        # Train each head sequentially (simulates parallel training)
        head_losses = []
        for k in range(LTE_N_HEADS):
            # Load this head's LoRA params
            for m_idx, m in enumerate(lora_mods):
                if m_idx in head_params[k]:
                    m.lora_A.data.copy_(head_params[k][m_idx][0])
                    m.lora_B.data.copy_(head_params[k][m_idx][1])

            # Create optimizer for this head
            lora_params = [p for m in lora_mods for p in [m.lora_A, m.lora_B]]
            optimizer = torch.optim.AdamW(lora_params, lr=LTE_LR, weight_decay=0.01)
            scaler = torch.amp.GradScaler("cuda")

            model.train()
            head_loss_accum = 0.0
            for step in range(LTE_MERGE_EVERY):
                # LR schedule with warmup within cycle
                if cycle > 0 and step < LTE_REWARMUP:
                    lr = LTE_LR * (step + 1) / LTE_REWARMUP
                else:
                    lr = LTE_LR
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                optimizer.zero_grad()
                try:
                    x, y = next(head_loaders[k])
                except StopIteration:
                    ds_k = TokenDataset(train_tokens, MAX_SEQ_LEN)
                    g = torch.Generator()
                    g.manual_seed(SEED + k * 7919 + cycle * 1000)
                    head_loaders[k] = iter(DataLoader(ds_k, batch_size=BATCH_SIZE, shuffle=True,
                                                       num_workers=1, pin_memory=True, generator=g))
                    x, y = next(head_loaders[k])
                x, y = x.to(DEVICE), y.to(DEVICE)

                with torch.autocast(device_type="cuda", dtype=DTYPE):
                    _, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                scaler.step(optimizer)
                scaler.update()

                head_loss_accum += loss.item()
                global_step += 1
                loss_history.append(loss.item())

            head_losses.append(head_loss_accum / LTE_MERGE_EVERY)

            # Save trained params back
            for m_idx, m in enumerate(lora_mods):
                head_params[k][m_idx] = (m.lora_A.data.clone(), m.lora_B.data.clone())

        # Merge: average all K head deltas into base
        with torch.no_grad():
            for m_idx, m in enumerate(lora_mods):
                avg_delta = torch.zeros_like(m.base.weight)
                for k in range(LTE_N_HEADS):
                    A, B = head_params[k][m_idx]
                    delta = (A.float() @ B.float()) * m.scaling  # (d_in, d_out)
                    avg_delta += delta.T.to(avg_delta.dtype)  # (d_out, d_in)
                avg_delta /= LTE_N_HEADS
                m.base.weight.data += avg_delta

                if use_reset:
                    # Reset LoRA to fresh init
                    nn.init.normal_(m.lora_A, std=1.0 / math.sqrt(m.rank))
                    nn.init.zeros_(m.lora_B)
                else:
                    # No-reset: keep average of head params as starting point
                    avg_A = sum(head_params[k][m_idx][0] for k in range(LTE_N_HEADS)) / LTE_N_HEADS
                    avg_B = sum(head_params[k][m_idx][1] for k in range(LTE_N_HEADS)) / LTE_N_HEADS
                    m.lora_A.data.copy_(avg_A)
                    m.lora_B.data.copy_(avg_B)

        # Evaluate after merge
        merge_val_loss = evaluate(model, val_loader)
        merge_losses.append(merge_val_loss)
        avg_head_loss = np.mean(head_losses)

        log(f"  Cycle {cycle+1}/{n_cycles}: head_train={avg_head_loss:.4f}, "
            f"val_after_merge={merge_val_loss:.4f}, "
            f"steps={global_step}/{LTE_STEPS}")

        # K1 check: early stopping on divergence
        if merge_val_loss > base_val_loss * 2.0:
            log(f"  ** K1 TRIGGERED: val_loss {merge_val_loss:.4f} > 2x base {base_val_loss:.4f}")
            log(f"  ** {condition} DIVERGED at cycle {cycle+1}")
            break

    # Final eval
    final_val = evaluate(model, val_loader, max_batches=200)
    elapsed = time.time() - t0

    # Strip LoRA (merge final delta if no-reset has residual)
    if not use_reset:
        for m in lora_mods:
            m.merge_no_reset()
    strip_lora(model)

    # Save
    torch.save(model.state_dict(), ckpt_path)

    meta = {
        "condition": condition,
        "use_reset": use_reset,
        "base_val_loss": base_val_loss,
        "final_val_loss": final_val,
        "merge_losses": merge_losses,
        "loss_ratio_vs_base": final_val / base_val_loss if base_val_loss > 0 else float('inf'),
        "n_cycles_completed": len(merge_losses),
        "n_cycles_planned": n_cycles,
        "diverged": final_val > base_val_loss * 2.0,
        "elapsed_s": elapsed,
        "total_steps": global_step,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log(f"  {condition} done: val={final_val:.4f}, ratio={meta['loss_ratio_vs_base']:.4f}, "
        f"diverged={meta['diverged']}, time={elapsed/60:.1f}min")
    return meta


# ── Phase 3: Expert Training & Composition ───────────────────────────────────

def train_experts(condition: str, train_tokens, val_tokens, domain_data) -> dict:
    results_path = OUTPUT_DIR / f"{condition}_experts.json"
    if results_path.exists():
        log(f"Expert results for {condition} already exist, loading")
        with open(results_path) as f:
            return json.load(f)

    log("=" * 70)
    log(f"Phase 3: Expert Training on {condition}")
    log("=" * 70)

    ckpt_path = OUTPUT_DIR / f"{condition}_base.pt"
    val_ds = TokenDataset(val_tokens, MAX_SEQ_LEN)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

    expert_deltas = []
    expert_losses = []
    t0 = time.time()

    for i, domain in enumerate(DOMAINS[:N_EXPERTS]):
        log(f"  Training expert {i+1}/{N_EXPERTS}: {domain}")

        # Load fresh base
        model = create_model()
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))

        # Apply LoRA
        apply_lora(model, EXPERT_RANK, EXPERT_ALPHA, freeze_base=True)

        # Domain data
        dtokens = domain_data[domain]
        domain_ds = TokenDataset(dtokens, MAX_SEQ_LEN)
        domain_loader = DataLoader(domain_ds, batch_size=EXPERT_BATCH_SIZE, shuffle=True,
                                    num_workers=1, pin_memory=True)

        lora_params = [p for m in get_lora_modules(model) for p in [m.lora_A, m.lora_B]]
        optimizer = torch.optim.AdamW(lora_params, lr=EXPERT_LR, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda")

        model.train()
        domain_iter = iter(domain_loader)
        losses = []

        for step in range(EXPERT_STEPS):
            optimizer.zero_grad()
            accum_loss = 0.0
            for _ in range(EXPERT_GRAD_ACCUM):
                try:
                    x, y = next(domain_iter)
                except StopIteration:
                    domain_iter = iter(domain_loader)
                    x, y = next(domain_iter)
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.autocast(device_type="cuda", dtype=DTYPE):
                    _, loss = model(x, y)
                (loss / EXPERT_GRAD_ACCUM).backward()
                accum_loss += loss.item() / EXPERT_GRAD_ACCUM

            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(accum_loss)

        # Extract delta and evaluate
        delta = extract_expert_delta(model)
        expert_deltas.append(delta)

        val_loss = evaluate(model, val_loader)
        expert_losses.append({"domain": domain, "final_train_loss": losses[-1], "val_loss": val_loss})
        log(f"    {domain}: train={losses[-1]:.4f}, val={val_loss:.4f}")

        del model, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()

    # Compute pairwise cosines
    cos_matrix = np.zeros((N_EXPERTS, N_EXPERTS))
    for i in range(N_EXPERTS):
        for j in range(N_EXPERTS):
            cos = np.dot(expert_deltas[i], expert_deltas[j]) / (
                np.linalg.norm(expert_deltas[i]) * np.linalg.norm(expert_deltas[j]) + 1e-10)
            cos_matrix[i, j] = cos

    # Same vs different domain cosine
    same_cos = [cos_matrix[i, i] for i in range(N_EXPERTS)]
    diff_cos = [cos_matrix[i, j] for i in range(N_EXPERTS) for j in range(N_EXPERTS) if i != j]
    mean_same = float(np.mean(same_cos))
    mean_diff = float(np.mean(diff_cos))

    elapsed = time.time() - t0
    result = {
        "condition": condition,
        "expert_losses": expert_losses,
        "mean_same_cos": mean_same,
        "mean_diff_cos": mean_diff,
        "cos_matrix": cos_matrix.tolist(),
        "elapsed_s": elapsed,
    }
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    log(f"  {condition} experts: mean_diff_cos={mean_diff:.6f}, mean_same_cos={mean_same:.6f}")
    return result


# ── Main Experiment ──────────────────────────────────────────────────────────

def run_experiment():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("=" * 70)
    log("LTE No-Reset at Macro Scale")
    log(f"  Device: {DEVICE}")
    log(f"  Smoke test: {IS_SMOKE}")
    log(f"  d_model={D_MODEL}, n_layers={N_LAYERS}, lte_rank={LTE_RANK}")
    log(f"  alpha/r scaling = {LTE_ALPHA/LTE_RANK}")
    log("=" * 70)

    # Prepare data
    train_tokens, val_tokens = prepare_data()
    domain_data = prepare_domain_data()

    # Phase 1: Conventional base (shared)
    conv_meta = train_conventional_base(train_tokens, val_tokens)
    log(f"  Conventional base: val_loss={conv_meta['final_val_loss']:.4f}")

    # Phase 2a: LTE with reset
    reset_meta = train_lte(None, train_tokens, val_tokens, "lte_reset", use_reset=True)

    # Phase 2b: LTE without reset
    no_reset_meta = train_lte(None, train_tokens, val_tokens, "lte_no_reset", use_reset=False)

    # Phase 3: Expert composition on both bases
    reset_experts = train_experts("lte_reset", train_tokens, val_tokens, domain_data)
    no_reset_experts = train_experts("lte_no_reset", train_tokens, val_tokens, domain_data)

    # ── Analysis ─────────────────────────────────────────────────────────

    log("")
    log("=" * 70)
    log("ANALYSIS")
    log("=" * 70)

    base_val = conv_meta['final_val_loss']
    reset_val = reset_meta['final_val_loss']
    no_reset_val = no_reset_meta['final_val_loss']

    log(f"  Base val_loss:      {base_val:.4f}")
    log(f"  LTE reset val_loss: {reset_val:.4f} (ratio: {reset_val/base_val:.4f})")
    log(f"  LTE no-reset val:   {no_reset_val:.4f} (ratio: {no_reset_val/base_val:.4f})")

    # K1: Divergence check
    k1_diverged = no_reset_meta.get('diverged', False)
    log(f"  K1 (divergence): {'KILLED' if k1_diverged else 'SURVIVES'}")

    # K2: Quality comparison (>20% worse than reset)
    if reset_val > 0:
        quality_ratio = no_reset_val / reset_val
    else:
        quality_ratio = float('inf')
    k2_killed = quality_ratio > 1.20
    log(f"  K2 (quality vs reset): ratio={quality_ratio:.4f} {'KILLED' if k2_killed else 'SURVIVES'} (threshold: 1.20)")

    # K3: Composition quality (cos > 5x worse)
    reset_cos = reset_experts['mean_diff_cos']
    no_reset_cos = no_reset_experts['mean_diff_cos']
    if abs(reset_cos) > 1e-10:
        cos_ratio = abs(no_reset_cos) / abs(reset_cos)
    else:
        cos_ratio = 0.0
    k3_killed = cos_ratio > 5.0
    log(f"  K3 (composition cos): reset={reset_cos:.6f}, no_reset={no_reset_cos:.6f}, "
        f"ratio={cos_ratio:.4f} {'KILLED' if k3_killed else 'SURVIVES'}")

    any_killed = k1_diverged or k2_killed or k3_killed
    status = "KILLED" if any_killed else "SURVIVES"

    log("")
    log("=" * 70)
    log(f"OVERALL: {status}")
    log("=" * 70)

    # Additional analysis: track loss trajectory
    log("")
    log("Loss trajectory per merge cycle:")
    log(f"  Reset merge losses:    {[f'{x:.4f}' for x in reset_meta.get('merge_losses', [])]}")
    log(f"  No-reset merge losses: {[f'{x:.4f}' for x in no_reset_meta.get('merge_losses', [])]}")

    # Weight norm analysis (quantify the alpha/r relative contribution)
    log("")
    log("Scaling analysis:")
    log(f"  d_model={D_MODEL}, lte_rank={LTE_RANK}, alpha/r={LTE_ALPHA/LTE_RANK}")
    log(f"  Theoretical relative contribution: alpha/r / sqrt(d) = {LTE_ALPHA/LTE_RANK / math.sqrt(D_MODEL):.4f}")

    total_elapsed = time.time() - t0

    # Save final results
    final = {
        "experiment": "lte_no_reset_macro",
        "status": status,
        "kill_criteria": {
            "K1_diverged": k1_diverged,
            "K2_quality_ratio": quality_ratio,
            "K2_killed": k2_killed,
            "K3_cos_ratio": cos_ratio,
            "K3_killed": k3_killed,
        },
        "conditions": {
            "base_val_loss": base_val,
            "reset_val_loss": reset_val,
            "no_reset_val_loss": no_reset_val,
            "reset_loss_ratio": reset_val / base_val if base_val > 0 else None,
            "no_reset_loss_ratio": no_reset_val / base_val if base_val > 0 else None,
        },
        "composition": {
            "reset_mean_diff_cos": reset_cos,
            "no_reset_mean_diff_cos": no_reset_cos,
        },
        "config": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "lte_rank": LTE_RANK,
            "lte_alpha": LTE_ALPHA,
            "lte_n_heads": LTE_N_HEADS,
            "lte_steps": LTE_STEPS,
            "n_experts": N_EXPERTS,
        },
        "elapsed_s": total_elapsed,
        "elapsed_min": total_elapsed / 60,
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(final, f, indent=2)

    log(f"\nTotal elapsed: {total_elapsed/60:.1f}min")
    log(f"Results saved to {results_path}")


if __name__ == "__main__":
    run_experiment()
