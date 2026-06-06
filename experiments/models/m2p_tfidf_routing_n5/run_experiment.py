#!/usr/bin/env python3
"""M2P TF-IDF Routing N=5: Distribution-agnostic sequence-level routing.

TYPE: verification (Type 1)
MATH: experiments/models/m2p_tfidf_routing_n5/MATH.md

ROOT CAUSE FIXED (exp_m2p_composition_n5, Finding #351):
  Per-token MLP router failed at 36.6% (K852 FAIL) because:
  1. Train/test distribution mismatch: router trained on BASE hidden states,
     deployed on COMPOSED hidden states — structural covariate shift.
  2. Per-token vocabulary ambiguity: sort/reverse/repeat share {a-h} alphabet,
     so any single token is uninformative about which of 3 domains generated it.

SOLUTION (Theorem 1, MATH.md):
  TF-IDF routing operates on INPUT TEXT SEQUENCES only (before any model
  forward pass). It is trivially invariant to model distribution by
  construction. Domain format strings (">", "*", "=", "sort:", etc.)
  provide TF-IDF-separable sequence-level features (Theorem 2).

PRIOR EVIDENCE:
  Finding #207, #247: TF-IDF + logistic regression → 90% routing on 5 SFT
  domains (contrastive_routing_n5). Same architecture applied here.
  Finding #351: Per-domain M2P quality 93.3% of SFT — bottleneck is routing.
  LoraRetriever (arXiv:2402.09997): text-based routing decoupled from NLL.

APPROACH:
  1. Reuse ALL code from m2p_composition_n5 (same arch, data, training).
  2. REMOVE: per-token MLP router (replaced entirely).
  3. ADD: TF-IDF + logistic regression on full input string (sequence-level).
  4. ADD: oracle routing path (ground-truth domain at inference).
  5. Compare: base / SFT / oracle-routed / tfidf-routed composition.

REUSE POLICY:
  - If adapters exist in m2p_composition_n5/adapters/, load them (skip retraining).
  - Otherwise, retrain from scratch (self-contained experiment).

Kill criteria (MATH.md Corollary):
  K867: TF-IDF routing accuracy > 70% on 5 M2P domains (vs 36.6% failure)
  K868: M2P composition quality > 70% of SFT with TF-IDF routing
  K869: Oracle routing quality ceiling > 80% of SFT (ground-truth labels)
"""

import gc
import json
import math
import os
import pickle
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety (CODING_GUIDELINES §2)
_dev = mx.device_info()
mx.set_memory_limit(_dev["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
ADAPTER_DIR = EXPERIMENT_DIR / "adapters"
ADAPTER_DIR.mkdir(exist_ok=True)

# Source adapters from m2p_composition_n5 (reuse if available)
SOURCE_ADAPTER_DIR = EXPERIMENT_DIR.parent / "m2p_composition_n5" / "adapters"

SEED = 42
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

# ── Architecture constants (IDENTICAL to m2p_composition_n5) ──────────────
D_MODEL = 256
N_LAYERS = 2
N_HEADS = 4
VOCAB_SIZE = 128
BLOCK_SIZE = 48
LORA_RANK = 4
LORA_SCALE = 2.0
N_DOMAINS = 5

D_M2P = 64
N_MEMORY = 32
M2P_LAYERS = 2

MODULE_NAMES = ["wq", "wk", "wv", "wo", "fc1"]
MODULE_OUT_DIMS = [D_MODEL, D_MODEL, D_MODEL, D_MODEL, 4 * D_MODEL]
N_MODULES = len(MODULE_NAMES)

BASE_STEPS = 1200 if not SMOKE_TEST else 60
SFT_STEPS  = 400  if not SMOKE_TEST else 30
M2P_STEPS  = 500  if not SMOKE_TEST else 30
LR = 3e-4
SFT_LR = 1e-3
M2P_LR = 1e-3

DOMAIN_NAMES = ["arithmetic", "sort", "parity", "reverse", "repeat"]


# ── Utilities ─────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


def log(m): print(m, flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def log_memory(label=""):
    a = mx.get_active_memory() / 1e9
    c = mx.get_cache_memory() / 1e9
    p = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB peak={p:.2f}GB")


# ── Data generation (IDENTICAL to m2p_composition_n5) ─────────────────────

def gen_domain_data(domain_id: int, n: int, rng: np.random.RandomState) -> list:
    """Generate n text samples for a given domain."""
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        if domain_id == 0:  # arithmetic
            a, b = rng.randint(0, 50), rng.randint(0, 50)
            data.append(f"{a}+{b}={a+b}")
        elif domain_id == 1:  # sort
            s = "".join(rng.choice(list(chars)) for _ in range(rng.randint(2, 5)))
            data.append(f"{s}>{''.join(sorted(s))}")
        elif domain_id == 2:  # parity
            bits = "".join(str(rng.randint(0, 2)) for _ in range(rng.randint(2, 6)))
            data.append(f"{bits}>{'even' if bits.count('1') % 2 == 0 else 'odd'}")
        elif domain_id == 3:  # reverse
            s = "".join(rng.choice(list(chars)) for _ in range(rng.randint(2, 5)))
            data.append(f"{s}>{''.join(reversed(s))}")
        elif domain_id == 4:  # repeat
            p = "".join(rng.choice(list(chars)) for _ in range(rng.randint(1, 3)))
            r = rng.randint(2, 4)
            data.append(f"{p}*{r}={p*r}")
    return data


def encode_text(text: str) -> list:
    return [ord(c) % VOCAB_SIZE for c in text[:BLOCK_SIZE + 1]]


def make_batches(texts: list) -> list:
    return [mx.array(encode_text(t)) for t in texts if len(t) >= 4]


def tokens_to_string(tokens) -> str:
    """Decode token array back to string (for TF-IDF input).

    Uses the inverse of encode_text: chr(token) for tokens < 128.
    This recovers the original ASCII characters.
    """
    if hasattr(tokens, 'tolist'):
        tlist = tokens.tolist()
    else:
        tlist = list(tokens)
    return "".join(chr(t) for t in tlist if t > 0)


# ── Toy GPT (IDENTICAL to m2p_composition_n5) ─────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((d,))
        self.eps = eps

    def __call__(self, x):
        rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * rms * self.weight


class Attention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
        scale = self.head_dim ** -0.5
        attn = mx.softmax(q @ k.transpose(0, 1, 3, 2) * scale + mask, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = Attention(d, n_heads)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ToyGPT(nn.Module):
    """Toy GPT: d=256, L=2, 4 heads, vocab=128."""

    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.wpe = nn.Embedding(BLOCK_SIZE + 1, D_MODEL)
        self.blocks = [Block(D_MODEL, N_HEADS) for _ in range(N_LAYERS)]
        self.norm_f = RMSNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm_f(x))

    def get_hidden_states(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        states = []
        for block in self.blocks:
            x = block(x)
            states.append(x)
        return states


# ── Grassmannian A-matrices (IDENTICAL to m2p_composition_n5) ─────────────

def generate_grassmannian_A(n_domains, n_layers, n_modules, d, rank, seed=42):
    total_rank = n_domains * rank
    assert total_rank <= d
    rng = np.random.RandomState(seed)
    A_matrices = {}
    for li in range(n_layers):
        for mi in range(n_modules):
            X = rng.randn(d, total_rank).astype(np.float32)
            Q, _ = np.linalg.qr(X)
            for di in range(n_domains):
                start = di * rank
                A_matrices[(di, li, mi)] = mx.array(Q[:, start:start + rank])
    return A_matrices


def verify_grassmannian_orthogonality(A_matrices, n_domains, n_layers, n_modules):
    cos_values = []
    for li in range(n_layers):
        for mi in range(n_modules):
            for di in range(n_domains):
                for dj in range(di + 1, n_domains):
                    ai = A_matrices[(di, li, mi)].reshape(-1)
                    aj = A_matrices[(dj, li, mi)].reshape(-1)
                    cos = mx.abs(
                        mx.sum(ai * aj) /
                        (mx.linalg.norm(ai) * mx.linalg.norm(aj) + 1e-12)
                    ).item()
                    cos_values.append(cos)
    return {
        "mean_cos": float(np.mean(cos_values)),
        "max_cos": float(np.max(cos_values)),
        "n_pairs": len(cos_values),
    }


# ── LoRA forward pass (IDENTICAL to m2p_composition_n5) ───────────────────

def lora_forward_with_B(base, tokens, A_matrices, domain_id, B_matrices):
    """Forward pass with Grassmannian LoRA applied for given domain."""
    B_batch, T = tokens.shape
    pos = mx.arange(T)
    x = base.wte(tokens) + base.wpe(pos)

    for li, block in enumerate(base.blocks):
        x_norm = block.norm1(x)
        B_b, T_b, C = x_norm.shape
        attn = block.attn
        h, hd = attn.n_heads, attn.head_dim

        def _apply_lora(linear_fn, x_in, li, mi):
            base_out = linear_fn(x_in)
            A = A_matrices[(domain_id, li, mi)]
            B = B_matrices[(li, mi)]
            return base_out + LORA_SCALE * (x_in @ A) @ B

        q = _apply_lora(attn.wq, x_norm, li, 0)
        k = _apply_lora(attn.wk, x_norm, li, 1)
        v = _apply_lora(attn.wv, x_norm, li, 2)

        q = q.reshape(B_b, T_b, h, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B_b, T_b, h, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B_b, T_b, h, hd).transpose(0, 2, 1, 3)

        mask = mx.triu(mx.full((T_b, T_b), float("-inf")), k=1)
        scale_factor = hd ** -0.5
        a_mat = mx.softmax(q @ k.transpose(0, 1, 3, 2) * scale_factor + mask, axis=-1)
        attn_ctx = (a_mat @ v).transpose(0, 2, 1, 3).reshape(B_b, T_b, C)

        attn_out = _apply_lora(attn.wo, attn_ctx, li, 3)
        x = x + attn_out

        x_norm2 = block.norm2(x)
        fc1_base = block.mlp.fc1(x_norm2)
        A_fc1 = A_matrices[(domain_id, li, 4)]
        B_fc1 = B_matrices[(li, 4)]
        fc1_out = fc1_base + LORA_SCALE * (x_norm2 @ A_fc1) @ B_fc1
        mlp_out = block.mlp.fc2(nn.gelu(fc1_out))
        x = x + mlp_out

    return base.lm_head(base.norm_f(x))


# ── SFT training (IDENTICAL to m2p_composition_n5) ────────────────────────

class BMatrices(nn.Module):
    def __init__(self):
        super().__init__()
        for li in range(N_LAYERS):
            for mi in range(N_MODULES):
                d_out = MODULE_OUT_DIMS[mi]
                setattr(self, f"B_{li}_{mi}", mx.zeros((LORA_RANK, d_out)))

    def as_dict(self):
        return {
            (li, mi): getattr(self, f"B_{li}_{mi}")
            for li in range(N_LAYERS) for mi in range(N_MODULES)
        }


def sft_loss_fn(b_container, base, tokens, A_matrices, domain_id):
    B_matrices = b_container.as_dict()
    logits = lora_forward_with_B(base, tokens, A_matrices, domain_id, B_matrices)
    return nn.losses.cross_entropy(logits[:, :-1], tokens[:, 1:], reduction="mean")


# ── M2P Transformer (IDENTICAL to m2p_composition_n5) ─────────────────────

class M2PAttention(nn.Module):
    def __init__(self, d, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        h, hd = self.n_heads, self.head_dim
        q = self.wq(x).reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
        scale = hd ** -0.5
        a = mx.softmax(q @ k.transpose(0, 1, 3, 2) * scale + mask, axis=-1)
        out = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)


class M2PMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class M2PBlock(nn.Module):
    def __init__(self, d, n_heads=4):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = M2PAttention(d, n_heads)
        self.norm2 = RMSNorm(d)
        self.mlp = M2PMLP(d)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class M2PTransformer(nn.Module):
    def __init__(self, d_base=D_MODEL, d_m2p=D_M2P):
        super().__init__()
        self.d_base = d_base
        self.d_m2p = d_m2p
        self.input_proj = nn.Linear(d_base, d_m2p, bias=False)
        self.memory_tokens = mx.random.normal(shape=(N_MEMORY, d_m2p)) * 0.02
        self.pos_embed = nn.Embedding(N_MEMORY, d_m2p)
        self.blocks = [M2PBlock(d_m2p, n_heads=4) for _ in range(M2P_LAYERS)]
        self.norm_f = RMSNorm(d_m2p)
        self.out_heads = {}
        for mi, (mname, d_out) in enumerate(zip(MODULE_NAMES, MODULE_OUT_DIMS)):
            total_out = N_LAYERS * LORA_RANK * d_out
            self.out_heads[mname] = nn.Linear(d_m2p, total_out, bias=False)

    def __call__(self, hidden_states_list):
        layer_encodings = []
        for h in hidden_states_list:
            pooled = mx.mean(h[0], axis=0)
            enc = self.input_proj(pooled)
            layer_encodings.append(enc)

        pos_ids = mx.arange(N_MEMORY)
        memory = self.memory_tokens + self.pos_embed(pos_ids)
        context_enc = mx.mean(mx.stack(layer_encodings, axis=0), axis=0)
        memory = memory + context_enc[None, :]

        x = memory[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)

        pooled_memory = mx.mean(x[0], axis=0)
        B_matrices = {}
        for mi, (mname, d_out) in enumerate(zip(MODULE_NAMES, MODULE_OUT_DIMS)):
            out = self.out_heads[mname](pooled_memory)
            out = out.reshape(N_LAYERS, LORA_RANK, d_out)
            for li in range(N_LAYERS):
                B_matrices[(li, mi)] = out[li]
        return B_matrices


def m2p_ntp_loss(m2p, base, A_matrices, domain_id, tokens):
    hidden_states = base.get_hidden_states(tokens)
    B_matrices = m2p(hidden_states)
    logits = lora_forward_with_B(base, tokens, A_matrices, domain_id, B_matrices)
    return nn.losses.cross_entropy(logits[:, :-1], tokens[:, 1:], reduction="mean")


# ── Evaluation helpers ─────────────────────────────────────────────────────

def eval_ntp_loss(base, batches, A_matrices=None, domain_id=None, B_matrices=None):
    total = 0.0
    n = 0
    for tokens in batches[:50]:
        tokens_2d = tokens[None, :]
        if A_matrices is not None and B_matrices is not None:
            logits = lora_forward_with_B(base, tokens_2d, A_matrices,
                                          domain_id, B_matrices)
        else:
            logits = base(tokens_2d)
        loss = nn.losses.cross_entropy(
            logits[:, :-1], tokens_2d[:, 1:], reduction="mean"
        )
        mx.eval(loss)
        total += loss.item()
        n += 1
        del logits, loss
    return total / max(n, 1)


def load_B_matrices(path: str) -> dict:
    data = np.load(path)
    B_matrices = {}
    for li in range(N_LAYERS):
        for mi in range(N_MODULES):
            key = f"{li}_{mi}"
            B_matrices[(li, mi)] = mx.array(data[key])
    return B_matrices


# ── TF-IDF Sequence-Level Router (NEW — replaces per-token MLP router) ─────

def build_domain_text_corpus(domain_data: dict) -> tuple:
    """Build text corpus from domain token sequences for TF-IDF training.

    Decodes token sequences back to strings.
    Returns: (texts, labels) for all train + val sequences.
    """
    train_texts, train_labels = [], []
    val_texts, val_labels = [], []

    for di, name in enumerate(DOMAIN_NAMES):
        for tokens in domain_data[name]["train"]:
            t = tokens_to_string(np.array(tokens))
            if t:
                train_texts.append(t)
                train_labels.append(di)
        for tokens in domain_data[name]["val"][:20]:
            t = tokens_to_string(np.array(tokens))
            if t:
                val_texts.append(t)
                val_labels.append(di)

    return (train_texts, np.array(train_labels),
            val_texts, np.array(val_labels))


def phase_train_tfidf_router(domain_data: dict) -> tuple:
    """Train TF-IDF + logistic regression sequence-level router.

    Theorem 1 (MATH.md): This router is trivially invariant to model
    distribution — it operates on input text only, before any forward pass.
    Theorem 2 (MATH.md): Format strings in toy data provide TF-IDF-separable
    features even for domains sharing the same character alphabet.

    Returns: (vectorizer, clf, routing_accuracy, per_domain_accuracy)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    log("\n=== Phase: Train TF-IDF Sequence Router ===")

    train_texts, train_labels, val_texts, val_labels = \
        build_domain_text_corpus(domain_data)

    log(f"  Train: {len(train_texts)} sequences, Val: {len(val_texts)} sequences")

    # TF-IDF settings: character-level n-grams to capture format strings
    # (">", "*", "=") even across word boundaries.
    # Match contrastive_routing_n5 settings (Finding #207: 90% accuracy).
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",   # character n-grams (better for format tokens)
        ngram_range=(1, 3),   # unigrams to trigrams
        max_features=2000,    # sufficient for vocab=128 char set
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts) if val_texts else X_train[:0]

    log(f"  TF-IDF features: {X_train.shape[1]}")

    clf = LogisticRegression(
        max_iter=500,
        C=1.0,
        solver="lbfgs",
        random_state=SEED,
    )
    clf.fit(X_train, train_labels)

    # Training accuracy
    train_preds = clf.predict(X_train)
    train_acc = float(np.mean(train_preds == train_labels))
    log(f"  Train accuracy: {train_acc:.1%}")

    # Validation accuracy + per-domain breakdown
    val_acc = 0.0
    per_domain_acc = {}
    if len(val_texts) > 0:
        val_preds = clf.predict(X_val)
        val_acc = float(np.mean(val_preds == val_labels))
        log(f"  Val accuracy: {val_acc:.1%} (K867 threshold: 70%)")

        for di, name in enumerate(DOMAIN_NAMES):
            mask = val_labels == di
            if mask.sum() > 0:
                acc = float(np.mean(val_preds[mask] == val_labels[mask]))
                per_domain_acc[name] = round(acc, 4)
                log(f"    {name:12s}: {acc:.1%} ({int(mask.sum())} samples)")
    else:
        # Fallback: use train accuracy as estimate
        val_acc = train_acc
        for name in DOMAIN_NAMES:
            per_domain_acc[name] = round(train_acc, 4)

    return vectorizer, clf, round(val_acc, 4), per_domain_acc


def route_sequence(vectorizer, clf, tokens) -> int:
    """Route a token sequence to a domain using TF-IDF + logistic regression.

    Args:
      vectorizer: fitted TfidfVectorizer
      clf: fitted LogisticRegression
      tokens: 1D token array (decoded to string internally)

    Returns: predicted domain index (0-4)
    """
    text = tokens_to_string(np.array(tokens))
    X = vectorizer.transform([text])
    return int(clf.predict(X)[0])


# ── Composition forward (single adapter, no router MLP) ───────────────────

def single_adapter_forward(base, tokens, A_matrices, domain_id, B_matrices):
    """Forward pass applying a SINGLE domain's adapter (hard routing).

    This replaces the soft-routing composed_forward from m2p_composition_n5.
    Routing decision is made BEFORE calling this function (by TF-IDF or oracle).
    """
    return lora_forward_with_B(base, tokens, A_matrices, domain_id, B_matrices)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def phase_generate_data(rng: np.random.RandomState) -> dict:
    """Generate train/val data for all 5 domains."""
    log("\n=== Phase: Generate Data ===")
    domain_data = {}
    n_per_domain = 500 if not SMOKE_TEST else 60
    for di, name in enumerate(DOMAIN_NAMES):
        texts = gen_domain_data(di, n_per_domain, rng)
        split = int(0.8 * len(texts))
        domain_data[name] = {
            "train": make_batches(texts[:split]),
            "val": make_batches(texts[split:]),
            "texts": texts,
            "domain_id": di,
        }
        log(f"  {name}: {len(domain_data[name]['train'])} train, "
            f"{len(domain_data[name]['val'])} val")
    return domain_data


def phase_pretrain_base(domain_data: dict) -> tuple:
    """Pre-train ToyGPT on all domains."""
    log("\n=== Phase: Pre-train Base Model ===")
    mx.random.seed(SEED)

    base = ToyGPT()
    mx.eval(base.parameters())

    all_train = []
    for name in DOMAIN_NAMES:
        all_train.extend(domain_data[name]["train"])

    optimizer = opt.Adam(learning_rate=LR)

    def loss_fn(model, tokens):
        tokens_2d = tokens[None, :]
        logits = model(tokens_2d)
        return nn.losses.cross_entropy(
            logits[:, :-1], tokens_2d[:, 1:], reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(base, loss_fn)

    gc.disable()
    for step in range(BASE_STEPS):
        tokens = all_train[step % len(all_train)]
        loss, grads = loss_and_grad(base, tokens)
        optimizer.update(base, grads)
        mx.eval(base.parameters(), optimizer.state, loss)
        if (step + 1) % max(1, BASE_STEPS // 4) == 0:
            log(f"  Step {step+1}/{BASE_STEPS}: loss={loss.item():.4f}")
    gc.enable()

    base.freeze()
    base_losses = {}
    for name in DOMAIN_NAMES:
        bl = eval_ntp_loss(base, domain_data[name]["val"])
        base_losses[name] = round(bl, 4)
    log(f"  Base losses: {base_losses}")

    # Save weights
    base_weights_path = EXPERIMENT_DIR / "base_weights.npz"
    weights_dict = {}
    for k, v in tree_flatten(base.parameters()):
        weights_dict[k.replace(".", "_")] = np.array(v)
    np.savez(str(base_weights_path), **weights_dict)
    log(f"  Saved base weights: {base_weights_path}")

    cleanup(optimizer)
    return base, base_losses, str(base_weights_path)


def phase_grassmannian(base: ToyGPT) -> tuple:
    """Generate and verify Grassmannian A-matrices."""
    log("\n=== Phase: Grassmannian A-matrices ===")
    A_matrices = generate_grassmannian_A(
        N_DOMAINS, N_LAYERS, N_MODULES, D_MODEL, LORA_RANK, seed=SEED
    )
    ortho = verify_grassmannian_orthogonality(
        A_matrices, N_DOMAINS, N_LAYERS, N_MODULES
    )
    log(f"  Orthogonality: mean|cos|={ortho['mean_cos']:.6f}, "
        f"max|cos|={ortho['max_cos']:.6f} ({ortho['n_pairs']} pairs)")
    assert ortho["max_cos"] < 1e-5, \
        f"Grassmannian guarantee failed: max|cos|={ortho['max_cos']}"
    return A_matrices, ortho


def phase_sft_domain(domain_name, domain_id, domain_data, base,
                      A_matrices, base_loss) -> dict:
    """Train SFT LoRA adapter for one domain."""
    log(f"  SFT {domain_name} (domain {domain_id})...")

    # Check if adapter already exists (from m2p_composition_n5 or this run)
    local_path = ADAPTER_DIR / f"sft_{domain_name}.npz"
    source_path = SOURCE_ADAPTER_DIR / f"sft_{domain_name}.npz"

    if local_path.exists():
        log(f"    Reusing existing local adapter: {local_path}")
        b_container = BMatrices()
        data = np.load(str(local_path))
        for li in range(N_LAYERS):
            for mi in range(N_MODULES):
                setattr(b_container, f"B_{li}_{mi}",
                        mx.array(data[f"{li}_{mi}"]))
        mx.eval(b_container.parameters())
    elif source_path.exists():
        log(f"    Reusing adapter from m2p_composition_n5: {source_path}")
        # Copy to local dir
        import shutil
        shutil.copy(str(source_path), str(local_path))
        b_container = BMatrices()
        data = np.load(str(local_path))
        for li in range(N_LAYERS):
            for mi in range(N_MODULES):
                setattr(b_container, f"B_{li}_{mi}",
                        mx.array(data[f"{li}_{mi}"]))
        mx.eval(b_container.parameters())
    else:
        # Train from scratch
        b_container = BMatrices()
        mx.eval(b_container.parameters())
        optimizer = opt.Adam(learning_rate=SFT_LR)

        def _loss(b_cont, tokens):
            return sft_loss_fn(b_cont, base, tokens[None, :], A_matrices, domain_id)

        grad_fn = nn.value_and_grad(b_container, _loss)
        train_batches = domain_data["train"]

        gc.disable()
        for step in range(SFT_STEPS):
            tokens = train_batches[step % len(train_batches)]
            loss, grads = grad_fn(b_container, tokens)
            optimizer.update(b_container, grads)
            mx.eval(b_container.parameters(), optimizer.state, loss)
        gc.enable()
        cleanup(optimizer)

        np_dict = {f"{li}_{mi}": np.array(getattr(b_container, f"B_{li}_{mi}"))
                   for li in range(N_LAYERS) for mi in range(N_MODULES)}
        np.savez(str(local_path), **np_dict)

    B_matrices = b_container.as_dict()
    sft_loss = eval_ntp_loss(base, domain_data["val"],
                              A_matrices, domain_id, B_matrices)
    quality_ratio = ((base_loss - sft_loss) / base_loss) if base_loss > 0.01 else 0.0
    log(f"    SFT loss={sft_loss:.4f} base={base_loss:.4f} "
        f"improvement={quality_ratio:.1%}")

    cleanup(b_container)
    return {"sft_loss": round(sft_loss, 4), "save_path": str(local_path)}


def phase_sft_all_domains(domain_data, base, A_matrices, base_losses) -> dict:
    log("\n=== Phase: SFT Baselines ===")
    sft_results = {}
    for di, name in enumerate(DOMAIN_NAMES):
        result = phase_sft_domain(
            name, di, domain_data[name], base, A_matrices, base_losses[name]
        )
        sft_results[name] = result
    return sft_results


def phase_m2p_domain(domain_name, domain_id, domain_data, base,
                      A_matrices, base_loss, sft_loss) -> dict:
    """Train M2P for ONE domain independently."""
    log(f"  M2P {domain_name} (domain {domain_id})...")

    local_path = ADAPTER_DIR / f"m2p_{domain_name}.npz"
    source_path = SOURCE_ADAPTER_DIR / f"m2p_{domain_name}.npz"

    trained_fresh = False

    if local_path.exists():
        log(f"    Reusing existing local M2P adapter: {local_path}")
    elif source_path.exists():
        log(f"    Reusing M2P adapter from m2p_composition_n5: {source_path}")
        import shutil
        shutil.copy(str(source_path), str(local_path))
    else:
        # Train M2P from scratch
        trained_fresh = True
        m2p = M2PTransformer(d_base=D_MODEL, d_m2p=D_M2P)
        mx.eval(m2p.parameters())
        m2p_param_count = sum(p.size for _, p in tree_flatten(m2p.parameters()))
        log(f"    M2P params: {m2p_param_count:,}")

        optimizer = opt.Adam(learning_rate=M2P_LR)

        def _loss(m2p_model, tokens):
            return m2p_ntp_loss(m2p_model, base, A_matrices, domain_id,
                                 tokens[None, :])

        grad_fn = nn.value_and_grad(m2p, _loss)
        train_batches = domain_data["train"]

        gc.disable()
        for step in range(M2P_STEPS):
            tokens = train_batches[step % len(train_batches)]
            loss, grads = grad_fn(m2p, tokens)
            optimizer.update(m2p, grads)
            mx.eval(m2p.parameters(), optimizer.state, loss)
            if (step + 1) % max(1, M2P_STEPS // 4) == 0:
                log(f"    Step {step+1}/{M2P_STEPS}: loss={loss.item():.4f}")
        gc.enable()
        cleanup(optimizer)

        # Save representative B-matrices (from first training context)
        context_tokens = domain_data["train"][0][None, :]
        hidden_states = base.get_hidden_states(context_tokens)
        B_matrices = m2p(hidden_states)
        mx.eval(*[B_matrices[(li, mi)] for li in range(N_LAYERS) for mi in range(N_MODULES)])

        np_dict = {f"{li}_{mi}": np.array(B_matrices[(li, mi)])
                   for li in range(N_LAYERS) for mi in range(N_MODULES)}
        np.savez(str(local_path), **np_dict)
        cleanup(m2p)

    # Load and evaluate
    B_matrices = load_B_matrices(str(local_path))
    m2p_loss = eval_ntp_loss(base, domain_data["val"],
                              A_matrices, domain_id, B_matrices)

    quality_ratio = 0.0
    if (base_loss - sft_loss) > 0.01:
        quality_ratio = (base_loss - m2p_loss) / (base_loss - sft_loss)
    log(f"    M2P loss={m2p_loss:.4f} SFT={sft_loss:.4f} "
        f"quality={quality_ratio:.1%}")

    cleanup(B_matrices)
    return {
        "m2p_loss": round(m2p_loss, 4),
        "quality_ratio": round(quality_ratio, 3),
        "save_path": str(local_path),
        "trained_fresh": trained_fresh,
    }


def phase_m2p_all_domains(domain_data, base, A_matrices, base_losses,
                            sft_results) -> dict:
    log("\n=== Phase: M2P Adapters (per-domain) ===")
    m2p_results = {}
    for di, name in enumerate(DOMAIN_NAMES):
        result = phase_m2p_domain(
            name, di, domain_data[name], base,
            A_matrices, base_losses[name], sft_results[name]["sft_loss"]
        )
        m2p_results[name] = result
    return m2p_results


def phase_routing_accuracy(domain_data, vectorizer, clf) -> dict:
    """Evaluate TF-IDF routing accuracy on held-out val sequences.

    This is a KEY new phase replacing phase_train_router from m2p_composition_n5.
    Measures: per-domain accuracy, overall accuracy, confusion matrix.
    """
    log("\n=== Phase: TF-IDF Routing Accuracy Evaluation ===")

    correct = 0
    total = 0
    per_domain = {}
    confusion = np.zeros((N_DOMAINS, N_DOMAINS), dtype=int)

    for di, name in enumerate(DOMAIN_NAMES):
        dom_correct = 0
        dom_total = 0
        for tokens in domain_data[name]["val"]:
            pred_di = route_sequence(vectorizer, clf, tokens)
            confusion[di, pred_di] += 1
            if pred_di == di:
                dom_correct += 1
                correct += 1
            total += 1
            dom_total += 1

        acc = dom_correct / max(dom_total, 1)
        per_domain[name] = round(acc, 4)
        log(f"  {name:12s}: {acc:.1%} ({dom_correct}/{dom_total})")

    overall = correct / max(total, 1)
    log(f"\n  Overall routing accuracy: {overall:.1%} ({correct}/{total})")
    log(f"  Baseline (per-token MLP): 36.6% (K852 FAIL in m2p_composition_n5)")
    log(f"  K867 threshold: 70%")

    # Confusion matrix display
    log("\n  Confusion matrix (rows=true, cols=pred):")
    header = "           " + "  ".join(f"{n[:4]:>4}" for n in DOMAIN_NAMES)
    log(f"  {header}")
    for i, name in enumerate(DOMAIN_NAMES):
        row = "  ".join(f"{confusion[i, j]:>4}" for j in range(N_DOMAINS))
        log(f"  {name[:10]:>10}:  {row}")

    return {
        "overall": round(overall, 4),
        "per_domain": per_domain,
        "confusion_matrix": confusion.tolist(),
        "n_correct": correct,
        "n_total": total,
    }


def phase_oracle_composition(domain_data, base, A_matrices, m2p_results,
                               sft_results, base_losses) -> dict:
    """Evaluate composition with ORACLE routing (ground-truth domain labels).

    This measures the CEILING quality achievable with perfect routing.
    K869: oracle quality > 80% of SFT.

    Oracle routing: at inference, use true domain label to select adapter.
    """
    log("\n=== Phase: Oracle-Routed Composition ===")

    oracle_quality = {}
    for di, name in enumerate(DOMAIN_NAMES):
        B_matrices = load_B_matrices(m2p_results[name]["save_path"])

        total = 0.0
        n = 0
        for tokens in domain_data[name]["val"][:30]:
            tokens_2d = tokens[None, :]
            logits = lora_forward_with_B(base, tokens_2d, A_matrices, di, B_matrices)
            loss = nn.losses.cross_entropy(
                logits[:, :-1], tokens_2d[:, 1:], reduction="mean"
            )
            mx.eval(loss)
            total += loss.item()
            n += 1
            del logits, loss

        oracle_loss = total / max(n, 1)
        sft_loss = sft_results[name]["sft_loss"]
        base_loss = base_losses[name]

        # Quality ratio: how much improvement vs SFT (oracle = M2P single-domain)
        if (base_loss - sft_loss) > 0.01:
            q = (base_loss - oracle_loss) / (base_loss - sft_loss)
        else:
            q = 0.0

        oracle_quality[name] = {
            "oracle_loss": round(oracle_loss, 4),
            "sft_loss": sft_loss,
            "base_loss": base_loss,
            "quality_ratio": round(q, 3),
        }
        log(f"  {name:12s}: oracle={oracle_loss:.4f} SFT={sft_loss:.4f} "
            f"quality={q:.1%}")

        cleanup(B_matrices)

    quality_ratios = [oracle_quality[n]["quality_ratio"] for n in DOMAIN_NAMES]
    median_q = float(np.median(quality_ratios))
    mean_q = float(np.mean(quality_ratios))
    log(f"\n  Oracle quality: median={median_q:.1%} mean={mean_q:.1%}")
    log(f"  K869 threshold: >80% of SFT")

    return {
        "per_domain": oracle_quality,
        "median_quality": round(median_q, 4),
        "mean_quality": round(mean_q, 4),
    }


def phase_tfidf_composition(domain_data, base, A_matrices, m2p_results,
                              sft_results, base_losses,
                              vectorizer, clf) -> dict:
    """Evaluate composition with TF-IDF routing.

    For each validation sequence:
    1. Convert tokens → string
    2. TF-IDF vectorize → logistic regression predict → domain d
    3. Load M2P adapter for domain d
    4. Apply single-adapter forward pass
    5. Measure NTP loss vs SFT

    K868: TF-IDF composition quality > 70% of SFT.
    """
    log("\n=== Phase: TF-IDF-Routed Composition ===")

    # Pre-load all B-matrices (5 adapters, each small: rank*d_out per slot)
    all_B = {}
    for di, name in enumerate(DOMAIN_NAMES):
        all_B[di] = load_B_matrices(m2p_results[name]["save_path"])

    tfidf_quality = {}
    routing_stats = {name: {"correct": 0, "total": 0} for name in DOMAIN_NAMES}

    for di, name in enumerate(DOMAIN_NAMES):
        total_loss = 0.0
        n = 0

        for tokens in domain_data[name]["val"][:30]:
            # Routing decision (sequence-level, text-only — Theorem 1)
            pred_di = route_sequence(vectorizer, clf, tokens)
            routing_stats[name]["total"] += 1
            if pred_di == di:
                routing_stats[name]["correct"] += 1

            # Apply predicted domain's adapter
            tokens_2d = tokens[None, :]
            B_matrices = all_B[pred_di]
            logits = lora_forward_with_B(base, tokens_2d, A_matrices,
                                          pred_di, B_matrices)
            loss = nn.losses.cross_entropy(
                logits[:, :-1], tokens_2d[:, 1:], reduction="mean"
            )
            mx.eval(loss)
            total_loss += loss.item()
            n += 1
            del logits, loss

        comp_loss = total_loss / max(n, 1)
        sft_loss = sft_results[name]["sft_loss"]
        base_loss = base_losses[name]

        if (base_loss - sft_loss) > 0.01:
            q = (base_loss - comp_loss) / (base_loss - sft_loss)
        else:
            q = 0.0

        acc = routing_stats[name]["correct"] / max(routing_stats[name]["total"], 1)
        tfidf_quality[name] = {
            "comp_loss": round(comp_loss, 4),
            "sft_loss": sft_loss,
            "base_loss": base_loss,
            "quality_ratio": round(q, 3),
            "routing_acc_here": round(acc, 4),
        }
        log(f"  {name:12s}: comp={comp_loss:.4f} SFT={sft_loss:.4f} "
            f"quality={q:.1%} routing={acc:.0%}")

    # Cleanup B matrices
    for B in all_B.values():
        del B

    quality_ratios = [tfidf_quality[n]["quality_ratio"] for n in DOMAIN_NAMES]
    median_q = float(np.median(quality_ratios))
    mean_q = float(np.mean(quality_ratios))
    log(f"\n  TF-IDF composition: median={median_q:.1%} mean={mean_q:.1%}")
    log(f"  K868 threshold: >70% of SFT")

    return {
        "per_domain": tfidf_quality,
        "median_quality": round(median_q, 4),
        "mean_quality": round(mean_q, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("M2P TF-IDF Routing N=5 — Distribution-Agnostic Sequence Routing")
    log(f"SMOKE_TEST={SMOKE_TEST}")
    log("=" * 70)
    log_memory("start")

    mx.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    # ── Data ──────────────────────────────────────────────────────────────
    domain_data = phase_generate_data(rng)
    log_memory("after data")

    # ── Base model ────────────────────────────────────────────────────────
    base, base_losses, base_weights_path = phase_pretrain_base(domain_data)
    log_memory("after base pretrain")

    # ── Grassmannian A-matrices ───────────────────────────────────────────
    A_matrices, ortho_result = phase_grassmannian(base)
    log_memory("after grassmannian")

    # ── SFT baselines ────────────────────────────────────────────────────
    sft_results = phase_sft_all_domains(domain_data, base, A_matrices, base_losses)
    log_memory("after SFT")

    # ── M2P adapters (reuse from m2p_composition_n5 if available) ─────────
    m2p_results = phase_m2p_all_domains(
        domain_data, base, A_matrices, base_losses, sft_results
    )
    log_memory("after M2P")

    # ── TF-IDF Router (NEW — replaces per-token MLP router) ───────────────
    vectorizer, clf, tfidf_train_acc, tfidf_train_per_domain = \
        phase_train_tfidf_router(domain_data)
    log_memory("after tfidf router training")

    # ── Routing accuracy evaluation (K867) ────────────────────────────────
    routing_eval = phase_routing_accuracy(domain_data, vectorizer, clf)
    log_memory("after routing eval")

    # ── Oracle composition (K869) ─────────────────────────────────────────
    oracle_results = phase_oracle_composition(
        domain_data, base, A_matrices, m2p_results, sft_results, base_losses
    )
    log_memory("after oracle composition")

    # ── TF-IDF-routed composition (K868) ──────────────────────────────────
    tfidf_results = phase_tfidf_composition(
        domain_data, base, A_matrices, m2p_results, sft_results, base_losses,
        vectorizer, clf
    )
    log_memory("after tfidf composition")

    # ── Kill criteria ─────────────────────────────────────────────────────
    routing_acc = routing_eval["overall"]
    median_tfidf_quality = tfidf_results["median_quality"]
    median_oracle_quality = oracle_results["median_quality"]

    k867_pass = routing_acc > 0.70
    k868_pass = median_tfidf_quality > 0.70
    k869_pass = median_oracle_quality > 0.80

    # ── SFT quality summary (for quality ratio computation) ───────────────
    sft_quality_per_domain = {}
    for name in DOMAIN_NAMES:
        sft_quality_per_domain[name] = {
            "sft_loss": sft_results[name]["sft_loss"],
            "base_loss": base_losses[name],
        }

    # ── Results assembly ──────────────────────────────────────────────────
    results = {
        "experiment": "exp_m2p_tfidf_routing_n5",
        "total_time_s": round(time.time() - t0, 1),
        "smoke_test": SMOKE_TEST,
        # Routing results (K867)
        "routing_accuracy": routing_acc,
        "routing_per_domain": routing_eval["per_domain"],
        "routing_confusion_matrix": routing_eval["confusion_matrix"],
        # Oracle quality ceiling (K869)
        "oracle_quality": {n: oracle_results["per_domain"][n]["quality_ratio"]
                           for n in DOMAIN_NAMES},
        "median_oracle_quality": median_oracle_quality,
        "mean_oracle_quality": oracle_results["mean_quality"],
        # TF-IDF composition quality (K868)
        "tfidf_quality": {n: tfidf_results["per_domain"][n]["quality_ratio"]
                          for n in DOMAIN_NAMES},
        "median_tfidf_quality": median_tfidf_quality,
        "mean_tfidf_quality": tfidf_results["mean_quality"],
        # SFT reference
        "sft_quality": sft_quality_per_domain,
        # M2P per-domain quality (should reproduce Finding #351)
        "m2p_quality_per_domain": {n: m2p_results[n]["quality_ratio"]
                                    for n in DOMAIN_NAMES},
        # Grassmannian verification
        "grassmannian_A_cos_max": ortho_result["max_cos"],
        # Base model losses
        "base_losses": base_losses,
        "sft_losses": {n: sft_results[n]["sft_loss"] for n in DOMAIN_NAMES},
        "m2p_losses": {n: m2p_results[n]["m2p_loss"] for n in DOMAIN_NAMES},
        # Kill criteria
        "k867_pass": k867_pass,
        "k868_pass": k868_pass,
        "k869_pass": k869_pass,
        "all_pass": k867_pass and k868_pass and k869_pass,
        # Predictions vs measurements (for PAPER.md prediction table)
        "predictions_vs_measurements": {
            "routing_accuracy": {
                "predicted_gte": 0.70,
                "predicted_expected": 0.90,
                "measured": routing_acc,
                "pass": k867_pass,
            },
            "tfidf_composition_quality": {
                "predicted_gte": 0.70,
                "measured": median_tfidf_quality,
                "pass": k868_pass,
            },
            "oracle_composition_quality": {
                "predicted_gte": 0.80,
                "measured": median_oracle_quality,
                "pass": k869_pass,
            },
        },
    }

    # ── Summary report ────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)
    log(f"Grassmannian A cos_max: {ortho_result['max_cos']:.8f} (should be ~0)")
    log("")
    log(f"TF-IDF routing accuracy: {routing_acc:.1%}")
    log(f"  K867: {'PASS' if k867_pass else 'FAIL'} (threshold: >70%, baseline: 36.6%)")
    log("")
    log(f"Oracle composition quality (median): {median_oracle_quality:.1%} of SFT")
    log(f"  K869: {'PASS' if k869_pass else 'FAIL'} (threshold: >80%)")
    log("")
    log(f"TF-IDF composition quality (median): {median_tfidf_quality:.1%} of SFT")
    log(f"  K868: {'PASS' if k868_pass else 'FAIL'} (threshold: >70%)")
    log("")
    log("Per-domain breakdown:")
    for name in DOMAIN_NAMES:
        ra = routing_eval["per_domain"].get(name, 0.0)
        oq = oracle_results["per_domain"][name]["quality_ratio"]
        tq = tfidf_results["per_domain"][name]["quality_ratio"]
        log(f"  {name:12s}: route={ra:.0%}  oracle={oq:.1%}  tfidf={tq:.1%}")
    log("")
    log(f"OVERALL: {'ALL PASS' if results['all_pass'] else 'PARTIAL/FAIL'} "
        f"in {results['total_time_s']}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
