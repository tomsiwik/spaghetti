#!/usr/bin/env python3
"""Multi-cycle promotion: 3 cycles on toy 2-layer transformer with Grassmannian A-slots.

Kill criteria:
  K928: All domain qualities >= 80% of SFT after 3 cycles (on promoted base, no adapter)
  K929: No domain degrades >20% from post-promotion quality across subsequent cycles
  K930 (KILL): Any domain drops below 50% on promoted base

Theorem (MATH.md): With orthogonal Grassmannian A-slots, sequential promotion gives
exact Pythagorean norm growth: ||W_K - W_0||_F = sqrt(sum_k ||dW_k||_F^2).
"""

import gc
import json
import math
import os
import time
from functools import partial
from pathlib import Path

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Config
D_MODEL = 128
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS       # 32
N_LAYERS = 2
VOCAB_SIZE = 32
SEQ_LEN = 4                         # [task_marker, a, b, result]
LORA_RANK = 4
N_DOMAINS = 3
LORA_SCALE = 1.0

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
N_PRETRAIN_STEPS = 100 if IS_SMOKE else 800   # base model pre-training
N_TRAIN_STEPS    = 100 if IS_SMOKE else 600   # per-domain LoRA fine-tune
N_EVAL = 20 if IS_SMOKE else 100
LR = 5e-4
GRAD_CLIP = 1.0
SEED = 42

# Vocabulary layout (VOCAB_SIZE = 32, no conflicts):
# 0–2:  task markers (domain 0=add, 1=sub, 2=mul)
# 3–12: operands 0–9 (token = value + 3)
# 13–22: results 0–9 (token = value + 13)
TASK_IDS = {"add": 0, "sub": 1, "mul": 2}
DOMAINS = ["add", "sub", "mul"]
OP_OFFSET = 3
RES_OFFSET = 13
MOD = 10


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    a = mx.get_active_memory() / 1e9
    c = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_data(domain: str, n: int, seed: int = 42) -> mx.array:
    """Generate (n, 4) integer array: [task_id, a_tok, b_tok, result_tok]."""
    rng = np.random.default_rng(seed)
    pairs = [(a, b) for a in range(MOD) for b in range(MOD)]  # 100 pairs
    idxs = rng.choice(len(pairs), min(n, len(pairs)), replace=False)
    task_id = TASK_IDS[domain]
    rows = []
    for i in idxs:
        a, b = pairs[i]
        if domain == "add":
            r = (a + b) % MOD
        elif domain == "sub":
            r = (a - b) % MOD
        else:
            r = (a * b) % MOD
        rows.append([task_id, a + OP_OFFSET, b + OP_OFFSET, r + RES_OFFSET])
    return mx.array(rows, dtype=mx.int32)


# ---------------------------------------------------------------------------
# Tiny 2-layer causal transformer
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def __call__(self, x, B_q=None, A_q=None, B_v=None, A_v=None, mask=None):
        # x: (batch, T, D)
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim

        q = self.q_proj(x)
        if B_q is not None:
            # LoRA: x @ A_q @ B_q where A_q:(D,r), B_q:(r,D)
            q = q + LORA_SCALE * (x @ A_q) @ B_q

        k = self.k_proj(x)

        v = self.v_proj(x)
        if B_v is not None:
            v = v + LORA_SCALE * (x @ A_v) @ B_v

        # Multi-head attention
        q = q.reshape(B, T, H, hd).transpose(0, 2, 1, 3)  # (B,H,T,hd)
        k = k.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, hd).transpose(0, 2, 1, 3)

        scale = hd ** -0.5
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # (B,H,T,T)
        if mask is not None:
            scores = scores + mask
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ff1 = nn.Linear(d_model, d_model * 2, bias=False)
        self.ff2 = nn.Linear(d_model * 2, d_model, bias=False)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def __call__(self, x, B_q=None, A_q=None, B_v=None, A_v=None, mask=None):
        x = x + self.attn(self.norm1(x), B_q=B_q, A_q=A_q, B_v=B_v, A_v=A_v, mask=mask)
        x = x + self.ff2(mx.maximum(self.ff1(self.norm2(x)), 0))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, x, lora_B_q=None, lora_A_q=None, lora_B_v=None, lora_A_v=None):
        # x: (batch, T) int32
        B, T = x.shape
        h = self.embed(x)           # (batch, T, D)

        # Causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)

        for i, block in enumerate(self.blocks):
            Bq = lora_B_q[i] if lora_B_q is not None else None
            Aq = lora_A_q[i] if lora_A_q is not None else None
            Bv = lora_B_v[i] if lora_B_v is not None else None
            Av = lora_A_v[i] if lora_A_v is not None else None
            h = block(h, B_q=Bq, A_q=Aq, B_v=Bv, A_v=Av, mask=mask)

        h = self.norm(h)
        return self.lm_head(h)   # (batch, T, vocab)


# ---------------------------------------------------------------------------
# LoRA trainable module (just B matrices)
# ---------------------------------------------------------------------------

class LoRAParams(nn.Module):
    def __init__(self, rank: int, d_model: int, n_layers: int):
        super().__init__()
        # B: (rank, d_model) — initialized to zero
        # LoRA forward: x @ A @ B where A:(D,r), B:(r,D)
        self.B_q = [mx.zeros((rank, d_model)) for _ in range(n_layers)]
        self.B_v = [mx.zeros((rank, d_model)) for _ in range(n_layers)]


# ---------------------------------------------------------------------------
# Grassmannian A-slot generation
# ---------------------------------------------------------------------------

def make_grassmannian_slots(d_model: int, rank: int, n_domains: int, seed: int):
    """
    QR decomposition of (d_model, n_domains*rank) random matrix.
    Returns list of n_domains arrays, each (d_model, rank), with orthonormal columns.
    For i != j: A_i.T @ A_j = 0 (exactly).
    """
    np.random.seed(seed)
    X = np.random.randn(d_model, n_domains * rank).astype(np.float32)
    Q, _ = np.linalg.qr(X)  # thin QR: Q shape (d_model, n_domains*rank)
    slots = []
    for i in range(n_domains):
        col = Q[:, i*rank:(i+1)*rank]  # (d_model, rank)
        slots.append(mx.array(col))
    return slots


# ---------------------------------------------------------------------------
# Pre-training phase (base model on all domains jointly, no adapters)
# ---------------------------------------------------------------------------

def phase_pretrain(base_model, all_train_data):
    """Pre-train base model on all 3 domains jointly (no LoRA) for N_PRETRAIN_STEPS.

    This gives the base model useful arithmetic representations before LoRA fine-tuning.
    """
    log(f"[pretrain] Training base on all domains for {N_PRETRAIN_STEPS} steps...")
    base_model.unfreeze()

    # Combine all domain data
    all_data = mx.concatenate([all_train_data[d] for d in DOMAINS], axis=0)
    n_data = all_data.shape[0]
    mx.eval(all_data)

    optimizer = optim.AdamW(learning_rate=LR, weight_decay=1e-4)

    def base_loss_fn(model, x):
        logits = model(x)
        targets = x[:, 1:]
        preds = logits[:, :-1, :]
        return nn.losses.cross_entropy(preds.reshape(-1, VOCAB_SIZE), targets.reshape(-1)).mean()

    vg_fn = nn.value_and_grad(base_model, base_loss_fn)

    rng = np.random.default_rng(SEED + 999)
    BATCH = 32

    gc.disable()
    last_loss = float("nan")
    for step in range(N_PRETRAIN_STEPS):
        idxs = rng.integers(0, n_data, BATCH)
        x_batch = all_data[mx.array(idxs)]

        loss, grads = vg_fn(base_model, x_batch)
        # Gradient clipping
        grads, _ = optim.clip_grad_norm(grads, max_norm=GRAD_CLIP)
        optimizer.update(base_model, grads)
        mx.eval(base_model.parameters(), optimizer.state, loss)
        last_loss = loss.item()
        if step % 200 == 0:
            log(f"  [pretrain] step {step}: loss={last_loss:.4f}")
    gc.enable()
    log(f"  [pretrain] final_loss={last_loss:.4f}")
    cleanup(optimizer, all_data)
    base_model.freeze()


# ---------------------------------------------------------------------------
# Training phase (per-domain, per-cycle)
# ---------------------------------------------------------------------------

def phase_train_adapter(base_model, A_q_slot, A_v_slot, train_data, domain):
    """Train LoRA B matrices for one domain on the current base model.

    Returns (B_q_numpy, B_v_numpy, sft_loss) — B arrays saved as numpy for disk storage.
    """
    log(f"  [train] domain={domain}, steps={N_TRAIN_STEPS}")
    base_model.freeze()  # base is frozen

    lora = LoRAParams(rank=LORA_RANK, d_model=D_MODEL, n_layers=N_LAYERS)
    mx.eval(lora.parameters())

    # A-slots: same for all layers (shared Grassmannian slot per projection type)
    A_q_all = [A_q_slot] * N_LAYERS  # list of (D_MODEL, LORA_RANK)
    A_v_all = [A_v_slot] * N_LAYERS

    optimizer = optim.AdamW(learning_rate=LR, weight_decay=0.0)

    def loss_fn(lora, base_model, A_q, A_v, x):
        logits = base_model(
            x, lora_B_q=lora.B_q, lora_A_q=A_q, lora_B_v=lora.B_v, lora_A_v=A_v
        )
        # Predict position 3 (result token) given positions 0–2
        # Cross-entropy on position 3 only (index 3 prediction from context [0,1,2,3])
        # Actually: predict token at each position; focus on position 2 → position 3
        # CE on all positions (autoregressive), but we only care about position 3
        targets = x[:, 1:]                   # (batch, 3): positions 1,2,3
        preds = logits[:, :-1, :]            # (batch, 3, vocab)
        loss = nn.losses.cross_entropy(preds.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        return loss.mean()

    vg_fn = nn.value_and_grad(lora, loss_fn)

    n_data = train_data.shape[0]
    rng = np.random.default_rng(SEED)
    BATCH = 16

    gc.disable()
    last_loss = float("nan")
    for step in range(N_TRAIN_STEPS):
        idxs = rng.integers(0, n_data, BATCH)
        x_batch = train_data[mx.array(idxs)]

        loss, grads = vg_fn(lora, base_model, A_q_all, A_v_all, x_batch)
        grads = optim.clip_grad_norm(grads, max_norm=GRAD_CLIP)
        optimizer.update(lora, grads)
        mx.eval(lora.parameters(), optimizer.state, loss)
        last_loss = loss.item()

        if step % 100 == 0:
            log(f"    step {step}: loss={last_loss:.4f}")
    gc.enable()

    log(f"  [train] final_loss={last_loss:.4f}")

    # Save B matrices as numpy (disk-safe)
    B_q_np = [np.array(b) for b in lora.B_q]
    B_v_np = [np.array(b) for b in lora.B_v]

    cleanup(lora, optimizer)
    base_model.unfreeze()
    return B_q_np, B_v_np, last_loss


# ---------------------------------------------------------------------------
# Evaluation phase
# ---------------------------------------------------------------------------

def phase_evaluate(base_model, domain, eval_data, B_q_np=None, B_v_np=None, A_q_slot=None, A_v_slot=None):
    """Evaluate accuracy on domain.

    If B_q_np/B_v_np given: evaluate WITH LoRA adapter (SFT quality).
    Otherwise: evaluate WITHOUT adapter (promoted base quality).
    """
    use_lora = (B_q_np is not None)
    A_q_all = [A_q_slot] * N_LAYERS if use_lora else None
    A_v_all = [A_v_slot] * N_LAYERS if use_lora else None
    B_q_mlx = [mx.array(b) for b in B_q_np] if use_lora else None
    B_v_mlx = [mx.array(b) for b in B_v_np] if use_lora else None

    correct = 0
    total = 0
    for i in range(min(eval_data.shape[0], N_EVAL)):
        x = eval_data[i:i+1]    # (1, 4)
        logits = base_model(
            x, lora_B_q=B_q_mlx, lora_A_q=A_q_all, lora_B_v=B_v_mlx, lora_A_v=A_v_all
        )
        mx.eval(logits)
        # Prediction at position 2 (predict token at pos 3 from context [0,1,2,3])
        pred_idx = int(mx.argmax(logits[0, 2, :]).item())  # logit for position 2 → token 3
        target = int(x[0, 3].item())
        if pred_idx == target:
            correct += 1
        total += 1
        del logits

    if B_q_mlx:
        cleanup(*B_q_mlx, *B_v_mlx)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Promotion phase: merge LoRA into base
# ---------------------------------------------------------------------------

def phase_promote(base_model, B_q_np, B_v_np, A_q_slot, A_v_slot):
    """Merge LoRA B matrices into base model weights in-place.

    Returns Frobenius norms of the added deltas per projection type.
    """
    delta_norms_q = []
    delta_norms_v = []

    for l_idx, block in enumerate(base_model.blocks):
        A_q = np.array(A_q_slot, dtype=np.float64)   # (D_MODEL, LORA_RANK)
        A_v = np.array(A_v_slot, dtype=np.float64)
        B_q = B_q_np[l_idx].astype(np.float64)       # (LORA_RANK, D_MODEL)
        B_v = B_v_np[l_idx].astype(np.float64)

        # LoRA forward: y += scale * (x @ A) @ B
        # Promotion: weight += scale * (A @ B).T = scale * B.T @ A.T
        # weight.shape = (D_out, D_in) in nn.Linear
        delta_q = (LORA_SCALE * B_q.T @ A_q.T).astype(np.float32)   # (D_MODEL, D_MODEL)
        delta_v = (LORA_SCALE * B_v.T @ A_v.T).astype(np.float32)

        # Update q_proj weight
        old_q_w = np.array(block.attn.q_proj.weight)
        new_q_w = mx.array((old_q_w + delta_q).astype(np.float32))
        block.attn.q_proj.weight = new_q_w

        # Update v_proj weight
        old_v_w = np.array(block.attn.v_proj.weight)
        new_v_w = mx.array((old_v_w + delta_v).astype(np.float32))
        block.attn.v_proj.weight = new_v_w

        delta_norms_q.append(float(np.linalg.norm(delta_q, "fro")))
        delta_norms_v.append(float(np.linalg.norm(delta_v, "fro")))

    mx.eval(base_model.parameters())
    log(f"  [promote] delta_norms q={[f'{n:.4f}' for n in delta_norms_q]} v={[f'{n:.4f}' for n in delta_norms_v]}")
    return delta_norms_q, delta_norms_v


# ---------------------------------------------------------------------------
# Pythagorean bound measurement
# ---------------------------------------------------------------------------

def measure_pythagorean_bound(initial_weights_np, base_model, all_delta_norms_q, all_delta_norms_v):
    """Verify Theorem 1: ||W_K - W_0||_F = sqrt(sum_k ||dW_k||_F^2)."""
    total_sq = sum(
        sum(n**2 for n in norms_q) + sum(n**2 for n in norms_v)
        for norms_q, norms_v in zip(all_delta_norms_q, all_delta_norms_v)
    )
    pythagorean_pred = math.sqrt(total_sq)

    # Compute actual cumulative change
    actual_sq = 0.0
    for l_idx, block in enumerate(base_model.blocks):
        q_init = initial_weights_np[f"block{l_idx}_q_w"]
        v_init = initial_weights_np[f"block{l_idx}_v_w"]
        q_curr = np.array(block.attn.q_proj.weight)
        v_curr = np.array(block.attn.v_proj.weight)
        actual_sq += float(np.linalg.norm(q_curr - q_init, "fro")**2)
        actual_sq += float(np.linalg.norm(v_curr - v_init, "fro")**2)
    actual = math.sqrt(actual_sq)

    rel_err = abs(actual - pythagorean_pred) / max(actual, 1e-10)
    log(f"  [pyth] predicted={pythagorean_pred:.6f} actual={actual:.6f} rel_err={rel_err:.2e}")
    return pythagorean_pred, actual, rel_err


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    log("=== Multi-Cycle Promotion Experiment ===")
    log_memory("start")

    # 1. Generate Grassmannian A-slots (one per domain, shared across layers)
    log("\n[Step 1] Generating Grassmannian A-slots...")
    # We need separate A_q and A_v slots for each domain
    # Total slots: n_domains = 3, rank = 4 → need 12-dim orthogonal basis per projection
    A_q_slots = make_grassmannian_slots(D_MODEL, LORA_RANK, N_DOMAINS, seed=SEED)
    A_v_slots = make_grassmannian_slots(D_MODEL, LORA_RANK, N_DOMAINS, seed=SEED + 1)

    # Verify orthogonality
    for i in range(N_DOMAINS):
        for j in range(i+1, N_DOMAINS):
            orth_q = float(mx.abs(A_q_slots[i].T @ A_q_slots[j]).max().item())
            orth_v = float(mx.abs(A_v_slots[i].T @ A_v_slots[j]).max().item())
            log(f"  A_q[{i}]^T A_q[{j}] max = {orth_q:.2e}  (should be ~0)")
            log(f"  A_v[{i}]^T A_v[{j}] max = {orth_v:.2e}  (should be ~0)")

    # 2. Initialize base model
    log("\n[Step 2] Initializing base model...")
    mx.random.seed(SEED)
    base_model = TinyLM(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS)
    mx.eval(base_model.parameters())

    # Save initial weights for Pythagorean bound measurement
    initial_weights_np = {}
    for l_idx, block in enumerate(base_model.blocks):
        initial_weights_np[f"block{l_idx}_q_w"] = np.array(block.attn.q_proj.weight)
        initial_weights_np[f"block{l_idx}_v_w"] = np.array(block.attn.v_proj.weight)

    # 3. Generate data for all domains
    log("\n[Step 3] Generating training/eval data...")
    train_data = {d: make_data(d, n=100, seed=SEED + i) for i, d in enumerate(DOMAINS)}
    eval_data  = {d: make_data(d, n=N_EVAL, seed=SEED + 100 + i) for i, d in enumerate(DOMAINS)}
    for d in DOMAINS:
        mx.eval(train_data[d], eval_data[d])

    # Results tracking
    sft_acc = {}          # accuracy WITH adapter (best case)
    promoted_acc = {}     # {cycle: {domain: accuracy}} — on base WITHOUT adapter
    all_delta_norms_q = []
    all_delta_norms_v = []

    # 3b. Pre-train base on all domains jointly
    log("\n[Step 3b] Pre-training base model on all domains...")
    phase_pretrain(base_model, train_data)
    log_memory("after-pretrain")

    # Re-save initial weights AFTER pre-training (this IS the base for adapter training)
    initial_weights_np = {}
    for l_idx, block in enumerate(base_model.blocks):
        initial_weights_np[f"block{l_idx}_q_w"] = np.array(block.attn.q_proj.weight)
        initial_weights_np[f"block{l_idx}_v_w"] = np.array(block.attn.v_proj.weight)

    # 4. Three-cycle promotion loop
    for cycle, domain in enumerate(DOMAINS):
        log(f"\n{'='*60}")
        log(f"[Cycle {cycle+1}] Domain: {domain}")
        log(f"{'='*60}")

        # Phase A: Train adapter for this domain on current base
        B_q_np, B_v_np, final_loss = phase_train_adapter(
            base_model, A_q_slots[cycle], A_v_slots[cycle], train_data[domain], domain
        )
        log_memory(f"after-train-{domain}")

        # Phase B: Evaluate WITH adapter (SFT quality)
        acc_sft = phase_evaluate(
            base_model, domain, eval_data[domain],
            B_q_np=B_q_np, B_v_np=B_v_np,
            A_q_slot=A_q_slots[cycle], A_v_slot=A_v_slots[cycle]
        )
        sft_acc[domain] = acc_sft
        log(f"  [eval] sft_acc({domain}) = {acc_sft:.3f}")

        # Phase C: Promote adapter into base
        dn_q, dn_v = phase_promote(base_model, B_q_np, B_v_np, A_q_slots[cycle], A_v_slots[cycle])
        all_delta_norms_q.append(dn_q)
        all_delta_norms_v.append(dn_v)
        del B_q_np, B_v_np

        # Phase D: Evaluate ALL previously-promoted domains on new base (no adapter)
        promoted_acc[cycle+1] = {}
        for d_eval in DOMAINS[:cycle+1]:
            acc = phase_evaluate(base_model, d_eval, eval_data[d_eval])
            promoted_acc[cycle+1][d_eval] = acc
            ratio = acc / max(sft_acc[d_eval], 1e-6)
            log(f"  [eval] promoted_acc({d_eval}) after cycle {cycle+1} = {acc:.3f} "
                f"(quality_ratio = {ratio:.3f})")

        log_memory(f"after-cycle-{domain}")

    # 5. Pythagorean bound verification
    log("\n[Step 5] Pythagorean bound verification (Theorem 1):")
    pyth_pred, pyth_actual, pyth_rel_err = measure_pythagorean_bound(
        initial_weights_np, base_model, all_delta_norms_q, all_delta_norms_v
    )

    # 6. Kill criteria evaluation
    log("\n[Step 6] Kill criteria evaluation:")
    final_cycle = len(DOMAINS)

    # K928: After 3 cycles, all domains >= 80% of SFT
    k928_results = {}
    k928_pass = True
    for d in DOMAINS:
        if d in promoted_acc.get(final_cycle, {}):
            prom_acc = promoted_acc[final_cycle][d]
            sft = sft_acc.get(d, 0.0)
            ratio = prom_acc / max(sft, 1e-6)
            passes = ratio >= 0.80
            k928_results[d] = {"promoted_acc": prom_acc, "sft_acc": sft, "ratio": ratio, "pass": passes}
            if not passes:
                k928_pass = False
            log(f"  K928 {d}: promoted={prom_acc:.3f} sft={sft:.3f} ratio={ratio:.3f} {'PASS' if passes else 'FAIL'}")

    # K929: No domain degrades >20% from its post-promotion quality
    k929_results = {}
    k929_pass = True
    for d_idx, d in enumerate(DOMAINS):
        promo_cycle = d_idx + 1
        initial_prom = promoted_acc.get(promo_cycle, {}).get(d)
        final_prom = promoted_acc.get(final_cycle, {}).get(d)
        if initial_prom is not None and final_prom is not None and initial_prom > 0:
            degradation = (initial_prom - final_prom) / initial_prom
            passes = degradation <= 0.20
            k929_results[d] = {
                "initial_promoted": initial_prom, "final_promoted": final_prom,
                "degradation": degradation, "pass": passes
            }
            if not passes:
                k929_pass = False
            log(f"  K929 {d}: initial={initial_prom:.3f} final={final_prom:.3f} "
                f"degradation={degradation:.1%} {'PASS' if passes else 'FAIL'}")

    # K930 (KILL): Any domain < 50%
    k930_kill = False
    for d in DOMAINS:
        if d in promoted_acc.get(final_cycle, {}):
            if promoted_acc[final_cycle][d] < 0.50:
                k930_kill = True
                log(f"  K930 KILL: {d} dropped to {promoted_acc[final_cycle][d]:.3f}")
    if not k930_kill:
        log("  K930: No domain below 50% — KILL not triggered")

    # Summary
    total_time = time.time() - t0
    log(f"\n[Summary] K928={'PASS' if k928_pass else 'FAIL'} "
        f"K929={'PASS' if k929_pass else 'FAIL'} "
        f"K930={'KILL' if k930_kill else 'NO-KILL'} "
        f"time={total_time:.1f}s")

    results = {
        "experiment": "exp_m2p_multi_cycle_promotion",
        "config": {
            "d_model": D_MODEL, "n_layers": N_LAYERS, "n_heads": N_HEADS,
            "lora_rank": LORA_RANK, "lora_scale": LORA_SCALE,
            "n_train_steps": N_TRAIN_STEPS, "n_eval": N_EVAL, "lr": LR,
            "domains": DOMAINS, "mod": MOD, "seed": SEED,
        },
        "sft_acc": sft_acc,
        "promoted_acc": {str(k): v for k, v in promoted_acc.items()},
        "pythagorean_bound": {
            "predicted": pyth_pred, "actual": pyth_actual, "rel_error": pyth_rel_err,
            "pass": pyth_rel_err < 1e-3,
        },
        "k928": {"pass": k928_pass, "results": k928_results},
        "k929": {"pass": k929_pass, "results": k929_results},
        "k930": {"kill": k930_kill},
        "all_criteria_pass": k928_pass and k929_pass and not k930_kill,
        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults written to {RESULTS_FILE}")
    log_memory("end")
    cleanup(base_model)
    return results


if __name__ == "__main__":
    main()
