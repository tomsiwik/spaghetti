"""
T0.3: p-RoPE Semantic Channels Are Position-Invariant

Kill criteria:
  K997: max ||NoPE(v, pos=100) - NoPE(v, pos=100000)||_inf == 0 (algebraic)
  K998: mean ||RoPE(v, pos=100) - RoPE(v, pos=100000)||_2 > 0 (control)
  K999: NoPE-only adapter quality >= 90% of full-dim adapter

Gemma 4 global attention: head_dim=512, partial_rotary_factor=0.25
RoPE dims: [0, 128), NoPE dims: [128, 512)
"""

import json
import math
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

IS_SMOKE = "--smoke" in sys.argv

# ── Gemma 4 global attention config ──────────────────────────────────────────
HEAD_DIM = 512
PARTIAL_ROTARY_FACTOR = 0.25
ROPE_DIM = int(PARTIAL_ROTARY_FACTOR * HEAD_DIM)   # 128
NOPE_DIM = HEAD_DIM - ROPE_DIM                     # 384
ROPE_BASE = 10000.0

# Adapter / training config
RANK = 4
D_OUT = 64       # output dim of Q_proj proxy
HIDDEN = HEAD_DIM
BATCH = 64
N_TRAIN = 500 if not IS_SMOKE else 20
N_STEPS = 300 if not IS_SMOKE else 10
N_TEST = 200 if not IS_SMOKE else 20
SEED = 42
LR = 1e-3


# ── Phase 1: p-RoPE algebraic verification ───────────────────────────────────

def compute_gemma4_inv_freq():
    """Reproduce Gemma 4's _compute_proportional_rope_parameters in numpy."""
    n_rope_pairs = ROPE_DIM // 2   # 64 pairs
    n_total_pairs = HEAD_DIM // 2  # 256 pairs

    inv_freq_rope = np.array([
        1.0 / (ROPE_BASE ** (2 * k / HEAD_DIM))
        for k in range(n_rope_pairs)
    ])  # shape (64,), all > 0

    # Zero-pad to total pairs (NoPE pairs get 0)
    inv_freq_full = np.zeros(n_total_pairs)
    inv_freq_full[:n_rope_pairs] = inv_freq_rope
    return inv_freq_full  # shape (256,)


def apply_rope_numpy(v, position, inv_freq):
    """
    Apply RoPE to vector v at scalar position.
    v: (head_dim,) float array
    inv_freq: (head_dim // 2,) float array
    Returns: (head_dim,) rotated vector
    """
    theta = position * inv_freq  # (256,)
    cos_theta = np.cos(theta)    # (256,)
    sin_theta = np.sin(theta)    # (256,)

    v_out = np.empty_like(v)
    for i in range(HEAD_DIM // 2):
        v0 = v[2 * i]
        v1 = v[2 * i + 1]
        c = cos_theta[i]
        s = sin_theta[i]
        v_out[2 * i]     = v0 * c - v1 * s
        v_out[2 * i + 1] = v0 * s + v1 * c
    return v_out


def phase1_algebraic_verification():
    """
    K997: NoPE dims (128-511) unchanged between pos 100 and pos 100000.
    K998: RoPE dims (0-127) DO change (control).
    """
    print("\n── Phase 1: Algebraic Verification ──────────────────────────────────")
    rng = np.random.default_rng(SEED)
    inv_freq = compute_gemma4_inv_freq()

    # Verify inv_freq structure
    n_nonzero = np.sum(inv_freq > 0)
    n_zero = np.sum(inv_freq == 0)
    print(f"inv_freq: {n_nonzero} non-zero (RoPE), {n_zero} zero (NoPE)")
    print(f"  RoPE pairs: freq[0]={inv_freq[0]:.6f} .. freq[63]={inv_freq[63]:.2e}")
    print(f"  NoPE pairs: freq[64]={inv_freq[64]:.6f} .. freq[255]={inv_freq[255]:.6f}")

    assert n_nonzero == ROPE_DIM // 2, f"Expected 64 non-zero, got {n_nonzero}"
    assert n_zero == NOPE_DIM // 2, f"Expected 192 zero, got {n_zero}"

    N_VEC = 100
    pos_a = 100.0
    pos_b = 100000.0

    nope_diffs = []
    rope_diffs = []

    for _ in range(N_VEC):
        v = rng.normal(size=(HEAD_DIM,)).astype(np.float64)
        va = apply_rope_numpy(v, pos_a, inv_freq)
        vb = apply_rope_numpy(v, pos_b, inv_freq)

        # NoPE slice: scalar indices [128, 512)
        nope_diff = np.max(np.abs(va[ROPE_DIM:] - vb[ROPE_DIM:]))
        rope_diff = np.linalg.norm(va[:ROPE_DIM] - vb[:ROPE_DIM])

        nope_diffs.append(nope_diff)
        rope_diffs.append(rope_diff)

    max_nope_diff = float(np.max(nope_diffs))
    mean_rope_diff = float(np.mean(rope_diffs))

    print(f"\nK997 — NoPE max |Δ| (pos {int(pos_a)} vs {int(pos_b)}): {max_nope_diff:.2e}")
    print(f"K998 — RoPE mean ||Δ||_2: {mean_rope_diff:.6f}")

    k997_pass = max_nope_diff == 0.0
    k998_pass = mean_rope_diff > 0.0

    print(f"  K997 PASS (==0.0): {k997_pass}")
    print(f"  K998 PASS (>0.0):  {k998_pass}")

    return {
        "max_nope_diff": max_nope_diff,
        "mean_rope_diff": mean_rope_diff,
        "k997_pass": k997_pass,
        "k998_pass": k998_pass,
    }


# ── Phase 2: Adapter Capacity Test ───────────────────────────────────────────

class FullDimAdapter(nn.Module):
    """Rank-r LoRA on all HEAD_DIM dims of Q_proj."""

    def __init__(self, r=RANK, scale=5.0):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, D_OUT, bias=False)
        self.lora_a = nn.Linear(HIDDEN, r, bias=False)
        self.lora_b = nn.Linear(r, D_OUT, bias=False)
        self.scale = scale
        # Init: lora_b zero so adapter starts as identity offset
        self.lora_b.weight = mx.zeros_like(self.lora_b.weight)

    def __call__(self, x):
        base = self.q_proj(x)
        delta = self.lora_b(self.lora_a(x))
        return base + self.scale * delta


class NoPEAdapter(nn.Module):
    """Rank-r LoRA restricted to NoPE dims [ROPE_DIM:HEAD_DIM] of Q_proj."""

    def __init__(self, r=RANK, scale=5.0):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, D_OUT, bias=False)
        self.lora_a = nn.Linear(NOPE_DIM, r, bias=False)
        self.lora_b = nn.Linear(r, D_OUT, bias=False)
        self.scale = scale
        self.lora_b.weight = mx.zeros_like(self.lora_b.weight)

    def __call__(self, x):
        base = self.q_proj(x)
        x_nope = x[..., ROPE_DIM:]   # slice NoPE dims
        delta = self.lora_b(self.lora_a(x_nope))
        return base + self.scale * delta


def make_dataset(rng_np, n, seed_offset=0):
    """
    Synthetic linear classification: y = sign(w^T x)
    w uniform over all HEAD_DIM dims.
    """
    rng2 = np.random.default_rng(SEED + seed_offset)
    w = rng2.normal(size=(HEAD_DIM,)).astype(np.float32)
    w /= np.linalg.norm(w)

    x = rng_np.normal(size=(n, HEAD_DIM)).astype(np.float32)
    y_score = x @ w
    y = (y_score > 0).astype(np.int32)
    return mx.array(x), mx.array(y), w


def train_adapter(model, x_train, y_train, n_steps, name):
    """Train binary classifier with Adam, return final train acc."""
    optimizer = optim.Adam(learning_rate=LR)

    def loss_fn(model, x, y):
        logits = model(x)          # (batch, D_OUT)
        # Binary classification: project to scalar via first dim
        score = logits[:, 0]
        loss = nn.losses.binary_cross_entropy(
            score, y.astype(mx.float32), with_logits=True
        )
        return loss

    grad_fn = nn.value_and_grad(model, loss_fn)

    losses = []
    for step in range(n_steps):
        # Mini-batch
        idx = np.random.randint(0, len(y_train), size=(BATCH,))
        x_b = x_train[mx.array(idx)]
        y_b = y_train[mx.array(idx)]

        loss, grads = grad_fn(model, x_b, y_b)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        losses.append(loss.item())

        if step % 100 == 0:
            print(f"  [{name}] step {step}/{n_steps} loss={loss.item():.4f}")

    return float(np.mean(losses[-10:]))


def evaluate_adapter(model, x_test, y_test):
    logits = model(x_test)
    score = logits[:, 0]
    preds = (score > 0).astype(mx.int32)
    mx.eval(preds)
    acc = float(mx.mean(preds == y_test).item())
    return acc


def phase2_adapter_capacity():
    """K999: NoPE-only adapter quality >= 90% of full-dim adapter."""
    print("\n── Phase 2: Adapter Capacity Test ───────────────────────────────────")
    rng = np.random.default_rng(SEED + 100)

    x_train, y_train, w = make_dataset(rng, N_TRAIN, seed_offset=0)
    x_test, y_test, _ = make_dataset(
        np.random.default_rng(SEED + 200), N_TEST, seed_offset=0
    )

    # Signal fraction in NoPE dims
    w_nope_norm_sq = float(np.sum(w[ROPE_DIM:] ** 2))
    print(f"Signal in NoPE dims: {w_nope_norm_sq:.3f} (expected {NOPE_DIM / HEAD_DIM:.3f})")
    print(f"Signal in RoPE dims: {1 - w_nope_norm_sq:.3f} (expected {ROPE_DIM / HEAD_DIM:.3f})")

    # Baseline (no adapter)
    w_mx = mx.array(w.reshape(HEAD_DIM, 1))
    baseline_score = (x_test @ w_mx).squeeze()
    mx.eval(baseline_score)
    baseline_acc = float(mx.mean(
        ((baseline_score > 0).astype(mx.int32) == y_test)
    ).item())
    print(f"Oracle accuracy (using true w): {baseline_acc:.4f}")

    # Full-dim adapter
    mx.random.seed(SEED)
    full_model = FullDimAdapter()
    mx.eval(full_model.parameters())
    print(f"\nTraining full-dim adapter ({N_STEPS} steps)...")
    train_adapter(full_model, x_train, y_train, N_STEPS, "full")
    full_acc = evaluate_adapter(full_model, x_test, y_test)

    # NoPE-only adapter (fresh random for fair comparison)
    mx.random.seed(SEED)
    nope_model = NoPEAdapter()
    mx.eval(nope_model.parameters())
    print(f"\nTraining NoPE-only adapter ({N_STEPS} steps)...")
    train_adapter(nope_model, x_train, y_train, N_STEPS, "nope")
    nope_acc = evaluate_adapter(nope_model, x_test, y_test)

    quality_ratio = nope_acc / full_acc if full_acc > 0 else 0.0
    k999_pass = quality_ratio >= 0.90

    print(f"\nK999 — quality_ratio = {quality_ratio:.4f} (NoPE={nope_acc:.4f} / full={full_acc:.4f})")
    print(f"  K999 PASS (>=0.90): {k999_pass}")

    return {
        "baseline_acc": baseline_acc,
        "full_acc": full_acc,
        "nope_acc": nope_acc,
        "quality_ratio": quality_ratio,
        "signal_in_nope": w_nope_norm_sq,
        "k999_pass": k999_pass,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("T0.3: p-RoPE Channel Isolation")
    print(f"  head_dim={HEAD_DIM}, rope_dim={ROPE_DIM}, nope_dim={NOPE_DIM}")
    print(f"  partial_rotary_factor={PARTIAL_ROTARY_FACTOR}")
    print(f"  smoke={IS_SMOKE}")
    print("=" * 60)

    r1 = phase1_algebraic_verification()
    r2 = phase2_adapter_capacity()

    elapsed = time.time() - t0
    print(f"\n── Summary ──────────────────────────────────────────────────────────")
    print(f"K997 PASS={r1['k997_pass']}  max_nope_diff={r1['max_nope_diff']:.2e}")
    print(f"K998 PASS={r1['k998_pass']}  mean_rope_diff={r1['mean_rope_diff']:.6f}")
    print(f"K999 PASS={r2['k999_pass']}  quality_ratio={r2['quality_ratio']:.4f}")
    print(f"Runtime: {elapsed:.1f}s")

    results = {
        "experiment": "exp_p1_t0_prope_channel_isolation",
        "is_smoke": IS_SMOKE,
        "config": {
            "head_dim": HEAD_DIM,
            "rope_dim": ROPE_DIM,
            "nope_dim": NOPE_DIM,
            "partial_rotary_factor": PARTIAL_ROTARY_FACTOR,
            "rank": RANK,
            "n_steps": N_STEPS,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
        },
        "phase1": r1,
        "phase2": r2,
        "kill_criteria": {
            "K997": {"pass": r1["k997_pass"], "value": r1["max_nope_diff"],
                     "threshold": "==0.0", "metric": "max_nope_diff_inf"},
            "K998": {"pass": r1["k998_pass"], "value": r1["mean_rope_diff"],
                     "threshold": ">0.0", "metric": "mean_rope_diff_l2"},
            "K999": {"pass": r2["k999_pass"], "value": r2["quality_ratio"],
                     "threshold": ">=0.90", "metric": "nope_acc/full_acc"},
        },
        "runtime_seconds": elapsed,
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults written to results.json")

    all_pass = r1["k997_pass"] and r1["k998_pass"] and r2["k999_pass"]
    print(f"\nOverall: {'ALL K PASS' if all_pass else 'SOME K FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
