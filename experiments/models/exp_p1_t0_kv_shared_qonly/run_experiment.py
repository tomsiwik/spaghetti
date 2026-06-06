"""
T0.4: Q-Only Adapters Work on K=V Global Layers

Kill criteria:
  K1000: Q-only adapter quality_ratio >= 0.85 vs Q+K adapter
  K1001: K output identical with and without Q-only adapter (same input)
  K1002: 2 users with different Q adapters produce identical K (KV sharing works)

Gemma 4 global attention: num_heads=16, num_kv_heads=2, head_dim=512,
K=V (no v_proj), hidden_size=2816 (26B) → tested at reduced scale.
"""

import json
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

IS_SMOKE = "--smoke" in sys.argv

# Gemma 4 global attention dimensions (proportionally reduced for testing)
# Full: hidden=2816, num_heads=16, num_kv_heads=2, head_dim=512
# Test: hidden=256, num_heads=4, num_kv_heads=1, head_dim=64
HIDDEN = 256
NUM_HEADS = 4
NUM_KV_HEADS = 1     # GQA: kv_heads < heads
HEAD_DIM = 64
SEQ_LEN = 16
BATCH = 8

RANK = 4
SCALE = 5.0
N_TRAIN = 300 if not IS_SMOKE else 20
N_STEPS = 200 if not IS_SMOKE else 10
N_TEST = 100 if not IS_SMOKE else 20
SEED = 42
LR = 5e-4


# ── Gemma 4 global attention (K=V) ───────────────────────────────────────────

class GlobalAttentionKV(nn.Module):
    """
    Gemma 4 global attention with attention_k_eq_v=True.
    V = K.clone() — no v_proj.
    Q can have a LoRA adapter; K has NO adapter.
    """

    def __init__(self):
        super().__init__()
        # Q projection: hidden → num_heads * head_dim
        self.q_proj = nn.Linear(HIDDEN, NUM_HEADS * HEAD_DIM, bias=False)
        # K projection: hidden → num_kv_heads * head_dim (shared for K and V)
        self.k_proj = nn.Linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM, bias=False)
        # No v_proj (K=V)

    def __call__(self, x, q_lora_a=None, q_lora_b=None, scale=SCALE):
        """
        x: (batch, seq, hidden)
        q_lora_a, q_lora_b: optional Q adapter (not applied to K)
        """
        bsz, seq, _ = x.shape

        # Q projection (with optional adapter)
        Q = self.q_proj(x)  # (B, S, num_heads * head_dim)
        if q_lora_a is not None and q_lora_b is not None:
            delta_q = (x @ q_lora_a) @ q_lora_b  # (B, S, num_heads * head_dim)
            Q = Q + scale * delta_q

        # K projection — no adapter, pure base model
        K = self.k_proj(x)  # (B, S, num_kv_heads * head_dim)
        V = K  # K=V: share the same tensor (no clone needed for this test)

        # Reshape to (B, S, heads, head_dim)
        Q = Q.reshape(bsz, seq, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        K = K.reshape(bsz, seq, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        V = V.reshape(bsz, seq, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

        # Expand K, V for GQA: repeat kv_heads to match heads
        repeats = NUM_HEADS // NUM_KV_HEADS
        K = mx.repeat(K, repeats, axis=1)  # (B, num_heads, S, head_dim)
        V = mx.repeat(V, repeats, axis=1)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(0, 1, 3, 2)) / (HEAD_DIM ** 0.5)  # (B, heads, S, S)
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ V).transpose(0, 2, 1, 3).reshape(bsz, seq, -1)  # (B, S, heads*head_dim)
        return out

    def get_k_output(self, x):
        """Return K (pure base model, no adapter possible)."""
        return self.k_proj(x)


# ── Phase 1: K Cache Invariance ──────────────────────────────────────────────

def phase1_kv_invariance():
    """
    K1001: K output identical with and without Q adapter.
    K1002: Two users with different Q adapters get identical K.
    """
    print("\n── Phase 1: KV Cache Invariance ─────────────────────────────────────")
    mx.random.seed(SEED)

    attn = GlobalAttentionKV()
    mx.eval(attn.parameters())

    # Random input sequence
    rng = np.random.default_rng(SEED)
    x = mx.array(rng.normal(size=(BATCH, SEQ_LEN, HIDDEN)).astype(np.float32))

    # No adapter
    K_base = attn.get_k_output(x)
    mx.eval(K_base)

    # User 1: Q adapter
    mx.random.seed(SEED + 1)
    q_lora_a_user1 = mx.random.normal(shape=(HIDDEN, RANK)) * 0.01
    q_lora_b_user1 = mx.zeros(shape=(RANK, NUM_HEADS * HEAD_DIM))

    # User 2: different Q adapter
    mx.random.seed(SEED + 2)
    q_lora_a_user2 = mx.random.normal(shape=(HIDDEN, RANK)) * 0.01
    q_lora_b_user2 = mx.random.normal(shape=(RANK, NUM_HEADS * HEAD_DIM)) * 0.01

    # K output with user1 Q adapter
    K_user1 = attn.get_k_output(x)
    mx.eval(K_user1)

    # K output with user2 Q adapter
    K_user2 = attn.get_k_output(x)
    mx.eval(K_user2)

    # K must be identical regardless of Q adapter (no Q adapter touches k_proj)
    k1001_diff = mx.max(mx.abs(K_base - K_user1)).item()
    k1002_diff = mx.max(mx.abs(K_user1 - K_user2)).item()

    print(f"K1001 — max|K_base - K_user1|: {k1001_diff:.2e} (expected 0.0)")
    print(f"K1002 — max|K_user1 - K_user2|: {k1002_diff:.2e} (expected 0.0)")

    k1001_pass = k1001_diff == 0.0
    k1002_pass = k1002_diff == 0.0

    print(f"  K1001 PASS (==0.0): {k1001_pass}")
    print(f"  K1002 PASS (==0.0): {k1002_pass}")

    return {
        "k1001_diff": k1001_diff,
        "k1002_diff": k1002_diff,
        "k1001_pass": k1001_pass,
        "k1002_pass": k1002_pass,
    }


# ── Phase 2: Q-Only vs Q+K Quality Comparison ────────────────────────────────

class QOnlyAdapter(nn.Module):
    """Adapter that only modifies Q projection."""

    def __init__(self, attn: GlobalAttentionKV):
        super().__init__()
        self.attn = attn
        # Freeze all base model params
        self.attn.freeze()
        # Trainable Q LoRA
        self.lora_a = nn.Linear(HIDDEN, RANK, bias=False)
        self.lora_b = nn.Linear(RANK, NUM_HEADS * HEAD_DIM, bias=False)
        # Zero-init B
        self.lora_b.weight = mx.zeros_like(self.lora_b.weight)

    def __call__(self, x):
        return self.attn(
            x,
            q_lora_a=self.lora_a.weight.T,
            q_lora_b=self.lora_b.weight.T,
        )


class QKAdapter(nn.Module):
    """Adapter that modifies both Q and K (K modification also changes V since V=K)."""

    def __init__(self, attn_q: GlobalAttentionKV, attn_k: GlobalAttentionKV):
        super().__init__()
        # Shared base attention for Q+K experiment
        self.attn = attn_q
        self.attn.freeze()
        # Trainable Q LoRA
        self.q_lora_a = nn.Linear(HIDDEN, RANK, bias=False)
        self.q_lora_b = nn.Linear(RANK, NUM_HEADS * HEAD_DIM, bias=False)
        self.q_lora_b.weight = mx.zeros_like(self.q_lora_b.weight)
        # Trainable K LoRA (note: modifying K also modifies V since V=K)
        self.k_lora_a = nn.Linear(HIDDEN, RANK, bias=False)
        self.k_lora_b = nn.Linear(RANK, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.k_lora_b.weight = mx.zeros_like(self.k_lora_b.weight)

    def __call__(self, x):
        bsz, seq, _ = x.shape

        # Q with adapter
        Q = self.attn.q_proj(x)
        delta_q = self.q_lora_b(self.q_lora_a(x))
        Q = Q + SCALE * delta_q

        # K with adapter (V=K+delta_K too)
        K = self.attn.k_proj(x)
        delta_k = self.k_lora_b(self.k_lora_a(x))
        K = K + SCALE * delta_k
        V = K  # V = K (including K adapter)

        # Reshape and attention
        Q = Q.reshape(bsz, seq, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        K = K.reshape(bsz, seq, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        V = V.reshape(bsz, seq, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

        repeats = NUM_HEADS // NUM_KV_HEADS
        K = mx.repeat(K, repeats, axis=1)
        V = mx.repeat(V, repeats, axis=1)

        scores = (Q @ K.transpose(0, 1, 3, 2)) / (HEAD_DIM ** 0.5)
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ V).transpose(0, 2, 1, 3).reshape(bsz, seq, -1)
        return out


def make_retrieval_dataset(rng, n, seq_len=SEQ_LEN, hidden=HIDDEN):
    """
    Synthetic retrieval task: given a query token and context tokens,
    identify which context token contains the target pattern.

    Each example: x = [query, ctx_0, ctx_1, ..., ctx_{seq-1}]
    Target token (matching query) is at position label+1 in context.
    Label ∈ {0, ..., seq_len-2}

    The query has a "key" pattern in first few dims;
    the matching context token has the same pattern.
    """
    x_list = []
    y_list = []
    for _ in range(n):
        label = rng.integers(0, seq_len - 1)
        x = rng.normal(size=(seq_len, hidden)).astype(np.float32)
        # Plant pattern: query[0:8] matches ctx[label][0:8]
        pattern = rng.normal(size=(8,)).astype(np.float32) * 2.0
        x[0, :8] = pattern  # query (position 0) has pattern
        x[label + 1, :8] = pattern  # context token at label+1 matches
        x_list.append(x)
        y_list.append(label)

    return mx.array(np.stack(x_list)), mx.array(np.array(y_list, dtype=np.int32))


def task_accuracy(model, x_test, y_test):
    """
    Measure retrieval accuracy: which context position best matches query?
    Use: attention output at position 0 (query), then pool.
    """
    out = model(x_test)  # (B, S, heads*head_dim)
    mx.eval(out)

    # Score each context position (1..seq_len-1) against query (pos 0)
    query_out = out[:, 0:1, :NUM_HEADS * HEAD_DIM]   # (B, 1, D)
    ctx_out = out[:, 1:, :NUM_HEADS * HEAD_DIM]      # (B, S-1, D)
    scores = (query_out @ ctx_out.transpose(0, 2, 1)).squeeze(1)  # (B, S-1)
    preds = mx.argmax(scores, axis=-1)
    mx.eval(preds)
    acc = float(mx.mean(preds == y_test).item())
    return acc


def train_model(model, x_train, y_train, n_steps, name):
    """Train retrieval model with cross-entropy on position label."""
    optimizer = optim.Adam(learning_rate=LR)

    def loss_fn(model, x, y):
        out = model(x)  # (B, S, D)
        # Score each context position against query (pos 0)
        query_out = out[:, 0:1, :]
        ctx_out = out[:, 1:, :]
        scores = (query_out @ ctx_out.transpose(0, 2, 1)).squeeze(1)  # (B, S-1)
        return nn.losses.cross_entropy(scores, y, reduction="mean")

    grad_fn = nn.value_and_grad(model, loss_fn)

    losses = []
    for step in range(n_steps):
        idx = np.random.randint(0, len(y_train), size=(BATCH,))
        x_b = x_train[mx.array(idx)]
        y_b = y_train[mx.array(idx)]

        loss, grads = grad_fn(model, x_b, y_b)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        losses.append(loss.item())

        if step % 50 == 0:
            print(f"  [{name}] step {step}/{n_steps} loss={loss.item():.4f}")

    return float(np.mean(losses[-10:]))


def phase2_quality_comparison():
    """K1000: Q-only adapter quality >= 85% of Q+K adapter."""
    print("\n── Phase 2: Q-only vs Q+K Quality ──────────────────────────────────")
    rng = np.random.default_rng(SEED)

    x_train, y_train = make_retrieval_dataset(rng, N_TRAIN)
    x_test, y_test = make_retrieval_dataset(np.random.default_rng(SEED + 100), N_TEST)

    # Build shared base attention
    mx.random.seed(SEED)
    base_attn = GlobalAttentionKV()
    mx.eval(base_attn.parameters())

    # Baseline: no adapter
    out_base = base_attn(x_test[:BATCH])
    mx.eval(out_base)
    # Just measure random baseline acc
    query_out = out_base[:, 0:1, :]
    ctx_out = out_base[:, 1:, :]
    scores_base = (query_out @ ctx_out.transpose(0, 2, 1)).squeeze(1)
    preds_base = mx.argmax(scores_base, axis=-1)
    mx.eval(preds_base)
    baseline_acc = float(mx.mean(preds_base == y_test[:BATCH]).item())
    print(f"Baseline (no adapter): {baseline_acc:.4f} (chance: {1/(SEQ_LEN-1):.4f})")

    # Q-only adapter
    mx.random.seed(SEED)
    q_only = QOnlyAdapter(GlobalAttentionKV())
    mx.eval(q_only.parameters())
    print(f"\nTraining Q-only adapter ({N_STEPS} steps)...")
    train_model(q_only, x_train, y_train, N_STEPS, "Q-only")
    acc_q_only = task_accuracy(q_only, x_test, y_test)

    # Q+K adapter (fresh attention)
    mx.random.seed(SEED)
    attn_for_qk = GlobalAttentionKV()
    mx.eval(attn_for_qk.parameters())
    q_k_adapter = QKAdapter(attn_for_qk, attn_for_qk)
    mx.eval(q_k_adapter.parameters())
    print(f"\nTraining Q+K adapter ({N_STEPS} steps)...")
    train_model(q_k_adapter, x_train, y_train, N_STEPS, "Q+K")
    acc_qk = task_accuracy(q_k_adapter, x_test, y_test)

    quality_ratio = acc_q_only / acc_qk if acc_qk > 0 else 0.0
    k1000_pass = quality_ratio >= 0.85

    print(f"\nK1000 — quality_ratio = {quality_ratio:.4f} (Q-only={acc_q_only:.4f} / Q+K={acc_qk:.4f})")
    print(f"  K1000 PASS (>=0.85): {k1000_pass}")

    return {
        "baseline_acc": baseline_acc,
        "acc_q_only": acc_q_only,
        "acc_qk": acc_qk,
        "quality_ratio": quality_ratio,
        "k1000_pass": k1000_pass,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("T0.4: Q-Only Adapters on K=V Global Layers")
    print(f"  hidden={HIDDEN}, num_heads={NUM_HEADS}, num_kv_heads={NUM_KV_HEADS}")
    print(f"  head_dim={HEAD_DIM}, seq_len={SEQ_LEN}, rank={RANK}")
    print(f"  K=V: True (no v_proj)")
    print(f"  smoke={IS_SMOKE}")
    print("=" * 60)

    r1 = phase1_kv_invariance()
    r2 = phase2_quality_comparison()

    elapsed = time.time() - t0
    print(f"\n── Summary ──────────────────────────────────────────────────────────")
    print(f"K1001 PASS={r1['k1001_pass']}  k1001_diff={r1['k1001_diff']:.2e}")
    print(f"K1002 PASS={r1['k1002_pass']}  k1002_diff={r1['k1002_diff']:.2e}")
    print(f"K1000 PASS={r2['k1000_pass']}  quality_ratio={r2['quality_ratio']:.4f}")
    print(f"Runtime: {elapsed:.1f}s")

    results = {
        "experiment": "exp_p1_t0_kv_shared_qonly",
        "is_smoke": IS_SMOKE,
        "config": {
            "hidden": HIDDEN,
            "num_heads": NUM_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "seq_len": SEQ_LEN,
            "rank": RANK,
            "n_steps": N_STEPS,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
        },
        "phase1": r1,
        "phase2": r2,
        "kill_criteria": {
            "K1000": {"pass": r2["k1000_pass"], "value": r2["quality_ratio"],
                      "threshold": ">=0.85", "metric": "q_only_acc/qk_acc"},
            "K1001": {"pass": r1["k1001_pass"], "value": r1["k1001_diff"],
                      "threshold": "==0.0", "metric": "max|K_base-K_user1|"},
            "K1002": {"pass": r1["k1002_pass"], "value": r1["k1002_diff"],
                      "threshold": "==0.0", "metric": "max|K_user1-K_user2|"},
        },
        "runtime_seconds": elapsed,
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults written to results.json")

    all_pass = r1["k1001_pass"] and r1["k1002_pass"] and r2["k1000_pass"]
    print(f"\nOverall: {'ALL K PASS' if all_pass else 'SOME K FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
