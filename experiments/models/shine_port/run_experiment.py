#!/usr/bin/env python3
"""SHINE: port M2P Transformer to MLX, test on toy model.

Extracts the core M2P (Memory-to-Parameter) Transformer from SHINE
(arXiv:2602.06358) and implements it in MLX. Tests on a toy transformer
to verify the architecture generates meaningful adapter weights from
hidden state memory.

Type: Guided Exploration (Type 2).
Proven framework: SHINE M2P architecture (arXiv:2602.06358).
Unknown being probed: does the MLX port produce outputs statistically
  distinguishable from random projections?

Kill criteria:
  K826: Core architecture not portable to MLX
  K827 (PRIMARY): M2P output cosine similarity distribution must be
        statistically distinguishable from random baseline at p < 0.05
        (two-sample t-test, n=30 pairs each) AND absolute difference
        in means > 0.05 (effect size threshold).
        If NOT distinguishable: architecture provides no structure beyond
        random projection — port is mechanically functional but scientifically
        undemonstrated as a structured adapter generator.
"""

import gc, json, math, os, time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import scipy.stats
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

# Memory safety: leave 8 GB for system, cap cache at 2 GB
_dev = mx.device_info()
mx.set_memory_limit(_dev["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
SEED = 42


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

def log(m): print(m, flush=True)
def cleanup(*o):
    for x in o: del x
    gc.collect(); mx.clear_cache(); mx.reset_peak_memory()


# ── M2P Transformer (ported from SHINE §3.4) ────────────────────────────

class M2PAttention(nn.Module):
    """Standard bidirectional self-attention for M2P Transformer."""
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        scale = self.head_dim ** -0.5
        attn = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * scale, axis=-1)
        return self.wo((attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C))


class M2PBlock(nn.Module):
    """One M2P Transformer block with alternating row/column attention.

    From SHINE §3.4: odd layers use column attention (across layers),
    even layers use row attention (across memory tokens).
    """
    def __init__(self, dim, n_heads=4, is_column=True):
        super().__init__()
        self.attn = M2PAttention(dim, n_heads)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * dim, dim, bias=False),
        )
        self.is_column = is_column

    def __call__(self, x):
        """x: (L, M, H) where L=layers, M=memory tokens, H=hidden dim."""
        L, M, H = x.shape

        if self.is_column:
            # Column attention: attend across LAYERS for each memory token
            # Reshape to (M, L, H) so each "batch" is one memory token attending across layers
            x_t = x.transpose(1, 0, 2)  # (M, L, H)
            x_t = x_t + self.attn(self.norm1(x_t))
            x_t = x_t + self.mlp(self.norm2(x_t))
            return x_t.transpose(1, 0, 2)  # back to (L, M, H)
        else:
            # Row attention: attend across MEMORY TOKENS for each layer
            # x is already (L, M, H) — each "batch" is one layer attending across memory tokens
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x


class M2PTransformer(nn.Module):
    """Memory-to-Parameter Transformer (SHINE §3.4).

    Takes memory states from all layers of a base LLM and generates
    adapter weights (LoRA A and B matrices).

    Input: (L, M, H) — L layers, M memory tokens, H hidden dim
    Output: (L, M, H) — contextualized memory grid

    Positional embeddings per SHINE §3.4 Eq. 5:
      P_layer: (L, 1, H) — learned layer-position embedding, broadcast across M
      P_token: (1, M, H) — learned token-position embedding, broadcast across L
      Applied: memory_states = memory_states + P_layer + P_token
    Without these, attention is permutation-equivariant in the layer dimension
    and cannot distinguish layer i from layer j (Vaswani et al. 2017).
    """
    def __init__(self, n_lm_layers, n_mem_tokens, dim, n_layers_m2p=4, n_heads=4):
        super().__init__()
        # Positional embeddings per SHINE §3.4 Eq. 5.
        # Xavier normal initialization (Fix 4): provides non-trivial positional
        # signal from the start, not just after training.
        # scale = sqrt(2 / (fan_in + fan_out)); for positional embedding,
        # fan_in=1 (scalar position index), fan_out=dim.
        scale = math.sqrt(2.0 / (1 + dim))
        self.p_layer = mx.random.normal((n_lm_layers, 1, dim)) * scale   # (L, 1, H)
        self.p_token = mx.random.normal((1, n_mem_tokens, dim)) * scale  # (1, M, H)

        self.blocks = []
        for i in range(n_layers_m2p):
            is_column = (i % 2 == 0)  # alternate column/row attention
            self.blocks.append(M2PBlock(dim, n_heads, is_column))
        self.final_norm = nn.RMSNorm(dim)

    def __call__(self, memory_states):
        """memory_states: (L, M, H)"""
        # Apply positional embeddings per SHINE §3.4 Eq. 5
        x = memory_states + self.p_layer + self.p_token
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


def cosine_sim_flat(a, b):
    """Cosine similarity between two arbitrary-shape arrays, treated as flat vectors."""
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_flat * b_flat)
    norm_a = mx.linalg.norm(a_flat)
    norm_b = mx.linalg.norm(b_flat)
    result = dot / (norm_a * norm_b + 1e-8)
    mx.eval(result)
    return result.item()


# ── Phase functions (each scoped to avoid memory accumulation) ───────────

def phase_instantiate(L, M, H, N_M2P_LAYERS):
    """Phase 1: instantiate the M2P transformer and report size."""
    log("\n=== Phase 1: Instantiate M2P Transformer ===")
    m2p = M2PTransformer(L, M, H, n_layers_m2p=N_M2P_LAYERS, n_heads=4)
    mx.eval(m2p.parameters())

    n_params = sum(p.size for _, p in nn.utils.tree_flatten(m2p.parameters()))
    log(f"  M2P parameters: {n_params:,}")
    log(f"  M2P parameter size: {n_params * 4 / 1e6:.2f} MB (float32)")
    log(f"  P_layer shape: {m2p.p_layer.shape}  (SHINE Eq. 5, Xavier init)")
    log(f"  P_token shape: {m2p.p_token.shape}  (SHINE Eq. 5, Xavier init)")

    # Verify Xavier init is non-zero
    p_layer_norm = mx.linalg.norm(m2p.p_layer.reshape(-1)).item()
    p_token_norm = mx.linalg.norm(m2p.p_token.reshape(-1)).item()
    log(f"  P_layer norm: {p_layer_norm:.4f} (non-zero = correct)")
    log(f"  P_token norm: {p_token_norm:.4f} (non-zero = correct)")
    return m2p, n_params


def phase_forward(m2p, L, M, H):
    """Phase 2: verify forward pass shape and timing."""
    log("\n=== Phase 2: Forward pass ===")
    memory_states = mx.random.normal((L, M, H))
    t1 = time.time()
    output = m2p(memory_states)
    mx.eval(output)
    fwd_time = time.time() - t1
    log(f"  Input: {memory_states.shape}")
    log(f"  Output: {output.shape}")
    log(f"  Forward time: {fwd_time*1000:.1f}ms")
    return fwd_time, memory_states, output


def phase_parameter_generation(output, L, M, H, LORA_RANK):
    """Phase 3: verify M2P output has enough values to fill LoRA matrices."""
    log("\n=== Phase 3: Parameter generation ===")
    results = []
    for li in range(L):
        layer_output = output[li]   # (M, H)
        flat = layer_output.reshape(-1)
        needed = H * LORA_RANK + LORA_RANK * H   # 2 * H * rank
        available = flat.shape[0]

        if available >= needed:
            A = flat[:H * LORA_RANK].reshape(H, LORA_RANK)
            B = flat[H * LORA_RANK:H * LORA_RANK + LORA_RANK * H].reshape(LORA_RANK, H)
            mx.eval(A, B)
            delta = A @ B
            mx.eval(delta)
            delta_norm = mx.linalg.norm(delta.reshape(-1)).item()
            log(f"  Layer {li}: A={A.shape} B={B.shape} delta_norm={delta_norm:.4f}")
            results.append({"layer": li, "delta_norm": delta_norm})
        else:
            log(f"  Layer {li}: INSUFFICIENT (need M={M} >= 2*rank={2*LORA_RANK})")
            results.append({"layer": li, "delta_norm": None})
    return results


def phase_gradient_flow(m2p, memory_states):
    """Phase 4: verify gradients reach all layers."""
    log("\n=== Phase 4: Gradient flow ===")

    def dummy_loss(m2p, memory):
        out = m2p(memory)
        return mx.mean(out ** 2)

    loss_and_grad = nn.value_and_grad(m2p, dummy_loss)
    loss, grads = loss_and_grad(m2p, memory_states)
    mx.eval(loss)
    grad_norms = [mx.linalg.norm(g.reshape(-1)).item()
                  for _, g in nn.utils.tree_flatten(grads) if g.size > 0]
    log(f"  Loss: {loss.item():.6f}")
    log(f"  Grad norms: min={min(grad_norms):.6f} max={max(grad_norms):.6f}")
    all_nonzero = all(g > 1e-10 for g in grad_norms)
    log(f"  All grads non-zero: {all_nonzero}")
    return all_nonzero, grad_norms


def phase_input_sensitivity(m2p, L, M, H, n_pairs=30, label=""):
    """Phase 6a/6b: Input-sensitivity test.

    Two distinct random memory states should produce M2P outputs with
    cosine similarity near 0 (random matrix theory prediction for n=2048).

    Random matrix theory predicts: E[cos] = 0, std ≈ 1/sqrt(L*M*H) ≈ 0.022.

    Run this BEFORE and AFTER Phase 5 training to measure whether training
    changes input sensitivity.
    """
    phase_label = f"Phase 6{label}: Input Sensitivity Test"
    log(f"\n=== {phase_label} ===")
    n = L * M * H
    rmt_std = 1.0 / math.sqrt(n)
    log(f"  Random matrix theory prediction: E[cos] = 0, std ≈ {rmt_std:.4f} (n={n})")
    log(f"  n_pairs = {n_pairs}")

    cos_sims = []
    for i in range(n_pairs):
        m1 = mx.random.normal((L, M, H))
        m2 = mx.random.normal((L, M, H))
        out1 = m2p(m1)
        out2 = m2p(m2)
        mx.eval(out1, out2)
        cos = cosine_sim_flat(out1, out2)
        cos_sims.append(cos)

    mean_cos = float(np.mean(cos_sims))
    std_cos = float(np.std(cos_sims))
    log(f"  Mean cosine similarity: {mean_cos:.4f}  (RMT prediction: ≈ 0)")
    log(f"  Std cosine similarity:  {std_cos:.4f}  (RMT prediction: ≈ {rmt_std:.4f})")
    log(f"  Mean in [-0.1, 0.1]: {abs(mean_cos) <= 0.1} (P1 sanity check)")

    return mean_cos, std_cos, cos_sims


def phase_convergence(m2p, memory_states, L, M, H):
    """Phase 5: quick convergence test on synthetic target."""
    log("\n=== Phase 5: Convergence test ===")
    target = mx.random.normal((L, M, H)) * 0.1
    optimizer = opt.Adam(learning_rate=1e-3)

    def target_loss(m2p, memory, target):
        return mx.mean((m2p(memory) - target) ** 2)

    losses = []
    gc.disable()
    for step in range(100):
        loss, grads = nn.value_and_grad(m2p, target_loss)(m2p, memory_states, target)
        optimizer.update(m2p, grads)
        mx.eval(m2p.parameters(), optimizer.state, loss)
        losses.append(loss.item())
        if (step + 1) % 25 == 0:
            log(f"  Step {step+1}: loss={loss.item():.6f}")
    gc.enable()

    converged = losses[-1] < losses[0] * 0.5
    log(f"  Converged: {converged} (final/initial = {losses[-1]/losses[0]:.3f})")
    return converged, losses


def phase_random_baseline(m2p, L, M, H, n_pairs=30):
    """Phase 7 — PRIMARY KILL CRITERION (K827).

    Compares M2P output cosine distribution vs random baseline cosine
    distribution using a two-sample t-test.

    Random matrix theory null: for two independent standard normal vectors
    in R^n, E[cos] = 0 with std ≈ 1/sqrt(n).

    K827 PASS: p_value < 0.05 AND |diff_in_means| > 0.05
      => M2P outputs are statistically distinguishable from random noise
      => The architecture imposes detectable structure beyond random projection

    K827 FAIL: p_value >= 0.05 OR |diff_in_means| <= 0.05
      => M2P outputs are NOT statistically distinguishable from random noise
      => The port is mechanically functional but scientifically undemonstrated
         as a structured adapter generator
    """
    log(f"\n=== Phase 7: Random-Baseline Comparison (PRIMARY KILL CRITERION K827) ===")
    log(f"  n_pairs = {n_pairs} (each group) — two-sample t-test, alpha=0.05")
    log(f"  K827 PASS requires: p < 0.05 AND |diff in means| > 0.05")

    m2p_cos_list = []
    rand_cos_list = []

    for i in range(n_pairs):
        # M2P cosine: two distinct inputs through the same trained M2P
        m1 = mx.random.normal((L, M, H))
        m2 = mx.random.normal((L, M, H))
        out1 = m2p(m1)
        out2 = m2p(m2)
        mx.eval(out1, out2)
        m2p_cos = cosine_sim_flat(out1, out2)
        m2p_cos_list.append(m2p_cos)

        # Random baseline: two independent random vectors of same shape
        r1 = mx.random.normal((L, M, H))
        r2 = mx.random.normal((L, M, H))
        mx.eval(r1, r2)
        rand_cos = cosine_sim_flat(r1, r2)
        rand_cos_list.append(rand_cos)

    mean_m2p = float(np.mean(m2p_cos_list))
    std_m2p = float(np.std(m2p_cos_list))
    mean_rand = float(np.mean(rand_cos_list))
    std_rand = float(np.std(rand_cos_list))
    diff_means = abs(mean_m2p - mean_rand)

    # Two-sample t-test (Welch's, unequal variances)
    t_stat, p_value = scipy.stats.ttest_ind(m2p_cos_list, rand_cos_list, equal_var=False)
    p_value = float(p_value)
    t_stat = float(t_stat)

    log(f"  M2P output cosine:   mean={mean_m2p:.4f}  std={std_m2p:.4f}  n={n_pairs}")
    log(f"  Random baseline cos: mean={mean_rand:.4f}  std={std_rand:.4f}  n={n_pairs}")
    log(f"  Difference in means: {diff_means:.4f}")
    log(f"  Welch t-test: t={t_stat:.3f}, p={p_value:.4f}")

    k827_p_pass = p_value < 0.05
    k827_effect_pass = diff_means > 0.05
    k827_pass = k827_p_pass and k827_effect_pass

    log(f"  p < 0.05:         {k827_p_pass}  (p={p_value:.4f})")
    log(f"  |diff| > 0.05:    {k827_effect_pass}  (|diff|={diff_means:.4f})")
    log(f"  K827 (PRIMARY):   {'PASS' if k827_pass else 'FAIL'}")
    if not k827_pass:
        log("  *** PHASE 7 FAIL: M2P outputs are NOT statistically distinguishable")
        log("      from random noise. The port is mechanically functional but")
        log("      does not demonstrate structured adapter generation on toy data. ***")

    return (k827_pass, k827_p_pass, k827_effect_pass,
            mean_m2p, std_m2p, mean_rand, std_rand,
            diff_means, t_stat, p_value,
            m2p_cos_list, rand_cos_list)


def main():
    t0 = time.time()
    log("SHINE M2P Transformer Port to MLX (Revision 3)")
    log("=" * 60)
    log("Experiment type: Guided Exploration (Type 2)")
    log("Unknown: Are M2P outputs statistically distinguishable from random noise?")
    log("PRIMARY KILL CRITERION: K827 — two-sample t-test, p < 0.05 AND |diff| > 0.05")
    mx.random.seed(SEED)

    # Architecture test parameters
    L = 4       # LLM layers
    M = 8       # memory tokens
    H = 64      # hidden dim
    LORA_RANK = 4
    N_M2P_LAYERS = 4
    N_PAIRS = 30   # Fix 3: increased from 10 to 30 for statistical power

    log(f"M2P config: L={L}, M={M}, H={H}, rank={LORA_RANK}, m2p_layers={N_M2P_LAYERS}")
    n_flat = L * M * H
    rmt_std = 1.0 / math.sqrt(n_flat)
    log(f"Random matrix theory prediction: E[cos]=0, std≈{rmt_std:.4f} (n={n_flat})")

    # --- Phase 1: Instantiation ---
    m2p, n_params = phase_instantiate(L, M, H, N_M2P_LAYERS)

    # --- Phase 2: Forward pass ---
    fwd_time, memory_states, output = phase_forward(m2p, L, M, H)

    # --- Phase 3: Parameter generation ---
    gen_results = phase_parameter_generation(output, L, M, H, LORA_RANK)

    # --- Phase 4: Gradient flow ---
    all_grads_nonzero, grad_norms = phase_gradient_flow(m2p, memory_states)

    # --- Phase 6a: Input sensitivity BEFORE training ---
    mean_cos_pre, std_cos_pre, cos_sims_pre = phase_input_sensitivity(
        m2p, L, M, H, n_pairs=N_PAIRS, label="a (pre-training)"
    )

    # --- Phase 5: Convergence (modifies m2p weights) ---
    converged, losses = phase_convergence(m2p, memory_states, L, M, H)

    # --- Phase 6b: Input sensitivity AFTER training ---
    mean_cos_post, std_cos_post, cos_sims_post = phase_input_sensitivity(
        m2p, L, M, H, n_pairs=N_PAIRS, label="b (post-training)"
    )

    # --- Phase 7: Random baseline (PRIMARY KILL CRITERION K827) ---
    (k827_pass, k827_p_pass, k827_effect_pass,
     mean_m2p, std_m2p, mean_rand, std_rand,
     diff_means, t_stat, p_value,
     m2p_cos_list, rand_cos_list) = phase_random_baseline(m2p, L, M, H, n_pairs=N_PAIRS)

    # --- K826: architecture portability ---
    k826_pass = True  # if we reached here, it compiled and ran

    # --- Results ---
    results = {
        "experiment": "shine_port",
        "revision": 3,
        "status": "provisional",
        "total_time_s": round(time.time() - t0, 1),
        "m2p_params": n_params,
        "m2p_params_mb": round(n_params * 4 / 1e6, 2),
        "forward_time_ms": round(fwd_time * 1000, 1),
        "gradient_flow": all_grads_nonzero,
        "converged": converged,
        "loss_ratio": round(losses[-1] / losses[0], 4),
        "config": {"L": L, "M": M, "H": H, "rank": LORA_RANK,
                   "m2p_layers": N_M2P_LAYERS, "n_pairs": N_PAIRS},
        "positional_embeddings": {
            "implemented": True,
            "init": "xavier_normal",
            "p_layer_shape": list(m2p.p_layer.shape),
            "p_token_shape": list(m2p.p_token.shape),
            "reference": "SHINE §3.4 Eq. 5",
        },
        "rmt_prediction": {
            "n_flat": n_flat,
            "expected_mean": 0.0,
            "expected_std": round(rmt_std, 4),
        },
        "input_sensitivity_pre_training": {
            "n_pairs": N_PAIRS,
            "mean_cosine": round(mean_cos_pre, 4),
            "std_cosine": round(std_cos_pre, 4),
            "mean_in_rmt_range": abs(mean_cos_pre) <= 0.1,
            "cosine_sims": [round(c, 4) for c in cos_sims_pre],
        },
        "input_sensitivity_post_training": {
            "n_pairs": N_PAIRS,
            "mean_cosine": round(mean_cos_post, 4),
            "std_cosine": round(std_cos_post, 4),
            "mean_in_rmt_range": abs(mean_cos_post) <= 0.1,
            "cosine_sims": [round(c, 4) for c in cos_sims_post],
        },
        "random_baseline": {
            "n_pairs": N_PAIRS,
            "mean_m2p_cos": round(mean_m2p, 4),
            "std_m2p_cos": round(std_m2p, 4),
            "mean_rand_cos": round(mean_rand, 4),
            "std_rand_cos": round(std_rand, 4),
            "diff_means": round(diff_means, 4),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 4),
            "p_less_than_0_05": k827_p_pass,
            "effect_size_gt_0_05": k827_effect_pass,
            "m2p_cos_list": [round(c, 4) for c in m2p_cos_list],
            "rand_cos_list": [round(c, 4) for c in rand_cos_list],
        },
        "parameter_generation": gen_results,
        "kill_criteria": {
            "K826": {
                "pass": k826_pass,
                "detail": "M2P Transformer (with Xavier-init positional embeddings) ported to MLX successfully",
            },
            "K827": {
                "pass": k827_pass,
                "criterion": "p < 0.05 AND |diff in means| > 0.05 (two-sample t-test, n=30)",
                "p_value": round(p_value, 4),
                "diff_means": round(diff_means, 4),
                "p_criterion_pass": k827_p_pass,
                "effect_criterion_pass": k827_effect_pass,
                "detail": (
                    f"t={t_stat:.3f}, p={p_value:.4f} {'< 0.05 PASS' if k827_p_pass else '>= 0.05 FAIL'}; "
                    f"|diff|={diff_means:.4f} {'> 0.05 PASS' if k827_effect_pass else '<= 0.05 FAIL'}"
                ),
                "note": "PRIMARY KILL CRITERION — tests whether M2P imposes structure beyond random projection",
            },
        },
        "all_pass": k826_pass and k827_pass,
    }

    log(f"\n{'='*60}")
    log(f"SUMMARY")
    log(f"{'='*60}")
    log(f"M2P Transformer: {n_params:,} params, {fwd_time*1000:.1f}ms forward")
    log(f"Positional embeddings: Xavier init (non-zero from start)")
    log(f"Pre-training  input sensitivity: mean_cos={mean_cos_pre:.4f}, std={std_cos_pre:.4f}")
    log(f"Post-training input sensitivity: mean_cos={mean_cos_post:.4f}, std={std_cos_post:.4f}")
    log(f"")
    log(f"PRIMARY K827 RESULT:")
    log(f"  M2P cosine:    mean={mean_m2p:.4f}, std={std_m2p:.4f}")
    log(f"  Random cosine: mean={mean_rand:.4f}, std={std_rand:.4f}")
    log(f"  t={t_stat:.3f}, p={p_value:.4f}, |diff|={diff_means:.4f}")
    log(f"  K827: {'PASS' if k827_pass else 'FAIL'}")
    log(f"")
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} — {v['detail']}")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED (K827 FAIL)'}")
    log(f"Status: {results['status']}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
