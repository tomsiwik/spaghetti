#!/usr/bin/env python3
"""MLX adapter inference speed benchmark on Apple Silicon.

Measures latency overhead of LoRA adapter composition using MLX (Metal GPU).
Three strategies: base (no adapters), pre-merge, runtime activation path.
Varies N in {1, 2, 4, 8}.

Usage:
    uv run python experiments/models/adapter_inference_speed_mlx/run_experiment.py
"""

import json
import math
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np


def flatten_params(params):
    """Flatten nested parameter dict/list into list of mx.array."""
    arrays = []
    if isinstance(params, dict):
        for v in params.values():
            arrays.extend(flatten_params(v))
    elif isinstance(params, (list, tuple)):
        for v in params:
            arrays.extend(flatten_params(v))
    elif isinstance(params, mx.array):
        arrays.append(params)
    return arrays


# ── Config ──────────────────────────────────────────────────────────────────

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
VOCAB_SIZE = 256
SEQ_LEN = 32
BATCH_SIZE = 1
LORA_RANK = 8
N_VALUES = [1, 2, 4, 8]

WARMUP_ITERS = 100
TIMED_ITERS = 500


# ── Model Components ────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        ms = mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True)
        x = x * mx.rsqrt(ms + self.eps)
        return (x * self.weight).astype(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x, mask=None):
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention with causal mask
        scores = (q @ k.transpose(0, 1, 3, 2)) / self.scale
        if mask is not None:
            scores = scores + mask
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=False)

    def __call__(self, x):
        return self.fc2(mx.maximum(self.fc1(x), 0))  # ReLU activation


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = RMSNorm(d_model)
        self.mlp = MLP(d_model)

    def __call__(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class MicroTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, x):
        B, T = x.shape
        h = self.embed(x)
        # Build causal mask
        mask = mx.triu(mx.full((T, T), float('-inf')), k=1)
        mask = mask.reshape(1, 1, T, T)
        for layer in self.layers:
            h = layer(h, mask)
        h = self.ln_f(h)
        return self.lm_head(h)


# ── LoRA Adapters ───────────────────────────────────────────────────────────

def create_lora_adapters(n_adapters: int, d_model: int, rank: int, n_layers: int):
    """Create random LoRA A, B matrices for all layers.

    Returns list of adapters, each adapter is a list of layer dicts.
    Each layer dict maps projection_name -> (A, B) where:
      A: (rank, d_in), B: (d_out, rank)
    """
    adapters = []
    for _ in range(n_adapters):
        adapter = []
        for _ in range(n_layers):
            layer_loras = {}
            # Attention projections: d -> d
            for name in ['wq', 'wk', 'wv', 'wo']:
                A = mx.random.normal((rank, d_model)) * 0.01
                B = mx.random.normal((d_model, rank)) * 0.01
                layer_loras[name] = (A, B)
            # MLP: fc1 d->4d, fc2 4d->d
            A = mx.random.normal((rank, d_model)) * 0.01
            B = mx.random.normal((4 * d_model, rank)) * 0.01
            layer_loras['fc1'] = (A, B)

            A = mx.random.normal((rank, 4 * d_model)) * 0.01
            B = mx.random.normal((d_model, rank)) * 0.01
            layer_loras['fc2'] = (A, B)

            adapter.append(layer_loras)
        adapters.append(adapter)
    mx.eval(*[v for a in adapters for layer in a for pair in layer.values() for v in pair])
    return adapters


def pre_merge_adapters(model: MicroTransformer, adapters: list, n_adapters: int):
    """Merge N adapters into model weights. Modifies weights in-place."""
    scale = 1.0 / n_adapters
    for layer_idx, layer in enumerate(model.layers):
        proj_map = {
            'wq': layer.attn.wq,
            'wk': layer.attn.wk,
            'wv': layer.attn.wv,
            'wo': layer.attn.wo,
            'fc1': layer.mlp.fc1,
            'fc2': layer.mlp.fc2,
        }
        for proj_name, proj in proj_map.items():
            delta = mx.zeros_like(proj.weight)
            for adapter in adapters[:n_adapters]:
                A, B = adapter[layer_idx][proj_name]
                # W += scale * B @ A
                delta = delta + B @ A
            proj.weight = proj.weight + scale * delta
    # Force evaluation of all modified weights
    mx.eval(*flatten_params(model.parameters()))


class RuntimeLoRATransformer(nn.Module):
    """Transformer that applies LoRA adapters at runtime via activation path."""

    def __init__(self, base_model: MicroTransformer, adapters: list, k: int):
        super().__init__()
        self.base = base_model
        self.adapters = adapters[:k]
        self.k = k
        self.scale = 1.0 / k

    def _apply_lora_linear(self, proj, x, layer_idx, proj_name):
        """Compute proj(x) + scale * sum_i B_i @ (A_i @ x)."""
        base_out = proj(x)
        lora_sum = None
        for adapter in self.adapters:
            A, B = adapter[layer_idx][proj_name]
            # x: (B, T, d_in), A: (r, d_in), B: (d_out, r)
            # A @ x^T -> (r, T) per batch, then B @ that -> (d_out, T)
            # More efficient: x @ A^T -> (B, T, r), then @ B^T -> (B, T, d_out)
            h = x @ A.T  # (B, T, r)
            h = h @ B.T  # (B, T, d_out)
            if lora_sum is None:
                lora_sum = h
            else:
                lora_sum = lora_sum + h
        return base_out + self.scale * lora_sum

    def __call__(self, tokens):
        B, T = tokens.shape
        h = self.base.embed(tokens)
        mask = mx.triu(mx.full((T, T), float('-inf')), k=1).reshape(1, 1, T, T)

        for layer_idx, layer in enumerate(self.base.layers):
            # Pre-norm
            normed = layer.ln1(h)

            # Attention with LoRA
            d = self.base.layers[0].attn.head_dim
            n_h = self.base.layers[0].attn.n_heads
            scale_attn = math.sqrt(d)

            q = self._apply_lora_linear(layer.attn.wq, normed, layer_idx, 'wq')
            k_proj = self._apply_lora_linear(layer.attn.wk, normed, layer_idx, 'wk')
            v = self._apply_lora_linear(layer.attn.wv, normed, layer_idx, 'wv')

            q = q.reshape(B, T, n_h, d).transpose(0, 2, 1, 3)
            k_proj = k_proj.reshape(B, T, n_h, d).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, n_h, d).transpose(0, 2, 1, 3)

            scores = (q @ k_proj.transpose(0, 1, 3, 2)) / scale_attn
            scores = scores + mask
            attn = mx.softmax(scores, axis=-1)
            attn_out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)

            attn_out = self._apply_lora_linear(layer.attn.wo, attn_out, layer_idx, 'wo')
            h = h + attn_out

            # MLP with LoRA
            normed2 = layer.ln2(h)
            mlp_h = self._apply_lora_linear(layer.mlp.fc1, normed2, layer_idx, 'fc1')
            mlp_h = mx.maximum(mlp_h, 0)
            mlp_out = self._apply_lora_linear(layer.mlp.fc2, mlp_h, layer_idx, 'fc2')
            h = h + mlp_out

        h = self.base.ln_f(h)
        return self.base.lm_head(h)


# ── Benchmarking ────────────────────────────────────────────────────────────

def benchmark(forward_fn, input_tokens, warmup: int, iters: int):
    """Benchmark a forward function with proper MLX synchronization."""
    # Warmup: compile Metal shaders, stabilize
    for _ in range(warmup):
        out = forward_fn(input_tokens)
        mx.eval(out)

    # Timed iterations
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = forward_fn(input_tokens)
        mx.eval(out)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    latencies = np.array(latencies)
    return {
        'mean_ms': float(np.mean(latencies) * 1000),
        'std_ms': float(np.std(latencies) * 1000),
        'median_ms': float(np.median(latencies) * 1000),
        'p95_ms': float(np.percentile(latencies, 95) * 1000),
        'min_ms': float(np.min(latencies) * 1000),
        'tokens_per_sec': float(SEQ_LEN / np.mean(latencies)),
    }


def copy_model_weights(model):
    """Deep copy all model parameters for restoration."""
    import copy
    saved = {}
    for name, param in model.parameters().items():
        # Store a copy
        saved[name] = mx.array(param)
    mx.eval(*saved.values())
    return saved


def restore_model_weights(model, saved):
    """Restore model weights from saved copy."""
    model.load_weights(list(saved.items()))
    mx.eval(*flatten_params(model.parameters()))


def main():
    print("=" * 70)
    print("MLX Adapter Inference Speed Benchmark")
    print("=" * 70)
    print(f"MLX version: {mx.__version__}")
    print(f"Config: d={D_MODEL}, L={N_LAYERS}, heads={N_HEADS}, rank={LORA_RANK}")
    print(f"Sequence: B={BATCH_SIZE}, T={SEQ_LEN}, V={VOCAB_SIZE}")
    print(f"Warmup: {WARMUP_ITERS}, Timed: {TIMED_ITERS}")
    print()

    # Create input tokens
    input_tokens = mx.array(np.random.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).astype(np.int32))
    mx.eval(input_tokens)

    # Build base model
    print("Building base model...")
    base_model = MicroTransformer(D_MODEL, N_HEADS, N_LAYERS, VOCAB_SIZE)
    mx.eval(*flatten_params(base_model.parameters()))

    # Count parameters
    n_params = sum(p.size for p in flatten_params(base_model.parameters()))
    print(f"Base model parameters: {n_params:,}")

    # Create all adapters (max N=8)
    print(f"Creating {max(N_VALUES)} LoRA adapters (rank={LORA_RANK})...")
    all_adapters = create_lora_adapters(max(N_VALUES), D_MODEL, LORA_RANK, N_LAYERS)

    # Count LoRA params per adapter
    lora_params = sum(A.size + B.size for pair in all_adapters[0] for (A, B) in pair.values())
    print(f"LoRA params per adapter: {lora_params:,}")
    print()

    results = {}

    # ── Benchmark 1: Base model ─────────────────────────────────────────
    print("Benchmarking: Base model (no adapters)...")
    base_stats = benchmark(base_model, input_tokens, WARMUP_ITERS, TIMED_ITERS)
    results['base'] = base_stats
    print(f"  Mean: {base_stats['mean_ms']:.3f} ms, Std: {base_stats['std_ms']:.3f} ms")
    print(f"  Tokens/sec: {base_stats['tokens_per_sec']:.1f}")
    print()

    # ── Benchmark 2: Pre-merge strategy ─────────────────────────────────
    print("--- Pre-Merge Strategy ---")
    pre_merge_results = {}
    for N in N_VALUES:
        # Rebuild fresh model each time
        model = MicroTransformer(D_MODEL, N_HEADS, N_LAYERS, VOCAB_SIZE)
        mx.eval(*flatten_params(model.parameters()))

        # Pre-merge N adapters
        pre_merge_adapters(model, all_adapters, N)

        print(f"  Pre-merge N={N}...")
        stats = benchmark(model, input_tokens, WARMUP_ITERS, TIMED_ITERS)
        overhead = (stats['mean_ms'] - base_stats['mean_ms']) / base_stats['mean_ms'] * 100
        stats['overhead_pct'] = overhead
        pre_merge_results[f'N={N}'] = stats
        print(f"    Mean: {stats['mean_ms']:.3f} ms, Overhead: {overhead:+.2f}%")

    results['pre_merge'] = pre_merge_results
    print()

    # ── Benchmark 3: Runtime activation path ────────────────────────────
    print("--- Runtime Activation Path ---")
    runtime_results = {}
    for N in N_VALUES:
        # Use k=N (apply all N adapters at runtime)
        rt_model = RuntimeLoRATransformer(base_model, all_adapters, k=N)

        print(f"  Runtime k={N}...")
        stats = benchmark(rt_model, input_tokens, WARMUP_ITERS, TIMED_ITERS)
        overhead = (stats['mean_ms'] - base_stats['mean_ms']) / base_stats['mean_ms'] * 100
        stats['overhead_pct'] = overhead
        runtime_results[f'k={N}'] = stats
        print(f"    Mean: {stats['mean_ms']:.3f} ms, Overhead: {overhead:+.2f}%")

    results['runtime'] = runtime_results
    print()

    # ── Analysis ────────────────────────────────────────────────────────
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Kill Criteria K1: single adapter overhead <= 15%
    k1_overhead = runtime_results['k=1']['overhead_pct']
    k1_pass = k1_overhead <= 15.0
    print(f"\nK1: Single-adapter runtime overhead = {k1_overhead:.2f}%")
    print(f"    Threshold: <= 15%")
    print(f"    Verdict: {'PASS' if k1_pass else 'FAIL'}")

    # Kill Criteria K2: scaling is sub-linear or linear
    # Fit overhead(k) = alpha * k^beta
    ks = np.array(N_VALUES, dtype=float)
    overheads = np.array([runtime_results[f'k={k}']['overhead_pct'] for k in N_VALUES])

    # Log-log fit for beta
    # Only fit where overhead > 0 (to avoid log issues)
    valid = overheads > 0
    if valid.sum() >= 2:
        log_k = np.log(ks[valid])
        log_oh = np.log(overheads[valid])
        beta, log_alpha = np.polyfit(log_k, log_oh, 1)
        alpha = np.exp(log_alpha)
    else:
        beta = 0.0
        alpha = 0.0

    k2_pass = beta <= 1.05  # small tolerance for noise
    print(f"\nK2: Scaling exponent beta = {beta:.3f}")
    print(f"    Fit: overhead = {alpha:.2f} * k^{beta:.3f}")
    print(f"    Threshold: beta <= 1.0 (linear or sub-linear)")
    print(f"    Verdict: {'PASS' if k2_pass else 'FAIL'}")

    # Pre-merge analysis
    pre_merge_overheads = [pre_merge_results[f'N={N}']['overhead_pct'] for N in N_VALUES]
    max_pre_merge = max(abs(o) for o in pre_merge_overheads)
    print(f"\nPre-merge: max |overhead| = {max_pre_merge:.2f}% (expected ~0%)")

    # Linearity check for runtime
    print(f"\nRuntime overhead by k:")
    print(f"  {'k':>4}  {'Overhead %':>12}  {'Theoretical %':>14}  {'Ratio':>8}")
    for k_val in N_VALUES:
        actual = runtime_results[f'k={k_val}']['overhead_pct']
        theoretical = k_val * 9.375
        ratio = actual / theoretical if theoretical > 0 else 0
        print(f"  {k_val:>4}  {actual:>12.2f}  {theoretical:>14.2f}  {ratio:>8.2f}")

    # ── Summary Table ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE")
    print(f"{'=' * 70}")
    print(f"{'Strategy':<25} {'N/k':>5} {'Mean ms':>10} {'Std ms':>10} {'Overhead':>10} {'Tok/s':>10}")
    print("-" * 70)
    print(f"{'Base':<25} {'--':>5} {base_stats['mean_ms']:>10.3f} {base_stats['std_ms']:>10.3f} {'0.00%':>10} {base_stats['tokens_per_sec']:>10.1f}")
    for N in N_VALUES:
        s = pre_merge_results[f'N={N}']
        print(f"{'Pre-merge':<25} {N:>5} {s['mean_ms']:>10.3f} {s['std_ms']:>10.3f} {s['overhead_pct']:>+9.2f}% {s['tokens_per_sec']:>10.1f}")
    for k_val in N_VALUES:
        s = runtime_results[f'k={k_val}']
        print(f"{'Runtime activation':<25} {k_val:>5} {s['mean_ms']:>10.3f} {s['std_ms']:>10.3f} {s['overhead_pct']:>+9.2f}% {s['tokens_per_sec']:>10.1f}")

    # ── Save results ────────────────────────────────────────────────────
    output_dir = Path(__file__).parent
    results_file = output_dir / "results.json"
    save_data = {
        'config': {
            'd_model': D_MODEL,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS,
            'vocab_size': VOCAB_SIZE,
            'seq_len': SEQ_LEN,
            'batch_size': BATCH_SIZE,
            'lora_rank': LORA_RANK,
            'warmup_iters': WARMUP_ITERS,
            'timed_iters': TIMED_ITERS,
            'mlx_version': mx.__version__,
            'n_base_params': int(n_params),
            'n_lora_params_per_adapter': int(lora_params),
        },
        'results': results,
        'analysis': {
            'k1_overhead_pct': float(k1_overhead),
            'k1_pass': bool(k1_pass),
            'k2_beta': float(beta),
            'k2_alpha': float(alpha),
            'k2_pass': bool(k2_pass),
            'pre_merge_max_overhead_pct': float(max_pre_merge),
        },
    }
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
