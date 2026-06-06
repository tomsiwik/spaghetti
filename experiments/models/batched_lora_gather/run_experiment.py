#!/usr/bin/env python3
"""Batched LoRA dispatch via stacked matmul on MLX.

Tests whether stacking K adapter matrices and using batched matmul
is faster than sequential per-adapter dispatch for runtime LoRA.

Kill criteria:
  K769: Batched LoRA speed < 85 tok/s (vs v3 73 tok/s baseline) → KILL
  K770: Quality MSE > 1e-6 vs sequential RuntimeLoRA → KILL
  K771: Memory > 3 GB total → KILL

Platform: Apple M5 Pro 48GB, MLX
"""

import gc
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Memory safety (CODING_GUIDELINES mandatory)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# ──────────────────────────────────────────────────────────────────────
# Config: two scales to test the mechanism + production relevance
# ──────────────────────────────────────────────────────────────────────

# Micro scale (fast iteration, mechanism test)
MICRO = {
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 4,
    "vocab_size": 256,
    "lora_rank": 8,
    "seq_len": 1,       # Single token (autoregressive decode)
    "warmup": 200,
    "timed": 1000,
}

# Production scale (d=2560, matching BitNet-2B-4T)
PROD = {
    "d_model": 2560,
    "n_heads": 32,
    "n_layers": 30,
    "vocab_size": 151936,  # Qwen tokenizer
    "lora_rank": 16,
    "seq_len": 1,
    "warmup": 50,
    "timed": 200,
}

K_VALUES = [1, 2, 3, 5]  # Number of active adapters


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def sdpa(q, k, v, mask=None):
    """Wrapper for scaled_dot_product_attention that handles None mask."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    if mask is not None:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


# ──────────────────────────────────────────────────────────────────────
# Model definition (minimal transformer for speed benchmarking)
# ──────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x, mask=None):
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        out = sdpa(q, k, v, mask=mask)
        return self.wo(out.transpose(0, 2, 1, 3).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden = 4 * d_model
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.ln2 = nn.RMSNorm(d_model)
        self.mlp = MLP(d_model)

    def __call__(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class MicroTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.embed = nn.Embedding(cfg["vocab_size"], d)
        self.layers = [TransformerBlock(d, cfg["n_heads"])
                       for _ in range(cfg["n_layers"])]
        self.ln_f = nn.RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg["vocab_size"], bias=False)

    def __call__(self, x):
        B, T = x.shape
        h = self.embed(x)
        mask = None
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        for layer in self.layers:
            h = layer(h, mask)
        return self.lm_head(self.ln_f(h))


# ──────────────────────────────────────────────────────────────────────
# LoRA adapters
# ──────────────────────────────────────────────────────────────────────

# Module targets for LoRA (7 per layer: q, k, v, o, gate, up, down)
MODULE_NAMES = ["wq", "wk", "wv", "wo", "gate", "up", "down"]


def get_module_shapes(cfg):
    """Get (d_in, d_out) for each LoRA target module."""
    d = cfg["d_model"]
    hidden = 4 * d
    return {
        "wq": (d, d), "wk": (d, d), "wv": (d, d), "wo": (d, d),
        "gate": (d, hidden), "up": (d, hidden), "down": (hidden, d),
    }


def create_adapters(n_adapters, cfg):
    """Create LoRA adapter A, B pairs for all layers and modules.

    Returns: list of n_adapters, each is:
      list of n_layers, each is:
        dict of module_name -> (A, B)
        A: (r, d_in), B: (d_out, r)
    """
    r = cfg["lora_rank"]
    shapes = get_module_shapes(cfg)
    adapters = []
    for _ in range(n_adapters):
        adapter = []
        for _ in range(cfg["n_layers"]):
            layer_loras = {}
            for name, (d_in, d_out) in shapes.items():
                A = mx.random.normal((r, d_in)) * 0.01
                B = mx.random.normal((d_out, r)) * 0.01
                layer_loras[name] = (A, B)
            adapter.append(layer_loras)
        adapters.append(adapter)
    # Force evaluation
    all_params = [v for a in adapters for layer in a
                  for pair in layer.values() for v in pair]
    mx.eval(*all_params)
    return adapters


# ──────────────────────────────────────────────────────────────────────
# Strategy 1: Sequential (v3 baseline)
# Each adapter applied one at a time in a Python loop.
# ──────────────────────────────────────────────────────────────────────

class SequentialLoRATransformer(nn.Module):
    """v3-style sequential adapter dispatch."""

    def __init__(self, base, adapters, k):
        super().__init__()
        self.base = base
        self.adapters = adapters[:k]
        self.k = k
        self.scale = 1.0 / k

    def _apply_lora(self, proj, x, layer_idx, proj_name):
        base_out = proj(x)
        lora_sum = None
        for adapter in self.adapters:
            A, B = adapter[layer_idx][proj_name]
            h = x @ A.T   # (B, T, r)
            h = h @ B.T   # (B, T, d_out)
            if lora_sum is None:
                lora_sum = h
            else:
                lora_sum = lora_sum + h
        return base_out + self.scale * lora_sum

    def __call__(self, x):
        B, T = x.shape
        h = self.base.embed(x)
        mask = None
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)

        for li, layer in enumerate(self.base.layers):
            normed = layer.ln1(h)
            q = self._apply_lora(layer.attn.wq, normed, li, "wq")
            k_proj = self._apply_lora(layer.attn.wk, normed, li, "wk")
            v = self._apply_lora(layer.attn.wv, normed, li, "wv")

            B_sz, T_sz, C = q.shape
            hd = layer.attn.head_dim
            nh = layer.attn.n_heads
            q = q.reshape(B_sz, T_sz, nh, hd).transpose(0, 2, 1, 3)
            k_proj = k_proj.reshape(B_sz, T_sz, nh, hd).transpose(0, 2, 1, 3)
            v = v.reshape(B_sz, T_sz, nh, hd).transpose(0, 2, 1, 3)
            attn_out = sdpa(q, k_proj, v, mask=mask)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B_sz, T_sz, C)
            attn_out = self._apply_lora(layer.attn.wo, attn_out, li, "wo")
            h = h + attn_out

            normed2 = layer.ln2(h)
            gate_out = self._apply_lora(layer.mlp.gate, normed2, li, "gate")
            up_out = self._apply_lora(layer.mlp.up, normed2, li, "up")
            mlp_h = nn.silu(gate_out) * up_out
            mlp_out = self._apply_lora(layer.mlp.down, mlp_h, li, "down")
            h = h + mlp_out

        return self.base.lm_head(self.base.ln_f(h))


# ──────────────────────────────────────────────────────────────────────
# Strategy 2: Stacked batched matmul
# Stack K adapter A matrices into (K, r, d), compute all at once.
# ──────────────────────────────────────────────────────────────────────

class StackedLoRATransformer(nn.Module):
    """Batched adapter dispatch via stacked matmul."""

    def __init__(self, base, adapters, k):
        super().__init__()
        self.base = base
        self.k = k
        self.scale = 1.0 / k

        # Pre-stack adapter matrices per layer per module
        # A_stacked[layer][module] = (K, r, d_in)
        # B_stacked[layer][module] = (K, d_out, r)
        self.A_stacked = []
        self.B_stacked = []
        for li in range(len(base.layers)):
            layer_A = {}
            layer_B = {}
            for name in MODULE_NAMES:
                As = [adapters[i][li][name][0] for i in range(k)]
                Bs = [adapters[i][li][name][1] for i in range(k)]
                layer_A[name] = mx.stack(As)  # (K, r, d_in)
                layer_B[name] = mx.stack(Bs)  # (K, d_out, r)
            self.A_stacked.append(layer_A)
            self.B_stacked.append(layer_B)

        # Force eval of stacked tensors
        all_stacked = []
        for la, lb in zip(self.A_stacked, self.B_stacked):
            for name in MODULE_NAMES:
                all_stacked.extend([la[name], lb[name]])
        mx.eval(*all_stacked)

    def _apply_lora_stacked(self, proj, x, layer_idx, proj_name):
        base_out = proj(x)

        A_stack = self.A_stacked[layer_idx][proj_name]  # (K, r, d_in)
        B_stack = self.B_stacked[layer_idx][proj_name]  # (K, d_out, r)

        # x: (B, T, d_in) -> broadcast matmul with (K, d_in, r) = A_stack transposed
        # (B, T, d_in) @ (K, d_in, r) -> (K, B, T, r) via broadcast
        # We need: for each k, compute x @ A_k.T = x @ A_stack[k].T
        # MLX broadcast: (1, B, T, d_in) @ (K, 1, d_in, r) -> (K, B, T, r)

        # Reshape for broadcast: x -> (1, B, T, d) ; A.T -> (K, 1, d, r)
        # But MLX matmul broadcasts from the left. Let's use explicit einsum.
        # h_k = x @ A_k.T for all k -> einsum('btd,krd->kbtr')
        # which is equivalent to x @ A_stack.transpose(0,2,1) with broadcast

        # Method: x[None] @ A_stack[:, None].transpose(0, 1, 3, 2)
        # x[None]: (1, B, T, d), A.T: (K, 1, d, r) -> (K, B, T, r)
        A_T = mx.transpose(A_stack, axes=(0, 2, 1))  # (K, d_in, r)
        H = x[None] @ A_T[:, None]  # (K, B, T, r)

        # y_k = h_k @ B_k.T for all k
        B_T = mx.transpose(B_stack, axes=(0, 2, 1))  # (K, r, d_out) -- wait, B is (K, d_out, r)
        # B.T => (K, r, d_out)
        Y = H @ B_T[:, None]  # (K, B, T, d_out)

        # Sum over K dimension
        lora_sum = mx.sum(Y, axis=0)  # (B, T, d_out)

        return base_out + self.scale * lora_sum

    def __call__(self, x):
        B, T = x.shape
        h = self.base.embed(x)
        mask = None
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)

        for li, layer in enumerate(self.base.layers):
            normed = layer.ln1(h)
            q = self._apply_lora_stacked(layer.attn.wq, normed, li, "wq")
            k_proj = self._apply_lora_stacked(layer.attn.wk, normed, li, "wk")
            v = self._apply_lora_stacked(layer.attn.wv, normed, li, "wv")

            B_sz, T_sz, C = q.shape
            hd = layer.attn.head_dim
            nh = layer.attn.n_heads
            q = q.reshape(B_sz, T_sz, nh, hd).transpose(0, 2, 1, 3)
            k_proj = k_proj.reshape(B_sz, T_sz, nh, hd).transpose(0, 2, 1, 3)
            v = v.reshape(B_sz, T_sz, nh, hd).transpose(0, 2, 1, 3)
            attn_out = sdpa(q, k_proj, v, mask=mask)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B_sz, T_sz, C)
            attn_out = self._apply_lora_stacked(layer.attn.wo, attn_out, li, "wo")
            h = h + attn_out

            normed2 = layer.ln2(h)
            gate_out = self._apply_lora_stacked(layer.mlp.gate, normed2, li, "gate")
            up_out = self._apply_lora_stacked(layer.mlp.up, normed2, li, "up")
            mlp_h = nn.silu(gate_out) * up_out
            mlp_out = self._apply_lora_stacked(layer.mlp.down, mlp_h, li, "down")
            h = h + mlp_out

        return self.base.lm_head(self.base.ln_f(h))


# ──────────────────────────────────────────────────────────────────────
# Strategy 3: Concatenated A approach
# Concatenate K adapter A matrices along rank dim: (K*r, d)
# Single matmul, then split and multiply B separately.
# ──────────────────────────────────────────────────────────────────────

class ConcatLoRATransformer(nn.Module):
    """Concatenated A-matrix approach: one large matmul for all A projections."""

    def __init__(self, base, adapters, k):
        super().__init__()
        self.base = base
        self.k = k
        self.scale = 1.0 / k
        self.rank = adapters[0][0][MODULE_NAMES[0]][0].shape[0]

        # Pre-concatenate A matrices: (K*r, d_in) per layer per module
        # Pre-stack B matrices: (K, d_out, r) per layer per module
        self.A_concat = []
        self.B_list = []
        for li in range(len(base.layers)):
            layer_A = {}
            layer_B = {}
            for name in MODULE_NAMES:
                As = [adapters[i][li][name][0] for i in range(k)]  # each (r, d_in)
                Bs = [adapters[i][li][name][1] for i in range(k)]  # each (d_out, r)
                layer_A[name] = mx.concatenate(As, axis=0)  # (K*r, d_in)
                layer_B[name] = Bs  # keep as list for sequential B application
            self.A_concat.append(layer_A)
            self.B_list.append(layer_B)

        # Eval
        all_concat = []
        for la in self.A_concat:
            for name in MODULE_NAMES:
                all_concat.append(la[name])
        for lb in self.B_list:
            for name in MODULE_NAMES:
                all_concat.extend(lb[name])
        mx.eval(*all_concat)

    def _apply_lora_concat(self, proj, x, layer_idx, proj_name):
        base_out = proj(x)

        A_cat = self.A_concat[layer_idx][proj_name]  # (K*r, d_in)
        Bs = self.B_list[layer_idx][proj_name]        # list of K (d_out, r)

        # Single matmul for all A projections: x @ A_cat.T -> (B, T, K*r)
        h_all = x @ A_cat.T  # (B, T, K*r)

        # Split into K chunks of size r, apply B to each
        r = self.rank
        lora_sum = None
        for i in range(self.k):
            h_i = h_all[..., i * r:(i + 1) * r]  # (B, T, r)
            y_i = h_i @ Bs[i].T                    # (B, T, d_out)
            if lora_sum is None:
                lora_sum = y_i
            else:
                lora_sum = lora_sum + y_i

        return base_out + self.scale * lora_sum

    def __call__(self, x):
        B, T = x.shape
        h = self.base.embed(x)
        mask = None
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)

        for li, layer in enumerate(self.base.layers):
            normed = layer.ln1(h)
            q = self._apply_lora_concat(layer.attn.wq, normed, li, "wq")
            k_proj = self._apply_lora_concat(layer.attn.wk, normed, li, "wk")
            v = self._apply_lora_concat(layer.attn.wv, normed, li, "wv")

            B_sz, T_sz, C = q.shape
            hd = layer.attn.head_dim
            nh = layer.attn.n_heads
            q = q.reshape(B_sz, T_sz, nh, hd).transpose(0, 2, 1, 3)
            k_proj = k_proj.reshape(B_sz, T_sz, nh, hd).transpose(0, 2, 1, 3)
            v = v.reshape(B_sz, T_sz, nh, hd).transpose(0, 2, 1, 3)
            attn_out = sdpa(q, k_proj, v, mask=mask)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B_sz, T_sz, C)
            attn_out = self._apply_lora_concat(layer.attn.wo, attn_out, li, "wo")
            h = h + attn_out

            normed2 = layer.ln2(h)
            gate_out = self._apply_lora_concat(layer.mlp.gate, normed2, li, "gate")
            up_out = self._apply_lora_concat(layer.mlp.up, normed2, li, "up")
            mlp_h = nn.silu(gate_out) * up_out
            mlp_out = self._apply_lora_concat(layer.mlp.down, mlp_h, li, "down")
            h = h + mlp_out

        return self.base.lm_head(self.base.ln_f(h))


# ──────────────────────────────────────────────────────────────────────
# Strategy 4: addmm-fused sequential (best known v3 optimization)
# Uses mx.addmm where possible for base + lora in one call.
# ──────────────────────────────────────────────────────────────────────

class AddmmLoRATransformer(nn.Module):
    """v3 with addmm fusion: base_out + scale * lora_out in single fused op."""

    def __init__(self, base, adapters, k):
        super().__init__()
        self.base = base
        self.adapters = adapters[:k]
        self.k = k
        self.scale = 1.0 / k

    def _apply_lora_addmm(self, proj, x, layer_idx, proj_name):
        # First adapter with addmm: base + scale * (x@A0.T)@B0.T
        A0, B0 = self.adapters[0][layer_idx][proj_name]
        h0 = x @ A0.T  # (B, T, r)
        # proj(x) = x @ W.T; we want x @ W.T + scale * h0 @ B0.T
        base_out = proj(x)
        result = mx.addmm(base_out, h0, B0.T, alpha=self.scale)

        # Remaining adapters
        for i in range(1, self.k):
            A, B = self.adapters[i][layer_idx][proj_name]
            h = x @ A.T
            result = mx.addmm(result, h, B.T, alpha=self.scale)

        return result

    def __call__(self, x):
        B, T = x.shape
        h = self.base.embed(x)
        mask = None
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)

        for li, layer in enumerate(self.base.layers):
            normed = layer.ln1(h)
            q = self._apply_lora_addmm(layer.attn.wq, normed, li, "wq")
            k_proj = self._apply_lora_addmm(layer.attn.wk, normed, li, "wk")
            v = self._apply_lora_addmm(layer.attn.wv, normed, li, "wv")

            B_sz, T_sz, C = q.shape
            hd = layer.attn.head_dim
            nh = layer.attn.n_heads
            q = q.reshape(B_sz, T_sz, nh, hd).transpose(0, 2, 1, 3)
            k_proj = k_proj.reshape(B_sz, T_sz, nh, hd).transpose(0, 2, 1, 3)
            v = v.reshape(B_sz, T_sz, nh, hd).transpose(0, 2, 1, 3)
            attn_out = sdpa(q, k_proj, v, mask=mask)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B_sz, T_sz, C)
            attn_out = self._apply_lora_addmm(layer.attn.wo, attn_out, li, "wo")
            h = h + attn_out

            normed2 = layer.ln2(h)
            gate_out = self._apply_lora_addmm(layer.mlp.gate, normed2, li, "gate")
            up_out = self._apply_lora_addmm(layer.mlp.up, normed2, li, "up")
            mlp_h = nn.silu(gate_out) * up_out
            mlp_out = self._apply_lora_addmm(layer.mlp.down, mlp_h, li, "down")
            h = h + mlp_out

        return self.base.lm_head(self.base.ln_f(h))


# ──────────────────────────────────────────────────────────────────────
# Benchmarking
# ──────────────────────────────────────────────────────────────────────

def benchmark_model(model, tokens, warmup, timed):
    """Benchmark end-to-end forward pass."""
    # Warmup
    for _ in range(warmup):
        out = model(tokens)
        mx.eval(out)

    # Timed
    latencies = []
    for _ in range(timed):
        t0 = time.perf_counter()
        out = model(tokens)
        mx.eval(out)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    lat = np.array(latencies)
    mean_ms = float(np.mean(lat)) * 1000
    std_ms = float(np.std(lat)) * 1000
    tok_s = 1.0 / float(np.mean(lat))  # tokens per second (seq_len=1)
    return {
        "mean_ms": round(mean_ms, 4),
        "std_ms": round(std_ms, 4),
        "p50_ms": round(float(np.median(lat)) * 1000, 4),
        "p95_ms": round(float(np.percentile(lat, 95)) * 1000, 4),
        "tok_s": round(tok_s, 1),
    }


# ──────────────────────────────────────────────────────────────────────
# Phase 1: Numerical equivalence test
# ──────────────────────────────────────────────────────────────────────

def phase_numerical_equivalence():
    """Verify stacked and concat produce same output as sequential."""
    log("\n" + "=" * 70)
    log("[Phase 1] Numerical equivalence test (micro scale)")
    log("=" * 70)

    cfg = MICRO
    mx.random.seed(42)
    base = MicroTransformer(cfg)
    mx.eval(base.parameters())

    adapters = create_adapters(max(K_VALUES), cfg)
    tokens = mx.array([[42]], dtype=mx.int32)  # single token
    mx.eval(tokens)

    results = {}
    for k in K_VALUES:
        seq_model = SequentialLoRATransformer(base, adapters, k)
        stack_model = StackedLoRATransformer(base, adapters, k)
        concat_model = ConcatLoRATransformer(base, adapters, k)
        addmm_model = AddmmLoRATransformer(base, adapters, k)

        out_seq = seq_model(tokens)
        out_stack = stack_model(tokens)
        out_concat = concat_model(tokens)
        out_addmm = addmm_model(tokens)
        mx.eval(out_seq, out_stack, out_concat, out_addmm)

        mse_stack = float(mx.mean((out_seq - out_stack) ** 2).item())
        mse_concat = float(mx.mean((out_seq - out_concat) ** 2).item())
        mse_addmm = float(mx.mean((out_seq - out_addmm) ** 2).item())
        max_diff_stack = float(mx.max(mx.abs(out_seq - out_stack)).item())
        max_diff_concat = float(mx.max(mx.abs(out_seq - out_concat)).item())
        max_diff_addmm = float(mx.max(mx.abs(out_seq - out_addmm)).item())

        log(f"  K={k}: stacked MSE={mse_stack:.2e}, max_diff={max_diff_stack:.2e}")
        log(f"        concat  MSE={mse_concat:.2e}, max_diff={max_diff_concat:.2e}")
        log(f"        addmm   MSE={mse_addmm:.2e}, max_diff={max_diff_addmm:.2e}")

        results[f"K={k}"] = {
            "stacked_mse": mse_stack,
            "stacked_max_diff": max_diff_stack,
            "concat_mse": mse_concat,
            "concat_max_diff": max_diff_concat,
            "addmm_mse": mse_addmm,
            "addmm_max_diff": max_diff_addmm,
        }

    # K770 assessment
    max_mse = max(
        max(v["stacked_mse"], v["concat_mse"], v["addmm_mse"])
        for v in results.values()
    )
    k770_pass = max_mse < 1e-6
    log(f"\n  K770: max MSE = {max_mse:.2e} (threshold 1e-6) -> {'PASS' if k770_pass else 'FAIL'}")

    cleanup(base, seq_model, stack_model, concat_model, addmm_model)
    del adapters
    gc.collect()
    mx.clear_cache()

    return results, k770_pass


# ──────────────────────────────────────────────────────────────────────
# Phase 2: Micro-scale speed benchmark
# ──────────────────────────────────────────────────────────────────────

def phase_micro_speed():
    """Benchmark at micro scale (d=128) to test mechanism."""
    log("\n" + "=" * 70)
    log("[Phase 2] Micro-scale speed benchmark (d=128)")
    log("=" * 70)

    cfg = MICRO
    mx.random.seed(42)
    base = MicroTransformer(cfg)
    mx.eval(base.parameters())

    adapters = create_adapters(max(K_VALUES), cfg)
    tokens = mx.array([[42]], dtype=mx.int32)
    mx.eval(tokens)

    n_params = sum(p.size for _, p in nn.utils.tree_flatten(base.parameters()))
    log(f"  Base params: {n_params:,}")

    # Base model speed
    base_stats = benchmark_model(base, tokens, cfg["warmup"], cfg["timed"])
    log(f"  Base: {base_stats['tok_s']:.0f} tok/s ({base_stats['mean_ms']:.3f} ms)")

    results = {"base": base_stats}

    for k in K_VALUES:
        log(f"\n  --- K={k} adapters ---")

        seq_model = SequentialLoRATransformer(base, adapters, k)
        stack_model = StackedLoRATransformer(base, adapters, k)
        concat_model = ConcatLoRATransformer(base, adapters, k)
        addmm_model = AddmmLoRATransformer(base, adapters, k)

        seq_stats = benchmark_model(seq_model, tokens, cfg["warmup"], cfg["timed"])
        stack_stats = benchmark_model(stack_model, tokens, cfg["warmup"], cfg["timed"])
        concat_stats = benchmark_model(concat_model, tokens, cfg["warmup"], cfg["timed"])
        addmm_stats = benchmark_model(addmm_model, tokens, cfg["warmup"], cfg["timed"])

        log(f"  Sequential: {seq_stats['tok_s']:.0f} tok/s ({seq_stats['mean_ms']:.3f} ms)")
        log(f"  Stacked:    {stack_stats['tok_s']:.0f} tok/s ({stack_stats['mean_ms']:.3f} ms)")
        log(f"  Concat:     {concat_stats['tok_s']:.0f} tok/s ({concat_stats['mean_ms']:.3f} ms)")
        log(f"  Addmm:      {addmm_stats['tok_s']:.0f} tok/s ({addmm_stats['mean_ms']:.3f} ms)")

        speedup_stack = seq_stats["mean_ms"] / stack_stats["mean_ms"]
        speedup_concat = seq_stats["mean_ms"] / concat_stats["mean_ms"]
        speedup_addmm = seq_stats["mean_ms"] / addmm_stats["mean_ms"]
        log(f"  Speedup stacked/seq:  {speedup_stack:.3f}x")
        log(f"  Speedup concat/seq:   {speedup_concat:.3f}x")
        log(f"  Speedup addmm/seq:    {speedup_addmm:.3f}x")

        results[f"K={k}"] = {
            "sequential": seq_stats,
            "stacked": stack_stats,
            "concat": concat_stats,
            "addmm": addmm_stats,
            "speedup_stacked": round(speedup_stack, 4),
            "speedup_concat": round(speedup_concat, 4),
            "speedup_addmm": round(speedup_addmm, 4),
        }

    cleanup(base)
    del adapters
    gc.collect()
    mx.clear_cache()

    return results


# ──────────────────────────────────────────────────────────────────────
# Phase 3: Production-scale speed benchmark
# ──────────────────────────────────────────────────────────────────────

def phase_prod_speed():
    """Benchmark at production scale (d=2560, 30 layers) for K769/K771."""
    log("\n" + "=" * 70)
    log("[Phase 3] Production-scale speed benchmark (d=2560, L=30)")
    log("=" * 70)

    cfg = PROD
    mx.random.seed(42)

    log("  Building base model...")
    base = MicroTransformer(cfg)
    mx.eval(base.parameters())

    n_params = sum(p.size for _, p in nn.utils.tree_flatten(base.parameters()))
    log(f"  Base params: {n_params:,}")
    log_memory("after-base-build")

    # Create adapters for max K
    max_k = max(K_VALUES)
    log(f"  Creating {max_k} adapters (rank={cfg['lora_rank']})...")
    adapters = create_adapters(max_k, cfg)
    log_memory("after-adapter-create")

    tokens = mx.array([[42]], dtype=mx.int32)
    mx.eval(tokens)

    # Base model speed
    base_stats = benchmark_model(base, tokens, cfg["warmup"], cfg["timed"])
    log(f"  Base: {base_stats['tok_s']:.0f} tok/s ({base_stats['mean_ms']:.3f} ms)")

    results = {"base": base_stats}

    # Find the best strategy (from Phase 2 micro results, test all at prod)
    for k in K_VALUES:
        log(f"\n  --- K={k} adapters ---")

        # Sequential (v3 baseline)
        seq_model = SequentialLoRATransformer(base, adapters, k)
        seq_stats = benchmark_model(seq_model, tokens, cfg["warmup"], cfg["timed"])
        log(f"  Sequential: {seq_stats['tok_s']:.0f} tok/s ({seq_stats['mean_ms']:.3f} ms)")
        peak_seq = mx.get_peak_memory() / 1e9
        log(f"  Peak memory (seq): {peak_seq:.2f} GB")
        mx.reset_peak_memory()

        # Stacked
        stack_model = StackedLoRATransformer(base, adapters, k)
        stack_stats = benchmark_model(stack_model, tokens, cfg["warmup"], cfg["timed"])
        log(f"  Stacked:    {stack_stats['tok_s']:.0f} tok/s ({stack_stats['mean_ms']:.3f} ms)")
        peak_stack = mx.get_peak_memory() / 1e9
        log(f"  Peak memory (stacked): {peak_stack:.2f} GB")
        mx.reset_peak_memory()

        # Concat
        concat_model = ConcatLoRATransformer(base, adapters, k)
        concat_stats = benchmark_model(concat_model, tokens, cfg["warmup"], cfg["timed"])
        log(f"  Concat:     {concat_stats['tok_s']:.0f} tok/s ({concat_stats['mean_ms']:.3f} ms)")
        peak_concat = mx.get_peak_memory() / 1e9
        log(f"  Peak memory (concat): {peak_concat:.2f} GB")
        mx.reset_peak_memory()

        # Addmm
        addmm_model = AddmmLoRATransformer(base, adapters, k)
        addmm_stats = benchmark_model(addmm_model, tokens, cfg["warmup"], cfg["timed"])
        log(f"  Addmm:      {addmm_stats['tok_s']:.0f} tok/s ({addmm_stats['mean_ms']:.3f} ms)")
        peak_addmm = mx.get_peak_memory() / 1e9
        log(f"  Peak memory (addmm): {peak_addmm:.2f} GB")
        mx.reset_peak_memory()

        speedup_stack = seq_stats["mean_ms"] / stack_stats["mean_ms"]
        speedup_concat = seq_stats["mean_ms"] / concat_stats["mean_ms"]
        speedup_addmm = seq_stats["mean_ms"] / addmm_stats["mean_ms"]
        log(f"  Speedup stacked/seq:  {speedup_stack:.3f}x")
        log(f"  Speedup concat/seq:   {speedup_concat:.3f}x")
        log(f"  Speedup addmm/seq:    {speedup_addmm:.3f}x")

        results[f"K={k}"] = {
            "sequential": seq_stats,
            "stacked": stack_stats,
            "concat": concat_stats,
            "addmm": addmm_stats,
            "speedup_stacked": round(speedup_stack, 4),
            "speedup_concat": round(speedup_concat, 4),
            "speedup_addmm": round(speedup_addmm, 4),
            "peak_memory_gb": {
                "sequential": round(peak_seq, 3),
                "stacked": round(peak_stack, 3),
                "concat": round(peak_concat, 3),
                "addmm": round(peak_addmm, 3),
            },
        }

    cleanup(base)
    del adapters
    gc.collect()
    mx.clear_cache()

    return results


# ──────────────────────────────────────────────────────────────────────
# Phase 4: Isolated matmul microbenchmark
# Tests stacked vs sequential at the single-operation level
# ──────────────────────────────────────────────────────────────────────

def phase_isolated_matmul():
    """Isolate the matmul performance: stacked vs sequential for A projection."""
    log("\n" + "=" * 70)
    log("[Phase 4] Isolated matmul microbenchmark")
    log("=" * 70)

    results = {}

    for d, r, label in [(128, 8, "micro"), (2560, 16, "prod")]:
        log(f"\n  --- {label}: d={d}, r={r} ---")
        x = mx.random.normal((1, 1, d))  # single token
        mx.eval(x)

        for k in K_VALUES:
            # Create K A matrices
            As = [mx.random.normal((r, d)) * 0.01 for _ in range(k)]
            A_stack = mx.stack(As)  # (K, r, d)
            A_concat = mx.concatenate(As, axis=0)  # (K*r, d)
            mx.eval(*As, A_stack, A_concat)

            warmup = 500
            iters = 2000

            # Sequential
            for _ in range(warmup):
                hs = [x @ A.T for A in As]
                mx.eval(*hs)

            times_seq = []
            for _ in range(iters):
                t0 = time.perf_counter()
                hs = [x @ A.T for A in As]
                mx.eval(*hs)
                times_seq.append(time.perf_counter() - t0)

            # Stacked batched matmul
            A_T = mx.transpose(A_stack, axes=(0, 2, 1))  # (K, d, r)
            mx.eval(A_T)

            for _ in range(warmup):
                H = x[None] @ A_T[:, None]
                mx.eval(H)

            times_stack = []
            for _ in range(iters):
                t0 = time.perf_counter()
                H = x[None] @ A_T[:, None]
                mx.eval(H)
                times_stack.append(time.perf_counter() - t0)

            # Concat single matmul
            for _ in range(warmup):
                h_all = x @ A_concat.T
                mx.eval(h_all)

            times_concat = []
            for _ in range(iters):
                t0 = time.perf_counter()
                h_all = x @ A_concat.T
                mx.eval(h_all)
                times_concat.append(time.perf_counter() - t0)

            mean_seq = float(np.mean(times_seq)) * 1e6  # microseconds
            mean_stack = float(np.mean(times_stack)) * 1e6
            mean_concat = float(np.mean(times_concat)) * 1e6

            speedup_stack = mean_seq / mean_stack if mean_stack > 0 else 0
            speedup_concat = mean_seq / mean_concat if mean_concat > 0 else 0

            log(f"  K={k}: seq={mean_seq:.1f}us  stacked={mean_stack:.1f}us  "
                f"concat={mean_concat:.1f}us  "
                f"speedup_stack={speedup_stack:.2f}x  "
                f"speedup_concat={speedup_concat:.2f}x")

            key = f"{label}_K={k}"
            results[key] = {
                "d": d, "r": r, "k": k,
                "seq_us": round(mean_seq, 2),
                "stacked_us": round(mean_stack, 2),
                "concat_us": round(mean_concat, 2),
                "speedup_stacked": round(speedup_stack, 3),
                "speedup_concat": round(speedup_concat, 3),
            }

            del As, A_stack, A_concat, A_T

    cleanup()
    return results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 70)
    log("Batched LoRA Dispatch via Stacked Matmul on MLX")
    log("=" * 70)
    log(f"MLX version: {mx.__version__}")
    log(f"Device: {mx.default_device()}")
    log_memory("start")

    # Phase 1: Numerical equivalence
    equiv_results, k770_pass = phase_numerical_equivalence()
    log_memory("after-phase-1")

    # Phase 2: Micro speed
    micro_results = phase_micro_speed()
    log_memory("after-phase-2")

    # Phase 3: Production speed
    prod_results = phase_prod_speed()
    log_memory("after-phase-3")

    # Phase 4: Isolated matmul
    matmul_results = phase_isolated_matmul()
    log_memory("after-phase-4")

    # ══════════════════════════════════════════════════════════════════
    # Kill criteria assessment
    # ══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    # K769: Best strategy speed >= 85 tok/s at K=1 (prod scale)
    # Compare all strategies at K=1
    prod_k1 = prod_results.get("K=1", {})
    best_speed_k1 = 0
    best_strategy_k1 = "none"
    for strat in ["sequential", "stacked", "concat", "addmm"]:
        s = prod_k1.get(strat, {}).get("tok_s", 0)
        if s > best_speed_k1:
            best_speed_k1 = s
            best_strategy_k1 = strat

    # Also check at K with best combined performance
    all_speeds = {}
    for k_key in [f"K={k}" for k in K_VALUES]:
        kr = prod_results.get(k_key, {})
        for strat in ["sequential", "stacked", "concat", "addmm"]:
            s = kr.get(strat, {}).get("tok_s", 0)
            all_speeds[f"{k_key}_{strat}"] = s

    best_overall = max(all_speeds.values()) if all_speeds else 0
    best_overall_key = max(all_speeds, key=all_speeds.get) if all_speeds else "none"

    k769_pass = best_speed_k1 >= 85
    log(f"\nK769: Best speed at K=1 = {best_speed_k1:.1f} tok/s ({best_strategy_k1})")
    log(f"      Best overall = {best_overall:.1f} tok/s ({best_overall_key})")
    log(f"      Threshold: >= 85 tok/s")
    log(f"      K769 {'PASS' if k769_pass else 'FAIL'}")

    # K770: Numerical MSE
    log(f"\nK770: Numerical MSE < 1e-6 -> {'PASS' if k770_pass else 'FAIL'}")

    # K771: Memory < 3 GB
    max_mem = 0
    for k_key in [f"K={k}" for k in K_VALUES]:
        kr = prod_results.get(k_key, {})
        mems = kr.get("peak_memory_gb", {})
        for strat, mem in mems.items():
            if mem > max_mem:
                max_mem = mem

    k771_pass = max_mem < 3.0
    log(f"\nK771: Peak memory = {max_mem:.2f} GB (threshold 3.0 GB)")
    log(f"      K771 {'PASS' if k771_pass else 'FAIL'}")

    # Find best strategy overall
    log("\n" + "=" * 70)
    log("STRATEGY COMPARISON (Production Scale)")
    log("=" * 70)
    log(f"{'K':>3} {'Strategy':>12} {'tok/s':>8} {'ms':>8} {'Speedup':>8} {'Peak GB':>8}")
    log("-" * 60)

    for k in K_VALUES:
        k_key = f"K={k}"
        kr = prod_results.get(k_key, {})
        seq_ms = kr.get("sequential", {}).get("mean_ms", 999)
        for strat in ["sequential", "stacked", "concat", "addmm"]:
            stats = kr.get(strat, {})
            if stats:
                speedup = seq_ms / stats["mean_ms"] if stats["mean_ms"] > 0 else 0
                mem = kr.get("peak_memory_gb", {}).get(strat, 0)
                log(f"{k:>3} {strat:>12} {stats['tok_s']:>8.1f} {stats['mean_ms']:>8.3f} "
                    f"{speedup:>7.3f}x {mem:>8.2f}")

    # Overall verdict
    overall = "SUPPORTED" if (k769_pass and k770_pass and k771_pass) else "KILLED"
    log(f"\nOverall verdict: {overall}")
    if not k769_pass:
        log(f"  K769 FAIL: {best_speed_k1:.1f} tok/s < 85 tok/s threshold")
    if not k770_pass:
        log("  K770 FAIL: Numerical MSE > 1e-6")
    if not k771_pass:
        log(f"  K771 FAIL: Peak memory {max_mem:.2f} GB > 3.0 GB")

    # Save results
    all_results = {
        "experiment": "batched_lora_gather",
        "config": {
            "micro": MICRO,
            "prod": PROD,
            "k_values": K_VALUES,
            "mlx_version": mx.__version__,
        },
        "numerical_equivalence": equiv_results,
        "micro_speed": micro_results,
        "prod_speed": prod_results,
        "isolated_matmul": matmul_results,
        "kill_criteria": {
            "K769_speed_pass": k769_pass,
            "K769_best_tok_s": best_speed_k1,
            "K769_best_strategy": best_strategy_k1,
            "K770_mse_pass": k770_pass,
            "K771_memory_pass": k771_pass,
            "K771_peak_gb": max_mem,
        },
        "best_strategy": {
            "at_k1": best_strategy_k1,
            "overall": best_overall_key,
            "overall_tok_s": best_overall,
        },
        "verdict": overall,
        "total_time_s": round(time.time() - t0, 1),
    }

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {all_results['total_time_s']:.1f}s")


if __name__ == "__main__":
    main()
