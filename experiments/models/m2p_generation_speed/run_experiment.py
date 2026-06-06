#!/usr/bin/env python3
"""M2P generation speed benchmark — measure inference latency on M5 Pro.

Kill criteria:
  K947: M2P forward pass (mean of 100 calls) < 100 ms

Measures:
  1. Isolated M2P forward pass (given pre-extracted hidden states)
  2. Full pipeline: hidden-state extraction + M2P forward

Design:
  - Load Qwen3-0.6B (4-bit) and M2P weights from v4
  - Warm up with WARMUP_RUNS calls (discarded)
  - Time BENCH_RUNS calls for each component
  - mx.eval() after each call to force MLX lazy evaluation

SMOKE_TEST=1 reduces benchmark to 5+10 calls.

References:
  MATH.md Theorem 1 (BW lower bound)
  MATH.md Theorem 3 (K947 feasibility proof)
"""

import gc
import json
import os
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np

import mlx.core as mx
import mlx.nn as nn

# Memory safety — MANDATORY per CODING_GUIDELINES
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load as mlx_load
from mlx_lm.models.base import create_attention_mask
from mlx.utils import tree_flatten  # noqa: E402

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

# ---- Config ----------------------------------------------------------------
MODEL_ID = "mlx-community/Qwen3-0.6B-4bit"
LORA_RANK = 4
LORA_SCALE = 5.0
D_M2P = 1024
OUTPUT_SCALE = 0.032

INPUT_LEN = 10 if IS_SMOKE else 64  # token sequence length for hidden extraction
WARMUP_RUNS = 3 if IS_SMOKE else 10
BENCH_RUNS = 5 if IS_SMOKE else 100

EXPERIMENT_DIR = Path(__file__).parent
V4_DIR = EXPERIMENT_DIR.parent / "m2p_qwen06b_gsm8k_v4"
V4_M2P_PATH = V4_DIR / "m2p_weights.npz"
V2_DIR = EXPERIMENT_DIR.parent / "m2p_qwen06b_gsm8k_v2"
V2_LORA_A_PATH = V2_DIR / "lora_a_matrices.npz"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"


# ---- Logging ---------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def log_memory(label: str = "") -> None:
    active = mx.get_active_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB peak={peak:.2f}GB")


# ---- M2P Network (identical to v4) ----------------------------------------

class M2PNetwork(nn.Module):
    """Hypernetwork: context hidden states → LoRA B-matrices (v4 architecture)."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_m2p: int,
        rank: int,
        q_proj_out: int,
        v_proj_out: int,
        output_scale: float = 0.032,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.rank = rank
        self.d_m2p = d_m2p
        self.output_scale = output_scale

        self.enc_linear1 = nn.Linear(d_model, 2 * d_m2p)
        self.enc_linear2 = nn.Linear(2 * d_m2p, d_m2p)
        self.b_heads_q = [nn.Linear(d_m2p, rank * q_proj_out) for _ in range(n_layers)]
        self.b_heads_v = [nn.Linear(d_m2p, rank * v_proj_out) for _ in range(n_layers)]

    def __call__(self, layer_hs: mx.array):
        """Generate B-matrices from per-layer hidden states.

        Args:
            layer_hs: (n_layers, d_model)

        Returns:
            B_q_layers: list of (rank, q_proj_out) per layer
            B_v_layers: list of (rank, v_proj_out) per layer
        """
        h = mx.mean(layer_hs, axis=0)
        h = nn.gelu(self.enc_linear1(h))
        z = self.enc_linear2(h)

        B_q_layers = []
        B_v_layers = []
        for li in range(self.n_layers):
            b_q_flat = self.b_heads_q[li](z)
            b_v_flat = self.b_heads_v[li](z)
            B_q_layers.append(b_q_flat.reshape(self.rank, -1) * self.output_scale)
            B_v_layers.append(b_v_flat.reshape(self.rank, -1) * self.output_scale)

        return B_q_layers, B_v_layers


# ---- Hidden state extraction (simplified version of v4) --------------------

def extract_hidden_states(model, tokens_arr: mx.array) -> mx.array:
    """Extract per-layer mean-pooled hidden states (base model, no LoRA).

    Returns: (n_layers, d_model)
    """
    qwen3_model = model.model
    h = qwen3_model.embed_tokens(tokens_arr)  # (1, T, d_model)
    mask = create_attention_mask(h, None)

    layer_states = []
    for layer in qwen3_model.layers:
        normed = layer.input_layernorm(h)
        attn_out = layer.self_attn(normed, mask=mask, cache=None)
        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        layer_states.append(mx.mean(h[0], axis=0))

    return mx.stop_gradient(mx.stack(layer_states, axis=0))


# ---- Timing utility --------------------------------------------------------

def time_fn(fn, warmup: int, runs: int, label: str) -> dict:
    """Time a zero-argument callable that returns one or more MLX arrays.

    fn() must return an mx.array or a flat list/tuple of mx.array.
    Calls mx.eval() after each call to force evaluation of all outputs.
    Returns dict with mean_ms, std_ms, min_ms, max_ms.
    """
    log(f"\n[TIMING] {label}")
    log(f"  warmup={warmup} runs={runs}")

    def eval_result(r):
        if isinstance(r, (list, tuple)):
            mx.eval(*r)
        else:
            mx.eval(r)

    # Warm up
    for _ in range(warmup):
        eval_result(fn())
    log(f"  warmup complete")

    times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        eval_result(result)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    m = mean(times_ms)
    s = stdev(times_ms) if len(times_ms) > 1 else 0.0
    log(f"  mean={m:.2f}ms std={s:.2f}ms min={min(times_ms):.2f}ms max={max(times_ms):.2f}ms")
    return {
        "mean_ms": m,
        "std_ms": s,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


# ---- Main ------------------------------------------------------------------

def main():
    log("=" * 70)
    log("exp_m2p_generation_speed — M2P inference latency benchmark")
    log(f"SMOKE_TEST={IS_SMOKE} | INPUT_LEN={INPUT_LEN} | BENCH_RUNS={BENCH_RUNS}")
    log("=" * 70)

    # Phase 1: Load model and M2P weights
    log("\n[Phase 1] Loading Qwen3-0.6B...")
    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    log_memory("after model load")

    # Introspect model dims from args (weight shapes are packed for 4-bit)
    args = model.args
    n_layers = args.num_hidden_layers
    d_model = args.hidden_size
    q_proj_out = args.num_attention_heads * args.head_dim
    v_proj_out = args.num_key_value_heads * args.head_dim
    log(f"  n_layers={n_layers} d_model={d_model} q_proj_out={q_proj_out} v_proj_out={v_proj_out}")

    # Phase 2: Load M2P weights
    log("\n[Phase 2] Loading M2P weights from v4...")
    if not V4_M2P_PATH.exists():
        raise FileNotFoundError(f"M2P weights not found: {V4_M2P_PATH}")

    m2p = M2PNetwork(
        n_layers=n_layers,
        d_model=d_model,
        d_m2p=D_M2P,
        rank=LORA_RANK,
        q_proj_out=q_proj_out,
        v_proj_out=v_proj_out,
        output_scale=OUTPUT_SCALE,
    )
    weights = dict(mx.load(str(V4_M2P_PATH)))
    m2p.load_weights(list(weights.items()))
    mx.eval(m2p.parameters())

    # Count M2P params
    flat_params = tree_flatten(m2p.parameters())
    m2p_param_count = sum(v.size for _, v in flat_params if isinstance(v, mx.array))
    log(f"  M2P params: {m2p_param_count/1e6:.2f}M")
    log_memory("after M2P load")

    # Phase 3: Prepare sample input
    log("\n[Phase 3] Preparing sample input...")
    sample_text = "Natalia sold clips to 48 of her friends in April."
    tokens = tokenizer.encode(sample_text)[:INPUT_LEN]
    # Pad to INPUT_LEN if short
    while len(tokens) < INPUT_LEN:
        tokens = tokens + [tokenizer.eos_token_id or 0]
    tokens = tokens[:INPUT_LEN]
    tokens_arr = mx.array(tokens)[None]  # (1, T)
    mx.eval(tokens_arr)
    log(f"  Input shape: {tokens_arr.shape}")

    # Phase 4: Pre-extract hidden states for isolated M2P timing
    log("\n[Phase 4] Pre-extracting hidden states for isolated M2P timing...")
    hs = extract_hidden_states(model, tokens_arr)  # (n_layers, d_model)
    mx.eval(hs)
    log(f"  Hidden states shape: {hs.shape}")
    log_memory("after hidden extraction")

    # Phase 5: Benchmark isolated M2P forward pass
    log("\n[Phase 5] Benchmarking isolated M2P forward pass...")
    hs_const = hs  # Already evaluated

    def m2p_forward():
        B_q, B_v = m2p(hs_const)
        return B_q + B_v  # eval all outputs for accurate timing

    m2p_timing = time_fn(m2p_forward, warmup=WARMUP_RUNS, runs=BENCH_RUNS, label="M2P forward (isolated)")

    # Phase 6: Benchmark full pipeline (hidden extraction + M2P forward)
    log("\n[Phase 6] Benchmarking full pipeline (extraction + M2P)...")

    def full_pipeline():
        hs_new = extract_hidden_states(model, tokens_arr)
        B_q, B_v = m2p(hs_new)
        return B_q + B_v  # eval all outputs

    pipeline_timing = time_fn(full_pipeline, warmup=WARMUP_RUNS, runs=BENCH_RUNS, label="Full pipeline (extraction + M2P)")

    # Phase 7: Compute derived metrics
    log("\n[Phase 7] Results summary")
    log("=" * 70)
    m2p_mean = m2p_timing["mean_ms"]
    pipeline_mean = pipeline_timing["mean_ms"]
    extraction_mean = pipeline_mean - m2p_mean  # Approximate extraction cost

    log(f"  M2P forward (isolated):  {m2p_mean:.2f} ± {m2p_timing['std_ms']:.2f} ms")
    log(f"  Full pipeline:           {pipeline_mean:.2f} ± {pipeline_timing['std_ms']:.2f} ms")
    log(f"  Extraction overhead:     {extraction_mean:.2f} ms (approx)")
    log(f"  M2P param count:         {m2p_param_count/1e6:.2f}M")

    # K947: M2P forward < 100 ms
    k947_pass = m2p_mean < 100.0
    log(f"\n  K947: M2P forward mean {m2p_mean:.2f}ms < 100ms → {'PASS' if k947_pass else 'KILL'}")

    # BW efficiency check
    m2p_bytes = m2p_param_count * 4  # fp32
    actual_bw = (m2p_bytes / 1e9) / (m2p_mean / 1e3)  # GB/s
    peak_bw = 400.0  # M5 Pro spec
    bw_efficiency = actual_bw / peak_bw * 100
    log(f"\n  Memory BW used:          {actual_bw:.1f} GB/s ({bw_efficiency:.1f}% of {peak_bw} GB/s peak)")
    log(f"  BW lower bound:          {m2p_bytes/1e9/peak_bw*1000:.2f} ms")

    # Phase 8: Save results
    results = {
        "m2p_forward_mean_ms": m2p_mean,
        "m2p_forward_std_ms": m2p_timing["std_ms"],
        "m2p_forward_min_ms": m2p_timing["min_ms"],
        "m2p_forward_max_ms": m2p_timing["max_ms"],
        "pipeline_mean_ms": pipeline_mean,
        "pipeline_std_ms": pipeline_timing["std_ms"],
        "pipeline_min_ms": pipeline_timing["min_ms"],
        "pipeline_max_ms": pipeline_timing["max_ms"],
        "extraction_approx_ms": extraction_mean,
        "m2p_param_count": m2p_param_count,
        "n_layers": n_layers,
        "d_model": d_model,
        "input_len": INPUT_LEN,
        "bench_runs": BENCH_RUNS,
        "warmup_runs": WARMUP_RUNS,
        "actual_bw_gbps": actual_bw,
        "bw_efficiency_pct": bw_efficiency,
        "k947_pass": k947_pass,
        "kill_criteria": {
            "K947": {
                "text": "M2P forward pass mean < 100 ms",
                "result": "pass" if k947_pass else "kill",
                "value": m2p_mean,
                "threshold": 100.0,
            }
        }
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n  Results saved: {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    main()
