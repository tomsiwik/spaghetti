#!/usr/bin/env python3
"""Adapter hot-swap latency benchmark on Qwen3-0.6B.

Kill criteria:
  K951: total swap latency (inject + first token) < 50ms

Three components measured:
  (a) t_inject_only: inject_lora_b() cost (reference swap, no generation)
  (b) t_ttft_noswap: TTFT with adapter already injected (no inject before)
  (c) t_ttft_after_swap: inject B_new then generate (combined, realistic scenario)
  swap_overhead = t_ttft_after_swap - t_ttft_noswap

SMOKE_TEST=1 uses 5+10 runs instead of 10+50.

References:
  MATH.md (Theorems 1-3)
  Hu et al. arXiv:2106.09685 (LoRA)
  Finding #388 (M2P generation speed: 5.31ms forward, 268.7 GB/s BW)
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
from mlx.utils import tree_flatten

# Memory safety — MANDATORY per CODING_GUIDELINES
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load as mlx_load
from mlx_lm import generate as mlx_generate
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.base import create_attention_mask

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

# ---- Config ----------------------------------------------------------------
MODEL_ID = "mlx-community/Qwen3-0.6B-4bit"
LORA_RANK = 4
LORA_SCALE = 5.0
D_M2P = 1024
OUTPUT_SCALE = 0.032

WARMUP_RUNS = 3 if IS_SMOKE else 10
BENCH_RUNS = 5 if IS_SMOKE else 50
MAX_GEN_TOKENS = 1  # TTFT only — single decode step

EXPERIMENT_DIR = Path(__file__).parent
V4_DIR = EXPERIMENT_DIR.parent / "m2p_qwen06b_gsm8k_v4"
V4_M2P_PATH = V4_DIR / "m2p_weights.npz"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Two different prompts → two different hidden states → two different B-matrix sets
PROMPT_A = "Natalia sold clips to 48 of her friends in April and then sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
PROMPT_B = "A train travels at 60 mph for 2 hours, then at 80 mph for 3 hours. What is the total distance traveled?"

# ---- Logging ---------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def log_memory(label: str = "") -> None:
    active = mx.get_active_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB peak={peak:.2f}GB")


# ---- M2P Network (identical to v4) ----------------------------------------

class M2PNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_m2p, rank, q_proj_out, v_proj_out,
                 output_scale=0.032):
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
        h = mx.mean(layer_hs, axis=0)
        h = nn.gelu(self.enc_linear1(h))
        z = self.enc_linear2(h)
        B_q = []
        B_v = []
        for li in range(self.n_layers):
            B_q.append((self.b_heads_q[li](z).reshape(self.rank, -1) * self.output_scale))
            B_v.append((self.b_heads_v[li](z).reshape(self.rank, -1) * self.output_scale))
        return B_q, B_v


# ---- LoRA helpers ----------------------------------------------------------

def apply_lora_structure(model, n_layers: int) -> None:
    """Wrap q_proj/v_proj with LoRALinear; initialize lora_a to zeros."""
    d_model = model.args.hidden_size
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.q_proj = LoRALinear.from_base(attn.q_proj, r=LORA_RANK, scale=LORA_SCALE)
        attn.v_proj = LoRALinear.from_base(attn.v_proj, r=LORA_RANK, scale=LORA_SCALE)
    model.freeze()


def inject_lora_b(model, B_q_layers: list, B_v_layers: list) -> None:
    """Set lora_b on all attention layers. This is the 'adapter swap' operation."""
    for li, layer in enumerate(model.model.layers):
        layer.self_attn.q_proj.lora_b = B_q_layers[li]
        layer.self_attn.v_proj.lora_b = B_v_layers[li]
    mx.eval(model.parameters())


# ---- Hidden state extraction (base model, no LoRA) -------------------------

def extract_hidden_states(model, tokens_arr: mx.array) -> mx.array:
    """Extract per-layer mean-pooled hidden states from base model.

    Returns: (n_layers, d_model)
    """
    qwen3 = model.model
    h = qwen3.embed_tokens(tokens_arr)
    mask = create_attention_mask(h, None)
    layer_states = []
    for layer in qwen3.layers:
        normed = layer.input_layernorm(h)
        attn_out = layer.self_attn(normed, mask=mask, cache=None)
        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        layer_states.append(mx.mean(h[0], axis=0))
    return mx.stop_gradient(mx.stack(layer_states, axis=0))


# ---- Timing ----------------------------------------------------------------

def time_fn(fn, warmup: int, runs: int, label: str) -> dict:
    """Time a callable that triggers MLX computation. Calls mx.eval() to force sync."""
    log(f"\n[TIMING] {label}")
    for _ in range(warmup):
        fn()
    log(f"  warmup({warmup}) done")

    times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    m = mean(times_ms)
    s = stdev(times_ms) if len(times_ms) > 1 else 0.0
    log(f"  mean={m:.3f}ms std={s:.3f}ms min={min(times_ms):.3f}ms max={max(times_ms):.3f}ms")
    return {"mean_ms": m, "std_ms": s, "min_ms": min(times_ms), "max_ms": max(times_ms)}


# ---- Main ------------------------------------------------------------------

def main():
    log("=" * 70)
    log("exp_adapter_hotswap_latency — adapter swap latency benchmark")
    log(f"SMOKE={IS_SMOKE} | WARMUP={WARMUP_RUNS} | BENCH_RUNS={BENCH_RUNS}")
    log("=" * 70)

    # Phase 1: Load model
    log("\n[Phase 1] Loading Qwen3-0.6B (4-bit)...")
    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    log_memory("after model load")

    args = model.args
    n_layers = args.num_hidden_layers
    d_model = args.hidden_size
    q_proj_out = args.num_attention_heads * args.head_dim
    v_proj_out = args.num_key_value_heads * args.head_dim
    log(f"  n_layers={n_layers} d_model={d_model}")
    log(f"  q_proj_out={q_proj_out} v_proj_out={v_proj_out}")

    # Phase 2: Patch model with LoRA layers
    log("\n[Phase 2] Patching model with LoRA structure...")
    apply_lora_structure(model, n_layers)

    # Phase 3: Load M2P network
    log("\n[Phase 3] Loading M2P v4 network...")
    if not V4_M2P_PATH.exists():
        raise FileNotFoundError(f"M2P weights not found: {V4_M2P_PATH}")
    m2p = M2PNetwork(
        n_layers=n_layers, d_model=d_model, d_m2p=D_M2P, rank=LORA_RANK,
        q_proj_out=q_proj_out, v_proj_out=v_proj_out, output_scale=OUTPUT_SCALE,
    )
    weights = dict(mx.load(str(V4_M2P_PATH)))
    m2p.load_weights(list(weights.items()))
    mx.eval(m2p.parameters())
    log_memory("after M2P load")

    # Phase 4: Precompute two B-matrix sets (adapters for two "domains")
    log("\n[Phase 4] Precomputing B-matrix sets for two prompts...")

    def encode_prompt(text: str, max_len: int = 64) -> mx.array:
        toks = tokenizer.encode(text)[:max_len]
        return mx.array(toks)[None]

    tokens_a = encode_prompt(PROMPT_A)
    tokens_b = encode_prompt(PROMPT_B)
    mx.eval(tokens_a, tokens_b)

    # Temporarily remove LoRA B so extract_hidden_states uses base model
    # (inject zeros so LoRA contribution is zero during extraction)
    B_q_zero = [mx.zeros((LORA_RANK, q_proj_out), dtype=mx.float16) for _ in range(n_layers)]
    B_v_zero = [mx.zeros((LORA_RANK, v_proj_out), dtype=mx.float16) for _ in range(n_layers)]
    inject_lora_b(model, B_q_zero, B_v_zero)

    hs_a = extract_hidden_states(model, tokens_a)
    mx.eval(hs_a)
    log(f"  Hidden states A: {hs_a.shape}")

    hs_b = extract_hidden_states(model, tokens_b)
    mx.eval(hs_b)
    log(f"  Hidden states B: {hs_b.shape}")

    B_q_a, B_v_a = m2p(hs_a)
    mx.eval(*B_q_a, *B_v_a)
    log("  B-matrix set A (adapter_a): computed")

    B_q_b, B_v_b = m2p(hs_b)
    mx.eval(*B_q_b, *B_v_b)
    log("  B-matrix set B (adapter_b): computed")

    # Compute sizes
    b_size_bytes = sum(b.nbytes for b in B_q_a + B_v_a)
    log(f"  B-matrix size per adapter: {b_size_bytes / 1024:.1f} KB")
    log_memory("after B precompute")

    del m2p, hs_a, hs_b, B_q_zero, B_v_zero
    gc.collect()
    mx.clear_cache()
    log_memory("after cleanup")

    # Phase 5: Benchmark inject_only (reference swap with no generation)
    log("\n[Phase 5] Benchmarking inject_only (no generation)...")
    alternating = [True]  # alternate between A and B

    def inject_only():
        if alternating[0]:
            inject_lora_b(model, B_q_a, B_v_a)
        else:
            inject_lora_b(model, B_q_b, B_v_b)
        alternating[0] = not alternating[0]

    result_inject_only = time_fn(inject_only, WARMUP_RUNS, BENCH_RUNS, "inject_lora_b only")

    # Phase 6: Benchmark TTFT — no swap (same adapter already injected)
    log("\n[Phase 6] Benchmarking TTFT with no swap (baseline)...")
    inject_lora_b(model, B_q_a, B_v_a)  # Pre-inject adapter_a

    def generate_noswap():
        out = mlx_generate(
            model, tokenizer,
            prompt=PROMPT_A,
            max_tokens=MAX_GEN_TOKENS,
            verbose=False,
        )
        mx.eval()  # Ensure all metal work completes
        return out

    result_ttft_noswap = time_fn(generate_noswap, WARMUP_RUNS, BENCH_RUNS, "TTFT no-swap (baseline)")

    # Phase 7: Benchmark TTFT — with swap (inject new adapter then generate)
    log("\n[Phase 7] Benchmarking TTFT after swap (inject + generate)...")
    inject_lora_b(model, B_q_a, B_v_a)  # Start with adapter_a
    swap_state = [True]  # alternate injections

    def generate_with_swap():
        # Swap to the OTHER adapter before generating
        if swap_state[0]:
            inject_lora_b(model, B_q_b, B_v_b)
            prompt = PROMPT_B
        else:
            inject_lora_b(model, B_q_a, B_v_a)
            prompt = PROMPT_A
        swap_state[0] = not swap_state[0]

        out = mlx_generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=MAX_GEN_TOKENS,
            verbose=False,
        )
        mx.eval()
        return out

    result_ttft_swap = time_fn(generate_with_swap, WARMUP_RUNS, BENCH_RUNS, "TTFT after swap")

    # Phase 8: Compute derived metrics
    log("\n[Phase 8] Computing metrics...")

    t_inject = result_inject_only["mean_ms"]
    t_ttft_baseline = result_ttft_noswap["mean_ms"]
    t_ttft_after_swap = result_ttft_swap["mean_ms"]
    swap_overhead = t_ttft_after_swap - t_ttft_baseline

    # K951: total swap latency = inject + TTFT_after_swap
    # (conservative: t_inject is already embedded in t_ttft_after_swap,
    #  but we also measure it independently for decomposition)
    k951_value = t_ttft_after_swap  # This includes inject as it's measured end-to-end
    k951_pass = k951_value < 50.0

    log(f"\n  t_inject_only:     {t_inject:.3f} ms")
    log(f"  t_TTFT_baseline:   {t_ttft_baseline:.3f} ms")
    log(f"  t_TTFT_after_swap: {t_ttft_after_swap:.3f} ms")
    log(f"  swap_overhead:     {swap_overhead:.3f} ms")
    log(f"  K951 value:        {k951_value:.3f} ms (threshold 50ms)")
    log(f"  K951:              {'PASS' if k951_pass else 'FAIL'}")

    results = {
        "experiment": "exp_adapter_hotswap_latency",
        "smoke_test": IS_SMOKE,
        "config": {
            "model": MODEL_ID,
            "lora_rank": LORA_RANK,
            "warmup_runs": WARMUP_RUNS,
            "bench_runs": BENCH_RUNS,
            "max_gen_tokens": MAX_GEN_TOKENS,
            "b_size_bytes": b_size_bytes,
        },
        "results": {
            "inject_only_ms": result_inject_only,
            "ttft_noswap_ms": result_ttft_noswap,
            "ttft_after_swap_ms": result_ttft_swap,
            "swap_overhead_ms": swap_overhead,
            "k951_value_ms": k951_value,
        },
        "kill_criteria": {
            "K951": {"threshold_ms": 50.0, "measured_ms": k951_value, "pass": k951_pass},
        },
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n[DONE] Results written to {RESULTS_FILE}")

    if k951_pass:
        log("\n✓ K951 PASS: Hot-swap latency within product threshold")
    else:
        log(f"\n✗ K951 FAIL: {k951_value:.1f}ms exceeds 50ms threshold")


if __name__ == "__main__":
    main()
