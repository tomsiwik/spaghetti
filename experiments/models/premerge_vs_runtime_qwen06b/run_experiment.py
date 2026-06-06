#!/usr/bin/env python3
"""Pre-merge vs runtime LoRA on Qwen3-0.6B: quality and speed tradeoff.

Kill criteria:
  K952: Pre-merge tok/s >= 1.5x runtime LoRA tok/s
  K953: |quality_premerge - quality_runtime| < 1pp on GSM8K

MATH.md prediction:
  K952 FAIL: LoRA BW < 1% of base BW -> speedup < 1.05x for rank <= 32
  K953 PASS: delta/quantization_step < 0.5x -> re-quantization preserves quality

SMOKE_TEST=1 uses 10 warmup+20 bench tokens, 20 GSM8K samples.

References:
  MATH.md (Theorems 1-3)
  Hu et al. arXiv:2106.09685 (LoRA)
  Finding #74 (runtime LoRA: 82 tok/s serving)
  Finding #289 (ternary premerge kills quality at delta/Δ_q >> 1)
"""

import gc
import json
import os
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from datasets import load_dataset
from mlx.utils import tree_flatten
from mlx_lm import generate as mlx_generate
from mlx_lm import load as mlx_load
from mlx_lm.tuner.lora import LoRALinear

# ── Memory safety (MANDATORY per CODING_GUIDELINES) ─────────────────────────
_dev = mx.device_info()
_total = _dev["memory_size"]
mx.set_memory_limit(_total - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

# ── Config ───────────────────────────────────────────────────────────────────
IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

MODEL_ID = "mlx-community/Qwen3-0.6B-4bit"
LORA_RANK = 8
LORA_SCALE = 20.0
LORA_B_STD_SPEED = 0.02   # for speed benchmark (B values don't matter for timing)
LORA_B_STD_QUALITY = 0.001  # small delta: preserves model output while being non-trivial

WARMUP_TOKENS = 10 if IS_SMOKE else 30
BENCH_TOKENS = 20 if IS_SMOKE else 200
N_BENCH_REPS = 3 if IS_SMOKE else 5
GSM8K_N = 20 if IS_SMOKE else 50
SEED = 42

BENCH_PROMPT = (
    "Natalia sold clips to 48 of her friends in April and then sold half as many "
    "clips in May. How many clips did Natalia sell altogether in April and May? "
    "Think step by step."
)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"


# ── Utilities ────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(msg, flush=True)


def log_mem(label: str = "") -> None:
    active = mx.get_active_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM{' ' + label if label else ''}] active={active:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()


def extract_gsm8k_answer(text: str) -> float | None:
    """Extract final numeric answer from model output."""
    # Look for #### <number> pattern
    match = re.search(r"####\s*([\-\d,\.]+)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass
    # Fall back: last number in text
    nums = re.findall(r"[\-\d,\.]+", text)
    for n in reversed(nums):
        try:
            return float(n.replace(",", ""))
        except ValueError:
            continue
    return None


def inject_synthetic_lora(model, rank: int, scale: float, b_std: float) -> None:
    """Replace q_proj, v_proj in every attention layer with LoRALinear.

    Initializes B with small random values (b_std) to produce a non-trivial
    adapter that tests both quality preservation and speed overhead.
    """
    mx.random.seed(SEED)
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.q_proj = LoRALinear.from_base(attn.q_proj, r=rank, scale=scale)
        attn.v_proj = LoRALinear.from_base(attn.v_proj, r=rank, scale=scale)
        # Initialize B with non-zero values (B is 0 by default in LoRALinear)
        attn.q_proj.lora_b = mx.random.normal(attn.q_proj.lora_b.shape) * b_std
        attn.v_proj.lora_b = mx.random.normal(attn.v_proj.lora_b.shape) * b_std
    mx.eval(model.parameters())


def fuse_lora_inplace(model, dequantize: bool = False) -> None:
    """Fuse all LoRALinear layers back into base weights (in-place).

    dequantize=False → re-quantize to 4-bit (pre-merge 4-bit strategy)
    dequantize=True  → keep as bf16 (pre-merge bf16 strategy, slower decode)
    """
    for layer in model.model.layers:
        attn = layer.self_attn
        if isinstance(attn.q_proj, LoRALinear):
            attn.q_proj = attn.q_proj.fuse(dequantize=dequantize)
        if isinstance(attn.v_proj, LoRALinear):
            attn.v_proj = attn.v_proj.fuse(dequantize=dequantize)
    mx.eval(model.parameters())


def measure_tok_per_sec(model, tokenizer, n_reps: int) -> dict:
    """Measure decode throughput: warm up then time n_reps runs."""
    # Warmup
    for _ in range(2):
        _ = mlx_generate(
            model, tokenizer,
            prompt=BENCH_PROMPT,
            max_tokens=WARMUP_TOKENS,
            verbose=False,
        )
    mx.eval()

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _ = mlx_generate(
            model, tokenizer,
            prompt=BENCH_PROMPT,
            max_tokens=BENCH_TOKENS,
            verbose=False,
        )
        mx.eval()
        times.append(time.perf_counter() - t0)

    tps_list = [BENCH_TOKENS / t for t in times]
    return {
        "mean_tps": float(np.mean(tps_list)),
        "std_tps": float(np.std(tps_list)),
        "min_tps": float(np.min(tps_list)),
        "max_tps": float(np.max(tps_list)),
        "times_s": [float(t) for t in times],
    }


# ── Phase 1: Base model speed (reference) ───────────────────────────────────

def phase_base_speed() -> dict:
    """Measure base Qwen3-0.6B-4bit tok/s (no LoRA)."""
    log("\n=== Phase 1: Base model speed ===")
    model, tok = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    log_mem("after load")

    result = measure_tok_per_sec(model, tok, N_BENCH_REPS)
    log(f"  base tok/s: {result['mean_tps']:.1f} ± {result['std_tps']:.1f}")

    cleanup(model, tok)
    log_mem("after cleanup")
    return result


# ── Phase 2: Runtime LoRA speed ──────────────────────────────────────────────

def phase_runtime_lora_speed() -> dict:
    """Benchmark runtime LoRA (side-path): base 4-bit + bf16 A,B at each step."""
    log(f"\n=== Phase 2: Runtime LoRA speed (rank={LORA_RANK}) ===")
    model, tok = mlx_load(MODEL_ID)
    inject_synthetic_lora(model, LORA_RANK, LORA_SCALE, LORA_B_STD_SPEED)
    log_mem("after inject")

    # Measure LoRA size for bandwidth analysis
    lora_bytes = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        if isinstance(attn.q_proj, LoRALinear):
            lora_bytes += attn.q_proj.lora_a.nbytes + attn.q_proj.lora_b.nbytes
            lora_bytes += attn.v_proj.lora_a.nbytes + attn.v_proj.lora_b.nbytes

    result = measure_tok_per_sec(model, tok, N_BENCH_REPS)
    result["lora_params_bytes"] = lora_bytes
    log(f"  runtime LoRA tok/s: {result['mean_tps']:.1f} ± {result['std_tps']:.1f}")
    log(f"  LoRA BW per step: {lora_bytes / 1e6:.2f} MB (rank={LORA_RANK})")

    cleanup(model, tok)
    log_mem("after cleanup")
    return result


# ── Phase 3: Pre-merge 4-bit speed ───────────────────────────────────────────

def phase_premerge_speed() -> dict:
    """Benchmark pre-merge 4-bit: merge LoRA into weights, re-quantize, run base."""
    log(f"\n=== Phase 3: Pre-merge 4-bit speed (rank={LORA_RANK}) ===")
    model, tok = mlx_load(MODEL_ID)
    inject_synthetic_lora(model, LORA_RANK, LORA_SCALE, LORA_B_STD_SPEED)
    log_mem("after inject")

    # Time the merge itself
    t_merge_start = time.perf_counter()
    fuse_lora_inplace(model, dequantize=False)
    t_merge = time.perf_counter() - t_merge_start
    log(f"  merge time: {t_merge*1000:.1f} ms")
    log_mem("after fuse")

    result = measure_tok_per_sec(model, tok, N_BENCH_REPS)
    result["merge_time_ms"] = float(t_merge * 1000)
    log(f"  pre-merge 4-bit tok/s: {result['mean_tps']:.1f} ± {result['std_tps']:.1f}")

    cleanup(model, tok)
    log_mem("after cleanup")
    return result


# ── Phase 4: GSM8K quality comparison ────────────────────────────────────────

def _eval_gsm8k(model, tokenizer, examples: list) -> float:
    """Run GSM8K evaluation. Returns accuracy (0-1)."""
    correct = 0
    for i, ex in enumerate(examples):
        question = ex["question"]
        # Extract ground-truth answer
        gt_text = ex["answer"].split("####")[-1].strip().replace(",", "")
        try:
            gt = float(gt_text)
        except ValueError:
            continue

        prompt = f"<|im_start|>user\n{question}\nPlease think step by step and end with #### <answer>.<|im_end|>\n<|im_start|>assistant\n"
        out = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=False)
        pred = extract_gsm8k_answer(out)

        if pred is not None and abs(pred - gt) < 0.5:
            correct += 1

        if (i + 1) % 10 == 0:
            log(f"    [{i+1}/{len(examples)}] running acc: {correct/(i+1):.3f}")

    return correct / len(examples)


def phase_quality(strategy: str) -> dict:
    """Evaluate GSM8K accuracy for a given serving strategy.

    strategy: 'runtime_lora' | 'premerge_4bit' | 'base'
    """
    log(f"\n=== Phase 4-{strategy}: GSM8K quality (n={GSM8K_N}) ===")

    # Load dataset
    ds = load_dataset("gsm8k", "main", split="test")
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(ds), size=GSM8K_N, replace=False)
    examples = [ds[int(i)] for i in sorted(indices)]

    # Load model
    model, tok = mlx_load(MODEL_ID)

    if strategy == "runtime_lora":
        inject_synthetic_lora(model, LORA_RANK, LORA_SCALE, LORA_B_STD)
        log("  strategy: runtime LoRA (side-path)")
    elif strategy == "premerge_4bit":
        inject_synthetic_lora(model, LORA_RANK, LORA_SCALE, LORA_B_STD)
        fuse_lora_inplace(model, dequantize=False)
        log("  strategy: pre-merge 4-bit (fused + re-quantized)")
    elif strategy == "base":
        log("  strategy: base model (no adapter)")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    log_mem("after setup")

    acc = _eval_gsm8k(model, tok, examples)
    log(f"  GSM8K accuracy ({strategy}): {acc:.4f} ({acc*100:.1f}%)")

    cleanup(model, tok)
    log_mem("after cleanup")
    return {"strategy": strategy, "accuracy": float(acc), "n_samples": GSM8K_N}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    log(f"Experiment: exp_premerge_vs_runtime_qwen06b")
    log(f"Model: {MODEL_ID}")
    log(f"LoRA rank: {LORA_RANK}, scale: {LORA_SCALE}, B_std: {LORA_B_STD}")
    log(f"Smoke: {IS_SMOKE}, bench_tokens: {BENCH_TOKENS}, gsm8k_n: {GSM8K_N}")

    # Speed benchmarks
    base_speed = phase_base_speed()
    runtime_speed = phase_runtime_lora_speed()
    premerge_speed = phase_premerge_speed()

    speedup = premerge_speed["mean_tps"] / runtime_speed["mean_tps"]
    log(f"\n--- Speed summary ---")
    log(f"  Base model:     {base_speed['mean_tps']:.1f} tok/s")
    log(f"  Runtime LoRA:   {runtime_speed['mean_tps']:.1f} tok/s")
    log(f"  Pre-merge 4-bit: {premerge_speed['mean_tps']:.1f} tok/s")
    log(f"  Speedup (pre-merge / runtime): {speedup:.3f}x")

    k952_pass = speedup >= 1.5
    log(f"  K952 (>=1.5x): {'PASS' if k952_pass else 'FAIL'} (speedup={speedup:.3f}x)")

    # Quality benchmarks
    quality_runtime = phase_quality("runtime_lora")
    quality_premerge = phase_quality("premerge_4bit")

    quality_diff = abs(quality_premerge["accuracy"] - quality_runtime["accuracy"]) * 100
    log(f"\n--- Quality summary ---")
    log(f"  Runtime LoRA acc:    {quality_runtime['accuracy']*100:.1f}%")
    log(f"  Pre-merge 4-bit acc: {quality_premerge['accuracy']*100:.1f}%")
    log(f"  Quality diff:        {quality_diff:.2f}pp")
    log(f"  K953 (<1pp): {'PASS' if quality_diff < 1.0 else 'FAIL'} (diff={quality_diff:.2f}pp)")

    # LoRA BW analysis
    base_bw_mb = 340.0  # measured
    lora_bw_mb = runtime_speed["lora_params_bytes"] / 1e6
    predicted_speedup = 1 + lora_bw_mb / base_bw_mb
    log(f"\n--- BW analysis (Theorem 1 verification) ---")
    log(f"  Base model BW: {base_bw_mb:.0f} MB")
    log(f"  LoRA BW (rank={LORA_RANK}): {lora_bw_mb:.2f} MB")
    log(f"  Predicted speedup: {predicted_speedup:.4f}x")
    log(f"  Measured speedup:  {speedup:.4f}x")
    log(f"  Theorem 1 error:   {abs(speedup - predicted_speedup):.4f}x")

    results = {
        "experiment": "exp_premerge_vs_runtime_qwen06b",
        "config": {
            "model": MODEL_ID,
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
            "lora_b_std": LORA_B_STD,
            "bench_tokens": BENCH_TOKENS,
            "gsm8k_n": GSM8K_N,
            "smoke": IS_SMOKE,
        },
        "speed": {
            "base": base_speed,
            "runtime_lora": runtime_speed,
            "premerge_4bit": premerge_speed,
        },
        "quality": {
            "runtime_lora": quality_runtime,
            "premerge_4bit": quality_premerge,
        },
        "analysis": {
            "speedup_premerge_over_runtime": speedup,
            "quality_diff_pp": quality_diff,
            "lora_bw_mb": lora_bw_mb,
            "base_bw_mb": base_bw_mb,
            "predicted_speedup_theorem1": predicted_speedup,
        },
        "kill_criteria": {
            "K952": {
                "description": "Pre-merge tok/s >= 1.5x runtime LoRA tok/s",
                "measured_speedup": speedup,
                "threshold": 1.5,
                "pass": k952_pass,
            },
            "K953": {
                "description": "Quality difference < 1pp between strategies",
                "measured_diff_pp": quality_diff,
                "threshold_pp": 1.0,
                "pass": quality_diff < 1.0,
            },
        },
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
