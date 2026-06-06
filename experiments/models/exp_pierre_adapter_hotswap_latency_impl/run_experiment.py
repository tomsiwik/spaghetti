#!/usr/bin/env python3
"""exp_pierre_adapter_hotswap_latency_impl — measure K1953 and K1954.

Inherits theorems from parent MATH.md §3/§4. F#666: both KCs are target-metrics
(latency, token-identity); no proxy pairing needed.

Scope (parent §8):
  - mlx-community/gemma-4-e4b-it-4bit
  - targets: self_attn.v_proj + self_attn.o_proj (F#627)
  - rank 6, alpha 8.0
  - N=5 synthetic adapter_B sets
  - A: partitioned QR per (layer, key) -> N orthogonal domain-A blocks (F#562)
  - B: random N(0, 0.01^2)

Kill criteria:
  K1953: t_attach_median over 20 runs > 100 ms -> FAIL
  K1954: same-adapter detach/re-attach glitch-count > 1 over positions {1,2,4,8} -> FAIL

SMOKE_TEST=1 -> reduced runs; PROVISIONAL verdict not supported.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np

import mlx.core as mx
import mlx.nn as nn

# Memory safety (researcher hat)
_info = mx.device_info()
mx.set_memory_limit(_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

# pierre.py lives at repo_root/pierre/pierre.py
EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pierre.pierre import RuntimeLoRA  # noqa: E402

from mlx_lm import load as mlx_load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx.utils import tree_unflatten  # noqa: E402


# Gemma 4 E4B uses `model.layers` (42 DecoderLayers), not `model.model.layers`
# (qwen3/llama-style). Re-implement attach/detach here to honor the model layout.
# pierre.pierre is untouched.

def attach_adapter(model, frozen_A, adapter_B, domain_idx, alpha):
    layers = model.layers
    count = 0
    for li in range(len(layers)):
        updates = []
        for key in TARGET_KEYS:
            bk = f"model.layers.{li}.{key}.lora_b"
            ak = f"layer_{li}_{key}_domain_{domain_idx}"
            if bk not in adapter_B or ak not in frozen_A:
                continue
            m = layers[li]
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is None:
                continue
            A = mx.array(frozen_A[ak]).astype(mx.bfloat16)
            B = adapter_B[bk].astype(mx.bfloat16)
            updates.append((key, RuntimeLoRA(m, A, B, alpha)))
            count += 1
        if updates:
            layers[li].update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    return count


def detach_adapters(model):
    layers = model.layers
    count = 0
    for layer in layers:
        updates = []
        for key in TARGET_KEYS:
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if isinstance(m, RuntimeLoRA):
                updates.append((key, m.base))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    return count


# ---- Config --------------------------------------------------------------

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
TARGET_KEYS = ["self_attn.v_proj", "self_attn.o_proj"]  # F#627
LORA_RANK = 6
LORA_ALPHA = 8.0
N_ADAPTERS = 5

WARMUP_RUNS = 2 if IS_SMOKE else 3
BENCH_RUNS = 5 if IS_SMOKE else 20
GEN_TOKENS = 8 if IS_SMOKE else 16
SWAP_POSITIONS = [1, 2, 4, 8]

K1953_THRESHOLD_MS = 100.0
K1954_THRESHOLD_GLITCHES = 1  # > 1 fails

RESULTS_FILE = EXP_DIR / "results.json"

PROMPT = (
    "Explain in three sentences the role of the Pythagorean theorem in "
    "Euclidean geometry and give one concrete numerical example."
)


def log(msg: str) -> None:
    print(msg, flush=True)


def log_mem(label: str = "") -> None:
    active = mx.get_active_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB peak={peak:.2f}GB")


# ---- Dimension inference -------------------------------------------------

def _module_for_key(layer, key: str):
    m = layer
    for part in key.split("."):
        m = getattr(m, part)
    return m


def _dims_of(mod) -> tuple:
    """Return (in_features, out_features) for a Linear or QuantizedLinear."""
    W = mod.weight
    out_f = int(W.shape[0])
    in_packed = int(W.shape[1])
    bits = getattr(mod, "bits", None)
    if bits is not None:
        in_f = in_packed * (32 // int(bits))
    else:
        in_f = in_packed
    return (in_f, out_f)


def infer_per_layer_dims(model) -> list:
    """Return list[{key: (in, out)}] per layer.

    Gemma 4 E4B mixes layer types (e.g. sliding-window vs global attention)
    with different head configurations, so per-layer inference is required.
    """
    per_layer = []
    for layer in model.layers:
        d = {}
        for key in TARGET_KEYS:
            mod = _module_for_key(layer, key)
            d[key] = _dims_of(mod)
        per_layer.append(d)
    return per_layer


# ---- Adapter synthesis ---------------------------------------------------

def synthesize_adapters(per_layer_dims: list, seed: int = 42):
    """N=5 adapter sets, partitioned-QR Grassmannian-A + random-B.

    per_layer_dims: list[layer_idx] -> {key: (in, out)}.
    Returns:
      frozen_A: dict keyed 'layer_{li}_{key}_domain_{d}' -> (in, r) bf16
      adapter_Bs: list of N dicts keyed 'model.layers.{li}.{key}.lora_b' -> (r, out) bf16
    """
    rng = np.random.default_rng(seed)
    frozen_A: dict = {}
    adapter_Bs = [dict() for _ in range(N_ADAPTERS)]

    for li, dims in enumerate(per_layer_dims):
        for key in TARGET_KEYS:
            in_f, out_f = dims[key]
            # Partitioned QR (F#562): (in_f, N*r) random normal -> Q has
            # orthonormal columns; slice into N blocks of r columns.
            M = rng.standard_normal((in_f, N_ADAPTERS * LORA_RANK)).astype(np.float32)
            Q, _ = np.linalg.qr(M)
            for d in range(N_ADAPTERS):
                A_np = Q[:, d * LORA_RANK:(d + 1) * LORA_RANK].astype(np.float32)
                ak = f"layer_{li}_{key}_domain_{d}"
                frozen_A[ak] = mx.array(A_np).astype(mx.bfloat16)
                B_np = rng.standard_normal((LORA_RANK, out_f)).astype(np.float32) * 0.01
                bk = f"model.layers.{li}.{key}.lora_b"
                adapter_Bs[d][bk] = mx.array(B_np).astype(mx.bfloat16)

    # Materialize
    mx.eval(list(frozen_A.values()))
    for ab in adapter_Bs:
        mx.eval(list(ab.values()))

    return frozen_A, adapter_Bs


# ---- Timing -------------------------------------------------------------

def time_attach_cycle(model, frozen_A, adapter_Bs, n_layers) -> float:
    """One attach+detach cycle. Measures attach wall-clock only."""
    d = time_attach_cycle._counter % N_ADAPTERS
    time_attach_cycle._counter += 1

    t0 = time.perf_counter()
    attach_adapter(model, frozen_A, adapter_Bs[d], d, LORA_ALPHA)
    t1 = time.perf_counter()
    # Clean up before next cycle (not timed)
    detach_adapters(model)
    mx.eval(model.parameters())
    return (t1 - t0) * 1000.0


time_attach_cycle._counter = 0


# ---- Generation with KV cache ------------------------------------------

def _next_input(tok: mx.array) -> mx.array:
    """Shape tok (1,) -> (1, 1) for next decode step."""
    return tok.reshape(1, 1)


def greedy_generate(
    model,
    prompt_ids: mx.array,
    n_tokens: int,
    swap_at_pos: int | None = None,
    frozen_A: dict | None = None,
    adapter_B_0: dict | None = None,
) -> list[int]:
    """Greedy decode with KV cache. Optionally detach+reattach at step=swap_at_pos.

    step=0 is the FIRST generated token (after prefill). swap_at_pos=k means
    the swap happens BEFORE generating token at index k (i.e. after k tokens
    have been generated, we swap, then generate the remainder).
    """
    cache = make_prompt_cache(model)

    # Prefill
    logits = model(prompt_ids[None], cache=cache)
    next_tok = mx.argmax(logits[:, -1, :], axis=-1)  # (1,)
    mx.eval(next_tok, cache)

    tokens = [int(next_tok.item())]

    for step in range(1, n_tokens):
        if swap_at_pos is not None and step == swap_at_pos:
            detach_adapters(model)
            attach_adapter(model, frozen_A, adapter_B_0, 0, LORA_ALPHA)
            mx.eval(model.parameters())

        logits = model(_next_input(next_tok), cache=cache)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_tok, cache)
        tokens.append(int(next_tok.item()))

    return tokens


# ---- Main --------------------------------------------------------------

def main() -> int:
    log("=" * 70)
    log("exp_pierre_adapter_hotswap_latency_impl")
    log(f"SMOKE={IS_SMOKE} BENCH_RUNS={BENCH_RUNS} GEN_TOKENS={GEN_TOKENS}")
    log("=" * 70)

    # Phase 1: load model
    log("\n[Phase 1] Loading model...")
    t0 = time.perf_counter()
    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    log(f"  load time: {time.perf_counter() - t0:.2f}s")
    log_mem("after load")

    n_layers = len(model.layers)
    per_layer_dims = infer_per_layer_dims(model)
    log(f"  n_layers={n_layers}")
    # Log unique dim signatures to reveal per-layer variation
    uniq = {}
    for li, d in enumerate(per_layer_dims):
        sig = tuple(sorted((k, v) for k, v in d.items()))
        uniq.setdefault(sig, []).append(li)
    for sig, lis in uniq.items():
        log(f"  dims_group {dict(sig)}: layers[{lis[0]}..{lis[-1]}] count={len(lis)}")

    # Phase 2: synthesize adapters
    log("\n[Phase 2] Synthesizing N=5 adapters (partitioned-QR A, random-B)...")
    t0 = time.perf_counter()
    frozen_A, adapter_Bs = synthesize_adapters(per_layer_dims, seed=42)
    log(f"  synth time: {time.perf_counter() - t0:.2f}s")
    log(f"  frozen_A keys: {len(frozen_A)}")
    log(f"  adapter_Bs: {N_ADAPTERS} sets, {len(adapter_Bs[0])} keys each")
    mx.clear_cache()
    log_mem("after synth")

    # Phase 3: warmup
    log("\n[Phase 3] Warmup...")
    for _ in range(WARMUP_RUNS):
        time_attach_cycle(model, frozen_A, adapter_Bs, n_layers)
    detach_adapters(model)
    mx.eval(model.parameters())
    log_mem("after warmup")

    # Phase 4: K1953 benchmark
    log(f"\n[Phase 4] K1953 benchmark ({BENCH_RUNS} runs)...")
    attach_times_ms = []
    for i in range(BENCH_RUNS):
        t_ms = time_attach_cycle(model, frozen_A, adapter_Bs, n_layers)
        attach_times_ms.append(t_ms)

    t_median = median(attach_times_ms)
    t_mean = mean(attach_times_ms)
    t_std = stdev(attach_times_ms) if len(attach_times_ms) > 1 else 0.0
    t_min = min(attach_times_ms)
    t_max = max(attach_times_ms)
    log(f"  attach_times median={t_median:.3f}ms mean={t_mean:.3f}ms "
        f"std={t_std:.3f}ms min={t_min:.3f}ms max={t_max:.3f}ms")

    k1953_pass = t_median <= K1953_THRESHOLD_MS
    log(f"  K1953: {'PASS' if k1953_pass else 'FAIL'} "
        f"(median {t_median:.3f} ms vs threshold {K1953_THRESHOLD_MS} ms)")

    # Phase 5: K1954 determinism
    log(f"\n[Phase 5] K1954 same-adapter detach/reattach determinism "
        f"at positions {SWAP_POSITIONS}...")
    prompt_ids_list = tokenizer.encode(PROMPT)[:64]
    prompt_ids = mx.array(prompt_ids_list, dtype=mx.int32)
    log(f"  prompt tokens: {len(prompt_ids_list)}")

    # Ensure a clean attach of adapter_B[0]
    detach_adapters(model)
    mx.eval(model.parameters())
    attach_adapter(model, frozen_A, adapter_Bs[0], 0, LORA_ALPHA)
    mx.eval(model.parameters())

    # Baseline T_0
    t0 = time.perf_counter()
    T0 = greedy_generate(model, prompt_ids, GEN_TOKENS, swap_at_pos=None)
    log(f"  T_0 ({time.perf_counter() - t0:.2f}s): {T0}")

    # Decoded preview
    try:
        log(f"  T_0 text: {tokenizer.decode(T0)!r}")
    except Exception as e:  # noqa: BLE001
        log(f"  T_0 decode skipped: {e}")

    T_swap = {}
    glitch_per_k = {}
    total_glitches = 0

    for k in SWAP_POSITIONS:
        if k >= GEN_TOKENS:
            log(f"  skip k={k} (>= GEN_TOKENS={GEN_TOKENS})")
            continue

        # Reset: detach + reattach same adapter to start fresh
        detach_adapters(model)
        mx.eval(model.parameters())
        attach_adapter(model, frozen_A, adapter_Bs[0], 0, LORA_ALPHA)
        mx.eval(model.parameters())

        t0 = time.perf_counter()
        Tk = greedy_generate(
            model, prompt_ids, GEN_TOKENS,
            swap_at_pos=k, frozen_A=frozen_A, adapter_B_0=adapter_Bs[0],
        )
        elapsed = time.perf_counter() - t0
        T_swap[str(k)] = Tk
        diffs = sum(1 for i in range(GEN_TOKENS) if T0[i] != Tk[i])
        glitch_per_k[str(k)] = diffs
        total_glitches += diffs
        log(f"  k={k}: T_swap={Tk} diffs={diffs} ({elapsed:.2f}s)")

    k1954_pass = total_glitches <= K1954_THRESHOLD_GLITCHES
    log(f"  K1954: {'PASS' if k1954_pass else 'FAIL'} "
        f"(total_glitches={total_glitches} vs threshold>{K1954_THRESHOLD_GLITCHES})")

    # Cleanup
    detach_adapters(model)
    mx.eval(model.parameters())

    # Phase 6: results
    log("\n[Phase 6] Writing results...")
    all_pass = bool(k1953_pass and k1954_pass)
    verdict = "SUPPORTED" if all_pass else "KILLED"

    results = {
        "experiment_id": "exp_pierre_adapter_hotswap_latency_impl",
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": IS_SMOKE,
        "config": {
            "model": MODEL_ID,
            "n_layers": n_layers,
            "target_keys": TARGET_KEYS,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "n_adapters": N_ADAPTERS,
            "bench_runs": BENCH_RUNS,
            "warmup_runs": WARMUP_RUNS,
            "gen_tokens": GEN_TOKENS,
            "swap_positions": SWAP_POSITIONS,
            "prompt": PROMPT,
        },
        "per_layer_dims": [
            {k: {"in": v[0], "out": v[1]} for k, v in d.items()}
            for d in per_layer_dims
        ],
        "K1953": {
            "text": "t_attach_median over 20 runs > 100ms",
            "threshold_ms": K1953_THRESHOLD_MS,
            "measured": {
                "median_ms": t_median,
                "mean_ms": t_mean,
                "std_ms": t_std,
                "min_ms": t_min,
                "max_ms": t_max,
                "runs_ms": attach_times_ms,
            },
            "pass": bool(k1953_pass),
        },
        "K1954": {
            "text": "same-adapter detach/re-attach glitch-count > 1",
            "threshold_glitches": K1954_THRESHOLD_GLITCHES,
            "measured": {
                "T_0": T0,
                "T_swap": T_swap,
                "glitches_per_k": glitch_per_k,
                "total_glitches": total_glitches,
            },
            "pass": bool(k1954_pass),
        },
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n[DONE] {verdict}: K1953={'PASS' if k1953_pass else 'FAIL'} "
        f"K1954={'PASS' if k1954_pass else 'FAIL'}")
    log(f"  Results: {RESULTS_FILE}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
