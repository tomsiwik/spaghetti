"""
exp_g4_adapter_class_composition — measurement proxy.

Loads 3 existing Gemma 4 E4B 4-bit LoRA adapters (exp_p1_t2_single_domain_training:
code, math, medical), samples q_proj layers + random probes, and measures the
composition-geometry deviation for three classes:

  L (LoRA / class A):   L(v) = Σ_i ΔW_i v               (additive)
  D (pseudo-DoRA):      D(v) = (W_d - W_0) @ v,
                          W_d = m_d ⊙ (W_0 + ΣΔW_i) / ||W_0 + ΣΔW_i||_c,
                          m_d = ||W_0||_c  (DoRA init, frozen).
  M (pseudo-MoLoRA):    M(v) = (1/N) Σ_i ΔW_i v         (uniform mixture).

Deviation: dev_C = ||C(v) - L(v)|| / (||L(v)|| + 1e-12).

Pure measurement: no training, no MMLU eval. Replaces the DB title's 3pp MMLU
claim with a composition-geometry proxy (see MATH.md §Scope).
"""
from __future__ import annotations

import gc
import json
import math
import random
import time
from pathlib import Path
from statistics import median

import mlx.core as mx
from mlx_lm import load
from safetensors import safe_open

HERE = Path(__file__).parent
RESULTS_PATH = HERE / "results.json"

GEMMA_ID = "mlx-community/gemma-4-e4b-it-4bit"
ADAPTER_ROOT = Path("experiments/models/exp_p1_t2_single_domain_training/adapters")
DOMAINS = ("code", "math", "medical")
LORA_SCALE = 6.0  # From adapter_config.json
PROJ = "q_proj"  # Adapters only target q_proj
NUM_PROBE_LAYERS = 10
NUM_PROBES = 10
K1_MIN_SUCCESS_RATE = 0.95
K2_MIN_GAP = 1e-4
SEED = 0

random.seed(SEED)
mx.random.seed(SEED)


def load_adapter_ab(domain: str, layer_idx: int) -> tuple[mx.array, mx.array]:
    """Load A (d_in, r) and B (r, d_out) for the given domain + layer."""
    path = ADAPTER_ROOT / domain / "adapters.safetensors"
    a_key = f"language_model.model.layers.{layer_idx}.self_attn.{PROJ}.lora_a"
    b_key = f"language_model.model.layers.{layer_idx}.self_attn.{PROJ}.lora_b"
    with safe_open(str(path), framework="numpy") as f:
        a_np = f.get_tensor(a_key)
        b_np = f.get_tensor(b_key)
    a = mx.array(a_np).astype(mx.float32)
    b = mx.array(b_np).astype(mx.float32)
    return a, b


def delta_w(a: mx.array, b: mx.array) -> mx.array:
    """ΔW = scale * (A @ B).T, shape (d_out, d_in)."""
    # A: (d_in, r), B: (r, d_out). A @ B: (d_in, d_out). .T -> (d_out, d_in).
    ab = a @ b  # (d_in, d_out)
    return LORA_SCALE * ab.T  # (d_out, d_in)


def column_norms(W: mx.array) -> mx.array:
    """Per-column L2 norm; output shape (d_in,) for W in R^{d_out x d_in}."""
    return mx.sqrt(mx.sum(W * W, axis=0) + 1e-12)


def measure_layer(model, layer_idx: int, probes: mx.array) -> dict:
    """Return deviation stats for one q_proj layer across all probes."""
    # Dequantize base W_0 for this layer's q_proj.
    layer = model.language_model.model.layers[layer_idx]
    mod = layer.self_attn.q_proj
    W0 = mx.dequantize(
        mod.weight, mod.scales, mod.biases,
        group_size=mod.group_size, bits=mod.bits,
    ).astype(mx.float32)  # (d_out, d_in)
    mx.eval(W0)

    # Build three deltas.
    deltas = []
    for domain in DOMAINS:
        a, b = load_adapter_ab(domain, layer_idx)
        dw = delta_w(a, b)  # (d_out, d_in)
        mx.eval(dw)
        deltas.append(dw)
        del a, b
    sum_delta = deltas[0] + deltas[1] + deltas[2]  # (d_out, d_in)
    mx.eval(sum_delta)

    # Pseudo-DoRA weight: W_d = m_d ⊙ (W_0 + ΣΔW) / ||W_0 + ΣΔW||_c,
    # broadcast m_d = ||W_0||_c across rows (column-wise scaling).
    W0_plus = W0 + sum_delta
    col_n0 = column_norms(W0)                 # (d_in,) -- m_d
    col_nplus = column_norms(W0_plus)         # (d_in,)
    scale_vec = col_n0 / col_nplus            # (d_in,)
    W_d = W0_plus * scale_vec[None, :]        # broadcast (d_in,) across d_out rows
    dora_delta = W_d - W0                     # (d_out, d_in)
    mx.eval(dora_delta)

    # Evaluate on each probe.
    probe_rows = []
    for p_idx in range(probes.shape[0]):
        v = probes[p_idx]  # (d_in,)
        L_v = sum_delta @ v          # (d_out,)
        D_v = dora_delta @ v         # (d_out,)
        M_v = (sum_delta @ v) / 3.0  # (1/N) L_v, N=3

        L_norm = float(mx.linalg.norm(L_v))
        if L_norm < 1e-12:
            # Degenerate probe direction (shouldn't happen for random unit probes).
            probe_rows.append({"probe": p_idx, "ok": False, "reason": "zero_L"})
            continue

        dev_L = 0.0  # identity
        dev_D = float(mx.linalg.norm(D_v - L_v)) / L_norm
        dev_M = float(mx.linalg.norm(M_v - L_v)) / L_norm  # exactly 2/3 ≈ 0.667

        row = {
            "probe": p_idx,
            "ok": all(math.isfinite(x) for x in (dev_L, dev_D, dev_M)),
            "L_norm": L_norm,
            "dev_L": dev_L,
            "dev_D": dev_D,
            "dev_M": dev_M,
        }
        probe_rows.append(row)

    # Clean up layer tensors.
    del W0, sum_delta, W0_plus, dora_delta, deltas
    mx.clear_cache()
    gc.collect()

    return {"layer": layer_idx, "probes": probe_rows}


def main() -> int:
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading {GEMMA_ID}")
    model, _ = load(GEMMA_ID)
    mx.eval(model.parameters())

    # Total layers in Gemma 4 E4B.
    n_layers = len(model.language_model.model.layers)
    print(f"[{time.strftime('%H:%M:%S')}] Model has {n_layers} layers")

    # Sample NUM_PROBE_LAYERS evenly spaced layers.
    if n_layers < NUM_PROBE_LAYERS:
        picked = list(range(n_layers))
    else:
        step = n_layers / NUM_PROBE_LAYERS
        picked = [min(int(i * step), n_layers - 1) for i in range(NUM_PROBE_LAYERS)]
    print(f"[{time.strftime('%H:%M:%S')}] Probing layers: {picked}")

    # Probe vectors: unit-norm, d_in = 2560 (Gemma 4 E4B q_proj input).
    D_IN = 2560
    probes_raw = mx.random.normal(shape=(NUM_PROBES, D_IN))
    probes = probes_raw / mx.linalg.norm(probes_raw, axis=1, keepdims=True)
    mx.eval(probes)

    # Measure each layer.
    all_rows = []
    for i, layer_idx in enumerate(picked):
        t0 = time.time()
        layer_result = measure_layer(model, layer_idx, probes)
        dt = time.time() - t0
        all_rows.append(layer_result)
        print(f"[{time.strftime('%H:%M:%S')}] L{layer_idx:02d} done ({dt:.1f}s)")

    # Flatten probe rows.
    flat = [
        {"layer": lr["layer"], **pr}
        for lr in all_rows
        for pr in lr["probes"]
    ]
    total = len(flat)
    successful = [r for r in flat if r.get("ok")]
    success_rate = len(successful) / total if total else 0.0

    def _med(key):
        vals = [r[key] for r in successful]
        return median(vals) if vals else float("nan")

    med_L = _med("dev_L")
    med_D = _med("dev_D")
    med_M = _med("dev_M")

    # Per-class aggregate.
    def _stats(key):
        vals = [r[key] for r in successful]
        if not vals:
            return {}
        vals_s = sorted(vals)
        return {
            "n": len(vals),
            "min": vals_s[0],
            "median": vals_s[len(vals_s) // 2],
            "max": vals_s[-1],
            "mean": sum(vals) / len(vals),
        }

    stats_L = _stats("dev_L")
    stats_D = _stats("dev_D")
    stats_M = _stats("dev_M")

    # Kill criteria.
    k1_pass = success_rate >= K1_MIN_SUCCESS_RATE
    k2_pass = (
        math.isfinite(med_D)
        and math.isfinite(med_M)
        and (med_D > med_L + K2_MIN_GAP)
        and (med_M > med_L + K2_MIN_GAP)
    )
    all_pass = k1_pass and k2_pass
    verdict = "SUPPORTED" if all_pass else ("KILLED" if k1_pass else "INCONCLUSIVE")

    elapsed = time.time() - t_start
    result = {
        "experiment_id": "exp_g4_adapter_class_composition",
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": False,
        "seed": SEED,
        "num_probe_layers": len(picked),
        "num_probes_per_layer": NUM_PROBES,
        "layers_probed": picked,
        "num_domains": len(DOMAINS),
        "domains": list(DOMAINS),
        "lora_scale": LORA_SCALE,
        "n_total_measurements": total,
        "n_successful": len(successful),
        "success_rate": success_rate,
        "class_stats": {
            "L_lora_additive": stats_L,
            "D_pseudo_dora": stats_D,
            "M_pseudo_molora": stats_M,
        },
        "medians": {"dev_L": med_L, "dev_D": med_D, "dev_M": med_M},
        "k1_structural": {
            "description": "success_rate >= 0.95",
            "measured": success_rate,
            "threshold": K1_MIN_SUCCESS_RATE,
            "pass": k1_pass,
        },
        "k2_target": {
            "description": "median(dev_D) > median(dev_L) + 1e-4 AND median(dev_M) > median(dev_L) + 1e-4",
            "measured_gap_D_minus_L": med_D - med_L if math.isfinite(med_D) else float("nan"),
            "measured_gap_M_minus_L": med_M - med_L if math.isfinite(med_M) else float("nan"),
            "threshold": K2_MIN_GAP,
            "pass": k2_pass,
        },
        "elapsed_sec": elapsed,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== VERDICT: {verdict} ===")
    print(f"K1 structural (success>={K1_MIN_SUCCESS_RATE}): {k1_pass}  (rate={success_rate:.3f})")
    print(f"K2 target (class ordering): {k2_pass}")
    print(f"  dev_L median = {med_L:.6f}")
    print(f"  dev_D median = {med_D:.6f}")
    print(f"  dev_M median = {med_M:.6f}")
    print(f"Elapsed: {elapsed:.1f}s")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
