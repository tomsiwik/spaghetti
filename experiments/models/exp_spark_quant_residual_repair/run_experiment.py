#!/usr/bin/env python3
"""exp_spark_quant_residual_repair — does the q_proj LoRA delta align with the 4-bit quant residual?

Pure weight-space test (no decode, no training). Frozen mlx-community/gemma-4-e4b-it-4bit is the base the
r=6 q_proj adapters (math, medical, python) were trained on. We ask whether the fp16 LoRA delta B@A literally
repairs the projection's OWN 4-bit quantization residual.

Reference: true fp16 google/gemma-4-e4b-it is a 16GB gated download not present locally, so we use the local
8-bit snapshot mlx-community/gemma-4-e4b-it-8bit as a high-fidelity fp proxy: 8-bit step is 16x finer than
4-bit, so dequant(W_int8) recovers fp16 to within ~6% of the 4-bit residual norm (<< the 2x null margin).
  R_l = dequant(W_int8_l) - dequant(W_int4_l)    # the 4-bit quant residual (real, both on disk)

Per layer l, the LoRA weight delta on W (out,in) is  dW_l = (A_l @ B_l).T  (forward x@A@B => weight (A@B).T).
We measure  cosbar = mean_l cos(flatten(dW_l), flatten(R_l))  for q_proj across all 42 layers, and compare
against 2x the 95th-percentile of a norm-matched random-rank6 null (200 draws/layer, pooled, seed=42).

K2301: KILL if cosbar_real(math) < 2 * p95_null. SUPPORTED if cosbar_real(math) >= 2 * p95_null.

NO MOCKS. Real 4-bit + 8-bit Gemma-4 loaded, all 42 q_proj layers dequantized. is_smoke=False. mlx-lm 0.31.2.
"""

import gc
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

EXP_DIR = Path(__file__).resolve().parent
RESULTS_FILE = EXP_DIR / "results.json"

MODEL_4BIT = "mlx-community/gemma-4-e4b-it-4bit"
MODEL_8BIT = "mlx-community/gemma-4-e4b-it-8bit"
ADAPTER_ROOT = Path("/Users/tom/Code/tomsiwik/llm/data/adapters")
ADAPTERS = {
    "math": ADAPTER_ROOT / "math" / "adapters.safetensors",
    "medical": ADAPTER_ROOT / "medical" / "adapters.safetensors",
    "python": ADAPTER_ROOT / "python" / "adapters.safetensors",
}
PRIMARY = "math"  # the adapter named in the spec; verdict is decided on this one

N_LAYERS_EXPECTED = 42
N_NULL = 200          # random rank-6 draws per layer
NULL_MARGIN = 2.0     # real must clear 2x the null p95
SEED = 42


def log(m):
    print(m, flush=True)


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def dequant_qproj(model):
    """Return {layer_idx: dequantized q_proj weight (out,in) float32} for every layer."""
    lm = get_lm(model)
    out = {}
    for li, layer in enumerate(lm.model.layers):
        q = layer.self_attn.q_proj
        # Must be a QuantizedLinear with its own bits/group_size — dequant with the module's params.
        assert hasattr(q, "scales") and hasattr(q, "biases"), f"layer {li} q_proj not quantized"
        W = mx.dequantize(q.weight, q.scales, q.biases,
                          group_size=q.group_size, bits=q.bits).astype(mx.float32)
        out[li] = W
    return out


def lora_delta(ad, li):
    """Weight-space delta dW = (A @ B).T with shape (out, in), float32."""
    ak = f"language_model.model.layers.{li}.self_attn.q_proj.lora_a"
    bk = f"language_model.model.layers.{li}.self_attn.q_proj.lora_b"
    if ak not in ad or bk not in ad:
        return None
    A = ad[ak].astype(mx.float32)   # (in, r)
    B = ad[bk].astype(mx.float32)   # (r, out)
    return (A @ B).T                # (out, in)


def cos_flat(X, Y):
    x = X.reshape(-1)
    y = Y.reshape(-1)
    nx = mx.linalg.norm(x)
    ny = mx.linalg.norm(y)
    if float(nx) < 1e-12 or float(ny) < 1e-12:
        return 0.0
    return float((x @ y) / (nx * ny))


def null_p95_for_layer(R, target_norm, r, n_draws, rng_key):
    """|cos| of n_draws norm-matched random rank-6 matrices vs flatten(R). Returns the |cos| array."""
    out_d, in_d = R.shape
    rflat = R.reshape(-1)
    rnorm = mx.linalg.norm(rflat)
    cosines = []
    for i in range(n_draws):
        k1, k2, rng_key = mx.random.split(rng_key, 3)
        A = mx.random.normal((out_d, r), key=k1)
        B = mx.random.normal((r, in_d), key=k2)
        D = A @ B                                  # rank-r
        dflat = D.reshape(-1)
        dnorm = mx.linalg.norm(dflat)
        if float(dnorm) < 1e-12:
            continue
        # norm-matched: cosine is scale-invariant, but we keep the contract explicit.
        c = abs(float((dflat @ rflat) / (dnorm * rnorm)))
        cosines.append(c)
    return cosines


def percentile(vals, p):
    s = sorted(vals)
    if not s:
        return float("nan")
    idx = (len(s) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log("=" * 72)
    log("exp_spark_quant_residual_repair")
    log(f"4bit base: {MODEL_4BIT}")
    log(f"8bit ref : {MODEL_8BIT}")
    log("=" * 72)

    for name, p in ADAPTERS.items():
        assert p.exists(), f"missing adapter {name}: {p}"

    log("\n=== Load 4-bit base + dequant q_proj ===")
    m4, _ = load(MODEL_4BIT)
    W4 = dequant_qproj(m4)
    assert len(W4) == N_LAYERS_EXPECTED, f"expected {N_LAYERS_EXPECTED} layers, got {len(W4)}"
    del m4
    gc.collect(); mx.clear_cache()

    log("=== Load 8-bit ref + dequant q_proj ===")
    m8, _ = load(MODEL_8BIT)
    W8 = dequant_qproj(m8)
    assert len(W8) == N_LAYERS_EXPECTED
    del m8
    gc.collect(); mx.clear_cache()

    # 4-bit quantization residual reference, per layer.
    log("=== Build 4-bit quant residual R = dequant(W8) - dequant(W4) ===")
    R = {}
    res_rel = []
    for li in range(N_LAYERS_EXPECTED):
        Rli = (W8[li] - W4[li]).astype(mx.float32)
        mx.eval(Rli)
        R[li] = Rli
        rn = float(mx.linalg.norm(Rli.reshape(-1)))
        wn = float(mx.linalg.norm(W8[li].reshape(-1)))
        res_rel.append(rn / wn if wn > 0 else 0.0)
    log(f"  mean rel residual ||R||/||W8|| = {sum(res_rel)/len(res_rel):.4f}")
    del W8
    gc.collect()

    rng = mx.random.key(SEED)

    per_adapter = {}
    for name, path in ADAPTERS.items():
        log(f"\n=== Adapter: {name} ===")
        ad = mx.load(str(path))
        real_cos = []
        null_pool = []
        delta_norms = []
        for li in range(N_LAYERS_EXPECTED):
            dW = lora_delta(ad, li)
            if dW is None:
                log(f"  layer {li}: no q_proj lora — SKIP")
                continue
            dW = dW.astype(mx.float32)
            mx.eval(dW)
            c = cos_flat(dW, R[li])
            real_cos.append(c)
            delta_norms.append(float(mx.linalg.norm(dW.reshape(-1))))
            k_layer, rng = mx.random.split(rng, 2)
            target = float(mx.linalg.norm(dW.reshape(-1)))
            null_cos = null_p95_for_layer(R[li], target, r=6, n_draws=N_NULL, rng_key=k_layer)
            null_pool.extend(null_cos)
        n_layers = len(real_cos)
        cosbar = sum(real_cos) / n_layers
        abs_cosbar = sum(abs(c) for c in real_cos) / n_layers
        p95 = percentile(null_pool, 95.0)
        p50 = percentile(null_pool, 50.0)
        threshold = NULL_MARGIN * p95
        supported = cosbar >= threshold
        ratio = cosbar / p95 if p95 > 1e-12 else float("inf")
        log(f"  n_layers={n_layers}  cosbar={cosbar:.5f}  |cosbar|={abs_cosbar:.5f}")
        log(f"  null p50={p50:.5f}  p95={p95:.5f}  2*p95={threshold:.5f}  ratio cosbar/p95={ratio:.2f}")
        log(f"  -> {'SUPPORTED' if supported else 'KILLED'} (margin {NULL_MARGIN}x)")
        per_adapter[name] = {
            "n_layers": n_layers,
            "cosbar_real": cosbar,
            "abs_cosbar_real": abs_cosbar,
            "per_layer_cos": real_cos,
            "null_p50": p50,
            "null_p95": p95,
            "threshold_2x_p95": threshold,
            "ratio_cosbar_over_p95": ratio,
            "supported": bool(supported),
            "mean_delta_fro": sum(delta_norms) / len(delta_norms),
            "n_null_pooled": len(null_pool),
        }
        del ad
        gc.collect()

    prim = per_adapter[PRIMARY]
    killed = not prim["supported"]
    verdict = "killed" if killed else "supported"
    all_pass = not killed

    results = {
        "experiment_id": "exp_spark_quant_residual_repair",
        "config": {
            "model_4bit": MODEL_4BIT,
            "model_8bit_ref": MODEL_8BIT,
            "reference_note": "8-bit dequant used as fp16 proxy (16x finer step, ~6% residual error, < 2x null margin)",
            "adapters": {k: str(v) for k, v in ADAPTERS.items()},
            "primary_adapter": PRIMARY,
            "n_layers": N_LAYERS_EXPECTED,
            "n_null_per_layer": N_NULL,
            "null_margin": NULL_MARGIN,
            "lora_rank": 6,
            "seed": SEED,
            "mlx_lm": "0.31.2",
        },
        "per_adapter": per_adapter,
        "kill_criteria": {
            "2301": {
                "text": "mean per-layer cos(B@A, W_fp16 - dequant(W_4bit)) for q_proj does NOT exceed 2x the 95th-pct of a norm-matched random-rank6 null",
                "primary_adapter": PRIMARY,
                "cosbar_real": prim["cosbar_real"],
                "threshold_2x_p95": prim["threshold_2x_p95"],
                "ratio_cosbar_over_p95": prim["ratio_cosbar_over_p95"],
                "result": "fail" if killed else "pass",
            }
        },
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": False,
        "total_wall_clock_sec": time.time() - t0,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    log("\n" + "=" * 72)
    for name, d in per_adapter.items():
        tag = "PRIMARY" if name == PRIMARY else ""
        log(f"  {name:8s} cosbar={d['cosbar_real']:+.5f}  2*p95={d['threshold_2x_p95']:.5f}  "
            f"ratio={d['ratio_cosbar_over_p95']:.2f}  {'SUPPORTED' if d['supported'] else 'KILLED'} {tag}")
    log(f"VERDICT ({PRIMARY}): {verdict}  all_pass={all_pass}")
    log(f"Wrote {RESULTS_FILE}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
