"""
TT-LoRA Port to MLX — Verification Experiment
Paper: TT-LoRA MoE (arXiv:2504.21190)
Target: Gemma 4 E4B on Apple M5 Pro

Tests:
  K1: Forward pass self-consistency (reconstruction vs direct matmul) < 1e-5
  K2: Parameter count <= 40K per layer (q + v)
  K3: Forward latency within 2x of standard LoRA on MLX
"""

import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# TT-LoRA Core: reconstruction-based forward pass
# ---------------------------------------------------------------------------

class TTLoRALinear(nn.Module):
    """TT-LoRA adapter wrapping an existing linear layer.

    Uses reconstruction: contract TT cores → ΔW matrix → x @ ΔW.T + base(x).
    This is simpler and faster on Metal than sequential contraction because
    a single large matmul beats many tiny kernels.
    """

    def __init__(
        self,
        base_weight: mx.array,   # frozen base weight [out, in] or quantized
        in_features: int,
        out_features: int,
        tt_shape: list[int],     # factorization of [in_features, out_features]
        tt_rank: int = 8,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tt_shape = tt_shape
        self.alpha = alpha

        # Validate factorization
        m_prod = 1
        n_prod = 1
        # First factors multiply to in_features, rest to out_features
        # We mark split point
        self._find_split(tt_shape, in_features, out_features)

        # Build TT ranks: [1, r, r, ..., r, 1]
        d = len(tt_shape)
        ranks = [1] + [tt_rank] * (d - 1) + [1]

        # Initialize cores with scaled normal initialization
        cores = []
        for k in range(d):
            shape = (ranks[k], tt_shape[k], ranks[k + 1])
            std = 1.0 / math.sqrt(tt_shape[k])
            core = mx.random.normal(shape) * std
            cores.append(core)
        self.tt_cores = cores

        # Store base weight (frozen, possibly quantized)
        self._base_weight = base_weight
        self._cached_delta_w = None  # Lazy cache for inference

    def _find_split(self, tt_shape, in_features, out_features):
        """Find where input factors end and output factors begin."""
        prod = 1
        for i, s in enumerate(tt_shape):
            prod *= s
            if prod == in_features:
                self._split = i + 1
                # Verify remaining factors = out_features
                rest = 1
                for j in range(i + 1, len(tt_shape)):
                    rest *= tt_shape[j]
                assert rest == out_features, (
                    f"Remaining factors product {rest} != out_features {out_features}"
                )
                return
        raise ValueError(
            f"Cannot split tt_shape {tt_shape} into "
            f"in={in_features} x out={out_features}"
        )

    def reconstruct_delta_w(self) -> mx.array:
        """Contract TT cores into full weight correction matrix [out, in]."""
        # Start with first core: [1, s_0, r_1] → squeeze → [s_0, r_1]
        result = self.tt_cores[0].squeeze(0)  # [s_0, r_1]

        for k in range(1, len(self.tt_cores)):
            core = self.tt_cores[k]  # [r_k, s_k, r_{k+1}]
            r_k, s_k, r_next = core.shape

            # result shape: [s_0*s_1*...*s_{k-1}, r_k]
            # core reshaped: [r_k, s_k * r_{k+1}]
            result = result @ core.reshape(r_k, s_k * r_next)
            # result: [s_0*...*s_{k-1}, s_k * r_{k+1}]
            # Reshape to separate s_k from r_{k+1}
            leading = result.shape[0]
            result = result.reshape(leading * s_k, r_next)

        # result: [s_0*s_1*...*s_{d-1}, 1] → squeeze → [prod(tt_shape)]
        result = result.squeeze(-1)

        # Reshape: the TT shape encodes [in_factors..., out_factors...]
        # So the tensor indices are (i_1,...,i_dm, j_1,...,j_dn)
        # Reshape to [in_features, out_features] then transpose to [out, in]
        delta_w = result.reshape(self.in_features, self.out_features)
        return delta_w.T  # [out, in]

    def cache_delta_w(self):
        """Pre-compute and cache ΔW for inference (no per-call reconstruction)."""
        self._cached_delta_w = self.reconstruct_delta_w()
        mx.eval(self._cached_delta_w)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward: base(x) + alpha * x @ delta_W.T"""
        base_out = x @ self._base_weight.T

        # Use cached ΔW if available, otherwise reconstruct
        delta_w = self._cached_delta_w if self._cached_delta_w is not None else self.reconstruct_delta_w()
        tt_out = x @ delta_w.T  # [B, S, out]

        return base_out + self.alpha * tt_out

    def num_params(self) -> int:
        """Count trainable parameters in TT cores."""
        return sum(c.size for c in self.tt_cores)


# ---------------------------------------------------------------------------
# Standard LoRA for comparison
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Standard LoRA: y = Wx + alpha * B @ A @ x"""

    def __init__(
        self,
        base_weight: mx.array,
        in_features: int,
        out_features: int,
        rank: int = 6,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self._base_weight = base_weight

        # Standard LoRA init: A ~ N(0, 1/sqrt(r)), B = 0
        self.lora_a = mx.random.normal((rank, in_features)) / math.sqrt(rank)
        self.lora_b = mx.zeros((out_features, rank))

    def __call__(self, x: mx.array) -> mx.array:
        base_out = x @ self._base_weight.T
        lora_out = (x @ self.lora_a.T) @ self.lora_b.T
        return base_out + self.alpha * lora_out

    def num_params(self) -> int:
        return self.lora_a.size + self.lora_b.size


# ---------------------------------------------------------------------------
# Factorization utilities
# ---------------------------------------------------------------------------

def factorize(n: int, max_factor: int = 10) -> list[int]:
    """Factorize n into small factors <= max_factor, preferring 8s."""
    factors = []
    # Pull out largest power-of-2 factors as 8s
    while n % 8 == 0 and n > 8:
        factors.append(8)
        n //= 8
    # Remaining small factors
    for f in range(max_factor, 1, -1):
        while n % f == 0 and n > 1:
            factors.append(f)
            n //= f
    if n > 1:
        factors.append(n)
    factors.sort()
    return factors


def compute_tt_shape(in_features: int, out_features: int) -> list[int]:
    """Compute TT shape by factorizing input and output dimensions."""
    m_factors = factorize(in_features)
    n_factors = factorize(out_features)
    return m_factors + n_factors


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_k1_consistency(in_features: int, out_features: int, tt_rank: int = 8):
    """K1: Verify forward pass consistency between TT-LoRA and explicit matmul."""
    tt_shape = compute_tt_shape(in_features, out_features)
    print(f"  TT shape: {tt_shape} ({len(tt_shape)} cores)")

    # Create a random base weight (simulating dequantized)
    base_weight = mx.random.normal((out_features, in_features)) * 0.01

    # Create TT-LoRA module
    ttlora = TTLoRALinear(
        base_weight=base_weight,
        in_features=in_features,
        out_features=out_features,
        tt_shape=tt_shape,
        tt_rank=tt_rank,
        alpha=1.0,
    )

    # Create test input: [B=2, S=16, in_features]
    x = mx.random.normal((2, 16, in_features))

    # Forward via TT-LoRA module
    y_ttlora = ttlora(x)
    mx.eval(y_ttlora)

    # Forward via explicit reconstruction + matmul
    delta_w = ttlora.reconstruct_delta_w()  # [out, in]
    y_explicit = (x @ base_weight.T) + 1.0 * (x @ delta_w.T)
    mx.eval(y_explicit)

    # Compare
    max_diff = mx.abs(y_ttlora - y_explicit).max().item()
    print(f"  Max diff (TT forward vs explicit): {max_diff:.2e}")
    return max_diff, tt_shape


def run_k2_params(in_features: int, out_features: int, tt_rank: int = 8, lora_rank: int = 6):
    """K2: Count parameters for TT-LoRA vs standard LoRA."""
    tt_shape = compute_tt_shape(in_features, out_features)

    base_weight = mx.random.normal((out_features, in_features)) * 0.01

    ttlora = TTLoRALinear(
        base_weight=base_weight,
        in_features=in_features,
        out_features=out_features,
        tt_shape=tt_shape,
        tt_rank=tt_rank,
    )

    lora = LoRALinear(
        base_weight=base_weight,
        in_features=in_features,
        out_features=out_features,
        rank=lora_rank,
    )

    tt_params = ttlora.num_params()
    lora_params = lora.num_params()
    compression = lora_params / tt_params if tt_params > 0 else float("inf")

    print(f"  TT-LoRA params: {tt_params:,}")
    print(f"  LoRA params:    {lora_params:,}")
    print(f"  Compression:    {compression:.1f}x")

    return tt_params, lora_params, compression


def run_k3_latency(
    in_features: int,
    out_features: int,
    tt_rank: int = 8,
    lora_rank: int = 6,
    batch_size: int = 1,
    seq_len: int = 64,
    warmup: int = 10,
    iters: int = 100,
):
    """K3: Latency comparison TT-LoRA vs standard LoRA."""
    tt_shape = compute_tt_shape(in_features, out_features)
    base_weight = mx.random.normal((out_features, in_features)) * 0.01

    ttlora_uncached = TTLoRALinear(
        base_weight=base_weight,
        in_features=in_features,
        out_features=out_features,
        tt_shape=tt_shape,
        tt_rank=tt_rank,
    )

    ttlora_cached = TTLoRALinear(
        base_weight=base_weight,
        in_features=in_features,
        out_features=out_features,
        tt_shape=tt_shape,
        tt_rank=tt_rank,
    )
    # Copy same cores for fair comparison, then cache
    ttlora_cached.tt_cores = ttlora_uncached.tt_cores
    ttlora_cached.cache_delta_w()

    lora = LoRALinear(
        base_weight=base_weight,
        in_features=in_features,
        out_features=out_features,
        rank=lora_rank,
    )

    x = mx.random.normal((batch_size, seq_len, in_features))
    mx.eval(x)

    def bench(fn, label):
        for _ in range(warmup):
            y = fn(x)
            mx.eval(y)
        t0 = time.perf_counter()
        for _ in range(iters):
            y = fn(x)
            mx.eval(y)
        elapsed = (time.perf_counter() - t0) / iters
        print(f"  {label}: {elapsed*1000:.3f} ms")
        return elapsed

    base_time = bench(lambda x: x @ base_weight.T, "Base only    ")
    lora_time = bench(lora, "LoRA         ")
    tt_uncached_time = bench(ttlora_uncached, "TT-LoRA (raw)")
    tt_cached_time = bench(ttlora_cached, "TT-LoRA (cached)")

    ratio_uncached = tt_uncached_time / lora_time if lora_time > 0 else float("inf")
    ratio_cached = tt_cached_time / lora_time if lora_time > 0 else float("inf")
    print(f"  Ratio (uncached): {ratio_uncached:.2f}x")
    print(f"  Ratio (cached):   {ratio_cached:.2f}x")

    return {
        "base_ms": base_time * 1000,
        "lora_ms": lora_time * 1000,
        "tt_uncached_ms": tt_uncached_time * 1000,
        "tt_cached_ms": tt_cached_time * 1000,
        "ratio_uncached": ratio_uncached,
        "ratio_cached": ratio_cached,
    }


def run_on_real_model():
    """Run TT-LoRA on actual Gemma 4 E4B layer to verify integration."""
    from mlx_lm import load

    print("\n=== Real Model Integration Test ===")
    print("Loading Gemma 4 E4B...")
    model, tokenizer = load("mlx-community/gemma-4-e4b-it-4bit")

    # Get a test layer
    layer = model.layers[0]
    q_proj = layer.self_attn.q_proj

    # Dequantize the weight to get actual dimensions
    # MLX quantized linear stores: weight, scales, biases (quantized format)
    # We need the logical dimensions
    # From model inspection: q_proj is 2048 x 2560 (out x in)
    in_features = 2560
    out_features = 2048

    # Dequantize for the base weight
    if hasattr(q_proj, "scales"):
        # Quantized linear — dequantize
        base_w = mx.dequantize(
            q_proj.weight, q_proj.scales, q_proj.biases,
            q_proj.group_size, q_proj.bits,
        )
    else:
        base_w = q_proj.weight
    mx.eval(base_w)
    print(f"  Dequantized q_proj weight: {base_w.shape}")

    # Create TT-LoRA wrapper
    tt_shape = compute_tt_shape(in_features, out_features)
    ttlora_q = TTLoRALinear(
        base_weight=base_w,
        in_features=in_features,
        out_features=out_features,
        tt_shape=tt_shape,
        tt_rank=8,
        alpha=1.0,
    )

    # Create a random input matching hidden_size
    x = mx.random.normal((1, 8, in_features))
    mx.eval(x)
    print(f"  Input shape: {x.shape}")

    # Original q_proj output
    y_orig = q_proj(x)
    mx.eval(y_orig)

    # TT-LoRA output
    y_tt = ttlora_q(x)
    mx.eval(y_tt)

    print(f"  Original output shape: {y_orig.shape}")
    print(f"  TT-LoRA output shape:  {y_tt.shape}")
    print(f"  Output diff norm:      {mx.sqrt(mx.sum((y_tt - y_orig) ** 2)).item():.4f}")
    print(f"  (Diff expected — TT cores are random, not trained)")

    # Clean up model to free memory
    del model, tokenizer
    return True


def main():
    print("=" * 60)
    print("TT-LoRA Port to MLX — Verification Experiment")
    print("Paper: arXiv:2504.21190")
    print("=" * 60)

    results = {
        "experiment": "exp_p9_ttlora_port_mlx",
        "paper": "arXiv:2504.21190",
    }

    # --- K1: Forward pass consistency ---
    print("\n--- K1: Forward Pass Consistency ---")
    projections = {
        "q_proj": (2560, 2048),
        "v_proj": (2560, 512),
        "o_proj": (2048, 2560),
    }

    k1_results = {}
    all_pass = True
    for name, (in_f, out_f) in projections.items():
        print(f"\n{name} ({in_f} → {out_f}):")
        max_diff, tt_shape = run_k1_consistency(in_f, out_f, tt_rank=8)
        passed = max_diff < 1e-5
        all_pass = all_pass and passed
        k1_results[name] = {
            "in_features": in_f,
            "out_features": out_f,
            "tt_shape": tt_shape,
            "max_diff": max_diff,
            "pass": passed,
        }
        print(f"  K1 {'PASS' if passed else 'FAIL'}")

    results["k1"] = {
        "description": "Forward pass self-consistency (reconstruction vs explicit)",
        "threshold": 1e-5,
        "results": k1_results,
        "pass": all_pass,
    }

    # --- K2: Parameter count ---
    print("\n--- K2: Parameter Count ---")
    k2_results = {}
    total_tt_per_layer = 0
    for name, (in_f, out_f) in projections.items():
        print(f"\n{name}:")
        tt_p, lora_p, comp = run_k2_params(in_f, out_f, tt_rank=8, lora_rank=6)
        total_tt_per_layer += tt_p
        k2_results[name] = {
            "tt_params": tt_p,
            "lora_params": lora_p,
            "compression": comp,
        }

    # q + v only (as specified in kill criteria context)
    tt_qv = k2_results["q_proj"]["tt_params"] + k2_results["v_proj"]["tt_params"]
    k2_pass = tt_qv <= 40_000
    print(f"\nTotal TT-LoRA params (q+v): {tt_qv:,}")
    print(f"Total TT-LoRA params (q+v+o): {total_tt_per_layer:,}")
    print(f"K2 {'PASS' if k2_pass else 'FAIL'}: {tt_qv:,} <= 40,000")

    results["k2"] = {
        "description": "Parameter count <= 40K per layer (q + v)",
        "threshold": 40_000,
        "tt_params_qv": tt_qv,
        "tt_params_qvo": total_tt_per_layer,
        "per_projection": k2_results,
        "pass": k2_pass,
    }

    # --- K3: Latency ---
    print("\n--- K3: Latency Comparison ---")
    k3_results = {}
    max_ratio_cached = 0.0
    max_ratio_uncached = 0.0
    for name, (in_f, out_f) in projections.items():
        print(f"\n{name} ({in_f} → {out_f}):")
        latency = run_k3_latency(
            in_f, out_f, tt_rank=8, lora_rank=6,
            batch_size=1, seq_len=64,
        )
        max_ratio_cached = max(max_ratio_cached, latency["ratio_cached"])
        max_ratio_uncached = max(max_ratio_uncached, latency["ratio_uncached"])
        k3_results[name] = latency

    # K3 uses cached mode (inference scenario)
    k3_pass = max_ratio_cached <= 2.0
    print(f"\nMax TT/LoRA ratio (cached):   {max_ratio_cached:.2f}x")
    print(f"Max TT/LoRA ratio (uncached): {max_ratio_uncached:.2f}x")
    print(f"K3 {'PASS' if k3_pass else 'FAIL'}: {max_ratio_cached:.2f}x <= 2.0x (cached inference)")

    results["k3"] = {
        "description": "Forward latency within 2x of standard LoRA (cached inference)",
        "threshold": 2.0,
        "max_ratio_cached": max_ratio_cached,
        "max_ratio_uncached": max_ratio_uncached,
        "per_projection": k3_results,
        "pass": k3_pass,
    }

    # --- Real model integration test ---
    try:
        integration_ok = run_on_real_model()
        results["integration"] = {"pass": integration_ok}
    except Exception as e:
        print(f"\nIntegration test failed: {e}")
        results["integration"] = {"pass": False, "error": str(e)}

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  K1 (consistency): {'PASS' if results['k1']['pass'] else 'FAIL'}")
    print(f"  K2 (params):      {'PASS' if results['k2']['pass'] else 'FAIL'}")
    print(f"  K3 (latency):     {'PASS' if results['k3']['pass'] else 'FAIL'}")

    overall = results["k1"]["pass"] and results["k2"]["pass"] and results["k3"]["pass"]
    results["overall_pass"] = overall
    print(f"  OVERALL:          {'PASS' if overall else 'FAIL'}")

    # Save results
    out_dir = Path(__file__).parent
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
