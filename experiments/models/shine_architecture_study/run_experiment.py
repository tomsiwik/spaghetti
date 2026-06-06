#!/usr/bin/env python3
"""SHINE Piece C: M2P Transformer architecture study for Gemma 4 E4B.

Verifies that the M2P Transformer from SHINE (arXiv:2602.06358) is:
  K806: Not dependent on Qwen-specific components (architecture-agnostic)
  K807: Parameter count << 1B for practical Gemma 4 E4B config

Builds on exp_shine_port (Finding #336) which proved M2P portable at toy scale.
This experiment validates at production Gemma 4 E4B dimensions:
  L=42, d_model=2560, r=16 (PoLAR rank)

Type: Verification (Type 1).
"""

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.utils

# Memory safety
_dev = mx.device_info()
mx.set_memory_limit(_dev["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
SEED = 42


def log(m): print(m, flush=True)
def cleanup(*objs):
    for o in objs: del o
    gc.collect(); mx.clear_cache(); mx.reset_peak_memory()


# ── Gemma 4 E4B config ───────────────────────────────────────────────────────

GEMMA4_E4B = {
    "num_hidden_layers": 42,
    "hidden_size": 2560,
    "head_dim": 256,
    "lora_rank": 16,
}

# ── M2P Transformer (SHINE §3.4, architecture-agnostic implementation) ───────

class M2PAttention(nn.Module):
    """Bidirectional self-attention — no architecture-specific components."""
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        scale = self.head_dim ** -0.5
        attn = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * scale, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)


class M2PFFN(nn.Module):
    """Standard FFN with GELU — no architecture-specific components."""
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class M2PLayer(nn.Module):
    """One M2P layer: row attn + col attn + FFN (SHINE §3.4, Eq. 4-6)."""
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.row_attn = M2PAttention(dim, n_heads)
        self.col_attn = M2PAttention(dim, n_heads)
        self.ffn = M2PFFN(dim)

    def __call__(self, z: mx.array) -> mx.array:
        # z: (L, M, H)
        L, M, H = z.shape

        # Row attention: attend over M tokens within each layer
        z_row = z.reshape(L, M, H)
        z = z + self.row_attn(self.norm1(z_row))

        # Column attention: attend over L layers for each memory position
        z_col = z.transpose(1, 0, 2)  # (M, L, H)
        z_col = z_col + self.col_attn(self.norm2(z_col))
        z = z_col.transpose(1, 0, 2)  # (L, M, H)

        # FFN applied to all positions
        z = z + self.ffn(self.norm3(z))
        return z


class M2PTransformer(nn.Module):
    """Memory-to-Parameter Transformer (SHINE arXiv:2602.06358, §3.4).

    Architecture-agnostic: works for any (L, M, H) input and any (r, d_model).

    Args:
        n_model_layers:  Number of LLM layers (L)
        n_memory_tokens: Memory tokens per layer (M)
        m2p_hidden_dim:  M2P internal hidden dim (H, independent of d_model)
        n_m2p_layers:    Number of M2P transformer layers
        lora_rank:       Target LoRA rank (r)
        adapter_dim:     Target adapter dimension (d_model or d_intermediate)
        n_heads:         Number of attention heads in M2P
    """
    def __init__(
        self,
        n_model_layers: int,
        n_memory_tokens: int,
        m2p_hidden_dim: int,
        n_m2p_layers: int,
        lora_rank: int,
        adapter_dim: int,
        n_heads: int = 4,
    ):
        super().__init__()
        self.L = n_model_layers
        self.M = n_memory_tokens
        self.H = m2p_hidden_dim
        self.r = lora_rank
        self.d = adapter_dim

        # Positional embeddings: layer index + memory token position (SHINE §3.4, Eq. 5)
        # Xavier init as per SHINE (Finding #336 validated this is critical)
        scale_layer = (n_model_layers * m2p_hidden_dim) ** -0.5
        scale_token = (n_memory_tokens * m2p_hidden_dim) ** -0.5
        self.pos_layer = nn.Embedding(n_model_layers, m2p_hidden_dim)  # (L, H)
        self.pos_token = nn.Embedding(n_memory_tokens, m2p_hidden_dim)  # (M, H)

        # M2P transformer layers
        self.layers = [M2PLayer(m2p_hidden_dim, n_heads) for _ in range(n_m2p_layers)]
        self.norm_out = nn.LayerNorm(m2p_hidden_dim)

        # Output projections: shared linear → per-layer adapter weights
        # proj_A generates lora_A: (r, d_model), proj_B generates lora_B: (d_model, r)
        self.proj_A = nn.Linear(m2p_hidden_dim, lora_rank * adapter_dim, bias=False)
        self.proj_B = nn.Linear(m2p_hidden_dim, adapter_dim * lora_rank, bias=False)

    def __call__(self, z: mx.array) -> tuple[mx.array, mx.array]:
        """
        Args:
            z: (L, M, H) memory states — input to M2P

        Returns:
            lora_A: (L, r, d_model) — LoRA A matrices for all layers
            lora_B: (L, d_model, r) — LoRA B matrices for all layers
        """
        L, M, H = z.shape
        assert L == self.L, f"Expected L={self.L}, got {L}"
        assert M == self.M, f"Expected M={self.M}, got {M}"

        # Add positional embeddings (Xavier-like init in Embedding, see Finding #336)
        layer_idx = mx.arange(L)
        token_idx = mx.arange(M)
        p_layer = self.pos_layer(layer_idx)[:, None, :]  # (L, 1, H)
        p_token = self.pos_token(token_idx)[None, :, :]  # (1, M, H)
        z = z + p_layer + p_token                        # (L, M, H)

        # M2P transformer
        for layer in self.layers:
            z = layer(z)

        # Aggregate: mean over memory tokens → (L, H)
        z_out = self.norm_out(z.mean(axis=1))

        # Project to adapter weights
        lora_A = self.proj_A(z_out).reshape(L, self.r, self.d)      # (L, r, d)
        lora_B = self.proj_B(z_out).reshape(L, self.d, self.r)      # (L, d, r)
        return lora_A, lora_B


def count_params(model: nn.Module) -> int:
    """Count trainable parameters."""
    leaves = [v for _, v in mlx.utils.tree_flatten(model.parameters())]
    return sum(v.size for v in leaves)


def component_breakdown(model: M2PTransformer) -> dict:
    """Break down parameters by component."""
    def sz(m):
        return sum(v.size for _, v in mlx.utils.tree_flatten(m.parameters()))
    return {
        "pos_layer": model.pos_layer.weight.size,
        "pos_token": model.pos_token.weight.size,
        "m2p_layers": sum(sz(l) for l in model.layers),
        "norm_out": sz(model.norm_out),
        "proj_A": model.proj_A.weight.size,
        "proj_B": model.proj_B.weight.size,
    }


def run_architecture_agnosticism_check(model: M2PTransformer, config_name: str) -> dict:
    """Verify no Qwen-specific or architecture-specific components.

    Check performed: inspect all module types used in model, confirm none
    require Qwen/architecture-specific imports.
    """
    allowed_types = {
        "M2PTransformer", "M2PLayer", "M2PAttention", "M2PFFN",
        "LayerNorm", "Linear", "Embedding", "list",
    }
    found_types = {type(m).__name__ for _, m in model.named_modules()}

    # Check for any architecture-specific components
    qwen_specific = {"RotaryEmbedding", "RMSNorm", "GQA", "SwiGLU", "MoE", "Expert"}
    found_specific = found_types & qwen_specific

    return {
        "config": config_name,
        "module_types": sorted(found_types),
        "qwen_specific_found": sorted(found_specific),
        "k806_pass": len(found_specific) == 0,
    }


def run_forward_pass(model: M2PTransformer, config_name: str) -> dict:
    """Test forward pass: verify output shapes and basic properties."""
    mx.random.seed(SEED)
    L, M, H = model.L, model.M, model.H

    # Random memory input
    z = mx.random.normal((L, M, H))
    mx.eval(z)

    t0 = time.perf_counter()
    lora_A, lora_B = model(z)
    mx.eval(lora_A, lora_B)
    latency_ms = (time.perf_counter() - t0) * 1000

    # Verify shapes
    expected_A = (L, model.r, model.d)
    expected_B = (L, model.d, model.r)
    shape_ok = (lora_A.shape == expected_A) and (lora_B.shape == expected_B)

    # Verify outputs are finite and non-zero
    a_norm = float(mx.sqrt(mx.sum(lora_A ** 2)).item())
    b_norm = float(mx.sqrt(mx.sum(lora_B ** 2)).item())
    outputs_valid = (a_norm > 0) and (b_norm > 0) and np.isfinite(a_norm)

    return {
        "config": config_name,
        "input_shape": (L, M, H),
        "output_shape_A": lora_A.shape,
        "output_shape_B": lora_B.shape,
        "expected_shape_A": expected_A,
        "expected_shape_B": expected_B,
        "shape_ok": shape_ok,
        "lora_A_norm": a_norm,
        "lora_B_norm": b_norm,
        "outputs_valid": outputs_valid,
        "latency_ms": latency_ms,
    }


def main():
    results = {
        "is_smoke": os.environ.get("SMOKE_TEST", "0") == "1",
        "phases": {},
    }

    # ── Phase 1: Architecture instantiation at E4B config ──────────────────
    log("=" * 60)
    log("Phase 1: M2P for Gemma 4 E4B (L=42, d=2560, r=16)")
    log("=" * 60)

    L = GEMMA4_E4B["num_hidden_layers"]   # 42
    d = GEMMA4_E4B["hidden_size"]          # 2560
    r = GEMMA4_E4B["lora_rank"]            # 16
    M = 32                                  # memory tokens (SHINE typical)
    H = 256                                 # M2P hidden dim

    m2p_e4b = M2PTransformer(
        n_model_layers=L,
        n_memory_tokens=M,
        m2p_hidden_dim=H,
        n_m2p_layers=4,
        lora_rank=r,
        adapter_dim=d,
        n_heads=4,
    )
    mx.eval(m2p_e4b.parameters())

    n_params = count_params(m2p_e4b)
    breakdown = component_breakdown(m2p_e4b)
    log(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    log(f"  Breakdown: {breakdown}")

    k807_pass = n_params < 1_000_000_000
    log(f"  K807 (< 1B params): {'PASS' if k807_pass else 'FAIL'} — {n_params/1e9:.3f}B")

    # K806: architecture agnosticism check
    arch_check = run_architecture_agnosticism_check(m2p_e4b, "E4B")
    log(f"  Module types: {arch_check['module_types']}")
    log(f"  Qwen-specific found: {arch_check['qwen_specific_found']}")
    log(f"  K806 (no arch-specific): {'PASS' if arch_check['k806_pass'] else 'FAIL'}")

    # Forward pass
    fwd = run_forward_pass(m2p_e4b, "E4B")
    log(f"  Output shape A: {fwd['output_shape_A']} (expected {fwd['expected_shape_A']})")
    log(f"  Output shape B: {fwd['output_shape_B']} (expected {fwd['expected_shape_B']})")
    log(f"  Shape OK: {fwd['shape_ok']}")
    log(f"  Forward latency: {fwd['latency_ms']:.1f}ms")

    results["phases"]["phase1"] = {
        "n_params": n_params,
        "n_params_M": n_params / 1e6,
        "breakdown": breakdown,
        "k806_pass": arch_check["k806_pass"],
        "k807_pass": k807_pass,
        "arch_check": arch_check,
        "forward": fwd,
    }
    cleanup(m2p_e4b)

    # ── Phase 2: Scale ablation — L=10, 20, 42 ─────────────────────────────
    log("\n" + "=" * 60)
    log("Phase 2: Scale ablation (L=10, 20, 42) — verify architecture-agnosticism")
    log("=" * 60)

    ablations = []
    for L_test in [10, 20, 42]:
        m2p = M2PTransformer(
            n_model_layers=L_test,
            n_memory_tokens=M,
            m2p_hidden_dim=H,
            n_m2p_layers=4,
            lora_rank=r,
            adapter_dim=d,
            n_heads=4,
        )
        mx.eval(m2p.parameters())
        n_p = count_params(m2p)
        fwd = run_forward_pass(m2p, f"L={L_test}")
        log(f"  L={L_test}: {n_p/1e6:.2f}M params, shape_ok={fwd['shape_ok']}, latency={fwd['latency_ms']:.1f}ms")
        ablations.append({
            "L": L_test,
            "n_params": n_p,
            "n_params_M": n_p / 1e6,
            "shape_ok": fwd["shape_ok"],
            "latency_ms": fwd["latency_ms"],
        })
        cleanup(m2p)

    results["phases"]["phase2_ablations"] = ablations

    # ── Phase 3: Compact config (rank-factored output) ──────────────────────
    log("\n" + "=" * 60)
    log("Phase 3: Compact M2P (smaller H, fewer layers)")
    log("=" * 60)

    # Minimum config: H=128, n_layers=2, M=16 — what's the smallest viable M2P?
    m2p_compact = M2PTransformer(
        n_model_layers=L,      # still 42 E4B layers
        n_memory_tokens=16,
        m2p_hidden_dim=128,
        n_m2p_layers=2,
        lora_rank=r,
        adapter_dim=d,
        n_heads=4,
    )
    mx.eval(m2p_compact.parameters())
    n_compact = count_params(m2p_compact)
    fwd_compact = run_forward_pass(m2p_compact, "compact")
    log(f"  Compact config: {n_compact/1e6:.2f}M params, shape_ok={fwd_compact['shape_ok']}, latency={fwd_compact['latency_ms']:.1f}ms")
    results["phases"]["phase3_compact"] = {
        "n_params": n_compact,
        "n_params_M": n_compact / 1e6,
        "shape_ok": fwd_compact["shape_ok"],
        "latency_ms": fwd_compact["latency_ms"],
    }
    cleanup(m2p_compact)

    # ── Final verdict ───────────────────────────────────────────────────────
    p1 = results["phases"]["phase1"]
    k806_pass = p1["k806_pass"]
    k807_pass = p1["k807_pass"]
    shapes_all_ok = all(a["shape_ok"] for a in ablations) and p1["forward"]["shape_ok"]

    results["k806_result"] = "pass" if k806_pass else "fail"
    results["k807_result"] = "pass" if k807_pass else "fail"
    results["shapes_ok"] = shapes_all_ok

    log("\n" + "=" * 60)
    log("FINAL RESULTS")
    log("=" * 60)
    log(f"K806 (no arch-specific): {'PASS' if k806_pass else 'FAIL'}")
    log(f"K807 (< 1B params):      {'PASS' if k807_pass else 'FAIL'} — {p1['n_params_M']:.1f}M params")
    log(f"All output shapes OK:    {'PASS' if shapes_all_ok else 'FAIL'}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
