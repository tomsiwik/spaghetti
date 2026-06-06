"""
exp_method_adapter_depth_audit — measurability audit of pre-registered KCs
against available dependency artifacts.

Rationale in MATH.md: the pre-registered KCs (K1727, K1728, K1729) require
(a) a method adapter with weight support across all layers, (b) a validated
domain adapter, and (c) a residual-stream intervention harness. The only
available candidate adapter (method_multi from exp_method_vs_domain_adapter)
has num_layers=16 -> support ⊂ [L/2, L]. Per PLAN.md §1 we cannot reformulate
the KCs after the fact; we measure the geometry and report failure.

This script does NOT invent a result. It enumerates:
  - Gemma-4-E4B layer count L
  - adapter layer support (from safetensors keys)
  - KC-measurability booleans
and writes results.json + a verdict.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
from safetensors import safe_open

import gc

HERE = Path(__file__).resolve().parent
DEP_DIR = HERE.parent / "exp_method_vs_domain_adapter"
METHOD_ADAPTER = DEP_DIR / "adapters" / "method_multi" / "adapters.safetensors"
METHOD_CONFIG = DEP_DIR / "adapters" / "method_multi" / "adapter_config.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

OUT = HERE / "results.json"


def count_base_layers() -> int:
    """Load the base model metadata and return the number of transformer layers."""
    from mlx_lm import load

    model, _ = load(MODEL_ID)
    # Gemma-like models expose .model.layers (or .layers)
    layers = None
    for attr in ("model", "language_model"):
        inner = getattr(model, attr, None)
        if inner is not None and hasattr(inner, "layers"):
            layers = inner.layers
            break
    if layers is None and hasattr(model, "layers"):
        layers = model.layers
    n = len(layers) if layers is not None else -1
    del model
    gc.collect()
    mx.clear_cache()
    return n


def enumerate_adapter_layers(weight_path: Path) -> dict[str, Any]:
    """Read the safetensors file and extract which layer indices have LoRA weights."""
    indices: set[int] = set()
    keys: list[str] = []
    modules: set[str] = set()
    with safe_open(str(weight_path), framework="numpy") as f:
        for k in f.keys():
            keys.append(k)
            # expected key form: ...layers.<idx>....lora_a.weight etc.
            parts = k.split(".")
            for i, tok in enumerate(parts):
                if tok == "layers" and i + 1 < len(parts):
                    try:
                        indices.add(int(parts[i + 1]))
                    except ValueError:
                        pass
            if "v_proj" in k:
                modules.add("v_proj")
            if "o_proj" in k:
                modules.add("o_proj")
            if "q_proj" in k:
                modules.add("q_proj")
            if "k_proj" in k:
                modules.add("k_proj")
    return {
        "n_keys": len(keys),
        "layer_indices": sorted(indices),
        "modules": sorted(modules),
        "sample_keys": keys[:6],
    }


def measurability(kc_band: tuple[int, int], adapter_layers: list[int]) -> dict[str, Any]:
    """Return whether zero-mask-ablation at the KC band would be non-degenerate."""
    lo, hi = kc_band
    overlap = [ell for ell in adapter_layers if lo <= ell <= hi]
    return {
        "kc_band": [lo, hi],
        "adapter_layers_in_band": overlap,
        "overlap_size": len(overlap),
        "degenerate": len(overlap) == 0,  # zero-mask on zero weights == no-op
    }


def main() -> None:
    print("[audit] enumerating base-model layer count…")
    try:
        n_base_layers = count_base_layers()
    except Exception as e:  # platform/model load failure is informative
        n_base_layers = -1
        load_error = str(e)
    else:
        load_error = None

    print(f"[audit] base layers: {n_base_layers}")

    print("[audit] reading adapter weights…")
    adapter_info = enumerate_adapter_layers(METHOD_ADAPTER)
    config = json.loads(METHOD_CONFIG.read_text())
    cfg_num_layers = config.get("num_layers")
    cfg_keys = config.get("lora_parameters", {}).get("keys", [])

    adapter_layers = adapter_info["layer_indices"]

    # KC bands as written in the pre-registered criteria
    k1727_low = (0, 8)
    k1727_high = (24, n_base_layers - 1 if n_base_layers > 0 else 31)

    k1727_low_meas = measurability(k1727_low, adapter_layers)
    k1727_high_meas = measurability(k1727_high, adapter_layers)

    # K1727 is measurable only if BOTH ablation arms are non-degenerate
    k1727_measurable = (
        not k1727_low_meas["degenerate"] and not k1727_high_meas["degenerate"]
    )

    # K1728 requires a validated domain adapter
    domain_adapter_available = False
    domain_notes = (
        "No validated domain adapter in repo. "
        "exp_knowledge_disentanglement_control KILLED (Finding kills), "
        "exp_prompt_erasure_gemma4 KILLED (Finding #588). "
        "exp_method_vs_domain_adapter trained only method adapters (method_multi, method_single_math)."
    )

    # K1729 requires a residual-stream intervention harness
    residual_harness_available = False
    residual_notes = (
        "No residual-stream intervention harness exists in the repo. "
        "Building one (per Todd et al. §3) requires forward hooks on all 32 transformer "
        "blocks + calibration against a known function vector; out of scope for one hat iteration."
    )

    results = {
        "experiment": "exp_method_adapter_depth_audit",
        "model": MODEL_ID,
        "is_smoke": False,
        "mode": "measurability_audit",
        "base_layer_count": n_base_layers,
        "base_load_error": load_error,
        "adapter": {
            "path": str(METHOD_ADAPTER.relative_to(HERE.parent.parent)),
            "config_num_layers": cfg_num_layers,
            "config_keys": cfg_keys,
            "enumerated_layer_indices": adapter_layers,
            "adapter_n_keys": adapter_info["n_keys"],
            "modules": adapter_info["modules"],
        },
        "kc_analysis": {
            "K1727": {
                "text": "Ablating method adapter at layers 0-8 causes >=2x drop in procedural benchmark vs ablating layers 24-32",
                "low_band_measurability": k1727_low_meas,
                "high_band_measurability": k1727_high_meas,
                "measurable": k1727_measurable,
                "pass": False,
                "reason": (
                    "degenerate_low_band" if k1727_low_meas["degenerate"]
                    else ("degenerate_high_band" if k1727_high_meas["degenerate"] else "not_run")
                ),
            },
            "K1728": {
                "text": "Domain adapter ablation shows opposite pattern (later layers more causal)",
                "domain_adapter_available": domain_adapter_available,
                "measurable": domain_adapter_available,
                "pass": False,
                "reason": "prerequisite_missing_domain_adapter",
                "notes": domain_notes,
            },
            "K1729": {
                "text": "Residual stream intervention at early-middle layers reproduces method effect within 1pp",
                "residual_harness_available": residual_harness_available,
                "measurable": residual_harness_available,
                "pass": False,
                "reason": "prerequisite_missing_residual_stream_harness",
                "notes": residual_notes,
            },
        },
        "all_pass": False,
        "verdict": "KILLED",
        "kill_reason": "dependency_structural_incompatibility",
        "notes": (
            "Pre-registered KCs assume an adapter with weight support across all layer "
            "bands and a validated method+domain adapter pair. The only candidate adapter "
            "(method_multi from exp_method_vs_domain_adapter) was trained with num_layers=16 "
            "in mlx-lm, which applies LoRA to the LAST 16 layers only. Layers 0-8 have zero "
            "LoRA weights, making K1727's low-band ablation a degenerate no-op. K1728 has no "
            "validated domain adapter. K1729 has no intervention harness. Per PLAN.md §1 "
            "kill-criteria discipline (no post-hoc reformulation), this is KILLED. v2 design "
            "guidance recorded in MATH.md."
        ),
    }

    OUT.write_text(json.dumps(results, indent=2))
    print(f"[audit] wrote {OUT}")
    print(f"[audit] verdict: {results['verdict']}")
    print(f"[audit] K1727 measurable: {k1727_measurable} (low band degenerate: {k1727_low_meas['degenerate']})")
    print(f"[audit] K1728 measurable: {domain_adapter_available}")
    print(f"[audit] K1729 measurable: {residual_harness_available}")


if __name__ == "__main__":
    main()
