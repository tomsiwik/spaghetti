"""
T4.5: Pierre Adapter Format Compatibility (AUDIT-RERUN, code-bug fix)

Audit findings (LOOPHOLE_{FINDING,CODE,METHODOLOGY}.md) flagged:
  (1) K1088 silently passed when peft library absent (hardcoded bypass).
  (2) K1089 claimed via string suffix match — no vLLM runtime test.
  (3) K1090 claimed via subset fallacy (Theorem 3) — no Unsloth runtime test.
  (4) K1091 metadata lied: measured Grassmannian deviation 0.579 vs predicted <1e-6,
      but the `"property": "orthonormal_rows"` tag was written regardless.
  (5) Upstream dep `exp_p1_t2_single_domain_training` is KILLED — no
      `adapters.safetensors` exists on disk.

Rerun strategy (honest scope):
  - Use SYNTHETIC Grassmannian-initialized adapters (QR decomposition). This
    isolates the FORMAT question from the training-drift question — the latter
    belongs in interference experiments, not format-compat.
  - HARD-require peft: raise on ImportError. Actually call
    peft.LoraConfig(**cfg) and peft.LoraConfig.from_pretrained(dir). This is
    the minimum K1088 claim.
  - K1089 (vLLM runtime) and K1090 (Unsloth training) are CUDA-only and
    unreachable on Apple Silicon per PLAN.md Part 2 ("MLX only — no CUDA").
    We report `skip_reason="cuda_unavailable_on_platform"` and do NOT mark PASS.
    This is a platform constraint, not a bug. Runtime verification lives in
    follow-up `exp_followup_format_compat_peft_required`.
  - K1091 asserts `max_deviation < 1e-6` on synthetic adapters. If exceeded,
    FAIL — do not silently record drifted values under an `orthonormal_rows`
    property tag.

Kill criteria (DB, unchanged):
  K1088: Adapter loads in HF PEFT via LoraConfig (standard format)
  K1089: Adapter loads in vLLM runtime LoRA
  K1090: Adapter trains in Unsloth QLoRA pipeline
  K1091: Grassmannian A stored as adapter_config.json metadata (no code changes)
"""
import json
import time
from pathlib import Path

import numpy as np
import safetensors.numpy as st_np
from safetensors import safe_open

# --- Config ---------------------------------------------------------------
EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / "peft_adapters"
RESULTS_PATH = EXP_DIR / "results.json"

D_IN = 2560   # Gemma 4 q_proj input
D_OUT = 2048  # Gemma 4 q_proj output
RANK = 6
ALPHA = 6.0
N_LAYERS = 42
SEED = 42

GRASSMANNIAN_TOL = 1e-6

REQUIRED_PEFT_FIELDS = {
    "peft_type", "base_model_name_or_path", "r", "lora_alpha",
    "target_modules", "bias",
}


def build_synthetic_grassmannian_adapter():
    """Build synthetic adapter with Grassmannian A (orthonormal rows via QR)."""
    rng = np.random.default_rng(SEED)
    max_deviation = 0.0
    tensors = {}
    for layer in range(N_LAYERS):
        # A: [d_in, r] with orthonormal columns ⇒ A^T A = I_r (i.e. rows of A^T orthonormal)
        M = rng.standard_normal((D_IN, RANK)).astype(np.float32)
        Q, _ = np.linalg.qr(M)          # Q: [d_in, r] with Q^T Q = I_r
        A = Q                           # MLX layout: [d_in, r]
        B = rng.standard_normal((RANK, D_OUT)).astype(np.float32) * 0.01
        tensors[f"language_model.model.layers.{layer}.self_attn.q_proj.lora_a"] = A
        tensors[f"language_model.model.layers.{layer}.self_attn.q_proj.lora_b"] = B

        AtA = A.T @ A
        dev = float(np.max(np.abs(AtA - np.eye(RANK))))
        if dev > max_deviation:
            max_deviation = dev
    return tensors, max_deviation


def convert_to_peft_format(mlx_tensors):
    """MLX (lora_a [d_in,r], lora_b [r,d_out]) → PEFT (lora_A.weight [r,d_in], lora_B.weight [d_out,r])."""
    peft_tensors = {}
    for key, val in mlx_tensors.items():
        if key.endswith(".lora_a"):
            new_key = "base_model.model." + key.replace(".lora_a", ".lora_A.weight")
            peft_tensors[new_key] = val.T  # [r, d_in]
        elif key.endswith(".lora_b"):
            new_key = "base_model.model." + key.replace(".lora_b", ".lora_B.weight")
            peft_tensors[new_key] = val.T  # [d_out, r]
    return peft_tensors


def write_adapter(peft_tensors, max_deviation):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "math_peft"
    out.mkdir(parents=True, exist_ok=True)

    st_np.save_file(peft_tensors, str(out / "adapter_model.safetensors"))

    # Only claim `orthonormal_rows` if deviation is actually under tolerance
    property_claim = "orthonormal_rows" if max_deviation < GRASSMANNIAN_TOL else "drifted_from_orthonormal"

    cfg = {
        "peft_type": "LORA",
        "base_model_name_or_path": "mlx-community/gemma-4-e4b-it-4bit",
        "r": RANK,
        "lora_alpha": ALPHA,
        "target_modules": ["q_proj"],
        "bias": "none",
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "modules_to_save": None,
        "task_type": "CAUSAL_LM",
        "pierre_metadata": {
            "construction": "qr",
            "property": property_claim,
            "rank": RANK,
            "scale": ALPHA,
            "verified_max_deviation": max_deviation,
            "grassmannian_tolerance": GRASSMANNIAN_TOL,
        },
    }
    with open(out / "adapter_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg, out


def k1088_peft_load(cfg, adapter_dir):
    """K1088: adapter loads in HF PEFT via LoraConfig. HARD-require peft — no bypass."""
    # No ImportError bypass. If peft is missing, the KC fails outright.
    import peft as peft_lib

    # (a) Construct LoraConfig directly from our dict
    lora_cfg = peft_lib.LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["target_modules"],
        bias=cfg["bias"],
        lora_dropout=cfg["lora_dropout"],
        task_type=cfg.get("task_type", "CAUSAL_LM"),
    )
    assert lora_cfg.r == cfg["r"]
    assert lora_cfg.lora_alpha == cfg["lora_alpha"]

    # (b) Round-trip through PeftConfig.from_pretrained (reads adapter_config.json)
    loaded = peft_lib.PeftConfig.from_pretrained(str(adapter_dir))
    assert loaded.peft_type == peft_lib.PeftType.LORA or str(loaded.peft_type).endswith("LORA")
    assert getattr(loaded, "r", None) == cfg["r"]
    assert list(getattr(loaded, "target_modules", [])) == cfg["target_modules"]

    # (c) Safetensor keys follow PEFT naming (lora_A.weight / lora_B.weight)
    st_path = adapter_dir / "adapter_model.safetensors"
    with safe_open(str(st_path), framework="numpy") as f:
        keys = list(f.keys())
    a_keys = [k for k in keys if k.endswith(".lora_A.weight")]
    b_keys = [k for k in keys if k.endswith(".lora_B.weight")]
    assert len(a_keys) == N_LAYERS, f"expected {N_LAYERS} A keys, got {len(a_keys)}"
    assert len(b_keys) == N_LAYERS, f"expected {N_LAYERS} B keys, got {len(b_keys)}"

    # (d) Required PEFT fields present
    missing = REQUIRED_PEFT_FIELDS - set(cfg.keys())
    assert not missing, f"missing PEFT fields: {missing}"

    return {
        "pass": True,
        "peft_version": peft_lib.__version__,
        "loraconfig_constructed": True,
        "peftconfig_from_pretrained": True,
        "a_keys": len(a_keys),
        "b_keys": len(b_keys),
        "required_fields_present": len(REQUIRED_PEFT_FIELDS),
    }


def k1089_vllm_runtime():
    """K1089: vLLM runtime LoRA load. CUDA-only — unreachable on Apple Silicon."""
    return {
        "pass": False,
        "skip_reason": "cuda_unavailable_on_platform",
        "note": "vLLM runtime LoRA requires CUDA; PLAN.md Part 2 restricts repo to MLX. "
                "Structural check (string suffix + shapes) is NOT equivalent to runtime "
                "load (vLLM often requires fused qkv_proj; Theorem 3 subset fallacy "
                "acknowledged). Runtime verification scoped to "
                "exp_followup_format_compat_peft_required.",
    }


def k1090_unsloth_runtime():
    """K1090: Unsloth QLoRA training pipeline. CUDA-only — unreachable on Apple Silicon."""
    return {
        "pass": False,
        "skip_reason": "cuda_unavailable_on_platform",
        "note": "Unsloth QLoRA requires CUDA + bitsandbytes. Cannot be run on MLX target. "
                "Theorem 3 (PEFT-compat ⟹ Unsloth-compat) was a subset-direction fallacy "
                "and is retracted.",
    }


def k1091_grassmannian_metadata(cfg, max_deviation):
    """K1091: Grassmannian A stored as metadata; claim only when deviation < tolerance."""
    meta = cfg.get("pierre_metadata", {})
    required = {"construction", "property", "rank", "scale", "verified_max_deviation"}
    fields_ok = required.issubset(set(meta.keys()))

    deviation_ok = max_deviation < GRASSMANNIAN_TOL
    property_truthful = (
        (deviation_ok and meta.get("property") == "orthonormal_rows")
        or (not deviation_ok and meta.get("property") == "drifted_from_orthonormal")
    )

    # Round-trip the config
    rt_ok = False
    try:
        with open(OUT_DIR / "math_peft" / "adapter_config.json") as f:
            reloaded = json.load(f)
        rt_meta = reloaded.get("pierre_metadata", {})
        rt_ok = (
            rt_meta.get("construction") == meta["construction"]
            and rt_meta.get("rank") == meta["rank"]
            and abs(rt_meta.get("verified_max_deviation", -1) - meta["verified_max_deviation"]) < 1e-12
        )
    except Exception as e:  # noqa: BLE001
        rt_ok = False

    return {
        "pass": bool(fields_ok and deviation_ok and property_truthful and rt_ok),
        "fields_ok": fields_ok,
        "deviation_under_tol": deviation_ok,
        "property_claim_truthful": property_truthful,
        "roundtrip_ok": rt_ok,
        "max_deviation": float(max_deviation),
        "tolerance": GRASSMANNIAN_TOL,
    }


def main():
    t0 = time.time()
    results = {
        "experiment": "exp_p1_t4_adapter_format_compat",
        "rerun_context": "audit-2026-04-17 code-bug fix; synthetic Grassmannian because upstream dep KILLED",
        "assumptions": [
            "Synthetic QR-initialized A-matrices are an acceptable substrate for a FORMAT test "
            "(the KC concerns file/schema bijection, not training-drift).",
            "K1089/K1090 require CUDA runtimes; unreachable on MLX/Apple Silicon. "
            "Reported as skip, not pass.",
            "Theorem 3 (PEFT-compat ⟹ vLLM/Unsloth-compat) is a subset-direction fallacy; retracted.",
        ],
    }

    print("\n=== Phase 1: Build synthetic Grassmannian adapter ===")
    mlx_tensors, max_dev = build_synthetic_grassmannian_adapter()
    print(f"synthetic Grassmannian max_deviation = {max_dev:.2e} (tol {GRASSMANNIAN_TOL:.0e})")
    results["phases"] = {
        "synthetic_build": {
            "n_layers": N_LAYERS,
            "rank": RANK,
            "d_in": D_IN,
            "d_out": D_OUT,
            "max_deviation": float(max_dev),
        }
    }

    print("\n=== Phase 2: Convert to PEFT format ===")
    peft_tensors = convert_to_peft_format(mlx_tensors)
    cfg, adapter_dir = write_adapter(peft_tensors, max_dev)
    results["phases"]["convert"] = {
        "peft_tensors": len(peft_tensors),
        "adapter_dir": str(adapter_dir),
    }

    print("\n=== K1088: PEFT LoraConfig (hard-required peft) ===")
    try:
        k1088 = k1088_peft_load(cfg, adapter_dir)
    except ImportError as e:
        k1088 = {"pass": False, "error": f"peft not installed: {e}"}
    except Exception as e:  # noqa: BLE001
        k1088 = {"pass": False, "error": f"{type(e).__name__}: {e}"}
    print(f"K1088: {'PASS' if k1088['pass'] else 'FAIL'}  {k1088}")
    results["k1088"] = k1088

    print("\n=== K1089: vLLM runtime (platform-unavailable) ===")
    k1089 = k1089_vllm_runtime()
    print(f"K1089: SKIP ({k1089['skip_reason']})")
    results["k1089"] = k1089

    print("\n=== K1090: Unsloth runtime (platform-unavailable) ===")
    k1090 = k1090_unsloth_runtime()
    print(f"K1090: SKIP ({k1090['skip_reason']})")
    results["k1090"] = k1090

    print("\n=== K1091: Grassmannian metadata ===")
    k1091 = k1091_grassmannian_metadata(cfg, max_dev)
    print(f"K1091: {'PASS' if k1091['pass'] else 'FAIL'}  {k1091}")
    results["k1091"] = k1091

    all_testable_pass = k1088["pass"] and k1091["pass"]
    verdict = (
        "KILLED"  # cannot satisfy K1089/K1090 on platform → cannot mark supported
    )

    results["elapsed_s"] = round(time.time() - t0, 2)
    results["all_pass"] = False  # K1089/K1090 are not pass — verdict CANNOT be `supported`
    results["testable_on_platform_pass"] = bool(all_testable_pass)
    results["verdict"] = verdict

    print("\n=== SUMMARY ===")
    print(f"K1088 (PEFT LoraConfig):     {'PASS' if k1088['pass'] else 'FAIL'}")
    print(f"K1089 (vLLM runtime):        SKIP (cuda unavailable)")
    print(f"K1090 (Unsloth runtime):     SKIP (cuda unavailable)")
    print(f"K1091 (Grassmannian meta):   {'PASS' if k1091['pass'] else 'FAIL'}")
    print(f"Verdict: {verdict}  (K1089/K1090 unreachable on platform)")
    print(f"Elapsed: {results['elapsed_s']}s")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
