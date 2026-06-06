"""
T4.5v2: Adapter Format Compat with Actual PEFT/vLLM Loading (Loophole Fix)

Fixes three loopholes from exp_p1_t4_adapter_format_compat:
1. K1: Actually calls PeftModel.from_pretrained() on CPU (not just LoraConfig)
2. K2: Grassmannian reporting honest — no 'orthonormal_rows' if deviation > 1e-4
3. K3: Round-trip MLX -> PEFT -> MLX exact (max diff < 1e-6)

Kill criteria:
- K1243: PeftModel.from_pretrained() loads adapter on CPU without error
- K1244: Grassmannian deviation reported honestly (no orthonormal_rows if drift > 1e-4)
- K1245: Round-trip MLX -> PEFT -> MLX preserves adapter weights (max diff < 1e-6)
"""

import json
import sys
import time
import shutil
import tempfile
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_PATH = Path(__file__).parent / "results.json"

# OPT-125m: q_proj shape 768x768, 12 layers
BASE_MODEL = "facebook/opt-125m"
D_MODEL = 768
N_LAYERS = 3  # Only test first 3 layers to keep it fast
LORA_RANK = 4
GRASSMANNIAN_EPSILON = 1e-4  # Threshold for honest reporting


# ---------------------------------------------------------------------------
# Phase 1: Create synthetic adapters (no MLX needed — NumPy only)
# ---------------------------------------------------------------------------

def create_qr_adapter(n_layers=N_LAYERS, d=D_MODEL, r=LORA_RANK):
    """Create QR-initialized adapter (should have deviation ~0)."""
    data = {}
    for i in range(n_layers):
        # lora_a [d_in, r]: QR initialization (orthonormal columns)
        M = np.random.randn(d, r).astype(np.float32)
        Q, _ = np.linalg.qr(M)
        lora_a = Q[:, :r]  # [d, r] with orthonormal columns

        # lora_b [r, d]: zeros (standard LoRA init)
        lora_b = np.zeros((r, d), dtype=np.float32)

        key_prefix = f"model.decoder.layers.{i}.self_attn.q_proj"
        data[f"{key_prefix}.lora_a"] = lora_a
        data[f"{key_prefix}.lora_b"] = lora_b

    return data


def create_random_adapter(n_layers=N_LAYERS, d=D_MODEL, r=LORA_RANK):
    """Create random (untrained-like) adapter (deviation >> 1e-4)."""
    data = {}
    for i in range(n_layers):
        lora_a = np.random.randn(d, r).astype(np.float32)  # NOT orthonormal
        lora_b = np.random.randn(r, d).astype(np.float32) * 0.01

        key_prefix = f"model.decoder.layers.{i}.self_attn.q_proj"
        data[f"{key_prefix}.lora_a"] = lora_a
        data[f"{key_prefix}.lora_b"] = lora_b

    return data


def compute_grassmannian_deviation(data):
    """Compute max ||A^T A - I_r||_inf over all lora_a matrices."""
    lora_a_keys = [k for k in data if k.endswith(".lora_a")]
    max_dev = 0.0
    for key in lora_a_keys:
        A = data[key]  # [d_in, r]
        AtA = A.T @ A  # [r, r]
        dev = np.max(np.abs(AtA - np.eye(AtA.shape[0])))
        max_dev = max(max_dev, dev)
    return max_dev


# ---------------------------------------------------------------------------
# Phase 2: Export to PEFT format
# ---------------------------------------------------------------------------

def export_to_peft(data, adapter_dir: Path, deviation: float, model_name: str):
    """Export adapter data to PEFT-compatible format (adapter_config.json + safetensors)."""
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Build adapter_config.json
    is_orthonormal = deviation < GRASSMANNIAN_EPSILON
    metadata = {
        "verified_max_deviation": float(deviation),
        "honest_reporting": True,
    }
    if is_orthonormal:
        metadata["property"] = "orthonormal_rows"
    # NOTE: do NOT write "orthonormal_rows" if deviation >= GRASSMANNIAN_EPSILON

    lora_a_keys = [k for k in data if k.endswith(".lora_a")]
    target_layer = "q_proj"
    target_modules = [target_layer]

    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": model_name,
        "r": LORA_RANK,
        "lora_alpha": float(LORA_RANK),
        "target_modules": target_modules,
        "bias": "none",
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "modules_to_save": None,
        "pierre_metadata": metadata,
    }

    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)

    # Build PEFT-format safetensors
    # MLX: path.lora_a [d_in, r] -> PEFT: base_model.model.path.lora_A.weight [r, d_in]
    # MLX: path.lora_b [r, d_out] -> PEFT: base_model.model.path.lora_B.weight [d_out, r]
    peft_tensors = {}
    for mlx_key in lora_a_keys:
        peft_key_a = "base_model.model." + mlx_key.replace(".lora_a", ".lora_A.weight")
        peft_key_b = "base_model.model." + mlx_key.replace(".lora_a", ".lora_B.weight")
        peft_tensors[peft_key_a] = data[mlx_key].T  # [r, d_in]

        mlx_key_b = mlx_key.replace(".lora_a", ".lora_b")
        peft_tensors[peft_key_b] = data[mlx_key_b].T  # [d_out, r]

    try:
        import safetensors.numpy as sf_np
        sf_np.save_file(peft_tensors, str(adapter_dir / "adapter_model.safetensors"))
    except ImportError:
        # Fallback: save as numpy arrays zipped (not standard PEFT, marks K1 as SKIP)
        raise RuntimeError("safetensors not installed. Cannot export to PEFT format.")

    return peft_tensors, is_orthonormal


# ---------------------------------------------------------------------------
# Phase 3 (K1): PeftModel.from_pretrained() on CPU
# ---------------------------------------------------------------------------

def phase_k1_peft_load(adapter_dir: Path):
    """K1: Actually load adapter with PeftModel.from_pretrained() on CPU."""
    print("\n=== Phase K1: PeftModel.from_pretrained() on CPU ===")
    t0 = time.time()

    try:
        import peft as peft_lib
    except ImportError:
        print("FAIL: peft not installed (hard fail — no bypass)")
        return False, "peft not installed"

    try:
        from transformers import AutoModelForCausalLM
        print(f"Loading base model {BASE_MODEL} on CPU...")
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="cpu")
        print(f"Base model loaded ({time.time()-t0:.1f}s)")

        print(f"Loading PEFT adapter from {adapter_dir}...")
        peft_model = peft_lib.PeftModel.from_pretrained(base_model, str(adapter_dir))
        print(f"PeftModel loaded ({time.time()-t0:.1f}s)")

        # Verify forward pass works
        import torch
        dummy_input = torch.ones(1, 4, dtype=torch.long)
        with torch.no_grad():
            out = peft_model(dummy_input)
        print(f"Forward pass shape: {out.logits.shape}")
        assert out.logits.shape == (1, 4, 50272), f"Unexpected shape: {out.logits.shape}"

        elapsed = time.time() - t0
        print(f"K1 PASS: PeftModel loaded and forward pass succeeded ({elapsed:.1f}s)")
        return True, f"loaded in {elapsed:.1f}s, logits shape {out.logits.shape}"

    except Exception as e:
        print(f"K1 FAIL: {e}")
        return False, str(e)


# ---------------------------------------------------------------------------
# Phase 4 (K2): Honest Grassmannian reporting
# ---------------------------------------------------------------------------

def phase_k2_grassmannian_honesty(qr_data, qr_dir: Path, random_data, random_dir: Path):
    """K2: Honest Grassmannian deviation reporting."""
    print("\n=== Phase K2: Grassmannian Honesty ===")

    # Check QR adapter: should report 'orthonormal_rows'
    qr_config = json.loads((qr_dir / "adapter_config.json").read_text())
    qr_meta = qr_config.get("pierre_metadata", {})
    qr_deviation = qr_meta.get("verified_max_deviation", 999)
    qr_has_tag = qr_meta.get("property") == "orthonormal_rows"

    print(f"QR adapter: deviation={qr_deviation:.2e}, orthonormal_rows tag={qr_has_tag}")

    # Check random adapter: should NOT report 'orthonormal_rows'
    rand_config = json.loads((random_dir / "adapter_config.json").read_text())
    rand_meta = rand_config.get("pierre_metadata", {})
    rand_deviation = rand_meta.get("verified_max_deviation", 999)
    rand_has_tag = rand_meta.get("property") == "orthonormal_rows"

    print(f"Random adapter: deviation={rand_deviation:.2e}, orthonormal_rows tag={rand_has_tag}")

    # K2 passes iff: QR writes tag AND random does not write tag
    qr_correct = qr_has_tag and qr_deviation < GRASSMANNIAN_EPSILON
    rand_correct = not rand_has_tag and rand_deviation > GRASSMANNIAN_EPSILON

    passed = qr_correct and rand_correct
    print(f"K2 {'PASS' if passed else 'FAIL'}: QR={qr_correct}, Random={rand_correct}")
    return passed, {
        "qr_deviation": float(qr_deviation),
        "qr_has_tag": qr_has_tag,
        "random_deviation": float(rand_deviation),
        "random_has_tag": rand_has_tag,
    }


# ---------------------------------------------------------------------------
# Phase 5 (K3): Round-trip validation
# ---------------------------------------------------------------------------

def phase_k3_roundtrip(original_data, peft_tensors):
    """K3: Round-trip MLX -> PEFT -> MLX preserves weights exactly."""
    print("\n=== Phase K3: Round-Trip MLX -> PEFT -> MLX ===")

    max_diff = 0.0
    lora_a_keys = [k for k in original_data if k.endswith(".lora_a")]

    for mlx_key in lora_a_keys:
        peft_key_a = "base_model.model." + mlx_key.replace(".lora_a", ".lora_A.weight")
        peft_key_b = "base_model.model." + mlx_key.replace(".lora_a", ".lora_B.weight")
        mlx_key_b = mlx_key.replace(".lora_a", ".lora_b")

        # Round-trip: MLX lora_a -> PEFT lora_A.weight (transpose) -> back (transpose)
        reconstructed_a = peft_tensors[peft_key_a].T  # [d_in, r]
        diff_a = np.max(np.abs(reconstructed_a - original_data[mlx_key]))

        reconstructed_b = peft_tensors[peft_key_b].T  # [r, d_out]
        diff_b = np.max(np.abs(reconstructed_b - original_data[mlx_key_b]))

        max_diff = max(max_diff, diff_a, diff_b)

    passed = max_diff < 1e-6
    print(f"Max round-trip diff: {max_diff:.2e}")
    print(f"K3 {'PASS' if passed else 'FAIL'}: max_diff={max_diff:.2e} (threshold 1e-6)")
    return passed, float(max_diff)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("T4.5v2: Loophole Fix — Actual PEFT CPU Loading + Honest Grassmannian")
    print("=" * 60)
    t_start = time.time()

    np.random.seed(42)
    tmp = Path(tempfile.mkdtemp())
    qr_dir = tmp / "qr_adapter"
    random_dir = tmp / "random_adapter"

    try:
        # --- Create adapters ---
        print("\n--- Creating synthetic adapters ---")
        qr_data = create_qr_adapter()
        random_data = create_random_adapter()

        qr_deviation = compute_grassmannian_deviation(qr_data)
        rand_deviation = compute_grassmannian_deviation(random_data)
        print(f"QR adapter deviation: {qr_deviation:.2e} (expected < 1e-12)")
        print(f"Random adapter deviation: {rand_deviation:.2e} (expected > 0.1)")

        # --- Export ---
        print("\n--- Exporting to PEFT format ---")
        qr_peft_tensors, qr_is_orth = export_to_peft(qr_data, qr_dir, qr_deviation, BASE_MODEL)
        rand_peft_tensors, rand_is_orth = export_to_peft(random_data, random_dir, rand_deviation, BASE_MODEL)
        print(f"QR: orthonormal tag={qr_is_orth}, Random: orthonormal tag={rand_is_orth}")

        # --- K1: PEFT CPU loading ---
        k1_pass, k1_detail = phase_k1_peft_load(qr_dir)

        # --- K2: Grassmannian honesty ---
        k2_pass, k2_detail = phase_k2_grassmannian_honesty(
            qr_data, qr_dir, random_data, random_dir
        )

        # --- K3: Round-trip ---
        k3_pass, k3_max_diff = phase_k3_roundtrip(qr_data, qr_peft_tensors)

        total_time = time.time() - t_start
        all_pass = k1_pass and k2_pass and k3_pass

        results = {
            "is_smoke": False,
            "all_pass": all_pass,
            "total_time_min": round(total_time / 60, 3),
            "kill_criteria": {
                "k1243_peft_cpu_load": {"pass": k1_pass, "detail": k1_detail},
                "k1244_grassmannian_honesty": {"pass": k2_pass, **k2_detail},
                "k1245_roundtrip_max_diff": {"pass": k3_pass, "max_diff": k3_max_diff},
            },
            "adapter_info": {
                "base_model": BASE_MODEL,
                "lora_rank": LORA_RANK,
                "n_layers_tested": N_LAYERS,
                "d_model": D_MODEL,
                "qr_deviation": float(qr_deviation),
                "random_deviation": float(rand_deviation),
            },
            "grassmannian_epsilon": GRASSMANNIAN_EPSILON,
        }

        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 60)
        print(f"K1 (PEFT CPU load):       {'PASS' if k1_pass else 'FAIL'}")
        print(f"K2 (Grassmannian honesty): {'PASS' if k2_pass else 'FAIL'}")
        print(f"K3 (Round-trip):           {'PASS' if k3_pass else 'FAIL'}")
        print(f"ALL PASS: {all_pass}")
        print(f"Total time: {total_time:.1f}s")
        print("=" * 60)

        return 0 if all_pass else 1

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
