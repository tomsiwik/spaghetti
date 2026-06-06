"""
exp_followup_format_compat_peft_required — K1576 stand-or-fall.

Closes the honest gap left by `exp_p1_t4_adapter_format_compat` (KILLED via
silent-bypass + subset-fallacy, Finding #585) on a CPU-reachable path:

  K1576.a: `import peft` is HARD-required at module top.  No ImportError bypass.
  K1576.b: peft.PeftModel.from_pretrained(base, adapter_dir) + forward pass
           must both succeed on a real HF tiny LlamaForCausalLM.
  K1576.c: Fused-QKV stack reshape structurally valid (Theorem 2 of MATH.md)
           AND adapter enumerates separate q_proj/k_proj/v_proj (subset-fallacy
           guard — exposes the non-trivial transformation required for fused
           runtimes like vLLM).

Design notes:
  - No MLX, no HF-Hub download: build a tiny LlamaConfig locally.
  - Synthetic Grassmannian A via QR (matches parent substrate).
  - Hard-fail on missing peft / transformers / torch at top-level import.
  - No `try/except` around the load call: a real exception → KC fails and
    the traceback is preserved in results.json["k1576_failure"].
"""
# --- Hard-required imports (K1576.a). No ImportError bypass below.
import peft                    # noqa: E402 — intentional hard-require
import transformers            # noqa: E402
import torch                   # noqa: E402

import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import safetensors.torch as st_t


EXP_DIR = Path(__file__).parent
ADAPTER_DIR = EXP_DIR / "peft_adapters" / "qkv_tiny"
RESULTS_PATH = EXP_DIR / "results.json"

# Tiny base dims — keeps CPU run <10s and avoids HF-Hub downloads.
HIDDEN = 64
N_LAYERS = 4
N_HEADS = 4
HEAD_DIM = HIDDEN // N_HEADS
VOCAB = 256
RANK = 6
ALPHA = 6.0
SEED = 42
GRASSMANNIAN_TOL = 1e-6


# ---------------------------------------------------------------------------
# Synthetic Grassmannian adapter builder
# ---------------------------------------------------------------------------
def build_grassmannian_A(d_in, rank, rng):
    """QR-based Grassmannian: A ∈ ℝ^{d_in, rank} with orthonormal columns."""
    M = rng.standard_normal((d_in, rank)).astype(np.float32)
    Q, _ = np.linalg.qr(M)
    return Q  # [d_in, rank], Q^T Q = I_r


def build_adapter_peft_layout(d_in, d_out, rank, n_layers, rng):
    """Build PEFT-native state_dict for separate q/k/v targets.

    PEFT convention (for base_model.model.<path>.<proj>.default):
      base_model.model.<path>.<proj>.lora_A.weight : [r, d_in]
      base_model.model.<path>.<proj>.lora_B.weight : [d_out, r]
    """
    sd = {}
    max_dev = 0.0
    for layer in range(n_layers):
        for proj in ("q_proj", "k_proj", "v_proj"):
            A = build_grassmannian_A(d_in, rank, rng)         # [d_in, r]
            B = rng.standard_normal((rank, d_out)).astype(np.float32) * 0.01

            key_prefix = (
                f"base_model.model.model.layers.{layer}"
                f".self_attn.{proj}"
            )
            # PEFT stores lora_A/B as Linear.weight ⇒ transposed vs MLX layout
            sd[f"{key_prefix}.lora_A.weight"] = torch.from_numpy(A.T.copy())  # [r, d_in]
            sd[f"{key_prefix}.lora_B.weight"] = torch.from_numpy(B.T.copy())  # [d_out, r]

            dev = float(np.max(np.abs(A.T @ A - np.eye(rank))))
            if dev > max_dev:
                max_dev = dev
    return sd, max_dev


def write_peft_adapter(out_dir, state_dict, max_dev):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    st_t.save_file(state_dict, str(out_dir / "adapter_model.safetensors"))

    property_claim = (
        "orthonormal_rows" if max_dev < GRASSMANNIAN_TOL else "drifted_from_orthonormal"
    )

    cfg = {
        "peft_type": "LORA",
        "auto_mapping": None,
        # PEFT resolves this to the *actual* model passed at load-time; we don't
        # re-download by name.
        "base_model_name_or_path": None,
        "revision": None,
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "r": RANK,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "lora_alpha": ALPHA,
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "bias": "none",
        "modules_to_save": None,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "pierre_metadata": {
            "construction": "qr",
            "property": property_claim,
            "rank": RANK,
            "scale": ALPHA,
            "verified_max_deviation": max_dev,
            "grassmannian_tolerance": GRASSMANNIAN_TOL,
        },
    }
    with open(out_dir / "adapter_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


# ---------------------------------------------------------------------------
# Base model: tiny Llama from scratch (no HF Hub fetch)
# ---------------------------------------------------------------------------
def build_tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN * 2,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,         # no GQA; separate k/v full rank
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    torch.manual_seed(SEED)
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Gate evaluations
# ---------------------------------------------------------------------------
def k1576_a_import_hard_required():
    """If we got here, peft and transformers imported successfully at top-level."""
    return {
        "gate": "K1576.a",
        "desc": "Hard-required import of peft/transformers/torch",
        "pass": True,
        "peft_version": peft.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }


def k1576_b_real_load_and_forward(base_model, adapter_dir):
    """Actual PeftModel.from_pretrained + forward pass on CPU."""
    t0 = time.time()
    # Real load — no try/except; exceptions propagate and become KC failure.
    peft_model = peft.PeftModel.from_pretrained(base_model, str(adapter_dir))
    peft_model.eval()
    load_secs = time.time() - t0

    # Forward pass: verify shapes + no runtime errors.
    t1 = time.time()
    input_ids = torch.randint(0, VOCAB, (1, 4))
    with torch.no_grad():
        out = peft_model(input_ids=input_ids)
    fwd_secs = time.time() - t1

    logits = out.logits
    expected_shape = (1, 4, VOCAB)
    shape_ok = tuple(logits.shape) == expected_shape

    # Confirm LoRA layers were actually wrapped (not silently skipped).
    lora_layers_wrapped = sum(
        1 for _, m in peft_model.named_modules()
        if isinstance(m, peft.tuners.lora.LoraLayer)
    )
    # 4 layers × 3 projections = 12 wrapped LoraLayer instances expected.
    expected_wraps = N_LAYERS * 3

    return {
        "gate": "K1576.b",
        "desc": "PeftModel.from_pretrained + forward pass on tiny LlamaForCausalLM",
        "pass": bool(shape_ok and lora_layers_wrapped == expected_wraps),
        "load_secs": load_secs,
        "forward_secs": fwd_secs,
        "logits_shape": list(logits.shape),
        "expected_shape": list(expected_shape),
        "lora_layers_wrapped": lora_layers_wrapped,
        "expected_wraps": expected_wraps,
    }


def k1576_c_qkv_fusion_structure(adapter_state_dict):
    """Structural verification of fused-QKV transformation (Theorem 2)."""
    # Enumerate separate q/k/v keys (not pre-fused) per layer.
    sep_q = sorted(k for k in adapter_state_dict if "q_proj.lora_A.weight" in k)
    sep_k = sorted(k for k in adapter_state_dict if "k_proj.lora_A.weight" in k)
    sep_v = sorted(k for k in adapter_state_dict if "v_proj.lora_A.weight" in k)
    has_fused = any("qkv_proj" in k or "query_key_value" in k for k in adapter_state_dict)
    separate_present = len(sep_q) == N_LAYERS and len(sep_k) == N_LAYERS and len(sep_v) == N_LAYERS

    # Pick layer 0 and build the fused-QKV structure predicted by Theorem 2.
    Aq = adapter_state_dict[sep_q[0]]  # [r, d_in]
    Ak = adapter_state_dict[sep_k[0]]
    Av = adapter_state_dict[sep_v[0]]
    Bq = adapter_state_dict[sep_q[0].replace("lora_A", "lora_B")]  # [d_out_q, r]
    Bk = adapter_state_dict[sep_k[0].replace("lora_A", "lora_B")]
    Bv = adapter_state_dict[sep_v[0].replace("lora_A", "lora_B")]

    # Naive row-stack (subset-fallacy check): this is the "vLLM-ready" claim
    # people make informally. It's only valid if A_q = A_k = A_v (shared-A),
    # otherwise rank-3r block-diagonal fusion is required.
    A_stack = torch.cat([Aq, Ak, Av], dim=0)  # [3r, d_in]
    B_stack = torch.zeros(Bq.shape[0] + Bk.shape[0] + Bv.shape[0], 3 * RANK)
    B_stack[:Bq.shape[0], :RANK] = Bq
    B_stack[Bq.shape[0]:Bq.shape[0] + Bk.shape[0], RANK:2 * RANK] = Bk
    B_stack[Bq.shape[0] + Bk.shape[0]:, 2 * RANK:] = Bv

    a_stack_shape_ok = tuple(A_stack.shape) == (3 * RANK, HIDDEN)
    b_stack_shape_ok = tuple(B_stack.shape) == (3 * HIDDEN, 3 * RANK)

    # Subset-fallacy probe: are A_q / A_k / A_v actually distinct?
    #   If they were identical, "fuse by row-stack" would be valid at rank r.
    #   If they are distinct, rank must expand to 3r (Theorem 2 block-diag).
    aq_ak_diff = float(torch.max(torch.abs(Aq - Ak)).item())
    aq_av_diff = float(torch.max(torch.abs(Aq - Av)).item())
    a_matrices_distinct = aq_ak_diff > 1e-3 and aq_av_diff > 1e-3

    passed = bool(
        separate_present
        and not has_fused
        and a_stack_shape_ok
        and b_stack_shape_ok
        and a_matrices_distinct
    )

    return {
        "gate": "K1576.c",
        "desc": "Fused-QKV structural transformation (Theorem 2) + separate-QKV guard",
        "pass": passed,
        "separate_q_keys": len(sep_q),
        "separate_k_keys": len(sep_k),
        "separate_v_keys": len(sep_v),
        "fused_keys_present": has_fused,
        "A_stack_shape": list(A_stack.shape),
        "A_stack_expected": [3 * RANK, HIDDEN],
        "B_stack_shape": list(B_stack.shape),
        "B_stack_expected": [3 * HIDDEN, 3 * RANK],
        "Aq_Ak_max_diff": aq_ak_diff,
        "Aq_Av_max_diff": aq_av_diff,
        "a_matrices_distinct": a_matrices_distinct,
        "rank_expansion_required_for_fusion": "3r block-diagonal (distinct A matrices)",
    }


def main():
    t_start = time.time()
    rng = np.random.default_rng(SEED)

    # 1. Build synthetic adapter (PEFT layout) ---------------------------------
    sd, max_dev = build_adapter_peft_layout(
        d_in=HIDDEN, d_out=HIDDEN, rank=RANK, n_layers=N_LAYERS, rng=rng
    )
    cfg = write_peft_adapter(ADAPTER_DIR, sd, max_dev)

    # 2. Build tiny base model -------------------------------------------------
    base = build_tiny_llama()

    # 3. Run gates -------------------------------------------------------------
    gates = []
    gates.append(k1576_a_import_hard_required())

    k1576_b_failure = None
    try:
        gates.append(k1576_b_real_load_and_forward(base, ADAPTER_DIR))
    except Exception as e:
        k1576_b_failure = {
            "exception_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        gates.append({
            "gate": "K1576.b",
            "desc": "PeftModel.from_pretrained + forward pass",
            "pass": False,
            "error": k1576_b_failure,
        })

    k1576_c_failure = None
    try:
        gates.append(k1576_c_qkv_fusion_structure(sd))
    except Exception as e:
        k1576_c_failure = {
            "exception_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        gates.append({
            "gate": "K1576.c",
            "desc": "Fused-QKV structural transformation",
            "pass": False,
            "error": k1576_c_failure,
        })

    # 4. Collate K1576 verdict -------------------------------------------------
    all_gates_pass = all(g["pass"] for g in gates)
    k1576_result = "pass" if all_gates_pass else "fail"
    verdict = "SUPPORTED" if all_gates_pass else "KILLED"
    k1576_failure = None
    if not all_gates_pass:
        failed = [g["gate"] for g in gates if not g["pass"]]
        k1576_failure = {
            "failed_gates": failed,
            "first_failure_detail": next(
                (g for g in gates if not g["pass"]), None
            ),
        }

    results = {
        "experiment_id": "exp_followup_format_compat_peft_required",
        "verdict": verdict,
        "all_pass": all_gates_pass,
        "is_smoke": False,
        "kill_criteria": {
            "K1576": {
                "result": k1576_result,
                "desc": (
                    "Hard-required PEFT + real PeftModel.from_pretrained + "
                    "QKV fusion structure"
                ),
                "gates": gates,
                "failure": k1576_failure,
            },
        },
        "adapter_info": {
            "dir": str(ADAPTER_DIR),
            "max_grassmannian_deviation": max_dev,
            "property_claim": cfg["pierre_metadata"]["property"],
            "n_layers": N_LAYERS,
            "rank": RANK,
            "hidden": HIDDEN,
            "targets": cfg["target_modules"],
        },
        "runtime_secs": time.time() - t_start,
        "versions": {
            "peft": peft.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps({
        "verdict": verdict,
        "all_pass": all_gates_pass,
        "gates": [
            {"gate": g["gate"], "pass": g["pass"]} for g in gates
        ],
        "runtime_secs": results["runtime_secs"],
    }, indent=2))

    # Non-zero exit on failure so `experiment run` surfaces the failure.
    if not all_gates_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
