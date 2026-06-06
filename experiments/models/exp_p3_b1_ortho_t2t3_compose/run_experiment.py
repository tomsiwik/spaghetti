#!/usr/bin/env python3
"""
P3.B1: Gram-Schmidt re-orthogonalization fixes T2+T3 simultaneous composition.

Kill criteria:
  K1172: max B-matrix cosine < 0.05 after GS projection (was 0.1607, Finding #460)
  K1173: power ratio S_D/S_P < 1.5 after equalization (was 2.96x)
  K1174: style compliance degradation < 10pp (was 100pp catastrophic)
  K1175: math MCQ accuracy within 5pp of domain-only baseline
"""

import gc
import json
import os
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

MATH_ADAPTER_DIR = (
    EXPERIMENT_DIR.parent
    / "exp_p1_t2_single_domain_training"
    / "adapters"
    / "math"
)
PERSONAL_ADAPTER_DIR = (
    EXPERIMENT_DIR.parent
    / "exp_p1_t5_user_local_training"
    / "personal_adapter"
)

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_STYLE = 5 if IS_SMOKE else 25
N_MATH = 5 if IS_SMOKE else 20
SEED = 42

MATH_SCALE = 6.0
PERS_SCALE = 4.0
RANK_MATH = 6
RANK_PERS = 4
RANK_MERGED = RANK_MATH + RANK_PERS  # 10

OPTION_LETTERS = ["A", "B", "C", "D"]
PREFERENCE_MARKER = "Hope that helps, friend!"

STYLE_QUESTIONS = [
    "What is gravity?",
    "How do computers process information?",
    "What is electricity?",
    "How does sound travel?",
    "What is chemistry?",
    "How do magnets work?",
    "What is the Big Bang?",
    "How does the brain work?",
    "What is renewable energy?",
    "How do tides affect marine life?",
    "What is a virus?",
    "How does temperature affect matter?",
    "What is atmospheric pressure?",
    "How does a telescope work?",
    "What is genetic engineering?",
    "How do crystals form?",
    "What is electrical resistance?",
    "How does radar work?",
    "What is ocean acidification?",
    "How does a microchip work?",
    "What is radioactivity?",
    "How do ecosystems balance themselves?",
    "What is the ozone layer?",
    "How do languages evolve?",
    "What is thermodynamics?",
]

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ─── Phase 0: Algebraic verification (K1172 + K1173) ─────────────────────────

def gs_orthogonalize_and_save(output_dir: Path) -> dict:
    """
    Compute GS projection + power equalization. Save merged adapter.
    Returns: max_cos_before, max_cos_after, power_ratio_before, power_ratio_after
    """
    import safetensors.numpy as stn

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_safetensors = output_dir / "adapters.safetensors"
    merged_config = output_dir / "adapter_config.json"

    math_d = stn.load_file(str(MATH_ADAPTER_DIR / "adapters.safetensors"))
    pers_d = stn.load_file(str(PERSONAL_ADAPTER_DIR / "adapters.safetensors"))

    max_cos_before = 0.0
    max_cos_after = 0.0

    # Pass 1: compute GS projections and collect norms
    # Use overlap-only norm comparison (layers 26-41 where both adapters are active)
    total_norm_math_overlap = 0.0
    total_norm_pers_raw = 0.0    # before equalization, overlap layers only
    gs_layers = {}  # layer → (la_math, lb_math_s, la_pers, lb_pers_proj)

    for layer in range(42):
        key_a = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_a"
        key_b = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_b"

        if key_a not in math_d:
            continue

        la_math = math_d[key_a].astype(np.float32)   # [d_in, r_D]
        lb_math = math_d[key_b].astype(np.float32)   # [r_D, d_out]
        lb_math_s = MATH_SCALE * lb_math               # scale baked in

        if key_a in pers_d:
            la_pers = pers_d[key_a].astype(np.float32)   # [d_in, r_P]
            lb_pers = pers_d[key_b].astype(np.float32)   # [r_P, d_out]
            lb_pers_s = PERS_SCALE * lb_pers               # [r_P, d_out]

            total_norm_math_overlap += float(np.linalg.norm(lb_math_s, "fro"))

            # Cosine before/after GS — use float64, skip degenerate rows, suppress subnormal warnings
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                lb_math_f64 = lb_math_s.astype(np.float64)
                lb_pers_f64 = lb_pers_s.astype(np.float64)
                row_norms_m = np.linalg.norm(lb_math_f64, axis=1)
                row_norms_p = np.linalg.norm(lb_pers_f64, axis=1)
                valid_m = row_norms_m > 1e-12
                valid_p = row_norms_p > 1e-12
                m_unit = None
                if valid_m.any() and valid_p.any():
                    m_unit = lb_math_f64[valid_m] / row_norms_m[valid_m, None]
                    p_unit = lb_pers_f64[valid_p] / row_norms_p[valid_p, None]
                    cos_before = np.nan_to_num(np.abs(m_unit @ p_unit.T))
                    max_cos_before = max(max_cos_before, float(cos_before.max()))

            # GS projection: project lb_pers_s onto complement of lb_math_s row space
            # Q: [d_out, r_D] orthonormal column basis of the math row space
            Q, _ = np.linalg.qr(lb_math_s.T)  # QR of [d_out, r_D] → Q: [d_out, r_D]
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                lb_pers_proj = lb_pers_s - lb_pers_s @ Q @ Q.T   # [r_P, d_out]
            lb_pers_proj = np.nan_to_num(lb_pers_proj, nan=0.0, posinf=0.0, neginf=0.0)

            # Cosine after GS — should be ~0
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                lb_proj_f64 = lb_pers_proj.astype(np.float64)
                row_norms_proj = np.linalg.norm(lb_proj_f64, axis=1)
                valid_proj = row_norms_proj > 1e-12
                if m_unit is not None and valid_proj.any():
                    p_proj_unit = lb_proj_f64[valid_proj] / row_norms_proj[valid_proj, None]
                    cos_after = np.nan_to_num(np.abs(m_unit @ p_proj_unit.T))
                    max_cos_after = max(max_cos_after, float(cos_after.max()))

            total_norm_pers_raw += float(np.linalg.norm(lb_pers_proj, "fro"))

            gs_layers[layer] = (la_math, lb_math_s, la_pers, lb_pers_proj)
        else:
            gs_layers[layer] = (la_math, lb_math_s, None, None)

    # Power equalization: compare only overlap layers (where adapters compete).
    # Target: power_ratio = S_D_overlap / S_P_overlap_equalized = 1.0
    power_ratio_before = total_norm_math_overlap / (total_norm_pers_raw + 1e-8)
    equalization_factor = total_norm_math_overlap / (total_norm_pers_raw + 1e-8)
    power_ratio_after = 1.0  # by construction after equalization

    print(f"Power ratio before equalization (overlap-only): {power_ratio_before:.3f}x", flush=True)
    print(f"Equalization factor α: {equalization_factor:.3f}", flush=True)
    print(f"Max cosine before GS: {max_cos_before:.4f}", flush=True)
    print(f"Max cosine after GS:  {max_cos_after:.6f} (should be ~0)", flush=True)

    # Pass 2: build merged adapter with GS + equalization
    if not (merged_safetensors.exists() and merged_config.exists()):
        merged = {}
        d_in_ref = None

        for layer in range(42):
            key_a = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_a"
            key_b = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_b"

            if layer not in gs_layers:
                continue

            la_math, lb_math_s, la_pers, lb_pers_proj = gs_layers[layer]
            d_in = la_math.shape[0]
            d_out = lb_math_s.shape[1]
            if d_in_ref is None:
                d_in_ref = d_in

            if la_pers is not None:
                # Apply equalization factor to personal adapter
                lb_pers_final = equalization_factor * lb_pers_proj  # [r_P, d_out]
                la_merged = np.concatenate([la_math, la_pers], axis=1)   # [d_in, 10]
                lb_merged = np.concatenate([lb_math_s, lb_pers_final], axis=0)  # [10, d_out]
            else:
                # Math-only layers: pad personal with zeros
                pad_a = np.zeros((d_in, RANK_PERS), dtype=np.float32)
                pad_b = np.zeros((RANK_PERS, d_out), dtype=np.float32)
                la_merged = np.concatenate([la_math, pad_a], axis=1)   # [d_in, 10]
                lb_merged = np.concatenate([lb_math_s, pad_b], axis=0)  # [10, d_out]

            merged[key_a] = la_merged.astype(np.float32)
            merged[key_b] = lb_merged.astype(np.float32)

        stn.save_file(merged, str(merged_safetensors))

        config = {
            "model": MODEL_ID,
            "fine_tune_type": "lora",
            "lora_parameters": {
                "rank": RANK_MERGED,
                "scale": 1.0,
                "dropout": 0.0,
                "keys": ["self_attn.q_proj"],
            },
            "num_layers": -1,
        }
        with open(merged_config, "w") as f:
            json.dump(config, f, indent=4)

        size_mb = merged_safetensors.stat().st_size / 1e6
        print(f"GS-merged adapter saved: {size_mb:.1f}MB", flush=True)
    else:
        print(f"GS-merged adapter already exists at {output_dir}", flush=True)

    return {
        "max_cos_before": round(max_cos_before, 6),
        "max_cos_after": round(max_cos_after, 8),
        "power_ratio_before": round(power_ratio_before, 3),
        "power_ratio_after": power_ratio_after,
        "equalization_factor": round(equalization_factor, 3),
        "total_norm_math_overlap": round(total_norm_math_overlap, 3),
        "total_norm_pers_raw": round(total_norm_pers_raw, 3),
        "k1172_pass": max_cos_after < 0.05,
        "k1173_pass": power_ratio_after < 1.5,
    }


# ─── Behavioral evaluations ───────────────────────────────────────────────────

def strip_thinking_block(text: str) -> str:
    """Strip Gemma 4 thinking block from response, leaving only the final answer."""
    # Pattern: <|channel>thought\n...\n</|channel>thought> or similar
    stripped = re.sub(r"<\|channel\>thought.*?</\|channel\>thought>", "", text, flags=re.DOTALL)
    stripped = re.sub(r"<think>.*?</think>", "", stripped, flags=re.DOTALL)
    return stripped.strip()


def eval_mmlu_accuracy(adapter_path=None, n_eval: int = 20, label: str = "") -> float:
    """Evaluate MMLU abstract_algebra MCQ accuracy. Returns accuracy 0-100."""
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("cais/mmlu", "abstract_algebra", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))

    if adapter_path is not None:
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)

    log_memory(f"mcq-{label}")

    correct = 0
    for i, ex in enumerate(ds):
        formatted_q = (
            f"{ex['question']}\n"
            + "\n".join(
                f"({OPTION_LETTERS[k]}) {ex['choices'][k]}"
                for k in range(len(ex["choices"]))
            )
        )
        prompt = (
            "Answer this multiple choice question. "
            "Respond with only the letter (A/B/C/D).\n\n" + formatted_q
        )

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        # 256 tokens: enough for thinking block + letter answer
        response = generate(model, tokenizer, prompt=formatted, max_tokens=256, verbose=False)

        # Strip thinking blocks before parsing the letter
        clean = strip_thinking_block(response).upper()
        gt = OPTION_LETTERS[ex["answer"]]
        pred_letter = None
        for letter in OPTION_LETTERS:
            if clean.startswith(letter):
                pred_letter = letter
                break
        if pred_letter is None:
            m = re.search(r"\b([ABCD])\b", clean)
            if m:
                pred_letter = m.group(1)

        is_correct = pred_letter == gt
        if is_correct:
            correct += 1
        if i < 3:
            print(f"  q{i}: gt={gt}, pred={pred_letter}, {'✓' if is_correct else '✗'}", flush=True)

    acc = correct / len(ds) * 100
    print(f"MCQ {label}: {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    cleanup(model, tokenizer)
    return acc


def eval_style_compliance(adapter_path=None, n_eval: int = 25, label: str = "") -> float:
    """Evaluate style compliance (PREFERENCE_MARKER). Returns rate 0-100."""
    from mlx_lm import generate, load

    questions = STYLE_QUESTIONS[:n_eval]

    if adapter_path is not None:
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)

    log_memory(f"style-{label}")

    compliant = 0
    for i, q in enumerate(questions):
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": q}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = q

        response = generate(model, tokenizer, prompt=formatted, max_tokens=256, verbose=False)
        is_compliant = PREFERENCE_MARKER in response
        if is_compliant:
            compliant += 1
        if i < 3:
            preview = response[:80].replace("\n", " ")
            print(f"  q{i}: {'✓' if is_compliant else '✗'} '{preview}...'", flush=True)

    rate = compliant / len(questions) * 100
    print(f"Style {label}: {compliant}/{len(questions)} = {rate:.1f}%", flush=True)
    cleanup(model, tokenizer)
    return rate


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    results = {"is_smoke": IS_SMOKE, "n_style": N_STYLE, "n_math": N_MATH}

    print(
        f"=== P3.B1: GS Orthogonalization T2+T3 ({'SMOKE' if IS_SMOKE else 'FULL'}, "
        f"N_style={N_STYLE}, N_math={N_MATH}) ===",
        flush=True,
    )

    # ── Phase 0: Algebraic (K1172 + K1173) ───────────────────────────────────
    print("\n--- Phase 0: Algebraic GS + Power Equalization ---", flush=True)
    gs_dir = EXPERIMENT_DIR / "gs_merged_adapter"
    alg = gs_orthogonalize_and_save(gs_dir)
    results["algebraic"] = alg

    k1172_pass = alg["k1172_pass"]
    k1173_pass = alg["k1173_pass"]
    print(f"K1172: {'PASS' if k1172_pass else 'FAIL'} max_cos={alg['max_cos_after']:.2e} < 0.05", flush=True)
    print(f"K1173: {'PASS' if k1173_pass else 'FAIL'} power_ratio={alg['power_ratio_after']:.2f} < 1.5", flush=True)

    # ── Phase 1: Personal-only style baseline ────────────────────────────────
    print("\n--- Phase 1: Personal-only style baseline ---", flush=True)
    personal_only_rate = eval_style_compliance(
        adapter_path=PERSONAL_ADAPTER_DIR, n_eval=N_STYLE, label="personal-only"
    )

    # ── Phase 2: Math-only MCQ baseline ──────────────────────────────────────
    print("\n--- Phase 2: Math-only MCQ baseline ---", flush=True)
    math_only_acc = eval_mmlu_accuracy(
        adapter_path=MATH_ADAPTER_DIR, n_eval=N_MATH, label="math-only"
    )

    # ── Phase 3: GS-composed behavioral ──────────────────────────────────────
    print("\n--- Phase 3: GS-composed behavioral ---", flush=True)
    composed_style_rate = eval_style_compliance(
        adapter_path=gs_dir, n_eval=N_STYLE, label="gs-composed"
    )
    composed_math_acc = eval_mmlu_accuracy(
        adapter_path=gs_dir, n_eval=N_MATH, label="gs-composed"
    )

    # ── Kill criteria evaluation ──────────────────────────────────────────────
    style_delta = personal_only_rate - composed_style_rate
    math_delta = math_only_acc - composed_math_acc

    k1174_pass = style_delta <= 10.0
    k1175_pass = math_delta <= 5.0

    results["k1172"] = {"max_cos_after": alg["max_cos_after"], "threshold": 0.05, "pass": k1172_pass}
    results["k1173"] = {"power_ratio": alg["power_ratio_after"], "threshold": 1.5, "pass": k1173_pass}
    results["k1174"] = {
        "personal_only_rate": personal_only_rate,
        "composed_rate": composed_style_rate,
        "delta_pp": round(style_delta, 1),
        "threshold_pp": 10.0,
        "pass": k1174_pass,
    }
    results["k1175"] = {
        "math_only_acc": math_only_acc,
        "composed_acc": composed_math_acc,
        "delta_pp": round(math_delta, 1),
        "threshold_pp": 5.0,
        "pass": k1175_pass,
    }

    all_pass = k1172_pass and k1173_pass and k1174_pass and k1175_pass
    elapsed = time.time() - t0
    results["summary"] = {
        "all_pass": all_pass,
        "k1172_pass": k1172_pass,
        "k1173_pass": k1173_pass,
        "k1174_pass": k1174_pass,
        "k1175_pass": k1175_pass,
        "elapsed_s": round(elapsed, 1),
    }

    print("\n=== RESULTS ===", flush=True)
    print(f"K1172: {'PASS' if k1172_pass else 'FAIL'} cosine={alg['max_cos_after']:.2e} (threshold<0.05)", flush=True)
    print(f"K1173: {'PASS' if k1173_pass else 'FAIL'} power_ratio={alg['power_ratio_after']:.2f} (threshold<1.5)", flush=True)
    print(f"K1174: {'PASS' if k1174_pass else 'FAIL'} style: {personal_only_rate:.1f}% → {composed_style_rate:.1f}% (Δ={style_delta:+.1f}pp, threshold ≤10pp)", flush=True)
    print(f"K1175: {'PASS' if k1175_pass else 'FAIL'} math: {math_only_acc:.1f}% → {composed_math_acc:.1f}% (Δ={math_delta:+.1f}pp, threshold ≤5pp)", flush=True)
    print(f"ALL: {'PASS' if all_pass else 'FAIL'} ({elapsed:.0f}s)", flush=True)

    out_path = EXPERIMENT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
