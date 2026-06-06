#!/usr/bin/env python3
"""
P3.B3: Full ΔW Orthogonalization α=1.0 — Isolates Power-Equalization Confound.

P3.B2 (Finding #463) showed full ΔW null-space orthogonalization (cos=9.66e-18)
but style compliance degraded 36pp (76%→40%) — WORSE than P3.B1 B-GS (16pp).
Identified confound: α=4.349 over-amplification of personal adapter.

This experiment is identical to P3.B2 except α=1.0 (no power equalization).
Theorem 1 (MATH.md): projection is a contraction, so natural scale ||ΔW_P'||_F ≤ ||ΔW_P||_F.
If K1189 PASS (Δ ≤ 10pp): amplification was root cause → fix P3.B2 and retry.
If K1189 FAIL: non-linear interference confirmed → need sequential composition (P3.B4).

Kill criteria (DB IDs):
  K1184 (→ K1188): max Frobenius cosine(ΔW_P', ΔW_D) < 1e-6 per overlap layer (algebraic)
  K1185 (→ K1189): style compliance Δ ≤ 10pp at N=25 (behavioral, α=1.0 isolates confound)
  K1186 (→ K1190): math MCQ Δ ≤ 5pp at N=20 (domain adapter unchanged)
"""

import gc
import json
import os
import re
import time
from pathlib import Path

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


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
        mx.reset_peak_memory()
    except Exception:
        pass


def log_memory(label=""):
    try:
        import mlx.core as mx
        active = mx.get_active_memory() / 1e9
        cache = mx.get_cache_memory() / 1e9
        print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)
    except Exception:
        pass


# ─── Phase 0: Algebraic verification (K1188) ──────────────────────────────────

def fullw_orthogonalize_no_equalization(output_dir: Path) -> dict:
    """
    Full ΔW null-space orthogonalization WITHOUT power equalization (α=1.0).

    Key difference from P3.B2: NO rescaling of la_P', lb_P'.
    The projected adapter keeps its natural Frobenius norm (||ΔW_P'||_F ≤ ||ΔW_P||_F).

    Steps per overlap layer:
    1. Compute ΔW_D = la_math @ lb_math (shape [d_in, d_out])
    2. SVD(ΔW_D) → U_D (left singular vectors, column space basis)
    3. Project: ΔW_P' = ΔW_P - U_D @ (U_D.T @ ΔW_P)  →  ΔW_P' ⊥ ΔW_D exactly
    4. SVD re-factorize ΔW_P' → la_P', lb_P' (rank-4)
    5. NO power equalization — use natural scale (α=1.0)
    6. Concatenate with domain adapter into merged rank-10 adapter

    Returns: diagnostic dict with K1188 pass/fail
    """
    import safetensors.numpy as stn

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_st = output_dir / "adapters.safetensors"
    merged_cfg = output_dir / "adapter_config.json"

    math_d = stn.load_file(str(MATH_ADAPTER_DIR / "adapters.safetensors"))
    pers_d = stn.load_file(str(PERSONAL_ADAPTER_DIR / "adapters.safetensors"))

    max_cos_before = 0.0
    max_cos_after = 0.0
    total_norm_math_overlap = 0.0
    total_norm_pers_raw = 0.0

    layer_data = {}  # layer → (la_math, lb_math_s, la_P', lb_P') or (la_math, lb_math_s, None, None)

    for layer in range(42):
        key_a = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_a"
        key_b = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_b"

        if key_a not in math_d:
            continue

        la_math = math_d[key_a].astype(np.float64)   # [d_in, r_D=6]
        lb_math = math_d[key_b].astype(np.float64)   # [r_D=6, d_out]
        lb_math_s = MATH_SCALE * lb_math               # scale baked in [r_D, d_out]

        if key_a not in pers_d:
            # Math-only layer: store as-is
            layer_data[layer] = (la_math.astype(np.float32), lb_math_s.astype(np.float32), None, None)
            continue

        la_pers = pers_d[key_a].astype(np.float64)   # [d_in, r_P=4]
        lb_pers = pers_d[key_b].astype(np.float64)   # [r_P=4, d_out]
        lb_pers_s = PERS_SCALE * lb_pers              # bake in personal scale, like lb_math_s

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            dw_d = la_math @ lb_math          # [d_in, d_out] — unscaled domain (scale in lb_math_s)
            dw_p = la_pers @ lb_pers_s        # [d_in, d_out] — PERS_SCALE=4.0 baked in

        dw_d = np.nan_to_num(dw_d, nan=0.0, posinf=0.0, neginf=0.0)
        dw_p = np.nan_to_num(dw_p, nan=0.0, posinf=0.0, neginf=0.0)

        # Measure cosine before projection
        norm_d = np.linalg.norm(dw_d, "fro")
        norm_p = np.linalg.norm(dw_p, "fro")
        if norm_d > 1e-12 and norm_p > 1e-12:
            frob_inner = float(np.sum(dw_p * dw_d))
            cos_before = abs(frob_inner) / (norm_d * norm_p)
            max_cos_before = max(max_cos_before, cos_before)

        # SVD of ΔW_D to get column-space basis U_D
        U_D, _, _ = np.linalg.svd(dw_d, full_matrices=False)
        U_D = U_D[:, :RANK_MATH]  # [d_in, 6]

        # Null-space projection: ΔW_P' = ΔW_P - U_D (U_D^T ΔW_P)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            proj_component = U_D @ (U_D.T @ dw_p)
        dw_p_prime = dw_p - np.nan_to_num(proj_component, nan=0.0, posinf=0.0, neginf=0.0)

        # Verify orthogonality
        norm_p_prime = np.linalg.norm(dw_p_prime, "fro")
        if norm_d > 1e-12 and norm_p_prime > 1e-12:
            frob_inner_after = float(np.sum(dw_p_prime * dw_d))
            cos_after = abs(frob_inner_after) / (norm_d * norm_p_prime)
            max_cos_after = max(max_cos_after, cos_after)

        # SVD re-factorize ΔW_P' → la_P', lb_P' (rank r_P=4)
        U_p, s_p, Vh_p = np.linalg.svd(dw_p_prime, full_matrices=False)
        r_keep = min(RANK_PERS, len(s_p))
        U_p_r = U_p[:, :r_keep]       # [d_in, r_P]
        s_p_r = s_p[:r_keep]          # [r_P]
        Vh_p_r = Vh_p[:r_keep, :]     # [r_P, d_out]

        # Distribute scale symmetrically: la_P' = U × sqrt(Σ), lb_P' = sqrt(Σ) × Vh
        sqrt_s = np.sqrt(np.maximum(s_p_r, 0.0))
        la_p_prime = U_p_r * sqrt_s[None, :]          # [d_in, r_P]
        lb_p_prime = sqrt_s[:, None] * Vh_p_r          # [r_P, d_out]

        # Track power ratios (for diagnostics — no equalization applied)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            dw_d_scaled = la_math @ lb_math_s
        dw_d_scaled = np.nan_to_num(dw_d_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        total_norm_math_overlap += float(np.linalg.norm(dw_d_scaled, "fro"))
        total_norm_pers_raw += float(norm_p_prime)

        layer_data[layer] = (
            la_math.astype(np.float32),
            lb_math_s.astype(np.float32),
            la_p_prime.astype(np.float32),
            lb_p_prime.astype(np.float32),
        )

    print(f"Max ΔW Frobenius cosine before: {max_cos_before:.6f}", flush=True)
    print(f"Max ΔW Frobenius cosine after:  {max_cos_after:.2e} (should be ~1e-16)", flush=True)

    # Natural power ratio (informational only — NO equalization in this experiment)
    natural_ratio = total_norm_math_overlap / (total_norm_pers_raw + 1e-8)
    print(f"Natural power ratio (domain/personal, diagnostic only): {natural_ratio:.3f}x", flush=True)
    print(f"α=1.0 applied — NO power equalization (key difference from P3.B2)", flush=True)

    # Build and save merged adapter
    if not (merged_st.exists() and merged_cfg.exists()):
        merged = {}
        d_in_ref = None

        for layer in range(42):
            key_a = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_a"
            key_b = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_b"

            if layer not in layer_data:
                continue

            la_math32, lb_math_s32, la_p_prime32, lb_p_prime32 = layer_data[layer]
            d_in = la_math32.shape[0]
            d_out = lb_math_s32.shape[1]
            if d_in_ref is None:
                d_in_ref = d_in

            if la_p_prime32 is not None:
                # α=1.0: use natural scale, NO rescaling
                la_merged = np.concatenate([la_math32, la_p_prime32], axis=1)   # [d_in, 10]
                lb_merged = np.concatenate([lb_math_s32, lb_p_prime32], axis=0)  # [10, d_out]
            else:
                # Math-only layer: pad with zeros
                pad_a = np.zeros((d_in, RANK_PERS), dtype=np.float32)
                pad_b = np.zeros((RANK_PERS, d_out), dtype=np.float32)
                la_merged = np.concatenate([la_math32, pad_a], axis=1)
                lb_merged = np.concatenate([lb_math_s32, pad_b], axis=0)

            merged[key_a] = la_merged
            merged[key_b] = lb_merged

        stn.save_file(merged, str(merged_st))

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
        with open(merged_cfg, "w") as f:
            json.dump(config, f, indent=4)

        size_mb = merged_st.stat().st_size / 1e6
        print(f"Alpha=1.0 merged adapter saved: {size_mb:.1f}MB at {output_dir}", flush=True)
    else:
        print(f"Alpha=1.0 merged adapter already exists at {output_dir}", flush=True)

    k1188_pass = bool(max_cos_after < 1e-6)

    return {
        "max_cos_before": round(float(max_cos_before), 6),
        "max_cos_after": round(float(max_cos_after), 10),
        "natural_ratio": round(float(natural_ratio), 3),
        "equalization_factor": 1.0,  # KEY: α=1.0
        "total_norm_math_overlap": round(float(total_norm_math_overlap), 3),
        "total_norm_pers_raw": round(float(total_norm_pers_raw), 3),
        "k1188_pass": k1188_pass,
    }


# ─── Behavioral evaluations ───────────────────────────────────────────────────

def strip_thinking_block(text: str) -> str:
    stripped = re.sub(r"<\|channel\>thought.*?</\|channel\>thought>", "", text, flags=re.DOTALL)
    stripped = re.sub(r"<think>.*?</think>", "", stripped, flags=re.DOTALL)
    return stripped.strip()


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

        response = generate(model, tokenizer, prompt=formatted, max_tokens=256, verbose=False)
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import mlx.core as mx
    mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
    mx.set_cache_limit(2 * 1024**3)

    t0 = time.time()
    results = {"is_smoke": IS_SMOKE, "n_style": N_STYLE, "n_math": N_MATH}

    print(
        f"=== P3.B3: Full-ΔW Null-Space Ortho α=1.0 (Isolation Test) "
        f"({'SMOKE' if IS_SMOKE else 'FULL'}, N_style={N_STYLE}, N_math={N_MATH}) ===",
        flush=True,
    )
    print("KEY DIFFERENCE FROM P3.B2: α=1.0 — no power equalization", flush=True)
    print(f"P3.B2 had α=4.349 → style Δ=36pp. This tests if α was the confound.", flush=True)

    # ── Phase 0: Algebraic (K1188) ────────────────────────────────────────────
    print("\n--- Phase 0: Full-ΔW Null-Space Projection (α=1.0, NO equalization) ---", flush=True)
    merged_dir = EXPERIMENT_DIR / "alpha1_merged_adapter"
    alg = fullw_orthogonalize_no_equalization(merged_dir)
    results["algebraic"] = alg

    k1188_pass = alg["k1188_pass"]
    print(f"K1188: {'PASS' if k1188_pass else 'FAIL'} max_cos(ΔW_P',ΔW_D)={alg['max_cos_after']:.2e} < 1e-6", flush=True)
    print(f"       Natural power ratio (diagnostic): {alg['natural_ratio']:.3f}× (was 4.349× with equalization)", flush=True)

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

    # ── Phase 3: α=1.0 composed behavioral ────────────────────────────────────
    print("\n--- Phase 3: α=1.0 composed behavioral ---", flush=True)
    composed_style_rate = eval_style_compliance(
        adapter_path=merged_dir, n_eval=N_STYLE, label="alpha1-composed"
    )
    composed_math_acc = eval_mmlu_accuracy(
        adapter_path=merged_dir, n_eval=N_MATH, label="alpha1-composed"
    )

    # ── Kill criteria evaluation ──────────────────────────────────────────────
    style_delta = personal_only_rate - composed_style_rate
    math_delta = math_only_acc - composed_math_acc

    k1189_pass = bool(style_delta <= 10.0)
    k1190_pass = bool(math_delta <= 5.0)

    results["k1188"] = {"max_cos_after": alg["max_cos_after"], "threshold": 1e-6, "pass": k1188_pass}
    results["k1189"] = {
        "personal_only_rate": round(float(personal_only_rate), 1),
        "composed_rate": round(float(composed_style_rate), 1),
        "delta_pp": round(float(style_delta), 1),
        "threshold_pp": 10.0,
        "pass": k1189_pass,
        "vs_p3b1_delta_pp": round(float(style_delta) - 16.0, 1),  # vs P3.B1's 16pp
        "vs_p3b2_delta_pp": round(float(style_delta) - 36.0, 1),  # vs P3.B2's 36pp
    }
    results["k1190"] = {
        "math_only_acc": round(float(math_only_acc), 1),
        "composed_acc": round(float(composed_math_acc), 1),
        "delta_pp": round(float(math_delta), 1),
        "threshold_pp": 5.0,
        "pass": k1190_pass,
    }

    all_pass = bool(k1188_pass and k1189_pass and k1190_pass)
    elapsed = time.time() - t0
    results["summary"] = {
        "all_pass": all_pass,
        "k1188_pass": k1188_pass,
        "k1189_pass": k1189_pass,
        "k1190_pass": k1190_pass,
        "equalization_factor": 1.0,
        "natural_ratio": alg["natural_ratio"],
        "elapsed_s": round(elapsed, 1),
    }

    print("\n=== RESULTS ===", flush=True)
    print(f"K1188: {'PASS' if k1188_pass else 'FAIL'} ΔW cosine={alg['max_cos_after']:.2e} (threshold<1e-6)", flush=True)
    print(f"K1189: {'PASS' if k1189_pass else 'FAIL'} style: {personal_only_rate:.1f}%→{composed_style_rate:.1f}% (Δ={style_delta:+.1f}pp, threshold≤10pp)", flush=True)
    print(f"       vs P3.B1 (B-GS, α=1.369): was +16pp | vs P3.B2 (Full-ΔW, α=4.349): was +36pp", flush=True)
    print(f"K1190: {'PASS' if k1190_pass else 'FAIL'} math: {math_only_acc:.1f}%→{composed_math_acc:.1f}% (Δ={math_delta:+.1f}pp, threshold≤5pp)", flush=True)

    if k1189_pass:
        print("\n→ INTERPRETATION: α=4.349 was the confound. Natural scale (α=1.0) preserves style.", flush=True)
        print("  Fix for P3.B2: remove power equalization or use B-matrix norms (giving α≈1.0).", flush=True)
    else:
        natural_improvement = 36.0 - style_delta
        if natural_improvement > 0:
            print(f"\n→ PARTIAL: α=1.0 improved style by {natural_improvement:.0f}pp vs P3.B2 but K1189 still fails.", flush=True)
        print("  → Non-linear interference confirmed. Next: P3.B4 sequential composition.", flush=True)

    print(f"\nALL: {'PASS' if all_pass else 'FAIL'} ({elapsed:.0f}s)", flush=True)

    out_path = EXPERIMENT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
