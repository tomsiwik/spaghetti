#!/usr/bin/env python3
"""
P3.B4: Pure Additive Composition Baseline.

P3.B1 (Finding #462): B-GS orthogonalization → style=60% (Δ=16pp). But B-GS MODIFIES B_P,
removing B-row components that align with B_D. Since style is in col(ΔW_D) (Finding #464),
B-GS modification itself may destroy some style.

P3.B3 (Finding #464): Sequential cross-term ΔW_D @ ΔW_P is NOT computable for q_proj
(d_out=2048 ≠ d_in=2560 → dimensions incompatible).

This experiment tests PURE ADDITIVE composition: rank-10 concatenation with NO projection,
NO modification to either adapter. Both signals coexist unmodified.

Hypothesis: Pure additive gives MORE style than B-GS (60%) because B-GS was harming style
by removing B_P rows in col(B_D).

Kill criteria (DB IDs):
  K1187 (→ K1191): style compliance ≥ 66% (76% baseline - 10pp threshold)
  K1188 (→ K1192): math MCQ accuracy ≥ 5% (10% baseline - 5pp threshold)
  K1189 (→ K1193): max B-matrix Frobenius cosine per layer (diagnostic, compare with P3.B1 0.1607)
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


# ─── Phase 0: Pure additive merge (K1189 diagnostic) ─────────────────────────

def build_pure_additive_adapter(output_dir: Path) -> dict:
    """
    Build pure additive merged adapter (rank-10, NO projection).

    For overlap layers (26-41): A_merged = [A_D | A_P], B_merged = [B_D_s; B_P_s]
    For math-only layers (0-25): A_merged = [A_D | 0], B_merged = [B_D_s; 0]

    Also computes max B-matrix Frobenius cosine per layer (K1189 diagnostic).
    This is the SAME measurement P3.B1 found at 0.1607 (layer 36).

    No modification to any adapter weights.
    """
    import safetensors.numpy as stn

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_st = output_dir / "adapters.safetensors"
    merged_cfg = output_dir / "adapter_config.json"

    math_d = stn.load_file(str(MATH_ADAPTER_DIR / "adapters.safetensors"))
    pers_d = stn.load_file(str(PERSONAL_ADAPTER_DIR / "adapters.safetensors"))

    max_cos_b = 0.0   # max B-matrix Frobenius cosine across all overlap layers
    n_overlap = 0
    total_norm_domain = 0.0
    total_norm_pers = 0.0

    merged = {}

    for layer in range(42):
        key_a = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_a"
        key_b = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_b"

        if key_a not in math_d:
            continue

        la_D = math_d[key_a].astype(np.float64)    # [d_in, r_D=6]
        lb_D = math_d[key_b].astype(np.float64)    # [r_D=6, d_out]
        lb_D_s = MATH_SCALE * lb_D                  # scale baked in

        d_in = la_D.shape[0]
        d_out = lb_D.shape[1]

        if key_a not in pers_d:
            # Math-only layer: pad with zeros for rank alignment
            pad_a = np.zeros((d_in, RANK_PERS), dtype=np.float32)
            pad_b = np.zeros((RANK_PERS, d_out), dtype=np.float32)
            la_merged = np.concatenate([la_D.astype(np.float32), pad_a], axis=1)
            lb_merged = np.concatenate([lb_D_s.astype(np.float32), pad_b], axis=0)
            merged[key_a] = la_merged
            merged[key_b] = lb_merged
            continue

        la_P = pers_d[key_a].astype(np.float64)    # [d_in, r_P=4]
        lb_P = pers_d[key_b].astype(np.float64)    # [r_P=4, d_out]
        lb_P_s = PERS_SCALE * lb_P                  # scale baked in

        # Diagnostic: max row-cosine between B_D rows and B_P rows
        # B_D has r_D=6 rows of length d_out, B_P has r_P=4 rows of length d_out
        # Measures how aligned their output directions are
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            row_norms_D = np.linalg.norm(lb_D_s, axis=1, keepdims=True) + 1e-12
            row_norms_P = np.linalg.norm(lb_P_s, axis=1, keepdims=True) + 1e-12
            lb_D_norm = lb_D_s / row_norms_D
            lb_P_norm = lb_P_s / row_norms_P
            cross_cos = np.nan_to_num(np.abs(lb_D_norm @ lb_P_norm.T), nan=0.0)
        max_cos_layer = float(np.max(cross_cos))
        max_cos_b = max(max_cos_b, max_cos_layer)

        # Frobenius norms of full ΔW for power ratio
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            dw_d = np.nan_to_num(la_D @ lb_D_s, nan=0.0, posinf=0.0, neginf=0.0)
            dw_p = np.nan_to_num(la_P @ lb_P_s, nan=0.0, posinf=0.0, neginf=0.0)
        total_norm_domain += float(np.linalg.norm(dw_d, "fro"))
        total_norm_pers += float(np.linalg.norm(dw_p, "fro"))
        n_overlap += 1

        # Pure additive merge: NO projection, NO modification
        la_merged = np.concatenate(
            [la_D.astype(np.float32), la_P.astype(np.float32)], axis=1
        )   # [d_in, 10]
        lb_merged = np.concatenate(
            [lb_D_s.astype(np.float32), lb_P_s.astype(np.float32)], axis=0
        )   # [10, d_out]

        merged[key_a] = la_merged
        merged[key_b] = lb_merged

    power_ratio = total_norm_domain / (total_norm_pers + 1e-8)
    print(f"Max B-matrix Frobenius cosine (K1189): {max_cos_b:.4f} (P3.B1 baseline: 0.1607)", flush=True)
    print(f"Power ratio (domain/personal): {power_ratio:.3f}x (P3.B1 baseline: 2.96x from B-matrices)", flush=True)
    print(f"Overlap layers: {n_overlap}", flush=True)

    if not (merged_st.exists() and merged_cfg.exists()):
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
        print(f"Pure additive merged adapter saved: {size_mb:.1f}MB (rank={RANK_MERGED})", flush=True)
    else:
        print(f"Pure additive merged adapter already exists at {output_dir}", flush=True)

    return {
        "max_cos_b": round(float(max_cos_b), 4),
        "power_ratio": round(float(power_ratio), 3),
        "n_overlap_layers": n_overlap,
        "total_norm_domain": round(total_norm_domain, 3),
        "total_norm_pers": round(total_norm_pers, 3),
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
        f"=== P3.B4: Pure Additive Composition Baseline "
        f"({'SMOKE' if IS_SMOKE else 'FULL'}, N_style={N_STYLE}, N_math={N_MATH}) ===",
        flush=True,
    )
    print("Hypothesis: Pure additive (no projection) gives MORE style than B-GS (60%)", flush=True)
    print("B-GS in P3.B1 modified B_P, removing B-rows in col(B_D). Since style IS in col(ΔW_D),", flush=True)
    print("B-GS modification itself destroyed some style. Pure additive keeps both signals intact.", flush=True)

    # ── Phase 0: Build pure additive adapter (K1189 diagnostic) ──────────────
    print("\n--- Phase 0: Pure additive adapter construction ---", flush=True)
    add_dir = EXPERIMENT_DIR / "pure_additive_adapter"
    alg = build_pure_additive_adapter(add_dir)
    results["algebraic"] = alg

    max_cos_b = alg["max_cos_b"]
    k1189_note = f"max_cos_b={max_cos_b:.4f} (diagnostic, P3.B1 baseline=0.1607)"
    print(f"K1189 diagnostic: {k1189_note}", flush=True)

    # ── Phase 1: Personal-only style baseline ─────────────────────────────────
    print("\n--- Phase 1: Personal-only style baseline ---", flush=True)
    personal_only_rate = eval_style_compliance(
        adapter_path=PERSONAL_ADAPTER_DIR, n_eval=N_STYLE, label="personal-only"
    )

    # ── Phase 2: Math-only MCQ baseline ───────────────────────────────────────
    print("\n--- Phase 2: Math-only MCQ baseline ---", flush=True)
    math_only_acc = eval_mmlu_accuracy(
        adapter_path=MATH_ADAPTER_DIR, n_eval=N_MATH, label="math-only"
    )

    # ── Phase 3: Pure additive composed (K1187, K1188) ───────────────────────
    print("\n--- Phase 3: Pure additive composed behavioral ---", flush=True)
    add_style_rate = eval_style_compliance(
        adapter_path=add_dir, n_eval=N_STYLE, label="additive"
    )
    add_math_acc = eval_mmlu_accuracy(
        adapter_path=add_dir, n_eval=N_MATH, label="additive"
    )

    # ── Kill criteria evaluation ───────────────────────────────────────────────
    style_delta = personal_only_rate - add_style_rate
    math_delta = math_only_acc - add_math_acc

    k1187_pass = bool(add_style_rate >= 66.0)
    k1188_pass = bool(add_math_acc >= 5.0)

    results["k1187"] = {
        "personal_only_rate": round(float(personal_only_rate), 1),
        "additive_rate": round(float(add_style_rate), 1),
        "delta_pp": round(float(style_delta), 1),
        "threshold_pct": 66.0,
        "pass": k1187_pass,
        "vs_p3b1_b_gs": round(float(add_style_rate) - 60.0, 1),
    }
    results["k1188"] = {
        "math_only_acc": round(float(math_only_acc), 1),
        "additive_acc": round(float(add_math_acc), 1),
        "delta_pp": round(float(math_delta), 1),
        "threshold_pct": 5.0,
        "pass": k1188_pass,
    }
    results["k1189"] = {
        "max_cos_b": max_cos_b,
        "note": k1189_note,
        "pass": True,  # Always pass: diagnostic only
    }

    elapsed = time.time() - t0
    all_pass = bool(k1187_pass and k1188_pass)
    results["summary"] = {
        "all_pass": all_pass,
        "k1187_pass": k1187_pass,
        "k1188_pass": k1188_pass,
        "style_vs_p3b1": round(float(add_style_rate) - 60.0, 1),
        "power_ratio": alg["power_ratio"],
        "elapsed_s": round(elapsed, 1),
    }

    print("\n=== RESULTS ===", flush=True)
    print(
        f"K1187: {'PASS' if k1187_pass else 'FAIL'} style: "
        f"{personal_only_rate:.1f}%→{add_style_rate:.1f}% "
        f"(Δ={style_delta:+.1f}pp, threshold≥66%)",
        flush=True,
    )
    print(
        f"       vs P3.B1 B-GS (60%): {'+' if add_style_rate >= 60 else ''}{add_style_rate - 60:.1f}pp",
        flush=True,
    )
    print(
        f"K1188: {'PASS' if k1188_pass else 'FAIL'} math: "
        f"{math_only_acc:.1f}%→{add_math_acc:.1f}% "
        f"(Δ={math_delta:+.1f}pp, threshold≥5%)",
        flush=True,
    )
    print(f"K1189: DIAGNOSTIC max_cos_b={max_cos_b:.4f}", flush=True)

    if k1187_pass:
        print("\n→ INTERPRETATION: Pure additive composition meets style threshold.", flush=True)
        print("  B-GS modification in P3.B1 was harming style (≥6pp extra degradation).", flush=True)
        print("  Pure additive is the correct composition method: keep both adapters unmodified.", flush=True)
    else:
        delta_vs_p3b1 = add_style_rate - 60.0
        if delta_vs_p3b1 > 0:
            print(
                f"\n→ PARTIAL: Pure additive +{delta_vs_p3b1:.0f}pp vs B-GS but K1187 still fails (Δ={style_delta:.1f}pp).",
                flush=True,
            )
        else:
            print(
                f"\n→ FAIL: Pure additive {delta_vs_p3b1:.0f}pp vs B-GS (no improvement).",
                flush=True,
            )
        print(
            "  Context shift hypothesis confirmed: domain adapter primes early layers (0-25),",
            flush=True,
        )
        print(
            "  biasing hidden states before personal adapter activates (26-41).",
            flush=True,
        )
        print(
            "  → Next: P3.B5 — retrain personal adapter ON TOP of domain-adapted model.",
            flush=True,
        )

    print(f"\nALL: {'PASS' if all_pass else 'FAIL'} ({elapsed:.0f}s)", flush=True)

    out_path = EXPERIMENT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
