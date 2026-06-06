#!/usr/bin/env python3
"""
P2.A1: Tier 2 + Tier 3 Simultaneous Activation
Test: domain (math) + personal (style) adapter composition preserves both behaviors.

Kill criteria:
  K1: Math MCQ accuracy with composed adapter within 5pp of math-only adapter
  K2: Style compliance with composed adapter within 10pp of personal-only adapter
  K3: Max B-matrix cosine between math and personal adapters < 0.1
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

# Adapter paths
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
N_EVAL = 5 if IS_SMOKE else 25
SEED = 42

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

# Memory safety
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


# ─── Phase 0: K3 — Analytical B-matrix cosine ────────────────────────────────

def compute_bmatrix_cosine() -> dict:
    """
    Compute max B-matrix cosine between math and personal adapters (K3).
    Measures: max_{i,j,l} |cos(b_math^i_l, b_personal^j_l)| over overlap layers.
    """
    import safetensors.numpy as stn

    math_d = stn.load_file(str(MATH_ADAPTER_DIR / "adapters.safetensors"))
    personal_d = stn.load_file(str(PERSONAL_ADAPTER_DIR / "adapters.safetensors"))

    MATH_SCALE = 6.0
    PERSONAL_SCALE = 4.0

    cosines_by_layer = {}
    max_cos_overall = 0.0

    for layer in range(26, 42):  # overlap layers only (personal adapter covers 26-41)
        key_b = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_b"
        if key_b not in math_d or key_b not in personal_d:
            continue

        lb_math = math_d[key_b].astype(np.float32)    # [6, d_out]
        lb_pers = personal_d[key_b].astype(np.float32) # [4, d_out]

        # Apply scales as used during composition
        lb_math_scaled = MATH_SCALE * lb_math    # [6, d_out]
        lb_pers_scaled = PERSONAL_SCALE * lb_pers # [4, d_out]

        # Normalize rows to unit vectors
        norms_m = np.linalg.norm(lb_math_scaled, axis=1, keepdims=True) + 1e-8
        norms_p = np.linalg.norm(lb_pers_scaled, axis=1, keepdims=True) + 1e-8
        lb_math_unit = lb_math_scaled / norms_m  # [6, d_out]
        lb_pers_unit = lb_pers_scaled / norms_p  # [4, d_out]

        # Cosine matrix [6, 4]: max absolute cosine between any pair of output directions
        cos_matrix = np.abs(lb_math_unit @ lb_pers_unit.T)
        max_cos_layer = float(cos_matrix.max())
        cosines_by_layer[str(layer)] = round(max_cos_layer, 6)
        max_cos_overall = max(max_cos_overall, max_cos_layer)

    print(f"K3: max B-matrix cosine across overlap layers 26-41 = {max_cos_overall:.4f}", flush=True)
    return {
        "max_cos": max_cos_overall,
        "layer_max_cos": cosines_by_layer,
        "n_overlap_layers": len(cosines_by_layer),
        "threshold": 0.1,
    }


# ─── Merged adapter creation ──────────────────────────────────────────────────

def create_merged_adapter(output_dir: Path) -> Path:
    """
    Create merged math+personal adapter safetensors.
    Strategy: rank=10 (6+4) for all layers. Scales baked into lora_b.
    Overlap layers (26-41): concat math+personal adapters.
    Math-only layers (0-25): concat math adapter + zero padding.
    """
    import safetensors.numpy as stn

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_safetensors = output_dir / "adapters.safetensors"
    merged_config = output_dir / "adapter_config.json"

    if merged_safetensors.exists() and merged_config.exists():
        print(f"Merged adapter already exists at {output_dir}", flush=True)
        return output_dir

    math_d = stn.load_file(str(MATH_ADAPTER_DIR / "adapters.safetensors"))
    personal_d = stn.load_file(str(PERSONAL_ADAPTER_DIR / "adapters.safetensors"))

    MATH_SCALE = 6.0
    PERSONAL_SCALE = 4.0
    RANK_MATH = 6
    RANK_PERSONAL = 4
    RANK_MERGED = RANK_MATH + RANK_PERSONAL  # 10

    merged = {}

    for layer in range(42):
        key_a = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_a"
        key_b = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_b"

        if key_a not in math_d:
            continue  # should not happen — math adapter covers all 42 layers

        la_math = math_d[key_a].astype(np.float32)  # [d_in, 6]
        lb_math = math_d[key_b].astype(np.float32)  # [6, d_out]
        d_in = la_math.shape[0]
        d_out = lb_math.shape[1]

        if key_a in personal_d:
            # Overlap layer: concat adapters, bake in scales
            la_pers = personal_d[key_a].astype(np.float32)  # [d_in, 4]
            lb_pers = personal_d[key_b].astype(np.float32)  # [4, d_out]

            la_merged = np.concatenate([la_math, la_pers], axis=1)            # [d_in, 10]
            lb_merged = np.concatenate(
                [MATH_SCALE * lb_math, PERSONAL_SCALE * lb_pers], axis=0
            )  # [10, d_out]
        else:
            # Math-only layer: pad personal slots with zeros
            pad_a = np.zeros((d_in, RANK_PERSONAL), dtype=np.float32)
            pad_b = np.zeros((RANK_PERSONAL, d_out), dtype=np.float32)
            la_merged = np.concatenate([la_math, pad_a], axis=1)              # [d_in, 10]
            lb_merged = np.concatenate([MATH_SCALE * lb_math, pad_b], axis=0) # [10, d_out]

        merged[key_a] = la_merged.astype(np.float32)
        merged[key_b] = lb_merged.astype(np.float32)

    stn.save_file(merged, str(merged_safetensors))

    # adapter_config.json: scale=1.0 (original scales baked into lora_b)
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
    print(f"Merged adapter saved: {size_mb:.1f}MB at {output_dir}", flush=True)
    return output_dir


# ─── K1: Math MCQ accuracy ────────────────────────────────────────────────────

def eval_mmlu_accuracy(adapter_path=None, n_eval: int = 25, label: str = "") -> float:
    """Evaluate MMLU abstract_algebra MCQ accuracy. Returns accuracy 0-100."""
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("cais/mmlu", "abstract_algebra", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))

    if adapter_path is not None:
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)

    log_memory(f"K1-{label}")

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

        response = generate(model, tokenizer, prompt=formatted, max_tokens=20, verbose=False)

        gt = OPTION_LETTERS[ex["answer"]]
        pred = response.strip().upper()
        pred_letter = None
        for letter in OPTION_LETTERS:
            if pred.startswith(letter):
                pred_letter = letter
                break
        if pred_letter is None:
            m = re.search(r"\b([ABCD])\b", pred)
            if m:
                pred_letter = m.group(1)

        is_correct = pred_letter == gt
        if is_correct:
            correct += 1
        if i < 3:
            print(f"  q{i}: gt={gt}, pred={pred_letter}, {'✓' if is_correct else '✗'}", flush=True)

    acc = correct / len(ds) * 100
    print(f"K1 {label}: {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    cleanup(model, tokenizer)
    return acc


# ─── K2: Style compliance ─────────────────────────────────────────────────────

def eval_style_compliance(adapter_path=None, n_eval: int = 25, label: str = "") -> float:
    """Evaluate style compliance ('Hope that helps, friend!'). Returns rate 0-100."""
    from mlx_lm import generate, load

    questions = STYLE_QUESTIONS[:n_eval]

    if adapter_path is not None:
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)

    log_memory(f"K2-{label}")

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
    print(f"K2 {label}: {compliant}/{len(questions)} = {rate:.1f}%", flush=True)
    cleanup(model, tokenizer)
    return rate


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    results = {"is_smoke": IS_SMOKE, "n_eval": N_EVAL}

    print(f"=== P2.A1: Tier 2+3 Composition ({'SMOKE' if IS_SMOKE else 'FULL'}, N={N_EVAL}) ===",
          flush=True)

    # ── K3: B-matrix cosine (analytical, fast) ────────────────────────────────
    print("\n--- K3: B-matrix cosine ---", flush=True)
    k3_data = compute_bmatrix_cosine()
    k3_pass = k3_data["max_cos"] < k3_data["threshold"]
    results["k3"] = {**k3_data, "pass": k3_pass}
    print(f"K3: {'PASS' if k3_pass else 'FAIL'} max_cos={k3_data['max_cos']:.4f} < {k3_data['threshold']}", flush=True)

    # ── Create merged adapter ─────────────────────────────────────────────────
    print("\n--- Creating merged adapter ---", flush=True)
    merged_dir = EXPERIMENT_DIR / "merged_adapter"
    create_merged_adapter(merged_dir)

    # ── K1: Math MCQ accuracy ─────────────────────────────────────────────────
    print("\n--- K1: Math MCQ accuracy ---", flush=True)

    math_only_acc = eval_mmlu_accuracy(
        adapter_path=MATH_ADAPTER_DIR, n_eval=N_EVAL, label="math-only"
    )
    composed_math_acc = eval_mmlu_accuracy(
        adapter_path=merged_dir, n_eval=N_EVAL, label="composed"
    )

    k1_delta = math_only_acc - composed_math_acc
    k1_pass = k1_delta <= 5.0
    results["k1"] = {
        "math_only_acc": math_only_acc,
        "composed_acc": composed_math_acc,
        "delta_pp": round(k1_delta, 1),
        "threshold_pp": 5.0,
        "pass": k1_pass,
    }
    print(f"K1: {'PASS' if k1_pass else 'FAIL'} math-only={math_only_acc:.1f}% → composed={composed_math_acc:.1f}% (Δ={k1_delta:+.1f}pp, threshold ≤5pp)", flush=True)

    # ── K2: Style compliance ──────────────────────────────────────────────────
    print("\n--- K2: Style compliance ---", flush=True)

    personal_only_rate = eval_style_compliance(
        adapter_path=PERSONAL_ADAPTER_DIR, n_eval=N_EVAL, label="personal-only"
    )
    composed_style_rate = eval_style_compliance(
        adapter_path=merged_dir, n_eval=N_EVAL, label="composed"
    )

    k2_delta = personal_only_rate - composed_style_rate
    k2_pass = k2_delta <= 10.0
    results["k2"] = {
        "personal_only_rate": personal_only_rate,
        "composed_rate": composed_style_rate,
        "delta_pp": round(k2_delta, 1),
        "threshold_pp": 10.0,
        "pass": k2_pass,
    }
    print(f"K2: {'PASS' if k2_pass else 'FAIL'} personal-only={personal_only_rate:.1f}% → composed={composed_style_rate:.1f}% (Δ={k2_delta:+.1f}pp, threshold ≤10pp)", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    all_pass = k1_pass and k2_pass and k3_pass
    results["summary"] = {
        "all_pass": all_pass,
        "k1_pass": k1_pass,
        "k2_pass": k2_pass,
        "k3_pass": k3_pass,
        "math_only_acc": math_only_acc,
        "composed_math_acc": composed_math_acc,
        "personal_only_rate": personal_only_rate,
        "composed_style_rate": composed_style_rate,
        "max_bmatrix_cos": k3_data["max_cos"],
        "elapsed_min": round(elapsed / 60, 1),
    }

    with open(EXPERIMENT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'ALL PASS ✓' if all_pass else 'SOME FAIL ✗'} "
          f"K1={'PASS' if k1_pass else 'FAIL'} "
          f"K2={'PASS' if k2_pass else 'FAIL'} "
          f"K3={'PASS' if k3_pass else 'FAIL'} "
          f"({elapsed/60:.1f}min)", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
