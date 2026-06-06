#!/usr/bin/env python3
"""
T3.2: MMLU preserved under N=5 composition (0pp degradation target)

Tests whether each of the 5 domain adapters (math/code/medical/legal/finance)
degrades general MMLU on neutral subjects when activated individually (as routing does).
Also verifies Gemma 4's QK-norm makes this scale-invariant.

Kill criteria:
  K1053: MMLU(routing, any adapter) >= MMLU(base) - 1pp
  K1054: MMLU at scale=12,18 within 3pp of scale=6 (QK-norm scale invariance)
  K1055: Each adapter individually: MMLU(base+adapter_i) >= MMLU(base) - 1pp

Adapter paths:
  math/code/medical: exp_p1_t2_single_domain_training/adapters/{domain}/
  legal/finance:     exp_p1_t2_multi_domain_5/adapters/{domain}/
"""

import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import mlx.core as mx

# Memory safety — leave 8GB for OS + overhead
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_PER_SUBJECT = 3 if IS_SMOKE else 25
SEED = 42
OPTION_LETTERS = ["A", "B", "C", "D"]

# Neutral MMLU subjects (no overlap with any adapter training domain)
MMLU_SUBJECTS = [
    "high_school_geography",
    "world_religions",
    "philosophy",
    "high_school_world_history",
    "sociology",
]

# Subjects for scale sensitivity test (3 subjects × N_PER_SUBJECT questions)
SCALE_TEST_SUBJECTS = ["high_school_geography", "world_religions", "philosophy"]

T21_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_single_domain_training"
T26_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_multi_domain_5"

ADAPTER_PATHS = {
    "math":    T21_DIR / "adapters" / "math",
    "code":    T21_DIR / "adapters" / "code",
    "medical": T21_DIR / "adapters" / "medical",
    "legal":   T26_DIR / "adapters" / "legal",
    "finance": T26_DIR / "adapters" / "finance",
}

DOMAINS = list(ADAPTER_PATHS.keys())


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()


def set_lora_scale(model, new_scale: float):
    """Override LoRA scale in all adapted linear layers (for K1054 test)."""
    count = 0
    for _, module in model.named_modules():
        if hasattr(module, "scale") and hasattr(module, "lora_a") and hasattr(module, "lora_b"):
            module.scale = new_scale
            count += 1
    print(f"  Set scale={new_scale} on {count} LoRA layers", flush=True)
    return count


def load_mmlu_questions(subjects, n_per_subject):
    """Load MMLU questions from HuggingFace datasets, return list of (subject, question, choices, answer_idx)."""
    from datasets import load_dataset

    all_questions = []
    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split="test")
        ds = ds.shuffle(seed=SEED).select(range(min(n_per_subject, len(ds))))
        for ex in ds:
            all_questions.append({
                "subject": subject,
                "question": ex["question"],
                "choices": ex["choices"],
                "answer": ex["answer"],  # int 0-3
            })
        print(f"  Loaded {min(n_per_subject, len(ds))} questions from {subject}", flush=True)

    return all_questions


def eval_mmlu_questions(model, tokenizer, questions, label=""):
    """Evaluate model on a list of MMLU question dicts. Returns accuracy (0-100)."""
    correct = 0
    for q in questions:
        formatted_q = (
            q["question"] + "\n"
            + "\n".join(f"({OPTION_LETTERS[i]}) {q['choices'][i]}" for i in range(len(q["choices"])))
        )
        prompt = "Answer this multiple choice question. Respond with only the letter (A/B/C/D).\n\n" + formatted_q

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted = prompt

        from mlx_lm import generate
        response = generate(model, tokenizer, prompt=formatted, max_tokens=20, verbose=False)

        gt = OPTION_LETTERS[q["answer"]]
        pred = response.strip().upper()
        pred_letter = next((l for l in OPTION_LETTERS if pred.startswith(l)), None)
        if pred_letter is None:
            m = re.search(r"\b([ABCD])\b", pred)
            if m:
                pred_letter = m.group(1)

        if pred_letter == gt:
            correct += 1

    acc = correct / len(questions) * 100
    print(f"  [{label}] {correct}/{len(questions)} = {acc:.1f}%", flush=True)
    return acc


# ─────────────────────────────────────────────
# Phase 1: Base MMLU
# ─────────────────────────────────────────────

def measure_base_mmlu(questions):
    from mlx_lm import load

    print("\n=== Phase 1: Base MMLU (no adapter) ===", flush=True)
    log_memory("pre-base-load")

    model, tokenizer = load(MODEL_ID)
    log_memory("base-loaded")

    acc = eval_mmlu_questions(model, tokenizer, questions, label="base")
    cleanup(model, tokenizer)
    log_memory("post-base-cleanup")
    return acc


# ─────────────────────────────────────────────
# Phase 2: Per-adapter MMLU (K1055)
# ─────────────────────────────────────────────

def measure_adapter_mmlu(domain, adapter_path, questions):
    from mlx_lm import load

    print(f"\n--- {domain} adapter (scale=6) ---", flush=True)
    log_memory(f"pre-{domain}-load")

    model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    log_memory(f"{domain}-loaded")

    acc = eval_mmlu_questions(model, tokenizer, questions, label=domain)
    cleanup(model, tokenizer)
    log_memory(f"post-{domain}-cleanup")
    return acc


# ─────────────────────────────────────────────
# Phase 3: Scale sensitivity (K1054)
# ─────────────────────────────────────────────

def measure_scale_sensitivity(scale_questions):
    from mlx_lm import load

    print("\n=== Phase 3: Scale sensitivity (K1054) — math adapter ===", flush=True)
    log_memory("pre-scale-load")

    # Load with math adapter (base scale=6)
    math_path = ADAPTER_PATHS["math"]
    model, tokenizer = load(MODEL_ID, adapter_path=str(math_path))
    log_memory("scale-loaded")

    scale_results = {}

    for scale in [6.0, 12.0, 18.0]:
        n_modified = set_lora_scale(model, scale)
        if n_modified == 0:
            print(f"  WARNING: No LoRA layers found for scale override. Skipping scale={scale}", flush=True)
            scale_results[f"scale_{int(scale)}"] = None
            continue
        acc = eval_mmlu_questions(model, tokenizer, scale_questions, label=f"math@scale={int(scale)}")
        scale_results[f"scale_{int(scale)}"] = round(acc, 1)

    cleanup(model, tokenizer)
    log_memory("post-scale-cleanup")
    return scale_results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    t_start = time.time()
    print("T3.2: MMLU Preservation Under N=5 Composition", flush=True)
    print(f"SMOKE_TEST={IS_SMOKE}, N_PER_SUBJECT={N_PER_SUBJECT}", flush=True)
    log_memory("start")

    # Verify all adapter paths
    for domain, path in ADAPTER_PATHS.items():
        adapter_file = path / "adapters.safetensors"
        if not adapter_file.exists():
            print(f"ERROR: Missing adapter: {adapter_file}", flush=True)
            sys.exit(1)
    print(f"All {len(ADAPTER_PATHS)} adapters found.", flush=True)

    # Load MMLU questions once (reused across all evals)
    print("\n--- Loading MMLU questions ---", flush=True)
    full_questions = load_mmlu_questions(MMLU_SUBJECTS, N_PER_SUBJECT)
    scale_questions = load_mmlu_questions(SCALE_TEST_SUBJECTS, N_PER_SUBJECT)
    print(f"Full eval: {len(full_questions)} questions, Scale test: {len(scale_questions)} questions", flush=True)

    results = {
        "is_smoke": IS_SMOKE,
        "n_per_subject": N_PER_SUBJECT,
        "n_full": len(full_questions),
        "n_scale": len(scale_questions),
        "subjects": MMLU_SUBJECTS,
        "scale_subjects": SCALE_TEST_SUBJECTS,
    }

    # Phase 1: Base MMLU
    base_acc = measure_base_mmlu(full_questions)
    results["base_mmlu"] = round(base_acc, 1)
    print(f"Base MMLU: {base_acc:.1f}%", flush=True)

    # Phase 2: Per-adapter MMLU (K1055)
    print("\n=== Phase 2: Per-adapter MMLU (K1055) ===", flush=True)
    adapter_accs = {}
    for domain in DOMAINS:
        acc = measure_adapter_mmlu(domain, ADAPTER_PATHS[domain], full_questions)
        adapter_accs[domain] = round(acc, 1)

    results["adapter_mmlu"] = adapter_accs

    # Phase 3: Scale sensitivity (K1054)
    if not IS_SMOKE:
        scale_results = measure_scale_sensitivity(scale_questions)
    else:
        print("SMOKE MODE: skipping scale sensitivity test", flush=True)
        scale_results = {"scale_6": None, "scale_12": None, "scale_18": None}

    results["scale_sensitivity"] = scale_results

    # Phase 4: Kill criteria
    print("\n=== Phase 4: Kill criteria ===", flush=True)
    kill_criteria = {}

    # K1055: each adapter <= 1pp degradation
    k1055_detail = {}
    k1055_pass = True
    for domain in DOMAINS:
        delta = base_acc - adapter_accs[domain]
        passes = delta <= 1.0
        k1055_detail[domain] = {
            "base_pct": round(base_acc, 1),
            "adapter_pct": adapter_accs[domain],
            "delta_pp": round(delta, 2),
            "pass": passes,
        }
        if not passes:
            k1055_pass = False
        status = "PASS" if passes else "FAIL"
        print(f"K1055 {domain}: base={base_acc:.1f}% adapter={adapter_accs[domain]:.1f}% Δ={delta:.2f}pp → {status}", flush=True)

    kill_criteria["K1055"] = {"pass": k1055_pass, "detail": k1055_detail}
    print(f"K1055 overall: {'PASS' if k1055_pass else 'FAIL'}", flush=True)

    # K1053: routing MMLU = best adapter MMLU >= base - 1pp
    # With routing, worst case is the adapter that degrades MMLU most
    worst_delta = max(base_acc - acc for acc in adapter_accs.values())
    k1053_pass = worst_delta <= 1.0
    kill_criteria["K1053"] = {
        "pass": k1053_pass,
        "worst_delta_pp": round(worst_delta, 2),
        "worst_domain": min(adapter_accs, key=adapter_accs.get),
        "routing_guarantee": "selects single adapter per query → inherits worst-case from K1055",
    }
    print(f"K1053 (routing MMLU): worst_delta={worst_delta:.2f}pp → {'PASS' if k1053_pass else 'FAIL'}", flush=True)

    # K1054: scale invariance within 3pp
    k1054_pass = True
    k1054_detail = {}
    if scale_results.get("scale_6") is not None:
        ref = scale_results["scale_6"]
        for scale_key in ["scale_12", "scale_18"]:
            if scale_results.get(scale_key) is not None:
                delta_from_6 = abs(scale_results[scale_key] - ref)
                passes = delta_from_6 <= 3.0
                k1054_detail[scale_key] = {
                    "acc": scale_results[scale_key],
                    "delta_from_scale6": round(delta_from_6, 2),
                    "pass": passes,
                }
                if not passes:
                    k1054_pass = False
                print(f"K1054 {scale_key}: {scale_results[scale_key]:.1f}% (Δ={delta_from_6:.2f}pp from scale=6) → {'PASS' if passes else 'FAIL'}", flush=True)
    else:
        print("K1054: scale sensitivity skipped (smoke mode or LoRA layer access failed)", flush=True)
        k1054_pass = None  # indeterminate

    kill_criteria["K1054"] = {"pass": k1054_pass, "detail": k1054_detail, "scale_results": scale_results}
    print(f"K1054 overall: {k1054_pass}", flush=True)

    results["kill_criteria"] = kill_criteria

    # Summary
    elapsed = time.time() - t_start
    results["total_time_s"] = round(elapsed, 1)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Base MMLU:                {base_acc:.1f}%", flush=True)
    for domain in DOMAINS:
        delta = base_acc - adapter_accs[domain]
        status = "PASS" if delta <= 1.0 else "FAIL"
        print(f"  {domain:10s}: {adapter_accs[domain]:.1f}% (Δ={delta:.2f}pp) [{status}]", flush=True)
    print(f"K1053 routing MMLU:       {'PASS' if k1053_pass else 'FAIL'} (worst Δ={worst_delta:.2f}pp)", flush=True)
    print(f"K1054 scale invariance:   {k1054_pass}", flush=True)
    print(f"K1055 per-adapter MMLU:   {'PASS' if k1055_pass else 'FAIL'}", flush=True)
    print(f"Total time: {elapsed:.0f}s ({elapsed/3600:.2f}h)", flush=True)


if __name__ == "__main__":
    main()
