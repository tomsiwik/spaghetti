#!/usr/bin/env python3
"""
exp_model_thinking_preservation_training — smoke-scale implementation.

Trains a LoRA adapter on s1K reasoning traces WITH <think>...</think> tokens
in the target, then evaluates MMLU-Pro with enable_thinking=True across 3
categories (math, comp_sci, health) to test the recipe from MATH.md.

Smoke mode runs a tiny pipeline end-to-end. KC cannot be conclusively
verified at smoke scale — `results.json.is_smoke=True` and the verdict is
provisional.
"""

import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_DIR = EXPERIMENT_DIR / "data"
ADAPTER_DIR = EXPERIMENT_DIR / "adapters" / "thinking_preservation"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "1") == "1"  # default smoke
SEED = 42

LORA_RANK = 8
LORA_SCALE = 1.0
LORA_DROPOUT = 0.0
LORA_KEYS = ["self_attn.v_proj", "self_attn.o_proj"]
N_STEPS = 20 if IS_SMOKE else 1000
BATCH_SIZE = 1
LR = 1e-5
MAX_SEQ_LEN = 2048
MAX_TOTAL_CHARS = 6000

EVAL_PER_CAT = 2 if IS_SMOKE else 20
EVAL_CATEGORIES = ["math", "computer science", "health"]


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ── Phase 1: reuse s1K training data from sibling experiment ───────────────

def phase_prepare_training_data():
    """Reuse s1K train.jsonl from exp_p11_reasoning_sft_s1k (thinking-enabled).

    MATH.md (A1): every target must contain <think>...</think> — s1K traces
    satisfy this by construction. We simply copy/link the prepared data.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    sibling = EXPERIMENT_DIR.parent / "exp_p11_reasoning_sft_s1k" / "data"
    local_train = DATA_DIR / "train.jsonl"
    local_valid = DATA_DIR / "valid.jsonl"

    if not local_train.exists():
        src_train = sibling / "train.jsonl"
        src_valid = sibling / "valid.jsonl"
        if not src_train.exists():
            raise FileNotFoundError(
                f"Expected s1K training data at {src_train}. Run exp_p11_reasoning_sft_s1k first."
            )
        import shutil as _sh
        _sh.copy(src_train, local_train)
        _sh.copy(src_valid, local_valid)
        log(f"  Copied s1K data from {sibling}")

    # Verify A1 — every target has <think>
    n_train = 0
    n_with_think = 0
    think_loss_frac = []
    with open(local_train) as f:
        for line in f:
            ex = json.loads(line)
            n_train += 1
            assistant = next(
                (m["content"] for m in ex["messages"] if m["role"] == "assistant"),
                "",
            )
            if "<think>" in assistant and "</think>" in assistant:
                n_with_think += 1
                think_match = re.search(r"<think>(.*?)</think>", assistant, re.DOTALL)
                if think_match:
                    think_loss_frac.append(len(think_match.group(1)) / max(len(assistant), 1))

    a1_pass = (n_with_think / max(n_train, 1)) >= 0.99
    mean_think_frac = float(np.mean(think_loss_frac)) if think_loss_frac else 0.0
    log(
        f"  A1 check: {n_with_think}/{n_train} examples contain <think> "
        f"(think fraction mean={mean_think_frac:.2f})"
    )

    n_valid = sum(1 for _ in open(local_valid))
    return {
        "n_train": n_train,
        "n_valid": n_valid,
        "a1_pass": a1_pass,
        "mean_think_frac": round(mean_think_frac, 3),
    }


# ── Phase 2: LoRA training via mlx_lm.lora CLI ────────────────────────────

def phase_train():
    """Train LoRA adapter using mlx_lm.lora CLI with the prepared data."""
    import yaml
    log("\n[Phase 2] LoRA training (thinking-enabled targets)...")
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    lora_config_path = EXPERIMENT_DIR / "lora_config.yaml"
    lora_config = {
        "lora_parameters": {
            "rank": LORA_RANK,
            "scale": LORA_SCALE,
            "dropout": LORA_DROPOUT,
            "keys": LORA_KEYS,
        }
    }
    with open(lora_config_path, "w") as f:
        yaml.dump(lora_config, f)

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", MODEL_ID,
        "--train",
        "--data", str(DATA_DIR),
        "--iters", str(N_STEPS),
        "--batch-size", str(BATCH_SIZE),
        "--learning-rate", str(LR),
        "--adapter-path", str(ADAPTER_DIR),
        "--save-every", "20" if IS_SMOKE else "200",
        "--max-seq-length", str(MAX_SEQ_LEN),
        "--grad-checkpoint",
        "-c", str(lora_config_path),
    ]

    log(f"  Running: mlx_lm.lora --iters {N_STEPS} (smoke={IS_SMOKE})")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        log(f"ERROR: Training failed with code {result.returncode}")
        return {"status": "failed", "time_s": round(elapsed, 1)}

    log(f"  Training done in {elapsed:.0f}s")
    return {"status": "ok", "time_s": round(elapsed, 1), "steps": N_STEPS}


# ── Phase 3: MMLU-Pro eval with enable_thinking=True ─────────────────────

def strip_thinking(response):
    """Measure thinking chars and return cleaned answer.

    Gemma 4 emits thinking as either <think>...</think> OR on the
    `<|channel|>thought` channel. We match both — mem-antipattern-008.
    """
    thinking_len = 0
    m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if m:
        thinking_len = len(m.group(1))
    m_channel = re.search(
        r"<\|channel\|>thought(.*?)(?:<\|end_channel\|>|<\|channel\|>final)",
        response, re.DOTALL,
    )
    if m_channel:
        thinking_len = max(thinking_len, len(m_channel.group(1)))
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    cleaned = re.sub(
        r"<\|channel\|>thought.*?(<\|end_channel\|>|<\|channel\|>final)",
        "", cleaned, flags=re.DOTALL,
    )
    return cleaned.strip(), thinking_len


def parse_mcq_answer(response):
    """Extract the MCQ letter from a possibly-thinking response."""
    answer_text, thinking_len = strip_thinking(response)
    for pattern in [
        r"\banswer is\s*([A-J])",
        r"\banswer:\s*([A-J])",
        r"\b([A-J])\b\s*$",
        r"^\s*([A-J])\b",
    ]:
        m = re.search(pattern, answer_text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper(), thinking_len
    m = re.search(r"\b([A-J])\b", answer_text)
    if m:
        return m.group(1).upper(), thinking_len
    return None, thinking_len


def phase_eval_mmlu_pro(adapter_path=None):
    """Evaluate MMLU-Pro on 3 categories with thinking enabled."""
    from mlx_lm import load, generate

    label = "ADAPTED" if adapter_path else "BASE"
    log(f"\n[Eval] MMLU-Pro with thinking ({label})")

    mmlu_path = EXPERIMENT_DIR.parent / "exp_bench_mmlu_pro" / "data" / "test.parquet"
    if not mmlu_path.exists():
        log(f"  MMLU-Pro data missing at {mmlu_path}")
        return {"accuracy": None, "error": "mmlu-data-missing"}

    df = pd.read_parquet(mmlu_path)
    available = set(df["category"].unique())
    log(f"  Available categories (sample): {sorted(available)[:5]}...")

    load_kwargs = {"adapter_path": str(adapter_path)} if adapter_path else {}
    try:
        model, tokenizer = load(MODEL_ID, **load_kwargs)
    except Exception as e:
        log(f"  ERROR loading model: {e}")
        return {"accuracy": None, "error": str(e)}

    log_memory(f"{label}-post-load")

    rng = np.random.RandomState(SEED)
    correct_total, total, thinking_total = 0, 0, 0
    per_cat = {}
    OPTION_LETTERS = "ABCDEFGHIJ"

    for cat in EVAL_CATEGORIES:
        cat_df = df[df["category"].str.lower() == cat.lower()]
        if len(cat_df) == 0:
            log(f"  Category '{cat}' not in dataset; skipping.")
            continue
        n_sample = min(EVAL_PER_CAT, len(cat_df))
        idx = rng.choice(len(cat_df), n_sample, replace=False)
        sample = cat_df.iloc[idx]

        cat_correct, cat_thinking = 0, 0
        for _, row in sample.iterrows():
            options = row.get("options", [])
            n_opts = len(options)
            option_text = "\n".join(
                f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)
            )
            correct_letter = OPTION_LETTERS[int(row["answer_index"])]

            user_content = (
                f"Answer the following multiple choice question. "
                f"Select the single best answer letter "
                f"(A through {OPTION_LETTERS[n_opts - 1]}).\n\n"
                f"Question: {row['question']}\n\n"
                f"Options:\n{option_text}\n\n"
                f"Answer:"
            )
            messages = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )

            try:
                # mem-antipattern-008: keep max_tokens large so thinking trace
                # isn't truncated and mis-scored as "thinking suppressed".
                response = generate(model, tokenizer, prompt=prompt, max_tokens=2048)
                predicted, t_chars = parse_mcq_answer(response)
                cat_thinking += t_chars
                if predicted == correct_letter:
                    cat_correct += 1
            except Exception as e:
                log(f"    ERROR generating: {e}")

            mx.eval()

        thinking_total += cat_thinking
        correct_total += cat_correct
        total += n_sample
        acc = cat_correct / n_sample * 100 if n_sample else 0.0
        avg_think = cat_thinking / n_sample if n_sample else 0.0
        per_cat[cat] = {
            "correct": cat_correct,
            "total": n_sample,
            "acc": round(acc, 1),
            "avg_thinking_chars": round(avg_think, 0),
        }
        log(
            f"  {cat}: {acc:.0f}% ({cat_correct}/{n_sample}), "
            f"avg_thinking={avg_think:.0f} chars"
        )

    cleanup(model, tokenizer)

    overall_acc = correct_total / total * 100 if total else 0.0
    overall_think = thinking_total / total if total else 0.0
    log(f"  Overall: {overall_acc:.1f}% ({correct_total}/{total}), "
        f"avg_thinking={overall_think:.0f}")

    return {
        "accuracy": round(overall_acc, 1),
        "correct": correct_total,
        "total": total,
        "avg_thinking_chars": round(overall_think, 0),
        "per_category": per_cat,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 60)
    log("exp_model_thinking_preservation_training")
    log(f"IS_SMOKE={IS_SMOKE}  N_STEPS={N_STEPS}  EVAL_PER_CAT={EVAL_PER_CAT}")
    log("=" * 60)
    log_memory("start")

    results = {
        "experiment": "exp_model_thinking_preservation_training",
        "model": MODEL_ID,
        "is_smoke": IS_SMOKE,
        "lora_rank": LORA_RANK,
        "lora_scale": LORA_SCALE,
        "lora_keys": LORA_KEYS,
        "n_steps": N_STEPS,
        "eval_categories": EVAL_CATEGORIES,
        "eval_per_cat": EVAL_PER_CAT,
    }

    # Phase 1
    log("\n[Phase 1] Prepare training data")
    data_info = phase_prepare_training_data()
    results["training_data"] = data_info

    # Phase 2
    train_results = phase_train()
    results["training"] = train_results
    adapter_path = ADAPTER_DIR if train_results.get("status") == "ok" else None

    # Phase 3a: base eval
    base = phase_eval_mmlu_pro(adapter_path=None)
    results["base"] = base

    # Phase 3b: adapter eval
    if adapter_path is not None:
        adapted = phase_eval_mmlu_pro(adapter_path=adapter_path)
        results["adapted"] = adapted
    else:
        results["adapted"] = {"accuracy": None, "error": "training_failed"}

    # Kill criteria scoring
    base_acc = results["base"].get("accuracy") or 0.0
    adapt_acc = results["adapted"].get("accuracy") or 0.0
    adapt_think = results["adapted"].get("avg_thinking_chars") or 0.0

    # K1685: within 2pp of base+thinking
    k1685_delta = adapt_acc - base_acc
    k1685_pass = abs(k1685_delta) <= 2.0 and adapt_acc is not None

    # K1686: thinking chars >= 1500
    k1686_pass = adapt_think >= 1500

    # K1687: recipe holds across 3 categories (each within 5pp of base)
    k1687_details = {}
    k1687_all_pass = True
    for cat in EVAL_CATEGORIES:
        b = results["base"].get("per_category", {}).get(cat, {}).get("acc")
        a = results["adapted"].get("per_category", {}).get(cat, {}).get("acc")
        if b is None or a is None:
            k1687_details[cat] = {"base": b, "adapted": a, "pass": False}
            k1687_all_pass = False
            continue
        delta = a - b
        passed = abs(delta) <= 5.0
        k1687_details[cat] = {
            "base": b, "adapted": a, "delta": round(delta, 1), "pass": passed,
        }
        k1687_all_pass = k1687_all_pass and passed

    results["kill_criteria"] = {
        "K1685": {
            "desc": "adapter+thinking within 2pp of base+thinking MMLU-Pro",
            "base_acc": base_acc,
            "adapt_acc": adapt_acc,
            "delta_pp": round(k1685_delta, 2),
            "pass": bool(k1685_pass),
        },
        "K1686": {
            "desc": "avg thinking chars >= 1500",
            "value": adapt_think,
            "pass": bool(k1686_pass),
        },
        "K1687": {
            "desc": "recipe holds across 3 categories (each within 5pp)",
            "per_category": k1687_details,
            "pass": bool(k1687_all_pass),
        },
    }

    all_pass = bool(k1685_pass and k1686_pass and k1687_all_pass)
    results["all_pass"] = all_pass
    # Smoke runs are inherently provisional — cannot be supported/killed.
    results["verdict"] = "PROVISIONAL" if IS_SMOKE else (
        "SUPPORTED" if all_pass else "KILLED"
    )
    results["total_time_s"] = round(time.time() - t0, 1)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    log("\n" + "=" * 60)
    log("RESULTS")
    log("=" * 60)
    log(f"Base MMLU-Pro+thinking:    {base_acc:.1f}%")
    log(f"Adapter MMLU-Pro+thinking: {adapt_acc:.1f}%  (Δ={k1685_delta:+.1f}pp)")
    log(f"Adapter avg thinking chars:{adapt_think:.0f}")
    log(f"K1685 within 2pp:       {'PASS' if k1685_pass else 'FAIL'}")
    log(f"K1686 thinking>=1500:   {'PASS' if k1686_pass else 'FAIL'}")
    log(f"K1687 3-cat each 5pp:   {'PASS' if k1687_all_pass else 'FAIL'}")
    log(f"Verdict: {results['verdict']}")
    log(f"Total time: {results['total_time_s']:.0f}s")
    log(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
