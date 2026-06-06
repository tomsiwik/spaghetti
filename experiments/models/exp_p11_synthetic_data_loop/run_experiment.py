#!/usr/bin/env python3
"""
P11.I0: Synthetic Reasoning Data Generation Loop (STAR)

STAR approach (arXiv:2203.14465): generate thinking traces from base model,
filter by answer correctness, fine-tune LoRA, iterate (Round 2).

Phases:
  1. Generate 70 thinking traces (base model, MMLU-Pro pool 1)
  2. Filter correct traces → train Round 1 adapter
  3. Eval Round 1 on held-out MMLU-Pro (70 questions)
  4. Generate 70 more traces using Round 1 adapter (pool 2)
  5. Train Round 2 adapter on ALL correct traces (pool 1 + pool 2 combined)
  6. Eval Round 2 on same held-out set

Kill criteria:
  K1544: Round 1 generation yield >= 45% (Theorem 1)
  K1545: Round 1 accuracy >= 59% MMLU-Pro (= P11.F0 floor, Theorem 2)
  K1546: Round 2 yield >= Round 1 yield - 5pp (training preserves generation)
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
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_DIR = EXPERIMENT_DIR / "data"
ADAPTER_DIR_R1 = REPO_ROOT / "data" / "adapters" / "math-star-r1-v0"
ADAPTER_DIR_R2 = REPO_ROOT / "data" / "adapters" / "math-star-r2-v0"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
MMLU_PATH = REPO_ROOT / "experiments/models/exp_bench_mmlu_pro/data/test.parquet"
SEED = 42
OPTION_LETTERS = "ABCDEFGHIJ"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

# Data partitioning (per category, by stable row index)
N_GEN_PER_CAT = 1 if IS_SMOKE else 5   # Questions per cat per generation round
N_EVAL_PER_CAT = 2 if IS_SMOKE else 5  # Questions for eval

# Training config (same as P11.F0)
LORA_RANK = 8
LORA_SCALE = 1.0
LORA_DROPOUT = 0.0
LORA_KEYS = ["self_attn.v_proj", "self_attn.o_proj"]
MAX_SEQ_LEN = 8192
N_STEPS_R1 = 5 if IS_SMOKE else 200
N_STEPS_R2 = 5 if IS_SMOKE else 150
BATCH_SIZE = 1
LR = 1e-5

MAX_TOKENS = 2048  # Enough for thinking + answer


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


def strip_thinking(response):
    """Extract and strip Gemma 4 thinking tokens from response."""
    # Primary: Gemma 4 channel tags
    m = re.search(r'<\|channel>thought.*?<channel\|>', response, flags=re.DOTALL)
    if m:
        thinking = m.group(0)
        cleaned = re.sub(r'<\|channel>thought.*?<channel\|>', '', response, flags=re.DOTALL).strip()
        return cleaned, thinking
    # Fallback: <think>...</think>
    m = re.search(r'<think>(.*?)</think>', response, flags=re.DOTALL)
    if m:
        thinking = m.group(0)
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        return cleaned, thinking
    return response, ""


def parse_mcq_answer(response):
    """Extract MCQ letter answer from response."""
    answer_text, thinking = strip_thinking(response)
    for pattern in [
        r'\b([A-J])\b(?:\s*$|\s*\.|\s*\))',
        r'(?:^|\s)([A-J])(?:\s*$|\s*\.)',
        r'answer is ([A-J])',
        r'answer: ([A-J])',
    ]:
        m = re.search(pattern, answer_text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper(), thinking
    m = re.search(r'\b([A-J])\b', answer_text)
    if m:
        return m.group(1).upper(), thinking
    return None, thinking


def load_mmlu_and_partition():
    """Load MMLU-Pro and split into gen1 / gen2 / eval pools (disjoint)."""
    if not MMLU_PATH.exists():
        raise FileNotFoundError(f"MMLU-Pro data not found: {MMLU_PATH}")

    df = pd.read_parquet(MMLU_PATH)
    categories = sorted(df["category"].unique())
    log(f"Loaded MMLU-Pro: {len(df)} rows, {len(categories)} categories")

    pool_gen1, pool_gen2, pool_eval = [], [], []
    rng = np.random.default_rng(SEED)

    for cat in categories:
        cat_df = df[df["category"] == cat].reset_index(drop=True)
        n = len(cat_df)
        needed = N_GEN_PER_CAT * 2 + N_EVAL_PER_CAT
        if n < needed:
            log(f"WARNING: {cat} has only {n} rows, needed {needed}. Adjusting.")
            # Use what we have, divided equally
            per = max(1, n // 3)
        else:
            per = max(N_GEN_PER_CAT, 1)

        # Use non-overlapping slices
        idx = rng.permutation(n)
        gen1_idx = idx[:per]
        gen2_idx = idx[per:2*per]
        eval_idx = idx[2*per:2*per + N_EVAL_PER_CAT]

        pool_gen1.extend(cat_df.iloc[gen1_idx].to_dict("records"))
        pool_gen2.extend(cat_df.iloc[gen2_idx].to_dict("records"))
        pool_eval.extend(cat_df.iloc[eval_idx].to_dict("records"))

    log(f"Partitioned: gen1={len(pool_gen1)}, gen2={len(pool_gen2)}, eval={len(pool_eval)}")
    return pool_gen1, pool_gen2, pool_eval


def build_prompt(row, tokenizer):
    """Build MMLU-Pro prompt for a single row."""
    options = row.get("options", [])
    if isinstance(options, np.ndarray):
        options = options.tolist()
    n_opts = len(options)
    option_text = "\n".join(f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    user_content = (
        f"Answer the following multiple choice question. "
        f"Select the single best answer letter (A through {OPTION_LETTERS[n_opts-1]}).\n\n"
        f"Question: {row['question']}\n\nOptions:\n{option_text}\n\nAnswer:"
    )
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def generate_traces(pool, adapter_path=None, label="BASE"):
    """Generate thinking traces for a pool of questions. Returns list of dicts."""
    from mlx_lm import load, generate

    log(f"\n[Generate] {label} on {len(pool)} questions...")
    load_kwargs = {"adapter_path": str(adapter_path)} if adapter_path else {}
    model, tokenizer = load(MODEL_ID, **load_kwargs)
    log_memory(f"post-load {label}")

    results = []
    correct = 0

    for i, row in enumerate(pool):
        options = row.get("options", [])
        if isinstance(options, np.ndarray):
            options = options.tolist()
        correct_letter = OPTION_LETTERS[int(row["answer_index"])]
        n_opts = len(options)
        option_text = "\n".join(f"{OPTION_LETTERS[j]}. {opt}" for j, opt in enumerate(options))
        user_content = (
            f"Answer the following multiple choice question. "
            f"Select the single best answer letter (A through {OPTION_LETTERS[n_opts-1]}).\n\n"
            f"Question: {row['question']}\n\nOptions:\n{option_text}\n\nAnswer:"
        )
        messages = [{"role": "user", "content": user_content}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        response = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)
        pred, thinking = parse_mcq_answer(response)
        is_correct = (pred == correct_letter) if pred else False
        if is_correct:
            correct += 1

        results.append({
            "question": row["question"],
            "options": options,
            "answer_index": int(row["answer_index"]),
            "correct_letter": correct_letter,
            "predicted": pred,
            "is_correct": is_correct,
            "thinking": thinking,
            "response": response,
            "category": row.get("category", ""),
        })

        if (i + 1) % 10 == 0 or i == len(pool) - 1:
            log(f"  {i+1}/{len(pool)}: correct={correct}, yield={correct/(i+1)*100:.1f}%")

    yield_pct = correct / len(pool) * 100 if pool else 0
    log(f"[Generate] {label} done: {correct}/{len(pool)} correct = {yield_pct:.1f}%")
    cleanup(model, tokenizer)
    return results, yield_pct


def prepare_sft_data(traces_list, output_dir, label=""):
    """Format correct traces as SFT JSONL for mlx_lm.lora training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flatten all trace lists
    all_traces = []
    for traces in traces_list:
        all_traces.extend([t for t in traces if t["is_correct"]])

    log(f"[Data] {label} preparing {len(all_traces)} correct traces for SFT")

    if len(all_traces) == 0:
        log("WARNING: No correct traces — training will be empty")
        return 0

    examples = []
    for t in all_traces:
        options = t["options"]
        if isinstance(options, np.ndarray):
            options = options.tolist()
        n_opts = len(options)
        option_text = "\n".join(f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
        user_content = (
            f"Answer the following multiple choice question. "
            f"Select the single best answer letter (A through {OPTION_LETTERS[n_opts-1]}).\n\n"
            f"Question: {t['question']}\n\nOptions:\n{option_text}\n\nAnswer:"
        )
        # Assistant: include thinking + short answer
        thinking = t["thinking"]
        answer_letter = t["correct_letter"]
        if thinking:
            assistant_content = f"{thinking}\nThe answer is {answer_letter}."
        else:
            assistant_content = f"The answer is {answer_letter}."

        examples.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        })

    # Shuffle + split 90/10; ensure valid.jsonl always has >= 1 row
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(examples))
    if len(examples) < 2:
        train_idx = idx
        valid_idx = idx  # duplicate single example to satisfy mlx_lm.lora requirement
    else:
        split = max(1, int(len(examples) * 0.9))
        train_idx = idx[:split]
        valid_idx = idx[split:] if split < len(idx) else idx[:1]

    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"

    with open(train_path, "w") as f:
        for i in train_idx:
            f.write(json.dumps(examples[i]) + "\n")
    with open(valid_path, "w") as f:
        for i in valid_idx:
            f.write(json.dumps(examples[i]) + "\n")

    log(f"  Wrote {len(train_idx)} train, {len(valid_idx)} valid to {output_dir}")
    return len(train_idx)


def train_adapter(data_dir, adapter_dir, n_steps, label=""):
    """Train LoRA adapter using mlx_lm.lora."""
    import yaml
    log(f"\n[Train] {label}: {n_steps} steps from {data_dir}...")
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    lora_config_path = EXPERIMENT_DIR / f"lora_config_{label.lower().replace(' ', '_')}.yaml"
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
        "--data", str(data_dir),
        "--iters", str(n_steps),
        "--batch-size", str(BATCH_SIZE),
        "--learning-rate", str(LR),
        "--adapter-path", str(adapter_dir),
        "--save-every", str(max(50, n_steps // 4)),
        "--max-seq-length", str(MAX_SEQ_LEN),
        "--grad-checkpoint",
        "-c", str(lora_config_path),
    ]

    log(f"  Running: {' '.join(cmd[:6])} ...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        log(f"ERROR: Training failed with code {result.returncode}")
        return {"status": "failed", "time_s": round(elapsed, 1)}

    log(f"  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return {"status": "ok", "time_s": round(elapsed, 1), "steps": n_steps}


def eval_mmlu(pool_eval, adapter_path=None, label="BASE"):
    """Evaluate model on held-out eval pool."""
    from mlx_lm import load, generate

    log(f"\n[Eval] {label} on {len(pool_eval)} questions...")
    load_kwargs = {"adapter_path": str(adapter_path)} if adapter_path else {}
    model, tokenizer = load(MODEL_ID, **load_kwargs)
    log_memory(f"post-load {label}")

    correct = 0
    total_thinking_chars = 0
    per_cat = {}

    for row in pool_eval:
        options = row.get("options", [])
        if isinstance(options, np.ndarray):
            options = options.tolist()
        correct_letter = OPTION_LETTERS[int(row["answer_index"])]
        n_opts = len(options)
        option_text = "\n".join(f"{OPTION_LETTERS[j]}. {opt}" for j, opt in enumerate(options))
        user_content = (
            f"Answer the following multiple choice question. "
            f"Select the single best answer letter (A through {OPTION_LETTERS[n_opts-1]}).\n\n"
            f"Question: {row['question']}\n\nOptions:\n{option_text}\n\nAnswer:"
        )
        messages = [{"role": "user", "content": user_content}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        response = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)
        pred, thinking = parse_mcq_answer(response)
        is_correct = (pred == correct_letter) if pred else False
        if is_correct:
            correct += 1
        total_thinking_chars += len(thinking)

        cat = row.get("category", "other")
        if cat not in per_cat:
            per_cat[cat] = {"correct": 0, "total": 0}
        per_cat[cat]["total"] += 1
        if is_correct:
            per_cat[cat]["correct"] += 1

    accuracy = correct / len(pool_eval) * 100 if pool_eval else 0
    avg_thinking = total_thinking_chars / len(pool_eval) if pool_eval else 0
    log(f"[Eval] {label}: {correct}/{len(pool_eval)} = {accuracy:.1f}%, avg_thinking={avg_thinking:.0f}c")
    cleanup(model, tokenizer)
    return {
        "accuracy": round(accuracy, 2),
        "correct": correct,
        "total": len(pool_eval),
        "avg_thinking_chars": round(avg_thinking, 0),
        "per_category": {
            k: round(v["correct"] / v["total"] * 100, 1) if v["total"] > 0 else 0
            for k, v in per_cat.items()
        }
    }


def main():
    log("=" * 60)
    log("P11.I0: Synthetic Reasoning Data Generation Loop (STAR)")
    log(f"IS_SMOKE={IS_SMOKE}, N_GEN_PER_CAT={N_GEN_PER_CAT}, N_EVAL_PER_CAT={N_EVAL_PER_CAT}")
    log("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # ── Phase 0: Partition MMLU-Pro ──────────────────────────
    pool_gen1, pool_gen2, pool_eval = load_mmlu_and_partition()

    # ── Phase 1: Generate traces (base model, pool 1) ────────
    traces_r1, yield_r1 = generate_traces(pool_gen1, adapter_path=None, label="BASE-R1")
    correct_r1 = [t for t in traces_r1 if t["is_correct"]]
    results["phase1_generation"] = {
        "n_questions": len(pool_gen1),
        "n_correct": len(correct_r1),
        "yield_pct": round(yield_r1, 2),
    }

    # ── Phase 2: Train Round 1 adapter ───────────────────────
    data_r1_dir = DATA_DIR / "r1"
    n_train_r1 = prepare_sft_data([traces_r1], data_r1_dir, label="R1")
    if n_train_r1 > 0:
        train_r1 = train_adapter(data_r1_dir, ADAPTER_DIR_R1, N_STEPS_R1, label="R1")
    else:
        train_r1 = {"status": "skipped", "reason": "no correct traces"}
        log("WARNING: No correct traces for R1 training — skipping")
    results["phase2_train_r1"] = train_r1

    # ── Phase 3: Eval Round 1 ────────────────────────────────
    r1_adapter = ADAPTER_DIR_R1 if train_r1.get("status") == "ok" else None
    eval_r1 = eval_mmlu(pool_eval, adapter_path=r1_adapter, label="R1")
    results["phase3_eval_r1"] = eval_r1

    # ── Phase 4: Generate traces (R1 adapter, pool 2) ────────
    traces_r2, yield_r2 = generate_traces(pool_gen2, adapter_path=r1_adapter, label="R1-R2gen")
    correct_r2 = [t for t in traces_r2 if t["is_correct"]]
    results["phase4_generation_r2"] = {
        "n_questions": len(pool_gen2),
        "n_correct": len(correct_r2),
        "yield_pct": round(yield_r2, 2),
    }

    # ── Phase 5: Train Round 2 adapter (ALL traces combined) ─
    data_r2_dir = DATA_DIR / "r2"
    n_train_r2 = prepare_sft_data([traces_r1, traces_r2], data_r2_dir, label="R2")
    if n_train_r2 > 0:
        train_r2 = train_adapter(data_r2_dir, ADAPTER_DIR_R2, N_STEPS_R2, label="R2")
    else:
        train_r2 = {"status": "skipped", "reason": "no correct traces"}
        log("WARNING: No correct traces for R2 training — skipping")
    results["phase5_train_r2"] = train_r2

    # ── Phase 6: Eval Round 2 ────────────────────────────────
    r2_adapter = ADAPTER_DIR_R2 if train_r2.get("status") == "ok" else None
    eval_r2 = eval_mmlu(pool_eval, adapter_path=r2_adapter, label="R2")
    results["phase6_eval_r2"] = eval_r2

    # ── Kill Criteria Evaluation ─────────────────────────────
    k1544 = yield_r1 >= 45.0
    k1545 = eval_r1.get("accuracy", 0) >= 59.0
    k1546 = yield_r2 >= (yield_r1 - 5.0)

    results["kill_criteria"] = {
        "K1544_yield_ge45": {"pass": k1544, "value": round(yield_r1, 2), "threshold": 45.0},
        "K1545_r1_acc_ge59": {"pass": k1545, "value": eval_r1.get("accuracy"), "threshold": 59.0},
        "K1546_r2_yield_nondegrading": {"pass": k1546, "value": round(yield_r2, 2), "threshold": round(yield_r1 - 5.0, 2)},
    }

    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log(f"K1544 (yield >= 45%): {'PASS' if k1544 else 'FAIL'} — {yield_r1:.1f}%")
    log(f"K1545 (R1 >= 59%):    {'PASS' if k1545 else 'FAIL'} — {eval_r1.get('accuracy')}%")
    log(f"K1546 (R2 yield non-degrading): {'PASS' if k1546 else 'FAIL'} — R1={yield_r1:.1f}% → R2={yield_r2:.1f}%")
    log(f"R1 accuracy: {eval_r1.get('accuracy')}% | R2 accuracy: {eval_r2.get('accuracy')}%")
    log("=" * 60)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
