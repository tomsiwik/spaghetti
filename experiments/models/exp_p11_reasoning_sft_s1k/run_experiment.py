#!/usr/bin/env python3
"""
P11.A0: Reasoning SFT on s1K Dataset (Thinking-Compatible)

Train LoRA adapter on s1K reasoning traces WITH thinking tokens in target.
Fix: Finding #536 showed MCQ adapter suppressed thinking (0 chars) because
training targets had no <think>...</think> tokens. This experiment fixes that.

Kill criteria:
  K1490: MMLU-Pro with thinking >= 65% (baseline 62.1%)
  K1491: GSM8K >= 80% (baseline 77%)
  K1492: Adapter does NOT suppress thinking (>0 thinking chars per response)

Reference: arXiv:2501.19393 (s1: Simple Test-Time Scaling)
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
import requests

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_DIR = EXPERIMENT_DIR / "data"
ADAPTER_DIR = EXPERIMENT_DIR / "adapters" / "reasoning"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42

# Training config
LORA_RANK = 8
LORA_SCALE = 1.0
LORA_DROPOUT = 0.0
LORA_KEYS = ["self_attn.v_proj", "self_attn.o_proj"]
N_STEPS = 20 if IS_SMOKE else 1000
BATCH_SIZE = 1
LR = 1e-5
MAX_SEQ_LEN = 2048

# s1K: use examples where thinking_len + q_len + a_len fits in context
# ~3 chars per token rough estimate; budget 2048*3=6144 chars total
MAX_TOTAL_CHARS = 6000

# Eval config
EVAL_PER_CAT = 2 if IS_SMOKE else 20
GSM8K_N = 5 if IS_SMOKE else 50


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


# ─────────────────────────────────────────────
# Phase 1: Download s1K dataset
# ─────────────────────────────────────────────

def phase_download_s1k():
    """Download s1K dataset from HuggingFace parquet."""
    parquet_path = DATA_DIR / "s1k.parquet"
    if parquet_path.exists():
        log("s1K parquet already downloaded.")
        df = pd.read_parquet(parquet_path)
        log(f"Loaded {len(df)} examples.")
        return df

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log("Downloading s1K parquet from HuggingFace...")
    url = ("https://huggingface.co/datasets/simplescaling/s1K/resolve/"
           "refs%2Fconvert%2Fparquet/default/train/0000.parquet")
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()

    total = 0
    with open(parquet_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            total += len(chunk)
    log(f"Downloaded {total / 1e6:.1f} MB")

    df = pd.read_parquet(parquet_path)
    log(f"Loaded {len(df)} examples with columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
# Phase 2: Prepare training data
# ─────────────────────────────────────────────

def prepare_training_data(df):
    """
    Format s1K as thinking-compatible JSONL for mlx_lm.lora.
    Assistant content: <think>{thinking_trajectory}</think>\n\n{attempt}
    This ensures the adapter learns to USE the thinking channel (Theorem 1).
    """
    train_path = DATA_DIR / "train.jsonl"
    valid_path = DATA_DIR / "valid.jsonl"

    if train_path.exists() and valid_path.exists():
        train_count = sum(1 for _ in open(train_path))
        valid_count = sum(1 for _ in open(valid_path))
        log(f"Training data exists: {train_count} train, {valid_count} valid.")
        return train_count

    examples = []
    skipped = 0

    for _, row in df.iterrows():
        question = str(row.get("question", "")).strip()
        attempt = str(row.get("attempt", "")).strip()

        # thinking_trajectories is a list; use the first one
        thinking_traj = row.get("thinking_trajectories", [])
        if isinstance(thinking_traj, (list, np.ndarray)) and len(thinking_traj) > 0:
            thinking = str(thinking_traj[0]).strip()
        elif isinstance(thinking_traj, str):
            thinking = thinking_traj.strip()
        else:
            thinking = ""

        if not question or not attempt or not thinking:
            skipped += 1
            continue

        # Filter by total length to fit in context
        total_chars = len(question) + len(thinking) + len(attempt)
        if total_chars > MAX_TOTAL_CHARS:
            skipped += 1
            continue

        # Format: user=question, assistant=<think>thinking</think>\n\nanswer
        # This is the key fix: include thinking tokens in the target (Theorem 1)
        assistant_content = f"<think>{thinking}</think>\n\n{attempt}"

        example = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_content},
            ]
        }
        examples.append(example)

    log(f"  Kept {len(examples)} examples, skipped {skipped} (too long or missing)")

    if len(examples) < 10:
        log("WARNING: Too few examples after filtering. Relaxing length constraint.")
        examples = []
        skipped = 0
        for _, row in df.iterrows():
            question = str(row.get("question", "")).strip()
            attempt = str(row.get("attempt", "")).strip()
            thinking_traj = row.get("thinking_trajectories", [])
            if isinstance(thinking_traj, (list, np.ndarray)) and len(thinking_traj) > 0:
                thinking = str(thinking_traj[0]).strip()
            elif isinstance(thinking_traj, str):
                thinking = thinking_traj.strip()
            else:
                thinking = ""
            if not question or not attempt or not thinking:
                continue
            # Truncate thinking to fit
            max_think_chars = MAX_TOTAL_CHARS - len(question) - len(attempt) - 50
            if max_think_chars < 100:
                continue
            thinking = thinking[:max_think_chars]
            assistant_content = f"<think>{thinking}</think>\n\n{attempt}"
            examples.append({
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": assistant_content},
                ]
            })

    # Shuffle with seed
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(examples))
    examples = [examples[i] for i in idx]

    # 90/10 train/valid split
    n_valid = max(1, len(examples) // 10)
    valid_examples = examples[:n_valid]
    train_examples = examples[n_valid:]

    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(valid_path, "w") as f:
        for ex in valid_examples:
            f.write(json.dumps(ex) + "\n")

    log(f"  Wrote {len(train_examples)} train, {len(valid_examples)} valid examples")
    return len(train_examples)


# ─────────────────────────────────────────────
# Phase 3: LoRA training via mlx_lm.lora CLI
# ─────────────────────────────────────────────

def phase_train():
    """Train reasoning SFT adapter using mlx_lm.lora."""
    import yaml
    log("\n[Phase 3] LoRA training on s1K reasoning traces...")
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    # Write lora_config.yaml — mlx_lm.lora requires -c flag, not --lora-parameters
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
    log(f"  Wrote {lora_config_path}")

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", MODEL_ID,
        "--train",
        "--data", str(DATA_DIR),
        "--iters", str(N_STEPS),
        "--batch-size", str(BATCH_SIZE),
        "--learning-rate", str(LR),
        "--adapter-path", str(ADAPTER_DIR),
        "--save-every", "200" if not IS_SMOKE else "10",
        "--max-seq-length", str(MAX_SEQ_LEN),
        "--grad-checkpoint",
        "-c", str(lora_config_path),
    ]

    log(f"  Running: {' '.join(cmd[:8])} ...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        log(f"ERROR: Training failed with code {result.returncode}")
        return {"status": "failed", "time_s": elapsed}

    log(f"  Training done in {elapsed:.0f}s")
    return {"status": "ok", "time_s": round(elapsed, 1), "steps": N_STEPS}


# ─────────────────────────────────────────────
# Phase 4: Evaluate on MMLU-Pro + thinking
# ─────────────────────────────────────────────

def strip_thinking(response):
    """Extract thinking chars and clean answer from Gemma 4 response.

    CRITICAL (audit-2026-04-17 fix): Gemma 4 emits `<|channel>thought...<channel|>`
    NOT `<think>...</think>`. Original regex missed the channel format and left
    thinking content in the answer text, causing base MMLU-Pro to measure 12.5%
    (should be 62.1% — Finding #536). Both patterns supported.
    """
    if not response:
        return response or "", 0
    thinking_len = 0
    m = re.search(r'<\|channel>thought.*?<channel\|>', response, flags=re.DOTALL)
    if m:
        thinking_len = len(m.group(0))
        cleaned = re.sub(r'<\|channel>thought.*?<channel\|>', '', response, flags=re.DOTALL).strip()
        return cleaned, thinking_len
    m = re.search(r'<think>(.*?)</think>', response, flags=re.DOTALL)
    if m:
        thinking_len = len(m.group(1))
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    return cleaned, thinking_len


def parse_mcq_answer(response):
    """Extract MCQ letter from response."""
    answer_text, thinking_len = strip_thinking(response)
    # Look for single letter answer
    for pattern in [
        r'\b([A-J])\b(?:\s*$|\s*\.|\s*\))',
        r'(?:^|\s)([A-J])(?:\s*$|\s*\.)',
        r'answer is ([A-J])',
        r'answer: ([A-J])',
    ]:
        m = re.search(pattern, answer_text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper(), thinking_len
    # Last fallback: first letter found
    m = re.search(r'\b([A-J])\b', answer_text)
    if m:
        return m.group(1).upper(), thinking_len
    return None, thinking_len


def phase_eval_mmlu_pro(adapter_path=None):
    """Evaluate on MMLU-Pro with thinking mode enabled."""
    from mlx_lm import load, generate

    log(f"\n[Eval] MMLU-Pro with thinking (adapter={'YES' if adapter_path else 'BASE'})")

    # Load MMLU-Pro data from sibling experiment
    mmlu_path = EXPERIMENT_DIR.parent / "exp_bench_mmlu_pro" / "data" / "test.parquet"
    if not mmlu_path.exists():
        # Try exp_bench_mmlu_pro_thinking which may have data
        alt_path = EXPERIMENT_DIR.parent / "exp_bench_mmlu_pro_thinking" / "data"
        log(f"MMLU-Pro data not found at {mmlu_path}, checking alternatives...")
        # Fall back to downloading
        log("Downloading MMLU-Pro test data...")
        mmlu_url = ("https://datasets-server.huggingface.co/rows?"
                   "dataset=TIGER-Lab/MMLU-Pro&config=default&split=test&offset=0&length=100")
        # This is a simplified fetch; use the existing approach from other experiments
        return {"accuracy": None, "error": "MMLU-Pro data not available"}

    df = pd.read_parquet(mmlu_path)
    categories = sorted(df["category"].unique())

    # Load model
    load_kwargs = {"adapter_path": str(adapter_path)} if adapter_path else {}
    try:
        model, tokenizer = load(MODEL_ID, **load_kwargs)
    except Exception as e:
        log(f"  ERROR loading model: {e}")
        return {"accuracy": None, "error": str(e)}

    log_memory("post-load")

    correct_total = 0
    total = 0
    total_thinking_chars = 0
    per_cat = {}
    OPTION_LETTERS = "ABCDEFGHIJ"

    rng = np.random.RandomState(SEED)

    for cat in categories:
        cat_df = df[df["category"] == cat]
        n_sample = min(EVAL_PER_CAT, len(cat_df))
        sample_idx = rng.choice(len(cat_df), n_sample, replace=False)
        sample = cat_df.iloc[sample_idx]

        cat_correct = 0
        cat_thinking = 0

        for _, row in sample.iterrows():
            options = row.get("options", [])
            n_opts = len(options)
            option_text = "\n".join(f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
            correct_letter = OPTION_LETTERS[int(row["answer_index"])]

            user_content = (
                f"Answer the following multiple choice question. "
                f"Select the single best answer letter "
                f"(A through {OPTION_LETTERS[n_opts-1]}).\n\n"
                f"Question: {row['question']}\n\n"
                f"Options:\n{option_text}\n\n"
                f"Answer:"
            )

            messages = [{"role": "user", "content": user_content}]
            # CRITICAL: apply_chat_template with add_generation_prompt=True
            # Gemma 4 with <|think|> in system already supports thinking
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            max_tokens = 2048  # allow full thinking chain

            try:
                response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
                predicted, t_chars = parse_mcq_answer(response)
                cat_thinking += t_chars
                if predicted == correct_letter:
                    cat_correct += 1
            except Exception as e:
                log(f"    ERROR: {e}")

            del response
            mx.eval()

        total_thinking_chars += cat_thinking
        correct_total += cat_correct
        total += n_sample
        cat_acc = cat_correct / n_sample * 100 if n_sample > 0 else 0
        per_cat[cat] = {"correct": cat_correct, "total": n_sample, "acc": round(cat_acc, 1)}
        log(f"  {cat}: {cat_acc:.0f}% ({cat_correct}/{n_sample})")

    cleanup(model, tokenizer)

    accuracy = correct_total / total * 100 if total > 0 else 0
    avg_thinking = total_thinking_chars / total if total > 0 else 0
    log(f"  MMLU-Pro overall: {accuracy:.1f}% ({correct_total}/{total})")
    log(f"  Avg thinking chars/q: {avg_thinking:.0f}")

    return {
        "accuracy": round(accuracy, 1),
        "correct": correct_total,
        "total": total,
        "avg_thinking_chars": round(avg_thinking, 0),
        "total_thinking_chars": total_thinking_chars,
        "per_category": per_cat,
    }


# ─────────────────────────────────────────────
# Phase 5: Evaluate on GSM8K
# ─────────────────────────────────────────────

def phase_eval_gsm8k(adapter_path=None):
    """Evaluate on GSM8K with thinking mode."""
    from mlx_lm import load, generate

    log(f"\n[Eval] GSM8K (adapter={'YES' if adapter_path else 'BASE'})")

    # Download GSM8K test set via HF API
    gsm_path = DATA_DIR / "gsm8k_test.jsonl"
    if not gsm_path.exists():
        log("  Fetching GSM8K test data...")
        url = ("https://datasets-server.huggingface.co/rows?"
               "dataset=openai/gsm8k&config=main&split=test&offset=0&length=500")
        resp = requests.get(url, timeout=60)
        if not resp.ok:
            log(f"  ERROR: {resp.status_code}")
            return {"accuracy": None, "error": f"HTTP {resp.status_code}"}
        rows = resp.json().get("rows", [])
        with open(gsm_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r["row"]) + "\n")
        log(f"  Saved {len(rows)} GSM8K examples")

    gsm_data = []
    with open(gsm_path) as f:
        for line in f:
            gsm_data.append(json.loads(line.strip()))

    rng = np.random.RandomState(SEED)
    n = min(GSM8K_N, len(gsm_data))
    sample = [gsm_data[i] for i in rng.choice(len(gsm_data), n, replace=False)]

    load_kwargs = {"adapter_path": str(adapter_path)} if adapter_path else {}
    try:
        model, tokenizer = load(MODEL_ID, **load_kwargs)
    except Exception as e:
        log(f"  ERROR loading model: {e}")
        return {"accuracy": None, "error": str(e)}

    log_memory("gsm8k-post-load")

    def extract_number(text):
        """Extract final numeric answer from GSM8K response."""
        cleaned, _ = strip_thinking(text)
        # Look for #### pattern (standard GSM8K format)
        m = re.search(r'####\s*([\d,.-]+)', cleaned)
        if m:
            return m.group(1).replace(",", "").strip()
        # Last number in text
        nums = re.findall(r'-?[\d,]+\.?\d*', cleaned)
        if nums:
            return nums[-1].replace(",", "").strip()
        return None

    def get_ground_truth(answer_str):
        """Extract numeric ground truth from GSM8K answer."""
        m = re.search(r'####\s*([\d,.-]+)', answer_str)
        if m:
            return m.group(1).replace(",", "").strip()
        return answer_str.strip()

    correct = 0
    total_thinking = 0

    for item in sample:
        question = item["question"]
        gt = get_ground_truth(item["answer"])

        messages = [{"role": "user",
                     "content": f"Solve this math problem step by step.\n\n{question}\n\nAnswer:"}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        try:
            response = generate(model, tokenizer, prompt=prompt, max_tokens=2048)
            _, t_chars = strip_thinking(response)
            total_thinking += t_chars
            predicted = extract_number(response)
            if predicted and gt and predicted == gt:
                correct += 1
        except Exception as e:
            log(f"  ERROR: {e}")

        del response
        mx.eval()

    cleanup(model, tokenizer)

    accuracy = correct / n * 100 if n > 0 else 0
    log(f"  GSM8K: {accuracy:.1f}% ({correct}/{n})")
    log(f"  Avg thinking chars: {total_thinking / n:.0f}")

    return {
        "accuracy": round(accuracy, 1),
        "correct": correct,
        "total": n,
        "avg_thinking_chars": round(total_thinking / n, 0) if n > 0 else 0,
    }


# ─────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 60)
    log("P11.A0: Reasoning SFT on s1K (Thinking-Compatible)")
    log("=" * 60)
    log_memory("start")

    results = {
        "experiment": "exp_p11_reasoning_sft_s1k",
        "model": MODEL_ID,
        "smoke_test": IS_SMOKE,
        "lora_rank": LORA_RANK,
        "n_steps": N_STEPS,
    }

    # Phase 1: Download dataset
    log("\n[Phase 1] Download s1K dataset")
    df = phase_download_s1k()
    log_memory("after-download")

    # Phase 2: Prepare training data
    log("\n[Phase 2] Prepare training data")
    n_train = prepare_training_data(df)
    results["n_train"] = n_train
    log_memory("after-data-prep")
    del df
    gc.collect()

    # Phase 3: Train
    train_results = phase_train()
    results["training"] = train_results
    log_memory("after-training")

    if train_results.get("status") == "failed":
        log("Training failed; running base-only evaluation for comparison")
        adapter_path = None
    else:
        adapter_path = ADAPTER_DIR

    # Phase 4: Evaluate MMLU-Pro (base, no adapter)
    log("\n[Phase 4a] Base model MMLU-Pro evaluation")
    base_mmlu = phase_eval_mmlu_pro(adapter_path=None)
    results["base_mmlu_pro"] = base_mmlu
    log_memory("after-base-mmlu")

    # Phase 4b: Evaluate MMLU-Pro (with reasoning adapter)
    if adapter_path is not None:
        log("\n[Phase 4b] Reasoning adapter MMLU-Pro evaluation")
        adapted_mmlu = phase_eval_mmlu_pro(adapter_path=adapter_path)
        results["adapted_mmlu_pro"] = adapted_mmlu
        log_memory("after-adapted-mmlu")
    else:
        results["adapted_mmlu_pro"] = {"accuracy": None, "error": "training_failed"}

    # Phase 5a: Evaluate GSM8K (base)
    log("\n[Phase 5a] Base model GSM8K evaluation")
    base_gsm = phase_eval_gsm8k(adapter_path=None)
    results["base_gsm8k"] = base_gsm
    log_memory("after-base-gsm")

    # Phase 5b: Evaluate GSM8K (with adapter)
    if adapter_path is not None:
        log("\n[Phase 5b] Reasoning adapter GSM8K evaluation")
        adapted_gsm = phase_eval_gsm8k(adapter_path=adapter_path)
        results["adapted_gsm8k"] = adapted_gsm
        log_memory("after-adapted-gsm")
    else:
        results["adapted_gsm8k"] = {"accuracy": None, "error": "training_failed"}

    # Kill criteria evaluation
    base_mmlu_acc = results["base_mmlu_pro"].get("accuracy") or 0
    adapted_mmlu_acc = results["adapted_mmlu_pro"].get("accuracy") or 0
    base_gsm_acc = results["base_gsm8k"].get("accuracy") or 0
    adapted_gsm_acc = results["adapted_gsm8k"].get("accuracy") or 0
    adapted_thinking = results["adapted_mmlu_pro"].get("avg_thinking_chars", 0) or 0

    k1490_pass = adapted_mmlu_acc >= 65.0
    k1491_pass = adapted_gsm_acc >= 80.0
    k1492_pass = adapted_thinking > 0

    results["kill_criteria"] = {
        "K1490": {"desc": "MMLU-Pro+thinking >= 65%", "value": adapted_mmlu_acc, "pass": k1490_pass},
        "K1491": {"desc": "GSM8K >= 80%", "value": adapted_gsm_acc, "pass": k1491_pass},
        "K1492": {"desc": "thinking NOT suppressed", "value": adapted_thinking, "pass": k1492_pass},
    }

    results["total_time_s"] = round(time.time() - t0, 1)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)
    log(f"Base MMLU-Pro (thinking):    {base_mmlu_acc:.1f}%")
    log(f"Adapted MMLU-Pro (thinking): {adapted_mmlu_acc:.1f}%")
    log(f"Base GSM8K:                  {base_gsm_acc:.1f}%")
    log(f"Adapted GSM8K:               {adapted_gsm_acc:.1f}%")
    log(f"Avg thinking chars (adapter):{adapted_thinking:.0f}")
    log(f"K1490 (MMLU-Pro>=65%):      {'PASS' if k1490_pass else 'FAIL'}")
    log(f"K1491 (GSM8K>=80%):         {'PASS' if k1491_pass else 'FAIL'}")
    log(f"K1492 (thinking>0):          {'PASS' if k1492_pass else 'FAIL'}")
    log(f"Total time: {results['total_time_s']:.0f}s")
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
