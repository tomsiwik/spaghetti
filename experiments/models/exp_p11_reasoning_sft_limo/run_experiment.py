#!/usr/bin/env python3
"""
P11.A1: Reasoning SFT on LIMO Dataset (817 Maximally-Hard Traces)

Train LoRA adapter on LIMO reasoning traces WITH thinking tokens in target.
LIMO (arXiv:2502.03387): 817 problems selected where model barely succeeds
(1-3/32 attempts) — maximally hard but solvable, near capability boundary.

Kill criteria:
  K1493: MMLU-Pro with thinking >= 65% (baseline 62.1%)
  K1494: GSM8K >= 85% (LIMO claims 95.6% MATH500 at larger scale)
  K1495: Training completes in < 1h on M5 Pro (817 samples)

Reference: arXiv:2502.03387 (LIMO: Less is More for Reasoning)
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
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent  # llm/
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_DIR = EXPERIMENT_DIR / "data"
ADAPTER_DIR = EXPERIMENT_DIR / "adapters" / "reasoning"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42

# Training config — identical to s1K for fair comparison
LORA_RANK = 8
LORA_SCALE = 1.0
LORA_DROPOUT = 0.0
LORA_KEYS = ["self_attn.v_proj", "self_attn.o_proj"]
N_STEPS = 20 if IS_SMOKE else 817  # 1 epoch over 817 LIMO examples
BATCH_SIZE = 1
LR = 1e-5
MAX_SEQ_LEN = 2048

# LIMO solutions average ~3K chars — fit more than s1K
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
# Phase 1: Download LIMO dataset
# ─────────────────────────────────────────────

def phase_download_limo():
    """Download LIMO dataset (GAIR/LIMO) from HuggingFace."""
    parquet_path = DATA_DIR / "limo.parquet"
    if parquet_path.exists():
        log("LIMO parquet already downloaded.")
        df = pd.read_parquet(parquet_path)
        log(f"Loaded {len(df)} examples with columns: {list(df.columns)}")
        return df

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Try parquet endpoint first
    url = ("https://huggingface.co/datasets/GAIR/LIMO/resolve/"
           "refs%2Fconvert%2Fparquet/default/train/0000.parquet")
    log(f"Downloading LIMO parquet from HuggingFace...")
    try:
        resp = requests.get(url, timeout=120, stream=True)
        if resp.status_code == 200:
            total = 0
            with open(parquet_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    total += len(chunk)
            log(f"Downloaded {total / 1e6:.1f} MB")
            df = pd.read_parquet(parquet_path)
            log(f"Loaded {len(df)} examples with columns: {list(df.columns)}")
            return df
        else:
            log(f"Parquet endpoint returned {resp.status_code}, trying datasets API...")
    except Exception as e:
        log(f"Parquet download failed: {e}, trying datasets API...")

    # Fallback: HuggingFace datasets rows API
    api_url = ("https://datasets-server.huggingface.co/rows?"
               "dataset=GAIR/LIMO&config=default&split=train&offset=0&length=1000")
    resp = requests.get(api_url, timeout=120)
    if not resp.ok:
        raise RuntimeError(f"Failed to download LIMO: {resp.status_code}")

    data = resp.json()
    rows = [r["row"] for r in data.get("rows", [])]
    log(f"Downloaded {len(rows)} examples via API")

    # Convert to DataFrame and save as parquet
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path)
    log(f"Saved {len(df)} examples to {parquet_path}")
    log(f"Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
# Phase 2: Prepare training data
# ─────────────────────────────────────────────

def extract_boxed_answer(solution_text):
    """Extract the final \\boxed{...} answer from a competition math solution."""
    # Find last \boxed{...} — handles nested braces
    matches = list(re.finditer(r'\\boxed\{', solution_text))
    if not matches:
        return None
    # Take the last match and find its closing brace
    start = matches[-1].end()
    depth = 1
    pos = start
    while pos < len(solution_text) and depth > 0:
        if solution_text[pos] == '{':
            depth += 1
        elif solution_text[pos] == '}':
            depth -= 1
        pos += 1
    if depth == 0:
        return solution_text[start:pos - 1].strip()
    return None


def prepare_training_data(df):
    """
    Format LIMO as thinking-compatible JSONL for mlx_lm.lora.
    Format: <think>{solution}</think>\n\nThe answer is \\boxed{X}
    This preserves the thinking channel (Theorem 2 in MATH.md).
    """
    train_path = DATA_DIR / "train.jsonl"
    valid_path = DATA_DIR / "valid.jsonl"

    if train_path.exists() and valid_path.exists():
        train_count = sum(1 for _ in open(train_path))
        valid_count = sum(1 for _ in open(valid_path))
        log(f"Training data exists: {train_count} train, {valid_count} valid.")
        return train_count

    # Identify column names — LIMO may use different names
    cols = list(df.columns)
    log(f"  LIMO columns: {cols}")

    q_col = next((c for c in ["problem", "question", "query", "input"] if c in cols), None)
    a_col = next((c for c in ["solution", "answer", "output", "response"] if c in cols), None)

    if q_col is None or a_col is None:
        raise ValueError(f"Cannot find question/answer columns in {cols}")
    log(f"  Using q_col='{q_col}', a_col='{a_col}'")

    examples = []
    skipped = 0
    no_boxed = 0

    for _, row in df.iterrows():
        question = str(row.get(q_col, "")).strip()
        solution = str(row.get(a_col, "")).strip()

        if not question or not solution:
            skipped += 1
            continue

        # Try to extract final boxed answer
        boxed = extract_boxed_answer(solution)

        # Format: full solution as thinking, boxed answer as final output
        if boxed:
            assistant_content = f"<think>{solution}</think>\n\nThe answer is \\boxed{{{boxed}}}"
        else:
            # No boxed answer — use full solution as both thinking and output
            assistant_content = f"<think>{solution}</think>"
            no_boxed += 1

        total_chars = len(question) + len(assistant_content)
        if total_chars > MAX_TOTAL_CHARS:
            # Truncate solution to fit, keeping the end (answer extraction part)
            budget = MAX_TOTAL_CHARS - len(question) - 50
            if budget < 200:
                skipped += 1
                continue
            if boxed:
                suffix = f"</think>\n\nThe answer is \\boxed{{{boxed}}}"
                thinking_budget = budget - len(suffix)
                if thinking_budget < 100:
                    skipped += 1
                    continue
                truncated_solution = solution[:thinking_budget]
                assistant_content = f"<think>{truncated_solution}{suffix}"
            else:
                assistant_content = f"<think>{solution[:budget]}</think>"

        example = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_content},
            ]
        }
        examples.append(example)

    log(f"  Kept {len(examples)}, skipped {skipped} (empty), no_boxed={no_boxed}")

    if len(examples) < 10:
        raise RuntimeError(f"Too few examples: {len(examples)}. Check dataset format.")

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
    """Train reasoning SFT adapter using mlx_lm.lora on LIMO traces."""
    import yaml
    log("\n[Phase 3] LoRA training on LIMO reasoning traces...")
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
        "--save-every", "100" if not IS_SMOKE else "10",
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
# Eval helpers — Gemma 4 correct thinking regex
# ─────────────────────────────────────────────

def strip_thinking(response):
    """Remove Gemma 4 thinking channel tokens.

    Gemma 4 uses <|channel>thought...<channel|> for thinking,
    NOT <think>...</think>. This is the p10/p11-validated approach (62.1%).
    """
    thinking_len = 0
    # Primary: Gemma 4 channel tags
    m = re.search(r'<\|channel>thought.*?<channel\|>', response, flags=re.DOTALL)
    if m:
        thinking_len = len(m.group(0))
    cleaned = re.sub(r'<\|channel>thought.*?<channel\|>', '', response, flags=re.DOTALL)
    # Fallback: <think>...</think> (some tokenizer versions)
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    return cleaned.strip(), thinking_len


def parse_mcq_answer(response):
    """Extract MCQ letter from response."""
    answer_text, thinking_len = strip_thinking(response)
    for pattern in [
        r'\b([A-J])\b(?:\s*$|\s*\.|\s*\))',
        r'(?:^|\s)([A-J])(?:\s*$|\s*\.)',
        r'answer is ([A-J])',
        r'answer: ([A-J])',
    ]:
        m = re.search(pattern, answer_text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper(), thinking_len
    m = re.search(r'\b([A-J])\b', answer_text)
    if m:
        return m.group(1).upper(), thinking_len
    return None, thinking_len


def format_mmlu_prompt(row, tokenizer, enable_thinking=True):
    """Format an MMLU-Pro row into a prompt string."""
    OPTION_LETTERS = "ABCDEFGHIJ"
    options = row.get("options", [])
    n_opts = len(options)
    option_text = "\n".join(f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
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
        enable_thinking=enable_thinking,
    )
    return prompt


# ─────────────────────────────────────────────
# Phase 4: MMLU-Pro evaluation
# ─────────────────────────────────────────────

def load_mmlu_data():
    """Load MMLU-Pro test data from sibling experiment directories."""
    OPTION_LETTERS = "ABCDEFGHIJ"
    candidates = [
        REPO_ROOT / "experiments/models/exp_bench_mmlu_pro/data/test.parquet",
        REPO_ROOT / "experiments/models/exp_bench_mmlu_pro_thinking/data/test.parquet",
        REPO_ROOT / "experiments/models/exp_p11_baseline_eval/data/mmlu_pro_test.parquet",
    ]
    for p in candidates:
        if p.exists():
            log(f"  Loading MMLU-Pro from {p}")
            return pd.read_parquet(p)

    # Download if not available
    log("  Downloading MMLU-Pro test data...")
    url = ("https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/"
           "refs%2Fconvert%2Fparquet/default/test/0000.parquet")
    resp = requests.get(url, timeout=120, stream=True)
    if not resp.ok:
        log(f"  Download failed: {resp.status_code}")
        return None

    save_path = DATA_DIR / "mmlu_pro_test.parquet"
    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    return pd.read_parquet(save_path)


def phase_eval_mmlu_pro(adapter_path=None):
    """Evaluate on MMLU-Pro with thinking mode enabled."""
    from mlx_lm import load, generate

    OPTION_LETTERS = "ABCDEFGHIJ"
    log(f"\n[Eval] MMLU-Pro with thinking (adapter={'YES' if adapter_path else 'BASE'})")

    df = load_mmlu_data()
    if df is None:
        return {"accuracy": None, "error": "MMLU-Pro data unavailable"}

    categories = sorted(df["category"].unique())

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
    rng = np.random.RandomState(SEED)

    for cat in categories:
        cat_df = df[df["category"] == cat]
        n_sample = min(EVAL_PER_CAT, len(cat_df))
        sample_idx = rng.choice(len(cat_df), n_sample, replace=False)
        sample = cat_df.iloc[sample_idx]

        cat_correct = 0
        cat_thinking = 0

        for _, row in sample.iterrows():
            correct_letter = OPTION_LETTERS[int(row["answer_index"])]
            prompt = format_mmlu_prompt(row, tokenizer, enable_thinking=True)

            try:
                response = generate(model, tokenizer, prompt=prompt, max_tokens=2048)
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
# Phase 5: GSM8K evaluation
# ─────────────────────────────────────────────

def phase_eval_gsm8k(adapter_path=None):
    """Evaluate on GSM8K with thinking mode."""
    from mlx_lm import load, generate

    log(f"\n[Eval] GSM8K (adapter={'YES' if adapter_path else 'BASE'})")

    gsm_path = DATA_DIR / "gsm8k_test.jsonl"
    # Reuse from s1K if available
    s1k_gsm = EXPERIMENT_DIR.parent / "exp_p11_reasoning_sft_s1k" / "data" / "gsm8k_test.jsonl"
    if not gsm_path.exists() and s1k_gsm.exists():
        import shutil
        shutil.copy(s1k_gsm, gsm_path)
        log(f"  Copied GSM8K from s1K experiment")

    if not gsm_path.exists():
        log("  Fetching GSM8K test data...")
        url = ("https://datasets-server.huggingface.co/rows?"
               "dataset=openai/gsm8k&config=main&split=test&offset=0&length=500")
        resp = requests.get(url, timeout=60)
        if not resp.ok:
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
        return {"accuracy": None, "error": str(e)}

    log_memory("gsm8k-post-load")

    def extract_number(text):
        cleaned, _ = strip_thinking(text)
        m = re.search(r'####\s*([\d,.-]+)', cleaned)
        if m:
            return m.group(1).replace(",", "").strip()
        nums = re.findall(r'-?[\d,]+\.?\d*', cleaned)
        if nums:
            return nums[-1].replace(",", "").strip()
        return None

    def get_ground_truth(answer_str):
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
# Main
# ─────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 60)
    log("P11.A1: Reasoning SFT on LIMO (Thinking-Compatible)")
    log("=" * 60)
    log_memory("start")

    results = {
        "experiment": "exp_p11_reasoning_sft_limo",
        "model": MODEL_ID,
        "smoke_test": IS_SMOKE,
        "lora_rank": LORA_RANK,
        "n_steps": N_STEPS,
    }

    # Phase 1: Download dataset
    log("\n[Phase 1] Download LIMO dataset")
    df = phase_download_limo()
    results["dataset_size"] = len(df)
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

    adapter_path = None if train_results.get("status") == "failed" else ADAPTER_DIR

    # Phase 4a: Base model MMLU-Pro
    log("\n[Phase 4a] Base model MMLU-Pro evaluation")
    base_mmlu = phase_eval_mmlu_pro(adapter_path=None)
    results["base_mmlu_pro"] = base_mmlu
    log_memory("after-base-mmlu")

    # Phase 4b: Adapted MMLU-Pro
    if adapter_path is not None:
        log("\n[Phase 4b] LIMO adapter MMLU-Pro evaluation")
        adapted_mmlu = phase_eval_mmlu_pro(adapter_path=adapter_path)
        results["adapted_mmlu_pro"] = adapted_mmlu
        log_memory("after-adapted-mmlu")
    else:
        results["adapted_mmlu_pro"] = {"accuracy": None, "error": "training_failed"}

    # Phase 5a: Base GSM8K
    log("\n[Phase 5a] Base model GSM8K evaluation")
    base_gsm = phase_eval_gsm8k(adapter_path=None)
    results["base_gsm8k"] = base_gsm
    log_memory("after-base-gsm")

    # Phase 5b: Adapted GSM8K
    if adapter_path is not None:
        log("\n[Phase 5b] LIMO adapter GSM8K evaluation")
        adapted_gsm = phase_eval_gsm8k(adapter_path=adapter_path)
        results["adapted_gsm8k"] = adapted_gsm
        log_memory("after-adapted-gsm")
    else:
        results["adapted_gsm8k"] = {"accuracy": None, "error": "training_failed"}

    # Kill criteria
    base_mmlu_acc = (results["base_mmlu_pro"].get("accuracy") or 0)
    adapted_mmlu_acc = (results["adapted_mmlu_pro"].get("accuracy") or 0)
    base_gsm_acc = (results["base_gsm8k"].get("accuracy") or 0)
    adapted_gsm_acc = (results["adapted_gsm8k"].get("accuracy") or 0)
    adapted_thinking = (results["adapted_mmlu_pro"].get("avg_thinking_chars", 0) or 0)
    train_time = results["training"].get("time_s", 9999)

    k1493_pass = adapted_mmlu_acc >= 65.0
    k1494_pass = adapted_gsm_acc >= 85.0
    k1495_pass = train_time < 3600

    results["kill_criteria"] = {
        "K1493": {"desc": "MMLU-Pro+thinking >= 65%", "value": adapted_mmlu_acc, "pass": k1493_pass},
        "K1494": {"desc": "GSM8K >= 85%", "value": adapted_gsm_acc, "pass": k1494_pass},
        "K1495": {"desc": "Training < 1h", "value": round(train_time, 0), "pass": k1495_pass},
    }

    results["total_time_s"] = round(time.time() - t0, 1)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)
    log(f"Base MMLU-Pro (thinking):    {base_mmlu_acc:.1f}%")
    log(f"LIMO adapter MMLU-Pro:       {adapted_mmlu_acc:.1f}%")
    log(f"Base GSM8K:                  {base_gsm_acc:.1f}%")
    log(f"LIMO adapter GSM8K:          {adapted_gsm_acc:.1f}%")
    log(f"Avg thinking chars (adapter):{adapted_thinking:.0f}")
    log(f"Training time:               {train_time:.0f}s")
    log(f"K1493 (MMLU-Pro>=65%):       {'PASS' if k1493_pass else 'FAIL'}")
    log(f"K1494 (GSM8K>=85%):          {'PASS' if k1494_pass else 'FAIL'}")
    log(f"K1495 (Training<1h):         {'PASS' if k1495_pass else 'FAIL'}")
    log(f"Total time: {results['total_time_s']:.0f}s")
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
