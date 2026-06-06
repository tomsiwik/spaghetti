#!/usr/bin/env python3
"""
P11.F0: Train math-s1k-reasoning-v0 (Proper Context Window Fix)

Key fix over P11.A0 (exp_p11_reasoning_sft_s1k):
  - P11.A0 used MAX_TOTAL_CHARS=6000 → only 27/1000 examples (37 epochs → overfitting)
  - P11.F0 uses MAX_TOTAL_CHARS=32000 → all 1000 examples (1 epoch, proper training)
  - P11.F0 uses MAX_SEQ_LEN=8192 (Gemma 4 supports this)
  - P11.F0 fixes Gemma 4 thinking token regex (<|channel>thought...<channel|>)
  - P11.F0 fixes GSM8K loading via datasets library (not broken datasets-server API)
  - P11.F0 saves adapter to adapters/math-s1k-reasoning-v0/ for registry

Kill criteria:
  K1508: MMLU-Pro + thinking >= 59% (vs 62.1% base — Theorem 1: expected ~61-63%)
  K1509: GSM8K >= 80% (vs 77% base — Theorem 2 gradient diversity)
  K1510: Adapter registered in registry.json with eval scores

Reference: arXiv:2501.19393 (s1: Simple Test-Time Scaling)
           exp_p11_reasoning_sft_s1k PAPER.md (prior run: N=27, -26pp catastrophic forgetting)
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

# Adapter goes to canonical adapters/ directory (for registry)
ADAPTER_DIR = REPO_ROOT / "data" / "adapters" / "math-s1k-reasoning-v0"
REGISTRY_PATH = REPO_ROOT / "data" / "adapters" / "registry.json"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
SEED = 42

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

# Training config — KEY CHANGES vs P11.A0
LORA_RANK = 8
LORA_SCALE = 1.0
LORA_DROPOUT = 0.0
LORA_KEYS = ["self_attn.v_proj", "self_attn.o_proj"]
MAX_SEQ_LEN = 8192        # Was 2048 — Gemma 4 supports 8192
MAX_TOTAL_CHARS = 32000   # Was 6000 — now includes all 1000 examples
N_STEPS = 20 if IS_SMOKE else 1000
BATCH_SIZE = 1
LR = 1e-5

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


def strip_thinking(response):
    """Extract thinking chars and clean answer from Gemma 4 response.

    CRITICAL: Gemma 4 uses <|channel>thought...content...<channel|> NOT <think>...</think>
    Both patterns supported for robustness.
    """
    thinking_len = 0
    # Primary: Gemma 4 channel tags
    m = re.search(r'<\|channel>thought.*?<channel\|>', response, flags=re.DOTALL)
    if m:
        thinking_len = len(m.group(0))
        cleaned = re.sub(r'<\|channel>thought.*?<channel\|>', '', response, flags=re.DOTALL).strip()
        return cleaned, thinking_len
    # Fallback: <think>...</think> (training format)
    m = re.search(r'<think>(.*?)</think>', response, flags=re.DOTALL)
    if m:
        thinking_len = len(m.group(1))
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        return cleaned, thinking_len
    return response, 0


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


# ─────────────────────────────────────────────
# Phase 1: Load s1K dataset (already in sibling exp)
# ─────────────────────────────────────────────

def phase_load_s1k():
    """Load s1K dataset from sibling experiment (already downloaded)."""
    s1k_parquet = REPO_ROOT / "experiments/models/exp_p11_reasoning_sft_s1k/data/s1k.parquet"

    if s1k_parquet.exists():
        log(f"Loading s1K from {s1k_parquet}")
        df = pd.read_parquet(s1k_parquet)
        log(f"Loaded {len(df)} examples.")
        return df

    # Fallback: download fresh
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log("Downloading s1K parquet from HuggingFace...")
    import requests
    local_parquet = DATA_DIR / "s1k.parquet"
    url = ("https://huggingface.co/datasets/simplescaling/s1K/resolve/"
           "refs%2Fconvert%2Fparquet/default/train/0000.parquet")
    resp = requests.get(url, timeout=180, stream=True)
    resp.raise_for_status()
    with open(local_parquet, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    df = pd.read_parquet(local_parquet)
    log(f"Downloaded and loaded {len(df)} examples.")
    return df


# ─────────────────────────────────────────────
# Phase 2: Prepare training data (larger context)
# ─────────────────────────────────────────────

def prepare_training_data(df):
    """Format s1K as thinking-compatible JSONL.

    KEY CHANGE: MAX_TOTAL_CHARS=32000 (was 6000) → keeps all 1000 examples.
    With 1000 examples and 1000 steps → ~1 epoch (not 37 epochs as in P11.A0).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "train.jsonl"
    valid_path = DATA_DIR / "valid.jsonl"

    if train_path.exists() and valid_path.exists():
        train_count = sum(1 for _ in open(train_path))
        valid_count = sum(1 for _ in open(valid_path))
        log(f"Training data already prepared: {train_count} train, {valid_count} valid.")
        return train_count

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
            skipped += 1
            continue

        total_chars = len(question) + len(thinking) + len(attempt)
        if total_chars > MAX_TOTAL_CHARS:
            # Truncate thinking to fit (don't skip — keep all examples)
            max_think_chars = MAX_TOTAL_CHARS - len(question) - len(attempt) - 50
            if max_think_chars < 500:
                skipped += 1
                continue
            thinking = thinking[:max_think_chars]

        # Format with thinking tokens in target
        # Using <think>...</think> for training (standard format)
        assistant_content = f"<think>{thinking}</think>\n\n{attempt}"

        examples.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_content},
            ]
        })

    log(f"Prepared {len(examples)} examples, skipped {skipped}")

    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(examples))
    examples = [examples[i] for i in idx]

    n_valid = max(1, len(examples) // 10)
    valid_examples = examples[:n_valid]
    train_examples = examples[n_valid:]

    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    with open(valid_path, "w") as f:
        for ex in valid_examples:
            f.write(json.dumps(ex) + "\n")

    log(f"Wrote {len(train_examples)} train, {len(valid_examples)} valid examples")
    return len(train_examples)


# ─────────────────────────────────────────────
# Phase 3: LoRA training
# ─────────────────────────────────────────────

def phase_train():
    """Train reasoning SFT adapter using mlx_lm.lora with 8192 context."""
    import yaml
    log("\n[Phase 3] LoRA training (max_seq_len=8192, ~1 epoch on 1000 examples)...")
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
        "--save-every", "200" if not IS_SMOKE else "10",
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
        return {"status": "failed", "time_s": elapsed}

    log(f"  Training done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return {"status": "ok", "time_s": round(elapsed, 1), "steps": N_STEPS}


# ─────────────────────────────────────────────
# Phase 4a: Eval MMLU-Pro (base)
# ─────────────────────────────────────────────

def phase_eval_mmlu_pro(adapter_path=None, label="BASE"):
    """Evaluate on MMLU-Pro with thinking mode enabled."""
    from mlx_lm import load, generate

    log(f"\n[Eval] MMLU-Pro + thinking ({label})")
    mmlu_path = REPO_ROOT / "experiments/models/exp_bench_mmlu_pro/data/test.parquet"
    if not mmlu_path.exists():
        log(f"MMLU-Pro data not found at {mmlu_path}")
        return {"accuracy": None, "error": "data not found"}

    df = pd.read_parquet(mmlu_path)
    categories = sorted(df["category"].unique())

    load_kwargs = {"adapter_path": str(adapter_path)} if adapter_path else {}
    try:
        model, tokenizer = load(MODEL_ID, **load_kwargs)
    except Exception as e:
        log(f"ERROR loading model: {e}")
        return {"accuracy": None, "error": str(e)}

    log_memory(f"post-load {label}")

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
                f"Select the single best answer letter (A through {OPTION_LETTERS[n_opts-1]}).\n\n"
                f"Question: {row['question']}\n\nOptions:\n{option_text}\n\nAnswer:"
            )

            messages = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )

            try:
                response = generate(model, tokenizer, prompt=prompt, max_tokens=2048)
                predicted, t_chars = parse_mcq_answer(response)
                cat_thinking += t_chars
                if predicted == correct_letter:
                    cat_correct += 1
            except Exception as e:
                log(f"  Error on question: {e}")

        per_cat[cat] = {"correct": cat_correct, "total": n_sample,
                        "accuracy": cat_correct / n_sample if n_sample else 0}
        correct_total += cat_correct
        total += n_sample
        total_thinking_chars += cat_thinking
        log(f"  {cat}: {cat_correct}/{n_sample} ({cat_correct/n_sample*100:.0f}%)")

    accuracy = correct_total / total if total else 0
    avg_thinking = total_thinking_chars / total if total else 0
    log(f"  Overall: {correct_total}/{total} = {accuracy*100:.1f}%, thinking={avg_thinking:.0f} chars/q")

    cleanup(model, tokenizer)
    return {
        "accuracy": round(accuracy * 100, 1),
        "correct": correct_total,
        "total": total,
        "avg_thinking_chars": round(avg_thinking),
        "per_category": per_cat,
    }


# ─────────────────────────────────────────────
# Phase 4b: Eval GSM8K (adapter)
# ─────────────────────────────────────────────

def phase_eval_gsm8k(adapter_path=None, label="BASE"):
    """Evaluate on GSM8K using datasets library (not broken API)."""
    from mlx_lm import load, generate

    log(f"\n[Eval] GSM8K ({label})")

    try:
        from datasets import load_dataset
        gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        log(f"ERROR loading GSM8K: {e}")
        return {"accuracy": None, "error": str(e)}

    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(gsm8k), min(GSM8K_N, len(gsm8k)), replace=False)
    sample = [gsm8k[int(i)] for i in indices]

    load_kwargs = {"adapter_path": str(adapter_path)} if adapter_path else {}
    try:
        model, tokenizer = load(MODEL_ID, **load_kwargs)
    except Exception as e:
        log(f"ERROR loading model: {e}")
        return {"accuracy": None, "error": str(e)}

    log_memory(f"post-load GSM8K {label}")

    correct = 0
    total = len(sample)

    for item in sample:
        question = item["question"]
        answer_str = item["answer"]
        # Extract numeric answer from "#### N" format
        m = re.search(r"####\s*([0-9,\-\.]+)", answer_str)
        if not m:
            total -= 1
            continue
        true_ans = m.group(1).replace(",", "").strip()

        messages = [{"role": "user", "content": f"Solve step by step: {question}\nAnswer:"}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        try:
            response = generate(model, tokenizer, prompt=prompt, max_tokens=1024)
            # Look for numeric answer (last number in response)
            nums = re.findall(r"[-]?\d+(?:\.\d+)?(?:,\d{3})*", response.replace(",", ""))
            pred = nums[-1].replace(",", "") if nums else None
            if pred == true_ans:
                correct += 1
        except Exception as e:
            log(f"  Error: {e}")

    accuracy = correct / total if total else 0
    log(f"  GSM8K {label}: {correct}/{total} = {accuracy*100:.1f}%")

    cleanup(model, tokenizer)
    return {"accuracy": round(accuracy * 100, 1), "correct": correct, "total": total}


# ─────────────────────────────────────────────
# Phase 5: Register in registry.json
# ─────────────────────────────────────────────

def phase_register(mmlu_adapter_result, gsm8k_adapter_result):
    """Add or update math-s1k-reasoning-v0 in registry.json."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
    else:
        registry = {"schema_version": 1, "base_model": MODEL_ID, "adapters": []}

    # Check adapter file size
    adapter_files = list(ADAPTER_DIR.glob("*.safetensors"))
    size_mb = sum(f.stat().st_size for f in adapter_files) / 1e6

    entry = {
        "name": "math-s1k-reasoning-v0",
        "domain": "math",
        "source": "simplescaling/s1K",
        "type": "reasoning",
        "version": 0,
        "path": f"adapters/math-s1k-reasoning-v0/",
        "training": {
            "method": "sft_ntp",
            "polar": False,
            "dataset": "simplescaling/s1K",
            "n_examples": 1000,
            "steps": N_STEPS,
            "rank": LORA_RANK,
            "target_modules": LORA_KEYS,
            "thinking_enabled": True,
            "max_seq_len": MAX_SEQ_LEN,
            "experiment_id": "exp_p11_s1k_reasoning_train_eval",
        },
        "evals": {
            "mmlu_pro_thinking": mmlu_adapter_result.get("accuracy"),
            "gsm8k": gsm8k_adapter_result.get("accuracy"),
        },
        "size_mb": round(size_mb, 2),
        "created": "2026-04-14",
        "status": "reasoning",
        "notes": "s1K-1.1 reasoning adapter. Full 1000 examples (fixed from P11.A0 which had 27).",
    }

    # Remove existing entry if present
    registry["adapters"] = [a for a in registry["adapters"] if a["name"] != "math-s1k-reasoning-v0"]
    registry["adapters"].append(entry)

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    log(f"Registered math-s1k-reasoning-v0 in registry.json (size={size_mb:.1f}MB)")
    return {"status": "ok", "size_mb": round(size_mb, 2)}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    results = {}
    t_start = time.time()

    log("=" * 60)
    log("P11.F0: Train math-s1k-reasoning-v0 (Proper Context Window)")
    log(f"MAX_SEQ_LEN={MAX_SEQ_LEN}, MAX_TOTAL_CHARS={MAX_TOTAL_CHARS}")
    log(f"IS_SMOKE={IS_SMOKE}, N_STEPS={N_STEPS}")
    log("=" * 60)

    # Phase 1: Load data
    log("\n[Phase 1] Load s1K dataset")
    df = phase_load_s1k()
    results["phase1_examples"] = len(df)

    # Phase 2: Prepare training data
    log("\n[Phase 2] Prepare training data")
    n_train = prepare_training_data(df)
    results["phase2_n_train"] = n_train
    log(f"Training examples: {n_train} (epoch size ~1.0 with {N_STEPS} steps)")

    # Phase 3: Train
    log("\n[Phase 3] LoRA training")
    train_result = phase_train()
    results["phase3_train"] = train_result

    if train_result.get("status") != "ok" and not IS_SMOKE:
        log("Training failed — saving partial results")
        results["killed"] = True
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        return

    # Pre-flight: adapter weights present after training (mem-antipattern-017).
    st_files = list(ADAPTER_DIR.glob("*.safetensors"))
    st_size_mb = sum(f.stat().st_size for f in st_files) / 1e6
    if not st_files or st_size_mb < 0.1:
        log(f"FATAL: no usable adapters.safetensors after training "
            f"({len(st_files)} files, {st_size_mb:.2f} MB)")
        results["killed"] = True
        results["kill_reason"] = "adapter_weights_missing_post_save"
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        return
    log(f"[Pre-flight] adapter weights present: {len(st_files)} files, {st_size_mb:.2f} MB")

    # Phase 4a: Eval base MMLU-Pro (for delta computation)
    log("\n[Phase 4a] Base model eval — MMLU-Pro")
    mmlu_base = phase_eval_mmlu_pro(adapter_path=None, label="BASE")
    results["phase4a_mmlu_base"] = mmlu_base

    # Phase 4b: Eval adapter MMLU-Pro
    log("\n[Phase 4b] Adapter eval — MMLU-Pro")
    mmlu_adapter = phase_eval_mmlu_pro(adapter_path=ADAPTER_DIR, label="ADAPTER")
    results["phase4b_mmlu_adapter"] = mmlu_adapter

    # Phase 4c: Eval adapter GSM8K
    log("\n[Phase 4c] Adapter eval — GSM8K")
    gsm8k_adapter = phase_eval_gsm8k(adapter_path=ADAPTER_DIR, label="ADAPTER")
    results["phase4c_gsm8k_adapter"] = gsm8k_adapter

    # Kill criteria evaluation
    mmlu_acc = mmlu_adapter.get("accuracy") or 0
    gsm8k_acc = gsm8k_adapter.get("accuracy") or 0
    avg_thinking = mmlu_adapter.get("avg_thinking_chars", 0)

    k1508_pass = mmlu_acc >= 59.0
    k1509_pass = gsm8k_acc >= 80.0
    k1510_pass = False  # will be set after registration

    log("\n[Kill Criteria]")
    log(f"  K1508 (MMLU-Pro + thinking >= 59%): {mmlu_acc:.1f}% → {'PASS' if k1508_pass else 'FAIL'}")
    log(f"  K1509 (GSM8K >= 80%): {gsm8k_acc:.1f}% → {'PASS' if k1509_pass else 'FAIL'}")
    log(f"  Thinking: {avg_thinking:.0f} chars/q")

    # Phase 5: Register
    log("\n[Phase 5] Register adapter")
    reg_result = phase_register(mmlu_adapter, gsm8k_adapter)
    results["phase5_registry"] = reg_result
    k1510_pass = reg_result.get("status") == "ok"
    log(f"  K1510 (Registered): {'PASS' if k1510_pass else 'FAIL'}")

    # Summary
    elapsed = time.time() - t_start
    results["kill_criteria"] = {
        "K1508_mmlu_thinking": {"value": mmlu_acc, "threshold": 59.0, "pass": k1508_pass},
        "K1509_gsm8k": {"value": gsm8k_acc, "threshold": 80.0, "pass": k1509_pass},
        "K1510_registered": {"value": k1510_pass, "pass": k1510_pass},
    }
    results["total_time_s"] = round(elapsed)

    log(f"\n{'='*60}")
    log(f"Total time: {elapsed/60:.1f} min")
    log(f"K1508: {'PASS' if k1508_pass else 'FAIL'} ({mmlu_acc:.1f}%)")
    log(f"K1509: {'PASS' if k1509_pass else 'FAIL'} ({gsm8k_acc:.1f}%)")
    log(f"K1510: {'PASS' if k1510_pass else 'FAIL'}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
