#!/usr/bin/env python3
"""
P11.E0: Baseline Evaluation Suite — Base Model + All Existing Adapters

Evaluates every adapter in registry.json on:
  - MMLU-Pro (20 per category, 14 cats = 280 questions) with thinking ON and OFF
  - GSM8K (50 questions) with thinking ON

Updates registry.json with eval scores.

Kill criteria:
  K1505: All 5 adapters evaluated on GSM8K + MMLU-Pro (thinking ON and OFF)
  K1506: Base model evaluated (thinking ON target ~62.1%, matching Finding #530)
  K1507: registry.json updated with all eval scores
"""

import gc
import json
import os
import re
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
REGISTRY_PATH = REPO_ROOT / "data" / "adapters" / "registry.json"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
SEED = 42

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
BASE_ONLY = os.environ.get("BASE_ONLY", "0") == "1"
EVAL_PER_CAT = 2 if IS_SMOKE else 20
GSM8K_N = 5 if IS_SMOKE else 50
THINKING_MAX_TOKENS = 2048

MMLU_PATH = REPO_ROOT / "experiments/models/exp_bench_mmlu_pro/data/test.parquet"

# All adapters to evaluate (from registry.json paths)
ADAPTERS = [
    {
        "name": "base",
        "path": None,
        "domain": "base",
    },
    {
        "name": "math-gsm8k-knowledge-v0",
        "path": REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/math",
        "domain": "math",
    },
    {
        "name": "code-codealpaca-knowledge-v0",
        "path": REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/code",
        "domain": "code",
    },
    {
        "name": "medical-medmcqa-knowledge-v0",
        "path": REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/medical",
        "domain": "medical",
    },
    {
        "name": "legal-mmlu-knowledge-v0",
        "path": REPO_ROOT / "experiments/models/exp_p1_t2_multi_domain_5/adapters/legal",
        "domain": "legal",
    },
    {
        "name": "finance-mmlu-knowledge-v0",
        "path": REPO_ROOT / "experiments/models/exp_p1_t2_multi_domain_5/adapters/finance",
        "domain": "finance",
    },
]


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
    # Gemma 4 thinking: <|channel>thought\n...<channel|> (primary)
    # Fallback: <think>...</think> (generic)
    if not response:
        return response or "", 0
    thinking_len = 0
    m = re.search(r'<\|channel>thought.*?<channel\|>', response, flags=re.DOTALL)
    if m:
        thinking_len = len(m.group(0))
    cleaned = re.sub(r'<\|channel>thought.*?<channel\|>', '', response, flags=re.DOTALL)
    if thinking_len == 0:
        m2 = re.search(r'<think>(.*?)</think>', cleaned, flags=re.DOTALL)
        if m2:
            thinking_len = len(m2.group(1))
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
    return cleaned, thinking_len


def parse_mcq_answer(response):
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


def eval_mmlu_pro(model, tokenizer, thinking: bool) -> dict:
    """Evaluate on MMLU-Pro. Returns accuracy + thinking stats."""
    from mlx_lm import generate

    df = pd.read_parquet(MMLU_PATH)
    categories = sorted(df["category"].unique())
    OPTION_LETTERS = "ABCDEFGHIJ"
    rng = np.random.RandomState(SEED)

    correct_total = 0
    total = 0
    total_thinking_chars = 0
    per_cat = {}

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

            if thinking:
                user_content = (
                    f"Answer the following multiple choice question. "
                    f"Select the single best answer letter "
                    f"(A through {OPTION_LETTERS[n_opts-1]}).\n\n"
                    f"Question: {row['question']}\n\n"
                    f"Options:\n{option_text}\n\n"
                    f"Answer:"
                )
            else:
                user_content = (
                    f"The following is a multiple choice question. "
                    f"Answer with ONLY the letter of the correct option "
                    f"(A through {OPTION_LETTERS[n_opts-1]}). "
                    f"Do not explain.\n\n"
                    f"Question: {row['question']}\n\n"
                    f"Options:\n{option_text}\n\n"
                    f"Answer:"
                )

            messages = [{"role": "user", "content": user_content}]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking,
                )
            except Exception:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )

            try:
                response = generate(model, tokenizer, prompt=prompt, max_tokens=THINKING_MAX_TOKENS if thinking else 16)
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

    accuracy = correct_total / total * 100 if total > 0 else 0
    avg_thinking = total_thinking_chars / total if total > 0 else 0
    log(f"  MMLU-Pro: {accuracy:.1f}% ({correct_total}/{total}), "
        f"avg_thinking={avg_thinking:.0f} chars")

    return {
        "accuracy": round(accuracy, 1),
        "correct": correct_total,
        "total": total,
        "avg_thinking_chars": round(avg_thinking, 0),
        "total_thinking_chars": total_thinking_chars,
        "per_category": per_cat,
    }


def eval_gsm8k(model, tokenizer, thinking: bool) -> dict:
    """Evaluate on GSM8K. Returns accuracy."""
    from mlx_lm import generate
    import requests

    gsm_path = EXPERIMENT_DIR / "data" / "gsm8k_test.jsonl"
    gsm_path.parent.mkdir(parents=True, exist_ok=True)
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
        try:
            if thinking:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=True,
                )
            else:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        except Exception:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        try:
            response = generate(model, tokenizer, prompt=prompt,
                                max_tokens=THINKING_MAX_TOKENS if thinking else 512)
            _, t_chars = strip_thinking(response)
            total_thinking += t_chars
            predicted = extract_number(response)
            if predicted and gt and predicted == gt:
                correct += 1
        except Exception as e:
            log(f"  ERROR: {e}")

        del response
        mx.eval()

    accuracy = correct / n * 100 if n > 0 else 0
    log(f"  GSM8K: {accuracy:.1f}% ({correct}/{n})")

    return {
        "accuracy": round(accuracy, 1),
        "correct": correct,
        "total": n,
        "avg_thinking_chars": round(total_thinking / n, 0) if n > 0 else 0,
    }


def eval_adapter(adapter_info: dict) -> dict:
    """Run full eval suite for one adapter. Returns result dict."""
    from mlx_lm import load

    name = adapter_info["name"]
    adapter_path = adapter_info["path"]

    log(f"\n{'='*60}")
    log(f"Evaluating: {name}")
    if adapter_path:
        log(f"  Adapter: {adapter_path}")
        if not Path(adapter_path).exists():
            log(f"  ERROR: adapter path does not exist!")
            return {"name": name, "error": "path_not_found"}
    else:
        log("  Base model (no adapter)")
    log('='*60)

    load_kwargs = {"adapter_path": str(adapter_path)} if adapter_path else {}
    try:
        model, tokenizer = load(MODEL_ID, **load_kwargs)
    except Exception as e:
        log(f"  ERROR loading: {e}")
        return {"name": name, "error": str(e)}

    log_memory(f"{name}-post-load")
    result = {"name": name}

    # MMLU-Pro without thinking
    log(f"\n[{name}] MMLU-Pro (thinking=OFF)")
    result["mmlu_pro"] = eval_mmlu_pro(model, tokenizer, thinking=False)
    log_memory(f"{name}-after-mmlu-nothink")

    # MMLU-Pro with thinking
    log(f"\n[{name}] MMLU-Pro (thinking=ON)")
    result["mmlu_pro_thinking"] = eval_mmlu_pro(model, tokenizer, thinking=True)
    log_memory(f"{name}-after-mmlu-think")

    # GSM8K with thinking
    log(f"\n[{name}] GSM8K (thinking=ON)")
    result["gsm8k"] = eval_gsm8k(model, tokenizer, thinking=True)
    log_memory(f"{name}-after-gsm8k")

    cleanup(model, tokenizer)
    return result


def update_registry(all_results: list):
    """Update registry.json with eval scores from this run."""
    if not REGISTRY_PATH.exists():
        log(f"WARNING: registry.json not found at {REGISTRY_PATH}")
        return False

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    today = time.strftime("%Y-%m-%d")
    result_by_name = {r["name"]: r for r in all_results}

    for adapter in registry["adapters"]:
        name = adapter["name"]
        if name not in result_by_name:
            continue
        res = result_by_name[name]
        if "error" in res:
            continue
        evals = adapter.setdefault("evals", {})
        if "mmlu_pro" in res:
            evals["mmlu_pro"] = {
                "score": res["mmlu_pro"]["accuracy"],
                "n": res["mmlu_pro"]["total"],
                "thinking": False,
                "date": today,
                "experiment_id": "exp_p11_baseline_eval",
            }
        if "mmlu_pro_thinking" in res:
            evals["mmlu_pro_thinking"] = {
                "score": res["mmlu_pro_thinking"]["accuracy"],
                "n": res["mmlu_pro_thinking"]["total"],
                "thinking": True,
                "avg_thinking_chars": res["mmlu_pro_thinking"].get("avg_thinking_chars", 0),
                "date": today,
                "experiment_id": "exp_p11_baseline_eval",
            }
        if "gsm8k" in res:
            evals["gsm8k"] = {
                "score": res["gsm8k"]["accuracy"],
                "n": res["gsm8k"]["total"],
                "thinking": True,
                "date": today,
                "experiment_id": "exp_p11_baseline_eval",
            }

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    log(f"  Updated {REGISTRY_PATH}")
    return True


def main():
    t0 = time.time()
    log("=" * 60)
    log("P11.E0: Baseline Evaluation Suite")
    log("=" * 60)
    log_memory("start")

    all_results = []

    adapters_to_run = [a for a in ADAPTERS if a["name"] == "base"] if BASE_ONLY else ADAPTERS
    log(f"Running {len(adapters_to_run)} adapter(s); BASE_ONLY={BASE_ONLY}, IS_SMOKE={IS_SMOKE}")

    for adapter_info in adapters_to_run:
        t_start = time.time()
        result = eval_adapter(adapter_info)
        result["elapsed_s"] = round(time.time() - t_start, 1)
        all_results.append(result)

        # Save intermediate results after each adapter
        RESULTS_FILE.write_text(json.dumps({
            "experiment": "exp_p11_baseline_eval",
            "partial": True,
            "results": all_results,
        }, indent=2))
        log(f"  [checkpoint] Saved intermediate results.")

    # Finalize
    results = {
        "experiment": "exp_p11_baseline_eval",
        "model": MODEL_ID,
        "smoke_test": IS_SMOKE,
        "eval_per_cat": EVAL_PER_CAT,
        "gsm8k_n": GSM8K_N,
        "results": all_results,
        "total_time_s": round(time.time() - t0, 1),
    }

    # Kill criteria check
    base_res = next((r for r in all_results if r["name"] == "base"), {})
    base_mmlu_think = base_res.get("mmlu_pro_thinking", {}).get("accuracy", 0) or 0
    adapters_complete = sum(
        1 for r in all_results
        if r["name"] != "base"
        and "error" not in r
        and "mmlu_pro" in r
        and "gsm8k" in r
    )

    k1505_pass = adapters_complete >= 5
    k1506_pass = abs(base_mmlu_think - 62.1) <= 5.0  # within 5pp of expected
    k1507_pass = update_registry(all_results)

    results["kill_criteria"] = {
        "K1505": {"desc": "All 5 adapters evaluated", "value": adapters_complete, "pass": k1505_pass},
        "K1506": {"desc": "Base MMLU-Pro+thinking ~62.1%", "value": base_mmlu_think, "pass": k1506_pass},
        "K1507": {"desc": "registry.json updated", "value": k1507_pass, "pass": k1507_pass},
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)
    for r in all_results:
        name = r["name"]
        mmlu_no = r.get("mmlu_pro", {}).get("accuracy", "N/A")
        mmlu_yes = r.get("mmlu_pro_thinking", {}).get("accuracy", "N/A")
        gsm = r.get("gsm8k", {}).get("accuracy", "N/A")
        log(f"  {name:35s} MMLU(no/yes)={mmlu_no}/{mmlu_yes}  GSM8K={gsm}")

    log(f"\nK1505 (5 adapters done): {'PASS' if k1505_pass else 'FAIL'} ({adapters_complete}/5)")
    log(f"K1506 (base MMLU ~62%):  {'PASS' if k1506_pass else 'FAIL'} ({base_mmlu_think:.1f}%)")
    log(f"K1507 (registry updated): {'PASS' if k1507_pass else 'FAIL'}")
    log(f"Total time: {results['total_time_s']:.0f}s")
    log(f"Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
