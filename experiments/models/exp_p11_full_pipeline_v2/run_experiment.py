#!/usr/bin/env python3
"""
P11.M0: Full Pipeline v2 — Best Adapter + Plan-and-Solve + Injection Decoding

Combines validated P11 components:
  1. Best available adapter (RSD-aligned > GRPO > STAR > s1K, checked at runtime)
  2. Plan-and-Solve prompting (P1_ps template from P11.Z0)
  3. Injection decoding (Wait token at < 1500 thinking chars, from P11.Z1)

Ablation: 4 conditions × 5 MMLU-Pro cats × 7 q = 140 generations
GSM8K: 25 questions on full_pipeline condition

Kill criteria:
  K1544: full_pipeline MMLU-Pro >= 70%
  K1545: full_pipeline GSM8K >= 85%
  K1546: All components contribute: adapter >= 1pp, PS >= 1pp (injection expected ~0pp)

References:
  arXiv:2209.01510 (Plan-and-Solve Prompting)
  arXiv:2509.22230 (Reverse Speculative Decoding)
  Room Model (our finding — exclusive routing additive independence)
"""

import gc
import json
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

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

# Adapter priority list (highest to lowest quality)
ADAPTER_PRIORITY = [
    ("rsd_aligned",   REPO_ROOT / "data" / "adapters" / "math-rsd-aligned-v0"),
    ("grpo",          REPO_ROOT / "data" / "adapters" / "math-s1k-grpo-v0"),
    ("star_r2",       REPO_ROOT / "data" / "adapters" / "math-star-r1-v0"),
    ("s1k_reasoning", REPO_ROOT / "data" / "adapters" / "math-s1k-reasoning-v0"),
]

# MMLU-Pro data (sibling experiment directory)
MMLU_PATH = (REPO_ROOT / "experiments/models/exp_bench_mmlu_pro/data"
             / "test-00000-of-00001.parquet")
MMLU_ALT_PATH = (REPO_ROOT / "experiments/models/exp_p11_plan_and_solve_prompt"
                 / "data/mmlu_pro_sample.parquet")

# GSM8K data
GSM8K_PATH = REPO_ROOT / "experiments/models/exp_p11_s1k_reasoning_train_eval" / "data" / "gsm8k_test.parquet"

SEED = 42
EVAL_CATS = ["math", "physics", "biology", "history", "law"]  # 5 representative cats
EVAL_PER_CAT = 7       # 5 cats × 7q = 35 per condition
N_GSM8K = 25           # GSM8K eval questions
MAX_TOKENS = 2048
OPTION_LETTERS = list("ABCDEFGHIJ")

# Injection threshold (from P11.Z1: mean thinking=2614, but use 1500 as raised threshold)
MIN_THINKING_CHARS = 1500
INJECT_TEXT = "Wait, let me reconsider this more carefully."

# Plan-and-Solve prompt template (P1_ps from P11.Z0)
PS_PROMPT = (
    "The following is a multiple choice question. "
    "Let's first understand the problem and devise a plan to solve it. "
    "Then, let's carry out the plan and solve the problem step by step "
    "to get the correct answer.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Answer:"
)

DIRECT_PROMPT = (
    "The following is a multiple choice question. "
    "Answer with ONLY the letter of the correct option "
    "(A through {max_letter}). "
    "Do not explain.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Answer:"
)

_log_lines = []


def log(msg):
    print(msg, flush=True)
    _log_lines.append(msg)


def strip_thinking(response):
    """Remove Gemma 4 thinking channel (p10-validated regex)."""
    thinking_chars = 0
    m = re.search(r'<\|channel>thought.*?<channel\|>', response, flags=re.DOTALL)
    if m:
        thinking_chars = len(m.group(0))
    cleaned = re.sub(r'<\|channel>thought.*?<channel\|>', '', response, flags=re.DOTALL)
    return cleaned.strip(), thinking_chars


def parse_answer(response):
    """Extract answer letter from Gemma 4 response (thinking stripped)."""
    if not response:
        return None, 0
    answer_text, thinking_chars = strip_thinking(response)
    if not answer_text:
        return None, thinking_chars
    # Single letter
    if len(answer_text) == 1 and answer_text.upper() in OPTION_LETTERS:
        return answer_text.upper(), thinking_chars
    # Starts with letter + punctuation
    m = re.match(r'^([A-J])[.\s:)\-,]', answer_text)
    if m:
        return m.group(1), thinking_chars
    # "answer is X"
    m = re.search(r'(?:answer|correct)\s+(?:is\s+)?([A-J])\b', answer_text, re.IGNORECASE)
    if m:
        return m.group(1).upper(), thinking_chars
    # Last standalone letter
    matches = list(re.finditer(r'\b([A-J])\b', answer_text))
    if matches:
        return matches[-1].group(1), thinking_chars
    return None, thinking_chars


def load_mmlu_sample():
    """Load MMLU-Pro questions for target categories."""
    path = MMLU_PATH if MMLU_PATH.exists() else MMLU_ALT_PATH
    if not path.exists():
        # Try downloading via parquet URL
        try:
            url = ("https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro"
                   "/resolve/main/data/test-00000-of-00001.parquet")
            import urllib.request
            log(f"  Downloading MMLU-Pro from HuggingFace...")
            tmp_path = EXPERIMENT_DIR / "data" / "mmlu_pro_test.parquet"
            tmp_path.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(url, tmp_path)
            path = tmp_path
        except Exception as e:
            log(f"  ERROR: Cannot load MMLU-Pro data: {e}")
            sys.exit(1)
    df = pd.read_parquet(path)
    log(f"  MMLU-Pro: {len(df)} total rows, {df['category'].nunique()} categories")

    # Filter to target categories (case-insensitive substring match)
    sampled = []
    rng = np.random.default_rng(SEED)
    for cat_target in EVAL_CATS:
        cat_rows = df[df["category"].str.lower().str.contains(cat_target, na=False)]
        if len(cat_rows) == 0:
            # Try exact match
            cat_rows = df[df["category"].str.lower() == cat_target.lower()]
        if len(cat_rows) == 0:
            log(f"  WARNING: No rows for category '{cat_target}' — skipping")
            continue
        n = min(EVAL_PER_CAT, len(cat_rows))
        idx = rng.choice(len(cat_rows), n, replace=False)
        sampled.append(cat_rows.iloc[idx].copy())
        log(f"  Category '{cat_target}': {n} questions (from {len(cat_rows)} available)")

    if not sampled:
        log("  ERROR: No MMLU-Pro categories found — exiting")
        sys.exit(1)
    result = pd.concat(sampled, ignore_index=True)
    log(f"  Total MMLU sample: {len(result)} questions across {len(sampled)} categories")
    return result


def load_gsm8k():
    """Load GSM8K test questions."""
    if GSM8K_PATH.exists():
        df = pd.read_parquet(GSM8K_PATH)
        log(f"  GSM8K: {len(df)} rows loaded from {GSM8K_PATH}")
    else:
        # Download
        try:
            url = ("https://huggingface.co/datasets/openai/gsm8k"
                   "/resolve/main/main/test-00000-of-00001.parquet")
            import urllib.request
            log("  Downloading GSM8K from HuggingFace...")
            tmp = EXPERIMENT_DIR / "data" / "gsm8k_test.parquet"
            tmp.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(url, tmp)
            df = pd.read_parquet(tmp)
            log(f"  GSM8K: {len(df)} rows")
        except Exception as e:
            log(f"  WARNING: Cannot load GSM8K: {e} — skipping Phase 4")
            return None
    rng = np.random.default_rng(SEED + 100)
    idx = rng.choice(len(df), min(N_GSM8K, len(df)), replace=False)
    return df.iloc[idx].reset_index(drop=True)


def format_mmlu_prompt(row, tokenizer, use_ps=False):
    """Format MMLU-Pro question with optional PS prompt."""
    opts = row["options"]
    option_text = "\n".join(f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(opts))
    max_letter = OPTION_LETTERS[len(opts) - 1]
    if use_ps:
        content = PS_PROMPT.format(
            max_letter=max_letter,
            question=row["question"],
            options=option_text,
        )
    else:
        content = DIRECT_PROMPT.format(
            max_letter=max_letter,
            question=row["question"],
            options=option_text,
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def format_gsm8k_prompt(row, tokenizer):
    """Format GSM8K question."""
    question = row.get("question", row.get("Question", ""))
    content = (
        "Solve this math problem step by step. "
        "At the end, write 'The answer is: <number>'.\n\n"
        f"Problem: {question}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def parse_gsm8k_answer(response):
    """Extract numeric answer from GSM8K response."""
    _, thinking_chars = strip_thinking(response)
    # Look for "The answer is: X" pattern
    m = re.search(r'(?:answer\s+is\s*:?\s*)([\d,\.]+)', response, re.IGNORECASE)
    if m:
        val = m.group(1).replace(',', '')
        try:
            return float(val), thinking_chars
        except ValueError:
            pass
    # Last number in response
    nums = re.findall(r'\b(\d[\d,\.]*)\b', response)
    if nums:
        try:
            return float(nums[-1].replace(',', '')), thinking_chars
        except ValueError:
            pass
    return None, thinking_chars


def get_gsm8k_target(row):
    """Extract numeric answer from GSM8K ground truth."""
    answer = row.get("answer", row.get("Answer", ""))
    # GSM8K answers end with "#### <number>"
    m = re.search(r'####\s*([\d,\.]+)', str(answer))
    if m:
        try:
            return float(m.group(1).replace(',', ''))
        except ValueError:
            pass
    # Try last number
    nums = re.findall(r'\b(\d[\d,\.]*)\b', str(answer))
    if nums:
        try:
            return float(nums[-1].replace(',', ''))
        except ValueError:
            pass
    return None


def generate_with_injection(prompt, model, tokenizer, max_tokens=MAX_TOKENS,
                             inject_text=INJECT_TEXT, min_thinking=MIN_THINKING_CHARS):
    """
    Generate with optional thinking injection.
    If response has < min_thinking thinking chars, injects text and regenerates.
    Since Gemma 4 mean=2614 >> 1500, this rarely triggers (T3 in MATH.md).
    """
    from mlx_lm import generate
    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                        verbose=False)
    _, thinking_chars = strip_thinking(response)

    injected = False
    if thinking_chars < min_thinking:
        # Inject: append to prompt and regenerate
        injected_prompt = prompt + response + f"\n{inject_text}\n"
        response2 = generate(model, tokenizer, prompt=injected_prompt,
                             max_tokens=max_tokens, verbose=False)
        response = response + inject_text + response2
        injected = True

    return response, injected


def evaluate_mmlu(df, model, tokenizer, condition_name, use_ps=False, use_injection=False):
    """Evaluate one condition on MMLU-Pro subset."""
    from mlx_lm import generate
    total_correct = 0
    total_count = 0
    total_thinking = 0
    n_injected = 0
    category_results = {}

    for _, row in df.iterrows():
        prompt = format_mmlu_prompt(row, tokenizer, use_ps=use_ps)
        if use_injection:
            response, injected = generate_with_injection(prompt, model, tokenizer)
            n_injected += int(injected)
        else:
            response = generate(model, tokenizer, prompt=prompt,
                                max_tokens=MAX_TOKENS, verbose=False)
        pred, thinking_chars = parse_answer(response)
        label = row.get("answer", row.get("Answer", ""))
        if isinstance(label, (int, float)):
            label = OPTION_LETTERS[int(label)]
        elif isinstance(label, str) and len(label) == 1:
            label = label.upper()

        correct = (pred == label) if pred else False
        total_correct += int(correct)
        total_count += 1
        total_thinking += thinking_chars

        cat = row.get("category", "unknown")
        if cat not in category_results:
            category_results[cat] = {"correct": 0, "total": 0}
        category_results[cat]["correct"] += int(correct)
        category_results[cat]["total"] += 1

    accuracy = total_correct / total_count if total_count > 0 else 0.0
    avg_thinking = total_thinking / total_count if total_count > 0 else 0
    log(f"  [{condition_name}] acc={accuracy:.3f} ({total_correct}/{total_count}) "
        f"avg_thinking={avg_thinking:.0f}c injected={n_injected}")
    return {
        "condition": condition_name,
        "accuracy": accuracy,
        "correct": total_correct,
        "total": total_count,
        "avg_thinking_chars": avg_thinking,
        "n_injected": n_injected,
        "category_results": category_results,
    }


def evaluate_gsm8k(df, model, tokenizer):
    """Evaluate on GSM8K."""
    from mlx_lm import generate
    if df is None:
        return None
    total_correct = 0
    total_count = 0
    for _, row in df.iterrows():
        prompt = format_gsm8k_prompt(row, tokenizer)
        response = generate(model, tokenizer, prompt=prompt,
                            max_tokens=MAX_TOKENS, verbose=False)
        pred, _ = parse_gsm8k_answer(response)
        target = get_gsm8k_target(row)
        correct = False
        if pred is not None and target is not None:
            correct = abs(pred - target) < 0.5
        total_correct += int(correct)
        total_count += 1
    accuracy = total_correct / total_count if total_count > 0 else 0.0
    log(f"  [gsm8k] acc={accuracy:.3f} ({total_correct}/{total_count})")
    return {"accuracy": accuracy, "correct": total_correct, "total": total_count}


def main():
    from mlx_lm import load

    log("=" * 60)
    log("P11.M0: Full Pipeline v2")
    log("=" * 60)

    t0 = time.time()
    results = {}

    # ── Phase 1: Component inventory ────────────────────────────────
    log("\n[Phase 1] Component Inventory")

    best_adapter_name = None
    best_adapter_path = None
    for name, path in ADAPTER_PRIORITY:
        if path.exists():
            log(f"  ✓ {name}: {path}")
            if best_adapter_name is None:
                best_adapter_name = name
                best_adapter_path = path
        else:
            log(f"  ✗ {name}: NOT FOUND")

    if best_adapter_name is None:
        log("  WARNING: No adapters found — adapter_only and full_pipeline will run on base model")
    else:
        log(f"  Selected adapter: {best_adapter_name} at {best_adapter_path}")
    results["best_adapter"] = best_adapter_name

    # Load MMLU-Pro and GSM8K data
    log("\n[Phase 1] Loading data...")
    mmlu_df = load_mmlu_sample()
    gsm8k_df = load_gsm8k()

    log(f"\n[Phase 1] Budget: {len(mmlu_df)*4} MMLU-Pro + {N_GSM8K} GSM8K generations")

    # ── Phase 2: Base model (condition 1) ───────────────────────────
    log("\n[Phase 2] Condition 1: base_thinking (no adapter, direct prompt)")
    model, tokenizer = load(MODEL_ID)
    res_base = evaluate_mmlu(mmlu_df, model, tokenizer, "base_thinking",
                             use_ps=False, use_injection=False)
    results["base_thinking"] = res_base
    del model
    gc.collect()
    mx.metal.clear_cache() if hasattr(mx.metal, 'clear_cache') else None
    try:
        mx.clear_cache()
    except Exception:
        pass

    # ── Phase 3: Adapter only (condition 2) ────────────────────────
    log("\n[Phase 3] Condition 2: adapter_only (best adapter, direct prompt)")
    if best_adapter_path:
        model, tokenizer = load(MODEL_ID, adapter_path=str(best_adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)
    res_adapter = evaluate_mmlu(mmlu_df, model, tokenizer, "adapter_only",
                                use_ps=False, use_injection=False)
    results["adapter_only"] = res_adapter

    # ── Phase 4: Adapter + PS prompt (condition 3) ──────────────────
    log("\n[Phase 4] Condition 3: adapter_ps (best adapter + PS prompt)")
    res_adapter_ps = evaluate_mmlu(mmlu_df, model, tokenizer, "adapter_ps",
                                   use_ps=True, use_injection=False)
    results["adapter_ps"] = res_adapter_ps

    # ── Phase 5: Full pipeline (condition 4) ────────────────────────
    log("\n[Phase 5] Condition 4: full_pipeline (adapter + PS + injection)")
    res_full = evaluate_mmlu(mmlu_df, model, tokenizer, "full_pipeline",
                             use_ps=True, use_injection=True)
    results["full_pipeline"] = res_full

    # ── Phase 6: GSM8K on full pipeline ─────────────────────────────
    log("\n[Phase 6] GSM8K eval on full pipeline condition")
    gsm8k_result = evaluate_gsm8k(gsm8k_df, model, tokenizer)
    results["gsm8k_full_pipeline"] = gsm8k_result

    del model
    gc.collect()

    # ── Kill criteria ────────────────────────────────────────────────
    log("\n[Kill Criteria]")

    acc_full = results["full_pipeline"]["accuracy"]
    acc_adapter_ps = results["adapter_ps"]["accuracy"]
    acc_adapter = results["adapter_only"]["accuracy"]
    acc_base = results["base_thinking"]["accuracy"]
    gsm8k_acc = gsm8k_result["accuracy"] if gsm8k_result else None

    delta_adapter = acc_adapter - acc_base         # adapter vs base
    delta_ps = acc_adapter_ps - acc_adapter        # PS prompt contribution
    delta_inject = acc_full - acc_adapter_ps       # injection contribution

    log(f"  base_thinking:  {acc_base:.3f}")
    log(f"  adapter_only:   {acc_adapter:.3f}  (delta_adapter={delta_adapter:+.3f})")
    log(f"  adapter_ps:     {acc_adapter_ps:.3f}  (delta_ps={delta_ps:+.3f})")
    log(f"  full_pipeline:  {acc_full:.3f}  (delta_inject={delta_inject:+.3f})")
    if gsm8k_acc is not None:
        log(f"  gsm8k:          {gsm8k_acc:.3f}")

    k1544_pass = acc_full >= 0.70
    k1545_pass = gsm8k_acc is not None and gsm8k_acc >= 0.85
    k1546a_pass = delta_ps >= 0.01           # PS adds >= 1pp
    k1546b_pass = delta_adapter >= 0.01      # adapter adds >= 1pp
    k1546c_pass = delta_inject >= 0.01       # injection adds >= 1pp (expected FAIL per T3)
    k1546_pass = k1546a_pass and k1546b_pass and k1546c_pass

    results["kill_criteria"] = {
        "K1544": {"pass": k1544_pass, "value": acc_full, "threshold": 0.70},
        "K1545": {"pass": k1545_pass, "value": gsm8k_acc, "threshold": 0.85},
        "K1546a_ps": {"pass": k1546a_pass, "delta": delta_ps, "threshold": 0.01},
        "K1546b_adapter": {"pass": k1546b_pass, "delta": delta_adapter, "threshold": 0.01},
        "K1546c_inject": {"pass": k1546c_pass, "delta": delta_inject, "threshold": 0.01},
        "K1546_all": {"pass": k1546_pass},
    }

    log(f"\n  K1544 MMLU-Pro >= 70%: {'PASS' if k1544_pass else 'FAIL'} ({acc_full:.1%})")
    log(f"  K1545 GSM8K >= 85%:   {'PASS' if k1545_pass else 'FAIL'} ({gsm8k_acc:.1%})" if gsm8k_acc else "  K1545: N/A (no GSM8K)")
    log(f"  K1546a PS >= +1pp:    {'PASS' if k1546a_pass else 'FAIL'} ({delta_ps:+.1%})")
    log(f"  K1546b adapter >= +1pp: {'PASS' if k1546b_pass else 'FAIL'} ({delta_adapter:+.1%})")
    log(f"  K1546c inject >= +1pp: {'PASS' if k1546c_pass else 'FAIL'} ({delta_inject:+.1%}) [expected FAIL, T3]")
    log(f"  K1546 all components: {'PASS' if k1546_pass else 'FAIL'}")

    results["elapsed_seconds"] = time.time() - t0
    results["log"] = _log_lines

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {results['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
