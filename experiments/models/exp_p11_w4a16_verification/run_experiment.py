"""
P11.K1: W4A16 Precision and Reasoning Chain Quality on Gemma 4

Kill criteria:
  K1538: Identify exact quantization scheme of mlx-community/gemma-4-e4b-it-4bit
  K1539: If W4A4, measure gap to W4A16 (expected: significant for reasoning)
  K1540: 8-bit model thinking score >= 4-bit + 5pp (confirms quantization hurts reasoning)

Paper: 2504.04823 — W4A16 achieves near-lossless reasoning quantization.
"""

import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent

MODEL_4BIT = "mlx-community/gemma-4-e4b-it-4bit"
MODEL_8BIT = "mlx-community/gemma-4-e4b-it-8bit"

SEED = 42
EVAL_PER_CAT = 20  # 20 × 14 = 280 questions, same as p10/p11 experiments
OPTION_LETTERS = "ABCDEFGHIJ"

IS_SMOKE = "--smoke" in sys.argv


def log(msg):
    print(msg, flush=True)


def log_memory(label):
    info = mx.metal.get_active_memory() / 1e9
    peak = mx.metal.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={info:.2f}GB peak={peak:.2f}GB")
    mx.metal.reset_peak_memory()


# ─────────────────────────────────────────────
# Phase 1: Config check (K1538, K1539)
# ─────────────────────────────────────────────

def phase_config_check():
    """Verify quantization config of 4-bit and 8-bit models. Answers K1538."""
    from huggingface_hub import hf_hub_download
    import json as _json

    log("\n[Phase 1] Quantization config check")

    results = {}

    # Check 4-bit model
    try:
        cfg_path = hf_hub_download(repo_id=MODEL_4BIT, filename="config.json")
        with open(cfg_path) as f:
            cfg = _json.load(f)
        q = cfg.get("quantization", cfg.get("quantization_config", {}))
        dtype = cfg.get("dtype", "unknown")
        results["model_4bit"] = {
            "model_id": MODEL_4BIT,
            "bits": q.get("bits"),
            "group_size": q.get("group_size"),
            "mode": q.get("mode"),
            "activation_dtype": dtype,
            "is_w4a16": q.get("bits") == 4 and dtype in ("bfloat16", "float16"),
        }
        r = results["model_4bit"]
        log(f"  4-bit model: bits={r['bits']}, group_size={r['group_size']}, "
            f"mode={r['mode']}, act_dtype={r['activation_dtype']}, "
            f"W4A16={r['is_w4a16']}")
    except Exception as e:
        log(f"  ERROR checking 4-bit config: {e}")
        results["model_4bit"] = {"error": str(e)}

    # Check 8-bit model
    try:
        cfg_path = hf_hub_download(repo_id=MODEL_8BIT, filename="config.json")
        with open(cfg_path) as f:
            cfg = _json.load(f)
        q = cfg.get("quantization", cfg.get("quantization_config", {}))
        dtype = cfg.get("dtype", "unknown")
        results["model_8bit"] = {
            "model_id": MODEL_8BIT,
            "bits": q.get("bits"),
            "group_size": q.get("group_size"),
            "mode": q.get("mode"),
            "activation_dtype": dtype,
            "is_w8a16": q.get("bits") == 8 and dtype in ("bfloat16", "float16"),
        }
        r = results["model_8bit"]
        log(f"  8-bit model: bits={r['bits']}, group_size={r['group_size']}, "
            f"mode={r['mode']}, act_dtype={r['activation_dtype']}, "
            f"W8A16={r['is_w8a16']}")
    except Exception as e:
        log(f"  ERROR checking 8-bit config: {e}")
        results["model_8bit"] = {"error": str(e)}

    # K1538 verdict
    m4 = results.get("model_4bit", {})
    k1538_pass = m4.get("is_w4a16", False)
    # K1539: N/A if W4A16 (no activation quantization)
    k1539_status = "N/A" if k1538_pass else "MEASURE_NEEDED"
    log(f"  K1538 (W4A16 confirmed): {'PASS' if k1538_pass else 'FAIL'}")
    log(f"  K1539 (W4A4 gap measure): {k1539_status}")

    results["k1538_pass"] = k1538_pass
    results["k1539_status"] = k1539_status
    return results


# ─────────────────────────────────────────────
# MMLU-Pro prompt and parsing (from p10, validated at 62.1%)
# ─────────────────────────────────────────────

def format_mmlu_prompt(row, tokenizer, enable_thinking=True):
    """Format MMLU-Pro question. Based on p10 format validated at 62.1%."""
    options = row["options"]
    option_text = "\n".join(
        f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)
    )
    content = (
        f"The following is a multiple choice question. "
        f"Answer with ONLY the letter of the correct option "
        f"(A through {OPTION_LETTERS[len(options)-1]}). "
        f"Do not explain.\n\n"
        f"Question: {row['question']}\n\n"
        f"Options:\n{option_text}\n\n"
        f"Answer:"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def strip_thinking(response):
    """Remove Gemma 4 thinking channel tokens.

    Gemma 4 uses <|channel>thought...content...<channel|> for thinking,
    NOT <think>...</think>. This is the p10-validated approach (62.1%).
    """
    if not response:
        return response, 0
    thinking_len = 0
    # Primary: Gemma 4 channel tags
    m = re.search(r'<\|channel>thought.*?<channel\|>', response, flags=re.DOTALL)
    if m:
        thinking_len = len(m.group(0))
    cleaned = re.sub(r'<\|channel>thought.*?<channel\|>', '', response, flags=re.DOTALL)
    # Fallback: <think>...</think> (some tokenizer versions)
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    return cleaned.strip(), thinking_len


def parse_answer(response):
    """Extract answer letter from model response."""
    if not response:
        return None, 0
    answer_text, thinking_chars = strip_thinking(response)
    if not answer_text:
        return None, thinking_chars
    # Direct single letter
    if len(answer_text) == 1 and answer_text.upper() in OPTION_LETTERS:
        return answer_text.upper(), thinking_chars
    # Starts with letter followed by delimiter
    m = re.match(r"^([A-J])[.\s:)\-,]", answer_text)
    if m:
        return m.group(1), thinking_chars
    # "The answer is X" pattern
    m = re.search(r"(?:answer|correct)\s+(?:is\s+)?([A-J])", answer_text, re.IGNORECASE)
    if m:
        return m.group(1).upper(), thinking_chars
    # Last letter in text
    m = re.search(r"([A-J])", answer_text)
    if m:
        return m.group(1).upper(), thinking_chars
    return None, thinking_chars


# ─────────────────────────────────────────────
# Phase 2/3: MMLU-Pro eval with thinking
# ─────────────────────────────────────────────

def phase_eval_mmlu_pro(model_id, label):
    """Evaluate a model on MMLU-Pro with thinking. Returns accuracy + thinking stats."""
    from mlx_lm import load, generate

    log(f"\n[Phase eval] {label} ({model_id})")

    # Load MMLU-Pro data
    mmlu_path = REPO_ROOT / "experiments/models/exp_bench_mmlu_pro/data/test.parquet"
    if not mmlu_path.exists():
        # Try thinking experiment data
        alt = REPO_ROOT / "experiments/models/exp_bench_mmlu_pro_thinking/data/test.parquet"
        if alt.exists():
            mmlu_path = alt
        else:
            log("  ERROR: MMLU-Pro data not found")
            return {"error": "MMLU-Pro data not found", "accuracy": None}

    df = pd.read_parquet(mmlu_path)
    categories = sorted(df["category"].unique())
    log(f"  Loaded {len(df)} questions, {len(categories)} categories")

    # Load model
    try:
        model, tokenizer = load(model_id)
    except Exception as e:
        log(f"  ERROR loading {model_id}: {e}")
        return {"error": str(e), "accuracy": None}
    log_memory("post-load")

    rng = np.random.RandomState(SEED)
    correct_total = 0
    total = 0
    total_thinking_chars = 0
    per_cat = {}
    t_start = time.time()

    n_smoke = 2  # questions per category in smoke mode
    for cat in categories:
        cat_df = df[df["category"] == cat]
        n_sample = min(n_smoke if IS_SMOKE else EVAL_PER_CAT, len(cat_df))
        idxs = rng.choice(len(cat_df), n_sample, replace=False)
        sample = cat_df.iloc[idxs]

        cat_correct = 0
        cat_thinking = 0

        for _, row in sample.iterrows():
            correct_letter = OPTION_LETTERS[int(row["answer_index"])]
            prompt = format_mmlu_prompt(row, tokenizer, enable_thinking=True)
            max_tokens = 512 if IS_SMOKE else 2048

            try:
                response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
                predicted, t_chars = parse_answer(response)
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
        cat_acc = cat_correct / n_sample * 100 if n_sample else 0
        per_cat[cat] = {
            "correct": cat_correct,
            "total": n_sample,
            "acc": round(cat_acc, 1),
            "avg_thinking_chars": round(cat_thinking / n_sample, 0) if n_sample else 0,
        }
        log(f"  {cat}: {cat_acc:.0f}% ({cat_correct}/{n_sample})")

    elapsed = time.time() - t_start
    overall_acc = correct_total / total * 100 if total else 0
    avg_thinking = total_thinking_chars / total if total else 0

    log(f"  OVERALL: {overall_acc:.1f}% ({correct_total}/{total})")
    log(f"  Avg thinking chars/q: {avg_thinking:.0f}")
    log(f"  Elapsed: {elapsed:.0f}s")
    log_memory("post-eval")

    del model, tokenizer
    mx.metal.clear_cache()
    mx.eval()

    return {
        "model_id": model_id,
        "label": label,
        "accuracy": round(overall_acc / 100, 4),
        "accuracy_pct": round(overall_acc, 2),
        "correct": correct_total,
        "total": total,
        "avg_thinking_chars": round(avg_thinking, 0),
        "elapsed_s": round(elapsed, 1),
        "per_category": per_cat,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    log("=" * 60)
    log("P11.K1: W4A16 Verification")
    log(f"Smoke mode: {IS_SMOKE}")
    log("=" * 60)

    results = {}

    # Phase 1: Config check (K1538, K1539)
    results["config"] = phase_config_check()

    # Phase 2: 4-bit model (W4A16)
    results["model_4bit"] = phase_eval_mmlu_pro(MODEL_4BIT, "4-bit W4A16")

    # Phase 3: 8-bit model (W8A16)
    results["model_8bit"] = phase_eval_mmlu_pro(MODEL_8BIT, "8-bit W8A16")

    # K1540 verdict
    acc_4bit = results["model_4bit"].get("accuracy_pct")
    acc_8bit = results["model_8bit"].get("accuracy_pct")
    gap = None
    k1540_pass = None
    if acc_4bit is not None and acc_8bit is not None:
        gap = round(acc_8bit - acc_4bit, 2)
        k1540_pass = gap >= 5.0
        log(f"\n[K1540] 8-bit ({acc_8bit:.1f}%) - 4-bit ({acc_4bit:.1f}%) = {gap:.1f}pp")
        log(f"  K1540 {'PASS (quantization hurts reasoning!)' if k1540_pass else 'FAIL (W4A16 near-lossless)'}")

    results["summary"] = {
        "k1538_pass": results["config"].get("k1538_pass"),
        "k1539_status": results["config"].get("k1539_status"),
        "k1540_pass": k1540_pass,
        "accuracy_4bit_pct": acc_4bit,
        "accuracy_8bit_pct": acc_8bit,
        "gap_8bit_minus_4bit_pp": gap,
        "avg_thinking_4bit": results["model_4bit"].get("avg_thinking_chars"),
        "avg_thinking_8bit": results["model_8bit"].get("avg_thinking_chars"),
    }

    # Save results
    out_path = EXPERIMENT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n[Results saved to {out_path}]")

    # Final summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    for k, v in results["summary"].items():
        log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
