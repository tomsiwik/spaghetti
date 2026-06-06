#!/usr/bin/env python3
"""exp_followup_ntp_sft_extraction_fixed — Qwen2.5-3B GSM8K re-run with fixed extraction.

Kill criteria (pre-registered, DB):
  K1547: Fixed-extraction GSM8K reproduces Qwen baseline within 2 pp at n>=200
         (acc_new >= 0.667 given Qwen2.5 Tech Report baseline 0.687).

Rationale: parent exp_competitive_benchmark reported Qwen = 36% on n=50. Published
baseline is 68.7%. This re-run uses tokenizer.apply_chat_template (chatml) + canonical
GSM8K extraction at n=200 to separate extraction/prompt bug from quant degradation.

Platform: Apple M5 Pro 48GB, MLX.
Model: mlx-community/Qwen2.5-3B-Instruct-4bit (already cached).
Decoding: greedy (temp=0), max_tokens=512.

References:
  - Qwen2.5 Technical Report (arxiv:2412.15115) Table 8: 68.7% GSM8K.
  - lm-evaluation-harness canonical GSM8K extraction: "#### <num>" first, else last num.
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-4bit"
BASELINE_ACCURACY = 0.687          # Qwen2.5 Tech Report Table 8 (fp16, chat 8-shot).
KC_PASS_THRESHOLD = 0.667          # Baseline - 2 pp (K1547 "within 2 pp").
N_SAMPLES = 200
MAX_NEW_TOKENS = 512
K1547_ID = 1547


# ------------------------------------------------------------------
# Memory safety (MLX unified memory)
# ------------------------------------------------------------------
def setup_memory():
    dev_info = mx.metal.device_info()
    total = dev_info["memory_size"]
    # Leave 8 GB headroom on 48 GB M5 Pro.
    mx.metal.set_memory_limit(total - 8 * 1024**3)
    mx.metal.set_cache_limit(2 * 1024**3)


# ------------------------------------------------------------------
# GSM8K answer extraction (canonical: lm-evaluation-harness style).
# Prefers "#### <num>" anchor, then trailing "\\boxed{...}" or "answer is ...", then
# last signed number in generation. Reject None as wrong.
# ------------------------------------------------------------------
_NUM_RE = re.compile(r"-?[\d,]*\.?\d+")


def extract_gsm8k_answer(text: str):
    if not text:
        return None
    # 1. Canonical "#### <num>" anchor.
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    # 2. \\boxed{X}  (chat-formatted CoT often uses this).
    m = re.search(r"\\boxed\{\s*(-?[\d,]*\.?\d+)\s*\}", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    # 3. "answer is: X" / "answer: X".
    m = re.search(
        r"(?:final\s+answer|the\s+answer)\s*(?:is|:)\s*\$?(-?[\d,]*\.?\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    # 4. Last numeric token in generation (canonical lm-eval-harness fallback).
    nums = _NUM_RE.findall(text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def extract_gsm8k_gold(answer_text: str):
    # GSM8K gold is formatted as "...\n#### <num>".
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", answer_text)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def is_correct(pred, gold, tol: float = 1e-4):
    if pred is None or gold is None:
        return False
    if gold == 0:
        return abs(pred) < tol
    return abs(pred - gold) / max(abs(gold), 1.0) < tol


# ------------------------------------------------------------------
# Prompt format (chatml via tokenizer.apply_chat_template)
# ------------------------------------------------------------------
def build_prompt(tokenizer, question: str) -> str:
    system = (
        "You are a helpful math assistant. Solve the problem step by step. "
        "Put your final numerical answer after '####' on the last line."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
def load_gsm8k(n: int):
    """Load GSM8K test split via hf_hub_download (datasets lib broken on py3.14)."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    path = hf_hub_download(
        "openai/gsm8k",
        "main/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    out = []
    for _, row in df.iterrows():
        gold = extract_gsm8k_gold(row["answer"])
        if gold is None:
            continue
        out.append({"question": row["question"], "gold": gold})
        if len(out) >= n:
            break
    return out


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    setup_memory()

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"Loading {MODEL_ID}")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  loaded in {time.time() - t0:.1f}s")

    problems = load_gsm8k(N_SAMPLES)
    log(f"Loaded {len(problems)} GSM8K problems")
    assert len(problems) >= N_SAMPLES, f"need {N_SAMPLES} problems, got {len(problems)}"

    sampler = make_sampler(temp=0.0)

    correct = 0
    preds = []
    t_eval = time.time()
    for i, p in enumerate(problems):
        prompt = build_prompt(tokenizer, p["question"])
        try:
            gen = generate(
                model,
                tokenizer,
                prompt,
                max_tokens=MAX_NEW_TOKENS,
                sampler=sampler,
                verbose=False,
            )
        except Exception as e:
            log(f"  gen failed at {i}: {e}")
            gen = ""
        pred = extract_gsm8k_answer(gen)
        ok = is_correct(pred, p["gold"])
        correct += int(ok)
        preds.append({"idx": i, "gold": p["gold"], "pred": pred, "correct": ok})
        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t_eval
            log(
                f"  [{i + 1:3d}/{len(problems)}] acc={correct / (i + 1):.3f} "
                f"elapsed={elapsed:.1f}s"
            )
        # Periodic memory hygiene.
        if (i + 1) % 20 == 0:
            mx.metal.clear_cache()
            gc.collect()

    accuracy = correct / len(problems)
    elapsed = time.time() - t_eval
    log(f"FINAL: {correct}/{len(problems)} = {accuracy:.3f}  ({elapsed:.1f}s)")

    # Kill criteria evaluation.
    k1547_pass = accuracy >= KC_PASS_THRESHOLD
    kill_results = [
        {
            "id": K1547_ID,
            "text": "Fixed-extraction GSM8K reproduces Qwen baseline within 2 pp at n>=200",
            "pass": bool(k1547_pass),
            "measured": float(accuracy),
            "threshold": float(KC_PASS_THRESHOLD),
            "baseline": float(BASELINE_ACCURACY),
        }
    ]
    all_pass = all(k["pass"] for k in kill_results)
    verdict = "SUPPORTED" if all_pass else "KILLED"

    results = {
        "experiment": "exp_followup_ntp_sft_extraction_fixed",
        "model": MODEL_ID,
        "n_samples": len(problems),
        "accuracy": float(accuracy),
        "correct": int(correct),
        "baseline": float(BASELINE_ACCURACY),
        "pass_threshold": float(KC_PASS_THRESHOLD),
        "kill_results": kill_results,
        "all_pass": bool(all_pass),
        "verdict": verdict,
        "is_smoke": False,
        "elapsed_seconds": float(elapsed),
        "predictions_sample": preds[:10],
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"Wrote {RESULTS_FILE}")
    log(f"verdict={verdict}  k1547_pass={k1547_pass}")


if __name__ == "__main__":
    main()
