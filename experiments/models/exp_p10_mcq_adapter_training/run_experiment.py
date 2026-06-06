#!/usr/bin/env python3
"""
P10.C0: MCQ-Format Adapter (Fix NTP Degradation on Benchmarks)

Train standard LoRA adapter with MCQ classification loss on MMLU-Pro data.
Test with and without thinking mode. Compare to base model.

Kill criteria:
  K1470: MCQ adapter + thinking >= 65% MMLU-Pro
  K1471: MCQ adapter does NOT degrade generative quality (HumanEval within 5pp)
  K1472: MCQ mixed training completes in < 30 min on M5 Pro

Grounded by:
  Finding #517 — NTP adapters degrade MCQ (-6.2pp on MMLU-Pro)
  Finding #522 — MCQ classification loss +14.5pp under TT-LoRA r6
  Finding #528 — Thinking mode zero benefit on GPQA Diamond (4-bit)
"""

import gc
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
ADAPTER_DIR = EXPERIMENT_DIR / "adapters" / "mcq"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
DATA_FILE = EXPERIMENT_DIR.parent / "exp_bench_mmlu_pro" / "data" / "test.parquet"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42
OPTION_LETTERS = "ABCDEFGHIJ"

# Training config
LORA_RANK = 6
LORA_SCALE = 1.0  # alpha/rank ratio
LORA_DROPOUT = 0.0
LORA_KEYS = ["self_attn.v_proj", "self_attn.o_proj"]
N_STEPS = 20 if IS_SMOKE else 500
BATCH_SIZE = 2
LR = 2e-4
MCQ_WEIGHT = 1.0  # λ for MCQ loss
MAX_SEQ_LEN = 384

# Eval config
EVAL_PER_CAT = 3 if IS_SMOKE else 50
THINKING_PER_CAT = 2 if IS_SMOKE else 20
HUMANEVAL_N = 3 if IS_SMOKE else 20


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


# ────────────────────────────────────────────────
# Data Loading & Splitting
# ────────────────────────────────────────────────

def load_and_split_data():
    """Load MMLU-Pro, stratified split 80/20."""
    df = pd.read_parquet(DATA_FILE)
    train_dfs, eval_dfs = [], []
    rng = np.random.RandomState(SEED)

    for cat in sorted(df["category"].unique()):
        cat_df = df[df["category"] == cat].copy()
        idx = rng.permutation(len(cat_df))
        split = int(0.8 * len(cat_df))
        train_dfs.append(cat_df.iloc[idx[:split]])
        eval_dfs.append(cat_df.iloc[idx[split:]])

    train_df = pd.concat(train_dfs, ignore_index=True)
    eval_df = pd.concat(eval_dfs, ignore_index=True)
    log(f"  Data split: {len(train_df)} train, {len(eval_df)} eval "
        f"across {df['category'].nunique()} categories")
    return train_df, eval_df


# ────────────────────────────────────────────────
# Training Data Preparation
# ────────────────────────────────────────────────

def prepare_training_examples(train_df, tokenizer):
    """Convert MMLU-Pro questions to training format with MCQ targets."""
    answer_to_idx = {letter: i for i, letter in enumerate(OPTION_LETTERS)}
    examples = []

    for _, row in train_df.iterrows():
        options = row["options"]
        n_options = len(options)
        correct_letter = row["answer"]
        correct_idx = answer_to_idx.get(correct_letter, -1)
        if correct_idx == -1 or correct_idx >= n_options:
            continue

        option_text = "\n".join(
            f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)
        )
        user_content = (
            f"The following is a multiple choice question. "
            f"Answer with ONLY the letter of the correct option "
            f"(A through {OPTION_LETTERS[len(options)-1]}). "
            f"Do not explain.\n\n"
            f"Question: {row['question']}\n\n"
            f"Options:\n{option_text}\n\n"
            f"Answer:"
        )
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": correct_letter},
        ]

        full = tokenizer.apply_chat_template(messages, tokenize=False)
        full_ids = tokenizer.encode(full)
        prompt = tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True)
        prompt_len = len(tokenizer.encode(prompt))

        if len(full_ids) > MAX_SEQ_LEN:
            full_ids = full_ids[:MAX_SEQ_LEN]
        if prompt_len >= len(full_ids):
            continue

        examples.append({
            "input_ids": full_ids,
            "prompt_len": prompt_len,
            "length": len(full_ids),
            "correct_idx": correct_idx,
            "n_options": n_options,
        })

    log(f"  Prepared {len(examples)} training examples "
        f"(from {len(train_df)} questions)")
    return examples


def get_answer_token_ids(tokenizer):
    """Get token IDs for A through J answer letters."""
    ids = []
    for letter in OPTION_LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        tid = encoded[-1]
        ids.append(tid)
    log(f"  Answer token IDs: {dict(zip(OPTION_LETTERS, ids))}")
    return ids


# ────────────────────────────────────────────────
# LoRA Setup
# ────────────────────────────────────────────────

def apply_lora(model):
    """Apply standard LoRA to model using mlx_lm's LoRA layers."""
    from mlx_lm.tuner.utils import linear_to_lora_layers

    num_layers = len(model.model.layers) if hasattr(model, "model") else len(model.layers)
    config = {
        "rank": LORA_RANK,
        "scale": LORA_SCALE,
        "dropout": LORA_DROPOUT,
        "keys": LORA_KEYS,
    }
    linear_to_lora_layers(model, num_layers, config)

    # Freeze all except LoRA params
    model.freeze()
    from mlx_lm.tuner.utils import LoRALinear
    layers = model.model.layers if hasattr(model, "model") else model.layers
    for layer in layers:
        for key in LORA_KEYS:
            parts = key.split(".")
            module = layer
            for p in parts:
                module = getattr(module, p)
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], recurse=False)

    trainable = sum(p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    total = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    log(f"  LoRA applied: {trainable:,} trainable / {total:,} total params "
        f"(r={LORA_RANK})")
    return trainable


# ────────────────────────────────────────────────
# Mixed Training (NTP + MCQ)
# ────────────────────────────────────────────────

def train_mixed(model, tokenizer, examples, answer_token_ids):
    """Train LoRA with mixed NTP + MCQ classification loss."""
    random.seed(SEED)
    random.shuffle(examples)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    optimizer = optim.AdamW(learning_rate=LR, weight_decay=0.01)

    # Pre-convert token IDs to Python ints for slicing
    atids = [int(t) for t in answer_token_ids]

    def loss_fn(model, input_ids, lengths, prompt_lens, correct_idxs, n_options_batch):
        logits = model(input_ids)
        logits = logits.astype(mx.float32)
        shift_logits = logits[:, :-1, :]
        shift_targets = input_ids[:, 1:]

        # ── NTP loss (prompt-masked) ──
        ce = nn.losses.cross_entropy(shift_logits, shift_targets, reduction="none")
        S = shift_targets.shape[1]
        pos = mx.arange(S)[None, :]
        mask = (pos >= (prompt_lens[:, None] - 1)) & (pos < (lengths[:, None] - 1))
        mask = mask.astype(mx.float32)
        ntp_loss = (ce * mask).sum() / mx.maximum(mask.sum(), 1.0)

        # ── MCQ classification loss ──
        # Get logits at answer position (prompt_len - 1 in shift_logits)
        B = input_ids.shape[0]
        V = shift_logits.shape[-1]
        answer_pos = (prompt_lens - 1)[:, None, None]
        answer_pos_expanded = mx.broadcast_to(answer_pos, (B, 1, V))
        answer_vocab_logits = mx.take_along_axis(
            shift_logits, answer_pos_expanded, axis=1
        ).squeeze(1)  # (B, V)

        # Extract logits for all 10 answer tokens (A-J)
        option_logits = mx.concatenate([
            answer_vocab_logits[:, atids[i]:atids[i] + 1]
            for i in range(10)
        ], axis=1)  # (B, 10)

        # Mask to valid options per question (some have < 10 options)
        option_mask = mx.arange(10)[None, :] < n_options_batch[:, None]
        # Set invalid option logits to large negative
        option_logits = mx.where(option_mask, option_logits, mx.array(-1e9))

        mcq_loss = nn.losses.cross_entropy(
            option_logits, correct_idxs, reduction="mean"
        )

        return ntp_loss + MCQ_WEIGHT * mcq_loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    t0 = time.time()
    idx = 0

    for step in range(N_STEPS):
        batch_exs = []
        for _ in range(BATCH_SIZE):
            batch_exs.append(examples[idx % len(examples)])
            idx += 1

        max_len = max(e["length"] for e in batch_exs)
        input_ids = mx.array([
            e["input_ids"] + [pad_id] * (max_len - e["length"])
            for e in batch_exs
        ])
        lengths = mx.array([e["length"] for e in batch_exs])
        prompt_lens = mx.array([e["prompt_len"] for e in batch_exs])
        correct_idxs = mx.array([e["correct_idx"] for e in batch_exs])
        n_options_batch = mx.array([e["n_options"] for e in batch_exs])

        loss, grads = loss_and_grad(
            model, input_ids, lengths, prompt_lens, correct_idxs, n_options_batch
        )
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

        losses.append(loss.item())

        if (step + 1) % max(1, N_STEPS // 10) == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - t0
            log(f"    Step {step+1}/{N_STEPS}: loss={avg:.4f} ({elapsed:.1f}s)")

    total_time = time.time() - t0
    log(f"  Training complete: {N_STEPS} steps in {total_time:.0f}s "
        f"({total_time/60:.1f} min)")
    return losses, total_time


# ────────────────────────────────────────────────
# MMLU-Pro Evaluation
# ────────────────────────────────────────────────

def format_mmlu_prompt(row, tokenizer, enable_thinking=False):
    """Format MMLU-Pro question for evaluation.

    Matches exp_bench_mmlu_pro prompt format (validated at 42.3% base).
    CRITICAL: must pass enable_thinking=False explicitly or Gemma 4 defaults
    to thinking mode and wastes max_tokens on thinking tokens.
    """
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
    """Remove thinking tokens from Gemma 4 response."""
    if not response:
        return response, 0
    thinking_len = 0
    m = re.search(r'<\|channel>thought.*?<channel\|>', response, flags=re.DOTALL)
    if m:
        thinking_len = len(m.group(0))
    cleaned = re.sub(r'<\|channel>thought.*?<channel\|>', '', response, flags=re.DOTALL)
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
    # Starts with letter
    m = re.match(r"^([A-J])[.\s:)\-,]", answer_text)
    if m:
        return m.group(1), thinking_chars
    # "The answer is X" pattern
    m = re.search(r"(?:answer|correct)\s+(?:is\s+)?([A-J])", answer_text, re.IGNORECASE)
    if m:
        return m.group(1).upper(), thinking_chars
    # Last letter match
    last_letter = None
    for ch in answer_text:
        if ch.upper() in OPTION_LETTERS:
            last_letter = ch.upper()
    if last_letter:
        return last_letter, thinking_chars
    return None, thinking_chars


def evaluate_mmlu(model, tokenizer, eval_df, label="eval",
                  per_cat=50, enable_thinking=False):
    """Evaluate on MMLU-Pro subset."""
    from mlx_lm import generate

    max_tokens = 2048 if enable_thinking else 32
    total_correct = 0
    total_count = 0
    total_thinking_chars = 0
    category_results = {}
    t0 = time.time()

    categories = sorted(eval_df["category"].unique())
    for cat in categories:
        cat_df = eval_df[eval_df["category"] == cat]
        if per_cat and per_cat < len(cat_df):
            cat_df = cat_df.sample(n=per_cat, random_state=SEED)

        correct = 0
        cat_thinking = 0
        for idx, (_, row) in enumerate(cat_df.iterrows()):
            prompt = format_mmlu_prompt(row, tokenizer, enable_thinking)
            try:
                response = generate(model, tokenizer, prompt=prompt,
                                    max_tokens=max_tokens, verbose=False)
            except Exception as e:
                response = f"ERROR: {e}"

            predicted, thinking_chars = parse_answer(response)
            cat_thinking += thinking_chars
            if predicted == row["answer"]:
                correct += 1

            if (idx + 1) % 20 == 0:
                elapsed = time.time() - t0
                log(f"    [{label}] {cat} {idx+1}/{len(cat_df)}: "
                    f"{correct}/{idx+1} ({100*correct/(idx+1):.1f}%) | {elapsed:.0f}s")

        cat_acc = correct / len(cat_df) if len(cat_df) > 0 else 0
        category_results[cat] = {
            "correct": correct,
            "total": len(cat_df),
            "accuracy": round(cat_acc, 4),
        }
        total_correct += correct
        total_count += len(cat_df)
        total_thinking_chars += cat_thinking
        elapsed = time.time() - t0
        log(f"  [{label}] {cat}: {correct}/{len(cat_df)} = {100*cat_acc:.1f}% "
            f"(cumulative: {100*total_correct/total_count:.1f}%, {elapsed:.0f}s)")

    overall_acc = total_correct / total_count if total_count > 0 else 0
    elapsed = time.time() - t0
    return {
        "overall_accuracy": round(overall_acc, 4),
        "total_correct": total_correct,
        "total_count": total_count,
        "total_thinking_chars": total_thinking_chars,
        "categories": category_results,
        "elapsed_s": round(elapsed, 1),
    }


# ────────────────────────────────────────────────
# HumanEval Quick Check
# ────────────────────────────────────────────────

def eval_humaneval_quick(model, tokenizer, n_eval=20):
    """Quick HumanEval check for generative quality."""
    from mlx_lm import generate

    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet",
                               repo_type="dataset")
        he_df = pd.read_parquet(path)
    except Exception:
        try:
            from datasets import load_dataset
            ds = load_dataset("openai_humaneval", split="test")
            he_df = ds.to_pandas()
        except Exception as e:
            log(f"  HumanEval: SKIP (cannot load dataset: {e})")
            return {"status": "skipped", "reason": str(e)}

    he_df = he_df.head(n_eval)
    passed = 0

    for i, (_, row) in enumerate(he_df.iterrows()):
        prompt_code = row["prompt"]
        messages = [{"role": "user", "content":
            f"Complete the following Python function:\n\n```python\n{prompt_code}\n```\n\nRespond with only the function body, no markdown."}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=formatted,
                            max_tokens=512, verbose=False)

        # Extract code
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        completion = code_match.group(1) if code_match else response

        full_code = prompt_code + completion + "\n\n" + row["test"] + f"\n\ncheck({row['entry_point']})\n"

        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                timeout=10, capture_output=True, text=True,
            )
            if result.returncode == 0:
                passed += 1
        except (subprocess.TimeoutExpired, Exception):
            pass

    acc = passed / len(he_df) * 100
    log(f"  HumanEval: {passed}/{len(he_df)} = {acc:.1f}%")
    return {
        "passed": passed,
        "total": len(he_df),
        "accuracy": round(acc, 1),
    }


# ────────────────────────────────────────────────
# Adapter Save
# ────────────────────────────────────────────────

def save_lora_adapter(model):
    """Save LoRA adapter weights."""
    from mlx_lm.tuner.utils import LoRALinear
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    adapter_weights = {}
    layers = model.model.layers if hasattr(model, "model") else model.layers
    for li, layer in enumerate(layers):
        for key in LORA_KEYS:
            parts = key.split(".")
            module = layer
            for p in parts:
                module = getattr(module, p, None)
                if module is None:
                    break
            if module is not None and isinstance(module, LoRALinear):
                prefix = f"model.layers.{li}.{key}"
                adapter_weights[f"{prefix}.lora_a"] = module.lora_a
                adapter_weights[f"{prefix}.lora_b"] = module.lora_b

    mx.savez(str(ADAPTER_DIR / "adapters.safetensors"), **adapter_weights)
    # Save adapter config
    config = {
        "rank": LORA_RANK,
        "scale": LORA_SCALE,
        "dropout": LORA_DROPOUT,
        "keys": LORA_KEYS,
        "num_layers": len(layers),
    }
    (ADAPTER_DIR / "adapter_config.json").write_text(json.dumps(config, indent=2))
    log(f"  Adapter saved to {ADAPTER_DIR}")


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────

def main():
    t_start = time.time()
    log("=" * 70)
    log("P10.C0: MCQ-Format Adapter (Fix NTP Degradation on Benchmarks)")
    log(f"LoRA r{LORA_RANK}, Mixed loss (NTP + MCQ λ={MCQ_WEIGHT})")
    log(f"SMOKE={IS_SMOKE}, N_STEPS={N_STEPS}, EVAL_PER_CAT={EVAL_PER_CAT}")
    log("=" * 70)

    results = {
        "experiment": "exp_p10_mcq_adapter_training",
        "smoke": IS_SMOKE,
        "seed": SEED,
        "lora_rank": LORA_RANK,
        "mcq_weight": MCQ_WEIGHT,
        "n_steps": N_STEPS,
        "eval_per_cat": EVAL_PER_CAT,
        "thinking_per_cat": THINKING_PER_CAT,
    }

    # ── Phase 0: Load data ──────────────────────
    log("\n[Phase 0] Loading data...")
    train_df, eval_df = load_and_split_data()
    results["n_train"] = len(train_df)
    results["n_eval"] = len(eval_df)

    # ── Phase 1: Base model evaluation (no thinking) ──
    log("\n" + "=" * 70)
    log("[Phase 1] Base model — MMLU-Pro (no thinking)")
    log("=" * 70)

    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    log_memory("model-loaded")

    base_eval = evaluate_mmlu(model, tokenizer, eval_df,
                              label="base", per_cat=EVAL_PER_CAT)
    results["base_no_thinking"] = base_eval
    log(f"\n  Phase 1: Base = {base_eval['overall_accuracy']*100:.1f}% "
        f"({base_eval['total_correct']}/{base_eval['total_count']}) "
        f"in {base_eval['elapsed_s']:.0f}s")
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    # ── Phase 2: Train MCQ adapter ──────────────
    log("\n" + "=" * 70)
    log("[Phase 2] Training MCQ adapter (mixed NTP + MCQ loss)")
    log("=" * 70)

    answer_token_ids = get_answer_token_ids(tokenizer)
    results["answer_token_ids"] = dict(zip(OPTION_LETTERS, [int(t) for t in answer_token_ids]))

    training_examples = prepare_training_examples(train_df, tokenizer)
    results["n_training_examples"] = len(training_examples)

    trainable_params = apply_lora(model)
    results["trainable_params"] = trainable_params
    log_memory("lora-applied")

    model.train()
    losses, train_time = train_mixed(model, tokenizer, training_examples, answer_token_ids)
    results["train_time_s"] = round(train_time, 1)
    results["train_time_min"] = round(train_time / 60, 1)
    results["final_loss"] = round(losses[-1], 4) if losses else None
    results["loss_history"] = [round(l, 4) for l in losses[::max(1, len(losses)//20)]]

    # Save adapter
    save_lora_adapter(model)

    # ── Phase 3: MCQ adapter evaluation (no thinking) ──
    log("\n" + "=" * 70)
    log("[Phase 3] MCQ adapter — MMLU-Pro (no thinking)")
    log("=" * 70)

    model.eval()
    adapted_eval = evaluate_mmlu(model, tokenizer, eval_df,
                                 label="adapted", per_cat=EVAL_PER_CAT)
    results["adapted_no_thinking"] = adapted_eval
    log(f"\n  Phase 3: Adapted = {adapted_eval['overall_accuracy']*100:.1f}% "
        f"(base was {base_eval['overall_accuracy']*100:.1f}%)")
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    # ── Phase 4: MCQ adapter + thinking ─────────
    log("\n" + "=" * 70)
    log("[Phase 4] MCQ adapter — MMLU-Pro (WITH thinking)")
    log("=" * 70)

    adapted_thinking_eval = evaluate_mmlu(
        model, tokenizer, eval_df,
        label="adapted+thinking", per_cat=THINKING_PER_CAT,
        enable_thinking=True)
    results["adapted_with_thinking"] = adapted_thinking_eval
    log(f"\n  Phase 4: Adapted+Thinking = {adapted_thinking_eval['overall_accuracy']*100:.1f}%")
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    # ── Phase 5: Base + thinking (control) ──────
    log("\n" + "=" * 70)
    log("[Phase 5] Base model — MMLU-Pro (WITH thinking, same subset)")
    log("=" * 70)

    # Remove LoRA, reload base
    del model
    cleanup()
    model, tokenizer = load(MODEL_ID)

    base_thinking_eval = evaluate_mmlu(
        model, tokenizer, eval_df,
        label="base+thinking", per_cat=THINKING_PER_CAT,
        enable_thinking=True)
    results["base_with_thinking"] = base_thinking_eval
    log(f"\n  Phase 5: Base+Thinking = {base_thinking_eval['overall_accuracy']*100:.1f}%")
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    # ── Phase 6: HumanEval quick check ──────────
    log("\n" + "=" * 70)
    log("[Phase 6] HumanEval generative quality check")
    log("=" * 70)

    # Reload with adapter for HumanEval
    del model
    cleanup()
    model, tokenizer = load(MODEL_ID)
    apply_lora(model)
    # Load saved adapter weights
    adapter_file = ADAPTER_DIR / "adapters.safetensors"
    if adapter_file.exists():
        weights = mx.load(str(adapter_file))
        model.load_weights(list(weights.items()))
        log("  Loaded saved adapter weights")

    model.eval()
    humaneval_result = eval_humaneval_quick(model, tokenizer, HUMANEVAL_N)
    results["humaneval"] = humaneval_result
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    # ── Summary & Kill Criteria ─────────────────
    log("\n" + "=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)

    base_acc = base_eval["overall_accuracy"] * 100
    adapted_acc = adapted_eval["overall_accuracy"] * 100
    adapted_thinking_acc = adapted_thinking_eval["overall_accuracy"] * 100
    base_thinking_acc = base_thinking_eval["overall_accuracy"] * 100

    mcq_effect = adapted_acc - base_acc
    thinking_effect_base = base_thinking_acc - base_acc
    thinking_effect_adapted = adapted_thinking_acc - adapted_acc
    combined_effect = adapted_thinking_acc - base_acc

    log(f"\n  Base (no thinking):      {base_acc:.1f}%")
    log(f"  Base + thinking:         {base_thinking_acc:.1f}% ({thinking_effect_base:+.1f}pp thinking effect)")
    log(f"  MCQ adapter (no think):  {adapted_acc:.1f}% ({mcq_effect:+.1f}pp MCQ effect)")
    log(f"  MCQ adapter + thinking:  {adapted_thinking_acc:.1f}% ({combined_effect:+.1f}pp combined)")

    he_acc = humaneval_result.get("accuracy")
    he_str = f"{he_acc:.1f}%" if he_acc is not None else "N/A"
    log(f"  HumanEval (adapted):     {he_str}")

    # Kill criteria
    k1470_pass = adapted_thinking_acc >= 65.0
    k1471_pass = he_acc is not None and he_acc >= 55.0  # base ~60%, within 5pp
    k1472_pass = train_time < 30 * 60  # < 30 min

    results["kill_criteria"] = {
        "K1470": {
            "pass": k1470_pass,
            "detail": f"Adapted+thinking={adapted_thinking_acc:.1f}% (target >=65%)",
        },
        "K1471": {
            "pass": k1471_pass,
            "detail": f"HumanEval={he_str} (target >=55%)",
        },
        "K1472": {
            "pass": k1472_pass,
            "detail": f"Train time={train_time/60:.1f}min (target <30min)",
        },
    }

    results["summary"] = {
        "base_no_thinking": round(base_acc, 1),
        "base_with_thinking": round(base_thinking_acc, 1),
        "adapted_no_thinking": round(adapted_acc, 1),
        "adapted_with_thinking": round(adapted_thinking_acc, 1),
        "mcq_effect_pp": round(mcq_effect, 1),
        "thinking_effect_base_pp": round(thinking_effect_base, 1),
        "thinking_effect_adapted_pp": round(thinking_effect_adapted, 1),
        "combined_effect_pp": round(combined_effect, 1),
        "humaneval_acc": he_acc,
        "train_time_min": round(train_time / 60, 1),
    }

    log(f"\nKILL CRITERIA:")
    log(f"  K1470 (adapted+thinking >= 65%): {'PASS' if k1470_pass else 'FAIL'} ({adapted_thinking_acc:.1f}%)")
    log(f"  K1471 (HumanEval >= 55%):        {'PASS' if k1471_pass else 'FAIL'} ({he_str})")
    log(f"  K1472 (train < 30min):           {'PASS' if k1472_pass else 'FAIL'} ({train_time/60:.1f}min)")

    # Per-category comparison
    log("\n  Per-category (base vs adapted, no thinking):")
    for cat in sorted(base_eval["categories"]):
        b = base_eval["categories"][cat]["accuracy"] * 100
        a = adapted_eval["categories"][cat]["accuracy"] * 100
        d = a - b
        log(f"    {cat:20s}: {b:5.1f}% -> {a:5.1f}% ({d:+.1f}pp)")

    total_time = time.time() - t_start
    results["total_time_s"] = round(total_time, 1)
    results["total_time_h"] = round(total_time / 3600, 2)

    log(f"\nTotal time: {total_time:.0f}s ({total_time/3600:.2f}h)")

    cleanup(model, tokenizer)
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
