#!/usr/bin/env python3
"""exp_score_kl_constrained_mcq — smoke-scale implementation.

SCoRe (arxiv:2409.12917) stage-I KL-constrained SFT on s1K reasoning traces.
Trains a LoRA adapter on Gemma-4-E4B-it-4bit with custom loss

    L = CE(π_θ, y)  +  β · KL( π_0(·|x) ‖ π_θ(·|x) )

then evaluates MMLU-Pro (thinking mode) across 3 categories.

Smoke mode pipes-through the whole pipeline at n_train=27, n_steps=20,
eval_per_cat=2 — KC values are informative only; verdict = PROVISIONAL.
"""

import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd

# Memory hygiene (mem-antipattern ref: MLX unified memory)
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_DIR = EXPERIMENT_DIR / "data"
ADAPTER_DIR = EXPERIMENT_DIR / "adapters" / "score_kl"
KL_LOG_FILE = EXPERIMENT_DIR / "kl_trace.jsonl"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "1") == "1"
SEED = 42

LORA_RANK = 8
LORA_SCALE = 1.0
LORA_DROPOUT = 0.0
LORA_KEYS = ["self_attn.v_proj", "self_attn.o_proj"]
LORA_NUM_LAYERS = 16  # last N transformer blocks
N_STEPS = 20 if IS_SMOKE else 1000
BATCH_SIZE = 1
LR = 1e-5
BETA_KL = 1.0
KL_BOUND = 0.1  # K1726 threshold (nats)
MAX_SEQ_LEN = 2048

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


# ── Phase 1: reuse s1K training data ───────────────────────────────────────

def phase_prepare_training_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sibling = EXPERIMENT_DIR.parent / "exp_p11_reasoning_sft_s1k" / "data"
    local_train = DATA_DIR / "train.jsonl"
    local_valid = DATA_DIR / "valid.jsonl"

    if not local_train.exists():
        import shutil as _sh
        _sh.copy(sibling / "train.jsonl", local_train)
        _sh.copy(sibling / "valid.jsonl", local_valid)
        log(f"  Copied s1K data from {sibling}")

    n_train = sum(1 for _ in open(local_train))
    n_valid = sum(1 for _ in open(local_valid))
    n_with_think = 0
    with open(local_train) as f:
        for line in f:
            ex = json.loads(line)
            assistant = next(
                (m["content"] for m in ex["messages"] if m["role"] == "assistant"),
                "",
            )
            if "<think>" in assistant and "</think>" in assistant:
                n_with_think += 1
    a1_pass = (n_with_think / max(n_train, 1)) >= 0.99
    return {
        "n_train": n_train,
        "n_valid": n_valid,
        "a1_pass": bool(a1_pass),
        "n_with_think": n_with_think,
    }


# ── Phase 2: KL-constrained LoRA training ──────────────────────────────────

def phase_train_kl_constrained():
    """Custom training loop: CE + β·KL(π_0 ‖ π_θ).

    Memory layout on M5 Pro 48GB (IS_SMOKE):
      π_0 (base, frozen):        ~3 GB (4-bit Gemma-4-E4B)
      π_θ (model, LoRA-attach):  ~3 GB weights + LoRA ≪ 50MB
      activations + grad:        ~4 GB at BATCH=1, SEQ=2048
      total peak:                ~10 GB  (well within budget)
    """
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx.utils import tree_flatten

    log("\n[Phase 2] Loading base (frozen π_0) and training model (π_θ)…")
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    # --- π_0: frozen base reference, no LoRA
    base_model, tokenizer = load(MODEL_ID)
    base_model.freeze()
    base_model.eval()
    log_memory("base-loaded")

    # --- π_θ: same weights, LoRA trainable
    model, _ = load(MODEL_ID)
    model.freeze()
    lora_config = {
        "rank": LORA_RANK,
        "scale": LORA_SCALE,
        "dropout": LORA_DROPOUT,
        "keys": LORA_KEYS,
    }
    linear_to_lora_layers(model, LORA_NUM_LAYERS, lora_config, use_dora=False)
    log_memory("model+lora-attached")

    # Save adapter_config.json so the adapter can be re-loaded by mlx_lm.load
    (ADAPTER_DIR / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora",
        "num_layers": LORA_NUM_LAYERS,
        "lora_parameters": lora_config,
    }, indent=2))

    # --- Load training data manually (messages format → token ids) ---
    def _render_messages(msgs):
        return tokenizer.apply_chat_template(msgs, tokenize=False)

    def _load_jsonl_to_ids(path):
        out = []
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                text = _render_messages(ex["messages"])
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) > MAX_SEQ_LEN:
                    ids = ids[:MAX_SEQ_LEN]
                if len(ids) < 2:
                    continue
                out.append(ids)
        return out

    train_ids = _load_jsonl_to_ids(DATA_DIR / "train.jsonl")
    log(f"  Tokenised {len(train_ids)} train examples; "
        f"median length {int(np.median([len(x) for x in train_ids]))} tokens")

    # --- KL-regularised loss — plain (not mx.compiled) closure ---
    kl_trace = []

    def kl_loss(model_, input_ids):
        """CE + β · KL(π_0 ‖ π_θ); masks everything except the answer tokens
        (positions >= len(prompt)). Here we mask padding only — full supervised."""
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        logits = model_(inputs)                                    # (B, T, V)
        base_logits = mx.stop_gradient(base_model(inputs))         # (B, T, V)
        ntoks = mx.array(targets.size, dtype=mx.float32)

        ce = nn.losses.cross_entropy(logits, targets, reduction="mean")

        log_p0 = nn.log_softmax(base_logits.astype(mx.float32), axis=-1)
        log_pt = nn.log_softmax(logits.astype(mx.float32), axis=-1)
        p0 = mx.exp(log_p0)
        kl = (p0 * (log_p0 - log_pt)).sum(axis=-1).mean()

        total = ce + BETA_KL * kl
        return total, kl

    loss_and_grad = nn.value_and_grad(model, kl_loss)
    optimizer = optim.AdamW(learning_rate=LR)

    log(f"  Starting KL-constrained train: iters={N_STEPS}, β={BETA_KL}, lr={LR}")
    model.train()
    rng = np.random.RandomState(SEED)
    t0 = time.time()
    for step in range(1, N_STEPS + 1):
        idx = int(rng.randint(0, len(train_ids)))
        ids = train_ids[idx]
        batch = mx.array([ids], dtype=mx.int32)
        try:
            (loss_val, kl_val), grads = loss_and_grad(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss_val, kl_val)
            kl_f = float(kl_val.item())
            loss_f = float(loss_val.item())
        except Exception as e:
            log(f"  step {step}: train raised {e!r}; aborting")
            cleanup(model, base_model, tokenizer, optimizer)
            return {
                "status": "failed",
                "error": repr(e),
                "time_s": round(time.time() - t0, 1),
                "kl_trace": kl_trace,
            }
        kl_trace.append({"step": step, "kl": kl_f, "loss": loss_f})
        if step % max(1, N_STEPS // 5) == 0 or step == 1:
            log(f"    step {step}: loss={loss_f:.3f}  kl={kl_f:.4f}")
        mx.clear_cache()
    elapsed = time.time() - t0

    with open(KL_LOG_FILE, "w") as f:
        for rec in kl_trace:
            f.write(json.dumps(rec) + "\n")

    # Save LoRA adapter weights
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(ADAPTER_DIR / "adapters.safetensors"), adapter_weights)

    kls = [r["kl"] for r in kl_trace]
    max_kl = max(kls) if kls else None
    mean_kl = float(np.mean(kls)) if kls else None

    cleanup(model, base_model, tokenizer, optimizer)

    return {
        "status": "ok",
        "time_s": round(elapsed, 1),
        "steps": N_STEPS,
        "kl_trace": kl_trace,
        "max_kl": max_kl,
        "mean_kl": mean_kl,
    }


# ── Pre-training diagnostic gate: verify base emits thinking ───────────────

def phase_diagnostic_gate():
    """Sanity check — generate 3 base-model responses (one per category) with
    enable_thinking=True and verify the updated strip_thinking regex captures
    non-zero thinking chars. Aborts the run if all three samples show 0 chars.

    Saves raw responses to data/channel_diagnostic.jsonl for offline inspection.
    Required fix per REVIEW-adversarial.md: prevents burning ~1 h of compute
    on a behaviorally-uninterpretable eval.
    """
    from mlx_lm import load, generate

    log("\n[Gate] 3-sample base-model thinking sanity check")
    mmlu_path = EXPERIMENT_DIR.parent / "exp_bench_mmlu_pro" / "data" / "test.parquet"
    if not mmlu_path.exists():
        log("  WARN: MMLU-Pro data missing — cannot run gate, skipping")
        return {"gate_run": False, "reason": "mmlu-data-missing", "pass": False}

    df = pd.read_parquet(mmlu_path)
    model, tokenizer = load(MODEL_ID)
    log_memory("gate-base-loaded")

    rng = np.random.RandomState(SEED)
    diag_path = DATA_DIR / "channel_diagnostic.jsonl"
    OPTION_LETTERS = "ABCDEFGHIJ"
    samples = []
    think_lens = []

    for cat in EVAL_CATEGORIES:
        cat_df = df[df["category"].str.lower() == cat.lower()]
        if len(cat_df) == 0:
            continue
        idx = int(rng.choice(len(cat_df), 1)[0])
        row = cat_df.iloc[idx]
        options = row.get("options", [])
        n_opts = len(options)
        option_text = "\n".join(
            f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)
        )
        user_content = (
            f"Answer the following multiple choice question. Select the single "
            f"best answer letter (A through {OPTION_LETTERS[n_opts - 1]}).\n\n"
            f"Question: {row['question']}\n\n"
            f"Options:\n{option_text}\n\nAnswer:"
        )
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        response = generate(model, tokenizer, prompt=prompt, max_tokens=2048)
        _, t_chars = strip_thinking(response)
        think_lens.append(t_chars)
        samples.append({
            "category": cat,
            "thinking_chars": int(t_chars),
            "response_prefix": response[:800],
            "response_len_chars": len(response),
        })
        log(f"  {cat}: thinking_chars={t_chars}, resp_len={len(response)}")
        mx.eval()

    with open(diag_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    cleanup(model, tokenizer)

    max_think = max(think_lens) if think_lens else 0
    avg_think = sum(think_lens) / max(len(think_lens), 1)
    gate_pass = max_think > 0
    return {
        "gate_run": True,
        "pass": bool(gate_pass),
        "max_thinking_chars": int(max_think),
        "avg_thinking_chars": round(avg_think, 0),
        "n_samples": len(samples),
        "diagnostic_file": str(diag_path),
    }


# ── Phase 3: MMLU-Pro eval ─────────────────────────────────────────────────

def strip_thinking(response):
    # Gemma-4 emits `<|channel>thought...<channel|>` (asymmetric delimiters;
    # confirmed working in exp_p11_baseline_eval:107). Fallback `<think>...</think>`
    # covers alternative training formats.
    if not response:
        return response or "", 0
    thinking_len = 0
    m = re.search(r"<\|channel>thought.*?<channel\|>", response, flags=re.DOTALL)
    if m:
        thinking_len = len(m.group(0))
    cleaned = re.sub(r"<\|channel>thought.*?<channel\|>", "", response, flags=re.DOTALL)
    if thinking_len == 0:
        m2 = re.search(r"<think>(.*?)</think>", cleaned, flags=re.DOTALL)
        if m2:
            thinking_len = len(m2.group(1))
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    return cleaned, thinking_len


def parse_mcq_answer(response):
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
    from mlx_lm import load, generate

    label = "ADAPTED" if adapter_path else "BASE"
    log(f"\n[Eval] MMLU-Pro thinking ({label})")

    mmlu_path = EXPERIMENT_DIR.parent / "exp_bench_mmlu_pro" / "data" / "test.parquet"
    if not mmlu_path.exists():
        return {"accuracy": None, "error": "mmlu-data-missing"}

    df = pd.read_parquet(mmlu_path)
    load_kwargs = {"adapter_path": str(adapter_path)} if adapter_path else {}
    try:
        model, tokenizer = load(MODEL_ID, **load_kwargs)
    except Exception as e:
        return {"accuracy": None, "error": f"load-failed: {e!r}"}

    log_memory(f"{label}-post-load")

    rng = np.random.RandomState(SEED)
    correct_total, total, thinking_total = 0, 0, 0
    per_cat = {}
    OPTION_LETTERS = "ABCDEFGHIJ"

    for cat in EVAL_CATEGORIES:
        cat_df = df[df["category"].str.lower() == cat.lower()]
        if len(cat_df) == 0:
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
                f"Answer the following multiple choice question. Select the single "
                f"best answer letter (A through {OPTION_LETTERS[n_opts - 1]}).\n\n"
                f"Question: {row['question']}\n\n"
                f"Options:\n{option_text}\n\nAnswer:"
            )
            messages = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            try:
                # mem-antipattern-008: keep max_tokens large so thinking isn't truncated.
                response = generate(model, tokenizer, prompt=prompt, max_tokens=2048)
                predicted, t_chars = parse_mcq_answer(response)
                cat_thinking += t_chars
                if predicted == correct_letter:
                    cat_correct += 1
            except Exception as e:
                log(f"    gen err: {e!r}")
            mx.eval()

        thinking_total += cat_thinking
        correct_total += cat_correct
        total += n_sample
        acc = cat_correct / n_sample * 100 if n_sample else 0.0
        per_cat[cat] = {
            "correct": cat_correct,
            "total": n_sample,
            "acc": round(acc, 1),
            "avg_thinking_chars": round(cat_thinking / max(n_sample, 1), 0),
        }
        log(f"  {cat}: {acc:.0f}% ({cat_correct}/{n_sample})")

    cleanup(model, tokenizer)

    overall_acc = correct_total / total * 100 if total else 0.0
    overall_think = thinking_total / total if total else 0.0
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
    log("exp_score_kl_constrained_mcq")
    log(f"IS_SMOKE={IS_SMOKE}  N_STEPS={N_STEPS}  β={BETA_KL}")
    log("=" * 60)
    log_memory("start")

    results = {
        "experiment": "exp_score_kl_constrained_mcq",
        "model": MODEL_ID,
        "is_smoke": IS_SMOKE,
        "lora_rank": LORA_RANK,
        "lora_scale": LORA_SCALE,
        "lora_keys": LORA_KEYS,
        "n_steps": N_STEPS,
        "beta_kl": BETA_KL,
        "kl_bound_k1726": KL_BOUND,
        "eval_categories": EVAL_CATEGORIES,
        "eval_per_cat": EVAL_PER_CAT,
    }

    log("\n[Phase 1] Prepare s1K training data")
    data_info = phase_prepare_training_data()
    results["training_data"] = data_info
    log(f"  {data_info}")

    # Phase 1.5 — pre-training diagnostic gate (prevents ~1h of uninterpretable
    # compute if Gemma-4 isn't emitting thinking under current template/regex).
    gate_info = phase_diagnostic_gate()
    results["diagnostic_gate"] = gate_info
    log(f"  gate: {gate_info}")
    if not IS_SMOKE and gate_info.get("gate_run") and not gate_info.get("pass"):
        results["verdict"] = "ABORTED"
        results["abort_reason"] = (
            "diagnostic_gate: 3-sample base eval returned 0 thinking_chars — "
            "strip_thinking regex or chat template still broken. "
            "Inspect data/channel_diagnostic.jsonl."
        )
        results["total_time_s"] = round(time.time() - t0, 1)
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        log(f"\nABORTED: {results['abort_reason']}")
        return

    # Phase 2
    train_results = phase_train_kl_constrained()
    results["training"] = train_results
    adapter_path = ADAPTER_DIR if train_results.get("status") == "ok" else None

    # Phase 3
    results["base"] = phase_eval_mmlu_pro(adapter_path=None)
    if adapter_path is not None:
        results["adapted"] = phase_eval_mmlu_pro(adapter_path=adapter_path)
    else:
        results["adapted"] = {"accuracy": None, "error": "training_failed"}

    base_acc = results["base"].get("accuracy") or 0.0
    adapt_acc = results["adapted"].get("accuracy") or 0.0
    adapt_think = results["adapted"].get("avg_thinking_chars") or 0.0

    # K1724: MCQ ≥ plain-SFT baseline (50.4%)
    plain_sft_baseline = 50.4
    k1724_pass = (results["adapted"].get("accuracy") is not None
                  and adapt_acc >= plain_sft_baseline)

    # K1725: within 2pp of base
    k1725_delta = adapt_acc - base_acc
    k1725_pass = (results["adapted"].get("accuracy") is not None
                  and abs(k1725_delta) <= 2.0)

    # K1726: KL bound ≤ 0.1 at every logged step
    max_kl = train_results.get("max_kl")
    k1726_pass = (max_kl is not None and max_kl <= KL_BOUND)

    results["kill_criteria"] = {
        "K1724": {
            "desc": "MCQ ≥ plain-SFT baseline 50.4%",
            "adapt_acc": adapt_acc,
            "baseline": plain_sft_baseline,
            "pass": bool(k1724_pass),
        },
        "K1725": {
            "desc": "adapter within 2pp of base MMLU-Pro+thinking",
            "base_acc": base_acc,
            "adapt_acc": adapt_acc,
            "delta_pp": round(k1725_delta, 2),
            "pass": bool(k1725_pass),
        },
        "K1726": {
            "desc": "max KL(π_0‖π_θ) ≤ 0.1 across all logged training steps",
            "max_kl": max_kl,
            "mean_kl": train_results.get("mean_kl"),
            "bound": KL_BOUND,
            "pass": bool(k1726_pass),
        },
    }

    all_pass = bool(k1724_pass and k1725_pass and k1726_pass)
    results["all_pass"] = all_pass
    results["adapt_think_chars"] = adapt_think
    # Smoke is always PROVISIONAL.
    if IS_SMOKE:
        results["verdict"] = "PROVISIONAL"
    else:
        results["verdict"] = "SUPPORTED" if all_pass else "KILLED"
    results["total_time_s"] = round(time.time() - t0, 1)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    log("\n" + "=" * 60)
    log("RESULTS")
    log("=" * 60)
    log(f"Base MMLU-Pro+thinking:    {base_acc:.1f}%")
    log(f"Adapter MMLU-Pro+thinking: {adapt_acc:.1f}%  (Δ={k1725_delta:+.1f}pp)")
    log(f"Max KL during train:       {max_kl}")
    log(f"K1724 ≥50.4%:  {'PASS' if k1724_pass else 'FAIL'}")
    log(f"K1725 ≤2pp:    {'PASS' if k1725_pass else 'FAIL'}")
    log(f"K1726 KL≤0.1:  {'PASS' if k1726_pass else 'FAIL'}")
    log(f"Verdict: {results['verdict']}")
    log(f"Total time: {results['total_time_s']:.0f}s")


if __name__ == "__main__":
    main()
