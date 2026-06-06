#!/usr/bin/env python3
"""
P11.C0: ThinkPO — DPO with Long CoT as Chosen, Short CoT as Rejected

Applies DPO (arXiv:2305.18290) after GRPO adapter to prefer longer thinking traces.
Implements offline DPO: reference log-probs precomputed once, training uses cached values.

Kill criteria:
  K1499: ThinkPO MMLU-Pro (thinking) >= GRPO + 2pp
  K1500: avg_thinking_chars (ThinkPO) >= avg_thinking_chars (GRPO) * 1.10
  K1501: GSM8K accuracy (ThinkPO) >= GSM8K (GRPO) - 5pp

References:
  arXiv:2502.13173 (ThinkPO)
  arXiv:2305.18290 (DPO)
  arXiv:2501.12948 (DeepSeek-R1: RS-SFT warmup)
"""

import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent  # llm/
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_DIR = EXPERIMENT_DIR / "data"
ADAPTER_DIR = EXPERIMENT_DIR / "adapters" / "thinkpo"
GRPO_ADAPTER_PATH = REPO_ROOT / "experiments/models/exp_p11_grpo_reasoning_adapter/adapters/rs_sft"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1" or "--smoke" in sys.argv
N_PREF_QUESTIONS = 10 if IS_SMOKE else 100  # questions per phase-1 sample
N_COMPLETIONS_PER_Q = 2 if IS_SMOKE else 4   # completions to generate per question
DPO_STEPS = 5 if IS_SMOKE else 100
BETA = 0.1
LR = 5e-6
N_EVAL_PER_CAT = 2 if IS_SMOKE else 7  # MMLU-Pro categories × N_EVAL_PER_CAT
MAX_TOKENS_SAMPLE = 2048
MAX_TOKENS_GSM8K = 1024
MIN_THINKING_DIFF = 100 if IS_SMOKE else 200  # min char diff to form preference pair

THINKING_OPEN = "<|channel>thought"
THINKING_CLOSE = "<channel|>"

# Memory limits
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

MMLU_PRO_CATS = [
    "math", "physics", "chemistry", "biology",
    "law", "business", "economics", "psychology",
    "computer science", "health", "history", "philosophy",
    "engineering", "other"
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_thinking(text):
    """Extract thinking channel content from Gemma 4 output."""
    m = re.search(r"<\|channel>thought(.*?)<channel\|>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def parse_answer(text):
    """Extract multiple choice answer letter from model output."""
    after_think = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    after_think = re.sub(r"<think>.*?</think>", "", after_think, flags=re.DOTALL)
    m = re.search(r"\b([A-J])\b", after_think.strip()[:200])
    if m:
        return m.group(1)
    m = re.search(r"answer[^A-Ja-j]*([A-J])", after_think, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def load_mmlu_pro(n_per_cat=7):
    """Load MMLU-Pro from the cached local data dir (used in other experiments)."""
    # Try local data directory (from exp_p11_baseline_eval or exp_bench_mmlu_pro)
    candidates = [
        REPO_ROOT / "experiments/models/exp_bench_mmlu_pro/data/mmlu_pro.parquet",
        REPO_ROOT / "experiments/models/exp_p11_baseline_eval/data/mmlu_pro_test.parquet",
    ]
    import pandas as pd
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            log(f"Loaded MMLU-Pro from {p}: {len(df)} rows")
            # Sample n_per_cat per category
            cats = df["category"].unique() if "category" in df.columns else df["subject"].unique()
            col = "category" if "category" in df.columns else "subject"
            samples = []
            for cat in cats:
                cat_df = df[df[col] == cat]
                n = min(n_per_cat, len(cat_df))
                samples.append(cat_df.sample(n, random_state=42))
            return pd.concat(samples, ignore_index=True)

    # Fallback: download via datasets
    log("Downloading MMLU-Pro via datasets library...")
    try:
        from datasets import load_dataset
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        import pandas as pd
        df = ds.to_pandas()
        col = "category" if "category" in df.columns else "subject"
        samples = []
        for cat in df[col].unique():
            cat_df = df[df[col] == cat]
            n = min(n_per_cat, len(cat_df))
            samples.append(cat_df.sample(n, random_state=42))
        return pd.concat(samples, ignore_index=True)
    except Exception as e:
        raise RuntimeError(f"Cannot load MMLU-Pro: {e}")


def format_mmlu_pro_prompt(row, tokenizer):
    """Format a MMLU-Pro row as a thinking-enabled chat prompt."""
    opts = row.get("options", row.get("choices", []))
    opt_str = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(opts))
    question = row.get("question", row.get("input", ""))
    user_content = (
        f"Question: {question}\n\nOptions:\n{opt_str}\n\n"
        "Think step by step, then answer with a single letter (A-J)."
    )
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Enable thinking channel
    if THINKING_OPEN not in prompt:
        prompt += f"{THINKING_OPEN}\n"
    return prompt


def generate_completion(model, tokenizer, prompt, max_tokens=MAX_TOKENS_SAMPLE):
    """Generate a single completion and return (text, thinking_chars)."""
    from mlx_lm import generate
    output = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    thinking = extract_thinking(output)
    return output, len(thinking)


# ─── Phase 1: Generate preference pairs ──────────────────────────────────────

def phase1_generate_pairs(model, tokenizer, df):
    """
    Generate N_COMPLETIONS_PER_Q completions per question.
    Create preference pair: (long, short) where diff > MIN_THINKING_DIFF.
    """
    log(f"Phase 1: generating preference pairs (N_Q={N_PREF_QUESTIONS}, K={N_COMPLETIONS_PER_Q})")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Sample N_PREF_QUESTIONS questions
    sample_df = df.sample(min(N_PREF_QUESTIONS, len(df)), random_state=42)

    pairs = []
    stats = {"n_questions": 0, "n_pairs": 0, "n_skipped_no_diff": 0}

    for idx, row in sample_df.iterrows():
        col = "category" if "category" in row.index else "subject"
        prompt = format_mmlu_pro_prompt(row, tokenizer)
        completions = []

        for k in range(N_COMPLETIONS_PER_Q):
            try:
                text, thinking_len = generate_completion(model, tokenizer, prompt)
                completions.append({"text": text, "thinking_len": thinking_len, "prompt": prompt})
            except Exception as e:
                log(f"  Generation failed: {e}")
                continue

        stats["n_questions"] += 1

        if len(completions) < 2:
            stats["n_skipped_no_diff"] += 1
            continue

        # Sort by thinking length
        completions.sort(key=lambda c: c["thinking_len"])
        short = completions[0]
        long_ = completions[-1]

        if long_["thinking_len"] - short["thinking_len"] < MIN_THINKING_DIFF:
            stats["n_skipped_no_diff"] += 1
            continue

        pairs.append({
            "prompt": prompt,
            "chosen": long_["text"],
            "rejected": short["text"],
            "chosen_thinking_len": long_["thinking_len"],
            "rejected_thinking_len": short["thinking_len"],
        })
        stats["n_pairs"] += 1

        if stats["n_pairs"] % 10 == 0:
            log(f"  {stats['n_pairs']} pairs collected so far")

    log(f"Phase 1 done: {stats['n_pairs']}/{stats['n_questions']} pairs valid")
    log(f"  Skipped (no diff): {stats['n_skipped_no_diff']}")

    pairs_path = DATA_DIR / "preference_pairs.json"
    with open(pairs_path, "w") as f:
        json.dump(pairs, f, indent=2)

    return pairs, stats


# ─── Phase 2: Precompute reference log-probs ──────────────────────────────────

def compute_sequence_log_probs(model, tokenizer, prompt, completion):
    """
    Compute sum of log-probs for completion tokens given prompt.
    Returns scalar (sum over completion tokens, not averaged).
    """
    full_text = prompt + completion
    tokens = tokenizer.encode(full_text)
    prompt_tokens = tokenizer.encode(prompt)
    n_prompt = len(prompt_tokens)

    if len(tokens) <= n_prompt:
        return 0.0

    input_ids = mx.array(tokens[:-1])[None]  # (1, T-1)
    target_ids = mx.array(tokens[1:])         # (T-1,)

    logits = model(input_ids)  # (1, T-1, vocab)
    logits = logits[0]          # (T-1, vocab)

    # Cross-entropy per token (nats)
    log_probs = -nn.losses.cross_entropy(logits, target_ids, reduction="none")  # (T-1,)

    # Only sum over completion tokens (after prompt)
    completion_start = n_prompt - 1  # index in T-1 space
    completion_start = max(0, min(completion_start, len(log_probs) - 1))
    completion_lp = mx.sum(log_probs[completion_start:])
    mx.eval(completion_lp)
    result = completion_lp.item()

    # Free memory
    del logits, log_probs, input_ids, target_ids, completion_lp
    gc.collect()
    mx.metal.clear_cache()

    return result


def phase2_precompute_ref_logprobs(model, tokenizer, pairs):
    """
    Precompute reference log-probs for all pairs.
    Saves to data/ref_logprobs.json.
    """
    log(f"Phase 2: precomputing reference log-probs for {len(pairs)} pairs")
    ref_lps = []

    for i, pair in enumerate(pairs):
        chosen_lp = compute_sequence_log_probs(
            model, tokenizer, pair["prompt"], pair["chosen"]
        )
        rejected_lp = compute_sequence_log_probs(
            model, tokenizer, pair["prompt"], pair["rejected"]
        )
        ref_lps.append({
            "chosen_ref_lp": chosen_lp,
            "rejected_ref_lp": rejected_lp,
        })
        if (i + 1) % 10 == 0:
            log(f"  {i+1}/{len(pairs)} ref log-probs computed")

    ref_lps_path = DATA_DIR / "ref_logprobs.json"
    with open(ref_lps_path, "w") as f:
        json.dump(ref_lps, f)

    log(f"Reference log-probs saved to {ref_lps_path}")
    return ref_lps


# ─── Phase 3: DPO Training ────────────────────────────────────────────────────

def dpo_loss_fn(model, tokenizer, batch):
    """
    Offline DPO loss for a single batch item.
    batch: dict with prompt, chosen, rejected, chosen_ref_lp, rejected_ref_lp
    """
    prompt = batch["prompt"]
    chosen = batch["chosen"]
    rejected = batch["rejected"]
    ref_chosen_lp = mx.array(batch["chosen_ref_lp"])
    ref_rejected_lp = mx.array(batch["rejected_ref_lp"])

    # Policy log-probs (computed with gradients)
    full_chosen = prompt + chosen
    full_rejected = prompt + rejected

    chosen_tokens = tokenizer.encode(full_chosen)
    rejected_tokens = tokenizer.encode(full_rejected)
    prompt_tokens = tokenizer.encode(prompt)
    n_prompt = len(prompt_tokens)

    def get_policy_lp(tokens, n_prompt):
        if len(tokens) <= n_prompt:
            return mx.array(0.0)
        input_ids = mx.array(tokens[:-1])[None]
        target_ids = mx.array(tokens[1:])
        logits = model(input_ids)[0]
        log_probs = -nn.losses.cross_entropy(logits, target_ids, reduction="none")
        comp_start = max(0, n_prompt - 1)
        return mx.sum(log_probs[comp_start:])

    policy_chosen_lp = get_policy_lp(chosen_tokens, n_prompt)
    policy_rejected_lp = get_policy_lp(rejected_tokens, n_prompt)

    # DPO loss: -log σ(β*(log π_θ(y_w) - log π_ref(y_w)) - β*(log π_θ(y_l) - log π_ref(y_l)))
    reward_chosen = BETA * (policy_chosen_lp - ref_chosen_lp)
    reward_rejected = BETA * (policy_rejected_lp - ref_rejected_lp)
    loss = -mx.log(mx.sigmoid(reward_chosen - reward_rejected))

    return loss


def phase3_dpo_training(model, tokenizer, pairs, ref_lps):
    """Train DPO with offline reference log-probs."""
    log(f"Phase 3: DPO training ({DPO_STEPS} steps, β={BETA}, lr={LR})")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    # Apply LoRA for trainable policy on top of GRPO adapter
    from mlx_lm.tuner.lora import LoRALinear
    import mlx.nn as nn_mlx

    # Freeze all existing parameters, add new LoRA layers
    model.freeze()

    # Apply LoRA to q_proj and v_proj in each attention layer
    n_lora_applied = 0
    for layer in model.model.layers:
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            for proj_name in ["q_proj", "v_proj"]:
                if hasattr(attn, proj_name):
                    proj = getattr(attn, proj_name)
                    lora_proj = LoRALinear.from_base(
                        proj, r=4, dropout=0.0, scale=1.0
                    )
                    lora_proj.unfreeze(keys=["lora_a", "lora_b"])
                    setattr(attn, proj_name, lora_proj)
                    n_lora_applied += 1

    log(f"  Applied LoRA to {n_lora_applied} projection layers (r=4)")

    optimizer = optim.Adam(learning_rate=LR)

    # Training loop
    all_items = [
        {**pairs[i], **ref_lps[i]}
        for i in range(len(pairs))
    ]

    train_losses = []
    step = 0

    def grad_fn(batch):
        return nn.value_and_grad(model, lambda m: dpo_loss_fn(m, tokenizer, batch))(model)

    for step in range(DPO_STEPS):
        idx = step % len(all_items)
        batch = all_items[idx]

        loss, grads = grad_fn(batch)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())

        loss_val = loss.item()
        train_losses.append(loss_val)

        if (step + 1) % 10 == 0:
            avg_loss = np.mean(train_losses[-10:])
            log(f"  Step {step+1}/{DPO_STEPS}: loss={avg_loss:.4f}")

        del loss, grads
        gc.collect()

    # Save adapter
    from mlx_lm import save
    save(ADAPTER_DIR, model, tokenizer)
    log(f"ThinkPO adapter saved to {ADAPTER_DIR}")

    return {
        "n_steps": DPO_STEPS,
        "final_loss": float(np.mean(train_losses[-10:])) if train_losses else None,
        "adapter_path": str(ADAPTER_DIR),
        "n_lora_layers": n_lora_applied,
    }


# ─── Phase 4: Evaluation ──────────────────────────────────────────────────────

def eval_mmlu_pro(model, tokenizer, df, label, n_per_cat=N_EVAL_PER_CAT):
    """Evaluate on MMLU-Pro with thinking mode."""
    log(f"Evaluating {label} on MMLU-Pro ({n_per_cat}/category)...")
    t0 = time.time()
    col = "category" if "category" in df.columns else "subject"

    per_cat = {}
    thinking_chars = []

    for cat in df[col].unique():
        cat_df = df[df[col] == cat].sample(min(n_per_cat, sum(df[col] == cat)), random_state=42)
        correct = 0
        for _, row in cat_df.iterrows():
            prompt = format_mmlu_pro_prompt(row, tokenizer)
            try:
                text, t_len = generate_completion(model, tokenizer, prompt)
                thinking_chars.append(t_len)
                ans = parse_answer(text)
                gt = row.get("answer", row.get("correct_answer", ""))
                if isinstance(gt, int):
                    gt = chr(65 + gt)
                if ans and ans.upper() == str(gt).upper():
                    correct += 1
            except Exception as e:
                log(f"  Eval error: {e}")

        per_cat[cat] = correct / max(1, len(cat_df))

    overall = np.mean(list(per_cat.values()))
    avg_thinking = np.mean(thinking_chars) if thinking_chars else 0.0
    elapsed = time.time() - t0

    log(f"  {label}: overall={overall:.3f}, avg_thinking={avg_thinking:.0f} chars, t={elapsed:.0f}s")
    return {
        "label": label,
        "overall_accuracy": float(overall),
        "per_category": {k: float(v) for k, v in per_cat.items()},
        "avg_thinking_chars": float(avg_thinking),
        "eval_time_s": elapsed,
    }


def eval_gsm8k(model, tokenizer, label, n=20):
    """Quick GSM8K eval."""
    log(f"Evaluating {label} on GSM8K (n={n})...")
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        import pandas as pd
        df = pd.DataFrame(ds).sample(n, random_state=42)
    except Exception as e:
        log(f"  GSM8K load failed: {e}")
        return {"label": label, "accuracy": None, "error": str(e)}

    correct = 0
    for _, row in df.iterrows():
        q = row["question"]
        messages = [{"role": "user", "content": f"Solve: {q}\nShow your work and give the final numerical answer."}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if THINKING_OPEN not in prompt:
            prompt += f"{THINKING_OPEN}\n"
        try:
            text, _ = generate_completion(model, tokenizer, prompt, max_tokens=MAX_TOKENS_GSM8K)
            # Extract final answer
            ans_text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
            nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", ans_text.replace(",", ""))
            pred = nums[-1] if nums else None
            gt_nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", row["answer"].replace(",", ""))
            gt = gt_nums[-1] if gt_nums else None
            if pred and gt and abs(float(pred) - float(gt)) < 0.01:
                correct += 1
        except Exception:
            pass

    acc = correct / n
    log(f"  GSM8K {label}: {correct}/{n} = {acc:.3f}")
    return {"label": label, "accuracy": float(acc), "n": n}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    from mlx_lm import load

    results = {
        "experiment": "exp_p11_thinkpo_polish",
        "model": MODEL_ID,
        "grpo_adapter": str(GRPO_ADAPTER_PATH),
        "is_smoke": IS_SMOKE,
        "beta": BETA,
        "dpo_steps": DPO_STEPS,
    }

    log(f"ThinkPO experiment starting (smoke={IS_SMOKE})")
    log(f"GRPO adapter: {GRPO_ADAPTER_PATH}")

    if not GRPO_ADAPTER_PATH.exists():
        raise RuntimeError(
            f"GRPO adapter not found at {GRPO_ADAPTER_PATH}. "
            "Run exp_p11_grpo_reasoning_adapter first."
        )

    # ── Load model with GRPO adapter ──────────────────────────────────────────
    log("Loading model + GRPO adapter...")
    model, tokenizer = load(MODEL_ID, adapter_path=str(GRPO_ADAPTER_PATH))
    mx.eval(model.parameters())

    # ── Load MMLU-Pro data ────────────────────────────────────────────────────
    mmlu_df = load_mmlu_pro(n_per_cat=N_EVAL_PER_CAT * 2)  # 2x for train + eval split

    # Split: half for preference generation, half for eval
    train_df = mmlu_df.sample(frac=0.5, random_state=42)
    eval_df = mmlu_df.drop(train_df.index)

    # ── Phase 1: Generate preference pairs ────────────────────────────────────
    pairs, phase1_stats = phase1_generate_pairs(model, tokenizer, train_df)
    results["phase1"] = phase1_stats

    if len(pairs) < 2:
        log("WARNING: Too few preference pairs. Checking fallback...")
        # Fallback: generate pairs using base model (no adapter) as "short"
        log("Falling back to base vs GRPO as short/long split")
        # Load base without adapter
        model_base, _ = load(MODEL_ID)
        mx.eval(model_base.parameters())
        fallback_pairs = []
        for _, row in train_df.head(10).iterrows():
            prompt = format_mmlu_pro_prompt(row, tokenizer)
            try:
                base_text, base_len = generate_completion(model_base, tokenizer, prompt)
                grpo_text, grpo_len = generate_completion(model, tokenizer, prompt)
                # GRPO model typically thinks longer due to RS-SFT training
                if grpo_len > base_len:
                    fallback_pairs.append({
                        "prompt": prompt,
                        "chosen": grpo_text,
                        "rejected": base_text,
                        "chosen_thinking_len": grpo_len,
                        "rejected_thinking_len": base_len,
                    })
            except Exception as e:
                log(f"  Fallback pair error: {e}")
        del model_base
        gc.collect()
        mx.metal.clear_cache()
        if fallback_pairs:
            pairs = fallback_pairs
            results["phase1"]["fallback_used"] = True
            results["phase1"]["n_pairs"] = len(pairs)

    results["phase1"]["avg_chosen_thinking"] = float(
        np.mean([p["chosen_thinking_len"] for p in pairs]) if pairs else 0.0
    )
    results["phase1"]["avg_rejected_thinking"] = float(
        np.mean([p["rejected_thinking_len"] for p in pairs]) if pairs else 0.0
    )

    # ── Phase 2: Precompute reference log-probs ───────────────────────────────
    ref_lps = phase2_precompute_ref_logprobs(model, tokenizer, pairs)
    results["phase2"] = {
        "n_pairs_with_ref_lps": len(ref_lps),
        "avg_ref_chosen_lp": float(np.mean([r["chosen_ref_lp"] for r in ref_lps])) if ref_lps else None,
        "avg_ref_rejected_lp": float(np.mean([r["rejected_ref_lp"] for r in ref_lps])) if ref_lps else None,
    }

    # ── Evaluate GRPO baseline (before ThinkPO) ───────────────────────────────
    grpo_mmlu = eval_mmlu_pro(model, tokenizer, eval_df, "grpo_baseline")
    grpo_gsm8k = eval_gsm8k(model, tokenizer, "grpo_baseline", n=5 if IS_SMOKE else 20)
    results["grpo_baseline_mmlu"] = grpo_mmlu
    results["grpo_baseline_gsm8k"] = grpo_gsm8k

    # ── Phase 3: DPO training ─────────────────────────────────────────────────
    phase3_stats = phase3_dpo_training(model, tokenizer, pairs, ref_lps)
    results["phase3"] = phase3_stats

    # Free cache after training
    gc.collect()
    mx.metal.clear_cache()

    # ── Evaluate ThinkPO model ────────────────────────────────────────────────
    thinkpo_mmlu = eval_mmlu_pro(model, tokenizer, eval_df, "thinkpo")
    thinkpo_gsm8k = eval_gsm8k(model, tokenizer, "thinkpo", n=5 if IS_SMOKE else 20)
    results["thinkpo_mmlu"] = thinkpo_mmlu
    results["thinkpo_gsm8k"] = thinkpo_gsm8k

    # ── Kill criteria ─────────────────────────────────────────────────────────
    grpo_acc = grpo_mmlu["overall_accuracy"]
    thinkpo_acc = thinkpo_mmlu["overall_accuracy"]
    grpo_thinking = grpo_mmlu["avg_thinking_chars"]
    thinkpo_thinking = thinkpo_mmlu["avg_thinking_chars"]
    grpo_gsm = grpo_gsm8k.get("accuracy") or 0.0
    thinkpo_gsm = thinkpo_gsm8k.get("accuracy") or 0.0

    k1499_pass = thinkpo_acc >= grpo_acc + 0.02
    k1500_pass = thinkpo_thinking >= grpo_thinking * 1.10
    k1501_pass = thinkpo_gsm >= grpo_gsm - 0.05

    results["kill_criteria"] = {
        "K1499_thinkpo_ge_grpo_plus_2pp": {
            "pass": k1499_pass,
            "thinkpo": float(thinkpo_acc),
            "grpo": float(grpo_acc),
            "delta": float(thinkpo_acc - grpo_acc),
            "threshold": 0.02,
        },
        "K1500_thinking_len_increase_10pct": {
            "pass": k1500_pass,
            "thinkpo_thinking": float(thinkpo_thinking),
            "grpo_thinking": float(grpo_thinking),
            "ratio": float(thinkpo_thinking / max(1.0, grpo_thinking)),
            "threshold": 1.10,
        },
        "K1501_no_gsm8k_regression": {
            "pass": k1501_pass,
            "thinkpo_gsm8k": float(thinkpo_gsm),
            "grpo_gsm8k": float(grpo_gsm),
            "delta": float(thinkpo_gsm - grpo_gsm),
            "threshold": -0.05,
        },
    }

    log(f"\n=== Kill Criteria ===")
    log(f"K1499 (ThinkPO >= GRPO+2pp): {'PASS' if k1499_pass else 'FAIL'} ({thinkpo_acc:.3f} vs {grpo_acc:.3f}+0.02)")
    log(f"K1500 (thinking +10%):       {'PASS' if k1500_pass else 'FAIL'} ({thinkpo_thinking:.0f} vs {grpo_thinking:.0f}*1.1)")
    log(f"K1501 (no GSM8K regression): {'PASS' if k1501_pass else 'FAIL'} ({thinkpo_gsm:.3f} vs {grpo_gsm:.3f}-0.05)")

    # ── Save results ──────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    main()
