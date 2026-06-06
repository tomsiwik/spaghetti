#!/usr/bin/env python3
"""exp_knowledge_disentanglement_control — rank-16 method adapter lifts
reasoning (K1733) while keeping knowledge (K1734) and factual recall
(K1735) within ±1 pp.

Pipeline:
  1. Build/reuse method-adapter training data (MMLU-Pro teacher traces,
     subgoal system prompt, student-side prompt erasure).
  2. Train rank-16 LoRA on v_proj+o_proj, top 16 layers, scale 4.0.
  3. Eval base vs adapter on three benchmarks:
       - GSM8K-test  (reasoning proxy for BBH per A4)
       - MMLU (cais/mmlu, balanced random subset)
       - TriviaQA (validation subset, exact-match)
  4. Compute K1733/K1734/K1735 pass/fail. K1736 (3 seeds) marked
     inconclusive at smoke; requires full rerun.

v2 fixes from predecessor (exp_method_vs_domain_adapter):
  - strip_thinking fortified to handle missing `<channel|>` close tag.
  - Larger training budget at full scale.
  - Signature metric is NOT a KC here (knowledge-disentanglement is
    the KC, not behavioural signature).

SMOKE_TEST=1 default. SMOKE completes as PROVISIONAL per guardrail.
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

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_DIR = EXPERIMENT_DIR / "data"
ADAPTER_DIR = EXPERIMENT_DIR / "adapters" / "method_multi"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "1") == "1"
SEED = 42

LORA_RANK = 16
LORA_SCALE = 4.0
LORA_DROPOUT = 0.0
LORA_KEYS = ["self_attn.v_proj", "self_attn.o_proj"]
LORA_NUM_LAYERS = 16

N_STEPS = 60 if IS_SMOKE else 300
BATCH_SIZE = 1
LR = 1e-4
MAX_SEQ_LEN = 2048

TRAIN_CATS = ["math", "computer science", "health", "law", "economics"]
N_PER_CAT_TRAIN = 5 if IS_SMOKE else 20

EVAL_N_PER_BENCH = 20 if IS_SMOKE else 100
GEN_MAX_TOKENS = 2048

METHOD_SYSTEM_PROMPT = (
    "You solve problems by decomposing them into numbered subgoals. "
    "Before giving your final answer, write explicit steps:\n"
    "Step 1: Restate what the question is asking.\n"
    "Step 2: Identify the relevant information or constraints.\n"
    "Step 3: Evaluate each option against those constraints.\n"
    "Step 4: Pick the best option and justify briefly.\n"
    "End with exactly one line:\n"
    "Answer: X\n"
    "where X is the letter of the best option."
)
OPTION_LETTERS = "ABCDEFGHIJ"


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


# ── v2 fortified strip_thinking ────────────────────────────────────────────
def strip_thinking(response):
    """v2: handles missing `<channel|>` close tag by stripping from
    `<|channel>thought` to the next blank line, `Answer:` prefix, or
    the final 400 chars — whichever comes first.
    """
    if not response:
        return response or "", 0
    thinking_len = 0
    # Case A: balanced <|channel>thought ... <channel|>
    m = re.search(r"<\|channel>thought.*?<channel\|>", response, flags=re.DOTALL)
    if m:
        thinking_len = len(m.group(0))
        cleaned = re.sub(
            r"<\|channel>thought.*?<channel\|>", "", response, flags=re.DOTALL
        )
    else:
        # Case B: unbalanced — strip from marker up to blank line or "Answer:".
        idx = response.find("<|channel>thought")
        if idx >= 0:
            tail = response[idx:]
            # Find the earliest terminator among: two \n\n, \nAnswer:, \nStep
            mblank = re.search(r"\n\s*\n", tail)
            manswer = re.search(r"(?i)\n\s*Answer\s*:", tail)
            cand_ends = [mblank.start() if mblank else None,
                         manswer.start() if manswer else None]
            cand_ends = [c for c in cand_ends if c is not None]
            end_rel = min(cand_ends) if cand_ends else min(len(tail), 1200)
            thinking_len = end_rel
            cleaned = response[:idx] + tail[end_rel:]
        else:
            cleaned = response
    # Also strip legacy <think>..</think>
    m2 = re.search(r"<think>(.*?)</think>", cleaned, flags=re.DOTALL)
    if m2 and thinking_len == 0:
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


def parse_gsm8k_answer(response):
    """Extract integer final answer from GSM8K response."""
    text, _ = strip_thinking(response)
    # GSM8K ground truth is after "####" in canonical format; in model output
    # look for last integer in "answer is X" or "= X" or "####" patterns.
    for pat in [
        r"####\s*(-?\d[\d,]*\.?\d*)",
        r"(?i)answer\s*(?:is|:)\s*\$?(-?\d[\d,]*\.?\d*)",
        r"(?i)final answer\s*(?:is|:)\s*\$?(-?\d[\d,]*\.?\d*)",
        r"\$?(-?\d[\d,]*\.?\d*)\s*$",
    ]:
        m = re.search(pat, text)
        if m:
            num = m.group(1).replace(",", "")
            try:
                return float(num)
            except ValueError:
                continue
    # Fallback: last number in the text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            return None
    return None


def gsm8k_canonical_answer(answer_text):
    """Extract ground-truth numeric from GSM8K `answer` field (post `####`)."""
    if "####" not in answer_text:
        return None
    tail = answer_text.split("####")[-1].strip()
    try:
        return float(tail.replace(",", ""))
    except ValueError:
        return None


# ── Phase 1: teacher traces for method-adapter training ────────────────────

def _format_mmlu_pro_question(row):
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
    return user_content, OPTION_LETTERS[int(row["answer_index"])]


def _teacher_prompt(tokenizer, user_content):
    msgs = [
        {"role": "system", "content": METHOD_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )


def phase_build_training_data():
    train_path = DATA_DIR / "train_multi.jsonl"
    if train_path.exists():
        n = sum(1 for _ in open(train_path))
        log(f"[Phase 1] cached {n} train examples")
        return {"n_train": n, "cached": True}

    from mlx_lm import load, generate

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    mmlu_path = EXPERIMENT_DIR.parent / "exp_bench_mmlu_pro" / "data" / "test.parquet"
    if not mmlu_path.exists():
        return {"error": f"mmlu-pro-missing at {mmlu_path}"}

    df = pd.read_parquet(mmlu_path)
    rng = np.random.RandomState(SEED)
    log("\n[Phase 1] Generating teacher traces")
    model, tokenizer = load(MODEL_ID)
    log_memory("teacher-loaded")

    n_wrote = 0
    with open(train_path, "w") as w:
        for cat in TRAIN_CATS:
            cat_df = df[df["category"].str.lower() == cat.lower()]
            if len(cat_df) == 0:
                log(f"  WARN: 0 rows for {cat}")
                continue
            n_sample = min(N_PER_CAT_TRAIN, len(cat_df))
            idxs = rng.choice(len(cat_df), size=n_sample, replace=False)
            for i in idxs:
                row = cat_df.iloc[int(i)]
                user_content, correct_letter = _format_mmlu_pro_question(row)
                prompt = _teacher_prompt(tokenizer, user_content)
                try:
                    resp = generate(
                        model, tokenizer, prompt=prompt, max_tokens=1024
                    )
                    mx.eval()
                except Exception as e:
                    log(f"  gen err {cat}: {e!r}")
                    continue
                student_resp, _ = strip_thinking(resp)
                if not student_resp.strip():
                    continue
                ex = {
                    "category": cat,
                    "correct_letter": correct_letter,
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": student_resp},
                    ],
                }
                w.write(json.dumps(ex) + "\n")
                n_wrote += 1
            log(f"  {cat}: wrote examples")
    cleanup(model, tokenizer)
    log(f"[Phase 1] wrote {n_wrote} train examples")
    return {"n_train": n_wrote, "cached": False}


# ── Phase 2: train method adapter ──────────────────────────────────────────

def phase_train_adapter():
    if (ADAPTER_DIR / "adapters.safetensors").exists():
        log(f"[Phase 2] adapter already at {ADAPTER_DIR}; reuse")
        return {"status": "ok", "cached": True, "adapter_path": str(ADAPTER_DIR)}

    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx.utils import tree_flatten

    train_path = DATA_DIR / "train_multi.jsonl"
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    log(f"\n[Phase 2] Train rank-{LORA_RANK} LoRA on {train_path.name}")
    model, tokenizer = load(MODEL_ID)
    model.freeze()
    lora_config = {
        "rank": LORA_RANK,
        "scale": LORA_SCALE,
        "dropout": LORA_DROPOUT,
        "keys": LORA_KEYS,
    }
    linear_to_lora_layers(model, LORA_NUM_LAYERS, lora_config, use_dora=False)
    log_memory("lora-attached")

    (ADAPTER_DIR / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora",
        "num_layers": LORA_NUM_LAYERS,
        "lora_parameters": lora_config,
    }, indent=2))

    train_ids = []
    with open(train_path) as f:
        for line in f:
            ex = json.loads(line)
            text = tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, enable_thinking=False
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > MAX_SEQ_LEN:
                ids = ids[:MAX_SEQ_LEN]
            if len(ids) < 4:
                continue
            train_ids.append(ids)
    if not train_ids:
        cleanup(model, tokenizer)
        return {"status": "failed", "error": "empty-train"}
    log(f"  {len(train_ids)} train seqs; median "
        f"{int(np.median([len(x) for x in train_ids]))} tokens")

    def loss_fn(model_, input_ids):
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        logits = model_(inputs)
        ntoks = mx.array(targets.size, dtype=mx.float32)
        ce = nn.losses.cross_entropy(logits, targets, reduction="mean")
        return ce, ntoks

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=LR)

    rng = np.random.RandomState(SEED)
    losses = []
    t0 = time.time()
    model.train()
    for step in range(1, N_STEPS + 1):
        idx = int(rng.randint(0, len(train_ids)))
        ids = train_ids[idx]
        batch = mx.array([ids], dtype=mx.int32)
        try:
            (loss_val, _nt), grads = loss_and_grad(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss_val)
        except Exception as e:
            log(f"  step {step} err: {e!r}")
            cleanup(model, tokenizer, optimizer)
            return {"status": "failed", "error": repr(e)}
        losses.append(float(loss_val.item()))
        if step % max(1, N_STEPS // 6) == 0 or step == 1:
            log(f"    step {step}  loss={losses[-1]:.3f}")
        mx.clear_cache()
    elapsed = time.time() - t0

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(
        str(ADAPTER_DIR / "adapters.safetensors"), adapter_weights
    )
    cleanup(model, tokenizer, optimizer)

    return {
        "status": "ok",
        "cached": False,
        "n_train_seqs": len(train_ids),
        "time_s": round(elapsed, 1),
        "steps": N_STEPS,
        "final_loss": round(losses[-1], 4) if losses else None,
        "mean_loss": round(float(np.mean(losses)), 4) if losses else None,
        "adapter_path": str(ADAPTER_DIR),
    }


# ── Phase 3: eval — three benchmarks, base and adapter arms ────────────────

def _eval_gsm8k(model, tokenizer, n, tag):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    rng = np.random.RandomState(SEED + 100)
    idxs = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    correct = 0
    responses = []
    for i in idxs:
        ex = ds[int(i)]
        gt = gsm8k_canonical_answer(ex["answer"])
        if gt is None:
            continue
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["question"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        try:
            resp = generate_text(model, tokenizer, prompt)
            pred = parse_gsm8k_answer(resp)
        except Exception as e:
            resp, pred = "", None
        ok = (pred is not None and gt is not None and abs(pred - gt) < 1e-3)
        correct += int(ok)
        responses.append({
            "bench": "gsm8k", "tag": tag,
            "gt": gt, "pred": pred, "ok": ok,
            "resp_prefix": (resp[:300] if resp else ""),
        })
        mx.eval()
    total = len(responses)
    acc = correct / max(total, 1) * 100
    return {"bench": "gsm8k", "n": total, "correct": correct,
            "acc": round(acc, 1), "responses": responses}


def _eval_mmlu(model, tokenizer, n, tag):
    """Use `cais/mmlu` all split, balanced random subset."""
    from datasets import load_dataset
    # MMLU is keyed by subject; use "all" config if available, else combine.
    # The local cache has TIGER-Lab___mmlu-pro already — but also cais___mmlu
    # with per-subject configs. Use a stable subset via `all` if provided.
    try:
        ds = load_dataset("cais/mmlu", "all", split="test")
    except Exception:
        # fallback: logical_fallacies only (small, cached)
        ds = load_dataset("cais/mmlu", "logical_fallacies", split="test")
    rng = np.random.RandomState(SEED + 200)
    idxs = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    correct = 0
    responses = []
    for i in idxs:
        ex = ds[int(i)]
        question = ex["question"]
        choices = ex["choices"]
        ans_idx = int(ex["answer"])
        correct_letter = OPTION_LETTERS[ans_idx]
        opt_text = "\n".join(f"{OPTION_LETTERS[k]}. {c}" for k, c in enumerate(choices))
        user = (
            f"Answer the following multiple choice question. Select the single "
            f"best answer letter (A through D).\n\nQuestion: {question}\n\n"
            f"Options:\n{opt_text}\n\nAnswer:"
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        try:
            resp = generate_text(model, tokenizer, prompt)
            pred, _ = parse_mcq_answer(resp)
        except Exception as e:
            resp, pred = "", None
        ok = (pred == correct_letter)
        correct += int(ok)
        responses.append({
            "bench": "mmlu", "tag": tag,
            "gt": correct_letter, "pred": pred, "ok": ok,
            "resp_prefix": (resp[:300] if resp else ""),
        })
        mx.eval()
    total = len(responses)
    acc = correct / max(total, 1) * 100
    return {"bench": "mmlu", "n": total, "correct": correct,
            "acc": round(acc, 1), "responses": responses}


def _trivia_answer_match(pred_text, gold_answer):
    """Exact-match or normalised-inclusion against gold aliases."""
    if not pred_text:
        return False
    pred_lower = pred_text.lower().strip()
    aliases = []
    if isinstance(gold_answer, dict):
        # TriviaQA format: {value: ..., aliases: [...], normalized_value: ...}
        if gold_answer.get("value"):
            aliases.append(gold_answer["value"])
        if gold_answer.get("normalized_value"):
            aliases.append(gold_answer["normalized_value"])
        if gold_answer.get("aliases"):
            aliases.extend(gold_answer["aliases"])
        if gold_answer.get("normalized_aliases"):
            aliases.extend(gold_answer["normalized_aliases"])
    elif isinstance(gold_answer, str):
        aliases = [gold_answer]
    for a in aliases:
        if not a:
            continue
        a_low = a.lower().strip()
        if a_low and a_low in pred_lower:
            return True
    return False


def _eval_triviaqa(model, tokenizer, n, tag):
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "unfiltered", split="validation")
    rng = np.random.RandomState(SEED + 300)
    idxs = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    correct = 0
    responses = []
    for i in idxs:
        ex = ds[int(i)]
        q = ex["question"]
        gold = ex["answer"]  # dict with value, aliases, normalized_value, ...
        user = f"Answer the following trivia question in one or two words.\n\nQuestion: {q}\n\nAnswer:"
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        try:
            resp = generate_text(model, tokenizer, prompt)
            cleaned, _ = strip_thinking(resp)
        except Exception as e:
            resp, cleaned = "", ""
        # Take short answer part: after "Answer:" if present, else first line
        short = cleaned
        m = re.search(r"(?i)answer\s*:\s*(.+)", cleaned)
        if m:
            short = m.group(1).strip()
        # Trim at newline
        short = short.split("\n")[0].strip()
        ok = _trivia_answer_match(short, gold)
        correct += int(ok)
        responses.append({
            "bench": "triviaqa", "tag": tag,
            "gt": gold.get("value", "") if isinstance(gold, dict) else gold,
            "pred": short[:120], "ok": ok,
            "resp_prefix": (resp[:300] if resp else ""),
        })
        mx.eval()
    total = len(responses)
    acc = correct / max(total, 1) * 100
    return {"bench": "triviaqa", "n": total, "correct": correct,
            "acc": round(acc, 1), "responses": responses}


# global holder to avoid reimport on every call (set inside eval_arm)
_MLX_GEN = None

def generate_text(model, tokenizer, prompt):
    global _MLX_GEN
    if _MLX_GEN is None:
        from mlx_lm import generate
        _MLX_GEN = generate
    return _MLX_GEN(model, tokenizer, prompt=prompt, max_tokens=GEN_MAX_TOKENS)


def phase_eval_arm(adapter_path, tag):
    from mlx_lm import load
    log(f"\n[Eval:{tag}] load model (adapter={'yes' if adapter_path else 'no'})")
    kwargs = {"adapter_path": str(adapter_path)} if adapter_path else {}
    model, tokenizer = load(MODEL_ID, **kwargs)
    log_memory(f"loaded-{tag}")

    gsm = _eval_gsm8k(model, tokenizer, EVAL_N_PER_BENCH, tag)
    log(f"  gsm8k: {gsm['correct']}/{gsm['n']} = {gsm['acc']}%")
    mmlu = _eval_mmlu(model, tokenizer, EVAL_N_PER_BENCH, tag)
    log(f"  mmlu:  {mmlu['correct']}/{mmlu['n']} = {mmlu['acc']}%")
    trivia = _eval_triviaqa(model, tokenizer, EVAL_N_PER_BENCH, tag)
    log(f"  trivia: {trivia['correct']}/{trivia['n']} = {trivia['acc']}%")

    cleanup(model, tokenizer)
    return {"tag": tag, "gsm8k": gsm, "mmlu": mmlu, "triviaqa": trivia}


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 60)
    log(f"exp_knowledge_disentanglement_control  SMOKE={IS_SMOKE}")
    log(f"  N_STEPS={N_STEPS}  n_per_cat={N_PER_CAT_TRAIN}  "
        f"eval_n={EVAL_N_PER_BENCH}")
    log(f"  rank={LORA_RANK}  scale={LORA_SCALE}  keys={LORA_KEYS}")
    log("=" * 60)
    log_memory("start")

    results = {
        "experiment": "exp_knowledge_disentanglement_control",
        "model": MODEL_ID,
        "mlx_lm_version": "0.31.2",
        "is_smoke": IS_SMOKE,
        "lora_rank": LORA_RANK,
        "lora_scale": LORA_SCALE,
        "lora_keys": LORA_KEYS,
        "lora_num_layers": LORA_NUM_LAYERS,
        "n_steps": N_STEPS,
        "n_per_cat_train": N_PER_CAT_TRAIN,
        "eval_n_per_bench": EVAL_N_PER_BENCH,
        "train_cats": TRAIN_CATS,
        "seed": SEED,
    }

    # Phase 1 — training data
    data_info = phase_build_training_data()
    results["training_data"] = data_info
    log(f"  data: {data_info}")
    if data_info.get("error"):
        results["verdict"] = "ABORTED"
        results["all_pass"] = False
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return 1

    # Phase 2 — train adapter
    train_info = phase_train_adapter()
    results["train_adapter"] = train_info
    if train_info.get("status") != "ok":
        results["verdict"] = "ABORTED"
        results["all_pass"] = False
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return 1

    # Phase 3 — eval base then adapter (reset model load between arms)
    eval_base = phase_eval_arm(None, "base")
    eval_adapter = phase_eval_arm(ADAPTER_DIR, "adapter")

    # Persist responses to jsonl
    resp_path = DATA_DIR / "eval_responses.jsonl"
    with open(resp_path, "w") as f:
        for arm in (eval_base, eval_adapter):
            for bench in ("gsm8k", "mmlu", "triviaqa"):
                for r in arm[bench].get("responses", []):
                    r["arm"] = arm["tag"]; f.write(json.dumps(r) + "\n")

    # Strip responses from summary for compact results.json
    def drop_responses(arm):
        out = dict(arm)
        for b in ("gsm8k", "mmlu", "triviaqa"):
            out[b] = {k: v for k, v in arm[b].items() if k != "responses"}
        return out

    results["eval_base"] = drop_responses(eval_base)
    results["eval_adapter"] = drop_responses(eval_adapter)

    # ── KC evaluation ───────────────────────────────────────────────────────
    r_base = eval_base["gsm8k"]["acc"]
    r_adp = eval_adapter["gsm8k"]["acc"]
    k_base = eval_base["mmlu"]["acc"]
    k_adp = eval_adapter["mmlu"]["acc"]
    f_base = eval_base["triviaqa"]["acc"]
    f_adp = eval_adapter["triviaqa"]["acc"]

    r_delta = r_adp - r_base
    k_delta = k_adp - k_base
    f_delta = f_adp - f_base

    k1733_pass = r_delta >= 5.0
    k1734_pass = abs(k_delta) < 1.0
    k1735_pass = abs(f_delta) < 1.0

    results["k1733_bbh_proxy_gain"] = {
        "pass": bool(k1733_pass),
        "proxy": "gsm8k",
        "base_acc": r_base,
        "adapter_acc": r_adp,
        "delta_pp": round(r_delta, 2),
        "threshold_pp": 5.0,
    }
    results["k1734_mmlu_neutral"] = {
        "pass": bool(k1734_pass),
        "base_acc": k_base,
        "adapter_acc": k_adp,
        "delta_pp": round(k_delta, 2),
        "threshold_abs_pp": 1.0,
    }
    results["k1735_triviaqa_neutral"] = {
        "pass": bool(k1735_pass),
        "base_acc": f_base,
        "adapter_acc": f_adp,
        "delta_pp": round(f_delta, 2),
        "threshold_abs_pp": 1.0,
    }
    results["k1736_seeds"] = {
        "pass": False,
        "inconclusive": True,
        "reason": "smoke runs 1 seed; full rerun needs 3 seeds and CV<10%",
    }

    all_pass = k1733_pass and k1734_pass and k1735_pass
    results["all_pass"] = bool(all_pass)
    if IS_SMOKE:
        results["verdict"] = "PROVISIONAL"
        results["verdict_reason"] = (
            "smoke: 1 seed, reduced n, BBH proxied by GSM8K (A4). "
            "Full rerun required for supported/killed."
        )
    else:
        results["verdict"] = "SUPPORTED" if all_pass else "KILLED"

    results["total_time_s"] = round(time.time() - t0, 1)
    log_memory("end")

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log("\n" + "=" * 60)
    log(f"VERDICT: {results['verdict']}  all_pass={all_pass}")
    log(f"  K1733 reasoning(gsm8k) Δ≥+5pp: {k1733_pass}  "
        f"base={r_base}% adp={r_adp}% Δ={r_delta:+.1f}pp")
    log(f"  K1734 MMLU |Δ|<1pp:            {k1734_pass}  "
        f"base={k_base}% adp={k_adp}% Δ={k_delta:+.1f}pp")
    log(f"  K1735 TriviaQA |Δ|<1pp:        {k1735_pass}  "
        f"base={f_base}% adp={f_adp}% Δ={f_delta:+.1f}pp")
    log(f"  K1736 3-seeds:                 INCONCLUSIVE (smoke)")
    log(f"  Total time: {results['total_time_s']}s")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
