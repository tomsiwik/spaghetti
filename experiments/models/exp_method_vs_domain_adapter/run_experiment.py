#!/usr/bin/env python3
"""exp_method_vs_domain_adapter — rank-16 method adapter vs single-domain control.

Hypothesis (MATH.md):
  Training a rank-16 LoRA on decompose-into-subgoals teacher traces across 5
  diverse MMLU-Pro categories encodes a transferable method subspace, while the
  same procedure on one category encodes method+domain and fails to transfer.

Pipeline:
  1. Load MMLU-Pro, partition train_cats (math, cs, health, law, economics)
     and heldout_cats (physics, biology, philosophy, psychology, history).
  2. Generate teacher traces from π_0 under a subgoal-decomposition system
     prompt. Two training mixtures:
       - multi: n_per_cat × 5 cats
       - single: n_per_cat × 5 copies of `math` only (same total examples)
  3. Train two rank-16 LoRA adapters with identical budgets.
  4. Eval base + both adapters on heldout cats × eval_per_cat. Measure:
       - K1718 (multi beats base on ≥3/5 heldout)
       - K1719 (single beats base on ≤1/5 heldout)
       - K1720 (multi signature rate ≥70 % AND ≥+20 pp vs base)

SMOKE_TEST=1 default (≈25 min on M5 Pro). SMOKE_TEST=0 is full run (≈2.5 h);
SMOKE verdict is always PROVISIONAL per guardrail.
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
ADAPTER_MULTI_DIR = EXPERIMENT_DIR / "adapters" / "method_multi"
ADAPTER_SINGLE_DIR = EXPERIMENT_DIR / "adapters" / "method_single_math"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "1") == "1"
SEED = 42

LORA_RANK = 16
LORA_SCALE = 4.0
LORA_DROPOUT = 0.0
LORA_KEYS = ["self_attn.v_proj", "self_attn.o_proj"]
LORA_NUM_LAYERS = 16

N_STEPS = 40 if IS_SMOKE else 300
BATCH_SIZE = 1
LR = 1e-4
MAX_SEQ_LEN = 2048

TRAIN_CATS = ["math", "computer science", "health", "law", "economics"]
HELDOUT_CATS = ["physics", "biology", "philosophy", "psychology", "history"]
SINGLE_DOMAIN = "math"

N_PER_CAT_TRAIN = 3 if IS_SMOKE else 20          # examples per category for multi
# Single-domain adapter uses N_PER_CAT_TRAIN * len(TRAIN_CATS) math examples:
N_SINGLE_DOMAIN = N_PER_CAT_TRAIN * len(TRAIN_CATS)
EVAL_PER_CAT = 3 if IS_SMOKE else 15

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


# ── strip_thinking (copied from exp_score_kl_constrained_mcq, verified) ────
def strip_thinking(response):
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


def count_subgoal_markers(response):
    """Behavioural signature per MATH.md K1720.

    Counts distinct categories of subgoal markers in `response` (post-think-strip):
      - explicit `Step N` form
      - `^N.` numbered enumeration at line start
      - connectives: First, Second, Next, Finally
      - bullet markers (at least 2 bullets to qualify)

    Returns an int count. count >= 2 == signature-present.
    """
    text, _ = strip_thinking(response or "")
    score = 0
    # Rule 1: "Step 1" ... "Step 2" — need ≥2 distinct step indices
    steps = re.findall(r"(?i)\bStep\s*(\d+)\b", text)
    if len(set(steps)) >= 2:
        score += 1
    # Rule 2: numbered enumeration at line start
    numbered = re.findall(r"(?m)^\s*(\d+)[.)]\s", text)
    if len(set(numbered)) >= 2:
        score += 1
    # Rule 3: connectives
    connectives_hit = sum(
        1 for w in ("first", "second", "third", "next", "finally", "then")
        if re.search(rf"(?i)\b{w}\b[,:]", text)
    )
    if connectives_hit >= 2:
        score += 1
    # Rule 4: bullet list of length ≥ 2
    bullets = re.findall(r"(?m)^\s*[-*]\s+\S", text)
    if len(bullets) >= 2:
        score += 1
    return score


# ── Phase 1: build training data via teacher generation ─────────────────────

def _format_question(row):
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


def _build_teacher_prompt(tokenizer, user_content, with_method_prompt):
    """Render the chat prompt. If with_method_prompt, prepend a system message
    that instructs subgoal decomposition — this is the teacher-only prompt that
    gets ERASED from the student training data (Orca-2 prompt erasure)."""
    msgs = []
    if with_method_prompt:
        msgs.append({"role": "system", "content": METHOD_SYSTEM_PROMPT})
    msgs.append({"role": "user", "content": user_content})
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )


def phase_build_training_data():
    """Generate teacher traces once for (train_cats × N_PER_CAT) questions and
    once for (math × N_SINGLE_DOMAIN) questions. Also: validate the teacher's
    subgoal signature rate as a pre-training gate (must be ≥ 0.70, else abort
    — anti-pattern: training on teacher traces without the method would be
    an unmeasured control failure).

    Writes:
      data/train_multi.jsonl  — {"question_id", "category", "messages" (user+assistant)}
      data/train_single.jsonl
      data/teacher_stats.json — teacher signature rate + sample responses
    """
    from mlx_lm import load, generate

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    multi_path = DATA_DIR / "train_multi.jsonl"
    single_path = DATA_DIR / "train_single.jsonl"
    stats_path = DATA_DIR / "teacher_stats.json"

    if multi_path.exists() and single_path.exists() and stats_path.exists():
        log("[Phase 1] Training data exists; skipping teacher generation.")
        n_multi = sum(1 for _ in open(multi_path))
        n_single = sum(1 for _ in open(single_path))
        stats = json.loads(stats_path.read_text())
        return {
            "n_multi": n_multi,
            "n_single": n_single,
            "teacher_signature_rate": stats.get("signature_rate"),
            "teacher_signature_pass": stats.get("signature_rate", 0) >= 0.70,
            "cached": True,
        }

    mmlu_path = EXPERIMENT_DIR.parent / "exp_bench_mmlu_pro" / "data" / "test.parquet"
    if not mmlu_path.exists():
        return {"error": "mmlu-pro-missing", "path": str(mmlu_path)}

    df = pd.read_parquet(mmlu_path)
    rng = np.random.RandomState(SEED)

    log("\n[Phase 1] Generating teacher traces for training data")
    model, tokenizer = load(MODEL_ID)
    log_memory("teacher-loaded")

    def gen_examples(cat, n, writer, log_list=None):
        cat_df = df[df["category"].str.lower() == cat.lower()]
        if len(cat_df) == 0:
            log(f"  WARN: 0 rows for category {cat}")
            return 0
        n_avail = min(n, len(cat_df))
        idxs = rng.choice(len(cat_df), size=n_avail, replace=False)
        wrote = 0
        for i in idxs:
            row = cat_df.iloc[int(i)]
            user_content, correct_letter = _format_question(row)
            prompt_teacher = _build_teacher_prompt(
                tokenizer, user_content, with_method_prompt=True
            )
            try:
                resp = generate(
                    model, tokenizer, prompt=prompt_teacher, max_tokens=1024
                )
                mx.eval()
            except Exception as e:
                log(f"  teacher gen err cat={cat}: {e!r}")
                continue
            # Student training: STRIP thinking but KEEP the subgoal text; erase
            # the system prompt by rendering the student-side messages WITHOUT
            # the method system prompt (Orca-2 prompt erasure).
            student_resp, _tchars = strip_thinking(resp)
            if not student_resp.strip():
                continue
            msgs_student = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": student_resp},
            ]
            ex = {
                "question_id": int(row.get("question_id", -1)),
                "category": cat,
                "correct_letter": correct_letter,
                "messages": msgs_student,
                "subgoal_markers": count_subgoal_markers(student_resp),
            }
            writer.write(json.dumps(ex) + "\n")
            if log_list is not None:
                log_list.append(ex)
            wrote += 1
        return wrote

    # Multi-domain mixture
    n_multi = 0
    all_train_examples = []
    with open(multi_path, "w") as w:
        for cat in TRAIN_CATS:
            wrote = gen_examples(cat, N_PER_CAT_TRAIN, w, log_list=all_train_examples)
            log(f"  multi {cat}: wrote {wrote}")
            n_multi += wrote

    # Single-domain ablation: same total examples, math only
    n_single = 0
    with open(single_path, "w") as w:
        wrote = gen_examples(SINGLE_DOMAIN, N_SINGLE_DOMAIN, w)
        log(f"  single {SINGLE_DOMAIN}: wrote {wrote}")
        n_single = wrote

    # Teacher signature stats on the multi-domain set (these are the sentinel
    # examples — if the teacher isn't producing subgoal structure, training on
    # them won't either).
    sig_hits = sum(1 for ex in all_train_examples if ex["subgoal_markers"] >= 2)
    signature_rate = sig_hits / max(len(all_train_examples), 1)
    stats = {
        "n_train_multi_examples": n_multi,
        "n_train_single_examples": n_single,
        "signature_hits": sig_hits,
        "signature_rate": round(signature_rate, 3),
        "sample_response": (all_train_examples[0]["messages"][-1]["content"][:600]
                            if all_train_examples else None),
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    log(f"  teacher signature rate: {signature_rate:.2%} ({sig_hits}/{len(all_train_examples)})")

    cleanup(model, tokenizer)

    return {
        "n_multi": n_multi,
        "n_single": n_single,
        "teacher_signature_rate": round(signature_rate, 3),
        "teacher_signature_pass": signature_rate >= 0.70,
        "cached": False,
    }


# ── Phase 2: train a LoRA adapter from a JSONL of student (user, assistant) ─

def phase_train_adapter(train_jsonl_path, adapter_out_dir, tag):
    """Train rank-16 LoRA on student-side (user, assistant) messages.

    The student data has the method system prompt ERASED (prompt erasure).
    Training objective: standard LM cross-entropy on the full rendered
    chat template, masking nothing (all tokens supervised — small training
    set, avoids having to split prompt from answer).
    """
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx.utils import tree_flatten

    log(f"\n[Phase 2:{tag}] Train adapter from {train_jsonl_path.name} -> {adapter_out_dir.name}")
    adapter_out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load(MODEL_ID)
    model.freeze()
    lora_config = {
        "rank": LORA_RANK,
        "scale": LORA_SCALE,
        "dropout": LORA_DROPOUT,
        "keys": LORA_KEYS,
    }
    linear_to_lora_layers(model, LORA_NUM_LAYERS, lora_config, use_dora=False)
    log_memory(f"lora-attached-{tag}")

    (adapter_out_dir / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora",
        "num_layers": LORA_NUM_LAYERS,
        "lora_parameters": lora_config,
    }, indent=2))

    train_ids = []
    with open(train_jsonl_path) as f:
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
        log(f"  ERR: no training ids for {tag}")
        cleanup(model, tokenizer)
        return {"status": "failed", "error": "empty-train"}

    log(f"  {len(train_ids)} train seqs; median len "
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

    rng = np.random.RandomState(SEED + (0 if tag == "multi" else 1))
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
            log(f"  step {step}:{tag} train err: {e!r}")
            cleanup(model, tokenizer, optimizer)
            return {"status": "failed", "error": repr(e)}
        losses.append(float(loss_val.item()))
        if step % max(1, N_STEPS // 5) == 0 or step == 1:
            log(f"    step {step}:{tag}  loss={losses[-1]:.3f}")
        mx.clear_cache()
    elapsed = time.time() - t0

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(
        str(adapter_out_dir / "adapters.safetensors"), adapter_weights
    )

    cleanup(model, tokenizer, optimizer)

    return {
        "status": "ok",
        "tag": tag,
        "n_train_seqs": len(train_ids),
        "time_s": round(elapsed, 1),
        "steps": N_STEPS,
        "final_loss": round(losses[-1], 4) if losses else None,
        "mean_loss": round(float(np.mean(losses)), 4) if losses else None,
        "adapter_path": str(adapter_out_dir),
    }


# ── Phase 3: eval — base, multi, single on HELD-OUT cats, NO method prompt ──

def phase_eval_heldout(adapter_path, tag):
    """Generate WITHOUT the method system prompt (prompt erasure check).
    Returns per-category accuracy and per-response subgoal-marker counts."""
    from mlx_lm import load, generate

    log(f"\n[Eval:{tag}] Held-out MMLU-Pro (no method prompt)")
    mmlu_path = EXPERIMENT_DIR.parent / "exp_bench_mmlu_pro" / "data" / "test.parquet"
    if not mmlu_path.exists():
        return {"error": "mmlu-missing"}

    df = pd.read_parquet(mmlu_path)
    load_kwargs = {"adapter_path": str(adapter_path)} if adapter_path else {}
    try:
        model, tokenizer = load(MODEL_ID, **load_kwargs)
    except Exception as e:
        return {"error": f"load-failed: {e!r}"}
    log_memory(f"eval-{tag}-loaded")

    # IMPORTANT: fixed eval seed so all three arms (base, multi, single) see
    # the SAME held-out questions for paired comparison.
    rng = np.random.RandomState(SEED + 1000)

    per_cat = {}
    all_responses = []   # for signature scoring
    for cat in HELDOUT_CATS:
        cat_df = df[df["category"].str.lower() == cat.lower()]
        if len(cat_df) == 0:
            continue
        n_sample = min(EVAL_PER_CAT, len(cat_df))
        # Re-seed per cat so all arms pick the same rows independent of cat order
        cat_rng = np.random.RandomState(SEED + 1000 + hash(cat) % 997)
        idxs = cat_rng.choice(len(cat_df), size=n_sample, replace=False)
        cat_correct, cat_sig = 0, 0
        for i in idxs:
            row = cat_df.iloc[int(i)]
            user_content, correct_letter = _format_question(row)
            # NO method system prompt at eval — prompt erasure test.
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            try:
                resp = generate(model, tokenizer, prompt=prompt, max_tokens=2048)
                predicted, _tchars = parse_mcq_answer(resp)
            except Exception as e:
                log(f"    eval err: {e!r}")
                resp, predicted = "", None
            sig_count = count_subgoal_markers(resp)
            sig_hit = int(sig_count >= 2)
            all_responses.append({
                "category": cat,
                "correct": correct_letter,
                "predicted": predicted,
                "sig_count": sig_count,
                "sig_hit": sig_hit,
                "response_prefix": resp[:300] if resp else "",
            })
            if predicted == correct_letter:
                cat_correct += 1
            cat_sig += sig_hit
            mx.eval()

        acc = cat_correct / n_sample * 100 if n_sample else 0
        sig_rate = cat_sig / n_sample if n_sample else 0
        per_cat[cat] = {
            "correct": cat_correct,
            "total": n_sample,
            "acc": round(acc, 1),
            "signature_rate": round(sig_rate, 3),
        }
        log(f"  {cat}: acc={acc:.1f}% ({cat_correct}/{n_sample})  sig={sig_rate:.0%}")

    cleanup(model, tokenizer)

    total_n = sum(v["total"] for v in per_cat.values())
    total_correct = sum(v["correct"] for v in per_cat.values())
    overall_sig = sum(1 for r in all_responses if r["sig_hit"]) / max(len(all_responses), 1)
    return {
        "tag": tag,
        "per_category": per_cat,
        "overall_accuracy": round(total_correct / max(total_n, 1) * 100, 1),
        "overall_signature_rate": round(overall_sig, 3),
        "n_eval": total_n,
        "responses": all_responses,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 60)
    log(f"exp_method_vs_domain_adapter  SMOKE={IS_SMOKE}  "
        f"N_STEPS={N_STEPS}  rank={LORA_RANK}  scale={LORA_SCALE}")
    log(f"train_cats={TRAIN_CATS}")
    log(f"heldout_cats={HELDOUT_CATS}")
    log("=" * 60)
    log_memory("start")

    results = {
        "experiment": "exp_method_vs_domain_adapter",
        "model": MODEL_ID,
        "is_smoke": IS_SMOKE,
        "lora_rank": LORA_RANK,
        "lora_scale": LORA_SCALE,
        "lora_keys": LORA_KEYS,
        "lora_num_layers": LORA_NUM_LAYERS,
        "n_steps": N_STEPS,
        "n_per_cat_train": N_PER_CAT_TRAIN,
        "n_single_domain": N_SINGLE_DOMAIN,
        "eval_per_cat": EVAL_PER_CAT,
        "train_cats": TRAIN_CATS,
        "heldout_cats": HELDOUT_CATS,
        "single_domain": SINGLE_DOMAIN,
        "seed": SEED,
    }

    # Phase 1 — teacher data (and teacher gate)
    data_info = phase_build_training_data()
    results["training_data"] = data_info
    log(f"  data: {data_info}")
    if data_info.get("error"):
        results["verdict"] = "ABORTED"
        results["all_pass"] = False
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return 1
    if not IS_SMOKE and not data_info.get("teacher_signature_pass", False):
        results["verdict"] = "ABORTED"
        results["abort_reason"] = (
            f"teacher signature rate {data_info.get('teacher_signature_rate')} < 0.70"
        )
        results["all_pass"] = False
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return 1

    # Phase 2a — multi-domain adapter
    train_multi = phase_train_adapter(
        DATA_DIR / "train_multi.jsonl", ADAPTER_MULTI_DIR, "multi"
    )
    results["train_multi"] = train_multi
    if train_multi.get("status") != "ok":
        results["verdict"] = "ABORTED"
        results["all_pass"] = False
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return 1

    # Phase 2b — single-domain control
    train_single = phase_train_adapter(
        DATA_DIR / "train_single.jsonl", ADAPTER_SINGLE_DIR, "single"
    )
    results["train_single"] = train_single
    if train_single.get("status") != "ok":
        results["verdict"] = "ABORTED"
        results["all_pass"] = False
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return 1

    # Phase 3 — paired eval on held-out cats
    eval_base = phase_eval_heldout(None, "base")
    results["eval_base"] = {k: v for k, v in eval_base.items() if k != "responses"}
    eval_multi = phase_eval_heldout(ADAPTER_MULTI_DIR, "multi")
    results["eval_multi"] = {k: v for k, v in eval_multi.items() if k != "responses"}
    eval_single = phase_eval_heldout(ADAPTER_SINGLE_DIR, "single")
    results["eval_single"] = {k: v for k, v in eval_single.items() if k != "responses"}

    # Persist responses separately for offline inspection
    resp_path = DATA_DIR / "eval_responses.jsonl"
    with open(resp_path, "w") as f:
        for r in eval_base.get("responses", []):
            r["arm"] = "base"; f.write(json.dumps(r) + "\n")
        for r in eval_multi.get("responses", []):
            r["arm"] = "multi"; f.write(json.dumps(r) + "\n")
        for r in eval_single.get("responses", []):
            r["arm"] = "single"; f.write(json.dumps(r) + "\n")
    results["eval_responses_file"] = str(resp_path)

    # ── KC evaluation ───────────────────────────────────────────────────────
    def cats_beaten(arm_eval, base_eval):
        beaten = []
        for cat in HELDOUT_CATS:
            a = arm_eval.get("per_category", {}).get(cat, {}).get("acc", 0)
            b = base_eval.get("per_category", {}).get(cat, {}).get("acc", 0)
            if a > b:
                beaten.append(cat)
        return beaten

    multi_beaten = cats_beaten(eval_multi, eval_base)
    single_beaten = cats_beaten(eval_single, eval_base)
    multi_sig = eval_multi.get("overall_signature_rate", 0.0)
    base_sig = eval_base.get("overall_signature_rate", 0.0)
    sig_delta_pp = (multi_sig - base_sig) * 100

    k1718_pass = len(multi_beaten) >= 3
    k1719_pass = len(single_beaten) <= 1
    k1720_pass = (multi_sig >= 0.70) and (sig_delta_pp >= 20.0)

    results["k1718_multi_beats_base_ge_3"] = {
        "pass": bool(k1718_pass),
        "multi_beaten_cats": multi_beaten,
        "count": len(multi_beaten),
    }
    results["k1719_single_fails_le_1"] = {
        "pass": bool(k1719_pass),
        "single_beaten_cats": single_beaten,
        "count": len(single_beaten),
    }
    results["k1720_signature_rate"] = {
        "pass": bool(k1720_pass),
        "multi_signature_rate": multi_sig,
        "base_signature_rate": base_sig,
        "delta_pp": round(sig_delta_pp, 2),
        "threshold_rate": 0.70,
        "threshold_delta_pp": 20.0,
    }

    all_pass = k1718_pass and k1719_pass and k1720_pass
    results["all_pass"] = bool(all_pass)
    if IS_SMOKE:
        results["verdict"] = "PROVISIONAL"
        results["verdict_reason"] = "smoke mode — rerun at full N for supported/killed"
    else:
        results["verdict"] = "SUPPORTED" if all_pass else "KILLED"

    results["total_time_s"] = round(time.time() - t0, 1)
    log_memory("end")

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log("\n" + "=" * 60)
    log(f"VERDICT: {results['verdict']}  all_pass={all_pass}")
    log(f"  K1718 multi≥3/5: {k1718_pass}  beaten={multi_beaten}")
    log(f"  K1719 single≤1/5: {k1719_pass}  beaten={single_beaten}")
    log(f"  K1720 sig≥0.70 AND Δ≥+20pp: {k1720_pass}  "
        f"sig_multi={multi_sig:.2%}  Δ={sig_delta_pp:+.1f}pp")
    log(f"  Total time: {results['total_time_s']}s")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
