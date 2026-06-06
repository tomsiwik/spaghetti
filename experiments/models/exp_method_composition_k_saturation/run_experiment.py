#!/usr/bin/env python3
"""exp_method_composition_k_saturation — Skill-Mix k-saturation on Gemma 4 LoRA.

MATH: experiments/models/exp_method_composition_k_saturation/MATH.md

5 method adapters, each with a distinct textually-disjoint signature.
Compose k-at-a-time (rank-stacked LoRA sum), measure signature preservation
and MCQ accuracy on held-out MMLU-Pro questions.

Pipeline:
  Phase 1: teacher-gate + generate training data per method (n_train each)
           — gate: teacher signature rate for method i >= 0.70, else exclude
  Phase 2: train 5 rank-8 LoRA method adapters on v_proj+o_proj (top 16
           layers, scale 4.0)
  Phase 3: for k in {1..5}, pick the first-k subset of methods, build the
           composed adapter by concat-stacking A and B along the rank axis,
           eval on held-out MMLU-Pro n=eval_per_cond questions
  Phase 4: KC evaluation, write results.json

Smoke reports PROVISIONAL per guardrail 1009. Full rerun plan in PAPER.md.
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
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "1") == "1"
SEED = 42

LORA_RANK = 8
LORA_SCALE = 4.0
LORA_DROPOUT = 0.0
LORA_KEYS = ["self_attn.v_proj", "self_attn.o_proj"]
LORA_NUM_LAYERS = 16

N_STEPS = 30 if IS_SMOKE else 200
BATCH_SIZE = 1
LR = 1e-4
MAX_SEQ_LEN = 2048

N_PER_METHOD_TRAIN = 6 if IS_SMOKE else 30
EVAL_PER_COND = 10 if IS_SMOKE else 30
TRAIN_CATS = ["math", "computer science", "health", "law", "economics"]
HELDOUT_CATS = ["physics", "biology", "philosophy", "psychology", "history"]

# Five method definitions — each textually disjoint.
METHODS = [
    {
        "name": "restate",
        "system": (
            "You answer multiple-choice questions using this format:\n"
            "Begin with the line: Problem restated: <one-sentence paraphrase>\n"
            "Then give your reasoning. End with: Answer: X"
        ),
        "signature_regex": r"(?m)^\s*Problem restated:\s",
    },
    {
        "name": "numbered",
        "system": (
            "You answer multiple-choice questions using explicit numbered steps.\n"
            "Use the format:\n"
            "Step 1: <work>\n"
            "Step 2: <work>\n"
            "Step 3: <work>\n"
            "Answer: X"
        ),
        "signature_regex": r"\bStep\s+2\b",
    },
    {
        "name": "verify",
        "system": (
            "You answer multiple-choice questions and self-verify.\n"
            "After your reasoning, add a line:\n"
            "Check: <one-sentence sanity check that the answer is consistent>\n"
            "Then finish with: Answer: X"
        ),
        "signature_regex": r"(?mi)^\s*(Verification|Check):\s",
    },
    {
        "name": "principle",
        "system": (
            "You answer multiple-choice questions by first naming the governing rule.\n"
            "Begin with a line:\n"
            "Principle: <the rule or law being applied>\n"
            "Then give your reasoning. End with: Answer: X"
        ),
        "signature_regex": r"(?mi)^\s*(Principle|Rule):\s",
    },
    {
        "name": "tldr",
        "system": (
            "You answer multiple-choice questions concisely.\n"
            "Give your reasoning, then finish with two lines:\n"
            "Answer: X\n"
            "TL;DR: <one-sentence summary of why>"
        ),
        "signature_regex": r"(?mi)\bTL;?DR:\s",
    },
]

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


# Fresh regex helpers (not copied from parent — each verified against unit test)

def strip_thinking(response):
    """Remove Gemma-4 thinking channel. Fortified for missing close tags."""
    if not response:
        return response or ""
    out = re.sub(r"<\|channel>thought.*?<channel\|>", "", response, flags=re.DOTALL)
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL)
    # Fallback: open tag but no close tag — strip to next blank line or "Answer:"
    m = re.search(r"<\|channel>thought", out)
    if m:
        tail = out[m.end():]
        stop = re.search(r"\n\s*\n|Answer:", tail)
        cut = stop.end() if stop else len(tail)
        out = out[:m.start()] + tail[cut:]
    return out.strip()


def parse_mcq_answer(response):
    text = strip_thinking(response)
    for pattern in [
        r"\banswer is\s*([A-J])",
        r"(?mi)^\s*Answer:\s*([A-J])",
        r"\banswer:\s*([A-J])",
        r"\b([A-J])\b\s*$",
    ]:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    m = re.search(r"\b([A-J])\b", text)
    return m.group(1).upper() if m else None


def signature_hit(response, method):
    """Return 1 if method's target signature present in post-strip response."""
    text = strip_thinking(response)
    return int(bool(re.search(method["signature_regex"], text)))


def _format_question(row):
    options = row.get("options", [])
    n_opts = len(options)
    option_text = "\n".join(
        f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)
    )
    user_content = (
        f"Answer the following multiple choice question. Select the single "
        f"best answer letter (A through {OPTION_LETTERS[n_opts - 1]}).\n\n"
        f"Question: {row['question']}\n\nOptions:\n{option_text}\n\nAnswer:"
    )
    return user_content, OPTION_LETTERS[int(row["answer_index"])]


# ── Phase 1: teacher-gate + training data per method ────────────────────────

def phase_build_teacher_data():
    from mlx_lm import load, generate

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_cached = all(
        (DATA_DIR / f"train_{m['name']}.jsonl").exists() for m in METHODS
    )
    stats_path = DATA_DIR / "teacher_stats.json"
    if all_cached and stats_path.exists():
        log("[Phase 1] Teacher data cached.")
        return json.loads(stats_path.read_text())

    mmlu_path = EXPERIMENT_DIR.parent / "exp_bench_mmlu_pro" / "data" / "test.parquet"
    if not mmlu_path.exists():
        return {"error": f"mmlu-pro-missing: {mmlu_path}"}

    df = pd.read_parquet(mmlu_path)
    rng = np.random.RandomState(SEED)
    log("\n[Phase 1] Generating teacher data per method")
    model, tokenizer = load(MODEL_ID)
    log_memory("teacher-loaded")

    per_method_stats = {}
    for method in METHODS:
        out_path = DATA_DIR / f"train_{method['name']}.jsonl"
        if out_path.exists():
            # count existing + compute rate by re-reading (cheap)
            lines = [json.loads(l) for l in open(out_path)]
            n = len(lines)
            sig_hits = sum(l.get("sig_hit", 0) for l in lines)
            per_method_stats[method["name"]] = {
                "n": n, "sig_hits": sig_hits,
                "signature_rate": round(sig_hits / max(n, 1), 3),
                "gate_pass": (sig_hits / max(n, 1)) >= 0.70,
                "cached": True,
            }
            continue
        # sample TRAIN_CATS × N_PER_METHOD_TRAIN rows (evenly over cats)
        per_cat = max(1, N_PER_METHOD_TRAIN // len(TRAIN_CATS))
        n_wrote, sig_hits = 0, 0
        with open(out_path, "w") as w:
            for cat in TRAIN_CATS:
                cat_df = df[df["category"].str.lower() == cat.lower()]
                if len(cat_df) == 0:
                    continue
                idxs = rng.choice(len(cat_df), size=min(per_cat, len(cat_df)),
                                  replace=False)
                for i in idxs:
                    row = cat_df.iloc[int(i)]
                    user_content, correct = _format_question(row)
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "system", "content": method["system"]},
                         {"role": "user", "content": user_content}],
                        tokenize=False, add_generation_prompt=True,
                        enable_thinking=True,
                    )
                    try:
                        resp = generate(model, tokenizer, prompt=prompt,
                                        max_tokens=768)
                        mx.eval()
                    except Exception as e:
                        log(f"  teacher err {method['name']}/{cat}: {e!r}")
                        continue
                    student = strip_thinking(resp)
                    if not student:
                        continue
                    sig = signature_hit(resp, method)
                    sig_hits += sig
                    ex = {
                        "category": cat,
                        "correct_letter": correct,
                        "sig_hit": sig,
                        # Prompt-erased student view: keep the teacher's
                        # method-style response but drop the method system
                        # prompt (Orca-2 pattern)
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": student},
                        ],
                    }
                    w.write(json.dumps(ex) + "\n")
                    n_wrote += 1
        rate = sig_hits / max(n_wrote, 1)
        per_method_stats[method["name"]] = {
            "n": n_wrote, "sig_hits": sig_hits,
            "signature_rate": round(rate, 3),
            "gate_pass": rate >= 0.70,
            "cached": False,
        }
        log(f"  {method['name']}: wrote {n_wrote}, sig_rate={rate:.2%}")

    stats_path.write_text(json.dumps(per_method_stats, indent=2))
    cleanup(model, tokenizer)
    return per_method_stats


# ── Phase 2: train one rank-8 LoRA per method ───────────────────────────────

def _train_one_adapter(method_name, train_jsonl_path, adapter_out_dir):
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx.utils import tree_flatten

    log(f"\n[Phase 2:{method_name}] Training rank={LORA_RANK} adapter")
    adapter_out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load(MODEL_ID)
    model.freeze()
    lora_config = {
        "rank": LORA_RANK, "scale": LORA_SCALE,
        "dropout": LORA_DROPOUT, "keys": LORA_KEYS,
    }
    linear_to_lora_layers(model, LORA_NUM_LAYERS, lora_config, use_dora=False)
    log_memory(f"lora-attached-{method_name}")

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
        cleanup(model, tokenizer)
        return {"status": "failed", "error": "empty-train"}

    def loss_fn(m, input_ids):
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        logits = m(inputs)
        ntoks = mx.array(targets.size, dtype=mx.float32)
        ce = nn.losses.cross_entropy(logits, targets, reduction="mean")
        return ce, ntoks

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=LR)
    rng = np.random.RandomState(SEED + hash(method_name) % 997)
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
            log(f"  step {step}:{method_name} train err: {e!r}")
            cleanup(model, tokenizer, optimizer)
            return {"status": "failed", "error": repr(e)}
        losses.append(float(loss_val.item()))
        if step % max(1, N_STEPS // 4) == 0 or step == 1:
            log(f"    step {step}:{method_name}  loss={losses[-1]:.3f}")
        mx.clear_cache()
    elapsed = time.time() - t0

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(adapter_out_dir / "adapters.safetensors"),
                        adapter_weights)

    cleanup(model, tokenizer, optimizer)
    return {
        "status": "ok", "method": method_name,
        "n_train_seqs": len(train_ids),
        "time_s": round(elapsed, 1), "steps": N_STEPS,
        "final_loss": round(losses[-1], 4) if losses else None,
        "mean_loss": round(float(np.mean(losses)), 4) if losses else None,
        "adapter_path": str(adapter_out_dir),
    }


def phase_train_all_adapters(teacher_stats):
    train_results = {}
    for method in METHODS:
        name = method["name"]
        ts = teacher_stats.get(name, {})
        if not ts.get("gate_pass", False):
            log(f"  SKIP training {name} (gate_pass={ts.get('gate_pass')}, "
                f"rate={ts.get('signature_rate')})")
            train_results[name] = {
                "status": "skipped-gate",
                "teacher_rate": ts.get("signature_rate"),
            }
            continue
        adapter_dir = ADAPTERS_DIR / f"method_{name}"
        # skip train if adapter already exists
        if (adapter_dir / "adapters.safetensors").exists():
            train_results[name] = {"status": "ok-cached",
                                   "adapter_path": str(adapter_dir)}
            log(f"  {name}: cached")
            continue
        train_jsonl = DATA_DIR / f"train_{name}.jsonl"
        r = _train_one_adapter(name, train_jsonl, adapter_dir)
        train_results[name] = r
    return train_results


# ── Phase 3: composition (rank-stack) + eval ────────────────────────────────

def _compose_adapter_weights(subset_names):
    """Load each method's adapter and stack A/B along rank axis.

    Returns a dict {key: mx.array} ready to save as a composed adapter file.
    Composed rank = LORA_RANK * len(subset_names).
    """
    loaded = []
    for name in subset_names:
        p = ADAPTERS_DIR / f"method_{name}" / "adapters.safetensors"
        w = mx.load(str(p))
        loaded.append(w)

    # Gather unique module roots across all adapters (they should match).
    roots = set()
    for w in loaded:
        for k in w.keys():
            if k.endswith(".lora_a"):
                roots.add(k[:-len(".lora_a")])
    composed = {}
    for root in sorted(roots):
        a_key = f"{root}.lora_a"
        b_key = f"{root}.lora_b"
        a_mats, b_mats = [], []
        for w in loaded:
            if a_key not in w or b_key not in w:
                continue
            a_mats.append(w[a_key])      # (in, r)
            b_mats.append(w[b_key])      # (r, out)
        if not a_mats:
            continue
        # Stack along rank axis — matches Δ = s * A @ B = s * Σ A_i @ B_i
        a_cat = mx.concatenate(a_mats, axis=1)   # (in, k*r)
        b_cat = mx.concatenate(b_mats, axis=0)   # (k*r, out)
        composed[a_key] = a_cat
        composed[b_key] = b_cat
    return composed


def _write_composed_adapter(composed_weights, out_dir, rank):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora",
        "num_layers": LORA_NUM_LAYERS,
        "lora_parameters": {
            "rank": rank, "scale": LORA_SCALE,
            "dropout": LORA_DROPOUT, "keys": LORA_KEYS,
        },
    }, indent=2))
    mx.save_safetensors(str(out_dir / "adapters.safetensors"),
                        dict(composed_weights))


def _eval_condition(adapter_path, cond_label, methods_in_cond, df, rng_master):
    """Eval base+adapter on held-out MCQ. Returns per-method sig_rate + acc."""
    from mlx_lm import load, generate

    load_kwargs = {"adapter_path": str(adapter_path)} if adapter_path else {}
    model, tokenizer = load(MODEL_ID, **load_kwargs)
    log_memory(f"eval-{cond_label}")

    # Re-seed per condition on a FIXED pool — so the same held-out questions
    # are scored across every k and solo condition.
    rng = np.random.RandomState(SEED + 5000)

    correct, n = 0, 0
    sig_counts = {m["name"]: 0 for m in METHODS}
    responses = []
    for cat in HELDOUT_CATS:
        cat_df = df[df["category"].str.lower() == cat.lower()]
        if len(cat_df) == 0:
            continue
        # Stable per-cat sub-rng so same qs across conds
        cat_rng = np.random.RandomState(SEED + 5000 + hash(cat) % 997)
        n_sample = min(EVAL_PER_COND // len(HELDOUT_CATS) + 1, len(cat_df))
        idxs = cat_rng.choice(len(cat_df), size=n_sample, replace=False)
        for i in idxs:
            if n >= EVAL_PER_COND:
                break
            row = cat_df.iloc[int(i)]
            user_content, correct_letter = _format_question(row)
            # NO method system prompt at eval — prompt-erasure test
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            try:
                resp = generate(model, tokenizer, prompt=prompt,
                                max_tokens=1024)
                predicted = parse_mcq_answer(resp)
            except Exception as e:
                log(f"    eval err: {e!r}")
                resp, predicted = "", None
            n += 1
            if predicted == correct_letter:
                correct += 1
            for m in METHODS:
                sig_counts[m["name"]] += signature_hit(resp, m)
            responses.append({
                "cond": cond_label, "category": cat,
                "correct": correct_letter, "predicted": predicted,
                "sig_hits": {
                    m["name"]: int(signature_hit(resp, m))
                    for m in METHODS
                },
                "response_prefix": resp[:500] if resp else "",
            })
            mx.eval()
        if n >= EVAL_PER_COND:
            break

    cleanup(model, tokenizer)
    per_method_rate = {
        name: round(sig_counts[name] / max(n, 1), 3)
        for name in sig_counts
    }
    # Mean rate over the subset of methods present in the condition
    if methods_in_cond:
        focal_rate = float(np.mean([per_method_rate[m] for m in methods_in_cond]))
    else:
        focal_rate = 0.0
    return {
        "cond": cond_label,
        "methods_in_cond": methods_in_cond,
        "n": n,
        "accuracy": round(correct / max(n, 1) * 100, 1),
        "per_method_signature_rate": per_method_rate,
        "focal_signature_rate": round(focal_rate, 3),
        "responses": responses,
    }


def phase_compose_and_eval(teacher_stats, train_results):
    from mlx_lm import load
    mmlu_path = EXPERIMENT_DIR.parent / "exp_bench_mmlu_pro" / "data" / "test.parquet"
    if not mmlu_path.exists():
        return {"error": f"mmlu-pro-missing: {mmlu_path}"}
    df = pd.read_parquet(mmlu_path)
    rng = np.random.RandomState(SEED)

    # Only include methods that trained successfully and gate-passed
    available = [m["name"] for m in METHODS
                 if train_results.get(m["name"], {}).get("status", "") in
                 ("ok", "ok-cached")]
    log(f"\n[Phase 3] Available methods for sweep: {available}")
    if len(available) == 0:
        return {"error": "no-trained-adapters"}

    all_conds = {}
    # Solo eval per available method (k=1 individually — to compute M_solo)
    solo_rates = {}
    for name in available:
        adapter_dir = ADAPTERS_DIR / f"method_{name}"
        r = _eval_condition(adapter_dir, f"solo_{name}", [name], df, rng)
        # strip responses in aggregated view
        summary = {k: v for k, v in r.items() if k != "responses"}
        all_conds[f"solo_{name}"] = summary
        solo_rates[name] = summary["focal_signature_rate"]

    M_solo = float(np.mean(list(solo_rates.values()))) if solo_rates else 0.0
    log(f"  M_solo (mean solo signature rate) = {M_solo:.3f}")

    # Composed eval: for each k in {1..5} capped at len(available), use the
    # first-k subset (deterministic — same subset inclusion used by PAPER).
    # We also run a "k=N all" condition.
    k_sweep = list(range(1, len(available) + 1))
    sat_curve = {}
    for k in k_sweep:
        subset = available[:k]
        compose_dir = EXPERIMENT_DIR / "adapters" / f"composed_k{k}"
        composed = _compose_adapter_weights(subset)
        _write_composed_adapter(composed, compose_dir, rank=LORA_RANK * k)
        r = _eval_condition(compose_dir, f"k{k}", subset, df, rng)
        summary = {k2: v for k2, v in r.items() if k2 != "responses"}
        all_conds[f"k{k}"] = summary
        sat_curve[k] = summary["focal_signature_rate"]
        log(f"  k={k} subset={subset} focal_sig={summary['focal_signature_rate']:.3f}  "
            f"acc={summary['accuracy']}%")

    # Persist per-response logs
    resp_path = DATA_DIR / "eval_responses.jsonl"
    with open(resp_path, "w") as f:
        for cond, summary in all_conds.items():
            # we dropped responses; re-read is cheap → skipped. We left the
            # aggregate summary in all_conds. Keep this file as stub for future.
            pass
    return {
        "available": available,
        "M_solo": round(M_solo, 3),
        "solo_rates": solo_rates,
        "saturation_curve": sat_curve,
        "conditions": all_conds,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 60)
    log(f"exp_method_composition_k_saturation  SMOKE={IS_SMOKE}")
    log(f"  N_STEPS={N_STEPS}  rank={LORA_RANK}  scale={LORA_SCALE}  "
        f"n_per_method={N_PER_METHOD_TRAIN}  eval={EVAL_PER_COND}")
    log("=" * 60)
    log_memory("start")

    results = {
        "experiment": "exp_method_composition_k_saturation",
        "model": MODEL_ID,
        "is_smoke": IS_SMOKE,
        "seed": SEED,
        "lora_rank": LORA_RANK,
        "lora_scale": LORA_SCALE,
        "lora_keys": LORA_KEYS,
        "lora_num_layers": LORA_NUM_LAYERS,
        "n_steps": N_STEPS,
        "n_per_method_train": N_PER_METHOD_TRAIN,
        "eval_per_cond": EVAL_PER_COND,
        "methods": [m["name"] for m in METHODS],
        "train_cats": TRAIN_CATS,
        "heldout_cats": HELDOUT_CATS,
    }

    teacher_stats = phase_build_teacher_data()
    results["teacher_stats"] = teacher_stats
    if isinstance(teacher_stats, dict) and teacher_stats.get("error"):
        results["verdict"] = "ABORTED"
        results["all_pass"] = False
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return 1

    train_results = phase_train_all_adapters(teacher_stats)
    results["train_results"] = train_results

    sweep = phase_compose_and_eval(teacher_stats, train_results)
    results["sweep"] = sweep
    if sweep.get("error"):
        results["verdict"] = "ABORTED"
        results["all_pass"] = False
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return 1

    # ── KC eval ─────────────────────────────────────────────────────────────
    M_solo = sweep.get("M_solo", 0.0)
    curve = sweep.get("saturation_curve", {})
    # map k→rate (k is int)
    curve_int = {int(k): float(v) for k, v in curve.items()}

    # K1730: k=2 preservation ≥ 0.95 * M_solo
    k2_rate = curve_int.get(2)
    k1730_pass = (k2_rate is not None) and (k2_rate >= 0.95 * M_solo)

    # K1731: k=5 preservation ≤ 0.80 * M_solo (saturation)
    k5_rate = curve_int.get(5)
    k1731_pass = (k5_rate is not None) and (k5_rate <= 0.80 * M_solo)

    # K1732: monotonic with no +0.05 spike
    ks = sorted(curve_int.keys())
    non_monotonic_spikes = []
    for i in range(1, len(ks)):
        prev, curr = curve_int[ks[i-1]], curve_int[ks[i]]
        if curr > prev + 0.05:
            non_monotonic_spikes.append({"k": ks[i], "prev": prev, "curr": curr})
    k1732_pass = len(non_monotonic_spikes) == 0

    results["k1730_k2_survival_ge_0p95_Msolo"] = {
        "pass": bool(k1730_pass),
        "M_solo": M_solo, "k2_rate": k2_rate,
        "threshold": round(0.95 * M_solo, 3),
    }
    results["k1731_k5_saturation_le_0p80_Msolo"] = {
        "pass": bool(k1731_pass),
        "M_solo": M_solo, "k5_rate": k5_rate,
        "threshold": round(0.80 * M_solo, 3),
    }
    results["k1732_monotonic_no_spikes"] = {
        "pass": bool(k1732_pass),
        "spikes": non_monotonic_spikes,
        "curve": curve_int,
    }

    all_pass = k1730_pass and k1731_pass and k1732_pass
    results["all_pass"] = bool(all_pass)
    if IS_SMOKE:
        results["verdict"] = "PROVISIONAL"
        results["verdict_reason"] = "smoke mode — rerun at full N"
    else:
        results["verdict"] = "SUPPORTED" if all_pass else "KILLED"

    results["total_time_s"] = round(time.time() - t0, 1)
    log_memory("end")
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log("\n" + "=" * 60)
    log(f"VERDICT: {results['verdict']}  all_pass={all_pass}")
    log(f"  M_solo={M_solo:.3f}  curve={curve_int}")
    log(f"  K1730 k=2 ≥ {0.95*M_solo:.3f}: {k1730_pass}")
    log(f"  K1731 k=5 ≤ {0.80*M_solo:.3f}: {k1731_pass}")
    log(f"  K1732 monotonic: {k1732_pass}  spikes={non_monotonic_spikes}")
    log(f"  total_time={results['total_time_s']}s")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
