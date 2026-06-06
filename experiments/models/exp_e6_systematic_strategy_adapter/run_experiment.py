#!/usr/bin/env python3
"""
exp_e6_systematic_strategy_adapter
===================================

Hedgehog per-layer cos-sim distillation of "systematic decomposition" strategy.
Teacher: Gemma 4 E4B + decomposition system prompt.
Student: same model + LoRA v_proj+o_proj (F#627).

Pre-registered KCs (canonical DB text — do not edit):
  K_struct: mean per-layer cos > 0.85 on n>=50 heldout prompts
  K2028: >3pp improvement on ANY domain (GSM8K, ARC-C, MMLU-STEM)
  K2029: improvement on 2+ domains (cross-domain transfer)

Skills invoked: /mlx-dev (mx.eval, mx.clear_cache, nn.value_and_grad) + /fast-mlx.
"""
from __future__ import annotations

import gc
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

mx.set_memory_limit(mx.metal.device_info()["memory_size"] - 8 * 1024**3) \
    if hasattr(mx, "metal") and hasattr(mx.metal, "device_info") \
    else mx.set_memory_limit(40 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_TRAIN = 24 if IS_SMOKE else 200
N_HELDOUT = 8 if IS_SMOKE else 50
N_STEPS = 30 if IS_SMOKE else 800
N_EVAL = 20 if IS_SMOKE else 100
SEED = 42

ADAPTER_RANK = 8
ADAPTER_TARGETS = ("v_proj", "o_proj")  # F#627
LORA_SCALE = 6.0                        # ≤ 8 per F#328/F#330
BATCH_SIZE = 1
SEQLEN = 256 if IS_SMOKE else 512
LR = 1e-4
WEIGHT_DECAY = 0.01
GEN_MAX_TOKENS = 8
ENABLE_THINKING = False  # F#790 + F#786 mitigation

DECOMP_SYSTEM_PROMPT = (
    "For every problem: (1) identify the sub-problems, "
    "(2) solve each sub-problem independently showing your work, "
    "(3) verify each sub-result, (4) combine into a final answer."
)
NEUTRAL_SYSTEM_PROMPT = ""

SMOKE_NEUTRAL_PROMPTS = [
    "Explain how a binary search works.",
    "List three causes of inflation.",
    "What is the chemical formula of glucose?",
    "Give one example of a matrix multiplication.",
    "Describe the difference between TCP and UDP.",
    "Summarize Newton's second law.",
    "Name three programming languages used for systems work.",
    "What does the acronym DNS stand for?",
    "Define entropy in thermodynamics.",
    "Explain what a lambda function is in Python.",
    "What is the capital of Belgium?",
    "Describe one method to sort an array.",
    "Define a context manager.",
    "What is REST?",
    "How does a hash table work?",
    "Define encapsulation in OOP.",
    "List two benefits of sleep.",
    "Define a kernel in operating systems.",
    "What is gradient descent?",
    "Describe the structure of an HTTP request.",
    "What is a primary key in a relational database?",
    "Explain the difference between a stack and a queue.",
    "Name three units of digital storage.",
    "What is a syntax tree?",
    "Define recursion.",
    "What is the boiling point of water in Celsius?",
    "Describe one application of Bayes' theorem.",
    "Explain what an ORM is.",
    "What is a thread in computing?",
    "Define mass-energy equivalence.",
    "What is dynamic programming?",
    "Define overfitting.",
    "What does CPU stand for?",
    "Explain what a regular expression is.",
    "Define a microservice.",
    "What is the difference between HTTP and HTTPS?",
    "Define cache coherence.",
    "What is a binary tree?",
    "Explain the role of DNA polymerase.",
    "Define the term 'pivot' in a sorting algorithm.",
]


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()


# ──────────────────────────────────────────────
# Chat-template helper
# ──────────────────────────────────────────────

def apply_chat(tokenizer, messages, *, tokenize=False):
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=tokenize,
            enable_thinking=ENABLE_THINKING,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=tokenize,
        )


# ──────────────────────────────────────────────
# Phase 0 — neutral prompt curation
# ──────────────────────────────────────────────

def prepare_neutral_prompts(n_train: int, n_heldout: int) -> dict:
    if IS_SMOKE or n_train + n_heldout <= len(SMOKE_NEUTRAL_PROMPTS):
        all_prompts = list(SMOKE_NEUTRAL_PROMPTS)
        train = all_prompts[:n_train]
        heldout = all_prompts[n_train:n_train + n_heldout]
        return {"train": train, "heldout": heldout, "source": "embedded_smoke"}

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("datasets package required for full Phase 0")

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    collected = []
    for ex in ds:
        if len(collected) >= (n_train + n_heldout) * 3:
            break
        msgs = ex.get("messages") or []
        if not msgs:
            continue
        first_user = next((m for m in msgs if m.get("role") == "user"), None)
        if not first_user:
            continue
        text = first_user.get("content", "").strip()
        if len(text) < 20 or len(text) > 600:
            continue
        collected.append(text)
    if len(collected) < n_train + n_heldout:
        raise RuntimeError(f"UltraChat yielded only {len(collected)}, need {n_train + n_heldout}")
    return {
        "train": collected[:n_train],
        "heldout": collected[n_train:n_train + n_heldout],
        "source": "ultrachat_filtered",
    }


# ──────────────────────────────────────────────
# Hedgehog attention capture + distillation
# ──────────────────────────────────────────────

class AttnOutCapture(nn.Module):
    def __init__(self, inner, store, layer_idx):
        super().__init__()
        self.inner = inner
        self._store = store
        self._layer_idx = layer_idx

    def __call__(self, x):
        out = self.inner(x)
        self._store[self._layer_idx] = out
        return out


def install_hooks(model, store):
    layers = model.language_model.layers
    for i, layer in enumerate(layers):
        layer.self_attn.o_proj = AttnOutCapture(layer.self_attn.o_proj, store, i)
    return len(layers)


def encode_prompt(tokenizer, system_prompt: str, user: str) -> mx.array:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user})
    ids = apply_chat(tokenizer, msgs, tokenize=True)
    return mx.array(ids, dtype=mx.int32)


def per_layer_cossim_loss(teacher_attns, student_attns, t_offset, s_offset, x_len):
    losses = []
    for layer_idx in sorted(teacher_attns.keys()):
        t = mx.stop_gradient(teacher_attns[layer_idx])
        s = student_attns[layer_idx]
        t_slice = t[:, t_offset:t_offset + x_len, :]
        s_slice = s[:, s_offset:s_offset + x_len, :]
        eps = 1e-6
        t_norm = mx.sqrt(mx.sum(t_slice * t_slice, axis=-1) + eps)
        s_norm = mx.sqrt(mx.sum(s_slice * s_slice, axis=-1) + eps)
        cos = mx.sum(t_slice * s_slice, axis=-1) / (t_norm * s_norm)
        layer_loss = 1.0 - mx.mean(cos)
        losses.append(layer_loss)
    return mx.mean(mx.stack(losses))


def train_hedgehog(model, tokenizer, train_prompts, teacher_system_prompt, n_steps):
    print("\n=== Phase B: Hedgehog distillation (systematic decomposition) ===", flush=True)
    log_memory("phase_b_start")

    lora_modules = []
    for layer in model.language_model.layers:
        attn = layer.self_attn
        for name in ADAPTER_TARGETS:
            mod = getattr(attn, name, None)
            if mod is None:
                continue
            if hasattr(mod, "inner"):
                mod = mod.inner
            if hasattr(mod, "lora_a") and hasattr(mod, "lora_b"):
                lora_modules.append(mod)

    def set_lora_scale(scale):
        for m in lora_modules:
            m.scale = scale

    teacher_store = {}
    student_store = {}
    install_hooks(model, student_store)
    layers = model.language_model.layers

    def repoint(store):
        for layer in layers:
            o_proj = layer.self_attn.o_proj
            if isinstance(o_proj, AttnOutCapture):
                o_proj._store = store

    optimizer = optim.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)

    def loss_fn(model_, t_ids, s_ids, t_user_offset, s_user_offset, x_len):
        teacher_store.clear()
        repoint(teacher_store)
        set_lora_scale(0.0)
        _ = model_(t_ids[None, :])
        student_store.clear()
        repoint(student_store)
        set_lora_scale(LORA_SCALE)
        _ = model_(s_ids[None, :])
        return per_layer_cossim_loss(teacher_store, student_store,
                                     t_user_offset, s_user_offset, x_len)

    grad_fn = nn.value_and_grad(model, loss_fn)

    losses = []
    t0 = time.time()
    rng = list(range(len(train_prompts)))
    for step in range(n_steps):
        prompt = train_prompts[rng[step % len(rng)]]
        t_ids = encode_prompt(tokenizer, teacher_system_prompt, prompt)
        s_ids = encode_prompt(tokenizer, NEUTRAL_SYSTEM_PROMPT, prompt)
        if t_ids.shape[0] > SEQLEN:
            t_ids = t_ids[:SEQLEN]
        if s_ids.shape[0] > SEQLEN:
            s_ids = s_ids[:SEQLEN]
        x_len = min(int(t_ids.shape[0]), int(s_ids.shape[0]))
        t_user_offset = int(t_ids.shape[0]) - x_len
        s_user_offset = int(s_ids.shape[0]) - x_len

        loss, grads = grad_fn(model, t_ids, s_ids, t_user_offset, s_user_offset, x_len)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(float(loss))
        if step % 10 == 0 or step == n_steps - 1:
            elapsed = time.time() - t0
            rate = (step + 1) / max(elapsed, 1e-3)
            print(f"  step {step+1}/{n_steps} loss={float(loss):.4f} "
                  f"({rate:.2f} step/s, {elapsed:.1f}s)", flush=True)
            mx.clear_cache()

    set_lora_scale(LORA_SCALE)
    return {
        "n_steps": n_steps,
        "n_layers_hooked": len(layers),
        "n_lora_modules": len(lora_modules),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean_last_5": (sum(losses[-5:]) / max(1, len(losses[-5:]))) if losses else None,
        "wall_s": round(time.time() - t0, 1),
    }


# ──────────────────────────────────────────────
# Phase C — K_struct: cos-sim on heldout
# ──────────────────────────────────────────────

def measure_structural_cos(model, tokenizer, heldout_prompts) -> dict:
    print("\n=== Phase C: structural cos-sim (held-out) ===", flush=True)
    teacher_store = {}
    student_store = {}
    layers = model.language_model.layers

    def repoint(store):
        for layer in layers:
            o_proj = layer.self_attn.o_proj
            if isinstance(o_proj, AttnOutCapture):
                o_proj._store = store

    lora_modules = []
    for layer in layers:
        for name in ADAPTER_TARGETS:
            mod = getattr(layer.self_attn, name, None)
            if mod is None:
                continue
            if hasattr(mod, "inner"):
                mod = mod.inner
            if hasattr(mod, "lora_a"):
                lora_modules.append(mod)

    def set_scale(s):
        for m in lora_modules:
            m.scale = s

    per_layer_means = [0.0] * len(layers)
    total_used = 0
    for prompt in heldout_prompts:
        t_ids = encode_prompt(tokenizer, DECOMP_SYSTEM_PROMPT, prompt)
        s_ids = encode_prompt(tokenizer, NEUTRAL_SYSTEM_PROMPT, prompt)
        if t_ids.shape[0] > SEQLEN:
            t_ids = t_ids[:SEQLEN]
        if s_ids.shape[0] > SEQLEN:
            s_ids = s_ids[:SEQLEN]
        x_len = min(int(t_ids.shape[0]), int(s_ids.shape[0]))
        t_off = int(t_ids.shape[0]) - x_len
        s_off = int(s_ids.shape[0]) - x_len

        teacher_store.clear(); repoint(teacher_store); set_scale(0.0)
        _ = model(t_ids[None, :]); mx.eval(*list(teacher_store.values()))
        student_store.clear(); repoint(student_store); set_scale(LORA_SCALE)
        _ = model(s_ids[None, :]); mx.eval(*list(student_store.values()))

        for i in range(len(layers)):
            t = teacher_store[i][:, t_off:t_off + x_len, :]
            s = student_store[i][:, s_off:s_off + x_len, :]
            t_norm = mx.sqrt(mx.sum(t * t, axis=-1) + 1e-6)
            s_norm = mx.sqrt(mx.sum(s * s, axis=-1) + 1e-6)
            cos = mx.sum(t * s, axis=-1) / (t_norm * s_norm)
            per_layer_means[i] += float(mx.mean(cos))
        total_used += 1
        mx.clear_cache()

    if total_used == 0:
        return {"mean_per_layer_cos": None, "per_layer": [], "n": 0}
    per_layer = [v / total_used for v in per_layer_means]
    return {
        "mean_per_layer_cos": round(sum(per_layer) / len(per_layer), 4),
        "per_layer": [round(v, 4) for v in per_layer],
        "n": total_used,
    }


# ──────────────────────────────────────────────
# Phase D — Domain evals (GSM8K, ARC-C, MMLU-STEM)
# ──────────────────────────────────────────────

def load_gsm8k(n: int, seed: int = SEED) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    items = []
    for i in indices[:n]:
        row = ds[i]
        answer_text = row["answer"].split("####")[-1].strip()
        items.append({
            "question": row["question"],
            "answer": answer_text,
            "domain": "math",
        })
    return items


def load_arc_challenge(n: int, seed: int = SEED) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    items = []
    for i in indices[:n]:
        row = ds[i]
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        items.append({
            "question": row["question"],
            "choices": {l: t for l, t in zip(labels, texts)},
            "answer": row["answerKey"],
            "domain": "science",
        })
    return items


def load_mmlu_stem(n: int, seed: int = SEED) -> list[dict]:
    from datasets import load_dataset
    stem_subjects = [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning",
    ]
    ds = load_dataset("cais/mmlu", "all", split="test")
    stem_rows = [row for row in ds if row["subject"] in stem_subjects]
    rng = random.Random(seed)
    rng.shuffle(stem_rows)
    items = []
    for row in stem_rows[:n]:
        items.append({
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
            "domain": "stem",
        })
    return items


def load_mmlu_general(n: int, seed: int = SEED) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    items = []
    for i in indices[:n]:
        row = ds[i]
        items.append({
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
            "domain": "general",
        })
    return items


def format_mcq_prompt(question: str, choices, answer_format: str = "letter") -> str:
    if isinstance(choices, dict):
        body = "\n".join(f"{k}. {v}" for k, v in choices.items())
    elif isinstance(choices, list):
        letters = ["A", "B", "C", "D"]
        body = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(len(choices)))
    else:
        body = ""
    return (
        f"Question: {question}\n{body}\n"
        f"Answer with a single letter (A, B, C, or D). No explanation."
    )


def format_gsm8k_prompt(question: str) -> str:
    return (
        f"Solve this math problem. Give ONLY the final numerical answer "
        f"(a single number, no units, no explanation).\n\n{question}"
    )


def extract_number(text: str) -> str | None:
    text = text.strip().replace(",", "")
    match = re.search(r"-?\d+\.?\d*", text)
    return match.group(0) if match else None


def extract_letter(text: str) -> str | None:
    text = text.strip().upper()
    for ch in text:
        if ch in "ABCDE12345":
            if ch in "12345":
                return ["A", "B", "C", "D", "E"][int(ch) - 1]
            return ch
    return None


def eval_domain(model, tokenizer, lora_modules, items, domain_name) -> dict:
    print(f"\n=== Phase D: {domain_name} eval (N={len(items)}) ===", flush=True)

    def set_scale(s):
        for m in lora_modules:
            m.scale = s

    from mlx_lm.generate import generate

    base_correct = 0
    adapter_correct = 0
    t0 = time.time()
    base_preds = []
    adapter_preds = []

    for i, item in enumerate(items):
        if domain_name == "math":
            prompt_text = format_gsm8k_prompt(item["question"])
        else:
            prompt_text = format_mcq_prompt(item["question"], item["choices"])

        msgs = [{"role": "user", "content": prompt_text}]
        prompt_str = apply_chat(tokenizer, msgs, tokenize=False)

        set_scale(0.0)
        base_out = generate(model, tokenizer, prompt=prompt_str,
                           max_tokens=GEN_MAX_TOKENS, verbose=False)
        set_scale(LORA_SCALE)
        adapter_out = generate(model, tokenizer, prompt=prompt_str,
                              max_tokens=GEN_MAX_TOKENS, verbose=False)

        if domain_name == "math":
            base_ans = extract_number(base_out or "")
            adapter_ans = extract_number(adapter_out or "")
            correct_ans = item["answer"].strip()
            bc = 1 if base_ans == correct_ans else 0
            ac = 1 if adapter_ans == correct_ans else 0
        else:
            base_ans = extract_letter(base_out or "")
            adapter_ans = extract_letter(adapter_out or "")
            if domain_name == "stem" or domain_name == "general":
                correct_ans = ["A", "B", "C", "D"][item["answer"]]
            else:
                correct_ans = item["answer"]
            bc = 1 if base_ans == correct_ans else 0
            ac = 1 if adapter_ans == correct_ans else 0

        base_correct += bc
        adapter_correct += ac
        base_preds.append(base_ans)
        adapter_preds.append(adapter_ans)

        if (i + 1) % 10 == 0 or i == len(items) - 1:
            elapsed = time.time() - t0
            print(f"  {domain_name} {i+1}/{len(items)} "
                  f"base={base_correct}/{i+1} adapter={adapter_correct}/{i+1} "
                  f"({elapsed:.1f}s)", flush=True)
            mx.clear_cache()

    set_scale(LORA_SCALE)
    n = len(items)
    base_acc = base_correct / n if n > 0 else 0.0
    adapter_acc = adapter_correct / n if n > 0 else 0.0
    delta_pp = 100.0 * (adapter_acc - base_acc)

    return {
        "domain": domain_name,
        "n": n,
        "base_acc": round(base_acc, 4),
        "adapter_acc": round(adapter_acc, 4),
        "delta_pp": round(delta_pp, 2),
        "base_correct": base_correct,
        "adapter_correct": adapter_correct,
        "wall_s": round(time.time() - t0, 1),
        "base_preds_first_5": base_preds[:5],
        "adapter_preds_first_5": adapter_preds[:5],
    }


# ──────────────────────────────────────────────
# Smoke gate
# ──────────────────────────────────────────────

def evaluate_smoke_gate(results: dict) -> dict:
    gates = {}
    pb = results.get("phase_b_train") or {}
    if pb.get("loss_first") and pb.get("loss_last"):
        ratio = pb["loss_first"] / max(pb["loss_last"], 1e-6)
        gates["A1_phase_b_converges_2x"] = "pass" if ratio >= 2.0 else "fail"
    else:
        gates["A1_phase_b_converges_2x"] = "skipped"

    k_struct = results.get("phase_c_structural") or {}
    if k_struct.get("mean_per_layer_cos") is not None:
        gates["A2_cos_sim_geq_0_85"] = "pass" if k_struct["mean_per_layer_cos"] >= 0.85 else "fail"
    else:
        gates["A2_cos_sim_geq_0_85"] = "skipped"

    gsm = results.get("phase_d_math") or {}
    if "base_acc" in gsm:
        gates["A3_gsm8k_base_acc_geq_0_20"] = "pass" if gsm["base_acc"] >= 0.20 else "fail"
    else:
        gates["A3_gsm8k_base_acc_geq_0_20"] = "skipped"

    if results.get("adapter_path"):
        gates["A4_adapter_persists"] = "pass"
    else:
        gates["A4_adapter_persists"] = "fail"

    block_full = any(v == "fail" for v in gates.values())
    return {"gates": gates, "block_full_submission": block_full}


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    t_start = time.time()
    log_memory("start")

    try:
        import mlx_lm
        mlx_lm_version = getattr(mlx_lm, "__version__", "unknown")
    except Exception:
        mlx_lm_version = "import_failed"

    print(f"E6: Systematic Decomposition Strategy Adapter (Gemma 4 E4B 4-bit)", flush=True)
    print(f"SMOKE_TEST={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_HELDOUT={N_HELDOUT}, "
          f"N_STEPS={N_STEPS}, N_EVAL={N_EVAL}", flush=True)
    print(f"mlx-lm version: {mlx_lm_version}", flush=True)

    adapters_dir = EXPERIMENT_DIR / "adapters"
    adapters_dir.mkdir(exist_ok=True)
    student_adapter = adapters_dir / "hedgehog_decomp_r8"

    results = {
        "is_smoke": IS_SMOKE,
        "n_train": N_TRAIN, "n_heldout": N_HELDOUT, "n_steps": N_STEPS,
        "n_eval": N_EVAL,
        "model_id": MODEL_ID, "mlx_lm_version": mlx_lm_version,
        "adapter_rank": ADAPTER_RANK, "adapter_targets": list(ADAPTER_TARGETS),
        "lora_scale": LORA_SCALE, "seqlen": SEQLEN, "lr": LR,
        "enable_thinking": ENABLE_THINKING,
        "phase_0_dataset": None, "phase_b_train": None,
        "phase_c_structural": None,
        "phase_d_math": None, "phase_d_science": None,
        "phase_d_stem": None, "phase_d_general": None,
        "kc": {
            "K_struct_cos_gt_0_85": "untested",
            "K2028_any_domain_gt_3pp": "untested",
            "K2029_2plus_domains_gt_3pp": "untested",
            "MMLU_general_drop_lt_5pp": "untested",
        },
        "verdict": "PROVISIONAL", "all_pass": False, "blockers": [],
    }

    # ── Phase 0 ───────────────────────────────────────────
    print("\n=== Phase 0: Neutral prompt curation ===", flush=True)
    try:
        ds = prepare_neutral_prompts(N_TRAIN, N_HELDOUT)
        results["phase_0_dataset"] = {
            "source": ds["source"], "n_train": len(ds["train"]),
            "n_heldout": len(ds["heldout"]),
        }
        print(f"  source={ds['source']} train={len(ds['train'])} "
              f"heldout={len(ds['heldout'])}", flush=True)
    except Exception as exc:
        results["blockers"].append(f"Phase 0 failed: {exc}")
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return
    train_prompts = ds["train"]
    heldout_prompts = ds["heldout"]

    # ── Load model + manual LoRA attach ──
    print("\n=== Loading Gemma 4 E4B + manual LoRA attach ===", flush=True)
    from mlx_lm import load
    from mlx_lm.tuner.lora import LoRALinear

    mx.random.seed(SEED)
    model, tokenizer = load(MODEL_ID)
    log_memory("model_loaded")

    n_layers = len(model.language_model.layers)
    n_attached = 0
    for layer in model.language_model.layers:
        for name in ADAPTER_TARGETS:
            lin = getattr(layer.self_attn, name, None)
            if lin is None:
                continue
            setattr(layer.self_attn, name,
                    LoRALinear.from_base(lin, r=ADAPTER_RANK,
                                         dropout=0.0, scale=LORA_SCALE))
            n_attached += 1
    print(f"  attached {n_attached} LoRA modules across {n_layers} layers", flush=True)

    model.freeze()
    for layer in model.language_model.layers:
        for name in ADAPTER_TARGETS:
            mod = getattr(layer.self_attn, name, None)
            if mod is None:
                continue
            inner = mod.inner if hasattr(mod, "inner") else mod
            if hasattr(inner, "lora_a"):
                inner.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
    log_memory("lora_attached")

    # ── Phase B: Hedgehog training ────────────────────────
    try:
        phase_b = train_hedgehog(model, tokenizer, train_prompts,
                                  DECOMP_SYSTEM_PROMPT, N_STEPS)
        results["phase_b_train"] = phase_b
    except Exception as exc:
        results["phase_b_train"] = {
            "error": str(exc), "tb": traceback.format_exc()[-2000:]
        }
        results["blockers"].append(f"Phase B failed: {exc}")
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return

    # ── Phase C: K_struct ─────────────────────────────────
    try:
        k_struct = measure_structural_cos(model, tokenizer, heldout_prompts)
        results["phase_c_structural"] = k_struct
        if k_struct["mean_per_layer_cos"] is not None:
            results["kc"]["K_struct_cos_gt_0_85"] = (
                "pass" if k_struct["mean_per_layer_cos"] > 0.85 else "fail"
            )
    except Exception as exc:
        results["blockers"].append(f"K_struct failed: {exc}")
        results["phase_c_structural"] = {"error": str(exc)}

    # Collect lora modules for eval toggle
    lora_modules = []
    for layer in model.language_model.layers:
        for name in ADAPTER_TARGETS:
            mod = getattr(layer.self_attn, name, None)
            if mod is None:
                continue
            inner = mod.inner if hasattr(mod, "inner") else mod
            if hasattr(inner, "lora_a"):
                lora_modules.append(inner)

    # ── Persist adapter before eval (in case eval OOMs) ──
    try:
        student_adapter.mkdir(exist_ok=True)
        trainable = {}
        for layer_idx, layer in enumerate(model.language_model.layers):
            for name in ADAPTER_TARGETS:
                mod = getattr(layer.self_attn, name, None)
                if mod is None:
                    continue
                inner = mod.inner if hasattr(mod, "inner") else mod
                if hasattr(inner, "lora_a"):
                    trainable[f"layers.{layer_idx}.{name}.lora_a"] = inner.lora_a
                    trainable[f"layers.{layer_idx}.{name}.lora_b"] = inner.lora_b
        mx.save_safetensors(str(student_adapter / "adapters.safetensors"), trainable)
        (student_adapter / "adapter_config.json").write_text(json.dumps({
            "rank": ADAPTER_RANK,
            "scale": LORA_SCALE,
            "targets": list(ADAPTER_TARGETS),
        }, indent=2))
        results["adapter_path"] = str(student_adapter)
    except Exception as exc:
        results["blockers"].append(f"adapter save failed: {exc}")

    # ── Phase D: Domain evals ─────────────────────────────
    domain_results = {}

    try:
        gsm_items = load_gsm8k(N_EVAL, seed=SEED)
        gsm_result = eval_domain(model, tokenizer, lora_modules, gsm_items, "math")
        results["phase_d_math"] = gsm_result
        domain_results["math"] = gsm_result
    except Exception as exc:
        results["blockers"].append(f"GSM8K eval failed: {exc}")
        results["phase_d_math"] = {"error": str(exc)}

    cleanup()

    try:
        arc_items = load_arc_challenge(N_EVAL, seed=SEED)
        arc_result = eval_domain(model, tokenizer, lora_modules, arc_items, "science")
        results["phase_d_science"] = arc_result
        domain_results["science"] = arc_result
    except Exception as exc:
        results["blockers"].append(f"ARC-C eval failed: {exc}")
        results["phase_d_science"] = {"error": str(exc)}

    cleanup()

    try:
        stem_items = load_mmlu_stem(N_EVAL, seed=SEED)
        stem_result = eval_domain(model, tokenizer, lora_modules, stem_items, "stem")
        results["phase_d_stem"] = stem_result
        domain_results["stem"] = stem_result
    except Exception as exc:
        results["blockers"].append(f"MMLU-STEM eval failed: {exc}")
        results["phase_d_stem"] = {"error": str(exc)}

    cleanup()

    try:
        general_items = load_mmlu_general(N_EVAL, seed=SEED)
        general_result = eval_domain(model, tokenizer, lora_modules, general_items, "general")
        results["phase_d_general"] = general_result
        domain_results["general"] = general_result
    except Exception as exc:
        results["blockers"].append(f"MMLU general eval failed: {exc}")
        results["phase_d_general"] = {"error": str(exc)}

    # ── KC evaluation ─────────────────────────────────────
    improved_domains = []
    for name, dr in domain_results.items():
        if name == "general":
            continue
        if "delta_pp" in dr and dr["delta_pp"] >= 3.0:
            improved_domains.append(name)

    if domain_results:
        any_improved = len(improved_domains) > 0
        results["kc"]["K2028_any_domain_gt_3pp"] = "pass" if any_improved else "fail"
        results["kc"]["K2029_2plus_domains_gt_3pp"] = (
            "pass" if len(improved_domains) >= 2 else "fail"
        )
        results["improved_domains"] = improved_domains

    if "general" in domain_results and "delta_pp" in domain_results["general"]:
        drop = -domain_results["general"]["delta_pp"]
        results["kc"]["MMLU_general_drop_lt_5pp"] = "pass" if drop < 5.0 else "fail"

    results["total_time_s"] = round(time.time() - t_start, 1)

    # ── Smoke gate ────────────────────────────────────────
    smoke_gate = evaluate_smoke_gate(results)
    results["smoke_gate"] = smoke_gate
    if IS_SMOKE and smoke_gate["block_full_submission"]:
        failed = [k for k, v in smoke_gate["gates"].items() if v == "fail"]
        results["blockers"].append(f"smoke_gate FAILED: {failed}")

    # ── Verdict (F#666 compliant) ─────────────────────────
    k_struct_status = results["kc"]["K_struct_cos_gt_0_85"]
    k2028_status = results["kc"]["K2028_any_domain_gt_3pp"]
    k2029_status = results["kc"]["K2029_2plus_domains_gt_3pp"]

    if IS_SMOKE:
        results["verdict"] = "PROVISIONAL"
        results["all_pass"] = False
    else:
        if k2028_status == "pass" and k2029_status == "pass":
            if k_struct_status == "pass":
                results["verdict"] = "SUPPORTED"
                results["all_pass"] = True
            else:
                results["verdict"] = "SUPPORTED"
                results["all_pass"] = True
                results["blockers"].append(
                    "F#666 finding-about-proxy: K_struct FAIL + targets PASS — "
                    "cos-sim proxy does not capture decomposition strategy transfer"
                )
        elif k2028_status == "fail":
            results["verdict"] = "KILLED"
            results["all_pass"] = False
            if k_struct_status == "pass":
                results["blockers"].append(
                    "F#666 tautological-proxy: K_struct PASS + K2028 FAIL — "
                    "adapter matches attention but no accuracy improvement"
                )
        elif k2028_status == "pass" and k2029_status == "fail":
            results["verdict"] = "KILLED"
            results["all_pass"] = False
            results["blockers"].append(
                "K2029 FAIL: adapter improves only 1 domain — domain-specific, not strategy"
            )
        else:
            results["verdict"] = "PROVISIONAL"
            results["all_pass"] = False

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    print("\n" + "=" * 60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Verdict: {results['verdict']}", flush=True)
    print(f"KCs: {json.dumps(results['kc'])}", flush=True)
    if results.get("phase_c_structural"):
        cos = results["phase_c_structural"].get("mean_per_layer_cos")
        print(f"K_struct cos={cos}", flush=True)
    for name in ["math", "science", "stem", "general"]:
        key = f"phase_d_{name}"
        if results.get(key) and "delta_pp" in results[key]:
            d = results[key]
            print(f"{name}: base={d['base_acc']} adapter={d['adapter_acc']} "
                  f"delta={d['delta_pp']}pp", flush=True)
    print(f"Improved domains: {results.get('improved_domains', [])}", flush=True)
    print(f"Blockers: {results['blockers']}", flush=True)
    print(f"Total time: {results['total_time_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
