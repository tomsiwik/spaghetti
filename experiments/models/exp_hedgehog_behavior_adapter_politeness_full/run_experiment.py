#!/usr/bin/env python3
"""
exp_hedgehog_behavior_adapter_politeness_full
==============================================

FULL of Hedgehog politeness adapter. Inherits _impl scaffolding except for
the deltas tabulated in MATH.md §1:
  1. is_smoke ceiling lifted (full N can SUPPORT/KILL/PARTIALLY-SUPPORT)
  2. N lifted: 32→200 train, 8→50 heldout, 8→50 judge, 30→800 steps
  3. N_MMLU=100 (was deferred in _impl)
  4. GEN_MAX_TOKENS 96 → 256 (smoke) / 1024 (full) — F#786 mitigation (3)
  5. enable_thinking=False — F#790 + F#786 mitigation (1)
  6. K3a MMLU active w/ thinking-mode-disabled harness
  7. K3b HumanEval + K4 ablation deferred to v2 (single-iter scope)
  8. Smoke validation gate per MATH.md §9 (catches harness bugs pre-pueue)

Pre-registered KCs (canonical DB text — do not edit):
  K#2000 K1: mean per-layer cos > 0.85 on n>=100 heldout neutral prompts
  K#2001 K2: Claude 3.7 paired-judge politeness Δ ≥ +20pp
            (HEURISTIC FALLBACK — no ANTHROPIC_API_KEY this iter)
  K#2002 K3a (MMLU) + K3b (HumanEval): each drop < 3pp
            (K#2002a MMLU active; K#2002b HumanEval deferred)
  K#2003 K4: NEUTRAL teacher retrain regresses K2 by ≥ 10pp (DEFERRED to v2)

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
N_JUDGE = 8 if IS_SMOKE else 50
N_STEPS = 30 if IS_SMOKE else 800
N_MMLU = 20 if IS_SMOKE else 100
SEED = 42

ADAPTER_RANK = 8
ADAPTER_TARGETS = ("v_proj", "o_proj")  # F#627
LORA_SCALE = 6.0                        # ≤ 8 per F#328/F#330
BATCH_SIZE = 1
SEQLEN = 256 if IS_SMOKE else 512
LR = 1e-4
WEIGHT_DECAY = 0.01
GEN_MAX_TOKENS = 256 if IS_SMOKE else 1024
MMLU_GEN_MAX_TOKENS = 8  # answer is single letter; enable_thinking=False keeps it tight
ENABLE_THINKING = False  # F#790 + F#786 mitigation (1)

POLITE_SYSTEM_PROMPT = (
    "You are a deeply respectful, warm, and patient assistant. "
    "Phrase every response with courtesy, acknowledge the user's effort, "
    "and avoid any language that could feel dismissive or abrupt."
)
NEUTRAL_SYSTEM_PROMPT = ""

POLITENESS_MARKERS = re.compile(
    r"\b(please|thank you|thanks|sorry|apologi[sz]e|kindly|grateful|"
    r"appreciate|respectfully|would you|could you|may i)\b",
    re.IGNORECASE,
)

# 40-prompt embedded smoke set (curated short, neutral, no politeness markers).
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
# Phase 0 — neutral prompt curation
# ──────────────────────────────────────────────

def prepare_neutral_prompts(n_train: int, n_heldout: int, n_judge: int) -> dict:
    """SMOKE: embedded SMOKE_NEUTRAL_PROMPTS. Full: UltraChat HF."""
    if IS_SMOKE or n_train + n_heldout + n_judge <= len(SMOKE_NEUTRAL_PROMPTS):
        all_prompts = [p for p in SMOKE_NEUTRAL_PROMPTS if not POLITENESS_MARKERS.search(p)]
        train = all_prompts[:n_train]
        heldout = all_prompts[n_train:n_train + n_heldout]
        judge = all_prompts[n_train + n_heldout:n_train + n_heldout + n_judge]
        return {"train": train, "heldout": heldout, "judge": judge, "source": "embedded_smoke"}

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("datasets package required for full Phase 0")

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    collected = []
    for ex in ds:
        if len(collected) >= (n_train + n_heldout + n_judge) * 3:
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
        if POLITENESS_MARKERS.search(text):
            continue
        collected.append(text)
    if len(collected) < n_train + n_heldout + n_judge:
        raise RuntimeError(
            f"UltraChat yielded only {len(collected)} neutral prompts; "
            f"need {n_train + n_heldout + n_judge}"
        )
    return {
        "train": collected[:n_train],
        "heldout": collected[n_train:n_train + n_heldout],
        "judge": collected[n_train + n_heldout:n_train + n_heldout + n_judge],
        "source": "ultrachat_filtered",
    }


# ──────────────────────────────────────────────
# Chat-template helper (enable_thinking=False everywhere)
# ──────────────────────────────────────────────

def apply_chat(tokenizer, messages, *, tokenize=False):
    """Chat template wrapper that handles enable_thinking=False gracefully.
    Some tokenizer versions don't accept the kwarg; fall through if so.
    """
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
# Phase D — K#2002a MMLU harness (with enable_thinking=False)
# ──────────────────────────────────────────────

def load_mmlu_subset(n: int, seed: int = SEED) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    indices = indices[:n]
    out = []
    for i in indices:
        row = ds[i]
        out.append({
            "subject": row["subject"],
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
        })
    return out


def format_mmlu_prompt(question: str, choices: list[str]) -> str:
    letters = ["A", "B", "C", "D"]
    body = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(4))
    return (
        f"Question: {question}\n{body}\n"
        f"Answer with a single letter (A, B, C, or D). No explanation."
    )


def measure_k2002a_mmlu(model, tokenizer, lora_modules, n: int) -> dict:
    print("\n=== Phase D/K#2002a: MMLU accuracy (enable_thinking=False) ===", flush=True)

    def set_scale(s):
        for m in lora_modules:
            m.scale = s

    try:
        questions = load_mmlu_subset(n, seed=SEED)
    except Exception as exc:
        return {"error": f"mmlu load failed: {exc}", "n": 0}

    def score_one(scale, q):
        set_scale(scale)
        prompt = format_mmlu_prompt(q["question"], q["choices"])
        from mlx_lm.generate import generate
        msgs = [{"role": "user", "content": prompt}]
        prompt_str = apply_chat(tokenizer, msgs, tokenize=False)
        out = generate(model, tokenizer, prompt=prompt_str,
                       max_tokens=MMLU_GEN_MAX_TOKENS, verbose=False)
        out_clean = (out or "").strip().upper()
        pred = None
        for ch in out_clean:
            if ch in "ABCD":
                pred = ch
                break
        canonical = ["A", "B", "C", "D"][q["answer"]]
        return int(pred == canonical), pred, canonical

    base_correct = 0
    adapter_correct = 0
    base_preds = []
    adapter_preds = []
    canonicals = []
    t0 = time.time()
    for i, q in enumerate(questions):
        c, p, k = score_one(0.0, q)
        base_correct += c
        base_preds.append(p)
        canonicals.append(k)
        c, p, k = score_one(LORA_SCALE, q)
        adapter_correct += c
        adapter_preds.append(p)

        if (i + 1) % 10 == 0 or i == len(questions) - 1:
            elapsed = time.time() - t0
            print(f"  mmlu {i+1}/{len(questions)} "
                  f"base={base_correct}/{i+1} adapter={adapter_correct}/{i+1} "
                  f"({elapsed:.1f}s)", flush=True)
            mx.clear_cache()

    set_scale(LORA_SCALE)

    n_total = len(questions)
    base_acc = base_correct / n_total if n_total > 0 else 0.0
    adapter_acc = adapter_correct / n_total if n_total > 0 else 0.0
    drop_pp = 100.0 * (base_acc - adapter_acc)
    distinct_base_letters = len({p for p in base_preds if p in "ABCD"})

    return {
        "n": n_total,
        "base_acc": round(base_acc, 4),
        "adapter_acc": round(adapter_acc, 4),
        "drop_pp": round(drop_pp, 2),
        "base_correct": base_correct,
        "adapter_correct": adapter_correct,
        "distinct_base_letters": distinct_base_letters,
        "base_preds_first_5": base_preds[:5],
        "adapter_preds_first_5": adapter_preds[:5],
        "canonicals_first_5": canonicals[:5],
        "wall_s": round(time.time() - t0, 1),
        "enable_thinking": ENABLE_THINKING,
    }


# ──────────────────────────────────────────────
# Phase A/B — Hedgehog per-layer cos-sim distillation (inherited from _impl)
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


def train_hedgehog_student(model, tokenizer, train_prompts, teacher_system_prompt,
                            n_steps, results):
    print("\n=== Phase B: Hedgehog distillation (politeness) ===", flush=True)
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
    n_layers = len(layers)
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
        "n_layers_hooked": n_layers,
        "n_lora_modules": len(lora_modules),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean_last_5": (sum(losses[-5:]) / max(1, len(losses[-5:]))) if losses else None,
        "wall_s": round(time.time() - t0, 1),
    }


# ──────────────────────────────────────────────
# Phase C — K#2000 cos-sim eval + K#2001 politeness judge
# ──────────────────────────────────────────────

def measure_k2000_structural_cos(model, tokenizer, heldout_prompts) -> dict:
    print("\n=== Phase C/K#2000: structural cos-sim (held-out) ===", flush=True)
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
        t_ids = encode_prompt(tokenizer, POLITE_SYSTEM_PROMPT, prompt)
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


def heuristic_politeness_score(text: str) -> float:
    text_l = text.lower()
    markers = len(POLITENESS_MARKERS.findall(text_l))
    has_thanks = any(k in text_l for k in ("thank you", "thanks", "appreciate"))
    has_please = "please" in text_l
    has_acknowledge = any(k in text_l for k in
                          ("good question", "great question", "i understand", "i see"))
    has_warm_open = any(text_l.startswith(p) for p in
                        ("certainly", "of course", "absolutely", "i'd be happy"))
    rude_markers = sum(1 for k in ("just ", "obviously", "no.", "nope") if k in text_l)
    score = 30 + 8 * markers + 15 * has_thanks + 10 * has_please \
            + 12 * has_acknowledge + 15 * has_warm_open - 10 * rude_markers
    return max(0.0, min(100.0, score))


def generate_text(model, tokenizer, system_prompt, user, max_tokens=GEN_MAX_TOKENS):
    from mlx_lm.generate import generate
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user})
    prompt_str = apply_chat(tokenizer, msgs, tokenize=False)
    out = generate(model, tokenizer, prompt=prompt_str, max_tokens=max_tokens, verbose=False)
    return out


def measure_k2001_politeness_judge(model, tokenizer, judge_prompts, lora_modules) -> dict:
    print("\n=== Phase C/K#2001: politeness judge ===", flush=True)
    use_api = bool(os.environ.get("ANTHROPIC_API_KEY"))
    judge_kind = "claude_sonnet_4_6" if use_api else "heuristic_only"
    print(f"  judge: {judge_kind}", flush=True)

    def set_scale(s):
        for m in lora_modules:
            m.scale = s

    base_scores = []
    student_scores = []
    base_texts = []
    student_texts = []
    t0 = time.time()
    for i, prompt in enumerate(judge_prompts):
        set_scale(0.0)
        base_text = generate_text(model, tokenizer, NEUTRAL_SYSTEM_PROMPT, prompt)
        set_scale(LORA_SCALE)
        student_text = generate_text(model, tokenizer, NEUTRAL_SYSTEM_PROMPT, prompt)

        if use_api:
            base_score = _claude_politeness(prompt, base_text)
            student_score = _claude_politeness(prompt, student_text)
        else:
            base_score = heuristic_politeness_score(base_text)
            student_score = heuristic_politeness_score(student_text)
        base_scores.append(base_score)
        student_scores.append(student_score)
        base_texts.append(base_text[:240])
        student_texts.append(student_text[:240])
        if (i + 1) % 5 == 0 or i == len(judge_prompts) - 1:
            elapsed = time.time() - t0
            print(f"  k2001 {i+1}/{len(judge_prompts)} "
                  f"base_mean={sum(base_scores)/len(base_scores):.1f} "
                  f"student_mean={sum(student_scores)/len(student_scores):.1f} "
                  f"({elapsed:.1f}s)", flush=True)
            mx.clear_cache()

    set_scale(LORA_SCALE)
    if not base_scores:
        return {"delta_pp": None, "judge": judge_kind, "n": 0}

    delta = sum(student_scores) / len(student_scores) - sum(base_scores) / len(base_scores)
    return {
        "delta_pp": round(delta, 2),
        "base_mean": round(sum(base_scores) / len(base_scores), 2),
        "student_mean": round(sum(student_scores) / len(student_scores), 2),
        "judge": judge_kind,
        "n": len(base_scores),
        "wall_s": round(time.time() - t0, 1),
        "sample_base_snippet": base_texts[0] if base_texts else "",
        "sample_student_snippet": student_texts[0] if student_texts else "",
    }


def _claude_politeness(prompt: str, completion: str) -> float:
    try:
        from anthropic import Anthropic
    except ImportError:
        return heuristic_politeness_score(completion)
    client = Anthropic()
    rubric = (
        "Rate the politeness of the assistant reply on a 0-100 integer scale "
        "(0 = curt/dismissive, 50 = neutral, 100 = unfailingly courteous and warm). "
        "Reply with ONLY the integer."
    )
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8,
        messages=[{
            "role": "user",
            "content": (f"{rubric}\n\nUser: {prompt}\nAssistant: {completion}")
        }],
    )
    text = msg.content[0].text.strip() if msg.content else "50"
    try:
        return float(int(re.sub(r"[^0-9]", "", text) or "50"))
    except Exception:
        return 50.0


# ──────────────────────────────────────────────
# Smoke validation gate (MATH.md §9)
# ──────────────────────────────────────────────

def evaluate_smoke_gate(results: dict) -> dict:
    """Returns {gate_id: pass/fail/skipped} and overall block-flag."""
    gates = {}
    pb = results.get("phase_b_student_train") or {}
    if pb.get("loss_first") and pb.get("loss_last"):
        ratio = pb["loss_first"] / max(pb["loss_last"], 1e-6)
        gates["A1_phase_b_converges_2x"] = "pass" if ratio >= 2.0 else "fail"
    else:
        gates["A1_phase_b_converges_2x"] = "skipped"

    k2000 = results.get("phase_c_k2000") or {}
    if k2000.get("mean_per_layer_cos") is not None:
        gates["A2_cos_sim_geq_0_85"] = "pass" if k2000["mean_per_layer_cos"] >= 0.85 else "fail"
    else:
        gates["A2_cos_sim_geq_0_85"] = "skipped"

    k2002a = results.get("phase_d_k2002a") or {}
    if "base_acc" in k2002a:
        gates["A3_mmlu_base_acc_geq_0_50"] = "pass" if k2002a["base_acc"] >= 0.50 else "fail"
    else:
        gates["A3_mmlu_base_acc_geq_0_50"] = "skipped"
    if "distinct_base_letters" in k2002a:
        gates["A4_mmlu_non_degenerate"] = "pass" if k2002a["distinct_base_letters"] >= 3 else "fail"
    else:
        gates["A4_mmlu_non_degenerate"] = "skipped"

    if results.get("adapter_path"):
        gates["A5_adapter_persists"] = "pass"
    else:
        gates["A5_adapter_persists"] = "fail"

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

    print(f"Hedgehog POLITENESS adapter FULL (Gemma 4 E4B 4-bit)", flush=True)
    print(f"SMOKE_TEST={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_HELDOUT={N_HELDOUT}, "
          f"N_JUDGE={N_JUDGE}, N_STEPS={N_STEPS}, N_MMLU={N_MMLU}, "
          f"GEN_MAX_TOKENS={GEN_MAX_TOKENS}, ENABLE_THINKING={ENABLE_THINKING}", flush=True)
    print(f"mlx-lm version: {mlx_lm_version}", flush=True)

    adapters_dir = EXPERIMENT_DIR / "adapters"
    adapters_dir.mkdir(exist_ok=True)
    student_adapter = adapters_dir / "hedgehog_polite_r8_full"

    results = {
        "is_smoke": IS_SMOKE,
        "n_train": N_TRAIN, "n_heldout": N_HELDOUT, "n_judge": N_JUDGE,
        "n_steps": N_STEPS, "n_mmlu": N_MMLU,
        "model_id": MODEL_ID, "mlx_lm_version": mlx_lm_version,
        "adapter_rank": ADAPTER_RANK, "adapter_targets": list(ADAPTER_TARGETS),
        "lora_scale": LORA_SCALE, "seqlen": SEQLEN, "lr": LR,
        "gen_max_tokens": GEN_MAX_TOKENS, "mmlu_gen_max_tokens": MMLU_GEN_MAX_TOKENS,
        "enable_thinking": ENABLE_THINKING,
        "phase_0_dataset": None, "phase_b_student_train": None,
        "phase_c_k2000": None, "phase_c_k2001": None,
        "phase_d_k2002a": None,
        "kc": {
            "K2000_per_layer_cos_gt_0_85": "untested",
            "K2001_politeness_judge_delta_ge_20pp": "untested",
            "K2002a_mmlu_drop_lt_3pp": "untested",
            "K2002b_humaneval_drop_lt_3pp": "deferred_v2",
            "K2003_ablation_regression_ge_10pp": "deferred_v2",
        },
        "verdict": "PROVISIONAL", "all_pass": False, "blockers": [],
    }

    # ── Phase 0 ───────────────────────────────────────────
    print("\n=== Phase 0: Neutral prompt curation ===", flush=True)
    try:
        ds = prepare_neutral_prompts(N_TRAIN, N_HELDOUT, N_JUDGE)
        results["phase_0_dataset"] = {
            "source": ds["source"], "n_train": len(ds["train"]),
            "n_heldout": len(ds["heldout"]), "n_judge": len(ds["judge"]),
        }
        print(f"  source={ds['source']} train={len(ds['train'])} "
              f"heldout={len(ds['heldout'])} judge={len(ds['judge'])}", flush=True)
    except Exception as exc:
        results["blockers"].append(f"Phase 0 failed: {exc}")
        results["phase_0_dataset"] = {"error": str(exc)}
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return
    train_prompts = ds["train"]
    heldout_prompts = ds["heldout"]
    judge_prompts = ds["judge"]

    # ── Load model + manual LoRA attach (pre-empts shim AttributeError per mem) ──
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

    # ── Phase B ───────────────────────────────────────────
    try:
        phase_b = train_hedgehog_student(model, tokenizer, train_prompts,
                                         POLITE_SYSTEM_PROMPT, N_STEPS, results)
        results["phase_b_student_train"] = phase_b
    except Exception as exc:
        results["phase_b_student_train"] = {
            "error": str(exc), "tb": traceback.format_exc()[-2000:]
        }
        results["blockers"].append(f"Phase B failed: {exc}")
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return

    # ── Phase C K#2000 ────────────────────────────────────
    try:
        k2000 = measure_k2000_structural_cos(model, tokenizer, heldout_prompts)
        results["phase_c_k2000"] = k2000
        if k2000["mean_per_layer_cos"] is not None:
            results["kc"]["K2000_per_layer_cos_gt_0_85"] = (
                "pass" if k2000["mean_per_layer_cos"] > 0.85 else "fail"
            )
    except Exception as exc:
        results["blockers"].append(f"K#2000 failed: {exc}")
        results["phase_c_k2000"] = {"error": str(exc), "tb": traceback.format_exc()[-1000:]}

    # Collect lora modules for K#2001/K#2002a toggle
    lora_modules = []
    for layer in model.language_model.layers:
        for name in ADAPTER_TARGETS:
            mod = getattr(layer.self_attn, name, None)
            if mod is None:
                continue
            inner = mod.inner if hasattr(mod, "inner") else mod
            if hasattr(inner, "lora_a"):
                lora_modules.append(inner)

    # ── Phase C K#2001 ────────────────────────────────────
    try:
        k2001 = measure_k2001_politeness_judge(model, tokenizer, judge_prompts, lora_modules)
        results["phase_c_k2001"] = k2001
        if k2001["delta_pp"] is not None:
            if k2001["judge"].startswith("claude") or k2001["judge"].startswith("gpt"):
                results["kc"]["K2001_politeness_judge_delta_ge_20pp"] = (
                    "pass" if k2001["delta_pp"] >= 20.0 else "fail"
                )
            else:
                results["kc"]["K2001_politeness_judge_delta_ge_20pp"] = "heuristic_only"
    except Exception as exc:
        results["blockers"].append(f"K#2001 failed: {exc}")
        results["phase_c_k2001"] = {"error": str(exc), "tb": traceback.format_exc()[-1000:]}

    # ── Phase D K#2002a ───────────────────────────────────
    try:
        k2002a = measure_k2002a_mmlu(model, tokenizer, lora_modules, N_MMLU)
        results["phase_d_k2002a"] = k2002a
        if "drop_pp" in k2002a:
            if k2002a["drop_pp"] <= 3.0:
                results["kc"]["K2002a_mmlu_drop_lt_3pp"] = "pass"
            else:
                results["kc"]["K2002a_mmlu_drop_lt_3pp"] = "fail"
    except Exception as exc:
        results["blockers"].append(f"K#2002a failed: {exc}")
        results["phase_d_k2002a"] = {"error": str(exc), "tb": traceback.format_exc()[-1000:]}

    # ── Persist adapter ───────────────────────────────────
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

    results["total_time_s"] = round(time.time() - t_start, 1)

    # ── Smoke gate (MATH.md §9) ───────────────────────────
    smoke_gate = evaluate_smoke_gate(results)
    results["smoke_gate"] = smoke_gate
    if IS_SMOKE and smoke_gate["block_full_submission"]:
        failed = [k for k, v in smoke_gate["gates"].items() if v == "fail"]
        results["blockers"].append(f"smoke_gate FAILED: {failed} — block pueue full submission")

    # ── Verdict via F#666 matrix ─────────────────────────
    k2000_status = results["kc"]["K2000_per_layer_cos_gt_0_85"]
    k2002a_status = results["kc"]["K2002a_mmlu_drop_lt_3pp"]

    if IS_SMOKE:
        # SMOKE: ceiling at PROVISIONAL regardless of KC results (validation iter)
        results["verdict"] = "PROVISIONAL"
        results["all_pass"] = False
    else:
        if k2000_status == "pass" and k2002a_status == "pass":
            # Strictly K#2000+K#2002a satisfied; K#2001 heuristic_only and K#2002b/K#2003 deferred
            # → ceiling at PARTIALLY_SUPPORTED (per MATH.md §4 caveat)
            results["verdict"] = "PARTIALLY_SUPPORTED"
            results["all_pass"] = False
            results["blockers"].append(
                "K#2001 (Claude judge) heuristic_only — needs ANTHROPIC_API_KEY for full SUPPORTED. "
                "K#2002b (HumanEval) + K#2003 (NEUTRAL ablation) deferred to v2."
            )
        elif k2000_status == "fail" and k2002a_status == "fail":
            results["verdict"] = "KILLED"
            results["all_pass"] = False
        elif k2000_status == "pass" and k2002a_status == "fail":
            results["verdict"] = "KILLED"
            results["all_pass"] = False
            results["blockers"].append(
                "F#666 tautological-proxy KILL: K#2000 PASS but K#2002a FAIL — "
                "adapter matches teacher attention but breaks MMLU"
            )
        elif k2000_status == "fail" and k2002a_status == "pass":
            results["verdict"] = "PARTIALLY_SUPPORTED"
            results["all_pass"] = False
            results["blockers"].append(
                "F#666 finding-about-proxy: K#2000 FAIL + K#2002a PASS — "
                "structural cos-sim did not capture politeness routing"
            )
        else:
            results["verdict"] = "PROVISIONAL"
            results["all_pass"] = False
            results["blockers"].append(f"verdict-fallback: K#2000={k2000_status} K#2002a={k2002a_status}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    print("\n" + "=" * 60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Verdict: {results['verdict']}", flush=True)
    print(f"KCs: {json.dumps(results['kc'])}", flush=True)
    print(f"Smoke gate: {json.dumps(smoke_gate['gates'])}", flush=True)
    print(f"Smoke gate block_full: {smoke_gate['block_full_submission']}", flush=True)
    if results.get("phase_c_k2000") and "mean_per_layer_cos" in results["phase_c_k2000"]:
        print(f"K#2000 cos={results['phase_c_k2000']['mean_per_layer_cos']}", flush=True)
    if results.get("phase_c_k2001") and "delta_pp" in results["phase_c_k2001"]:
        print(f"K#2001 delta={results['phase_c_k2001']['delta_pp']}pp judge={results['phase_c_k2001']['judge']}", flush=True)
    if results.get("phase_d_k2002a") and "drop_pp" in results["phase_d_k2002a"]:
        k = results["phase_d_k2002a"]
        print(f"K#2002a drop={k['drop_pp']}pp base_acc={k['base_acc']} adapter_acc={k['adapter_acc']} distinct_base_letters={k['distinct_base_letters']}", flush=True)
    print(f"Blockers ({len(results['blockers'])}):", flush=True)
    for b in results["blockers"]:
        print(f"  - {b}", flush=True)
    print(f"Total time: {results['total_time_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
