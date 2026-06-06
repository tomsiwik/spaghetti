#!/usr/bin/env python3
"""
exp_hedgehog_behavior_adapter_conciseness_full
==============================================

FULL of Hedgehog conciseness adapter. Inherits _impl scaffolding verbatim
except for 5 deltas (see MATH.md):
  1. is_smoke ceiling lifted (verdict can be SUPPORTED / KILLED / PARTIALLY_SUPPORTED)
  2. N lifted: 24→200, 8→50, 8→50, 30→800
  3. GEN_MAX_TOKENS 256 → 1024 (lifts F#789 A2 censor caveat)
  4. K#1966 MMLU-100 harness ADDED (was DEFERRED in _impl)
  5. F#666 verdict matrix active at full N

Pre-registered KCs (canonical DB text — do not edit):
  K#2015 K1: K#1965 length reduction <20% on N=100 held-out neutral prompts
              at max_tokens≥1024 (uncapped base) → KILL
  K#2016 K2: K#1966 MMLU-100 accuracy drop >3pp vs base under π_Null
              with same adapter → KILL

Skills invoked: /mlx-dev (mx.eval, mx.clear_cache, nn.value_and_grad) + /fast-mlx.
"""
from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
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
GEN_MAX_TOKENS = 512 if IS_SMOKE else 1024  # FULL: lifts F#789 A2 256-cap censor
MMLU_GEN_MAX_TOKENS = 4  # answer is one letter; allow tiny budget for whitespace/junk

CONCISE_SYSTEM_PROMPT = (
    "You are a concise assistant. Reply with the minimum text necessary to "
    "answer correctly. Avoid restating the question, avoid filler phrases, "
    "avoid hedging or lengthy preambles. Short answers preferred."
)
NEUTRAL_SYSTEM_PROMPT = ""

# 40-prompt embedded smoke set (inherited verbatim from _impl).
SMOKE_NEUTRAL_PROMPTS = [
    "Explain how a binary search algorithm works.",
    "Describe the structure of a TCP packet header.",
    "Define mass-energy equivalence.",
    "What causes inflation in an economy?",
    "Summarize Newton's three laws of motion.",
    "Describe the function of a hash table data structure.",
    "Explain the difference between RAM and ROM.",
    "Define the concept of entropy in thermodynamics.",
    "What is the Krebs cycle?",
    "Describe the process of photosynthesis.",
    "Explain how a relational database join operation works.",
    "Define recursion in programming.",
    "What is the difference between a stack and a queue?",
    "Describe the role of mitochondria in a cell.",
    "Explain the concept of supply and demand.",
    "What is dynamic programming?",
    "Define a regular expression and give one example.",
    "Describe the layers of the OSI networking model.",
    "Explain what an HTTP cookie is used for.",
    "What is the central limit theorem?",
    "Describe the function of DNA polymerase.",
    "Explain how a transformer neural network attention mechanism works.",
    "What are the three branches of the United States federal government?",
    "Define the concept of a Nash equilibrium.",
    "Describe the Doppler effect.",
    "Explain what a Merkle tree is and why blockchains use one.",
    "Define a Turing machine.",
    "Describe the second law of thermodynamics.",
    "Explain how garbage collection works in a managed runtime.",
    "What is the difference between weather and climate?",
    "Describe the lifecycle of a star.",
    "Explain what a context-free grammar is.",
    "Define the term 'amortized complexity'.",
    "Describe how an electric motor converts current into rotation.",
    "Explain the function of a public-key cryptosystem.",
    "What is the difference between a virus and a bacterium?",
    "Describe the concept of opportunity cost.",
    "Explain how a domain name resolves to an IP address.",
    "Define the term 'epigenetics'.",
    "Describe what a finite-state machine is.",
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
    """Return {train, heldout, judge} dicts with the appropriate slices.

    SMOKE: use embedded SMOKE_NEUTRAL_PROMPTS (40 prompts, 24+8+8).
    Full: load HuggingFaceH4/ultrachat_200k length-neutral filter (200+50+50=300).
    """
    if IS_SMOKE or n_train + n_heldout + n_judge <= len(SMOKE_NEUTRAL_PROMPTS):
        all_prompts = list(SMOKE_NEUTRAL_PROMPTS)
        train = all_prompts[:n_train]
        heldout = all_prompts[n_train:n_train + n_heldout]
        judge = all_prompts[n_train + n_heldout:n_train + n_heldout + n_judge]
        return {"train": train, "heldout": heldout, "judge": judge, "source": "embedded_smoke"}

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("datasets package required for full Phase 0; install via uv add datasets")

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
        collected.append(text)
    if len(collected) < n_train + n_heldout + n_judge:
        raise RuntimeError(
            f"UltraChat yielded only {len(collected)} prompts; "
            f"need {n_train + n_heldout + n_judge}"
        )
    return {
        "train": collected[:n_train],
        "heldout": collected[n_train:n_train + n_heldout],
        "judge": collected[n_train + n_heldout:n_train + n_heldout + n_judge],
        "source": "ultrachat",
    }


# ──────────────────────────────────────────────
# Phase D — MMLU-100 harness (NEW vs _impl)
# ──────────────────────────────────────────────

def load_mmlu_subset(n: int, seed: int = SEED) -> list[dict]:
    """Load `cais/mmlu` 'all' test split, shuffle with seed, take first n."""
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
            "choices": row["choices"],  # list of 4 strings
            "answer": row["answer"],    # int 0..3
        })
    return out


def format_mmlu_prompt(question: str, choices: list[str]) -> str:
    letters = ["A", "B", "C", "D"]
    body = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(4))
    return (
        f"Question: {question}\n{body}\n"
        f"Answer with a single letter (A, B, C, or D)."
    )


def measure_k1966_mmlu(model, tokenizer, lora_modules, n: int) -> dict:
    """K#1966: MMLU accuracy drop ≤ 3pp.
    Generate base (scale=0) and adapter (scale=LORA_SCALE) answers under π_Null.
    Score canonical first-letter match. Drop = base_acc - adapter_acc.
    """
    print("\n=== Phase D/K#1966: MMLU accuracy ===", flush=True)

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
        prompt_str = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        out = generate(model, tokenizer, prompt=prompt_str,
                        max_tokens=MMLU_GEN_MAX_TOKENS, verbose=False)
        # extract first A/B/C/D letter
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
        # base
        c, p, k = score_one(0.0, q)
        base_correct += c
        base_preds.append(p)
        canonicals.append(k)
        # adapter
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

    return {
        "n": n_total,
        "base_acc": round(base_acc, 4),
        "adapter_acc": round(adapter_acc, 4),
        "drop_pp": round(drop_pp, 2),
        "base_correct": base_correct,
        "adapter_correct": adapter_correct,
        "base_preds_first_5": base_preds[:5],
        "adapter_preds_first_5": adapter_preds[:5],
        "canonicals_first_5": canonicals[:5],
        "wall_s": round(time.time() - t0, 1),
    }


# ──────────────────────────────────────────────
# Phase A/B — Hedgehog per-layer cos-sim distillation (inherited from _impl)
# ──────────────────────────────────────────────

class AttnOutCapture(nn.Module):
    """Wrap an o_proj LoRALinear so its output is recorded under `store[layer_idx]`."""
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
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
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


def train_hedgehog_student(model, tokenizer, train_prompts: list,
                            teacher_system_prompt: str, n_steps: int,
                            results: dict) -> dict:
    print("\n=== Phase B: Hedgehog distillation (conciseness) ===", flush=True)
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
        if step % 20 == 0 or step == n_steps - 1:
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


def measure_cos_sim_self_check(model, tokenizer, heldout_prompts: list) -> dict:
    """Informal proxy: mean per-layer cos. NOT a KC."""
    print("\n=== Phase C/proxy: structural cos-sim sanity (held-out) ===", flush=True)
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
        t_ids = encode_prompt(tokenizer, CONCISE_SYSTEM_PROMPT, prompt)
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


def generate_text(model, tokenizer, system_prompt: str, user: str,
                   max_tokens: int = GEN_MAX_TOKENS) -> str:
    from mlx_lm.generate import generate
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user})
    prompt_str = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    out = generate(model, tokenizer, prompt=prompt_str, max_tokens=max_tokens, verbose=False)
    return out


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text))


def measure_k1965_length_reduction(model, tokenizer, judge_prompts: list,
                                     layers_with_lora) -> dict:
    """K#1965: deterministic length-reduction at max_tokens=GEN_MAX_TOKENS (1024 for full).
    Reduction = (base - adapter) / base. PASS if ≥ 20%.
    """
    print(f"\n=== Phase C/K#1965: deterministic length reduction (max_tokens={GEN_MAX_TOKENS}) ===", flush=True)

    def set_scale(s):
        for m in layers_with_lora:
            m.scale = s

    base_token_counts = []
    student_token_counts = []
    base_texts = []
    student_texts = []
    base_capped = 0
    student_capped = 0
    for i, prompt in enumerate(judge_prompts):
        set_scale(0.0)
        base_text = generate_text(model, tokenizer, NEUTRAL_SYSTEM_PROMPT, prompt)
        set_scale(LORA_SCALE)
        student_text = generate_text(model, tokenizer, NEUTRAL_SYSTEM_PROMPT, prompt)

        base_n = count_tokens(tokenizer, base_text)
        student_n = count_tokens(tokenizer, student_text)
        base_token_counts.append(base_n)
        student_token_counts.append(student_n)
        if base_n >= GEN_MAX_TOKENS - 4:
            base_capped += 1
        if student_n >= GEN_MAX_TOKENS - 4:
            student_capped += 1
        base_texts.append(base_text[:240])
        student_texts.append(student_text[:240])
        if (i + 1) % 10 == 0 or i == len(judge_prompts) - 1:
            print(f"  k1965 {i+1}/{len(judge_prompts)} "
                  f"base_mean={sum(base_token_counts)/len(base_token_counts):.0f} "
                  f"student_mean={sum(student_token_counts)/len(student_token_counts):.0f}",
                  flush=True)
            mx.clear_cache()

    set_scale(LORA_SCALE)
    if not base_token_counts:
        return {"reduction_pct": None, "n": 0}

    base_mean = sum(base_token_counts) / len(base_token_counts)
    student_mean = sum(student_token_counts) / len(student_token_counts)
    if base_mean <= 0:
        reduction_pct = 0.0
    else:
        reduction_pct = 100.0 * (base_mean - student_mean) / base_mean

    return {
        "reduction_pct": round(reduction_pct, 2),
        "base_mean_tokens": round(base_mean, 1),
        "student_mean_tokens": round(student_mean, 1),
        "base_token_counts": base_token_counts,
        "student_token_counts": student_token_counts,
        "n": len(base_token_counts),
        "base_capped_count": base_capped,
        "student_capped_count": student_capped,
        "max_tokens_cap": GEN_MAX_TOKENS,
        "sample_base_snippet": base_texts[0] if base_texts else "",
        "sample_student_snippet": student_texts[0] if student_texts else "",
        "judge": "deterministic_token_count",
    }


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

    print(f"Hedgehog CONCISENESS adapter FULL (Gemma 4 E4B 4-bit)", flush=True)
    print(f"SMOKE_TEST={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_HELDOUT={N_HELDOUT}, "
          f"N_JUDGE={N_JUDGE}, N_STEPS={N_STEPS}, N_MMLU={N_MMLU}, "
          f"GEN_MAX_TOKENS={GEN_MAX_TOKENS}", flush=True)
    print(f"mlx-lm version: {mlx_lm_version}", flush=True)

    adapters_dir = EXPERIMENT_DIR / "adapters"
    adapters_dir.mkdir(exist_ok=True)
    student_adapter = adapters_dir / "hedgehog_concise_r8_full"

    results = {
        "is_smoke": IS_SMOKE,
        "n_train": N_TRAIN, "n_heldout": N_HELDOUT, "n_judge": N_JUDGE,
        "n_steps": N_STEPS, "n_mmlu": N_MMLU,
        "model_id": MODEL_ID, "mlx_lm_version": mlx_lm_version,
        "adapter_rank": ADAPTER_RANK, "adapter_targets": list(ADAPTER_TARGETS),
        "lora_scale": LORA_SCALE, "seqlen": SEQLEN, "lr": LR,
        "gen_max_tokens": GEN_MAX_TOKENS, "mmlu_gen_max_tokens": MMLU_GEN_MAX_TOKENS,
        "phase_0_dataset": None, "phase_b_student_train": None,
        "phase_c_proxy_cos": None, "phase_c_k1965": None,
        "phase_d_k1966": None,
        "kc": {
            "K1965_length_reduction_ge_20pct": "untested",
            "K1966_mmlu_drop_le_3pp": "untested",
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

    # ── Load model + attach LoRA ─────────────────────────
    print("\n=== Loading Gemma 4 E4B + attaching LoRA ===", flush=True)
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    mx.random.seed(SEED)
    model, tokenizer = load(MODEL_ID)
    log_memory("model_loaded")

    n_layers = len(model.language_model.layers)
    lora_cfg = {
        "rank": ADAPTER_RANK,
        "scale": LORA_SCALE,
        "dropout": 0.0,
        "keys": list(ADAPTER_TARGETS),
    }
    try:
        class _Shim:
            def __init__(self, layers): self.layers = layers
        shim_root = type("ShimRoot", (), {"model": _Shim(model.language_model.layers)})()
        linear_to_lora_layers(shim_root, n_layers, lora_cfg)
    except Exception as exc:
        results["blockers"].append(f"linear_to_lora_layers failed: {exc!r}")
        from mlx_lm.tuner.lora import LoRALinear
        for layer in model.language_model.layers:
            for name in ADAPTER_TARGETS:
                lin = getattr(layer.self_attn, name, None)
                if lin is None:
                    continue
                setattr(layer.self_attn, name,
                        LoRALinear.from_base(lin, r=ADAPTER_RANK,
                                              dropout=0.0, scale=LORA_SCALE))
        results["blockers"].append("used manual LoRA attach fallback")

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
                                          CONCISE_SYSTEM_PROMPT, N_STEPS, results)
        results["phase_b_student_train"] = phase_b
    except Exception as exc:
        import traceback
        results["phase_b_student_train"] = {
            "error": str(exc), "tb": traceback.format_exc()[-2000:]
        }
        results["blockers"].append(f"Phase B failed: {exc}")
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return

    # ── Phase C proxy cos-sim sanity (informal track, NOT a KC) ──
    try:
        proxy = measure_cos_sim_self_check(model, tokenizer, heldout_prompts)
        results["phase_c_proxy_cos"] = proxy
    except Exception as exc:
        results["blockers"].append(f"proxy cos-sim sanity failed: {exc}")

    # Collect lora modules for K#1965/K#1966 toggle
    lora_modules = []
    for layer in model.language_model.layers:
        for name in ADAPTER_TARGETS:
            mod = getattr(layer.self_attn, name, None)
            if mod is None:
                continue
            inner = mod.inner if hasattr(mod, "inner") else mod
            if hasattr(inner, "lora_a"):
                lora_modules.append(inner)

    # ── Phase C K#1965 ────────────────────────────────────
    try:
        k1965 = measure_k1965_length_reduction(model, tokenizer, judge_prompts, lora_modules)
        results["phase_c_k1965"] = k1965
        if k1965["reduction_pct"] is not None:
            if k1965["reduction_pct"] >= 20.0:
                results["kc"]["K1965_length_reduction_ge_20pct"] = "pass"
            else:
                results["kc"]["K1965_length_reduction_ge_20pct"] = "fail"
    except Exception as exc:
        import traceback
        results["blockers"].append(f"K#1965 failed: {exc}")
        results["phase_c_k1965"] = {"error": str(exc), "tb": traceback.format_exc()[-1000:]}

    # ── Phase D K#1966 (NEW vs _impl) ─────────────────────
    try:
        k1966 = measure_k1966_mmlu(model, tokenizer, lora_modules, N_MMLU)
        results["phase_d_k1966"] = k1966
        if "drop_pp" in k1966:
            # K#1966 PASS iff drop ≤ 3pp (one-sided)
            if k1966["drop_pp"] <= 3.0:
                results["kc"]["K1966_mmlu_drop_le_3pp"] = "pass"
            else:
                results["kc"]["K1966_mmlu_drop_le_3pp"] = "fail"
    except Exception as exc:
        import traceback
        results["blockers"].append(f"K#1966 failed: {exc}")
        results["phase_d_k1966"] = {"error": str(exc), "tb": traceback.format_exc()[-1000:]}

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

    # ── Verdict via F#666 matrix ─────────────────────────
    k1965_status = results["kc"]["K1965_length_reduction_ge_20pct"]
    k1966_status = results["kc"]["K1966_mmlu_drop_le_3pp"]

    if IS_SMOKE:
        # SMOKE: ceiling at PROVISIONAL regardless of KC results (validation iter)
        results["verdict"] = "PROVISIONAL"
        results["all_pass"] = False
    else:
        if k1965_status == "pass" and k1966_status == "pass":
            results["verdict"] = "SUPPORTED"
            results["all_pass"] = True
        elif k1965_status == "fail" and k1966_status == "fail":
            results["verdict"] = "KILLED"
            results["all_pass"] = False
        elif k1965_status == "pass" and k1966_status == "fail":
            # Tautological-proxy KILL on target per F#666
            results["verdict"] = "KILLED"
            results["all_pass"] = False
            results["blockers"].append("F#666 tautological-proxy KILL: K#1965 PASS but K#1966 FAIL — adapter compresses length but breaks task quality")
        elif k1965_status == "fail" and k1966_status == "pass":
            results["verdict"] = "PARTIALLY_SUPPORTED"
            results["all_pass"] = False
            results["blockers"].append("F#666 finding-about-proxy: K#1965 FAIL + K#1966 PASS — deterministic length proxy did not capture concise behavior")
        else:
            # untested (KC harness errored)
            results["verdict"] = "PROVISIONAL"
            results["all_pass"] = False
            results["blockers"].append(f"verdict-fallback: K#1965={k1965_status} K#1966={k1966_status}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    print("\n" + "=" * 60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Verdict: {results['verdict']}", flush=True)
    print(f"KCs: {json.dumps(results['kc'])}", flush=True)
    if results.get("phase_c_k1965") and "reduction_pct" in results["phase_c_k1965"]:
        print(f"K#1965 reduction={results['phase_c_k1965']['reduction_pct']}%", flush=True)
    if results.get("phase_d_k1966") and "drop_pp" in results["phase_d_k1966"]:
        print(f"K#1966 drop={results['phase_d_k1966']['drop_pp']}pp", flush=True)
    print(f"Blockers ({len(results['blockers'])}):", flush=True)
    for b in results["blockers"]:
        print(f"  - {b}", flush=True)
    print(f"Total time: {results['total_time_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
