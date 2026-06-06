#!/usr/bin/env python3
"""
exp_hedgehog_behavior_adapter_politeness_impl
==============================================

IMPL of Hedgehog per-layer cos-sim distillation for politeness adapter.

Inherits MATH.md from parent exp_hedgehog_behavior_adapter_politeness.

Pre-registered KCs (canonical DB text — do not edit):
  K#1821 K1 structural: mean per-layer cos > 0.85 on held-out neutral prompts
  K#1822 K2 target:     auto-judge politeness Δ ≥ +20pp (paired w/ K1 per F#666)
  K#1823 K3 target:     MMLU and HumanEval each drop < 3pp vs base
  K#1824 K4 ablation:   teacher w/ NEUTRAL_SYSTEM_PROMPT regresses K2 by ≥ 10pp

Skills invoked: /mlx-dev (mx.eval discipline, lazy eval, nn.value_and_grad,
                          mx.clear_cache between phases).

Phase plan:
  Phase 0 — neutral-prompt curation (UltraChat HF dataset for full; smoke uses
            embedded list). Filter politeness-marker regex.
  Phase A — teacher capture: base + POLITE_SYSTEM_PROMPT, scale=0 (no LoRA).
  Phase B — student train: base + LoRA, scale=LORA_SCALE, neutral prompt.
            Per-layer cos-sim loss on attention output (o_proj output).
  Phase C — K1 cos-sim eval; K2 politeness judge.
            K2 falls back to deterministic-heuristic judge if no
            ANTHROPIC_API_KEY (PROVISIONAL caveat documented in PAPER.md).
  Phase D — K3 MMLU + HumanEval (deferred to follow-on iteration; flagged).
  Phase E — K4 ablation retrain w/ NEUTRAL teacher (deferred to follow-on).

For SMOKE_TEST=1: tiny dataset, 50 steps, K1+K2-heuristic only.
For full run:    UltraChat N=1000, 500 steps, K1+K2(API)+K3+K4.
                 Submit via pueue; budget ~3-5h on M5 Pro 48GB.
"""
from __future__ import annotations

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

# Honest memory cap for 48GB M5 Pro: leave 8GB headroom for system + judge API.
mx.set_memory_limit(mx.metal.device_info()["memory_size"] - 8 * 1024**3) \
    if hasattr(mx, "metal") and hasattr(mx.metal, "device_info") \
    else mx.set_memory_limit(40 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_TRAIN = 32 if IS_SMOKE else 1000
N_HELDOUT = 8 if IS_SMOKE else 100
N_JUDGE = 8 if IS_SMOKE else 100
N_STEPS = 30 if IS_SMOKE else 500
SEED = 42

ADAPTER_RANK = 8
ADAPTER_TARGETS = ("v_proj", "o_proj")  # F#627
LORA_SCALE = 6.0                        # ≤ 8 per F#328/F#330
BATCH_SIZE = 1                          # MLX 4-bit Gemma 4 E4B fits at bs=1
SEQLEN = 256 if IS_SMOKE else 512
LR = 1e-4
WEIGHT_DECAY = 0.01

POLITE_SYSTEM_PROMPT = (
    "You are a deeply respectful, warm, and patient assistant. "
    "Phrase every response with courtesy, acknowledge the user's effort, "
    "and avoid any language that could feel dismissive or abrupt."
)
NEUTRAL_SYSTEM_PROMPT = ""  # K4 ablation replaces teacher prompt with this

POLITENESS_MARKERS = re.compile(
    r"\b(please|thank you|thanks|sorry|apologi[sz]e|kindly|grateful|"
    r"appreciate|respectfully|would you|could you|may i)\b",
    re.IGNORECASE,
)

# Smoke neutral prompts — short factual instructions, no politeness or rudeness cues.
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

def prepare_neutral_prompts(n_train: int, n_heldout: int) -> dict:
    """Return {train: [prompts], heldout: [prompts]}.

    SMOKE: uses embedded SMOKE_NEUTRAL_PROMPTS (40 prompts).
    Full:  loads HuggingFaceH4/ultrachat_200k and filters politeness markers.
    """
    if IS_SMOKE or n_train + n_heldout <= len(SMOKE_NEUTRAL_PROMPTS):
        all_prompts = list(SMOKE_NEUTRAL_PROMPTS)
        # Drop politeness markers (defense in depth)
        all_prompts = [p for p in all_prompts if not POLITENESS_MARKERS.search(p)]
        train = all_prompts[:n_train]
        heldout = all_prompts[n_train:n_train + n_heldout]
        return {"train": train, "heldout": heldout, "source": "embedded_smoke"}

    # Full run path — UltraChat
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("datasets package required for full Phase 0; install via uv add datasets")

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
        if POLITENESS_MARKERS.search(text):
            continue
        collected.append(text)
    if len(collected) < n_train + n_heldout:
        raise RuntimeError(f"UltraChat yielded only {len(collected)} neutral prompts; need {n_train+n_heldout}")
    return {
        "train": collected[:n_train],
        "heldout": collected[n_train:n_train + n_heldout],
        "source": "ultrachat_filtered",
    }


# ──────────────────────────────────────────────
# Phase A/B — Hedgehog per-layer cos-sim distillation
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
    """Replace each layer.self_attn.o_proj with AttnOutCapture wrapper.
    Caller is responsible for clearing `store` before each forward pass."""
    layers = model.language_model.layers
    for i, layer in enumerate(layers):
        layer.self_attn.o_proj = AttnOutCapture(layer.self_attn.o_proj, store, i)
    return len(layers)


def encode_prompt(tokenizer, system_prompt: str, user: str) -> mx.array:
    """Apply chat template and return token ids as 1D mx.array (no batch dim)."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user})
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    return mx.array(ids, dtype=mx.int32)


def per_layer_cossim_loss(teacher_attns, student_attns, t_offset, s_offset, x_len):
    """Mean over layers of (1 - cos(t[t_offset:t_offset+x_len], s[s_offset:s_offset+x_len])).
    Cos is taken per token over the hidden axis, then mean-pooled across tokens.
    Teacher tensors are detached (mx.stop_gradient) so no gradient flows through teacher path.
    """
    losses = []
    for layer_idx in sorted(teacher_attns.keys()):
        t = mx.stop_gradient(teacher_attns[layer_idx])  # (1, T_t, H)
        s = student_attns[layer_idx]                    # (1, T_s, H)
        # Slice the x portion
        t_slice = t[:, t_offset:t_offset + x_len, :]
        s_slice = s[:, s_offset:s_offset + x_len, :]
        # Cosine per token: (t·s)/(||t||·||s||) along hidden axis
        eps = 1e-6
        t_norm = mx.sqrt(mx.sum(t_slice * t_slice, axis=-1) + eps)
        s_norm = mx.sqrt(mx.sum(s_slice * s_slice, axis=-1) + eps)
        cos = mx.sum(t_slice * s_slice, axis=-1) / (t_norm * s_norm)  # (1, x_len)
        # Mean over tokens, then layer loss = 1 - mean cos
        layer_loss = 1.0 - mx.mean(cos)
        losses.append(layer_loss)
    return mx.mean(mx.stack(losses))


def train_hedgehog_student(model, tokenizer, train_prompts: list,
                            teacher_system_prompt: str, n_steps: int,
                            results: dict) -> dict:
    """Phase B: Hedgehog cos-sim distillation training.

    Same model holds LoRA: teacher = scale=0 (no perturbation, polite prompt);
    student = scale=LORA_SCALE (neutral prompt). Two forward passes per step;
    cos-sim loss on captured o_proj outputs aligned on the user-message portion.
    """
    print("\n=== Phase B: Hedgehog distillation ===", flush=True)
    log_memory("phase_b_start")

    # Find LoRA modules to set scale toggles
    lora_modules = []
    for layer in model.language_model.layers:
        attn = layer.self_attn
        for name in ADAPTER_TARGETS:
            mod = getattr(attn, name, None)
            if mod is None:
                continue
            # If wrapped with AttnOutCapture, look inside
            if hasattr(mod, "inner"):
                mod = mod.inner
            if hasattr(mod, "lora_a") and hasattr(mod, "lora_b"):
                lora_modules.append(mod)

    def set_lora_scale(scale):
        for m in lora_modules:
            m.scale = scale

    # Hooks store
    teacher_store = {}
    student_store = {}
    install_hooks(model, student_store)
    # The same hooks fill both stores depending on the active dict; we swap by
    # re-pointing _store. Implement re-pointing via a small helper.
    layers = model.language_model.layers
    def repoint(store):
        for layer in layers:
            o_proj = layer.self_attn.o_proj
            if isinstance(o_proj, AttnOutCapture):
                o_proj._store = store

    optimizer = optim.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)

    # Trainable parameters: only LoRA a/b
    def loss_fn(model_, t_ids, s_ids, t_user_offset, s_user_offset, x_len):
        # Teacher pass — scale=0
        teacher_store.clear()
        repoint(teacher_store)
        set_lora_scale(0.0)
        _ = model_(t_ids[None, :])  # batch dim
        # Student pass — scale=LORA_SCALE
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
        # Build teacher / student token sequences
        t_ids = encode_prompt(tokenizer, teacher_system_prompt, prompt)
        s_ids = encode_prompt(tokenizer, NEUTRAL_SYSTEM_PROMPT, prompt)
        # Truncate to seqlen
        if t_ids.shape[0] > SEQLEN:
            t_ids = t_ids[:SEQLEN]
        if s_ids.shape[0] > SEQLEN:
            s_ids = s_ids[:SEQLEN]
        # User-message offsets: align the user content tail.
        # Both sequences end with the same user text + generation marker.
        # Compare the trailing min(T_t, T_s) tokens (after their respective system prefixes).
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

    set_lora_scale(LORA_SCALE)  # leave model in student configuration

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
# Phase C — K1 cos-sim eval (held-out) + K2 politeness judge
# ──────────────────────────────────────────────

def measure_k1_structural_cos(model, tokenizer, heldout_prompts: list) -> dict:
    """K1: mean per-layer cos(teacher_attn, student_attn) on held-out prompts.
    Same model with LoRA scale toggle (scale=0 for teacher, LORA_SCALE for student).
    """
    print("\n=== Phase C/K1: structural cos-sim (held-out) ===", flush=True)
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
    """Deterministic 0–100 politeness heuristic for smoke-mode K2 fallback.
    NOT a substitute for a Claude/GPT-4 paired judge; documented as PROVISIONAL
    in PAPER.md when used.
    """
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


def generate_text(model, tokenizer, system_prompt: str, user: str,
                   max_tokens: int = 96) -> str:
    from mlx_lm.generate import generate
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user})
    prompt_str = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    out = generate(model, tokenizer, prompt=prompt_str, max_tokens=max_tokens, verbose=False)
    return out


def measure_k2_politeness_judge(model, tokenizer, judge_prompts: list,
                                  layers_with_lora) -> dict:
    """K2: politeness Δ = score(student) - score(base) on neutral prompts.

    Uses Anthropic Claude API if ANTHROPIC_API_KEY env var present;
    falls back to deterministic heuristic_politeness_score otherwise.
    Score is on a 0-100 scale; we report Δ as percentage points.
    """
    print("\n=== Phase C/K2: politeness judge ===", flush=True)
    use_api = bool(os.environ.get("ANTHROPIC_API_KEY")) and not IS_SMOKE
    judge_kind = "claude_3_7_sonnet" if use_api else "heuristic_smoke"
    print(f"  judge: {judge_kind}", flush=True)

    # Toggle LoRA scale for base vs student generations.
    def set_scale(s):
        for m in layers_with_lora:
            m.scale = s

    base_scores = []
    student_scores = []
    for prompt in judge_prompts:
        # Base generation (no adapter)
        set_scale(0.0)
        base_text = generate_text(model, tokenizer, NEUTRAL_SYSTEM_PROMPT, prompt)
        # Student generation (adapter on)
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
    }


def _claude_politeness(prompt: str, completion: str) -> float:
    """Score a completion for politeness on 0-100 scale via Claude. Disabled in smoke."""
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

    print(f"Hedgehog politeness adapter IMPL (Gemma 4 E4B 4-bit)", flush=True)
    print(f"SMOKE_TEST={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_HELDOUT={N_HELDOUT}, "
          f"N_JUDGE={N_JUDGE}, N_STEPS={N_STEPS}", flush=True)
    print(f"mlx-lm version: {mlx_lm_version}", flush=True)

    adapters_dir = EXPERIMENT_DIR / "adapters"
    adapters_dir.mkdir(exist_ok=True)
    student_adapter = adapters_dir / "hedgehog_polite_r8"

    results = {
        "is_smoke": IS_SMOKE,
        "n_train": N_TRAIN, "n_heldout": N_HELDOUT, "n_judge": N_JUDGE, "n_steps": N_STEPS,
        "model_id": MODEL_ID, "mlx_lm_version": mlx_lm_version,
        "adapter_rank": ADAPTER_RANK, "adapter_targets": list(ADAPTER_TARGETS),
        "lora_scale": LORA_SCALE, "seqlen": SEQLEN, "lr": LR,
        "phase_0_dataset": None, "phase_b_student_train": None,
        "phase_c_k1_k2": {}, "phase_d_k3": None, "phase_e_k4_ablation": None,
        "kc": {
            "K1821_per_layer_cos_gt_0_85": "untested",
            "K1822_politeness_judge_delta_ge_20pp": "untested",
            "K1823a_mmlu_drop_lt_3pp": "untested",
            "K1823b_humaneval_drop_lt_3pp": "untested",
            "K1824_ablation_regression_ge_10pp": "untested",
        },
        "verdict": "PROVISIONAL", "all_pass": False, "blockers": [],
    }

    # ── Phase 0 ───────────────────────────────────────────
    print("\n=== Phase 0: Neutral prompt curation ===", flush=True)
    try:
        ds = prepare_neutral_prompts(N_TRAIN, N_HELDOUT)
        results["phase_0_dataset"] = {
            "source": ds["source"], "n_train": len(ds["train"]), "n_heldout": len(ds["heldout"])
        }
        print(f"  source={ds['source']} train={len(ds['train'])} heldout={len(ds['heldout'])}", flush=True)
    except Exception as exc:
        results["blockers"].append(f"Phase 0 failed: {exc}")
        results["phase_0_dataset"] = {"error": str(exc)}
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return
    train_prompts = ds["train"]
    heldout_prompts = ds["heldout"]

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
    # mlx_lm.tuner.utils.linear_to_lora_layers expects model.model.layers shape;
    # for Gemma 4 the equivalent root is model.language_model. We patch it on
    # the helper directly via a duck-typed call: walk and replace.
    try:
        # Try mlx_lm helper first — it walks model.model.layers
        # Gemma 4 model exposes .layers at top via model.layers (a list of language_model.layers).
        # mlx_lm v0.31 walks model.model.layers — we route via a tiny shim.
        class _Shim:
            def __init__(self, layers): self.layers = layers
        shim_root = type("ShimRoot", (), {"model": _Shim(model.language_model.layers)})()
        linear_to_lora_layers(shim_root, n_layers, lora_cfg)
    except Exception as exc:
        results["blockers"].append(f"linear_to_lora_layers failed: {exc!r}")
        # Fallback: manual replacement
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

    # Freeze everything, then unfreeze LoRA params.
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
        import traceback
        results["phase_b_student_train"] = {
            "error": str(exc), "tb": traceback.format_exc()[-2000:]
        }
        results["blockers"].append(f"Phase B failed: {exc}")
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return

    # ── Phase C K1 ────────────────────────────────────────
    try:
        k1 = measure_k1_structural_cos(model, tokenizer, heldout_prompts)
        results["phase_c_k1_k2"]["K1"] = k1
        if k1["mean_per_layer_cos"] is not None:
            results["kc"]["K1821_per_layer_cos_gt_0_85"] = (
                "pass" if k1["mean_per_layer_cos"] > 0.85 else "fail"
            )
    except Exception as exc:
        results["blockers"].append(f"K1 failed: {exc}")

    # Collect lora modules for K2 toggle
    lora_modules = []
    for layer in model.language_model.layers:
        for name in ADAPTER_TARGETS:
            mod = getattr(layer.self_attn, name, None)
            if mod is None:
                continue
            inner = mod.inner if hasattr(mod, "inner") else mod
            if hasattr(inner, "lora_a"):
                lora_modules.append(inner)

    # ── Phase C K2 ────────────────────────────────────────
    try:
        k2 = measure_k2_politeness_judge(model, tokenizer, heldout_prompts[:N_JUDGE], lora_modules)
        results["phase_c_k1_k2"]["K2"] = k2
        if k2["delta_pp"] is not None:
            # Only mark pass/fail under non-heuristic judge.
            if k2["judge"].startswith("claude") or k2["judge"].startswith("gpt"):
                results["kc"]["K1822_politeness_judge_delta_ge_20pp"] = (
                    "pass" if k2["delta_pp"] >= 20.0 else "fail"
                )
            else:
                results["kc"]["K1822_politeness_judge_delta_ge_20pp"] = "heuristic_only"
    except Exception as exc:
        results["blockers"].append(f"K2 failed: {exc}")

    # ── Phase D K3 ────────────────────────────────────────
    results["phase_d_k3"] = {
        "deferred": True,
        "reason": "MMLU subset + HumanEval pass@1 require dedicated harness; "
                  "scheduled for follow-on _full iteration once K1+K2 land in pueue.",
    }
    results["blockers"].append("K3 deferred to follow-on full-iteration (MMLU/HumanEval harness)")

    # ── Phase E K4 ────────────────────────────────────────
    results["phase_e_k4_ablation"] = {
        "deferred": True,
        "reason": "K4 requires full retrain with NEUTRAL teacher prompt; "
                  "scheduled for follow-on _full iteration after K1+K2 stabilize.",
    }
    results["blockers"].append("K4 ablation deferred to follow-on full-iteration")

    # ── Persist adapter ───────────────────────────────────
    try:
        student_adapter.mkdir(exist_ok=True)
        # Save trainable params only
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

    # Verdict — never SUPPORTED in this iteration: K2 is heuristic, K3+K4 deferred.
    results["verdict"] = "PROVISIONAL"
    results["all_pass"] = False

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    print("\n" + "=" * 60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Verdict: {results['verdict']}", flush=True)
    print(f"KCs: {json.dumps(results['kc'])}", flush=True)
    print(f"Blockers ({len(results['blockers'])}):", flush=True)
    for b in results["blockers"]:
        print(f"  - {b}", flush=True)
    print(f"Total time: {results['total_time_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
