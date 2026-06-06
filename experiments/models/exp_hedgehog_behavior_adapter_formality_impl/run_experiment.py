#!/usr/bin/env python3
"""
exp_hedgehog_behavior_adapter_formality_impl
============================================

IMPL of Hedgehog per-layer cos-sim distillation for FORMALITY adapter.
Inherits MATH.md design from parent exp_hedgehog_behavior_adapter_formality
(F#724 PROVISIONAL design-lock). Adapted from politeness_impl template with:
  - π_Formal teacher prompt (academic register) replacing POLITE_SYSTEM_PROMPT
  - formality heuristic scorer (lexical register markers + hedging + inverse contractions)
  - sizing-bug fix: SMOKE_NEUTRAL_FORMALITY_PROMPTS sized to fit N_TRAIN + N_HELDOUT + N_JUDGE

Pre-registered KCs (canonical DB text — do not edit):
  K#1963 K1 target: formality auto-judge < +10pp vs base (FAIL if Δ<10)
  K#1964 K2 target: |MMLU-100 accuracy delta| > 2pp two-sided (FAIL if drift)

Skills invoked: /mlx-dev (mx.eval, mx.clear_cache, nn.value_and_grad) + /fast-mlx.

Phase plan:
  Phase 0 — neutral-prompt curation (embedded smoke list — sized to fit).
  Phase A — teacher capture: base + π_Formal, scale=0 (no LoRA).
  Phase B — student train: base + LoRA, scale=LORA_SCALE, neutral prompt.
            Per-layer cos-sim loss on attention output (o_proj output).
  Phase C — K#1963 formality judge (heuristic in smoke; Claude in full).
  Phase D — K#1964 MMLU subset (deferred to _full).
  Phase E — F#666 verdict matrix.

For SMOKE_TEST=1: 40-prompt embedded set, 30 steps, K#1963 heuristic_only.
For full run:    UltraChat N=200 train + 50 held-out + 100 MMLU,
                 800 steps, K#1963 Claude-judge + K#1964 MMLU.
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
SEED = 42

ADAPTER_RANK = 8
ADAPTER_TARGETS = ("v_proj", "o_proj")  # F#627
LORA_SCALE = 6.0                        # ≤ 8 per F#328/F#330
BATCH_SIZE = 1
SEQLEN = 256 if IS_SMOKE else 512
LR = 1e-4
WEIGHT_DECAY = 0.01
GEN_MAX_TOKENS = 256  # raised from 192 (K2-collapse antipattern mitigation, MATH §8 A4)

FORMAL_SYSTEM_PROMPT = (
    "You are a formal-register academic assistant. Reply in formal English "
    "with academic tone, no contractions, full sentences with subordinate "
    "clauses where appropriate. Use precise terminology and qualified, "
    "evidence-based phrasing throughout."
)
NEUTRAL_SYSTEM_PROMPT = ""

# Formality marker regex. Used by heuristic scorer; also used to filter
# the neutral-prompt set so the prompts themselves don't bias the register.
FORMAL_MARKERS = re.compile(
    r"\b(furthermore|moreover|however|therefore|consequently|notwithstanding|"
    r"henceforth|whereas|hereinafter|aforementioned|aforesaid|inasmuch|"
    r"insofar|subsequent|preceding|comprising|constitutes|exemplifies|"
    r"demonstrates|illustrates|elucidates|necessitates|substantiates|"
    r"corroborates|investigation|methodology|hypothesis|empirical|"
    r"theoretical|fundamental|comprehensive|systematic)\b",
    re.IGNORECASE,
)
INFORMAL_MARKERS = re.compile(
    r"\b(gonna|wanna|gotta|kinda|sorta|yeah|nope|yep|nah|cool|awesome|"
    r"super|totally|basically|like\s|stuff|things|guys|y'all|hey|hi|"
    r"yo|huh|um|uh|whatever|anyway|anyhow|so\s+yeah|so\s+anyway)\b",
    re.IGNORECASE,
)
CONTRACTION_RE = re.compile(
    r"\b(it's|that's|what's|here's|there's|let's|don't|doesn't|didn't|"
    r"won't|wouldn't|shouldn't|couldn't|can't|isn't|aren't|wasn't|weren't|"
    r"hasn't|haven't|hadn't|i'm|you're|we're|they're|i've|you've|we've|"
    r"they've|i'll|you'll|we'll|they'll|i'd|you'd|we'd|they'd)\b",
    re.IGNORECASE,
)
HEDGE_MARKERS = re.compile(
    r"\b(arguably|presumably|ostensibly|conceivably|plausibly|tentatively|"
    r"possibly|perhaps|potentially|generally|typically|commonly|frequently|"
    r"largely|primarily|principally|notably|particularly|specifically|"
    r"is\s+often|may\s+be|might\s+be|can\s+be|tend\s+to|tends\s+to)\b",
    re.IGNORECASE,
)

# 40 neutral knowledge questions — sized exactly to N_TRAIN(24) + N_HELDOUT(8) + N_JUDGE(8).
SMOKE_NEUTRAL_FORMALITY_PROMPTS = [
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

# Sizing assertion (analyst antipattern from refactor_impl iter ~62)
assert N_TRAIN + N_HELDOUT + N_JUDGE <= len(SMOKE_NEUTRAL_FORMALITY_PROMPTS), (
    f"sizing-bug: need {N_TRAIN + N_HELDOUT + N_JUDGE} prompts but smoke list "
    f"has {len(SMOKE_NEUTRAL_FORMALITY_PROMPTS)}"
)


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

    SMOKE: use embedded SMOKE_NEUTRAL_FORMALITY_PROMPTS (40 prompts).
    Full: load HuggingFaceH4/ultrachat_200k and filter both formal+informal markers.
    """
    if IS_SMOKE or n_train + n_heldout + n_judge <= len(SMOKE_NEUTRAL_FORMALITY_PROMPTS):
        all_prompts = list(SMOKE_NEUTRAL_FORMALITY_PROMPTS)
        # Drop any prompt with explicit formal/informal cues (defense in depth)
        all_prompts = [
            p for p in all_prompts
            if not FORMAL_MARKERS.search(p) and not INFORMAL_MARKERS.search(p)
        ]
        train = all_prompts[:n_train]
        heldout = all_prompts[n_train:n_train + n_heldout]
        judge = all_prompts[n_train + n_heldout:n_train + n_heldout + n_judge]
        return {"train": train, "heldout": heldout, "judge": judge, "source": "embedded_smoke"}

    # Full run path — UltraChat
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
        if FORMAL_MARKERS.search(text) or INFORMAL_MARKERS.search(text):
            continue
        collected.append(text)
    if len(collected) < n_train + n_heldout + n_judge:
        raise RuntimeError(
            f"UltraChat yielded only {len(collected)} register-neutral prompts; "
            f"need {n_train + n_heldout + n_judge}"
        )
    return {
        "train": collected[:n_train],
        "heldout": collected[n_train:n_train + n_heldout],
        "judge": collected[n_train + n_heldout:n_train + n_heldout + n_judge],
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
    print("\n=== Phase B: Hedgehog distillation (formality) ===", flush=True)
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
# Phase C — informal cos-sim self-check (proxy track) + K#1963 formality judge
# ──────────────────────────────────────────────

def measure_cos_sim_self_check(model, tokenizer, heldout_prompts: list) -> dict:
    """Informal proxy track (NOT a KC): mean per-layer cos(teacher_attn, student_attn).
    Same model with LoRA scale toggle. Used to validate the training signal
    converged before judging behavior. Reported in PAPER.md as a sanity-check,
    not a kill criterion (parent MATH §3 says 'informally tracked').
    """
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
        t_ids = encode_prompt(tokenizer, FORMAL_SYSTEM_PROMPT, prompt)
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


def heuristic_formality_score(text: str) -> float:
    """Deterministic 0–100 formality heuristic for smoke-mode K#1963 fallback.
    Components (parent MATH §4): (a) lexical register (formal vocab), (b) hedging,
    (c) inverse contraction-rate, (d) inverse informal markers. NOT a substitute
    for Claude paired judge; documented as PROVISIONAL in PAPER.md.
    """
    if not text:
        return 0.0
    text_l = text.lower()
    n_words = max(1, len(text.split()))
    formal_n = len(FORMAL_MARKERS.findall(text_l))
    informal_n = len(INFORMAL_MARKERS.findall(text_l))
    contractions_n = len(CONTRACTION_RE.findall(text_l))
    hedge_n = len(HEDGE_MARKERS.findall(text_l))
    # Density per 100 words
    density = lambda c: 100.0 * c / n_words
    score = 50.0
    score += 5.0 * min(density(formal_n), 4.0)        # formal vocab boost (cap)
    score += 3.0 * min(density(hedge_n), 4.0)         # hedging boost (cap)
    score -= 6.0 * min(density(informal_n), 4.0)      # informal penalty
    score -= 4.0 * min(density(contractions_n), 4.0)  # contraction penalty
    # Subordinate-clause proxy: comma-density (very rough)
    commas = text.count(",")
    score += 2.0 * min(commas / max(1, n_words / 20), 4.0)
    return max(0.0, min(100.0, score))


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


def measure_k1963_formality_judge(model, tokenizer, judge_prompts: list,
                                    layers_with_lora) -> dict:
    """K#1963: formality Δ = score(student) - score(base) on neutral prompts.
    Uses Claude API if ANTHROPIC_API_KEY env var present (and not smoke);
    falls back to heuristic_formality_score otherwise.
    """
    print("\n=== Phase C/K#1963: formality judge ===", flush=True)
    use_api = bool(os.environ.get("ANTHROPIC_API_KEY")) and not IS_SMOKE
    judge_kind = "claude_3_7_sonnet" if use_api else "heuristic_smoke"
    print(f"  judge: {judge_kind}", flush=True)

    def set_scale(s):
        for m in layers_with_lora:
            m.scale = s

    base_scores = []
    student_scores = []
    base_texts = []
    student_texts = []
    for prompt in judge_prompts:
        set_scale(0.0)
        base_text = generate_text(model, tokenizer, NEUTRAL_SYSTEM_PROMPT, prompt)
        set_scale(LORA_SCALE)
        student_text = generate_text(model, tokenizer, NEUTRAL_SYSTEM_PROMPT, prompt)

        if use_api:
            base_score = _claude_formality(prompt, base_text)
            student_score = _claude_formality(prompt, student_text)
        else:
            base_score = heuristic_formality_score(base_text)
            student_score = heuristic_formality_score(student_text)
        base_scores.append(base_score)
        student_scores.append(student_score)
        base_texts.append(base_text[:240])  # snippet for PAPER.md
        student_texts.append(student_text[:240])

    set_scale(LORA_SCALE)
    if not base_scores:
        return {"delta_pp": None, "judge": judge_kind, "n": 0}

    delta = sum(student_scores) / len(student_scores) - sum(base_scores) / len(base_scores)
    return {
        "delta_pp": round(delta, 2),
        "base_mean": round(sum(base_scores) / len(base_scores), 2),
        "student_mean": round(sum(student_scores) / len(student_scores), 2),
        "base_scores": [round(s, 2) for s in base_scores],
        "student_scores": [round(s, 2) for s in student_scores],
        "judge": judge_kind,
        "n": len(base_scores),
        "sample_base_snippet": base_texts[0] if base_texts else "",
        "sample_student_snippet": student_texts[0] if student_texts else "",
    }


def _claude_formality(prompt: str, completion: str) -> float:
    try:
        from anthropic import Anthropic
    except ImportError:
        return heuristic_formality_score(completion)
    client = Anthropic()
    rubric = (
        "Rate the formality of the assistant reply on a 0-100 integer scale "
        "(0 = very casual/colloquial, 50 = neutral, 100 = highly formal/academic register). "
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

    print(f"Hedgehog FORMALITY adapter IMPL (Gemma 4 E4B 4-bit)", flush=True)
    print(f"SMOKE_TEST={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_HELDOUT={N_HELDOUT}, "
          f"N_JUDGE={N_JUDGE}, N_STEPS={N_STEPS}", flush=True)
    print(f"mlx-lm version: {mlx_lm_version}", flush=True)

    adapters_dir = EXPERIMENT_DIR / "adapters"
    adapters_dir.mkdir(exist_ok=True)
    student_adapter = adapters_dir / "hedgehog_formal_r8"

    results = {
        "is_smoke": IS_SMOKE,
        "n_train": N_TRAIN, "n_heldout": N_HELDOUT, "n_judge": N_JUDGE, "n_steps": N_STEPS,
        "model_id": MODEL_ID, "mlx_lm_version": mlx_lm_version,
        "adapter_rank": ADAPTER_RANK, "adapter_targets": list(ADAPTER_TARGETS),
        "lora_scale": LORA_SCALE, "seqlen": SEQLEN, "lr": LR,
        "gen_max_tokens": GEN_MAX_TOKENS,
        "phase_0_dataset": None, "phase_b_student_train": None,
        "phase_c_proxy_cos": None, "phase_c_k1963": None,
        "phase_d_k1964": None,
        "kc": {
            "K1963_formality_judge_delta_ge_10pp": "untested",
            "K1964_mmlu_drift_le_2pp": "untested",
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
                                          FORMAL_SYSTEM_PROMPT, N_STEPS, results)
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

    # Collect lora modules for K#1963 toggle
    lora_modules = []
    for layer in model.language_model.layers:
        for name in ADAPTER_TARGETS:
            mod = getattr(layer.self_attn, name, None)
            if mod is None:
                continue
            inner = mod.inner if hasattr(mod, "inner") else mod
            if hasattr(inner, "lora_a"):
                lora_modules.append(inner)

    # ── Phase C K#1963 ────────────────────────────────────
    try:
        k1963 = measure_k1963_formality_judge(model, tokenizer, judge_prompts, lora_modules)
        results["phase_c_k1963"] = k1963
        if k1963["delta_pp"] is not None:
            if k1963["judge"].startswith("claude") or k1963["judge"].startswith("gpt"):
                results["kc"]["K1963_formality_judge_delta_ge_10pp"] = (
                    "pass" if k1963["delta_pp"] >= 10.0 else "fail"
                )
            else:
                # Smoke heuristic-only: report PASS_SMOKE if heuristic Δ ≥ 10 pp,
                # else heuristic_only (PROVISIONAL); never a final pass/fail.
                if k1963["delta_pp"] >= 10.0:
                    results["kc"]["K1963_formality_judge_delta_ge_10pp"] = "pass_smoke"
                else:
                    results["kc"]["K1963_formality_judge_delta_ge_10pp"] = "heuristic_only"
    except Exception as exc:
        results["blockers"].append(f"K#1963 failed: {exc}")

    # ── Phase D K#1964 (DEFERRED to _full) ────────────────
    results["phase_d_k1964"] = {
        "deferred": True,
        "reason": "MMLU 100-question subset requires dedicated harness + canonical "
                  "answer scoring; scheduled for follow-on _full iteration once "
                  "K#1963 PASS_SMOKE confirmed and pueue budget secured.",
    }
    results["blockers"].append("K#1964 deferred to follow-on _full iteration (MMLU harness)")

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

    # Verdict — PROVISIONAL by construction (smoke + heuristic K#1963 + deferred K#1964).
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
