#!/usr/bin/env python3
"""
P3.C2: Few-Shot Style Injection — In-Context Learning Breaks Rank Bottleneck.

Tests whether 2-3 in-context examples raise style compliance from 60% (LoRA ceiling,
Findings #467, #468) to ≥70% with zero training cost.

Reuses P3.B5 artifacts (no training required):
  - domain_fused_base: FP16 math-fused base model
  - new_personal_adapter: personal adapter trained on domain_fused_base

Phases:
  0. Verify B5 artifacts exist (domain_fused_base, new_personal_adapter)
  1. Zero-shot baseline (N_STYLE queries, no examples in prompt)
  2. Few-shot style injection (N_STYLE queries, k=3 examples in system prompt)
  3. k-scaling check (k=1,2,3,5 on N_SCALE diverse queries)

Kill criteria:
  K1199: style_compliance_fewshot >= 70% (above LoRA ceiling 60%)
  K1200: zero_training_cost = True (no adapter trained)
  K1201: context_overhead_ratio <= 15.0 (token overhead vs zero-shot)

References:
  - Brown et al. 2020 (arxiv 2005.14165): GPT-3 few-shot learning
  - Finding #468: rank-4 LoRA ceiling at 60%
"""

import gc
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
B5_DIR = EXPERIMENT_DIR.parent / "exp_p3_b5_domain_conditional_retrain"
DOMAIN_FUSED_DIR = B5_DIR / "domain_fused_base"
PERSONAL_ADAPTER_DIR = B5_DIR / "new_personal_adapter"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_STYLE = 5 if IS_SMOKE else 15     # style compliance test
N_SCALE = 3 if IS_SMOKE else 5      # k-scaling test
SEED = 42

PREFERENCE_MARKER = "Hope that helps, friend!"

# ──────────────────────────────────────────────────────────────────────
# Few-shot examples for style injection
# ──────────────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    {
        "q": "What is gravity?",
        "a": "Gravity is a fundamental force that attracts objects with mass toward one another. On Earth, it gives weight to objects and causes them to fall when dropped. It's described by Newton's law of universal gravitation and Einstein's general relativity. Hope that helps, friend!"
    },
    {
        "q": "How do computers store information?",
        "a": "Computers store information as binary data — sequences of 0s and 1s. Hard drives use magnetic or flash storage, while RAM uses electrical charges. Each character or image is encoded as patterns of these binary digits. Hope that helps, friend!"
    },
    {
        "q": "What causes seasons?",
        "a": "Seasons are caused by Earth's axial tilt of about 23.5 degrees as it orbits the Sun. When the Northern Hemisphere tilts toward the Sun, it gets more direct sunlight and experiences summer. Six months later, it tilts away, giving us winter. Hope that helps, friend!"
    },
    {
        "q": "What is democracy?",
        "a": "Democracy is a system of government where citizens have the power to make decisions, either directly or through elected representatives. It typically includes free elections, protection of individual rights, and the rule of law. Hope that helps, friend!"
    },
    {
        "q": "Why is the sky blue?",
        "a": "The sky appears blue due to Rayleigh scattering. Sunlight contains all colors, but when it passes through Earth's atmosphere, the shorter blue wavelengths scatter more than red ones. Our eyes see this scattered blue light coming from all directions. Hope that helps, friend!"
    },
]

# Style test prompts (same as P3.C0/C1 for comparability)
STYLE_PROMPTS = [
    "What is machine learning?",
    "Explain quantum entanglement in simple terms.",
    "How does photosynthesis work?",
    "What is the difference between a virus and a bacterium?",
    "Can you explain the concept of recursion in programming?",
    "What causes rainbows?",
    "How does the stock market work?",
    "What is the meaning of life according to philosophy?",
    "Explain the theory of relativity.",
    "How do vaccines work?",
    "What is the difference between weather and climate?",
    "Explain how neural networks learn.",
    "What is blockchain technology?",
    "How does the immune system fight infections?",
    "What is the significance of the speed of light?",
]

# Diverse prompts for k-scaling (different domains than training examples)
SCALING_PROMPTS = [
    "What is inflation in economics?",
    "How do plants grow from seeds?",
    "What is the difference between art and design?",
    "Explain how GPS works.",
    "What causes earthquakes?",
]

# ──────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ──────────────────────────────────────────────────────────────────────

_cached_model = None
_cached_tokenizer = None


def load_model():
    """Load domain_fused_base + personal adapter (B5 artifacts). Cached."""
    global _cached_model, _cached_tokenizer
    if _cached_model is not None:
        return _cached_model, _cached_tokenizer
    from mlx_lm import load as mlx_load
    print(f"  Loading model from {DOMAIN_FUSED_DIR}...")
    print(f"  Adapter: {PERSONAL_ADAPTER_DIR}")
    model, tokenizer = mlx_load(
        str(DOMAIN_FUSED_DIR),
        adapter_path=str(PERSONAL_ADAPTER_DIR)
    )
    mx.eval(model.parameters())
    _cached_model = model
    _cached_tokenizer = tokenizer
    print("  Model loaded.")
    return model, tokenizer


def apply_chat_template(tokenizer, messages: list) -> str:
    """Apply chat template for Gemma 4 multi-turn format."""
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        pass
    # Fallback: Gemma 4 format
    result = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        result += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
    result += "<start_of_turn>model\n"
    return result


def count_tokens(tokenizer, text: str) -> int:
    """Count tokens in a string."""
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return len(text.split())  # fallback: word count


def generate_response(prompt_text: str, model, tokenizer, max_tokens: int = 256) -> str:
    """Generate response for a formatted prompt."""
    from mlx_lm import generate as mlx_generate
    try:
        response = mlx_generate(
            model, tokenizer, prompt=prompt_text, max_tokens=max_tokens, verbose=False
        )
        return response
    except Exception as e:
        print(f"  [generate error: {e}]")
        return ""


def check_style(response: str) -> bool:
    """Returns True if response contains the PREFERENCE_MARKER."""
    return PREFERENCE_MARKER.lower() in response.lower()


# ──────────────────────────────────────────────────────────────────────
# Phase 0: Verify B5 artifacts
# ──────────────────────────────────────────────────────────────────────

def phase0_verify_artifacts() -> dict:
    print("\n== Phase 0: Verify B5 artifacts ==")
    ok = True
    if not DOMAIN_FUSED_DIR.exists():
        print(f"  ERROR: domain_fused_base not found at {DOMAIN_FUSED_DIR}")
        ok = False
    else:
        print(f"  domain_fused_base: OK ({DOMAIN_FUSED_DIR})")
    if not PERSONAL_ADAPTER_DIR.exists():
        print(f"  ERROR: personal_adapter not found at {PERSONAL_ADAPTER_DIR}")
        ok = False
    else:
        print(f"  personal_adapter: OK ({PERSONAL_ADAPTER_DIR})")
    return {"artifacts_ok": ok}


# ──────────────────────────────────────────────────────────────────────
# Phase 1: Zero-shot baseline
# ──────────────────────────────────────────────────────────────────────

def phase1_zeroshot_baseline(n: int) -> dict:
    print(f"\n== Phase 1: Zero-shot baseline (N={n}) ==")
    model, tokenizer = load_model()
    prompts = STYLE_PROMPTS[:n]
    compliant = 0
    total_tokens = 0

    for i, question in enumerate(prompts):
        messages = [{"role": "user", "content": question}]
        formatted = apply_chat_template(tokenizer, messages)
        total_tokens += count_tokens(tokenizer, formatted)
        response = generate_response(formatted, model, tokenizer)
        ok = check_style(response)
        if ok:
            compliant += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{i+1}/{n}] {status}: {question[:50]}...")
        if not ok and i < 3:
            print(f"    Response tail: ...{response[-80:]!r}")

    rate = compliant / n
    avg_tokens = total_tokens / n
    print(f"\n  Zero-shot style compliance: {compliant}/{n} = {rate:.1%}")
    print(f"  Avg prompt tokens: {avg_tokens:.0f}")
    return {
        "zeroshot_compliant": compliant,
        "zeroshot_n": n,
        "zeroshot_rate": rate,
        "zeroshot_avg_tokens": avg_tokens,
    }


# ──────────────────────────────────────────────────────────────────────
# Phase 2: Few-shot style injection (k=3 examples)
# ──────────────────────────────────────────────────────────────────────

def build_fewshot_messages(question: str, k: int) -> list:
    """Build few-shot message list with k examples before the query."""
    messages = []
    examples = FEW_SHOT_EXAMPLES[:k]
    for ex in examples:
        messages.append({"role": "user", "content": ex["q"]})
        messages.append({"role": "assistant", "content": ex["a"]})
    messages.append({"role": "user", "content": question})
    return messages


def phase2_fewshot_injection(n: int, k: int = 3) -> dict:
    print(f"\n== Phase 2: Few-shot style injection (N={n}, k={k}) ==")
    model, tokenizer = load_model()
    prompts = STYLE_PROMPTS[:n]
    compliant = 0
    total_tokens_fs = 0

    for i, question in enumerate(prompts):
        messages = build_fewshot_messages(question, k=k)
        formatted = apply_chat_template(tokenizer, messages)
        total_tokens_fs += count_tokens(tokenizer, formatted)
        response = generate_response(formatted, model, tokenizer)
        ok = check_style(response)
        if ok:
            compliant += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{i+1}/{n}] {status}: {question[:50]}...")
        if not ok and i < 3:
            print(f"    Response tail: ...{response[-80:]!r}")

    rate = compliant / n
    avg_tokens_fs = total_tokens_fs / n
    print(f"\n  Few-shot (k={k}) style compliance: {compliant}/{n} = {rate:.1%}")
    print(f"  Avg few-shot prompt tokens: {avg_tokens_fs:.0f}")
    return {
        "fewshot_compliant": compliant,
        "fewshot_n": n,
        "fewshot_rate": rate,
        "fewshot_avg_tokens": avg_tokens_fs,
        "k": k,
    }


# ──────────────────────────────────────────────────────────────────────
# Phase 3: k-scaling check (k=1,2,3,5)
# ──────────────────────────────────────────────────────────────────────

def phase3_k_scaling(n: int) -> dict:
    print(f"\n== Phase 3: k-scaling check (N={n} per k) ==")
    model, tokenizer = load_model()
    prompts = SCALING_PROMPTS[:n]
    results = {}
    k_values = [0, 1, 2, 3, 5] if not IS_SMOKE else [0, 1, 3]

    for k in k_values:
        compliant = 0
        for question in prompts:
            if k == 0:
                messages = [{"role": "user", "content": question}]
            else:
                messages = build_fewshot_messages(question, k=k)
            formatted = apply_chat_template(tokenizer, messages)
            response = generate_response(formatted, model, tokenizer)
            if check_style(response):
                compliant += 1
        rate = compliant / n
        results[f"k{k}"] = rate
        print(f"  k={k}: {compliant}/{n} = {rate:.1%}")

    return {"k_scaling": results}


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 60)
    print("P3.C2: Few-Shot Style Injection")
    print(f"IS_SMOKE={IS_SMOKE}, N_STYLE={N_STYLE}, N_SCALE={N_SCALE}")
    print("=" * 60)

    results = {"is_smoke": IS_SMOKE, "timestamp": time.time()}

    # Phase 0: Verify artifacts
    p0 = phase0_verify_artifacts()
    results.update(p0)
    if not p0["artifacts_ok"]:
        print("\nERROR: Missing B5 artifacts. Run P3.B5 first.")
        results["error"] = "missing_artifacts"
        results["elapsed_s"] = time.time() - t_start
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return

    # Phase 1: Zero-shot baseline
    p1 = phase1_zeroshot_baseline(N_STYLE)
    results.update(p1)
    zeroshot_tokens = p1["zeroshot_avg_tokens"]

    # Phase 2: Few-shot injection (k=3)
    p2 = phase2_fewshot_injection(N_STYLE, k=3)
    results.update(p2)
    fewshot_tokens = p2["fewshot_avg_tokens"]

    # Compute overhead ratio
    overhead_ratio = fewshot_tokens / max(zeroshot_tokens, 1.0)
    results["context_overhead_ratio"] = overhead_ratio
    print(f"\n  Context overhead ratio: {overhead_ratio:.1f}x")

    # Phase 3: k-scaling
    p3 = phase3_k_scaling(N_SCALE)
    results.update(p3)

    # Kill criteria evaluation
    elapsed = time.time() - t_start
    results["elapsed_s"] = elapsed

    fewshot_rate = p2["fewshot_rate"]
    k1199_pass = fewshot_rate >= 0.70
    k1200_pass = True  # by construction — no training
    k1201_pass = overhead_ratio <= 15.0

    results["k1199_pass"] = k1199_pass
    results["k1200_pass"] = k1200_pass
    results["k1201_pass"] = k1201_pass

    print("\n" + "=" * 60)
    print("KILL CRITERIA SUMMARY")
    print("=" * 60)
    print(f"  K1199 style_fewshot >= 70%: {fewshot_rate:.1%} → {'PASS' if k1199_pass else 'FAIL'}")
    print(f"  K1200 zero_training_cost:   True → PASS")
    print(f"  K1201 overhead_ratio <= 15: {overhead_ratio:.1f}x → {'PASS' if k1201_pass else 'FAIL'}")
    print(f"\n  Total elapsed: {elapsed:.1f}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
