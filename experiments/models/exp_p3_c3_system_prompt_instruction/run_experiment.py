#!/usr/bin/env python3
"""
P3.C3: System Prompt Instruction for Style Injection.

Tests whether an explicit system prompt instruction raises style compliance from
60% (LoRA ceiling) to ≥80% with zero training cost. Uses RLHF instruction following
as the mechanism, orthogonal to adapter weight perturbation.

Reuses P3.B5 artifacts (no training required):
  - domain_fused_base: FP16 math-fused base model
  - new_personal_adapter: personal adapter trained on domain_fused_base

Phases:
  0. Verify B5 artifacts exist
  1. Control: zero-shot without system prompt (N_STYLE queries)
  2. System prompt instruction (N_STYLE queries)
  3. Ablation: instruction variants + base model vs adapted (N_ABLATION each)

Kill criteria:
  K1202: style_instruction >= 80% (above LoRA ceiling 60%)
  K1203: zero_training_cost = True
  K1204: context_overhead_ratio <= 2.0

References:
  - Ouyang et al. 2022 (arxiv 2203.02155): InstructGPT instruction following
  - Finding #469: few-shot context-prior conflict degrades to 20%
  - Finding #468: rank-4 LoRA ceiling at 60%
"""

import gc
import json
import os
import time
from pathlib import Path

import mlx.core as mx

EXPERIMENT_DIR = Path(__file__).parent
B5_DIR = EXPERIMENT_DIR.parent / "exp_p3_b5_domain_conditional_retrain"
DOMAIN_FUSED_DIR = B5_DIR / "domain_fused_base"
PERSONAL_ADAPTER_DIR = B5_DIR / "new_personal_adapter"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_STYLE = 5 if IS_SMOKE else 15       # main style test
N_ABLATION = 3 if IS_SMOKE else 5     # ablation per variant
SEED = 42

PREFERENCE_MARKER = "Hope that helps, friend!"

# System prompt variants for ablation
SYSTEM_LONG = (
    "Always end your response with the exact phrase 'Hope that helps, friend!' — "
    "include this precise phrase at the very end of your response, after your explanation. "
    "Do not paraphrase it."
)
SYSTEM_SHORT = "End every response with: Hope that helps, friend!"
SYSTEM_MEDIUM = "Always end your response with 'Hope that helps, friend!'"

# ──────────────────────────────────────────────────────────────────────
# Style test prompts (consistent with P3.C0/C1/C2)
# ──────────────────────────────────────────────────────────────────────

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

ABLATION_PROMPTS = [
    "What is inflation in economics?",
    "How do plants grow from seeds?",
    "What is the difference between art and design?",
    "Explain how GPS works.",
    "What causes earthquakes?",
]

# ──────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ──────────────────────────────────────────────────────────────────────

_cache = {}


def load_model(model_path: str, adapter_path: str = None):
    """Load model with optional adapter. Cached by (model_path, adapter_path)."""
    key = (model_path, adapter_path)
    if key in _cache:
        return _cache[key]
    from mlx_lm import load as mlx_load
    print(f"  Loading {model_path} (adapter={adapter_path or 'none'})...")
    model, tokenizer = mlx_load(model_path, adapter_path=adapter_path)
    mx.eval(model.parameters())
    _cache[key] = (model, tokenizer)
    print("  Loaded.")
    return model, tokenizer


def apply_chat_template(tokenizer, messages: list) -> str:
    """Apply Gemma 4 chat template."""
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        pass
    result = ""
    for msg in messages:
        result += f"<start_of_turn>{msg['role']}\n{msg['content']}<end_of_turn>\n"
    result += "<start_of_turn>model\n"
    return result


def count_tokens(tokenizer, text: str) -> int:
    """Count tokens."""
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return len(text.split())


def generate_response(formatted_prompt: str, model, tokenizer, max_tokens: int = 256) -> str:
    """Generate response."""
    from mlx_lm import generate as mlx_generate
    try:
        return mlx_generate(model, tokenizer, prompt=formatted_prompt,
                           max_tokens=max_tokens, verbose=False)
    except Exception as e:
        print(f"  [generate error: {e}]")
        return ""


def check_style(response: str) -> bool:
    """True if PREFERENCE_MARKER present."""
    return PREFERENCE_MARKER.lower() in response.lower()


def run_style_eval(prompts: list, model, tokenizer, system_prompt: str = None) -> dict:
    """Run style compliance eval on prompts with optional system prompt."""
    compliant = 0
    total_prompt_tokens = 0

    for i, question in enumerate(prompts):
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        else:
            messages = [{"role": "user", "content": question}]

        formatted = apply_chat_template(tokenizer, messages)
        total_prompt_tokens += count_tokens(tokenizer, formatted)
        response = generate_response(formatted, model, tokenizer)
        ok = check_style(response)
        if ok:
            compliant += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{i+1}/{len(prompts)}] {status}: {question[:50]}...")
        if not ok and i < 2:
            print(f"    Tail: ...{response[-80:]!r}")

    rate = compliant / len(prompts)
    avg_tokens = total_prompt_tokens / len(prompts)
    return {"compliant": compliant, "n": len(prompts), "rate": rate, "avg_tokens": avg_tokens}


# ──────────────────────────────────────────────────────────────────────
# Phases
# ──────────────────────────────────────────────────────────────────────

def phase0_verify() -> dict:
    print("\n== Phase 0: Verify B5 artifacts ==")
    ok = DOMAIN_FUSED_DIR.exists() and PERSONAL_ADAPTER_DIR.exists()
    print(f"  domain_fused_base: {'OK' if DOMAIN_FUSED_DIR.exists() else 'MISSING'}")
    print(f"  personal_adapter:  {'OK' if PERSONAL_ADAPTER_DIR.exists() else 'MISSING'}")
    return {"artifacts_ok": ok}


def phase1_control(n: int) -> dict:
    """Phase 1: Zero-shot control without system prompt."""
    print(f"\n== Phase 1: Control (no system prompt, N={n}) ==")
    model, tokenizer = load_model(str(DOMAIN_FUSED_DIR), str(PERSONAL_ADAPTER_DIR))
    res = run_style_eval(STYLE_PROMPTS[:n], model, tokenizer, system_prompt=None)
    print(f"\n  Control compliance: {res['compliant']}/{res['n']} = {res['rate']:.1%}")
    print(f"  Avg prompt tokens: {res['avg_tokens']:.0f}")
    return {"control_rate": res["rate"], "control_n": res["n"],
            "control_compliant": res["compliant"], "control_avg_tokens": res["avg_tokens"]}


def phase2_instruction(n: int) -> dict:
    """Phase 2: System prompt instruction (SYSTEM_LONG)."""
    print(f"\n== Phase 2: System prompt instruction (N={n}) ==")
    print(f"  System: {SYSTEM_LONG[:80]}...")
    model, tokenizer = load_model(str(DOMAIN_FUSED_DIR), str(PERSONAL_ADAPTER_DIR))
    res = run_style_eval(STYLE_PROMPTS[:n], model, tokenizer, system_prompt=SYSTEM_LONG)
    print(f"\n  Instruction compliance: {res['compliant']}/{res['n']} = {res['rate']:.1%}")
    print(f"  Avg prompt tokens (with system): {res['avg_tokens']:.0f}")
    return {"instruction_rate": res["rate"], "instruction_n": res["n"],
            "instruction_compliant": res["compliant"], "instruction_avg_tokens": res["avg_tokens"]}


def phase3_ablation(n: int) -> dict:
    """Phase 3: Ablation over instruction variants."""
    print(f"\n== Phase 3: Ablation (N={n} per variant) ==")
    model, tokenizer = load_model(str(DOMAIN_FUSED_DIR), str(PERSONAL_ADAPTER_DIR))
    ablation = {}

    # Short instruction
    print("  Variant: short instruction")
    res = run_style_eval(ABLATION_PROMPTS[:n], model, tokenizer, system_prompt=SYSTEM_SHORT)
    ablation["short"] = res["rate"]
    print(f"  Short: {res['rate']:.1%}")

    # Medium instruction
    print("  Variant: medium instruction")
    res = run_style_eval(ABLATION_PROMPTS[:n], model, tokenizer, system_prompt=SYSTEM_MEDIUM)
    ablation["medium"] = res["rate"]
    print(f"  Medium: {res['rate']:.1%}")

    # Long instruction (same as phase 2, on ablation prompts)
    print("  Variant: long instruction")
    res = run_style_eval(ABLATION_PROMPTS[:n], model, tokenizer, system_prompt=SYSTEM_LONG)
    ablation["long"] = res["rate"]
    print(f"  Long: {res['rate']:.1%}")

    # No instruction (ablation control)
    print("  Variant: no instruction")
    res = run_style_eval(ABLATION_PROMPTS[:n], model, tokenizer, system_prompt=None)
    ablation["none"] = res["rate"]
    print(f"  None: {res['rate']:.1%}")

    return {"ablation": ablation}


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 60)
    print("P3.C3: System Prompt Instruction Style Injection")
    print(f"IS_SMOKE={IS_SMOKE}, N_STYLE={N_STYLE}, N_ABLATION={N_ABLATION}")
    print("=" * 60)

    results = {"is_smoke": IS_SMOKE, "timestamp": time.time()}

    # Phase 0
    p0 = phase0_verify()
    results.update(p0)
    if not p0["artifacts_ok"]:
        print("\nERROR: Missing B5 artifacts.")
        results["error"] = "missing_artifacts"
        results["elapsed_s"] = time.time() - t_start
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return

    # Phase 1: Control
    p1 = phase1_control(N_STYLE)
    results.update(p1)
    control_tokens = p1["control_avg_tokens"]

    # Phase 2: System prompt instruction
    p2 = phase2_instruction(N_STYLE)
    results.update(p2)
    instruction_tokens = p2["instruction_avg_tokens"]

    # Compute overhead ratio
    overhead_ratio = instruction_tokens / max(control_tokens, 1.0)
    results["context_overhead_ratio"] = overhead_ratio
    print(f"\n  Context overhead ratio: {overhead_ratio:.2f}x")

    # Phase 3: Ablation
    p3 = phase3_ablation(N_ABLATION)
    results.update(p3)

    # Kill criteria
    elapsed = time.time() - t_start
    results["elapsed_s"] = elapsed

    instruction_rate = p2["instruction_rate"]
    k1202_pass = instruction_rate >= 0.80
    k1203_pass = True
    k1204_pass = overhead_ratio <= 2.0

    results["k1202_pass"] = k1202_pass
    results["k1203_pass"] = k1203_pass
    results["k1204_pass"] = k1204_pass

    print("\n" + "=" * 60)
    print("KILL CRITERIA SUMMARY")
    print("=" * 60)
    print(f"  K1202 instruction_style >= 80%: {instruction_rate:.1%} → {'PASS' if k1202_pass else 'FAIL'}")
    print(f"  K1203 zero_training_cost:       True → PASS")
    print(f"  K1204 overhead_ratio <= 2.0:    {overhead_ratio:.2f}x → {'PASS' if k1204_pass else 'FAIL'}")
    print(f"\n  Total elapsed: {elapsed:.1f}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
