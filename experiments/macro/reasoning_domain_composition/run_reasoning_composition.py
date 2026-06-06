#!/usr/bin/env python3
"""Test universal reasoning adapter composition with ALL pilot50 domain experts.

Hypothesis: ONE reasoning adapter composes with N domain experts without
retraining — improving reasoning on all domains while preserving domain quality.

Design:
  For each domain expert in pilot50:
    1. Baseline domain: domain expert alone on domain eval data → PPL
    2. Baseline reasoning: domain expert alone on math reasoning prompts → accuracy
    3. Composed domain: domain + reasoning adapter on domain eval data → PPL
    4. Composed reasoning: domain + reasoning adapter on math prompts → accuracy
    5. Trace quality: does composed model produce coherent <think>...</think>?

Kill criteria:
  K1: reasoning adapter improves <50% of domain experts on reasoning tasks
  K2: average domain quality degradation >3% when reasoning adapter is added
  K3: reasoning traces become incoherent when composed with domain experts
      (measured by: <think> tag completion rate and non-empty trace rate)

Expected runtime: ~60-90 min on RTX 4090 (tests all available adapters)
"""

import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

# ── Configuration ─────────────────────────────────────────────────────────────

REPO_ROOT = Path("/workspace/llm")
BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
HF_CACHE = os.environ.get("HF_CACHE", "/workspace/hf_cache")

ADAPTER_DIR = REPO_ROOT / "data" / "adapters"
DATA_DIR = REPO_ROOT / "data" / "distillation"
REASONING_ADAPTER = (
    REPO_ROOT / "micro" / "models" / "reasoning_expert_distillation" / "reasoning_adapter"
)
RESULTS_DIR = REPO_ROOT / "results" / "reasoning_domain_composition"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Eval config
EVAL_SAMPLES_DOMAIN = 5 if IS_SMOKE else 30  # per domain for PPL
EVAL_SAMPLES_REASONING = 3 if IS_SMOKE else 20  # math prompts for accuracy
MAX_SEQ_LEN = 256 if IS_SMOKE else 512
MAX_GEN_TOKENS = 64 if IS_SMOKE else 512  # for reasoning trace generation
MAX_DOMAINS = 5 if IS_SMOKE else 999  # cap for smoke test

SEED = 42

# Math reasoning prompts (subset of MATH test, diverse difficulty)
REASONING_PROMPTS = [
    "What is the remainder when 2^100 is divided by 7?",
    "Find all integer solutions to x^2 - 5x + 6 = 0.",
    "If f(x) = 3x + 2 and g(x) = x^2 - 1, what is f(g(3))?",
    "A bag contains 5 red and 3 blue balls. What is the probability of drawing 2 red balls without replacement?",
    "What is the sum of the first 20 positive integers?",
    "Simplify: (x^2 - 9) / (x + 3).",
    "In a right triangle with legs 3 and 4, what is the hypotenuse?",
    "How many ways can you arrange the letters in the word MISSISSIPPI?",
    "What is the derivative of x^3 * sin(x)?",
    "Find the GCD of 84 and 120.",
    "If log_2(x) = 5, what is x?",
    "A train travels 60 km/h for 2 hours, then 80 km/h for 3 hours. What is the average speed?",
    "What is the integral of 1/x from 1 to e?",
    "Solve: 2x + 3y = 12 and x - y = 1.",
    "How many diagonals does a regular hexagon have?",
    "What is the sum of the infinite geometric series 1 + 1/2 + 1/4 + ...?",
    "Find the area of a circle with circumference 10*pi.",
    "If 3^x = 81, what is x?",
    "What is the determinant of the matrix [[1,2],[3,4]]?",
    "A fair coin is flipped 5 times. What is the probability of exactly 3 heads?",
]

# Ground truth answers for accuracy checking (order matches REASONING_PROMPTS)
GROUND_TRUTH = [
    "2",       # 2^100 mod 7 = 2
    "2, 3",    # x=2 or x=3
    "26",      # f(8) = 26
    "5/14",    # C(5,2)/C(8,2) = 10/28 = 5/14
    "210",     # 20*21/2
    "x - 3",   # factor cancellation
    "5",       # 3-4-5 triangle
    "34650",   # 11!/(4!4!2!)
    "3x^2*sin(x) + x^3*cos(x)",  # product rule
    "12",      # GCD(84,120)
    "32",      # 2^5
    "72",      # 360/5 = 72 km/h
    "1",       # ln(e) - ln(1) = 1
    "x=3, y=2",  # system of equations
    "9",       # n(n-3)/2 = 6*3/2 = 9
    "2",       # a/(1-r) = 1/(1-1/2) = 2
    "25*pi",   # r=5, A=25pi
    "4",       # 3^4=81
    "-2",      # 1*4-2*3 = -2
    "5/16",    # C(5,3)/32 = 10/32 = 5/16
]


def log(msg: str):
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def discover_adapters():
    """Find all valid adapters with eval data."""
    adapters = []
    if not ADAPTER_DIR.exists():
        log(f"ERROR: adapter dir {ADAPTER_DIR} not found")
        sys.exit(1)
    for d in sorted(ADAPTER_DIR.iterdir()):
        if d.is_dir() and (d / "adapter_model.safetensors").exists():
            # Check if domain has eval data
            data_dir = DATA_DIR / d.name
            if data_dir.exists() and any(data_dir.iterdir()):
                adapters.append(d.name)
    return adapters


def load_eval_texts(domain, tokenizer, n):
    """Load evaluation texts for a domain from tail of training data."""
    texts = []
    for fname in ["eval.jsonl", "train.jsonl"]:
        f = DATA_DIR / domain / fname
        if not f.exists():
            continue
        with open(f) as fh:
            lines = fh.readlines()
        if fname == "train.jsonl":
            lines = lines[-min(200, len(lines)):]
        for line in lines:
            record = json.loads(line)
            if "messages" in record:
                text = tokenizer.apply_chat_template(
                    record["messages"], tokenize=False, add_generation_prompt=False)
                texts.append(text)
            elif "text" in record:
                texts.append(record["text"])
            if len(texts) >= n:
                return texts
    return texts


def compute_ppl(model, tokenizer, texts):
    """Compute mean perplexity over texts."""
    import torch
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=MAX_SEQ_LEN).to(model.device)
            if inputs["input_ids"].shape[1] < 2:
                continue
            outputs = model(**inputs, labels=inputs["input_ids"])
            n_tok = inputs["input_ids"].shape[1]
            total_loss += outputs.loss.item() * n_tok
            total_tokens += n_tok
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def check_answer(response: str, ground_truth: str) -> bool:
    """Loose check: does the response contain the ground truth answer?"""
    # Strip <think>...</think> to get the final answer
    final = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if not final:
        final = response
    # Normalize: lowercase, strip whitespace
    resp_norm = final.lower().replace(" ", "")
    gt_norm = ground_truth.lower().replace(" ", "")
    # Check containment (loose)
    return gt_norm in resp_norm


def check_trace_quality(response: str) -> dict:
    """Assess reasoning trace quality."""
    has_think_open = "<think>" in response
    has_think_close = "</think>" in response
    # Extract trace content
    trace_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    trace_text = trace_match.group(1).strip() if trace_match else ""
    trace_len = len(trace_text)
    # Non-trivial: at least 20 chars of reasoning
    is_nontrivial = trace_len >= 20

    return {
        "has_think_tags": has_think_open and has_think_close,
        "trace_length": trace_len,
        "is_nontrivial": is_nontrivial,
    }


def eval_reasoning(model, tokenizer, n_prompts):
    """Evaluate model on math reasoning prompts. Returns accuracy + trace info."""
    import torch

    prompts = REASONING_PROMPTS[:n_prompts]
    truths = GROUND_TRUTH[:n_prompts]

    results = []
    model.eval()
    for prompt, gt in zip(prompts, truths):
        messages = [
            {"role": "system", "content": (
                "You are a helpful math assistant. Show your reasoning "
                "step by step inside <think>...</think> tags before giving "
                "your final answer."
            )},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_GEN_TOKENS,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode only the generated tokens
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)

        correct = check_answer(response, gt)
        trace = check_trace_quality(response)

        results.append({
            "prompt": prompt,
            "ground_truth": gt,
            "correct": correct,
            "trace": trace,
            "response_preview": response[:200],
        })

    accuracy = sum(1 for r in results if r["correct"]) / max(len(results), 1)
    trace_rate = sum(1 for r in results if r["trace"]["has_think_tags"]) / max(len(results), 1)
    nontrivial_rate = sum(1 for r in results if r["trace"]["is_nontrivial"]) / max(len(results), 1)

    return {
        "accuracy": round(accuracy, 4),
        "trace_rate": round(trace_rate, 4),
        "nontrivial_trace_rate": round(nontrivial_rate, 4),
        "n_prompts": len(results),
        "details": results,
    }


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    t0 = time.time()
    log("=" * 72)
    log("REASONING DOMAIN COMPOSITION — UNIVERSALITY TEST")
    log("=" * 72)
    log(f"Smoke test: {IS_SMOKE}")

    # Verify reasoning adapter
    if not REASONING_ADAPTER.exists():
        log(f"ERROR: Reasoning adapter not found at {REASONING_ADAPTER}")
        log("Run train_reasoning_expert.py first.")
        sys.exit(1)

    # Discover domain adapters
    all_domains = discover_adapters()
    if len(all_domains) > MAX_DOMAINS:
        all_domains = all_domains[:MAX_DOMAINS]
    log(f"Testing {len(all_domains)} domain experts: {all_domains[:10]}{'...' if len(all_domains) > 10 else ''}")

    # Load tokenizer and base model
    log("\nLoading base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto",
        cache_dir=HF_CACHE, trust_remote_code=True)
    log("Base model loaded.")

    # ── Phase 1: Base model baselines ─────────────────────────────────────
    log("\n=== Phase 1: Base Model Baselines ===")

    # Base model reasoning
    log("  Base model reasoning accuracy...")
    base_reasoning = eval_reasoning(base_model, tokenizer, EVAL_SAMPLES_REASONING)
    log(f"  Base reasoning accuracy: {base_reasoning['accuracy']:.1%}")
    log(f"  Base trace rate: {base_reasoning['trace_rate']:.1%}")

    # ── Phase 2: Per-domain evaluation ────────────────────────────────────
    log("\n=== Phase 2: Per-Domain Evaluation ===")

    domain_results = {}
    for i, domain in enumerate(all_domains):
        log(f"\n  [{i+1}/{len(all_domains)}] {domain}")

        eval_texts = load_eval_texts(domain, tokenizer, EVAL_SAMPLES_DOMAIN)
        if not eval_texts:
            log(f"    SKIP: no eval data")
            continue

        # 2a. Domain expert alone — domain PPL
        domain_model = PeftModel.from_pretrained(
            base_model, str(ADAPTER_DIR / domain))
        domain_ppl = compute_ppl(domain_model, tokenizer, eval_texts)
        log(f"    Domain-only PPL: {domain_ppl:.4f} ({len(eval_texts)} samples)")

        # 2b. Domain expert alone — reasoning accuracy
        domain_reasoning = eval_reasoning(domain_model, tokenizer, EVAL_SAMPLES_REASONING)
        log(f"    Domain-only reasoning acc: {domain_reasoning['accuracy']:.1%}")

        del domain_model
        gc.collect()
        torch.cuda.empty_cache()

        # 2c. Composed (domain + reasoning) — domain PPL
        domain_model = PeftModel.from_pretrained(
            base_model, str(ADAPTER_DIR / domain))
        merged = domain_model.merge_and_unload()
        composed = PeftModel.from_pretrained(merged, str(REASONING_ADAPTER))
        composed_ppl = compute_ppl(composed, tokenizer, eval_texts)
        log(f"    Composed PPL: {composed_ppl:.4f}")

        # 2d. Composed — reasoning accuracy + trace quality
        composed_reasoning = eval_reasoning(composed, tokenizer, EVAL_SAMPLES_REASONING)
        log(f"    Composed reasoning acc: {composed_reasoning['accuracy']:.1%}")
        log(f"    Composed trace rate: {composed_reasoning['trace_rate']:.1%}")

        # Compute deltas
        ppl_degradation = (composed_ppl - domain_ppl) / domain_ppl * 100
        reasoning_delta = composed_reasoning["accuracy"] - domain_reasoning["accuracy"]
        reasoning_improved = composed_reasoning["accuracy"] > domain_reasoning["accuracy"]

        log(f"    PPL degradation: {ppl_degradation:+.2f}%")
        log(f"    Reasoning improvement: {reasoning_delta:+.1%}")

        domain_results[domain] = {
            "domain_ppl": round(float(domain_ppl), 4),
            "composed_ppl": round(float(composed_ppl), 4),
            "ppl_degradation_pct": round(float(ppl_degradation), 2),
            "domain_reasoning_acc": round(float(domain_reasoning["accuracy"]), 4),
            "composed_reasoning_acc": round(float(composed_reasoning["accuracy"]), 4),
            "reasoning_delta": round(float(reasoning_delta), 4),
            "reasoning_improved": reasoning_improved,
            "composed_trace_rate": round(float(composed_reasoning["trace_rate"]), 4),
            "composed_nontrivial_trace_rate": round(float(composed_reasoning["nontrivial_trace_rate"]), 4),
            "n_eval": len(eval_texts),
        }

        del composed, merged
        gc.collect()
        torch.cuda.empty_cache()

    # ── Phase 3: Kill Criteria Assessment ────────────────────────────────
    log("\n" + "=" * 72)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 72)

    n_tested = len(domain_results)
    if n_tested == 0:
        log("ERROR: No domains evaluated!")
        sys.exit(1)

    # K1: reasoning adapter improves <50% of domain experts
    n_improved = sum(1 for r in domain_results.values() if r["reasoning_improved"])
    improvement_rate = n_improved / n_tested
    k1_pass = improvement_rate >= 0.50
    log(f"\n  K1: Reasoning improves >=50% of domains")
    log(f"      Improved: {n_improved}/{n_tested} ({improvement_rate:.1%})")
    log(f"      Verdict: {'PASS' if k1_pass else 'KILL'}")

    # K2: average domain quality degradation >3%
    degradations = [r["ppl_degradation_pct"] for r in domain_results.values()]
    mean_degradation = sum(degradations) / len(degradations)
    max_degradation = max(degradations)
    k2_pass = mean_degradation <= 3.0
    log(f"\n  K2: Mean domain degradation <=3%")
    log(f"      Mean: {mean_degradation:+.2f}%")
    log(f"      Max:  {max_degradation:+.2f}%")
    log(f"      Verdict: {'PASS' if k2_pass else 'KILL'}")

    # K3: reasoning traces incoherent (tag completion + non-trivial)
    trace_rates = [r["composed_trace_rate"] for r in domain_results.values()]
    nontrivial_rates = [r["composed_nontrivial_trace_rate"] for r in domain_results.values()]
    mean_trace_rate = sum(trace_rates) / len(trace_rates)
    mean_nontrivial = sum(nontrivial_rates) / len(nontrivial_rates)
    k3_pass = mean_trace_rate >= 0.50 and mean_nontrivial >= 0.30
    log(f"\n  K3: Coherent reasoning traces")
    log(f"      Mean trace tag rate: {mean_trace_rate:.1%}")
    log(f"      Mean nontrivial trace rate: {mean_nontrivial:.1%}")
    log(f"      Verdict: {'PASS' if k3_pass else 'KILL'}")

    # Overall
    all_pass = k1_pass and k2_pass and k3_pass
    verdict = "PASS" if all_pass else "KILLED"
    log(f"\n  OVERALL VERDICT: {verdict}")

    # Reasoning accuracy summary
    domain_accs = [r["domain_reasoning_acc"] for r in domain_results.values()]
    composed_accs = [r["composed_reasoning_acc"] for r in domain_results.values()]
    log(f"\n  Reasoning Accuracy Summary:")
    log(f"    Base model: {base_reasoning['accuracy']:.1%}")
    log(f"    Domain-only mean: {sum(domain_accs)/len(domain_accs):.1%}")
    log(f"    Composed mean: {sum(composed_accs)/len(composed_accs):.1%}")
    log(f"    Delta: {sum(composed_accs)/len(composed_accs) - sum(domain_accs)/len(domain_accs):+.1%}")

    # ── Save Results ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    results = {
        "experiment": "reasoning_domain_composition",
        "hypothesis": "ONE reasoning adapter composes with N domain experts universally",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "base_model": BASE_MODEL,
            "reasoning_adapter": str(REASONING_ADAPTER),
            "n_domains_tested": n_tested,
            "eval_samples_domain": EVAL_SAMPLES_DOMAIN,
            "eval_samples_reasoning": EVAL_SAMPLES_REASONING,
            "max_seq_len": MAX_SEQ_LEN,
            "max_gen_tokens": MAX_GEN_TOKENS,
            "smoke_test": IS_SMOKE,
        },
        "base_reasoning": {
            "accuracy": base_reasoning["accuracy"],
            "trace_rate": base_reasoning["trace_rate"],
        },
        "per_domain": domain_results,
        "kill_criteria": {
            "K1_universality_ge_50pct": {
                "threshold": 0.50,
                "actual": round(improvement_rate, 4),
                "n_improved": n_improved,
                "n_tested": n_tested,
                "pass": k1_pass,
            },
            "K2_mean_degradation_le_3pct": {
                "threshold": 3.0,
                "actual": round(mean_degradation, 2),
                "max_degradation": round(max_degradation, 2),
                "pass": k2_pass,
            },
            "K3_coherent_traces": {
                "mean_trace_rate": round(mean_trace_rate, 4),
                "mean_nontrivial_rate": round(mean_nontrivial, 4),
                "pass": k3_pass,
            },
        },
        "verdict": verdict,
        "elapsed_s": round(elapsed, 1),
    }

    out_file = RESULTS_DIR / "reasoning_composition_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {out_file}")
    log(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
