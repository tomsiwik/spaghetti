#!/usr/bin/env python3
"""Pierre Tiny: full benchmark suite for leaderboard positioning.

Runs MMLU (50Q), GSM8K (30Q), HumanEval (15Q) on BitNet-2B-4T with:
  1. Base (no adapter)                      -- the floor
  2. NTP math adapter, scale=20             -- best for reasoning (Finding #262)
  3. NTP code adapter, scale=20             -- best for code gen
  4. SFT code adapter, scale=1              -- low-scale preserves MMLU (Finding #320)
  5. Composed N=5 SFT, DARE p=0.5, scale=1  -- full composition (Finding #266)

Kill criteria:
  K820: All benchmarks below base model (adapters make everything worse)

Prior base scores (Finding #213, n=50/50/20):
  MMLU: 38%, GSM8K: 58%, HumanEval: 60%
"""

import ast
import gc
import json
import math
import os
import re
import sys
import subprocess
import tempfile
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from pierre import (
    attach_adapter, detach_adapters, compose_adapters,
    load_adapter, load_frozen_A, ADAPTER_TARGETS,
)
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

SFT_SOURCE = EXPERIMENT_DIR.parent / "bitnet_sft_generation_v3" / "sft_adapters"
NTP_SOURCE = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "adapters"
SKELETON_PATH = NTP_SOURCE / "grassmannian_skeleton.npz"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
SEED = 42
DOMAINS = ["medical", "code", "math", "legal", "finance"]

# Sample counts -- enough for directional signal, fast enough for <2hr
N_MMLU = 50
N_GSM8K = 30
N_HUMANEVAL = 15


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


def log(m): print(m, flush=True)

def log_memory(label=""):
    a = mx.get_active_memory() / 1e9; p = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB peak={p:.2f}GB")

def cleanup(*o):
    for x in o: del x
    gc.collect(); mx.clear_cache(); mx.reset_peak_memory()


# ============================================================================
# Data loading (from HuggingFace datasets)
# ============================================================================

def load_mmlu_questions(n=N_MMLU):
    """Load MMLU questions from diverse subjects."""
    from datasets import load_dataset
    log(f"Loading {n} MMLU questions...")
    subjects = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics",
        "computer_security", "conceptual_physics", "econometrics",
        "electrical_engineering", "elementary_mathematics", "formal_logic",
        "global_facts", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_european_history",
        "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_mathematics",
        "high_school_microeconomics", "high_school_physics",
        "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history",
        "human_aging", "human_sexuality", "international_law",
        "jurisprudence", "logical_fallacies", "machine_learning",
        "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios",
        "nutrition", "philosophy", "prehistory",
        "professional_accounting", "professional_law",
        "professional_medicine", "professional_psychology",
    ]
    questions = []
    per_subject = max(1, n // len(subjects))
    remaining = n
    for subj in subjects:
        if remaining <= 0: break
        try:
            ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
            take = min(per_subject, len(ds), remaining)
            for i in range(take):
                item = ds[i]
                questions.append({
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": item["answer"],
                    "subject": subj,
                })
                remaining -= 1
        except Exception as e:
            log(f"  Warning: skip {subj}: {e}")
    log(f"  Loaded {len(questions)} MMLU questions from {len(set(q['subject'] for q in questions))} subjects")
    return questions


def load_gsm8k_problems(n=N_GSM8K):
    """Load GSM8K math reasoning problems."""
    from datasets import load_dataset
    log(f"Loading {n} GSM8K problems...")
    ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
    problems = []
    for i in range(min(n, len(ds))):
        item = ds[i]
        match = re.search(r'####\s*([\-\d,]+(?:\.\d+)?)', item["answer"])
        answer_num = float(match.group(1).replace(',', '')) if match else None
        problems.append({
            "question": item["question"],
            "answer_text": item["answer"],
            "answer_num": answer_num,
        })
    log(f"  Loaded {len(problems)} GSM8K problems")
    return problems


def load_humaneval_problems(n=N_HUMANEVAL):
    """Load HumanEval code generation problems."""
    from datasets import load_dataset
    log(f"Loading {n} HumanEval problems...")
    ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
    problems = []
    for i in range(min(n, len(ds))):
        item = ds[i]
        problems.append({
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "canonical_solution": item["canonical_solution"],
            "test": item["test"],
            "entry_point": item["entry_point"],
        })
    log(f"  Loaded {len(problems)} HumanEval problems")
    return problems


# ============================================================================
# MMLU evaluation -- logit-based (no generation needed, faster + more stable)
# ============================================================================

def eval_mmlu_logit(model, tokenizer, questions):
    """Evaluate MMLU via logit comparison at last token position.

    For each question, compute P(A), P(B), P(C), P(D) from logits.
    This is the standard approach for LLM benchmarks (lm-eval-harness style).
    """
    labels = ["A", "B", "C", "D"]
    # Pre-compute token IDs for each answer label
    label_ids = {}
    for l in labels:
        toks = tokenizer.encode(f" {l}")
        # Use last token (handles BPE variations)
        label_ids[l] = toks[-1] if toks else tokenizer.encode(l)[-1]

    correct = 0
    total = 0
    details = []

    for q in questions:
        choices_text = "\n".join(f"{l}) {c}" for l, c in zip(labels, q["choices"]))
        prompt = f"Q: {q['question']}\n{choices_text}\nAnswer:"
        tokens = tokenizer.encode(prompt)
        if len(tokens) > 512:
            tokens = tokens[:512]

        logits = model(mx.array(tokens)[None, :])
        mx.eval(logits)
        last_logits = logits[0, -1]

        probs = {l: last_logits[label_ids[l]].item() for l in labels}
        predicted = max(probs, key=probs.get)
        correct_label = labels[q["answer"]]
        is_correct = predicted == correct_label

        if is_correct: correct += 1
        total += 1
        details.append({
            "subject": q["subject"],
            "predicted": predicted,
            "correct": correct_label,
            "is_correct": is_correct,
        })
        del logits, last_logits

    acc = correct / total if total else 0.0
    return acc, correct, total, details


# ============================================================================
# GSM8K evaluation -- generation-based
# ============================================================================

def eval_gsm8k(model, tokenizer, problems):
    """Evaluate GSM8K with chain-of-thought prompting."""
    sampler = make_sampler(temp=0.0)
    correct = 0
    total = 0
    details = []

    for p in problems:
        prompt = (
            f"### Instruction:\n"
            f"Solve this math problem step by step. "
            f"End your answer with #### followed by the numerical answer.\n\n"
            f"{p['question']}\n\n"
            f"### Response:\n"
        )
        try:
            output = mlx_generate(
                model, tokenizer, prompt=prompt,
                max_tokens=300, sampler=sampler, verbose=False,
            )
        except Exception as e:
            output = ""
            log(f"  GSM8K gen failed: {e}")

        predicted = extract_gsm8k_answer(output)
        expected = p["answer_num"]

        is_correct = False
        if predicted is not None and expected is not None:
            is_correct = abs(predicted - expected) < 0.01

        if is_correct: correct += 1
        total += 1
        details.append({
            "predicted": predicted,
            "expected": expected,
            "is_correct": is_correct,
            "output_preview": output[:150] if output else "",
        })

    acc = correct / total if total else 0.0
    return acc, correct, total, details


def extract_gsm8k_answer(text):
    """Extract numerical answer from GSM8K response."""
    # Pattern: "#### X"
    matches = re.findall(r'####\s*([\-\d,]+(?:\.\d+)?)', text)
    if matches:
        try: return float(matches[-1].replace(',', ''))
        except ValueError: pass
    # Pattern: "the answer is X"
    m = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*\$?([\-\d,]+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        try: return float(m.group(1).replace(',', ''))
        except ValueError: pass
    # Pattern: "= X"
    matches = re.findall(r'=\s*\$?([\-\d,]+(?:\.\d+)?)', text)
    if matches:
        try: return float(matches[-1].replace(',', ''))
        except ValueError: pass
    # Last number
    matches = re.findall(r'([\-\d,]+(?:\.\d+)?)', text)
    if matches:
        try: return float(matches[-1].replace(',', ''))
        except ValueError: pass
    return None


# ============================================================================
# HumanEval evaluation -- generation + execution
# ============================================================================

def clean_completion(completion):
    """Clean generated completion: remove markdown fences and trailing non-code."""
    clean = re.sub(r'```python\s*\n?', '', completion)
    clean = re.sub(r'```\s*\n?', '', clean)

    lines = clean.split('\n')
    body_lines = []
    found_body = False
    for line in lines:
        stripped = line.strip()
        # Stop at second top-level def/class
        if (stripped.startswith('def ') or stripped.startswith('class ')) and found_body:
            break
        # Stop at prose lines (non-indented, non-comment, non-code-keyword)
        if found_body and stripped and not line.startswith(' ') and not line.startswith('\t'):
            if not stripped.startswith('#') and not stripped.startswith('def ') \
               and not stripped.startswith('class ') and not stripped.startswith('return') \
               and not stripped.startswith('from ') and not stripped.startswith('import '):
                break
        if stripped:
            found_body = True
        body_lines.append(line)

    # Trim trailing empty/comment lines
    while body_lines and (not body_lines[-1].strip() or body_lines[-1].strip().startswith('#')):
        body_lines.pop()

    return '\n'.join(body_lines)


def try_fix_indent(completion_text):
    """Try to fix indentation issues in generated code.

    Common issue: model generates 3-space indent instead of 4-space.
    Strategy: for each line, if indent % 4 != 0, round up to nearest multiple of 4.
    """
    lines = completion_text.split('\n')
    fixed = []
    for line in lines:
        if not line.strip():
            fixed.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        if indent % 4 != 0:
            new_indent = ((indent + 3) // 4) * 4  # round up
            fixed.append(' ' * new_indent + line.lstrip())
        else:
            fixed.append(line)
    return '\n'.join(fixed)


def extract_function_body(prompt, completion):
    """Extract clean function code from prompt + completion.

    Tries multiple strategies to produce parseable Python.
    """
    cleaned = clean_completion(completion)

    # Strategy 1: direct concatenation
    code = prompt + cleaned
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        pass

    # Strategy 2: fix indentation (round non-4-aligned to nearest 4)
    code = prompt + try_fix_indent(cleaned)
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        pass

    # Strategy 3: add 1 space to all completion lines (3->4 fix)
    shifted = '\n'.join((' ' + l if l.strip() else l) for l in cleaned.split('\n'))
    code = prompt + shifted
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        pass

    # Give up: return best attempt (raw concatenation)
    return prompt + cleaned


def eval_humaneval(model, tokenizer, problems):
    """Evaluate HumanEval: generate code completion, run test cases.

    Uses pass@1: model gets one attempt per problem, greedy decoding.
    """
    sampler = make_sampler(temp=0.0)
    correct = 0
    total = 0
    details = []

    for p in problems:
        # HumanEval prompt IS the function signature + docstring
        prompt = p["prompt"]
        try:
            completion = mlx_generate(
                model, tokenizer, prompt=prompt,
                max_tokens=256, sampler=sampler, verbose=False,
            )
        except Exception as e:
            completion = ""
            log(f"  HumanEval gen failed: {e}")

        # Extract clean function code
        full_code = extract_function_body(prompt, completion)

        # Run test
        passed = run_humaneval_test(full_code, p["test"], p["entry_point"])
        if passed: correct += 1
        total += 1
        details.append({
            "task_id": p["task_id"],
            "passed": passed,
            "code_preview": full_code[:200],
        })

    acc = correct / total if total else 0.0
    return acc, correct, total, details


def run_humaneval_test(code, test_code, entry_point):
    """Run HumanEval test in subprocess (sandboxed, 10s timeout)."""
    full = code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full)
            f.flush()
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, timeout=10,
            )
            os.unlink(f.name)
            return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        try: os.unlink(f.name)
        except: pass
        return False


# ============================================================================
# DARE sparsification (Finding #266)
# ============================================================================

def dare_sparsify(adapter_b, p=0.5, seed=42):
    """DARE: randomly drop p fraction, rescale by 1/(1-p). E[B_sparse] = B."""
    rng = np.random.RandomState(seed)
    sparsified = {}
    for key, val in adapter_b.items():
        mask = mx.array(rng.binomial(1, 1.0 - p, size=val.shape).astype(np.float32))
        sparsified[key] = (val * mask / (1.0 - p)).astype(val.dtype)
    return sparsified


# ============================================================================
# Configuration definitions
# ============================================================================

def get_configurations():
    """Define all benchmark configurations.

    Returns list of (name, adapter_type, adapter_domain, domain_idx, scale, compose_all).
    """
    return [
        # (name, adapter_source, domain, domain_idx, scale, compose_all_flag)
        ("base",              None,  None,   None, None,  False),
        ("ntp_math_s20",      "ntp", "math", 2,    20.0, False),
        ("ntp_code_s20",      "ntp", "code", 1,    20.0, False),
        ("sft_code_s1",       "sft", "code", 1,    1.0,  False),
        ("sft_math_s1",       "sft", "math", 2,    1.0,  False),
        ("composed_n5_dare_s1", "sft", None, 0,    1.0,  True),
    ]


# ============================================================================
# Phase functions (MANDATORY scoping per CODING_GUIDELINES)
# ============================================================================

def phase_load_data():
    """Phase 0: Load all benchmark datasets (no GPU memory)."""
    log("\n=== Phase 0: Loading Data ===")
    mmlu = load_mmlu_questions(N_MMLU)
    gsm8k = load_gsm8k_problems(N_GSM8K)
    humaneval = load_humaneval_problems(N_HUMANEVAL)
    return mmlu, gsm8k, humaneval


def phase_evaluate_config(config_name, adapter_source, domain, domain_idx,
                          scale, compose_all, frozen_A, mmlu, gsm8k, humaneval):
    """Phase N: Evaluate a single configuration across all benchmarks.

    Loads model fresh, attaches adapter if needed, evaluates, cleans up.
    """
    log(f"\n{'='*60}")
    log(f"=== Config: {config_name} ===")
    log(f"{'='*60}")

    model, tok = load(MODEL_ID)
    log_memory(f"{config_name}-loaded")

    if adapter_source is not None and not compose_all:
        # Single adapter
        if adapter_source == "ntp":
            adapter_path = str(NTP_SOURCE / domain / "adapter.npz")
        else:
            adapter_path = str(SFT_SOURCE / domain / "adapter.npz")
        adapter = load_adapter(adapter_path)
        n_wrapped = attach_adapter(model, frozen_A, adapter, domain_idx, scale)
        log(f"  Attached {adapter_source}/{domain} adapter (scale={scale}, wrapped={n_wrapped})")
        del adapter

    elif compose_all:
        # Compose N=5 SFT adapters with DARE p=0.5
        log(f"  Composing N={len(DOMAINS)} SFT adapters with DARE p=0.5, scale={scale}")
        adapters = []
        for di, d in enumerate(DOMAINS):
            a = load_adapter(str(SFT_SOURCE / d / "adapter.npz"))
            a = dare_sparsify(a, p=0.5, seed=SEED + di)
            adapters.append(a)
        composed = compose_adapters(adapters)
        n_wrapped = attach_adapter(model, frozen_A, composed, 0, scale)
        log(f"  Attached composed adapter (wrapped={n_wrapped})")
        del adapters, composed

    gc.collect(); mx.clear_cache()

    # --- MMLU (logit-based) ---
    log(f"\n  --- MMLU ({len(mmlu)}Q, logit-based) ---")
    t0 = time.time()
    mmlu_acc, mmlu_c, mmlu_t, mmlu_det = eval_mmlu_logit(model, tok, mmlu)
    mmlu_time = time.time() - t0
    log(f"  MMLU: {mmlu_c}/{mmlu_t} = {mmlu_acc:.1%} ({mmlu_time:.0f}s)")

    # --- GSM8K (generation-based) ---
    log(f"\n  --- GSM8K ({len(gsm8k)}Q, generation) ---")
    t0 = time.time()
    gsm_acc, gsm_c, gsm_t, gsm_det = eval_gsm8k(model, tok, gsm8k)
    gsm_time = time.time() - t0
    log(f"  GSM8K: {gsm_c}/{gsm_t} = {gsm_acc:.1%} ({gsm_time:.0f}s)")

    # --- HumanEval (generation + execution) ---
    log(f"\n  --- HumanEval ({len(humaneval)}Q, generation+exec) ---")
    t0 = time.time()
    he_acc, he_c, he_t, he_det = eval_humaneval(model, tok, humaneval)
    he_time = time.time() - t0
    log(f"  HumanEval: {he_c}/{he_t} = {he_acc:.1%} ({he_time:.0f}s)")

    result = {
        "mmlu": {"correct": mmlu_c, "total": mmlu_t, "acc": round(mmlu_acc, 4),
                 "time_s": round(mmlu_time, 1)},
        "gsm8k": {"correct": gsm_c, "total": gsm_t, "acc": round(gsm_acc, 4),
                  "time_s": round(gsm_time, 1)},
        "humaneval": {"correct": he_c, "total": he_t, "acc": round(he_acc, 4),
                      "time_s": round(he_time, 1)},
    }

    log_memory(f"{config_name}-done")
    cleanup(model, tok)
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("Pierre Tiny: Benchmark Suite")
    log("=" * 60)
    log(f"Platform: {device_info.get('arch', 'unknown')}, "
        f"Memory: {device_info['memory_size'] / 1e9:.0f}GB")
    mx.random.seed(SEED)

    # Load frozen A matrices (shared across all configs)
    frozen_A = load_frozen_A(str(SKELETON_PATH))
    log(f"Loaded Grassmannian skeleton: {len(frozen_A)} keys")

    # Phase 0: Load data
    mmlu, gsm8k, humaneval = phase_load_data()

    # Evaluate each configuration
    configs = get_configurations()
    results = {"benchmarks": {}, "meta": {
        "model": MODEL_ID,
        "n_mmlu": len(mmlu),
        "n_gsm8k": len(gsm8k),
        "n_humaneval": len(humaneval),
        "seed": SEED,
        "domains": DOMAINS,
    }}

    for config_name, asrc, domain, didx, scale, compose in configs:
        r = phase_evaluate_config(
            config_name, asrc, domain, didx, scale, compose,
            frozen_A, mmlu, gsm8k, humaneval,
        )
        results["benchmarks"][config_name] = r

    # ========== Analysis ==========
    log(f"\n{'='*60}")
    log("RESULTS SUMMARY")
    log(f"{'='*60}")

    base = results["benchmarks"]["base"]
    benchmarks = ["mmlu", "gsm8k", "humaneval"]

    # Print comparison table
    header = f"{'Config':<25} {'MMLU':>8} {'GSM8K':>8} {'HumanEval':>10}"
    log(header)
    log("-" * len(header))
    for c, b in results["benchmarks"].items():
        mmlu_s = f"{b['mmlu']['acc']:.1%}"
        gsm_s = f"{b['gsm8k']['acc']:.1%}"
        he_s = f"{b['humaneval']['acc']:.1%}"
        tag = " *" if c == "base" else ""
        log(f"{c:<25} {mmlu_s:>8} {gsm_s:>8} {he_s:>10}{tag}")

    # Print deltas vs base
    log(f"\nDeltas vs base:")
    log(f"{'Config':<25} {'MMLU':>8} {'GSM8K':>8} {'HumanEval':>10}")
    log("-" * 60)
    for c, b in results["benchmarks"].items():
        if c == "base": continue
        deltas = {}
        for bm in benchmarks:
            delta = b[bm]["acc"] - base[bm]["acc"]
            deltas[bm] = delta
        log(f"{c:<25} {deltas['mmlu']:>+7.1%} {deltas['gsm8k']:>+7.1%} {deltas['humaneval']:>+9.1%}")

    # Kill criterion: K820 -- any adapter config beats base on any benchmark?
    any_improvement = False
    best_improvements = {}
    for bm in benchmarks:
        best_config = None
        best_delta = 0.0
        for c, b in results["benchmarks"].items():
            if c == "base": continue
            delta = b[bm]["acc"] - base[bm]["acc"]
            if delta > best_delta:
                best_delta = delta
                best_config = c
        best_improvements[bm] = {"config": best_config, "delta": round(best_delta, 4)}
        if best_delta > 0:
            any_improvement = True

    k820_pass = any_improvement
    results["kill_criteria"] = {
        "K820": {
            "pass": k820_pass,
            "detail": f"Any adapter beats base on any benchmark: {any_improvement}",
            "best_improvements": best_improvements,
        }
    }

    # Success criterion: S82 -- domain-specific improvement without general degradation
    # Check: best single adapter per benchmark > base, AND composed doesn't degrade MMLU
    s82_detail = {}
    for bm in benchmarks:
        best = best_improvements[bm]
        s82_detail[bm] = {
            "best_config": best["config"],
            "delta": best["delta"],
            "improved": best["delta"] > 0,
        }
    composed_mmlu_delta = (
        results["benchmarks"]["composed_n5_dare_s1"]["mmlu"]["acc"]
        - base["mmlu"]["acc"]
    )
    s82_pass = any_improvement and composed_mmlu_delta > -0.10  # within 10pp
    results["success_criteria"] = {
        "S82": {
            "pass": s82_pass,
            "detail": f"Domain improvement={any_improvement}, composed MMLU delta={composed_mmlu_delta:+.1%}",
            "composed_mmlu_delta": round(composed_mmlu_delta, 4),
        }
    }

    results["total_time_s"] = round(time.time() - t0, 1)
    results["all_pass"] = k820_pass

    log(f"\n{'='*60}")
    log(f"K820 ({'PASS' if k820_pass else 'FAIL'}): "
        f"Any adapter beats base = {any_improvement}")
    log(f"S82  ({'PASS' if s82_pass else 'FAIL'}): "
        f"Domain improvement + composition preserves MMLU")
    log(f"Total time: {results['total_time_s']:.0f}s")
    log(f"{'='*60}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
