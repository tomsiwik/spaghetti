#!/usr/bin/env python3
"""Competitive Benchmark: BitNet-2B + SOLE vs Qwen2.5-3B vs Gemma-2-2B.

Head-to-head comparison of our composed system against monolithic competitors
on real benchmarks (GSM8K, MMLU).

Kill criteria:
  K1 (#512): Composed worse than Qwen2.5-3B on > 60% of benchmarks -> KILL
  K2 (#513): Composed worse than base BitNet-2B alone on any benchmark -> KILL
  K3 (#514): Total memory exceeds Qwen2.5-3B quantized -> KILL

Success criteria:
  S1 (#40): Beats Qwen2.5-3B on >= 3/5 domain benchmarks at < 3GB memory

Platform: Apple M5 Pro 48GB, MLX
"""

import gc
import json
import math
import os
import re
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Source experiment with trained adapters + Grassmannian skeleton
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"

MODEL_ID_BITNET = "microsoft/BitNet-b1.58-2B-4T"
MODEL_ID_QWEN = "mlx-community/Qwen2.5-3B-Instruct-4bit"
MODEL_ID_GEMMA = "mlx-community/gemma-2-2b-it-4bit"

LORA_RANK = 16
LORA_SCALE = 20.0

DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = len(DOMAINS)

# MMLU subject mapping per adapter domain
MMLU_SUBJECTS = {
    "medical": ["clinical_knowledge", "professional_medicine", "anatomy", "medical_genetics"],
    "code": ["college_computer_science", "high_school_computer_science", "machine_learning"],
    "math": ["high_school_mathematics", "elementary_mathematics", "college_mathematics"],
    "legal": ["professional_law", "jurisprudence", "international_law"],
    "finance": ["professional_accounting", "econometrics", "high_school_macroeconomics"],
}

# Evaluation sizes
MMLU_N_PER_DOMAIN = 20
GSM8K_N = 50
MAX_NEW_TOKENS_GSM8K = 256
MAX_NEW_TOKENS_MMLU = 32

# Target layers for LoRA
TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def log(msg, end="\n"):
    print(msg, end=end, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ============================================================================
# BitNet unpacking (reused from prior experiments)
# ============================================================================

from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.bitlinear_layers import BitLinear


def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    """Unpack uint8-packed ternary weights to bfloat16."""
    w0 = (packed_weights & 3).astype(mx.bfloat16) - 1
    w1 = ((packed_weights >> 2) & 3).astype(mx.bfloat16) - 1
    w2 = ((packed_weights >> 4) & 3).astype(mx.bfloat16) - 1
    w3 = ((packed_weights >> 6) & 3).astype(mx.bfloat16) - 1
    unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:out_features]
    scale = weight_scale.astype(mx.bfloat16)
    if invert_scale:
        unpacked = unpacked / scale
    else:
        unpacked = unpacked * scale
    return unpacked


def replace_bitlinear_with_linear(model):
    """Replace BitLinear with nn.Linear for differentiable forward pass."""
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                unpacked_w = unpack_ternary(
                    module.weight, module.out_features,
                    module.weight_scale, module.invert_weight_scales,
                )
                has_bias = module.bias is not None
                linear = nn.Linear(module.in_features, module.out_features, bias=has_bias)
                linear.weight = unpacked_w
                if has_bias:
                    linear.bias = module.bias
                updates.append((key, linear))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    log(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


# ============================================================================
# Pre-merge composition (from e2e_demo_pipeline_mlx)
# ============================================================================

def load_skeleton():
    """Load the Grassmannian A matrices."""
    skeleton_path = ADAPTERS_DIR / "grassmannian_skeleton.npz"
    return dict(np.load(str(skeleton_path)))


def load_all_adapters():
    """Load all domain adapter B matrices from disk."""
    adapters = {}
    for domain in DOMAINS:
        adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
        adapters[domain] = dict(mx.load(str(adapter_path)))
        log(f"  Loaded adapter: {domain} ({len(adapters[domain])} tensors)")
    return adapters


def premerge_adapters_into_model(model, skeleton, adapters, domain_weights):
    """Pre-merge selected adapters into model weights.

    W_new = W_base + sum_d w_d * scale * B_d^T @ A_d^T
    """
    n_layers = len(model.model.layers)
    merge_count = 0

    for li in range(n_layers):
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = model.model.layers[li]
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue

            delta = None
            for domain, w in domain_weights.items():
                if w < 1e-6:
                    continue
                di = DOMAINS.index(domain)
                skey = f"layer_{li}_{key}_domain_{di}"
                if skey not in skeleton:
                    continue
                a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)

                b_key = f"model.layers.{li}.{key}.lora_b"
                if b_key not in adapters[domain]:
                    continue
                b_mx = adapters[domain][b_key]

                lora_delta = w * LORA_SCALE * (b_mx.T @ a_mx.T)
                if delta is None:
                    delta = lora_delta
                else:
                    delta = delta + lora_delta

            if delta is not None:
                module.weight = module.weight + delta
                merge_count += 1

    mx.eval(model.parameters())
    active_domains = [d for d, w in domain_weights.items() if w >= 1e-6]
    log(f"  Pre-merged {len(active_domains)} adapters into {merge_count} layers")
    return model


# ============================================================================
# Data loading
# ============================================================================

def load_gsm8k_data(n=50):
    """Load GSM8K test problems."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    for i in range(min(n, len(ds))):
        item = ds[i]
        answer_text = item["answer"]
        match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', answer_text)
        if match:
            answer = float(match.group(1).replace(',', ''))
        else:
            nums = re.findall(r'([\d,]+(?:\.\d+)?)', answer_text)
            answer = float(nums[-1].replace(',', '')) if nums else None
        problems.append({
            "question": item["question"],
            "answer": answer,
            "answer_text": answer_text,
        })
    log(f"  Loaded {len(problems)} GSM8K problems")
    return problems


def load_mmlu_data(n_per_domain=20):
    """Load MMLU test questions mapped to our adapter domains."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")

    by_subject = {}
    for item in ds:
        subj = item["subject"]
        if subj not in by_subject:
            by_subject[subj] = []
        by_subject[subj].append(item)

    mmlu_data = {}
    for domain, subjects in MMLU_SUBJECTS.items():
        questions = []
        for subj in subjects:
            if subj in by_subject:
                questions.extend(by_subject[subj])
        rng = np.random.RandomState(42)
        rng.shuffle(questions)
        mmlu_data[domain] = questions[:n_per_domain]
        log(f"  MMLU {domain}: {len(mmlu_data[domain])} questions from {subjects}")

    return mmlu_data


# ============================================================================
# Prompt formatting (model-specific)
# ============================================================================

def format_gsm8k_prompt_generic(question):
    """Generic instruction format for GSM8K."""
    return (
        f"### Instruction:\n"
        f"Solve the following math problem step by step. "
        f"Show your work and give the final numerical answer after ####.\n\n"
        f"{question}\n\n"
        f"### Response:\n"
    )


def format_gsm8k_prompt_chatml(question):
    """ChatML format for Qwen models."""
    return (
        f"<|im_start|>system\nYou are a helpful math tutor. Solve problems step by step "
        f"and give the final numerical answer after ####.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def format_gsm8k_prompt_gemma(question):
    """Gemma chat format for GSM8K."""
    return (
        f"<start_of_turn>user\n"
        f"Solve the following math problem step by step. "
        f"Show your work and give the final numerical answer after ####.\n\n"
        f"{question}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def format_mmlu_prompt_generic(question, choices):
    """Generic format for MMLU."""
    choice_labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(f"{label}. {choice}" for label, choice in zip(choice_labels, choices))
    return (
        f"### Instruction:\n"
        f"Answer the following multiple choice question. "
        f"Reply with just the letter (A, B, C, or D).\n\n"
        f"{question}\n\n{choices_text}\n\n"
        f"### Response:\n"
    )


def format_mmlu_prompt_chatml(question, choices):
    """ChatML format for Qwen models."""
    choice_labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(f"{label}. {choice}" for label, choice in zip(choice_labels, choices))
    return (
        f"<|im_start|>system\nYou are a helpful assistant. Answer multiple choice questions "
        f"with just the letter (A, B, C, or D).<|im_end|>\n"
        f"<|im_start|>user\n{question}\n\n{choices_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def format_mmlu_prompt_gemma(question, choices):
    """Gemma chat format for MMLU."""
    choice_labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(f"{label}. {choice}" for label, choice in zip(choice_labels, choices))
    return (
        f"<start_of_turn>user\n"
        f"Answer the following multiple choice question. "
        f"Reply with just the letter (A, B, C, or D).\n\n"
        f"{question}\n\n{choices_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


# ============================================================================
# Answer extraction
# ============================================================================

def extract_gsm8k_answer(generated_text):
    """Extract numerical answer from GSM8K-style generation."""
    match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', generated_text)
    if match:
        return float(match.group(1).replace(',', ''))
    match = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*\$?([\d,]+(?:\.\d+)?)', generated_text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(',', ''))
    matches = re.findall(r'=\s*\$?([\d,]+(?:\.\d+)?)', generated_text)
    if matches:
        return float(matches[-1].replace(',', ''))
    matches = re.findall(r'\$([\d,]+(?:\.\d+)?)', generated_text)
    if matches:
        return float(matches[-1].replace(',', ''))
    matches = re.findall(r'([\d,]+(?:\.\d+)?)', generated_text)
    if matches:
        try:
            return float(matches[-1].replace(',', ''))
        except ValueError:
            pass
    return None


def check_gsm8k_correct(predicted, ground_truth, tolerance=0.01):
    """Check if GSM8K answer is correct (within tolerance)."""
    if predicted is None or ground_truth is None:
        return False
    if ground_truth == 0:
        return abs(predicted) < tolerance
    return abs(predicted - ground_truth) / abs(ground_truth) < tolerance


def extract_mmlu_answer(generated_text):
    """Extract letter answer (A/B/C/D) from generated text."""
    text = generated_text.strip()
    if text and text[0].upper() in "ABCD":
        return text[0].upper()
    match = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*([A-Da-d])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    for line in text.split('\n'):
        line = line.strip()
        if len(line) == 1 and line.upper() in "ABCD":
            return line.upper()
    match = re.search(r'[\(\s]([A-Da-d])[\)\.\s]', text)
    if match:
        return match.group(1).upper()
    match = re.search(r'\b([A-Da-d])\b', text)
    if match:
        return match.group(1).upper()
    return None


# ============================================================================
# Generation
# ============================================================================

def generate_text(model, tokenizer, prompt, max_tokens=256):
    """Generate text using mlx_lm with greedy decoding (temp=0.0)."""
    try:
        sampler = make_sampler(temp=0.0)  # Greedy for reproducibility
        text = mlx_generate(
            model, tokenizer, prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return text
    except Exception as e:
        log(f"  WARNING: generation failed: {e}")
        return ""


# ============================================================================
# Evaluation functions
# ============================================================================

def eval_gsm8k(model_name, model, tokenizer, problems, format_fn):
    """Evaluate on GSM8K problems."""
    log(f"\n  [GSM8K] Evaluating {model_name}...")
    t0 = time.time()
    correct = 0
    total = len(problems)

    for i, prob in enumerate(problems):
        prompt = format_fn(prob["question"])
        generated = generate_text(model, tokenizer, prompt, max_tokens=MAX_NEW_TOKENS_GSM8K)
        predicted = extract_gsm8k_answer(generated)
        is_correct = check_gsm8k_correct(predicted, prob["answer"])
        if is_correct:
            correct += 1
        if (i + 1) % 10 == 0:
            log(f"    {i+1}/{total}: {correct}/{i+1} correct ({100*correct/(i+1):.1f}%)")
        if (i + 1) % 25 == 0:
            gc.collect()

    accuracy = correct / total if total > 0 else 0
    elapsed = time.time() - t0
    log(f"  [GSM8K] {model_name}: {correct}/{total} = {accuracy:.3f} ({elapsed:.1f}s)")
    return {"accuracy": accuracy, "correct": correct, "total": total, "time_s": elapsed}


def eval_mmlu(model_name, model, tokenizer, mmlu_data, format_fn):
    """Evaluate on MMLU questions per domain."""
    log(f"\n  [MMLU] Evaluating {model_name}...")
    results = {}
    choice_labels = ["A", "B", "C", "D"]

    for domain in DOMAINS:
        questions = mmlu_data[domain]
        if not questions:
            results[domain] = {"accuracy": 0, "correct": 0, "total": 0}
            continue

        correct = 0
        total = len(questions)

        for i, q in enumerate(questions):
            prompt = format_fn(q["question"], q["choices"])
            generated = generate_text(model, tokenizer, prompt, max_tokens=MAX_NEW_TOKENS_MMLU)
            predicted = extract_mmlu_answer(generated)
            gt_label = choice_labels[q["answer"]]
            if predicted == gt_label:
                correct += 1

        accuracy = correct / total if total > 0 else 0
        results[domain] = {"accuracy": accuracy, "correct": correct, "total": total}
        log(f"    MMLU {domain}: {correct}/{total} = {accuracy:.3f}")

    return results


# ============================================================================
# Phase functions (each self-contained per CODING_GUIDELINES)
# ============================================================================

def phase_load_data():
    """Load all benchmark data."""
    log("\n" + "=" * 70)
    log("LOADING BENCHMARK DATA")
    log("=" * 70)
    gsm8k = load_gsm8k_data(GSM8K_N)
    mmlu = load_mmlu_data(MMLU_N_PER_DOMAIN)
    return gsm8k, mmlu


def phase_eval_bitnet_base(gsm8k, mmlu):
    """Evaluate BitNet-2B base model (no adapters)."""
    log("\n" + "=" * 70)
    log("EVALUATING: BitNet-2B-4T BASE (no adapters)")
    log("=" * 70)
    t0 = time.time()
    mx.reset_peak_memory()

    model, tokenizer = load(MODEL_ID_BITNET)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("bitnet-base-loaded")

    gsm8k_results = eval_gsm8k("bitnet-base", model, tokenizer, gsm8k, format_gsm8k_prompt_generic)
    mmlu_results = eval_mmlu("bitnet-base", model, tokenizer, mmlu, format_mmlu_prompt_generic)

    peak_mem = mx.get_peak_memory() / 1e9
    elapsed = time.time() - t0
    log(f"\nBitNet base total: {elapsed:.1f}s, peak memory: {peak_mem:.2f}GB")

    results = {
        "gsm8k": gsm8k_results,
        "mmlu": mmlu_results,
        "peak_memory_gb": peak_mem,
        "time_s": elapsed,
    }
    cleanup(model, tokenizer)
    return results


def phase_eval_bitnet_sole(gsm8k, mmlu):
    """Evaluate BitNet-2B + SOLE (uniform 1/N pre-merge composition)."""
    log("\n" + "=" * 70)
    log("EVALUATING: BitNet-2B-4T + SOLE (uniform 1/N pre-merge)")
    log("=" * 70)
    t0 = time.time()
    mx.reset_peak_memory()

    model, tokenizer = load(MODEL_ID_BITNET)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("bitnet-sole-loaded-base")

    # Pre-merge all 5 adapters with uniform 1/N weighting
    skeleton = load_skeleton()
    adapters = load_all_adapters()
    domain_weights = {d: 1.0 / N_DOMAINS for d in DOMAINS}
    model = premerge_adapters_into_model(model, skeleton, adapters, domain_weights)
    del skeleton, adapters
    gc.collect()
    mx.clear_cache()
    log_memory("bitnet-sole-merged")

    gsm8k_results = eval_gsm8k("bitnet-sole", model, tokenizer, gsm8k, format_gsm8k_prompt_generic)
    mmlu_results = eval_mmlu("bitnet-sole", model, tokenizer, mmlu, format_mmlu_prompt_generic)

    peak_mem = mx.get_peak_memory() / 1e9
    elapsed = time.time() - t0
    log(f"\nBitNet SOLE total: {elapsed:.1f}s, peak memory: {peak_mem:.2f}GB")

    results = {
        "gsm8k": gsm8k_results,
        "mmlu": mmlu_results,
        "peak_memory_gb": peak_mem,
        "time_s": elapsed,
    }
    cleanup(model, tokenizer)
    return results


def phase_eval_competitor(model_id, model_name, gsm8k, mmlu, gsm8k_fmt, mmlu_fmt):
    """Evaluate a competitor model."""
    log("\n" + "=" * 70)
    log(f"EVALUATING: {model_name} ({model_id})")
    log("=" * 70)
    t0 = time.time()
    mx.reset_peak_memory()

    try:
        model, tokenizer = load(model_id)
        log_memory(f"{model_name}-loaded")
    except Exception as e:
        log(f"  FAILED to load {model_name}: {e}")
        log(f"  Downloading may take a few minutes...")
        try:
            model, tokenizer = load(model_id)
            log_memory(f"{model_name}-loaded-retry")
        except Exception as e2:
            log(f"  FAILED to load {model_name} after retry: {e2}")
            return None

    gsm8k_results = eval_gsm8k(model_name, model, tokenizer, gsm8k, gsm8k_fmt)
    mmlu_results = eval_mmlu(model_name, model, tokenizer, mmlu, mmlu_fmt)

    peak_mem = mx.get_peak_memory() / 1e9
    elapsed = time.time() - t0
    log(f"\n{model_name} total: {elapsed:.1f}s, peak memory: {peak_mem:.2f}GB")

    results = {
        "gsm8k": gsm8k_results,
        "mmlu": mmlu_results,
        "peak_memory_gb": peak_mem,
        "time_s": elapsed,
    }
    cleanup(model, tokenizer)
    return results


# ============================================================================
# Analysis
# ============================================================================

def analyze_results(all_results):
    """Compare all models and evaluate kill/success criteria."""
    log("\n" + "=" * 70)
    log("ANALYSIS")
    log("=" * 70)

    # Extract accuracy per benchmark per model
    benchmarks = ["gsm8k", "mmlu_medical", "mmlu_code", "mmlu_math", "mmlu_legal", "mmlu_finance"]
    model_scores = {}

    for model_name, res in all_results.items():
        if res is None:
            continue
        scores = {}
        scores["gsm8k"] = res["gsm8k"]["accuracy"]
        for domain in DOMAINS:
            scores[f"mmlu_{domain}"] = res["mmlu"][domain]["accuracy"]
        model_scores[model_name] = scores

    # Print comparison table
    log("\n  ACCURACY COMPARISON:")
    header = f"  {'Benchmark':<15}"
    for name in model_scores:
        header += f"  {name:<18}"
    log(header)
    log("  " + "-" * (15 + 20 * len(model_scores)))

    for bench in benchmarks:
        row = f"  {bench:<15}"
        for name in model_scores:
            val = model_scores[name].get(bench, None)
            if val is not None:
                row += f"  {val:<18.3f}"
            else:
                row += f"  {'N/A':<18}"
        log(row)

    # Memory comparison
    log("\n  MEMORY COMPARISON:")
    for name, res in all_results.items():
        if res is not None:
            log(f"  {name}: {res['peak_memory_gb']:.2f} GB peak")

    # Kill criteria evaluation
    analysis = {"model_scores": model_scores, "kill_criteria": {}, "success_criteria": {}}

    if "bitnet_sole" in model_scores and "qwen25_3b" in model_scores:
        sole = model_scores["bitnet_sole"]
        qwen = model_scores["qwen25_3b"]

        # K1: Composed worse than Qwen2.5-3B on > 60% benchmarks
        worse_count = sum(1 for b in benchmarks if sole.get(b, 0) < qwen.get(b, 0))
        total_benchmarks = len(benchmarks)
        k1_fail = worse_count > total_benchmarks * 0.6
        analysis["kill_criteria"]["k1_512"] = {
            "result": "fail" if k1_fail else "pass",
            "evidence": f"SOLE worse on {worse_count}/{total_benchmarks} benchmarks (threshold >60% = >{total_benchmarks * 0.6:.0f})",
            "worse_benchmarks": [b for b in benchmarks if sole.get(b, 0) < qwen.get(b, 0)],
            "better_benchmarks": [b for b in benchmarks if sole.get(b, 0) >= qwen.get(b, 0)],
        }
        log(f"\n  K1 (#512): SOLE worse on {worse_count}/{total_benchmarks} vs Qwen2.5-3B -> {'KILL' if k1_fail else 'PASS'}")

    if "bitnet_sole" in model_scores and "bitnet_base" in model_scores:
        sole = model_scores["bitnet_sole"]
        base = model_scores["bitnet_base"]

        # K2: Composed worse than base on ANY benchmark
        worse_any = [b for b in benchmarks if sole.get(b, 0) < base.get(b, 0)]
        k2_fail = len(worse_any) > 0
        analysis["kill_criteria"]["k2_513"] = {
            "result": "fail" if k2_fail else "pass",
            "evidence": f"SOLE worse than base on {len(worse_any)}/{len(benchmarks)} benchmarks: {worse_any}",
        }
        log(f"  K2 (#513): SOLE worse than base on {len(worse_any)} benchmarks: {worse_any} -> {'KILL' if k2_fail else 'PASS'}")

    if "bitnet_sole" in all_results and all_results["bitnet_sole"] is not None:
        sole_mem = all_results["bitnet_sole"]["peak_memory_gb"]
        qwen_mem = all_results.get("qwen25_3b", {})
        if isinstance(qwen_mem, dict) and qwen_mem is not None:
            qwen_mem_val = qwen_mem.get("peak_memory_gb", float("inf"))
        else:
            qwen_mem_val = float("inf")

        # K3: Memory exceeds Qwen2.5-3B
        k3_fail = sole_mem > qwen_mem_val
        analysis["kill_criteria"]["k3_514"] = {
            "result": "fail" if k3_fail else "pass",
            "evidence": f"SOLE {sole_mem:.2f}GB vs Qwen {qwen_mem_val:.2f}GB",
        }
        log(f"  K3 (#514): Memory SOLE={sole_mem:.2f}GB vs Qwen={qwen_mem_val:.2f}GB -> {'KILL' if k3_fail else 'PASS'}")

    # S1: Beats Qwen on >= 3/5 domain benchmarks at < 3GB
    if "bitnet_sole" in model_scores and "qwen25_3b" in model_scores:
        sole = model_scores["bitnet_sole"]
        qwen = model_scores["qwen25_3b"]
        domain_benchmarks = [f"mmlu_{d}" for d in DOMAINS]
        beats_count = sum(1 for b in domain_benchmarks if sole.get(b, 0) > qwen.get(b, 0))
        sole_mem = all_results["bitnet_sole"]["peak_memory_gb"]
        s1_pass = beats_count >= 3 and sole_mem < 3.0
        analysis["success_criteria"]["s1_40"] = {
            "result": "pass" if s1_pass else "fail",
            "evidence": f"Beats Qwen on {beats_count}/5 domains, memory={sole_mem:.2f}GB",
        }
        log(f"  S1 (#40): Beats Qwen on {beats_count}/5 domains at {sole_mem:.2f}GB -> {'PASS' if s1_pass else 'FAIL'}")

    return analysis


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log_memory("start")

    # Phase 0: Load data
    gsm8k, mmlu = phase_load_data()

    all_results = {}

    # Phase 1: BitNet base
    all_results["bitnet_base"] = phase_eval_bitnet_base(gsm8k, mmlu)

    # Phase 2: BitNet + SOLE
    all_results["bitnet_sole"] = phase_eval_bitnet_sole(gsm8k, mmlu)

    # Phase 3: Qwen2.5-3B (primary competitor)
    all_results["qwen25_3b"] = phase_eval_competitor(
        MODEL_ID_QWEN, "Qwen2.5-3B-4bit", gsm8k, mmlu,
        format_gsm8k_prompt_chatml, format_mmlu_prompt_chatml,
    )

    # Phase 4: Gemma-2-2B (size-matched)
    all_results["gemma2_2b"] = phase_eval_competitor(
        MODEL_ID_GEMMA, "Gemma-2-2B-4bit", gsm8k, mmlu,
        format_gsm8k_prompt_gemma, format_mmlu_prompt_gemma,
    )

    # Phase 5: Analysis
    analysis = analyze_results(all_results)

    # Save results
    total_time = time.time() - t0
    output = {
        "experiment": "competitive_benchmark",
        "models": {
            "bitnet_base": {"id": MODEL_ID_BITNET, "description": "BitNet-2B-4T base, no adapters"},
            "bitnet_sole": {"id": MODEL_ID_BITNET, "description": "BitNet-2B-4T + 5 domain adapters, uniform 1/N pre-merge"},
            "qwen25_3b": {"id": MODEL_ID_QWEN, "description": "Qwen2.5-3B-Instruct 4-bit quantized"},
            "gemma2_2b": {"id": MODEL_ID_GEMMA, "description": "Gemma-2-2B-IT 4-bit quantized"},
        },
        "results": all_results,
        "analysis": analysis,
        "params": {
            "gsm8k_n": GSM8K_N,
            "mmlu_n_per_domain": MMLU_N_PER_DOMAIN,
            "max_new_tokens_gsm8k": MAX_NEW_TOKENS_GSM8K,
            "max_new_tokens_mmlu": MAX_NEW_TOKENS_MMLU,
            "temperature": 0.0,
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
            "composition": "uniform 1/N pre-merge",
            "n_adapters": N_DOMAINS,
        },
        "total_time_s": total_time,
    }

    RESULTS_FILE.write_text(json.dumps(output, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Print summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    for k, v in analysis.get("kill_criteria", {}).items():
        log(f"  {k}: {v['result'].upper()} - {v['evidence']}")
    for k, v in analysis.get("success_criteria", {}).items():
        log(f"  {k}: {v['result'].upper()} - {v['evidence']}")


if __name__ == "__main__":
    main()
