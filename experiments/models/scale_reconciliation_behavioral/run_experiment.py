#!/usr/bin/env python3
"""Scale Reconciliation Behavioral Validation.

Compares behavioral quality across three LoRA scale configurations:
  1. Per-domain optimal: {medical:20, code:20, math:20, legal:4, finance:1}
  2. Uniform scale=2.0 (contrastive training optimal from Finding #246)
  3. Uniform scale=20.0 (original default)

Kill criteria:
  K1 (#654): scale=2.0 uniform produces WORSE behavioral quality than per-domain
             optimal on >=3/5 domains (>20% degradation)
  K2 (#655): scale=2.0 math domain loses >50% of the 8x behavioral gain seen
             at scale=20 (Finding #238)
  K3 (#656): scale=2.0 produces incoherent output on any domain (format quality
             collapses)

Type: guided-exploration
Platform: Apple M5 Pro 48GB, MLX
"""

import ast
import gc
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.bitlinear_layers import BitLinear
from mlx.utils import tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Paths to existing infrastructure
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]
NUM_PROMPTS_PER_DOMAIN = 10

# Three scale configurations to compare
SCALE_CONFIGS = {
    "per_domain_optimal": {
        "medical": 20.0, "code": 20.0, "math": 20.0, "legal": 4.0, "finance": 1.0,
    },
    "uniform_2.0": {
        "medical": 2.0, "code": 2.0, "math": 2.0, "legal": 2.0, "finance": 2.0,
    },
    "uniform_20.0": {
        "medical": 20.0, "code": 20.0, "math": 20.0, "legal": 20.0, "finance": 20.0,
    },
}

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
# Model utilities (from behavioral_eval_routed)
# ============================================================================

def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
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
# Pre-merge composition (from behavioral_eval_routed)
# ============================================================================

def load_skeleton():
    skeleton_path = ADAPTERS_DIR / "grassmannian_skeleton.npz"
    return dict(np.load(str(skeleton_path)))


def load_adapter(domain):
    adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
    adapter = dict(mx.load(str(adapter_path)))
    log(f"  Loaded adapter: {domain} ({len(adapter)} tensors)")
    return adapter


def premerge_single_adapter(model, skeleton, adapter, domain, scale):
    """Pre-merge a single adapter into model weights: W_new = W_base + scale * B^T @ A^T"""
    n_layers = len(model.model.layers)
    merge_count = 0
    di = DOMAINS.index(domain)

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

            skey = f"layer_{li}_{key}_domain_{di}"
            if skey not in skeleton:
                continue
            a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)

            b_key = f"model.layers.{li}.{key}.lora_b"
            if b_key not in adapter:
                continue
            b_mx = adapter[b_key]

            delta = scale * (b_mx.T @ a_mx.T)
            module.weight = module.weight + delta
            merge_count += 1

    mx.eval(model.parameters())
    log(f"  Pre-merged {domain} adapter (scale={scale}) into {merge_count} layers")
    return model


def save_base_weights(model):
    base_weights = []
    for layer in model.model.layers:
        layer_w = {}
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None and isinstance(module, nn.Linear):
                layer_w[key] = module.weight
        base_weights.append(layer_w)
    return base_weights


def restore_base_weights(model, base_weights):
    for li, layer_weights in enumerate(base_weights):
        for key, weight in layer_weights.items():
            parts = key.split(".")
            module = model.model.layers[li]
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None and isinstance(module, nn.Linear):
                module.weight = weight
    mx.eval(model.parameters())


# ============================================================================
# Generation
# ============================================================================

def format_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_text(model, tokenizer, prompt, max_tokens=128):
    try:
        sampler = make_sampler(temp=0.0)
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
# Evaluation metrics (from behavioral_eval_routed)
# ============================================================================

def eval_code_syntax(text):
    """Check if generated text contains valid Python syntax."""
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        pass
    code_blocks = re.findall(r'```(?:python)?\s*\n?(.*?)```', text, re.DOTALL)
    for block in code_blocks:
        try:
            ast.parse(block.strip())
            return True
        except SyntaxError:
            continue
    lines = text.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('def ', 'class ', 'import ', 'from ', 'for ',
                                'while ', 'if ', 'try:', 'except', 'with ',
                                'return ', 'print(', '#')):
            in_code = True
        if in_code:
            code_lines.append(line)
    if code_lines:
        try:
            ast.parse('\n'.join(code_lines))
            return True
        except SyntaxError:
            pass
    return False


def extract_math_answer(text):
    """Extract numerical answer from generated text."""
    m = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*\$?([\d,]+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(',', ''))
    matches = re.findall(r'####\s*([\d,]+(?:\.\d+)?)', text)
    if matches:
        return float(matches[-1].replace(',', ''))
    matches = re.findall(r'=\s*\$?([\d,]+(?:\.\d+)?)', text)
    if matches:
        return float(matches[-1].replace(',', ''))
    matches = re.findall(r'\$([\d,]+(?:\.\d+)?)', text)
    if matches:
        return float(matches[-1].replace(',', ''))
    matches = re.findall(r'([\d,]+(?:\.\d+)?)', text)
    if matches:
        try:
            return float(matches[-1].replace(',', ''))
        except ValueError:
            pass
    return None


def extract_ground_truth_answer(response_text):
    """Extract ground truth from training data response."""
    matches = re.findall(r'<<[^>]*?=(\d+(?:\.\d+)?)>>', response_text)
    if matches:
        return float(matches[-1])
    m = re.search(r'####\s*([\d,]+(?:\.\d+)?)', response_text)
    if m:
        return float(m.group(1).replace(',', ''))
    m = re.search(r'(?:is|=)\s*\$?([\d,]+(?:\.\d+)?)\s*$', response_text.strip(), re.IGNORECASE)
    if m:
        return float(m.group(1).replace(',', ''))
    return None


def eval_math_correct(gen_answer, gt_answer, eps=0.01):
    if gen_answer is None or gt_answer is None:
        return False
    if gt_answer == 0:
        return abs(gen_answer) < eps
    return abs(gen_answer - gt_answer) / abs(gt_answer) < eps


def extract_key_facts(text):
    """Extract key factual elements from a reference text."""
    facts = set()
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'shall', 'must', 'need', 'ought',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'mine',
        'yours', 'hers', 'ours', 'theirs', 'this', 'that', 'these', 'those',
        'who', 'whom', 'which', 'what', 'whose', 'where', 'when', 'how',
        'not', 'no', 'nor', 'but', 'and', 'or', 'so', 'if', 'then',
        'than', 'too', 'very', 'just', 'only', 'also', 'more', 'most',
        'some', 'any', 'all', 'each', 'every', 'both', 'few', 'many',
        'much', 'such', 'own', 'other', 'another', 'same', 'different',
        'about', 'after', 'again', 'against', 'at', 'before', 'between',
        'by', 'down', 'during', 'for', 'from', 'in', 'into', 'of', 'off',
        'on', 'out', 'over', 'through', 'to', 'under', 'up', 'with',
        'as', 'because', 'while', 'until', 'although', 'since', 'whether',
        'here', 'there', 'now', 'still', 'already', 'yet', 'even',
        'well', 'back', 'way', 'get', 'got', 'make', 'made', 'take',
        'took', 'go', 'went', 'come', 'came', 'see', 'saw', 'know',
        'knew', 'think', 'thought', 'say', 'said', 'give', 'gave',
        'find', 'found', 'tell', 'told', 'ask', 'asked', 'use', 'used',
        'work', 'try', 'call', 'keep', 'let', 'begin', 'seem', 'help',
        'show', 'hear', 'play', 'run', 'move', 'live', 'believe',
        'bring', 'happen', 'write', 'provide', 'sit', 'stand', 'lose',
        'pay', 'meet', 'include', 'continue', 'set', 'learn', 'change',
        'lead', 'understand', 'watch', 'follow', 'stop', 'create',
        'speak', 'read', 'allow', 'add', 'spend', 'grow', 'open',
        'walk', 'win', 'offer', 'remember', 'love', 'consider',
        'appear', 'buy', 'wait', 'serve', 'die', 'send', 'expect',
        'build', 'stay', 'fall', 'cut', 'reach', 'kill', 'remain',
        'one', 'two', 'first', 'new', 'good', 'old', 'great', 'big',
        'small', 'long', 'high', 'little', 'large', 'thing', 'things',
        'part', 'like', 'people', 'person', 'time', 'year', 'day',
        'example', 'important', 'however', 'therefore', 'thus',
        'means', 'based', 'often', 'usually', 'typically', 'generally',
        'particular', 'specific', 'certain', 'several', 'various',
        'common', 'similar', 'possible', 'likely', 'actually',
        'really', 'simply', 'especially', 'particularly',
    }
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    for w in words:
        if len(w) >= 4 and w not in stopwords:
            facts.add(w)
    number_patterns = re.findall(
        r'\b(\d+(?:\.\d+)?)\s*(%|percent|years?|months?|days?|hours?|mg|ml|kg|lb|dollars?|\$)?',
        text.lower())
    for num, unit in number_patterns:
        if unit:
            facts.add(f"{num} {unit}".strip())
        facts.add(num)
    non_stop = [w for w in words if w not in stopwords and len(w) >= 3]
    for i in range(len(non_stop) - 1):
        bigram = f"{non_stop[i]} {non_stop[i+1]}"
        facts.add(bigram)
    return facts


def eval_factual_recall(generated_text, reference_text):
    """Compute factual recall: fraction of reference facts found in generated text."""
    ref_facts = extract_key_facts(reference_text)
    gen_facts = extract_key_facts(generated_text)
    if not ref_facts:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0,
                "ref_facts": 0, "gen_facts": 0, "matched": 0}
    gen_lower = generated_text.lower()
    matched = 0
    for fact in ref_facts:
        if fact in gen_lower:
            matched += 1
    recall = matched / len(ref_facts) if ref_facts else 0.0
    ref_lower = reference_text.lower()
    gen_matched = 0
    for fact in gen_facts:
        if fact in ref_lower:
            gen_matched += 1
    precision = gen_matched / len(gen_facts) if gen_facts else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "recall": recall, "precision": precision, "f1": f1,
        "ref_facts": len(ref_facts), "gen_facts": len(gen_facts), "matched": matched,
    }


def eval_numerical_accuracy(generated_text, reference_text):
    """Extract and compare numbers between generated and reference text."""
    def extract_numbers(text):
        matches = re.findall(r'(?:\$)?([\d,]+(?:\.\d+)?)\s*(%)?', text)
        numbers = set()
        for num_str, pct in matches:
            try:
                val = float(num_str.replace(',', ''))
                numbers.add(val)
            except ValueError:
                pass
        return numbers
    ref_nums = extract_numbers(reference_text)
    gen_nums = extract_numbers(generated_text)
    if not ref_nums:
        return {"numerical_accuracy": 0.0, "ref_numbers": 0, "gen_numbers": 0, "matched": 0}
    matched = 0
    for rn in ref_nums:
        for gn in gen_nums:
            if rn == 0:
                if abs(gn) < 0.01:
                    matched += 1
                    break
            elif abs(gn - rn) / abs(rn) < 0.01:
                matched += 1
                break
    accuracy = matched / len(ref_nums) if ref_nums else 0.0
    return {
        "numerical_accuracy": accuracy,
        "ref_numbers": len(ref_nums), "gen_numbers": len(gen_nums), "matched": matched,
    }


def evaluate_response(generated_text, reference_text, domain):
    """Evaluate a single generated response using domain-appropriate metrics."""
    result = {"domain": domain, "generated_len": len(generated_text)}

    if domain == "code":
        syntax_ok = eval_code_syntax(generated_text)
        factual = eval_factual_recall(generated_text, reference_text)
        score = 0.7 * (1.0 if syntax_ok else 0.0) + 0.3 * factual["recall"]
        result.update({
            "score": score, "syntax_valid": syntax_ok,
            "factual_recall": factual["recall"], "factual_f1": factual["f1"],
            "method": "syntax_parse + factual_recall",
        })

    elif domain == "math":
        gen_answer = extract_math_answer(generated_text)
        gt_answer = extract_ground_truth_answer(reference_text)
        correct = eval_math_correct(gen_answer, gt_answer)
        score = 1.0 if correct else 0.0
        result.update({
            "score": score, "answer_correct": correct,
            "gen_answer": gen_answer, "gt_answer": gt_answer,
            "method": "numerical_answer_match (eps=0.01)",
        })

    elif domain == "medical":
        factual = eval_factual_recall(generated_text, reference_text)
        score = factual["recall"]
        result.update({
            "score": score, "factual_recall": factual["recall"],
            "factual_precision": factual["precision"], "factual_f1": factual["f1"],
            "method": "factual_recall (medical facts vs reference)",
        })

    elif domain == "legal":
        factual = eval_factual_recall(generated_text, reference_text)
        score = factual["recall"]
        result.update({
            "score": score, "factual_recall": factual["recall"],
            "factual_precision": factual["precision"], "factual_f1": factual["f1"],
            "method": "factual_recall (legal facts vs reference)",
        })

    elif domain == "finance":
        factual = eval_factual_recall(generated_text, reference_text)
        numerical = eval_numerical_accuracy(generated_text, reference_text)
        num_weight = 0.4 if numerical["ref_numbers"] > 0 else 0.0
        fact_weight = 1.0 - num_weight
        score = fact_weight * factual["recall"] + num_weight * numerical["numerical_accuracy"]
        result.update({
            "score": score, "factual_recall": factual["recall"],
            "factual_f1": factual["f1"],
            "numerical_accuracy": numerical["numerical_accuracy"],
            "method": "factual_recall + numerical_accuracy",
        })

    return result


# ============================================================================
# Format quality check (for K3: incoherent output detection)
# ============================================================================

def check_format_quality(text, domain):
    """Check if generated text has minimum format quality (not gibberish)."""
    if len(text.strip()) < 10:
        return False, "too_short"
    # Check for repetition (sign of degenerate generation)
    words = text.split()
    if len(words) > 5:
        word_counts = Counter(words)
        most_common_count = word_counts.most_common(1)[0][1]
        if most_common_count > len(words) * 0.5:
            return False, "repetitive"
    # Check for non-ASCII garbage
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    if ascii_ratio < 0.8:
        return False, "non_ascii"
    return True, "ok"


# ============================================================================
# Data loading
# ============================================================================

def extract_prompts_with_answers(domain, n_prompts=10):
    val_path = DATA_DIR / domain / "valid.jsonl"
    prompts = []
    with open(val_path) as f:
        for line in f:
            text = json.loads(line)["text"]
            if "### Instruction:" in text and "### Response:" in text:
                instruction = text.split("### Instruction:")[1].split("### Response:")[0].strip()
                response = text.split("### Response:")[1].strip()
                prompts.append({"instruction": instruction, "response": response})
            if len(prompts) >= n_prompts:
                break
    return prompts


# ============================================================================
# Phase: Generate with base model
# ============================================================================

def phase_generate_base(prompts_by_domain):
    log("\n" + "=" * 70)
    log("PHASE: GENERATE WITH BASE MODEL")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    mx.random.seed(SEED)
    np.random.seed(SEED)

    results = {}
    for domain in DOMAINS:
        domain_results = []
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            formatted = format_prompt(prompt_data["instruction"])
            generated = generate_text(model, tokenizer, formatted,
                                      max_tokens=MAX_NEW_TOKENS)
            domain_results.append(generated)
            log(f"  [{domain}][{i}] generated {len(generated)} chars")
        results[domain] = domain_results
        log(f"  {domain}: {len(domain_results)} generations complete")

    elapsed = time.time() - t0
    del model, tokenizer
    cleanup()
    log_memory("post-base-gen")
    log(f"  Base generation: {elapsed:.1f}s")
    return results, elapsed


# ============================================================================
# Phase: Generate with a specific scale configuration
# ============================================================================

def phase_generate_scaled(prompts_by_domain, scale_config, config_name):
    """Generate with oracle-routed adapters at the given scale configuration.

    Loads model ONCE, saves base weights, then for each domain:
      1. Restore base weights
      2. Pre-merge domain adapter at the configured scale
      3. Generate for that domain's prompts
    """
    log("\n" + "=" * 70)
    log(f"PHASE: GENERATE WITH {config_name.upper()}")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    base_weights = save_base_weights(model)
    log("  Saved base weights for restoration")

    skeleton = load_skeleton()
    log(f"  Loaded Grassmannian skeleton ({len(skeleton)} tensors)")

    mx.random.seed(SEED)
    np.random.seed(SEED)

    results = {}
    for domain in DOMAINS:
        restore_base_weights(model, base_weights)

        adapter = load_adapter(domain)
        scale = scale_config[domain]
        model = premerge_single_adapter(model, skeleton, adapter, domain, scale)
        del adapter

        domain_results = []
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            formatted = format_prompt(prompt_data["instruction"])
            generated = generate_text(model, tokenizer, formatted,
                                      max_tokens=MAX_NEW_TOKENS)
            domain_results.append(generated)
            log(f"  [{domain}][{i}] s={scale} generated {len(generated)} chars")
        results[domain] = domain_results
        log(f"  {domain} (s={scale}): {len(domain_results)} generations complete")

    elapsed = time.time() - t0
    del model, tokenizer, skeleton, base_weights
    cleanup()
    log_memory(f"post-{config_name}-gen")
    log(f"  {config_name} generation: {elapsed:.1f}s")
    return results, elapsed


# ============================================================================
# Phase: Evaluate all generations
# ============================================================================

def phase_evaluate(prompts_by_domain, generations_by_config):
    """Evaluate all scale configs against the behavioral framework."""
    log("\n" + "=" * 70)
    log("PHASE: BEHAVIORAL EVALUATION")
    log("=" * 70)
    t0 = time.time()

    all_evals = {}
    for config_name, generations in generations_by_config.items():
        config_evals = {}
        for domain in DOMAINS:
            domain_evals = []
            for i, (prompt_data, gen_text) in enumerate(zip(
                    prompts_by_domain[domain], generations[domain])):
                result = evaluate_response(gen_text, prompt_data["response"], domain)
                result["prompt"] = prompt_data["instruction"][:100]
                result["generated_preview"] = gen_text[:200]

                # Format quality check
                fmt_ok, fmt_reason = check_format_quality(gen_text, domain)
                result["format_quality_ok"] = fmt_ok
                result["format_quality_reason"] = fmt_reason

                domain_evals.append(result)
            config_evals[domain] = domain_evals

            scores = [r["score"] for r in domain_evals]
            mean_score = float(np.mean(scores)) if scores else 0.0
            log(f"  [{config_name}][{domain}] mean={mean_score:.4f}")

        all_evals[config_name] = config_evals

    elapsed = time.time() - t0
    log(f"\n  Evaluation: {elapsed:.1f}s")
    return all_evals, elapsed


# ============================================================================
# Main orchestrator
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("SCALE RECONCILIATION: BEHAVIORAL VALIDATION")
    log("=" * 70)
    log(f"Model: {MODEL_ID}")
    log(f"Domains: {DOMAINS}")
    log(f"Prompts/domain: {NUM_PROMPTS_PER_DOMAIN}")
    log(f"Max tokens: {MAX_NEW_TOKENS}")
    log(f"Scale configs: {list(SCALE_CONFIGS.keys())}")
    for name, cfg in SCALE_CONFIGS.items():
        log(f"  {name}: {cfg}")
    log_memory("start")

    # Load prompts
    log("\nLoading evaluation prompts...")
    prompts_by_domain = {}
    for domain in DOMAINS:
        prompts = extract_prompts_with_answers(domain, NUM_PROMPTS_PER_DOMAIN)
        prompts_by_domain[domain] = prompts
        log(f"  {domain}: {len(prompts)} prompts loaded")

    # Generate with base model
    base_gen, base_time = phase_generate_base(prompts_by_domain)

    # Generate with each scale configuration
    config_generations = {"base": base_gen}
    config_times = {"base": base_time}

    for config_name, scale_config in SCALE_CONFIGS.items():
        gen, gen_time = phase_generate_scaled(
            prompts_by_domain, scale_config, config_name)
        config_generations[config_name] = gen
        config_times[config_name] = gen_time

    # Evaluate all
    all_evals, eval_time = phase_evaluate(prompts_by_domain, config_generations)

    # ============================================================================
    # Compute comparison tables
    # ============================================================================
    log("\n" + "=" * 70)
    log("COMPARISON TABLE")
    log("=" * 70)

    comparison = {}
    for domain in DOMAINS:
        domain_comp = {}
        for config_name in config_generations:
            scores = [r["score"] for r in all_evals[config_name][domain]]
            mean_score = float(np.mean(scores)) if scores else 0.0
            fmt_ok_count = sum(1 for r in all_evals[config_name][domain]
                               if r.get("format_quality_ok", True))

            entry = {
                "mean_score": round(mean_score, 4),
                "scores": [round(s, 4) for s in scores],
                "format_ok_count": fmt_ok_count,
                "n_samples": len(scores),
            }

            # Domain-specific details
            if domain == "math":
                entry["correct_count"] = sum(
                    1 for r in all_evals[config_name][domain]
                    if r.get("answer_correct", False))
            elif domain == "code":
                entry["syntax_valid_count"] = sum(
                    1 for r in all_evals[config_name][domain]
                    if r.get("syntax_valid", False))

            domain_comp[config_name] = entry
        comparison[domain] = domain_comp

    # Print comparison
    log(f"\n{'Domain':<10} {'Base':>8} {'PerDom':>8} {'Uni2.0':>8} {'Uni20.0':>8}")
    log("-" * 50)
    for domain in DOMAINS:
        base_s = comparison[domain]["base"]["mean_score"]
        pd_s = comparison[domain]["per_domain_optimal"]["mean_score"]
        u2_s = comparison[domain]["uniform_2.0"]["mean_score"]
        u20_s = comparison[domain]["uniform_20.0"]["mean_score"]
        log(f"{domain:<10} {base_s:>8.3f} {pd_s:>8.3f} {u2_s:>8.3f} {u20_s:>8.3f}")

    # ============================================================================
    # Kill criteria assessment
    # ============================================================================
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    # K1 (#654): scale=2.0 WORSE than per-domain optimal on >=3/5 domains (>20% degradation)
    k1_worse_count = 0
    k1_details = {}
    for domain in DOMAINS:
        pd_score = comparison[domain]["per_domain_optimal"]["mean_score"]
        u2_score = comparison[domain]["uniform_2.0"]["mean_score"]

        if pd_score > 0.001:
            degradation = (pd_score - u2_score) / pd_score
        else:
            degradation = 0.0 if u2_score >= pd_score else 1.0

        worse = degradation > 0.20  # >20% degradation
        if worse:
            k1_worse_count += 1

        k1_details[domain] = {
            "per_domain_score": round(pd_score, 4),
            "uniform_2_score": round(u2_score, 4),
            "degradation_pct": round(degradation * 100, 1),
            "worse": worse,
        }
        log(f"  K1 {domain}: pd={pd_score:.3f} u2={u2_score:.3f} "
            f"degradation={degradation*100:.1f}% {'WORSE' if worse else 'OK'}")

    k1_pass = k1_worse_count < 3
    log(f"\n  K1 (#654): {'PASS' if k1_pass else 'FAIL'} -- "
        f"s=2.0 worse on {k1_worse_count}/5 domains (threshold: >=3 = FAIL)")

    # K2 (#655): scale=2.0 math domain loses >50% of the 8x behavioral gain
    # Finding #238: math base=0.10, per-domain=0.80 (gain=0.70)
    # K2 checks if s=2.0 retains at least 50% of that gain (>=0.45)
    math_base = comparison["math"]["base"]["mean_score"]
    math_pd = comparison["math"]["per_domain_optimal"]["mean_score"]
    math_u2 = comparison["math"]["uniform_2.0"]["mean_score"]
    math_gain_pd = math_pd - math_base
    math_gain_u2 = math_u2 - math_base
    math_gain_retained = math_gain_u2 / math_gain_pd if math_gain_pd > 0 else 0.0

    k2_pass = math_gain_retained >= 0.50  # retains >=50% of the gain
    log(f"\n  K2 (#655): {'PASS' if k2_pass else 'FAIL'} -- "
        f"math gain retained: {math_gain_retained*100:.1f}%")
    log(f"    base={math_base:.3f}, per_domain(s=20)={math_pd:.3f}, "
        f"uniform_2={math_u2:.3f}")
    log(f"    gain_pd={math_gain_pd:.3f}, gain_u2={math_gain_u2:.3f}")

    # K3 (#656): scale=2.0 produces incoherent output on ANY domain
    k3_incoherent_domains = []
    for domain in DOMAINS:
        fmt_ok = comparison[domain]["uniform_2.0"]["format_ok_count"]
        total = comparison[domain]["uniform_2.0"]["n_samples"]
        fmt_ratio = fmt_ok / total if total > 0 else 0
        if fmt_ratio < 0.5:  # <50% format quality = collapsed
            k3_incoherent_domains.append(domain)
        log(f"  K3 {domain}: format_ok={fmt_ok}/{total} ({fmt_ratio*100:.0f}%)")

    k3_pass = len(k3_incoherent_domains) == 0
    log(f"\n  K3 (#656): {'PASS' if k3_pass else 'FAIL'} -- "
        f"incoherent domains: {k3_incoherent_domains if k3_incoherent_domains else 'none'}")

    # ============================================================================
    # Additional analysis: compare uniform_2.0 vs uniform_20.0
    # ============================================================================
    log("\n" + "=" * 70)
    log("SCALE 2.0 vs 20.0 COMPARISON")
    log("=" * 70)

    u2_vs_u20 = {}
    for domain in DOMAINS:
        u2_score = comparison[domain]["uniform_2.0"]["mean_score"]
        u20_score = comparison[domain]["uniform_20.0"]["mean_score"]
        delta = u2_score - u20_score
        u2_vs_u20[domain] = {
            "u2": round(u2_score, 4),
            "u20": round(u20_score, 4),
            "delta": round(delta, 4),
            "u2_better": delta > 0.02,
            "u20_better": delta < -0.02,
        }
        label = "u2 BETTER" if delta > 0.02 else ("u20 BETTER" if delta < -0.02 else "NEUTRAL")
        log(f"  {domain:<10} u2={u2_score:.3f} u20={u20_score:.3f} "
            f"delta={delta:+.3f} {label}")

    u2_better_count = sum(1 for d in u2_vs_u20.values() if d["u2_better"])
    u20_better_count = sum(1 for d in u2_vs_u20.values() if d["u20_better"])
    log(f"\n  s=2.0 better on {u2_better_count}/5 domains")
    log(f"  s=20.0 better on {u20_better_count}/5 domains")

    # ============================================================================
    # Compile results
    # ============================================================================
    results = {
        "experiment": "scale_reconciliation_behavioral",
        "description": "Behavioral quality comparison: per-domain optimal vs uniform s=2.0 vs uniform s=20.0",
        "model": MODEL_ID,
        "routing": "oracle_top1",
        "scale_configs": SCALE_CONFIGS,
        "n_domains": len(DOMAINS),
        "n_prompts_per_domain": NUM_PROMPTS_PER_DOMAIN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "comparison": comparison,
        "kill_criteria": {
            "K1_654": {
                "description": "s=2.0 WORSE than per-domain optimal on >=3/5 domains (>20% degradation)",
                "domains_worse": k1_worse_count,
                "threshold": 3,
                "details": k1_details,
                "result": "PASS" if k1_pass else "FAIL",
            },
            "K2_655": {
                "description": "s=2.0 math domain loses >50% of behavioral gain from s=20",
                "math_base": round(math_base, 4),
                "math_per_domain": round(math_pd, 4),
                "math_uniform_2": round(math_u2, 4),
                "gain_per_domain": round(math_gain_pd, 4),
                "gain_uniform_2": round(math_gain_u2, 4),
                "gain_retained_pct": round(math_gain_retained * 100, 1),
                "result": "PASS" if k2_pass else "FAIL",
            },
            "K3_656": {
                "description": "s=2.0 produces incoherent output on any domain",
                "incoherent_domains": k3_incoherent_domains,
                "result": "PASS" if k3_pass else "FAIL",
            },
        },
        "u2_vs_u20": u2_vs_u20,
        "predictions_vs_measured": {
            "P1_format_preserved": {
                "predicted": "s=2.0 produces coherent output on all 5 domains (LIMA hypothesis)",
                "measured": k3_incoherent_domains,
                "match": k3_pass,
            },
            "P2_math_degrades": {
                "predicted": "math at s=2.0 scores 0.10-0.30 (below half of s=20's 0.80)",
                "measured": round(math_u2, 4),
                "match": math_u2 < 0.40,
            },
            "P3_knowledge_preserved": {
                "predicted": "legal/finance at s=2.0 >= per-domain optimal",
                "legal_u2": comparison["legal"]["uniform_2.0"]["mean_score"],
                "legal_pd": comparison["legal"]["per_domain_optimal"]["mean_score"],
                "finance_u2": comparison["finance"]["uniform_2.0"]["mean_score"],
                "finance_pd": comparison["finance"]["per_domain_optimal"]["mean_score"],
                "match": (comparison["legal"]["uniform_2.0"]["mean_score"] >=
                         comparison["legal"]["per_domain_optimal"]["mean_score"] - 0.02) and
                        (comparison["finance"]["uniform_2.0"]["mean_score"] >=
                         comparison["finance"]["per_domain_optimal"]["mean_score"] - 0.02),
            },
            "P4_u2_within_20pct": {
                "predicted": "s=2.0 within 20% of per-domain optimal on >=3/5 domains",
                "domains_within_20pct": 5 - k1_worse_count,
                "match": k1_pass,
            },
        },
        "eval_details": {
            config: {domain: evals for domain, evals in config_evals.items()}
            for config, config_evals in all_evals.items()
        },
        "timing": {
            config: round(t, 1) for config, t in config_times.items()
        } | {"eval_time_s": round(eval_time, 1), "total_time_s": round(time.time() - t0, 1)},
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"\n{'Domain':<10} {'Base':>8} {'PerDom':>8} {'Uni2.0':>8} {'Uni20.0':>8}")
    log("-" * 50)
    for domain in DOMAINS:
        base_s = comparison[domain]["base"]["mean_score"]
        pd_s = comparison[domain]["per_domain_optimal"]["mean_score"]
        u2_s = comparison[domain]["uniform_2.0"]["mean_score"]
        u20_s = comparison[domain]["uniform_20.0"]["mean_score"]
        log(f"{domain:<10} {base_s:>8.3f} {pd_s:>8.3f} {u2_s:>8.3f} {u20_s:>8.3f}")

    log(f"\n  K1 (#654): {results['kill_criteria']['K1_654']['result']} "
        f"(worse on {k1_worse_count}/5, threshold >=3)")
    log(f"  K2 (#655): {results['kill_criteria']['K2_655']['result']} "
        f"(math gain retained: {math_gain_retained*100:.1f}%)")
    log(f"  K3 (#656): {results['kill_criteria']['K3_656']['result']} "
        f"(incoherent: {k3_incoherent_domains if k3_incoherent_domains else 'none'})")

    total_time = time.time() - t0
    log(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    return results


if __name__ == "__main__":
    main()
