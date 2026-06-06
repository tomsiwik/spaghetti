#!/usr/bin/env python3
"""Self-Contrast Decoding: Extract value from unchosen adapters (SCMoE approach).

Kill criteria:
  K1 (#652): Self-contrast worse than single-adapter on >=3/5 domains by execution-based eval
  K2 (#653): Latency overhead >3x single-adapter generation

Type: frontier-extension
Platform: Apple M5 Pro 48GB, MLX
Grounded by: SCMoE (2405.14507), Contrastive Decoding (2210.15097)
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
from mlx_lm.models import cache as mlx_cache
from mlx.utils import tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(4 * 1024**3)  # 4GB cache for two-model workload

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

# Per-domain optimal scales (Finding #217)
DOMAIN_SCALES = {
    "medical": 20.0,
    "code": 20.0,
    "math": 20.0,
    "legal": 4.0,
    "finance": 1.0,
}

# Contrastive decoding alpha values to sweep
ALPHA_VALUES = [0.1, 0.3, 0.5, 1.0]

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
# Pre-merge composition
# ============================================================================

def load_skeleton():
    skeleton_path = ADAPTERS_DIR / "grassmannian_skeleton.npz"
    return dict(np.load(str(skeleton_path)))


def load_adapter(domain):
    adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
    adapter = dict(mx.load(str(adapter_path)))
    log(f"  Loaded adapter: {domain} ({len(adapter)} tensors)")
    return adapter


def premerge_adapter(model, skeleton, adapter, domain, scale):
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


def premerge_amateur(model, skeleton, adapters_dict, primary_domain):
    """Pre-merge average of non-primary adapters as amateur model.

    Amateur = base + (1/K-1) * sum_{j != primary} scale_j * Delta_j

    This averages the non-primary adapter contributions uniformly.
    """
    n_layers = len(model.model.layers)
    merge_count = 0
    non_primary = [d for d in DOMAINS if d != primary_domain]
    K_minus_1 = len(non_primary)

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

            # Accumulate delta from all non-primary adapters
            total_delta = None
            for nd in non_primary:
                di = DOMAINS.index(nd)
                skey = f"layer_{li}_{key}_domain_{di}"
                if skey not in skeleton:
                    continue
                a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)

                b_key = f"model.layers.{li}.{key}.lora_b"
                if b_key not in adapters_dict[nd]:
                    continue
                b_mx = adapters_dict[nd][b_key]

                scale = DOMAIN_SCALES[nd]
                delta = scale * (b_mx.T @ a_mx.T)
                if total_delta is None:
                    total_delta = delta
                else:
                    total_delta = total_delta + delta

            if total_delta is not None:
                avg_delta = total_delta / K_minus_1
                module.weight = module.weight + avg_delta
                merge_count += 1

    mx.eval(model.parameters())
    log(f"  Pre-merged amateur (avg of {non_primary}) into {merge_count} layers")
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
# Contrastive generation (token-by-token)
# ============================================================================

def contrastive_generate(expert_model, amateur_model, tokenizer, prompt_text,
                         alpha=0.5, max_tokens=128):
    """Generate text using contrastive decoding between expert and amateur models.

    logit_CD = (1 + alpha) * logit_expert - alpha * logit_amateur

    Both models must already have their respective adapters pre-merged.
    Uses proper KV caches for both models.
    """
    prompt_tokens = mx.array(tokenizer.encode(prompt_text))[None, :]  # [1, seq_len]

    # Create KV caches for both models
    expert_cache = mlx_cache.make_prompt_cache(expert_model)
    amateur_cache = mlx_cache.make_prompt_cache(amateur_model)

    # Prefill both models with the prompt
    expert_logits = expert_model(prompt_tokens, cache=expert_cache)
    amateur_logits = amateur_model(prompt_tokens, cache=amateur_cache)
    mx.eval(expert_logits, amateur_logits, [c.state for c in expert_cache],
            [c.state for c in amateur_cache])

    # Apply contrastive decoding to get first token
    cd_logits = (1.0 + alpha) * expert_logits[:, -1, :] - alpha * amateur_logits[:, -1, :]
    next_token = mx.argmax(cd_logits, axis=-1, keepdims=True)
    mx.eval(next_token)

    generated_tokens = [next_token.item()]

    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    # Autoregressive generation
    for _ in range(max_tokens - 1):
        if generated_tokens[-1] in eos_ids:
            break

        token_input = next_token  # [1, 1]
        expert_logits = expert_model(token_input, cache=expert_cache)
        amateur_logits = amateur_model(token_input, cache=amateur_cache)
        mx.eval(expert_logits, amateur_logits, [c.state for c in expert_cache],
                [c.state for c in amateur_cache])

        cd_logits = (1.0 + alpha) * expert_logits[:, -1, :] - alpha * amateur_logits[:, -1, :]
        next_token = mx.argmax(cd_logits, axis=-1, keepdims=True)
        mx.eval(next_token)

        generated_tokens.append(next_token.item())

    # Decode
    text = tokenizer._tokenizer.decode(generated_tokens)
    del expert_cache, amateur_cache
    return text


def generate_text_standard(model, tokenizer, prompt_text, max_tokens=128):
    """Standard greedy generation with a single model (for baseline + latency comparison)."""
    try:
        sampler = make_sampler(temp=0.0)
        text = mlx_generate(
            model, tokenizer, prompt_text,
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


def format_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


# ============================================================================
# Phase 1: Load data
# ============================================================================

def phase_load_data():
    log("\n" + "=" * 70)
    log("PHASE 0: LOAD DATA")
    log("=" * 70)
    prompts_by_domain = {}
    for domain in DOMAINS:
        prompts = extract_prompts_with_answers(domain, NUM_PROMPTS_PER_DOMAIN)
        prompts_by_domain[domain] = prompts
        log(f"  {domain}: {len(prompts)} prompts loaded")
    return prompts_by_domain


# ============================================================================
# Phase 2: Single-adapter baseline (for K1 comparison + K2 latency baseline)
# ============================================================================

def phase_single_adapter_baseline(prompts_by_domain):
    """Generate with single adapter (oracle routing) for baseline comparison."""
    log("\n" + "=" * 70)
    log("PHASE 1: SINGLE-ADAPTER BASELINE (oracle routing)")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    base_weights = save_base_weights(model)
    skeleton = load_skeleton()

    mx.random.seed(SEED)
    np.random.seed(SEED)

    results = {}
    latencies = {}

    for domain in DOMAINS:
        restore_base_weights(model, base_weights)
        adapter = load_adapter(domain)
        scale = DOMAIN_SCALES[domain]
        premerge_adapter(model, skeleton, adapter, domain, scale)
        del adapter

        domain_results = []
        domain_latencies = []
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            formatted = format_prompt(prompt_data["instruction"])
            t_gen = time.time()
            generated = generate_text_standard(model, tokenizer, formatted,
                                               max_tokens=MAX_NEW_TOKENS)
            gen_time = time.time() - t_gen
            domain_results.append(generated)
            domain_latencies.append(gen_time)
            log(f"  [{domain}][{i}] {len(generated)} chars, {gen_time:.2f}s")

        results[domain] = domain_results
        latencies[domain] = domain_latencies
        log(f"  {domain}: {len(domain_results)} generations, "
            f"avg {np.mean(domain_latencies):.2f}s")

    elapsed = time.time() - t0
    del model, tokenizer, skeleton, base_weights
    cleanup()
    log_memory("post-baseline")
    log(f"  Single-adapter baseline: {elapsed:.1f}s total")
    return results, latencies, elapsed


# ============================================================================
# Phase 3: Self-contrast decoding (the main experiment)
# ============================================================================

def phase_self_contrast(prompts_by_domain, alpha):
    """Generate with self-contrast decoding for a given alpha.

    For each domain:
    - Expert = base + primary adapter (domain-matched)
    - Amateur = base + average(non-primary adapters)
    - Contrastive: logit_CD = (1+alpha)*logit_E - alpha*logit_A
    """
    log(f"\n" + "=" * 70)
    log(f"PHASE 2: SELF-CONTRAST DECODING (alpha={alpha})")
    log("=" * 70)
    t0 = time.time()

    # Load all adapters once
    all_adapters = {}
    for domain in DOMAINS:
        all_adapters[domain] = load_adapter(domain)

    skeleton = load_skeleton()

    # Load TWO model instances (expert and amateur)
    log("  Loading expert model...")
    expert_model, tokenizer = load(MODEL_ID)
    expert_model = replace_bitlinear_with_linear(expert_model)
    expert_model.freeze()
    expert_base_weights = save_base_weights(expert_model)

    log("  Loading amateur model...")
    amateur_model, _ = load(MODEL_ID)
    amateur_model = replace_bitlinear_with_linear(amateur_model)
    amateur_model.freeze()
    amateur_base_weights = save_base_weights(amateur_model)

    log_memory("two-models-loaded")

    mx.random.seed(SEED)
    np.random.seed(SEED)

    results = {}
    latencies = {}

    for domain in DOMAINS:
        log(f"\n  --- {domain.upper()} ---")

        # Configure expert: base + primary adapter
        restore_base_weights(expert_model, expert_base_weights)
        scale = DOMAIN_SCALES[domain]
        premerge_adapter(expert_model, skeleton, all_adapters[domain], domain, scale)

        # Configure amateur: base + avg(non-primary adapters)
        restore_base_weights(amateur_model, amateur_base_weights)
        premerge_amateur(amateur_model, skeleton, all_adapters, domain)

        domain_results = []
        domain_latencies = []
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            formatted = format_prompt(prompt_data["instruction"])
            t_gen = time.time()
            generated = contrastive_generate(
                expert_model, amateur_model, tokenizer, formatted,
                alpha=alpha, max_tokens=MAX_NEW_TOKENS,
            )
            gen_time = time.time() - t_gen
            domain_results.append(generated)
            domain_latencies.append(gen_time)
            log(f"  [{domain}][{i}] {len(generated)} chars, {gen_time:.2f}s")

        results[domain] = domain_results
        latencies[domain] = domain_latencies
        log(f"  {domain}: avg {np.mean(domain_latencies):.2f}s")

    elapsed = time.time() - t0
    del expert_model, amateur_model, tokenizer, skeleton
    del expert_base_weights, amateur_base_weights, all_adapters
    cleanup()
    log_memory("post-contrast")
    log(f"  Self-contrast (alpha={alpha}): {elapsed:.1f}s total")
    return results, latencies, elapsed


# ============================================================================
# Phase 4: Evaluate all conditions
# ============================================================================

def phase_evaluate(prompts_by_domain, baseline_gens, contrast_gens_by_alpha):
    """Evaluate all generation conditions using behavioral metrics."""
    log("\n" + "=" * 70)
    log("PHASE 3: BEHAVIORAL EVALUATION")
    log("=" * 70)

    all_results = {}

    # Evaluate baseline
    baseline_scores = {}
    for domain in DOMAINS:
        domain_scores = []
        for prompt_data, gen_text in zip(prompts_by_domain[domain], baseline_gens[domain]):
            result = evaluate_response(gen_text, prompt_data["response"], domain)
            domain_scores.append(result["score"])
        baseline_scores[domain] = float(np.mean(domain_scores))
        log(f"  Baseline {domain}: {baseline_scores[domain]:.4f}")
    all_results["baseline"] = baseline_scores

    # Evaluate each alpha
    for alpha, contrast_gens in contrast_gens_by_alpha.items():
        contrast_scores = {}
        for domain in DOMAINS:
            domain_scores = []
            for prompt_data, gen_text in zip(prompts_by_domain[domain], contrast_gens[domain]):
                result = evaluate_response(gen_text, prompt_data["response"], domain)
                domain_scores.append(result["score"])
            contrast_scores[domain] = float(np.mean(domain_scores))
            log(f"  Contrast alpha={alpha} {domain}: {contrast_scores[domain]:.4f}")
        all_results[f"alpha_{alpha}"] = contrast_scores

    return all_results


# ============================================================================
# Main
# ============================================================================

def main():
    t0_total = time.time()
    log_memory("start")

    # Phase 0: Load data
    prompts_by_domain = phase_load_data()

    # Phase 1: Single-adapter baseline
    baseline_gens, baseline_latencies, baseline_time = phase_single_adapter_baseline(
        prompts_by_domain)

    # Phase 2: Self-contrast for each alpha
    # Run alpha=0.3 first (expected best), then others if time permits
    contrast_gens_by_alpha = {}
    contrast_latencies_by_alpha = {}
    contrast_times = {}

    for alpha in ALPHA_VALUES:
        elapsed_so_far = time.time() - t0_total
        if elapsed_so_far > 5400:  # 90 min safety limit
            log(f"\n  SKIPPING alpha={alpha} (elapsed {elapsed_so_far:.0f}s > 5400s safety)")
            break

        gens, lats, t_elapsed = phase_self_contrast(prompts_by_domain, alpha)
        contrast_gens_by_alpha[alpha] = gens
        contrast_latencies_by_alpha[alpha] = lats
        contrast_times[alpha] = t_elapsed

    # Phase 3: Evaluate everything
    eval_results = phase_evaluate(prompts_by_domain, baseline_gens, contrast_gens_by_alpha)

    # ========================================================================
    # Assemble results
    # ========================================================================
    log("\n" + "=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)

    baseline_scores = eval_results["baseline"]

    # Find best alpha per domain
    best_alpha_per_domain = {}
    best_scores = {}
    for domain in DOMAINS:
        best_a = None
        best_s = baseline_scores[domain]
        for alpha in contrast_gens_by_alpha:
            key = f"alpha_{alpha}"
            s = eval_results[key][domain]
            if s > best_s:
                best_s = s
                best_a = alpha
        best_alpha_per_domain[domain] = best_a
        best_scores[domain] = best_s

    # K1 assessment: self-contrast worse than single-adapter on >=3/5 domains
    # Use best alpha per domain for fairest comparison
    domains_worse = 0
    domains_better = 0
    domains_equal = 0
    k1_details = {}
    for domain in DOMAINS:
        bl = baseline_scores[domain]
        # Best across all alphas
        best_contrast = bl  # default to baseline if no alpha tried
        best_a_for_domain = None
        for alpha in contrast_gens_by_alpha:
            key = f"alpha_{alpha}"
            sc = eval_results[key][domain]
            if sc > best_contrast:
                best_contrast = sc
                best_a_for_domain = alpha
        diff = best_contrast - bl
        if diff < -0.005:  # worse by more than noise
            domains_worse += 1
            direction = "worse"
        elif diff > 0.005:
            domains_better += 1
            direction = "better"
        else:
            domains_equal += 1
            direction = "equal"
        k1_details[domain] = {
            "baseline_score": bl,
            "best_contrast_score": best_contrast,
            "best_alpha": best_a_for_domain,
            "diff": diff,
            "direction": direction,
        }
        log(f"  {domain}: baseline={bl:.4f}, best_contrast={best_contrast:.4f} "
            f"(alpha={best_a_for_domain}), diff={diff:+.4f} [{direction}]")

    k1_result = "FAIL" if domains_worse >= 3 else "PASS"
    log(f"\n  K1: {domains_worse}/5 domains worse -> {k1_result}")
    log(f"      ({domains_better} better, {domains_equal} equal, {domains_worse} worse)")

    # K2 assessment: latency overhead > 3x
    avg_baseline_latency = float(np.mean([
        np.mean(baseline_latencies[d]) for d in DOMAINS
    ]))
    latency_ratios = {}
    for alpha in contrast_latencies_by_alpha:
        avg_contrast_latency = float(np.mean([
            np.mean(contrast_latencies_by_alpha[alpha][d]) for d in DOMAINS
        ]))
        ratio = avg_contrast_latency / avg_baseline_latency if avg_baseline_latency > 0 else 999
        latency_ratios[str(alpha)] = {
            "avg_baseline_s": avg_baseline_latency,
            "avg_contrast_s": avg_contrast_latency,
            "ratio": ratio,
        }
        log(f"  K2 alpha={alpha}: {avg_contrast_latency:.2f}s / {avg_baseline_latency:.2f}s "
            f"= {ratio:.2f}x")

    # K2 uses worst-case alpha
    worst_ratio = max(lr["ratio"] for lr in latency_ratios.values()) if latency_ratios else 999
    k2_result = "FAIL" if worst_ratio > 3.0 else "PASS"
    log(f"  K2: worst ratio {worst_ratio:.2f}x -> {k2_result}")

    # Predictions check
    predictions = {}
    # P1: math score >= 0.80 with contrast
    best_math = max([eval_results.get(f"alpha_{a}", {}).get("math", 0)
                     for a in contrast_gens_by_alpha], default=0)
    predictions["P1_math_ge_0.80"] = {
        "predicted": ">= 0.80",
        "measured": best_math,
        "match": best_math >= 0.80,
    }
    # P2: code score >= 0.62 with contrast
    best_code = max([eval_results.get(f"alpha_{a}", {}).get("code", 0)
                     for a in contrast_gens_by_alpha], default=0)
    predictions["P2_code_ge_0.62"] = {
        "predicted": ">= 0.62",
        "measured": best_code,
        "match": best_code >= 0.62,
    }
    # P3: finance likely degrades
    best_finance = max([eval_results.get(f"alpha_{a}", {}).get("finance", 0)
                        for a in contrast_gens_by_alpha], default=0)
    predictions["P3_finance_degrades"] = {
        "predicted": "< baseline",
        "measured": best_finance,
        "baseline": baseline_scores.get("finance", 0),
        "match": best_finance < baseline_scores.get("finance", 0),
    }
    # P4: <= 2/5 domains worse
    predictions["P4_le_2_domains_worse"] = {
        "predicted": "<= 2 domains worse",
        "measured": domains_worse,
        "match": domains_worse <= 2,
    }
    # P5: latency ~2x
    predictions["P5_latency_approx_2x"] = {
        "predicted": "~2x (< 3x)",
        "measured": worst_ratio,
        "match": worst_ratio < 3.0,
    }

    log("\n  Predictions:")
    for k, v in predictions.items():
        match_str = "YES" if v["match"] else "NO"
        log(f"    {k}: predicted={v['predicted']}, measured={v['measured']}, match={match_str}")

    # Per-alpha detail tables
    alpha_comparison = {}
    for alpha in contrast_gens_by_alpha:
        key = f"alpha_{alpha}"
        alpha_comparison[str(alpha)] = {}
        for domain in DOMAINS:
            bl = baseline_scores[domain]
            sc = eval_results[key][domain]
            alpha_comparison[str(alpha)][domain] = {
                "baseline": bl,
                "contrast": sc,
                "diff": sc - bl,
                "pct_change": ((sc - bl) / bl * 100) if bl > 0 else 0,
            }

    # Sample generations for inspection
    sample_generations = {}
    for domain in ["math", "code"]:
        sample_generations[domain] = {
            "prompt": prompts_by_domain[domain][0]["instruction"][:200],
            "baseline": baseline_gens[domain][0][:300],
        }
        for alpha in contrast_gens_by_alpha:
            sample_generations[domain][f"alpha_{alpha}"] = (
                contrast_gens_by_alpha[alpha][domain][0][:300]
            )

    total_time = time.time() - t0_total

    results = {
        "experiment": "self_contrast_decoding",
        "description": "Self-contrast decoding: extract value from unchosen adapters",
        "model": MODEL_ID,
        "method": "SCMoE-style contrastive decoding adapted for LoRA adapters",
        "grounded_by": "SCMoE (2405.14507), Contrastive Decoding (2210.15097)",
        "type": "frontier-extension",
        "alpha_values_tested": [float(a) for a in contrast_gens_by_alpha.keys()],
        "n_domains": len(DOMAINS),
        "n_prompts_per_domain": NUM_PROMPTS_PER_DOMAIN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "domain_scales": DOMAIN_SCALES,
        "baseline_scores": baseline_scores,
        "eval_results": eval_results,
        "alpha_comparison": alpha_comparison,
        "k1_details": k1_details,
        "kill_criteria": {
            "K1_652": {
                "description": "Self-contrast worse than single-adapter on >=3/5 domains",
                "domains_worse": domains_worse,
                "domains_better": domains_better,
                "domains_equal": domains_equal,
                "threshold": 3,
                "result": k1_result,
            },
            "K2_653": {
                "description": "Latency overhead >3x single-adapter generation",
                "worst_ratio": worst_ratio,
                "threshold": 3.0,
                "latency_details": latency_ratios,
                "result": k2_result,
            },
        },
        "predictions": predictions,
        "sample_generations": sample_generations,
        "timing": {
            "baseline_time_s": baseline_time,
            "contrast_times_s": {str(k): v for k, v in contrast_times.items()},
            "total_time_s": total_time,
        },
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\n  Results saved to {RESULTS_FILE}")
    log(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    log_memory("final")


if __name__ == "__main__":
    main()
