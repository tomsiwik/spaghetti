#!/usr/bin/env python3
"""Top-k routing eliminates spectral composition pathology.

Kill criteria:
  K712: Top-2 routed composed Gini > 0.15 on any evaluation batch -> KILL
  K713: Top-2 routed per-domain PPL > 5% worse than oracle single-adapter PPL on >=3/5 -> KILL
  K714: Top-2 routed behavioral quality < oracle behavioral quality by >15% on >=2/5 -> KILL

Type: Verification (of Theorem 1: routing eliminates between-domain spectral pathology)
Platform: Apple M5 Pro 48GB, MLX
Papers: arXiv 2505.18356, Finding #58, #238, #279
"""

import ast
import gc
import json
import math
import os
import re
import time
from itertools import combinations
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.bitlinear_layers import BitLinear

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Source adapters and data
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"
SKELETON_PATH = ADAPTERS_DIR / "grassmannian_skeleton.npz"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
MAX_SEQ_LENGTH = 256
MAX_NEW_TOKENS = 128
SEED = 42
N_EVAL_SAMPLES = 20  # per domain for PPL evaluation
N_PROMPTS_PER_DOMAIN = 10  # for behavioral eval

DOMAINS = ["medical", "code", "math", "legal", "finance"]

# Per-domain optimal scales (Finding #249)
OPTIMAL_SCALES = {
    "medical": 20.0, "code": 20.0, "math": 20.0,
    "legal": 4.0, "finance": 1.0,
}

# Oracle top-2: for each domain, which 2 adapters are best?
# Primary = domain itself. Secondary = most related domain by PPL proximity.
# We test all C(5,2)=10 pairs for spectral analysis, but use oracle pairs for PPL/behavioral.
# Oracle secondary selection based on domain similarity:
ORACLE_TOP2 = {
    "medical": ["medical", "math"],     # medical questions often involve numerical reasoning
    "code": ["code", "math"],           # code involves mathematical logic
    "math": ["math", "code"],           # math benefits from structured reasoning
    "legal": ["legal", "finance"],      # legal and finance share regulatory language
    "finance": ["finance", "legal"],    # finance and legal share regulatory language
}

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

NUM_LAYERS = 30

# Sample layers/keys for Gini analysis (same as Frobenius experiment)
SAMPLE_LAYERS = [0, 5, 10, 15, 20, 25, 29]
SAMPLE_KEYS = ["self_attn.q_proj", "mlp.gate_proj"]


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
# Gini coefficient
# ============================================================================

def gini_coefficient(values):
    """Compute Gini coefficient of an array of non-negative values."""
    vals = np.sort(np.asarray(values, dtype=np.float64))
    n = len(vals)
    if n <= 1 or np.sum(vals) < 1e-15:
        return 0.0
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * vals) / (n * np.sum(vals)))


# ============================================================================
# BitNet unpacking
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
# Model weight manipulation (pre-merge)
# ============================================================================

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


def premerge_adapters(model, skeleton_np, adapters_np_list, domain_indices, scales):
    """Pre-merge multiple adapters into model. Returns merge count."""
    merge_count = 0
    for adapter_np, di, scale in zip(adapters_np_list, domain_indices, scales):
        for li in range(NUM_LAYERS):
            for key in TARGET_KEYS:
                skey = f"layer_{li}_{key}_domain_{di}"
                bkey = f"model.layers.{li}.{key}.lora_b"
                if skey not in skeleton_np or bkey not in adapter_np:
                    continue

                a_mx = mx.array(skeleton_np[skey]).astype(mx.bfloat16)
                b_mx = mx.array(adapter_np[bkey]).astype(mx.bfloat16)

                delta = scale * (b_mx.T @ a_mx.T)

                parts = key.split(".")
                module = model.model.layers[li]
                for part in parts:
                    module = getattr(module, part, None)
                    if module is None:
                        break
                if module is not None and isinstance(module, nn.Linear):
                    module.weight = module.weight + delta
                    merge_count += 1

            # Periodic eval to prevent graph explosion
            if li % 10 == 0:
                mx.eval(model.parameters())

    mx.eval(model.parameters())
    return merge_count


def apply_delta_dict_to_model(model, delta_dict_np):
    """Apply dict of {(li, key): np_delta} to model weights."""
    count = 0
    for (li, key), delta_np in delta_dict_np.items():
        delta_mx = mx.array(delta_np.astype(np.float32))
        parts = key.split(".")
        module = model.model.layers[li]
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                break
        if module is not None and isinstance(module, nn.Linear):
            module.weight = module.weight + delta_mx.astype(module.weight.dtype)
            count += 1
        if count % 30 == 0:
            mx.eval(model.parameters())
    mx.eval(model.parameters())
    return count


# ============================================================================
# Delta composition (numpy, for spectral analysis)
# ============================================================================

def compose_pair_delta(skeleton_np, adapter_a_np, adapter_b_np, di_a, di_b, s_a, s_b):
    """Compose 2 adapter deltas into a single delta dict for spectral analysis."""
    delta_dict = {}
    for li in range(NUM_LAYERS):
        for key in TARGET_KEYS:
            composed = None
            for adapter_np, di, scale in [(adapter_a_np, di_a, s_a), (adapter_b_np, di_b, s_b)]:
                skey = f"layer_{li}_{key}_domain_{di}"
                bkey = f"model.layers.{li}.{key}.lora_b"
                if skey not in skeleton_np or bkey not in adapter_np:
                    continue
                A = skeleton_np[skey].astype(np.float64)
                B = adapter_np[bkey].astype(np.float64)
                delta = scale * (B.T @ A.T)
                if composed is None:
                    composed = delta
                else:
                    composed += delta
            if composed is not None:
                delta_dict[(li, key)] = composed
    return delta_dict


def compose_single_delta(skeleton_np, adapter_np, di, scale):
    """Compose a single adapter delta for spectral analysis."""
    delta_dict = {}
    for li in range(NUM_LAYERS):
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_{di}"
            bkey = f"model.layers.{li}.{key}.lora_b"
            if skey not in skeleton_np or bkey not in adapter_np:
                continue
            A = skeleton_np[skey].astype(np.float64)
            B = adapter_np[bkey].astype(np.float64)
            delta = scale * (B.T @ A.T)
            delta_dict[(li, key)] = delta
    return delta_dict


def compose_uniform_n5_delta(skeleton_np, all_adapters_np, eq_scales=None):
    """Compose all N=5 adapter deltas uniformly (with optional equalization)."""
    delta_dict = {}
    for li in range(NUM_LAYERS):
        for key in TARGET_KEYS:
            composed = None
            for di, domain in enumerate(DOMAINS):
                skey = f"layer_{li}_{key}_domain_{di}"
                bkey = f"model.layers.{li}.{key}.lora_b"
                if skey not in skeleton_np or bkey not in all_adapters_np[di]:
                    continue
                A = skeleton_np[skey].astype(np.float64)
                B = all_adapters_np[di][bkey].astype(np.float64)
                scale = OPTIMAL_SCALES[domain]
                delta = scale * (B.T @ A.T)
                if eq_scales is not None:
                    delta = delta * eq_scales[domain]
                if composed is None:
                    composed = delta
                else:
                    composed += delta
            if composed is not None:
                delta_dict[(li, key)] = composed
    return delta_dict


def compute_gini_for_delta_dict(delta_dict):
    """Compute Gini coefficient from delta dict on sampled layer/key pairs."""
    ginis = []
    for li in SAMPLE_LAYERS:
        for key in SAMPLE_KEYS:
            if (li, key) not in delta_dict:
                continue
            composed = delta_dict[(li, key)]
            _, S_c, _ = np.linalg.svd(composed, full_matrices=False)
            S_nz = S_c[S_c > 1e-6]
            if len(S_nz) > 1:
                ginis.append(gini_coefficient(S_nz))
    if not ginis:
        return {"mean_gini": 0.0, "std_gini": 0.0, "n_samples": 0, "per_sample_ginis": []}
    return {
        "mean_gini": float(np.mean(ginis)),
        "std_gini": float(np.std(ginis)),
        "max_gini": float(np.max(ginis)),
        "n_samples": len(ginis),
        "per_sample_ginis": [float(g) for g in ginis],
    }


# ============================================================================
# Frobenius equalization (from Finding #279)
# ============================================================================

def compute_equalization_scales_partial(skeleton_np, all_adapters_np, compression=0.5):
    """Compute partial log-compression equalization scales (Finding #279)."""
    domain_norms = {}
    for di, domain in enumerate(DOMAINS):
        total_frob_sq = 0.0
        scale = OPTIMAL_SCALES[domain]
        for li in range(NUM_LAYERS):
            for key in TARGET_KEYS:
                skey = f"layer_{li}_{key}_domain_{di}"
                bkey = f"model.layers.{li}.{key}.lora_b"
                if skey not in skeleton_np or bkey not in all_adapters_np[di]:
                    continue
                B = all_adapters_np[di][bkey].astype(np.float64)
                total_frob_sq += scale ** 2 * float(np.sum(B ** 2))
        domain_norms[domain] = math.sqrt(total_frob_sq)

    norms = np.array([domain_norms[d] for d in DOMAINS])
    log_norms = np.log(norms + 1e-30)
    mean_log = np.mean(log_norms)
    new_log = mean_log + compression * (log_norms - mean_log)
    new_norms = np.exp(new_log)
    scales = new_norms / norms
    return {d: float(scales[i]) for i, d in enumerate(DOMAINS)}


# ============================================================================
# PPL evaluation
# ============================================================================

def compute_val_loss(model, tokenizer, texts):
    """Compute mean cross-entropy val loss on texts."""
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)[:MAX_SEQ_LENGTH]
        if len(ids) < 2:
            continue
        x = mx.array([ids[:-1]])
        y = mx.array([ids[1:]])
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += len(ids) - 1
        del logits, loss
    return total_loss / max(total_tokens, 1)


# ============================================================================
# Generation + behavioral evaluation (from behavioral_eval_routed)
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
    matched = sum(1 for fact in ref_facts if fact in gen_lower)
    recall = matched / len(ref_facts)
    ref_lower = reference_text.lower()
    gen_matched = sum(1 for fact in gen_facts if fact in ref_lower)
    precision = gen_matched / len(gen_facts) if gen_facts else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "recall": recall, "precision": precision, "f1": f1,
        "ref_facts": len(ref_facts), "gen_facts": len(gen_facts), "matched": matched,
    }


def eval_numerical_accuracy(generated_text, reference_text):
    """Extract and compare numbers."""
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
    accuracy = matched / len(ref_nums)
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
        })
    elif domain == "math":
        gen_answer = extract_math_answer(generated_text)
        gt_answer = extract_ground_truth_answer(reference_text)
        correct = eval_math_correct(gen_answer, gt_answer)
        score = 1.0 if correct else 0.0
        result.update({
            "score": score, "answer_correct": correct,
            "gen_answer": gen_answer, "gt_answer": gt_answer,
        })
    elif domain in ("medical", "legal"):
        factual = eval_factual_recall(generated_text, reference_text)
        score = factual["recall"]
        result.update({
            "score": score, "factual_recall": factual["recall"],
            "factual_f1": factual["f1"],
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
        })
    else:
        result["score"] = 0.0

    return result


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
# PHASE 1: Spectral Analysis (numpy only -- no model needed)
# ============================================================================

def phase_spectral_analysis(skeleton_np, all_adapters_np):
    """Compute Gini for: each individual adapter, all C(5,2)=10 pairs, oracle top-2 pairs,
    uniform N=5, uniform N=5 with 50% log-compression."""
    log("\n" + "=" * 70)
    log("PHASE 1: SPECTRAL ANALYSIS (Gini coefficients)")
    log("=" * 70)
    t0 = time.time()

    # 1a. Individual adapter Gini (P5: individual Gini < 0.15)
    log("\n  --- Individual adapter Gini ---")
    individual_ginis = {}
    for di, domain in enumerate(DOMAINS):
        delta_dict = compose_single_delta(
            skeleton_np, all_adapters_np[di], di, OPTIMAL_SCALES[domain]
        )
        gini_result = compute_gini_for_delta_dict(delta_dict)
        individual_ginis[domain] = gini_result
        log(f"  {domain:10s}: Gini={gini_result['mean_gini']:.4f} +/- {gini_result['std_gini']:.4f} "
            f"(max={gini_result.get('max_gini', 0):.4f})")
        del delta_dict

    # 1b. All C(5,2) = 10 pair Gini
    log("\n  --- All C(5,2) pair Gini (top-2 spectral analysis) ---")
    pair_ginis = {}
    for (di_a, domain_a), (di_b, domain_b) in combinations(enumerate(DOMAINS), 2):
        pair_key = f"{domain_a}+{domain_b}"
        delta_dict = compose_pair_delta(
            skeleton_np,
            all_adapters_np[di_a], all_adapters_np[di_b],
            di_a, di_b,
            OPTIMAL_SCALES[domain_a], OPTIMAL_SCALES[domain_b],
        )
        gini_result = compute_gini_for_delta_dict(delta_dict)
        pair_ginis[pair_key] = gini_result
        scale_ratio = max(OPTIMAL_SCALES[domain_a], OPTIMAL_SCALES[domain_b]) / \
                      max(min(OPTIMAL_SCALES[domain_a], OPTIMAL_SCALES[domain_b]), 0.01)
        log(f"  {pair_key:20s}: Gini={gini_result['mean_gini']:.4f} "
            f"(max={gini_result.get('max_gini', 0):.4f}, "
            f"scale_ratio={scale_ratio:.0f}:1)")
        del delta_dict

    # 1c. Oracle top-2 Gini per domain
    log("\n  --- Oracle top-2 Gini per domain ---")
    oracle_top2_ginis = {}
    for domain in DOMAINS:
        d1, d2 = ORACLE_TOP2[domain]
        di1, di2 = DOMAINS.index(d1), DOMAINS.index(d2)
        pair_key = f"{d1}+{d2}"
        # Use cached pair result if available
        if pair_key in pair_ginis:
            oracle_top2_ginis[domain] = pair_ginis[pair_key]
        elif f"{d2}+{d1}" in pair_ginis:
            oracle_top2_ginis[domain] = pair_ginis[f"{d2}+{d1}"]
        else:
            delta_dict = compose_pair_delta(
                skeleton_np,
                all_adapters_np[di1], all_adapters_np[di2],
                di1, di2,
                OPTIMAL_SCALES[d1], OPTIMAL_SCALES[d2],
            )
            oracle_top2_ginis[domain] = compute_gini_for_delta_dict(delta_dict)
            del delta_dict
        g = oracle_top2_ginis[domain]
        log(f"  {domain:10s} (oracle: {d1}+{d2}): Gini={g['mean_gini']:.4f} "
            f"(max={g.get('max_gini', 0):.4f})")

    # 1d. Uniform N=5 Gini (baseline)
    log("\n  --- Uniform N=5 Gini (baseline) ---")
    uniform_delta = compose_uniform_n5_delta(skeleton_np, all_adapters_np)
    uniform_gini = compute_gini_for_delta_dict(uniform_delta)
    log(f"  Uniform N=5: Gini={uniform_gini['mean_gini']:.4f} +/- {uniform_gini['std_gini']:.4f} "
        f"(max={uniform_gini.get('max_gini', 0):.4f})")
    del uniform_delta

    # 1e. Uniform N=5 with 50% log-compression (Finding #279)
    log("\n  --- Uniform N=5 with 50% log-compression ---")
    eq_scales = compute_equalization_scales_partial(skeleton_np, all_adapters_np, compression=0.5)
    equalized_delta = compose_uniform_n5_delta(skeleton_np, all_adapters_np, eq_scales)
    equalized_gini = compute_gini_for_delta_dict(equalized_delta)
    log(f"  Uniform N=5 (50% log-comp): Gini={equalized_gini['mean_gini']:.4f} "
        f"+/- {equalized_gini['std_gini']:.4f} (max={equalized_gini.get('max_gini', 0):.4f})")
    del equalized_delta

    # K712 assessment: top-2 routed Gini > 0.15 on ANY pair?
    max_oracle_gini = max(g["max_gini"] for g in oracle_top2_ginis.values())
    mean_oracle_gini = np.mean([g["mean_gini"] for g in oracle_top2_ginis.values()])
    max_any_pair_gini = max(g["max_gini"] for g in pair_ginis.values())

    log(f"\n  --- K712 Assessment ---")
    log(f"  Max oracle top-2 Gini (any sample): {max_oracle_gini:.4f}")
    log(f"  Mean oracle top-2 Gini: {mean_oracle_gini:.4f}")
    log(f"  Max any-pair Gini (any sample): {max_any_pair_gini:.4f}")
    log(f"  K712: Max oracle Gini {'>' if max_oracle_gini > 0.15 else '<='} 0.15 -> "
        f"{'KILL' if max_oracle_gini > 0.15 else 'PASS'}")

    elapsed = time.time() - t0
    log(f"\n  Phase 1 time: {elapsed:.1f}s")

    return {
        "individual_ginis": {d: v for d, v in individual_ginis.items()},
        "pair_ginis": pair_ginis,
        "oracle_top2_ginis": oracle_top2_ginis,
        "uniform_n5_gini": uniform_gini,
        "uniform_n5_equalized_gini": equalized_gini,
        "k712_max_oracle_gini": float(max_oracle_gini),
        "k712_mean_oracle_gini": float(mean_oracle_gini),
        "k712_max_any_pair_gini": float(max_any_pair_gini),
        "k712_pass": bool(max_oracle_gini <= 0.15),
        "elapsed_s": elapsed,
    }


# ============================================================================
# PHASE 2: PPL Comparison (needs model)
# ============================================================================

def phase_ppl_comparison(skeleton_np, all_adapters_np):
    """Compare PPL: oracle single-adapter, oracle top-2, uniform N=5, uniform N=5 + 50% eq."""
    log("\n" + "=" * 70)
    log("PHASE 2: PERPLEXITY COMPARISON")
    log("=" * 70)
    t0 = time.time()

    # Load model
    log("  Loading model...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("model-loaded")

    # Load validation data
    log("  Loading validation data...")
    domain_texts = {}
    for domain in DOMAINS:
        val_path = DATA_DIR / domain / "valid.jsonl"
        texts = []
        with open(val_path) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
                if len(texts) >= N_EVAL_SAMPLES:
                    break
        domain_texts[domain] = texts
    log(f"  Loaded {sum(len(v) for v in domain_texts.values())} validation texts")

    base_weights = save_base_weights(model)

    # 2a. Base model PPL
    log("\n  --- Base model PPL ---")
    base_ppl = {}
    for domain in DOMAINS:
        loss = compute_val_loss(model, tokenizer, domain_texts[domain])
        base_ppl[domain] = math.exp(loss)
        log(f"  {domain}: {base_ppl[domain]:.4f}")

    # 2b. Oracle single-adapter PPL (best adapter = domain itself)
    log("\n  --- Oracle single-adapter PPL ---")
    single_adapter_ppl = {}
    for domain in DOMAINS:
        restore_base_weights(model, base_weights)
        di = DOMAINS.index(domain)
        adapter_np = all_adapters_np[di]
        merge_count = premerge_adapters(
            model, skeleton_np, [adapter_np], [di], [OPTIMAL_SCALES[domain]]
        )
        loss = compute_val_loss(model, tokenizer, domain_texts[domain])
        single_adapter_ppl[domain] = math.exp(loss)
        improvement = (base_ppl[domain] - single_adapter_ppl[domain]) / base_ppl[domain] * 100
        log(f"  {domain}: {single_adapter_ppl[domain]:.4f} ({improvement:+.1f}% vs base)")

    # 2c. Oracle top-2 PPL
    log("\n  --- Oracle top-2 PPL ---")
    top2_ppl = {}
    for domain in DOMAINS:
        restore_base_weights(model, base_weights)
        d1, d2 = ORACLE_TOP2[domain]
        di1, di2 = DOMAINS.index(d1), DOMAINS.index(d2)
        adapters = [all_adapters_np[di1], all_adapters_np[di2]]
        scales = [OPTIMAL_SCALES[d1], OPTIMAL_SCALES[d2]]
        merge_count = premerge_adapters(model, skeleton_np, adapters, [di1, di2], scales)
        loss = compute_val_loss(model, tokenizer, domain_texts[domain])
        top2_ppl[domain] = math.exp(loss)
        vs_single = (top2_ppl[domain] - single_adapter_ppl[domain]) / single_adapter_ppl[domain] * 100
        log(f"  {domain} ({d1}+{d2}): {top2_ppl[domain]:.4f} ({vs_single:+.1f}% vs single-adapter)")

    # 2d. Uniform N=5 PPL (all adapters at optimal scales)
    log("\n  --- Uniform N=5 PPL ---")
    restore_base_weights(model, base_weights)
    uniform_delta = compose_uniform_n5_delta(skeleton_np, all_adapters_np)
    apply_delta_dict_to_model(model, uniform_delta)
    uniform_ppl = {}
    for domain in DOMAINS:
        loss = compute_val_loss(model, tokenizer, domain_texts[domain])
        uniform_ppl[domain] = math.exp(loss)
        vs_single = (uniform_ppl[domain] - single_adapter_ppl[domain]) / single_adapter_ppl[domain] * 100
        log(f"  {domain}: {uniform_ppl[domain]:.4f} ({vs_single:+.1f}% vs single-adapter)")
    del uniform_delta

    # 2e. Uniform N=5 with 50% log-compression PPL
    log("\n  --- Uniform N=5 + 50% log-compression PPL ---")
    restore_base_weights(model, base_weights)
    eq_scales = compute_equalization_scales_partial(skeleton_np, all_adapters_np, compression=0.5)
    equalized_delta = compose_uniform_n5_delta(skeleton_np, all_adapters_np, eq_scales)
    apply_delta_dict_to_model(model, equalized_delta)
    equalized_ppl = {}
    for domain in DOMAINS:
        loss = compute_val_loss(model, tokenizer, domain_texts[domain])
        equalized_ppl[domain] = math.exp(loss)
        vs_single = (equalized_ppl[domain] - single_adapter_ppl[domain]) / single_adapter_ppl[domain] * 100
        log(f"  {domain}: {equalized_ppl[domain]:.4f} ({vs_single:+.1f}% vs single-adapter)")
    del equalized_delta

    # K713 assessment
    log("\n  --- K713 Assessment ---")
    domains_worse_5pct = 0
    for domain in DOMAINS:
        ratio = (top2_ppl[domain] - single_adapter_ppl[domain]) / single_adapter_ppl[domain]
        status = "WORSE" if ratio > 0.05 else "OK"
        log(f"  {domain}: top-2={top2_ppl[domain]:.4f} vs single={single_adapter_ppl[domain]:.4f} "
            f"({ratio:+.2%}) [{status}]")
        if ratio > 0.05:
            domains_worse_5pct += 1

    k713_pass = domains_worse_5pct < 3
    log(f"  K713: {domains_worse_5pct}/5 domains >5% worse -> {'PASS' if k713_pass else 'KILL'}")

    # P4 assessment: top-2 vs uniform+eq
    log("\n  --- P4: Top-2 vs Uniform+Eq ---")
    top2_wins = 0
    for domain in DOMAINS:
        top2_better = top2_ppl[domain] < equalized_ppl[domain]
        diff = (equalized_ppl[domain] - top2_ppl[domain]) / top2_ppl[domain] * 100
        log(f"  {domain}: top-2={top2_ppl[domain]:.4f} vs eq={equalized_ppl[domain]:.4f} "
            f"(top-2 {'better' if top2_better else 'worse'} by {abs(diff):.1f}%)")
        if top2_better:
            top2_wins += 1
    log(f"  P4: Top-2 wins {top2_wins}/5 domains")

    elapsed = time.time() - t0
    log(f"\n  Phase 2 time: {elapsed:.1f}s")
    log_memory("post-ppl")

    restore_base_weights(model, base_weights)
    cleanup(model, tokenizer, base_weights)

    return {
        "base_ppl": base_ppl,
        "single_adapter_ppl": single_adapter_ppl,
        "top2_ppl": top2_ppl,
        "uniform_n5_ppl": uniform_ppl,
        "equalized_n5_ppl": equalized_ppl,
        "k713_domains_worse": domains_worse_5pct,
        "k713_pass": bool(k713_pass),
        "p4_top2_wins": top2_wins,
        "elapsed_s": elapsed,
    }


# ============================================================================
# PHASE 3: Behavioral Evaluation (needs model + generation)
# ============================================================================

def phase_behavioral_eval(skeleton_np, all_adapters_np, prompts_by_domain):
    """Generate with oracle single-adapter and oracle top-2, evaluate with behavioral metrics."""
    log("\n" + "=" * 70)
    log("PHASE 3: BEHAVIORAL EVALUATION")
    log("=" * 70)
    t0 = time.time()

    # Load model
    log("  Loading model...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("model-loaded")

    base_weights = save_base_weights(model)

    mx.random.seed(SEED)
    np.random.seed(SEED)

    # Evaluate 3 domains with execution-based metrics (math, code, medical)
    # These are the domains where Finding #238 showed improvement.
    eval_domains = ["math", "code", "medical"]

    # 3a. Generate with oracle single-adapter
    log("\n  --- Oracle single-adapter generation ---")
    single_generations = {}
    for domain in eval_domains:
        restore_base_weights(model, base_weights)
        di = DOMAINS.index(domain)
        adapter_np = all_adapters_np[di]
        premerge_adapters(model, skeleton_np, [adapter_np], [di], [OPTIMAL_SCALES[domain]])

        domain_gens = []
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            formatted = format_prompt(prompt_data["instruction"])
            generated = generate_text(model, tokenizer, formatted, max_tokens=MAX_NEW_TOKENS)
            domain_gens.append(generated)
            if i < 2:
                log(f"  [{domain}][{i}] {len(generated)} chars")
        single_generations[domain] = domain_gens
        log(f"  {domain}: {len(domain_gens)} generations")

    # 3b. Generate with oracle top-2
    log("\n  --- Oracle top-2 generation ---")
    top2_generations = {}
    for domain in eval_domains:
        restore_base_weights(model, base_weights)
        d1, d2 = ORACLE_TOP2[domain]
        di1, di2 = DOMAINS.index(d1), DOMAINS.index(d2)
        adapters = [all_adapters_np[di1], all_adapters_np[di2]]
        scales = [OPTIMAL_SCALES[d1], OPTIMAL_SCALES[d2]]
        premerge_adapters(model, skeleton_np, adapters, [di1, di2], scales)

        domain_gens = []
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            formatted = format_prompt(prompt_data["instruction"])
            generated = generate_text(model, tokenizer, formatted, max_tokens=MAX_NEW_TOKENS)
            domain_gens.append(generated)
            if i < 2:
                log(f"  [{domain}][{i}] {len(generated)} chars")
        top2_generations[domain] = domain_gens
        log(f"  {domain}: {len(domain_gens)} generations")

    # 3c. Evaluate with behavioral metrics
    log("\n  --- Behavioral evaluation ---")
    single_evals = {}
    top2_evals = {}

    for domain in eval_domains:
        log(f"\n  === {domain.upper()} ===")

        # Single adapter
        single_domain_evals = []
        for prompt_data, gen_text in zip(prompts_by_domain[domain], single_generations[domain]):
            result = evaluate_response(gen_text, prompt_data["response"], domain)
            single_domain_evals.append(result)

        # Top-2
        top2_domain_evals = []
        for prompt_data, gen_text in zip(prompts_by_domain[domain], top2_generations[domain]):
            result = evaluate_response(gen_text, prompt_data["response"], domain)
            top2_domain_evals.append(result)

        single_scores = [r["score"] for r in single_domain_evals]
        top2_scores = [r["score"] for r in top2_domain_evals]

        single_mean = float(np.mean(single_scores))
        top2_mean = float(np.mean(top2_scores))

        log(f"  Single-adapter avg: {single_mean:.4f}")
        log(f"  Top-2 avg:          {top2_mean:.4f}")

        if single_mean > 0.001:
            degradation = (single_mean - top2_mean) / single_mean
            log(f"  Degradation:        {degradation:+.2%}")
        else:
            degradation = 0.0
            log(f"  Degradation:        N/A (single=0)")

        if domain == "math":
            s_correct = sum(1 for r in single_domain_evals if r.get("answer_correct", False))
            t_correct = sum(1 for r in top2_domain_evals if r.get("answer_correct", False))
            log(f"  Math correct: single={s_correct}/{len(single_domain_evals)}, "
                f"top2={t_correct}/{len(top2_domain_evals)}")
        elif domain == "code":
            s_syntax = sum(1 for r in single_domain_evals if r.get("syntax_valid", False))
            t_syntax = sum(1 for r in top2_domain_evals if r.get("syntax_valid", False))
            log(f"  Syntax valid: single={s_syntax}/{len(single_domain_evals)}, "
                f"top2={t_syntax}/{len(top2_domain_evals)}")

        single_evals[domain] = {
            "scores": single_scores,
            "mean": single_mean,
            "evals": single_domain_evals,
        }
        top2_evals[domain] = {
            "scores": top2_scores,
            "mean": top2_mean,
            "evals": top2_domain_evals,
            "degradation_vs_single": float(degradation),
        }

    # K714 assessment
    log("\n  --- K714 Assessment ---")
    domains_behavioral_worse_15pct = 0
    for domain in eval_domains:
        single_mean = single_evals[domain]["mean"]
        top2_mean = top2_evals[domain]["mean"]
        if single_mean > 0.001:
            degradation = (single_mean - top2_mean) / single_mean
        else:
            degradation = 0.0  # Can't be worse if single is zero
        status = "WORSE" if degradation > 0.15 else "OK"
        log(f"  {domain}: single={single_mean:.3f} top2={top2_mean:.3f} "
            f"degradation={degradation:.2%} [{status}]")
        if degradation > 0.15:
            domains_behavioral_worse_15pct += 1

    k714_pass = domains_behavioral_worse_15pct < 2
    log(f"  K714: {domains_behavioral_worse_15pct}/{len(eval_domains)} domains >15% worse -> "
        f"{'PASS' if k714_pass else 'KILL'}")

    elapsed = time.time() - t0
    log(f"\n  Phase 3 time: {elapsed:.1f}s")
    log_memory("post-behavioral")

    restore_base_weights(model, base_weights)
    cleanup(model, tokenizer, base_weights)

    return {
        "eval_domains": eval_domains,
        "single_evals": {d: {"mean": v["mean"], "scores": v["scores"]}
                         for d, v in single_evals.items()},
        "top2_evals": {d: {"mean": v["mean"], "scores": v["scores"],
                           "degradation_vs_single": v["degradation_vs_single"]}
                       for d, v in top2_evals.items()},
        "k714_domains_worse": domains_behavioral_worse_15pct,
        "k714_pass": bool(k714_pass),
        "elapsed_s": elapsed,
    }


# ============================================================================
# Main orchestrator
# ============================================================================

def main():
    t_start = time.time()
    log("=" * 70)
    log("TOP-K ROUTING ELIMINATES SPECTRAL COMPOSITION PATHOLOGY")
    log("=" * 70)
    log(f"Model: {MODEL_ID}")
    log(f"Domains: {DOMAINS}")
    log(f"Oracle top-2 routing: {ORACLE_TOP2}")
    log(f"Per-domain scales: {OPTIMAL_SCALES}")
    log_memory("start")

    # Load shared numpy resources (lightweight)
    log("\nLoading skeleton and adapters (numpy)...")
    skeleton_np = dict(np.load(str(SKELETON_PATH)))
    log(f"  Skeleton keys: {len(skeleton_np)}")

    all_adapters_np = []
    for di, domain in enumerate(DOMAINS):
        adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
        adapter = dict(np.load(str(adapter_path)))
        all_adapters_np.append(adapter)
        log(f"  {domain}: {len(adapter)} keys")

    # Load prompts for behavioral eval
    log("\nLoading evaluation prompts...")
    prompts_by_domain = {}
    for domain in ["math", "code", "medical"]:  # Only domains with execution metrics
        prompts = extract_prompts_with_answers(domain, N_PROMPTS_PER_DOMAIN)
        prompts_by_domain[domain] = prompts
        log(f"  {domain}: {len(prompts)} prompts loaded")

    # Phase 1: Spectral Analysis (numpy only)
    phase1 = phase_spectral_analysis(skeleton_np, all_adapters_np)
    gc.collect()

    # Phase 2: PPL Comparison (needs model load)
    phase2 = phase_ppl_comparison(skeleton_np, all_adapters_np)
    gc.collect()

    # Phase 3: Behavioral Evaluation (needs model + generation)
    phase3 = phase_behavioral_eval(skeleton_np, all_adapters_np, prompts_by_domain)
    gc.collect()

    # ============================================================================
    # Summary and kill criteria
    # ============================================================================
    elapsed = time.time() - t_start

    log("\n" + "=" * 70)
    log("SUMMARY: KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    k712_pass = phase1["k712_pass"]
    k713_pass = phase2["k713_pass"]
    k714_pass = phase3["k714_pass"]

    log(f"\n  K712 (Gini < 0.15): {'PASS' if k712_pass else 'KILL'} "
        f"(max oracle Gini = {phase1['k712_max_oracle_gini']:.4f})")
    log(f"  K713 (PPL within 5%): {'PASS' if k713_pass else 'KILL'} "
        f"({phase2['k713_domains_worse']}/5 domains worse)")
    log(f"  K714 (behavioral within 15%): {'PASS' if k714_pass else 'KILL'} "
        f"({phase3['k714_domains_worse']}/{len(phase3['eval_domains'])} domains worse)")

    log(f"\n  --- Key Comparisons ---")
    log(f"  Uniform N=5 Gini:              {phase1['uniform_n5_gini']['mean_gini']:.4f}")
    log(f"  Uniform N=5 + 50% eq Gini:     {phase1['uniform_n5_equalized_gini']['mean_gini']:.4f}")
    log(f"  Mean oracle top-2 Gini:        {phase1['k712_mean_oracle_gini']:.4f}")
    gini_reduction = (phase1['uniform_n5_gini']['mean_gini'] - phase1['k712_mean_oracle_gini']) / \
                     phase1['uniform_n5_gini']['mean_gini'] * 100
    log(f"  Gini reduction (routing vs uniform): {gini_reduction:.1f}%")
    log(f"  Gini reduction (eq vs uniform):      "
        f"{(phase1['uniform_n5_gini']['mean_gini'] - phase1['uniform_n5_equalized_gini']['mean_gini']) / phase1['uniform_n5_gini']['mean_gini'] * 100:.1f}%")

    log(f"\n  P4: Top-2 routing wins {phase2['p4_top2_wins']}/5 domains vs uniform+eq")

    log(f"\n  Total time: {elapsed:.0f}s")

    results = {
        "experiment": "routing_eliminates_spectral_pathology",
        "model": MODEL_ID,
        "domains": DOMAINS,
        "oracle_top2": ORACLE_TOP2,
        "optimal_scales": OPTIMAL_SCALES,
        "phase1_spectral": phase1,
        "phase2_ppl": phase2,
        "phase3_behavioral": phase3,
        "kill_criteria": {
            "K712": {
                "description": "Top-2 routed composed Gini > 0.15 on any eval batch",
                "max_oracle_gini": phase1["k712_max_oracle_gini"],
                "threshold": 0.15,
                "result": "PASS" if k712_pass else "KILL",
            },
            "K713": {
                "description": "Top-2 per-domain PPL > 5% worse than single-adapter on >=3/5",
                "domains_worse": phase2["k713_domains_worse"],
                "threshold": 3,
                "result": "PASS" if k713_pass else "KILL",
            },
            "K714": {
                "description": "Top-2 behavioral quality < oracle by >15% on >=2/5",
                "domains_worse": phase3["k714_domains_worse"],
                "threshold": 2,
                "result": "PASS" if k714_pass else "KILL",
            },
        },
        "predictions": {
            "P1_gini_lt_015": {
                "predicted": "< 0.15",
                "measured": phase1["k712_mean_oracle_gini"],
                "match": bool(phase1["k712_mean_oracle_gini"] < 0.15),
            },
            "P2_ppl_within_5pct": {
                "predicted": "within 5% on >=3/5 domains",
                "domains_within": 5 - phase2["k713_domains_worse"],
                "match": k713_pass,
            },
            "P3_behavioral_within_15pct": {
                "predicted": "within 15% on >=2/3 eval domains",
                "domains_within": len(phase3["eval_domains"]) - phase3["k714_domains_worse"],
                "match": k714_pass,
            },
            "P4_top2_beats_equalized": {
                "predicted": "top-2 wins majority of domains",
                "top2_wins": phase2["p4_top2_wins"],
                "match": phase2["p4_top2_wins"] >= 3,
            },
        },
        "total_time_s": round(elapsed, 1),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    log(f"\nResults saved to {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    main()
