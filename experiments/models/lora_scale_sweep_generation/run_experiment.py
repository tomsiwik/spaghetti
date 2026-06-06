#!/usr/bin/env python3
"""LoRA Scale Sweep: Generation Quality Across 5 Domains.

Kill criteria:
  K620: No lora_scale produces domain adapter that beats base on its own domain by >10%
  K621: All lora_scale values show code adapter still universal best
  K622: Lower lora_scale produces incoherent output (format quality degrades)

Type: guided-exploration
Platform: Apple M5 Pro 48GB, MLX

OPTIMIZATION: Model loaded once per scale. Adapter weights swapped in-place.
Each adapter only evaluated on OWN domain (for K620) + code adapter on all 5 (for K621).
This reduces 1250 generations to ~200, cutting runtime from ~4h to ~40min.
"""

import ast
import gc
import json
import math
import os
import re
import time
from collections import defaultdict
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
SFT_DIR = EXPERIMENT_DIR.parent / "bitnet_sft_generation_v3"
ADAPTERS_DIR = SFT_DIR / "sft_adapters"
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
DATA_DIR = SOURCE_DIR / "data"
NTP_ADAPTERS_DIR = SOURCE_DIR / "adapters"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
MAX_SEQ_LENGTH = 256
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]
SCALES = [1.0, 2.0, 4.0, 8.0, 20.0]
NUM_PROMPTS_PER_DOMAIN = 10
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0

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
# Model utilities
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


class TernaryLoRALinear(nn.Module):
    """LoRA with STE-ternary B and optional Grassmannian A."""
    def __init__(self, base_linear: nn.Linear, rank: int = 16,
                 scale: float = 20.0, a_init: mx.array = None):
        super().__init__()
        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]
        self.linear = base_linear
        if a_init is not None:
            self.lora_a = a_init
        else:
            s = 1.0 / math.sqrt(in_features)
            self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_features, rank))
        self.lora_b = mx.zeros((rank, out_features))
        self.scale = scale
        self.rank = rank
        self.linear.freeze()
        self.freeze(keys=["lora_a"], strict=False)

    def __call__(self, x):
        base_out = self.linear(x)
        b = self.lora_b
        alpha = mx.mean(mx.abs(b))
        b_scaled = b / (alpha + 1e-7)
        b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
        b_ste = b + mx.stop_gradient(b_q - b)
        lora_out = (x @ self.lora_a) @ b_ste * self.scale
        return base_out + lora_out


# ============================================================================
# Model loading helpers
# ============================================================================

def load_skeleton():
    skeleton_path = NTP_ADAPTERS_DIR / "grassmannian_skeleton.npz"
    return dict(np.load(str(skeleton_path)))


def apply_lora_with_skeleton(model, skeleton, domain_idx, lora_scale):
    count = 0
    for li, layer in enumerate(model.model.layers):
        lora_updates = []
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue
            skey = f"layer_{li}_{key}_domain_{domain_idx}"
            a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16) if skey in skeleton else None
            lora = TernaryLoRALinear(module, rank=LORA_RANK, scale=lora_scale, a_init=a_mx)
            lora_updates.append((key, lora))
            count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    mx.eval(model.parameters())
    return model


def apply_single_adapter(model, skeleton, domain_idx, adapter_path, lora_scale):
    model = apply_lora_with_skeleton(model, skeleton, domain_idx, lora_scale)
    adapter_params = dict(mx.load(str(adapter_path)))
    model.update(tree_unflatten(list(adapter_params.items())))
    mx.eval(model.parameters())
    return model


def format_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_text(model, tokenizer, prompt, max_tokens=128, temperature=0.0):
    try:
        sampler = make_sampler(temp=temperature)
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


def generate_for_prompts(model, tokenizer, prompts, domain_label=""):
    """Generate responses for a list of prompts."""
    results = []
    for prompt_data in prompts:
        formatted = format_prompt(prompt_data["instruction"])
        generated = generate_text(model, tokenizer, formatted,
                                  max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
        results.append(generated)
    return results


# ============================================================================
# Evaluation metrics
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


STOPWORDS = {
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


def extract_key_facts(text):
    facts = set()
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    for w in words:
        if len(w) >= 4 and w not in STOPWORDS:
            facts.add(w)
    number_patterns = re.findall(
        r'\b(\d+(?:\.\d+)?)\s*(%|percent|years?|months?|days?|hours?|mg|ml|kg|lb|dollars?|\$)?',
        text.lower())
    for num, unit in number_patterns:
        if unit:
            facts.add(f"{num} {unit}".strip())
        facts.add(num)
    non_stop = [w for w in words if w not in STOPWORDS and len(w) >= 3]
    for i in range(len(non_stop) - 1):
        facts.add(f"{non_stop[i]} {non_stop[i+1]}")
    return facts


def eval_factual_recall(generated_text, reference_text):
    ref_facts = extract_key_facts(reference_text)
    if not ref_facts:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    gen_lower = generated_text.lower()
    matched = sum(1 for fact in ref_facts if fact in gen_lower)
    recall = matched / len(ref_facts)
    gen_facts = extract_key_facts(generated_text)
    ref_lower = reference_text.lower()
    gen_matched = sum(1 for fact in gen_facts if fact in ref_lower) if gen_facts else 0
    precision = gen_matched / len(gen_facts) if gen_facts else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}


def eval_numerical_accuracy(generated_text, reference_text):
    def extract_numbers(text):
        matches = re.findall(r'(?:\$)?([\d,]+(?:\.\d+)?)\s*(%)?', text)
        numbers = set()
        for num_str, pct in matches:
            try:
                numbers.add(float(num_str.replace(',', '')))
            except ValueError:
                pass
        return numbers
    ref_nums = extract_numbers(reference_text)
    gen_nums = extract_numbers(generated_text)
    if not ref_nums:
        return 0.0
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
    return matched / len(ref_nums)


def evaluate_response(generated_text, reference_text, domain):
    """Returns score in [0,1]."""
    if domain == "code":
        syntax_ok = eval_code_syntax(generated_text)
        factual = eval_factual_recall(generated_text, reference_text)
        return 0.7 * (1.0 if syntax_ok else 0.0) + 0.3 * factual["recall"]
    elif domain == "math":
        gen_answer = extract_math_answer(generated_text)
        gt_answer = extract_ground_truth_answer(reference_text)
        return 1.0 if eval_math_correct(gen_answer, gt_answer) else 0.0
    elif domain in ("medical", "legal"):
        return eval_factual_recall(generated_text, reference_text)["recall"]
    elif domain == "finance":
        factual = eval_factual_recall(generated_text, reference_text)
        num_acc = eval_numerical_accuracy(generated_text, reference_text)
        return 0.6 * factual["recall"] + 0.4 * num_acc
    return 0.0


def eval_format_quality(generated_text):
    """Check basic format quality: non-empty, has words, not just repetition."""
    if not generated_text.strip():
        return 0.0
    words = generated_text.split()
    if len(words) < 3:
        return 0.1
    unique_words = set(w.lower() for w in words)
    diversity = len(unique_words) / len(words)
    if diversity < 0.1:
        return 0.1
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    if len(alpha_words) < 3:
        return 0.2
    return min(1.0, diversity + 0.3)


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
# Phase 1: Base model generations (all 5 domains)
# ============================================================================

def phase_generate_base(prompts_by_domain):
    log("\n" + "=" * 70)
    log("PHASE 1: BASE MODEL (50 generations)")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    mx.random.seed(SEED)
    np.random.seed(SEED)

    results = {}
    for domain in DOMAINS:
        results[domain] = generate_for_prompts(model, tokenizer, prompts_by_domain[domain])
        log(f"  {domain}: {len(results[domain])} generations")

    elapsed = time.time() - t0
    del model, tokenizer
    cleanup()
    log_memory("post-base")
    log(f"  Base done: {elapsed:.1f}s")
    return results, elapsed


# ============================================================================
# Phase 2: Each adapter on OWN domain at all scales
# ============================================================================

def phase_own_domain_sweep(prompts_by_domain):
    """For each adapter, generate on its own domain at all 5 scales.

    This is the core data for K620 (does any adapter beat base?).
    One model load per (adapter, scale) pair = 25 loads.
    But only 10 generations per load (own domain only) instead of 50.
    """
    log("\n" + "=" * 70)
    log("PHASE 2: OWN-DOMAIN SWEEP (5 adapters x 5 scales, 10 gens each)")
    log("=" * 70)

    skeleton = load_skeleton()
    # own_gens[adapter_domain][scale] = list[str]
    own_gens = {d: {} for d in DOMAINS}
    total_t0 = time.time()

    for adapter_domain in DOMAINS:
        domain_idx = DOMAINS.index(adapter_domain)
        adapter_path = ADAPTERS_DIR / adapter_domain / "adapter.npz"
        if not adapter_path.exists():
            log(f"  WARNING: {adapter_path} not found, skipping")
            continue

        for scale in SCALES:
            t0 = time.time()
            model, tokenizer = load(MODEL_ID)
            model = replace_bitlinear_with_linear(model)
            model = apply_single_adapter(model, skeleton, domain_idx, adapter_path, scale)
            model.freeze()
            mx.random.seed(SEED)
            np.random.seed(SEED)

            gens = generate_for_prompts(
                model, tokenizer, prompts_by_domain[adapter_domain])
            own_gens[adapter_domain][scale] = gens

            elapsed = time.time() - t0
            log(f"  {adapter_domain}@s={scale}: {len(gens)} gens in {elapsed:.1f}s")

            del model, tokenizer
            cleanup()

    del skeleton
    cleanup()
    total_elapsed = time.time() - total_t0
    log(f"  Own-domain sweep: {total_elapsed:.1f}s")
    log_memory("post-own-domain")
    return own_gens, total_elapsed


# ============================================================================
# Phase 3: Code adapter on ALL domains at scale=2 and best scale
# (for K621 — is code still universal best?)
# ============================================================================

def phase_code_cross_domain(prompts_by_domain, best_scale_per_domain):
    """Code adapter on all 5 domains at scale=2 and each domain's best scale.

    This tests K621: is code adapter still universally best?
    """
    log("\n" + "=" * 70)
    log("PHASE 3: CODE ADAPTER CROSS-DOMAIN")
    log("=" * 70)

    skeleton = load_skeleton()
    code_idx = DOMAINS.index("code")
    code_path = ADAPTERS_DIR / "code" / "adapter.npz"

    # Determine which scales to test
    test_scales = list(set([2.0] + list(best_scale_per_domain.values())))
    test_scales.sort()

    # code_gens[scale][eval_domain] = list[str]
    code_gens = {}
    total_t0 = time.time()

    for scale in test_scales:
        t0 = time.time()
        model, tokenizer = load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)
        model = apply_single_adapter(model, skeleton, code_idx, code_path, scale)
        model.freeze()
        mx.random.seed(SEED)
        np.random.seed(SEED)

        code_gens[scale] = {}
        for eval_domain in DOMAINS:
            gens = generate_for_prompts(model, tokenizer, prompts_by_domain[eval_domain])
            code_gens[scale][eval_domain] = gens

        elapsed = time.time() - t0
        log(f"  code@s={scale}: all 5 domains in {elapsed:.1f}s")

        del model, tokenizer
        cleanup()

    del skeleton
    cleanup()
    total_elapsed = time.time() - total_t0
    log(f"  Code cross-domain: {total_elapsed:.1f}s")
    return code_gens, total_elapsed


# ============================================================================
# Phase 4: Evaluate and analyze
# ============================================================================

def phase_evaluate_and_analyze(prompts_by_domain, base_gens, own_gens, code_gens):
    log("\n" + "=" * 70)
    log("PHASE 4: EVALUATE & ANALYZE")
    log("=" * 70)

    # --- Base scores ---
    base_scores = {}
    for domain in DOMAINS:
        scores = []
        format_scores = []
        for prompt_data, gen_text in zip(prompts_by_domain[domain], base_gens[domain]):
            scores.append(evaluate_response(gen_text, prompt_data["response"], domain))
            format_scores.append(eval_format_quality(gen_text))
        base_scores[domain] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "stderr": float(np.std(scores) / np.sqrt(len(scores))),
            "scores": [float(s) for s in scores],
            "format_mean": float(np.mean(format_scores)),
        }
        log(f"  Base {domain}: {base_scores[domain]['mean']:.4f} "
            f"(+/-{base_scores[domain]['stderr']:.4f})")

    # --- Own-domain adapter scores ---
    # own_scores[adapter_domain][scale] = {mean, std, stderr, advantage_pct, ...}
    own_scores = {}
    for adapter_domain in DOMAINS:
        own_scores[adapter_domain] = {}
        for scale in SCALES:
            if scale not in own_gens[adapter_domain]:
                continue
            gens = own_gens[adapter_domain][scale]
            scores = []
            format_scores = []
            for prompt_data, gen_text in zip(prompts_by_domain[adapter_domain], gens):
                scores.append(evaluate_response(gen_text, prompt_data["response"], adapter_domain))
                format_scores.append(eval_format_quality(gen_text))

            mean = float(np.mean(scores))
            base_mean = base_scores[adapter_domain]["mean"]
            advantage = (mean / max(base_mean, 0.001) - 1) * 100

            own_scores[adapter_domain][scale] = {
                "mean": mean,
                "std": float(np.std(scores)),
                "stderr": float(np.std(scores) / np.sqrt(len(scores))),
                "scores": [float(s) for s in scores],
                "format_mean": float(np.mean(format_scores)),
                "advantage_pct": advantage,
            }

    # Print scale profiles
    log("\n  === SCALE PROFILES (own-domain advantage vs base) ===")
    log(f"  {'Domain':<10} | " + " | ".join(f"s={s:<4}" for s in SCALES) + " |")
    log(f"  {'-'*10}-+-" + "-+-".join(f"{'-'*6}" for _ in SCALES) + "-+")
    for domain in DOMAINS:
        parts = []
        for scale in SCALES:
            if scale in own_scores[domain]:
                adv = own_scores[domain][scale]["advantage_pct"]
                parts.append(f"{adv:+5.1f}%")
            else:
                parts.append("  N/A ")
        log(f"  {domain:<10} | " + " | ".join(parts) + " |")

    # --- K620: Any adapter beats base by >10% on own domain? ---
    k620_found = False
    best_advantages = {}
    best_scale_per_domain = {}
    for domain in DOMAINS:
        best_adv = -999
        best_s = None
        for scale in SCALES:
            if scale in own_scores[domain]:
                adv = own_scores[domain][scale]["advantage_pct"]
                best_advantages[f"{domain}@s={scale}"] = adv
                if adv > best_adv:
                    best_adv = adv
                    best_s = scale
                if adv > 10.0:
                    k620_found = True
        best_scale_per_domain[domain] = best_s

    log(f"\n  K620 (adapter beats base >10%): {'PASS' if k620_found else 'FAIL -> KILL'}")
    sorted_adv = sorted(best_advantages.items(), key=lambda x: x[1], reverse=True)
    for key, adv in sorted_adv[:8]:
        log(f"    {key}: {adv:+.1f}%")

    # --- K622: Low scale coherent? ---
    low_scale_format_ok = True
    format_by_scale = {}
    for scale in [1.0, 2.0]:
        fmts = []
        for domain in DOMAINS:
            if scale in own_scores[domain]:
                fmts.append(own_scores[domain][scale]["format_mean"])
        avg_fmt = float(np.mean(fmts)) if fmts else 0.0
        format_by_scale[scale] = avg_fmt
        if avg_fmt < 0.3:
            low_scale_format_ok = False

    log(f"\n  K622 (low scale coherent): {'PASS' if low_scale_format_ok else 'FAIL -> KILL'}")
    for scale, fmt in format_by_scale.items():
        log(f"    Scale {scale}: avg format={fmt:.3f}")

    # --- Code adapter cross-domain scores ---
    code_cross_scores = {}
    for scale in code_gens:
        code_cross_scores[scale] = {}
        for eval_domain in DOMAINS:
            gens = code_gens[scale][eval_domain]
            scores = []
            for prompt_data, gen_text in zip(prompts_by_domain[eval_domain], gens):
                scores.append(evaluate_response(gen_text, prompt_data["response"], eval_domain))
            code_cross_scores[scale][eval_domain] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }

    # --- K621: Is code adapter still universal best? ---
    # At scale=2 (our candidate best), compare code vs domain adapter
    log(f"\n  === CODE vs DOMAIN ADAPTER (at best domain scales) ===")
    code_beats_count = 0
    domain_beats_count = 0
    k621_details = {}

    for domain in DOMAINS:
        best_s = best_scale_per_domain[domain]
        domain_score = own_scores[domain][best_s]["mean"] if best_s in own_scores[domain] else 0
        # Find code score at this domain's best scale (or closest available)
        code_score = 0
        for cs in sorted(code_cross_scores.keys(), key=lambda x: abs(x - best_s)):
            if domain in code_cross_scores[cs]:
                code_score = code_cross_scores[cs][domain]["mean"]
                code_scale_used = cs
                break

        winner = "code" if code_score > domain_score else domain
        if winner == "code":
            code_beats_count += 1
        else:
            domain_beats_count += 1

        k621_details[domain] = {
            "domain_adapter_score": domain_score,
            "domain_adapter_scale": best_s,
            "code_score": code_score,
            "code_scale": code_scale_used,
            "winner": winner,
        }
        log(f"  {domain}: domain={domain_score:.4f}@s={best_s} vs code={code_score:.4f}@s={code_scale_used} -> {winner}")

    code_universally_best = code_beats_count == 5
    log(f"\n  K621 (code NOT universal best): {'PASS' if not code_universally_best else 'FAIL -> KILL'}")
    log(f"    Code wins: {code_beats_count}/5, Domain wins: {domain_beats_count}/5")

    # --- Prediction verification ---
    log("\n  === PREDICTION VERIFICATION ===")

    # P4: alpha(20, legal) < 0
    alpha_legal_20 = own_scores["legal"].get(20.0, {}).get("advantage_pct", 0) / 100
    p4_pass = alpha_legal_20 < 0
    log(f"  P4 (alpha(20,legal) < 0): {'CONFIRMED' if p4_pass else 'REFUTED'} ({alpha_legal_20:.3f})")

    # P5: s* varies by domain type
    prose_scales = [best_scale_per_domain[d] for d in ["medical", "legal", "finance"]
                    if best_scale_per_domain.get(d)]
    struct_scales = [best_scale_per_domain[d] for d in ["code", "math"]
                     if best_scale_per_domain.get(d)]
    if prose_scales and struct_scales:
        p5_pass = np.mean(prose_scales) != np.mean(struct_scales)
        log(f"  P5 (s* varies by domain): {'CONFIRMED' if p5_pass else 'REFUTED'} "
            f"(prose avg={np.mean(prose_scales):.1f}, struct avg={np.mean(struct_scales):.1f})")
    else:
        p5_pass = False

    log(f"\n  Best scale by domain:")
    for d in DOMAINS:
        s = best_scale_per_domain.get(d)
        if s and s in own_scores[d]:
            log(f"    {d}: s*={s}, score={own_scores[d][s]['mean']:.4f}, "
                f"adv={own_scores[d][s]['advantage_pct']:+.1f}%")

    # --- Compile analysis ---
    analysis = {
        "kill_criteria": {
            "K620": {
                "pass": k620_found,
                "description": "At least one adapter beats base by >10% on own domain",
            },
            "K621": {
                "pass": not code_universally_best,
                "description": "Code adapter is NOT universal best at all domains",
                "code_wins": code_beats_count,
                "domain_wins": domain_beats_count,
                "details": k621_details,
            },
            "K622": {
                "pass": low_scale_format_ok,
                "description": "Low scale output is coherent (format > 0.3)",
                "format_by_scale": format_by_scale,
            },
        },
        "predictions": {
            "P1_alpha_gt_10pct": k620_found,
            "P4_legal_20_negative": p4_pass,
            "P5_scale_varies": p5_pass,
        },
        "best_scale_by_domain": best_scale_per_domain,
        "best_advantages": best_advantages,
    }

    return base_scores, own_scores, code_cross_scores, analysis


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("LoRA SCALE SWEEP: GENERATION QUALITY")
    log("=" * 70)
    log(f"Scales: {SCALES}")
    log(f"Domains: {DOMAINS}")
    log(f"Prompts/domain: {NUM_PROMPTS_PER_DOMAIN}")
    log(f"Total est. generations: base=50, own-domain=250, code-cross=~50-100")
    log_memory("start")

    # Load prompts
    prompts_by_domain = {}
    for domain in DOMAINS:
        prompts = extract_prompts_with_answers(domain, NUM_PROMPTS_PER_DOMAIN)
        prompts_by_domain[domain] = prompts
        log(f"  {domain}: {len(prompts)} prompts")

    # Phase 1: Base model
    base_gens, base_time = phase_generate_base(prompts_by_domain)

    # Phase 2: Each adapter on own domain at all scales
    own_gens, sweep_time = phase_own_domain_sweep(prompts_by_domain)

    # Quick analysis to find best scales per domain
    quick_best_scales = {}
    for domain in DOMAINS:
        best_s = 1.0
        best_score = -1
        for scale in SCALES:
            if scale in own_gens[domain]:
                # Quick eval: just check average score
                scores = []
                for prompt_data, gen_text in zip(prompts_by_domain[domain], own_gens[domain][scale]):
                    scores.append(evaluate_response(gen_text, prompt_data["response"], domain))
                avg = float(np.mean(scores))
                if avg > best_score:
                    best_score = avg
                    best_s = scale
        quick_best_scales[domain] = best_s
    log(f"\n  Quick best scales: {quick_best_scales}")

    # Phase 3: Code adapter cross-domain at relevant scales
    code_gens, code_time = phase_code_cross_domain(prompts_by_domain, quick_best_scales)

    # Phase 4: Full evaluation and analysis
    base_scores, own_scores, code_cross_scores, analysis = phase_evaluate_and_analyze(
        prompts_by_domain, base_gens, own_gens, code_gens)

    # Compile results
    total_time = time.time() - t0
    results = {
        "experiment": "lora_scale_sweep_generation",
        "model": MODEL_ID,
        "scales": SCALES,
        "domains": DOMAINS,
        "n_prompts": NUM_PROMPTS_PER_DOMAIN,
        "base_scores": base_scores,
        "own_domain_scores": {
            d: {str(s): v for s, v in scales.items()}
            for d, scales in own_scores.items()
        },
        "code_cross_domain_scores": {
            str(s): {d: v for d, v in domains.items()}
            for s, domains in code_cross_scores.items()
        },
        "analysis": analysis,
        "timing": {
            "base_generation_s": base_time,
            "own_domain_sweep_s": sweep_time,
            "code_cross_domain_s": code_time,
            "total_s": total_time,
        },
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Summary
    log("\n" + "=" * 70)
    log("KILL CRITERIA SUMMARY")
    log("=" * 70)
    for k, v in analysis["kill_criteria"].items():
        status = "PASS" if v["pass"] else "FAIL"
        log(f"  {k}: {status} - {v['description']}")

    all_pass = all(v["pass"] for v in analysis["kill_criteria"].values())
    log(f"\n  Overall: {'ALL PASS -> SUPPORTED' if all_pass else 'KILL TRIGGERED'}")


if __name__ == "__main__":
    main()
