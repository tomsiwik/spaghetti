#!/usr/bin/env python3
"""Generation Quality Retest: Per-Domain Optimal Scales.

Retest of exp_generation_quality_test (killed) with per-domain optimal scales
from Finding #217. Original used uniform scale=20 for all adapters.

Three conditions:
  1. Base model (no adapter)
  2. Uniform: oracle top-1 routing, all adapters at s=20
  3. Scale-aware: oracle top-1 routing, per-domain optimal scales

Kill criteria:
  K631: Scale-aware routing STILL worse on >= 3/5 domains
  K632: Per-domain scale < 5% improvement over uniform scale=20
  K633: All generated text is incoherent

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
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]
NUM_PROMPTS_PER_DOMAIN = 10
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0

# Per-domain optimal scales from Finding #217
OPTIMAL_SCALES = {
    "medical": 20.0,
    "code": 20.0,
    "math": 20.0,
    "legal": 4.0,
    "finance": 1.0,
}
UNIFORM_SCALE = 20.0

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
# Model utilities (same as lora_scale_sweep_generation)
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


def generate_for_prompts(model, tokenizer, prompts, label=""):
    results = []
    for prompt_data in prompts:
        formatted = format_prompt(prompt_data["instruction"])
        generated = generate_text(model, tokenizer, formatted,
                                  max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
        results.append(generated)
    return results


# ============================================================================
# Evaluation metrics (same as lora_scale_sweep_generation)
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
# Phase 1: Base model generations
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
# Phase 2: Oracle top-1 routing with given scale map
# ============================================================================

def phase_generate_routed(prompts_by_domain, scale_map, label=""):
    """Generate with oracle top-1 routing: each domain uses its own adapter at given scale."""
    log(f"\n" + "=" * 70)
    log(f"PHASE 2: ROUTED GENERATION ({label})")
    log(f"  Scales: {scale_map}")
    log("=" * 70)

    skeleton = load_skeleton()
    results = {}
    total_t0 = time.time()

    for domain in DOMAINS:
        domain_idx = DOMAINS.index(domain)
        adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
        scale = scale_map[domain]

        if not adapter_path.exists():
            log(f"  WARNING: {adapter_path} not found, skipping {domain}")
            results[domain] = [""] * len(prompts_by_domain[domain])
            continue

        t0 = time.time()
        model, tokenizer = load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)
        model = apply_single_adapter(model, skeleton, domain_idx, adapter_path, scale)
        model.freeze()
        mx.random.seed(SEED)
        np.random.seed(SEED)

        gens = generate_for_prompts(model, tokenizer, prompts_by_domain[domain])
        results[domain] = gens

        elapsed = time.time() - t0
        log(f"  {domain}@s={scale}: {len(gens)} gens in {elapsed:.1f}s")

        del model, tokenizer
        cleanup()

    del skeleton
    cleanup()
    total_elapsed = time.time() - total_t0
    log_memory(f"post-{label}")
    log(f"  {label} done: {total_elapsed:.1f}s")
    return results, total_elapsed


# ============================================================================
# Phase 3: Evaluate all conditions
# ============================================================================

def evaluate_condition(prompts_by_domain, generations, condition_name):
    """Score all generations for a condition."""
    scores = {}
    for domain in DOMAINS:
        domain_scores = []
        format_scores = []
        for prompt_data, gen_text in zip(prompts_by_domain[domain], generations[domain]):
            domain_scores.append(evaluate_response(gen_text, prompt_data["response"], domain))
            format_scores.append(eval_format_quality(gen_text))
        scores[domain] = {
            "mean": float(np.mean(domain_scores)),
            "std": float(np.std(domain_scores)),
            "stderr": float(np.std(domain_scores) / np.sqrt(len(domain_scores))),
            "scores": [float(s) for s in domain_scores],
            "format_mean": float(np.mean(format_scores)),
        }
        log(f"  [{condition_name}] {domain}: {scores[domain]['mean']:.4f} "
            f"(+/-{scores[domain]['stderr']:.4f}, fmt={scores[domain]['format_mean']:.3f})")
    return scores


def phase_evaluate(prompts_by_domain, base_gens, uniform_gens, perscale_gens):
    log("\n" + "=" * 70)
    log("PHASE 3: EVALUATE ALL CONDITIONS")
    log("=" * 70)

    base_scores = evaluate_condition(prompts_by_domain, base_gens, "base")
    uniform_scores = evaluate_condition(prompts_by_domain, uniform_gens, "uniform-s20")
    perscale_scores = evaluate_condition(prompts_by_domain, perscale_gens, "per-scale")

    # --- Compute advantages ---
    log("\n  === ADVANTAGE vs BASE ===")
    log(f"  {'Domain':<10} | {'Base':>8} | {'Uniform':>8} | {'Per-Scale':>8} | {'Uni %':>8} | {'PS %':>8} |")
    log(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+")

    uniform_better_count = 0
    perscale_better_count = 0
    perscale_vs_uniform_better = 0
    domain_results = {}

    for domain in DOMAINS:
        b = base_scores[domain]["mean"]
        u = uniform_scores[domain]["mean"]
        p = perscale_scores[domain]["mean"]

        u_adv = ((u / max(b, 0.001)) - 1) * 100
        p_adv = ((p / max(b, 0.001)) - 1) * 100
        p_vs_u = ((p / max(u, 0.001)) - 1) * 100

        if u > b:
            uniform_better_count += 1
        if p > b:
            perscale_better_count += 1
        if p > u:
            perscale_vs_uniform_better += 1

        domain_results[domain] = {
            "base": b,
            "uniform": u,
            "perscale": p,
            "uniform_adv_pct": u_adv,
            "perscale_adv_pct": p_adv,
            "perscale_vs_uniform_pct": p_vs_u,
        }

        log(f"  {domain:<10} | {b:>8.4f} | {u:>8.4f} | {p:>8.4f} | {u_adv:>+7.1f}% | {p_adv:>+7.1f}% |")

    # --- Kill criteria ---
    log("\n  === KILL CRITERIA ===")

    # K631: Scale-aware routing STILL worse on >= 3/5 domains
    perscale_worse_count = sum(1 for d in DOMAINS if domain_results[d]["perscale"] < domain_results[d]["base"])
    k631_pass = perscale_worse_count < 3
    log(f"  K631 (per-scale worse on <3/5 domains): {'PASS' if k631_pass else 'FAIL -> KILL'}")
    log(f"    Per-scale worse on {perscale_worse_count}/5 domains, better on {perscale_better_count}/5")

    # K632: Per-domain scale >= 5% improvement over uniform on >= 1 domain
    perscale_improvement_domains = []
    for domain in DOMAINS:
        imp = domain_results[domain]["perscale_vs_uniform_pct"]
        if imp > 5.0:
            perscale_improvement_domains.append(domain)
    k632_pass = len(perscale_improvement_domains) > 0
    log(f"  K632 (per-scale >5% over uniform on >=1 domain): {'PASS' if k632_pass else 'FAIL -> KILL'}")
    for domain in DOMAINS:
        log(f"    {domain}: per-scale vs uniform = {domain_results[domain]['perscale_vs_uniform_pct']:+.1f}%")

    # K633: All text is coherent
    all_coherent = True
    for domain in DOMAINS:
        if perscale_scores[domain]["format_mean"] < 0.3:
            all_coherent = False
    k633_pass = all_coherent
    log(f"  K633 (text is coherent): {'PASS' if k633_pass else 'FAIL -> KILL'}")

    # --- Hypothesis verification ---
    log("\n  === HYPOTHESIS VERIFICATION ===")

    # H1: Per-scale beats base on >= 4/5
    h1 = perscale_better_count >= 4
    log(f"  H1 (per-scale beats base >=4/5): {'CONFIRMED' if h1 else 'REFUTED'} ({perscale_better_count}/5)")

    # H2: Per-scale beats uniform on >= 3/5
    h2 = perscale_vs_uniform_better >= 3
    log(f"  H2 (per-scale beats uniform >=3/5): {'CONFIRMED' if h2 else 'REFUTED'} ({perscale_vs_uniform_better}/5)")

    # H3: Legal flips to >= 0%
    h3 = domain_results["legal"]["perscale_adv_pct"] >= 0
    log(f"  H3 (legal flips to >=0%): {'CONFIRMED' if h3 else 'REFUTED'} ({domain_results['legal']['perscale_adv_pct']:+.1f}%)")

    # H4: Finance flips to >= 0%
    h4 = domain_results["finance"]["perscale_adv_pct"] >= 0
    log(f"  H4 (finance flips to >=0%): {'CONFIRMED' if h4 else 'REFUTED'} ({domain_results['finance']['perscale_adv_pct']:+.1f}%)")

    # H5: Math >= +100%
    h5 = domain_results["math"]["perscale_adv_pct"] >= 100
    log(f"  H5 (math >=+100%): {'CONFIRMED' if h5 else 'REFUTED'} ({domain_results['math']['perscale_adv_pct']:+.1f}%)")

    analysis = {
        "kill_criteria": {
            "K631": {"pass": k631_pass, "perscale_worse_count": perscale_worse_count},
            "K632": {"pass": k632_pass, "improvement_domains": perscale_improvement_domains},
            "K633": {"pass": k633_pass},
        },
        "hypotheses": {
            "H1_perscale_beats_base_4of5": h1,
            "H2_perscale_beats_uniform_3of5": h2,
            "H3_legal_flips": h3,
            "H4_finance_flips": h4,
            "H5_math_strong": h5,
        },
        "domain_results": domain_results,
        "counts": {
            "uniform_better_than_base": uniform_better_count,
            "perscale_better_than_base": perscale_better_count,
            "perscale_better_than_uniform": perscale_vs_uniform_better,
        },
    }

    return base_scores, uniform_scores, perscale_scores, analysis


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("GENERATION QUALITY RETEST: PER-DOMAIN OPTIMAL SCALES")
    log("=" * 70)
    log(f"Domains: {DOMAINS}")
    log(f"Optimal scales: {OPTIMAL_SCALES}")
    log(f"Uniform scale: {UNIFORM_SCALE}")
    log(f"Prompts/domain: {NUM_PROMPTS_PER_DOMAIN}")
    log(f"3 conditions x 5 domains x 10 prompts = 150 generations")
    log_memory("start")

    # Load prompts
    prompts_by_domain = {}
    for domain in DOMAINS:
        prompts = extract_prompts_with_answers(domain, NUM_PROMPTS_PER_DOMAIN)
        prompts_by_domain[domain] = prompts
        log(f"  {domain}: {len(prompts)} prompts")

    # Phase 1: Base model
    base_gens, base_time = phase_generate_base(prompts_by_domain)

    # Phase 2a: Uniform scale=20 routing
    uniform_map = {d: UNIFORM_SCALE for d in DOMAINS}
    uniform_gens, uniform_time = phase_generate_routed(
        prompts_by_domain, uniform_map, label="uniform-s20")

    # Phase 2b: Per-domain optimal scale routing
    perscale_gens, perscale_time = phase_generate_routed(
        prompts_by_domain, OPTIMAL_SCALES, label="per-scale")

    # Phase 3: Evaluate all conditions
    base_scores, uniform_scores, perscale_scores, analysis = phase_evaluate(
        prompts_by_domain, base_gens, uniform_gens, perscale_gens)

    # Compile results
    total_time = time.time() - t0
    results = {
        "experiment": "generation_quality_perscale",
        "model": MODEL_ID,
        "optimal_scales": OPTIMAL_SCALES,
        "uniform_scale": UNIFORM_SCALE,
        "domains": DOMAINS,
        "n_prompts": NUM_PROMPTS_PER_DOMAIN,
        "base_scores": base_scores,
        "uniform_scores": uniform_scores,
        "perscale_scores": perscale_scores,
        "analysis": analysis,
        "timing": {
            "base_s": base_time,
            "uniform_s": uniform_time,
            "perscale_s": perscale_time,
            "total_s": total_time,
        },
        "sample_generations": {},
    }

    # Save sample generations for qualitative review
    for domain in DOMAINS:
        results["sample_generations"][domain] = {
            "prompt": prompts_by_domain[domain][0]["instruction"][:200],
            "reference": prompts_by_domain[domain][0]["response"][:200],
            "base": base_gens[domain][0][:300],
            "uniform": uniform_gens[domain][0][:300],
            "perscale": perscale_gens[domain][0][:300],
        }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Summary
    log("\n" + "=" * 70)
    log("KILL CRITERIA SUMMARY")
    log("=" * 70)
    for k, v in analysis["kill_criteria"].items():
        status = "PASS" if v["pass"] else "FAIL -> KILL"
        log(f"  {k}: {status}")

    all_pass = all(v["pass"] for v in analysis["kill_criteria"].values())
    log(f"\n  Overall: {'ALL PASS -> SUPPORTED' if all_pass else 'KILL TRIGGERED'}")

    log("\n" + "=" * 70)
    log("HYPOTHESIS SUMMARY")
    log("=" * 70)
    for k, v in analysis["hypotheses"].items():
        log(f"  {k}: {'CONFIRMED' if v else 'REFUTED'}")


if __name__ == "__main__":
    main()
