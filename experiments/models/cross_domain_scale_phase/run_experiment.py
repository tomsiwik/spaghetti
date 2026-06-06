#!/usr/bin/env python3
"""Cross-Domain Scale Phase Transition: Code and Medical at s={1,2,4,6,8,12,16,20}.

Extends Finding #250 (math phase transition at s*=[4,6]) to code and medical domains.
Tests whether the critical LoRA scale s* is universal or domain-dependent.

Kill criteria:
  K1 (#660): At least one non-math domain shows phase transition (accuracy jump >= 0.3
              between consecutive scales). PASS if yes.
  K2 (#661): Phase transition location s* differs by >4 scale units across domains.
              PASS if domain-dependent, FAIL if universal.
  K3 (#662): All tested domains reach plateau accuracy >= 0.5 at some scale. PASS if yes.

Type: guided-exploration
Platform: Apple M5 Pro 48GB, MLX
"""

import ast
import gc
import json
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

SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0
SEED = 42

# Scale sweep: 8 points matching kill criteria specification
SCALES = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0]
DOMAINS = ["code", "medical"]
ALL_DOMAINS = ["medical", "code", "math", "legal", "finance"]
NUM_PROMPTS = 10

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
# Model utilities (from scale_phase_transition)
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


def premerge_single_adapter(model, skeleton, adapter, domain, scale):
    """Pre-merge a single adapter: W_new = W_base + scale * B^T @ A^T"""
    n_layers = len(model.model.layers)
    merge_count = 0
    di = ALL_DOMAINS.index(domain)

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
# Evaluation metrics
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


def evaluate_response(generated_text, reference_text, domain):
    """Evaluate a single generated response using domain-appropriate metrics."""
    result = {"domain": domain, "generated_len": len(generated_text)}

    if domain == "code":
        syntax_ok = eval_code_syntax(generated_text)
        factual = eval_factual_recall(generated_text, reference_text)
        # Code score: 70% syntax validity + 30% factual recall (token overlap with reference)
        score = 0.7 * (1.0 if syntax_ok else 0.0) + 0.3 * factual["recall"]
        result.update({
            "score": score, "syntax_valid": syntax_ok,
            "factual_recall": factual["recall"], "factual_f1": factual["f1"],
            "method": "syntax_parse + factual_recall",
        })

    elif domain == "medical":
        factual = eval_factual_recall(generated_text, reference_text)
        score = factual["recall"]
        result.update({
            "score": score, "factual_recall": factual["recall"],
            "factual_precision": factual["precision"], "factual_f1": factual["f1"],
            "method": "factual_recall (medical facts vs reference)",
        })

    return result


def check_format_quality(text):
    """Check if generated text has minimum format quality (not gibberish)."""
    if len(text.strip()) < 10:
        return False, "too_short"
    words = text.split()
    if len(words) > 5:
        word_counts = Counter(words)
        most_common_count = word_counts.most_common(1)[0][1]
        if most_common_count > len(words) * 0.5:
            return False, "repetitive"
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
# Phase: Generate base (no adapter) for both domains
# ============================================================================

def phase_generate_base(prompts_by_domain):
    log("\n" + "=" * 70)
    log("PHASE: BASE MODEL (no adapter)")
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
            generated = generate_text(model, tokenizer, formatted, max_tokens=MAX_NEW_TOKENS)
            domain_results.append(generated)
            log(f"  [{domain}][base][{i}] {len(generated)} chars")
        results[domain] = domain_results

    elapsed = time.time() - t0
    del model, tokenizer
    cleanup()
    log_memory("post-base")
    log(f"  Base generation: {elapsed:.1f}s")
    return results, elapsed


# ============================================================================
# Phase: Generate sweep for one domain (single model load)
# ============================================================================

def phase_generate_sweep_domain(domain, prompts):
    """Generate outputs at each scale for one domain. Load model once."""
    log("\n" + "=" * 70)
    log(f"PHASE: SCALE SWEEP - {domain.upper()}")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    base_weights = save_base_weights(model)
    log("  Saved base weights")

    skeleton = load_skeleton()
    log(f"  Loaded skeleton ({len(skeleton)} tensors)")

    adapter = load_adapter(domain)

    all_results = {}
    for scale in SCALES:
        log(f"\n  --- {domain} Scale = {scale} ---")
        mx.random.seed(SEED)
        np.random.seed(SEED)

        restore_base_weights(model, base_weights)
        model = premerge_single_adapter(model, skeleton, adapter, domain, scale)

        scale_results = []
        for i, prompt_data in enumerate(prompts):
            formatted = format_prompt(prompt_data["instruction"])
            generated = generate_text(model, tokenizer, formatted, max_tokens=MAX_NEW_TOKENS)
            scale_results.append(generated)
            log(f"  [{domain}][s={scale}][{i}] {len(generated)} chars")

        all_results[scale] = scale_results
        log(f"  {domain} Scale {scale}: {len(scale_results)} generations")
        log_memory(f"{domain}-s={scale}")

    elapsed = time.time() - t0
    del model, tokenizer, skeleton, adapter, base_weights
    cleanup()
    log_memory(f"post-sweep-{domain}")
    log(f"  {domain} sweep: {elapsed:.1f}s")
    return all_results, elapsed


# ============================================================================
# Phase: Evaluate
# ============================================================================

def phase_evaluate(prompts_by_domain, base_generations, sweep_generations):
    log("\n" + "=" * 70)
    log("PHASE: EVALUATION")
    log("=" * 70)
    t0 = time.time()

    all_results = {}

    for domain in DOMAINS:
        prompts = prompts_by_domain[domain]

        # Evaluate base
        base_scores = []
        base_details = []
        for i, (prompt_data, gen_text) in enumerate(zip(prompts, base_generations[domain])):
            eval_result = evaluate_response(gen_text, prompt_data["response"], domain)
            eval_result["prompt"] = prompt_data["instruction"][:100]
            eval_result["generated_preview"] = gen_text[:300]
            format_ok, format_reason = check_format_quality(gen_text)
            eval_result["format_ok"] = format_ok
            eval_result["format_reason"] = format_reason
            base_scores.append(eval_result["score"])
            base_details.append(eval_result)

        base_mean = np.mean(base_scores)
        log(f"  [{domain}][base] mean_score={base_mean:.3f}")

        # Evaluate each scale
        scale_data = {}
        for scale in SCALES:
            scale_scores = []
            scale_details = []
            for i, (prompt_data, gen_text) in enumerate(
                zip(prompts, sweep_generations[domain][scale])
            ):
                eval_result = evaluate_response(gen_text, prompt_data["response"], domain)
                eval_result["prompt"] = prompt_data["instruction"][:100]
                eval_result["generated_preview"] = gen_text[:300]
                format_ok, format_reason = check_format_quality(gen_text)
                eval_result["format_ok"] = format_ok
                eval_result["format_reason"] = format_reason
                scale_scores.append(eval_result["score"])
                scale_details.append(eval_result)

            scale_mean = np.mean(scale_scores)
            log(f"  [{domain}][s={scale}] mean_score={scale_mean:.3f}")
            scale_data[scale] = {
                "mean_score": round(float(scale_mean), 4),
                "scores": [round(float(s), 4) for s in scale_scores],
                "details": scale_details,
            }

        all_results[domain] = {
            "base": {
                "mean_score": round(float(base_mean), 4),
                "scores": [round(float(s), 4) for s in base_scores],
                "details": base_details,
            },
            "scales": scale_data,
        }

    elapsed = time.time() - t0
    log(f"\n  Evaluation: {elapsed:.1f}s")
    return all_results, elapsed


# ============================================================================
# Analysis: transition detection
# ============================================================================

def analyze_transitions(eval_results):
    """Detect phase transitions and compute sigmoid fits per domain."""
    from scipy.optimize import curve_fit

    analysis = {}

    for domain in DOMAINS:
        domain_data = eval_results[domain]
        base_score = domain_data["base"]["mean_score"]
        scale_scores = {s: domain_data["scales"][s]["mean_score"] for s in SCALES}

        # Largest jump between consecutive scales
        max_jump = 0
        max_jump_scales = (0, 0)
        for i in range(len(SCALES) - 1):
            jump = scale_scores[SCALES[i + 1]] - scale_scores[SCALES[i]]
            if jump > max_jump:
                max_jump = jump
                max_jump_scales = (SCALES[i], SCALES[i + 1])

        # Find s* (scale where first exceeds base + 0.15)
        threshold = base_score + 0.15
        s_star = None
        for s in SCALES:
            if scale_scores[s] > threshold:
                s_star = s
                break

        # Monotonicity check
        accs_ordered = [scale_scores[s] for s in SCALES]
        monotonic = all(accs_ordered[i] <= accs_ordered[i + 1] + 0.05
                        for i in range(len(accs_ordered) - 1))

        # Plateau: max score achieved
        max_score = max(scale_scores[s] for s in SCALES)

        # Sigmoid fit
        s_mid_fit, tau_fit, r2 = None, None, None
        try:
            max_s_score = scale_scores[20.0]
            if max_s_score > base_score + 0.05:
                def sigmoid_model(s, s_mid, tau):
                    return base_score + (max_s_score - base_score) / (1 + np.exp(-(s - s_mid) / tau))

                x_data = np.array(SCALES)
                y_data = np.array([scale_scores[s] for s in SCALES])
                popt, pcov = curve_fit(sigmoid_model, x_data, y_data, p0=[10.0, 3.0],
                                       bounds=([1.0, 0.1], [20.0, 10.0]))
                s_mid_fit, tau_fit = popt
                y_pred = sigmoid_model(x_data, *popt)
                ss_res = np.sum((y_data - y_pred) ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0
        except Exception as e:
            log(f"  [{domain}] Sigmoid fit failed: {e}")

        # Per-prompt transition analysis
        per_prompt_transitions = []
        base_scores_list = domain_data["base"]["scores"]
        for pi in range(NUM_PROMPTS):
            prompt_curve = {"prompt_idx": pi}
            prompt_curve["base_score"] = base_scores_list[pi]
            prompt_scores = {}
            for s in SCALES:
                prompt_scores[str(s)] = domain_data["scales"][s]["scores"][pi]
            prompt_curve["scale_scores"] = prompt_scores

            # Find per-prompt transition scale
            base_s = base_scores_list[pi]
            per_prompt_s_star = None
            for s in SCALES:
                if domain_data["scales"][s]["scores"][pi] > base_s + 0.15:
                    per_prompt_s_star = s
                    break
            prompt_curve["s_star"] = per_prompt_s_star
            per_prompt_transitions.append(prompt_curve)

        analysis[domain] = {
            "base_score": round(float(base_score), 4),
            "max_score": round(float(max_score), 4),
            "max_jump": round(float(max_jump), 4),
            "max_jump_between": list(max_jump_scales),
            "s_star": s_star,
            "monotonic": bool(monotonic),
            "sigmoid_fit": {
                "s_mid": round(float(s_mid_fit), 2) if s_mid_fit is not None else None,
                "tau": round(float(tau_fit), 2) if tau_fit is not None else None,
                "r_squared": round(float(r2), 4) if r2 is not None else None,
            },
            "transition_curve": {str(s): round(float(scale_scores[s]), 4) for s in SCALES},
            "per_prompt_transitions": per_prompt_transitions,
        }

    return analysis


# ============================================================================
# Kill criteria assessment
# ============================================================================

def assess_kill_criteria(analysis):
    """Evaluate K1, K2, K3 from the analysis."""

    # K1: At least one non-math domain shows phase transition (jump >= 0.3)
    k1_jumps = {d: analysis[d]["max_jump"] for d in DOMAINS}
    k1_pass = any(j >= 0.3 for j in k1_jumps.values())

    # K2: s* differs by >4 scale units across domains (include math s*=5 from F250)
    # Math s* is in [4,6], use midpoint 5
    math_s_star = 5.0
    s_stars = {"math": math_s_star}
    for d in DOMAINS:
        if analysis[d]["s_star"] is not None:
            s_stars[d] = analysis[d]["s_star"]

    if len(s_stars) >= 2:
        all_s_stars = list(s_stars.values())
        max_diff = max(all_s_stars) - min(all_s_stars)
        k2_pass = max_diff > 4
    else:
        max_diff = 0
        k2_pass = False  # Cannot assess

    # K3: All tested domains reach plateau >= 0.5
    k3_scores = {d: analysis[d]["max_score"] for d in DOMAINS}
    k3_pass = all(s >= 0.5 for s in k3_scores.values())

    return {
        "K1": {
            "description": "At least one non-math domain shows phase transition (jump >= 0.3)",
            "domain_jumps": {d: round(float(v), 4) for d, v in k1_jumps.items()},
            "result": "PASS" if k1_pass else "FAIL",
        },
        "K2": {
            "description": "Phase transition location s* differs by >4 scale units across domains",
            "s_stars": {d: v for d, v in s_stars.items()},
            "max_diff": round(float(max_diff), 1),
            "result": "PASS" if k2_pass else "FAIL",
        },
        "K3": {
            "description": "All tested domains reach plateau accuracy >= 0.5",
            "domain_max_scores": {d: round(float(v), 4) for d, v in k3_scores.items()},
            "result": "PASS" if k3_pass else "FAIL",
        },
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("CROSS-DOMAIN SCALE PHASE TRANSITION")
    log("=" * 70)
    log(f"Model: {MODEL_ID}")
    log(f"Domains: {DOMAINS}")
    log(f"Prompts per domain: {NUM_PROMPTS}")
    log(f"Scales: {SCALES}")
    log(f"Max tokens: {MAX_NEW_TOKENS}")
    log_memory("start")

    # Load prompts
    prompts_by_domain = {}
    for domain in DOMAINS:
        prompts_by_domain[domain] = extract_prompts_with_answers(domain, NUM_PROMPTS)
        log(f"Loaded {len(prompts_by_domain[domain])} {domain} prompts")

    # Generate base
    base_gen, base_time = phase_generate_base(prompts_by_domain)

    # Generate sweeps (one domain at a time to manage memory)
    sweep_gen = {}
    sweep_times = {}
    for domain in DOMAINS:
        domain_sweep, domain_time = phase_generate_sweep_domain(domain, prompts_by_domain[domain])
        sweep_gen[domain] = domain_sweep
        sweep_times[domain] = domain_time

    # Evaluate
    eval_results, eval_time = phase_evaluate(prompts_by_domain, base_gen, sweep_gen)

    # Analyze transitions
    analysis = analyze_transitions(eval_results)

    # Kill criteria
    kill_criteria = assess_kill_criteria(analysis)

    # ========================================================================
    # Print summary
    # ========================================================================
    log("\n" + "=" * 70)
    log("TRANSITION CURVES")
    log("=" * 70)

    for domain in DOMAINS:
        a = analysis[domain]
        log(f"\n  {domain.upper()}:")
        log(f"  {'Scale':>8} {'Score':>10}")
        log(f"  {'-' * 20}")
        log(f"  {'base':>8} {a['base_score']:>10.3f}")
        for s in SCALES:
            log(f"  {s:>8.1f} {a['transition_curve'][str(s)]:>10.3f}")
        log(f"  Max jump: {a['max_jump']:.3f} between s={a['max_jump_between'][0]} and s={a['max_jump_between'][1]}")
        log(f"  s*: {a['s_star']}")
        log(f"  Monotonic: {a['monotonic']}")
        if a['sigmoid_fit']['s_mid'] is not None:
            log(f"  Sigmoid: s_mid={a['sigmoid_fit']['s_mid']}, tau={a['sigmoid_fit']['tau']}, R2={a['sigmoid_fit']['r_squared']}")

    log("\n" + "=" * 70)
    log("KILL CRITERIA")
    log("=" * 70)
    for k, v in kill_criteria.items():
        log(f"\n  {k}: {v['result']}")
        log(f"     {v['description']}")
        for detail_k, detail_v in v.items():
            if detail_k not in ("description", "result"):
                log(f"     {detail_k}: {detail_v}")

    # Per-prompt transition analysis
    log("\n" + "=" * 70)
    log("PER-PROMPT TRANSITION ANALYSIS")
    log("=" * 70)
    for domain in DOMAINS:
        a = analysis[domain]
        log(f"\n  {domain.upper()} per-prompt s*:")
        for pt in a["per_prompt_transitions"]:
            log(f"    Prompt {pt['prompt_idx']}: s*={pt['s_star']} "
                f"(base={pt['base_score']:.3f}, "
                f"scores at s=6:{pt['scale_scores'].get('6.0', 'N/A'):.3f}, "
                f"s=20:{pt['scale_scores'].get('20.0', 'N/A'):.3f})")

    # ========================================================================
    # Save results
    # ========================================================================
    results = {
        "experiment": "cross_domain_scale_phase",
        "description": "Scale sweep on code and medical domains to test phase transition universality",
        "model": MODEL_ID,
        "domains": DOMAINS,
        "n_prompts": NUM_PROMPTS,
        "scales": SCALES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "math_s_star_from_F250": [4, 6],
        "evaluation": eval_results,
        "analysis": analysis,
        "kill_criteria": kill_criteria,
        "timing": {
            "base_gen_s": round(base_time, 1),
            **{f"sweep_{d}_s": round(sweep_times[d], 1) for d in DOMAINS},
            "eval_s": round(eval_time, 1),
            "total_s": round(time.time() - t0, 1),
        },
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {time.time() - t0:.0f}s")

    return results


if __name__ == "__main__":
    main()
