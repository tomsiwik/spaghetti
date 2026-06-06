#!/usr/bin/env python3
"""Scale Phase Transition: Math Reasoning Activation Boundary.

Sweeps LoRA scale from 1 to 20 on math domain to map the behavioral transition
from FORMAT regime (0% math gain) to CAPABILITY regime (700% math gain).

Kill criteria:
  K1: No scale in [4, 16] achieves math accuracy > 0.20 (sharp transition only in [16,20])
  K2: All scales score <= 0.20 or >= 0.60 (pure phase transition, no intermediates)
  K3: Math accuracy is non-monotonic in scale

Type: guided-exploration
Platform: Apple M5 Pro 48GB, MLX
"""

import gc
import json
import re
import time
from pathlib import Path

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

# Scale sweep: 9 points from 1 to 20
SCALES = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]
DOMAIN = "math"
NUM_PROMPTS = 10

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

DOMAINS = ["medical", "code", "math", "legal", "finance"]


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
# Math evaluation
# ============================================================================

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


def has_chain_of_thought(text):
    """Detect whether the generation contains chain-of-thought reasoning structure."""
    cot_indicators = [
        r'step\s*\d',
        r'first[,\s]',
        r'then[,\s]',
        r'therefore',
        r'so\s+(?:the|we)',
        r'let\s*(?:\'s|us)',
        r'<<.*?>>',
        r'\d+\s*[\+\-\*\/\×]\s*\d+',
        r'=\s*\d+',
    ]
    count = 0
    text_lower = text.lower()
    for pattern in cot_indicators:
        if re.search(pattern, text_lower):
            count += 1
    return count >= 2


# ============================================================================
# Data loading
# ============================================================================

def extract_prompts_with_answers(n_prompts=10):
    val_path = DATA_DIR / DOMAIN / "valid.jsonl"
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
# Phase: Generate base (no adapter)
# ============================================================================

def phase_generate_base(prompts):
    log("\n" + "=" * 70)
    log("PHASE: BASE MODEL (no adapter)")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    mx.random.seed(SEED)
    np.random.seed(SEED)

    results = []
    for i, prompt_data in enumerate(prompts):
        formatted = format_prompt(prompt_data["instruction"])
        generated = generate_text(model, tokenizer, formatted, max_tokens=MAX_NEW_TOKENS)
        results.append(generated)
        log(f"  [base][{i}] {len(generated)} chars")

    elapsed = time.time() - t0
    del model, tokenizer
    cleanup()
    log_memory("post-base")
    log(f"  Base generation: {elapsed:.1f}s")
    return results, elapsed


# ============================================================================
# Phase: Generate at all scales (single model load)
# ============================================================================

def phase_generate_sweep(prompts):
    """Generate math outputs at each scale. Load model once, restore base weights between scales."""
    log("\n" + "=" * 70)
    log("PHASE: SCALE SWEEP")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    base_weights = save_base_weights(model)
    log("  Saved base weights")

    skeleton = load_skeleton()
    log(f"  Loaded skeleton ({len(skeleton)} tensors)")

    adapter = load_adapter(DOMAIN)

    all_results = {}
    for scale in SCALES:
        log(f"\n  --- Scale = {scale} ---")
        mx.random.seed(SEED)
        np.random.seed(SEED)

        restore_base_weights(model, base_weights)
        model = premerge_single_adapter(model, skeleton, adapter, DOMAIN, scale)

        scale_results = []
        for i, prompt_data in enumerate(prompts):
            formatted = format_prompt(prompt_data["instruction"])
            generated = generate_text(model, tokenizer, formatted, max_tokens=MAX_NEW_TOKENS)
            scale_results.append(generated)
            log(f"  [s={scale}][{i}] {len(generated)} chars")

        all_results[scale] = scale_results
        log(f"  Scale {scale}: {len(scale_results)} generations")
        log_memory(f"s={scale}")

    elapsed = time.time() - t0
    del model, tokenizer, skeleton, adapter, base_weights
    cleanup()
    log_memory("post-sweep")
    log(f"  Sweep generation: {elapsed:.1f}s")
    return all_results, elapsed


# ============================================================================
# Phase: Evaluate
# ============================================================================

def phase_evaluate(prompts, base_generations, sweep_generations):
    log("\n" + "=" * 70)
    log("PHASE: EVALUATION")
    log("=" * 70)
    t0 = time.time()

    def eval_set(generations, label):
        results = []
        for i, (prompt_data, gen_text) in enumerate(zip(prompts, generations)):
            gt_answer = extract_ground_truth_answer(prompt_data["response"])
            gen_answer = extract_math_answer(gen_text)
            correct = eval_math_correct(gen_answer, gt_answer)
            cot = has_chain_of_thought(gen_text)
            results.append({
                "prompt": prompt_data["instruction"][:100],
                "gt_answer": gt_answer,
                "gen_answer": gen_answer,
                "correct": correct,
                "has_cot": cot,
                "generated_preview": gen_text[:300],
                "generated_len": len(gen_text),
            })
        accuracy = sum(1 for r in results if r["correct"]) / len(results)
        cot_rate = sum(1 for r in results if r["has_cot"]) / len(results)
        log(f"  [{label}] accuracy={accuracy:.2f} cot_rate={cot_rate:.2f}")
        return results, accuracy, cot_rate

    # Evaluate base
    base_results, base_acc, base_cot = eval_set(base_generations, "base")

    # Evaluate each scale
    scale_results = {}
    scale_accuracies = {}
    scale_cot_rates = {}
    for scale in SCALES:
        results, acc, cot = eval_set(sweep_generations[scale], f"s={scale}")
        scale_results[scale] = results
        scale_accuracies[scale] = acc
        scale_cot_rates[scale] = cot

    elapsed = time.time() - t0
    log(f"\n  Evaluation: {elapsed:.1f}s")
    return base_results, base_acc, base_cot, scale_results, scale_accuracies, scale_cot_rates, elapsed


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("SCALE PHASE TRANSITION: MATH REASONING ACTIVATION")
    log("=" * 70)
    log(f"Model: {MODEL_ID}")
    log(f"Domain: {DOMAIN}")
    log(f"Prompts: {NUM_PROMPTS}")
    log(f"Scales: {SCALES}")
    log(f"Max tokens: {MAX_NEW_TOKENS}")
    log_memory("start")

    # Load prompts
    prompts = extract_prompts_with_answers(NUM_PROMPTS)
    log(f"Loaded {len(prompts)} math prompts")

    # Generate
    base_gen, base_time = phase_generate_base(prompts)
    sweep_gen, sweep_time = phase_generate_sweep(prompts)

    # Evaluate
    (base_results, base_acc, base_cot,
     scale_results, scale_accs, scale_cots, eval_time) = phase_evaluate(
        prompts, base_gen, sweep_gen)

    # ========================================================================
    # Transition curve
    # ========================================================================
    log("\n" + "=" * 70)
    log("TRANSITION CURVE")
    log("=" * 70)
    log(f"\n{'Scale':>8} {'Accuracy':>10} {'CoT Rate':>10} {'Correct':>8}")
    log("-" * 40)
    log(f"{'base':>8} {base_acc:>10.2f} {base_cot:>10.2f} {sum(1 for r in base_results if r['correct']):>8d}/10")
    for scale in SCALES:
        acc = scale_accs[scale]
        cot = scale_cots[scale]
        n_correct = sum(1 for r in scale_results[scale] if r["correct"])
        log(f"{scale:>8.1f} {acc:>10.2f} {cot:>10.2f} {n_correct:>8d}/10")

    # ========================================================================
    # Kill criteria
    # ========================================================================
    log("\n" + "=" * 70)
    log("KILL CRITERIA")
    log("=" * 70)

    # K1: No scale in [4, 16] achieves accuracy > 0.20
    intermediate_scales = [s for s in SCALES if 4 <= s <= 16]
    max_intermediate_acc = max(scale_accs[s] for s in intermediate_scales)
    k1_pass = max_intermediate_acc > 0.20
    log(f"\n  K1: Max accuracy in [4,16] = {max_intermediate_acc:.2f}")
    log(f"      {'PASS (intermediate activation exists)' if k1_pass else 'FAIL (sharp transition only in [16,20])'}")

    # K2: All scales score <= 0.20 or >= 0.60 (no intermediates)
    intermediate_values = [scale_accs[s] for s in SCALES if 0.20 < scale_accs[s] < 0.60]
    k2_has_intermediates = len(intermediate_values) > 0
    log(f"\n  K2: Intermediate accuracy values (0.20 < acc < 0.60): {[round(v, 2) for v in intermediate_values]}")
    log(f"      {'PASS (gradual transition)' if k2_has_intermediates else 'FAIL (pure phase transition)'}")

    # K3: Non-monotonic
    accs_ordered = [scale_accs[s] for s in SCALES]
    monotonic = all(accs_ordered[i] <= accs_ordered[i + 1] + 0.05  # allow small noise
                    for i in range(len(accs_ordered) - 1))
    k3_pass = monotonic
    log(f"\n  K3: Monotonic (with 0.05 tolerance): {monotonic}")
    log(f"      {'PASS (monotonic)' if k3_pass else 'FAIL (non-monotonic — perturbation model wrong)'}")

    # Detect largest jump between adjacent scales
    max_jump = 0
    max_jump_scales = (0, 0)
    for i in range(len(SCALES) - 1):
        jump = scale_accs[SCALES[i + 1]] - scale_accs[SCALES[i]]
        if jump > max_jump:
            max_jump = jump
            max_jump_scales = (SCALES[i], SCALES[i + 1])

    log(f"\n  Largest jump: {max_jump:.2f} between s={max_jump_scales[0]} and s={max_jump_scales[1]}")
    if max_jump >= 0.40:
        log(f"  -> Supports Model 1 (phase transition)")
    elif max_jump <= 0.20:
        log(f"  -> Supports Model 2 (gradual sigmoid)")
    else:
        log(f"  -> Ambiguous (moderate jump)")

    # ========================================================================
    # Model discrimination
    # ========================================================================
    log("\n" + "=" * 70)
    log("MODEL DISCRIMINATION")
    log("=" * 70)

    # Fit sigmoid: f(s) = base + gain * sigmoid((s - s_mid) / tau)
    from scipy.optimize import curve_fit

    def sigmoid_model(s, s_mid, tau):
        return base_acc + (scale_accs[20.0] - base_acc) / (1 + np.exp(-(s - s_mid) / tau))

    try:
        x_data = np.array(SCALES)
        y_data = np.array([scale_accs[s] for s in SCALES])
        popt, pcov = curve_fit(sigmoid_model, x_data, y_data, p0=[10.0, 3.0],
                               bounds=([1.0, 0.1], [20.0, 10.0]))
        s_mid_fit, tau_fit = popt
        y_pred = sigmoid_model(x_data, *popt)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        log(f"  Sigmoid fit: s_mid={s_mid_fit:.1f}, tau={tau_fit:.1f}, R²={r2:.3f}")
        if tau_fit < 1.5:
            log(f"  -> Sharp transition (tau < 1.5): phase transition model")
        elif tau_fit > 4.0:
            log(f"  -> Broad transition (tau > 4.0): gradual crossover model")
        else:
            log(f"  -> Moderate transition width")
    except Exception as e:
        log(f"  Sigmoid fit failed: {e}")
        s_mid_fit, tau_fit, r2 = None, None, None

    # ========================================================================
    # Results
    # ========================================================================
    results = {
        "experiment": "scale_phase_transition",
        "description": "Scale sweep on math domain to map FORMAT-to-CAPABILITY transition",
        "model": MODEL_ID,
        "domain": DOMAIN,
        "n_prompts": NUM_PROMPTS,
        "scales": SCALES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "base": {
            "accuracy": round(base_acc, 4),
            "cot_rate": round(base_cot, 4),
            "details": base_results,
        },
        "transition_curve": {
            str(s): {
                "accuracy": round(scale_accs[s], 4),
                "cot_rate": round(scale_cots[s], 4),
                "n_correct": sum(1 for r in scale_results[s] if r["correct"]),
                "n_cot": sum(1 for r in scale_results[s] if r["has_cot"]),
                "details": scale_results[s],
            }
            for s in SCALES
        },
        "kill_criteria": {
            "K1": {
                "description": "No scale in [4,16] achieves accuracy > 0.20",
                "max_intermediate_accuracy": round(max_intermediate_acc, 4),
                "result": "PASS" if k1_pass else "FAIL",
            },
            "K2": {
                "description": "All scores <= 0.20 or >= 0.60 (no intermediates)",
                "intermediate_values": [round(v, 4) for v in intermediate_values],
                "result": "PASS" if k2_has_intermediates else "FAIL",
            },
            "K3": {
                "description": "Non-monotonic accuracy",
                "monotonic": monotonic,
                "result": "PASS" if k3_pass else "FAIL",
            },
        },
        "analysis": {
            "largest_jump": round(max_jump, 4),
            "largest_jump_between": list(max_jump_scales),
            "sigmoid_fit": {
                "s_mid": round(s_mid_fit, 2) if s_mid_fit is not None else None,
                "tau": round(tau_fit, 2) if tau_fit is not None else None,
                "r_squared": round(r2, 4) if r2 is not None else None,
            },
        },
        "timing": {
            "base_gen_s": round(base_time, 1),
            "sweep_gen_s": round(sweep_time, 1),
            "eval_s": round(eval_time, 1),
            "total_s": round(time.time() - t0, 1),
        },
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"\n{'Scale':>8} {'Accuracy':>10} {'CoT':>6}")
    log("-" * 28)
    log(f"{'base':>8} {base_acc:>10.2f} {base_cot:>6.2f}")
    for s in SCALES:
        log(f"{s:>8.1f} {scale_accs[s]:>10.2f} {scale_cots[s]:>6.2f}")

    log(f"\n  K1: {results['kill_criteria']['K1']['result']}")
    log(f"  K2: {results['kill_criteria']['K2']['result']}")
    log(f"  K3: {results['kill_criteria']['K3']['result']}")
    if s_mid_fit is not None:
        log(f"  Sigmoid: s_mid={s_mid_fit:.1f}, tau={tau_fit:.1f}, R²={r2:.3f}")
    log(f"\n  Total time: {time.time() - t0:.0f}s")

    return results


if __name__ == "__main__":
    main()
