#!/usr/bin/env python3
"""Targeted test: Individual vs Routed 12pp GSM8K gap investigation.

The main experiment found:
  - Individual math adapter on GSM8K: 30% (TernaryLoRALinear, single adapter loaded)
  - Routed to math on GSM8K: 42% (RoutedMultiAdapterLoRALinear, all 5 adapters loaded,
    weights=[0,0,1,0,0])

These should be mathematically equivalent. This script isolates whether the gap
comes from (a) having all 5 B matrices loaded, or (b) the code path itself.

Test conditions:
  A. Individual: TernaryLoRALinear with math adapter only (replicates original)
  B. Routed-all: RoutedMultiAdapterLoRALinear with all 5 B matrices, weights=[0,0,1,0,0]
  C. Routed-math-only: RoutedMultiAdapterLoRALinear with ONLY math B matrix loaded
     (zeros for all other B matrices — same as initialization)

If C ~ A (~30%), the gap comes from the other loaded B matrices.
If C ~ B (~42%), the gap comes from the code path itself.

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

# Memory safety
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = len(DOMAINS)
MAX_NEW_TOKENS = 256
GSM8K_N = 50


def log(msg):
    print(msg, flush=True)


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
# Import shared code from run_experiment
# ============================================================================

from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.bitlinear_layers import BitLinear


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
# LoRA layer classes (copied from run_experiment.py)
# ============================================================================

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


class TernaryLoRALinear(nn.Module):
    def __init__(self, base_linear, rank=16, scale=20.0, a_init=None):
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


class RoutedMultiAdapterLoRALinear(nn.Module):
    def __init__(self, base_linear, rank=16, scale=20.0, a_inits=None):
        super().__init__()
        out_features = base_linear.weight.shape[0]
        self.linear = base_linear
        self.scale = scale
        self.rank = rank
        self.n_experts = len(a_inits) if a_inits else 0
        self.a_matrices = a_inits if a_inits else []
        self.b_matrices = [mx.zeros((rank, out_features)) for _ in range(self.n_experts)]
        self.linear.freeze()
        self._routing_weights = [1.0 / self.n_experts] * self.n_experts if self.n_experts > 0 else []

    def set_routing_weights(self, weights):
        self._routing_weights = weights

    def __call__(self, x):
        base_out = self.linear(x)
        if self.n_experts == 0:
            return base_out
        lora_sum = mx.zeros_like(base_out)
        for i in range(self.n_experts):
            w = self._routing_weights[i]
            if w < 1e-6:
                continue
            b = self.b_matrices[i]
            alpha = mx.mean(mx.abs(b))
            b_scaled = b / (alpha + 1e-7)
            b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
            b_ste = b + mx.stop_gradient(b_q - b)
            lora_sum = lora_sum + w * ((x @ self.a_matrices[i]) @ b_ste)
        return base_out + lora_sum * self.scale


# ============================================================================
# Model setup
# ============================================================================

def load_skeleton():
    skeleton_path = ADAPTERS_DIR / "grassmannian_skeleton.npz"
    return dict(np.load(str(skeleton_path)))


def apply_single_adapter(model, skeleton, domain_idx, domain_name):
    """Apply single adapter using TernaryLoRALinear."""
    n_layers = len(model.model.layers)
    a_matrices = {}
    for li in range(n_layers):
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_{domain_idx}"
            if skey in skeleton:
                a_matrices[(li, key)] = skeleton[skey]

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
            a_key = (li, key)
            a_mx = mx.array(a_matrices[a_key]).astype(mx.bfloat16) if a_key in a_matrices else None
            lora = TernaryLoRALinear(module, rank=LORA_RANK, scale=LORA_SCALE, a_init=a_mx)
            lora_updates.append((key, lora))
            count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    adapter_path = ADAPTERS_DIR / domain_name / "adapter.npz"
    adapter_params = dict(mx.load(str(adapter_path)))
    model.update(tree_unflatten(list(adapter_params.items())))
    mx.eval(model.parameters())
    log(f"  Applied {domain_name} adapter via TernaryLoRALinear ({count} layers)")
    return model


def apply_routed_adapter(model, skeleton, load_all_b_matrices=True):
    """Apply routed multi-adapter.

    If load_all_b_matrices=True: load all 5 domain B matrices (replicates original routed).
    If load_all_b_matrices=False: load ONLY math B matrix (zeros for others).
    """
    n_layers = len(model.model.layers)
    all_adapter_params = {}
    if load_all_b_matrices:
        for di, domain in enumerate(DOMAINS):
            adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
            all_adapter_params[domain] = dict(mx.load(str(adapter_path)))
    else:
        # Only load math adapter
        adapter_path = ADAPTERS_DIR / "math" / "adapter.npz"
        all_adapter_params["math"] = dict(mx.load(str(adapter_path)))

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
            a_inits = []
            for di in range(N_DOMAINS):
                skey = f"layer_{li}_{key}_domain_{di}"
                if skey in skeleton:
                    a_inits.append(mx.array(skeleton[skey]).astype(mx.bfloat16))
                else:
                    a_inits.append(None)
            routed_lora = RoutedMultiAdapterLoRALinear(
                module, rank=LORA_RANK, scale=LORA_SCALE, a_inits=a_inits
            )
            # Load B matrices
            for di, domain in enumerate(DOMAINS):
                if domain in all_adapter_params:
                    b_key = f"model.layers.{li}.{key}.lora_b"
                    if b_key in all_adapter_params[domain]:
                        routed_lora.b_matrices[di] = all_adapter_params[domain][b_key]
                # Otherwise B stays as zeros (initialized in __init__)
            lora_updates.append((key, routed_lora))
            count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    mx.eval(model.parameters())
    mode = "all 5 B matrices" if load_all_b_matrices else "math B only (others zero)"
    log(f"  Applied routed multi-adapter ({mode}, {count} layers)")
    return model


def set_all_routing_weights(model, weights):
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, RoutedMultiAdapterLoRALinear):
                module.set_routing_weights(weights)


# ============================================================================
# GSM8K evaluation (copied from run_experiment.py)
# ============================================================================

def load_gsm8k_data(n=50):
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


def format_gsm8k_prompt(question):
    return (
        f"### Instruction:\n"
        f"Solve the following math problem step by step. "
        f"Show your work and give the final numerical answer after ####.\n\n"
        f"{question}\n\n"
        f"### Response:\n"
    )


def extract_gsm8k_answer(generated_text):
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
    if predicted is None or ground_truth is None:
        return False
    if ground_truth == 0:
        return abs(predicted) < tolerance
    return abs(predicted - ground_truth) / abs(ground_truth) < tolerance


def generate_text(model, tokenizer, prompt, max_tokens=256):
    try:
        sampler = make_sampler(temp=0.1, top_p=0.95)
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


def eval_gsm8k(config_name, model, tokenizer, problems):
    """Evaluate on GSM8K and return accuracy."""
    log(f"\n  [GSM8K] Evaluating {config_name}...")
    t0 = time.time()
    correct = 0
    total = len(problems)

    for i, prob in enumerate(problems):
        prompt = format_gsm8k_prompt(prob["question"])
        generated = generate_text(model, tokenizer, prompt, max_tokens=MAX_NEW_TOKENS)
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
    log(f"  [GSM8K] {config_name}: {correct}/{total} = {accuracy:.3f} ({elapsed:.1f}s)")
    return {"accuracy": accuracy, "correct": correct, "total": total, "time_s": elapsed}


# ============================================================================
# Three test conditions
# ============================================================================

def condition_a_individual(problems):
    """Condition A: TernaryLoRALinear with math adapter only."""
    log("\n" + "=" * 70)
    log("CONDITION A: Individual (TernaryLoRALinear, math only)")
    log("=" * 70)

    skeleton = load_skeleton()
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    math_idx = DOMAINS.index("math")
    model = apply_single_adapter(model, skeleton, math_idx, "math")
    model.freeze()
    log_memory("post-load-A")

    result = eval_gsm8k("individual(math)", model, tokenizer, problems)

    cleanup(model, tokenizer)
    del skeleton
    gc.collect()
    return result


def condition_b_routed_all(problems):
    """Condition B: RoutedMultiAdapterLoRALinear with ALL 5 B matrices, route to math."""
    log("\n" + "=" * 70)
    log("CONDITION B: Routed-all (RoutedMultiAdapterLoRALinear, all 5 B matrices, w=[0,0,1,0,0])")
    log("=" * 70)

    skeleton = load_skeleton()
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model = apply_routed_adapter(model, skeleton, load_all_b_matrices=True)
    model.freeze()

    math_idx = DOMAINS.index("math")
    weights = [0.0] * N_DOMAINS
    weights[math_idx] = 1.0
    set_all_routing_weights(model, weights)
    log_memory("post-load-B")

    result = eval_gsm8k("routed-all(math)", model, tokenizer, problems)

    cleanup(model, tokenizer)
    del skeleton
    gc.collect()
    return result


def condition_c_routed_math_only(problems):
    """Condition C: RoutedMultiAdapterLoRALinear with ONLY math B matrix, route to math."""
    log("\n" + "=" * 70)
    log("CONDITION C: Routed-math-only (RoutedMultiAdapterLoRALinear, only math B, w=[0,0,1,0,0])")
    log("=" * 70)

    skeleton = load_skeleton()
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model = apply_routed_adapter(model, skeleton, load_all_b_matrices=False)
    model.freeze()

    math_idx = DOMAINS.index("math")
    weights = [0.0] * N_DOMAINS
    weights[math_idx] = 1.0
    set_all_routing_weights(model, weights)
    log_memory("post-load-C")

    result = eval_gsm8k("routed-math-only(math)", model, tokenizer, problems)

    cleanup(model, tokenizer)
    del skeleton
    gc.collect()
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    log("=" * 70)
    log("ROUTING GAP INVESTIGATION")
    log("Individual (30%) vs Routed (42%) on GSM8K -- 12pp gap")
    log("=" * 70)
    t0 = time.time()

    problems = load_gsm8k_data(GSM8K_N)

    result_a = condition_a_individual(problems)
    result_b = condition_b_routed_all(problems)
    result_c = condition_c_routed_math_only(problems)

    # Analysis
    log("\n" + "=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)
    log(f"  A. Individual (TernaryLoRALinear):              {result_a['accuracy']:.3f} ({result_a['correct']}/{result_a['total']})")
    log(f"  B. Routed-all (5 B matrices, w=[0,0,1,0,0]):   {result_b['accuracy']:.3f} ({result_b['correct']}/{result_b['total']})")
    log(f"  C. Routed-math-only (1 B matrix, w=[0,0,1,0,0]): {result_c['accuracy']:.3f} ({result_c['correct']}/{result_c['total']})")

    log("\nINTERPRETATION:")
    a_acc = result_a['accuracy']
    b_acc = result_b['accuracy']
    c_acc = result_c['accuracy']

    if abs(c_acc - a_acc) <= 0.04:  # C ~ A
        log("  C ~ A: The gap comes from having other B matrices loaded.")
        log("  The non-math B matrices affect the routed computation even at w=0.")
    elif abs(c_acc - b_acc) <= 0.04:  # C ~ B
        log("  C ~ B: The gap comes from the code path itself (RoutedMultiAdapterLoRALinear vs TernaryLoRALinear).")
        log("  The other B matrices do NOT explain the difference.")
    else:
        log("  C is between A and B: Both factors contribute.")
        log(f"  Code path effect: {abs(c_acc - a_acc):.3f}")
        log(f"  Other B matrices effect: {abs(b_acc - c_acc):.3f}")

    elapsed = time.time() - t0
    log(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    output = {
        "test": "routing_gap_investigation",
        "condition_a_individual": result_a,
        "condition_b_routed_all": result_b,
        "condition_c_routed_math_only": result_c,
        "interpretation": {
            "a_acc": a_acc,
            "b_acc": b_acc,
            "c_acc": c_acc,
            "a_vs_c_gap": abs(c_acc - a_acc),
            "b_vs_c_gap": abs(b_acc - c_acc),
        },
        "total_time_s": round(elapsed, 1),
    }
    output_path = EXPERIMENT_DIR / "routing_gap_results.json"
    output_path.write_text(json.dumps(output, indent=2))
    log(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
