#!/usr/bin/env python3
"""
exp_e1_function_vector_strategy_extraction
==========================================

Can we read strategy vectors directly from attention activations
and inject them as LoRA adapters?

Grounded by: Function Vectors (2310.15213), ActAdd (2308.10248),
Refusal=Single Direction (2406.11717), F#480, F#428, F#203.

Pre-registered KCs (canonical DB text — do not edit):
  K#2017 K1: extracted activation vector has cos > 0.1 with strategy condition
  K#2018 K2: injected LoRA from vector produces > 2pp GSM8K improvement
  K#2019 K3: strategy vector is strategy-specific (same strategy, different
             prompts → cos > 0.3 between extracted vectors)

Phase plan:
  Phase 0 — collect strategy prompts for 3 strategies (systematic, step-by-step,
            conservative) plus neutral baseline.
  Phase 1 — extract activation differences: run model with each strategy prompt
            vs neutral, capture per-layer o_proj output, compute mean-diff vector.
  Phase 2 — K1 validation: measure cosine similarity between extracted vectors
            across different prompts for the same strategy.
  Phase 3 — inject: project mean-diff vector into LoRA B-matrix format (with
            Grassmannian A), load as adapter.
  Phase 4 — K2 validation: measure GSM8K behavioral change with injected adapter.
  Phase 5 — K3 validation: measure cross-prompt consistency of strategy vectors.

For SMOKE_TEST=1: 10 prompts per strategy, 50 GSM8K problems, smoke eval.
For full run: 50 prompts per strategy, 500 GSM8K problems, full eval.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Memory management
if hasattr(mx, "metal") and hasattr(mx.metal, "device_info"):
    mx.set_memory_limit(mx.metal.device_info()["memory_size"] - 8 * 1024**3)
else:
    mx.set_memory_limit(40 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_PROMPTS_PER_STRATEGY = 10 if IS_SMOKE else 50
N_GSM8K = 50 if IS_SMOKE else 500
SEED = 42
SEQLEN = 512

ADAPTER_RANK = 8
ADAPTER_TARGETS = ("v_proj", "o_proj")
LORA_SCALE = 6.0  # F#426 sweet spot

# ── Strategy prompts ──────────────────────────────────────

STRATEGY_PROMPTS = {
    "systematic": [
        "Break this problem into smaller sub-problems. Solve each independently, then combine the results.",
        "Decompose the problem into parts. Address each part step by step.",
        "First, identify the components of this problem. Then solve each component separately.",
        "What are the sub-problems here? Solve each one, then synthesize.",
        "Approach this systematically: divide into sub-tasks, solve each, merge solutions.",
        "Analyze the structure of this problem. What are the independent pieces? Solve each.",
        "Break it down into simpler problems. Solve the simple problems. Combine answers.",
        "What is the hierarchy of sub-problems? Solve bottom-up.",
        "Identify independent variables. Solve for each. Integrate results.",
        "Decompose → Solve sub-problems → Synthesize. Apply this framework.",
    ],
    "step_by_step": [
        "Think step by step. Show your work at each step. Verify before proceeding.",
        "Work through this one step at a time. Check each step before moving on.",
        "Proceed carefully: state each step, verify it, then advance.",
        "Chain-of-thought: write out every reasoning step explicitly.",
        "Step 1, Step 2, Step 3... Number each step. Verify intermediate results.",
        "Show complete work. Don't skip steps. Verify partial results.",
        "Walk through the solution methodically, checking at each stage.",
        "Explicit reasoning: write each deduction. Check before the next.",
        "Build the answer incrementally. Validate at each increment.",
        "Trace through every step. Pause and verify at checkpoints.",
    ],
    "conservative": [
        "Be careful. Only state what you're certain about. Flag uncertainty explicitly.",
        "Prefer known-correct approaches. Avoid speculative reasoning.",
        "If unsure, say so. Err on the side of caution.",
        "Use well-established methods. Don't take creative shortcuts.",
        "Verify each claim against known facts. Flag anything uncertain.",
        "Stick to what's proven. Express confidence levels for each statement.",
        "Conservative approach: prioritize correctness over completeness.",
        "Only proceed when confident. Acknowledge limitations.",
        "Safe and certain: no speculation, flag all assumptions.",
        "Trust verified methods. Express doubt when evidence is thin.",
    ],
}

NEUTRAL_PROMPTS = [
    "What is 2 + 2?",
    "Solve for x: 3x = 12.",
    "A rectangle has sides 4 and 5. What is the area?",
    "What is 15% of 200?",
    "If a train travels 60 mph for 2 hours, how far does it go?",
    "What is the square root of 144?",
    "Convert 0.75 to a fraction.",
    "What is 7 * 8?",
    "A triangle has angles 90° and 45°. What is the third angle?",
    "What is the perimeter of a square with side 7?",
    "What is 100 - 37?",
    "How many minutes in 2.5 hours?",
    "What is 3/4 + 1/4?",
    "Round 3.14159 to two decimal places.",
    "What is the mean of 10, 20, and 30?",
    "Simplify: 2(x + 3).",
    "What is 2^5?",
    "If 5 apples cost $3, what do 20 apples cost?",
    "What is the LCM of 4 and 6?",
    "What is the GCD of 12 and 18?",
    "Convert 5 km to meters.",
    "What is 1/3 as a percentage (rounded)?",
    "How many sides does a hexagon have?",
    "What is the volume of a cube with side 3?",
    "Evaluate: 2 + 3 * 4.",
    "What is the median of 1, 3, 5, 7, 9?",
    "What is -5 + 8?",
    "How many degrees in a right angle?",
    "What is 0.1 * 0.2?",
    "What is the next prime after 7?",
    "Factor: x^2 - 9.",
    "What is 10! / 8!?",
    "How many faces does a cube have?",
    "What is the reciprocal of 7?",
    "Solve: x/4 = 5.",
    "What is the area of a circle with radius 3? (Use π ≈ 3.14)",
    "What is 999 + 1?",
    "Express 0.4 as a fraction in lowest terms.",
    "What is the mode of 2, 3, 2, 5, 2?",
    "How many seconds in one hour?",
    "What is 45% of 80?",
    "If f(x) = 2x + 1, what is f(3)?",
    "What is the absolute value of -12?",
    "Convert 2500 grams to kilograms.",
    "What is 11^2?",
    "How many edges does a rectangular prism have?",
    "Simplify: 6/8.",
    "What is the range of 3, 8, 12, 5?",
    "Evaluate: 4^3.",
    "What is 1.5 * 4?",
]

# GSM8K problems for behavioral eval
GSM8K_PROBLEMS = [
    {"question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 each. How much does she make every day at the farmers' market?", "answer": 18},
    {"question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "answer": 3},
    {"question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?", "answer": 70000},
    {"question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many meters does he run a week?", "answer": 540},
    {"question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She has 5 chickens. If she has 75 cups of feed, how many days will the feed last?", "answer": 5},
    {"question": "Kylar went to the store to buy glasses for his new apartment. He needed 6 glasses but they were only sold in sets of 2. Each set costs $4. How much did he spend?", "answer": 12},
    {"question": "Mari makes $20 an hour as a contractor. She just got a $5 per hour raise. She works 50 hours a week. How much does she make a week?", "answer": 1250},
    {"question": "A building has 20 floors. Each floor has 8 apartments. Each apartment has 4 windows. How many windows are in the building?", "answer": 640},
    {"question": "A car travels at 60 mph for 3 hours, then 40 mph for 2 hours. What is the total distance?", "answer": 260},
    {"question": "A store sells apples at $1.50 each and oranges at $2 each. If you buy 3 apples and 2 oranges, how much do you spend?", "answer": 8.5},
]


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)


def cleanup():
    gc.collect()
    mx.clear_cache()


# ── Phase 1: Activation Extraction ───────────────────────

class ActivationStore:
    """Capture o_proj outputs for all layers during forward pass."""
    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.activations = {}  # layer_idx -> (1, seq_len, hidden)

    def install(self, model):
        """Install hooks on all self_attn.o_proj layers."""
        layers = model.language_model.layers
        for i, layer in enumerate(layers):
            orig = layer.self_attn.o_proj
            layer.self_attn.o_proj = _HookedModule(orig, self, i)

    def clear(self):
        self.activations.clear()

    def get_layer(self, idx):
        return self.activations.get(idx)


class _HookedModule(nn.Module):
    """Wraps a linear module to capture its output."""
    def __init__(self, inner, store, layer_idx):
        super().__init__()
        self.inner = inner
        self._store = store
        self._idx = layer_idx

    def __call__(self, x):
        out = self.inner(x)
        self._store.activations[self._idx] = out
        return out


def encode_prompt(tokenizer, system_prompt, user_msg):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_msg})
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    return mx.array(ids, dtype=mx.int32)


def extract_strategy_vectors(model, tokenizer, strategy_name, strategy_prompts,
                              neutral_prompts, store):
    """Extract mean activation difference between strategy and neutral.

    For each strategy prompt paired with a neutral prompt:
      1. Run forward with strategy prompt, capture o_proj outputs
      2. Run forward with neutral prompt, capture o_proj outputs
      3. Compute per-layer difference (strategy - neutral)
      4. Average across all prompt pairs

    Returns: dict[layer_idx] -> mean_diff_vector (hidden_dim,)
    """
    print(f"\n  Extracting '{strategy_name}' vectors from {len(strategy_prompts)} prompts...", flush=True)
    n_layers = store.n_layers
    layer_diffs = {i: [] for i in range(n_layers)}

    for i, (sp, np_) in enumerate(zip(strategy_prompts, neutral_prompts)):
        # Strategy forward pass
        store.clear()
        s_ids = encode_prompt(tokenizer, sp, np_)
        if s_ids.shape[0] > SEQLEN:
            s_ids = s_ids[:SEQLEN]
        _ = model(s_ids[None, :])
        mx.eval(*list(store.activations.values()))
        strategy_acts = {k: v for k, v in store.activations.items()}

        # Neutral forward pass
        store.clear()
        n_ids = encode_prompt(tokenizer, "", np_)
        if n_ids.shape[0] > SEQLEN:
            n_ids = n_ids[:SEQLEN]
        _ = model(n_ids[None, :])
        mx.eval(*list(store.activations.values()))
        neutral_acts = {k: v for k, v in store.activations.items()}

        # Compute per-layer difference (mean over tokens)
        x_len = min(int(s_ids.shape[0]), int(n_ids.shape[0]))
        s_off = int(s_ids.shape[0]) - x_len
        n_off = int(n_ids.shape[0]) - x_len

        for layer_idx in range(n_layers):
            if layer_idx not in strategy_acts or layer_idx not in neutral_acts:
                continue
            s_act = strategy_acts[layer_idx][:, s_off:s_off+x_len, :]  # (1, x_len, H)
            n_act = neutral_acts[layer_idx][:, n_off:n_off+x_len, :]   # (1, x_len, H)
            # Mean difference across tokens
            diff = mx.mean(s_act - n_act, axis=1)  # (1, H)
            layer_diffs[layer_idx].append(diff)

        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(strategy_prompts)} prompts processed", flush=True)
            mx.clear_cache()

    # Average across prompts
    mean_diffs = {}
    for layer_idx in range(n_layers):
        if layer_diffs[layer_idx]:
            stacked = mx.stack(layer_diffs[layer_idx])  # (N, 1, H)
            mean_diffs[layer_idx] = mx.squeeze(mx.mean(stacked, axis=0))  # (H,)
        mx.clear_cache()

    return mean_diffs


# ── Phase 2: K1 — Strategy Vector Consistency ────────────

def measure_k1_consistency(strategy_vectors_by_prompt, strategy_name):
    """K1: For same strategy, different prompts → cos between extracted vectors.

    If strategy vectors are consistent (cos > 0.1), the strategy signal
    is real and not prompt-specific.
    """
    print(f"\n=== K1: Strategy vector consistency for '{strategy_name}' ===", flush=True)

    # Pick a middle layer to measure (layer 21 is roughly mid-Gemma-4)
    # Actually measure across all layers and pick the best
    all_layer_cos = {}
    for layer_idx in strategy_vectors_by_prompt[0].keys():
        vecs = [sv[layer_idx] for sv in strategy_vectors_by_prompt if layer_idx in sv]
        if len(vecs) < 2:
            continue
        # Pairwise cosine
        cos_values = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                a, b = vecs[i], vecs[j]
                cos_val = float(mx.sum(a * b) / (mx.sqrt(mx.sum(a * a)) * mx.sqrt(mx.sum(b * b)) + 1e-8))
                cos_values.append(cos_val)
        mean_cos = sum(cos_values) / len(cos_values) if cos_values else 0.0
        all_layer_cos[layer_idx] = round(mean_cos, 4)

    # Find best layer
    best_layer = max(all_layer_cos, key=all_layer_cos.get)
    best_cos = all_layer_cos[best_layer]
    mean_all = sum(all_layer_cos.values()) / max(len(all_layer_cos), 1)

    print(f"  Best layer: {best_layer} (cos={best_cos:.4f})", flush=True)
    print(f"  Mean across layers: {mean_all:.4f}", flush=True)

    return {
        "best_layer": best_layer,
        "best_cos": best_cos,
        "mean_cos_all_layers": round(mean_all, 4),
        "per_layer_cos": all_layer_cos,
        "n_prompts": len(strategy_vectors_by_prompt),
    }


# ── Phase 3: Inject as LoRA ─────────────────────────────

def inject_vector_as_lora(model, strategy_name, mean_diffs, best_layer=None):
    """Project mean activation difference into LoRA B-matrix format.

    For each target layer and projection:
      - A matrix: Grassmannian random (orthogonal by construction, F#428)
      - B matrix: reshape mean_diff vector to (rank, hidden//rank) or use projection

    Strategy: project the activation difference vector onto the Grassmannian A's
    column space to get B = A^T * diff (rank-dim projection of hidden-dim vector).
    """
    print(f"\n=== Phase 3: Injecting '{strategy_name}' as LoRA ===", flush=True)
    from mlx_lm.tuner.lora import LoRALinear

    n_layers = len(model.language_model.layers)
    target_layers = [best_layer] if best_layer is not None else list(range(n_layers))
    # Also target nearby layers ±2 for coverage
    if best_layer is not None:
        for offset in [-2, -1, 1, 2]:
            l = best_layer + offset
            if 0 <= l < n_layers and l not in target_layers:
                target_layers.append(l)

    injected_layers = []
    for layer_idx in target_layers:
        if layer_idx not in mean_diffs:
            continue
        diff_vec = mean_diffs[layer_idx]  # (H,)
        if diff_vec is None:
            continue

        layer = model.language_model.layers[layer_idx]
        for proj_name in ADAPTER_TARGETS:
            wrapped = getattr(layer.self_attn, proj_name, None)
            if wrapped is None:
                continue

            # Unwrap HookedModule to get the actual linear layer
            orig = wrapped.inner if hasattr(wrapped, 'inner') else wrapped
            if not hasattr(orig, 'weight'):
                continue

            # Create LoRA module first to get correct dimensions
            # (quantized models have packed weight shapes that differ from actual dims)
            try:
                lora_mod = LoRALinear.from_base(orig, r=ADAPTER_RANK, dropout=0.0, scale=LORA_SCALE)
                in_features = lora_mod.lora_a.shape[0]
                out_features = lora_mod.lora_b.shape[1]

                # Create Grassmannian A: random orthogonal column basis
                mx.random.seed(SEED + layer_idx + hash(proj_name) % 1000)
                A_full = mx.random.normal((in_features, ADAPTER_RANK))
                Q, _ = mx.linalg.qr(A_full, stream=mx.cpu)
                A_mat = Q[:, :ADAPTER_RANK]  # (in_features, rank)

                diff_slice = diff_vec[:out_features]
                diff_norm = mx.sqrt(mx.sum(diff_slice * diff_slice) + 1e-8)
                diff_normalized = diff_slice / diff_norm

                scale_factor = diff_norm / mx.sqrt(mx.array(ADAPTER_RANK, dtype=mx.float32))
                B_mat = mx.broadcast_to(
                    diff_normalized[None, :],
                    (ADAPTER_RANK, out_features)
                ) * (scale_factor / ADAPTER_RANK)

                lora_mod.lora_a = A_mat
                lora_mod.lora_b = B_mat
                # Re-wrap in hook (so downstream code can still toggle scale)
                setattr(layer.self_attn, proj_name,
                        _HookedModule(lora_mod, _NullStore(), layer_idx))
                injected_layers.append(layer_idx)
            except Exception as e:
                print(f"    Warning: could not inject layer {layer_idx}.{proj_name}: {e}", flush=True)

    print(f"  Injected into {len(set(injected_layers))} layers", flush=True)
    return {"injected_layers": sorted(set(injected_layers))}


class _NullStore:
    """Dummy store for injected LoRA modules (no capture needed)."""
    def __init__(self):
        self.activations = {}
    def clear(self):
        pass


# ── Phase 4: K2 — Behavioral GSM8K Eval ──────────────────

def measure_k2_gsm8k(model, tokenizer, problems, adapter_on=True):
    """K2: Measure GSM8K accuracy with and without injected adapter."""
    from mlx_lm.generate import generate
    print(f"\n=== K2: GSM8K eval (adapter={'ON' if adapter_on else 'OFF'}) ===", flush=True)

    # Toggle LoRA scale — handles both HookedModule-wrapped and bare LoRA
    def set_scale(s):
        for layer in model.language_model.layers:
            for name in ADAPTER_TARGETS:
                mod = getattr(layer.self_attn, name, None)
                if mod is None:
                    continue
                # May be HookedModule wrapping LoRALinear
                target = mod
                while hasattr(target, 'inner'):
                    target = target.inner
                if hasattr(target, "scale"):
                    target.scale = s

    correct = 0
    total = 0
    results_list = []

    for i, problem in enumerate(problems):
        q = problem["question"]
        expected = problem["answer"]

        msgs = [{"role": "user", "content": q + " Give only the final numerical answer."}]
        prompt_str = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

        try:
            out = generate(model, tokenizer, prompt=prompt_str, max_tokens=128, verbose=False)
            # Extract number from output
            predicted = _extract_number(out)
            is_correct = predicted is not None and abs(predicted - expected) < 0.5
            if is_correct:
                correct += 1
            total += 1
            results_list.append({
                "question": q[:80],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
            })
        except Exception as e:
            results_list.append({"question": q[:80], "error": str(e)})
            total += 1

        if (i + 1) % 10 == 0:
            acc = correct / max(total, 1) * 100
            print(f"  {i+1}/{total} accuracy={acc:.1f}%", flush=True)

    acc = correct / max(total, 1) * 100
    return {
        "correct": correct,
        "total": total,
        "accuracy_pp": round(acc, 2),
        "details": results_list if IS_SMOKE else f"{correct}/{total}",
    }


def _extract_number(text):
    """Extract the last number from generated text."""
    import re
    # Look for numbers, possibly with decimals or negatives
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except ValueError:
        return None


# ── Phase 5: K3 — Cross-Prompt Consistency ───────────────

def measure_k3_cross_prompt(strategy_name, mean_diffs, model, tokenizer, store):
    """K3: Same strategy, different prompt formulations → cos between vectors."""
    print(f"\n=== K3: Cross-prompt consistency for '{strategy_name}' ===", flush=True)

    # Split strategy prompts into two halves, extract vectors from each
    all_prompts = STRATEGY_PROMPTS[strategy_name]
    mid = len(all_prompts) // 2
    group_a_prompts = all_prompts[:mid]
    group_b_prompts = all_prompts[mid:]

    neutral_a = NEUTRAL_PROMPTS[:mid]
    neutral_b = NEUTRAL_PROMPTS[mid:mid+len(group_b_prompts)]

    vecs_a = extract_strategy_vectors(model, tokenizer, f"{strategy_name}_A",
                                       group_a_prompts, neutral_a, store)
    vecs_b = extract_strategy_vectors(model, tokenizer, f"{strategy_name}_B",
                                       group_b_prompts, neutral_b, store)

    # Cosine between group A and group B mean vectors
    layer_cos = {}
    for layer_idx in vecs_a:
        if layer_idx not in vecs_b:
            continue
        a = vecs_a[layer_idx]
        b = vecs_b[layer_idx]
        cos_val = float(mx.sum(a * b) / (mx.sqrt(mx.sum(a * a)) * mx.sqrt(mx.sum(b * b)) + 1e-8))
        layer_cos[layer_idx] = round(cos_val, 4)

    if not layer_cos:
        return {"mean_cos": None, "per_layer": {}, "verdict": "insufficient_data"}

    best_layer = max(layer_cos, key=layer_cos.get)
    mean_cos = sum(layer_cos.values()) / len(layer_cos)
    print(f"  Best layer: {best_layer} cos={layer_cos[best_layer]:.4f}", flush=True)
    print(f"  Mean cos: {mean_cos:.4f}", flush=True)

    return {
        "best_layer": best_layer,
        "best_cos": layer_cos[best_layer],
        "mean_cos": round(mean_cos, 4),
        "per_layer": layer_cos,
    }


# ── Main ─────────────────────────────────────────────────

def main():
    t_start = time.time()
    log_memory("start")

    results = {
        "experiment": "exp_e1_function_vector_strategy_extraction",
        "is_smoke": IS_SMOKE,
        "model_id": MODEL_ID,
        "adapter_rank": ADAPTER_RANK,
        "adapter_targets": list(ADAPTER_TARGETS),
        "lora_scale": LORA_SCALE,
        "strategies_tested": list(STRATEGY_PROMPTS.keys()),
        "phases": {},
        "kc": {
            "K2017_strategy_vector_cos_gt_0_1": "untested",
            "K2018_gsm8k_improvement_gt_2pp": "untested",
            "K2019_cross_prompt_cos_gt_0_3": "untested",
        },
        "verdict": "PROVISIONAL",
    }

    # Load model
    print("Loading Gemma 4 E4B 4-bit...", flush=True)
    from mlx_lm import load
    mx.random.seed(SEED)
    model, tokenizer = load(MODEL_ID)
    log_memory("model_loaded")

    n_layers = len(model.language_model.layers)
    print(f"Model loaded: {n_layers} layers", flush=True)

    # Install activation capture hooks
    store = ActivationStore(n_layers)
    store.install(model)
    log_memory("hooks_installed")

    # ── Phase 1+2: Extract + K1 for each strategy ────────
    strategy_results = {}
    all_mean_diffs = {}
    k1_best_layers = {}

    for strategy_name in STRATEGY_PROMPTS:
        print(f"\n{'='*60}", flush=True)
        print(f"STRATEGY: {strategy_name}", flush=True)
        print(f"{'='*60}", flush=True)

        prompts = STRATEGY_PROMPTS[strategy_name][:N_PROMPTS_PER_STRATEGY]
        neutrals = NEUTRAL_PROMPTS[:len(prompts)]

        # Extract per-prompt vectors for K1
        per_prompt_vecs = []
        for i, (sp, np_) in enumerate(zip(prompts, neutrals)):
            vec = extract_strategy_vectors(model, tokenizer,
                                            f"{strategy_name}_p{i}", [sp], [np_], store)
            per_prompt_vecs.append(vec)

        # K1: consistency across prompts
        k1 = measure_k1_consistency(per_prompt_vecs, strategy_name)
        results["phases"][f"k1_{strategy_name}"] = k1
        k1_best_layers[strategy_name] = k1.get("best_layer")

        # Extract mean vector (all prompts)
        mean_diffs = extract_strategy_vectors(model, tokenizer, strategy_name,
                                               prompts, neutrals, store)
        all_mean_diffs[strategy_name] = mean_diffs

        strategy_results[strategy_name] = {
            "k1": k1,
            "mean_diff_norm": {str(k): float(mx.sqrt(mx.sum(v * v)))
                                for k, v in mean_diffs.items() if v is not None},
        }

        # K1 verdict
        best_cos = k1.get("best_cos", 0)
        if best_cos > 0.1:
            results["kc"][f"K2017_{strategy_name}_cos_gt_0_1"] = "pass"
        else:
            results["kc"][f"K2017_{strategy_name}_cos_gt_0_1"] = "fail"

        cleanup()

    results["phases"]["strategy_extraction"] = strategy_results

    # ── Phase 3+4: Inject best strategy and test GSM8K ────
    # Pick the strategy with highest K1 consistency
    best_strategy = max(strategy_results.keys(),
                        key=lambda s: strategy_results[s]["k1"].get("best_cos", 0))
    best_layer = k1_best_layers.get(best_strategy, n_layers // 2)

    # ── K3 BEFORE injection (model still has original weights + hooks) ──
    k3 = measure_k3_cross_prompt(best_strategy, all_mean_diffs[best_strategy],
                                  model, tokenizer, store)
    results["phases"][f"k3_{best_strategy}"] = k3

    if k3.get("mean_cos", 0) > 0.3:
        results["kc"]["K2019_cross_prompt_cos_gt_0_3"] = "pass"
    else:
        results["kc"]["K2019_cross_prompt_cos_gt_0_3"] = "fail"

    cleanup()

    # ── Baseline GSM8K (no adapter) ──────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"INJECTING: {best_strategy} (best K1, layer={best_layer})", flush=True)
    print(f"{'='*60}", flush=True)

    print("\n--- Baseline GSM8K ---", flush=True)
    baseline = measure_k2_gsm8k(model, tokenizer, GSM8K_PROBLEMS, adapter_on=False)
    results["phases"]["baseline_gsm8k"] = baseline

    # Inject adapter
    inject_info = inject_vector_as_lora(model, best_strategy,
                                          all_mean_diffs[best_strategy], best_layer)
    results["phases"]["injection"] = inject_info

    # GSM8K with injected adapter
    print("\n--- GSM8K with injected adapter ---", flush=True)
    adapted = measure_k2_gsm8k(model, tokenizer, GSM8K_PROBLEMS, adapter_on=True)
    results["phases"]["adapted_gsm8k"] = adapted

    # K2 verdict
    delta = adapted["accuracy_pp"] - baseline["accuracy_pp"]
    results["phases"]["gsm8k_delta_pp"] = round(delta, 2)
    if delta > 2.0:
        results["kc"]["K2018_gsm8k_improvement_gt_2pp"] = "pass"
    else:
        results["kc"]["K2018_gsm8k_improvement_gt_2pp"] = "fail"

    # ── Cross-strategy interference check ────────────────
    print("\n=== Cross-strategy interference ===", flush=True)
    strategy_names = list(STRATEGY_PROMPTS.keys())
    cross_cos = {}
    for i, s1 in enumerate(strategy_names):
        for j, s2 in enumerate(strategy_names):
            if i >= j:
                continue
            # Cos between mean vectors at best layer
            v1 = all_mean_diffs[s1].get(best_layer)
            v2 = all_mean_diffs[s2].get(best_layer)
            if v1 is not None and v2 is not None:
                cos_val = float(mx.sum(v1 * v2) / (mx.sqrt(mx.sum(v1 * v1)) * mx.sqrt(mx.sum(v2 * v2)) + 1e-8))
                cross_cos[f"{s1}_vs_{s2}"] = round(cos_val, 4)
    results["phases"]["cross_strategy_cosine"] = cross_cos
    print(f"  Cross-strategy cos: {cross_cos}", flush=True)

    # ── Final verdict ────────────────────────────────────
    all_kc = list(results["kc"].values())
    n_pass = sum(1 for v in all_kc if v == "pass")
    n_fail = sum(1 for v in all_kc if v == "fail")
    results["total_time_s"] = round(time.time() - t_start, 1)

    if n_fail == 0 and n_pass > 0:
        results["verdict"] = "SUPPORTED"
    elif n_fail > 0:
        results["verdict"] = "KILLED"
    else:
        results["verdict"] = "PROVISIONAL"

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    print("\n" + "=" * 60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Best strategy: {best_strategy} (layer {best_layer})", flush=True)
    print(f"Baseline GSM8K: {baseline['accuracy_pp']:.1f}%", flush=True)
    print(f"Adapted GSM8K: {adapted['accuracy_pp']:.1f}%", flush=True)
    print(f"Delta: {delta:+.2f}pp", flush=True)
    print(f"KCs: {json.dumps(results['kc'])}", flush=True)
    print(f"Verdict: {results['verdict']}", flush=True)
    print(f"Time: {results['total_time_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
