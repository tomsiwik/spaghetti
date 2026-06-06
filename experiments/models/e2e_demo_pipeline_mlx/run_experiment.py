#!/usr/bin/env python3
"""E2E Demo Pipeline: query -> entropy gate -> route -> compose -> generate on M5 Pro.

The first end-to-end integration of all proven BitNet-SOLE components:
  1. BitNet-2B-4T base model (1.7GB, ternary)
  2. 5 real-data trained adapters (medical, code, math, legal, finance)
  3. Entropy gating (skip routing when base is confident)
  4. Oracle routing (select domain expert based on query)
  5. Pre-merge composition (0% per-token overhead)
  6. Text generation with quality measurement

Kill criteria:
  K1 (#245): E2E latency > 2x base generation
  K2 (#246): Quality worse than base alone on any domain

Success criteria:
  S1 (#20): E2E pipeline serves interactively (<100ms/token) with quality >= base

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
from mlx.utils import tree_flatten, tree_unflatten

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
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256

DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = len(DOMAINS)

# Generation settings
NUM_PROMPTS_PER_DOMAIN = 10
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.9

# Entropy gating (from proven experiment: entropy_gated_experts)
OTSU_THRESHOLD = 2.10  # nats, proven Otsu threshold

# Pipeline configs to test
PIPELINE_CONFIGS = {
    "base": "No adapters, base model only",
    "e2e_oracle_top1": "Full pipeline: entropy gate + oracle top-1 routing + pre-merge",
    "e2e_oracle_top2": "Full pipeline: entropy gate + oracle top-2 routing + pre-merge",
}


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
# BitNet unpacking and model utilities
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
# Pre-merge composition (proven 0% overhead)
# ============================================================================

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


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

    This is the core of the 0% overhead approach: modify W_base in place so that
    the composed model is just a standard nn.Linear with no per-token LoRA overhead.

    W_new = W_base + sum_d w_d * scale * B_d @ A_d

    Args:
        model: Model with nn.Linear layers (post BitLinear replacement)
        skeleton: dict of Grassmannian A matrices
        adapters: dict of domain -> adapter params
        domain_weights: dict of domain -> weight (0.0 to skip)
    """
    n_layers = len(model.model.layers)
    merge_count = 0

    for li in range(n_layers):
        for key in TARGET_KEYS:
            # Navigate to the linear layer
            parts = key.split(".")
            module = model.model.layers[li]
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue

            # Accumulate LoRA deltas from selected adapters
            delta = None
            for domain, w in domain_weights.items():
                if w < 1e-6:
                    continue

                # Get A matrix from skeleton
                di = DOMAINS.index(domain)
                skey = f"layer_{li}_{key}_domain_{di}"
                if skey not in skeleton:
                    continue
                a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)

                # Get B matrix from adapter
                b_key = f"model.layers.{li}.{key}.lora_b"
                if b_key not in adapters[domain]:
                    continue
                b_mx = adapters[domain][b_key]

                # LoRA delta: B @ A^T (note: A is (in, r), B is (r, out))
                # In mlx_lm LoRA convention: output = x @ A @ B * scale
                # So delta_W = (A @ B)^T for weight matrix W (out, in)
                # But we need to add to W which is (out, in):
                # y = W @ x + scale * B @ A^T @ x = (W + scale * B @ A^T) @ x
                # Wait -- need to match the convention exactly.
                # In the training code: lora_out = (x @ self.lora_a) @ self.lora_b * self.scale
                # So: lora_out = x @ (A @ B) * scale
                # Base: base_out = x @ W^T (since nn.Linear stores W as (out, in))
                # Combined: y = x @ (W^T + A @ B * scale) = x @ W_new^T
                # So: W_new = W + (A @ B * scale)^T = W + scale * B^T @ A^T
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
# Entropy computation
# ============================================================================

def compute_query_entropy(model, tokenizer, text):
    """Compute mean entropy of base model output on query text.

    Returns (mean_entropy, fraction_above_threshold).
    """
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return 0.0, 0.0

    tokens = tokens[:MAX_SEQ_LENGTH]
    x = mx.array(tokens)[None, :]
    logits = model(x)
    mx.eval(logits)

    # Compute per-token entropy
    probs = mx.softmax(logits[0], axis=-1)
    log_p = mx.log(mx.clip(probs, 1e-10, 1.0))
    ent = -mx.sum(probs * log_p, axis=-1)  # (seq_len,)
    mx.eval(ent)

    ent_list = ent.tolist()
    mean_ent = float(np.mean(ent_list))
    frac_above = float(np.mean([1.0 if e >= OTSU_THRESHOLD else 0.0 for e in ent_list]))

    del logits, probs, log_p, ent, x
    return mean_ent, frac_above


# ============================================================================
# Oracle routing (proven 100% accuracy on 5 genuine domains)
# ============================================================================

def oracle_route(domain, k=1):
    """Oracle routing: returns domain weights for top-k selection.

    For top-1: full weight to correct domain.
    For top-2: 0.8 to correct domain, 0.2 to most related domain.
    """
    # Domain affinity (based on empirical cross-domain PPL from prior experiments)
    affinity = {
        "medical": ["legal", "finance", "math", "code"],
        "code": ["math", "finance", "legal", "medical"],
        "math": ["code", "finance", "medical", "legal"],
        "legal": ["finance", "medical", "math", "code"],
        "finance": ["legal", "math", "medical", "code"],
    }

    weights = {d: 0.0 for d in DOMAINS}
    if k == 1:
        weights[domain] = 1.0
    elif k == 2:
        weights[domain] = 0.8
        secondary = affinity[domain][0]
        weights[secondary] = 0.2
    return weights


# ============================================================================
# Text generation
# ============================================================================

def generate_text(model, tokenizer, prompt, max_tokens=128, temperature=0.7, top_p=0.9):
    """Generate text using mlx_lm.generate."""
    try:
        sampler = make_sampler(temp=temperature, top_p=top_p)
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


def generate_text_timed(model, tokenizer, prompt, max_tokens=128, temperature=0.7, top_p=0.9):
    """Generate text and return (text, elapsed_seconds, tokens_generated)."""
    t0 = time.time()
    text = generate_text(model, tokenizer, prompt, max_tokens, temperature, top_p)
    elapsed = time.time() - t0

    # Count generated tokens (approximate: re-tokenize the output)
    gen_tokens = len(tokenizer.encode(text)) if text else 0
    return text, elapsed, gen_tokens


# ============================================================================
# Quality metrics (domain-appropriate, NOT keyword density for prose)
# ============================================================================

def code_syntax_valid(text):
    """Check if generated text contains valid Python code."""
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
        if stripped.startswith(('def ', 'class ', 'import ', 'from ', 'for ', 'while ',
                                'if ', 'try:', 'except', 'with ', 'return ', 'print(', '#')):
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
    """Extract numeric answer from generated text."""
    m = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*\$?([\d,]+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(',', ''))
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
    """Extract ground truth answer from training data."""
    matches = re.findall(r'<<[^>]*?=(\d+(?:\.\d+)?)>>', response_text)
    if matches:
        return float(matches[-1])
    m = re.search(r'####\s*([\d,]+(?:\.\d+)?)', response_text)
    if m:
        return float(m.group(1).replace(',', ''))
    return None


def math_answer_correct(gen_answer, gt_answer):
    """Check answer correctness within 1% tolerance."""
    if gen_answer is None or gt_answer is None:
        return False
    if gt_answer == 0:
        return abs(gen_answer) < 0.01
    return abs(gen_answer - gt_answer) / abs(gt_answer) < 0.01


def compute_ppl_on_text(model, tokenizer, text):
    """Compute perplexity of model on a single text."""
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return float("inf")
    tokens = tokens[:MAX_SEQ_LENGTH + 1]
    x = mx.array(tokens[:-1])[None, :]
    y = mx.array(tokens[1:])[None, :]
    logits = model(x)
    loss = nn.losses.cross_entropy(logits, y, reduction="sum")
    mx.eval(loss)
    n_tokens = y.size
    ppl = math.exp(min(loss.item() / n_tokens, 100))
    del logits, loss, x, y
    return ppl


def is_incoherent(text):
    """Detect obviously incoherent text."""
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 3:
        return True
    counter = Counter(words)
    most_common_freq = counter.most_common(1)[0][1] / len(words)
    if most_common_freq > 0.4:
        return True
    if len(set(words)) < 5 and len(words) > 20:
        return True
    return False


def word_count(text):
    return len(re.findall(r'\b\w+\b', text))


# ============================================================================
# Prompt extraction
# ============================================================================

def extract_prompts_with_answers(domain, n_prompts=10):
    """Extract instruction prompts from validation data."""
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
# Phase 1: Base generation (latency + quality baseline)
# ============================================================================

def phase_base_generation(prompts_by_domain):
    """Generate with base model only. Returns results dict."""
    log("\n" + "=" * 70)
    log("[Phase 1] BASE MODEL generation (no adapters)")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("post-load-base")

    mx.random.seed(42)
    np.random.seed(42)

    results = {}
    total_gen_time = 0.0
    total_gen_tokens = 0

    for domain in DOMAINS:
        domain_results = []
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            formatted = format_prompt(prompt_data["instruction"])
            text, gen_time, gen_tokens = generate_text_timed(
                model, tokenizer, formatted,
                max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P
            )
            total_gen_time += gen_time
            total_gen_tokens += gen_tokens

            # Domain-appropriate quality metrics
            result = {
                "prompt": prompt_data["instruction"][:100],
                "generated": text[:200],
                "gen_time_s": round(gen_time, 3),
                "gen_tokens": gen_tokens,
                "word_count": word_count(text),
                "incoherent": is_incoherent(text),
            }

            if domain == "code":
                result["syntax_valid"] = code_syntax_valid(text)
            elif domain == "math":
                gen_ans = extract_math_answer(text)
                gt_ans = extract_ground_truth_answer(prompt_data["response"])
                result["answer_correct"] = math_answer_correct(gen_ans, gt_ans)

            domain_results.append(result)

            if (i + 1) % 5 == 0:
                log(f"  {domain}: {i+1}/{len(prompts_by_domain[domain])} done")

        results[domain] = domain_results

    # Also compute PPL on validation data
    ppl_results = {}
    for domain in DOMAINS:
        val_path = DATA_DIR / domain / "valid.jsonl"
        ppls = []
        with open(val_path) as f:
            for j, line in enumerate(f):
                if j >= 25:
                    break
                text = json.loads(line)["text"]
                ppl = compute_ppl_on_text(model, tokenizer, text)
                ppls.append(ppl)
                del text
        ppl_results[domain] = round(float(np.mean(ppls)), 4)
        log(f"  {domain} PPL: {ppl_results[domain]}")

    elapsed = time.time() - t0
    tokens_per_sec = total_gen_tokens / total_gen_time if total_gen_time > 0 else 0
    ms_per_token = (total_gen_time / total_gen_tokens * 1000) if total_gen_tokens > 0 else 0

    summary = {
        "generation_results": results,
        "ppl": ppl_results,
        "total_gen_time_s": round(total_gen_time, 2),
        "total_gen_tokens": total_gen_tokens,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "ms_per_token": round(ms_per_token, 2),
        "phase_time_s": round(elapsed, 1),
    }

    log(f"\n  Base: {tokens_per_sec:.1f} tok/s, {ms_per_token:.1f} ms/tok")
    log(f"  Phase 1 done in {elapsed:.1f}s")
    log_memory("post-base-gen")
    cleanup(model, tokenizer)
    return summary


# ============================================================================
# Phase 2: E2E pipeline generation
# ============================================================================

def save_base_weights(model):
    """Save a copy of the base (unmerged) weights for restoration after pre-merge."""
    base_weights = {}
    for li, layer in enumerate(model.model.layers):
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None and isinstance(module, nn.Linear):
                # Deep copy the weight
                base_weights[(li, key)] = mx.array(module.weight)
    mx.eval(base_weights)
    return base_weights


def restore_base_weights(model, base_weights):
    """Restore base weights after pre-merge (undo composition)."""
    for (li, key), w in base_weights.items():
        parts = key.split(".")
        module = model.model.layers[li]
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                break
        if module is not None and isinstance(module, nn.Linear):
            module.weight = w
    mx.eval(model.parameters())


def phase_e2e_generation(prompts_by_domain, top_k=1):
    """Full E2E pipeline: entropy gate -> route -> pre-merge -> generate.

    Loads model ONCE, saves base weights, and for each query:
    1. Compute entropy on query (entropy gate)
    2. If entropy low: use base directly
    3. Otherwise: oracle route, pre-merge adapters, generate, restore base weights

    Args:
        top_k: Number of experts to route to (1 or 2)
    """
    log(f"\n{'=' * 70}")
    log(f"[Phase 2] E2E PIPELINE: entropy gate + oracle top-{top_k} + pre-merge")
    log("=" * 70)

    t0_phase = time.time()
    skeleton = load_skeleton()
    all_adapters = load_all_adapters()

    # Load model ONCE for the entire phase
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("post-load-e2e")

    # Save base weights for restoration after each pre-merge
    base_weights = save_base_weights(model)
    log(f"  Saved {len(base_weights)} base weight matrices for restoration")

    results = {}
    ppl_results = {}
    total_gen_time = 0.0
    total_gen_tokens = 0
    total_routing_time = 0.0
    total_merge_time = 0.0
    entropy_skips = 0
    total_queries = 0

    for domain in DOMAINS:
        log(f"\n  --- Domain: {domain} ---")
        domain_results = []

        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            total_queries += 1
            formatted = format_prompt(prompt_data["instruction"])

            # Step 1: Entropy gate -- compute on the query itself
            t_route = time.time()
            mean_ent, frac_above = compute_query_entropy(model, tokenizer, prompt_data["instruction"])
            entropy_gate_open = mean_ent >= OTSU_THRESHOLD

            merged = False
            if not entropy_gate_open:
                # Base is confident -- skip routing and composition
                entropy_skips += 1
                routing_time = time.time() - t_route
                total_routing_time += routing_time
                merge_time = 0.0
            else:
                # Step 2: Oracle routing
                domain_weights = oracle_route(domain, k=top_k)
                routing_time = time.time() - t_route
                total_routing_time += routing_time

                # Step 3: Pre-merge composition
                t_merge = time.time()
                model = premerge_adapters_into_model(model, skeleton, all_adapters, domain_weights)
                merge_time = time.time() - t_merge
                total_merge_time += merge_time
                merged = True

            # Step 4: Generate
            mx.random.seed(42)
            text, gen_time, gen_tokens = generate_text_timed(
                model, tokenizer, formatted,
                max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P
            )
            total_gen_time += gen_time
            total_gen_tokens += gen_tokens

            # Step 5: Restore base weights if we merged
            if merged:
                t_restore = time.time()
                restore_base_weights(model, base_weights)
                restore_time = time.time() - t_restore
            else:
                restore_time = 0.0

            # Quality metrics
            result = {
                "prompt": prompt_data["instruction"][:100],
                "generated": text[:200],
                "gen_time_s": round(gen_time, 3),
                "gen_tokens": gen_tokens,
                "word_count": word_count(text),
                "incoherent": is_incoherent(text),
                "entropy_gate": "skip" if not entropy_gate_open else "open",
                "mean_entropy": round(mean_ent, 4),
                "routing_time_ms": round(routing_time * 1000, 2),
                "merge_time_ms": round(merge_time * 1000, 2),
                "restore_time_ms": round(restore_time * 1000, 2),
                "total_e2e_overhead_ms": round((routing_time + merge_time + restore_time) * 1000, 2),
            }

            if domain == "code":
                result["syntax_valid"] = code_syntax_valid(text)
            elif domain == "math":
                gen_ans = extract_math_answer(text)
                gt_ans = extract_ground_truth_answer(prompt_data["response"])
                result["answer_correct"] = math_answer_correct(gen_ans, gt_ans)

            domain_results.append(result)

            if (i + 1) % 5 == 0:
                log(f"  {domain}: {i+1}/{len(prompts_by_domain[domain])} done "
                    f"(skip={entropy_skips}/{total_queries})")

        results[domain] = domain_results

    # PPL evaluation: per-domain with composed model
    for domain in DOMAINS:
        domain_weights = oracle_route(domain, k=top_k)
        premerge_adapters_into_model(model, skeleton, all_adapters, domain_weights)

        val_path = DATA_DIR / domain / "valid.jsonl"
        ppls = []
        with open(val_path) as f:
            for j, line in enumerate(f):
                if j >= 25:
                    break
                text = json.loads(line)["text"]
                ppl = compute_ppl_on_text(model, tokenizer, text)
                ppls.append(ppl)
        ppl_results[domain] = round(float(np.mean(ppls)), 4)
        log(f"  {domain} PPL (composed): {ppl_results[domain]}")

        # Restore for next domain
        restore_base_weights(model, base_weights)

    elapsed = time.time() - t0_phase
    tokens_per_sec = total_gen_tokens / total_gen_time if total_gen_time > 0 else 0
    ms_per_token = (total_gen_time / total_gen_tokens * 1000) if total_gen_tokens > 0 else 0

    summary = {
        "generation_results": results,
        "ppl": ppl_results,
        "total_gen_time_s": round(total_gen_time, 2),
        "total_gen_tokens": total_gen_tokens,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "ms_per_token": round(ms_per_token, 2),
        "total_routing_time_s": round(total_routing_time, 2),
        "total_merge_time_s": round(total_merge_time, 2),
        "entropy_skip_rate": round(entropy_skips / total_queries, 4) if total_queries > 0 else 0,
        "entropy_skips": entropy_skips,
        "total_queries": total_queries,
        "phase_time_s": round(elapsed, 1),
    }

    log(f"\n  E2E top-{top_k}: {tokens_per_sec:.1f} tok/s, {ms_per_token:.1f} ms/tok")
    log(f"  Entropy skip rate: {entropy_skips}/{total_queries} ({summary['entropy_skip_rate']*100:.1f}%)")
    log(f"  Phase 2 (top-{top_k}) done in {elapsed:.1f}s")
    log_memory("post-e2e-gen")
    cleanup(model, tokenizer, base_weights)
    return summary


# ============================================================================
# Phase 3: Analyze and compare
# ============================================================================

def phase_analyze(base_results, e2e_top1_results, e2e_top2_results):
    """Compare base vs E2E pipeline on latency and quality."""
    log(f"\n{'=' * 70}")
    log("[Phase 3] ANALYSIS: Base vs E2E Pipeline")
    log("=" * 70)

    analysis = {
        "latency": {},
        "quality": {},
        "kill_criteria": {},
    }

    # --- Latency comparison (K1) ---
    base_ms = base_results["ms_per_token"]
    e2e1_ms = e2e_top1_results["ms_per_token"]
    e2e2_ms = e2e_top2_results["ms_per_token"]

    latency_ratio_top1 = e2e1_ms / base_ms if base_ms > 0 else float("inf")
    latency_ratio_top2 = e2e2_ms / base_ms if base_ms > 0 else float("inf")

    analysis["latency"] = {
        "base_ms_per_token": base_ms,
        "e2e_top1_ms_per_token": e2e1_ms,
        "e2e_top2_ms_per_token": e2e2_ms,
        "base_tokens_per_sec": base_results["tokens_per_sec"],
        "e2e_top1_tokens_per_sec": e2e_top1_results["tokens_per_sec"],
        "e2e_top2_tokens_per_sec": e2e_top2_results["tokens_per_sec"],
        "latency_ratio_top1": round(latency_ratio_top1, 4),
        "latency_ratio_top2": round(latency_ratio_top2, 4),
        "e2e_top1_routing_overhead_s": e2e_top1_results["total_routing_time_s"],
        "e2e_top1_merge_overhead_s": e2e_top1_results["total_merge_time_s"],
        "e2e_top2_routing_overhead_s": e2e_top2_results["total_routing_time_s"],
        "e2e_top2_merge_overhead_s": e2e_top2_results["total_merge_time_s"],
        "entropy_skip_rate_top1": e2e_top1_results["entropy_skip_rate"],
        "entropy_skip_rate_top2": e2e_top2_results["entropy_skip_rate"],
    }

    # Compute merged-only (worst-case) latency ratio: only queries where entropy_gate="open"
    merged_only_ratios = {}
    for config_name, config_results in [("top1", e2e_top1_results), ("top2", e2e_top2_results)]:
        merged_gen_time = 0.0
        merged_gen_tokens = 0
        for domain in DOMAINS:
            for r in config_results["generation_results"][domain]:
                if r["entropy_gate"] == "open":
                    merged_gen_time += r["gen_time_s"]
                    merged_gen_tokens += r["gen_tokens"]
        if merged_gen_tokens > 0:
            merged_ms_per_tok = merged_gen_time / merged_gen_tokens * 1000
            merged_only_ratios[config_name] = round(merged_ms_per_tok / base_ms, 4)
        else:
            merged_only_ratios[config_name] = None

    analysis["latency"]["merged_only_ratio_top1"] = merged_only_ratios.get("top1")
    analysis["latency"]["merged_only_ratio_top2"] = merged_only_ratios.get("top2")

    k1_pass = latency_ratio_top1 <= 2.0 and latency_ratio_top2 <= 2.0
    analysis["kill_criteria"]["K1"] = {
        "test": "E2E latency > 2x base generation",
        "result": "PASS" if k1_pass else "FAIL (KILL)",
        "evidence": (
            f"average ratio: top1={latency_ratio_top1:.3f}, top2={latency_ratio_top2:.3f}; "
            f"worst-case (merged-only): top1={merged_only_ratios.get('top1', 'N/A')}x, "
            f"top2={merged_only_ratios.get('top2', 'N/A')}x"
        ),
    }

    log(f"\n  K1 Latency:")
    log(f"    Base: {base_ms:.1f} ms/tok ({base_results['tokens_per_sec']:.1f} tok/s)")
    log(f"    E2E top-1: {e2e1_ms:.1f} ms/tok ({e2e_top1_results['tokens_per_sec']:.1f} tok/s)")
    log(f"    E2E top-2: {e2e2_ms:.1f} ms/tok ({e2e_top2_results['tokens_per_sec']:.1f} tok/s)")
    log(f"    Average ratio top-1: {latency_ratio_top1:.3f}x, top-2: {latency_ratio_top2:.3f}x")
    log(f"    Merged-only (worst-case) ratio top-1: {merged_only_ratios.get('top1', 'N/A')}x, "
        f"top-2: {merged_only_ratios.get('top2', 'N/A')}x")
    log(f"    K1: {'PASS' if k1_pass else 'FAIL (KILL)'}")

    # --- Quality comparison (K2) ---
    k2_failures = []

    for domain in DOMAINS:
        base_ppl = base_results["ppl"][domain]
        e2e1_ppl = e2e_top1_results["ppl"][domain]
        e2e2_ppl = e2e_top2_results["ppl"][domain]

        # PPL: lower is better. E2E worse = higher PPL (strict, as pre-registered)
        e2e1_worse = e2e1_ppl > base_ppl
        e2e2_worse = e2e2_ppl > base_ppl

        quality_entry = {
            "base_ppl": base_ppl,
            "e2e_top1_ppl": e2e1_ppl,
            "e2e_top2_ppl": e2e2_ppl,
            "e2e_top1_improvement_pct": round((base_ppl - e2e1_ppl) / base_ppl * 100, 2),
            "e2e_top2_improvement_pct": round((base_ppl - e2e2_ppl) / base_ppl * 100, 2),
        }

        # Task-specific metrics
        if domain == "code":
            base_gen = base_results["generation_results"][domain]
            e2e1_gen = e2e_top1_results["generation_results"][domain]
            e2e2_gen = e2e_top2_results["generation_results"][domain]
            base_syntax = np.mean([r["syntax_valid"] for r in base_gen])
            e2e1_syntax = np.mean([r["syntax_valid"] for r in e2e1_gen])
            e2e2_syntax = np.mean([r["syntax_valid"] for r in e2e2_gen])
            quality_entry["base_syntax_valid_rate"] = round(float(base_syntax), 3)
            quality_entry["e2e_top1_syntax_valid_rate"] = round(float(e2e1_syntax), 3)
            quality_entry["e2e_top2_syntax_valid_rate"] = round(float(e2e2_syntax), 3)

        elif domain == "math":
            base_gen = base_results["generation_results"][domain]
            e2e1_gen = e2e_top1_results["generation_results"][domain]
            e2e2_gen = e2e_top2_results["generation_results"][domain]
            base_correct = np.mean([r["answer_correct"] for r in base_gen])
            e2e1_correct = np.mean([r["answer_correct"] for r in e2e1_gen])
            e2e2_correct = np.mean([r["answer_correct"] for r in e2e2_gen])
            quality_entry["base_answer_correct_rate"] = round(float(base_correct), 3)
            quality_entry["e2e_top1_answer_correct_rate"] = round(float(e2e1_correct), 3)
            quality_entry["e2e_top2_answer_correct_rate"] = round(float(e2e2_correct), 3)

        analysis["quality"][domain] = quality_entry

        # K2 check: quality worse on ANY domain
        # Use PPL as primary metric, task metrics as supporting evidence
        if e2e1_worse or e2e2_worse:
            k2_failures.append({
                "domain": domain,
                "top1_worse": e2e1_worse,
                "top2_worse": e2e2_worse,
                "base_ppl": base_ppl,
                "top1_ppl": e2e1_ppl,
                "top2_ppl": e2e2_ppl,
            })

    # K2: Quality worse than base alone on ANY domain (strict comparison)
    k2_pass = len(k2_failures) == 0
    analysis["kill_criteria"]["K2"] = {
        "test": "Quality worse than base alone on any domain",
        "result": "PASS" if k2_pass else "FAIL (KILL)",
        "failures": k2_failures,
        "evidence": f"{len(k2_failures)}/{N_DOMAINS} domains worse",
    }

    log(f"\n  K2 Quality (PPL, lower=better):")
    for domain in DOMAINS:
        q = analysis["quality"][domain]
        log(f"    {domain}: base={q['base_ppl']:.2f}, "
            f"top1={q['e2e_top1_ppl']:.2f} ({q['e2e_top1_improvement_pct']:+.1f}%), "
            f"top2={q['e2e_top2_ppl']:.2f} ({q['e2e_top2_improvement_pct']:+.1f}%)")
    log(f"    K2: {'PASS' if k2_pass else 'FAIL (KILL)'}")

    if not k2_pass:
        log(f"    Failures: {[f['domain'] for f in k2_failures]}")

    # Task-specific quality
    log(f"\n  Task-specific quality:")
    for domain in ["code", "math"]:
        q = analysis["quality"][domain]
        if domain == "code":
            log(f"    Code syntax: base={q.get('base_syntax_valid_rate', 'N/A')}, "
                f"top1={q.get('e2e_top1_syntax_valid_rate', 'N/A')}, "
                f"top2={q.get('e2e_top2_syntax_valid_rate', 'N/A')}")
        elif domain == "math":
            log(f"    Math correct: base={q.get('base_answer_correct_rate', 'N/A')}, "
                f"top1={q.get('e2e_top1_answer_correct_rate', 'N/A')}, "
                f"top2={q.get('e2e_top2_answer_correct_rate', 'N/A')}")

    # S1: Interactive serving
    s1_pass = e2e1_ms < 100  # <100ms per token
    analysis["kill_criteria"]["S1"] = {
        "test": "E2E pipeline serves interactively (<100ms/token) with quality >= base",
        "result": "PASS" if (s1_pass and k2_pass) else "FAIL",
        "evidence": f"top1={e2e1_ms:.1f}ms/tok, quality={'PASS' if k2_pass else 'FAIL'}",
    }
    log(f"\n  S1: Interactive serving: {e2e1_ms:.1f}ms/tok (<100ms target), "
        f"quality {'PASS' if k2_pass else 'FAIL'} -> {'PASS' if (s1_pass and k2_pass) else 'FAIL'}")

    # Overall verdict
    overall = "SUPPORTED" if k1_pass and k2_pass else "KILLED"
    analysis["verdict"] = overall
    log(f"\n  VERDICT: {overall}")

    return analysis


# ============================================================================
# Main
# ============================================================================

def main():
    t0_total = time.time()
    log("=" * 70)
    log("E2E Demo Pipeline: BitNet-SOLE on M5 Pro")
    log("=" * 70)
    log(f"Model: {MODEL_ID}")
    log(f"Domains: {DOMAINS}")
    log(f"Prompts per domain: {NUM_PROMPTS_PER_DOMAIN}")
    log(f"Max new tokens: {MAX_NEW_TOKENS}")
    log(f"Entropy threshold: {OTSU_THRESHOLD} nats")
    log_memory("start")

    # --- Extract prompts ---
    log("\nExtracting prompts from validation data...")
    prompts_by_domain = {}
    for domain in DOMAINS:
        prompts = extract_prompts_with_answers(domain, n_prompts=NUM_PROMPTS_PER_DOMAIN)
        prompts_by_domain[domain] = prompts
        log(f"  {domain}: {len(prompts)} prompts")

    # --- Phase 1: Base generation ---
    base_results = phase_base_generation(prompts_by_domain)

    # --- Phase 2a: E2E pipeline, top-1 ---
    e2e_top1_results = phase_e2e_generation(prompts_by_domain, top_k=1)

    # --- Phase 2b: E2E pipeline, top-2 ---
    e2e_top2_results = phase_e2e_generation(prompts_by_domain, top_k=2)

    # --- Phase 3: Analysis ---
    analysis = phase_analyze(base_results, e2e_top1_results, e2e_top2_results)

    # --- Save results ---
    total_time = time.time() - t0_total
    results = {
        "experiment": "e2e_demo_pipeline_mlx",
        "model": MODEL_ID,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_s": round(total_time, 1),
        "config": {
            "domains": DOMAINS,
            "num_prompts_per_domain": NUM_PROMPTS_PER_DOMAIN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "entropy_threshold": OTSU_THRESHOLD,
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
        },
        "base": base_results,
        "e2e_top1": e2e_top1_results,
        "e2e_top2": e2e_top2_results,
        "analysis": analysis,
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    log_memory("final")


if __name__ == "__main__":
    main()
