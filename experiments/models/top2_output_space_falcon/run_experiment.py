#!/usr/bin/env python3
"""Top-2 Output-Space Composition on Falcon-E-3B.

Tests whether output-space composition (average logits of 2 independently-run
adapters) beats single-best adapter and parameter merge, matching MoE practice.

Kill criteria:
  K1 (#556): Output-space top-2 not better than single best adapter
  K2 (#557): Speed below 30 tok/s (unusable)

Success criteria:
  S1 (#65): Output-space top-2 superlinear (beats single adapter on >=3/5 domains) at >40 tok/s

Prior art:
  - arXiv:2506.13479: k=2 enables superlinear gains, k>=3 degrades
  - LoRI (arXiv:2504.07448): output-space eliminates cross-terms mathematically
  - MoE practice: Mixtral top-2/8, DeepSeek-V3 top-2/256
  - exp_falcon_e3b_composition: Falcon-E-3B base beats Qwen 5/6, uniform merge hurts
  - exp_lora_soups_cat: orthogonal adapters prevent superlinear in param-merge

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
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse adapters from falcon_e3b_composition
FALCON_ADAPTERS_DIR = EXPERIMENT_DIR.parent / "falcon_e3b_composition" / "adapters"
DATA_DIR = EXPERIMENT_DIR.parent / "real_data_25_domain_adapters" / "data"

MODEL_ID_FALCON = "mlx-community/Falcon-E-3B-Instruct-1.58bit"

# LoRA config (must match training)
LORA_RANK = 16
LORA_SCALE = 20.0
TARGET_KEYS = ["self_attn.q_proj", "self_attn.v_proj", "self_attn.o_proj"]

DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = len(DOMAINS)

# Eval config
MMLU_SUBJECTS = {
    "medical": ["clinical_knowledge", "professional_medicine", "anatomy", "medical_genetics"],
    "code": ["college_computer_science", "high_school_computer_science", "machine_learning"],
    "math": ["high_school_mathematics", "elementary_mathematics", "college_mathematics"],
    "legal": ["professional_law", "jurisprudence", "international_law"],
    "finance": ["professional_accounting", "econometrics", "high_school_macroeconomics"],
}
MMLU_N_PER_DOMAIN = 20
GSM8K_N = 50
MAX_NEW_TOKENS_GSM8K = 256
MAX_NEW_TOKENS_MMLU = 32
SEED = 42

# Speed test config
SPEED_TEST_TOKENS = 100
SPEED_TEST_PROMPT = "Explain the concept of supply and demand in economics."


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
# Model loading and ternary unpacking
# ============================================================================

from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.lora import LoRALinear
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
# LoRA application helpers
# ============================================================================

def apply_adapter_to_model(model, adapter_path):
    """Apply a single LoRA adapter's delta directly into model weights.

    Instead of using LoRALinear, we merge the adapter delta into the base weights:
    W_eff = W + scale * B^T @ A^T  (in weight matrix form)

    This way we can run a standard forward pass without LoRA overhead.
    """
    adapter = dict(mx.load(str(adapter_path)))
    merge_count = 0

    for name, param in adapter.items():
        parts = name.split(".")
        if parts[-1] != "lora_a":
            continue

        b_name = ".".join(parts[:-1]) + ".lora_b"
        if b_name not in adapter:
            continue

        lora_a = param  # (in_features, r)
        lora_b = adapter[b_name]  # (r, out_features)

        # Navigate to the module
        module = model
        for p in parts[:-1]:
            if p.isdigit():
                module = module[int(p)]
            else:
                module = getattr(module, p, None)
            if module is None:
                break

        if module is None:
            continue

        if hasattr(module, "weight"):
            weight = module.weight
        else:
            continue

        # delta_W = scale * B^T @ A^T
        delta = LORA_SCALE * (lora_b.T @ lora_a.T)
        module.weight = weight + delta
        merge_count += 1

    mx.eval(model.parameters())
    del adapter
    return merge_count


def remove_adapter_from_model(model, adapter_path):
    """Remove a previously applied adapter delta from model weights."""
    adapter = dict(mx.load(str(adapter_path)))
    count = 0

    for name, param in adapter.items():
        parts = name.split(".")
        if parts[-1] != "lora_a":
            continue

        b_name = ".".join(parts[:-1]) + ".lora_b"
        if b_name not in adapter:
            continue

        lora_a = param
        lora_b = adapter[b_name]

        module = model
        for p in parts[:-1]:
            if p.isdigit():
                module = module[int(p)]
            else:
                module = getattr(module, p, None)
            if module is None:
                break

        if module is None:
            continue

        if hasattr(module, "weight"):
            weight = module.weight
        else:
            continue

        delta = LORA_SCALE * (lora_b.T @ lora_a.T)
        module.weight = weight - delta
        count += 1

    mx.eval(model.parameters())
    del adapter
    return count


# ============================================================================
# Routing: simple embedding-similarity based
# ============================================================================

def compute_domain_embeddings(tokenizer):
    """Compute simple domain embeddings using domain description tokens.

    For each domain, tokenize a representative description and use the
    mean token embedding as the domain's routing signature.
    """
    domain_descriptions = {
        "medical": "medicine health clinical diagnosis treatment patient disease",
        "code": "programming software code function algorithm variable class",
        "math": "mathematics equation theorem proof calculus algebra number",
        "legal": "law court judge statute regulation contract liability",
        "finance": "finance investment banking stock market portfolio economy",
    }
    return domain_descriptions


def route_to_top2(text, domain_descriptions):
    """Route a text to top-2 domains using keyword overlap scoring.

    Simple but effective: count domain keyword matches in the input text.
    This avoids the binary-head-collapse problem (exp_binary_routing_head_collapse)
    and the depth-routing failure (exp_depth_routed_adapters).
    """
    text_lower = text.lower()
    scores = {}
    for domain, desc in domain_descriptions.items():
        keywords = desc.split()
        score = sum(1 for kw in keywords if kw in text_lower)
        # Add domain-specific bonus keywords
        bonus_keywords = {
            "medical": ["symptom", "drug", "therapy", "hospital", "clinical", "doctor",
                       "surgery", "prescription", "diagnosis", "pathology"],
            "code": ["python", "javascript", "function", "class", "error", "debug",
                    "compile", "syntax", "api", "database"],
            "math": ["solve", "equation", "derivative", "integral", "probability",
                    "matrix", "vector", "theorem", "calculate", "formula"],
            "legal": ["plaintiff", "defendant", "verdict", "statute", "amendment",
                     "jurisdiction", "rights", "constitution", "criminal", "civil"],
            "finance": ["profit", "revenue", "interest", "dividend", "asset",
                       "liability", "budget", "inflation", "gdp", "trade"],
        }
        if domain in bonus_keywords:
            score += sum(1 for kw in bonus_keywords[domain] if kw in text_lower)
        scores[domain] = score

    # Sort by score descending, take top 2
    ranked = sorted(scores.items(), key=lambda x: -x[1])

    # If no keywords match (common for MMLU), use domain label from the question context
    if ranked[0][1] == 0:
        # Fallback: return the first two domains (will be compared fairly)
        return [ranked[0][0], ranked[1][0]]

    return [ranked[0][0], ranked[1][0]]


def route_with_oracle(domain):
    """Oracle routing: always include the correct domain adapter."""
    other_domains = [d for d in DOMAINS if d != domain]
    # Pick the domain adapter + one complementary adapter
    # Use a fixed secondary based on empirical complementarity
    complementary = {
        "medical": "code",      # medical often needs structured reasoning
        "code": "math",         # code and math share analytical thinking
        "math": "code",         # math benefits from algorithmic thinking
        "legal": "finance",     # legal and finance overlap in contracts/regulation
        "finance": "legal",     # finance benefits from regulatory knowledge
    }
    secondary = complementary.get(domain, other_domains[0])
    return [domain, secondary]


# ============================================================================
# Evaluation helpers (reused from falcon_e3b_composition)
# ============================================================================

def format_gsm8k_prompt(question):
    return (
        f"### Instruction:\n"
        f"Solve the following math problem step by step. "
        f"Show your work and give the final numerical answer after ####.\n\n"
        f"{question}\n\n"
        f"### Response:\n"
    )


def format_mmlu_prompt(question, choices):
    choice_labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(f"{l}. {c}" for l, c in zip(choice_labels, choices))
    return (
        f"### Instruction:\n"
        f"Answer the following multiple choice question. "
        f"Reply with just the letter (A, B, C, or D).\n\n"
        f"{question}\n\n{choices_text}\n\n"
        f"### Response:\n"
    )


def extract_gsm8k_answer(text):
    match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', text)
    if match:
        return float(match.group(1).replace(',', ''))
    match = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*\$?([\d,]+(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(',', ''))
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


def check_gsm8k_correct(predicted, ground_truth, tolerance=0.01):
    if predicted is None or ground_truth is None:
        return False
    if ground_truth == 0:
        return abs(predicted) < tolerance
    return abs(predicted - ground_truth) / abs(ground_truth) < tolerance


def extract_mmlu_answer(text):
    text = text.strip()
    if text and text[0].upper() in "ABCD":
        return text[0].upper()
    match = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*([A-Da-d])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    for line in text.split('\n'):
        line = line.strip()
        if len(line) == 1 and line.upper() in "ABCD":
            return line.upper()
    match = re.search(r'[\(\s]([A-Da-d])[\)\.\s]', text)
    if match:
        return match.group(1).upper()
    match = re.search(r'\b([A-Da-d])\b', text)
    if match:
        return match.group(1).upper()
    return None


def generate_text(model, tokenizer, prompt, max_tokens=256):
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
# Output-space composition: generate with logit averaging
# ============================================================================

def generate_output_space_top2(model, tokenizer, prompt, adapter_paths,
                                top2_domains, max_tokens=256):
    """Generate text using output-space top-2 composition.

    For each token:
    1. Apply adapter 1, get logits_1
    2. Apply adapter 2, get logits_2
    3. Average logits: logits = 0.5 * logits_1 + 0.5 * logits_2
    4. Sample from averaged logits
    5. Append token, repeat

    This is the core of the experiment: MoE-style output-space composition.
    """
    # Tokenize prompt
    tokens = tokenizer.encode(prompt)
    token_ids = mx.array([tokens])

    # Load both adapter weight dicts
    path_1 = adapter_paths[top2_domains[0]]
    path_2 = adapter_paths[top2_domains[1]]

    adapter_1 = dict(mx.load(str(path_1)))
    adapter_2 = dict(mx.load(str(path_2)))

    # Pre-compute deltas for both adapters (list of (module_path, delta) pairs)
    def compute_deltas(adapter_dict):
        deltas = []
        for name, param in adapter_dict.items():
            parts = name.split(".")
            if parts[-1] != "lora_a":
                continue
            b_name = ".".join(parts[:-1]) + ".lora_b"
            if b_name not in adapter_dict:
                continue
            lora_a = param
            lora_b = adapter_dict[b_name]
            delta = LORA_SCALE * (lora_b.T @ lora_a.T)
            deltas.append((parts[:-1], delta))
        return deltas

    deltas_1 = compute_deltas(adapter_1)
    deltas_2 = compute_deltas(adapter_2)
    mx.eval([d for _, d in deltas_1] + [d for _, d in deltas_2])
    del adapter_1, adapter_2

    def navigate_to_module(model, parts):
        module = model
        for p in parts:
            if p.isdigit():
                module = module[int(p)]
            else:
                module = getattr(module, p, None)
            if module is None:
                return None
        return module

    # Save base weights for the affected modules
    base_weights = {}
    for parts, _ in deltas_1:
        key = ".".join(parts)
        module = navigate_to_module(model, parts)
        if module is not None and hasattr(module, "weight"):
            base_weights[key] = module.weight

    def apply_deltas(deltas):
        for parts, delta in deltas:
            key = ".".join(parts)
            module = navigate_to_module(model, parts)
            if module is not None and key in base_weights:
                module.weight = base_weights[key] + delta

    def restore_base():
        for parts, _ in deltas_1:
            key = ".".join(parts)
            module = navigate_to_module(model, parts)
            if module is not None and key in base_weights:
                module.weight = base_weights[key]

    generated_tokens = []
    cache_1 = None
    cache_2 = None

    try:
        for step in range(max_tokens):
            if step == 0:
                input_ids = token_ids
            else:
                input_ids = mx.array([[generated_tokens[-1]]])

            # Forward pass with adapter 1
            apply_deltas(deltas_1)
            if step == 0:
                logits_1 = model(input_ids)
            else:
                logits_1 = model(input_ids, cache=cache_1)
            # We cannot use KV cache across adapter swaps because the
            # KV values were computed with different weights. So we must
            # recompute from scratch each time, OR accept no caching.
            # For simplicity and correctness: no KV cache (slower but correct).
            # Actually let's use a simpler approach: full sequence each time.

            # Restart: simpler approach without KV cache
            # Just do full forward pass each time
            break

        # Simpler approach: build full sequence, do 2 full forward passes per token
        # This is slow but correct. For speed test we'll measure separately.
        generated_tokens = []
        current_ids = list(tokens)

        for step in range(max_tokens):
            input_tensor = mx.array([current_ids])

            # Forward with adapter 1
            apply_deltas(deltas_1)
            logits_1 = model(input_tensor)
            logits_1 = logits_1[:, -1, :]  # last token logits
            mx.eval(logits_1)

            # Forward with adapter 2
            apply_deltas(deltas_2)
            logits_2 = model(input_tensor)
            logits_2 = logits_2[:, -1, :]
            mx.eval(logits_2)

            # Average logits (output-space composition)
            avg_logits = 0.5 * logits_1 + 0.5 * logits_2
            mx.eval(avg_logits)

            # Greedy decode (temp=0)
            next_token = mx.argmax(avg_logits, axis=-1).item()

            del logits_1, logits_2, avg_logits

            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token)
            current_ids.append(next_token)

            # Limit sequence length to prevent OOM
            if len(current_ids) > 512:
                break

        # Restore base weights
        restore_base()

        result_text = tokenizer.decode(generated_tokens)
        return result_text

    except Exception as e:
        restore_base()
        log(f"  WARNING: output-space generation failed: {e}")
        return ""


# ============================================================================
# Simplified output-space: pre-compute full logits then average
# For MMLU (short generation), this is more practical
# ============================================================================

def get_next_token_logits(model, tokenizer, prompt):
    """Get logits for the next token after prompt. Used for MMLU."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits = model(input_ids)
    next_logits = logits[:, -1, :]
    mx.eval(next_logits)
    del logits
    return next_logits


def output_space_mmlu_answer(model, tokenizer, prompt, adapter_paths, top2_domains):
    """Get MMLU answer using output-space top-2 composition.

    For MMLU, we only need the FIRST generated token (A/B/C/D).
    So: apply adapter 1, get logits; apply adapter 2, get logits; average; pick best.
    """
    path_1 = adapter_paths[top2_domains[0]]
    path_2 = adapter_paths[top2_domains[1]]

    # Apply adapter 1, get logits
    n1 = apply_adapter_to_model(model, path_1)
    logits_1 = get_next_token_logits(model, tokenizer, prompt)
    remove_adapter_from_model(model, path_1)

    # Apply adapter 2, get logits
    n2 = apply_adapter_to_model(model, path_2)
    logits_2 = get_next_token_logits(model, tokenizer, prompt)
    remove_adapter_from_model(model, path_2)

    # Average logits
    avg_logits = 0.5 * logits_1 + 0.5 * logits_2
    mx.eval(avg_logits)

    # Decode the top token
    next_token = mx.argmax(avg_logits, axis=-1).item()
    text = tokenizer.decode([next_token])
    del logits_1, logits_2, avg_logits
    return text


# ============================================================================
# Benchmark data loading
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


def load_mmlu_data(n_per_domain=20):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    by_subject = {}
    for item in ds:
        subj = item["subject"]
        if subj not in by_subject:
            by_subject[subj] = []
        by_subject[subj].append(item)

    mmlu_data = {}
    rng = np.random.RandomState(42)
    for domain, subjects in MMLU_SUBJECTS.items():
        questions = []
        for subj in subjects:
            if subj in by_subject:
                questions.extend(by_subject[subj])
        rng.shuffle(questions)
        mmlu_data[domain] = questions[:n_per_domain]
        log(f"  MMLU {domain}: {len(mmlu_data[domain])} questions")
    return mmlu_data


# ============================================================================
# Phase 1: Evaluate base model (no adapters)
# ============================================================================

def phase_eval_base(mmlu):
    """Evaluate Falcon-E-3B base on MMLU domains."""
    log("\n" + "=" * 70)
    log("PHASE 1: FALCON-E-3B BASE (no adapters)")
    log("=" * 70)
    t0 = time.time()
    mx.reset_peak_memory()

    model, tokenizer = load(MODEL_ID_FALCON)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("base-loaded")

    results = {}
    choice_labels = ["A", "B", "C", "D"]

    for domain in DOMAINS:
        questions = mmlu[domain]
        correct = 0
        total = len(questions)

        for i, q in enumerate(questions):
            prompt = format_mmlu_prompt(q["question"], q["choices"])
            generated = generate_text(model, tokenizer, prompt, max_tokens=MAX_NEW_TOKENS_MMLU)
            predicted = extract_mmlu_answer(generated)
            gt_label = choice_labels[q["answer"]]
            if predicted == gt_label:
                correct += 1

        accuracy = correct / total if total > 0 else 0
        results[domain] = {"accuracy": accuracy, "correct": correct, "total": total}
        log(f"  Base MMLU {domain}: {correct}/{total} = {accuracy:.3f}")

    peak_mem = mx.get_peak_memory() / 1e9
    elapsed = time.time() - t0
    log(f"  Base eval done in {elapsed:.1f}s, peak={peak_mem:.2f}GB")

    cleanup(model, tokenizer)
    return {"mmlu": results, "peak_memory_gb": peak_mem, "time_s": elapsed}


# ============================================================================
# Phase 2: Evaluate single-best adapter per domain
# ============================================================================

def phase_eval_single_adapter(mmlu):
    """Evaluate each domain with its single matched adapter."""
    log("\n" + "=" * 70)
    log("PHASE 2: SINGLE BEST ADAPTER (oracle: domain-matched)")
    log("=" * 70)
    t0 = time.time()

    adapter_paths = {d: FALCON_ADAPTERS_DIR / d / "adapter.npz" for d in DOMAINS}
    results = {}
    choice_labels = ["A", "B", "C", "D"]

    for domain in DOMAINS:
        mx.reset_peak_memory()
        log(f"\n  --- Evaluating {domain} with {domain} adapter ---")

        model, tokenizer = load(MODEL_ID_FALCON)
        model = replace_bitlinear_with_linear(model)

        # Apply only the matched domain adapter
        n_merged = apply_adapter_to_model(model, adapter_paths[domain])
        model.freeze()
        log(f"  Applied {domain} adapter ({n_merged} weight matrices)")

        questions = mmlu[domain]
        correct = 0
        total = len(questions)

        for i, q in enumerate(questions):
            prompt = format_mmlu_prompt(q["question"], q["choices"])
            generated = generate_text(model, tokenizer, prompt, max_tokens=MAX_NEW_TOKENS_MMLU)
            predicted = extract_mmlu_answer(generated)
            gt_label = choice_labels[q["answer"]]
            if predicted == gt_label:
                correct += 1

        accuracy = correct / total if total > 0 else 0
        results[domain] = {"accuracy": accuracy, "correct": correct, "total": total}
        log(f"  Single-adapter MMLU {domain}: {correct}/{total} = {accuracy:.3f}")

        cleanup(model, tokenizer)

    elapsed = time.time() - t0
    log(f"  Single-adapter eval done in {elapsed:.1f}s")
    return {"mmlu": results, "time_s": elapsed}


# ============================================================================
# Phase 3: Evaluate parameter merge (uniform 1/N)
# ============================================================================

def phase_eval_param_merge(mmlu):
    """Evaluate uniform 1/N parameter merge of all 5 adapters."""
    log("\n" + "=" * 70)
    log("PHASE 3: PARAMETER MERGE (uniform 1/5)")
    log("=" * 70)
    t0 = time.time()
    mx.reset_peak_memory()

    model, tokenizer = load(MODEL_ID_FALCON)
    model = replace_bitlinear_with_linear(model)

    adapter_paths = {d: FALCON_ADAPTERS_DIR / d / "adapter.npz" for d in DOMAINS}

    # Merge all 5 adapters with uniform 1/5 weight
    total_merged = 0
    for domain in DOMAINS:
        adapter = dict(mx.load(str(adapter_paths[domain])))
        merge_count = 0
        for name, param in adapter.items():
            parts = name.split(".")
            if parts[-1] != "lora_a":
                continue
            b_name = ".".join(parts[:-1]) + ".lora_b"
            if b_name not in adapter:
                continue
            lora_a = param
            lora_b = adapter[b_name]

            module = model
            for p in parts[:-1]:
                if p.isdigit():
                    module = module[int(p)]
                else:
                    module = getattr(module, p, None)
                if module is None:
                    break

            if module is None or not hasattr(module, "weight"):
                continue

            delta = (1.0 / N_DOMAINS) * LORA_SCALE * (lora_b.T @ lora_a.T)
            module.weight = module.weight + delta
            merge_count += 1
        total_merged += merge_count
        del adapter
        gc.collect()

    mx.eval(model.parameters())
    model.freeze()
    log(f"  Merged {N_DOMAINS} adapters into {total_merged} weight matrices")
    log_memory("param-merge-loaded")

    results = {}
    choice_labels = ["A", "B", "C", "D"]

    for domain in DOMAINS:
        questions = mmlu[domain]
        correct = 0
        total = len(questions)

        for i, q in enumerate(questions):
            prompt = format_mmlu_prompt(q["question"], q["choices"])
            generated = generate_text(model, tokenizer, prompt, max_tokens=MAX_NEW_TOKENS_MMLU)
            predicted = extract_mmlu_answer(generated)
            gt_label = choice_labels[q["answer"]]
            if predicted == gt_label:
                correct += 1

        accuracy = correct / total if total > 0 else 0
        results[domain] = {"accuracy": accuracy, "correct": correct, "total": total}
        log(f"  Param-merge MMLU {domain}: {correct}/{total} = {accuracy:.3f}")

    peak_mem = mx.get_peak_memory() / 1e9
    elapsed = time.time() - t0
    log(f"  Param-merge eval done in {elapsed:.1f}s, peak={peak_mem:.2f}GB")

    cleanup(model, tokenizer)
    return {"mmlu": results, "peak_memory_gb": peak_mem, "time_s": elapsed}


# ============================================================================
# Phase 4: Evaluate output-space top-2 (THE KEY TEST)
# ============================================================================

def phase_eval_output_space_top2(mmlu):
    """Evaluate output-space top-2 composition on MMLU.

    For each domain question:
    1. Oracle-route to top-2 adapters (domain + complementary)
    2. Run forward pass with adapter 1, get logits
    3. Run forward pass with adapter 2, get logits
    4. Average logits, pick answer
    """
    log("\n" + "=" * 70)
    log("PHASE 4: OUTPUT-SPACE TOP-2 (oracle routing)")
    log("=" * 70)
    t0 = time.time()
    mx.reset_peak_memory()

    model, tokenizer = load(MODEL_ID_FALCON)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("output-space-loaded")

    adapter_paths = {d: FALCON_ADAPTERS_DIR / d / "adapter.npz" for d in DOMAINS}

    results = {}
    choice_labels = ["A", "B", "C", "D"]

    for domain in DOMAINS:
        top2 = route_with_oracle(domain)
        questions = mmlu[domain]
        correct = 0
        total = len(questions)

        log(f"\n  --- {domain}: output-space top-2 with [{top2[0]}, {top2[1]}] ---")

        for i, q in enumerate(questions):
            prompt = format_mmlu_prompt(q["question"], q["choices"])

            # Output-space: get logits from each adapter independently
            text = output_space_mmlu_answer(
                model, tokenizer, prompt, adapter_paths, top2
            )
            predicted = extract_mmlu_answer(text)
            gt_label = choice_labels[q["answer"]]
            if predicted == gt_label:
                correct += 1

            if (i + 1) % 10 == 0:
                log(f"    {i+1}/{total}: {correct}/{i+1} ({100*correct/(i+1):.0f}%)")

        accuracy = correct / total if total > 0 else 0
        results[domain] = {"accuracy": accuracy, "correct": correct, "total": total,
                           "adapters_used": top2}
        log(f"  Output-space top-2 MMLU {domain}: {correct}/{total} = {accuracy:.3f}")

        gc.collect()
        mx.clear_cache()

    peak_mem = mx.get_peak_memory() / 1e9
    elapsed = time.time() - t0
    log(f"  Output-space top-2 eval done in {elapsed:.1f}s, peak={peak_mem:.2f}GB")

    cleanup(model, tokenizer)
    return {"mmlu": results, "peak_memory_gb": peak_mem, "time_s": elapsed}


# ============================================================================
# Phase 4b: Output-space top-2 with keyword routing (not oracle)
# ============================================================================

def phase_eval_output_space_top2_routed(mmlu):
    """Evaluate output-space top-2 with keyword-based routing."""
    log("\n" + "=" * 70)
    log("PHASE 4b: OUTPUT-SPACE TOP-2 (keyword routing)")
    log("=" * 70)
    t0 = time.time()
    mx.reset_peak_memory()

    model, tokenizer = load(MODEL_ID_FALCON)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("output-space-routed-loaded")

    adapter_paths = {d: FALCON_ADAPTERS_DIR / d / "adapter.npz" for d in DOMAINS}
    domain_descriptions = compute_domain_embeddings(tokenizer)

    results = {}
    routing_stats = {d: {} for d in DOMAINS}
    choice_labels = ["A", "B", "C", "D"]

    for domain in DOMAINS:
        questions = mmlu[domain]
        correct = 0
        total = len(questions)
        routes_used = []

        log(f"\n  --- {domain}: output-space top-2 with keyword routing ---")

        for i, q in enumerate(questions):
            prompt = format_mmlu_prompt(q["question"], q["choices"])
            top2 = route_to_top2(q["question"], domain_descriptions)
            routes_used.append(top2)

            text = output_space_mmlu_answer(
                model, tokenizer, prompt, adapter_paths, top2
            )
            predicted = extract_mmlu_answer(text)
            gt_label = choice_labels[q["answer"]]
            if predicted == gt_label:
                correct += 1

        accuracy = correct / total if total > 0 else 0

        # Analyze routing
        correct_routes = sum(1 for r in routes_used if domain in r)
        results[domain] = {
            "accuracy": accuracy, "correct": correct, "total": total,
            "routing_accuracy": correct_routes / total if total > 0 else 0,
        }
        log(f"  Routed top-2 MMLU {domain}: {correct}/{total} = {accuracy:.3f} "
            f"(routing accuracy: {correct_routes}/{total})")

        gc.collect()
        mx.clear_cache()

    peak_mem = mx.get_peak_memory() / 1e9
    elapsed = time.time() - t0
    log(f"  Routed output-space eval done in {elapsed:.1f}s, peak={peak_mem:.2f}GB")

    cleanup(model, tokenizer)
    return {"mmlu": results, "peak_memory_gb": peak_mem, "time_s": elapsed}


# ============================================================================
# Phase 5: Speed benchmark
# ============================================================================

def phase_speed_benchmark():
    """Measure tok/s for base, single adapter, and output-space top-2."""
    log("\n" + "=" * 70)
    log("PHASE 5: SPEED BENCHMARK")
    log("=" * 70)
    mx.reset_peak_memory()

    model, tokenizer = load(MODEL_ID_FALCON)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    adapter_paths = {d: FALCON_ADAPTERS_DIR / d / "adapter.npz" for d in DOMAINS}
    prompt = SPEED_TEST_PROMPT
    n_tokens = SPEED_TEST_TOKENS

    results = {}

    # 1. Base model speed
    log("\n  Measuring base model speed...")
    sampler = make_sampler(temp=0.0)
    # Warmup
    _ = mlx_generate(model, tokenizer, prompt, max_tokens=10, sampler=sampler, verbose=False)

    t0 = time.time()
    _ = mlx_generate(model, tokenizer, prompt, max_tokens=n_tokens, sampler=sampler, verbose=False)
    elapsed = time.time() - t0
    base_tps = n_tokens / elapsed
    results["base"] = {"tok_per_sec": base_tps, "time_s": elapsed, "n_tokens": n_tokens}
    log(f"  Base: {base_tps:.1f} tok/s ({elapsed:.2f}s for {n_tokens} tokens)")

    # 2. Single adapter speed
    log("\n  Measuring single adapter speed...")
    apply_adapter_to_model(model, adapter_paths["math"])
    # Warmup
    _ = mlx_generate(model, tokenizer, prompt, max_tokens=10, sampler=sampler, verbose=False)

    t0 = time.time()
    _ = mlx_generate(model, tokenizer, prompt, max_tokens=n_tokens, sampler=sampler, verbose=False)
    elapsed = time.time() - t0
    single_tps = n_tokens / elapsed
    results["single_adapter"] = {"tok_per_sec": single_tps, "time_s": elapsed, "n_tokens": n_tokens}
    log(f"  Single adapter: {single_tps:.1f} tok/s ({elapsed:.2f}s for {n_tokens} tokens)")
    remove_adapter_from_model(model, adapter_paths["math"])

    # 3. Output-space top-2 speed (the expensive one)
    # For speed: measure time to do 2 full forward passes and average logits
    # This is per-token cost, not full generation (which would need the slow loop)
    log("\n  Measuring output-space top-2 per-token speed...")

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Warmup
    logits_warm = model(input_ids)
    mx.eval(logits_warm)
    del logits_warm

    # Measure: 2 forward passes + logit average, repeated n_tokens times
    n_measure = 50  # enough for reliable timing
    t0 = time.time()
    for _ in range(n_measure):
        # Apply adapter 1
        apply_adapter_to_model(model, adapter_paths["medical"])
        logits_1 = model(input_ids)
        mx.eval(logits_1)
        remove_adapter_from_model(model, adapter_paths["medical"])

        # Apply adapter 2
        apply_adapter_to_model(model, adapter_paths["code"])
        logits_2 = model(input_ids)
        mx.eval(logits_2)
        remove_adapter_from_model(model, adapter_paths["code"])

        # Average
        avg = 0.5 * logits_1 + 0.5 * logits_2
        mx.eval(avg)
        del logits_1, logits_2, avg

    elapsed = time.time() - t0
    # Each iteration = 1 "token" worth of compute (2 forward passes)
    output_space_tps = n_measure / elapsed
    results["output_space_top2"] = {
        "tok_per_sec": output_space_tps,
        "time_s": elapsed,
        "n_iterations": n_measure,
        "note": "per-token cost: 2 full forward passes + logit average. "
                "Actual generation would be slower due to sequential token dependency."
    }
    log(f"  Output-space top-2: {output_space_tps:.1f} tok/s equivalent "
        f"({elapsed:.2f}s for {n_measure} iterations)")

    # 4. More realistic: measure actual generation with adapter-swap per token
    # For a short generation (10 tokens), measure wall-clock
    log("\n  Measuring output-space actual generation speed (10 tokens)...")
    top2 = ["medical", "code"]

    t0 = time.time()
    current_ids = list(tokens)
    for step in range(10):
        input_tensor = mx.array([current_ids])

        apply_adapter_to_model(model, adapter_paths[top2[0]])
        logits_1 = model(input_tensor)
        logits_1_last = logits_1[:, -1, :]
        mx.eval(logits_1_last)
        remove_adapter_from_model(model, adapter_paths[top2[0]])

        apply_adapter_to_model(model, adapter_paths[top2[1]])
        logits_2 = model(input_tensor)
        logits_2_last = logits_2[:, -1, :]
        mx.eval(logits_2_last)
        remove_adapter_from_model(model, adapter_paths[top2[1]])

        avg_logits = 0.5 * logits_1_last + 0.5 * logits_2_last
        mx.eval(avg_logits)
        next_token = mx.argmax(avg_logits, axis=-1).item()
        current_ids.append(next_token)
        del logits_1, logits_2, logits_1_last, logits_2_last, avg_logits

    elapsed = time.time() - t0
    actual_tps = 10 / elapsed
    results["output_space_actual_gen"] = {
        "tok_per_sec": actual_tps,
        "time_s": elapsed,
        "n_tokens": 10,
        "note": "Actual generation with adapter swap per token. No KV cache."
    }
    log(f"  Output-space actual generation: {actual_tps:.1f} tok/s "
        f"({elapsed:.2f}s for 10 tokens)")

    peak_mem = mx.get_peak_memory() / 1e9
    results["peak_memory_gb"] = peak_mem
    log(f"\n  Speed test peak memory: {peak_mem:.2f}GB")

    cleanup(model, tokenizer)
    return results


# ============================================================================
# Analysis
# ============================================================================

def analyze_results(base_res, single_res, merge_res, output_space_res,
                    output_space_routed_res, speed_res):
    log("\n" + "=" * 70)
    log("ANALYSIS")
    log("=" * 70)

    # Compare MMLU accuracy across methods
    log("\n  MMLU ACCURACY BY DOMAIN:")
    log(f"  {'Domain':<12} {'Base':<8} {'Single':<8} {'Merge':<8} {'OS-Top2':<8} {'OS-Routed':<10}")
    log("  " + "-" * 56)

    domains_output_beats_single = 0
    domains_output_beats_base = 0

    for domain in DOMAINS:
        base_acc = base_res["mmlu"][domain]["accuracy"]
        single_acc = single_res["mmlu"][domain]["accuracy"]
        merge_acc = merge_res["mmlu"][domain]["accuracy"]
        os_acc = output_space_res["mmlu"][domain]["accuracy"]
        os_routed_acc = output_space_routed_res["mmlu"][domain]["accuracy"]

        if os_acc > single_acc:
            domains_output_beats_single += 1
        if os_acc > base_acc:
            domains_output_beats_base += 1

        log(f"  {domain:<12} {base_acc:<8.3f} {single_acc:<8.3f} {merge_acc:<8.3f} "
            f"{os_acc:<8.3f} {os_routed_acc:<10.3f}")

    # Averages
    avg_base = np.mean([base_res["mmlu"][d]["accuracy"] for d in DOMAINS])
    avg_single = np.mean([single_res["mmlu"][d]["accuracy"] for d in DOMAINS])
    avg_merge = np.mean([merge_res["mmlu"][d]["accuracy"] for d in DOMAINS])
    avg_os = np.mean([output_space_res["mmlu"][d]["accuracy"] for d in DOMAINS])
    avg_os_routed = np.mean([output_space_routed_res["mmlu"][d]["accuracy"] for d in DOMAINS])

    log(f"  {'AVERAGE':<12} {avg_base:<8.3f} {avg_single:<8.3f} {avg_merge:<8.3f} "
        f"{avg_os:<8.3f} {avg_os_routed:<10.3f}")

    # Speed
    log("\n  SPEED:")
    for method, data in speed_res.items():
        if isinstance(data, dict) and "tok_per_sec" in data:
            log(f"  {method}: {data['tok_per_sec']:.1f} tok/s")

    # Determine the most representative speed for K2
    # Use actual generation speed if available, else per-token estimate
    if "output_space_actual_gen" in speed_res:
        k2_speed = speed_res["output_space_actual_gen"]["tok_per_sec"]
    else:
        k2_speed = speed_res.get("output_space_top2", {}).get("tok_per_sec", 0)

    # Kill criteria
    log("\n  KILL CRITERIA:")

    k1_pass = domains_output_beats_single >= 3
    log(f"  K1 (#556): Output-space top-2 beats single adapter on "
        f"{domains_output_beats_single}/5 domains -> {'PASS' if k1_pass else 'FAIL'} "
        f"(threshold: >=3)")

    k2_pass = k2_speed >= 30
    log(f"  K2 (#557): Speed = {k2_speed:.1f} tok/s -> {'PASS' if k2_pass else 'FAIL'} "
        f"(threshold: >=30)")

    # Success criteria
    s1_pass = domains_output_beats_single >= 3 and k2_speed >= 40
    log(f"\n  SUCCESS CRITERIA:")
    log(f"  S1 (#65): Superlinear on {domains_output_beats_single}/5 domains at "
        f"{k2_speed:.1f} tok/s -> {'PASS' if s1_pass else 'FAIL'} "
        f"(threshold: >=3 domains at >40 tok/s)")

    return {
        "averages": {
            "base": avg_base, "single_adapter": avg_single,
            "param_merge": avg_merge, "output_space_top2": avg_os,
            "output_space_routed": avg_os_routed,
        },
        "domains_os_beats_single": domains_output_beats_single,
        "domains_os_beats_base": domains_output_beats_base,
        "k1_556": {"result": "pass" if k1_pass else "fail",
                    "evidence": f"OS top-2 beats single on {domains_output_beats_single}/5 domains"},
        "k2_557": {"result": "pass" if k2_pass else "fail",
                    "evidence": f"Speed = {k2_speed:.1f} tok/s"},
        "s1_65": {"result": "pass" if s1_pass else "fail",
                   "evidence": f"{domains_output_beats_single}/5 superlinear at {k2_speed:.1f} tok/s"},
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log_memory("start")

    # Verify adapters exist
    for domain in DOMAINS:
        path = FALCON_ADAPTERS_DIR / domain / "adapter.npz"
        if not path.exists():
            log(f"FATAL: Adapter not found: {path}")
            log("Run experiments/models/falcon_e3b_composition/run_experiment.py first.")
            return
    log("All 5 Falcon-E-3B adapters found.")

    # Load benchmark data
    log("\n" + "=" * 70)
    log("LOADING BENCHMARK DATA")
    log("=" * 70)
    mmlu = load_mmlu_data(MMLU_N_PER_DOMAIN)

    # Phase 1: Base model
    base_results = phase_eval_base(mmlu)

    # Phase 2: Single best adapter per domain
    single_results = phase_eval_single_adapter(mmlu)

    # Phase 3: Parameter merge (uniform 1/5)
    merge_results = phase_eval_param_merge(mmlu)

    # Phase 4: Output-space top-2 (oracle routing)
    output_space_results = phase_eval_output_space_top2(mmlu)

    # Phase 4b: Output-space top-2 (keyword routing)
    output_space_routed_results = phase_eval_output_space_top2_routed(mmlu)

    # Phase 5: Speed benchmark
    speed_results = phase_speed_benchmark()

    # Analysis
    analysis = analyze_results(
        base_results, single_results, merge_results,
        output_space_results, output_space_routed_results, speed_results
    )

    # Save results
    total_time = time.time() - t0
    output = {
        "experiment": "top2_output_space_falcon",
        "model": MODEL_ID_FALCON,
        "methods": {
            "base": "Falcon-E-3B-Instruct, no adapters",
            "single_adapter": "Oracle domain-matched single adapter (full delta)",
            "param_merge": "Uniform 1/5 parameter merge of all 5 adapters",
            "output_space_top2": "Output-space top-2 with oracle routing (domain + complementary)",
            "output_space_routed": "Output-space top-2 with keyword routing",
        },
        "results": {
            "base": base_results,
            "single_adapter": single_results,
            "param_merge": merge_results,
            "output_space_top2": output_space_results,
            "output_space_routed": output_space_routed_results,
        },
        "speed": speed_results,
        "analysis": analysis,
        "params": {
            "mmlu_n_per_domain": MMLU_N_PER_DOMAIN,
            "max_new_tokens_mmlu": MAX_NEW_TOKENS_MMLU,
            "temperature": 0.0,
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
            "n_adapters": N_DOMAINS,
            "composition": "output-space logit average (top-2)",
            "routing": "oracle (domain + complementary) and keyword-based",
        },
        "total_time_s": total_time,
    }

    RESULTS_FILE.write_text(json.dumps(output, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    for k, v in analysis.items():
        if isinstance(v, dict) and "result" in v:
            log(f"  {k}: {v['result'].upper()} - {v['evidence']}")


if __name__ == "__main__":
    main()
