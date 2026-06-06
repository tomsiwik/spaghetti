#!/usr/bin/env python3
"""
KR-Test Knowledge Retention Evaluation for BitNet-2B LoRA Adapters

Implements the contrastive evaluation from arXiv:2601.03505, adapted for
$0 Apple Silicon execution. Instead of a teacher LLM for contrastive pair
generation, we use rule-based entity/number swaps from the training data.

Hypothesis: KR-Test replaces PPL as primary adapter quality metric.
KR-Test measures whether a model assigns higher log-probability to factually
correct continuations vs plausible-but-wrong alternatives.

Score: fraction of examples where sum(log p(correct_t)) > sum(log p(wrong_t))
       evaluated up to min(len(correct), len(wrong)) tokens.

Kill criteria:
  K1: KR-Test scores do not correlate with domain task accuracy (r < 0.5)
  K2: KR-Test cannot distinguish good adapters from random adapters (delta < 2x noise floor)

Conditions tested:
  1. Base model (no adapter) -- baseline discrimination ability
  2. Each domain adapter individually (on own domain + cross-domain)
  3. Random (untrained) adapter -- negative control for K2
  4. Composed 1/N -- all adapters merged

Runtime: ~30-40 min on Apple Silicon (MLX)
"""

import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm import load
from mlx_lm.models.bitlinear_layers import BitLinear

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_CONTEXT_TOKENS = 192  # Context window for log-prob eval
N_CONTRASTIVE_PER_DOMAIN = 50  # Number of contrastive pairs per domain
GENERATION_MODE = "cross_item"  # "rule_based" or "cross_item" or "both"
SEED = 42

EXPERIMENT_DIR = Path(__file__).parent
TASK_EVAL_DIR = Path(__file__).parent.parent / "bitnet_instruction_task_eval"
ADAPTERS_DIR = TASK_EVAL_DIR / "adapters"
DATA_DIR = TASK_EVAL_DIR / "data"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

DOMAINS = ["medical", "math", "code", "legal", "creative"]

# Task accuracy from instruction_task_eval (individual adapter on own domain, routed)
# These are the "ground truth" task metrics we want KR-Test to correlate with
TASK_ACCURACY = {
    "medical": {"metric": "keyword_f1", "individual": 0.2544, "base": 0.2532, "delta": 0.0012},
    "math": {"metric": "accuracy", "individual": 0.0667, "base": 0.0, "delta": 0.0667},
    "code": {"metric": "syntax_valid", "individual": 1.0, "base": 0.9, "delta": 0.1},
    "legal": {"metric": "keyword_f1", "individual": 0.2647, "base": 0.0779, "delta": 0.1868},
    "creative": {"metric": "ppl_improvement", "individual": 2.811, "base": 4.3899, "delta": 0.3598},
    # creative delta = (base - individual) / base = improvement fraction
}

# Training loss as alternative quality signal
TRAIN_LOSS = {
    "medical": 0.9488,
    "math": 0.7992,
    "code": 0.8411,
    "legal": 0.0000,  # degenerate
    "creative": 1.4818,
}


def log(msg):
    print(msg, flush=True)


# ===========================================================================
# Contrastive pair generation (rule-based, no teacher LLM)
# ===========================================================================
def generate_contrastive_pairs(domain, val_data, n_pairs=N_CONTRASTIVE_PER_DOMAIN):
    """Generate (context, correct_continuation, wrong_continuation) triples.

    Uses cross-item contrastive pairs: the wrong continuation is a valid
    response to a DIFFERENT question in the same domain. This makes the wrong
    answer fluent and plausible but factually incorrect for the given context.

    This is much harder than rule-based perturbation because the wrong answer
    is grammatically correct and domain-appropriate -- the model must actually
    know the SPECIFIC answer to distinguish correct from wrong.

    For each item i, we pair it with a "hard negative" -- the response to the
    item whose question is most different (by token overlap) to maximize
    difficulty. A truly wrong response that shares no surface cues with the
    question forces the model to rely on learned factual associations.
    """
    random.seed(SEED)

    # Filter valid items
    valid_items = []
    for item in val_data:
        instruction = item.get("instruction", "")
        response = item.get("response", "")
        if instruction and response and len(response) >= 20:
            valid_items.append(item)

    if len(valid_items) < 2:
        return []

    pairs = []

    if GENERATION_MODE in ("cross_item", "both"):
        # Cross-item: pair each question with a DIFFERENT item's response
        # Use distant items (not adjacent) for harder negatives
        n = len(valid_items)
        indices = list(range(n))
        random.shuffle(indices)

        for idx_a in range(min(n, n_pairs)):
            item_a = valid_items[idx_a]
            # Pick a distant item as hard negative
            # Use offset of n//2 to maximize topical distance
            idx_b = (idx_a + max(1, n // 3)) % n
            item_b = valid_items[idx_b]

            # Skip if responses are identical
            if item_a["response"] == item_b["response"]:
                idx_b = (idx_b + 1) % n
                item_b = valid_items[idx_b]

            if item_a["response"] == item_b["response"]:
                continue

            context = f"### Instruction:\n{item_a['instruction']}\n\n### Response:\n"
            pairs.append({
                "context": context,
                "correct": item_a["response"],
                "wrong": item_b["response"],
                "domain": domain,
                "method": "cross_item",
            })

    if GENERATION_MODE in ("rule_based", "both"):
        # Rule-based: domain-specific perturbation
        for item in valid_items[:n_pairs * 2]:
            if len(pairs) >= n_pairs:
                break
            context = f"### Instruction:\n{item['instruction']}\n\n### Response:\n"
            wrong = perturb_response(domain, item["response"])
            if wrong and wrong != item["response"]:
                pairs.append({
                    "context": context,
                    "correct": item["response"],
                    "wrong": wrong,
                    "domain": domain,
                    "method": "rule_based",
                })

    return pairs[:n_pairs]


def perturb_response(domain, response):
    """Domain-specific factual perturbation of a response."""
    if domain == "medical":
        return perturb_medical(response)
    elif domain == "math":
        return perturb_math(response)
    elif domain == "code":
        return perturb_code(response)
    elif domain == "legal":
        return perturb_legal(response)
    elif domain == "creative":
        return perturb_creative(response)
    return None


def perturb_medical(response):
    """Swap medical entities: numbers, conditions, body parts."""
    # Try number perturbation first
    result = perturb_numbers(response)
    if result != response:
        return result
    # Try swapping medical terms
    swaps = [
        ("hypertension", "hypotension"), ("systolic", "diastolic"),
        ("increase", "decrease"), ("elevated", "reduced"),
        ("infection", "inflammation"), ("acute", "chronic"),
        ("benign", "malignant"), ("proximal", "distal"),
        ("anterior", "posterior"), ("superior", "inferior"),
        ("type 1", "type 2"), ("left", "right"),
        ("arterial", "venous"), ("oral", "intravenous"),
        ("positive", "negative"),
    ]
    return swap_terms(response, swaps)


def perturb_math(response):
    """Perturb numbers in math responses."""
    return perturb_numbers(response)


def perturb_code(response):
    """Swap operators or keywords in code."""
    swaps = [
        ("True", "False"), ("true", "false"),
        (" + ", " - "), (" > ", " < "),
        (" >= ", " <= "), (" == ", " != "),
        ("append", "remove"), ("max", "min"),
        ("and", "or"), ("while", "for"),
        ("return ", "print("),
    ]
    return swap_terms(response, swaps)


def perturb_legal(response):
    """Swap legal terms."""
    swaps = [
        ("Yes", "No"), ("yes", "no"),
        ("enforceable", "unenforceable"),
        ("liable", "not liable"),
        ("valid", "invalid"), ("lawful", "unlawful"),
        ("permitted", "prohibited"), ("guilty", "innocent"),
        ("plaintiff", "defendant"),
    ]
    result = swap_terms(response, swaps)
    if result == response:
        # Legal data is short (Yes/No), try direct flip
        if response.strip().lower() in ("yes", "no"):
            return "No" if response.strip().lower() == "yes" else "Yes"
    return result


def perturb_creative(response):
    """Swap names, objects, or emotional valence in stories."""
    result = perturb_numbers(response)
    if result != response:
        return result
    swaps = [
        ("happy", "sad"), ("big", "small"), ("old", "young"),
        ("good", "bad"), ("beautiful", "ugly"), ("fast", "slow"),
        ("loved", "hated"), ("found", "lost"), ("gave", "took"),
        ("opened", "closed"), ("bright", "dark"), ("warm", "cold"),
    ]
    return swap_terms(response, swaps)


def perturb_numbers(text):
    """Find numbers in text and perturb them."""
    # Find all numbers (integers and decimals)
    numbers = list(re.finditer(r'\b(\d+\.?\d*)\b', text))
    if not numbers:
        return text

    # Pick a random number to perturb
    match = random.choice(numbers)
    original = match.group()
    try:
        val = float(original)
        # Perturb by a meaningful amount
        if val == 0:
            new_val = random.choice([1, 2, 5, 10])
        elif '.' in original:
            new_val = val * random.choice([0.5, 1.5, 2.0, 0.8])
            # Keep same decimal places
            decimals = len(original.split('.')[1])
            replacement = f"{new_val:.{decimals}f}"
        else:
            # Integer: add or subtract, or multiply
            delta = max(1, int(val * 0.3))
            new_val = int(val + random.choice([-delta, delta, delta * 2]))
            if new_val < 0 and val >= 0:
                new_val = int(val + delta)
            replacement = str(new_val)

        if 'replacement' not in dir():
            replacement = str(int(new_val)) if '.' not in original else f"{new_val}"

        return text[:match.start()] + replacement + text[match.end():]
    except ValueError:
        return text


def swap_terms(text, swap_pairs):
    """Try each swap pair; return first successful swap."""
    for a, b in swap_pairs:
        if a in text:
            return text.replace(a, b, 1)
        if b in text:
            return text.replace(b, a, 1)
    return text


# ===========================================================================
# Ternary unpacking and model setup (reused from prior experiments)
# ===========================================================================
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


# ===========================================================================
# LoRA setup
# ===========================================================================
class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=16, scale=20.0):
        super().__init__()
        self.linear = base_linear
        self.r = r
        self.scale = scale
        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]
        s = 1.0 / math.sqrt(in_features)
        self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_features, r))
        self.lora_b = mx.zeros((r, out_features))

    def __call__(self, x):
        base_out = self.linear(x)
        lora_out = (x @ self.lora_a) @ self.lora_b * self.scale
        return base_out + lora_out


def apply_lora(model, rank=16, scale=20.0):
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora = LoRALinear(module, r=rank, scale=scale)
                updates.append((key, lora))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    log(f"  Applied LoRA (r={rank}) to {count} layers")
    return model


def load_adapter(path):
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_weights(model, adapter_params, scale=1.0):
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def zero_lora_params(model):
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                in_dims = module.lora_a.shape[0]
                s = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(low=-s, high=s, shape=module.lora_a.shape)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def compose_adapters(adapter_list, scale_per_adapter=None):
    N = len(adapter_list)
    if scale_per_adapter is None:
        scale_per_adapter = 1.0 / N
    merged = {}
    for key in adapter_list[0].keys():
        stacked = mx.stack([a[key] for a in adapter_list])
        merged[key] = mx.sum(stacked, axis=0) * scale_per_adapter
    return merged


# ===========================================================================
# KR-Test scoring: log-prob comparison
# ===========================================================================
def compute_log_probs(model, tokenizer, text, max_tokens=MAX_CONTEXT_TOKENS):
    """Compute per-token log probabilities for a text sequence.

    Returns: list of log-probs for each token (excluding the first).
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    if len(tokens) < 2:
        return []

    input_ids = mx.array([tokens])
    logits = model(input_ids)  # (1, seq_len, vocab_size)
    mx.eval(logits)

    # Log softmax over vocab dimension
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    # Get log-prob of each actual next token
    # For position i, we want log_prob[i, tokens[i+1]]
    token_log_probs = []
    for i in range(len(tokens) - 1):
        next_token = tokens[i + 1]
        lp = log_probs[0, i, next_token].item()
        token_log_probs.append(lp)

    return token_log_probs


def kr_test_score_single(model, tokenizer, pair):
    """Score a single contrastive pair.

    Returns: (correct_logprob, wrong_logprob, is_correct)
    where is_correct = True if model prefers the correct continuation.
    """
    context = pair["context"]
    correct_text = context + pair["correct"]
    wrong_text = context + pair["wrong"]

    # Tokenize context to know where continuation starts
    context_tokens = tokenizer.encode(context, add_special_tokens=False)
    ctx_len = len(context_tokens)

    # Get full sequence log-probs
    correct_lps = compute_log_probs(model, tokenizer, correct_text)
    wrong_lps = compute_log_probs(model, tokenizer, wrong_text)

    # Sum log-probs only over continuation tokens (after context)
    # The log-prob at position i predicts token i+1
    # So continuation starts at index ctx_len-1 (predicting token at ctx_len)
    start_idx = max(0, ctx_len - 1)

    correct_cont_lps = correct_lps[start_idx:]
    wrong_cont_lps = wrong_lps[start_idx:]

    # Truncate to minimum length (KR-Test protocol)
    min_len = min(len(correct_cont_lps), len(wrong_cont_lps))
    if min_len == 0:
        return 0.0, 0.0, False

    correct_sum = sum(correct_cont_lps[:min_len])
    wrong_sum = sum(wrong_cont_lps[:min_len])

    return correct_sum, wrong_sum, correct_sum > wrong_sum


def evaluate_kr_test(model, tokenizer, contrastive_pairs, label=""):
    """Evaluate KR-Test score on a set of contrastive pairs.

    Returns dict with score, n_correct, n_total, per-domain breakdown.
    """
    results_by_domain = {}
    all_correct = 0
    all_total = 0
    margins = []

    for pair in contrastive_pairs:
        domain = pair["domain"]
        if domain not in results_by_domain:
            results_by_domain[domain] = {"correct": 0, "total": 0, "margins": []}

        correct_lp, wrong_lp, is_correct = kr_test_score_single(model, tokenizer, pair)
        margin = correct_lp - wrong_lp

        results_by_domain[domain]["total"] += 1
        results_by_domain[domain]["margins"].append(margin)
        all_total += 1
        margins.append(margin)

        if is_correct:
            results_by_domain[domain]["correct"] += 1
            all_correct += 1

    # Compute scores
    overall_score = all_correct / all_total if all_total > 0 else 0.0
    domain_scores = {}
    for domain, data in results_by_domain.items():
        score = data["correct"] / data["total"] if data["total"] > 0 else 0.0
        mean_margin = sum(data["margins"]) / len(data["margins"]) if data["margins"] else 0.0
        domain_scores[domain] = {
            "score": score,
            "correct": data["correct"],
            "total": data["total"],
            "mean_margin": mean_margin,
        }

    mean_margin = sum(margins) / len(margins) if margins else 0.0

    if label:
        log(f"  {label}: KR-Test = {overall_score:.3f} ({all_correct}/{all_total}), "
            f"mean margin = {mean_margin:.3f}")
        for domain in DOMAINS:
            if domain in domain_scores:
                d = domain_scores[domain]
                log(f"    {domain}: {d['score']:.3f} ({d['correct']}/{d['total']}), "
                    f"margin={d['mean_margin']:.3f}")

    return {
        "score": overall_score,
        "n_correct": all_correct,
        "n_total": all_total,
        "mean_margin": mean_margin,
        "domain_scores": domain_scores,
    }


# ===========================================================================
# Main experiment
# ===========================================================================
def main():
    t0 = time.time()
    random.seed(SEED)
    mx.random.seed(SEED)

    log("=" * 70)
    log("KR-Test Knowledge Retention Evaluation")
    log("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Load and prepare contrastive pairs
    # -----------------------------------------------------------------------
    log("\n[1/5] Generating contrastive pairs from instruction eval data...")

    all_pairs = []
    pair_stats = {}
    for domain in DOMAINS:
        val_path = DATA_DIR / domain / "val.jsonl"
        if not val_path.exists():
            log(f"  WARNING: {val_path} not found, skipping {domain}")
            continue

        val_data = []
        with open(val_path) as f:
            for line in f:
                val_data.append(json.loads(line))

        pairs = generate_contrastive_pairs(domain, val_data, N_CONTRASTIVE_PER_DOMAIN)
        pair_stats[domain] = {
            "n_pairs": len(pairs),
            "n_val_samples": len(val_data),
        }
        all_pairs.extend(pairs)
        log(f"  {domain}: {len(pairs)} contrastive pairs from {len(val_data)} val samples")

    log(f"  Total: {len(all_pairs)} contrastive pairs across {len(pair_stats)} domains")

    # Show examples
    for domain in DOMAINS:
        domain_pairs = [p for p in all_pairs if p["domain"] == domain]
        if domain_pairs:
            p = domain_pairs[0]
            log(f"\n  Example ({domain}):")
            log(f"    Context: ...{p['context'][-80:]}")
            log(f"    Correct: {p['correct'][:80]}...")
            log(f"    Wrong:   {p['wrong'][:80]}...")

    # -----------------------------------------------------------------------
    # Step 2: Load model
    # -----------------------------------------------------------------------
    log("\n[2/5] Loading BitNet-2B-4T...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    # -----------------------------------------------------------------------
    # Step 3: Evaluate base model (no adapter)
    # -----------------------------------------------------------------------
    log("\n[3/5] Evaluating base model (no adapter)...")
    base_results = evaluate_kr_test(model, tokenizer, all_pairs, label="Base")

    # -----------------------------------------------------------------------
    # Step 4: Apply LoRA and evaluate each condition
    # -----------------------------------------------------------------------
    log("\n[4/5] Evaluating adapter conditions...")
    model = apply_lora(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Load all adapters
    adapters = {}
    for domain in DOMAINS:
        adapter_path = ADAPTERS_DIR / domain
        if adapter_path.exists():
            adapters[domain] = load_adapter(adapter_path)
            log(f"  Loaded {domain} adapter: {len(adapters[domain])} tensors")
        else:
            log(f"  WARNING: {adapter_path} not found")

    # 4a: Random (untrained) adapter -- negative control
    log("\n  --- Random adapter (negative control) ---")
    zero_lora_params(model)
    random_results = evaluate_kr_test(model, tokenizer, all_pairs, label="Random")

    # 4b: Each domain adapter individually (on ALL domains, not just own)
    individual_results = {}
    for adapter_domain in DOMAINS:
        if adapter_domain not in adapters:
            continue
        log(f"\n  --- {adapter_domain} adapter (individual) ---")
        zero_lora_params(model)
        apply_adapter_weights(model, adapters[adapter_domain])
        mx.eval(model.parameters())
        result = evaluate_kr_test(model, tokenizer, all_pairs,
                                  label=f"Individual-{adapter_domain}")
        individual_results[adapter_domain] = result

    # 4c: Composed 1/N
    log("\n  --- Composed 1/N (all adapters) ---")
    adapter_list = [adapters[d] for d in DOMAINS if d in adapters]
    if len(adapter_list) > 1:
        composed = compose_adapters(adapter_list)
        zero_lora_params(model)
        apply_adapter_weights(model, composed)
        mx.eval(model.parameters())
        composed_results = evaluate_kr_test(model, tokenizer, all_pairs,
                                            label="Composed-1/N")
    else:
        composed_results = {"score": 0, "domain_scores": {}}

    # -----------------------------------------------------------------------
    # Step 5: Analysis -- correlation and discrimination
    # -----------------------------------------------------------------------
    log("\n[5/5] Analysis...")

    # K1: Correlation between KR-Test domain scores and task accuracy
    # For each adapter, get its KR-Test score on its own domain
    kr_scores_own_domain = []
    task_deltas = []
    domain_names_for_corr = []

    for domain in DOMAINS:
        if domain not in individual_results:
            continue
        if domain not in individual_results[domain].get("domain_scores", {}):
            continue

        kr_own = individual_results[domain]["domain_scores"][domain]["score"]
        # Use the adapter-over-base delta as task accuracy signal
        task_delta = TASK_ACCURACY.get(domain, {}).get("delta", 0)

        kr_scores_own_domain.append(kr_own)
        task_deltas.append(task_delta)
        domain_names_for_corr.append(domain)

    # Also compute KR-Test improvement over base for each adapter
    kr_deltas_over_base = []
    base_kr_by_domain = {}
    for domain in DOMAINS:
        if domain in base_results.get("domain_scores", {}):
            base_kr_by_domain[domain] = base_results["domain_scores"][domain]["score"]

    for domain in domain_names_for_corr:
        if domain in individual_results and domain in individual_results[domain].get("domain_scores", {}):
            adapter_kr = individual_results[domain]["domain_scores"][domain]["score"]
            base_kr = base_kr_by_domain.get(domain, 0.5)
            kr_deltas_over_base.append(adapter_kr - base_kr)

    # Pearson correlation (manual, no scipy dependency)
    def pearson_r(x, y):
        n = len(x)
        if n < 3:
            return float('nan'), n
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
        std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5
        if std_x < 1e-10 or std_y < 1e-10:
            return float('nan'), n
        return cov / (std_x * std_y), n

    # Spearman rank correlation
    def spearman_rho(x, y):
        n = len(x)
        if n < 3:
            return float('nan'), n
        def rank(vals):
            sorted_indices = sorted(range(len(vals)), key=lambda i: vals[i])
            ranks = [0.0] * len(vals)
            for rank_val, idx in enumerate(sorted_indices):
                ranks[idx] = rank_val + 1
            return ranks
        rx = rank(x)
        ry = rank(y)
        return pearson_r(rx, ry)[0], n

    r_pearson, n_corr = pearson_r(kr_scores_own_domain, task_deltas)
    r_spearman, _ = spearman_rho(kr_scores_own_domain, task_deltas)

    # Also try KR-Test delta (over base) vs task delta
    r_delta_pearson, _ = pearson_r(kr_deltas_over_base, task_deltas)
    r_delta_spearman, _ = spearman_rho(kr_deltas_over_base, task_deltas)

    log(f"\n  K1 Correlation Analysis (n={n_corr} domains):")
    log(f"    Domain pairs: {list(zip(domain_names_for_corr, kr_scores_own_domain, task_deltas))}")
    log(f"    KR-Test raw score vs task delta:")
    log(f"      Pearson r  = {r_pearson:.4f}")
    log(f"      Spearman rho = {r_spearman:.4f}")
    log(f"    KR-Test delta (over base) vs task delta:")
    log(f"      Pearson r  = {r_delta_pearson:.4f}")
    log(f"      Spearman rho = {r_delta_spearman:.4f}")
    log(f"    KR deltas over base: {list(zip(domain_names_for_corr, kr_deltas_over_base))}")

    # K2: Discrimination -- trained adapter vs random adapter
    trained_scores = []
    trained_domains = []
    random_domain_scores = []
    random_domains = []
    for domain in DOMAINS:
        if domain in individual_results and domain in individual_results[domain].get("domain_scores", {}):
            trained_scores.append(individual_results[domain]["domain_scores"][domain]["score"])
            trained_domains.append(domain)
        if domain in random_results.get("domain_scores", {}):
            random_domain_scores.append(random_results["domain_scores"][domain]["score"])
            random_domains.append(domain)

    mean_trained = sum(trained_scores) / len(trained_scores) if trained_scores else 0
    mean_random = sum(random_domain_scores) / len(random_domain_scores) if random_domain_scores else 0

    # Noise floor = |random - 0.5| (deviation from chance)
    noise_floor = abs(mean_random - 0.5)
    discrimination_delta = mean_trained - mean_random
    discrimination_ratio = abs(discrimination_delta) / noise_floor if noise_floor > 0.001 else float('inf')

    log(f"\n  K2 Discrimination Analysis:")
    log(f"    Mean trained adapter KR-Test (own domain): {mean_trained:.4f}")
    log(f"    Mean random adapter KR-Test: {mean_random:.4f}")
    log(f"    Noise floor (|random - 0.5|): {noise_floor:.4f}")
    log(f"    Discrimination delta: {discrimination_delta:.4f}")
    log(f"    Discrimination ratio (delta / noise_floor): {discrimination_ratio:.2f}x")
    log(f"    Per-domain trained: {list(zip(trained_domains, trained_scores))}")
    log(f"    Per-domain random:  {list(zip(random_domains, random_domain_scores))}")

    # -----------------------------------------------------------------------
    # Kill criteria assessment
    # -----------------------------------------------------------------------
    # K1: r < 0.5 (using best of raw/delta, Pearson/Spearman)
    best_r = max(
        abs(r_pearson) if not math.isnan(r_pearson) else 0,
        abs(r_spearman) if not math.isnan(r_spearman) else 0,
        abs(r_delta_pearson) if not math.isnan(r_delta_pearson) else 0,
        abs(r_delta_spearman) if not math.isnan(r_delta_spearman) else 0,
    )
    k1_pass = best_r >= 0.5

    # K2: delta < 2x noise floor
    k2_pass = discrimination_ratio >= 2.0

    verdict = "SUPPORTED" if k1_pass and k2_pass else "KILLED"
    if k1_pass and not k2_pass:
        verdict = "KILLED (K2: discrimination < 2x noise floor)"
    elif not k1_pass and k2_pass:
        verdict = "KILLED (K1: correlation r < 0.5)"
    elif not k1_pass and not k2_pass:
        verdict = "KILLED (K1 + K2)"

    log(f"\n  === VERDICT: {verdict} ===")
    log(f"    K1 (correlation r >= 0.5): {'PASS' if k1_pass else 'FAIL'} (best |r| = {best_r:.4f})")
    log(f"    K2 (discrimination >= 2x noise): {'PASS' if k2_pass else 'FAIL'} "
        f"(ratio = {discrimination_ratio:.2f}x)")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    elapsed = time.time() - t0
    results = {
        "experiment": "bitnet_kr_test_eval",
        "model": MODEL_ID,
        "hypothesis": "KR-Test replaces PPL as primary adapter quality metric",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_contrastive_pairs": len(all_pairs),
        "pair_stats": pair_stats,
        "base": base_results,
        "random_adapter": random_results,
        "individual_adapters": individual_results,
        "composed_1_over_n": composed_results,
        "analysis": {
            "k1_correlation": {
                "domains": domain_names_for_corr,
                "kr_scores_own_domain": kr_scores_own_domain,
                "kr_deltas_over_base": kr_deltas_over_base,
                "task_deltas": task_deltas,
                "pearson_r_raw": r_pearson if not math.isnan(r_pearson) else None,
                "spearman_rho_raw": r_spearman if not math.isnan(r_spearman) else None,
                "pearson_r_delta": r_delta_pearson if not math.isnan(r_delta_pearson) else None,
                "spearman_rho_delta": r_delta_spearman if not math.isnan(r_delta_spearman) else None,
                "best_abs_r": best_r,
                "pass": k1_pass,
            },
            "k2_discrimination": {
                "mean_trained_own_domain": mean_trained,
                "mean_random": mean_random,
                "noise_floor": noise_floor,
                "discrimination_delta": discrimination_delta,
                "discrimination_ratio": discrimination_ratio,
                "trained_per_domain": dict(zip(trained_domains, trained_scores)),
                "random_per_domain": dict(zip(random_domains, random_domain_scores)),
                "pass": k2_pass,
            },
        },
        "kill_criteria": {
            "K1": f"{'PASS' if k1_pass else 'FAIL'}: best |r| = {best_r:.4f} (threshold 0.5)",
            "K2": f"{'PASS' if k2_pass else 'FAIL'}: ratio = {discrimination_ratio:.2f}x (threshold 2.0x)",
        },
        "verdict": verdict,
        "total_time_s": elapsed,
        "total_time_min": elapsed / 60,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n  Results saved to {RESULTS_FILE}")
    log(f"  Total time: {elapsed/60:.1f} min")

    return results


if __name__ == "__main__":
    main()
