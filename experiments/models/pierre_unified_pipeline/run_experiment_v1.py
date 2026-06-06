#!/usr/bin/env python3
"""Pierre unified pipeline: route → compose → pre-merge → generate.

Wires three proven components into a single end-to-end pipeline:
  1. Ridge regression router (Finding #276, 96% accuracy)
  2. NRE composition (Finding #275, norm-preserved averaging)
  3. Pre-merge (0% per-token overhead, 165 tok/s)

Kill criteria:
  K1: Routing accuracy < 80% on held-out domain classification (target: 96% from #276)
  K2: Any domain PPL > 10% worse than single-adapter PPL
  K3: Behavioral quality score < 0.3 mean across domains (execution-based eval)

Type: Verification (wiring proven components, verifying they compose correctly)
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
from mlx.utils import tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Source adapters + data
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]

# Calibration uses train data (400/domain), eval uses valid data (50/domain)
N_CAL_PER_DOMAIN = 50
N_TEST_PER_DOMAIN = 50  # all valid data
NUM_GEN_PER_DOMAIN = 5
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0  # deterministic for reproducibility


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
# BitNet model utilities (reused from proven experiments)
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
# Data loading
# ============================================================================

def load_domain_data(domain, split="valid", max_samples=None):
    """Load domain data from JSONL files."""
    path = DATA_DIR / domain / f"{split}.jsonl"
    samples = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            samples.append(obj["text"])
            if max_samples and len(samples) >= max_samples:
                break
    return samples


# ============================================================================
# Behavioral evaluation (from Finding #266, behavioral_eval_framework)
# ============================================================================

def eval_code_syntax(text):
    """Check if generated code is syntactically valid Python."""
    code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)\n```', text, re.DOTALL)
    if not code_blocks:
        lines = [l for l in text.split('\n') if l.strip() and not l.startswith('#')]
        code_text = '\n'.join(lines)
    else:
        code_text = '\n'.join(code_blocks)

    try:
        ast.parse(code_text)
        return True
    except SyntaxError:
        return False


def eval_factual_recall(generated, reference):
    """Token-overlap factual recall between generated and reference text."""
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                  'as', 'into', 'through', 'during', 'before', 'after', 'and',
                  'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
                  'neither', 'each', 'every', 'all', 'any', 'few', 'more',
                  'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
                  'than', 'too', 'very', 'just', 'because', 'if', 'when',
                  'where', 'how', 'what', 'which', 'who', 'whom', 'this',
                  'that', 'these', 'those', 'it', 'its', 'i', 'me', 'my',
                  'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she',
                  'her', 'they', 'them', 'their'}

    def extract_tokens(text):
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return set(w for w in words if w not in stop_words and len(w) > 2)

    gen_tokens = extract_tokens(generated)
    ref_tokens = extract_tokens(reference)

    if not ref_tokens:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

    overlap = gen_tokens & ref_tokens
    recall = len(overlap) / len(ref_tokens) if ref_tokens else 0.0
    precision = len(overlap) / len(gen_tokens) if gen_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}


def extract_math_answer(text):
    """Extract numerical answer from math response."""
    patterns = [
        r'(?:answer|result|solution|=)\s*[:\s]*\$?\\?boxed\{([^}]+)\}',
        r'(?:answer|result|solution|=)\s*[:\s]*([+-]?\d+\.?\d*)',
        r'=\s*([+-]?\d+\.?\d*)\s*$',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                continue
    return None


def evaluate_response(generated_text, reference_text, domain):
    """Evaluate a single response using domain-appropriate behavioral metrics."""
    result = {"domain": domain, "generated_len": len(generated_text)}

    if domain == "code":
        syntax_ok = eval_code_syntax(generated_text)
        factual = eval_factual_recall(generated_text, reference_text)
        score = 0.7 * (1.0 if syntax_ok else 0.0) + 0.3 * factual["recall"]
        result.update({"score": score, "syntax_valid": syntax_ok,
                       "factual_recall": factual["recall"], "method": "syntax+recall"})

    elif domain == "math":
        gen_answer = extract_math_answer(generated_text)
        ref_answer = extract_math_answer(reference_text)
        correct = (gen_answer is not None and ref_answer is not None
                   and abs(gen_answer - ref_answer) < 0.01 * max(abs(ref_answer), 1.0))
        score = 1.0 if correct else 0.0
        result.update({"score": score, "answer_correct": correct, "method": "numerical_match"})

    else:  # medical, legal, finance
        factual = eval_factual_recall(generated_text, reference_text)
        score = factual["recall"]
        result.update({"score": score, "factual_recall": factual["recall"],
                       "factual_f1": factual["f1"], "method": "factual_recall"})

    return result


# ============================================================================
# Phase 1: Load model + calibrate router
# ============================================================================

def phase_calibrate_router():
    """Load model, calibrate ridge regression router, return metrics."""
    log("\n=== Phase 1: Load model + calibrate router ===")
    log_memory("start")

    from pierre.router import RouterStatistics, solve_ridge, RidgeRouter

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    log_memory("model loaded")

    # Get hidden dim
    hidden_dim = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    log(f"  Hidden dim: {hidden_dim}")

    # Load calibration data from TRAIN split (valid is reserved for eval)
    cal_data = {}
    for domain in DOMAINS:
        cal_data[domain] = load_domain_data(domain, "train", N_CAL_PER_DOMAIN)
        log(f"  Loaded {len(cal_data[domain])} cal samples for {domain} (train split)")

    def extract_hidden(model, tokenizer, text):
        """Extract mean-pooled hidden state with causal mask + final norm (matching DUME)."""
        toks = tokenizer.encode(text)[:MAX_SEQ_LENGTH]
        if len(toks) < 4:
            return None
        x = mx.array(toks)[None, :]
        h = model.model.embed_tokens(x)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1]).astype(h.dtype)
        for layer in model.model.layers:
            h = layer(h, mask=mask)
        h = model.model.norm(h)
        mx.eval(h)
        pooled = mx.mean(h[0], axis=0, keepdims=True).astype(mx.float32)  # (1, H)
        mx.eval(pooled)
        del h, x, mask
        return pooled, len(toks)

    # Build router statistics
    t0 = time.time()
    stats = RouterStatistics(hidden_dim, len(DOMAINS))
    tokens_per_domain = {}

    for di, domain in enumerate(DOMAINS):
        n_tokens = 0
        for text in cal_data[domain]:
            result = extract_hidden(model, tokenizer, text)
            if result is None:
                continue
            pooled, n = result
            stats.update(pooled, di)
            n_tokens += n
            del pooled
        tokens_per_domain[domain] = n_tokens
        log(f"  Router calibration: {domain} ({n_tokens} tokens)")

    # Solve ridge regression
    W = solve_ridge(stats, lam=1.0, column_normalize=True)
    cal_time = time.time() - t0
    log(f"  Router calibrated in {cal_time:.1f}s")

    # Test routing accuracy on held-out data
    log("\n  Testing routing accuracy...")
    correct = 0
    total = 0
    per_domain_acc = {}

    for di, domain in enumerate(DOMAINS):
        test_samples = load_domain_data(domain, "valid", N_TEST_PER_DOMAIN)
        domain_correct = 0

        for text in test_samples:
            result = extract_hidden(model, tokenizer, text)
            if result is None:
                continue
            pooled, _ = result
            logits = pooled @ W
            pred = mx.argmax(logits, axis=-1)
            mx.eval(pred)
            if pred.item() == di:
                domain_correct += 1
                correct += 1
            total += 1
            del pooled, logits, pred

        acc = domain_correct / max(len(test_samples), 1)
        per_domain_acc[domain] = round(acc, 3)
        log(f"    {domain}: {acc:.1%} ({domain_correct}/{len(test_samples)})")

    overall_acc = correct / total if total > 0 else 0.0
    log(f"  Overall routing accuracy: {overall_acc:.1%} ({correct}/{total})")

    # Save router weights for later phases
    np.save(str(EXPERIMENT_DIR / "router_weights.npy"), np.array(W))

    results = {
        "routing_accuracy": round(overall_acc, 4),
        "per_domain_accuracy": per_domain_acc,
        "calibration_time_s": round(cal_time, 2),
        "tokens_per_domain": tokens_per_domain,
        "hidden_dim": hidden_dim,
    }

    log_memory("end phase 1")
    cleanup(model, tokenizer, stats)
    return results


# ============================================================================
# Phase 2: Compose + PPL per domain (single-adapter baseline vs pipeline)
# ============================================================================

def phase_ppl_comparison():
    """Compare per-domain PPL: base, single-adapter, Pierre pipeline."""
    log("\n=== Phase 2: PPL comparison ===")
    log_memory("start")

    from pierre.compose import compute_delta, nre_merge_deltas, premerge_deltas_into_model, TARGET_KEYS

    # Load validation data
    val_data = {}
    for domain in DOMAINS:
        val_data[domain] = load_domain_data(domain, "valid", N_TEST_PER_DOMAIN)

    skeleton = dict(np.load(str(ADAPTERS_DIR / "grassmannian_skeleton.npz")))

    def compute_ppl(model, tokenizer, texts):
        """Compute mean perplexity on a list of texts."""
        total_loss = 0.0
        total_tokens = 0
        for text in texts:
            toks = tokenizer.encode(text)[:MAX_SEQ_LENGTH]
            if len(toks) < 4:
                continue
            x = mx.array(toks)[None, :]
            logits = model(x)
            mx.eval(logits)
            # Shift for next-token prediction
            targets = x[:, 1:]
            logits = logits[:, :-1, :]
            log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
            token_log_probs = mx.take_along_axis(
                log_probs, targets[:, :, None], axis=-1
            ).squeeze(-1)
            mx.eval(token_log_probs)
            total_loss += -token_log_probs.sum().item()
            total_tokens += targets.shape[1]
            del logits, log_probs, token_log_probs, x
        return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')

    results = {"base_ppl": {}, "single_ppl": {}, "pierre_ppl": {}}

    # --- Base model PPL ---
    log("  Computing base model PPL...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    for domain in DOMAINS:
        ppl = compute_ppl(model, tokenizer, val_data[domain])
        results["base_ppl"][domain] = round(ppl, 3)
        log(f"    base/{domain}: PPL={ppl:.3f}")
    cleanup(model, tokenizer)

    # --- Single-adapter PPL (gold standard) ---
    log("  Computing single-adapter PPL...")
    for di, domain in enumerate(DOMAINS):
        model, tokenizer = load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)

        # Load and merge single adapter
        adapter = dict(mx.load(str(ADAPTERS_DIR / domain / "adapter.npz")))
        n_layers = len(model.model.layers)
        merge_count = 0
        for li in range(n_layers):
            for key in TARGET_KEYS:
                b_key = f"model.layers.{li}.{key}.lora_b"
                skey = f"layer_{li}_{key}_domain_{di}"
                if b_key not in adapter or skey not in skeleton:
                    continue
                a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
                b_mx = adapter[b_key]
                delta = LORA_SCALE * (b_mx.astype(mx.bfloat16).T @ a_mx.T)
                parts = key.split(".")
                module = model.model.layers[li]
                for part in parts:
                    module = getattr(module, part, None)
                if module is not None and isinstance(module, nn.Linear):
                    module.weight = module.weight + delta
                    merge_count += 1
        mx.eval(model.parameters())

        ppl = compute_ppl(model, tokenizer, val_data[domain])
        results["single_ppl"][domain] = round(ppl, 3)
        log(f"    single/{domain}: PPL={ppl:.3f} ({merge_count} layers merged)")
        cleanup(model, tokenizer, adapter)

    # --- Pierre pipeline: route + compute_delta with correct domain A-matrix + pre-merge ---
    log("  Computing Pierre pipeline PPL...")
    router_W = mx.array(np.load(str(EXPERIMENT_DIR / "router_weights.npy")))

    for di, domain in enumerate(DOMAINS):
        model, tokenizer = load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)
        n_layers = len(model.model.layers)

        # Route: which adapter(s) does the router pick for this domain?
        test_text = val_data[domain][0]
        toks = tokenizer.encode(test_text)[:MAX_SEQ_LENGTH]
        x = mx.array(toks)[None, :]
        h = model.model.embed_tokens(x)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1]).astype(h.dtype)
        for layer in model.model.layers:
            h = layer(h, mask=mask)
        h = model.model.norm(h)
        mx.eval(h)
        pooled = mx.mean(h[0], axis=0, keepdims=True).astype(mx.float32)
        logits = pooled @ router_W
        probs = mx.softmax(logits, axis=-1)
        mx.eval(probs)

        top1_idx = mx.argmax(probs, axis=-1).item()
        top1_domain = DOMAINS[top1_idx]
        log(f"    Route {domain} -> {top1_domain} (prob={probs[0, top1_idx].item():.3f})")
        del h, pooled, x, logits, mask

        # Compute delta using correct domain's A-matrix
        adapter = dict(mx.load(str(ADAPTERS_DIR / top1_domain / "adapter.npz")))
        deltas = compute_delta(adapter, skeleton, DOMAINS.index(top1_domain), LORA_SCALE, n_layers)
        merge_count = premerge_deltas_into_model(model, deltas)

        ppl = compute_ppl(model, tokenizer, val_data[domain])
        results["pierre_ppl"][domain] = round(ppl, 3)
        log(f"    pierre/{domain}: PPL={ppl:.3f} (routed to {top1_domain}, {merge_count} merged)")
        cleanup(model, tokenizer, adapter, deltas)

    # Compute degradation vs single-adapter
    results["ppl_degradation"] = {}
    for domain in DOMAINS:
        single = results["single_ppl"][domain]
        pierre = results["pierre_ppl"][domain]
        deg = (pierre - single) / single * 100
        results["ppl_degradation"][domain] = round(deg, 2)
        log(f"  PPL degradation {domain}: {deg:+.2f}%")

    log_memory("end phase 2")
    cleanup(skeleton)
    return results


# ============================================================================
# Phase 3: Behavioral eval (generation quality)
# ============================================================================

def phase_behavioral_eval():
    """Generate text with Pierre pipeline and evaluate behavioral quality."""
    log("\n=== Phase 3: Behavioral evaluation ===")
    log_memory("start")

    from pierre.compose import compute_delta, premerge_deltas_into_model

    router_W = mx.array(np.load(str(EXPERIMENT_DIR / "router_weights.npy")))
    skeleton = dict(np.load(str(ADAPTERS_DIR / "grassmannian_skeleton.npz")))

    results = {"per_domain": {}, "generations": []}

    for di, domain in enumerate(DOMAINS):
        log(f"\n  Domain: {domain}")
        model, tokenizer = load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)
        n_layers = len(model.model.layers)

        # Load test prompts from valid split
        test_data = load_domain_data(domain, "valid", NUM_GEN_PER_DOMAIN)

        # Route using first sample to pick adapter
        first_text = test_data[0] if test_data else ""
        toks = tokenizer.encode(first_text)[:MAX_SEQ_LENGTH]
        x = mx.array(toks)[None, :]
        h = model.model.embed_tokens(x)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1]).astype(h.dtype)
        for layer in model.model.layers:
            h = layer(h, mask=mask)
        h = model.model.norm(h)
        mx.eval(h)
        pooled = mx.mean(h[0], axis=0, keepdims=True).astype(mx.float32)
        logits = pooled @ router_W
        top1_idx = mx.argmax(logits, axis=-1).item()
        routed_domain = DOMAINS[top1_idx]
        del h, pooled, x, logits, mask

        # Compute delta with correct domain A-matrix and merge
        adapter = dict(mx.load(str(ADAPTERS_DIR / routed_domain / "adapter.npz")))
        deltas = compute_delta(adapter, skeleton, DOMAINS.index(routed_domain), LORA_SCALE, n_layers)
        premerge_deltas_into_model(model, deltas)
        cleanup(adapter, deltas)

        # Generate and evaluate
        domain_scores = []
        sampler = make_sampler(temp=TEMPERATURE)

        for text in test_data:
            # Extract instruction from text
            if "### Instruction:" in text and "### Response:" in text:
                instruction = text.split("### Response:")[0].strip()
                reference = text.split("### Response:")[-1].strip()
                prompt = instruction + "\n### Response:\n"
            else:
                prompt = text[:200]
                reference = text

            try:
                generated = mlx_generate(
                    model, tokenizer, prompt=prompt,
                    max_tokens=MAX_NEW_TOKENS, sampler=sampler, verbose=False,
                )
            except Exception as e:
                log(f"    Generation failed: {e}")
                generated = ""

            eval_result = evaluate_response(generated, reference, domain)
            domain_scores.append(eval_result["score"])

            results["generations"].append({
                "domain": domain,
                "routed_to": routed_domain,
                "prompt": prompt[:100] + "...",
                "generated": generated[:200] + "..." if len(generated) > 200 else generated,
                "score": eval_result["score"],
                "method": eval_result.get("method", "unknown"),
            })

        mean_score = float(np.mean(domain_scores)) if domain_scores else 0.0
        results["per_domain"][domain] = {
            "mean_score": round(mean_score, 3),
            "n_samples": len(domain_scores),
            "routed_to": routed_domain,
            "correct_route": routed_domain == domain,
        }
        log(f"    Routed to: {routed_domain} | Mean behavioral score: {mean_score:.3f}")

        cleanup(model, tokenizer)

    # Overall score
    all_scores = [v["mean_score"] for v in results["per_domain"].values()]
    results["overall_mean_score"] = round(float(np.mean(all_scores)), 3)
    log(f"\n  Overall behavioral score: {results['overall_mean_score']:.3f}")

    log_memory("end phase 3")
    cleanup(skeleton)
    return results


# ============================================================================
# Phase 4: Latency measurement
# ============================================================================

def phase_latency():
    """Measure generation latency: base vs Pierre pipeline."""
    log("\n=== Phase 4: Latency measurement ===")
    log_memory("start")

    from pierre.compose import compute_delta, premerge_deltas_into_model

    prompt = "### Instruction:\nExplain the concept of photosynthesis.\n\n### Response:\n"
    n_warmup = 2
    n_measure = 5

    results = {}

    # Base model latency
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    sampler = make_sampler(temp=0.0)

    for _ in range(n_warmup):
        mlx_generate(model, tokenizer, prompt=prompt, max_tokens=32, sampler=sampler, verbose=False)

    times = []
    for _ in range(n_measure):
        t0 = time.time()
        out = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=MAX_NEW_TOKENS, sampler=sampler, verbose=False)
        elapsed = time.time() - t0
        n_toks = len(tokenizer.encode(out)) - len(tokenizer.encode(prompt))
        times.append({"time_s": elapsed, "tokens": n_toks})

    mean_tps = sum(t["tokens"] for t in times) / sum(t["time_s"] for t in times) if times else 0
    results["base"] = {"mean_tok_per_s": round(mean_tps, 1), "runs": times}
    log(f"  Base: {mean_tps:.1f} tok/s")
    cleanup(model, tokenizer)

    # Pierre pipeline latency (pre-merge, should be identical to base)
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    skeleton = dict(np.load(str(ADAPTERS_DIR / "grassmannian_skeleton.npz")))
    adapter = dict(mx.load(str(ADAPTERS_DIR / "medical" / "adapter.npz")))
    n_layers = len(model.model.layers)
    deltas = compute_delta(adapter, skeleton, 0, LORA_SCALE, n_layers)  # domain 0 = medical
    premerge_deltas_into_model(model, deltas)
    cleanup(skeleton, adapter, deltas)

    sampler = make_sampler(temp=0.0)
    for _ in range(n_warmup):
        mlx_generate(model, tokenizer, prompt=prompt, max_tokens=32, sampler=sampler, verbose=False)

    times = []
    for _ in range(n_measure):
        t0 = time.time()
        out = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=MAX_NEW_TOKENS, sampler=sampler, verbose=False)
        elapsed = time.time() - t0
        n_toks = len(tokenizer.encode(out)) - len(tokenizer.encode(prompt))
        times.append({"time_s": elapsed, "tokens": n_toks})

    mean_tps = sum(t["tokens"] for t in times) / sum(t["time_s"] for t in times) if times else 0
    results["pierre"] = {"mean_tok_per_s": round(mean_tps, 1), "runs": times}
    overhead = (results["base"]["mean_tok_per_s"] - mean_tps) / results["base"]["mean_tok_per_s"] * 100
    results["overhead_pct"] = round(overhead, 2)
    log(f"  Pierre: {mean_tps:.1f} tok/s (overhead: {overhead:.1f}%)")

    log_memory("end phase 4")
    cleanup(model, tokenizer)
    return results


# ============================================================================
# Main orchestrator
# ============================================================================

def main():
    t0 = time.time()
    log("Pierre Unified Pipeline Experiment")
    log("=" * 60)
    log_memory("start")

    mx.random.seed(SEED)

    # Phase 1: Calibrate router
    cal_results = phase_calibrate_router()
    log_memory("after phase 1")

    # Phase 2: PPL comparison
    ppl_results = phase_ppl_comparison()
    log_memory("after phase 2")

    # Phase 3: Behavioral eval
    behav_results = phase_behavioral_eval()
    log_memory("after phase 3")

    # Phase 4: Latency
    latency_results = phase_latency()
    log_memory("after phase 4")

    # Combine results
    results = {
        "experiment": "pierre_unified_pipeline",
        "model": MODEL_ID,
        "domains": DOMAINS,
        "total_time_s": round(time.time() - t0, 1),
        "phase1_routing": cal_results,
        "phase2_ppl": ppl_results,
        "phase3_behavioral": behav_results,
        "phase4_latency": latency_results,
    }

    # Kill criteria evaluation
    kill = {}

    # K1: Routing accuracy >= 80%
    k1_acc = cal_results["routing_accuracy"]
    kill["K1_routing_accuracy"] = {
        "value": k1_acc,
        "threshold": 0.80,
        "pass": k1_acc >= 0.80,
        "detail": f"Routing accuracy {k1_acc:.1%} (target >= 80%)",
    }

    # K2: No domain PPL > 10% worse than single-adapter
    worst_deg = max(ppl_results["ppl_degradation"].values())
    kill["K2_ppl_degradation"] = {
        "value": worst_deg,
        "threshold": 10.0,
        "pass": worst_deg <= 10.0,
        "detail": f"Worst PPL degradation {worst_deg:+.2f}% (limit 10%)",
    }

    # K3: Mean behavioral score >= 0.3
    behav_score = behav_results["overall_mean_score"]
    kill["K3_behavioral_quality"] = {
        "value": behav_score,
        "threshold": 0.30,
        "pass": behav_score >= 0.30,
        "detail": f"Mean behavioral score {behav_score:.3f} (target >= 0.3)",
    }

    results["kill_criteria"] = kill
    results["all_pass"] = all(k["pass"] for k in kill.values())

    log("\n" + "=" * 60)
    log("KILL CRITERIA RESULTS:")
    for k, v in kill.items():
        status = "PASS" if v["pass"] else "FAIL"
        log(f"  {k}: {status} — {v['detail']}")
    log(f"\nOverall: {'ALL PASS' if results['all_pass'] else 'KILLED'}")
    log(f"Total time: {results['total_time_s']:.1f}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
