#!/usr/bin/env python3
"""Input-Dependent Adapter Scaling via Embedding Similarity.

Tests whether per-query scale selection (using TF-IDF cosine similarity to domain
centroids) improves behavioral quality over fixed per-domain scales.

Type: guided-exploration
Platform: Apple M5 Pro 48GB, MLX

Kill criteria:
  K1 (#663): Per-query scaling improves behavioral quality over fixed per-domain on 0/3 domains
  K2 (#664): Embedding-scale correlation less than 0.3
  K3 (#665): Per-query system produces incoherent output on more than 20% of queries
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

# Paths to existing infrastructure
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]
# Focus on 3 domains for kill criteria assessment
EVAL_DOMAINS = ["math", "code", "medical"]
NUM_PROMPTS_PER_DOMAIN = 10

# Per-domain optimal scales (Finding #249)
DOMAIN_SCALES = {
    "medical": 20.0,
    "code": 20.0,
    "math": 20.0,
    "legal": 4.0,
    "finance": 1.0,
}

# Alpha: floor for scale modulation (minimum fraction of optimal scale)
# alpha=0.3 means even maximally OOD queries get 30% of optimal scale
ALPHA = 0.3

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
# Model utilities (from behavioral_eval_routed)
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
    """Pre-merge a single adapter into model weights: W_new = W_base + scale * B^T @ A^T"""
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
    log(f"  Pre-merged {domain} adapter (scale={scale:.2f}) into {merge_count} layers")
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
# Evaluation metrics (from behavioral_eval_routed)
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


def extract_key_facts(text):
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
    result = {"domain": domain, "generated_len": len(generated_text)}

    if domain == "code":
        syntax_ok = eval_code_syntax(generated_text)
        factual = eval_factual_recall(generated_text, reference_text)
        score = 0.7 * (1.0 if syntax_ok else 0.0) + 0.3 * factual["recall"]
        result.update({
            "score": score, "syntax_valid": syntax_ok,
            "factual_recall": factual["recall"], "factual_f1": factual["f1"],
            "method": "syntax_parse + factual_recall",
        })

    elif domain == "math":
        gen_answer = extract_math_answer(generated_text)
        gt_answer = extract_ground_truth_answer(reference_text)
        correct = eval_math_correct(gen_answer, gt_answer)
        score = 1.0 if correct else 0.0
        result.update({
            "score": score, "answer_correct": correct,
            "gen_answer": gen_answer, "gt_answer": gt_answer,
            "method": "numerical_answer_match (eps=0.01)",
        })

    elif domain == "medical":
        factual = eval_factual_recall(generated_text, reference_text)
        score = factual["recall"]
        result.update({
            "score": score, "factual_recall": factual["recall"],
            "factual_precision": factual["precision"], "factual_f1": factual["f1"],
            "method": "factual_recall (medical facts vs reference)",
        })

    elif domain == "legal":
        factual = eval_factual_recall(generated_text, reference_text)
        score = factual["recall"]
        result.update({
            "score": score, "factual_recall": factual["recall"],
            "factual_precision": factual["precision"], "factual_f1": factual["f1"],
            "method": "factual_recall (legal facts vs reference)",
        })

    elif domain == "finance":
        factual = eval_factual_recall(generated_text, reference_text)
        score = factual["recall"]
        result.update({
            "score": score, "factual_recall": factual["recall"],
            "factual_f1": factual["f1"],
            "method": "factual_recall",
        })

    return result


def check_coherence(text):
    """Check if generated text is coherent (not degenerate)."""
    if len(text.strip()) < 10:
        return False, "too_short"
    # Check for excessive repetition
    words = text.split()
    if len(words) > 10:
        counter = Counter(words)
        most_common_freq = counter.most_common(1)[0][1] / len(words)
        if most_common_freq > 0.5:
            return False, "repetitive"
    # Check for garbled output (mostly non-alphabetic)
    alpha_chars = sum(1 for c in text if c.isalpha())
    if len(text) > 0 and alpha_chars / len(text) < 0.3:
        return False, "garbled"
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


def load_domain_instructions(domain, split="train", max_samples=400):
    path = DATA_DIR / domain / f"{split}.jsonl"
    instructions = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            text = json.loads(line)["text"]
            if "### Instruction:" in text:
                instruction = text.split("### Instruction:")[1].split("### Response:")[0].strip()
                instructions.append(instruction)
    return instructions


# ============================================================================
# Phase 0: Build TF-IDF router and compute per-query similarities
# ============================================================================

def phase_build_router():
    """Train TF-IDF vectorizer and compute domain centroids.

    Returns vectorizer, centroids, and per-domain similarity statistics.
    """
    log("\n" + "=" * 70)
    log("PHASE 0: BUILD TF-IDF ROUTER + COMPUTE CENTROIDS")
    log("=" * 70)
    t0 = time.time()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    # Load training data
    train_texts = []
    train_labels = []
    domain_texts = {}
    for di, domain in enumerate(DOMAINS):
        instructions = load_domain_instructions(domain, split="train", max_samples=400)
        train_texts.extend(instructions)
        train_labels.extend([di] * len(instructions))
        domain_texts[domain] = instructions
        log(f"  {domain}: {len(instructions)} train instructions")

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    log(f"  TF-IDF features: {X_train.shape[1]}")

    # Compute domain centroids
    centroids = {}
    centroid_norms = {}
    for di, domain in enumerate(DOMAINS):
        mask = np.array(train_labels) == di
        domain_vecs = X_train[mask].toarray()
        centroid = domain_vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids[domain] = centroid
        centroid_norms[domain] = norm
        log(f"  {domain} centroid norm: {norm:.4f}")

    # Compute within-domain similarity statistics (for calibration)
    sim_stats = {}
    for di, domain in enumerate(DOMAINS):
        mask = np.array(train_labels) == di
        domain_vecs = X_train[mask].toarray()
        centroid = centroids[domain]
        centroid_norm = centroid_norms[domain]

        # Cosine similarity of each training sample to its own centroid
        sims = []
        for vec in domain_vecs:
            vec_norm = np.linalg.norm(vec)
            if vec_norm > 0 and centroid_norm > 0:
                sim = np.dot(vec, centroid) / (vec_norm * centroid_norm)
                sims.append(sim)
        sims = np.array(sims)
        sim_stats[domain] = {
            "mean": float(np.mean(sims)),
            "std": float(np.std(sims)),
            "min": float(np.min(sims)),
            "max": float(np.max(sims)),
            "p25": float(np.percentile(sims, 25)),
            "p75": float(np.percentile(sims, 75)),
        }
        log(f"  {domain} in-domain sim: mean={sim_stats[domain]['mean']:.4f}, "
            f"std={sim_stats[domain]['std']:.4f}, "
            f"range=[{sim_stats[domain]['min']:.4f}, {sim_stats[domain]['max']:.4f}]")

    # Train classifier (for routing accuracy comparison)
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=SEED)
    clf.fit(X_train, np.array(train_labels))
    train_acc = float(np.mean(clf.predict(X_train) == np.array(train_labels)))
    log(f"  Router train accuracy: {train_acc:.4f}")

    elapsed = time.time() - t0
    log(f"  Phase 0: {elapsed:.1f}s")

    return vectorizer, centroids, centroid_norms, sim_stats, clf


def compute_query_similarity(query_text, vectorizer, centroids, centroid_norms):
    """Compute cosine similarity of query to each domain centroid."""
    q_vec = vectorizer.transform([query_text]).toarray()[0]
    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0:
        return {d: 0.0 for d in DOMAINS}

    sims = {}
    for domain in DOMAINS:
        centroid = centroids[domain]
        c_norm = centroid_norms[domain]
        if c_norm > 0:
            sims[domain] = float(np.dot(q_vec, centroid) / (q_norm * c_norm))
        else:
            sims[domain] = 0.0
    return sims


def compute_input_dependent_scale(sim, domain, sim_stats):
    """Map similarity to scale using normalized linear mapping.

    Scale = s_d * f(sim) where f(sim) = max(alpha, sim_normalized)
    sim_normalized = (sim - min_sim) / (max_sim - min_sim) for domain calibration.
    """
    s_d = DOMAIN_SCALES[domain]
    stats = sim_stats[domain]

    # Normalize similarity relative to in-domain distribution
    # Map [p25, p75] -> [0, 1] (more robust than min/max)
    sim_range = stats["p75"] - stats["p25"]
    if sim_range > 0:
        sim_normalized = (sim - stats["p25"]) / sim_range
    else:
        sim_normalized = 1.0  # If all similarities are the same, use full scale

    # Clamp to [alpha, 1.0]
    f_sim = max(ALPHA, min(1.0, sim_normalized))

    return s_d * f_sim


# ============================================================================
# Phase 1: Generate with fixed per-domain scales (baseline from prior work)
# ============================================================================

def phase_generate_fixed(prompts_by_domain):
    """Generate with fixed per-domain scales (oracle routing)."""
    log("\n" + "=" * 70)
    log("PHASE 1: GENERATE WITH FIXED PER-DOMAIN SCALES (BASELINE)")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    base_weights = save_base_weights(model)
    skeleton = load_skeleton()
    log(f"  Loaded Grassmannian skeleton ({len(skeleton)} tensors)")

    mx.random.seed(SEED)
    np.random.seed(SEED)

    results = {}
    for domain in EVAL_DOMAINS:
        restore_base_weights(model, base_weights)
        adapter = load_adapter(domain)
        scale = DOMAIN_SCALES[domain]
        model = premerge_single_adapter(model, skeleton, adapter, domain, scale)
        del adapter

        domain_results = []
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            formatted = format_prompt(prompt_data["instruction"])
            generated = generate_text(model, tokenizer, formatted,
                                      max_tokens=MAX_NEW_TOKENS)
            domain_results.append(generated)
            log(f"  [fixed][{domain}][{i}] {len(generated)} chars")
        results[domain] = domain_results

    elapsed = time.time() - t0
    del model, tokenizer, skeleton, base_weights
    cleanup()
    log_memory("post-fixed-gen")
    log(f"  Fixed generation: {elapsed:.1f}s")
    return results, elapsed


# ============================================================================
# Phase 2: Generate with input-dependent scales
# ============================================================================

def phase_generate_input_dependent(prompts_by_domain, vectorizer, centroids,
                                    centroid_norms, sim_stats):
    """Generate with per-query scale selection based on embedding similarity."""
    log("\n" + "=" * 70)
    log("PHASE 2: GENERATE WITH INPUT-DEPENDENT SCALES")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    base_weights = save_base_weights(model)
    skeleton = load_skeleton()
    log(f"  Loaded Grassmannian skeleton ({len(skeleton)} tensors)")

    mx.random.seed(SEED)
    np.random.seed(SEED)

    results = {}
    per_query_info = {}

    for domain in EVAL_DOMAINS:
        adapter = load_adapter(domain)
        domain_results = []
        domain_query_info = []

        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            instruction = prompt_data["instruction"]

            # Compute similarity to all domain centroids
            sims = compute_query_similarity(instruction, vectorizer,
                                            centroids, centroid_norms)
            sim_to_own = sims[domain]

            # Compute input-dependent scale
            scale = compute_input_dependent_scale(sim_to_own, domain, sim_stats)
            fixed_scale = DOMAIN_SCALES[domain]

            log(f"  [dynamic][{domain}][{i}] sim={sim_to_own:.4f} "
                f"scale={scale:.2f} (fixed={fixed_scale:.1f})")

            # Restore base weights and merge with dynamic scale
            restore_base_weights(model, base_weights)
            model = premerge_single_adapter(model, skeleton, adapter, domain, scale)

            formatted = format_prompt(instruction)
            generated = generate_text(model, tokenizer, formatted,
                                      max_tokens=MAX_NEW_TOKENS)
            domain_results.append(generated)

            domain_query_info.append({
                "prompt_idx": i,
                "instruction_preview": instruction[:80],
                "similarities": sims,
                "sim_to_routed_domain": sim_to_own,
                "input_dependent_scale": scale,
                "fixed_scale": fixed_scale,
                "scale_ratio": scale / fixed_scale if fixed_scale > 0 else 0,
            })

        results[domain] = domain_results
        per_query_info[domain] = domain_query_info
        del adapter
        gc.collect()

    elapsed = time.time() - t0
    del model, tokenizer, skeleton, base_weights
    cleanup()
    log_memory("post-dynamic-gen")
    log(f"  Input-dependent generation: {elapsed:.1f}s")
    return results, per_query_info, elapsed


# ============================================================================
# Phase 3: Evaluate and compare
# ============================================================================

def phase_evaluate(prompts_by_domain, fixed_generations, dynamic_generations,
                   per_query_info):
    """Evaluate both fixed and input-dependent generations."""
    log("\n" + "=" * 70)
    log("PHASE 3: BEHAVIORAL EVALUATION")
    log("=" * 70)
    t0 = time.time()

    fixed_evals = {}
    dynamic_evals = {}

    for domain in EVAL_DOMAINS:
        log(f"\n  === {domain.upper()} ===")

        fixed_domain_evals = []
        for i, (prompt_data, gen_text) in enumerate(zip(
                prompts_by_domain[domain], fixed_generations[domain])):
            result = evaluate_response(gen_text, prompt_data["response"], domain)
            result["prompt"] = prompt_data["instruction"][:100]
            result["generated_preview"] = gen_text[:200]
            coherent, reason = check_coherence(gen_text)
            result["coherent"] = coherent
            result["coherence_reason"] = reason
            fixed_domain_evals.append(result)

        dynamic_domain_evals = []
        for i, (prompt_data, gen_text) in enumerate(zip(
                prompts_by_domain[domain], dynamic_generations[domain])):
            result = evaluate_response(gen_text, prompt_data["response"], domain)
            result["prompt"] = prompt_data["instruction"][:100]
            result["generated_preview"] = gen_text[:200]
            coherent, reason = check_coherence(gen_text)
            result["coherent"] = coherent
            result["coherence_reason"] = reason
            # Attach similarity info
            qi = per_query_info[domain][i]
            result["sim_to_domain"] = qi["sim_to_routed_domain"]
            result["input_dependent_scale"] = qi["input_dependent_scale"]
            result["fixed_scale"] = qi["fixed_scale"]
            dynamic_domain_evals.append(result)

        fixed_scores = [r["score"] for r in fixed_domain_evals]
        dynamic_scores = [r["score"] for r in dynamic_domain_evals]

        fixed_mean = np.mean(fixed_scores)
        dynamic_mean = np.mean(dynamic_scores)
        improvement = dynamic_mean - fixed_mean

        log(f"  Fixed avg score:     {fixed_mean:.4f}")
        log(f"  Dynamic avg score:   {dynamic_mean:.4f}")
        log(f"  Improvement:         {improvement:+.4f} "
            f"({improvement/max(fixed_mean,0.001)*100:+.1f}%)")

        # Per-prompt comparison
        for i in range(len(fixed_scores)):
            qi = per_query_info[domain][i]
            f_score = fixed_scores[i]
            d_score = dynamic_scores[i]
            delta = d_score - f_score
            log(f"    [{i}] sim={qi['sim_to_routed_domain']:.3f} "
                f"scale={qi['input_dependent_scale']:.1f} "
                f"fixed={f_score:.3f} dynamic={d_score:.3f} "
                f"delta={delta:+.3f}")

        fixed_evals[domain] = fixed_domain_evals
        dynamic_evals[domain] = dynamic_domain_evals

    elapsed = time.time() - t0
    log(f"\n  Evaluation: {elapsed:.1f}s")
    return fixed_evals, dynamic_evals, elapsed


# ============================================================================
# Phase 4: Correlation analysis
# ============================================================================

def phase_correlation_analysis(dynamic_evals, per_query_info):
    """Compute embedding-score correlation (K2)."""
    log("\n" + "=" * 70)
    log("PHASE 4: CORRELATION ANALYSIS")
    log("=" * 70)

    all_sims = []
    all_scores = []
    per_domain_corr = {}

    for domain in EVAL_DOMAINS:
        sims = [r["sim_to_domain"] for r in dynamic_evals[domain]]
        scores = [r["score"] for r in dynamic_evals[domain]]

        all_sims.extend(sims)
        all_scores.extend(scores)

        if len(set(scores)) > 1 and len(set(sims)) > 1:
            corr = float(np.corrcoef(sims, scores)[0, 1])
        else:
            corr = 0.0

        per_domain_corr[domain] = {
            "correlation": corr,
            "n": len(sims),
            "sim_mean": float(np.mean(sims)),
            "sim_std": float(np.std(sims)),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
        }
        log(f"  {domain}: r={corr:.4f} (n={len(sims)}, "
            f"sim={np.mean(sims):.4f}+/-{np.std(sims):.4f}, "
            f"score={np.mean(scores):.4f}+/-{np.std(scores):.4f})")

    # Overall correlation
    if len(set(all_scores)) > 1 and len(set(all_sims)) > 1:
        overall_corr = float(np.corrcoef(all_sims, all_scores)[0, 1])
    else:
        overall_corr = 0.0

    log(f"\n  Overall correlation: r={overall_corr:.4f} (n={len(all_sims)})")

    return per_domain_corr, overall_corr


# ============================================================================
# Main orchestrator
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("INPUT-DEPENDENT ADAPTER SCALING VIA EMBEDDING SIMILARITY")
    log("=" * 70)
    log(f"Model: {MODEL_ID}")
    log(f"Eval domains: {EVAL_DOMAINS}")
    log(f"Prompts/domain: {NUM_PROMPTS_PER_DOMAIN}")
    log(f"Max tokens: {MAX_NEW_TOKENS}")
    log(f"Alpha (floor): {ALPHA}")
    log(f"Per-domain scales: {DOMAIN_SCALES}")
    log_memory("start")

    # Load prompts
    log("\nLoading evaluation prompts...")
    prompts_by_domain = {}
    for domain in EVAL_DOMAINS:
        prompts = extract_prompts_with_answers(domain, NUM_PROMPTS_PER_DOMAIN)
        prompts_by_domain[domain] = prompts
        log(f"  {domain}: {len(prompts)} prompts loaded")

    # Phase 0: Build TF-IDF router
    vectorizer, centroids, centroid_norms, sim_stats, clf = phase_build_router()

    # Compute similarities for eval prompts (for analysis)
    eval_similarities = {}
    for domain in EVAL_DOMAINS:
        domain_sims = []
        for prompt_data in prompts_by_domain[domain]:
            sims = compute_query_similarity(prompt_data["instruction"],
                                            vectorizer, centroids, centroid_norms)
            domain_sims.append(sims)
        eval_similarities[domain] = domain_sims
        own_sims = [s[domain] for s in domain_sims]
        log(f"  {domain} eval similarities to own centroid: "
            f"mean={np.mean(own_sims):.4f}, "
            f"range=[{min(own_sims):.4f}, {max(own_sims):.4f}]")

    # Phase 1: Fixed per-domain scales (baseline)
    fixed_gen, fixed_time = phase_generate_fixed(prompts_by_domain)

    # Phase 2: Input-dependent scales
    dynamic_gen, per_query_info, dynamic_time = phase_generate_input_dependent(
        prompts_by_domain, vectorizer, centroids, centroid_norms, sim_stats)

    # Phase 3: Evaluate
    fixed_evals, dynamic_evals, eval_time = phase_evaluate(
        prompts_by_domain, fixed_gen, dynamic_gen, per_query_info)

    # Phase 4: Correlation analysis
    per_domain_corr, overall_corr = phase_correlation_analysis(
        dynamic_evals, per_query_info)

    # ============================================================================
    # Kill criteria assessment
    # ============================================================================
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    comparison = {}
    domains_improved = 0

    for domain in EVAL_DOMAINS:
        f_scores = [r["score"] for r in fixed_evals[domain]]
        d_scores = [r["score"] for r in dynamic_evals[domain]]
        f_mean = float(np.mean(f_scores))
        d_mean = float(np.mean(d_scores))
        improvement = d_mean - f_mean
        improvement_pct = improvement / max(f_mean, 0.001) * 100

        improved = improvement > 0.02  # 2% threshold for "improved"
        if improved:
            domains_improved += 1

        comparison[domain] = {
            "fixed_mean": round(f_mean, 4),
            "dynamic_mean": round(d_mean, 4),
            "improvement": round(improvement, 4),
            "improvement_pct": round(improvement_pct, 1),
            "improved": improved,
            "fixed_scale": DOMAIN_SCALES[domain],
            "n_samples": len(f_scores),
        }

        log(f"\n  {domain.upper()}:")
        log(f"    Fixed:     {f_mean:.4f} (scale={DOMAIN_SCALES[domain]})")
        log(f"    Dynamic:   {d_mean:.4f}")
        log(f"    Delta:     {improvement:+.4f} ({improvement_pct:+.1f}%)")
        log(f"    Improved:  {improved}")

    # K1: Per-query scaling improves on 0/3 domains
    k1_domains_improved = domains_improved
    k1_pass = domains_improved >= 1  # PASS if at least 1 domain improves
    k1_kill = domains_improved == 0  # KILL if 0/3 improve

    # K2: Embedding-scale correlation < 0.3
    k2_corr = overall_corr
    k2_pass = abs(k2_corr) >= 0.3
    k2_kill = abs(k2_corr) < 0.3

    # K3: Incoherent output > 20%
    total_queries = 0
    incoherent_queries = 0
    for domain in EVAL_DOMAINS:
        for r in dynamic_evals[domain]:
            total_queries += 1
            if not r.get("coherent", True):
                incoherent_queries += 1
    incoherent_pct = incoherent_queries / max(total_queries, 1) * 100
    k3_pass = incoherent_pct <= 20
    k3_kill = incoherent_pct > 20

    log(f"\n  K1 (#663): Domains improved: {k1_domains_improved}/3 -> "
        f"{'PASS' if k1_pass else 'FAIL (KILL: 0/3 improved)'}")
    log(f"  K2 (#664): Overall correlation: r={k2_corr:.4f} -> "
        f"{'PASS (|r|>=0.3)' if k2_pass else 'FAIL (|r|<0.3)'}")
    log(f"  K3 (#665): Incoherent: {incoherent_queries}/{total_queries} "
        f"({incoherent_pct:.1f}%) -> {'PASS' if k3_pass else 'FAIL (>20%)'}")

    # ============================================================================
    # Save results
    # ============================================================================

    total_time = time.time() - t0

    results = {
        "experiment": "exp_input_dependent_scaling",
        "description": "Input-dependent adapter scaling via TF-IDF embedding similarity",
        "type": "guided-exploration",
        "model": MODEL_ID,
        "eval_domains": EVAL_DOMAINS,
        "n_prompts_per_domain": NUM_PROMPTS_PER_DOMAIN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "alpha": ALPHA,
        "domain_scales": DOMAIN_SCALES,
        "sim_stats": sim_stats,
        "comparison": comparison,
        "correlation": {
            "per_domain": per_domain_corr,
            "overall": overall_corr,
        },
        "kill_criteria": {
            "K1_domains_improved": {
                "id": 663,
                "value": k1_domains_improved,
                "threshold": ">=1 to pass, 0 to kill",
                "result": "pass" if k1_pass else "fail",
            },
            "K2_correlation": {
                "id": 664,
                "value": round(k2_corr, 4),
                "threshold": "|r| >= 0.3",
                "result": "pass" if k2_pass else "fail",
            },
            "K3_incoherent": {
                "id": 665,
                "value": round(incoherent_pct, 1),
                "threshold": "<= 20%",
                "result": "pass" if k3_pass else "fail",
            },
        },
        "per_query_info": {d: info for d, info in per_query_info.items()},
        "fixed_eval_details": {d: evals for d, evals in fixed_evals.items()},
        "dynamic_eval_details": {d: evals for d, evals in dynamic_evals.items()},
        "timing": {
            "fixed_generation_s": round(fixed_time, 1),
            "dynamic_generation_s": round(dynamic_time, 1),
            "evaluation_s": round(eval_time, 1),
            "total_s": round(total_time, 1),
        },
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    log(f"\nResults saved to {RESULTS_FILE}")

    log(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f}m)")
    log_memory("end")


if __name__ == "__main__":
    main()
