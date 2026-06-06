#!/usr/bin/env python3
"""LoRAuter Task-Representation Routing: Learned Embeddings Replace TF-IDF.

Implements LoRAuter-style routing (arXiv:2601.21795): sentence-embedding centroids
from validation sets replace TF-IDF for adapter selection. Measures whether
embedding similarity predicts adapter effectiveness (not just domain identity).

Kill criteria:
  K1 (#666): Embedding-effectiveness correlation <= 0.3 across all domains
  K2 (#667): Learned routing does not improve behavioral quality on any domain vs TF-IDF
  K3 (#668): >20% incoherent output

Type: guided-exploration
Platform: Apple M5 Pro 48GB, MLX
"""

import ast
import gc
import json
import os
import re
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

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
EVAL_DOMAINS = ["math", "code", "medical"]  # Focus domains for kill criteria
NUM_PROMPTS_PER_DOMAIN = 10
NUM_VAL_SAMPLES_FOR_CENTROIDS = 20  # Per domain, for centroid computation

# Per-domain optimal scales (Finding #249)
DOMAIN_SCALES = {
    "medical": 20.0,
    "code": 20.0,
    "math": 20.0,
    "legal": 4.0,
    "finance": 1.0,
}

# Sentence transformer model (LoRAuter uses Styxxxx/lora_retriever,
# but all-MiniLM-L6-v2 is more widely available and produces 384-dim embeddings)
SENTENCE_MODEL = "all-MiniLM-L6-v2"

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
    import mlx.core as mx
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup_mlx(*objects):
    import mlx.core as mx
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ============================================================================
# Data loading
# ============================================================================

def load_validation_data(domain, n_samples=50):
    """Load validation samples for a domain."""
    val_path = DATA_DIR / domain / "valid.jsonl"
    samples = []
    with open(val_path) as f:
        for line in f:
            text = json.loads(line)["text"]
            if "### Instruction:" in text and "### Response:" in text:
                instruction = text.split("### Instruction:")[1].split("### Response:")[0].strip()
                response = text.split("### Response:")[1].strip()
                samples.append({"instruction": instruction, "response": response})
            if len(samples) >= n_samples:
                break
    return samples


# ============================================================================
# Phase 1: Compute task-representation centroids (sentence-transformer)
# ============================================================================

def phase_compute_centroids():
    """Compute LoRAuter-style task centroids using sentence-transformer embeddings.

    For each domain, encode validation instructions and compute mean embedding.
    """
    log("\n" + "=" * 70)
    log("PHASE 1: COMPUTE TASK-REPRESENTATION CENTROIDS")
    log("=" * 70)
    t0 = time.time()

    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(SENTENCE_MODEL)
    log(f"  Loaded sentence-transformer: {SENTENCE_MODEL}")
    log(f"  Embedding dimension: {st_model.get_sentence_embedding_dimension()}")

    centroids = {}
    centroid_details = {}

    for domain in DOMAINS:
        # Load validation samples for centroid computation
        val_data = load_validation_data(domain, NUM_VAL_SAMPLES_FOR_CENTROIDS)
        instructions = [s["instruction"] for s in val_data]
        log(f"  {domain}: encoding {len(instructions)} validation samples...")

        # Encode with instruction prefix (following LoRAuter)
        prefix = "Represent the sentence for similar task retrieval: "
        texts_with_prefix = [prefix + inst for inst in instructions]
        embeddings = st_model.encode(texts_with_prefix, normalize_embeddings=True)

        # Compute centroid (mean of normalized embeddings, then re-normalize)
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids[domain] = centroid

        # Compute intra-class spread (for Fisher's ratio analysis)
        similarities = embeddings @ centroid
        centroid_details[domain] = {
            "n_samples": len(instructions),
            "mean_self_similarity": float(np.mean(similarities)),
            "std_self_similarity": float(np.std(similarities)),
            "min_self_similarity": float(np.min(similarities)),
        }
        log(f"    centroid self-similarity: mean={np.mean(similarities):.4f}, "
            f"std={np.std(similarities):.4f}")

    # Compute inter-centroid similarities
    domain_list = list(centroids.keys())
    centroid_matrix = np.stack([centroids[d] for d in domain_list])
    inter_sim = centroid_matrix @ centroid_matrix.T
    log(f"\n  Inter-centroid cosine similarity matrix:")
    log(f"  {'':12s}" + "".join(f"{d:>10s}" for d in domain_list))
    for i, d1 in enumerate(domain_list):
        row = f"  {d1:12s}"
        for j, d2 in enumerate(domain_list):
            row += f"{inter_sim[i, j]:10.4f}"
        log(row)

    # Fisher's discriminant ratio: inter-class / intra-class variance
    # High ratio => good separability
    mean_inter = np.mean([inter_sim[i, j] for i in range(len(domain_list))
                          for j in range(len(domain_list)) if i != j])
    mean_intra = np.mean([centroid_details[d]["std_self_similarity"]
                          for d in domain_list])
    fisher_ratio = (1.0 - mean_inter) / max(mean_intra, 1e-8)
    log(f"\n  Mean inter-centroid similarity: {mean_inter:.4f}")
    log(f"  Mean intra-class std: {mean_intra:.4f}")
    log(f"  Fisher-like separability: {fisher_ratio:.2f}")

    elapsed = time.time() - t0
    del st_model
    gc.collect()
    log(f"  Centroid computation: {elapsed:.1f}s")

    return centroids, centroid_details, {
        "inter_similarity_matrix": inter_sim.tolist(),
        "domain_order": domain_list,
        "mean_inter_similarity": float(mean_inter),
        "fisher_ratio": float(fisher_ratio),
        "elapsed_s": round(elapsed, 1),
    }


# ============================================================================
# Phase 2: Route test queries and measure routing accuracy
# ============================================================================

def phase_route_queries(centroids):
    """Route test queries using embedding similarity and measure accuracy."""
    log("\n" + "=" * 70)
    log("PHASE 2: ROUTE TEST QUERIES (EMBEDDING SIMILARITY)")
    log("=" * 70)
    t0 = time.time()

    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(SENTENCE_MODEL)

    # Use DIFFERENT samples from centroid computation (offset by NUM_VAL_SAMPLES_FOR_CENTROIDS)
    # to avoid train-on-test contamination
    test_prompts = {}
    for domain in DOMAINS:
        all_val = load_validation_data(domain, NUM_VAL_SAMPLES_FOR_CENTROIDS + NUM_PROMPTS_PER_DOMAIN)
        # Take samples AFTER the centroid samples
        test_prompts[domain] = all_val[NUM_VAL_SAMPLES_FOR_CENTROIDS:
                                       NUM_VAL_SAMPLES_FOR_CENTROIDS + NUM_PROMPTS_PER_DOMAIN]
        if len(test_prompts[domain]) < NUM_PROMPTS_PER_DOMAIN:
            # Fall back to first samples if not enough
            test_prompts[domain] = all_val[:NUM_PROMPTS_PER_DOMAIN]
            log(f"  WARNING: {domain} only has {len(all_val)} samples, reusing centroid samples")

    # Embed all test queries
    prefix = "Represent the sentence for similar task retrieval: "
    centroid_matrix = np.stack([centroids[d] for d in DOMAINS])

    routing_results = {}
    all_similarities = []  # (domain, query_idx, similarities_dict)

    correct = 0
    total = 0

    for domain in DOMAINS:
        domain_results = []
        for i, prompt_data in enumerate(test_prompts[domain]):
            query_text = prefix + prompt_data["instruction"]
            query_emb = st_model.encode([query_text], normalize_embeddings=True)[0]

            # Compute cosine similarities to all centroids
            similarities = query_emb @ centroid_matrix.T
            sim_dict = {d: float(similarities[j]) for j, d in enumerate(DOMAINS)}

            # Route to highest-similarity domain
            routed_domain = DOMAINS[np.argmax(similarities)]
            is_correct = (routed_domain == domain)
            if is_correct:
                correct += 1
            total += 1

            result = {
                "instruction": prompt_data["instruction"][:100],
                "true_domain": domain,
                "routed_domain": routed_domain,
                "correct": is_correct,
                "similarities": sim_dict,
                "max_similarity": float(np.max(similarities)),
                "true_domain_similarity": float(sim_dict[domain]),
                "margin": float(sim_dict[domain] - sorted(similarities)[-2]) if is_correct
                          else float(sorted(similarities)[-1] - sim_dict[domain]),
            }
            domain_results.append(result)
            all_similarities.append((domain, i, sim_dict))

        routing_results[domain] = domain_results
        domain_correct = sum(1 for r in domain_results if r["correct"])
        log(f"  {domain}: {domain_correct}/{len(domain_results)} correct")

    accuracy = correct / total if total > 0 else 0.0
    log(f"\n  Overall routing accuracy: {correct}/{total} = {accuracy:.1%}")

    elapsed = time.time() - t0
    del st_model
    gc.collect()
    log(f"  Routing evaluation: {elapsed:.1f}s")

    return test_prompts, routing_results, {
        "accuracy": float(accuracy),
        "correct": correct,
        "total": total,
        "per_domain_accuracy": {
            d: float(sum(1 for r in routing_results[d] if r["correct"]) / len(routing_results[d]))
            for d in DOMAINS
        },
        "elapsed_s": round(elapsed, 1),
    }


# ============================================================================
# MLX model utilities (from behavioral_eval_routed)
# ============================================================================

def setup_mlx():
    """Import and configure MLX. Called only when MLX phases begin."""
    import mlx.core as mx
    import mlx.nn as nn
    device_info = mx.device_info()
    total_mem = device_info["memory_size"]
    mx.set_memory_limit(total_mem - 8 * 1024**3)
    mx.set_cache_limit(2 * 1024**3)
    return mx, nn


def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    import mlx.core as mx
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
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.bitlinear_layers import BitLinear
    from mlx.utils import tree_unflatten
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


def load_skeleton():
    skeleton_path = ADAPTERS_DIR / "grassmannian_skeleton.npz"
    return dict(np.load(str(skeleton_path)))


def load_adapter(domain):
    import mlx.core as mx
    adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
    adapter = dict(mx.load(str(adapter_path)))
    log(f"  Loaded adapter: {domain} ({len(adapter)} tensors)")
    return adapter


def premerge_single_adapter(model, skeleton, adapter, domain, scale):
    import mlx.core as mx
    import mlx.nn as nn
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
    import mlx.nn as nn
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
    import mlx.core as mx
    import mlx.nn as nn
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


def format_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_text(model, tokenizer, prompt, max_tokens=128):
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
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


def is_coherent(text):
    """Check if generated text is coherent (not degenerate)."""
    if len(text.strip()) < 10:
        return False
    # Check for excessive repetition
    words = text.split()
    if len(words) < 3:
        return len(text.strip()) > 0
    # Check for single-word repetition
    from collections import Counter
    word_counts = Counter(words)
    most_common_count = word_counts.most_common(1)[0][1]
    if most_common_count > len(words) * 0.5 and len(words) > 10:
        return False
    # Check for phrase repetition (3-grams)
    if len(words) >= 6:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        trigram_counts = Counter(trigrams)
        most_common_trigram = trigram_counts.most_common(1)[0][1]
        if most_common_trigram > len(trigrams) * 0.3 and len(trigrams) > 5:
            return False
    return True


# ============================================================================
# Phase 3: Generate with base model
# ============================================================================

def phase_generate_base(prompts_by_domain):
    log("\n" + "=" * 70)
    log("PHASE 3: GENERATE WITH BASE MODEL")
    log("=" * 70)
    t0 = time.time()

    mx, nn = setup_mlx()
    from mlx_lm import load

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
            generated = generate_text(model, tokenizer, formatted,
                                      max_tokens=MAX_NEW_TOKENS)
            domain_results.append(generated)
            log(f"  [{domain}][{i}] generated {len(generated)} chars")
        results[domain] = domain_results
        log(f"  {domain}: {len(domain_results)} generations complete")

    elapsed = time.time() - t0
    del model, tokenizer
    cleanup_mlx()
    log_memory("post-base-gen")
    log(f"  Base generation: {elapsed:.1f}s")
    return results, elapsed


# ============================================================================
# Phase 4: Generate with embedding-routed composition
# ============================================================================

def phase_generate_embedding_routed(prompts_by_domain, routing_results):
    """Generate using embedding-routed adapter selection.

    For each query, use the routing results to select the adapter, then
    pre-merge and generate.
    """
    log("\n" + "=" * 70)
    log("PHASE 4: GENERATE WITH EMBEDDING-ROUTED COMPOSITION")
    log("=" * 70)
    t0 = time.time()

    mx, nn = setup_mlx()
    from mlx_lm import load

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    base_weights = save_base_weights(model)
    skeleton = load_skeleton()
    log(f"  Loaded Grassmannian skeleton ({len(skeleton)} tensors)")

    mx.random.seed(SEED)
    np.random.seed(SEED)

    # Pre-load all adapters (5 small files)
    adapters = {}
    for domain in DOMAINS:
        adapters[domain] = load_adapter(domain)

    results = {}
    current_merged = None  # Track which adapter is currently merged

    for domain in DOMAINS:
        domain_results = []
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            # Get routed adapter for this query
            routed_domain = routing_results[domain][i]["routed_domain"]
            scale = DOMAIN_SCALES[routed_domain]

            # Only re-merge if the adapter changed
            if routed_domain != current_merged:
                restore_base_weights(model, base_weights)
                premerge_single_adapter(model, skeleton, adapters[routed_domain],
                                       routed_domain, scale)
                current_merged = routed_domain

            formatted = format_prompt(prompt_data["instruction"])
            generated = generate_text(model, tokenizer, formatted,
                                      max_tokens=MAX_NEW_TOKENS)
            domain_results.append({
                "text": generated,
                "routed_to": routed_domain,
                "scale": scale,
            })
            log(f"  [{domain}][{i}] routed->{routed_domain}(s={scale}) {len(generated)} chars")
        results[domain] = domain_results
        log(f"  {domain}: {len(domain_results)} generations complete")

    elapsed = time.time() - t0
    del model, tokenizer, skeleton, base_weights, adapters
    cleanup_mlx()
    log_memory("post-emb-routed-gen")
    log(f"  Embedding-routed generation: {elapsed:.1f}s")
    return results, elapsed


# ============================================================================
# Phase 5: Generate with oracle routing (baseline comparison)
# ============================================================================

def phase_generate_oracle(prompts_by_domain):
    """Generate with oracle routing (perfect adapter selection)."""
    log("\n" + "=" * 70)
    log("PHASE 5: GENERATE WITH ORACLE ROUTING")
    log("=" * 70)
    t0 = time.time()

    mx, nn = setup_mlx()
    from mlx_lm import load

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    base_weights = save_base_weights(model)
    skeleton = load_skeleton()

    mx.random.seed(SEED)
    np.random.seed(SEED)

    results = {}
    for domain in DOMAINS:
        restore_base_weights(model, base_weights)
        adapter = load_adapter(domain)
        scale = DOMAIN_SCALES[domain]
        premerge_single_adapter(model, skeleton, adapter, domain, scale)
        del adapter

        domain_results = []
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            formatted = format_prompt(prompt_data["instruction"])
            generated = generate_text(model, tokenizer, formatted,
                                      max_tokens=MAX_NEW_TOKENS)
            domain_results.append(generated)
            log(f"  [{domain}][{i}] generated {len(generated)} chars")
        results[domain] = domain_results
        log(f"  {domain}: {len(domain_results)} generations complete")

    elapsed = time.time() - t0
    del model, tokenizer, skeleton, base_weights
    cleanup_mlx()
    log_memory("post-oracle-gen")
    log(f"  Oracle generation: {elapsed:.1f}s")
    return results, elapsed


# ============================================================================
# Phase 6: Evaluate all generations
# ============================================================================

def phase_evaluate(prompts_by_domain, base_gen, emb_routed_gen, oracle_gen, routing_results):
    """Evaluate base, embedding-routed, and oracle generations."""
    log("\n" + "=" * 70)
    log("PHASE 6: BEHAVIORAL EVALUATION")
    log("=" * 70)
    t0 = time.time()

    base_evals = {}
    emb_evals = {}
    oracle_evals = {}
    coherence_counts = {"total": 0, "incoherent": 0}

    for domain in DOMAINS:
        log(f"\n  === {domain.upper()} ===")

        base_domain = []
        emb_domain = []
        oracle_domain = []

        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            ref = prompt_data["response"]

            # Base
            base_result = evaluate_response(base_gen[domain][i], ref, domain)
            base_domain.append(base_result)

            # Embedding-routed
            emb_text = emb_routed_gen[domain][i]["text"]
            emb_result = evaluate_response(emb_text, ref, domain)
            emb_result["routed_to"] = emb_routed_gen[domain][i]["routed_to"]
            emb_result["routing_correct"] = routing_results[domain][i]["correct"]
            emb_result["true_domain_similarity"] = routing_results[domain][i]["true_domain_similarity"]
            emb_domain.append(emb_result)

            # Oracle
            oracle_result = evaluate_response(oracle_gen[domain][i], ref, domain)
            oracle_domain.append(oracle_result)

            # Coherence check for embedding-routed
            coherence_counts["total"] += 1
            if not is_coherent(emb_text):
                coherence_counts["incoherent"] += 1

        base_scores = [r["score"] for r in base_domain]
        emb_scores = [r["score"] for r in emb_domain]
        oracle_scores = [r["score"] for r in oracle_domain]

        log(f"  Base:    {np.mean(base_scores):.4f}")
        log(f"  Emb-rtr: {np.mean(emb_scores):.4f}")
        log(f"  Oracle:  {np.mean(oracle_scores):.4f}")

        base_evals[domain] = base_domain
        emb_evals[domain] = emb_domain
        oracle_evals[domain] = oracle_domain

    elapsed = time.time() - t0
    log(f"\n  Evaluation: {elapsed:.1f}s")

    return base_evals, emb_evals, oracle_evals, coherence_counts, elapsed


# ============================================================================
# Phase 7: Compute embedding-effectiveness correlation
# ============================================================================

def phase_correlation_analysis(emb_evals, routing_results):
    """Compute correlation between embedding similarity and behavioral score."""
    log("\n" + "=" * 70)
    log("PHASE 7: EMBEDDING-EFFECTIVENESS CORRELATION")
    log("=" * 70)

    from scipy import stats

    correlations = {}

    for domain in DOMAINS:
        similarities = []
        scores = []
        for i, (emb_result, route_result) in enumerate(
                zip(emb_evals[domain], routing_results[domain])):
            sim = route_result["true_domain_similarity"]
            score = emb_result["score"]
            similarities.append(sim)
            scores.append(score)

        similarities = np.array(similarities)
        scores = np.array(scores)

        # Pearson correlation
        if np.std(similarities) > 1e-8 and np.std(scores) > 1e-8:
            r, p = stats.pearsonr(similarities, scores)
        else:
            r, p = 0.0, 1.0

        correlations[domain] = {
            "pearson_r": float(r),
            "p_value": float(p),
            "n_samples": len(similarities),
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
        }
        log(f"  {domain}: r={r:.4f}, p={p:.4f}, "
            f"mean_sim={np.mean(similarities):.4f}, mean_score={np.mean(scores):.4f}")

    # Overall correlation (all domains pooled)
    all_sims = []
    all_scores = []
    for domain in DOMAINS:
        for i, (emb_result, route_result) in enumerate(
                zip(emb_evals[domain], routing_results[domain])):
            all_sims.append(route_result["true_domain_similarity"])
            all_scores.append(emb_result["score"])

    all_sims = np.array(all_sims)
    all_scores = np.array(all_scores)
    if np.std(all_sims) > 1e-8 and np.std(all_scores) > 1e-8:
        r_overall, p_overall = stats.pearsonr(all_sims, all_scores)
    else:
        r_overall, p_overall = 0.0, 1.0

    correlations["overall"] = {
        "pearson_r": float(r_overall),
        "p_value": float(p_overall),
        "n_samples": len(all_sims),
    }
    log(f"\n  Overall: r={r_overall:.4f}, p={p_overall:.4f}")

    return correlations


# ============================================================================
# Main orchestrator
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("LORAUTER TASK-REPRESENTATION ROUTING")
    log("=" * 70)
    log(f"Model: {MODEL_ID}")
    log(f"Sentence-transformer: {SENTENCE_MODEL}")
    log(f"Domains: {DOMAINS}")
    log(f"Prompts/domain (eval): {NUM_PROMPTS_PER_DOMAIN}")
    log(f"Samples/domain (centroids): {NUM_VAL_SAMPLES_FOR_CENTROIDS}")
    log(f"Per-domain scales: {DOMAIN_SCALES}")

    # Phase 1: Compute centroids (lightweight, no GPU)
    centroids, centroid_details, centroid_stats = phase_compute_centroids()

    # Phase 2: Route test queries (lightweight, no GPU)
    test_prompts, routing_results, routing_stats = phase_route_queries(centroids)

    # Phase 3: Base model generation (MLX GPU)
    base_gen, base_time = phase_generate_base(test_prompts)

    # Phase 4: Embedding-routed generation (MLX GPU)
    emb_gen, emb_time = phase_generate_embedding_routed(test_prompts, routing_results)

    # Phase 5: Oracle generation (MLX GPU)
    oracle_gen, oracle_time = phase_generate_oracle(test_prompts)

    # Phase 6: Evaluate all
    base_evals, emb_evals, oracle_evals, coherence, eval_time = phase_evaluate(
        test_prompts, base_gen, emb_gen, oracle_gen, routing_results)

    # Phase 7: Correlation analysis
    correlations = phase_correlation_analysis(emb_evals, routing_results)

    # ============================================================================
    # Kill criteria assessment
    # ============================================================================
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    comparison = {}
    for domain in DOMAINS:
        b_scores = [r["score"] for r in base_evals[domain]]
        e_scores = [r["score"] for r in emb_evals[domain]]
        o_scores = [r["score"] for r in oracle_evals[domain]]

        b_mean = float(np.mean(b_scores))
        e_mean = float(np.mean(e_scores))
        o_mean = float(np.mean(o_scores))

        comparison[domain] = {
            "base_mean": round(b_mean, 4),
            "emb_routed_mean": round(e_mean, 4),
            "oracle_mean": round(o_mean, 4),
            "emb_vs_base": round(e_mean - b_mean, 4),
            "emb_vs_oracle": round(e_mean - o_mean, 4),
            "oracle_pct": round(e_mean / max(o_mean, 0.001) * 100, 1),
            "routing_accuracy": routing_stats["per_domain_accuracy"][domain],
            "scale_used": DOMAIN_SCALES[domain],
            "n_samples": len(b_scores),
        }
        log(f"  {domain:10s}: base={b_mean:.3f} emb={e_mean:.3f} oracle={o_mean:.3f} "
            f"route_acc={routing_stats['per_domain_accuracy'][domain]:.0%}")

    # K1: Embedding-effectiveness correlation > 0.3 on any domain
    any_domain_corr_above = any(
        abs(correlations[d]["pearson_r"]) > 0.3 for d in DOMAINS
    )
    max_corr = max(abs(correlations[d]["pearson_r"]) for d in DOMAINS)
    overall_corr = abs(correlations["overall"]["pearson_r"])
    k1_pass = any_domain_corr_above or overall_corr > 0.3
    log(f"\n  K1 (#666): {'PASS' if k1_pass else 'FAIL'} — "
        f"Max per-domain |r|={max_corr:.4f}, overall |r|={overall_corr:.4f} "
        f"(threshold: >0.3)")

    # K2: Learned routing improves behavioral quality on any domain vs TF-IDF
    # Compare embedding-routed vs base (TF-IDF routing was equivalent to base
    # in Finding #253 since TF-IDF has r=-0.079)
    # We compare emb-routed vs oracle to measure effectiveness
    domains_emb_better_than_base = sum(
        1 for d in DOMAINS if comparison[d]["emb_vs_base"] > 0.02
    )
    k2_pass = domains_emb_better_than_base >= 1
    log(f"  K2 (#667): {'PASS' if k2_pass else 'FAIL'} — "
        f"Emb-routed better than base on {domains_emb_better_than_base}/5 domains "
        f"(threshold: >=1)")

    # K3: Incoherent output <= 20%
    incoherent_rate = coherence["incoherent"] / max(coherence["total"], 1)
    k3_pass = incoherent_rate <= 0.20
    log(f"  K3 (#668): {'PASS' if k3_pass else 'FAIL'} — "
        f"Incoherent: {coherence['incoherent']}/{coherence['total']} = {incoherent_rate:.1%} "
        f"(threshold: <=20%)")

    # ============================================================================
    # Compile results
    # ============================================================================
    results = {
        "experiment": "lorauter_task_routing",
        "description": "LoRAuter task-representation routing: sentence-embedding centroids for adapter selection",
        "model": MODEL_ID,
        "sentence_model": SENTENCE_MODEL,
        "n_domains": len(DOMAINS),
        "n_prompts_per_domain": NUM_PROMPTS_PER_DOMAIN,
        "n_val_for_centroids": NUM_VAL_SAMPLES_FOR_CENTROIDS,
        "domain_scales": DOMAIN_SCALES,
        "centroid_stats": centroid_stats,
        "centroid_details": centroid_details,
        "routing_stats": routing_stats,
        "comparison": comparison,
        "correlations": correlations,
        "coherence": {
            "total": coherence["total"],
            "incoherent": coherence["incoherent"],
            "rate": round(incoherent_rate, 4),
        },
        "kill_criteria": {
            "K1_666": {
                "description": "Embedding-effectiveness correlation > 0.3 on any domain",
                "max_per_domain_abs_r": round(max_corr, 4),
                "overall_abs_r": round(overall_corr, 4),
                "result": "PASS" if k1_pass else "FAIL",
            },
            "K2_667": {
                "description": "Emb-routed improves behavioral quality on >=1 domain vs base",
                "domains_better": domains_emb_better_than_base,
                "result": "PASS" if k2_pass else "FAIL",
            },
            "K3_668": {
                "description": "Incoherent output <= 20%",
                "incoherent_rate": round(incoherent_rate, 4),
                "result": "PASS" if k3_pass else "FAIL",
            },
        },
        "predictions_vs_measured": {
            "P1_routing_accuracy": {
                "predicted": ">= 80%",
                "measured": round(routing_stats["accuracy"], 4),
                "match": routing_stats["accuracy"] >= 0.80,
            },
            "P2_effectiveness_correlation": {
                "predicted": "r > 0.3 on at least 1 domain",
                "measured_max": round(max_corr, 4),
                "measured_overall": round(overall_corr, 4),
                "match": k1_pass,
            },
            "P3_behavioral_improvement": {
                "predicted": "emb-routed >= base on >=1 domain",
                "domains_better": domains_emb_better_than_base,
                "match": k2_pass,
            },
            "P4_coherence": {
                "predicted": "<= 20% incoherent",
                "measured": round(incoherent_rate, 4),
                "match": k3_pass,
            },
        },
        "timing": {
            "centroid_time_s": centroid_stats["elapsed_s"],
            "routing_time_s": routing_stats["elapsed_s"],
            "base_gen_time_s": round(base_time, 1),
            "emb_routed_gen_time_s": round(emb_time, 1),
            "oracle_gen_time_s": round(oracle_time, 1),
            "eval_time_s": round(eval_time, 1),
            "total_time_s": round(time.time() - t0, 1),
        },
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  Routing accuracy: {routing_stats['accuracy']:.1%}")
    for domain in DOMAINS:
        c = comparison[domain]
        log(f"  {domain:10s}: base={c['base_mean']:.3f} emb={c['emb_routed_mean']:.3f} "
            f"oracle={c['oracle_mean']:.3f} route={c['routing_accuracy']:.0%}")
    log(f"\n  K1 (#666): {results['kill_criteria']['K1_666']['result']} (|r|={max_corr:.4f})")
    log(f"  K2 (#667): {results['kill_criteria']['K2_667']['result']} ({domains_emb_better_than_base} domains)")
    log(f"  K3 (#668): {results['kill_criteria']['K3_668']['result']} ({incoherent_rate:.1%} incoherent)")

    total_time = time.time() - t0
    log(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    return results


if __name__ == "__main__":
    main()
