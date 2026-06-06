#!/usr/bin/env python3
"""CPT vs SFT Adapters: Knowledge Injection via Continued Pre-Training.

Tests whether continued pre-training (CPT) adapters trained with causal LM
on raw domain text inject domain knowledge and improve prose domain behavioral
quality vs SFT adapters.

Kill criteria:
  K1 (#672): CPT WORSE behavioral quality than SFT on legal AND medical -> KILL
  K2 (#673): CPT incoherent output >20% -> KILL
  K3 (#674): CPT training fails to converge or >2 hours on M5 Pro -> KILL

Type: Guided exploration (proven framework: two-regime model Finding #249)
Platform: Apple M5 Pro 48GB, MLX
"""

import ast
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
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.bitlinear_layers import BitLinear

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
CPT_ADAPTERS_DIR = EXPERIMENT_DIR / "cpt_adapters"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0  # Use CAPABILITY regime scale (Finding #249)
TRAIN_ITERS = 200
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0
SEED = 42

# Only test prose domains where SFT degrades (Finding #209)
DOMAINS = ["legal", "medical"]
NUM_PROMPTS_PER_DOMAIN = 10

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
# Model utilities (from real_data_domain_experts)
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
# TernaryLoRALinear (from real_data_domain_experts)
# ============================================================================

class TernaryLoRALinear(nn.Module):
    """LoRA with frozen Grassmannian A and STE-ternary B."""
    def __init__(self, base_linear: nn.Linear, rank: int = 16,
                 scale: float = 20.0, a_init: mx.array = None):
        super().__init__()
        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]

        self.linear = base_linear

        if a_init is not None:
            self.lora_a = a_init
        else:
            s = 1.0 / math.sqrt(in_features)
            self.lora_a = mx.random.uniform(
                low=-s, high=s, shape=(in_features, rank)
            )

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


def apply_ternary_lora(model, rank, scale, a_matrices_per_layer):
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
            if a_key in a_matrices_per_layer:
                a_np = a_matrices_per_layer[a_key]
                a_mx = mx.array(a_np).astype(mx.bfloat16)
            else:
                a_mx = None

            lora = TernaryLoRALinear(module, rank=rank, scale=scale, a_init=a_mx)
            lora_updates.append((key, lora))
            count += 1

        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    log(f"  Applied TernaryLoRA (r={rank}) to {count} layers")
    return model


def save_adapter(model, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        params[name] = mx.array(p)
    mx.eval(params)
    mx.savez(str(path / "adapter.npz"), **params)
    log(f"  Saved adapter: {len(params)} tensors to {path}")


# ============================================================================
# Pre-merge composition (from behavioral_eval_routed)
# ============================================================================

def load_skeleton():
    skeleton_path = ADAPTERS_DIR / "grassmannian_skeleton.npz"
    return dict(np.load(str(skeleton_path)))


def load_adapter_from_path(path):
    adapter = dict(mx.load(str(path / "adapter.npz")))
    log(f"  Loaded adapter from {path} ({len(adapter)} tensors)")
    return adapter


def premerge_single_adapter(model, skeleton, adapter, domain, scale):
    """Pre-merge a single adapter into model weights: W_new = W_base + scale * B^T @ A^T"""
    all_domain_names = ["medical", "code", "math", "legal", "finance"]
    n_layers = len(model.model.layers)
    merge_count = 0
    di = all_domain_names.index(domain)

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
# Data preparation
# ============================================================================

def prepare_cpt_data(domain):
    """Extract raw domain text from SFT training data for CPT.

    For CPT, we strip the instruction/response format and use only the
    response text (which contains the actual domain knowledge). We also
    concatenate instruction + response as continuous text since CPT should
    model the full domain text distribution.
    """
    train_path = DATA_DIR / domain / "train.jsonl"
    cpt_texts = []

    with open(train_path) as f:
        for line in f:
            text = json.loads(line)["text"]
            # Extract the response portion (domain knowledge)
            if "### Response:" in text:
                response = text.split("### Response:")[1].strip()
                if len(response) > 20:  # Skip very short responses
                    cpt_texts.append(response)

    log(f"  CPT data for {domain}: {len(cpt_texts)} texts")
    return cpt_texts


def prepare_sft_data(domain):
    """Load SFT training data (instruction-response format)."""
    train_path = DATA_DIR / domain / "train.jsonl"
    sft_texts = []
    with open(train_path) as f:
        for line in f:
            sft_texts.append(json.loads(line)["text"])
    log(f"  SFT data for {domain}: {len(sft_texts)} texts")
    return sft_texts


def extract_prompts_with_answers(domain, n_prompts=10):
    """Load validation prompts with reference answers for behavioral eval."""
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
# Behavioral evaluation metrics (from behavioral_eval_framework)
# ============================================================================

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


def eval_coherence(text):
    """Check if text is coherent (not degenerate/repetitive/empty).

    Returns True if coherent, False if incoherent.
    Incoherence indicators:
      - Empty or very short (<20 chars)
      - Highly repetitive (same token repeated many times)
      - No real words
    """
    if len(text.strip()) < 20:
        return False

    # Check for degenerate repetition
    words = text.split()
    if len(words) < 3:
        return False

    # Check if >50% of words are the same
    from collections import Counter
    word_counts = Counter(words)
    most_common_count = word_counts.most_common(1)[0][1]
    if most_common_count / len(words) > 0.5:
        return False

    # Check for char-level repetition (e.g., "aaaaaa...")
    if len(set(text.strip())) < 5:
        return False

    return True


def evaluate_prose_response(generated_text, reference_text, domain):
    """Evaluate a single prose domain response."""
    factual = eval_factual_recall(generated_text, reference_text)
    coherent = eval_coherence(generated_text)

    result = {
        "domain": domain,
        "score": factual["recall"],
        "factual_recall": factual["recall"],
        "factual_precision": factual["precision"],
        "factual_f1": factual["f1"],
        "coherent": coherent,
        "generated_len": len(generated_text),
        "ref_facts": factual["ref_facts"],
        "matched_facts": factual["matched"],
    }
    return result


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
# Phase 1: Train CPT adapters
# ============================================================================

def phase_train_cpt(domain):
    """Train a CPT adapter for one domain using raw text CLM objective."""
    log(f"\n{'=' * 70}")
    log(f"PHASE 1: TRAIN CPT ADAPTER - {domain.upper()}")
    log(f"{'=' * 70}")
    t0 = time.time()

    # Load model
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)

    # Load Grassmannian skeleton for this domain
    skeleton = dict(np.load(str(ADAPTERS_DIR / "grassmannian_skeleton.npz")))
    all_domain_names = ["medical", "code", "math", "legal", "finance"]
    di = all_domain_names.index(domain)

    a_matrices = {}
    n_layers = len(model.model.layers)
    for li in range(n_layers):
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_{di}"
            if skey in skeleton:
                a_matrices[(li, key)] = skeleton[skey]

    del skeleton
    gc.collect()

    # Apply LoRA
    model = apply_ternary_lora(model, LORA_RANK, LORA_SCALE, a_matrices)
    del a_matrices
    gc.collect()

    # Freeze everything except lora_b
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.unfreeze(keys=["lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"  Trainable params: {trainable:,}")

    # Prepare CPT data: raw domain text (response only, no instruction format)
    cpt_texts = prepare_cpt_data(domain)

    # Tokenize
    train_tokens = []
    for text in cpt_texts:
        toks = tokenizer.encode(text)
        if len(toks) > 2:
            train_tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))
    log(f"  {len(train_tokens)} training sequences for CPT")

    # Training loop (CLM on raw text)
    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    gc.disable()
    for step in range(TRAIN_ITERS):
        idx = step % len(train_tokens)
        tokens = train_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        loss_val = loss.item()
        losses.append(loss_val)

        if (step + 1) % 50 == 0 or step == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            log(f"    Step {step+1}/{TRAIN_ITERS}: loss={loss_val:.4f} (avg50={avg:.4f})")

    gc.enable()
    gc.collect()

    train_time = time.time() - t0
    first_50 = sum(losses[:50]) / 50
    last_50 = sum(losses[-50:]) / 50
    converged = last_50 < first_50 * 0.95

    log(f"  Done in {train_time:.1f}s. Loss: {first_50:.4f} -> {last_50:.4f} "
        f"({'converged' if converged else 'NOT converged'})")

    # Save adapter
    save_adapter(model, CPT_ADAPTERS_DIR / domain)

    result = {
        "train_time_s": round(train_time, 1),
        "first_50_avg_loss": round(first_50, 4),
        "last_50_avg_loss": round(last_50, 4),
        "converged": converged,
        "trainable_params": trainable,
        "n_train_sequences": len(train_tokens),
    }

    log_memory(f"post-cpt-train-{domain}")
    del model, tokenizer, optimizer, train_tokens
    cleanup()
    return result


# ============================================================================
# Phase 2: Generate with base, SFT adapter, and CPT adapter
# ============================================================================

def phase_generate(prompts_by_domain, config_name, adapter_dir=None):
    """Generate responses for all domains.

    config_name: "base", "sft", or "cpt"
    adapter_dir: path to adapter directory (None for base)
    """
    log(f"\n{'=' * 70}")
    log(f"PHASE 2: GENERATE WITH {config_name.upper()}")
    log(f"{'=' * 70}")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    if adapter_dir is not None:
        # Pre-merge adapter into model
        skeleton = load_skeleton()
        base_weights = save_base_weights(model)

        for domain in DOMAINS:
            adapter_path = adapter_dir / domain
            if adapter_path.exists():
                adapter = load_adapter_from_path(adapter_path)
                # Determine scale based on domain
                scale = LORA_SCALE
                model = premerge_single_adapter(model, skeleton, adapter, domain, scale)
                del adapter

        # NOTE: For per-domain generation, we need to swap adapters.
        # Since we generate domain by domain, let's do it properly.
        # Restore and re-merge per domain.
        restore_base_weights(model, base_weights)

    mx.random.seed(SEED)
    np.random.seed(SEED)

    results = {}

    if adapter_dir is not None:
        skeleton = load_skeleton()
        base_weights = save_base_weights(model)

        for domain in DOMAINS:
            restore_base_weights(model, base_weights)

            adapter_path = adapter_dir / domain
            if adapter_path.exists():
                adapter = load_adapter_from_path(adapter_path)
                model = premerge_single_adapter(model, skeleton, adapter, domain, LORA_SCALE)
                del adapter

            domain_results = []
            for i, prompt_data in enumerate(prompts_by_domain[domain]):
                formatted = format_prompt(prompt_data["instruction"])
                generated = generate_text(model, tokenizer, formatted,
                                          max_tokens=MAX_NEW_TOKENS)
                domain_results.append(generated)
                log(f"  [{config_name}][{domain}][{i}] {len(generated)} chars")
            results[domain] = domain_results

        del skeleton, base_weights
    else:
        # Base model: generate directly
        for domain in DOMAINS:
            domain_results = []
            for i, prompt_data in enumerate(prompts_by_domain[domain]):
                formatted = format_prompt(prompt_data["instruction"])
                generated = generate_text(model, tokenizer, formatted,
                                          max_tokens=MAX_NEW_TOKENS)
                domain_results.append(generated)
                log(f"  [{config_name}][{domain}][{i}] {len(generated)} chars")
            results[domain] = domain_results

    elapsed = time.time() - t0
    del model, tokenizer
    cleanup()
    log_memory(f"post-gen-{config_name}")
    log(f"  {config_name} generation: {elapsed:.1f}s")
    return results, elapsed


# ============================================================================
# Phase 3: Evaluate all configurations
# ============================================================================

def phase_evaluate(prompts_by_domain, generations_dict):
    """Evaluate all configs (base, sft, cpt) with behavioral metrics."""
    log(f"\n{'=' * 70}")
    log(f"PHASE 3: BEHAVIORAL EVALUATION")
    log(f"{'=' * 70}")
    t0 = time.time()

    all_evals = {}

    for config_name, generations in generations_dict.items():
        config_evals = {}
        for domain in DOMAINS:
            domain_evals = []
            for i, (prompt_data, gen_text) in enumerate(zip(
                    prompts_by_domain[domain], generations[domain])):
                result = evaluate_prose_response(
                    gen_text, prompt_data["response"], domain)
                result["prompt"] = prompt_data["instruction"][:100]
                result["generated_preview"] = gen_text[:200]
                domain_evals.append(result)

            scores = [r["score"] for r in domain_evals]
            mean_score = np.mean(scores) if scores else 0.0
            coherent_count = sum(1 for r in domain_evals if r["coherent"])
            coherent_pct = coherent_count / len(domain_evals) * 100 if domain_evals else 0

            log(f"\n  [{config_name}][{domain}]")
            log(f"    Mean factual recall: {mean_score:.4f}")
            log(f"    Coherent: {coherent_count}/{len(domain_evals)} ({coherent_pct:.0f}%)")

            config_evals[domain] = {
                "per_prompt": domain_evals,
                "mean_score": float(mean_score),
                "coherent_count": coherent_count,
                "coherent_pct": float(coherent_pct),
                "n_prompts": len(domain_evals),
            }

        all_evals[config_name] = config_evals

    elapsed = time.time() - t0
    log(f"\n  Evaluation: {elapsed:.1f}s")
    return all_evals, elapsed


# ============================================================================
# Main orchestrator
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("CPT vs SFT ADAPTERS: KNOWLEDGE INJECTION EXPERIMENT")
    log("=" * 70)
    log(f"Model: {MODEL_ID}")
    log(f"Domains: {DOMAINS}")
    log(f"Prompts/domain: {NUM_PROMPTS_PER_DOMAIN}")
    log(f"Max tokens: {MAX_NEW_TOKENS}")
    log(f"LoRA scale: {LORA_SCALE}")
    log(f"Train iters: {TRAIN_ITERS}")
    log_memory("start")

    # ---------------------------------------------------------------
    # Phase 1: Train CPT adapters (one per domain)
    # ---------------------------------------------------------------
    train_results = {}
    total_train_time = 0
    for domain in DOMAINS:
        result = phase_train_cpt(domain)
        train_results[domain] = result
        total_train_time += result["train_time_s"]

    log(f"\nTotal CPT training time: {total_train_time:.1f}s")

    # K3 check: training time
    k3_pass = total_train_time < 7200  # 2 hours
    k3_converged = all(r["converged"] for r in train_results.values())
    log(f"K3 pre-check: time={total_train_time:.0f}s (<7200s: {'PASS' if k3_pass else 'FAIL'}), "
        f"converged={'PASS' if k3_converged else 'FAIL'}")

    # ---------------------------------------------------------------
    # Phase 2: Load eval prompts
    # ---------------------------------------------------------------
    log("\nLoading evaluation prompts...")
    prompts_by_domain = {}
    for domain in DOMAINS:
        prompts = extract_prompts_with_answers(domain, NUM_PROMPTS_PER_DOMAIN)
        prompts_by_domain[domain] = prompts
        log(f"  {domain}: {len(prompts)} prompts loaded")

    # ---------------------------------------------------------------
    # Phase 3: Generate with base model
    # ---------------------------------------------------------------
    base_gen, base_time = phase_generate(prompts_by_domain, "base", adapter_dir=None)

    # ---------------------------------------------------------------
    # Phase 4: Generate with SFT adapters (existing)
    # ---------------------------------------------------------------
    sft_gen, sft_time = phase_generate(prompts_by_domain, "sft", adapter_dir=ADAPTERS_DIR)

    # ---------------------------------------------------------------
    # Phase 5: Generate with CPT adapters (newly trained)
    # ---------------------------------------------------------------
    cpt_gen, cpt_time = phase_generate(prompts_by_domain, "cpt", adapter_dir=CPT_ADAPTERS_DIR)

    # ---------------------------------------------------------------
    # Phase 6: Evaluate all configurations
    # ---------------------------------------------------------------
    all_evals, eval_time = phase_evaluate(prompts_by_domain, {
        "base": base_gen,
        "sft": sft_gen,
        "cpt": cpt_gen,
    })

    # ============================================================================
    # Kill criteria assessment
    # ============================================================================
    log(f"\n{'=' * 70}")
    log("KILL CRITERIA ASSESSMENT")
    log(f"{'=' * 70}")

    comparison = {}
    cpt_worse_than_sft = 0
    cpt_incoherent_pct = []

    for domain in DOMAINS:
        base_score = all_evals["base"][domain]["mean_score"]
        sft_score = all_evals["sft"][domain]["mean_score"]
        cpt_score = all_evals["cpt"][domain]["mean_score"]

        cpt_vs_sft_delta = cpt_score - sft_score
        cpt_vs_base_delta = cpt_score - base_score
        sft_vs_base_delta = sft_score - base_score

        cpt_coherent = all_evals["cpt"][domain]["coherent_pct"]
        cpt_incoherent_pct.append(100 - cpt_coherent)

        if cpt_score < sft_score:
            cpt_worse_than_sft += 1

        improvement_vs_sft = (cpt_vs_sft_delta / max(sft_score, 0.001)) * 100

        comp = {
            "base_score": round(base_score, 4),
            "sft_score": round(sft_score, 4),
            "cpt_score": round(cpt_score, 4),
            "cpt_vs_sft_delta": round(cpt_vs_sft_delta, 4),
            "cpt_vs_sft_pct": round(improvement_vs_sft, 1),
            "cpt_vs_base_delta": round(cpt_vs_base_delta, 4),
            "sft_vs_base_delta": round(sft_vs_base_delta, 4),
            "cpt_coherent_pct": round(cpt_coherent, 1),
        }
        comparison[domain] = comp

        log(f"\n  {domain.upper()}:")
        log(f"    Base:  {base_score:.4f}")
        log(f"    SFT:   {sft_score:.4f} (vs base: {sft_vs_base_delta:+.4f})")
        log(f"    CPT:   {cpt_score:.4f} (vs base: {cpt_vs_base_delta:+.4f}, "
            f"vs SFT: {cpt_vs_sft_delta:+.4f} = {improvement_vs_sft:+.1f}%)")
        log(f"    CPT coherence: {cpt_coherent:.0f}%")

    # K1: CPT worse than SFT on BOTH legal AND medical -> KILL
    k1_pass = cpt_worse_than_sft < len(DOMAINS)  # Not ALL domains worse
    k1_result = "pass" if k1_pass else "fail"

    # K2: CPT incoherent >20%
    max_incoherent = max(cpt_incoherent_pct) if cpt_incoherent_pct else 0
    k2_pass = max_incoherent <= 20
    k2_result = "pass" if k2_pass else "fail"

    # K3: Training time + convergence
    k3_overall = k3_pass and k3_converged
    k3_result = "pass" if k3_overall else "fail"

    log(f"\n  K1 (#672): CPT worse than SFT on {cpt_worse_than_sft}/{len(DOMAINS)} domains "
        f"-> {'PASS' if k1_pass else 'KILL (worse on ALL domains)'}")
    log(f"  K2 (#673): Max incoherent = {max_incoherent:.0f}% "
        f"-> {'PASS' if k2_pass else 'KILL (>20% incoherent)'}")
    log(f"  K3 (#674): Time={total_train_time:.0f}s, converged={k3_converged} "
        f"-> {'PASS' if k3_overall else 'KILL'}")

    overall = "supported" if (k1_pass and k2_pass and k3_overall) else "killed"
    log(f"\n  OVERALL: {overall.upper()}")

    # ============================================================================
    # Save results
    # ============================================================================
    total_time = time.time() - t0

    results = {
        "experiment": "cpt_vs_sft_prose_adapters",
        "hypothesis": "CPT adapters improve prose domain behavioral quality vs SFT",
        "status": overall,
        "total_time_s": round(total_time, 1),
        "kill_criteria": {
            "K1_cpt_vs_sft": {
                "result": k1_result,
                "cpt_worse_count": cpt_worse_than_sft,
                "total_domains": len(DOMAINS),
            },
            "K2_coherence": {
                "result": k2_result,
                "max_incoherent_pct": round(max_incoherent, 1),
                "threshold": 20,
            },
            "K3_training": {
                "result": k3_result,
                "total_train_time_s": round(total_train_time, 1),
                "all_converged": k3_converged,
                "threshold_s": 7200,
            },
        },
        "training": train_results,
        "comparison": comparison,
        "per_domain_details": {},
        "generation_times": {
            "base_s": round(base_time, 1),
            "sft_s": round(sft_time, 1),
            "cpt_s": round(cpt_time, 1),
        },
    }

    # Add per-domain detail (without full text to keep JSON manageable)
    for domain in DOMAINS:
        domain_detail = {}
        for config in ["base", "sft", "cpt"]:
            evals = all_evals[config][domain]
            domain_detail[config] = {
                "mean_score": evals["mean_score"],
                "coherent_pct": evals["coherent_pct"],
                "per_prompt_scores": [r["score"] for r in evals["per_prompt"]],
                "per_prompt_coherent": [r["coherent"] for r in evals["per_prompt"]],
                "sample_generations": [
                    {
                        "prompt": r["prompt"],
                        "preview": r["generated_preview"],
                        "score": r["score"],
                        "coherent": r["coherent"],
                    }
                    for r in evals["per_prompt"][:3]  # First 3 samples
                ],
            }
        results["per_domain_details"][domain] = domain_detail

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total experiment time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
