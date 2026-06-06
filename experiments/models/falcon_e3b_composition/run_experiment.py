#!/usr/bin/env python3
"""Falcon-E-3B LoRA Adapter Composition Experiment.

Tests whether Falcon-E-3B (3B ternary, Llama-compatible, 999MB) supports
LoRA adapter composition and can compete with Qwen2.5-3B on domain benchmarks.

Kill criteria:
  K1 (#532): Falcon-E-3B doesn't support LoRA -> KILL
  K2 (#533): Still loses >4/6 vs Qwen-3B -> KILL
  K3 (#534): Total memory >6GB -> KILL

Success criteria:
  S1 (#54): Beats Qwen-3B on >=3/6 benchmarks at <4GB -> unlocks competitive ternary composition

Prior art:
  - BitNet-2B competitive benchmark: KILLED vs Qwen-3B (4/6 loss, 10.98GB memory)
  - Key failures: (1) MMLU factual knowledge gap, (2) uniform composition hurts math/legal,
    (3) BitNet bf16 unpack = 5.35GB. Falcon-E-3B has instruction-tuned variant.
  - Falcon-E-3B: 53.17 avg benchmark (arxiv from tiiuae/onebitllms)

Architecture:
  Base: tiiuae/Falcon-E-3B (ternary 1.58-bit, Llama-compatible)
    - hidden_size=2048, 32 layers, GQA 16:2, SwiGLU intermediate=13312
    - 999MB packed, ~6.1GB unpacked to bf16
  LoRA: rank-16 on q/v/o projections (attention-only for speed)
  Training: 200 iters/adapter, instruction format, seq_len=256
  Composition: uniform 1/N pre-merge into bf16 weights

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
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse data from prior experiment
DATA_DIR = EXPERIMENT_DIR.parent / "real_data_25_domain_adapters" / "data"

# Model IDs
MODEL_ID_FALCON = "mlx-community/Falcon-E-3B-Instruct-1.58bit"
MODEL_ID_QWEN = "mlx-community/Qwen2.5-3B-Instruct-4bit"

# LoRA config
LORA_RANK = 16
LORA_SCALE = 20.0
TRAIN_ITERS = 200
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
VAL_BATCHES = 25
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = len(DOMAINS)

# Target layers for LoRA (attention only for 3B model -- faster training)
TARGET_KEYS = ["self_attn.q_proj", "self_attn.v_proj", "self_attn.o_proj"]

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
# Ternary unpacking (same as BitNet -- Falcon-E uses identical BitLinear)
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
# LoRA application and training
# ============================================================================

def apply_lora(model):
    """Apply LoRA to target attention projections."""
    lora_count = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        for full_key in TARGET_KEYS:
            parts = full_key.split(".")
            module = layer
            for p in parts:
                module = getattr(module, p, None)
                if module is None:
                    break
            if module is not None and isinstance(module, nn.Linear):
                lora = LoRALinear.from_base(module, r=LORA_RANK, scale=LORA_SCALE)
                # Navigate to parent and set
                parent = layer
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], lora)
                lora_count += 1
    mx.eval(model.parameters())
    log(f"  Applied LoRA to {lora_count} layers")
    return model


def freeze_base_train_lora(model):
    """Freeze base weights, unfreeze only LoRA parameters."""
    model.freeze()
    model.unfreeze(keys=["lora_a", "lora_b"], strict=False)
    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    total = sum(p.size for _, p in tree_flatten(model.parameters()))
    log(f"  Trainable: {trainable:,} ({100*trainable/total:.4f}%)")
    return trainable, total


def load_training_data(domain, tokenizer, max_samples=500):
    """Load training data from existing preprocessed JSONL files."""
    train_path = DATA_DIR / domain / "train.jsonl"
    val_path = DATA_DIR / domain / "valid.jsonl"

    def load_jsonl(path, max_n):
        texts = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= max_n:
                    break
                item = json.loads(line)
                texts.append(item["text"])
        return texts

    train_texts = load_jsonl(train_path, max_samples)
    val_texts = load_jsonl(val_path, 50)
    log(f"  Data {domain}: {len(train_texts)} train, {len(val_texts)} val")
    return train_texts, val_texts


def tokenize_batch(texts, tokenizer, max_len=256):
    """Tokenize a batch of texts, truncating to max_len."""
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        if len(tokens) >= 2:
            all_tokens.append(tokens)
    return all_tokens


def get_batch(all_tokens, batch_idx):
    """Get a single training example as mx.array."""
    idx = batch_idx % len(all_tokens)
    return mx.array([all_tokens[idx]])


def compute_ppl(model, tokenizer, texts, max_batches=25, max_len=256):
    """Compute perplexity on a set of texts."""
    tokens_list = tokenize_batch(texts, tokenizer, max_len)
    if not tokens_list:
        return float("inf")

    total_loss = 0.0
    total_tokens = 0
    n_batches = min(max_batches, len(tokens_list))

    for i in range(n_batches):
        batch = mx.array([tokens_list[i]])
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        loss = nn.losses.cross_entropy(logits, targets, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += targets.size
        del logits, loss
        if i % 10 == 0:
            gc.collect()

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(min(avg_loss, 20))


# ============================================================================
# Phase 1: Train adapters
# ============================================================================

def phase_train_adapters():
    """Train 5 domain adapters on Falcon-E-3B."""
    log("\n" + "=" * 70)
    log("PHASE 1: TRAIN 5 DOMAIN ADAPTERS ON FALCON-E-3B")
    log("=" * 70)

    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    all_train_results = {}

    for domain in DOMAINS:
        log(f"\n--- Training adapter: {domain} ---")
        t0 = time.time()
        mx.reset_peak_memory()

        # Load model fresh for each adapter (CODING_GUIDELINES: separate phases)
        model, tokenizer = load(MODEL_ID_FALCON)
        model = replace_bitlinear_with_linear(model)
        model = apply_lora(model)
        freeze_base_train_lora(model)

        # Load data
        train_texts, val_texts = load_training_data(domain, tokenizer)
        train_tokens = tokenize_batch(train_texts, tokenizer, MAX_SEQ_LENGTH)

        # Base PPL (before training)
        base_ppl = compute_ppl(model, tokenizer, val_texts, max_batches=VAL_BATCHES)
        log(f"  Base PPL ({domain}): {base_ppl:.2f}")

        # Train
        optimizer = opt.Adam(learning_rate=LEARNING_RATE)

        def loss_fn(model, tokens):
            logits = model(tokens[:, :-1])
            targets = tokens[:, 1:]
            return nn.losses.cross_entropy(logits, targets, reduction="mean")

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        losses = []

        gc.disable()
        for step in range(TRAIN_ITERS):
            batch = get_batch(train_tokens, step)
            loss, grads = loss_and_grad(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            loss_val = loss.item()
            losses.append(loss_val)
            if (step + 1) % 50 == 0:
                log(f"  Step {step+1}/{TRAIN_ITERS}: loss={loss_val:.4f}")
        gc.enable()
        gc.collect()

        # Trained PPL
        trained_ppl = compute_ppl(model, tokenizer, val_texts, max_batches=VAL_BATCHES)
        log(f"  Trained PPL ({domain}): {trained_ppl:.2f} (base: {base_ppl:.2f}, delta: {(trained_ppl-base_ppl)/base_ppl*100:.1f}%)")

        # Save adapter weights (LoRA A and B only)
        adapter_dir = ADAPTERS_DIR / domain
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_weights = {}
        for name, param in tree_flatten(model.trainable_parameters()):
            adapter_weights[name] = param
        mx.savez(str(adapter_dir / "adapter.npz"), **adapter_weights)
        adapter_file = adapter_dir / "adapter.npz"
        adapter_size = os.path.getsize(adapter_file) if adapter_file.exists() else 0
        log(f"  Saved adapter to {adapter_dir} ({adapter_size/1024:.1f} KB)")

        elapsed = time.time() - t0
        peak_mem = mx.get_peak_memory() / 1e9

        all_train_results[domain] = {
            "base_ppl": base_ppl,
            "trained_ppl": trained_ppl,
            "ppl_improvement_pct": (base_ppl - trained_ppl) / base_ppl * 100,
            "final_loss": losses[-1],
            "peak_memory_gb": peak_mem,
            "time_s": elapsed,
        }

        log(f"  {domain} done in {elapsed:.1f}s, peak={peak_mem:.2f}GB")
        cleanup(model, tokenizer, optimizer)

    return all_train_results


# ============================================================================
# Phase 2: Compose and evaluate
# ============================================================================

def premerge_adapters(model, adapter_paths, weights):
    """Pre-merge LoRA adapters into model weights.

    W_composed = W_base + sum_d w_d * scale * (x @ lora_a_d) @ lora_b_d
    In weight space: W += w_d * scale * lora_b_d^T @ lora_a_d^T
    Wait -- LoRALinear computes: y = W @ x + scale * (x @ lora_a) @ lora_b
    So the effective delta: dW = scale * lora_b^T @ lora_a^T applied to W
    Actually: y = Wx + scale * (xA)B = (W + scale * B^T A^T)x... no.
    Let me check: LoRALinear forward = linear(x) + scale * dropout(x @ lora_a) @ lora_b
    So y = Wx + scale * x @ A @ B = (W^T + scale * A @ B)^T x... in row form:
    y = x @ W^T + scale * x @ A @ B
    So effective weight (in row form): W_eff^T = W^T + scale * A @ B
    Delta to W (column form): dW = (scale * A @ B)^T = scale * B^T @ A^T
    """
    merge_count = 0
    for domain, path in adapter_paths.items():
        w = weights.get(domain, 1.0 / len(adapter_paths))
        if w < 1e-6:
            continue

        adapter = dict(mx.load(str(path)))

        for name, param in adapter.items():
            # name like "model.layers.0.self_attn.q_proj.lora_a"
            parts = name.split(".")
            if parts[-1] == "lora_a":
                # Find corresponding lora_b
                b_name = ".".join(parts[:-1]) + ".lora_b"
                if b_name not in adapter:
                    continue
                lora_a = param  # shape: (in_features, r)
                lora_b = adapter[b_name]  # shape: (r, out_features)

                # Navigate to the weight matrix
                # name = model.layers.X.self_attn.Y_proj.lora_a
                # We need model.layers.X.self_attn.Y_proj.weight
                module = model
                for p in parts[:-1]:  # up to Y_proj
                    if p.isdigit():
                        module = module[int(p)]
                    else:
                        module = getattr(module, p, None)
                    if module is None:
                        break

                if module is None:
                    continue

                # module is a LoRALinear or after fuse, nn.Linear
                # For base model (no LoRA applied), we need the weight directly
                if hasattr(module, "linear"):
                    weight = module.linear.weight
                elif hasattr(module, "weight"):
                    weight = module.weight
                else:
                    continue

                # delta_W = scale * B^T @ A^T (in column-major / weight form)
                delta = w * LORA_SCALE * (lora_b.T @ lora_a.T)

                if hasattr(module, "linear"):
                    module.linear.weight = weight + delta
                else:
                    module.weight = weight + delta
                merge_count += 1

        del adapter
        gc.collect()

    mx.eval(model.parameters())
    log(f"  Pre-merged {len(adapter_paths)} adapters into {merge_count} weight matrices")
    return model


def phase_eval_falcon_composed(gsm8k, mmlu):
    """Evaluate Falcon-E-3B with composed adapters."""
    log("\n" + "=" * 70)
    log("EVALUATING: Falcon-E-3B + 5 adapters (uniform 1/N pre-merge)")
    log("=" * 70)
    t0 = time.time()
    mx.reset_peak_memory()

    model, tokenizer = load(MODEL_ID_FALCON)
    model = replace_bitlinear_with_linear(model)
    log_memory("falcon-composed-loaded")

    # Pre-merge adapters
    adapter_paths = {d: ADAPTERS_DIR / d / "adapter.npz" for d in DOMAINS}
    weights = {d: 1.0 / N_DOMAINS for d in DOMAINS}
    model = premerge_adapters(model, adapter_paths, weights)
    model.freeze()
    log_memory("falcon-composed-merged")

    gsm8k_results = eval_gsm8k("falcon-composed", model, tokenizer, gsm8k,
                                format_gsm8k_prompt_falcon)
    mmlu_results = eval_mmlu("falcon-composed", model, tokenizer, mmlu,
                             format_mmlu_prompt_falcon)

    peak_mem = mx.get_peak_memory() / 1e9
    elapsed = time.time() - t0
    log(f"\nFalcon composed total: {elapsed:.1f}s, peak memory: {peak_mem:.2f}GB")

    results = {
        "gsm8k": gsm8k_results,
        "mmlu": mmlu_results,
        "peak_memory_gb": peak_mem,
        "time_s": elapsed,
    }
    cleanup(model, tokenizer)
    return results


def phase_eval_falcon_base(gsm8k, mmlu):
    """Evaluate Falcon-E-3B base (no adapters)."""
    log("\n" + "=" * 70)
    log("EVALUATING: Falcon-E-3B Base (no adapters)")
    log("=" * 70)
    t0 = time.time()
    mx.reset_peak_memory()

    model, tokenizer = load(MODEL_ID_FALCON)
    # Note: Falcon-E-3B-Instruct uses BitLinear natively but mlx-lm handles it
    # We keep it as-is for inference (no LoRA, no unpack needed for eval)
    # Actually we DO need to unpack for fair comparison (same forward pass)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("falcon-base-loaded")

    gsm8k_results = eval_gsm8k("falcon-base", model, tokenizer, gsm8k,
                                format_gsm8k_prompt_falcon)
    mmlu_results = eval_mmlu("falcon-base", model, tokenizer, mmlu,
                             format_mmlu_prompt_falcon)

    peak_mem = mx.get_peak_memory() / 1e9
    elapsed = time.time() - t0
    log(f"\nFalcon base total: {elapsed:.1f}s, peak memory: {peak_mem:.2f}GB")

    results = {
        "gsm8k": gsm8k_results,
        "mmlu": mmlu_results,
        "peak_memory_gb": peak_mem,
        "time_s": elapsed,
    }
    cleanup(model, tokenizer)
    return results


def phase_eval_qwen(gsm8k, mmlu):
    """Evaluate Qwen2.5-3B-Instruct-4bit."""
    log("\n" + "=" * 70)
    log(f"EVALUATING: Qwen2.5-3B-Instruct-4bit")
    log("=" * 70)
    t0 = time.time()
    mx.reset_peak_memory()

    model, tokenizer = load(MODEL_ID_QWEN)
    log_memory("qwen-loaded")

    gsm8k_results = eval_gsm8k("qwen-3b", model, tokenizer, gsm8k,
                                format_gsm8k_prompt_chatml)
    mmlu_results = eval_mmlu("qwen-3b", model, tokenizer, mmlu,
                             format_mmlu_prompt_chatml)

    peak_mem = mx.get_peak_memory() / 1e9
    elapsed = time.time() - t0
    log(f"\nQwen total: {elapsed:.1f}s, peak memory: {peak_mem:.2f}GB")

    results = {
        "gsm8k": gsm8k_results,
        "mmlu": mmlu_results,
        "peak_memory_gb": peak_mem,
        "time_s": elapsed,
    }
    cleanup(model, tokenizer)
    return results


# ============================================================================
# Prompt formatting
# ============================================================================

def format_gsm8k_prompt_falcon(question):
    """Falcon-E-3B uses Llama chat template. Use instruction format."""
    return (
        f"### Instruction:\n"
        f"Solve the following math problem step by step. "
        f"Show your work and give the final numerical answer after ####.\n\n"
        f"{question}\n\n"
        f"### Response:\n"
    )


def format_gsm8k_prompt_chatml(question):
    return (
        f"<|im_start|>system\nYou are a helpful math tutor. Solve problems step by step "
        f"and give the final numerical answer after ####.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def format_mmlu_prompt_falcon(question, choices):
    choice_labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(f"{l}. {c}" for l, c in zip(choice_labels, choices))
    return (
        f"### Instruction:\n"
        f"Answer the following multiple choice question. "
        f"Reply with just the letter (A, B, C, or D).\n\n"
        f"{question}\n\n{choices_text}\n\n"
        f"### Response:\n"
    )


def format_mmlu_prompt_chatml(question, choices):
    choice_labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(f"{l}. {c}" for l, c in zip(choice_labels, choices))
    return (
        f"<|im_start|>system\nYou are a helpful assistant. Answer multiple choice questions "
        f"with just the letter (A, B, C, or D).<|im_end|>\n"
        f"<|im_start|>user\n{question}\n\n{choices_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ============================================================================
# Data loading
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
# Answer extraction
# ============================================================================

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


# ============================================================================
# Generation & evaluation
# ============================================================================

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


def eval_gsm8k(model_name, model, tokenizer, problems, format_fn):
    log(f"\n  [GSM8K] Evaluating {model_name}...")
    t0 = time.time()
    correct = 0
    total = len(problems)

    for i, prob in enumerate(problems):
        prompt = format_fn(prob["question"])
        generated = generate_text(model, tokenizer, prompt, max_tokens=MAX_NEW_TOKENS_GSM8K)
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
    log(f"  [GSM8K] {model_name}: {correct}/{total} = {accuracy:.3f} ({elapsed:.1f}s)")
    return {"accuracy": accuracy, "correct": correct, "total": total, "time_s": elapsed}


def eval_mmlu(model_name, model, tokenizer, mmlu_data, format_fn):
    log(f"\n  [MMLU] Evaluating {model_name}...")
    results = {}
    choice_labels = ["A", "B", "C", "D"]

    for domain in DOMAINS:
        questions = mmlu_data[domain]
        if not questions:
            results[domain] = {"accuracy": 0, "correct": 0, "total": 0}
            continue

        correct = 0
        total = len(questions)

        for i, q in enumerate(questions):
            prompt = format_fn(q["question"], q["choices"])
            generated = generate_text(model, tokenizer, prompt, max_tokens=MAX_NEW_TOKENS_MMLU)
            predicted = extract_mmlu_answer(generated)
            gt_label = choice_labels[q["answer"]]
            if predicted == gt_label:
                correct += 1

        accuracy = correct / total if total > 0 else 0
        results[domain] = {"accuracy": accuracy, "correct": correct, "total": total}
        log(f"    MMLU {domain}: {correct}/{total} = {accuracy:.3f}")

    return results


# ============================================================================
# Analysis
# ============================================================================

def analyze_results(all_results):
    log("\n" + "=" * 70)
    log("ANALYSIS")
    log("=" * 70)

    benchmarks = ["gsm8k", "mmlu_medical", "mmlu_code", "mmlu_math", "mmlu_legal", "mmlu_finance"]
    model_scores = {}

    for model_name, res in all_results.items():
        if res is None:
            continue
        scores = {}
        scores["gsm8k"] = res["gsm8k"]["accuracy"]
        for domain in DOMAINS:
            scores[f"mmlu_{domain}"] = res["mmlu"][domain]["accuracy"]
        model_scores[model_name] = scores

    # Print table
    log("\n  ACCURACY COMPARISON:")
    header = f"  {'Benchmark':<15}"
    for name in model_scores:
        header += f"  {name:<20}"
    log(header)
    log("  " + "-" * (15 + 22 * len(model_scores)))

    for bench in benchmarks:
        row = f"  {bench:<15}"
        for name in model_scores:
            val = model_scores[name].get(bench, None)
            if val is not None:
                row += f"  {val:<20.3f}"
            else:
                row += f"  {'N/A':<20}"
        log(row)

    # Memory
    log("\n  MEMORY COMPARISON:")
    for name, res in all_results.items():
        if res is not None:
            log(f"  {name}: {res['peak_memory_gb']:.2f} GB peak")

    analysis = {"model_scores": model_scores, "kill_criteria": {}, "success_criteria": {}}

    # K2: Falcon composed loses >4/6 vs Qwen-3B
    if "falcon_composed" in model_scores and "qwen_3b" in model_scores:
        falcon = model_scores["falcon_composed"]
        qwen = model_scores["qwen_3b"]
        worse_count = sum(1 for b in benchmarks if falcon.get(b, 0) < qwen.get(b, 0))
        k2_kill = worse_count > 4
        analysis["kill_criteria"]["k2_533"] = {
            "result": "fail" if k2_kill else "pass",
            "evidence": f"Falcon loses {worse_count}/6 vs Qwen (threshold >4)",
            "worse_benchmarks": [b for b in benchmarks if falcon.get(b, 0) < qwen.get(b, 0)],
            "better_benchmarks": [b for b in benchmarks if falcon.get(b, 0) >= qwen.get(b, 0)],
        }
        log(f"\n  K2 (#533): Falcon loses {worse_count}/6 vs Qwen -> {'KILL' if k2_kill else 'PASS'}")

    # K3: Total memory > 6GB
    if "falcon_composed" in all_results and all_results["falcon_composed"] is not None:
        falcon_mem = all_results["falcon_composed"]["peak_memory_gb"]
        k3_kill = falcon_mem > 6.0
        analysis["kill_criteria"]["k3_534"] = {
            "result": "fail" if k3_kill else "pass",
            "evidence": f"Falcon composed peak memory: {falcon_mem:.2f} GB (threshold 6.0 GB)",
        }
        log(f"  K3 (#534): Falcon memory={falcon_mem:.2f}GB -> {'KILL' if k3_kill else 'PASS'}")

    # S1: Beats Qwen on >=3/6 at <4GB
    if "falcon_composed" in model_scores and "qwen_3b" in model_scores:
        falcon = model_scores["falcon_composed"]
        qwen = model_scores["qwen_3b"]
        beats_count = sum(1 for b in benchmarks if falcon.get(b, 0) > qwen.get(b, 0))
        falcon_mem = all_results["falcon_composed"]["peak_memory_gb"]
        s1_pass = beats_count >= 3 and falcon_mem < 4.0
        analysis["success_criteria"]["s1_54"] = {
            "result": "pass" if s1_pass else "fail",
            "evidence": f"Beats Qwen on {beats_count}/6, memory={falcon_mem:.2f}GB",
        }
        log(f"  S1 (#54): Beats Qwen on {beats_count}/6 at {falcon_mem:.2f}GB -> {'PASS' if s1_pass else 'FAIL'}")

    return analysis


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log_memory("start")

    # Phase 1: Train adapters (skip if all adapters already exist)
    all_adapters_exist = all(
        (ADAPTERS_DIR / d / "adapter.npz").exists() for d in DOMAINS
    )
    if all_adapters_exist:
        log("All adapters already exist, skipping training phase.")
        train_results = {"note": "skipped - adapters pre-trained"}
    else:
        train_results = phase_train_adapters()

    # Phase 2: Load benchmark data
    log("\n" + "=" * 70)
    log("LOADING BENCHMARK DATA")
    log("=" * 70)
    gsm8k = load_gsm8k_data(GSM8K_N)
    mmlu = load_mmlu_data(MMLU_N_PER_DOMAIN)

    all_results = {}

    # Phase 3: Evaluate Falcon base
    all_results["falcon_base"] = phase_eval_falcon_base(gsm8k, mmlu)

    # Phase 4: Evaluate Falcon + composed adapters
    all_results["falcon_composed"] = phase_eval_falcon_composed(gsm8k, mmlu)

    # Phase 5: Evaluate Qwen competitor
    all_results["qwen_3b"] = phase_eval_qwen(gsm8k, mmlu)

    # Phase 6: Analysis
    analysis = analyze_results(all_results)

    # Save
    total_time = time.time() - t0
    output = {
        "experiment": "falcon_e3b_composition",
        "models": {
            "falcon_base": {"id": MODEL_ID_FALCON, "description": "Falcon-E-3B-Instruct 1.58-bit, no adapters"},
            "falcon_composed": {"id": MODEL_ID_FALCON, "description": "Falcon-E-3B + 5 domain LoRA adapters, uniform 1/5 pre-merge"},
            "qwen_3b": {"id": MODEL_ID_QWEN, "description": "Qwen2.5-3B-Instruct 4-bit quantized"},
        },
        "training": train_results,
        "results": all_results,
        "analysis": analysis,
        "params": {
            "gsm8k_n": GSM8K_N,
            "mmlu_n_per_domain": MMLU_N_PER_DOMAIN,
            "max_new_tokens_gsm8k": MAX_NEW_TOKENS_GSM8K,
            "max_new_tokens_mmlu": MAX_NEW_TOKENS_MMLU,
            "temperature": 0.0,
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
            "train_iters": TRAIN_ITERS,
            "composition": "uniform 1/N pre-merge",
            "n_adapters": N_DOMAINS,
            "target_keys": TARGET_KEYS,
        },
        "total_time_s": total_time,
        "k1_lora_supported": True,
    }

    RESULTS_FILE.write_text(json.dumps(output, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log("  K1 (#532): LoRA supported on Falcon-E-3B -> PASS")
    for k, v in analysis.get("kill_criteria", {}).items():
        log(f"  {k}: {v['result'].upper()} - {v['evidence']}")
    for k, v in analysis.get("success_criteria", {}).items():
        log(f"  {k}: {v['result'].upper()} - {v['evidence']}")


if __name__ == "__main__":
    main()
