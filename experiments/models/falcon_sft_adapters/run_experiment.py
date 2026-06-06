#!/usr/bin/env python3
"""Falcon-E-3B SFT Adapter Experiment.

Tests whether SFT-loss (response-only masking) LoRA adapters fix the degradation
observed with NTP-trained adapters on instruction-tuned base models.

Kill criteria:
  K1 (#562): SFT adapters still degrade base performance on >3/5 benchmarks
  K2 (#563): Composed model worse than best single adapter on >3/5 benchmarks

Prior results (NTP adapters, exp_falcon_e3b_composition):
  Falcon base: GSM8K=0.44, MMLU avg=0.54
  NTP composed: GSM8K=0.36, MMLU avg=0.43 (DEGRADED)

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

# Prior experiment data (instruction-formatted JSONL)
DATA_DIR = EXPERIMENT_DIR.parent / "real_data_25_domain_adapters" / "data"

# Model
MODEL_ID = "mlx-community/Falcon-E-3B-Instruct-1.58bit"

# LoRA config (same as prior experiment for fair comparison)
LORA_RANK = 16
LORA_SCALE = 20.0
TRAIN_ITERS = 300  # Increased from 200: SFT has fewer loss terms per sample
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
VAL_BATCHES = 25
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = len(DOMAINS)

# Target layers for LoRA
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

# Response marker for SFT masking
RESPONSE_MARKER = "### Response:\n"


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
# Model loading (reused from falcon_e3b_composition)
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


def apply_lora(model):
    """Apply LoRA to target attention projections."""
    lora_count = 0
    for layer in model.model.layers:
        for full_key in TARGET_KEYS:
            parts = full_key.split(".")
            module = layer
            for p in parts:
                module = getattr(module, p, None)
                if module is None:
                    break
            if module is not None and isinstance(module, nn.Linear):
                lora = LoRALinear.from_base(module, r=LORA_RANK, scale=LORA_SCALE)
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


# ============================================================================
# SFT data loading with response masking
# ============================================================================

def load_sft_data(domain, tokenizer, max_samples=500):
    """Load training data and create SFT masks (response-only loss).

    The data is formatted as:
        ### Instruction:\n<instruction>\n\n### Response:\n<response>

    We tokenize the full text, then find the position of "### Response:\n"
    and create a mask that is 0 for instruction tokens and 1 for response tokens.
    """
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


def tokenize_with_sft_mask(text, tokenizer, max_len=256):
    """Tokenize text and return (tokens, loss_mask) for SFT.

    loss_mask[t] = 1 if token t is a RESPONSE token (should be predicted)
    loss_mask[t] = 0 if token t is an INSTRUCTION token (should be ignored)
    """
    # Find the response marker in the text
    response_idx = text.find(RESPONSE_MARKER)
    if response_idx < 0:
        # Fallback: treat entire text as response (NTP behavior)
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        return tokens, [1] * len(tokens)

    # Split into instruction part and response part
    instruction_part = text[:response_idx + len(RESPONSE_MARKER)]
    # response_part = text[response_idx + len(RESPONSE_MARKER):]

    # Tokenize instruction prefix to find boundary
    instruction_tokens = tokenizer.encode(instruction_part, add_special_tokens=True)
    instruction_len = len(instruction_tokens)

    # Tokenize full text
    full_tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(full_tokens) > max_len:
        full_tokens = full_tokens[:max_len]

    # Create mask: 0 for instruction tokens, 1 for response tokens
    mask = [0] * min(instruction_len, len(full_tokens))
    mask += [1] * (len(full_tokens) - len(mask))

    return full_tokens, mask


def prepare_sft_batches(texts, tokenizer, max_len=256):
    """Prepare all training examples with SFT masks."""
    batches = []
    n_response_tokens = 0
    n_total_tokens = 0
    for text in texts:
        tokens, mask = tokenize_with_sft_mask(text, tokenizer, max_len)
        if len(tokens) >= 4:  # Need at least a few tokens
            batches.append((tokens, mask))
            n_response_tokens += sum(mask)
            n_total_tokens += len(tokens)
    if batches:
        response_ratio = n_response_tokens / max(n_total_tokens, 1)
        log(f"  SFT masking: {response_ratio:.1%} response tokens, "
            f"{1-response_ratio:.1%} instruction tokens masked out")
    return batches


def get_sft_batch(batches, batch_idx):
    """Get a single SFT training example."""
    idx = batch_idx % len(batches)
    tokens, mask = batches[idx]
    return mx.array([tokens]), mx.array([mask])


# ============================================================================
# SFT loss function
# ============================================================================

def sft_loss_fn(model, tokens, mask):
    """Compute cross-entropy loss ONLY on response tokens.

    Args:
        model: the model
        tokens: [1, T] token IDs
        mask: [1, T] binary mask (1 = response token, 0 = instruction token)

    Returns:
        Scalar loss averaged over response tokens only.
    """
    logits = model(tokens[:, :-1])  # [1, T-1, vocab]
    targets = tokens[:, 1:]  # [1, T-1]
    response_mask = mask[:, 1:]  # shift mask to align with targets

    # Per-token cross-entropy
    per_token_loss = nn.losses.cross_entropy(logits, targets, reduction="none")  # [1, T-1]

    # Mask: only response tokens contribute
    masked_loss = per_token_loss * response_mask
    n_response = mx.maximum(response_mask.sum(), mx.array(1.0))
    loss = masked_loss.sum() / n_response

    return loss


# ============================================================================
# NTP loss (baseline comparison)
# ============================================================================

def ntp_loss_fn(model, tokens, mask):
    """Standard NTP loss on ALL tokens (for comparison). Mask is ignored."""
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


# ============================================================================
# Evaluation functions (same as falcon_e3b_composition)
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
# Phase 1: Train SFT adapters
# ============================================================================

def phase_train_adapters():
    """Train 5 domain SFT adapters on Falcon-E-3B."""
    log("\n" + "=" * 70)
    log("PHASE 1: TRAIN 5 DOMAIN SFT ADAPTERS ON FALCON-E-3B")
    log("=" * 70)

    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    all_train_results = {}

    for domain in DOMAINS:
        log(f"\n--- Training SFT adapter: {domain} ---")
        t0 = time.time()
        mx.reset_peak_memory()

        # Load model fresh (CODING_GUIDELINES: separate scopes)
        model, tokenizer = load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)
        model = apply_lora(model)
        freeze_base_train_lora(model)

        # Load and prepare SFT data
        train_texts, val_texts = load_sft_data(domain, tokenizer)
        train_batches = prepare_sft_batches(train_texts, tokenizer, MAX_SEQ_LENGTH)
        val_batches_data = prepare_sft_batches(val_texts, tokenizer, MAX_SEQ_LENGTH)

        if not train_batches:
            log(f"  WARNING: No valid training data for {domain}, skipping")
            cleanup(model, tokenizer)
            continue

        # Compute base validation loss (before training)
        base_val_loss = 0.0
        n_val = min(VAL_BATCHES, len(val_batches_data))
        for i in range(n_val):
            tokens, mask = get_sft_batch(val_batches_data, i)
            loss = sft_loss_fn(model, tokens, mask)
            mx.eval(loss)
            base_val_loss += loss.item()
            del loss, tokens, mask
        base_val_loss /= max(n_val, 1)
        base_ppl = math.exp(min(base_val_loss, 20))
        log(f"  Base SFT val loss: {base_val_loss:.4f} (PPL: {base_ppl:.2f})")

        # Train with SFT loss
        optimizer = opt.Adam(learning_rate=LEARNING_RATE)
        loss_and_grad = nn.value_and_grad(model, sft_loss_fn)
        losses = []

        gc.disable()
        for step in range(TRAIN_ITERS):
            tokens, mask = get_sft_batch(train_batches, step)
            loss, grads = loss_and_grad(model, tokens, mask)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            loss_val = loss.item()
            losses.append(loss_val)
            if (step + 1) % 50 == 0:
                log(f"  Step {step+1}/{TRAIN_ITERS}: sft_loss={loss_val:.4f}")
        gc.enable()
        gc.collect()

        # Trained validation loss
        trained_val_loss = 0.0
        for i in range(n_val):
            tokens, mask = get_sft_batch(val_batches_data, i)
            loss = sft_loss_fn(model, tokens, mask)
            mx.eval(loss)
            trained_val_loss += loss.item()
            del loss, tokens, mask
        trained_val_loss /= max(n_val, 1)
        trained_ppl = math.exp(min(trained_val_loss, 20))
        log(f"  Trained SFT val loss: {trained_val_loss:.4f} (PPL: {trained_ppl:.2f})")
        log(f"  PPL improvement: {base_ppl:.2f} -> {trained_ppl:.2f} "
            f"({(base_ppl - trained_ppl) / base_ppl * 100:.1f}%)")

        # Save adapter weights
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
            "sft_response_ratio": "see log",
            "peak_memory_gb": peak_mem,
            "time_s": elapsed,
        }

        log(f"  {domain} done in {elapsed:.1f}s, peak={peak_mem:.2f}GB")
        cleanup(model, tokenizer, optimizer)

    return all_train_results


# ============================================================================
# Phase 2: Evaluate base model
# ============================================================================

def eval_gsm8k(model_name, model, tokenizer, problems):
    log(f"\n  [GSM8K] Evaluating {model_name}...")
    t0 = time.time()
    correct = 0
    total = len(problems)

    for i, prob in enumerate(problems):
        prompt = format_gsm8k_prompt(prob["question"])
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


def eval_mmlu(model_name, model, tokenizer, mmlu_data):
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
            prompt = format_mmlu_prompt(q["question"], q["choices"])
            generated = generate_text(model, tokenizer, prompt, max_tokens=MAX_NEW_TOKENS_MMLU)
            predicted = extract_mmlu_answer(generated)
            gt_label = choice_labels[q["answer"]]
            if predicted == gt_label:
                correct += 1

        accuracy = correct / total if total > 0 else 0
        results[domain] = {"accuracy": accuracy, "correct": correct, "total": total}
        log(f"    MMLU {domain}: {correct}/{total} = {accuracy:.3f}")

    return results


def phase_eval_base(gsm8k, mmlu):
    """Evaluate Falcon-E-3B base (no adapters)."""
    log("\n" + "=" * 70)
    log("PHASE 2: EVALUATE FALCON-E-3B BASE (no adapters)")
    log("=" * 70)
    t0 = time.time()
    mx.reset_peak_memory()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("base-loaded")

    gsm8k_results = eval_gsm8k("falcon-base", model, tokenizer, gsm8k)
    mmlu_results = eval_mmlu("falcon-base", model, tokenizer, mmlu)

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


# ============================================================================
# Phase 3: Evaluate single SFT adapters
# ============================================================================

def load_adapter_into_model(model, adapter_path):
    """Load saved LoRA adapter weights into model."""
    adapter = dict(mx.load(str(adapter_path)))
    model.load_weights(list(adapter.items()), strict=False)
    mx.eval(model.parameters())
    return model


def phase_eval_single_adapters(gsm8k, mmlu):
    """Evaluate each SFT adapter individually."""
    log("\n" + "=" * 70)
    log("PHASE 3: EVALUATE INDIVIDUAL SFT ADAPTERS")
    log("=" * 70)

    single_results = {}

    for domain in DOMAINS:
        adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
        if not adapter_path.exists():
            log(f"  Adapter {domain} not found, skipping")
            continue

        log(f"\n--- Evaluating SFT adapter: {domain} ---")
        t0 = time.time()
        mx.reset_peak_memory()

        model, tokenizer = load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)
        model = apply_lora(model)
        model = load_adapter_into_model(model, adapter_path)
        model.freeze()
        log_memory(f"single-{domain}-loaded")

        gsm8k_results = eval_gsm8k(f"sft-{domain}", model, tokenizer, gsm8k)
        mmlu_results = eval_mmlu(f"sft-{domain}", model, tokenizer, mmlu)

        peak_mem = mx.get_peak_memory() / 1e9
        elapsed = time.time() - t0

        single_results[domain] = {
            "gsm8k": gsm8k_results,
            "mmlu": mmlu_results,
            "peak_memory_gb": peak_mem,
            "time_s": elapsed,
        }

        log(f"  {domain} adapter eval done in {elapsed:.1f}s, peak={peak_mem:.2f}GB")
        cleanup(model, tokenizer)

    return single_results


# ============================================================================
# Phase 4: Evaluate composed model (uniform 1/N pre-merge)
# ============================================================================

def premerge_adapters(model, adapter_paths, weights):
    """Pre-merge LoRA adapters into model weights."""
    merge_count = 0
    for domain, path in adapter_paths.items():
        w = weights.get(domain, 1.0 / len(adapter_paths))
        if w < 1e-6:
            continue

        adapter = dict(mx.load(str(path)))

        for name, param in adapter.items():
            parts = name.split(".")
            if parts[-1] == "lora_a":
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

                if hasattr(module, "linear"):
                    weight = module.linear.weight
                elif hasattr(module, "weight"):
                    weight = module.weight
                else:
                    continue

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


def phase_eval_composed(gsm8k, mmlu):
    """Evaluate Falcon-E-3B with uniform 1/N pre-merged SFT adapters."""
    log("\n" + "=" * 70)
    log("PHASE 4: EVALUATE COMPOSED MODEL (uniform 1/N pre-merge)")
    log("=" * 70)
    t0 = time.time()
    mx.reset_peak_memory()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    log_memory("composed-loaded")

    adapter_paths = {d: ADAPTERS_DIR / d / "adapter.npz" for d in DOMAINS}
    weights = {d: 1.0 / N_DOMAINS for d in DOMAINS}
    model = premerge_adapters(model, adapter_paths, weights)
    model.freeze()
    log_memory("composed-merged")

    gsm8k_results = eval_gsm8k("sft-composed", model, tokenizer, gsm8k)
    mmlu_results = eval_mmlu("sft-composed", model, tokenizer, mmlu)

    peak_mem = mx.get_peak_memory() / 1e9
    elapsed = time.time() - t0
    log(f"\nComposed total: {elapsed:.1f}s, peak memory: {peak_mem:.2f}GB")

    results = {
        "gsm8k": gsm8k_results,
        "mmlu": mmlu_results,
        "peak_memory_gb": peak_mem,
        "time_s": elapsed,
    }
    cleanup(model, tokenizer)
    return results


# ============================================================================
# Phase 5: Evaluate routed model (best single adapter per domain)
# ============================================================================

def phase_eval_routed(gsm8k, mmlu, single_results):
    """Simulate routing: for each domain benchmark, use the domain-specific adapter."""
    log("\n" + "=" * 70)
    log("PHASE 5: EVALUATE ROUTED MODEL (best adapter per domain)")
    log("=" * 70)

    # For MMLU, use domain-specific adapter for each domain
    # For GSM8K, use the math adapter
    routed_results = {
        "gsm8k": single_results.get("math", {}).get("gsm8k", {"accuracy": 0, "correct": 0, "total": 0}),
        "mmlu": {},
    }

    for domain in DOMAINS:
        if domain in single_results and "mmlu" in single_results[domain]:
            # Use the domain-specific adapter's score for that domain's MMLU
            routed_results["mmlu"][domain] = single_results[domain]["mmlu"][domain]
        else:
            routed_results["mmlu"][domain] = {"accuracy": 0, "correct": 0, "total": 0}

    log("  Routed results (simulated oracle routing):")
    log(f"    GSM8K (math adapter): {routed_results['gsm8k']['accuracy']:.3f}")
    for domain in DOMAINS:
        log(f"    MMLU {domain}: {routed_results['mmlu'][domain]['accuracy']:.3f}")

    return routed_results


# ============================================================================
# Analysis
# ============================================================================

def analyze_results(base_results, single_results, composed_results, routed_results):
    log("\n" + "=" * 70)
    log("ANALYSIS: SFT vs NTP vs BASE")
    log("=" * 70)

    # Prior NTP results (from exp_falcon_e3b_composition)
    ntp_composed = {
        "gsm8k": 0.36,
        "mmlu_medical": 0.30, "mmlu_code": 0.50, "mmlu_math": 0.55,
        "mmlu_legal": 0.35, "mmlu_finance": 0.45,
    }

    benchmarks = ["gsm8k", "mmlu_medical", "mmlu_code", "mmlu_math", "mmlu_legal", "mmlu_finance"]

    def get_scores(results):
        scores = {}
        scores["gsm8k"] = results["gsm8k"]["accuracy"]
        for domain in DOMAINS:
            scores[f"mmlu_{domain}"] = results["mmlu"][domain]["accuracy"]
        return scores

    base_scores = get_scores(base_results)
    composed_scores = get_scores(composed_results)
    routed_scores = get_scores(routed_results)

    # Best single adapter per benchmark
    best_single = {}
    for bench in benchmarks:
        best_val = 0
        best_domain = None
        for domain in DOMAINS:
            if domain in single_results:
                scores = get_scores(single_results[domain])
                if scores.get(bench, 0) > best_val:
                    best_val = scores[bench]
                    best_domain = domain
        best_single[bench] = {"accuracy": best_val, "domain": best_domain}

    # Print comparison table
    log("\n  ACCURACY COMPARISON:")
    log(f"  {'Benchmark':<15} {'Base':<8} {'NTP-comp':<10} {'SFT-comp':<10} {'Routed':<10} {'BestSingle':<12}")
    log("  " + "-" * 70)

    for bench in benchmarks:
        base_v = base_scores.get(bench, 0)
        ntp_v = ntp_composed.get(bench, 0)
        sft_v = composed_scores.get(bench, 0)
        route_v = routed_scores.get(bench, 0)
        best_v = best_single[bench]["accuracy"]
        log(f"  {bench:<15} {base_v:<8.3f} {ntp_v:<10.3f} {sft_v:<10.3f} {route_v:<10.3f} {best_v:<12.3f}")

    # K1: SFT adapters degrade base on >3/5 benchmarks
    sft_degrades = 0
    sft_improves = 0
    for bench in benchmarks:
        if routed_scores.get(bench, 0) < base_scores.get(bench, 0) - 0.01:
            sft_degrades += 1
        elif routed_scores.get(bench, 0) > base_scores.get(bench, 0) + 0.01:
            sft_improves += 1

    k1_kill = sft_degrades > 3
    log(f"\n  K1 (#562): SFT routed degrades {sft_degrades}/6, improves {sft_improves}/6 -> {'KILL' if k1_kill else 'PASS'}")

    # K2: Composed worse than best single on >3/5
    composed_worse = 0
    for bench in benchmarks:
        if composed_scores.get(bench, 0) < best_single[bench]["accuracy"] - 0.01:
            composed_worse += 1
    k2_kill = composed_worse > 3
    log(f"  K2 (#563): Composed worse than best single on {composed_worse}/6 -> {'KILL' if k2_kill else 'PASS'}")

    # Delta from NTP
    log("\n  SFT vs NTP IMPROVEMENT:")
    for bench in benchmarks:
        sft_v = composed_scores.get(bench, 0)
        ntp_v = ntp_composed.get(bench, 0)
        delta = sft_v - ntp_v
        log(f"    {bench}: NTP={ntp_v:.3f} -> SFT={sft_v:.3f} (delta={delta:+.3f})")

    analysis = {
        "base_scores": base_scores,
        "ntp_composed_scores": ntp_composed,
        "sft_composed_scores": composed_scores,
        "sft_routed_scores": routed_scores,
        "best_single_scores": {b: v["accuracy"] for b, v in best_single.items()},
        "k1_562": {
            "result": "fail" if k1_kill else "pass",
            "degrades": sft_degrades,
            "improves": sft_improves,
            "evidence": f"SFT routed degrades {sft_degrades}/6 benchmarks (threshold >3)",
        },
        "k2_563": {
            "result": "fail" if k2_kill else "pass",
            "composed_worse": composed_worse,
            "evidence": f"Composed worse than best single on {composed_worse}/6 (threshold >3)",
        },
    }

    return analysis


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log_memory("start")

    # Phase 1: Train SFT adapters
    all_adapters_exist = all(
        (ADAPTERS_DIR / d / "adapter.npz").exists() for d in DOMAINS
    )
    if all_adapters_exist:
        log("All SFT adapters already exist, skipping training phase.")
        train_results = {"note": "skipped - adapters pre-trained"}
    else:
        train_results = phase_train_adapters()

    # Load benchmark data
    log("\n" + "=" * 70)
    log("LOADING BENCHMARK DATA")
    log("=" * 70)
    gsm8k = load_gsm8k_data(GSM8K_N)
    mmlu = load_mmlu_data(MMLU_N_PER_DOMAIN)

    # Phase 2: Evaluate base
    base_results = phase_eval_base(gsm8k, mmlu)

    # Phase 3: Evaluate single SFT adapters
    single_results = phase_eval_single_adapters(gsm8k, mmlu)

    # Phase 4: Evaluate composed (uniform 1/N)
    composed_results = phase_eval_composed(gsm8k, mmlu)

    # Phase 5: Evaluate routed (oracle routing)
    routed_results = phase_eval_routed(gsm8k, mmlu, single_results)

    # Phase 6: Analysis
    analysis = analyze_results(base_results, single_results, composed_results, routed_results)

    # Save results
    total_time = time.time() - t0
    output = {
        "experiment": "falcon_sft_adapters",
        "model": MODEL_ID,
        "training": train_results,
        "base": base_results,
        "single_adapters": single_results,
        "composed": composed_results,
        "routed": routed_results,
        "analysis": analysis,
        "params": {
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
            "train_iters": TRAIN_ITERS,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "loss_type": "SFT (response-only masking)",
            "response_marker": RESPONSE_MARKER,
            "target_keys": TARGET_KEYS,
            "composition": "uniform 1/N pre-merge",
            "gsm8k_n": GSM8K_N,
            "mmlu_n_per_domain": MMLU_N_PER_DOMAIN,
        },
        "prior_ntp_results": {
            "gsm8k": 0.36,
            "mmlu_medical": 0.30, "mmlu_code": 0.50, "mmlu_math": 0.55,
            "mmlu_legal": 0.35, "mmlu_finance": 0.45,
            "note": "From exp_falcon_e3b_composition (NTP loss, same model)"
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
    for k in ["k1_562", "k2_563"]:
        v = analysis[k]
        log(f"  {k}: {v['result'].upper()} - {v['evidence']}")


if __name__ == "__main__":
    main()
