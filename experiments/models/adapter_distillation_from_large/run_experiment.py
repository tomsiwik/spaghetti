#!/usr/bin/env python3
"""
Experiment: Adapter Distillation from Large Teacher

Distill knowledge from Qwen2.5-7B (teacher) into ternary LoRA adapters on
BitNet-2B-4T (student) via sequence-level knowledge distillation.

Kill criteria:
  K1: Distilled adapter PPL must be >= 5% better than self-supervised baseline
  K2: Peak memory must stay under 40GB (teacher + student fit in 48GB)

Approach:
  Phase 1: Teacher (Qwen2.5-7B-4bit) generates enhanced domain text from prompts
  Phase 2: Student (BitNet-2B-4T + LoRA) trains on teacher-generated text
  Phase 3: Evaluate distilled vs self-supervised adapter PPL on same val set

The teacher and student never coexist in memory -- teacher generates first,
then is unloaded before student training begins.
"""

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety
device = mx.device_info()
total_mem = device["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
DISTILL_DATA_DIR = EXPERIMENT_DIR / "distill_data"
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Use the same data dir as the baseline experiment for validation
BASELINE_DATA_DIR = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"

# Models
TEACHER_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
STUDENT_ID = "microsoft/BitNet-b1.58-2B-4T"

# Training config (match baseline for fair comparison)
LORA_RANK = 16
LORA_SCALE = 20.0
TRAIN_ITERS = 200
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
VAL_BATCHES = 25

# Distillation config
TEACHER_GEN_SAMPLES = 100       # samples per domain (100 keeps generation under 30 min)
TEACHER_MAX_NEW_TOKENS = 128    # max tokens per generation (shorter = faster)
TEACHER_TEMPERATURE = 0.7       # lower than 1.0 for focused text
TEACHER_TOP_P = 0.9

# Domains (same 5 as baseline)
DOMAINS = {
    "python": {
        "hf_dataset": "iamtarun/python_code_instructions_18k_alpaca",
        "text_key": "output",
        "prompt_key": "instruction",
        "max_samples_train": 100,
        "max_samples_val": 50,
        "system_prompt": "You are an expert Python programmer. Write clean, correct Python code.",
    },
    "math": {
        "hf_dataset": "gsm8k",
        "hf_subset": "main",
        "text_key": "answer",
        "prompt_key": "question",
        "max_samples_train": 100,
        "max_samples_val": 50,
        "system_prompt": "You are a math tutor. Solve the problem step by step.",
    },
    "medical": {
        "hf_dataset": "medalpaca/medical_meadow_medical_flashcards",
        "text_key": "output",
        "prompt_key": "input",
        "max_samples_train": 100,
        "max_samples_val": 50,
        "system_prompt": "You are a medical expert. Provide accurate medical information.",
    },
    "legal": {
        "hf_dataset": "jonathanli/law-stack-exchange",
        "text_key": "body",
        "prompt_key": None,  # no prompt key, use first sentence as prompt
        "max_samples_train": 100,
        "max_samples_val": 50,
        "system_prompt": "You are a legal expert. Provide clear legal analysis.",
    },
    "creative": {
        "hf_dataset": "roneneldan/TinyStories",
        "text_key": "text",
        "prompt_key": None,
        "max_samples_train": 100,
        "max_samples_val": 50,
        "system_prompt": "You are a creative writer. Write engaging stories.",
    },
}

# Self-supervised baseline PPL (from bitnet_2b_real_composition/results.json)
BASELINE_PPLS = {
    "python": 2.217879865748791,
    "math": 3.6036968713946225,
    "medical": 4.7421944845839885,
    "legal": 16.530172045991478,
    "creative": 4.923663481280721,
}


def log_memory(label=""):
    """Print current memory usage."""
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    """Release MLX memory."""
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ===========================================================================
# Ternary unpacking (from baseline experiment)
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
    from mlx_lm.models.bitlinear_layers import BitLinear
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
    print(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


# ===========================================================================
# LoRA utilities (from baseline experiment)
# ===========================================================================
def apply_lora_to_model(model, rank=16, scale=1.0):
    from mlx_lm.tuner.lora import LoRALinear
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        lora_updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora_layer = LoRALinear.from_base(module, r=rank, scale=scale, dropout=0.0)
                lora_updates.append((key, lora_layer))
                count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    print(f"  Applied LoRA (r={rank}) to {count} layers")
    return model


def zero_lora_params(model):
    from mlx_lm.tuner.lora import LoRALinear
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                in_dims = module.lora_a.shape[0]
                r = module.lora_a.shape[1]
                scale = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(
                    low=-scale, high=scale, shape=module.lora_a.shape
                )
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def get_lora_params(model):
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_a" in name or "lora_b" in name:
            params[name] = mx.array(p)
    mx.eval(params)
    return params


def save_adapter(model, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    params = get_lora_params(model)
    mx.savez(str(path / "adapter.npz"), **params)
    print(f"  Saved adapter: {len(params)} tensors to {path}")


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_weights(model, adapter_params: dict, scale: float = 1.0):
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


# ===========================================================================
# Phase 1: Teacher generates distillation data
# ===========================================================================
def extract_prompts(domain_name, domain_config):
    """Extract prompts from domain training data for teacher generation."""
    from datasets import load_dataset as hf_load

    kwargs = {}
    if "hf_subset" in domain_config:
        kwargs["name"] = domain_config["hf_subset"]

    ds = hf_load(domain_config["hf_dataset"], **kwargs)
    if "train" in ds:
        split_data = ds["train"]
    else:
        split_data = ds[list(ds.keys())[0]]

    prompts = []
    text_key = domain_config["text_key"]
    prompt_key = domain_config.get("prompt_key")

    # Find actual text key
    if text_key not in split_data.column_names:
        for alt in ["text", "content", "output", "answer", "response", "question"]:
            if alt in split_data.column_names:
                text_key = alt
                break

    if prompt_key and prompt_key not in split_data.column_names:
        prompt_key = None

    for row in split_data:
        text = row[text_key]
        if not isinstance(text, str) or len(text.strip()) < 20:
            continue

        if prompt_key and prompt_key in row and isinstance(row[prompt_key], str) and len(row[prompt_key].strip()) > 5:
            prompt = row[prompt_key].strip()
        else:
            # Use first sentence as prompt
            sentences = text.strip().split(".")
            if len(sentences) >= 2:
                prompt = sentences[0].strip() + "."
            else:
                prompt = text[:100].strip()

        prompts.append(prompt)
        if len(prompts) >= domain_config["max_samples_train"]:
            break

    return prompts


def make_sampler(temperature=0.7, top_p=0.9):
    """Create a sampler function for mlx_lm generate."""
    from mlx_lm.sample_utils import make_sampler as _make_sampler
    return _make_sampler(temp=temperature, top_p=top_p)


def phase_teacher_generate():
    """Load teacher, generate domain text, save, unload."""
    from mlx_lm import load, generate

    print("\n[Phase 1] Teacher generation")
    print(f"  Teacher: {TEACHER_ID}")

    DISTILL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if distillation data already exists
    all_exist = all(
        (DISTILL_DATA_DIR / f"{domain}.jsonl").exists()
        for domain in DOMAINS
    )
    if all_exist:
        print("  Distillation data already exists, skipping generation")
        for domain in DOMAINS:
            path = DISTILL_DATA_DIR / f"{domain}.jsonl"
            n = sum(1 for _ in open(path))
            print(f"    {domain}: {n} samples")
        return None

    # Extract prompts first (before loading teacher to save memory)
    print("  Extracting prompts from domain datasets...")
    domain_prompts = {}
    for domain_name, config in DOMAINS.items():
        prompts = extract_prompts(domain_name, config)
        domain_prompts[domain_name] = prompts
        print(f"    {domain_name}: {len(prompts)} prompts")

    # Load teacher
    print(f"\n  Loading teacher model...")
    t0 = time.time()
    teacher_model, teacher_tokenizer = load(TEACHER_ID)
    load_time = time.time() - t0
    print(f"  Teacher loaded in {load_time:.1f}s")
    log_memory("teacher-loaded")

    peak_memory = mx.get_peak_memory() / 1e9

    # Create sampler for generation
    sampler = make_sampler(temperature=TEACHER_TEMPERATURE, top_p=TEACHER_TOP_P)

    # Generate for each domain
    for domain_name, config in DOMAINS.items():
        print(f"\n  --- Generating {domain_name} distillation data ---")
        output_path = DISTILL_DATA_DIR / f"{domain_name}.jsonl"

        if output_path.exists():
            print(f"    Already exists, skipping")
            continue

        prompts = domain_prompts[domain_name]
        system_prompt = config["system_prompt"]
        generated = []
        t_start = time.time()

        for i, prompt in enumerate(prompts):
            # Format as chat for instruct model
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            formatted = teacher_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Generate using mlx_lm.generate with sampler
            try:
                output = generate(
                    teacher_model,
                    teacher_tokenizer,
                    prompt=formatted,
                    max_tokens=TEACHER_MAX_NEW_TOKENS,
                    sampler=sampler,
                    verbose=False,
                )
                if isinstance(output, str) and len(output.strip()) > 20:
                    generated.append({
                        "prompt": prompt,
                        "text": output.strip(),
                    })
            except Exception as e:
                print(f"    Warning: generation failed for prompt {i}: {e}")
                continue

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                print(f"    {i+1}/{len(prompts)} ({rate:.1f} samples/s)")

        # Save
        with open(output_path, "w") as f:
            for item in generated:
                json.dump(item, f)
                f.write("\n")

        elapsed = time.time() - t_start
        print(f"    Generated {len(generated)}/{len(prompts)} samples in {elapsed:.0f}s")

        # Check peak memory
        current_peak = mx.get_peak_memory() / 1e9
        if current_peak > peak_memory:
            peak_memory = current_peak

    print(f"\n  Peak memory during generation: {peak_memory:.2f} GB")

    # Unload teacher
    print("  Unloading teacher...")
    cleanup(teacher_model, teacher_tokenizer)
    log_memory("teacher-unloaded")

    return peak_memory


# ===========================================================================
# Phase 2: Train student on distilled data
# ===========================================================================
def compute_ppl(model, tokenizer, data_path: Path, max_batches: int = 25):
    """Compute perplexity on validation data."""
    valid_path = data_path / "valid.jsonl"
    if not valid_path.exists():
        return float("inf")

    texts = []
    with open(valid_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    total_loss = 0.0
    total_tokens = 0

    for text in texts[:max_batches]:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:MAX_SEQ_LENGTH + 1]

        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]

        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)

        total_loss += loss.item()
        total_tokens += y.size
        del logits, loss, x, y

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 100))


def phase_train_distilled():
    """Load student, train on distilled data, evaluate."""
    from mlx_lm import load
    from mlx_lm.tuner.lora import LoRALinear

    print("\n[Phase 2] Student training on distilled data")
    print(f"  Student: {STUDENT_ID}")

    # Load student
    t0 = time.time()
    model, tokenizer = load(STUDENT_ID)
    print(f"  Student loaded in {time.time() - t0:.1f}s")

    # Unpack ternary
    print("  Unpacking ternary weights...")
    model = replace_bitlinear_with_linear(model)
    log_memory("student-unpacked")

    # Apply LoRA
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"  Trainable LoRA parameters: {trainable:,}")

    # Verify baseline data exists
    if not BASELINE_DATA_DIR.exists():
        print(f"  ERROR: Baseline data not found at {BASELINE_DATA_DIR}")
        print("  Run bitnet_2b_real_composition first to create validation data")
        return None

    # Train each domain adapter on distilled data
    distilled_ppls = {}
    self_supervised_ppls = {}
    train_results = {}
    peak_memory = mx.get_peak_memory() / 1e9

    for domain_name in DOMAINS:
        print(f"\n  --- Training {domain_name} (distilled) ---")

        # Reset LoRA
        zero_lora_params(model)

        # Load distilled data
        distill_path = DISTILL_DATA_DIR / f"{domain_name}.jsonl"
        if not distill_path.exists():
            print(f"  ERROR: No distillation data at {distill_path}")
            continue

        texts = []
        with open(distill_path) as f:
            for line in f:
                item = json.loads(line)
                texts.append(item["text"])

        # Tokenize
        train_tokens = []
        for text in texts:
            toks = tokenizer.encode(text)
            if len(toks) > 2:
                train_tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))

        print(f"    {len(train_tokens)} training sequences (distilled)")

        if len(train_tokens) < 10:
            print(f"    ERROR: Too few training sequences")
            continue

        # Training loop
        optimizer = opt.Adam(learning_rate=LEARNING_RATE)

        def loss_fn(model, x, y):
            logits = model(x)
            return nn.losses.cross_entropy(logits, y, reduction="mean")

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        t_start = time.time()
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
                print(f"      Step {step+1}/{TRAIN_ITERS}: loss={loss_val:.4f} (avg50={avg:.4f})")
        gc.enable()
        gc.collect()

        train_time = time.time() - t_start
        first_50 = sum(losses[:50]) / 50
        last_50 = sum(losses[-50:]) / 50
        converged = last_50 < first_50 * 0.95

        print(f"    Done in {train_time:.1f}s. Loss: {first_50:.4f} -> {last_50:.4f} "
              f"({'converged' if converged else 'NOT converged'})")

        train_results[domain_name] = {
            "train_time_s": round(train_time, 1),
            "first_50_avg_loss": round(first_50, 4),
            "last_50_avg_loss": round(last_50, 4),
            "converged": converged,
        }

        # Save adapter
        save_adapter(model, ADAPTERS_DIR / domain_name)

        # Evaluate on ORIGINAL validation data (same as baseline)
        val_data_dir = BASELINE_DATA_DIR / domain_name
        if val_data_dir.exists():
            ppl = compute_ppl(model, tokenizer, val_data_dir)
            distilled_ppls[domain_name] = ppl
            baseline = BASELINE_PPLS.get(domain_name, float("inf"))
            improvement = (baseline - ppl) / baseline * 100
            print(f"    Distilled PPL: {ppl:.2f} (baseline: {baseline:.2f}, {improvement:+.1f}%)")
        else:
            print(f"    WARNING: No validation data at {val_data_dir}")

        current_peak = mx.get_peak_memory() / 1e9
        if current_peak > peak_memory:
            peak_memory = current_peak

        # Clean up optimizer state between domains
        del optimizer, train_tokens, losses
        gc.collect()
        mx.clear_cache()

    log_memory("post-training")

    # Also train self-supervised baselines for fair comparison
    # (retrain on original data with same hyperparams to control for randomness)
    print("\n  --- Training self-supervised baselines (control) ---")
    for domain_name in DOMAINS:
        print(f"\n  --- Training {domain_name} (self-supervised control) ---")

        zero_lora_params(model)

        # Load original training data
        orig_data_dir = BASELINE_DATA_DIR / domain_name
        train_path = orig_data_dir / "train.jsonl"
        if not train_path.exists():
            print(f"    No original training data at {train_path}")
            self_supervised_ppls[domain_name] = BASELINE_PPLS.get(domain_name, float("inf"))
            continue

        texts = []
        with open(train_path) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
        # Limit to same number of samples as distilled for fair comparison
        texts = texts[:TEACHER_GEN_SAMPLES]

        train_tokens = []
        for text in texts:
            toks = tokenizer.encode(text)
            if len(toks) > 2:
                train_tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))

        print(f"    {len(train_tokens)} training sequences (self-supervised)")

        optimizer = opt.Adam(learning_rate=LEARNING_RATE)

        def loss_fn(model, x, y):
            logits = model(x)
            return nn.losses.cross_entropy(logits, y, reduction="mean")

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        t_start = time.time()
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
                print(f"      Step {step+1}/{TRAIN_ITERS}: loss={loss_val:.4f} (avg50={avg:.4f})")
        gc.enable()
        gc.collect()

        train_time = time.time() - t_start
        first_50 = sum(losses[:50]) / 50
        last_50 = sum(losses[-50:]) / 50

        print(f"    Done in {train_time:.1f}s. Loss: {first_50:.4f} -> {last_50:.4f}")

        # Evaluate
        val_data_dir = BASELINE_DATA_DIR / domain_name
        if val_data_dir.exists():
            ppl = compute_ppl(model, tokenizer, val_data_dir)
            self_supervised_ppls[domain_name] = ppl
            print(f"    Self-supervised PPL: {ppl:.2f}")

        del optimizer, train_tokens, losses
        gc.collect()
        mx.clear_cache()

    # Cleanup student
    cleanup(model, tokenizer)

    return {
        "distilled_ppls": distilled_ppls,
        "self_supervised_ppls": self_supervised_ppls,
        "train_results": train_results,
        "peak_memory_gb": round(peak_memory, 2),
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_total = time.time()

    results = {
        "experiment": "adapter_distillation_from_large",
        "teacher": TEACHER_ID,
        "student": STUDENT_ID,
        "lora_rank": LORA_RANK,
        "train_iters": TRAIN_ITERS,
        "teacher_temperature": TEACHER_TEMPERATURE,
        "domains": list(DOMAINS.keys()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=" * 70)
    print("Adapter Distillation from Large Teacher")
    print("=" * 70)
    log_memory("start")

    # Phase 1: Teacher generates distillation data
    teacher_peak = phase_teacher_generate()
    if teacher_peak is not None:
        results["teacher_peak_memory_gb"] = round(teacher_peak, 2)
    log_memory("after-teacher")

    # Phase 2: Train student on distilled data + self-supervised control
    train_output = phase_train_distilled()
    if train_output is None:
        results["error"] = "Training phase failed"
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        return

    results.update(train_output)

    # Phase 3: Analysis
    print("\n" + "=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70)

    distilled = train_output["distilled_ppls"]
    self_sup = train_output["self_supervised_ppls"]
    baseline = BASELINE_PPLS

    print(f"\n{'Domain':<12} {'Baseline':>10} {'Self-Sup':>10} {'Distilled':>10} {'Imp vs SS':>10} {'Imp vs BL':>10}")
    print("-" * 62)

    improvements_vs_ss = []
    improvements_vs_bl = []

    for domain in DOMAINS:
        bl = baseline.get(domain, float("inf"))
        ss = self_sup.get(domain, float("inf"))
        dist = distilled.get(domain, float("inf"))

        imp_ss = (ss - dist) / ss * 100 if ss < float("inf") and dist < float("inf") else 0
        imp_bl = (bl - dist) / bl * 100 if bl < float("inf") and dist < float("inf") else 0

        improvements_vs_ss.append(imp_ss)
        improvements_vs_bl.append(imp_bl)

        print(f"{domain:<12} {bl:>10.2f} {ss:>10.2f} {dist:>10.2f} {imp_ss:>+9.1f}% {imp_bl:>+9.1f}%")

    avg_imp_ss = sum(improvements_vs_ss) / len(improvements_vs_ss) if improvements_vs_ss else 0
    avg_imp_bl = sum(improvements_vs_bl) / len(improvements_vs_bl) if improvements_vs_bl else 0

    avg_distilled = sum(distilled.values()) / len(distilled) if distilled else float("inf")
    avg_self_sup = sum(self_sup.values()) / len(self_sup) if self_sup else float("inf")
    avg_baseline = sum(baseline.values()) / len(baseline)

    print(f"\n{'Average':<12} {avg_baseline:>10.2f} {avg_self_sup:>10.2f} {avg_distilled:>10.2f} {avg_imp_ss:>+9.1f}% {avg_imp_bl:>+9.1f}%")

    results["avg_distilled_ppl"] = round(avg_distilled, 4)
    results["avg_self_supervised_ppl"] = round(avg_self_sup, 4)
    results["avg_baseline_ppl"] = round(avg_baseline, 4)
    results["avg_improvement_vs_self_supervised_pct"] = round(avg_imp_ss, 2)
    results["avg_improvement_vs_baseline_pct"] = round(avg_imp_bl, 2)
    results["per_domain_improvement_vs_ss"] = {
        d: round(imp, 2) for d, imp in zip(DOMAINS, improvements_vs_ss)
    }

    # Kill criteria assessment
    print("\n" + "=" * 70)
    print("KILL CRITERIA ASSESSMENT")
    print("=" * 70)

    # K1: Distilled PPL >= 5% better than self-supervised
    # Compare against same-run self-supervised (controls for randomness)
    k1_pass = avg_imp_ss >= 5.0
    results["k1_threshold"] = 5.0
    results["k1_actual"] = round(avg_imp_ss, 2)
    results["k1_pass"] = k1_pass
    print(f"\n  K1: Distilled PPL >= 5% better than self-supervised")
    print(f"      Actual: {avg_imp_ss:+.1f}% -> {'PASS' if k1_pass else 'FAIL'}")

    # K2: Peak memory under 40GB
    peak = train_output["peak_memory_gb"]
    if teacher_peak is not None:
        peak = max(peak, teacher_peak)
    k2_pass = peak < 40.0
    results["k2_peak_memory_gb"] = round(peak, 2)
    results["k2_pass"] = k2_pass
    print(f"\n  K2: Peak memory < 40GB")
    print(f"      Actual: {peak:.2f} GB -> {'PASS' if k2_pass else 'FAIL'}")

    # Overall verdict
    verdict = "SUPPORTED" if k1_pass and k2_pass else "KILLED"
    results["verdict"] = verdict

    # Domains where distillation helped/hurt
    n_helped = sum(1 for imp in improvements_vs_ss if imp > 0)
    n_total = len(improvements_vs_ss)
    results["domains_improved_vs_ss"] = n_helped
    print(f"\n  Domains where distillation helped: {n_helped}/{n_total}")

    total_time = time.time() - t_total
    results["total_time_s"] = round(total_time, 1)
    print(f"\n  Total experiment time: {total_time:.0f}s ({total_time/60:.1f} min)")

    print(f"\n  VERDICT: {verdict}")

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
