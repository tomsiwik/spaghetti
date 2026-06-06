#!/usr/bin/env python3
"""Task Accuracy on Real Benchmarks: Does routed composition improve accuracy?

Tests the composed ternary model on real benchmarks:
  - GSM8K (50 problems): exact match on final numerical answer
  - MMLU subsets (20 questions per domain x 5 domains = 100): multiple-choice accuracy

4 configurations compared:
  1. Base: BitNet-2B-4T alone, no adapters
  2. Individual expert: single best adapter per benchmark (oracle selection)
  3. Uniform 1/N: all 5 adapters with equal weight (1/5 each)
  4. Routed (top-1): oracle routing to best adapter

Kill criteria:
  K1 (id=233): Routing doesn't improve task accuracy over uniform on any benchmark -> KILL
  K2 (id=234): Composed model worse than base on >50% of benchmarks -> KILL

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

# Source experiment with trained adapters
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0

DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = len(DOMAINS)

# MMLU subject mapping per adapter domain
MMLU_SUBJECTS = {
    "medical": ["clinical_knowledge", "professional_medicine", "anatomy", "medical_genetics"],
    "code": ["college_computer_science", "high_school_computer_science", "machine_learning"],
    "math": ["high_school_mathematics", "elementary_mathematics", "college_mathematics"],
    "legal": ["professional_law", "jurisprudence", "international_law"],
    "finance": ["professional_accounting", "econometrics", "high_school_macroeconomics"],
}

# How many MMLU questions per domain
MMLU_N_PER_DOMAIN = 20
# How many GSM8K problems
GSM8K_N = 50
# Generation tokens
MAX_NEW_TOKENS = 256


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
# BitNet unpacking and model utilities (reused from generation_quality_test)
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
# LoRA layers for composition (reused from generation_quality_test)
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
            self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_features, rank))
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


class MultiAdapterLoRALinear(nn.Module):
    """LoRA with multiple A/B pairs for multi-expert composition (uniform 1/N)."""
    def __init__(self, base_linear: nn.Linear, rank: int = 16,
                 scale: float = 20.0, a_inits: list = None):
        super().__init__()
        out_features = base_linear.weight.shape[0]
        self.linear = base_linear
        self.scale = scale
        self.rank = rank
        self.n_experts = len(a_inits) if a_inits else 0
        self.a_matrices = a_inits if a_inits else []
        self.b_matrices = [mx.zeros((rank, out_features)) for _ in range(self.n_experts)]
        self.linear.freeze()

    def __call__(self, x):
        base_out = self.linear(x)
        if self.n_experts == 0:
            return base_out
        lora_sum = mx.zeros_like(base_out)
        for i in range(self.n_experts):
            b = self.b_matrices[i]
            alpha = mx.mean(mx.abs(b))
            b_scaled = b / (alpha + 1e-7)
            b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
            b_ste = b + mx.stop_gradient(b_q - b)
            lora_sum = lora_sum + (x @ self.a_matrices[i]) @ b_ste
        return base_out + lora_sum * (self.scale / self.n_experts)


class RoutedMultiAdapterLoRALinear(nn.Module):
    """LoRA with multiple A/B pairs and per-expert routing weights."""
    def __init__(self, base_linear: nn.Linear, rank: int = 16,
                 scale: float = 20.0, a_inits: list = None):
        super().__init__()
        out_features = base_linear.weight.shape[0]
        self.linear = base_linear
        self.scale = scale
        self.rank = rank
        self.n_experts = len(a_inits) if a_inits else 0
        self.a_matrices = a_inits if a_inits else []
        self.b_matrices = [mx.zeros((rank, out_features)) for _ in range(self.n_experts)]
        self.linear.freeze()
        self._routing_weights = [1.0 / self.n_experts] * self.n_experts if self.n_experts > 0 else []

    def set_routing_weights(self, weights):
        self._routing_weights = weights

    def __call__(self, x):
        base_out = self.linear(x)
        if self.n_experts == 0:
            return base_out
        lora_sum = mx.zeros_like(base_out)
        for i in range(self.n_experts):
            w = self._routing_weights[i]
            if w < 1e-6:
                continue
            b = self.b_matrices[i]
            alpha = mx.mean(mx.abs(b))
            b_scaled = b / (alpha + 1e-7)
            b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
            b_ste = b + mx.stop_gradient(b_q - b)
            lora_sum = lora_sum + w * ((x @ self.a_matrices[i]) @ b_ste)
        return base_out + lora_sum * self.scale


# ============================================================================
# Model setup utilities
# ============================================================================

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def load_skeleton():
    """Load the Grassmannian A matrices."""
    skeleton_path = ADAPTERS_DIR / "grassmannian_skeleton.npz"
    return dict(np.load(str(skeleton_path)))


def apply_single_adapter(model, skeleton, domain_idx, domain_name):
    """Apply a single adapter (with correct A matrix) to the model."""
    n_layers = len(model.model.layers)
    a_matrices = {}
    for li in range(n_layers):
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_{domain_idx}"
            if skey in skeleton:
                a_matrices[(li, key)] = skeleton[skey]

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
            a_mx = mx.array(a_matrices[a_key]).astype(mx.bfloat16) if a_key in a_matrices else None
            lora = TernaryLoRALinear(module, rank=LORA_RANK, scale=LORA_SCALE, a_init=a_mx)
            lora_updates.append((key, lora))
            count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    adapter_path = ADAPTERS_DIR / domain_name / "adapter.npz"
    adapter_params = dict(mx.load(str(adapter_path)))
    model.update(tree_unflatten(list(adapter_params.items())))
    mx.eval(model.parameters())
    log(f"  Applied {domain_name} adapter ({count} layers)")
    return model


def apply_multi_adapter_uniform(model, skeleton):
    """Apply all 5 adapters with uniform 1/N weighting."""
    n_layers = len(model.model.layers)
    all_adapter_params = {}
    for di, domain in enumerate(DOMAINS):
        adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
        all_adapter_params[domain] = dict(mx.load(str(adapter_path)))

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
            a_inits = []
            for di in range(N_DOMAINS):
                skey = f"layer_{li}_{key}_domain_{di}"
                if skey in skeleton:
                    a_inits.append(mx.array(skeleton[skey]).astype(mx.bfloat16))
                else:
                    a_inits.append(None)
            multi_lora = MultiAdapterLoRALinear(
                module, rank=LORA_RANK, scale=LORA_SCALE, a_inits=a_inits
            )
            for di, domain in enumerate(DOMAINS):
                b_key = f"model.layers.{li}.{key}.lora_b"
                if b_key in all_adapter_params[domain]:
                    multi_lora.b_matrices[di] = all_adapter_params[domain][b_key]
            lora_updates.append((key, multi_lora))
            count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    mx.eval(model.parameters())
    log(f"  Applied uniform multi-adapter composition ({count} layers)")
    return model


def apply_multi_adapter_routed(model, skeleton):
    """Apply all 5 adapters with routing weights (set per-sequence)."""
    n_layers = len(model.model.layers)
    all_adapter_params = {}
    for di, domain in enumerate(DOMAINS):
        adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
        all_adapter_params[domain] = dict(mx.load(str(adapter_path)))

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
            a_inits = []
            for di in range(N_DOMAINS):
                skey = f"layer_{li}_{key}_domain_{di}"
                if skey in skeleton:
                    a_inits.append(mx.array(skeleton[skey]).astype(mx.bfloat16))
                else:
                    a_inits.append(None)
            routed_lora = RoutedMultiAdapterLoRALinear(
                module, rank=LORA_RANK, scale=LORA_SCALE, a_inits=a_inits
            )
            for di, domain in enumerate(DOMAINS):
                b_key = f"model.layers.{li}.{key}.lora_b"
                if b_key in all_adapter_params[domain]:
                    routed_lora.b_matrices[di] = all_adapter_params[domain][b_key]
            lora_updates.append((key, routed_lora))
            count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    mx.eval(model.parameters())
    log(f"  Applied routed multi-adapter composition ({count} layers)")
    return model


def set_all_routing_weights(model, weights):
    """Set routing weights on all RoutedMultiAdapterLoRALinear modules."""
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, RoutedMultiAdapterLoRALinear):
                module.set_routing_weights(weights)


# ============================================================================
# Data loading
# ============================================================================

def load_gsm8k_data(n=50):
    """Load GSM8K test problems."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    # Take first n problems (deterministic)
    problems = []
    for i in range(min(n, len(ds))):
        item = ds[i]
        # Extract final answer after ####
        answer_text = item["answer"]
        match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', answer_text)
        if match:
            answer = float(match.group(1).replace(',', ''))
        else:
            # Fallback: last number
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
    """Load MMLU test questions mapped to our adapter domains."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")

    # Index by subject
    by_subject = {}
    for item in ds:
        subj = item["subject"]
        if subj not in by_subject:
            by_subject[subj] = []
        by_subject[subj].append(item)

    mmlu_data = {}
    for domain, subjects in MMLU_SUBJECTS.items():
        questions = []
        for subj in subjects:
            if subj in by_subject:
                questions.extend(by_subject[subj])
        # Shuffle deterministically and take n_per_domain
        rng = np.random.RandomState(42)
        rng.shuffle(questions)
        mmlu_data[domain] = questions[:n_per_domain]
        log(f"  MMLU {domain}: {len(mmlu_data[domain])} questions from {subjects}")

    return mmlu_data


# ============================================================================
# Evaluation functions
# ============================================================================

def format_gsm8k_prompt(question):
    """Format GSM8K question for the model."""
    return (
        f"### Instruction:\n"
        f"Solve the following math problem step by step. "
        f"Show your work and give the final numerical answer after ####.\n\n"
        f"{question}\n\n"
        f"### Response:\n"
    )


def format_mmlu_prompt(question, choices):
    """Format MMLU multiple-choice question."""
    choice_labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(f"{label}. {choice}" for label, choice in zip(choice_labels, choices))
    return (
        f"### Instruction:\n"
        f"Answer the following multiple choice question. "
        f"Reply with just the letter (A, B, C, or D).\n\n"
        f"{question}\n\n{choices_text}\n\n"
        f"### Response:\n"
    )


def extract_gsm8k_answer(generated_text):
    """Extract numerical answer from GSM8K-style generation."""
    # Pattern: #### X
    match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', generated_text)
    if match:
        return float(match.group(1).replace(',', ''))

    # Pattern: "the answer is X"
    match = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*\$?([\d,]+(?:\.\d+)?)', generated_text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(',', ''))

    # Pattern: "= X" (last equation result)
    matches = re.findall(r'=\s*\$?([\d,]+(?:\.\d+)?)', generated_text)
    if matches:
        return float(matches[-1].replace(',', ''))

    # Pattern: "$X"
    matches = re.findall(r'\$([\d,]+(?:\.\d+)?)', generated_text)
    if matches:
        return float(matches[-1].replace(',', ''))

    # Last number in text
    matches = re.findall(r'([\d,]+(?:\.\d+)?)', generated_text)
    if matches:
        try:
            return float(matches[-1].replace(',', ''))
        except ValueError:
            pass
    return None


def check_gsm8k_correct(predicted, ground_truth, tolerance=0.01):
    """Check if GSM8K answer is correct (within tolerance)."""
    if predicted is None or ground_truth is None:
        return False
    if ground_truth == 0:
        return abs(predicted) < tolerance
    return abs(predicted - ground_truth) / abs(ground_truth) < tolerance


def extract_mmlu_answer(generated_text):
    """Extract letter answer (A/B/C/D) from generated text."""
    text = generated_text.strip()

    # Direct letter at start
    if text and text[0].upper() in "ABCD":
        return text[0].upper()

    # Pattern: "The answer is X" or "Answer: X"
    match = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*([A-Da-d])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern: single letter on its own line
    for line in text.split('\n'):
        line = line.strip()
        if len(line) == 1 and line.upper() in "ABCD":
            return line.upper()

    # Pattern: "(X)" or "X." or "X)"
    match = re.search(r'[\(\s]([A-Da-d])[\)\.\s]', text)
    if match:
        return match.group(1).upper()

    # First letter A-D found anywhere
    match = re.search(r'\b([A-Da-d])\b', text)
    if match:
        return match.group(1).upper()

    return None


def generate_text(model, tokenizer, prompt, max_tokens=256):
    """Generate text using mlx_lm."""
    try:
        sampler = make_sampler(temp=0.1, top_p=0.95)  # Low temp for accuracy tasks
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
# Phase 1: Evaluate on GSM8K
# ============================================================================

def phase_eval_gsm8k(config_name, model, tokenizer, problems):
    """Evaluate a model config on GSM8K problems."""
    log(f"\n  [GSM8K] Evaluating {config_name}...")
    t0 = time.time()
    correct = 0
    total = len(problems)
    details = []

    for i, prob in enumerate(problems):
        prompt = format_gsm8k_prompt(prob["question"])
        generated = generate_text(model, tokenizer, prompt, max_tokens=MAX_NEW_TOKENS)
        predicted = extract_gsm8k_answer(generated)
        is_correct = check_gsm8k_correct(predicted, prob["answer"])
        if is_correct:
            correct += 1
        details.append({
            "question": prob["question"][:100],
            "ground_truth": prob["answer"],
            "predicted": predicted,
            "correct": is_correct,
        })
        if (i + 1) % 10 == 0:
            log(f"    {i+1}/{total}: {correct}/{i+1} correct ({100*correct/(i+1):.1f}%)")
        # Free memory periodically
        if (i + 1) % 25 == 0:
            gc.collect()

    accuracy = correct / total if total > 0 else 0
    elapsed = time.time() - t0
    log(f"  [GSM8K] {config_name}: {correct}/{total} = {accuracy:.3f} ({elapsed:.1f}s)")
    return {"accuracy": accuracy, "correct": correct, "total": total, "time_s": elapsed, "details": details}


# ============================================================================
# Phase 2: Evaluate on MMLU
# ============================================================================

def phase_eval_mmlu(config_name, model, tokenizer, mmlu_data, domain=None):
    """Evaluate a model config on MMLU questions for a specific domain (or all)."""
    if domain:
        domains_to_eval = [domain]
    else:
        domains_to_eval = list(mmlu_data.keys())

    all_results = {}
    for d in domains_to_eval:
        questions = mmlu_data[d]
        if not questions:
            all_results[d] = {"accuracy": 0, "correct": 0, "total": 0}
            continue

        correct = 0
        total = len(questions)
        choice_labels = ["A", "B", "C", "D"]

        for i, q in enumerate(questions):
            prompt = format_mmlu_prompt(q["question"], q["choices"])
            generated = generate_text(model, tokenizer, prompt, max_tokens=32)  # Short for MC
            predicted = extract_mmlu_answer(generated)
            gt_label = choice_labels[q["answer"]]
            is_correct = (predicted == gt_label)
            if is_correct:
                correct += 1

        accuracy = correct / total if total > 0 else 0
        all_results[d] = {"accuracy": accuracy, "correct": correct, "total": total}
        log(f"    MMLU {d}: {correct}/{total} = {accuracy:.3f}")

    return all_results


# ============================================================================
# Orchestration: evaluate one configuration on all benchmarks
# ============================================================================

def evaluate_config_base(gsm8k_problems, mmlu_data):
    """Evaluate base model (no adapters) on all benchmarks."""
    log("\n" + "="*70)
    log("CONFIG: BASE (no adapters)")
    log("="*70)
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("post-load-base")

    gsm8k_results = phase_eval_gsm8k("base", model, tokenizer, gsm8k_problems)
    log(f"\n  Evaluating MMLU for base...")
    mmlu_results = phase_eval_mmlu("base", model, tokenizer, mmlu_data)

    elapsed = time.time() - t0
    log(f"\nBASE total time: {elapsed:.1f}s")
    log_memory("post-eval-base")
    cleanup(model, tokenizer)

    return {"gsm8k": gsm8k_results, "mmlu": mmlu_results, "time_s": elapsed}


def evaluate_config_individual(gsm8k_problems, mmlu_data):
    """Evaluate individual best adapter per benchmark (oracle selection).

    For GSM8K: use math adapter.
    For MMLU per domain: use matching domain adapter.
    """
    log("\n" + "="*70)
    log("CONFIG: INDIVIDUAL EXPERT (oracle selection)")
    log("="*70)
    t0 = time.time()

    skeleton = load_skeleton()

    # GSM8K with math adapter (domain index 2 = "math")
    log("\n  Loading math adapter for GSM8K...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    math_idx = DOMAINS.index("math")
    model = apply_single_adapter(model, skeleton, math_idx, "math")
    model.freeze()
    gsm8k_results = phase_eval_gsm8k("individual(math)", model, tokenizer, gsm8k_problems)
    cleanup(model, tokenizer)

    # MMLU per domain with matching adapter
    mmlu_results = {}
    for di, domain in enumerate(DOMAINS):
        if domain not in mmlu_data or not mmlu_data[domain]:
            continue
        log(f"\n  Loading {domain} adapter for MMLU...")
        model, tokenizer = load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)
        model = apply_single_adapter(model, skeleton, di, domain)
        model.freeze()
        domain_result = phase_eval_mmlu(f"individual({domain})", model, tokenizer, mmlu_data, domain=domain)
        mmlu_results.update(domain_result)
        cleanup(model, tokenizer)

    elapsed = time.time() - t0
    log(f"\nINDIVIDUAL total time: {elapsed:.1f}s")
    del skeleton
    gc.collect()

    return {"gsm8k": gsm8k_results, "mmlu": mmlu_results, "time_s": elapsed}


def evaluate_config_uniform(gsm8k_problems, mmlu_data):
    """Evaluate uniform 1/N composition on all benchmarks."""
    log("\n" + "="*70)
    log("CONFIG: UNIFORM 1/N (all 5 adapters equally weighted)")
    log("="*70)
    t0 = time.time()

    skeleton = load_skeleton()
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model = apply_multi_adapter_uniform(model, skeleton)
    model.freeze()
    log_memory("post-load-uniform")

    gsm8k_results = phase_eval_gsm8k("uniform", model, tokenizer, gsm8k_problems)
    log(f"\n  Evaluating MMLU for uniform...")
    mmlu_results = phase_eval_mmlu("uniform", model, tokenizer, mmlu_data)

    elapsed = time.time() - t0
    log(f"\nUNIFORM total time: {elapsed:.1f}s")
    log_memory("post-eval-uniform")
    cleanup(model, tokenizer)
    del skeleton
    gc.collect()

    return {"gsm8k": gsm8k_results, "mmlu": mmlu_results, "time_s": elapsed}


def evaluate_config_routed(gsm8k_problems, mmlu_data):
    """Evaluate routed top-1 composition on all benchmarks.

    Oracle routing: for GSM8K, route to math adapter.
    For MMLU, route to matching domain adapter.
    """
    log("\n" + "="*70)
    log("CONFIG: ROUTED TOP-1 (oracle routing to best adapter)")
    log("="*70)
    t0 = time.time()

    skeleton = load_skeleton()
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model = apply_multi_adapter_routed(model, skeleton)
    model.freeze()
    log_memory("post-load-routed")

    # GSM8K: route to math adapter
    math_idx = DOMAINS.index("math")
    weights = [0.0] * N_DOMAINS
    weights[math_idx] = 1.0
    set_all_routing_weights(model, weights)
    gsm8k_results = phase_eval_gsm8k("routed(math)", model, tokenizer, gsm8k_problems)

    # MMLU per domain: route to matching adapter
    mmlu_results = {}
    for di, domain in enumerate(DOMAINS):
        if domain not in mmlu_data or not mmlu_data[domain]:
            continue
        weights = [0.0] * N_DOMAINS
        weights[di] = 1.0
        set_all_routing_weights(model, weights)
        domain_result = phase_eval_mmlu(f"routed({domain})", model, tokenizer, mmlu_data, domain=domain)
        mmlu_results.update(domain_result)

    elapsed = time.time() - t0
    log(f"\nROUTED total time: {elapsed:.1f}s")
    log_memory("post-eval-routed")
    cleanup(model, tokenizer)
    del skeleton
    gc.collect()

    return {"gsm8k": gsm8k_results, "mmlu": mmlu_results, "time_s": elapsed}


# ============================================================================
# Analysis and Kill Criteria
# ============================================================================

def analyze_results(base, individual, uniform, routed):
    """Analyze results and evaluate kill criteria."""
    log("\n" + "="*70)
    log("ANALYSIS")
    log("="*70)

    # Collect all benchmark results
    benchmarks = {}

    # GSM8K
    benchmarks["gsm8k"] = {
        "base": base["gsm8k"]["accuracy"],
        "individual": individual["gsm8k"]["accuracy"],
        "uniform": uniform["gsm8k"]["accuracy"],
        "routed": routed["gsm8k"]["accuracy"],
    }

    # MMLU per domain
    for domain in DOMAINS:
        key = f"mmlu_{domain}"
        benchmarks[key] = {
            "base": base["mmlu"].get(domain, {}).get("accuracy", 0),
            "individual": individual["mmlu"].get(domain, {}).get("accuracy", 0),
            "uniform": uniform["mmlu"].get(domain, {}).get("accuracy", 0),
            "routed": routed["mmlu"].get(domain, {}).get("accuracy", 0),
        }

    # Print summary table
    log("\nBenchmark Results Summary:")
    log(f"{'Benchmark':<20} {'Base':>8} {'Individual':>12} {'Uniform':>10} {'Routed':>10} {'R>U?':>6} {'R>B?':>6}")
    log("-" * 78)

    routed_beats_uniform_count = 0
    routed_worse_than_base_count = 0
    total_benchmarks = 0

    for bench_name, scores in benchmarks.items():
        b = scores["base"]
        ind = scores["individual"]
        u = scores["uniform"]
        r = scores["routed"]
        r_gt_u = r > u
        r_gt_b = r >= b  # >= because equal means not worse

        if r_gt_u:
            routed_beats_uniform_count += 1
        if r < b:
            routed_worse_than_base_count += 1
        total_benchmarks += 1

        log(f"{bench_name:<20} {b:>8.3f} {ind:>12.3f} {u:>10.3f} {r:>10.3f} {'YES' if r_gt_u else 'no':>6} {'YES' if r_gt_b else 'no':>6}")

    log("-" * 78)
    log(f"Routed beats uniform: {routed_beats_uniform_count}/{total_benchmarks}")
    log(f"Routed worse than base: {routed_worse_than_base_count}/{total_benchmarks}")

    # K1: Routing doesn't improve task accuracy over uniform on ANY benchmark
    k1_pass = routed_beats_uniform_count > 0
    k1_result = "pass" if k1_pass else "fail"
    k1_evidence = (f"Routed beats uniform on {routed_beats_uniform_count}/{total_benchmarks} benchmarks")

    # K2: Composed model worse than base on >50% of benchmarks
    k2_pass = routed_worse_than_base_count <= total_benchmarks / 2
    k2_result = "pass" if k2_pass else "fail"
    k2_evidence = (f"Routed worse than base on {routed_worse_than_base_count}/{total_benchmarks} benchmarks")

    # S1: Gumbel-routed composed model beats base on majority of benchmarks
    routed_beats_base = sum(1 for s in benchmarks.values() if s["routed"] > s["base"])
    s1_pass = routed_beats_base > total_benchmarks / 2
    s1_evidence = f"Routed beats base on {routed_beats_base}/{total_benchmarks} benchmarks"

    log(f"\nKill Criteria:")
    log(f"  K1 (233): {k1_result.upper()} -- {k1_evidence}")
    log(f"  K2 (234): {k2_result.upper()} -- {k2_evidence}")
    log(f"  S1 (18):  {'PASS' if s1_pass else 'FAIL'} -- {s1_evidence}")

    if not k1_pass:
        log("\n  *** K1 TRIGGERED: KILL -- routing provides zero improvement over uniform ***")
    if not k2_pass:
        log("\n  *** K2 TRIGGERED: KILL -- composed model worse than base on majority ***")

    return {
        "benchmarks": benchmarks,
        "kill_criteria": {
            "k1_233": {"result": k1_result, "evidence": k1_evidence},
            "k2_234": {"result": k2_result, "evidence": k2_evidence},
        },
        "success_criteria": {
            "s1_18": {"result": "pass" if s1_pass else "fail", "evidence": s1_evidence},
        },
        "summary": {
            "routed_beats_uniform": routed_beats_uniform_count,
            "routed_worse_than_base": routed_worse_than_base_count,
            "total_benchmarks": total_benchmarks,
        },
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("="*70)
    log("Task Accuracy on Real Benchmarks")
    log("="*70)
    log_memory("start")

    # Load benchmark data
    log("\n[Data] Loading benchmark data...")
    gsm8k_problems = load_gsm8k_data(n=GSM8K_N)
    mmlu_data = load_mmlu_data(n_per_domain=MMLU_N_PER_DOMAIN)

    # Evaluate all configurations
    base_results = evaluate_config_base(gsm8k_problems, mmlu_data)
    individual_results = evaluate_config_individual(gsm8k_problems, mmlu_data)
    uniform_results = evaluate_config_uniform(gsm8k_problems, mmlu_data)
    routed_results = evaluate_config_routed(gsm8k_problems, mmlu_data)

    # Analyze and evaluate kill criteria
    analysis = analyze_results(base_results, individual_results, uniform_results, routed_results)

    # Save full results
    total_time = time.time() - t0
    results = {
        "experiment": "task_accuracy_real_benchmarks",
        "model": MODEL_ID,
        "configs": {
            "base": base_results,
            "individual": individual_results,
            "uniform": uniform_results,
            "routed": routed_results,
        },
        "analysis": analysis,
        "params": {
            "gsm8k_n": GSM8K_N,
            "mmlu_n_per_domain": MMLU_N_PER_DOMAIN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.1,
            "top_p": 0.95,
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
        },
        "total_time_s": round(total_time, 1),
    }

    # Strip details from saved JSON to keep file manageable
    for config_name in ["base", "individual", "uniform", "routed"]:
        if "gsm8k" in results["configs"][config_name]:
            results["configs"][config_name]["gsm8k"].pop("details", None)

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total experiment time: {total_time:.1f}s ({total_time/60:.1f} min)")
    log_memory("final")


if __name__ == "__main__":
    main()
