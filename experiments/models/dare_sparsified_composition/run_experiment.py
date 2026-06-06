#!/usr/bin/env python3
"""DARE sparsified adapter composition: reduce OOD interference via delta dropout.

Kill criteria:
  K681: DARE composition STILL degrades OOD benchmarks by >=5pp on majority (>=3/5) domains
  K682: In-distribution behavioral gains drop below 50% of non-DARE composition
  K683: DARE + ternary produces degenerate output (incompatible with ternary quantization)

Type: Guided exploration (Type 2)
Prior math: DARE (arXiv:2311.03099)
Unknown: Optimal drop rate p for ternary adapters

Approach:
  1. Evaluate base model on OOD benchmarks (GSM8K, code gen, MMLU)
  2. Evaluate NTP adapters WITHOUT DARE (baseline composition)
  3. Evaluate NTP adapters WITH DARE at p in {0.5, 0.7, 0.9, 0.95}
  4. Compare OOD degradation and in-distribution behavioral gains
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

# Adapter sources (NTP adapters from real_data_domain_experts)
NTP_SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
NTP_ADAPTERS_DIR = NTP_SOURCE_DIR / "adapters"
NTP_DATA_DIR = NTP_SOURCE_DIR / "data"
SKELETON_PATH = NTP_ADAPTERS_DIR / "grassmannian_skeleton.npz"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]

# Per-domain optimal scales (Finding #249)
OPTIMAL_SCALES = {
    "medical": 20.0,
    "code": 20.0,
    "math": 20.0,
    "legal": 4.0,
    "finance": 1.0,
}

# DARE drop rates to sweep
DROP_RATES = [0.5, 0.7, 0.9, 0.95]

# Benchmark sizes
GSM8K_N = 50
CODE_GEN_N = 10
MMLU_N_PER_DOMAIN = 20

MAX_TOKENS_GSM8K = 256
MAX_TOKENS_CODE = 256
MAX_TOKENS_MMLU = 32
MAX_TOKENS_DOMAIN = 128

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

MMLU_SUBJECTS = {
    "medical": ["clinical_knowledge", "professional_medicine", "anatomy", "medical_genetics"],
    "code": ["college_computer_science", "high_school_computer_science", "machine_learning"],
    "math": ["high_school_mathematics", "elementary_mathematics", "college_mathematics"],
    "legal": ["professional_law", "jurisprudence", "international_law"],
    "finance": ["professional_accounting", "econometrics", "high_school_macroeconomics"],
}


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
# BitNet unpacking
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
# Adapter loading & composition
# ============================================================================

def load_skeleton():
    return dict(np.load(str(SKELETON_PATH)))


def load_adapter(adapter_dir, domain):
    adapter_path = adapter_dir / domain / "adapter.npz"
    adapter = dict(mx.load(str(adapter_path)))
    n_tensors = len(adapter)
    log(f"  Loaded adapter: {domain} ({n_tensors} tensors)")
    return adapter


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


def compute_delta(skeleton, adapter, domain, scale):
    """Compute the materialized delta for a single adapter.
    Returns dict mapping (layer_idx, key) -> delta matrix.
    """
    di = DOMAINS.index(domain)
    deltas = {}
    n_layers = 30  # BitNet-2B has 30 layers
    for li in range(n_layers):
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_{di}"
            if skey not in skeleton:
                continue
            a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)  # (d_in, r)
            b_key = f"model.layers.{li}.{key}.lora_b"
            if b_key not in adapter:
                continue
            b_mx = adapter[b_key]  # (r, d_out)
            delta = scale * (b_mx.T @ a_mx.T)  # (d_out, d_in)
            deltas[(li, key)] = delta
    return deltas


def apply_dare_to_deltas(deltas, drop_rate, rng_key=None):
    """Apply DARE sparsification: Bernoulli mask + rescale.

    For each delta matrix, randomly drop `drop_rate` fraction of entries
    and rescale survivors by 1/(1-drop_rate).

    Args:
        deltas: dict mapping (layer_idx, key) -> delta matrix
        drop_rate: probability of dropping each entry (p in DARE paper)
        rng_key: random key for reproducibility

    Returns:
        dict with same keys, DARE-sparsified delta matrices
    """
    if rng_key is None:
        rng_key = mx.random.key(SEED)

    rescale = 1.0 / (1.0 - drop_rate)
    dare_deltas = {}
    total_params = 0
    total_kept = 0

    for (li, key), delta in deltas.items():
        # Generate Bernoulli mask: 1 = keep, 0 = drop
        rng_key, subkey = mx.random.split(rng_key)
        mask = mx.random.bernoulli(
            p=(1.0 - drop_rate),
            shape=delta.shape,
            key=subkey,
        ).astype(mx.bfloat16)

        # Apply mask and rescale
        dare_delta = delta * mask * rescale
        dare_deltas[(li, key)] = dare_delta

        total_params += delta.size
        total_kept += int(mx.sum(mask).item())

    actual_keep_rate = total_kept / total_params if total_params > 0 else 0
    log(f"    DARE p={drop_rate}: kept {total_kept}/{total_params} = {actual_keep_rate:.3f} "
        f"(target {1-drop_rate:.3f}), rescale={rescale:.2f}")

    mx.eval(dare_deltas)
    return dare_deltas


def apply_deltas_to_model(model, deltas):
    """Apply precomputed delta matrices to model weights."""
    merge_count = 0
    for (li, key), delta in deltas.items():
        parts = key.split(".")
        module = model.model.layers[li]
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                break
        if module is not None and isinstance(module, nn.Linear):
            module.weight = module.weight + delta
            merge_count += 1
    mx.eval(model.parameters())
    return merge_count


def premerge_single_adapter(model, skeleton, adapter, domain, scale):
    """Pre-merge without DARE (baseline). Returns merge count."""
    deltas = compute_delta(skeleton, adapter, domain, scale)
    count = apply_deltas_to_model(model, deltas)
    del deltas
    return count


def premerge_single_adapter_dare(model, skeleton, adapter, domain, scale, drop_rate, rng_key=None):
    """Pre-merge with DARE sparsification. Returns merge count."""
    deltas = compute_delta(skeleton, adapter, domain, scale)
    dare_deltas = apply_dare_to_deltas(deltas, drop_rate, rng_key)
    del deltas
    count = apply_deltas_to_model(model, dare_deltas)
    del dare_deltas
    gc.collect()
    mx.clear_cache()
    return count


# ============================================================================
# Generation & evaluation (reused from ntp_vs_sft_ood_benchmark)
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


def format_gsm8k_prompt(question):
    return (
        f"### Instruction:\n"
        f"Solve the following math problem step by step. "
        f"Show your work and give the final numerical answer after ####.\n\n"
        f"{question}\n\n"
        f"### Response:\n"
    )


def format_code_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def format_mmlu_prompt(question, choices):
    choice_labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(f"{label}. {choice}" for label, choice in zip(choice_labels, choices))
    return (
        f"### Instruction:\n"
        f"Answer the following multiple choice question. "
        f"Reply with just the letter (A, B, C, or D).\n\n"
        f"{question}\n\n{choices_text}\n\n"
        f"### Response:\n"
    )


def format_domain_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


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


# ============================================================================
# Data loading
# ============================================================================

def load_gsm8k_data(n=50):
    from datasets import load_dataset
    log(f"  Loading GSM8K ({n} problems)...")
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
        problems.append({"question": item["question"], "answer": answer})
    log(f"  GSM8K: {len(problems)} problems loaded")
    return problems


def load_code_gen_data(n=10):
    log(f"  Loading code generation ({n} problems)...")
    val_path = NTP_DATA_DIR / "code" / "valid.jsonl"
    problems = []
    with open(val_path) as f:
        for line in f:
            text = json.loads(line)["text"]
            if "### Instruction:" in text and "### Response:" in text:
                instruction = text.split("### Instruction:")[1].split("### Response:")[0].strip()
                response = text.split("### Response:")[1].strip()
                problems.append({"instruction": instruction, "reference": response})
            if len(problems) >= n:
                break
    log(f"  Code gen: {len(problems)} problems loaded")
    return problems


def load_mmlu_data(n_per_domain=20):
    from datasets import load_dataset
    log(f"  Loading MMLU ({n_per_domain} per domain)...")
    ds = load_dataset("cais/mmlu", "all", split="test")
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
        rng = np.random.RandomState(42)
        rng.shuffle(questions)
        mmlu_data[domain] = questions[:n_per_domain]
        log(f"    MMLU {domain}: {len(mmlu_data[domain])} questions")
    return mmlu_data


def load_indist_eval_data():
    log("  Loading in-distribution eval data...")
    indist = {}

    math_val = NTP_DATA_DIR / "math" / "valid.jsonl"
    math_problems = []
    with open(math_val) as f:
        for line in f:
            text = json.loads(line)["text"]
            if "### Instruction:" in text and "### Response:" in text:
                inst = text.split("### Instruction:")[1].split("### Response:")[0].strip()
                resp = text.split("### Response:")[1].strip()
                math_problems.append({"instruction": inst, "reference": resp})
            if len(math_problems) >= 20:
                break
    indist["math"] = math_problems
    log(f"    Math in-dist: {len(math_problems)} problems")

    code_val = NTP_DATA_DIR / "code" / "valid.jsonl"
    code_problems = []
    with open(code_val) as f:
        for line in f:
            text = json.loads(line)["text"]
            if "### Instruction:" in text and "### Response:" in text:
                inst = text.split("### Instruction:")[1].split("### Response:")[0].strip()
                resp = text.split("### Response:")[1].strip()
                code_problems.append({"instruction": inst, "reference": resp})
            if len(code_problems) >= 20:
                break
    indist["code"] = code_problems
    log(f"    Code in-dist: {len(code_problems)} problems")

    return indist


# ============================================================================
# Benchmark evaluation functions
# ============================================================================

def eval_gsm8k(label, model, tokenizer, problems):
    log(f"\n  [GSM8K] Evaluating {label}...")
    t0 = time.time()
    correct = 0
    total = len(problems)
    for i, prob in enumerate(problems):
        prompt = format_gsm8k_prompt(prob["question"])
        gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_GSM8K)
        predicted = extract_gsm8k_answer(gen)
        if check_gsm8k_correct(predicted, prob["answer"]):
            correct += 1
        if (i + 1) % 10 == 0:
            log(f"    {i+1}/{total}: {correct}/{i+1} ({100*correct/(i+1):.1f}%)")
        if (i + 1) % 25 == 0:
            gc.collect()
    accuracy = correct / total if total > 0 else 0
    elapsed = time.time() - t0
    log(f"  [GSM8K] {label}: {correct}/{total} = {accuracy:.1%} ({elapsed:.1f}s)")
    return {"accuracy": accuracy, "correct": correct, "total": total, "time_s": round(elapsed, 1)}


def eval_code_gen(label, model, tokenizer, problems):
    log(f"\n  [Code Gen] Evaluating {label}...")
    t0 = time.time()
    syntax_ok = 0
    total = len(problems)
    for prob in problems:
        prompt = format_code_prompt(prob["instruction"])
        gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_CODE)
        if eval_code_syntax(gen):
            syntax_ok += 1
    rate = syntax_ok / total if total > 0 else 0
    elapsed = time.time() - t0
    log(f"  [Code Gen] {label}: {syntax_ok}/{total} = {rate:.1%} ({elapsed:.1f}s)")
    return {"syntax_rate": rate, "correct": syntax_ok, "total": total, "time_s": round(elapsed, 1)}


def eval_mmlu(label, model, tokenizer, mmlu_data):
    log(f"\n  [MMLU] Evaluating {label}...")
    choice_labels = ["A", "B", "C", "D"]
    results_by_domain = {}
    total_correct = 0
    total_q = 0
    for domain in DOMAINS:
        questions = mmlu_data[domain]
        correct = 0
        for q in questions:
            prompt = format_mmlu_prompt(q["question"], q["choices"])
            gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_MMLU)
            predicted = extract_mmlu_answer(gen)
            gt_label = choice_labels[q["answer"]]
            if predicted == gt_label:
                correct += 1
        n = len(questions)
        acc = correct / n if n > 0 else 0
        results_by_domain[domain] = {"accuracy": acc, "correct": correct, "total": n}
        total_correct += correct
        total_q += n
        log(f"    MMLU {domain}: {correct}/{n} = {acc:.1%}")

    overall_acc = total_correct / total_q if total_q > 0 else 0
    log(f"  [MMLU] {label} overall: {total_correct}/{total_q} = {overall_acc:.1%}")
    return {
        "overall_accuracy": overall_acc,
        "overall_correct": total_correct,
        "overall_total": total_q,
        "by_domain": results_by_domain,
    }


def eval_math_correctness(model, tokenizer, problems):
    correct = 0
    total = len(problems)
    for prob in problems:
        prompt = format_domain_prompt(prob["instruction"])
        gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_DOMAIN)
        ref_nums = re.findall(r'[\d]+(?:\.\d+)?', prob["reference"])
        if ref_nums:
            answer_num = ref_nums[-1]
            if answer_num in gen:
                correct += 1
        else:
            ref_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', prob["reference"].lower()))
            gen_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', gen.lower()))
            overlap = len(ref_words & gen_words) / max(len(ref_words), 1)
            if overlap > 0.3:
                correct += 1
    return {"correctness": correct / total if total > 0 else 0, "correct": correct, "total": total}


def eval_code_passrate(model, tokenizer, problems):
    passed = 0
    total = len(problems)
    for prob in problems:
        prompt = format_domain_prompt(prob["instruction"])
        gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_CODE)
        if eval_code_syntax(gen):
            passed += 1
    return {"pass_rate": passed / total if total > 0 else 0, "passed": passed, "total": total}


# ============================================================================
# Phase functions (each self-contained per CODING_GUIDELINES)
# ============================================================================

def phase_load_data():
    """Load all benchmark data."""
    log("\n" + "=" * 70)
    log("PHASE 0: LOADING ALL BENCHMARK DATA")
    log("=" * 70)
    gsm8k = load_gsm8k_data(GSM8K_N)
    code_gen = load_code_gen_data(CODE_GEN_N)
    mmlu = load_mmlu_data(MMLU_N_PER_DOMAIN)
    indist = load_indist_eval_data()
    return gsm8k, code_gen, mmlu, indist


def phase_eval_base(gsm8k, code_gen, mmlu):
    """Evaluate base model (no adapters) on OOD benchmarks."""
    log("\n" + "=" * 70)
    log("PHASE 1: BASE MODEL (no adapters)")
    log("=" * 70)
    t0 = time.time()
    mx.reset_peak_memory()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("base-loaded")

    gsm8k_results = eval_gsm8k("base", model, tokenizer, gsm8k)
    code_results = eval_code_gen("base", model, tokenizer, code_gen)
    mmlu_results = eval_mmlu("base", model, tokenizer, mmlu)

    peak_mem = mx.get_peak_memory() / 1e9
    elapsed = time.time() - t0
    log(f"\nBase model total: {elapsed:.1f}s, peak memory: {peak_mem:.2f}GB")

    results = {
        "gsm8k": gsm8k_results,
        "code_gen": code_results,
        "mmlu": mmlu_results,
        "peak_memory_gb": round(peak_mem, 2),
        "time_s": round(elapsed, 1),
    }
    cleanup(model, tokenizer)
    return results


def phase_eval_ntp_no_dare(gsm8k, code_gen, mmlu, indist):
    """Evaluate NTP adapters WITHOUT DARE (baseline composition)."""
    log("\n" + "=" * 70)
    log("PHASE 2: NTP ADAPTERS WITHOUT DARE (baseline)")
    log("=" * 70)
    t0 = time.time()
    mx.reset_peak_memory()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    skeleton = load_skeleton()
    adapters = {}
    for domain in DOMAINS:
        adapters[domain] = load_adapter(NTP_ADAPTERS_DIR, domain)

    base_weights = save_base_weights(model)
    log_memory("ntp-baseline-loaded")

    # GSM8K with math adapter
    log("\n  --- GSM8K (math adapter, s=20) ---")
    n = premerge_single_adapter(model, skeleton, adapters["math"], "math", OPTIMAL_SCALES["math"])
    log(f"  Pre-merged math adapter into {n} layers")
    gsm8k_results = eval_gsm8k("ntp-no-dare", model, tokenizer, gsm8k)
    restore_base_weights(model, base_weights)

    # Code gen with code adapter
    log("\n  --- Code Gen (code adapter, s=20) ---")
    n = premerge_single_adapter(model, skeleton, adapters["code"], "code", OPTIMAL_SCALES["code"])
    log(f"  Pre-merged code adapter into {n} layers")
    code_results = eval_code_gen("ntp-no-dare", model, tokenizer, code_gen)
    restore_base_weights(model, base_weights)

    # MMLU per-domain routing
    log("\n  --- MMLU (per-domain routing) ---")
    choice_labels = ["A", "B", "C", "D"]
    mmlu_results_by_domain = {}
    total_correct = 0
    total_q = 0
    for domain in DOMAINS:
        scale = OPTIMAL_SCALES[domain]
        n = premerge_single_adapter(model, skeleton, adapters[domain], domain, scale)
        log(f"  Pre-merged {domain} adapter (s={scale}) into {n} layers")
        questions = mmlu[domain]
        correct = 0
        for q in questions:
            prompt = format_mmlu_prompt(q["question"], q["choices"])
            gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_MMLU)
            predicted = extract_mmlu_answer(gen)
            gt_label = choice_labels[q["answer"]]
            if predicted == gt_label:
                correct += 1
        acc = correct / len(questions) if questions else 0
        mmlu_results_by_domain[domain] = {"accuracy": acc, "correct": correct, "total": len(questions)}
        total_correct += correct
        total_q += len(questions)
        log(f"    MMLU {domain}: {correct}/{len(questions)} = {acc:.1%}")
        restore_base_weights(model, base_weights)
        gc.collect()

    overall_mmlu = total_correct / total_q if total_q > 0 else 0
    mmlu_results = {
        "overall_accuracy": overall_mmlu,
        "overall_correct": total_correct,
        "overall_total": total_q,
        "by_domain": mmlu_results_by_domain,
    }

    # In-distribution eval
    log("\n  --- In-Distribution Behavioral Eval ---")
    n = premerge_single_adapter(model, skeleton, adapters["math"], "math", OPTIMAL_SCALES["math"])
    math_indist = eval_math_correctness(model, tokenizer, indist["math"])
    log(f"  Math in-dist correctness: {math_indist['correct']}/{math_indist['total']} = {math_indist['correctness']:.1%}")
    restore_base_weights(model, base_weights)

    n = premerge_single_adapter(model, skeleton, adapters["code"], "code", OPTIMAL_SCALES["code"])
    code_indist = eval_code_passrate(model, tokenizer, indist["code"])
    log(f"  Code in-dist pass rate: {code_indist['passed']}/{code_indist['total']} = {code_indist['pass_rate']:.1%}")
    restore_base_weights(model, base_weights)

    elapsed = time.time() - t0
    results = {
        "gsm8k": gsm8k_results,
        "code_gen": code_results,
        "mmlu": mmlu_results,
        "in_distribution": {
            "math_correctness": math_indist,
            "code_pass_rate": code_indist,
        },
        "time_s": round(elapsed, 1),
    }
    cleanup(model, tokenizer, skeleton, adapters, base_weights)
    return results


def phase_eval_dare(gsm8k, code_gen, mmlu, indist, drop_rate):
    """Evaluate NTP adapters WITH DARE at given drop rate."""
    log("\n" + "=" * 70)
    log(f"PHASE DARE p={drop_rate}: NTP ADAPTERS WITH DARE SPARSIFICATION")
    log("=" * 70)
    t0 = time.time()
    mx.reset_peak_memory()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    skeleton = load_skeleton()
    adapters = {}
    for domain in DOMAINS:
        adapters[domain] = load_adapter(NTP_ADAPTERS_DIR, domain)

    base_weights = save_base_weights(model)
    log_memory(f"dare-p{drop_rate}-loaded")

    # Use different seeds per drop rate for reproducibility but independence
    seed_offset = int(drop_rate * 1000)

    # GSM8K with math adapter + DARE
    log(f"\n  --- GSM8K (math adapter, s=20, DARE p={drop_rate}) ---")
    rng_key = mx.random.key(SEED + seed_offset)
    n = premerge_single_adapter_dare(
        model, skeleton, adapters["math"], "math",
        OPTIMAL_SCALES["math"], drop_rate, rng_key
    )
    log(f"  Pre-merged math adapter (DARE) into {n} layers")

    # Check for degenerate output (K683)
    test_prompt = "### Instruction:\nWhat is 2+2?\n\n### Response:\n"
    test_gen = generate_text(model, tokenizer, test_prompt, max_tokens=32)
    is_degenerate = (
        len(test_gen.strip()) < 2
        or test_gen.strip() == test_prompt.strip()
        or all(c == test_gen[0] for c in test_gen.strip())
    )
    log(f"  Degenerate check: {'FAIL' if is_degenerate else 'PASS'} (output: {test_gen[:80]!r})")

    gsm8k_results = eval_gsm8k(f"dare-p{drop_rate}", model, tokenizer, gsm8k)
    restore_base_weights(model, base_weights)

    # Code gen with code adapter + DARE
    log(f"\n  --- Code Gen (code adapter, s=20, DARE p={drop_rate}) ---")
    rng_key = mx.random.key(SEED + seed_offset + 1)
    n = premerge_single_adapter_dare(
        model, skeleton, adapters["code"], "code",
        OPTIMAL_SCALES["code"], drop_rate, rng_key
    )
    log(f"  Pre-merged code adapter (DARE) into {n} layers")
    code_results = eval_code_gen(f"dare-p{drop_rate}", model, tokenizer, code_gen)
    restore_base_weights(model, base_weights)

    # MMLU per-domain routing with DARE
    log(f"\n  --- MMLU (per-domain routing, DARE p={drop_rate}) ---")
    choice_labels = ["A", "B", "C", "D"]
    mmlu_results_by_domain = {}
    total_correct = 0
    total_q = 0
    for di, domain in enumerate(DOMAINS):
        scale = OPTIMAL_SCALES[domain]
        rng_key = mx.random.key(SEED + seed_offset + 10 + di)
        n = premerge_single_adapter_dare(
            model, skeleton, adapters[domain], domain,
            scale, drop_rate, rng_key
        )
        log(f"  Pre-merged {domain} adapter (DARE s={scale}) into {n} layers")
        questions = mmlu[domain]
        correct = 0
        for q in questions:
            prompt = format_mmlu_prompt(q["question"], q["choices"])
            gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_MMLU)
            predicted = extract_mmlu_answer(gen)
            gt_label = choice_labels[q["answer"]]
            if predicted == gt_label:
                correct += 1
        acc = correct / len(questions) if questions else 0
        mmlu_results_by_domain[domain] = {"accuracy": acc, "correct": correct, "total": len(questions)}
        total_correct += correct
        total_q += len(questions)
        log(f"    MMLU {domain}: {correct}/{len(questions)} = {acc:.1%}")
        restore_base_weights(model, base_weights)
        gc.collect()

    overall_mmlu = total_correct / total_q if total_q > 0 else 0
    mmlu_results = {
        "overall_accuracy": overall_mmlu,
        "overall_correct": total_correct,
        "overall_total": total_q,
        "by_domain": mmlu_results_by_domain,
    }

    # In-distribution eval with DARE
    log(f"\n  --- In-Distribution Behavioral Eval (DARE p={drop_rate}) ---")
    rng_key = mx.random.key(SEED + seed_offset + 100)
    n = premerge_single_adapter_dare(
        model, skeleton, adapters["math"], "math",
        OPTIMAL_SCALES["math"], drop_rate, rng_key
    )
    math_indist = eval_math_correctness(model, tokenizer, indist["math"])
    log(f"  Math in-dist correctness: {math_indist['correct']}/{math_indist['total']} = {math_indist['correctness']:.1%}")
    restore_base_weights(model, base_weights)

    rng_key = mx.random.key(SEED + seed_offset + 101)
    n = premerge_single_adapter_dare(
        model, skeleton, adapters["code"], "code",
        OPTIMAL_SCALES["code"], drop_rate, rng_key
    )
    code_indist = eval_code_passrate(model, tokenizer, indist["code"])
    log(f"  Code in-dist pass rate: {code_indist['passed']}/{code_indist['total']} = {code_indist['pass_rate']:.1%}")
    restore_base_weights(model, base_weights)

    elapsed = time.time() - t0
    results = {
        "drop_rate": drop_rate,
        "gsm8k": gsm8k_results,
        "code_gen": code_results,
        "mmlu": mmlu_results,
        "in_distribution": {
            "math_correctness": math_indist,
            "code_pass_rate": code_indist,
        },
        "degenerate_output": is_degenerate,
        "time_s": round(elapsed, 1),
    }
    cleanup(model, tokenizer, skeleton, adapters, base_weights)
    return results


# ============================================================================
# Kill criteria assessment
# ============================================================================

def assess_kill_criteria(base_results, no_dare_results, dare_results_by_p):
    """Assess kill criteria across all DARE drop rates."""
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    base_gsm8k = base_results["gsm8k"]["accuracy"]
    base_code = base_results["code_gen"]["syntax_rate"]
    base_mmlu = base_results["mmlu"]["overall_accuracy"]

    no_dare_gsm8k = no_dare_results["gsm8k"]["accuracy"]
    no_dare_code = no_dare_results["code_gen"]["syntax_rate"]
    no_dare_mmlu = no_dare_results["mmlu"]["overall_accuracy"]
    no_dare_math_indist = no_dare_results["in_distribution"]["math_correctness"]["correctness"]
    no_dare_code_indist = no_dare_results["in_distribution"]["code_pass_rate"]["pass_rate"]

    log(f"\n  Base: GSM8K={base_gsm8k:.1%}, Code={base_code:.1%}, MMLU={base_mmlu:.1%}")
    log(f"  No-DARE: GSM8K={no_dare_gsm8k:.1%}, Code={no_dare_code:.1%}, MMLU={no_dare_mmlu:.1%}")
    log(f"  No-DARE in-dist: math={no_dare_math_indist:.1%}, code={no_dare_code_indist:.1%}")

    # OOD benchmarks for kill criteria assessment
    # 5 domains: GSM8K, code_gen, MMLU_medical, MMLU_code, MMLU_math
    # (We use MMLU by-domain since that's where degradation was measured)
    ood_benchmarks = ["gsm8k", "code_gen", "mmlu_medical", "mmlu_code", "mmlu_math"]

    def get_ood_degradations(results, base):
        """Return per-benchmark degradation in pp."""
        degs = {}
        degs["gsm8k"] = (results["gsm8k"]["accuracy"] - base["gsm8k"]["accuracy"]) * 100
        degs["code_gen"] = (results["code_gen"]["syntax_rate"] - base["code_gen"]["syntax_rate"]) * 100
        for mmlu_dom in ["medical", "code", "math"]:
            if mmlu_dom in results["mmlu"]["by_domain"]:
                degs[f"mmlu_{mmlu_dom}"] = (
                    results["mmlu"]["by_domain"][mmlu_dom]["accuracy"]
                    - base["mmlu"]["by_domain"][mmlu_dom]["accuracy"]
                ) * 100
        return degs

    assessment = {}

    for p, dare_results in sorted(dare_results_by_p.items()):
        log(f"\n  --- DARE p={p} ---")

        dare_gsm8k = dare_results["gsm8k"]["accuracy"]
        dare_code = dare_results["code_gen"]["syntax_rate"]
        dare_mmlu = dare_results["mmlu"]["overall_accuracy"]
        dare_math_indist = dare_results["in_distribution"]["math_correctness"]["correctness"]
        dare_code_indist = dare_results["in_distribution"]["code_pass_rate"]["pass_rate"]

        log(f"  OOD: GSM8K={dare_gsm8k:.1%}, Code={dare_code:.1%}, MMLU={dare_mmlu:.1%}")
        log(f"  In-dist: math={dare_math_indist:.1%}, code={dare_code_indist:.1%}")

        # K681: OOD degradation >= 5pp on majority (>=3/5) domains
        degs = get_ood_degradations(dare_results, base_results)
        n_degraded_5pp = sum(1 for v in degs.values() if v <= -5.0)
        k681_fail = n_degraded_5pp >= 3
        log(f"  K681: {n_degraded_5pp}/5 domains degrade >=5pp -> {'FAIL' if k681_fail else 'PASS'}")
        for bm, deg in degs.items():
            log(f"    {bm}: {deg:+.1f}pp {'(>=5pp)' if deg <= -5.0 else ''}")

        # K682: In-dist gains < 50% of no-DARE
        if no_dare_math_indist > 0:
            math_ratio = dare_math_indist / no_dare_math_indist
        else:
            math_ratio = 1.0
        if no_dare_code_indist > 0:
            code_ratio = dare_code_indist / no_dare_code_indist
        else:
            code_ratio = 1.0
        k682_fail = math_ratio < 0.5 or code_ratio < 0.5
        log(f"  K682: math_ratio={math_ratio:.2f}, code_ratio={code_ratio:.2f} -> {'FAIL' if k682_fail else 'PASS'}")

        # K683: Degenerate output
        k683_fail = dare_results.get("degenerate_output", False)
        log(f"  K683: degenerate={'YES' if k683_fail else 'NO'} -> {'FAIL' if k683_fail else 'PASS'}")

        assessment[p] = {
            "ood_degradations": degs,
            "n_degraded_5pp": n_degraded_5pp,
            "k681_fail": k681_fail,
            "math_indist_ratio": round(math_ratio, 3),
            "code_indist_ratio": round(code_ratio, 3),
            "k682_fail": k682_fail,
            "k683_fail": k683_fail,
            "any_kill": k681_fail or k682_fail or k683_fail,
        }

    # Find best drop rate (fewest kills, then fewest degraded domains)
    best_p = None
    best_score = None
    for p, a in sorted(assessment.items()):
        score = (int(a["any_kill"]), a["n_degraded_5pp"], -a["math_indist_ratio"])
        if best_score is None or score < best_score:
            best_score = score
            best_p = p

    log(f"\n  Best drop rate: p={best_p}")
    log(f"  Assessment: {assessment[best_p]}")

    return assessment, best_p


# ============================================================================
# Main
# ============================================================================

def main():
    t_total = time.time()
    results = {
        "experiment": "dare_sparsified_composition",
        "model": MODEL_ID,
        "drop_rates": DROP_RATES,
        "domains": DOMAINS,
        "scales": OPTIMAL_SCALES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    log("=" * 70)
    log("DARE Sparsified Adapter Composition")
    log("=" * 70)
    log(f"  Model: {MODEL_ID}")
    log(f"  Drop rates: {DROP_RATES}")
    log(f"  Benchmarks: GSM8K ({GSM8K_N}), Code Gen ({CODE_GEN_N}), MMLU ({MMLU_N_PER_DOMAIN}/domain)")

    # Phase 0: Load all benchmark data
    gsm8k, code_gen, mmlu, indist = phase_load_data()

    # Phase 1: Base model evaluation
    base_results = phase_eval_base(gsm8k, code_gen, mmlu)
    results["base"] = base_results

    # Save intermediate
    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

    # Phase 2: NTP adapters without DARE (baseline)
    no_dare_results = phase_eval_ntp_no_dare(gsm8k, code_gen, mmlu, indist)
    results["ntp_no_dare"] = no_dare_results

    # Save intermediate
    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

    # Phase 3+: DARE at each drop rate
    dare_results = {}
    for drop_rate in DROP_RATES:
        dr = phase_eval_dare(gsm8k, code_gen, mmlu, indist, drop_rate)
        dare_results[drop_rate] = dr
        results[f"dare_p{drop_rate}"] = dr

        # Save intermediate after each drop rate
        RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

    # Kill criteria assessment
    assessment, best_p = assess_kill_criteria(base_results, no_dare_results, dare_results)
    results["kill_criteria_assessment"] = {str(k): v for k, v in assessment.items()}
    results["best_drop_rate"] = best_p

    # Overall verdict
    best_assessment = assessment[best_p]
    if best_assessment["any_kill"]:
        verdict = "KILLED"
    else:
        verdict = "SUPPORTED"

    results["verdict"] = verdict
    results["total_time_s"] = round(time.time() - t_total, 1)

    log("\n" + "=" * 70)
    log("FINAL VERDICT")
    log("=" * 70)
    log(f"  Best drop rate: p={best_p}")
    log(f"  K681 (OOD >=5pp majority): {'FAIL' if best_assessment['k681_fail'] else 'PASS'}")
    log(f"  K682 (in-dist <50%): {'FAIL' if best_assessment['k682_fail'] else 'PASS'}")
    log(f"  K683 (degenerate): {'FAIL' if best_assessment['k683_fail'] else 'PASS'}")
    log(f"  Verdict: {verdict}")
    log(f"  Total time: {results['total_time_s']:.0f}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
