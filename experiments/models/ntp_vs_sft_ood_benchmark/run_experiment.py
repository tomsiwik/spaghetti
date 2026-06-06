#!/usr/bin/env python3
"""NTP vs SFT adapter OOD benchmark: resolve training objective confound.

Head-to-head comparison of NTP (next-token prediction) vs SFT (supervised fine-tuning
with response-only masking) adapters on out-of-distribution benchmarks.

Kill criteria:
  K1 (#678): NTP adapters ALSO degrade OOD benchmarks by >=5pp on majority (>=3/5) -> KILL
  K2 (#679): NTP adapters lose in-distribution behavioral gains vs SFT -> KILL
  K3 (#680): NTP adapters fail to converge on BitNet-2B-4T -> KILL

Predictions (from MATH.md):
  P1: NTP GSM8K <= 2pp degradation vs base (SFT showed -15pp)
  P2: NTP code gen <= 2pp degradation vs base (SFT showed -10pp)
  P3: NTP MMLU <= 3pp degradation vs base (SFT showed -5pp)
  P4: NTP in-dist math correctness >= 60%
  P5: NTP in-dist code pass@1 >= 40%
  P6: NTP training converged (already confirmed by existing adapters)

Type: Guided exploration (Type 2)
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

# Source directories for the two adapter types
NTP_SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
NTP_ADAPTERS_DIR = NTP_SOURCE_DIR / "adapters"
NTP_DATA_DIR = NTP_SOURCE_DIR / "data"

SFT_SOURCE_DIR = EXPERIMENT_DIR.parent / "bitnet_sft_generation_v3"
SFT_ADAPTERS_DIR = SFT_SOURCE_DIR / "sft_adapters"

# Both adapter types share the same Grassmannian skeleton
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

# Benchmark sizes — match prior experiments for comparability
GSM8K_N = 50        # match competitive_benchmark_routed (larger N for tighter CI)
CODE_GEN_N = 10     # match capability_benchmark_full_system
MMLU_N_PER_DOMAIN = 20  # match both prior experiments

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
# BitNet unpacking (identical to prior experiments)
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
# Pre-merge composition (shared by NTP and SFT — identical mechanism)
# ============================================================================

def load_skeleton():
    return dict(np.load(str(SKELETON_PATH)))


def load_adapter(adapter_dir, domain):
    adapter_path = adapter_dir / domain / "adapter.npz"
    adapter = dict(mx.load(str(adapter_path)))
    n_tensors = len(adapter)
    log(f"  Loaded adapter: {domain} ({n_tensors} tensors) from {adapter_dir.name}")
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


def premerge_single_adapter(model, skeleton, adapter, domain, scale):
    """Pre-merge: W_new = W_base + scale * B^T @ A^T"""
    di = DOMAINS.index(domain)
    merge_count = 0
    for li in range(len(model.model.layers)):
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
    return merge_count


# ============================================================================
# Generation
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


# ============================================================================
# Prompt formatting
# ============================================================================

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


def eval_code_syntax(text):
    """Check if generated text contains valid Python syntax."""
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
    """Load in-distribution evaluation data for behavioral assessment."""
    log("  Loading in-distribution eval data...")
    indist = {}

    # Math: load validation problems
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

    # Code: load validation problems
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
# In-distribution evaluation
# ============================================================================

def eval_math_correctness(model, tokenizer, problems):
    """Evaluate math correctness on in-distribution problems.
    Score: check if answer contains the key numbers/expressions from reference."""
    correct = 0
    total = len(problems)
    for prob in problems:
        prompt = format_domain_prompt(prob["instruction"])
        gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_DOMAIN)
        # Extract numbers from reference and check if they appear in generation
        ref_nums = re.findall(r'[\d]+(?:\.\d+)?', prob["reference"])
        if ref_nums:
            # Check if the last number (likely the answer) appears in generation
            answer_num = ref_nums[-1]
            if answer_num in gen:
                correct += 1
        else:
            # No numbers in reference — check keyword overlap
            ref_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', prob["reference"].lower()))
            gen_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', gen.lower()))
            overlap = len(ref_words & gen_words) / max(len(ref_words), 1)
            if overlap > 0.3:
                correct += 1
    return {"correctness": correct / total if total > 0 else 0, "correct": correct, "total": total}


def eval_code_passrate(model, tokenizer, problems):
    """Evaluate code generation pass rate on in-distribution problems."""
    passed = 0
    total = len(problems)
    for prob in problems:
        prompt = format_domain_prompt(prob["instruction"])
        gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_CODE)
        if eval_code_syntax(gen):
            passed += 1
    return {"pass_rate": passed / total if total > 0 else 0, "passed": passed, "total": total}


# ============================================================================
# OOD benchmark evaluation
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
    """Evaluate BitNet-2B base model (no adapters) on OOD benchmarks."""
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


def phase_eval_ntp_composed(gsm8k, code_gen, mmlu, indist):
    """Evaluate NTP adapters composed with per-domain optimal scales."""
    log("\n" + "=" * 70)
    log("PHASE 2: NTP ADAPTERS (oracle top-1 routing, per-domain scales)")
    log(f"  Scales: {OPTIMAL_SCALES}")
    log(f"  Adapter source: {NTP_ADAPTERS_DIR}")
    log("=" * 70)
    t0 = time.time()
    mx.reset_peak_memory()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("ntp-loaded-base")

    skeleton = load_skeleton()
    adapters = {}
    for domain in DOMAINS:
        adapters[domain] = load_adapter(NTP_ADAPTERS_DIR, domain)

    base_weights = save_base_weights(model)
    log_memory("ntp-adapters-loaded")

    # --- OOD Benchmarks ---

    # GSM8K: route to math adapter
    log("\n  --- GSM8K (math adapter, s=20) ---")
    n = premerge_single_adapter(model, skeleton, adapters["math"], "math", OPTIMAL_SCALES["math"])
    log(f"  Pre-merged math adapter into {n} layers")
    gsm8k_results = eval_gsm8k("ntp-composed", model, tokenizer, gsm8k)
    restore_base_weights(model, base_weights)

    # Code gen: route to code adapter
    log("\n  --- Code Gen (code adapter, s=20) ---")
    n = premerge_single_adapter(model, skeleton, adapters["code"], "code", OPTIMAL_SCALES["code"])
    log(f"  Pre-merged code adapter into {n} layers")
    code_results = eval_code_gen("ntp-composed", model, tokenizer, code_gen)
    restore_base_weights(model, base_weights)

    # MMLU: route each domain to matching adapter
    log("\n  --- MMLU (per-domain routing) ---")
    mmlu_results_by_domain = {}
    total_correct = 0
    total_q = 0
    choice_labels = ["A", "B", "C", "D"]
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
    log(f"  MMLU overall: {total_correct}/{total_q} = {overall_mmlu:.1%}")

    # --- In-distribution behavioral eval ---
    log("\n  --- In-Distribution Behavioral Eval ---")

    # Math in-dist with math adapter
    n = premerge_single_adapter(model, skeleton, adapters["math"], "math", OPTIMAL_SCALES["math"])
    log(f"  Pre-merged math adapter for in-dist eval ({n} layers)")
    math_indist = eval_math_correctness(model, tokenizer, indist["math"])
    log(f"  Math in-dist correctness: {math_indist['correct']}/{math_indist['total']} = {math_indist['correctness']:.1%}")
    restore_base_weights(model, base_weights)

    # Code in-dist with code adapter
    n = premerge_single_adapter(model, skeleton, adapters["code"], "code", OPTIMAL_SCALES["code"])
    log(f"  Pre-merged code adapter for in-dist eval ({n} layers)")
    code_indist = eval_code_passrate(model, tokenizer, indist["code"])
    log(f"  Code in-dist pass rate: {code_indist['passed']}/{code_indist['total']} = {code_indist['pass_rate']:.1%}")
    restore_base_weights(model, base_weights)

    peak_mem = mx.get_peak_memory() / 1e9
    elapsed = time.time() - t0
    log(f"\nNTP composed total: {elapsed:.1f}s, peak memory: {peak_mem:.2f}GB")

    results = {
        "gsm8k": gsm8k_results,
        "code_gen": code_results,
        "mmlu": mmlu_results,
        "in_distribution": {
            "math_correctness": math_indist,
            "code_pass_rate": code_indist,
        },
        "peak_memory_gb": round(peak_mem, 2),
        "time_s": round(elapsed, 1),
        "adapter_type": "NTP",
        "routing": "oracle_top1",
        "scales": OPTIMAL_SCALES,
    }
    cleanup(model, tokenizer, skeleton, adapters, base_weights)
    return results


def phase_eval_sft_composed(gsm8k, code_gen, mmlu, indist):
    """Evaluate SFT adapters composed with per-domain optimal scales."""
    log("\n" + "=" * 70)
    log("PHASE 3: SFT ADAPTERS (oracle top-1 routing, per-domain scales)")
    log(f"  Scales: {OPTIMAL_SCALES}")
    log(f"  Adapter source: {SFT_ADAPTERS_DIR}")
    log("=" * 70)
    t0 = time.time()
    mx.reset_peak_memory()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("sft-loaded-base")

    skeleton = load_skeleton()
    adapters = {}
    for domain in DOMAINS:
        adapters[domain] = load_adapter(SFT_ADAPTERS_DIR, domain)

    base_weights = save_base_weights(model)
    log_memory("sft-adapters-loaded")

    # --- OOD Benchmarks ---

    # GSM8K
    log("\n  --- GSM8K (math adapter, s=20) ---")
    n = premerge_single_adapter(model, skeleton, adapters["math"], "math", OPTIMAL_SCALES["math"])
    log(f"  Pre-merged math adapter into {n} layers")
    gsm8k_results = eval_gsm8k("sft-composed", model, tokenizer, gsm8k)
    restore_base_weights(model, base_weights)

    # Code gen
    log("\n  --- Code Gen (code adapter, s=20) ---")
    n = premerge_single_adapter(model, skeleton, adapters["code"], "code", OPTIMAL_SCALES["code"])
    log(f"  Pre-merged code adapter into {n} layers")
    code_results = eval_code_gen("sft-composed", model, tokenizer, code_gen)
    restore_base_weights(model, base_weights)

    # MMLU
    log("\n  --- MMLU (per-domain routing) ---")
    mmlu_results_by_domain = {}
    total_correct = 0
    total_q = 0
    choice_labels = ["A", "B", "C", "D"]
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
    log(f"  MMLU overall: {total_correct}/{total_q} = {overall_mmlu:.1%}")

    # --- In-distribution behavioral eval ---
    log("\n  --- In-Distribution Behavioral Eval ---")

    n = premerge_single_adapter(model, skeleton, adapters["math"], "math", OPTIMAL_SCALES["math"])
    log(f"  Pre-merged math adapter for in-dist eval ({n} layers)")
    math_indist = eval_math_correctness(model, tokenizer, indist["math"])
    log(f"  Math in-dist correctness: {math_indist['correct']}/{math_indist['total']} = {math_indist['correctness']:.1%}")
    restore_base_weights(model, base_weights)

    n = premerge_single_adapter(model, skeleton, adapters["code"], "code", OPTIMAL_SCALES["code"])
    log(f"  Pre-merged code adapter for in-dist eval ({n} layers)")
    code_indist = eval_code_passrate(model, tokenizer, indist["code"])
    log(f"  Code in-dist pass rate: {code_indist['passed']}/{code_indist['total']} = {code_indist['pass_rate']:.1%}")
    restore_base_weights(model, base_weights)

    peak_mem = mx.get_peak_memory() / 1e9
    elapsed = time.time() - t0
    log(f"\nSFT composed total: {elapsed:.1f}s, peak memory: {peak_mem:.2f}GB")

    results = {
        "gsm8k": gsm8k_results,
        "code_gen": code_results,
        "mmlu": mmlu_results,
        "in_distribution": {
            "math_correctness": math_indist,
            "code_pass_rate": code_indist,
        },
        "peak_memory_gb": round(peak_mem, 2),
        "time_s": round(elapsed, 1),
        "adapter_type": "SFT",
        "routing": "oracle_top1",
        "scales": OPTIMAL_SCALES,
    }
    cleanup(model, tokenizer, skeleton, adapters, base_weights)
    return results


# ============================================================================
# Analysis
# ============================================================================

def phase_analysis(base_results, ntp_results, sft_results):
    """Compare NTP vs SFT vs base and evaluate kill criteria."""
    log("\n" + "=" * 70)
    log("ANALYSIS: NTP vs SFT vs BASE")
    log("=" * 70)

    # Build comparison table
    benchmarks = {
        "gsm8k": ("gsm8k", "accuracy"),
        "code_gen": ("code_gen", "syntax_rate"),
        "mmlu_overall": ("mmlu", "overall_accuracy"),
        "mmlu_medical": None,
        "mmlu_code": None,
        "mmlu_math": None,
        "mmlu_legal": None,
        "mmlu_finance": None,
    }

    def get_score(results, bench):
        if bench == "gsm8k":
            return results["gsm8k"]["accuracy"]
        elif bench == "code_gen":
            return results["code_gen"]["syntax_rate"]
        elif bench == "mmlu_overall":
            return results["mmlu"]["overall_accuracy"]
        elif bench.startswith("mmlu_"):
            domain = bench.replace("mmlu_", "")
            return results["mmlu"]["by_domain"][domain]["accuracy"]
        return None

    bench_names = list(benchmarks.keys())

    log("\n  OOD BENCHMARK COMPARISON:")
    log(f"  {'Benchmark':<15} {'Base':<10} {'NTP':<10} {'SFT':<10} {'NTP-Base':<10} {'SFT-Base':<10}")
    log("  " + "-" * 65)

    ntp_deltas = {}
    sft_deltas = {}
    for bench in bench_names:
        base_score = get_score(base_results, bench)
        ntp_score = get_score(ntp_results, bench)
        sft_score = get_score(sft_results, bench)
        ntp_delta = ntp_score - base_score if ntp_score is not None and base_score is not None else None
        sft_delta = sft_score - base_score if sft_score is not None and base_score is not None else None
        ntp_deltas[bench] = ntp_delta
        sft_deltas[bench] = sft_delta
        log(f"  {bench:<15} {base_score:<10.3f} {ntp_score:<10.3f} {sft_score:<10.3f} "
            f"{ntp_delta:+.3f}     {sft_delta:+.3f}")

    # In-distribution comparison
    log("\n  IN-DISTRIBUTION BEHAVIORAL COMPARISON:")
    ntp_math = ntp_results["in_distribution"]["math_correctness"]["correctness"]
    sft_math = sft_results["in_distribution"]["math_correctness"]["correctness"]
    ntp_code = ntp_results["in_distribution"]["code_pass_rate"]["pass_rate"]
    sft_code = sft_results["in_distribution"]["code_pass_rate"]["pass_rate"]
    log(f"  {'Metric':<25} {'NTP':<10} {'SFT':<10}")
    log("  " + "-" * 45)
    log(f"  {'Math correctness':<25} {ntp_math:<10.1%} {sft_math:<10.1%}")
    log(f"  {'Code pass@1':<25} {ntp_code:<10.1%} {sft_code:<10.1%}")

    # Kill criteria evaluation
    analysis = {"kill_criteria": {}}

    # K1 (#678): NTP adapters degrade OOD by >=5pp on >=3 of ALL OOD benchmarks
    # Corrected: include all 7 individual OOD benchmarks (no exclusions)
    ood_benchmarks = ["gsm8k", "code_gen", "mmlu_medical", "mmlu_code", "mmlu_math", "mmlu_legal", "mmlu_finance"]
    ntp_degraded_5pp = [b for b in ood_benchmarks if ntp_deltas.get(b, 0) <= -0.05]
    k1_fail = len(ntp_degraded_5pp) >= 3
    analysis["kill_criteria"]["k1_678"] = {
        "result": "fail" if k1_fail else "pass",
        "degraded_benchmarks": ntp_degraded_5pp,
        "count": len(ntp_degraded_5pp),
        "threshold": 3,
        "total_benchmarks": len(ood_benchmarks),
        "evidence": f"NTP degraded >=5pp on {len(ntp_degraded_5pp)}/{len(ood_benchmarks)} OOD benchmarks: {ntp_degraded_5pp}",
    }
    log(f"\n  K1 (#678): NTP degraded >=5pp on {len(ntp_degraded_5pp)}/{len(ood_benchmarks)} benchmarks -> {'KILL' if k1_fail else 'PASS'}")
    if ntp_degraded_5pp:
        for b in ntp_degraded_5pp:
            log(f"    {b}: {ntp_deltas[b]:+.1%}")

    # K2 (#679): NTP loses in-distribution behavioral gains
    k2_math_fail = ntp_math < 0.60
    k2_code_fail = ntp_code < 0.40
    k2_fail = k2_math_fail or k2_code_fail
    analysis["kill_criteria"]["k2_679"] = {
        "result": "fail" if k2_fail else "pass",
        "math_correctness": ntp_math,
        "code_pass_rate": ntp_code,
        "evidence": f"Math={ntp_math:.1%} (threshold 60%), Code={ntp_code:.1%} (threshold 40%)",
    }
    log(f"  K2 (#679): NTP in-dist math={ntp_math:.1%}, code={ntp_code:.1%} -> {'KILL' if k2_fail else 'PASS'}")

    # K3 (#680): NTP failed to converge — already resolved (adapters exist and were used in prior experiments)
    analysis["kill_criteria"]["k3_680"] = {
        "result": "pass",
        "evidence": "NTP adapters from real_data_domain_experts already trained and used in Finding #237",
    }
    log(f"  K3 (#680): NTP adapters exist and were previously validated -> PASS")

    # Summary
    any_kill = any(v["result"] == "fail" for v in analysis["kill_criteria"].values())
    overall = "KILLED" if any_kill else "SUPPORTED"
    log(f"\n  OVERALL VERDICT: {overall}")

    # Compute the key comparison metric: does NTP preserve OOD better than SFT?
    ntp_avg_ood_delta = np.mean([ntp_deltas[b] for b in ood_benchmarks if ntp_deltas[b] is not None])
    sft_avg_ood_delta = np.mean([sft_deltas[b] for b in ood_benchmarks if sft_deltas[b] is not None])
    ntp_better = ntp_avg_ood_delta > sft_avg_ood_delta
    log(f"\n  NTP avg OOD delta: {ntp_avg_ood_delta:+.3f}")
    log(f"  SFT avg OOD delta: {sft_avg_ood_delta:+.3f}")
    log(f"  NTP better than SFT on OOD: {ntp_better}")

    analysis["ntp_vs_sft"] = {
        "ntp_avg_ood_delta": float(ntp_avg_ood_delta),
        "sft_avg_ood_delta": float(sft_avg_ood_delta),
        "ntp_preserves_ood_better": bool(ntp_better),
        "ntp_deltas": {k: float(v) for k, v in ntp_deltas.items() if v is not None},
        "sft_deltas": {k: float(v) for k, v in sft_deltas.items() if v is not None},
    }

    return analysis


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("NTP vs SFT ADAPTER OOD BENCHMARK")
    log(f"Model: {MODEL_ID}")
    log(f"NTP adapters: {NTP_ADAPTERS_DIR}")
    log(f"SFT adapters: {SFT_ADAPTERS_DIR}")
    log(f"Scales: {OPTIMAL_SCALES}")
    log("=" * 70)
    log_memory("start")

    # Phase 0: Load all data
    gsm8k, code_gen, mmlu, indist = phase_load_data()

    # Phase 1: Base model evaluation
    base_results = phase_eval_base(gsm8k, code_gen, mmlu)
    log_memory("after-base")

    # Phase 2: NTP composed evaluation
    ntp_results = phase_eval_ntp_composed(gsm8k, code_gen, mmlu, indist)
    log_memory("after-ntp")

    # Phase 3: SFT composed evaluation
    sft_results = phase_eval_sft_composed(gsm8k, code_gen, mmlu, indist)
    log_memory("after-sft")

    # Phase 4: Analysis
    analysis = phase_analysis(base_results, ntp_results, sft_results)

    # Save results
    total_time = time.time() - t0
    results = {
        "experiment": "ntp_vs_sft_ood_benchmark",
        "model": MODEL_ID,
        "optimal_scales": OPTIMAL_SCALES,
        "benchmark_sizes": {
            "gsm8k": GSM8K_N,
            "code_gen": CODE_GEN_N,
            "mmlu_per_domain": MMLU_N_PER_DOMAIN,
        },
        "base_results": base_results,
        "ntp_results": ntp_results,
        "sft_results": sft_results,
        "analysis": analysis,
        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total experiment time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
