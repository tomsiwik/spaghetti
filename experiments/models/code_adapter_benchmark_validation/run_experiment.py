#!/usr/bin/env python3
"""Code Adapter Benchmark Validation: MMLU/GSM8K/HumanEval on BitNet-2B + code LoRA.

Kill criteria:
  K614: Code adapter DEGRADES MMLU accuracy vs base (worse by >2%) -> KILL
  K615: Code adapter shows <5% improvement on GSM8K AND HumanEval vs base -> KILL
  K616: All benchmark scores below random baseline (base model too weak) -> KILL

Type: guided-exploration
Platform: Apple M5 Pro 48GB, MLX

Compares: (1) base BitNet-2B-4T vs (2) base + code SFT adapter
Benchmarks: MMLU (50 questions), GSM8K (50 problems), HumanEval (20 problems)
"""

import ast
import gc
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
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

# Existing infrastructure paths
SFT_DIR = EXPERIMENT_DIR.parent / "bitnet_sft_generation_v3"
SFT_ADAPTERS_DIR = SFT_DIR / "sft_adapters"
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
NTP_ADAPTERS_DIR = SOURCE_DIR / "adapters"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]

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
# Model utilities (from behavioral_eval_framework)
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


class TernaryLoRALinear(nn.Module):
    """LoRA with STE-ternary B and optional Grassmannian A."""
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


def load_base_model():
    """Load and prepare base model (BitLinear unpacked)."""
    log("Loading BitNet-2B-4T base model...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("base-model-loaded")
    return model, tokenizer


def load_skeleton():
    skeleton_path = NTP_ADAPTERS_DIR / "grassmannian_skeleton.npz"
    return dict(np.load(str(skeleton_path)))


def apply_code_adapter(model, skeleton):
    """Apply code SFT adapter (domain_idx=1 for 'code' in DOMAINS list)."""
    code_domain_idx = DOMAINS.index("code")  # = 1
    adapter_path = SFT_ADAPTERS_DIR / "code" / "adapter.npz"

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
            skey = f"layer_{li}_{key}_domain_{code_domain_idx}"
            a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16) if skey in skeleton else None
            lora = TernaryLoRALinear(module, rank=LORA_RANK, scale=LORA_SCALE, a_init=a_mx)
            lora_updates.append((key, lora))
            count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    # Load trained B-matrices
    adapter_params = dict(mx.load(str(adapter_path)))
    model.update(tree_unflatten(list(adapter_params.items())))
    mx.eval(model.parameters())
    log(f"  Applied code adapter: {count} LoRA modules")
    return model


def generate_text(model, tokenizer, prompt, max_tokens=256, temperature=0.0):
    """Generate text from prompt."""
    try:
        sampler = make_sampler(temp=temperature)
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
# Benchmark data loading
# ============================================================================

def load_mmlu_questions(n=50):
    """Load MMLU questions from HuggingFace datasets."""
    from datasets import load_dataset
    log("Loading MMLU dataset...")
    # Use a diverse set of subjects
    subjects = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics",
        "computer_security", "conceptual_physics", "econometrics",
        "electrical_engineering", "elementary_mathematics", "formal_logic",
        "global_facts", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_european_history",
        "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_mathematics",
        "high_school_microeconomics", "high_school_physics",
        "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history",
        "human_aging", "human_sexuality", "international_law",
        "jurisprudence", "logical_fallacies", "machine_learning",
        "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios",
        "nutrition", "philosophy", "prehistory",
        "professional_accounting", "professional_law",
        "professional_medicine", "professional_psychology",
    ]

    questions = []
    # Collect from validation split for each subject, round-robin
    per_subject = max(1, n // len(subjects))
    remaining = n

    for subj in subjects:
        if remaining <= 0:
            break
        try:
            ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
            take = min(per_subject, len(ds), remaining)
            for i in range(take):
                item = ds[i]
                questions.append({
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": item["answer"],  # 0-3 index
                    "subject": subj,
                })
                remaining -= 1
        except Exception as e:
            log(f"  Warning: could not load {subj}: {e}")
            continue

    log(f"  Loaded {len(questions)} MMLU questions across {len(set(q['subject'] for q in questions))} subjects")
    return questions


def load_gsm8k_problems(n=50):
    """Load GSM8K math problems."""
    from datasets import load_dataset
    log("Loading GSM8K dataset...")
    ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)

    # Take first n problems
    problems = []
    for i in range(min(n, len(ds))):
        item = ds[i]
        # Extract numerical answer from "#### X" format
        answer_text = item["answer"]
        match = re.search(r'####\s*([\-\d,]+(?:\.\d+)?)', answer_text)
        if match:
            answer_num = float(match.group(1).replace(',', ''))
        else:
            # Try last number
            nums = re.findall(r'[\-\d,]+(?:\.\d+)?', answer_text)
            answer_num = float(nums[-1].replace(',', '')) if nums else None

        problems.append({
            "question": item["question"],
            "answer_text": answer_text,
            "answer_num": answer_num,
        })

    log(f"  Loaded {len(problems)} GSM8K problems")
    return problems


def load_humaneval_problems(n=20):
    """Load HumanEval coding problems."""
    from datasets import load_dataset
    log("Loading HumanEval dataset...")
    ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)

    problems = []
    for i in range(min(n, len(ds))):
        item = ds[i]
        problems.append({
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "canonical_solution": item["canonical_solution"],
            "test": item["test"],
            "entry_point": item["entry_point"],
        })

    log(f"  Loaded {len(problems)} HumanEval problems")
    return problems


# ============================================================================
# Evaluation functions
# ============================================================================

def format_mmlu_prompt(question, choices):
    """Format MMLU question for the model."""
    choice_labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(f"{label}) {text}" for label, text in zip(choice_labels, choices))
    return (
        f"### Instruction:\n"
        f"Answer the following multiple choice question. "
        f"Reply with ONLY the letter of the correct answer (A, B, C, or D).\n\n"
        f"Question: {question}\n"
        f"{choices_text}\n\n"
        f"### Response:\n"
    )


def extract_mmlu_answer(text):
    """Extract answer letter from model output."""
    text = text.strip()
    # Look for standalone letter at the start
    if text and text[0] in "ABCD":
        return text[0]
    # Look for "The answer is X" pattern
    m = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*([A-D])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Look for any A/B/C/D in the first line
    first_line = text.split('\n')[0]
    m = re.search(r'\b([A-D])\b', first_line)
    if m:
        return m.group(1)
    # Last resort: first letter found anywhere
    m = re.search(r'([A-D])', text)
    if m:
        return m.group(1)
    return None


def evaluate_mmlu(model, tokenizer, questions):
    """Evaluate model on MMLU questions. Returns accuracy."""
    choice_labels = ["A", "B", "C", "D"]
    correct = 0
    total = 0
    details = []

    for q in questions:
        prompt = format_mmlu_prompt(q["question"], q["choices"])
        output = generate_text(model, tokenizer, prompt, max_tokens=32, temperature=0.0)
        predicted = extract_mmlu_answer(output)
        correct_label = choice_labels[q["answer"]]
        is_correct = predicted == correct_label

        if is_correct:
            correct += 1
        total += 1

        details.append({
            "subject": q["subject"],
            "predicted": predicted,
            "correct": correct_label,
            "is_correct": is_correct,
            "output_preview": output[:100],
        })

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, details


def format_gsm8k_prompt(question):
    """Format GSM8K question for the model."""
    return (
        f"### Instruction:\n"
        f"Solve this math problem step by step. "
        f"End your answer with #### followed by the numerical answer.\n\n"
        f"{question}\n\n"
        f"### Response:\n"
    )


def extract_gsm8k_answer(text):
    """Extract numerical answer from GSM8K response."""
    # Pattern: "#### X"
    matches = re.findall(r'####\s*([\-\d,]+(?:\.\d+)?)', text)
    if matches:
        try:
            return float(matches[-1].replace(',', ''))
        except ValueError:
            pass
    # Pattern: "the answer is X"
    m = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*\$?([\-\d,]+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1).replace(',', ''))
        except ValueError:
            pass
    # Pattern: "= X" (last one)
    matches = re.findall(r'=\s*\$?([\-\d,]+(?:\.\d+)?)', text)
    if matches:
        try:
            return float(matches[-1].replace(',', ''))
        except ValueError:
            pass
    # Last number in text
    matches = re.findall(r'([\-\d,]+(?:\.\d+)?)', text)
    if matches:
        try:
            return float(matches[-1].replace(',', ''))
        except ValueError:
            pass
    return None


def evaluate_gsm8k(model, tokenizer, problems):
    """Evaluate model on GSM8K. Returns exact match accuracy."""
    correct = 0
    total = 0
    details = []

    for p in problems:
        prompt = format_gsm8k_prompt(p["question"])
        output = generate_text(model, tokenizer, prompt, max_tokens=300, temperature=0.0)
        predicted = extract_gsm8k_answer(output)
        expected = p["answer_num"]

        # Exact match with tolerance
        is_correct = False
        if predicted is not None and expected is not None:
            is_correct = abs(predicted - expected) < 0.01

        if is_correct:
            correct += 1
        total += 1

        details.append({
            "question_preview": p["question"][:100],
            "predicted": predicted,
            "expected": expected,
            "is_correct": is_correct,
            "output_preview": output[:200],
        })

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, details


def format_humaneval_prompt(prompt_text):
    """Format HumanEval problem for the model."""
    return (
        f"### Instruction:\n"
        f"Complete the following Python function. "
        f"Write ONLY the function body (no imports, no extra code).\n\n"
        f"```python\n{prompt_text}```\n\n"
        f"### Response:\n"
    )


def extract_code_completion(output, prompt_text, entry_point):
    """Extract function completion from model output."""
    # Try to find code block
    code_blocks = re.findall(r'```(?:python)?\s*\n?(.*?)```', output, re.DOTALL)
    if code_blocks:
        code = code_blocks[0].strip()
        # If the code contains the full function, use it
        if f"def {entry_point}" in code:
            return code
        # Otherwise, assume it's just the body
        return prompt_text + code

    # No code block — try to use raw output as function body
    # Strip leading explanation text
    lines = output.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('def ', 'return ', 'if ', 'for ', 'while ',
                                'try:', 'except', 'with ', 'import ', 'from ',
                                '#', '    ', '\t')) or in_code:
            in_code = True
            code_lines.append(line)
        elif stripped == '' and in_code:
            code_lines.append(line)

    if code_lines:
        code = '\n'.join(code_lines)
        if f"def {entry_point}" in code:
            return code
        return prompt_text + code

    # Fallback: append entire output as body
    return prompt_text + output


def execute_humaneval_test(code, test_code, entry_point, timeout=10):
    """Execute HumanEval test in a sandboxed subprocess."""
    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        passed = result.returncode == 0
        error = result.stderr[:500] if result.stderr else ""
        return passed, error
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)[:500]
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def evaluate_humaneval(model, tokenizer, problems):
    """Evaluate model on HumanEval. Returns pass@1."""
    passed = 0
    total = 0
    details = []

    for p in problems:
        prompt = format_humaneval_prompt(p["prompt"])
        output = generate_text(model, tokenizer, prompt, max_tokens=512, temperature=0.0)
        code = extract_code_completion(output, p["prompt"], p["entry_point"])
        success, error = execute_humaneval_test(code, p["test"], p["entry_point"])

        if success:
            passed += 1
        total += 1

        details.append({
            "task_id": p["task_id"],
            "passed": success,
            "error": error[:200] if error else "",
            "output_preview": output[:300],
        })

    accuracy = passed / total if total > 0 else 0.0
    return accuracy, details


# ============================================================================
# Phase 1: Load benchmark data
# ============================================================================

def phase_load_data():
    """Load all benchmark datasets."""
    log("\n" + "=" * 70)
    log("PHASE 0: LOADING BENCHMARK DATA")
    log("=" * 70)

    mmlu = load_mmlu_questions(n=50)
    gsm8k = load_gsm8k_problems(n=50)
    humaneval = load_humaneval_problems(n=20)

    return mmlu, gsm8k, humaneval


# ============================================================================
# Phase 2: Evaluate base model
# ============================================================================

def phase_eval_base(mmlu, gsm8k, humaneval):
    """Evaluate base BitNet-2B-4T without any adapter."""
    log("\n" + "=" * 70)
    log("PHASE 1: BASE MODEL EVALUATION")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = load_base_model()

    log("\n  --- MMLU (50 questions) ---")
    mmlu_acc, mmlu_details = evaluate_mmlu(model, tokenizer, mmlu)
    log(f"  MMLU accuracy: {mmlu_acc:.4f} ({int(mmlu_acc * len(mmlu))}/{len(mmlu)})")
    # Per-subject breakdown
    subjects = {}
    for d in mmlu_details:
        s = d["subject"]
        if s not in subjects:
            subjects[s] = {"correct": 0, "total": 0}
        subjects[s]["total"] += 1
        if d["is_correct"]:
            subjects[s]["correct"] += 1
    for s, v in sorted(subjects.items()):
        log(f"    {s}: {v['correct']}/{v['total']}")

    log("\n  --- GSM8K (50 problems) ---")
    gsm8k_acc, gsm8k_details = evaluate_gsm8k(model, tokenizer, gsm8k)
    log(f"  GSM8K accuracy: {gsm8k_acc:.4f} ({int(gsm8k_acc * len(gsm8k))}/{len(gsm8k)})")

    log("\n  --- HumanEval (20 problems) ---")
    humaneval_acc, humaneval_details = evaluate_humaneval(model, tokenizer, humaneval)
    log(f"  HumanEval pass@1: {humaneval_acc:.4f} ({int(humaneval_acc * len(humaneval))}/{len(humaneval)})")

    elapsed = time.time() - t0
    log(f"\n  Base model eval time: {elapsed:.1f}s")
    log_memory("base-eval-done")

    results = {
        "mmlu": {"accuracy": mmlu_acc, "details": mmlu_details},
        "gsm8k": {"accuracy": gsm8k_acc, "details": gsm8k_details},
        "humaneval": {"accuracy": humaneval_acc, "details": humaneval_details},
        "eval_time_s": round(elapsed, 1),
    }

    cleanup(model, tokenizer)
    return results


# ============================================================================
# Phase 3: Evaluate base + code adapter
# ============================================================================

def phase_eval_adapter(mmlu, gsm8k, humaneval):
    """Evaluate base BitNet-2B-4T + code SFT adapter."""
    log("\n" + "=" * 70)
    log("PHASE 2: CODE ADAPTER EVALUATION")
    log("=" * 70)
    t0 = time.time()

    skeleton = load_skeleton()
    model, tokenizer = load_base_model()
    model = apply_code_adapter(model, skeleton)
    del skeleton
    gc.collect()
    log_memory("code-adapter-loaded")

    log("\n  --- MMLU (50 questions) ---")
    mmlu_acc, mmlu_details = evaluate_mmlu(model, tokenizer, mmlu)
    log(f"  MMLU accuracy: {mmlu_acc:.4f} ({int(mmlu_acc * len(mmlu))}/{len(mmlu)})")
    subjects = {}
    for d in mmlu_details:
        s = d["subject"]
        if s not in subjects:
            subjects[s] = {"correct": 0, "total": 0}
        subjects[s]["total"] += 1
        if d["is_correct"]:
            subjects[s]["correct"] += 1
    for s, v in sorted(subjects.items()):
        log(f"    {s}: {v['correct']}/{v['total']}")

    log("\n  --- GSM8K (50 problems) ---")
    gsm8k_acc, gsm8k_details = evaluate_gsm8k(model, tokenizer, gsm8k)
    log(f"  GSM8K accuracy: {gsm8k_acc:.4f} ({int(gsm8k_acc * len(gsm8k))}/{len(gsm8k)})")

    log("\n  --- HumanEval (20 problems) ---")
    humaneval_acc, humaneval_details = evaluate_humaneval(model, tokenizer, humaneval)
    log(f"  HumanEval pass@1: {humaneval_acc:.4f} ({int(humaneval_acc * len(humaneval))}/{len(humaneval)})")

    elapsed = time.time() - t0
    log(f"\n  Adapter model eval time: {elapsed:.1f}s")
    log_memory("adapter-eval-done")

    results = {
        "mmlu": {"accuracy": mmlu_acc, "details": mmlu_details},
        "gsm8k": {"accuracy": gsm8k_acc, "details": gsm8k_details},
        "humaneval": {"accuracy": humaneval_acc, "details": humaneval_details},
        "eval_time_s": round(elapsed, 1),
    }

    cleanup(model, tokenizer)
    return results


# ============================================================================
# Kill criteria assessment
# ============================================================================

def assess_kill_criteria(base_results, adapter_results):
    """Assess all kill criteria and return verdict."""
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    base_mmlu = base_results["mmlu"]["accuracy"]
    adapter_mmlu = adapter_results["mmlu"]["accuracy"]
    base_gsm8k = base_results["gsm8k"]["accuracy"]
    adapter_gsm8k = adapter_results["gsm8k"]["accuracy"]
    base_humaneval = base_results["humaneval"]["accuracy"]
    adapter_humaneval = adapter_results["humaneval"]["accuracy"]

    mmlu_delta = adapter_mmlu - base_mmlu
    gsm8k_delta = adapter_gsm8k - base_gsm8k
    humaneval_delta = adapter_humaneval - base_humaneval

    log(f"\n  MMLU:      base={base_mmlu:.4f}  adapter={adapter_mmlu:.4f}  delta={mmlu_delta:+.4f}")
    log(f"  GSM8K:     base={base_gsm8k:.4f}  adapter={adapter_gsm8k:.4f}  delta={gsm8k_delta:+.4f}")
    log(f"  HumanEval: base={base_humaneval:.4f}  adapter={adapter_humaneval:.4f}  delta={humaneval_delta:+.4f}")

    # K614: Code adapter DEGRADES MMLU accuracy vs base (worse by >2%)
    k614_fail = mmlu_delta < -0.02
    k614_result = "FAIL" if k614_fail else "PASS"
    log(f"\n  K614 (MMLU non-degradation): {k614_result}")
    log(f"    Criterion: adapter MMLU >= base MMLU - 0.02")
    log(f"    Measured: {adapter_mmlu:.4f} {'<' if k614_fail else '>='} {base_mmlu - 0.02:.4f}")

    # K615: Code adapter shows <5% improvement on GSM8K AND HumanEval vs base
    gsm8k_improved = gsm8k_delta >= 0.05
    humaneval_improved = humaneval_delta >= 0.05
    k615_fail = (not gsm8k_improved) and (not humaneval_improved)
    k615_result = "FAIL" if k615_fail else "PASS"
    log(f"\n  K615 (Structured task improvement): {k615_result}")
    log(f"    Criterion: GSM8K delta >= 5pp OR HumanEval delta >= 5pp")
    log(f"    Measured: GSM8K delta={gsm8k_delta:+.4f} {'YES' if gsm8k_improved else 'NO'}")
    log(f"             HumanEval delta={humaneval_delta:+.4f} {'YES' if humaneval_improved else 'NO'}")

    # K616: All benchmark scores below random baseline
    mmlu_above_random = base_mmlu >= 0.25 or adapter_mmlu >= 0.25
    gsm8k_above_floor = base_gsm8k >= 0.05 or adapter_gsm8k >= 0.05
    humaneval_above_floor = base_humaneval > 0.0 or adapter_humaneval > 0.0
    k616_fail = not (mmlu_above_random or gsm8k_above_floor or humaneval_above_floor)
    k616_result = "FAIL" if k616_fail else "PASS"
    log(f"\n  K616 (Above random baseline): {k616_result}")
    log(f"    MMLU above 25%: {'YES' if mmlu_above_random else 'NO'}")
    log(f"    GSM8K above 5%: {'YES' if gsm8k_above_floor else 'NO'}")
    log(f"    HumanEval > 0%: {'YES' if humaneval_above_floor else 'NO'}")

    any_kill = k614_fail or k615_fail or k616_fail
    overall = "KILLED" if any_kill else "SUPPORTED"
    log(f"\n  OVERALL: {overall}")

    return {
        "k614": {"result": k614_result, "mmlu_delta": mmlu_delta},
        "k615": {"result": k615_result, "gsm8k_delta": gsm8k_delta, "humaneval_delta": humaneval_delta},
        "k616": {"result": k616_result, "mmlu_above_random": mmlu_above_random},
        "overall": overall,
        "deltas": {
            "mmlu": mmlu_delta,
            "gsm8k": gsm8k_delta,
            "humaneval": humaneval_delta,
        },
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t0_total = time.time()
    log("=" * 70)
    log("CODE ADAPTER BENCHMARK VALIDATION")
    log("BitNet-2B-4T base vs base + code SFT adapter")
    log("Benchmarks: MMLU (50), GSM8K (50), HumanEval (20)")
    log("=" * 70)
    log_memory("start")

    mx.random.seed(SEED)
    np.random.seed(SEED)

    # Phase 0: Load data
    mmlu, gsm8k, humaneval = phase_load_data()

    # Phase 1: Base model
    base_results = phase_eval_base(mmlu, gsm8k, humaneval)

    # Phase 2: Code adapter
    adapter_results = phase_eval_adapter(mmlu, gsm8k, humaneval)

    # Kill criteria
    kill = assess_kill_criteria(base_results, adapter_results)

    # Save results
    total_time = time.time() - t0_total
    results = {
        "experiment": "code_adapter_benchmark_validation",
        "model": MODEL_ID,
        "adapter": "code_sft_adapter",
        "lora_rank": LORA_RANK,
        "lora_scale": LORA_SCALE,
        "base_results": {
            "mmlu_accuracy": base_results["mmlu"]["accuracy"],
            "gsm8k_accuracy": base_results["gsm8k"]["accuracy"],
            "humaneval_pass1": base_results["humaneval"]["accuracy"],
            "eval_time_s": base_results["eval_time_s"],
        },
        "adapter_results": {
            "mmlu_accuracy": adapter_results["mmlu"]["accuracy"],
            "gsm8k_accuracy": adapter_results["gsm8k"]["accuracy"],
            "humaneval_pass1": adapter_results["humaneval"]["accuracy"],
            "eval_time_s": adapter_results["eval_time_s"],
        },
        "deltas": kill["deltas"],
        "kill_criteria": {
            "k614_mmlu_nondegradation": kill["k614"],
            "k615_structured_improvement": kill["k615"],
            "k616_above_random": kill["k616"],
        },
        "overall": kill["overall"],
        "total_time_s": round(total_time, 1),
        "details": {
            "base_mmlu": base_results["mmlu"]["details"],
            "base_gsm8k": base_results["gsm8k"]["details"],
            "base_humaneval": base_results["humaneval"]["details"],
            "adapter_mmlu": adapter_results["mmlu"]["details"],
            "adapter_gsm8k": adapter_results["gsm8k"]["details"],
            "adapter_humaneval": adapter_results["humaneval"]["details"],
        },
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total experiment time: {total_time:.1f}s")
    log(f"\nFINAL VERDICT: {kill['overall']}")


if __name__ == "__main__":
    main()
