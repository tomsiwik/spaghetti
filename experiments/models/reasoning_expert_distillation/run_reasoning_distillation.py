#!/usr/bin/env python3
"""Train a reasoning LoRA expert + evaluate on MATH-500 in subprocess-isolated phases.

OOM-safe design: each GPU-intensive phase runs as a SEPARATE subprocess so VRAM
is fully reclaimed by the OS between phases. This is the ONLY reliable way to
avoid cumulative memory fragmentation on a 24GB RTX 4090.

Phases:
  1. train          -- QLoRA rank-16 on Qwen2.5-7B with rasbt/math_distill (500 steps)
  2. eval_base      -- MATH-500 eval on base Qwen2.5-7B (no adapter)
  3. eval_reasoning -- MATH-500 eval on base + reasoning adapter
  4. compare        -- Load result JSONs, compute accuracy delta, assess kill criteria

Usage (on RunPod):
    cd /workspace/llm
    python experiments/models/reasoning_expert_distillation/run_reasoning_distillation.py

    # Smoke test (<60s, 5 training steps, 10 eval examples):
    SMOKE_TEST=1 python experiments/models/reasoning_expert_distillation/run_reasoning_distillation.py

    # With runtime cap:
    MAX_RUNTIME=7200 python experiments/models/reasoning_expert_distillation/run_reasoning_distillation.py

    # Run a single phase:
    python experiments/models/reasoning_expert_distillation/run_reasoning_distillation.py --phase train

Expected runtime: ~2-3 hours on RTX 4090 (24GB)
Expected cost: ~$0.70-$1.00
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Environment: MUST be set before any torch import ─────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Monkey-patch set_submodule for PyTorch builds missing it (RunPod 2.4.1) ──
import torch
if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: "torch.nn.Module") -> None:
        atoms = target.split(".")
        mod = self
        for item in atoms[:-1]:
            mod = getattr(mod, item)
        setattr(mod, atoms[-1], module)
    torch.nn.Module.set_submodule = _set_submodule
    print(f"[patch] set_submodule monkey-patched onto torch {torch.__version__}")

# ── Configuration ─────────────────────────────────────────────────────────────

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
MAX_RUNTIME = int(os.environ.get("MAX_RUNTIME", "0"))  # seconds, 0 = no limit

# Paths (RunPod layout)
BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
HF_CACHE = os.environ.get("HF_CACHE", "/workspace/hf_cache")
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # /workspace/llm
OUTPUT_DIR = REPO_ROOT / "micro" / "models" / "reasoning_expert_distillation"
ADAPTER_DIR = OUTPUT_DIR / "reasoning_adapter"
RESULTS_DIR = OUTPUT_DIR

# Dataset config
DATASET_NAME = "rasbt/math_distill"
DATASET_CONFIG = "math_train"
DATASET_SPLIT = "train"
MAX_TRAIN_EXAMPLES = 10000
MAX_SEQ_LENGTH = 2048

# QLoRA config (matches pilot50 for composition compatibility)
LORA_RANK = 16
LORA_ALPHA = 16
LORA_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training config
TRAIN_STEPS = 500
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 1e-4
WARMUP_STEPS = 20
SEED = 42

# Eval config
MATH500_URL = (
    "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
    "main/ch03/01_main-chapter-code/math500_test.json"
)
MAX_EVAL_EXAMPLES = 500
MAX_NEW_TOKENS = 2048

# Smoke test overrides
if SMOKE_TEST:
    TRAIN_STEPS = 5
    MAX_TRAIN_EXAMPLES = 20
    MAX_EVAL_EXAMPLES = 10
    MAX_NEW_TOKENS = 256
    MAX_SEQ_LENGTH = 512
    print("[SMOKE_TEST] Reduced parameters for quick validation")


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    """Timestamped logging to stdout."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Timeout handler ───────────────────────────────────────────────────────────

def _timeout_handler(signum, frame):
    log(f"MAX_RUNTIME={MAX_RUNTIME}s exceeded. Terminating.")
    sys.exit(1)


if MAX_RUNTIME > 0 and hasattr(signal, "SIGALRM"):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(MAX_RUNTIME)
    log(f"Timeout armed: {MAX_RUNTIME}s")


# =============================================================================
# PHASE 1: TRAIN
# =============================================================================

def phase_train() -> None:
    """Train the reasoning QLoRA adapter.

    Loads Qwen2.5-7B with 4-bit NF4 quantization, applies LoRA rank-16 to all
    attention + FFN projection modules, trains on rasbt/math_distill reasoning
    traces for 500 steps.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    adapter_out = Path(ADAPTER_DIR)
    adapter_out.mkdir(parents=True, exist_ok=True)

    # Nanochat pattern: disable GC during training to avoid ~500ms pauses
    # from Python cycle detection scanning long-lived PyTorch tensors
    gc.disable()
    gc.collect()

    log("=" * 72)
    log("PHASE 1: TRAIN REASONING ADAPTER")
    log(f"  Base model: {BASE_MODEL}")
    log(f"  Output:     {adapter_out}")
    log(f"  Rank: {LORA_RANK}, Steps: {TRAIN_STEPS}, LR: {LR}")
    log(f"  Max seq length: {MAX_SEQ_LENGTH}")
    log(f"  Smoke test: {SMOKE_TEST}")
    log("=" * 72)

    # ── Load tokenizer ────────────────────────────────────────────────────
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Prepare dataset ───────────────────────────────────────────────────
    log(f"Loading dataset: {DATASET_NAME} (config={DATASET_CONFIG})")
    ds = load_dataset(
        DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, cache_dir=HF_CACHE
    )
    log(f"  Raw dataset size: {len(ds)}")

    # Filter: only examples with non-empty thinking traces
    ds = ds.filter(
        lambda x: (x.get("message_thinking") or "").strip() != "",
        desc="Filtering empty thinking traces",
    )
    log(f"  After filtering empty traces: {len(ds)}")

    # Filter: exclude very long traces (>10K chars) to avoid OOM during training
    ds = ds.filter(
        lambda x: len(x.get("message_thinking", "")) < 10000,
        desc="Filtering very long traces",
    )
    log(f"  After length filter: {len(ds)}")

    # Shuffle and cap
    ds = ds.shuffle(seed=SEED)
    if len(ds) > MAX_TRAIN_EXAMPLES:
        ds = ds.select(range(MAX_TRAIN_EXAMPLES))
    log(f"  Training examples: {len(ds)}")

    # Format as chat messages with <think> tags
    def format_reasoning(example):
        """Format a single example as chat message with <think> reasoning trace."""
        problem = example["problem"]
        thinking = example.get("message_thinking", "")
        content = example.get("message_content", "")

        if thinking:
            assistant_response = f"<think>\n{thinking}\n</think>\n\n{content}"
        else:
            assistant_response = content

        messages = [
            {"role": "system", "content": (
                "You are a helpful math assistant that shows your reasoning "
                "step by step inside <think>...</think> tags before giving "
                "your final answer."
            )},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": assistant_response},
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds = ds.map(
        format_reasoning,
        remove_columns=ds.column_names,
        desc="Formatting reasoning traces",
    )

    # ── Load base model with 4-bit quantization ──────────────────────────
    log("Loading base model with QLoRA (4-bit NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── Add LoRA adapters (all modules for composition compatibility) ─────
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_MODULES,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ── Explicit bf16 dtype detection (not autocast) ─────────────────────
    # SM 80+ (A100/H100/A5000/4090) supports bf16; older GPUs fall back to fp16
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    log(f"  Precision: {'bf16' if use_bf16 else 'fp16'}")

    # ── Train ─────────────────────────────────────────────────────────────
    ckpt_dir = adapter_out / "checkpoints"
    t0 = time.time()

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=str(ckpt_dir),
            max_steps=TRAIN_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            warmup_steps=WARMUP_STEPS,
            lr_scheduler_type="cosine",
            logging_steps=25,
            save_steps=100,
            save_total_limit=2,
            bf16=use_bf16,
            fp16=(not use_bf16) if torch.cuda.is_available() else False,
            optim="adamw_8bit",
            seed=SEED,
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            packing=True,
            report_to="none",
            warmup_ratio=0.0,  # use warmup_steps instead
            dataloader_pin_memory=True,
            dataloader_num_workers=2,
        ),
    )

    log("Starting training...")
    train_result = trainer.train()
    train_loss = train_result.training_loss
    elapsed = time.time() - t0

    log(f"Training complete: loss={train_loss:.4f}, time={elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # ── Save adapter ─────────────────────────────────────────────────────
    model.save_pretrained(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))

    # Cleanup intermediate checkpoints to save disk
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    # Save training metadata
    meta = {
        "experiment": "reasoning_expert_distillation",
        "type": "reasoning_capability_adapter",
        "base_model": BASE_MODEL,
        "dataset": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "n_train_examples": len(ds),
        "max_seq_length": MAX_SEQ_LENGTH,
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "target_modules": LORA_MODULES,
        "steps": TRAIN_STEPS,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "lr": LR,
        "warmup_steps": WARMUP_STEPS,
        "train_loss": float(train_loss),
        "train_time_s": float(elapsed),
        "trainable_params": trainable,
        "total_params": total,
        "seed": SEED,
        "smoke_test": SMOKE_TEST,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(adapter_out / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log(f"Adapter saved to {adapter_out}")
    est_cost = elapsed / 3600 * 0.34
    log(f"Estimated cost: ${est_cost:.2f}")


# =============================================================================
# MATH-500 ANSWER PARSING
# Robust extraction and grading adapted from rasbt/reasoning-from-scratch ch03.
# Handles nested braces, LaTeX normalization, fraction parsing, sympy fallback.
# =============================================================================

# Regex patterns for answer extraction
RE_BOXED = re.compile(r"\\boxed\s*\{", re.DOTALL)
RE_NUMBER = re.compile(r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

# LaTeX normalization rules
LATEX_FIXES = [
    (r"\\left\s*", ""),
    (r"\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot", "*"),
    (r"\u00B7|\u00D7", "*"),
    (r"\\\^\\circ", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"\u00B0", ""),  # degree symbol
]

RE_SPECIAL = re.compile(r"<\|[^>]+?\|>")

SUPERSCRIPT_MAP = {
    "\u2070": "0", "\u00B9": "1", "\u00B2": "2", "\u00B3": "3", "\u2074": "4",
    "\u2075": "5", "\u2076": "6", "\u2077": "7", "\u2078": "8", "\u2079": "9",
    "\u207A": "+", "\u207B": "-", "\u207D": "(", "\u207E": ")",
}


def get_last_boxed(text: str) -> str | None:
    """Extract content from the last \\boxed{...} in text, handling nested braces."""
    matches = list(RE_BOXED.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    if depth != 0:
        return None
    return text[start : pos - 1]


def extract_final_answer(text: str) -> str:
    """Extract the final answer: prefer \\boxed{}, fall back to last number."""
    if not text:
        return ""
    boxed = get_last_boxed(text.strip())
    if boxed:
        return boxed.strip().strip("$ ")
    numbers = RE_NUMBER.findall(text)
    return numbers[-1] if numbers else text.strip()


def normalize_text(text: str) -> str:
    """Normalize a math answer for comparison.

    Applies LaTeX canonicalization, removes formatting artifacts, converts
    fractions and exponents to evaluable form.
    """
    if not text:
        return ""
    text = RE_SPECIAL.sub("", text).strip()

    # Strip leading multiple-choice labels (e.g. "c. 3" -> "3")
    m = re.match(r"^[A-Za-z]\s*[.:]\s*(.+)$", text)
    if m:
        text = m.group(1)

    # Remove degree markers
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)
    text = re.sub(r"\^\s*\\circ", "", text)
    text = text.replace("\u00B0", "")

    # Unwrap \text{...}
    m = re.match(r"^\\text\{(?P<x>.+?)\}$", text)
    if m:
        text = m.group("x")

    # Strip inline/display math wrappers
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)

    # Apply LaTeX normalization rules
    for pat, rep in LATEX_FIXES:
        text = re.sub(pat, rep, text)

    # Convert unicode superscripts to exponent form
    def _convert_superscripts(s, base=None):
        converted = "".join(SUPERSCRIPT_MAP.get(ch, ch) for ch in s)
        return f"{base}**{converted}" if base is not None else converted

    text = re.sub(
        r"([0-9A-Za-z\)\]\}])([\u2070\u00B9\u00B2\u00B3\u2074\u2075\u2076\u2077\u2078\u2079\u207A\u207B]+)",
        lambda m: _convert_superscripts(m.group(2), base=m.group(1)),
        text,
    )
    text = _convert_superscripts(text)

    # Numbers and roots
    text = text.replace("\\%", "%").replace("$", "").replace("%", "")
    text = re.sub(
        r"\\sqrt\s*\{([^}]*)\}",
        lambda m: f"sqrt({m.group(1)})",
        text,
    )
    text = re.sub(
        r"\\sqrt\s+([^\\\s{}]+)",
        lambda m: f"sqrt({m.group(1)})",
        text,
    )

    # Fractions: \frac{a}{b} -> (a)/(b)
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda m: f"({m.group(1)})/({m.group(2)})",
        text,
    )
    text = re.sub(
        r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)",
        lambda m: f"({m.group(1)})/({m.group(2)})",
        text,
    )

    # Exponents and mixed numbers
    text = text.replace("^", "**")
    text = re.sub(
        r"(?<=\d)\s+(\d+/\d+)",
        lambda m: "+" + m.group(1),
        text,
    )

    # Remove thousands separators: 1,234 -> 1234
    text = re.sub(r"(?<=\d),(?=\d\d\d(\D|$))", "", text)

    return text.replace("{", "").replace("}", "").strip().lower()


def _sympy_parse(expr: str):
    """Parse a math expression with sympy. Returns None on failure."""
    if expr is None or len(expr) > 2000:
        return None
    try:
        from sympy.parsing import sympy_parser as spp
        return spp.parse_expr(
            expr,
            transformations=(
                *spp.standard_transformations,
                spp.implicit_multiplication_application,
            ),
            evaluate=True,
        )
    except Exception:
        return None


def _equality_check(expr_gt: str, expr_pred: str) -> bool:
    """Check mathematical equality: string match, then numeric, then sympy."""
    # Exact string match after normalization
    if expr_gt == expr_pred:
        return True

    # Numeric comparison with tolerance
    try:
        val_gt = float(expr_gt) if "/" not in expr_gt else eval(expr_gt)  # noqa: S307
        val_pred = float(expr_pred) if "/" not in expr_pred else eval(expr_pred)  # noqa: S307
        if abs(val_gt - val_pred) < 1e-6:
            return True
    except Exception:
        pass

    # Sympy symbolic comparison (optional, may not be installed)
    try:
        from sympy import simplify
        gt_sym = _sympy_parse(expr_gt)
        pred_sym = _sympy_parse(expr_pred)
        if gt_sym is not None and pred_sym is not None:
            if simplify(gt_sym - pred_sym) == 0:
                return True
    except Exception:
        pass

    return False


def _split_tuple(text: str) -> list[str]:
    """Split tuple/list answers like '(a, b)' into parts."""
    if (
        text
        and len(text) >= 2
        and text[0] in "(["
        and text[-1] in ")]"
        and "," in text[1:-1]
    ):
        items = [p.strip() for p in text[1:-1].split(",")]
        if all(items):
            return items
    return [text] if text else []


def grade_answer(predicted: str, ground_truth: str) -> bool:
    """Grade a predicted answer against ground truth.

    Pipeline: extract -> normalize -> split tuples -> check equality per part.
    Uses string match, numeric comparison (1e-6 tolerance), and sympy fallback.
    """
    if predicted is None or ground_truth is None:
        return False

    gt_parts = _split_tuple(normalize_text(ground_truth))
    pred_parts = _split_tuple(normalize_text(predicted))

    if not gt_parts or not pred_parts or len(gt_parts) != len(pred_parts):
        return False

    return all(
        _equality_check(gt, pred) for gt, pred in zip(gt_parts, pred_parts)
    )


# =============================================================================
# MATH-500 DATASET LOADING
# =============================================================================

def load_math500(max_examples: int = MAX_EVAL_EXAMPLES) -> list[dict]:
    """Load the MATH-500 test set. Tries local copy first, then downloads."""
    import requests

    local_path = OUTPUT_DIR / "math500_test.json"

    # Try local copy
    if local_path.exists():
        with open(local_path) as f:
            data = json.load(f)
        return data[:max_examples]

    # Try reference copy in repo
    ref_path = (
        REPO_ROOT / "references" / "reasoning-from-scratch"
        / "ch03" / "01_main-chapter-code" / "math500_test.json"
    )
    if ref_path.exists():
        with open(ref_path) as f:
            data = json.load(f)
    else:
        # Download
        log(f"Downloading MATH-500 from {MATH500_URL}")
        r = requests.get(MATH500_URL, timeout=30)
        r.raise_for_status()
        data = r.json()

    # Cache locally
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w") as f:
        json.dump(data, f, indent=2)

    return data[:max_examples]


# =============================================================================
# PHASE 2 & 3: EVAL (base and reasoning)
# =============================================================================

def _build_eval_prompts(
    problems: list[dict], tokenizer
) -> list[str]:
    """Build Qwen2.5 chat-formatted prompts for MATH-500 evaluation."""
    prompts = []
    for ex in problems:
        messages = [
            {"role": "system", "content": (
                "You are a helpful math assistant. Solve the problem step by step "
                "and write your final answer as \\boxed{ANSWER}."
            )},
            {"role": "user", "content": ex["problem"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    return prompts


def _run_eval(
    problems: list[dict],
    condition_name: str,
    adapter_path: str | None = None,
) -> dict:
    """Evaluate a model condition on MATH-500 problems.

    Loads the base model (optionally with adapter), generates answers one at a
    time with explicit memory cleanup after each example, and grades them.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    log(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"Loading model for condition: {condition_name}")
    if adapter_path:
        # Load quantized base + adapter for reasoning eval (matches training config)
        from peft import PeftModel

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=HF_CACHE,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        log(f"  Loaded adapter from {adapter_path}")
    else:
        # Load base model in fp16 for eval (no quantization needed -- saves memory
        # vs 4bit because no double quant overhead, and fp16 is faster for generation)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=HF_CACHE,
            trust_remote_code=True,
        )

    model.eval()

    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        log(f"  Model loaded. Peak GPU memory: {mem_gb:.1f} GB")

    prompts = _build_eval_prompts(problems, tokenizer)

    log(f"  Evaluating {len(problems)} problems...")
    t0 = time.time()
    correct = 0
    results_per_example = []

    for i, (prompt, example) in enumerate(zip(prompts, problems)):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated tokens (skip the prompt)
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True)

        # Explicit cleanup -- prevents KV cache memory accumulation
        del outputs, inputs, generated
        torch.cuda.empty_cache()

        predicted = extract_final_answer(text)
        is_correct = grade_answer(predicted, example["answer"])
        correct += int(is_correct)

        results_per_example.append({
            "idx": i,
            "problem": (
                example["problem"][:100] + "..."
                if len(example["problem"]) > 100
                else example["problem"]
            ),
            "ground_truth": example["answer"],
            "predicted": predicted,
            "correct": is_correct,
        })

        # Progress logging every 25 examples
        if (i + 1) % 25 == 0 or (i + 1) == len(problems):
            acc_so_far = 100 * correct / (i + 1)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(problems) - i - 1)
            log(f"    [{i+1}/{len(problems)}] accuracy: {acc_so_far:.1f}% "
                f"(elapsed: {elapsed:.0f}s, eta: {eta:.0f}s)")

        # Periodic deep cleanup every 50 examples
        if (i + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    accuracy = correct / len(problems) if problems else 0.0
    log(f"  {condition_name}: {correct}/{len(problems)} = {100 * accuracy:.1f}% ({elapsed:.0f}s)")

    result = {
        "condition": condition_name,
        "correct": correct,
        "total": len(problems),
        "accuracy": accuracy,
        "accuracy_pct": round(100 * accuracy, 2),
        "elapsed_s": round(elapsed, 1),
        "per_example": results_per_example,
        "adapter_path": adapter_path,
        "smoke_test": SMOKE_TEST,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return result


def phase_eval_base() -> None:
    """Phase 2: Evaluate base Qwen2.5-7B on MATH-500 (no adapter)."""
    log("=" * 72)
    log("PHASE 2: EVALUATE BASE MODEL ON MATH-500")
    log("=" * 72)

    problems = load_math500(MAX_EVAL_EXAMPLES)
    log(f"  Loaded {len(problems)} MATH-500 problems")

    result = _run_eval(problems, "base", adapter_path=None)

    out_path = RESULTS_DIR / "math500_base_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"  Results saved to {out_path}")


def phase_eval_reasoning() -> None:
    """Phase 3: Evaluate base + reasoning adapter on MATH-500."""
    log("=" * 72)
    log("PHASE 3: EVALUATE REASONING ADAPTER ON MATH-500")
    log("=" * 72)

    adapter_path = str(ADAPTER_DIR)
    if not (ADAPTER_DIR / "adapter_config.json").exists():
        log(f"ERROR: Adapter not found at {adapter_path}. Run phase 'train' first.")
        sys.exit(1)

    problems = load_math500(MAX_EVAL_EXAMPLES)
    log(f"  Loaded {len(problems)} MATH-500 problems")

    result = _run_eval(problems, "reasoning_only", adapter_path=adapter_path)

    out_path = RESULTS_DIR / "math500_reasoning_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"  Results saved to {out_path}")


# =============================================================================
# PHASE 4: COMPARE
# =============================================================================

def phase_compare() -> None:
    """Phase 4: Load results from both evals, compute delta, assess kill criteria."""
    log("=" * 72)
    log("PHASE 4: COMPARE RESULTS & KILL CRITERIA")
    log("=" * 72)

    base_path = RESULTS_DIR / "math500_base_results.json"
    reasoning_path = RESULTS_DIR / "math500_reasoning_results.json"

    if not base_path.exists():
        log(f"ERROR: Base results not found at {base_path}. Run phase 'eval_base' first.")
        sys.exit(1)
    if not reasoning_path.exists():
        log(f"ERROR: Reasoning results not found at {reasoning_path}. Run phase 'eval_reasoning' first.")
        sys.exit(1)

    with open(base_path) as f:
        base_result = json.load(f)
    with open(reasoning_path) as f:
        reasoning_result = json.load(f)

    base_acc = base_result["accuracy_pct"]
    reasoning_acc = reasoning_result["accuracy_pct"]
    improvement_pp = reasoning_acc - base_acc

    log(f"\n  Accuracy Summary:")
    log(f"    Base (no adapter):         {base_acc:.1f}%  ({base_result['correct']}/{base_result['total']})")
    log(f"    Reasoning adapter:         {reasoning_acc:.1f}%  ({reasoning_result['correct']}/{reasoning_result['total']})")
    log(f"    Improvement:               {improvement_pp:+.1f} pp")

    # Kill criteria K1: reasoning LoRA must improve >10pp over base
    k1_threshold = 10
    k1_pass = improvement_pp > k1_threshold
    log(f"\n  K1: Reasoning improvement > {k1_threshold}pp over base")
    log(f"      Improvement: {improvement_pp:+.1f}pp")
    log(f"      Verdict: {'PASS' if k1_pass else 'KILL'}")

    if k1_pass:
        verdict = "PASS (reasoning distillation effective)"
    else:
        verdict = f"KILLED (K1: {improvement_pp:+.1f}pp < {k1_threshold}pp threshold)"

    log(f"\n  Overall: {verdict}")

    # Save combined results
    combined = {
        "experiment": "reasoning_expert_distillation",
        "base_model": BASE_MODEL,
        "adapter_path": str(ADAPTER_DIR),
        "smoke_test": SMOKE_TEST,
        "conditions": {
            "base": {
                "accuracy_pct": base_acc,
                "correct": base_result["correct"],
                "total": base_result["total"],
                "elapsed_s": base_result["elapsed_s"],
            },
            "reasoning_only": {
                "accuracy_pct": reasoning_acc,
                "correct": reasoning_result["correct"],
                "total": reasoning_result["total"],
                "elapsed_s": reasoning_result["elapsed_s"],
            },
        },
        "improvement_pp": round(improvement_pp, 2),
        "kill_criteria": {
            "K1_reasoning_gt_10pp": {
                "threshold": k1_threshold,
                "actual": round(improvement_pp, 2),
                "pass": k1_pass,
            },
        },
        "verdict": verdict,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Also include training metadata if available
    train_meta_path = ADAPTER_DIR / "train_meta.json"
    if train_meta_path.exists():
        with open(train_meta_path) as f:
            combined["train_meta"] = json.load(f)

    out_path = RESULTS_DIR / "math500_results.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    log(f"\n  Combined results saved to {out_path}")

    # Estimate total cost
    total_time_s = base_result.get("elapsed_s", 0) + reasoning_result.get("elapsed_s", 0)
    train_time_s = combined.get("train_meta", {}).get("train_time_s", 0)
    total_cost = (total_time_s + train_time_s) / 3600 * 0.34
    log(f"  Total GPU time: {(total_time_s + train_time_s) / 60:.1f} min")
    log(f"  Estimated cost: ${total_cost:.2f}")


# =============================================================================
# SUBPROCESS ORCHESTRATOR
# =============================================================================

PHASES = {
    "train": phase_train,
    "eval_base": phase_eval_base,
    "eval_reasoning": phase_eval_reasoning,
    "compare": phase_compare,
}


def run_phase_in_subprocess(phase_name: str) -> None:
    """Fork a new Python process for a single phase.

    This is the ONLY reliable way to fully reclaim GPU VRAM between phases on a
    24GB card. Calling torch.cuda.empty_cache() + gc.collect() is NOT sufficient
    because PyTorch's CUDA allocator retains memory pools, and fragmentation
    accumulates across model loads.

    By spawning a fresh process, the OS reclaims ALL GPU memory when the process
    exits, giving the next phase a clean slate.
    """
    log(f"\n{'#' * 72}")
    log(f"# Launching phase '{phase_name}' as subprocess")
    log(f"{'#' * 72}\n")

    # Build the subprocess command: re-invoke this script with --phase
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase", phase_name,
    ]

    # Forward environment variables
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if SMOKE_TEST:
        env["SMOKE_TEST"] = "1"
    if MAX_RUNTIME > 0:
        env["MAX_RUNTIME"] = str(MAX_RUNTIME)

    t0 = time.time()
    result = subprocess.run(
        cmd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        log(f"Phase '{phase_name}' FAILED (exit code {result.returncode}) after {elapsed:.0f}s")
        sys.exit(result.returncode)

    log(f"Phase '{phase_name}' completed in {elapsed:.0f}s ({elapsed / 60:.1f} min)")


def main():
    """Entry point: orchestrate all phases as subprocesses, or run a single phase."""
    parser = argparse.ArgumentParser(
        description="Reasoning expert distillation: train + eval in subprocess-isolated phases"
    )
    parser.add_argument(
        "--phase",
        choices=list(PHASES.keys()),
        default=None,
        help="Run a single phase directly (used by subprocess orchestrator). "
             "If not specified, runs all phases sequentially as subprocesses.",
    )
    args = parser.parse_args()

    if args.phase:
        # Direct execution of a single phase (called from subprocess)
        log(f"Executing phase: {args.phase}")
        PHASES[args.phase]()
        return

    # Orchestrator mode: run all phases as separate subprocesses
    log("=" * 72)
    log("REASONING EXPERT DISTILLATION -- FULL PIPELINE")
    log(f"  Base model:    {BASE_MODEL}")
    log(f"  Adapter dir:   {ADAPTER_DIR}")
    log(f"  Results dir:   {RESULTS_DIR}")
    log(f"  Smoke test:    {SMOKE_TEST}")
    if MAX_RUNTIME > 0:
        log(f"  Max runtime:   {MAX_RUNTIME}s")
    else:
        log(f"  Max runtime:   unlimited")
    if torch.cuda.is_available():
        log(f"  GPU:           {torch.cuda.get_device_name(0)}")
    else:
        log(f"  GPU:           CPU (no CUDA)")
    log("=" * 72)

    pipeline_t0 = time.time()

    # Phase 1: Train
    run_phase_in_subprocess("train")

    # Phase 2: Eval base
    run_phase_in_subprocess("eval_base")

    # Phase 3: Eval reasoning
    run_phase_in_subprocess("eval_reasoning")

    # Phase 4: Compare (CPU-only, fast -- but still subprocess for consistency)
    run_phase_in_subprocess("compare")

    pipeline_elapsed = time.time() - pipeline_t0
    log(f"\n{'=' * 72}")
    log(f"PIPELINE COMPLETE in {pipeline_elapsed:.0f}s ({pipeline_elapsed / 60:.1f} min)")
    log(f"Estimated cost: ${pipeline_elapsed / 3600 * 0.34:.2f}")
    log(f"Results: {RESULTS_DIR / 'math500_results.json'}")
    log(f"{'=' * 72}")


if __name__ == "__main__":
    main()
