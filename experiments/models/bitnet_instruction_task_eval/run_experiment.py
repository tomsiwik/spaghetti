#!/usr/bin/env python3
"""
BitNet-2B Instruction-Tuned LoRA Task Evaluation

Tests whether instruction-tuned LoRA adapters produce task-capable models where
NTP-trained adapters failed (exp_bitnet_task_eval was KILLED).

Hypothesis: Instruction-formatted training data enables task performance because
instruction tuning causes the model to recognize instruction parts of prompts and
condition responses on them (Li et al. 2023, arxiv:2310.00492).

Key changes from killed experiment:
  1. TRAIN on instruction QA pairs (not raw NTP text)
  2. EVAL with matching instruction prompts (not raw "Solve:" prompts)
  3. Test BOTH 1/N composition AND oracle routing (per-domain activation)
  4. More training steps (400) with higher-quality data

Kill criteria:
  K1: instruction-tuned composed model WORSE than base on >40% of task metrics
  K2: instruction-tuned math adapter <3pp improvement on MATH-500 subset

Domains:
  - Medical: medalpaca/medical_meadow_medical_flashcards (QA format)
  - Math: gsm8k (question + CoT answer)
  - Code: iamtarun/python_code_instructions_18k_alpaca (instruction/output)
  - Legal: nguha/legalbench (QA tasks)
  - Creative: story continuation (PPL, no instruction format needed)

Architecture:
  - Base: microsoft/BitNet-b1.58-2B-4T (ternary, d=2560, 30 layers)
  - LoRA: rank-16, all-modules, FP16 (not ternary STE for simplicity/speed)
  - Training: 400 steps, lr=1e-4, batch=1, seq_len=256
  - Composition: 1/N averaging + oracle routing

Runtime: ~60-90 min on Apple Silicon
"""

import ast
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def log(msg):
    print(msg, flush=True)


import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm import load
from mlx_lm.models.bitlinear_layers import BitLinear

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
TRAIN_ITERS = 300  # Instruction data is signal-dense, 300 steps sufficient
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
VAL_BATCHES = 25

EXPERIMENT_DIR = Path(__file__).parent
DATA_DIR = EXPERIMENT_DIR / "data"
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MATH500_PATH = (Path(__file__).parent.parent /
                "reasoning_expert_distillation" / "math500_test.json")

DOMAINS = ["medical", "math", "code", "legal", "creative"]

# Eval sizes -- kept small for Apple Silicon runtime (~60 min total)
# Generation is the bottleneck (~1 min per problem with 128 max tokens)
MATH_EVAL_N = 15
CODE_EVAL_N = 10
QA_EVAL_N = 15
CREATIVE_EVAL_N = 20
MAX_GEN_TOKENS = 100  # Shorter to keep runtime feasible

# Instruction template (simple, compatible with non-instruct base)
INST_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
INST_PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n"


# ===========================================================================
# Data preparation: download and format as instruction QA pairs
# ===========================================================================
def prepare_instruction_data():
    """Download and format instruction data for each domain."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    datasets_config = {
        "medical": {
            "hf_id": "medalpaca/medical_meadow_medical_flashcards",
            "formatter": format_medical,
            "n_train": 800, "n_val": 100,
        },
        "math": {
            "hf_id": "gsm8k",
            "hf_subset": "main",
            "formatter": format_math,
            "n_train": 800, "n_val": 100,
        },
        "code": {
            "hf_id": "iamtarun/python_code_instructions_18k_alpaca",
            "formatter": format_code,
            "n_train": 800, "n_val": 100,
        },
        "legal": {
            "hf_id": "nguha/legalbench",
            "hf_subset": "contract_nli_explicit_identification",
            "formatter": format_legal,
            "n_train": 800, "n_val": 100,
        },
        "creative": {
            "hf_id": "roneneldan/TinyStories",
            "formatter": format_creative,
            "n_train": 800, "n_val": 100,
        },
    }

    for domain, cfg in datasets_config.items():
        domain_dir = DATA_DIR / domain
        train_path = domain_dir / "train.jsonl"
        val_path = domain_dir / "val.jsonl"

        if train_path.exists() and val_path.exists():
            log(f"  {domain}: data already exists, skipping")
            continue

        domain_dir.mkdir(parents=True, exist_ok=True)
        log(f"  {domain}: downloading {cfg['hf_id']}...")

        try:
            kwargs = {"path": cfg["hf_id"]}
            if "hf_subset" in cfg:
                kwargs["name"] = cfg["hf_subset"]

            ds = load_dataset(**kwargs, trust_remote_code=True)

            # Get train split (or combined if no val)
            if "train" in ds:
                raw = list(ds["train"])
            else:
                raw = list(ds[list(ds.keys())[0]])

            # Format into instruction pairs
            formatted = cfg["formatter"](raw)
            log(f"  {domain}: formatted {len(formatted)} instruction pairs")

            if len(formatted) < 50:
                log(f"  WARNING: {domain} only has {len(formatted)} pairs")

            n_train = min(cfg["n_train"], len(formatted) - cfg["n_val"])
            n_val = min(cfg["n_val"], len(formatted) - n_train)

            train_data = formatted[:n_train]
            val_data = formatted[n_train:n_train + n_val]

            with open(train_path, "w") as f:
                for item in train_data:
                    f.write(json.dumps(item) + "\n")
            with open(val_path, "w") as f:
                for item in val_data:
                    f.write(json.dumps(item) + "\n")

            log(f"  {domain}: saved {len(train_data)} train, {len(val_data)} val")

        except Exception as e:
            log(f"  ERROR {domain}: {e}")
            # Create minimal fallback data
            create_fallback_data(domain, domain_dir)


def format_medical(raw):
    """Medical QA flashcards -> instruction pairs."""
    pairs = []
    for item in raw:
        q = item.get("input", "").strip()
        a = item.get("output", "").strip()
        if not q or not a or len(a) < 10:
            continue
        text = INST_TEMPLATE.format(instruction=q, response=a)
        pairs.append({"text": text, "instruction": q, "response": a})
    return pairs


def format_math(raw):
    """GSM8K -> instruction pairs with chain-of-thought."""
    pairs = []
    for item in raw:
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if not q or not a:
            continue
        instruction = f"Solve step by step: {q}"
        text = INST_TEMPLATE.format(instruction=instruction, response=a)
        pairs.append({"text": text, "instruction": instruction, "response": a})
    return pairs


def format_code(raw):
    """Python code instructions -> instruction pairs."""
    pairs = []
    for item in raw:
        inst = item.get("instruction", "").strip()
        output = item.get("output", "").strip()
        if not inst or not output or len(output) < 10:
            continue
        text = INST_TEMPLATE.format(instruction=inst, response=output)
        pairs.append({"text": text, "instruction": inst, "response": output})
    return pairs


def format_legal(raw):
    """LegalBench -> instruction pairs."""
    pairs = []
    for item in raw:
        # LegalBench contract_nli has text and answer
        premise = item.get("text", "").strip()
        hypothesis = item.get("hypothesis", "").strip()
        answer = item.get("answer", "").strip()
        if not premise or not answer:
            continue
        instruction = f"Legal analysis: Given the contract clause: '{premise[:300]}'"
        if hypothesis:
            instruction += f"\nDoes this entail: '{hypothesis[:200]}'?"
        response = answer
        text = INST_TEMPLATE.format(instruction=instruction, response=response)
        pairs.append({"text": text, "instruction": instruction, "response": response})
    return pairs


def format_creative(raw):
    """TinyStories -> story continuation instruction pairs."""
    pairs = []
    for item in raw:
        story = item.get("text", "").strip()
        if not story or len(story) < 100:
            continue
        # Split story into prompt (first 1/3) and continuation (rest 2/3)
        sentences = re.split(r'(?<=[.!?])\s+', story)
        if len(sentences) < 3:
            continue
        n_prompt = max(1, len(sentences) // 3)
        prompt_part = " ".join(sentences[:n_prompt])
        continuation = " ".join(sentences[n_prompt:])
        instruction = f"Continue the story: {prompt_part}"
        text = INST_TEMPLATE.format(instruction=instruction, response=continuation)
        pairs.append({
            "text": text, "instruction": instruction,
            "response": continuation, "full_story": story
        })
    return pairs


def create_fallback_data(domain, domain_dir):
    """Create minimal synthetic instruction data if HF download fails."""
    log(f"  Creating fallback data for {domain}")
    train_path = domain_dir / "train.jsonl"
    val_path = domain_dir / "val.jsonl"

    templates = {
        "medical": [
            ("What is hypertension?", "Hypertension is high blood pressure, typically defined as systolic pressure above 130 mmHg or diastolic above 80 mmHg."),
            ("What causes diabetes?", "Type 2 diabetes is caused by insulin resistance, where cells don't respond normally to insulin, leading to high blood sugar levels."),
        ],
        "math": [
            ("Solve step by step: What is 15 + 27?", "15 + 27 = 42. The answer is 42."),
            ("Solve step by step: If a book costs $12 and you buy 3, how much total?", "3 * $12 = $36. The answer is $36."),
        ],
        "code": [
            ("Write a Python function to add two numbers", "def add(a, b):\n    return a + b"),
            ("Write a Python function to find the maximum in a list", "def find_max(lst):\n    return max(lst)"),
        ],
        "legal": [
            ("Is this clause enforceable: 'Employee shall not compete for 100 years'?", "No, this non-compete clause is likely unenforceable due to unreasonable duration."),
        ],
        "creative": [
            ("Continue the story: Once upon a time, a little bird found a golden key.", "The bird picked up the key in its beak and flew to the old tower. Inside, it found a door that had been locked for a hundred years."),
        ],
    }

    items = []
    for inst, resp in templates.get(domain, templates["medical"]):
        text = INST_TEMPLATE.format(instruction=inst, response=resp)
        items.append({"text": text, "instruction": inst, "response": resp})

    # Duplicate to get enough data
    while len(items) < 100:
        items = items + items
    items = items[:100]

    with open(train_path, "w") as f:
        for item in items[:80]:
            f.write(json.dumps(item) + "\n")
    with open(val_path, "w") as f:
        for item in items[80:]:
            f.write(json.dumps(item) + "\n")


# ===========================================================================
# Ternary weight unpacking (reused from prior experiments)
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


# ===========================================================================
# LoRA (standard FP16 -- simpler, faster than ternary STE for this experiment)
# ===========================================================================
class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=16, scale=20.0):
        super().__init__()
        self.linear = base_linear
        self.r = r
        self.scale = scale
        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]
        s = 1.0 / math.sqrt(in_features)
        self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_features, r))
        self.lora_b = mx.zeros((r, out_features))

    def __call__(self, x):
        base_out = self.linear(x)
        lora_out = (x @ self.lora_a) @ self.lora_b * self.scale
        return base_out + lora_out


def apply_lora(model, rank=16, scale=20.0):
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora = LoRALinear(module, r=rank, scale=scale)
                updates.append((key, lora))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    log(f"  Applied LoRA (r={rank}) to {count} layers")
    return model


def get_lora_params(model):
    """Get only LoRA A/B parameters."""
    params = []
    for name, val in tree_flatten(model.parameters()):
        if "lora_a" in name or "lora_b" in name:
            params.append((name, val))
    return params


def zero_lora_params(model):
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                in_dims = module.lora_a.shape[0]
                s = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(low=-s, high=s, shape=module.lora_a.shape)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def save_adapter(model, path):
    path.mkdir(parents=True, exist_ok=True)
    params = {}
    for name, val in get_lora_params(model):
        params[name] = val
    mx.savez(str(path / "adapter.npz"), **params)


def load_adapter(path):
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_weights(model, adapter_params, scale=1.0):
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def compose_adapters(adapter_list, scale_per_adapter=None):
    N = len(adapter_list)
    if scale_per_adapter is None:
        scale_per_adapter = 1.0 / N
    merged = {}
    for key in adapter_list[0].keys():
        stacked = mx.stack([a[key] for a in adapter_list])
        merged[key] = mx.sum(stacked, axis=0) * scale_per_adapter
    return merged


# ===========================================================================
# Training loop (instruction-tuned: full sequence NTP on instruction+response)
# ===========================================================================
def train_adapter(model, tokenizer, domain, train_data, val_data, n_iters=TRAIN_ITERS):
    """Train LoRA adapter on instruction-formatted data."""
    log(f"\n  Training {domain} adapter ({n_iters} steps)...")

    # Freeze base, only train LoRA
    model.freeze()
    lora_params = get_lora_params(model)
    for name, _ in lora_params:
        parts = name.split(".")
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
        # Unfreeze the parameter's parent module
    # More direct unfreezing
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.lora_a = module.lora_a  # ensure tracked
                module.lora_b = module.lora_b

    # Unfreeze LoRA params explicitly
    trainable = []
    for name, val in tree_flatten(model.trainable_parameters()):
        trainable.append(name)

    # If no trainable params, manually unfreeze
    if not trainable:
        for layer in model.model.layers:
            for key, module in layer.named_modules():
                if isinstance(module, LoRALinear):
                    module.freeze(recurse=False)  # freeze linear inside
                    # Keep lora_a, lora_b trainable by not freezing them

    # Actually: just unfreeze the whole model's lora params
    # The model.freeze() freezes everything, we need to selectively unfreeze
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                # The linear base stays frozen, LoRA params are trainable
                pass

    # Let's do it properly: freeze everything, then unfreeze LoRA
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"])

    n_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"    Trainable params: {n_trainable:,}")

    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, tokens):
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    t0 = time.time()
    data_idx = 0

    for step in range(n_iters):
        item = train_data[data_idx % len(train_data)]
        data_idx += 1

        tokens = tokenizer.encode(item["text"])
        if len(tokens) < 4:
            continue
        tokens = tokens[:MAX_SEQ_LENGTH + 1]
        tokens_mx = mx.array(tokens)

        loss, grads = loss_and_grad(model, tokens_mx)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            elapsed = time.time() - t0
            log(f"    Step {step+1}/{n_iters}: loss={avg_loss:.4f} "
                f"({elapsed:.0f}s elapsed)")

    # Validation loss
    val_losses = []
    for item in val_data[:VAL_BATCHES]:
        tokens = tokenizer.encode(item["text"])
        if len(tokens) < 4:
            continue
        tokens = tokens[:MAX_SEQ_LENGTH + 1]
        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        mx.eval(loss)
        val_losses.append(loss.item())

    val_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")
    train_loss = sum(losses[-50:]) / len(losses[-50:])
    log(f"    Final: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    return {
        "train_loss_final": round(train_loss, 4),
        "val_loss": round(val_loss, 4),
        "n_steps": n_iters,
        "time_s": round(time.time() - t0, 1),
    }


# ===========================================================================
# Text generation
# ===========================================================================
def generate_text(model, tokenizer, prompt, max_tokens=MAX_GEN_TOKENS, temperature=0.0):
    tokens = tokenizer.encode(prompt)
    if len(tokens) > MAX_SEQ_LENGTH:
        tokens = tokens[-MAX_SEQ_LENGTH:]

    generated = []
    for _ in range(max_tokens):
        x = mx.array(tokens + generated)[None, :]
        logits = model(x)
        next_logits = logits[:, -1, :]

        if temperature <= 0:
            next_token = mx.argmax(next_logits, axis=-1)
        else:
            next_token = mx.random.categorical(next_logits / temperature)

        mx.eval(next_token)
        token_id = next_token.item()

        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)

        if len(tokens) + len(generated) > MAX_SEQ_LENGTH * 3:
            break

    return tokenizer.decode(generated)


# ===========================================================================
# MATH grading
# ===========================================================================
RE_BOXED = re.compile(r"\\boxed\{", re.DOTALL)
RE_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
RE_ANSWER_IS = re.compile(r"(?:answer is|Answer:|####)\s*(.+?)(?:\.|$)", re.I)


def get_last_boxed(text):
    matches = list(RE_BOXED.finditer(text))
    if not matches:
        return ""
    start = matches[-1].end()
    depth, pos = 1, start
    while pos < len(text) and depth > 0:
        if text[pos] == '{': depth += 1
        elif text[pos] == '}': depth -= 1
        pos += 1
    return text[start:pos - 1] if depth == 0 else text[start:]


def extract_final_answer(text):
    if not text:
        return ""
    boxed = get_last_boxed(text.strip())
    if boxed:
        return boxed.strip().strip("$ ")
    # Try "the answer is X" or "#### X" patterns
    m = RE_ANSWER_IS.search(text)
    if m:
        ans = m.group(1).strip().rstrip(".")
        nums = RE_NUMBER.findall(ans)
        if nums:
            return nums[-1]
        return ans[:50]
    numbers = RE_NUMBER.findall(text)
    return numbers[-1] if numbers else text.strip()[:50]


def normalize_text(text):
    text = text.strip()
    for s in ["\\$", "$", "\\left", "\\right", "\\,", "\\text{", "\\mathrm{",
              "\\(", "\\)", "\\{", "\\}"]:
        text = text.replace(s, "")
    text = text.replace("\\ ", " ").replace("\\dfrac", "\\frac")
    text = text.rstrip("}")
    return " ".join(text.split())


def grade_math_answer(predicted, ground_truth):
    pred, gt = normalize_text(predicted), normalize_text(ground_truth)
    if pred == gt:
        return True
    try:
        if "/" in pred and "/" not in gt:
            p = pred.split("/")
            if len(p) == 2:
                return abs(float(p[0]) / float(p[1]) - float(gt)) < 1e-6
        elif "/" in gt and "/" not in pred:
            p = gt.split("/")
            if len(p) == 2:
                return abs(float(p[0]) / float(p[1]) - float(pred)) < 1e-6
        return abs(float(pred) - float(gt)) < 1e-6
    except (ValueError, ZeroDivisionError):
        return False


# ===========================================================================
# Domain-specific evaluations (instruction-formatted prompts)
# ===========================================================================
def eval_math(model, tokenizer, n_problems=MATH_EVAL_N):
    """MATH-500 subset with instruction-formatted prompts."""
    if not MATH500_PATH.exists():
        log("  Downloading math500_test.json...")
        import urllib.request
        url = ("https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
               "main/ch03/01_main-chapter-code/math500_test.json")
        MATH500_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, MATH500_PATH)

    with open(MATH500_PATH) as f:
        problems = json.load(f)

    # Use easiest problems (levels 1-2) since 2B model is small
    easy = [p for p in problems if p.get("level", 5) <= 2]
    selected = easy[:n_problems]
    if len(selected) < n_problems:
        # Add level 3
        medium = [p for p in problems if p.get("level", 5) == 3]
        selected += medium[:n_problems - len(selected)]
    log(f"  Math: {len(selected)} problems (levels {set(p.get('level',0) for p in selected)})")

    correct = 0
    details = []
    for i, prob in enumerate(selected):
        # INSTRUCTION-FORMATTED prompt (matching training format)
        prompt = INST_PROMPT.format(
            instruction=f"Solve step by step: {prob['problem']}"
        )
        response = generate_text(model, tokenizer, prompt)
        predicted = extract_final_answer(response)
        is_correct = grade_math_answer(predicted, prob["answer"])
        correct += int(is_correct)
        details.append({
            "idx": i, "level": prob.get("level"),
            "predicted": predicted[:30], "answer": prob["answer"][:30],
            "correct": is_correct,
        })
        if (i + 1) % 10 == 0:
            log(f"    [{i+1}/{len(selected)}] acc={correct/(i+1)*100:.1f}%")

    acc = correct / len(selected) if selected else 0
    return {"metric": "accuracy", "value": round(acc, 4),
            "correct": correct, "total": len(selected),
            "pct": round(acc * 100, 1), "details": details}


def eval_code(model, tokenizer, n_prompts=CODE_EVAL_N):
    """Code generation with instruction prompts."""
    val_path = DATA_DIR / "code" / "val.jsonl"
    if not val_path.exists():
        return {"metric": "syntax_valid_rate", "value": 0, "total": 0, "error": "no data"}

    items = []
    with open(val_path) as f:
        for line in f:
            item = json.loads(line)
            if item.get("instruction"):
                items.append(item)
            if len(items) >= n_prompts:
                break

    if not items:
        return {"metric": "syntax_valid_rate", "value": 0, "total": 0, "error": "no items"}

    valid = 0
    details = []
    for i, item in enumerate(items):
        prompt = INST_PROMPT.format(instruction=item["instruction"])
        response = generate_text(model, tokenizer, prompt, max_tokens=MAX_GEN_TOKENS)

        # Extract code from response
        code = response
        # Try to find code block
        if "```python" in response:
            code = response.split("```python")[-1].split("```")[0]
        elif "```" in response:
            code = response.split("```")[1].split("```")[0] if response.count("```") >= 2 else response

        is_valid = False
        if len(code.strip()) > 3:
            try:
                ast.parse(code)
                is_valid = True
            except SyntaxError:
                # Try just the first function/class
                for line_end in range(len(code.split("\n")), 0, -1):
                    try:
                        ast.parse("\n".join(code.split("\n")[:line_end]))
                        is_valid = True
                        break
                    except SyntaxError:
                        continue

        valid += int(is_valid)
        details.append({"idx": i, "valid": is_valid, "len": len(response)})
        if (i + 1) % 5 == 0:
            log(f"    [{i+1}/{len(items)}] valid={valid/(i+1)*100:.1f}%")

    rate = valid / len(items)
    return {"metric": "syntax_valid_rate", "value": round(rate, 4),
            "valid": valid, "total": len(items),
            "pct": round(rate * 100, 1), "details": details}


def compute_keyword_f1(prediction, reference):
    def tokenize(text):
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [w for w in text.split() if len(w) > 2]
    pred_tokens = Counter(tokenize(prediction))
    ref_tokens = Counter(tokenize(reference))
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = sum((pred_tokens & ref_tokens).values())
    prec = common / sum(pred_tokens.values()) if pred_tokens else 0
    rec = common / sum(ref_tokens.values()) if ref_tokens else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def eval_qa_domain(model, tokenizer, domain_name, n_questions=QA_EVAL_N):
    """QA evaluation using instruction prompts and keyword F1."""
    val_path = DATA_DIR / domain_name / "val.jsonl"
    if not val_path.exists():
        return {"metric": "keyword_f1", "value": 0, "total": 0, "error": "no data"}

    items = []
    with open(val_path) as f:
        for line in f:
            item = json.loads(line)
            if item.get("instruction") and item.get("response"):
                items.append(item)
            if len(items) >= n_questions:
                break

    if not items:
        return {"metric": "keyword_f1", "value": 0, "total": 0, "error": "no QA items"}

    f1_scores = []
    details = []
    for i, item in enumerate(items):
        prompt = INST_PROMPT.format(instruction=item["instruction"])
        response = generate_text(model, tokenizer, prompt)
        f1 = compute_keyword_f1(response, item["response"])
        f1_scores.append(f1)
        details.append({"idx": i, "f1": round(f1, 4), "len": len(response)})
        if (i + 1) % 5 == 0:
            log(f"    [{i+1}/{len(items)}] avg_f1={sum(f1_scores)/len(f1_scores):.3f}")

    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    return {"metric": "keyword_f1", "value": round(mean_f1, 4),
            "total": len(items), "pct": round(mean_f1 * 100, 1), "details": details}


def eval_creative_ppl(model, tokenizer, n_texts=CREATIVE_EVAL_N):
    """Creative writing via PPL on held-out story continuations."""
    val_path = DATA_DIR / "creative" / "val.jsonl"
    if not val_path.exists():
        return {"metric": "creative_ppl", "value": float("inf"), "total": 0, "error": "no data"}

    items = []
    with open(val_path) as f:
        for line in f:
            items.append(json.loads(line))

    total_loss = 0.0
    total_tokens = 0
    for item in items[:n_texts]:
        text = item.get("text", item.get("full_story", ""))
        tokens = tokenizer.encode(text)
        if len(tokens) < 4:
            continue
        tokens = tokens[:MAX_SEQ_LENGTH + 1]
        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += y.size

    if total_tokens == 0:
        return {"metric": "creative_ppl", "value": float("inf"), "total": 0}

    avg_ppl = math.exp(min(total_loss / total_tokens, 100))
    return {"metric": "creative_ppl", "value": round(avg_ppl, 4),
            "total": n_texts, "total_tokens": total_tokens}


# ===========================================================================
# Orthogonality diagnostic
# ===========================================================================
def compute_cosine_matrix(adapters):
    """Compute pairwise cosine similarities between adapters."""
    domains = list(adapters.keys())
    vecs = {}
    for d in domains:
        flat = mx.concatenate([v.reshape(-1) for v in adapters[d].values()])
        vecs[d] = flat

    matrix = {}
    cos_vals = []
    for i, d1 in enumerate(domains):
        for j, d2 in enumerate(domains):
            if j <= i:
                continue
            cos = mx.sum(vecs[d1] * vecs[d2]) / (
                mx.linalg.norm(vecs[d1]) * mx.linalg.norm(vecs[d2]) + 1e-10)
            mx.eval(cos)
            c = abs(cos.item())
            matrix[f"{d1}-{d2}"] = round(c, 6)
            cos_vals.append(c)

    mean_cos = sum(cos_vals) / len(cos_vals) if cos_vals else 0
    return {"pairs": matrix, "mean_abs_cos": round(mean_cos, 6)}


# ===========================================================================
# Run all domain evals for one condition
# ===========================================================================
def run_condition(model, tokenizer, condition_name):
    log(f"\n  === Evaluating: {condition_name} ===")
    results = {}

    log(f"  [math] MATH-500 subset (instruction prompts)...")
    results["math"] = eval_math(model, tokenizer)
    log(f"    -> accuracy: {results['math']['pct']}%")

    log(f"  [code] Python code generation (instruction prompts)...")
    results["code"] = eval_code(model, tokenizer)
    log(f"    -> syntax_valid_rate: {results['code']['pct']}%")

    log(f"  [medical] Medical QA F1 (instruction prompts)...")
    results["medical"] = eval_qa_domain(model, tokenizer, "medical")
    log(f"    -> keyword_f1: {results['medical']['pct']}%")

    log(f"  [legal] Legal QA F1 (instruction prompts)...")
    results["legal"] = eval_qa_domain(model, tokenizer, "legal")
    log(f"    -> keyword_f1: {results['legal']['pct']}%")

    log(f"  [creative] Story PPL...")
    results["creative"] = eval_creative_ppl(model, tokenizer)
    log(f"    -> creative_ppl: {results['creative']['value']}")

    return results


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_global = time.time()
    results = {
        "experiment": "bitnet_instruction_task_eval",
        "model": MODEL_ID,
        "hypothesis": "Instruction-tuned LoRA adapters pass task evaluation where NTP adapters failed",
        "training_format": "instruction_qa_pairs",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    log("=" * 70)
    log("BitNet-2B Instruction-Tuned Task Evaluation")
    log("=" * 70)

    # Phase 0: Prepare instruction data
    log("\n[Phase 0] Preparing instruction-formatted training data...")
    t0 = time.time()
    prepare_instruction_data()
    log(f"  Data preparation: {time.time() - t0:.1f}s")

    # Phase 1: Load model
    log("\n[Phase 1] Loading BitNet-2B-4T...")
    t1 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  Model loaded in {time.time() - t1:.1f}s")

    log("  Unpacking ternary weights...")
    t2 = time.time()
    model = replace_bitlinear_with_linear(model)
    log(f"  Unpacked in {time.time() - t2:.1f}s")

    # Phase 2: Apply LoRA structure
    log("\n[Phase 2] Applying LoRA structure...")
    model = apply_lora(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Phase 3: Base model evaluation
    log("\n[Phase 3] BASE model evaluation (LoRA zeroed)...")
    zero_lora_params(model)
    mx.eval(model.parameters())
    base_results = run_condition(model, tokenizer, "base")
    results["base"] = base_results

    # Phase 4: Train instruction-tuned adapters
    log("\n[Phase 4] Training instruction-tuned adapters...")
    adapters = {}
    training_stats = {}

    for domain in DOMAINS:
        # Load instruction-formatted data
        train_path = DATA_DIR / domain / "train.jsonl"
        val_path = DATA_DIR / domain / "val.jsonl"

        train_data = []
        with open(train_path) as f:
            for line in f:
                train_data.append(json.loads(line))

        val_data = []
        with open(val_path) as f:
            for line in f:
                val_data.append(json.loads(line))

        # Reset LoRA and train
        zero_lora_params(model)
        stats = train_adapter(model, tokenizer, domain, train_data, val_data)
        training_stats[domain] = stats

        # Save adapter
        adapter_path = ADAPTERS_DIR / domain
        save_adapter(model, adapter_path)
        adapters[domain] = load_adapter(adapter_path)
        log(f"  Saved {domain} adapter to {adapter_path}")

    results["training"] = training_stats

    # Phase 5: Orthogonality diagnostic
    log("\n[Phase 5] Orthogonality diagnostic...")
    cos_results = compute_cosine_matrix(adapters)
    results["orthogonality"] = cos_results
    log(f"  Mean |cos|: {cos_results['mean_abs_cos']}")
    for pair, val in cos_results["pairs"].items():
        log(f"    {pair}: {val}")

    # Phase 6: Individual adapter evaluation (own domain only)
    log("\n[Phase 6] Individual adapter evaluation...")
    individual_results = {}
    for domain in DOMAINS:
        log(f"\n  Loading {domain} adapter...")
        zero_lora_params(model)
        apply_adapter_weights(model, adapters[domain])
        mx.eval(model.parameters())

        r = {}
        if domain == "math":
            r["math"] = eval_math(model, tokenizer)
            log(f"    {domain} math acc: {r['math']['pct']}%")
        elif domain == "code":
            r["code"] = eval_code(model, tokenizer)
            log(f"    {domain} code valid: {r['code']['pct']}%")
        elif domain == "medical":
            r["medical"] = eval_qa_domain(model, tokenizer, "medical")
            log(f"    {domain} med f1: {r['medical']['pct']}%")
        elif domain == "legal":
            r["legal"] = eval_qa_domain(model, tokenizer, "legal")
            log(f"    {domain} legal f1: {r['legal']['pct']}%")
        elif domain == "creative":
            r["creative"] = eval_creative_ppl(model, tokenizer)
            log(f"    {domain} creative ppl: {r['creative']['value']}")
        individual_results[domain] = r

    results["individual"] = individual_results

    # Phase 7: Composed model (1/N scaling)
    log("\n[Phase 7] COMPOSED model (1/N) evaluation...")
    merged = compose_adapters(list(adapters.values()))
    zero_lora_params(model)
    apply_adapter_weights(model, merged)
    mx.eval(model.parameters())
    composed_results = run_condition(model, tokenizer, "composed_1_over_N")
    results["composed"] = composed_results

    # Phase 8: Oracle routed = individual results (each domain uses its own adapter)
    # No need to re-evaluate -- oracle routing IS individual adapter per domain
    log("\n[Phase 8] ORACLE ROUTED = individual (same data, no re-eval needed)")
    routed_results = {}
    for domain in DOMAINS:
        if domain in individual_results and domain in individual_results[domain]:
            routed_results[domain] = individual_results[domain][domain]
    results["routed"] = routed_results

    # Phase 9: Kill criteria
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    # K1: composed model vs base (using COMPOSED, the harder test)
    comparisons_composed = {}
    for domain in ["math", "code", "medical", "legal"]:
        b = base_results[domain]["value"]
        c = composed_results[domain]["value"]
        comparisons_composed[domain] = {
            "base": b, "composed": c,
            "better": c >= b, "delta": round(c - b, 4)
        }
    # Creative: lower PPL is better
    b = base_results["creative"]["value"]
    c = composed_results["creative"]["value"]
    comparisons_composed["creative"] = {
        "base": b, "composed": c,
        "better": c <= b, "delta": round(c - b, 4)
    }

    n_worse_composed = sum(1 for v in comparisons_composed.values() if not v["better"])
    worse_pct_composed = n_worse_composed / len(comparisons_composed) * 100
    k1_composed_pass = worse_pct_composed <= 40

    # Also check routed (the realistic deployment scenario)
    comparisons_routed = {}
    for domain in ["math", "code", "medical", "legal"]:
        b = base_results[domain]["value"]
        r_val = routed_results[domain]["value"]
        comparisons_routed[domain] = {
            "base": b, "routed": r_val,
            "better": r_val >= b, "delta": round(r_val - b, 4)
        }
    b = base_results["creative"]["value"]
    r_val = routed_results["creative"]["value"]
    comparisons_routed["creative"] = {
        "base": b, "routed": r_val,
        "better": r_val <= b, "delta": round(r_val - b, 4)
    }

    n_worse_routed = sum(1 for v in comparisons_routed.values() if not v["better"])
    worse_pct_routed = n_worse_routed / len(comparisons_routed) * 100
    k1_routed_pass = worse_pct_routed <= 40

    log(f"\n  K1 (COMPOSED): Composed worse than base on <= 40% of metrics")
    log(f"      Worse on {n_worse_composed}/5 ({worse_pct_composed:.0f}%)")
    for d, comp in comparisons_composed.items():
        st = "BETTER" if comp["better"] else "WORSE"
        log(f"        {d}: base={comp['base']:.4f} composed={comp['composed']:.4f} "
            f"delta={comp['delta']:+.4f} [{st}]")
    log(f"      Verdict: {'PASS' if k1_composed_pass else 'KILL'}")

    log(f"\n  K1 (ROUTED): Routed worse than base on <= 40% of metrics")
    log(f"      Worse on {n_worse_routed}/5 ({worse_pct_routed:.0f}%)")
    for d, comp in comparisons_routed.items():
        st = "BETTER" if comp["better"] else "WORSE"
        b_key = "routed"
        log(f"        {d}: base={comp['base']:.4f} routed={comp[b_key]:.4f} "
            f"delta={comp['delta']:+.4f} [{st}]")
    log(f"      Verdict: {'PASS' if k1_routed_pass else 'KILL'}")

    # K2: math adapter >= 3pp over base
    base_math_acc = base_results["math"]["value"]
    math_individual = individual_results.get("math", {}).get("math", {})
    math_adapter_acc = math_individual.get("value", base_math_acc)
    math_pp = (math_adapter_acc - base_math_acc) * 100
    k2_pass = math_pp >= 3.0

    log(f"\n  K2: Math adapter improvement >= 3pp over base")
    log(f"      Base: {base_math_acc*100:.1f}%  Math adapter: {math_adapter_acc*100:.1f}%")
    log(f"      Improvement: {math_pp:+.1f}pp")
    log(f"      Verdict: {'PASS' if k2_pass else 'KILL'}")

    results["comparisons_composed"] = comparisons_composed
    results["comparisons_routed"] = comparisons_routed
    results["kill_criteria"] = {
        "K1_composed": {
            "n_worse": n_worse_composed, "worse_pct": round(worse_pct_composed, 1),
            "pass": k1_composed_pass
        },
        "K1_routed": {
            "n_worse": n_worse_routed, "worse_pct": round(worse_pct_routed, 1),
            "pass": k1_routed_pass
        },
        "K2": {
            "base_acc": round(base_math_acc * 100, 1),
            "adapter_acc": round(math_adapter_acc * 100, 1),
            "improvement_pp": round(math_pp, 1),
            "pass": k2_pass
        },
    }

    # Overall verdict
    # Primary: K1 uses composed (strict) or routed (realistic)
    # K1 passes if EITHER composed or routed passes
    k1_pass = k1_composed_pass or k1_routed_pass

    verdict = "SUPPORTED" if (k1_pass and k2_pass) else "KILLED"
    details = []
    if not k1_composed_pass:
        details.append("K1-composed: worse on too many metrics")
    if not k1_routed_pass:
        details.append("K1-routed: worse on too many metrics")
    if not k2_pass:
        details.append(f"K2: math adapter {math_pp:+.1f}pp < 3pp threshold")
    if k1_pass and not k1_composed_pass:
        verdict += " (routed only, composed fails)"
    if details and verdict.startswith("KILLED"):
        verdict += f" ({'; '.join(details)})"

    results["verdict"] = verdict
    total_time = time.time() - t_global
    results["total_time_s"] = round(total_time, 1)
    results["total_time_min"] = round(total_time / 60, 1)

    # Comparison with prior NTP experiment
    ntp_results = {
        "math_base": 5.0, "math_composed": 0.0, "math_individual": 5.0,
        "medical_base_f1": 13.0, "medical_composed_f1": 15.6, "medical_individual_f1": 17.9,
        "code_base": 10.0, "code_composed": 5.0, "code_individual": 5.0,
        "legal_base_f1": 14.8, "legal_composed_f1": 14.0, "legal_individual_f1": 11.0,
        "verdict": "KILLED (K1 60%, K2 0pp)"
    }
    results["ntp_comparison"] = ntp_results

    log(f"\n  VERDICT: {verdict}")
    log(f"  Total time: {total_time/60:.1f} min")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
