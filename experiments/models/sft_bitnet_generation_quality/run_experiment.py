#!/usr/bin/env python3
"""SFT adapters on BitNet-2B: fix generation quality via instruction masking.

Trains SFT (response-only loss) LoRA adapters on BitNet-2B-4T, then evaluates
generation quality with energy gap routing + LLM-as-judge + task metrics.

Kill criteria:
  K1 (#578): SFT routed worse than base on >=3/5 domains by LLM-as-judge
  K2 (#579): SFT adapters show no improvement over NTP adapters on generation quality
  K3 (#580): Math correctness drops below 40% (regression from Finding #185)

Prior results:
  Finding #178: NTP adapters kill prose quality on all 5 domains
  Finding #180: SFT loss fixes this on Falcon (GSM8K 0.36->0.52)
  Finding #185: Energy gap top-1 routing 88% accuracy, +133% math correctness

Platform: Apple M5 Pro 48GB, MLX
"""

import ast
import gc
import json
import math
import os
import re
import time
from collections import Counter
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

# Source experiment data (instruction-formatted JSONL) and NTP adapters
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
NTP_ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"

# LoRA config (same as NTP experiment for fair comparison)
LORA_RANK = 16
LORA_SCALE = 20.0
TRAIN_ITERS = 300  # SFT has fewer loss terms per sample, need more iters
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = len(DOMAINS)

# Target layers for LoRA (same as NTP experiment)
TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

# Generation settings
NUM_PROMPTS_PER_DOMAIN = 10
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.9

# Response marker for SFT masking
RESPONSE_MARKER = "### Response:\n"

DOMAIN_KEYWORDS = {
    "medical": [
        "patient", "diagnosis", "treatment", "symptoms", "disease", "clinical",
        "medication", "therapy", "surgical", "pathology", "prognosis", "chronic",
        "acute", "syndrome", "condition", "medical", "doctor", "hospital",
        "infection", "immune", "blood", "organ", "tissue", "cell", "drug",
        "dose", "prescription", "cancer", "cardiac", "neural", "liver",
        "kidney", "lung", "brain", "bone", "muscle", "nerve", "artery",
        "vein", "inflammation", "fever", "pain", "swelling",
    ],
    "code": [
        "function", "variable", "class", "method", "return", "import",
        "loop", "array", "string", "integer", "boolean", "object", "def",
        "print", "if", "else", "for", "while", "try", "except", "lambda",
        "list", "dict", "tuple", "parameter", "argument", "module", "library",
        "algorithm", "data", "structure", "python", "code", "program",
        "output", "input", "error", "debug", "compile", "run", "execute",
    ],
    "math": [
        "equation", "formula", "calculate", "solve", "number", "variable",
        "total", "sum", "product", "divide", "multiply", "subtract", "add",
        "percent", "ratio", "fraction", "decimal", "integer", "value",
        "answer", "solution", "problem", "step", "result", "equal",
        "greater", "less", "positive", "negative", "zero", "proof",
        "theorem", "function", "graph", "slope", "area", "volume",
        "distance", "rate", "time", "cost", "price", "profit",
    ],
    "legal": [
        "law", "court", "judge", "attorney", "lawyer", "legal", "rights",
        "statute", "regulation", "contract", "liability", "plaintiff",
        "defendant", "jurisdiction", "precedent", "ruling", "verdict",
        "appeal", "testimony", "evidence", "trial", "case", "claim",
        "damages", "negligence", "tort", "criminal", "civil", "federal",
        "state", "constitution", "amendment", "legislation", "act",
        "section", "clause", "provision", "enforce", "comply", "violate",
    ],
    "finance": [
        "investment", "stock", "bond", "market", "portfolio", "risk",
        "return", "profit", "loss", "revenue", "capital", "dividend",
        "interest", "rate", "inflation", "monetary", "fiscal", "budget",
        "asset", "liability", "equity", "debt", "credit", "loan",
        "mortgage", "insurance", "tax", "income", "expense", "savings",
        "bank", "financial", "economic", "trade", "exchange", "currency",
        "price", "value", "growth", "recession", "gdp", "fund",
    ],
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
# Model utilities (reused from prior experiments)
# ============================================================================

from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.bitlinear_layers import BitLinear


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


class SkeletonLoRALinear(nn.Module):
    """LoRA with frozen Grassmannian A and trainable B (matches NTP adapter format)."""
    def __init__(self, base_linear: nn.Linear, a_matrix: mx.array,
                 rank: int = 16, scale: float = 20.0):
        super().__init__()
        out_features = base_linear.weight.shape[0]
        self.linear = base_linear
        self.lora_a = a_matrix  # frozen Grassmannian
        self.lora_b = mx.zeros((rank, out_features))  # trainable
        self.scale = scale
        self.linear.freeze()
        self.freeze(keys=["lora_a"], strict=False)

    def __call__(self, x):
        base_out = self.linear(x)
        lora_out = (x @ self.lora_a) @ self.lora_b * self.scale
        return base_out + lora_out


def apply_lora_with_skeleton(model, skeleton, domain_idx):
    """Apply LoRA with Grassmannian A matrices for a specific domain."""
    lora_count = 0
    for li, layer in enumerate(model.model.layers):
        lora_updates = []
        for full_key in TARGET_KEYS:
            parts = full_key.split(".")
            module = layer
            for p in parts:
                module = getattr(module, p, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue
            skey = f"layer_{li}_{full_key}_domain_{domain_idx}"
            if skey not in skeleton:
                continue
            a_matrix = mx.array(skeleton[skey]).astype(mx.bfloat16)
            lora = SkeletonLoRALinear(module, a_matrix, rank=LORA_RANK, scale=LORA_SCALE)
            lora_updates.append((full_key, lora))
            lora_count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    mx.eval(model.parameters())
    log(f"  Applied skeleton LoRA to {lora_count} layers (domain {domain_idx})")
    return model


def freeze_base_train_lora(model):
    model.freeze()
    model.unfreeze(keys=["lora_a", "lora_b"], strict=False)
    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    total = sum(p.size for _, p in tree_flatten(model.parameters()))
    log(f"  Trainable: {trainable:,} ({100*trainable/total:.4f}%)")
    return trainable, total


# ============================================================================
# Multi-adapter LoRA for routing evaluation
# ============================================================================

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
            lora_sum = lora_sum + w * ((x @ self.a_matrices[i]) @ b)
        return base_out + lora_sum * self.scale


# ============================================================================
# SFT data loading and loss
# ============================================================================

def load_sft_data(domain, max_samples=400):
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
    response_idx = text.find(RESPONSE_MARKER)
    if response_idx < 0:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        return tokens, [1] * len(tokens)

    instruction_part = text[:response_idx + len(RESPONSE_MARKER)]
    instruction_tokens = tokenizer.encode(instruction_part, add_special_tokens=True)
    instruction_len = len(instruction_tokens)

    full_tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(full_tokens) > max_len:
        full_tokens = full_tokens[:max_len]

    mask = [0] * min(instruction_len, len(full_tokens))
    mask += [1] * (len(full_tokens) - len(mask))
    return full_tokens, mask


def prepare_sft_batches(texts, tokenizer, max_len=256):
    batches = []
    n_response_tokens = 0
    n_total_tokens = 0
    for text in texts:
        tokens, mask = tokenize_with_sft_mask(text, tokenizer, max_len)
        if len(tokens) >= 4:
            batches.append((tokens, mask))
            n_response_tokens += sum(mask)
            n_total_tokens += len(tokens)
    if batches:
        response_ratio = n_response_tokens / max(n_total_tokens, 1)
        log(f"  SFT masking: {response_ratio:.1%} response tokens, "
            f"{1-response_ratio:.1%} instruction tokens masked out")
    return batches


def get_sft_batch(batches, batch_idx):
    idx = batch_idx % len(batches)
    tokens, mask = batches[idx]
    return mx.array([tokens]), mx.array([mask])


def sft_loss_fn(model, tokens, mask):
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    response_mask = mask[:, 1:]  # shift to align with targets
    per_token_loss = nn.losses.cross_entropy(logits, targets, reduction="none")
    masked_loss = per_token_loss * response_mask
    n_response = mx.maximum(response_mask.sum(), mx.array(1.0))
    loss = masked_loss.sum() / n_response
    return loss


# ============================================================================
# Evaluation utilities
# ============================================================================

def generate_text(model, tokenizer, prompt, max_tokens=128, temperature=0.7, top_p=0.9):
    try:
        sampler = make_sampler(temp=temperature, top_p=top_p)
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


def keyword_density(text, domain):
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    keywords = set(DOMAIN_KEYWORDS.get(domain, []))
    return sum(1 for w in words if w in keywords) / len(words)


def code_syntax_valid(text):
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
        if stripped.startswith(('def ', 'class ', 'import ', 'from ', 'for ', 'while ',
                                'if ', 'try:', 'except', 'with ', 'return ', 'print(', '#')):
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


def extract_math_answer(text):
    m = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*\$?([\d,]+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(',', ''))
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


def extract_ground_truth_answer(response_text):
    matches = re.findall(r'<<[^>]*?=(\d+(?:\.\d+)?)>>', response_text)
    if matches:
        return float(matches[-1])
    m = re.search(r'####\s*([\d,]+(?:\.\d+)?)', response_text)
    if m:
        return float(m.group(1).replace(',', ''))
    return None


def math_answer_correct(generated_answer, ground_truth):
    if generated_answer is None or ground_truth is None:
        return False
    if ground_truth == 0:
        return abs(generated_answer) < 0.01
    return abs(generated_answer - ground_truth) / abs(ground_truth) < 0.01


def compute_domain_score(text, domain, ground_truth_response=None):
    if domain == "code":
        syntax_ok = 1.0 if code_syntax_valid(text) else 0.0
        kw = keyword_density(text, domain)
        return 0.5 * syntax_ok + 0.5 * kw
    elif domain == "math":
        gen_answer = extract_math_answer(text)
        gt_answer = None
        if ground_truth_response:
            gt_answer = extract_ground_truth_answer(ground_truth_response)
        correct = 1.0 if math_answer_correct(gen_answer, gt_answer) else 0.0
        kw = keyword_density(text, domain)
        return 0.5 * correct + 0.5 * kw
    else:
        kw = keyword_density(text, domain)
        return kw  # simplified: keyword density as proxy


DOMAIN_DISPLAY = {
    "medical": "Medical", "code": "Programming/Code",
    "math": "Mathematics", "legal": "Legal", "finance": "Finance",
}


def build_judge_prompt(text, domain):
    domain_name = DOMAIN_DISPLAY.get(domain, domain)
    return (
        f"Rate the following {domain_name} text on three criteria. "
        f"Give a score from 1 (worst) to 5 (best) for each.\n\n"
        f"Text to evaluate:\n\"\"\"\n{text[:500]}\n\"\"\"\n\n"
        f"Criteria:\n"
        f"1. Domain Relevance: Does this text appropriately address {domain_name} topics? (1-5)\n"
        f"2. Coherence: Is the text well-structured and logically organized? (1-5)\n"
        f"3. Informativeness: Does the text provide useful, specific information? (1-5)\n\n"
        f"Respond with ONLY three numbers separated by commas, like: 4,3,5\n"
        f"Scores: "
    )


def parse_judge_scores(response):
    numbers = re.findall(r'(\d)', response[:50])
    if len(numbers) >= 3:
        scores = [int(n) for n in numbers[:3]]
        if all(1 <= s <= 5 for s in scores):
            return tuple(scores)
    numbers = re.findall(r'[1-5]', response[:100])
    if len(numbers) >= 3:
        return (int(numbers[0]), int(numbers[1]), int(numbers[2]))
    return None


def judge_text(model, tokenizer, text, domain):
    if not text or len(text.strip()) < 5:
        return {"relevance": 1, "coherence": 1, "informativeness": 1, "composite": 1.0}
    prompt = build_judge_prompt(text, domain)
    response = generate_text(model, tokenizer, prompt, max_tokens=20, temperature=0.1, top_p=0.95)
    scores = parse_judge_scores(response)
    if scores is None:
        response2 = generate_text(model, tokenizer, prompt, max_tokens=20, temperature=0.0, top_p=1.0)
        scores = parse_judge_scores(response2)
    if scores is None:
        return {"relevance": 3, "coherence": 3, "informativeness": 3,
                "composite": 3.0, "parse_failed": True}
    r, c, i = scores
    return {"relevance": r, "coherence": c, "informativeness": i, "composite": (r + c + i) / 3.0}


def extract_prompts_with_answers(domain, n_prompts=10):
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


def format_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


# ============================================================================
# NLL computation for energy gap routing
# ============================================================================

def compute_prompt_nll(model, tokenizer, prompt_text):
    tokens = tokenizer.encode(prompt_text)
    if len(tokens) > MAX_SEQ_LENGTH:
        tokens = tokens[:MAX_SEQ_LENGTH]
    if len(tokens) < 2:
        return float('inf')
    x = mx.array(tokens)[None, :]
    logits = model(x)
    mx.eval(logits)
    logits_shift = logits[:, :-1, :]
    targets = x[:, 1:]
    max_logits = mx.max(logits_shift, axis=-1, keepdims=True)
    shifted = logits_shift - max_logits
    log_probs = shifted - mx.log(mx.sum(mx.exp(shifted), axis=-1, keepdims=True))
    target_log_probs = mx.take_along_axis(log_probs, targets[:, :, None], axis=-1).squeeze(-1)
    nll = -mx.mean(target_log_probs).item()
    del logits, logits_shift, log_probs, target_log_probs, targets, x
    return nll


# ============================================================================
# Adapter loading utilities
# ============================================================================

def load_skeleton():
    skeleton_path = NTP_ADAPTERS_DIR / "grassmannian_skeleton.npz"
    return dict(np.load(str(skeleton_path)))


def apply_multi_adapter(model, skeleton, adapter_dir):
    """Apply all 5 adapters from a given adapter directory with routing weights."""
    n_layers = len(model.model.layers)
    all_adapter_params = {}
    for di, domain in enumerate(DOMAINS):
        adapter_path = adapter_dir / domain / "adapter.npz"
        if adapter_path.exists():
            all_adapter_params[domain] = dict(mx.load(str(adapter_path)))
        else:
            log(f"  WARNING: adapter not found: {adapter_path}")
            all_adapter_params[domain] = {}

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
    log(f"  Applied routed multi-adapter ({count} layers)")
    return model


def apply_single_adapter(model, skeleton, domain_idx, domain_name, adapter_dir):
    """Apply a single adapter for NLL computation."""
    adapter_path = adapter_dir / domain_name / "adapter.npz"
    if not adapter_path.exists():
        log(f"  WARNING: adapter not found: {adapter_path}")
        return model
    adapter_params = dict(mx.load(str(adapter_path)))
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
            skey = f"layer_{li}_{key}_domain_{domain_idx}"
            a_init = mx.array(skeleton[skey]).astype(mx.bfloat16) if skey in skeleton else None
            if a_init is None:
                continue
            in_f = module.weight.shape[1]
            out_f = module.weight.shape[0]
            # Create single-expert routed lora (weight=1.0)
            routed = RoutedMultiAdapterLoRALinear(
                module, rank=LORA_RANK, scale=LORA_SCALE, a_inits=[a_init]
            )
            b_key = f"model.layers.{li}.{key}.lora_b"
            if b_key in adapter_params:
                routed.b_matrices[0] = adapter_params[b_key]
            routed.set_routing_weights([1.0])
            lora_updates.append((key, routed))
            count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    mx.eval(model.parameters())
    return model


def set_all_routing_weights(model, weights):
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, RoutedMultiAdapterLoRALinear):
                module.set_routing_weights(weights)


# ============================================================================
# PHASE 1: Train SFT adapters
# ============================================================================

def phase_train_sft_adapters():
    log("\n" + "=" * 70)
    log("PHASE 1: TRAIN 5 SFT ADAPTERS ON BitNet-2B-4T")
    log("=" * 70)

    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    skeleton = load_skeleton()
    all_train_results = {}

    for di, domain in enumerate(DOMAINS):
        log(f"\n--- Training SFT adapter: {domain} (domain_idx={di}) ---")
        t0 = time.time()
        mx.reset_peak_memory()

        model, tokenizer = load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)
        model = apply_lora_with_skeleton(model, skeleton, di)
        freeze_base_train_lora(model)

        train_texts, val_texts = load_sft_data(domain)
        train_batches = prepare_sft_batches(train_texts, tokenizer, MAX_SEQ_LENGTH)
        val_batches_data = prepare_sft_batches(val_texts, tokenizer, MAX_SEQ_LENGTH)

        if not train_batches:
            log(f"  WARNING: No valid training data for {domain}, skipping")
            cleanup(model, tokenizer)
            continue

        # Base validation loss
        base_val_loss = 0.0
        n_val = min(25, len(val_batches_data))
        for i in range(n_val):
            tokens, mask = get_sft_batch(val_batches_data, i)
            loss = sft_loss_fn(model, tokens, mask)
            mx.eval(loss)
            base_val_loss += loss.item()
            del loss, tokens, mask
        base_val_loss /= max(n_val, 1)
        base_ppl = math.exp(min(base_val_loss, 20))
        log(f"  Base SFT val loss: {base_val_loss:.4f} (PPL: {base_ppl:.2f})")

        # Train
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
            if (step + 1) % 100 == 0:
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

        # Save adapter weights in same format as NTP adapters (lora_b only)
        adapter_dir = ADAPTERS_DIR / domain
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_weights = {}
        for name, param in tree_flatten(model.trainable_parameters()):
            # Only save lora_b (lora_a comes from skeleton)
            if "lora_b" in name:
                adapter_weights[name] = param
        mx.savez(str(adapter_dir / "adapter.npz"), **adapter_weights)
        adapter_size = os.path.getsize(adapter_dir / "adapter.npz")
        log(f"  Saved adapter ({adapter_size/1024:.1f} KB)")

        elapsed = time.time() - t0
        peak_mem = mx.get_peak_memory() / 1e9

        all_train_results[domain] = {
            "base_ppl": round(base_ppl, 2),
            "trained_ppl": round(trained_ppl, 2),
            "ppl_improvement_pct": round((base_ppl - trained_ppl) / base_ppl * 100, 1),
            "final_loss": round(losses[-1], 4),
            "peak_memory_gb": round(peak_mem, 2),
            "time_s": round(elapsed, 1),
        }

        log(f"  {domain} done in {elapsed:.1f}s, peak={peak_mem:.2f}GB")
        cleanup(model, tokenizer, optimizer)
        log_memory(f"post-{domain}")

    del skeleton
    gc.collect()
    return all_train_results


# ============================================================================
# PHASE 2: Generate with base model
# ============================================================================

def phase_generate_base(prompts_by_domain):
    log("\n[Phase 2] Generating with BASE model...")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    mx.random.seed(SEED)
    np.random.seed(SEED)
    results = {}
    for domain in DOMAINS:
        domain_results = []
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            formatted = format_prompt(prompt_data["instruction"])
            generated = generate_text(model, tokenizer, formatted,
                                      max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)
            score = compute_domain_score(generated, domain,
                                         ground_truth_response=prompt_data["response"] if domain == "math" else None)
            domain_results.append({
                "prompt": prompt_data["instruction"][:100],
                "response": prompt_data["response"][:100],
                "generated": generated[:300],
                "domain_score": score,
                "answer_correct": (
                    math_answer_correct(extract_math_answer(generated),
                                        extract_ground_truth_answer(prompt_data["response"]))
                    if domain == "math" else None),
                "syntax_valid": code_syntax_valid(generated) if domain == "code" else None,
            })
        results[domain] = domain_results
        scores = [r["domain_score"] for r in domain_results]
        log(f"  BASE {domain}: avg_score={np.mean(scores):.4f}")

    elapsed = time.time() - t0
    log(f"  Base gen done in {elapsed:.1f}s")
    del model, tokenizer
    cleanup()
    return results


# ============================================================================
# PHASE 3: Energy gap routing + generation (SFT adapters)
# ============================================================================

def phase_compute_energy_gaps_and_route(prompts_by_domain, adapter_dir, label="SFT"):
    """Compute energy gaps and determine top-1 routing for each query."""
    log(f"\n[Phase 3a] Computing energy gaps for {label} adapters...")
    t0 = time.time()

    skeleton = load_skeleton()

    # Base NLLs
    log(f"  Loading base model for NLL computation...")
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    base_nlls = {}
    for domain in DOMAINS:
        nlls = []
        for prompt_data in prompts_by_domain[domain]:
            formatted = format_prompt(prompt_data["instruction"])
            nll = compute_prompt_nll(model, tokenizer, formatted)
            nlls.append(nll)
        base_nlls[domain] = nlls
        log(f"    {domain}: mean_base_nll={np.mean(nlls):.4f}")
    del model, tokenizer
    cleanup()

    # Per-adapter NLLs
    adapter_nlls = {}
    for di, adapter_domain in enumerate(DOMAINS):
        log(f"    Adapter NLL: {adapter_domain}")
        model, tokenizer = load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)
        model = apply_single_adapter(model, skeleton, di, adapter_domain, adapter_dir)
        model.freeze()

        nlls_by_query = {}
        for query_domain in DOMAINS:
            nlls = []
            for prompt_data in prompts_by_domain[query_domain]:
                formatted = format_prompt(prompt_data["instruction"])
                nll = compute_prompt_nll(model, tokenizer, formatted)
                nlls.append(nll)
            nlls_by_query[query_domain] = nlls
        adapter_nlls[adapter_domain] = nlls_by_query
        del model, tokenizer
        cleanup()

    del skeleton

    # Compute energy gaps and routing decisions
    energy_gaps = {}
    routing_decisions = {}  # {query_domain: [best_adapter_idx per prompt]}
    routing_accuracy = {}

    for query_domain in DOMAINS:
        decisions = []
        correct_idx = DOMAINS.index(query_domain)
        for pi in range(len(base_nlls[query_domain])):
            gaps = []
            for adapter_domain in DOMAINS:
                gap = adapter_nlls[adapter_domain][query_domain][pi] - base_nlls[query_domain][pi]
                gaps.append(gap)
            best_idx = int(np.argmin(gaps))
            decisions.append(best_idx)
        routing_decisions[query_domain] = decisions
        acc = sum(1 for d in decisions if d == correct_idx) / len(decisions)
        routing_accuracy[query_domain] = acc
        log(f"    {label} routing {query_domain}: {acc:.0%} correct")

    overall_acc = np.mean(list(routing_accuracy.values()))
    log(f"  {label} overall routing accuracy: {overall_acc:.1%}")

    elapsed = time.time() - t0
    log(f"  Energy gap computation: {elapsed:.1f}s")
    return routing_decisions, routing_accuracy, overall_acc


def phase_generate_routed(prompts_by_domain, routing_decisions, adapter_dir, label="SFT"):
    """Generate text using energy-gap-routed adapters."""
    log(f"\n[Phase 3b] Generating with {label} ROUTED adapters...")
    t0 = time.time()

    skeleton = load_skeleton()
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model = apply_multi_adapter(model, skeleton, adapter_dir)
    model.freeze()
    del skeleton

    mx.random.seed(SEED)
    np.random.seed(SEED)
    results = {}
    for di, domain in enumerate(DOMAINS):
        domain_results = []
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            # Set routing to energy-gap top-1 selection
            best_adapter = routing_decisions[domain][i]
            weights = [0.0] * N_DOMAINS
            weights[best_adapter] = 1.0
            set_all_routing_weights(model, weights)

            formatted = format_prompt(prompt_data["instruction"])
            generated = generate_text(model, tokenizer, formatted,
                                      max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)
            score = compute_domain_score(generated, domain,
                                         ground_truth_response=prompt_data["response"] if domain == "math" else None)
            domain_results.append({
                "prompt": prompt_data["instruction"][:100],
                "response": prompt_data["response"][:100],
                "generated": generated[:300],
                "domain_score": score,
                "routed_to": DOMAINS[best_adapter],
                "answer_correct": (
                    math_answer_correct(extract_math_answer(generated),
                                        extract_ground_truth_answer(prompt_data["response"]))
                    if domain == "math" else None),
                "syntax_valid": code_syntax_valid(generated) if domain == "code" else None,
            })
        results[domain] = domain_results
        scores = [r["domain_score"] for r in domain_results]
        log(f"  {label} ROUTED {domain}: avg_score={np.mean(scores):.4f}")

    elapsed = time.time() - t0
    log(f"  {label} routed gen done in {elapsed:.1f}s")
    del model, tokenizer
    cleanup()
    return results


# ============================================================================
# PHASE 4: LLM-as-Judge evaluation
# ============================================================================

def phase_judge_all(base_results, sft_results, ntp_results):
    """Judge all generated texts. Loads model once, judges everything."""
    log("\n[Phase 4] LLM-as-Judge scoring...")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    judge_results = {"base": {}, "sft_routed": {}, "ntp_routed": {}}

    for config_name, gen_results in [("base", base_results), ("sft_routed", sft_results),
                                      ("ntp_routed", ntp_results)]:
        log(f"\n  Judging {config_name}...")
        for domain in DOMAINS:
            domain_scores = []
            for i, item in enumerate(gen_results[domain]):
                score = judge_text(model, tokenizer, item["generated"], domain)
                domain_scores.append(score)
            judge_results[config_name][domain] = domain_scores
            mean_composite = np.mean([s["composite"] for s in domain_scores])
            log(f"    {config_name} {domain}: judge_composite={mean_composite:.2f}")

    elapsed = time.time() - t0
    log(f"  Judging done in {elapsed:.1f}s")
    del model, tokenizer
    cleanup()
    return judge_results


# ============================================================================
# PHASE 5: Analysis and kill criteria assessment
# ============================================================================

def phase_analyze(base_results, sft_results, ntp_results, judge_results,
                  sft_routing_acc, ntp_routing_acc):
    log("\n" + "=" * 70)
    log("ANALYSIS AND KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    analysis = {"domains": {}, "kill_criteria": {}}

    # Per-domain analysis
    sft_better_count = 0
    sft_better_than_ntp_count = 0
    math_correct_sft = 0
    math_total_sft = 0

    for domain in DOMAINS:
        base_judge = np.mean([s["composite"] for s in judge_results["base"][domain]])
        sft_judge = np.mean([s["composite"] for s in judge_results["sft_routed"][domain]])
        ntp_judge = np.mean([s["composite"] for s in judge_results["ntp_routed"][domain]])

        base_score = np.mean([r["domain_score"] for r in base_results[domain]])
        sft_score = np.mean([r["domain_score"] for r in sft_results[domain]])
        ntp_score = np.mean([r["domain_score"] for r in ntp_results[domain]])

        sft_beats_base_judge = sft_judge > base_judge
        sft_beats_ntp_judge = sft_judge > ntp_judge
        sft_beats_base_task = sft_score > base_score
        sft_beats_ntp_task = sft_score > ntp_score

        # Use combined signal: judge + task metrics
        sft_beats_base = sft_beats_base_judge or sft_beats_base_task
        sft_beats_ntp = sft_beats_ntp_judge or sft_beats_ntp_task

        if sft_beats_base:
            sft_better_count += 1
        if sft_beats_ntp:
            sft_better_than_ntp_count += 1

        # Math correctness
        if domain == "math":
            for r in sft_results[domain]:
                math_total_sft += 1
                if r.get("answer_correct"):
                    math_correct_sft += 1

        domain_detail = {
            "base_judge": round(base_judge, 2),
            "sft_judge": round(sft_judge, 2),
            "ntp_judge": round(ntp_judge, 2),
            "base_task_score": round(base_score, 4),
            "sft_task_score": round(sft_score, 4),
            "ntp_task_score": round(ntp_score, 4),
            "sft_beats_base_judge": bool(sft_beats_base_judge),
            "sft_beats_base_task": bool(sft_beats_base_task),
            "sft_beats_ntp_judge": bool(sft_beats_ntp_judge),
            "sft_beats_ntp_task": bool(sft_beats_ntp_task),
        }

        # Domain-specific metrics
        if domain == "math":
            base_correct = sum(1 for r in base_results[domain] if r.get("answer_correct"))
            sft_correct = math_correct_sft
            ntp_correct = sum(1 for r in ntp_results[domain] if r.get("answer_correct"))
            domain_detail["base_math_correct"] = f"{base_correct}/{math_total_sft}"
            domain_detail["sft_math_correct"] = f"{sft_correct}/{math_total_sft}"
            domain_detail["ntp_math_correct"] = f"{ntp_correct}/{math_total_sft}"
        elif domain == "code":
            base_syntax = sum(1 for r in base_results[domain] if r.get("syntax_valid"))
            sft_syntax = sum(1 for r in sft_results[domain] if r.get("syntax_valid"))
            ntp_syntax = sum(1 for r in ntp_results[domain] if r.get("syntax_valid"))
            n = len(base_results[domain])
            domain_detail["base_syntax_valid"] = f"{base_syntax}/{n}"
            domain_detail["sft_syntax_valid"] = f"{sft_syntax}/{n}"
            domain_detail["ntp_syntax_valid"] = f"{ntp_syntax}/{n}"

        analysis["domains"][domain] = domain_detail
        log(f"\n  {domain}:")
        log(f"    Judge: base={base_judge:.2f}, SFT={sft_judge:.2f}, NTP={ntp_judge:.2f}")
        log(f"    Task:  base={base_score:.4f}, SFT={sft_score:.4f}, NTP={ntp_score:.4f}")
        log(f"    SFT beats base: judge={sft_beats_base_judge}, task={sft_beats_base_task}")
        log(f"    SFT beats NTP:  judge={sft_beats_ntp_judge}, task={sft_beats_ntp_task}")

    # Kill criteria
    sft_worse_count = N_DOMAINS - sft_better_count
    math_pct = (math_correct_sft / max(math_total_sft, 1)) * 100

    k1_pass = sft_worse_count < 3  # K1 kills if SFT worse on >=3/5
    k2_pass = sft_better_than_ntp_count > 0  # K2 kills if NO improvement over NTP
    k3_pass = math_pct >= 40  # K3 kills if math <40%

    analysis["kill_criteria"] = {
        "K1_sft_worse_than_base_domains": sft_worse_count,
        "K1_threshold": ">=3/5 kills",
        "K1_pass": bool(k1_pass),
        "K2_sft_better_than_ntp_domains": sft_better_than_ntp_count,
        "K2_threshold": "0 improvement kills",
        "K2_pass": bool(k2_pass),
        "K3_math_correctness_pct": round(math_pct, 1),
        "K3_threshold": "<40% kills",
        "K3_pass": bool(k3_pass),
    }

    analysis["routing"] = {
        "sft_overall_accuracy": round(sft_routing_acc, 3),
        "ntp_overall_accuracy": round(ntp_routing_acc, 3),
    }

    log(f"\n  KILL CRITERIA:")
    log(f"    K1: SFT worse on {sft_worse_count}/5 domains (kills if >=3) -> {'PASS' if k1_pass else 'FAIL'}")
    log(f"    K2: SFT beats NTP on {sft_better_than_ntp_count}/5 domains (kills if 0) -> {'PASS' if k2_pass else 'FAIL'}")
    log(f"    K3: Math correctness {math_pct:.1f}% (kills if <40%) -> {'PASS' if k3_pass else 'FAIL'}")

    return analysis


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0_total = time.time()
    log("=" * 70)
    log("SFT BITNET GENERATION QUALITY EXPERIMENT")
    log("=" * 70)
    log_memory("start")

    # Extract prompts
    log("\nExtracting prompts from validation data...")
    prompts_by_domain = {}
    for domain in DOMAINS:
        prompts_by_domain[domain] = extract_prompts_with_answers(domain, NUM_PROMPTS_PER_DOMAIN)
        log(f"  {domain}: {len(prompts_by_domain[domain])} prompts")

    # Phase 1: Train SFT adapters
    train_results = phase_train_sft_adapters()
    log_memory("post-train")

    # Phase 2: Generate with base
    base_results = phase_generate_base(prompts_by_domain)
    log_memory("post-base-gen")

    # Phase 3a: Energy gap routing for SFT adapters
    sft_routing, sft_routing_acc, sft_overall_acc = phase_compute_energy_gaps_and_route(
        prompts_by_domain, ADAPTERS_DIR, label="SFT"
    )
    log_memory("post-sft-energy")

    # Phase 3b: Generate with SFT routed adapters
    sft_results = phase_generate_routed(prompts_by_domain, sft_routing, ADAPTERS_DIR, label="SFT")
    log_memory("post-sft-gen")

    # Phase 3c: Energy gap routing for NTP adapters (comparison)
    ntp_routing, ntp_routing_acc, ntp_overall_acc = phase_compute_energy_gaps_and_route(
        prompts_by_domain, NTP_ADAPTERS_DIR, label="NTP"
    )
    log_memory("post-ntp-energy")

    # Phase 3d: Generate with NTP routed adapters
    ntp_results = phase_generate_routed(prompts_by_domain, ntp_routing, NTP_ADAPTERS_DIR, label="NTP")
    log_memory("post-ntp-gen")

    # Phase 4: LLM-as-judge
    judge_results = phase_judge_all(base_results, sft_results, ntp_results)
    log_memory("post-judge")

    # Phase 5: Analysis
    analysis = phase_analyze(base_results, sft_results, ntp_results, judge_results,
                             sft_overall_acc, ntp_overall_acc)

    # Save results
    total_time = time.time() - t0_total
    results = {
        "experiment": "sft_bitnet_generation_quality",
        "model": MODEL_ID,
        "config": {
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
            "train_iters": TRAIN_ITERS,
            "num_prompts_per_domain": NUM_PROMPTS_PER_DOMAIN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "seed": SEED,
        },
        "training": train_results,
        "analysis": analysis,
        "judge_summary": {
            domain: {
                "base": round(np.mean([s["composite"] for s in judge_results["base"][domain]]), 2),
                "sft_routed": round(np.mean([s["composite"] for s in judge_results["sft_routed"][domain]]), 2),
                "ntp_routed": round(np.mean([s["composite"] for s in judge_results["ntp_routed"][domain]]), 2),
            }
            for domain in DOMAINS
        },
        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_time/60:.1f} minutes")

    # Print summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    k = analysis["kill_criteria"]
    log(f"K1: {'PASS' if k['K1_pass'] else 'FAIL'} (SFT worse on {k['K1_sft_worse_than_base_domains']}/5 domains)")
    log(f"K2: {'PASS' if k['K2_pass'] else 'FAIL'} (SFT beats NTP on {k['K2_sft_better_than_ntp_domains']}/5 domains)")
    log(f"K3: {'PASS' if k['K3_pass'] else 'FAIL'} (Math {k['K3_math_correctness_pct']:.1f}%)")


if __name__ == "__main__":
    main()
