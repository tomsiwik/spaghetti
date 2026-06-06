#!/usr/bin/env python3
"""Generation Quality Retest: LLM-as-Judge + Task Benchmarks Replace Keyword Density

Retests the prior exp_generation_quality_test which was KILLED (3/5 domains worse)
using keyword density. The kill was suspected to be an evaluation artifact because
keyword density penalizes format-appropriate responses.

This experiment uses:
  - LLM-as-judge (base model self-evaluation) for prose domains
  - Task-specific metrics retained: code syntax validity, math answer correctness
  - 50 prompts per domain (up from 10), 1 seed (50 paired samples/domain)
  - Wilcoxon signed-rank test for statistical comparison
  - Correlation analysis (Spearman) between old and new metrics (K2)

Configurations:
  1. Base only (no adapters)
  2. Routed top-1 (oracle routing: single expert, weight=1.0)

Kill criteria:
  K1 (#560): Routed worse than base on >= 3/5 domains using LLM-as-judge scoring
  K2 (#561): LLM-as-judge agrees with keyword density r>0.7 (old metric was fine)

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
from scipy import stats

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
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256

DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = len(DOMAINS)

# Generation settings
NUM_PROMPTS_PER_DOMAIN = 50
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.9
# 1 seed x 50 prompts = 50 paired samples/domain.
# Power analysis: >95% power for medium effects (d=0.5) with Wilcoxon at n=50.
# 3 seeds would require ~4+ hours; 1 seed completes in ~70 min.
SEEDS = [42]

# Domain keyword lists (same as prior experiment, for correlation analysis)
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

DOMAIN_DISPLAY = {
    "medical": "Medical",
    "code": "Programming/Code",
    "math": "Mathematics",
    "legal": "Legal",
    "finance": "Finance",
}


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
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
# BitNet unpacking and model utilities (reused from prior experiment)
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
# LoRA layers for composition (reused from prior experiment)
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
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, RoutedMultiAdapterLoRALinear):
                module.set_routing_weights(weights)


def get_oracle_routing_weights_top1(domain_idx, n_experts=5):
    weights = [0.0] * n_experts
    weights[domain_idx] = 1.0
    return weights


# ============================================================================
# Text generation
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


# ============================================================================
# Old metrics (for correlation analysis with K2)
# ============================================================================

def keyword_density(text, domain):
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    keywords = set(DOMAIN_KEYWORDS.get(domain, []))
    return sum(1 for w in words if w in keywords) / len(words)


def ngram_diversity(text, n=3):
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def repetition_score(text):
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def coherence_score(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    word_counts = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    avg_len = np.mean(word_counts)
    return max(0, 1.0 - abs(avg_len - 15) / 30)


def old_composite_score(text, domain):
    """The OLD scoring function from the prior experiment (for K2 correlation)."""
    if domain == "code":
        syntax_ok = 1.0 if code_syntax_valid(text) else 0.0
        kw = keyword_density(text, domain)
        return 0.5 * syntax_ok + 0.5 * kw
    elif domain == "math":
        # Can't compute without ground truth, use keyword density only
        kw = keyword_density(text, domain)
        return kw
    else:
        kw = keyword_density(text, domain)
        div = ngram_diversity(text)
        coh = coherence_score(text)
        rep = repetition_score(text)
        return 0.45 * kw + 0.25 * div + 0.10 * coh + 0.20 * rep


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
                                'if ', 'try:', 'except', 'with ', 'return ', 'print(',
                                '#')):
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


def is_incoherent(text):
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 3:
        return True
    counter = Counter(words)
    most_common_freq = counter.most_common(1)[0][1] / len(words)
    if most_common_freq > 0.4:
        return True
    if len(set(words)) < 5 and len(words) > 20:
        return True
    return False


# ============================================================================
# LLM-as-Judge scoring
# ============================================================================

def build_judge_prompt(text, domain):
    """Build a prompt for the base model to judge generated text quality.

    The judge evaluates on three criteria (1-5 each):
    - Domain Relevance
    - Coherence
    - Informativeness

    Returns a prompt that asks for structured output.
    """
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
    """Parse the judge's response to extract three scores.

    Returns (relevance, coherence, informativeness) or None if parsing fails.
    """
    # Try to find three numbers in the response
    # Pattern: "4,3,5" or "4, 3, 5" or "4 3 5"
    numbers = re.findall(r'(\d)', response[:50])
    if len(numbers) >= 3:
        scores = [int(n) for n in numbers[:3]]
        # Validate range
        if all(1 <= s <= 5 for s in scores):
            return tuple(scores)

    # Fallback: try to find any three numbers 1-5
    numbers = re.findall(r'[1-5]', response[:100])
    if len(numbers) >= 3:
        return (int(numbers[0]), int(numbers[1]), int(numbers[2]))

    return None


def judge_text(model, tokenizer, text, domain):
    """Have the base model judge a generated text.

    Returns dict with relevance, coherence, informativeness, and composite score.
    """
    if not text or len(text.strip()) < 5:
        return {"relevance": 1, "coherence": 1, "informativeness": 1, "composite": 1.0}

    prompt = build_judge_prompt(text, domain)
    # Use low temperature for more consistent judging
    response = generate_text(
        model, tokenizer, prompt,
        max_tokens=20, temperature=0.1, top_p=0.95
    )

    scores = parse_judge_scores(response)
    if scores is None:
        # If parsing fails, try once more with greedy decoding
        response2 = generate_text(
            model, tokenizer, prompt,
            max_tokens=20, temperature=0.0, top_p=1.0
        )
        scores = parse_judge_scores(response2)

    if scores is None:
        # Default to neutral score if judging fails
        return {"relevance": 3, "coherence": 3, "informativeness": 3,
                "composite": 3.0, "parse_failed": True, "raw_response": response[:100]}

    r, c, i = scores
    return {
        "relevance": r,
        "coherence": c,
        "informativeness": i,
        "composite": (r + c + i) / 3.0,
        "raw_response": response[:100],
    }


# ============================================================================
# Prompt extraction
# ============================================================================

def extract_prompts_with_answers(domain, n_prompts=50):
    """Extract instruction prompts from validation data."""
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
# Phase 1: Generate text with base and routed models
# ============================================================================

def phase_generate_base(prompts_by_domain, seed):
    """Generate text with base model (no adapters) for a single seed."""
    log(f"\n[Generate BASE] seed={seed}...")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("post-load-base")

    mx.random.seed(seed)
    np.random.seed(seed)
    results = {}
    for domain in DOMAINS:
        domain_results = []
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            formatted = format_prompt(prompt_data["instruction"])
            generated = generate_text(
                model, tokenizer, formatted,
                max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P
            )
            domain_results.append({
                "prompt": prompt_data["instruction"],
                "response": prompt_data["response"],
                "generated": generated,
            })
            if (i + 1) % 10 == 0:
                log(f"  {domain}: {i+1}/{len(prompts_by_domain[domain])}")
        results[domain] = domain_results
    elapsed = time.time() - t0
    log(f"  Base generation (seed={seed}) done in {elapsed:.1f}s")
    log_memory("post-gen-base")
    cleanup(model, tokenizer)
    return results


def phase_generate_routed(prompts_by_domain, seed):
    """Generate text with routed top-1 adapter composition for a single seed."""
    log(f"\n[Generate ROUTED] seed={seed}...")
    t0 = time.time()

    skeleton = load_skeleton()
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model = apply_multi_adapter_routed(model, skeleton)
    model.freeze()
    log_memory("post-load-routed")

    mx.random.seed(seed)
    np.random.seed(seed)
    results = {}
    for di, domain in enumerate(DOMAINS):
        weights = get_oracle_routing_weights_top1(di, N_DOMAINS)
        set_all_routing_weights(model, weights)
        domain_results = []
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            formatted = format_prompt(prompt_data["instruction"])
            generated = generate_text(
                model, tokenizer, formatted,
                max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P
            )
            domain_results.append({
                "prompt": prompt_data["instruction"],
                "response": prompt_data["response"],
                "generated": generated,
            })
            if (i + 1) % 10 == 0:
                log(f"  {domain}: {i+1}/{len(prompts_by_domain[domain])}")
        results[domain] = domain_results
    elapsed = time.time() - t0
    log(f"  Routed generation (seed={seed}) done in {elapsed:.1f}s")
    log_memory("post-gen-routed")
    cleanup(model, tokenizer)
    del skeleton
    gc.collect()
    return results


# ============================================================================
# Phase 2: LLM-as-Judge evaluation
# ============================================================================

def phase_judge(all_base, all_routed):
    """Judge all generated texts using the base model.

    Loads the base model once and judges all texts from all seeds/configs.
    Returns per-seed, per-domain, per-prompt judge scores for base and routed.
    """
    log("\n[Phase: LLM-as-Judge] Scoring all generated texts...")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("post-load-judge")

    n_seeds = len(all_base)
    judge_base = []
    judge_routed = []

    for seed_idx in range(n_seeds):
        seed = SEEDS[seed_idx]
        log(f"\n  Judging seed={seed}...")
        base_scores = {}
        routed_scores = {}

        for domain in DOMAINS:
            base_domain_scores = []
            routed_domain_scores = []

            base_texts = all_base[seed_idx][domain]
            routed_texts = all_routed[seed_idx][domain]

            for i in range(len(base_texts)):
                # Judge base text
                b_score = judge_text(model, tokenizer, base_texts[i]["generated"], domain)
                base_domain_scores.append(b_score)

                # Judge routed text
                r_score = judge_text(model, tokenizer, routed_texts[i]["generated"], domain)
                routed_domain_scores.append(r_score)

                if (i + 1) % 10 == 0:
                    log(f"    {domain}: judged {i+1}/{len(base_texts)} pairs")

            base_scores[domain] = base_domain_scores
            routed_scores[domain] = routed_domain_scores

            # Quick summary
            b_mean = np.mean([s["composite"] for s in base_domain_scores])
            r_mean = np.mean([s["composite"] for s in routed_domain_scores])
            log(f"    {domain}: base_judge={b_mean:.2f}, routed_judge={r_mean:.2f}")

        judge_base.append(base_scores)
        judge_routed.append(routed_scores)

    elapsed = time.time() - t0
    log(f"  Judging done in {elapsed:.1f}s")
    log_memory("post-judge")
    cleanup(model, tokenizer)
    return judge_base, judge_routed


# ============================================================================
# Phase 3: Compute old metrics (for K2 correlation)
# ============================================================================

def phase_compute_old_metrics(all_base, all_routed):
    """Compute old keyword-density-based metrics for correlation analysis (K2)."""
    log("\n[Phase: Old Metrics] Computing keyword density scores for K2 correlation...")

    n_seeds = len(all_base)
    old_base = []
    old_routed = []

    for seed_idx in range(n_seeds):
        base_scores = {}
        routed_scores = {}
        for domain in DOMAINS:
            base_domain = []
            routed_domain = []
            for i in range(len(all_base[seed_idx][domain])):
                b_text = all_base[seed_idx][domain][i]["generated"]
                r_text = all_routed[seed_idx][domain][i]["generated"]

                b_old = old_composite_score(b_text, domain)
                r_old = old_composite_score(r_text, domain)

                # Also compute task-specific for code/math
                b_extra = {}
                r_extra = {}
                if domain == "code":
                    b_extra["syntax_valid"] = code_syntax_valid(b_text)
                    r_extra["syntax_valid"] = code_syntax_valid(r_text)
                elif domain == "math":
                    gt = extract_ground_truth_answer(
                        all_base[seed_idx][domain][i]["response"]
                    )
                    b_extra["answer_correct"] = math_answer_correct(
                        extract_math_answer(b_text), gt
                    )
                    r_extra["answer_correct"] = math_answer_correct(
                        extract_math_answer(r_text), gt
                    )

                base_domain.append({"old_score": b_old, **b_extra})
                routed_domain.append({"old_score": r_old, **r_extra})

            base_scores[domain] = base_domain
            routed_scores[domain] = routed_domain

        old_base.append(base_scores)
        old_routed.append(routed_scores)

    log("  Old metrics computed.")
    return old_base, old_routed


# ============================================================================
# Analysis
# ============================================================================

def analyze_results(all_base, all_routed, judge_base, judge_routed, old_base, old_routed):
    """Full analysis: judge scores, statistical tests, kill criteria, correlation."""
    log("\n[Analysis] Computing results...")
    n_seeds = len(all_base)

    results = {
        "per_domain": {},
        "kill_criteria": {},
        "k2_correlation": {},
        "statistical_tests": {},
    }

    routed_worse_count = 0
    all_domain_correlations = []

    for domain in DOMAINS:
        # Collect all paired scores across seeds
        all_base_judge = []
        all_routed_judge = []
        all_base_old = []
        all_routed_old = []
        all_base_relevance = []
        all_routed_relevance = []
        all_base_coherence_j = []
        all_routed_coherence_j = []
        all_base_info = []
        all_routed_info = []
        parse_fails_base = 0
        parse_fails_routed = 0

        for seed_idx in range(n_seeds):
            for i in range(len(judge_base[seed_idx][domain])):
                b_j = judge_base[seed_idx][domain][i]
                r_j = judge_routed[seed_idx][domain][i]

                all_base_judge.append(b_j["composite"])
                all_routed_judge.append(r_j["composite"])
                all_base_relevance.append(b_j["relevance"])
                all_routed_relevance.append(r_j["relevance"])
                all_base_coherence_j.append(b_j["coherence"])
                all_routed_coherence_j.append(r_j["coherence"])
                all_base_info.append(b_j["informativeness"])
                all_routed_info.append(r_j["informativeness"])
                if b_j.get("parse_failed"):
                    parse_fails_base += 1
                if r_j.get("parse_failed"):
                    parse_fails_routed += 1

                b_o = old_base[seed_idx][domain][i]
                r_o = old_routed[seed_idx][domain][i]
                all_base_old.append(b_o["old_score"])
                all_routed_old.append(r_o["old_score"])

        n_samples = len(all_base_judge)
        base_judge_mean = float(np.mean(all_base_judge))
        routed_judge_mean = float(np.mean(all_routed_judge))
        base_judge_std = float(np.std(all_base_judge))
        routed_judge_std = float(np.std(all_routed_judge))

        base_old_mean = float(np.mean(all_base_old))
        routed_old_mean = float(np.mean(all_routed_old))

        # Wilcoxon signed-rank test (paired)
        diffs = np.array(all_routed_judge) - np.array(all_base_judge)
        nonzero_diffs = diffs[diffs != 0]
        if len(nonzero_diffs) > 0:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(nonzero_diffs)
        else:
            wilcoxon_stat, wilcoxon_p = 0.0, 1.0

        # Direction: is routed better?
        routed_wins = routed_judge_mean > base_judge_mean
        if not routed_wins:
            routed_worse_count += 1

        # Effect size (matched-pairs rank-biserial correlation)
        if len(nonzero_diffs) > 0:
            n_nz = len(nonzero_diffs)
            effect_size = 1 - (2 * wilcoxon_stat) / (n_nz * (n_nz + 1))
        else:
            effect_size = 0.0

        # Spearman correlation between old and new metrics (K2)
        # Combine base and routed scores to get a range of scores
        combined_old = all_base_old + all_routed_old
        combined_new = all_base_judge + all_routed_judge
        if len(set(combined_old)) > 1 and len(set(combined_new)) > 1:
            spearman_r, spearman_p = stats.spearmanr(combined_old, combined_new)
        else:
            spearman_r, spearman_p = 0.0, 1.0

        all_domain_correlations.append(spearman_r)

        # Task-specific metrics for code/math
        task_metrics = {}
        if domain == "code":
            b_syntax = [old_base[si][domain][i].get("syntax_valid", False)
                        for si in range(n_seeds) for i in range(len(old_base[si][domain]))]
            r_syntax = [old_routed[si][domain][i].get("syntax_valid", False)
                        for si in range(n_seeds) for i in range(len(old_routed[si][domain]))]
            task_metrics["base_syntax_valid_rate"] = float(np.mean(b_syntax))
            task_metrics["routed_syntax_valid_rate"] = float(np.mean(r_syntax))
        elif domain == "math":
            b_correct = [old_base[si][domain][i].get("answer_correct", False)
                         for si in range(n_seeds) for i in range(len(old_base[si][domain]))]
            r_correct = [old_routed[si][domain][i].get("answer_correct", False)
                         for si in range(n_seeds) for i in range(len(old_routed[si][domain]))]
            task_metrics["base_answer_correct_rate"] = float(np.mean(b_correct))
            task_metrics["routed_answer_correct_rate"] = float(np.mean(r_correct))

        # Bonferroni-adjusted significance
        bonferroni_alpha = 0.05 / len(DOMAINS)
        significant = wilcoxon_p < bonferroni_alpha

        results["per_domain"][domain] = {
            "n_samples": n_samples,
            "base_judge_mean": round(base_judge_mean, 3),
            "base_judge_std": round(base_judge_std, 3),
            "routed_judge_mean": round(routed_judge_mean, 3),
            "routed_judge_std": round(routed_judge_std, 3),
            "judge_delta": round(routed_judge_mean - base_judge_mean, 3),
            "judge_delta_pct": round((routed_judge_mean - base_judge_mean) / max(base_judge_mean, 0.01) * 100, 1),
            "routed_wins_judge": bool(routed_wins),
            "base_old_mean": round(base_old_mean, 4),
            "routed_old_mean": round(routed_old_mean, 4),
            "old_delta_pct": round((routed_old_mean - base_old_mean) / max(base_old_mean, 0.01) * 100, 1),
            "routed_wins_old": bool(routed_old_mean > base_old_mean),
            "wilcoxon_stat": round(float(wilcoxon_stat), 1),
            "wilcoxon_p": float(wilcoxon_p),
            "effect_size_r": round(float(effect_size), 3),
            "significant_bonferroni": bool(significant),
            "spearman_r_old_vs_new": round(float(spearman_r), 3),
            "spearman_p": float(spearman_p),
            "parse_fails_base": parse_fails_base,
            "parse_fails_routed": parse_fails_routed,
            "sub_scores": {
                "base_relevance_mean": round(float(np.mean(all_base_relevance)), 2),
                "routed_relevance_mean": round(float(np.mean(all_routed_relevance)), 2),
                "base_coherence_mean": round(float(np.mean(all_base_coherence_j)), 2),
                "routed_coherence_mean": round(float(np.mean(all_routed_coherence_j)), 2),
                "base_informativeness_mean": round(float(np.mean(all_base_info)), 2),
                "routed_informativeness_mean": round(float(np.mean(all_routed_info)), 2),
            },
            **task_metrics,
        }

        results["statistical_tests"][domain] = {
            "test": "Wilcoxon signed-rank (two-sided)",
            "n_pairs": n_samples,
            "n_nonzero_diffs": int(len(nonzero_diffs)),
            "stat": round(float(wilcoxon_stat), 1),
            "p_value": float(wilcoxon_p),
            "bonferroni_alpha": bonferroni_alpha,
            "significant": bool(significant),
            "effect_size_r": round(float(effect_size), 3),
            "direction": "routed > base" if routed_wins else "base > routed",
        }

    # Kill criteria
    k1_kill = routed_worse_count >= 3
    overall_spearman = float(np.mean(all_domain_correlations))
    k2_kill = overall_spearman > 0.7

    results["kill_criteria"] = {
        "k1": {
            "id": 560,
            "test": "Routed worse than base on >= 3/5 domains (LLM-as-judge)",
            "value": f"{routed_worse_count}/5 domains worse",
            "result": "FAIL" if k1_kill else "PASS",
        },
        "k2": {
            "id": 561,
            "test": "LLM-as-judge agrees with keyword density r>0.7",
            "value": f"mean Spearman r = {overall_spearman:.3f}",
            "per_domain": {d: round(float(all_domain_correlations[i]), 3)
                          for i, d in enumerate(DOMAINS)},
            "result": "FAIL" if k2_kill else "PASS",
        },
    }

    verdict = "KILLED" if k1_kill else "SUPPORTED"
    results["verdict"] = verdict
    results["k2_interpretation"] = (
        "Old metric agreed with new -> prior kill was real, not artifact"
        if k2_kill else
        "Old metric disagreed with new -> prior kill was likely artifact"
    )

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("Generation Quality Retest: LLM-as-Judge + Task Benchmarks")
    log("K1: Routed worse on >= 3/5 domains (judge) -> KILL")
    log("K2: Judge agrees with keyword density r>0.7 -> old metric was fine")
    log("=" * 70)
    log_memory("start")

    # Verify adapters exist
    for domain in DOMAINS:
        adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
        if not adapter_path.exists():
            log(f"ERROR: Missing adapter for {domain} at {adapter_path}")
            return

    # Extract prompts
    log("\n[Setup] Extracting prompts from validation data...")
    prompts_by_domain = {}
    for domain in DOMAINS:
        prompts = extract_prompts_with_answers(domain, NUM_PROMPTS_PER_DOMAIN)
        prompts_by_domain[domain] = prompts
        log(f"  {domain}: {len(prompts)} prompts")

    # Phase 1: Generate text with base and routed for all seeds
    all_base = []
    all_routed = []

    for seed_idx, seed in enumerate(SEEDS):
        log(f"\n{'='*70}")
        log(f"SEED {seed_idx+1}/{len(SEEDS)}: {seed}")
        log(f"{'='*70}")

        base_results = phase_generate_base(prompts_by_domain, seed)
        all_base.append(base_results)

        routed_results = phase_generate_routed(prompts_by_domain, seed)
        all_routed.append(routed_results)

    # Phase 2: LLM-as-Judge evaluation (single model load for all texts)
    judge_base, judge_routed = phase_judge(all_base, all_routed)

    # Phase 3: Compute old metrics for K2 correlation
    old_base, old_routed = phase_compute_old_metrics(all_base, all_routed)

    # Analysis
    analysis = analyze_results(all_base, all_routed, judge_base, judge_routed, old_base, old_routed)

    # Build full results
    full_results = {
        "experiment": "generation_quality_llm_judge",
        "model": MODEL_ID,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "num_prompts_per_domain": NUM_PROMPTS_PER_DOMAIN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seeds": SEEDS,
            "routing": "oracle_top1 (single expert, weight=1.0)",
            "judge_model": MODEL_ID + " (self-evaluation)",
            "judge_temperature": 0.1,
        },
        **analysis,
        "total_time_s": round(time.time() - t0, 1),
    }

    # Save sample generations for inspection
    samples = {}
    for domain in DOMAINS:
        samples[domain] = {
            "prompt": all_base[0][domain][0]["prompt"],
            "base": all_base[0][domain][0]["generated"][:500],
            "routed": all_routed[0][domain][0]["generated"][:500],
        }
    full_results["generated_samples"] = samples

    RESULTS_FILE.write_text(json.dumps(full_results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")

    # Print summary
    log("\n" + "=" * 70)
    log("SUMMARY: LLM-as-Judge Scores (mean across 3 seeds, 50 prompts/domain)")
    log("=" * 70)

    for domain in DOMAINS:
        d = analysis["per_domain"][domain]
        sig_marker = "*" if d["significant_bonferroni"] else ""
        log(f"\n{domain}:")
        log(f"  Judge: base={d['base_judge_mean']:.3f}+/-{d['base_judge_std']:.3f}  "
            f"routed={d['routed_judge_mean']:.3f}+/-{d['routed_judge_std']:.3f}  "
            f"delta={d['judge_delta']:+.3f} ({d['judge_delta_pct']:+.1f}%){sig_marker}")
        log(f"  Old:   base={d['base_old_mean']:.4f}  routed={d['routed_old_mean']:.4f}  "
            f"delta={d['old_delta_pct']:+.1f}%")
        log(f"  Wilcoxon p={d['wilcoxon_p']:.4f}, effect_size={d['effect_size_r']:.3f}")
        log(f"  Spearman(old,new)={d['spearman_r_old_vs_new']:.3f}")
        log(f"  Sub-scores: rel={d['sub_scores']['base_relevance_mean']:.1f}->{d['sub_scores']['routed_relevance_mean']:.1f} "
            f"coh={d['sub_scores']['base_coherence_mean']:.1f}->{d['sub_scores']['routed_coherence_mean']:.1f} "
            f"inf={d['sub_scores']['base_informativeness_mean']:.1f}->{d['sub_scores']['routed_informativeness_mean']:.1f}")
        log(f"  Routed wins (judge): {d['routed_wins_judge']}  |  Routed wins (old): {d['routed_wins_old']}")
        if "base_syntax_valid_rate" in d:
            log(f"  Syntax valid: base={d['base_syntax_valid_rate']:.3f} routed={d['routed_syntax_valid_rate']:.3f}")
        if "base_answer_correct_rate" in d:
            log(f"  Answer correct: base={d['base_answer_correct_rate']:.3f} routed={d['routed_answer_correct_rate']:.3f}")
        log(f"  Parse failures: base={d['parse_fails_base']}, routed={d['parse_fails_routed']}")

    log(f"\nKill Criteria:")
    for k, v in analysis["kill_criteria"].items():
        log(f"  {k} (#{v['id']}): {v['result']} -- {v['test']} ({v['value']})")
        if "per_domain" in v:
            for d, r in v["per_domain"].items():
                log(f"    {d}: r={r}")

    log(f"\nVERDICT: {analysis['verdict']}")
    log(f"K2 interpretation: {analysis['k2_interpretation']}")
    log(f"Total time: {full_results['total_time_s']:.1f}s")


if __name__ == "__main__":
    main()
