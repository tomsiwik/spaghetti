#!/usr/bin/env python3
"""Generation Quality Test v2: Does routed composition produce better TEXT than base alone?

REVISION of v1 incorporating 6 fixes from adversarial review:
  Fix 1: Top-1 routing only (single expert, weight=1.0, no secondary)
  Fix 2: Cross-PPL is DIAGNOSTIC only, NOT in primary scoring
  Fix 3: Domain-appropriate metrics:
         - Code: syntax validity (ast.parse) + keyword density
         - Math: answer correctness (binary) + keyword density
         - Medical/Legal/Finance: composite with coherence=10%
  Fix 4: 3 seeds minimum (42, 137, 2024), report mean+std
  Fix 5: Same K1 criterion: routed worse on >= 3/5 domains -> KILL
  Fix 6: XPPL normalization asymmetry documented in MATH.md

Configurations tested:
  1. Base only (no adapters)
  2. Uniform 1/N composition (all 5 adapters equally weighted)
  3. Routed top-1 (oracle routing: single expert, weight=1.0)

Kill criteria:
  K1 (id=272): Routed composition worse than base on >= 3/5 domains -> KILL
  K2 (id=273): No measurable difference between routed and base -> KILL
  K3 (id=274): All generated text is incoherent (base model too weak at 2B) -> KILL

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
NUM_PROMPTS_PER_DOMAIN = 10
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.9
SEEDS = [42, 137, 2024]  # Fix 4: 3 seeds minimum

# Domain keyword lists for scoring
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
# BitNet unpacking and model utilities (reused from real_data_domain_experts)
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
# LoRA layers for composition
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
    """LoRA with multiple A/B pairs and per-expert routing weights.

    Forward: y = base(x) + sum_i[w_i * (x @ A_i) @ ternary(B_i)] * scale
    where w_i are the routing weights (set externally per sequence).

    Fix 1: Top-1 routing means only one w_i is 1.0, rest are 0.0.
    """
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
        """Set per-expert routing weights for this forward pass."""
        self._routing_weights = weights

    def __call__(self, x):
        base_out = self.linear(x)
        if self.n_experts == 0:
            return base_out

        lora_sum = mx.zeros_like(base_out)
        for i in range(self.n_experts):
            w = self._routing_weights[i]
            if w < 1e-6:
                continue  # Skip experts with zero weight
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
# Fix 1: Top-1 oracle routing (single expert, weight=1.0)
# ============================================================================

def get_oracle_routing_weights_top1(domain_idx, n_experts=5):
    """Oracle top-1 routing: full weight to the correct domain expert only.

    Fix 1: Removes the arbitrary secondary expert pairing that caused
    interference (especially legal+finance) in v1. This is the clean test
    of whether a single domain-specialized adapter improves generation.
    """
    weights = [0.0] * n_experts
    weights[domain_idx] = 1.0
    return weights


# ============================================================================
# Text generation
# ============================================================================

def generate_text(model, tokenizer, prompt, max_tokens=128, temperature=0.7, top_p=0.9):
    """Generate text using mlx_lm.generate which handles KV cache properly."""
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
# Scoring metrics
# ============================================================================

def keyword_density(text, domain):
    """Fraction of words that are domain keywords."""
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    keywords = set(DOMAIN_KEYWORDS.get(domain, []))
    matches = sum(1 for w in words if w in keywords)
    return matches / len(words)


def ngram_diversity(text, n=3):
    """Fraction of unique n-grams. Higher = more diverse."""
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def word_count(text):
    """Number of words in text."""
    return len(re.findall(r'\b\w+\b', text))


def repetition_score(text):
    """Ratio of unique words to total words. Higher = less repetitive."""
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def coherence_score(text):
    """Simple coherence metric: average sentence length (proxy for coherence).
    Score is 1.0 for sentences of ~15 words, lower for extremes.
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    avg_len = np.mean([word_count(s) for s in sentences])
    return max(0, 1.0 - abs(avg_len - 15) / 30)


def is_incoherent(text):
    """Detect obviously incoherent text (repeated tokens, gibberish)."""
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
# Fix 3: Domain-appropriate metrics
# ============================================================================

def code_syntax_valid(text):
    """Check if generated text contains valid Python code (via ast.parse).

    Tries to parse the entire text, then tries to extract code blocks
    and parse those. Returns True if ANY parseable Python is found.
    """
    # Try parsing the whole text as Python
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        pass

    # Try extracting code blocks (```python ... ```)
    code_blocks = re.findall(r'```(?:python)?\s*\n?(.*?)```', text, re.DOTALL)
    for block in code_blocks:
        try:
            ast.parse(block.strip())
            return True
        except SyntaxError:
            continue

    # Try lines that look like Python (start with def, class, import, etc.)
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
        code_text = '\n'.join(code_lines)
        try:
            ast.parse(code_text)
            return True
        except SyntaxError:
            pass

    return False


def extract_math_answer(text):
    """Extract a numeric answer from generated math text.

    Looks for patterns like:
    - "the answer is 42"
    - "= 42"
    - "$42" or "42 dollars"
    - Final number in the text
    """
    # Pattern: "the answer is X" or "answer: X"
    m = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*\$?([\d,]+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(',', ''))

    # Pattern: "= X" (final equation result)
    matches = re.findall(r'=\s*\$?([\d,]+(?:\.\d+)?)', text)
    if matches:
        return float(matches[-1].replace(',', ''))

    # Pattern: "$X" (dollar amount)
    matches = re.findall(r'\$([\d,]+(?:\.\d+)?)', text)
    if matches:
        return float(matches[-1].replace(',', ''))

    # Last number in the text
    matches = re.findall(r'([\d,]+(?:\.\d+)?)', text)
    if matches:
        try:
            return float(matches[-1].replace(',', ''))
        except ValueError:
            pass

    return None


def extract_ground_truth_answer(response_text):
    """Extract the ground truth numeric answer from the training data response.

    The math data uses <<calc>>result format, e.g. <<5*2=10>>10.
    The final such number is typically the answer.
    """
    # Look for <<...=X>>X patterns - get the last one (final answer)
    matches = re.findall(r'<<[^>]*?=(\d+(?:\.\d+)?)>>', response_text)
    if matches:
        return float(matches[-1])

    # Fallback: last number after "####"
    m = re.search(r'####\s*([\d,]+(?:\.\d+)?)', response_text)
    if m:
        return float(m.group(1).replace(',', ''))

    return None


def math_answer_correct(generated_answer, ground_truth):
    """Check if extracted answer matches ground truth (within tolerance)."""
    if generated_answer is None or ground_truth is None:
        return False
    # Allow 1% tolerance for floating point
    if ground_truth == 0:
        return abs(generated_answer) < 0.01
    return abs(generated_answer - ground_truth) / abs(ground_truth) < 0.01


def compute_domain_score(text, domain, ground_truth_response=None):
    """Compute the PRIMARY domain-appropriate quality score for generated text.

    Fix 3: Domain-specific scoring:
    - Code: syntax_valid (0 or 1) * 0.5 + keyword_density * 0.5
    - Math: answer_correct (0 or 1) * 0.5 + keyword_density * 0.5
    - Medical/Legal/Finance: reweighted composite:
        keyword_density * 45% + ngram_diversity * 25% + coherence * 10% + repetition * 20%

    Fix 2: Cross-PPL is NOT included here. It is diagnostic only.
    """
    if domain == "code":
        syntax_ok = 1.0 if code_syntax_valid(text) else 0.0
        kw = keyword_density(text, domain)
        return 0.5 * syntax_ok + 0.5 * kw

    elif domain == "math":
        # Extract answer from generated text
        gen_answer = extract_math_answer(text)
        # Extract ground truth from training data response
        gt_answer = None
        if ground_truth_response:
            gt_answer = extract_ground_truth_answer(ground_truth_response)
        correct = 1.0 if math_answer_correct(gen_answer, gt_answer) else 0.0
        kw = keyword_density(text, domain)
        return 0.5 * correct + 0.5 * kw

    else:
        # Medical, Legal, Finance: reweighted composite
        # Fix 3: coherence dropped to 10% (penalizes non-prose unfairly)
        kw = keyword_density(text, domain)
        div = ngram_diversity(text)
        coh = coherence_score(text)
        rep = repetition_score(text)
        return 0.45 * kw + 0.25 * div + 0.10 * coh + 0.20 * rep


# ============================================================================
# Prompt extraction (with ground truth for math)
# ============================================================================

def extract_prompts_with_answers(domain, n_prompts=10):
    """Extract instruction prompts (and ground truth responses for math) from validation data."""
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
    """Format instruction as model input (instruction-response format)."""
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


# ============================================================================
# Phase 1: Generate with base model (across all seeds)
# ============================================================================

def phase_generate_base(prompts_by_domain, seed):
    """Generate text with base model (no adapters) for a single seed."""
    log(f"\n[Phase 1] Generating with BASE model (seed={seed})...")
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
            score = compute_domain_score(
                generated, domain,
                ground_truth_response=prompt_data["response"] if domain == "math" else None
            )
            domain_results.append({
                "prompt": prompt_data["instruction"],
                "ground_truth_response": prompt_data["response"] if domain == "math" else None,
                "generated": generated,
                "word_count": word_count(generated),
                "keyword_density": keyword_density(generated, domain),
                "ngram_diversity": ngram_diversity(generated),
                "repetition": repetition_score(generated),
                "coherence": coherence_score(generated),
                "incoherent": is_incoherent(generated),
                "domain_score": score,
                # Domain-specific sub-metrics
                "syntax_valid": code_syntax_valid(generated) if domain == "code" else None,
                "answer_correct": (
                    math_answer_correct(
                        extract_math_answer(generated),
                        extract_ground_truth_answer(prompt_data["response"])
                    ) if domain == "math" else None
                ),
            })
            if (i + 1) % 5 == 0:
                log(f"  {domain}: {i+1}/{len(prompts_by_domain[domain])} prompts done")

        results[domain] = domain_results
        avg_score = np.mean([r["domain_score"] for r in domain_results])
        log(f"  {domain}: avg_domain_score={avg_score:.3f}")

    elapsed = time.time() - t0
    log(f"  Base generation (seed={seed}) done in {elapsed:.1f}s")
    log_memory("post-gen-base")
    cleanup(model, tokenizer)
    return results


# ============================================================================
# Phase 2: Generate with uniform 1/N composition
# ============================================================================

def phase_generate_uniform(prompts_by_domain, seed):
    """Generate text with uniform 1/N adapter composition for a single seed."""
    log(f"\n[Phase 2] Generating with UNIFORM 1/N composition (seed={seed})...")
    t0 = time.time()

    skeleton = load_skeleton()
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model = apply_multi_adapter_uniform(model, skeleton)
    model.freeze()
    log_memory("post-load-uniform")

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
            score = compute_domain_score(
                generated, domain,
                ground_truth_response=prompt_data["response"] if domain == "math" else None
            )
            domain_results.append({
                "prompt": prompt_data["instruction"],
                "generated": generated,
                "word_count": word_count(generated),
                "keyword_density": keyword_density(generated, domain),
                "ngram_diversity": ngram_diversity(generated),
                "repetition": repetition_score(generated),
                "coherence": coherence_score(generated),
                "incoherent": is_incoherent(generated),
                "domain_score": score,
                "syntax_valid": code_syntax_valid(generated) if domain == "code" else None,
                "answer_correct": (
                    math_answer_correct(
                        extract_math_answer(generated),
                        extract_ground_truth_answer(prompt_data["response"])
                    ) if domain == "math" else None
                ),
            })
            if (i + 1) % 5 == 0:
                log(f"  {domain}: {i+1}/{len(prompts_by_domain[domain])} prompts done")

        results[domain] = domain_results
        avg_score = np.mean([r["domain_score"] for r in domain_results])
        log(f"  {domain}: avg_domain_score={avg_score:.3f}")

    elapsed = time.time() - t0
    log(f"  Uniform generation (seed={seed}) done in {elapsed:.1f}s")
    log_memory("post-gen-uniform")
    cleanup(model, tokenizer)
    del skeleton
    gc.collect()
    return results


# ============================================================================
# Phase 3: Generate with routed top-1 composition (Fix 1: single expert)
# ============================================================================

def phase_generate_routed(prompts_by_domain, seed):
    """Generate text with routed top-1 adapter composition for a single seed.

    Fix 1: Uses top-1 oracle routing (single expert, weight=1.0).
    No secondary expert. This removes the confound of arbitrary secondary pairing.
    """
    log(f"\n[Phase 3] Generating with ROUTED top-1 composition (seed={seed})...")
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
        # Fix 1: Top-1 routing - single expert only
        weights = get_oracle_routing_weights_top1(di, N_DOMAINS)
        set_all_routing_weights(model, weights)
        log(f"  {domain}: routing weights = {weights}")

        domain_results = []
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            formatted = format_prompt(prompt_data["instruction"])
            generated = generate_text(
                model, tokenizer, formatted,
                max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P
            )
            score = compute_domain_score(
                generated, domain,
                ground_truth_response=prompt_data["response"] if domain == "math" else None
            )
            domain_results.append({
                "prompt": prompt_data["instruction"],
                "generated": generated,
                "word_count": word_count(generated),
                "keyword_density": keyword_density(generated, domain),
                "ngram_diversity": ngram_diversity(generated),
                "repetition": repetition_score(generated),
                "coherence": coherence_score(generated),
                "incoherent": is_incoherent(generated),
                "domain_score": score,
                "syntax_valid": code_syntax_valid(generated) if domain == "code" else None,
                "answer_correct": (
                    math_answer_correct(
                        extract_math_answer(generated),
                        extract_ground_truth_answer(prompt_data["response"])
                    ) if domain == "math" else None
                ),
            })
            if (i + 1) % 5 == 0:
                log(f"  {domain}: {i+1}/{len(prompts_by_domain[domain])} prompts done")

        results[domain] = domain_results
        avg_score = np.mean([r["domain_score"] for r in domain_results])
        log(f"  {domain}: avg_domain_score={avg_score:.3f}")

    elapsed = time.time() - t0
    log(f"  Routed generation (seed={seed}) done in {elapsed:.1f}s")
    log_memory("post-gen-routed")
    cleanup(model, tokenizer)
    del skeleton
    gc.collect()
    return results


# ============================================================================
# Phase 4: Cross-PPL evaluation (Fix 2: DIAGNOSTIC ONLY)
# ============================================================================

def phase_cross_ppl(all_base, all_uniform, all_routed):
    """Evaluate generated text quality via cross-perplexity.

    Fix 2: Cross-PPL is DIAGNOSTIC ONLY. It is NOT used in the primary
    domain_score or kill criterion evaluation. It is computed for informational
    purposes and reported separately.

    Uses text from seed=42 (first seed) to avoid redundant computation.
    """
    log("\n[Phase 4] Computing cross-PPL (DIAGNOSTIC ONLY)...")
    t0 = time.time()

    # Use first seed's results for cross-PPL (diagnostic, not primary metric)
    base_results = all_base[0]
    uniform_results = all_uniform[0]
    routed_results = all_routed[0]

    skeleton = load_skeleton()
    cross_ppls = {"base": {}, "uniform": {}, "routed": {}}

    for di, domain in enumerate(DOMAINS):
        log(f"\n  Loading {domain} adapter for cross-PPL...")
        model, tokenizer = load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)
        model = apply_single_adapter(model, skeleton, di, domain)
        model.freeze()

        for config_name, config_results in [
            ("base", base_results),
            ("uniform", uniform_results),
            ("routed", routed_results),
        ]:
            total_loss = 0.0
            total_tokens = 0
            for result in config_results[domain]:
                text = result["generated"]
                if not text or len(text.strip()) < 5:
                    continue
                full_text = format_prompt(result["prompt"]) + text
                tokens = tokenizer.encode(full_text)
                tokens = tokens[:MAX_SEQ_LENGTH + 1]
                if len(tokens) < 2:
                    continue
                x = mx.array(tokens[:-1])[None, :]
                y = mx.array(tokens[1:])[None, :]
                logits = model(x)
                loss = nn.losses.cross_entropy(logits, y, reduction="sum")
                mx.eval(loss)
                total_loss += loss.item()
                total_tokens += y.size
                del logits, loss, x, y

            if total_tokens > 0:
                avg_loss = total_loss / total_tokens
                ppl = math.exp(min(avg_loss, 100))
            else:
                ppl = float("inf")
            cross_ppls[config_name][domain] = ppl
            log(f"    {config_name} {domain}: cross-PPL = {ppl:.2f}")

        cleanup(model, tokenizer)

    elapsed = time.time() - t0
    log(f"  Cross-PPL done in {elapsed:.1f}s")
    del skeleton
    gc.collect()
    return cross_ppls


# ============================================================================
# Analysis and scoring
# ============================================================================

def analyze_results(all_base, all_uniform, all_routed, cross_ppls):
    """Compute aggregate scores and kill criteria assessment across all seeds.

    Fix 2: primary metric is domain_score (domain-appropriate, no cross-PPL)
    Fix 4: aggregates across 3 seeds with mean and std
    Fix 5: same K1 criterion (>= 3/5 domains worse -> KILL)
    """
    log("\n[Analysis] Computing aggregate metrics across seeds...")

    n_seeds = len(all_base)
    analysis = {
        "per_domain": {},
        "aggregates": {},
        "kill_criteria": {},
        "generated_samples": {},
        "per_seed_details": {},
    }

    # Compute per-domain, per-seed scores
    domain_scores_by_seed = {domain: {"base": [], "uniform": [], "routed": []}
                             for domain in DOMAINS}

    for seed_idx in range(n_seeds):
        seed = SEEDS[seed_idx]
        seed_detail = {}
        for domain in DOMAINS:
            base_scores = [r["domain_score"] for r in all_base[seed_idx][domain]]
            uniform_scores = [r["domain_score"] for r in all_uniform[seed_idx][domain]]
            routed_scores = [r["domain_score"] for r in all_routed[seed_idx][domain]]

            base_mean = np.mean(base_scores)
            uniform_mean = np.mean(uniform_scores)
            routed_mean = np.mean(routed_scores)

            domain_scores_by_seed[domain]["base"].append(base_mean)
            domain_scores_by_seed[domain]["uniform"].append(uniform_mean)
            domain_scores_by_seed[domain]["routed"].append(routed_mean)

            seed_detail[domain] = {
                "base_mean": round(float(base_mean), 4),
                "uniform_mean": round(float(uniform_mean), 4),
                "routed_mean": round(float(routed_mean), 4),
            }
        analysis["per_seed_details"][f"seed_{seed}"] = seed_detail

    # Aggregate across seeds: mean and std
    routed_better_count = 0
    measurable_diff_count = 0
    incoherent_count = 0

    for domain in DOMAINS:
        base_mean = float(np.mean(domain_scores_by_seed[domain]["base"]))
        base_std = float(np.std(domain_scores_by_seed[domain]["base"]))
        uniform_mean = float(np.mean(domain_scores_by_seed[domain]["uniform"]))
        uniform_std = float(np.std(domain_scores_by_seed[domain]["uniform"]))
        routed_mean = float(np.mean(domain_scores_by_seed[domain]["routed"]))
        routed_std = float(np.std(domain_scores_by_seed[domain]["routed"]))

        # Routed wins if mean score is higher
        routed_wins = routed_mean > base_mean
        if routed_wins:
            routed_better_count += 1

        # Measurable difference: > 5% relative improvement
        if base_mean > 0:
            relative_improvement = (routed_mean - base_mean) / base_mean
        else:
            relative_improvement = 0
        if abs(relative_improvement) > 0.05:
            measurable_diff_count += 1

        # Incoherence check (across all seeds)
        all_routed_incoh = sum(
            sum(1 for r in all_routed[si][domain] if r["incoherent"])
            for si in range(n_seeds)
        )
        total_routed_samples = sum(len(all_routed[si][domain]) for si in range(n_seeds))
        if all_routed_incoh > total_routed_samples * 0.5:
            incoherent_count += 1

        # Domain-specific sub-metrics (aggregated across seeds)
        all_base_kw = [r["keyword_density"] for si in range(n_seeds) for r in all_base[si][domain]]
        all_uniform_kw = [r["keyword_density"] for si in range(n_seeds) for r in all_uniform[si][domain]]
        all_routed_kw = [r["keyword_density"] for si in range(n_seeds) for r in all_routed[si][domain]]

        domain_data = {
            "base": {
                "domain_score_mean": round(base_mean, 4),
                "domain_score_std": round(base_std, 4),
                "keyword_density_mean": round(float(np.mean(all_base_kw)), 4),
                "cross_ppl": round(cross_ppls["base"].get(domain, float("inf")), 2),
            },
            "uniform": {
                "domain_score_mean": round(uniform_mean, 4),
                "domain_score_std": round(uniform_std, 4),
                "keyword_density_mean": round(float(np.mean(all_uniform_kw)), 4),
                "cross_ppl": round(cross_ppls["uniform"].get(domain, float("inf")), 2),
            },
            "routed": {
                "domain_score_mean": round(routed_mean, 4),
                "domain_score_std": round(routed_std, 4),
                "keyword_density_mean": round(float(np.mean(all_routed_kw)), 4),
                "cross_ppl": round(cross_ppls["routed"].get(domain, float("inf")), 2),
            },
            "routed_wins": bool(routed_wins),
            "relative_improvement_pct": round(relative_improvement * 100, 2),
        }

        # Add domain-specific sub-metrics
        if domain == "code":
            for config_name, all_data in [("base", all_base), ("uniform", all_uniform), ("routed", all_routed)]:
                syntax_results = [r["syntax_valid"] for si in range(n_seeds) for r in all_data[si][domain] if r["syntax_valid"] is not None]
                domain_data[config_name]["syntax_valid_rate"] = round(float(np.mean(syntax_results)), 3) if syntax_results else 0.0

        if domain == "math":
            for config_name, all_data in [("base", all_base), ("uniform", all_uniform), ("routed", all_routed)]:
                answer_results = [r["answer_correct"] for si in range(n_seeds) for r in all_data[si][domain] if r["answer_correct"] is not None]
                domain_data[config_name]["answer_correct_rate"] = round(float(np.mean(answer_results)), 3) if answer_results else 0.0

        analysis["per_domain"][domain] = domain_data

        # Save sample from first seed
        analysis["generated_samples"][domain] = {
            "prompt": all_base[0][domain][0]["prompt"],
            "base": all_base[0][domain][0]["generated"][:500],
            "uniform": all_uniform[0][domain][0]["generated"][:500],
            "routed": all_routed[0][domain][0]["generated"][:500],
        }

    # Kill criteria assessment (Fix 5: same criterion)
    routed_worse_count = len(DOMAINS) - routed_better_count
    k1_kill = routed_worse_count >= 3
    k2_kill = measurable_diff_count == 0
    k3_kill = incoherent_count == len(DOMAINS)

    analysis["aggregates"] = {
        "routed_better_count": routed_better_count,
        "routed_worse_count": routed_worse_count,
        "measurable_diff_count": measurable_diff_count,
        "incoherent_domains": incoherent_count,
        "n_seeds": n_seeds,
        "seeds": SEEDS,
    }

    analysis["kill_criteria"] = {
        "k1": {
            "id": 272,
            "test": "Routed worse than base on >= 3/5 domains (domain-appropriate score, mean across seeds)",
            "value": f"{routed_worse_count}/5 domains worse",
            "result": "FAIL" if k1_kill else "PASS",
        },
        "k2": {
            "id": 273,
            "test": "No measurable difference (0/5 domains with >5% improvement)",
            "value": f"{measurable_diff_count}/5 domains with measurable diff",
            "result": "FAIL" if k2_kill else "PASS",
        },
        "k3": {
            "id": 274,
            "test": "All generated text incoherent",
            "value": f"{incoherent_count}/5 domains incoherent",
            "result": "FAIL" if k3_kill else "PASS",
        },
    }

    verdict = "KILLED" if (k1_kill or k2_kill or k3_kill) else "SUPPORTED"
    analysis["verdict"] = verdict

    return analysis


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("Generation Quality Test v2: Routed Top-1 Composition vs Base")
    log("Fixes: top-1 only, domain-appropriate metrics, 3 seeds, no XPPL primary")
    log("=" * 70)
    log_memory("start")

    # Verify adapters exist
    for domain in DOMAINS:
        adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
        if not adapter_path.exists():
            log(f"ERROR: Missing adapter for {domain} at {adapter_path}")
            log("Run exp_real_data_domain_experts first!")
            return

    # Extract prompts (with ground truth responses for math)
    log("\n[Setup] Extracting prompts from validation data...")
    prompts_by_domain = {}
    for domain in DOMAINS:
        prompts = extract_prompts_with_answers(domain, NUM_PROMPTS_PER_DOMAIN)
        prompts_by_domain[domain] = prompts
        log(f"  {domain}: {len(prompts)} prompts")

    # Run all 3 seeds (Fix 4)
    all_base = []
    all_uniform = []
    all_routed = []

    for seed_idx, seed in enumerate(SEEDS):
        log(f"\n{'='*70}")
        log(f"SEED {seed_idx+1}/{len(SEEDS)}: {seed}")
        log(f"{'='*70}")

        # Phase 1: Base generation
        base_results = phase_generate_base(prompts_by_domain, seed)
        all_base.append(base_results)

        # Phase 2: Uniform composition
        uniform_results = phase_generate_uniform(prompts_by_domain, seed)
        all_uniform.append(uniform_results)

        # Phase 3: Routed top-1 composition (Fix 1)
        routed_results = phase_generate_routed(prompts_by_domain, seed)
        all_routed.append(routed_results)

    # Phase 4: Cross-PPL evaluation (Fix 2: diagnostic only)
    cross_ppls = phase_cross_ppl(all_base, all_uniform, all_routed)

    # Analysis across all seeds
    analysis = analyze_results(all_base, all_uniform, all_routed, cross_ppls)

    # Final results
    results = {
        "experiment": "generation_quality_test_v2",
        "model": MODEL_ID,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "v2 (revised per adversarial review)",
        "fixes_applied": [
            "Fix 1: Top-1 routing only (single expert, weight=1.0)",
            "Fix 2: Cross-PPL diagnostic only, not in primary scoring",
            "Fix 3: Domain-appropriate metrics (code: syntax, math: correctness, others: reweighted)",
            "Fix 4: 3 seeds (42, 137, 2024) with mean+std",
            "Fix 5: Same K1 criterion (>= 3/5 domains worse -> KILL)",
            "Fix 6: XPPL normalization asymmetry documented in MATH.md",
        ],
        "config": {
            "num_prompts_per_domain": NUM_PROMPTS_PER_DOMAIN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seeds": SEEDS,
            "routing": "oracle_top1 (single expert, weight=1.0)",
        },
        "per_domain": analysis["per_domain"],
        "aggregates": analysis["aggregates"],
        "kill_criteria": analysis["kill_criteria"],
        "cross_ppls_diagnostic": cross_ppls,
        "verdict": analysis["verdict"],
        "generated_samples": analysis["generated_samples"],
        "per_seed_details": analysis["per_seed_details"],
        "total_time_s": round(time.time() - t0, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")

    # Print summary
    log("\n" + "=" * 70)
    log("SUMMARY (v2: domain-appropriate scores, mean across 3 seeds)")
    log("=" * 70)
    for domain in DOMAINS:
        d = analysis["per_domain"][domain]
        log(f"\n{domain}:")
        log(f"  Base:    score={d['base']['domain_score_mean']:.4f} +/- {d['base']['domain_score_std']:.4f}  kw={d['base']['keyword_density_mean']:.4f}  xppl={d['base']['cross_ppl']:.1f}")
        log(f"  Uniform: score={d['uniform']['domain_score_mean']:.4f} +/- {d['uniform']['domain_score_std']:.4f}  kw={d['uniform']['keyword_density_mean']:.4f}  xppl={d['uniform']['cross_ppl']:.1f}")
        log(f"  Routed:  score={d['routed']['domain_score_mean']:.4f} +/- {d['routed']['domain_score_std']:.4f}  kw={d['routed']['keyword_density_mean']:.4f}  xppl={d['routed']['cross_ppl']:.1f}")
        log(f"  Routed wins: {d['routed_wins']} ({d['relative_improvement_pct']:+.1f}%)")
        if domain == "code":
            log(f"  Syntax valid rate: base={d['base'].get('syntax_valid_rate', 'N/A')} uniform={d['uniform'].get('syntax_valid_rate', 'N/A')} routed={d['routed'].get('syntax_valid_rate', 'N/A')}")
        if domain == "math":
            log(f"  Answer correct rate: base={d['base'].get('answer_correct_rate', 'N/A')} uniform={d['uniform'].get('answer_correct_rate', 'N/A')} routed={d['routed'].get('answer_correct_rate', 'N/A')}")

    log(f"\nKill Criteria (Fix 5: same pre-registered K1):")
    for k, v in analysis["kill_criteria"].items():
        log(f"  {k} (id={v['id']}): {v['result']} -- {v['test']} ({v['value']})")

    log(f"\nVERDICT: {analysis['verdict']}")
    log(f"Total time: {results['total_time_s']:.1f}s ({len(SEEDS)} seeds)")

    # Show sample generations (first seed only)
    log("\n" + "=" * 70)
    log("SAMPLE GENERATIONS (seed=42, first prompt per domain)")
    log("=" * 70)
    for domain in DOMAINS:
        samples = analysis["generated_samples"][domain]
        log(f"\n--- {domain.upper()} ---")
        log(f"Prompt: {samples['prompt'][:200]}")
        log(f"\nBase:    {samples['base'][:300]}")
        log(f"\nUniform: {samples['uniform'][:300]}")
        log(f"\nRouted:  {samples['routed'][:300]}")


if __name__ == "__main__":
    main()
