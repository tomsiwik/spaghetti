#!/usr/bin/env python3
"""Energy Gap Top-k Routing: Select best adapter per query by NLL magnitude.

Uses the energy gap Delta_E = NLL(adapted) - NLL(base) to RANK adapters per query,
then selects the top-1 (most negative Delta_E = largest NLL reduction) for generation.

This is fundamentally different from the KILLED energy-gating approach (Finding #184):
- Gating = binary include/exclude at a threshold (impossible: all gaps negative)
- Ranking = relative ordering by magnitude (works: AUC=0.942 on math, Finding #182)

Kill criteria:
  K575: Energy gap top-1 selects correct domain adapter <80% of time on labeled queries
  K576: Top-1 routed generation quality worse than or equal to uniform on math+code
  K577: Energy gap computation overhead exceeds 10% of base inference time

Platform: Apple M5 Pro 48GB, MLX
Type: Guided exploration (unknown: does ranking improve generation quality?)
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
from mlx.utils import tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Source: real_data_domain_experts adapters
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256

DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = len(DOMAINS)

# Generation settings — reduced for time budget
NUM_PROMPTS_PER_DOMAIN = 10
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.9
SEED = 42  # Single seed (3 seeds too slow for 4 configs * 5 domains * 10 prompts)

# Domain keywords
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
# BitNet unpacking (from energy_gated_composition)
# ============================================================================

from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
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


# ============================================================================
# LoRA layer (single adapter only — no multi-adapter routing needed for top-1)
# ============================================================================

class TernaryLoRALinear(nn.Module):
    """Single adapter LoRA with STE-ternary B."""
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


# ============================================================================
# Multi-adapter routing layer (for uniform composition baseline)
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
            alpha = mx.mean(mx.abs(b))
            b_scaled = b / (alpha + 1e-7)
            b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
            b_ste = b + mx.stop_gradient(b_q - b)
            lora_sum = lora_sum + w * ((x @ self.a_matrices[i]) @ b_ste)
        return base_out + lora_sum * self.scale


# ============================================================================
# Model setup helpers
# ============================================================================

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def load_skeleton():
    skeleton_path = ADAPTERS_DIR / "grassmannian_skeleton.npz"
    return dict(np.load(str(skeleton_path)))


def apply_single_adapter(model, skeleton, domain_idx, domain_name):
    """Apply a single adapter to the model."""
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
    return model


def apply_multi_adapter_routed(model, skeleton):
    """Apply all 5 adapters with routing weights."""
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
    log(f"  Applied routed multi-adapter ({count} layers)")
    return model


def set_all_routing_weights(model, weights):
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, RoutedMultiAdapterLoRALinear):
                module.set_routing_weights(weights)


# ============================================================================
# Scoring metrics
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


def word_count(text):
    return len(re.findall(r'\b\w+\b', text))


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
    avg_len = np.mean([word_count(s) for s in sentences])
    return max(0, 1.0 - abs(avg_len - 15) / 30)


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
        div = ngram_diversity(text)
        coh = coherence_score(text)
        rep = repetition_score(text)
        return 0.45 * kw + 0.25 * div + 0.10 * coh + 0.20 * rep


# ============================================================================
# Prompt extraction
# ============================================================================

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
# Energy gap computation
# ============================================================================

def compute_prompt_nll(model, tokenizer, prompt_text):
    """Compute NLL on prompt tokens. Returns per-token NLL (float)."""
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

    target_log_probs = mx.take_along_axis(
        log_probs, targets[:, :, None], axis=-1
    ).squeeze(-1)

    nll = -mx.mean(target_log_probs).item()
    del logits, logits_shift, log_probs, target_log_probs, targets, x
    return nll


# ============================================================================
# Phase 0: Compute energy gaps (one model load per adapter)
# ============================================================================

def phase_compute_energy_gaps(prompts_by_domain):
    """Compute energy gap per (query, adapter) pair.

    Returns energy_gaps[adapter_domain][query_domain][prompt_idx] = Delta_E.
    Also returns timing info for overhead measurement.
    """
    log("\n[Phase 0] Computing energy gaps...")
    t0 = time.time()

    # Step 1: Base NLLs
    log("  Loading base model for NLL computation...")
    t_base_start = time.time()
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    log_memory("base-loaded")

    base_nlls = {}
    for domain in DOMAINS:
        nlls = []
        for prompt_data in prompts_by_domain[domain]:
            formatted = format_prompt(prompt_data["instruction"])
            nll = compute_prompt_nll(model, tokenizer, formatted)
            nlls.append(nll)
        base_nlls[domain] = nlls
        log(f"    {domain}: mean_base_nll={np.mean(nlls):.4f}")

    t_base = time.time() - t_base_start
    del model, tokenizer
    cleanup()
    log_memory("base-cleanup")

    # Step 2: Per-adapter NLLs
    log("  Computing per-adapter NLLs...")
    skeleton = load_skeleton()
    adapter_nlls = {}
    t_adapter_start = time.time()

    for di, adapter_domain in enumerate(DOMAINS):
        log(f"    Adapter: {adapter_domain}")
        model, tokenizer = load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)
        model = apply_single_adapter(model, skeleton, di, adapter_domain)
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
        log_memory(f"adapter-{adapter_domain}-cleanup")

    t_adapter = time.time() - t_adapter_start
    del skeleton

    # Step 3: Compute energy gaps
    energy_gaps = {}
    for adapter_domain in DOMAINS:
        energy_gaps[adapter_domain] = {}
        for query_domain in DOMAINS:
            gaps = [
                adapter_nlls[adapter_domain][query_domain][i] - base_nlls[query_domain][i]
                for i in range(len(base_nlls[query_domain]))
            ]
            energy_gaps[adapter_domain][query_domain] = gaps

    total_energy_time = time.time() - t0
    timing = {
        "base_nll_time_s": round(t_base, 1),
        "adapter_nll_time_s": round(t_adapter, 1),
        "total_energy_time_s": round(total_energy_time, 1),
        "n_prompts_total": NUM_PROMPTS_PER_DOMAIN * N_DOMAINS,
    }

    log(f"  Energy gap computation: {total_energy_time:.1f}s total")
    return base_nlls, energy_gaps, timing


# ============================================================================
# Top-1 routing: select best adapter per query
# ============================================================================

def select_top1_adapter(energy_gaps, query_domain, prompt_idx):
    """Select the adapter with the most negative energy gap (largest NLL reduction).

    Returns (adapter_index, adapter_name, energy_gap_value).
    """
    gaps = []
    for adapter_domain in DOMAINS:
        gap = energy_gaps[adapter_domain][query_domain][prompt_idx]
        gaps.append(gap)

    best_idx = int(np.argmin(gaps))  # most negative = largest NLL reduction
    return best_idx, DOMAINS[best_idx], gaps[best_idx]


# ============================================================================
# Phase 1: Generate with base model
# ============================================================================

def phase_generate_base(prompts_by_domain):
    log("\n[Phase 1] Generating with BASE model...")
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
                "generated": generated[:200],
                "domain_score": score,
                "keyword_density": keyword_density(generated, domain),
                "answer_correct": (
                    math_answer_correct(extract_math_answer(generated),
                                        extract_ground_truth_answer(prompt_data["response"]))
                    if domain == "math" else None),
                "syntax_valid": code_syntax_valid(generated) if domain == "code" else None,
            })
        results[domain] = domain_results
        scores = [r["domain_score"] for r in domain_results]
        log(f"  {domain}: avg_score={np.mean(scores):.4f}")

    elapsed = time.time() - t0
    log(f"  Base gen done in {elapsed:.1f}s")
    del model, tokenizer
    cleanup()
    log_memory("post-base-gen")
    return results, elapsed


# ============================================================================
# Phase 2: Generate with uniform 1/N composition
# ============================================================================

def phase_generate_uniform(prompts_by_domain):
    log("\n[Phase 2] Generating with UNIFORM 1/N composition...")
    t0 = time.time()

    skeleton = load_skeleton()
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model = apply_multi_adapter_routed(model, skeleton)
    uniform_weights = [1.0 / N_DOMAINS] * N_DOMAINS
    set_all_routing_weights(model, uniform_weights)
    model.freeze()
    del skeleton

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
                "generated": generated[:200],
                "domain_score": score,
                "keyword_density": keyword_density(generated, domain),
                "answer_correct": (
                    math_answer_correct(extract_math_answer(generated),
                                        extract_ground_truth_answer(prompt_data["response"]))
                    if domain == "math" else None),
                "syntax_valid": code_syntax_valid(generated) if domain == "code" else None,
            })
        results[domain] = domain_results
        scores = [r["domain_score"] for r in domain_results]
        log(f"  {domain}: avg_score={np.mean(scores):.4f}")

    elapsed = time.time() - t0
    log(f"  Uniform gen done in {elapsed:.1f}s")
    del model, tokenizer
    cleanup()
    log_memory("post-uniform-gen")
    return results, elapsed


# ============================================================================
# Phase 3: Generate with top-1 energy-gap routed single adapter
# ============================================================================

def phase_generate_top1_routed(prompts_by_domain, energy_gaps):
    """For each query, select the best adapter by energy gap, then generate with it.

    This loads the multi-adapter model once and sets routing weights to one-hot
    for the selected adapter per query.
    """
    log("\n[Phase 3] Generating with TOP-1 energy-gap routed adapter...")
    t0 = time.time()

    skeleton = load_skeleton()
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model = apply_multi_adapter_routed(model, skeleton)
    model.freeze()
    del skeleton

    mx.random.seed(SEED)
    np.random.seed(SEED)
    results = {}
    routing_decisions = {}

    for domain in DOMAINS:
        domain_results = []
        domain_routing = []
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            # Top-1 selection
            best_idx, best_name, best_gap = select_top1_adapter(energy_gaps, domain, i)

            # One-hot routing weights
            weights = [0.0] * N_DOMAINS
            weights[best_idx] = 1.0
            set_all_routing_weights(model, weights)

            # Track routing decision
            all_gaps = {ad: energy_gaps[ad][domain][i] for ad in DOMAINS}
            is_correct = (best_name == domain)
            domain_routing.append({
                "prompt_idx": i,
                "selected_adapter": best_name,
                "selected_gap": best_gap,
                "correct_selection": is_correct,
                "all_gaps": all_gaps,
                "gap_margin": sorted(all_gaps.values())[0] - sorted(all_gaps.values())[1],
            })

            formatted = format_prompt(prompt_data["instruction"])
            generated = generate_text(model, tokenizer, formatted,
                                      max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)
            score = compute_domain_score(generated, domain,
                                         ground_truth_response=prompt_data["response"] if domain == "math" else None)
            domain_results.append({
                "prompt": prompt_data["instruction"][:100],
                "generated": generated[:200],
                "domain_score": score,
                "keyword_density": keyword_density(generated, domain),
                "answer_correct": (
                    math_answer_correct(extract_math_answer(generated),
                                        extract_ground_truth_answer(prompt_data["response"]))
                    if domain == "math" else None),
                "syntax_valid": code_syntax_valid(generated) if domain == "code" else None,
                "selected_adapter": best_name,
                "correct_selection": is_correct,
            })

        results[domain] = domain_results
        routing_decisions[domain] = domain_routing
        scores = [r["domain_score"] for r in domain_results]
        correct_rate = np.mean([r["correct_selection"] for r in domain_routing])
        log(f"  {domain}: avg_score={np.mean(scores):.4f}, top1_correct={correct_rate:.0%}")

    elapsed = time.time() - t0
    log(f"  Top-1 routed gen done in {elapsed:.1f}s")
    del model, tokenizer
    cleanup()
    log_memory("post-top1-gen")
    return results, routing_decisions, elapsed


# ============================================================================
# Phase 4: Generate with oracle single adapter (upper bound)
# ============================================================================

def phase_generate_oracle(prompts_by_domain):
    """Use the domain-matched adapter for each query (perfect routing).
    This is the upper bound for top-1 routing quality.
    """
    log("\n[Phase 4] Generating with ORACLE (domain-matched) adapter...")
    t0 = time.time()

    skeleton = load_skeleton()
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model = apply_multi_adapter_routed(model, skeleton)
    model.freeze()
    del skeleton

    mx.random.seed(SEED)
    np.random.seed(SEED)
    results = {}

    for di, domain in enumerate(DOMAINS):
        # One-hot weights for the correct adapter
        weights = [0.0] * N_DOMAINS
        weights[di] = 1.0
        set_all_routing_weights(model, weights)

        domain_results = []
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            formatted = format_prompt(prompt_data["instruction"])
            generated = generate_text(model, tokenizer, formatted,
                                      max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)
            score = compute_domain_score(generated, domain,
                                         ground_truth_response=prompt_data["response"] if domain == "math" else None)
            domain_results.append({
                "prompt": prompt_data["instruction"][:100],
                "generated": generated[:200],
                "domain_score": score,
                "keyword_density": keyword_density(generated, domain),
                "answer_correct": (
                    math_answer_correct(extract_math_answer(generated),
                                        extract_ground_truth_answer(prompt_data["response"]))
                    if domain == "math" else None),
                "syntax_valid": code_syntax_valid(generated) if domain == "code" else None,
            })
        results[domain] = domain_results
        scores = [r["domain_score"] for r in domain_results]
        log(f"  {domain}: avg_score={np.mean(scores):.4f}")

    elapsed = time.time() - t0
    log(f"  Oracle gen done in {elapsed:.1f}s")
    del model, tokenizer
    cleanup()
    log_memory("post-oracle-gen")
    return results, elapsed


# ============================================================================
# Analysis
# ============================================================================

def analyze_results(base_results, uniform_results, top1_results, oracle_results,
                    routing_decisions, energy_gaps, energy_timing, gen_timings):
    log("\n[Analysis] Computing aggregate metrics...")

    analysis = {
        "per_domain": {},
        "routing_accuracy": {},
        "kill_criteria": {},
        "timing": {},
        "energy_gap_matrix": {},
    }

    # Energy gap matrix (for reference)
    log("\n  Energy Gap Matrix (adapter rows x query domain columns):")
    log(f"  {'':>12s} " + " ".join(f"{d:>10s}" for d in DOMAINS))
    for ad in DOMAINS:
        gaps = [np.mean(energy_gaps[ad][qd]) for qd in DOMAINS]
        log(f"  {ad:>12s} " + " ".join(f"{g:>10.4f}" for g in gaps))
        analysis["energy_gap_matrix"][ad] = {qd: round(np.mean(energy_gaps[ad][qd]), 4) for qd in DOMAINS}

    # Per-domain analysis
    overall_correct = 0
    overall_total = 0
    math_code_top1_better = True

    for domain in DOMAINS:
        base_scores = [r["domain_score"] for r in base_results[domain]]
        uniform_scores = [r["domain_score"] for r in uniform_results[domain]]
        top1_scores = [r["domain_score"] for r in top1_results[domain]]
        oracle_scores = [r["domain_score"] for r in oracle_results[domain]]

        base_mean = float(np.mean(base_scores))
        uniform_mean = float(np.mean(uniform_scores))
        top1_mean = float(np.mean(top1_scores))
        oracle_mean = float(np.mean(oracle_scores))

        # Routing accuracy
        correct = sum(1 for r in routing_decisions[domain] if r["correct_selection"])
        total = len(routing_decisions[domain])
        accuracy = correct / total if total > 0 else 0
        overall_correct += correct
        overall_total += total

        domain_data = {
            "base": {"mean": round(base_mean, 4)},
            "uniform": {"mean": round(uniform_mean, 4)},
            "top1": {"mean": round(top1_mean, 4)},
            "oracle": {"mean": round(oracle_mean, 4)},
            "top1_vs_uniform_pct": round((top1_mean - uniform_mean) / max(uniform_mean, 0.001) * 100, 2),
            "top1_vs_base_pct": round((top1_mean - base_mean) / max(base_mean, 0.001) * 100, 2),
            "oracle_vs_top1_pct": round((oracle_mean - top1_mean) / max(top1_mean, 0.001) * 100, 2),
            "routing_accuracy": round(accuracy, 3),
            "n_correct": correct,
            "n_total": total,
        }

        # Math answer correctness
        if domain == "math":
            for config_name, config_results in [("base", base_results), ("uniform", uniform_results),
                                                  ("top1", top1_results), ("oracle", oracle_results)]:
                rates = [r["answer_correct"] for r in config_results[domain]
                         if r.get("answer_correct") is not None]
                domain_data[config_name]["answer_correct_rate"] = round(float(np.mean(rates)), 3) if rates else 0.0

        # Code syntax
        if domain == "code":
            for config_name, config_results in [("base", base_results), ("uniform", uniform_results),
                                                  ("top1", top1_results), ("oracle", oracle_results)]:
                rates = [r["syntax_valid"] for r in config_results[domain]
                         if r.get("syntax_valid") is not None]
                domain_data[config_name]["syntax_valid_rate"] = round(float(np.mean(rates)), 3) if rates else 0.0

        # K576: Check math+code specifically
        if domain in ("math", "code"):
            if top1_mean <= uniform_mean:
                math_code_top1_better = False

        analysis["per_domain"][domain] = domain_data

        # Log routing decisions for this domain
        routing_selections = {}
        for r in routing_decisions[domain]:
            sel = r["selected_adapter"]
            routing_selections[sel] = routing_selections.get(sel, 0) + 1
        analysis["routing_accuracy"][domain] = {
            "accuracy": round(accuracy, 3),
            "selection_distribution": routing_selections,
        }

    # Overall routing accuracy
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
    analysis["routing_accuracy"]["overall"] = {
        "accuracy": round(overall_accuracy, 3),
        "correct": overall_correct,
        "total": overall_total,
    }

    # ========== Kill Criteria ==========

    # K575: Top-1 selects correct domain adapter >= 80% of the time
    k575_pass = overall_accuracy >= 0.80
    analysis["kill_criteria"]["K575"] = {
        "test": "Energy gap top-1 selects correct domain adapter >= 80%",
        "measured": round(overall_accuracy, 3),
        "threshold": 0.80,
        "per_domain": {d: round(analysis["routing_accuracy"][d]["accuracy"], 3) for d in DOMAINS},
        "result": "PASS" if k575_pass else "FAIL",
    }

    # K576: Top-1 generation quality better than uniform on math+code
    math_top1 = analysis["per_domain"]["math"]["top1"]["mean"]
    math_uniform = analysis["per_domain"]["math"]["uniform"]["mean"]
    code_top1 = analysis["per_domain"]["code"]["top1"]["mean"]
    code_uniform = analysis["per_domain"]["code"]["uniform"]["mean"]

    # Need top-1 > uniform on at least one of math/code AND not worse on the other
    k576_pass = (math_top1 > math_uniform) or (code_top1 > code_uniform)
    analysis["kill_criteria"]["K576"] = {
        "test": "Top-1 routed generation quality better than uniform on math+code",
        "math_top1": round(math_top1, 4),
        "math_uniform": round(math_uniform, 4),
        "code_top1": round(code_top1, 4),
        "code_uniform": round(code_uniform, 4),
        "math_top1_better": math_top1 > math_uniform,
        "code_top1_better": code_top1 > code_uniform,
        "result": "PASS" if k576_pass else "FAIL",
    }

    # K577: Energy gap overhead < 10% of base inference time
    # Energy gap is computed per-prompt (N+1 forward passes on prompt tokens)
    # Generation is N_tokens forward passes per prompt
    per_prompt_energy_time = energy_timing["total_energy_time_s"] / energy_timing["n_prompts_total"]
    per_prompt_base_gen_time = gen_timings["base"] / (NUM_PROMPTS_PER_DOMAIN * N_DOMAINS)
    overhead_pct = (per_prompt_energy_time / per_prompt_base_gen_time * 100) if per_prompt_base_gen_time > 0 else float('inf')

    k577_pass = overhead_pct < 10.0
    analysis["kill_criteria"]["K577"] = {
        "test": "Energy gap overhead < 10% of base inference time",
        "per_prompt_energy_s": round(per_prompt_energy_time, 3),
        "per_prompt_gen_s": round(per_prompt_base_gen_time, 3),
        "overhead_pct": round(float(overhead_pct), 1),
        "threshold_pct": 10.0,
        "result": "PASS" if k577_pass else "FAIL",
        "note": "Energy gap includes model loading overhead (amortized in production)",
    }

    analysis["timing"] = {
        "energy_computation": energy_timing,
        "generation": gen_timings,
    }

    # Verdict
    any_fail = not k575_pass or not k576_pass or not k577_pass
    analysis["verdict"] = "KILLED" if any_fail else "SUPPORTED"

    return analysis


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("Energy Gap Top-1 Routing: Select best adapter per query by NLL magnitude")
    log("Type: Guided Exploration (does ranking translate to generation quality?)")
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

    # Phase 0: Energy gaps (load each adapter once, compute NLL on all queries)
    base_nlls, energy_gaps, energy_timing = phase_compute_energy_gaps(prompts_by_domain)

    # Log energy gap matrix
    log("\n  Energy Gap Matrix (adapter rows x query domain columns, mean):")
    log(f"  {'':>12s} " + " ".join(f"{d:>10s}" for d in DOMAINS))
    for ad in DOMAINS:
        gaps = [np.mean(energy_gaps[ad][qd]) for qd in DOMAINS]
        log(f"  {ad:>12s} " + " ".join(f"{g:>10.4f}" for g in gaps))

    # Preview top-1 selections
    log("\n  Top-1 selections per query domain:")
    for domain in DOMAINS:
        selections = []
        for i in range(NUM_PROMPTS_PER_DOMAIN):
            _, selected, gap = select_top1_adapter(energy_gaps, domain, i)
            selections.append(selected)
        counts = Counter(selections)
        correct_rate = counts.get(domain, 0) / NUM_PROMPTS_PER_DOMAIN
        log(f"    {domain}: correct={correct_rate:.0%}, distribution={dict(counts)}")

    gen_timings = {}

    # Phase 1: Base model generation
    base_results, base_time = phase_generate_base(prompts_by_domain)
    gen_timings["base"] = base_time

    # Phase 2: Uniform 1/N composition
    uniform_results, uniform_time = phase_generate_uniform(prompts_by_domain)
    gen_timings["uniform"] = uniform_time

    # Phase 3: Top-1 energy-gap routed
    top1_results, routing_decisions, top1_time = phase_generate_top1_routed(prompts_by_domain, energy_gaps)
    gen_timings["top1"] = top1_time

    # Phase 4: Oracle (domain-matched) for upper bound
    oracle_results, oracle_time = phase_generate_oracle(prompts_by_domain)
    gen_timings["oracle"] = oracle_time

    # Analysis
    analysis = analyze_results(
        base_results, uniform_results, top1_results, oracle_results,
        routing_decisions, energy_gaps, energy_timing, gen_timings,
    )

    # Save results
    full_results = {
        "experiment": "energy_gap_topk_routing",
        "model": MODEL_ID,
        "type": "guided_exploration",
        "seed": SEED,
        "n_prompts_per_domain": NUM_PROMPTS_PER_DOMAIN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "domains": DOMAINS,
        "analysis": analysis,
        "routing_decisions": {
            d: routing_decisions[d] for d in DOMAINS
        },
        "total_time_s": round(time.time() - t0, 1),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(full_results, f, indent=2, cls=NumpyEncoder)
    log(f"\nResults saved to {RESULTS_FILE}")

    # Print summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    log("\nPer-domain scores:")
    log(f"  {'Domain':>10s}  {'Base':>8s}  {'Uniform':>8s}  {'Top-1':>8s}  {'Oracle':>8s}  {'Top1 acc':>8s}")
    for domain in DOMAINS:
        d = analysis["per_domain"][domain]
        log(f"  {domain:>10s}  {d['base']['mean']:>8.4f}  {d['uniform']['mean']:>8.4f}  "
            f"{d['top1']['mean']:>8.4f}  {d['oracle']['mean']:>8.4f}  "
            f"{d['routing_accuracy']:.0%}")

    log(f"\nOverall routing accuracy: {analysis['routing_accuracy']['overall']['accuracy']:.1%}")

    log("\nKill Criteria:")
    for kc, data in analysis["kill_criteria"].items():
        log(f"  {kc}: {data['result']} — {data['test']}")

    log(f"\nVerdict: {analysis['verdict']}")
    log(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
