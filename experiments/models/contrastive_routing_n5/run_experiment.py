#!/usr/bin/env python3
"""Contrastive retrieval routing for SFT adapters (exp_contrastive_routing_n5).

Replaces energy gap (NLL-based) routing with text-classification-based routing.
Energy gap fails at 36% accuracy because code adapter universally reduces NLL.
Text classification routes on input features, independent of adapter NLL.

Kill criteria:
  K605: Contrastive routing accuracy >= 70% on 5 SFT domains (vs 36% energy gap)
  K606: Routed composition improves math correctness >= 60%
  K607: Routed composition does not degrade prose domains vs base (0/5 worse)

Platform: Apple M5 Pro 48GB, MLX
Type: Guided exploration (proven framework, unknown: domain separability at N=5)
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
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Source data and adapters from prior experiments
SFT_DIR = EXPERIMENT_DIR.parent / "bitnet_sft_generation_v3"
SFT_ADAPTERS_DIR = SFT_DIR / "sft_adapters"
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
DATA_DIR = SOURCE_DIR / "data"
NTP_ADAPTERS_DIR = SOURCE_DIR / "adapters"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = len(DOMAINS)

# Generation settings
NUM_PROMPTS_PER_DOMAIN = 10
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0

RESPONSE_MARKER = "### Response:\n"

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
# Model utilities (from bitnet_sft_generation_v3)
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
# TernaryLoRA layer (from bitnet_sft_generation_v3)
# ============================================================================

class TernaryLoRALinear(nn.Module):
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
# Model setup helpers (Grassmannian skeleton)
# ============================================================================

def load_skeleton():
    skeleton_path = NTP_ADAPTERS_DIR / "grassmannian_skeleton.npz"
    return dict(np.load(str(skeleton_path)))


def apply_lora_with_skeleton(model, skeleton, domain_idx):
    n_layers = len(model.model.layers)
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
            a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16) if skey in skeleton else None
            lora = TernaryLoRALinear(module, rank=LORA_RANK, scale=LORA_SCALE, a_init=a_mx)
            lora_updates.append((key, lora))
            count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    mx.eval(model.parameters())
    return model


def apply_single_adapter_from_file(model, skeleton, domain_idx, adapter_path):
    model = apply_lora_with_skeleton(model, skeleton, domain_idx)
    adapter_params = dict(mx.load(str(adapter_path)))
    model.update(tree_unflatten(list(adapter_params.items())))
    mx.eval(model.parameters())
    return model


# ============================================================================
# Data loading
# ============================================================================

def load_domain_instructions(domain, split="train", max_samples=400):
    """Load instruction text from domain data."""
    path = DATA_DIR / domain / f"{split}.jsonl"
    instructions = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            text = json.loads(line)["text"]
            if "### Instruction:" in text:
                instruction = text.split("### Instruction:")[1].split("### Response:")[0].strip()
                instructions.append(instruction)
    return instructions


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
# Scoring metrics (execution-based, from bitnet_sft_generation_v3)
# ============================================================================

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


def coherence_score(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    avg_len = np.mean([len(re.findall(r'\b\w+\b', s)) for s in sentences])
    return max(0, 1.0 - abs(avg_len - 15) / 30)


def repetition_score(text):
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


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
    matches = re.findall(r'####\s*([\d,]+(?:\.\d+)?)', text)
    if matches:
        return float(matches[-1].replace(',', ''))
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
    m = re.search(r'(?:is|=)\s*\$?([\d,]+(?:\.\d+)?)\s*$', response_text.strip(), re.IGNORECASE)
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
        return 0.6 * syntax_ok + 0.4 * kw
    elif domain == "math":
        gen_answer = extract_math_answer(text)
        gt_answer = None
        if ground_truth_response:
            gt_answer = extract_ground_truth_answer(ground_truth_response)
        correct = 1.0 if math_answer_correct(gen_answer, gt_answer) else 0.0
        kw = keyword_density(text, domain)
        return 0.7 * correct + 0.3 * kw
    else:
        kw = keyword_density(text, domain)
        div = ngram_diversity(text)
        coh = coherence_score(text)
        rep = repetition_score(text)
        return 0.45 * kw + 0.25 * div + 0.10 * coh + 0.20 * rep


# ============================================================================
# Phase 1: Train text classifier for routing (TF-IDF + logistic regression)
# ============================================================================

def phase_train_classifier():
    """Train TF-IDF + logistic regression classifier on domain instruction text.

    This is the contrastive routing replacement for energy gap routing.
    The classifier maps input text -> domain label, independent of any adapter NLL.
    """
    log("\n" + "=" * 70)
    log("PHASE 1: TRAIN TEXT CLASSIFIER FOR ROUTING")
    log("=" * 70)
    t0 = time.time()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # Load training data: instructions from all 5 domains
    train_texts = []
    train_labels = []
    for di, domain in enumerate(DOMAINS):
        instructions = load_domain_instructions(domain, split="train", max_samples=400)
        train_texts.extend(instructions)
        train_labels.extend([di] * len(instructions))
        log(f"  {domain}: {len(instructions)} train instructions")

    # Load validation data for cross-validation
    val_texts = []
    val_labels = []
    for di, domain in enumerate(DOMAINS):
        instructions = load_domain_instructions(domain, split="valid", max_samples=50)
        val_texts.extend(instructions)
        val_labels.extend([di] * len(instructions))
        log(f"  {domain}: {len(instructions)} val instructions")

    log(f"  Total: {len(train_texts)} train, {len(val_texts)} val")

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Unigrams + bigrams
        stop_words="english",
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    log(f"  TF-IDF features: {X_train.shape[1]}")

    # Train logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=SEED,
    )
    clf.fit(X_train, y_train)

    # Evaluate on training set (sanity check)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    log(f"  Train accuracy: {train_acc:.4f}")

    # Evaluate on validation set
    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    log(f"  Val accuracy: {val_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_val, val_preds)
    log(f"\n  Confusion matrix (rows=true, cols=pred):")
    header = "         " + "  ".join(f"{d[:5]:>5}" for d in DOMAINS)
    log(f"  {header}")
    for i, domain in enumerate(DOMAINS):
        row = "  ".join(f"{cm[i, j]:>5}" for j in range(N_DOMAINS))
        log(f"  {domain[:5]:>5}:  {row}")

    # Per-class accuracy
    per_class_acc = {}
    for di, domain in enumerate(DOMAINS):
        mask = y_val == di
        if mask.sum() > 0:
            acc = accuracy_score(y_val[mask], val_preds[mask])
            per_class_acc[domain] = acc
            log(f"  {domain}: {acc:.2%} ({int(mask.sum())} samples)")

    # Classification report
    report = classification_report(y_val, val_preds, target_names=DOMAINS, output_dict=True)

    elapsed = time.time() - t0
    log(f"\n  Classifier training: {elapsed:.1f}s")

    results = {
        "train_accuracy": round(float(train_acc), 4),
        "val_accuracy": round(float(val_acc), 4),
        "per_class_accuracy": {k: round(v, 4) for k, v in per_class_acc.items()},
        "confusion_matrix": cm.tolist(),
        "n_features": int(X_train.shape[1]),
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "time_s": round(elapsed, 1),
    }

    return vectorizer, clf, results


# ============================================================================
# Phase 2: Evaluate routing on test prompts (same prompts as energy gap exp)
# ============================================================================

def phase_evaluate_routing(vectorizer, clf, prompts_by_domain):
    """Route test prompts using text classifier and compare to energy gap."""
    log("\n" + "=" * 70)
    log("PHASE 2: EVALUATE ROUTING ACCURACY ON TEST PROMPTS")
    log("=" * 70)

    routing_decisions = {}
    total_correct = 0
    total_queries = 0

    for di, domain in enumerate(DOMAINS):
        domain_correct = 0
        decisions = []
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            instruction = prompt_data["instruction"]
            # Classify instruction text
            X = vectorizer.transform([instruction])
            pred_idx = int(clf.predict(X)[0])
            pred_domain = DOMAINS[pred_idx]
            probs = clf.predict_proba(X)[0]

            correct = pred_idx == di
            if correct:
                domain_correct += 1
                total_correct += 1
            total_queries += 1

            decisions.append({
                "instruction": instruction[:100],
                "true_domain": domain,
                "pred_domain": pred_domain,
                "correct": correct,
                "confidence": round(float(probs[pred_idx]), 4),
                "all_probs": {DOMAINS[j]: round(float(probs[j]), 4) for j in range(N_DOMAINS)},
            })

        accuracy = domain_correct / len(prompts_by_domain[domain])
        routing_decisions[domain] = {
            "accuracy": round(accuracy, 4),
            "correct": domain_correct,
            "total": len(prompts_by_domain[domain]),
            "decisions": decisions,
        }
        log(f"  {domain}: {accuracy:.0%} ({domain_correct}/{len(prompts_by_domain[domain])})")

    overall_accuracy = total_correct / total_queries
    log(f"\n  Overall routing accuracy: {overall_accuracy:.0%} ({total_correct}/{total_queries})")
    log(f"  Energy gap baseline: 36% (18/50)")
    log(f"  Improvement: {overall_accuracy:.0%} vs 36%")

    return {
        "overall_accuracy": round(overall_accuracy, 4),
        "total_correct": total_correct,
        "total_queries": total_queries,
        "per_domain": routing_decisions,
        "energy_gap_baseline": 0.36,
    }


# ============================================================================
# Phase 3: Generate with base model (no adapters)
# ============================================================================

def generate_text(model, tokenizer, prompt, max_tokens=128, temperature=0.0):
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


def evaluate_generation(model, tokenizer, prompts_by_domain, label=""):
    results = {}
    for domain in DOMAINS:
        domain_results = []
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            formatted = format_prompt(prompt_data["instruction"])
            generated = generate_text(model, tokenizer, formatted,
                                      max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
            score = compute_domain_score(
                generated, domain,
                ground_truth_response=prompt_data["response"] if domain == "math" else None
            )
            result = {
                "prompt": prompt_data["instruction"][:100],
                "generated": generated[:300],
                "domain_score": score,
                "keyword_density": keyword_density(generated, domain),
            }
            if domain == "math":
                gen_ans = extract_math_answer(generated)
                gt_ans = extract_ground_truth_answer(prompt_data["response"])
                result["answer_correct"] = math_answer_correct(gen_ans, gt_ans)
                result["gen_answer"] = gen_ans
                result["gt_answer"] = gt_ans
            elif domain == "code":
                result["syntax_valid"] = code_syntax_valid(generated)
            domain_results.append(result)

        results[domain] = domain_results
        scores = [r["domain_score"] for r in domain_results]
        log(f"  [{label}] {domain}: avg_score={np.mean(scores):.4f}")

    return results


def phase_generate_base(prompts_by_domain):
    """Generate with base model (no adapters)."""
    log("\n" + "=" * 70)
    log("PHASE 3: GENERATE WITH BASE MODEL")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()

    mx.random.seed(SEED)
    np.random.seed(SEED)
    results = evaluate_generation(model, tokenizer, prompts_by_domain, label="BASE")

    elapsed = time.time() - t0
    del model, tokenizer
    cleanup()
    log_memory("post-base-gen")
    return results, elapsed


# ============================================================================
# Phase 4: Generate with classifier-routed adapters
# ============================================================================

def phase_generate_routed(prompts_by_domain, vectorizer, clf):
    """Generate with classifier-routed SFT adapters."""
    log("\n" + "=" * 70)
    log("PHASE 4: GENERATE WITH CLASSIFIER-ROUTED SFT ADAPTERS")
    log("=" * 70)
    t0 = time.time()

    skeleton = load_skeleton()

    mx.random.seed(SEED)
    np.random.seed(SEED)

    # Route all prompts using classifier
    adapter_queries = {d: [] for d in DOMAINS}
    for domain in DOMAINS:
        for i, prompt_data in enumerate(prompts_by_domain[domain]):
            X = vectorizer.transform([prompt_data["instruction"]])
            pred_idx = int(clf.predict(X)[0])
            pred_domain = DOMAINS[pred_idx]
            adapter_queries[pred_domain].append((domain, i, prompt_data, pred_idx, pred_domain))

    # Process adapter by adapter (1 model load per adapter)
    all_gen_results = {}
    all_routing = {}

    for adapter_domain in DOMAINS:
        queries = adapter_queries[adapter_domain]
        if not queries:
            log(f"  Adapter {adapter_domain}: 0 queries, skipping")
            continue

        log(f"  Loading adapter: {adapter_domain} ({len(queries)} queries)")
        model, tokenizer = load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)
        di = DOMAINS.index(adapter_domain)
        adapter_path = SFT_ADAPTERS_DIR / adapter_domain / "adapter.npz"

        if not adapter_path.exists():
            log(f"  WARNING: adapter not found at {adapter_path}, skipping")
            del model, tokenizer
            cleanup()
            continue

        model = apply_single_adapter_from_file(model, skeleton, di, adapter_path)
        model.freeze()

        for query_domain, prompt_idx, prompt_data, pred_idx, pred_name in queries:
            formatted = format_prompt(prompt_data["instruction"])
            generated = generate_text(model, tokenizer, formatted,
                                      max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
            score = compute_domain_score(
                generated, query_domain,
                ground_truth_response=prompt_data["response"] if query_domain == "math" else None
            )

            result = {
                "prompt": prompt_data["instruction"][:100],
                "generated": generated[:300],
                "domain_score": score,
                "keyword_density": keyword_density(generated, query_domain),
                "routed_to": pred_name,
                "true_domain": query_domain,
            }
            if query_domain == "math":
                gen_ans = extract_math_answer(generated)
                gt_ans = extract_ground_truth_answer(prompt_data["response"])
                result["answer_correct"] = math_answer_correct(gen_ans, gt_ans)
                result["gen_answer"] = gen_ans
                result["gt_answer"] = gt_ans
            elif query_domain == "code":
                result["syntax_valid"] = code_syntax_valid(generated)

            key = f"{query_domain}_{prompt_idx}"
            all_gen_results[key] = result
            all_routing[key] = {
                "query_domain": query_domain,
                "prompt_idx": prompt_idx,
                "routed_to": pred_name,
            }

        del model, tokenizer
        cleanup()
        log_memory(f"post-{adapter_domain}")

    del skeleton

    # Reorganize by domain
    routed_results = {}
    for domain in DOMAINS:
        domain_results = []
        for i in range(NUM_PROMPTS_PER_DOMAIN):
            key = f"{domain}_{i}"
            if key in all_gen_results:
                domain_results.append(all_gen_results[key])
            else:
                domain_results.append({"domain_score": 0.0, "routed_to": "MISSING"})
        routed_results[domain] = domain_results
        scores = [r["domain_score"] for r in domain_results]
        log(f"  [ROUTED] {domain}: avg_score={np.mean(scores):.4f}")

    elapsed = time.time() - t0
    return routed_results, elapsed


# ============================================================================
# Phase 5: Analysis and kill criteria
# ============================================================================

def phase_analysis(routing_results, base_results, routed_results):
    """Analyze results against kill criteria."""
    log("\n" + "=" * 70)
    log("PHASE 5: ANALYSIS AND KILL CRITERIA")
    log("=" * 70)

    analysis = {"domains": {}}

    domains_worse = 0
    for domain in DOMAINS:
        base_scores = [r["domain_score"] for r in base_results[domain]]
        routed_scores = [r["domain_score"] for r in routed_results[domain]]
        base_mean = np.mean(base_scores)
        routed_mean = np.mean(routed_scores)
        improvement = routed_mean - base_mean
        improvement_pct = 100 * improvement / max(base_mean, 0.001)

        domain_info = {
            "base_mean_score": round(float(base_mean), 4),
            "routed_mean_score": round(float(routed_mean), 4),
            "improvement": round(float(improvement), 4),
            "improvement_pct": round(float(improvement_pct), 1),
            "routed_better": bool(routed_mean > base_mean),
            "routing_accuracy": routing_results["per_domain"][domain]["accuracy"],
        }

        # Domain-specific behavioral metrics
        if domain == "math":
            base_correct = sum(1 for r in base_results[domain] if r.get("answer_correct", False))
            routed_correct = sum(1 for r in routed_results[domain] if r.get("answer_correct", False))
            domain_info["base_math_correct"] = f"{base_correct}/{len(base_results[domain])}"
            domain_info["routed_math_correct"] = f"{routed_correct}/{len(routed_results[domain])}"
            domain_info["math_correctness_base"] = base_correct / len(base_results[domain])
            domain_info["math_correctness_routed"] = routed_correct / len(routed_results[domain])
        elif domain == "code":
            base_syntax = sum(1 for r in base_results[domain] if r.get("syntax_valid", False))
            routed_syntax = sum(1 for r in routed_results[domain] if r.get("syntax_valid", False))
            domain_info["base_code_syntax"] = f"{base_syntax}/{len(base_results[domain])}"
            domain_info["routed_code_syntax"] = f"{routed_syntax}/{len(routed_results[domain])}"
            domain_info["syntax_pass_base"] = base_syntax / len(base_results[domain])
            domain_info["syntax_pass_routed"] = routed_syntax / len(routed_results[domain])

        if not domain_info["routed_better"]:
            domains_worse += 1

        analysis["domains"][domain] = domain_info
        log(f"  {domain}: base={base_mean:.4f} routed={routed_mean:.4f} "
            f"({'BETTER' if domain_info['routed_better'] else 'WORSE'}) "
            f"routing={domain_info['routing_accuracy']:.0%}")

    # Kill criteria assessment
    routing_accuracy = routing_results["overall_accuracy"]

    math_correctness = 0.0
    if "math" in analysis["domains"]:
        math_correctness = analysis["domains"]["math"].get("math_correctness_routed", 0.0)

    k605_pass = routing_accuracy >= 0.70
    k606_pass = math_correctness >= 0.60
    k607_pass = domains_worse == 0

    analysis["kill_criteria"] = {
        "K605": {
            "description": "Contrastive routing accuracy >= 70%",
            "value": round(routing_accuracy, 4),
            "threshold": 0.70,
            "result": "PASS" if k605_pass else "FAIL",
            "comparison": f"{routing_accuracy:.0%} vs 36% energy gap",
        },
        "K606": {
            "description": "Math correctness >= 60%",
            "value": round(math_correctness, 4),
            "threshold": 0.60,
            "result": "PASS" if k606_pass else "FAIL",
        },
        "K607": {
            "description": "0/5 prose domains worse than base",
            "domains_worse": domains_worse,
            "threshold": 0,
            "result": "PASS" if k607_pass else "FAIL",
            "comparison": f"{domains_worse}/5 worse (vs 3/5 with energy gap)",
        },
    }

    all_pass = k605_pass and k606_pass and k607_pass
    analysis["overall"] = "SUPPORTED" if all_pass else "MIXED" if k605_pass else "KILLED"

    log(f"\n  === KILL CRITERIA ===")
    log(f"  K605 (routing >= 70%): {'PASS' if k605_pass else 'FAIL'} ({routing_accuracy:.0%})")
    log(f"  K606 (math >= 60%): {'PASS' if k606_pass else 'FAIL'} ({math_correctness:.0%})")
    log(f"  K607 (0 worse): {'PASS' if k607_pass else 'FAIL'} ({domains_worse}/5 worse)")
    log(f"  Overall: {analysis['overall']}")

    return analysis


# ============================================================================
# Main
# ============================================================================

def main():
    t0_total = time.time()
    log("=" * 70)
    log("EXPERIMENT: Contrastive Retrieval Routing for SFT Adapters")
    log("=" * 70)
    log_memory("start")

    # Verify SFT adapters exist
    for domain in DOMAINS:
        adapter_path = SFT_ADAPTERS_DIR / domain / "adapter.npz"
        if not adapter_path.exists():
            log(f"FATAL: SFT adapter not found: {adapter_path}")
            log("Run bitnet_sft_generation_v3 first to train SFT adapters.")
            return
    log("  All 5 SFT adapters found.")

    # Load test prompts (same as energy gap experiment)
    prompts_by_domain = {}
    for domain in DOMAINS:
        prompts = extract_prompts_with_answers(domain, n_prompts=NUM_PROMPTS_PER_DOMAIN)
        prompts_by_domain[domain] = prompts
        log(f"  {domain}: {len(prompts)} test prompts")

    # Phase 1: Train text classifier
    vectorizer, clf, classifier_results = phase_train_classifier()
    log_memory("post-classifier")

    # Phase 2: Evaluate routing accuracy
    routing_results = phase_evaluate_routing(vectorizer, clf, prompts_by_domain)
    log_memory("post-routing")

    # Phase 3: Generate with base model
    base_results, base_gen_time = phase_generate_base(prompts_by_domain)

    # Phase 4: Generate with routed adapters
    routed_results, routed_gen_time = phase_generate_routed(prompts_by_domain, vectorizer, clf)

    # Phase 5: Analysis
    analysis = phase_analysis(routing_results, base_results, routed_results)

    # Save results
    total_time = time.time() - t0_total
    results = {
        "experiment": "contrastive_routing_n5",
        "model": MODEL_ID,
        "n_domains": N_DOMAINS,
        "n_prompts_per_domain": NUM_PROMPTS_PER_DOMAIN,
        "lora_rank": LORA_RANK,
        "lora_scale": LORA_SCALE,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "classifier": classifier_results,
        "routing": routing_results,
        "base_gen_time_s": round(base_gen_time, 1),
        "routed_gen_time_s": round(routed_gen_time, 1),
        "analysis": analysis,
        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
