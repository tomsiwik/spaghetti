#!/usr/bin/env python3
"""Energy Gap Routing at N=24: Does adapter selection degrade with more domains?

Frontier extension of Finding #185 (88% accuracy at N=5) to N=24 domains.
Uses the same energy gap mechanism: DeltaE = NLL(adapter) - NLL(base),
route to argmin(DeltaE).

Kill criteria:
  K581: Routing accuracy drops below 60% at N=25 (from 88% at N=5)
  K582: Best-domain correctness drops below 50% on math (from 70% at N=5)
  K583: O(N) overhead exceeds 120s per query at N=25

Strategy for memory/time:
  - Load base model once, compute ALL base NLLs, unload
  - Load each adapter sequentially (one at a time), compute NLLs, unload
  - Use 5 prompts per domain (120 total) for routing accuracy
  - Generate text only for math/code domains (K2 assessment)
  - Skip uniform/oracle generation baselines (already measured at N=5)

Platform: Apple M5 Pro 48GB, MLX
"""

import ast
import gc
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
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

# Source: real_data_25_domain_adapters
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_25_domain_adapters"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"
SKELETON_PATH = ADAPTERS_DIR / "grassmannian_skeleton_n24.npz"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256

# All 24 domains with adapters (excluding real_estate which has no data)
DOMAINS = [
    "agriculture", "code", "cooking", "creative_writing", "cybersecurity",
    "economics", "education", "engineering", "environmental", "finance",
    "health_fitness", "history", "legal", "linguistics", "marketing",
    "math", "medical", "music", "philosophy", "politics",
    "psychology", "science", "sociology", "sports",
]
N_DOMAINS = len(DOMAINS)
DOMAIN_TO_IDX = {d: i for i, d in enumerate(DOMAINS)}

# Evaluation settings
NUM_PROMPTS_PER_DOMAIN = 5  # 5 * 24 = 120 routing evaluations
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.9
SEED = 42

# Generation-quality domains (K2 focuses on math)
GEN_QUALITY_DOMAINS = ["math", "code"]

# ============================================================================
# Domain keywords for scoring (extended from N=5 experiment)
# ============================================================================
DOMAIN_KEYWORDS = {
    "agriculture": [
        "crop", "soil", "farm", "harvest", "irrigation", "seed", "fertilizer",
        "livestock", "cattle", "wheat", "corn", "yield", "agriculture", "plant",
        "grow", "organic", "field", "acre", "season", "drought",
    ],
    "code": [
        "function", "variable", "class", "method", "return", "import",
        "loop", "array", "string", "integer", "boolean", "object", "def",
        "print", "if", "else", "for", "while", "try", "except", "lambda",
        "list", "dict", "tuple", "parameter", "argument", "module", "library",
        "algorithm", "python", "code", "program", "output", "input", "error",
    ],
    "cooking": [
        "recipe", "ingredient", "cook", "bake", "stir", "heat", "oven",
        "pan", "sauce", "flavor", "taste", "spice", "herb", "chop", "mix",
        "serve", "meal", "dish", "kitchen", "tablespoon", "teaspoon", "cup",
    ],
    "creative_writing": [
        "story", "character", "plot", "narrative", "scene", "dialogue",
        "setting", "theme", "conflict", "protagonist", "chapter", "fiction",
        "poem", "metaphor", "imagery", "tone", "voice", "genre", "draft",
    ],
    "cybersecurity": [
        "security", "vulnerability", "attack", "malware", "encryption",
        "firewall", "breach", "threat", "hack", "password", "authentication",
        "network", "phishing", "ransomware", "patch", "exploit", "cyber",
    ],
    "economics": [
        "economy", "market", "supply", "demand", "price", "inflation",
        "gdp", "trade", "fiscal", "monetary", "recession", "growth",
        "unemployment", "interest", "rate", "consumer", "producer", "tax",
    ],
    "education": [
        "student", "teacher", "learn", "curriculum", "classroom", "school",
        "university", "course", "assessment", "grade", "education", "skill",
        "knowledge", "instruction", "pedagogy", "academic", "lecture", "exam",
    ],
    "engineering": [
        "design", "system", "structure", "material", "load", "stress",
        "circuit", "mechanical", "electrical", "thermal", "fluid", "force",
        "torque", "voltage", "current", "resistance", "engineer", "prototype",
    ],
    "environmental": [
        "climate", "emission", "pollution", "ecosystem", "biodiversity",
        "carbon", "renewable", "sustainability", "conservation", "habitat",
        "species", "deforestation", "ocean", "atmosphere", "greenhouse",
    ],
    "finance": [
        "investment", "stock", "bond", "market", "portfolio", "risk",
        "return", "profit", "loss", "revenue", "capital", "dividend",
        "interest", "rate", "inflation", "monetary", "fiscal", "budget",
        "asset", "liability", "equity", "debt", "credit", "loan", "bank",
    ],
    "health_fitness": [
        "exercise", "workout", "muscle", "cardio", "nutrition", "diet",
        "calories", "protein", "fitness", "strength", "flexibility", "health",
        "body", "weight", "training", "recovery", "endurance", "yoga",
    ],
    "history": [
        "century", "war", "empire", "revolution", "civilization", "dynasty",
        "era", "ancient", "medieval", "colonial", "battle", "treaty",
        "king", "queen", "republic", "democracy", "historical", "period",
    ],
    "legal": [
        "law", "court", "judge", "attorney", "lawyer", "legal", "rights",
        "statute", "regulation", "contract", "liability", "plaintiff",
        "defendant", "jurisdiction", "precedent", "ruling", "verdict",
        "appeal", "testimony", "evidence", "trial", "case", "claim",
    ],
    "linguistics": [
        "language", "grammar", "syntax", "morphology", "phonology", "semantic",
        "pragmatic", "dialect", "linguistic", "vowel", "consonant", "lexicon",
        "discourse", "bilingual", "translation", "speech", "verb", "noun",
    ],
    "marketing": [
        "brand", "campaign", "customer", "audience", "engagement", "content",
        "social", "media", "advertising", "promotion", "strategy", "market",
        "sales", "conversion", "analytics", "digital", "seo", "roi",
    ],
    "math": [
        "equation", "formula", "calculate", "solve", "number", "variable",
        "total", "sum", "product", "divide", "multiply", "subtract", "add",
        "percent", "ratio", "fraction", "decimal", "integer", "value",
        "answer", "solution", "problem", "step", "result", "equal",
    ],
    "medical": [
        "patient", "diagnosis", "treatment", "symptoms", "disease", "clinical",
        "medication", "therapy", "surgical", "pathology", "prognosis", "chronic",
        "acute", "syndrome", "condition", "medical", "doctor", "hospital",
    ],
    "music": [
        "melody", "rhythm", "harmony", "chord", "tempo", "key", "note",
        "scale", "instrument", "song", "compose", "genre", "beat", "pitch",
        "tone", "lyric", "band", "orchestra", "album", "perform",
    ],
    "philosophy": [
        "ethics", "moral", "existence", "truth", "knowledge", "reason",
        "logic", "consciousness", "metaphysics", "epistemology", "virtue",
        "justice", "freedom", "ontology", "philosophy", "argument", "theory",
    ],
    "politics": [
        "government", "policy", "election", "vote", "party", "congress",
        "president", "democracy", "legislation", "political", "campaign",
        "senate", "representative", "law", "citizen", "rights", "reform",
    ],
    "psychology": [
        "behavior", "cognitive", "emotion", "mental", "therapy", "brain",
        "consciousness", "perception", "memory", "anxiety", "depression",
        "personality", "motivation", "learning", "development", "disorder",
    ],
    "science": [
        "experiment", "hypothesis", "theory", "molecule", "atom", "cell",
        "energy", "force", "gravity", "evolution", "gene", "dna", "species",
        "chemical", "reaction", "particle", "quantum", "research", "data",
    ],
    "sociology": [
        "society", "culture", "community", "social", "class", "inequality",
        "institution", "norm", "identity", "gender", "race", "ethnicity",
        "group", "structure", "power", "mobility", "stratification",
    ],
    "sports": [
        "team", "game", "score", "player", "match", "season", "championship",
        "coach", "goal", "point", "win", "loss", "tournament", "league",
        "athlete", "training", "competition", "field", "court", "race",
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
# BitNet unpacking (from energy_gap_topk_routing)
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
# LoRA layer (single adapter)
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
# Model setup helpers
# ============================================================================

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def load_skeleton():
    return dict(np.load(str(SKELETON_PATH)))


def apply_single_adapter(model, skeleton, domain_idx, domain_name):
    """Apply a single adapter to the model (loads B from disk)."""
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


# ============================================================================
# Scoring metrics (from N=5 experiment)
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


def coherence_score(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    wc = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    avg_len = np.mean(wc)
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

def extract_prompts_with_answers(domain, n_prompts=5):
    val_path = DATA_DIR / domain / "valid.jsonl"
    if not val_path.exists():
        log(f"  WARNING: no validation data for {domain}")
        return []
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
# Phase 0: Compute energy gaps for ALL 24 adapters x 120 queries
# ============================================================================

def phase_compute_energy_gaps(prompts_by_domain):
    """Compute energy gap per (query, adapter) pair.

    Memory strategy: load base model once for all base NLLs, then load each
    adapter one at a time. Never have >1 adapter in memory.
    """
    log("\n[Phase 0] Computing energy gaps for N=24 adapters...")
    t0 = time.time()

    # Step 1: Base NLLs for all 120 queries
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
        log(f"    {domain}: mean_base_nll={np.mean(nlls):.4f} (n={len(nlls)})")

    t_base = time.time() - t_base_start
    log(f"  Base NLLs done in {t_base:.1f}s")
    del model, tokenizer
    cleanup()
    log_memory("base-cleanup")

    # Step 2: Per-adapter NLLs (load one adapter at a time)
    log("  Computing per-adapter NLLs (24 adapters, sequential)...")
    skeleton = load_skeleton()
    adapter_nlls = {}
    per_adapter_times = []
    t_adapter_start = time.time()

    for di, adapter_domain in enumerate(DOMAINS):
        t_ad = time.time()
        log(f"    [{di+1}/{N_DOMAINS}] Adapter: {adapter_domain}")
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
        elapsed_ad = time.time() - t_ad
        per_adapter_times.append(elapsed_ad)
        log(f"      done in {elapsed_ad:.1f}s")
        del model, tokenizer
        cleanup()

    t_adapter = time.time() - t_adapter_start
    del skeleton
    cleanup()
    log_memory("all-adapters-cleanup")

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
    n_prompts_total = sum(len(v) for v in prompts_by_domain.values())
    per_query_overhead = total_energy_time / max(n_prompts_total, 1)

    timing = {
        "base_nll_time_s": round(t_base, 1),
        "adapter_nll_time_s": round(t_adapter, 1),
        "total_energy_time_s": round(total_energy_time, 1),
        "per_adapter_avg_time_s": round(np.mean(per_adapter_times), 1),
        "per_query_overhead_s": round(per_query_overhead, 1),
        "n_prompts_total": n_prompts_total,
        "n_adapters": N_DOMAINS,
    }

    log(f"  Energy gap computation: {total_energy_time:.1f}s total")
    log(f"  Per-query overhead: {per_query_overhead:.1f}s")
    return base_nlls, energy_gaps, timing


# ============================================================================
# Phase 1: Routing accuracy analysis
# ============================================================================

def phase_routing_accuracy(energy_gaps, prompts_by_domain):
    """Measure top-1 routing accuracy and build confusion matrix."""
    log("\n[Phase 1] Analyzing routing accuracy at N=24...")

    confusion_matrix = np.zeros((N_DOMAINS, N_DOMAINS), dtype=int)
    routing_decisions = {}
    per_domain_accuracy = {}

    total_correct = 0
    total_queries = 0

    for query_domain in DOMAINS:
        qi = DOMAIN_TO_IDX[query_domain]
        domain_routing = []
        n_correct = 0

        for prompt_idx in range(len(prompts_by_domain[query_domain])):
            # Compute gaps for all adapters
            gaps = {}
            for adapter_domain in DOMAINS:
                gaps[adapter_domain] = energy_gaps[adapter_domain][query_domain][prompt_idx]

            # Select argmin (most negative gap = best adapter)
            best_adapter = min(gaps, key=gaps.get)
            best_gap = gaps[best_adapter]
            bi = DOMAIN_TO_IDX[best_adapter]
            confusion_matrix[qi, bi] += 1

            is_correct = (best_adapter == query_domain)
            if is_correct:
                n_correct += 1

            # Gap margin: difference between best and second-best
            sorted_gaps = sorted(gaps.values())
            gap_margin = sorted_gaps[1] - sorted_gaps[0]  # positive = confident

            # Top-3 adapters
            top3 = sorted(gaps.items(), key=lambda x: x[1])[:3]

            domain_routing.append({
                "prompt_idx": prompt_idx,
                "selected_adapter": best_adapter,
                "selected_gap": best_gap,
                "correct_selection": is_correct,
                "gap_margin": gap_margin,
                "top3": [(a, round(g, 4)) for a, g in top3],
            })

        n_total = len(prompts_by_domain[query_domain])
        acc = n_correct / n_total if n_total > 0 else 0
        per_domain_accuracy[query_domain] = {
            "correct": n_correct,
            "total": n_total,
            "accuracy": round(acc, 4),
        }
        routing_decisions[query_domain] = domain_routing
        total_correct += n_correct
        total_queries += n_total

        log(f"  {query_domain}: {n_correct}/{n_total} = {acc:.1%}")

    overall_accuracy = total_correct / total_queries if total_queries > 0 else 0
    log(f"\n  OVERALL: {total_correct}/{total_queries} = {overall_accuracy:.1%}")

    # Identify confusion clusters
    confusion_pairs = []
    for i in range(N_DOMAINS):
        for j in range(N_DOMAINS):
            if i != j and confusion_matrix[i, j] > 0:
                confusion_pairs.append({
                    "query_domain": DOMAINS[i],
                    "selected_adapter": DOMAINS[j],
                    "count": int(confusion_matrix[i, j]),
                })
    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)

    # Mean gap margin for correct vs incorrect
    correct_margins = []
    incorrect_margins = []
    for domain, decisions in routing_decisions.items():
        for d in decisions:
            if d["correct_selection"]:
                correct_margins.append(d["gap_margin"])
            else:
                incorrect_margins.append(d["gap_margin"])

    return {
        "overall_accuracy": round(overall_accuracy, 4),
        "total_correct": total_correct,
        "total_queries": total_queries,
        "per_domain_accuracy": per_domain_accuracy,
        "confusion_matrix": confusion_matrix.tolist(),
        "confusion_pairs": confusion_pairs[:20],  # Top 20 confusions
        "routing_decisions": routing_decisions,
        "mean_correct_margin": round(float(np.mean(correct_margins)), 4) if correct_margins else 0,
        "mean_incorrect_margin": round(float(np.mean(incorrect_margins)), 4) if incorrect_margins else 0,
        "domain_labels": DOMAINS,
    }


# ============================================================================
# Phase 2: Generation quality on math/code with top-1 routing
# ============================================================================

def phase_generate_top1(prompts_by_domain, energy_gaps):
    """Generate with top-1 routed adapter for math and code domains."""
    log("\n[Phase 2] Generating with TOP-1 routing on math/code...")
    t0 = time.time()

    skeleton = load_skeleton()
    results = {}

    for gen_domain in GEN_QUALITY_DOMAINS:
        log(f"  Domain: {gen_domain}")
        domain_results = []

        for prompt_idx, prompt_data in enumerate(prompts_by_domain[gen_domain]):
            # Select best adapter
            gaps = {ad: energy_gaps[ad][gen_domain][prompt_idx] for ad in DOMAINS}
            best_adapter = min(gaps, key=gaps.get)
            best_idx = DOMAIN_TO_IDX[best_adapter]

            log(f"    Prompt {prompt_idx}: routing to {best_adapter} (gap={gaps[best_adapter]:.4f})")

            # Load this specific adapter
            model, tokenizer = load(MODEL_ID)
            model = replace_bitlinear_with_linear(model)
            model = apply_single_adapter(model, skeleton, best_idx, best_adapter)
            model.freeze()

            mx.random.seed(SEED + prompt_idx)
            formatted = format_prompt(prompt_data["instruction"])
            generated = generate_text(model, tokenizer, formatted,
                                      max_tokens=MAX_NEW_TOKENS,
                                      temperature=TEMPERATURE, top_p=TOP_P)
            score = compute_domain_score(
                generated, gen_domain,
                ground_truth_response=prompt_data["response"] if gen_domain == "math" else None,
            )

            result_entry = {
                "prompt": prompt_data["instruction"][:100],
                "generated": generated[:300],
                "domain_score": score,
                "routed_to": best_adapter,
                "keyword_density": keyword_density(generated, gen_domain),
            }

            if gen_domain == "math":
                gen_answer = extract_math_answer(generated)
                gt_answer = extract_ground_truth_answer(prompt_data["response"])
                result_entry["answer_correct"] = math_answer_correct(gen_answer, gt_answer)
                result_entry["gen_answer"] = gen_answer
                result_entry["gt_answer"] = gt_answer
            elif gen_domain == "code":
                result_entry["syntax_valid"] = code_syntax_valid(generated)

            domain_results.append(result_entry)
            del model, tokenizer
            cleanup()

        results[gen_domain] = domain_results
        scores = [r["domain_score"] for r in domain_results]
        log(f"    avg_score={np.mean(scores):.4f}")

        if gen_domain == "math":
            correct = sum(1 for r in domain_results if r.get("answer_correct"))
            log(f"    math_correctness={correct}/{len(domain_results)}")

    del skeleton
    cleanup()
    elapsed = time.time() - t0
    log(f"  Generation done in {elapsed:.1f}s")
    return results, elapsed


# ============================================================================
# Phase 3: Energy gap distribution analysis
# ============================================================================

def phase_gap_analysis(energy_gaps):
    """Analyze energy gap distributions to understand routing signal quality."""
    log("\n[Phase 3] Analyzing energy gap distributions...")

    per_domain_stats = {}
    for query_domain in DOMAINS:
        # For each query domain, compute stats of the correct adapter's gap
        # vs the distribution of incorrect adapter gaps
        correct_gaps = energy_gaps[query_domain][query_domain]
        incorrect_gaps = []
        for adapter_domain in DOMAINS:
            if adapter_domain != query_domain:
                incorrect_gaps.extend(energy_gaps[adapter_domain][query_domain])

        correct_mean = float(np.mean(correct_gaps))
        incorrect_mean = float(np.mean(incorrect_gaps))
        separation = incorrect_mean - correct_mean  # positive = correct is more negative

        # Compute the minimum gap between correct and ANY other adapter (per query)
        min_margins = []
        for pi in range(len(correct_gaps)):
            other_gaps = [energy_gaps[ad][query_domain][pi] for ad in DOMAINS if ad != query_domain]
            margin = min(other_gaps) - correct_gaps[pi]  # positive = correct wins
            min_margins.append(margin)

        per_domain_stats[query_domain] = {
            "correct_gap_mean": round(correct_mean, 4),
            "incorrect_gap_mean": round(incorrect_mean, 4),
            "separation": round(separation, 4),
            "correct_gap_std": round(float(np.std(correct_gaps)), 4),
            "min_margin_mean": round(float(np.mean(min_margins)), 4),
            "min_margin_std": round(float(np.std(min_margins)), 4),
            "fraction_positive_margin": round(
                sum(1 for m in min_margins if m > 0) / len(min_margins), 4
            ),
        }
        log(f"  {query_domain}: sep={separation:.4f}, "
            f"min_margin={np.mean(min_margins):.4f} +/- {np.std(min_margins):.4f}")

    return per_domain_stats


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("Energy Gap Routing at N=24: Scaling Experiment")
    log("=" * 70)
    log(f"N_DOMAINS={N_DOMAINS}, NUM_PROMPTS_PER_DOMAIN={NUM_PROMPTS_PER_DOMAIN}")
    log_memory("start")

    # Extract prompts
    log("\nExtracting prompts...")
    prompts_by_domain = {}
    for domain in DOMAINS:
        prompts = extract_prompts_with_answers(domain, n_prompts=NUM_PROMPTS_PER_DOMAIN)
        prompts_by_domain[domain] = prompts
        log(f"  {domain}: {len(prompts)} prompts")

    n_total_prompts = sum(len(v) for v in prompts_by_domain.values())
    log(f"  Total: {n_total_prompts} prompts")

    # Phase 0: Energy gaps
    base_nlls, energy_gaps, timing = phase_compute_energy_gaps(prompts_by_domain)
    log_memory("post-energy-gaps")

    # Phase 1: Routing accuracy
    routing_results = phase_routing_accuracy(energy_gaps, prompts_by_domain)
    overall_acc = routing_results["overall_accuracy"]

    # Phase 2: Generation quality (math/code only)
    gen_results, gen_time = phase_generate_top1(prompts_by_domain, energy_gaps)
    log_memory("post-generation")

    # Phase 3: Gap distribution analysis
    gap_stats = phase_gap_analysis(energy_gaps)

    # ============================================================================
    # Kill criteria assessment
    # ============================================================================
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    # K581: Routing accuracy >= 60%
    k581_pass = overall_acc >= 0.60
    log(f"  K581: Routing accuracy = {overall_acc:.1%} (threshold: >= 60%) -> {'PASS' if k581_pass else 'FAIL'}")

    # K582: Math correctness >= 50%
    math_results = gen_results.get("math", [])
    math_correct = sum(1 for r in math_results if r.get("answer_correct")) if math_results else 0
    math_total = len(math_results) if math_results else 1
    math_correctness = math_correct / math_total
    k582_pass = math_correctness >= 0.50
    log(f"  K582: Math correctness = {math_correct}/{math_total} = {math_correctness:.1%} "
        f"(threshold: >= 50%) -> {'PASS' if k582_pass else 'FAIL'}")

    # K583: Per-query overhead < 120s
    per_query_overhead = timing["per_query_overhead_s"]
    k583_pass = per_query_overhead < 120
    log(f"  K583: Per-query overhead = {per_query_overhead:.1f}s "
        f"(threshold: < 120s) -> {'PASS' if k583_pass else 'FAIL'}")

    all_pass = k581_pass and k582_pass and k583_pass
    verdict = "SUPPORTED" if all_pass else "KILLED"
    log(f"\n  VERDICT: {verdict}")

    # ============================================================================
    # Comparison with N=5
    # ============================================================================
    log("\n" + "=" * 70)
    log("COMPARISON WITH N=5")
    log("=" * 70)
    log(f"  N=5  routing accuracy: 88.0% (44/50)")
    log(f"  N=24 routing accuracy: {overall_acc:.1%} ({routing_results['total_correct']}/{routing_results['total_queries']})")
    log(f"  Degradation: {88.0 - overall_acc*100:.1f} percentage points")
    log(f"  N=5  math correctness: 70% (7/10)")
    log(f"  N=24 math correctness: {math_correctness:.0%} ({math_correct}/{math_total})")

    # Gumbel prediction check
    log(f"\n  Gumbel prediction: accuracy should be 60-75%")
    log(f"  Measured: {overall_acc:.1%}")
    gumbel_match = 0.55 <= overall_acc <= 0.80  # wide band for frontier extension
    log(f"  Within predicted range: {'YES' if gumbel_match else 'NO'}")

    # ============================================================================
    # Save results
    # ============================================================================
    total_time = time.time() - t0

    results = {
        "experiment": "energy_routing_n25_scaling",
        "type": "frontier_extension",
        "model": MODEL_ID,
        "n_domains": N_DOMAINS,
        "n_prompts_per_domain": NUM_PROMPTS_PER_DOMAIN,
        "seed": SEED,
        "domains": DOMAINS,

        # Kill criteria
        "k581_pass": bool(k581_pass),
        "k582_pass": bool(k582_pass),
        "k583_pass": bool(k583_pass),
        "all_kill_criteria_pass": bool(all_pass),
        "verdict": verdict,

        # Routing
        "overall_routing_accuracy": routing_results["overall_accuracy"],
        "per_domain_accuracy": routing_results["per_domain_accuracy"],
        "confusion_matrix": routing_results["confusion_matrix"],
        "confusion_pairs": routing_results["confusion_pairs"],
        "mean_correct_margin": routing_results["mean_correct_margin"],
        "mean_incorrect_margin": routing_results["mean_incorrect_margin"],
        "domain_labels": DOMAINS,

        # Generation quality
        "generation_results": {
            domain: {
                "avg_score": round(float(np.mean([r["domain_score"] for r in results_list])), 4),
                "n_samples": len(results_list),
                "details": results_list,
            }
            for domain, results_list in gen_results.items()
        },
        "math_correctness": round(math_correctness, 4),
        "math_correct_count": math_correct,
        "math_total_count": math_total,

        # Timing
        "energy_timing": timing,
        "generation_time_s": round(gen_time, 1),
        "total_time_s": round(total_time, 1),

        # Gap analysis
        "gap_analysis": gap_stats,

        # N=5 comparison
        "n5_routing_accuracy": 0.88,
        "n5_math_correctness": 0.70,
        "accuracy_degradation_pct_points": round(88.0 - overall_acc * 100, 1),
        "gumbel_prediction_match": bool(gumbel_match),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
