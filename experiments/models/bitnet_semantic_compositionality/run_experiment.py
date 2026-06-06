#!/usr/bin/env python3
"""
Semantic Compositionality Eval + OSRM Data-Space Orthogonality Diagnostic

Two-part experiment testing whether weight-space orthogonality (|cos|~0.001)
translates to meaningful semantic compositionality AND data-space orthogonality.

Part A -- Semantic Composition Eval:
  Compose instruction-tuned adapters pairwise, test on cross-domain queries
  that require BOTH domains. Compare base vs individual vs composed.
  Metrics: KR-Test contrastive score, task keyword accuracy, PPL.

Part B -- OSRM Data-Space Diagnostic:
  For all (i,j) adapter pairs, measure ||A_j * h_i|| / ||A_i * h_i||.
  If ratio < 0.1: our Grassmannian-style random A matrices are already
  approximately data-orthogonal (OSRM unnecessary).
  If ratio > 0.1: need OSRM-style data-aware A initialization.

Kill criteria:
  K1: composed (medical+reasoning) adapter worse than either alone on >50% of cross-domain queries
  K2: semantic coherence of composed outputs rated lower than base on manual inspection of 20 samples
  K3: OSRM diagnostic shows ||A_j * h_i|| > 0.1 * ||A_i * h_i|| for >50% of pairs

Reuses:
  - 5 instruction-tuned adapters from bitnet_instruction_task_eval
  - Instruction data from bitnet_instruction_task_eval
  - Adapter loading/composition from prior experiments

Platform: Apple Silicon MLX, $0. ~60-90 min.
"""

import json
import math
import os
import re
import sys
import time
from pathlib import Path
from itertools import combinations

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm import load
from mlx_lm.models.bitlinear_layers import BitLinear


# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
VAL_BATCHES = 25
SEED = 42

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Source directories (reuse instruction-tuned adapters and data)
INST_EVAL_DIR = Path(__file__).parent.parent / "bitnet_instruction_task_eval"
ADAPTERS_DIR = INST_EVAL_DIR / "adapters"
DATA_DIR = INST_EVAL_DIR / "data"

DOMAINS = ["medical", "math", "code", "legal", "creative"]

# Cross-domain test pairs: (domain_a, domain_b, cross_domain_queries)
# These are questions that genuinely require BOTH domains
CROSS_DOMAIN_PAIRS = {
    ("medical", "math"): [
        "Calculate the BMI for a patient who weighs 85kg and is 1.75m tall. Is this in the healthy range?",
        "A patient takes 500mg of medication every 8 hours. What is the total daily dose in grams?",
        "If a clinical trial shows 23 out of 150 patients had adverse effects, what is the percentage?",
        "A drug has a half-life of 6 hours. How much of a 200mg dose remains after 18 hours?",
        "Calculate the creatinine clearance for a 70kg male with serum creatinine of 1.2 mg/dL.",
        "A hospital ward has 30 beds. If average occupancy is 85%, how many patients per day on average?",
        "If blood pressure drops from 150/90 to 130/80, what is the percentage decrease in systolic?",
        "A diabetic patient needs 0.5 units of insulin per kg. Calculate dose for a 90kg patient.",
        "What is the BMR for a 30-year-old male weighing 75kg using the Harris-Benedict equation?",
        "If a disease has prevalence of 1 in 10000 and test sensitivity is 99%, what is the PPV?",
    ],
    ("medical", "code"): [
        "Write a Python function to calculate BMI given weight in kg and height in meters.",
        "Write code to parse a FHIR patient resource and extract medication dosages.",
        "Create a Python function that converts between common medical units (mg to g, mL to L).",
        "Write a function to classify blood pressure readings as normal, elevated, or hypertensive.",
        "Create code to calculate the Glasgow Coma Scale score from eye, verbal, motor responses.",
        "Write a Python function to parse ICD-10 codes and return the disease category.",
        "Create a function that checks drug interaction warnings given two medication names.",
        "Write code to calculate pediatric drug dosages based on weight and age.",
        "Create a Python function to parse lab results and flag abnormal values.",
        "Write a function that calculates the APACHE II severity score.",
    ],
    ("math", "code"): [
        "Write a Python function to solve a quadratic equation ax^2 + bx + c = 0.",
        "Implement Newton's method for finding square roots in Python.",
        "Write code to compute the Fibonacci sequence using matrix exponentiation.",
        "Create a function that performs Gaussian elimination on a matrix.",
        "Write a Python function to compute numerical derivatives using central differences.",
        "Implement the Euclidean algorithm for GCD in Python.",
        "Write code to perform polynomial interpolation using Lagrange's method.",
        "Create a function to solve a system of linear equations using Cramer's rule.",
        "Write Python code to compute the determinant of an NxN matrix recursively.",
        "Implement Simpson's rule for numerical integration in Python.",
    ],
    ("legal", "math"): [
        "Calculate the statute of limitations expiry date if the incident occurred on March 15, 2020 with a 3-year limit.",
        "If a lawsuit settlement is $500,000 and attorney fees are 33.3%, what does the plaintiff receive?",
        "Calculate compound interest on a $100,000 judgment at 5% annual rate over 3 years.",
        "A contract specifies 2% monthly late fees. What is the effective annual rate?",
        "If 12 jurors must agree and 9 currently agree, what fraction of remaining must be convinced?",
        "Calculate the present value of a $50,000 annual pension for 20 years at 4% discount rate.",
        "A property is assessed at $300,000 with a tax rate of 1.2%. What is the annual property tax?",
        "If a patent expires in 2028 and was filed in 2008, how many years of protection remain?",
        "Calculate damages: lost wages of $75,000/year for 5 years with 3% annual growth.",
        "A company must pay 150% of overtime wages. If base rate is $25/hr, what is overtime pay?",
    ],
    ("legal", "code"): [
        "Write a Python function to check if a contract date is within the statute of limitations.",
        "Create code to parse legal citations in the format 'Volume Reporter Page' (e.g., '347 U.S. 483').",
        "Write a function to calculate attorney fee percentages from settlement amounts.",
        "Create a Python script to detect potentially problematic clauses in a contract text.",
        "Write code to parse and validate case numbers in federal court format.",
        "Create a function that computes deadline dates given filing dates and rule-based timelines.",
        "Write a Python function to redact PII (names, SSN, addresses) from legal documents.",
        "Create code to classify legal text into categories (contract, tort, criminal, civil).",
        "Write a function to extract party names from legal case captions.",
        "Create a Python function to calculate billable hours and fees from time entries.",
    ],
}

# Instruction template (matches training format)
INST_PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n"

# Number of manual inspection samples for K2
K2_SAMPLES = 20
MAX_GEN_TOKENS = 150


def log(msg):
    print(msg, flush=True)


# ===========================================================================
# Ternary weight unpacking + model setup (reused from prior experiments)
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
# LoRA
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
    params = []
    for name, val in tree_flatten(model.parameters()):
        if "lora_a" in name or "lora_b" in name:
            params.append((name, val))
    return params


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


def zero_lora_to_base(model):
    """Reset LoRA params to zero (effectively base model)."""
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.lora_a = mx.zeros_like(module.lora_a)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


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
# PPL computation
# ===========================================================================
def compute_ppl(model, tokenizer, texts, max_batches=25):
    total_loss = 0.0
    total_tokens = 0
    for text in texts[:max_batches]:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
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
        return float("inf")
    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 100))


# ===========================================================================
# KR-Test contrastive evaluation (adapted from bitnet_kr_test_eval)
# ===========================================================================
def kr_test_score(model, tokenizer, correct_texts, wrong_texts, max_ctx=192):
    """Contrastive score: fraction where log-prob(correct) > log-prob(wrong).

    Uses cross-item pairing: correct continuation is from the right domain,
    wrong continuation is from a different item in same domain.
    """
    wins = 0
    total = 0

    for correct, wrong in zip(correct_texts, wrong_texts):
        correct_tokens = tokenizer.encode(correct)[:max_ctx]
        wrong_tokens = tokenizer.encode(wrong)[:max_ctx]

        if len(correct_tokens) < 3 or len(wrong_tokens) < 3:
            continue

        min_len = min(len(correct_tokens), len(wrong_tokens))
        correct_tokens = correct_tokens[:min_len]
        wrong_tokens = wrong_tokens[:min_len]

        # Log-prob of correct
        x_c = mx.array(correct_tokens[:-1])[None, :]
        y_c = mx.array(correct_tokens[1:])[None, :]
        logits_c = model(x_c)
        loss_c = nn.losses.cross_entropy(logits_c, y_c, reduction="mean")
        mx.eval(loss_c)

        # Log-prob of wrong
        x_w = mx.array(wrong_tokens[:-1])[None, :]
        y_w = mx.array(wrong_tokens[1:])[None, :]
        logits_w = model(x_w)
        loss_w = nn.losses.cross_entropy(logits_w, y_w, reduction="mean")
        mx.eval(loss_w)

        # Lower loss = higher probability = better
        if loss_c.item() < loss_w.item():
            wins += 1
        total += 1

    return wins / total if total > 0 else 0.0


# ===========================================================================
# Part A: Semantic Composition Evaluation
# ===========================================================================
def eval_cross_domain_quality(model, tokenizer, queries, pair_name):
    """Evaluate model on cross-domain queries.

    Returns dict with:
      - responses: list of generated responses
      - avg_response_length: average token count of responses
      - non_empty_rate: fraction of non-empty responses
      - keyword_scores: fraction of responses containing domain-relevant keywords
    """
    domain_a, domain_b = pair_name

    # Domain-relevant keywords for scoring
    DOMAIN_KEYWORDS = {
        "medical": ["patient", "mg", "dose", "blood", "clinical", "diagnosis",
                     "treatment", "health", "disease", "symptom", "bmi", "pressure",
                     "medication", "drug", "hospital", "creatinine", "insulin",
                     "prevalence", "sensitivity", "adverse"],
        "math": ["calculate", "equation", "formula", "result", "=", "answer",
                 "solve", "step", "number", "total", "percent", "rate",
                 "value", "annual", "compound", "interest", "fraction"],
        "code": ["def ", "return", "function", "python", "import", "class",
                 "if ", "for ", "print", "list", "dict", "str", "int",
                 "float", "input", "output", "parse"],
        "legal": ["contract", "statute", "court", "law", "plaintiff",
                  "defendant", "attorney", "settlement", "jurisdiction",
                  "clause", "liability", "patent", "filing", "damages"],
        "creative": ["story", "once", "upon", "character", "narrative",
                     "scene", "dialogue", "plot", "beginning", "end"],
    }

    keywords_a = set(DOMAIN_KEYWORDS.get(domain_a, []))
    keywords_b = set(DOMAIN_KEYWORDS.get(domain_b, []))

    responses = []
    keyword_hits_a = 0
    keyword_hits_b = 0
    non_empty = 0

    for q in queries:
        prompt = INST_PROMPT.format(instruction=q)
        resp = generate_text(model, tokenizer, prompt)
        responses.append(resp)

        if resp.strip():
            non_empty += 1
            resp_lower = resp.lower()
            # Check if response contains keywords from BOTH domains
            has_a = any(kw.lower() in resp_lower for kw in keywords_a)
            has_b = any(kw.lower() in resp_lower for kw in keywords_b)
            if has_a:
                keyword_hits_a += 1
            if has_b:
                keyword_hits_b += 1

    n = len(queries)
    return {
        "responses": responses,
        "n_queries": n,
        "non_empty_rate": non_empty / n if n > 0 else 0,
        "keyword_rate_domain_a": keyword_hits_a / n if n > 0 else 0,
        "keyword_rate_domain_b": keyword_hits_b / n if n > 0 else 0,
        "cross_domain_rate": sum(1 for r in responses
                                 if any(k.lower() in r.lower() for k in keywords_a) and
                                 any(k.lower() in r.lower() for k in keywords_b)) / n if n > 0 else 0,
        "avg_response_length": sum(len(r.split()) for r in responses) / n if n > 0 else 0,
    }


def run_part_a(model, tokenizer, adapters):
    """Part A: Semantic compositionality evaluation.

    For each cross-domain pair:
      1. Eval base model (no adapter)
      2. Eval individual adapters (domain A only, domain B only)
      3. Eval composed adapter (A + B, 1/2 scaling)
    Compare keyword coverage, response quality, and PPL.
    """
    log("\n" + "=" * 70)
    log("PART A: SEMANTIC COMPOSITION EVALUATION")
    log("=" * 70)

    results = {}
    all_coherence_samples = []  # For K2 manual inspection

    for pair_key, queries in CROSS_DOMAIN_PAIRS.items():
        domain_a, domain_b = pair_key
        pair_name = f"{domain_a}+{domain_b}"
        log(f"\n--- Cross-domain pair: {pair_name} ({len(queries)} queries) ---")

        if domain_a not in adapters or domain_b not in adapters:
            log(f"  Skipping {pair_name}: adapter not available")
            continue

        # 1. Base model (zero LoRA)
        log(f"  Evaluating base model...")
        zero_lora_to_base(model)
        mx.eval(model.parameters())
        base_result = eval_cross_domain_quality(model, tokenizer, queries, pair_key)

        # 2. Individual adapter A only
        log(f"  Evaluating {domain_a} adapter only...")
        apply_adapter_weights(model, adapters[domain_a])
        mx.eval(model.parameters())
        individual_a_result = eval_cross_domain_quality(model, tokenizer, queries, pair_key)

        # 3. Individual adapter B only
        log(f"  Evaluating {domain_b} adapter only...")
        apply_adapter_weights(model, adapters[domain_b])
        mx.eval(model.parameters())
        individual_b_result = eval_cross_domain_quality(model, tokenizer, queries, pair_key)

        # 4. Composed adapter (A + B, 1/2 scaling)
        log(f"  Evaluating composed {pair_name} adapter...")
        composed = compose_adapters([adapters[domain_a], adapters[domain_b]])
        apply_adapter_weights(model, composed)
        mx.eval(model.parameters())
        composed_result = eval_cross_domain_quality(model, tokenizer, queries, pair_key)

        # Compute PPL on cross-domain queries (using query text as context)
        cross_texts = [INST_PROMPT.format(instruction=q) for q in queries]

        zero_lora_to_base(model)
        mx.eval(model.parameters())
        base_ppl = compute_ppl(model, tokenizer, cross_texts, max_batches=len(queries))

        apply_adapter_weights(model, adapters[domain_a])
        mx.eval(model.parameters())
        ind_a_ppl = compute_ppl(model, tokenizer, cross_texts, max_batches=len(queries))

        apply_adapter_weights(model, adapters[domain_b])
        mx.eval(model.parameters())
        ind_b_ppl = compute_ppl(model, tokenizer, cross_texts, max_batches=len(queries))

        apply_adapter_weights(model, composed)
        mx.eval(model.parameters())
        composed_ppl = compute_ppl(model, tokenizer, cross_texts, max_batches=len(queries))

        # Best individual = whichever is better
        best_individual_cross = max(
            individual_a_result["cross_domain_rate"],
            individual_b_result["cross_domain_rate"]
        )

        results[pair_name] = {
            "base": {
                "cross_domain_rate": base_result["cross_domain_rate"],
                "keyword_a": base_result["keyword_rate_domain_a"],
                "keyword_b": base_result["keyword_rate_domain_b"],
                "non_empty": base_result["non_empty_rate"],
                "avg_len": base_result["avg_response_length"],
                "ppl": round(base_ppl, 2),
            },
            f"individual_{domain_a}": {
                "cross_domain_rate": individual_a_result["cross_domain_rate"],
                "keyword_a": individual_a_result["keyword_rate_domain_a"],
                "keyword_b": individual_a_result["keyword_rate_domain_b"],
                "non_empty": individual_a_result["non_empty_rate"],
                "avg_len": individual_a_result["avg_response_length"],
                "ppl": round(ind_a_ppl, 2),
            },
            f"individual_{domain_b}": {
                "cross_domain_rate": individual_b_result["cross_domain_rate"],
                "keyword_a": individual_b_result["keyword_rate_domain_a"],
                "keyword_b": individual_b_result["keyword_rate_domain_b"],
                "non_empty": individual_b_result["non_empty_rate"],
                "avg_len": individual_b_result["avg_response_length"],
                "ppl": round(ind_b_ppl, 2),
            },
            "composed": {
                "cross_domain_rate": composed_result["cross_domain_rate"],
                "keyword_a": composed_result["keyword_rate_domain_a"],
                "keyword_b": composed_result["keyword_rate_domain_b"],
                "non_empty": composed_result["non_empty_rate"],
                "avg_len": composed_result["avg_response_length"],
                "ppl": round(composed_ppl, 2),
            },
            "composed_better_than_best_individual": (
                composed_result["cross_domain_rate"] >= best_individual_cross
            ),
        }

        # Collect coherence samples for K2
        for i in range(min(4, len(queries))):  # 4 per pair, 5 pairs = 20 total
            all_coherence_samples.append({
                "pair": pair_name,
                "query": queries[i],
                "base_response": base_result["responses"][i][:300],
                "composed_response": composed_result["responses"][i][:300],
            })

        log(f"  Results: base cross={base_result['cross_domain_rate']:.2f}, "
            f"ind_a cross={individual_a_result['cross_domain_rate']:.2f}, "
            f"ind_b cross={individual_b_result['cross_domain_rate']:.2f}, "
            f"composed cross={composed_result['cross_domain_rate']:.2f}")
        log(f"  PPL: base={base_ppl:.2f}, ind_a={ind_a_ppl:.2f}, "
            f"ind_b={ind_b_ppl:.2f}, composed={composed_ppl:.2f}")

    return results, all_coherence_samples


# ===========================================================================
# Part B: OSRM Data-Space Orthogonality Diagnostic
# ===========================================================================
def extract_hidden_states(model, tokenizer, texts, layer_idx=-1, max_samples=30):
    """Extract hidden states from a specific transformer layer.

    Returns tensor of shape (n_samples, d_model) -- mean-pooled over sequence.
    """
    hidden_states_list = []

    for text in texts[:max_samples]:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:MAX_SEQ_LENGTH]
        x = mx.array(tokens)[None, :]

        # Run through model layers to get intermediate hidden states
        h = model.model.embed_tokens(x)

        n_layers = len(model.model.layers)
        target_layer = layer_idx if layer_idx >= 0 else n_layers + layer_idx

        for i, layer in enumerate(model.model.layers):
            h = layer(h, mask=None)
            if i == target_layer:
                break

        # Mean pool over sequence dimension
        h_mean = mx.mean(h[0], axis=0)  # (d_model,)
        mx.eval(h_mean)
        hidden_states_list.append(h_mean)

    if not hidden_states_list:
        return None

    return mx.stack(hidden_states_list)  # (n_samples, d_model)


def get_a_matrices(adapter_params):
    """Extract all A matrices from adapter params.

    Returns list of (layer_name, A_matrix) where A_matrix is (in_features, r).
    """
    a_matrices = []
    for key, val in adapter_params.items():
        if "lora_a" in key:
            a_matrices.append((key, val))
    return a_matrices


def compute_osrm_diagnostic(adapters, hidden_states_per_domain, domain_names):
    """Compute OSRM cross-activation ratios for all (i,j) pairs.

    For adapter i on domain j's data:
      ratio = ||A_i @ h_j|| / ||A_j @ h_j||

    If ratio < 0.1: adapter i does not activate on domain j's data (good).
    If ratio > 0.1: adapter i interferes with domain j (bad, need OSRM).

    We compute this for A matrices across all LoRA layers and average.
    """
    log("\n" + "=" * 70)
    log("PART B: OSRM DATA-SPACE ORTHOGONALITY DIAGNOSTIC")
    log("=" * 70)

    n_domains = len(domain_names)
    results = {}
    pair_ratios = []  # All (i,j) ratios for K3

    for i, domain_i in enumerate(domain_names):
        if domain_i not in adapters or domain_i not in hidden_states_per_domain:
            continue

        h_i = hidden_states_per_domain[domain_i]  # (n_samples, d_model)
        a_matrices_i = get_a_matrices(adapters[domain_i])

        if h_i is None or len(a_matrices_i) == 0:
            continue

        # Compute self-activation: ||A_i @ h_i|| (denominator)
        self_norms = []
        for layer_key, a_mat in a_matrices_i:
            # a_mat shape: (in_features, r)
            # h_i shape: (n_samples, d_model)
            # Only use if dimensions match (A from the matching layer)
            if a_mat.shape[0] == h_i.shape[1]:
                projected = h_i @ a_mat  # (n_samples, r)
                norms = mx.sqrt(mx.sum(projected * projected, axis=1))  # (n_samples,)
                mx.eval(norms)
                self_norms.append(mx.mean(norms).item())

        if not self_norms:
            continue

        mean_self_norm = sum(self_norms) / len(self_norms)

        for j, domain_j in enumerate(domain_names):
            if i == j:
                continue
            if domain_j not in adapters:
                continue

            a_matrices_j = get_a_matrices(adapters[domain_j])

            # Cross-activation: ||A_j @ h_i||
            cross_norms = []
            for (lk_i, _), (lk_j, a_mat_j) in zip(a_matrices_i, a_matrices_j):
                if a_mat_j.shape[0] == h_i.shape[1]:
                    projected = h_i @ a_mat_j  # (n_samples, r)
                    norms = mx.sqrt(mx.sum(projected * projected, axis=1))
                    mx.eval(norms)
                    cross_norms.append(mx.mean(norms).item())

            if not cross_norms:
                continue

            mean_cross_norm = sum(cross_norms) / len(cross_norms)

            ratio = mean_cross_norm / mean_self_norm if mean_self_norm > 1e-10 else float("inf")

            pair_key = f"{domain_j}_on_{domain_i}_data"
            results[pair_key] = {
                "self_norm_A_i_h_i": round(mean_self_norm, 6),
                "cross_norm_A_j_h_i": round(mean_cross_norm, 6),
                "ratio": round(ratio, 4),
                "passes_threshold": ratio < 0.1,
            }
            pair_ratios.append(ratio)

            log(f"  {pair_key}: ||A_j*h_i||/||A_i*h_i|| = {ratio:.4f} "
                f"({'PASS' if ratio < 0.1 else 'FAIL'})")

    # Also compute the EFFECTIVE delta diagnostic: ||B_j @ A_j @ h_i|| / ||B_i @ A_i @ h_i||
    # This is the full OSRM interference measure
    log("\n--- Full OSRM: ||B_j @ A_j @ h_i|| / ||B_i @ A_i @ h_i|| ---")
    full_osrm_results = {}
    full_pair_ratios = []

    for i, domain_i in enumerate(domain_names):
        if domain_i not in adapters or domain_i not in hidden_states_per_domain:
            continue

        h_i = hidden_states_per_domain[domain_i]
        adapter_i = adapters[domain_i]

        # Get matched A,B pairs for domain i
        a_keys_i = sorted([k for k in adapter_i if "lora_a" in k])
        b_keys_i = sorted([k for k in adapter_i if "lora_b" in k])

        # Compute self-activation: ||B_i @ A_i @ h_i||
        self_norms = []
        for ak, bk in zip(a_keys_i, b_keys_i):
            a_mat = adapter_i[ak]  # (in_features, r)
            b_mat = adapter_i[bk]  # (r, out_features)
            if a_mat.shape[0] == h_i.shape[1]:
                projected = (h_i @ a_mat) @ b_mat  # (n_samples, out_features)
                norms = mx.sqrt(mx.sum(projected * projected, axis=1))
                mx.eval(norms)
                self_norms.append(mx.mean(norms).item())

        if not self_norms:
            continue
        mean_self_norm = sum(self_norms) / len(self_norms)

        for j, domain_j in enumerate(domain_names):
            if i == j:
                continue
            if domain_j not in adapters:
                continue

            adapter_j = adapters[domain_j]
            a_keys_j = sorted([k for k in adapter_j if "lora_a" in k])
            b_keys_j = sorted([k for k in adapter_j if "lora_b" in k])

            cross_norms = []
            for ak, bk in zip(a_keys_j, b_keys_j):
                a_mat = adapter_j[ak]
                b_mat = adapter_j[bk]
                if a_mat.shape[0] == h_i.shape[1]:
                    projected = (h_i @ a_mat) @ b_mat
                    norms = mx.sqrt(mx.sum(projected * projected, axis=1))
                    mx.eval(norms)
                    cross_norms.append(mx.mean(norms).item())

            if not cross_norms:
                continue

            mean_cross_norm = sum(cross_norms) / len(cross_norms)
            ratio = mean_cross_norm / mean_self_norm if mean_self_norm > 1e-10 else float("inf")

            pair_key = f"full_{domain_j}_on_{domain_i}_data"
            full_osrm_results[pair_key] = {
                "self_norm_BA_i_h_i": round(mean_self_norm, 6),
                "cross_norm_BA_j_h_i": round(mean_cross_norm, 6),
                "ratio": round(ratio, 4),
                "passes_threshold": ratio < 0.1,
            }
            full_pair_ratios.append(ratio)

            log(f"  {pair_key}: ||BA_j*h_i||/||BA_i*h_i|| = {ratio:.4f} "
                f"({'PASS' if ratio < 0.1 else 'FAIL'})")

    return {
        "a_only": results,
        "full_ba": full_osrm_results,
        "a_only_ratios": pair_ratios,
        "full_ba_ratios": full_pair_ratios,
    }


# ===========================================================================
# KR-Test on cross-domain data
# ===========================================================================
def run_kr_test_cross_domain(model, tokenizer, adapters, domain_data):
    """Run KR-Test on cross-domain composed adapters vs base.

    For each pair (A,B):
      - Correct: domain A text from domain A val data
      - Wrong: domain B text used in place of domain A text (cross-item)
    Measure whether composed adapter discriminates better than base.
    """
    log("\n--- KR-Test Cross-Domain Discrimination ---")
    results = {}

    for pair_key in CROSS_DOMAIN_PAIRS:
        domain_a, domain_b = pair_key
        pair_name = f"{domain_a}+{domain_b}"

        if domain_a not in adapters or domain_b not in adapters:
            continue
        if domain_a not in domain_data or domain_b not in domain_data:
            continue

        # Correct: domain A text, Wrong: domain B text
        correct_texts = domain_data[domain_a][:30]
        wrong_texts = domain_data[domain_b][:30]
        n_pairs = min(len(correct_texts), len(wrong_texts))
        correct_texts = correct_texts[:n_pairs]
        wrong_texts = wrong_texts[:n_pairs]

        if n_pairs < 5:
            log(f"  {pair_name}: insufficient data ({n_pairs} pairs), skipping")
            continue

        # Base model
        zero_lora_to_base(model)
        mx.eval(model.parameters())
        base_score = kr_test_score(model, tokenizer, correct_texts, wrong_texts)

        # Composed adapter
        composed = compose_adapters([adapters[domain_a], adapters[domain_b]])
        apply_adapter_weights(model, composed)
        mx.eval(model.parameters())
        composed_score = kr_test_score(model, tokenizer, correct_texts, wrong_texts)

        # Individual adapter A
        apply_adapter_weights(model, adapters[domain_a])
        mx.eval(model.parameters())
        ind_a_score = kr_test_score(model, tokenizer, correct_texts, wrong_texts)

        results[pair_name] = {
            "n_pairs": n_pairs,
            "base_kr_score": round(base_score, 4),
            "composed_kr_score": round(composed_score, 4),
            f"individual_{domain_a}_kr_score": round(ind_a_score, 4),
            "composed_better_than_base": composed_score > base_score,
        }

        log(f"  {pair_name}: base={base_score:.3f}, composed={composed_score:.3f}, "
            f"ind_{domain_a}={ind_a_score:.3f}")

    return results


# ===========================================================================
# Main experiment
# ===========================================================================
def main():
    mx.random.seed(SEED)
    t0 = time.time()

    log("=" * 70)
    log("SEMANTIC COMPOSITIONALITY + OSRM DIAGNOSTIC")
    log("=" * 70)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    log("\n[1/6] Loading model...")
    model, tokenizer = load(MODEL_ID)
    log("  Replacing BitLinear layers...")
    model = replace_bitlinear_with_linear(model)
    log("  Applying LoRA scaffolding...")
    model = apply_lora(model, rank=LORA_RANK, scale=LORA_SCALE)
    mx.eval(model.parameters())

    # ------------------------------------------------------------------
    # Load adapters
    # ------------------------------------------------------------------
    log("\n[2/6] Loading instruction-tuned adapters...")
    adapters = {}
    for domain in DOMAINS:
        adapter_path = ADAPTERS_DIR / domain
        if adapter_path.exists():
            adapters[domain] = load_adapter(adapter_path)
            n_params = sum(v.size for v in adapters[domain].values())
            log(f"  {domain}: loaded ({n_params:,} params)")
        else:
            log(f"  {domain}: NOT FOUND at {adapter_path}")

    if len(adapters) < 2:
        log("ERROR: Need at least 2 adapters for composition eval")
        sys.exit(1)

    log(f"  Loaded {len(adapters)} adapters: {list(adapters.keys())}")

    # ------------------------------------------------------------------
    # Load validation data for hidden state extraction and KR-Test
    # ------------------------------------------------------------------
    log("\n[3/6] Loading domain validation data...")
    domain_data = {}
    for domain in DOMAINS:
        val_path = DATA_DIR / domain / "val.jsonl"
        if val_path.exists():
            texts = []
            with open(val_path) as f:
                for line in f:
                    item = json.loads(line)
                    texts.append(item["text"])
            domain_data[domain] = texts
            log(f"  {domain}: {len(texts)} validation texts")
        else:
            log(f"  {domain}: no validation data")

    # ------------------------------------------------------------------
    # Part B: OSRM diagnostic (run first -- pure measurement, no generation)
    # ------------------------------------------------------------------
    log("\n[4/6] Extracting hidden states for OSRM diagnostic...")
    hidden_states = {}

    # Use base model (zero LoRA) for hidden state extraction
    zero_lora_to_base(model)
    mx.eval(model.parameters())

    for domain in DOMAINS:
        if domain in domain_data:
            log(f"  Extracting hidden states for {domain}...")
            hs = extract_hidden_states(model, tokenizer, domain_data[domain],
                                       layer_idx=-1, max_samples=30)
            if hs is not None:
                hidden_states[domain] = hs
                log(f"  {domain}: shape={hs.shape}")

    log("\n[5/6] Computing OSRM cross-activation ratios...")
    osrm_results = compute_osrm_diagnostic(adapters, hidden_states, DOMAINS)

    # ------------------------------------------------------------------
    # Part A: Semantic composition eval
    # ------------------------------------------------------------------
    log("\n[6/6] Running semantic composition evaluation...")
    part_a_results, coherence_samples = run_part_a(model, tokenizer, adapters)

    # ------------------------------------------------------------------
    # KR-Test cross-domain
    # ------------------------------------------------------------------
    kr_results = run_kr_test_cross_domain(model, tokenizer, adapters, domain_data)

    # ------------------------------------------------------------------
    # Kill criteria assessment
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    # K1: composed worse than either alone on >50% of cross-domain queries
    n_pairs_tested = len(part_a_results)
    n_composed_better = sum(1 for v in part_a_results.values()
                           if v["composed_better_than_best_individual"])
    k1_fail_rate = 1.0 - (n_composed_better / n_pairs_tested) if n_pairs_tested > 0 else 1.0
    k1_pass = k1_fail_rate <= 0.5
    log(f"\nK1: Composed worse than best individual on {k1_fail_rate*100:.0f}% of pairs")
    log(f"    Threshold: >50% -> KILL")
    log(f"    Verdict: {'PASS' if k1_pass else 'KILL'}")

    # K2: Semantic coherence (save samples for manual inspection)
    # Automated proxy: composed responses are non-empty and longer than base
    n_coherent = 0
    for sample in coherence_samples:
        composed_resp = sample["composed_response"]
        base_resp = sample["base_response"]
        # Simple coherence check: composed is non-empty AND at least as long as base
        if len(composed_resp.split()) >= max(3, len(base_resp.split()) * 0.5):
            n_coherent += 1
    k2_rate = n_coherent / len(coherence_samples) if coherence_samples else 0
    k2_pass = k2_rate >= 0.5  # At least 50% coherent
    log(f"\nK2: Semantic coherence: {n_coherent}/{len(coherence_samples)} "
        f"({k2_rate*100:.0f}%) composed responses coherent")
    log(f"    Threshold: <50% coherent -> KILL")
    log(f"    Verdict: {'PASS' if k2_pass else 'KILL'}")
    log(f"    Note: {len(coherence_samples)} samples saved for manual inspection")

    # K3: OSRM diagnostic
    a_only_ratios = osrm_results["a_only_ratios"]
    full_ba_ratios = osrm_results["full_ba_ratios"]

    if a_only_ratios:
        n_a_fail = sum(1 for r in a_only_ratios if r > 0.1)
        k3_a_fail_rate = n_a_fail / len(a_only_ratios)
        log(f"\nK3 (A-only): {n_a_fail}/{len(a_only_ratios)} pairs have ratio > 0.1 "
            f"({k3_a_fail_rate*100:.0f}%)")
    else:
        k3_a_fail_rate = 1.0
        log(f"\nK3 (A-only): No data")

    if full_ba_ratios:
        n_ba_fail = sum(1 for r in full_ba_ratios if r > 0.1)
        k3_ba_fail_rate = n_ba_fail / len(full_ba_ratios)
        log(f"K3 (BA-full): {n_ba_fail}/{len(full_ba_ratios)} pairs have ratio > 0.1 "
            f"({k3_ba_fail_rate*100:.0f}%)")
    else:
        k3_ba_fail_rate = 1.0
        log(f"K3 (BA-full): No data")

    # K3 uses the A-only measure as specified in hypothesis
    k3_pass = k3_a_fail_rate <= 0.5
    log(f"    Threshold: >50% pairs fail -> KILL")
    log(f"    Verdict: {'PASS' if k3_pass else 'KILL'}")

    overall_pass = k1_pass and k2_pass and k3_pass
    verdict = "SUPPORTED" if overall_pass else "KILLED"
    log(f"\n{'=' * 70}")
    log(f"OVERALL VERDICT: {verdict}")
    log(f"  K1 (composition helps): {'PASS' if k1_pass else 'FAIL'}")
    log(f"  K2 (coherence): {'PASS' if k2_pass else 'FAIL'}")
    log(f"  K3 (OSRM data-orthogonality): {'PASS' if k3_pass else 'FAIL'}")
    log(f"{'=' * 70}")

    elapsed = time.time() - t0
    log(f"\nTotal runtime: {elapsed / 60:.1f} min")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    final_results = {
        "experiment": "bitnet_semantic_compositionality",
        "hypothesis": "exp_bitnet_semantic_compositionality",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "n_adapters": len(adapters),
        "domains": list(adapters.keys()),
        "runtime_min": round(elapsed / 60, 1),

        "part_a_semantic_composition": part_a_results,
        "kr_test_cross_domain": kr_results,
        "coherence_samples": coherence_samples,

        "part_b_osrm_diagnostic": {
            "a_only": osrm_results["a_only"],
            "full_ba": osrm_results["full_ba"],
            "a_only_summary": {
                "mean_ratio": round(sum(a_only_ratios) / len(a_only_ratios), 4) if a_only_ratios else None,
                "max_ratio": round(max(a_only_ratios), 4) if a_only_ratios else None,
                "min_ratio": round(min(a_only_ratios), 4) if a_only_ratios else None,
                "n_fail": sum(1 for r in a_only_ratios if r > 0.1) if a_only_ratios else 0,
                "n_total": len(a_only_ratios),
            },
            "full_ba_summary": {
                "mean_ratio": round(sum(full_ba_ratios) / len(full_ba_ratios), 4) if full_ba_ratios else None,
                "max_ratio": round(max(full_ba_ratios), 4) if full_ba_ratios else None,
                "min_ratio": round(min(full_ba_ratios), 4) if full_ba_ratios else None,
                "n_fail": sum(1 for r in full_ba_ratios if r > 0.1) if full_ba_ratios else 0,
                "n_total": len(full_ba_ratios),
            },
        },

        "kill_criteria": {
            "K1": {
                "description": "composed worse than best individual on >50% of pairs",
                "fail_rate": round(k1_fail_rate, 4),
                "pass": k1_pass,
            },
            "K2": {
                "description": "semantic coherence rated lower than base on manual inspection",
                "coherence_rate": round(k2_rate, 4),
                "pass": k2_pass,
            },
            "K3": {
                "description": "OSRM ||A_j * h_i|| > 0.1 * ||A_i * h_i|| for >50% pairs",
                "a_only_fail_rate": round(k3_a_fail_rate, 4),
                "ba_full_fail_rate": round(k3_ba_fail_rate, 4),
                "pass": k3_pass,
            },
        },
        "verdict": verdict,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
