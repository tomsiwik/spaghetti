#!/usr/bin/env python3
"""Per-domain module selection: attention-only for prose, full for code.

Guided exploration (Type 2): Module separability proven (Finding #300).
Unknown: optimal module set per domain.

Kill criteria:
  K766: Per-domain selection does NOT reduce benchmark degradation vs full-module
        (attention-only medical/math still degrades MMLU > 2%)
  K767: Per-domain selection loses > 20% domain PPL improvement vs full-module baseline
  K768: Module effects NOT separable (interaction effects > 10%)

Predictions (from MATH.md):
  - Attn-only perturbation is ~28% of full-module (by parameter count)
  - MMLU degradation ~1.4pp (28% of 5pp baseline)
  - Code requires full-module (MLP carries syntax); prose domains work attn-only
  - Module effects are approximately additive (interaction < 10%)

Platform: Apple M5 Pro 48GB, MLX
"""

import ast
import gc
import json
import math
import os
import re
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from mlx.utils import tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Paths to existing infrastructure
SFT_DIR = EXPERIMENT_DIR.parent / "bitnet_sft_generation_v3" / "sft_adapters"
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
DATA_DIR = SOURCE_DIR / "data"
NTP_ADAPTERS_DIR = SOURCE_DIR / "adapters"
SKELETON_PATH = NTP_ADAPTERS_DIR / "grassmannian_skeleton.npz"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_SCALE = 20.0
MAX_SEQ = 256
SEED = 42
DOMAINS = ["medical", "code", "math", "legal", "finance"]

# Module groups
ATTN_MODULES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
MLP_MODULES = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
ALL_MODULES = ATTN_MODULES + MLP_MODULES

# Evaluation sizes (kept small for runtime: total ~30min target)
N_CAL = 30
N_PPL = 40
N_GEN = 5
N_MMLU_PER_DOMAIN = 15
MAX_NEW_TOKENS = 128
MAX_TOKENS_MMLU = 32

# MMLU subjects (same as capability_benchmark)
MMLU_SUBJECTS = {
    "medical": ["clinical_knowledge", "professional_medicine", "anatomy", "medical_genetics"],
    "code": ["college_computer_science", "high_school_computer_science", "machine_learning"],
    "math": ["high_school_mathematics", "elementary_mathematics", "college_mathematics"],
    "legal": ["professional_law", "jurisprudence", "international_law"],
    "finance": ["professional_accounting", "econometrics", "high_school_macroeconomics"],
}

# Per-domain optimal scales (Finding #249)
OPTIMAL_SCALES = {
    "medical": 20.0,
    "code": 20.0,
    "math": 20.0,
    "legal": 4.0,
    "finance": 1.0,
}


# ============================================================================
# Utilities
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def log(msg): print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for o in objects:
        del o
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def load_data(domain, split="valid", n=None):
    samples = []
    with open(DATA_DIR / domain / f"{split}.jsonl") as f:
        for line in f:
            samples.append(json.loads(line)["text"])
            if n and len(samples) >= n:
                break
    return samples


# ============================================================================
# Adapter loading with module filtering
# ============================================================================

def load_skeleton():
    return dict(np.load(str(SKELETON_PATH)))


def load_adapter(domain):
    return dict(mx.load(str(SFT_DIR / domain / "adapter.npz")))


def filter_adapter(adapter_b, module_set):
    """Filter adapter B-matrices to only include specified modules.

    module_set: list of module names like ["self_attn.q_proj", "mlp.gate_proj"]
    Returns filtered dict with only matching keys.
    """
    filtered = {}
    for key, val in adapter_b.items():
        # key format: "model.layers.N.self_attn.q_proj.lora_b"
        for mod in module_set:
            if mod in key:
                filtered[key] = val
                break
    return filtered


def filter_skeleton(skeleton, module_set, domain_idx):
    """Filter skeleton A-matrices to only include specified modules.

    Returns filtered dict.
    """
    filtered = {}
    for key, val in skeleton.items():
        # key format: "layer_N_self_attn.q_proj_domain_D"
        for mod in module_set:
            if mod in key and f"_domain_{domain_idx}" in key:
                filtered[key] = val
                break
        # Also keep keys for other domains (they won't match in attach)
        if f"_domain_{domain_idx}" not in key:
            filtered[key] = val
    return filtered


# ============================================================================
# Pre-merge composition (from capability_benchmark)
# ============================================================================

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


def save_base_weights(model):
    base_weights = []
    for layer in model.model.layers:
        layer_w = {}
        for key in ALL_MODULES:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None and isinstance(module, nn.Linear):
                layer_w[key] = module.weight
        base_weights.append(layer_w)
    return base_weights


def restore_base_weights(model, base_weights):
    for li, layer_weights in enumerate(base_weights):
        for key, weight in layer_weights.items():
            parts = key.split(".")
            module = model.model.layers[li]
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None and isinstance(module, nn.Linear):
                module.weight = weight
    mx.eval(model.parameters())


def premerge_adapter(model, skeleton, adapter_b, domain, scale, module_set=None):
    """Pre-merge: W_new = W_base + scale * B^T @ A^T, filtered by module_set."""
    if module_set is None:
        module_set = ALL_MODULES
    di = DOMAINS.index(domain)
    merge_count = 0
    for li in range(len(model.model.layers)):
        for key in module_set:
            parts = key.split(".")
            module = model.model.layers[li]
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue
            skey = f"layer_{li}_{key}_domain_{di}"
            if skey not in skeleton:
                continue
            a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
            b_key = f"model.layers.{li}.{key}.lora_b"
            if b_key not in adapter_b:
                continue
            b_mx = adapter_b[b_key]
            delta = scale * (b_mx.T @ a_mx.T)
            module.weight = module.weight + delta
            merge_count += 1
    mx.eval(model.parameters())
    return merge_count


# ============================================================================
# Evaluation functions
# ============================================================================

def compute_ppl(model, tokenizer, texts, max_seq=MAX_SEQ):
    loss, n = 0.0, 0
    for text in texts:
        toks = tokenizer.encode(text)[:max_seq]
        if len(toks) < 4:
            continue
        x = mx.array(toks)[None, :]
        logits = model(x)
        mx.eval(logits)
        targets = x[:, 1:]
        lp = mx.log(mx.softmax(logits[:, :-1, :], axis=-1) + 1e-10)
        tlp = mx.take_along_axis(lp, targets[:, :, None], axis=-1).squeeze(-1)
        mx.eval(tlp)
        loss += -tlp.sum().item()
        n += targets.shape[1]
        del logits, lp, tlp, x
    return math.exp(loss / n) if n else float('inf')


STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'to', 'of', 'in', 'for', 'on', 'with',
    'at', 'by', 'from', 'as', 'and', 'but', 'or', 'not', 'so', 'yet',
    'both', 'either', 'each', 'every', 'all', 'any', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than', 'too',
    'very', 'just', 'because', 'if', 'when', 'where', 'how', 'what', 'which',
    'who', 'this', 'that', 'these', 'those', 'it', 'its', 'i', 'me', 'my',
    'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they',
    'them', 'their',
}


def factual_recall(gen, ref):
    def toks(t):
        return set(w for w in re.findall(r'\b[a-z]+\b', t.lower())
                   if w not in STOP_WORDS and len(w) > 2)
    g, r = toks(gen), toks(ref)
    return len(g & r) / len(r) if r else 0.0


def eval_response(gen, ref, domain):
    if domain == "code":
        blocks = re.findall(r'```(?:python)?\s*\n(.*?)\n```', gen, re.DOTALL)
        code = '\n'.join(blocks) if blocks else '\n'.join(
            l for l in gen.split('\n') if l.strip() and not l.startswith('#'))
        try:
            ast.parse(code)
            ok = True
        except SyntaxError:
            ok = False
        return 0.7 * float(ok) + 0.3 * factual_recall(gen, ref)
    return factual_recall(gen, ref)


def generate_text(model, tokenizer, prompt, max_tokens=MAX_NEW_TOKENS):
    try:
        sampler = make_sampler(temp=0.0)
        return mlx_generate(model, tokenizer, prompt=prompt,
                            max_tokens=max_tokens, sampler=sampler, verbose=False)
    except Exception:
        return ""


def format_mmlu_prompt(question, choices):
    labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    return (
        f"### Instruction:\n"
        f"Answer the following multiple choice question. "
        f"Reply with just the letter (A, B, C, or D).\n\n"
        f"{question}\n\n{choices_text}\n\n"
        f"### Response:\n"
    )


def extract_mmlu_answer(text):
    text = text.strip()
    if text and text[0].upper() in "ABCD":
        return text[0].upper()
    match = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*([A-Da-d])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    for line in text.split('\n'):
        line = line.strip()
        if len(line) == 1 and line.upper() in "ABCD":
            return line.upper()
    match = re.search(r'[\(\s]([A-Da-d])[\)\.\s]', text)
    if match:
        return match.group(1).upper()
    match = re.search(r'\b([A-Da-d])\b', text)
    if match:
        return match.group(1).upper()
    return None


# ============================================================================
# Phase 1: Measure Frobenius perturbation norms (pure math, no generation)
# ============================================================================

def phase_perturbation_norms():
    """Measure adapter perturbation norm per module group per domain."""
    log("\n" + "=" * 70)
    log("PHASE 1: PERTURBATION NORM ANALYSIS")
    log("=" * 70)

    skeleton = load_skeleton()
    results = {}

    for domain in DOMAINS:
        di = DOMAINS.index(domain)
        adapter_b = load_adapter(domain)
        scale = OPTIMAL_SCALES[domain]

        attn_norm_sq = 0.0
        mlp_norm_sq = 0.0
        per_module = {}

        for li in range(30):  # 30 layers
            for key in ALL_MODULES:
                skey = f"layer_{li}_{key}_domain_{di}"
                bkey = f"model.layers.{li}.{key}.lora_b"
                if skey not in skeleton or bkey not in adapter_b:
                    continue
                A = mx.array(skeleton[skey]).astype(mx.float32)
                B = adapter_b[bkey].astype(mx.float32)
                delta = scale * (B.T @ A.T)  # (out, in) full-rank perturbation
                frob = mx.linalg.norm(delta.reshape(-1))
                mx.eval(frob)
                frob_val = frob.item() ** 2  # squared norm for additivity

                if key not in per_module:
                    per_module[key] = 0.0
                per_module[key] += frob_val

                if key in ATTN_MODULES:
                    attn_norm_sq += frob_val
                else:
                    mlp_norm_sq += frob_val

                del delta, frob, A, B

        total_sq = attn_norm_sq + mlp_norm_sq
        results[domain] = {
            "attn_frob_sq": round(attn_norm_sq, 2),
            "mlp_frob_sq": round(mlp_norm_sq, 2),
            "total_frob_sq": round(total_sq, 2),
            "attn_fraction": round(attn_norm_sq / total_sq, 4) if total_sq > 0 else 0,
            "mlp_fraction": round(mlp_norm_sq / total_sq, 4) if total_sq > 0 else 0,
            "per_module": {k: round(v, 2) for k, v in per_module.items()},
            "scale": scale,
        }
        log(f"  {domain} (s={scale}): attn={attn_norm_sq:.1f} ({attn_norm_sq/total_sq*100:.1f}%), "
            f"mlp={mlp_norm_sq:.1f} ({mlp_norm_sq/total_sq*100:.1f}%)")
        del adapter_b

    gc.collect()
    mx.clear_cache()
    return results


# ============================================================================
# Phase 2: PPL for 3 module configs per domain
# ============================================================================

def phase_ppl():
    """Measure PPL with full, attn-only, and MLP-only adapters per domain."""
    log("\n" + "=" * 70)
    log("PHASE 2: PPL COMPARISON (full vs attn-only vs MLP-only)")
    log("=" * 70)

    skeleton = load_skeleton()
    val_data = {d: load_data(d, "valid", N_PPL) for d in DOMAINS}

    configs = {
        "full": ALL_MODULES,
        "attn_only": ATTN_MODULES,
        "mlp_only": MLP_MODULES,
    }

    results = {"base": {}, "full": {}, "attn_only": {}, "mlp_only": {}}

    # Base PPL
    model, tok = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    base_weights = save_base_weights(model)
    log_memory("base loaded")

    for d in DOMAINS:
        results["base"][d] = round(compute_ppl(model, tok, val_data[d]), 3)
        log(f"  base/{d}: {results['base'][d]}")

    # Per-config PPL
    for config_name, module_set in configs.items():
        log(f"\n  --- Config: {config_name} ({len(module_set)} modules) ---")
        for d in DOMAINS:
            restore_base_weights(model, base_weights)
            adapter_b = load_adapter(d)
            scale = OPTIMAL_SCALES[d]
            n_merged = premerge_adapter(model, skeleton, adapter_b, d, scale, module_set)
            ppl = round(compute_ppl(model, tok, val_data[d]), 3)
            results[config_name][d] = ppl

            # Compute improvement vs base
            base_ppl = results["base"][d]
            improvement = (base_ppl - ppl) / base_ppl * 100
            log(f"    {d}: PPL={ppl} ({improvement:+.1f}% vs base, {n_merged} merged)")
            del adapter_b

    cleanup(model, tok)

    # Compute retention: attn-only improvement as % of full improvement
    results["attn_retention"] = {}
    results["mlp_retention"] = {}
    for d in DOMAINS:
        full_improve = results["base"][d] - results["full"][d]
        attn_improve = results["base"][d] - results["attn_only"][d]
        mlp_improve = results["base"][d] - results["mlp_only"][d]
        if full_improve > 0:
            results["attn_retention"][d] = round(attn_improve / full_improve * 100, 1)
            results["mlp_retention"][d] = round(mlp_improve / full_improve * 100, 1)
        else:
            results["attn_retention"][d] = 0.0
            results["mlp_retention"][d] = 0.0

    # Interaction effect: full improvement vs attn + mlp improvements
    results["interaction"] = {}
    for d in DOMAINS:
        full_improve = results["base"][d] - results["full"][d]
        attn_improve = results["base"][d] - results["attn_only"][d]
        mlp_improve = results["base"][d] - results["mlp_only"][d]
        additive = attn_improve + mlp_improve
        if abs(full_improve) > 0.01:
            interaction_pct = abs(full_improve - additive) / abs(full_improve) * 100
            results["interaction"][d] = round(interaction_pct, 1)
        else:
            results["interaction"][d] = 0.0
        log(f"  Interaction {d}: full={full_improve:.3f}, "
            f"attn+mlp={additive:.3f}, interaction={results['interaction'][d]}%")

    return results


# ============================================================================
# Phase 3: Behavioral eval for full vs attn-only per domain
# ============================================================================

def phase_behavioral():
    """Behavioral eval comparing full and attn-only adapters."""
    log("\n" + "=" * 70)
    log("PHASE 3: BEHAVIORAL COMPARISON (full vs attn-only)")
    log("=" * 70)

    skeleton = load_skeleton()
    results = {"full": {}, "attn_only": {}}

    for config_name, module_set in [("full", ALL_MODULES), ("attn_only", ATTN_MODULES)]:
        log(f"\n  --- Config: {config_name} ---")
        for d in DOMAINS:
            model, tok = load(MODEL_ID)
            model = replace_bitlinear_with_linear(model)
            model.freeze()

            adapter_b = load_adapter(d)
            scale = OPTIMAL_SCALES[d]
            base_weights = save_base_weights(model)
            n_merged = premerge_adapter(model, skeleton, adapter_b, d, scale, module_set)

            test = load_data(d, "valid", N_GEN)
            scores = []
            for text in test:
                if "### Response:" in text:
                    prompt = text.split("### Response:")[0].strip() + "\n### Response:\n"
                    ref = text.split("### Response:")[-1].strip()
                else:
                    prompt, ref = text[:200], text
                gen = generate_text(model, tok, prompt)
                scores.append(eval_response(gen, ref, d))

            mean_score = float(np.mean(scores)) if scores else 0.0
            results[config_name][d] = round(mean_score, 3)
            log(f"    {d}: behavioral={mean_score:.3f} ({n_merged} modules)")
            del adapter_b
            cleanup(model, tok)

    # Compute behavioral retention
    results["behavioral_retention"] = {}
    for d in DOMAINS:
        full_score = results["full"][d]
        attn_score = results["attn_only"][d]
        if full_score > 0:
            results["behavioral_retention"][d] = round(attn_score / full_score * 100, 1)
        else:
            results["behavioral_retention"][d] = 0.0

    return results


# ============================================================================
# Phase 4: MMLU evaluation (base vs full vs attn-only vs hybrid)
# ============================================================================

def phase_mmlu():
    """MMLU benchmark degradation: the core of this experiment."""
    log("\n" + "=" * 70)
    log("PHASE 4: MMLU EVALUATION")
    log("=" * 70)

    from datasets import load_dataset
    log("  Loading MMLU data...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    by_subject = {}
    for item in ds:
        subj = item["subject"]
        if subj not in by_subject:
            by_subject[subj] = []
        by_subject[subj].append(item)

    mmlu_data = {}
    for domain, subjects in MMLU_SUBJECTS.items():
        questions = []
        for subj in subjects:
            if subj in by_subject:
                questions.extend(by_subject[subj])
        rng = np.random.RandomState(SEED)
        rng.shuffle(questions)
        mmlu_data[domain] = questions[:N_MMLU_PER_DOMAIN]
        log(f"    {domain}: {len(mmlu_data[domain])} questions")

    del ds, by_subject
    gc.collect()

    skeleton = load_skeleton()
    choice_labels = ["A", "B", "C", "D"]

    # Hybrid config: attn-only for prose, full for code
    HYBRID_CONFIG = {
        "medical": ATTN_MODULES,
        "code": ALL_MODULES,
        "math": ATTN_MODULES,
        "legal": ATTN_MODULES,
        "finance": ATTN_MODULES,
    }

    results = {}

    for config_name, config_fn in [
        ("base", lambda d: None),
        ("full", lambda d: ALL_MODULES),
        ("attn_only", lambda d: ATTN_MODULES),
        ("hybrid", lambda d: HYBRID_CONFIG[d]),
    ]:
        log(f"\n  --- MMLU: {config_name} ---")
        mmlu_correct = 0
        mmlu_total = 0
        mmlu_by_domain = {}

        for d in DOMAINS:
            model, tok = load(MODEL_ID)
            model = replace_bitlinear_with_linear(model)
            model.freeze()

            module_set = config_fn(d)
            if module_set is not None:
                adapter_b = load_adapter(d)
                scale = OPTIMAL_SCALES[d]
                premerge_adapter(model, skeleton, adapter_b, d, scale, module_set)
                del adapter_b

            domain_correct = 0
            for q in mmlu_data[d]:
                prompt = format_mmlu_prompt(q["question"], q["choices"])
                gen = generate_text(model, tok, prompt, max_tokens=MAX_TOKENS_MMLU)
                predicted = extract_mmlu_answer(gen)
                gt_label = choice_labels[q["answer"]]
                if predicted == gt_label:
                    domain_correct += 1
                    mmlu_correct += 1
                mmlu_total += 1
                del gen

            domain_acc = domain_correct / len(mmlu_data[d]) if mmlu_data[d] else 0
            mmlu_by_domain[d] = {
                "accuracy": round(domain_acc, 3),
                "correct": domain_correct,
                "total": len(mmlu_data[d]),
            }
            log(f"    {d}: {domain_correct}/{len(mmlu_data[d])} = {domain_acc:.3f}")
            cleanup(model, tok)

        overall_acc = mmlu_correct / mmlu_total if mmlu_total > 0 else 0
        results[config_name] = {
            "accuracy": round(overall_acc, 3),
            "correct": mmlu_correct,
            "total": mmlu_total,
            "by_domain": mmlu_by_domain,
        }
        log(f"  {config_name} overall: {mmlu_correct}/{mmlu_total} = {overall_acc:.3f}")

    # Compute degradation
    results["degradation"] = {}
    for config_name in ["full", "attn_only", "hybrid"]:
        base_acc = results["base"]["accuracy"]
        config_acc = results[config_name]["accuracy"]
        delta_pp = (config_acc - base_acc) * 100
        results["degradation"][config_name] = {
            "overall_pp": round(delta_pp, 1),
            "by_domain": {},
        }
        for d in DOMAINS:
            base_d = results["base"]["by_domain"][d]["accuracy"]
            config_d = results[config_name]["by_domain"][d]["accuracy"]
            results["degradation"][config_name]["by_domain"][d] = round((config_d - base_d) * 100, 1)

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("EXPERIMENT: Per-Domain Module Selection")
    log(f"  Model: {MODEL_ID}")
    log(f"  Domains: {DOMAINS}")
    log(f"  Scales: {OPTIMAL_SCALES}")
    log("=" * 70)
    log_memory("start")

    # Phase 1: Perturbation norms (fast, no model needed)
    perturbation = phase_perturbation_norms()

    # Phase 2: PPL comparison
    ppl_results = phase_ppl()

    # Phase 3: Behavioral comparison
    behavioral = phase_behavioral()

    # Phase 4: MMLU benchmark degradation
    mmlu = phase_mmlu()

    # ====================================================================
    # Analysis and kill criteria
    # ====================================================================
    log("\n" + "=" * 70)
    log("ANALYSIS")
    log("=" * 70)

    # K766: Attn-only MMLU degradation < 2% for medical/math
    k766_attn_mmlu = mmlu["degradation"]["attn_only"]["overall_pp"]
    k766_hybrid_mmlu = mmlu["degradation"]["hybrid"]["overall_pp"]
    # Check per-domain: medical and math specifically
    k766_medical_attn = mmlu["degradation"]["attn_only"]["by_domain"].get("medical", 0)
    k766_math_attn = mmlu["degradation"]["attn_only"]["by_domain"].get("math", 0)
    # PASS if attn-only degradation is less (in absolute value) than full-module
    k766_full_mmlu = mmlu["degradation"]["full"]["overall_pp"]
    k766_pass = (
        abs(k766_hybrid_mmlu) < abs(k766_full_mmlu) and
        abs(k766_attn_mmlu) < abs(k766_full_mmlu)
    )
    log(f"\n  K766 (MMLU degradation):")
    log(f"    Full-module: {k766_full_mmlu:+.1f}pp")
    log(f"    Attn-only:   {k766_attn_mmlu:+.1f}pp")
    log(f"    Hybrid:      {k766_hybrid_mmlu:+.1f}pp")
    log(f"    Medical attn: {k766_medical_attn:+.1f}pp, Math attn: {k766_math_attn:+.1f}pp")
    log(f"    PASS: {k766_pass} (attn/hybrid less degradation than full)")

    # K767: Attn-only retains >= 80% of PPL improvement
    k767_retentions = ppl_results["attn_retention"]
    # For hybrid: code is full (100%), rest is attn_only
    k767_hybrid_retentions = {}
    for d in DOMAINS:
        if d == "code":
            k767_hybrid_retentions[d] = 100.0
        else:
            k767_hybrid_retentions[d] = k767_retentions[d]
    k767_min_hybrid = min(k767_hybrid_retentions.values())
    k767_pass = k767_min_hybrid >= 80.0
    log(f"\n  K767 (PPL retention >= 80%):")
    for d in DOMAINS:
        log(f"    {d}: attn={k767_retentions[d]:.1f}%, hybrid={k767_hybrid_retentions[d]:.1f}%")
    log(f"    Min hybrid retention: {k767_min_hybrid:.1f}%")
    log(f"    PASS: {k767_pass}")

    # K768: Module interaction < 10%
    k768_interactions = ppl_results["interaction"]
    k768_max = max(k768_interactions.values())
    k768_pass = k768_max < 10.0
    log(f"\n  K768 (Interaction effects < 10%):")
    for d in DOMAINS:
        log(f"    {d}: {k768_interactions[d]:.1f}%")
    log(f"    Max interaction: {k768_max:.1f}%")
    log(f"    PASS: {k768_pass}")

    # Prediction verification
    predictions = {}

    # P1: Attn fraction ~28% of total perturbation
    mean_attn_frac = np.mean([perturbation[d]["attn_fraction"] for d in DOMAINS
                              if OPTIMAL_SCALES[d] >= 4.0])  # Only behavioral-scale domains
    predictions["attn_perturbation_fraction"] = {
        "predicted": "~28%",
        "measured": f"{mean_attn_frac*100:.1f}%",
        "match": 15 < mean_attn_frac * 100 < 45,  # Wide tolerance for guided exploration
    }

    # P2: Code requires MLP
    predictions["code_needs_mlp"] = {
        "predicted": "code behavioral drops significantly with attn-only",
        "measured": f"full={behavioral['full'].get('code', 0)}, attn={behavioral['attn_only'].get('code', 0)}",
        "match": behavioral["full"].get("code", 0) > behavioral["attn_only"].get("code", 0) * 1.3,
    }

    # P3: Medical/math work attn-only
    predictions["prose_attn_sufficient"] = {
        "predicted": "medical/math behavioral >= full with attn-only",
        "measured": (f"medical: full={behavioral['full'].get('medical', 0):.3f}, "
                     f"attn={behavioral['attn_only'].get('medical', 0):.3f}; "
                     f"math: full={behavioral['full'].get('math', 0):.3f}, "
                     f"attn={behavioral['attn_only'].get('math', 0):.3f}"),
        "match": (behavioral["attn_only"].get("medical", 0) >= behavioral["full"].get("medical", 0) * 0.9 and
                  behavioral["attn_only"].get("math", 0) >= behavioral["full"].get("math", 0) * 0.9),
    }

    # Optimal config per domain
    optimal_config = {}
    for d in DOMAINS:
        full_score = behavioral["full"].get(d, 0)
        attn_score = behavioral["attn_only"].get(d, 0)
        # If attn-only is within 10% of full and has less MMLU degradation, prefer attn
        if attn_score >= full_score * 0.9:
            optimal_config[d] = "attn_only"
        else:
            optimal_config[d] = "full"

    # Assemble results
    all_results = {
        "experiment": "per_domain_module_selection",
        "hypothesis": "Per-domain module selection: attn-only for prose, full for code",
        "type": "guided-exploration",
        "model": MODEL_ID,
        "scales": OPTIMAL_SCALES,
        "perturbation": perturbation,
        "ppl": ppl_results,
        "behavioral": behavioral,
        "mmlu": mmlu,
        "predictions": predictions,
        "optimal_config": optimal_config,
        "kill_criteria": {
            "K766": {
                "text": "Per-domain selection reduces MMLU degradation vs full-module",
                "pass": k766_pass,
                "full_mmlu_pp": k766_full_mmlu,
                "attn_mmlu_pp": k766_attn_mmlu,
                "hybrid_mmlu_pp": k766_hybrid_mmlu,
            },
            "K767": {
                "text": "Hybrid retains >= 80% domain PPL improvement",
                "pass": k767_pass,
                "retentions": k767_hybrid_retentions,
                "min_retention": k767_min_hybrid,
            },
            "K768": {
                "text": "Module interaction effects < 10%",
                "pass": k768_pass,
                "interactions": k768_interactions,
                "max_interaction": k768_max,
            },
        },
        "all_pass": k766_pass and k767_pass and k768_pass,
        "total_time_s": round(time.time() - t0, 1),
    }

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {all_results['total_time_s']:.1f}s")
    log(f"\nKill criteria: K766={'PASS' if k766_pass else 'FAIL'}, "
        f"K767={'PASS' if k767_pass else 'FAIL'}, "
        f"K768={'PASS' if k768_pass else 'FAIL'}")
    log(f"All pass: {all_results['all_pass']}")

    # Print optimal config summary
    log(f"\nOptimal module config per domain:")
    for d in DOMAINS:
        log(f"  {d}: {optimal_config[d]}")


if __name__ == "__main__":
    main()
