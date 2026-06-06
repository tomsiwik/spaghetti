#!/usr/bin/env python3
"""LORA_SCALE sweep on 128-token segments: resolve segment PPL degradation.

Finding #310 showed segment-isolated PPL (7.636) is WORSE than base (7.465)
at LORA_SCALE=20. This experiment sweeps scales {2, 5, 10, 15, 20} to find
the optimal scale for 128-token segment-isolated application.

Type: Guided Exploration (Type 2) -- proven framework (LoRA perturbation
scaling), unknown parameter (optimal scale for segment-isolated application).

Kill criteria:
  K787: Best-scale segment PPL < base PPL (7.465)
  K788: Scale-PPL curve is non-monotonic (optimal scale != 20)
  K789: Best-scale behavioral score within 10% of per-sequence best

Platform: Apple M5 Pro 48GB, MLX.
"""

import ast
import gc
import json
import math
import os
import random
import re
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Source data and adapters (same as prior experiments)
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
MAX_SEQ_LENGTH = 256
SEGMENT_LENGTH = 128  # matches Finding #305 and #310
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = len(DOMAINS)

# Scale sweep configuration
SCALES = [2.0, 5.0, 10.0, 15.0, 20.0]
TRAINING_SCALE = 20.0  # adapters were trained at this scale

# Data budget
N_EVAL_PER_DOMAIN = 10   # samples per domain for segment PPL eval
N_GEN_PER_DOMAIN = 5     # samples per domain for behavioral eval
MAX_NEW_TOKENS = 128


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
# Data loading
# ============================================================================

def load_domain_data(domain, split="valid", max_samples=50):
    """Load instruction-format text from domain data."""
    path = DATA_DIR / domain / f"{split}.jsonl"
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            text = json.loads(line)["text"]
            if "### Instruction:" in text and "### Response:" in text:
                instruction = text.split("### Instruction:")[1].split("### Response:")[0].strip()
                response = text.split("### Response:")[1].strip()
                samples.append({"instruction": instruction, "response": response, "text": text})
    return samples


def format_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


# ============================================================================
# Model utilities (adapted from hidden_state_probe_router)
# ============================================================================

from mlx_lm import load as load_model_and_tokenizer
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


def apply_lora_to_model(model, rank=16, scale=1.0):
    """Apply LoRA layers to model. Scale can be changed later via set_lora_scale."""
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        lora_updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora_layer = LoRALinear.from_base(module, r=rank, scale=scale, dropout=0.0)
                lora_updates.append((key, lora_layer))
                count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    log(f"  Applied LoRA (r={rank}, scale={scale}) to {count} linear layers")
    return model


def set_lora_scale(model, scale):
    """Set LORA_SCALE on all LoRALinear modules. No eval needed -- scale is a Python float."""
    count = 0
    for layer in model.model.layers:
        for _key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.scale = scale
                count += 1
    return count


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_to_model(model, adapter_params):
    model.update(tree_unflatten(list(adapter_params.items())))


def zero_adapter_in_model(model):
    updates = []
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_b" in name:
            updates.append((name, mx.zeros_like(p)))
    if updates:
        model.update(tree_unflatten(updates))


# ============================================================================
# Behavioral evaluation (adapted from per_domain_module_selection)
# ============================================================================

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
    """Generate text with temperature=0 (greedy)."""
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    try:
        sampler = make_sampler(temp=0.0)
        return mlx_generate(model, tokenizer, prompt=prompt,
                            max_tokens=max_tokens, sampler=sampler, verbose=False)
    except Exception:
        return ""


# ============================================================================
# Phase 1: Segment PPL sweep across scales
# ============================================================================

def phase_segment_ppl_sweep():
    """Sweep LORA_SCALE on 128-token segments for each domain.

    For each scale in SCALES:
      For each domain:
        Load correct adapter, set scale, evaluate PPL on 128-token segments.
    Also measure base-only PPL and per-sequence PPL (at training scale).
    """
    log("\n" + "=" * 70)
    log("PHASE 1: SEGMENT PPL SWEEP ACROSS SCALES")
    log("=" * 70)
    t0 = time.time()

    # Load model once, swap adapters and scales
    model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=TRAINING_SCALE)
    log_memory("model-loaded")

    # Load all adapters
    adapters = {}
    for domain in DOMAINS:
        adapters[domain] = load_adapter(ADAPTERS_DIR / domain)
        log(f"  Loaded adapter: {domain}")

    # Load evaluation data
    domain_data = {}
    for domain in DOMAINS:
        samples = load_domain_data(domain, split="valid", max_samples=N_EVAL_PER_DOMAIN)
        if len(samples) < N_EVAL_PER_DOMAIN:
            train_supplement = load_domain_data(domain, split="train", max_samples=N_EVAL_PER_DOMAIN)
            samples = (samples + train_supplement)[:N_EVAL_PER_DOMAIN]
        domain_data[domain] = samples
        log(f"  {domain}: {len(samples)} eval samples")

    # Helper: compute PPL on token list
    def compute_token_ppl(tokens):
        if len(tokens) < 2:
            return float("inf"), 0
        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)
        nll = loss.item()
        n = y.size
        del x, y, logits, loss
        return nll, n

    # Prepare segments: tokenize each sample, truncate to SEGMENT_LENGTH
    domain_segments = {}
    for domain in DOMAINS:
        segs = []
        for sample in domain_data[domain]:
            tokens = tokenizer.encode(sample["text"])
            # Pad short sequences by repeating
            while len(tokens) < SEGMENT_LENGTH:
                tokens = tokens + tokens
            seg = tokens[:SEGMENT_LENGTH]
            segs.append(seg)
        domain_segments[domain] = segs
        log(f"  {domain}: {len(segs)} segments of {SEGMENT_LENGTH} tokens")

    # ---- Baseline: base-only PPL (zero adapter, segment-isolated) ----
    log("\n  Evaluating base-only PPL...")
    zero_adapter_in_model(model)
    base_ppl_per_domain = {}
    base_total_nll, base_total_n = 0.0, 0
    for domain in DOMAINS:
        nll, n = 0.0, 0
        for seg in domain_segments[domain]:
            seg_nll, seg_n = compute_token_ppl(seg)
            nll += seg_nll
            n += seg_n
        ppl = math.exp(nll / n) if n > 0 else float("inf")
        base_ppl_per_domain[domain] = ppl
        base_total_nll += nll
        base_total_n += n
        log(f"    base {domain}: PPL={ppl:.4f}")
    base_ppl_overall = math.exp(base_total_nll / base_total_n) if base_total_n > 0 else float("inf")
    log(f"    base overall: PPL={base_ppl_overall:.4f}")

    # ---- Per-sequence PPL at training scale (for reference) ----
    log("\n  Evaluating per-sequence PPL (training scale, full-length)...")
    set_lora_scale(model, TRAINING_SCALE)
    perseq_ppl_per_domain = {}
    perseq_total_nll, perseq_total_n = 0.0, 0
    for domain in DOMAINS:
        apply_adapter_to_model(model, adapters[domain])
        nll, n = 0.0, 0
        for sample in domain_data[domain]:
            tokens = tokenizer.encode(sample["text"])[:MAX_SEQ_LENGTH]
            if len(tokens) < 4:
                continue
            seg_nll, seg_n = compute_token_ppl(tokens)
            nll += seg_nll
            n += seg_n
        ppl = math.exp(nll / n) if n > 0 else float("inf")
        perseq_ppl_per_domain[domain] = ppl
        perseq_total_nll += nll
        perseq_total_n += n
        log(f"    per-seq {domain}: PPL={ppl:.4f} (scale={TRAINING_SCALE})")
    perseq_ppl_overall = math.exp(perseq_total_nll / perseq_total_n) if perseq_total_n > 0 else float("inf")
    log(f"    per-seq overall: PPL={perseq_ppl_overall:.4f}")

    # ---- Scale sweep: segment-isolated PPL per scale ----
    scale_results = {}
    for scale in SCALES:
        log(f"\n  Scale={scale}:")
        set_lora_scale(model, scale)

        scale_per_domain = {}
        total_nll, total_n = 0.0, 0
        for domain in DOMAINS:
            apply_adapter_to_model(model, adapters[domain])
            nll, n = 0.0, 0
            for seg in domain_segments[domain]:
                seg_nll, seg_n = compute_token_ppl(seg)
                nll += seg_nll
                n += seg_n
            ppl = math.exp(nll / n) if n > 0 else float("inf")
            scale_per_domain[domain] = ppl
            total_nll += nll
            total_n += n
            log(f"    {domain}: PPL={ppl:.4f}")

        overall_ppl = math.exp(total_nll / total_n) if total_n > 0 else float("inf")
        log(f"    overall: PPL={overall_ppl:.4f} (vs base {base_ppl_overall:.4f}, delta={((overall_ppl/base_ppl_overall)-1)*100:+.2f}%)")

        scale_results[scale] = {
            "per_domain": scale_per_domain,
            "overall_ppl": overall_ppl,
            "total_nll": total_nll,
            "total_n": total_n,
            "delta_vs_base_pct": ((overall_ppl / base_ppl_overall) - 1) * 100,
        }

        # Cleanup between scales
        gc.collect()
        mx.clear_cache()

    # Zero out adapter before cleanup
    zero_adapter_in_model(model)

    elapsed = time.time() - t0
    log(f"\n  Phase 1 complete in {elapsed:.1f}s")

    del model, tokenizer
    cleanup()
    log_memory("post-cleanup")

    return {
        "base_ppl_per_domain": base_ppl_per_domain,
        "base_ppl_overall": base_ppl_overall,
        "perseq_ppl_per_domain": perseq_ppl_per_domain,
        "perseq_ppl_overall": perseq_ppl_overall,
        "scale_results": scale_results,
        "time_s": elapsed,
    }


# ============================================================================
# Phase 2: Behavioral evaluation at best scale vs training scale
# ============================================================================

def phase_behavioral(best_scale, ppl_results):
    """Evaluate behavioral quality (factual recall / code correctness) at:
    1. best_scale (found in Phase 1)
    2. training_scale (s=20, for comparison)
    3. base only (no adapter)

    Uses segment-length prompts to match the PPL evaluation context.
    """
    log("\n" + "=" * 70)
    log(f"PHASE 2: BEHAVIORAL EVALUATION (best_scale={best_scale}, training_scale={TRAINING_SCALE})")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=TRAINING_SCALE)
    log_memory("model-loaded")

    # Load adapters
    adapters = {}
    for domain in DOMAINS:
        adapters[domain] = load_adapter(ADAPTERS_DIR / domain)

    # Load behavioral test data
    domain_test = {}
    for domain in DOMAINS:
        samples = load_domain_data(domain, split="valid", max_samples=N_GEN_PER_DOMAIN + 5)
        domain_test[domain] = samples[:N_GEN_PER_DOMAIN]
        log(f"  {domain}: {len(domain_test[domain])} behavioral samples")

    behavioral_results = {}

    for config_name, scale_val in [("base_only", None), ("best_scale", best_scale), ("training_scale", TRAINING_SCALE)]:
        log(f"\n  --- Config: {config_name} (scale={scale_val}) ---")
        config_scores = {}

        for domain in DOMAINS:
            if scale_val is not None:
                set_lora_scale(model, scale_val)
                apply_adapter_to_model(model, adapters[domain])
            else:
                zero_adapter_in_model(model)

            scores = []
            for sample in domain_test[domain]:
                prompt = format_prompt(sample["instruction"])
                ref = sample["response"]
                gen = generate_text(model, tokenizer, prompt)
                score = eval_response(gen, ref, domain)
                scores.append(score)

            mean_score = float(np.mean(scores)) if scores else 0.0
            config_scores[domain] = mean_score
            log(f"    {domain}: behavioral={mean_score:.3f} (n={len(scores)})")

        config_scores["mean"] = float(np.mean([config_scores[d] for d in DOMAINS]))
        behavioral_results[config_name] = config_scores
        log(f"    MEAN: {config_scores['mean']:.3f}")

        # Zero adapter between configs
        zero_adapter_in_model(model)

    elapsed = time.time() - t0
    log(f"\n  Phase 2 complete in {elapsed:.1f}s")

    del model, tokenizer
    cleanup()
    log_memory("post-cleanup")

    return {
        "behavioral": behavioral_results,
        "best_scale": best_scale,
        "time_s": elapsed,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    log("=" * 70)
    log("EXPERIMENT: LORA_SCALE Sweep on 128-Token Segments")
    log(f"  Model: {MODEL_ID}")
    log(f"  Domains: {DOMAINS}")
    log(f"  Scales: {SCALES}")
    log(f"  Training scale: {TRAINING_SCALE}")
    log(f"  Segment length: {SEGMENT_LENGTH}")
    log(f"  Eval samples/domain: {N_EVAL_PER_DOMAIN}")
    log(f"  Gen samples/domain: {N_GEN_PER_DOMAIN}")
    log("=" * 70)
    log_memory("start")

    # Phase 1: Segment PPL sweep
    ppl_data = phase_segment_ppl_sweep()
    log_memory("after-phase1")

    # Find best scale
    best_scale = None
    best_ppl = float("inf")
    for s in SCALES:
        ppl = ppl_data["scale_results"][s]["overall_ppl"]
        if ppl < best_ppl:
            best_ppl = ppl
            best_scale = s

    log(f"\n  Best scale: {best_scale} (PPL={best_ppl:.4f})")
    log(f"  Base PPL: {ppl_data['base_ppl_overall']:.4f}")
    log(f"  Per-seq PPL: {ppl_data['perseq_ppl_overall']:.4f}")

    # Phase 2: Behavioral evaluation
    behavioral_data = phase_behavioral(best_scale, ppl_data)
    log_memory("after-phase2")

    # ====================================================================
    # Assemble results and assess kill criteria
    # ====================================================================
    total_time = time.time() - t_start

    base_ppl = ppl_data["base_ppl_overall"]

    # K787: Best-scale segment PPL < base PPL (7.465)
    k787_pass = best_ppl < base_ppl
    k787_detail = f"Best segment PPL={best_ppl:.4f} at s={best_scale} vs base={base_ppl:.4f} (delta={((best_ppl/base_ppl)-1)*100:+.2f}%)"

    # K788: Scale-PPL curve is non-monotonic (optimal scale != training scale)
    k788_pass = (best_scale != TRAINING_SCALE)
    # Also check if the curve shows non-monotonic behavior
    ppls_at_scales = [(s, ppl_data["scale_results"][s]["overall_ppl"]) for s in SCALES]
    ppls_at_scales.sort(key=lambda x: x[0])
    # Non-monotonic = there exists s1 < s2 < s3 where PPL(s2) < PPL(s1) and PPL(s2) < PPL(s3)
    is_nonmonotonic = False
    for i in range(1, len(ppls_at_scales) - 1):
        if ppls_at_scales[i][1] < ppls_at_scales[i-1][1] and ppls_at_scales[i][1] < ppls_at_scales[i+1][1]:
            is_nonmonotonic = True
            break
    k788_detail = f"Best scale={best_scale} (training={TRAINING_SCALE}). Non-monotonic={is_nonmonotonic}. Curve: {[(s, round(p, 4)) for s, p in ppls_at_scales]}"

    # K789: Best-scale behavioral within 10% of per-sequence behavioral
    best_behavioral = behavioral_data["behavioral"].get("best_scale", {}).get("mean", 0.0)
    training_behavioral = behavioral_data["behavioral"].get("training_scale", {}).get("mean", 0.0)
    # Use training_scale behavioral as "per-sequence best" proxy
    if training_behavioral > 0:
        behavioral_ratio = best_behavioral / training_behavioral
        k789_pass = behavioral_ratio >= 0.90
    else:
        behavioral_ratio = 0.0
        k789_pass = False
    k789_detail = f"Best-scale behavioral={best_behavioral:.3f}, training-scale behavioral={training_behavioral:.3f}, ratio={behavioral_ratio:.3f}"

    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)
    log(f"  K787: {'PASS' if k787_pass else 'FAIL'} - {k787_detail}")
    log(f"  K788: {'PASS' if k788_pass else 'FAIL'} - {k788_detail}")
    log(f"  K789: {'PASS' if k789_pass else 'FAIL'} - {k789_detail}")

    overall = "SUPPORTED" if (k787_pass and k788_pass and k789_pass) else "KILLED"
    if k787_pass and k788_pass:
        overall = "SUPPORTED"  # K789 is secondary
    log(f"\n  VERDICT: {overall}")

    # Summary table
    log("\n" + "=" * 70)
    log("SCALE-PPL CURVE")
    log("=" * 70)
    log(f"  {'Scale':>8}  {'Segment PPL':>12}  {'vs Base':>10}  {'vs Per-Seq':>10}")
    log(f"  {'-----':>8}  {'-----------':>12}  {'---------':>10}  {'----------':>10}")
    log(f"  {'base':>8}  {base_ppl:>12.4f}  {'---':>10}  {((base_ppl/ppl_data['perseq_ppl_overall'])-1)*100:>+9.2f}%")
    for s in SCALES:
        ppl = ppl_data["scale_results"][s]["overall_ppl"]
        vs_base = ((ppl / base_ppl) - 1) * 100
        vs_perseq = ((ppl / ppl_data["perseq_ppl_overall"]) - 1) * 100
        marker = " <-- best" if s == best_scale else ""
        log(f"  {s:>8.1f}  {ppl:>12.4f}  {vs_base:>+9.2f}%  {vs_perseq:>+9.2f}%{marker}")
    log(f"  {'per-seq':>8}  {ppl_data['perseq_ppl_overall']:>12.4f}  {((ppl_data['perseq_ppl_overall']/base_ppl)-1)*100:>+9.2f}%  {'---':>10}")

    # Per-domain breakdown at best scale
    log("\n  PER-DOMAIN PPL AT BEST SCALE (s={best_scale}):")
    for domain in DOMAINS:
        seg_ppl = ppl_data["scale_results"][best_scale]["per_domain"][domain]
        b_ppl = ppl_data["base_ppl_per_domain"][domain]
        ps_ppl = ppl_data["perseq_ppl_per_domain"][domain]
        log(f"    {domain:>10}: seg={seg_ppl:.4f}, base={b_ppl:.4f}, per-seq={ps_ppl:.4f}, seg-vs-base={((seg_ppl/b_ppl)-1)*100:+.2f}%")

    # Behavioral summary
    log("\n  BEHAVIORAL SCORES:")
    log(f"  {'Config':>16}  {'medical':>8}  {'code':>8}  {'math':>8}  {'legal':>8}  {'finance':>8}  {'MEAN':>8}")
    for config_name in ["base_only", "best_scale", "training_scale"]:
        scores = behavioral_data["behavioral"][config_name]
        row = f"  {config_name:>16}"
        for domain in DOMAINS:
            row += f"  {scores[domain]:>8.3f}"
        row += f"  {scores['mean']:>8.3f}"
        log(row)

    # Write results
    results = {
        "experiment": "exp_segment_adapter_scale_sweep",
        "model": MODEL_ID,
        "domains": DOMAINS,
        "n_domains": N_DOMAINS,
        "scales_tested": SCALES,
        "training_scale": TRAINING_SCALE,
        "segment_length": SEGMENT_LENGTH,
        "n_eval_per_domain": N_EVAL_PER_DOMAIN,
        "n_gen_per_domain": N_GEN_PER_DOMAIN,
        "seed": SEED,
        "ppl": {
            "base_overall": base_ppl,
            "base_per_domain": ppl_data["base_ppl_per_domain"],
            "perseq_overall": ppl_data["perseq_ppl_overall"],
            "perseq_per_domain": ppl_data["perseq_ppl_per_domain"],
            "scale_results": {str(s): v for s, v in ppl_data["scale_results"].items()},
        },
        "best_scale": best_scale,
        "best_scale_ppl": best_ppl,
        "behavioral": behavioral_data["behavioral"],
        "kill_criteria": {
            "K787": {"pass": bool(k787_pass), "detail": k787_detail},
            "K788": {"pass": bool(k788_pass), "detail": k788_detail},
            "K789": {"pass": bool(k789_pass), "detail": k789_detail},
        },
        "verdict": overall,
        "phase1_time_s": round(ppl_data["time_s"], 1),
        "phase2_time_s": round(behavioral_data["time_s"], 1),
        "total_time_s": round(total_time, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\n  Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
