#!/usr/bin/env python3
"""Purpose-trained per-domain adapters with optimal module sets.

Guided Exploration (Type 2): Resolves Finding #304 Limitation 2.
  Finding #304 used post-hoc ablation of adapters trained with all 7 modules.
  This experiment trains adapters with ONLY the optimal module set per domain:
    medical=attn, code=full, math=attn, legal=attn, finance=attn.

Kill criteria:
  K778: Purpose-trained attn-only medical behavioral >= 0.39
  K779: Purpose-trained attn-only math PPL <= 3.43
  K780: Purpose-trained full-module code behavioral >= 0.25

Discriminating predictions:
  P4: If co-adaptation exists, purpose-trained attn-only OUTPERFORMS post-hoc
  P5: If independence holds, purpose-trained matches post-hoc (< 5% diff)
  P6: B-matrix cosine(purpose-trained, post-hoc) < 0.95 implies co-adaptation

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
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.bitlinear_layers import BitLinear

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
ADAPTERS_DIR = EXPERIMENT_DIR / "purpose_adapters"

# Source data and skeleton
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
DATA_DIR = SOURCE_DIR / "data"
NTP_ADAPTERS_DIR = SOURCE_DIR / "adapters"
SKELETON_PATH = NTP_ADAPTERS_DIR / "grassmannian_skeleton.npz"

# Post-hoc adapters for comparison (from SFT v3)
POSTHOC_DIR = EXPERIMENT_DIR.parent / "bitnet_sft_generation_v3" / "sft_adapters"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
TRAIN_ITERS = 300
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]

# Module groups
ATTN_MODULES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
MLP_MODULES = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
ALL_MODULES = ATTN_MODULES + MLP_MODULES

# Optimal module config per domain (from Finding #304)
DOMAIN_MODULE_CONFIG = {
    "medical": ATTN_MODULES,
    "code": ALL_MODULES,
    "math": ATTN_MODULES,
    "legal": ATTN_MODULES,
    "finance": ATTN_MODULES,
}

# Per-domain optimal scales (Finding #249)
OPTIMAL_SCALES = {
    "medical": 20.0,
    "code": 20.0,
    "math": 20.0,
    "legal": 4.0,
    "finance": 1.0,
}

# Evaluation sizes
N_PPL = 40
N_GEN = 5
MAX_NEW_TOKENS = 128

# SFT training
RESPONSE_MARKER = "### Response:\n"


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
# Model utilities
# ============================================================================

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
# TernaryLoRA layer
# ============================================================================

class TernaryLoRALinear(nn.Module):
    """LoRA with STE-ternary B and Grassmannian A."""
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
# SFT data loading and masking
# ============================================================================

def tokenize_with_sft_mask(text, tokenizer, max_len=256):
    """Tokenize and return (tokens, loss_mask). mask=1 for response tokens only."""
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
    for text in texts:
        tokens, mask = tokenize_with_sft_mask(text, tokenizer, max_len)
        if len(tokens) >= 4:
            batches.append((tokens, mask))
    return batches


def get_sft_batch(batches, batch_idx):
    idx = batch_idx % len(batches)
    tokens, mask = batches[idx]
    return mx.array([tokens]), mx.array([mask])


# ============================================================================
# Loss function
# ============================================================================

def sft_loss_fn(model, tokens, mask):
    """Cross-entropy loss ONLY on response tokens."""
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    response_mask = mask[:, 1:]
    per_token_loss = nn.losses.cross_entropy(logits, targets, reduction="none")
    masked_loss = per_token_loss * response_mask
    n_response = mx.maximum(response_mask.sum(), mx.array(1.0))
    return masked_loss.sum() / n_response


# ============================================================================
# Evaluation functions
# ============================================================================

def compute_ppl(model, tokenizer, texts, max_seq=MAX_SEQ_LENGTH):
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


# ============================================================================
# Pre-merge composition (for evaluation)
# ============================================================================

def save_base_weights(model, module_set=ALL_MODULES):
    base_weights = []
    for layer in model.model.layers:
        layer_w = {}
        for key in module_set:
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


def premerge_adapter(model, skeleton, adapter_b, domain, scale, module_set):
    """Pre-merge: W_new = W_base + scale * B^T @ A^T, filtered by module_set."""
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
# Skeleton loading with module-set filtering
# ============================================================================

def load_skeleton():
    return dict(np.load(str(SKELETON_PATH)))


def apply_lora_with_skeleton(model, skeleton, domain_idx, module_set):
    """Apply TernaryLoRALinear with Grassmannian A ONLY for specified modules."""
    count = 0
    for li, layer in enumerate(model.model.layers):
        lora_updates = []
        for key in module_set:
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
    log(f"  Applied TernaryLoRA to {count} modules "
        f"({len(module_set)} per layer, domain {domain_idx})")
    return model


# ============================================================================
# Phase 1: Train purpose-trained adapters
# ============================================================================

def phase_train_purpose_adapters():
    """Train 5 domain adapters with purpose-specific module sets."""
    log("\n" + "=" * 70)
    log("PHASE 1: TRAIN PURPOSE-TRAINED ADAPTERS")
    log("  medical/math/legal/finance: attn-only (4 modules)")
    log("  code: full-module (7 modules)")
    log("=" * 70)

    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    skeleton = load_skeleton()
    all_train_results = {}

    for di, domain in enumerate(DOMAINS):
        module_set = DOMAIN_MODULE_CONFIG[domain]
        config_name = "attn_only" if module_set == ATTN_MODULES else "full"
        log(f"\n--- Training {domain} ({config_name}, {len(module_set)} modules) ---")
        t0 = time.time()
        mx.reset_peak_memory()

        # Load fresh model
        model, tokenizer = load(MODEL_ID)
        model = replace_bitlinear_with_linear(model)

        # Apply LoRA ONLY to the purpose module set
        model = apply_lora_with_skeleton(model, skeleton, di, module_set)

        # Freeze base, train only lora_b
        model.freeze()
        model.unfreeze(keys=["lora_b"], strict=False)
        trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
        total = sum(p.size for _, p in tree_flatten(model.parameters()))
        log(f"  Trainable: {trainable:,} ({100*trainable/total:.4f}%)")

        # Load and prepare SFT data
        train_texts = load_data(domain, "train", 400)
        val_texts = load_data(domain, "valid", 50)
        log(f"  Data: {len(train_texts)} train, {len(val_texts)} val")

        train_batches = prepare_sft_batches(train_texts, tokenizer, MAX_SEQ_LENGTH)
        val_batches = prepare_sft_batches(val_texts, tokenizer, MAX_SEQ_LENGTH)

        if not train_batches:
            log(f"  WARNING: No valid training data for {domain}, skipping")
            cleanup(model, tokenizer)
            continue

        # Compute base validation loss (before training)
        base_val_loss = 0.0
        n_val = min(25, len(val_batches))
        for i in range(n_val):
            tokens, mask = get_sft_batch(val_batches, i)
            loss = sft_loss_fn(model, tokens, mask)
            mx.eval(loss)
            base_val_loss += loss.item()
            del loss, tokens, mask
        base_val_loss /= max(n_val, 1)
        log(f"  Base SFT val loss: {base_val_loss:.4f}")

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
                log(f"  Step {step+1}/{TRAIN_ITERS}: loss={loss_val:.4f}")
        gc.enable()
        gc.collect()

        # Post-training validation loss
        trained_val_loss = 0.0
        for i in range(n_val):
            tokens, mask = get_sft_batch(val_batches, i)
            loss = sft_loss_fn(model, tokens, mask)
            mx.eval(loss)
            trained_val_loss += loss.item()
            del loss, tokens, mask
        trained_val_loss /= max(n_val, 1)
        log(f"  Trained SFT val loss: {trained_val_loss:.4f}")

        # Save adapter weights (lora_b only)
        adapter_dir = ADAPTERS_DIR / domain
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_weights = {}
        for name, param in tree_flatten(model.trainable_parameters()):
            adapter_weights[name] = param
        mx.savez(str(adapter_dir / "adapter.npz"), **adapter_weights)
        log(f"  Saved to {adapter_dir}")

        elapsed = time.time() - t0
        peak_mem = mx.get_peak_memory() / 1e9

        all_train_results[domain] = {
            "module_config": config_name,
            "n_modules_per_layer": len(module_set),
            "trainable_params": trainable,
            "base_val_loss": round(base_val_loss, 4),
            "trained_val_loss": round(trained_val_loss, 4),
            "converged": trained_val_loss < base_val_loss,
            "final_loss": round(losses[-1], 4),
            "loss_reduction_pct": round(
                (base_val_loss - trained_val_loss) / base_val_loss * 100, 1
            ) if base_val_loss > 0 else 0,
            "time_s": round(elapsed, 1),
            "peak_memory_gb": round(peak_mem, 2),
        }
        log(f"  {domain}: converged={trained_val_loss < base_val_loss}, "
            f"loss {base_val_loss:.4f} -> {trained_val_loss:.4f}, "
            f"{elapsed:.1f}s, peak={peak_mem:.2f}GB")

        cleanup(model, tokenizer, optimizer)
        log_memory(f"after-{domain}")

    del skeleton
    return all_train_results


# ============================================================================
# Phase 2: PPL evaluation (purpose-trained vs post-hoc)
# ============================================================================

def phase_ppl_comparison():
    """Compare PPL: purpose-trained vs post-hoc ablated vs base."""
    log("\n" + "=" * 70)
    log("PHASE 2: PPL COMPARISON")
    log("=" * 70)

    skeleton = load_skeleton()
    val_data = {d: load_data(d, "valid", N_PPL) for d in DOMAINS}

    results = {"base": {}, "posthoc": {}, "purpose": {}}

    # Load model for evaluation
    model, tok = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    base_weights = save_base_weights(model)
    log_memory("base loaded")

    # Base PPL
    for d in DOMAINS:
        results["base"][d] = round(compute_ppl(model, tok, val_data[d]), 3)
        log(f"  base/{d}: {results['base'][d]}")

    # Post-hoc ablated PPL (using full-module adapters with attn-only or full eval)
    log("\n  --- Post-hoc ablated (Finding #304 config) ---")
    for d in DOMAINS:
        restore_base_weights(model, base_weights)
        module_set = DOMAIN_MODULE_CONFIG[d]
        adapter_b = dict(mx.load(str(POSTHOC_DIR / d / "adapter.npz")))
        scale = OPTIMAL_SCALES[d]
        n_merged = premerge_adapter(model, skeleton, adapter_b, d, scale, module_set)
        ppl = round(compute_ppl(model, tok, val_data[d]), 3)
        results["posthoc"][d] = ppl
        log(f"    {d}: PPL={ppl} ({n_merged} modules, post-hoc {len(module_set)}-module)")
        del adapter_b

    # Purpose-trained PPL
    log("\n  --- Purpose-trained ---")
    for d in DOMAINS:
        restore_base_weights(model, base_weights)
        module_set = DOMAIN_MODULE_CONFIG[d]
        adapter_path = ADAPTERS_DIR / d / "adapter.npz"
        if not adapter_path.exists():
            log(f"    {d}: SKIPPED (no adapter)")
            results["purpose"][d] = float('inf')
            continue
        adapter_b = dict(mx.load(str(adapter_path)))
        scale = OPTIMAL_SCALES[d]
        n_merged = premerge_adapter(model, skeleton, adapter_b, d, scale, module_set)
        ppl = round(compute_ppl(model, tok, val_data[d]), 3)
        results["purpose"][d] = ppl
        log(f"    {d}: PPL={ppl} ({n_merged} modules, purpose-trained)")
        del adapter_b

    cleanup(model, tok)
    return results


# ============================================================================
# Phase 3: Behavioral evaluation (purpose-trained vs post-hoc)
# ============================================================================

def phase_behavioral_comparison():
    """Compare behavioral scores: purpose-trained vs post-hoc ablated."""
    log("\n" + "=" * 70)
    log("PHASE 3: BEHAVIORAL COMPARISON")
    log("=" * 70)

    skeleton = load_skeleton()
    results = {"posthoc": {}, "purpose": {}}

    for config_name, adapter_source in [
        ("posthoc", POSTHOC_DIR),
        ("purpose", ADAPTERS_DIR),
    ]:
        log(f"\n  --- {config_name} adapters ---")
        for d in DOMAINS:
            model, tok = load(MODEL_ID)
            model = replace_bitlinear_with_linear(model)
            model.freeze()

            module_set = DOMAIN_MODULE_CONFIG[d]
            adapter_path = adapter_source / d / "adapter.npz"
            if not adapter_path.exists():
                log(f"    {d}: SKIPPED (no adapter)")
                results[config_name][d] = 0.0
                cleanup(model, tok)
                continue

            adapter_b = dict(mx.load(str(adapter_path)))
            scale = OPTIMAL_SCALES[d]
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

    return results


# ============================================================================
# Phase 4: B-matrix divergence analysis
# ============================================================================

def phase_b_matrix_analysis():
    """Compare B-matrices between purpose-trained and post-hoc adapters."""
    log("\n" + "=" * 70)
    log("PHASE 4: B-MATRIX DIVERGENCE ANALYSIS")
    log("=" * 70)

    results = {}
    for d in DOMAINS:
        module_set = DOMAIN_MODULE_CONFIG[d]

        posthoc_path = POSTHOC_DIR / d / "adapter.npz"
        purpose_path = ADAPTERS_DIR / d / "adapter.npz"
        if not purpose_path.exists() or not posthoc_path.exists():
            log(f"  {d}: SKIPPED (missing adapter)")
            continue

        posthoc_b = dict(mx.load(str(posthoc_path)))
        purpose_b = dict(mx.load(str(purpose_path)))

        # Compute per-module cosine similarity and norm ratio
        cosines = []
        norm_ratios = []
        for key in sorted(purpose_b.keys()):
            if key not in posthoc_b:
                continue
            # Only compare modules in the domain config
            key_matches = any(mod in key for mod in module_set)
            if not key_matches:
                continue

            p_flat = purpose_b[key].reshape(-1).astype(mx.float32)
            h_flat = posthoc_b[key].reshape(-1).astype(mx.float32)
            mx.eval(p_flat, h_flat)

            p_norm = mx.linalg.norm(p_flat).item()
            h_norm = mx.linalg.norm(h_flat).item()

            if p_norm > 1e-8 and h_norm > 1e-8:
                cos = (mx.sum(p_flat * h_flat) / (p_norm * h_norm)).item()
                cosines.append(cos)
                norm_ratios.append(p_norm / h_norm)

            del p_flat, h_flat

        if cosines:
            mean_cos = float(np.mean(cosines))
            std_cos = float(np.std(cosines))
            mean_norm_ratio = float(np.mean(norm_ratios))
            results[d] = {
                "mean_cosine": round(mean_cos, 4),
                "std_cosine": round(std_cos, 4),
                "min_cosine": round(float(np.min(cosines)), 4),
                "max_cosine": round(float(np.max(cosines)), 4),
                "mean_norm_ratio": round(mean_norm_ratio, 4),
                "n_modules_compared": len(cosines),
            }
            log(f"  {d}: cos={mean_cos:.4f} +/- {std_cos:.4f}, "
                f"norm_ratio={mean_norm_ratio:.4f}, n={len(cosines)}")
        else:
            results[d] = {"mean_cosine": None, "n_modules_compared": 0}
            log(f"  {d}: no comparable modules")

        del posthoc_b, purpose_b

    gc.collect()
    mx.clear_cache()
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("EXPERIMENT: Purpose-Trained Module Adapters")
    log(f"  Model: {MODEL_ID}")
    log(f"  Domains: {DOMAINS}")
    log(f"  Module configs: " +
        ", ".join(f"{d}={'attn' if DOMAIN_MODULE_CONFIG[d]==ATTN_MODULES else 'full'}"
                  for d in DOMAINS))
    log("=" * 70)
    log_memory("start")

    # Phase 1: Train adapters with purpose-specific module sets
    train_results = phase_train_purpose_adapters()
    log_memory("after-training")

    # Phase 2: PPL comparison
    ppl_results = phase_ppl_comparison()
    log_memory("after-ppl")

    # Phase 3: Behavioral comparison
    behavioral = phase_behavioral_comparison()
    log_memory("after-behavioral")

    # Phase 4: B-matrix divergence
    b_matrix = phase_b_matrix_analysis()
    log_memory("after-bmatrix")

    # ====================================================================
    # Analysis and kill criteria
    # ====================================================================
    log("\n" + "=" * 70)
    log("ANALYSIS")
    log("=" * 70)

    # K778: Purpose-trained attn-only medical behavioral >= 0.39
    k778_val = behavioral["purpose"].get("medical", 0)
    k778_posthoc = behavioral["posthoc"].get("medical", 0)
    k778_pass = k778_val >= 0.39
    log(f"\n  K778 (medical behavioral >= 0.39):")
    log(f"    Purpose-trained: {k778_val:.3f}")
    log(f"    Post-hoc:        {k778_posthoc:.3f}")
    log(f"    PASS: {k778_pass}")

    # K779: Purpose-trained attn-only math PPL <= 3.43
    k779_val = ppl_results["purpose"].get("math", float('inf'))
    k779_posthoc = ppl_results["posthoc"].get("math", float('inf'))
    k779_pass = k779_val <= 3.43
    log(f"\n  K779 (math PPL <= 3.43):")
    log(f"    Purpose-trained: {k779_val}")
    log(f"    Post-hoc:        {k779_posthoc}")
    log(f"    PASS: {k779_pass}")

    # K780: Purpose-trained full-module code behavioral >= 0.25
    k780_val = behavioral["purpose"].get("code", 0)
    k780_posthoc = behavioral["posthoc"].get("code", 0)
    k780_pass = k780_val >= 0.25
    log(f"\n  K780 (code behavioral >= 0.25):")
    log(f"    Purpose-trained: {k780_val:.3f}")
    log(f"    Post-hoc:        {k780_posthoc:.3f}")
    log(f"    PASS: {k780_pass}")

    # Discriminating predictions
    log("\n  Discriminating predictions:")

    # P4/P5: Co-adaptation vs independence
    # Compare purpose vs posthoc for attn-only domains
    attn_domains = ["medical", "math", "legal", "finance"]
    purpose_better_count = 0
    posthoc_better_count = 0
    for d in attn_domains:
        pb = behavioral["purpose"].get(d, 0)
        hb = behavioral["posthoc"].get(d, 0)
        diff_pct = ((pb - hb) / max(hb, 0.001)) * 100 if hb > 0 else 0
        if pb > hb * 1.05:
            purpose_better_count += 1
        elif hb > pb * 1.05:
            posthoc_better_count += 1
        log(f"    {d}: purpose={pb:.3f}, posthoc={hb:.3f}, diff={diff_pct:+.1f}%")

    if purpose_better_count >= 2:
        coadaptation = "H2 (co-adaptation): purpose-trained outperforms post-hoc"
    elif posthoc_better_count >= 2:
        coadaptation = "UNEXPECTED: post-hoc outperforms purpose-trained"
    else:
        coadaptation = "H1 (independence): no significant difference"
    log(f"    Verdict: {coadaptation}")

    # P6: B-matrix divergence
    mean_cos_all = []
    for d in attn_domains:
        if d in b_matrix and b_matrix[d].get("mean_cosine") is not None:
            mean_cos_all.append(b_matrix[d]["mean_cosine"])
    if mean_cos_all:
        overall_cos = float(np.mean(mean_cos_all))
        if overall_cos < 0.95:
            p6_verdict = f"B-matrices diverge (cos={overall_cos:.4f} < 0.95): co-adaptation confirmed"
        else:
            p6_verdict = f"B-matrices converge (cos={overall_cos:.4f} >= 0.95): independence confirmed"
        log(f"    P6: {p6_verdict}")
    else:
        overall_cos = None
        p6_verdict = "No B-matrix data"
        log(f"    P6: {p6_verdict}")

    # PPL comparison summary
    log("\n  PPL Summary:")
    for d in DOMAINS:
        base = ppl_results["base"].get(d, 0)
        posthoc_ppl = ppl_results["posthoc"].get(d, 0)
        purpose_ppl = ppl_results["purpose"].get(d, 0)
        log(f"    {d}: base={base}, posthoc={posthoc_ppl}, purpose={purpose_ppl}")

    # ====================================================================
    # Assemble results
    # ====================================================================
    all_results = {
        "experiment": "purpose_trained_module_adapters",
        "hypothesis": "Adapters trained with only optimal module set match/exceed post-hoc ablated",
        "type": "verification",
        "model": MODEL_ID,
        "domain_module_config": {d: ("attn_only" if c == ATTN_MODULES else "full")
                                 for d, c in DOMAIN_MODULE_CONFIG.items()},
        "training": train_results,
        "ppl": ppl_results,
        "behavioral": behavioral,
        "b_matrix_analysis": b_matrix,
        "kill_criteria": {
            "K778": {
                "text": "Purpose-trained attn-only medical behavioral >= 0.39",
                "pass": k778_pass,
                "purpose_value": round(k778_val, 3),
                "posthoc_value": round(k778_posthoc, 3),
            },
            "K779": {
                "text": "Purpose-trained attn-only math PPL <= 3.43",
                "pass": k779_pass,
                "purpose_value": k779_val,
                "posthoc_value": k779_posthoc,
            },
            "K780": {
                "text": "Purpose-trained full-module code behavioral >= 0.25",
                "pass": k780_pass,
                "purpose_value": round(k780_val, 3),
                "posthoc_value": round(k780_posthoc, 3),
            },
        },
        "discriminating": {
            "coadaptation_verdict": coadaptation,
            "b_matrix_overall_cosine": overall_cos,
            "b_matrix_verdict": p6_verdict,
        },
        "all_pass": k778_pass and k779_pass and k780_pass,
        "total_time_s": round(time.time() - t0, 1),
    }

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {all_results['total_time_s']:.1f}s")
    log(f"\nKill criteria: K778={'PASS' if k778_pass else 'FAIL'}, "
        f"K779={'PASS' if k779_pass else 'FAIL'}, "
        f"K780={'PASS' if k780_pass else 'FAIL'}")
    log(f"All pass: {all_results['all_pass']}")


if __name__ == "__main__":
    main()
