#!/usr/bin/env python3
"""Pierre Pro: integrated serving on Qwen3-4B-4bit with full composition pipeline.

Replicates the integrated pipeline proven on BitNet-2B-4T (Finding #323)
on the larger Qwen3-4B-4bit model.

Components:
  1. Block-diagonal causal mask (Finding #322: no RoPE reset needed)
  2. MLP-only per-token adapter routing (Finding #312-313)
  3. SFT adapters at scale<=5 (Findings #319, #320, #330)
  4. DARE p=0.5 for OOD robustness (Finding #266)
  5. Ridge regression router (Finding #276/#310)

Key corrections from prior findings:
  - LORA_SCALE = 5.0 (NOT 20 -- Finding #330: scale=20 causes -42pp MMLU loss)
  - No RoPE reset (Finding #322: unnecessary, block-diag mask sufficient)
  - Speed measures actual integrated pipeline forward pass, not mlx_generate

Kill criteria:
  K821: Quality below base Qwen3-4B on majority of benchmarks (behavioral < 0.3)
"""

import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

BASE_DIR = EXPERIMENT_DIR.parent / "pro_base_validation"
INIT_DIR = EXPERIMENT_DIR.parent / "pro_grassmannian_init"
ADAPTER_DIR = EXPERIMENT_DIR.parent / "pro_sft_5_adapters" / "adapters"
DATA_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "data"

# --- Configuration ---
LORA_RANK = 16
LORA_SCALE = 5.0  # NOT 20! Finding #330: scale<=5 preserves MMLU
MAX_SEQ = 256
SEGMENT_LEN = 128
SEED = 42
DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_CAL = 50     # calibration texts per domain for router
N_EVAL = 5     # eval texts per domain
N_PAIRS = 6    # domain pairs for integrated eval
DARE_P = 0.5

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def log(m):
    print(m, flush=True)


def log_memory(label=""):
    a = mx.get_active_memory() / 1e9
    p = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB peak={p:.2f}GB")


def cleanup(*o):
    for x in o:
        del x
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def load_data(domain, split="valid", n=None):
    texts = []
    p = DATA_DIR / domain / f"{split}.jsonl"
    if not p.exists():
        return []
    with open(p) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
            if n and len(texts) >= n:
                break
    return texts


# ---- DARE sparsification (Finding #266) ----

def dare_sparsify(adapter_b, p=0.5, seed=42):
    """Apply DARE: randomly drop p fraction of adapter B params, rescale by 1/(1-p)."""
    rng = np.random.RandomState(seed)
    sparsified = {}
    for key, val in adapter_b.items():
        val_np = np.array(val)
        mask = rng.binomial(1, 1.0 - p, size=val_np.shape).astype(np.float32)
        sparsified[key] = mx.array(val_np * mask / (1.0 - p)).astype(mx.bfloat16)
    return sparsified


# ---- Block-diagonal causal mask (Finding #322) ----

def create_block_diagonal_mask_2seg(seq_len, boundary):
    """Block-diagonal mask for 2 segments. No RoPE reset needed (Finding #322)."""
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    positions = mx.arange(seq_len)
    seg_i = positions >= boundary
    seg_j = positions >= boundary
    cross_seg = seg_i[:, None] != seg_j[None, :]
    cross_penalty = mx.where(cross_seg, mx.array(float("-inf")), mx.array(0.0))
    return mask + cross_penalty


# ---- Encoder for router ----

_mask_cache = {}

def encode(model, input_ids):
    """Mean-pooled post-norm hidden state. (B, T) -> (B, H)."""
    T = input_ids.shape[1]
    if T not in _mask_cache:
        _mask_cache[T] = nn.MultiHeadAttention.create_additive_causal_mask(T)
    mask = _mask_cache[T].astype(mx.bfloat16)
    h = model.model.embed_tokens(input_ids)
    for layer in model.model.layers:
        h = layer(h, mask=mask)
    h = model.model.norm(h)
    mx.eval(h)
    return mx.mean(h, axis=1).astype(mx.float32)


# ---- Ridge router (Finding #276/#310) ----

def fit_router(model, tok, domain_texts, lam=1.0, max_seq=256):
    """Closed-form ridge regression router: W* = (X^TX + lambda*I)^{-1} X^TY."""
    domains = list(domain_texts.keys())
    D = len(domains)
    H = model.args.hidden_size

    XtX = mx.zeros((H, H))
    XtY = mx.zeros((H, D))

    for di, domain in enumerate(domains):
        for text in domain_texts[domain]:
            toks = tok.encode(text)[:max_seq]
            if len(toks) < 4:
                continue
            h = encode(model, mx.array(toks)[None, :])
            XtX = XtX + h.T @ h
            XtY = XtY.at[:, di].add(h.squeeze(0))

    W = mx.linalg.solve(XtX + lam * mx.eye(H), XtY, stream=mx.cpu)
    W = W / mx.maximum(mx.linalg.norm(W, axis=0, keepdims=True), 1e-8)
    mx.eval(W)
    return W


def route(model, tok, text, W, max_seq=256):
    """Route query to best domain. Returns domain index."""
    h = encode(model, mx.array(tok.encode(text)[:max_seq])[None, :])
    return mx.argmax(h @ W, axis=-1).item()


# ---- Adapter management ----

class RuntimeLoRA(nn.Module):
    """y = base(x) + alpha * (x @ A) @ B"""
    def __init__(self, base, A, B, alpha):
        super().__init__()
        self.base = base
        self.lora_a = A.astype(mx.bfloat16)
        self.lora_b = B.astype(mx.bfloat16)
        self.alpha = alpha
        self.freeze(keys=["base", "lora_a"], strict=False)

    def __call__(self, x):
        y = self.base(x)
        return y + ((x @ self.lora_a) @ self.lora_b * self.alpha).astype(y.dtype)


def attach_adapter(model, frozen_A, adapter_B, domain_idx, alpha):
    """Attach single domain adapter via RuntimeLoRA wrapping."""
    count = 0
    for li in range(len(model.model.layers)):
        updates = []
        for key in TARGET_KEYS:
            bk = f"model.layers.{li}.{key}.lora_b"
            ak = f"layer_{li}_{key}_domain_{domain_idx}"
            if bk not in adapter_B or ak not in frozen_A:
                continue
            m = model.model.layers[li]
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is None:
                continue
            A = mx.array(frozen_A[ak]).astype(mx.bfloat16)
            B = adapter_B[bk].astype(mx.bfloat16)
            updates.append((key, RuntimeLoRA(m, A, B, alpha)))
            count += 1
        if updates:
            model.model.layers[li].update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    return count


def detach_adapters(model):
    """Remove all RuntimeLoRA wrappers, restoring base modules."""
    count = 0
    for layer in model.model.layers:
        updates = []
        for key in TARGET_KEYS:
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if isinstance(m, RuntimeLoRA):
                updates.append((key, m.base))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    return count


# ---- Behavioral evaluation ----

STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'to', 'of', 'in', 'for', 'on', 'with',
    'at', 'by', 'from', 'as', 'and', 'but', 'or', 'not', 'so', 'yet', 'both',
    'either', 'each', 'every', 'all', 'any', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
    'just', 'because', 'if', 'when', 'where', 'how', 'what', 'which', 'who',
    'this', 'that', 'these', 'those', 'it', 'its', 'i', 'me', 'my', 'we',
    'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them',
    'their',
}


def factual_recall(generated, reference):
    def tokenize(x):
        return set(
            w for w in re.findall(r'\b[a-z]+\b', x.lower())
            if w not in STOP_WORDS and len(w) > 2
        )
    gt, rt = tokenize(generated), tokenize(reference)
    return len(gt & rt) / len(rt) if rt else 0.0


# ---- PPL helpers ----

def compute_nll_with_mask(model, tokens, mask):
    """Forward pass with custom attention mask. Returns per-token NLL array."""
    x = mx.array(tokens)[None, :]
    h = model.model.embed_tokens(x)
    for layer in model.model.layers:
        h = layer(h, mask=mask, cache=None)
    h = model.model.norm(h)
    # Qwen3 uses tie_word_embeddings
    logits = model.model.embed_tokens.as_linear(h)
    mx.eval(logits)
    log_probs = mx.log(mx.softmax(logits[0, :-1], axis=-1) + 1e-10)
    targets = mx.array(tokens[1:])
    nll = -mx.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(-1)
    mx.eval(nll)
    del logits, log_probs, h
    return nll


# ---- Single-pass per-token MLP routing (Finding #313) ----

def single_pass_mixed_mlp_forward(model, tokens, mask, frozen_A,
                                   adapter_A, adapter_B,
                                   domain_idx_A, domain_idx_B,
                                   boundary, alpha=LORA_SCALE):
    """Single-pass forward with per-token MLP adapter routing.

    tokens 0..boundary-1 use adapter_A, tokens boundary..end use adapter_B.
    Attention layers use BASE weights only (no LoRA).
    MLP layers apply per-token adapter via mx.where mask.

    Adapted from tiny_integrated_serving for Qwen3 architecture:
    - Uses SiLU activation (not squared ReLU)
    - Uses q_norm/k_norm (QK-RMSNorm)
    - GQA with 32 query heads, 8 KV heads
    - No ffn_sub_norm (Qwen3 doesn't have it)
    """
    x = mx.array(tokens)[None, :]
    h = model.model.embed_tokens(x)
    T = len(tokens)

    # Per-token mask for broadcasting: (1, T, 1) -- True = adapter A
    tok_mask = mx.arange(T)[None, :, None] < boundary

    for li, layer in enumerate(model.model.layers):
        # -- Attention sub-layer with base weights + custom mask --
        h_norm = layer.input_layernorm(h)

        attn = layer.self_attn
        B_dim, L, D = h_norm.shape

        queries = attn.q_proj(h_norm)
        keys = attn.k_proj(h_norm)
        values = attn.v_proj(h_norm)

        # Qwen3: q_norm and k_norm applied to reshaped heads
        queries = attn.q_norm(
            queries.reshape(B_dim, L, attn.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        keys = attn.k_norm(
            keys.reshape(B_dim, L, attn.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(B_dim, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = attn.rope(queries)
        keys = attn.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=attn.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B_dim, L, -1)
        output = attn.o_proj(output)
        h = h + output

        # -- MLP sub-layer with per-token adapter routing --
        h_norm2 = layer.post_attention_layernorm(h)
        mlp = layer.mlp

        def mlp_with_mixed_lora(module, h_in, module_key):
            """Compute base(x) + per-token LoRA delta."""
            base_out = module(h_in)

            ak_A = f"layer_{li}_{module_key}_domain_{domain_idx_A}"
            ak_B = f"layer_{li}_{module_key}_domain_{domain_idx_B}"
            bk = f"model.layers.{li}.{module_key}.lora_b"

            if ak_A not in frozen_A or bk not in adapter_A:
                return base_out

            A_a = mx.array(frozen_A[ak_A]).astype(mx.bfloat16)
            A_b = mx.array(frozen_A[ak_B]).astype(mx.bfloat16)
            B_a = adapter_A[bk].astype(mx.bfloat16)
            B_b = adapter_B[bk].astype(mx.bfloat16)

            lora_out_A = ((h_in @ A_a) @ B_a * alpha).astype(base_out.dtype)
            lora_out_B = ((h_in @ A_b) @ B_b * alpha).astype(base_out.dtype)

            mixed_lora = mx.where(tok_mask, lora_out_A, lora_out_B)
            return base_out + mixed_lora

        # Qwen3 MLP: down_proj(silu(gate_proj(x)) * up_proj(x))
        gate_out = mlp_with_mixed_lora(mlp.gate_proj, h_norm2, "mlp.gate_proj")
        up_out = mlp_with_mixed_lora(mlp.up_proj, h_norm2, "mlp.up_proj")

        x_mid = nn.silu(gate_out) * up_out  # SiLU, not squared ReLU

        down_out = mlp_with_mixed_lora(mlp.down_proj, x_mid, "mlp.down_proj")
        h = h + down_out

    h = model.model.norm(h)
    logits = model.model.embed_tokens.as_linear(h)
    mx.eval(logits)

    log_probs = mx.log(mx.softmax(logits[0, :-1], axis=-1) + 1e-10)
    targets = mx.array(tokens[1:])
    nll = -mx.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(-1)
    mx.eval(nll)
    del logits, log_probs, h, output, queries, keys, values
    return nll


# ============================================================
# Phase functions (each self-contained per CODING_GUIDELINES)
# ============================================================

def phase_calibrate_router(model_id, frozen_A):
    """Phase 1: Fit ridge regression router on calibration data."""
    log("\n=== Phase 1: Ridge Router Calibration ===")

    model, tok = load(model_id)
    log_memory("model loaded for router")

    cal_data = {d: load_data(d, "train", N_CAL) for d in DOMAINS}
    W = fit_router(model, tok, cal_data, max_seq=MAX_SEQ)
    log(f"  Router fitted (W shape: {W.shape})")

    # Test routing accuracy
    correct, total = 0, 0
    for di, d in enumerate(DOMAINS):
        for text in load_data(d, "valid", 10):
            if route(model, tok, text, W, MAX_SEQ) == di:
                correct += 1
            total += 1
    accuracy = correct / total if total > 0 else 0
    log(f"  Routing accuracy: {accuracy:.1%} ({correct}/{total})")

    np.save(str(EXPERIMENT_DIR / "router_W.npy"), np.array(W))
    cleanup(model, tok)
    log_memory("after router cleanup")
    return W, accuracy


def phase_behavioral_isolated(model_id, frozen_A):
    """Phase 2: Per-domain isolated quality (single adapter per domain)."""
    log("\n=== Phase 2: Per-Domain Isolated Quality ===")

    behavioral = {}
    sampler = make_sampler(temp=0.0)

    for di, d in enumerate(DOMAINS):
        adapter_path = ADAPTER_DIR / d / "adapter.npz"
        if not adapter_path.exists():
            log(f"  SKIP {d}: no adapter")
            continue

        model, tok = load(model_id)
        adapter_b = dict(mx.load(str(adapter_path)))
        n_modules = attach_adapter(model, frozen_A, adapter_b, di, LORA_SCALE)

        test_texts = load_data(d, "valid", N_EVAL)
        scores = []
        for text in test_texts:
            if "### Response:" in text:
                prompt = text.split("### Response:")[0].strip() + "\n### Response:\n"
                ref = text.split("### Response:")[-1].strip()
            else:
                prompt, ref = text[:200], text
            try:
                gen = mlx_generate(
                    model, tok, prompt=prompt, max_tokens=128,
                    sampler=sampler, verbose=False
                )
                scores.append(factual_recall(gen, ref))
            except Exception as e:
                log(f"    ERROR generating for {d}: {e}")
                scores.append(0.0)

        mean_score = float(np.mean(scores)) if scores else 0.0
        behavioral[d] = {
            "score": round(mean_score, 3),
            "n_modules": n_modules,
            "n_samples": len(scores),
        }
        log(f"  {d}: behavioral={mean_score:.3f} ({n_modules} modules, {len(scores)} samples)")
        cleanup(model, tok, adapter_b)

    log_memory("after isolated eval cleanup")
    return behavioral


def phase_integrated_pipeline(model_id, frozen_A):
    """Phase 3: Integrated pipeline with block-diag mask + per-token MLP routing.

    For each domain pair:
    1. Concatenate segments from two domains
    2. Create block-diagonal causal mask
    3. Run single forward pass with per-token MLP adapter routing
    4. Compare PPL to per-sequence baseline and segment-isolated oracle
    """
    log("\n=== Phase 3: Integrated Pipeline (Block-Diag + Per-Token MLP) ===")

    model, tok = load(model_id)
    log_memory("model loaded for integrated pipeline")

    from itertools import combinations
    domain_pairs = list(combinations(range(len(DOMAINS)), 2))[:N_PAIRS]

    # Load all DARE-sparsified adapters
    adapters = {}
    for di, d in enumerate(DOMAINS):
        adapter_path = ADAPTER_DIR / d / "adapter.npz"
        if not adapter_path.exists():
            continue
        raw = dict(mx.load(str(adapter_path)))
        adapters[di] = dare_sparsify(raw, p=DARE_P, seed=SEED + di)
        del raw
    gc.collect()

    results = {"pairs": [], "summary": {}}
    all_gaps_vs_iso = []
    all_gaps_vs_perseq = []

    for di_A, di_B in domain_pairs:
        d_A, d_B = DOMAINS[di_A], DOMAINS[di_B]
        texts_A = load_data(d_A, "valid", 5)
        texts_B = load_data(d_B, "valid", 5)

        pair_results = []
        for text_A, text_B in zip(texts_A[:3], texts_B[:3]):
            tokens_A = tok.encode(text_A)[:SEGMENT_LEN]
            tokens_B = tok.encode(text_B)[:SEGMENT_LEN]

            if len(tokens_A) < 16 or len(tokens_B) < 16:
                continue

            combined = tokens_A + tokens_B
            boundary = len(tokens_A)
            T = len(combined)

            # -- Per-sequence baseline (one adapter on full sequence) --
            attach_adapter(model, frozen_A, adapters[di_A], di_A, LORA_SCALE)
            causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(mx.bfloat16)
            nll_perseq_A = compute_nll_with_mask(model, combined, causal_mask)
            perseq_A_mean = mx.mean(nll_perseq_A).item()
            detach_adapters(model)

            attach_adapter(model, frozen_A, adapters[di_B], di_B, LORA_SCALE)
            nll_perseq_B = compute_nll_with_mask(model, combined, causal_mask)
            perseq_B_mean = mx.mean(nll_perseq_B).item()
            detach_adapters(model)

            perseq_ppl = min(math.exp(perseq_A_mean), math.exp(perseq_B_mean))

            # -- Segment-isolated oracle --
            attach_adapter(model, frozen_A, adapters[di_A], di_A, LORA_SCALE)
            iso_mask_A = nn.MultiHeadAttention.create_additive_causal_mask(len(tokens_A)).astype(mx.bfloat16)
            iso_nll_A = compute_nll_with_mask(model, tokens_A, iso_mask_A)
            mx.eval(iso_nll_A)
            detach_adapters(model)

            attach_adapter(model, frozen_A, adapters[di_B], di_B, LORA_SCALE)
            iso_mask_B = nn.MultiHeadAttention.create_additive_causal_mask(len(tokens_B)).astype(mx.bfloat16)
            iso_nll_B = compute_nll_with_mask(model, tokens_B, iso_mask_B)
            mx.eval(iso_nll_B)
            detach_adapters(model)

            iso_nll_total = mx.sum(iso_nll_A).item() + mx.sum(iso_nll_B).item()
            iso_n = iso_nll_A.size + iso_nll_B.size
            iso_ppl = math.exp(iso_nll_total / iso_n) if iso_n > 0 else float('inf')

            # -- Integrated pipeline: block-diag + per-token MLP --
            bd_mask = create_block_diagonal_mask_2seg(T, boundary).astype(mx.bfloat16)

            nll_integrated = single_pass_mixed_mlp_forward(
                model, combined, bd_mask, frozen_A,
                adapters[di_A], adapters[di_B],
                di_A, di_B, boundary
            )

            # Fair PPL (exclude boundary token which predicts first token of seg B from seg A context)
            nll_seg_A = nll_integrated[:boundary - 1]
            nll_seg_B = nll_integrated[boundary:]
            int_nll_fair = mx.sum(nll_seg_A).item() + mx.sum(nll_seg_B).item()
            int_n_fair = nll_seg_A.size + nll_seg_B.size
            int_ppl_fair = math.exp(int_nll_fair / int_n_fair) if int_n_fair > 0 else float('inf')

            # Gaps
            gap_int_vs_iso = (int_ppl_fair - iso_ppl) / iso_ppl * 100 if iso_ppl > 0 else 0
            gap_int_vs_perseq = (int_ppl_fair - perseq_ppl) / perseq_ppl * 100 if perseq_ppl > 0 else 0

            log(f"  {d_A}+{d_B}: iso={iso_ppl:.3f} perseq={perseq_ppl:.3f} "
                f"integrated={int_ppl_fair:.3f} "
                f"(vs iso: {gap_int_vs_iso:+.1f}%, vs perseq: {gap_int_vs_perseq:+.1f}%)")

            pair_results.append({
                "domains": [d_A, d_B],
                "iso_ppl": round(iso_ppl, 3),
                "perseq_ppl": round(perseq_ppl, 3),
                "integrated_ppl_fair": round(int_ppl_fair, 3),
                "gap_int_vs_iso_pct": round(gap_int_vs_iso, 3),
                "gap_int_vs_perseq_pct": round(gap_int_vs_perseq, 3),
                "boundary": boundary,
                "T": T,
            })

            all_gaps_vs_iso.append(gap_int_vs_iso)
            all_gaps_vs_perseq.append(gap_int_vs_perseq)

            del (nll_perseq_A, nll_perseq_B, nll_integrated, nll_seg_A, nll_seg_B,
                 bd_mask, causal_mask, iso_nll_A, iso_nll_B, iso_mask_A, iso_mask_B)
            gc.collect()

        results["pairs"].append({"pair": f"{d_A}+{d_B}", "samples": pair_results})

    del adapters
    gc.collect()
    mx.clear_cache()

    results["summary"] = {
        "mean_gap_vs_iso_pct": round(float(np.mean(all_gaps_vs_iso)), 3) if all_gaps_vs_iso else 0,
        "mean_gap_vs_perseq_pct": round(float(np.mean(all_gaps_vs_perseq)), 3) if all_gaps_vs_perseq else 0,
        "max_gap_vs_iso_pct": round(float(np.max(all_gaps_vs_iso)), 3) if all_gaps_vs_iso else 0,
        "max_gap_vs_perseq_pct": round(float(np.max(all_gaps_vs_perseq)), 3) if all_gaps_vs_perseq else 0,
        "n_samples": len(all_gaps_vs_iso),
    }

    log(f"\n  Summary: mean gap vs iso = {results['summary']['mean_gap_vs_iso_pct']:.3f}%")
    log(f"  Summary: mean gap vs perseq = {results['summary']['mean_gap_vs_perseq_pct']:.3f}%")

    cleanup(model, tok)
    log_memory("after integrated pipeline cleanup")
    return results


def phase_speed(model_id, frozen_A):
    """Phase 4: Speed measurement of ACTUAL integrated pipeline forward pass.

    This measures the real integrated forward pass with block-diagonal mask
    + per-token MLP routing, NOT single-adapter mlx_generate (which was the
    confound in Finding #323).
    """
    log("\n=== Phase 4: Speed (Integrated Pipeline Forward Pass) ===")

    model, tok = load(model_id)

    # Load 2 DARE-sparsified adapters for the speed test
    adapter_A_raw = dict(mx.load(str(ADAPTER_DIR / "medical" / "adapter.npz")))
    adapter_B_raw = dict(mx.load(str(ADAPTER_DIR / "code" / "adapter.npz")))
    adapter_A = dare_sparsify(adapter_A_raw, p=DARE_P, seed=SEED)
    adapter_B = dare_sparsify(adapter_B_raw, p=DARE_P, seed=SEED + 1)
    del adapter_A_raw, adapter_B_raw

    # Create a representative mixed input
    text_A = load_data("medical", "valid", 1)[0]
    text_B = load_data("code", "valid", 1)[0]
    tokens_A = tok.encode(text_A)[:SEGMENT_LEN]
    tokens_B = tok.encode(text_B)[:SEGMENT_LEN]
    combined = tokens_A + tokens_B
    boundary = len(tokens_A)
    T = len(combined)

    bd_mask = create_block_diagonal_mask_2seg(T, boundary).astype(mx.bfloat16)

    # Warmup
    log("  Warming up...")
    for _ in range(3):
        nll = single_pass_mixed_mlp_forward(
            model, combined, bd_mask, frozen_A,
            adapter_A, adapter_B, 0, 1, boundary
        )
        del nll
    gc.collect()

    # Benchmark: measure forward pass time
    log("  Benchmarking integrated forward pass...")
    times = []
    for _ in range(10):
        t1 = time.time()
        nll = single_pass_mixed_mlp_forward(
            model, combined, bd_mask, frozen_A,
            adapter_A, adapter_B, 0, 1, boundary
        )
        mx.eval(nll)
        dt = time.time() - t1
        times.append({"s": dt, "tokens": T})
        del nll

    avg_time = float(np.mean([t["s"] for t in times]))
    integrated_tps = T / avg_time
    log(f"  Integrated pipeline: {integrated_tps:.1f} tok/s "
        f"(avg {avg_time*1000:.1f}ms for {T} tokens, {len(times)} runs)")

    # Also measure single-adapter mlx_generate for comparison
    log("  Measuring single-adapter mlx_generate for comparison...")
    attach_adapter(model, frozen_A, adapter_A, 0, LORA_SCALE)
    prompt = "### Instruction:\nExplain photosynthesis.\n\n### Response:\n"
    sampler = make_sampler(temp=0.0)

    for _ in range(2):
        mlx_generate(model, tok, prompt=prompt, max_tokens=32, sampler=sampler, verbose=False)

    gen_times = []
    for _ in range(5):
        t1 = time.time()
        out = mlx_generate(model, tok, prompt=prompt, max_tokens=128, sampler=sampler, verbose=False)
        dt = time.time() - t1
        n_toks = len(tok.encode(out)) - len(tok.encode(prompt))
        if n_toks > 0:
            gen_times.append({"s": dt, "toks": n_toks})

    gen_tps = sum(t["toks"] for t in gen_times) / sum(t["s"] for t in gen_times) if gen_times else 0
    log(f"  Single-adapter mlx_generate: {gen_tps:.1f} tok/s")

    detach_adapters(model)
    cleanup(model, tok, adapter_A, adapter_B)
    log_memory("after speed cleanup")

    return {
        "integrated_tps": round(integrated_tps, 1),
        "generate_tps": round(gen_tps, 1),
        "integrated_avg_ms": round(avg_time * 1000, 1),
        "n_tokens": T,
        "n_runs": len(times),
    }


def phase_behavioral_routed(model_id, frozen_A, W):
    """Phase 5: Behavioral evaluation with router + DARE adapters."""
    log("\n=== Phase 5: Behavioral (Routed + DARE) ===")

    model, tok = load(model_id)
    behavioral = {}
    sampler = make_sampler(temp=0.0)

    for di, d in enumerate(DOMAINS):
        test = load_data(d, "valid", N_EVAL)
        if not test:
            log(f"  SKIP {d}: no data")
            continue

        # Route to determine domain
        ri = route(model, tok, test[0], W, MAX_SEQ)
        rd = DOMAINS[ri]

        # Load DARE-sparsified adapter for routed domain
        adapter_path = ADAPTER_DIR / rd / "adapter.npz"
        if not adapter_path.exists():
            log(f"  SKIP {d}: no adapter for routed domain {rd}")
            continue
        raw = dict(mx.load(str(adapter_path)))
        adapter_dare = dare_sparsify(raw, p=DARE_P, seed=SEED)
        del raw

        attach_adapter(model, frozen_A, adapter_dare, ri, LORA_SCALE)

        scores = []
        for text in test:
            if "### Response:" in text:
                prompt = text.split("### Response:")[0].strip() + "\n### Response:\n"
                ref = text.split("### Response:")[-1].strip()
            else:
                prompt, ref = text[:200], text
            try:
                gen = mlx_generate(
                    model, tok, prompt=prompt, max_tokens=128,
                    sampler=sampler, verbose=False
                )
                scores.append(factual_recall(gen, ref))
            except Exception:
                scores.append(0.0)

        mean = float(np.mean(scores)) if scores else 0.0
        behavioral[d] = {
            "score": round(mean, 3),
            "routed_to": rd,
            "correct": rd == d,
            "n_samples": len(scores),
        }
        log(f"  {d} -> {rd}: {mean:.3f} {'(correct)' if rd == d else '(MISROUTED)'}")

        detach_adapters(model)
        del adapter_dare
        gc.collect()

    cleanup(model, tok)
    log_memory("after behavioral routed cleanup")
    return behavioral


# ============================================================
# Main orchestrator
# ============================================================

def main():
    t0 = time.time()
    log("Pierre Pro: Integrated Serving Pipeline on Qwen3-4B-4bit")
    log("=" * 60)
    log(f"Components: block-diag mask + per-token MLP routing + DARE p={DARE_P} + ridge router")
    log(f"LORA_SCALE = {LORA_SCALE} (NOT 20 -- Finding #330)")
    log(f"Kill: K821 (behavioral >= 0.3)")
    log("=" * 60)
    mx.random.seed(SEED)

    # Load skeleton (shared across all phases)
    frozen_A = dict(np.load(str(INIT_DIR / "grassmannian_skeleton_n5.npz")))
    log(f"Skeleton loaded: {len(frozen_A)} matrices")

    # Get model ID from base validation
    base_data = {}
    if (BASE_DIR / "results.json").exists():
        base_data = json.loads((BASE_DIR / "results.json").read_text())
    model_id = base_data.get("model_id", "mlx-community/Qwen3-4B-4bit")
    log(f"Model: {model_id}")
    log_memory("start")

    # Phase 1: Router calibration
    W, routing_accuracy = phase_calibrate_router(model_id, frozen_A)

    # Phase 2: Per-domain isolated behavioral quality
    isolated_behavioral = phase_behavioral_isolated(model_id, frozen_A)

    # Phase 3: Integrated pipeline PPL comparison
    integrated = phase_integrated_pipeline(model_id, frozen_A)

    # Phase 4: Speed measurement (actual integrated pipeline)
    speed = phase_speed(model_id, frozen_A)

    # Phase 5: Behavioral with routing + DARE
    routed_behavioral = phase_behavioral_routed(model_id, frozen_A, W)

    # ---- Aggregate results ----
    overall_isolated = float(np.mean(
        [v["score"] for v in isolated_behavioral.values()]
    ))
    overall_routed = float(np.mean(
        [v["score"] for v in routed_behavioral.values()]
    ))

    # K821: behavioral >= 0.3 (use routed behavioral as the pipeline output)
    k821_pass = overall_routed >= 0.3

    # Also check isolated for reference
    mean_gap_vs_perseq = integrated["summary"]["mean_gap_vs_perseq_pct"]
    mean_gap_vs_iso = integrated["summary"]["mean_gap_vs_iso_pct"]

    results = {
        "experiment": "pro_integrated_serving",
        "model_id": model_id,
        "lora_scale": LORA_SCALE,
        "dare_p": DARE_P,
        "total_time_s": round(time.time() - t0, 1),
        "routing_accuracy": round(routing_accuracy, 4),
        "isolated_behavioral": isolated_behavioral,
        "overall_isolated_behavioral": round(overall_isolated, 3),
        "integrated_pipeline": integrated,
        "speed": speed,
        "routed_behavioral": routed_behavioral,
        "overall_routed_behavioral": round(overall_routed, 3),
        "kill_criteria": {
            "K821": {
                "pass": bool(k821_pass),
                "metric": "overall_routed_behavioral",
                "value": round(overall_routed, 3),
                "threshold": 0.3,
                "detail": (f"Routed behavioral score {overall_routed:.3f} "
                           f"(threshold: >= 0.3)")
            },
        },
        "supplementary": {
            "integrated_vs_perseq_gap_pct": round(mean_gap_vs_perseq, 3),
            "integrated_vs_iso_gap_pct": round(mean_gap_vs_iso, 3),
            "integrated_speed_tps": speed["integrated_tps"],
            "generate_speed_tps": speed["generate_tps"],
        },
        "all_pass": bool(k821_pass),
    }

    log(f"\n{'=' * 60}")
    log("RESULTS SUMMARY")
    log(f"{'=' * 60}")
    log(f"Routing accuracy:          {routing_accuracy:.1%}")
    log(f"Isolated behavioral (avg): {overall_isolated:.3f}")
    log(f"Routed behavioral (avg):   {overall_routed:.3f}")
    log(f"Integrated vs isolated:    {mean_gap_vs_iso:+.1f}%")
    log(f"Integrated vs per-seq:     {mean_gap_vs_perseq:+.1f}%")
    log(f"Integrated pipeline speed: {speed['integrated_tps']:.1f} tok/s")
    log(f"mlx_generate speed:        {speed['generate_tps']:.1f} tok/s")
    log("")
    log("Per-domain (isolated):")
    for d, v in isolated_behavioral.items():
        log(f"  {d}: {v['score']:.3f}")
    log("Per-domain (routed + DARE):")
    for d, v in routed_behavioral.items():
        log(f"  {d} -> {v['routed_to']}: {v['score']:.3f} {'(correct)' if v['correct'] else '(MISROUTED)'}")
    log("")
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} -- {v['detail']}")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'} in {results['total_time_s']:.0f}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
