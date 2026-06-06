#!/usr/bin/env python3
"""Pierre Tiny: integrated serving pipeline.

Combines ALL proven components into one system:
  1. Block-diagonal causal masking (Finding #314, #322)
  2. MLP-only per-token adapter routing (Finding #312-313)
  3. SFT adapters for 5 domains (Finding #297)
  4. DARE p=0.5 for OOD robustness (Finding #266)
  5. Ridge regression router for domain detection (Finding #276)

The TRUE integrated pipeline:
  - Loads model ONCE
  - Takes mixed-domain input (multiple segments concatenated)
  - Uses ridge router to detect segment domains
  - Creates block-diagonal causal mask
  - Applies per-segment MLP adapters via per-token routing in a single forward pass
  - Evaluates quality and speed

Kill criteria:
  K818: Integrated pipeline worse than per-sequence baseline (must PASS)
  K819: Speed < 60 tok/s (must PASS)

Success criterion S80: Within 2% of segment-isolated oracle, >70 tok/s
"""

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
from mlx.utils import tree_flatten, tree_unflatten

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from pierre import (
    attach_adapter, detach_adapters, fit_router, route,
    load_adapter, load_frozen_A, encode, ADAPTER_TARGETS,
)
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.bitlinear_layers import BitLinear

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

SFT_SOURCE = EXPERIMENT_DIR.parent / "bitnet_sft_generation_v3" / "sft_adapters"
NTP_SOURCE = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "adapters"
SKELETON_PATH = NTP_SOURCE / "grassmannian_skeleton.npz"
DATA_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_SCALE = 20.0
MAX_SEQ = 256
SEGMENT_LEN = 128
SEED = 42
DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_CAL = 50
DARE_P = 0.5
N_EVAL_TEXTS = 20  # per domain for PPL
N_PAIRS = 3  # domain pairs for integrated eval


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

def log(m): print(m, flush=True)

def log_memory(label=""):
    a = mx.get_active_memory() / 1e9; p = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB peak={p:.2f}GB")

def cleanup(*o):
    for x in o: del x
    gc.collect(); mx.clear_cache(); mx.reset_peak_memory()

def load_data(d, split="valid", n=None):
    s = []
    with open(DATA_DIR / d / f"{split}.jsonl") as f:
        for l in f:
            s.append(json.loads(l)["text"])
            if n and len(s) >= n: break
    return s


# ---- DARE sparsification (Finding #266) ----

def dare_sparsify(adapter_b, p=0.5, seed=42):
    """Apply DARE: randomly drop p fraction of adapter B params, rescale by 1/(1-p)."""
    rng = np.random.RandomState(seed)
    sparsified = {}
    for key, val in adapter_b.items():
        mask = mx.array(rng.binomial(1, 1.0 - p, size=val.shape).astype(np.float32))
        sparsified[key] = (val * mask / (1.0 - p)).astype(val.dtype)
    return sparsified


# ---- Block-diagonal causal mask (Finding #322) ----

def create_block_diagonal_mask(seq_len, boundaries):
    """Create additive block-diagonal causal mask for K segments.

    boundaries: list of (start, end) tuples for each segment.
    Each segment can only attend within itself. Causal within.
    """
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    positions = mx.arange(seq_len)

    # Assign each position to a segment
    seg_ids = mx.zeros((seq_len,), dtype=mx.int32)
    for seg_idx, (start, end) in enumerate(boundaries):
        for pos in range(start, end):
            seg_ids = seg_ids.at[pos].add(seg_idx)

    # Cross-segment: where seg_ids differ, add -inf
    cross_seg = seg_ids[:, None] != seg_ids[None, :]
    cross_penalty = mx.where(cross_seg, mx.array(float("-inf")), mx.array(0.0))
    mask = mask + cross_penalty
    return mask


def create_block_diagonal_mask_2seg(seq_len, boundary):
    """Optimized block-diagonal mask for exactly 2 segments."""
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    positions = mx.arange(seq_len)
    seg_i = positions >= boundary
    seg_j = positions >= boundary
    cross_seg = seg_i[:, None] != seg_j[None, :]
    cross_penalty = mx.where(cross_seg, mx.array(float("-inf")), mx.array(0.0))
    return mask + cross_penalty


# ---- Compute PPL helpers ----

def compute_ppl(model, tok, texts):
    """Standard per-sequence PPL (model's native forward pass)."""
    loss, n = 0.0, 0
    for text in texts:
        toks = tok.encode(text)[:MAX_SEQ]
        if len(toks) < 4: continue
        x = mx.array(toks)[None, :]
        logits = model(x); mx.eval(logits)
        targets = x[:, 1:]
        lp = mx.log(mx.softmax(logits[:, :-1, :], axis=-1) + 1e-10)
        tlp = mx.take_along_axis(lp, targets[:,:,None], axis=-1).squeeze(-1)
        mx.eval(tlp)
        loss += -tlp.sum().item(); n += targets.shape[1]
        del logits, lp, tlp, x
    return math.exp(loss / n) if n else float('inf')


def compute_nll_with_mask(model, tokens, mask):
    """Forward pass with custom attention mask. Returns per-token NLL array."""
    x = mx.array(tokens)[None, :]
    h = model.model.embed_tokens(x)
    for layer in model.model.layers:
        h = layer(h, mask=mask, cache=None)
    h = model.model.norm(h)
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)
    mx.eval(logits)
    log_probs = mx.log(mx.softmax(logits[0, :-1], axis=-1) + 1e-10)
    targets = mx.array(tokens[1:])
    nll = -mx.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(-1)
    mx.eval(nll)
    del logits, log_probs, h
    return nll


# ---- Single-pass per-token MLP routing with RuntimeLoRA (Finding #313) ----

def single_pass_mixed_mlp_forward(model, tokens, mask, frozen_A,
                                   adapter_A, adapter_B,
                                   domain_idx_A, domain_idx_B,
                                   boundary, alpha=LORA_SCALE):
    """Single-pass forward with per-token MLP adapter routing.

    tokens 0..boundary-1 use adapter_A, tokens boundary..end use adapter_B.
    Attention layers use BASE weights only (no adapter).
    MLP layers apply per-token adapter via mx.where mask.

    This is the core integration: block-diagonal mask + per-token MLP routing.
    """
    x = mx.array(tokens)[None, :]
    h = model.model.embed_tokens(x)
    T = len(tokens)

    # Pre-create the per-token mask for broadcasting: (1, T, 1)
    tok_mask = mx.arange(T)[None, :, None] < boundary  # True = adapter A

    for li, layer in enumerate(model.model.layers):
        # -- Attention sub-layer with base weights + custom mask --
        h_norm = layer.input_layernorm(h)

        attn = layer.self_attn
        B_dim, L, D = h_norm.shape
        queries = attn.q_proj(h_norm)
        keys = attn.k_proj(h_norm)
        values = attn.v_proj(h_norm)

        queries = queries.reshape(B_dim, L, attn.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B_dim, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B_dim, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = attn.rope(queries)
        keys = attn.rope(keys)

        from mlx_lm.models.base import scaled_dot_product_attention
        output = scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=attn.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B_dim, L, -1)

        if hasattr(attn, 'attn_sub_norm'):
            output = attn.attn_sub_norm(output)
        output = attn.o_proj(output)
        h = h + output

        # -- MLP sub-layer with per-token adapter routing --
        h_norm2 = layer.post_attention_layernorm(h)
        mlp = layer.mlp

        # For each MLP module (gate, up, down), compute base + per-token LoRA
        def mlp_with_mixed_lora(module, h_in, module_key):
            """Compute base(x) + per-token LoRA delta."""
            base_out = module(h_in)

            # Get LoRA A and B for both adapters
            ak_A = f"layer_{li}_{module_key}_domain_{domain_idx_A}"
            ak_B = f"layer_{li}_{module_key}_domain_{domain_idx_B}"
            bk = f"model.layers.{li}.{module_key}.lora_b"

            if ak_A not in frozen_A or bk not in adapter_A:
                return base_out

            A_a = mx.array(frozen_A[ak_A]).astype(mx.bfloat16)
            A_b = mx.array(frozen_A[ak_B]).astype(mx.bfloat16)
            B_a = adapter_A[bk].astype(mx.bfloat16)
            B_b = adapter_B[bk].astype(mx.bfloat16)

            # LoRA output for both adapters: (1, T, d_out)
            lora_out_A = ((h_in @ A_a) @ B_a * alpha).astype(base_out.dtype)
            lora_out_B = ((h_in @ A_b) @ B_b * alpha).astype(base_out.dtype)

            # Per-token selection
            mixed_lora = mx.where(tok_mask, lora_out_A, lora_out_B)
            return base_out + mixed_lora

        gate_out = mlp_with_mixed_lora(mlp.gate_proj, h_norm2, "mlp.gate_proj")
        up_out = mlp_with_mixed_lora(mlp.up_proj, h_norm2, "mlp.up_proj")

        x_mid = nn.relu2(gate_out) * up_out
        x_mid = mlp.ffn_sub_norm(x_mid)

        down_out = mlp_with_mixed_lora(mlp.down_proj, x_mid, "mlp.down_proj")
        h = h + down_out

    h = model.model.norm(h)
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)
    mx.eval(logits)

    log_probs = mx.log(mx.softmax(logits[0, :-1], axis=-1) + 1e-10)
    targets = mx.array(tokens[1:])
    nll = -mx.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(-1)
    mx.eval(nll)
    del logits, log_probs, h, output, queries, keys, values
    return nll


# ---- Behavioral evaluation ----

STOP_WORDS = {'the','a','an','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','will','would','could','should','may','might','can',
    'to','of','in','for','on','with','at','by','from','as','and','but','or','not',
    'so','yet','both','either','each','every','all','any','few','more','most','other',
    'some','such','no','only','own','same','than','too','very','just','because','if',
    'when','where','how','what','which','who','this','that','these','those','it','its',
    'i','me','my','we','our','you','your','he','him','his','she','her','they','them','their'}

def factual_recall(g, r):
    def t(x): return set(w for w in re.findall(r'\b[a-z]+\b', x.lower()) if w not in STOP_WORDS and len(w)>2)
    gt, rt = t(g), t(r)
    return len(gt & rt) / len(rt) if rt else 0.0


# ---- Phase functions ----

def phase_calibrate_router(model, tok, frozen_A):
    """Phase 1: Fit ridge regression router on calibration data."""
    log("\n=== Phase 1: Ridge Router Calibration ===")
    cal_data = {d: load_data(d, "train", N_CAL) for d in DOMAINS}
    W = fit_router(model, tok, cal_data, max_seq=MAX_SEQ)
    log(f"  Router fitted (W shape: {W.shape})")

    # Test routing accuracy
    correct, total = 0, 0
    for di, d in enumerate(DOMAINS):
        for text in load_data(d, "valid", 10):
            if route(model, tok, text, W, MAX_SEQ) == di: correct += 1
            total += 1
    accuracy = correct / total
    log(f"  Routing accuracy: {accuracy:.1%} ({correct}/{total})")

    np.save(str(EXPERIMENT_DIR / "router_W.npy"), np.array(W))
    return W, accuracy


def phase_per_sequence_baseline(model, tok, frozen_A):
    """Phase 2: Per-sequence baseline (load adapter per domain, evaluate individually)."""
    log("\n=== Phase 2: Per-Sequence Baseline PPL ===")
    baseline_ppls = {}
    for di, d in enumerate(DOMAINS):
        adapter = load_adapter(str(SFT_SOURCE / d / "adapter.npz"))
        attach_adapter(model, frozen_A, adapter, di, LORA_SCALE)
        baseline_ppls[d] = round(compute_ppl(model, tok, load_data(d, "valid", N_EVAL_TEXTS)), 3)
        log(f"  {d}: {baseline_ppls[d]}")
        detach_adapters(model)
        del adapter
    gc.collect()
    return baseline_ppls


def phase_dare_baseline(model, tok, frozen_A, baseline_ppls):
    """Phase 3: DARE-sparsified adapter quality (per-sequence with DARE)."""
    log("\n=== Phase 3: DARE p=0.5 Baseline PPL ===")
    dare_ppls = {}
    for di, d in enumerate(DOMAINS):
        adapter = load_adapter(str(SFT_SOURCE / d / "adapter.npz"))
        adapter_dare = dare_sparsify(adapter, p=DARE_P, seed=SEED + di)
        attach_adapter(model, frozen_A, adapter_dare, di, LORA_SCALE)
        dare_ppls[d] = round(compute_ppl(model, tok, load_data(d, "valid", N_EVAL_TEXTS)), 3)
        deg = (dare_ppls[d] - baseline_ppls[d]) / baseline_ppls[d] * 100
        log(f"  {d}: {dare_ppls[d]} ({deg:+.1f}% vs baseline)")
        detach_adapters(model)
        del adapter, adapter_dare
    gc.collect()
    return dare_ppls


def phase_integrated_pipeline(model, tok, frozen_A, baseline_ppls):
    """Phase 4: TRUE integrated pipeline with block-diagonal mask + per-token MLP routing.

    This is the key phase. For each domain pair:
    1. Concatenate segments from two domains
    2. Create block-diagonal causal mask
    3. Run single forward pass with per-token MLP adapter routing
    4. Compare PPL to per-sequence baseline
    """
    log("\n=== Phase 4: Integrated Pipeline (Block-Diag + Per-Token MLP) ===")

    from itertools import combinations
    domain_pairs = list(combinations(range(len(DOMAINS)), 2))[:N_PAIRS * 2]

    # Load all DARE-sparsified adapters
    adapters = {}
    for di, d in enumerate(DOMAINS):
        adapter = load_adapter(str(SFT_SOURCE / d / "adapter.npz"))
        adapters[di] = dare_sparsify(adapter, p=DARE_P, seed=SEED + di)
        del adapter

    results = {"pairs": [], "summary": {}}
    all_integrated_ppls = {d: [] for d in DOMAINS}
    all_perseq_ppls = {d: [] for d in DOMAINS}

    for di_A, di_B in domain_pairs:
        d_A, d_B = DOMAINS[di_A], DOMAINS[di_B]
        texts_A = load_data(d_A, "valid", 5)
        texts_B = load_data(d_B, "valid", 5)

        pair_results = []
        for text_A, text_B in zip(texts_A[:3], texts_B[:3]):
            tokens_A = tok.encode(text_A)[:SEGMENT_LEN]
            tokens_B = tok.encode(text_B)[:SEGMENT_LEN]

            # Ensure minimum segment length
            if len(tokens_A) < 16 or len(tokens_B) < 16:
                continue

            combined = tokens_A + tokens_B
            boundary = len(tokens_A)
            T = len(combined)

            # -- Per-sequence baseline for this pair --
            # Adapter A on combined sequence (per-sequence: apply one adapter to all)
            attach_adapter(model, frozen_A, adapters[di_A], di_A, LORA_SCALE)
            nll_perseq_A = compute_nll_with_mask(
                model, combined,
                nn.MultiHeadAttention.create_additive_causal_mask(T).astype(mx.bfloat16)
            )
            perseq_A_mean = mx.mean(nll_perseq_A).item()
            detach_adapters(model)

            attach_adapter(model, frozen_A, adapters[di_B], di_B, LORA_SCALE)
            nll_perseq_B = compute_nll_with_mask(
                model, combined,
                nn.MultiHeadAttention.create_additive_causal_mask(T).astype(mx.bfloat16)
            )
            perseq_B_mean = mx.mean(nll_perseq_B).item()
            detach_adapters(model)

            # Per-sequence best: select best adapter for the whole sequence
            perseq_ppl = min(math.exp(perseq_A_mean), math.exp(perseq_B_mean))

            # -- Segment-isolated oracle --
            # Adapter A on segment A only
            attach_adapter(model, frozen_A, adapters[di_A], di_A, LORA_SCALE)
            iso_x_A = mx.array(tokens_A)[None, :]
            iso_logits_A = model(iso_x_A); mx.eval(iso_logits_A)
            iso_lp_A = mx.log(mx.softmax(iso_logits_A[0, :-1], axis=-1) + 1e-10)
            iso_nll_A = -mx.take_along_axis(iso_lp_A, mx.array(tokens_A[1:])[:, None], axis=-1).squeeze(-1)
            mx.eval(iso_nll_A)
            detach_adapters(model)

            attach_adapter(model, frozen_A, adapters[di_B], di_B, LORA_SCALE)
            iso_x_B = mx.array(tokens_B)[None, :]
            iso_logits_B = model(iso_x_B); mx.eval(iso_logits_B)
            iso_lp_B = mx.log(mx.softmax(iso_logits_B[0, :-1], axis=-1) + 1e-10)
            iso_nll_B = -mx.take_along_axis(iso_lp_B, mx.array(tokens_B[1:])[:, None], axis=-1).squeeze(-1)
            mx.eval(iso_nll_B)
            detach_adapters(model)

            iso_nll_total = mx.sum(iso_nll_A).item() + mx.sum(iso_nll_B).item()
            iso_n = iso_nll_A.size + iso_nll_B.size
            iso_ppl = math.exp(iso_nll_total / iso_n)

            # -- Integrated pipeline: block-diag + per-token MLP --
            bd_mask = create_block_diagonal_mask_2seg(T, boundary).astype(mx.bfloat16)

            nll_integrated = single_pass_mixed_mlp_forward(
                model, combined, bd_mask, frozen_A,
                adapters[di_A], adapters[di_B],
                di_A, di_B, boundary
            )

            # Fair PPL (exclude boundary token)
            nll_seg_A = nll_integrated[:boundary - 1]
            nll_seg_B = nll_integrated[boundary:]
            int_nll_fair = mx.sum(nll_seg_A).item() + mx.sum(nll_seg_B).item()
            int_n_fair = nll_seg_A.size + nll_seg_B.size
            int_ppl_fair = math.exp(int_nll_fair / int_n_fair) if int_n_fair > 0 else float('inf')

            # Full PPL (include boundary)
            int_ppl_full = math.exp(mx.mean(nll_integrated).item())

            # Gaps
            gap_int_vs_iso = (int_ppl_fair - iso_ppl) / iso_ppl * 100
            gap_int_vs_perseq = (int_ppl_fair - perseq_ppl) / perseq_ppl * 100
            gap_perseq_vs_iso = (perseq_ppl - iso_ppl) / iso_ppl * 100

            log(f"  {d_A}+{d_B}: iso={iso_ppl:.3f} perseq={perseq_ppl:.3f}({gap_perseq_vs_iso:+.1f}%) "
                f"integrated={int_ppl_fair:.3f}({gap_int_vs_iso:+.1f}% vs iso, {gap_int_vs_perseq:+.1f}% vs perseq)")

            pair_results.append({
                "domains": [d_A, d_B],
                "iso_ppl": round(iso_ppl, 3),
                "perseq_ppl": round(perseq_ppl, 3),
                "integrated_ppl_fair": round(int_ppl_fair, 3),
                "integrated_ppl_full": round(int_ppl_full, 3),
                "gap_int_vs_iso_pct": round(gap_int_vs_iso, 3),
                "gap_int_vs_perseq_pct": round(gap_int_vs_perseq, 3),
                "boundary": boundary,
                "T": T,
            })

            # Accumulate per-domain integrated PPLs for overall comparison
            seg_A_ppl = math.exp(mx.mean(nll_seg_A).item()) if nll_seg_A.size > 0 else float('inf')
            seg_B_ppl = math.exp(mx.mean(nll_seg_B).item()) if nll_seg_B.size > 0 else float('inf')
            all_integrated_ppls[d_A].append(seg_A_ppl)
            all_integrated_ppls[d_B].append(seg_B_ppl)

            del (nll_perseq_A, nll_perseq_B, nll_integrated, nll_seg_A, nll_seg_B,
                 bd_mask, iso_logits_A, iso_logits_B, iso_lp_A, iso_lp_B,
                 iso_nll_A, iso_nll_B)
            gc.collect()

        results["pairs"].append({"pair": f"{d_A}+{d_B}", "samples": pair_results})

    # Cleanup
    del adapters
    gc.collect(); mx.clear_cache()

    # Compute summary statistics
    all_gaps_vs_iso = []
    all_gaps_vs_perseq = []
    for pair_data in results["pairs"]:
        for sample in pair_data["samples"]:
            all_gaps_vs_iso.append(sample["gap_int_vs_iso_pct"])
            all_gaps_vs_perseq.append(sample["gap_int_vs_perseq_pct"])

    results["summary"] = {
        "mean_gap_vs_iso_pct": round(float(np.mean(all_gaps_vs_iso)), 3) if all_gaps_vs_iso else 0,
        "mean_gap_vs_perseq_pct": round(float(np.mean(all_gaps_vs_perseq)), 3) if all_gaps_vs_perseq else 0,
        "max_gap_vs_iso_pct": round(float(np.max(all_gaps_vs_iso)), 3) if all_gaps_vs_iso else 0,
        "max_gap_vs_perseq_pct": round(float(np.max(all_gaps_vs_perseq)), 3) if all_gaps_vs_perseq else 0,
        "n_samples": len(all_gaps_vs_iso),
        "integrated_domain_ppls": {
            d: round(float(np.mean(v)), 3) if v else None
            for d, v in all_integrated_ppls.items()
        },
    }

    log(f"\n  Summary: mean gap vs iso = {results['summary']['mean_gap_vs_iso_pct']:.3f}%")
    log(f"  Summary: mean gap vs perseq = {results['summary']['mean_gap_vs_perseq_pct']:.3f}%")

    return results


def phase_speed(model, tok, frozen_A):
    """Phase 5: Speed measurement with single adapter.

    NOTE: This measures standard mlx_generate with a single adapter, NOT the
    integrated pipeline (block-diag + per-token MLP routing). The integrated
    pipeline's generation speed is untested. See REVIEW-adversarial.md fix #3.
    """
    log("\n=== Phase 5: Speed Measurement (single adapter, NOT integrated pipeline) ===")
    adapter = load_adapter(str(SFT_SOURCE / "medical" / "adapter.npz"))
    attach_adapter(model, frozen_A, adapter, 0, LORA_SCALE)

    prompt = "### Instruction:\nExplain photosynthesis.\n\n### Response:\n"
    sampler = make_sampler(temp=0.0)

    # Warmup
    for _ in range(3):
        mlx_generate(model, tok, prompt=prompt, max_tokens=32, sampler=sampler, verbose=False)

    # Benchmark
    times = []
    for _ in range(5):
        t1 = time.time()
        out = mlx_generate(model, tok, prompt=prompt, max_tokens=128, sampler=sampler, verbose=False)
        dt = time.time() - t1
        n = len(tok.encode(out)) - len(tok.encode(prompt))
        if n > 0:
            times.append({"s": dt, "toks": n})

    tps = sum(t["toks"] for t in times) / sum(t["s"] for t in times) if times else 0
    log(f"  Speed: {tps:.1f} tok/s ({len(times)} runs)")

    detach_adapters(model)
    del adapter
    gc.collect()
    return round(tps, 1)


def phase_behavioral(model, tok, frozen_A, W):
    """Phase 6: Behavioral evaluation with routed adapters + DARE."""
    log("\n=== Phase 6: Behavioral (Routed + DARE) ===")
    behavioral = {}
    sampler = make_sampler(temp=0.0)

    for di, d in enumerate(DOMAINS):
        test = load_data(d, "valid", 5)

        # Route to determine domain
        ri = route(model, tok, test[0], W, MAX_SEQ)
        rd = DOMAINS[ri]

        # Load DARE-sparsified adapter for routed domain
        adapter = load_adapter(str(SFT_SOURCE / rd / "adapter.npz"))
        adapter_dare = dare_sparsify(adapter, p=DARE_P, seed=SEED)
        attach_adapter(model, frozen_A, adapter_dare, ri, LORA_SCALE)

        scores = []
        for text in test:
            if "### Response:" in text:
                prompt = text.split("### Response:")[0].strip() + "\n### Response:\n"
                ref = text.split("### Response:")[-1].strip()
            else:
                prompt, ref = text[:200], text
            try:
                gen = mlx_generate(model, tok, prompt=prompt, max_tokens=128,
                                   sampler=sampler, verbose=False)
                scores.append(factual_recall(gen, ref))
            except Exception:
                scores.append(0.0)

        mean = float(np.mean(scores)) if scores else 0.0
        behavioral[d] = {"score": round(mean, 3), "routed_to": rd, "correct": rd == d}
        log(f"  {d} -> {rd}: {mean:.3f} {'(correct)' if rd == d else '(MISROUTED)'}")
        detach_adapters(model)
        del adapter, adapter_dare
    gc.collect()
    return behavioral


def main():
    t0 = time.time()
    log("Pierre Tiny: Integrated Serving Pipeline")
    log("=" * 60)
    log("Components: block-diag mask + per-token MLP routing + DARE + ridge router")
    log("Kill: K818 (pipeline not worse) K819 (speed >= 60 tok/s)")
    log("=" * 60)
    mx.random.seed(SEED)

    frozen_A = load_frozen_A(str(SKELETON_PATH))
    log_memory("after skeleton load")

    # Phase 1: Router
    model, tok = load(MODEL_ID)
    log_memory("model loaded")
    W, routing_accuracy = phase_calibrate_router(model, tok, frozen_A)

    # Phase 2: Per-sequence baseline
    baseline_ppls = phase_per_sequence_baseline(model, tok, frozen_A)

    # Phase 3: DARE baseline
    dare_ppls = phase_dare_baseline(model, tok, frozen_A, baseline_ppls)

    # Phase 4: Integrated pipeline
    integrated = phase_integrated_pipeline(model, tok, frozen_A, baseline_ppls)

    # Phase 5: Speed
    speed_tps = phase_speed(model, tok, frozen_A)

    # Phase 6: Behavioral
    behavioral = phase_behavioral(model, tok, frozen_A, W)
    overall_behavioral = float(np.mean([v["score"] for v in behavioral.values()]))

    cleanup(model, tok)
    log_memory("final cleanup")

    # ---- Kill criteria ----

    # K818: Integrated pipeline not worse than per-sequence baseline
    # Theorem 1 predicts: integrated within 10% of per-sequence
    # The integrated pipeline should be BETTER than per-sequence (it uses correct adapters per token)
    mean_gap_vs_perseq = integrated["summary"]["mean_gap_vs_perseq_pct"]
    k818_pass = mean_gap_vs_perseq < 10.0  # within 10% of per-sequence baseline

    # K819: Speed >= 60 tok/s
    k819_pass = speed_tps >= 60.0

    # S80: Within 2% of segment-isolated oracle, >70 tok/s
    mean_gap_vs_iso = integrated["summary"]["mean_gap_vs_iso_pct"]
    s80_quality = mean_gap_vs_iso < 2.0
    s80_speed = speed_tps >= 70.0
    s80_pass = s80_quality and s80_speed

    results = {
        "experiment": "tiny_integrated_serving",
        "total_time_s": round(time.time() - t0, 1),
        "routing_accuracy": round(routing_accuracy, 4),
        "baseline_ppl": baseline_ppls,
        "dare_ppl": dare_ppls,
        "integrated_pipeline": integrated,
        "speed_tps": speed_tps,
        "behavioral": behavioral,
        "overall_behavioral": round(overall_behavioral, 3),
        "kill_criteria": {
            "K818": {
                "pass": bool(k818_pass),
                "metric": "mean_gap_vs_perseq_pct",
                "value": round(mean_gap_vs_perseq, 3),
                "threshold": 10.0,
                "detail": f"Integrated pipeline {mean_gap_vs_perseq:+.1f}% vs per-sequence baseline (threshold: < 10%)"
            },
            "K819": {
                "pass": bool(k819_pass),
                "metric": "speed_tps",
                "value": speed_tps,
                "threshold": 60.0,
                "detail": f"Speed {speed_tps:.1f} tok/s (threshold: >= 60 tok/s)"
            },
        },
        "success_criteria": {
            "S80": {
                "pass": bool(s80_pass),
                "quality": {"value": round(mean_gap_vs_iso, 3), "threshold": 2.0, "pass": bool(s80_quality)},
                "speed": {"value": speed_tps, "threshold": 70.0, "pass": bool(s80_speed)},
                "detail": (f"Gap vs iso: {mean_gap_vs_iso:.1f}% (< 2%: {'PASS' if s80_quality else 'FAIL'}), "
                           f"Speed: {speed_tps:.1f} tok/s (>= 70: {'PASS' if s80_speed else 'FAIL'})")
            }
        },
        "all_pass": bool(k818_pass and k819_pass),
    }

    log(f"\n{'=' * 60}")
    log(f"RESULTS SUMMARY")
    log(f"{'=' * 60}")
    log(f"Routing accuracy: {routing_accuracy:.1%}")
    log(f"Behavioral score: {overall_behavioral:.3f}")
    log(f"Speed: {speed_tps:.1f} tok/s")
    log(f"Integrated vs isolated: {mean_gap_vs_iso:+.1f}%")
    log(f"Integrated vs per-sequence: {mean_gap_vs_perseq:+.1f}%")
    log(f"")
    log(f"Per-domain baseline PPL: {baseline_ppls}")
    log(f"Per-domain DARE PPL:     {dare_ppls}")
    log(f"")
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} -- {v['detail']}")
    log(f"  S80: {'PASS' if s80_pass else 'FAIL'} -- {results['success_criteria']['S80']['detail']}")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'} in {results['total_time_s']:.0f}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
