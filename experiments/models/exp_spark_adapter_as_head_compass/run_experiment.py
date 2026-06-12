#!/usr/bin/env python3
"""
exp_spark_adapter_as_head_compass

HYPOTHESIS (frame-break): a net-negative math q_proj LoRA adapter (rank 6, scale 6.0) is NEVER applied
to the forward pass. Instead its per-layer B (output) column space is used as a READ-ONLY compass to
SCORE each base attention head's q_proj output block, then we AMPLIFY only the top-scoring BASE heads
by a small factor gamma. The adapter delta never enters the residual stream / logits.

Pre-registered kill 2310 (verbatim):
  "GSM8K EM (n>=60) for amplifying adapter-compass-selected base q_proj heads does NOT exceed BOTH
   (a) no-intervention base AND (b) amplifying random-selected heads (same count/factor) by >=+4pp
   each; OR best result requires applying the adapter delta to logits at all"

Three real arms (GSM8K EM, n>=60):
  (a) base            — no intervention
  (b) compass-amplify — amplify adapter-compass-selected top-K base heads by gamma (sweep)
  (c) random-amplify  — amplify same COUNT of randomly-selected base heads by same gamma (seed 1234)
  (ctx, labeled refuting) delta-applied — actually inject the adapter delta (scale 6.0). If this is the
       best arm, the hypothesis is refuted by clause 2 of kill 2310.

ASSERTION: the compass/random intervention is a pure coordinate-wise rescale of the BASE q_proj output;
no additive B@A term enters those arms. Enforced by `assert_no_delta_in_mask`.
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
REPO = Path("/Users/tom/Code/tomsiwik/llm")
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
ADAPTER_FILE = REPO / "data/adapters/math/adapters.safetensors"
GSM8K_PARQUET = REPO / "experiments/models/exp_p11_rsd_aligned_traces/data/gsm8k_test.parquet"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_EVAL = 6 if IS_SMOKE else 80          # >= 60 required for real
SEED = 42
RANDOM_ARM_SEED = 1234                   # fixed; reported
N_HEADS = 8
HEAD_DIM = 256
DQ = N_HEADS * HEAD_DIM                  # 2048
ADAPTER_SCALE = 6.0                      # the adapter's trained scale (only used in refuting ctx arm)
TOP_K_HEADS = 12                         # number of (layer,head) slots amplified (compass & random)
GAMMA_SWEEP = [1.2] if IS_SMOKE else [1.1, 1.2, 1.3, 1.5]

MAX_TOKENS = 512 if IS_SMOKE else 800


def log_mem(label=""):
    print(f"[MEM {label}] active={mx.get_active_memory()/1e9:.2f}GB cache={mx.get_cache_memory()/1e9:.2f}GB", flush=True)


def cleanup(*objs):
    for o in objs:
        del o
    gc.collect()
    mx.clear_cache()


# ──────────────────────────────────────────────────────────
# Compass: score base heads from frozen adapter B column space
# ──────────────────────────────────────────────────────────

def load_adapter_BA():
    """Return per-layer (A, B) numpy arrays from frozen math adapter (READ-ONLY)."""
    A, B = {}, {}
    with safe_open(str(ADAPTER_FILE), "numpy") as f:
        for k in f.keys():
            m = re.search(r"layers\.(\d+)\.self_attn\.q_proj\.lora_(a|b)", k)
            if not m:
                continue
            ell = int(m.group(1))
            if m.group(2) == "a":
                A[ell] = f.get_tensor(k)   # (2560, 6)
            else:
                B[ell] = f.get_tensor(k)   # (6, 2048)
    return A, B


def compass_scores(B: dict, head_dims: dict) -> dict:
    """score(ell,h) = ||B[:, head-block h]||_F^2 — energy the delta writes into head h's q-output block.

    Pure function of frozen adapter weights; no model forward, no x. Head-block width is the live
    per-layer head_dim (256 for sliding, 512 for full-attention layers); B output dim == n_heads*head_dim.
    """
    scores = {}
    for ell, Bm in B.items():
        hd = head_dims[ell]
        assert Bm.shape[1] == N_HEADS * hd, f"adapter B layer {ell} dim {Bm.shape[1]} != {N_HEADS*hd}"
        for h in range(N_HEADS):
            block = Bm[:, h * hd:(h + 1) * hd]
            # mean squared per coordinate so 256-dim (sliding) and 512-dim (full) heads compete fairly
            scores[(ell, h)] = float(np.mean(block * block))
    return scores


def select_top_heads(scores: dict, k: int):
    return sorted(scores.keys(), key=lambda kh: scores[kh], reverse=True)[:k]


def select_random_heads(all_slots, k: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_slots), size=k, replace=False)
    return [all_slots[i] for i in sorted(idx)]


def build_layer_masks(selected, gamma: float, head_dims: dict, qdims: dict):
    """Return {layer: mx.array (qdim,)} multiplicative mask; gamma on selected head blocks, else 1.0.

    The mask is built ONLY from integer (layer,head) selection + scalar gamma — never from delta values.
    """
    masks = {}
    for (ell, h) in selected:
        hd = head_dims[ell]
        if ell not in masks:
            masks[ell] = np.ones(qdims[ell], dtype=np.float32)
        masks[ell][h * hd:(h + 1) * hd] = gamma
    return {ell: mx.array(v) for ell, v in masks.items()}


def assert_no_delta_in_mask(masks, gamma):
    """Falsifiable assertion: every mask entry is exactly 1.0 or exactly gamma (no delta leakage)."""
    for ell, m in masks.items():
        vals = np.array(m).reshape(-1).astype(np.float64)
        is_one = np.isclose(vals, 1.0, atol=1e-5)
        is_gamma = np.isclose(vals, float(gamma), atol=1e-5)
        bad = ~(is_one | is_gamma)
        assert not bad.any(), f"mask layer {ell} has {int(bad.sum())} non-{{1,gamma}} values e.g. {vals[bad][:3]}"


# ──────────────────────────────────────────────────────────
# q_proj wrappers
# ──────────────────────────────────────────────────────────

class AmplifyQProj(nn.Module):
    """Wrap base q_proj: out = base_q_proj(x) * mask. Pure coordinate rescale of BASE output.

    NO adapter delta is added. This is the compass/random arm operator.
    """
    def __init__(self, base, mask):
        super().__init__()
        self.base = base
        self._mask = mask  # mx.array (2048,) of 1.0/gamma

    def __call__(self, x):
        return self.base(x) * self._mask


class DeltaQProj(nn.Module):
    """REFUTING context arm ONLY: out = base_q_proj(x) + scale * (x @ A) @ B (the adapter applied)."""
    def __init__(self, base, A, B, scale):
        super().__init__()
        self.base = base
        self._A = mx.array(A)
        self._B = mx.array(B)
        self._scale = scale

    def __call__(self, x):
        delta = (x @ self._A) @ self._B
        return self.base(x) + self._scale * delta


def get_layers(model):
    lm = model.language_model if hasattr(model, "language_model") else model
    return lm.model.layers


def install_amplify(model, masks):
    layers = get_layers(model)
    originals = {}
    for ell, m in masks.items():
        base = layers[ell].self_attn.q_proj
        originals[ell] = base
        layers[ell].self_attn.q_proj = AmplifyQProj(base, m)
    return originals


def install_delta(model, A, B, scale, layers_set):
    layers = get_layers(model)
    originals = {}
    for ell in sorted(layers_set):
        base = layers[ell].self_attn.q_proj
        originals[ell] = base
        layers[ell].self_attn.q_proj = DeltaQProj(base, A[ell], B[ell], scale)
    return originals


def layer_dims(model, B):
    """Per-layer (head_dim, q_proj_out_dim) from the live model; assert adapter B matches q_proj out."""
    layers = get_layers(model)
    head_dims, qdims = {}, {}
    for ell in B.keys():
        attn = layers[ell].self_attn
        qd = int(attn.q_proj.weight.shape[0])  # quantized: weight rows == output dim
        assert qd == B[ell].shape[1], f"layer {ell}: live q_proj out {qd} != adapter B {B[ell].shape[1]}"
        head_dims[ell] = int(attn.head_dim)
        qdims[ell] = qd
    return head_dims, qdims


def restore(model, originals):
    layers = get_layers(model)
    for ell, base in originals.items():
        layers[ell].self_attn.q_proj = base


# ──────────────────────────────────────────────────────────
# GSM8K eval (offline parquet, greedy, EM on #### answer)
# ──────────────────────────────────────────────────────────

def load_gsm8k(n):
    import pandas as pd
    df = pd.read_parquet(GSM8K_PARQUET)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df.iloc[:n].to_dict("records")


def extract_gt(answer):
    m = re.search(r"####\s*([\-\d,\.]+)", answer)
    return m.group(1).replace(",", "").strip() if m else None


def extract_pred(text):
    m = re.search(r"####\s*([\-\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1].replace(",", "").strip() if nums else None


def eval_em(model, tokenizer, examples):
    from mlx_lm import generate
    correct = 0
    for ex in examples:
        prompt = f"Solve the following math problem step by step.\n\n{ex['question']}\n\nAnswer:"
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        resp = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)
        gt = extract_gt(ex["answer"])
        pred = extract_pred(resp)
        if gt is not None and pred is not None and pred == gt:
            correct += 1
    return correct / len(examples) * 100.0


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    from mlx_lm import load
    t0 = time.time()
    log_mem("start")

    A, B = load_adapter_BA()
    examples = load_gsm8k(N_EVAL)
    model, tokenizer = load(MODEL_ID)

    head_dims, qdims = layer_dims(model, B)
    print(f"Per-layer head_dim verified against adapter B for {len(head_dims)} layers", flush=True)

    scores = compass_scores(B, head_dims)
    all_slots = sorted(scores.keys())
    compass_heads = select_top_heads(scores, TOP_K_HEADS)
    random_heads = select_random_heads(all_slots, TOP_K_HEADS, RANDOM_ARM_SEED)

    print(f"Compass top-{TOP_K_HEADS} heads (layer,head): {compass_heads}", flush=True)
    print(f"Random heads (seed {RANDOM_ARM_SEED}): {random_heads}", flush=True)

    results = {
        "experiment": "exp_spark_adapter_as_head_compass",
        "is_smoke": IS_SMOKE,
        "model_id": MODEL_ID,
        "adapter_file": str(ADAPTER_FILE),
        "n_eval": N_EVAL,
        "seed": SEED,
        "random_arm_seed": RANDOM_ARM_SEED,
        "top_k_heads": TOP_K_HEADS,
        "gamma_sweep": GAMMA_SWEEP,
        "adapter_scale_refuting_arm": ADAPTER_SCALE,
        "compass_selected_heads": [list(x) for x in compass_heads],
        "random_selected_heads": [list(x) for x in random_heads],
        "compass_scores_top": {f"L{e}H{h}": round(scores[(e, h)], 6) for (e, h) in compass_heads},
        "per_layer_head_dim": {str(k): v for k, v in sorted(head_dims.items())},
        "kill_2310": ("GSM8K EM (n>=60) for amplifying adapter-compass-selected base q_proj heads does "
                      "NOT exceed BOTH (a) no-intervention base AND (b) amplifying random-selected heads "
                      "(same count/factor) by >=+4pp each; OR best result requires applying the adapter "
                      "delta to logits at all"),
        "arms": {},
    }

    # ── Arm (a): base ────────────────────────────────────
    base_acc = eval_em(model, tokenizer, examples)
    results["arms"]["base"] = round(base_acc, 2)
    print(f"[base] EM = {base_acc:.2f}", flush=True)

    # ── Arm (b)+(c): compass / random sweep ──────────────
    compass_by_gamma, random_by_gamma = {}, {}
    for gamma in GAMMA_SWEEP:
        c_masks = build_layer_masks(compass_heads, gamma, head_dims, qdims)
        assert_no_delta_in_mask(c_masks, gamma)
        orig = install_amplify(model, c_masks)
        c_acc = eval_em(model, tokenizer, examples)
        restore(model, orig)
        compass_by_gamma[str(gamma)] = round(c_acc, 2)
        print(f"[compass gamma={gamma}] EM = {c_acc:.2f}", flush=True)

        r_masks = build_layer_masks(random_heads, gamma, head_dims, qdims)
        assert_no_delta_in_mask(r_masks, gamma)
        orig = install_amplify(model, r_masks)
        r_acc = eval_em(model, tokenizer, examples)
        restore(model, orig)
        random_by_gamma[str(gamma)] = round(r_acc, 2)
        print(f"[random  gamma={gamma}] EM = {r_acc:.2f}", flush=True)

    results["arms"]["compass_amplify_by_gamma"] = compass_by_gamma
    results["arms"]["random_amplify_by_gamma"] = random_by_gamma

    # ── Arm (ctx, refuting): delta applied ───────────────
    orig = install_delta(model, A, B, ADAPTER_SCALE, set(B.keys()))
    delta_acc = eval_em(model, tokenizer, examples)
    restore(model, orig)
    results["arms"]["delta_applied_refuting_ctx"] = round(delta_acc, 2)
    print(f"[delta-applied REFUTING ctx] EM = {delta_acc:.2f}", flush=True)

    # ── Verdict ──────────────────────────────────────────
    best_gamma = max(compass_by_gamma, key=lambda g: compass_by_gamma[g])
    c_star = compass_by_gamma[best_gamma]
    r_star = max(random_by_gamma.values())
    beats_base = (c_star - base_acc) >= 4.0
    beats_random = (c_star - r_star) >= 4.0
    delta_is_best = delta_acc > c_star and delta_acc > base_acc

    supported = beats_base and beats_random and not delta_is_best
    results["best_gamma"] = best_gamma
    results["compass_best_em"] = c_star
    results["random_best_em"] = r_star
    results["delta_diff_vs_base"] = round(c_star - base_acc, 2)
    results["compass_minus_base_pp"] = round(c_star - base_acc, 2)
    results["compass_minus_random_pp"] = round(c_star - r_star, 2)
    results["delta_is_best_arm"] = bool(delta_is_best)
    results["all_pass"] = bool(supported)
    results["verdict"] = "supported" if supported else "killed"
    results["kill_2310_result"] = "passed" if supported else "killed"
    results["total_time_s"] = round(time.time() - t0, 1)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 60, flush=True)
    print(f"VERDICT: {results['verdict']}", flush=True)
    print(f"base={base_acc:.2f}  compass*={c_star:.2f}(g={best_gamma})  random*={r_star:.2f}  delta={delta_acc:.2f}", flush=True)
    print(f"compass-base={results['compass_minus_base_pp']}pp  compass-random={results['compass_minus_random_pp']}pp", flush=True)
    cleanup(model, tokenizer)


if __name__ == "__main__":
    main()
