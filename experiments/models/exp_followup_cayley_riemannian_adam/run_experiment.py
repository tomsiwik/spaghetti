#!/usr/bin/env python3
"""exp_followup_cayley_riemannian_adam — Riemannian Adam on Stiefel for HRA.

TYPE: frontier-extension
MATH: experiments/models/exp_followup_cayley_riemannian_adam/MATH.md
PARENT: exp_p1_t1_householder_vs_lora (K1013 FAIL → this followup tests resurrection).

WHAT THIS TESTS
  Parent experiment killed HRA on equal-rank comparison vs LoRA: K1013 (HRA never
  converges in 300 steps) and K1012 (HRA MMLU -6pp). Parent impossibility analysis
  attributed K1013 to Euclidean Adam leaving the Stiefel manifold. This followup
  replaces Euclidean Adam with Riemannian Adam via QR retraction (a first-order
  Stiefel retraction; the math derives with Cayley but QR is algebraically
  equivalent to first order — see MATH.md Assumption 1).

  Training setup: Gemma 4 E4B-4bit, v_proj adapter r=16, 150 SFT steps on GSM8K,
  MMLU+GSM8K eval n=30 each. Three adapters trained under identical data/steps:
    - LoRA       (baseline — vanilla AdamW on lora_a/lora_b)
    - HRA_euc    (control — vanilla AdamW on V; expected to NOT converge)
    - HRA_riem   (hypothesis — Riemannian Adam with QR retraction on Stiefel)

KILL CRITERIA (target-gated per F#666):
  K1559 (proxy):      steps_to_loss<0.5 (HRA_riem) <= steps_to_loss<0.5 (LoRA)
  K_target:           MMLU_acc(HRA_riem) - MMLU_acc(LoRA) >= -3 pp
  K_stiefel (struct): max_layer ||V_final V_finalᵀ - I_r||_F <= 1e-4
  K_euc_control:      steps_to_loss<0.5 (HRA_euc) > 1.5 × LoRA_conv  (reproduces parent F#416)

  SUPPORTED = K1559 PASS AND K_target PASS.
  KILLED    = K1559 FAIL AND K_target FAIL.
  Other     = partial (see PAPER.md).
"""

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
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from mlx_lm import load as mlx_load
from mlx_lm import generate as mlx_generate
from mlx_lm.tuner.lora import LoRALinear

# Memory safety (phased execution, mlx-dev skill rules)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
EXPERIMENT_DIR = Path(__file__).parent

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

# Gemma 4 E4B native dims (verified via mlx_lm.load 0.31.2)
N_LAYERS = 42
D_MODEL = 2560   # hidden_size (v_proj input)

HRA_RANK = 16
LORA_RANK = 16

TRAIN_STEPS = 5 if IS_SMOKE else 150
LR = 5e-5
GRAD_CLIP = 1.0
MAX_SEQ_LEN = 64 if IS_SMOKE else 256
MAX_GEN_TOKENS = 32 if IS_SMOKE else 96
SEED = 42

N_EVAL_GSM8K = 3 if IS_SMOKE else 30
N_EVAL_MMLU = 3 if IS_SMOKE else 30
LOSS_CONV_THRESH = 0.5

# Riemannian Adam hyperparams (standard)
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# ---------------------------------------------------------------------------
# Stiefel geometry utilities (canonical metric, rows-orthonormal convention)
# ---------------------------------------------------------------------------

def project_tangent(V: mx.array, G: mx.array) -> mx.array:
    """Projection onto T_V St(r, d_in).

    V: (r, d_in) with VVᵀ = I_r.
    G: (r, d_in) Euclidean gradient.
    Returns Ξ = G - 0.5 * (G Vᵀ + V Gᵀ) V, which satisfies V Ξᵀ + Ξ Vᵀ = 0.
    """
    GVt = G @ V.T               # (r, r)
    sym = 0.5 * (GVt + GVt.T)   # symmetric part
    return G - sym @ V


def qr_retraction(V_plus_delta: mx.array) -> mx.array:
    """QR-based retraction onto St(r, d_in).

    Given V̂ = V + τΞ (off manifold after step), return (r, d_in) with rows
    orthonormal via QR of V̂ᵀ. Valid first-order retraction (Absil et al. 2008
    §4.1.1); equivalent to Cayley to O(τ²). mx.linalg.qr is CPU-only per 0.31.2.
    """
    # V̂ shape (r, n). QR on (n, r): Q is (n, r), columns orthonormal → V⁺ = Qᵀ.
    Q, _ = mx.linalg.qr(V_plus_delta.T, stream=mx.cpu)
    return Q.T


def grassmannian_init(r: int, n: int, seed: int) -> mx.array:
    """Partitioned-QR Grassmannian init (F#415, F#562 — orthonormal rows)."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, r)).astype(np.float32)  # (n, r)
    Q, _ = np.linalg.qr(M)                              # Q: (n, r), orthonormal cols
    return mx.array(Q.T)                                # (r, n), orthonormal rows

# ---------------------------------------------------------------------------
# HRALinear — Householder reflection adapter
# ---------------------------------------------------------------------------

class HRALinear(nn.Module):
    """HRA adapter: y = linear(H^(r) x), H^(r) = H_r ∘ ... ∘ H_1.

    V ∈ R^{r × d_in}, trainable. When rows are orthonormal (Stiefel), H^(r) is
    a proper orthogonal transform. This module does NOT itself enforce
    orthonormality — the caller (Riemannian optimizer or GS-retraction) must.
    """

    def __init__(self, linear: nn.Module, r: int, d_in: int, seed: int):
        super().__init__()
        self.r = r
        self.d_in = d_in
        self.linear = linear
        self.V = grassmannian_init(r, d_in, seed)

    def __call__(self, x: mx.array) -> mx.array:
        # Vectorized: for each of r reflections, x ← x − 2 (x·v/v·v) v.
        for i in range(self.r):
            v = self.V[i]                                       # (d_in,)
            vv = mx.sum(v * v) + 1e-8
            xv = mx.sum(x * v, axis=-1, keepdims=True)          # (..., 1)
            x = x - (2.0 / vv) * xv * v                         # (..., d_in)
        return self.linear(x)

# ---------------------------------------------------------------------------
# Data loaders (GSM8K + MMLU)
# ---------------------------------------------------------------------------

def load_gsm8k(split: str, n: int, seed: int):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    out = []
    for i in indices:
        item = ds[i]
        m = re.search(r"####\s*([0-9,\-\.]+)", item["answer"])
        if m:
            out.append({"question": item["question"],
                        "answer": m.group(1).replace(",", "")})
        if len(out) >= n:
            break
    return out


def load_mmlu(n: int, seed: int):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), min(n * 2, len(ds)))
    out = []
    for i in idx:
        it = ds[i]
        if len(it["choices"]) == 4:
            out.append({"question": it["question"], "choices": it["choices"],
                        "answer": it["answer"]})
        if len(out) >= n:
            break
    return out

# ---------------------------------------------------------------------------
# Prompt / loss helpers
# ---------------------------------------------------------------------------

def fmt_gsm8k(q, a=None):
    return f"Q: {q}\nA: {a}" if a is not None else f"Q: {q}\nA:"


def fmt_mmlu(q, choices):
    labels = "ABCD"
    opts = "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))
    return f"{q}\n{opts}\nAnswer:"


def loss_fn(model, tokens):
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    logits = model(x)
    return nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="mean"
    )

# ---------------------------------------------------------------------------
# Adapter injection / weight-state helpers
# ---------------------------------------------------------------------------

def inject_hra(model, r=HRA_RANK, d_in=D_MODEL, seed=SEED):
    count = 0
    for i, layer in enumerate(model.layers):
        original = layer.self_attn.v_proj
        layer.self_attn.v_proj = HRALinear(original, r=r, d_in=d_in,
                                           seed=seed + i)   # per-layer seed
        count += 1
    return count


def inject_lora(model, r=LORA_RANK):
    count = 0
    for layer in model.layers:
        original = layer.self_attn.v_proj
        layer.self_attn.v_proj = LoRALinear.from_base(
            original, r=r, scale=float(r)
        )
        count += 1
    return count


def count_trainable(model):
    return sum(p.size for _, p in tree_flatten(model.trainable_parameters()))

# ---------------------------------------------------------------------------
# Training — three regimes
# ---------------------------------------------------------------------------

def train_lora(model, tok, train_data):
    """Baseline: LoRA + AdamW."""
    print("\n=== Training LoRA r=16 (Euclidean AdamW) ===")
    optimizer = optim.AdamW(learning_rate=LR, weight_decay=0.01)
    return _train_loop(model, tok, optimizer, train_data, tag="lora")


def train_hra_euc(model, tok, train_data):
    """Control: HRA + vanilla Euclidean AdamW (parent regime)."""
    print("\n=== Training HRA r=16 (Euclidean AdamW — parent regime control) ===")
    optimizer = optim.AdamW(learning_rate=LR, weight_decay=0.0)   # no WD for manifold params
    return _train_loop(model, tok, optimizer, train_data, tag="hra_euc")


def _train_loop(model, tok, optimizer, train_data, tag: str):
    """Euclidean AdamW training loop (shared by lora + hra_euc)."""
    prompts = [fmt_gsm8k(ex["question"], ex["answer"]) for ex in train_data]
    value_and_grad = nn.value_and_grad(model, loss_fn)

    step_times, loss_curve = [], []
    conv_step = TRAIN_STEPS + 1
    rng = random.Random(SEED)

    for step in range(TRAIN_STEPS):
        prompt = rng.choice(prompts)
        toks = tok.encode(prompt)[:MAX_SEQ_LEN]
        if len(toks) < 4:
            continue
        batch = mx.array([toks], dtype=mx.uint32)

        t0 = time.perf_counter()
        loss, grads = value_and_grad(model, batch)

        all_grads = [g for _, g in tree_flatten(grads) if g is not None]
        grad_norm = mx.sqrt(sum(mx.sum(g * g) for g in all_grads))
        if mx.isfinite(grad_norm):
            scale = mx.minimum(mx.array(1.0), GRAD_CLIP / (grad_norm + 1e-8))
            grads = tree_map(lambda g: g * scale if g is not None else None, grads)

        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        t1 = time.perf_counter()

        lv = loss.item()
        step_times.append(t1 - t0)
        loss_curve.append(lv)
        if lv < LOSS_CONV_THRESH and conv_step > TRAIN_STEPS:
            conv_step = step + 1
        if step % 25 == 0 or step < 3:
            print(f"  [{tag}] step {step:3d}/{TRAIN_STEPS}: loss={lv:.4f} "
                  f"time={t1-t0:.3f}s")

    avg_step = float(np.mean(step_times[1:]) if len(step_times) > 1 else np.mean(step_times or [0.0]))
    print(f"  [{tag}] avg step={avg_step:.3f}s, conv_step={conv_step}")
    return {"loss_curve": loss_curve, "avg_step_time": avg_step,
            "conv_step": conv_step, "n_trainable": int(count_trainable(model))}


def train_hra_riem(model, tok, train_data):
    """Hypothesis: HRA + Riemannian Adam (QR retraction on Stiefel).

    Implements Bécigneul–Ganea (2019) Riemannian Adam with per-layer per-V state:
        m (first moment)  — tangent-space momentum
        v (second moment) — element-wise (standard Adam-style)
    Retraction: QR (first-order equivalent to Cayley; simpler implementation).

    Base-model parameters are frozen; only V matrices update per-layer.
    """
    print("\n=== Training HRA r=16 (Riemannian Adam + QR retraction) ===")

    # Collect V handles (one per layer) and init moments.
    V_state = []   # list of dicts {"V": mx.array, "m": mx.array, "v": mx.array}
    hra_layers = []
    for layer in model.layers:
        hra = layer.self_attn.v_proj
        assert isinstance(hra, HRALinear)
        zeros = mx.zeros_like(hra.V)
        V_state.append({"V": hra.V, "m": zeros, "v": zeros})
        hra_layers.append(hra)

    n_trainable = sum(s["V"].size for s in V_state)
    print(f"  HRA adapter trainable V-params: {n_trainable:,} "
          f"({len(V_state)} layers × {V_state[0]['V'].shape})")

    prompts = [fmt_gsm8k(ex["question"], ex["answer"]) for ex in train_data]
    value_and_grad = nn.value_and_grad(model, loss_fn)

    step_times, loss_curve = [], []
    conv_step = TRAIN_STEPS + 1
    rng = random.Random(SEED)

    for step in range(TRAIN_STEPS):
        prompt = rng.choice(prompts)
        toks = tok.encode(prompt)[:MAX_SEQ_LEN]
        if len(toks) < 4:
            continue
        batch = mx.array([toks], dtype=mx.uint32)

        t0 = time.perf_counter()
        loss, grads = value_and_grad(model, batch)

        # Build a list of Euclidean gradients w.r.t. each layer's V.
        # mlx_lm Gemma 4 wraps layers under `language_model.model.layers.*` —
        # flatten grads and match by full key suffix ".self_attn.v_proj.V".
        flat = dict(tree_flatten(grads))
        euc_grads = [None] * len(hra_layers)
        for key, g in flat.items():
            if not key.endswith(".self_attn.v_proj.V"):
                continue
            # extract integer layer index from "...layers.<i>.self_attn..."
            m_idx = re.search(r"layers\.(\d+)\.self_attn", key)
            if m_idx is None:
                continue
            i = int(m_idx.group(1))
            if 0 <= i < len(hra_layers):
                euc_grads[i] = g
        if any(g is None for g in euc_grads):
            missing = [i for i, g in enumerate(euc_grads) if g is None]
            raise RuntimeError(f"HRA_riem: grads missing for layers {missing[:5]}... "
                               f"(got {len(flat)} leaf grads total)")

        # Global gradient clipping on the concatenated adapter grads.
        total_sq = sum(mx.sum(g * g) for g in euc_grads)
        gnorm = mx.sqrt(total_sq)
        if mx.isfinite(gnorm):
            gscale = mx.minimum(mx.array(1.0), GRAD_CLIP / (gnorm + 1e-8))
            euc_grads = [g * gscale for g in euc_grads]

        t = step + 1
        bc1 = 1.0 - ADAM_BETA1 ** t
        bc2 = 1.0 - ADAM_BETA2 ** t

        # Per-layer Riemannian Adam update.
        for i, hra in enumerate(hra_layers):
            st = V_state[i]
            V, G_euc = st["V"], euc_grads[i]

            # 1. Project Euclidean grad to tangent space at V.
            Xi = project_tangent(V, G_euc)

            # 2. Adam moments (in tangent).
            m = ADAM_BETA1 * st["m"] + (1.0 - ADAM_BETA1) * Xi
            v = ADAM_BETA2 * st["v"] + (1.0 - ADAM_BETA2) * (Xi * Xi)
            m_hat = m / bc1
            v_hat = v / bc2
            d = m_hat / (mx.sqrt(v_hat) + ADAM_EPS)

            # 3. Re-project search direction onto tangent (simple transport).
            d_t = project_tangent(V, d)

            # 4. QR retraction on Stiefel.
            V_new = qr_retraction(V - LR * d_t)

            # 5. Transport momentum to new tangent (re-projection).
            m_transported = project_tangent(V_new, m)

            st["V"] = V_new
            st["m"] = m_transported
            st["v"] = v
            hra.V = V_new   # write back to the module

        mx.eval(loss, *[st["V"] for st in V_state],
                *[st["m"] for st in V_state], *[st["v"] for st in V_state])
        t1 = time.perf_counter()

        lv = loss.item()
        step_times.append(t1 - t0)
        loss_curve.append(lv)
        if lv < LOSS_CONV_THRESH and conv_step > TRAIN_STEPS:
            conv_step = step + 1
        if step % 25 == 0 or step < 3:
            print(f"  [hra_riem] step {step:3d}/{TRAIN_STEPS}: loss={lv:.4f} "
                  f"time={t1-t0:.3f}s")

    avg_step = float(np.mean(step_times[1:]) if len(step_times) > 1 else np.mean(step_times or [0.0]))

    # K_stiefel: measure final Stiefel deviation across all layers.
    stiefel_errs = []
    eye_r = mx.eye(HRA_RANK)
    for st in V_state:
        V = st["V"]
        err = mx.sqrt(mx.sum((V @ V.T - eye_r) ** 2)).item()   # Frobenius
        stiefel_errs.append(err)
    max_stiefel_err = float(max(stiefel_errs))
    print(f"  [hra_riem] avg step={avg_step:.3f}s, conv_step={conv_step}, "
          f"max |VVᵀ-I|_F = {max_stiefel_err:.2e}")

    return {"loss_curve": loss_curve, "avg_step_time": avg_step,
            "conv_step": conv_step, "n_trainable": int(n_trainable),
            "max_stiefel_err": max_stiefel_err}

# ---------------------------------------------------------------------------
# Phased execution — one function per phase, cleanup between (mlx-dev pattern)
# ---------------------------------------------------------------------------

def save_hra_weights(model, path):
    d = {}
    for i, layer in enumerate(model.layers):
        hra = layer.self_attn.v_proj
        if isinstance(hra, HRALinear):
            mx.eval(hra.V)
            d[f"V_{i}"] = np.array(hra.V)
    np.savez(path, **d)


def save_lora_weights(model, path):
    d = {}
    for i, layer in enumerate(model.layers):
        lora = layer.self_attn.v_proj
        if isinstance(lora, LoRALinear):
            mx.eval(lora.lora_a, lora.lora_b)
            d[f"lora_a_{i}"] = np.array(lora.lora_a)
            d[f"lora_b_{i}"] = np.array(lora.lora_b)
    np.savez(path, **d)


def load_hra_weights(model, path):
    w = np.load(path)
    for i, layer in enumerate(model.layers):
        key = f"V_{i}"
        if key in w.files and isinstance(layer.self_attn.v_proj, HRALinear):
            layer.self_attn.v_proj.V = mx.array(w[key])


def load_lora_weights(model, path):
    w = np.load(path)
    for i, layer in enumerate(model.layers):
        a, b = f"lora_a_{i}", f"lora_b_{i}"
        if a in w.files and isinstance(layer.self_attn.v_proj, LoRALinear):
            layer.self_attn.v_proj.lora_a = mx.array(w[a])
            layer.self_attn.v_proj.lora_b = mx.array(w[b])


def phase_train(adapter: str, train_data):
    """Run one training phase end-to-end; return (stats, weight_path)."""
    model, tok = mlx_load(MODEL_ID)
    model.freeze()
    if adapter == "lora":
        inject_lora(model)
    else:
        inject_hra(model)

    if adapter == "lora":
        stats = train_lora(model, tok, train_data)
        wpath = EXPERIMENT_DIR / "lora_weights.npz"
        save_lora_weights(model, wpath)
    elif adapter == "hra_euc":
        stats = train_hra_euc(model, tok, train_data)
        wpath = EXPERIMENT_DIR / "hra_euc_weights.npz"
        save_hra_weights(model, wpath)
    elif adapter == "hra_riem":
        stats = train_hra_riem(model, tok, train_data)
        wpath = EXPERIMENT_DIR / "hra_riem_weights.npz"
        save_hra_weights(model, wpath)
    else:
        raise ValueError(adapter)

    del model, tok
    gc.collect()
    mx.clear_cache()
    return stats, str(wpath)


def phase_eval(adapter: str, weight_path, gsm8k_eval, mmlu_eval):
    """Eval one adapter (or base if adapter='base'); return dict of accs."""
    model, tok = mlx_load(MODEL_ID)
    model.freeze()
    if adapter == "lora":
        inject_lora(model)
        load_lora_weights(model, weight_path)
    elif adapter in ("hra_euc", "hra_riem"):
        inject_hra(model)
        load_hra_weights(model, weight_path)
    # adapter == "base": nothing injected
    mx.eval(model.parameters())

    # GSM8K
    correct = 0
    for ex in gsm8k_eval:
        out = mlx_generate(model, tok, prompt=fmt_gsm8k(ex["question"]),
                           max_tokens=MAX_GEN_TOKENS, verbose=False)
        nums = re.findall(r"-?[\d,]+\.?\d*", out)
        if nums:
            pred = nums[-1].replace(",", "")
            if pred == ex["answer"] or pred.rstrip("0").rstrip(".") == ex["answer"]:
                correct += 1
    gsm8k_acc = correct / len(gsm8k_eval) if gsm8k_eval else 0.0

    # MMLU
    correct = 0
    for ex in mmlu_eval:
        out = mlx_generate(model, tok, prompt=fmt_mmlu(ex["question"], ex["choices"]),
                           max_tokens=16, verbose=False)
        m = re.search(r"\b([ABCD])\b", out.strip().upper())
        pred = m.group(1) if m else "X"
        truth = "ABCD"[ex["answer"]]
        if pred == truth:
            correct += 1
    mmlu_acc = correct / len(mmlu_eval) if mmlu_eval else 0.0

    print(f"  [{adapter}] GSM8K={gsm8k_acc:.3f} MMLU={mmlu_acc:.3f}")

    del model, tok
    gc.collect()
    mx.clear_cache()
    return {"gsm8k": gsm8k_acc, "mmlu": mmlu_acc}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    mx.random.seed(SEED)

    print(f"=== Cayley/Riemannian Adam on Stiefel for HRA "
          f"({'SMOKE' if IS_SMOKE else 'FULL'}) ===")
    print(f"Model: {MODEL_ID}  |  Target: v_proj (d_in={D_MODEL}) "
          f"× {N_LAYERS} layers")
    print(f"HRA r={HRA_RANK}  LoRA r={LORA_RANK}  Steps={TRAIN_STEPS}  "
          f"GSM8K eval n={N_EVAL_GSM8K}  MMLU eval n={N_EVAL_MMLU}")

    train_data = load_gsm8k("train", 20 if IS_SMOKE else 500, SEED)
    gsm8k_eval = load_gsm8k("test", N_EVAL_GSM8K, SEED + 1)
    mmlu_eval = load_mmlu(N_EVAL_MMLU, SEED + 2)
    print(f"Data: train={len(train_data)}  gsm8k_eval={len(gsm8k_eval)}  "
          f"mmlu_eval={len(mmlu_eval)}")

    results = {
        "is_smoke": IS_SMOKE, "model_id": MODEL_ID,
        "train_steps": TRAIN_STEPS, "hra_rank": HRA_RANK, "lora_rank": LORA_RANK,
        "d_model": D_MODEL, "n_layers": N_LAYERS,
        "hra_params_per_layer": HRA_RANK * D_MODEL,
        "lora_params_per_layer": LORA_RANK * (D_MODEL + 512),   # v_proj out_dim=512
    }

    # Base eval (Phase 0).
    print("\n--- Phase 0: base eval ---")
    base = phase_eval("base", None, gsm8k_eval, mmlu_eval)
    results["base"] = base

    # Phase 1: LoRA train + eval.
    print("\n--- Phase 1: LoRA ---")
    lora_stats, lora_w = phase_train("lora", train_data)
    lora_eval = phase_eval("lora", lora_w, gsm8k_eval, mmlu_eval)
    results["lora_train"] = lora_stats
    results["lora_eval"] = lora_eval

    # Phase 2: HRA_euc control train + eval (reproduces parent F#416 regime).
    print("\n--- Phase 2: HRA_euc (parent regime control) ---")
    hra_euc_stats, hra_euc_w = phase_train("hra_euc", train_data)
    hra_euc_eval = phase_eval("hra_euc", hra_euc_w, gsm8k_eval, mmlu_eval)
    results["hra_euc_train"] = hra_euc_stats
    results["hra_euc_eval"] = hra_euc_eval

    # Phase 3: HRA_riem train + eval (hypothesis).
    print("\n--- Phase 3: HRA_riem (Riemannian Adam + QR retraction) ---")
    hra_riem_stats, hra_riem_w = phase_train("hra_riem", train_data)
    hra_riem_eval = phase_eval("hra_riem", hra_riem_w, gsm8k_eval, mmlu_eval)
    results["hra_riem_train"] = hra_riem_stats
    results["hra_riem_eval"] = hra_riem_eval

    # --- KC evaluation ---
    lora_conv = lora_stats["conv_step"]
    hra_euc_conv = hra_euc_stats["conv_step"]
    hra_riem_conv = hra_riem_stats["conv_step"]

    lora_converged = lora_conv <= TRAIN_STEPS
    hra_riem_converged = hra_riem_conv <= TRAIN_STEPS
    hra_euc_converged = hra_euc_conv <= TRAIN_STEPS

    # K1559 proxy: HRA_riem converges in ≤ LoRA steps (explicit branches; propagates
    # the parent F#416 code-bug fix — sentinel conv_step cannot inflate to false PASS).
    if not lora_converged and not hra_riem_converged:
        k1559 = False           # neither converged — can't measure
    elif hra_riem_converged and not lora_converged:
        k1559 = True            # riem converged, lora DNF — strictly better
    elif not hra_riem_converged and lora_converged:
        k1559 = False           # riem DNF, lora converged — fail
    else:
        k1559 = hra_riem_conv <= lora_conv

    # K_target: MMLU(HRA_riem) within -3pp of LoRA.
    mmlu_delta_pp = (hra_riem_eval["mmlu"] - lora_eval["mmlu"]) * 100
    k_target = mmlu_delta_pp >= -3.0

    # K_stiefel structural: Frobenius ||VVᵀ − I_r||_F ≤ 1e-4.
    stiefel_err = hra_riem_stats.get("max_stiefel_err", float("inf"))
    k_stiefel = stiefel_err <= 1e-4

    # K_euc_control: parent F#416 should reproduce — HRA_euc should NOT converge
    # within 1.5× LoRA's step count (control validity).
    if not lora_converged:
        k_euc_control = not hra_euc_converged   # both DNF = consistent parent behavior
    else:
        k_euc_control = (not hra_euc_converged) or (hra_euc_conv > 1.5 * lora_conv)

    # Target-gated verdict (F#666).
    proxy_pass = k1559
    target_pass = k_target
    if proxy_pass and target_pass:
        verdict = "SUPPORTED"
    elif (not proxy_pass) and (not target_pass):
        verdict = "KILLED"
    else:
        verdict = "PARTIAL"

    results.update({
        "k1559": {"pass": bool(k1559),
                  "hra_riem_conv_step": hra_riem_conv,
                  "lora_conv_step": lora_conv,
                  "hra_riem_converged": bool(hra_riem_converged),
                  "lora_converged": bool(lora_converged),
                  "train_steps": TRAIN_STEPS,
                  "conv_threshold": LOSS_CONV_THRESH},
        "k_target": {"pass": bool(k_target),
                     "hra_riem_mmlu": hra_riem_eval["mmlu"],
                     "lora_mmlu": lora_eval["mmlu"],
                     "delta_pp": round(mmlu_delta_pp, 2),
                     "threshold_pp": -3.0},
        "k_stiefel": {"pass": bool(k_stiefel),
                      "max_err_frobenius": stiefel_err,
                      "threshold": 1e-4},
        "k_euc_control": {"pass": bool(k_euc_control),
                          "hra_euc_conv_step": hra_euc_conv,
                          "lora_conv_step": lora_conv,
                          "hra_euc_converged": bool(hra_euc_converged)},
        "all_pass": bool(proxy_pass and target_pass),
        "verdict": verdict,
        "ran": True,
    })

    out = EXPERIMENT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults written: {out}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Base:       GSM8K={base['gsm8k']:.3f}  MMLU={base['mmlu']:.3f}")
    print(f"LoRA:       GSM8K={lora_eval['gsm8k']:.3f}  MMLU={lora_eval['mmlu']:.3f}  "
          f"conv={lora_conv}/{TRAIN_STEPS}  t={lora_stats['avg_step_time']:.3f}s")
    print(f"HRA_euc:    GSM8K={hra_euc_eval['gsm8k']:.3f}  MMLU={hra_euc_eval['mmlu']:.3f}  "
          f"conv={hra_euc_conv}/{TRAIN_STEPS}  t={hra_euc_stats['avg_step_time']:.3f}s")
    print(f"HRA_riem:   GSM8K={hra_riem_eval['gsm8k']:.3f}  MMLU={hra_riem_eval['mmlu']:.3f}  "
          f"conv={hra_riem_conv}/{TRAIN_STEPS}  t={hra_riem_stats['avg_step_time']:.3f}s")
    print()
    print(f"K1559   (proxy:conv)   : {'PASS' if k1559 else 'FAIL'}  "
          f"hra_riem={hra_riem_conv} vs lora={lora_conv}")
    print(f"K_target (MMLU delta)  : {'PASS' if k_target else 'FAIL'}  "
          f"Δ={mmlu_delta_pp:+.1f}pp (thresh ≥ -3pp)")
    print(f"K_stiefel (struct)     : {'PASS' if k_stiefel else 'FAIL'}  "
          f"max ||VVᵀ-I|_F={stiefel_err:.2e}")
    print(f"K_euc_ctrl (parent F416): {'PASS' if k_euc_control else 'FAIL'}  "
          f"hra_euc={hra_euc_conv} vs lora={lora_conv}")
    print(f"\nVerdict: {verdict}")
    return results


if __name__ == "__main__":
    main()
