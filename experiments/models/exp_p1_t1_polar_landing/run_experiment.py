#!/usr/bin/env python3
"""T1.5: PoLAR Landing Field — Stiefel retraction prevents rank collapse.

TYPE: guided-exploration
MATH: experiments/models/exp_p1_t1_polar_landing/MATH.md

WHAT THIS TESTS:
  PoLAR adapter: ΔW = V @ U with U retracted to Stiefel every RETRACT_EVERY steps.
  Retraction = polar projection: U ← UV^T from SVD(U) — all rows become orthonormal.
  This is the discrete version of the gradient flow toward Stiefel (Theorem 1: landing field).

  Theorem 2: near-orthogonal U prevents rank collapse (sr(ΔW) >= sr(V) / (1+ε)^2).
  With periodic retraction: ε ≈ 0 → sr(ΔW) ≈ sr(V) >> sr(LoRA_trained).

  K1021: ||U U^T - I||_F_max < 0.01 after retraction (trivially ≈ float32_floor)
  K1022: sr(ΔW = V@U) >= 5 at r=32 (vs LoRA rank collapse to sr~1-3)
  K1023: PoLAR quality >= LoRA quality on GSM8K at matched rank (r=32)

NOTE: Gemma4 not loadable by mlx_lm 0.29.1. Using Qwen3-4B-4bit as proxy.
SMOKE_TEST=1: 5 steps, n=5 eval, completes in <5 min.
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

# Memory safety — MANDATORY per CODING_GUIDELINES
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
EXPERIMENT_DIR = Path(__file__).parent

MODEL_ID = "mlx-community/Qwen3-4B-4bit"

RANK = 32          # r=32 as required by K1022
SCALE = 1.0        # adapter scale
D_IN = 2560        # Qwen3-4B hidden_size (q_proj input)
D_OUT = 4096       # Qwen3-4B q_proj output (32 heads × 128 head_dim)

TRAIN_STEPS = 5 if IS_SMOKE else 200
RETRACT_EVERY = 2 if IS_SMOKE else 10  # Stiefel retraction interval
LR = 5e-5
GRAD_CLIP = 1.0
MAX_SEQ_LEN = 64 if IS_SMOKE else 256
MAX_GEN_TOKENS = 32 if IS_SMOKE else 128
SEED = 42

N_EVAL = 5 if IS_SMOKE else 30  # GSM8K eval size


# ---------------------------------------------------------------------------
# PoLAR adapter (Polar-Decomposed Low-Rank via periodic retraction)
# ---------------------------------------------------------------------------

class PoLARLinear(nn.Module):
    """PoLAR adapter: ΔW = V @ U with U constrained to near-Stiefel.

    U ∈ R^{r × d_in}: initialized with QR (orthonormal rows).
    V ∈ R^{d_out × r}: initialized to zero (residual property at t=0).

    After every RETRACT_EVERY gradient steps, U is retracted to Stiefel
    via polar projection: SVD(U) → U ← UV^T (all rows orthonormal).

    The retraction is the discrete step of the landing field gradient flow.
    Theorem 1 guarantees this converges; retraction makes it exact.

    Forward: output = linear(x) + scale * (x @ U^T) @ V^T
    Param count = LoRA r=32: r*d_in + d_out*r.
    """

    def __init__(self, linear: nn.Module, r: int = RANK, d_in: int = D_IN,
                 d_out: int = D_OUT, scale: float = SCALE):
        super().__init__()
        self.r = r
        self.linear = linear  # frozen base
        self.scale = scale

        # Initialize U with orthonormal rows via QR
        rng = np.random.default_rng(SEED)
        U_rand = rng.standard_normal((d_in, r)).astype(np.float32)
        Q, _ = np.linalg.qr(U_rand)  # Q: (d_in, r) orthonormal columns
        self.U = mx.array(Q.T)  # (r, d_in): orthonormal rows
        self.V = mx.zeros((d_out, r))

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.linear(x)
        delta = (x @ self.U.T) @ self.V.T  # (batch, d_out)
        return base_out + self.scale * delta

    def retract_to_stiefel(self):
        """Project U rows to Stiefel via polar decomposition: U ← UV^T from SVD(U).

        Cheap: r=32 << d_in=2560, so SVD of (32 × 2560) is fast.
        After retraction: ||U U^T - I||_F ≤ float32_floor ≈ 1e-6.
        """
        U_np = np.array(self.U.tolist(), dtype=np.float64)  # float64 for stability
        # Guard: skip retraction if U contains NaN/inf (prevents SVD from producing NaN W/Vh)
        if not np.all(np.isfinite(U_np)):
            print("WARNING: U contains NaN/inf before retraction — skipping retraction step")
            return
        W, _, Vh = np.linalg.svd(U_np, full_matrices=False)
        # W: (r, r), Vh: (r, d_in) — orthonormal rows
        U_retracted = W @ Vh  # (r, d_in) with orthonormal rows
        if not np.all(np.isfinite(U_retracted)):
            print("WARNING: SVD retraction produced NaN/inf — skipping update")
            return
        self.U = mx.array(U_retracted.astype(np.float32))

    def measure_stiefel_distance(self) -> float:
        """||U U^T - I||_F after retraction (K1021 measurement)."""
        U_np = np.array(self.U.tolist(), dtype=np.float64)
        G = U_np @ U_np.T
        I = np.eye(self.r)
        E = G - I
        return float(np.sqrt(np.sum(E ** 2)))


# ---------------------------------------------------------------------------
# LoRA adapter (baseline for K1023 comparison)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Standard LoRA: ΔW = B @ A. A~Normal(0,σ), B=0. No orthogonality."""

    def __init__(self, linear: nn.Module, r: int = RANK, d_in: int = D_IN,
                 d_out: int = D_OUT, scale: float = SCALE):
        super().__init__()
        self.r = r
        self.linear = linear
        self.scale = scale
        rng = np.random.default_rng(SEED)
        A_init = rng.standard_normal((r, d_in)).astype(np.float32) * (1.0 / math.sqrt(r))
        self.A = mx.array(A_init)
        self.B = mx.zeros((d_out, r))

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.linear(x)
        delta = (x @ self.A.T) @ self.B.T
        return base_out + self.scale * delta


# ---------------------------------------------------------------------------
# Stable rank (K1022)
# ---------------------------------------------------------------------------

def stable_rank_of_product(V_np: np.ndarray, U_np: np.ndarray) -> float:
    """Compute sr(V @ U) exactly via thin SVD of V then (r × d_in) matrix M.

    V: (d_out, r), U: (r, d_in). Method:
    1. Thin SVD of V: V = U_V @ S_V @ Vh_V
    2. M = diag(S_V) @ (Vh_V @ U)   — (r × d_in)
    3. Thin SVD of M → singular values S_M
    4. sr = sum(S_M_i^2) / max(S_M_i)^2

    Complexity: O(d_out * r^2 + r^2 * d_in) — cheap for r=32.
    """
    V_norm_sq = float(np.sum(V_np ** 2))
    if V_norm_sq < 1e-12:
        return 0.0
    V64 = V_np.astype(np.float64)
    U64 = U_np.astype(np.float64)
    _, S_V, Vh_V = np.linalg.svd(V64, full_matrices=False)  # S_V: (r,)
    M = np.diag(S_V) @ (Vh_V @ U64)  # (r, d_in)
    _, S_M, _ = np.linalg.svd(M, full_matrices=False)  # S_M: (r,)
    frob_sq = float(np.sum(S_M ** 2))
    spectral_sq = float(S_M[0] ** 2) if len(S_M) > 0 else 0.0
    return frob_sq / spectral_sq if spectral_sq > 1e-12 else 0.0


def measure_stable_ranks(model, adapter_cls) -> dict:
    srs = []
    for layer in model.layers:
        adapter = layer.self_attn.q_proj
        if not isinstance(adapter, adapter_cls):
            continue
        if adapter_cls is PoLARLinear:
            V_np = np.array(adapter.V.tolist(), dtype=np.float32)
            U_np = np.array(adapter.U.tolist(), dtype=np.float32)
        else:
            V_np = np.array(adapter.B.tolist(), dtype=np.float32)
            U_np = np.array(adapter.A.tolist(), dtype=np.float32)
        srs.append(stable_rank_of_product(V_np, U_np))
    if not srs:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "mean": float(np.mean(srs)),
        "min": float(np.min(srs)),
        "max": float(np.max(srs)),
        "median": float(np.median(srs)),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gsm8k(split="train", n=500, seed=SEED):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    random.seed(seed)
    indices = list(range(min(n * 2, len(ds))))
    random.shuffle(indices)
    examples = []
    for i in indices:
        item = ds[i]
        match = re.search(r"####\s*([0-9,\-\.]+)", item["answer"])
        if match:
            final = match.group(1).replace(",", "")
            examples.append({"question": item["question"], "answer": final})
        if len(examples) >= n:
            break
    return examples[:n]


def make_gsm8k_prompt(q, answer=None):
    if answer is not None:
        return f"Q: {q}\nA: {answer}"
    return f"Q: {q}\nA:"


def extract_number(text: str):
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def tokenize_for_training(tokenizer, examples, max_len):
    result = []
    for ex in examples:
        prompt = make_gsm8k_prompt(ex["question"], ex["answer"])
        ids = tokenizer.encode(prompt)
        ids = ids[:max_len]
        result.append(ids)
    return result


# ---------------------------------------------------------------------------
# Adapter injection
# ---------------------------------------------------------------------------

def freeze_model(model):
    model.freeze()


def inject_adapters(model, adapter_cls):
    count = 0
    for layer in model.layers:
        original = layer.self_attn.q_proj
        layer.self_attn.q_proj = adapter_cls(
            linear=original, r=RANK, d_in=D_IN, d_out=D_OUT, scale=SCALE
        )
        count += 1
    return count


def retract_all_adapters(model):
    """Retract U to Stiefel in all PoLAR layers. Cheap: r=32 SVD per layer."""
    for layer in model.layers:
        adapter = layer.self_attn.q_proj
        if isinstance(adapter, PoLARLinear):
            adapter.retract_to_stiefel()
    mx.eval(model.parameters())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_ce_loss(model, input_ids: list) -> mx.array:
    losses = []
    for ids in input_ids:
        if len(ids) < 2:
            continue
        x = mx.array(ids[:-1], dtype=mx.uint32)[None, :]
        targets = mx.array(ids[1:], dtype=mx.uint32)
        logits = model(x)[0]  # (T-1, vocab)
        loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
        losses.append(loss)
    return sum(losses) / len(losses) if losses else mx.array(0.0)


def train_adapter(model, tokenizer, train_data, adapter_cls, n_steps, retract_every=0):
    """Train adapter. For PoLAR: retract_every > 0 applies Stiefel retraction."""
    optimizer = optim.AdamW(learning_rate=LR)
    value_and_grad = nn.value_and_grad(model, compute_ce_loss)

    random.seed(SEED)
    step_times = []
    losses = []

    for step in range(n_steps):
        batch_ids = random.sample(train_data, min(2, len(train_data)))

        t0 = time.time()
        loss, grads = value_and_grad(model, batch_ids)
        mx.eval(loss, grads)

        # Gradient clipping
        grad_list = [(k, v) for k, v in tree_flatten(grads)]
        gnorm = math.sqrt(sum(float(mx.sum(g * g).item()) for _, g in grad_list))
        if gnorm > GRAD_CLIP:
            scale = GRAD_CLIP / (gnorm + 1e-6)
            grads = tree_map(lambda g: g * scale, grads)

        optimizer.update(model, grads)
        mx.eval(model.parameters())

        # Stiefel retraction for PoLAR
        if retract_every > 0 and (step + 1) % retract_every == 0:
            retract_all_adapters(model)

        step_time = time.time() - t0
        step_times.append(step_time)
        losses.append(float(loss.item()))

        if step % 20 == 0 or step == n_steps - 1:
            print(f"  step {step:3d}: loss={losses[-1]:.4f}  time={step_time:.2f}s")

    # Final retraction for PoLAR (ensures K1021 measurement is post-retraction)
    if retract_every > 0:
        retract_all_adapters(model)

    return {
        "losses": losses,
        "mean_step_time": float(np.mean(step_times)),
        "final_loss": losses[-1] if losses else float("nan"),
    }


# ---------------------------------------------------------------------------
# Quality evaluation (K1023)
# ---------------------------------------------------------------------------

def eval_gsm8k(model, tokenizer, examples):
    correct = 0
    for ex in examples:
        prompt = make_gsm8k_prompt(ex["question"])
        try:
            response = mlx_generate(
                model, tokenizer, prompt=prompt, max_tokens=MAX_GEN_TOKENS, verbose=False
            )
        except Exception:
            response = ""
        pred = extract_number(response)
        correct += int(pred == ex["answer"])
    return correct / len(examples) if examples else 0.0


# ---------------------------------------------------------------------------
# Near-Stiefel measurement (K1021)
# ---------------------------------------------------------------------------

def measure_stiefel_stats(model) -> dict:
    dists = []
    for layer in model.layers:
        adapter = layer.self_attn.q_proj
        if isinstance(adapter, PoLARLinear):
            dists.append(adapter.measure_stiefel_distance())
    if not dists:
        return {"mean": 0.0, "max": 0.0, "min": 0.0}
    return {
        "mean": float(np.mean(dists)),
        "max": float(np.max(dists)),
        "min": float(np.min(dists)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mx.random.seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    results = {"is_smoke": IS_SMOKE}
    print(f"Loading model {MODEL_ID}...")
    model, tokenizer = mlx_load(MODEL_ID)

    print("Loading GSM8K data...")
    train_examples = load_gsm8k("train", n=200 if not IS_SMOKE else 10)
    test_examples = load_gsm8k("test", n=N_EVAL, seed=SEED + 1)
    train_data = tokenize_for_training(tokenizer, train_examples, MAX_SEQ_LEN)
    print(f"Train: {len(train_examples)} | Test: {len(test_examples)}")

    # -------------------------------------------------------------------------
    # Phase 1: PoLAR training with Stiefel retraction
    # -------------------------------------------------------------------------
    print(f"\n=== Phase 1: PoLAR (r={RANK}, retract every {RETRACT_EVERY} steps) ===")
    freeze_model(model)
    n_layers = inject_adapters(model, PoLARLinear)
    print(f"Injected PoLAR into {n_layers} layers | params: "
          f"{(RANK * D_IN + D_OUT * RANK) * n_layers:,}")

    t0 = time.time()
    polar_train = train_adapter(
        model, tokenizer, train_data, PoLARLinear, TRAIN_STEPS, RETRACT_EVERY
    )
    polar_time = time.time() - t0
    print(f"PoLAR training: {polar_time:.1f}s")

    # K1021: near-Stiefel (post-retraction)
    stiefel = measure_stiefel_stats(model)
    k1021_pass = stiefel["max"] < 0.01
    print(f"K1021 ||UU^T-I||_F: max={stiefel['max']:.2e}  "
          f"mean={stiefel['mean']:.2e}  → {'PASS' if k1021_pass else 'FAIL'}")

    # K1022: stable rank (post-retraction)
    polar_sr = measure_stable_ranks(model, PoLARLinear)
    k1022_pass = polar_sr["mean"] >= 5.0
    print(f"K1022 sr(PoLAR): mean={polar_sr['mean']:.2f}  "
          f"min={polar_sr['min']:.2f}  → {'PASS' if k1022_pass else 'FAIL'} (threshold 5)")

    # K1023 part 1: PoLAR quality
    print("Evaluating PoLAR on GSM8K...")
    polar_acc = eval_gsm8k(model, tokenizer, test_examples)
    print(f"PoLAR GSM8K: {polar_acc:.1%}")

    del model
    mx.clear_cache()
    gc.collect()

    # -------------------------------------------------------------------------
    # Phase 2: LoRA baseline
    # -------------------------------------------------------------------------
    print(f"\n=== Phase 2: LoRA baseline (r={RANK}, no retraction) ===")
    model, tokenizer = mlx_load(MODEL_ID)
    freeze_model(model)
    inject_adapters(model, LoRALinear)

    t0 = time.time()
    lora_train = train_adapter(
        model, tokenizer, train_data, LoRALinear, TRAIN_STEPS, retract_every=0
    )
    lora_time = time.time() - t0
    print(f"LoRA training: {lora_time:.1f}s")

    lora_sr = measure_stable_ranks(model, LoRALinear)
    print(f"sr(LoRA): mean={lora_sr['mean']:.2f}  min={lora_sr['min']:.2f}")

    print("Evaluating LoRA on GSM8K...")
    lora_acc = eval_gsm8k(model, tokenizer, test_examples)
    print(f"LoRA GSM8K: {lora_acc:.1%}")

    del model
    mx.clear_cache()
    gc.collect()

    # -------------------------------------------------------------------------
    # Final results
    # -------------------------------------------------------------------------
    k1023_pass = polar_acc >= lora_acc

    print("\n=== Kill Criteria ===")
    print(f"K1021 ||UU^T-I||_F_max={stiefel['max']:.2e}: {'PASS' if k1021_pass else 'FAIL'} (<0.01)")
    print(f"K1022 sr(PoLAR)={polar_sr['mean']:.2f} vs sr(LoRA)={lora_sr['mean']:.2f}: "
          f"{'PASS' if k1022_pass else 'FAIL'} (>=5)")
    print(f"K1023 PoLAR={polar_acc:.1%} vs LoRA={lora_acc:.1%}: "
          f"{'PASS' if k1023_pass else 'FAIL'}")

    results.update({
        "polar": {
            "train": polar_train,
            "stiefel_distance": stiefel,
            "stable_rank": polar_sr,
            "gsm8k_accuracy": polar_acc,
            "train_time_s": polar_time,
        },
        "lora": {
            "train": lora_train,
            "stable_rank": lora_sr,
            "gsm8k_accuracy": lora_acc,
            "train_time_s": lora_time,
        },
        "kill_criteria": {
            "K1021": {"value": stiefel["max"], "threshold": 0.01, "pass": k1021_pass},
            "K1022": {"polar_sr": polar_sr["mean"], "lora_sr": lora_sr["mean"],
                      "threshold": 5.0, "pass": k1022_pass},
            "K1023": {"polar_acc": polar_acc, "lora_acc": lora_acc, "pass": k1023_pass},
        },
        "hyperparams": {
            "rank": RANK, "scale": SCALE, "retract_every": RETRACT_EVERY,
            "train_steps": TRAIN_STEPS, "lr": LR, "model": MODEL_ID,
            "n_eval": N_EVAL,
        }
    })

    out_path = EXPERIMENT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out_path}")

    all_pass = k1021_pass and k1022_pass and k1023_pass
    print(f"\n{'ALL K PASS' if all_pass else 'SOME K FAIL'}: "
          f"K1021={'P' if k1021_pass else 'F'} "
          f"K1022={'P' if k1022_pass else 'F'} "
          f"K1023={'P' if k1023_pass else 'F'}")


if __name__ == "__main__":
    main()
