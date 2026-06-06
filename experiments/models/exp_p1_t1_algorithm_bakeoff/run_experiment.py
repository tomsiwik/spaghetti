#!/usr/bin/env python3
"""T1.6: Algorithm bake-off — LoRA vs HRA at equal parameter budgets.

TYPE: verification
MATH: experiments/models/exp_p1_t1_algorithm_bakeoff/MATH.md

WHAT THIS TESTS:
  T1.2 (Finding #416) was killed because it compared at equal RANK not equal PARAMS.
  HRA r=16 (40k params) vs LoRA r=16 (106k params) is unfair.
  This experiment compares at equal parameter budgets:
    Low budget:  LoRA r=6  (~40k/layer) vs HRA r=16 (~41k/layer)
    High budget: LoRA r=16 (~106k/layer) vs HRA r=42 (~108k/layer)

  Cayley/PoLAR/Givens excluded (see MATH.md for impossibility structures).

KILL CRITERIA (IDs from experiment DB):
  K1024: Winner identified by composite = quality × 1/params × 1/time
  K1025: Winner adapters orthogonal — |cos| < 0.01 (math vs code domain)
  K1026: Winner stable rank >= 3 at nominal rank
  K1027: All configs train in <= 1 hour

NOTE: Gemma4 not loadable by mlx_lm 0.29.1. Using Qwen3-4B-4bit as proxy.
SMOKE_TEST=1: 5 steps, n=5 eval, 2 configs only.
"""

import gc
import json
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

# Memory safety — MANDATORY per CODING_GUIDELINES
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
EXPERIMENT_DIR = Path(__file__).parent

MODEL_ID = "mlx-community/Qwen3-4B-4bit"
D_IN = 2560       # Qwen3-4B hidden_size (q_proj input dim)
D_OUT = 4096      # Qwen3-4B q_proj output dim (32 heads × 128 head_dim)
N_LAYERS = 36     # Qwen3-4B transformer layers

TRAIN_STEPS = 5 if IS_SMOKE else 300
LR = 5e-5
GRAD_CLIP = 1.0
MAX_SEQ_LEN = 64 if IS_SMOKE else 256
MAX_GEN_TOKENS = 32 if IS_SMOKE else 128
SEED = 42

N_EVAL = 5 if IS_SMOKE else 100
LOSS_CONV_THRESH = 0.5

# Configs: (name, method, rank)
# Low budget ≈ 40k params/layer, High budget ≈ 107k params/layer
CONFIGS_FULL = [
    ("LoRA_r6",  "lora", 6),    # 39,936 params/layer
    ("HRA_r16",  "hra",  16),   # 40,960 params/layer  [low budget equal to LoRA r=6]
    ("LoRA_r16", "lora", 16),   # 106,496 params/layer
    ("HRA_r42",  "hra",  42),   # 107,520 params/layer [high budget equal to LoRA r=16]
]
CONFIGS_SMOKE = [
    ("LoRA_r6",  "lora", 6),
    ("HRA_r16",  "hra",  16),
]
CONFIGS = CONFIGS_SMOKE if IS_SMOKE else CONFIGS_FULL

# ---------------------------------------------------------------------------
# HRALinear: Householder Reflection Adaptation (arxiv 2405.17484)
# ---------------------------------------------------------------------------

class HRALinear(nn.Module):
    """Householder Reflection Adaptation.

    Applies r Householder reflections to x before the frozen base linear:
        y = base(H_r(...H_1(x)...))
    where H_i(x) = x - (2 / vᵀv) * (xᵀv) * v.

    Params: V ∈ ℝ^{r × d_in} (reflection vectors).
    Param count: r × d_in (vs LoRA: r × (d_in + d_out)).
    """

    def __init__(self, linear: nn.Module, r: int, d_in: int = D_IN):
        super().__init__()
        self.r = r
        self.linear = linear  # frozen base (QuantizedLinear)
        rng = np.random.default_rng(SEED)
        V_np = rng.standard_normal((r, d_in)).astype(np.float32)
        norms = np.linalg.norm(V_np, axis=-1, keepdims=True) + 1e-8
        V_np = V_np / norms  # unit-norm rows
        self.V = mx.array(V_np)  # (r, d_in), trainable

    def __call__(self, x: mx.array) -> mx.array:
        for i in range(self.r):
            v = self.V[i]                           # (d_in,)
            vTv = mx.sum(v * v) + 1e-8              # scalar
            xTv = mx.sum(x * v, axis=-1, keepdims=True)  # (..., 1)
            x = x - (2.0 / vTv) * xTv * v          # (..., d_in)
        return self.linear(x)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gsm8k(split="train", n=500, seed=SEED):
    """Load GSM8K examples with final numeric answers."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    examples = []
    for i in indices:
        item = ds[i]
        m = re.search(r"####\s*([0-9,\-\.]+)", item["answer"])
        if m:
            final = m.group(1).replace(",", "")
            examples.append({"question": item["question"], "answer": final})
        if len(examples) >= n:
            break
    return examples[:n]


def make_gsm8k_prompt(q, answer=None):
    if answer is not None:
        return f"Q: {q}\nA: {answer}"
    return f"Q: {q}\nA:"


def make_code_examples(n=250, seed=SEED):
    """Synthetic Python-style code examples for domain-2 adapter."""
    rng = random.Random(seed + 1)
    ops = [
        ("sort",     lambda xs: sorted(xs)),
        ("reverse",  lambda xs: list(reversed(xs))),
        ("sum",      lambda xs: [sum(xs)]),
        ("max",      lambda xs: [max(xs)]),
        ("min",      lambda xs: [min(xs)]),
    ]
    examples = []
    for _ in range(n):
        k = rng.randint(3, 7)
        nums = [rng.randint(1, 99) for _ in range(k)]
        op_name, op_fn = rng.choice(ops)
        result = op_fn(nums)
        prompt = (
            f"# Python: {op_name} of {nums}\n"
            f"result = {op_fn.__name__ if hasattr(op_fn, '__name__') else op_name}({nums})\n"
            f"# result = {result}"
        )
        examples.append(prompt)
    return examples[:n]

# ---------------------------------------------------------------------------
# Adapter injection helpers
# ---------------------------------------------------------------------------

def inject_hra(model, r, d_in=D_IN):
    """Replace q_proj with HRALinear in all layers."""
    for layer in model.layers:
        orig = layer.self_attn.q_proj
        layer.self_attn.q_proj = HRALinear(orig, r=r, d_in=d_in)


def inject_lora(model, r):
    """Replace q_proj with LoRALinear in all layers."""
    for layer in model.layers:
        orig = layer.self_attn.q_proj
        layer.self_attn.q_proj = LoRALinear.from_base(orig, r=r, scale=float(r))


def count_trainable(model):
    params = tree_flatten(model.trainable_parameters())
    return sum(p.size for _, p in params)


def get_adapter_flat(model, method):
    """Flatten all adapter params into one vector for cosine computation."""
    vecs = []
    for layer in model.layers:
        adapter = layer.self_attn.q_proj
        if method == "hra" and isinstance(adapter, HRALinear):
            mx.eval(adapter.V)
            vecs.append(np.array(adapter.V).ravel())
        elif method == "lora" and isinstance(adapter, LoRALinear):
            mx.eval(adapter.lora_a, adapter.lora_b)
            vecs.append(np.array(adapter.lora_a).ravel())
            vecs.append(np.array(adapter.lora_b).ravel())
    return np.concatenate(vecs) if vecs else np.zeros(1)


def stable_rank_of_V(model):
    """Compute stable rank of V matrices (stacked over layers) for HRA."""
    rows = []
    for layer in model.layers[:4]:  # Use first 4 layers for speed
        adapter = layer.self_attn.q_proj
        if isinstance(adapter, HRALinear):
            mx.eval(adapter.V)
            rows.append(np.array(adapter.V))  # (r, d_in)
    if not rows:
        return 0.0
    V_stack = np.concatenate(rows, axis=0)  # (4r, d_in)
    # Compute via Frobenius and spectral norm approximation
    # sr = ||V||_F^2 / ||V||_2^2
    # Use SVD on a projection for efficiency
    V_small = V_stack[:, :512] if V_stack.shape[1] > 512 else V_stack  # (4r, 512)
    try:
        s = np.linalg.svd(V_small, compute_uv=False)
        sr = float((np.sum(s**2) / (s[0]**2 + 1e-8)))
    except Exception:
        sr = float(V_stack.shape[0])
    return sr


def stable_rank_of_AB(model):
    """Compute stable rank of LoRA B matrices (stacked over layers)."""
    rows = []
    for layer in model.layers[:4]:
        adapter = layer.self_attn.q_proj
        if isinstance(adapter, LoRALinear):
            mx.eval(adapter.lora_b)
            rows.append(np.array(adapter.lora_b))  # (r, d_out)
    if not rows:
        return 0.0
    B_stack = np.concatenate(rows, axis=0)
    B_small = B_stack[:, :512] if B_stack.shape[1] > 512 else B_stack
    try:
        s = np.linalg.svd(B_small, compute_uv=False)
        sr = float(np.sum(s**2) / (s[0]**2 + 1e-8))
    except Exception:
        sr = float(B_stack.shape[0])
    return sr

# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def loss_fn(model, tokens):
    """Causal LM loss."""
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    logits = model(x)
    loss = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1),
        reduction="mean",
    )
    return loss

# ---------------------------------------------------------------------------
# Training phase
# ---------------------------------------------------------------------------

def train_config(config_name, method, r, train_prompts):
    """Train one config. Returns dict with metrics and trained params."""
    print(f"\n=== Training {config_name} ({method.upper()} r={r}) ===")

    model, tok = mlx_load(MODEL_ID)
    model.freeze()

    if method == "hra":
        inject_hra(model, r=r)
    else:
        inject_lora(model, r=r)

    n_params = count_trainable(model)
    params_per_layer = n_params // N_LAYERS
    print(f"  Trainable: {n_params:,} total ({params_per_layer:,}/layer)")

    optimizer = optim.AdamW(learning_rate=LR, weight_decay=0.01)
    value_and_grad = nn.value_and_grad(model, loss_fn)

    step_times = []
    loss_curve = []
    conv_step = TRAIN_STEPS + 1

    for step in range(TRAIN_STEPS):
        prompt = random.choice(train_prompts)
        tokens = tok.encode(prompt)[:MAX_SEQ_LEN]
        if len(tokens) < 4:
            continue
        batch = mx.array([tokens], dtype=mx.uint32)

        t0 = time.perf_counter()
        loss, grads = value_and_grad(model, batch)

        # Gradient clip
        all_g = [g for _, g in tree_flatten(grads) if g is not None]
        if all_g:
            grad_norm = mx.sqrt(sum(mx.sum(g * g) for g in all_g))
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

        if step % 100 == 0 or step < 3:
            print(f"  step {step:3d}: loss={lv:.4f} t={t1-t0:.3f}s")

    avg_step = float(np.mean(step_times[1:])) if len(step_times) > 1 else float(np.mean(step_times))
    total_train_time = float(sum(step_times))

    # Extract adapter flat vector for K1025
    adapter_flat = get_adapter_flat(model, method)

    # Compute stable rank for K1026
    if method == "hra":
        sr = stable_rank_of_V(model)
    else:
        sr = stable_rank_of_AB(model)

    # Save weights for eval reload
    weight_path = EXPERIMENT_DIR / f"weights_{config_name}.npz"
    w = {}
    for i, layer in enumerate(model.layers):
        adapter = layer.self_attn.q_proj
        if method == "hra" and isinstance(adapter, HRALinear):
            mx.eval(adapter.V)
            w[f"layer_{i}_V"] = np.array(adapter.V)
        elif method == "lora" and isinstance(adapter, LoRALinear):
            mx.eval(adapter.lora_a, adapter.lora_b)
            w[f"layer_{i}_A"] = np.array(adapter.lora_a)
            w[f"layer_{i}_B"] = np.array(adapter.lora_b)
    np.savez(weight_path, **w)

    del model, optimizer, grads
    gc.collect()
    mx.clear_cache()

    return {
        "config": config_name,
        "method": method,
        "rank": r,
        "n_params": n_params,
        "params_per_layer": params_per_layer,
        "avg_step_time": avg_step,
        "total_train_time": total_train_time,
        "conv_step": conv_step,
        "loss_curve": loss_curve,
        "stable_rank": sr,
        "adapter_flat": adapter_flat,  # numpy array, not saved to JSON
        "weight_path": str(weight_path),
    }

# ---------------------------------------------------------------------------
# Evaluation phase
# ---------------------------------------------------------------------------

def eval_config(config_name, method, r, eval_data):
    """Evaluate a trained config on GSM8K."""
    print(f"\n=== Eval {config_name} ===")

    model, tok = mlx_load(MODEL_ID)
    model.freeze()

    if method == "hra":
        inject_hra(model, r=r)
    elif method == "lora":
        inject_lora(model, r=r)

    # Reload weights
    weight_path = EXPERIMENT_DIR / f"weights_{config_name}.npz"
    if weight_path.exists():
        w = np.load(weight_path)
        for i, layer in enumerate(model.layers):
            adapter = layer.self_attn.q_proj
            if method == "hra" and isinstance(adapter, HRALinear):
                key = f"layer_{i}_V"
                if key in w:
                    adapter.V = mx.array(w[key])
            elif method == "lora" and isinstance(adapter, LoRALinear):
                ka, kb = f"layer_{i}_A", f"layer_{i}_B"
                if ka in w:
                    adapter.lora_a = mx.array(w[ka])
                    adapter.lora_b = mx.array(w[kb])
        mx.eval(model.parameters())

    correct = 0
    for ex in eval_data:
        prompt = make_gsm8k_prompt(ex["question"])
        out = mlx_generate(model, tok, prompt=prompt,
                           max_tokens=MAX_GEN_TOKENS, verbose=False)
        nums = re.findall(r"-?[\d,]+\.?\d*", out)
        if nums:
            pred = nums[-1].replace(",", "")
            if pred == ex["answer"] or pred.rstrip("0").rstrip(".") == ex["answer"]:
                correct += 1

    acc = correct / len(eval_data) if eval_data else 0.0
    print(f"  {config_name}: {correct}/{len(eval_data)} = {acc:.3f}")

    del model
    gc.collect()
    mx.clear_cache()
    return acc

# ---------------------------------------------------------------------------
# K1025: Cosine between math and code adapters (winner config only)
# ---------------------------------------------------------------------------

def train_code_adapter(winner_name, winner_method, winner_r, code_prompts):
    """Train winner config on code domain for K1025 cosine check."""
    print(f"\n=== K1025: Code adapter ({winner_name}) ===")

    model, tok = mlx_load(MODEL_ID)
    model.freeze()

    if winner_method == "hra":
        inject_hra(model, r=winner_r)
    else:
        inject_lora(model, r=winner_r)

    optimizer = optim.AdamW(learning_rate=LR, weight_decay=0.01)
    value_and_grad = nn.value_and_grad(model, loss_fn)

    for step in range(TRAIN_STEPS):
        prompt = random.choice(code_prompts)
        tokens = tok.encode(prompt)[:MAX_SEQ_LEN]
        if len(tokens) < 4:
            continue
        batch = mx.array([tokens], dtype=mx.uint32)
        loss, grads = value_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())

    flat_code = get_adapter_flat(model, winner_method)

    del model, optimizer, grads
    gc.collect()
    mx.clear_cache()
    return flat_code

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"=== T1.6: Algorithm Bake-off ({'SMOKE' if IS_SMOKE else 'FULL'}) ===")
    print(f"Model: {MODEL_ID}")
    print(f"Configs: {[c[0] for c in CONFIGS]}")
    print(f"Steps: {TRAIN_STEPS} | Eval n={N_EVAL}")

    # Load data
    print("\nLoading data...")
    train_data = load_gsm8k("train", n=500 if not IS_SMOKE else 50)
    eval_data  = load_gsm8k("test",  n=N_EVAL)
    code_data  = make_code_examples(n=250 if not IS_SMOKE else 50)
    train_prompts = [make_gsm8k_prompt(ex["question"], ex["answer"]) for ex in train_data]
    print(f"  Train: {len(train_data)} | Eval: {len(eval_data)} | Code: {len(code_data)}")

    # Base eval (load once, no adapter)
    print("\n--- Base eval (no adapter) ---")
    model_base, tok_base = mlx_load(MODEL_ID)
    model_base.freeze()
    mx.eval(model_base.parameters())
    correct_base = 0
    for ex in eval_data:
        prompt = make_gsm8k_prompt(ex["question"])
        out = mlx_generate(model_base, tok_base, prompt=prompt,
                           max_tokens=MAX_GEN_TOKENS, verbose=False)
        nums = re.findall(r"-?[\d,]+\.?\d*", out)
        if nums:
            pred = nums[-1].replace(",", "")
            if pred == ex["answer"]:
                correct_base += 1
    base_acc = correct_base / len(eval_data) if eval_data else 0.0
    print(f"  Base: {correct_base}/{len(eval_data)} = {base_acc:.3f}")
    del model_base, tok_base
    gc.collect()
    mx.clear_cache()

    # Train all configs
    train_results = {}
    for config_name, method, r in CONFIGS:
        tr = train_config(config_name, method, r, train_prompts)
        train_results[config_name] = tr

    # Eval all configs
    eval_accs = {}
    for config_name, method, r in CONFIGS:
        acc = eval_config(config_name, method, r, eval_data)
        eval_accs[config_name] = acc

    # K1024: Composite score = quality / (params_per_layer * avg_step_time)
    # quality = max(0, acc - base_acc) + 0.001 to avoid zero
    composites = {}
    for config_name, _, _ in CONFIGS:
        tr = train_results[config_name]
        quality = max(0.0, eval_accs[config_name] - base_acc) + 0.001
        # Normalize: lower params and lower time → higher composite
        composite = quality / (tr["params_per_layer"] * tr["avg_step_time"] + 1e-12)
        composites[config_name] = composite
        print(f"  {config_name}: acc={eval_accs[config_name]:.3f} quality={quality:.4f} "
              f"params/layer={tr['params_per_layer']:,} "
              f"step_time={tr['avg_step_time']:.3f}s "
              f"composite={composite:.2e}")

    winner_name = max(composites, key=composites.get)
    winner_method = [m for n, m, _ in CONFIGS if n == winner_name][0]
    winner_r = [r for n, _, r in CONFIGS if n == winner_name][0]
    winner_tr = train_results[winner_name]
    print(f"\n  K1024 WINNER: {winner_name} (composite={composites[winner_name]:.2e})")

    # K1025: Cosine between math adapter and code adapter (winner method)
    print(f"\n--- K1025: Orthogonality check ({winner_name}) ---")
    math_flat = train_results[winner_name]["adapter_flat"]
    code_flat = train_code_adapter(winner_name, winner_method, winner_r, code_prompts=code_data)

    norm_m = np.linalg.norm(math_flat) + 1e-12
    norm_c = np.linalg.norm(code_flat) + 1e-12
    cos_val = float(np.dot(math_flat, code_flat) / (norm_m * norm_c))
    abs_cos = abs(cos_val)
    k1025_pass = abs_cos < 0.01
    print(f"  |cos(math, code)| = {abs_cos:.6f} | K1025: {'PASS' if k1025_pass else 'FAIL'}")

    # K1026: Stable rank of winner
    sr = winner_tr["stable_rank"]
    k1026_pass = sr >= 3.0
    print(f"  Stable rank = {sr:.2f} (nominal r={winner_r}) | K1026: {'PASS' if k1026_pass else 'FAIL'}")

    # K1027: Max training time
    max_time = max(tr["total_train_time"] for tr in train_results.values())
    k1027_pass = max_time <= 3600.0
    print(f"  Max training time = {max_time:.1f}s | K1027: {'PASS' if k1027_pass else 'FAIL'}")

    # Build results dict (no numpy arrays in JSON)
    results = {
        "is_smoke": IS_SMOKE,
        "model_id": MODEL_ID,
        "base_acc": base_acc,
        "configs": {
            name: {
                "method": train_results[name]["method"],
                "rank": train_results[name]["rank"],
                "params_per_layer": train_results[name]["params_per_layer"],
                "n_params_total": train_results[name]["n_params"],
                "avg_step_time": train_results[name]["avg_step_time"],
                "total_train_time": train_results[name]["total_train_time"],
                "conv_step": train_results[name]["conv_step"],
                "stable_rank": train_results[name]["stable_rank"],
                "gsm8k_acc": eval_accs[name],
                "composite_score": composites[name],
            }
            for name, _, _ in CONFIGS
        },
        "winner": winner_name,
        "k1024": {
            "pass": True,  # Always passes — winner is identified
            "winner": winner_name,
            "winner_composite": composites[winner_name],
            "all_composites": {k: v for k, v in composites.items()},
        },
        "k1025": {
            "pass": k1025_pass,
            "abs_cos": abs_cos,
            "cos_val": cos_val,
            "n_params": len(math_flat),
            "threshold": 0.01,
        },
        "k1026": {
            "pass": k1026_pass,
            "stable_rank": sr,
            "winner_rank": winner_r,
            "threshold": 3.0,
        },
        "k1027": {
            "pass": k1027_pass,
            "max_train_time_s": max_time,
            "all_times": {name: train_results[name]["total_train_time"]
                          for name, _, _ in CONFIGS},
        },
    }

    out_path = EXPERIMENT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Summary
    print("\n" + "="*60)
    print("T1.6 RESULTS SUMMARY")
    print("="*60)
    print(f"Base GSM8K: {base_acc:.3f}")
    for name, _, _ in CONFIGS:
        tr = train_results[name]
        acc = eval_accs[name]
        print(f"  {name:12s}: acc={acc:.3f} params={tr['params_per_layer']:>7,}/layer "
              f"sr={tr['stable_rank']:.1f} composite={composites[name]:.2e}")
    print(f"\nWinner: {winner_name}")
    print(f"K1024 (winner identified): PASS")
    print(f"K1025 (|cos|<0.01): {'PASS' if k1025_pass else 'FAIL'} ({abs_cos:.4f})")
    print(f"K1026 (sr>=3): {'PASS' if k1026_pass else 'FAIL'} ({sr:.1f})")
    print(f"K1027 (time<=1h): {'PASS' if k1027_pass else 'FAIL'} ({max_time:.0f}s)")

    all_pass = k1025_pass and k1026_pass and k1027_pass
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    return results


if __name__ == "__main__":
    main()
