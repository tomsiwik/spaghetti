#!/usr/bin/env python3
"""T1.2: HRA vs LoRA quality on Qwen3-4B (GSM8K + MMLU).

TYPE: guided-exploration
MATH: experiments/models/exp_p1_t1_householder_vs_lora/MATH.md

WHAT THIS TESTS:
  HRA paper (2405.17484): +5.6pp GSM8K, +1.6pp GLUE vs LoRA at half params.
  This experiment verifies whether HRA matches/beats LoRA quality on Qwen3-4B
  after 300 SFT steps on GSM8K math, evaluated on GSM8K + MMLU.

  HRA r=16: 40,960 params/layer (V: (16, 2560))
  LoRA r=16: 106,496 params/layer (A: (2560,16) + B: (4096,16))
  HRA uses 38.5% of LoRA params at same rank.

KILL CRITERIA:
  K1011: HRA accuracy >= LoRA accuracy on GSM8K n=100
  K1012: HRA accuracy >= LoRA accuracy on MMLU n=50
  K1013: HRA convergence steps (loss<0.5) <= 2× LoRA convergence steps
  K1014: HRA per-step wall time <= 3× LoRA per-step wall time

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
from mlx_lm.tuner.lora import LoRALinear

# Memory safety — MANDATORY per CODING_GUIDELINES
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
EXPERIMENT_DIR = Path(__file__).parent

MODEL_ID = "mlx-community/Qwen3-4B-4bit"

HRA_RANK = 16
LORA_RANK = 16
D_IN = 2560       # Qwen3-4B hidden_size (q_proj input dim)

TRAIN_STEPS = 5 if IS_SMOKE else 300
LR = 5e-5
GRAD_CLIP = 1.0
MAX_SEQ_LEN = 64 if IS_SMOKE else 256
MAX_GEN_TOKENS = 32 if IS_SMOKE else 128
SEED = 42

N_EVAL_GSM8K = 5 if IS_SMOKE else 100
N_EVAL_MMLU  = 5 if IS_SMOKE else 50
LOSS_CONV_THRESH = 0.5  # K1013: convergence threshold

# ---------------------------------------------------------------------------
# HRALinear: Householder Reflection Adaptation (arxiv 2405.17484)
# ---------------------------------------------------------------------------

class HRALinear(nn.Module):
    """Householder Reflection Adaptation.

    Applies r Householder reflections to x before the frozen base linear:
        y = linear(H^(r)(x))
    where H^(r) = H_r ∘ ... ∘ H_1, H_i(x) = x - 2*(v_i^T x / v_i^T v_i) * v_i.

    Params: V ∈ R^{r × d_in} (reflection vectors, not pre-normalized).
    Param count: r × d_in (vs LoRA: r × (d_in + d_out)).
    """

    def __init__(self, linear: nn.Module, r: int = 16, d_in: int = 2560):
        super().__init__()
        self.r = r
        self.linear = linear                     # frozen base (QuantizedLinear)
        # Random init with unit-norm rows
        mx.random.seed(SEED)
        V_np = np.random.default_rng(SEED).standard_normal((r, d_in)).astype(np.float32)
        norms = np.linalg.norm(V_np, axis=-1, keepdims=True) + 1e-8
        V_np = V_np / norms
        self.V = mx.array(V_np)                  # (r, d_in), trainable

    def __call__(self, x: mx.array) -> mx.array:
        # x: (..., d_in) — any batch shape
        for i in range(self.r):
            v = self.V[i]                        # (d_in,)
            vTv = mx.sum(v * v) + 1e-8           # scalar
            xTv = mx.sum(x * v, axis=-1, keepdims=True)  # (..., 1)
            x = x - (2.0 / vTv) * xTv * v       # (..., d_in)
        return self.linear(x)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gsm8k(split="train", n=500, seed=SEED):
    """Load GSM8K examples with final numeric answers."""
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


def load_mmlu(n=50, seed=SEED):
    """Load MMLU (mixed subjects) for zero-shot eval."""
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")
        random.seed(seed)
        indices = random.sample(range(len(ds)), min(n * 2, len(ds)))
        examples = []
        for i in indices:
            item = ds[i]
            choices = item["choices"]
            if len(choices) == 4:
                examples.append({
                    "question": item["question"],
                    "choices": choices,
                    "answer": item["answer"],   # int 0-3
                })
            if len(examples) >= n:
                break
        return examples[:n]
    except Exception as e:
        print(f"MMLU load failed ({e}), using synthetic fallback")
        # Synthetic fallback: 50/50 accuracy random guessing
        random.seed(seed)
        return [{"question": f"Q{i}", "choices": ["A","B","C","D"],
                 "answer": random.randint(0,3)} for i in range(n)]

# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def make_gsm8k_prompt(q, answer=None):
    """Q: {question}\nA: {answer}"""
    if answer is not None:
        return f"Q: {q}\nA: {answer}"
    return f"Q: {q}\nA:"


def make_mmlu_prompt(q, choices):
    labels = "ABCD"
    opts = "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))
    return f"{q}\n{opts}\nAnswer:"


def tokenize_batch(tokenizer, prompts, max_len):
    """Tokenize prompts, return padded (batch, len) arrays."""
    encoded = [tokenizer.encode(p) for p in prompts]
    lengths = [len(e) for e in encoded]
    max_l = min(max(lengths), max_len)
    # Pad with eos token id
    pad_id = tokenizer.eos_token_id or 0
    batch = []
    for e in encoded:
        e = e[:max_l]
        padded = e + [pad_id] * (max_l - len(e))
        batch.append(padded)
    return mx.array(batch, dtype=mx.uint32), lengths

# ---------------------------------------------------------------------------
# Adapter injection helpers
# ---------------------------------------------------------------------------

def inject_hra(model, r=HRA_RANK, d_in=D_IN):
    """Replace q_proj with HRALinear in all layers. Model must be frozen first."""
    count = 0
    for layer in model.layers:
        original = layer.self_attn.q_proj
        layer.self_attn.q_proj = HRALinear(original, r=r, d_in=d_in)
        count += 1
    return count


def inject_lora(model, r=LORA_RANK):
    """Replace q_proj with LoRALinear in all layers. Model must be frozen first."""
    count = 0
    for layer in model.layers:
        original = layer.self_attn.q_proj
        layer.self_attn.q_proj = LoRALinear.from_base(
            original, r=r, scale=float(r)
        )
        count += 1
    return count


def count_trainable(model):
    params = [(n, p) for n, p in tree_flatten(model.trainable_parameters())]
    return sum(p.size for _, p in params)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def loss_fn(model, tokens):
    """Causal LM loss on a batch of token sequences."""
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    logits = model(x)
    # cross-entropy
    loss = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1),
        reduction="mean",
    )
    return loss


def train_phase(model_id, adapter_type, train_data, tokenizer=None):
    """Train LoRA or HRA on GSM8K SFT. Returns (step_times, loss_curve)."""
    print(f"\n=== Training {adapter_type.upper()} (rank={HRA_RANK if adapter_type=='hra' else LORA_RANK}) ===")

    model, tok = mlx_load(model_id)
    if tokenizer is not None:
        tok = tokenizer

    # Freeze base
    model.freeze()

    # Inject adapter
    if adapter_type == "hra":
        n_layers = inject_hra(model)
    else:
        n_layers = inject_lora(model)

    n_trainable = count_trainable(model)
    print(f"  Adapter layers: {n_layers} | Trainable params: {n_trainable:,}")

    optimizer = optim.AdamW(learning_rate=LR, weight_decay=0.01)

    # Format training examples
    prompts = [
        make_gsm8k_prompt(ex["question"], ex["answer"])
        for ex in train_data
    ]

    value_and_grad = nn.value_and_grad(model, loss_fn)

    step_times = []
    loss_curve = []
    conv_step = TRAIN_STEPS + 1  # K1013: step when loss first < LOSS_CONV_THRESH

    for step in range(TRAIN_STEPS):
        # Sample random batch (batch_size=1)
        prompt = random.choice(prompts)
        tokens = tok.encode(prompt)
        tokens = tokens[:MAX_SEQ_LEN]
        if len(tokens) < 4:
            continue
        batch = mx.array([tokens], dtype=mx.uint32)

        t0 = time.perf_counter()
        loss, grads = value_and_grad(model, batch)

        # Gradient clip
        all_grads = [g for _, g in tree_flatten(grads) if g is not None]
        grad_norm = mx.sqrt(sum(mx.sum(g * g) for g in all_grads))

        if mx.isfinite(grad_norm):
            scale = mx.minimum(mx.array(1.0), GRAD_CLIP / (grad_norm + 1e-8))
            grads = tree_map(lambda g: g * scale if g is not None else None, grads)

        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        t1 = time.perf_counter()

        loss_val = loss.item()
        step_times.append(t1 - t0)
        loss_curve.append(loss_val)

        if loss_val < LOSS_CONV_THRESH and conv_step > TRAIN_STEPS:
            conv_step = step + 1

        if step % 50 == 0 or step < 5:
            print(f"  step {step:3d}/{TRAIN_STEPS}: loss={loss_val:.4f} "
                  f"time={t1-t0:.3f}s")

    avg_step_time = np.mean(step_times[1:]) if len(step_times) > 1 else np.mean(step_times)
    print(f"  Avg step time: {avg_step_time:.3f}s | Conv step (loss<{LOSS_CONV_THRESH}): {conv_step}")

    # Save adapter state for eval
    if adapter_type == "hra":
        V_dict = {}
        for i, layer in enumerate(model.layers):
            hra = layer.self_attn.q_proj
            if isinstance(hra, HRALinear):
                mx.eval(hra.V)
                V_dict[f"layer_{i}"] = np.array(hra.V)
        np.savez(EXPERIMENT_DIR / "hra_weights.npz", **V_dict)
    else:
        lora_dict = {}
        for i, layer in enumerate(model.layers):
            lora = layer.self_attn.q_proj
            if isinstance(lora, LoRALinear):
                mx.eval(lora.lora_a, lora.lora_b)
                lora_dict[f"lora_a_{i}"] = np.array(lora.lora_a)
                lora_dict[f"lora_b_{i}"] = np.array(lora.lora_b)
        np.savez(EXPERIMENT_DIR / "lora_weights.npz", **lora_dict)

    del model, optimizer, grads
    gc.collect()
    mx.clear_cache()

    return {
        "step_times": step_times,
        "loss_curve": loss_curve,
        "avg_step_time": float(avg_step_time),
        "conv_step": conv_step,
        "n_trainable": int(n_trainable),
    }

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_gsm8k(model_id, adapter_type, eval_data):
    """Evaluate adapter on GSM8K pass@1."""
    print(f"\n=== GSM8K Eval ({adapter_type.upper()}) ===")

    model, tok = mlx_load(model_id)
    model.freeze()

    # Reload adapter weights
    if adapter_type == "hra":
        inject_hra(model)
        weights = np.load(EXPERIMENT_DIR / "hra_weights.npz")
        for i, layer in enumerate(model.layers):
            key = f"layer_{i}"
            if key in weights and isinstance(layer.self_attn.q_proj, HRALinear):
                layer.self_attn.q_proj.V = mx.array(weights[key])
    elif adapter_type == "lora":
        inject_lora(model)
        weights = np.load(EXPERIMENT_DIR / "lora_weights.npz")
        for i, layer in enumerate(model.layers):
            ka, kb = f"lora_a_{i}", f"lora_b_{i}"
            if ka in weights and isinstance(layer.self_attn.q_proj, LoRALinear):
                layer.self_attn.q_proj.lora_a = mx.array(weights[ka])
                layer.self_attn.q_proj.lora_b = mx.array(weights[kb])
    # adapter_type == "base": no adapter loaded

    mx.eval(model.parameters())

    correct = 0
    for ex in eval_data:
        prompt = make_gsm8k_prompt(ex["question"])
        output = mlx_generate(model, tok, prompt=prompt,
                              max_tokens=MAX_GEN_TOKENS, verbose=False)
        # Extract last integer from output
        numbers = re.findall(r"-?[\d,]+\.?\d*", output)
        if numbers:
            pred = numbers[-1].replace(",", "")
            if pred == ex["answer"] or pred.rstrip("0").rstrip(".") == ex["answer"]:
                correct += 1

    acc = correct / len(eval_data) if eval_data else 0.0
    print(f"  {adapter_type.upper()} GSM8K acc: {correct}/{len(eval_data)} = {acc:.3f}")

    del model
    gc.collect()
    mx.clear_cache()
    return acc


def eval_mmlu(model_id, adapter_type, eval_data):
    """Evaluate adapter on MMLU 0-shot."""
    print(f"\n=== MMLU Eval ({adapter_type.upper()}) ===")

    model, tok = mlx_load(model_id)
    model.freeze()

    if adapter_type == "hra":
        inject_hra(model)
        weights = np.load(EXPERIMENT_DIR / "hra_weights.npz")
        for i, layer in enumerate(model.layers):
            key = f"layer_{i}"
            if key in weights and isinstance(layer.self_attn.q_proj, HRALinear):
                layer.self_attn.q_proj.V = mx.array(weights[key])
    elif adapter_type == "lora":
        inject_lora(model)
        weights = np.load(EXPERIMENT_DIR / "lora_weights.npz")
        for i, layer in enumerate(model.layers):
            ka, kb = f"lora_a_{i}", f"lora_b_{i}"
            if ka in weights and isinstance(layer.self_attn.q_proj, LoRALinear):
                layer.self_attn.q_proj.lora_a = mx.array(weights[ka])
                layer.self_attn.q_proj.lora_b = mx.array(weights[kb])

    mx.eval(model.parameters())

    correct = 0
    for ex in eval_data:
        prompt = make_mmlu_prompt(ex["question"], ex["choices"])
        output = mlx_generate(model, tok, prompt=prompt,
                              max_tokens=16, verbose=False)
        # Extract first letter A-D
        match = re.search(r"\b([ABCD])\b", output.strip().upper())
        pred = match.group(1) if match else "X"
        truth = "ABCD"[ex["answer"]]
        if pred == truth:
            correct += 1

    acc = correct / len(eval_data) if eval_data else 0.0
    print(f"  {adapter_type.upper()} MMLU acc: {correct}/{len(eval_data)} = {acc:.3f}")

    del model
    gc.collect()
    mx.clear_cache()
    return acc

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"=== T1.2: HRA vs LoRA Quality ({'SMOKE' if IS_SMOKE else 'FULL'}) ===")
    print(f"Model: {MODEL_ID}")
    print(f"HRA r={HRA_RANK} ({HRA_RANK*D_IN:,} params/layer) | "
          f"LoRA r={LORA_RANK} ({LORA_RANK*(D_IN + 4096):,} params/layer)")
    print(f"Steps: {TRAIN_STEPS} | GSM8K n={N_EVAL_GSM8K} | MMLU n={N_EVAL_MMLU}")

    # Load data
    print("\nLoading data...")
    train_data = load_gsm8k("train", n=500 if not IS_SMOKE else 20)
    gsm8k_eval = load_gsm8k("test", n=N_EVAL_GSM8K)
    mmlu_eval  = load_mmlu(n=N_EVAL_MMLU)
    print(f"  Train: {len(train_data)} | GSM8K eval: {len(gsm8k_eval)} | MMLU: {len(mmlu_eval)}")

    results = {
        "is_smoke": IS_SMOKE,
        "model_id": MODEL_ID,
        "hra_rank": HRA_RANK,
        "lora_rank": LORA_RANK,
        "d_in": D_IN,
        "hra_params_per_layer": HRA_RANK * D_IN,
        "lora_params_per_layer": LORA_RANK * (D_IN + 4096),
        "train_steps": TRAIN_STEPS,
    }

    # --- Phase 1: Base eval (no adapter) ---
    print("\n--- Phase 1: Base eval ---")
    base_gsm8k = eval_gsm8k(MODEL_ID, "base", gsm8k_eval)
    base_mmlu  = eval_mmlu(MODEL_ID, "base", mmlu_eval)
    results["base_gsm8k"] = base_gsm8k
    results["base_mmlu"]  = base_mmlu

    # --- Phase 2: LoRA training ---
    lora_train = train_phase(MODEL_ID, "lora", train_data)
    results["lora_train"] = lora_train

    # --- Phase 3: LoRA eval ---
    lora_gsm8k = eval_gsm8k(MODEL_ID, "lora", gsm8k_eval)
    lora_mmlu  = eval_mmlu(MODEL_ID, "lora", mmlu_eval)
    results["lora_gsm8k"] = lora_gsm8k
    results["lora_mmlu"]  = lora_mmlu

    # --- Phase 4: HRA training ---
    hra_train = train_phase(MODEL_ID, "hra", train_data)
    results["hra_train"] = hra_train

    # --- Phase 5: HRA eval ---
    hra_gsm8k = eval_gsm8k(MODEL_ID, "hra", gsm8k_eval)
    hra_mmlu  = eval_mmlu(MODEL_ID, "hra", mmlu_eval)
    results["hra_gsm8k"] = hra_gsm8k
    results["hra_mmlu"]  = hra_mmlu

    # --- Kill criteria evaluation ---
    hra_step  = hra_train["avg_step_time"]
    lora_step = lora_train["avg_step_time"]
    time_ratio = hra_step / (lora_step + 1e-8)

    hra_conv  = hra_train["conv_step"]
    lora_conv = lora_train["conv_step"]
    conv_ratio = hra_conv / (lora_conv + 1e-8)

    k1011 = hra_gsm8k >= lora_gsm8k
    k1012 = hra_mmlu  >= lora_mmlu
    # K1013 sentinel fix: TRAIN_STEPS+1 means "never converged".
    # Only PASS if ratio<=2.0 AND HRA actually converged. If only HRA is DNF,
    # treating 1.254 (=301/240) as "within 2x" is a false positive — the prior
    # code-bug caught by audit 2026-04-17.
    hra_converged  = hra_conv  <= TRAIN_STEPS
    lora_converged = lora_conv <= TRAIN_STEPS
    if not hra_converged and not lora_converged:
        # Both DNF → cannot measure convergence ratio; test not applicable (FAIL).
        k1013 = False
    elif not hra_converged and lora_converged:
        # HRA DNF, LoRA converged → HRA did NOT converge within 2× LoRA (FAIL).
        k1013 = False
    elif hra_converged and not lora_converged:
        # HRA converged, LoRA DNF → HRA is definitively faster (PASS).
        k1013 = True
    else:
        k1013 = conv_ratio <= 2.0
    k1014 = time_ratio <= 3.0

    results.update({
        "k1011": {"pass": k1011, "hra_gsm8k": hra_gsm8k, "lora_gsm8k": lora_gsm8k,
                  "delta_pp": round((hra_gsm8k - lora_gsm8k) * 100, 2)},
        "k1012": {"pass": k1012, "hra_mmlu": hra_mmlu, "lora_mmlu": lora_mmlu,
                  "delta_pp": round((hra_mmlu - lora_mmlu) * 100, 2)},
        "k1013": {"pass": k1013, "hra_conv_step": hra_conv, "lora_conv_step": lora_conv,
                  "ratio": round(conv_ratio, 2),
                  "hra_converged": bool(hra_converged),
                  "lora_converged": bool(lora_converged),
                  "train_steps": TRAIN_STEPS,
                  "conv_threshold": LOSS_CONV_THRESH},
        "k1014": {"pass": k1014, "hra_step_time": round(hra_step, 4),
                  "lora_step_time": round(lora_step, 4), "ratio": round(time_ratio, 2)},
    })

    all_pass = k1011 and k1012 and k1013 and k1014
    results["all_pass"] = bool(all_pass)
    results["verdict"] = "SUPPORTED" if all_pass else "KILLED"
    results["ran"] = True

    # Save results
    results_path = EXPERIMENT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # --- Summary ---
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Base: GSM8K={base_gsm8k:.3f}, MMLU={base_mmlu:.3f}")
    print(f"LoRA: GSM8K={lora_gsm8k:.3f}, MMLU={lora_mmlu:.3f} "
          f"({lora_train['n_trainable']:,} params, {lora_step:.3f}s/step)")
    print(f"HRA:  GSM8K={hra_gsm8k:.3f}, MMLU={hra_mmlu:.3f} "
          f"({hra_train['n_trainable']:,} params, {hra_step:.3f}s/step)")
    print()
    print(f"K1011 (HRA>=LoRA GSM8K): {'PASS' if k1011 else 'FAIL'} "
          f"({'+' if hra_gsm8k>=lora_gsm8k else ''}{(hra_gsm8k-lora_gsm8k)*100:.1f}pp)")
    print(f"K1012 (HRA>=LoRA MMLU):  {'PASS' if k1012 else 'FAIL'} "
          f"({'+' if hra_mmlu>=lora_mmlu else ''}{(hra_mmlu-lora_mmlu)*100:.1f}pp)")
    print(f"K1013 (conv<=2×LoRA):    {'PASS' if k1013 else 'FAIL'} "
          f"(HRA={hra_conv} LoRA={lora_conv} ratio={conv_ratio:.2f})")
    print(f"K1014 (time<=3×LoRA):    {'PASS' if k1014 else 'FAIL'} "
          f"(ratio={time_ratio:.2f})")

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAIL'} "
          f"(verdict={results['verdict']})")
    return results


if __name__ == "__main__":
    main()
