#!/usr/bin/env python3
"""
Shared PoLAR training/eval primitives for beehive experiments.

Imports the validated PoLARLinear from F#442 and provides reusable phases:
  - inject_polar_adapters(model, rank, scale)
  - train(model, tokenizer, records, modules, n_steps, lr, ...)
  - eval_gsm8k / eval_humaneval / eval_medqa
  - eval_principle_following(model, tokenizer, valid_ids, n_eval)
  - composed_delta_load_apply (for N>1 composition tests)
"""
from __future__ import annotations

import gc
import math
import re
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tooling.scripts.beehive_to_mlx import fetch_eval_pairs, principle_following_score  # noqa: E402

SEED = 42
RANK = 6
SCALE = 6.0
RETRACT_EVERY = 20
LR = 1e-4
GRAD_CLIP = 1.0
BATCH_SIZE = 2
MAX_SEQ_LEN = 512


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ─────────────────────────────────────────────
# PoLAR module — F#442 verified
# ─────────────────────────────────────────────

class PoLARLinear(nn.Module):
    def __init__(self, base_linear: nn.Module, rank: int, scale: float, seed: int = SEED):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.scale = scale

        if hasattr(base_linear, "group_size"):
            d_out = base_linear.weight.shape[0]
            d_in = base_linear.scales.shape[1] * base_linear.group_size
        else:
            d_in = base_linear.weight.shape[1]
            d_out = base_linear.weight.shape[0]
        self.d_in, self.d_out = d_in, d_out

        rng = np.random.default_rng(seed)
        rand_mat = rng.standard_normal((d_in, rank)).astype(np.float32)
        Q, _ = np.linalg.qr(rand_mat)
        self.lora_a = mx.array(Q)
        self.lora_b = mx.zeros((rank, d_out))

    def __call__(self, x):
        return self.base(x) + self.scale * ((x @ self.lora_a) @ self.lora_b)

    def retract_to_stiefel(self) -> tuple[float, float]:
        I_r = np.eye(self.rank)
        A_np = np.array(self.lora_a.tolist(), dtype=np.float64)
        if not np.all(np.isfinite(A_np)) or np.sum(A_np ** 2) < 1e-12:
            dist_A = float(np.sqrt(np.sum((A_np.T @ A_np - I_r) ** 2))) if np.all(np.isfinite(A_np)) else float("inf")
        else:
            W, _, Vh = np.linalg.svd(A_np, full_matrices=False)
            A_ret = W @ Vh
            self.lora_a = mx.array(A_ret.astype(np.float32))
            dist_A = float(np.sqrt(np.sum((A_ret.T @ A_ret - I_r) ** 2)))

        B_np = np.array(self.lora_b.tolist(), dtype=np.float64)
        if not np.all(np.isfinite(B_np)) or np.sum(B_np ** 2) < 1e-12:
            dist_B = float(np.sqrt(self.rank))
        else:
            W2, _, Vh2 = np.linalg.svd(B_np, full_matrices=False)
            B_ret = W2 @ Vh2
            self.lora_b = mx.array(B_ret.astype(np.float32))
            dist_B = float(np.sqrt(np.sum((B_ret @ B_ret.T - I_r) ** 2)))
        return dist_A, dist_B


def _get_layers(model):
    """Navigate to transformer layers regardless of model wrapper structure."""
    # Gemma 4 via mlx_lm: model.language_model.model.layers
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        return model.language_model.model.layers
    # Fallback: model.model.language_model.layers (older structure)
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return model.model.language_model.layers
    # Direct: model.model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    raise AttributeError(f"Cannot find transformer layers in {type(model).__name__}")


def inject_polar_adapters(model, rank=RANK, scale=SCALE, seed=SEED):
    modules = []
    for layer in _get_layers(model):
        wrapped = PoLARLinear(layer.self_attn.q_proj, rank=rank, scale=scale, seed=seed)
        layer.self_attn.q_proj = wrapped
        modules.append(wrapped)
    return modules


def retract_all(modules) -> tuple[float, float]:
    max_A, max_B = 0.0, 0.0
    for m in modules:
        dA, dB = m.retract_to_stiefel()
        max_A, max_B = max(max_A, dA), max(max_B, dB)
    return max_A, max_B


# ─────────────────────────────────────────────
# Tokenization + collation
# ─────────────────────────────────────────────

def tokenize_record(tokenizer, rec: dict, max_len: int = MAX_SEQ_LEN):
    msgs = rec["messages"]
    prompt_only = tokenizer.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer.encode(prompt_only)
    full_ids = tokenizer.encode(full)
    if len(full_ids) > max_len:
        full_ids = full_ids[:max_len]
    labels = list(full_ids)
    n_prompt = min(len(prompt_ids), len(full_ids))
    for i in range(n_prompt):
        labels[i] = -100
    return mx.array(full_ids, dtype=mx.uint32), mx.array(labels, dtype=mx.int32)


def collate(batch, pad_id: int = 0):
    max_len = max(int(t[0].shape[0]) for t in batch)
    inputs, labels = [], []
    for ids, lab in batch:
        n = int(ids.shape[0])
        if n < max_len:
            ids = mx.concatenate([ids, mx.zeros(max_len - n, dtype=mx.uint32) + pad_id])
            lab = mx.concatenate([lab, mx.full((max_len - n,), -100, dtype=mx.int32)])
        inputs.append(ids)
        labels.append(lab)
    return mx.stack(inputs), mx.stack(labels)


def loss_fn(model, inputs, labels):
    logits = model(inputs)[:, :-1, :]
    targets = labels[:, 1:]
    log_probs = nn.log_softmax(logits.astype(mx.float32), axis=-1)
    # Clip -100 ignore-index to valid range; mask zeroes them out below.
    safe_targets = mx.clip(targets, 0, log_probs.shape[-1] - 1)
    target_lp = mx.take_along_axis(log_probs, safe_targets[:, :, None], axis=-1).squeeze(-1)
    mask = targets != -100
    return -(target_lp * mask).sum() / mx.maximum(mask.sum(), 1)


def _grad_clip(grads, max_norm: float):
    flat = []

    def walk(g):
        if isinstance(g, dict):
            for v in g.values():
                walk(v)
        elif isinstance(g, list):
            for v in g:
                walk(v)
        elif isinstance(g, mx.array):
            flat.append(g)

    walk(grads)
    if not flat:
        return grads
    total_norm = mx.sqrt(sum(mx.sum(g.astype(mx.float32) ** 2) for g in flat) + 1e-12)
    factor = mx.minimum(max_norm / total_norm, mx.array(1.0))

    def scale(g):
        if isinstance(g, dict):
            return {k: scale(v) for k, v in g.items()}
        if isinstance(g, list):
            return [scale(v) for v in g]
        if isinstance(g, mx.array):
            return g * factor
        return g

    return scale(grads)


def train(model, tokenizer, records, modules, n_steps, lr=LR, batch_size=BATCH_SIZE,
          retract_every=RETRACT_EVERY, grad_clip=GRAD_CLIP, log_every=50, seed=SEED,
          stop_at_loss: float | None = None, stop_window: int = 20):
    optimizer = optim.Adam(learning_rate=lr)
    rng = np.random.default_rng(seed)
    tokenized = [tokenize_record(tokenizer, r) for r in records]
    n_data = len(tokenized)
    grad_fn = nn.value_and_grad(model, loss_fn)
    losses = []
    monotonic_break = None

    for step in range(n_steps):
        idx = rng.choice(n_data, size=min(batch_size, n_data), replace=(n_data < batch_size))
        batch = [tokenized[i] for i in idx]
        inputs, labels = collate(batch)
        loss, grads = grad_fn(model, inputs, labels)
        grads = _grad_clip(grads, grad_clip)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        loss_val = float(loss.item())
        if not math.isfinite(loss_val):
            losses.append(loss_val)
            print(f"[step {step}] DIVERGED loss={loss_val}", flush=True)
            break
        losses.append(loss_val)

        if step >= 50 and monotonic_break is None:
            recent = sum(losses[-20:]) / 20
            older = sum(losses[-40:-20]) / 20
            if recent > older + 0.5:
                monotonic_break = step

        if (step + 1) % retract_every == 0:
            retract_all(modules)
            mx.eval(model.parameters())

        if step % log_every == 0:
            print(f"[step {step}/{n_steps}] loss={loss_val:.4f}", flush=True)

        # Early stop to prevent over-confidence (cause of N>1 composition collapse in exp_beehive_polar_strategy_n3)
        if stop_at_loss is not None and step >= stop_window:
            recent_avg = sum(losses[-stop_window:]) / stop_window
            if recent_avg <= stop_at_loss:
                print(f"[step {step}] early stop: recent_avg={recent_avg:.4f} ≤ {stop_at_loss}", flush=True)
                break

    max_A, max_B = retract_all(modules)
    mx.eval(model.parameters())
    return {
        "losses": losses,
        "first_loss": losses[0] if losses else float("nan"),
        "final_loss": losses[-1] if losses else float("nan"),
        "any_nan": any(not math.isfinite(l) for l in losses),
        "monotonic_break_step": monotonic_break,
        "stiefel_max_A": max_A,
        "stiefel_max_B": max_B,
    }


# ─────────────────────────────────────────────
# Composition (Σ B_i @ A_i with 1/N scaling)
# ─────────────────────────────────────────────

def extract_adapter_state(modules) -> list[dict]:
    """Snapshot lora_a, lora_b per layer for later composition."""
    return [{"a": mx.array(np.array(m.lora_a.tolist())), "b": mx.array(np.array(m.lora_b.tolist())), "scale": m.scale} for m in modules]


def compose_apply(model, modules, adapter_states_list: list[list[dict]]):
    """Replace each PoLAR's effective ΔW with sum_i (a_i @ b_i) * scale_i / N.

    Mutates in-place. modules must already be PoLARLinear-wrapped.
    """
    n = len(adapter_states_list)
    if n == 0:
        return
    for layer_idx, m in enumerate(modules):
        # Build composed a, b such that x @ a @ b == sum of contributions.
        # Easiest: zero current, then accumulate using a fresh stacked rank.
        # To keep rank fixed, we instead pre-compute Σ a_i b_i and inject a virtual delta.
        # Simpler route: replace (a,b) with random projection of stacked = block-diag — but
        # that doubles rank. Use explicit ΔW accumulation via small auxiliary delta tensor.
        delta = None
        for state in adapter_states_list:
            a = state[layer_idx]["a"]
            b = state[layer_idx]["b"]
            d = (a @ b)
            delta = d if delta is None else delta + d
        delta = delta / n
        # Stash composed delta on the module; override forward.
        m._composed_delta = delta
        mx.eval(m._composed_delta)

        # Replace forward to use composed_delta directly (bypass a@b)
        def make_fwd(layer):
            def fwd(x):
                return layer.base(x) + layer.scale * (x @ layer._composed_delta)
            return fwd
        m.__call__ = make_fwd(m).__get__(m)


def reset_compose(modules):
    """Restore standard PoLAR forward."""
    for m in modules:
        if hasattr(m, "_composed_delta"):
            del m._composed_delta
        m.__call__ = PoLARLinear.__call__.__get__(m)


# ─────────────────────────────────────────────
# Eval primitives
# ─────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, n_eval=30, seed=SEED) -> float:
    from datasets import load_dataset
    from mlx_lm import generate

    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed).select(range(min(n_eval, 1319)))
    correct = 0
    for ex in ds:
        msgs = [{"role": "user", "content": f"Solve step by step.\n\n{ex['question']}\n\nAnswer:"}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=formatted, max_tokens=1024, verbose=False)
        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt = gt_match.group(1).replace(",", "").strip()
        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match and pred_match.group(1).replace(",", "").strip() == gt:
            correct += 1
        else:
            nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
            if nums and nums[-1] == gt:
                correct += 1
    return correct / len(ds) * 100


def eval_humaneval(model, tokenizer, n_eval=30) -> float:
    from datasets import load_dataset
    from mlx_lm import generate

    ds = load_dataset("openai_humaneval", split="test").select(range(min(n_eval, 164)))
    passed = 0
    for ex in ds:
        msgs = [{"role": "user", "content": f"Complete this Python function:\n\n```python\n{ex['prompt']}\n```\n\nRespond with only the function body."}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=formatted, max_tokens=512, verbose=False)
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        completion = code_match.group(1) if code_match else response
        full_code = ex["prompt"] + completion + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"
        try:
            r = subprocess.run([sys.executable, "-c", full_code], timeout=10, capture_output=True, text=True)
            if r.returncode == 0:
                passed += 1
        except (subprocess.TimeoutExpired, Exception):
            pass
    return passed / len(ds) * 100


def eval_medqa(model, tokenizer, n_eval=30, seed=SEED) -> float:
    from datasets import load_dataset
    from mlx_lm import generate

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test").shuffle(seed=seed).select(range(min(n_eval, 1273)))
    correct = 0
    for ex in ds:
        opts = ex["options"]
        question = f"{ex['question']}\n(A) {opts['A']}\n(B) {opts['B']}\n(C) {opts['C']}\n(D) {opts['D']}"
        msgs = [{"role": "user", "content": f"Answer with only the letter (A/B/C/D).\n\n{question}"}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=formatted, max_tokens=20, verbose=False)
        gt = ex["answer_idx"]
        pred = response.strip().upper()
        pred_letter = next((L for L in "ABCD" if pred.startswith(L)), None)
        if not pred_letter:
            m = re.search(r"\b([ABCD])\b", pred)
            pred_letter = m.group(1) if m else None
        if pred_letter == gt:
            correct += 1
    return correct / len(ds) * 100


def eval_principle_following(model, tokenizer, valid_ids: list[int], n_eval: int) -> dict:
    from mlx_lm import generate

    pairs = fetch_eval_pairs(valid_ids)[:n_eval]
    if not pairs:
        return {"n": 0, "aggregate_mean": 0.0, "format_mean": 0.0, "keyword_mean": 0.0, "structure_mean": 0.0}

    scores = []
    for p in pairs:
        msgs = [{"role": "user", "content": p["prompt"]}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=formatted, max_tokens=600, verbose=False)
        scores.append(principle_following_score(response, p["expected"], p["principle"]))
    return {
        "n": len(scores),
        "aggregate_mean": float(np.mean([s["aggregate"] for s in scores]) * 100),
        "format_mean": float(np.mean([s["format"] for s in scores]) * 100),
        "keyword_mean": float(np.mean([s["keyword_recall"] for s in scores]) * 100),
        "structure_mean": float(np.mean([s["structure"] for s in scores]) * 100),
    }


def per_trajectory_nll(model, tokenizer, records: list[dict]) -> list[float]:
    """For score-vs-NLL correlation: per-record NLL on the assistant span only."""
    results = []
    for r in records:
        ids, labels = tokenize_record(tokenizer, r)
        ids = ids[None, :]
        labels = labels[None, :]
        l = float(loss_fn(model, ids, labels).item())
        results.append(l)
    return results
