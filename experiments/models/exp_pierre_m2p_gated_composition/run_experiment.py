#!/usr/bin/env python3
"""
M2P-gated continuous-weight composition over 7 PoLAR adapters.

Tests the proposed architecture: tiny MLP on prompt embedding → softmax weights →
weighted sum of 7 adapter deltas. Replaces uniform 1/N (which destructively interferes
when adapters are aggressive) with LEARNED PEAKED gating.

Compute argument: per-adapter delta is ~0.0002× base forward FLOPs at rank 6, so N
parallel adapter contributions ≈ single-inference cost. The bottleneck is shared base
forward, NOT the per-adapter compute.

Architecture:
  prompt → embed_tokens → mean-pool → 2-layer MLP → 7 logits → softmax × (1+buffer)
  composed_delta_per_layer = Σ w_i × (a_i @ b_i)  (weights from gate, adapters frozen)
  forward = base(x) + scale × x @ composed_delta

Gate trained as classification (prompt → best-matching adapter label):
  beehive prompts → labeled by trajectory_type (full / prepare / act / integrate)
  GSM8K prompts → labeled "math" (matches the math domain adapter slot)
  CodeAlpaca prompts → labeled "code"
  MedQA prompts → labeled "medical"
  Loss = cross-entropy + λ_sparse × entropy(softmax) + α × buffer-term

Kill criteria:
  K2116: Gated ≥ best single-adapter on each of GSM8K/HumanEval/MedQA
  K2117: Avg gate entropy ≤ 1.5 nats (peaked, no flat collapse; max is ln(7)≈1.95)
  K2118: For homogeneous-domain queries, top-1 gate weight ≥ 0.5
  K2119: P95 first-token latency ≤ 250ms
  K2120: Calibration — high-entropy bucket has lower accuracy than low-entropy bucket
"""

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tooling.scripts.beehive_to_mlx import fetch_rows
from tooling.scripts.polar_train import (
    inject_polar_adapters, cleanup,
    eval_gsm8k, eval_humaneval, eval_medqa,
    PoLARLinear, RANK, SCALE,
)

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42
N_BENCH_EVAL = 5 if IS_SMOKE else 30

# Gate config
GATE_HIDDEN_DIM = 256
GATE_N_OUT = 7              # 4 strategy + 3 domain
GATE_TRAIN_STEPS = 30 if IS_SMOKE else 1500
GATE_LR = 1e-3
GATE_BATCH = 32
SPARSITY_WEIGHT = 0.10      # entropy penalty coefficient
BUFFER = 0.05               # weights × (1+buffer) at inference (slight oversaturation)
TEMPERATURE = 1.0           # softmax temperature; lower = sharper

ADAPTER_DIR = REPO_ROOT / "data" / "adapters"
# Order matters — gate output index → adapter slot
ADAPTER_SLOTS = [
    ("strategy_full",      ADAPTER_DIR / "strategy_full_polar"   / "polar.safetensors"),
    ("strategy_prepare",   ADAPTER_DIR / "strategy_prepare_polar" / "polar.safetensors"),
    ("strategy_act",       ADAPTER_DIR / "strategy_act_polar"     / "polar.safetensors"),
    ("strategy_integrate", ADAPTER_DIR / "strategy_integrate_polar"/ "polar.safetensors"),
    ("domain_math",        ADAPTER_DIR / "math_polar"             / "polar.safetensors"),
    ("domain_code",        ADAPTER_DIR / "code_polar"             / "polar.safetensors"),
    ("domain_medical",     ADAPTER_DIR / "medical_polar"          / "polar.safetensors"),
]
SLOT_NAMES = [s[0] for s in ADAPTER_SLOTS]


def log(m): print(m, flush=True)


def log_memory(label=""):
    a = mx.get_active_memory() / 1e9
    c = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB")


# ─────────────────────────────────────────────
# Adapter state I/O
# ─────────────────────────────────────────────

def load_state(path: Path) -> list[dict]:
    raw = mx.load(str(path))
    n_layers = max(int(k.split(".")[0].split("_")[1]) for k in raw if k.startswith("layer_")) + 1
    return [{
        "a": np.array(raw[f"layer_{i}.lora_a"].tolist(), dtype=np.float32),
        "b": np.array(raw[f"layer_{i}.lora_b"].tolist(), dtype=np.float32),
    } for i in range(n_layers)]


def stiefel_distance(state: list[dict]) -> tuple[float, float]:
    max_A, max_B = 0.0, 0.0
    for s in state:
        a, b = s["a"], s["b"]
        I = np.eye(a.shape[1])
        max_A = max(max_A, float(np.sqrt(np.sum((a.T @ a - I) ** 2))))
        max_B = max(max_B, float(np.sqrt(np.sum((b @ b.T - I) ** 2))))
    return max_A, max_B


# ─────────────────────────────────────────────
# Embedding extraction (robust to mlx_lm wrapper structure)
# ─────────────────────────────────────────────

def _get_embed_tokens(model):
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model.embed_tokens
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.embed_tokens
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    raise AttributeError("Cannot find embed_tokens")


def prompt_embedding(model, tokenizer, text: str) -> np.ndarray:
    """Mean-pooled token embedding as fingerprint for the gate input."""
    ids = mx.array(tokenizer.encode(text), dtype=mx.uint32)[None, :]
    emb_layer = _get_embed_tokens(model)
    emb = emb_layer(ids)
    v = mx.mean(emb, axis=1).squeeze().astype(mx.float32)
    return np.array(v.tolist(), dtype=np.float32)


# ─────────────────────────────────────────────
# Gate MLP (trained on cross-entropy with sparsity reg)
# ─────────────────────────────────────────────

class GateMLP(nn.Module):
    """2-layer MLP: prompt embedding → 7 logits."""
    def __init__(self, embed_dim: int, hidden_dim: int, n_out: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_out)

    def __call__(self, x):
        h = nn.gelu(self.fc1(x))
        return self.fc2(h)


def gate_loss_fn(model: GateMLP, X, y):
    """CE loss + entropy penalty (encourage peaked outputs)."""
    logits = model(X) / TEMPERATURE
    log_probs = nn.log_softmax(logits, axis=-1)
    ce = -mx.mean(mx.take_along_axis(log_probs, y[:, None], axis=-1).squeeze(-1))
    probs = mx.exp(log_probs)
    entropy = -mx.sum(probs * log_probs, axis=-1)  # per-sample
    sparsity_loss = mx.mean(entropy)
    return ce + SPARSITY_WEIGHT * sparsity_loss


def train_gate(embeddings: np.ndarray, labels: np.ndarray, embed_dim: int) -> GateMLP:
    """Train a small MLP gate. Inputs are pre-computed embeddings."""
    log(f"\n[Gate train] {len(embeddings)} samples, {GATE_N_OUT} classes, {GATE_TRAIN_STEPS} steps")
    gate = GateMLP(embed_dim, GATE_HIDDEN_DIM, GATE_N_OUT)
    optimizer = optim.Adam(learning_rate=GATE_LR)
    rng = np.random.default_rng(SEED)

    X = mx.array(embeddings)
    y = mx.array(labels.astype(np.int32))
    grad_fn = nn.value_and_grad(gate, gate_loss_fn)

    losses = []
    n = len(embeddings)
    for step in range(GATE_TRAIN_STEPS):
        idx = rng.choice(n, size=min(GATE_BATCH, n), replace=False)
        X_batch = X[mx.array(idx)]
        y_batch = y[mx.array(idx)]
        loss, grads = grad_fn(gate, X_batch, y_batch)
        optimizer.update(gate, grads)
        mx.eval(gate.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))
        if step % max(1, GATE_TRAIN_STEPS // 10) == 0:
            log(f"  [step {step}/{GATE_TRAIN_STEPS}] loss={loss.item():.4f}")

    log(f"  final loss: {losses[-1]:.4f}")
    return gate


# ─────────────────────────────────────────────
# Build training corpus for the gate
# ─────────────────────────────────────────────

def build_gate_training_data(model, tokenizer) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Mixed corpus: beehive (strategy labels) + benchmarks (domain labels).

    Returns: (embeddings, label_indices, holdout_prompts, holdout_label_indices)
    """
    log("\n[Build gate corpus]")

    # Beehive — label by trajectory type
    type_to_slot = {"full": "strategy_full", "prepare": "strategy_prepare",
                    "act": "strategy_act", "integrate": "strategy_integrate"}
    beehive = fetch_rows(quality="approved")
    beehive_pairs = [(r.user_prompt, type_to_slot[r.type]) for r in beehive]
    log(f"  beehive: {len(beehive_pairs)} samples")

    # Benchmark TRAIN splits — label by domain
    from datasets import load_dataset
    bench_pairs = []

    # GSM8K train (test is held out)
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=SEED).select(range(80))
    for ex in ds:
        bench_pairs.append((f"Solve step by step.\n\n{ex['question']}\n\nAnswer:", "domain_math"))

    # CodeAlpaca
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train").shuffle(seed=SEED).select(range(80))
    for ex in ds:
        prompt = ex["instruction"] + (f"\n\nInput:\n{ex['input']}" if ex.get("input") else "")
        bench_pairs.append((f"Complete this Python function:\n\n```python\n{prompt}\n```", "domain_code"))

    # MedQA train
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train").shuffle(seed=SEED).select(range(80))
        for ex in ds:
            opts = ex["options"]
            q = f"{ex['question']}\n(A) {opts['A']}\n(B) {opts['B']}\n(C) {opts['C']}\n(D) {opts['D']}"
            bench_pairs.append((f"Answer with only the letter (A/B/C/D).\n\n{q}", "domain_medical"))
    except Exception as e:
        log(f"  WARN: skipping medqa: {e}")

    log(f"  benchmark: {len(bench_pairs)} samples")

    all_pairs = beehive_pairs + bench_pairs
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(all_pairs))
    all_pairs = [all_pairs[i] for i in perm]

    # 90/10 holdout
    cut = int(len(all_pairs) * 0.9)
    train_pairs, holdout_pairs = all_pairs[:cut], all_pairs[cut:]

    log(f"  train: {len(train_pairs)}, holdout: {len(holdout_pairs)}")

    # Compute embeddings for training
    log("  computing prompt embeddings...")
    label_to_idx = {name: i for i, name in enumerate(SLOT_NAMES)}
    train_emb, train_lab = [], []
    for prompt, label in train_pairs:
        train_emb.append(prompt_embedding(model, tokenizer, prompt))
        train_lab.append(label_to_idx[label])
    holdout_prompts = [p for p, _ in holdout_pairs]
    holdout_labels = [label_to_idx[l] for _, l in holdout_pairs]

    return (np.stack(train_emb), np.array(train_lab, dtype=np.int32),
            holdout_prompts, holdout_labels)


# ─────────────────────────────────────────────
# Composition with gate weights
# ─────────────────────────────────────────────

def gate_predict_weights(gate: GateMLP, model, tokenizer, prompt: str) -> np.ndarray:
    """Predict softmax weights × (1+buffer) for a prompt."""
    emb = prompt_embedding(model, tokenizer, prompt)
    x = mx.array(emb)[None, :]
    logits = gate(x) / TEMPERATURE
    probs = nn.softmax(logits, axis=-1).squeeze()
    p = np.array(probs.tolist(), dtype=np.float64)
    return p * (1.0 + BUFFER)


def apply_gated_composition(modules, all_states: list[list[dict]], weights: np.ndarray):
    """ΔW_combined = Σ w_i × (a_i @ b_i)  per layer."""
    for layer_idx, m in enumerate(modules):
        delta = None
        for w_i, state in zip(weights, all_states):
            if w_i < 1e-4:
                continue
            a = mx.array(state[layer_idx]["a"])
            b = mx.array(state[layer_idx]["b"])
            d = (a @ b) * float(w_i)
            delta = d if delta is None else delta + d
        if delta is None:
            delta = mx.zeros((all_states[0][layer_idx]["a"].shape[0], all_states[0][layer_idx]["b"].shape[1]))
        mx.eval(delta)
        m._composed_delta = delta

        def make_fwd(layer):
            def fwd(x):
                return layer.base(x) + layer.scale * (x @ layer._composed_delta)
            return fwd
        m.__call__ = make_fwd(m).__get__(m)


# ─────────────────────────────────────────────
# Eval helpers (per-prompt gating; bucket by gate selection)
# ─────────────────────────────────────────────

def evaluate_gated(gate: GateMLP, all_states: list[list[dict]], benchmark: str) -> dict:
    """For each benchmark prompt: compute gate weights → compose → generate → score.

    Buckets prompts by top-2 gate selection to reuse model loads.
    """
    from datasets import load_dataset
    from mlx_lm import load, generate

    if benchmark == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=SEED).select(range(N_BENCH_EVAL))
        pairs = [(f"Solve step by step.\n\n{ex['question']}\n\nAnswer:", ex["answer"]) for ex in ds]
    elif benchmark == "humaneval":
        ds = load_dataset("openai_humaneval", split="test").select(range(N_BENCH_EVAL))
        pairs = [(f"Complete this Python function:\n\n```python\n{ex['prompt']}\n```", ex) for ex in ds]
    elif benchmark == "medqa":
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test").shuffle(seed=SEED).select(range(N_BENCH_EVAL))
        pairs = []
        for ex in ds:
            opts = ex["options"]
            q = f"{ex['question']}\n(A) {opts['A']}\n(B) {opts['B']}\n(C) {opts['C']}\n(D) {opts['D']}"
            pairs.append((f"Answer with only the letter (A/B/C/D).\n\n{q}", ex))

    log(f"\n[Gated eval: {benchmark}]")

    # Phase 1: compute gate weights for every prompt (using base model only)
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    weights_per_prompt = []
    for prompt, _ in pairs:
        w = gate_predict_weights(gate, model, tokenizer, prompt)
        weights_per_prompt.append(w)
    cleanup(model, tokenizer)

    # Phase 2: bucket by top-2 indices for batched composition
    buckets = {}  # key=(top1_idx, top2_idx) sorted → list of (prompt_idx, prompt, gold, weights)
    for i, ((prompt, gold), w) in enumerate(zip(pairs, weights_per_prompt)):
        top2_idx = tuple(sorted(np.argsort(w)[-2:].tolist()))
        buckets.setdefault(top2_idx, []).append((i, prompt, gold, w))

    log(f"  bucketed into {len(buckets)} unique top-2 selections")

    results_by_idx = {}
    entropies = []
    top1_weights = []

    for bucket_key, items in buckets.items():
        avg_w = np.mean([item[3] for item in items], axis=0)
        # Track entropy + top1 for KCs
        for _, _, _, w in items:
            normalized = w / w.sum()
            entropies.append(float(-(normalized * np.log(normalized + 1e-12)).sum()))
            top1_weights.append(float(normalized.max()))

        log(f"  bucket {bucket_key} ({[SLOT_NAMES[i] for i in bucket_key]}): {len(items)} prompts, avg_w_top2={avg_w[list(bucket_key)].round(2).tolist()}")

        from mlx_lm import load as _load
        model, tokenizer = _load(MODEL_ID)
        mx.eval(model.parameters())
        modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
        apply_gated_composition(modules, all_states, avg_w)
        mx.eval(model.parameters())

        for prompt_idx, prompt, gold, _ in items:
            msgs = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            max_t = 1024 if benchmark == "gsm8k" else (512 if benchmark == "humaneval" else 20)
            response = generate(model, tokenizer, prompt=formatted, max_tokens=max_t, verbose=False)
            results_by_idx[prompt_idx] = (response, gold)

        cleanup(model, tokenizer, modules)

    # Score
    correct = 0
    correctness_per_prompt = []
    for i in range(len(pairs)):
        response, gold = results_by_idx[i]
        ok = False
        if benchmark == "gsm8k":
            gt_match = re.search(r"####\s*([\d,\-\.]+)", gold)
            if gt_match:
                gt = gt_match.group(1).replace(",", "").strip()
                pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
                ok = pred_match and pred_match.group(1).replace(",", "").strip() == gt
                if not ok:
                    nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
                    ok = bool(nums) and nums[-1] == gt
        elif benchmark == "humaneval":
            ex = gold
            code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
            completion = code_match.group(1) if code_match else response
            full_code = ex["prompt"] + completion + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"
            try:
                r = subprocess.run([sys.executable, "-c", full_code], timeout=10, capture_output=True, text=True)
                ok = r.returncode == 0
            except Exception:
                ok = False
        elif benchmark == "medqa":
            ex = gold
            pred = response.strip().upper()
            pred_letter = next((L for L in "ABCD" if pred.startswith(L)), None)
            if not pred_letter:
                m = re.search(r"\b([ABCD])\b", pred)
                pred_letter = m.group(1) if m else None
            ok = pred_letter == ex["answer_idx"]
        if ok:
            correct += 1
        correctness_per_prompt.append(ok)

    return {
        "accuracy": round(correct / len(pairs) * 100, 1) if pairs else 0.0,
        "n": len(pairs),
        "n_buckets": len(buckets),
        "avg_entropy": round(float(np.mean(entropies)), 3),
        "median_entropy": round(float(np.median(entropies)), 3),
        "avg_top1_weight": round(float(np.mean(top1_weights)), 3),
        "entropies": entropies,
        "top1_weights": top1_weights,
        "correctness_per_prompt": correctness_per_prompt,
    }


def evaluate_single_state(state: list[dict]) -> dict:
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
    for i, m in enumerate(modules):
        m.lora_a = mx.array(state[i]["a"]); m.lora_b = mx.array(state[i]["b"])
    mx.eval(model.parameters())
    g = eval_gsm8k(model, tokenizer, N_BENCH_EVAL)
    h = eval_humaneval(model, tokenizer, N_BENCH_EVAL)
    d = eval_medqa(model, tokenizer, N_BENCH_EVAL)
    cleanup(model, tokenizer, modules)
    return {"gsm8k": round(g, 1), "humaneval": round(h, 1), "medqa": round(d, 1)}


def measure_p95_first_token_latency(gate: GateMLP, all_states: list[list[dict]], n_trials: int = 12) -> dict:
    from mlx_lm import load, generate

    prompts = [
        "What is 5 plus 7?",
        "Write a Python function that returns the sum of a list.",
        "What's the typical dosage of acetaminophen?",
        "Explain inheritance in OOP.",
        "What is photosynthesis?",
        "Sort [3, 1, 4, 1, 5].",
    ]

    samples = []
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)

    for trial in range(n_trials):
        prompt = prompts[trial % len(prompts)]
        # 1. Gate prediction (very cheap)
        # 2. Composition application
        # 3. First-token generation
        t_start = time.perf_counter()
        w = gate_predict_weights(gate, model, tokenizer, prompt)
        apply_gated_composition(modules, all_states, w)
        mx.eval(model.parameters())
        msgs = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        _ = generate(model, tokenizer, prompt=formatted, max_tokens=1, verbose=False)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        samples.append(elapsed_ms)

    cleanup(model, tokenizer, modules)
    s = sorted(samples)
    return {
        "p50_ms": round(s[len(s) // 2], 1),
        "p95_ms": round(s[int(len(s) * 0.95)], 1),
        "samples_ms": [round(x, 1) for x in samples],
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    t0 = time.time()
    log_memory("start")
    log(f"=== M2P-Gated Composition (SMOKE={IS_SMOKE}) ===")

    # Phase 0: Load 7 adapters
    log("\n[Phase 0] Load 7 PoLAR adapters")
    all_states = []
    for slot_name, path in ADAPTER_SLOTS:
        if not path.exists():
            log(f"  FATAL: missing {slot_name}: {path}")
            sys.exit(1)
        state = load_state(path)
        all_states.append(state)
        dA, dB = stiefel_distance(state)
        log(f"  {slot_name}: {len(state)} layers, Stiefel A={dA:.2e} B={dB:.2e}")

    # Phase 1: Per-adapter single eval (baseline numbers)
    log("\n[Phase 1] Per-single-adapter baseline")
    per_adapter = {}
    for (slot_name, _), state in zip(ADAPTER_SLOTS, all_states):
        per_adapter[slot_name] = evaluate_single_state(state)
        log(f"  {slot_name}: {per_adapter[slot_name]}")
    BENCH = ["gsm8k", "humaneval", "medqa"]
    best_single = {b: max(per_adapter[s][b] for s in per_adapter) for b in BENCH}
    log(f"  best_single: {best_single}")

    # Phase 2: Build gate training corpus + train gate
    log("\n[Phase 2] Build training corpus + train gate")
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())

    # Probe embed dim
    _probe = _get_embed_tokens(model)(mx.array([[1]], dtype=mx.uint32))
    embed_dim = int(_probe.shape[-1])
    log(f"  embed_dim={embed_dim}")

    train_emb, train_lab, holdout_prompts, holdout_labels = build_gate_training_data(model, tokenizer)
    cleanup(model, tokenizer)

    gate = train_gate(train_emb, train_lab, embed_dim)

    # Phase 3: Evaluate gate on held-out
    log("\n[Phase 3] Gate held-out classification accuracy")
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    correct = 0
    for prompt, gold_label in zip(holdout_prompts, holdout_labels):
        emb = prompt_embedding(model, tokenizer, prompt)
        logits = gate(mx.array(emb)[None, :])
        pred = int(mx.argmax(logits, axis=-1).item())
        if pred == gold_label:
            correct += 1
    holdout_acc = correct / max(len(holdout_prompts), 1) * 100
    log(f"  holdout classification accuracy: {holdout_acc:.1f}% ({correct}/{len(holdout_prompts)})")
    cleanup(model, tokenizer)

    # Phase 4: Gated composition eval on benchmarks
    log("\n[Phase 4] Gated composition eval on benchmarks")
    gated_results = {}
    for b in BENCH:
        gated_results[b] = evaluate_gated(gate, all_states, b)

    # Phase 5: Latency
    log("\n[Phase 5] First-token latency under gated composition")
    latency = measure_p95_first_token_latency(gate, all_states, n_trials=12)
    log(f"  latency p50={latency['p50_ms']}ms p95={latency['p95_ms']}ms")

    # ── KCs ────────────────────────────────────
    log("\n=== Kill Criteria ===")

    # K2116: gated ≥ best single on each benchmark
    drops = {b: round(best_single[b] - gated_results[b]["accuracy"], 1) for b in BENCH}
    k2116 = all(d <= 0.0 for d in drops.values())  # gated >= best_single

    # K2117: avg gate entropy ≤ 1.5
    avg_entropy = float(np.mean([gated_results[b]["avg_entropy"] for b in BENCH]))
    k2117 = avg_entropy <= 1.5

    # K2118: top-1 weight ≥ 0.5 on average across benchmark prompts
    avg_top1 = float(np.mean([gated_results[b]["avg_top1_weight"] for b in BENCH]))
    k2118 = avg_top1 >= 0.5

    # K2119: P95 latency ≤ 250ms
    k2119 = latency["p95_ms"] <= 250.0

    # K2120: calibration — split prompts by gate entropy median, check accuracy gap
    cal_per_bench = {}
    for b in BENCH:
        ents = gated_results[b]["entropies"]
        corrects = gated_results[b]["correctness_per_prompt"]
        if not ents:
            continue
        med = float(np.median(ents))
        low_acc = np.mean([c for c, e in zip(corrects, ents) if e <= med]) * 100 if any(e <= med for e in ents) else 0
        high_acc = np.mean([c for c, e in zip(corrects, ents) if e > med]) * 100 if any(e > med for e in ents) else 0
        cal_per_bench[b] = {"low_entropy_acc": round(low_acc, 1), "high_entropy_acc": round(high_acc, 1),
                            "median_entropy": med, "delta_pp": round(low_acc - high_acc, 1)}
    k2120 = all(v.get("delta_pp", 0) >= 3.0 for v in cal_per_bench.values()) if cal_per_bench else False

    all_pass = k2116 and k2117 and k2118 and k2119 and k2120
    verdict = "PROVISIONAL" if IS_SMOKE else ("SUPPORTED" if all_pass else "KILLED")

    results = {
        "is_smoke": IS_SMOKE,
        "config": {
            "gate_hidden_dim": GATE_HIDDEN_DIM,
            "n_adapters": GATE_N_OUT,
            "sparsity_weight": SPARSITY_WEIGHT,
            "buffer": BUFFER,
            "temperature": TEMPERATURE,
        },
        "adapters": SLOT_NAMES,
        "per_single_adapter": per_adapter,
        "best_single_per_bench": best_single,
        "gate_holdout_accuracy": round(holdout_acc, 1),
        "gated_results": {b: {k: v for k, v in r.items() if k not in ("entropies", "top1_weights", "correctness_per_prompt")}
                          for b, r in gated_results.items()},
        "drops_vs_best_single": drops,
        "calibration": cal_per_bench,
        "latency": latency,
        "kill_criteria": {
            "K2116_gated_beats_best_single": {"pass": k2116, "drops_pp": drops},
            "K2117_avg_entropy_below_1_5": {"pass": k2117, "avg_entropy": round(avg_entropy, 3)},
            "K2118_top1_weight_above_0_5": {"pass": k2118, "avg_top1": round(avg_top1, 3)},
            "K2119_p95_latency_250ms": {"pass": k2119, "p95_ms": latency["p95_ms"]},
            "K2120_calibration": {"pass": k2120, "per_bench": cal_per_bench},
        },
        "verdict": verdict, "all_pass": all_pass,
        "total_time_s": round(time.time() - t0, 1),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    log(f"K2116 gated ≥ best single:    {'PASS' if k2116 else 'FAIL'}  drops={drops}")
    log(f"K2117 avg entropy ≤ 1.5:      {'PASS' if k2117 else 'FAIL'}  ({avg_entropy:.3f})")
    log(f"K2118 avg top-1 ≥ 0.5:        {'PASS' if k2118 else 'FAIL'}  ({avg_top1:.3f})")
    log(f"K2119 P95 latency ≤ 250ms:    {'PASS' if k2119 else 'FAIL'}  ({latency['p95_ms']}ms)")
    log(f"K2120 calibration ≥3pp:       {'PASS' if k2120 else 'FAIL'}  {cal_per_bench}")
    log(f"\nVERDICT: {verdict}")
    log(f"Total: {results['total_time_s']:.0f}s")


if __name__ == "__main__":
    main()
