#!/usr/bin/env python3
"""
Composition-method ablation: uniform 1/N vs hard top-1 vs M2P-gated continuous mixing.

Same 7 PoLAR adapters, same eval slice, head-to-head comparison. Determines Pierre v1
composition mechanism empirically.

Methods tested:
  M1 — uniform 1/N composition: ΔW = Σ ΔW_i / N
  M2 — hard top-1 routing: pick single best adapter per benchmark domain (oracle)
  M3 — M2P-gated continuous: trained MLP gate → softmax weights → weighted composition

All applied via _FusedDeltaLinear module replacement (Finding #828).

Kill criteria:
  K2121: M2P-gated > both uniform-1/N AND hard top-1 on average accuracy
  K2122: All three methods within 1.5× latency of best
  K2123: M2P-gated Spearman ρ(confidence, correctness) ≥ 0.3
  K2124: Failure-mode characterization (diagnostic only)
"""

import gc
import json
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
from scipy import stats as scipy_stats

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tooling.scripts.polar_train import (
    inject_polar_adapters, cleanup,
    eval_gsm8k, eval_humaneval, eval_medqa,
    PoLARLinear, RANK, SCALE,
)

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42
N_BENCH_EVAL = 5 if IS_SMOKE else 30

GATE_HIDDEN_DIM = 256
GATE_N_OUT = 7
GATE_TRAIN_STEPS = 30 if IS_SMOKE else 1500
GATE_LR = 1e-3
GATE_BATCH = 32
SPARSITY_WEIGHT = 0.10
BUFFER = 0.05
TEMPERATURE = 1.0

ADAPTER_DIR = REPO_ROOT / "data" / "adapters"
ADAPTER_SLOTS = [
    ("strategy_full",      ADAPTER_DIR / "strategy_full_polar"      / "polar.safetensors"),
    ("strategy_prepare",   ADAPTER_DIR / "strategy_prepare_polar"   / "polar.safetensors"),
    ("strategy_act",       ADAPTER_DIR / "strategy_act_polar"       / "polar.safetensors"),
    ("strategy_integrate", ADAPTER_DIR / "strategy_integrate_polar" / "polar.safetensors"),
    ("domain_math",        ADAPTER_DIR / "math_polar"               / "polar.safetensors"),
    ("domain_code",        ADAPTER_DIR / "code_polar"               / "polar.safetensors"),
    ("domain_medical",     ADAPTER_DIR / "medical_polar"            / "polar.safetensors"),
]
SLOT_NAMES = [s[0] for s in ADAPTER_SLOTS]
N_ADAPTERS = len(ADAPTER_SLOTS)


def log(m): print(m, flush=True)


def log_memory(label=""):
    a = mx.get_active_memory() / 1e9; c = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB")


# ─────────────────────────────────────────────
# Adapter I/O
# ─────────────────────────────────────────────

def load_state(path: Path) -> list[dict]:
    raw = mx.load(str(path))
    n_layers = max(int(k.split(".")[0].split("_")[1]) for k in raw if k.startswith("layer_")) + 1
    return [{
        "a": np.array(raw[f"layer_{i}.lora_a"].tolist(), dtype=np.float32),
        "b": np.array(raw[f"layer_{i}.lora_b"].tolist(), dtype=np.float32),
    } for i in range(n_layers)]


# ─────────────────────────────────────────────
# FusedDeltaLinear — correct composition (Finding #828)
# ─────────────────────────────────────────────

class _FusedDeltaLinear(nn.Module):
    """y = base_linear(x) + x @ fused_delta. Proper nn.Module, no __call__ override."""
    def __init__(self, base_layer, fused):
        super().__init__()
        self.base = base_layer
        self._fused = fused
    def __call__(self, x):
        return self.base(x) + (x @ self._fused)


def apply_fused_composition(model, all_states, weights):
    """Replace q_proj with _FusedDeltaLinear using weighted sum of task vectors.
    weights: list of N floats. ΔW = Σ w_i * scale * (A_i @ B_i)."""
    layers_iter = (model.language_model.model.layers if hasattr(model, "language_model")
                   else model.model.language_model.layers)
    n_layers = len(all_states[0])
    for layer_idx, layer in enumerate(layers_iter):
        if layer_idx >= n_layers:
            break
        delta = None
        for w, state in zip(weights, all_states):
            if abs(w) < 1e-6:
                continue
            a = state[layer_idx]["a"]; b = state[layer_idx]["b"]
            d = (a @ b) * float(w) * SCALE
            delta = d if delta is None else delta + d
        if delta is None:
            delta = np.zeros((all_states[0][layer_idx]["a"].shape[0],
                              all_states[0][layer_idx]["b"].shape[1]), dtype=np.float32)
        delta_mx = mx.array(delta)
        mx.eval(delta_mx)
        q_proj = layer.self_attn.q_proj
        base_layer = q_proj.base if isinstance(q_proj, PoLARLinear) else q_proj
        layer.self_attn.q_proj = _FusedDeltaLinear(base_layer, delta_mx)


def apply_single_adapter(model, state):
    """Load single adapter weights directly into PoLAR modules."""
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
    for i, m in enumerate(modules):
        m.lora_a = mx.array(state[i]["a"]); m.lora_b = mx.array(state[i]["b"])
    mx.eval(model.parameters())
    return modules


# ─────────────────────────────────────────────
# M2P Gate (inline training, same as exp_pierre_m2p_gated_composition)
# ─────────────────────────────────────────────

class GateMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_out):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_out)
    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


def _get_embed_tokens(model):
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model.embed_tokens
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.embed_tokens
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    raise AttributeError("Cannot find embed_tokens")


def prompt_embedding(model, tokenizer, text):
    ids = mx.array(tokenizer.encode(text), dtype=mx.uint32)[None, :]
    emb = _get_embed_tokens(model)(ids)
    v = mx.mean(emb, axis=1).squeeze().astype(mx.float32)
    return np.array(v.tolist(), dtype=np.float32)


def gate_loss_fn(gate, X, y):
    logits = gate(X) / TEMPERATURE
    log_probs = nn.log_softmax(logits, axis=-1)
    ce = -mx.mean(mx.take_along_axis(log_probs, y[:, None], axis=-1).squeeze(-1))
    probs = mx.exp(log_probs)
    entropy = -mx.sum(probs * log_probs, axis=-1)
    return ce + SPARSITY_WEIGHT * mx.mean(entropy)


def train_gate(embeddings, labels, embed_dim):
    log(f"  [Gate train] {len(embeddings)} samples, {GATE_TRAIN_STEPS} steps")
    gate = GateMLP(embed_dim, GATE_HIDDEN_DIM, GATE_N_OUT)
    optimizer = optim.Adam(learning_rate=GATE_LR)
    rng = np.random.default_rng(SEED)
    X = mx.array(embeddings); y = mx.array(labels.astype(np.int32))
    grad_fn = nn.value_and_grad(gate, gate_loss_fn)
    n = len(embeddings)
    for step in range(GATE_TRAIN_STEPS):
        idx = rng.choice(n, size=min(GATE_BATCH, n), replace=False)
        X_b = X[mx.array(idx)]; y_b = y[mx.array(idx)]
        loss, grads = grad_fn(gate, X_b, y_b)
        optimizer.update(gate, grads)
        mx.eval(gate.parameters(), optimizer.state, loss)
        if step % max(1, GATE_TRAIN_STEPS // 5) == 0:
            log(f"    step {step}/{GATE_TRAIN_STEPS} loss={loss.item():.4f}")
    log(f"    final loss={loss.item():.4f}")
    return gate


def build_gate_corpus(model, tokenizer):
    from tooling.scripts.beehive_to_mlx import fetch_rows
    from datasets import load_dataset

    type_to_slot = {"full": "strategy_full", "prepare": "strategy_prepare",
                    "act": "strategy_act", "integrate": "strategy_integrate"}
    beehive = fetch_rows(quality="approved")
    pairs = [(r.user_prompt, type_to_slot[r.type]) for r in beehive]
    log(f"  beehive: {len(pairs)} samples")

    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=SEED).select(range(80))
    for ex in ds:
        pairs.append((f"Solve step by step.\n\n{ex['question']}\n\nAnswer:", "domain_math"))
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train").shuffle(seed=SEED).select(range(80))
    for ex in ds:
        prompt = ex["instruction"] + (f"\n\nInput:\n{ex['input']}" if ex.get("input") else "")
        pairs.append((f"Complete this Python function:\n\n```python\n{prompt}\n```", "domain_code"))
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train").shuffle(seed=SEED).select(range(80))
        for ex in ds:
            opts = ex["options"]
            q = f"{ex['question']}\n(A) {opts['A']}\n(B) {opts['B']}\n(C) {opts['C']}\n(D) {opts['D']}"
            pairs.append((f"Answer with only the letter (A/B/C/D).\n\n{q}", "domain_medical"))
    except Exception as e:
        log(f"  WARN: skipping medqa train: {e}")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in perm]
    cut = int(len(pairs) * 0.9)
    train_pairs, holdout_pairs = pairs[:cut], pairs[cut:]

    label_to_idx = {name: i for i, name in enumerate(SLOT_NAMES)}
    embs, labs = [], []
    for prompt, label in train_pairs:
        embs.append(prompt_embedding(model, tokenizer, prompt))
        labs.append(label_to_idx[label])

    holdout_prompts = [p for p, _ in holdout_pairs]
    holdout_labels = [label_to_idx[l] for _, l in holdout_pairs]
    return np.stack(embs), np.array(labs, dtype=np.int32), holdout_prompts, holdout_labels


# ─────────────────────────────────────────────
# Build eval tuples — common slice for all methods
# ─────────────────────────────────────────────

def build_eval_tuples():
    from datasets import load_dataset
    out = {}
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=SEED).select(range(N_BENCH_EVAL))
    out["gsm8k"] = [(f"Solve step by step.\n\n{ex['question']}\n\nAnswer:", ex["answer"]) for ex in ds]
    ds = load_dataset("openai_humaneval", split="test").select(range(N_BENCH_EVAL))
    out["humaneval"] = [(f"Complete this Python function:\n\n```python\n{ex['prompt']}\n```", ex) for ex in ds]
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test").shuffle(seed=SEED).select(range(N_BENCH_EVAL))
    medqa = []
    for ex in ds:
        opts = ex["options"]
        q = f"{ex['question']}\n(A) {opts['A']}\n(B) {opts['B']}\n(C) {opts['C']}\n(D) {opts['D']}"
        medqa.append((f"Answer with only the letter (A/B/C/D).\n\n{q}", ex))
    out["medqa"] = medqa
    return out


def score_response(response, gold, benchmark):
    if benchmark == "gsm8k":
        gt_match = re.search(r"####\s*([\d,\-\.]+)", gold)
        if not gt_match: return False
        gt = gt_match.group(1).replace(",", "").strip()
        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match and pred_match.group(1).replace(",", "").strip() == gt:
            return True
        nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
        return bool(nums) and nums[-1] == gt
    elif benchmark == "humaneval":
        ex = gold
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        completion = code_match.group(1) if code_match else response
        full_code = ex["prompt"] + completion + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"
        try:
            r = subprocess.run([sys.executable, "-c", full_code], timeout=10, capture_output=True, text=True)
            return r.returncode == 0
        except Exception:
            return False
    elif benchmark == "medqa":
        ex = gold
        pred = response.strip().upper()
        pred_letter = next((L for L in "ABCD" if pred.startswith(L)), None)
        if not pred_letter:
            m = re.search(r"\b([ABCD])\b", pred)
            pred_letter = m.group(1) if m else None
        return pred_letter == ex["answer_idx"]
    return False


# ─────────────────────────────────────────────
# Evaluate methods
# ─────────────────────────────────────────────

def generate_responses(model, tokenizer, pairs, benchmark):
    from mlx_lm import generate
    results = []
    latencies = []
    for i, (prompt, gold) in enumerate(pairs):
        msgs = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        max_t = 1024 if benchmark == "gsm8k" else (512 if benchmark == "humaneval" else 20)
        t0 = time.perf_counter()
        response = generate(model, tokenizer, prompt=formatted, max_tokens=max_t, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)
        ok = score_response(response, gold, benchmark)
        results.append(ok)
    return results, latencies


def evaluate_m1_uniform(all_states, eval_tuples):
    """M1: uniform 1/N composition via FusedDeltaLinear."""
    from mlx_lm import load
    log("\n[M1: Uniform 1/N]")
    BENCH = ["gsm8k", "humaneval", "medqa"]
    out = {}
    all_latencies = []
    weights = [1.0 / N_ADAPTERS] * N_ADAPTERS
    for benchmark in BENCH:
        model, tokenizer = load(MODEL_ID)
        mx.eval(model.parameters())
        apply_fused_composition(model, all_states, weights)
        mx.eval(model.parameters())
        results, latencies = generate_responses(model, tokenizer, eval_tuples[benchmark], benchmark)
        acc = round(sum(results) / len(results) * 100, 1)
        out[benchmark] = {"acc": acc, "per_prompt": results}
        all_latencies.extend(latencies)
        log(f"  {benchmark}: {acc}%")
        del model, tokenizer; gc.collect(); mx.clear_cache()
    out["_latency_ms"] = round(float(np.median(all_latencies)), 1)
    return out


def evaluate_m2_top1(all_states, eval_tuples):
    """M2: hard top-1 oracle routing — pick best single adapter per benchmark domain."""
    from mlx_lm import load
    log("\n[M2: Hard top-1 oracle routing]")
    BENCH = ["gsm8k", "humaneval", "medqa"]
    bench_to_adapter = {
        "gsm8k": SLOT_NAMES.index("domain_math"),
        "humaneval": SLOT_NAMES.index("domain_code"),
        "medqa": SLOT_NAMES.index("domain_medical"),
    }
    out = {}
    all_latencies = []
    for benchmark in BENCH:
        chosen = bench_to_adapter[benchmark]
        log(f"  {benchmark} → {SLOT_NAMES[chosen]}")
        model, tokenizer = load(MODEL_ID)
        mx.eval(model.parameters())
        modules = apply_single_adapter(model, all_states[chosen])
        results, latencies = generate_responses(model, tokenizer, eval_tuples[benchmark], benchmark)
        acc = round(sum(results) / len(results) * 100, 1)
        out[benchmark] = {"acc": acc, "per_prompt": results, "routed_to": SLOT_NAMES[chosen]}
        all_latencies.extend(latencies)
        log(f"    acc={acc}%")
        cleanup(model, tokenizer, modules)
    out["_latency_ms"] = round(float(np.median(all_latencies)), 1)
    return out


def evaluate_m3_gated(all_states, eval_tuples, gate, embed_model, embed_tokenizer):
    """M3: M2P-gated continuous composition via FusedDeltaLinear.
    Per-prompt gate weights → weighted composition → generate."""
    from mlx_lm import load, generate as mlx_generate
    log("\n[M3: M2P-gated continuous]")
    BENCH = ["gsm8k", "humaneval", "medqa"]
    out = {}
    all_latencies = []
    all_confidences = []
    all_correctness = []

    for benchmark in BENCH:
        pairs = eval_tuples[benchmark]
        log(f"  {benchmark} ({len(pairs)} prompts):")

        prompt_weights = []
        for prompt, _ in pairs:
            emb = prompt_embedding(embed_model, embed_tokenizer, prompt)
            logits = gate(mx.array(emb)[None, :]) / TEMPERATURE
            probs = nn.softmax(logits, axis=-1).squeeze()
            w = np.array(probs.tolist(), dtype=np.float64) * (1.0 + BUFFER)
            prompt_weights.append(w)

        results = []
        latencies = []
        for i, ((prompt, gold), w) in enumerate(zip(pairs, prompt_weights)):
            model, tokenizer = load(MODEL_ID)
            mx.eval(model.parameters())
            apply_fused_composition(model, all_states, w.tolist())
            mx.eval(model.parameters())

            msgs = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            max_t = 1024 if benchmark == "gsm8k" else (512 if benchmark == "humaneval" else 20)
            t0 = time.perf_counter()
            response = mlx_generate(model, tokenizer, prompt=formatted, max_tokens=max_t, verbose=False)
            latencies.append((time.perf_counter() - t0) * 1000)
            ok = score_response(response, gold, benchmark)
            results.append(ok)

            confidence = float(w.max() / w.sum())
            all_confidences.append(confidence)
            all_correctness.append(1.0 if ok else 0.0)

            del model, tokenizer; gc.collect(); mx.clear_cache()

        acc = round(sum(results) / len(results) * 100, 1)
        out[benchmark] = {"acc": acc, "per_prompt": results}
        all_latencies.extend(latencies)
        log(f"    acc={acc}%")

    out["_latency_ms"] = round(float(np.median(all_latencies)), 1)
    out["_confidences"] = all_confidences
    out["_correctness"] = all_correctness
    return out


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    t0 = time.time()
    log_memory("start")
    log(f"=== Composition Method Ablation (SMOKE={IS_SMOKE}) ===")

    log("\n[Phase 0] Load 7 PoLAR adapters")
    all_states = []
    for slot_name, path in ADAPTER_SLOTS:
        if not path.exists():
            log(f"  FATAL: missing {slot_name}: {path}"); sys.exit(1)
        all_states.append(load_state(path))
        log(f"  {slot_name}: {len(all_states[-1])} layers")

    log("\n[Phase 1] Build common eval tuples")
    eval_tuples = build_eval_tuples()
    for b, p in eval_tuples.items():
        log(f"  {b}: {len(p)} prompts")

    log("\n[Phase 2] Train M2P gate")
    from mlx_lm import load
    embed_model, embed_tokenizer = load(MODEL_ID)
    mx.eval(embed_model.parameters())
    _probe = _get_embed_tokens(embed_model)(mx.array([[1]], dtype=mx.uint32))
    embed_dim = int(_probe.shape[-1])
    log(f"  embed_dim={embed_dim}")
    train_emb, train_lab, holdout_prompts, holdout_labels = build_gate_corpus(embed_model, embed_tokenizer)
    gate = train_gate(train_emb, train_lab, embed_dim)

    correct = 0
    for prompt, gold_label in zip(holdout_prompts, holdout_labels):
        emb = prompt_embedding(embed_model, embed_tokenizer, prompt)
        logits = gate(mx.array(emb)[None, :])
        pred = int(mx.argmax(logits, axis=-1).item())
        if pred == gold_label: correct += 1
    holdout_acc = round(correct / max(len(holdout_prompts), 1) * 100, 1)
    log(f"  gate holdout accuracy: {holdout_acc}%")

    BENCH = ["gsm8k", "humaneval", "medqa"]

    log("\n[Phase 3] Evaluate M1 (uniform 1/N)")
    del embed_model, embed_tokenizer; gc.collect(); mx.clear_cache()
    m1 = evaluate_m1_uniform(all_states, eval_tuples)
    m1_avg = round(float(np.mean([m1[b]["acc"] for b in BENCH])), 1)

    log("\n[Phase 4] Evaluate M2 (hard top-1 oracle)")
    m2 = evaluate_m2_top1(all_states, eval_tuples)
    m2_avg = round(float(np.mean([m2[b]["acc"] for b in BENCH])), 1)

    log("\n[Phase 5] Evaluate M3 (M2P-gated continuous)")
    embed_model, embed_tokenizer = load(MODEL_ID)
    mx.eval(embed_model.parameters())
    m3 = evaluate_m3_gated(all_states, eval_tuples, gate, embed_model, embed_tokenizer)
    m3_avg = round(float(np.mean([m3[b]["acc"] for b in BENCH])), 1)
    del embed_model, embed_tokenizer; gc.collect(); mx.clear_cache()

    # ── KCs ────────────────────────────────────
    log("\n=== Kill Criteria ===")

    k2121 = m3_avg > m1_avg and m3_avg > m2_avg
    log(f"K2121 M2P-gated > uniform AND > top-1: {'PASS' if k2121 else 'FAIL'}")
    log(f"  m1_avg={m1_avg}, m2_avg={m2_avg}, m3_avg={m3_avg}")

    latencies = [m1["_latency_ms"], m2["_latency_ms"], m3["_latency_ms"]]
    best_lat = min(latencies)
    k2122 = all(L <= 1.5 * best_lat for L in latencies)
    log(f"K2122 latency within 1.5×: {'PASS' if k2122 else 'FAIL'}  latencies={latencies} best={best_lat}")

    confidences = m3.get("_confidences", [])
    correctness = m3.get("_correctness", [])
    if len(confidences) >= 5 and len(set(confidences)) > 1:
        rho, p_val = scipy_stats.spearmanr(confidences, correctness)
        calibration_rho = round(float(rho), 3) if not np.isnan(rho) else 0.0
    else:
        calibration_rho = 0.0
        p_val = 1.0
    k2123 = calibration_rho >= 0.3
    log(f"K2123 calibration ρ ≥ 0.3: {'PASS' if k2123 else 'FAIL'}  ρ={calibration_rho} p={p_val:.4f}")

    k2124_diag = {}
    for b in BENCH:
        m1_fails = set(i for i, ok in enumerate(m1[b]["per_prompt"]) if not ok)
        m2_fails = set(i for i, ok in enumerate(m2[b]["per_prompt"]) if not ok)
        m3_fails = set(i for i, ok in enumerate(m3[b]["per_prompt"]) if not ok)
        all_fail = m1_fails & m2_fails & m3_fails
        k2124_diag[b] = {
            "all_fail": len(all_fail), "m1_fail": len(m1_fails),
            "m2_fail": len(m2_fails), "m3_fail": len(m3_fails),
            "total": len(m1[b]["per_prompt"]),
        }
    log(f"K2124 failure diagnostic: {k2124_diag}")

    all_pass = k2121 and k2122 and k2123
    verdict = "PROVISIONAL" if IS_SMOKE else ("SUPPORTED" if all_pass else "KILLED")

    results = {
        "is_smoke": IS_SMOKE,
        "gate_holdout_accuracy": holdout_acc,
        "method_results": {
            "M1_uniform": {b: m1[b]["acc"] for b in BENCH} | {"avg": m1_avg, "latency_ms": m1["_latency_ms"]},
            "M2_top1": {b: m2[b]["acc"] for b in BENCH} | {"avg": m2_avg, "latency_ms": m2["_latency_ms"]},
            "M3_gated": {b: m3[b]["acc"] for b in BENCH} | {"avg": m3_avg, "latency_ms": m3["_latency_ms"]},
        },
        "kill_criteria": {
            "K2121_gated_beats_others": {"pass": k2121, "m1_avg": m1_avg, "m2_avg": m2_avg, "m3_avg": m3_avg},
            "K2122_latency_parity": {"pass": k2122, "latencies_ms": latencies, "best_ms": best_lat},
            "K2123_calibration": {"pass": k2123, "spearman_rho": calibration_rho, "p_value": round(float(p_val), 4)},
            "K2124_failure_diagnostic": {"pass": True, "per_bench": k2124_diag},
        },
        "verdict": verdict, "all_pass": all_pass,
        "total_time_s": round(time.time() - t0, 1),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    log(f"\nVERDICT: {verdict}")
    log(f"Total: {results['total_time_s']:.0f}s")


if __name__ == "__main__":
    main()
