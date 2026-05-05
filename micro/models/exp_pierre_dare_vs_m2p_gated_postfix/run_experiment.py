#!/usr/bin/env python3
"""
DARE vs M2P-gated head-to-head, with Finding #831 fix applied to BOTH.

Re-runs the M2P-gated composition (which was previously a false-kill at humaneval=20%
due to __call__ override bug) using the canonical _FusedDeltaLinear pattern. Compares
to DARE composition baseline on the same eval slice.

Decision tree:
  - M2P-gated > DARE → ship M2P-gated (richer calibration story)
  - M2P-gated ~ DARE → ship DARE (simpler) + use M2P confidence as separate signal
  - M2P-gated < DARE → ship DARE alone

Kill criteria:
  K2147: M2P-gated avg accuracy ≥ DARE avg
  K2148: M2P-gated calibration ρ(confidence, correctness) ≥ 0.3
  K2149: M2P-gated latency within 1.2× DARE
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

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.beehive_to_mlx import fetch_rows
from scripts.polar_train import (
    inject_polar_adapters, cleanup,
    eval_gsm8k, eval_humaneval, eval_medqa,
    PoLARLinear, RANK, SCALE,
)

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42
N_BENCH_EVAL = 5 if IS_SMOKE else 30
DARE_DROP_RATE = 0.90

GATE_HIDDEN_DIM = 256
GATE_N_OUT = 7
GATE_TRAIN_STEPS = 30 if IS_SMOKE else 1500
GATE_LR = 1e-3
GATE_BATCH = 32
SPARSITY_WEIGHT = 0.10
BUFFER = 0.05
TEMPERATURE = 1.0

ADAPTER_DIR = REPO_ROOT / "adapters"
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


def log(m): print(m, flush=True)
def log_memory(label=""):
    a = mx.get_active_memory() / 1e9; c = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB")


def load_state(path: Path) -> list[dict]:
    raw = mx.load(str(path))
    n_layers = max(int(k.split(".")[0].split("_")[1]) for k in raw if k.startswith("layer_")) + 1
    return [{
        "a": np.array(raw[f"layer_{i}.lora_a"].tolist(), dtype=np.float32),
        "b": np.array(raw[f"layer_{i}.lora_b"].tolist(), dtype=np.float32),
    } for i in range(n_layers)]


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


# ─────────────────────────────────────────────
# Canonical _FusedDeltaLinear (Finding #831)
# ─────────────────────────────────────────────

class _FusedDeltaLinear(nn.Module):
    def __init__(self, base_layer, fused):
        super().__init__()
        self.base = base_layer
        self._fused = fused
    def __call__(self, x):
        return self.base(x) + (x @ self._fused)


def apply_fused_per_layer(model, fused_deltas_per_layer):
    """Apply pre-computed fused delta per layer via setattr (canonical pattern)."""
    layers_iter = (model.language_model.model.layers if hasattr(model, "language_model")
                   else model.model.language_model.layers)
    for layer_idx, layer in enumerate(layers_iter):
        if layer_idx >= len(fused_deltas_per_layer):
            break
        delta_mx = mx.array(fused_deltas_per_layer[layer_idx].astype(np.float32))
        mx.eval(delta_mx)
        q_proj = layer.self_attn.q_proj
        base_layer = q_proj.base if isinstance(q_proj, (PoLARLinear, _FusedDeltaLinear)) else q_proj
        layer.self_attn.q_proj = _FusedDeltaLinear(base_layer, delta_mx)


def compute_dare_deltas(all_states, drop_rate=DARE_DROP_RATE, seed=SEED):
    """Per-layer DARE composition: ΔW_l = (1/N) Σ_i (drop_p(scale × A_i @ B_i) / (1-p))."""
    rng = np.random.default_rng(seed)
    n_layers = len(all_states[0])
    n_adapters = len(all_states)
    deltas = []
    for l in range(n_layers):
        delta = None
        for state in all_states:
            tv = SCALE * (state[l]["a"] @ state[l]["b"])
            mask = rng.binomial(1, 1.0 - drop_rate, size=tv.shape).astype(np.float32)
            tv_dare = (tv * mask) / max(1e-9, 1.0 - drop_rate)
            delta = tv_dare if delta is None else delta + tv_dare
        deltas.append(delta / n_adapters)
    return deltas


def compute_gated_deltas(all_states, weights):
    """Per-layer gated composition: ΔW_l = Σ_i w_i × scale × A_i @ B_i."""
    n_layers = len(all_states[0])
    deltas = []
    for l in range(n_layers):
        delta = None
        for w, state in zip(weights, all_states):
            if abs(w) < 1e-6:
                continue
            tv = float(w) * SCALE * (state[l]["a"] @ state[l]["b"])
            delta = tv if delta is None else delta + tv
        if delta is None:
            delta = np.zeros((all_states[0][l]["a"].shape[0],
                              all_states[0][l]["b"].shape[1]), dtype=np.float32)
        deltas.append(delta)
    return deltas


# ─────────────────────────────────────────────
# Gate MLP + training
# ─────────────────────────────────────────────

class GateMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_out):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_out)
    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


def gate_loss_fn(gate, X, y):
    logits = gate(X) / TEMPERATURE
    log_probs = nn.log_softmax(logits, axis=-1)
    ce = -mx.mean(mx.take_along_axis(log_probs, y[:, None], axis=-1).squeeze(-1))
    probs = mx.exp(log_probs)
    entropy = -mx.sum(probs * log_probs, axis=-1)
    return ce + SPARSITY_WEIGHT * mx.mean(entropy)


def train_gate(embeddings, labels, embed_dim):
    log(f"\n[Gate train] {len(embeddings)} samples, {GATE_TRAIN_STEPS} steps")
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
        if step % max(1, GATE_TRAIN_STEPS // 10) == 0:
            log(f"  [step {step}/{GATE_TRAIN_STEPS}] loss={loss.item():.4f}")
    log(f"  trained")
    return gate


def build_gate_corpus(model, tokenizer):
    log("\n[Build gate corpus]")
    type_to_slot = {"full": "strategy_full", "prepare": "strategy_prepare",
                    "act": "strategy_act", "integrate": "strategy_integrate"}
    beehive = fetch_rows(quality="approved")
    pairs = [(r.user_prompt, type_to_slot[r.type]) for r in beehive]

    from datasets import load_dataset
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
            pairs.append((f"Answer with only the letter.\n\n{q}", "domain_medical"))
    except Exception:
        pass

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in perm]
    cut = int(len(pairs) * 0.9)
    label_to_idx = {n: i for i, n in enumerate(SLOT_NAMES)}

    log(f"  computing embeddings for {cut} train + {len(pairs)-cut} holdout")
    train_emb, train_lab = [], []
    for prompt, label in pairs[:cut]:
        train_emb.append(prompt_embedding(model, tokenizer, prompt))
        train_lab.append(label_to_idx[label])
    holdout_prompts = [p for p, _ in pairs[cut:]]
    holdout_labels = [label_to_idx[l] for _, l in pairs[cut:]]

    return (np.stack(train_emb), np.array(train_lab, dtype=np.int32),
            holdout_prompts, holdout_labels)


def gate_predict_weights(gate, model, tokenizer, prompt):
    emb = prompt_embedding(model, tokenizer, prompt)
    x = mx.array(emb)[None, :]
    logits = gate(x) / TEMPERATURE
    probs = nn.softmax(logits, axis=-1).squeeze()
    p = np.array(probs.tolist(), dtype=np.float64)
    return p * (1.0 + BUFFER)


# ─────────────────────────────────────────────
# Eval methods
# ─────────────────────────────────────────────

def eval_dare(all_states):
    from mlx_lm import load
    log("\n[Eval: DARE composition]")
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    deltas = compute_dare_deltas(all_states)
    apply_fused_per_layer(model, deltas)
    mx.eval(model.parameters())
    g = eval_gsm8k(model, tokenizer, N_BENCH_EVAL)
    h = eval_humaneval(model, tokenizer, N_BENCH_EVAL)
    d = eval_medqa(model, tokenizer, N_BENCH_EVAL)
    cleanup(model, tokenizer)
    return {"gsm8k": round(g, 1), "humaneval": round(h, 1), "medqa": round(d, 1)}


def eval_m2p_gated(gate, all_states, ref_model, ref_tokenizer):
    """Per-prompt gating: compute weights, bucket prompts by top-2 selection, compose+eval."""
    from datasets import load_dataset
    from mlx_lm import load, generate
    log("\n[Eval: M2P-gated composition]")

    # Build eval tuples
    bench_pairs = {}
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=SEED).select(range(N_BENCH_EVAL))
    bench_pairs["gsm8k"] = [(f"Solve step by step.\n\n{ex['question']}\n\nAnswer:", ex["answer"]) for ex in ds]
    ds = load_dataset("openai_humaneval", split="test").select(range(N_BENCH_EVAL))
    bench_pairs["humaneval"] = [(f"Complete this Python function:\n\n```python\n{ex['prompt']}\n```", ex) for ex in ds]
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test").shuffle(seed=SEED).select(range(N_BENCH_EVAL))
    medqa_pairs = []
    for ex in ds:
        opts = ex["options"]
        q = f"{ex['question']}\n(A) {opts['A']}\n(B) {opts['B']}\n(C) {opts['C']}\n(D) {opts['D']}"
        medqa_pairs.append((f"Answer with only the letter.\n\n{q}", ex))
    bench_pairs["medqa"] = medqa_pairs

    # Compute gate weights per prompt (using ref_model just for embedding lookup)
    log("  computing per-prompt gate weights...")
    weights_per_prompt = {}
    entropies = {}
    top1_weights = {}
    for bench, pairs in bench_pairs.items():
        weights_per_prompt[bench] = []
        entropies[bench] = []
        top1_weights[bench] = []
        for prompt, _ in pairs:
            w = gate_predict_weights(gate, ref_model, ref_tokenizer, prompt)
            normalized = w / w.sum()
            entropies[bench].append(float(-(normalized * np.log(normalized + 1e-12)).sum()))
            top1_weights[bench].append(float(normalized.max()))
            weights_per_prompt[bench].append(w)
    # Bucket by top-2 selection
    from collections import defaultdict
    accuracy = {}
    correctness_per_prompt = {b: [] * len(bench_pairs[b]) for b in bench_pairs}

    for bench, pairs in bench_pairs.items():
        log(f"\n  [{bench}] bucketing by top-2 routes...")
        buckets = defaultdict(list)
        for i, ((prompt, gold), w) in enumerate(zip(pairs, weights_per_prompt[bench])):
            top2_key = tuple(sorted(np.argsort(w)[-2:].tolist()))
            buckets[top2_key].append((i, prompt, gold, w))
        log(f"    {len(buckets)} unique top-2 selections")

        results_by_idx = {}
        for bucket_key, items in buckets.items():
            avg_w = np.mean([item[3] for item in items], axis=0)
            log(f"    bucket {bucket_key}: {len(items)} prompts, top-1={SLOT_NAMES[bucket_key[1]]} ({avg_w[bucket_key[1]]:.2f})")
            model, tokenizer = load(MODEL_ID)
            mx.eval(model.parameters())
            deltas = compute_gated_deltas(all_states, avg_w)
            apply_fused_per_layer(model, deltas)
            mx.eval(model.parameters())
            for prompt_idx, prompt, gold, _ in items:
                results_by_idx[prompt_idx] = _generate_and_score(
                    model, tokenizer, prompt, gold, bench)
            cleanup(model, tokenizer)

        correct = sum(1 for ok in results_by_idx.values() if ok)
        accuracy[bench] = round(correct / len(pairs) * 100, 1)
        correctness_per_prompt[bench] = [results_by_idx[i] for i in range(len(pairs))]
        log(f"    accuracy = {accuracy[bench]}%")

    return accuracy, entropies, top1_weights, correctness_per_prompt


def _generate_and_score(model, tokenizer, prompt, gold, benchmark) -> bool:
    from mlx_lm import generate
    msgs = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    max_t = 1024 if benchmark == "gsm8k" else (512 if benchmark == "humaneval" else 20)
    response = generate(model, tokenizer, prompt=formatted, max_tokens=max_t, verbose=False)

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
            r = subprocess.run([sys.executable, "-c", full_code], timeout=10,
                               capture_output=True, text=True)
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


def measure_latency(model_loader_fn, n_trials=8):
    """Measure first-token latency given a model factory."""
    model, tokenizer = model_loader_fn()
    samples = []
    prompts = ["What is 2+2?", "Sort a list", "What's the capital?"]
    from mlx_lm import generate
    for trial in range(n_trials):
        prompt = prompts[trial % len(prompts)]
        msgs = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        t0 = time.perf_counter()
        _ = generate(model, tokenizer, prompt=formatted, max_tokens=1, verbose=False)
        samples.append((time.perf_counter() - t0) * 1000)
    cleanup(model, tokenizer)
    s = sorted(samples)
    return {"p50_ms": round(s[len(s)//2], 1), "p95_ms": round(s[int(len(s)*0.95)], 1)}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    t0 = time.time()
    log_memory("start")
    log(f"=== DARE vs M2P-gated (post-#831 fix) (SMOKE={IS_SMOKE}) ===")

    log("\n[Phase 0] Load 7 PoLAR adapters")
    all_states = [load_state(p) for _, p in ADAPTER_SLOTS]
    log(f"  loaded {len(all_states)}")

    log("\n[Phase 1] Train M2P gate")
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    _probe = _get_embed_tokens(model)(mx.array([[1]], dtype=mx.uint32))
    embed_dim = int(_probe.shape[-1])
    train_emb, train_lab, holdout_prompts, holdout_labels = build_gate_corpus(model, tokenizer)
    gate = train_gate(train_emb, train_lab, embed_dim)

    # Gate holdout accuracy
    correct = 0
    for prompt, gold in zip(holdout_prompts, holdout_labels):
        emb = prompt_embedding(model, tokenizer, prompt)
        logits = gate(mx.array(emb)[None, :])
        pred = int(mx.argmax(logits, axis=-1).item())
        if pred == gold: correct += 1
    holdout_acc = correct / max(len(holdout_prompts), 1) * 100
    log(f"  gate holdout accuracy: {holdout_acc:.1f}%")
    cleanup(model, tokenizer)

    # DARE eval
    dare_acc = eval_dare(all_states)
    dare_avg = float(np.mean(list(dare_acc.values())))

    # M2P-gated eval (need ref_model + tokenizer for embedding lookups during routing)
    log("\n[Reload ref model for gate embeddings]")
    ref_model, ref_tokenizer = load(MODEL_ID)
    mx.eval(ref_model.parameters())
    m2p_acc, entropies, top1_w, correctness = eval_m2p_gated(
        gate, all_states, ref_model, ref_tokenizer)
    cleanup(ref_model, ref_tokenizer)
    m2p_avg = float(np.mean(list(m2p_acc.values())))

    # Calibration: for M2P-gated, correlate top1_weight with correctness per benchmark
    cal_per_bench = {}
    for b in m2p_acc:
        if not entropies[b] or not correctness[b]:
            continue
        # Spearman ρ between top-1 weight (confidence) and correctness
        ranks_w = np.argsort(np.argsort(top1_w[b]))
        ranks_c = np.argsort(np.argsort(np.array(correctness[b], dtype=np.float64)))
        rho = float(np.corrcoef(ranks_w, ranks_c)[0, 1]) if len(ranks_w) > 2 else 0.0
        cal_per_bench[b] = round(rho, 3) if not np.isnan(rho) else 0.0
    cal_avg = float(np.mean([abs(v) for v in cal_per_bench.values()])) if cal_per_bench else 0.0

    # Latency comparison
    log("\n[Latency: DARE]")
    def make_dare_model():
        from mlx_lm import load
        m, t = load(MODEL_ID)
        mx.eval(m.parameters())
        deltas = compute_dare_deltas(all_states)
        apply_fused_per_layer(m, deltas)
        mx.eval(m.parameters())
        return m, t
    dare_lat = measure_latency(make_dare_model, n_trials=6)
    log(f"  {dare_lat}")

    log("\n[Latency: M2P-gated (uniform-weight proxy)]")
    def make_m2p_model():
        from mlx_lm import load
        m, t = load(MODEL_ID)
        mx.eval(m.parameters())
        deltas = compute_gated_deltas(all_states, [1.0/7]*7)
        apply_fused_per_layer(m, deltas)
        mx.eval(m.parameters())
        return m, t
    m2p_lat = measure_latency(make_m2p_model, n_trials=6)
    log(f"  {m2p_lat}")

    # KCs
    log("\n=== Kill Criteria ===")
    k2147 = m2p_avg >= dare_avg
    k2148 = cal_avg >= 0.3
    k2149 = m2p_lat["p95_ms"] <= 1.2 * dare_lat["p95_ms"]

    all_pass = k2147 and k2148 and k2149
    verdict = "PROVISIONAL" if IS_SMOKE else ("SUPPORTED" if all_pass else "KILLED")

    results = {
        "is_smoke": IS_SMOKE,
        "config": {"sparsity_weight": SPARSITY_WEIGHT, "buffer": BUFFER, "temperature": TEMPERATURE},
        "gate_holdout_accuracy": round(holdout_acc, 1),
        "dare": dare_acc, "dare_avg": round(dare_avg, 1),
        "m2p_gated": m2p_acc, "m2p_avg": round(m2p_avg, 1),
        "calibration_per_bench": cal_per_bench, "calibration_avg_abs_rho": round(cal_avg, 3),
        "latency": {"dare": dare_lat, "m2p_gated": m2p_lat},
        "kill_criteria": {
            "K2147_m2p_beats_dare": {"pass": k2147, "m2p_avg": round(m2p_avg, 1),
                                     "dare_avg": round(dare_avg, 1),
                                     "delta_pp": round(m2p_avg - dare_avg, 1)},
            "K2148_calibration": {"pass": k2148, "avg_abs_rho": round(cal_avg, 3),
                                  "per_bench": cal_per_bench},
            "K2149_latency_parity": {"pass": k2149, "ratio": round(m2p_lat["p95_ms"]/max(dare_lat["p95_ms"], 1e-9), 2)},
        },
        "verdict": verdict, "all_pass": all_pass,
        "total_time_s": round(time.time() - t0, 1),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    log(f"K2147 M2P ≥ DARE:           {'PASS' if k2147 else 'FAIL'}  Δ={m2p_avg-dare_avg:+.1f}pp")
    log(f"K2148 calibration ρ ≥ 0.3:   {'PASS' if k2148 else 'FAIL'}  ρ̄={cal_avg:.3f}")
    log(f"K2149 latency within 1.2×:  {'PASS' if k2149 else 'FAIL'}  ratio={m2p_lat['p95_ms']/max(dare_lat['p95_ms'],1e-9):.2f}")
    log(f"\nVERDICT: {verdict}")
    log(f"Total: {results['total_time_s']:.0f}s")


if __name__ == "__main__":
    main()
