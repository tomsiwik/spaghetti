#!/usr/bin/env python3
"""
Per-task routing for math: hybrid serving that closes the GSM8K -6.7pp regression.

Strategy:
  - Math-shaped queries → single best math-domain adapter (no fused delta)
  - Everything else → DARE composition over 7 adapters via _FusedDeltaLinear

Math-vs-non-math classifier: TF-IDF + ridge (cheap, F#431 validated at 96.6% on
similar Gemma 4 domain classification).

Uses the canonical _FusedDeltaLinear pattern (Finding #831).

Kill criteria:
  K2143: GSM8K accuracy ≥ best single-adapter (70.0%) within 2pp
  K2144: HumanEval/MedQA composed accuracy preserved within 2pp of DARE result
  K2145: Routing accuracy ≥85% on math-vs-non-math binary
  K2146: Routing overhead ≤5ms per query
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
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.polar_train import (
    inject_polar_adapters, cleanup,
    eval_gsm8k, eval_humaneval, eval_medqa,
    PoLARLinear, RANK, SCALE,
)

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42
N_BENCH_EVAL = 5 if IS_SMOKE else 30
DARE_DROP_RATE = 0.90  # validated by exp_pierre_ties_dare_composition

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
MATH_ADAPTER_IDX = SLOT_NAMES.index("domain_math")


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


# ─────────────────────────────────────────────
# Canonical _FusedDeltaLinear (Finding #831)
# ─────────────────────────────────────────────

class _FusedDeltaLinear(nn.Module):
    """y = base(x) + x @ fused_delta. Proper nn.Module subclass — never override __call__."""
    def __init__(self, base_layer, fused):
        super().__init__()
        self.base = base_layer
        self._fused = fused
    def __call__(self, x):
        return self.base(x) + (x @ self._fused)


def apply_dare_composition(model, all_states, drop_rate=DARE_DROP_RATE, seed=SEED):
    """DARE: drop p% randomly + rescale by 1/(1-p) + linear average. Apply via setattr."""
    rng = np.random.default_rng(seed)
    layers_iter = (model.language_model.model.layers if hasattr(model, "language_model")
                   else model.model.language_model.layers)
    n_layers = len(all_states[0])
    for layer_idx, layer in enumerate(layers_iter):
        if layer_idx >= n_layers:
            break
        # Build per-adapter task vectors with DARE preprocessing
        delta_sum = None
        for state in all_states:
            a = state[layer_idx]["a"]; b = state[layer_idx]["b"]
            tv = SCALE * (a @ b)
            mask = rng.binomial(1, 1.0 - drop_rate, size=tv.shape).astype(np.float32)
            tv_dare = (tv * mask) / max(1e-9, 1.0 - drop_rate)
            delta_sum = tv_dare if delta_sum is None else delta_sum + tv_dare
        delta_avg = delta_sum / len(all_states)
        delta_mx = mx.array(delta_avg.astype(np.float32))
        mx.eval(delta_mx)
        q_proj = layer.self_attn.q_proj
        base_layer = q_proj.base if isinstance(q_proj, (PoLARLinear, _FusedDeltaLinear)) else q_proj
        layer.self_attn.q_proj = _FusedDeltaLinear(base_layer, delta_mx)


def apply_single_adapter(model, state):
    """Load single adapter via PoLAR injection — used for math route."""
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
    for i, m in enumerate(modules):
        m.lora_a = mx.array(state[i]["a"]); m.lora_b = mx.array(state[i]["b"])
    mx.eval(model.parameters())
    return modules


# ─────────────────────────────────────────────
# Math-vs-non-math classifier
# ─────────────────────────────────────────────

def build_math_classifier_corpus():
    """Math = GSM8K-train. Non-math = beehive + CodeAlpaca + MedQA-train."""
    from datasets import load_dataset

    math_prompts, nonmath_prompts = [], []

    # Math (positive class)
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=SEED).select(range(150))
    for ex in ds:
        math_prompts.append(f"Solve step by step.\n\n{ex['question']}\n\nAnswer:")

    # Non-math: beehive (50), CodeAlpaca (50), MedQA-train (50)
    snapshot = REPO_ROOT / "data" / "beehive_snapshot" / "approved.jsonl"
    if snapshot.exists():
        with open(snapshot) as f:
            for i, line in enumerate(f):
                if i >= 50: break
                rec = json.loads(line)
                user_msg = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
                nonmath_prompts.append(user_msg)
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train").shuffle(seed=SEED).select(range(50))
    for ex in ds:
        prompt = ex["instruction"] + (f"\n\nInput:\n{ex['input']}" if ex.get("input") else "")
        nonmath_prompts.append(f"Complete this Python function:\n\n```python\n{prompt}\n```")
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train").shuffle(seed=SEED).select(range(50))
        for ex in ds:
            opts = ex["options"]
            q = f"{ex['question']}\n(A) {opts['A']}\n(B) {opts['B']}\n(C) {opts['C']}\n(D) {opts['D']}"
            nonmath_prompts.append(f"Answer with only the letter.\n\n{q}")
    except Exception:
        pass

    texts = math_prompts + nonmath_prompts
    labels = ["math"] * len(math_prompts) + ["nonmath"] * len(nonmath_prompts)
    return texts, labels


def train_math_classifier():
    log("\n[Phase 1a] Train math-vs-non-math classifier")
    texts, labels = build_math_classifier_corpus()
    log(f"  corpus: {sum(1 for l in labels if l=='math')} math, "
        f"{sum(1 for l in labels if l=='nonmath')} non-math")

    X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.2,
                                              random_state=SEED, stratify=labels)
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Xv_tr = vec.fit_transform(X_tr)
    Xv_te = vec.transform(X_te)
    clf = RidgeClassifier()
    clf.fit(Xv_tr, y_tr)

    train_acc = clf.score(Xv_tr, y_tr) * 100
    test_acc = clf.score(Xv_te, y_te) * 100
    log(f"  train_acc={train_acc:.1f}%  test_acc={test_acc:.1f}%")
    return vec, clf, test_acc


def is_math_query(prompt: str, vec, clf) -> bool:
    return clf.predict(vec.transform([prompt]))[0] == "math"


def time_routing_overhead(vec, clf, n_trials=100) -> float:
    samples = ["What is 7 + 5?", "Write a Python function.", "Explain photosynthesis."] * (n_trials // 3 + 1)
    samples = samples[:n_trials]
    t0 = time.perf_counter()
    for s in samples:
        _ = is_math_query(s, vec, clf)
    elapsed_ms = (time.perf_counter() - t0) * 1000 / n_trials
    return elapsed_ms


# ─────────────────────────────────────────────
# Per-task routed evaluation
# ─────────────────────────────────────────────

def eval_with_math_routing(vec, clf, all_states):
    """For each benchmark prompt: route, then dispatch to math-single or DARE-composed."""
    from datasets import load_dataset
    from mlx_lm import load, generate

    log("\n[Phase 3] Per-task routed evaluation")

    benchmarks = {}

    # Build all eval prompts up-front so we can bucket by route decision
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

    # For each benchmark, bucket by routing decision (most prompts in a benchmark
    # will route to the same place, so two model loads max per bench)
    routing_summary = {}
    accuracy = {}

    for bench, pairs in bench_pairs.items():
        log(f"\n  [{bench}] routing decisions...")
        math_route, dare_route = [], []
        for i, (prompt, gold) in enumerate(pairs):
            if is_math_query(prompt, vec, clf):
                math_route.append((i, prompt, gold))
            else:
                dare_route.append((i, prompt, gold))
        log(f"    {len(math_route)} → math-single, {len(dare_route)} → DARE-composed")
        routing_summary[bench] = {"math_route": len(math_route), "dare_route": len(dare_route)}

        results_by_idx = {}

        # MATH-ROUTE: load model + single math adapter
        if math_route:
            model, tokenizer = load(MODEL_ID)
            mx.eval(model.parameters())
            apply_single_adapter(model, all_states[MATH_ADAPTER_IDX])
            for prompt_idx, prompt, gold in math_route:
                results_by_idx[prompt_idx] = _generate_and_score(
                    model, tokenizer, prompt, gold, bench)
            cleanup(model, tokenizer)

        # DARE-ROUTE: load model + DARE composition
        if dare_route:
            model, tokenizer = load(MODEL_ID)
            mx.eval(model.parameters())
            apply_dare_composition(model, all_states)
            mx.eval(model.parameters())
            for prompt_idx, prompt, gold in dare_route:
                results_by_idx[prompt_idx] = _generate_and_score(
                    model, tokenizer, prompt, gold, bench)
            cleanup(model, tokenizer)

        correct = sum(1 for ok in results_by_idx.values() if ok)
        accuracy[bench] = round(correct / len(pairs) * 100, 1)
        log(f"    accuracy = {accuracy[bench]}%")

    return accuracy, routing_summary


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


# ─────────────────────────────────────────────
# Reference baselines (single-adapter best, DARE-only)
# ─────────────────────────────────────────────

def eval_dare_only(all_states):
    """Reference: pure DARE composition (the prior winner) for comparison."""
    from mlx_lm import load
    log("\n[Phase 2] Reference: DARE-only composition")
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    apply_dare_composition(model, all_states)
    mx.eval(model.parameters())
    g = eval_gsm8k(model, tokenizer, N_BENCH_EVAL)
    h = eval_humaneval(model, tokenizer, N_BENCH_EVAL)
    d = eval_medqa(model, tokenizer, N_BENCH_EVAL)
    cleanup(model, tokenizer)
    return {"gsm8k": round(g, 1), "humaneval": round(h, 1), "medqa": round(d, 1)}


def eval_single_math(state):
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    apply_single_adapter(model, state)
    g = eval_gsm8k(model, tokenizer, N_BENCH_EVAL)
    cleanup(model, tokenizer)
    return round(g, 1)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    t0 = time.time()
    log_memory("start")
    log(f"=== Per-Task Routing for Math (SMOKE={IS_SMOKE}) ===")

    # Load adapters
    log("\n[Phase 0] Load 7 PoLAR adapters")
    all_states = []
    for slot_name, path in ADAPTER_SLOTS:
        if not path.exists():
            log(f"  FATAL: missing {slot_name}: {path}")
            sys.exit(1)
        all_states.append(load_state(path))
    log(f"  loaded {len(all_states)} adapters")

    # Train math classifier
    vec, clf, classifier_test_acc = train_math_classifier()
    routing_overhead_ms = time_routing_overhead(vec, clf)
    log(f"  routing overhead: {routing_overhead_ms:.3f} ms/query")

    # Reference baseline: single math adapter on GSM8K (the bar to beat)
    log("\n[Phase 2a] Single math-adapter on GSM8K (reference)")
    math_only_gsm8k = eval_single_math(all_states[MATH_ADAPTER_IDX])
    log(f"  domain_math single → GSM8K = {math_only_gsm8k}%")

    # DARE-only (the prior result we're trying to improve on for GSM8K)
    dare_only = eval_dare_only(all_states)
    log(f"  DARE-only: {dare_only}")

    # Per-task routed evaluation
    routed_acc, routing_summary = eval_with_math_routing(vec, clf, all_states)
    log(f"  routed: {routed_acc}")

    # KCs
    log("\n=== Kill Criteria ===")
    BENCH = ["gsm8k", "humaneval", "medqa"]
    BEST_SINGLE = {"gsm8k": 70.0, "humaneval": 86.7, "medqa": 50.0}  # from prior runs
    DARE_REFERENCE = {"gsm8k": 63.3, "humaneval": 90.0, "medqa": 66.7}  # known DARE numbers

    # K2143: GSM8K within 2pp of best single (70.0%)
    gsm8k_drop = BEST_SINGLE["gsm8k"] - routed_acc["gsm8k"]
    k2143 = gsm8k_drop <= 2.0

    # K2144: HumanEval/MedQA within 2pp of DARE result
    he_drop = DARE_REFERENCE["humaneval"] - routed_acc["humaneval"]
    md_drop = DARE_REFERENCE["medqa"] - routed_acc["medqa"]
    k2144 = he_drop <= 2.0 and md_drop <= 2.0

    # K2145: classifier accuracy ≥ 85%
    k2145 = classifier_test_acc >= 85.0

    # K2146: routing overhead ≤ 5ms
    k2146 = routing_overhead_ms <= 5.0

    all_pass = k2143 and k2144 and k2145 and k2146
    verdict = "PROVISIONAL" if IS_SMOKE else ("SUPPORTED" if all_pass else "KILLED")

    results = {
        "is_smoke": IS_SMOKE,
        "config": {"dare_drop_rate": DARE_DROP_RATE, "best_single_per_bench": BEST_SINGLE,
                   "dare_reference": DARE_REFERENCE},
        "math_classifier_test_acc": round(classifier_test_acc, 1),
        "routing_overhead_ms": round(routing_overhead_ms, 3),
        "single_math_gsm8k": math_only_gsm8k,
        "dare_only": dare_only,
        "routed_per_task": routed_acc,
        "routing_summary": routing_summary,
        "drops": {"gsm8k_vs_best_single": round(gsm8k_drop, 1),
                  "humaneval_vs_dare": round(he_drop, 1),
                  "medqa_vs_dare": round(md_drop, 1)},
        "kill_criteria": {
            "K2143_gsm8k_within_2pp": {"pass": k2143, "drop_pp": round(gsm8k_drop, 1)},
            "K2144_he_md_within_2pp": {"pass": k2144,
                                       "humaneval_drop": round(he_drop, 1),
                                       "medqa_drop": round(md_drop, 1)},
            "K2145_classifier_85": {"pass": k2145, "test_acc": round(classifier_test_acc, 1)},
            "K2146_routing_5ms": {"pass": k2146, "ms_per_query": round(routing_overhead_ms, 3)},
        },
        "verdict": verdict, "all_pass": all_pass,
        "total_time_s": round(time.time() - t0, 1),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    log(f"K2143 GSM8K within 2pp:    {'PASS' if k2143 else 'FAIL'}  drop={gsm8k_drop:+.1f}pp")
    log(f"K2144 HE/MD within 2pp:    {'PASS' if k2144 else 'FAIL'}  he={he_drop:+.1f} md={md_drop:+.1f}")
    log(f"K2145 classifier ≥85%:     {'PASS' if k2145 else 'FAIL'}  ({classifier_test_acc:.1f}%)")
    log(f"K2146 routing ≤5ms:        {'PASS' if k2146 else 'FAIL'}  ({routing_overhead_ms:.2f}ms)")
    log(f"\nVERDICT: {verdict}")
    log(f"Total: {results['total_time_s']:.0f}s")


if __name__ == "__main__":
    main()
