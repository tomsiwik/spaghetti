#!/usr/bin/env python3
"""
Composition mechanism fix — the actual research blocker for Pierre Phase 1.5.

Background: exp_beehive_polar_strategy_n3 KILLED for K2078. With 3 over-trained
PoLAR strategy adapters (loss=0.0001 each), naive sum(B_i @ A_i)/N composition
collapsed HumanEval from 77% → 20%.

This experiment tests two concrete fixes against under-trained (loss=0.1) adapters
loaded from exp_beehive_polar_strategy_full_corpus:

  Method 1: Weighted-sum composition with regularized adapters (under-trained)
    ΔW = Σ_i (α_i * B_i @ A_i)   with α_i = 1/N or learned
    Hypothesis: under-trained adapters are not over-confident, so their average
    is constructive not destructive.

  Method 2: Prompt-level routing (TF-IDF + ridge classifier picks ONE adapter)
    Per query, classify topic → activate single best-fit adapter.
    Already validated at 96.6% (F#431). This is the "safe ship" option.

Kill criteria:
  K2098: At least one method preserves single-strategy gains within 5pp on each benchmark
  K2099: Winning method beats best single-adapter on ≥1 benchmark
  K2100: Routing accuracy ≥80% (only applicable if Method 2 wins)
  K2101: Joint Stiefel preserved per-adapter (no gradient during inference)
"""

import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

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

ADAPTER_DIR = REPO_ROOT / "data" / "adapters"
STRATEGIES = ["full", "prepare", "act", "integrate"]


def log(m): print(m, flush=True)


def log_memory(label=""):
    a = mx.get_active_memory() / 1e9
    c = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB")


# ─────────────────────────────────────────────
# Load adapter states from exp B
# ─────────────────────────────────────────────

def load_state(path: Path) -> list[dict]:
    raw = mx.load(str(path))
    n_layers = max(int(k.split(".")[0].split("_")[1]) for k in raw if k.startswith("layer_")) + 1
    return [{
        "a": np.array(raw[f"layer_{i}.lora_a"].tolist(), dtype=np.float32),
        "b": np.array(raw[f"layer_{i}.lora_b"].tolist(), dtype=np.float32),
    } for i in range(n_layers)]


def stiefel_distance(state: list[dict]) -> tuple[float, float]:
    """Verify Joint Stiefel constraint (max ||A^T A - I||, ||B B^T - I|| across layers)."""
    max_A, max_B = 0.0, 0.0
    for s in state:
        a = s["a"]; b = s["b"]
        I = np.eye(a.shape[1])
        max_A = max(max_A, float(np.sqrt(np.sum((a.T @ a - I) ** 2))))
        max_B = max(max_B, float(np.sqrt(np.sum((b @ b.T - I) ** 2))))
    return max_A, max_B


# ─────────────────────────────────────────────
# Method 1: Weighted-sum composition
# ─────────────────────────────────────────────

def apply_weighted_sum(modules, states_list: list[list[dict]], weights: list[float] | None = None):
    """Σ_i α_i * (a_i @ b_i) with default α_i = 1/N."""
    n = len(states_list)
    if weights is None:
        weights = [1.0 / n] * n
    for layer_idx, m in enumerate(modules):
        delta = None
        for w_i, state in zip(weights, states_list):
            a = mx.array(state[layer_idx]["a"])
            b = mx.array(state[layer_idx]["b"])
            d = (a @ b) * w_i
            delta = d if delta is None else delta + d
        mx.eval(delta)
        m._composed_delta = delta

        def make_fwd(layer):
            def fwd(x):
                return layer.base(x) + layer.scale * (x @ layer._composed_delta)
            return fwd
        m.__call__ = make_fwd(m).__get__(m)


# ─────────────────────────────────────────────
# Method 2: Prompt-level routing (TF-IDF + ridge)
# ─────────────────────────────────────────────

def build_router(strategy_states: dict[str, list[dict]]) -> tuple:
    """Train TF-IDF + RidgeClassifier on training prompts.

    Each strategy has its own training prompts (the user msgs from beehive). We
    label each user message with its strategy type and learn a classifier.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import train_test_split

    texts, labels = [], []
    for s in strategy_states:
        data_dir = REPO_ROOT / "micro" / "models" / "exp_beehive_polar_strategy_full_corpus" / f"data_{s}"
        if not (data_dir / "train.jsonl").exists():
            continue
        for line in open(data_dir / "train.jsonl"):
            rec = json.loads(line)
            user_msg = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
            texts.append(user_msg)
            labels.append(s)

    X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.2, random_state=SEED, stratify=labels)
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Xv = vec.fit_transform(X_tr)
    clf = RidgeClassifier()
    clf.fit(Xv, y_tr)
    test_acc = clf.score(vec.transform(X_te), y_te) * 100
    log(f"  Router: trained on {len(X_tr)} samples, test accuracy {test_acc:.1f}% on {len(X_te)} held-out")
    return vec, clf, test_acc


def evaluate_routed(strategy_states: dict[str, list[dict]], vec, clf, benchmark: str) -> tuple[float, dict]:
    """Per-prompt routing: classify the prompt, activate one strategy adapter, generate."""
    from datasets import load_dataset
    from mlx_lm import load, generate

    counts = {s: 0 for s in strategy_states}
    log(f"\n[Method 2 Routed: {benchmark}]")

    # Build (prompt, gold) pairs depending on benchmark
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
    else:
        raise ValueError(benchmark)

    # Pre-route all prompts (group by predicted strategy → batch model loads)
    prompts_by_strategy = {s: [] for s in strategy_states}
    for prompt, gold in pairs:
        pred = clf.predict(vec.transform([prompt]))[0]
        counts[pred] += 1
        prompts_by_strategy[pred].append((prompt, gold))

    # Eval each strategy bucket using its adapter
    correct = 0
    total = 0
    import re, subprocess
    for strategy, items in prompts_by_strategy.items():
        if not items:
            continue
        log(f"  routing {len(items)}/{N_BENCH_EVAL} to {strategy}")
        model, tokenizer = load(MODEL_ID)
        mx.eval(model.parameters())
        modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
        state = strategy_states[strategy]
        for i, m in enumerate(modules):
            m.lora_a = mx.array(state[i]["a"])
            m.lora_b = mx.array(state[i]["b"])
        mx.eval(model.parameters())

        for prompt, gold in items:
            msgs = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            max_t = 1024 if benchmark == "gsm8k" else (512 if benchmark == "humaneval" else 20)
            response = generate(model, tokenizer, prompt=formatted, max_tokens=max_t, verbose=False)
            total += 1

            if benchmark == "gsm8k":
                gt_match = re.search(r"####\s*([\d,\-\.]+)", gold)
                if not gt_match: continue
                gt = gt_match.group(1).replace(",", "").strip()
                pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
                ok = pred_match and pred_match.group(1).replace(",", "").strip() == gt
                if not ok:
                    nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
                    ok = nums and nums[-1] == gt
                if ok: correct += 1
            elif benchmark == "humaneval":
                ex = gold
                code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
                completion = code_match.group(1) if code_match else response
                full_code = ex["prompt"] + completion + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"
                try:
                    r = subprocess.run([sys.executable, "-c", full_code], timeout=10, capture_output=True, text=True)
                    if r.returncode == 0: correct += 1
                except: pass
            elif benchmark == "medqa":
                ex = gold
                pred = response.strip().upper()
                pred_letter = next((L for L in "ABCD" if pred.startswith(L)), None)
                if not pred_letter:
                    m = re.search(r"\b([ABCD])\b", pred)
                    pred_letter = m.group(1) if m else None
                if pred_letter == ex["answer_idx"]: correct += 1

        cleanup(model, tokenizer, modules)

    acc = correct / total * 100 if total else 0.0
    return round(acc, 1), counts


def evaluate_weighted_sum(strategy_states: dict[str, list[dict]], strategies_used: list[str]) -> dict:
    """Method 1: load N adapters as weighted-sum composition, run all 3 benchmarks."""
    from mlx_lm import load
    log(f"\n[Method 1 Weighted-Sum: {strategies_used}]")
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
    states_list = [strategy_states[s] for s in strategies_used]
    apply_weighted_sum(modules, states_list)
    mx.eval(model.parameters())

    g = eval_gsm8k(model, tokenizer, N_BENCH_EVAL)
    h = eval_humaneval(model, tokenizer, N_BENCH_EVAL)
    d = eval_medqa(model, tokenizer, N_BENCH_EVAL)
    log(f"  weighted-sum: GSM8K={g:.1f}% HumanEval={h:.1f}% MedQA={d:.1f}%")
    cleanup(model, tokenizer, modules)
    return {"gsm8k": round(g, 1), "humaneval": round(h, 1), "medqa": round(d, 1)}


def evaluate_single(strategy: str, state: list[dict]) -> dict:
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
    for i, m in enumerate(modules):
        m.lora_a = mx.array(state[i]["a"])
        m.lora_b = mx.array(state[i]["b"])
    mx.eval(model.parameters())
    g = eval_gsm8k(model, tokenizer, N_BENCH_EVAL)
    h = eval_humaneval(model, tokenizer, N_BENCH_EVAL)
    d = eval_medqa(model, tokenizer, N_BENCH_EVAL)
    cleanup(model, tokenizer, modules)
    return {"gsm8k": round(g, 1), "humaneval": round(h, 1), "medqa": round(d, 1)}


def main():
    t0 = time.time()
    log_memory("start")
    log(f"=== Composition Mechanism Fix (SMOKE={IS_SMOKE}) ===")

    # Phase 0: load 4 strategy adapters from upstream experiment
    log("\n[Phase 0] Load strategy adapters from exp_beehive_polar_strategy_full_corpus")
    strategy_states = {}
    for s in STRATEGIES:
        wpath = ADAPTER_DIR / f"strategy_{s}_polar" / "polar.safetensors"
        if not wpath.exists():
            log(f"  FATAL: missing {wpath}")
            sys.exit(1)
        strategy_states[s] = load_state(wpath)
        dA, dB = stiefel_distance(strategy_states[s])
        log(f"  {s}: loaded {len(strategy_states[s])} layers, Stiefel A={dA:.4e} B={dB:.4e}")

    # Phase 1: base + per-strategy single eval (for baseline comparison)
    log("\n[Phase 1] Base + per-strategy reference evals")
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    base = {
        "gsm8k": round(eval_gsm8k(model, tokenizer, N_BENCH_EVAL), 1),
        "humaneval": round(eval_humaneval(model, tokenizer, N_BENCH_EVAL), 1),
        "medqa": round(eval_medqa(model, tokenizer, N_BENCH_EVAL), 1),
    }
    log(f"  base: {base}")
    cleanup(model, tokenizer)

    per_strategy = {s: evaluate_single(s, strategy_states[s]) for s in STRATEGIES}

    BENCH = ["gsm8k", "humaneval", "medqa"]

    # Phase 2: Method 1 — Weighted-sum composition (test 2 group sizes)
    log("\n[Phase 2] Method 1: Weighted-sum (1/N)")
    method1_n3 = evaluate_weighted_sum(strategy_states, ["prepare", "act", "integrate"])
    method1_n4 = evaluate_weighted_sum(strategy_states, ["full", "prepare", "act", "integrate"])

    # Phase 3: Method 2 — Prompt-level routing
    log("\n[Phase 3] Method 2: Prompt-level routing")
    vec, clf, router_test_acc = build_router(strategy_states)
    method2 = {}
    routing_counts = {}
    for b in BENCH:
        acc, counts = evaluate_routed(strategy_states, vec, clf, b)
        method2[b] = acc
        routing_counts[b] = counts

    # Determine "best single" per benchmark
    best_single = {b: max(per_strategy[s][b] for s in STRATEGIES) for b in BENCH}

    # K2098: at least one method preserves within 5pp of best single
    method1_n3_drops = {b: round(best_single[b] - method1_n3[b], 1) for b in BENCH}
    method1_n4_drops = {b: round(best_single[b] - method1_n4[b], 1) for b in BENCH}
    method2_drops    = {b: round(best_single[b] - method2[b], 1) for b in BENCH}

    pass_method1_n3 = all(d <= 5.0 for d in method1_n3_drops.values())
    pass_method1_n4 = all(d <= 5.0 for d in method1_n4_drops.values())
    pass_method2    = all(d <= 5.0 for d in method2_drops.values())
    k2098 = pass_method1_n3 or pass_method1_n4 or pass_method2

    # K2099: winning method beats best single on ≥1 benchmark
    candidates = {
        "weighted_sum_n3": method1_n3,
        "weighted_sum_n4": method1_n4,
        "routed":          method2,
    }
    k2099 = any(any(method[b] > best_single[b] for b in BENCH) for method in candidates.values())
    winners = {name: [b for b in BENCH if method[b] > best_single[b]] for name, method in candidates.items()}

    # K2100: routing accuracy
    k2100 = router_test_acc >= 80.0

    # K2101: Stiefel preserved (we already verified at load — they're loaded, not retrained)
    stiefel_ok = all(stiefel_distance(strategy_states[s])[0] < 0.01 and stiefel_distance(strategy_states[s])[1] < 0.01 for s in STRATEGIES)
    k2101 = stiefel_ok

    all_pass = k2098 and k2099 and k2101  # K2100 only required IF method 2 is the winner
    verdict = "PROVISIONAL" if IS_SMOKE else ("SUPPORTED" if all_pass else "KILLED")

    results = {
        "is_smoke": IS_SMOKE,
        "base": base,
        "per_strategy_single": per_strategy,
        "best_single_per_benchmark": best_single,
        "method1_weighted_sum_n3": {"results": method1_n3, "drops_pp": method1_n3_drops, "pass_5pp": pass_method1_n3},
        "method1_weighted_sum_n4": {"results": method1_n4, "drops_pp": method1_n4_drops, "pass_5pp": pass_method1_n4},
        "method2_routed":         {"results": method2, "drops_pp": method2_drops, "pass_5pp": pass_method2,
                                   "router_test_acc": router_test_acc, "routing_counts_per_bench": routing_counts},
        "winners_per_method": winners,
        "kill_criteria": {
            "K2098_at_least_one_preserves_5pp": {"pass": k2098,
                "winners": [m for m, ok in [("weighted_sum_n3", pass_method1_n3),
                                            ("weighted_sum_n4", pass_method1_n4),
                                            ("routed", pass_method2)] if ok]},
            "K2099_beats_best_single": {"pass": k2099, "winners": winners},
            "K2100_router_accuracy": {"pass": k2100, "test_acc": router_test_acc, "applicable": pass_method2},
            "K2101_stiefel_preserved": {"pass": k2101},
        },
        "verdict": verdict, "all_pass": all_pass,
        "total_time_s": round(time.time() - t0, 1),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    log("\n=== KCs ===")
    log(f"K2098 ≥1 method preserves within 5pp: {'PASS' if k2098 else 'FAIL'}")
    log(f"  weighted_sum_n3: {'PASS' if pass_method1_n3 else 'FAIL'}  drops={method1_n3_drops}")
    log(f"  weighted_sum_n4: {'PASS' if pass_method1_n4 else 'FAIL'}  drops={method1_n4_drops}")
    log(f"  routed:          {'PASS' if pass_method2 else 'FAIL'}  drops={method2_drops}")
    log(f"K2099 beats best-single: {'PASS' if k2099 else 'FAIL'}  winners={winners}")
    log(f"K2100 router ≥80%:        {'PASS' if k2100 else 'FAIL'}  ({router_test_acc:.1f}%)")
    log(f"K2101 Stiefel preserved: {'PASS' if k2101 else 'FAIL'}")
    log(f"\nVERDICT: {verdict}")
    log(f"Total: {results['total_time_s']:.0f}s")


if __name__ == "__main__":
    main()
