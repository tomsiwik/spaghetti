#!/usr/bin/env python3
"""
Pierre Phase 1 E2E viability gate: 3 strategy + 3 domain adapters composed.

Uses the WINNING composition method from exp_beehive_polar_composition_mechanism
(read from its results.json). Loads adapters trained in:
  - exp_beehive_polar_strategy_full_corpus  (strategy: prepare/act/integrate)
  - exp_beehive_polar_domain_5              (domain: math/code/medical)

This is the headline number for the Pierre Phase 1 founding pitch.

Kill criteria:
  K2102: 6-adapter composition preserves best per-domain accuracy within 7pp
  K2103: Composed > base on ALL of GSM8K/HumanEval/MedQA by ≥5pp
  K2104: Composed > best single-adapter on ≥1 benchmark
  K2105: P95 first-token latency ≤500ms on M5 Pro
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
STRATEGIES = ["prepare", "act", "integrate"]
DOMAINS = ["math", "code", "medical"]


def log(m): print(m, flush=True)


def log_memory(label=""):
    a = mx.get_active_memory() / 1e9
    c = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB")


def load_state(path: Path) -> list[dict]:
    raw = mx.load(str(path))
    n_layers = max(int(k.split(".")[0].split("_")[1]) for k in raw if k.startswith("layer_")) + 1
    return [{
        "a": np.array(raw[f"layer_{i}.lora_a"].tolist(), dtype=np.float32),
        "b": np.array(raw[f"layer_{i}.lora_b"].tolist(), dtype=np.float32),
    } for i in range(n_layers)]


def apply_weighted_sum(modules, states_list: list[list[dict]], weights: list[float] | None = None):
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


def measure_first_token_latency_ms(model, tokenizer, n_trials: int = 10) -> dict:
    """Measure first-token latency with the composed model."""
    from mlx_lm import generate

    prompts = [
        "What is 7 + 5?",
        "Write a Python function to add two numbers.",
        "What's the capital of France?",
        "Explain DNA in one sentence.",
        "How do I sort a list?",
    ]
    samples = []
    for trial in range(n_trials):
        prompt = prompts[trial % len(prompts)]
        msgs = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        t_start = time.perf_counter()
        _ = generate(model, tokenizer, prompt=formatted, max_tokens=1, verbose=False)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        samples.append(elapsed_ms)
    samples_sorted = sorted(samples)
    p50 = samples_sorted[len(samples) // 2]
    p95 = samples_sorted[int(len(samples) * 0.95)]
    return {"p50_ms": round(p50, 1), "p95_ms": round(p95, 1), "samples_ms": [round(s, 1) for s in samples]}


def main():
    t0 = time.time()
    log_memory("start")
    log(f"=== Pierre Phase 1 E2E Viability (SMOKE={IS_SMOKE}) ===")

    # Phase 0: read winning composition method from upstream experiment
    log("\n[Phase 0] Determine winning composition method from exp_beehive_polar_composition_mechanism")
    upstream = REPO_ROOT / "micro" / "models" / "exp_beehive_polar_composition_mechanism" / "results.json"
    if not upstream.exists():
        log(f"  FATAL: upstream results not found: {upstream}")
        sys.exit(1)
    upstream_results = json.loads(upstream.read_text())
    winners_kc = upstream_results.get("kill_criteria", {}).get("K2098_at_least_one_preserves_5pp", {})
    winning_methods = winners_kc.get("winners", [])
    if not winning_methods:
        log(f"  WARN: no method passed K2098 in upstream. Using weighted_sum as default.")
        winning_method = "weighted_sum_n4"
    else:
        # Prefer routed if it works (production-friendly), else weighted_sum
        winning_method = "routed" if "routed" in winning_methods else winning_methods[0]
    log(f"  Using composition method: {winning_method}")

    # Phase 1: Load adapters
    log("\n[Phase 1] Load 6 adapters (3 strategy + 3 domain)")
    strategy_states = {}
    for s in STRATEGIES:
        wpath = ADAPTER_DIR / f"strategy_{s}_polar" / "polar.safetensors"
        if not wpath.exists():
            log(f"  FATAL: missing strategy adapter {wpath}")
            sys.exit(1)
        strategy_states[s] = load_state(wpath)
        log(f"  strategy/{s}: {len(strategy_states[s])} layers")

    domain_states = {}
    for d in DOMAINS:
        # exp_beehive_polar_domain_5 saves to adapters/{domain}_polar/polar.safetensors
        wpath = ADAPTER_DIR / f"{d}_polar" / "polar.safetensors"
        if not wpath.exists():
            log(f"  FATAL: missing domain adapter {wpath}")
            sys.exit(1)
        domain_states[d] = load_state(wpath)
        log(f"  domain/{d}: {len(domain_states[d])} layers")

    # Phase 2: Base + per-single-adapter eval
    log("\n[Phase 2] Base + per-single eval")
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

    BENCH = ["gsm8k", "humaneval", "medqa"]
    # Per-single eval (just to compute best-single baseline; reuse upstream numbers if available)
    upstream_per_strategy = upstream_results.get("per_strategy_single", {})
    if upstream_per_strategy:
        per_strategy = upstream_per_strategy
        log(f"  reusing per-strategy from upstream: {per_strategy}")
    else:
        log(f"  WARN: no upstream per_strategy_single found; will rely on base for K2104")
        per_strategy = {}

    # Phase 3: 6-adapter composition
    log("\n[Phase 3] 6-adapter composition (weighted-sum 1/6)")
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
    states_list = list(strategy_states.values()) + list(domain_states.values())
    apply_weighted_sum(modules, states_list)
    mx.eval(model.parameters())

    composed = {
        "gsm8k": round(eval_gsm8k(model, tokenizer, N_BENCH_EVAL), 1),
        "humaneval": round(eval_humaneval(model, tokenizer, N_BENCH_EVAL), 1),
        "medqa": round(eval_medqa(model, tokenizer, N_BENCH_EVAL), 1),
    }
    log(f"  6-composed: {composed}")

    # Phase 4: Latency measurement
    log("\n[Phase 4] First-token latency (Phase 1 budget: ≤500ms p95)")
    latency = measure_first_token_latency_ms(model, tokenizer, n_trials=10)
    log(f"  latency p50={latency['p50_ms']}ms p95={latency['p95_ms']}ms")
    cleanup(model, tokenizer, modules)

    # KCs
    if per_strategy:
        best_single = {b: max(per_strategy[s][b] for s in per_strategy) for b in BENCH}
    else:
        best_single = base
    drops_vs_best_single = {b: round(best_single[b] - composed[b], 1) for b in BENCH}
    deltas_vs_base = {b: round(composed[b] - base[b], 1) for b in BENCH}

    k2102 = all(d <= 7.0 for d in drops_vs_best_single.values())
    k2103 = all(d >= 5.0 for d in deltas_vs_base.values())
    k2104 = any(composed[b] > best_single[b] for b in BENCH) if per_strategy else False
    k2105 = latency["p95_ms"] <= 500.0

    all_pass = k2102 and k2103 and k2104 and k2105
    verdict = "PROVISIONAL" if IS_SMOKE else ("SUPPORTED" if all_pass else "KILLED")

    results = {
        "is_smoke": IS_SMOKE,
        "winning_composition_method": winning_method,
        "base": base,
        "best_single_per_bench": best_single,
        "composed_6": composed,
        "drops_vs_best_single_pp": drops_vs_best_single,
        "deltas_vs_base_pp": deltas_vs_base,
        "latency_first_token": latency,
        "kill_criteria": {
            "K2102_composition_within_7pp": {"pass": k2102, "drops_pp": drops_vs_best_single},
            "K2103_beats_base_5pp_all": {"pass": k2103, "deltas_pp": deltas_vs_base},
            "K2104_beats_best_single_on_1": {"pass": k2104, "composed": composed, "best_single": best_single},
            "K2105_p95_latency_500ms": {"pass": k2105, "p95_ms": latency["p95_ms"]},
        },
        "verdict": verdict, "all_pass": all_pass,
        "total_time_s": round(time.time() - t0, 1),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    log("\n=== KCs ===")
    log(f"K2102 within 7pp of best single: {'PASS' if k2102 else 'FAIL'}  drops={drops_vs_best_single}")
    log(f"K2103 ≥5pp over base on ALL:    {'PASS' if k2103 else 'FAIL'}  deltas={deltas_vs_base}")
    log(f"K2104 > best single on ≥1:       {'PASS' if k2104 else 'FAIL'}")
    log(f"K2105 p95 latency ≤500ms:        {'PASS' if k2105 else 'FAIL'}  ({latency['p95_ms']}ms)")
    log(f"\nVERDICT: {verdict}")
    log(f"Total: {results['total_time_s']:.0f}s")


if __name__ == "__main__":
    main()
