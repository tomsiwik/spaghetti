#!/usr/bin/env python3
"""
Beehive PoLAR strategy adapters — N=3 composition on Gemma 4 (Pierre Phase 1.3).

Trains 3 PoLAR strategy adapters on beehive trajectory_type splits:
  - prepare adapter (planning/decomposition strategy)
  - act adapter (concrete tool-use/execution strategy)
  - integrate adapter (synthesis/closure strategy)

Then composes them via correct Σ(B_i @ A_i)/N math and tests on standard
benchmarks to verify F#203 prediction (strategies transfer across domains).

Kill criteria:
  K2077: Each strategy adapter improves ≥2 of {GSM8K, HumanEval, MedQA} by ≥3pp
  K2078: N=3 PoLAR composition preserves single-strategy gains within 5pp on each benchmark
  K2079: Joint Stiefel maintained: max ||A^T A − I||_F < 0.01 AND ||B B^T − I||_F < 0.01
  K2080: 3-strategy composition beats best single-strategy on at least 1 benchmark
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

from tooling.scripts.beehive_to_mlx import export_split
from tooling.scripts.polar_train import (
    inject_polar_adapters, train, cleanup,
    eval_gsm8k, eval_humaneval, eval_medqa,
    PoLARLinear, RANK, SCALE,
)

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42

N_STEPS = 20 if IS_SMOKE else 800
N_BENCH_EVAL = 5 if IS_SMOKE else 30
STRATEGIES = ["prepare", "act", "integrate"]


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    a = mx.get_active_memory() / 1e9
    c = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB")


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ─────────────────────────────────────────────
# Composition: pre-compute Σ(B_i @ A_i)/N on top of fresh PoLAR layers
# ─────────────────────────────────────────────

def compose_apply(modules, adapter_states_list: list[list[dict]]):
    """For each layer, attach pre-computed Σ_i(a_i b_i)/N as `_composed_delta`
    and override forward to use it. Quantization-safe (base is untouched)."""
    n = len(adapter_states_list)
    assert n > 0
    for layer_idx, m in enumerate(modules):
        delta = None
        for state in adapter_states_list:
            a = state[layer_idx]["a"]
            b = state[layer_idx]["b"]
            d = a @ b
            delta = d if delta is None else delta + d
        delta = delta / n
        mx.eval(delta)
        m._composed_delta = delta

        def make_fwd(layer):
            def fwd(x):
                return layer.base(x) + layer.scale * (x @ layer._composed_delta)
            return fwd
        m.__call__ = make_fwd(m).__get__(m)


def reset_compose(modules):
    for m in modules:
        if hasattr(m, "_composed_delta"):
            del m._composed_delta
        m.__call__ = PoLARLinear.__call__.__get__(m)


# ─────────────────────────────────────────────
# Train one strategy adapter
# ─────────────────────────────────────────────

def train_strategy(label: str, train_records: list[dict], n_steps: int) -> tuple[dict, list[dict]]:
    """Returns (train_stats, adapter_state_per_layer)."""
    log(f"\n[Train: strategy={label}, n_records={len(train_records)}]")
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
    log_memory(f"polar-injected-{label}")

    stats = train(model, tokenizer, train_records, modules, n_steps)
    log(f"  loss: {stats['first_loss']:.4f} → {stats['final_loss']:.4f}, Stiefel A={stats['stiefel_max_A']:.4f} B={stats['stiefel_max_B']:.4f}")

    # Snapshot per-layer state to numpy (so we can re-inject onto fresh model)
    state = []
    for m in modules:
        state.append({
            "a": np.array(m.lora_a.tolist(), dtype=np.float32),
            "b": np.array(m.lora_b.tolist(), dtype=np.float32),
        })

    # Save for inspection
    weights = {}
    for i, m in enumerate(modules):
        weights[f"layer_{i}.lora_a"] = m.lora_a
        weights[f"layer_{i}.lora_b"] = m.lora_b
    mx.eval(weights)
    out = EXPERIMENT_DIR / "adapter_weights" / f"strategy_{label}.safetensors"
    out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out), weights)

    cleanup(model, tokenizer, modules)
    return stats, state


def eval_with_state(label: str, state: list[dict]) -> dict:
    """Inject state into fresh model and run all 3 benchmarks."""
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
    for i, m in enumerate(modules):
        m.lora_a = mx.array(state[i]["a"])
        m.lora_b = mx.array(state[i]["b"])
    mx.eval(model.parameters())

    log(f"\n[Eval: {label}]")
    g = eval_gsm8k(model, tokenizer, N_BENCH_EVAL)
    h = eval_humaneval(model, tokenizer, N_BENCH_EVAL)
    d = eval_medqa(model, tokenizer, N_BENCH_EVAL)
    log(f"  {label}: GSM8K={g:.1f}%  HumanEval={h:.1f}%  MedQA={d:.1f}%")
    cleanup(model, tokenizer, modules)
    return {"gsm8k": round(g, 1), "humaneval": round(h, 1), "medqa": round(d, 1)}


def eval_composed(states: list[list[dict]]) -> dict:
    """Apply N-way composition to fresh model and run all 3 benchmarks."""
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)

    # Convert numpy state back to mx for compose_apply
    mx_states = []
    for state in states:
        mx_states.append([{"a": mx.array(s["a"]), "b": mx.array(s["b"])} for s in state])
    compose_apply(modules, mx_states)
    mx.eval(model.parameters())
    log(f"\n[Eval: composed N={len(states)}]")
    g = eval_gsm8k(model, tokenizer, N_BENCH_EVAL)
    h = eval_humaneval(model, tokenizer, N_BENCH_EVAL)
    d = eval_medqa(model, tokenizer, N_BENCH_EVAL)
    log(f"  composed: GSM8K={g:.1f}%  HumanEval={h:.1f}%  MedQA={d:.1f}%")
    cleanup(model, tokenizer, modules)
    return {"gsm8k": round(g, 1), "humaneval": round(h, 1), "medqa": round(d, 1)}


def main():
    t_start = time.time()
    log_memory("start")
    log(f"=== Beehive PoLAR Strategy N=3 (SMOKE={IS_SMOKE}) ===")

    # Phase 0: export 3 strategy datasets (approved only — quality validated separately)
    log("\n[Phase 0] Export 3 strategy datasets")
    manifests = {}
    for s in STRATEGIES:
        out = EXPERIMENT_DIR / f"data_{s}"
        m = export_split(out_dir=out, quality="approved", traj_type=s, val_frac=0.2, seed=SEED)
        manifests[s] = m
        log(f"  {s}: train={m['n_train']}, val={m['n_valid']}")

    # Phase 1: base
    log("\n[Phase 1] Base eval")
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

    # Phase 2: train 3 strategy adapters
    log("\n[Phase 2] Train 3 strategy adapters")
    train_stats = {}
    states = {}
    for s in STRATEGIES:
        records = load_jsonl(EXPERIMENT_DIR / f"data_{s}" / "train.jsonl")
        if len(records) < 3:
            log(f"  SKIP {s}: only {len(records)} records (need ≥3)")
            train_stats[s] = {"skipped": True, "reason": f"{len(records)} records insufficient"}
            states[s] = None
            continue
        stats, state = train_strategy(s, records, N_STEPS)
        train_stats[s] = {k: v for k, v in stats.items() if k != "losses"}
        states[s] = state

    # Phase 3: per-strategy eval
    log("\n[Phase 3] Per-strategy eval")
    per_strategy = {}
    for s in STRATEGIES:
        if states[s] is None:
            per_strategy[s] = None
            continue
        per_strategy[s] = eval_with_state(s, states[s])

    # Phase 4: N=3 composition
    valid_states = [states[s] for s in STRATEGIES if states[s] is not None]
    composed = eval_composed(valid_states) if len(valid_states) >= 2 else None
    if composed:
        log(f"  composed N={len(valid_states)}: {composed}")

    # KCs
    log("\n=== Kill Criteria ===")
    benchmarks = ["gsm8k", "humaneval", "medqa"]

    # K2077: each strategy improves ≥2 of 3 benchmarks by ≥3pp
    k2077_per_strategy = {}
    for s in STRATEGIES:
        if per_strategy[s] is None:
            k2077_per_strategy[s] = {"pass": False, "reason": "skipped"}
            continue
        improvements = {b: per_strategy[s][b] - base[b] for b in benchmarks}
        n_passing = sum(1 for v in improvements.values() if v >= 3.0)
        k2077_per_strategy[s] = {"pass": n_passing >= 2, "n_passing": n_passing, "improvements_pp": {b: round(v, 1) for b, v in improvements.items()}}
    k2077 = all(v["pass"] for v in k2077_per_strategy.values())

    # K2078: composition preserves gains within 5pp
    k2078_per_bench = {}
    if composed:
        # For each benchmark, find max single-strategy score; ensure composed within 5pp.
        for b in benchmarks:
            single_best = max((per_strategy[s][b] for s in STRATEGIES if per_strategy[s] is not None), default=base[b])
            drop = single_best - composed[b]
            k2078_per_bench[b] = {"pass": drop <= 5.0, "single_best": single_best, "composed": composed[b], "drop_pp": round(drop, 1)}
        k2078 = all(v["pass"] for v in k2078_per_bench.values())
    else:
        k2078 = False

    # K2079: Stiefel maintained for all trained strategies
    stiefel_max = {s: {"A": train_stats[s].get("stiefel_max_A", float("inf")),
                       "B": train_stats[s].get("stiefel_max_B", float("inf"))}
                   for s in STRATEGIES if states[s] is not None}
    k2079 = all(v["A"] < 0.01 and v["B"] < 0.01 for v in stiefel_max.values())

    # K2080: composition beats best single on ≥1 benchmark
    k2080 = False
    if composed:
        for b in benchmarks:
            single_best = max((per_strategy[s][b] for s in STRATEGIES if per_strategy[s] is not None), default=base[b])
            if composed[b] > single_best:
                k2080 = True
                break

    all_pass = k2077 and k2078 and k2079 and k2080
    verdict = "PROVISIONAL" if IS_SMOKE else ("SUPPORTED" if all_pass else "KILLED")

    results = {
        "is_smoke": IS_SMOKE,
        "n_steps": N_STEPS,
        "manifests": {s: {k: m[k] for k in ["n_train", "n_valid", "stratification"]} for s, m in manifests.items()},
        "train_stats": train_stats,
        "base": base,
        "per_strategy": per_strategy,
        "composed": composed,
        "kill_criteria": {
            "K2077_each_strategy_2of3": {"pass": k2077, "per_strategy": k2077_per_strategy},
            "K2078_composition_preserves_5pp": {"pass": k2078, "per_benchmark": k2078_per_bench},
            "K2079_stiefel": {"pass": k2079, "per_strategy_max": stiefel_max},
            "K2080_composition_beats_best": {"pass": k2080, "composed": composed, "single_best_per_bench": {
                b: max((per_strategy[s][b] for s in STRATEGIES if per_strategy[s] is not None), default=base[b])
                for b in benchmarks
            } if composed else None},
        },
        "verdict": verdict,
        "all_pass": all_pass,
        "total_time_s": round(time.time() - t_start, 1),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    log(f"K2077 each strategy ≥2 of 3 +3pp:  {'PASS' if k2077 else 'FAIL'}")
    for s, v in k2077_per_strategy.items():
        log(f"  {s}: {v}")
    log(f"K2078 composition within 5pp:      {'PASS' if k2078 else 'FAIL'}  {k2078_per_bench}")
    log(f"K2079 joint Stiefel:               {'PASS' if k2079 else 'FAIL'}  {stiefel_max}")
    log(f"K2080 composition > single (1+):   {'PASS' if k2080 else 'FAIL'}")
    log(f"\nVERDICT: {verdict}")
    log(f"Total time: {results['total_time_s']:.0f}s")


if __name__ == "__main__":
    main()
