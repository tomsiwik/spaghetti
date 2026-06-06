#!/usr/bin/env python3
"""
Per-strategy PoLAR adapters at full Turso scale (370+ rows each), early-stopped.

Trains 4 PoLAR adapters (full, prepare, act, integrate). Early stop at avg(loss_last_20)≤0.1
to prevent over-confidence — root cause of composition collapse in exp_beehive_polar_strategy_n3.

Adapters saved to adapters/strategy_{type}_polar/ for downstream composition test.

Kill criteria:
  K2094: Each strategy improves ≥2 of {GSM8K,HumanEval,MedQA} by ≥3pp (F#203 transfer at scale)
  K2095: Final loss in [0.05, 0.15] window (early-stop discipline)
  K2096: Joint Stiefel for ALL 4 adapters (max ||...|| < 0.01)
  K2097: Per-strategy gains beat exp_beehive_polar_strategy_n3 on ≥2 of 3 benchmarks (data scale value)
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
    RANK, SCALE,
)

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42

N_STEPS_MAX = 30 if IS_SMOKE else 1500    # cap; early stop at loss=0.1 will usually fire much earlier
STOP_AT_LOSS = 0.1
N_BENCH_EVAL = 5 if IS_SMOKE else 30

STRATEGIES = ["full", "prepare", "act", "integrate"]

# v1 (small data) reference numbers from exp_beehive_polar_strategy_n3 results.json
N3_REFERENCE = {
    "prepare":   {"gsm8k": 46.7, "humaneval": 66.7, "medqa": 26.7},
    "act":       {"gsm8k": 60.0, "humaneval": 53.3, "medqa": 23.3},
    "integrate": {"gsm8k": 63.3, "humaneval": 76.7, "medqa": 16.7},
}


def log(m): print(m, flush=True)


def log_memory(label=""):
    a = mx.get_active_memory() / 1e9
    c = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB")


def load_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def train_strategy(label: str, records: list[dict], n_steps_max: int) -> tuple[dict, list[dict], str]:
    log(f"\n[Train: strategy={label}, n_records={len(records)}, max_steps={n_steps_max}]")
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
    log_memory(f"polar-injected-{label}")

    stats = train(model, tokenizer, records, modules, n_steps_max, stop_at_loss=STOP_AT_LOSS)
    log(f"  loss: {stats['first_loss']:.4f} → {stats['final_loss']:.4f} (n_steps={len(stats['losses'])}, target={STOP_AT_LOSS}), Stiefel A={stats['stiefel_max_A']:.4f} B={stats['stiefel_max_B']:.4f}")

    # Snapshot for later composition
    state = [{
        "a": np.array(m.lora_a.tolist(), dtype=np.float32),
        "b": np.array(m.lora_b.tolist(), dtype=np.float32),
    } for m in modules]

    # Save canonically for pierre-server consumption
    out_dir = REPO_ROOT / "data" / "adapters" / f"strategy_{label}_polar"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = {f"layer_{i}.lora_a": m.lora_a for i, m in enumerate(modules)}
    weights.update({f"layer_{i}.lora_b": m.lora_b for i, m in enumerate(modules)})
    mx.eval(weights)
    mx.save_safetensors(str(out_dir / "polar.safetensors"), weights)
    cleanup(model, tokenizer, modules)
    return {k: v for k, v in stats.items() if k != "losses"}, state, str(out_dir / "polar.safetensors")


def eval_with_state(label: str, state: list[dict]) -> dict:
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
    log(f"  [Eval {label}]: GSM8K={g:.1f}%  HumanEval={h:.1f}%  MedQA={d:.1f}%")
    cleanup(model, tokenizer, modules)
    return {"gsm8k": round(g, 1), "humaneval": round(h, 1), "medqa": round(d, 1)}


def main():
    t0 = time.time()
    log_memory("start")
    log(f"=== Strategy Full Corpus N=4 (SMOKE={IS_SMOKE}, stop_at={STOP_AT_LOSS}) ===")

    log("\n[Phase 0] Export 4 strategy datasets from live Turso")
    manifests = {}
    for s in STRATEGIES:
        out = EXPERIMENT_DIR / f"data_{s}"
        m = export_split(out_dir=out, quality="approved", traj_type=s, val_frac=0.1, seed=SEED)
        manifests[s] = m
        log(f"  {s}: train={m['n_train']}, val={m['n_valid']}")

    log("\n[Phase 1] Base eval (3 benchmarks)")
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

    log("\n[Phase 2] Train 4 strategies with early-stopping")
    train_stats = {}
    states = {}
    weights_paths = {}
    for s in STRATEGIES:
        recs = load_jsonl(EXPERIMENT_DIR / f"data_{s}" / "train.jsonl")
        # Cap steps so smaller buckets don't waste compute, but allow early-stop to fire
        local_max = min(N_STEPS_MAX, len(recs) * 3)
        stats, state, wpath = train_strategy(s, recs, local_max)
        train_stats[s] = stats
        states[s] = state
        weights_paths[s] = wpath

    log("\n[Phase 3] Per-strategy eval (single adapter active)")
    per_strategy = {s: eval_with_state(s, states[s]) for s in STRATEGIES}

    # KCs
    BENCH = ["gsm8k", "humaneval", "medqa"]
    benchmarks = {b: base[b] for b in BENCH}

    # K2094: each strategy passes ≥2 of 3 benchmarks at +3pp
    k2094_per = {}
    for s in STRATEGIES:
        improvements = {b: per_strategy[s][b] - benchmarks[b] for b in BENCH}
        n_passing = sum(1 for v in improvements.values() if v >= 3.0)
        k2094_per[s] = {"pass": n_passing >= 2, "n_passing": n_passing,
                         "improvements_pp": {b: round(v, 1) for b, v in improvements.items()}}
    k2094 = all(v["pass"] for v in k2094_per.values())

    # K2095: final loss in [0.05, 0.15] window
    k2095_per = {s: {"pass": 0.05 <= train_stats[s]["final_loss"] <= 0.15,
                     "final_loss": train_stats[s]["final_loss"]} for s in STRATEGIES}
    k2095 = all(v["pass"] for v in k2095_per.values())

    # K2096: Stiefel for all 4
    stiefel = {s: {"A": train_stats[s]["stiefel_max_A"], "B": train_stats[s]["stiefel_max_B"]} for s in STRATEGIES}
    k2096 = all(v["A"] < 0.01 and v["B"] < 0.01 for v in stiefel.values())

    # K2097: gains beat exp_beehive_polar_strategy_n3 on ≥2 of 3 per strategy that has reference
    k2097_per = {}
    for s in ["prepare", "act", "integrate"]:
        ref = N3_REFERENCE[s]
        wins = sum(1 for b in BENCH if per_strategy[s][b] >= ref[b])
        k2097_per[s] = {"pass": wins >= 2, "wins_vs_n3": wins, "ref": ref, "current": per_strategy[s]}
    k2097 = all(v["pass"] for v in k2097_per.values())

    all_pass = k2094 and k2095 and k2096 and k2097
    verdict = "PROVISIONAL" if IS_SMOKE else ("SUPPORTED" if all_pass else "KILLED")

    results = {
        "is_smoke": IS_SMOKE, "stop_at_loss": STOP_AT_LOSS,
        "manifests": {s: {k: m[k] for k in ["n_train", "n_valid", "stratification"]} for s, m in manifests.items()},
        "train_stats": train_stats,
        "weights_paths": weights_paths,
        "base": base,
        "per_strategy": per_strategy,
        "kill_criteria": {
            "K2094_each_strategy_2of3": {"pass": k2094, "per_strategy": k2094_per},
            "K2095_loss_window_0.05-0.15": {"pass": k2095, "per_strategy": k2095_per},
            "K2096_joint_stiefel": {"pass": k2096, "per_strategy": stiefel},
            "K2097_beats_n3_at_scale": {"pass": k2097, "per_strategy": k2097_per},
        },
        "verdict": verdict, "all_pass": all_pass,
        "total_time_s": round(time.time() - t0, 1),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    log(f"\nK2094 each strategy ≥2/3 +3pp: {'PASS' if k2094 else 'FAIL'}")
    for s, v in k2094_per.items(): log(f"  {s}: {v}")
    log(f"K2095 loss in [0.05,0.15]:     {'PASS' if k2095 else 'FAIL'}  {k2095_per}")
    log(f"K2096 Stiefel < 0.01:          {'PASS' if k2096 else 'FAIL'}")
    log(f"K2097 beats n3 (≥2/3):         {'PASS' if k2097 else 'FAIL'}  {k2097_per}")
    log(f"\nVERDICT: {verdict}")
    log(f"Adapters saved: adapters/strategy_{{full,prepare,act,integrate}}_polar/polar.safetensors")
    log(f"Total: {results['total_time_s']:.0f}s")


if __name__ == "__main__":
    main()
