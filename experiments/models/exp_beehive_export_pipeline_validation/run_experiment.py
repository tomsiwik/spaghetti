#!/usr/bin/env python3
"""
Beehive→mlx_lm pipeline validation: train one PoLAR adapter on all approved
beehive trajectories, verify convergence, no benchmark regression, principle
learning, and Stiefel maintenance.

Kill criteria:
  K2069: loss decreases monotonically; no NaN/divergence
  K2070: no >2pp regression on any of GSM8K/HumanEval/MedQA
  K2071: principle-following ≥ base + 5pp on held-out beehive eval
  K2072: max ||A^T A − I||_F < 0.01 AND max ||B B^T − I||_F < 0.01
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
    eval_gsm8k, eval_humaneval, eval_medqa, eval_principle_following,
    RANK, SCALE,
)

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42

N_STEPS = 20 if IS_SMOKE else 1000
N_BENCH_EVAL = 5 if IS_SMOKE else 30
N_PRINCIPLE_EVAL = 5 if IS_SMOKE else 21


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    a = mx.get_active_memory() / 1e9
    c = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB")


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    t_start = time.time()
    log_memory("start")
    log(f"=== Beehive→mlx_lm pipeline validation (SMOKE={IS_SMOKE}) ===")

    # Phase 0
    log("\n[Phase 0] Export beehive approved trajectories")
    data_dir = EXPERIMENT_DIR / "data"
    manifest = export_split(out_dir=data_dir, quality="approved", val_frac=0.2, seed=SEED)
    log(f"  n_train={manifest['n_train']}, n_valid={manifest['n_valid']}, valid_by_type={manifest['stratification']['valid_by_type']}")
    valid_ids = manifest["valid_ids"]
    train_records = load_jsonl(data_dir / "train.jsonl")

    # Phase 1: base
    log("\n[Phase 1] Base model evaluation")
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    log_memory("base-loaded")

    base_gsm8k = eval_gsm8k(model, tokenizer, N_BENCH_EVAL)
    base_humaneval = eval_humaneval(model, tokenizer, N_BENCH_EVAL)
    base_medqa = eval_medqa(model, tokenizer, N_BENCH_EVAL)
    base_principle = eval_principle_following(model, tokenizer, valid_ids, N_PRINCIPLE_EVAL)
    log(f"  base GSM8K={base_gsm8k:.1f}%  HumanEval={base_humaneval:.1f}%  MedQA={base_medqa:.1f}%")
    log(f"  base principle={base_principle['aggregate_mean']:.1f}% (fmt={base_principle['format_mean']:.1f}, kw={base_principle['keyword_mean']:.1f}, struct={base_principle['structure_mean']:.1f})")
    cleanup(model, tokenizer)

    # Phase 2: train
    log("\n[Phase 2] Train PoLAR on approved beehive corpus")
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
    log(f"  injected {len(modules)} PoLAR adapters on q_proj")
    log_memory("polar-injected")

    stats = train(model, tokenizer, train_records, modules, N_STEPS)
    log(f"  loss: {stats['first_loss']:.4f} → {stats['final_loss']:.4f}")
    log(f"  Stiefel: max_A={stats['stiefel_max_A']:.4f}, max_B={stats['stiefel_max_B']:.4f}")

    # Phase 3: trained eval
    log("\n[Phase 3] Trained adapter eval")
    a_gsm8k = eval_gsm8k(model, tokenizer, N_BENCH_EVAL)
    a_humaneval = eval_humaneval(model, tokenizer, N_BENCH_EVAL)
    a_medqa = eval_medqa(model, tokenizer, N_BENCH_EVAL)
    a_principle = eval_principle_following(model, tokenizer, valid_ids, N_PRINCIPLE_EVAL)
    log(f"  adapter GSM8K={a_gsm8k:.1f}%  HumanEval={a_humaneval:.1f}%  MedQA={a_medqa:.1f}%")
    log(f"  adapter principle={a_principle['aggregate_mean']:.1f}%")

    # Save adapter
    adapter_out = EXPERIMENT_DIR / "adapter_weights"
    adapter_out.mkdir(exist_ok=True)
    weights = {}
    for i, m in enumerate(modules):
        weights[f"layer_{i}.lora_a"] = m.lora_a
        weights[f"layer_{i}.lora_b"] = m.lora_b
    mx.eval(weights)
    mx.save_safetensors(str(adapter_out / "polar_approved.safetensors"), weights)

    cleanup(model, tokenizer)

    # KCs
    log("\n=== Kill Criteria ===")
    k2069 = (not stats["any_nan"]) and (stats["final_loss"] < stats["first_loss"]) and (stats["monotonic_break_step"] is None or stats["monotonic_break_step"] > N_STEPS - 100)
    drops = {"gsm8k": base_gsm8k - a_gsm8k, "humaneval": base_humaneval - a_humaneval, "medqa": base_medqa - a_medqa}
    k2070 = all(d <= 2.0 for d in drops.values())
    p_delta = a_principle["aggregate_mean"] - base_principle["aggregate_mean"]
    k2071 = p_delta >= 5.0
    k2072 = stats["stiefel_max_A"] < 0.01 and stats["stiefel_max_B"] < 0.01

    all_pass = k2069 and k2070 and k2071 and k2072
    verdict = "PROVISIONAL" if IS_SMOKE else ("SUPPORTED" if all_pass else "KILLED")

    results = {
        "is_smoke": IS_SMOKE,
        "n_steps": N_STEPS,
        "manifest": {k: manifest[k] for k in ["n_train", "n_valid", "stratification", "score_stats"]},
        "train_stats": {k: v for k, v in stats.items() if k != "losses"},
        "loss_curve_summary": {
            "first": stats["first_loss"],
            "final": stats["final_loss"],
            "min": min(stats["losses"]) if stats["losses"] else None,
            "max": max(stats["losses"]) if stats["losses"] else None,
            "n_steps": len(stats["losses"]),
        },
        "base": {"gsm8k": round(base_gsm8k, 1), "humaneval": round(base_humaneval, 1), "medqa": round(base_medqa, 1), "principle": base_principle},
        "adapter": {"gsm8k": round(a_gsm8k, 1), "humaneval": round(a_humaneval, 1), "medqa": round(a_medqa, 1), "principle": a_principle},
        "kill_criteria": {
            "K2069_loss_monotonic": {"pass": k2069, "any_nan": stats["any_nan"], "first": stats["first_loss"], "final": stats["final_loss"], "break_step": stats["monotonic_break_step"]},
            "K2070_no_regression": {"pass": k2070, "drops_pp": {k: round(v, 1) for k, v in drops.items()}},
            "K2071_principle_5pp": {"pass": k2071, "delta_pp": round(p_delta, 1), "base": base_principle["aggregate_mean"], "adapter": a_principle["aggregate_mean"]},
            "K2072_stiefel": {"pass": k2072, "max_A": stats["stiefel_max_A"], "max_B": stats["stiefel_max_B"]},
        },
        "verdict": verdict,
        "all_pass": all_pass,
        "total_time_s": round(time.time() - t_start, 1),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    log(f"\nK2069 loss monotonic:    {'PASS' if k2069 else 'FAIL'}")
    log(f"K2070 no regression ≤2pp: {'PASS' if k2070 else 'FAIL'} (drops={drops})")
    log(f"K2071 principle ≥+5pp:    {'PASS' if k2071 else 'FAIL'} (Δ={p_delta:+.1f}pp)")
    log(f"K2072 joint Stiefel:      {'PASS' if k2072 else 'FAIL'} (A={stats['stiefel_max_A']:.4f}, B={stats['stiefel_max_B']:.4f})")
    log(f"\nVERDICT: {verdict}")
    log(f"Total time: {results['total_time_s']:.0f}s")


if __name__ == "__main__":
    main()
