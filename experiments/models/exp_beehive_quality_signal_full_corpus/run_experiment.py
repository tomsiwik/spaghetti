#!/usr/bin/env python3
"""
Beehive quality signal validation on FULL Turso corpus (2069 approved + 24 rejected)
with non-tautological principle-following scorer.

Re-confirms exp_beehive_quality_signal_validation (which used 5% of data) at full scale.

Kill criteria:
  K2090: Approved-only PoLAR > rejected-only on principle eval (Δ ≥ 5pp)
  K2091: Approved-only PoLAR > random-mix on principle eval (Δ ≥ 3pp)
  K2092: Approved-only improves base by ≥3pp on at least 1 of GSM8K/HumanEval/MedQA
  K2093: Base-model keyword_recall < 0.5 (eval is non-tautological)
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

from tooling.scripts.beehive_to_mlx import export_split, fetch_rows
from tooling.scripts.polar_train import (
    inject_polar_adapters, train, cleanup,
    eval_gsm8k, eval_humaneval, eval_medqa, eval_principle_following,
    RANK, SCALE,
)

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42

N_STEPS = 30 if IS_SMOKE else 1500     # 1500 steps × bs=2 ≈ ~1.5 epochs over 1862 train rows
N_BENCH_EVAL = 5 if IS_SMOKE else 30
N_PRINCIPLE_EVAL = 5 if IS_SMOKE else 50  # bigger now: 207 valid available


def log(m): print(m, flush=True)


def log_memory(label=""):
    a = mx.get_active_memory() / 1e9
    c = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB")


def load_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def export_random_mix(out_dir: Path, val_frac: float, seed: int) -> dict:
    """50/50 mix of approved + rejected. Note: rejected is small (24), so total mix ~48 rows."""
    import random as _r
    from tooling.scripts.beehive_to_mlx import _stratified_split, _to_messages_record, _stratify_summary, _score_stats

    approved = fetch_rows(quality="approved")
    rejected = fetch_rows(quality="rejected")
    n = min(len(approved), len(rejected))
    rng = _r.Random(seed)
    rng.shuffle(approved); rng.shuffle(rejected)
    mix = approved[:n] + rejected[:n]   # full balanced (24+24=48 here)
    rng.shuffle(mix)

    train_rows, val_rows = _stratified_split(mix, val_frac, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for items, fname in [(train_rows, "train.jsonl"), (val_rows, "valid.jsonl")]:
        with open(out_dir / fname, "w") as f:
            for r in items:
                rec = _to_messages_record(r)
                f.write(json.dumps({"messages": rec["messages"]}) + "\n")
    manifest = {
        "filters": {"quality": "balanced-mix"},
        "n_total": len(mix), "n_train": len(train_rows), "n_valid": len(val_rows),
        "stratification": _stratify_summary(train_rows, val_rows),
        "score_stats": _score_stats(mix),
        "valid_ids": [r.id for r in val_rows], "train_ids": [r.id for r in train_rows],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def train_adapter(label: str, train_records: list[dict], n_steps: int) -> tuple[dict, str]:
    log(f"\n[Train: {label}, n={len(train_records)}]")
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
    log_memory(f"polar-injected-{label}")

    stats = train(model, tokenizer, train_records, modules, n_steps, stop_at_loss=0.1)
    log(f"  loss: {stats['first_loss']:.4f} → {stats['final_loss']:.4f}, Stiefel A={stats['stiefel_max_A']:.4f} B={stats['stiefel_max_B']:.4f}, n_actual_steps={len(stats['losses'])}")

    weights = {f"layer_{i}.lora_a": m.lora_a for i, m in enumerate(modules)}
    weights.update({f"layer_{i}.lora_b": m.lora_b for i, m in enumerate(modules)})
    mx.eval(weights)
    out = EXPERIMENT_DIR / "adapter_weights" / f"{label}.safetensors"
    out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out), weights)
    cleanup(model, tokenizer, modules)
    return {k: v for k, v in stats.items() if k != "losses"}, str(out)


def eval_adapter(label: str, weights_path: str, valid_ids: list[int]) -> dict:
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
    raw = mx.load(weights_path)
    for i, m in enumerate(modules):
        m.lora_a = raw[f"layer_{i}.lora_a"]
        m.lora_b = raw[f"layer_{i}.lora_b"]
    mx.eval(model.parameters())

    log(f"\n[Eval: {label}]")
    p = eval_principle_following(model, tokenizer, valid_ids, N_PRINCIPLE_EVAL)
    g = eval_gsm8k(model, tokenizer, N_BENCH_EVAL)
    h = eval_humaneval(model, tokenizer, N_BENCH_EVAL)
    d = eval_medqa(model, tokenizer, N_BENCH_EVAL)
    log(f"  {label}: principle={p['aggregate_mean']:.1f}% GSM8K={g:.1f}% HumanEval={h:.1f}% MedQA={d:.1f}%")
    cleanup(model, tokenizer, modules)
    return {"principle": p, "gsm8k": round(g, 1), "humaneval": round(h, 1), "medqa": round(d, 1)}


def main():
    t0 = time.time()
    log_memory("start")
    log(f"=== Quality Signal Full Corpus (SMOKE={IS_SMOKE}) ===")

    log("\n[Phase 0] Export 3 datasets from live Turso")
    approved_dir = EXPERIMENT_DIR / "data_approved"
    rejected_dir = EXPERIMENT_DIR / "data_rejected"
    mix_dir = EXPERIMENT_DIR / "data_mix"
    m_app = export_split(out_dir=approved_dir, quality="approved", val_frac=0.1, seed=SEED)
    m_rej = export_split(out_dir=rejected_dir, quality="rejected", val_frac=0.2, seed=SEED)
    m_mix = export_random_mix(mix_dir, val_frac=0.1, seed=SEED)
    log(f"  approved: train={m_app['n_train']} val={m_app['n_valid']}")
    log(f"  rejected: train={m_rej['n_train']} val={m_rej['n_valid']}")
    log(f"  mix:      train={m_mix['n_train']} val={m_mix['n_valid']}")

    canonical_valid_ids = m_app["valid_ids"]

    # Phase 1: base
    log("\n[Phase 1] Base model on canonical principle eval")
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    base_p = eval_principle_following(model, tokenizer, canonical_valid_ids, N_PRINCIPLE_EVAL)
    base_g = eval_gsm8k(model, tokenizer, N_BENCH_EVAL)
    base_h = eval_humaneval(model, tokenizer, N_BENCH_EVAL)
    base_d = eval_medqa(model, tokenizer, N_BENCH_EVAL)
    log(f"  base: principle={base_p['aggregate_mean']:.1f}% (kw_recall={base_p['keyword_mean']:.1f}) GSM8K={base_g:.1f}% HumanEval={base_h:.1f}% MedQA={base_d:.1f}%")
    cleanup(model, tokenizer)

    # Phase 2: train 3 adapters
    train_results = {}
    paths = {}
    for label, jsonl in [
        ("approved", approved_dir / "train.jsonl"),
        ("rejected", rejected_dir / "train.jsonl"),
        ("mix", mix_dir / "train.jsonl"),
    ]:
        recs = load_jsonl(jsonl)
        # Adjust n_steps for small training sets: cap to ≤5 epochs
        local_steps = min(N_STEPS, len(recs) * 5)
        stats, wpath = train_adapter(label, recs, local_steps)
        train_results[label] = stats
        paths[label] = wpath

    # Phase 3: eval each
    eval_results = {label: eval_adapter(label, paths[label], canonical_valid_ids) for label in paths}

    # Compute KCs
    p = lambda x: eval_results[x]["principle"]["aggregate_mean"]
    delta_app_rej = p("approved") - p("rejected")
    delta_app_mix = p("approved") - p("mix")
    bench_delta = max(
        eval_results["approved"]["gsm8k"] - base_g,
        eval_results["approved"]["humaneval"] - base_h,
        eval_results["approved"]["medqa"] - base_d,
    )
    base_kw_recall = base_p["keyword_mean"] / 100  # back to [0,1]

    k2090 = delta_app_rej >= 5
    k2091 = delta_app_mix >= 3
    k2092 = bench_delta >= 3
    k2093 = base_kw_recall < 0.5

    all_pass = k2090 and k2091 and k2092 and k2093
    verdict = "PROVISIONAL" if IS_SMOKE else ("SUPPORTED" if all_pass else "KILLED")

    results = {
        "is_smoke": IS_SMOKE, "n_steps": N_STEPS,
        "manifests": {k: {"n_train": m["n_train"], "n_valid": m["n_valid"], "stratification": m["stratification"]}
                      for k, m in [("approved", m_app), ("rejected", m_rej), ("mix", m_mix)]},
        "train_stats": train_results,
        "base": {"principle": base_p, "gsm8k": round(base_g,1), "humaneval": round(base_h,1), "medqa": round(base_d,1)},
        "adapter": eval_results,
        "deltas": {
            "approved_minus_rejected_principle": round(delta_app_rej, 1),
            "approved_minus_mix_principle": round(delta_app_mix, 1),
            "approved_best_benchmark_delta_pp": round(bench_delta, 1),
        },
        "kill_criteria": {
            "K2090_approved_beats_rejected": {"pass": k2090, "delta_pp": round(delta_app_rej, 1)},
            "K2091_approved_beats_mix": {"pass": k2091, "delta_pp": round(delta_app_mix, 1)},
            "K2092_approved_beats_base_3pp": {"pass": k2092, "best_benchmark_delta_pp": round(bench_delta, 1)},
            "K2093_eval_non_tautological": {"pass": k2093, "base_kw_recall": round(base_kw_recall, 3)},
        },
        "verdict": verdict, "all_pass": all_pass,
        "total_time_s": round(time.time() - t0, 1),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    log("\n=== KCs ===")
    log(f"K2090 approved>rejected ≥5pp:  {'PASS' if k2090 else 'FAIL'} (Δ={delta_app_rej:+.1f})")
    log(f"K2091 approved>mix ≥3pp:        {'PASS' if k2091 else 'FAIL'} (Δ={delta_app_mix:+.1f})")
    log(f"K2092 approved>base on 1 bench: {'PASS' if k2092 else 'FAIL'} (best={bench_delta:+.1f})")
    log(f"K2093 eval non-tautological:    {'PASS' if k2093 else 'FAIL'} (kw={base_kw_recall:.3f})")
    log(f"\nVERDICT: {verdict}")
    log(f"Total: {results['total_time_s']:.0f}s")


if __name__ == "__main__":
    main()
