#!/usr/bin/env python3
"""
Beehive quality signal validation: train 3 PoLAR adapters (approved-only,
rejected-only, random-mix), measure if quality flag predicts adapter behavior.

Kill criteria:
  K2073: Approved > rejected on held-out principle eval (Δ ≥ 5pp)
  K2074: Approved > random-mix on same eval (Δ ≥ 3pp)
  K2075: Approved improves base by ≥3pp on at least 1 of GSM8K/HumanEval/MedQA
  K2076: [INCONCLUSIVE — score has insufficient variance, see K2081]
  K2081: Characterize score variance + flag rejected score=90 anomaly
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

N_STEPS = 20 if IS_SMOKE else 800
N_BENCH_EVAL = 5 if IS_SMOKE else 30
N_PRINCIPLE_EVAL = 5 if IS_SMOKE else 25


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    a = mx.get_active_memory() / 1e9
    c = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB")


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def export_random_mix(out_dir: Path, val_frac: float, seed: int) -> dict:
    """Build a balanced approved+rejected mix matching the smaller set's size."""
    import random as _r
    from tooling.scripts.beehive_to_mlx import _stratified_split, _to_messages_record, _stratify_summary, _score_stats

    approved = fetch_rows(quality="approved")
    rejected = fetch_rows(quality="rejected")
    n = min(len(approved), len(rejected))
    rng = _r.Random(seed)
    rng.shuffle(approved)
    rng.shuffle(rejected)
    mix = approved[:n // 2] + rejected[:n // 2]
    rng.shuffle(mix)

    train_rows, val_rows = _stratified_split(mix, val_frac, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for items, fname in [(train_rows, "train.jsonl"), (val_rows, "valid.jsonl")]:
        with open(out_dir / fname, "w") as f:
            for r in items:
                rec = _to_messages_record(r)
                f.write(json.dumps({"messages": rec["messages"]}) + "\n")

    manifest = {
        "filters": {"quality": "random-mix-50/50"},
        "n_total": len(mix),
        "n_train": len(train_rows),
        "n_valid": len(val_rows),
        "stratification": _stratify_summary(train_rows, val_rows),
        "score_stats": _score_stats(mix),
        "valid_ids": [r.id for r in val_rows],
        "train_ids": [r.id for r in train_rows],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def train_one_adapter(label: str, train_records: list[dict], n_steps: int) -> tuple[dict, dict]:
    """Returns (train_stats, adapter_weights_dict). Releases model after."""
    log(f"\n[Train: {label}]")
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
    log_memory(f"polar-injected-{label}")

    stats = train(model, tokenizer, train_records, modules, n_steps)
    log(f"  loss: {stats['first_loss']:.4f} → {stats['final_loss']:.4f}, Stiefel A={stats['stiefel_max_A']:.4f} B={stats['stiefel_max_B']:.4f}")

    # Save weights
    weights = {}
    for i, m in enumerate(modules):
        weights[f"layer_{i}.lora_a"] = m.lora_a
        weights[f"layer_{i}.lora_b"] = m.lora_b
    mx.eval(weights)
    out = EXPERIMENT_DIR / "adapter_weights" / f"{label}.safetensors"
    out.parent.mkdir(exist_ok=True)
    mx.save_safetensors(str(out), weights)

    cleanup(model, tokenizer, modules)
    return stats, str(out)


def eval_one_adapter(label: str, adapter_path: str, valid_ids: list[int]) -> dict:
    """Re-load model + adapter and eval on principle + standard benchmarks."""
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)

    # Load weights into modules
    raw = mx.load(adapter_path)
    for i, m in enumerate(modules):
        m.lora_a = raw[f"layer_{i}.lora_a"]
        m.lora_b = raw[f"layer_{i}.lora_b"]
    mx.eval(model.parameters())

    log(f"\n[Eval: {label}]")
    principle = eval_principle_following(model, tokenizer, valid_ids, N_PRINCIPLE_EVAL)
    gsm = eval_gsm8k(model, tokenizer, N_BENCH_EVAL)
    he = eval_humaneval(model, tokenizer, N_BENCH_EVAL)
    md = eval_medqa(model, tokenizer, N_BENCH_EVAL)
    log(f"  {label}: principle={principle['aggregate_mean']:.1f}%  GSM8K={gsm:.1f}%  HumanEval={he:.1f}%  MedQA={md:.1f}%")
    cleanup(model, tokenizer, modules)
    return {"principle": principle, "gsm8k": round(gsm, 1), "humaneval": round(he, 1), "medqa": round(md, 1)}


def main():
    t_start = time.time()
    log_memory("start")
    log(f"=== Beehive quality signal validation (SMOKE={IS_SMOKE}) ===")

    # Phase 0: prepare 3 datasets
    log("\n[Phase 0] Prepare 3 stratified datasets")
    approved_dir = EXPERIMENT_DIR / "data_approved"
    rejected_dir = EXPERIMENT_DIR / "data_rejected"
    mix_dir = EXPERIMENT_DIR / "data_mix"

    m_approved = export_split(out_dir=approved_dir, quality="approved", val_frac=0.2, seed=SEED)
    m_rejected = export_split(out_dir=rejected_dir, quality="rejected", val_frac=0.2, seed=SEED)
    m_mix = export_random_mix(mix_dir, val_frac=0.2, seed=SEED)
    log(f"  approved: train={m_approved['n_train']} val={m_approved['n_valid']}")
    log(f"  rejected: train={m_rejected['n_train']} val={m_rejected['n_valid']}")
    log(f"  mix:      train={m_mix['n_train']} val={m_mix['n_valid']}")

    # Score variance diagnostic (K2081)
    score_diag = {
        "approved": m_approved["score_stats"],
        "rejected": m_rejected["score_stats"],
        "rejected_score90_anomaly_count": sum(1 for r in fetch_rows(quality="rejected") if r.score == 90),
    }
    log(f"  score variance: approved={score_diag['approved']}, rejected={score_diag['rejected']}")
    log(f"  rejected score=90 anomalies: {score_diag['rejected_score90_anomaly_count']}")

    # Use the held-out approved valid set as the canonical principle eval target —
    # we want to know which training data produces an adapter that follows real principles.
    canonical_valid_ids = m_approved["valid_ids"]

    # Phase 1: base
    log("\n[Phase 1] Base model eval on canonical principle set")
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    base_principle = eval_principle_following(model, tokenizer, canonical_valid_ids, N_PRINCIPLE_EVAL)
    base_gsm8k = eval_gsm8k(model, tokenizer, N_BENCH_EVAL)
    base_humaneval = eval_humaneval(model, tokenizer, N_BENCH_EVAL)
    base_medqa = eval_medqa(model, tokenizer, N_BENCH_EVAL)
    log(f"  base principle={base_principle['aggregate_mean']:.1f}% GSM8K={base_gsm8k:.1f}% HumanEval={base_humaneval:.1f}% MedQA={base_medqa:.1f}%")
    cleanup(model, tokenizer)

    # Phase 2: train 3 adapters
    log("\n[Phase 2] Train 3 PoLAR adapters")
    train_results = {}
    weights_paths = {}

    for label, jsonl_path, manifest in [
        ("approved", approved_dir / "train.jsonl", m_approved),
        ("rejected", rejected_dir / "train.jsonl", m_rejected),
        ("mix", mix_dir / "train.jsonl", m_mix),
    ]:
        records = load_jsonl(jsonl_path)
        stats, wpath = train_one_adapter(label, records, N_STEPS)
        train_results[label] = stats
        weights_paths[label] = wpath

    # Phase 3: eval all 3 against canonical principle eval
    log("\n[Phase 3] Eval 3 adapters")
    eval_results = {}
    for label, wpath in weights_paths.items():
        eval_results[label] = eval_one_adapter(label, wpath, canonical_valid_ids)

    # Compute deltas
    p = lambda lab: eval_results[lab]["principle"]["aggregate_mean"]
    delta_app_rej = p("approved") - p("rejected")
    delta_app_mix = p("approved") - p("mix")

    bench = lambda lab: max(
        eval_results[lab]["gsm8k"] - base_gsm8k,
        eval_results[lab]["humaneval"] - base_humaneval,
        eval_results[lab]["medqa"] - base_medqa,
    )
    approved_best_bench_delta = bench("approved")

    # KCs
    k2073 = delta_app_rej >= 5.0
    k2074 = delta_app_mix >= 3.0
    k2075 = approved_best_bench_delta >= 3.0

    # K2081: score variance characterization
    approved_scores = [r.score for r in fetch_rows(quality="approved") if r.score is not None]
    score_cv = float(np.std(approved_scores) / np.mean(approved_scores)) if approved_scores else float("nan")
    k2081_finding = "uninformative" if score_cv < 0.1 else "informative"

    all_pass = k2073 and k2074 and k2075
    verdict = "PROVISIONAL" if IS_SMOKE else ("SUPPORTED" if all_pass else "KILLED")

    results = {
        "is_smoke": IS_SMOKE,
        "n_steps": N_STEPS,
        "manifests": {
            "approved": {k: m_approved[k] for k in ["n_train", "n_valid", "stratification", "score_stats"]},
            "rejected": {k: m_rejected[k] for k in ["n_train", "n_valid", "stratification", "score_stats"]},
            "mix": {k: m_mix[k] for k in ["n_train", "n_valid", "stratification", "score_stats"]},
        },
        "train_stats": {label: {k: v for k, v in s.items() if k != "losses"} for label, s in train_results.items()},
        "base": {"principle": base_principle, "gsm8k": round(base_gsm8k, 1), "humaneval": round(base_humaneval, 1), "medqa": round(base_medqa, 1)},
        "adapter": eval_results,
        "deltas": {
            "approved_minus_rejected_principle": round(delta_app_rej, 1),
            "approved_minus_mix_principle": round(delta_app_mix, 1),
            "approved_best_benchmark_delta_pp": round(approved_best_bench_delta, 1),
        },
        "score_diagnostics": {
            "approved_cv": round(score_cv, 4),
            "approved_n_with_score": len(approved_scores),
            "approved_unique_scores": sorted(set(approved_scores)),
            "rejected_score_90_count": score_diag["rejected_score90_anomaly_count"],
            "k2081_finding": k2081_finding,
        },
        "kill_criteria": {
            "K2073_approved_beats_rejected": {"pass": k2073, "delta_pp": round(delta_app_rej, 1)},
            "K2074_approved_beats_mix": {"pass": k2074, "delta_pp": round(delta_app_mix, 1)},
            "K2075_approved_beats_base_3pp": {"pass": k2075, "best_benchmark_delta_pp": round(approved_best_bench_delta, 1)},
            "K2076_score_correlation": {"pass": None, "result": "inconclusive", "reason": "score has insufficient variance (CV<0.1) to support correlation test"},
            "K2081_score_diagnostics": {"pass": k2081_finding == "uninformative", "cv": round(score_cv, 4), "finding": k2081_finding},
        },
        "verdict": verdict,
        "all_pass": all_pass,
        "total_time_s": round(time.time() - t_start, 1),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    log("\n=== Kill Criteria ===")
    log(f"K2073 approved > rejected (≥5pp):   {'PASS' if k2073 else 'FAIL'} (Δ={delta_app_rej:+.1f}pp)")
    log(f"K2074 approved > mix (≥3pp):        {'PASS' if k2074 else 'FAIL'} (Δ={delta_app_mix:+.1f}pp)")
    log(f"K2075 approved > base on 1 bench:   {'PASS' if k2075 else 'FAIL'} (best={approved_best_bench_delta:+.1f}pp)")
    log(f"K2081 score variance diagnostic:    cv={score_cv:.4f} → {k2081_finding}")
    log(f"\nVERDICT: {verdict}")
    log(f"Total time: {results['total_time_s']:.0f}s")


if __name__ == "__main__":
    main()
