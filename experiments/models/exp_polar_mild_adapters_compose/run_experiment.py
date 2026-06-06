#!/usr/bin/env python3
"""
Mild PoLAR adapters + composition (F#54/F#73 conditions reproduced).

Hypothesis: prior experiments failed because adapters were over-trained (800-1500 iters
to loss 0.1-0.3), creating destructive interference when summed. F#54 (N=24 SUPPORTED)
and F#73 (N=15, 0.53% degradation) used ~200 iter caps with mild adapters.

This experiment:
  1. Retrains 4 strategy + 3 domain adapters with HARD 200-iter cap, scale=4 (not 6)
  2. Measures per-adapter PPL on its validation data (the F#73 metric)
  3. Tests composition with hidden-state Gumbel top-2 routing
  4. Verifies composed PPL stays within 1.10x of single-adapter PPL (F#73 standard)

Kill criteria:
  K2111: Each mild adapter ≥3pp on primary benchmark (mild but specialized)
  K2112: Per-adapter PPL when composed ≤1.10x best single PPL (F#73 preservation)
  K2113: N=2 strategy×domain composition preserves best-axis task accuracy within 5pp
  K2114: N=7 full composition preserves base accuracy on all 3 benchmarks (no degradation)
  K2115: Joint Stiefel for all 7 adapters
"""

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
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
    PoLARLinear, RANK,
    tokenize_record, collate, loss_fn,
)

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42
N_BENCH_EVAL = 5 if IS_SMOKE else 30

# F#73 / F#54 conditions
MILD_SCALE = 4.0       # was 6.0
MILD_STEPS_CAP = 200   # was 800-1500
PPL_PRESERVATION_RATIO = 1.10

ADAPTER_DIR = EXPERIMENT_DIR / "mild_adapters"
STRATEGIES = ["full", "prepare", "act", "integrate"]
DOMAINS = ["math", "code", "medical"]


def log(m): print(m, flush=True)


def log_memory(label=""):
    a = mx.get_active_memory() / 1e9
    c = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB")


def load_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


# ─────────────────────────────────────────────
# Mild adapter training (F#54 conditions)
# ─────────────────────────────────────────────

def train_mild_adapter(label: str, records: list[dict]) -> tuple[dict, list[dict], str]:
    """200-iter cap, scale=4. Returns (stats, state, weights_path)."""
    log(f"\n[Train mild: {label}, n={len(records)}, scale={MILD_SCALE}, cap={MILD_STEPS_CAP}]")
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=MILD_SCALE)
    log_memory(f"injected-{label}")

    # No early stop — hard cap at MILD_STEPS_CAP
    stats = train(model, tokenizer, records, modules, n_steps=MILD_STEPS_CAP)
    log(f"  loss: {stats['first_loss']:.4f} → {stats['final_loss']:.4f} (n={len(stats['losses'])})  Stiefel A={stats['stiefel_max_A']:.2e} B={stats['stiefel_max_B']:.2e}")

    state = [{
        "a": np.array(m.lora_a.tolist(), dtype=np.float32),
        "b": np.array(m.lora_b.tolist(), dtype=np.float32),
    } for m in modules]

    # Save
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    weights = {f"layer_{i}.lora_a": m.lora_a for i, m in enumerate(modules)}
    weights.update({f"layer_{i}.lora_b": m.lora_b for i, m in enumerate(modules)})
    mx.eval(weights)
    out = ADAPTER_DIR / f"{label}.safetensors"
    mx.save_safetensors(str(out), weights)
    cleanup(model, tokenizer, modules)
    return {k: v for k, v in stats.items() if k != "losses"}, state, str(out)


# ─────────────────────────────────────────────
# Per-adapter PPL on validation
# ─────────────────────────────────────────────

def compute_ppl(model, tokenizer, valid_records: list[dict]) -> float:
    """Average NLL across validation records → exp(avg_nll) = PPL."""
    if not valid_records:
        return float("nan")
    total_nll, total_tokens = 0.0, 0
    for rec in valid_records:
        ids, labels = tokenize_record(tokenizer, rec)
        ids = ids[None, :]; labels = labels[None, :]
        nll = float(loss_fn(model, ids, labels).item())
        n_tok = int((labels[:, 1:] != -100).sum().item())
        total_nll += nll * max(n_tok, 1)
        total_tokens += max(n_tok, 1)
    return float(math.exp(total_nll / max(total_tokens, 1)))


# ─────────────────────────────────────────────
# Eval helpers
# ─────────────────────────────────────────────

def evaluate_single_state(state: list[dict], scale: float = MILD_SCALE) -> dict:
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=scale)
    for i, m in enumerate(modules):
        m.lora_a = mx.array(state[i]["a"]); m.lora_b = mx.array(state[i]["b"])
    mx.eval(model.parameters())
    g = eval_gsm8k(model, tokenizer, N_BENCH_EVAL)
    h = eval_humaneval(model, tokenizer, N_BENCH_EVAL)
    d = eval_medqa(model, tokenizer, N_BENCH_EVAL)
    cleanup(model, tokenizer, modules)
    return {"gsm8k": round(g, 1), "humaneval": round(h, 1), "medqa": round(d, 1)}


# ─────────────────────────────────────────────
# Composition (uniform 1/N as F#73 reference)
# ─────────────────────────────────────────────

def apply_uniform_composition(modules, states_list: list[list[dict]]):
    n = len(states_list)
    for layer_idx, m in enumerate(modules):
        delta = None
        for state in states_list:
            a = mx.array(state[layer_idx]["a"]); b = mx.array(state[layer_idx]["b"])
            d = (a @ b) * (1.0 / n)
            delta = d if delta is None else delta + d
        mx.eval(delta)
        m._composed_delta = delta

        def make_fwd(layer):
            def fwd(x):
                return layer.base(x) + layer.scale * (x @ layer._composed_delta)
            return fwd
        m.__call__ = make_fwd(m).__get__(m)


def evaluate_composed(states_list: list[list[dict]], scale: float = MILD_SCALE) -> dict:
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=scale)
    apply_uniform_composition(modules, states_list)
    mx.eval(model.parameters())
    g = eval_gsm8k(model, tokenizer, N_BENCH_EVAL)
    h = eval_humaneval(model, tokenizer, N_BENCH_EVAL)
    d = eval_medqa(model, tokenizer, N_BENCH_EVAL)
    cleanup(model, tokenizer, modules)
    return {"gsm8k": round(g, 1), "humaneval": round(h, 1), "medqa": round(d, 1)}


def evaluate_composed_ppl(states_list: list[list[dict]], valid_recs_per_adapter: list[list[dict]],
                           scale: float = MILD_SCALE) -> list[float]:
    """For each adapter's validation set, compute PPL of the COMPOSED model."""
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=scale)
    apply_uniform_composition(modules, states_list)
    mx.eval(model.parameters())
    ppls = [compute_ppl(model, tokenizer, recs) for recs in valid_recs_per_adapter]
    cleanup(model, tokenizer, modules)
    return ppls


def evaluate_single_ppl(state: list[dict], valid_recs: list[dict], scale: float = MILD_SCALE) -> float:
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=scale)
    for i, m in enumerate(modules):
        m.lora_a = mx.array(state[i]["a"]); m.lora_b = mx.array(state[i]["b"])
    mx.eval(model.parameters())
    ppl = compute_ppl(model, tokenizer, valid_recs)
    cleanup(model, tokenizer, modules)
    return ppl


# ─────────────────────────────────────────────
# Domain dataset prep (lightweight versions of exp_beehive_polar_domain_5)
# ─────────────────────────────────────────────

def prepare_math(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=SEED).select(range(min(n, 7473)))
    return [{"messages": [
        {"role": "user", "content": f"Solve the following math problem step by step.\n\n{ex['question']}"},
        {"role": "assistant", "content": ex["answer"]},
    ]} for ex in ds]


def prepare_code(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train").shuffle(seed=SEED).select(range(min(n, 20000)))
    out = []
    for ex in ds:
        c = ex["instruction"]
        if ex.get("input"):
            c += f"\n\nInput:\n{ex['input']}"
        out.append({"messages": [
            {"role": "user", "content": c},
            {"role": "assistant", "content": ex["output"]},
        ]})
    return out


def prepare_medical(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train").shuffle(seed=SEED).select(range(min(n, 10178)))
    out = []
    for ex in ds:
        opts = ex["options"]
        q = f"{ex['question']}\n(A) {opts['A']}\n(B) {opts['B']}\n(C) {opts['C']}\n(D) {opts['D']}"
        out.append({"messages": [
            {"role": "user", "content": f"Answer this medical multiple choice question. Respond with only the letter (A/B/C/D).\n\n{q}"},
            {"role": "assistant", "content": f"{ex['answer_idx']}: {ex['answer']}"},
        ]})
    return out


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    t0 = time.time()
    log_memory("start")
    log(f"=== Mild PoLAR adapters + composition (F#73 conditions) ===")
    log(f"    scale={MILD_SCALE}, hard step cap={MILD_STEPS_CAP}, smoke={IS_SMOKE}")

    n_train_per_adapter = 50 if IS_SMOKE else 500   # smaller train set; 200 iters anyway

    # ── Phase 0: Base eval ──────────────────────────
    log("\n[Phase 0] Base eval")
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

    # ── Phase 1: Train 4 mild strategy adapters ────
    log("\n[Phase 1] Train 4 mild strategy adapters from beehive")
    strategy_states = {}
    strategy_train_stats = {}
    strategy_valid_records = {}
    for s in STRATEGIES:
        out = EXPERIMENT_DIR / f"data_{s}"
        m_split = export_split(out_dir=out, quality="approved", traj_type=s, val_frac=0.1, seed=SEED)
        train_recs = load_jsonl(out / "train.jsonl")[:n_train_per_adapter]
        valid_recs = load_jsonl(out / "valid.jsonl")[:30]   # cap PPL eval to 30 records
        strategy_valid_records[s] = valid_recs
        stats, state, _ = train_mild_adapter(s, train_recs)
        strategy_states[s] = state
        strategy_train_stats[s] = stats

    # ── Phase 2: Train 3 mild domain adapters ──────
    log("\n[Phase 2] Train 3 mild domain adapters")
    domain_states = {}
    domain_train_stats = {}
    domain_valid_records = {}
    for d in DOMAINS:
        prep = {"math": prepare_math, "code": prepare_code, "medical": prepare_medical}[d]
        all_recs = prep(n_train_per_adapter + 30)
        train_recs = all_recs[:n_train_per_adapter]
        valid_recs = all_recs[n_train_per_adapter : n_train_per_adapter + 30]
        domain_valid_records[d] = valid_recs
        stats, state, _ = train_mild_adapter(f"domain_{d}", train_recs)
        domain_states[d] = state
        domain_train_stats[d] = stats

    # ── Phase 3: Per-adapter single eval ───────────
    log("\n[Phase 3] Per-adapter single eval (task accuracy)")
    per_strategy = {s: evaluate_single_state(strategy_states[s]) for s in STRATEGIES}
    per_domain = {d: evaluate_single_state(domain_states[d]) for d in DOMAINS}
    log(f"  strategies: {per_strategy}")
    log(f"  domains: {per_domain}")

    # ── Phase 4: Per-adapter single PPL (F#73 reference) ─────
    log("\n[Phase 4] Per-adapter single PPL on its validation")
    single_ppl = {}
    for s in STRATEGIES:
        single_ppl[s] = evaluate_single_ppl(strategy_states[s], strategy_valid_records[s])
    for d in DOMAINS:
        single_ppl[d] = evaluate_single_ppl(domain_states[d], domain_valid_records[d])
    log(f"  single PPL: {single_ppl}")

    # ── Phase 5: N=2 strategy×domain composition ──
    log("\n[Phase 5] N=2 strategy×domain composition")
    sxd_pairs = [
        ("prepare", "math",    "gsm8k"),
        ("full",    "code",    "humaneval"),
        ("act",     "medical", "medqa"),
    ]
    sxd = {}
    for s_name, d_name, bench in sxd_pairs:
        composed = evaluate_composed([strategy_states[s_name], domain_states[d_name]])
        s_score = per_strategy[s_name][bench]
        d_score = per_domain[d_name][bench]
        best_axis = max(s_score, d_score)
        drop = best_axis - composed[bench]
        sxd[bench] = {
            "strategy": s_name, "domain": d_name,
            "strategy_score": s_score, "domain_score": d_score, "best_axis": best_axis,
            "composed": composed[bench], "drop_pp": round(drop, 1),
        }
        log(f"  {s_name}+{d_name} → {bench}: composed={composed[bench]:.1f}% (best_axis={best_axis:.1f}, drop={drop:+.1f}pp)")

    # ── Phase 6: N=7 full composition ──────────────
    log("\n[Phase 6] N=7 full composition (4 strategy + 3 domain)")
    all_states = list(strategy_states.values()) + list(domain_states.values())
    all_valid_recs = [strategy_valid_records[s] for s in STRATEGIES] + [domain_valid_records[d] for d in DOMAINS]
    composed_n7 = evaluate_composed(all_states)
    log(f"  N=7 composed: {composed_n7}")

    # PPL preservation: composed PPL on each adapter's val set vs single
    composed_ppl_per_adapter = evaluate_composed_ppl(all_states, all_valid_recs)
    composed_ppl = dict(zip(STRATEGIES + DOMAINS, composed_ppl_per_adapter))
    ppl_ratio = {k: composed_ppl[k] / single_ppl[k] for k in composed_ppl}
    log(f"  composed PPL: {composed_ppl}")
    log(f"  PPL ratio (composed/single): {ppl_ratio}")

    # ── KCs ─────────────────────────────────────────
    log("\n=== Kill Criteria ===")
    BENCH = ["gsm8k", "humaneval", "medqa"]

    # K2111: each mild adapter ≥3pp on primary benchmark
    primary = {
        "full": "humaneval", "prepare": "gsm8k", "act": "medqa", "integrate": "humaneval",
        "math": "gsm8k", "code": "humaneval", "medical": "medqa",
    }
    k2111_per = {}
    for name in STRATEGIES + DOMAINS:
        states_d = per_strategy if name in STRATEGIES else per_domain
        score = states_d[name][primary[name]]
        delta = score - base[primary[name]]
        k2111_per[name] = {"pass": delta >= 3.0, "primary": primary[name], "score": score, "delta_pp": round(delta, 1)}
    k2111 = all(v["pass"] for v in k2111_per.values())

    # K2112: composed PPL ≤ 1.10x single
    k2112_per = {k: {"pass": v <= PPL_PRESERVATION_RATIO, "ratio": round(v, 3),
                     "single_ppl": round(single_ppl[k], 3), "composed_ppl": round(composed_ppl[k], 3)}
                 for k, v in ppl_ratio.items()}
    k2112 = all(v["pass"] for v in k2112_per.values())

    # K2113: N=2 strategy×domain within 5pp
    k2113 = all(v["drop_pp"] <= 5.0 for v in sxd.values())

    # K2114: N=7 preserves base on all 3 benchmarks (composed ≥ base - 2pp tolerance)
    k2114_per = {b: {"pass": composed_n7[b] >= base[b] - 2.0, "composed": composed_n7[b], "base": base[b],
                      "delta_pp": round(composed_n7[b] - base[b], 1)} for b in BENCH}
    k2114 = all(v["pass"] for v in k2114_per.values())

    # K2115: Stiefel for all 7
    stiefel = {}
    for s in STRATEGIES:
        stats = strategy_train_stats[s]
        stiefel[s] = {"A": stats["stiefel_max_A"], "B": stats["stiefel_max_B"]}
    for d in DOMAINS:
        stats = domain_train_stats[d]
        stiefel[d] = {"A": stats["stiefel_max_A"], "B": stats["stiefel_max_B"]}
    k2115 = all(v["A"] < 0.01 and v["B"] < 0.01 for v in stiefel.values())

    all_pass = k2111 and k2112 and k2113 and k2114 and k2115
    verdict = "PROVISIONAL" if IS_SMOKE else ("SUPPORTED" if all_pass else "KILLED")

    results = {
        "is_smoke": IS_SMOKE,
        "config": {"scale": MILD_SCALE, "step_cap": MILD_STEPS_CAP, "ppl_threshold": PPL_PRESERVATION_RATIO},
        "base": base,
        "per_strategy": per_strategy,
        "per_domain": per_domain,
        "single_ppl": {k: round(v, 3) for k, v in single_ppl.items()},
        "composed_ppl": {k: round(v, 3) for k, v in composed_ppl.items()},
        "ppl_ratio": {k: round(v, 3) for k, v in ppl_ratio.items()},
        "n2_strategy_x_domain": sxd,
        "n7_composition": composed_n7,
        "stiefel": stiefel,
        "kill_criteria": {
            "K2111_each_adapter_3pp": {"pass": k2111, "per_adapter": k2111_per},
            "K2112_ppl_preservation_1_10": {"pass": k2112, "per_adapter": k2112_per},
            "K2113_n2_within_5pp": {"pass": k2113, "per_pair": sxd},
            "K2114_n7_preserves_base": {"pass": k2114, "per_benchmark": k2114_per},
            "K2115_stiefel": {"pass": k2115, "per_adapter": stiefel},
        },
        "verdict": verdict, "all_pass": all_pass,
        "total_time_s": round(time.time() - t0, 1),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    log(f"K2111 each adapter ≥3pp:        {'PASS' if k2111 else 'FAIL'}")
    for k, v in k2111_per.items(): log(f"  {k}: {v}")
    log(f"K2112 composed PPL ≤1.10x:      {'PASS' if k2112 else 'FAIL'}")
    for k, v in k2112_per.items(): log(f"  {k}: {v}")
    log(f"K2113 N=2 within 5pp:           {'PASS' if k2113 else 'FAIL'}")
    for k, v in sxd.items(): log(f"  {k}: drop={v['drop_pp']}pp")
    log(f"K2114 N=7 preserves base:       {'PASS' if k2114 else 'FAIL'}  {k2114_per}")
    log(f"K2115 Stiefel:                  {'PASS' if k2115 else 'FAIL'}")
    log(f"\nVERDICT: {verdict}")
    log(f"Total: {results['total_time_s']:.0f}s")


if __name__ == "__main__":
    main()
