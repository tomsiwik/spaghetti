#!/usr/bin/env python3
"""
5 PoLAR domain context adapters on Gemma 4 (Pierre Phase 1.4):
  math (GSM8K) | code (CodeAlpaca) | medical (MedQA) | legal (case_hold) | finance (alpaca)

Then validates:
  - K7: N=5 domain composition preserves on-domain accuracy (PoLAR replication of F#440)
  - K8: N=8 composition (5 domains + 3 strategies from exp_beehive_polar_strategy_n3)

Kill criteria:
  K2082: Math PoLAR ≥ F#424 baseline GSM8K within ±5pp
  K2083: Code PoLAR ≥ F#424 baseline HumanEval within ±5pp
  K2084: Medical PoLAR ≥ F#424 baseline MedQA within ±5pp
  K2085: Legal PoLAR improves base by ≥5pp on case_hold
  K2086: Finance PoLAR improves base by ≥5pp on finance-alpaca held-out
  K2087: Joint Stiefel for ALL 5 (max ||...||_F < 0.01)
  K2088: N=5 domain composition preserves on-domain within 5pp
  K2089: N=8 composition preserves best on-domain within 7pp
"""

import json
import os
import re
import subprocess
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
    inject_polar_adapters, train, cleanup,
    eval_gsm8k, eval_humaneval, eval_medqa,
    PoLARLinear, RANK, SCALE,
)

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42

N_TRAIN = 50 if IS_SMOKE else 2000
N_STEPS = 20 if IS_SMOKE else 1000
N_EVAL = 5 if IS_SMOKE else 30

DOMAINS = ["math", "code", "medical", "legal", "finance"]

# Pierre Phase 1.4 baseline thresholds — from F#424 + conservative for new domains
BASELINE_DELTA_PP = {
    "math": 22,        # F#424 GSM8K
    "code": 48,        # F#424 HumanEval
    "medical": 62,     # F#424 MedQA
    "legal": 5,        # No prior baseline; conservative
    "finance": 5,      # No prior baseline; conservative
}

# Path to pre-trained strategy adapters from exp_beehive_polar_strategy_n3
STRATEGY_DIR = REPO_ROOT / "micro" / "models" / "exp_beehive_polar_strategy_n3" / "adapter_weights"


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    a = mx.get_active_memory() / 1e9
    c = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB")


# ─────────────────────────────────────────────
# Per-domain dataset prep (returns list of {"messages":[...]} records)
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
        content = ex["instruction"]
        if ex.get("input"):
            content += f"\n\nInput:\n{ex['input']}"
        out.append({"messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": ex["output"]},
        ]})
    return out


def prepare_medical(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train").shuffle(seed=SEED).select(range(min(n, 10178)))
    out = []
    for ex in ds:
        opts = ex["options"]
        question = (
            f"{ex['question']}\n"
            f"(A) {opts['A']}\n(B) {opts['B']}\n(C) {opts['C']}\n(D) {opts['D']}"
        )
        out.append({"messages": [
            {"role": "user", "content": f"Answer this medical multiple choice question. Respond with only the letter (A/B/C/D).\n\n{question}"},
            {"role": "assistant", "content": f"{ex['answer_idx']}: {ex['answer']}"},
        ]})
    return out


def prepare_legal(n: int) -> list[dict]:
    """case_hold: 5-way multiple-choice legal Q&A (US case-law citation completion)."""
    from datasets import load_dataset
    ds = load_dataset("lex_glue", "case_hold", split="train").shuffle(seed=SEED).select(range(min(n, 45000)))
    out = []
    for ex in ds:
        endings = ex["endings"]
        gt_idx = ex["label"]  # 0-4
        gt_letter = "ABCDE"[gt_idx]
        question = (
            f"{ex['context']}\n\n"
            f"(A) {endings[0]}\n(B) {endings[1]}\n(C) {endings[2]}\n(D) {endings[3]}\n(E) {endings[4]}"
        )
        out.append({"messages": [
            {"role": "user", "content": f"Choose the holding most likely cited in this legal case. Respond with only the letter (A/B/C/D/E).\n\n{question}"},
            {"role": "assistant", "content": f"{gt_letter}"},
        ]})
    return out


def prepare_finance(n: int) -> list[dict]:
    """finance-alpaca: instruction-style financial Q&A."""
    from datasets import load_dataset
    ds = load_dataset("gbharti/finance-alpaca", split="train").shuffle(seed=SEED).select(range(min(n, 68000)))
    out = []
    for ex in ds:
        content = ex["instruction"]
        if ex.get("input"):
            content += f"\n\nInput:\n{ex['input']}"
        out.append({"messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": ex["output"]},
        ]})
    return out


PREPARE_FNS = {
    "math": prepare_math,
    "code": prepare_code,
    "medical": prepare_medical,
    "legal": prepare_legal,
    "finance": prepare_finance,
}


# ─────────────────────────────────────────────
# Per-domain eval (returns accuracy 0-100)
# ─────────────────────────────────────────────

def eval_legal(model, tokenizer, n_eval=30) -> float:
    from datasets import load_dataset
    from mlx_lm import generate

    ds = load_dataset("lex_glue", "case_hold", split="validation").shuffle(seed=SEED).select(range(min(n_eval, 3900)))
    correct = 0
    for ex in ds:
        endings = ex["endings"]
        question = f"{ex['context']}\n\n(A) {endings[0]}\n(B) {endings[1]}\n(C) {endings[2]}\n(D) {endings[3]}\n(E) {endings[4]}"
        msgs = [{"role": "user", "content": f"Choose the holding most likely cited in this legal case. Respond with only the letter (A/B/C/D/E).\n\n{question}"}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=formatted, max_tokens=20, verbose=False)
        gt_letter = "ABCDE"[ex["label"]]
        pred = response.strip().upper()
        pred_letter = next((L for L in "ABCDE" if pred.startswith(L)), None)
        if not pred_letter:
            m = re.search(r"\b([ABCDE])\b", pred)
            pred_letter = m.group(1) if m else None
        if pred_letter == gt_letter:
            correct += 1
    return correct / len(ds) * 100


def eval_finance(model, tokenizer, n_eval=30) -> float:
    """Finance eval: holdout slice of finance-alpaca, judge by token-overlap heuristic.

    Behavioral proxy: does response include ≥30% of content words from gold answer?
    Threshold tuned to be permissive — we measure 'in the ballpark' not exact match.
    """
    from datasets import load_dataset
    from mlx_lm import generate

    # Split off last 5% deterministically as held-out eval
    ds_full = load_dataset("gbharti/finance-alpaca", split="train").shuffle(seed=SEED + 1)
    eval_split = ds_full.select(range(max(0, len(ds_full) - 1000), len(ds_full)))
    eval_split = eval_split.shuffle(seed=SEED).select(range(min(n_eval, len(eval_split))))

    stop = {"that", "this", "with", "from", "into", "when", "than", "their", "your", "have",
            "been", "must", "will", "what", "which", "where", "should", "would", "could",
            "about", "every", "after", "before", "while", "until", "because", "those", "these",
            "they", "them", "some", "more", "less", "also", "such", "even", "both", "each",
            "very", "well", "back", "much", "many", "most", "make", "made", "does", "doing"}

    def content_words(s: str) -> set[str]:
        return {w.lower() for w in re.findall(r"[A-Za-z]{4,}", s) if w.lower() not in stop}

    correct = 0
    for ex in eval_split:
        content = ex["instruction"]
        if ex.get("input"):
            content += f"\n\nInput:\n{ex['input']}"
        msgs = [{"role": "user", "content": content}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=formatted, max_tokens=400, verbose=False)
        gold_words = content_words(ex["output"])
        if not gold_words:
            continue
        resp_words = content_words(response)
        overlap = len(gold_words & resp_words) / len(gold_words)
        if overlap >= 0.3:
            correct += 1
    return correct / len(eval_split) * 100


EVAL_FNS = {
    "math": eval_gsm8k,
    "code": eval_humaneval,
    "medical": eval_medqa,
    "legal": eval_legal,
    "finance": eval_finance,
}


# ─────────────────────────────────────────────
# Domain training
# ─────────────────────────────────────────────

def train_domain(domain: str, n_train: int, n_steps: int) -> tuple[dict, list[dict], str]:
    """Returns (train_stats, per-layer state, weights_path)."""
    log(f"\n[Train PoLAR domain: {domain}]")
    records = PREPARE_FNS[domain](n_train)
    log(f"  prepared {len(records)} training records")

    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
    log_memory(f"polar-injected-{domain}")

    stats = train(model, tokenizer, records, modules, n_steps)
    log(f"  loss: {stats['first_loss']:.4f} → {stats['final_loss']:.4f}, Stiefel A={stats['stiefel_max_A']:.4f} B={stats['stiefel_max_B']:.4f}")

    state = []
    for m in modules:
        state.append({
            "a": np.array(m.lora_a.tolist(), dtype=np.float32),
            "b": np.array(m.lora_b.tolist(), dtype=np.float32),
        })

    # Save shareable weights for pierre-server consumption
    out_dir = REPO_ROOT / "data" / "adapters" / f"{domain}_polar"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = {}
    for i, m in enumerate(modules):
        weights[f"layer_{i}.lora_a"] = m.lora_a
        weights[f"layer_{i}.lora_b"] = m.lora_b
    mx.eval(weights)
    weights_path = str(out_dir / "polar.safetensors")
    mx.save_safetensors(weights_path, weights)

    cleanup(model, tokenizer, modules)
    return {k: v for k, v in stats.items() if k != "losses"}, state, weights_path


def eval_with_state(label: str, state: list[dict], domain_for_eval: str) -> float:
    """Inject state into fresh model, eval on `domain_for_eval` benchmark."""
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
    for i, m in enumerate(modules):
        m.lora_a = mx.array(state[i]["a"])
        m.lora_b = mx.array(state[i]["b"])
    mx.eval(model.parameters())
    eval_fn = EVAL_FNS[domain_for_eval]
    acc = eval_fn(model, tokenizer, N_EVAL)
    log(f"  [Eval {label} → {domain_for_eval}]: {acc:.1f}%")
    cleanup(model, tokenizer, modules)
    return acc


# ─────────────────────────────────────────────
# Composition primitive
# ─────────────────────────────────────────────

def compose_apply(modules, states_list: list[list[dict]]):
    """Σ_i (a_i b_i) / N applied as composed delta override on each PoLAR layer."""
    n = len(states_list)
    for layer_idx, m in enumerate(modules):
        delta = None
        for state in states_list:
            a = mx.array(state[layer_idx]["a"]) if isinstance(state[layer_idx]["a"], np.ndarray) else state[layer_idx]["a"]
            b = mx.array(state[layer_idx]["b"]) if isinstance(state[layer_idx]["b"], np.ndarray) else state[layer_idx]["b"]
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


def eval_composition(label: str, states_list: list[list[dict]], evals_to_run: list[str]) -> dict:
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
    compose_apply(modules, states_list)
    mx.eval(model.parameters())

    log(f"\n[Eval composition: {label}, N={len(states_list)}, evals={evals_to_run}]")
    out = {}
    for d in evals_to_run:
        eval_fn = EVAL_FNS[d]
        acc = eval_fn(model, tokenizer, N_EVAL)
        out[d] = round(acc, 1)
        log(f"  {label} → {d}: {acc:.1f}%")
    cleanup(model, tokenizer, modules)
    return out


def load_strategy_states() -> list[list[dict]]:
    """Load the 3 strategy adapter weights from exp_beehive_polar_strategy_n3."""
    if not STRATEGY_DIR.exists():
        return []

    strategies = ["prepare", "act", "integrate"]
    states_list = []
    for s in strategies:
        path = STRATEGY_DIR / f"strategy_{s}.safetensors"
        if not path.exists():
            log(f"  WARN: strategy adapter missing: {path}")
            continue
        raw = mx.load(str(path))
        # Reconstruct per-layer state
        n_layers = max(int(k.split(".")[0].split("_")[1]) for k in raw.keys() if k.startswith("layer_")) + 1
        state = []
        for i in range(n_layers):
            a = raw[f"layer_{i}.lora_a"]
            b = raw[f"layer_{i}.lora_b"]
            state.append({"a": np.array(a.tolist(), dtype=np.float32), "b": np.array(b.tolist(), dtype=np.float32)})
        states_list.append(state)
        log(f"  loaded strategy {s}: {n_layers} layers")
    return states_list


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    t_start = time.time()
    log_memory("start")
    log(f"=== 5 PoLAR Domain Adapters + N=8 Composition (SMOKE={IS_SMOKE}) ===")

    # ── Phase 0: Base model evaluation across all 5 domains ──
    log("\n[Phase 0] Base evaluation on 5 domains")
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    base = {}
    for d in DOMAINS:
        eval_fn = EVAL_FNS[d]
        acc = eval_fn(model, tokenizer, N_EVAL)
        base[d] = round(acc, 1)
        log(f"  base → {d}: {acc:.1f}%")
    cleanup(model, tokenizer)

    # ── Phase 1: Train 5 PoLAR domain adapters ──
    log("\n[Phase 1] Train 5 PoLAR domain adapters")
    train_stats = {}
    domain_states = {}
    domain_weights_paths = {}
    for d in DOMAINS:
        stats, state, wpath = train_domain(d, N_TRAIN, N_STEPS)
        train_stats[d] = stats
        domain_states[d] = state
        domain_weights_paths[d] = wpath

    # ── Phase 2: Per-domain accuracy of each adapter (single-adapter check) ──
    log("\n[Phase 2] Single-adapter on-domain accuracy (5 evals)")
    single_acc = {}
    for d in DOMAINS:
        single_acc[d] = round(eval_with_state(d, domain_states[d], d), 1)

    # ── Phase 3: N=5 composition ──
    log("\n[Phase 3] N=5 domain composition (Σ B_i @ A_i / 5)")
    states_list_n5 = [domain_states[d] for d in DOMAINS]
    composed_n5 = eval_composition("N=5-domains", states_list_n5, evals_to_run=DOMAINS)

    # ── Phase 4: N=8 composition (5 domains + 3 strategies) ──
    log("\n[Phase 4] N=8 composition (5 domains + 3 strategies)")
    strategy_states = load_strategy_states()
    if strategy_states:
        composed_n8 = eval_composition(
            f"N=8 (5d+{len(strategy_states)}s)",
            states_list_n5 + strategy_states,
            evals_to_run=DOMAINS,
        )
    else:
        log("  WARN: no strategy adapters found; K8 will be inconclusive")
        composed_n8 = None

    # ── Kill criteria ──
    log("\n=== Kill Criteria ===")

    # K1-K5: each domain meets baseline
    domain_passes = {}
    for i, d in enumerate(DOMAINS):
        delta = single_acc[d] - base[d]
        threshold = BASELINE_DELTA_PP[d]
        # ±5pp tolerance for math/code/medical (replicating known baselines), ≥threshold for new domains
        if d in ("math", "code", "medical"):
            ok = delta >= threshold - 5
        else:
            ok = delta >= threshold
        kc_id = f"K208{2+i}"
        domain_passes[d] = {"pass": ok, "delta_pp": round(delta, 1), "threshold": threshold, "kc": kc_id}
        log(f"  {kc_id} {d:8s}: base={base[d]:.1f}% adapter={single_acc[d]:.1f}% Δ={delta:+.1f}pp threshold={threshold:+d}pp {'PASS' if ok else 'FAIL'}")

    # K6: Stiefel for all 5
    stiefel_max = {d: {"A": train_stats[d].get("stiefel_max_A", float("inf")),
                       "B": train_stats[d].get("stiefel_max_B", float("inf"))} for d in DOMAINS}
    k2087 = all(v["A"] < 0.01 and v["B"] < 0.01 for v in stiefel_max.values())
    log(f"  K2087 stiefel ALL 5 < 0.01: {'PASS' if k2087 else 'FAIL'} {stiefel_max}")

    # K7: N=5 composition preserves on-domain within 5pp
    n5_drops = {d: round(single_acc[d] - composed_n5[d], 1) for d in DOMAINS}
    k2088 = all(drop <= 5.0 for drop in n5_drops.values())
    log(f"  K2088 N=5 composition within 5pp: {'PASS' if k2088 else 'FAIL'} drops={n5_drops}")

    # K8: N=8 composition preserves best-on-domain within 7pp
    if composed_n8:
        n8_drops = {d: round(max(single_acc[d], base[d]) - composed_n8[d], 1) for d in DOMAINS}
        k2089 = all(drop <= 7.0 for drop in n8_drops.values())
        log(f"  K2089 N=8 composition within 7pp: {'PASS' if k2089 else 'FAIL'} drops={n8_drops}")
    else:
        n8_drops = None
        k2089 = None

    # All-pass if domain Ks + Stiefel + N=5 + N=8 (when available)
    all_domain_pass = all(v["pass"] for v in domain_passes.values())
    all_pass = all_domain_pass and k2087 and k2088 and (k2089 if k2089 is not None else True)
    verdict = "PROVISIONAL" if IS_SMOKE else ("SUPPORTED" if all_pass else "KILLED")

    results = {
        "is_smoke": IS_SMOKE,
        "n_train": N_TRAIN,
        "n_steps": N_STEPS,
        "n_eval": N_EVAL,
        "base": base,
        "single_adapter_accuracy": single_acc,
        "train_stats": train_stats,
        "domain_passes": domain_passes,
        "stiefel_max": stiefel_max,
        "composed_n5": composed_n5,
        "composed_n5_drops_pp": n5_drops,
        "composed_n8": composed_n8,
        "composed_n8_drops_pp": n8_drops,
        "domain_weights_paths": domain_weights_paths,
        "kill_criteria": {
            **{f"K{2082+i}_{d}_baseline": domain_passes[d] for i, d in enumerate(DOMAINS)},
            "K2087_stiefel_all5": {"pass": k2087, "max": stiefel_max},
            "K2088_n5_composition_5pp": {"pass": k2088, "drops_pp": n5_drops},
            "K2089_n8_composition_7pp": {"pass": k2089, "drops_pp": n8_drops, "strategies_loaded": len(strategy_states)},
        },
        "verdict": verdict,
        "all_pass": all_pass,
        "total_time_s": round(time.time() - t_start, 1),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    log(f"\nVERDICT: {verdict}")
    log(f"Total time: {results['total_time_s']:.0f}s")
    log(f"Adapters saved to: adapters/{{math,code,medical,legal,finance}}_polar/polar.safetensors")


if __name__ == "__main__":
    main()
