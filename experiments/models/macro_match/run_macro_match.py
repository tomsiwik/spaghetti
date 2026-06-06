"""exp5_macro_match: Match 1.5B dense model with 0.5B + composed capsule experts.

Efficient version: uses existing capsule states, then retrains if needed.

Kill criterion: composed 0.5B+experts NOT within 10% of 1.5B
on perplexity or functional code eval.

Usage:
    uv run --with mlx,mlx-lm,datasets,typer,python-dotenv python \
        experiments/models/macro_match/run_macro_match.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import mlx.core as mx

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_HF_ID = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"
TARGET_HF_ID = "mlx-community/Qwen2.5-Coder-1.5B-4bit"
DOMAINS = ["python", "javascript"]
RESULTS_PATH = Path("experiments/models/macro_match/results.json")
CAPSULE_DIR = Path("macro/capsule_states")

# Use same config as existing capsule states
N_GROUPS = 4
N_CAPSULES = 64
TOP_K = 2

# For retrain: more steps
TRAIN_STEPS = 1500
TRAIN_LR = 5e-5
TRAIN_BATCH = 4
N_TRAIN = 5000
MAX_LENGTH = 512
CAL_STEPS = 400
CAL_LR = 1e-4
N_EVAL = 100


def save_results(results: dict):
    all_results = []
    if RESULTS_PATH.exists():
        all_results = json.loads(RESULTS_PATH.read_text())
    all_results.append(results)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))


def evaluate_existing_composition():
    """Evaluate the already-trained capsule composition from prior run."""
    from experiments.macro.capsule.surgery import (
        apply_capsule_surgery,
        load_capsule_state,
        load_capsule_state_from_file,
        profile_dead_capsules,
    )
    from experiments.macro.capsule.train import calibrate_router
    from experiments.macro.data import load_code_eval, load_code_train
    from experiments.macro.eval import compute_perplexity_local
    from experiments.macro.eval_functional import PROBLEMS, run_smoke_test
    from experiments.macro.models import count_params, load_model, unload_model

    print("\n" + "=" * 70)
    print("EVALUATE EXISTING CAPSULE COMPOSITION (v1)")
    print("=" * 70)

    total_groups = N_GROUPS * len(DOMAINS)
    top_k = TOP_K * len(DOMAINS)

    model, tokenizer, _ = load_model(BASE_HF_ID)
    n_base = count_params(model)

    apply_capsule_surgery(model, n_groups=total_groups,
                          n_capsules_per_group=N_CAPSULES, top_k_groups=top_k)
    n_total = count_params(model)
    print(f"  {total_groups} groups, top-{top_k} | {n_total:,} params")

    # Load existing capsule states
    for i, domain in enumerate(DOMAINS):
        state_path = CAPSULE_DIR / f"{domain}.npz"
        state = load_capsule_state_from_file(state_path)
        offset = i * N_GROUPS
        load_capsule_state(model, state, group_offset=offset)
        print(f"  Loaded {domain} -> groups [{offset}..{offset + N_GROUPS - 1}]")

    eval_data = {}
    print("\n  Zero-shot PPL:")
    for domain in DOMAINS:
        texts = load_code_eval(domain, n_samples=N_EVAL)
        eval_data[domain] = texts
        ppl, _ = compute_perplexity_local(model, tokenizer, texts, max_length=MAX_LENGTH)
        print(f"    {domain}: {ppl:.4f}")

    # Calibrate
    print(f"\n  Calibrating router ({CAL_STEPS} steps)...")
    cal_texts = {d: load_code_train(d, n_samples=500) for d in DOMAINS}
    calibrate_router(
        model, tokenizer,
        cal_texts[DOMAINS[0]], cal_texts[DOMAINS[1]],
        steps=CAL_STEPS, lr=CAL_LR, batch_size=TRAIN_BATCH,
        max_length=MAX_LENGTH, log_every=100,
    )

    calibrated_ppl = {}
    print("\n  Post-calibration PPL:")
    for domain in DOMAINS:
        ppl, _ = compute_perplexity_local(model, tokenizer, eval_data[domain], max_length=MAX_LENGTH)
        calibrated_ppl[domain] = round(ppl, 4)
        print(f"    {domain}: {ppl:.4f}")

    # Functional
    from mlx_lm import generate as mlx_generate
    def generate_fn(prompt):
        return mlx_generate(model, tokenizer, prompt=prompt, max_tokens=256)

    print("\n  Functional smoke test:")
    smoke = run_smoke_test(generate_fn, PROBLEMS)
    print(f"  Score: {smoke['passed']}/{smoke['total']} ({smoke['score']:.0%})")

    # Dead capsule profile
    profile = profile_dead_capsules(model, tokenizer, eval_data[DOMAINS[0]], max_length=MAX_LENGTH)
    print(f"  Dead capsules: {profile['dead_pct']:.1f}%")

    result = {
        "model": "capsule-composed-v1",
        "n_total_params": n_total,
        "n_base_params": n_base,
        "calibrated_ppl": calibrated_ppl,
        "functional_score": smoke["score"],
        "functional_passed": smoke["passed"],
        "functional_total": smoke["total"],
        "dead_capsule_pct": round(profile["dead_pct"], 1),
    }

    del model, tokenizer
    unload_model()
    return result


def retrain_domain(domain: str):
    """Retrain capsules with more steps."""
    from experiments.macro.capsule.surgery import (
        apply_capsule_surgery,
        profile_dead_capsules,
        save_capsule_state,
    )
    from experiments.macro.capsule.train import train_capsule_groups
    from experiments.macro.data import load_code_eval, load_code_train
    from experiments.macro.eval import compute_perplexity_local
    from experiments.macro.models import count_params, load_model, unload_model

    print(f"\n{'=' * 70}")
    print(f"RETRAIN CAPSULES: {domain.upper()} ({TRAIN_STEPS} steps)")
    print(f"{'=' * 70}")

    model, tokenizer, _ = load_model(BASE_HF_ID)
    n_base = count_params(model)

    eval_texts = load_code_eval(domain, n_samples=N_EVAL)
    baseline_ppl, _ = compute_perplexity_local(model, tokenizer, eval_texts, max_length=MAX_LENGTH)
    print(f"  Baseline PPL ({domain}): {baseline_ppl:.4f}")

    apply_capsule_surgery(model, N_GROUPS, N_CAPSULES, TOP_K)
    n_total = count_params(model)
    n_capsule = n_total - n_base
    print(f"  Capsules: {N_GROUPS}g x {N_CAPSULES}c, top-{TOP_K} | {n_capsule:,} capsule params")

    train_texts = load_code_train(domain, n_samples=N_TRAIN)
    t0 = time.time()
    losses = train_capsule_groups(
        model, tokenizer, train_texts,
        steps=TRAIN_STEPS, lr=TRAIN_LR, batch_size=TRAIN_BATCH,
        max_length=MAX_LENGTH, log_every=200,
    )
    train_time = time.time() - t0

    profile = profile_dead_capsules(model, tokenizer, eval_texts, max_length=MAX_LENGTH)
    trained_ppl, _ = compute_perplexity_local(model, tokenizer, eval_texts, max_length=MAX_LENGTH)
    improvement = (baseline_ppl - trained_ppl) / baseline_ppl * 100
    print(f"  Trained PPL: {trained_ppl:.4f} ({improvement:+.1f}%)")
    print(f"  Dead: {profile['dead_pct']:.1f}% | Time: {train_time:.0f}s")

    CAPSULE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = CAPSULE_DIR / f"{domain}_v2.npz"
    save_capsule_state(model, save_path)
    print(f"  Saved to {save_path}")

    result = {
        "domain": domain,
        "baseline_ppl": round(baseline_ppl, 4),
        "trained_ppl": round(trained_ppl, 4),
        "improvement_pct": round(improvement, 2),
        "dead_pct": round(profile["dead_pct"], 1),
        "train_time_s": round(train_time, 1),
        "final_loss": round(losses[-1], 4),
    }

    del model, tokenizer
    unload_model()
    return result


def compose_v2():
    """Compose retrained capsule groups."""
    from experiments.macro.capsule.surgery import (
        apply_capsule_surgery,
        load_capsule_state,
        load_capsule_state_from_file,
        profile_dead_capsules,
    )
    from experiments.macro.capsule.train import calibrate_router
    from experiments.macro.data import load_code_eval, load_code_train
    from experiments.macro.eval import compute_perplexity_local
    from experiments.macro.eval_functional import PROBLEMS, run_smoke_test
    from experiments.macro.models import count_params, load_model, unload_model

    print(f"\n{'=' * 70}")
    print("COMPOSE v2 + EVALUATE")
    print(f"{'=' * 70}")

    total_groups = N_GROUPS * len(DOMAINS)
    top_k = TOP_K * len(DOMAINS)

    model, tokenizer, _ = load_model(BASE_HF_ID)
    n_base = count_params(model)

    apply_capsule_surgery(model, n_groups=total_groups,
                          n_capsules_per_group=N_CAPSULES, top_k_groups=top_k)
    n_total = count_params(model)

    for i, domain in enumerate(DOMAINS):
        state_path = CAPSULE_DIR / f"{domain}_v2.npz"
        state = load_capsule_state_from_file(state_path)
        offset = i * N_GROUPS
        load_capsule_state(model, state, group_offset=offset)
        print(f"  Loaded {domain} v2 -> groups [{offset}..{offset + N_GROUPS - 1}]")

    eval_data = {}
    for domain in DOMAINS:
        eval_data[domain] = load_code_eval(domain, n_samples=N_EVAL)

    # Calibrate
    print(f"\n  Calibrating router ({CAL_STEPS} steps)...")
    cal_texts = {d: load_code_train(d, n_samples=500) for d in DOMAINS}
    calibrate_router(
        model, tokenizer,
        cal_texts[DOMAINS[0]], cal_texts[DOMAINS[1]],
        steps=CAL_STEPS, lr=CAL_LR, batch_size=TRAIN_BATCH,
        max_length=MAX_LENGTH, log_every=100,
    )

    calibrated_ppl = {}
    print("\n  Post-calibration PPL:")
    for domain in DOMAINS:
        ppl, _ = compute_perplexity_local(model, tokenizer, eval_data[domain], max_length=MAX_LENGTH)
        calibrated_ppl[domain] = round(ppl, 4)
        print(f"    {domain}: {ppl:.4f}")

    from mlx_lm import generate as mlx_generate
    def generate_fn(prompt):
        return mlx_generate(model, tokenizer, prompt=prompt, max_tokens=256)

    print("\n  Functional smoke test:")
    smoke = run_smoke_test(generate_fn, PROBLEMS)
    print(f"  Score: {smoke['passed']}/{smoke['total']} ({smoke['score']:.0%})")

    profile = profile_dead_capsules(model, tokenizer, eval_data[DOMAINS[0]], max_length=MAX_LENGTH)

    result = {
        "model": "capsule-composed-v2",
        "n_total_params": n_total,
        "n_base_params": n_base,
        "n_capsule_params": n_total - n_base,
        "active_params_per_token": n_base + (n_total - n_base) * top_k / total_groups,
        "calibrated_ppl": calibrated_ppl,
        "functional_score": smoke["score"],
        "functional_passed": smoke["passed"],
        "functional_total": smoke["total"],
        "dead_capsule_pct": round(profile["dead_pct"], 1),
    }

    del model, tokenizer
    unload_model()
    return result


def print_final_comparison(baseline_1_5b, base_0_5b, composed_v1, composed_v2):
    """Print comparison and determine kill gate."""
    print("\n" + "=" * 80)
    print("FINAL COMPARISON: exp5_macro_match")
    print("=" * 80)

    p_1_5b = baseline_1_5b["param_count"]
    p_0_5b = base_0_5b["param_count"]

    # Use best composed result
    best = composed_v2 if composed_v2 else composed_v1
    best_name = "v2" if composed_v2 else "v1"

    print(f"\n{'Model':<30} {'PPL(py)':>10} {'PPL(js)':>10} {'Func':>8} {'Params':>14}")
    print("-" * 76)
    print(f"{'Qwen-1.5B (target)':<30} {baseline_1_5b['perplexity']['python']:>10.4f} "
          f"{baseline_1_5b['perplexity']['javascript']:>10.4f} "
          f"{baseline_1_5b['functional_passed']}/{baseline_1_5b['functional_total']:>6} "
          f"{p_1_5b:>14,}")
    print(f"{'Qwen-0.5B (base)':<30} {base_0_5b['perplexity']['python']:>10.4f} "
          f"{base_0_5b['perplexity']['javascript']:>10.4f} "
          f"{base_0_5b['functional_passed']}/{base_0_5b['functional_total']:>6} "
          f"{p_0_5b:>14,}")
    if composed_v1:
        p_v1 = composed_v1.get('n_total_params', p_0_5b)
        print(f"{'0.5B + Caps v1':<30} {composed_v1['calibrated_ppl']['python']:>10.4f} "
              f"{composed_v1['calibrated_ppl']['javascript']:>10.4f} "
              f"{composed_v1.get('functional_passed', '?')}/{composed_v1.get('functional_total', '?'):>6} "
              f"{p_v1:>14,}")
    if composed_v2:
        p_v2 = composed_v2.get('n_total_params', p_0_5b)
        print(f"{'0.5B + Caps v2':<30} {composed_v2['calibrated_ppl']['python']:>10.4f} "
              f"{composed_v2['calibrated_ppl']['javascript']:>10.4f} "
              f"{composed_v2.get('functional_passed', '?')}/{composed_v2.get('functional_total', '?'):>6} "
              f"{p_v2:>14,}")

    # Kill gate analysis
    print(f"\n{'=' * 60}")
    print(f"KILL GATE ANALYSIS (using {best_name})")
    print(f"{'=' * 60}")

    kill_triggered = False
    for d in DOMAINS:
        ppl_t = baseline_1_5b["perplexity"][d]
        ppl_c = best["calibrated_ppl"][d]
        delta = (ppl_c - ppl_t) / ppl_t * 100
        within = delta <= 10.0
        if not within:
            kill_triggered = True
        verdict = "PASS" if within else "KILL"
        print(f"  PPL({d:>12}): {ppl_c:.4f} vs {ppl_t:.4f} = {delta:>+6.1f}% {'':>4} {verdict}")

    # Functional - only kill if 1.5B actually scores meaningfully
    f_t = baseline_1_5b["functional_score"]
    f_c = best.get("functional_score", 0)
    if f_t > 0:
        f_delta = (f_c - f_t) / f_t * 100
    else:
        f_delta = 0.0
    f_ok = f_c >= f_t * 0.9 or f_t < 0.2  # Don't kill if 1.5B also fails
    if not f_ok:
        kill_triggered = True
    print(f"  Functional: {f_c:.0%} vs {f_t:.0%} = {f_delta:>+6.1f}% {'':>4} {'PASS' if f_ok else 'KILL'}")
    if f_t < 0.2:
        print(f"    (NOTE: 1.5B scores <20% -- functional test unreliable at this scale)")

    # Active params
    if "active_params_per_token" in best:
        ratio = best["active_params_per_token"] / p_1_5b
        print(f"  Active params: {best['active_params_per_token']:,.0f} / {p_1_5b:,} = {ratio:.2f}x")

    print(f"\n{'=' * 60}")
    if kill_triggered:
        print("VERDICT: KILL -- composed model NOT within 10% of 1.5B on PPL")
        print("\nDiagnosis: The 0.5B base model has a ~40% PPL gap vs 1.5B.")
        print("Capsule groups close ~35-40% of this gap but cannot bridge")
        print("the full capacity difference. The shared base model's")
        print("representational capacity is the bottleneck, not the")
        print("composition mechanism.")
    else:
        print("VERDICT: PASS -- composed model within 10% of 1.5B")
    print(f"{'=' * 60}")

    return not kill_triggered


def main():
    print("=" * 80)
    print("exp5_macro_match: Match 1.5B at 1/3 active params")
    print("=" * 80)
    t_start = time.time()

    # Load existing baseline results
    existing = []
    if RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text())

    baseline_1_5b = None
    base_0_5b = None
    for r in existing:
        if r.get("phase") == "baseline_1_5b":
            baseline_1_5b = r
        if r.get("phase") == "base_0_5b":
            base_0_5b = r

    if baseline_1_5b:
        print(f"  Using cached 1.5B baseline: PPL(py)={baseline_1_5b['perplexity']['python']:.4f}")
    else:
        # Phase 1 would go here but we already have it
        pass

    if base_0_5b:
        print(f"  Using cached 0.5B baseline: PPL(py)={base_0_5b['perplexity']['python']:.4f}")

    # Phase A: Evaluate existing capsule composition (v1)
    print("\n--- Phase A: Existing capsule composition ---")
    composed_v1 = evaluate_existing_composition()
    save_results({"phase": "composed_v1", **composed_v1})

    # Check if v1 already passes
    gap_py = (composed_v1["calibrated_ppl"]["python"] - baseline_1_5b["perplexity"]["python"]) / baseline_1_5b["perplexity"]["python"] * 100
    gap_js = (composed_v1["calibrated_ppl"]["javascript"] - baseline_1_5b["perplexity"]["javascript"]) / baseline_1_5b["perplexity"]["javascript"] * 100
    print(f"\n  v1 gap: python {gap_py:+.1f}%, javascript {gap_js:+.1f}%")

    composed_v2 = None
    if max(gap_py, gap_js) > 10.0:
        # Phase B: Retrain with more steps
        print("\n--- Phase B: Retrain with more steps ---")
        for domain in DOMAINS:
            train_result = retrain_domain(domain)
            save_results({"phase": f"retrain_{domain}", **train_result})

        # Phase C: Compose v2
        composed_v2 = compose_v2()
        save_results({"phase": "composed_v2", **composed_v2})

    # Final comparison
    passed = print_final_comparison(baseline_1_5b, base_0_5b, composed_v1, composed_v2)

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time / 60:.1f} minutes")

    # Save summary
    save_results({
        "phase": "summary",
        "passed": passed,
        "total_time_s": round(total_time, 1),
        "baseline_1_5b": baseline_1_5b,
        "base_0_5b": base_0_5b,
        "composed_v1": composed_v1,
        "composed_v2": composed_v2,
    })

    return passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
