"""High-level workflows for capsule training and composition at macro scale.

Orchestrates: model loading → surgery → training → evaluation → save.
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .surgery import (
    apply_capsule_surgery,
    extract_capsule_state,
    is_surgically_modified,
    load_capsule_state,
    load_capsule_state_from_file,
    profile_dead_capsules,
    save_capsule_state,
)
from ..data import load_code_eval, load_code_train
from ..eval import EvalResult, compute_perplexity_local
from ..models import count_params, load_model, unload_model
from .train import calibrate_router, train_capsule_groups

DEFAULT_HF_ID = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"
CAPSULE_DIR = Path("experiments/macro/capsule_states")
N_GROUPS = 4
N_CAPSULES_PER_GROUP = 64  # Was 224 (50% overhead). Now ~12% overhead.
TOP_K = 2


def _print_dead_profile(profile: dict):
    """Print dead capsule profiling results."""
    print(f"\n  --- Dead capsule profile ---")
    print(f"  Total: {profile['total_dead']}/{profile['total_capsules']} "
          f"({profile['dead_pct']:.1f}%) dead across {profile['total_tokens_profiled']} tokens")
    # Show a few representative layers
    layers = sorted(profile['per_layer'].keys())
    if len(layers) > 6:
        show = layers[:3] + layers[-3:]
    else:
        show = layers
    for l in show:
        s = profile['per_layer'][l]
        print(f"    layer {l:2d}: {s['dead']:3d}/{s['total']:3d} ({s['dead_pct']:.0f}%) dead")


def train_single_domain(
    domain: str,
    hf_id: str = DEFAULT_HF_ID,
    steps: int = 500,
    lr: float = 1e-4,
    batch_size: int = 4,
    n_train: int = 5000,
    n_eval: int = 100,
    max_length: int = 512,
) -> EvalResult:
    """Train capsule groups for a single domain. Returns EvalResult for leaderboard."""
    print(f"\n{'=' * 60}")
    print(f"CAPSULE TRAINING: {domain}")
    print(f"{'=' * 60}")

    # Load base model
    model, tokenizer, load_time = load_model(hf_id)
    n_base_params = count_params(model)
    print(f"  Base model loaded in {load_time:.1f}s | {n_base_params:,} params")

    # Baseline PPL before surgery
    eval_texts = load_code_eval(domain, n_samples=n_eval)
    baseline_ppl, _ = compute_perplexity_local(model, tokenizer, eval_texts, max_length=max_length)
    print(f"  Baseline PPL ({domain}): {baseline_ppl:.2f}")

    # Apply surgery
    apply_capsule_surgery(model, N_GROUPS, N_CAPSULES_PER_GROUP, TOP_K)
    n_total_params = count_params(model)
    n_capsule_params = n_total_params - n_base_params
    print(f"  Capsule params: {n_capsule_params:,} ({n_capsule_params / n_base_params * 100:.1f}% of base)")

    # Sanity: surgery shouldn't change PPL
    surgery_ppl, _ = compute_perplexity_local(model, tokenizer, eval_texts, max_length=max_length)
    print(f"  Post-surgery PPL ({domain}): {surgery_ppl:.2f} (should match baseline)")

    # Train capsule groups
    print(f"\n  --- Training capsule groups ({steps} steps) ---")
    train_texts = load_code_train(domain, n_samples=n_train)
    t0 = time.time()
    train_capsule_groups(
        model, tokenizer, train_texts,
        steps=steps, lr=lr, batch_size=batch_size, max_length=max_length,
    )
    train_time = time.time() - t0

    # Profile dead capsules
    profile = profile_dead_capsules(model, tokenizer, eval_texts, max_length=max_length)
    _print_dead_profile(profile)

    # Evaluate trained model
    trained_ppl, tps = compute_perplexity_local(model, tokenizer, eval_texts, max_length=max_length)
    print(f"  Trained PPL ({domain}): {trained_ppl:.2f}")

    improvement = (baseline_ppl - trained_ppl) / baseline_ppl * 100
    print(f"  Improvement: {improvement:+.1f}%")

    # Kill gate
    if improvement < 0.5:
        print(f"  ** KILL GATE: improvement {improvement:.1f}% < 0.5% threshold **")
    else:
        print(f"  OK: improvement {improvement:+.1f}% -- passes kill gate")

    # Save capsule state
    CAPSULE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = CAPSULE_DIR / f"{domain}.npz"
    save_capsule_state(model, save_path)
    print(f"  Saved capsule state to {save_path}")

    # Build EvalResult
    result = EvalResult(
        model_name=f"capsule-{domain}",
        hf_id=hf_id,
        tier="capsule",
        param_count=n_total_params,
        perplexity={domain: round(trained_ppl, 2)},
        tokens_per_sec=round(tps, 1),
        load_time_s=round(load_time, 2),
        eval_time_s=round(train_time, 2),
        peak_memory_gb=round((n_base_params * 0.5 + n_capsule_params * 2) / 1e9, 2),
        backend="local",
        arch=f"capsule({N_GROUPS}g×{N_CAPSULES_PER_GROUP}c,top{TOP_K})",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    del model, tokenizer
    unload_model()

    print(f"\n  Summary: baseline={baseline_ppl:.2f} → trained={trained_ppl:.2f} "
          f"({improvement:+.1f}%) | {train_time:.0f}s | "
          f"{profile['dead_pct']:.0f}% capsules dead")
    return result


def compose_domains(
    domains: list[str],
    hf_id: str = DEFAULT_HF_ID,
    calibration_steps: int = 200,
    lr: float = 1e-4,
    batch_size: int = 4,
    n_eval: int = 100,
    max_length: int = 512,
    single_domain_ppls: dict[str, float] | None = None,
) -> EvalResult:
    """Compose capsule groups from multiple domains. Returns EvalResult.

    If single_domain_ppls is provided, reports composition degradation
    vs single-domain capsule PPL (not just vs baseline).
    """
    print(f"\n{'=' * 60}")
    print(f"CAPSULE COMPOSITION: {', '.join(domains)}")
    print(f"{'=' * 60}")

    total_groups = N_GROUPS * len(domains)
    top_k = TOP_K * len(domains)

    # Load base model
    model, tokenizer, load_time = load_model(hf_id)
    n_base_params = count_params(model)
    print(f"  Base model loaded in {load_time:.1f}s")

    # Baseline PPL per domain
    eval_data = {}
    baseline_ppl = {}
    for domain in domains:
        texts = load_code_eval(domain, n_samples=n_eval)
        eval_data[domain] = texts
        ppl, _ = compute_perplexity_local(model, tokenizer, texts, max_length=max_length)
        baseline_ppl[domain] = ppl
        print(f"  Baseline PPL ({domain}): {ppl:.2f}")

    # Apply surgery with combined groups
    apply_capsule_surgery(model, n_groups=total_groups, top_k_groups=top_k)
    n_total_params = count_params(model)
    print(f"  Surgery: {total_groups} groups, top-{top_k} | {n_total_params:,} params")

    # Load domain capsule states
    for i, domain in enumerate(domains):
        state_path = CAPSULE_DIR / f"{domain}.npz"
        if not state_path.exists():
            print(f"  ERROR: {state_path} not found. Run train-capsule --domain {domain} first.")
            del model, tokenizer
            unload_model()
            raise FileNotFoundError(f"Capsule state not found: {state_path}")
        state = load_capsule_state_from_file(state_path)
        offset = i * N_GROUPS
        load_capsule_state(model, state, group_offset=offset)
        print(f"  Loaded {domain} capsules → groups [{offset}..{offset + N_GROUPS - 1}]")

    # Zero-shot evaluation (before calibration)
    print(f"\n  --- Zero-shot (no calibration) ---")
    composed_ppl = {}
    for domain in domains:
        ppl, _ = compute_perplexity_local(model, tokenizer, eval_data[domain], max_length=max_length)
        composed_ppl[domain] = ppl
        print(f"    {domain}: {ppl:.2f}")

    # Calibrate router
    calibrated_ppl = dict(composed_ppl)  # default to zero-shot if no calibration
    if calibration_steps > 0 and len(domains) >= 2:
        print(f"\n  --- Calibrating router ({calibration_steps} steps) ---")
        cal_texts = {d: load_code_train(d, n_samples=500) for d in domains}
        t0 = time.time()
        calibrate_router(
            model, tokenizer,
            cal_texts[domains[0]], cal_texts[domains[1]],
            steps=calibration_steps, lr=lr, batch_size=batch_size,
            max_length=max_length,
        )
        cal_time = time.time() - t0
        print(f"  Calibration took {cal_time:.0f}s")

        print(f"\n  --- Post-calibration ---")
        for domain in domains:
            ppl, _ = compute_perplexity_local(model, tokenizer, eval_data[domain], max_length=max_length)
            calibrated_ppl[domain] = ppl
            print(f"    {domain}: {ppl:.2f}")

    # Dead capsule profiling on composed model
    profile = profile_dead_capsules(model, tokenizer, eval_data[domains[0]], max_length=max_length)
    _print_dead_profile(profile)

    # Summary table — now includes single-domain comparison
    print(f"\n{'=' * 70}")
    print("COMPOSITION SUMMARY")
    print(f"{'=' * 70}")
    has_single = single_domain_ppls is not None
    if has_single:
        print(f"{'Domain':<12} {'Baseline':>9} {'Single':>9} {'Composed':>9} {'Calibr.':>9} {'vs Base':>9} {'vs Single':>10}")
        print("-" * 70)
    else:
        print(f"{'Domain':<12} {'Baseline':>9} {'Composed':>9} {'Calibrated':>10} {'vs Base':>9}")
        print("-" * 55)

    for domain in domains:
        base = baseline_ppl[domain]
        comp = composed_ppl[domain]
        cal = calibrated_ppl[domain]
        delta_base = (cal - base) / base * 100
        if has_single and domain in single_domain_ppls:
            single = single_domain_ppls[domain]
            delta_single = (cal - single) / single * 100
            print(f"{domain:<12} {base:>9.2f} {single:>9.2f} {comp:>9.2f} {cal:>9.2f} {delta_base:>+8.1f}% {delta_single:>+9.1f}%")
        else:
            print(f"{domain:<12} {base:>9.2f} {comp:>9.2f} {cal:>10.2f} {delta_base:>+8.1f}%")

    # Kill gate: composition degradation vs baseline
    avg_delta_base = sum(
        (calibrated_ppl[d] - baseline_ppl[d]) / baseline_ppl[d] * 100
        for d in domains
    ) / len(domains)

    if avg_delta_base > 5.0:
        print(f"\n  ** KILL GATE: avg degradation vs baseline {avg_delta_base:+.1f}% > 5% threshold **")
    else:
        print(f"\n  OK: avg delta vs baseline {avg_delta_base:+.1f}% -- within 5% threshold")

    # Composition degradation vs single-domain (the real question)
    if has_single:
        available = [d for d in domains if d in single_domain_ppls]
        if available:
            avg_delta_single = sum(
                (calibrated_ppl[d] - single_domain_ppls[d]) / single_domain_ppls[d] * 100
                for d in available
            ) / len(available)
            print(f"  Composition cost vs single-domain: {avg_delta_single:+.1f}% "
                  f"(micro validated: +1.6% at N=5)")

    # Compute tok/s
    _, tps = compute_perplexity_local(model, tokenizer, eval_data[domains[0]], max_length=max_length)

    result = EvalResult(
        model_name="capsule-composed",
        hf_id=hf_id,
        tier="capsule",
        param_count=n_total_params,
        perplexity={d: round(calibrated_ppl[d], 2) for d in domains},
        tokens_per_sec=round(tps, 1),
        load_time_s=round(load_time, 2),
        eval_time_s=0.0,
        peak_memory_gb=round((n_base_params * 0.5 + (n_total_params - n_base_params) * 2) / 1e9, 2),
        backend="local",
        arch=f"capsule-composed({total_groups}g,top{top_k})",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Optional functional smoke test (advisory, not a kill gate)
    try:
        from ..eval_functional import print_smoke_report, run_smoke_test

        def generate_fn(prompt: str) -> str:
            from mlx_lm import generate as mlx_generate
            return mlx_generate(model, tokenizer, prompt=prompt, max_tokens=256)

        print(f"\n  --- Functional smoke test (advisory) ---")
        report = run_smoke_test(generate_fn)
        print_smoke_report(report)
    except Exception as e:
        print(f"  Smoke test skipped: {e}")

    del model, tokenizer
    unload_model()
    return result
