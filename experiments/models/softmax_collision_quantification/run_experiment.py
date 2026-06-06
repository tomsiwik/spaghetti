"""Experiment: Softmax collision rate scaling and mitigation (REVISED).

Protocol:
1. Phase 1 (Collision Scaling): Train CapsuleMoEGPT at N=8,16,32,64 with k=2.
   Measure collision rate at epsilon={0.01, 0.05, 0.10} post-training.
   KC1: collision rate must increase with N.

2. Phase 2 (Mitigation): At N=32, test:
   - Temperature scaling: T=0.5, T=1.0 (baseline), T=2.0
   - Margin loss: target_margin=0.1, weight=0.1
   KC2: best mitigation must improve quality by >0.5% (at p<0.05).

REVISION NOTES (2026-03-07):
- 5 seeds for Phase 2 (statistical significance)
- Dual-temperature collision measurement: at training T AND T=1.0
- p-values via Welch's t-test for KC2
- Phase 1 kept from v1 (3 seeds, already solid r^2=0.978)
"""

import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.data import load_names, CharTokenizer, CharDataset, train_val_split
from experiments.train import train, evaluate
from experiments.models.capsule_moe.capsule_moe import CapsuleMoEGPT
from experiments.models.softmax_collision_quantification.softmax_collision_quantification import (
    TempScaledMoEGPT,
    MarginLossMoEGPT,
)


def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def measure_collision_rates(model, val_ds, n_batches=10, epsilons=(0.01, 0.05, 0.10),
                            temperature=1.0):
    """Measure collision rates at multiple epsilon thresholds.

    A 'collision' is when the top-1 and top-2 softmax probabilities are
    within epsilon of each other (near-tie).

    Args:
        temperature: Temperature applied to logits before softmax for measurement.
            Use 1.0 to measure raw logit quality; use model's training temperature
            to measure inference-time behavior.

    Returns dict mapping epsilon -> collision_rate, plus distribution stats.
    """
    rng = random.Random(0)
    collision_counts = {eps: 0 for eps in epsilons}
    total_tokens = 0
    all_gaps = []

    for _ in range(n_batches):
        inputs, _ = val_ds.get_batch(32, rng)
        B, T = inputs.shape
        pos = mx.arange(T)
        x = model.wte(inputs) + model.wpe(pos)
        x = model.norm0(x)

        for layer in model.layers:
            h = x + layer.attn(layer.norm1(x))
            h_normed = layer.norm2(h)

            # Get router scores
            pool = layer.capsule_pool
            scores = pool.router(h_normed)  # (B, T, G)
            # Apply measurement temperature
            probs = mx.softmax(scores / temperature, axis=-1)
            mx.eval(probs)

            # Get top-2 gap
            sorted_probs = mx.sort(probs, axis=-1)
            mx.eval(sorted_probs)
            top1 = sorted_probs[..., -1]
            top2 = sorted_probs[..., -2]
            gap = top1 - top2
            mx.eval(gap)

            gap_flat = gap.reshape(-1)
            mx.eval(gap_flat)
            all_gaps.extend(gap_flat.tolist())

            for eps in epsilons:
                collisions = mx.sum((gap < eps).astype(mx.float32)).item()
                collision_counts[eps] += collisions

            total_tokens += B * T
            x = layer(x)

    rates = {}
    for eps in epsilons:
        rates[eps] = collision_counts[eps] / total_tokens if total_tokens > 0 else 0

    all_gaps_sorted = sorted(all_gaps)
    n = len(all_gaps_sorted)
    mean_gap = sum(all_gaps_sorted) / n if n > 0 else 0
    median_gap = all_gaps_sorted[n // 2] if n > 0 else 0
    p10 = all_gaps_sorted[n // 10] if n > 0 else 0
    p90 = all_gaps_sorted[9 * n // 10] if n > 0 else 0

    return {
        "collision_rates": {str(eps): rate for eps, rate in rates.items()},
        "gap_stats": {
            "mean": mean_gap,
            "median": median_gap,
            "p10": p10,
            "p90": p90,
        },
        "total_tokens": total_tokens,
        "measurement_temperature": temperature,
    }


def welch_t_test(vals_a, vals_b):
    """Welch's t-test for unequal variances. Returns (t_stat, p_value, df)."""
    n_a, n_b = len(vals_a), len(vals_b)
    mean_a = sum(vals_a) / n_a
    mean_b = sum(vals_b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in vals_a) / (n_a - 1) if n_a > 1 else 0
    var_b = sum((x - mean_b) ** 2 for x in vals_b) / (n_b - 1) if n_b > 1 else 0

    se = math.sqrt(var_a / n_a + var_b / n_b) if (var_a / n_a + var_b / n_b) > 0 else 1e-10
    t_stat = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)) if (n_a > 1 and n_b > 1) else 1
    df = num / denom if denom > 0 else 1

    # Approximate two-tailed p-value using Student's t CDF approximation
    # Use the regularized incomplete beta function approximation
    p_value = _t_cdf_approx(abs(t_stat), df)
    return t_stat, p_value, df


def _t_cdf_approx(t, df):
    """Approximate two-tailed p-value for Student's t distribution.

    Uses the relationship between t and F distributions:
    P(|T| > t) = I_x(df/2, 1/2) where x = df/(df + t^2)
    Approximated via the normal distribution for large df.
    """
    if df <= 0:
        return 1.0
    # For small df, use a simple approximation
    x = df / (df + t * t)
    # Regularized incomplete beta function approximation
    # For moderate t and df, use the normal approximation with correction
    # Cornish-Fisher: z ~ t * (1 - 1/(4*df)) * sqrt(1/df * (df - 1))
    # Simpler: for df > 2, use normal approx with adjusted z
    if df > 30:
        z = t
    else:
        # Adjusted z-score for t distribution
        z = t * math.sqrt(math.log(1 + t * t / df) / (t * t / df)) if t > 0 else 0

    # Normal CDF approximation (Abramowitz & Stegun)
    if z > 8:
        return 2e-16  # essentially zero
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p_const = 0.3275911
    sign = 1 if z >= 0 else -1
    z = abs(z)
    tt = 1.0 / (1.0 + p_const * z)
    normal_cdf = 1.0 - (a1 * tt + a2 * tt**2 + a3 * tt**3 + a4 * tt**4 + a5 * tt**5) * math.exp(-z * z / 2.0) / math.sqrt(2 * math.pi)
    # Two-tailed
    return 2.0 * (1.0 - normal_cdf)


def run_phase1_scaling(seeds=(42, 123, 777), steps=500):
    """Phase 1: Measure collision rate vs N. (Unchanged from v1.)"""
    print("=" * 70)
    print("  PHASE 1: Collision Rate Scaling with N")
    print("=" * 70)

    N_values = [8, 16, 32, 64]
    epsilons = [0.01, 0.05, 0.10]
    results = []

    for N in N_values:
        caps_per_group = max(4, 256 // N)

        seed_results = []
        for seed in seeds:
            print(f"\n--- N={N}, caps/group={caps_per_group}, seed={seed} ---")

            docs = load_names()
            tokenizer = CharTokenizer(docs)
            docs_train, docs_val = train_val_split(docs, seed=seed)
            train_ds = CharDataset(docs_train, tokenizer, block_size=32)
            val_ds = CharDataset(docs_val, tokenizer, block_size=32)

            model = CapsuleMoEGPT(
                vocab_size=tokenizer.vocab_size, block_size=32,
                n_embd=64, n_head=4, n_layer=4,
                n_groups=N, n_capsules_per_group=caps_per_group, top_k_groups=2,
            )
            mx.eval(model.parameters())
            n_params = count_params(model)
            print(f"  Params: {n_params:,}")

            result = train(model, train_ds, val_ds, steps=steps,
                          batch_size=32, lr=3e-3, seed=seed, log_every=200)
            val_loss = result["val_loss"]
            print(f"  Val loss: {val_loss:.4f}")

            # Measure collision rates at T=1.0 (standard)
            collision_data = measure_collision_rates(model, val_ds, n_batches=10,
                                                     epsilons=epsilons, temperature=1.0)
            print(f"  Collision rates: " + " | ".join(
                f"eps={eps}: {collision_data['collision_rates'][str(eps)]:.3f}"
                for eps in epsilons
            ))
            print(f"  Gap stats: mean={collision_data['gap_stats']['mean']:.4f}, "
                  f"median={collision_data['gap_stats']['median']:.4f}, "
                  f"p10={collision_data['gap_stats']['p10']:.4f}")

            seed_results.append({
                "seed": seed,
                "val_loss": val_loss,
                "params": n_params,
                "collision_data": collision_data,
                "tokens_per_sec": result["tokens_per_sec"],
            })

        results.append({
            "N": N,
            "caps_per_group": caps_per_group,
            "seed_results": seed_results,
        })

    return results


def run_phase2_mitigation(seeds=(42, 123, 777, 2024, 314), steps=500):
    """Phase 2: Test collision mitigations at N=32 with dual-temperature measurement.

    REVISION: 5 seeds, measure collision at BOTH training T and T=1.0.
    """
    print("\n" + "=" * 70)
    print("  PHASE 2: Collision Mitigation at N=32 (REVISED: 5 seeds, dual-T)")
    print("=" * 70)

    N = 32
    caps_per_group = 8
    epsilons = [0.01, 0.05, 0.10]
    results = []

    # (config_name, model_class_name, extra_kwargs, training_temperature)
    configs = [
        ("baseline_T1.0", "CapsuleMoEGPT", {}, 1.0),
        ("temp_T0.5", "TempScaledMoEGPT", {"temperature": 0.5}, 0.5),
        ("temp_T2.0", "TempScaledMoEGPT", {"temperature": 2.0}, 2.0),
        ("margin_m0.1", "MarginLossMoEGPT", {"target_margin": 0.1, "margin_weight": 0.1}, 1.0),
    ]

    for config_name, model_class_name, extra_kwargs, train_temp in configs:
        seed_results = []
        for seed in seeds:
            print(f"\n--- {config_name}, seed={seed} ---")

            docs = load_names()
            tokenizer = CharTokenizer(docs)
            docs_train, docs_val = train_val_split(docs, seed=seed)
            train_ds = CharDataset(docs_train, tokenizer, block_size=32)
            val_ds = CharDataset(docs_val, tokenizer, block_size=32)

            base_kwargs = dict(
                vocab_size=tokenizer.vocab_size, block_size=32,
                n_embd=64, n_head=4, n_layer=4,
                n_groups=N, n_capsules_per_group=caps_per_group, top_k=2,
            )

            if model_class_name == "CapsuleMoEGPT":
                base_kwargs.pop("top_k")
                base_kwargs["top_k_groups"] = 2
                model = CapsuleMoEGPT(**base_kwargs)
            elif model_class_name == "TempScaledMoEGPT":
                model = TempScaledMoEGPT(**base_kwargs, **extra_kwargs)
            elif model_class_name == "MarginLossMoEGPT":
                model = MarginLossMoEGPT(**base_kwargs, **extra_kwargs)

            mx.eval(model.parameters())
            n_params = count_params(model)
            print(f"  Params: {n_params:,}")

            result = train(model, train_ds, val_ds, steps=steps,
                          batch_size=32, lr=3e-3, seed=seed, log_every=200)
            val_loss = result["val_loss"]
            print(f"  Val loss: {val_loss:.4f}")

            # Dual-temperature collision measurement:
            # 1. At T=1.0 (raw logit quality -- isolates training dynamics effect)
            collision_t1 = measure_collision_rates(model, val_ds, n_batches=10,
                                                    epsilons=epsilons, temperature=1.0)
            # 2. At training temperature (inference-time effect)
            collision_train_t = measure_collision_rates(model, val_ds, n_batches=10,
                                                         epsilons=epsilons,
                                                         temperature=train_temp)

            print(f"  Collision @T=1.0: " + " | ".join(
                f"eps={eps}: {collision_t1['collision_rates'][str(eps)]:.3f}"
                for eps in epsilons
            ))
            print(f"  Collision @T={train_temp}: " + " | ".join(
                f"eps={eps}: {collision_train_t['collision_rates'][str(eps)]:.3f}"
                for eps in epsilons
            ))

            seed_results.append({
                "seed": seed,
                "val_loss": val_loss,
                "params": n_params,
                "collision_at_T1": collision_t1,
                "collision_at_train_T": collision_train_t,
                "training_temperature": train_temp,
                "tokens_per_sec": result["tokens_per_sec"],
            })

        results.append({
            "config": config_name,
            "model_class": model_class_name,
            "extra_kwargs": extra_kwargs,
            "training_temperature": train_temp,
            "seed_results": seed_results,
        })

    return results


def analyze_scaling(phase1_results):
    """Analyze collision rate scaling with N."""
    print("\n" + "=" * 70)
    print("  ANALYSIS: Collision Rate Scaling")
    print("=" * 70)

    epsilons = ["0.01", "0.05", "0.1"]

    print(f"\n{'N':>4} | {'Params':>8} | {'Val Loss':>10} | "
          + " | ".join(f"eps={e}" for e in epsilons)
          + " | Mean Gap | Median Gap")
    print("-" * 100)

    scaling_data = {}
    for r in phase1_results:
        N = r["N"]
        seeds = r["seed_results"]

        mean_val = sum(s["val_loss"] for s in seeds) / len(seeds)
        params = seeds[0]["params"]

        mean_rates = {}
        for eps in epsilons:
            rates = [s["collision_data"]["collision_rates"][eps] for s in seeds]
            mean_rates[eps] = sum(rates) / len(rates)

        mean_gap = sum(s["collision_data"]["gap_stats"]["mean"] for s in seeds) / len(seeds)
        median_gap = sum(s["collision_data"]["gap_stats"]["median"] for s in seeds) / len(seeds)

        print(f"{N:>4} | {params:>8,} | {mean_val:>10.4f} | "
              + " | ".join(f"{mean_rates[e]:>7.3f}" for e in epsilons)
              + f" | {mean_gap:>8.4f} | {median_gap:>10.4f}")

        scaling_data[N] = {
            "val_loss": mean_val,
            "rates": mean_rates,
            "mean_gap": mean_gap,
            "median_gap": median_gap,
        }

    print("\n--- KC1: Does collision rate increase with N? ---")
    for eps in epsilons:
        rates = [(N, scaling_data[N]["rates"][eps]) for N in sorted(scaling_data)]
        is_monotonic = all(rates[i][1] <= rates[i + 1][1] for i in range(len(rates) - 1))
        ratio_8_64 = rates[-1][1] / rates[0][1] if rates[0][1] > 0 else float('inf')
        print(f"  eps={eps}: {' -> '.join(f'{r[1]:.3f}' for r in rates)} "
              f"(ratio 64/8: {ratio_8_64:.2f}x, monotonic: {is_monotonic})")

    return scaling_data


def analyze_mitigation(phase2_results):
    """Analyze mitigation effectiveness with statistical tests."""
    print("\n" + "=" * 70)
    print("  ANALYSIS: Collision Mitigation at N=32 (5 seeds, dual-T)")
    print("=" * 70)

    # --- Quality comparison ---
    print(f"\n{'Config':<20} | {'Val Loss':>10} | {'Diff%':>8} | {'Std':>7} | "
          "C(0.01)@T=1 | C(0.01)@trainT | p-value")
    print("-" * 105)

    baseline_losses = None
    baseline_mean = None

    for r in phase2_results:
        config = r["config"]
        seeds = r["seed_results"]
        losses = [s["val_loss"] for s in seeds]
        mean_val = sum(losses) / len(losses)
        std_val = math.sqrt(sum((x - mean_val) ** 2 for x in losses) / (len(losses) - 1))

        # Collision rates at T=1.0
        rates_t1 = [s["collision_at_T1"]["collision_rates"]["0.01"] for s in seeds]
        mean_rate_t1 = sum(rates_t1) / len(rates_t1)

        # Collision rates at training T
        rates_tt = [s["collision_at_train_T"]["collision_rates"]["0.01"] for s in seeds]
        mean_rate_tt = sum(rates_tt) / len(rates_tt)

        if baseline_losses is None:
            baseline_losses = losses
            baseline_mean = mean_val
            diff_pct = 0.0
            p_str = "---"
        else:
            diff_pct = (mean_val - baseline_mean) / baseline_mean * 100
            _, p_val, _ = welch_t_test(baseline_losses, losses)
            p_str = f"{p_val:.4f}"

        print(f"{config:<20} | {mean_val:>10.4f} | {diff_pct:>+8.2f}% | {std_val:>7.4f} | "
              f"{mean_rate_t1:>11.3f} | {mean_rate_tt:>13.3f} | {p_str:>7}")

    # --- Per-seed detail for T=2.0 anomaly investigation ---
    print("\n--- Per-seed collision rates @T=1.0 (investigating T=2.0 anomaly) ---")
    for r in phase2_results:
        config = r["config"]
        seeds = r["seed_results"]
        print(f"  {config}:")
        for s in seeds:
            c01 = s["collision_at_T1"]["collision_rates"]["0.01"]
            print(f"    seed={s['seed']}: C(0.01)@T=1.0={c01:.3f}, val_loss={s['val_loss']:.4f}")

    # --- KC2 with p-values ---
    print("\n--- KC2: Best mitigation >0.5% improvement at p<0.05? ---")
    best_improvement = 0.0
    best_config = None
    best_p = 1.0
    for r in phase2_results:
        if r["config"] == "baseline_T1.0":
            continue
        losses = [s["val_loss"] for s in r["seed_results"]]
        mean_val = sum(losses) / len(losses)
        improvement = (baseline_mean - mean_val) / baseline_mean * 100
        _, p_val, _ = welch_t_test(baseline_losses, losses)
        print(f"  {r['config']}: {improvement:+.2f}% improvement, p={p_val:.4f}")
        if improvement > best_improvement:
            best_improvement = improvement
            best_config = r["config"]
            best_p = p_val

    passes_magnitude = best_improvement > 0.5
    passes_significance = best_p < 0.05
    if passes_magnitude and passes_significance:
        verdict = "PASSES (magnitude and significance)"
    elif passes_magnitude and not passes_significance:
        verdict = f"INCONCLUSIVE (magnitude OK at {best_improvement:.2f}%, but p={best_p:.4f} > 0.05)"
    else:
        verdict = "KILLED"

    print(f"\n  Best: {best_config} with {best_improvement:+.2f}% (p={best_p:.4f})")
    print(f"  KC2 verdict: {verdict}")

    # --- Dual-T decomposition ---
    print("\n--- Dual-Temperature Decomposition ---")
    print("  Separating training dynamics (learned logits) from inference sharpening:")
    for r in phase2_results:
        config = r["config"]
        train_t = r["training_temperature"]
        seeds = r["seed_results"]

        rate_t1 = sum(s["collision_at_T1"]["collision_rates"]["0.01"] for s in seeds) / len(seeds)
        rate_tt = sum(s["collision_at_train_T"]["collision_rates"]["0.01"] for s in seeds) / len(seeds)

        if config == "baseline_T1.0":
            baseline_rate = rate_t1
            print(f"  {config}: C@T=1.0={rate_t1:.3f} (reference)")
        else:
            training_effect = baseline_rate - rate_t1  # positive = training produced sharper logits
            inference_effect = rate_t1 - rate_tt  # positive = temperature further sharpens
            total_effect = baseline_rate - rate_tt
            print(f"  {config}: C@T=1.0={rate_t1:.3f}, C@T={train_t}={rate_tt:.3f}")
            print(f"    Training effect (learned logits): {training_effect:+.3f}")
            print(f"    Inference effect (T scaling):     {inference_effect:+.3f}")
            print(f"    Total effect:                     {total_effect:+.3f}")


def fit_scaling_law(scaling_data):
    """Fit collision_rate = a * N^b to the data."""
    print("\n--- Scaling Law Fit: collision_rate = a * N^b ---")
    for eps in ["0.01", "0.05", "0.1"]:
        Ns = sorted(scaling_data.keys())
        rates = [scaling_data[N]["rates"][eps] for N in Ns]

        log_Ns = [math.log(N) for N in Ns]
        log_rates = [math.log(max(r, 1e-10)) for r in rates]

        n = len(Ns)
        mean_lnN = sum(log_Ns) / n
        mean_lnR = sum(log_rates) / n
        cov = sum((log_Ns[i] - mean_lnN) * (log_rates[i] - mean_lnR) for i in range(n))
        var = sum((log_Ns[i] - mean_lnN) ** 2 for i in range(n))

        b = cov / var if var > 0 else 0
        ln_a = mean_lnR - b * mean_lnN
        a = math.exp(ln_a)

        ss_res = sum((log_rates[i] - (ln_a + b * log_Ns[i])) ** 2 for i in range(n))
        ss_tot = sum((log_rates[i] - mean_lnR) ** 2 for i in range(n))
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"  eps={eps}: rate = {a:.4f} * N^{b:.3f} (r^2={r_sq:.3f})")
        for N_ext in [128, 256, 512]:
            predicted = min(1.0, a * (N_ext ** b))
            print(f"    N={N_ext}: predicted rate = {predicted:.3f}")


def main():
    t0 = time.time()

    # Phase 1: Scaling (keep from v1 -- 3 seeds is fine for KC1)
    phase1_seeds = (42, 123, 777)
    phase1 = run_phase1_scaling(seeds=phase1_seeds, steps=500)
    scaling_data = analyze_scaling(phase1)
    fit_scaling_law(scaling_data)

    # Phase 2: Mitigation (REVISED: 5 seeds, dual-T measurement)
    phase2_seeds = (42, 123, 777, 2024, 314)
    phase2 = run_phase2_mitigation(seeds=phase2_seeds, steps=500)
    analyze_mitigation(phase2)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save results
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump({
            "phase1_scaling": phase1,
            "phase2_mitigation": phase2,
            "revision_notes": "v2: 5 seeds, dual-T measurement, p-values",
            "total_runtime_s": elapsed,
        }, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
