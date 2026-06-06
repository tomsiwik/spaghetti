#!/usr/bin/env python3
"""Analyze FFN-only vs all-modules adapters: quality and orthogonality.

Runs on CPU using safetensors + numpy. No GPU needed.

This script:
1. Loads PPL metrics from training summaries
2. Computes orthogonality matrices for both adapter sets
3. Compares independently-trained FFN-only orthogonality with retroactive subset
4. Evaluates kill criteria

Usage:
    python3 experiments/models/ffn_only_matched_rank/analyze.py \
        --ffn-adapters adapters_ffn_only/ \
        --all-adapters adapters/ \
        --retroactive-results experiments/models/ffn_only_vs_all_modules/results.json
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


DOMAINS = ["bash", "math", "medical", "python", "sql"]


def load_adapter_weights(adapter_dir: Path) -> dict:
    """Load safetensors weights from an adapter directory."""
    from safetensors.numpy import load_file
    sf = adapter_dir / "adapter_model.safetensors"
    if not sf.exists():
        raise FileNotFoundError(f"No adapter_model.safetensors in {adapter_dir}")
    return load_file(str(sf))


def extract_module_vectors(weights: dict, module_filter: str = None) -> np.ndarray:
    """Flatten adapter weights into a single vector, optionally filtering by module type.

    module_filter: 'ffn' filters to .mlp. keys, 'attn' to .self_attn. keys, None = all
    """
    parts = []
    for k in sorted(weights.keys()):
        if module_filter == "ffn" and ".mlp." not in k:
            continue
        if module_filter == "attn" and ".self_attn." not in k:
            continue
        parts.append(weights[k].flatten())

    if not parts:
        return np.zeros(1)
    return np.concatenate(parts)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / (norm + 1e-12))


def compute_pairwise_cosines(adapter_dirs: dict[str, Path],
                              module_filter: str = None) -> dict:
    """Compute pairwise cosine matrix for a set of adapters."""
    vectors = {}
    for domain, path in sorted(adapter_dirs.items()):
        weights = load_adapter_weights(path)
        vectors[domain] = extract_module_vectors(weights, module_filter)

    domains = sorted(vectors.keys())
    cosines = {}
    abs_cosines = []

    for i, d1 in enumerate(domains):
        for j in range(i + 1, len(domains)):
            d2 = domains[j]
            cos = cosine_similarity(vectors[d1], vectors[d2])
            pair_key = f"{d1}-{d2}"
            cosines[pair_key] = cos
            abs_cosines.append(abs(cos))

    return {
        "pairs": cosines,
        "mean_abs_cos": statistics.mean(abs_cosines) if abs_cosines else 0.0,
        "std_abs_cos": statistics.stdev(abs_cosines) if len(abs_cosines) > 1 else 0.0,
        "max_abs_cos": max(abs_cosines) if abs_cosines else 0.0,
        "n_pairs": len(abs_cosines),
        "dims": {d: len(vectors[d]) for d in domains},
    }


def load_metrics(adapter_base_dir: Path) -> dict:
    """Load per-domain training metrics from adapter directories."""
    metrics = {}
    for domain in DOMAINS:
        metrics_file = adapter_base_dir / domain / "metrics.json"
        if metrics_file.exists():
            metrics[domain] = json.loads(metrics_file.read_text())
    return metrics


def ortho_stats_excl_outlier(cosines: dict, outlier_pair: str = "math-medical") -> dict:
    """Report orthogonality stats with and without the dominant pair (reviewer rec #3)."""
    all_abs = [abs(v) for v in cosines.values()]
    excl_abs = [abs(v) for k, v in cosines.items() if k != outlier_pair]

    result = {
        "mean_abs_cos": statistics.mean(all_abs) if all_abs else 0.0,
        "median_abs_cos": statistics.median(all_abs) if all_abs else 0.0,
    }
    if excl_abs:
        result["mean_abs_cos_excl_math_medical"] = statistics.mean(excl_abs)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FFN-only vs all-modules adapters")
    parser.add_argument("--ffn-adapters", default="adapters_ffn_only/",
                        help="Directory with FFN-only adapters")
    parser.add_argument("--all-adapters", default="adapters/",
                        help="Directory with all-modules adapters")
    parser.add_argument("--all-retrained", default=None,
                        help="Directory with retrained all-modules adapters (if available)")
    parser.add_argument("--retroactive-results",
                        default="experiments/models/ffn_only_vs_all_modules/results.json",
                        help="Path to previous retroactive subset results")
    parser.add_argument("--output",
                        default="experiments/models/ffn_only_matched_rank/results.json",
                        help="Output results JSON")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to aggregate across (looks for seed_N/ subdirs)")

    args = parser.parse_args()
    ffn_dir = Path(args.ffn_adapters)
    all_dir = Path(args.all_adapters)
    retro_path = Path(args.retroactive_results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    seeds = args.seeds or [None]  # None = no seed subdirs

    # ---- Part 1: Quality Comparison (PPL) ----
    print("=" * 70)
    print("  PART 1: Quality Comparison (Perplexity)")
    print("=" * 70)

    # Aggregate metrics across seeds
    all_seed_ffn_metrics = {}
    all_seed_all_metrics = {}
    for seed in seeds:
        seed_suffix = f"seed_{seed}" if seed is not None else ""
        ffn_seed_dir = ffn_dir / seed_suffix if seed_suffix else ffn_dir
        ffn_m = load_metrics(ffn_seed_dir)
        all_seed_ffn_metrics[seed] = ffn_m

        if args.all_retrained:
            all_ret_dir = Path(args.all_retrained) / seed_suffix if seed_suffix else Path(args.all_retrained)
            all_seed_all_metrics[seed] = load_metrics(all_ret_dir)

    # Use first seed's metrics for display, aggregate PPL across seeds
    ffn_metrics = all_seed_ffn_metrics.get(seeds[0], {})
    all_metrics_orig = load_metrics(all_dir)  # original adapters as fallback

    # Also check for retrained all-modules metrics
    all_retrained_metrics = all_seed_all_metrics.get(seeds[0], {})

    if ffn_metrics:
        multi_seed = len(seeds) > 1 and seeds[0] is not None

        if multi_seed:
            print(f"\n  Multi-seed analysis ({len(seeds)} seeds: {seeds})")
            print(f"\n  {'Domain':<12} {'FFN PPL (mean±std)':<22} {'All PPL (mean±std)':<22} {'Gap %':<10} {'Kill?'}")
        else:
            print(f"\n  {'Domain':<12} {'FFN-only PPL':<15} {'All-mods PPL':<15} {'Gap %':<10} {'Kill?'}")
        print("  " + "-" * 72)

        ppl_gaps = []
        kill_quality = False

        for domain in DOMAINS:
            # Collect PPL across seeds
            ffn_ppls_for_domain = []
            all_ppls_for_domain = []
            for seed in seeds:
                fm = all_seed_ffn_metrics.get(seed, {})
                if fm.get(domain, {}).get("eval_ppl") is not None:
                    ffn_ppls_for_domain.append(fm[domain]["eval_ppl"])
                am = all_seed_all_metrics.get(seed, {})
                if am.get(domain, {}).get("eval_ppl") is not None:
                    all_ppls_for_domain.append(am[domain]["eval_ppl"])

            # Fallback to original adapters if no retrained
            if not all_ppls_for_domain:
                orig_ppl = all_metrics_orig.get(domain, {}).get("eval_ppl")
                if orig_ppl is not None:
                    all_ppls_for_domain = [orig_ppl]

            if not ffn_ppls_for_domain:
                print(f"  {domain:<12} {'N/A':<22} {'-':<22}")
                continue

            ffn_mean = statistics.mean(ffn_ppls_for_domain)
            ffn_std = statistics.stdev(ffn_ppls_for_domain) if len(ffn_ppls_for_domain) > 1 else 0.0

            if all_ppls_for_domain:
                all_mean = statistics.mean(all_ppls_for_domain)
                all_std = statistics.stdev(all_ppls_for_domain) if len(all_ppls_for_domain) > 1 else 0.0
                gap_pct = (ffn_mean - all_mean) / all_mean * 100
                ppl_gaps.append(gap_pct)
                kill_domain = gap_pct > 5.0
                if kill_domain:
                    kill_quality = True
                if multi_seed:
                    print(f"  {domain:<12} {ffn_mean:<8.4f}±{ffn_std:<10.4f} {all_mean:<8.4f}±{all_std:<10.4f} "
                          f"{gap_pct:>+8.2f}%  {'KILL' if kill_domain else 'OK'}")
                else:
                    print(f"  {domain:<12} {ffn_mean:<15.4f} {all_mean:<15.4f} "
                          f"{gap_pct:>+8.2f}%  {'KILL' if kill_domain else 'OK'}")
            else:
                if multi_seed:
                    print(f"  {domain:<12} {ffn_mean:<8.4f}±{ffn_std:<10.4f} {'N/A':<22}")
                else:
                    print(f"  {domain:<12} {ffn_mean:<15.4f} {'N/A':<15}")

        if ppl_gaps:
            mean_gap = statistics.mean(ppl_gaps)
            max_gap = max(ppl_gaps)
            print(f"\n  Mean PPL gap: {mean_gap:+.2f}%")
            print(f"  Max PPL gap:  {max_gap:+.2f}%")
            print(f"  Kill criterion (>5% gap): {'TRIGGERED' if kill_quality else 'PASSED'}")

            results["quality"] = {
                "seeds": seeds if multi_seed else [None],
                "ffn_ppls": {d: ffn_metrics.get(d, {}).get("eval_ppl") for d in DOMAINS},
                "all_ppls": {d: (all_retrained_metrics.get(d, {}).get("eval_ppl")
                                or all_metrics_orig.get(d, {}).get("eval_ppl")) for d in DOMAINS},
                "ppl_gaps_pct": {d: g for d, g in zip(DOMAINS, ppl_gaps)} if ppl_gaps else {},
                "mean_gap_pct": mean_gap,
                "max_gap_pct": max_gap,
                "kill_triggered": kill_quality,
            }
    else:
        print("\n  No FFN-only metrics found. Training not yet completed.")
        print("  Run train_ffn_only.py on RunPod first.")
        results["quality"] = {"status": "NOT_AVAILABLE"}

    # ---- Part 2: Orthogonality Analysis ----
    print("\n" + "=" * 70)
    print("  PART 2: Orthogonality Analysis")
    print("=" * 70)

    # Check which adapter sets are available
    ffn_dirs = {d: ffn_dir / d for d in DOMAINS if (ffn_dir / d / "adapter_model.safetensors").exists()}
    all_dirs = {d: all_dir / d for d in DOMAINS if (all_dir / d / "adapter_model.safetensors").exists()}

    print(f"\n  FFN-only adapters found: {len(ffn_dirs)} ({', '.join(sorted(ffn_dirs))})")
    print(f"  All-modules adapters found: {len(all_dirs)} ({', '.join(sorted(all_dirs))})")

    # 2a: FFN-only adapter full orthogonality
    if len(ffn_dirs) >= 2:
        print("\n  --- FFN-only adapters (independently trained, all params) ---")
        ffn_ortho = compute_pairwise_cosines(ffn_dirs)
        excl_stats = ortho_stats_excl_outlier(ffn_ortho["pairs"])
        ffn_ortho.update(excl_stats)
        print(f"  Mean |cos|: {ffn_ortho['mean_abs_cos']:.6f}")
        print(f"  Median |cos|: {ffn_ortho['median_abs_cos']:.6f}")
        if "mean_abs_cos_excl_math_medical" in excl_stats:
            print(f"  Mean |cos| (excl math-medical): {excl_stats['mean_abs_cos_excl_math_medical']:.6f}")
        print(f"  Max  |cos|: {ffn_ortho['max_abs_cos']:.6f}")
        print(f"  Pairs:")
        for pair, cos in sorted(ffn_ortho["pairs"].items()):
            print(f"    {pair:<25} {cos:>10.6f}")
        results["ffn_independent_ortho"] = ffn_ortho
    else:
        print("\n  Insufficient FFN-only adapters for orthogonality analysis.")
        results["ffn_independent_ortho"] = {"status": "NOT_AVAILABLE"}

    # 2b: All-modules adapter orthogonality (full params)
    if len(all_dirs) >= 2:
        print("\n  --- All-modules adapters (full params) ---")
        all_ortho_full = compute_pairwise_cosines(all_dirs)
        print(f"  Mean |cos|: {all_ortho_full['mean_abs_cos']:.6f}")
        print(f"  Max  |cos|: {all_ortho_full['max_abs_cos']:.6f}")
        results["all_modules_full_ortho"] = all_ortho_full

        # 2c: Retroactive FFN subset from all-modules adapters
        print("\n  --- All-modules adapters (FFN subset only, retroactive) ---")
        all_ortho_ffn_subset = compute_pairwise_cosines(all_dirs, module_filter="ffn")
        print(f"  Mean |cos|: {all_ortho_ffn_subset['mean_abs_cos']:.6f}")
        print(f"  Max  |cos|: {all_ortho_ffn_subset['max_abs_cos']:.6f}")
        results["all_modules_ffn_subset_ortho"] = all_ortho_ffn_subset

        # 2d: Retroactive attention subset
        print("\n  --- All-modules adapters (Attn subset only, retroactive) ---")
        all_ortho_attn_subset = compute_pairwise_cosines(all_dirs, module_filter="attn")
        print(f"  Mean |cos|: {all_ortho_attn_subset['mean_abs_cos']:.6f}")
        print(f"  Max  |cos|: {all_ortho_attn_subset['max_abs_cos']:.6f}")
        results["all_modules_attn_subset_ortho"] = all_ortho_attn_subset

    # ---- Part 3: Compare Independent vs Retroactive ----
    print("\n" + "=" * 70)
    print("  PART 3: Independent vs Retroactive FFN-only Orthogonality")
    print("=" * 70)

    retro_data = None
    if retro_path.exists():
        retro_data = json.loads(retro_path.read_text())
        retro_ffn_mean = retro_data.get("real_adapters", {}).get("ffn_mean_abs_cos")
        print(f"\n  Previous retroactive FFN-only mean |cos|: {retro_ffn_mean:.6f}")
    else:
        print(f"\n  No retroactive results at {retro_path}")

    if len(ffn_dirs) >= 2 and retro_data:
        independent_mean = results.get("ffn_independent_ortho", {}).get("mean_abs_cos")
        retro_ffn_mean = retro_data.get("real_adapters", {}).get("ffn_mean_abs_cos")

        if independent_mean is not None and retro_ffn_mean is not None and retro_ffn_mean > 0:
            ortho_diff_pct = abs(independent_mean - retro_ffn_mean) / retro_ffn_mean * 100
            kill_ortho = ortho_diff_pct > 50.0

            print(f"  Independent FFN-only mean |cos|: {independent_mean:.6f}")
            print(f"  Retroactive FFN subset mean |cos|: {retro_ffn_mean:.6f}")
            print(f"  Difference: {ortho_diff_pct:.1f}%")
            print(f"  Kill criterion (>50% diff): {'TRIGGERED' if kill_ortho else 'PASSED'}")

            results["independent_vs_retroactive"] = {
                "independent_mean_abs_cos": independent_mean,
                "retroactive_mean_abs_cos": retro_ffn_mean,
                "difference_pct": ortho_diff_pct,
                "kill_triggered": kill_ortho,
            }

            # Pairwise comparison
            retro_ffn_cosines = retro_data.get("real_adapters", {}).get("ffn_cosines", [])
            if retro_ffn_cosines and "ffn_independent_ortho" in results:
                print(f"\n  Pairwise comparison (independent vs retroactive):")
                print(f"  {'Pair':<25} {'Independent':>12} {'Retroactive':>12} {'Diff':>10}")
                print("  " + "-" * 62)

                # Reconstruct retroactive pair order
                retro_domains = retro_data.get("real_adapters", {}).get("domains", DOMAINS)
                retro_pair_map = {}
                idx = 0
                for i, d1 in enumerate(retro_domains):
                    for j in range(i + 1, len(retro_domains)):
                        d2 = retro_domains[j]
                        pair_key = f"{d1}-{d2}"
                        retro_pair_map[pair_key] = retro_ffn_cosines[idx]
                        idx += 1

                indep_pairs = results["ffn_independent_ortho"]["pairs"]
                for pair_key in sorted(indep_pairs.keys()):
                    ind_cos = indep_pairs[pair_key]
                    retro_cos = retro_pair_map.get(pair_key)
                    if retro_cos is not None:
                        diff = ind_cos - retro_cos
                        print(f"  {pair_key:<25} {ind_cos:>12.6f} {retro_cos:>12.6f} {diff:>10.6f}")
    else:
        print("\n  Cannot compare: need both independent FFN-only adapters and retroactive data.")

    # ---- Part 4: Kill Criteria Summary ----
    print("\n" + "=" * 70)
    print("  KILL CRITERIA SUMMARY")
    print("=" * 70)

    quality_kill = results.get("quality", {}).get("kill_triggered", None)
    ortho_kill = results.get("independent_vs_retroactive", {}).get("kill_triggered", None)

    print(f"\n  K1: FFN-only PPL >5% higher than all-modules: "
          f"{'TRIGGERED' if quality_kill else 'PASSED' if quality_kill is not None else 'NOT TESTED'}")
    print(f"  K2: Orthogonality differs >50% from retroactive: "
          f"{'TRIGGERED' if ortho_kill else 'PASSED' if ortho_kill is not None else 'NOT TESTED'}")

    overall_kill = (quality_kill is True) or (ortho_kill is True)
    overall_pass = (quality_kill is False) and (ortho_kill is False)

    if overall_kill:
        print(f"\n  VERDICT: KILL -- FFN-only architecture pivot NOT confirmed")
    elif overall_pass:
        print(f"\n  VERDICT: PROVEN -- FFN-only matches all-modules at same rank")
    else:
        print(f"\n  VERDICT: INCOMPLETE -- awaiting training results")

    results["verdict"] = {
        "quality_kill": quality_kill,
        "ortho_kill": ortho_kill,
        "overall": "KILL" if overall_kill else "PROVEN" if overall_pass else "INCOMPLETE",
    }

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
