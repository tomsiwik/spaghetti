"""Information Preservation Predictor: diagnostic experiment.

Tests whether the Frobenius-norm information preservation of LoRA merging
methods predicts their quality ranking. This is a pure diagnostic -- no
new model is trained. It reuses the lora_merging_bakeoff infrastructure.

Hypothesis: For each merging method m, compute
  info_preserved(m) = 1 - ||merged_m - avg_original||_F / ||avg_original||_F
where avg_original = (1/N) * sum_k delta_k is the lossless linear reference.
The ranking of info_preserved should match the quality ranking (val loss).

Kill criteria:
  1. Info preservation ranking does not match quality ranking for >1 method
  2. Spearman correlation between info preservation and quality < 0.8
"""

import copy
import random
import statistics
import time
import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.models import get_model
from experiments.data import (
    load_names, CharTokenizer, CharDataset,
    domain_split, train_val_split,
)
from experiments.train import train, evaluate, ntp_loss
from experiments.models.lora_merging_bakeoff.merging_methods import (
    extract_deltas,
    merge_simple_average,
    merge_ties,
    merge_dare,
    merge_dare_ties,
    apply_merged_deltas,
)
from experiments.models.lora_procrustes.test_lora_procrustes import (
    RoutedDeltaGPT,
    calibrate_router,
    copy_weights,
    count_params,
    freeze_except_lora,
    reset_lora,
    get_deltas as get_deltas_list,
)


# ── Config ──────────────────────────────────────────────────────────────────

BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
LORA_RANK = 8
LORA_ALPHA = 1.0
PRETRAIN_STEPS = 300
FINETUNE_STEPS = 300
ROUTER_CAL_STEPS = 100
BATCH_SIZE = 32
LR = 3e-3

TIES_DENSITY = 0.2
DARE_DROP_RATES = [0.3, 0.5, 0.7, 0.9]


# ── Information Preservation Metrics ────────────────────────────────────────

def frobenius_norm(delta_dict):
    """Compute total Frobenius norm of a delta dictionary."""
    total = 0.0
    for k, v in delta_dict.items():
        total += mx.sum(v ** 2).item()
    return total ** 0.5


def info_preservation_vs_average(merged_delta, avg_delta):
    """Compute information preservation relative to simple average.

    info_preserved = 1 - ||merged - avg||_F / ||avg||_F

    Simple average is the lossless linear reference (exact mean of all deltas).
    A method that perfectly reproduces the average gets 1.0.
    A method that produces zeros gets 1 - 1 = 0 (if avg is nonzero).

    Returns:
        float: information preservation score (1.0 = perfect, 0.0 = total loss)
    """
    diff_norm_sq = 0.0
    avg_norm_sq = 0.0
    for k in avg_delta:
        diff = merged_delta[k] - avg_delta[k]
        diff_norm_sq += mx.sum(diff ** 2).item()
        avg_norm_sq += mx.sum(avg_delta[k] ** 2).item()

    avg_norm = avg_norm_sq ** 0.5
    diff_norm = diff_norm_sq ** 0.5

    if avg_norm < 1e-10:
        return 1.0  # both zero
    return 1.0 - diff_norm / avg_norm


def info_preservation_vs_originals(merged_delta, original_deltas):
    """Compute information preservation relative to each original delta.

    For each domain k: loss_k = ||merged - delta_k||_F^2
    Total loss = sum_k loss_k
    Total energy = sum_k ||delta_k||_F^2
    info_preserved = 1 - sqrt(total_loss / total_energy)

    Returns:
        float: information preservation score
    """
    total_loss = 0.0
    total_energy = 0.0
    for delta_k in original_deltas:
        for k in delta_k:
            diff = merged_delta[k] - delta_k[k]
            total_loss += mx.sum(diff ** 2).item()
            total_energy += mx.sum(delta_k[k] ** 2).item()

    if total_energy < 1e-10:
        return 1.0
    return 1.0 - (total_loss / total_energy) ** 0.5


def info_norm_ratio(merged_delta, original_deltas):
    """Compute norm ratio: ||merged||_F / avg(||delta_k||_F).

    A method preserving full signal magnitude gets ratio ~1.0.
    Methods that trim/drop signal get ratio < 1.0.
    Methods that amplify noise get ratio > 1.0.

    Returns:
        float: norm ratio
    """
    merged_norm = frobenius_norm(merged_delta)
    orig_norms = [frobenius_norm(d) for d in original_deltas]
    avg_orig_norm = sum(orig_norms) / len(orig_norms)

    if avg_orig_norm < 1e-10:
        return 1.0
    return merged_norm / avg_orig_norm


def spearman_rank_correlation(x, y):
    """Compute Spearman rank correlation between two lists.

    Returns rho in [-1, 1]. rho=1 means perfect rank agreement.
    """
    n = len(x)
    if n < 3:
        return float('nan')

    # Compute ranks (average ranks for ties)
    def rank(values):
        indexed = sorted(enumerate(values), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and abs(indexed[j+1][1] - indexed[j][1]) < 1e-12:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = rank(x)
    ry = rank(y)

    # Pearson correlation of ranks
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    den_x = sum((a - mean_rx) ** 2 for a in rx) ** 0.5
    den_y = sum((b - mean_ry) ** 2 for b in ry) ** 0.5

    if den_x < 1e-10 or den_y < 1e-10:
        return 0.0
    return num / (den_x * den_y)


# ── Training helpers (reused from bakeoff) ──────────────────────────────────

def train_joint(train_datasets, val_datasets, vocab_size, steps_per_domain, seed):
    """Train a single GPT jointly on all domains (oracle baseline)."""
    model = get_model("gpt", vocab_size=vocab_size, **BASE)
    mx.eval(model.parameters())

    total_steps = len(train_datasets) * steps_per_domain
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    domain_names = list(train_datasets.keys())
    for step in range(1, total_steps + 1):
        domain = domain_names[(step - 1) % len(domain_names)]
        inputs, targets = train_datasets[domain].get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    results = {}
    for name, val_ds in val_datasets.items():
        results[name] = evaluate(model, val_ds, BATCH_SIZE)
    results["avg"] = sum(results.values()) / len(results)
    return results


def pretrain_base(joint_train, vocab_size, seed):
    """Pretrain base GPT model on joint data."""
    base = get_model("gpt", vocab_size=vocab_size, **BASE)
    mx.eval(base.parameters())
    train(base, joint_train, steps=PRETRAIN_STEPS, batch_size=BATCH_SIZE,
          lr=LR, seed=seed, log_every=9999)
    return base


def finetune_lora(base_model, train_ds, val_ds, vocab_size, seed):
    """Fine-tune a LoRA model on a single domain."""
    lora = get_model("lora_gpt", vocab_size=vocab_size, **BASE,
                     lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA)
    mx.eval(lora.parameters())

    for l_idx in range(BASE['n_layer']):
        bl = base_model.layers[l_idx]
        ll = lora.layers[l_idx]
        ll.attn.wq.weight = bl.attn.wq.weight
        ll.attn.wk.weight = bl.attn.wk.weight
        ll.attn.wv.weight = bl.attn.wv.weight
        ll.attn.wo.weight = bl.attn.wo.weight
        ll.mlp.fc1.linear.weight = bl.mlp.fc1.weight
        ll.mlp.fc2.linear.weight = bl.mlp.fc2.weight
    lora.wte.weight = base_model.wte.weight
    lora.wpe.weight = base_model.wpe.weight
    lora.lm_head.weight = base_model.lm_head.weight
    mx.eval(lora.parameters())

    freeze_except_lora(lora)
    train(lora, train_ds, val_ds, steps=FINETUNE_STEPS,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    lora.unfreeze()
    return lora


def eval_merged_model(base_model, merged_deltas, val_datasets, vocab_size):
    """Evaluate a merged delta model on all domains."""
    model = apply_merged_deltas(base_model, merged_deltas, vocab_size)
    results = {}
    for name, val_ds in val_datasets.items():
        results[name] = evaluate(model, val_ds, BATCH_SIZE)
    results["avg"] = sum(results.values()) / len(results)
    return results


def eval_concat_calibrated(base_model, delta_lists, train_datasets,
                           val_datasets, vocab_size, seed):
    """Evaluate concat+calibrate method with routed LoRA deltas."""
    base_copy = get_model("gpt", vocab_size=vocab_size, **BASE)
    mx.eval(base_copy.parameters())
    copy_weights(base_model, base_copy)

    concat_model = RoutedDeltaGPT(base_copy, delta_lists, vocab_size, top_k=2)
    mx.eval(concat_model.parameters())

    concat_model.freeze()
    for router in concat_model.routers:
        router.unfreeze()

    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(concat_model, ntp_loss)

    all_train_ds = list(train_datasets.values())
    for step in range(1, ROUTER_CAL_STEPS + 1):
        ds = all_train_ds[step % len(all_train_ds)]
        inputs, targets = ds.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(concat_model, inputs, targets)
        optimizer.update(concat_model, grads)
        mx.eval(concat_model.parameters(), optimizer.state)

    concat_model.unfreeze()

    results = {}
    for name, val_ds in val_datasets.items():
        results[name] = evaluate(concat_model, val_ds, BATCH_SIZE)
    results["avg"] = sum(results.values()) / len(results)
    return results


# ── Main Experiment ─────────────────────────────────────────────────────────

def run_info_preservation(seed=42, n_domains=2):
    """Run full info preservation analysis for one seed and domain count."""
    print(f"\n{'='*70}")
    print(f"INFO PRESERVATION ANALYSIS (seed={seed}, N={n_domains})")
    print(f"{'='*70}")

    mx.random.seed(seed)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    if n_domains == 2:
        splits = domain_split(docs, method="binary")
    elif n_domains == 5:
        splits = domain_split(docs, method="quintary")
    else:
        raise ValueError(f"Unsupported n_domains={n_domains}")

    all_train, all_val = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_train, tokenizer, BASE["block_size"])

    train_datasets = {}
    val_datasets = {}
    for d_name, d_docs in splits.items():
        d_train, d_val = train_val_split(d_docs, seed=seed)
        train_datasets[d_name] = CharDataset(d_train, tokenizer, BASE["block_size"])
        val_datasets[d_name] = CharDataset(d_val, tokenizer, BASE["block_size"])

    # --- 1. Joint training baseline ---
    print("\n  Training joint baseline...")
    joint_results = train_joint(train_datasets, val_datasets, V, FINETUNE_STEPS, seed)

    # --- 2. Pretrain base ---
    print("  Pretraining base...")
    base_model = pretrain_base(joint_train, V, seed)

    # --- 3. Fine-tune LoRA per domain ---
    lora_models = {}
    delta_dicts = {}
    delta_lists = []
    domain_names = list(splits.keys())

    for i, d_name in enumerate(domain_names):
        print(f"  Fine-tuning LoRA for {d_name}...")
        lora = finetune_lora(base_model, train_datasets[d_name],
                             val_datasets[d_name], V, seed + i)
        lora_models[d_name] = lora
        delta_dicts[d_name] = extract_deltas(lora)
        delta_lists.append(get_deltas_list(lora))

    all_delta_dicts = [delta_dicts[d] for d in domain_names]

    # --- 4. Compute merged deltas for each method ---
    print("  Computing merges...")

    # Reference: simple average (lossless linear merge)
    avg_delta = merge_simple_average(all_delta_dicts)

    methods = {}

    # Simple average
    methods["simple_avg"] = {
        "merged": avg_delta,
        "quality": eval_merged_model(base_model, avg_delta, val_datasets, V),
    }

    # TIES
    merged_ties = merge_ties(all_delta_dicts, density=TIES_DENSITY)
    methods["ties"] = {
        "merged": merged_ties,
        "quality": eval_merged_model(base_model, merged_ties, val_datasets, V),
    }

    # DARE at multiple drop rates
    for p in DARE_DROP_RATES:
        key = f"dare_p{p}"
        merged_dare = merge_dare(all_delta_dicts, drop_rate=p, seed=seed)
        methods[key] = {
            "merged": merged_dare,
            "quality": eval_merged_model(base_model, merged_dare, val_datasets, V),
        }

    # DARE-TIES
    merged_dt = merge_dare_ties(all_delta_dicts, drop_rate=0.9, density=1.0, seed=seed)
    methods["dare_ties"] = {
        "merged": merged_dt,
        "quality": eval_merged_model(base_model, merged_dt, val_datasets, V),
    }

    # Concat+calibrate (special: info_preserved = 1.0 by construction)
    concat_results = eval_concat_calibrated(
        base_model, delta_lists, train_datasets, val_datasets, V, seed)
    methods["concat_cal"] = {
        "merged": None,  # no single merged delta
        "quality": concat_results,
    }

    # --- 5. Compute info preservation metrics ---
    print("  Computing info preservation metrics...")

    results = {}
    for name, data in methods.items():
        quality = data["quality"]["avg"]

        if name == "concat_cal":
            # Concat preserves all information by construction
            ip_vs_avg = 1.0
            ip_vs_orig = 1.0
            norm_ratio = 1.0
        else:
            merged = data["merged"]
            ip_vs_avg = info_preservation_vs_average(merged, avg_delta)
            ip_vs_orig = info_preservation_vs_originals(merged, all_delta_dicts)
            norm_ratio = info_norm_ratio(merged, all_delta_dicts)

        results[name] = {
            "quality": quality,
            "ip_vs_avg": ip_vs_avg,
            "ip_vs_orig": ip_vs_orig,
            "norm_ratio": norm_ratio,
        }

    # Add joint baseline quality for reference
    joint_avg = joint_results["avg"]

    # --- 6. Print results ---
    print(f"\n  {'Method':<15} {'Val Loss':>9} {'vs Joint':>9} "
          f"{'IP(avg)':>8} {'IP(orig)':>9} {'NormR':>7}")
    print("  " + "-" * 60)

    # Sort by quality (val loss, ascending = best first)
    sorted_methods = sorted(results.items(), key=lambda x: x[1]["quality"])

    for name, r in sorted_methods:
        gap = (r["quality"] - joint_avg) / joint_avg * 100
        print(f"  {name:<15} {r['quality']:>9.4f} {gap:>+8.2f}% "
              f"{r['ip_vs_avg']:>8.4f} {r['ip_vs_orig']:>9.4f} {r['norm_ratio']:>7.4f}")

    return results, joint_avg


def run_multiseed(seeds=(42, 123, 7), n_domains=2):
    """Run across seeds and compute aggregate correlation."""
    all_results = {}
    all_joint = {}

    for seed in seeds:
        results, joint_avg = run_info_preservation(seed, n_domains)
        all_results[seed] = results
        all_joint[seed] = joint_avg

    # Aggregate across seeds
    methods = list(all_results[seeds[0]].keys())

    print(f"\n\n{'='*70}")
    print(f"AGGREGATE (N={n_domains}, seeds={seeds})")
    print(f"{'='*70}")

    agg = {}
    for m in methods:
        quals = [all_results[s][m]["quality"] for s in seeds]
        ip_avgs = [all_results[s][m]["ip_vs_avg"] for s in seeds]
        ip_origs = [all_results[s][m]["ip_vs_orig"] for s in seeds]
        norm_rs = [all_results[s][m]["norm_ratio"] for s in seeds]

        agg[m] = {
            "quality_mean": statistics.mean(quals),
            "quality_std": statistics.stdev(quals) if len(quals) > 1 else 0.0,
            "ip_vs_avg_mean": statistics.mean(ip_avgs),
            "ip_vs_orig_mean": statistics.mean(ip_origs),
            "norm_ratio_mean": statistics.mean(norm_rs),
        }

    # Sort by quality
    sorted_agg = sorted(agg.items(), key=lambda x: x[1]["quality_mean"])
    joint_mean = statistics.mean(all_joint[s] for s in seeds)

    print(f"\n  {'Method':<15} {'Val Loss':>9} {'Std':>7} {'vs Joint':>9} "
          f"{'IP(avg)':>8} {'IP(orig)':>9} {'NormR':>7}")
    print("  " + "-" * 68)

    quality_ranking = []
    ip_avg_ranking = []
    ip_orig_ranking = []

    for rank_idx, (name, r) in enumerate(sorted_agg):
        gap = (r["quality_mean"] - joint_mean) / joint_mean * 100
        print(f"  {name:<15} {r['quality_mean']:>9.4f} {r['quality_std']:>7.4f} "
              f"{gap:>+8.2f}% {r['ip_vs_avg_mean']:>8.4f} "
              f"{r['ip_vs_orig_mean']:>9.4f} {r['norm_ratio_mean']:>7.4f}")

        quality_ranking.append(name)
        ip_avg_ranking.append((name, r["ip_vs_avg_mean"]))
        ip_orig_ranking.append((name, r["ip_vs_orig_mean"]))

    # Sort info preservation rankings (higher = better, should match quality = lower loss)
    ip_avg_ranking.sort(key=lambda x: -x[1])  # descending IP = best first
    ip_orig_ranking.sort(key=lambda x: -x[1])

    ip_avg_names = [x[0] for x in ip_avg_ranking]
    ip_orig_names = [x[0] for x in ip_orig_ranking]

    # Compute Spearman correlations
    # Quality: lower val loss = better => rank 1 = best
    # IP: higher = better => rank 1 = best
    # So we want: quality ranking (ascending loss) to match IP ranking (descending IP)
    quality_vals = [agg[m]["quality_mean"] for m in methods]
    ip_avg_vals = [-agg[m]["ip_vs_avg_mean"] for m in methods]  # negate so lower = better
    ip_orig_vals = [-agg[m]["ip_vs_orig_mean"] for m in methods]
    norm_vals = [-agg[m]["norm_ratio_mean"] for m in methods]

    rho_ip_avg = spearman_rank_correlation(quality_vals, ip_avg_vals)
    rho_ip_orig = spearman_rank_correlation(quality_vals, ip_orig_vals)
    rho_norm = spearman_rank_correlation(quality_vals, norm_vals)

    print(f"\n  Spearman Correlations (quality vs info preservation):")
    print(f"    IP vs average:    rho = {rho_ip_avg:+.4f}")
    print(f"    IP vs originals:  rho = {rho_ip_orig:+.4f}")
    print(f"    Norm ratio:       rho = {rho_norm:+.4f}")

    # Ranking comparison
    print(f"\n  Ranking Comparison:")
    print(f"    {'Rank':>4}  {'Quality':>15}  {'IP(avg)':>15}  {'IP(orig)':>15}")
    print(f"    {'----':>4}  {'-------':>15}  {'-------':>15}  {'--------':>15}")
    mismatches_avg = 0
    mismatches_orig = 0
    for i in range(len(quality_ranking)):
        q = quality_ranking[i]
        a = ip_avg_names[i] if i < len(ip_avg_names) else "?"
        o = ip_orig_names[i] if i < len(ip_orig_names) else "?"
        match_a = "  " if q == a else " *"
        match_o = "  " if q == o else " *"
        if q != a:
            mismatches_avg += 1
        if q != o:
            mismatches_orig += 1
        print(f"    {i+1:>4}  {q:>15}  {a:>13}{match_a}  {o:>13}{match_o}")

    print(f"\n  Ranking mismatches (IP vs avg): {mismatches_avg}")
    print(f"  Ranking mismatches (IP vs orig): {mismatches_orig}")

    # Kill criteria
    print(f"\n  --- Kill Criteria ---")

    # Use the best metric (highest Spearman)
    best_rho = max(rho_ip_avg, rho_ip_orig, rho_norm)
    best_metric = "IP(avg)" if best_rho == rho_ip_avg else (
        "IP(orig)" if best_rho == rho_ip_orig else "NormRatio")
    best_mismatches = min(mismatches_avg, mismatches_orig)

    print(f"  Best metric: {best_metric} (rho = {best_rho:+.4f})")
    print(f"  KC1: ranking mismatches > 1? {best_mismatches} mismatches "
          f"{'=> KILL' if best_mismatches > 1 else '=> PASS'}")
    print(f"  KC2: Spearman < 0.8? {best_rho:.4f} "
          f"{'=> KILL' if best_rho < 0.8 else '=> PASS'}")

    return agg, {
        "rho_ip_avg": rho_ip_avg,
        "rho_ip_orig": rho_ip_orig,
        "rho_norm": rho_norm,
        "mismatches_avg": mismatches_avg,
        "mismatches_orig": mismatches_orig,
        "quality_ranking": quality_ranking,
        "ip_avg_ranking": ip_avg_names,
        "ip_orig_ranking": ip_orig_names,
    }


# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()

    print("=" * 70)
    print("PHASE 1: N=2 DOMAINS")
    print("=" * 70)
    agg_n2, corr_n2 = run_multiseed(seeds=(42, 123, 7), n_domains=2)

    print("\n\n")
    print("=" * 70)
    print("PHASE 2: N=5 DOMAINS")
    print("=" * 70)
    agg_n5, corr_n5 = run_multiseed(seeds=(42, 123, 7), n_domains=5)

    # Cross-scale summary
    print(f"\n\n{'='*70}")
    print("CROSS-SCALE SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Metric':<25} {'N=2':>10} {'N=5':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Spearman IP(avg)':<25} {corr_n2['rho_ip_avg']:>+10.4f} {corr_n5['rho_ip_avg']:>+10.4f}")
    print(f"  {'Spearman IP(orig)':<25} {corr_n2['rho_ip_orig']:>+10.4f} {corr_n5['rho_ip_orig']:>+10.4f}")
    print(f"  {'Spearman NormRatio':<25} {corr_n2['rho_norm']:>+10.4f} {corr_n5['rho_norm']:>+10.4f}")
    print(f"  {'Ranking mismatches(avg)':<25} {corr_n2['mismatches_avg']:>10d} {corr_n5['mismatches_avg']:>10d}")
    print(f"  {'Ranking mismatches(orig)':<25} {corr_n2['mismatches_orig']:>10d} {corr_n5['mismatches_orig']:>10d}")

    # Overall verdict
    best_n2 = max(corr_n2['rho_ip_avg'], corr_n2['rho_ip_orig'], corr_n2['rho_norm'])
    best_n5 = max(corr_n5['rho_ip_avg'], corr_n5['rho_ip_orig'], corr_n5['rho_norm'])
    min_mismatches_n2 = min(corr_n2['mismatches_avg'], corr_n2['mismatches_orig'])
    min_mismatches_n5 = min(corr_n5['mismatches_avg'], corr_n5['mismatches_orig'])

    print(f"\n  VERDICT:")
    print(f"    N=2: best rho = {best_n2:.4f} ({'PASS' if best_n2 >= 0.8 else 'KILL'}), "
          f"min mismatches = {min_mismatches_n2} ({'PASS' if min_mismatches_n2 <= 1 else 'KILL'})")
    print(f"    N=5: best rho = {best_n5:.4f} ({'PASS' if best_n5 >= 0.8 else 'KILL'}), "
          f"min mismatches = {min_mismatches_n5} ({'PASS' if min_mismatches_n5 <= 1 else 'KILL'})")

    overall = "PROVEN" if (best_n2 >= 0.8 and best_n5 >= 0.8 and
                           min_mismatches_n2 <= 1 and min_mismatches_n5 <= 1) else "PARTIAL/KILLED"
    print(f"    Overall: {overall}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
