"""Experiment: LSH Capsule Routing vs Softmax Capsule MoE (revised with controls).

Compares:
1. Single-domain quality: LSH (T=1,2,4,8) vs softmax (with and without balance loss)
2. Uniform routing baseline (establishes floor)
3. Routing throughput: time per forward pass
4. Expert utilization balance

Controls added per adversarial review:
- softmax_no_balance: CapsuleMoE with aux_loss=0 (no balance loss handicap)
- uniform: CapsuleMoEUniform (no routing at all, 1/G weighting)

Kill criteria:
- LSH routing quality >3% worse than learned softmax routing
- LSH requires >4 hash tables to match softmax quality (diminishing returns)

We match total capsule count: 8 groups * 32 caps/group = 256 total for all models.
"""

import sys
import time
import random
import json
import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import ntp_loss, evaluate
from experiments.models import get_model


def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def paired_ttest(a, b):
    """Paired t-test for two lists of values. Returns t-statistic and p-value."""
    n = len(a)
    assert n == len(b) and n >= 2
    diffs = [ai - bi for ai, bi in zip(a, b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    se_d = (var_d / n) ** 0.5
    if se_d < 1e-12:
        return 0.0, 1.0
    t_stat = mean_d / se_d
    # Approximate two-tailed p-value using t-distribution with n-1 df
    # For n=3, df=2. Use a simple approximation.
    import math
    df = n - 1
    # Regularized incomplete beta function approximation for small df
    # For df=2: p = (1 + t^2/2)^(-3/2) ... use a numerical approach
    x = df / (df + t_stat ** 2)
    # Simple numerical integration isn't needed for reporting;
    # use the well-known formula for df=2: p = 1/sqrt(1 + t^2/df)^(df+1)/df
    # Actually, for df=2: CDF(t) = 0.5 + t/(2*sqrt(2+t^2))
    if df == 2:
        p_one_tail = 0.5 - t_stat / (2.0 * math.sqrt(2.0 + t_stat ** 2))
        p_two_tail = 2.0 * min(p_one_tail, 1.0 - p_one_tail)
    else:
        # Fallback: approximate using normal for larger df
        from math import erf
        z = t_stat * (1.0 - 1.0 / (4.0 * df))  # rough correction
        p_two_tail = 1.0 - erf(abs(z) / math.sqrt(2.0))
    return t_stat, max(0.0, min(1.0, p_two_tail))


def train_model(model, train_ds, val_ds, steps=500, batch_size=32, lr=3e-3,
                seed=42, log_every=100):
    """Train a model and return train/val loss."""
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    losses = []
    t0 = time.time()

    for step in range(1, steps + 1):
        inputs, targets = train_ds.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss.item())

        if step % log_every == 0 or step == steps:
            elapsed = time.time() - t0
            print(f"    step {step:4d}/{steps} | loss {loss.item():.4f} | {elapsed:.1f}s")

    val_loss = evaluate(model, val_ds, batch_size)
    elapsed = time.time() - t0
    return {"losses": losses, "val_loss": val_loss, "elapsed_s": elapsed}


def measure_throughput(model, batch_size=32, block_size=32, n_iters=50):
    """Measure forward-pass throughput (tokens/sec)."""
    tokens = mx.zeros((batch_size, block_size), dtype=mx.int32)
    # Warmup
    for _ in range(5):
        _ = model(tokens)
        mx.eval(model.parameters())

    t0 = time.time()
    for _ in range(n_iters):
        _ = model(tokens)
        mx.eval(model.parameters())
    elapsed = time.time() - t0
    tokens_per_sec = (n_iters * batch_size * block_size) / elapsed
    return tokens_per_sec


def zero_aux_loss(self):
    """Replacement aux_loss that returns 0."""
    return mx.array(0.0)


def run_single_domain_sweep(seeds=(42, 123, 777), steps=500):
    """Compare single-domain quality: LSH (T=1,2,4,8) vs softmax controls."""
    print("=" * 70)
    print("EXPERIMENT 1: Single-Domain Quality -- LSH vs Softmax (with controls)")
    print("=" * 70)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vs = tokenizer.vocab_size

    # Config definitions:
    # - softmax_k2: standard softmax with balance loss (original baseline)
    # - softmax_no_bal: softmax with aux_loss zeroed (fair comparison)
    # - uniform: all groups weighted 1/G (floor baseline)
    # - lsh_T{1,2,4,8}: LSH routing with varying table counts
    configs = {
        "softmax_k2": {
            "model": "capsule_moe",
            "kwargs": {"n_groups": 8, "n_capsules_per_group": 32, "top_k_groups": 2},
            "zero_aux": False,
        },
        "softmax_no_bal": {
            "model": "capsule_moe",
            "kwargs": {"n_groups": 8, "n_capsules_per_group": 32, "top_k_groups": 2},
            "zero_aux": True,
        },
        "uniform": {
            "model": "capsule_moe_uniform",
            "kwargs": {"n_groups": 8, "n_capsules_per_group": 32, "top_k_groups": 2},
            "zero_aux": True,  # uniform has no meaningful aux_loss anyway
        },
        "lsh_T1": {
            "model": "lsh_capsule_routing",
            "kwargs": {"n_groups": 8, "n_capsules_per_group": 32, "n_tables": 1, "top_k": 2},
            "zero_aux": True,  # LSH routing is not learned, aux_loss is diagnostic only
        },
        "lsh_T2": {
            "model": "lsh_capsule_routing",
            "kwargs": {"n_groups": 8, "n_capsules_per_group": 32, "n_tables": 2, "top_k": 2},
            "zero_aux": True,
        },
        "lsh_T4": {
            "model": "lsh_capsule_routing",
            "kwargs": {"n_groups": 8, "n_capsules_per_group": 32, "n_tables": 4, "top_k": 2},
            "zero_aux": True,
        },
        "lsh_T8": {
            "model": "lsh_capsule_routing",
            "kwargs": {"n_groups": 8, "n_capsules_per_group": 32, "n_tables": 8, "top_k": 2},
            "zero_aux": True,
        },
    }

    results = {name: [] for name in configs}
    param_counts = {}
    throughputs = {}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        docs_train, docs_val = train_val_split(docs, seed=seed)
        train_ds = CharDataset(docs_train, tokenizer, 32)
        val_ds = CharDataset(docs_val, tokenizer, 32)

        for name, cfg in configs.items():
            print(f"\n  [{name}]")
            mx.random.seed(seed)
            model = get_model(cfg["model"], vocab_size=vs, block_size=32,
                              **cfg["kwargs"])

            # Optionally zero out aux_loss to remove balance-loss confound
            if cfg.get("zero_aux", False):
                import types
                model.aux_loss = types.MethodType(zero_aux_loss, model)

            mx.eval(model.parameters())
            n_params = count_params(model)
            param_counts[name] = n_params
            print(f"    trainable params: {n_params:,}")

            result = train_model(model, train_ds, val_ds, steps=steps, seed=seed)
            results[name].append(result["val_loss"])
            print(f"    val_loss: {result['val_loss']:.4f}")

            # Measure throughput (only on first seed)
            if seed == seeds[0]:
                tps = measure_throughput(model)
                throughputs[name] = tps
                print(f"    throughput: {tps:.0f} tok/s")

    # Summary
    print(f"\n{'='*70}")
    print(f"SINGLE-DOMAIN SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*70}")

    # Use softmax_no_bal as the primary comparison baseline (fair comparison)
    softmax_nb_mean = sum(results["softmax_no_bal"]) / len(seeds)
    softmax_bal_mean = sum(results["softmax_k2"]) / len(seeds)

    print(f"\n  {'Config':<18} | {'Params':>8} | {'Val Loss':>9} | {'vs SM-noBal':>11} | {'vs SM-bal':>9} | {'Throughput':>12}")
    print(f"  {'-'*18}-+-{'-'*8}-+-{'-'*9}-+-{'-'*11}-+-{'-'*9}-+-{'-'*12}")

    kill_results = {}
    for name in configs:
        mean_val = sum(results[name]) / len(seeds)
        delta_nb_pct = 100 * (mean_val - softmax_nb_mean) / softmax_nb_mean
        delta_bal_pct = 100 * (mean_val - softmax_bal_mean) / softmax_bal_mean
        tps_str = f"{throughputs.get(name, 0):.0f} tok/s" if name in throughputs else "N/A"
        print(f"  {name:<18} | {param_counts[name]:>8,} | {mean_val:>9.4f} | {delta_nb_pct:>+10.2f}% | {delta_bal_pct:>+8.2f}% | {tps_str:>12}")
        kill_results[name] = {"mean": mean_val,
                               "delta_vs_nobal_pct": delta_nb_pct,
                               "delta_vs_bal_pct": delta_bal_pct,
                               "per_seed": results[name]}

    # Paired t-tests: each LSH config vs softmax_no_bal
    print(f"\n  --- Paired t-tests (vs softmax_no_bal, {len(seeds)} seeds) ---")
    for name in configs:
        if name.startswith("lsh_") or name in ("uniform", "softmax_k2"):
            t_stat, p_val = paired_ttest(results[name], results["softmax_no_bal"])
            sig = "*" if p_val < 0.05 else ""
            print(f"    {name:<18}: t={t_stat:+.3f}, p={p_val:.4f} {sig}")

    # Kill criteria evaluation (vs softmax_no_bal as fair baseline)
    print(f"\n  --- Kill Criteria (vs softmax_no_bal, the fair baseline) ---")

    best_lsh_delta = min(kill_results[n]["delta_vs_nobal_pct"] for n in kill_results if n.startswith("lsh_"))
    best_lsh_name = min((n for n in kill_results if n.startswith("lsh_")),
                        key=lambda n: kill_results[n]["delta_vs_nobal_pct"])
    kc1_triggered = best_lsh_delta > 3.0
    print(f"  KC1 (LSH >3% worse than softmax-no-bal): {'TRIGGERED' if kc1_triggered else 'PASSES'}")
    print(f"       Best LSH: {best_lsh_name} at {best_lsh_delta:+.2f}% vs softmax_no_bal")

    tables_needed = None
    for n_tables in [1, 2, 4, 8]:
        name = f"lsh_T{n_tables}"
        if kill_results[name]["delta_vs_nobal_pct"] <= 3.0:
            tables_needed = n_tables
            break
    kc2_triggered = tables_needed is not None and tables_needed > 4
    if tables_needed is None:
        kc2_triggered = True
        print(f"  KC2 (requires >4 tables): TRIGGERED (never matches at any T)")
    else:
        print(f"  KC2 (requires >4 tables): {'TRIGGERED' if kc2_triggered else 'PASSES'}")
        print(f"       First match at T={tables_needed}")

    # Check if uniform matches everything (routing irrelevant at this scale?)
    uniform_delta = kill_results["uniform"]["delta_vs_nobal_pct"]
    print(f"\n  Uniform routing delta vs softmax_no_bal: {uniform_delta:+.2f}%")
    if abs(uniform_delta) < 3.0:
        print(f"  WARNING: Uniform routing within 3% of softmax -- routing quality may be irrelevant at this scale.")
    else:
        print(f"  Uniform routing is {abs(uniform_delta):.1f}% from softmax -- routing quality matters.")

    return results, kill_results, throughputs


def run_expert_utilization_analysis(seeds=(42,)):
    """Analyze expert utilization balance for LSH vs softmax (both variants)."""
    print(f"\n{'='*70}")
    print("EXPERIMENT 2: Expert Utilization Analysis")
    print(f"{'='*70}")

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vs = tokenizer.vocab_size

    for seed in seeds:
        docs_train, docs_val = train_val_split(docs, seed=seed)
        train_ds = CharDataset(docs_train, tokenizer, 32)
        val_ds = CharDataset(docs_val, tokenizer, 32)

        # Train LSH T=4 model
        print(f"\n  Training LSH T=4 for utilization analysis (seed={seed})...")
        mx.random.seed(seed)
        import types
        lsh_model = get_model("lsh_capsule_routing", vocab_size=vs, block_size=32,
                               n_groups=8, n_capsules_per_group=32,
                               n_tables=4, top_k=2)
        lsh_model.aux_loss = types.MethodType(zero_aux_loss, lsh_model)
        mx.eval(lsh_model.parameters())
        _ = train_model(lsh_model, train_ds, val_ds, steps=500, seed=seed)

        # Run a batch through for diagnostics
        rng = random.Random(seed)
        inputs, _ = val_ds.get_batch(64, rng)
        _ = lsh_model(inputs)
        mx.eval(lsh_model.parameters())

        diag = lsh_model.get_routing_diagnostics()
        print(f"\n  LSH Routing Diagnostics (T=4, seed={seed}):")
        for layer_name, layer_diag in diag.items():
            if "expert_utilization" in layer_diag:
                util = layer_diag["expert_utilization"]
                util_std = (sum((u - 1/8)**2 for u in util) / len(util)) ** 0.5
                print(f"    {layer_name}:")
                print(f"      utilization: {['%.3f' % u for u in util]}")
                print(f"      std from uniform: {util_std:.4f}")
                print(f"      entropy: {layer_diag['normalized_entropy']:.3f}")
                print(f"      mean selected: {layer_diag['mean_selected']:.1f}")

        # Compare with softmax baseline (WITH balance loss) utilization
        print(f"\n  Training softmax (with balance loss) for utilization comparison...")
        mx.random.seed(seed)
        flat_model_bal = get_model("capsule_moe", vocab_size=vs, block_size=32,
                                n_groups=8, n_capsules_per_group=32, top_k_groups=2)
        mx.eval(flat_model_bal.parameters())
        _ = train_model(flat_model_bal, train_ds, val_ds, steps=500, seed=seed)

        _ = flat_model_bal(inputs)
        mx.eval(flat_model_bal.parameters())

        print(f"\n  Softmax (with balance loss) Routing Diagnostics (seed={seed}):")
        for li, layer in enumerate(flat_model_bal.layers):
            gp = layer.capsule_pool._gate_probs
            if gp is not None:
                selected = (gp > 1e-6).astype(mx.float32)
                util = mx.mean(selected, axis=(0, 1))
                mx.eval(util)
                eps = 1e-8
                entropy = -mx.sum(gp * mx.log(gp + eps), axis=-1)
                max_ent = math.log(8)
                norm_ent = mx.mean(entropy).item() / max_ent
                print(f"    layer_{li}:")
                print(f"      utilization: {['%.3f' % u for u in util.tolist()]}")
                print(f"      entropy: {norm_ent:.3f}")

        # Compare with softmax baseline (WITHOUT balance loss) utilization
        print(f"\n  Training softmax (no balance loss) for utilization comparison...")
        mx.random.seed(seed)
        flat_model_nb = get_model("capsule_moe", vocab_size=vs, block_size=32,
                                n_groups=8, n_capsules_per_group=32, top_k_groups=2)
        flat_model_nb.aux_loss = types.MethodType(zero_aux_loss, flat_model_nb)
        mx.eval(flat_model_nb.parameters())
        _ = train_model(flat_model_nb, train_ds, val_ds, steps=500, seed=seed)

        _ = flat_model_nb(inputs)
        mx.eval(flat_model_nb.parameters())

        print(f"\n  Softmax (no balance loss) Routing Diagnostics (seed={seed}):")
        for li, layer in enumerate(flat_model_nb.layers):
            gp = layer.capsule_pool._gate_probs
            if gp is not None:
                selected = (gp > 1e-6).astype(mx.float32)
                util = mx.mean(selected, axis=(0, 1))
                mx.eval(util)
                eps = 1e-8
                entropy = -mx.sum(gp * mx.log(gp + eps), axis=-1)
                max_ent = math.log(8)
                norm_ent = mx.mean(entropy).item() / max_ent
                print(f"    layer_{li}:")
                print(f"      utilization: {['%.3f' % u for u in util.tolist()]}")
                print(f"      entropy: {norm_ent:.3f}")


def main():
    t0 = time.time()

    # Experiment 1: Quality sweep with controls
    results, kill_results, throughputs = run_single_domain_sweep(
        seeds=(42, 123, 777), steps=500
    )

    # Experiment 2: Utilization analysis (all 3 variants)
    run_expert_utilization_analysis(seeds=(42,))

    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")

    # Save results
    save_results = {
        "kill_results": {k: {"mean": v["mean"],
                              "delta_vs_nobal_pct": v["delta_vs_nobal_pct"],
                              "delta_vs_bal_pct": v["delta_vs_bal_pct"],
                              "per_seed": v["per_seed"]}
                          for k, v in kill_results.items()},
        "throughputs": throughputs,
        "total_time_s": total_time,
    }
    with open("/Users/tom/Code/tomsiwik/llm/experiments/models/lsh_capsule_routing/results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print("Results saved to results.json")


if __name__ == "__main__":
    main()
