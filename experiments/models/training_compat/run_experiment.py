"""Exp 11: Training-time composition compatibility.

Research question: Can auxiliary losses during domain fine-tuning reduce the
composition gap between independently-composed and jointly-trained models?

Protocol:
  1. Pretrain base model on ALL data (300 steps)
  2. Snapshot base weights (for orthogonality loss reference)
  3. Fine-tune only capsule weights per domain with auxiliary losses (200 steps)
  4. Compose by concatenating A and B weight matrices from both domains
  5. Evaluate: zero-shot composition, scalar calibration, full calibration
  6. Compare gap reduction vs no-aux baseline

Conditions:
  A. no_aux     - standard fine-tuning (relu_router protocol, control)
  B. ortho_only - weight orthogonality loss only (coeff=0.1)
  C. norm_only  - output-norm matching loss only (coeff=0.1)
  D. combined   - both losses (ortho=0.1, norm=0.1)
  E. joint      - joint training baseline (upper bound)

Kill threshold: best aux condition reduces composition gap < 50% vs no_aux.
"""

import copy
import random
import statistics

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate, ntp_loss
from .training_compat import TrainingCompatGPT


# Shared config (matches relu_router experiments)
BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
N_CAPSULES = 128       # per domain (composed = 256)
STEPS_PRETRAIN = 300
STEPS_FINETUNE = 200
CALIBRATION_STEPS = 100
BATCH_SIZE = 32
LR = 3e-3


# Auxiliary loss coefficient configs
AUX_CONFIGS = {
    "no_aux":     {"ortho_coeff": 0.0, "norm_coeff": 0.0},
    "ortho_only": {"ortho_coeff": 0.1, "norm_coeff": 0.0},
    "norm_only":  {"ortho_coeff": 0.0, "norm_coeff": 0.1},
    "combined":   {"ortho_coeff": 0.1, "norm_coeff": 0.1},
}


def _make_model(vocab_size, n_capsules=N_CAPSULES,
                ortho_coeff=0.0, norm_coeff=0.0):
    """Create a TrainingCompatGPT model."""
    model = TrainingCompatGPT(
        vocab_size=vocab_size, n_capsules=n_capsules,
        ortho_coeff=ortho_coeff, norm_coeff=norm_coeff,
        **BASE,
    )
    mx.eval(model.parameters())
    return model


def _freeze_attention(model):
    """Freeze everything EXCEPT capsule pool weights."""
    model.freeze()
    for layer in model.layers:
        layer.capsule_pool.unfreeze()


def _eval_domains(model, domain_datasets, batch_size=BATCH_SIZE):
    """Evaluate model on all domains."""
    result = {}
    for d_name in domain_datasets:
        result[d_name] = evaluate(model, domain_datasets[d_name][1], batch_size)
    result["avg"] = sum(v for k, v in result.items() if k != "avg") / len(domain_datasets)
    return result


def compose_models(base_model, domain_models, vocab_size,
                   ortho_coeff=0.0, norm_coeff=0.0):
    """Compose domain-specific capsule pools by weight concatenation."""
    n_domains = len(domain_models)
    n_capsules_per_domain = domain_models[0].layers[0].capsule_pool.n_capsules
    n_capsules_total = n_capsules_per_domain * n_domains

    composed = _make_model(
        vocab_size=vocab_size,
        n_capsules=n_capsules_total,
        ortho_coeff=ortho_coeff,
        norm_coeff=norm_coeff,
    )

    # Copy shared parameters from base model
    base_params = dict(nn.utils.tree_flatten(base_model.parameters()))
    shared_weights = [(k, v) for k, v in base_params.items() if "capsule_pool" not in k]
    composed.load_weights(shared_weights, strict=False)

    for layer_idx in range(len(composed.layers)):
        A_parts = [dm.layers[layer_idx].capsule_pool.A.weight for dm in domain_models]
        B_parts = [dm.layers[layer_idx].capsule_pool.B.weight for dm in domain_models]

        A_composed = mx.concatenate(A_parts, axis=0)  # (P*D, d)
        B_composed = mx.concatenate(B_parts, axis=1)  # (d, P*D)

        comp_pool = composed.layers[layer_idx].capsule_pool
        comp_pool.A.load_weights([("weight", A_composed)])
        comp_pool.B.load_weights([("weight", B_composed)])

    mx.eval(composed.parameters())
    return composed


def weight_average_models(base_model, domain_models, vocab_size):
    """Compose by weight averaging (standard model merging baseline)."""
    n_domains = len(domain_models)
    n_capsules = domain_models[0].layers[0].capsule_pool.n_capsules

    averaged = _make_model(vocab_size=vocab_size, n_capsules=n_capsules)

    base_params = dict(nn.utils.tree_flatten(base_model.parameters()))
    shared_weights = [(k, v) for k, v in base_params.items() if "capsule_pool" not in k]
    averaged.load_weights(shared_weights, strict=False)

    for layer_idx in range(len(averaged.layers)):
        A_avg = sum(dm.layers[layer_idx].capsule_pool.A.weight for dm in domain_models) / n_domains
        B_avg = sum(dm.layers[layer_idx].capsule_pool.B.weight for dm in domain_models) / n_domains

        avg_pool = averaged.layers[layer_idx].capsule_pool
        avg_pool.A.load_weights([("weight", A_avg)])
        avg_pool.B.load_weights([("weight", B_avg)])

    mx.eval(averaged.parameters())
    return averaged


def measure_delta_orthogonality(base_model, domain_models):
    """Measure pairwise cosine similarity of weight deltas between domains."""
    n_domains = len(domain_models)
    n_layers = len(base_model.layers)

    all_sims = []
    for l_idx in range(n_layers):
        base_A = base_model.layers[l_idx].capsule_pool.A.weight
        base_B = base_model.layers[l_idx].capsule_pool.B.weight

        deltas = []
        for dm in domain_models:
            dA = dm.layers[l_idx].capsule_pool.A.weight - base_A
            dB = dm.layers[l_idx].capsule_pool.B.weight - base_B
            delta = mx.concatenate([dA.reshape(-1), dB.reshape(-1)])
            deltas.append(delta)

        for i in range(n_domains):
            for j in range(i + 1, n_domains):
                dot = mx.sum(deltas[i] * deltas[j]).item()
                ni = mx.sqrt(mx.sum(deltas[i] ** 2)).item()
                nj = mx.sqrt(mx.sum(deltas[j] ** 2)).item()
                cos = dot / (ni * nj + 1e-12)
                all_sims.append(cos)

    return {
        "mean": statistics.mean(all_sims) if all_sims else 0,
        "max": max(all_sims) if all_sims else 0,
        "all": all_sims,
    }


def measure_output_norm_variance(base_model, domain_models, dataset, batch_size=32, n_batches=5):
    """Measure variance in output norms across domain models.

    Lower variance = more compatible for composition (balanced contributions).
    """
    rng = random.Random(0)
    all_norms = {i: [] for i in range(len(domain_models))}

    for _ in range(n_batches):
        inputs, _ = dataset.get_batch(batch_size, rng)

        for d_idx, dm in enumerate(domain_models):
            B, T = inputs.shape
            pos = mx.arange(T)
            x = dm.wte(inputs) + dm.wpe(pos)
            x = dm.norm0(x)
            for layer in dm.layers:
                x_attn = x + layer.attn(layer.norm1(x))
                x_norm = layer.norm2(x_attn)
                pool_out = layer.capsule_pool(x_norm)
                out_norm = mx.sqrt(mx.mean(mx.sum(pool_out * pool_out, axis=-1)) + 1e-8).item()
                all_norms[d_idx].append(out_norm)
                x = x_attn + pool_out

    # Compute per-domain mean norms and cross-domain variance
    mean_norms = [statistics.mean(all_norms[i]) for i in range(len(domain_models))]
    norm_variance = statistics.variance(mean_norms) if len(mean_norms) > 1 else 0
    return {
        "per_domain_mean_norms": mean_norms,
        "cross_domain_variance": norm_variance,
    }


def run_experiment(seed=42):
    """Run the full training compatibility experiment for one seed."""
    print(f"\n{'='*70}")
    print(f"EXP 11: TRAINING-TIME COMPOSITION COMPATIBILITY (seed={seed})")
    print(f"{'='*70}")

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    splits = domain_split(docs)

    domain_datasets = {}
    for d_name, d_docs in splits.items():
        d_train, d_val = train_val_split(d_docs, seed=seed)
        domain_datasets[d_name] = (
            CharDataset(d_train, tokenizer, BASE["block_size"]),
            CharDataset(d_val, tokenizer, BASE["block_size"]),
        )

    all_docs_train, all_docs_val = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_docs_train, tokenizer, BASE["block_size"])
    joint_val = CharDataset(all_docs_val, tokenizer, BASE["block_size"])

    V = tokenizer.vocab_size
    domain_names = list(domain_datasets.keys())
    results = {}

    # ================================================================
    # JOINT TRAINING BASELINE (upper bound)
    # ================================================================
    print("\n--- Joint training baseline ---")
    model_joint = _make_model(V, n_capsules=N_CAPSULES * 2)
    train(model_joint, joint_train, steps=STEPS_PRETRAIN + STEPS_FINETUNE,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    results["joint"] = _eval_domains(model_joint, domain_datasets)
    print(f"  Joint: avg={results['joint']['avg']:.4f}")

    # ================================================================
    # PRETRAIN SHARED BASE (shared across all conditions)
    # ================================================================
    print("\n--- Pretraining shared base ---")
    base_model = _make_model(V, n_capsules=N_CAPSULES)
    train(base_model, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    # Measure base norm ratios for norm-matching loss
    norm_ratios = base_model.measure_norm_ratios(joint_train)
    print(f"  Base norm ratios: {['%.3f' % r for r in norm_ratios]}")

    # ================================================================
    # PER-CONDITION FINE-TUNING AND COMPOSITION
    # ================================================================
    for cond_name, aux_cfg in AUX_CONFIGS.items():
        print(f"\n--- Condition: {cond_name} (ortho={aux_cfg['ortho_coeff']}, norm={aux_cfg['norm_coeff']}) ---")

        domain_models = []
        for d_name in domain_names:
            # Create fresh copy from base
            model_d = _make_model(V, n_capsules=N_CAPSULES,
                                  ortho_coeff=aux_cfg["ortho_coeff"],
                                  norm_coeff=aux_cfg["norm_coeff"])

            # Copy base weights
            base_params_list = list(zip(
                [k for k, _ in nn.utils.tree_flatten(base_model.parameters())],
                [v for _, v in nn.utils.tree_flatten(base_model.parameters())]
            ))
            model_d.load_weights(base_params_list)
            mx.eval(model_d.parameters())

            # Set base snapshot for orthogonality loss
            model_d.snapshot_base()

            # Set norm targets
            for l_idx, layer in enumerate(model_d.layers):
                layer.capsule_pool.target_norm_ratio = norm_ratios[l_idx]

            # Freeze attention, fine-tune only capsules
            _freeze_attention(model_d)
            train(model_d, domain_datasets[d_name][0], steps=STEPS_FINETUNE,
                  batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
            model_d.unfreeze()
            domain_models.append(model_d)

        # Measure delta orthogonality
        ortho_stats = measure_delta_orthogonality(base_model, domain_models)
        print(f"  Delta ortho: mean_cos={ortho_stats['mean']:.4f}, max_cos={ortho_stats['max']:.4f}")

        # Measure output norm variance
        norm_stats = measure_output_norm_variance(
            base_model, domain_models, joint_train)
        print(f"  Norm variance: {norm_stats['cross_domain_variance']:.6f}")
        print(f"  Per-domain norms: {['%.3f' % n for n in norm_stats['per_domain_mean_norms']]}")

        # Zero-shot composition
        composed = compose_models(base_model, domain_models, V)
        results[f"{cond_name}_zero_shot"] = _eval_domains(composed, domain_datasets)
        print(f"  Zero-shot: avg={results[f'{cond_name}_zero_shot']['avg']:.4f}")

        # Weight averaging
        averaged = weight_average_models(base_model, domain_models, V)
        results[f"{cond_name}_weight_avg"] = _eval_domains(averaged, domain_datasets)
        print(f"  Weight avg: avg={results[f'{cond_name}_weight_avg']['avg']:.4f}")

        # Store diagnostics
        results[f"{cond_name}_diagnostics"] = {
            "ortho": ortho_stats,
            "norm": norm_stats,
        }

    return results


def analyze_results(all_results, seeds):
    """Analyze composition gap reduction across seeds."""
    print(f"\n{'='*70}")
    print("COMPOSITION GAP ANALYSIS")
    print(f"{'='*70}")

    # Compute gaps: gap = (composed - joint) / joint * 100
    conditions = ["no_aux", "ortho_only", "norm_only", "combined"]

    # Zero-shot gaps
    print(f"\n  {'Condition':<20} {'Composed avg':>14} {'Joint avg':>12} {'Gap (%)':>10} {'vs no_aux':>12}")
    print("  " + "-" * 72)

    gaps_zs = {}
    for cond in conditions:
        avgs = [r[f"{cond}_zero_shot"]["avg"] for r in all_results]
        joints = [r["joint"]["avg"] for r in all_results]
        mean_avg = statistics.mean(avgs)
        mean_joint = statistics.mean(joints)
        gap = (mean_avg - mean_joint) / mean_joint * 100
        gaps_zs[cond] = gap

        if cond == "no_aux":
            vs_noaux = "baseline"
        else:
            reduction = (gaps_zs["no_aux"] - gap) / abs(gaps_zs["no_aux"]) * 100 if gaps_zs["no_aux"] != 0 else 0
            vs_noaux = f"{reduction:+.1f}% red."
        print(f"  {cond:<20} {mean_avg:>14.4f} {mean_joint:>12.4f} {gap:>+9.1f}% {vs_noaux:>12}")

    # Weight-avg gaps
    print(f"\n  Weight-averaging gaps:")
    print(f"  {'Condition':<20} {'WA avg':>14} {'Joint avg':>12} {'Gap (%)':>10}")
    print("  " + "-" * 60)

    for cond in conditions:
        avgs = [r[f"{cond}_weight_avg"]["avg"] for r in all_results]
        joints = [r["joint"]["avg"] for r in all_results]
        mean_avg = statistics.mean(avgs)
        mean_joint = statistics.mean(joints)
        gap = (mean_avg - mean_joint) / mean_joint * 100
        print(f"  {cond:<20} {mean_avg:>14.4f} {mean_joint:>12.4f} {gap:>+9.1f}%")

    # Orthogonality diagnostics
    print(f"\n  Orthogonality diagnostics:")
    print(f"  {'Condition':<20} {'mean_cos':>10} {'max_cos':>10}")
    print("  " + "-" * 42)
    for cond in conditions:
        diags = [r[f"{cond}_diagnostics"]["ortho"] for r in all_results]
        mean_cos = statistics.mean([d["mean"] for d in diags])
        max_cos = statistics.mean([d["max"] for d in diags])
        print(f"  {cond:<20} {mean_cos:>10.4f} {max_cos:>10.4f}")

    # Norm diagnostics
    print(f"\n  Output norm variance (lower = more compatible):")
    print(f"  {'Condition':<20} {'norm_var':>12}")
    print("  " + "-" * 34)
    for cond in conditions:
        diags = [r[f"{cond}_diagnostics"]["norm"] for r in all_results]
        mean_var = statistics.mean([d["cross_domain_variance"] for d in diags])
        print(f"  {cond:<20} {mean_var:>12.6f}")

    # Kill criterion
    print(f"\n  KILL CRITERION: aux loss reduces composition gap < 50%?")
    print(f"  Baseline (no_aux) zero-shot gap: {gaps_zs['no_aux']:+.1f}%")
    best_reduction = 0
    best_cond = "none"
    for cond in ["ortho_only", "norm_only", "combined"]:
        if gaps_zs["no_aux"] != 0:
            reduction = (gaps_zs["no_aux"] - gaps_zs[cond]) / abs(gaps_zs["no_aux"]) * 100
        else:
            reduction = 0
        if reduction > best_reduction:
            best_reduction = reduction
            best_cond = cond
        print(f"  {cond}: gap={gaps_zs[cond]:+.1f}%, reduction={reduction:.1f}%")

    threshold = 50
    if best_reduction >= threshold:
        print(f"\n  VERDICT: PASS -- {best_cond} reduces gap by {best_reduction:.1f}% (>= {threshold}%)")
    else:
        print(f"\n  VERDICT: KILL -- best reduction is {best_reduction:.1f}% ({best_cond}), < {threshold}%")

    return gaps_zs


def main():
    """Run experiment across 3 seeds and report."""
    seeds = [42, 123, 7]
    all_results = []

    for seed in seeds:
        r = run_experiment(seed=seed)
        all_results.append(r)

        # Per-seed summary
        print(f"\n  Seed {seed} summary:")
        for method, vals in r.items():
            if isinstance(vals, dict) and "avg" in vals:
                print(f"    {method:<30} avg={vals['avg']:.4f}")

    gaps = analyze_results(all_results, seeds)
    return all_results, gaps


if __name__ == "__main__":
    main()
