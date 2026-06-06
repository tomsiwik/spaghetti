"""Composition experiment: SiLU vs ReLU capsule activation.

Same protocol as relu_router/test_composition.py but runs BOTH
activations side-by-side for direct comparison.

Protocol:
  1. Pretrain base model on ALL data (shared attention + embeddings)
  2. Fine-tune only MLP weights per domain (attention frozen)
  3. Compose by concatenating A and B weight matrices
  4. Evaluate: zero-shot, scalar calibration, full calibration, weight avg

Reports: val loss, effective sparsity, near-dead capsules for both.
"""

import copy
import random
import statistics

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate, ntp_loss
from .silu_capsule import SiLUCapsuleGPT, SiLUCapsulePool
from ..relu_router.relu_router import ReLURouterGPT, ReLUCapsulePool


# Shared experiment config (identical to relu_router)
BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
N_CAPSULES = 128
STEPS_PRETRAIN = 300
STEPS_FINETUNE = 200
CALIBRATION_STEPS = 100
BATCH_SIZE = 32
LR = 3e-3


def _make_silu_model(vocab_size, n_capsules=N_CAPSULES):
    model = SiLUCapsuleGPT(vocab_size=vocab_size, n_capsules=n_capsules, **BASE)
    mx.eval(model.parameters())
    return model


def _make_relu_model(vocab_size, n_capsules=N_CAPSULES):
    model = ReLURouterGPT(vocab_size=vocab_size, n_capsules=n_capsules, **BASE)
    mx.eval(model.parameters())
    return model


def _freeze_attention(model):
    model.freeze()
    for layer in model.layers:
        layer.capsule_pool.unfreeze()


def _eval_domains(model, domain_datasets, batch_size=BATCH_SIZE):
    result = {}
    for d_name in domain_datasets:
        result[d_name] = evaluate(model, domain_datasets[d_name][1], batch_size)
    result["avg"] = sum(v for k, v in result.items() if k != "avg") / len(domain_datasets)
    return result


def _get_sparsity_stats(model):
    """Get sparsity/death stats for either SiLU or ReLU model."""
    stats = model.capsule_stats()
    if "eff_sparsity" in stats:
        # SiLU model
        return {
            "sparsity": [s if s is not None else 0.0 for s in stats["eff_sparsity"]],
            "n_dead": [d if d is not None else 0 for d in stats["n_near_dead"]],
        }
    else:
        # ReLU model
        return {
            "sparsity": [s if s is not None else 0.0 for s in stats["sparsity"]],
            "n_dead": [d if d is not None else 0 for d in stats["n_dead"]],
        }


def compose_models(base_model, domain_models, make_fn, pool_cls, vocab_size):
    """Generic composition by weight concatenation."""
    n_domains = len(domain_models)
    n_cap_per = domain_models[0].layers[0].capsule_pool.n_capsules
    n_cap_total = n_cap_per * n_domains

    composed = make_fn(vocab_size, n_capsules=n_cap_total)

    base_params = dict(nn.utils.tree_flatten(base_model.parameters()))
    shared_weights = [(k, v) for k, v in base_params.items() if "capsule_pool" not in k]
    composed.load_weights(shared_weights, strict=False)

    for layer_idx in range(len(composed.layers)):
        A_parts = [dm.layers[layer_idx].capsule_pool.A.weight for dm in domain_models]
        B_parts = [dm.layers[layer_idx].capsule_pool.B.weight for dm in domain_models]

        A_composed = mx.concatenate(A_parts, axis=0)
        B_composed = mx.concatenate(B_parts, axis=1)

        comp_pool = composed.layers[layer_idx].capsule_pool
        comp_pool.A.load_weights([("weight", A_composed)])
        comp_pool.B.load_weights([("weight", B_composed)])

    mx.eval(composed.parameters())
    return composed


def weight_average_models(base_model, domain_models, make_fn, vocab_size):
    """Generic weight averaging."""
    n_domains = len(domain_models)
    n_cap = domain_models[0].layers[0].capsule_pool.n_capsules

    averaged = make_fn(vocab_size, n_capsules=n_cap)

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


def scalar_calibrate(model, train_ds_a, train_ds_b, n_capsules_per_domain,
                     activation_fn, steps=CALIBRATION_STEPS, lr=1e-2, seed=42):
    """Train per-pool scaling factors (2 scalars/layer)."""
    rng = random.Random(seed)
    n_layers = len(model.layers)

    model.freeze()

    def forward_with_scales(model, tokens, scales_flat, n_cap, act_fn):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = model.wte(tokens) + model.wpe(pos)
        x = model.norm0(x)
        for l_idx, layer in enumerate(model.layers):
            x_norm = layer.norm1(x)
            x = x + layer.attn(x_norm)
            x_norm = layer.norm2(x)
            pool = layer.capsule_pool
            h = act_fn(pool.A(x_norm))
            h_a = h[..., :n_cap] * scales_flat[2 * l_idx]
            h_b = h[..., n_cap:] * scales_flat[2 * l_idx + 1]
            h_scaled = mx.concatenate([h_a, h_b], axis=-1)
            x = x + pool.B(h_scaled)
        return model.lm_head(x)

    def loss_fn(scales_flat, model, inputs, targets, n_cap, act_fn):
        logits = forward_with_scales(model, inputs, scales_flat, n_cap, act_fn)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
        )

    scales_flat = [mx.array([1.0]) for _ in range(2 * n_layers)]

    for step in range(1, steps + 1):
        if step % 2 == 1:
            inputs, targets = train_ds_a.get_batch(BATCH_SIZE, rng)
        else:
            inputs, targets = train_ds_b.get_batch(BATCH_SIZE, rng)

        loss, grads = mx.value_and_grad(loss_fn)(
            scales_flat, model, inputs, targets, n_capsules_per_domain, activation_fn
        )
        scales_flat = [s - lr * g for s, g in zip(scales_flat, grads)]
        mx.eval(scales_flat, loss)

    # Apply learned scales to B matrix
    for l_idx in range(n_layers):
        pool = model.layers[l_idx].capsule_pool
        B_weight = pool.B.weight
        scale_a = scales_flat[2 * l_idx].item()
        scale_b = scales_flat[2 * l_idx + 1].item()
        B_a = B_weight[:, :n_capsules_per_domain] * scale_a
        B_b = B_weight[:, n_capsules_per_domain:] * scale_b
        B_new = mx.concatenate([B_a, B_b], axis=1)
        pool.B.load_weights([("weight", B_new)])

    mx.eval(model.parameters())
    model.unfreeze()
    return [s.item() for s in scales_flat]


def full_calibrate(model, joint_train, steps=CALIBRATION_STEPS, lr=3e-4, seed=42):
    """Full capsule calibration (continued training)."""
    _freeze_attention(model)
    train(model, joint_train, steps=steps, batch_size=BATCH_SIZE,
          lr=lr, seed=seed, log_every=9999)
    model.unfreeze()


def run_single_activation(activation_name, make_fn, pool_cls, activation_fn,
                          domain_datasets, joint_train, V, seed):
    """Run full composition protocol for one activation type."""
    domain_names = list(domain_datasets.keys())
    results = {}

    # 1. Joint training baseline
    print(f"  [{activation_name}] Joint training...")
    model_joint = make_fn(V, n_capsules=N_CAPSULES * 2)
    train(model_joint, joint_train, steps=STEPS_PRETRAIN + STEPS_FINETUNE,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    # Run forward for stats
    batch_inputs, _ = joint_train.get_batch(BATCH_SIZE, random.Random(seed))
    _ = model_joint(batch_inputs)
    mx.eval(model_joint.parameters())
    results[f"{activation_name}_joint"] = _eval_domains(model_joint, domain_datasets)
    results[f"{activation_name}_joint"]["_stats"] = _get_sparsity_stats(model_joint)

    # 2. Pretrain base
    print(f"  [{activation_name}] Pretrain base...")
    base = make_fn(V, n_capsules=N_CAPSULES)
    train(base, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    # 3. Fine-tune per domain
    print(f"  [{activation_name}] Fine-tune domains...")
    domain_models = []
    for d_name in domain_names:
        model_d = copy.deepcopy(base)
        _freeze_attention(model_d)
        train(model_d, domain_datasets[d_name][0], steps=STEPS_FINETUNE,
              batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
        domain_models.append(model_d)

    # 4. Zero-shot composition
    print(f"  [{activation_name}] Zero-shot composition...")
    composed = compose_models(base, domain_models, make_fn, pool_cls, V)
    _ = composed(batch_inputs)
    mx.eval(composed.parameters())
    results[f"{activation_name}_zero_shot"] = _eval_domains(composed, domain_datasets)
    results[f"{activation_name}_zero_shot"]["_stats"] = _get_sparsity_stats(composed)

    # 5. Scalar calibration
    print(f"  [{activation_name}] Scalar calibration...")
    composed_scalar = copy.deepcopy(composed)
    learned_scales = scalar_calibrate(
        composed_scalar,
        domain_datasets[domain_names[0]][0],
        domain_datasets[domain_names[1]][0],
        N_CAPSULES, activation_fn,
        steps=CALIBRATION_STEPS, lr=1e-2, seed=seed,
    )
    results[f"{activation_name}_scalar_cal"] = _eval_domains(composed_scalar, domain_datasets)

    # 6. Full capsule calibration
    print(f"  [{activation_name}] Full calibration...")
    composed_full = copy.deepcopy(composed)
    full_calibrate(composed_full, joint_train, steps=CALIBRATION_STEPS,
                   lr=LR * 0.1, seed=seed)
    results[f"{activation_name}_full_cal"] = _eval_domains(composed_full, domain_datasets)

    # 7. Weight averaging
    print(f"  [{activation_name}] Weight averaging...")
    averaged = weight_average_models(base, domain_models, make_fn, V)
    results[f"{activation_name}_weight_avg"] = _eval_domains(averaged, domain_datasets)

    return results


def run_composition_experiment(seed=42):
    """Run full composition experiment for both SiLU and ReLU."""
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

    all_docs_train, _ = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_docs_train, tokenizer, BASE["block_size"])
    V = tokenizer.vocab_size

    results = {}

    # Run ReLU
    relu_results = run_single_activation(
        "relu", _make_relu_model, ReLUCapsulePool, nn.relu,
        domain_datasets, joint_train, V, seed,
    )
    results.update(relu_results)

    # Run SiLU
    silu_results = run_single_activation(
        "silu", _make_silu_model, SiLUCapsulePool, nn.silu,
        domain_datasets, joint_train, V, seed,
    )
    results.update(silu_results)

    return results


def main():
    """Run comparison across 3 seeds."""
    seeds = [42, 123, 7]
    all_results = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  Seed {seed}")
        print(f"{'='*70}")
        r = run_composition_experiment(seed=seed)
        all_results.append(r)

        for method, vals in sorted(r.items()):
            if method.startswith("_"):
                continue
            domains = [k for k in vals if k not in ("avg", "_stats")]
            detail = " | ".join(f"{d}={vals[d]:.3f}" for d in domains)
            stats_str = ""
            if "_stats" in vals:
                s = vals["_stats"]
                mean_sp = statistics.mean(s["sparsity"])
                mean_dead = statistics.mean(s["n_dead"])
                stats_str = f" [sp={mean_sp:.3f}, dead={mean_dead:.0f}]"
            print(f"  {method:<25} avg={vals['avg']:.4f} ({detail}){stats_str}")

    # Aggregate
    print(f"\n{'='*70}")
    print("  3-Seed Aggregate: SiLU vs ReLU")
    print(f"{'='*70}")

    # Collect method names
    methods = sorted(all_results[0].keys())

    relu_joint_mean = statistics.mean([r["relu_joint"]["avg"] for r in all_results])
    silu_joint_mean = statistics.mean([r["silu_joint"]["avg"] for r in all_results])

    print(f"\n  {'Method':<25} {'avg':>8} {'std':>8} {'vs own joint':>14}")
    print("  " + "-" * 58)

    for method in methods:
        avgs = [r[method]["avg"] for r in all_results]
        mean = statistics.mean(avgs)
        std = statistics.stdev(avgs) if len(avgs) > 1 else 0

        joint_ref = relu_joint_mean if method.startswith("relu") else silu_joint_mean
        vs_joint = ((mean - joint_ref) / joint_ref) * 100
        print(f"  {method:<25} {mean:>8.4f} {std:>8.4f} {vs_joint:>+13.1f}%")

    # Head-to-head comparison
    print(f"\n  === Head-to-Head: SiLU vs ReLU ===")
    comparisons = [
        ("Joint training", "silu_joint", "relu_joint"),
        ("Zero-shot composition", "silu_zero_shot", "relu_zero_shot"),
        ("Scalar calibration", "silu_scalar_cal", "relu_scalar_cal"),
        ("Full calibration", "silu_full_cal", "relu_full_cal"),
        ("Weight averaging", "silu_weight_avg", "relu_weight_avg"),
    ]

    for label, silu_key, relu_key in comparisons:
        silu_avg = statistics.mean([r[silu_key]["avg"] for r in all_results])
        relu_avg = statistics.mean([r[relu_key]["avg"] for r in all_results])
        delta = ((silu_avg - relu_avg) / relu_avg) * 100
        winner = "SiLU" if silu_avg < relu_avg else "ReLU"
        print(f"  {label:<25} SiLU={silu_avg:.4f} ReLU={relu_avg:.4f} "
              f"delta={delta:>+.1f}% ({winner} wins)")

    # Sparsity comparison (from joint models)
    print(f"\n  === Sparsity & Death ===")
    for seed_idx, r in enumerate(all_results):
        if "_stats" in r.get("relu_joint", {}):
            relu_s = r["relu_joint"]["_stats"]
            silu_s = r["silu_joint"]["_stats"]
            relu_sp = statistics.mean(relu_s["sparsity"])
            silu_sp = statistics.mean(silu_s["sparsity"])
            relu_dead = statistics.mean(relu_s["n_dead"])
            silu_dead = statistics.mean(silu_s["n_dead"])
            print(f"  Seed {seeds[seed_idx]}: "
                  f"ReLU sp={relu_sp:.3f} dead={relu_dead:.0f} | "
                  f"SiLU eff_sp={silu_sp:.3f} near_dead={silu_dead:.0f}")

    # Kill criteria
    print(f"\n  === Kill Criteria ===")
    silu_vs_relu_joint = ((silu_joint_mean - relu_joint_mean) / relu_joint_mean) * 100
    print(f"  SiLU joint vs ReLU joint: {silu_vs_relu_joint:>+.1f}% "
          f"({'KILL (>5%)' if silu_vs_relu_joint > 5.0 else 'OK'})")

    silu_zero = statistics.mean([r["silu_zero_shot"]["avg"] for r in all_results])
    relu_zero = statistics.mean([r["relu_zero_shot"]["avg"] for r in all_results])
    silu_comp_degrad = ((silu_zero - silu_joint_mean) / silu_joint_mean) * 100
    relu_comp_degrad = ((relu_zero - relu_joint_mean) / relu_joint_mean) * 100
    comp_diff = silu_comp_degrad - relu_comp_degrad
    print(f"  SiLU composition degradation: {silu_comp_degrad:>+.1f}% "
          f"(ReLU: {relu_comp_degrad:>+.1f}%, diff: {comp_diff:>+.1f}%)")
    if comp_diff > 5.0:
        print(f"  KILL: SiLU composition >5% worse than ReLU")
    else:
        print(f"  OK: SiLU composition within 5% of ReLU")


if __name__ == "__main__":
    main()
