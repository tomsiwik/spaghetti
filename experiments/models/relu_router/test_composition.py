"""Composition experiment for ReLU Router (revised per peer review).

Tests composition by MLP weight concatenation. The architecture is a
standard ReLU MLP; the contribution is the composition protocol.

Protocol:
  1. Pretrain base model on ALL data (shared attention + embeddings)
  2. Fine-tune only MLP weights per domain (attention frozen)
  3. Compose by concatenating A and B weight matrices from both domains
  4. Evaluate: zero-shot, per-pool scalar calibration, full calibration

Controls:
  - Joint training baseline (upper bound)
  - Weight-averaging baseline (standard model merging)
  - Capsule_moe composition under identical conditions (fair comparison)

Calibration levels (Issue 4 from peer review):
  - Zero-shot: no calibration at all
  - Scalar-only: train ONLY 1 scaling factor per pool per layer
    (tests whether loudness is the only problem)
  - Full capsule calibration: train all capsule weights (= continued training)

Kill threshold: zero-shot composition > 5% vs joint = KILLED (already known).
"""

import copy
import random
import statistics

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate, ntp_loss
from experiments.models import get_model
from .relu_router import ReLURouterGPT, ReLUCapsulePool


# Shared experiment config
BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
N_CAPSULES = 128       # per domain (composed = 256)
STEPS_PRETRAIN = 300
STEPS_FINETUNE = 200
CALIBRATION_STEPS = 100
BATCH_SIZE = 32
LR = 3e-3

# Capsule MoE config (matched total capsules)
CAP_MOE = dict(n_groups=4, n_capsules_per_group=32, top_k_groups=2)  # 128 total per domain


def _make_relu_model(vocab_size, n_capsules=N_CAPSULES):
    """Create a ReLURouterGPT model."""
    model = ReLURouterGPT(
        vocab_size=vocab_size, n_capsules=n_capsules, **BASE,
    )
    mx.eval(model.parameters())
    return model


def _freeze_attention(model):
    """Freeze everything EXCEPT capsule pool / MLP weights."""
    model.freeze()
    for layer in model.layers:
        layer.capsule_pool.unfreeze()


def _eval_domains(model, domain_datasets, batch_size=BATCH_SIZE):
    """Evaluate model on all domains, return dict with per-domain and avg loss."""
    result = {}
    for d_name in domain_datasets:
        result[d_name] = evaluate(model, domain_datasets[d_name][1], batch_size)
    result["avg"] = sum(v for k, v in result.items() if k != "avg") / len(domain_datasets)
    return result


def compose_relu_models(base_model, domain_models):
    """Compose domain-specific MLP pools by weight concatenation.

    Concatenates A matrices vertically and B matrices horizontally.
    The composed output is mathematically: Pool_A(x) + Pool_B(x).
    """
    n_domains = len(domain_models)
    n_capsules_per_domain = domain_models[0].layers[0].capsule_pool.n_capsules
    n_capsules_total = n_capsules_per_domain * n_domains

    composed = _make_relu_model(
        vocab_size=base_model.lm_head.weight.shape[0],
        n_capsules=n_capsules_total,
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


def weight_average_relu_models(base_model, domain_models):
    """Compose domain-specific MLPs by weight averaging (standard model merging).

    This is the baseline: instead of concatenation, simply average the A and B
    matrices from all domain models. The result has the SAME size as a single
    domain model (no parameter increase).
    """
    n_domains = len(domain_models)
    n_capsules = domain_models[0].layers[0].capsule_pool.n_capsules

    averaged = _make_relu_model(
        vocab_size=base_model.lm_head.weight.shape[0],
        n_capsules=n_capsules,
    )

    # Copy shared parameters from base model
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
                     steps=CALIBRATION_STEPS, lr=1e-2, seed=42):
    """Fair calibration: train ONLY per-pool scaling factors.

    For a 2-domain composed model, this trains 2 scalars per layer
    (one for each domain's pool). All capsule weights are frozen.
    Total trainable parameters: 2 * n_layer = 8 at micro scale.

    This tests whether the "loudness problem" is truly the only issue
    with zero-shot composition.
    """
    rng = random.Random(seed)
    n_layers = len(model.layers)

    # Create per-pool scalars as plain arrays (one per domain per layer)
    pool_scales = [[mx.array([1.0]), mx.array([1.0])] for _ in range(n_layers)]

    # Freeze the entire model
    model.freeze()

    # We need a custom forward that applies per-pool scaling
    def forward_with_scales(model, tokens, pool_scales, n_cap):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = model.wte(tokens) + model.wpe(pos)
        x = model.norm0(x)
        for l_idx, layer in enumerate(model.layers):
            # Attention
            x_norm = layer.norm1(x)
            x = x + layer.attn(x_norm)
            # Capsule pool with per-pool scaling
            x_norm = layer.norm2(x)
            pool = layer.capsule_pool
            h = nn.relu(pool.A(x_norm))  # (B, T, 2*n_cap)
            # Split activations into two pools and scale
            h_a = h[..., :n_cap] * pool_scales[l_idx][0]
            h_b = h[..., n_cap:] * pool_scales[l_idx][1]
            h_scaled = mx.concatenate([h_a, h_b], axis=-1)
            x = x + pool.B(h_scaled)
        return model.lm_head(x)

    def loss_fn(scales_flat, model, inputs, targets, n_cap, n_layers):
        # Unflatten scales
        ps = [[scales_flat[2*l], scales_flat[2*l + 1]] for l in range(n_layers)]
        logits = forward_with_scales(model, inputs, ps, n_cap)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
        )

    # Flatten scales for gradient computation
    # Using plain SGD for 8 scalar parameters — the loss surface is well-behaved
    # and SGD at lr=1e-2 converges reliably across seeds.
    scales_flat = [mx.array([1.0]) for _ in range(2 * n_layers)]

    for step in range(1, steps + 1):
        # Alternate domain batches
        if step % 2 == 1:
            inputs, targets = train_ds_a.get_batch(BATCH_SIZE, rng)
        else:
            inputs, targets = train_ds_b.get_batch(BATCH_SIZE, rng)

        loss, grads = mx.value_and_grad(loss_fn)(
            scales_flat, model, inputs, targets, n_capsules_per_domain, n_layers
        )
        scales_flat = [s - lr * g for s, g in zip(scales_flat, grads)]
        mx.eval(scales_flat, loss)

    # Apply learned scales by modifying B matrix columns
    for l_idx in range(n_layers):
        pool = model.layers[l_idx].capsule_pool
        B_weight = pool.B.weight  # (d, 2*n_cap)
        scale_a = scales_flat[2 * l_idx].item()
        scale_b = scales_flat[2 * l_idx + 1].item()
        B_a = B_weight[:, :n_capsules_per_domain] * scale_a
        B_b = B_weight[:, n_capsules_per_domain:] * scale_b
        B_new = mx.concatenate([B_a, B_b], axis=1)
        pool.B.load_weights([("weight", B_new)])

    mx.eval(model.parameters())
    model.unfreeze()
    return [s.item() for s in scales_flat]


def full_capsule_calibrate(model, joint_train, steps=CALIBRATION_STEPS, lr=3e-4, seed=42):
    """Full calibration: train ALL capsule weights on joint data.

    NOTE: This is effectively continued joint training, not just
    "harmonizing scales." The reviewer correctly identified this as
    unfair. We include it for completeness but the scalar calibration
    is the honest test.
    """
    _freeze_attention(model)
    train(model, joint_train, steps=steps, batch_size=BATCH_SIZE,
          lr=lr, seed=seed, log_every=9999)
    model.unfreeze()


# === Capsule MoE composition (identical conditions) ===

def compose_capsule_moe(base_model, domain_models, vocab_size):
    """Compose capsule_moe domain models by concatenating groups.

    Identical protocol to relu_router but for capsule_moe architecture.
    """
    n_groups_per_domain = base_model.layers[0].capsule_pool.n_groups
    n_groups_total = n_groups_per_domain * len(domain_models)
    top_k_composed = base_model.layers[0].capsule_pool.top_k_groups * len(domain_models)

    composed = get_model("capsule_moe",
                         vocab_size=vocab_size,
                         n_groups=n_groups_total,
                         n_capsules_per_group=CAP_MOE["n_capsules_per_group"],
                         top_k_groups=top_k_composed,
                         **{k: BASE[k] for k in ["n_embd", "n_head", "n_layer", "block_size"]})

    # Copy shared parameters from base
    base_params = dict(nn.utils.tree_flatten(base_model.parameters()))
    shared_weights = [(k, v) for k, v in base_params.items()
                      if "capsule_pool" not in k]
    composed.load_weights(shared_weights, strict=False)

    # Copy domain-specific capsule groups
    for l_idx in range(len(composed.layers)):
        pool_c = composed.layers[l_idx].capsule_pool
        for d_idx, dm in enumerate(domain_models):
            pool_d = dm.layers[l_idx].capsule_pool
            offset = d_idx * n_groups_per_domain
            for g in range(n_groups_per_domain):
                pool_c.groups[offset + g].A.weight = pool_d.groups[g].A.weight
                pool_c.groups[offset + g].B.weight = pool_d.groups[g].B.weight

    mx.eval(composed.parameters())
    return composed


def calibrate_capsule_moe_router(model, train_ds_a, train_ds_b,
                                  steps=CALIBRATION_STEPS, lr=LR, seed=42):
    """Calibrate ONLY the router weights of a composed capsule_moe."""
    # Freeze everything
    model.freeze()
    # Unfreeze only router
    for layer in model.layers:
        layer.capsule_pool.router.unfreeze()

    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    for step in range(1, steps + 1):
        if step % 2 == 1:
            inputs, targets = train_ds_a.get_batch(BATCH_SIZE, rng)
        else:
            inputs, targets = train_ds_b.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    model.unfreeze()


# === Main experiment ===

def run_composition_experiment(seed=42):
    """Run the full composition experiment with all controls.

    Returns dict of {method_name: {domain: loss, "avg": avg_loss}}.
    """
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

    V = tokenizer.vocab_size
    results = {}
    domain_names = list(domain_datasets.keys())

    # ============================================================
    # 1. RELU ROUTER: Joint training baseline
    # ============================================================
    print("  [1/8] ReLU Router: joint training baseline...")
    model_joint = _make_relu_model(V, n_capsules=N_CAPSULES * 2)
    train(model_joint, joint_train, steps=STEPS_PRETRAIN + STEPS_FINETUNE,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    results["relu_joint"] = _eval_domains(model_joint, domain_datasets)

    # ============================================================
    # 2. RELU ROUTER: Pretrain base + domain fine-tune
    # ============================================================
    print("  [2/8] ReLU Router: pretrain base...")
    base_relu = _make_relu_model(V, n_capsules=N_CAPSULES)
    train(base_relu, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    print("        Fine-tuning per domain...")
    relu_domain_models = []
    for d_name in domain_names:
        model_d = copy.deepcopy(base_relu)
        _freeze_attention(model_d)
        train(model_d, domain_datasets[d_name][0], steps=STEPS_FINETUNE,
              batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
        relu_domain_models.append(model_d)

    # ============================================================
    # 3. RELU ROUTER: Zero-shot composition
    # ============================================================
    print("  [3/8] ReLU Router: zero-shot composition...")
    composed_relu = compose_relu_models(base_relu, relu_domain_models)
    results["relu_zero_shot"] = _eval_domains(composed_relu, domain_datasets)

    # ============================================================
    # 4. RELU ROUTER: Scalar-only calibration (FAIR test, Issue 4)
    # ============================================================
    print("  [4/8] ReLU Router: scalar-only calibration (2 params/layer)...")
    composed_scalar = copy.deepcopy(composed_relu)
    learned_scales = scalar_calibrate(
        composed_scalar,
        domain_datasets[domain_names[0]][0],
        domain_datasets[domain_names[1]][0],
        N_CAPSULES, steps=CALIBRATION_STEPS, lr=1e-2, seed=seed,
    )
    results["relu_scalar_cal"] = _eval_domains(composed_scalar, domain_datasets)
    print(f"        Learned scales: {['%.3f' % s for s in learned_scales]}")

    # ============================================================
    # 5. RELU ROUTER: Full capsule calibration (unfair, for comparison)
    # ============================================================
    print("  [5/8] ReLU Router: full capsule calibration (continued training)...")
    composed_full = copy.deepcopy(composed_relu)
    full_capsule_calibrate(composed_full, joint_train, steps=CALIBRATION_STEPS,
                           lr=LR * 0.1, seed=seed)
    results["relu_full_cal"] = _eval_domains(composed_full, domain_datasets)

    # ============================================================
    # 6. RELU ROUTER: Weight averaging baseline
    # ============================================================
    print("  [6/8] ReLU Router: weight averaging baseline...")
    averaged_relu = weight_average_relu_models(base_relu, relu_domain_models)
    results["relu_weight_avg"] = _eval_domains(averaged_relu, domain_datasets)

    # ============================================================
    # 7. CAPSULE MOE: Same protocol, for fair comparison
    # ============================================================
    print("  [7/8] Capsule MoE: joint training baseline...")
    cap_joint = get_model("capsule_moe", vocab_size=V,
                          n_groups=CAP_MOE["n_groups"] * 2,
                          n_capsules_per_group=CAP_MOE["n_capsules_per_group"],
                          top_k_groups=CAP_MOE["top_k_groups"] * 2,
                          **{k: BASE[k] for k in ["n_embd", "n_head", "n_layer", "block_size"]})
    mx.eval(cap_joint.parameters())
    train(cap_joint, joint_train, steps=STEPS_PRETRAIN + STEPS_FINETUNE,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    results["cap_joint"] = _eval_domains(cap_joint, domain_datasets)

    print("        Pretrain + fine-tune...")
    base_cap = get_model("capsule_moe", vocab_size=V, **CAP_MOE,
                         **{k: BASE[k] for k in ["n_embd", "n_head", "n_layer", "block_size"]})
    mx.eval(base_cap.parameters())
    train(base_cap, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    cap_domain_models = []
    for d_name in domain_names:
        model_d = copy.deepcopy(base_cap)
        # Freeze everything except capsule groups
        model_d.freeze()
        for layer in model_d.layers:
            for group in layer.capsule_pool.groups:
                group.unfreeze()
        train(model_d, domain_datasets[d_name][0], steps=STEPS_FINETUNE,
              batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
        model_d.unfreeze()
        cap_domain_models.append(model_d)

    # ============================================================
    # 8. CAPSULE MOE: Composition with router calibration
    # ============================================================
    print("  [8/8] Capsule MoE: composition + router calibration...")
    composed_cap = compose_capsule_moe(base_cap, cap_domain_models, V)

    # Zero-shot (uniform routing)
    for layer in composed_cap.layers:
        layer.capsule_pool.uniform_routing = True
    mx.eval(composed_cap.parameters())
    results["cap_zero_shot"] = _eval_domains(composed_cap, domain_datasets)

    # Calibrated (train router only)
    composed_cap_cal = compose_capsule_moe(base_cap, cap_domain_models, V)
    mx.eval(composed_cap_cal.parameters())
    calibrate_capsule_moe_router(
        composed_cap_cal,
        domain_datasets[domain_names[0]][0],
        domain_datasets[domain_names[1]][0],
        steps=CALIBRATION_STEPS, lr=LR, seed=seed,
    )
    results["cap_calibrated"] = _eval_domains(composed_cap_cal, domain_datasets)

    return results


def main():
    """Run composition experiment across 3 seeds and report."""
    seeds = [42, 123, 7]
    all_results = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  Seed {seed}")
        print(f"{'='*70}")
        r = run_composition_experiment(seed=seed)
        all_results.append(r)

        # Print per-seed summary
        for method, vals in r.items():
            domains = [k for k in vals if k != "avg"]
            detail = " | ".join(f"{d}={vals[d]:.3f}" for d in domains)
            print(f"  {method:<25} avg={vals['avg']:.4f} ({detail})")

    # Aggregate across seeds
    print(f"\n{'='*70}")
    print("  3-Seed Aggregate")
    print(f"{'='*70}")

    methods = list(all_results[0].keys())
    relu_joint_mean = statistics.mean([r["relu_joint"]["avg"] for r in all_results])
    cap_joint_mean = statistics.mean([r["cap_joint"]["avg"] for r in all_results])

    print(f"\n  {'Method':<25} {'avg':>8} {'std':>8} {'vs relu_joint':>14} {'vs cap_joint':>14}")
    print("  " + "-" * 72)
    for method in methods:
        avgs = [r[method]["avg"] for r in all_results]
        mean = statistics.mean(avgs)
        std = statistics.stdev(avgs) if len(avgs) > 1 else 0
        vs_relu = ((mean - relu_joint_mean) / relu_joint_mean) * 100
        vs_cap = ((mean - cap_joint_mean) / cap_joint_mean) * 100
        print(f"  {method:<25} {mean:>8.4f} {std:>8.4f} {vs_relu:>+13.1f}% {vs_cap:>+13.1f}%")

    # Kill threshold analysis
    print(f"\n  Kill threshold analysis (> 5% vs own joint baseline = KILLED):")
    print("  " + "-" * 60)
    relu_methods = ["relu_zero_shot", "relu_scalar_cal", "relu_full_cal", "relu_weight_avg"]
    cap_methods = ["cap_zero_shot", "cap_calibrated"]

    for method in relu_methods:
        avgs = [r[method]["avg"] for r in all_results]
        mean = statistics.mean(avgs)
        delta = ((mean - relu_joint_mean) / relu_joint_mean) * 100
        status = "KILLED" if delta > 5.0 else "OK"
        print(f"  {status:>6} {method:<25} {delta:>+.1f}% vs relu_joint")

    for method in cap_methods:
        avgs = [r[method]["avg"] for r in all_results]
        mean = statistics.mean(avgs)
        delta = ((mean - cap_joint_mean) / cap_joint_mean) * 100
        status = "KILLED" if delta > 5.0 else "OK"
        print(f"  {status:>6} {method:<25} {delta:>+.1f}% vs cap_joint")


if __name__ == "__main__":
    main()
