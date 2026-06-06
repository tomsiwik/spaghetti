"""Composition experiment for Capsule MoE.

Tests the core composability claim: can capsule groups trained independently
on different domains be concatenated at runtime to handle both domains?

3-phase experiment:
  Phase 1: Train Model_A on domain A (a-m names) -- learns groups for domain A
  Phase 2: Train Model_B on domain B (n-z names) -- learns groups for domain B
  Phase 3: Compose: extract capsule groups from A and B, concatenate into
           a single model with 2*G groups, train only the router on mixed data

Baselines:
  - Joint training: single model trained on both domains simultaneously
  - Sequential training: single model trained on A then B (forgetting baseline)
  - Composed (uniform): composed model with uniform routing (no router training)
  - Composed (calibrated): composed model with router trained on mixed data

Kill threshold: if composition degrades > 5% vs joint training, the
composability claim fails.
"""

import copy
import random
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate, ntp_loss


# Standard micro config
BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
CAP = dict(**BASE, n_groups=4, n_capsules_per_group=64, top_k_groups=2)
STEPS_PER_DOMAIN = 300
ROUTER_CALIBRATION_STEPS = 100
BATCH_SIZE = 32
LR = 3e-3


def build_domain_datasets(tokenizer, docs, block_size=32, seed=42):
    """Build train/val datasets for each domain."""
    splits = domain_split(docs)
    datasets = {}
    for name, ddocs in splits.items():
        dtrain, dval = train_val_split(ddocs, seed=seed)
        datasets[name] = (
            CharDataset(dtrain, tokenizer, block_size),
            CharDataset(dval, tokenizer, block_size),
        )
    return datasets


def compose_capsule_models(model_a, model_b, vocab_size, block_size,
                           attn_strategy="avg"):
    """Create a composed model by concatenating capsule groups from A and B.

    The composed model has 2*G groups (G from A + G from B) with top_k_groups
    doubled to maintain the same active fraction.

    attn_strategy:
      "a" -- use attention/embeddings from model_a only
      "b" -- use attention/embeddings from model_b only
      "avg" -- average attention/embedding weights from both models
    """
    n_groups_a = model_a.layers[0].capsule_pool.n_groups
    n_groups_b = model_b.layers[0].capsule_pool.n_groups
    composed_groups = n_groups_a + n_groups_b
    # Double top_k to maintain same active fraction: k/G stays constant
    composed_top_k = model_a.layers[0].capsule_pool.top_k_groups + \
                     model_b.layers[0].capsule_pool.top_k_groups

    composed = get_model("capsule_moe",
                         vocab_size=vocab_size, block_size=block_size,
                         n_groups=composed_groups,
                         n_capsules_per_group=CAP["n_capsules_per_group"],
                         top_k_groups=composed_top_k,
                         **{k: BASE[k] for k in ["n_embd", "n_head", "n_layer"]})

    # Helper: merge weights based on strategy
    def merge(wa, wb):
        if attn_strategy == "a":
            return wa
        elif attn_strategy == "b":
            return wb
        else:  # avg
            return (wa + wb) / 2.0

    # Copy shared parameters (embeddings, attention, lm_head)
    # Note: RMSNorm has no learnable parameters in this implementation
    composed.wte.weight = merge(model_a.wte.weight, model_b.wte.weight)
    composed.wpe.weight = merge(model_a.wpe.weight, model_b.wpe.weight)
    composed.lm_head.weight = merge(model_a.lm_head.weight, model_b.lm_head.weight)

    for l_idx in range(len(composed.layers)):
        layer_c = composed.layers[l_idx]
        layer_a = model_a.layers[l_idx]
        layer_b = model_b.layers[l_idx]

        # Copy/merge attention
        layer_c.attn.wq.weight = merge(layer_a.attn.wq.weight, layer_b.attn.wq.weight)
        layer_c.attn.wk.weight = merge(layer_a.attn.wk.weight, layer_b.attn.wk.weight)
        layer_c.attn.wv.weight = merge(layer_a.attn.wv.weight, layer_b.attn.wv.weight)
        layer_c.attn.wo.weight = merge(layer_a.attn.wo.weight, layer_b.attn.wo.weight)

        # Copy capsule groups: first G from A, next G from B
        pool_c = layer_c.capsule_pool
        pool_a = layer_a.capsule_pool
        pool_b = layer_b.capsule_pool

        for g in range(n_groups_a):
            pool_c.groups[g].A.weight = pool_a.groups[g].A.weight
            pool_c.groups[g].B.weight = pool_a.groups[g].B.weight

        for g in range(n_groups_b):
            pool_c.groups[n_groups_a + g].A.weight = pool_b.groups[g].A.weight
            pool_c.groups[n_groups_a + g].B.weight = pool_b.groups[g].B.weight

        # Router is randomly initialized -- will be calibrated separately

    mx.eval(composed.parameters())
    return composed


def compose_from_shared_base(base_model, groups_a, groups_b, vocab_size, block_size):
    """Create a composed model using a shared pretrained base + domain capsule groups.

    This is the more realistic scenario: start from a pretrained base model,
    add domain-specific capsule groups on top. The base attention/embeddings
    are shared and frozen.
    """
    n_groups_a = len(groups_a[0])  # groups per layer
    n_groups_b = len(groups_b[0])
    composed_groups = n_groups_a + n_groups_b
    composed_top_k = 2 + 2  # same active fraction

    composed = get_model("capsule_moe",
                         vocab_size=vocab_size, block_size=block_size,
                         n_groups=composed_groups,
                         n_capsules_per_group=CAP["n_capsules_per_group"],
                         top_k_groups=composed_top_k,
                         **{k: BASE[k] for k in ["n_embd", "n_head", "n_layer"]})

    # Copy base model shared params
    composed.wte.weight = base_model.wte.weight
    composed.wpe.weight = base_model.wpe.weight
    composed.lm_head.weight = base_model.lm_head.weight

    for l_idx in range(len(composed.layers)):
        layer_c = composed.layers[l_idx]
        layer_base = base_model.layers[l_idx]

        # Copy attention from base
        layer_c.attn.wq.weight = layer_base.attn.wq.weight
        layer_c.attn.wk.weight = layer_base.attn.wk.weight
        layer_c.attn.wv.weight = layer_base.attn.wv.weight
        layer_c.attn.wo.weight = layer_base.attn.wo.weight

        # Copy domain-specific capsule groups
        pool_c = layer_c.capsule_pool
        for g in range(n_groups_a):
            pool_c.groups[g].A.weight = groups_a[l_idx][g].A.weight
            pool_c.groups[g].B.weight = groups_a[l_idx][g].B.weight
        for g in range(n_groups_b):
            pool_c.groups[n_groups_a + g].A.weight = groups_b[l_idx][g].A.weight
            pool_c.groups[n_groups_a + g].B.weight = groups_b[l_idx][g].B.weight

    mx.eval(composed.parameters())
    return composed


def freeze_except_router(model):
    """Freeze all parameters except the capsule pool routers."""
    model.freeze()
    for layer in model.layers:
        layer.capsule_pool.router.unfreeze()


def calibrate_router(model, train_ds_a, train_ds_b, steps=100, lr=3e-3, seed=42):
    """Train only the router weights on mixed-domain data."""
    freeze_except_router(model)

    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    losses = []
    for step in range(1, steps + 1):
        # Alternate batches from domain A and domain B
        if step % 2 == 1:
            inputs, targets = train_ds_a.get_batch(BATCH_SIZE, rng)
        else:
            inputs, targets = train_ds_b.get_batch(BATCH_SIZE, rng)

        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss.item())

        if step % 50 == 0 or step == steps:
            print(f"    router cal step {step:3d}/{steps} | loss {loss.item():.4f}")

    model.unfreeze()
    return losses


def run_composition_experiment(seed=42):
    """Run the full composition experiment for one seed."""
    print(f"\n{'='*70}")
    print(f"COMPOSITION EXPERIMENT (seed={seed})")
    print(f"{'='*70}")

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    domain_datasets = build_domain_datasets(tokenizer, docs, seed=seed)
    train_a, val_a = domain_datasets["a_m"]
    train_b, val_b = domain_datasets["n_z"]

    # Also build a joint dataset (all data)
    all_train, all_val = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_train, tokenizer, BASE["block_size"])
    joint_val = CharDataset(all_val, tokenizer, BASE["block_size"])

    results = {}

    # --- Baseline 1: Joint training (both domains simultaneously) ---
    print("\n--- Baseline: Joint training ---")
    model_joint = get_model("capsule_moe", vocab_size=V, **CAP)
    mx.eval(model_joint.parameters())

    # Train on alternating domain batches for 2*STEPS_PER_DOMAIN total
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(model_joint, ntp_loss)
    total_steps = 2 * STEPS_PER_DOMAIN
    for step in range(1, total_steps + 1):
        if step % 2 == 1:
            inputs, targets = train_a.get_batch(BATCH_SIZE, rng)
        else:
            inputs, targets = train_b.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model_joint, inputs, targets)
        optimizer.update(model_joint, grads)
        mx.eval(model_joint.parameters(), optimizer.state)
        if step % 100 == 0:
            print(f"  step {step:4d}/{total_steps} | loss {loss.item():.4f}")

    joint_val_a = evaluate(model_joint, val_a, BATCH_SIZE)
    joint_val_b = evaluate(model_joint, val_b, BATCH_SIZE)
    joint_avg = (joint_val_a + joint_val_b) / 2
    results["joint"] = {"a_m": joint_val_a, "n_z": joint_val_b, "avg": joint_avg}
    print(f"  Joint: a_m={joint_val_a:.4f}, n_z={joint_val_b:.4f}, avg={joint_avg:.4f}")

    # --- Baseline 2: Sequential training (A then B, shows forgetting) ---
    print("\n--- Baseline: Sequential training ---")
    model_seq = get_model("capsule_moe", vocab_size=V, **CAP)
    mx.eval(model_seq.parameters())
    # Train on A
    result_a = train(model_seq, train_a, val_a, steps=STEPS_PER_DOMAIN,
                     batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=100)
    # Train on B
    result_b = train(model_seq, train_b, val_b, steps=STEPS_PER_DOMAIN,
                     batch_size=BATCH_SIZE, lr=LR, seed=seed+1, log_every=100)

    seq_val_a = evaluate(model_seq, val_a, BATCH_SIZE)
    seq_val_b = evaluate(model_seq, val_b, BATCH_SIZE)
    seq_avg = (seq_val_a + seq_val_b) / 2
    results["sequential"] = {"a_m": seq_val_a, "n_z": seq_val_b, "avg": seq_avg}
    print(f"  Sequential: a_m={seq_val_a:.4f}, n_z={seq_val_b:.4f}, avg={seq_avg:.4f}")

    # --- Phase 1: Train Model_A on domain A ---
    print("\n--- Phase 1: Train domain-A specialist ---")
    model_a = get_model("capsule_moe", vocab_size=V, **CAP)
    mx.eval(model_a.parameters())
    train(model_a, train_a, val_a, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=100)
    spec_a_on_a = evaluate(model_a, val_a, BATCH_SIZE)
    spec_a_on_b = evaluate(model_a, val_b, BATCH_SIZE)
    print(f"  Model_A: a_m={spec_a_on_a:.4f}, n_z={spec_a_on_b:.4f}")

    # --- Phase 2: Train Model_B on domain B ---
    print("\n--- Phase 2: Train domain-B specialist ---")
    model_b = get_model("capsule_moe", vocab_size=V, **CAP)
    mx.eval(model_b.parameters())
    train(model_b, train_b, val_b, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=100)
    spec_b_on_a = evaluate(model_b, val_a, BATCH_SIZE)
    spec_b_on_b = evaluate(model_b, val_b, BATCH_SIZE)
    print(f"  Model_B: a_m={spec_b_on_a:.4f}, n_z={spec_b_on_b:.4f}")

    # --- Phase 3a: Compose with uniform routing, avg attention ---
    print("\n--- Phase 3a: Composed (uniform routing, avg attention) ---")
    composed_uniform = compose_capsule_models(model_a, model_b, V,
                                               BASE["block_size"], attn_strategy="avg")
    for layer in composed_uniform.layers:
        layer.capsule_pool.uniform_routing = True
    mx.eval(composed_uniform.parameters())

    comp_u_val_a = evaluate(composed_uniform, val_a, BATCH_SIZE)
    comp_u_val_b = evaluate(composed_uniform, val_b, BATCH_SIZE)
    comp_u_avg = (comp_u_val_a + comp_u_val_b) / 2
    results["composed_uniform"] = {"a_m": comp_u_val_a, "n_z": comp_u_val_b, "avg": comp_u_avg}
    print(f"  Composed (uniform): a_m={comp_u_val_a:.4f}, n_z={comp_u_val_b:.4f}, avg={comp_u_avg:.4f}")

    # --- Phase 3b: Compose with calibrated router, avg attention ---
    print("\n--- Phase 3b: Composed (calibrated router, avg attention) ---")
    composed_cal = compose_capsule_models(model_a, model_b, V,
                                           BASE["block_size"], attn_strategy="avg")
    mx.eval(composed_cal.parameters())

    calibrate_router(composed_cal, train_a, train_b,
                     steps=ROUTER_CALIBRATION_STEPS, lr=LR, seed=seed)

    comp_c_val_a = evaluate(composed_cal, val_a, BATCH_SIZE)
    comp_c_val_b = evaluate(composed_cal, val_b, BATCH_SIZE)
    comp_c_avg = (comp_c_val_a + comp_c_val_b) / 2
    results["composed_calibrated"] = {"a_m": comp_c_val_a, "n_z": comp_c_val_b, "avg": comp_c_avg}
    print(f"  Composed (calibrated): a_m={comp_c_val_a:.4f}, n_z={comp_c_val_b:.4f}, avg={comp_c_avg:.4f}")

    # --- Phase 4: Shared-base composition ---
    # More realistic: pretrain a base, fine-tune only capsule groups per domain
    print("\n--- Phase 4: Shared base + domain capsule fine-tuning ---")

    # 4a: Pretrain base on all data (300 steps)
    print("  4a: Pretraining shared base on all data...")
    base_model = get_model("capsule_moe", vocab_size=V, **CAP)
    mx.eval(base_model.parameters())
    joint_train_ds = CharDataset(all_train, tokenizer, BASE["block_size"])
    train(base_model, joint_train_ds, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=100)

    # 4b: Fine-tune only capsule groups on domain A (freeze attention)
    print("  4b: Fine-tuning capsule groups on domain A...")
    model_a2 = get_model("capsule_moe", vocab_size=V, **CAP)
    mx.eval(model_a2.parameters())
    # Copy base weights
    model_a2.load_weights(list(zip(
        [k for k, _ in nn.utils.tree_flatten(base_model.parameters())],
        [v for _, v in nn.utils.tree_flatten(base_model.parameters())]
    )))
    mx.eval(model_a2.parameters())
    # Freeze everything except capsule groups
    model_a2.freeze()
    for layer in model_a2.layers:
        for group in layer.capsule_pool.groups:
            group.unfreeze()
    train(model_a2, train_a, val_a, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=100)
    model_a2.unfreeze()

    # 4c: Fine-tune only capsule groups on domain B (freeze attention)
    print("  4c: Fine-tuning capsule groups on domain B...")
    model_b2 = get_model("capsule_moe", vocab_size=V, **CAP)
    mx.eval(model_b2.parameters())
    model_b2.load_weights(list(zip(
        [k for k, _ in nn.utils.tree_flatten(base_model.parameters())],
        [v for _, v in nn.utils.tree_flatten(base_model.parameters())]
    )))
    mx.eval(model_b2.parameters())
    model_b2.freeze()
    for layer in model_b2.layers:
        for group in layer.capsule_pool.groups:
            group.unfreeze()
    train(model_b2, train_b, val_b, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=100)
    model_b2.unfreeze()

    # 4d: Compose by extracting groups and using base attention
    print("  4d: Composing shared-base model...")
    groups_a_all = []
    groups_b_all = []
    for l_idx in range(len(base_model.layers)):
        groups_a_all.append(model_a2.layers[l_idx].capsule_pool.groups)
        groups_b_all.append(model_b2.layers[l_idx].capsule_pool.groups)

    composed_base = compose_from_shared_base(base_model, groups_a_all, groups_b_all,
                                              V, BASE["block_size"])
    # Uniform routing first
    for layer in composed_base.layers:
        layer.capsule_pool.uniform_routing = True
    mx.eval(composed_base.parameters())

    sb_u_val_a = evaluate(composed_base, val_a, BATCH_SIZE)
    sb_u_val_b = evaluate(composed_base, val_b, BATCH_SIZE)
    sb_u_avg = (sb_u_val_a + sb_u_val_b) / 2
    results["shared_base_uniform"] = {"a_m": sb_u_val_a, "n_z": sb_u_val_b, "avg": sb_u_avg}
    print(f"  Shared-base (uniform): a_m={sb_u_val_a:.4f}, n_z={sb_u_val_b:.4f}, avg={sb_u_avg:.4f}")

    # Calibrate router
    print("  4e: Calibrating router...")
    composed_base_cal = compose_from_shared_base(base_model, groups_a_all, groups_b_all,
                                                  V, BASE["block_size"])
    mx.eval(composed_base_cal.parameters())
    calibrate_router(composed_base_cal, train_a, train_b,
                     steps=ROUTER_CALIBRATION_STEPS, lr=LR, seed=seed)

    sb_c_val_a = evaluate(composed_base_cal, val_a, BATCH_SIZE)
    sb_c_val_b = evaluate(composed_base_cal, val_b, BATCH_SIZE)
    sb_c_avg = (sb_c_val_a + sb_c_val_b) / 2
    results["shared_base_calibrated"] = {"a_m": sb_c_val_a, "n_z": sb_c_val_b, "avg": sb_c_avg}
    print(f"  Shared-base (calibrated): a_m={sb_c_val_a:.4f}, n_z={sb_c_val_b:.4f}, avg={sb_c_avg:.4f}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'a_m':>8} {'n_z':>8} {'avg':>8} {'vs joint':>10}")
    print("-" * 60)
    for method, vals in results.items():
        delta = (vals["avg"] - results["joint"]["avg"]) / results["joint"]["avg"] * 100
        print(f"{method:<25} {vals['a_m']:>8.4f} {vals['n_z']:>8.4f} {vals['avg']:>8.4f} {delta:>+9.1f}%")

    return results


def run_multiseed(seeds=(42, 123, 7)):
    """Run composition experiment across multiple seeds."""
    all_results = {}
    for seed in seeds:
        all_results[seed] = run_composition_experiment(seed)

    # Aggregate
    print(f"\n\n{'='*70}")
    print("MULTI-SEED AGGREGATE")
    print(f"{'='*70}")

    methods = ["joint", "sequential", "composed_uniform", "composed_calibrated",
               "shared_base_uniform", "shared_base_calibrated"]
    print(f"{'Method':<28} {'avg (mean)':>12} {'avg (range)':>14} {'vs joint':>10}")
    print("-" * 68)

    joint_mean = None
    for method in methods:
        avgs = [all_results[s][method]["avg"] for s in seeds]
        mean_avg = statistics.mean(avgs)
        range_avg = max(avgs) - min(avgs)
        if method == "joint":
            joint_mean = mean_avg
            delta_str = "baseline"
        else:
            delta = (mean_avg - joint_mean) / joint_mean * 100
            delta_str = f"{delta:+.1f}%"
        print(f"{method:<28} {mean_avg:>12.4f} {'+/- ' + f'{range_avg:.4f}':>14} {delta_str:>10}")

    # Kill threshold check
    composition_methods = ["composed_uniform", "composed_calibrated",
                           "shared_base_uniform", "shared_base_calibrated"]
    print()
    for method in composition_methods:
        avgs = [all_results[s][method]["avg"] for s in seeds]
        mean_avg = statistics.mean(avgs)
        delta_pct = (mean_avg - joint_mean) / joint_mean * 100
        if delta_pct > 5.0:
            print(f"  ** KILL THRESHOLD EXCEEDED for {method}: {delta_pct:+.1f}% > 5% **")
        else:
            print(f"  OK {method}: {delta_pct:+.1f}% vs joint -- within 5% threshold")


if __name__ == "__main__":
    run_multiseed()
