"""Contrastive routing key experiment.

Validates that InfoNCE-trained contrastive keys can:
1. Achieve >85% routing accuracy with 50 samples/domain, 50 steps
2. Match or beat softmax router calibration for composition quality
3. Outperform a simple linear probe baseline

Experiment phases:
  1. Pretrain shared base on all data (300 steps)
  2. Fine-tune capsule groups per domain (freeze attention, 300 steps each)
  3. Compose: shared base + domain groups
  4a. Baseline: calibrate softmax router (100 steps on mixed data)
  4b. Contrastive: calibrate InfoNCE keys (50 samples, 50 steps)
  5. Measure routing accuracy and composition quality
  6. Compare baselines (joint, softmax, linear probe)

Kill thresholds (from MATH.md Section 10):
  - Routing accuracy < 70%: dead
  - Composition quality > 10% worse than joint: dead
  - More data/steps than softmax router: no advantage
  - Same accuracy as linear probe: contrastive loss adds nothing
"""

import copy
import random
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.models import get_model
from experiments.data import (
    load_names, CharTokenizer, CharDataset,
    domain_split, train_val_split,
)
from experiments.train import train, evaluate, ntp_loss
from experiments.models.contrastive_router.contrastive_router import (
    extract_hidden_states, infonce_loss, routing_accuracy,
)


# Standard micro config
BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
CAP = dict(**BASE, n_groups=4, n_capsules_per_group=64, top_k_groups=2)
STEPS_PER_DOMAIN = 300
BATCH_SIZE = 32
LR = 3e-3

# Contrastive key calibration defaults
CAL_SAMPLES = 2       # batches per domain (2 * 32 tokens = 64 per domain)
CAL_STEPS = 50        # InfoNCE optimization steps
CAL_TAU = 0.1         # temperature
D_KEY = 8             # routing key dimension

# Softmax router calibration
ROUTER_CAL_STEPS = 100


def build_domain_datasets(tokenizer, docs, block_size=32, seed=42):
    """Build train/val datasets per domain."""
    splits = domain_split(docs)
    datasets = {}
    for name, ddocs in splits.items():
        dtrain, dval = train_val_split(ddocs, seed=seed)
        datasets[name] = (
            CharDataset(dtrain, tokenizer, block_size),
            CharDataset(dval, tokenizer, block_size),
        )
    return datasets


def compose_contrastive_model(base_model, groups_a, groups_b,
                               vocab_size, block_size, d_key=D_KEY):
    """Create composed ContrastiveRouterGPT from shared base + domain groups."""
    n_groups_a = len(groups_a[0])
    n_groups_b = len(groups_b[0])
    composed_groups = n_groups_a + n_groups_b
    composed_top_k = 2 + 2  # same active fraction

    composed = get_model("contrastive_router",
                         vocab_size=vocab_size, block_size=block_size,
                         n_groups=composed_groups,
                         n_capsules_per_group=CAP["n_capsules_per_group"],
                         top_k_groups=composed_top_k, d_key=d_key,
                         **{k: BASE[k] for k in ["n_embd", "n_head", "n_layer"]})

    # Copy shared params from base
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

        # Copy capsule groups: first G from A, next G from B
        pool_c = layer_c.capsule_pool
        for g in range(n_groups_a):
            pool_c.groups[g].A.weight = groups_a[l_idx][g].A.weight
            pool_c.groups[g].B.weight = groups_a[l_idx][g].B.weight
        for g in range(n_groups_b):
            pool_c.groups[n_groups_a + g].A.weight = groups_b[l_idx][g].A.weight
            pool_c.groups[n_groups_a + g].B.weight = groups_b[l_idx][g].B.weight

        # Routing keys: stay randomly initialized (will be trained)

    mx.eval(composed.parameters())
    return composed


def compose_softmax_model(base_model, groups_a, groups_b, vocab_size, block_size):
    """Create composed CapsuleMoEGPT with softmax router (baseline)."""
    from experiments.models.capsule_moe.test_composition import compose_from_shared_base
    return compose_from_shared_base(base_model, groups_a, groups_b,
                                     vocab_size, block_size)


def freeze_except_router(model):
    """Freeze all parameters except the softmax router."""
    model.freeze()
    for layer in model.layers:
        layer.capsule_pool.router.unfreeze()


def calibrate_softmax_router(model, train_ds_a, train_ds_b,
                              steps=ROUTER_CAL_STEPS, lr=LR, seed=42):
    """Train only the softmax router on mixed-domain data."""
    freeze_except_router(model)
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
        if step % 50 == 0 or step == steps:
            print(f"    softmax cal step {step:3d}/{steps} | loss {loss.item():.4f}")

    model.unfreeze()


def collect_hidden_states(model, train_ds, n_batches=CAL_SAMPLES, seed=42):
    """Extract hidden states from model for one domain."""
    rng = random.Random(seed)
    all_hiddens = None
    for _ in range(n_batches):
        inputs, _ = train_ds.get_batch(BATCH_SIZE, rng)
        hiddens = extract_hidden_states(model, inputs)
        mx.eval(hiddens)
        if all_hiddens is None:
            all_hiddens = [h.reshape(-1, h.shape[-1]) for h in hiddens]
        else:
            for i, h in enumerate(hiddens):
                all_hiddens[i] = mx.concatenate(
                    [all_hiddens[i], h.reshape(-1, h.shape[-1])], axis=0)
    mx.eval(all_hiddens)
    return all_hiddens  # list of (N_tokens, d), one per layer


def calibrate_contrastive_keys(model, train_ds_a, train_ds_b,
                                n_batches=CAL_SAMPLES, steps=CAL_STEPS,
                                tau=CAL_TAU, lr=LR, seed=42):
    """Train contrastive routing keys via InfoNCE on labeled hidden states.

    1. Set uniform routing for extraction
    2. Extract hidden states per domain
    3. Train each layer's keys independently
    """
    # Step 1: Enable uniform routing for extraction
    for layer in model.layers:
        layer.capsule_pool.uniform_routing = True

    # Step 2: Extract hidden states
    h_a = collect_hidden_states(model, train_ds_a, n_batches, seed)
    h_b = collect_hidden_states(model, train_ds_b, n_batches, seed + 1)

    # Disable uniform routing
    for layer in model.layers:
        layer.capsule_pool.uniform_routing = False

    n_layers = len(model.layers)
    groups_per_domain = model.layers[0].capsule_pool.n_groups // 2

    per_layer_acc = []
    per_layer_loss = []

    for layer_idx in range(n_layers):
        # Combine and label
        h_all = mx.concatenate([h_a[layer_idx], h_b[layer_idx]], axis=0)
        labels = mx.concatenate([
            mx.zeros(h_a[layer_idx].shape[0], dtype=mx.int32),
            mx.ones(h_b[layer_idx].shape[0], dtype=mx.int32),
        ])
        mx.eval(h_all, labels)

        pool = model.layers[layer_idx].capsule_pool

        # Freeze groups, train only keys
        pool.freeze()
        for key in pool.routing_keys:
            key.unfreeze()

        optimizer = optim.Adam(learning_rate=lr)

        def loss_fn(pool, h, labels):
            return infonce_loss(pool, h, labels, groups_per_domain, tau)

        loss_and_grad = nn.value_and_grad(pool, loss_fn)

        for step in range(1, steps + 1):
            loss, grads = loss_and_grad(pool, h_all, labels)
            optimizer.update(pool, grads)
            mx.eval(pool.parameters(), optimizer.state)

        final_loss = loss.item()
        acc = routing_accuracy(pool, h_all, labels, groups_per_domain)
        per_layer_acc.append(acc)
        per_layer_loss.append(final_loss)

        pool.unfreeze()
        print(f"    layer {layer_idx}: loss={final_loss:.4f}, train_acc={acc:.1%}")

    return {
        "per_layer_acc": per_layer_acc,
        "per_layer_loss": per_layer_loss,
        "avg_acc": statistics.mean(per_layer_acc),
    }


def measure_held_out_accuracy(model, test_ds_a, test_ds_b,
                               n_batches=CAL_SAMPLES, seed=99):
    """Measure routing accuracy on held-out data."""
    # Use uniform routing for hidden state extraction
    for layer in model.layers:
        layer.capsule_pool.uniform_routing = True

    h_a = collect_hidden_states(model, test_ds_a, n_batches, seed)
    h_b = collect_hidden_states(model, test_ds_b, n_batches, seed + 1)

    for layer in model.layers:
        layer.capsule_pool.uniform_routing = False

    groups_per_domain = model.layers[0].capsule_pool.n_groups // 2
    per_layer_acc = []

    for layer_idx in range(len(model.layers)):
        h_all = mx.concatenate([h_a[layer_idx], h_b[layer_idx]], axis=0)
        labels = mx.concatenate([
            mx.zeros(h_a[layer_idx].shape[0], dtype=mx.int32),
            mx.ones(h_b[layer_idx].shape[0], dtype=mx.int32),
        ])
        mx.eval(h_all, labels)
        pool = model.layers[layer_idx].capsule_pool
        acc = routing_accuracy(pool, h_all, labels, groups_per_domain)
        per_layer_acc.append(acc)

    return {
        "per_layer_acc": per_layer_acc,
        "avg_acc": statistics.mean(per_layer_acc),
    }


class LinearProbe(nn.Module):
    """Simple linear classifier for domain detection baseline."""
    def __init__(self, n_embd, n_domains=2):
        super().__init__()
        self.linear = nn.Linear(n_embd, n_domains, bias=False)

    def __call__(self, h):
        return self.linear(h)


def train_linear_probe(h_all, labels, n_embd, n_domains=2,
                        steps=CAL_STEPS, lr=LR):
    """Train a linear probe baseline and return accuracy."""
    probe = LinearProbe(n_embd, n_domains)
    mx.eval(probe.parameters())

    def loss_fn(probe, h, labels):
        return nn.losses.cross_entropy(probe(h), labels, reduction="mean")

    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(probe, loss_fn)

    for step in range(steps):
        loss, grads = loss_and_grad(probe, h_all, labels)
        optimizer.update(probe, grads)
        mx.eval(probe.parameters(), optimizer.state)

    preds = mx.argmax(probe(h_all), axis=-1)
    mx.eval(preds)
    return (mx.sum(preds == labels) / labels.size).item()


def run_experiment(seed=42):
    """Run the full contrastive routing experiment for one seed."""
    print(f"\n{'='*70}")
    print(f"CONTRASTIVE ROUTING EXPERIMENT (seed={seed})")
    print(f"{'='*70}")

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    domain_datasets = build_domain_datasets(tokenizer, docs, seed=seed)
    train_a, val_a = domain_datasets["a_m"]
    train_b, val_b = domain_datasets["n_z"]

    all_train, all_val = train_val_split(docs, seed=seed)
    results = {}

    # --- Baseline 1: Joint training ---
    print("\n--- Baseline: Joint training ---")
    model_joint = get_model("capsule_moe", vocab_size=V, **CAP)
    mx.eval(model_joint.parameters())
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
        if step % 200 == 0:
            print(f"  step {step:4d}/{total_steps} | loss {loss.item():.4f}")

    joint_val_a = evaluate(model_joint, val_a, BATCH_SIZE)
    joint_val_b = evaluate(model_joint, val_b, BATCH_SIZE)
    joint_avg = (joint_val_a + joint_val_b) / 2
    results["joint"] = {"a_m": joint_val_a, "n_z": joint_val_b, "avg": joint_avg}
    print(f"  Joint: a_m={joint_val_a:.4f}, n_z={joint_val_b:.4f}, avg={joint_avg:.4f}")

    # --- Shared base + domain fine-tuning ---
    print("\n--- Pretraining shared base (all data, 300 steps) ---")
    base_model = get_model("capsule_moe", vocab_size=V, **CAP)
    mx.eval(base_model.parameters())
    joint_ds = CharDataset(all_train, tokenizer, BASE["block_size"])
    train(base_model, joint_ds, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)

    # Fine-tune domain A (freeze attention, train capsule groups)
    print("\n--- Fine-tuning capsule groups on domain A ---")
    model_a = get_model("capsule_moe", vocab_size=V, **CAP)
    mx.eval(model_a.parameters())
    model_a.load_weights(list(zip(
        [k for k, _ in nn.utils.tree_flatten(base_model.parameters())],
        [v for _, v in nn.utils.tree_flatten(base_model.parameters())]
    )))
    mx.eval(model_a.parameters())
    model_a.freeze()
    for layer in model_a.layers:
        for group in layer.capsule_pool.groups:
            group.unfreeze()
    train(model_a, train_a, val_a, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)
    model_a.unfreeze()

    # Fine-tune domain B
    print("\n--- Fine-tuning capsule groups on domain B ---")
    model_b = get_model("capsule_moe", vocab_size=V, **CAP)
    mx.eval(model_b.parameters())
    model_b.load_weights(list(zip(
        [k for k, _ in nn.utils.tree_flatten(base_model.parameters())],
        [v for _, v in nn.utils.tree_flatten(base_model.parameters())]
    )))
    mx.eval(model_b.parameters())
    model_b.freeze()
    for layer in model_b.layers:
        for group in layer.capsule_pool.groups:
            group.unfreeze()
    train(model_b, train_b, val_b, steps=STEPS_PER_DOMAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=150)
    model_b.unfreeze()

    # Extract domain-specific groups
    groups_a = [layer.capsule_pool.groups for layer in model_a.layers]
    groups_b = [layer.capsule_pool.groups for layer in model_b.layers]

    # --- Compose with softmax router (baseline) ---
    print("\n--- Composed + softmax router calibration ---")
    composed_softmax = compose_softmax_model(base_model, groups_a, groups_b,
                                              V, BASE["block_size"])
    mx.eval(composed_softmax.parameters())
    calibrate_softmax_router(composed_softmax, train_a, train_b, seed=seed)

    sm_val_a = evaluate(composed_softmax, val_a, BATCH_SIZE)
    sm_val_b = evaluate(composed_softmax, val_b, BATCH_SIZE)
    sm_avg = (sm_val_a + sm_val_b) / 2
    results["softmax_calibrated"] = {"a_m": sm_val_a, "n_z": sm_val_b, "avg": sm_avg}
    print(f"  Softmax: a_m={sm_val_a:.4f}, n_z={sm_val_b:.4f}, avg={sm_avg:.4f}")

    # --- Compose with contrastive keys ---
    print("\n--- Composed + contrastive key calibration ---")
    composed_contrastive = compose_contrastive_model(
        base_model, groups_a, groups_b, V, BASE["block_size"], d_key=D_KEY)

    # Calibrate keys
    cal_result = calibrate_contrastive_keys(
        composed_contrastive, train_a, train_b,
        n_batches=CAL_SAMPLES, steps=CAL_STEPS, tau=CAL_TAU, lr=LR, seed=seed)

    train_acc = cal_result["avg_acc"]
    print(f"  Train routing accuracy: {train_acc:.1%}")

    # Held-out routing accuracy
    held_out = measure_held_out_accuracy(
        composed_contrastive, val_a, val_b, n_batches=CAL_SAMPLES, seed=seed+100)
    held_out_acc = held_out["avg_acc"]
    print(f"  Held-out routing accuracy: {held_out_acc:.1%}")
    for i, acc in enumerate(held_out["per_layer_acc"]):
        print(f"    layer {i}: {acc:.1%}")

    # Composition quality
    cr_val_a = evaluate(composed_contrastive, val_a, BATCH_SIZE)
    cr_val_b = evaluate(composed_contrastive, val_b, BATCH_SIZE)
    cr_avg = (cr_val_a + cr_val_b) / 2
    results["contrastive_keys"] = {"a_m": cr_val_a, "n_z": cr_val_b, "avg": cr_avg}
    print(f"  Contrastive: a_m={cr_val_a:.4f}, n_z={cr_val_b:.4f}, avg={cr_avg:.4f}")

    # --- Linear probe baseline ---
    print("\n--- Linear probe baseline ---")
    # Extract hidden states from composed model (uniform routing)
    for layer in composed_contrastive.layers:
        layer.capsule_pool.uniform_routing = True
    h_a = collect_hidden_states(composed_contrastive, train_a, CAL_SAMPLES, seed)
    h_b = collect_hidden_states(composed_contrastive, train_b, CAL_SAMPLES, seed+1)
    for layer in composed_contrastive.layers:
        layer.capsule_pool.uniform_routing = False

    probe_accs = []
    for layer_idx in range(len(composed_contrastive.layers)):
        h_all = mx.concatenate([h_a[layer_idx], h_b[layer_idx]], axis=0)
        labels = mx.concatenate([
            mx.zeros(h_a[layer_idx].shape[0], dtype=mx.int32),
            mx.ones(h_b[layer_idx].shape[0], dtype=mx.int32),
        ])
        mx.eval(h_all, labels)
        acc = train_linear_probe(h_all, labels, BASE["n_embd"],
                                  steps=CAL_STEPS, lr=LR)
        probe_accs.append(acc)
        print(f"    layer {layer_idx}: {acc:.1%}")
    probe_avg = statistics.mean(probe_accs)
    print(f"  Linear probe avg accuracy: {probe_avg:.1%}")

    # --- Tau sweep ---
    print("\n--- Tau sweep ---")
    tau_results = {}
    for tau in [0.05, 0.1, 0.5, 1.0]:
        model_tau = compose_contrastive_model(
            base_model, groups_a, groups_b, V, BASE["block_size"], d_key=D_KEY)
        cal = calibrate_contrastive_keys(
            model_tau, train_a, train_b,
            n_batches=CAL_SAMPLES, steps=CAL_STEPS, tau=tau, lr=LR, seed=seed)
        ho = measure_held_out_accuracy(model_tau, val_a, val_b,
                                        n_batches=CAL_SAMPLES, seed=seed+100)
        val_loss = (evaluate(model_tau, val_a, BATCH_SIZE) +
                    evaluate(model_tau, val_b, BATCH_SIZE)) / 2
        tau_results[tau] = {"train_acc": cal["avg_acc"],
                            "held_out_acc": ho["avg_acc"],
                            "val_loss": val_loss}
        print(f"  tau={tau}: train_acc={cal['avg_acc']:.1%}, "
              f"held_out_acc={ho['avg_acc']:.1%}, val_loss={val_loss:.4f}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'a_m':>8} {'n_z':>8} {'avg':>8} {'vs joint':>10}")
    print("-" * 62)
    for method, vals in results.items():
        delta = (vals["avg"] - results["joint"]["avg"]) / results["joint"]["avg"] * 100
        print(f"{method:<25} {vals['a_m']:>8.4f} {vals['n_z']:>8.4f} "
              f"{vals['avg']:>8.4f} {delta:>+9.1f}%")

    print(f"\nRouting accuracy (held-out): {held_out_acc:.1%}")
    print(f"Linear probe accuracy:      {probe_avg:.1%}")

    # Kill threshold checks
    print(f"\n--- Kill threshold checks ---")
    if held_out_acc < 0.70:
        print(f"  ** KILL: routing accuracy {held_out_acc:.1%} < 70% **")
    elif held_out_acc < 0.85:
        print(f"  MARGINAL: routing accuracy {held_out_acc:.1%} (70-85% range)")
    else:
        print(f"  PASS: routing accuracy {held_out_acc:.1%} >= 85%")

    cr_delta = (cr_avg - joint_avg) / joint_avg * 100
    if cr_delta > 10.0:
        print(f"  ** KILL: composition quality {cr_delta:+.1f}% > 10% worse than joint **")
    elif cr_delta > 5.0:
        print(f"  MARGINAL: composition quality {cr_delta:+.1f}% (5-10% range)")
    else:
        print(f"  PASS: composition quality {cr_delta:+.1f}% within 5% of joint")

    if held_out_acc <= probe_avg + 0.01:
        print(f"  ** KILL: contrastive keys ({held_out_acc:.1%}) not better than "
              f"linear probe ({probe_avg:.1%}) **")
    else:
        print(f"  PASS: contrastive keys ({held_out_acc:.1%}) > "
              f"linear probe ({probe_avg:.1%})")

    return {
        "results": results,
        "routing": {
            "train_acc": train_acc,
            "held_out_acc": held_out_acc,
            "per_layer": held_out["per_layer_acc"],
        },
        "linear_probe_acc": probe_avg,
        "tau_sweep": tau_results,
    }


def run_multiseed(seeds=(42, 123, 7)):
    """Run experiment across multiple seeds and aggregate."""
    all_results = {}
    for seed in seeds:
        all_results[seed] = run_experiment(seed)

    print(f"\n\n{'='*70}")
    print("MULTI-SEED AGGREGATE")
    print(f"{'='*70}")

    # Composition quality
    methods = ["joint", "softmax_calibrated", "contrastive_keys"]
    print(f"\n{'Method':<25} {'avg (mean)':>12} {'avg (std)':>12} {'vs joint':>10}")
    print("-" * 62)

    joint_mean = None
    for method in methods:
        avgs = [all_results[s]["results"][method]["avg"] for s in seeds]
        mean_avg = statistics.mean(avgs)
        std_avg = statistics.stdev(avgs) if len(avgs) > 1 else 0
        if method == "joint":
            joint_mean = mean_avg
            delta_str = "baseline"
        else:
            delta = (mean_avg - joint_mean) / joint_mean * 100
            delta_str = f"{delta:+.1f}%"
        print(f"{method:<25} {mean_avg:>12.4f} {'+/- ' + f'{std_avg:.4f}':>12} "
              f"{delta_str:>10}")

    # Routing accuracy
    held_out_accs = [all_results[s]["routing"]["held_out_acc"] for s in seeds]
    probe_accs = [all_results[s]["linear_probe_acc"] for s in seeds]
    print(f"\nRouting accuracy (mean): {statistics.mean(held_out_accs):.1%} "
          f"+/- {statistics.stdev(held_out_accs):.1%}" if len(held_out_accs) > 1
          else f"\nRouting accuracy: {held_out_accs[0]:.1%}")
    print(f"Linear probe (mean):    {statistics.mean(probe_accs):.1%}")

    # Final verdict
    mean_acc = statistics.mean(held_out_accs)
    mean_probe = statistics.mean(probe_accs)
    mean_contrastive = statistics.mean(
        [all_results[s]["results"]["contrastive_keys"]["avg"] for s in seeds])
    mean_joint = statistics.mean(
        [all_results[s]["results"]["joint"]["avg"] for s in seeds])
    delta_pct = (mean_contrastive - mean_joint) / mean_joint * 100

    print(f"\n--- Final verdict ---")
    if mean_acc >= 0.85 and delta_pct <= 5.0 and mean_acc > mean_probe + 0.01:
        print("  PASS: All success criteria met.")
    elif mean_acc < 0.70 or delta_pct > 10.0:
        print("  KILL: Kill threshold exceeded.")
    else:
        print("  MARGINAL: Promising but not all criteria met.")

    return all_results


if __name__ == "__main__":
    run_multiseed()
