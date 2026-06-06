"""Sequential Freeze-Graft-Calibrate Experiment (N=2,3,4).

Tests whether the freeze-graft protocol scales to multiple sequential grafts.

Protocol for each graft step:
  1. Freeze the domain-A subtree (or previously grafted subtrees)
  2. Reinitialize the target subtree for the new domain
  3. Train the new subtree on domain-specific data
  4. Calibrate all unfrozen params on mixed data from ALL domains seen so far
  5. Measure degradation on every previously frozen domain

Tree topology (depth=3, 8 leaves, 7 gates):
  Gate indices: 0(root), 1(left child of root), 2(right child of root),
                3,4(children of 1), 5,6(children of 2)
  Leaf indices: 0-7

Progressive allocation:
  Step 0: Train ALL on domain A (400 steps)
  Graft 1 (N=2): Freeze leaves 0-3 + gates 1,3,4.
                  Reinit leaves 4-7 + gates 2,5,6. Train on B. Calibrate.
  Graft 2 (N=3): Freeze leaves 4-5 + gate 5.
                  Reinit leaves 6-7 + gate 6. Train on C. Calibrate.
  Graft 3 (N=4): Freeze leaf 6.
                  Reinit leaf 7. Train on D. Calibrate.

Two calibration strategies tested at each step:
  (a) all-unfrozen: calibrate all unfrozen gates + leaves
  (b) selective: calibrate only root + graft-point gates (minimal_graft_recal approach)
"""

import sys
import time
import random
import copy

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

from experiments.data import (
    load_names, CharTokenizer, CharDataset, domain_split, train_val_split,
)
from experiments.train import ntp_loss, evaluate
from experiments.models import get_model


def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def train_model(model, train_ds, val_ds, steps=300, batch_size=32, lr=3e-3,
                seed=42, log_every=100):
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    t0 = time.time()
    for step in range(1, steps + 1):
        inputs, targets = train_ds.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % log_every == 0 or step == steps:
            elapsed = time.time() - t0
            print(f"    step {step:4d}/{steps} | loss {loss.item():.4f} | {elapsed:.1f}s")

    val_loss = evaluate(model, val_ds, batch_size) if val_ds else None
    return {"val_loss": val_loss, "elapsed_s": time.time() - t0}


def freeze_components(model, leaf_indices, gate_indices):
    """Freeze specific leaves and gates across all layers."""
    for layer in model.layers:
        for li in leaf_indices:
            if li < len(layer.tree.leaves):
                layer.tree.leaves[li].freeze()
        for gi in gate_indices:
            if gi < len(layer.tree.gates):
                layer.tree.gates[gi].freeze()


def reinit_components(model, leaf_indices, gate_indices, seed=0):
    """Reinitialize specific leaves and gates with fresh random weights."""
    mx.random.seed(seed)
    for layer in model.layers:
        for li in leaf_indices:
            if li < len(layer.tree.leaves):
                leaf = layer.tree.leaves[li]
                d = leaf.A.weight.shape[1]
                n_caps = leaf.A.weight.shape[0]
                scale = (2.0 / (d + n_caps)) ** 0.5
                new_A = mx.random.normal((n_caps, d)) * scale
                new_B = mx.random.normal((d, n_caps)) * scale
                leaf.A.load_weights([("weight", new_A)])
                leaf.B.load_weights([("weight", new_B)])
        for gi in gate_indices:
            if gi < len(layer.tree.gates):
                gate = layer.tree.gates[gi]
                d_in = gate.proj.weight.shape[1]
                s = (1.0 / d_in) ** 0.5
                new_w = mx.random.normal(gate.proj.weight.shape) * s
                new_b = mx.zeros(gate.proj.bias.shape)
                gate.proj.load_weights([("weight", new_w), ("bias", new_b)])
    mx.eval(model.parameters())


def freeze_shared_params(model):
    """Freeze embeddings, norms, attention, lm_head."""
    model.wte.freeze()
    model.wpe.freeze()
    model.norm0.freeze()
    model.lm_head.freeze()
    for layer in model.layers:
        layer.norm1.freeze()
        layer.attn.freeze()
        layer.norm2.freeze()


def freeze_all_tree(model):
    """Freeze all gates and leaves in the tree."""
    for layer in model.layers:
        for gate in layer.tree.gates:
            gate.freeze()
        for leaf in layer.tree.leaves:
            leaf.freeze()


def unfreeze_components(model, leaf_indices, gate_indices):
    """Unfreeze specific leaves and gates."""
    for layer in model.layers:
        for li in leaf_indices:
            if li < len(layer.tree.leaves):
                layer.tree.leaves[li].unfreeze()
        for gi in gate_indices:
            if gi < len(layer.tree.gates):
                layer.tree.gates[gi].unfreeze()


# ── Graft step definitions ─────────────────────────────────────────────────
# Each step defines what to freeze, what to reinit, and what to calibrate.

GRAFT_STEPS = [
    {
        "name": "graft_1 (N=2)",
        "description": "Freeze domain A (left subtree), graft domain B (right subtree)",
        "freeze_leaves": {0, 1, 2, 3},
        "freeze_gates": {1, 3, 4},
        "reinit_leaves": {4, 5, 6, 7},
        "reinit_gates": {2, 5, 6},
        # All-unfrozen calibration: root + right subtree
        "cal_all_gates": {0, 2, 5, 6},
        "cal_all_leaves": {4, 5, 6, 7},
        # Selective calibration: root + graft-point only
        "cal_sel_gates": {0, 2},
        "cal_sel_leaves": set(),
    },
    {
        "name": "graft_2 (N=3)",
        "description": "Freeze domain B half (leaves 4-5), graft domain C (leaves 6-7)",
        "freeze_leaves": {4, 5},
        "freeze_gates": {5},
        "reinit_leaves": {6, 7},
        "reinit_gates": {6},
        # All-unfrozen calibration: root + gate 2 + gate 6 + leaves 6-7
        "cal_all_gates": {0, 2, 6},
        "cal_all_leaves": {6, 7},
        # Selective: root + gate 2 (graft-point for right subtree)
        "cal_sel_gates": {0, 2},
        "cal_sel_leaves": set(),
    },
    {
        "name": "graft_3 (N=4)",
        "description": "Freeze domain C (leaf 6), graft domain D (leaf 7)",
        "freeze_leaves": {6},
        "freeze_gates": set(),  # gate 6 already routes between leaves 6 and 7
        "reinit_leaves": {7},
        "reinit_gates": set(),  # gate 6 needs recalibration, not reinit
        # All-unfrozen calibration: root + gate 2 + gate 6 + leaf 7
        "cal_all_gates": {0, 2, 6},
        "cal_all_leaves": {7},
        # Selective: root + gate 6 (graft-point for this specific graft)
        "cal_sel_gates": {0, 6},
        "cal_sel_leaves": set(),
    },
]


def run_sequential_experiment(seeds=(42, 123, 777),
                               steps_base=400,
                               steps_graft=200,
                               steps_calibrate=200,
                               batch_size=32,
                               lr=3e-3):
    """Run the full sequential freeze-graft experiment."""
    print("=" * 70)
    print("SEQUENTIAL FREEZE-GRAFT-CALIBRATE (N=2,3,4)")
    print("=" * 70)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vs = tokenizer.vocab_size
    splits = domain_split(docs, method="quaternary")
    domain_names = list(splits.keys())
    print(f"Domains: {domain_names}")
    print(f"Domain sizes: {[len(splits[d]) for d in domain_names]}")

    tree_cfg = dict(vocab_size=vs, block_size=32, tree_depth=3,
                    n_capsules_per_leaf=32, beam_width=2)

    # Track all results across seeds
    all_results = {
        "per_seed": [],
        "configs": ["all-unfrozen", "selective"],
    }

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        seed_results = {
            "seed": seed,
            "grafts": [],
        }

        # Prepare domain datasets
        domain_datasets = {}
        for d_name, d_docs in splits.items():
            d_train, d_val = train_val_split(d_docs, seed=seed)
            domain_datasets[d_name] = (
                CharDataset(d_train, tokenizer, 32),
                CharDataset(d_val, tokenizer, 32),
            )

        # Joint dataset (all domains)
        all_train_docs, all_val_docs = train_val_split(docs, seed=seed)
        joint_train = CharDataset(all_train_docs, tokenizer, 32)
        joint_val = CharDataset(all_val_docs, tokenizer, 32)

        # ── Step 0: Train base model on domain A ────────────────────────
        d_a_name = domain_names[0]
        d_a_train, d_a_val = domain_datasets[d_a_name]
        print(f"\n  [BASE] Training on {d_a_name} ({steps_base} steps)")

        mx.random.seed(seed)
        model = get_model("sequential_freeze_graft", **tree_cfg)
        mx.eval(model.parameters())
        _ = train_model(model, d_a_train, d_a_val, steps=steps_base,
                        seed=seed, log_every=200)

        # Evaluate baseline on domain A
        base_losses = {}
        base_losses[d_a_name] = evaluate(model, d_a_val, batch_size)
        print(f"    baseline {d_a_name}: {base_losses[d_a_name]:.4f}")

        # Save base weights for both calibration strategies
        base_weights = {k: mx.array(v) for k, v
                        in nn.utils.tree_flatten(model.parameters())}

        # We'll track the model state after each graft for BOTH strategies
        # Strategy weights track the model state across sequential grafts
        strategy_weights = {}
        for strat in ["all-unfrozen", "selective"]:
            strategy_weights[strat] = dict(base_weights)

        # ── Sequential graft steps ──────────────────────────────────────
        for graft_idx, graft in enumerate(GRAFT_STEPS):
            domain_idx = graft_idx + 1  # domain B=1, C=2, D=3
            new_domain_name = domain_names[domain_idx]
            d_train, d_val = domain_datasets[new_domain_name]

            # Domains frozen so far (A plus any previously grafted)
            frozen_domains = domain_names[:domain_idx]
            active_domains = domain_names[:domain_idx + 1]

            print(f"\n  [{graft['name']}] Grafting {new_domain_name}")
            print(f"    Frozen domains: {frozen_domains}")

            graft_result = {
                "name": graft["name"],
                "new_domain": new_domain_name,
                "n_domains": domain_idx + 1,
                "strategies": {},
            }

            for strat in ["all-unfrozen", "selective"]:
                print(f"\n    Strategy: {strat}")

                # Restore model to state after previous graft
                mx.random.seed(seed + graft_idx * 1000)
                strat_model = get_model("sequential_freeze_graft", **tree_cfg)
                strat_model.load_weights(list(strategy_weights[strat].items()))
                mx.eval(strat_model.parameters())

                # Apply freezes from this graft step
                freeze_components(strat_model,
                                  graft["freeze_leaves"],
                                  graft["freeze_gates"])

                # Reinitialize new subtree
                reinit_components(strat_model,
                                  graft["reinit_leaves"],
                                  graft["reinit_gates"],
                                  seed=seed + graft_idx * 100)

                # Freeze shared params (embeddings, attention, norms)
                freeze_shared_params(strat_model)

                # Train new subtree on new domain
                n_train = count_params(strat_model)
                print(f"      training params (graft phase): {n_train}")
                t0 = time.time()
                _ = train_model(strat_model, d_train, d_val,
                                steps=steps_graft, seed=seed + graft_idx,
                                log_every=200)
                graft_time = time.time() - t0

                # ── Calibration phase ───────────────────────────────────
                # Freeze everything, then unfreeze only calibration params
                freeze_all_tree(strat_model)

                if strat == "all-unfrozen":
                    cal_gates = graft["cal_all_gates"]
                    cal_leaves = graft["cal_all_leaves"]
                else:
                    cal_gates = graft["cal_sel_gates"]
                    cal_leaves = graft["cal_sel_leaves"]

                unfreeze_components(strat_model, cal_leaves, cal_gates)
                n_cal = count_params(strat_model)
                print(f"      calibration params: {n_cal}")

                # Calibrate on mixed data from all domains seen so far
                # Build mixed dataset
                mixed_docs = []
                for d_name in active_domains:
                    d_t, _ = domain_datasets[d_name]
                    mixed_docs.extend(d_t.sequences)
                mixed_train = CharDataset.__new__(CharDataset)
                mixed_train.tokenizer = tokenizer
                mixed_train.block_size = 32
                mixed_train.sequences = mixed_docs

                t0 = time.time()
                _ = train_model(strat_model, mixed_train, None,
                                steps=steps_calibrate, seed=seed + graft_idx * 10,
                                log_every=200)
                cal_time = time.time() - t0

                # ── Evaluation ──────────────────────────────────────────
                # Evaluate on ALL domains (including frozen ones)
                eval_losses = {}
                for d_name in active_domains:
                    _, d_v = domain_datasets[d_name]
                    eval_losses[d_name] = evaluate(strat_model, d_v, batch_size)

                # Compute degradation relative to baseline
                degradations = {}
                for d_name in frozen_domains:
                    if d_name in base_losses:
                        deg = 100 * (eval_losses[d_name] - base_losses[d_name]) / base_losses[d_name]
                        degradations[d_name] = deg

                # Print results
                for d_name in active_domains:
                    deg_str = ""
                    if d_name in degradations:
                        deg_str = f" (degradation: {degradations[d_name]:+.2f}%)"
                    print(f"      {d_name}: {eval_losses[d_name]:.4f}{deg_str}")

                # Record cumulative degradation (max over frozen domains)
                max_deg = max(degradations.values()) if degradations else 0.0
                mean_deg = (sum(degradations.values()) / len(degradations)
                            if degradations else 0.0)

                strat_result = {
                    "eval_losses": eval_losses,
                    "degradations": degradations,
                    "max_degradation": max_deg,
                    "mean_degradation": mean_deg,
                    "cal_params": n_cal,
                    "cal_time": cal_time,
                    "graft_time": graft_time,
                    "total_time": graft_time + cal_time,
                }
                graft_result["strategies"][strat] = strat_result

                # Save the model state for next graft step
                # Unfreeze everything to save full state
                for layer in strat_model.layers:
                    for gate in layer.tree.gates:
                        gate.unfreeze()
                    for leaf in layer.tree.leaves:
                        leaf.unfreeze()
                strat_model.wte.unfreeze()
                strat_model.wpe.unfreeze()
                strat_model.norm0.unfreeze()
                strat_model.lm_head.unfreeze()
                for layer in strat_model.layers:
                    layer.norm1.unfreeze()
                    layer.attn.unfreeze()
                    layer.norm2.unfreeze()

                strategy_weights[strat] = {
                    k: mx.array(v) for k, v
                    in nn.utils.tree_flatten(strat_model.parameters())
                }

                # Update base_losses with new domain for future reference
                if new_domain_name not in base_losses:
                    base_losses[new_domain_name] = eval_losses[new_domain_name]

            seed_results["grafts"].append(graft_result)

        all_results["per_seed"].append(seed_results)

    # ── Summary ──────────────────────────────────────────────────────────
    print_summary(all_results, seeds)
    return all_results


def print_summary(all_results, seeds):
    """Print comprehensive summary across all seeds."""
    print(f"\n{'='*70}")
    print(f"SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*70}")

    for strat in ["all-unfrozen", "selective"]:
        print(f"\n  Strategy: {strat}")
        print(f"  {'Graft':<20} | {'N':>3} | {'Mean Max Deg':>12} | {'Mean Deg':>10} | {'Cal Params':>10} | {'Cal Time':>8}")
        print(f"  {'-'*20}-+-{'-'*3}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

        graft_degradations = []  # Track for linearity check
        cal_params_list = []

        for graft_idx in range(3):
            max_degs = []
            mean_degs = []
            cal_params = []
            cal_times = []

            for seed_result in all_results["per_seed"]:
                graft = seed_result["grafts"][graft_idx]
                s = graft["strategies"][strat]
                max_degs.append(s["max_degradation"])
                mean_degs.append(s["mean_degradation"])
                cal_params.append(s["cal_params"])
                cal_times.append(s["cal_time"])

            avg_max_deg = sum(max_degs) / len(max_degs)
            avg_mean_deg = sum(mean_degs) / len(mean_degs)
            avg_cal_params = sum(cal_params) / len(cal_params)
            avg_cal_time = sum(cal_times) / len(cal_times)

            graft_degradations.append(avg_max_deg)
            cal_params_list.append(avg_cal_params)

            graft_name = all_results["per_seed"][0]["grafts"][graft_idx]["name"]
            n_domains = graft_idx + 2

            print(f"  {graft_name:<20} | {n_domains:>3} | {avg_max_deg:>+10.2f}% | "
                  f"{avg_mean_deg:>+8.2f}% | {avg_cal_params:>10.0f} | {avg_cal_time:>6.1f}s")

        # ── Kill criteria assessment ────────────────────────────────────
        print(f"\n  Kill Criteria Assessment ({strat}):")

        # KC1: Cumulative degradation ratio N=4 vs N=2
        if len(graft_degradations) >= 3 and abs(graft_degradations[0]) > 0.01:
            deg_ratio = abs(graft_degradations[2]) / abs(graft_degradations[0])
            kc1_pass = deg_ratio <= 2.0
            print(f"    KC1: |deg(N=4)| / |deg(N=2)| = {deg_ratio:.2f} "
                  f"(threshold: 2.0) -> {'PASS' if kc1_pass else 'KILL'}")
        else:
            print(f"    KC1: Cannot compute (N=2 degradation too small)")

        # KC2: Calibration cost superlinearity
        # Check if cal_params grow faster than linearly with N
        # Linear would be: cal_params[i] / (i+2) is constant
        # Superlinear: the ratio increases
        if len(cal_params_list) >= 3:
            ratios = [cal_params_list[i] / (i + 2) for i in range(3)]
            # If ratio at N=4 > 1.5 * ratio at N=2, it's superlinear
            if ratios[0] > 0:
                cost_ratio = ratios[2] / ratios[0]
                kc2_pass = cost_ratio <= 1.5  # Allow 50% growth in per-N cost
                print(f"    KC2: cal_cost_per_N ratio (N=4)/(N=2) = {cost_ratio:.2f} "
                      f"(threshold: 1.5) -> {'PASS' if kc2_pass else 'KILL'}")
            else:
                print(f"    KC2: Cannot compute (N=2 cost is zero)")

    # Per-seed detail
    print(f"\n  Per-Seed Detail (all-unfrozen):")
    for seed_result in all_results["per_seed"]:
        print(f"\n    Seed {seed_result['seed']}:")
        for graft in seed_result["grafts"]:
            s = graft["strategies"]["all-unfrozen"]
            degs = ", ".join(f"{d}={v:+.2f}%" for d, v in s["degradations"].items())
            print(f"      {graft['name']}: max_deg={s['max_degradation']:+.2f}% "
                  f"[{degs}]")


if __name__ == "__main__":
    t0 = time.time()
    results = run_sequential_experiment()
    total = time.time() - t0
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f} min)")
