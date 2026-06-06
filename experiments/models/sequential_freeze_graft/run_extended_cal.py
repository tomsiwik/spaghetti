"""Extended calibration experiment: can more calibration steps rescue N=4?

The main experiment showed cumulative degradation growing >2x (3.65x) between
N=2 and N=4. But calibration was fixed at 200 steps. Maybe more calibration
at later grafts can compensate for the accumulated routing drift.

Test: scale calibration steps with N:
  - N=2: 200 steps (baseline)
  - N=3: 300 steps (1.5x)
  - N=4: 400 steps (2x)
"""

import sys
import time
import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

from experiments.data import (
    load_names, CharTokenizer, CharDataset, domain_split, train_val_split,
)
from experiments.train import ntp_loss, evaluate
from experiments.models import get_model
from experiments.models.sequential_freeze_graft.run_experiment import (
    count_params, train_model, freeze_components, reinit_components,
    freeze_shared_params, freeze_all_tree, unfreeze_components,
    GRAFT_STEPS,
)


def run_extended_calibration(seeds=(42, 123, 777)):
    """Test whether scaling calibration budget with N helps."""
    print("=" * 70)
    print("EXTENDED CALIBRATION: Scale cal steps with N")
    print("=" * 70)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vs = tokenizer.vocab_size
    splits = domain_split(docs, method="quaternary")
    domain_names = list(splits.keys())

    tree_cfg = dict(vocab_size=vs, block_size=32, tree_depth=3,
                    n_capsules_per_leaf=32, beam_width=2)

    # Calibration schedules to test
    cal_schedules = {
        "fixed_200":   [200, 200, 200],
        "scaled_1.5x": [200, 300, 400],
        "scaled_2x":   [200, 400, 600],
    }

    all_results = {}
    for sched_name in cal_schedules:
        all_results[sched_name] = {"per_seed": []}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        domain_datasets = {}
        for d_name, d_docs in splits.items():
            d_train, d_val = train_val_split(d_docs, seed=seed)
            domain_datasets[d_name] = (
                CharDataset(d_train, tokenizer, 32),
                CharDataset(d_val, tokenizer, 32),
            )
        all_train_docs, all_val_docs = train_val_split(docs, seed=seed)

        # Train base model on domain A
        d_a_name = domain_names[0]
        d_a_train, d_a_val = domain_datasets[d_a_name]
        print(f"\n  [BASE] Training on {d_a_name} (400 steps)")

        mx.random.seed(seed)
        base_model = get_model("sequential_freeze_graft", **tree_cfg)
        mx.eval(base_model.parameters())
        _ = train_model(base_model, d_a_train, d_a_val, steps=400,
                        seed=seed, log_every=400)

        base_losses = {}
        base_losses[d_a_name] = evaluate(base_model, d_a_val, 32)
        print(f"    baseline {d_a_name}: {base_losses[d_a_name]:.4f}")

        base_weights = {k: mx.array(v) for k, v
                        in nn.utils.tree_flatten(base_model.parameters())}

        for sched_name, cal_steps_list in cal_schedules.items():
            print(f"\n  Schedule: {sched_name} ({cal_steps_list})")

            sched_weights = dict(base_weights)
            sched_base_losses = dict(base_losses)
            graft_results = []

            for graft_idx, graft in enumerate(GRAFT_STEPS):
                domain_idx = graft_idx + 1
                new_domain_name = domain_names[domain_idx]
                d_train, d_val = domain_datasets[new_domain_name]
                frozen_domains = domain_names[:domain_idx]
                active_domains = domain_names[:domain_idx + 1]
                cal_steps = cal_steps_list[graft_idx]

                mx.random.seed(seed + graft_idx * 1000)
                model = get_model("sequential_freeze_graft", **tree_cfg)
                model.load_weights(list(sched_weights.items()))
                mx.eval(model.parameters())

                freeze_components(model, graft["freeze_leaves"], graft["freeze_gates"])
                reinit_components(model, graft["reinit_leaves"], graft["reinit_gates"],
                                  seed=seed + graft_idx * 100)
                freeze_shared_params(model)

                _ = train_model(model, d_train, d_val, steps=200,
                                seed=seed + graft_idx, log_every=200)

                # Calibration
                freeze_all_tree(model)
                unfreeze_components(model, graft["cal_all_leaves"], graft["cal_all_gates"])

                mixed_docs = []
                for d_name in active_domains:
                    d_t, _ = domain_datasets[d_name]
                    mixed_docs.extend(d_t.sequences)
                mixed_train = CharDataset.__new__(CharDataset)
                mixed_train.tokenizer = tokenizer
                mixed_train.block_size = 32
                mixed_train.sequences = mixed_docs

                _ = train_model(model, mixed_train, None, steps=cal_steps,
                                seed=seed + graft_idx * 10, log_every=cal_steps)

                # Evaluate
                eval_losses = {}
                for d_name in active_domains:
                    _, d_v = domain_datasets[d_name]
                    eval_losses[d_name] = evaluate(model, d_v, 32)

                degradations = {}
                for d_name in frozen_domains:
                    if d_name in sched_base_losses:
                        deg = 100 * (eval_losses[d_name] - sched_base_losses[d_name]) / sched_base_losses[d_name]
                        degradations[d_name] = deg

                max_deg = max(degradations.values()) if degradations else 0.0
                print(f"    {graft['name']}: max_deg={max_deg:+.2f}% "
                      f"(cal_steps={cal_steps})")

                graft_results.append({
                    "max_degradation": max_deg,
                    "degradations": degradations,
                    "cal_steps": cal_steps,
                })

                if new_domain_name not in sched_base_losses:
                    sched_base_losses[new_domain_name] = eval_losses[new_domain_name]

                # Save state
                for layer in model.layers:
                    for gate in layer.tree.gates:
                        gate.unfreeze()
                    for leaf in layer.tree.leaves:
                        leaf.unfreeze()
                model.wte.unfreeze()
                model.wpe.unfreeze()
                model.norm0.unfreeze()
                model.lm_head.unfreeze()
                for layer in model.layers:
                    layer.norm1.unfreeze()
                    layer.attn.unfreeze()
                    layer.norm2.unfreeze()

                sched_weights = {k: mx.array(v) for k, v
                                  in nn.utils.tree_flatten(model.parameters())}

            all_results[sched_name]["per_seed"].append(graft_results)

    # Summary
    print(f"\n{'='*70}")
    print(f"EXTENDED CALIBRATION SUMMARY")
    print(f"{'='*70}")
    for sched_name, cal_steps_list in cal_schedules.items():
        print(f"\n  {sched_name} ({cal_steps_list}):")
        for graft_idx in range(3):
            degs = [sr[graft_idx]["max_degradation"]
                    for sr in all_results[sched_name]["per_seed"]]
            mean_deg = sum(degs) / len(degs)
            per_seed = ", ".join(f"{d:+.2f}%" for d in degs)
            n = graft_idx + 2
            print(f"    N={n}: max_deg={mean_deg:+.2f}% ({per_seed})")

        # Ratio
        degs_n2 = [sr[0]["max_degradation"]
                   for sr in all_results[sched_name]["per_seed"]]
        degs_n4 = [sr[2]["max_degradation"]
                   for sr in all_results[sched_name]["per_seed"]]
        mean_n2 = sum(degs_n2) / len(degs_n2)
        mean_n4 = sum(degs_n4) / len(degs_n4)
        if abs(mean_n2) > 0.01:
            ratio = abs(mean_n4) / abs(mean_n2)
            verdict = "PASS" if ratio <= 2.0 else "KILL"
            print(f"    ratio N=4/N=2 = {ratio:.2f} [{verdict}]")

        # Total calibration cost
        total_steps = sum(cal_steps_list)
        print(f"    total cal steps: {total_steps}")


if __name__ == "__main__":
    t0 = time.time()
    run_extended_calibration()
    total = time.time() - t0
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f} min)")
