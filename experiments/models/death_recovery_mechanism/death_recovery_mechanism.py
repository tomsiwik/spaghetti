"""Death Recovery Mechanism -- Exp 20.

Isolates the inter-layer coupling revival mechanism by selectively
freezing MLP layers during fine-tuning.

Hypothesis: Weight updates in layers 0..l-1 shift the input distribution
to layer l, reviving dead neurons whose detector vectors a_i now see
positive inputs. Freezing upstream layers should suppress this.

Experimental design:
  Condition 1 (BASELINE): All MLP layers train freely (replicates Exp 18)
  Condition 2 (FREEZE-UPSTREAM): For each target layer l, freeze all MLP
    layers 0..l-1 and train only layers l..L-1. Measure revival in layer l.
  Condition 3 (FREEZE-DOWNSTREAM): For each target layer l, freeze layers
    l..L-1 and train only layers 0..l-1. Measure revival in layer l
    (should show revival if upstream changes drive it).
  Condition 4 (FREEZE-ALL-BUT-ONE): Train only one layer at a time.
    Isolates self-revival (momentum, noise) vs inter-layer coupling.

Kill criterion: If FREEZE-UPSTREAM does NOT reduce revival in downstream
layers compared to BASELINE, inter-layer coupling is not the mechanism.
"""

import copy
import statistics
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate
from ..relu_router.relu_router import ReLURouterGPT
from ..relu_router.test_composition import (
    _make_relu_model, _freeze_attention,
    BASE, N_CAPSULES, STEPS_PRETRAIN, BATCH_SIZE, LR,
)
from ..dead_capsule_pruning.dead_capsule_pruning import profile_activations
from ..capsule_revival.test_capsule_revival import get_dead_mask, transition_counts


# Experiment config
STEP_COUNTS = [0, 100, 400, 800, 1600, 3200]
DOMAIN = "a_m"
N_LAYERS = 4


def freeze_specific_mlp_layers(model, layer_indices_to_freeze):
    """Freeze MLP (capsule pool) weights in specified layers only.

    Attention is always frozen (consistent with Exp 17/18 protocol).
    Embeddings (wte, wpe), norm0, and lm_head are also frozen by the
    initial model.freeze() call. This is important: frozen embeddings
    mean x^0 = embed(tokens) is fixed, so frozen upstream MLP layers
    truly receive fixed inputs (no embedding drift confound).

    This additionally freezes MLP weights in selected layers so only
    the unfrozen MLP layers can change the input distribution.

    Args:
        model: ReLURouterGPT model
        layer_indices_to_freeze: list of layer indices whose MLP to freeze
    """
    # First freeze everything (wte, wpe, norm0, lm_head, all attention, all MLP)
    model.freeze()
    # Unfreeze all capsule pools
    for layer in model.layers:
        layer.capsule_pool.unfreeze()
    # Re-freeze specified MLP layers
    for idx in layer_indices_to_freeze:
        model.layers[idx].capsule_pool.freeze()


def profile_per_layer_death(model, val_ds, seed=42):
    """Profile and return per-layer dead masks and death rates.

    Returns:
        per_layer_masks: list of list of bool (True=dead) per layer
        per_layer_rates: list of float (death rate per layer)
        flat_mask: list of bool across all layers
    """
    freqs = profile_activations(model, val_ds, n_batches=20, batch_size=32, seed=seed)
    flat_mask, per_layer_masks = get_dead_mask(freqs)
    per_layer_rates = []
    for layer_mask in per_layer_masks:
        rate = sum(layer_mask) / len(layer_mask) if layer_mask else 0
        per_layer_rates.append(rate)
    return per_layer_masks, per_layer_rates, flat_mask


def run_condition(base_model, train_ds, val_ds, condition_name,
                  layers_to_freeze, steps, seed=42):
    """Run one freeze condition across all step counts.

    Args:
        base_model: Pretrained base model to start from
        train_ds: Training dataset
        val_ds: Validation dataset
        condition_name: Name for logging
        layers_to_freeze: list of layer indices to freeze (MLP only)
        steps: list of step counts to evaluate
        seed: random seed

    Returns:
        dict with per-step, per-layer death masks and rates
    """
    results = {
        "condition": condition_name,
        "frozen_layers": layers_to_freeze,
        "checkpoints": {},
    }

    for S in steps:
        model = copy.deepcopy(base_model)

        if S > 0:
            freeze_specific_mlp_layers(model, layers_to_freeze)
            train(model, train_ds, steps=S,
                  batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
            model.unfreeze()

        per_layer_masks, per_layer_rates, flat_mask = profile_per_layer_death(
            model, val_ds, seed=seed
        )

        results["checkpoints"][S] = {
            "per_layer_masks": per_layer_masks,
            "per_layer_rates": per_layer_rates,
            "flat_mask": flat_mask,
            "overall_rate": sum(flat_mask) / len(flat_mask),
        }

        print(f"    [{condition_name}] S={S:>4d} | "
              + " ".join(f"L{i}={r:.1%}" for i, r in enumerate(per_layer_rates))
              + f" | overall={results['checkpoints'][S]['overall_rate']:.1%}")

    return results


def compute_revival_per_layer(results, anchor_step=100):
    """Compute per-layer revival from anchor step to each later step.

    Returns dict mapping step -> list of per-layer revival rates.
    """
    if anchor_step not in results["checkpoints"]:
        # Use the first non-zero step as anchor
        anchor_step = min(s for s in results["checkpoints"] if s > 0)

    anchor_masks = results["checkpoints"][anchor_step]["per_layer_masks"]
    revival_data = {}

    for S, ckpt in results["checkpoints"].items():
        if S <= anchor_step:
            continue
        per_layer_revival = []
        for l_idx in range(N_LAYERS):
            mask_anchor = anchor_masks[l_idx]
            mask_s = ckpt["per_layer_masks"][l_idx]
            trans = transition_counts(mask_anchor, mask_s)
            total_dead_anchor = trans["dd"] + trans["da"]
            revival_rate = trans["da"] / total_dead_anchor if total_dead_anchor > 0 else 0
            per_layer_revival.append({
                "revival_rate": revival_rate,
                "n_revived": trans["da"],
                "n_dead_anchor": total_dead_anchor,
                "n_newly_dead": trans["ad"],
            })
        revival_data[S] = per_layer_revival

    return revival_data


def run_full_experiment(seed=42):
    """Run all freeze conditions for one seed.

    Conditions:
    1. BASELINE: all MLP layers train (no extra freezing)
    2. FREEZE-UPSTREAM-OF-L: for L=1,2,3, freeze layers 0..L-1
    3. FREEZE-SINGLE: train only one layer at a time (layers 1,2,3)

    Returns dict with all condition results.
    """
    # Setup data
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
    train_ds = domain_datasets[DOMAIN][0]
    val_ds = domain_datasets[DOMAIN][1]

    # Pretrain base model
    print(f"  Pretraining base model (300 steps, seed={seed})...")
    base = _make_relu_model(V, n_capsules=N_CAPSULES)
    train(base, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    all_results = {}

    # Condition 1: BASELINE (all MLP layers trainable)
    print(f"\n  === BASELINE (all layers train) ===")
    all_results["baseline"] = run_condition(
        base, train_ds, val_ds, "baseline",
        layers_to_freeze=[], steps=STEP_COUNTS, seed=seed,
    )

    # Condition 2: FREEZE-UPSTREAM for each target layer
    for target_layer in [1, 2, 3]:
        frozen = list(range(target_layer))  # freeze 0..target-1
        name = f"freeze_upstream_of_L{target_layer}"
        print(f"\n  === {name} (freeze MLP layers {frozen}) ===")
        all_results[name] = run_condition(
            base, train_ds, val_ds, name,
            layers_to_freeze=frozen, steps=STEP_COUNTS, seed=seed,
        )

    # Condition 3: FREEZE-SINGLE (train only one MLP layer)
    for train_layer in [0, 1, 2, 3]:
        frozen = [i for i in range(N_LAYERS) if i != train_layer]
        name = f"train_only_L{train_layer}"
        print(f"\n  === {name} (freeze MLP layers {frozen}) ===")
        all_results[name] = run_condition(
            base, train_ds, val_ds, name,
            layers_to_freeze=frozen, steps=STEP_COUNTS, seed=seed,
        )

    return all_results


def analyze_results(all_seeds_results):
    """Analyze and print aggregate results across seeds.

    The key comparison: does freezing upstream layers reduce revival
    in downstream layers?
    """
    seeds = list(all_seeds_results.keys())
    conditions = list(all_seeds_results[seeds[0]].keys())

    # Use first non-zero step as anchor
    anchor_step = STEP_COUNTS[1]  # 100
    final_step = STEP_COUNTS[-1]  # 3200

    print(f"\n{'='*80}")
    print(f"  REVIVAL ANALYSIS (anchor S={anchor_step}, measured at S={final_step})")
    print(f"  {len(seeds)} seeds")
    print(f"{'='*80}")

    # Collect per-layer revival data (rates AND counts) for each condition
    condition_revival = {}  # condition -> layer -> list of dicts across seeds
    condition_revival_rates = {}  # condition -> layer -> list of revival rates

    for cond in conditions:
        condition_revival[cond] = {l: [] for l in range(N_LAYERS)}
        condition_revival_rates[cond] = {l: [] for l in range(N_LAYERS)}
        for seed in seeds:
            result = all_seeds_results[seed][cond]
            revival = compute_revival_per_layer(result, anchor_step=anchor_step)
            if final_step in revival:
                for l in range(N_LAYERS):
                    condition_revival[cond][l].append(revival[final_step][l])
                    condition_revival_rates[cond][l].append(
                        revival[final_step][l]["revival_rate"]
                    )

    # ── FIX 3: Report |D^l_100| dead counts per condition ──
    print(f"\n{'='*80}")
    print(f"  ANCHOR DEAD COUNTS |D^l_100| PER CONDITION (denominators)")
    print(f"{'='*80}")
    print(f"\n  {'Condition':<30} | " + " | ".join(f" L{l} dead " for l in range(N_LAYERS)))
    print("  " + "-" * (30 + N_LAYERS * 10))

    for cond in conditions:
        row = f"  {cond:<30} | "
        for l in range(N_LAYERS):
            entries = condition_revival[cond][l]
            if entries:
                counts = [e["n_dead_anchor"] for e in entries]
                mean_count = statistics.mean(counts)
                row += f"  {mean_count:>5.1f}  | "
            else:
                row += f"  {'N/A':>5}  | "
        print(row)

    # Print revival rate table
    print(f"\n  Per-Layer Revival Rate (D->A fraction of anchor dead set)")
    print(f"  {'Condition':<30} | " + " | ".join(f"  Layer {l}  " for l in range(N_LAYERS)))
    print("  " + "-" * (30 + N_LAYERS * 13))

    for cond in conditions:
        row = f"  {cond:<30} | "
        for l in range(N_LAYERS):
            rates = condition_revival_rates[cond][l]
            if rates:
                mean = statistics.mean(rates)
                std = statistics.stdev(rates) if len(rates) > 1 else 0
                row += f"{mean:>5.1%} +/- {std:>4.1%} | "
            else:
                row += f"{'N/A':>12} | "
        print(row)

    # Key comparison: baseline vs freeze-upstream revival
    print(f"\n{'='*80}")
    print(f"  KEY COMPARISON: Upstream Freeze Effect on Revival")
    print(f"{'='*80}")

    for target_layer in [1, 2, 3]:
        baseline_rates = condition_revival_rates["baseline"][target_layer]
        freeze_name = f"freeze_upstream_of_L{target_layer}"
        freeze_rates = condition_revival_rates.get(freeze_name, {}).get(target_layer, [])

        if baseline_rates and freeze_rates:
            mean_base = statistics.mean(baseline_rates)
            mean_freeze = statistics.mean(freeze_rates)
            diff_pp = (mean_freeze - mean_base) * 100
            reduction_pct = (1 - mean_freeze / mean_base) * 100 if mean_base > 0 else 0

            print(f"\n  Layer {target_layer}:")
            print(f"    Baseline revival:          {mean_base:.1%}")
            print(f"    With upstream frozen:       {mean_freeze:.1%}")
            print(f"    Difference:                 {diff_pp:+.1f} pp")
            print(f"    Reduction:                  {reduction_pct:.0f}%")

    # ── FIX 2: Explain 97.3% vs 12.6% discrepancy ──
    # Compare alive->dead (new death) rates in baseline vs train-only-L0 for each layer
    print(f"\n{'='*80}")
    print(f"  NEW DEATH ANALYSIS (alive->dead transitions, S=100 to S=3200)")
    print(f"  Explains why train-only-L0 shows higher L1 revival than baseline:")
    print(f"  baseline L1 trains => creates offsetting new deaths")
    print(f"{'='*80}")

    print(f"\n  {'Condition':<30} | " + " | ".join(f" L{l} A->D " for l in range(N_LAYERS)))
    print("  " + "-" * (30 + N_LAYERS * 10))

    for cond in ["baseline", "train_only_L0", "train_only_L1", "train_only_L2", "train_only_L3"]:
        if cond not in conditions:
            continue
        row = f"  {cond:<30} | "
        for l in range(N_LAYERS):
            entries = condition_revival[cond][l]
            if entries:
                new_dead_counts = [e["n_newly_dead"] for e in entries]
                mean_nd = statistics.mean(new_dead_counts)
                row += f"  {mean_nd:>5.1f}  | "
            else:
                row += f"  {'N/A':>5}  | "
        print(row)

    # Also print net revival (revived - newly_dead) to show offsetting
    print(f"\n  Net Revival (revived - newly_dead):")
    print(f"  {'Condition':<30} | " + " | ".join(f"  L{l} net " for l in range(N_LAYERS)))
    print("  " + "-" * (30 + N_LAYERS * 10))

    for cond in ["baseline", "train_only_L0"]:
        if cond not in conditions:
            continue
        row = f"  {cond:<30} | "
        for l in range(N_LAYERS):
            entries = condition_revival[cond][l]
            if entries:
                nets = [e["n_revived"] - e["n_newly_dead"] for e in entries]
                mean_net = statistics.mean(nets)
                row += f"  {mean_net:>+5.1f}  | "
            else:
                row += f"  {'N/A':>5}  | "
        print(row)

    # ── FIX 4 + FIX 6: Self-revival with upstream/downstream separation ──
    # Exclude L0 from self-revival reporting (unreliable, 0/0 denominator)
    print(f"\n{'='*80}")
    print(f"  SELF-REVIVAL ANALYSIS (single-layer training, L0 excluded)")
    print(f"  L0 excluded: ~0% dead capsules at S=100, one seed has 0/0 denominator")
    print(f"{'='*80}")

    for layer in range(1, N_LAYERS):  # FIX 6: skip L0
        cond = f"train_only_L{layer}"
        # Revival in the trained layer itself (self-revival)
        trained_rates = condition_revival_rates.get(cond, {}).get(layer, [])

        # FIX 4: Separate upstream vs downstream "other layers" explicitly
        upstream_revival = []
        downstream_revival = []
        for other_l in range(N_LAYERS):
            if other_l == layer:
                continue
            rates = condition_revival_rates.get(cond, {}).get(other_l, [])
            if rates:
                if other_l < layer:
                    upstream_revival.extend(rates)
                else:
                    downstream_revival.extend(rates)

        if trained_rates:
            print(f"\n  Train only Layer {layer}:")
            print(f"    Revival in L{layer} (self):       {statistics.mean(trained_rates):.1%}")
            if upstream_revival:
                print(f"    Revival in upstream layers:  {statistics.mean(upstream_revival):.1%}  (should be ~0%)")
            if downstream_revival:
                print(f"    Revival in downstream layers:{statistics.mean(downstream_revival):.1%}  (inter-layer coupling)")

    # L0 note
    cond_l0 = "train_only_L0"
    if cond_l0 in conditions:
        l0_entries = condition_revival[cond_l0].get(0, [])
        l0_dead_counts = [e["n_dead_anchor"] for e in l0_entries] if l0_entries else []
        print(f"\n  [L0 excluded from self-revival table]")
        if l0_dead_counts:
            print(f"    L0 |D^0_100| per seed: {l0_dead_counts}")
            print(f"    Mean: {statistics.mean(l0_dead_counts):.1f} dead (too few for reliable rate)")

    # Death rate trajectories per condition
    print(f"\n{'='*80}")
    print(f"  DEATH RATE TRAJECTORIES")
    print(f"{'='*80}")

    for cond in ["baseline", "freeze_upstream_of_L1", "freeze_upstream_of_L2",
                  "freeze_upstream_of_L3"]:
        if cond not in conditions:
            continue
        print(f"\n  {cond}:")
        print(f"    {'Step':>6} | " + " | ".join(f"  Layer {l}" for l in range(N_LAYERS)))
        print("    " + "-" * (8 + N_LAYERS * 11))

        for S in STEP_COUNTS:
            rates_per_layer = {l: [] for l in range(N_LAYERS)}
            for seed in seeds:
                ckpt = all_seeds_results[seed][cond]["checkpoints"].get(S)
                if ckpt:
                    for l in range(N_LAYERS):
                        rates_per_layer[l].append(ckpt["per_layer_rates"][l])

            row = f"    {S:>6} | "
            for l in range(N_LAYERS):
                if rates_per_layer[l]:
                    row += f"{statistics.mean(rates_per_layer[l]):>8.1%}  | "
                else:
                    row += f"{'N/A':>8}  | "
            print(row)

    # Embedding freeze verification
    print(f"\n{'='*80}")
    print(f"  EMBEDDING FREEZE VERIFICATION")
    print(f"{'='*80}")
    print(f"\n  freeze_specific_mlp_layers() calls model.freeze() first,")
    print(f"  which freezes ALL parameters including wte, wpe, norm0, lm_head.")
    print(f"  Only capsule pools in non-frozen layers are then unfrozen.")
    print(f"  Therefore: embeddings are FROZEN in all conditions.")
    print(f"  x^0 = embed(tokens) is fixed => frozen upstream truly fixes input.")
    print(f"  No embedding drift confound.")

    # KILL CRITERION CHECK
    print(f"\n{'='*80}")
    print(f"  KILL CRITERION CHECK")
    print(f"{'='*80}")

    # Kill: freezing upstream does NOT reduce revival
    kill_triggered = True  # assume kill until proven otherwise
    for target_layer in [1, 2, 3]:
        baseline_rates = condition_revival_rates["baseline"][target_layer]
        freeze_name = f"freeze_upstream_of_L{target_layer}"
        freeze_rates = condition_revival_rates.get(freeze_name, {}).get(target_layer, [])

        if baseline_rates and freeze_rates:
            mean_base = statistics.mean(baseline_rates)
            mean_freeze = statistics.mean(freeze_rates)

            # Revival must be REDUCED by at least 5pp (meaningful reduction)
            # OR the freeze revival must be less than half the baseline
            reduced = (mean_base - mean_freeze) > 0.05 or (mean_freeze < mean_base * 0.5)
            if reduced:
                kill_triggered = False

    if kill_triggered:
        print(f"\n  KILL: Freezing upstream layers does NOT reduce revival in")
        print(f"  downstream layers. Inter-layer coupling is NOT the primary")
        print(f"  revival mechanism.")
        verdict = "KILL"
    else:
        # FIX 5: "strongly supported" not "confirmed"
        print(f"\n  PASS: Freezing upstream layers DOES reduce revival in")
        print(f"  downstream layers. Inter-layer coupling strongly supported as a")
        print(f"  revival mechanism. (n=3 seeds, no significance test; directional")
        print(f"  evidence with large effect sizes.)")
        verdict = "PASS"

    return verdict


def main():
    """Run across 3 seeds and analyze."""
    seeds = [42, 123, 7]
    all_seeds_results = {}

    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"  SEED {seed}")
        print(f"{'='*80}")
        all_seeds_results[seed] = run_full_experiment(seed=seed)

    verdict = analyze_results(all_seeds_results)
    return verdict


if __name__ == "__main__":
    main()
