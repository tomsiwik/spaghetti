"""Revival Dynamics Under Composition -- exp_revival_under_composition.

Exp 18 showed 28.1% capsule revival under single-domain fine-tuning.
Exp 20 showed inter-layer coupling drives 79-94% of revival.
Exp 16 showed Jaccard=0.895 for dead set identity between single and composed.

This experiment asks: when capsules live inside a COMPOSED model and see
cross-domain inputs during continued training (calibration), does revival
change compared to single-domain training?

Two mechanisms could alter revival under composition:
  (a) SUPPRESSION: cross-domain gradients cancel out, reducing the net
      input distribution shift that drives inter-layer coupling revival.
  (b) AMPLIFICATION: cross-domain inputs provide diverse activation
      patterns that push dead capsules across the boundary more often.

Experimental design:
  CONDITION A (SINGLE-DOMAIN): Fine-tune on one domain only, track
    per-capsule revival from S=200 to S=3200 (replicates Exp 18).
  CONDITION B (COMPOSED-JOINT): Compose 2-domain model, then continue
    training on joint data, track per-capsule revival in each domain's
    capsule half from the same anchor checkpoint.
  CONDITION C (COMPOSED-OWN): Compose 2-domain model, then continue
    training on each domain's own data only (no cross-domain), track
    per-capsule revival. This isolates composition structure from
    cross-domain input effects.

Kill criterion: |revival_composed - revival_single| < 5 pp.
"""

import copy
import statistics
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate
from .. import register
from ..relu_router.relu_router import ReLURouterGPT
from ..relu_router.test_composition import (
    compose_relu_models,
    _make_relu_model, _freeze_attention,
    BASE, N_CAPSULES, STEPS_PRETRAIN, STEPS_FINETUNE,
    BATCH_SIZE, LR,
)
from ..dead_capsule_pruning.dead_capsule_pruning import profile_activations
from ..capsule_revival.test_capsule_revival import get_dead_mask, transition_counts


@register("revival_under_composition", parent="capsule_revival")
class RevivalUnderCompositionGPT(ReLURouterGPT):
    """ReLURouterGPT for revival-under-composition experiments.

    Thin wrapper for lineage tracking. No architectural changes.
    """
    pass


# Training checkpoints for revival tracking
# We start from a pre-composed anchor and train further
ANCHOR_STEP = 200   # Fine-tune steps before composition
POST_COMPOSE_STEPS = [0, 100, 400, 800, 1600, 3200]
N_LAYERS = 4


def profile_per_capsule_death(model, dataset, n_capsules_per_layer, seed=42):
    """Profile per-capsule death, returning per-layer dead masks.

    Args:
        model: ReLURouterGPT (possibly composed with 2*N_CAPSULES)
        dataset: validation dataset for profiling
        n_capsules_per_layer: expected capsules per layer
        seed: profiling seed

    Returns:
        flat_mask: list of bool (True=dead), length = n_layers * n_capsules_per_layer
        per_layer_masks: list of list of bool, per layer
        per_layer_rates: list of float, death rate per layer
    """
    freqs = profile_activations(model, dataset, n_batches=20, batch_size=32, seed=seed)
    flat_mask, per_layer_masks = get_dead_mask(freqs)

    per_layer_rates = []
    for layer_mask in per_layer_masks:
        rate = sum(layer_mask) / len(layer_mask) if layer_mask else 0
        per_layer_rates.append(rate)

    return flat_mask, per_layer_masks, per_layer_rates


def split_composed_mask_by_domain(flat_mask, n_capsules_per_domain, n_layers):
    """Split a composed model's flat dead mask into per-domain halves.

    In a composed model with 2*P capsules per layer, the first P belong
    to domain A and the second P to domain B.

    Returns:
        domain_a_mask: flat list of bool for domain A's capsules only
        domain_b_mask: flat list of bool for domain B's capsules only
    """
    P = n_capsules_per_domain
    domain_a_mask = []
    domain_b_mask = []

    for l in range(n_layers):
        start = l * (2 * P)
        layer_mask = flat_mask[start:start + 2 * P]
        domain_a_mask.extend(layer_mask[:P])
        domain_b_mask.extend(layer_mask[P:])

    return domain_a_mask, domain_b_mask


def run_single_domain_revival(base_model, train_ds, val_ds,
                              anchor_step, post_steps, seed=42):
    """Condition A: Single-domain fine-tuning revival tracking.

    Replicates Exp 18 protocol: fine-tune from base, profile at anchor,
    then continue training and track revival.

    Returns dict with anchor mask and per-step masks.
    """
    results = {"condition": "single_domain", "checkpoints": {}}

    # Fine-tune to anchor
    model_anchor = copy.deepcopy(base_model)
    _freeze_attention(model_anchor)
    train(model_anchor, train_ds, steps=anchor_step,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    model_anchor.unfreeze()

    # Profile at anchor
    flat_mask, per_layer, rates = profile_per_capsule_death(
        model_anchor, val_ds, N_CAPSULES, seed=seed)
    results["anchor_mask"] = flat_mask
    results["anchor_rates"] = rates
    results["checkpoints"][0] = {
        "flat_mask": flat_mask, "per_layer_rates": rates,
        "overall_rate": sum(flat_mask) / len(flat_mask),
    }

    print(f"    Anchor S={anchor_step}: death={sum(flat_mask)/len(flat_mask):.1%}")

    # Continue training from anchor for each post_step count
    for S in post_steps:
        if S == 0:
            continue
        model_s = copy.deepcopy(model_anchor)
        _freeze_attention(model_s)
        train(model_s, train_ds, steps=S,
              batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
        model_s.unfreeze()

        flat_mask_s, per_layer_s, rates_s = profile_per_capsule_death(
            model_s, val_ds, N_CAPSULES, seed=seed)
        results["checkpoints"][S] = {
            "flat_mask": flat_mask_s, "per_layer_rates": rates_s,
            "overall_rate": sum(flat_mask_s) / len(flat_mask_s),
        }
        print(f"    S={anchor_step}+{S}: death={sum(flat_mask_s)/len(flat_mask_s):.1%}")

    return results


def run_composed_revival(base_model, domain_models, domain_datasets,
                         domain_names, train_ds_joint, val_ds_joint,
                         post_steps, seed=42, train_on="joint"):
    """Condition B/C: Composed model revival tracking.

    Compose domain models, profile at anchor (S=0 post-compose),
    then continue training on joint or own-domain data and track revival.

    Args:
        train_on: "joint" for condition B, or domain name for condition C
    """
    condition_name = f"composed_{train_on}"
    results = {"condition": condition_name, "checkpoints": {}}

    # Compose
    composed = compose_relu_models(
        base_model, [domain_models[d] for d in domain_names])

    # Profile at anchor (just-composed, no further training)
    flat_mask, per_layer, rates = profile_per_capsule_death(
        composed, val_ds_joint, 2 * N_CAPSULES, seed=seed)

    # Split into domain halves
    mask_a, mask_b = split_composed_mask_by_domain(flat_mask, N_CAPSULES, N_LAYERS)
    results["anchor_mask_full"] = flat_mask
    results["anchor_mask_a"] = mask_a
    results["anchor_mask_b"] = mask_b
    results["anchor_rates"] = rates
    results["checkpoints"][0] = {
        "flat_mask": flat_mask,
        "mask_a": mask_a, "mask_b": mask_b,
        "per_layer_rates": rates,
        "overall_rate": sum(flat_mask) / len(flat_mask),
        "rate_a": sum(mask_a) / len(mask_a),
        "rate_b": sum(mask_b) / len(mask_b),
    }

    print(f"    Anchor (composed): death={sum(flat_mask)/len(flat_mask):.1%} "
          f"(A={sum(mask_a)/len(mask_a):.1%}, B={sum(mask_b)/len(mask_b):.1%})")

    # Select training dataset
    if train_on == "joint":
        train_ds = train_ds_joint
    else:
        train_ds = domain_datasets[train_on][0]

    # Continue training from composed anchor
    for S in post_steps:
        if S == 0:
            continue
        model_s = copy.deepcopy(composed)
        _freeze_attention(model_s)
        train(model_s, train_ds, steps=S,
              batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
        model_s.unfreeze()

        flat_mask_s, per_layer_s, rates_s = profile_per_capsule_death(
            model_s, val_ds_joint, 2 * N_CAPSULES, seed=seed)
        mask_a_s, mask_b_s = split_composed_mask_by_domain(
            flat_mask_s, N_CAPSULES, N_LAYERS)

        results["checkpoints"][S] = {
            "flat_mask": flat_mask_s,
            "mask_a": mask_a_s, "mask_b": mask_b_s,
            "per_layer_rates": rates_s,
            "overall_rate": sum(flat_mask_s) / len(flat_mask_s),
            "rate_a": sum(mask_a_s) / len(mask_a_s),
            "rate_b": sum(mask_b_s) / len(mask_b_s),
        }
        print(f"    S=+{S}: death={sum(flat_mask_s)/len(flat_mask_s):.1%} "
              f"(A={sum(mask_a_s)/len(mask_a_s):.1%}, B={sum(mask_b_s)/len(mask_b_s):.1%})")

    return results


def compute_revival_rates(results, is_composed=False):
    """Compute revival rates from anchor to each later checkpoint.

    For single-domain: uses flat_mask directly.
    For composed: computes revival for each domain half separately.

    Returns dict mapping step -> revival info.
    """
    anchor_step = 0  # anchor is always the first checkpoint
    if is_composed:
        anchor_a = results["anchor_mask_a"]
        anchor_b = results["anchor_mask_b"]
    else:
        anchor_mask = results["anchor_mask"]

    revival_data = {}

    for S, ckpt in results["checkpoints"].items():
        if S == 0:
            continue

        if is_composed:
            # Per-domain revival
            trans_a = transition_counts(anchor_a, ckpt["mask_a"])
            trans_b = transition_counts(anchor_b, ckpt["mask_b"])

            dead_anchor_a = trans_a["dd"] + trans_a["da"]
            dead_anchor_b = trans_b["dd"] + trans_b["da"]
            revival_a = trans_a["da"] / dead_anchor_a if dead_anchor_a > 0 else 0
            revival_b = trans_b["da"] / dead_anchor_b if dead_anchor_b > 0 else 0

            # Combined (full model)
            anchor_full = results["anchor_mask_full"]
            trans_full = transition_counts(anchor_full, ckpt["flat_mask"])
            dead_anchor_full = trans_full["dd"] + trans_full["da"]
            revival_full = trans_full["da"] / dead_anchor_full if dead_anchor_full > 0 else 0

            revival_data[S] = {
                "revival_a": revival_a,
                "revival_b": revival_b,
                "revival_full": revival_full,
                "n_revived_a": trans_a["da"],
                "n_revived_b": trans_b["da"],
                "n_revived_full": trans_full["da"],
                "n_dead_anchor_a": dead_anchor_a,
                "n_dead_anchor_b": dead_anchor_b,
                "n_dead_anchor_full": dead_anchor_full,
                "n_newly_dead_a": trans_a["ad"],
                "n_newly_dead_b": trans_b["ad"],
                "n_newly_dead_full": trans_full["ad"],
            }
        else:
            trans = transition_counts(anchor_mask, ckpt["flat_mask"])
            dead_anchor = trans["dd"] + trans["da"]
            revival_rate = trans["da"] / dead_anchor if dead_anchor > 0 else 0

            revival_data[S] = {
                "revival_rate": revival_rate,
                "n_revived": trans["da"],
                "n_dead_anchor": dead_anchor,
                "n_newly_dead": trans["ad"],
            }

    return revival_data


def run_full_experiment(seed=42):
    """Run all conditions for one seed.

    Returns dict with results for each condition.
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

    all_docs_train, all_docs_val = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_docs_train, tokenizer, BASE["block_size"])
    joint_val = CharDataset(all_docs_val, tokenizer, BASE["block_size"])

    V = tokenizer.vocab_size
    domain_names = sorted(domain_datasets.keys())

    # Pretrain base model
    print(f"  Pretraining base model (300 steps, seed={seed})...")
    base = _make_relu_model(V, n_capsules=N_CAPSULES)
    train(base, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    # Fine-tune per domain to ANCHOR_STEP
    print(f"  Fine-tuning per domain ({ANCHOR_STEP} steps)...")
    domain_models = {}
    for d_name in domain_names:
        model_d = copy.deepcopy(base)
        _freeze_attention(model_d)
        train(model_d, domain_datasets[d_name][0], steps=ANCHOR_STEP,
              batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
        model_d.unfreeze()
        domain_models[d_name] = model_d

    all_results = {}

    # Condition A: Single-domain revival (both domains)
    for d_name in domain_names:
        print(f"\n  === SINGLE-DOMAIN ({d_name}) ===")
        key = f"single_{d_name}"
        all_results[key] = run_single_domain_revival(
            base, domain_datasets[d_name][0], domain_datasets[d_name][1],
            anchor_step=ANCHOR_STEP, post_steps=POST_COMPOSE_STEPS, seed=seed,
        )

    # Condition B: Composed + joint training
    print(f"\n  === COMPOSED + JOINT TRAINING ===")
    all_results["composed_joint"] = run_composed_revival(
        base, domain_models, domain_datasets, domain_names,
        joint_train, joint_val,
        post_steps=POST_COMPOSE_STEPS, seed=seed, train_on="joint",
    )

    # Condition C: Composed + own-domain training (first domain only for efficiency)
    print(f"\n  === COMPOSED + OWN-DOMAIN TRAINING ({domain_names[0]}) ===")
    all_results[f"composed_{domain_names[0]}"] = run_composed_revival(
        base, domain_models, domain_datasets, domain_names,
        joint_train, joint_val,
        post_steps=POST_COMPOSE_STEPS, seed=seed, train_on=domain_names[0],
    )

    return all_results


def analyze_results(all_seeds_results):
    """Analyze and print aggregate results across seeds.

    The key comparison: does composed revival rate differ from single-domain
    revival rate by >= 5pp?
    """
    seeds = list(all_seeds_results.keys())

    print(f"\n{'='*80}")
    print(f"  REVIVAL UNDER COMPOSITION ANALYSIS")
    print(f"  {len(seeds)} seeds, anchor at S={ANCHOR_STEP}")
    print(f"{'='*80}")

    # Collect revival rates at each checkpoint for each condition
    final_step = POST_COMPOSE_STEPS[-1]

    # Single-domain revival rates (average both domains)
    single_revival_rates = {S: [] for S in POST_COMPOSE_STEPS if S > 0}
    for seed in seeds:
        for key in all_seeds_results[seed]:
            if not key.startswith("single_"):
                continue
            revival = compute_revival_rates(all_seeds_results[seed][key], is_composed=False)
            for S, data in revival.items():
                single_revival_rates[S].append(data["revival_rate"])

    # Composed-joint revival rates (full model)
    composed_joint_revival = {S: [] for S in POST_COMPOSE_STEPS if S > 0}
    # Per-domain revival in composed model
    composed_revival_a = {S: [] for S in POST_COMPOSE_STEPS if S > 0}
    composed_revival_b = {S: [] for S in POST_COMPOSE_STEPS if S > 0}
    for seed in seeds:
        revival = compute_revival_rates(
            all_seeds_results[seed]["composed_joint"], is_composed=True)
        for S, data in revival.items():
            composed_joint_revival[S].append(data["revival_full"])
            composed_revival_a[S].append(data["revival_a"])
            composed_revival_b[S].append(data["revival_b"])

    # Composed-own revival rates
    composed_own_revival = {S: [] for S in POST_COMPOSE_STEPS if S > 0}
    for seed in seeds:
        # Find the composed_<domain> key
        own_keys = [k for k in all_seeds_results[seed] if k.startswith("composed_") and k != "composed_joint"]
        if own_keys:
            revival = compute_revival_rates(
                all_seeds_results[seed][own_keys[0]], is_composed=True)
            for S, data in revival.items():
                composed_own_revival[S].append(data["revival_full"])

    # Print revival trajectory table
    print(f"\n  Revival Rate (fraction of anchor dead cohort that revived)")
    print(f"  {'Steps':>6} | {'Single':>12} | {'Composed+Joint':>14} | {'Composed+Own':>14}")
    print(f"  " + "-" * 60)

    for S in POST_COMPOSE_STEPS:
        if S == 0:
            continue
        sr = single_revival_rates.get(S, [])
        cj = composed_joint_revival.get(S, [])
        co = composed_own_revival.get(S, [])

        sr_str = f"{statistics.mean(sr):.1%}" if sr else "N/A"
        cj_str = f"{statistics.mean(cj):.1%}" if cj else "N/A"
        co_str = f"{statistics.mean(co):.1%}" if co else "N/A"

        print(f"  {S:>6} | {sr_str:>12} | {cj_str:>14} | {co_str:>14}")

    # Print per-domain revival in composed model
    print(f"\n  Per-Domain Revival in Composed Model (joint training)")
    print(f"  {'Steps':>6} | {'Domain A':>12} | {'Domain B':>12}")
    print(f"  " + "-" * 40)

    for S in POST_COMPOSE_STEPS:
        if S == 0:
            continue
        ca = composed_revival_a.get(S, [])
        cb = composed_revival_b.get(S, [])
        ca_str = f"{statistics.mean(ca):.1%}" if ca else "N/A"
        cb_str = f"{statistics.mean(cb):.1%}" if cb else "N/A"
        print(f"  {S:>6} | {ca_str:>12} | {cb_str:>12}")

    # Death rate trajectories
    print(f"\n{'='*80}")
    print(f"  DEATH RATE TRAJECTORIES")
    print(f"{'='*80}")

    print(f"\n  {'Steps':>6} | {'Single':>12} | {'Composed+Joint':>14} | {'Comp+Joint A':>12} | {'Comp+Joint B':>12}")
    print(f"  " + "-" * 75)

    for S in POST_COMPOSE_STEPS:
        # Single domain
        single_rates = []
        for seed in seeds:
            for key in all_seeds_results[seed]:
                if key.startswith("single_"):
                    single_rates.append(
                        all_seeds_results[seed][key]["checkpoints"][S]["overall_rate"])

        # Composed joint
        comp_rates = []
        comp_a_rates = []
        comp_b_rates = []
        for seed in seeds:
            ckpt = all_seeds_results[seed]["composed_joint"]["checkpoints"][S]
            comp_rates.append(ckpt["overall_rate"])
            comp_a_rates.append(ckpt["rate_a"])
            comp_b_rates.append(ckpt["rate_b"])

        sr_str = f"{statistics.mean(single_rates):.1%}" if single_rates else "N/A"
        cr_str = f"{statistics.mean(comp_rates):.1%}" if comp_rates else "N/A"
        ca_str = f"{statistics.mean(comp_a_rates):.1%}" if comp_a_rates else "N/A"
        cb_str = f"{statistics.mean(comp_b_rates):.1%}" if comp_b_rates else "N/A"

        print(f"  {S:>6} | {sr_str:>12} | {cr_str:>14} | {ca_str:>12} | {cb_str:>12}")

    # Newly dead analysis
    print(f"\n{'='*80}")
    print(f"  NEW DEATH ANALYSIS (alive->dead transitions from anchor)")
    print(f"{'='*80}")

    print(f"\n  {'Steps':>6} | {'Single A->D':>12} | {'Composed A->D':>14}")
    print(f"  " + "-" * 40)

    for S in POST_COMPOSE_STEPS:
        if S == 0:
            continue
        single_nd = []
        for seed in seeds:
            for key in all_seeds_results[seed]:
                if key.startswith("single_"):
                    revival = compute_revival_rates(
                        all_seeds_results[seed][key], is_composed=False)
                    if S in revival:
                        single_nd.append(revival[S]["n_newly_dead"])

        comp_nd = []
        for seed in seeds:
            revival = compute_revival_rates(
                all_seeds_results[seed]["composed_joint"], is_composed=True)
            if S in revival:
                comp_nd.append(revival[S]["n_newly_dead_full"])

        snd_str = f"{statistics.mean(single_nd):.0f}" if single_nd else "N/A"
        cnd_str = f"{statistics.mean(comp_nd):.0f}" if comp_nd else "N/A"
        print(f"  {S:>6} | {snd_str:>12} | {cnd_str:>14}")

    # KILL CRITERION CHECK
    print(f"\n{'='*80}")
    print(f"  KILL CRITERION CHECK")
    print(f"{'='*80}")
    print(f"\n  Kill: |revival_composed - revival_single| < 5 pp at S={final_step}")

    sr_final = single_revival_rates.get(final_step, [])
    cj_final = composed_joint_revival.get(final_step, [])
    co_final = composed_own_revival.get(final_step, [])

    if sr_final and cj_final:
        mean_single = statistics.mean(sr_final)
        mean_composed_joint = statistics.mean(cj_final)
        diff_joint_pp = (mean_composed_joint - mean_single) * 100

        print(f"\n  Single-domain revival at S=+{final_step}:    {mean_single:.1%}")
        print(f"  Composed+joint revival at S=+{final_step}:    {mean_composed_joint:.1%}")
        print(f"  Difference:                            {diff_joint_pp:+.1f} pp")

        if co_final:
            mean_composed_own = statistics.mean(co_final)
            diff_own_pp = (mean_composed_own - mean_single) * 100
            print(f"  Composed+own revival at S=+{final_step}:     {mean_composed_own:.1%}")
            print(f"  Difference (own):                      {diff_own_pp:+.1f} pp")

        # Check kill criterion using max absolute difference
        max_diff = abs(diff_joint_pp)
        if co_final:
            max_diff = max(max_diff, abs(diff_own_pp))

        if max_diff < 5.0:
            print(f"\n  KILL: Max |diff| = {max_diff:.1f} pp < 5 pp threshold.")
            print(f"  Composition does NOT meaningfully change revival dynamics.")
            print(f"  Implication: 'prune after training' recommendation holds")
            print(f"  regardless of whether model is composed or single-domain.")
            verdict = "KILL"
        else:
            direction = "amplifies" if diff_joint_pp > 0 else "suppresses"
            print(f"\n  PASS: Max |diff| = {max_diff:.1f} pp >= 5 pp threshold.")
            print(f"  Composition {direction} revival by {max_diff:.1f} pp.")
            if diff_joint_pp > 0:
                print(f"  Implication: composed models have MORE revival than single-domain.")
                print(f"  Cross-domain inputs drive additional input distribution shifts.")
                print(f"  Pruning after composition completes is even MORE important.")
            else:
                print(f"  Implication: composed models have LESS revival than single-domain.")
                print(f"  Cross-domain gradients may cancel inter-layer coupling effects.")
                print(f"  Pruning timing is LESS critical in composed setting.")
            verdict = "PASS"

        # Report per-step convergence
        print(f"\n  Revival trajectory comparison:")
        for S in POST_COMPOSE_STEPS:
            if S == 0:
                continue
            sr_s = single_revival_rates.get(S, [])
            cj_s = composed_joint_revival.get(S, [])
            if sr_s and cj_s:
                diff = (statistics.mean(cj_s) - statistics.mean(sr_s)) * 100
                print(f"    S=+{S:>4d}: single={statistics.mean(sr_s):.1%}, "
                      f"composed={statistics.mean(cj_s):.1%}, diff={diff:+.1f} pp")

        # Standard deviations at final step
        if len(sr_final) > 1 and len(cj_final) > 1:
            print(f"\n  Std at S=+{final_step}:")
            print(f"    Single:        {statistics.stdev(sr_final):.1%}")
            print(f"    Composed+joint: {statistics.stdev(cj_final):.1%}")
            if co_final and len(co_final) > 1:
                print(f"    Composed+own:  {statistics.stdev(co_final):.1%}")

        return verdict
    else:
        print(f"\n  ERROR: Missing data for kill criterion check.")
        return "ERROR"


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
