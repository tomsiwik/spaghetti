"""Experiment: Split Domain Specialization.

Tests whether split children specialize faster than independently-initialized
children when fine-tuned on different domains.

Protocol:
1. Train base tree on ALL data (300 steps)
2. Split leaf 0 into children (leaves 0, 1) -- the SPLIT condition
3. Reinitialize leaves 0, 1 randomly -- the INDEPENDENT condition
4. For EACH condition:
   a. Fine-tune child 0 on domain A only, child 1 on domain B only
      (alternating batches, each child + parent gate trainable)
   b. Track learning curves every 25 steps
   c. After training, profile active capsule sets per child per domain
5. Compare:
   KC1: Convergence speed (steps to 99% of final quality, >10% fewer = pass)
   KC2: Domain separation (Jaccard of active capsule sets, <0.95 = pass)

3 seeds x 2 conditions = 6 training runs + analysis.
"""

import sys
import time
import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import ntp_loss, evaluate
from experiments.models import get_model
from experiments.models.split_leaf_actual.split_leaf_actual import (
    split_leaf_into_tree,
    _get_parent_gate,
)
from experiments.models.capsule_moe.capsule_moe import CapsuleGroup
from experiments.models.split_domain_spec.split_domain_spec import (
    profile_leaf_activations,
    compute_domain_jaccard,
)


def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def save_weights(model):
    """Deep copy of model weights."""
    return {k: mx.array(v) for k, v in nn.utils.tree_flatten(model.parameters())}


def freeze_except_leaf_pair_and_gate(model, leaf_idx):
    """Freeze everything except the leaf pair (leaf_idx, leaf_idx+1) and their parent gate."""
    sibling_idx = leaf_idx + 1
    depth = model.layers[0].tree.depth
    parent_gate = _get_parent_gate(leaf_idx, depth)

    model.wte.freeze()
    model.wpe.freeze()
    model.norm0.freeze()
    model.lm_head.freeze()
    for layer in model.layers:
        layer.norm1.freeze()
        layer.attn.freeze()
        layer.norm2.freeze()
        for gi in range(len(layer.tree.gates)):
            if gi != parent_gate:
                layer.tree.gates[gi].freeze()
        for li in range(len(layer.tree.leaves)):
            if li not in (leaf_idx, sibling_idx):
                layer.tree.leaves[li].freeze()


def domain_alternating_train(model, domain_A_ds, domain_B_ds, val_ds_A, val_ds_B,
                              steps=400, batch_size=32, lr=3e-3, seed=42,
                              eval_every=25, log_every=100):
    """Train with alternating domain batches.

    Odd steps: batch from domain A (training child 0)
    Even steps: batch from domain B (training child 1)
    Both children and the parent gate are trainable; the gradient from
    domain A data naturally updates child 0 more (because the gate routes
    A-data toward child 0) and vice versa.

    Actually, since we start from 50/50 routing, both children see both
    domains equally. The specialization happens through the gate learning
    to route. We alternate domains so both get equal training signal.

    Returns dict with learning curves and final metrics.
    """
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    t0 = time.time()
    curve_A = []  # (step, val_loss_A)
    curve_B = []  # (step, val_loss_B)
    curve_mixed = []  # (step, avg(val_A, val_B))

    for step in range(1, steps + 1):
        # Alternate domains
        if step % 2 == 1:
            inputs, targets = domain_A_ds.get_batch(batch_size, rng)
        else:
            inputs, targets = domain_B_ds.get_batch(batch_size, rng)

        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % log_every == 0 or step == steps:
            elapsed = time.time() - t0
            print(f"    step {step:4d}/{steps} | loss {loss.item():.4f} | {elapsed:.1f}s")

        if eval_every > 0 and (step % eval_every == 0 or step == steps):
            vl_A = evaluate(model, val_ds_A, batch_size)
            vl_B = evaluate(model, val_ds_B, batch_size)
            mixed = (vl_A + vl_B) / 2.0
            curve_A.append((step, vl_A))
            curve_B.append((step, vl_B))
            curve_mixed.append((step, mixed))

    return {
        "curve_A": curve_A,
        "curve_B": curve_B,
        "curve_mixed": curve_mixed,
        "final_A": curve_A[-1][1] if curve_A else None,
        "final_B": curve_B[-1][1] if curve_B else None,
        "final_mixed": curve_mixed[-1][1] if curve_mixed else None,
        "elapsed_s": time.time() - t0,
    }


def convergence_step(curve, threshold_frac=0.99):
    """Find the first step where val_loss reaches within threshold_frac of final.

    'Convergence' = first step where val_loss < (1 + (1-threshold_frac)) * final_val.
    I.e., within 1% of final quality for threshold_frac=0.99.

    Returns the step number, or the last step if never reached.
    """
    if not curve:
        return float('inf')
    final = curve[-1][1]
    target = final * (1 + (1 - threshold_frac))  # e.g., 1.01 * final
    for step, val in curve:
        if val <= target:
            return step
    return curve[-1][0]


def run_experiment(seeds=(42, 123, 777), steps_base=300, steps_finetune=400):
    """Run the full split domain specialization experiment."""
    t0_total = time.time()

    print("=" * 70)
    print("EXPERIMENT: Split Domain Specialization")
    print("=" * 70)
    print(f"  Seeds: {seeds}")
    print(f"  Base training: {steps_base} steps")
    print(f"  Domain fine-tuning: {steps_finetune} steps")

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vs = tokenizer.vocab_size

    # Binary domain split: a-m vs n-z
    splits = domain_split(docs, method="binary")
    domain_names = list(splits.keys())
    print(f"  Domains: {domain_names}")
    print(f"  Domain sizes: {[len(v) for v in splits.values()]}")

    results = {
        "split": {"curves_A": [], "curves_B": [], "curves_mixed": [],
                  "finals_A": [], "finals_B": [], "finals_mixed": [],
                  "jaccards": [], "jaccard_details": []},
        "independent": {"curves_A": [], "curves_B": [], "curves_mixed": [],
                        "finals_A": [], "finals_B": [], "finals_mixed": [],
                        "jaccards": [], "jaccard_details": []},
        "base": [],
    }

    tree_cfg = dict(vocab_size=vs, block_size=32, tree_depth=3,
                    n_capsules_per_leaf=32, beam_width=2)

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        # Prepare domain datasets
        domain_datasets = {}
        for d_name, d_docs in splits.items():
            d_train, d_val = train_val_split(d_docs, seed=seed)
            domain_datasets[d_name] = (
                CharDataset(d_train, tokenizer, 32),
                CharDataset(d_val, tokenizer, 32),
            )

        domA_name, domB_name = domain_names[0], domain_names[1]
        trainA, valA = domain_datasets[domA_name]
        trainB, valB = domain_datasets[domB_name]

        # Mixed validation
        all_docs = docs
        _, all_val = train_val_split(all_docs, seed=seed)
        mixed_val = CharDataset(all_val, tokenizer, 32)

        # 1. Train base model on all data
        print(f"\n  [BASE] Training base tree ({steps_base} steps on all data)")
        mx.random.seed(seed)
        base_model = get_model("split_domain_spec", **tree_cfg)
        mx.eval(base_model.parameters())

        all_train, _ = train_val_split(docs, seed=seed)
        all_train_ds = CharDataset(all_train, tokenizer, 32)
        rng = random.Random(seed)
        optimizer = optim.Adam(learning_rate=3e-3)
        loss_and_grad = nn.value_and_grad(base_model, ntp_loss)
        for step in range(1, steps_base + 1):
            inputs, targets = all_train_ds.get_batch(32, rng)
            loss, grads = loss_and_grad(base_model, inputs, targets)
            optimizer.update(base_model, grads)
            mx.eval(base_model.parameters(), optimizer.state)
            if step == steps_base:
                print(f"    step {step}/{steps_base} | loss {loss.item():.4f}")

        base_vl = evaluate(base_model, mixed_val, 32)
        results["base"].append(base_vl)
        print(f"    base val_loss (mixed): {base_vl:.4f}")

        base_ws = save_weights(base_model)

        # ---- (a) SPLIT condition ----
        print(f"\n  [SPLIT] Splitting leaf 0, domain-specific fine-tuning")
        mx.random.seed(seed)
        split_model = get_model("split_domain_spec", **tree_cfg)
        split_model.load_weights(list(base_ws.items()))
        mx.eval(split_model.parameters())

        split_leaf_into_tree(split_model, leaf_idx=0, noise_scale=0.001)
        freeze_except_leaf_pair_and_gate(split_model, leaf_idx=0)
        n_train = count_params(split_model)
        print(f"    trainable params: {n_train}")

        split_result = domain_alternating_train(
            split_model, trainA, trainB, valA, valB,
            steps=steps_finetune, seed=seed, eval_every=25, log_every=100,
        )
        results["split"]["curves_A"].append(split_result["curve_A"])
        results["split"]["curves_B"].append(split_result["curve_B"])
        results["split"]["curves_mixed"].append(split_result["curve_mixed"])
        results["split"]["finals_A"].append(split_result["final_A"])
        results["split"]["finals_B"].append(split_result["final_B"])
        results["split"]["finals_mixed"].append(split_result["final_mixed"])
        print(f"    split final: A={split_result['final_A']:.4f}, B={split_result['final_B']:.4f}")

        # Profile domain separation for split
        print(f"    profiling active capsules...")
        split_active_A = profile_leaf_activations(
            split_model, valA, [0, 1], n_batches=20, batch_size=32, seed=seed)
        split_active_B = profile_leaf_activations(
            split_model, valB, [0, 1], n_batches=20, batch_size=32, seed=seed)

        # Jaccard: compare child0's active set on domain A vs domain B
        # Low Jaccard = child0 uses DIFFERENT capsules for A vs B = specialization
        n_layers = len(split_model.layers)

        # Cross-domain Jaccard for child 0
        j_detail_c0, j_mean_c0 = compute_domain_jaccard(
            split_active_A[0], split_active_B[0], n_layers)
        # Cross-domain Jaccard for child 1
        j_detail_c1, j_mean_c1 = compute_domain_jaccard(
            split_active_A[1], split_active_B[1], n_layers)
        # Combined: average Jaccard across both children
        j_combined = (j_mean_c0 + j_mean_c1) / 2.0

        results["split"]["jaccards"].append(j_combined)
        results["split"]["jaccard_details"].append({
            "child0": j_detail_c0, "child0_mean": j_mean_c0,
            "child1": j_detail_c1, "child1_mean": j_mean_c1,
        })
        print(f"    split Jaccard(domA vs domB): child0={j_mean_c0:.3f}, child1={j_mean_c1:.3f}, combined={j_combined:.3f}")

        # ---- (b) INDEPENDENT condition ----
        print(f"\n  [INDEPENDENT] Random init half-size leaves, domain-specific fine-tuning")
        mx.random.seed(seed + 1000)
        indep_model = get_model("split_domain_spec", **tree_cfg)
        indep_model.load_weights(list(base_ws.items()))
        mx.eval(indep_model.parameters())

        # Replace leaves 0, 1 with random half-size
        for layer in indep_model.layers:
            d = layer.tree.leaves[0].A.weight.shape[1]
            n_caps_full = layer.tree.leaves[0].A.weight.shape[0]
            half = n_caps_full // 2

            for li in [0, 1]:
                new_group = CapsuleGroup(d, half)
                mx.eval(new_group.parameters())
                layer.tree.leaves[li] = new_group

            parent_gate_idx = _get_parent_gate(0, layer.tree.depth)
            gate = layer.tree.gates[parent_gate_idx]
            gate.proj.load_weights([
                ("weight", mx.zeros_like(gate.proj.weight)),
                ("bias", mx.zeros_like(gate.proj.bias)),
            ])
        mx.eval(indep_model.parameters())

        freeze_except_leaf_pair_and_gate(indep_model, leaf_idx=0)
        n_train_indep = count_params(indep_model)
        print(f"    trainable params: {n_train_indep}")

        indep_result = domain_alternating_train(
            indep_model, trainA, trainB, valA, valB,
            steps=steps_finetune, seed=seed, eval_every=25, log_every=100,
        )
        results["independent"]["curves_A"].append(indep_result["curve_A"])
        results["independent"]["curves_B"].append(indep_result["curve_B"])
        results["independent"]["curves_mixed"].append(indep_result["curve_mixed"])
        results["independent"]["finals_A"].append(indep_result["final_A"])
        results["independent"]["finals_B"].append(indep_result["final_B"])
        results["independent"]["finals_mixed"].append(indep_result["final_mixed"])
        print(f"    indep final: A={indep_result['final_A']:.4f}, B={indep_result['final_B']:.4f}")

        # Profile domain separation for independent
        print(f"    profiling active capsules...")
        indep_active_A = profile_leaf_activations(
            indep_model, valA, [0, 1], n_batches=20, batch_size=32, seed=seed)
        indep_active_B = profile_leaf_activations(
            indep_model, valB, [0, 1], n_batches=20, batch_size=32, seed=seed)

        j_detail_c0_i, j_mean_c0_i = compute_domain_jaccard(
            indep_active_A[0], indep_active_B[0], n_layers)
        j_detail_c1_i, j_mean_c1_i = compute_domain_jaccard(
            indep_active_A[1], indep_active_B[1], n_layers)
        j_combined_i = (j_mean_c0_i + j_mean_c1_i) / 2.0

        results["independent"]["jaccards"].append(j_combined_i)
        results["independent"]["jaccard_details"].append({
            "child0": j_detail_c0_i, "child0_mean": j_mean_c0_i,
            "child1": j_detail_c1_i, "child1_mean": j_mean_c1_i,
        })
        print(f"    indep Jaccard(domA vs domB): child0={j_mean_c0_i:.3f}, child1={j_mean_c1_i:.3f}, combined={j_combined_i:.3f}")

    # ---- ANALYSIS ----
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    def mean(lst): return sum(lst) / len(lst)

    # KC1: Convergence speed
    print(f"\n--- KC1: Convergence Speed (steps to 99% of final quality) ---")
    print(f"Threshold: split must reach quality >10% faster than independent")

    split_conv_steps = []
    indep_conv_steps = []
    for si in range(len(seeds)):
        sc = results["split"]["curves_mixed"][si]
        ic = results["independent"]["curves_mixed"][si]
        s_step = convergence_step(sc, threshold_frac=0.99)
        i_step = convergence_step(ic, threshold_frac=0.99)
        split_conv_steps.append(s_step)
        indep_conv_steps.append(i_step)
        speedup = 100 * (i_step - s_step) / i_step if i_step > 0 else 0
        print(f"  Seed {seeds[si]}: split={s_step} steps, indep={i_step} steps, speedup={speedup:+.1f}%")

    mean_split_conv = mean(split_conv_steps)
    mean_indep_conv = mean(indep_conv_steps)
    mean_speedup = 100 * (mean_indep_conv - mean_split_conv) / mean_indep_conv if mean_indep_conv > 0 else 0

    print(f"\n  Mean: split={mean_split_conv:.0f} steps, indep={mean_indep_conv:.0f} steps")
    print(f"  Mean speedup: {mean_speedup:+.1f}%")
    kill_kc1 = mean_speedup < 10.0
    print(f"  KC1 (>10% fewer steps): {'KILLED' if kill_kc1 else 'PASSES'} ({mean_speedup:+.1f}%)")

    # KC2: Domain separation
    print(f"\n--- KC2: Domain Separation (Jaccard of active capsule sets) ---")
    print(f"Threshold: Jaccard < 0.95 (children use different capsules for different domains)")

    for si in range(len(seeds)):
        sj = results["split"]["jaccards"][si]
        ij = results["independent"]["jaccards"][si]
        print(f"  Seed {seeds[si]}: split J={sj:.3f}, indep J={ij:.3f}")

    mean_split_j = mean(results["split"]["jaccards"])
    mean_indep_j = mean(results["independent"]["jaccards"])
    print(f"\n  Mean: split J={mean_split_j:.3f}, indep J={mean_indep_j:.3f}")
    kill_kc2 = mean_split_j > 0.95
    print(f"  KC2 (split J < 0.95): {'KILLED' if kill_kc2 else 'PASSES'} (J={mean_split_j:.3f})")

    # Additional: split vs independent J comparison
    j_diff = mean_split_j - mean_indep_j
    print(f"  Split J - Indep J = {j_diff:+.3f} (negative = split has BETTER separation)")

    # Learning curves summary
    print(f"\n--- Learning Curves Summary ---")
    for si in range(len(seeds)):
        print(f"\n  Seed {seeds[si]}:")
        sc = results["split"]["curves_mixed"][si]
        ic = results["independent"]["curves_mixed"][si]
        print(f"    {'Step':>6} | {'Split':>8} | {'Indep':>8} | {'Delta':>8}")
        print(f"    {'-'*42}")
        for (ss, sv), (is_, iv) in zip(sc, ic):
            delta = sv - iv
            marker = "<-split" if delta < -0.002 else (">-indep" if delta > 0.002 else "~equal")
            print(f"    {ss:>6} | {sv:.4f}   | {iv:.4f}   | {delta:+.4f}  {marker}")

    # Final quality comparison
    print(f"\n--- Final Quality ---")
    split_mixed = mean(results["split"]["finals_mixed"])
    indep_mixed = mean(results["independent"]["finals_mixed"])
    quality_gap = 100 * (split_mixed - indep_mixed) / indep_mixed
    print(f"  Split (mean mixed): {split_mixed:.4f}")
    print(f"  Indep (mean mixed): {indep_mixed:.4f}")
    print(f"  Gap: {quality_gap:+.2f}%")

    # Jaccard detail
    print(f"\n--- Jaccard Detail (per-layer) ---")
    for si in range(len(seeds)):
        print(f"\n  Seed {seeds[si]}, SPLIT:")
        for l, j, nA, nB, nI in results["split"]["jaccard_details"][si]["child0"]:
            print(f"    child0 layer {l}: J={j:.3f} (activeA={nA}, activeB={nB}, intersection={nI})")
        for l, j, nA, nB, nI in results["split"]["jaccard_details"][si]["child1"]:
            print(f"    child1 layer {l}: J={j:.3f} (activeA={nA}, activeB={nB}, intersection={nI})")

        print(f"  Seed {seeds[si]}, INDEPENDENT:")
        for l, j, nA, nB, nI in results["independent"]["jaccard_details"][si]["child0"]:
            print(f"    child0 layer {l}: J={j:.3f} (activeA={nA}, activeB={nB}, intersection={nI})")
        for l, j, nA, nB, nI in results["independent"]["jaccard_details"][si]["child1"]:
            print(f"    child1 layer {l}: J={j:.3f} (activeA={nA}, activeB={nB}, intersection={nI})")

    # Verdict
    print(f"\n{'='*70}")
    print(f"FINAL VERDICT")
    print(f"{'='*70}")
    print(f"  KC1 (convergence speed, >10% faster): {'KILLED' if kill_kc1 else 'PASSES'} ({mean_speedup:+.1f}%)")
    print(f"  KC2 (domain separation, J<0.95):      {'KILLED' if kill_kc2 else 'PASSES'} (J={mean_split_j:.3f})")
    print(f"  Overall: {'KILLED' if (kill_kc1 and kill_kc2) else 'PARTIALLY PASSES' if (kill_kc1 or kill_kc2) else 'PASSES'}")
    total_time = time.time() - t0_total
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    return {
        "results": results,
        "kill_kc1": kill_kc1,
        "kill_kc2": kill_kc2,
        "mean_speedup": mean_speedup,
        "mean_split_jaccard": mean_split_j,
        "mean_indep_jaccard": mean_indep_j,
    }


if __name__ == "__main__":
    run_experiment()
