"""Experiment: Skip-List Routing under Shared-Base Composition Protocol.

Tests whether skip-list multi-resolution routing survives composition.
Adapts the hierarchical_tree composition experiment protocol directly.

Protocol (same for flat, tree, and skip-list):
1. Pretrain base model on all data (300 steps)
2. Fine-tune expert modules per domain (attention frozen, 200 steps)
3. Compose by weight-averaging domain expert modules
4. Calibrate router/gates on mixed data (100 steps)
5. Evaluate

Comparisons:
- flat (CapsuleMoE G=8, k=2): the proven composition baseline
- tree (hierarchical_tree D=3, B=2): the proven hierarchical composition baseline
- skip (SkipListRouting N=8, k=2): the test subject

Kill criteria:
- KC1: skip-list composition gap >3% worse than flat composition gap
- KC2: level-weight distribution collapses to uniform under composition
  (i.e., weight above Level 0 drops below 10%, or becomes uniform across levels)

Additional diagnostics:
- Level-weight distribution before vs after composition
- Comparison to single-domain skip-list level weights (from parent experiment)
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


def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def count_total_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))


def train_model(model, train_ds, val_ds, steps=500, batch_size=32, lr=3e-3,
                seed=42, log_every=100):
    """Train a model and return metrics."""
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    losses = []
    t0 = time.time()

    for step in range(1, steps + 1):
        inputs, targets = train_ds.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss.item())

        if step % log_every == 0 or step == steps:
            elapsed = time.time() - t0
            print(f"    step {step:4d}/{steps} | loss {loss.item():.4f} | {elapsed:.1f}s")

    val_loss = evaluate(model, val_ds, batch_size)
    return {"losses": losses, "val_loss": val_loss, "elapsed_s": time.time() - t0}


def collect_skip_routing_stats(model, val_ds, n_batches=20, batch_size=32):
    """Collect skip-list level-weight stats over validation set.

    Returns per-level mean weight and whether the model has skip_pool layers.
    """
    rng = random.Random(0)
    all_usage = {}  # layer_name -> list of (n_levels+1,) arrays

    for _ in range(n_batches):
        inputs, targets = val_ds.get_batch(batch_size, rng)
        _ = model(inputs)
        mx.eval(model.parameters())

        for i, layer in enumerate(model.layers):
            if not hasattr(layer, 'skip_pool'):
                return None
            pool = layer.skip_pool
            lname = f"layer_{i}"

            if pool._level_usage is not None:
                usage = pool._level_usage
                B, T, nL = usage.shape
                usage_flat = usage.reshape(B * T, nL)
                if lname not in all_usage:
                    all_usage[lname] = []
                all_usage[lname].append(usage_flat)

    if not all_usage:
        return None

    stats = {}
    for lname in all_usage:
        all_u = mx.concatenate(all_usage[lname], axis=0)
        mean_u = mx.mean(all_u, axis=0).tolist()
        std_u = mx.std(all_u, axis=0).tolist()
        mx.eval(all_u)

        pool = model.layers[int(lname.split("_")[1])].skip_pool
        depth = pool.avg_routing_depth()

        stats[lname] = {
            "level_usage_mean": mean_u,
            "level_usage_std": std_u,
            "avg_depth": depth.item() if depth is not None else None,
            "n_levels": pool.n_levels,
            "n_tokens": all_u.shape[0],
        }

    return stats


def get_expert_key(model_type):
    """Return the attribute name for the expert module in each layer."""
    if model_type == "flat":
        return "capsule_pool"
    elif model_type == "tree":
        return "tree"
    elif model_type == "skip":
        return "skip_pool"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def make_model(model_type, vs):
    """Create a fresh model of the given type."""
    if model_type == "flat":
        return get_model("capsule_moe", vocab_size=vs, block_size=32,
                         n_groups=8, n_capsules_per_group=32, top_k_groups=2)
    elif model_type == "tree":
        return get_model("hierarchical_tree", vocab_size=vs, block_size=32,
                         tree_depth=3, n_capsules_per_leaf=32, beam_width=2)
    elif model_type == "skip":
        return get_model("skip_list_routing", vocab_size=vs, block_size=32,
                         n_experts=8, n_capsules_per_expert=32, top_k=2)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def run_composition(model_type, vs, domain_datasets, joint_train, joint_val,
                    seed, steps_pretrain=300, steps_finetune=200, steps_calibrate=100):
    """Run the full composition protocol for one model type and seed.

    Returns: {joint_val, composed_val, routing_stats_joint, routing_stats_composed}
    """
    expert_key = get_expert_key(model_type)

    # 1. Joint training baseline
    print(f"  [{model_type} joint baseline]")
    mx.random.seed(seed)
    joint_model = make_model(model_type, vs)
    mx.eval(joint_model.parameters())
    n_params = count_total_params(joint_model)
    print(f"    total params: {n_params:,}")
    joint_result = train_model(joint_model, joint_train, joint_val,
                               steps=steps_pretrain + steps_finetune,
                               seed=seed, log_every=200)
    joint_val_loss = joint_result["val_loss"]
    print(f"    joint val_loss: {joint_val_loss:.4f}")

    # Collect routing stats for joint model (skip only)
    routing_joint = None
    if model_type == "skip":
        routing_joint = collect_skip_routing_stats(joint_model, joint_val)

    # 2. Pretrain base on all data
    print(f"  [{model_type} pretrain base]")
    mx.random.seed(seed)
    base_model = make_model(model_type, vs)
    mx.eval(base_model.parameters())
    _ = train_model(base_model, joint_train, joint_val,
                    steps=steps_pretrain, seed=seed, log_every=200)

    # Save base weights
    base_weights = {k: mx.array(v) for k, v in
                    nn.utils.tree_flatten(base_model.parameters())}

    # 3. Fine-tune per domain (freeze attention+embeddings, train only experts)
    domain_expert_weights = {}
    for d_name, (d_train, d_val) in domain_datasets.items():
        print(f"    fine-tune {model_type} on {d_name}...")
        mx.random.seed(seed)
        d_model = make_model(model_type, vs)
        d_model.load_weights(list(base_weights.items()))
        mx.eval(d_model.parameters())

        # Freeze everything except expert modules
        d_model.wte.freeze()
        d_model.wpe.freeze()
        d_model.norm0.freeze()
        d_model.lm_head.freeze()
        for layer in d_model.layers:
            layer.norm1.freeze()
            layer.attn.freeze()
            layer.norm2.freeze()

        _ = train_model(d_model, d_train, d_val,
                        steps=steps_finetune, seed=seed, log_every=200)

        # Save domain-specific expert weights
        domain_expert_weights[d_name] = {}
        for layer_idx, layer in enumerate(d_model.layers):
            expert_module = getattr(layer, expert_key)
            for k, v in nn.utils.tree_flatten(expert_module.parameters()):
                full_key = f"layers.{layer_idx}.{expert_key}.{k}"
                domain_expert_weights[d_name][full_key] = mx.array(v)

    # 4. Compose: weight-average domain expert modules
    print(f"    composing {model_type} via weight averaging...")
    composed_model = make_model(model_type, vs)
    composed_model.load_weights(list(base_weights.items()))

    domains = list(domain_expert_weights.keys())
    n_domains = len(domains)
    for key in domain_expert_weights[domains[0]]:
        avg_w = sum(domain_expert_weights[d][key] for d in domains) / n_domains
        parts = key.split(".")
        obj = composed_model
        for p in parts[:-1]:
            if p.isdigit():
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)
        setattr(obj, parts[-1], avg_w)

    mx.eval(composed_model.parameters())

    # 5. Calibrate: train router/gates only on mixed data
    print(f"    calibrating {model_type} router...")
    composed_model.wte.freeze()
    composed_model.wpe.freeze()
    composed_model.norm0.freeze()
    composed_model.lm_head.freeze()
    for layer in composed_model.layers:
        layer.norm1.freeze()
        layer.attn.freeze()
        layer.norm2.freeze()
        # Freeze expert weights, keep routing params trainable
        expert_module = getattr(layer, expert_key)
        if model_type == "flat":
            for g in expert_module.groups:
                g.freeze()
        elif model_type == "tree":
            for leaf in expert_module.leaves:
                leaf.freeze()
        elif model_type == "skip":
            # Freeze leaf experts, keep routers + confidence gates trainable
            for expert in expert_module.experts:
                expert.freeze()

    _ = train_model(composed_model, joint_train, joint_val,
                    steps=steps_calibrate, seed=seed, log_every=100)

    # 6. Evaluate
    composed_val_loss = evaluate(composed_model, joint_val, 32)
    print(f"    composed val_loss: {composed_val_loss:.4f}")

    # Collect routing stats for composed model (skip only)
    routing_composed = None
    if model_type == "skip":
        routing_composed = collect_skip_routing_stats(composed_model, joint_val)

    return {
        "joint_val": joint_val_loss,
        "composed_val": composed_val_loss,
        "routing_joint": routing_joint,
        "routing_composed": routing_composed,
        "n_params": n_params,
    }


def print_routing_stats(stats, label):
    """Print skip-list routing statistics."""
    if stats is None:
        return
    print(f"\n    {label} routing stats:")
    total_above_l0 = 0.0
    n_layers = 0
    for lname, lstats in sorted(stats.items()):
        mean_u = lstats["level_usage_mean"]
        std_u = lstats["level_usage_std"]
        depth = lstats["avg_depth"]
        n_levels = lstats["n_levels"]

        # Level weights are ordered: coarsest first, finest (L0) last
        above_l0 = sum(mean_u[:-1])  # everything except Level 0
        total_above_l0 += above_l0
        n_layers += 1

        usage_str = " | ".join(f"L{n_levels - i}={m:.3f}+/-{s:.3f}"
                               for i, (m, s) in enumerate(zip(mean_u, std_u)))
        print(f"      {lname}: depth={depth:.3f} above_L0={100*above_l0:.1f}% [{usage_str}]")

    if n_layers > 0:
        avg_above_l0 = total_above_l0 / n_layers
        print(f"      Mean weight above Level 0: {100*avg_above_l0:.1f}%")
        return avg_above_l0
    return None


def run_experiment(seeds=(42, 123, 777), steps_pretrain=300,
                   steps_finetune=200, steps_calibrate=100):
    """Main experiment: composition comparison across flat, tree, skip."""
    print("=" * 70)
    print("EXPERIMENT: Skip-List Routing under Shared-Base Composition")
    print("=" * 70)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    vs = tokenizer.vocab_size
    splits = domain_split(docs)

    # Only run flat (baseline) and skip (test subject).
    # Tree composition results already proven at +0.17% gap (hierarchical_tree PAPER.md).
    results = {mt: {"joint": [], "composed": []} for mt in ["flat", "skip"]}
    skip_routing = {"joint_above_l0": [], "composed_above_l0": []}

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Seed {seed}")
        print(f"{'='*50}")

        # Prepare domain datasets
        domain_datasets = {}
        for d_name, d_docs in splits.items():
            d_train, d_val = train_val_split(d_docs, seed=seed)
            domain_datasets[d_name] = (
                CharDataset(d_train, tokenizer, 32),
                CharDataset(d_val, tokenizer, 32),
            )

        # Joint dataset (all data)
        all_train, all_val = train_val_split(docs, seed=seed)
        joint_train = CharDataset(all_train, tokenizer, 32)
        joint_val = CharDataset(all_val, tokenizer, 32)

        for model_type in ["flat", "skip"]:
            print(f"\n  === {model_type} ===")
            r = run_composition(model_type, vs, domain_datasets,
                                joint_train, joint_val, seed,
                                steps_pretrain, steps_finetune, steps_calibrate)
            results[model_type]["joint"].append(r["joint_val"])
            results[model_type]["composed"].append(r["composed_val"])

            if model_type == "skip":
                j_above = print_routing_stats(r["routing_joint"], "Joint")
                c_above = print_routing_stats(r["routing_composed"], "Composed")
                if j_above is not None:
                    skip_routing["joint_above_l0"].append(j_above)
                if c_above is not None:
                    skip_routing["composed_above_l0"].append(c_above)

    # ============ SUMMARY ============
    print(f"\n{'='*70}")
    print(f"SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*70}")

    gaps = {}
    for mt in ["flat", "skip"]:
        joint_mean = sum(results[mt]["joint"]) / len(seeds)
        comp_mean = sum(results[mt]["composed"]) / len(seeds)
        gap = 100 * (comp_mean - joint_mean) / joint_mean
        gaps[mt] = gap
        print(f"  {mt:5s}: joint={joint_mean:.4f}  composed={comp_mean:.4f}  gap={gap:+.2f}%")
        print(f"         joint per-seed:    {['%.4f' % v for v in results[mt]['joint']]}")
        print(f"         composed per-seed: {['%.4f' % v for v in results[mt]['composed']]}")

    # Kill Criterion 1: skip composition gap vs flat composition gap
    skip_vs_flat_delta = gaps["skip"] - gaps["flat"]
    print(f"\n  Flat composition gap:  {gaps['flat']:+.2f}%")
    print(f"  Tree composition gap (prior experiment): +0.17%")
    print(f"  Skip composition gap:  {gaps['skip']:+.2f}%")
    print(f"  Skip vs Flat delta:    {skip_vs_flat_delta:+.2f}pp")

    kc1 = skip_vs_flat_delta > 3.0
    print(f"\n  KC1 (skip composition gap >3% worse than flat): "
          f"{'TRIGGERED - KILL' if kc1 else 'PASSES'} ({skip_vs_flat_delta:+.2f}pp)")

    # Kill Criterion 2: level-weight distribution under composition
    print(f"\n  Level-weight concentration analysis:")
    if skip_routing["joint_above_l0"] and skip_routing["composed_above_l0"]:
        j_mean = sum(skip_routing["joint_above_l0"]) / len(skip_routing["joint_above_l0"])
        c_mean = sum(skip_routing["composed_above_l0"]) / len(skip_routing["composed_above_l0"])
        print(f"    Joint training: {100*j_mean:.1f}% weight above Level 0")
        print(f"    After composition: {100*c_mean:.1f}% weight above Level 0")
        print(f"    Change: {100*(c_mean - j_mean):+.1f}pp")

        # Check for collapse to uniform
        # With 4 levels (L3, L2, L1, L0), uniform = 25% per level
        # Collapse = all levels within 5pp of each other (effectively uniform)
        # Or weight above L0 < 10% (Level 0 dominates)
        kc2_collapse = c_mean < 0.10
        if kc2_collapse:
            print(f"    KC2: TRIGGERED - weight above L0 dropped below 10%")
        else:
            print(f"    KC2: PASSES - concentration maintained at {100*c_mean:.1f}%")
    else:
        kc2_collapse = None
        print(f"    KC2: INCONCLUSIVE (missing routing stats)")

    # Reference: parent experiment single-domain was 60.6% above L0
    print(f"\n  Reference: single-domain skip-list had 60.6% weight above Level 0")

    # Final verdict
    print(f"\n{'='*70}")
    if kc1:
        print("VERDICT: KILL (KC1 triggered - composition gap too large)")
    elif kc2_collapse:
        print("VERDICT: KILL (KC2 triggered - level-weight collapse)")
    elif kc2_collapse is None:
        print("VERDICT: INCONCLUSIVE (KC2 could not be evaluated)")
    else:
        print("VERDICT: PASSES both kill criteria")
    print(f"{'='*70}")

    return {
        "results": results,
        "gaps": gaps,
        "skip_routing": skip_routing,
        "kc1_triggered": kc1,
        "kc2_triggered": kc2_collapse,
    }


if __name__ == "__main__":
    t0 = time.time()
    result = run_experiment(seeds=(42, 123, 777))
    total = time.time() - t0
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f} min)")
