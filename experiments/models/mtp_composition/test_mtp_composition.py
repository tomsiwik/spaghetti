"""Tests and experiment for MTP effect on capsule composition.

Protocol (matches capsule_moe/test_composition.py):
1. Pretrain shared base on all data (300 steps)
2. Fine-tune only capsule groups per domain (freeze attention) -- 300 steps
3. Compose: extract domain capsule groups, concatenate, calibrate router (100 steps)
4. Compare: joint vs composed, across NTP (depth=1) and MTP (depth=2,3)

Kill criteria:
  1. MTP-trained groups compose >5% worse than NTP groups
  2. MTP provides <2% quality improvement for composed models over NTP composed
"""

import time
import random
import json
import statistics

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.models import get_model
from experiments.data import (
    load_names, CharTokenizer, CharDataset,
    domain_split, train_val_split,
)
from experiments.train import evaluate


# --- Config ---
BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
CAP = dict(**BASE, n_groups=4, n_capsules_per_group=64, top_k_groups=2)
PRETRAIN_STEPS = 300
FINETUNE_STEPS = 300
CALIBRATION_STEPS = 100
BATCH_SIZE = 32
LR = 3e-3


# --- Unit tests ---

SMALL_CFG = dict(vocab_size=28, block_size=32, n_embd=32, n_head=4, n_layer=2,
                 n_groups=2, n_capsules_per_group=32, top_k_groups=2)


def test_forward_shape():
    """MTP model produces correct NTP logit shape."""
    print("=" * 60)
    print("test_forward_shape")

    for depth in [1, 2, 3]:
        model = get_model("mtp_capsule_moe", **SMALL_CFG, mtp_depth=depth)
        tokens = mx.zeros((2, 16), dtype=mx.int32)
        logits = model(tokens)
        mx.eval(logits)
        assert logits.shape == (2, 16, 28), f"depth={depth}: {logits.shape}"
        print(f"  mtp_depth={depth}: logits shape {logits.shape}  OK")

    print("  PASSED\n")


def test_mtp_loss_nonzero():
    """MTP loss is nonzero for depth>1 and zero for depth=1."""
    print("=" * 60)
    print("test_mtp_loss_nonzero")

    targets = mx.zeros((2, 16), dtype=mx.int32)

    for depth in [1, 2, 3]:
        model = get_model("mtp_capsule_moe", **SMALL_CFG, mtp_depth=depth)
        tokens = mx.zeros((2, 16), dtype=mx.int32)
        model(tokens)
        loss = model.mtp_loss(targets)
        mx.eval(loss)
        if depth == 1:
            assert loss.item() == 0.0, f"depth=1 should have 0 MTP loss"
        else:
            assert loss.item() > 0.0, f"depth={depth} should have nonzero MTP loss"
        print(f"  mtp_depth={depth}: mtp_loss={loss.item():.4f}  OK")

    print("  PASSED\n")


def test_param_count():
    """MTP adds expected parameter count."""
    print("=" * 60)
    print("test_param_count")

    d = SMALL_CFG["n_embd"]
    params = {}
    for depth in [1, 2, 3]:
        model = get_model("mtp_capsule_moe", **SMALL_CFG, mtp_depth=depth)
        p = sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))
        params[depth] = p
        print(f"  depth={depth}: {p:,} params")

    expected_per_module = d * d
    assert params[2] - params[1] == expected_per_module
    assert params[3] - params[1] == 2 * expected_per_module
    print("  PASSED\n")


# --- MTP-aware training ---

def mtp_ntp_loss(model, inputs, targets):
    """Combined NTP + MTP loss."""
    logits = model(inputs)
    B, T, V = logits.shape
    ntp_loss = nn.losses.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        reduction="mean",
    )
    mtp_loss = model.mtp_loss(targets)
    return ntp_loss + mtp_loss + model.aux_loss()


def ntp_only_loss(model, inputs, targets):
    """NTP loss only (for router calibration, no MTP)."""
    logits = model(inputs)
    B, T, V = logits.shape
    loss = nn.losses.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        reduction="mean",
    )
    return loss + model.aux_loss()


def train_model(model, dataset, val_dataset=None,
                steps=500, batch_size=32, lr=3e-3,
                seed=42, log_every=100, use_mtp=True):
    """Train with MTP-aware loss (or NTP only if use_mtp=False)."""
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_fn = mtp_ntp_loss if use_mtp else ntp_only_loss
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    t0 = time.time()

    for step in range(1, steps + 1):
        inputs, targets = dataset.get_batch(batch_size, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()
        losses.append(loss_val)

        if step % log_every == 0 or step == steps:
            elapsed = time.time() - t0
            print(f"  step {step:4d}/{steps} | loss {loss_val:.4f}")

    elapsed = time.time() - t0
    val_loss = evaluate(model, val_dataset, batch_size) if val_dataset else None

    return {"final_loss": losses[-1], "val_loss": val_loss,
            "losses": losses, "elapsed_s": elapsed}


def test_learns_with_mtp():
    """MTP model can learn (loss decreases)."""
    print("=" * 60)
    print("test_learns_with_mtp")

    docs = load_names()
    tok = CharTokenizer(docs)
    ds = CharDataset(docs[:500], tok, block_size=32)

    for depth in [1, 2, 3]:
        model = get_model("mtp_capsule_moe",
                          vocab_size=tok.vocab_size, block_size=32,
                          n_embd=64, n_head=4, n_layer=2,
                          n_groups=2, n_capsules_per_group=64,
                          top_k_groups=2, mtp_depth=depth)

        result = train_model(model, ds, steps=150, batch_size=32, lr=3e-3,
                             log_every=150, seed=42)

        first = result["losses"][0]
        final = result["final_loss"]
        print(f"  depth={depth}: {first:.4f} -> {final:.4f} (delta={final-first:.4f})")
        assert final < first, f"depth={depth} loss didn't decrease"

    print("  PASSED\n")


# --- Composition helpers (adapted from capsule_moe/test_composition.py) ---

def compose_from_shared_base(base_model, groups_a, groups_b, vocab_size, block_size,
                              n_capsules_per_group=64):
    """Create composed model: shared base + concatenated domain capsule groups.

    groups_a: list of lists -- groups_a[layer_idx] = list of CapsuleGroup modules
    groups_b: same structure for domain B
    """
    from experiments.models.capsule_moe.capsule_moe import CapsuleMoEGPT

    n_groups_a = len(groups_a[0])
    n_groups_b = len(groups_b[0])
    composed_groups = n_groups_a + n_groups_b
    composed_top_k = 4  # 2+2, same active fraction

    composed = get_model("capsule_moe",
                         vocab_size=vocab_size, block_size=block_size,
                         n_groups=composed_groups,
                         n_capsules_per_group=n_capsules_per_group,
                         top_k_groups=composed_top_k,
                         **{k: BASE[k] for k in ["n_embd", "n_head", "n_layer"]})

    # Copy base shared params
    composed.wte.weight = base_model.wte.weight
    composed.wpe.weight = base_model.wpe.weight
    composed.lm_head.weight = base_model.lm_head.weight

    for l_idx in range(len(composed.layers)):
        layer_c = composed.layers[l_idx]
        layer_base = base_model.layers[l_idx]

        # Attention from base
        layer_c.attn.wq.weight = layer_base.attn.wq.weight
        layer_c.attn.wk.weight = layer_base.attn.wk.weight
        layer_c.attn.wv.weight = layer_base.attn.wv.weight
        layer_c.attn.wo.weight = layer_base.attn.wo.weight

        # Domain capsule groups
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
    """Freeze all parameters except capsule pool routers."""
    model.freeze()
    for layer in model.layers:
        layer.capsule_pool.router.unfreeze()


def calibrate_router(model, train_ds_a, train_ds_b, steps=100, lr=3e-3, seed=42):
    """Train only router weights on mixed-domain data."""
    freeze_except_router(model)

    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_only_loss)

    for step in range(1, steps + 1):
        if step % 2 == 1:
            inputs, targets = train_ds_a.get_batch(BATCH_SIZE, rng)
        else:
            inputs, targets = train_ds_b.get_batch(BATCH_SIZE, rng)

        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step == steps:
            print(f"    router cal step {step:3d}/{steps} | loss {loss.item():.4f}")

    model.unfreeze()


# --- Main experiment ---

def run_one_depth(mtp_depth, seed=42):
    """Run full composition experiment for one MTP depth and one seed.

    Protocol:
    1. Pretrain shared base on all data
    2. Fine-tune capsule groups per domain (freeze attention)
    3. Train joint model for reference
    4. Compose and calibrate router
    5. Evaluate
    """
    print(f"\n{'='*60}")
    print(f"  DEPTH={mtp_depth}, SEED={seed}")
    print(f"{'='*60}")

    docs = load_names()
    tok = CharTokenizer(docs)
    V = tok.vocab_size
    splits = domain_split(docs, method="binary")

    # Prepare datasets
    domain_datasets = {}
    for d_name, d_docs in splits.items():
        d_tr, d_vl = train_val_split(d_docs, seed=seed)
        domain_datasets[d_name] = (
            CharDataset(d_tr, tok, block_size=32),
            CharDataset(d_vl, tok, block_size=32),
        )

    # Joint dataset
    all_docs = [d for ds in splits.values() for d in ds]
    all_tr, all_vl = train_val_split(all_docs, seed=seed)
    joint_train = CharDataset(all_tr, tok, block_size=32)
    joint_val = CharDataset(all_vl, tok, block_size=32)

    model_kwargs = dict(
        vocab_size=V,
        mtp_depth=mtp_depth, mtp_lambda=0.3,
        **CAP,
    )

    # --- 1. Pretrain base ---
    print("\n--- Pretrain base ---")
    base = get_model("mtp_capsule_moe", **model_kwargs)
    mx.eval(base.parameters())
    train_model(base, joint_train, steps=PRETRAIN_STEPS, batch_size=BATCH_SIZE,
                lr=LR, seed=seed, log_every=150)

    # Snapshot base weights
    base_params = [(k, mx.array(v)) for k, v in nn.utils.tree_flatten(base.parameters())]

    # --- 2. Fine-tune capsule groups per domain ---
    domain_models = {}
    for d_name, (train_ds, val_ds) in domain_datasets.items():
        print(f"\n--- Fine-tune capsules: {d_name} ---")
        model = get_model("mtp_capsule_moe", **model_kwargs)
        model.load_weights(base_params)
        mx.eval(model.parameters())

        # Freeze everything except capsule groups
        model.freeze()
        for layer in model.layers:
            for group in layer.capsule_pool.groups:
                group.unfreeze()

        train_model(model, train_ds, steps=FINETUNE_STEPS, batch_size=BATCH_SIZE,
                    lr=LR, seed=seed, log_every=150)
        model.unfreeze()
        domain_models[d_name] = model

    # --- 3. Joint model (trained from base on all data) ---
    print("\n--- Joint model ---")
    joint_model = get_model("mtp_capsule_moe", **model_kwargs)
    joint_model.load_weights(base_params)
    mx.eval(joint_model.parameters())
    train_model(joint_model, joint_train, steps=FINETUNE_STEPS, batch_size=BATCH_SIZE,
                lr=LR, seed=seed, log_every=150)

    # --- 4. Compose ---
    print("\n--- Compose + calibrate ---")
    domains = list(domain_models.keys())
    groups_a = [domain_models[domains[0]].layers[l].capsule_pool.groups
                for l in range(BASE["n_layer"])]
    groups_b = [domain_models[domains[1]].layers[l].capsule_pool.groups
                for l in range(BASE["n_layer"])]

    composed = compose_from_shared_base(
        base, groups_a, groups_b, V, BASE["block_size"],
        n_capsules_per_group=CAP["n_capsules_per_group"],
    )

    train_a, _ = domain_datasets[domains[0]]
    train_b, _ = domain_datasets[domains[1]]
    calibrate_router(composed, train_a, train_b,
                     steps=CALIBRATION_STEPS, lr=LR, seed=seed)

    # --- 5. Evaluate ---
    print("\n--- Evaluate ---")
    joint_per_domain = {}
    composed_per_domain = {}
    single_per_domain = {}
    for d_name, (_, val_ds) in domain_datasets.items():
        joint_per_domain[d_name] = evaluate(joint_model, val_ds, BATCH_SIZE)
        composed_per_domain[d_name] = evaluate(composed, val_ds, BATCH_SIZE)
        single_per_domain[d_name] = evaluate(domain_models[d_name], val_ds, BATCH_SIZE)

    joint_avg = sum(joint_per_domain.values()) / len(joint_per_domain)
    composed_avg = sum(composed_per_domain.values()) / len(composed_per_domain)
    single_avg = sum(single_per_domain.values()) / len(single_per_domain)
    gap_pct = (composed_avg - joint_avg) / joint_avg * 100

    print(f"  Single-domain avg:  {single_avg:.4f}")
    print(f"  Joint avg:          {joint_avg:.4f}")
    print(f"  Composed avg:       {composed_avg:.4f}")
    print(f"  Composition gap:    {gap_pct:+.2f}%")

    return {
        "mtp_depth": mtp_depth,
        "seed": seed,
        "single_per_domain": single_per_domain,
        "single_avg": single_avg,
        "joint_per_domain": joint_per_domain,
        "joint_avg": joint_avg,
        "composed_per_domain": composed_per_domain,
        "composed_avg": composed_avg,
        "composition_gap_pct": gap_pct,
    }


def run_full_experiment():
    """Run MTP composition experiment: 3 depths x 3 seeds."""
    print("\n" + "=" * 70)
    print("  MTP COMPOSITION EXPERIMENT")
    print("  Depths: {1 (NTP), 2 (MTP-2), 3 (MTP-3)} x 3 seeds")
    print("=" * 70)

    depths = [1, 2, 3]
    seeds = [42, 123, 777]
    all_results = {d: [] for d in depths}

    for depth in depths:
        for seed in seeds:
            result = run_one_depth(depth, seed)
            all_results[depth].append(result)

    # --- Aggregate ---
    print("\n" + "=" * 70)
    print("  AGGREGATED RESULTS (mean +/- std across seeds)")
    print("=" * 70)

    summaries = {}
    for depth in depths:
        results = all_results[depth]
        joint_avgs = [r["joint_avg"] for r in results]
        composed_avgs = [r["composed_avg"] for r in results]
        gaps = [r["composition_gap_pct"] for r in results]
        single_avgs = [r["single_avg"] for r in results]

        summaries[depth] = {
            "mean_single": statistics.mean(single_avgs),
            "mean_joint": statistics.mean(joint_avgs),
            "mean_composed": statistics.mean(composed_avgs),
            "mean_gap": statistics.mean(gaps),
            "std_gap": statistics.stdev(gaps) if len(gaps) > 1 else 0,
            "per_seed_gaps": gaps,
        }

    print(f"\n{'Depth':>6} | {'Single':>8} | {'Joint':>8} | {'Composed':>9} | {'Gap':>10} | {'Std':>6}")
    print("-" * 58)
    for depth in depths:
        s = summaries[depth]
        print(f"{depth:>6} | {s['mean_single']:>8.4f} | {s['mean_joint']:>8.4f} | "
              f"{s['mean_composed']:>9.4f} | {s['mean_gap']:>+8.2f}% | {s['std_gap']:>5.2f}%")

    # --- Kill criteria ---
    ntp = summaries[1]
    print("\n--- KILL CRITERIA ---")

    for depth in [2, 3]:
        mtp = summaries[depth]
        print(f"\n  Depth {depth} vs NTP (depth 1):")

        # Kill 1: MTP composes >5% worse than NTP
        # Interpretation: MTP's composition gap is >5pp worse than NTP's gap
        gap_diff = mtp["mean_gap"] - ntp["mean_gap"]
        kill1 = mtp["mean_gap"] > ntp["mean_gap"] + 5.0
        print(f"    Kill 1 (MTP gap >5pp worse than NTP gap):")
        print(f"      NTP gap={ntp['mean_gap']:+.2f}%, MTP gap={mtp['mean_gap']:+.2f}%, diff={gap_diff:+.2f}pp")
        print(f"      -> {'KILL' if kill1 else 'PASS'}")

        # Kill 2: MTP provides <2% quality improvement for composed models
        # (lower loss = better, so improvement = (ntp - mtp) / ntp * 100)
        improvement = (ntp["mean_composed"] - mtp["mean_composed"]) / ntp["mean_composed"] * 100
        kill2 = improvement < 2.0
        print(f"    Kill 2 (MTP provides <2% quality improvement for composed):")
        print(f"      NTP composed={ntp['mean_composed']:.4f}, MTP composed={mtp['mean_composed']:.4f}")
        print(f"      Improvement: {improvement:+.2f}%")
        print(f"      -> {'KILL' if kill2 else 'PASS'}")

    # --- Save results ---
    results_path = "/Users/tom/Code/tomsiwik/llm/experiments/models/mtp_composition/results.json"
    output = {
        "summaries": {str(d): s for d, s in summaries.items()},
        "per_seed": {str(d): all_results[d] for d in depths},
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    return summaries, all_results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "experiment":
        run_full_experiment()
    else:
        test_forward_shape()
        test_mtp_loss_nonzero()
        test_param_count()
        test_learns_with_mtp()
        print("\nAll MTP composition unit tests passed!")
