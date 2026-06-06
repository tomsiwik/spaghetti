"""Tests for Union-Find Expert Merging.

Test plan:
  1. Unit test: UnionFind data structure correctness
  2. Unit test: build_union_find_from_similarity with known matrices
  3. Integration test: full merge pipeline on a trained composed model
  4. Experiment: threshold sweep to find compression vs quality Pareto
  5. Experiment: union-find vs greedy pairing comparison
"""

import random
import copy

import mlx.core as mx
import mlx.nn as nn

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate
from experiments.models.relu_router.relu_router import ReLURouterGPT, ReLUCapsulePool

from .union_find_merge import (
    UnionFind,
    build_union_find_from_similarity,
    union_find_merge,
    threshold_sweep,
    compare_uf_vs_greedy,
)


# ── Unit tests ──────────────────────────────────────────────────────────────

def test_union_find_basic():
    """Test UnionFind data structure operations."""
    uf = UnionFind(6)

    # Initially 6 components
    assert uf.n_components() == 6

    # Union 0-1, 2-3
    assert uf.union(0, 1) == True
    assert uf.union(2, 3) == True
    assert uf.n_components() == 4

    # Union 1-2 (connects {0,1} with {2,3})
    assert uf.union(1, 2) == True
    assert uf.n_components() == 3

    # Find should return same root for 0,1,2,3
    assert uf.find(0) == uf.find(1) == uf.find(2) == uf.find(3)

    # 4 and 5 are still separate
    assert uf.find(4) != uf.find(0)
    assert uf.find(5) != uf.find(0)

    # Clusters
    clusters = uf.clusters()
    sizes = sorted([len(v) for v in clusters.values()], reverse=True)
    assert sizes == [4, 1, 1]

    # Redundant union returns False
    assert uf.union(0, 3) == False

    print("  PASS: test_union_find_basic")


def test_union_find_path_compression():
    """Verify path compression makes subsequent finds O(1)."""
    uf = UnionFind(100)

    # Build a long chain: 0 -> 1 -> 2 -> ... -> 99
    for i in range(99):
        uf.union(i, i + 1)

    # First find traverses full chain
    root = uf.find(0)

    # After path compression, parent[0] should point directly to root
    assert uf.parent[0] == root

    # All elements should now have same root
    for i in range(100):
        assert uf.find(i) == root

    print("  PASS: test_union_find_path_compression")


def test_build_union_find_from_similarity():
    """Test union-find construction from known similarity matrices."""
    P = 5
    # Construct Jaccard matrix where 0~1, 1~2, 3~4 are similar
    J = mx.zeros((P, P))
    # 0-1: J=0.8, 1-2: J=0.6, 3-4: J=0.7
    J_list = [[1.0, 0.8, 0.1, 0.0, 0.0],
              [0.8, 1.0, 0.6, 0.0, 0.0],
              [0.1, 0.6, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.7],
              [0.0, 0.0, 0.0, 0.7, 1.0]]
    J = mx.array(J_list)

    # Output correlation: all similar pairs have high correlation
    corr = mx.array([[1.0, 0.9, 0.5, 0.0, 0.0],
                     [0.9, 1.0, 0.7, 0.0, 0.0],
                     [0.5, 0.7, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0, 0.8],
                     [0.0, 0.0, 0.0, 0.8, 1.0]])

    alive = mx.array([True, True, True, True, True])

    # Threshold J>0.5, corr>0.4: should merge {0,1,2} and {3,4}
    uf = build_union_find_from_similarity(J, corr, alive,
                                          jaccard_threshold=0.5,
                                          output_corr_threshold=0.4)
    assert uf.n_components() == 2  # {0,1,2} and {3,4}
    assert uf.find(0) == uf.find(1) == uf.find(2)
    assert uf.find(3) == uf.find(4)
    assert uf.find(0) != uf.find(3)

    # Higher threshold J>0.75: should merge {0,1} only, rest separate
    # (J(0,1)=0.8 passes, J(1,2)=0.6 fails, J(3,4)=0.7 fails at 0.75)
    uf2 = build_union_find_from_similarity(J, corr, alive,
                                           jaccard_threshold=0.75,
                                           output_corr_threshold=0.4)
    assert uf2.n_components() == 4  # {0,1}, {2}, {3}, {4}
    assert uf2.find(0) == uf2.find(1)
    assert uf2.find(2) != uf2.find(0)

    # Transitive closure: at J>0.5, 0~1 and 1~2 means 0~2 even though J(0,2)=0.1
    # This is the KEY advantage over greedy pairing
    assert uf.find(0) == uf.find(2)

    print("  PASS: test_build_union_find_from_similarity")


# ── Integration test ────────────────────────────────────────────────────────

def test_union_find_merge_pipeline():
    """Full integration test: train, compose, merge, evaluate."""
    print("\n=== Union-Find Merge Pipeline Test ===")

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    splits = domain_split(docs, method="binary")
    V = tokenizer.vocab_size

    # Prepare datasets
    datasets = {}
    for name, d_docs in splits.items():
        d_train, d_val = train_val_split(d_docs, seed=42)
        datasets[name] = (
            CharDataset(d_train, tokenizer, 32),
            CharDataset(d_val, tokenizer, 32),
        )

    # 1. Train shared base
    print("\n  1. Training shared base (200 steps)...")
    base = ReLURouterGPT(vocab_size=V, n_capsules=256)
    mx.eval(base.parameters())
    all_train = CharDataset(
        [d for ds in splits.values() for d in ds],
        tokenizer, 32
    )
    train(base, all_train, steps=200, batch_size=32, lr=3e-3, seed=42, log_every=100)

    # 2. Fine-tune per domain (freeze attention, train MLP only)
    domain_models = {}
    for domain_name, (train_ds, val_ds) in datasets.items():
        print(f"\n  2. Fine-tuning domain '{domain_name}' (200 steps)...")
        model = copy.deepcopy(base)
        mx.eval(model.parameters())

        # Freeze attention + embeddings, train only capsule pools
        model.wte.freeze()
        model.wpe.freeze()
        for layer in model.layers:
            layer.attn.freeze()
            layer.norm1.freeze()

        train(model, train_ds, steps=200, batch_size=32, lr=3e-3,
              seed=42, log_every=100)
        domain_models[domain_name] = model

    # 3. Compose by concatenating capsule pools
    print("\n  3. Composing domain models...")
    composed = copy.deepcopy(base)
    mx.eval(composed.parameters())

    for l_idx, layer in enumerate(composed.layers):
        A_parts = []
        B_parts = []
        for dm in domain_models.values():
            dm_layer = dm.layers[l_idx]
            A_parts.append(dm_layer.capsule_pool.A.weight)
            B_parts.append(dm_layer.capsule_pool.B.weight)

        A_cat = mx.concatenate(A_parts, axis=0)  # (2*P, d)
        B_cat = mx.concatenate(B_parts, axis=1)  # (d, 2*P)
        mx.eval(A_cat, B_cat)

        P_total = A_cat.shape[0]
        d = A_cat.shape[1]
        new_pool = ReLUCapsulePool(d, P_total)
        new_pool.A.load_weights([("weight", A_cat)])
        new_pool.B.load_weights([("weight", B_cat)])
        layer.capsule_pool = new_pool
        mx.eval(layer.capsule_pool.parameters())

    # 4. Evaluate baseline (pre-merge)
    print("\n  4. Evaluating composed model (pre-merge)...")
    all_val = CharDataset(
        [d for ds in splits.values()
         for d in train_val_split(ds, seed=42)[1]],
        tokenizer, 32
    )
    baseline_loss = evaluate(composed, all_val, batch_size=32)
    print(f"  Baseline val loss: {baseline_loss:.4f}")

    total_capsules_before = sum(
        layer.capsule_pool.n_capsules for layer in composed.layers
    )
    print(f"  Total capsules before: {total_capsules_before}")

    # 5. Apply union-find merging
    print("\n  5. Applying union-find merging (J>0.3, corr>0.3)...")
    composed_merged = copy.deepcopy(composed)
    mx.eval(composed_merged.parameters())

    merge_result = union_find_merge(
        composed_merged, all_val,
        jaccard_threshold=0.3,
        output_corr_threshold=0.3,
        n_batches=20,
        batch_size=32,
        seed=0,
        verbose=True,
    )

    # 6. Evaluate post-merge quality
    post_loss = evaluate(composed_merged, all_val, batch_size=32)
    delta_pct = (post_loss - baseline_loss) / baseline_loss * 100
    print(f"\n  Post-merge val loss: {post_loss:.4f} ({delta_pct:+.2f}%)")
    print(f"  Merged: {merge_result['pct_merged']:.1f}%")

    # 7. Check kill criteria
    kill_quality = delta_pct > 3.0
    kill_compression = merge_result["pct_merged"] < 20.0
    print(f"\n  Kill criteria:")
    print(f"    Quality: {delta_pct:+.2f}% {'> 3% KILL' if kill_quality else '<= 3% PASS'}")
    print(f"    Compression: {merge_result['pct_merged']:.1f}% "
          f"{'< 20% KILL' if kill_compression else '>= 20% PASS'}")

    return {
        "baseline_loss": baseline_loss,
        "post_loss": post_loss,
        "delta_pct": delta_pct,
        "pct_merged": merge_result["pct_merged"],
        "killed": kill_quality or kill_compression,
    }


# ── Full experiment ─────────────────────────────────────────────────────────

def run_full_experiment(seeds: list[int] = [42, 123, 7]):
    """Run the complete experiment across multiple seeds.

    For each seed:
      1. Train base + fine-tune 2 domains + compose
      2. Threshold sweep: find best compression/quality trade-off
      3. Compare union-find vs greedy at best threshold
    """
    print("=" * 70)
    print("Union-Find Expert Merging: Full Experiment")
    print("=" * 70)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    splits = domain_split(docs, method="binary")
    V = tokenizer.vocab_size

    all_results = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Seed: {seed}")
        print(f"{'='*70}")

        # Prepare datasets
        datasets = {}
        for name, d_docs in splits.items():
            d_train, d_val = train_val_split(d_docs, seed=seed)
            datasets[name] = (
                CharDataset(d_train, tokenizer, 32),
                CharDataset(d_val, tokenizer, 32),
            )

        # Train shared base
        print("\n  Training shared base (300 steps)...")
        base = ReLURouterGPT(vocab_size=V, n_capsules=256)
        mx.eval(base.parameters())
        all_docs = [d for ds in splits.values() for d in ds]
        all_train_docs, all_val_docs = train_val_split(all_docs, seed=seed)
        all_train_ds = CharDataset(all_train_docs, tokenizer, 32)
        all_val_ds = CharDataset(all_val_docs, tokenizer, 32)
        train(base, all_train_ds, steps=300, batch_size=32, lr=3e-3,
              seed=seed, log_every=150)

        # Fine-tune per domain
        domain_models = {}
        for domain_name, (train_ds, val_ds) in datasets.items():
            print(f"\n  Fine-tuning domain '{domain_name}' (200 steps)...")
            model = copy.deepcopy(base)
            mx.eval(model.parameters())
            model.wte.freeze()
            model.wpe.freeze()
            for layer in model.layers:
                layer.attn.freeze()
                layer.norm1.freeze()
            train(model, train_ds, steps=200, batch_size=32, lr=3e-3,
                  seed=seed, log_every=100)
            domain_models[domain_name] = model

        # Compose
        print("\n  Composing domain models...")
        composed = copy.deepcopy(base)
        mx.eval(composed.parameters())
        for l_idx, layer in enumerate(composed.layers):
            A_parts = []
            B_parts = []
            for dm in domain_models.values():
                dm_layer = dm.layers[l_idx]
                A_parts.append(dm_layer.capsule_pool.A.weight)
                B_parts.append(dm_layer.capsule_pool.B.weight)
            A_cat = mx.concatenate(A_parts, axis=0)
            B_cat = mx.concatenate(B_parts, axis=1)
            mx.eval(A_cat, B_cat)
            P_total = A_cat.shape[0]
            d = A_cat.shape[1]
            new_pool = ReLUCapsulePool(d, P_total)
            new_pool.A.load_weights([("weight", A_cat)])
            new_pool.B.load_weights([("weight", B_cat)])
            layer.capsule_pool = new_pool
            mx.eval(layer.capsule_pool.parameters())

        # Threshold sweep
        print("\n  Running threshold sweep...")
        sweep_results = threshold_sweep(
            composed, all_train_ds, all_val_ds,
            thresholds=[
                (0.15, 0.15),  # very aggressive
                (0.2, 0.2),    # aggressive
                (0.3, 0.3),    # moderate
                (0.4, 0.3),    # moderate-conservative
                (0.5, 0.4),    # conservative
            ],
            n_batches=20,
            batch_size=32,
            seed=seed,
            verbose=True,
        )

        # Compare UF vs greedy at moderate threshold
        print("\n  Comparing union-find vs greedy (J>0.3, corr>0.3)...")
        comparison = compare_uf_vs_greedy(
            composed, all_train_ds, all_val_ds,
            jaccard_threshold=0.3,
            output_corr_threshold=0.3,
            n_batches=20,
            batch_size=32,
            seed=seed,
            verbose=True,
        )

        seed_result = {
            "seed": seed,
            "sweep": sweep_results,
            "comparison": comparison,
        }
        all_results.append(seed_result)

        # Print summary for this seed
        print(f"\n  === Seed {seed} Summary ===")
        print(f"  {'Threshold':<15} {'Compression':>12} {'Quality Delta':>15} {'Verdict':>8}")
        print(f"  {'-'*55}")
        for r in sweep_results:
            jt, ct = r["jaccard_threshold"], r["output_corr_threshold"]
            verdict = "PASS" if r["delta_pct"] <= 3.0 and r["pct_merged"] >= 20.0 else "FAIL"
            print(f"  J>{jt},c>{ct:<5} {r['pct_merged']:>10.1f}%  {r['delta_pct']:>+13.2f}%  {verdict:>8}")

        print(f"\n  UF vs Greedy (J>0.3, corr>0.3):")
        print(f"    Union-Find: {comparison['union_find']['pct_merged']:.1f}% merged, "
              f"{comparison['union_find']['delta_pct']:+.2f}% quality")
        print(f"    Greedy:     {comparison['greedy']['pct_merged']:.1f}% merged, "
              f"{comparison['greedy']['delta_pct']:+.2f}% quality")

    # Aggregate across seeds
    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS (across {len(seeds)} seeds)")
    print(f"{'='*70}")

    # For each threshold, average metrics across seeds
    n_thresholds = len(all_results[0]["sweep"])
    for t_idx in range(n_thresholds):
        jt = all_results[0]["sweep"][t_idx]["jaccard_threshold"]
        ct = all_results[0]["sweep"][t_idx]["output_corr_threshold"]
        compressions = [r["sweep"][t_idx]["pct_merged"] for r in all_results]
        deltas = [r["sweep"][t_idx]["delta_pct"] for r in all_results]
        mean_comp = sum(compressions) / len(compressions)
        mean_delta = sum(deltas) / len(deltas)
        max_delta = max(deltas)

        passes = sum(1 for c, d in zip(compressions, deltas)
                     if d <= 3.0 and c >= 20.0)
        print(f"  J>{jt},c>{ct}: compression={mean_comp:.1f}%, "
              f"quality={mean_delta:+.2f}% (max={max_delta:+.2f}%), "
              f"pass={passes}/{len(seeds)}")

    # UF vs greedy aggregate
    uf_comps = [r["comparison"]["union_find"]["pct_merged"] for r in all_results]
    uf_deltas = [r["comparison"]["union_find"]["delta_pct"] for r in all_results]
    gr_comps = [r["comparison"]["greedy"]["pct_merged"] for r in all_results]
    gr_deltas = [r["comparison"]["greedy"]["delta_pct"] for r in all_results]

    print(f"\n  UF vs Greedy (J>0.3, corr>0.3, mean of {len(seeds)} seeds):")
    print(f"    Union-Find: {sum(uf_comps)/len(uf_comps):.1f}% merged, "
          f"{sum(uf_deltas)/len(uf_deltas):+.2f}% quality")
    print(f"    Greedy:     {sum(gr_comps)/len(gr_comps):.1f}% merged, "
          f"{sum(gr_deltas)/len(gr_deltas):+.2f}% quality")

    # Final verdict
    best_threshold = None
    for t_idx in range(n_thresholds):
        compressions = [r["sweep"][t_idx]["pct_merged"] for r in all_results]
        deltas = [r["sweep"][t_idx]["delta_pct"] for r in all_results]
        mean_comp = sum(compressions) / len(compressions)
        mean_delta = sum(deltas) / len(deltas)
        if mean_comp >= 20.0 and mean_delta <= 3.0:
            best_threshold = (
                all_results[0]["sweep"][t_idx]["jaccard_threshold"],
                all_results[0]["sweep"][t_idx]["output_corr_threshold"],
                mean_comp,
                mean_delta,
            )

    print(f"\n  FINAL VERDICT:")
    if best_threshold:
        jt, ct, comp, delta = best_threshold
        print(f"    PASS: J>{jt}, corr>{ct} achieves {comp:.1f}% compression "
              f"at {delta:+.2f}% quality loss")
    else:
        print(f"    FAIL: No threshold achieves >=20% compression at <=3% quality loss")

    return all_results


if __name__ == "__main__":
    # Run unit tests first
    print("Running unit tests...")
    test_union_find_basic()
    test_union_find_path_compression()
    test_build_union_find_from_similarity()
    print("All unit tests passed.\n")

    # Run full experiment
    results = run_full_experiment(seeds=[42, 123, 7])
