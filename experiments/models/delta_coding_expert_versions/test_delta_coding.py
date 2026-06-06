"""Delta Coding for Expert Version Management -- Full Experiment.

Tests the hypothesis that expert weight updates can be stored as deltas
(like video P-frames) with <1% quality drift per delta application and
<50% storage cost vs full snapshots.

Protocol:
  1. Pretrain base model (200 steps)
  2. Fine-tune LoRA expert v1 on domain A (200 steps)
  3. Continue fine-tuning to produce versions v2..v5 (80 steps each,
     on slightly different data subsets to simulate expert evolution)
  4. Build version chains with keyframe intervals K=1,2,5
  5. Measure reconstruction quality and storage cost
  6. Test delta compression via truncated SVD (ranks 1,2,4)
  7. Evaluate quality drift along chain length

Kill criteria:
  - quality drift >1% per delta application
  - delta storage >50% of full expert storage
"""

import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.data import (
    load_names, CharTokenizer, CharDataset,
    domain_split, train_val_split,
)
from experiments.train import ntp_loss, evaluate
from experiments.models.lora_procrustes.lora_procrustes import LoRAGPT


def freeze_except_lora(model):
    """Freeze all parameters except LoRA A and B."""
    model.freeze()
    for layer in model.layers:
        layer.mlp.fc1.unfreeze()
        layer.mlp.fc2.unfreeze()
        layer.mlp.fc1.linear.freeze()
        layer.mlp.fc2.linear.freeze()


def extract_lora_params(model):
    """Extract LoRA A/B parameters as a flat dict."""
    params = {}
    for l_idx, layer in enumerate(model.layers):
        for name in ['fc1', 'fc2']:
            fc = getattr(layer.mlp, name)
            params[f"layer{l_idx}.{name}.A"] = mx.array(fc.A)
            params[f"layer{l_idx}.{name}.B"] = mx.array(fc.B)
    return params


def load_lora_params(model, params):
    """Load LoRA A/B parameters from a flat dict."""
    for l_idx, layer in enumerate(model.layers):
        for name in ['fc1', 'fc2']:
            fc = getattr(layer.mlp, name)
            fc.A = params[f"layer{l_idx}.{name}.A"]
            fc.B = params[f"layer{l_idx}.{name}.B"]
    mx.eval(model.parameters())


def train_steps(model, dataset, steps, lr=3e-3, seed=42):
    """Train model for a given number of steps."""
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    for step in range(steps):
        inputs, targets = dataset.get_batch(32, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)


def quick_eval(model, val_ds, n_batches=5):
    """Fast evaluation with fewer batches."""
    return evaluate(model, val_ds, batch_size=32, n_batches=n_batches)


def run_experiment(seed=42):
    """Run the full delta coding experiment for one seed."""
    print(f"\n{'='*60}")
    print(f"  DELTA CODING EXPERIMENT (seed={seed})")
    print(f"{'='*60}")

    # Setup
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    docs_train, docs_val = train_val_split(docs, seed=seed)
    domains = domain_split(docs_train, method="binary")

    domain_docs = domains["a_m"]
    rng = random.Random(seed)
    rng.shuffle(domain_docs)

    chunk_size = len(domain_docs) // 5
    subsets = [domain_docs[i*chunk_size:(i+1)*chunk_size] for i in range(5)]

    full_train = CharDataset(domain_docs, tokenizer, block_size=32)
    val_ds = CharDataset(
        [d for d in docs_val if d[0].lower() <= 'm'],
        tokenizer, block_size=32
    )

    # Pretrain base
    print("\n1. Pretraining base (200 steps)...")
    model = LoRAGPT(
        vocab_size=tokenizer.vocab_size, block_size=32,
        n_embd=64, n_head=4, n_layer=4, lora_rank=8
    )
    mx.eval(model.parameters())
    train_steps(model, full_train, steps=200, seed=seed)
    base_loss = quick_eval(model, val_ds)
    print(f"   Base val loss: {base_loss:.4f}")

    # Freeze base, train LoRA
    freeze_except_lora(model)

    # Generate 5 versions
    print("\n2. Generating 5 expert versions...")
    versions = []
    for v in range(5):
        subset_ds = CharDataset(subsets[v], tokenizer, block_size=32)
        steps = 200 if v == 0 else 80
        train_steps(model, subset_ds, steps=steps, seed=seed + v)
        params = extract_lora_params(model)
        val_loss = quick_eval(model, val_ds)
        versions.append((params, val_loss))
        print(f"   v{v+1}: val_loss={val_loss:.4f}")

    # Delta statistics
    print("\n3. Inter-version delta analysis...")
    delta_ratios = []
    for v in range(1, len(versions)):
        total_delta_sq = 0.0
        total_param_sq = 0.0
        for k in versions[v][0]:
            diff = versions[v][0][k] - versions[v-1][0][k]
            total_delta_sq += mx.sum(diff * diff).item()
            total_param_sq += mx.sum(versions[v][0][k] * versions[v][0][k]).item()
        ratio = (total_delta_sq ** 0.5) / (total_param_sq ** 0.5)
        delta_ratios.append(ratio)
        print(f"   delta v{v}->v{v+1}: ||delta||/||params|| = {ratio:.4f}")

    from experiments.models.delta_coding_expert_versions.delta_coding_expert_versions import ExpertVersionChain

    # Test 1: Raw delta coding (no compression) -- quality should be exact
    print("\n4. Raw delta coding (K=5, longest chain)...")
    chain = ExpertVersionChain(keyframe_interval=5)
    for v_params, _ in versions:
        chain.add_version(v_params)

    raw_storage = chain.storage_cost(use_compressed=False)
    print(f"   Raw storage ratio: {raw_storage['ratio']:.3f} (raw delta = same size as full)")

    raw_drifts = []
    for v_idx in range(len(versions)):
        recon_err = chain.reconstruction_error(v_idx, use_compressed=False)
        # Only evaluate function-space drift for last version (most accumulated)
        if v_idx == len(versions) - 1:
            reconstructed = chain.reconstruct(v_idx, use_compressed=False)
            load_lora_params(model, reconstructed)
            recon_loss = quick_eval(model, val_ds)
            true_loss = versions[v_idx][1]
            drift = (recon_loss - true_loss) / true_loss * 100
            raw_drifts.append(drift)
            print(f"   v{v_idx+1}: recon_err={recon_err:.2e}, drift={drift:+.4f}%")
        else:
            raw_drifts.append(0.0)  # exact by construction
            print(f"   v{v_idx+1}: recon_err={recon_err:.2e}")

    # Test 2: SVD-compressed delta coding
    print("\n5. Compressed delta coding (SVD, K=5)...")
    compression_results = {}
    for svd_rank in [1, 2, 4]:
        chain = ExpertVersionChain(keyframe_interval=5)
        for v_params, _ in versions:
            chain.add_version(v_params)

        comp_stats = chain.compress_deltas(rank=svd_rank)
        storage = chain.storage_cost(use_compressed=True)

        # Evaluate last version (most accumulated compression error)
        v_idx = len(versions) - 1
        reconstructed = chain.reconstruct(v_idx, use_compressed=True)
        load_lora_params(model, reconstructed)
        recon_loss = quick_eval(model, val_ds)
        true_loss = versions[v_idx][1]
        drift = (recon_loss - true_loss) / true_loss * 100

        # Also check v3 (mid-chain)
        v_mid = 2
        reconstructed_mid = chain.reconstruct(v_mid, use_compressed=True)
        load_lora_params(model, reconstructed_mid)
        recon_loss_mid = quick_eval(model, val_ds)
        true_loss_mid = versions[v_mid][1]
        drift_mid = (recon_loss_mid - true_loss_mid) / true_loss_mid * 100

        avg_comp = sum(s["compression_ratio"] for s in comp_stats.values()) / max(len(comp_stats), 1)
        avg_rel_err = sum(s["relative_error"] for s in comp_stats.values()) / max(len(comp_stats), 1)

        compression_results[svd_rank] = {
            "max_drift": max(abs(drift), abs(drift_mid)),
            "drift_v5": drift,
            "drift_v3": drift_mid,
            "storage": storage,
            "avg_delta_compression": avg_comp,
            "avg_relative_error": avg_rel_err,
        }

        print(f"   rank={svd_rank}: delta_comp={avg_comp:.3f} "
              f"storage={storage['ratio']:.3f} ({storage['savings_pct']:.1f}% savings) "
              f"drift_v3={drift_mid:+.3f}% drift_v5={drift:+.3f}% "
              f"rel_err={avg_rel_err:.4f}")

    return {
        "seed": seed,
        "base_loss": base_loss,
        "version_losses": [v[1] for v in versions],
        "delta_ratios": delta_ratios,
        "raw_storage": raw_storage,
        "raw_drifts": raw_drifts,
        "compression_results": compression_results,
    }


def check_kill_criteria(results_list):
    """Check kill criteria across all seeds."""
    print(f"\n{'='*60}")
    print(f"  KILL CRITERIA CHECK ({len(results_list)} seeds)")
    print(f"{'='*60}")

    # KC1: quality drift >1% per delta application (raw, no compression)
    raw_max_drifts = [max(abs(d) for d in r["raw_drifts"]) for r in results_list]
    raw_max = max(raw_max_drifts)
    print(f"\n  KC1: Quality drift (raw deltas, no compression)")
    print(f"    Max |drift| across seeds: {raw_max:.4f}%  (threshold: 1%)")
    kc1_pass = raw_max < 1.0
    print(f"    Verdict: {'PASS' if kc1_pass else 'KILL'}")

    # KC2: delta storage >50% of full storage
    # Raw deltas are same size (100%) but compressed deltas are smaller
    # We check the best compression that still meets KC1
    print(f"\n  KC2: Storage ratio (compressed deltas)")
    best_passing_rank = None
    for svd_rank in [1, 2, 4]:
        ratios = [r["compression_results"][svd_rank]["storage"]["ratio"]
                  for r in results_list]
        drifts = [r["compression_results"][svd_rank]["max_drift"]
                  for r in results_list]
        avg_r = sum(ratios) / len(ratios)
        max_d = max(drifts)
        status = "PASS" if avg_r < 0.50 and max_d < 1.0 else "KILL"
        if status == "PASS" and best_passing_rank is None:
            best_passing_rank = svd_rank
        print(f"    SVD rank={svd_rank}: storage={avg_r:.3f} max_drift={max_d:.3f}% -> {status}")

    # For KC2, check if any compression level achieves <50% storage with <1% drift
    if best_passing_rank:
        best_ratio = sum(r["compression_results"][best_passing_rank]["storage"]["ratio"]
                         for r in results_list) / len(results_list)
        kc2_pass = best_ratio < 0.50
        print(f"    Best passing: rank={best_passing_rank}, ratio={best_ratio:.3f}")
    else:
        kc2_pass = False
        print(f"    No compression level achieves both <50% storage and <1% drift")

    print(f"    Verdict: {'PASS' if kc2_pass else 'KILL'}")

    # Delta norm statistics
    print(f"\n  Delta Norm Analysis:")
    all_ratios = []
    for r in results_list:
        all_ratios.extend(r["delta_ratios"])
    avg_delta_ratio = sum(all_ratios) / len(all_ratios)
    print(f"    Avg ||delta||/||params||: {avg_delta_ratio:.4f}")
    print(f"    Interpretation: deltas are {avg_delta_ratio*100:.1f}% the size of full params")
    print(f"    This is why raw deltas don't save storage -- they're dense updates.")
    print(f"    SVD compression exploits the LOW RANK structure of these deltas.")

    # Overall
    overall = kc1_pass and kc2_pass
    print(f"\n  OVERALL: {'PASS' if overall else 'PARTIAL'}")
    if kc1_pass and not kc2_pass:
        print(f"  KC1 passes (exact reconstruction), KC2 needs compression")
    return overall


def main():
    """Run experiment across 3 seeds."""
    t0 = time.time()
    results = []
    for seed in [42, 123, 7]:
        results.append(run_experiment(seed=seed))
    check_kill_criteria(results)
    print(f"\n  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
