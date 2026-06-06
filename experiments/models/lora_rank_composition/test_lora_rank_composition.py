"""LoRA Rank Sensitivity Sweep for Composition Quality.

Tests whether LoRA rank constrains composition quality through a rate-distortion
tradeoff. Sweeps r in {2, 4, 8, 16, 32, 64} on the same 2-domain setup as
exp_lora_procrustes_linear.

For each rank, measures:
1. Composition quality (val loss of composed model vs joint baseline)
2. Orthogonality (cosine similarity between domain deltas)
3. Dead neuron rate (fraction of ReLU units that never fire post-composition)
4. Shared fraction (shared_norm / (shared_norm + unique_norm))
5. Effective rank of deltas (via SVD singular value spectrum)

Kill criteria:
- Rank has no effect on composition quality (all ranks within 1% of each other)
- Orthogonality does not correlate with rank (r^2 < 0.2)
"""

import random
import statistics
import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate, ntp_loss
from experiments.models.lora_procrustes.lora_procrustes import LoRAGPT
from experiments.models.lora_procrustes.test_lora_procrustes import (
    copy_weights, count_params, freeze_except_lora, reset_lora,
    get_deltas, decompose_deltas, apply_deltas_to_base,
    RoutedDeltaGPT, calibrate_router,
)


# ── Config ──────────────────────────────────────────────────────────────────

BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
RANKS = [2, 4, 8, 16, 32, 64]
LORA_ALPHA = 1.0
PRETRAIN_STEPS = 300
FINETUNE_STEPS = 300
ROUTER_CAL_STEPS = 100
BATCH_SIZE = 32
LR = 3e-3


# ── Dead Neuron Profiling ───────────────────────────────────────────────────

def profile_dead_neurons(model, dataset, n_batches=20):
    """Profile which ReLU neurons never fire across n_batches.

    Returns fraction of dead neurons (neurons that output 0 for all inputs).
    Works for both GPT (has mlp.fc1) and RoutedDeltaGPT (has expert weights).
    """
    rng = random.Random(0)

    # For standard GPT models, hook into the MLP layers
    if hasattr(model, 'layers'):
        n_layer = len(model.layers)
        hidden_dim = 4 * BASE['n_embd']
        ever_fired = [mx.zeros(hidden_dim) for _ in range(n_layer)]

        for _ in range(n_batches):
            inputs, _ = dataset.get_batch(BATCH_SIZE, rng)
            B, T = inputs.shape
            pos = mx.arange(T)
            x = model.wte(inputs) + model.wpe(pos)
            x = model.norm0(x)

            for l_idx, layer in enumerate(model.layers):
                x_attn = x + layer.attn(layer.norm1(x))
                h = layer.norm2(x_attn)

                # Get pre-ReLU activations
                if hasattr(layer.mlp, 'fc1'):
                    if hasattr(layer.mlp.fc1, 'linear'):
                        # LoRA model
                        pre_relu = layer.mlp.fc1(h)
                    else:
                        # Standard GPT
                        pre_relu = layer.mlp.fc1(h)
                else:
                    x = x_attn + layer.mlp(h)
                    continue

                # Track which neurons fire (pre_relu > 0 means ReLU output > 0)
                fired = (pre_relu > 0).astype(mx.float32)
                # Max across batch and sequence dims
                fired_any = mx.max(fired.reshape(-1, hidden_dim), axis=0)
                ever_fired[l_idx] = mx.maximum(ever_fired[l_idx], fired_any)

                # Complete forward pass for next layer
                x = x_attn + layer.mlp(h)

        mx.eval(ever_fired)
        dead_counts = [int((1.0 - ef).sum().item()) for ef in ever_fired]
        total_dead = sum(dead_counts)
        total_neurons = n_layer * hidden_dim
        return total_dead / total_neurons if total_neurons > 0 else 0.0

    return 0.0


# ── Effective Rank (via SVD) ────────────────────────────────────────────────

def compute_effective_rank(deltas):
    """Compute effective rank of concatenated delta matrices via SVD.

    Effective rank = exp(Shannon entropy of normalized singular values).
    This gives a smooth measure of dimensionality that is 1 for rank-1
    matrices and full_rank for uniform singular values.
    """
    # Concatenate all deltas into one big matrix
    flat_deltas = []
    for l_idx, name, delta in deltas:
        flat_deltas.append(delta.reshape(-1))
    all_flat = mx.concatenate(flat_deltas)

    # Reshape into a matrix for SVD (stack all deltas as rows)
    # Use individual delta matrices and compute effective rank per matrix
    eff_ranks = []
    for l_idx, name, delta in deltas:
        # SVD of the delta matrix
        # delta shape: (in_dim, out_dim)
        U, S, Vt = mx.linalg.svd(delta, stream=mx.cpu)
        # Normalize singular values to form a distribution
        S_pos = mx.maximum(S, mx.array(1e-12))
        S_norm = S_pos / mx.sum(S_pos)
        # Shannon entropy
        H = -mx.sum(S_norm * mx.log(S_norm + 1e-12))
        eff_rank = mx.exp(H).item()
        eff_ranks.append(eff_rank)

    return statistics.mean(eff_ranks)


# ── Single Rank Experiment ──────────────────────────────────────────────────

def run_single_rank(rank, seed=42, verbose=True):
    """Run the full composition pipeline for a single LoRA rank.

    Returns a dict with all metrics for this rank.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"RANK = {rank}, SEED = {seed}")
        print(f"{'='*60}")

    mx.random.seed(seed)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    splits = domain_split(docs)
    all_train, all_val = train_val_split(docs, seed=seed)

    train_a_docs, val_a_docs = train_val_split(splits["a_m"], seed=seed)
    train_b_docs, val_b_docs = train_val_split(splits["n_z"], seed=seed)

    train_a = CharDataset(train_a_docs, tokenizer, BASE["block_size"])
    val_a = CharDataset(val_a_docs, tokenizer, BASE["block_size"])
    train_b = CharDataset(train_b_docs, tokenizer, BASE["block_size"])
    val_b = CharDataset(val_b_docs, tokenizer, BASE["block_size"])
    joint_train = CharDataset(all_train, tokenizer, BASE["block_size"])
    joint_val = CharDataset(all_val, tokenizer, BASE["block_size"])

    # === 1. Joint training baseline (standard GPT, no LoRA) ===
    if verbose:
        print("\n--- Joint training baseline ---")
    model_joint = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(model_joint.parameters())
    total_steps = 2 * FINETUNE_STEPS
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(model_joint, ntp_loss)
    for step in range(1, total_steps + 1):
        if step % 2 == 1:
            inputs, targets = train_a.get_batch(BATCH_SIZE, rng)
        else:
            inputs, targets = train_b.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model_joint, inputs, targets)
        optimizer.update(model_joint, grads)
        mx.eval(model_joint.parameters(), optimizer.state)

    j_a = evaluate(model_joint, val_a, BATCH_SIZE)
    j_b = evaluate(model_joint, val_b, BATCH_SIZE)
    joint_avg = (j_a + j_b) / 2
    if verbose:
        print(f"  Joint: avg={joint_avg:.4f}")

    # === 2. Pretrain base model ===
    if verbose:
        print("\n--- Pretraining base model ---")
    base_model = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_model.parameters())
    train(base_model, joint_train, steps=PRETRAIN_STEPS,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=300)

    # === 3. Fine-tune LoRA adapters per domain ===
    def finetune_lora(domain_train, domain_val, domain_name):
        if verbose:
            print(f"\n--- Fine-tuning LoRA (r={rank}) for domain {domain_name} ---")
        lora_model = get_model("lora_gpt", vocab_size=V, **BASE,
                               lora_rank=rank, lora_alpha=LORA_ALPHA)
        mx.eval(lora_model.parameters())

        # Copy base weights
        for l_idx in range(BASE['n_layer']):
            bl = base_model.layers[l_idx]
            ll = lora_model.layers[l_idx]
            ll.attn.wq.weight = bl.attn.wq.weight
            ll.attn.wk.weight = bl.attn.wk.weight
            ll.attn.wv.weight = bl.attn.wv.weight
            ll.attn.wo.weight = bl.attn.wo.weight
            ll.mlp.fc1.linear.weight = bl.mlp.fc1.weight
            ll.mlp.fc2.linear.weight = bl.mlp.fc2.weight
        lora_model.wte.weight = base_model.wte.weight
        lora_model.wpe.weight = base_model.wpe.weight
        lora_model.lm_head.weight = base_model.lm_head.weight
        mx.eval(lora_model.parameters())

        freeze_except_lora(lora_model)
        n_train = count_params(lora_model)
        if verbose:
            print(f"  Trainable params: {n_train:,}")
        train(lora_model, domain_train, domain_val, steps=FINETUNE_STEPS,
              batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=300)
        lora_model.unfreeze()
        return lora_model

    lora_a = finetune_lora(train_a, val_a, "A (a-m)")
    lora_b = finetune_lora(train_b, val_b, "B (n-z)")

    # === 4. Extract deltas and compute metrics ===
    deltas_a = get_deltas(lora_a)
    deltas_b = get_deltas(lora_b)

    shared_deltas, unique_deltas, decomp_metrics = decompose_deltas([deltas_a, deltas_b])

    # Effective rank of deltas
    eff_rank_a = compute_effective_rank(deltas_a)
    eff_rank_b = compute_effective_rank(deltas_b)
    avg_eff_rank = (eff_rank_a + eff_rank_b) / 2

    if verbose:
        print(f"\n--- Decomposition metrics ---")
        print(f"  Shared fraction: {decomp_metrics['shared_fraction']:.3f}")
        print(f"  Cosine similarity: {decomp_metrics['cosine_similarity']:.4f}")
        print(f"  Effective rank (avg): {avg_eff_rank:.2f}")

    # === 5. Task arithmetic (base + mean deltas) ===
    ta_deltas = []
    for m_idx in range(len(deltas_a)):
        l_idx, name, d_a = deltas_a[m_idx]
        _, _, d_b = deltas_b[m_idx]
        ta_deltas.append((l_idx, name, (d_a + d_b) / 2))

    ta_model = apply_deltas_to_base(base_model, ta_deltas, V)
    ta_a = evaluate(ta_model, val_a, BATCH_SIZE)
    ta_b = evaluate(ta_model, val_b, BATCH_SIZE)
    ta_avg = (ta_a + ta_b) / 2

    # === 6. Concatenated LoRA + calibrated router ===
    if verbose:
        print("\n--- Concat + calibrated router ---")
    base_for_concat = get_model("gpt", vocab_size=V, **BASE)
    mx.eval(base_for_concat.parameters())
    copy_weights(base_model, base_for_concat)

    concat_model = RoutedDeltaGPT(base_for_concat, [deltas_a, deltas_b], V, top_k=2)
    mx.eval(concat_model.parameters())
    calibrate_router(concat_model, train_a, train_b,
                     steps=ROUTER_CAL_STEPS, lr=LR, seed=seed)
    cc_a = evaluate(concat_model, val_a, BATCH_SIZE)
    cc_b = evaluate(concat_model, val_b, BATCH_SIZE)
    concat_avg = (cc_a + cc_b) / 2

    # === 7. Dead neuron profiling on composed model ===
    dead_rate = profile_dead_neurons(ta_model, joint_val)
    if verbose:
        print(f"  Dead neuron rate (task arith): {dead_rate:.1%}")

    # === 8. Delta norm analysis ===
    delta_norm_a = sum(mx.sum(d[2] ** 2).item() for d in deltas_a) ** 0.5
    delta_norm_b = sum(mx.sum(d[2] ** 2).item() for d in deltas_b) ** 0.5

    # Compute LoRA trainable params for this rank
    # fc1: A is (d, r), B is (r, 4d) -> d*r + r*4d
    # fc2: A is (4d, r), B is (r, d) -> 4d*r + r*d
    # per layer: d*r + r*4d + 4d*r + r*d = 2*d*r + 2*4d*r = 2*r*(d + 4d) = 10*d*r
    d = BASE['n_embd']
    lora_params = BASE['n_layer'] * 2 * rank * (d + 4 * d)

    result = {
        'rank': rank,
        'seed': seed,
        'joint_avg': joint_avg,
        'ta_avg': ta_avg,
        'concat_avg': concat_avg,
        'ta_vs_joint': (ta_avg - joint_avg) / joint_avg * 100,
        'concat_vs_joint': (concat_avg - joint_avg) / joint_avg * 100,
        'shared_fraction': decomp_metrics['shared_fraction'],
        'cosine_similarity': decomp_metrics['cosine_similarity'],
        'effective_rank': avg_eff_rank,
        'dead_rate': dead_rate,
        'delta_norm_a': delta_norm_a,
        'delta_norm_b': delta_norm_b,
        'lora_params': lora_params,
    }

    if verbose:
        print(f"\n--- Summary (r={rank}, seed={seed}) ---")
        print(f"  Joint avg:        {joint_avg:.4f}")
        print(f"  Task arith avg:   {ta_avg:.4f} ({result['ta_vs_joint']:+.2f}% vs joint)")
        print(f"  Concat+cal avg:   {concat_avg:.4f} ({result['concat_vs_joint']:+.2f}% vs joint)")
        print(f"  Shared fraction:  {decomp_metrics['shared_fraction']:.3f}")
        print(f"  Cosine sim:       {decomp_metrics['cosine_similarity']:.4f}")
        print(f"  Effective rank:   {avg_eff_rank:.2f}")
        print(f"  Dead neuron rate: {dead_rate:.1%}")
        print(f"  LoRA params:      {lora_params:,}")

    return result


# ── Full Sweep ──────────────────────────────────────────────────────────────

def run_sweep(seeds=(42, 123, 7)):
    """Run the full rank sweep across all ranks and seeds."""
    all_results = []

    for rank in RANKS:
        for seed in seeds:
            result = run_single_rank(rank, seed=seed, verbose=True)
            all_results.append(result)

    # Aggregate by rank
    print(f"\n\n{'='*80}")
    print("RANK SWEEP AGGREGATE")
    print(f"{'='*80}")

    print(f"\n{'Rank':>5} {'Params':>8} {'TA vs Joint':>12} {'CC vs Joint':>12} "
          f"{'SharedFrac':>11} {'CosSim':>8} {'EffRank':>8} {'DeadRate':>9}")
    print("-" * 85)

    rank_means = {}
    for rank in RANKS:
        rank_results = [r for r in all_results if r['rank'] == rank]
        ta_gaps = [r['ta_vs_joint'] for r in rank_results]
        cc_gaps = [r['concat_vs_joint'] for r in rank_results]
        shared_fracs = [r['shared_fraction'] for r in rank_results]
        cos_sims = [r['cosine_similarity'] for r in rank_results]
        eff_ranks = [r['effective_rank'] for r in rank_results]
        dead_rates = [r['dead_rate'] for r in rank_results]
        lora_params = rank_results[0]['lora_params']

        means = {
            'ta_gap': statistics.mean(ta_gaps),
            'cc_gap': statistics.mean(cc_gaps),
            'shared_frac': statistics.mean(shared_fracs),
            'cos_sim': statistics.mean(cos_sims),
            'eff_rank': statistics.mean(eff_ranks),
            'dead_rate': statistics.mean(dead_rates),
            'lora_params': lora_params,
        }
        rank_means[rank] = means

        print(f"{rank:>5} {lora_params:>8,} {means['ta_gap']:>+11.2f}% "
              f"{means['cc_gap']:>+11.2f}% {means['shared_frac']:>10.3f} "
              f"{means['cos_sim']:>8.4f} {means['eff_rank']:>8.2f} "
              f"{means['dead_rate']:>8.1%}")

    # === Kill criteria checks ===
    print(f"\n{'='*80}")
    print("KILL CRITERIA CHECKS")
    print(f"{'='*80}")

    # Kill 1: all ranks within 1% of each other (composition quality)
    ta_gaps = [rank_means[r]['ta_gap'] for r in RANKS]
    gap_range = max(ta_gaps) - min(ta_gaps)
    print(f"\n1. Composition quality range across ranks: {gap_range:.2f}pp")
    if gap_range < 1.0:
        print(f"   KILL: All ranks within {gap_range:.2f}pp (<1pp threshold)")
    else:
        print(f"   PASS: Range {gap_range:.2f}pp exceeds 1pp threshold")

    cc_gaps = [rank_means[r]['cc_gap'] for r in RANKS]
    cc_range = max(cc_gaps) - min(cc_gaps)
    print(f"   (Concat+cal range: {cc_range:.2f}pp)")

    # Kill 2: orthogonality does not correlate with rank (r^2 < 0.2)
    cos_sims = [rank_means[r]['cos_sim'] for r in RANKS]
    log_ranks = [math.log2(r) for r in RANKS]

    # Compute r^2 (coefficient of determination)
    n = len(RANKS)
    mean_x = sum(log_ranks) / n
    mean_y = sum(cos_sims) / n
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_ranks, cos_sims))
    ss_xx = sum((x - mean_x) ** 2 for x in log_ranks)
    ss_yy = sum((y - mean_y) ** 2 for y in cos_sims)

    if ss_xx > 0 and ss_yy > 0:
        r_corr = ss_xy / (ss_xx * ss_yy) ** 0.5
        r_squared = r_corr ** 2
        slope = ss_xy / ss_xx
    else:
        r_corr = 0.0
        r_squared = 0.0
        slope = 0.0

    print(f"\n2. Orthogonality vs log2(rank) correlation:")
    print(f"   r = {r_corr:.4f}, r^2 = {r_squared:.4f}, slope = {slope:.6f}")
    if r_squared < 0.2:
        print(f"   KILL: r^2 = {r_squared:.4f} < 0.2 threshold")
    else:
        print(f"   PASS: r^2 = {r_squared:.4f} >= 0.2 threshold")

    # Additional analyses
    print(f"\n{'='*80}")
    print("ADDITIONAL ANALYSES")
    print(f"{'='*80}")

    # Effective rank vs nominal rank
    print(f"\n3. Effective rank vs nominal rank:")
    for r in RANKS:
        eff = rank_means[r]['eff_rank']
        ratio = eff / r
        print(f"   r={r:>2}: effective={eff:.2f}, ratio={ratio:.2f}")

    # Dead neuron rate vs rank
    print(f"\n4. Dead neuron rate vs rank:")
    dead_rates = [rank_means[r]['dead_rate'] for r in RANKS]
    for r in RANKS:
        print(f"   r={r:>2}: dead_rate={rank_means[r]['dead_rate']:.1%}")

    # Shared fraction vs rank
    print(f"\n5. Shared fraction vs rank:")
    for r in RANKS:
        print(f"   r={r:>2}: shared_frac={rank_means[r]['shared_frac']:.3f}")

    # Delta norm vs rank
    print(f"\n6. Delta norm growth with rank:")
    for r in RANKS:
        rank_results = [res for res in all_results if res['rank'] == r]
        avg_norm = statistics.mean(
            [(res['delta_norm_a'] + res['delta_norm_b']) / 2 for res in rank_results]
        )
        print(f"   r={r:>2}: avg_delta_norm={avg_norm:.4f}")

    # Rate-distortion summary
    print(f"\n7. Rate-distortion summary (rank vs quality):")
    print(f"   {'Rank':>5} {'LoRA params':>12} {'TA gap':>8} {'CC gap':>8} {'Quality$':>10}")
    for r in RANKS:
        m = rank_means[r]
        # Quality cost = params per percentage point of quality
        quality_gap = abs(m['ta_gap']) if m['ta_gap'] > 0 else 0.01
        quality_cost = m['lora_params'] / quality_gap
        print(f"   {r:>5} {m['lora_params']:>12,} {m['ta_gap']:>+7.2f}% "
              f"{m['cc_gap']:>+7.2f}% {quality_cost:>10,.0f}")

    return all_results, rank_means


if __name__ == "__main__":
    run_sweep()
