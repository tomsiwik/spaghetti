"""Concat+Calibrate N=5 calibration budget sweep.

Hypothesis: The N=5 concat+cal failure (+5.07% vs joint, with 100 cal steps) is
caused by router underfitting, not a fundamental mechanism failure.

Sweep: ROUTER_CAL_STEPS in {100, 200, 300, 500} at N=5, 3 seeds each.

Kill criteria:
- 500-step calibration does not close gap to <3% vs joint at N=5
- No calibration budget achieves concat+cal beating simple average at N=5

Baselines from merging bakeoff (3-seed aggregate, N=5):
- Joint training: 0.4984
- Simple average: 0.5150 (+3.33%)
- Concat+cal 100 steps: 0.5237 (+5.07%)
"""

import copy
import random
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.models import get_model
from experiments.data import (
    load_names, CharTokenizer, CharDataset,
    domain_split, train_val_split,
)
from experiments.train import train, evaluate, ntp_loss
from experiments.models.lora_merging_bakeoff.merging_methods import (
    extract_deltas,
    merge_simple_average,
)
from experiments.models.lora_procrustes.test_lora_procrustes import (
    RoutedDeltaGPT,
    copy_weights,
    count_params,
    freeze_except_lora,
    get_deltas as get_deltas_list,
)


# ── Config ──────────────────────────────────────────────────────────────────

BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
LORA_RANK = 8
LORA_ALPHA = 1.0
PRETRAIN_STEPS = 300
FINETUNE_STEPS = 300
BATCH_SIZE = 32
LR = 3e-3
N_DOMAINS = 5

# The sweep
CAL_STEPS_SWEEP = [100, 200, 300, 500]
SEEDS = (42, 123, 7)


# ── Helpers (reused from merging bakeoff) ──────────────────────────────────

def train_joint(train_datasets, val_datasets, vocab_size, steps_per_domain, seed):
    """Train a single GPT jointly on all domains (oracle baseline)."""
    model = get_model("gpt", vocab_size=vocab_size, **BASE)
    mx.eval(model.parameters())

    total_steps = len(train_datasets) * steps_per_domain
    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(model, ntp_loss)

    domain_names = list(train_datasets.keys())

    for step in range(1, total_steps + 1):
        domain = domain_names[(step - 1) % len(domain_names)]
        inputs, targets = train_datasets[domain].get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 500 == 0 or step == total_steps:
            print(f"  step {step:4d}/{total_steps} | loss {loss.item():.4f}")

    results = {}
    for name, val_ds in val_datasets.items():
        results[name] = evaluate(model, val_ds, BATCH_SIZE)
    results["avg"] = sum(results.values()) / len(results)
    return results


def pretrain_base(joint_train, vocab_size, seed):
    """Pretrain base GPT model on joint data."""
    base = get_model("gpt", vocab_size=vocab_size, **BASE)
    mx.eval(base.parameters())
    train(base, joint_train, steps=PRETRAIN_STEPS, batch_size=BATCH_SIZE,
          lr=LR, seed=seed, log_every=300)
    return base


def finetune_lora(base_model, train_ds, val_ds, vocab_size, seed):
    """Fine-tune a LoRA model on a single domain."""
    lora = get_model("lora_gpt", vocab_size=vocab_size, **BASE,
                     lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA)
    mx.eval(lora.parameters())

    for l_idx in range(BASE['n_layer']):
        bl = base_model.layers[l_idx]
        ll = lora.layers[l_idx]
        ll.attn.wq.weight = bl.attn.wq.weight
        ll.attn.wk.weight = bl.attn.wk.weight
        ll.attn.wv.weight = bl.attn.wv.weight
        ll.attn.wo.weight = bl.attn.wo.weight
        ll.mlp.fc1.linear.weight = bl.mlp.fc1.weight
        ll.mlp.fc2.linear.weight = bl.mlp.fc2.weight
    lora.wte.weight = base_model.wte.weight
    lora.wpe.weight = base_model.wpe.weight
    lora.lm_head.weight = base_model.lm_head.weight
    mx.eval(lora.parameters())

    freeze_except_lora(lora)
    train(lora, train_ds, val_ds, steps=FINETUNE_STEPS,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=300)
    lora.unfreeze()
    return lora


def eval_simple_average(base_model, all_delta_dicts, val_datasets, vocab_size):
    """Evaluate simple average merging (task arithmetic)."""
    from experiments.models.lora_merging_bakeoff.merging_methods import apply_merged_deltas
    merged = merge_simple_average(all_delta_dicts)
    model = apply_merged_deltas(base_model, merged, vocab_size)
    results = {}
    for name, val_ds in val_datasets.items():
        results[name] = evaluate(model, val_ds, BATCH_SIZE)
    results["avg"] = sum(results.values()) / len(results)
    return results


def eval_concat_calibrated(base_model, delta_lists, train_datasets,
                           val_datasets, vocab_size, seed, cal_steps):
    """Evaluate concat+calibrate with a specified number of calibration steps."""
    base_copy = get_model("gpt", vocab_size=vocab_size, **BASE)
    mx.eval(base_copy.parameters())
    copy_weights(base_model, base_copy)

    concat_model = RoutedDeltaGPT(
        base_copy, delta_lists, vocab_size, top_k=2)
    mx.eval(concat_model.parameters())

    # Calibrate router on mixed-domain data
    all_train_ds = list(train_datasets.values())

    concat_model.freeze()
    for router in concat_model.routers:
        router.unfreeze()

    rng = random.Random(seed)
    optimizer = optim.Adam(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(concat_model, ntp_loss)

    losses = []
    for step in range(1, cal_steps + 1):
        ds = all_train_ds[step % len(all_train_ds)]
        inputs, targets = ds.get_batch(BATCH_SIZE, rng)
        loss, grads = loss_and_grad(concat_model, inputs, targets)
        optimizer.update(concat_model, grads)
        mx.eval(concat_model.parameters(), optimizer.state)
        losses.append(loss.item())

    concat_model.unfreeze()

    results = {}
    for name, val_ds in val_datasets.items():
        results[name] = evaluate(concat_model, val_ds, BATCH_SIZE)
    results["avg"] = sum(results.values()) / len(results)
    results["final_cal_loss"] = losses[-1] if losses else None

    # Also record loss at key checkpoints for learning curve
    checkpoints = {}
    for ck in [50, 100, 200, 300, 500]:
        if ck <= len(losses):
            checkpoints[ck] = statistics.mean(losses[max(0, ck-10):ck])
    results["loss_checkpoints"] = checkpoints

    return results


# ── Main Experiment ─────────────────────────────────────────────────────────

def run_single_seed(seed=42):
    """Run the calibration budget sweep for one seed at N=5."""
    print(f"\n{'='*70}")
    print(f"CALIBRATION BUDGET SWEEP (seed={seed}, N={N_DOMAINS})")
    print(f"{'='*70}")

    mx.random.seed(seed)
    t0 = time.time()

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    splits = domain_split(docs, method="quintary")

    all_train, all_val = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_train, tokenizer, BASE["block_size"])

    train_datasets = {}
    val_datasets = {}
    for d_name, d_docs in splits.items():
        d_train, d_val = train_val_split(d_docs, seed=seed)
        train_datasets[d_name] = CharDataset(d_train, tokenizer, BASE["block_size"])
        val_datasets[d_name] = CharDataset(d_val, tokenizer, BASE["block_size"])

    results = {}

    # === 1. Joint training baseline ===
    print("\n--- Joint training baseline ---")
    results["joint"] = train_joint(
        train_datasets, val_datasets, V, FINETUNE_STEPS, seed)
    print(f"  Joint avg: {results['joint']['avg']:.4f}")

    # === 2. Pretrain base ===
    print("\n--- Pretraining base model ---")
    base_model = pretrain_base(joint_train, V, seed)

    # === 3. Fine-tune LoRA per domain ===
    delta_dicts = {}
    delta_lists = []

    domain_names = list(splits.keys())
    for i, d_name in enumerate(domain_names):
        print(f"\n--- Fine-tuning LoRA for {d_name} ---")
        lora = finetune_lora(
            base_model, train_datasets[d_name], val_datasets[d_name],
            V, seed + i)
        delta_dicts[d_name] = extract_deltas(lora)
        delta_lists.append(get_deltas_list(lora))

    all_delta_dicts = [delta_dicts[d] for d in domain_names]

    # === 4. Simple average baseline ===
    print("\n--- Simple Average ---")
    results["simple_avg"] = eval_simple_average(
        base_model, all_delta_dicts, val_datasets, V)
    print(f"  Simple avg: {results['simple_avg']['avg']:.4f}")

    # === 5. Concat+Calibrate at each budget ===
    for cal_steps in CAL_STEPS_SWEEP:
        key = f"concat_cal_{cal_steps}"
        print(f"\n--- Concat+Calibrate ({cal_steps} steps) ---")
        t_cc = time.time()
        results[key] = eval_concat_calibrated(
            base_model, delta_lists, train_datasets,
            val_datasets, V, seed, cal_steps)
        t_cc_cost = time.time() - t_cc
        results[key]["cal_time"] = t_cc_cost
        print(f"  cal_steps={cal_steps}: {results[key]['avg']:.4f} ({t_cc_cost:.1f}s)")

    # === Summary ===
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"SUMMARY (seed={seed}, elapsed={elapsed:.1f}s)")
    print(f"{'='*70}")

    joint_avg = results["joint"]["avg"]
    print(f"\n{'Method':<25} {'avg val_loss':>12} {'vs joint':>10}")
    print("-" * 50)

    method_order = ["joint", "simple_avg"] + [f"concat_cal_{s}" for s in CAL_STEPS_SWEEP]
    for method in method_order:
        r = results[method]
        if method == "joint":
            gap_str = "baseline"
        else:
            gap = (r["avg"] - joint_avg) / joint_avg * 100
            gap_str = f"{gap:+.2f}%"
        print(f"{method:<25} {r['avg']:>12.4f} {gap_str:>10}")

    return results


def run_full_sweep():
    """Run calibration budget sweep across all seeds and aggregate."""
    all_results = {}
    for seed in SEEDS:
        all_results[seed] = run_single_seed(seed)

    # ── Aggregate ──
    print(f"\n\n{'='*70}")
    print(f"MULTI-SEED AGGREGATE (N={N_DOMAINS}, seeds={SEEDS})")
    print(f"{'='*70}")

    methods = ["joint", "simple_avg"] + [f"concat_cal_{s}" for s in CAL_STEPS_SWEEP]
    joint_avgs = [all_results[s]["joint"]["avg"] for s in SEEDS]
    joint_mean = statistics.mean(joint_avgs)

    print(f"\n{'Method':<25} {'mean loss':>10} {'std':>8} {'vs joint':>10} {'vs avg':>10}")
    print("-" * 68)

    method_stats = {}
    simple_avg_mean = None

    for method in methods:
        avgs = [all_results[s][method]["avg"] for s in SEEDS]
        mean_avg = statistics.mean(avgs)
        std_avg = statistics.stdev(avgs) if len(avgs) > 1 else 0.0

        if method == "joint":
            gap_str = "baseline"
            vs_avg_str = ""
        else:
            gap = (mean_avg - joint_mean) / joint_mean * 100
            gap_str = f"{gap:+.2f}%"
            if simple_avg_mean is not None:
                vs_avg = (mean_avg - simple_avg_mean) / simple_avg_mean * 100
                vs_avg_str = f"{vs_avg:+.2f}%"
            else:
                vs_avg_str = "ref"

        if method == "simple_avg":
            simple_avg_mean = mean_avg

        method_stats[method] = {"mean": mean_avg, "std": std_avg, "per_seed": avgs}
        print(f"{method:<25} {mean_avg:>10.4f} {std_avg:>8.4f} {gap_str:>10} {vs_avg_str:>10}")

    # ── Per-seed detail ──
    print(f"\n  Per-seed detail:")
    for method in methods:
        per_seed = [f"s{s}={all_results[s][method]['avg']:.4f}" for s in SEEDS]
        print(f"    {method:<23} {' | '.join(per_seed)}")

    # ── Calibration learning curves ──
    print(f"\n  Calibration loss checkpoints (last 10-step avg):")
    for cal_steps in CAL_STEPS_SWEEP:
        key = f"concat_cal_{cal_steps}"
        for seed in SEEDS:
            ckpts = all_results[seed][key].get("loss_checkpoints", {})
            ckpt_str = " | ".join(f"{k}:{v:.4f}" for k, v in sorted(ckpts.items()))
            print(f"    {key} seed={seed}: {ckpt_str}")

    # ── Kill Criteria ──
    print(f"\n{'='*70}")
    print("KILL CRITERIA EVALUATION")
    print(f"{'='*70}")

    # KC1: 500-step calibration does not close gap to <3% vs joint at N=5
    best_gap = None
    best_method = None
    for cal_steps in CAL_STEPS_SWEEP:
        key = f"concat_cal_{cal_steps}"
        gap = (method_stats[key]["mean"] - joint_mean) / joint_mean * 100
        print(f"\n  {key}: gap vs joint = {gap:+.2f}%")
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_method = key

    gap_500 = (method_stats["concat_cal_500"]["mean"] - joint_mean) / joint_mean * 100
    print(f"\n  KC1: 500-step gap vs joint = {gap_500:+.2f}% (threshold: <3%)")
    if gap_500 < 3.0:
        print(f"  --> PASS: 500-step calibration closes gap to <3%")
    else:
        print(f"  --> KILL: 500-step calibration does NOT close gap to <3%")

    # KC2: no calibration budget achieves concat+cal beating simple average at N=5
    sa_mean = method_stats["simple_avg"]["mean"]
    any_beats_avg = False
    for cal_steps in CAL_STEPS_SWEEP:
        key = f"concat_cal_{cal_steps}"
        cc_mean = method_stats[key]["mean"]
        beats = cc_mean < sa_mean
        margin = (cc_mean - sa_mean) / sa_mean * 100
        status = "BEATS" if beats else "LOSES TO"
        print(f"\n  KC2: {key} {status} simple_avg by {margin:+.2f}%")
        if beats:
            any_beats_avg = True

    if any_beats_avg:
        print(f"\n  --> PASS: at least one budget beats simple average")
    else:
        print(f"\n  --> KILL: no calibration budget beats simple average at N=5")

    # ── Trend analysis ──
    print(f"\n{'='*70}")
    print("TREND ANALYSIS")
    print(f"{'='*70}")

    gaps = []
    for cal_steps in CAL_STEPS_SWEEP:
        key = f"concat_cal_{cal_steps}"
        gap = (method_stats[key]["mean"] - joint_mean) / joint_mean * 100
        gaps.append((cal_steps, gap))
        print(f"  {cal_steps:>4} steps: {gap:+.2f}% vs joint")

    # Is the trend monotonically decreasing?
    decreasing = all(gaps[i][1] >= gaps[i+1][1] for i in range(len(gaps)-1))
    print(f"\n  Monotonically decreasing with more steps: {decreasing}")

    # Diminishing returns check
    if len(gaps) >= 2:
        first_improvement = gaps[0][1] - gaps[1][1]
        last_improvement = gaps[-2][1] - gaps[-1][1]
        print(f"  First improvement (100->200): {first_improvement:+.3f}pp")
        print(f"  Last improvement ({gaps[-2][0]}->{gaps[-1][0]}): {last_improvement:+.3f}pp")
        if last_improvement > 0 and first_improvement > 0:
            ratio = last_improvement / first_improvement
            print(f"  Diminishing returns ratio: {ratio:.2f} (1.0 = linear, <1.0 = diminishing)")

    return all_results, method_stats


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_full_sweep()
