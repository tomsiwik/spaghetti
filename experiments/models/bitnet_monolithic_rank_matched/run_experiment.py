#!/usr/bin/env python3
"""
BitNet-SOLE vs Monolithic at Matched Total Parameters

The prior experiment (bitnet_sole_vs_monolithic) showed SOLE routed wins 4/5
domains over rank-16 monolithic. But SOLE uses 5x more total parameters
(5 x r=16 = 108M vs 1 x r=16 = 21.6M). A reviewer will dismiss this as
"more parameters = better."

This experiment trains a rank-80 monolithic LoRA (~108M params, matching SOLE's
total) on the same shuffled union data for the same 2000 gradient steps. We then
compare per-domain PPL against the SOLE routed results from the prior experiment.

If SOLE still wins per-domain: composition is genuinely better than scaling rank.
If monolithic wins: SOLE's value is purely operational (modularity), not quality.

Kill criterion:
  rank-80 monolithic beats SOLE routed on >60% of per-domain metrics
  (3+ out of 5 domains)

Design:
  - Reuse data from prior experiment (symlink or copy)
  - Reuse base PPL and SOLE routed PPL from prior results.json
  - Train ONLY the rank-80 monolithic condition
  - Same hyperparameters: lr=1e-4, seq_len=128, 2000 steps, seed=42

Platform: Apple Silicon MLX, $0 compute.
"""

import json
import math
import os
import sys
import time
import random
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm import load
from mlx_lm.models.bitlinear_layers import BitLinear


# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 80  # Key change: rank-80 to match SOLE's total params
LORA_SCALE = 20.0  # Same scaling as prior experiment
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 128
LEARNING_RATE = 1e-4
VAL_BATCHES = 50
MONOLITHIC_STEPS = 2000

EXPERIMENT_DIR = Path(__file__).parent
PRIOR_DIR = EXPERIMENT_DIR.parent / "bitnet_sole_vs_monolithic"
DATA_DIR = PRIOR_DIR / "data"  # Reuse prior experiment's data directly
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

DOMAINS = ["medical", "code", "math", "legal", "creative"]

# Load prior results for comparison
PRIOR_RESULTS_FILE = PRIOR_DIR / "results.json"


# ===========================================================================
# Ternary weight unpacking (from proven pipeline)
# ===========================================================================
def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    """Unpack uint8-packed ternary weights to bfloat16 dense matrix."""
    w0 = (packed_weights & 3).astype(mx.bfloat16) - 1
    w1 = ((packed_weights >> 2) & 3).astype(mx.bfloat16) - 1
    w2 = ((packed_weights >> 4) & 3).astype(mx.bfloat16) - 1
    w3 = ((packed_weights >> 6) & 3).astype(mx.bfloat16) - 1
    unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:out_features]
    scale = weight_scale.astype(mx.bfloat16)
    if invert_scale:
        unpacked = unpacked / scale
    else:
        unpacked = unpacked * scale
    return unpacked


def replace_bitlinear_with_linear(model):
    """Replace all BitLinear layers with standard nn.Linear for training."""
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                unpacked_w = unpack_ternary(
                    module.weight, module.out_features,
                    module.weight_scale, module.invert_weight_scales,
                )
                has_bias = module.bias is not None
                linear = nn.Linear(module.in_features, module.out_features, bias=has_bias)
                linear.weight = unpacked_w
                if has_bias:
                    linear.bias = module.bias
                updates.append((key, linear))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    print(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


# ===========================================================================
# Ternary LoRA with STE (from proven pipeline)
# ===========================================================================
class TernaryLoRALinear(nn.Module):
    """LoRA layer with STE ternary quantization of A/B matrices."""

    def __init__(self, base_linear, r=16, scale=20.0):
        super().__init__()
        self.linear = base_linear
        self.r = r
        self.scale = scale

        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]

        s = 1.0 / math.sqrt(in_features)
        self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_features, r))
        self.lora_b = mx.zeros((r, out_features))

    def _ste_ternary(self, W):
        """Ternary quantization with Straight-Through Estimator."""
        alpha = mx.mean(mx.abs(W)) + 1e-10
        W_scaled = W / alpha
        W_q = mx.clip(mx.round(W_scaled), -1.0, 1.0) * alpha
        return W + mx.stop_gradient(W_q - W)

    def __call__(self, x):
        base_out = self.linear(x)
        A = self._ste_ternary(self.lora_a)
        B = self._ste_ternary(self.lora_b)
        lora_out = (x @ A) @ B * self.scale
        return base_out + lora_out


def apply_ternary_lora(model, rank=80, scale=20.0):
    """Apply ternary LoRA to all linear layers in transformer blocks."""
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        lora_updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora_layer = TernaryLoRALinear(module, r=rank, scale=scale)
                lora_updates.append((key, lora_layer))
                count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    print(f"  Applied Ternary LoRA (r={rank}) to {count} layers")
    return model


# ===========================================================================
# PPL evaluation (identical to prior experiment)
# ===========================================================================
def compute_ppl(model, tokenizer, data_path: Path, max_batches: int = 50):
    """Compute perplexity on validation data."""
    valid_path = data_path / "valid.jsonl"
    if not valid_path.exists():
        return float("inf")

    texts = []
    with open(valid_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    total_loss = 0.0
    total_tokens = 0

    for text in texts[:max_batches]:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:MAX_SEQ_LENGTH + 1]

        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]

        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)

        total_loss += loss.item()
        total_tokens += y.size

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 100))


# ===========================================================================
# Training (identical to prior experiment)
# ===========================================================================
def train_adapter(model, tokenizer, data_dir, n_steps, label):
    """Train a single adapter. Returns training metrics."""
    train_texts = []
    with open(data_dir / "train.jsonl") as f:
        for line in f:
            train_texts.append(json.loads(line)["text"])

    train_tokens = []
    for text in train_texts:
        toks = tokenizer.encode(text)
        if len(toks) > 2:
            train_tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))

    print(f"  {len(train_tokens)} training sequences, {n_steps} steps")

    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    t_start = time.time()
    losses = []

    for step in range(n_steps):
        idx = step % len(train_tokens)
        tokens = train_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()
        losses.append(loss_val)

        if (step + 1) % 100 == 0 or step == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - t_start
            eta = elapsed / (step + 1) * (n_steps - step - 1)
            print(f"    [{label}] Step {step+1}/{n_steps}: loss={loss_val:.4f} "
                  f"(avg50={avg:.4f}) elapsed={elapsed:.0f}s eta={eta:.0f}s")

    train_time = time.time() - t_start
    first_50 = sum(losses[:50]) / 50
    last_50 = sum(losses[-50:]) / 50
    converged = last_50 < first_50 * 0.95

    print(f"  [{label}] Done in {train_time:.1f}s. Loss: {first_50:.4f} -> {last_50:.4f} "
          f"({'converged' if converged else 'NOT converged'})")

    return {
        "label": label,
        "n_steps": n_steps,
        "train_time_s": round(train_time, 1),
        "first_50_avg_loss": round(first_50, 4),
        "last_50_avg_loss": round(last_50, 4),
        "converged": converged,
        "all_losses": [round(l, 4) for l in losses],
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    # Load prior results
    if not PRIOR_RESULTS_FILE.exists():
        print(f"FATAL: Prior results not found at {PRIOR_RESULTS_FILE}")
        print("Run bitnet_sole_vs_monolithic first.")
        sys.exit(1)

    with open(PRIOR_RESULTS_FILE) as f:
        prior = json.load(f)

    sole_routed_ppls = prior["sole_routed_ppls"]
    base_ppls = prior["base_ppls"]
    mono_r16_ppls = prior["mono_ppls"]
    prior_trainable = prior["trainable_params_per_expert"]

    print("=" * 70)
    print("BitNet-SOLE vs Monolithic: RANK-MATCHED Parameter Fairness Test")
    print("=" * 70)
    print(f"\nPrior SOLE routed PPLs (from {PRIOR_RESULTS_FILE.name}):")
    for d in DOMAINS:
        print(f"  {d}: {sole_routed_ppls[d]:.2f}")
    print(f"Prior mono r=16 trainable params: {prior_trainable:,}")
    print(f"SOLE total params: 5 x {prior_trainable:,} = {5*prior_trainable:,}")
    print(f"Target rank-80 should match ~{5*prior_trainable:,} params")

    results = {
        "experiment": "bitnet_monolithic_rank_matched",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "monolithic_steps": MONOLITHIC_STEPS,
        "domains": DOMAINS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "design": "rank_matched_total_params",
        "prior_experiment": "bitnet_sole_vs_monolithic",
        "sole_routed_ppls_prior": sole_routed_ppls,
        "base_ppls_prior": base_ppls,
        "mono_r16_ppls_prior": mono_r16_ppls,
        "sole_total_params": 5 * prior_trainable,
    }

    # ==================================================================
    # Phase 0: Load model and unpack
    # ==================================================================
    print("\n[Phase 0] Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    print("  Unpacking ternary weights...")
    t1 = time.time()
    model = replace_bitlinear_with_linear(model)
    unpack_time = time.time() - t1
    print(f"  Unpacked in {unpack_time:.1f}s")

    results["load_time_s"] = round(load_time + unpack_time, 1)

    # ==================================================================
    # Phase 1: Verify data exists
    # ==================================================================
    print("\n[Phase 1] Verifying prior experiment data...")
    mono_data_dir = DATA_DIR / "monolithic"
    if not (mono_data_dir / "train.jsonl").exists():
        print(f"FATAL: Monolithic training data not found at {mono_data_dir}")
        sys.exit(1)

    with open(mono_data_dir / "train.jsonl") as f:
        n_train = sum(1 for _ in f)
    print(f"  Monolithic training data: {n_train} samples")

    for d in DOMAINS:
        vp = DATA_DIR / d / "valid.jsonl"
        if not vp.exists():
            print(f"FATAL: Validation data missing for {d}")
            sys.exit(1)
    print("  All domain validation data present.")
    results["n_train_samples"] = n_train

    # ==================================================================
    # Phase 2: Verify base PPLs match prior experiment
    # ==================================================================
    print("\n[Phase 2] Verifying base model PPL (sanity check)...")
    base_ppls_new = {}
    for d in DOMAINS:
        ppl = compute_ppl(model, tokenizer, DATA_DIR / d)
        base_ppls_new[d] = round(ppl, 4)
        prior_ppl = base_ppls[d]
        diff_pct = abs(ppl - prior_ppl) / prior_ppl * 100
        status = "OK" if diff_pct < 1.0 else "MISMATCH"
        print(f"  {d}: {ppl:.4f} (prior: {prior_ppl:.4f}, diff: {diff_pct:.2f}%) [{status}]")

    results["base_ppls_verified"] = base_ppls_new

    # ==================================================================
    # Phase 3: Apply rank-80 LoRA and train
    # ==================================================================
    print(f"\n[Phase 3] Applying rank-{LORA_RANK} ternary LoRA...")
    model = apply_ternary_lora(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Freeze base, unfreeze LoRA
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"  Trainable LoRA parameters (r={LORA_RANK}): {trainable:,}")
    print(f"  SOLE total params (5 x r=16):              {5*prior_trainable:,}")
    param_ratio = trainable / (5 * prior_trainable)
    print(f"  Ratio: {param_ratio:.2f}x")
    results["trainable_params_r80"] = trainable
    results["param_ratio_vs_sole"] = round(param_ratio, 4)

    # Verify gradients
    print("  Verifying gradient computation...")
    def test_loss(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")
    test_grad = nn.value_and_grad(model, test_loss)
    x_test = mx.array([[1, 2, 3, 4, 5]])
    y_test = mx.array([[2, 3, 4, 5, 6]])
    try:
        l, g = test_grad(model, x_test, y_test)
        mx.eval(l)
        print(f"  Gradient check PASSED (loss={l.item():.4f})")
    except Exception as e:
        print(f"  Gradient check FAILED: {e}")
        results["error"] = f"Gradient computation failed: {e}"
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        return

    # Train monolithic rank-80 on shuffled union data
    print(f"\n[Phase 3b] Training rank-{LORA_RANK} monolithic on union data ({MONOLITHIC_STEPS} steps)...")
    train_result = train_adapter(
        model, tokenizer, mono_data_dir,
        n_steps=MONOLITHIC_STEPS,
        label=f"MONO-r{LORA_RANK}",
    )
    results["train_result"] = {k: v for k, v in train_result.items() if k != "all_losses"}
    results["train_losses_sampled"] = train_result["all_losses"][::100]  # every 100th step

    # ==================================================================
    # Phase 4: Evaluate rank-80 monolithic on all domains
    # ==================================================================
    print(f"\n[Phase 4] Evaluating rank-{LORA_RANK} monolithic per-domain PPL...")
    mono_r80_ppls = {}
    for d in DOMAINS:
        ppl = compute_ppl(model, tokenizer, DATA_DIR / d)
        mono_r80_ppls[d] = round(ppl, 4)
        imp_vs_base = (base_ppls[d] - ppl) / base_ppls[d] * 100
        print(f"  {d}: PPL={ppl:.2f} (base={base_ppls[d]:.2f}, {imp_vs_base:+.1f}% vs base)")
    results["mono_r80_ppls"] = mono_r80_ppls

    # ==================================================================
    # Phase 5: Head-to-head comparison
    # ==================================================================
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD: SOLE Routed vs Monolithic r=80 (parameter-matched)")
    print("=" * 70)

    sole_wins = 0
    mono_wins = 0
    domain_comparison = {}

    print(f"\n  {'Domain':<12} {'Base':>8} {'SOLE r=16':>10} {'Mono r=16':>10} {'Mono r=80':>10} {'Winner':>8} {'Gap':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    for d in DOMAINS:
        sole_ppl = sole_routed_ppls[d]
        mono80_ppl = mono_r80_ppls[d]
        mono16_ppl = mono_r16_ppls[d]
        base_ppl = base_ppls[d]

        winner = "SOLE" if sole_ppl <= mono80_ppl else "MONO-80"
        gap_pct = (sole_ppl - mono80_ppl) / mono80_ppl * 100

        if sole_ppl <= mono80_ppl:
            sole_wins += 1
        else:
            mono_wins += 1

        domain_comparison[d] = {
            "base_ppl": base_ppl,
            "sole_routed_ppl": sole_ppl,
            "mono_r16_ppl": mono16_ppl,
            "mono_r80_ppl": mono80_ppl,
            "winner_vs_sole": winner,
            "gap_pct_vs_sole": round(gap_pct, 2),
            "mono_r80_vs_r16_pct": round((mono80_ppl - mono16_ppl) / mono16_ppl * 100, 2),
            "mono_r80_vs_base_pct": round((base_ppl - mono80_ppl) / base_ppl * 100, 2),
        }

        print(f"  {d:<12} {base_ppl:>8.2f} {sole_ppl:>10.2f} {mono16_ppl:>10.2f} {mono80_ppl:>10.2f} {winner:>8} {gap_pct:>+7.1f}%")

    results["domain_comparison"] = domain_comparison
    results["sole_wins"] = sole_wins
    results["mono_r80_wins"] = mono_wins

    # Averages
    avg_sole = sum(sole_routed_ppls[d] for d in DOMAINS) / len(DOMAINS)
    avg_mono80 = sum(mono_r80_ppls[d] for d in DOMAINS) / len(DOMAINS)
    avg_mono16 = sum(mono_r16_ppls[d] for d in DOMAINS) / len(DOMAINS)
    avg_base = sum(base_ppls[d] for d in DOMAINS) / len(DOMAINS)

    print(f"\n  {'Average':<12} {avg_base:>8.2f} {avg_sole:>10.2f} {avg_mono16:>10.2f} {avg_mono80:>10.2f}")

    overall_gap = (avg_sole - avg_mono80) / avg_mono80 * 100
    r80_vs_r16_gap = (avg_mono80 - avg_mono16) / avg_mono16 * 100

    results["avg_sole_routed_ppl"] = round(avg_sole, 4)
    results["avg_mono_r80_ppl"] = round(avg_mono80, 4)
    results["avg_mono_r16_ppl"] = round(avg_mono16, 4)
    results["avg_base_ppl"] = round(avg_base, 4)
    results["overall_gap_pct"] = round(overall_gap, 2)
    results["r80_vs_r16_gap_pct"] = round(r80_vs_r16_gap, 2)

    print(f"\n  SOLE routed vs Mono r=80 avg gap: {overall_gap:+.1f}%")
    print(f"  Mono r=80 vs Mono r=16 avg gap:   {r80_vs_r16_gap:+.1f}%")

    # ==================================================================
    # Phase 6: Kill criteria assessment
    # ==================================================================
    print("\n" + "=" * 70)
    print("KILL CRITERIA ASSESSMENT")
    print("=" * 70)

    # Kill if mono r=80 wins >60% of domains (3+ out of 5)
    k1_killed = mono_wins >= 3
    results["k1_killed"] = k1_killed
    results["k1_description"] = f"Mono r=80 wins {mono_wins}/5 domains (threshold: >=3 to kill)"

    print(f"\n  K1: rank-80 monolithic beats SOLE routed on >60% of per-domain metrics")
    print(f"      Mono r=80 wins: {mono_wins}/5 domains")
    print(f"      SOLE wins:      {sole_wins}/5 domains")
    print(f"      Threshold:      >=3 mono wins to KILL")
    print(f"      Result:         {'KILLED' if k1_killed else 'PASS'}")

    verdict = "KILLED" if k1_killed else "SUPPORTED"
    results["verdict"] = verdict

    print(f"\n  VERDICT: {verdict}")
    if not k1_killed:
        print(f"  Interpretation: SOLE routed wins {sole_wins}/5 domains even at matched")
        print(f"  total parameters. Specialization genuinely beats scaling rank.")
    else:
        print(f"  Interpretation: At matched parameters, monolithic is competitive.")
        print(f"  SOLE's value is operational (modularity, no forgetting), not quality.")

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE}")

    # Print total runtime
    total_time = results.get("load_time_s", 0) + train_result["train_time_s"]
    print(f"  Total runtime: ~{total_time/60:.0f} minutes")


if __name__ == "__main__":
    main()
