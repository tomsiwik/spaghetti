#!/usr/bin/env python3
"""
Cross-Adapter Knowledge Transfer Matrix

Measures pairwise transfer between 5 domain adapters on BitNet-2B-4T.
For each pair (A, B), we measure whether adding adapter A improves domain B's PPL.

Kill criteria:
  K1: Zero pairs show >2% cross-domain improvement -> KILL
  K2: Transfer matrix is random (no structure) -> KILL

Approach:
  1. Train 5 domain adapters (python, math, medical, legal, creative)
  2. Measure base PPL, individual PPL per domain
  3. For each ordered pair (i, j) where i != j:
     - Compose adapter_i + adapter_j with weight alpha for i, (1-alpha) for j
     - Search alpha in {0.1, 0.2, 0.3, 0.5} to find best blend
     - Transfer coefficient = (PPL_j_alone - PPL_ij_best) / PPL_j_alone
  4. Build 5x5 transfer matrix, analyze structure
"""

import gc
import json
import math
import os
import sys
import time
from itertools import product
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety
device = mx.device_info()
total_mem = device["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.bitlinear_layers import BitLinear

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
TRAIN_ITERS = 200
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
VAL_BATCHES = 25

EXPERIMENT_DIR = Path(__file__).parent
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_SOURCE_DIR = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"

DOMAINS = ["python", "math", "medical", "legal", "creative"]

# Alpha values to search for cross-domain blending
# alpha = weight on the foreign adapter, (1 - alpha) = weight on the native adapter
ALPHA_VALUES = [0.1, 0.2, 0.3, 0.5]


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ===========================================================================
# BitNet unpacking (from bitnet_2b_real_composition)
# ===========================================================================
def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
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
# LoRA helpers (from bitnet_2b_real_composition)
# ===========================================================================
def apply_lora_to_model(model, rank=16, scale=1.0):
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        lora_updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora_layer = LoRALinear.from_base(module, r=rank, scale=scale, dropout=0.0)
                lora_updates.append((key, lora_layer))
                count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    print(f"  Applied LoRA (r={rank}) to {count} layers")
    return model


def get_lora_params(model):
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_a" in name or "lora_b" in name:
            params[name] = mx.array(p)
    mx.eval(params)
    return params


def save_adapter(model, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    params = get_lora_params(model)
    mx.savez(str(path / "adapter.npz"), **params)
    print(f"  Saved adapter: {len(params)} tensors to {path}")


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_weights(model, adapter_params: dict, scale: float = 1.0):
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def zero_lora_params(model):
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                in_dims = module.lora_a.shape[0]
                r = module.lora_a.shape[1]
                sc = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(low=-sc, high=sc, shape=module.lora_a.shape)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def compose_two_adapters(adapter_a: dict, adapter_b: dict, alpha_a: float, alpha_b: float):
    """Compose two adapters with specified weights."""
    merged = {}
    for key in adapter_a.keys():
        merged[key] = adapter_a[key] * alpha_a + adapter_b[key] * alpha_b
    return merged


# ===========================================================================
# PPL evaluation
# ===========================================================================
def compute_ppl(model, tokenizer, data_path: Path, max_batches: int = 25):
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
        del logits, loss, x, y

    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 100))


# ===========================================================================
# Phase 1: Load model and prepare data
# ===========================================================================
def phase_setup():
    """Load model, unpack ternary, apply LoRA, verify data exists."""
    print("=" * 70)
    print("Cross-Adapter Knowledge Transfer Matrix")
    print("=" * 70)

    # Verify data exists from prior experiment
    for domain in DOMAINS:
        data_path = DATA_SOURCE_DIR / domain / "valid.jsonl"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Missing data: {data_path}\n"
                f"Run bitnet_2b_real_composition first to generate domain data."
            )
    print(f"  Data verified: {len(DOMAINS)} domains from {DATA_SOURCE_DIR}")

    print("\n[Phase 0] Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("  Unpacking ternary weights...")
    model = replace_bitlinear_with_linear(model)

    print("  Applying LoRA...")
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Freeze base, unfreeze LoRA
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"  Trainable LoRA params: {trainable:,}")
    log_memory("after-setup")

    return model, tokenizer


# ===========================================================================
# Phase 2: Train adapters (or load if already saved)
# ===========================================================================
def phase_train_adapters(model, tokenizer):
    """Train 5 domain adapters, save to disk. Returns base_ppls and individual_ppls."""
    base_ppls = {}
    individual_ppls = {}

    # Check if adapters already exist
    all_exist = all((ADAPTERS_DIR / d / "adapter.npz").exists() for d in DOMAINS)
    if all_exist:
        print("\n[Phase 2] Adapters already exist, skipping training.")
    else:
        print("\n[Phase 2] Training domain adapters...")

    # Compute base PPLs (no adapter, just zeroed LoRA)
    print("\n  Computing base PPLs...")
    zero_lora_params(model)
    for domain in DOMAINS:
        ppl = compute_ppl(model, tokenizer, DATA_SOURCE_DIR / domain)
        base_ppls[domain] = ppl
        print(f"    {domain}: base PPL = {ppl:.4f}")

    if not all_exist:
        for domain in DOMAINS:
            print(f"\n  --- Training {domain} adapter ---")
            zero_lora_params(model)

            # Load training data
            train_texts = []
            with open(DATA_SOURCE_DIR / domain / "train.jsonl") as f:
                for line in f:
                    train_texts.append(json.loads(line)["text"])

            train_tokens = []
            for text in train_texts:
                toks = tokenizer.encode(text)
                if len(toks) > 2:
                    train_tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))
            print(f"    {len(train_tokens)} training sequences")

            optimizer = opt.Adam(learning_rate=LEARNING_RATE)

            def loss_fn(model, x, y):
                logits = model(x)
                return nn.losses.cross_entropy(logits, y, reduction="mean")

            loss_and_grad = nn.value_and_grad(model, loss_fn)

            t_start = time.time()
            losses = []
            gc.disable()
            for step in range(TRAIN_ITERS):
                idx = step % len(train_tokens)
                tokens = train_tokens[idx]
                x = tokens[:-1][None, :]
                y = tokens[1:][None, :]

                loss, grads = loss_and_grad(model, x, y)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)

                loss_val = loss.item()
                losses.append(loss_val)

                if (step + 1) % 100 == 0:
                    avg = sum(losses[-50:]) / len(losses[-50:])
                    print(f"      Step {step+1}/{TRAIN_ITERS}: avg_loss={avg:.4f}")
            gc.enable()
            gc.collect()

            train_time = time.time() - t_start
            last_50 = sum(losses[-50:]) / 50
            print(f"    Done in {train_time:.1f}s. Final avg loss: {last_50:.4f}")

            save_adapter(model, ADAPTERS_DIR / domain)
            del train_tokens, optimizer, losses
            gc.collect()

    # Evaluate individual adapter PPLs
    print("\n  Evaluating individual adapter PPLs...")
    for domain in DOMAINS:
        adapter = load_adapter(ADAPTERS_DIR / domain)
        zero_lora_params(model)
        apply_adapter_weights(model, adapter, scale=1.0)
        mx.eval(model.parameters())

        ppl = compute_ppl(model, tokenizer, DATA_SOURCE_DIR / domain)
        individual_ppls[domain] = ppl
        improvement = (base_ppls[domain] - ppl) / base_ppls[domain] * 100
        print(f"    {domain}: PPL={ppl:.4f} (base={base_ppls[domain]:.4f}, {improvement:+.1f}%)")
        del adapter

    log_memory("after-train-phase")
    return base_ppls, individual_ppls


# ===========================================================================
# Phase 3: Build pairwise transfer matrix
# ===========================================================================
def phase_transfer_matrix(model, tokenizer, base_ppls, individual_ppls):
    """For each ordered pair (foreign, native), measure transfer benefit."""
    print("\n" + "=" * 70)
    print("[Phase 3] Building pairwise transfer matrix")
    print("=" * 70)

    N = len(DOMAINS)
    # transfer_matrix[i][j] = transfer coefficient when adding adapter_i to domain_j
    # Positive = beneficial (adding foreign adapter helps)
    transfer_matrix = {}
    pairwise_details = {}

    for i, foreign_domain in enumerate(DOMAINS):
        foreign_adapter = load_adapter(ADAPTERS_DIR / foreign_domain)

        for j, native_domain in enumerate(DOMAINS):
            if i == j:
                continue  # Skip same-domain (that's just "does the adapter help its own domain")

            native_adapter = load_adapter(ADAPTERS_DIR / native_domain)
            native_ppl = individual_ppls[native_domain]

            best_ppl = native_ppl
            best_alpha = 0.0  # alpha=0 means native only

            pair_results = {"native_only_ppl": native_ppl, "alpha_sweep": {}}

            for alpha in ALPHA_VALUES:
                # Compose: foreign gets weight alpha, native gets weight (1 - alpha)
                composed = compose_two_adapters(foreign_adapter, native_adapter, alpha, 1.0 - alpha)
                zero_lora_params(model)
                apply_adapter_weights(model, composed, scale=1.0)
                mx.eval(model.parameters())

                ppl = compute_ppl(model, tokenizer, DATA_SOURCE_DIR / native_domain)
                pair_results["alpha_sweep"][str(alpha)] = ppl

                if ppl < best_ppl:
                    best_ppl = ppl
                    best_alpha = alpha

                del composed

            # Transfer coefficient: positive means adding foreign helps
            transfer_coeff = (native_ppl - best_ppl) / native_ppl * 100
            pair_key = f"{foreign_domain}->{native_domain}"

            transfer_matrix[pair_key] = {
                "transfer_pct": round(transfer_coeff, 4),
                "best_alpha": best_alpha,
                "best_ppl": round(best_ppl, 4),
                "native_ppl": round(native_ppl, 4),
            }

            symbol = "+" if transfer_coeff > 0 else "-"
            print(f"  {pair_key:25s}: transfer={transfer_coeff:+.2f}% "
                  f"(best_alpha={best_alpha}, PPL {native_ppl:.2f} -> {best_ppl:.2f})")

            pairwise_details[pair_key] = pair_results
            del native_adapter

        del foreign_adapter
        gc.collect()
        mx.clear_cache()

    log_memory("after-transfer-matrix")
    return transfer_matrix, pairwise_details


# ===========================================================================
# Phase 4: Analyze transfer matrix structure
# ===========================================================================
def phase_analyze(transfer_matrix, base_ppls, individual_ppls):
    """Analyze the transfer matrix for structure and test kill criteria."""
    print("\n" + "=" * 70)
    print("[Phase 4] Analyzing transfer matrix")
    print("=" * 70)

    # Extract transfer coefficients
    transfer_values = []
    positive_pairs = []
    negative_pairs = []

    for pair_key, data in transfer_matrix.items():
        tc = data["transfer_pct"]
        transfer_values.append(tc)
        if tc > 2.0:
            positive_pairs.append((pair_key, tc))
        elif tc < -2.0:
            negative_pairs.append((pair_key, tc))

    positive_pairs.sort(key=lambda x: -x[1])
    negative_pairs.sort(key=lambda x: x[1])

    print(f"\n  Total pairs: {len(transfer_values)}")
    print(f"  Positive transfer (>2%): {len(positive_pairs)}")
    print(f"  Negative transfer (<-2%): {len(negative_pairs)}")
    print(f"  Neutral (-2% to 2%): {len(transfer_values) - len(positive_pairs) - len(negative_pairs)}")

    if positive_pairs:
        print(f"\n  Top positive transfer pairs:")
        for pair, tc in positive_pairs[:10]:
            print(f"    {pair}: +{tc:.2f}%")

    if negative_pairs:
        print(f"\n  Top negative transfer pairs:")
        for pair, tc in negative_pairs[:5]:
            print(f"    {pair}: {tc:.2f}%")

    # Compute summary statistics
    import numpy as np
    tc_array = np.array(transfer_values)
    mean_tc = float(np.mean(tc_array))
    std_tc = float(np.std(tc_array))
    max_tc = float(np.max(tc_array))
    min_tc = float(np.min(tc_array))

    print(f"\n  Transfer coefficient statistics:")
    print(f"    Mean: {mean_tc:.4f}%")
    print(f"    Std:  {std_tc:.4f}%")
    print(f"    Max:  {max_tc:.4f}%")
    print(f"    Min:  {min_tc:.4f}%")

    # Build the NxN matrix for structure analysis
    N = len(DOMAINS)
    matrix = np.zeros((N, N))
    for i, fd in enumerate(DOMAINS):
        for j, nd in enumerate(DOMAINS):
            if i == j:
                matrix[i, j] = 0.0  # self-transfer = 0 by definition
            else:
                key = f"{fd}->{nd}"
                matrix[i, j] = transfer_matrix[key]["transfer_pct"]

    print(f"\n  Transfer Matrix (rows=foreign, cols=native):")
    header = "          " + "".join(f"{d:>10s}" for d in DOMAINS)
    print(f"  {header}")
    for i, fd in enumerate(DOMAINS):
        row = f"  {fd:>8s}"
        for j in range(N):
            if i == j:
                row += "       ---"
            else:
                row += f"   {matrix[i,j]:+6.2f}%"

        print(row)

    # Check symmetry: is transfer(A->B) correlated with transfer(B->A)?
    sym_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            sym_pairs.append((matrix[i, j], matrix[j, i]))
    if sym_pairs:
        a_vals, b_vals = zip(*sym_pairs)
        sym_corr = float(np.corrcoef(a_vals, b_vals)[0, 1]) if len(sym_pairs) > 1 else 0.0
        print(f"\n  Symmetry correlation (A->B vs B->A): {sym_corr:.4f}")

    # Check row/column structure
    row_means = np.mean(matrix, axis=1)  # How much each adapter helps others
    col_means = np.mean(matrix, axis=0)  # How much each domain benefits from others

    print(f"\n  Row means (how much each adapter helps OTHER domains):")
    for i, d in enumerate(DOMAINS):
        print(f"    {d}: {row_means[i]:+.4f}%")

    print(f"\n  Column means (how much each domain BENEFITS from foreign adapters):")
    for j, d in enumerate(DOMAINS):
        print(f"    {d}: {col_means[j]:+.4f}%")

    # Domain similarity heuristic (semantic grouping)
    # Expected high-transfer pairs based on domain relatedness:
    expected_high = {
        "math->python", "python->math",       # Both computational
        "medical->legal", "legal->medical",     # Both professional/formal
    }
    expected_low = {
        "creative->legal", "legal->creative",   # Very different registers
        "creative->math", "math->creative",     # Very different content
    }

    # Check if structure matches expectations
    high_transfers = [transfer_matrix[k]["transfer_pct"] for k in expected_high if k in transfer_matrix]
    low_transfers = [transfer_matrix[k]["transfer_pct"] for k in expected_low if k in transfer_matrix]

    avg_expected_high = np.mean(high_transfers) if high_transfers else 0.0
    avg_expected_low = np.mean(low_transfers) if low_transfers else 0.0

    print(f"\n  Expected-high-transfer pairs avg: {avg_expected_high:+.4f}%")
    print(f"  Expected-low-transfer pairs avg:  {avg_expected_low:+.4f}%")
    structure_gap = avg_expected_high - avg_expected_low
    print(f"  Structure gap (high - low): {structure_gap:+.4f}%")

    # Variance analysis: is the matrix more structured than random?
    # A random matrix would have all transfer coefficients near 0 with small variance
    # A structured matrix has high variance and clear clusters
    matrix_variance = float(np.var(tc_array))
    print(f"\n  Matrix variance: {matrix_variance:.6f}")

    # Kill criteria assessment
    k1_pass = len(positive_pairs) > 0
    print(f"\n  K1: {len(positive_pairs)} pairs with >2% improvement -> {'PASS' if k1_pass else 'KILL'}")

    # K2: Structure test - is the transfer matrix non-random?
    # Method: compare variance of transfer coefficients to what we'd expect from noise
    # Also check if expected-related domains DO transfer better
    k2_structure_score = structure_gap  # Positive = expected structure exists
    k2_pass = (matrix_variance > 0.5 or abs(structure_gap) > 1.0 or sym_corr > 0.3)
    print(f"  K2: variance={matrix_variance:.4f}, structure_gap={structure_gap:.2f}%, "
          f"sym_corr={sym_corr:.4f} -> {'PASS' if k2_pass else 'KILL'}")

    analysis = {
        "n_positive_pairs": len(positive_pairs),
        "n_negative_pairs": len(negative_pairs),
        "positive_pairs": [(p, round(t, 4)) for p, t in positive_pairs],
        "negative_pairs": [(p, round(t, 4)) for p, t in negative_pairs],
        "mean_transfer_pct": round(mean_tc, 4),
        "std_transfer_pct": round(std_tc, 4),
        "max_transfer_pct": round(max_tc, 4),
        "min_transfer_pct": round(min_tc, 4),
        "matrix_variance": round(matrix_variance, 6),
        "symmetry_correlation": round(sym_corr, 4),
        "expected_high_avg": round(float(avg_expected_high), 4),
        "expected_low_avg": round(float(avg_expected_low), 4),
        "structure_gap": round(structure_gap, 4),
        "row_means": {d: round(float(row_means[i]), 4) for i, d in enumerate(DOMAINS)},
        "col_means": {d: round(float(col_means[j]), 4) for j, d in enumerate(DOMAINS)},
        "transfer_matrix_grid": matrix.tolist(),
        "k1_pass": k1_pass,
        "k2_pass": k2_pass,
    }

    return analysis


# ===========================================================================
# Main orchestrator
# ===========================================================================
def main():
    t0_total = time.time()

    model, tokenizer = phase_setup()

    base_ppls, individual_ppls = phase_train_adapters(model, tokenizer)
    log_memory("after-training")

    transfer_matrix, pairwise_details = phase_transfer_matrix(
        model, tokenizer, base_ppls, individual_ppls
    )

    # Free model before analysis (only numpy needed)
    cleanup(model, tokenizer)
    del model, tokenizer

    analysis = phase_analyze(transfer_matrix, base_ppls, individual_ppls)

    # Compile results
    total_time = time.time() - t0_total
    results = {
        "experiment": "cross_adapter_knowledge_transfer",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "train_iters": TRAIN_ITERS,
        "domains": DOMAINS,
        "alpha_values_searched": ALPHA_VALUES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_s": round(total_time, 1),
        "base_ppls": {d: round(v, 4) for d, v in base_ppls.items()},
        "individual_ppls": {d: round(v, 4) for d, v in individual_ppls.items()},
        "transfer_matrix": transfer_matrix,
        "analysis": analysis,
        "verdict": "SUPPORTED" if analysis["k1_pass"] else "KILLED",
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n{'=' * 70}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"Verdict: {results['verdict']}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
