"""Homomorphic Expert Composition: Privacy-Preserving LoRA Averaging.

Tests whether LoRA deltas can be composed (averaged) in encrypted space
using the Paillier partially-homomorphic encryption scheme, preserving
composition quality while enabling privacy.

The key insight: weight averaging IS additive composition, and Paillier
supports exact addition in ciphertext space. The only quality risk comes
from the float->int quantization needed for Paillier (which operates on
integers). We test multiple quantization precisions.

Kill criteria:
- Encryption noise degrades composition >5% vs plaintext
- Encrypted composition >100x slower than plaintext

Protocol:
1. Pretrain base GPT on joint data
2. Fine-tune N LoRA adapters on domain-specific data
3. Extract LoRA deltas as float matrices
4. Compose via 4 methods:
   a. Plaintext averaging (baseline)
   b. Paillier-encrypted averaging (batch-packed, 16-bit quantization)
   c. Paillier-encrypted averaging (batch-packed, 24-bit quantization)
   d. Paillier-encrypted averaging (batch-packed, 32-bit quantization)
5. Compare quality (val loss) and wall-clock time
"""

import copy
import json
import math
import random
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from phe import paillier

from experiments.models import get_model
from experiments.data import (
    load_names, CharTokenizer, CharDataset,
    domain_split, train_val_split,
)
from experiments.train import train, evaluate, ntp_loss


# ── Config ──────────────────────────────────────────────────────────────────

BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
LORA_RANK = 8
LORA_ALPHA = 1.0
PRETRAIN_STEPS = 300
FINETUNE_STEPS = 300
BATCH_SIZE = 32
LR = 3e-3

# Paillier config
PAILLIER_KEY_BITS = 2048  # Standard security level


# ── Utilities ───────────────────────────────────────────────────────────────

def copy_weights(src, dst):
    """Copy all weights from src to dst model."""
    pairs = list(zip(
        [k for k, _ in nn.utils.tree_flatten(src.parameters())],
        [v for _, v in nn.utils.tree_flatten(src.parameters())]
    ))
    dst.load_weights(pairs)
    mx.eval(dst.parameters())


def freeze_except_lora(model):
    """Freeze all parameters except LoRA A and B."""
    model.freeze()
    for layer in model.layers:
        layer.mlp.fc1.unfreeze()
        layer.mlp.fc2.unfreeze()
        layer.mlp.fc1.linear.freeze()
        layer.mlp.fc2.linear.freeze()


def reset_lora(model):
    """Reset all LoRA A/B to initialization."""
    for layer in model.layers:
        for fc in [layer.mlp.fc1, layer.mlp.fc2]:
            in_dim = fc.A.shape[0]
            scale = (2.0 / in_dim) ** 0.5
            fc.A = mx.random.normal(fc.A.shape) * scale
            fc.B = mx.zeros(fc.B.shape)
    mx.eval(model.parameters())


def extract_deltas(model) -> dict:
    """Extract LoRA deltas as {(layer, sublayer): numpy_array}."""
    deltas = {}
    for l_idx, layer in enumerate(model.layers):
        for name, fc in [('fc1', layer.mlp.fc1), ('fc2', layer.mlp.fc2)]:
            delta = (fc.alpha / fc.rank) * (fc.A @ fc.B)
            mx.eval(delta)
            deltas[(l_idx, name)] = np.array(delta)
    return deltas


def apply_merged_deltas(base_model, merged_deltas, vocab_size):
    """Create a GPT model with base weights + merged LoRA deltas baked in."""
    from experiments.models.gpt.gpt import GPT
    model = GPT(vocab_size=vocab_size, **BASE)
    mx.eval(model.parameters())
    copy_weights(base_model, model)

    for (l_idx, name), delta_np in merged_deltas.items():
        delta = mx.array(delta_np)
        layer = model.layers[l_idx]
        if name == 'fc1':
            layer.mlp.fc1.weight = layer.mlp.fc1.weight + delta.T
        elif name == 'fc2':
            layer.mlp.fc2.weight = layer.mlp.fc2.weight + delta.T

    mx.eval(model.parameters())
    return model


# ── Plaintext Composition ───────────────────────────────────────────────────

def compose_plaintext(delta_list: list[dict]) -> dict:
    """Simple averaging: merged[k] = (1/N) * sum(delta_i[k])."""
    N = len(delta_list)
    keys = list(delta_list[0].keys())
    merged = {}
    for k in keys:
        merged[k] = sum(d[k] for d in delta_list) / N
    return merged


# ── Fixed-Point Quantization ────────────────────────────────────────────────

def quantize_to_int(arr: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    """Quantize float array to signed fixed-point integers.

    Args:
        arr: float array
        bits: precision bits (e.g. 16, 24, 32)

    Returns:
        (int_arr, scale) where arr ~ int_arr / scale
    """
    max_val = np.max(np.abs(arr))
    if max_val == 0:
        return np.zeros_like(arr, dtype=np.int64), 1.0

    # Scale to fit in [-2^(bits-1), 2^(bits-1) - 1]
    scale = (2 ** (bits - 1) - 1) / max_val
    int_arr = np.round(arr * scale).astype(np.int64)
    return int_arr, float(scale)


def dequantize_from_int(int_arr: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize integer array back to float."""
    return int_arr.astype(np.float32) / scale


# ── Batch Paillier Encryption ───────────────────────────────────────────────

def batch_encrypt_array(pk, arr_int: np.ndarray, val_bits: int,
                        key_bits: int = None) -> list:
    """Encrypt integer array using batch packing.

    Packs multiple val_bits-width integers into each Paillier plaintext
    to amortize encryption cost.

    Args:
        pk: Paillier public key
        arr_int: 1D array of integers
        val_bits: bits per value (determines packing density)
        key_bits: key size in bits (auto-detected from pk if None)

    Returns:
        list of (encrypted_packed_value, n_values_in_this_pack)
    """
    flat = arr_int.flatten()
    n = len(flat)

    # Detect key size from public key's n value
    if key_bits is None:
        key_bits = pk.n.bit_length()

    # For safety with additions of up to 256 experts, add 8 bits headroom per value
    slot_bits = val_bits + 8  # headroom for accumulation
    # Available plaintext bits (leave 64 bits of margin for Paillier)
    available_bits = key_bits - 64
    vals_per_pack = max(1, available_bits // slot_bits)

    encrypted_packs = []
    for start in range(0, n, vals_per_pack):
        end = min(start + vals_per_pack, n)
        chunk = flat[start:end]
        # Pack values into a single large integer
        # Each value occupies slot_bits, offset to be non-negative
        offset = 2 ** (val_bits - 1)  # shift to make all values non-negative
        packed = 0
        for i, v in enumerate(chunk):
            shifted = int(v) + offset  # now in [0, 2^val_bits - 1]
            packed += shifted << (i * slot_bits)

        encrypted_packs.append((pk.encrypt(packed), end - start))

    return encrypted_packs


def batch_decrypt_array(sk, encrypted_packs: list, val_bits: int,
                        total_elements: int, n_addends: int = 1) -> np.ndarray:
    """Decrypt batch-packed Paillier ciphertexts back to integer array.

    Args:
        sk: Paillier secret key
        encrypted_packs: list of (encrypted_value, n_values_in_pack)
        val_bits: bits per value
        total_elements: total number of original values
        n_addends: how many encrypted arrays were summed (to correct offset)
    """
    slot_bits = val_bits + 8
    offset = 2 ** (val_bits - 1)
    mask = (1 << slot_bits) - 1

    result = []
    for enc_val, n_vals in encrypted_packs:
        packed = sk.decrypt(enc_val)
        for i in range(n_vals):
            shifted = (packed >> (i * slot_bits)) & mask
            # After summing N addends, offset is N * offset per value
            original = shifted - offset * n_addends
            result.append(original)

    return np.array(result[:total_elements], dtype=np.int64)


def homomorphic_add_packs(packs_list: list[list]) -> list:
    """Add multiple sets of encrypted packs element-wise.

    Args:
        packs_list: list of N pack lists (one per expert)

    Returns:
        summed packs (same structure)
    """
    result = []
    for pack_idx in range(len(packs_list[0])):
        enc_sum = packs_list[0][pack_idx][0]
        n_vals = packs_list[0][pack_idx][1]
        for expert_idx in range(1, len(packs_list)):
            enc_sum = enc_sum + packs_list[expert_idx][pack_idx][0]
        result.append((enc_sum, n_vals))
    return result


# ── Encrypted Composition ───────────────────────────────────────────────────

def compose_encrypted(delta_list: list[dict], bits: int = 24,
                      pk=None, sk=None) -> tuple[dict, dict]:
    """Compose LoRA deltas via Paillier-encrypted averaging.

    Each contributor:
    1. Quantizes their float delta to fixed-point integers
    2. Batch-packs and encrypts with the shared public key
    3. Sends encrypted packs to aggregator

    Aggregator:
    1. Adds all encrypted packs (homomorphic addition)
    2. Decrypts the sum
    3. Divides by N and dequantizes

    Args:
        delta_list: list of N delta dicts {(layer,sublayer): np.array}
        bits: quantization precision
        pk, sk: Paillier keypair (generated if None)

    Returns:
        (merged_deltas, timing_info)
    """
    N = len(delta_list)
    keys = list(delta_list[0].keys())

    if pk is None or sk is None:
        t0 = time.time()
        pk, sk = paillier.generate_paillier_keypair(n_length=PAILLIER_KEY_BITS)
        keygen_time = time.time() - t0
    else:
        keygen_time = 0.0

    merged = {}
    total_encrypt_time = 0.0
    total_add_time = 0.0
    total_decrypt_time = 0.0
    total_quant_error = 0.0
    n_values = 0

    for k in keys:
        shape = delta_list[0][k].shape
        flat_len = int(np.prod(shape))

        # --- Contributor side: quantize + encrypt ---
        # Each contributor uses the same scale (in practice, they'd agree on
        # a range or use per-contributor scales with a protocol).
        # Here we use a global scale from the max across all contributors
        # for fairness.
        all_vals = np.stack([d[k] for d in delta_list])
        global_max = np.max(np.abs(all_vals))
        if global_max == 0:
            merged[k] = np.zeros(shape, dtype=np.float32)
            continue

        scale = (2 ** (bits - 1) - 1) / global_max

        encrypted_per_expert = []
        for i in range(N):
            arr = delta_list[i][k]
            int_arr = np.round(arr * scale).astype(np.int64)

            # Measure quantization error
            recon = int_arr.astype(np.float32) / scale
            quant_err = np.max(np.abs(arr - recon))
            total_quant_error = max(total_quant_error, quant_err)

            t0 = time.time()
            packs = batch_encrypt_array(pk, int_arr, bits)
            total_encrypt_time += time.time() - t0
            encrypted_per_expert.append(packs)

        # --- Aggregator side: homomorphic addition ---
        t0 = time.time()
        summed_packs = homomorphic_add_packs(encrypted_per_expert)
        total_add_time += time.time() - t0

        # --- Aggregator side: decrypt + dequantize ---
        t0 = time.time()
        summed_int = batch_decrypt_array(sk, summed_packs, bits, flat_len, n_addends=N)
        total_decrypt_time += time.time() - t0

        # Divide by N (in integer domain) then dequantize
        # Note: integer division introduces rounding, so we use float division
        avg_float = summed_int.astype(np.float64) / N / scale
        merged[k] = avg_float.reshape(shape).astype(np.float32)
        n_values += flat_len

    timing = {
        'keygen_s': keygen_time,
        'encrypt_s': total_encrypt_time,
        'add_s': total_add_time,
        'decrypt_s': total_decrypt_time,
        'total_s': keygen_time + total_encrypt_time + total_add_time + total_decrypt_time,
        'n_values': n_values,
        'max_quant_error': float(total_quant_error),
        'bits': bits,
    }

    return merged, timing


# ── Training Helpers ────────────────────────────────────────────────────────

def pretrain_base(joint_train, vocab_size, seed):
    """Pretrain base GPT model on joint data."""
    base = get_model("gpt", vocab_size=vocab_size, **BASE)
    mx.eval(base.parameters())
    train(base, joint_train, steps=PRETRAIN_STEPS, batch_size=BATCH_SIZE,
          lr=LR, seed=seed, log_every=150)
    return base


def finetune_lora(base_model, train_ds, val_ds, vocab_size, seed):
    """Fine-tune a LoRA model on a single domain."""
    lora = get_model("lora_gpt", vocab_size=vocab_size, **BASE,
                     lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA)
    mx.eval(lora.parameters())

    # Copy base weights
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


# ── Main Experiment ─────────────────────────────────────────────────────────

def run_experiment(seed=42, n_domains=2):
    """Run the full homomorphic composition experiment."""
    print(f"\n{'='*70}")
    print(f"HOMOMORPHIC EXPERT COMPOSITION (seed={seed}, N={n_domains})")
    print(f"{'='*70}")

    mx.random.seed(seed)
    t_total = time.time()

    # Data prep
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    if n_domains == 2:
        splits = domain_split(docs, method="binary")
    elif n_domains == 5:
        splits = domain_split(docs, method="quintary")
    else:
        raise ValueError(f"Unsupported n_domains={n_domains}")

    all_train, all_val = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_train, tokenizer, BASE["block_size"])

    train_datasets = {}
    val_datasets = {}
    for d_name, d_docs in splits.items():
        d_train, d_val = train_val_split(d_docs, seed=seed)
        train_datasets[d_name] = CharDataset(d_train, tokenizer, BASE["block_size"])
        val_datasets[d_name] = CharDataset(d_val, tokenizer, BASE["block_size"])

    # === 1. Pretrain base ===
    print("\n--- 1. Pretraining base model ---")
    base_model = pretrain_base(joint_train, V, seed)

    # === 2. Fine-tune LoRA per domain ===
    lora_models = {}
    delta_dicts = {}  # {domain: {(layer, sublayer): np.array}}

    domain_names = list(splits.keys())
    for i, d_name in enumerate(domain_names):
        print(f"\n--- 2{chr(97+i)}. Fine-tuning LoRA for {d_name} ---")
        lora = finetune_lora(
            base_model, train_datasets[d_name], val_datasets[d_name],
            V, seed + i)
        lora_models[d_name] = lora
        delta_dicts[d_name] = extract_deltas(lora)

    all_deltas = [delta_dicts[d] for d in domain_names]

    results = {}

    # === 3. Plaintext averaging (baseline) ===
    print("\n--- 3. Plaintext averaging (baseline) ---")
    t0 = time.time()
    merged_plain = compose_plaintext(all_deltas)
    plaintext_time = time.time() - t0
    model_plain = apply_merged_deltas(base_model, merged_plain, V)

    plain_losses = {}
    for name, val_ds in val_datasets.items():
        plain_losses[name] = evaluate(model_plain, val_ds, BATCH_SIZE)
    plain_losses["avg"] = sum(v for k, v in plain_losses.items() if k != "avg") / n_domains

    results["plaintext"] = {
        "losses": plain_losses,
        "time_s": plaintext_time,
        "method": "simple_average",
    }
    print(f"  Plaintext avg loss: {plain_losses['avg']:.4f} ({plaintext_time*1000:.2f}ms)")

    # === 4. Generate Paillier keypair (shared cost) ===
    print("\n--- 4. Generating Paillier keypair ---")
    t0 = time.time()
    pk, sk = paillier.generate_paillier_keypair(n_length=PAILLIER_KEY_BITS)
    keygen_time = time.time() - t0
    print(f"  Keygen ({PAILLIER_KEY_BITS}-bit): {keygen_time:.2f}s")

    # === 5. Encrypted averaging at different quantization levels ===
    for bits in [16, 24, 32]:
        print(f"\n--- 5. Encrypted averaging ({bits}-bit quantization) ---")
        merged_enc, timing = compose_encrypted(
            all_deltas, bits=bits, pk=pk, sk=sk)
        model_enc = apply_merged_deltas(base_model, merged_enc, V)

        enc_losses = {}
        for name, val_ds in val_datasets.items():
            enc_losses[name] = evaluate(model_enc, val_ds, BATCH_SIZE)
        enc_losses["avg"] = sum(v for k, v in enc_losses.items() if k != "avg") / n_domains

        # Compute delta error (encrypted vs plaintext)
        max_delta_err = 0.0
        mean_delta_err = 0.0
        n_vals = 0
        for k in merged_plain:
            diff = np.abs(merged_enc[k] - merged_plain[k])
            max_delta_err = max(max_delta_err, np.max(diff))
            mean_delta_err += np.sum(diff)
            n_vals += diff.size
        mean_delta_err /= n_vals

        results[f"encrypted_{bits}bit"] = {
            "losses": enc_losses,
            "timing": timing,
            "max_delta_error": float(max_delta_err),
            "mean_delta_error": float(mean_delta_err),
            "quality_gap_pct": (enc_losses["avg"] - plain_losses["avg"]) / plain_losses["avg"] * 100,
            "slowdown_vs_plaintext": timing["total_s"] / max(plaintext_time, 1e-9),
        }

        print(f"  Encrypted avg loss: {enc_losses['avg']:.4f}")
        print(f"  vs plaintext: {results[f'encrypted_{bits}bit']['quality_gap_pct']:+.4f}%")
        print(f"  Timing: encrypt={timing['encrypt_s']:.2f}s, "
              f"add={timing['add_s']:.4f}s, decrypt={timing['decrypt_s']:.3f}s, "
              f"total={timing['total_s']:.2f}s")
        print(f"  Slowdown: {results[f'encrypted_{bits}bit']['slowdown_vs_plaintext']:.1f}x")
        print(f"  Max weight error: {max_delta_err:.2e}, Mean: {mean_delta_err:.2e}")

    # === Summary ===
    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"SUMMARY (N={n_domains}, seed={seed}, elapsed={elapsed:.1f}s)")
    print(f"{'='*70}")

    print(f"\n{'Method':<25} {'avg loss':>10} {'vs plain':>10} {'time':>10} {'slowdown':>10}")
    print("-" * 68)

    plain_avg = results["plaintext"]["losses"]["avg"]
    print(f"{'plaintext':<25} {plain_avg:>10.4f} {'baseline':>10} "
          f"{results['plaintext']['time_s']*1000:>9.2f}ms {'1.0x':>10}")

    for bits in [16, 24, 32]:
        key = f"encrypted_{bits}bit"
        r = results[key]
        enc_avg = r["losses"]["avg"]
        gap = r["quality_gap_pct"]
        total = r["timing"]["total_s"]
        slowdown = r["slowdown_vs_plaintext"]
        print(f"{'encrypted_' + str(bits) + 'bit':<25} {enc_avg:>10.4f} "
              f"{gap:>+9.4f}% {total:>9.2f}s {slowdown:>9.1f}x")

    # Per-domain detail
    print(f"\n  Per-domain losses:")
    for method_key in ["plaintext"] + [f"encrypted_{b}bit" for b in [16, 24, 32]]:
        losses = results[method_key]["losses"]
        domain_strs = [f"{d}={losses[d]:.4f}" for d in domain_names]
        print(f"    {method_key:<23} {' | '.join(domain_strs)}")

    # Kill criteria
    print(f"\n--- Kill Criteria ---")

    # KC1: encryption noise degrades composition >5%
    gaps = [results[f"encrypted_{b}bit"]["quality_gap_pct"] for b in [16, 24, 32]]
    worst_gap = max(abs(g) for g in gaps)
    kc1_pass = worst_gap < 5.0
    print(f"\n  KC1: encryption noise degrades >5%?")
    for bits in [16, 24, 32]:
        g = results[f"encrypted_{bits}bit"]["quality_gap_pct"]
        print(f"    {bits}-bit: {g:+.4f}% {'PASS' if abs(g) < 5.0 else 'KILL'}")
    print(f"  --> {'PASS' if kc1_pass else 'KILL'}: worst gap = {worst_gap:.4f}%")

    # KC2: encrypted >100x slower than plaintext
    slowdowns = [results[f"encrypted_{b}bit"]["slowdown_vs_plaintext"] for b in [16, 24, 32]]
    best_slowdown = min(slowdowns)
    kc2_pass = best_slowdown < 100.0
    print(f"\n  KC2: encrypted >100x slower?")
    for bits in [16, 24, 32]:
        s = results[f"encrypted_{bits}bit"]["slowdown_vs_plaintext"]
        print(f"    {bits}-bit: {s:.1f}x {'PASS' if s < 100.0 else 'KILL'}")
    print(f"  --> {'PASS' if kc2_pass else 'KILL'}: best slowdown = {best_slowdown:.1f}x")

    overall = "PROVEN" if (kc1_pass and kc2_pass) else "KILLED"
    print(f"\n  OVERALL: {overall}")

    return results


def run_multiseed(seeds=(42, 123, 7), n_domains=2):
    """Run experiment across multiple seeds."""
    all_results = {}
    for seed in seeds:
        all_results[seed] = run_experiment(seed, n_domains)

    # Aggregate
    print(f"\n\n{'='*70}")
    print(f"MULTI-SEED AGGREGATE (N={n_domains}, seeds={seeds})")
    print(f"{'='*70}")

    methods = ["plaintext"] + [f"encrypted_{b}bit" for b in [16, 24, 32]]

    print(f"\n{'Method':<25} {'mean loss':>10} {'std':>8} {'vs plain':>10} {'slowdown':>10}")
    print("-" * 68)

    plain_means = [all_results[s]["plaintext"]["losses"]["avg"] for s in seeds]
    plain_mean = statistics.mean(plain_means)
    plain_std = statistics.stdev(plain_means) if len(seeds) > 1 else 0
    print(f"{'plaintext':<25} {plain_mean:>10.4f} {plain_std:>8.4f} {'baseline':>10} {'1.0x':>10}")

    aggregate = {}
    for method in methods[1:]:
        avgs = [all_results[s][method]["losses"]["avg"] for s in seeds]
        mean_avg = statistics.mean(avgs)
        std_avg = statistics.stdev(avgs) if len(seeds) > 1 else 0
        gap = (mean_avg - plain_mean) / plain_mean * 100
        slowdowns = [all_results[s][method]["slowdown_vs_plaintext"] for s in seeds]
        mean_slow = statistics.mean(slowdowns)
        aggregate[method] = {"mean": mean_avg, "std": std_avg, "gap": gap, "slowdown": mean_slow}
        print(f"{method:<25} {mean_avg:>10.4f} {std_avg:>8.4f} {gap:>+9.4f}% {mean_slow:>9.1f}x")

    # Aggregate kill criteria
    print(f"\n--- Aggregate Kill Criteria ---")
    worst_gap = max(abs(aggregate[m]["gap"]) for m in aggregate)
    best_slow = min(aggregate[m]["slowdown"] for m in aggregate)
    print(f"  KC1: worst quality gap = {worst_gap:.4f}% ({'PASS' if worst_gap < 5 else 'KILL'})")
    print(f"  KC2: best slowdown = {best_slow:.1f}x ({'PASS' if best_slow < 100 else 'KILL'})")

    overall = "PROVEN" if (worst_gap < 5 and best_slow < 100) else "KILLED"
    print(f"  OVERALL: {overall}")

    return all_results, aggregate


# ── Unit Tests ──────────────────────────────────────────────────────────────

def test_quantize_roundtrip():
    """Test that quantize -> dequantize preserves values within precision."""
    np.random.seed(42)
    arr = np.random.randn(64, 64).astype(np.float32) * 0.01

    for bits in [16, 24, 32]:
        int_arr, scale = quantize_to_int(arr, bits)
        recon = dequantize_from_int(int_arr, scale)
        max_err = np.max(np.abs(arr - recon))
        # Error should be < 1 LSB = max_val / 2^(bits-1)
        # For 32-bit, float32 precision (~7 decimal digits) dominates
        max_val = np.max(np.abs(arr))
        expected_err = max(max_val / (2 ** (bits - 1)), max_val * 1e-7)
        assert max_err < 2 * expected_err, (
            f"{bits}-bit: max_err={max_err:.2e} > 2*expected={2*expected_err:.2e}")
        print(f"  {bits}-bit roundtrip: max_err={max_err:.2e}, expected<{expected_err:.2e}")

    print("PASS: quantize roundtrip")


def test_paillier_batch_roundtrip():
    """Test batch encrypt -> decrypt preserves integer values exactly."""
    pk, sk = paillier.generate_paillier_keypair(n_length=1024)

    arr = np.array([100, -50, 32767, -32768, 0, 1, -1], dtype=np.int64)
    bits = 16

    packs = batch_encrypt_array(pk, arr, bits)
    recovered = batch_decrypt_array(sk, packs, bits, len(arr))

    assert np.array_equal(arr, recovered), (
        f"Roundtrip failed: {arr} vs {recovered}")
    print("PASS: Paillier batch roundtrip (exact)")


def test_paillier_homomorphic_add():
    """Test that encrypting separately and adding gives same as adding then encrypting."""
    pk, sk = paillier.generate_paillier_keypair(n_length=1024)

    a = np.array([10, -20, 30], dtype=np.int64)
    b = np.array([5, 15, -10], dtype=np.int64)
    bits = 16

    packs_a = batch_encrypt_array(pk, a, bits)
    packs_b = batch_encrypt_array(pk, b, bits)

    summed = homomorphic_add_packs([packs_a, packs_b])
    result = batch_decrypt_array(sk, summed, bits, len(a), n_addends=2)

    expected = a + b
    assert np.array_equal(result, expected), (
        f"Homomorphic add failed: {result} vs {expected}")
    print("PASS: Paillier homomorphic addition")


def test_full_pipeline_tiny():
    """Test the full encrypt-compose-decrypt pipeline on tiny data."""
    np.random.seed(42)

    # Simulate 2 experts with tiny deltas
    delta1 = {(0, 'fc1'): np.random.randn(4, 4).astype(np.float32) * 0.01}
    delta2 = {(0, 'fc1'): np.random.randn(4, 4).astype(np.float32) * 0.01}

    # Plaintext average
    plain = compose_plaintext([delta1, delta2])

    # Encrypted average
    pk, sk = paillier.generate_paillier_keypair(n_length=1024)
    enc, _ = compose_encrypted([delta1, delta2], bits=32, pk=pk, sk=sk)

    # Compare
    max_err = np.max(np.abs(plain[(0, 'fc1')] - enc[(0, 'fc1')]))
    print(f"  Plain vs encrypted max error: {max_err:.2e}")

    # At 32-bit, error should be negligible
    assert max_err < 1e-5, f"Pipeline error too large: {max_err:.2e}"
    print("PASS: full pipeline tiny")


def run_all_tests():
    """Run all unit tests."""
    print("\n=== Unit Tests ===\n")
    test_quantize_roundtrip()
    print()
    test_paillier_batch_roundtrip()
    print()
    test_paillier_homomorphic_add()
    print()
    test_full_pipeline_tiny()
    print("\n=== All tests passed ===\n")


# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_all_tests()
    else:
        # Run unit tests first
        run_all_tests()

        # Run single-seed first to verify
        results = run_experiment(seed=42, n_domains=2)

        # Save results
        out_dir = Path(__file__).parent
        # Convert numpy types for JSON serialization
        def make_serializable(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            return obj

        with open(out_dir / "results.json", "w") as f:
            json.dump(make_serializable(results), f, indent=2)
        print(f"\nResults saved to {out_dir / 'results.json'}")
