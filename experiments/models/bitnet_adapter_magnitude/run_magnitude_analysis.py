#!/usr/bin/env python3
"""
BitNet Adapter Magnitude Analysis: Does ternary base bound adapter magnitudes?

Tests the hypothesis that ternary {-1,0,1} base weights constrain LoRA adapter
weight magnitudes to a tighter range than FP16, explaining why composition works
on ternary despite WORSE orthogonality (mean |cos| 0.276 vs 0.260).

Measurements:
  1. Per-adapter delta norms (||B_i @ A_i||_F) and their variance across adapters
  2. Per-layer delta norms to identify where magnitude differences concentrate
  3. Activation magnitudes (hidden state norms) when adapters are applied
  4. Logit-scale distributions (output logit std per domain)
  5. Composition delta norms and scaling behavior

Reuses architecture from bitnet_composition_stability: d=64, r=4, L=2, 5 domains, 3 seeds.

Kill criteria (from HYPOTHESES.yml):
  K1: adapter weight norm variance on BitNet >= variance on FP16
  K2: max/min adapter norm ratio on BitNet > 10x
"""

import json
import sys
import time
from pathlib import Path

import autograd.numpy as np
from autograd import grad
import numpy as onp

# Import everything from the composition stability experiment
sys.path.insert(0, str(Path(__file__).parent.parent / "bitnet_composition_stability"))
from bitnet_composition_stability import (
    CharTokenizer, DOMAIN_GENERATORS,
    init_model, quantize_model_to_ternary,
    init_lora, train_lora, lora_to_delta, apply_lora, apply_delta,
    compose_deltas, flatten_delta, cosine_sim,
    train_base, eval_loss, _prepare_batch, forward,
)


def per_layer_delta_norms(lora, base_params, rank=4):
    """Compute delta norm per layer and per weight matrix."""
    norms = {}
    for k in base_params:
        if k == '_config' or base_params[k].ndim < 2:
            continue
        A_key = f'{k}_A'
        B_key = f'{k}_B'
        if A_key in lora and B_key in lora:
            delta = onp.dot(lora[A_key], lora[B_key])
            norms[k] = {
                'delta_norm': float(onp.linalg.norm(delta)),
                'delta_mean_abs': float(onp.mean(onp.abs(delta))),
                'delta_max_abs': float(onp.max(onp.abs(delta))),
                'A_norm': float(onp.linalg.norm(lora[A_key])),
                'B_norm': float(onp.linalg.norm(lora[B_key])),
            }
    return norms


def measure_activation_magnitudes(params, data_encoded, pad_id, n_samples=50):
    """Run forward pass and measure hidden state norms at each layer.

    Returns per-layer activation statistics.
    """
    batch = data_encoded[:n_samples]
    inp, tgt, mask = _prepare_batch(batch, pad_id)

    cfg = params['_config']
    d, H, L = cfg['d'], cfg['H'], cfg['L']
    hd = d // H
    B, T = inp.shape

    # Manual forward pass to capture intermediate activations
    x = params['tok_emb'][inp] + params['pos_emb'][:T]
    causal_mask = onp.triu(onp.ones((T, T)) * (-1e9), k=1).astype(onp.float32)

    layer_stats = {}

    for li in range(L):
        # Pre-attention
        x_np = onp.array(x) if not isinstance(x, onp.ndarray) else x
        pre_norm = float(onp.mean(onp.linalg.norm(x_np.reshape(-1, d), axis=-1)))

        # RMS norm
        ms = onp.mean(x_np ** 2, axis=-1, keepdims=True)
        h = x_np / onp.sqrt(ms + 1e-5) * params[f'ln1_w_{li}']

        # Attention
        qkv = onp.dot(h, params[f'Wqkv_{li}'])
        qkv = qkv.reshape(B, T, 3, H, hd)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        scale = 1.0 / onp.sqrt(hd)
        attn = onp.einsum('bhqd,bhkd->bhqk', q, k) * scale + causal_mask
        attn = attn - onp.max(attn, axis=-1, keepdims=True)
        attn = onp.exp(attn)
        attn = attn / onp.sum(attn, axis=-1, keepdims=True)
        out = onp.einsum('bhqk,bhkd->bhqd', attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, d)
        out = onp.dot(out, params[f'Wo_{li}'])
        x_np = x_np + out

        post_attn_norm = float(onp.mean(onp.linalg.norm(x_np.reshape(-1, d), axis=-1)))

        # FFN
        ms = onp.mean(x_np ** 2, axis=-1, keepdims=True)
        h = x_np / onp.sqrt(ms + 1e-5) * params[f'ln2_w_{li}']
        ffn = onp.maximum(0, onp.dot(h, params[f'W1_{li}']))
        ffn = onp.dot(ffn, params[f'W2_{li}'])
        x_np = x_np + ffn

        post_ffn_norm = float(onp.mean(onp.linalg.norm(x_np.reshape(-1, d), axis=-1)))

        layer_stats[f'layer_{li}'] = {
            'pre_norm': pre_norm,
            'post_attn_norm': post_attn_norm,
            'post_ffn_norm': post_ffn_norm,
            'attn_residual_ratio': post_attn_norm / max(pre_norm, 1e-10),
            'ffn_residual_ratio': post_ffn_norm / max(post_attn_norm, 1e-10),
        }
        x = x_np

    # Final logits
    ms = onp.mean(x ** 2, axis=-1, keepdims=True)
    x_final = x / onp.sqrt(ms + 1e-5) * params['ln_f_w']
    logits = onp.dot(x_final, params['W_head'])

    # Logit statistics (over non-padding positions)
    valid_logits = []
    for b in range(B):
        for t in range(T):
            if mask[b, t] > 0:
                valid_logits.append(logits[b, t])
    valid_logits = onp.array(valid_logits)

    logit_stats = {
        'logit_mean': float(onp.mean(valid_logits)),
        'logit_std': float(onp.std(valid_logits)),
        'logit_max': float(onp.max(valid_logits)),
        'logit_min': float(onp.min(valid_logits)),
        'logit_range': float(onp.max(valid_logits) - onp.min(valid_logits)),
        'per_token_std_mean': float(onp.mean(onp.std(valid_logits, axis=-1))),
        'per_token_std_std': float(onp.std(onp.std(valid_logits, axis=-1))),
    }

    return layer_stats, logit_stats


def run_seed(seed=42, d=64, r=4, L=2, H=2, n_data=300, n_eval=100,
             base_epochs=30, lora_epochs=30, verbose=True):
    """Run one seed of the magnitude analysis experiment."""

    print(f"\n{'='*70}")
    print(f"  BitNet Adapter Magnitude Analysis (seed={seed}, d={d}, r={r})")
    print(f"{'='*70}")

    tok = CharTokenizer()
    V = tok.vocab_size
    rng = onp.random.RandomState(seed)
    domain_names = list(DOMAIN_GENERATORS.keys())

    # Generate data
    print("\n[1/5] Generating domain data...")
    domain_data = {}
    domain_eval = {}
    mixed_train = []
    for name, gen_fn in DOMAIN_GENERATORS.items():
        train = gen_fn(n_data, rng)
        test = gen_fn(n_eval, rng)
        domain_data[name] = [tok.encode(s) for s in train]
        domain_eval[name] = [tok.encode(s) for s in test]
        mixed_train.extend(train[:n_data // len(DOMAIN_GENERATORS)])
    mixed_encoded = [tok.encode(s) for s in mixed_train]

    # Train FP16 base
    print("\n[2/5] Training FP16 base model...")
    fp16_base = init_model(V, d=d, H=H, L=L, seed=seed)
    fp16_base = train_base(fp16_base, mixed_encoded, tok.pad_id,
                           epochs=base_epochs, verbose=verbose)

    # Create ternary base
    print("\n[3/5] Quantizing base to ternary...")
    ternary_base, scales = quantize_model_to_ternary(fp16_base)

    # Train LoRA adapters on each base
    print("\n[4/5] Training LoRA adapters (5 domains x 2 bases)...")
    fp16_loras = {}
    ternary_loras = {}
    fp16_deltas = {}
    ternary_deltas = {}

    for di, name in enumerate(domain_names):
        lora_seed = seed * 100 + di

        # FP16
        print(f"  Training on FP16 base: {name}...")
        lora_fp16 = init_lora(fp16_base, rank=r, seed=lora_seed)
        lora_fp16 = train_lora(fp16_base, lora_fp16, domain_data[name], tok.pad_id,
                                epochs=lora_epochs, lr=0.003, verbose=False)
        fp16_loras[name] = lora_fp16
        fp16_deltas[name] = lora_to_delta(lora_fp16, fp16_base)

        # Ternary
        print(f"  Training on ternary base: {name}...")
        lora_ternary = init_lora(ternary_base, rank=r, seed=lora_seed)
        lora_ternary = train_lora(ternary_base, lora_ternary, domain_data[name], tok.pad_id,
                                   epochs=lora_epochs, lr=0.003, verbose=False)
        ternary_loras[name] = lora_ternary
        ternary_deltas[name] = lora_to_delta(lora_ternary, ternary_base)

    # ===================================================================
    # MEASUREMENT 1: Per-adapter delta norms
    # ===================================================================
    print("\n[5/5] Running magnitude analysis...")

    fp16_total_norms = {}
    ternary_total_norms = {}
    fp16_layer_norms = {}
    ternary_layer_norms = {}

    for name in domain_names:
        fp16_flat = flatten_delta(fp16_deltas[name])
        ternary_flat = flatten_delta(ternary_deltas[name])
        fp16_total_norms[name] = float(onp.linalg.norm(fp16_flat))
        ternary_total_norms[name] = float(onp.linalg.norm(ternary_flat))

        fp16_layer_norms[name] = per_layer_delta_norms(fp16_loras[name], fp16_base, r)
        ternary_layer_norms[name] = per_layer_delta_norms(ternary_loras[name], ternary_base, r)

    fp16_norm_values = list(fp16_total_norms.values())
    ternary_norm_values = list(ternary_total_norms.values())

    fp16_norm_mean = float(onp.mean(fp16_norm_values))
    fp16_norm_std = float(onp.std(fp16_norm_values))
    fp16_norm_cv = fp16_norm_std / max(fp16_norm_mean, 1e-10)
    fp16_norm_var = float(onp.var(fp16_norm_values))

    ternary_norm_mean = float(onp.mean(ternary_norm_values))
    ternary_norm_std = float(onp.std(ternary_norm_values))
    ternary_norm_cv = ternary_norm_std / max(ternary_norm_mean, 1e-10)
    ternary_norm_var = float(onp.var(ternary_norm_values))

    fp16_max_min_ratio = max(fp16_norm_values) / max(min(fp16_norm_values), 1e-10)
    ternary_max_min_ratio = max(ternary_norm_values) / max(min(ternary_norm_values), 1e-10)

    print(f"\n  Delta Norms (||B_i @ A_i||_F):")
    print(f"    {'Domain':12s} {'FP16':>10s} {'Ternary':>10s} {'Ratio':>8s}")
    for name in domain_names:
        ratio = ternary_total_norms[name] / max(fp16_total_norms[name], 1e-10)
        print(f"    {name:12s} {fp16_total_norms[name]:10.4f} "
              f"{ternary_total_norms[name]:10.4f} {ratio:8.3f}")
    print(f"    {'---':12s} {'---':>10s} {'---':>10s} {'---':>8s}")
    print(f"    {'Mean':12s} {fp16_norm_mean:10.4f} {ternary_norm_mean:10.4f}")
    print(f"    {'Std':12s} {fp16_norm_std:10.4f} {ternary_norm_std:10.4f}")
    print(f"    {'CV':12s} {fp16_norm_cv:10.4f} {ternary_norm_cv:10.4f}")
    print(f"    {'Var':12s} {fp16_norm_var:10.4f} {ternary_norm_var:10.4f}")
    print(f"    {'Max/Min':12s} {fp16_max_min_ratio:10.4f} {ternary_max_min_ratio:10.4f}")

    # ===================================================================
    # MEASUREMENT 2: Per-layer delta norm decomposition
    # ===================================================================
    layer_keys = sorted(set(
        k for norms in fp16_layer_norms.values() for k in norms.keys()
    ))

    per_layer_summary = {}
    for lk in layer_keys:
        fp16_vals = [fp16_layer_norms[name][lk]['delta_norm']
                     for name in domain_names if lk in fp16_layer_norms[name]]
        ternary_vals = [ternary_layer_norms[name][lk]['delta_norm']
                        for name in domain_names if lk in ternary_layer_norms[name]]
        if fp16_vals and ternary_vals:
            per_layer_summary[lk] = {
                'fp16_mean': float(onp.mean(fp16_vals)),
                'fp16_std': float(onp.std(fp16_vals)),
                'fp16_cv': float(onp.std(fp16_vals) / max(onp.mean(fp16_vals), 1e-10)),
                'ternary_mean': float(onp.mean(ternary_vals)),
                'ternary_std': float(onp.std(ternary_vals)),
                'ternary_cv': float(onp.std(ternary_vals) / max(onp.mean(ternary_vals), 1e-10)),
                'norm_ratio': float(onp.mean(ternary_vals) / max(onp.mean(fp16_vals), 1e-10)),
            }

    print(f"\n  Per-Layer Delta Norm Summary (mean across domains):")
    print(f"    {'Layer':20s} {'FP16 mean':>10s} {'FP16 CV':>8s} "
          f"{'Tern mean':>10s} {'Tern CV':>8s} {'Ratio':>8s}")
    for lk in sorted(per_layer_summary.keys()):
        s = per_layer_summary[lk]
        print(f"    {lk:20s} {s['fp16_mean']:10.4f} {s['fp16_cv']:8.4f} "
              f"{s['ternary_mean']:10.4f} {s['ternary_cv']:8.4f} {s['norm_ratio']:8.3f}")

    # ===================================================================
    # MEASUREMENT 3: Activation magnitudes
    # ===================================================================
    fp16_act_stats = {}
    ternary_act_stats = {}
    fp16_logit_stats = {}
    ternary_logit_stats = {}

    for name in domain_names:
        # FP16 with adapter applied
        fp16_effective = apply_lora(fp16_base, fp16_loras[name])
        fp16_effective_np = {k: onp.array(v) if not isinstance(v, (dict, onp.ndarray)) else v
                             for k, v in fp16_effective.items()}
        fp16_layers, fp16_logits = measure_activation_magnitudes(
            fp16_effective_np, domain_eval[name], tok.pad_id)
        fp16_act_stats[name] = fp16_layers
        fp16_logit_stats[name] = fp16_logits

        # Ternary with adapter applied
        ternary_effective = apply_lora(ternary_base, ternary_loras[name])
        ternary_effective_np = {k: onp.array(v) if not isinstance(v, (dict, onp.ndarray)) else v
                                 for k, v in ternary_effective.items()}
        ternary_layers, ternary_logits = measure_activation_magnitudes(
            ternary_effective_np, domain_eval[name], tok.pad_id)
        ternary_act_stats[name] = ternary_layers
        ternary_logit_stats[name] = ternary_logits

    # Aggregate activation stats across domains
    print(f"\n  Activation Magnitudes (mean across domains):")
    for layer_key in ['layer_0', 'layer_1']:
        fp16_pre = onp.mean([fp16_act_stats[n][layer_key]['pre_norm'] for n in domain_names])
        ternary_pre = onp.mean([ternary_act_stats[n][layer_key]['pre_norm'] for n in domain_names])
        fp16_post_ffn = onp.mean([fp16_act_stats[n][layer_key]['post_ffn_norm'] for n in domain_names])
        ternary_post_ffn = onp.mean([ternary_act_stats[n][layer_key]['post_ffn_norm'] for n in domain_names])
        print(f"    {layer_key}: FP16 pre={fp16_pre:.3f} post_ffn={fp16_post_ffn:.3f} | "
              f"Ternary pre={ternary_pre:.3f} post_ffn={ternary_post_ffn:.3f}")

    # ===================================================================
    # MEASUREMENT 4: Logit-scale distributions
    # ===================================================================
    fp16_logit_stds = [fp16_logit_stats[n]['per_token_std_mean'] for n in domain_names]
    ternary_logit_stds = [ternary_logit_stats[n]['per_token_std_mean'] for n in domain_names]

    fp16_logit_std_cv = float(onp.std(fp16_logit_stds) / max(onp.mean(fp16_logit_stds), 1e-10))
    ternary_logit_std_cv = float(onp.std(ternary_logit_stds) / max(onp.mean(ternary_logit_stds), 1e-10))

    print(f"\n  Logit-Scale Distributions (per-token std):")
    print(f"    {'Domain':12s} {'FP16 std':>10s} {'Ternary std':>12s}")
    for name in domain_names:
        print(f"    {name:12s} {fp16_logit_stats[name]['per_token_std_mean']:10.4f} "
              f"{ternary_logit_stats[name]['per_token_std_mean']:12.4f}")
    print(f"    {'CV':12s} {fp16_logit_std_cv:10.4f} {ternary_logit_std_cv:12.4f}")

    # ===================================================================
    # MEASUREMENT 5: Composition delta norms
    # ===================================================================
    fp16_comp_delta = compose_deltas(list(fp16_deltas.values()), mode='equal')
    ternary_comp_delta = compose_deltas(list(ternary_deltas.values()), mode='equal')

    fp16_comp_norm = float(onp.linalg.norm(flatten_delta(fp16_comp_delta)))
    ternary_comp_norm = float(onp.linalg.norm(flatten_delta(ternary_comp_delta)))

    # Also measure sum (not averaged) composition
    fp16_sum_delta = compose_deltas(list(fp16_deltas.values()), mode='sum')
    ternary_sum_delta = compose_deltas(list(ternary_deltas.values()), mode='sum')
    fp16_sum_norm = float(onp.linalg.norm(flatten_delta(fp16_sum_delta)))
    ternary_sum_norm = float(onp.linalg.norm(flatten_delta(ternary_sum_delta)))

    # Expected norm under perfect orthogonality: sqrt(sum(||delta_i||^2)) / N
    fp16_expected_ortho = float(onp.sqrt(sum(n**2 for n in fp16_norm_values)) / len(fp16_norm_values))
    ternary_expected_ortho = float(onp.sqrt(sum(n**2 for n in ternary_norm_values)) / len(ternary_norm_values))

    # Composition efficiency: actual / expected (1.0 = perfect, >1 = constructive, <1 = destructive)
    fp16_comp_efficiency = fp16_comp_norm / max(fp16_expected_ortho, 1e-10)
    ternary_comp_efficiency = ternary_comp_norm / max(ternary_expected_ortho, 1e-10)

    print(f"\n  Composition Delta Norms:")
    print(f"    FP16    equal-weight: {fp16_comp_norm:.4f}, sum: {fp16_sum_norm:.4f}")
    print(f"    Ternary equal-weight: {ternary_comp_norm:.4f}, sum: {ternary_sum_norm:.4f}")
    print(f"    FP16    expected (ortho): {fp16_expected_ortho:.4f}, efficiency: {fp16_comp_efficiency:.4f}")
    print(f"    Ternary expected (ortho): {ternary_expected_ortho:.4f}, efficiency: {ternary_comp_efficiency:.4f}")

    # ===================================================================
    # MEASUREMENT 6: Base weight magnitude analysis
    # ===================================================================
    fp16_base_norms = {}
    ternary_base_norms = {}
    for k in fp16_base:
        if k == '_config' or not isinstance(fp16_base[k], onp.ndarray) or fp16_base[k].ndim < 2:
            continue
        fp16_base_norms[k] = float(onp.linalg.norm(fp16_base[k]))
        ternary_base_norms[k] = float(onp.linalg.norm(ternary_base[k]))

    # Delta-to-base ratio (signal strength of adapter relative to base)
    fp16_delta_base_ratios = {}
    ternary_delta_base_ratios = {}
    for name in domain_names:
        fp16_ratios = {}
        ternary_ratios = {}
        for k in fp16_deltas[name]:
            fp16_d_norm = float(onp.linalg.norm(fp16_deltas[name][k]))
            ternary_d_norm = float(onp.linalg.norm(ternary_deltas[name][k]))
            fp16_ratios[k] = fp16_d_norm / max(fp16_base_norms.get(k, 1e-10), 1e-10)
            ternary_ratios[k] = ternary_d_norm / max(ternary_base_norms.get(k, 1e-10), 1e-10)
        fp16_delta_base_ratios[name] = fp16_ratios
        ternary_delta_base_ratios[name] = ternary_ratios

    # Mean delta/base ratio across all layers and domains
    all_fp16_ratios = [v for d in fp16_delta_base_ratios.values() for v in d.values()]
    all_ternary_ratios = [v for d in ternary_delta_base_ratios.values() for v in d.values()]

    print(f"\n  Adapter Signal Strength (||delta|| / ||base_weight||):")
    print(f"    FP16    mean: {onp.mean(all_fp16_ratios):.4f}, std: {onp.std(all_fp16_ratios):.4f}")
    print(f"    Ternary mean: {onp.mean(all_ternary_ratios):.4f}, std: {onp.std(all_ternary_ratios):.4f}")

    # ===================================================================
    # Build results
    # ===================================================================
    results = {
        'seed': seed,
        'd': d,
        'r': r,
        'L': L,
        'n_domains': len(domain_names),
        'delta_norms': {
            'fp16': fp16_total_norms,
            'ternary': ternary_total_norms,
            'fp16_mean': fp16_norm_mean,
            'fp16_std': fp16_norm_std,
            'fp16_cv': fp16_norm_cv,
            'fp16_var': fp16_norm_var,
            'ternary_mean': ternary_norm_mean,
            'ternary_std': ternary_norm_std,
            'ternary_cv': ternary_norm_cv,
            'ternary_var': ternary_norm_var,
            'fp16_max_min_ratio': fp16_max_min_ratio,
            'ternary_max_min_ratio': ternary_max_min_ratio,
        },
        'per_layer_norms': per_layer_summary,
        'activation_magnitudes': {
            'fp16': fp16_act_stats,
            'ternary': ternary_act_stats,
        },
        'logit_distributions': {
            'fp16': fp16_logit_stats,
            'ternary': ternary_logit_stats,
            'fp16_cross_domain_cv': fp16_logit_std_cv,
            'ternary_cross_domain_cv': ternary_logit_std_cv,
        },
        'composition_norms': {
            'fp16_equal_weight': fp16_comp_norm,
            'ternary_equal_weight': ternary_comp_norm,
            'fp16_sum': fp16_sum_norm,
            'ternary_sum': ternary_sum_norm,
            'fp16_expected_ortho': fp16_expected_ortho,
            'ternary_expected_ortho': ternary_expected_ortho,
            'fp16_composition_efficiency': fp16_comp_efficiency,
            'ternary_composition_efficiency': ternary_comp_efficiency,
        },
        'signal_strength': {
            'fp16_mean_delta_base_ratio': float(onp.mean(all_fp16_ratios)),
            'fp16_std_delta_base_ratio': float(onp.std(all_fp16_ratios)),
            'ternary_mean_delta_base_ratio': float(onp.mean(all_ternary_ratios)),
            'ternary_std_delta_base_ratio': float(onp.std(all_ternary_ratios)),
        },
        'kill_criteria': {
            'K1_fp16_norm_var': fp16_norm_var,
            'K1_ternary_norm_var': ternary_norm_var,
            'K1_pass': bool(ternary_norm_var < fp16_norm_var),
            'K2_ternary_max_min_ratio': ternary_max_min_ratio,
            'K2_threshold': 10.0,
            'K2_pass': bool(ternary_max_min_ratio <= 10.0),
        },
    }

    return results


def run_all(seeds=(42, 123, 314), d=64, r=4):
    """Run across multiple seeds and aggregate."""
    t0 = time.time()
    all_results = []

    for seed in seeds:
        result = run_seed(seed=seed, d=d, r=r, verbose=True)
        all_results.append(result)

    # Aggregate
    agg = {
        'config': {'d': d, 'r': r, 'L': 2, 'seeds': list(seeds), 'n_domains': 5},
        'per_seed': all_results,
    }

    # Aggregate delta norm statistics
    fp16_vars = [r['delta_norms']['fp16_var'] for r in all_results]
    ternary_vars = [r['delta_norms']['ternary_var'] for r in all_results]
    fp16_cvs = [r['delta_norms']['fp16_cv'] for r in all_results]
    ternary_cvs = [r['delta_norms']['ternary_cv'] for r in all_results]
    fp16_ratios = [r['delta_norms']['fp16_max_min_ratio'] for r in all_results]
    ternary_ratios = [r['delta_norms']['ternary_max_min_ratio'] for r in all_results]
    fp16_means = [r['delta_norms']['fp16_mean'] for r in all_results]
    ternary_means = [r['delta_norms']['ternary_mean'] for r in all_results]

    # Logit CV
    fp16_logit_cvs = [r['logit_distributions']['fp16_cross_domain_cv'] for r in all_results]
    ternary_logit_cvs = [r['logit_distributions']['ternary_cross_domain_cv'] for r in all_results]

    # Composition efficiency
    fp16_effs = [r['composition_norms']['fp16_composition_efficiency'] for r in all_results]
    ternary_effs = [r['composition_norms']['ternary_composition_efficiency'] for r in all_results]

    # Signal strength
    fp16_signals = [r['signal_strength']['fp16_mean_delta_base_ratio'] for r in all_results]
    ternary_signals = [r['signal_strength']['ternary_mean_delta_base_ratio'] for r in all_results]

    agg['aggregate'] = {
        'delta_norm_variance': {
            'fp16_mean': float(onp.mean(fp16_vars)),
            'fp16_std': float(onp.std(fp16_vars)),
            'ternary_mean': float(onp.mean(ternary_vars)),
            'ternary_std': float(onp.std(ternary_vars)),
            'ternary_lower_all_seeds': bool(all(
                r['delta_norms']['ternary_var'] < r['delta_norms']['fp16_var']
                for r in all_results)),
        },
        'delta_norm_cv': {
            'fp16_mean': float(onp.mean(fp16_cvs)),
            'ternary_mean': float(onp.mean(ternary_cvs)),
        },
        'delta_norm_mean': {
            'fp16_mean': float(onp.mean(fp16_means)),
            'ternary_mean': float(onp.mean(ternary_means)),
            'ratio': float(onp.mean(ternary_means) / max(onp.mean(fp16_means), 1e-10)),
        },
        'max_min_ratio': {
            'fp16_mean': float(onp.mean(fp16_ratios)),
            'ternary_mean': float(onp.mean(ternary_ratios)),
        },
        'logit_cross_domain_cv': {
            'fp16_mean': float(onp.mean(fp16_logit_cvs)),
            'ternary_mean': float(onp.mean(ternary_logit_cvs)),
        },
        'composition_efficiency': {
            'fp16_mean': float(onp.mean(fp16_effs)),
            'ternary_mean': float(onp.mean(ternary_effs)),
        },
        'signal_strength': {
            'fp16_mean': float(onp.mean(fp16_signals)),
            'ternary_mean': float(onp.mean(ternary_signals)),
            'ratio': float(onp.mean(ternary_signals) / max(onp.mean(fp16_signals), 1e-10)),
        },
    }

    # Kill criteria aggregated
    k1_passes = sum(1 for r in all_results if r['kill_criteria']['K1_pass'])
    k2_passes = sum(1 for r in all_results if r['kill_criteria']['K2_pass'])

    agg['kill_criteria'] = {
        'K1_description': 'adapter norm variance on BitNet < variance on FP16',
        'K1_fp16_var_mean': float(onp.mean(fp16_vars)),
        'K1_ternary_var_mean': float(onp.mean(ternary_vars)),
        'K1_pass_rate': f"{k1_passes}/{len(seeds)}",
        'K1_pass': bool(k1_passes == len(seeds)),
        'K2_description': 'max/min adapter norm ratio on BitNet <= 10x',
        'K2_ternary_ratio_mean': float(onp.mean(ternary_ratios)),
        'K2_pass_rate': f"{k2_passes}/{len(seeds)}",
        'K2_pass': bool(k2_passes == len(seeds)),
    }

    elapsed = time.time() - t0
    agg['runtime_seconds'] = float(elapsed)

    # Print final summary
    print("\n" + "=" * 70)
    print("  AGGREGATE RESULTS")
    print("=" * 70)

    print(f"\n  Delta Norm Statistics (across {len(seeds)} seeds):")
    print(f"    FP16    mean norm: {onp.mean(fp16_means):.4f}")
    print(f"    Ternary mean norm: {onp.mean(ternary_means):.4f} "
          f"(ratio: {onp.mean(ternary_means)/max(onp.mean(fp16_means),1e-10):.3f})")
    print(f"    FP16    norm variance: {onp.mean(fp16_vars):.4f}")
    print(f"    Ternary norm variance: {onp.mean(ternary_vars):.4f}")
    print(f"    FP16    norm CV: {onp.mean(fp16_cvs):.4f}")
    print(f"    Ternary norm CV: {onp.mean(ternary_cvs):.4f}")
    print(f"    FP16    max/min ratio: {onp.mean(fp16_ratios):.4f}")
    print(f"    Ternary max/min ratio: {onp.mean(ternary_ratios):.4f}")

    print(f"\n  Logit Cross-Domain CV:")
    print(f"    FP16:    {onp.mean(fp16_logit_cvs):.4f}")
    print(f"    Ternary: {onp.mean(ternary_logit_cvs):.4f}")

    print(f"\n  Composition Efficiency (actual/expected norm):")
    print(f"    FP16:    {onp.mean(fp16_effs):.4f}")
    print(f"    Ternary: {onp.mean(ternary_effs):.4f}")

    print(f"\n  Signal Strength (||delta||/||base||):")
    print(f"    FP16:    {onp.mean(fp16_signals):.4f}")
    print(f"    Ternary: {onp.mean(ternary_signals):.4f} "
          f"(ratio: {onp.mean(ternary_signals)/max(onp.mean(fp16_signals),1e-10):.3f})")

    print(f"\n  Kill Criteria:")
    print(f"    K1 (ternary var < fp16 var): {agg['kill_criteria']['K1_pass_rate']} "
          f"({'PASS' if agg['kill_criteria']['K1_pass'] else 'FAIL'})")
    print(f"       FP16 var={onp.mean(fp16_vars):.4f}, Ternary var={onp.mean(ternary_vars):.4f}")
    print(f"    K2 (ternary max/min <= 10x): {agg['kill_criteria']['K2_pass_rate']} "
          f"({'PASS' if agg['kill_criteria']['K2_pass'] else 'FAIL'})")
    print(f"       Ternary max/min={onp.mean(ternary_ratios):.4f}")

    print(f"\n  Runtime: {elapsed:.1f}s")

    return agg


def main():
    out_dir = Path(__file__).parent
    agg = run_all(seeds=(42, 123, 314), d=64, r=4)

    results_path = out_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
