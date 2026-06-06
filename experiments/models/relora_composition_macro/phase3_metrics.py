#!/usr/bin/env python3
"""Phase 3 ONLY: Compute composition metrics from pre-trained adapters.

Standalone script -- only requires safetensors, numpy (no transformers/peft).

KEY OPTIMIZATION: Exploits rank-r factorization to avoid materializing full deltas.
  delta_m = s * B_m @ A_m    where B_m is (d_out, r), A_m is (r, d_in)
  dot(delta_i, delta_j) = s^2 * trace(A_i^T B_i^T B_j A_j)
                        = s^2 * sum(A_i * (C @ A_j))   where C = B_i^T @ B_j is (r, r)

This reduces O(d_out * d_in) per module to O(r^2 * d_in + r * d_out) -- 200x faster for
r=16, d_out=18944, d_in=3584.

Usage (on RunPod):
    cd /workspace/llm
    python experiments/models/relora_composition_macro/phase3_metrics.py
"""

import json
import math
import time
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

OUTPUT_DIR = Path(__file__).parent
DOMAINS = ["math", "python", "sql", "medical", "bash"]
LORA_RANK = 16
LORA_ALPHA = 16
ATTN_MODS = {"q_proj", "k_proj", "v_proj", "o_proj"}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_adapter_AB(adapter_path: Path) -> dict:
    """Load adapter and return dict of {mod_name: (A, B)} as numpy float32."""
    weights_file = adapter_path / "adapter_model.safetensors"
    if not weights_file.exists():
        import torch
        weights_file = adapter_path / "adapter_model.bin"
        weights = torch.load(weights_file, map_location="cpu", weights_only=True)
    else:
        weights = load_file(str(weights_file), device="cpu")

    modules = {}
    for key, tensor in weights.items():
        clean = key.replace("base_model.model.", "")
        if "lora_A" in clean:
            mod_name = clean.split(".lora_A")[0]
            modules.setdefault(mod_name, {})["A"] = tensor.float().numpy()
        elif "lora_B" in clean:
            mod_name = clean.split(".lora_B")[0]
            modules.setdefault(mod_name, {})["B"] = tensor.float().numpy()

    # Return only complete pairs
    return {k: (v["A"], v["B"]) for k, v in modules.items() if "A" in v and "B" in v}


def lowrank_dot(A_i, B_i, A_j, B_j, scaling_sq):
    """Compute dot(flat(s*B_i@A_i), flat(s*B_j@A_j)) without materializing full delta.

    = s^2 * trace(A_i^T @ B_i^T @ B_j @ A_j)
    = s^2 * sum(A_i * ((B_i^T @ B_j) @ A_j))

    A: (r, d_in), B: (d_out, r)
    B_i^T @ B_j: (r, r) -- tiny
    C @ A_j: (r, d_in) -- same size as A
    element-wise multiply with A_i and sum
    """
    C = B_i.T @ B_j         # (r, r) -- cheap
    CA_j = C @ A_j           # (r, d_in) -- cheap
    return scaling_sq * np.sum(A_i * CA_j)


def lowrank_sqnorm(A, B, scaling_sq):
    """Compute ||s * B @ A||^2 = s^2 * trace(A^T B^T B A) = s^2 * sum(A * (B^T B @ A))"""
    BtB = B.T @ B     # (r, r)
    BtBA = BtB @ A    # (r, d_in)
    return scaling_sq * np.sum(A * BtBA)


def main():
    results_file = OUTPUT_DIR / "results.json"
    log("=" * 72)
    log("ReLoRA COMPOSITION MACRO -- Phase 3: Low-rank Metrics")
    log(f"  Domains: {DOMAINS}")
    log("=" * 72)

    t0 = time.time()

    conv_dirs = [OUTPUT_DIR / "conventional" / d for d in DOMAINS]
    relora_dirs = [OUTPUT_DIR / "relora" / d for d in DOMAINS]

    # Verify
    missing = []
    for d in conv_dirs + relora_dirs:
        if not (d / "adapter_model.safetensors").exists() and \
           not (d / "adapter_model.bin").exists():
            missing.append(str(d))
    if missing:
        log(f"ERROR: Missing adapters: {missing}")
        return

    # Load training metadata
    expert_metas = {"conventional": {}, "relora": {}}
    for domain in DOMAINS:
        for cond in ["conventional", "relora"]:
            meta_path = OUTPUT_DIR / cond / domain / "training_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    expert_metas[cond][domain] = json.load(f)

    scaling_sq = (LORA_ALPHA / LORA_RANK) ** 2

    def compute_condition(dirs, label):
        """Compute pairwise cosines for a set of expert dirs using low-rank trick."""
        n = len(dirs)

        log(f"\n  Loading {label} adapter A/B matrices...")
        all_AB = []
        for p in dirs:
            ab = load_adapter_AB(p)
            all_AB.append(ab)
            log(f"    {p.name}: {len(ab)} modules loaded")

        mod_names = sorted(all_AB[0].keys())
        log(f"    {len(mod_names)} LoRA modules")

        # Accumulators
        sq_norms = np.zeros(n)
        dots = np.zeros((n, n))
        sq_norms_attn = np.zeros(n)
        sq_norms_ffn = np.zeros(n)
        dots_attn = np.zeros((n, n))
        dots_ffn = np.zeros((n, n))

        total_delta_dim = 0
        t_start = time.time()

        for m_idx, mod_name in enumerate(mod_names):
            is_attn = any(am in mod_name for am in ATTN_MODS)

            # Get A, B for each expert on this module
            ABs = []
            for i in range(n):
                if mod_name in all_AB[i]:
                    ABs.append(all_AB[i][mod_name])  # (A, B)
                else:
                    ABs.append(None)

            # Track dim from first valid entry
            if ABs[0] is not None:
                A0, B0 = ABs[0]
                total_delta_dim += B0.shape[0] * A0.shape[1]  # d_out * d_in

            # Accumulate
            for i in range(n):
                if ABs[i] is None:
                    continue
                A_i, B_i = ABs[i]
                sn = lowrank_sqnorm(A_i, B_i, scaling_sq)
                sq_norms[i] += sn
                if is_attn:
                    sq_norms_attn[i] += sn
                else:
                    sq_norms_ffn[i] += sn

                for j in range(i + 1, n):
                    if ABs[j] is None:
                        continue
                    A_j, B_j = ABs[j]
                    d = lowrank_dot(A_i, B_i, A_j, B_j, scaling_sq)
                    dots[i][j] += d
                    if is_attn:
                        dots_attn[i][j] += d
                    else:
                        dots_ffn[i][j] += d

            if (m_idx + 1) % 50 == 0 or m_idx == len(mod_names) - 1:
                elapsed_m = time.time() - t_start
                log(f"    Processed {m_idx+1}/{len(mod_names)} modules ({elapsed_m:.1f}s)")

        # Compute cosines
        domain_names = [p.name for p in dirs]
        pairs, cos_vals = [], []
        for i in range(n):
            for j in range(i + 1, n):
                ni, nj = math.sqrt(sq_norms[i]), math.sqrt(sq_norms[j])
                cos = dots[i][j] / (ni * nj) if ni > 1e-12 and nj > 1e-12 else 0.0
                pairs.append({
                    "expert_i": domain_names[i],
                    "expert_j": domain_names[j],
                    "cosine": float(cos),
                    "abs_cosine": float(abs(cos)),
                })
                cos_vals.append(abs(cos))

        # Per-type cosines
        attn_cos_vals, ffn_cos_vals = [], []
        for i in range(n):
            for j in range(i + 1, n):
                ni_a, nj_a = math.sqrt(sq_norms_attn[i]), math.sqrt(sq_norms_attn[j])
                if ni_a > 1e-12 and nj_a > 1e-12:
                    attn_cos_vals.append(abs(dots_attn[i][j] / (ni_a * nj_a)))
                ni_f, nj_f = math.sqrt(sq_norms_ffn[i]), math.sqrt(sq_norms_ffn[j])
                if ni_f > 1e-12 and nj_f > 1e-12:
                    ffn_cos_vals.append(abs(dots_ffn[i][j] / (ni_f * nj_f)))

        norms = {domain_names[i]: float(math.sqrt(sq_norms[i])) for i in range(n)}

        cosine_metrics = {
            "mean_abs_cos": float(np.mean(cos_vals)) if cos_vals else 0,
            "max_abs_cos": float(np.max(cos_vals)) if cos_vals else 0,
            "min_abs_cos": float(np.min(cos_vals)) if cos_vals else 0,
            "std_abs_cos": float(np.std(cos_vals)) if cos_vals else 0,
            "pairs": pairs,
            "n_experts": n,
            "delta_dim": total_delta_dim,
            "norms": norms,
        }

        per_mod = {
            "attn_mean_abs_cos": float(np.mean(attn_cos_vals)) if attn_cos_vals else 0,
            "ffn_mean_abs_cos": float(np.mean(ffn_cos_vals)) if ffn_cos_vals else 0,
            "attn_max_abs_cos": float(np.max(attn_cos_vals)) if attn_cos_vals else 0,
            "ffn_max_abs_cos": float(np.max(ffn_cos_vals)) if ffn_cos_vals else 0,
        }

        log(f"    mean|cos| = {cosine_metrics['mean_abs_cos']:.8f}")
        log(f"    max|cos|  = {cosine_metrics['max_abs_cos']:.8f}")
        log(f"    delta_dim = {cosine_metrics['delta_dim']:,}")
        log(f"    attn mean|cos| = {per_mod['attn_mean_abs_cos']:.8f}")
        log(f"    ffn  mean|cos| = {per_mod['ffn_mean_abs_cos']:.8f}")
        for d, n_val in norms.items():
            log(f"    norm({d}) = {n_val:.6f}")

        return cosine_metrics, per_mod

    # Run both conditions
    conv_metrics, conv_per_mod = compute_condition(conv_dirs, "conventional")
    relora_metrics, relora_per_mod = compute_condition(relora_dirs, "ReLoRA")

    # Kill criteria
    log("\n=== Kill Criteria Evaluation ===")

    cos_ratio = relora_metrics["mean_abs_cos"] / (conv_metrics["mean_abs_cos"] + 1e-12)

    conv_losses = {d: expert_metas["conventional"].get(d, {}).get("train_loss", 0)
                   for d in DOMAINS}
    relora_losses = {d: expert_metas["relora"].get(d, {}).get("train_loss", 0)
                     for d in DOMAINS}
    conv_mean_loss = np.mean([v for v in conv_losses.values() if v > 0])
    relora_mean_loss = np.mean([v for v in relora_losses.values() if v > 0])
    loss_ratio = relora_mean_loss / (conv_mean_loss + 1e-12)

    k1 = cos_ratio > 5.0
    log(f"  K1: cos_ratio = {cos_ratio:.4f} (threshold: >5x) -> "
        f"{'KILLED' if k1 else 'SURVIVES'}")

    k2 = loss_ratio > 1.20
    log(f"  K2: loss_ratio = {loss_ratio:.4f} (threshold: >1.20) -> "
        f"{'KILLED' if k2 else 'SURVIVES'}")

    k3 = False
    log(f"  K3: base gap = approximated by loss_ratio")

    if k1 or k2:
        verdict = "KILLED"
    elif cos_ratio < 2.0 and loss_ratio < 1.10:
        verdict = "PROVEN"
    else:
        verdict = "SUPPORTED"

    log(f"\n  VERDICT: {verdict}")

    # Random baseline
    D = conv_metrics["delta_dim"]
    random_expected = math.sqrt(2 / (math.pi * D)) if D > 0 else 0
    log(f"\n  Random baseline E[|cos|] = {random_expected:.2e} (D={D:,})")
    conv_vs_random = conv_metrics['mean_abs_cos'] / (random_expected + 1e-12)
    relora_vs_random = relora_metrics['mean_abs_cos'] / (random_expected + 1e-12)
    log(f"  Conventional / random = {conv_vs_random:.1f}x")
    log(f"  ReLoRA / random = {relora_vs_random:.1f}x")

    # Scaling comparison
    log("\n  Scaling trend (micro -> macro):")
    log(f"    Micro (d=64):   cos_ratio = 1.77x, loss_ratio = 1.052")
    log(f"    Macro (d=3584): cos_ratio = {cos_ratio:.4f}x, loss_ratio = {loss_ratio:.4f}")

    # Per-pair details
    log("\n  === Pairwise Cosine Details ===")
    log(f"  {'Pair':<25} {'Conv cos':>12} {'ReLoRA cos':>12} {'Ratio':>8}")
    log(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*8}")
    conv_pair_map = {(p["expert_i"], p["expert_j"]): p["cosine"]
                     for p in conv_metrics["pairs"]}
    for p in relora_metrics["pairs"]:
        key = (p["expert_i"], p["expert_j"])
        conv_cos = conv_pair_map.get(key, 0)
        pair_label = f"{p['expert_i']}-{p['expert_j']}"
        r = abs(p["cosine"]) / (abs(conv_cos) + 1e-12) if abs(conv_cos) > 1e-12 else float('inf')
        log(f"  {pair_label:<25} {conv_cos:>12.8f} {p['cosine']:>12.8f} {r:>8.2f}x")

    # Loss details
    log("\n  === Training Loss Details ===")
    log(f"  {'Domain':<12} {'Conv loss':>12} {'ReLoRA loss':>12} {'Ratio':>8}")
    log(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    for d in DOMAINS:
        cl = conv_losses.get(d, 0)
        rl = relora_losses.get(d, 0)
        r = rl / (cl + 1e-12)
        log(f"  {d:<12} {cl:>12.4f} {rl:>12.4f} {r:>8.4f}")
    log(f"  {'MEAN':<12} {conv_mean_loss:>12.4f} {relora_mean_loss:>12.4f} {loss_ratio:>8.4f}")

    elapsed = time.time() - t0

    # Load ReLoRA adapter metadata
    relora_meta = {}
    relora_meta_path = OUTPUT_DIR / "relora_adapter" / "relora_meta.json"
    if relora_meta_path.exists():
        with open(relora_meta_path) as f:
            relora_meta = json.load(f)

    # Save results
    results = {
        "experiment": "relora_composition_macro",
        "base_model": "Qwen2.5-7B",
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_layers": 28,
        "domains": DOMAINS,
        "config": {
            "relora_steps": 150,
            "relora_lr": 4e-4,
            "expert_steps": 100,
            "expert_lr": 2e-4,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
        },
        "conventional": {
            "cosines": {k: v for k, v in conv_metrics.items() if k != "norms"},
            "per_module": conv_per_mod,
            "train_losses": conv_losses,
            "mean_train_loss": float(conv_mean_loss),
            "delta_norms": conv_metrics.get("norms", {}),
        },
        "relora": {
            "cosines": {k: v for k, v in relora_metrics.items() if k != "norms"},
            "per_module": relora_per_mod,
            "train_losses": relora_losses,
            "mean_train_loss": float(relora_mean_loss),
            "delta_norms": relora_metrics.get("norms", {}),
            "adapter_meta": relora_meta,
        },
        "ratios": {
            "cos_ratio": float(cos_ratio),
            "loss_ratio": float(loss_ratio),
        },
        "random_baseline": {
            "expected_cos": float(random_expected),
            "delta_dim": D,
        },
        "kill_criteria": {
            "K1_cos_ratio_gt_5x": bool(k1),
            "K2_loss_ratio_gt_1_20": bool(k2),
            "K3_base_gap_gt_10pct": bool(k3),
        },
        "verdict": verdict,
        "scaling_comparison": {
            "micro_d64_cos_ratio": 1.77,
            "micro_d64_loss_ratio": 1.052,
            "macro_d3584_cos_ratio": float(cos_ratio),
            "macro_d3584_loss_ratio": float(loss_ratio),
        },
        "elapsed_phase3_s": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {results_file}")
    log(f"Phase 3 time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
