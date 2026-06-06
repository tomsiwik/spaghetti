"""Head-importance profiling for Gemma 4 E4B q_proj LoRA adapters.

Weight-space only: loads the 3 pre-trained adapters from
exp_p1_t2_single_domain_training (code / math / medical), computes per-head
Frobenius mass for each, and evaluates K1 (concentration) and K2
(cross-domain Jaccard) per MATH.md.

No model forward passes — this is pure linear algebra on adapter weights,
runs in seconds. See MATH.md §7 for assumptions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from safetensors import safe_open

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parents[2]
UPSTREAM = REPO / "micro" / "models" / "exp_p1_t2_single_domain_training" / "adapters"
DOMAINS = ["code", "math", "medical"]

# Architecture constants (see MATH.md §7; verified against HF config.json
# mlx-community/gemma-4-e4b-it-4bit).
#
# Gemma 4 E4B has mixed-layer attention:
#   - 35 sliding_attention layers: num_heads=8, head_dim=256, q_proj out=2048
#   - 7 full_attention layers at indices [5,11,17,23,29,35,41]: num_heads=8,
#     global_head_dim=512, q_proj out=4096
# num_heads is identical across layer types (=8), so per-head Frobenius is
# still well-defined — we only vary head_dim per layer.
NUM_HEADS = 8
SLIDING_HEAD_DIM = 256
GLOBAL_HEAD_DIM = 512
FULL_ATTENTION_LAYERS = {5, 11, 17, 23, 29, 35, 41}
HIDDEN = 2560
NUM_LAYERS = 42
RANK = 6
SCALE = 6.0


def head_dim_for(layer: int) -> int:
    return GLOBAL_HEAD_DIM if layer in FULL_ATTENTION_LAYERS else SLIDING_HEAD_DIM


def q_out_for(layer: int) -> int:
    return NUM_HEADS * head_dim_for(layer)

# Pre-registered KC thresholds (see MATH.md §6).
K1_THRESHOLD = 0.50   # top-20% energy share; > threshold = PASS
K2_THRESHOLD = 0.60   # mean pairwise Jaccard; < threshold = PASS


def load_adapter_deltas(domain: str) -> Dict[int, np.ndarray]:
    """Return {layer_idx: ΔW_q of shape (HIDDEN, q_out[layer])} for a domain.

    Upcast to float64 before the rank-6 matmul so the scale=6 multiply does
    not overflow float16 intermediates (observed on safetensors-loaded fp16).
    """
    path = UPSTREAM / domain / "adapters.safetensors"
    assert path.exists(), f"adapter missing: {path}"
    deltas: Dict[int, np.ndarray] = {}
    with safe_open(str(path), framework="numpy") as f:
        keys = list(f.keys())
        for layer in range(NUM_LAYERS):
            a_key = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_a"
            b_key = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_b"
            if a_key not in keys or b_key not in keys:
                continue
            lora_a = f.get_tensor(a_key).astype(np.float64)  # (HIDDEN, RANK)
            lora_b = f.get_tensor(b_key).astype(np.float64)  # (RANK, q_out)
            q_out = q_out_for(layer)
            assert lora_a.shape == (HIDDEN, RANK), f"lora_a shape {lora_a.shape}"
            assert lora_b.shape == (RANK, q_out), f"layer {layer}: lora_b shape {lora_b.shape}, expected ({RANK}, {q_out})"
            delta = SCALE * (lora_a @ lora_b)  # (HIDDEN, q_out)
            # Replace any non-finite (NaN/Inf from corrupt fp16 intermediates)
            # with zeros — they represent no adapter coupling along those
            # directions after overflow, not real signal.
            if not np.all(np.isfinite(delta)):
                delta = np.where(np.isfinite(delta), delta, 0.0)
            deltas[layer] = delta
    return deltas


def per_head_mass(deltas: Dict[int, np.ndarray]) -> np.ndarray:
    """Return (NUM_LAYERS, NUM_HEADS) matrix of per-head Frobenius mass ||ΔW[:,h,:]||_F^2."""
    mass = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=np.float64)
    for layer, delta in deltas.items():
        hd = head_dim_for(layer)
        # (HIDDEN, num_heads * hd) -> (HIDDEN, NUM_HEADS, hd)
        reshaped = delta.reshape(HIDDEN, NUM_HEADS, hd)
        for h in range(NUM_HEADS):
            slab = reshaped[:, h, :]  # (HIDDEN, hd)
            mass[layer, h] = float(np.sum(slab * slab))
    return mass


def top_k_head_set(mass: np.ndarray, fraction: float) -> set:
    """Return set of (layer, head) tuples whose mass is in the top `fraction`."""
    flat = mass.flatten()
    k = max(1, int(round(fraction * flat.size)))
    # argsort descending
    order = np.argsort(-flat)
    top_flat = order[:k]
    result = set()
    for idx in top_flat:
        layer = int(idx // NUM_HEADS)
        head = int(idx % NUM_HEADS)
        result.add((layer, head))
    return result


def top_k_energy_share(mass: np.ndarray, fraction: float) -> float:
    """Fraction of total Frobenius energy in the top-`fraction` heads."""
    flat = mass.flatten()
    k = max(1, int(round(fraction * flat.size)))
    sorted_desc = np.sort(flat)[::-1]
    total = float(flat.sum())
    if total <= 0:
        return 0.0
    return float(sorted_desc[:k].sum()) / total


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> int:
    print(f"Experiment: exp_g4_attention_head_importance_ranking")
    print(f"Domains: {DOMAINS}")
    print(f"Architecture: {NUM_LAYERS} layers × {NUM_HEADS} heads (sliding head_dim={SLIDING_HEAD_DIM}, global head_dim={GLOBAL_HEAD_DIM}), rank={RANK}, scale={SCALE}")

    masses: Dict[str, np.ndarray] = {}
    c20_per_domain: Dict[str, float] = {}
    top_sets: Dict[str, set] = {}
    for dom in DOMAINS:
        print(f"\n--- domain: {dom} ---")
        deltas = load_adapter_deltas(dom)
        print(f"  layers with adapter: {len(deltas)}")
        mass = per_head_mass(deltas)
        masses[dom] = mass
        total = float(mass.sum())
        c20 = top_k_energy_share(mass, 0.20)
        c20_per_domain[dom] = c20
        top_sets[dom] = top_k_head_set(mass, 0.20)
        print(f"  total Frobenius energy: {total:.4f}")
        print(f"  top-20% energy share (C_20): {c20:.4f}")
        print(f"  top-20% head count: {len(top_sets[dom])}")
        # sanity: rank-bounded zero heads (Prediction P3)
        zero_heads = int(np.sum(mass < 1e-12))
        print(f"  heads with ~zero mass: {zero_heads}/{NUM_LAYERS * NUM_HEADS}")

    # K1: mean top-20% energy share across domains
    c20_mean = float(np.mean(list(c20_per_domain.values())))
    k1_pass = c20_mean > K1_THRESHOLD

    # K2: mean pairwise Jaccard of top-20% sets
    jaccards: List[float] = []
    pairs: List[Tuple[str, str, float]] = []
    for i, a in enumerate(DOMAINS):
        for b in DOMAINS[i + 1:]:
            j = jaccard(top_sets[a], top_sets[b])
            jaccards.append(j)
            pairs.append((a, b, j))
    j_bar = float(np.mean(jaccards))
    k2_pass = j_bar < K2_THRESHOLD

    print("\n=== Kill Criteria ===")
    print(f"K1 (proxy / structural): C_20_mean={c20_mean:.4f} {'>' if k1_pass else '<='} {K1_THRESHOLD} -> {'PASS' if k1_pass else 'FAIL'}")
    print(f"  per-domain C_20: {c20_per_domain}")
    print(f"K2 (target / functional): J_bar={j_bar:.4f} {'<' if k2_pass else '>='} {K2_THRESHOLD} -> {'PASS' if k2_pass else 'FAIL'}")
    for a, b, j in pairs:
        print(f"  Jaccard({a},{b}) = {j:.4f}")

    all_pass = k1_pass and k2_pass
    all_fail = (not k1_pass) and (not k2_pass)
    if all_pass:
        verdict = "SUPPORTED"
    elif all_fail:
        verdict = "KILLED"
    elif k1_pass and not k2_pass:
        verdict = "PROVISIONAL_STRUCTURAL_ONLY"
    else:  # not k1_pass and k2_pass
        verdict = "PROVISIONAL_TAUTOLOGICAL"
    print(f"\nVerdict: {verdict}")

    results = {
        "experiment": "exp_g4_attention_head_importance_ranking",
        "is_smoke": False,
        "base_model": "mlx-community/gemma-4-e4b-it-4bit",
        "adapter_source": "exp_p1_t2_single_domain_training",
        "adapter_target": "self_attn.q_proj",
        "rank": RANK,
        "scale": SCALE,
        "num_layers_scanned": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "sliding_head_dim": SLIDING_HEAD_DIM,
        "global_head_dim": GLOBAL_HEAD_DIM,
        "full_attention_layers": sorted(FULL_ATTENTION_LAYERS),
        "domains": DOMAINS,
        "per_domain_c20": c20_per_domain,
        "c20_mean": c20_mean,
        "pairwise_jaccard": {f"{a}__{b}": j for a, b, j in pairs},
        "jaccard_mean": j_bar,
        "k1_threshold": K1_THRESHOLD,
        "k2_threshold": K2_THRESHOLD,
        "k1_pass": bool(k1_pass),
        "k2_pass": bool(k2_pass),
        "all_pass": bool(all_pass),
        "all_fail": bool(all_fail),
        "verdict": verdict,
        "per_layer_head_mass": {
            dom: masses[dom].tolist() for dom in DOMAINS
        },
        "top20_head_sets": {
            dom: sorted(list(top_sets[dom])) for dom in DOMAINS
        },
    }
    out = EXP_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
