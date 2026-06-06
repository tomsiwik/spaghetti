"""
exp_intrinsic_dim_real_tasks
----------------------------
Measure the intrinsic dimensionality of the SFT B-matrices from v2.
SVD all B-matrices stacked per projection type, find d_int at 90% energy threshold.

No MLX needed — pure numpy SVD on already-computed matrices.
Expected runtime: < 10 seconds.
"""

import json
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
V2_DIR = Path("experiments/models/m2p_qwen06b_gsm8k_v2")
OUT_DIR = Path("experiments/models/intrinsic_dim_real_tasks")
ENERGY_THRESHOLD = 0.90
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load B-matrices
# ---------------------------------------------------------------------------
print("Loading SFT B-matrices from v2 ...")
d = np.load(V2_DIR / "sft_b_matrices.npz")
keys = sorted(d.keys())  # e.g. layer_0_q_proj_B, layer_0_v_proj_B, ...

q_keys = sorted([k for k in keys if "q_proj" in k])
v_keys = sorted([k for k in keys if "v_proj" in k])

B_q = np.vstack([d[k] for k in q_keys])  # (L*r, d_q) = (112, 2048)
B_v = np.vstack([d[k] for k in v_keys])  # (L*r, d_v) = (112, 1024)

L = len(q_keys)
r = d[q_keys[0]].shape[0]
d_q = d[q_keys[0]].shape[1]
d_v = d[v_keys[0]].shape[1]

print(f"  L={L} layers, r={r} (LoRA rank), d_q={d_q}, d_v={d_v}")
print(f"  M_q shape: {B_q.shape}")
print(f"  M_v shape: {B_v.shape}")


# ---------------------------------------------------------------------------
# SVD + intrinsic dimensionality
# ---------------------------------------------------------------------------
def measure_d_int(M: np.ndarray, name: str, tau: float = ENERGY_THRESHOLD):
    """SVD M, return d_int at tau energy threshold."""
    t0 = time.time()
    # Use full_matrices=False for efficiency (min(m,n) singular values)
    _, sigma, _ = np.linalg.svd(M, full_matrices=False)
    elapsed = time.time() - t0

    energy = sigma ** 2
    total_energy = energy.sum()
    cumulative = np.cumsum(energy) / total_energy

    # d_int = first k where cumulative energy >= tau
    d_int = int(np.searchsorted(cumulative, tau)) + 1  # 1-indexed

    # Energy fractions
    sigma1_fraction = float(energy[0] / total_energy)
    top5_fraction = float(energy[:5].sum() / total_energy)
    top10_fraction = float(energy[:10].sum() / total_energy)
    top20_fraction = float(energy[:20].sum() / total_energy)
    top64_fraction = float(energy[:64].sum() / total_energy)

    max_rank = min(M.shape)

    print(f"\n  [{name}]  shape={M.shape}  max_rank={max_rank}  SVD in {elapsed:.2f}s")
    print(f"    σ_1² fraction: {sigma1_fraction:.3f}  ({sigma1_fraction*100:.1f}% of energy)")
    print(f"    top-5 energy:  {top5_fraction:.3f}  ({top5_fraction*100:.1f}%)")
    print(f"    top-10 energy: {top10_fraction:.3f}  ({top10_fraction*100:.1f}%)")
    print(f"    top-20 energy: {top20_fraction:.3f}  ({top20_fraction*100:.1f}%)")
    print(f"    top-64 energy: {top64_fraction:.3f}  ({top64_fraction*100:.1f}%)")
    print(f"    d_int (τ={tau}): {d_int}  {'< 64 ✓' if d_int < 64 else '≥ 64 ⚠'}")

    # Print cumulative energy profile
    milestones = [1, 2, 5, 10, 20, 30, 50, 64, 100]
    print(f"    Energy profile:")
    for k in milestones:
        if k <= max_rank:
            frac = float(energy[:k].sum() / total_energy)
            print(f"      k={k:3d}: {frac:.3f} ({frac*100:.1f}%)")

    return {
        "name": name,
        "shape": list(M.shape),
        "max_rank": max_rank,
        "d_int_90pct": d_int,
        "sigma1_energy_fraction": sigma1_fraction,
        "top5_energy_fraction": top5_fraction,
        "top10_energy_fraction": top10_fraction,
        "top20_energy_fraction": top20_fraction,
        "top64_energy_fraction": top64_fraction,
        "singular_values": sigma.tolist(),
        "cumulative_energy_fractions": cumulative.tolist(),
    }


print("\n--- SVD Analysis ---")
q_result = measure_d_int(B_q, "q_proj")
v_result = measure_d_int(B_v, "v_proj")


# ---------------------------------------------------------------------------
# K945 evaluation
# ---------------------------------------------------------------------------
d_int_q = q_result["d_int_90pct"]
d_int_v = v_result["d_int_90pct"]
d_int_max = max(d_int_q, d_int_v)

print(f"\n--- Kill Criteria ---")
print(f"K945 (d_int measured): d_int_q={d_int_q}, d_int_v={d_int_v}, max={d_int_max}")
k945 = "pass"  # always passes — it's a measurement
print(f"K945: {k945.upper()}")

# Interpretation
if d_int_max < 64:
    interpretation = f"d_int={d_int_max} < 64 → M2P bottleneck (d_M2P=64) is SUFFICIENT. v2 failure was gradient bug only."
else:
    interpretation = f"d_int={d_int_max} >= 64 → M2P bottleneck needs expansion. SHINE regime (d_M2P=d_model) required."

print(f"\nInterpretation: {interpretation}")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
results = {
    "experiment": "exp_intrinsic_dim_real_tasks",
    "source": str(V2_DIR),
    "energy_threshold": ENERGY_THRESHOLD,
    "L": L,
    "r": r,
    "d_q": d_q,
    "d_v": d_v,
    "q_proj": q_result,
    "v_proj": v_result,
    "d_int_q": d_int_q,
    "d_int_v": d_int_v,
    "d_int_max": d_int_max,
    "m2p_bottleneck_d": 64,
    "bottleneck_sufficient": d_int_max < 64,
    "interpretation": interpretation,
    "kill_criteria": {
        "K945": {"status": k945, "value": f"d_int_q={d_int_q}, d_int_v={d_int_v}"},
    },
}

# Remove large arrays from JSON
for proj_key in ["q_proj", "v_proj"]:
    results[proj_key]["singular_values"] = results[proj_key]["singular_values"][:20]
    results[proj_key]["cumulative_energy_fractions"] = results[proj_key]["cumulative_energy_fractions"][:20]

out_path = OUT_DIR / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {out_path}")
print(f"\nSummary:")
print(f"  d_int (q_proj, 90% energy): {d_int_q}")
print(f"  d_int (v_proj, 90% energy): {d_int_v}")
print(f"  M2P bottleneck d_M2P=64: {'SUFFICIENT' if d_int_max < 64 else 'INSUFFICIENT'}")
