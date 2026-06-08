
"""Micro mechanism check (numpy, <5s): does collapsing the interference gap g=f_A-f_B destroy A-vs-B
separability of the COMPOSED output? alpha 1->0 = routed->merged. Predict: separability -> chance.
Writes results.json in CWD. is_smoke micro."""
import json, numpy as np
np.seterr(all="ignore")
rng = np.random.RandomState(0)
d, r, n = 256, 8, 400
A = (rng.randn(d, r) @ rng.randn(r, d)) / np.sqrt(d)
B = (rng.randn(d, r) @ rng.randn(r, d)) / np.sqrt(d)
X = rng.randn(n, d); fA, fB = X @ A, X @ B
lab = np.arange(n) % 2
target = np.where(lab[:, None] == 0, fA, fB); direction = fA - fB
def sep(a):
    out = a * target + (1 - a) * 0.5 * (fA + fB)
    s = (out * direction).sum(1); p = (s < np.median(s)).astype(int)
    return float(max((p == lab).mean(), (p != lab).mean()))
full, collapsed = sep(1.0), sep(0.0); drop = full - collapsed
k1 = bool(drop >= 0.15 and collapsed <= 0.60)
json.dump({"is_smoke": True, "scale": "micro", "separability_full_gap": round(full, 3),
           "separability_collapsed": round(collapsed, 3), "separability_drop": round(drop, 3),
           "verdict": "SUPPORTED" if k1 else "KILLED", "all_pass": k1,
           "kc": {"K1_gap_collapse_reduces_separability": "pass" if k1 else "fail"},
           "note": "Micro numpy mechanism sim; not the full frozen-Gemma-4 run."},
          open("results.json", "w"), indent=2)
print("RESULT done")
