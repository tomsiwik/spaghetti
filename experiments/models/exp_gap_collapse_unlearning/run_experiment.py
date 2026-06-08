
"""Micro mechanism check for: gap-collapse destroys expert separability ("merging = forgetting").
Self-contained numpy simulation (no model load, no network). TARGET behavioral metric: can a probe
recover the A-vs-B context label from the COMPOSED output as the interference gap g = f_A - f_B is
driven to 0 (alpha 1->0, routed->merged)? Prediction: separability -> chance (0.5). is_smoke micro."""
import json, numpy as np
from pathlib import Path
np.seterr(all="ignore")
EXP_DIR = Path(__file__).resolve().parent
rng = np.random.RandomState(0)
d, r, n = 256, 8, 400
A = (rng.randn(d, r) @ rng.randn(r, d)) / np.sqrt(d)   # two low-rank experts
B = (rng.randn(d, r) @ rng.randn(r, d)) / np.sqrt(d)
X = rng.randn(n, d)
fA, fB = X @ A, X @ B
lab = np.arange(n) % 2                                  # 0 = A-context, 1 = B-context
target = np.where(lab[:, None] == 0, fA, fB)           # correct behavior = route to right expert
direction = fA - fB                                     # the gap direction that separates A from B
def separability(alpha):
    out = alpha * target + (1 - alpha) * 0.5 * (fA + fB)  # alpha=1 routed, alpha=0 merged/collapsed
    score = (out * direction).sum(1)
    pred = (score < np.median(score)).astype(int)
    return float(max((pred == lab).mean(), (pred != lab).mean()))
full = separability(1.0)         # full gap kept (routed)
collapsed = separability(0.0)    # gap collapsed (merged)  -> should be ~chance
drop = full - collapsed
# KC: collapsing the gap must reduce separability toward chance (>= 0.15 absolute drop, collapsed <= 0.6)
k1 = bool(drop >= 0.15 and collapsed <= 0.60)
res = {
  "is_smoke": True, "scale": "micro",
  "separability_full_gap": round(full,3),
  "separability_collapsed": round(collapsed,3),
  "separability_drop": round(drop,3),
  "verdict": "SUPPORTED" if k1 else "KILLED",
  "all_pass": k1,
  "kc": {"K1_gap_collapse_reduces_separability": "pass" if k1 else "fail"},
  "note": "Micro numpy mechanism sim of gap-collapse->forgetting. Not the full frozen-Gemma-4 run; an honest small-scale check of the predicted direction.",
}
(EXP_DIR / "results.json").write_text(json.dumps(res, indent=2))
print("RESULT", json.dumps(res))
