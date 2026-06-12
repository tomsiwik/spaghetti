# REVIEW-adversarial — exp_spark_strobe_multiplex

Route: **REVISE** (round 1 of max 2)

## Mock / integrity checks — ALL PASS
- verify-experiment.sh exits 0; is_smoke:false; verdict present; model-backed.
- 3 real, DISTINCT adapters (md5s all differ), real safetensors, B-norms asserted >1e-4.
- Real datasets: gsm8k, HumanEval (executed via subprocess), MedQA-USMLE.
- Scorer path is identical across STATIC/STROBE → no scoring asymmetry; the python
  5.88→82.35 swing is a real generation-quality difference, not a parse artifact.
- Clock is genuinely content-blind (phase = step index mod 3, advanced post-forward).
- results.json / PAPER.md verdicts agree; threshold +4pp pre-registered, untracked dir,
  no goalpost move. Not a mock, not a tautology.

## Blocking flaw — uncontrolled magnitude confound (NOT a clean test of the claim)
MATH.md claim: interference is a *simultaneity* artifact, **not a weighting** one, and
labels STATIC the "1/N composition." But code (run_experiment.py:97-104):
  STATIC = base + SCALE * Σ_{i=1..3}(xA_i)B_i     # raw sum, NO 1/N
  STROBE = base + SCALE * (xA_k)B_k                # one delta
STATIC injects a residual ~3× the magnitude STROBE injects. The two conditions differ
in BOTH simultaneity AND total residual magnitude. The python collapse (5.88pp) is the
signature of an over-driven adapter stack (effective scale ~18), which STROBE avoids by
running each adapter at its trained scale (6). The design cannot separate "simultaneity"
from "magnitude," so PAPER's causal claim ("simultaneous deltas, not weighting, cause
the interference") is unsupported. Also: code contradicts its own "1/N composition" label.

## Required fixes (≤3, bounded)
1. Add a magnitude-matched control: STATIC_NORM = base + SCALE * (1/N) Σ(xA_i)B_i
   (true 1/N, matching the doc's own label). Report STROBE vs STATIC_NORM.
2. If STROBE still beats STATIC_NORM by ≥ +4pp, the simultaneity claim survives → SUPPORTED.
   If the win collapses against STATIC_NORM, it was a magnitude effect → KILLED.
3. Fix the STATIC label in MATH.md/PAPER.md (raw Σ is not "1/N") or implement 1/N.

The aggregate +19.6pp vs raw-sum STATIC is real but interpretively confounded; the
pre-registered comparison must be magnitude-matched before the verdict can be sealed.
