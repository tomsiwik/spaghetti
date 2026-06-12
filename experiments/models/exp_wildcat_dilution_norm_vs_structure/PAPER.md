# PAPER — Interference dilution: norm reduction vs structure

**Experiment:** `exp_wildcat_dilution_norm_vs_structure`
**Run:** real (is_smoke: false), gemma-4-e4b-it-4bit, GSM8K n=100, greedy, max 1024 tokens, no-thinking harness, mlx_lm 0.31.2.

## Question
F#881 showed a random mask at keep-fraction f = 0.4335 on the dense thinking delta (v/o_proj, scale 1.0),
composed with the math q_proj adapter (scale 6.0), recovers +22pp of the F#862 +30pp gain. Is that purely
norm reduction (mask collapses to a scalar alpha knob) or destruction of a structured collision?

## Pre-registered prediction (MATH.md)
- **Predicted:** EM(C, dense α=√f=0.6584) ≥ mean EM(B, random mask ×3 seeds) − 3pp ⇒ pure norm reduction,
  mask arc killed. Predicted EM_B ≈ 0.66, EM_C within [EM_B − 0.03, EM_B + 0.05].
- **Refutation:** mean EM(B) − EM(C) ≥ 5pp ⇒ structural, mask arc lives.
- **Inconclusive band:** gap in (3pp, 5pp) ⇒ provisional.
- **Noise flag:** 3-seed B spread > 8pp ⇒ F#881 is noise.

## Measured
| Arm | Description | EM (n=100) |
|---|---|---|
| A | dense delta, f=1 | 0.60 |
| B s0/s1/s2 | random mask f=0.4335 | 0.82 / 0.83 / 0.87 (**mean 0.84**) |
| C | dense α=√f=0.6584 (norm-matched) | **0.79** |
| C2 | dense α=f=0.4335 (mean-field, exploratory) | 0.84 |
| D | keep-largest-\|dW\|, f=0.4335 | 0.68 |
| E | keep-smallest-\|dW\|, f=0.4335 | 0.92 |

- gap_b_minus_c = mean EM(B) − EM(C) = **0.0500 (5.0pp, float 0.04999…)** — lands exactly on the
  refutation boundary / top of the inconclusive band (3pp, 5pp).
- B 3-seed spread = 5.0pp ≤ 8pp ⇒ noise flag NOT fired; F#881 dilution effect replicates
  (mean +24pp over dense A=0.60).
- Kill #2335: pure_norm_fired = false, noise_flag_fired = false, result = pass.
- Frobenius check: ‖Δ‖_F dense A = 3.161; B = 2.081, C = 2.081 — the B-vs-C primary contrast is
  norm-matched as designed. The exploratory arms are NOT norm-matched: C2 = 1.370, D = 3.080,
  E = 0.431. Only B vs C isolates structure from norm.

## Prediction vs measurement
The pre-registered prediction (pure norm reduction, gap ≤ 3pp) is **not confirmed**: the norm-matched
dense control C (0.79) underperforms the random mask B (0.84) by 5.0pp, sitting on the boundary
between "inconclusive" and "structural". The runner classified it inside the inconclusive band
(structure_class: "n/a (not structural)") and emitted `verdict: provisional`, `all_pass: false`.

Secondary signals are suggestive but **norm-confounded** and cannot support a structural claim:
- E (outliers removed) = 0.92 ≥ B = 0.84 > D (outliers kept) = 0.68 matches the pre-registered
  outlier-driven ordering, but D carries ‖Δ‖_F = 3.080 (~97% of dense norm) and E only 0.431
  (~14%), so pure norm reduction predicts exactly the same ordering. Across all arms EM is
  monotone in ‖Δ‖_F (3.16→0.60, 3.08→0.68, 2.08→0.79–0.87, 1.37→0.84, 0.43→0.92); as run,
  the D/E probes are a norm proxy, not a structure probe.
- C2 (α=f) = 0.84 matches B and beats C = 0.79, but C2 is also not norm-matched (1.370 vs 2.081),
  so it fits the same monotone-in-norm pattern.
- The only norm-controlled evidence in this run is the 5.0pp B-vs-C gap.

## Verdict
**PROVISIONAL.** Primary gap B−C = 5.0pp falls exactly at the refutation threshold (numerically
0.04999… < 0.05, i.e. inside the pre-registered inconclusive band). The "pure norm reduction" kill
did NOT fire — the mask arc is not reducible to a scalar alpha on this evidence — but the structural
claim is not cleanly confirmed by the primary criterion either. The D/E probes are norm-confounded
(D=3.080, E=0.431 vs C=2.081) and consistent with pure norm reduction, so they provide no
independent evidence for an outlier-carried mechanism. Required follow-up before any supported
claim: (a) n ≥ 300 (or pre-registered paired significance) on B vs C to resolve the 5pp gap;
(b) norm-matched D'/E' rescaled to ‖C‖_F = 2.081 so the structure probe is not a norm proxy.
