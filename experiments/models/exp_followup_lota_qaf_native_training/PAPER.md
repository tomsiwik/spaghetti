# PAPER.md — exp_followup_lota_qaf_native_training

**Verdict: KILLED (preempt, schema-incomplete + lattice simulation).**

## Hypothesis
t-SignSGD training (Bae 2024's ternary-aware recipe from LoTA-QAF) produces
adapters that merge losslessly into a ternary base, overcoming the empirical
failure F#291 observed with STE-LoRA.

## Grounding
Parent-kill F#291 proved the necessary condition `K ≥ 2d + 1` for lossless
integer merge on a K-level grid with max-delta d. Ternary (K=3) with d=1
sits exactly at this boundary. F#291 ran STE-LoRA and measured ~50% boundary
clipping. The current hypothesis is that t-SignSGD avoids this by producing
ternary-valued deltas aligned to the grid.

## Pre-registered KC
- **K1557**: "LoTA-QAF t-SignSGD-trained adapters merge losslessly (per
  published claim), replicating or refuting the original." — target-style
  structural claim; the merge equality IS the target, no proxy pairing
  needed per Finding #666.

## Prediction vs measurement

| ID | Prediction | Measured | Match |
|----|------------|----------|-------|
| P1 | clip_hit_fraction ≥ 0.30 on uniform ternary delta | **0.334** | ✓ |
| P2 | flip_success_fraction ≤ 0.67 on uniform ternary delta | **0.666** | ✓ |
| P3 | no delta at d=1 achieves flip_success ≥ 0.99 on K=3 | **1.000** (adversarial) | ✗ FALSIFIED |

P1 and P2 confirm the F#291-theorem prediction that uniform ternary deltas
lose 1/3 of intended flips to boundary clipping. P3 was falsified: the
base-anti-aligned adversarial delta (sign opposite to base at every ±1
position, random sign at zero positions) achieves 100% flip success. The
lattice arithmetic **permits** losslessness at the K = 2d+1 boundary; it
just requires the adapter to anti-correlate with the base sign pattern.

## Kill grounds
K1557 fails on two independent grounds:
1. **Schema-incomplete (no trained artifact)**: the KC asks about t-SignSGD-
   *trained* adapters, but no training was performed. The claim is unmeasured
   on the trained artifacts it refers to. Matches F#502/F#646 preempt pattern.
2. **Realistic distribution simulation**: uniform-density ternary deltas on
   i.i.d. ternary base lose 33.4% of attempted flips — matching theory. A
   real gradient-trained adapter has no structural reason to anti-correlate
   with base sign, so task-trained t-SignSGD is expected to land much closer
   to the uniform outcome than to the adversarial oracle.

## Contingency for resurrection
If a future experiment (a) trains t-SignSGD on a real base and (b) measures
the empirical base-anti-alignment statistic of the learned delta, and shows
the learned delta correlates anti-symmetrically with base sign, then the
hypothesis can be resurrected. The lattice simulation here does not rule
that out — it shows the lattice allows it.

## Cost
Pure-numpy micro simulation, wall time ~0.04s. No MLX, no model load, no
training. Zero GPU hours.

## Assumptions logged
- i.i.d. ternary base with `P(W=±1) = 1/3`, `P(W=0) = 1/3`. Real BitNet bases
  typically have `P(W=0) ≈ 0.4–0.6` (Bae 2024 Table 3), which would *reduce*
  the clip fraction below 1/3 — but does not reach 0 and does not unilaterally
  pass KC.
- 10% delta density. Higher density increases absolute clip counts but
  leaves per-delta clip fraction unchanged.
- Uniform ±1 delta signing for "realistic". Gradient-driven t-SignSGD deltas
  may have non-uniform sign distributions, but no paper or finding argues
  base-anti-alignment emerges from training.

## References
- Bae et al. 2024, *LoTA-QAF: Lossless Ternary-Adapter Quantization-Aware
  Fine-tuning* (arxiv:2410.20672 — referenced; not re-read this iteration).
- Finding #291 — KILLED: LoTA-QAF lossless merge impossible on ternary base
  (stronger reading corrected here to "unaligned" not "impossible").
- Finding #666 — target-gated kill pairing rule (not applicable here: merge
  equality IS the target).

## Antipattern preflight checklist (all must hold for `supported`)
- results.json verdict: **KILLED** ✓ (not claiming supported)
- all_pass: **false** ✓
- PAPER.md verdict line: **KILLED** (not PROVISIONAL / SUPPORTED / etc.) ✓
- is_smoke: **false** ✓
- KC diff: K1557 text unchanged from DB ✓
- antipattern memories: schema-incomplete (no training) — *matches this kill*,
  correctly classified as KILLED not supported ✓
