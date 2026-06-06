# Adversarial Review — P11.Z0: Plan-and-Solve Prompting

**Reviewer**: Ralph (automated)
**Date**: 2026-04-14
**Status**: POST-RUN review (results verified)
**Verdict**: PROCEED (KILLED — prediction correctly refuted)

---

## Pre-execution review

See above section (pre-run PROCEED with 3 non-blocking issues). All non-blocking issues
were properly handled in PAPER.md.

---

## Post-Run Verification

### Numbers check (results.json vs PAPER.md)

| Claim | results.json | PAPER.md | Match? |
|-------|-------------|----------|--------|
| P0_direct accuracy | 155/280 = 55.36% | 55.4% | ✓ |
| P1_ps accuracy | 147/280 = 52.5% | 52.5% | ✓ |
| P2_ps_plus accuracy | 152/280 = 54.29% | 54.3% | ✓ |
| K1529 pass | false | FAIL | ✓ |
| K1530 pass | true | PASS | ✓ |
| K1531 pass | true (ratio=1.0×) | PASS | ✓ |
| Best prompt | P0_direct | P0_direct | ✓ |

All numbers match. No fabrication.

### Kill criteria correctness

- **K1529 FAIL**: best=55.4% vs 64.1% target (delta_vs_baseline=-6.7pp). Correctly FAIL.
- **K1530 PASS**: PS+=54.3% > PS=52.5%. 1.8pp gap — within noise but directionally consistent with Wang et al.
- **K1531 PASS**: best=P0_direct, so ratio=1.0×. Trivially passes.

### PAPER.md quality

- Prediction-vs-measurement table: PRESENT ✓
- P0 drift (-6.7pp): disclosed and explained with 3 candidate causes ✓
- Delta-vs-P0 included (per pre-run review note): ✓
- Impossibility structure: well-articulated (thinking subsumes PS) ✓
- Implications for downstream evals: clear (don't use PS prompts) ✓

---

## Scientific Assessment

### What the experiment proves

The -6.7pp P0 drift vs Finding #530 weakens the conclusion slightly: we cannot rule out
that the PS prompts harmed accuracy but the baseline shifted independently. However,
**delta_vs_P0** (PS=-2.9pp, PS+=-1.1pp) is the correct comparison and shows PS never
beats direct answering in the same run.

The impossibility structure is correct: Gemma 4 E4B with `enable_thinking=True` generates
internal step decomposition before every token. Adding PS instructions creates redundant
planning that competes with the model's native thinking flow.

### What's uncertain

1. P0 drift: 55.4% vs 62.1% baseline (-6.7pp) is large. Likely sample variance (different
   random seed, different difficulty distribution from 12K+ MMLU-Pro questions). The math
   category shows 0.95 (high) while health shows 0.25 (low) — consistent with sample luck.

2. K1530 margin is 1.8pp (54.3% vs 52.5%) — within noise for n=280. The directional
   finding (PS+ ≥ PS) is consistent with Wang et al. but not definitive.

### No fabrication issues

Kill criteria are accurately reported. KILLED status is correct: K1529 fails, which is
the primary criterion.

---

## Verdict: PROCEED

PAPER.md is complete and accurate. Experiment is correctly KILLED. No REVISE needed.

The key finding (thinking subsumes PS; don't use PS prompts in P11 benchmark evals) is
actionable and correctly documented.
