# PAPER: Zero-shot precision transfer 4→8 bit (Gemma 4 E4B)

## Verdict: **KILLED** — adapter benefit does not transfer within 5% tolerance from 4-bit to 8-bit base.

## Hypothesis

From F#97 (micro scale), adapter deltas in a low-rank subspace are approximately preserved under bounded weight perturbation. The 4→8-bit dequantization perturbation is smaller than the SVD perturbation of F#97. Prediction: R ≥ 0.95 per domain.

## Predictions vs Measurements

| Quantity | Predicted | Measured | Pass? |
|----------|-----------|----------|-------|
| code R | ≥ 0.95 | 0.8999 | FAIL |
| math R | ≥ 0.95 | 0.9459 | FAIL |
| medical R | ≥ 0.95 | 0.9842 | PASS |
| median R (3 domains) | ≥ 0.95 | 0.9459 | FAIL |
| min R | ≥ 0.85 | 0.8999 | PASS |
| adapter helps on 4-bit | all domains | all domains | PASS |
| finite PPL ≥ 95% samples | yes | 99–100% | PASS |

## Raw numbers

| Domain | 4-bit base | 4-bit adapter | gain₄ | 8-bit base | 8-bit adapter | gain₈ | R |
|--------|-----------|---------------|-------|-----------|---------------|-------|------|
| code | 3.8184 | 1.4386 | 0.6233 | 3.2437 | 1.4244 | 0.5609 | 0.8999 |
| math | 5.9038 | 1.4413 | 0.7559 | 5.2604 | 1.4995 | 0.7149 | 0.9459 |
| medical | 10.4202 | 1.0000 | 0.9040 | 9.0735 | 1.0000 | 0.8898 | 0.9842 |

## Why the prediction failed

The prediction assumed the dequantization perturbation between 4-bit and 8-bit is *smaller* than the SVD perturbation in F#97, so transfer should be at least as good. This was wrong for two reasons:

1. **The perturbation is structured, not random.** F#97 used SVD perturbation (removing a rank component), which is a controlled, spectrally smooth change. The 4→8-bit dequantization changes the weight realization in a way that is *correlated across dimensions* (block quantization boundaries). This structured shift moves the effective weight manifold more than a spectral truncation of equivalent Frobenius norm.

2. **Gain ratio is sensitive to base PPL shifts.** The 8-bit base has lower PPL than 4-bit (better base → less room for adapter improvement). Code: 3.82→3.24 (15% base PPL drop), math: 5.90→5.26 (11% drop). When the base improves, the adapter's *relative* contribution shrinks, reducing gain₈ relative to gain₄. This is a ceiling effect: the adapter trained to fix 4-bit degradation partially overlaps with what 8-bit already fixes.

## What this finding means

- **4→8 transfer loses 5–10% of adapter benefit**, with code domain worst (10% loss).
- **Medical is near-perfect** (R=0.98) because the adapter PPL is already 1.0 (near-saturated) — the adapter dominates regardless of base precision.
- **The F#97 micro-scale result does not directly extend** to real LLM precision changes. The micro experiment's SVD perturbation was a qualitatively different kind of weight shift than block quantization.
- **Not catastrophic**: all domains pass the 0.85 floor. The adapter still provides >89% of its benefit. But the strict "lossless transfer" claim (≤5% loss) is falsified.

## Scope note

This tests 4→8 transfer only. The title's full claim (4→bf16) would require an even larger precision jump and likely worse transfer ratios. The 4→8 result upper-bounds the 4→bf16 transfer: if 4→8 loses 5–10%, 4→bf16 likely loses more.

## K1/K2

- K1 (structural): **PASS** — all sweeps finite, adapter helps on 4-bit.
- K2 (transfer ratio): **FAIL** — median R=0.9459 < 0.95 threshold. Per-domain floor (0.85) met.
- Verdict: **KILLED** per pre-registered KC.
