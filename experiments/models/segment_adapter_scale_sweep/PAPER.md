# Segment Adapter Scale Sweep: Proof Verification Report

## Theorem

**Claim (PBR Scaling):** For LoRA perturbation h_adapted = h_base + s * B * A * x
with fixed adapter weights (A, B), the perturbation-to-base ratio (PBR) scales
linearly with s. On distribution-shifted inputs (128-token segments vs 256-token
training sequences), there exists an optimal scale s* < s_train that minimizes
PPL by matching the perturbation magnitude to the segment context.

## Predictions vs Measurements

| Prediction (from MATH.md) | Measured | Match? |
|---------------------------|----------|--------|
| P1: Exists s* < 20 with PPL < base | s*=2.0, PPL 7.988 < base 7.993 (-0.06%) | WEAK YES |
| P2: PPL curve is U-shaped | Monotonically increasing from s=2 to s=20 | NO (monotonic, not U-shaped) |
| P3: s* in [5, 15] | s*=2.0 (smallest tested) | NO (below predicted range) |
| P4: Behavioral at s* within 10% of per-seq | ratio=0.998 (0.2% gap) | YES |

## Hypothesis

Optimal LORA_SCALE for 128-token segment-isolated application is significantly
less than training scale s=20, and at optimal scale, segment-isolated PPL will
be below base PPL.

**Verdict: KILLED.** The hypothesis that there exists an optimal s* making
segment-isolated adapters effective is refuted. The PPL curve is monotonically
increasing from s=2 to s=20, with the "best" scale (s=2) providing only -0.06%
improvement over base -- approximately 5 nats over 6350 tokens, statistically
indistinguishable from noise (no CI, no bootstrap, no multi-seed). The
prediction of a U-shaped curve was wrong. The self-test falsification condition
(monotonically increasing PPL) was met. Adapters provide near-zero useful signal
on isolated 128-token segments at any scale. The failure is structural (context
dependency), not a scale problem.

## What This Experiment Is

A scale sweep measuring how LORA_SCALE affects perplexity when domain-specific
LoRA adapters (trained at s=20 on full sequences) are applied to 128-token
isolated segments. Five scales {2, 5, 10, 15, 20} were tested across 5 domains
(medical, code, math, legal, finance) with 10 samples/domain for PPL and 5
samples/domain for behavioral evaluation.

## Key References

- Hu et al. (2022), "LoRA: Low-Rank Adaptation" (arXiv 2106.09685) -- perturbation
  scaling analysis
- Finding #310 -- segment-isolated PPL degradation at s=20
- Finding #305 -- segment isolation architecture
- Finding #308 -- B-matrix norm variation as scale confound

## Empirical Results

### Scale-PPL Curve (Segment-Isolated, 128 tokens)

| Scale | Segment PPL | vs Base | vs Per-Seq |
|-------|-------------|---------|------------|
| base  | 7.993       | ---     | -23.7%     |
| 2.0   | 7.988       | -0.06%  | -23.8%     |
| 5.0   | 8.007       | +0.17%  | -23.6%     |
| 10.0  | 8.051       | +0.73%  | -23.2%     |
| 15.0  | 8.084       | +1.14%  | -22.9%     |
| 20.0  | 8.130       | +1.72%  | -22.4%     |
| per-seq (s=20) | 10.480 | +31.1% | ---    |

The monotonically increasing curve from s=2 to s=20 shows that ANY adapter
perturbation at ANY scale is harmful or neutral on isolated 128-token segments.
The "best" scale (s=2.0) provides only -0.06% improvement over base -- within
noise. The per-sequence PPL being WORSE than base (10.48 vs 7.99) indicates
that even on full sequences, these adapters are not improving PPL on this
particular validation set.

### Per-Domain PPL at s=2.0 (Best Scale)

| Domain | Segment (s=2) | Base | Per-Seq (s=20) | seg vs base |
|--------|--------------|------|----------------|-------------|
| medical | 5.342 | 5.334 | 6.615 | +0.16% |
| code | 4.100 | 4.107 | 6.147 | -0.16% |
| math | 4.895 | 4.887 | 4.283 | +0.16% |
| legal | 19.068 | 19.098 | 21.612 | -0.16% |
| finance | 15.910 | 15.960 | 16.176 | -0.31% |

All per-domain deltas are within +/-0.3% of base. The adapters provide
essentially zero domain-specific signal on isolated segments.

### Behavioral Scores (Factual Recall)

| Config | medical | code | math | legal | finance | MEAN |
|--------|---------|------|------|-------|---------|------|
| base_only | 0.391 | 0.245 | 0.711 | 0.123 | 0.121 | 0.318 |
| best_scale (s=2) | 0.394 | 0.245 | 0.711 | 0.109 | 0.127 | 0.317 |
| training_scale (s=20) | 0.391 | 0.245 | 0.711 | 0.120 | 0.122 | 0.318 |

Behavioral scores are identical across all configurations. The adapters have
no measurable behavioral effect on short prompts at any scale.

## Kill Criteria Assessment

| Criterion | Result | Detail |
|-----------|--------|--------|
| K787: Best PPL < base | FAIL | 7.988 vs 7.993 (-0.06%) is ~5 nats over 6350 tokens. No CI, no bootstrap, no multi-seed. This is noise, not evidence of improvement. |
| K788: Non-monotonic curve (U-shape) | FAIL | Curve is monotonically increasing (7.988, 8.007, 8.051, 8.084, 8.130). The U-shape prediction -- the actual prediction being tested -- is refuted. s*=2 != s_train=20 satisfies only the weaker sub-condition, not the predicted non-monotonicity. |
| K789: Behavioral within 10% | PASS (trivial) | ratio=0.998, but this passes because the adapter has zero behavioral effect at any scale, not because s* preserves quality. |

## Limitations

1. **PPL improvement is vanishingly small (-0.06%)**: This is within measurement
   noise. A practitioner cannot distinguish s=2 from s=0 (no adapter) on segments.

2. **Per-sequence PPL is also WORSE than base**: At s=20, the correct-domain
   adapter yields PPL 10.48 vs base 7.99 on full validation sequences. This
   contradicts Finding #310 (per-seq best 7.366 < base 7.465), but that finding
   used mixed-domain concatenated sequences and tried ALL adapters to find the
   best one. Using the correct single adapter on pure-domain text, the adapter
   hurts on most domains.

3. **Behavioral metrics show no adapter effect at all**: Factual recall is
   identical with or without adapters, at any scale. This is consistent with
   the PPL findings -- the adapters are not providing useful domain signal.

4. **Only 10 samples/domain for PPL, 5 for behavioral**: Low statistical power,
   but the signal (or lack thereof) is consistent across all domains and scales.

## Interpretation

The scale confound hypothesis was partially correct: lower scale IS better for
segments (monotonic improvement from s=20 to s=2). But the deeper finding is that
**adapters provide essentially zero domain-specific signal on isolated 128-token
segments** -- the improvement from s=2 over base is 0.06%, indistinguishable
from noise.

This suggests the failure mode is not about scale magnitude but about
**context dependency**: the adapters were trained on full sequences where they
could leverage long-range attention patterns and full instruction-response
structure. On 128-token segments (often mid-response text without the instruction
prompt), there is no structure for the adapter to hook into.

The implication for SOLE: segment-isolated routing solves domain classification
(98.3% token accuracy, Finding #310) but does NOT solve domain adaptation.
The routing mechanism is correct but the adapters need context-aware application,
not isolated segment application.

## What Would Kill This

Already partially killed: the U-shaped prediction was wrong. The adapters are
not useful on isolated segments at ANY scale. The remaining question is whether
the adapters are useful on full sequences with proper instruction framing -- the
per-sequence PPL numbers here (10.48 > base 7.99) suggest they may not be, which
would be a larger finding about these specific adapters' utility.

## Runtime

- Phase 1 (PPL sweep): 26.5s
- Phase 2 (Behavioral): 246.3s
- Total: 273.0s (4.5 minutes)
