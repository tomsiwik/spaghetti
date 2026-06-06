# PAPER — exp_followup_jointly_trained_routing_molora

**Status:** KILLED (preemptive, structural)
**Scale:** micro N=5
**Date:** 2026-04-19

## Abstract
Evaluated K1551 — "jointly-trained (MoLoRA-style) router beats post-hoc
TF-IDF routing by ≥3pp on held-out per-token task at N=5" — via
structural proof against four prior findings. **KILLED preemptively**:
TF-IDF ceiling + per-token full-sequence null + MLP-only +3.3% ceiling
+ ridge-router analog failure jointly bound the plausible Δ below 3pp.
No training or eval run.

## Prediction vs Measurement

| Quantity                           | Predicted (structural)                  | Measured     |
|------------------------------------|-----------------------------------------|--------------|
| A_TF-IDF (N=5 held-out per-token)  | 96.6% (F#431 reuse)                     | N/A (preempt)|
| A_MoLoRA required for K1551        | ≥99.6% (96.6 + 3.0)                     | N/A (preempt)|
| A_MoLoRA on full-sequence mixed    | = A_TF-IDF (F#305 null, Δ=0)            | N/A (preempt)|
| A_MoLoRA on MLP-only homogeneous   | ≤ A_TF-IDF + 3.3% (F#312 ceiling)       | N/A (preempt)|
| A_MoLoRA on ridge-router-analog    | catastrophic drop (F#340 8.6pp)         | N/A (preempt)|

## Method
**No code executed.** Preempt proof constructed from:
1. F#431: TF-IDF at 96.6% accuracy at N=5 on Gemma 4.
2. F#305: per-token full-sequence routing is null vs per-sequence
   (PPL 4.815 bit-exact).
3. F#312: MLP-only contamination-free per-token ceiling +3.3% PPL;
   segment isolation dominates at +16.1%.
4. F#193: routing architecture irrelevant when representation is the
   bottleneck.
5. F#340: ridge router + single-pass E2E on mixed-domain KILLED
   (8.6pp accuracy drop).

## Results

### K1551 FAIL (preempt)
`A_MoLoRA − A_TF-IDF ≥ 3pp` is incompatible with:
- TF-IDF ceiling (max 3.4pp headroom, F#431)
- Per-token full-sequence null (Δ=0 on mixed, F#305)
- MLP-only +3.3% cap (F#312)
- Architecture irrelevance at near-oracle (F#193)

Ridge-router + E2E (closest analog) already KILLED on mixed-domain
(F#340). No mechanism in the MoLoRA design changes this.

## Conclusion
**KILLED — preemptive, structural impossibility.** The ≥3pp threshold
sits at or above the mathematical ceiling set by the already-measured
TF-IDF baseline (96.6%) combined with the null-per-token and
MLP-only-cap findings. The experiment would consume ~2h MLX training
for a structurally-determined null result.

## Antipatterns flagged
1. `near-oracle-ceiling-vs-3pp-threshold` — absolute-pp threshold
   with <3.4pp headroom above a measured baseline.
2. `per-token-full-sequence-routing-null` — F#305 reuse; mixed-domain
   per-token on full-sequence forward passes is identical to
   per-sequence by construction (cross-attention contamination).
3. `representation-bottleneck-not-architecture` — F#193 reuse; at
   near-oracle ceilings, switching routers cannot close the gap;
   remaining error is in representation geometry, not classifier
   expressivity.
4. `ridge-analog-already-killed` — F#340 reuse; the closest analog
   architecture failed on mixed-domain with 8.6pp accuracy drop.

## Findings reused (no new finding registered)
F#431, F#305, F#312, F#193, F#340. Preempt is F#305/F#312/F#193
combined-branch reuse; F#431 sets the ceiling; F#340 sets the
analog-failure precedent.

## Assumptions (logged per G1007)
- Held-out per-token task assumed to follow F#431 methodology
  (Gemma 4 tokenization, 5 real domains).
- "Jointly-trained" interpreted as MoLoRA-style co-training
  per `exp_molora_per_token_mlx`.
- Deviating from these assumptions would require a v2 pre-reg with
  a new KC; the ≥3pp threshold is structurally bounded above
  regardless.

## Verdict
**KILLED (preemptive).** K1551 FAIL marked structurally.
