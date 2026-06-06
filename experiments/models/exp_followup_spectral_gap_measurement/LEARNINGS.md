# LEARNINGS — exp_followup_spectral_gap_measurement

**One-line:** The `sqrt(30)` Davis–Kahan placeholder in
`pro_composition_mmlu/MATH.md` is wrong by ~36×. Measured cross-model
ratio is `R = 0.84`, not `~30`.

## What happened
- Pure measurement: SVDs of all attn projections in BitNet-b1.58-2B-4T
  (120 mats) and Gemma-4-E4B-4bit (168 mats), 100 % success.
- Median relative spectral gap at k=16: BitNet 0.00347 / Gemma 0.00292.
- Singular-value ratio σ_16/σ_17 = 1.005–1.012 in **both** models.
- Runtime 130 s; verdict `SUPPORTED` (measurement produced).

## What this changes
- F#320's *observation* (0 pp MMLU at fp16 scale ≤ 5, −5.5 pp on ternary)
  stands — it is empirical.
- F#320's *mechanism* (Davis–Kahan via 30× spectral gap) is refuted by
  the measurement. A different protective mechanism is at work.
- Confirms F#603's σ_k/σ_{k+1}≈1 flatness observation, now extended to
  Gemma-4-E4B-4bit.

## What this does not change
- The Davis–Kahan theorem itself — only the gap values plugged into it.
- BitNet vs fp16 degradation asymmetry — the phenomenon is real, just
  not the proposed cause.

## Antipatterns avoided
- No composition math, no LoRA training, no adapter loading, no routing.
  Pure weight-matrix SVD on deployed models.

## Next candidates surfaced
- A targeted experiment separating `||ΔW||_2` magnitude from `delta`
  would identify the true MMLU-protection mechanism. Defer unless a
  planner prioritises it — the spectral arc is already closed
  (feedback memory `spectral_arc_closed`).
