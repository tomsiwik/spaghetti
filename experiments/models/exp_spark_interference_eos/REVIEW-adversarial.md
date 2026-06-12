# REVIEW (adversarial) — exp_spark_interference_eos

Verdict reviewed: **KILLED**. Route: **PROCEED** (seal killed).

## Mock / not-real checks — all clear
- `results.json` present, `is_smoke:false`, `verify-experiment.sh` exits 0.
- Model in MATH.md == code == results: `mlx-community/gemma-4-e4b-it-4bit`. Real mlx-lm 0.31.2, runtime 1282 s.
- Adapters are genuinely distinct (not a copy): layer-0 lora_a ||math-med||=2.21 > each individual norm
  (~1.5), i.e. they steer in materially different directions. 84 keys each, two separate files.

## User's specific concern — degenerate/noise-driven crossover? NO
The crossover is NOT an argmax-noise artifact of near-identical logits:
- Per-step deltas are large and real: on_delta_mean median ~1626, off_delta_mean median ~1504 (order 1e3,
  full-vocab L2 of logit difference, recomputed per step in code). Not near-zero, not constant.
- The off/on magnitude ratio sits at median 0.961 (range 0.66–1.41): the two adapters perturb the base by
  *comparable* magnitudes. THIS is precisely why "first off>on crossing" fires at T_cross∈{0..4} — when
  magnitudes are this close and order-unstable, the first crossing is near-immediate and content-blind.
  The paper's interpretation is exactly right; this strengthens the kill rather than undermining it.

## Consistency / integrity
- Δacc, median_savings, early_crossover_rate, and clause counts all recompute independently from `details`
  and match `measured` exactly (Δacc −0.68 = 0.02−0.70; KC-3 34/35 = 0.971 over 35 to-EOS-correct cases).
- idx 1 (sole early-correct) reconciles: answer at token 2, T_cross=3 → answer survives the cut.
- Kill thresholds in code == MATH.md §4 (0.02 / 0.15 / 0.20). Dir untracked in git → no post-hoc goalpost move.
- Clause 1 (Δacc) and Clause 3 (early crossover) fired; Clause 2 (savings 0.999) didn't, but only because
  the crossover amputates generation — passing it is evidence against, as the paper states. No tautology:
  baseline is the SAME math-on config to-EOS, like-for-like (F#866). Behavioral metric (GSM8K EM), not a proxy.

## Conclusion
Real run, distinct real adapters, non-degenerate per-step deltas, internally consistent, pre-registered
falsification crossed in the kill direction. Hypothesis refuted. Seal KILLED.
