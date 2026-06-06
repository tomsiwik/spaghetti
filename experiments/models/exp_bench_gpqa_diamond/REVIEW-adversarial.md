# REVIEW: exp_bench_gpqa_diamond

## Verdict: PROCEED

## Data Integrity
All numbers in PAPER.md match results.json exactly:
- Base: 63/198 = 31.82% (reported 31.8%) ✓
- Adapted: 63/198 = 31.82% (reported 31.8%) ✓
- Delta: 0.0pp ✓
- Per-domain breakdowns all verified ✓
- Runtime: 42.8s + 50.5s = 93.3s ✓

## Prediction Accuracy
- Base accuracy (31.8%): Slightly below predicted lower bound (32%), acknowledged in paper. Acceptable.
- Google gap (-26.8pp): At predicted lower bound of [-26.6, -16.6]. Correct direction.
- Adapter delta (0.0pp): Within predicted [-6, 0] range. Floor effect explanation is sound.
- All three kill criteria outcomes predicted correctly.

## Mathematical Rigor
The "theorems" are calibration models rather than formal proofs. Acceptable for a benchmark experiment — the purpose is establishing empirical baselines, not proving structural guarantees. The MMLU-Pro ratio transfer and floor-effect formula are reasonable statistical arguments.

## Key Finding
The ~27pp thinking penalty being consistent across MMLU-Pro (27.1pp) and GPQA Diamond (26.8pp) despite different formats (10-option vs 4-option) is a genuinely useful calibration result. This suggests thinking mode provides a fixed capability boost, not a format-dependent one.

## Non-blocking Notes
1. The adapter delta (0.0pp overall) masks per-domain variation (Biology -5.3pp, Chemistry +3.2pp, Physics -2.3pp). These are small-N effects but worth noting for future SFT experiments.
2. Biology at 47.4% is a notable outlier — may reflect more factual-recall vs computation-heavy questions.

## Status: SUPPORTED
Appropriate for a benchmark calibration experiment. Both the thinking penalty and NTP-adapter-at-floor findings are well-supported by data.
