# LEARNINGS.md — P3.B4: Pure Additive Composition

## Core Finding
Pure additive composition (76%→24%, Δ=52pp) is WORSE than B-GS (60%), contradicting the prediction that preserving unmodified signals would outperform projection. All five weight-space composition strategies (additive, B-GS, full-ΔW GS, α=1.0, pure additive) fail to preserve personal style above 60%.

## Why
Training distribution mismatch: personal adapter was trained on base model hidden states h_base but receives domain-shifted states h_base+ΔW_D at inference. Layers 26-41 (personal adapter's range) receive states that differ from training. B-GS serendipitously outperforms pure additive by forcing personal adapter into non-overlapping directions (n_overlap_layers=16), but even B-GS can only achieve 60% (16pp loss).

## Comparison Across P3.B Series
| Method | Style | Δ from baseline |
|---|---|---|
| Personal-only | 76% | 0pp |
| B-GS (P3.B1) | 60% | -16pp |
| Full-ΔW GS (P3.B2) | 40% | -36pp |
| Full-ΔW α=1.0 (P3.B3) | 0% | -76pp |
| Pure additive (P3.B4) | 24% | -52pp |

## Impossibility Structure
For any static weight-space additive composition of independently-trained adapters with overlapping layer ranges: the composition is doomed by training distribution mismatch. No weight projection (additive, B-GS, null-space GS) can fix a mismatch that originates at training time.

## Implications for Next Experiment
P3.B5: retrain personal adapter ON TOP of domain-adapted model (base + math adapter loaded). The personal adapter then learns to produce style FROM domain-shifted states — eliminating the distribution mismatch by construction. Prediction: style compliance ≥ 76% (matching personal-only baseline). arxiv 2402.03513 (Null-Space LoRA) confirms weight-space isolation alone is insufficient; sequential fine-tuning on the adapted model is the correct approach.
