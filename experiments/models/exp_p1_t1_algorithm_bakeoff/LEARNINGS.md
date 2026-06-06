# LEARNINGS — T1.6 Algorithm Bake-off (Finding #420, supported)

## Core Finding

LoRA r=6 is the optimal P1 adapter format: 1.44M params/domain, 7% GSM8K after 300 steps,
|cos|≈10⁻³ orthogonality, sr=3.76. HRA consistently underperforms LoRA at equal params by
2-3pp across both budget levels (40k and 107k params/layer).

## Why

HRA's step-time penalty (1.35-1.95× vs LoRA at equal params) compounds with a key impossibility:
single-domain SFT has a rank-1 gradient subspace, so HRA's high stable rank (26.5 vs LoRA's 9.4)
provides no benefit — all adapter rows collapse to the dominant task direction regardless of
architecture (Theorem: sr(V) ≈ rank(∇_ΔW L) = 1-2 for single-domain SFT, proven in T1.5 KILLED).
The HRA paper (2405.17484) claim holds only at larger scale / longer training / multi-domain data.

## Implications for Next Experiment

T2.1: Train first real Gemma 4 E4B domain adapter (math) with LoRA r=6. Predict ≥15% GSM8K
after 1000 steps (extrapolated from M2P 4B finding #403). This is the first T2 experiment —
actual Gemma 4 training after T0 foundation complete + adapter format decided.
