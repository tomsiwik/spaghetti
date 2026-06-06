# LEARNINGS — exp_p0_composition_benchmark

## Core Finding
Standard LoRA pre-merge at full scale catastrophically destroys benchmark accuracy
(GSM8K 0%, HumanEval 0%, MedMCQA 20% — vs solo 73%/63%/50%). MedMCQA merged (20%)
falls below the unmodified base (31%), meaning interference actively degrades existing
capability. TF-IDF routing survives at 90.7%.

## Why
Three independent LoRA A-matrices are not orthogonal — they share subspace directions
from gradient descent. Pre-merging creates O(N) total perturbation while the useful
signal for any domain remains O(1), yielding SNR ~ 1/(N-1). At N=3 the interference-to-
signal ratio is ~2:1. PPL (smooth, averaged) absorbed this in Finding #505; benchmark
accuracy (binary per question) does not — any flipped answer accumulates. Finding #505's
2.1% PPL degradation was a misleading proxy for composition quality.

## Implications for Next Experiment
Orthogonal adapters (Grassmannian/PoLAR, Finding #341: cos=1.7e-16) are structurally
required before any pre-merge attempt. The routed pipeline works (routing accuracy ×
solo accuracy ≈ 95-99% of solo), validating the Pierre architecture's dual strategy:
orthogonal pre-merge for top-N static domains, routing for dynamic domains.
Next: implement orthogonal adapter training and re-test composition on benchmarks.
