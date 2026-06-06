# LEARNINGS.md — 2-Domain M2P Composition at 4B Scale

## Core Finding
Grassmannian isolation + TF-IDF routing scale identically from 0.6B to 4B (d=2560).
All three kill criteria passed: isolation at bf16 floor (1.38e-05), routing 100%, quality_ratio=1.3125 (exceeds SFT by 31%).

## Why
The QR-decomposition guarantee for A^T A = 0 is exact and dimension-independent — it holds at d=2560 the same way it holds at d=768.
TF-IDF routing is model-invariant by construction (LoraRetriever, 2402.09997), so routing quality is independent of scale.
SFT-residual M2P (Finding #403) provided the stable 4B base needed to close the composition loop.

## Implications for Next Experiment
Two open items: (1) code domain quality under composition — Finding #395 found format overfitting at 0.6B; verify at 4B.
(2) Scale to N=5 domains — N_max=320 is proven, N=2 is verified; next natural step is N=5 with real heterogeneous domains.
M2P encoder size (808M params) remains unresolved technical debt; VeRA fix was killed (Finding #380), so a new parameter-reduction approach is needed before production scale.
