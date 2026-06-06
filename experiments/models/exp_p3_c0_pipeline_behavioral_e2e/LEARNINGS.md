# LEARNINGS.md — P3.C0: Full Pipeline Behavioral E2E

## Core Finding
The full Pierre pipeline (ridge routing → domain-conditional composition → personal adapter) works end-to-end: routing=100%, style=60% (at threshold), math=20%. All kill criteria pass (Finding #467, SUPPORTED).

## Why
Routing is fully solved via TF-IDF+Ridge vocabulary transfer (arxiv 2205.01068). Style compliance in-pipeline (60%) is lower than isolation (92%) because ρ̄_C is question-distribution-dependent — training used a fixed template; diverse questions escape the learned format. The 32pp gap (92%→60%) reflects template fragility, not composition failure.

## Implications for P3.C1
Increase personal adapter training examples (40→100+) or iterations (300→500) to improve robustness across diverse questions. Target: in-pipeline style ≥80%. Routing does not need changes. Math MCQ is a poor benchmark (near-chance); use word-problem evaluation instead.
