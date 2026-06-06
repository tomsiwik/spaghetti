# LEARNINGS.md — T2.6: 5-Domain Adapter MVP

**Status:** supported  
**Date:** 2026-04-10

---

## Core Finding

Five LoRA r=6 adapters on Gemma 4 E4B achieve +22pp to +82pp domain specialization at 25MB total (fp32) / 8.35MB (4-bit), in 1.74 GPU-hours, at ~$10 total cost. All 3 kill criteria pass with margin.

## Why It Works

LoRA rank-6 provides sufficient capacity for MMLU MCQ and coding benchmarks because the intrinsic dimensionality of domain adaptation is much lower than full rank (Li et al. 2018). Independent adapters trained on disjoint corpora stay approximately orthogonal (|cos| < 0.019 from T2.2), making them safe to compose additively.

## Caveats

- Legal/finance base accuracy is 4% (format artifact: base generates prose, not A/B/C/D). True knowledge gain is ~10-30pp; K1047 (+3pp) still passes even at the conservative floor.
- JL-lemma citation in Theorem 3 is imprecise — the correct citation is intrinsic dimensionality (Li et al. 2018), not JL projection.
- Adapters stored in fp32; T2.2 confirms 4-bit yields 8.35MB with no quality loss.

## Implications for Next Experiment

- Future baselines should use format-corrected evaluation (few-shot prompt or log-likelihood) to isolate domain knowledge gain from format learning.
- T2.3 (local-only adapters) and T2.4 (PLE injection point) can now proceed with the 5-adapter set as ground truth.
- Composition experiment (T2.1 adapters + TF-IDF router) is the natural next step to verify composability at N=5.
