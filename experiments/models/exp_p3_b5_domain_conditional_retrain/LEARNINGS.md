# LEARNINGS.md — P3.B5: Domain-Conditional Personal Adapter Retraining

## Status: SUPPORTED (Finding #466)

## Core Finding

Domain-conditional retraining (fuse domain adapter into base, then retrain personal adapter
on the fused base) achieves **0pp composition degradation** (92% composed = 92% personal-alone).
All prior P3.B1–B4 geometric/algebraic strategies failed (−16pp to −76pp loss).

## Why

The root cause for all P3.B1–B4 failures was covariate shift: the personal adapter was
trained on h_base but received h_base+ΔW_domain at inference. No static weight-space
composition can fix this — d_H(P_base, P_domain) > 0 is irreducible via linear algebra.
Domain-conditional retraining sets d_H = 0 exactly by aligning training and inference
distributions. Theorem 2 from MATH.md verified. Related: TIES-Merging (arxiv 2306.01708)
identified interference from independently trained adapters — same root cause.

## Implications for Next Experiment

P3.C0 (behavioral E2E pipeline) should adopt domain-conditional retraining as the default
composition strategy. Two open questions for P3.C0: (1) scaling — O(N_domains × N_users)
personal adapters may be impractical, consider multi-domain fused base; (2) FP16 deployment
cost (~14GB domain_fused_base vs ~4GB 4-bit) needs resolution before production use.
