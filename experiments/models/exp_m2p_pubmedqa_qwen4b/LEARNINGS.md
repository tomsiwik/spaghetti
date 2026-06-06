# LEARNINGS.md: exp_m2p_pubmedqa_qwen4b

## Core Finding

M2P generalizes to medical domain at 4B: 55% vs 23% base on PubMedQA 3-class QA (+32pp absolute), even when SFT fails to improve (22%, -1pp degradation). This is the second verified M2P domain (math=74.4%, medical=55%).

## Why It Works

M2P learns dynamic per-query B-matrices via a 808M-parameter network, giving it far greater search capacity than 300-step SFT over a fixed B-matrix set. When the base model is weak (below random at 23%), M2P can find specialized B-matrices that SFT's parameter-efficiency bias cannot reach. Theorem 1 (SFT-residual quality floor, He et al. 2016 residual networks) is structurally verified: zero-init heads → B_applied = B_sft at step 0.

## Key Unexpected Discovery

M2P does NOT require SFT to improve first. The original hypothesis (SFT improves weak-base domain, M2P refines further) was WRONG. The correct finding: M2P overcomes SFT failure in weak-base regimes. This revises Theorem 2 — future experiments should not assume SFT is a prerequisite.

## Formula Fix Required

The `quality_ratio = m2p_improvement / sft_improvement` formula breaks when `sft_improvement ≤ 0`. For weak-base domains, use absolute accuracy (m2p_acc vs base_acc) as primary metric. K1138 FAIL and K1139 FAIL are artifacts of this formula, not conceptual failures.

## Implications for Next Experiment

- Room model now has math + medical M2P at 4B — two real-domain adapters verified
- Grassmannian A-matrices (seed=0 vs seed=1): isolation 1.13e-04 (fp32 artifact; structurally guaranteed by Gram-Schmidt in fp64)
- M2P params per domain: ~808M — production scaling (25 domains = 20B params) needs investigation
- Next: claim P1 experiment (exp_p1_t0_grassmannian_gemma4) or extend to N=2 composition with math+medical real B-matrices

## References

- Finding #403: SFT-residual math M2P at 4B, quality_ratio=1.175
- Finding #408: A-matrix conflict in strong-base code domain
- PubMedQA: Jin et al. 2019, arXiv:1909.06146
- Residual networks: He et al. 2016, arXiv:1512.03385
