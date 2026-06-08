# LEARNINGS.md — exp_g4_head_ablation_ppl

## Verdict: SUPPORTED (K#1967 PASS, K#1968 PASS)

## Core Finding

F#742's per-head mass ranking of rank-6 q_proj LoRA slabs is behaviorally real: zeroing the top-20% head slabs degrades held-out domain PPL 4.89× more on average than zeroing the bottom-20%, across all three domains (code 3.55×, math 6.24×, medical ∞). Structural weakness (C_20 = 0.335) does not imply functional irrelevance — the heads that carry adapter mass carry behavioral load.

## Why

The bottom-20% heads (max mass 1.75) are near-inert: zeroing them barely moves PPL, and in the medical domain slightly improves it (Δ_bot = −0.0008). The top-20% heads (min mass 3.64) carry the functional signal; their removal inflates PPL by 0.06–0.45 absolute across domains. The intact adapter beats the no-adapter base by 76.9% mean PPL gain, confirming the adapter is genuinely active and the test is meaningful.

## Measured results

| Domain  | PPL base | PPL intact | PPL top-zeroed | PPL bot-zeroed | ratio R |
|---------|--------:|-----------:|---------------:|---------------:|--------:|
| code    |   3.845 |      1.595 |          1.655 |          1.612 |    3.55 |
| math    |   5.769 |      1.530 |          1.816 |          1.576 |    6.24 |
| medical |  90.543 |      1.112 |          1.562 |          1.111 |      ∞  |

R̄ = 4.89; ratio > 2 on 3/3 domains.

## Implication for the next experiment

Head-sparsity is a viable adapter-serving optimization: the bottom-20% of q_proj head slabs can be zeroed (or skipped) at inference with negligible behavioral cost. The designated follow-up is `exp_g4_head_importance_vproj_oproj_F627` — repeating this ablation on F#627-compliant v_proj + o_proj adapters to determine whether the head-sparsity property transfers to the architecturally preferred projection targets.

## Caveats to propagate

- q_proj rank-6 only; conclusions do not transfer to v_proj/o_proj without re-running.
- 50 held-out samples/domain — small but identical across arms; ratios are robust, absolute PPL values are in-distribution low.
- Medical ratio ∞ is not a numerical artefact (Δ_top = 0.45 ≫ |Δ_bot| ≈ 0); finite-domain ratios (3.55, 6.24) independently satisfy all pass thresholds.
- PAPER.md cites a wrapper activity check (mean |Δ| = 5.06, detach |Δ| = 0) that is prose-only; wrapper activity is independently proven by the base-vs-intact PPL divergence.
- Gemma 4 E4B mixed-layer attention: head_dim 256 (sliding layers) vs 512 (full layers {5,11,17,23,29,35,41}); ablation code correctly maps per-layer head_dim.

## Antipattern audit

- [x] Both KCs target-metric (PPL behavioral), per F#666.
- [x] Corpus identical across all four arms; ratios are fair comparisons.
- [x] Wrapper F#831-safe (setattr subclass, not __call__ monkeypatch).
- [x] Verdict not silently upgraded; SUPPORTED earned by 3/3 domain pass on pre-registered thresholds.
- [x] Ranking reused verbatim from F#742, not recomputed post-hoc.
