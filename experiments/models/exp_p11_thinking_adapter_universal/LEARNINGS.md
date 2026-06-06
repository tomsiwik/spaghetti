# LEARNINGS.md — P11.H0: thinking-universal-v0

## Core Finding

Two-domain (math + code) LoRA training on v_proj+o_proj (r=8, scale=1.0, 2000 examples,
1000 steps) produced catastrophic forgetting on humanities: MMLU-Pro dropped 62.1%→47.6%
(−14.5pp, K1517 fails by 17.5pp); MedMCQA collapsed to 40.0% (K1518 fails by 15pp).
Thinking channel was preserved (2902 chars/q, K1519 PASS). Overall: KILLED.

## Why

Theorem 1's precondition **GD > 0.5** was violated by the training distribution. Math and
code are both STEM; their gradients are correlated (procedural token structures, symbolic
reasoning), so effective GD ≈ 0.2–0.3. The LoRA aligned with the dominant STEM subspace
and destroyed orthogonal humanities capacity. Per-category evidence: STEM preserved
(physics 66.7%, economics 73.3%), humanities collapsed (engineering 13.3%,
philosophy 20.0%). Finding #560 codifies this.

## Baseline reconciliation flag

Gate was Finding #536 baseline = 62.1%, but `exp_p11_baseline_eval` re-measured 40.7% on
the same model. Kill is robust to either baseline, but "adapter degraded MMLU-Pro" is
only true under F#536. Future experiments must **re-measure baseline in-run** instead of
citing an older finding; use the locally-measured number as the gate reference.

## Implications for Next Experiment (v2)

1. **Domain diversity ≥5 truly-diverse shards**: STEM + humanities + social science +
   medical + legal. Two STEM domains do not satisfy GD > 0.5; measure GD empirically
   before training (per-batch gradient cosine) as a pre-training gate.
2. **Locally-measured baseline**: run a no-adapter eval on the exact eval set/template in
   the same script; gate against that number, not Finding #536.
3. **Revisit KC K1518**: MedMCQA target of 55% is unrealistic without medical data;
   either drop the gate or include a medical shard in the training mix.
4. **Consider smaller adapter (r=4) or broader target modules** to reduce per-domain
   subspace lock-in.
5. **Per-category distribution > aggregate accuracy** as the primary LoRA-universality
   signal: aggregate 47.6% hides the STEM-vs-humanities split that is the real diagnostic.
