# PAPER.md — exp_g4_svd_truncate_adapter

**Verdict:** KILLED (preempt-structural, pre-measurement)

**Antipattern:** tautological-inter-variant-delta-ignores-base-baseline — §5 clause (promoted at F#709, 3rd instance). This is the **4th instance**, sub-variant **intra-adapter-rank-delta**.

## Claim
K1611: "r=4 within 5% of r=6 on MMLU-Pro" (SVD truncation of Gemma 4 r=6 adapters to r∈{2,3,4}).

## Prediction-vs-measurement

| # | Prediction | Basis | Measured | Verdict |
|---|------------|-------|----------|---------|
| P1 | r=6 q_proj Gemma 4 ≈ M_base on MMLU-Pro | F#477 K1226 FAIL, adapted_acc 0.480 < 0.50 | Not measured (preempt) | Parent already measured; inherits |
| P2 | r=4 SVD-truncated ≈ r=6 ≈ M_base (degenerate-equivalence) | L1, Eckart-Young | Not measured (preempt) | Covered by parent |
| P3 | K1611 trivially PASS regardless of truncation quality | L1 | Not measured (preempt) | Structural — trivial |
| P4 | Adding per-variant floor `M_r ≥ M_base + 0.05` inverts PASS → FAIL | L2 / L4 remedy | Not measured (preempt) | Remedy path documented |
| P5 | Domain-PPL reformulation admits non-degenerate measurement | F#325 | Not measured (preempt) | Unblock path |

## Summary
K1611 is a comparison-only KC with no per-variant base-anchor. Under the parent-inherited target-FAIL regime (F#477: r=6 q_proj Gemma 4 adapted_acc 0.480 < 0.50, below 4-opt MCQ random chance), the comparison `|M_4 − M_6| ≤ 0.05` passes in the **degenerate-equivalence regime** where both variants collapse to base. "PASS" signals nothing about SVD-truncation quality preservation.

The §5 clause fires direction-symmetrically (|≤|, ≥, <) — this 4th instance adds the **intra-adapter-rank-delta sub-variant** (variants are SVD ranks of the same adapter), distinct from prior variant-axes (architecture K1552, training K1577/F#704, routing K1584/F#709).

Precondition check is favorable but moot: 3/25 r=6 q_proj Gemma 4 adapters exist on disk (code/math/medical, per F#627), but MMLU-Pro spans 14 categories — the 11 categories without a matching trained adapter are covered only by base priors per F#477 impossibility analysis.

## Assumptions logged (autonomy guardrail 1008)
- K1611's "within 5%" is read as absolute-delta `|M_4 − M_6| ≤ 0.05`, not relative `|M_4 − M_6| / M_6 ≤ 0.05`. Both readings are tautology-vulnerable; the absolute reading is more common and more lenient to the claim.
- MMLU-Pro is the canonical multi-domain MCQ benchmark; we do not substitute MMLU (easier) or GSM8K/HumanEval (single-domain).
- Parent adapter trained target is q_proj rank-6 scale-6.0 (per `adapter_config.json` on disk) — matches F#627 SUPPORTED configuration.
- Hygiene at 2 defects (success_criteria=[], references=[]) is below the 3+ promotion threshold per F#703 canon; reported as secondary, not trigger.

## Unblock path
File `exp_g4_svd_truncate_adapter_v2_domain_ppl`:
1. **Scope** to the 3 trained domains (code/math/medical) on held-out val splits — not MMLU-Pro.
2. **Metric** F#325 PPL ratio (non-MCQ, non-degenerate on these domains).
3. **KC design** per-variant base-anchor AND inter-variant delta: `PPL_r < PPL_base` for each r ∈ {2,3,4,6} AND `PPL_4 / PPL_6 ≤ 1.05` AND `PPL_2 / PPL_6 ≤ 1.15` (rank-sensitivity scan).
4. **Cite** F#325 (Qwen3-4B SUPPORTED) for the mechanism; frontier-extension to Gemma 4 architecture at scale=6 is the scientific increment.
5. **F#627** confirms parent adapter target-SUPPORTED on domain tasks (not MCQ), so shared-failure regime does not apply in this reformulation.
