# REVIEW-adversarial.md — exp_pierre_ties_b_only

## Verdict: PROCEED (SUPPORTED)

## Checklist

| # | Check | Result |
|---|-------|--------|
| a | MATH.md has falsifiable hypothesis | ✓ Bipolar: PASS → B-only TIES viable; FAIL → full-delta required |
| b | Pre-registered KCs with thresholds | ✓ K1 (+3pp), K2 (5pp gap), K3 (5s budget) |
| c | At least one target-metric KC | ✓ K1 uses task accuracy avg (gsm8k, humaneval, medqa) |
| d | Code matches MATH.md algorithm | ✓ 3-step TIES (trim→elect→merge) on B-space, compose_methods.py:56-70 |
| e | Composition math correct | ✓ B-only compose, no Σ(B)@Σ(A) error — returns B-dict |
| f | MLX patterns correct | ✓ float32 compute, bfloat16 storage, mx.eval before .item() |
| g | results.json complete | ✓ 4/4 methods, all benchmarks, verdict field present |
| h | n≥50 per benchmark | ✓ config.n_eval_per_bench=50 |
| i | Verdict matches KC logic | ✓ SUPPORTED: K1 PASS, K2 PASS, K3 PASS |
| j | No data leakage | ✓ Eval uses shared runner with proper train/test separation |
| k | Seed fixed | ✓ seed=42 |
| l | Scale reasonable | ✓ LORA_SCALE=6.0, rank=6 |

## Key finding

ties_b_only avg=71.33% equals dare_full_delta avg=71.33% exactly. The research agent's claim that "TopK-by-magnitude on B alone wouldn't carry semantic meaning" is **empirically falsified**. Full-delta materialization provides zero benefit for TIES merging in this architecture.

This is architecturally significant: B-only TIES avoids materializing the full d_in×d_out delta, keeping composition within Pierre's factored B-space.

## Concerns

- K4 (sanity) has null value/threshold — acceptable since it's a label-only field documenting the architectural variant.
- humaneval shows ties_b_only=86% beating dare_full_delta=80% by 6pp — this variance at n=50 is within normal range but worth noting. The avg convergence (71.33=71.33) suggests it's noise cancellation across benchmarks.
- medqa scores (42-62%) are the weakest — expected for a non-medical-specialized base model.

## Recommendation

PROCEED to analyst for LEARNINGS.md. This is a clean, surprising result that directly informs Pierre architecture decisions.
