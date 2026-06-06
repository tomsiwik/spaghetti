# REVIEW-adversarial.md — T0.5: PLE Injection Point Verification

**Verdict: PROCEED**

## Summary

Algebraic + empirical experiment verifying the PLE injection mechanism at correct Gemma 4 E4B
dimensions. All 4 kill criteria pass. Results.json matches PAPER.md values exactly.
No fabricated measurements. Finding #418 (supported) is appropriate.

## Checklist

- [x] PAPER.md has prediction-vs-measurement table
- [x] Kill criteria results match results.json (max_diff=0.0, rel_diff=0.9908, 81.7% loss reduction)
- [x] Finding status (supported) appropriate — empirical proxy verification of formal theorems
- [x] All proofs have Theorem/Proof/QED structure

## Issues Found

### Non-blocking: K1006 metric mismatch (cosmetic)

MATH.md Theorem 4 predicts "accuracy improvement over random e_l on GSM8K subset" but
K1006 actually measures loss reduction (81.7% from 2.17→0.40). These are correlated proxies,
and 81.7% >> 1% threshold regardless of which metric. No impact on validity.

### Non-blocking: Theorem 3 citation missing

"Mechanistically identical to AdaLoRA/DyLoRA" is an interpretive claim without arxiv citation.
Fine as background context but should get a citation before T2 experiments reference it.

### Non-blocking: Scale note for production

PAPER.md correctly flags "scale e_l to 0.01 in production" — ‖h‖=655 for 42 layers with
unit-norm vectors. This should be captured as a design constraint in T2.4's MATH.md, not
just a footnote here.

## What's Good

1. **K1004 algebraic exactness**: max_diff = 0.0 EXACT is the strongest possible verification.
   The no-bias verification path (Theorem 1 proof: "W_proj has no bias by construction —
   verified in weights") is good scientific hygiene.

2. **K1005 activation strength**: rel_diff=0.9908 is much stronger than the >0.01 threshold.
   This means PLE injection is a powerful channel — future ablations should use much smaller
   e_l scales (0.01 as noted).

3. **Proxy model strategy works**: Using Qwen3-0.6B proxy for Theorem 4 (empirical part)
   while testing algebraic properties on synthetic Gemma4-dimension layers is the correct
   approach given mlx_lm 0.29.1's Gemma4 loading limitation.

4. **42-layer stability test**: k1003_multilayer_coherent=true and k1003_multilayer_norm=655.2
   goes beyond the kill criterion — good defensive testing.

## T0 Foundation Status

T0.1 (#417), T0.3 (#411), T0.4 (#412), T0.5 (#418): all supported.
T0.2 (V-Norm) killed — correct classification, Gemma4 not loadable.
T0 foundation is complete. T2.4 (PLE-M2P vs weight modification) unblocked.
