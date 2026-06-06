# Adversarial Review: P5.C0 Standing Committee Adapter

**Verdict: KILL (confirmed)**

## Evidence Integrity

Kill criteria results match `results.json` exactly:
- K1276: composed 70% vs domain-only 80% = -10pp (threshold +3pp) -- FAIL
- K1277: composed 40% vs domain-only 100% = -60pp (threshold 2pp) -- FAIL
- K1278: cos=0.0 (threshold 1e-4) -- PASS by construction

No evidence fabrication. Measurements are internally consistent across all configurations (base/committee/domain/composed).

## Mathematical Assessment

**Theorem 1 (Module-Disjoint Orthogonality)**: Trivially correct. Disjoint parameter support => zero inner product. No issues.

**Theorem 2 (Additive Composition Preserves Capabilities)**: **Fundamentally flawed.** The "proof sketch" claims cross-interaction is O(||DW_r||*||DW_d||) ~ O(2e-6) and therefore negligible. This analysis treats softmax as approximately linear under small perturbation, which is categorically wrong:

1. Softmax is an argmax approximation. Small DQ perturbations can shift which key gets maximal attention weight, redirecting ~100% of information flow. This is a first-order discontinuity at decision boundaries, not O(e^2).
2. The norm ratio ||DW||/||W|| ~ r/d is a parameter-space measure, not a functional measure. A rank-4 perturbation to Q can completely reorganize which tokens attend to which.
3. Autoregressive generation amplifies per-token errors multiplicatively, not additively. The claimed "per-token O(e^2)" becomes O(T*e^2) at best, O(e^T) at worst.

The paper correctly identifies this flaw post-hoc. The impossibility structure -- that attention creates obligatory multiplicative Q*V coupling -- is the genuine contribution.

**Theorem 3**: Vacuous given Theorem 2's failure. The linear transfer assumption (f*Delta_r improvement) requires the independence Theorem 2 was supposed to guarantee.

## Key Confound

Base model reasoning accuracy of 10% is an artifact: Gemma 4's `<|channel>thought` tokens consume the 120-token generation budget before reaching the answer. The model is reasoning correctly internally but truncating. The domain adapter (v_proj+down_proj) somehow suppresses thinking tokens, yielding 80% "reasoning" that is actually direct recall, not chain-of-thought.

This means the domain adapter's 80% reasoning score and the committee's 70% may be measuring entirely different capabilities. The comparison is confounded but does not change the KILL verdict -- the 60pp format degradation alone is disqualifying.

## Behavioral Evidence (Strong)

The degeneration patterns are severe and well-documented:
- "MEET MEET MEET..." repetition loops (attention fixation)
- "MEETING\nStep 1: Calculate..." pattern contamination
- "Step 1: Define the answer to answer" vacuous CoT

These are not marginal degradations -- the composed model produces fundamentally broken text. This confirms the KILL beyond any metric argument.

## Impossibility Structure (Sound)

The analysis that no module partition avoids Q*V coupling in transformers is correct and valuable:
```
y = softmax((W_Q + DQ)x * ((W_K)x)^T / sqrt(d)) * (W_V + DV)x
```
The output is a multiplicative function of both DQ and DV through the attention mechanism. Module disjointness in parameter space provides zero functional isolation.

## Finding Value

Despite the KILL, this experiment produces a high-value negative result:
1. **Module-disjoint != functionally independent** in transformers (generalizable principle)
2. **Same-module Grassmannian isolation** (Finding #49) is the correct approach -- both adapters on the same modules with orthogonal subspaces avoids Q*V coupling
3. **Softmax sensitivity** makes linear perturbation analysis invalid for attention-based composition

## Non-Blocking Notes

- Theorem 2's "proof sketch" label doesn't excuse the fundamental error -- the conclusion was stated as fact in the predictions table
- The 120-token max_tokens budget is too low for fair reasoning evaluation; future experiments should use 256+
- Committee-only getting 60% format (vs base 100%) shows the committee adapter itself introduces degeneration even in isolation
