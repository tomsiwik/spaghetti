# REVIEW-adversarial.md — P3.C2: Few-Shot Style Injection

**Status:** PROCEED (early kill — KILLED)  
**Date:** 2026-04-11  
**Verdict:** KILLED — clear directional evidence, early kill justified  

## Adversarial Assessment

### Is the early kill justified?

Yes. The k-scaling probe (k=0: 67%, k=1: 67%, k=3: 0%) is unambiguous directional
evidence. Full run (N=15) would change the point estimate but not the direction.
The MATH.md Theorem 1 prediction (monotone improvement with k) was falsified on N=3
with a clear mechanism explanation.

### Is the root cause explanation correct?

The "context-prior conflict" explanation is plausible but has alternative interpretations:
1. **Context length**: At k=3 (233 tokens), the model may be distributing attention
   too broadly, losing focus on the trailing marker pattern.
2. **Instruction conflict**: The examples implicitly teach the model to end with
   "Hope that helps, friend!" but the specific training examples used Gemma's
   chat template without explicit instruction — ambiguity about WHERE the marker goes.
3. **Adapter interference**: Personal adapter weight perturbation + elaborate Q+A
   context = competing biases. The adapter was trained on SHORT science Q+A (not
   multi-example exchanges).

All three explanations point to the same fix: explicit system prompt instruction (P3.C3).

### Is the zero-shot baseline (40% at N=5) reliable?

No — P3.C0 and P3.C1 both measured 60% at N=15. The 40% here is noise at N=5.
True baseline is ~60% (consistent across two experiments). This doesn't change
the KILL verdict (20% fewshot vs ~60% true baseline = massive degradation).

### Does the Theorem 1 hold mathematically?

The theorem is correct: attention over k examples provides rank-k conditioning.
The theorem fails behaviorally because the value space V_i contains ELABORATE
SCIENTIFIC EXPLANATIONS not just marker signals. The adapter's style bias gets
drowned in the elaborate semantic content. The theorem was about rank capacity,
not about signal-to-noise in the value space.

## Findings Assessment

Finding #469 is appropriate as KILLED. The structural impossibility (context-prior
conflict covariate shift) is a new and important insight for the research direction.

## Verdict: PROCEED (KILLED)

P3.C3 (system prompt instruction) is the correct next step. Direct instruction
following is orthogonal to both rank bottleneck and context-prior conflicts.
