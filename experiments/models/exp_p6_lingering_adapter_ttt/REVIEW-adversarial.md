# Adversarial Review: P6.A1 TTT-Style Embedded Adapter Update

## Verdict: PROCEED (KILLED confirmed)

The KILLED status is correct and well-supported. All three kill criteria fail with
measurements matching results.json. The impossibility structure is mathematically
sound. No fabrication detected.

## Strengths

1. **Theorem 1 is rigorous**: The gradient path necessity argument correctly identifies
   that LoRA inside attention cannot exploit TTT's closed-form trick. The non-linearity
   chain (softmax + GELU + RMSNorm) is correctly identified.

2. **Prediction table is honest**: K1289 was predicted to FAIL before running — good
   scientific practice. The accuracy prediction of 50-60% was close but slightly
   optimistic (measured 40%).

3. **Behavioral evidence is strong**: The hallucination pattern ("ZephyrFlow's internal
   X" replacing specific facts) and topic contamination ("capital of Japan" -> "ZephyrFlow")
   demonstrate a clear failure mode, not just metric degradation.

## Issues (non-blocking)

1. **Topic contamination is UNDERREPORTED**: PAPER.md flags one general knowledge
   failure (capital of Japan). But results.json reveals pervasive "Zephyr" contamination
   across correct general knowledge answers too: "The Nile Zephyr", "The element with
   atomic number 1 is Zephyr", "The event for which I'm Zephyr is Zephyr." The 90%
   general accuracy (keyword match) masks severe qualitative degradation. This
   strengthens the kill — the model is leaking domain tokens into every response.

2. **P6.A0 comparison table correction is messy**: Lines 47-58 of PAPER.md first
   show Fly.io and 90 days as "missed by both" then correct this inline. Final numbers
   (TTT 4/10, P6.A0 6/10) are accurate per results.json, but the presentation is
   confusing. Non-blocking since the final comparison is correct.

3. **Theorem 2 is softer than Theorem 1**: The "local loss information bound" relies
   on data processing inequality qualitatively ("much of I(h_l; A|Q) is in the
   representation of Q") rather than providing a quantitative bound. This is adequate
   for a guided exploration but would need tightening for a verification-type experiment.

## Key Finding Validated

**Response-only masking is necessary for factual LoRA learning.** All-token loss
causes: (a) 2.5x signal dilution on factual tokens, (b) hallucination via pattern
learning without fact anchoring, (c) pervasive topic contamination even on unrelated
queries. This is a structural result, not a hyperparameter issue — no tuning of
learning rate or rank can fix it.

**TTT zero-cost is impossible for transformer LoRA.** The non-linear gradient path
through attention makes closed-form gradients impossible. This closes the TTT
direction for our architecture.

## Routing: PROCEED to Analyst for LEARNINGS.md
