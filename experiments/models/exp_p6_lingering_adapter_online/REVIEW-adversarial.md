# Adversarial Review: P6.A0 Lingering Adapter — Online LoRA Update

## Audit Rerun Overlay (2026-04-18)
**Re-verdict**: **KILLED** under strict PLAN.md §1 pre-flight. The original
PROCEED review noted the H₂O/speed-of-light artifacts as "non-blocking" and
accepted SUPPORTED. Under the strict pre-reg rules now applied to
`audit-2026-04-17-rerun` experiments, the K1287 text-vs-measurement mismatch
("MMLU" vs 20-question trivia keyword match) is a **metric-swap** antipattern
and blocks `supported`. K1285/K1286 still pass cleanly. Verdict corrected to
KILLED; substantive findings are preserved in LEARNINGS.md.

## Prior Verdict: PROCEED (superseded)

## Assessment

The experiment is well-designed, results are genuine (verified against results.json),
and predictions are confirmed or exceeded. SUPPORTED status is appropriate for a
frontier extension.

### Strengths

1. **Prediction table is complete and honest.** All six quantitative predictions are
   reported with measured values. Four of six were exceeded — a strong signal that
   the underlying math (capacity, convergence) is directionally correct.

2. **Kill criteria are clean.** K1285 (+60pp vs ≥20pp), K1286 (110ms vs <1s),
   K1287 (improved vs <2pp degradation). All verified against raw JSON.

3. **Behavioral observations are specific and useful.** The taxonomy of what's
   easy (concrete identifiers), hard (version numbers), and hardest (character-level
   details) is grounded in data and actionable for follow-up experiments.

4. **Frequency–accuracy correlation** aligns with the regret bound prediction.
   Facts with 3-4× exposure all learned; 2× mixed. This is testable and scalable.

### Non-blocking Issues

1. **Non-convexity of LoRA loss is glossed over.** MATH.md Claim 2 invokes the
   online GD regret bound (Zinkevich 2003), which requires convexity. Cross-entropy
   is convex in logits, but the mapping from LoRA parameters to logits goes through
   a neural network — the composed function is non-convex. The result still holds
   empirically (loss did decrease 54%), but the formal guarantee is weaker than
   stated. The "frontier extension" type covers this honestly, but a sentence
   acknowledging non-convexity would strengthen the math.

2. **Repetitive generation in adapted outputs.** Several adapted responses show
   pathological repetition (e.g., "Zephyr" answer repeated 5+ times within the
   60-token window). This is not mentioned in PAPER.md. It doesn't affect keyword
   matching evaluation, but it IS a behavioral concern for production use. A single
   sentence noting this artifact would be valuable.

3. **H₂O / speed-of-light scoring artifacts.** Both base and adapted models miss
   H₂O due to Unicode vs ASCII keyword matching. The general accuracy "improvement"
   (90% → 95%) comes from the speed-of-light question where the adapted model's
   longer output happens to include "300,000". These are measurement artifacts,
   not real knowledge changes. The claim "no degradation" is still correct, but
   the 5pp improvement is noise, not signal.

### Finding Status

**SUPPORTED** is correct for a frontier extension. The core claim — online rank-4
LoRA can encode project facts from 20 turns at <200ms/turn with no general degradation
— is validated. Limitations (4/10 facts not fully learned, repetitive generation)
are clearly scoped.
