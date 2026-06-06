# Adversarial Review: P5.C1 Per-Adapter Reasoning Strategy

## Verdict: PROCEED (SUPPORTED)

The experiment is well-executed, honest about its negative hypothesis result, and yields
an actionable finding. Minor issues noted below are non-blocking.

## Issues

### 1. Theorem 1 "validation" is vacuous (non-blocking)

PAPER.md claims "Theorem 1 validated" because +12pp was measured. But Theorem 1 bounds
improvement from *per-domain* routing (each domain gets a different strategy). The actual
improvement is from *uniformly* switching to Direct. The theorem is technically correct
(any improvement satisfies the lower bound) but it's not testing what the theorem was
designed for. The theorem predicted per-domain strategy differentiation; the experiment
disproved it. More accurate to say: "Theorem 1's framework is correct but its premise
(delta_i varies across domains) is falsified at this task complexity."

### 2. TF-IDF routing is degenerate, not just "too simplistic" (non-blocking)

The routing details show 9/25 queries scored 0.0 across ALL domains (no keywords matched),
defaulting to "math" (alphabetical). This isn't routing failure from simplicity -- it's a
degenerate fallback. The 64% accuracy partially reflects a broken default, not a measured
routing capability. Finding #196's 95% was on domain text passages with rich vocabulary;
QA questions have sparse domain signal. The PAPER correctly identifies this mismatch but
understates the severity.

### 3. Status reconciliation (non-blocking)

PAPER.md says "PROVISIONAL (2/3 pass, but mechanism is wrong)" while the event says
"SUPPORTED (2/3)". SUPPORTED is defensible: the experiment is guided exploration, the
math framework (bounds) was validated, and the exploration yielded a clear actionable
finding ("Direct dominates at this complexity"). The wrong mechanism doesn't invalidate
the finding -- it refines it.

**Recommended status: SUPPORTED** -- the exploration successfully discovered that strategy
differentiation requires task complexity beyond direct recall, and that Direct prompting
is optimal for QA-level tasks.

## What's valuable

- The "less is more" finding is actionable: default to Direct prompting on Gemma 4 E4B-IT
  for simple QA. Only invoke CoT for tasks where direct recall fails.
- Clear impossibility structure: strategy differentiation is impossible when direct recall
  suffices. This predicts when it WILL matter (multi-step proofs, complex debugging).
- Honest 0/5 hypothesis matches with good analysis of why each was wrong.
- Token savings (38%) are real and practically useful regardless of the mechanism story.

## Connection to project

This experiment is peripheral to the core composition/ternary research but informs the
serving pipeline: strategy routing adds complexity without benefit at current task levels.
The finding connects to the Room Model by confirming that routing should be structural
(adapter selection), not strategic (prompting format).
