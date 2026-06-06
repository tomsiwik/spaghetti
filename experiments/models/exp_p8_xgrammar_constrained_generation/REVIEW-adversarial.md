# REVIEW-adversarial.md — P8.A0: Grammar-Constrained Code Generation

**Reviewer**: Adversarial Reviewer
**Date**: 2026-04-14
**Verdict**: PROCEED (with 2 non-blocking notes)

---

## BLOCKING ISSUES: None

---

## NON-BLOCKING NOTES

### 1. Theorem 2 is circular — "valid derivation" undefined

MATH.md states: "if thinking trace T is a valid derivation, then P(correct_code | C, T) ≥ P(correct_code | C)".
This is circular: we can't know T is valid before the code is generated.
The experiment correctly measures empirically whether think mode helps, which is sufficient for K1334.
Accept Theorem 2 as a plausible motivation, not a formal guarantee.

**Impact**: K1334 remains testable regardless. Theorem 2 is motivation, not proof.

### 2. Think-then-code prompt contradiction

`build_think_prompt` tells the model "Return ONLY the function definition" but then
seeds the response with `# Reasoning: ` — this is contradictory. The model may:
- Ignore the ONLY constraint and output full reasoning + function (correct behavior)
- Try to compress reasoning into a comment (edge case)

This is an intentional design to test whether reasoning-before-code helps, and the
extract_function() parser handles mixed output correctly. Not a bug.

---

## CORRECTNESS CHECKS (all pass)

### Self-repair loop structure (run_experiment.py:347)
`for retry in range(1, N_RETRY)` with N_RETRY=3 gives range(1,3)=[1,2] = 2 retries.
Total attempts: 1 initial + 2 retries = 3 = N_RETRY. Theorem 1's N=3 bound is correct.

### Code adapter path (line 32)
`REPO_ROOT / "micro/models/exp_p1_t2_single_domain_training/adapters/code"` — confirmed
present on disk from prior LiveCodeBench REVISE verification. Graceful skip if missing.

### Greedy repair at temp=0
Each retry feeds a DIFFERENT prompt (original broken code + error message) → distinct
generation even at temperature=0. Repair loop is not degenerate.

### Memory cleanup
`del model, tokenizer; cleanup()` between phases. Correct.

### N=20 problems × 2 models × 3 modes = 120 generations
Estimated runtime: 120 × 300 tokens / 73 tok/s ≈ 8 minutes. Well within budget.

---

## MATH.md QUALITY

- Theorem 1 (self-repair convergence): correct probability argument via De Morgan's law ✓
- Theorem 3 (overhead): trivially correct direct calculation ✓
- Kill criteria K1333-K1335 are precise and testable ✓
- Citations to arXiv:2411.15100 + arXiv:2601.07525 ✓
