# LEARNINGS.md — exp_method_composition_k_saturation

**Verdict: KILLED (smoke aborted at Phase-1 teacher gate).** DB `killed`;
`results.json.verdict=ABORTED`; K1730/K1731/K1732 all unmeasured.
Guardrail 1009 observed: PAPER.md narrative says PROVISIONAL, DB uses the
closest available label.

## Core Finding

Gemma-4-E4B-it-4bit under `enable_thinking=True` does not reliably emit
method-specific format scaffolding (`Problem restated:`, `Step 2`,
`Check:`, `Principle:`, `TL;DR:`). Teacher signature rates on n=5 per
method: 20 / 40 / 40 / 20 / 40 % — all five methods below the 70 % gate.
Zero method adapters were trained; the Skill-Mix (Arora & Goyal 2023,
arxiv:2310.17567) k-saturation curve could not be measured.

## Why

1. **Thinking-channel absorbs format-scaffold budget** (Finding #589).
   The `<|channel>thought…<channel|>` preamble consumes the
   instruction-following budget; after `strip_thinking` the visible
   answer satisfies the `Answer: X` contract but drops the method
   prefix/suffix the teacher prompt requested. Siblingclassof
   exp_method_vs_domain_adapter's Issue 2 (strip_thinking regex
   brittleness) is a separate failure in the *same* thinking-channel
   family.
2. **Anchored-strict regexes.** Disjointness during composition
   required single-anchor patterns (e.g. literal `Step 2`), which
   rejects natural variants ("Let me restate", "Step 1→3") that
   preserve the *method* but miss the *marker*.
3. **n=5 gate noise.** ±20 pp CI means a 40 % observation is
   consistent with a true 60 % rate — still below gate, but the
   smoke signal was noisier than strict-binary diagnosis suggests.

The failure mode is the **same class** as the parent
`exp_method_vs_domain_adapter` teacher-gate failure (40 % signature
rate at smoke). Two sibling kills under thinking-mode suffice to
designate teacher-side `enable_thinking=False` + few-shot
calibration as the default for any future
method/signature-dependent experiment on Gemma-4-E4B-it-4bit.

## Implications for Next Experiment

- **Any method/signature experiment on Gemma-4-E4B-it-4bit MUST**
  disable thinking on the teacher, use disjunction-surface regex
  (≥2 variants per method), n≥20 per method for the gate, and
  include ≥2 few-shot demonstrations (Orca-2 / Skill-Mix pattern).
- **Composition math is sound** — the rank-stack identity
  `Δ = concat_rank(B) @ concat_rank(A) = Σ B_i @ A_i` is preserved
  (antipattern-001 does not apply here). A v2 pre-reg that fixes
  only the teacher pipeline can measure the Skill-Mix curve
  without re-deriving composition.
- **Do not REVISE this experiment.** The fixes change the
  pre-registered signature regexes — a KC-modifying edit — so the
  v2 work must enter the DB as a fresh experiment per guardrail
  1005.
- **No new antipattern memory needed.** The thinking-mode
  absorption is already covered by `mem-antipattern-008`
  (THINKING-MODE TRUNCATION) as a generalisation; this instance
  refines the failure to *instruction-following budget*, not
  max-token truncation. Finding #589 is the appropriate record.

## References

- Arora & Goyal (2023). *Skill-Mix.* `arxiv:2310.17567` — predicted
  k-saturation curve; unmeasured here.
- Sibling: `exp_method_vs_domain_adapter` (KILLED) — same teacher-gate
  class failure, 40 % signature rate at n=15 smoke.
- Finding #589 (logged by reviewer): thinking-mode absorbs
  format-scaffold instruction-following budget.
