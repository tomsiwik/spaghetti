# REVIEW-adversarial.md — P0: Behavioral E2E Quality

**Verdict: PROCEED (killed finding confirmed)**

## What This Experiment Got Right

1. **Proof structure is correct.** Theorem 1 (Q = ρ × δ) is a valid chain rule — no math errors.
2. **Prediction-vs-measurement table present.** PAPER.md satisfies the proof-first requirement.
3. **Root cause identified correctly.** MCQ training format mismatch is the right diagnosis:
   MMLU-style fine-tuning (select A/B/C/D) teaches concise answer format, not explanatory vocabulary.
4. **Impossibility structure derived.** δ_D ≤ 0 for MCQ-trained adapters on capable base → routing
   irrelevant. This is the correct mathematical conclusion.

## What Failed and Why

**Fatal prior extrapolation.** Theorem 2 used δ_D from Finding #436 (personal adapter, 76pp gain).
Finding #436's δ was high because the base model's *style compliance gap was 100%* — the base
never spontaneously adopted user phrasing. Domain adapters face δ ≈ 0 because Gemma 4 already
covers domain vocabulary in its pretraining. Using a personal-style prior for domain vocabulary
gain is an invalid domain transfer.

**Failure modes section had the answer.** MATH.md listed "δ_D < predicted" as a possible failure
but offered a faulty fix: "PPL reduced by 26.3% → domain-correct tokens must increase." This
conflates PPL reduction (distribution shift) with open-ended generation quality (behavioral shift).
These are orthogonal (r=0.08, established). The proof already had the tools to derive the
impossibility but didn't use them.

## Concerns (Non-Blocking)

1. **Vocabulary count rubric has direction bias.** A verbose base model scores "better" than a
   concise adapted model even if the adaptation is correct. The adapted model scoring LOWER may
   partially reflect the base model's verbose `<|channel>thought\n` preambles inflating counts —
   not true behavioral regression. This doesn't change the KILLED verdict but weakens confidence
   in the exact improvement rates.

2. **Medical 60% is an anomaly.** Medical is the one domain where base mean was LOW (1.4 terms/
   response) and adapted was higher (2.1). This is exactly the δ_D > 0 scenario — base had a
   real gap. This suggests the hypothesis "domain adapters help when base has a gap" is partially
   right. Worth noting for future experiments targeting domains with genuine base capability gaps.

3. **N=10/domain is underpowered.** At N=10, a rate of 30% has SE ≈ 14pp. The kill threshold
   of 80% is far enough above measured rates (30%, 20%) that this doesn't change the verdict,
   but it means the medical 60% result needs more data before concluding "medical is different."

## What This Tells Us For P1

The real finding is: **δ_D is a function of the base model's pre-existing capability gap in domain D.**
- Personal adapters: base gap = 100% (no spontaneous style compliance) → δ = 0.76
- Domain vocabulary: base gap ≈ 0% (Gemma 4 already covers domains) → δ ≈ 0
- Medical (potentially): base gap > 0% → δ = 0.6 (interesting)

This gives a testable prediction: **δ_D ≥ 0.5 iff baseline_accuracy(domain) < 50%.**
For next behavioral experiments, test on tasks where Gemma 4 demonstrably fails before adaptation.

## Decision

**PROCEED with killed finding.** Finding #457 is correctly stated. PAPER.md is complete.
Analyst should write LEARNINGS.md noting:
1. MCQ training creates format-register mismatch for open-ended tasks
2. δ_D requires pre-existing base capability gap (not just PPL reduction)
3. Medical exception: suggests vocabulary adaptation works when base gap exists
4. Next experiments must target behavioral domains where base model fails baseline tests
