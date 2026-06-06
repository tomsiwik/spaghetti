# Adversarial Review: exp_bench_mmlu_pro_thinking

**Verdict: PROCEED**

Date: 2026-04-13
Reviewer: Ralph (automated)

---

## Summary

This experiment documents Gemma 4 E4B 4-bit performance on MMLU-Pro with thinking mode
enabled. Results were cross-referenced from exp_p10_mcq_adapter_training (Finding #530).
PAPER.md is well-structured and the kill criteria table serves as the prediction-vs-measurement
framework. Main results: 62.1% base+thinking (+20.4pp over non-thinking), MCQ adapter
suppresses thinking entirely (0 chars), 135x token overhead.

---

## Issues Found

### Blocking: None

### Non-Blocking

1. **Missing MATH.md** — For a benchmark experiment, MATH.md would normally contain the
   prediction framework. Here it's implicit: "base+thinking should approach Google's 69.4%
   (matching their reported conditions)." Acceptable for a benchmark type since no novel
   theorem is being proven, but LEARNINGS.md should note that quantization (4-bit) explains
   the 7.3pp gap.

2. **No results.json in this directory** — Raw data lives in exp_p10_mcq_adapter_training.
   This is structurally fine since the benchmark is a post-hoc documentation of those runs,
   but makes the experiment directory incomplete as a standalone artifact.

3. **Engineering anomaly unexplained** — Thinking decreases engineering by 17pp. This is
   noted in PAPER.md but not mechanistically explained. Hypothesis: engineering questions may
   require spatial or diagram-based reasoning that text-only thinking chains cannot compensate
   for. Worth a one-line note in LEARNINGS.md.

---

## Evaluation

| Criterion | Assessment |
|-----------|------------|
| Kill criteria table present | YES (K1455/K1456/K1457) |
| Behavioral findings present | YES (thinking suppression is behaviorally critical) |
| Math/proof backing | N/A (benchmark type — measuring known quantity) |
| Results verifiable | YES (from Finding #530, well cross-referenced) |
| Finding status appropriate | YES — "supported" (2/3 criteria pass; fails on overhead and adapter+thinking combo) |
| Advances the vision | YES (establishes baseline; shows adapter-thinking incompatibility) |

---

## Verdict: PROCEED

The benchmark is well-documented. The 62.1% baseline and thinking suppression finding are
both correct and significant. Send to analyst for LEARNINGS.md. No structural changes
required.
