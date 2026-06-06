# PAPER.md — exp_followup_output_space_qa_adapters

**Status:** KILLED — preemptive, structurally uninformative KC.
**Verdict:** KILLED (no code executed; three-lemma proof, see MATH.md).
**is_smoke:** false. **all_pass:** false.

## One-line conclusion
K1552 ("QA-format adapters beat NTP-format adapters on QA by ≥5pp") is either
tautological (L1) or prerequisite-gate-unmet (L2) or base-beat-impossible (L3).
The KC does not test the thesis it claims to test.

## Prediction-vs-measurement

| Quantity                               | Predicted (MATH.md)      | Measured       | Verdict           |
| -------------------------------------- | ------------------------ | -------------- | ----------------- |
| `Q(A_NTP)` Falcon-E-3B MMLU            | ≤ 0.42 (F#165 reproduce) | not measured   | preempt           |
| `Q(A_QA)` Falcon-E-3B MMLU             | [0.47, 0.53]             | not measured   | preempt           |
| K1552 delta `Q(A_QA) − Q(A_NTP)`       | ≥ 5pp (tautological)     | not measured   | preempt FAIL      |
| Prerequisite gate `Q(A_QA) vs Q(B)`    | [−0.08, −0.01]           | not measured   | preempt FAIL      |

## Why we preempted instead of running

Three independent lemmas (MATH.md) show K1552 outcome carries no information:

- **L1.** NTP-format adapters emit continuation-style prose on MCQ items (F#165
  measured −24% vs base on Falcon-E-3B). QA-format adapters emit letter answers by
  construction of their training distribution. Any positive Q(A_QA) trivially
  crosses 5pp over a prose-emitter — format-alignment is the measured effect, not
  composition quality.
- **L2.** F#166 impossibility structure requires a prerequisite gate: **single
  adapter must beat base before testing composition.** K1552 compares
  inter-adapter delta, not adapter-vs-base. PASS on K1552 does not validate the
  thesis.
- **L3.** F#477 (Gemma 4) measured single-adapter base-beat rate 2/5, with K1226
  FAIL (adapted acc < 0.50). Falcon-E-3B is ternary-quantized and instruction-
  tuned; its MMLU calibration is not weaker than Gemma 4's. Therefore the
  prerequisite gate is structurally unlikely to hold.

## Findings reused
F#166, F#165, F#477, F#167, F#168.

## Antipattern flags
- `tautological-inter-adapter-delta-ignores-base-baseline`
- `prerequisite-gate-unmet-output-space-composition`
- `bundled-orthogonal-fixes-format-plus-speed-one-kc`
- `format-alignment-symptom-fix-not-disease`

## Recommended v2 (not run)

Any re-scoped experiment must pre-register:
1. Base-beat gate: `Q(A_QA,single,d) ≥ Q(B) + 3pp` for ≥3/5 domains, **before**
   composition.
2. KC comparing top-2 to base: `Q(A_QA,top2) ≥ Q(B) + 5pp` (not vs a straw NTP
   adapter).
3. Cite F#167/F#168 — composition is structurally output-space MoE; binding
   constraint is single-adapter base-beat capacity, not architecture.

## Assumptions
- Falcon-E-3B MMLU priors are at least as strong as Gemma 4's on the relevant
  domains. (Both are instruction-tuned consumer-scale models; F#477 tested Gemma 4
  at rank 6 and found the gate unmet; Falcon-E-3B at similar scale has no
  structural advantage.)
- "QA-format supervision" does not re-train the base enough to change its MMLU
  priors; it only changes the output distribution shape.
- F#166's "prerequisite gate" is authoritative (third independent confirmation of
  the output-space impossibility class).
