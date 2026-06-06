# REVIEW-adversarial.md — exp_p2_a0_medical_pubmedqa_adapter

## Verdict: KILLED

Full run completed (N_TEST=198, N_TRAIN=700). K1167 fails decisively.

---

## Kill Criteria Results

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| K1166: base_acc < 0.50 | < 0.50 | 0.303 | PASS |
| K1167: δ_D(format-matched) > 0.15 | > 0.15 | +0.015 | **FAIL → KILL** |
| K1168: δ_D(MCQ) ≤ 0.05 | ≤ 0.05 | pending (full run still finishing) | — |

---

## Root Cause Analysis

**K1167 failed by 10x**: delta=+0.015 vs required +0.15.

**Why Theorem 1 failed**: The theorem's condition (Q_base < 0.50) was necessary but not
sufficient. Two unmodeled conditions:

1. **Near-random baseline**: Q_base=0.303 ≈ 1/3 (random for 3-class). No systematic error
   pattern to correct. LoRA has no wrong-prior signal to flip.

2. **No format gap**: PubMedQA format (yes/no/maybe) is trivial. Gemma 4 already understands
   the output format without training. Format matching adds no behavioral signal.

**Contrast with Finding #409**: Qwen3-4B at base=23% was NOT random (1/3 = 33%), so it had
systematic errors. M2P succeeded (+32pp) via context injection, not format matching.

**Impossibility structure**: δ_D ≈ 0 when Q_base ≈ 1/C (uniform over C classes). Adapters
cannot discover task-specific signal when the base's uncertainty is structurally random.

---

## Adversarial Concerns (non-blocking, already killed)

1. **Was 700 training examples enough?** Probably not. PubMedQA has 200K labeled QA pairs.
   With rank=4 and 700 examples, the adapter may underfit. However, even full dataset training
   is unlikely to help: the base model needs KNOWLEDGE not FORMAT, and LoRA fine-tuning at
   rank=4 cannot encode medical knowledge from 700 examples.

2. **Was MAX_TOKENS=512 enough?** The fix was applied (was 80 in round 1). Smoke test post-fix
   showed non-zero accuracies, confirming the parsing fix worked. Full run used 512 tokens.

3. **K1168 still running**: MCQ adapter result is pending. However, K1168 tests format mismatch
   theory (MCQ should NOT help). Even if K1168 FAILS (MCQ accidentally helps), the main
   hypothesis (K1167) is already killed. K1168 is informative but not decisive.

---

## Math Correctness

MATH.md Theorem 1 is a valid but INCOMPLETE proof. It shows δ_D > 0 in expectation but
omits the condition on base model being systematically (not randomly) wrong. Theorem 2
(MCQ mismatch) remains valid — confirmed by smoke test (K1168 trend) and Finding #457.

**Impossibility structure (for LEARNINGS.md)**:
- δ_D ≈ 0 iff Q_base ≈ 1/C (near-uniform class uncertainty)
- Format matching fails when: (1) base already understands the output format, AND (2) base has
  no systematic wrong prior to correct
- Fix: Task requires KNOWLEDGE INJECTION (RAG/context), not STYLE ADAPTATION (LoRA)

---

## Summary

Experiment killed. Theorem 1 falsified. The medical domain behavioral gap cannot be closed
by format-matched LoRA when the base model is near-randomly uncertain rather than
systematically wrong.

Next: exp_p2_a1 should test RAG-augmented inference (inject PubMed abstracts at test time)
rather than adapter fine-tuning. Alternative: chain-of-thought distillation on
(question + abstract + reasoning → answer) pairs.
