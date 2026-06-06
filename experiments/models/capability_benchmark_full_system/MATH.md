# Capability Benchmark: Full System Verification

## Type: Verification (Type 1)

This experiment verifies a proven framework with quantitative predictions derived
from prior experimental findings.

---

## A. Failure Mode Identification

**The disease:** Composition may HELP on tasks where the adapter encodes task-relevant
FORMAT (reasoning chains, code syntax, extraction templates) but HURT on tasks where
the bottleneck is KNOWLEDGE retrieval from base model weights.

**Why it is a real risk:** exp_competitive_benchmark_routed was KILLED because routed
composition at s=20 degraded MMLU math by -20pp and MMLU legal by -10pp. These are
KNOWLEDGE-retrieval tasks (multiple choice factual recall) where the adapter's format
perturbation interferes with the base model's stored factual associations.

**The degenerate state:** A composition system that improves format-dependent tasks
but silently destroys knowledge-retrieval tasks, giving a false sense of quality.

---

## B. The Right Question (Reframe)

**Wrong question:** "How do we make composition not hurt MMLU?"

**Right question:** "For which task types does LoRA composition provably help, and for
which does it provably hurt, as a function of the task's format-dependency?"

**The answer:** The LIMA hypothesis (Zhou et al., 2305.11206) establishes that fine-tuning
primarily teaches FORMAT, not knowledge. Finding #258 confirmed: CPT is a no-op, adapters
encode FORMAT not KNOWLEDGE. Therefore:

- Tasks where FORMAT = CAPABILITY (GSM8K chain-of-thought, code generation, NER extraction):
  composition HELPS because the adapter provides the format the task requires.
- Tasks where FORMAT != bottleneck (MMLU factual recall): composition HURTS because the
  format perturbation interferes with knowledge retrieval without adding capability.

---

## C. Prior Mathematical Foundations

**LIMA Hypothesis (Zhou et al., 2305.11206):** A pre-trained model contains almost all
knowledge needed for tasks. Fine-tuning teaches the model the FORMAT of interaction
(style, structure, response patterns), not new factual knowledge.

**Finding #258 (this project):** CPT is a near-identity adapter at 200 iters / 80K tokens /
rank-16. SFT adapters encode format transformations, not knowledge injections. Confirmed
empirically: CPT loss did not converge, but CPT output preserved base capability.

**Finding #249 (this project):** Two behavioral regimes exist:
- FORMAT regime (s <= 4): preserves base model behavior, best for domains where base
  model knowledge is sufficient (legal, finance factual recall)
- CAPABILITY regime (s >= 20): activates learned FORMAT patterns (chain-of-thought,
  code syntax), required for domains where the task IS the format (math, code, medical QA)

**Finding #237 (this project):** GSM8K +10pp is the only consistent competitive advantage
across 3 independent experiments. Math adapter at s=20 reliably improves chain-of-thought
reasoning, which is a FORMAT skill.

---

## D. Theorem and Predictions

**Proposition 1 (Format-Capability Equivalence).**
Let M be a pre-trained LM, A_d a LoRA adapter trained via SFT on domain d, and
s_d the per-domain scale. Define format_dependency(T) in [0,1] as the fraction of
task T's difficulty attributable to output format vs. factual knowledge.

Then for a task T with format_dependency(T) = f:

  quality(M + s_d * A_d, T) - quality(M, T) ~ c * f - delta * (1 - f)

where c > 0 is the format-skill gain from the adapter and delta > 0 is the
knowledge-retrieval interference from the weight perturbation.

**Interpretation:**
- When f ~ 1 (GSM8K, HumanEval, NER): gain ~ c, composition helps
- When f ~ 0 (MMLU factual recall): gain ~ -delta, composition hurts
- Crossover at f* = delta / (c + delta)

**Quantitative Predictions (from prior findings):**

| Benchmark | Format Dependency | Adapter | Scale | Predicted Direction | Predicted Magnitude |
|-----------|------------------|---------|-------|--------------------|--------------------|
| GSM8K (chain-of-thought) | f ~ 0.9 | math | 20 | IMPROVE | +10pp (Finding #237: consistent +10pp across 3 experiments) |
| Code generation | f ~ 0.9 | code | 20 | IMPROVE | +15pp (Finding #249: code +36% behavioral gain) |
| Clinical NER | f ~ 0.8 | medical | 20 | IMPROVE | +5pp (Finding #249: medical +18% behavioral gain) |
| MMLU (factual) | f ~ 0.2 | mixed | mixed | NEUTRAL to HURT | -5pp to 0pp (Finding: -20pp at s=20, neutral at s=4) |

**Incoherence prediction:** Finding #249 showed 0% incoherent output with per-scale
routing across all 5 domains. K3 predicts < 5% incoherent.

---

## E. Assumptions & Breaking Conditions

1. **LIMA holds at 2B scale.** If the base model lacks reasoning/coding knowledge
   entirely, no amount of format training helps. Breaking: GSM8K base < 5%.
   (Current base = 38%, so this holds.)

2. **Adapter quality is sufficient.** SFT adapters must have learned useful format
   patterns. Breaking: adapter at optimal scale performs same as base.
   (Prior experiments show math adapter at s=20 gives +700% on math eval.)

3. **Oracle routing is accurate.** We use oracle domain routing (test prompts are
   labeled by domain). Breaking: wrong adapter applied to wrong domain.
   (Oracle routing = 100% accuracy by construction.)

4. **Scale parameters are correct.** Using {math:20, code:20, medical:20, legal:4, finance:1}
   from Finding #249. Breaking: optimal scales shifted.
   (These were validated in generation_quality_perscale.)

---

## F. Worked Example (Not applicable - this is a verification experiment)

The "worked example" IS the prior experiments:
- generation_quality_perscale: 5 domains, per-scale routing, behavioral eval
- competitive_benchmark_routed: MMLU + GSM8K, showing the knowledge/format split

This experiment unifies both into a single verification with the explicit
format-dependency framework.

---

## G. Complexity & Architecture Connection

- Model load: ~4GB (BitNet-2B-4T unpacked to bfloat16)
- Per-adapter overhead: 7 LoRA modules x 24 layers x rank-16 = ~27M params
- Pre-merge composition: 0% inference overhead (weights merged before generation)
- Total runtime: ~20 min (4 conditions x ~5 min each)

---

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   Format-dependency partitioning: by routing to per-domain optimal scales,
   format-tasks get amplified (s=20) while knowledge-tasks get minimal perturbation (s<=4).

2. Which existing theorem(s) does the proof build on?
   LIMA hypothesis (Zhou et al., 2305.11206): fine-tuning teaches format, not knowledge.
   Finding #249: two behavioral regimes with distinct optimal scales.

3. What specific numbers does the proof predict?
   GSM8K: +10pp (base 38% -> 48%); Code: +15pp; NER: +5pp; MMLU: -5pp to 0pp; Incoherence: <5%.

4. What would FALSIFY the proof?
   If GSM8K (a format-dependent task) DEGRADES with the math adapter at s=20, the
   format-capability equivalence is wrong.

5. How many hyperparameters does this approach add?
   Count: 5 (one per-domain scale). Each derived from Finding #249 behavioral optimization.

6. Hack check: Am I adding fix #N to an existing stack?
   No. This is a verification of the existing proven system, not a new mechanism.
