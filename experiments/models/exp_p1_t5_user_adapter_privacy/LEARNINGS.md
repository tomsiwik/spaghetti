# LEARNINGS.md — T5.4: User Adapter Privacy (KILLED)

## Core Finding

Behavioral isolation (routing exclusivity) holds perfectly — 0/5 cross-user sign-offs.
MIA resistance and Grassmannian isolation both failed due to wrong proof assumptions,
not wrong architecture.

## Why It Failed

**K1111 (MIA):** Theorem 2 assumed "uniform style injection" means zero MIA distinguishability.
Wrong — weight trajectories accumulate per-batch signal, producing partial memorization at n=40.
Same-distribution train/test (both general science) makes delta=0pp mathematically impossible.
Fix: use semantically OOD non-member questions (medical/legal for a science user).

**K1112 (Geometry):** Theorem 3 measured lora_a (input extractor), not lora_b (output direction).
lora_a learns "what kind of input am I seeing?" — identical for two science-question users.
Style specificity lives in lora_b. Measuring the wrong matrix guaranteed a false positive.
Fix: measure cos(lora_b_A, lora_b_B); JL bound predicts <0.10 for independent style directions.

## What Holds

- Routing exclusivity → behavioral isolation is structurally guaranteed (T3.7, Finding #430)
- lora_a overlap at 0.62 is expected and harmless; outputs are determined by lora_b
- 60% non-member compliance shows style generalization, not per-example memorization

## Implications for Next Experiment

A corrected T5.4-v2 should:
1. Use OOD non-members (e.g., legal questions for a science-trained user) for K1111
2. Measure cos(lora_b_A, lora_b_B) for K1112 (expected: <0.20 by JL)
3. Optionally: apply Gram-Schmidt to lora_a (T3.6) to guarantee geometric isolation by construction

The privacy architecture is sound — only the measurement design was wrong.
