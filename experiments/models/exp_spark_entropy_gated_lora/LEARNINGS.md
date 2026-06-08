# LEARNINGS — exp_spark_entropy_gated_lora

## Core Finding
The entropy-gated LoRA experiment was killed because its foundational premise — F#827's −12pp off-domain interference / +22pp on-domain lift — did not reproduce under the only in-repo math adapter (q_proj r6/scale6 on gemma-4-e4b-it-4bit, greedy, thinking, n=40). The fixed adapter produced +5pp HumanEval (not −12pp) and −7.5pp GSM8K (not +22pp), rendering both kill criteria undefined by construction.

## Why
The F#827 behavioral interference pattern is adapter-recipe-specific, not a stable property of the model or domain pair. A q_proj-only, r=6, scale=6.0 math adapter on this base/eval configuration produces weak and sign-inverted cross-domain effects compared to the adapter recipe that generated F#827. The entropy-gating mechanism itself fired correctly (mean gate 0.086 code / 0.063 math) — it had nothing to act on.

## Implication for the Next Experiment
Any experiment that cites F#827's interference/lift magnitudes as a substrate must first verify those magnitudes reproduce under the exact adapter/model/eval configuration it will use — this is now a mandatory pre-experiment gate. A v2 entropy-gating experiment is only viable after identifying (or training) an adapter that reproduces the F#827 pattern; absent that, entropy-gating as an axis-relocation strategy remains untested, not falsified.
