# LEARNINGS: exp_p0_discriminative_diagnosis

## Core Finding
Compression (20x, TT-LoRA rank-6) is the disease behind MedMCQA discriminative collapse,
not the NTP training objective. Standard LoRA r8 improved MCQ by +22pp (30.5%→52.5%);
TT-LoRA r6 degraded it by -12pp (18.5%), a 34pp gap.

## Why
TT-LoRA rank-6 (135K params) preserves only dominant NTP singular directions and discards
the discriminative tail. Standard LoRA rank-8 (2.7M params, 20x more) has capacity to
encode both generative and discriminative features. Critically: TT-LoRA achieved *lower*
NTP loss (0.169 vs 0.179) but far worse MCQ accuracy — confirming that training loss is a
poor proxy for discriminative capacity. (arXiv:2504.21190)

## Implications for Next Experiment
TT-LoRA needs higher rank or mixed MCQ training for discriminative tasks. The E2E
benchmark's 21% MedMCQA was a compression artifact, not a training objective failure —
so explicit MCQ fine-tuning or higher rank (r16+) are the fixes to test next.
