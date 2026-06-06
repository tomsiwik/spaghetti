# LEARNINGS.md — exp_p2_a0_medical_pubmedqa_adapter

## Status: KILLED — Finding #459

## Core Finding
Format-matched LoRA on PubMedQA (700 examples, rank=4) achieves δ_D=+0.015 vs required +0.15.
Standard LoRA adapter training cannot improve behavioral accuracy when the base model is
near-randomly uncertain (Q_base ≈ 1/C = 0.303 ≈ 1/3 for 3-class task).

## Why
δ_D ≈ 0 when two conditions hold simultaneously: (1) base has no systematic wrong prior
(near-uniform class uncertainty), and (2) format gap is absent (Gemma 4 already understands
yes/no/maybe format). Without a systematic error to correct, LoRA has no signal to amplify.
Contrast with Finding #409: Qwen3-4B base=23% (NOT near chance) benefited from M2P because
it had a structural wrong prior that could be corrected.

## Implications for Next Experiment
Domain adapters fail for medical QA because the bottleneck is KNOWLEDGE DENSITY, not format.
Fix requires RAG-style context injection at inference time — inject relevant PubMed abstracts
so the base model reasons over evidence rather than relying on parametric memory.
Alternative: chain-of-thought distillation on (question + abstract + reasoning → answer).
Citation: arxiv 2005.11401 (RAG, Lewis et al. 2020).

## Impossibility Structure (from REVIEW-adversarial.md)
δ_D ≈ 0 iff Q_base ≈ 1/C (uniform over C classes).
Q_pipeline = ρ_D × δ_D = 0.988 × 0.015 = 0.015 (effectively zero behavioral gain).
No routing improvement can compensate for δ_D ≈ 0.
