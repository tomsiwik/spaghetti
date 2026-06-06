# LEARNINGS: TT-LoRA E2E Benchmark (exp_p0_ttlora_e2e_benchmark)

## Core Finding
TT-LoRA (rank-6, v_proj+o_proj) achieves 65x compression with 93% GSM8K and 87% HumanEval
retention, but destroys MCQ discriminative capacity (21%, below 25% random chance). Retention
is task-type dependent, not uniform.

## Why
The uniform 84% retention model (Finding #516) assumed all tasks benefit equally from
dominant-subspace preservation. This is wrong: generative reasoning tasks (CoT chains) rely on
directional steering that survives low-rank projection; discriminative MCQ requires fine-grained
token probability differences that collapse under rank-6 TT truncation. Training loss converging
(0.179, lowest of 3 domains) while behavioral performance collapsed (-29pp vs baseline) is a
direct confirmation of the behavioral guardrail: metrics-as-proxies fail precisely where it matters.

## Two-Tier Retention Model (supersedes F#516 uniform 84%)
- Generative reasoning (CoT, code gen): ~90% retention at TT-rank 6
- Discriminative classification (MCQ, 4-way choice): ~42% retention at TT-rank 6

## Implications for Next Experiment
The 25-domain vision needs a mixed adapter strategy: TT-LoRA for reasoning domains (math, code,
law-drafting), higher-rank or different projection targets (q_proj/k_proj) for classification
domains (medical triage, routing decisions). Next priority: formalize the task-type sensitivity
theorem — what rank is needed to preserve discriminative capacity?

## Validated
- Theorem 2 (routing independence): confirmed exactly at 98.3% — routing operates on input text,
  not adapter weights, so compression never affects routing quality
- Compression economics: 325 KB/adapter → 25 domains = 8.1 MB total (vs 545 MB with standard LoRA)
- $2/domain vision: credible for reasoning-type tasks

## Open Question
What rank r guarantees MCQ retention >= 80%? The JL-lemma gives a lower bound on rank needed
to preserve pairwise distances in the token probability space. This is the next theorem to derive.
