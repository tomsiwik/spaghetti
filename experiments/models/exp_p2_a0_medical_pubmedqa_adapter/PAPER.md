# PAPER.md — exp_p2_a0_medical_pubmedqa_adapter

## Hypothesis

Format-matched LoRA training on PubMedQA data will improve behavioral accuracy on
PubMedQA 3-class classification (yes/no/maybe) when the base model accuracy < 50%.
MCQ-trained adapters will NOT improve accuracy due to format-register mismatch (Theorem 2).

## Prediction vs Measurement

| Metric | Prediction (MATH.md) | Measured | Pass? |
|--------|---------------------|----------|-------|
| Base PubMedQA accuracy | 0.30–0.40 | 0.303 (60/198) | K1166 PASS |
| Format-matched LoRA δ_D | > 0.15 (+15pp) | +0.015 (63/198=0.318) | K1167 FAIL |
| Format-matched LoRA accuracy | 0.45–0.60 | 0.318 | KILL |
| MCQ-trained LoRA δ_D | ≤ 0.05 | pending (full run) | K1168 pending |
| Training time | < 5 min | 7.8 min (470s pubmed + 165s mcq) | acceptable |

**Full run**: N_TRAIN=700, N_TEST=198 (after 500-iter pubmed + 200-iter MCQ training).

## Key Result: KILLED

K1167 fails catastrophically. Format-matched LoRA achieves only delta=+0.015 (+1.5pp) vs
the required +0.15 (+15pp) threshold. The gap is 10x smaller than predicted.

Base accuracy 0.303 ≈ 1/3 (random for 3-class). The adapter produces 0.318 ≈ 0.303 + noise.

## Why Theorem 1 Failed

Theorem 1 stated: "For Q_base < 0.5, format-matched LoRA achieves δ_D > 0 in expectation."

The theorem's premise was necessary but not sufficient. Two hidden conditions not in Theorem 1:

**Condition A (systematic error)**: Base model must have a SYSTEMATIC wrong prior, not random
uncertainty. At Q_base ≈ 1/3 on a 3-class task, the model is near-randomly uncertain. There
is no strong wrong prior for the adapter to correct.

**Condition B (format gap)**: The base model must be producing incorrect FORMAT, not just
lacking knowledge. PubMedQA format (yes/no/maybe) is trivially simple — Gemma 4 understands
the 3-class format without training. Format-matching adds no behavioral signal.

Compare to Finding #409 (Qwen3-4B, base=23%): Qwen3-4B showed stronger systematic errors,
lower base accuracy (23% < 30%), and benefited from M2P resampling (+32pp). The gap here
was not format but KNOWLEDGE.

## Impossibility Structure

**δ_D ≈ 0 when**: Q_base ≈ 1/num_classes AND no systematic error pattern exists.

Formally: If base logits are near-uniform over C classes, then:
- p(y=correct|x) ≈ 1/C for all x
- LoRA adapts the distribution but cannot discover task-specific signal from 700 examples
  when Gemma 4's pretraining already covered the domain (albeit without PubMedQA-specific
  reasoning chains)

**The right question was wrong**: "Does format matching help?" presupposes format is the
bottleneck. The actual bottleneck is KNOWLEDGE SIGNAL DENSITY — 700 PubMedQA examples
cannot overcome a base model trained on 10T+ tokens of medical text that already reaches
30.3% on a 3-class task (near chance).

## Structural Fix for Next Experiment

From Finding #409: M2P achieved +32pp via CONTEXT-AWARE retrieval (inject relevant
abstracts at inference time). This is NOT an adapter problem. The medical domain requires:
1. Relevant abstract retrieval (RAG-style context injection) at inference time
2. Adapter trained to EXPLOIT context (not to memorize answers)

Alternative: Chain-of-thought distillation — train on (question + abstract + reasoning chain → answer)
instead of (question → answer). This provides structural signal the model can learn.

## Kill Criteria Summary

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| K1166: base_acc < 0.50 | < 0.50 | 0.303 | PASS |
| K1167: δ_D(format-matched) > 0.15 | > 0.15 | +0.015 | FAIL → KILL |
| K1168: δ_D(MCQ) ≤ 0.05 | ≤ 0.05 | pending | — |

## Connection to Pierre Architecture

Q_pipeline = ρ_D × δ_D (from Findings #458, #457)
- ρ_D = 98.8% (PROVEN, ridge routing)
- δ_D = +0.015 for PubMedQA (not +0.15 as required)
- Q_pipeline = 0.988 × 0.015 = 0.015 (effectively zero behavioral gain)

The Pierre pipeline CANNOT deliver behavioral value for medical PubMedQA via standard LoRA.
Context injection (RAG) or chain-of-thought distillation required.

## References

- Finding #457: δ_D ≈ 0 for MCQ-trained domain adapters (MMLU data)
- Finding #409: PubMedQA base=23%, M2P=55%, SFT=22% on Qwen3-4B
- Finding #458: Ridge routing 98.8% at N=25
- arxiv 1909.06146: PubMedQA dataset (Jin et al. 2019)
- arxiv 2106.09685: LoRA (Hu et al. 2021)
- arxiv 2012.13255: Intrinsic dimensionality (Aghajanyan et al. 2020)
