# LEARNINGS: exp_pro_base_validation

## Core Finding

Qwen3-4B-4bit is a viable Pierre Pro composition base: 2.26 GB memory (5% of budget),
82.6 tok/s generation, 92% MMLU-mini-50 (logit-based). The critical discovery is d=2560
matching BitNet-2B-4T, enabling direct transfer of all Grassmannian A-matrices and
composition machinery. Base model cannot follow instructions (GSM8K 48%, IFEval 33%),
confirming instruction adapter is the mandatory first training step.

## Why This Happened

**Memory efficiency:** 4-bit quantization with group_size=64 compresses 3.67B params to
2.26 GB — consistent with QLoRA's (2305.14314) finding that 4-bit NF4 preserves model
quality at ~4.25 effective bits/weight. The 20% overprediction in MATH.md likely comes
from embeddings also being quantized by mlx-lm (not kept in bf16 as assumed).

**Throughput exceeded prediction:** 82.6 tok/s = 68% bandwidth utilization of M5 Pro's
273 GB/s. This is within the predicted 60-75% utilization range when computed against
actual model size (2.26 GB) rather than predicted size (2.7 GB). M5 Pro memory controller
achieves consistent utilization across our experiments.

**MMLU inflation:** The 92% score on 50 hand-curated questions is inflated relative to
full MMLU. An Empirical Study of Qwen3 Quantization (2505.02214) shows 4-bit quantization
degrades MMLU by several percentage points on the full benchmark. Qwen3 Technical Report
(2505.09388) positions Qwen3-4B above Qwen2.5-7B on many benchmarks, suggesting true
full-MMLU is likely 70-80%. The 92% still clears our 60% threshold at 95% CI lower bound
(81%), so the finding holds regardless.

**Base model instruction failure:** GSM8K 48% and IFEval 33% are expected for a pretrained
base model without instruction tuning. The model generates continuations rather than answers.
This is well-characterized in LoRA Learns Less and Forgets Less (2405.09673): base models
require instruction-format finetuning before task evaluation is meaningful.

## Confirming Evidence

- **QLoRA (2305.14314, NeurIPS 2023):** 4-bit quantization + LoRA finetuning matches
  full-finetuning quality. Validates that our 4-bit Qwen3 base is not degraded beyond
  recoverability by adapter composition.

- **Qwen3 Technical Report (2505.09388):** Qwen3-4B outperforms Qwen2.5-7B on STEM/coding
  benchmarks. Confirms the model's knowledge quality is strong for its parameter class.

- **Empirical Study of Qwen3 Quantization (2505.02214):** 4-bit quantization introduces
  measurable but modest degradation. Near-lossless at 8-bit; tradeoff at 4-bit acceptable
  for composition experiments.

- **LoRA Learns Less and Forgets Less (2405.09673):** LoRA maintains better source-domain
  performance than full finetuning. Directly relevant — our composition approach adds
  adapters to a frozen quantized base, inheriting this forgetting resistance.

## Contradicting Evidence

- **IR-QLoRA (2402.05445):** Finds that quantization information loss cannot be fully
  recovered by LoRA, especially at ultra-low bitwidths. For 4-bit LLaMA-30B, LoRA-finetuned
  quantized model underperforms the original unquantized model on MMLU. Risk: our 4-bit
  base may have a lower ceiling than full-precision. Mitigation: we measure composition
  degradation relative to the 4-bit baseline, so the absolute ceiling is less relevant.

- **LoRA vs Full Fine-tuning: An Illusion of Equivalence (2410.21228):** Sequential LoRA
  training accumulates "intruder dimensions" (high-ranking singular vectors dissimilar to
  pretrained weights) causing more forgetting than full finetuning at scale. Risk: composing
  many adapters sequentially could degrade the base more than expected. Mitigation: our
  architecture uses frozen A-matrices (Grassmannian) and only trains B, avoiding the
  intruder dimension mechanism.

- **How Much Knowledge Can You Pack into a LoRA Adapter (2502.14502):** Knowledge injection
  via LoRA degrades reasoning benchmarks (MMLU, TruthfulQA) as new fact count increases.
  Risk: domain adapters that inject domain knowledge may degrade the base's MMLU. This is
  exactly why MMLU degradation is our composition metric — we will detect this if it occurs.

## Alternative Approaches

- **LQ-LoRA (2311.12023):** Low-rank + quantized matrix decomposition enables sub-3-bit
  quantization with minor degradation. Could push model to ~1.5 GB, doubling adapter
  headroom. Worth investigating if memory becomes a constraint at high adapter counts.

- **CLoQ (2501.18475):** Calibrated LoRA initialization for quantized LLMs. Corrects the
  quantization-induced error in the forward pass before adapter training. Could improve
  adapter convergence on our 4-bit base.

- **Selective LoRA (2501.15377):** Fine-tuning without catastrophic forgetting by selecting
  which layers to adapt. Relevant for instruction adapter training where we want to minimize
  base knowledge degradation.

## Implications for Next Experiments

1. **d=2560 match is the most valuable result.** All Grassmannian A-matrices (N_max = 25,600)
   transfer from BitNet experiments without modification. This eliminates re-derivation of
   orthogonality guarantees — the proofs carry over exactly.

2. **MMLU-mini-50 (logit-based) is the composition degradation metric.** Consistent
   methodology matters more than absolute calibration. All future experiments must use the
   same 50-question subset and logit-based evaluation for comparable results.

3. **Instruction adapter is the mandatory first step.** Without it, no task-based evaluation
   (GSM8K, IFEval, behavioral generation) is possible. This is not a limitation — it's a
   design point: the instruction adapter becomes the first composable module.

4. **82.6 tok/s base sets the throughput budget.** Adapter overhead must keep total generation
   above ~60 tok/s to remain usable. BitNet achieved 97 tok/s with adapters; Qwen3-4B
   starts lower, so overhead tolerance is tighter.

5. **Intruder dimension risk (2410.21228) is mitigated by Grassmannian architecture.**
   Our frozen A-matrices prevent the accumulation of intruder dimensions that sequential
   LoRA training creates. This is a structural advantage worth verifying empirically.

## Recommended Follow-Up

**exp_pro_grassmannian_init (P0):** Initialize Grassmannian A-matrices for Qwen3-4B (d=2560,
r=16). Verify orthogonality guarantees transfer from BitNet track. Motivated by Finding #317
(d=2560 match) and the Grassmannian interference prevention chain (Findings #225, #304-#314).
The A-matrices are architecture-independent — they depend only on d and r, both identical.

**exp_pro_instruction_adapter (P0):** Train instruction-following adapter on Qwen3-4B base.
Motivated by Finding #317 (GSM8K 48%, IFEval 33% without instruction tuning). Required before
any behavioral evaluation is possible. Literature: LoRA Learns Less and Forgets Less (2405.09673)
shows LoRA instruction tuning on frozen base preserves knowledge while adding task capability.
