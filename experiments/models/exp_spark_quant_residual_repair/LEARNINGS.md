# LEARNINGS — exp_spark_quant_residual_repair

## Core finding
The fp16 LoRA delta at q_proj is essentially orthogonal to the 4-bit dequantization
residual: signed cosine similarity is −4.66e-5 (math), −6.14e-5 (medical), +2.95e-5 (python),
all at the O(1/√D) noise floor and indistinguishable from a norm-matched random rank-6 null.
Adapters do not repair quantization error.

## Why
LoRA minimizes task loss, not dequant error; the rank-6 subspace it discovers is determined
by the gradient signal from the downstream task, not by the structure of the quantization
residual. The two subspaces are effectively uncorrelated at d_model scale, so alignment
at 2–3 orders below the predicted 0.05–0.30 band is noise, not signal.

## Implication for the next experiment
Do not pursue quantization-residual repair via adapter training in any form: the subspace
mismatch is fundamental, not a tuning problem. Approaches that require the adapter to
correct low-level weight representation errors (rather than operating in task space) belong
to the same dead class and should be rejected at spark time.
