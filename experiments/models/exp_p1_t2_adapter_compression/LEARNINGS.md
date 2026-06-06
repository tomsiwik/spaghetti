# T2.2 Adapter Compression — Learnings

## Core Finding
4-bit mixed quantization (lora_a fp16, lora_b 4-bit) compresses LoRA adapters 3× (4.99MB → 1.67MB) with no measurable quality degradation at n=25 evaluation across math/code/medical domains.

## Why
MLX quantization error (7.61% weight-space for 4-bit) is absorbed by the residual stream because the adapter's ΔW contribution is small relative to the 4-bit base model's own quantization noise. Zero-mean errors cancel across 42 layers, giving 200× better orthogonality preservation than the Theorem 2 worst-case bound predicted.

**Key insight**: lora_a (d_in × r=6) cannot be quantized with MLX (group_size must divide last dim; 6 not in {32,64,128}). Mixed strategy (quantize only lora_b) is forced by MLX constraints but works well.

## Implications for Next Experiment
25-domain system fits in 41.8 MB (vs 124.8 MB fp32). T2.6 (5-domain training) can proceed with 4-bit compression as the default serving format. 2-bit offers marginal additional savings (1.52 MB) with no added risk — but the incremental benefit over 4-bit is negligible.
