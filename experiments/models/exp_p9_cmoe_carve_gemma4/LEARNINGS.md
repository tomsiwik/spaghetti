# LEARNINGS: exp_p9_cmoe_carve_gemma4

## Core Finding
CMoE training-free FFN carving (arXiv:2502.04416) is mathematically exact on
Gemma 4 E4B (decomposition verified, max diff 2.86e-6), but structurally blocked
on MLX: N small matmuls < 1 fused matmul at current expert granularity (N=8,
D=1280), making >1.0x speedup impossible without O(k) conditional compute.

## Why
MLX GPU dispatch overhead dominates the 50% compute reduction from sparse
activation. Splitting one (10240 × 2560) fused matmul into 8 × (1280 × 2560)
sequential matmuls loses kernel fusion efficiency. This is a platform-level
constraint, not a hyperparameter — O(k) conditional execution (gather/scatter)
is not efficiently supported in MLX at this expert size.

## Secondary Finding
PPL measurement was confounded: Gemma 4 E4B is instruction-tuned; evaluating
on raw text yields base PPL ≈37K (meaningless). Any carving error compounds
catastrophically in this high-entropy regime. Quality experiments on IT models
must use chat-formatted inputs matching the model's training distribution.

## Implications for Next Experiment
1. P9 line (CMoE-style MoE carving for speed) is structurally blocked on MLX.
   Do not pursue CMoE quality recovery unless O(k) compute barrier is solved first.
2. Any future FFN decomposition experiment must benchmark dequantized-not-carved
   baseline to isolate decomposition error from quantization error.
3. IT model evals must use chat-formatted prompts (not raw text) — add this as
   a standard check in all future experiments using Gemma 4 E4B IT.
4. Adapter-level composition (P0/P1 line) remains the correct path for domain
   specialization; base model restructuring via carving has no MLX speed path.

## Blocked Downstream
4 P9 experiments blocked: expert identity, PERFT-R, DES, sigmoid routing.
All remain blocked until O(k) MLX conditional compute is demonstrated feasible.
