# Mathematical Framework: Task-Based Evaluation of Composed Ternary Adapters

## Notation

| Symbol | Definition | Dimension |
|--------|-----------|-----------|
| W_base | Base model weights (ternary, frozen) | varies |
| A_i, B_i | LoRA factors for adapter i | (d, r), (r, d) |
| N | Number of composed adapters | scalar (5) |
| s | LoRA scaling factor | scalar (20.0) |
| alpha | Ternary STE scaling factor | scalar (mean absolute value) |

## Composition Method

Given N trained adapters with parameters {(A_i, B_i)}_{i=1}^N, the averaged-factor
composition merges A and B separately:

    A_merged = (1/N) * sum_i A_i
    B_merged = (1/N) * sum_i B_i

The effective composed output for input x is:

    y = W_base * x + s * (x @ A_merged) @ B_merged

Expanding:

    y = W_base * x + s/N^2 * sum_i sum_j (x @ A_i) @ B_j

This yields:
- N diagonal terms (i=j): contribute s/N^2 * (x @ A_i) @ B_i (own adapter)
- N(N-1) cross terms (i!=j): contribute s/N^2 * (x @ A_i) @ B_j (interference)

Cross-terms are negligible when adapters are near-orthogonal (|cos| ~ 0.002).

## Task Metrics

### Math Accuracy (MATH-500 subset)
    acc = (1/n) * sum_{k=1}^n I[grade(extract(response_k), answer_k)]

where extract() finds the last \boxed{} or trailing number, and grade() does
normalized string matching with numeric tolerance 1e-6.

### Code Syntax Validity
    valid_rate = (1/n) * sum_{k=1}^n I[ast.parse(code_k) succeeds]

### Keyword F1 (Medical, Legal)
    F1 = 2 * P * R / (P + R)
    P = |pred_tokens AND ref_tokens| / |pred_tokens|
    R = |pred_tokens AND ref_tokens| / |ref_tokens|

where tokens are lowercased, punctuation-stripped, length > 2.

### Creative PPL
    PPL = exp((1/T) * sum_{t=1}^T -log p(x_t | x_{<t}))

Lower PPL = better next-token prediction on creative text.

## Expected Behavior Under 1/N^2 Scaling

With N=5 adapters and averaged-factor composition, each adapter's diagonal
contribution is scaled by 1/N^2 = 1/25 = 4%. This means:

- Individual adapter effect is attenuated to 4% of its full strength
- Cross-terms add O(N(N-1)/N^2) = O(1-1/N) interference contributions
- Net effect: composition preserves direction but dramatically reduces magnitude

This predicts that composed model will be closer to base than to any individual
adapter, which is confirmed by the experimental results.

## Computational Cost

Per-token generation (no KV cache):
    FLOPs per token ~ 2 * d_model * n_params (full forward pass)
    For BitNet-2B: ~2 * 2560 * 2.4B ~ 12.3 TFLOPs per token

With 128 tokens max generation and 20 problems:
    Total math eval: ~20 * 128 * 12.3T ~ 31.5 PFLOPs

This explains the ~106 min runtime on Apple Silicon M-series.

## Assumptions

1. Ternary STE quantization in the LoRA forward pass preserves adapter specialization
2. Averaged-factor composition is the correct merge strategy (vs. unit-weight)
3. Task metrics (accuracy, F1, PPL) are more informative than perplexity for real utility
4. 20 eval examples per domain is sufficient for directional signal (not statistical power)
5. BitNet-2B-4T base model (not instruction-tuned) has meaningful baseline task performance
