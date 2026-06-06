# Unified Routing Pipeline: Mathematical Foundations

## Notation

| Symbol | Meaning | Shape/Range |
|--------|---------|-------------|
| x_t | Input token at position t | scalar (token ID) |
| h_t | Hidden state at position t | (d,), d=2560 |
| H(t) | Shannon entropy of base model output at position t | scalar >= 0 |
| tau | Entropy threshold (Otsu-derived) | scalar, ~2.10 |
| f_j(h_t) | Routing head j's score for token t | scalar in [0,1] |
| A_j | Adapter j's LoRA parameters | dict of tensors |
| w_j | Composition weight for adapter j | scalar in [0,1] |
| N | Total number of adapters | 5 |
| k | Number of selected adapters (top-k) | 2 |

## Pipeline Definition

For each token position t, the unified pipeline executes:

### Stage 1: Entropy Gate

Compute base model logit entropy:

    H(t) = -sum_v p(v|x_{<t}) * log p(v|x_{<t})

where p(v|x_{<t}) = softmax(logits_base(x_{<t}))[v].

Decision rule:
- If H(t) < tau: use base model output (skip all routing and composition)
- If H(t) >= tau: proceed to Stage 2

From proven results: f_skip = P(H(t) < tau) ~= 0.63 at Otsu tau = 2.10.

### Stage 2: Routing Head Selection (only for H(t) >= tau)

For each adapter j in {1, ..., N}, compute routing score:

    f_j(h_t) = sigmoid(MLP_j(mean_pool(h)))

where MLP_j: R^d -> R^1 is a 2-layer MLP (d -> 32 -> 1).

Select top-k adapters by score:

    S(t) = argtop_k({f_j(h_t) : j = 1, ..., N})

### Stage 3: Pre-Merge Composition

For selected adapters S(t) = {j_1, j_2}, compose with score-proportional weights:

    w_{j_i} = f_{j_i}(h_t) / sum_{j in S(t)} f_j(h_t)

    A_composed = sum_{j in S(t)} w_j * A_j

Apply A_composed to produce output for token t.

## Computational Cost Analysis

### Without unified pipeline (always-compose baseline)
Per token:
- 1 forward pass with composed adapter: C_fwd

### With unified pipeline
Per token:
- 1 base forward pass (always needed for entropy): C_fwd
- If H(t) >= tau (probability 1 - f_skip = 0.37):
  - Mean-pool hidden states: O(d * seq_len) -- negligible
  - N routing head evaluations: N * C_head (N * ~0.17ms)
  - Adapter composition: O(N_params * k) -- negligible
  - 1 composed forward pass: C_fwd

Expected cost per token:

    E[C_unified] = C_fwd + (1 - f_skip) * (N * C_head + C_fwd)
                 = C_fwd * (1 + (1 - f_skip)) + (1 - f_skip) * N * C_head
                 = C_fwd * 1.37 + 0.37 * N * C_head

### Overhead vs always-compose

The always-compose baseline is a single forward pass with pre-merged adapters:

    C_baseline = C_fwd (composed)

The unified pipeline overhead is:

    overhead = E[C_unified] / C_baseline - 1

The key question is whether the entropy computation and conditional second
pass cost less than the quality gain from selective routing.

NOTE: The overhead CANNOT be less than 0% because we always need the base
forward pass for entropy computation. The value proposition is NOT speed --
it is quality: the unified pipeline achieves routed-quality PPL (6.42) on
the 37% of uncertain tokens while using cheap base-only output for the 63%
of confident tokens.

## PPL Decomposition

The unified pipeline PPL decomposes as:

    L_unified = f_skip * L_base + (1 - f_skip) * L_routed

where:
- L_base = per-token cross-entropy of base model (no adapter)
- L_routed = per-token cross-entropy of head-routed top-2 composition

PPL_unified = exp(L_unified)

From proven results:
- L_base tokens (H < tau) are the easy/confident tokens where base model already
  does well, so L_base is low for these tokens specifically
- L_routed is near-oracle quality (6.42 vs 6.41 oracle PPL)

The hypothesis: PPL_unified < 6.42 (best individual method) because:
1. Confident tokens (63%) get base output -- which is already optimal for them
2. Uncertain tokens (37%) get routed top-2 -- which is near-oracle

## Worked Example (d=2560, N=5, k=2)

Given:
- Token with H(t) = 3.5 (> tau = 2.10, uncertain)
- Head scores: python=0.92, math=0.85, medical=0.12, legal=0.05, creative=0.08
- Top-2: {python, math}
- Weights: w_python = 0.92/(0.92+0.85) = 0.519, w_math = 0.85/1.77 = 0.481
- Composed adapter: A = 0.519 * A_python + 0.481 * A_math

Given:
- Token with H(t) = 0.8 (< tau = 2.10, confident)
- Output: use base model prediction directly, skip routing heads entirely

## Assumptions

1. Entropy from the base forward pass is a reliable confidence signal
   (PROVEN: CV=0.87, Otsu eta=0.68)
2. Low-entropy tokens do not benefit from adapter composition
   (PROVEN: 1.13% PPL cost at 63% skip rate)
3. Routing heads remain accurate when applied only to uncertain tokens
   (UNTESTED: heads were trained on all tokens, evaluated here on subset)
4. Pre-merge composition is the right paradigm for selected adapters
   (PROVEN: 0.80% overhead, N-independent latency)
5. The two-pass architecture (base for entropy, composed for uncertain tokens)
   is worth the extra compute for quality improvement
   (TESTABLE: this experiment)
