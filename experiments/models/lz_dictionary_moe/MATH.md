# LZ Dictionary MoE: Mathematical Foundations

## Notation

| Symbol | Shape/Type | Definition |
|--------|-----------|------------|
| d | scalar | Embedding dimension |
| N | scalar | Number of experts |
| D | scalar | Number of dictionary entries (codebook size) |
| r | scalar | Rank of dictionary entries |
| r_delta | scalar | Rank of per-expert residual |
| k | scalar | Top-k experts selected per token |
| x | (B, T, d) | Token hidden states |
| W^down_j | (r, d) | j-th dictionary entry down-projection |
| W^up_j | (d, r) | j-th dictionary entry up-projection |
| alpha_i | (D,) | Expert i's dictionary composition coefficients |
| Delta^down_i | (r_delta, d) | Expert i's residual down-projection |
| Delta^up_i | (d, r_delta) | Expert i's residual up-projection |

## Expert Decomposition

### Standard MoE Expert
Each independent expert is an MLP:

    expert_i(x) = W2_i * ReLU(W1_i * x)

where W1_i in R^{4d x d}, W2_i in R^{d x 4d}. Parameters per expert: 8d^2.

### Dictionary-Composed Expert
Each expert is decomposed into shared dictionary references + unique residual:

    expert_i(x) = sum_{j=1}^{D} alpha_{i,j} * dict_j(x) + delta_i(x)

where:

    dict_j(x) = W^up_j * ReLU(W^down_j * x)       (shared sub-module)
    delta_i(x) = Delta^up_i * ReLU(Delta^down_i * x)  (per-expert residual)
    alpha_i = softmax(logits_i)                       (composition weights)

This is analogous to LZ77 compression:
- Dictionary entries = "previously seen patterns" (shared across experts)
- alpha coefficients = "pointers into the dictionary"
- delta residual = "literal bytes" (unique to each expert)

## Composition Weights

The alpha coefficients are produced by softmax over learnable logits:

    alpha_{i,j} = exp(l_{i,j}) / sum_{j'} exp(l_{i,j'})

Properties:
- sum_j alpha_{i,j} = 1 for each expert i (convex combination)
- All alpha_{i,j} > 0 (full dictionary access, soft selection)
- Gradient: d_alpha / d_l = diag(alpha) - alpha * alpha^T (standard softmax Jacobian)

## Parameter Count

### Standard MoE (per layer)
- Router: d * N
- Experts: N * (d * 4d + 4d * d) = 8Nd^2
- **Total: dN + 8Nd^2**

### Dictionary MoE (per layer)
- Router: d * N
- Dictionary: D * (d * r + r * d) = 2Ddr
- Expert alphas: N * D
- Expert deltas: N * (d * r_delta + r_delta * d) = 2Nd * r_delta
- **Total: dN + 2Ddr + ND + 2Nd * r_delta**

### Savings Analysis

Ratio (ignoring router, small terms):

    R = (2Ddr + 2Nd * r_delta) / (8Nd^2)
      = (Dr + Nr_delta) / (4Nd^2) * d * 2
      = Dr/(4Nd) + r_delta/(4d)

**Worked example at micro scale (d=64, N=4):**

Config 1 (small): D=8, r=32, r_delta=16
- Standard: 4 * 8 * 64^2 = 131,072 params (MLP only)
- Dictionary: 2 * 8 * 64 * 32 + 4 * 8 + 2 * 4 * 64 * 16
            = 32,768 + 32 + 8,192 = 40,992 params
- Ratio: 31.3% (68.7% savings in MLP params)

Config 2 (large): D=8, r=64, r_delta=48
- Standard: 131,072 params (MLP only)
- Dictionary: 2 * 8 * 64 * 64 + 32 + 2 * 4 * 64 * 48
            = 65,536 + 32 + 24,576 = 90,144 params
- Ratio: 68.8% (31.2% savings in MLP params)

## Effective Rank per Expert

Each dictionary-composed expert has effective weight matrices:

    W1_eff_i = sum_j alpha_{i,j} * [W^up_j; 0] @ [W^down_j; 0] + [Delta^up_i; 0] @ [Delta^down_i; 0]

The rank of each term:
- Each dict term: rank r (reduced from 4d)
- Delta term: rank r_delta
- Sum of D rank-r matrices: rank <= D*r (but likely lower due to shared structure)
- Plus delta: rank <= D*r + r_delta

With D=8, r=32, r_delta=16: effective rank <= 272 (vs 256=4d for standard)

The key insight: the dictionary SHARES rank capacity across experts.
If experts overlap in function space (as behavioral_dedup found 19.3%
redundancy at Layer 0), sharing is more efficient than independent allocation.

## Routing Layer

Standard top-k softmax routing (identical to MoEGPT):

    scores = x @ W_router               (B, T, N)
    probs = softmax(scores, axis=-1)     (B, T, N)
    masked = probs * top_k_mask(scores)  (B, T, N) with k nonzero
    weights = masked / sum(masked)       (B, T, N) renormalized

    output = sum_i weights_i * expert_i(x, dictionary)

## Load Balancing Loss

Same as standard MoE:

    L_balance = N * sum_i (mean_prob_i)^2

where mean_prob_i = (1/BT) sum_{b,t} probs_{b,t,i}.
Minimized at uniform 1/N distribution.

## Dictionary Utilization Metric

For diagnostic purposes, define utilization as:

    alpha_mean_j = (1/N) sum_i alpha_{i,j}    (mean weight of entry j across experts)
    utilized(j) = 1 if alpha_mean_j > 1/(2D)  (above half of uniform)
    utilization_rate = (1/D) sum_j utilized(j)

Kill criterion: utilization_rate < 0.30 means most dictionary entries are unused,
indicating the codebook is not capturing shared structure.

## Assumptions

1. **Low-rank structure exists in expert MLPs**: Expert weight matrices contain
   shared low-rank components. Supported by behavioral_dedup finding 19.3%
   redundancy at Layer 0 and by Procrustes decomposition finding 54% shared
   fraction of fine-tuning deltas.

2. **Soft composition is sufficient**: Using softmax alpha (all entries contribute)
   rather than hard selection (only some entries). This avoids the top-1 phase
   transition observed in sparse_router experiment. Hard selection is deferred
   to macro scale.

3. **Per-expert residual captures unique behavior**: The delta term must be
   expressive enough to capture what the dictionary misses. At r_delta=d/4,
   this provides 25% of the information capacity of the dictionary.

4. **Dictionary entries specialize during training**: With sufficient training
   signal, different dictionary entries should learn different sub-functions.
   If all entries remain uniform (H_norm -> 1.0), specialization has failed
   but the model may still work via ensemble effect.
