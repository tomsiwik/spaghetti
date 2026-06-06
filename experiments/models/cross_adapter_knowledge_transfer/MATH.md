# Cross-Adapter Knowledge Transfer: Mathematical Framework

## Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| W_base | Frozen base model weights | (d_out, d_in) |
| A_i | LoRA down-projection for adapter i | (d_in, r) |
| B_i | LoRA up-projection for adapter i | (r, d_out) |
| Delta_i = B_i A_i | Rank-r weight update for domain i | (d_out, d_in) |
| alpha | Foreign adapter weight in composition | [0, 1] |
| PPL_i(D_j) | Perplexity of adapter i evaluated on domain j data | R+ |
| T(i,j) | Transfer coefficient: how much adapter i helps domain j | [-100, 100]% |

## Transfer Coefficient

For a pair (foreign adapter i, native domain j), define:

**Composed model weights:**
W_composed = W_base + alpha * Delta_i + (1 - alpha) * Delta_j

**Native-only weights:**
W_native = W_base + Delta_j

**Transfer coefficient (for best alpha):**

T(i, j) = (PPL_j(W_native) - PPL_j(W_composed*)) / PPL_j(W_native) * 100

where W_composed* uses the alpha that minimizes PPL_j.

- T(i,j) > 0: Constructive transfer (foreign adapter helps)
- T(i,j) < 0: Destructive interference (foreign adapter hurts)
- T(i,j) ~ 0: No transfer

## Why Constructive Transfer Can Exist

Given Grassmannian-initialized A matrices with near-orthogonality (|cos| ~ 0.001),
the composed update acts on nearly disjoint subspaces:

Delta_i + Delta_j = B_i A_i + B_j A_j

Since A_i^T A_j ~ 0, the updates don't destructively interfere. But they CAN
constructively help because:

1. **Shared linguistic features:** Math and code both use structured reasoning.
   The math adapter may learn reasoning patterns that improve code generation
   even though the weight subspaces are orthogonal.

2. **Regularization effect:** Adding a small foreign adapter (alpha << 1)
   acts as implicit regularization, similar to dropout or weight decay.
   This can reduce overfitting to native domain.

3. **Complementary representations:** The base model has latent capacity that
   a single adapter cannot fully exploit. A foreign adapter may activate
   useful features that the native adapter's B matrix doesn't reach.

## Transfer Matrix Structure

The full N x N transfer matrix T has structure if:
1. Related domains show higher mutual transfer (e.g., math <-> code)
2. The matrix is approximately low-rank (few underlying transfer factors)
3. Row means vary (some adapters are better "donors")
4. Column means vary (some domains benefit more from foreign knowledge)

**Kill criterion K2:** T is random if:
- Matrix variance < 0.5 (all values near zero, no signal)
- AND structure gap (expected-high avg - expected-low avg) < 1.0%
- AND symmetry correlation < 0.3

## Expected Domain Relationships

Based on semantic similarity:
- **High transfer:** math <-> python (computational), medical <-> legal (professional)
- **Low transfer:** creative <-> legal (opposite registers), creative <-> math

## Worked Example (micro scale)

With 5 domains and 4 alpha values:
- 5 x 4 = 20 ordered pairs (excluding self)
- 4 alpha values per pair = 80 PPL evaluations
- Plus 5 base + 5 individual = 90 total evaluations
- At ~2s per eval: ~3 minutes for the transfer matrix phase

## Computational Cost

| Phase | Operations | Estimated Time |
|-------|-----------|---------------|
| Model load + unpack | 1x | ~5s |
| Training (5 adapters x 200 steps) | 1000 fwd+bwd | ~10 min |
| Base PPL (5 domains) | 125 fwd | ~10s |
| Individual PPL (5 domains) | 125 fwd | ~10s |
| Transfer matrix (20 pairs x 4 alphas) | 2000 fwd | ~3 min |
| **Total** | | **~15 min** |
