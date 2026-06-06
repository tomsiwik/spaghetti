# Math: Composition Interpolation Landscape

## Mechanism Definition

Given N trained LoRA adapters, each producing a weight perturbation:

    delta_W_i = (1/s) * B_i @ A_i^T,  where A_i in R^{d x r}, B_i in R^{r x d_out}

The composed perturbation under weights w = (w_1, ..., w_N) is:

    delta_W(w) = sum_{i=1}^{N} w_i * delta_W_i

For a 2-adapter sweep with parameter alpha in [0, 1]:

    delta_W(alpha) = alpha * delta_W_A + (1 - alpha) * delta_W_B

The loss landscape L(alpha) = PPL(W_base + delta_W(alpha)) maps the 1D interpolation path.

For a 3-adapter simplex with w1 + w2 + w3 = 1, wi >= 0:

    delta_W(w) = w1 * delta_W_1 + w2 * delta_W_2 + w3 * delta_W_3

This maps a 2D simplex to the scalar PPL surface.

## Why This Matters

**Convexity determines optimization feasibility.** If L(w) is convex in the weight
simplex, then gradient-based methods (or even grid search) reliably find the optimum.
If L(w) is chaotic (non-smooth, many local minima), then optimal composition weights
require expensive discrete search.

**Prior art:**
- LoRA Soups (arXiv 2410.13025): CAT composition averages task vectors, implicitly
  assuming the landscape is smooth enough for uniform weighting to work.
- Naive LoRA Summation (arXiv 2508.11985): Orthogonal A-matrices enable additive
  composition. Their theory predicts interference-free addition, which implies
  near-linearity of PPL in the weight simplex (each adapter's contribution is
  independent).
- Model soups (Wortsman et al., ICML 2022): Showed that averaging fine-tuned
  models in weight space often finds a point with better accuracy than any
  individual model, implying a convex loss basin in the interpolation space.

## What We Expect (Predictions from Theory)

### Prediction 1: Near-convexity for in-domain eval

For adapter A evaluated on domain A's data, the contribution of adapter A is
dominant. As alpha increases from 0 to 1 (adding more of A), PPL on domain A
should decrease monotonically. The landscape should be roughly convex:

    d^2 L_A / d alpha^2 >= 0  (approximately)

### Prediction 2: Trade-off curve for cross-domain eval

On mixed data (50/50 domain A + domain B), the optimal alpha should be interior
(not at endpoints), reflecting a genuine trade-off. If adapters are orthogonal
(as Grassmannian guarantees), the optimal point should be near alpha = 0.5.

### Prediction 3: Smooth simplex for 3 adapters

With orthogonal A-matrices (|cos| = 0.004), the 3-adapter simplex should have
a single smooth basin. The Hessian should be approximately positive semi-definite
(convex). The Lipschitz constant of the gradient should be small.

## What Would Break It

**K1 (Flat landscape):** If all compositions give the same PPL (within noise),
then the weights carry no signal and routing/optimization is pointless for
quality. This would mean delta_W_i perturbations are too small relative to
W_base to matter — contradicting our -29.1% composition benefit at N=24.

**K2 (Chaotic landscape):** If PPL varies non-monotonically with small weight
changes (e.g., flipping between high and low PPL at adjacent alpha values),
then the loss surface has many local minima and gradient methods fail. This
would require |cos(delta_W_i, delta_W_j)| to be large (interference), which
contradicts our measured |cos| = 0.00125.

## Smoothness Metrics

**Lipschitz constant of the gradient:**

    L_grad = max_{i,j adjacent} |dPPL/dalpha(alpha_i) - dPPL/dalpha(alpha_j)| / |alpha_i - alpha_j|

Small L_grad implies smooth, well-conditioned optimization.

**Convexity ratio:** For three consecutive points alpha_{i-1}, alpha_i, alpha_{i+1}:

    C_i = (L(alpha_i) - 0.5 * (L(alpha_{i-1}) + L(alpha_{i+1}))) / L(alpha_i)

C_i < 0 implies local convexity (midpoint below chord). Fraction of C_i < 0
measures global convexity.

**Hessian positive semi-definiteness (3-adapter):**

Numerical Hessian via finite differences on the simplex. Eigenvalues of the
2x2 Hessian (in barycentric coordinates) should be non-negative for convexity.

## Complexity Analysis

Each PPL evaluation: O(S * L * d^2) where S = sequence length, L = 30 layers,
d = 2560. With max_batches = 10 samples at seq_len = 256:

- Phase 1: 21 alpha values x 3 pairs x 3 eval sets = 189 evals (but model
  reloaded per pair, not per alpha — composition is just weight arithmetic)
- Phase 2: ~50 simplex points x 3 eval sets = 150 evals
- Total: ~339 forward passes, each ~0.3s = ~100s of pure compute

Main cost: model loading + BitLinear unpacking (~15s per load). With 3+1 loads
(3 pairs + 1 triple), overhead is ~60s. Total estimated: ~4 minutes.

## Connection to Architecture

This experiment tests whether the Grassmannian orthogonality guarantee
(||delta_W_i^T delta_W_j|| -> 0) translates to smooth, convex loss landscapes
in weight space. If confirmed:

1. **Optimal composition weights are gradient-findable** — enables learned
   routing weights beyond top-1 selection
2. **Uniform 1/N is a reasonable default** — the landscape is smooth enough
   that uniform weighting isn't far from optimal
3. **Per-token soft routing is feasible** — continuous weights on the simplex
   produce smooth quality variation, enabling gradient-based router training

Production reference: DeepSeek-V3 uses auxiliary-loss-free load balancing with
continuous expert weights. Our landscape analysis tests whether continuous
weights are well-behaved for Grassmannian LoRA experts.
