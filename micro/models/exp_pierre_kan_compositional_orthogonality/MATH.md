# MATH.md — KAN compositional orthogonality (does additive composition produce zero cross-contribution?)

## Hypothesis

KAN adapters compose by per-edge spline-coefficient addition:
`c_merged_ij = Σ_k c_k_ij`. This is *parameter-additive* — but the
**output** at a node is `Σ_i ϕ_ij(z_i)`, which means two adapters whose
splines have overlapping input-space support still produce *non-trivially
combined* contributions for shared inputs.

The structural guarantee we want is **support-disjoint composition**:
if adapter `k` is only active in input region `R_k`, and `R_k ∩ R_j = ∅`
for `k ≠ j`, then the composition `Σ_k ϕ_k` reduces at any input `x` to
exactly the single `ϕ_k` whose support contains `x`.

> **For Pierre's existing 7 PoLAR adapters, what is the input-space
> overlap of their effective spline supports? If we warm-start as KAN,
> does composition naturally have low cross-contribution, or do we need
> disjoint-support regularization (separate experiment)?**

## Why this matters

This is the experiment that validates (or falsifies) the Lagrangian
framing. Composition-as-addition is mathematically trivial; the
*useful* property is composition-as-superposition-without-interference,
which holds iff supports are disjoint.

If existing PoLAR adapters already have low input-space overlap when
their B-matrices are interpreted as KAN skip-weights → composition is
naturally clean, no special training needed.

If they have high overlap → a future experiment must add a
disjoint-support regularizer during training (e.g., contrastive loss
between activations on each adapter's native domain).

## Procedure

For each of the 7 PoLAR adapters:

1. **Probe activations on its native task.** Run the adapter alone on a
   small set of in-domain prompts (e.g., math adapter on GSM8K). For
   each layer, record the rank-r intermediate `z = x @ A` distribution
   across all token positions.

2. **Estimate spline support.** For each `z` distribution, compute the
   convex hull (or 5th–95th percentile range) of values reached. This
   is the input range where the spline ϕ_ij is "active" for that adapter.

3. **Pairwise support overlap matrix.** For each pair (k, j) and each
   layer, compute the overlap `|R_k ∩ R_j| / |R_k ∪ R_j|` (Jaccard).
   Aggregate to a single per-pair score.

4. **Composition cross-contribution test.** For each adapter k on its
   native task:
   - Compute output with adapter k alone: `y_k(x)`
   - Compute output with composition (k + others): `y_compose(x)`
   - Cross-contribution: `‖y_compose - y_k‖ / ‖y_k‖`
   - Low cross-contribution (≤5%) confirms support-disjoint behavior.

5. **Behavioral validation.** Composed performance should be ≥ best-single
   on each native benchmark with low cross-contribution. If
   cross-contribution is low BUT composed performance drops, supports
   are disjoint but the composition operator is wrong — that would
   falsify the Lagrangian framing.

## Pre-registered Kill Criteria

- **K1 (SUPPORT OVERLAP)** Mean pairwise Jaccard overlap of input ranges
  across the 21 (7 choose 2) adapter pairs ≤ 0.4.
  PASS → existing adapters are naturally low-overlap; KAN composition
  is interference-free for free.

- **K2 (CROSS-CONTRIBUTION)** Mean cross-contribution `‖y_compose - y_k‖
  / ‖y_k‖` across the 7 native-task evaluations ≤ 0.10.
  PASS → adding other adapters perturbs each adapter's output by ≤10%
  on its own task. Lagrangian superposition holds approximately.

- **K3 (BEHAVIORAL)** Composed accuracy on each native benchmark
  (GSM8K, HumanEval, MedQA) within 3pp of single-best, OR equal to
  TIES-B baseline (71.3% avg).
  PASS → support-disjoint composition is at least as good as TIES-B
  while being structurally cleaner.

- **K4 (FALSIFICATION)** Three-way reading:
  - K1 ✓ K2 ✓ K3 ✓ → **SUPPORTED**: Lagrangian framing works for free.
  - K1 ✓ K2 ✓ K3 ✗ → **PARTIAL**: supports disjoint but composition
    operator is wrong. Investigate alternatives to plain coefficient sum.
  - K1 ✗ K2 * K3 * → **REGULARIZER NEEDED**: existing supports overlap.
    Spec follow-up experiment with disjoint-support regularization.
  - K1 ✓ K2 ✗ K3 * → **CONTRADICTION**: low overlap should imply low
    cross-contribution. Likely measurement bug — investigate.

## What this experiment is NOT

- Not a re-baseline of TIES-B (that's already 71.3% avg).
- Not a replacement for the parent KAN experiment — that one tests
  expressivity (Q1) and pure-KAN composition (Q2). This one tests the
  structural orthogonality property.
- Not a training experiment. Uses warm-start from existing B-matrices.
  If support overlap is high (K1 fail), the next experiment introduces
  training-time regularization.

## Honest gaps

- **Spline support is approximated by activation range.** A spline can
  technically be non-zero across its full grid even if activations only
  cover part of the range. We're measuring "where the spline contributes
  meaningfully," not "where ϕ is mathematically non-zero." For
  warm-started KANs with the skip-weight path equal to B, the
  contribution at unobserved inputs is determined by the linear path,
  not the spline coefficients.

- **Cross-contribution is measured at the layer output, not behavioral.**
  K2 catches "supports overlap less than they look" via output
  perturbation. K3 catches "composition is mathematically clean but
  empirically broken" via task accuracy.

- **Stiefel-KAN hybrid (sibling experiment)** is the architecture that
  *guarantees* support disjointness by construction. This experiment
  tests whether that guarantee is even necessary, or if existing
  adapters give it to us for free.

## References

- Parent: `exp_pierre_kan_adapter_lagrangian` (P1) — must complete first
  to confirm KAN parameterization preserves PoLAR baseline.
- Sibling: `exp_pierre_stiefel_kan_hybrid` (P2) — adds Stiefel constraint
  if K1 fails here.
- Lagrangian framing: superposition principle for adapter potentials.
- Kolmogorov-Arnold: arxiv 2404.19756.
- Prior: TIES-B at 71.3% avg (target floor).
