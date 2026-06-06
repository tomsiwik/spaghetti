# Soft-Routing Obstruction Collapse: Proof Verification Report

## Theorem
(Theorem 1 from MATH.md) If, for a given token x, the Gumbel-sigmoid router
activates K(x) >= 3 adapters (gates > 0.5), then the effective composition cover
has H^1 = 0 (all topological obstructions from Finding #242 are eliminated).

## Predictions vs Measurements

| Prediction (from proof) | Measured | Match? |
|------------------------|----------|--------|
| E[K] >= 2.5 at neutral logits | 0.93 (hard), 0.95 (stochastic) | NO |
| E[K] >= 3.0 with learned routing | 0.93 | NO |
| Fraction K>=3 > 0.50 | 0.000 | NO |
| PPL(k=3) / PPL(k=2) <= 1.05 | 1.0000 | YES (trivially -- all regimes identical) |
| H^1 of routing cover = 0 | 0 (trivially -- no overlaps) | VACUOUS |

## Hypothesis
Gumbel-sigmoid routing on 5-domain composition activates >=3 adapters on >50% of tokens.

**REFUTED.** The router activates exactly 1 adapter per token (mean K = 0.93).
The H^1 = 3 topological obstructions from Finding #242 persist under learned routing.

## What This Experiment Is

This experiment tested whether Gumbel-sigmoid routing, trained to select domain adapters
for 5-domain BitNet-2B-4T composition, naturally activates enough adapters per token
to place the system in the obstruction-free regime (H^1 = 0).

**Method:**
1. Extracted hidden states from layer 15 of BitNet-2B-4T (250 samples, 5 domains)
2. Trained a Gumbel-sigmoid router (2-layer MLP: 2560->128->5, sigmoid gates) with BCE
   loss against soft domain labels (label smoothing = 0.05), 300 steps, lr=3e-4
3. Analyzed activation statistics: hard (deterministic) and stochastic (10 Gumbel runs)
4. Compared PPL under natural routing, forced k=2, forced k=3, and equal-weight-all
5. Computed Cech nerve of the routing cover

## Key References
- Finding #242: Cech nerve H^1=3 at k=2, H^1=0 at k=3 (sheaf cohomology dimension)
- Finding #185: Energy gap routing 88% accuracy, sigmoid routing 44% better than softmax
- Jang et al. 2017: Categorical reparameterization with Gumbel-softmax (ICLR)

## Empirical Results

### Activation Statistics (Phase 2)
- **Hard (deterministic):** Mean K = 0.93, std = 0.25
  - K=0: 6.8%, K=1: 93.2%, K>=2: 0.0%, K>=3: 0.0%
- **Stochastic (Gumbel, tau=1.0):** Mean K = 0.95, frac K>=3: 0.0%
- **Gate logits:** mean = -3.20, std = 2.20, frac positive = 18.6%
- **Gate values (sigmoid):** mean per adapter = 0.15-0.16
- Per-domain: medical=1.00, code=0.98, math=1.00, legal=0.92, finance=0.76

### PPL Comparison (Phase 3, 50 samples)
| Regime | Mean PPL | Median PPL |
|--------|----------|------------|
| Natural Gumbel-sigmoid | 10.50 | 3.90 |
| Forced k=2 | 10.50 | 3.90 |
| Forced k=3 | 10.50 | 3.90 |
| Equal-weight all | 9.99 | 4.90 |

PPL ratio (k=3 / k=2) = 1.0000. This is vacuous: since all gates for non-primary
adapters are near zero (sigmoid(-3.2) = 0.039), forcing top-2 or top-3 merges
adapters with negligible weight (0.039 * scale), which has no measurable effect.

### Cech Nerve of Routing Cover (Phase 4)
- Cover sizes: medical=50, code=49, math=50, legal=46, finance=44
  (each adapter's cover contains only its own domain's samples)
- Nerve: 5 vertices, 0 edges, 0 triangles (totally disconnected)
- beta_0 = 5 (5 components), beta_1 = 0 (trivially, no cycles because no edges)

### Kill Criteria
- **K1 (#650): FAIL.** Mean activation = 0.93 < 2.5. Routing is extremely sparse.
- **K2 (#651): PASS (vacuous).** PPL ratio = 1.0 <= 1.05. But this is because
  all regimes are effectively identical (only 1 adapter active).

## Analysis: Why K = 1

The mathematical reason is straightforward. With BCE loss and soft domain labels
(primary = 0.80, others = 0.05), the optimal logit for the primary adapter is
logit(0.80) = 1.39 and for non-primary adapters is logit(0.05) = -2.94. At these
logits, sigmoid gives 0.80 and 0.05 respectively. The threshold at 0.5 requires
logit > 0, which is satisfied only for the primary adapter.

This is not a bug in the router -- it is the mathematical consequence of training
a Gumbel-sigmoid router for domain selection. **Domain selection is inherently
sparse.** To activate K >= 3, the routing objective must explicitly reward
multi-adapter activation (e.g., auxiliary diversity loss, or composition-aware loss
that measures quality of the composed output).

The worked example in MATH.md (Section F) predicted E[K] = 2.2 for logits
[1.0, 0.8, 0.5, -0.3, -1.0], but the actual learned logits are much sparser:
the primary adapter gets logit ~ +1.4 and all others get ~ -3.0.

## Implications for the Composition Architecture

1. **Bridge adapters ARE needed.** The routing system does not naturally place
   composition in the obstruction-free regime. The H^1 = 3 obstructions from
   Finding #242 persist at inference time.

2. **Obstruction collapse requires architectural intervention.** Possibilities:
   - Force minimum activation count (k >= 3 hard constraint in the router)
   - Add diversity regularizer (e.g., min-entropy loss on gate distribution)
   - Train router with composition-aware loss (optimize PPL of composed output,
     not domain classification accuracy)
   - Use top-k selection with k >= 3 as a hard architectural choice

3. **The phase transition at k=3 is architecturally actionable.** Since
   Finding #242 proved H^1 collapses at exactly k=3, the simplest fix is
   to architecturally enforce top-3 routing (always activate the 3 highest-gate
   adapters). This adds no hyperparameters beyond the threshold k=3, which is
   derived from the nerve topology.

## Limitations
- Router trained with domain-classification BCE, not composition-aware loss.
  A router trained to optimize composed-output quality might naturally activate
  more adapters. This is a different experiment.
- 250 samples (50/domain), single seed. Router training may be underfitting.
- Label smoothing (0.05) was chosen ad hoc; different smoothing could change K.
- PPL comparison vacuous because gate values for non-primary adapters are negligible.

## What Would Kill This
- If a composition-aware router (trained on PPL of composed output) also converges
  to K=1, that would strongly confirm that sparse activation is fundamental to
  learned routing, not just an artifact of BCE training.
- If forced k=3 routing with proper weights shows PPL degradation >5%, that would
  kill the "obstruction collapse helps quality" hypothesis.

## Total Runtime
128.8 seconds on Apple M5 Pro 48GB.
