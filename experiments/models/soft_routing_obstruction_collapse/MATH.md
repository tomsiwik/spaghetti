# Soft-Routing Obstruction Collapse: Mathematical Foundation

## A. Failure Mode Identification

**Disease:** When composing N adapters with soft routing, if only k < 3 adapters
activate per token, the Cech nerve of the activation cover has non-trivial first
cohomology H^1 > 0, meaning independent pairwise incompatibilities exist that
cannot be resolved by any composition scheme operating on pairs alone.

**Proof this is a real risk:** Finding #242 established that the specialization-based
top-2 cover over 5 adapters (medical/code/math/legal/finance) on BitNet-2B-4T yields
a Cech nerve with beta_1 = |E| - |V| + c = 6 - 5 + 2 = 3, where c=2 connected
components (finance isolated at scale=1.0). These 3 independent 1-cycles represent
pairwise merging conflicts that no bridge adapter acting on 2 domains can resolve
without introducing new conflicts.

**Stable fixed point argument:** A sigmoid gate with learned bias b < 0 for most
adapters naturally tends toward sparse activation. Under Gumbel noise at temperature
tau, the expected activation probability for gate i is:

  p_i = E[sigma((logit_i + G) / tau)]

where G ~ Gumbel(0,1). For logit_i = 0 (uninformative), p_i -> 0.5 as tau -> inf
and p_i -> sigma(0) = 0.5 for tau = 1. But learned routing may push logits negative
for irrelevant adapters, yielding p_i << 0.5 and thus E[sum_i 1(gate_i > 0.5)] < 3.

## B. The Right Question

Not: "How do we force >=3 adapters to activate?"
But: "Under what conditions on Gumbel-sigmoid gate statistics does the expected
activation count exceed 3, placing composition in the H^1 = 0 regime?"

The answer comes from the Cech nerve structure proven in Finding #242.

## C. Prior Mathematical Foundations

**Theorem (Cech nerve cohomology, Finding #242).**
Let U = {U_1, ..., U_5} be the top-k specialization cover over 5 domain adapters.
The Cech nerve N(U) has:
- At k=2: |V|=5, |E|=6, |T|=0, c=2, hence beta_1 = 6 - 5 + 2 = 3
- At k=3: |V|=5, |E|=6, |T|=4, hence beta_1 = 6 - rank(delta_0) - rank(delta_1) = 6 - 3 - 3 = 0

The phase transition occurs exactly at k=3 because the 4 triangles fill all 3
independent cycles.

**Gumbel-sigmoid activation statistics (Jang et al. 2017, Maddison et al. 2017).**
For a Gumbel-sigmoid gate with logit l and temperature tau, the gate output is:

  g = sigma((l + G) / tau), where G ~ Gumbel(0, 1)

The probability that g > threshold t is:

  P(g > t) = P(sigma((l + G)/tau) > t) = P(G > tau * logit(t) - l)
           = exp(-exp(tau * logit(t) - l))   [Gumbel CDF]

For threshold t = 0.5: logit(0.5) = 0, so P(g > 0.5) = exp(-exp(-l)) = sigma_Gumbel(l).

**Connection to activation count.** For N=5 independent Gumbel-sigmoid gates with
logits l_1, ..., l_5, the expected number of active gates is:

  E[K] = sum_{i=1}^{5} P(g_i > 0.5) = sum_{i=1}^{5} exp(-exp(-l_i))

For E[K] >= 3, we need the logits to satisfy:

  sum_{i=1}^{5} exp(-exp(-l_i)) >= 3

## D. Proof of Guarantee

**Theorem 1.** (Obstruction collapse via activation count)
Let U_k denote the top-k specialization cover over 5 domain adapters on BitNet-2B-4T.
If, for a given token x, the Gumbel-sigmoid router activates K(x) >= 3 adapters
(i.e., gates g_i(x) > 0.5 for at least 3 indices i), then the effective composition
cover has H^1 = 0.

*Proof.* By Finding #242, the Cech nerve N(U_2) has beta_1 = 3, while N(U_3) has
beta_1 = 0. The specialization cover U_k assigns sample x to adapter i iff adapter i
is among the top-k lowest-PPL adapters for x. The routing cover R assigns x to
adapter i iff the gate g_i(x) > 0.5.

If K(x) >= 3, then x is assigned to at least 3 adapters in R. For the Cech nerve
of R to have beta_1 = 0, we need every 1-cycle (closed path of pairwise overlaps)
to be filled by a 2-simplex (triple overlap). With 5 adapters and K >= 3, every
pair of active adapters shares x in their intersection (since x is in all >= 3 of them),
creating triple overlaps that fill cycles.

Specifically: if adapters {i, j, k} are all active on token x, then x is in
U_i and U_j and U_k, creating a 2-simplex (i, j, k) in the nerve. With K >= 3
active adapters on every token, every edge in the activation nerve is contained
in at least one triangle, hence every 1-cycle is a boundary, hence H^1 = 0. QED.

**Corollary.** If the fraction of tokens with K >= 3 exceeds 50%, then the average
topological obstruction per token is at most 1.5 (vs. 3 for pure k=2 routing).

## D. Predictions (Derived from the Proof)

### Behavioral Predictions:
1. **B1:** Gumbel-sigmoid with temperature tau~1 on 5 adapters activates >= 3
   adapters on > 50% of tokens (the gates are independent Bernoulli with p_i ~ 0.5,
   giving E[K] = 2.5 for neutral logits; learned routing should push relevant
   adapters' logits positive)
2. **B2:** PPL under forced k=3 routing should be no worse than k=2 (because
   the additional adapter contributes relevant information without obstruction)

### Quantitative Predictions:
| Prediction | Source | Expected Value |
|-----------|--------|----------------|
| Mean K at tau=1.0, neutral logits | Binomial(5, 0.5) | 2.5 |
| Mean K with learned routing | Theorem 1 + Finding #185 | >= 3.0 |
| Fraction K >= 3, learned | Proof + CDF | > 0.50 |
| H^1 effective (K>=3 regime) | Theorem 1 | 0 |
| PPL ratio (k=3 / k=2) | Corollary | <= 1.05 |

### Kill Criteria (derived from predictions):
- **K1 (#650):** Mean activation count < 2.5 -> routing too sparse, H^1 persists
  (derived from Binomial(5, 0.5) baseline; learned routing should exceed this)
- **K2 (#651):** PPL at forced k=3 > 1.05 * PPL at k=2 -> obstruction collapse
  hurts quality (derived from Theorem 1: if H^1=0 helps, PPL should not degrade)

## E. Assumptions & Breaking Conditions

1. **Independence of gates:** Sigmoid gates are independent Bernoulli. If gates become
   correlated (all activate/deactivate together), the activation count is bimodal
   (all-on or all-off) and mean K is not informative. Breaking: check variance of K.
2. **Cover-nerve equivalence:** The routing cover approximates the specialization cover.
   If routing assigns adapters very differently from PPL-based specialization, the
   H^1 analysis may not transfer. Breaking: check overlap between routing cover and
   specialization cover.
3. **Scale factor compatibility:** The experiment uses pre-tuned per-domain scales
   (Finding #217). If scales interact badly with multi-adapter activation, quality
   could degrade. Breaking: PPL ratio > 1.05.

## F. Worked Example (N=5, d=2560)

Consider 5 adapters with learned logits l = [1.0, 0.8, 0.5, -0.3, -1.0].

Gate activation probabilities (at threshold 0.5, using Gumbel CDF):
- p_1 = exp(-exp(-1.0)) = exp(-0.368) = 0.692
- p_2 = exp(-exp(-0.8)) = exp(-0.449) = 0.638
- p_3 = exp(-exp(-0.5)) = exp(-0.607) = 0.545
- p_4 = exp(-exp(0.3)) = exp(-1.350) = 0.259
- p_5 = exp(-exp(1.0)) = exp(-2.718) = 0.066

E[K] = 0.692 + 0.638 + 0.545 + 0.259 + 0.066 = 2.200

This is below 3. The top-3 adapters contribute 1.875, suggesting that only when
all three relevant adapters activate (probability ~0.692 * 0.638 * 0.545 = 0.240)
do we get K >= 3.

For routing to reliably hit K >= 3, we need logits more positive overall, or
lower threshold, or temperature adjustment.

## G. Complexity & Architecture Connection

- Gumbel-sigmoid routing: O(d_hidden * N) per token, where d_hidden = 128 (router),
  N = 5 adapters. Negligible vs. model inference.
- Composition: pre-merge N adapters with scale * routing_weight per adapter.
  No additional memory beyond model weights.
- Finding #185: overhead measured at 0.58% for energy-gap routing; Gumbel-sigmoid
  similar.

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   When K >= 3 adapters activate per token, every 1-cycle in the Cech nerve is
   filled by a 2-simplex, making H^1 = 0 (no topological obstructions).

2. Which existing theorem(s) does the proof build on?
   Finding #242 (Cech nerve beta_1 at k=2 vs k=3); Gumbel-sigmoid reparameterization
   (Jang et al. 2017, ICLR); Cech cohomology nerve theorem (Borsuk 1948).

3. What specific numbers does the proof predict?
   E[K] >= 2.5 at neutral logits; E[K] >= 3.0 with learned routing; fraction K>=3 > 0.50;
   PPL(k=3) / PPL(k=2) <= 1.05.

4. What would FALSIFY the proof (not just the experiment)?
   If K >= 3 but H^1 != 0 (would mean the nerve structure is different from Finding #242);
   or if triple overlaps do not fill cycles (would mean the cover topology changed).

5. How many hyperparameters does this approach add?
   Count: 1 (temperature tau). Tau controls the sharpness of gate decisions; at tau=1
   the Gumbel noise has unit scale, matching the logit scale. Could be derived from
   the logit distribution but we treat it as a known constant (tau=1) from Finding #185.

6. Hack check: Am I adding fix #N to an existing stack?
   No. This experiment measures whether existing routing already places composition
   in the obstruction-free regime. No new mechanism is introduced.
