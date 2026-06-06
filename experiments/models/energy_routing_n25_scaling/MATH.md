# Energy Gap Routing at N=25: Frontier Extension

## Type: Frontier Extension

**Proven result:** Energy gap top-1 routing achieves 88% accuracy at N=5 domains
(Finding #185). The router selects argmin_i DeltaE_i where
DeltaE_i = NLL(adapter_i, query) - NLL(base, query).

**Gap:** Does this mechanism degrade as candidate set grows from N=5 to N=24?
Finding #186 shows legal-finance confusion (gap = 0.041 nats), suggesting that
more semantically similar domains will increase misrouting.

## A. Failure Mode Identification

At N=5, the router must distinguish 5 candidates. At N=24, it must distinguish
24. Two failure modes:

1. **Confusion escalation:** With more domains, the probability of having a
   "near-duplicate" domain increases. If domain gaps cluster, argmin may select
   the wrong adapter.

2. **Signal dilution:** If adapters are less specialized (trained on fewer
   domain-specific tokens relative to total), energy gaps shrink and become
   indistinguishable from noise.

## B. The Right Question

Not "how do we prevent confusion at N=24?" but rather:
**"Given N candidates with energy gaps drawn from some distribution, what is
the probability that the true best has the most negative gap?"**

This is an order statistics problem.

## C. Prior Mathematical Foundations

### Order Statistics of the Minimum

Let DeltaE_1, ..., DeltaE_N be the energy gaps for N adapters on a given query.
The correct adapter c has gap DeltaE_c. Routing is correct when
DeltaE_c = min_i DeltaE_i.

**Theorem (Order Statistics).** If DeltaE_c has mean mu_c and the other N-1
gaps have mean mu_j > mu_c (the correct adapter has lower mean gap), then the
probability of correct selection depends on:
- The separation: delta = min_{j != c} (mu_j - mu_c)
- The variance sigma^2 of the gap distribution
- The number of competitors N-1

For Gaussian gaps with common variance sigma^2:

P(correct) = P(DeltaE_c < min_{j != c} DeltaE_j)

By independence of the incorrect adapters' gaps from the correct one:

P(correct) = integral P(DeltaE_c = x) * prod_{j != c} P(DeltaE_j > x) dx

### Extreme Value Theory (Gumbel)

For the minimum of N-1 i.i.d. random variables from a distribution with CDF F,
the minimum converges to a Gumbel distribution. For N-1 competitors each with
mean mu_other and variance sigma^2:

E[min_{j != c} DeltaE_j] ~ mu_other - sigma * sqrt(2 * ln(N-1))

This means: as N grows, the minimum of the competitors' gaps DECREASES
(becomes more negative), making it harder for the correct adapter to win.

### Scaling Prediction

The "effective margin" between the correct adapter and the best competitor:

effective_margin(N) = delta - sigma * sqrt(2 * ln(N-1))

where delta = mu_other - mu_c is the raw separation.

At N=5: effective_margin = delta - sigma * sqrt(2 * ln(4)) = delta - 1.665 * sigma
At N=24: effective_margin = delta - sigma * sqrt(2 * ln(23)) = delta - 2.505 * sigma

The margin shrinks by 0.84 * sigma going from N=5 to N=24.

## D. Predictions (Frontier Extension)

**This is a frontier extension, not a proof verification.** The Gumbel analysis
gives approximate predictions that the experiment will probe.

### Prediction 1: Accuracy Degradation

At N=5, accuracy = 88% (44/50). The 12% errors came from legal-finance confusion
(delta ~ 0.041 nats). With 24 domains, we expect more near-duplicate pairs.

Predicted accuracy at N=24: We expect degradation proportional to the number of
"confusable pairs." At N=5, ~2 domains confused (legal/finance). At N=24,
domain clusters (e.g., economics/finance/marketing, medical/health_fitness/psychology,
legal/politics, science/engineering/environmental) may create 4-6 confusion clusters.

**Prediction:** Accuracy drops from 88% to 60-75% range, but remains well above
random (1/24 = 4.2%). K1 threshold is 60%.

### Prediction 2: Cluster Structure

The confusion matrix should show block-diagonal structure: errors within semantic
clusters, not random scatter.

Expected confusion clusters:
- {economics, finance, marketing} — business/money vocabulary
- {medical, health_fitness, psychology} — health/body vocabulary
- {science, engineering, environmental} — STEM vocabulary
- {legal, politics, sociology} — governance vocabulary
- {education, linguistics, philosophy} — humanities vocabulary
- {creative_writing, music, cooking} — lifestyle/arts (more distinct)
- {code, math, cybersecurity} — technical/structured

### Prediction 3: Overhead Scaling

Energy gap computation is O(N) in the number of adapters (one forward pass per adapter).
At N=5, overhead was ~29.5% of base inference (dominated by model loading).
At N=24, overhead scales linearly: ~24/5 * 29.5% = ~142% without caching.

With sequential load-evaluate-unload: overhead per query ~ N * t_forward.
K3 threshold: 120s per query. With t_forward ~ 0.1-0.5s, N=24 gives
2.4-12s per query. Well within K3.

### Prediction 4: Math Correctness

At N=5, math correctness was 70% with top-1 routing. At N=24, math adapter
must be selected from 24 candidates. Math is distinctive (structured problems,
numbers), so we expect routing accuracy on math to remain high (>80%).

**Prediction:** Math correctness >= 50% (K2 threshold). The math adapter
should still be selected correctly most of the time due to its distinctive
energy profile.

## E. Assumptions & Breaking Conditions

1. **Adapters are specialized:** Each adapter has lower NLL on its target domain.
   If an adapter fails to specialize, its energy gap will not distinguish it.
   *Breaking:* If >5 adapters have no specialization, routing accuracy drops below 50%.

2. **Energy gaps are approximately independent:** The gap for adapter i on query q
   depends primarily on the (adapter, query) pair, not on other adapters.
   *Breaking:* If adapters share parameters (they share frozen A matrices from
   Grassmannian init), gaps may be correlated, reducing effective N.

3. **Domain overlap is moderate:** Some domains are similar but most have
   distinctive vocabulary/structure.
   *Breaking:* If most domains are interchangeable, the confusion matrix
   becomes dense and routing becomes random.

## F. Worked Example (N=5 to N=24 Gumbel scaling)

At N=5 with sigma=0.3 nats and delta=0.5 nats (typical medical-vs-others gap):
- effective_margin = 0.5 - 0.3 * 1.665 = 0.5 - 0.50 = 0.00 nats
- This is marginal — consistent with 88% accuracy (some noise flips it)

At N=24 with same parameters:
- effective_margin = 0.5 - 0.3 * 2.505 = 0.5 - 0.75 = -0.25 nats
- Negative margin: the best competitor likely beats the correct adapter
- Prediction: accuracy drops for domains with delta < 0.75

For well-separated domains (code, math) with delta > 1.0:
- effective_margin = 1.0 - 0.75 = 0.25 nats > 0
- Should maintain high accuracy even at N=24

## G. Complexity & Architecture Connection

- **Energy gap computation:** O(N) forward passes per query (one per adapter)
- **Memory:** O(1) adapters loaded at a time (sequential evaluation)
- **Routing decision:** O(N) argmin, negligible
- **Total per-query overhead:** N * (load_adapter + forward_pass)
- With N=24 and ~2s per adapter cycle: ~48s per query for full evaluation
- **Architectural note:** In production, adapters would be cached in GPU memory
  and only the forward pass needed, reducing to N * t_forward ~ 2.4s

## Self-Test

1. **One mathematical property:** The argmin selector's error probability increases
   logarithmically with N via extreme value theory (Gumbel), not linearly.

2. **Existing theorems:** Gumbel extreme value theorem (Fisher-Tippett, 1928);
   order statistics of the minimum of N random variables.

3. **Specific predictions:** Accuracy 60-75% at N=24 (from 88% at N=5).
   Block-diagonal confusion matrix. Math/code accuracy >80%.
   Overhead <120s per query.

4. **Falsification:** The frontier extension is wrong if accuracy drops BELOW 60%
   (K1), indicating the Gumbel analysis underestimates degradation — possibly
   because adapters are correlated (violating assumption 2).

5. **Hyperparameters added:** 0. Energy gap routing has zero hyperparameters.

6. **Hack check:** No. This is measuring scaling of an existing zero-parameter method.
