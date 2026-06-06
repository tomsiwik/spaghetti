# ELO Adapter Tournament: Mathematical Foundations

## 1. Mechanism Definition

### ELO Rating System

The ELO rating system (Elo, 1978) models pairwise comparison outcomes as Bernoulli
trials with logistic probability. Originally designed for chess, it is now standard
for model comparison (LMSYS Chatbot Arena, arxiv 2403.04132).

**State:** Each adapter variant i has a rating R_i in R, initialized to R_0 = 1500.

**Expected score:** For a match between adapters i and j:

  E_i = 1 / (1 + 10^((R_j - R_i) / 400))
  E_j = 1 - E_i

This is the logistic (Bradley-Terry) model: P(i beats j) = sigma((R_i - R_j) / s)
where s = 400/ln(10) ~ 173.7 and sigma is the sigmoid function.

**Outcome:** In our setting, adapter i "wins" against j if the composition including
adapter i achieves lower PPL than the composition including adapter j on held-out data.
Formally, for a fixed set of other adapters C = {a_1, ..., a_{N-1}}:

  S_i = 1  if  PPL(compose(C union {i})) < PPL(compose(C union {j}))
  S_i = 0  otherwise (S_j = 1)

No draws: PPL is continuous, ties have probability 0.

**Update rule:** After observing outcome S_i in {0, 1}:

  R_i <- R_i + K * (S_i - E_i)
  R_j <- R_j + K * (S_j - E_j)

where K is the step size (we use K=32, standard for established players).

**Key property:** The sum R_i + R_j is conserved per match (zero-sum update).
Global sum is conserved: sum_i R_i = N * R_0 at all times.

### Composition PPL as Match Outcome

For a domain d with M adapter variants {v_1, ..., v_M}, each match compares
two variants by swapping them into a fixed composition of K other domain adapters:

  PPL_i = perplexity(base + (1/K) * sum_{k != d} adapter_k + (1/K) * v_i, val_d)

This measures how well variant v_i composes with the rest of the system, which is
the operationally relevant quality metric (not standalone PPL).

### Ground Truth: Individual Adapter Quality

For correlation analysis, we define ground-truth quality as the standalone
adapted PPL improvement over base:

  quality_i = PPL_base(domain_i) / PPL_adapted(v_i, domain_i)

Higher is better (more improvement over base). This is the metric ELO rankings
should correlate with if the tournament is meaningful.

## 2. Why It Works

**Bradley-Terry consistency:** The ELO system converges to the maximum likelihood
estimate of the Bradley-Terry model. After T rounds, the MLE rating vector
satisfies R* = argmax_R prod_{matches} P(observed outcome | R). For M adapters
with O(M^2) pairwise comparisons, the MLE exists and is unique (up to translation)
when the comparison graph is connected (Zermelo, 1929).

**Composition as proxy for quality:** If adapter quality is a latent scalar theta_i
and composition PPL is monotonically decreasing in theta (better adapters improve
composition more), then the Bradley-Terry model is well-specified and ELO ratings
will recover the latent quality ranking.

**Key assumption:** Composition PPL is a monotone function of individual adapter
quality. This holds when: (a) adapters are for the same domain, (b) interference
is low (guaranteed by Grassmannian A-matrices), (c) the composition function
(1/N scaling) treats all adapters symmetrically.

## 3. What Breaks It

**Non-transitivity:** If adapter A beats B in composition, B beats C, but C beats A,
the Bradley-Terry model is misspecified. This can happen if different adapter variants
specialize on different sub-distributions within a domain (e.g., one medical adapter
is good at cardiology, another at neurology). The ELO system will still converge but
the ranking may not correlate with any single quality metric.

**K1 fails when:** Composition PPL is NOT monotone in individual quality. Specifically:
- If interference between specific adapter pairs dominates (pair-specific effects
  rather than individual quality effects), then pairwise ELO captures interaction
  quality, not individual quality, and Kendall tau with standalone PPL will be low.
- Threshold: Kendall tau < 0.5 means worse than a trivially biased estimator.

**K2 fails when:** The number of matches M*(M-1)/2 per domain times the cost per
PPL evaluation exceeds the time budget. For 10 adapters per domain, 45 matches.
If each PPL eval takes 20 seconds, that's 15 minutes per domain. With 3 domains,
45 minutes total. We need efficient evaluation (fewer val batches per match).

**Sample complexity:** For M items, O(M log M) comparisons suffice to learn the
ranking with high probability (Shah & Wainwright, 2017). Full round-robin
M*(M-1)/2 is more than sufficient but expensive. For M=4 variants per domain,
only 6 matches are needed (tractable).

## 4. Assumptions

1. **Adapters within a domain differ only in quality, not in kind.**
   Justified by: same data, same architecture, different only in seed/hyperparams.
   If wrong: non-transitivity breaks monotonicity assumption.

2. **Composition PPL reflects individual adapter quality.**
   Justified by: Grassmannian orthogonality ensures low cross-adapter interference
   (mean |cos| = 0.00125), so composition quality is approximately additive.
   If wrong: K1 fails (tau < 0.5).

3. **ELO converges in O(M^2) matches.**
   Justified by: with M=4 variants, 6 matches per domain is full round-robin.
   After 2-3 full rounds (12-18 matches), ratings stabilize.
   If wrong: K2 fails (too many rounds needed).

## 5. Complexity Analysis

**Training cost:** D domains x V variants x T iterations = D*V*T forward+backward passes.
With D=3, V=4, T=100: 1,200 training iterations total (~10 min on M5 Pro).

**Tournament cost:** D domains x V*(V-1)/2 matches x E eval_batches = D*6*E forward passes.
With D=3, E=15: 270 forward passes (~3 min on M5 Pro).

**Total memory:** One model in memory at a time (~4 GB unpacked). Adapters saved to disk
(~200 KB each). Peak: model + 2 adapters loaded for comparison = ~4.001 GB.

## 6. Worked Example (D=2, V=3)

Domain "medical", 3 variants with seeds {42, 123, 7}:
- v1 (seed 42): standalone PPL = 8.5, quality = 12.0/8.5 = 1.41
- v2 (seed 123): standalone PPL = 9.2, quality = 12.0/9.2 = 1.30
- v3 (seed 7): standalone PPL = 7.8, quality = 12.0/7.8 = 1.54

Round-robin matches (composition PPL on medical val set):
- v1 vs v2: compose({code_best, v1}) PPL=9.1, compose({code_best, v2}) PPL=9.8
  -> v1 wins. E_v1 = 0.5 (equal ratings). R_v1 = 1500 + 32*(1-0.5) = 1516, R_v2 = 1484.
- v1 vs v3: compose PPL v1=9.1, v3=8.6 -> v3 wins.
  E_v1 = 1/(1+10^((1500-1516)/400)) = 0.523. R_v1 = 1516 + 32*(0-0.523) = 1499.
  R_v3 = 1500 + 32*(1-0.477) = 1517.
- v2 vs v3: compose PPL v2=9.8, v3=8.6 -> v3 wins.
  E_v2 = 1/(1+10^((1517-1484)/400)) = 0.453. R_v2 = 1484 + 32*(0-0.453) = 1470.
  R_v3 = 1517 + 32*(1-0.547) = 1531.

Final ELO: v3=1531 > v1=1499 > v2=1470.
Quality ranking: v3 (1.54) > v1 (1.41) > v2 (1.30).
Kendall tau = 1.0 (perfect agreement).

## 7. Connection to Architecture

**Evolve track enabler:** The ELO tournament replaces the killed clone-compete mechanism.
Instead of inheriting parameters (which failed because warm-start = cold-start),
we train multiple fresh variants and SELECT the best via tournament.

**Composition pipeline:** The tournament naturally measures composition quality,
which is the deployment-relevant metric. An adapter that is good standalone but
composes poorly will be correctly penalized.

**Routing integration:** The winning adapter per domain feeds into the softmax router
(which is already proven: matches oracle at N=24, 0% fallback). The tournament
handles adapter QUALITY while the router handles adapter SELECTION at inference time.

**Scaling:** For N domains with V variants each, tournament cost is O(N*V^2) pairwise
evaluations. With V=4 and N=25 (proven scale), that is 25*6 = 150 matches, each
taking ~3 seconds = ~7.5 minutes total. Well within the 30-minute budget.
