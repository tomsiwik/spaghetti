# Expert Pruning from Composition: Mathematical Foundations

## Variables and Notation

| Symbol | Shape / Type | Description |
|--------|-------------|-------------|
| W | (d_out, d_in) per layer | Base model frozen weight matrices |
| delta_i = B_i A_i | (d_out, d_in) per layer | LoRA expert i's weight delta (rank r) |
| N | scalar | Total number of experts (N=50 for pilot) |
| k | scalar | Number of experts in a subset (k <= N) |
| S | subset of {1..N} | A specific subset of experts, |S| = k |
| r | scalar | LoRA rank (r=16 for pilot adapters) |
| d | scalar | Model hidden dimension (d=3584 for Qwen2.5-7B) |
| L | scalar | Number of transformer layers (L=28 for Qwen2.5-7B) |
| PPL(S, D) | scalar | Perplexity of base + sum_{i in S} delta_i evaluated on dataset D |
| PPL_base(D) | scalar | Base model perplexity on D (no adapters) |
| Q_i | scalar | Individual quality score for expert i |
| pi | permutation of {1..N} | Expert ordering by quality rank |
| cos_ij | scalar | Cosine similarity between delta_i and delta_j (flattened) |
| I_ij | scalar | Interaction index between experts i and j |
| D_cal | dataset | Calibration dataset (diverse, domain-balanced) |
| D_mmlu | dataset | MMLU evaluation set |
| D_domain_i | dataset | Expert i's domain-specific evaluation data |

## 1. The Pruning Problem

### 1.1 Formulation

Given N trained experts, find the subset S* of size k* that maximizes
composition quality:

```
S* = argmax_{S subseteq {1..N}} Quality(S)
```

where Quality(S) can be defined as:
- Negative PPL on calibration data: Quality(S) = -PPL(S, D_cal)
- MMLU accuracy: Quality(S) = Acc(S, D_mmlu)
- Domain-weighted average: Quality(S) = sum_i w_i * Quality_i(S)

The combinatorial search space is C(N, k) = N! / (k!(N-k)!). For N=50, k=40:
C(50,40) = C(50,10) = 10,272,278,170. Exhaustive search is infeasible.

### 1.2 Why Pruning Might Help

Under naive addition, the composed model has effective weights:

```
W_S^l = W^l + sum_{i in S} B_i^l A_i^l
```

Each expert contributes its full delta regardless of relevance. Three failure
modes create pruning opportunities:

**Mode 1: Off-domain noise.** Expert delta_i trained on domain D_i adds noise
to tokens outside D_i. If the calibration set has no D_i tokens, expert i's
entire contribution is noise.

**Mode 2: Within-cluster interference.** For experts i,j in the same semantic
cluster (e.g., both math-related), within-cluster |cos_ij| is 7.84x higher
than cross-cluster. Their deltas partially cancel or amplify unpredictably.

**Mode 3: Low-quality distillation.** Some experts may have been trained on
poor-quality synthetic data (pilot50: SQL regressed -21.4%). These contribute
harmful deltas that degrade the composition.

### 1.3 Why Pruning Might NOT Help (The Null Hypothesis)

Under near-perfect orthogonality (cos_ij ~ 0.0002 at d=3584, r=16):

1. Expert contributions are additive and independent
2. Each expert's delta affects a nearly disjoint subspace
3. Adding expert i cannot harm expert j's contribution
4. The marginal benefit of each expert is always non-negative

If this holds, S* = {1..N} and pruning never helps. This is the "more is
always better" property -- a strong scaling result for SOLE.

**The critical test:** If removing bottom-20% experts (k=40) does NOT improve
PPL by >1% over using all 50, the null hypothesis holds and pruning is
unnecessary.

## 2. Expert Quality Ranking

### 2.1 Individual Quality Score

For each expert i, compute quality on domain-specific evaluation data:

```
Q_domain_i = (PPL_base(D_domain_i) - PPL({i}, D_domain_i)) / PPL_base(D_domain_i)
```

This is the relative PPL improvement when adding expert i alone. From pilot50
benchmark data, Q_domain ranges from -21.4% (SQL, regression) to +80% (best
domains), with mean 42.2%.

### 2.2 Composition Contribution Score (LOO)

The leave-one-out contribution of expert i to the full composition:

```
C_i = PPL({1..N}\{i}, D_cal) - PPL({1..N}, D_cal)
```

- C_i > 0: expert i helps (removing it hurts PPL)
- C_i < 0: expert i harms (removing it improves PPL)

Under orthogonality, C_i should correlate strongly with Q_domain_i because
there are no interaction effects. The Spearman rank correlation rho(Q_domain,
C) quantifies this.

### 2.3 Relationship Between Rankings

We have three possible rankings:

1. **Domain PPL ranking** pi_domain: sort by Q_domain_i (descending)
2. **LOO ranking** pi_loo: sort by C_i (descending)
3. **MMLU ranking** pi_mmlu: sort by individual MMLU accuracy delta

Under the orthogonality hypothesis, all three should be concordant. We measure
pairwise Kendall tau:

```
tau(pi_domain, pi_loo) >= 0.6  (K2 threshold)
tau(pi_domain, pi_mmlu) >= 0.6  (supplementary)
```

If tau < 0.6, rankings are unstable across metrics, meaning "quality" is
metric-dependent and pruning decisions are unreliable.

## 3. Accumulation Curve: Quality as f(k)

### 3.1 Rank-Ordered Accumulation

Sort experts by quality rank pi (best to worst). Define the accumulation curve:

```
Q(k) = -PPL({pi(1), pi(2), ..., pi(k)}, D_cal)    for k = 1, 2, ..., N
```

This measures composition quality as we add experts in rank order. The shape
of Q(k) reveals the quality-quantity tradeoff:

**Case A: Monotonically increasing.** Every additional expert helps. Optimal
subset is S* = {1..N}. Pruning never beneficial.

```
Q(k) strictly increasing for all k
=> dQ/dk > 0 everywhere
=> Null hypothesis confirmed
```

**Case B: Peak at k* < N.** Quality peaks at some k*, then declines. The
bottom (N-k*) experts actively harm the composition.

```
Q(k*) > Q(N)
=> Pruning bottom (N-k*) experts improves quality
=> k* is the optimal composition size
```

**Case C: Diminishing returns.** Q(k) increases but with diminishing marginal
returns. No clear peak, but the last experts contribute negligibly.

```
dQ/dk > 0 but d^2Q/dk^2 < 0
=> Pruning saves compute without meaningful quality loss
=> Marginal cost-benefit analysis determines practical k
```

### 3.2 Expected Behavior Under SOLE Orthogonality

Under perfect orthogonality, each expert's contribution to PPL is independent.
The PPL of the composed model decomposes approximately as:

```
log PPL(S, D) ~= log PPL_base(D) - sum_{i in S} epsilon_i(D)
```

where epsilon_i(D) >= 0 is expert i's marginal log-PPL reduction on dataset D.

This gives:

```
Q(k) = -PPL(S_k, D) = -exp(log PPL_base(D) - sum_{j=1}^{k} epsilon_{pi(j)}(D))
```

Since epsilon_i >= 0 for all experts that help and the sum only increases, Q(k)
is monotonically increasing. This means Case A holds under perfect orthogonality.

**The interesting question:** At SOLE's empirical cosine (~0.0002 at d=3584 but
within-cluster 7.84x higher), is the departure from perfect orthogonality
enough to produce Case B?

### 3.3 Formal Bound on Pruning Benefit

The maximum PPL improvement from pruning the bottom (N-k) experts is bounded by
the interaction terms between pruned and retained experts:

```
|PPL(top-k) - PPL(all-N)| <= sum_{i in pruned} |epsilon_i| + sum_{i in pruned, j in kept} |I_ij|
```

The first term is the direct contribution of pruned experts (positive if they
help, negative if they harm). The second term captures interaction effects.

Under SOLE orthogonality:

```
|I_ij| <= c * |cos_ij| * ||delta_i||_F * ||delta_j||_F
```

With cos ~ 0.0002 (cross-cluster) and cos ~ 0.0016 (within-cluster, 7.84x):

```
|I_ij| ~ O(10^-4) for cross-cluster pairs
|I_ij| ~ O(10^-3) for within-cluster pairs
```

For N=50 with ~5 clusters of ~10 experts each, the total interaction is:

```
Total_I = sum_cross |I_ij| + sum_within |I_ij|
        ~ C(40,1)*10*0.0001 + C(10,2)*5*0.001
        ~ 0.4 + 0.225
        ~ 0.625
```

In relative PPL terms, this is at most ~0.6% effect. The K1 threshold (1%
improvement from pruning bottom-20%) is thus right at the boundary where
interaction effects could be detectable.

## 4. Greedy Forward Selection

### 4.1 Algorithm

Greedy forward selection builds the optimal subset incrementally:

```
S_0 = {}
For t = 1, 2, ..., k:
    i* = argmax_{i not in S_{t-1}} Quality(S_{t-1} union {i})
    S_t = S_{t-1} union {i*}
```

This requires N + (N-1) + ... + (N-k+1) = O(Nk) quality evaluations. For
N=50, k=50: 50*50/2 = 1250 evaluations.

### 4.2 Complexity Analysis

Each quality evaluation requires one PPL computation on the calibration set.
Using the subtraction/addition approach from the LOO experiment:

```
Cost per eval = T_subtract + T_ppl_eval + T_restore
              ~ 0.2s + 5s + 0.2s = 5.4s
```

For O(Nk) = O(N^2) evaluations at N=50:

```
Total = 1250 * 5.4s ~ 6750s ~ 112 min
```

This is feasible but expensive (K3 threshold is O(N log N) evaluations).

### 4.3 Optimized Greedy via Ranking Shortcut

Under orthogonality, the greedy order should match the rank order. We can
verify this cheaply:

1. Rank experts by individual quality Q_domain (O(N log N))
2. Run greedy forward selection on top-k candidates only
3. Compare greedy and rank-ordered accumulation curves

If the curves match (within noise), ranking-based pruning is sufficient and
greedy is unnecessary. This is the scalability test: O(N log N) vs O(N^2).

### 4.4 Scalability to N=500

At N=500:
- Ranking-based: O(N log N) = O(500 * 9) = O(4500) evaluations. Each is O(1)
  (individual quality computation, one forward pass). Total ~ 4500 * 5s = 22500s
  ~ 6.25 hrs. Feasible with batch optimization.
- Greedy: O(N^2) = O(250,000) evaluations. At 5s each: 1.25M seconds = 14 days.
  NOT feasible.
- Combinatorial: C(500,400) ~ 10^58. NOT feasible.

K3 requires O(N log N) to be sufficient. If greedy materially outperforms
ranking at N=50, the approach does not scale.

## 5. Bottom-K Pruning Analysis

### 5.1 The Key Test

Sort experts by Q_domain descending. Define bottom-20% as the 10 experts with
lowest individual quality scores.

```
S_full = {1, ..., 50}
S_pruned = S_full \ {bottom 10}
```

Compute:

```
Delta_prune = (PPL(S_pruned, D_cal) - PPL(S_full, D_cal)) / PPL(S_full, D_cal) * 100%
```

K1 kill criterion: |Delta_prune| < 1% (removing bottom-20% does not improve
PPL by >1%).

### 5.2 Expected Outcome

From pilot50 data:
- Mean Q_domain = 42.2%, std ~ 15%
- Bottom-20% (10 experts) have Q_domain ~ 10-25%
- These experts still provide positive domain-specific improvements

Under orthogonality, their contributions to general calibration PPL should be:
- Small but non-negative (they help on their specific domain tokens)
- Independent of other experts (no interaction penalty)

Expected Delta_prune: +0.1% to +0.5% (removing them HURTS PPL slightly because
we lose their small positive contributions). This would confirm the null
hypothesis.

However, if within-cluster interference is significant for low-quality experts
(who may be more likely to have overlapping domains), Delta_prune could be
negative (removing them HELPS). This would support the pruning hypothesis.

### 5.3 Sweep Over k

To fully characterize the tradeoff, sweep k from 5 to 50:

```
For k in {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}:
    S_k = {pi(1), ..., pi(k)}  (top-k by quality rank)
    PPL_k = PPL(S_k, D_cal)
    MMLU_k = Acc(S_k, D_mmlu)
```

Plot PPL_k and MMLU_k vs k. If there is a peak at k* < 50, that defines the
optimal composition size. If monotonically improving to k=50, more is better.

## 6. MMLU as Secondary Metric

### 6.1 Why PPL and MMLU May Disagree

The -3.71pp MMLU regression at N=50 suggests that PPL improvement does not
guarantee task accuracy improvement. Possible explanations:

1. **Dilution:** Each adapter shifts logits slightly. At N=50, the cumulative
   shift drowns out the base model's calibrated predictions.

2. **Domain mismatch:** Most MMLU subjects have no corresponding adapter. The
   adapters' contributions on those subjects are pure noise.

3. **Distribution shift:** Adapters trained on synthetic instruction data shift
   the model toward chat-style outputs, not multiple-choice answers.

### 6.2 MMLU Evaluation Protocol

For each subset S_k:
1. Compose via weight-space addition (naive sum, no scaling)
2. Evaluate on 30 held-out MMLU subjects (not in domain map)
3. Record 5-shot accuracy using standard MMLU prompt format
4. Compare to base model accuracy

This gives us two accumulation curves: PPL(k) and MMLU(k). If MMLU peaks at
k* < 50, that directly demonstrates that pruning improves task quality.

## 7. Worked Numerical Example

### Setup
- N=50, r=16, d=3584, L=28
- Pilot50 data: 42.2% mean domain PPL improvement, 98% win rate
- Base PPL on calibration ~ 6.0 (typical for Qwen2.5-7B on diverse text)
- Production cosine: ~0.0002 cross-cluster, ~0.0016 within-cluster

### Expected Accumulation Curve

**At k=1 (best expert only):**
- PPL improvement concentrated on best expert's domain tokens
- Estimated PPL ~ 5.7 (base 6.0, ~5% improvement from one expert on
  matched calibration tokens)

**At k=10 (top-10 experts):**
- Cover ~20% of calibration domains directly
- Estimated PPL ~ 5.2 (~13% improvement)

**At k=25 (top-25 experts):**
- Cover ~50% of calibration domains
- Estimated PPL ~ 4.8 (~20% improvement)

**At k=50 (all experts):**
- Full coverage
- Estimated PPL ~ 4.5 (~25% improvement)

**At k=40 (pruned bottom-20%):**
- Lose coverage of 10 lowest-quality domains
- Estimated PPL ~ 4.55 (~24.2% improvement)
- Delta_prune ~ +1.1% (removing bottom-10 HURTS PPL slightly)

### Expected MMLU Curve

**At k=0 (base):** ~55% accuracy (Qwen2.5-7B-Instruct baseline)
**At k=10:** ~54% (slight regression from noise)
**At k=50:** ~51.3% (the observed -3.71pp regression)
**At k=40 (pruned):** ~52% (slightly better than k=50, removing noise sources)

The MMLU curve is more likely to show a pruning benefit than the PPL curve
because MMLU is more sensitive to distribution shift than calibration PPL.

### Greedy vs Ranking Comparison

Under orthogonality, greedy and ranking should agree. Expected discordance:

```
|PPL_greedy(k) - PPL_ranked(k)| / PPL_ranked(k) < 0.5%
```

for all k. If this holds, ranking-based selection at O(N log N) is sufficient.

## 8. Assumptions

1. **Weight-space additive composition (no scaling).** Consistent with SOLE
   default. Each expert contributes B_i A_i at full magnitude. Not 1/N weighted.

2. **Pilot50 adapters share identical LoRA config.** rank=16, all-modules
   (q/k/v/o/gate/up/down). No rank mixing or partial-module adapters.

3. **Individual quality ranking from pilot50 benchmark is valid.** The
   benchmark used training-domain eval data (contaminated but directionally
   correct for ranking). Cross-validation with LOO and MMLU tests robustness.

4. **Calibration data is domain-balanced.** Uses diverse texts covering code,
   math, science, creative, QA, professional domains. No bias toward any
   specific adapter's training domain.

5. **4-bit NF4 quantization preserves ranking.** Both composed and base models
   use identical quantization. Absolute PPL changes, but relative ordering
   should be preserved.

6. **PPL computation is numerically stable.** Float32 loss accumulation with
   per-token cross-entropy. No padding artifacts (process one text at a time).

7. **Orthogonality holds for all cross-cluster pairs.** Within-cluster cos may
   be up to 7.84x higher. The experiment does NOT assume perfect orthogonality;
   it tests whether the departure matters for pruning.

## 9. Connection to Prior Art

### TIES-Merging (Yadav et al., 2023)

TIES resolves sign conflicts at the parameter level. In SOLE, sign conflicts
are minimal because orthogonal deltas operate in different subspaces. TIES
would be redundant if orthogonality holds. However, for within-cluster pairs
(cos ~ 0.0016), TIES-style sign conflict resolution could marginally help.

### DARE (Yu et al., 2023)

DARE randomly drops delta parameters and rescales. This is complementary to
expert-level pruning (we drop entire experts, they drop parameters within
experts). DARE's insight -- most delta parameters are redundant -- is consistent
with SOLE's low-rank structure (rank-16 in d=3584 space).

### REAP (Cerebras, 2025)

Router-weighted Expert Activation Pruning for traditional MoE. Uses router gate
values and L2 norms to rank experts. Achieves near-lossless 50% compression.
In SOLE, there is no router, so ranking must come from LOO or individual quality.

### Shapley-MoE

Monte Carlo Shapley values for expert pruning. O(M*N) forward passes where M
is the number of MC samples. At M=20, N=50: 1000 evaluations. Comparable to
greedy but provides theoretically grounded importance scores. Under SOLE
orthogonality, Shapley collapses to LOO (see LOO MATH.md Section 5).

### EASY-EP

Output-aware importance scoring for MoE pruning. On domain-specific tasks,
pruned model matched or exceeded full model -- removing irrelevant experts
reduces noise. This is the closest analogy to our hypothesis: pruning
irrelevant domain experts from SOLE composition to reduce noise on MMLU.
