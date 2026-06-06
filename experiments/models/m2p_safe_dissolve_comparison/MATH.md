# MATH.md: Safe Dissolve — Competitive Promotion Strategies

## TYPE: guided-exploration

## PROVEN FRAMEWORK: Interference additivity under independent adapter updates

**Theorem (prior result, Finding #353):** For N independently-trained domain adapters
{ΔW_1, ..., ΔW_N} trained with cross-domain pairs, naive uniform-scale merging
W' = W + scale·Σ_i ΔW_i produces loss degradation O(scale · ||ΔW_i|| · κ(W)) on
domains where SFT gain is small (base_loss ≈ sft_loss). Specifically, for parity
(SFT delta = 0.033 nats), merge at scale=5 increased base_loss from 0.59 to 3.73
(+532%). This establishes that parity-class domains (SFT delta < 0.05 nats) are
structurally vulnerable to cross-domain interference.

**What this exploration adds:** Given that naive merging is provably destructive for
parity-class domains, which protection strategy achieves the best quality/cost tradeoff?
This is a guided exploration within the interference-additivity framework: we test 5
mechanistically distinct protection strategies and measure their effectiveness.

## UNKNOWN: Which protection strategy gives the best quality/cost tradeoff for multi-adapter promotion

---

## Problem Statement

Merging N cross-domain adapters into the base at uniform scale destroys competent domains.
Finding from cross-domain graph: parity (base_loss=0.59) jumps to 3.73 (6.3x worse) when
10 adapters are merged at PROMOTE_SCALE=5. Meanwhile, repeat improves 36%.

The question is NOT "can we protect competent domains?" (yes, trivially — don't merge).
The question is: **what is the cheapest protection that preserves the most quality?**

Each approach adds cost. We measure that cost alongside quality.

---

## Five Approaches

### S0: Naive (baseline — current behavior)
Merge all N adapters at uniform scale. No protection.
```
W' = W + scale * Σ_i ΔW_i
```
- Hyperparams: 1 (scale)
- Merge cost: O(N) matmul additions
- Inference cost: 0 (merged into base)
- Expected: best quality on hard domains, catastrophic on easy domains

### S1: Loss-Gated Merging
Before merging each adapter, evaluate on all D domains. Skip if any domain degrades >τ%.
```
for adapter_i in adapters:
    losses = evaluate(W + scale * ΔW_i, all_domains)
    if all(loss_d <= base_loss_d * (1 + τ)):
        W = W + scale * ΔW_i
```
- Hyperparams: 2 (scale, τ)
- Merge cost: O(N * D) forward passes for evaluation (dominant cost)
- Inference cost: 0 (merged into base)
- Caveat: greedy — adapter order matters. First-merged adapters are more likely accepted.

### S2: Headroom-Proportional Scale
Scale each adapter proportional to how much the target domain can improve.
```
scale_i = clamp(base_loss_target - τ, 0, max_scale)
W' = W + Σ_i scale_i * ΔW_i
```
- Hyperparams: 2 (τ threshold, max_scale)
- Merge cost: O(N) (no evaluation needed — uses base_loss which we already have)
- Inference cost: 0 (merged into base)
- Property: competent domains get scale≈0 by construction (structural protection)

### S3: Selective Routing (don't merge, route instead)
Maintain two bases: original (for easy domains) and enriched (for hard domains).
Router decides per-input which base to use.
```
if base_loss[domain] < τ:
    output = original_base(x) + per_domain_adapter(x)
else:
    output = enriched_base(x) + per_domain_adapter(x)
```
- Hyperparams: 1 (τ threshold for easy/hard split)
- Merge cost: O(N) (merge only into enriched copy)
- Inference cost: 2x memory (two bases), router lookup
- Property: zero degradation on easy domains (they never see the merge)

### S4: Null-Space Projection
Before merging, project each adapter delta onto the null space of competent-domain representations.
```
# Compute representation basis for competent domains
R = hidden_states(base, competent_domain_data)  # (n_samples, d)
U, S, V = svd(R)
P_null = I - U[:k] @ U[:k].T  # project out top-k directions

# Project each adapter delta
ΔW_i_safe = P_null @ ΔW_i
W' = W + scale * Σ_i ΔW_i_safe
```
- Hyperparams: 3 (scale, k, competent threshold)
- Merge cost: O(D * d^2) for SVD + O(N * d^2) for projection
- Inference cost: 0 (merged into base)
- Property: provably preserves competent-domain representations (up to rank-k)

---

## Metrics (measured per approach)

### Quality
- Per-domain loss on enriched base
- Per-domain M2P quality ratio after recrystallization
- Median quality ratio across all domains
- Worst-case domain degradation (Δ vs original base)
- Number of domains protected (no degradation >5%)

### Cost
- Merge time (seconds) — wall clock for the dissolve step
- Training overhead (seconds) — any extra training required
- Evaluation calls — number of forward passes during merge
- Inference overhead (seconds per token) — additional latency at serving time
- Peak memory during merge (GB)
- Inference memory (GB) — any extra memory at serving time

### Complexity
- Number of hyperparameters introduced
- Number of new components (modules, data structures)
- Lines of code added beyond naive

### Scalability
- How merge cost grows with N (adapters) and D (domains)

---

## Kill Criteria

**K_quality:** At least one approach achieves median quality >90% with 0 domains degraded >5%.

**K_cost:** The best-quality approach runs in <2x the wall time of naive.

**K_tradeoff:** There exists an approach that Pareto-dominates naive (better worst-case, same or better median, acceptable cost overhead).

---

## Predictions

| Approach | Median Q | Worst Δ | Protected | Merge Cost | Inference Cost |
|----------|---------|---------|-----------|------------|---------------|
| S0 Naive | 91.5% | -6.3x (parity) | 3/5 | 1x | 0 |
| S1 Loss-gated | 85-90% | <5% | 5/5 | 50x (N*D evals) | 0 |
| S2 Headroom | 88-92% | ~0% | 5/5 | 1x | 0 |
| S3 Selective | 91.5% | 0% | 5/5 | 1x | 2x memory |
| S4 Null-space | 80-88% | <5% | 5/5 | 10x (SVD) | 0 |

**Prediction:** S2 (headroom) is the Pareto winner — no inference cost, no evaluation overhead,
structural protection. S3 is the quality winner but costs 2x memory. S4 is theoretically elegant
but practically expensive. S1 is safe but slow.

---

## Self-Test

1. **Impossibility property:** This is a guided-exploration (no impossibility theorem).
   The S1 result establishes that naive loss-gating cannot safely merge any cross-domain
   adapter at this scale — loss-gating with τ>0 is structurally equivalent to "do nothing"
   when cross-domain interference exceeds τ for all N adapters.

2. **Cited theorems:** S4 null-space projection relies on SVD rank-k approximation
   (Eckart-Young-Mirsky theorem). The projected delta ΔW_safe = P_null @ ΔW minimizes
   ||ΔW - ΔW_safe||_F subject to P_null being an orthogonal projector. However, this
   minimizes Frobenius distance in WEIGHT space, not hidden-state distance — cite this
   limitation explicitly. Reference: Eckart & Young (1936), Mirsky (1960).

3. **Predicted numbers:** Table in Predictions section gives specific ranges per strategy
   (see Section: Predictions). Prediction accuracy: 3/5 strategies correctly predicted
   directionally; S2 and S4 parity protection both WRONG.

4. **Falsification condition:** The exploration is falsified if S3 (selective routing)
   does NOT achieve 5/5 domain protection at ≤2× inference memory cost. (K884 measures this.)
   S1's merge count is a secondary diagnostic: if S1 merges >0 adapters, the loss-gating
   signal is more informative; currently merges=0 means the interference gap is universal.

5. **Hyperparameter count:** S0: 1 (scale). S1: 2 (scale, τ). S2: 2 (τ, max_scale).
   S3: 1 (τ for easy/hard split). S4: 3 (scale, k, competent threshold).
   S3 has the lowest hyperparameter count among protective strategies.

6. **Hack check:** This is a comparison study (5 pre-specified strategies), not a fix
   iteration. No hacks — strategies are independent designs evaluated head-to-head.
