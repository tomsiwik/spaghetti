# Cross-Domain Composition: Research Digest (Revision 2)

## Hypothesis

Merged experts handle cross-domain queries (e.g., "compute then reverse") without
degrading more than 20% vs a base model trained on all domains. Cross-domain queries
route to the correct involved expert(s) more than 80% of the time (tightened from 50%).

## Revision History

**v2 (2026-03-14):** Addresses 5 fixes from adversarial review:
1. Per-type gaps with stddev reported (was only aggregate mean)
2. Single-expert K1 metric added (hash-ring scenario)
3. "Reviewer attack neutralized" claim removed; reframed as oracle routing finding
4. Cancellation artifact in aggregate K1 explicitly acknowledged
5. All 10 domain pairs tested (was 5), 5 seeds (was 3), K2 threshold 20% (was 50%)

## What This Model Is

This experiment tests whether weight-space composition of LoRA experts can handle
queries that span 2 domains simultaneously. We construct synthetic cross-domain
queries by chaining domain operations (e.g., "compute 12+34=46, then reverse: 64").

We evaluate 3 composition strategies on all C(5,2)=10 cross-domain pairs:
1. **Best single expert** -- the hash-ring scenario (one expert, best of involved)
2. **Multi-expert (oracle 2)** -- compose the 2 relevant experts (requires a router)
3. **Naive merge (all 5)** -- average all expert deltas

## Lineage in the Arena

```
exp_gram_schmidt_composition (proven)
  |
  +-- exp_merge_order_dependence (proven, CV=0.029%)
  |
  +-- exp_composition_vs_monolithic (supported)
  |
  +-- exp_cross_domain_composition (THIS EXPERIMENT)
```

## Key References

- Ilharco et al., "Editing Models with Task Arithmetic", ICLR 2023
- Yadav et al., "TIES-Merging", NeurIPS 2023
- Huang et al., "LoRAHub", 2023 (arXiv:2307.13269)
- Prabhakar et al., "LoRA Soups", COLING 2025

## Empirical Results

### Aggregate Kill Criteria (5 seeds, 10 cross-domain types, N=50 trials)

| Criterion | Metric | Threshold | Measured | Status |
|-----------|--------|-----------|----------|--------|
| K1_multi | Oracle 2-expert mean gap | >20% | -1.0% +/- 16.7% | **PASS** |
| K1_single | Best single expert mean gap | >20% | -7.0% +/- 15.2% | **PASS** |
| K2 | Routing error rate | >20% | 8.0% (4/50) | **PASS** |

### Per-Type Multi-Expert Gaps (% vs base, mean +/- std across 5 seeds)

| Cross-Domain Type | Mean | Std | Max | Exceeds 20%? |
|-------------------|------|-----|-----|---------------|
| arith_reverse | +21.5% | 15.1% | +49.2% | YES (worst) |
| arith_sort | +10.5% | 8.2% | +24.3% | sometimes |
| arith_repeat | -8.4% | 4.1% | -4.7% | no |
| arith_parity | +0.9% | 7.1% | +7.6% | no |
| reverse_repeat | -15.8% | 5.1% | -7.1% | no |
| reverse_sort | -14.1% | 5.8% | -6.8% | no |
| reverse_parity | -18.2% | 7.0% | -5.9% | no |
| repeat_sort | +17.4% | 6.4% | +27.8% | sometimes |
| repeat_parity | +11.8% | 11.8% | +24.1% | sometimes |
| sort_parity | -15.9% | 3.6% | -9.8% | no |

### Per-Type Single-Expert Gaps (hash-ring scenario)

| Cross-Domain Type | Mean | Std | Max | Exceeds 20%? |
|-------------------|------|-----|-----|---------------|
| arith_reverse | -14.1% | 6.7% | -3.5% | no |
| arith_sort | -15.6% | 4.9% | -8.3% | no |
| arith_repeat | -14.1% | 2.9% | -9.4% | no |
| arith_parity | +21.3% | 6.3% | +30.4% | YES |
| reverse_repeat | -25.4% | 5.3% | -20.6% | no (better) |
| reverse_sort | -18.1% | 4.8% | -10.1% | no |
| reverse_parity | -5.2% | 6.8% | +8.0% | no |
| repeat_sort | +1.4% | 7.7% | +12.2% | no |
| repeat_parity | +10.8% | 12.9% | +31.2% | sometimes |
| sort_parity | -10.5% | 5.5% | -0.7% | no |

### Cancellation Artifact (Fix 4)

The aggregate mean gap of -1.0% (multi-expert) is an average across positive and
negative gaps that partially cancel:

- **22/50 trials**: multi-expert is WORSE than base (mean of positive gaps: +15.0%)
- **28/50 trials**: multi-expert is BETTER than base (mean of negative gaps: -13.6%)
- **7/50 trials** exceed the 20% threshold on individual type/seed combinations
- **Max single-trial gap: +49.2%** (arith_reverse, seed 142)

The aggregate passes K1 because negative gaps (where composition helps) outnumber
and partially offset the positive gaps. The max gap and per-type statistics give a
more honest picture of the variance.

### Routing Analysis

Best single expert is from the involved domains in 92% of trials (46/50).
The 4 routing errors all occur on `reverse_parity` -- the parity expert sometimes
outperforms both involved experts on this cross-domain type, likely because the
parity computation dominates the loss.

## Key Findings

1. **Oracle 2-expert composition passes K1 on aggregate** (-1.0% mean), but individual
   cross-domain types can exceed 20% (arith_reverse: +21.5% mean, +49.2% max).
   The mechanism is directionally sound but not uniformly reliable across all pairs.

2. **Single-expert composition (hash-ring scenario) also passes K1** (-7.0% mean).
   Surprisingly, the best single expert often handles cross-domain queries better
   than multi-expert composition. This is because multi-expert averaging dilutes
   the strong expert's signal when the second expert adds noise rather than
   complementary knowledge.

3. **Cross-domain queries are handled by top-k=2 routing with oracle selection.**
   This requires an extension to hash-ring (or a different router). Hash-ring
   routes to one expert, which is sufficient on average (-7.0%) but fails on
   specific types (arith_parity: +21.3%).

4. **GS orthogonalization is not evaluated in this revision** -- the focus is on
   the composition mechanism itself and honest gap reporting. See v1 for GS analysis.

## Micro-Scale Limitations

1. **Synthetic cross-domain queries are sequential concatenation, not semantic
   composition.** Real cross-domain queries (e.g., "convert Python to Bash") require
   simultaneous understanding of both domains. Our queries test "can a model handle
   two tasks in sequence?" which is a weaker test.

2. **Untrained base model.** The base model starts with random weights. A pretrained
   base provides substantial language knowledge that aids cross-domain transfer.

3. **Rank-4 at d=32 is severely capacity-constrained.** Signal retention is ~80%.
   At d=4096/r=16, retention is 95%+, so gaps should shrink.

4. **Oracle routing for multi-expert.** We assume correct identification of the
   2 relevant experts. Production systems must solve the routing problem. Hash-ring
   routes to 1 expert, which works on average but not for all types.

5. **High variance.** Per-type stddev ranges from 3.6% to 15.1%. Individual
   type/seed combinations can far exceed the aggregate. N=5 seeds is better than
   N=3 but still limited for rare-event analysis.

6. **N=5 domains is small.** At N=500, naive merge dilution is 250x worse.
   Multi-expert routing becomes critical, and the routing problem hardens.

## What Would Kill This

**At micro scale (would falsify if observed):**
- Multi-expert mean gap >20% across all types [NOT OBSERVED: -1.0%]
- Routing error rate >20% [NOT OBSERVED: 8.0%]
- Note: Individual types DO exceed 20% (arith_reverse: +21.5% mean).
  This is reported honestly but does not kill the aggregate hypothesis.

**At macro scale (would need validation):**
- Real cross-domain queries require semantic-level expert interaction that
  weight-space composition cannot achieve
- Multi-expert routing overhead makes k>1 impractical at production latency
- Expert specialization at macro scale creates harder cross-domain boundaries

## Conclusions

1. **Weight-space composition of relevant experts handles cross-domain queries
   on aggregate**, with mean gap -1.0% (multi-expert) and -7.0% (single expert).
   Both pass the 20% kill threshold.

2. **The aggregate hides substantial per-type variance.** arith_reverse has a mean
   gap of +21.5% with max +49.2%. The mechanism is not uniformly reliable.

3. **The hash-ring single-expert scenario works better than expected.** For most
   cross-domain types, the best involved expert handles the query well on its own.
   The exception is arith_parity (+21.3%), where neither arithmetic nor parity
   expert adequately covers the joint task.

4. **Multi-expert oracle composition is NOT equivalent to hash-ring routing.**
   Hash-ring routes to one expert; multi-expert requires knowing which 2 experts
   are relevant and composing them. A production deployment needs either top-k
   routing (extending hash-ring) or a learned router for cross-domain queries.

## Date
2026-03-14. Status: **supported** (all three kill criteria pass on aggregate;
per-type variance acknowledged; hash-ring limitation qualified).
