# Cross-Domain Dilution vs Top-K: Research Digest

## Hypothesis

Multi-expert composition quality improves with selective relevance-weighted
composition vs equal-weight averaging of all involved experts. Specifically:
a 10-example PPL probe can approximate oracle weighting (r > 0.9) and reduce
worst-case per-type degradation below 20%.

## What This Model Is

This experiment extends the proven cross_domain_composition experiment, which
found that equal-weight 2-expert composition has a mean gap of -0.6% but a
worst-case of +24.3% (arith_reverse). Single-expert often beat multi-expert
because the second expert added noise rather than complementary signal.

We test 6 composition strategies that weight experts differently:

1. **equal_weight** -- (Delta_i + Delta_j) / 2, the parent baseline
2. **activation_weighted** -- weight by ||Delta_k @ h_query|| (weight-space signal)
3. **logit_diff_weighted** -- weight by how much each expert changes predictions (no labels)
4. **ppl_probe_weighted** -- weight by PPL on a 10-example probe buffer (cheap oracle)
5. **loss_weighted** -- weight by full-dataset PPL (oracle upper bound)
6. **top1_oracle** -- best single expert by full loss (oracle)

Plus **top1_logit_diff** -- select single best expert by logit change magnitude.

## Lineage in the Arena

```
exp_gram_schmidt_composition (proven)
  |
  +-- exp_cross_domain_composition (proven, mean gap -1.0%)
        |
        +-- exp_cross_domain_dilution_vs_k (THIS EXPERIMENT)
```

## Key References

- Wang et al., "LoRA-Flow: Dynamic LoRA Fusion", 2024 (arXiv:2402.11455) -- per-token per-layer gates
- Prabhakar et al., "LoRA Soups", COLING 2025 -- CAT learned scalar weights
- Huang et al., "LoRAHub", 2023 (arXiv:2307.13269) -- gradient-free few-shot composition
- Ilharco et al., "Editing Models with Task Arithmetic", ICLR 2023

## Empirical Results

### Strategy Comparison (5 seeds, 10 cross-domain types, N=50 trials each)

| Strategy | Mean Gap | Std | Max | Imp vs EW | Labels? | Production? |
|----------|----------|-----|-----|-----------|---------|-------------|
| equal_weight | -0.60% | 18.80% | +34.1% | baseline | No | Yes |
| activation_weighted | -0.71% | 18.66% | +32.6% | +0.11pp | No | Yes |
| logit_diff_weighted | +1.59% | 21.84% | +62.5% | -2.19pp | No | Yes |
| **ppl_probe_weighted** | **-9.94%** | **13.72%** | **+23.4%** | **+9.34pp** | **No*** | **Yes** |
| loss_weighted | -10.06% | 13.69% | +23.8% | +9.46pp | Yes | No |
| top1_oracle | -5.85% | 16.65% | +37.4% | +5.25pp | Yes | No |
| top1_logit_diff | +22.80% | 40.01% | +140.9% | -23.40pp | No | No |

*PPL probe uses NTP loss on 10 representative examples -- no explicit labels needed.

### Proxy-Oracle Weight Correlation

| Proxy Method | Pearson r | Weight Variance |
|-------------|-----------|-----------------|
| activation | 0.023 | near-zero (0.007) |
| logit_diff | -0.245 | low (0.079) |
| **ppl_probe** | **0.990** | matches oracle (0.207) |

### Worst-Case Resolution (arith_reverse)

| Strategy | Mean Gap | Max Gap | Exceeds 20%? |
|----------|----------|---------|--------------|
| equal_weight | +24.3% | +34.1% | YES |
| ppl_probe_weighted | -8.5% | +6.3% | NO |
| loss_weighted | -8.5% | +6.3% | NO |

### Kill Criteria Assessment

| Criterion | Metric | Threshold | Measured | Status |
|-----------|--------|-----------|----------|--------|
| K1 | Improvement vs equal_weight | >2pp | +9.46pp (loss), +9.34pp (probe) | **PASS** |
| K2 | arith_reverse gap | <20% | -8.5% (loss/probe) | **PASS** |

Both kill criteria pass decisively.

## Key Findings

1. **Weight-space relevance signals are useless at micro scale.** Activation
   magnitude (r=0.023) and logit difference (r=-0.245) do not predict which
   expert is better for a given query. The activation scores have near-zero
   variance (std=0.007), meaning all experts look equally activated.

2. **PPL on a 10-example probe buffer is a near-perfect oracle proxy.** With
   r=0.990 correlation and only 0.12pp gap in performance vs the full-dataset
   oracle, a small probe buffer is sufficient for production-quality weighting.
   This is analogous to LoRAHub's few-shot approach but even cheaper (no
   gradient computation, just K+1 forward passes on 10 examples).

3. **Weighted composition completely resolves the worst-case dilution.** The
   arith_reverse pair goes from +24.3% degradation to -8.5% improvement --
   a 33pp swing. The mechanism works because the probe identifies that the
   arithmetic expert should receive ~70% weight for arith_reverse queries.

4. **Top-1 oracle is NOT always best.** While top-1 gives the best mean
   improvement for some types (arith_reverse: -7.9%), it is catastrophically
   worse for types where both experts contribute (arith_parity: +37.4% with
   top-1 vs +0.9% with loss_weighted). Smooth weighting > hard selection.

5. **Logit difference is anti-correlated with quality.** Experts that change
   predictions more are often the WRONG experts. This is because at micro
   scale, the noisiest expert makes the largest perturbation. Do NOT use
   logit difference as a relevance signal.

## Production Implications for SOLE

The ppl_probe_weighted strategy is directly production-viable:

1. **Probe buffer**: maintain a rolling buffer of 10-20 recent queries per
   domain cluster. Updated continuously from serving traffic.

2. **Routing**: when a cross-domain query arrives, score each candidate expert
   by PPL on the probe buffer (K+1 forward passes, ~K * 0.001s per expert).

3. **Composition**: apply softmax weights from probe scores to expert deltas
   before merging.

4. **Cost**: O(K) forward passes on 10 examples. At K=2, this is ~0.002s
   additional latency, well within the <5% overhead budget.

This is cheaper than LoRA-Flow (which requires training a gate with ~200
examples per composition) and scales better than CAT (which requires per-layer
weight optimization).

## Micro-Scale Limitations

1. **Weight-space signals may work better at macro scale.** At d=4096/r=16,
   expert deltas carry more distinctive information. The activation and logit
   signals may become discriminative when experts are truly specialized (the
   pilot-50 experts have 98% win rate, unlike micro experts with ~0%
   specialization).

2. **Synthetic cross-domain queries.** Sequential chaining ("compute then
   reverse") is a weaker test than real semantic transfer ("convert Python
   to Bash"). Weighted composition may help less when both experts are
   genuinely needed simultaneously.

3. **Random-initialized base.** Pretrained bases provide substantial prior
   knowledge that may change expert-query interaction patterns.

4. **K=2 only.** At K>2, the dilution problem worsens and the benefit of
   selective weighting should increase, but the probe cost also increases
   linearly.

5. **PPL probe assumes representative queries.** In production, the probe
   buffer contains recent queries for the domain. If the query distribution
   shifts, the probe may become stale. A staleness detector (e.g., detecting
   when probe variance exceeds a threshold) could mitigate this.

6. **d=32, r=4 is severely capacity-constrained.** Signal retention ~80%.
   At d=4096/r=16, retention is 95%+.

## What Would Kill This

**At micro scale (would falsify if observed):**
- All weighted strategies show <2% improvement over equal weight [NOT OBSERVED: +9.46pp]
- arith_reverse still >20% gap with best strategy [NOT OBSERVED: -8.5%]

**At macro scale (would need validation):**
- PPL probe fails with real expert specialization (probe buffer not
  representative of query distribution)
- Weight-space signals (activation, logit_diff) become sufficient,
  making the probe buffer unnecessary
- K+1 forward passes per composition exceeds latency budget at scale
- Cross-domain queries at macro are fundamentally different from
  sequential chaining and require genuine multi-expert activation
  (making equal weight better than selective)

## Date
2026-03-15. Status: **proven** (both kill criteria pass decisively;
ppl_probe_weighted identified as production-viable near-oracle proxy).
