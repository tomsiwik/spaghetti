# MoTE-SOLE Architecture: Research Digest

## Hypothesis

MoTE-style architecture (FP16 shared base + ternary routed experts with top-k
gating) outperforms equal-weight 1/N composition on domain-specific PPL.

**Falsifiable**: If routed composition produces worse mean PPL than equal-weight
composition, or ternary experts lose >10% quality vs FP16, the hypothesis is killed.

## What This Model Is

A micro-scale (d=64, r=4, L=2) implementation of the MoTE (Mixture of Ternary
Experts) architecture adapted for SOLE. The system has three components:

1. **Shared expert**: Frozen FP16 base model (post-quantized to ternary)
2. **Routed experts**: 5 ternary LoRA domain adapters trained with QAT+STE
3. **Router**: Linear layer mapping sequence-level hidden states to expert
   selection probabilities with top-k gating and load-balancing loss

The router is trained on mixed-domain data after expert training, learning
which expert(s) to activate for each input sequence.

## Key References

- MoTE (arXiv 2506.14435): Mixture of Ternary Experts for memory-efficient LMMs
- Switch Transformer (Fedus et al. 2022): load-balancing auxiliary loss
- exp_bitnet_ternary_adapter_composition (this project): ternary adapters compose
  4.4% better than FP16 under equal-weight composition
- exp_bitnet_composition_stability (this project): ternary base composition ratio = 0.63

## Empirical Results

### Configuration

| Parameter | Value |
|-----------|-------|
| d (hidden dim) | 64 |
| r (LoRA rank) | 4 |
| L (layers) | 2 |
| N (experts) | 5 |
| Seeds | 42, 123, 314 |
| Domains | arithmetic, reverse, repeat, sort, parity |
| Base training | 20 epochs on mixed data |
| Expert training | 20 epochs per domain (QAT for ternary) |
| Router training | 15 epochs on mixed data, k=2, alpha_balance=0.01 |
| Total runtime | 1184s (~20 min) |

### Method Comparison (mean PPL across 5 domains, 3 seeds)

| Method | Mean PPL | Std |
|--------|----------|-----|
| Base (ternary, no adapters) | 9.18 | 0.25 |
| **Routed k=1** | **7.81** | **4.82** |
| Routed k=2 | 6.53 | 3.06 |
| Routed k=3 | 6.05 | 2.43 |
| FP16 equal-weight | 5.85 | 0.18 |
| **Ternary equal-weight** | **5.49** | **0.47** |
| Oracle (best single expert) | 3.15 | 0.07 |

**Key finding: Routed composition is WORSE than equal-weight composition on mean PPL.**
Even at k=3 (selecting 3 of 5 experts), routed PPL (6.05) exceeds ternary
equal-weight (5.49) by 10.2%. The variance is also much higher (std 2.43 vs 0.47).

### Per-Domain Breakdown (mean PPL, 3 seeds)

| Domain | Base | Single | EqWt | k=1 | k=2 | Oracle |
|--------|------|--------|------|-----|-----|--------|
| arithmetic | 12.4 | 4.3 | 9.3 | 21.2 (+128%) | 15.1 (+62%) | 4.3 |
| reverse | 10.3 | 4.3 | 5.4 | 5.4 (+0%) | 5.5 (+2%) | 4.3 |
| repeat | 8.0 | 1.8 | 5.0 | 2.2 (-56%) | 2.7 (-46%) | 1.8 |
| sort | 9.7 | 3.6 | 4.9 | 5.0 (+2%) | 4.8 (-2%) | 3.6 |
| parity | 5.4 | 1.8 | 2.9 | 5.3 (+84%) | 4.6 (+60%) | 1.8 |

The router succeeds on "repeat" (-56% vs equal-weight) but catastrophically
fails on "arithmetic" (+128%) and "parity" (+84%). When the router misroutes,
it concentrates weight on the WRONG expert, producing worse results than
distributing weight uniformly.

### Routing Accuracy

| k | Accuracy (mean +/- std) |
|---|-------------------------|
| 1 | 0.501 +/- 0.206 |
| 2 | 0.617 +/- 0.171 |
| 3 | 0.721 +/- 0.193 |

The router achieves only 50.1% accuracy at k=1 (chance = 20%), which is
above random but far from reliable. The high variance across seeds (std=0.206)
indicates the router does not consistently learn domain features.

### Expert Interference (Cosine Similarities)

| Pair | |cos| |
|------|-------|
| reverse-sort | 0.796 |
| reverse-repeat | 0.463 |
| repeat-sort | 0.468 |
| arithmetic-repeat | 0.143 |
| sort-parity | 0.210 |

High cosine similarity between reverse-sort (0.796) and within the
{reverse, repeat, sort} cluster explains why routing within this cluster
is particularly difficult -- the experts are not sufficiently differentiated
for the router to distinguish.

### Kill Criteria Assessment

| Criterion | Result | Verdict |
|-----------|--------|---------|
| K1: Routed beats equal-weight on >50% of domains | 3.0/5 domains (pass rate 2/3), BUT mean PPL is 10-42% worse | **BORDERLINE** |
| K2: Router training < 1hr | 38.6s mean, 45.6s max | **PASS** |
| K3: Ternary/FP16 individual ratio < 1.10 | 1.135 +/- 0.025 (pass rate 0/3) | **FAIL** |

**Overall verdict: KILLED.**

K1 is misleading: while the router beats equal-weight on 3/5 domains by
count, the magnitude of failures (+128% on arithmetic, +84% on parity)
far exceeds the magnitude of wins (-56% on repeat). Mean PPL is worse
across all k values. The router's value is negative in expectation.

K3 definitively fails: ternary experts are 13.5% worse than FP16
individually, exceeding the 10% threshold. This is consistent with
prior exp_bitnet_ternary_adapter_composition which found 2.6% individual
degradation at r=4, d=64 -- but that experiment used a ternary base
for BOTH conditions. Here we compare ternary adapters on ternary base
vs FP16 adapters on ternary base, and the gap is larger (13.5%).

## Why Routing Fails at This Scale

1. **Domain confusion at d=64**: With only 64-dimensional hidden states,
   the router has insufficient information to reliably distinguish
   domains. The 50% accuracy at k=1 means half of all routing decisions
   are wrong.

2. **Asymmetric failure modes**: When equal-weight composition is wrong
   on a domain, the penalty is dilution (1/5 of each expert). When routing
   is wrong, the penalty is concentration on the WRONG expert. The
   downside of misrouting is unbounded; the downside of dilution is bounded.

3. **High expert overlap**: The cosine similarity between reverse-sort
   (0.796) means these experts are nearly redundant. The router cannot
   meaningfully distinguish between them, and selecting the wrong one
   of a near-identical pair wastes the routing budget.

4. **Small N amplifies errors**: With N=5, each routing decision affects
   20% of the expert budget. At N=50+, individual misrouting decisions
   would have 2% impact -- routing may become viable at scale.

## Limitations

1. **Micro scale only**: d=64, r=4, N=5 is far from production (d=4096, r=16, N=50+).
   The router may work better with richer hidden representations.

2. **Synthetic toy domains**: Character-level arithmetic, reversal, etc. Real
   domains (math, code, medical) have more distinctive features that may
   be easier to route.

3. **Sequence-level routing**: We route per-sequence (average hidden states),
   not per-token. Token-level routing might be more precise but more expensive.

4. **Router architecture**: A single linear layer may be too simple. MLP or
   attention-based routers might perform better.

5. **Post-quantized ternary base**: Native BitNet training (b1.58) may produce
   different base representations that are more amenable to routing.

## What Would Kill This (at macro scale)

Even with better routing accuracy at macro scale, the following would kill MoTE-SOLE:

- Routing overhead (compute or latency) exceeds the quality gain vs equal-weight
- Ternary experts still show >10% individual quality loss vs FP16 at d=4096
- Load-balancing loss causes mode collapse or expert underutilization
- The router requires per-domain labeled data that defeats the purpose of
  zero-shot routing

## Relationship to Prior Findings

- **exp_bitnet_ternary_adapter_composition** (SUPPORTED): Found ternary adapters
  compose 4.4% BETTER than FP16 under equal-weight. That finding used equal-weight
  composition, not routing. This experiment confirms that equal-weight remains
  the better strategy at N=5 micro scale.

- **exp_bitnet_composition_stability** (SUPPORTED): Composition ratio 0.63 on
  ternary base. This experiment's equal-weight results are consistent (ternary
  equal-weight PPL 5.49 vs ternary base 9.18, ratio 0.60).

- **content_aware_routing** (KILLED at micro): Oracle routing produced identical
  NTP loss to random routing at d=64. This experiment confirms the same pattern:
  the router cannot extract sufficient domain signal from d=64 hidden states.

## Conclusion

At micro scale, learned routing does not justify its overhead compared to
equal-weight composition. The fundamental issue is that d=64 hidden states
do not carry enough domain-discriminative information for a linear router to
make reliable decisions. When routing is correct, it approaches oracle quality;
when wrong, it can be catastrophically worse than equal-weight. The net effect
is negative.

This does NOT rule out MoTE-SOLE at macro scale. At d=4096 with real domain
adapters and richer hidden representations, routing may become viable. However,
the prior content_aware_routing kill (also at micro) and the present results
suggest that routing is a macro-scale mechanism, not a micro-scale one.
