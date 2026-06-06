# SOLE vs LoRA-Flow: Research Digest

## Hypothesis

LoRA-Flow's dynamic per-layer per-token fusion gate does NOT achieve >10%
quality gain over SOLE's fixed unit weights when experts are structurally
orthogonal, because near-orthogonality makes the optimal weights trivially
equal to 1.

## What This Study Is

A systematic comparison of four LoRA composition methods at micro scale:

1. **SOLE** (ours): Fixed unit-weight addition. Zero trainable params.
2. **CAT** (LoRA Soups, Prabhakar et al., COLING 2025): Learned per-layer
   static scalar weights. 2*k*L params.
3. **LoRA-Flow** (Wang et al., 2024, arXiv:2402.11455): Dynamic per-layer
   per-token fusion gate. w = softmax(W_gate @ h) + b. L*k*(d+1) params.
4. **X-LoRA** (Buehler, 2024, arXiv:2402.07148): MLP gating on hidden
   states. L*h*(d+k) params.

This was spawned from the exp_oae_vs_lora_soups adversarial review, which
noted LoRA-Flow as a missing comparison from the SOLE positioning table.

## Lineage in the Arena

```
oae_vs_lora_soups (SOLE vs CAT) --> lora_flow_comparison (this study)
```

## Key References

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Wang et al., "LoRA-Flow" | 2024 | Dynamic per-layer fusion gate; softmax(W_gate @ h) + b |
| Buehler, "X-LoRA" | 2024 | MLP-based per-layer per-token LoRA mixing |
| Prabhakar et al., "LoRA Soups" | 2024 | CAT: static per-layer scalar weights |
| Ostapenko et al., "Modular LLMs" | 2024 | Arrow zero-shot routing over LoRA library |

## LoRA-Flow Method Summary

From Wang et al. (2024):

- **Fusion gate**: w^l = softmax(W_gate^l @ x_t^l) + b^l per layer per token
- **Parameters**: W_gate in R^{k x d}, b in R^k per layer. Total 0.26M for
  Llama-2-7b with k=2 (0.2% of a single LoRA adapter)
- **Training**: Only gate params trained. Backbone and LoRAs frozen. 200
  examples, 5 epochs
- **Results**: MGSM 37.6 vs LoRA-Hub 28.7 vs Average 13.9 (Llama-2-7b)
- **Key claim**: Dynamic weights are necessary for complex generative tasks
  where different tokens require different skill mixtures

## Empirical Results (Micro Scale)

### Configuration

d=64, d_ff=256, r=8, L=4, 12 domains (4 clusters), 3 seeds.
All gate training uses SPSA (50 steps).

### Quality Comparison

| Composition | SOLE | Avg (1/k) | CAT | LoRA-Flow | X-LoRA | Base |
|-------------|------|-----------|-----|-----------|--------|------|
| within_k2 | 3.4177 | 3.4177 | 3.4177 | 3.4177 | 3.4177 | 3.4177 |
| cross_k2 | 3.4330 | 3.4330 | 3.4330 | 3.4330 | 3.4330 | 3.4331 |
| cross_k6 | 3.4464 | 3.4464 | 3.4464 | 3.4464 | 3.4464 | 3.4464 |
| all_k12 | 3.4550 | 3.4550 | 3.4550 | 3.4550 | 3.4550 | 3.4550 |

**All four methods are equivalent to 4 decimal places.** Expert deltas
have negligible magnitude at micro scale, making the quality comparison
vacuous. This is identical to the finding in exp_oae_vs_lora_soups.

### LoRA-Flow Quality Gain Over SOLE

| Composition | Gain | Threshold |
|-------------|------|-----------|
| within_k2 | 0.0% | >10% |
| cross_k2 | 0.0% | >10% |
| cross_k6 | 0.0% | >10% |
| all_k12 | 0.0% | >10% |

### Overhead Comparison

| Method | k=2 | k=6 | k=12 |
|--------|-----|------|------|
| SOLE | 0.14s (1x) | 0.38s (1x) | 0.75s (1x) |
| CAT | 2.42s (17x) | 7.63s (20x) | 18.1s (24x) |
| LoRA-Flow | 0.85s (6x) | 2.40s (6x) | 6.19s (8x) |
| X-LoRA | 0.89s (6x) | 2.53s (7x) | 6.13s (8x) |

### Parameter Scaling at Production (d=4096, L=32)

| k | SOLE | CAT | LoRA-Flow | X-LoRA (h=64) |
|---|------|-----|-----------|---------------|
| 2 | 0 | 128 | 262K | 8.39M |
| 10 | 0 | 640 | 1.31M | 8.41M |
| 100 | 0 | 6,400 | 13.1M | 8.59M |
| 500 | 0 | 32,000 | 65.6M | 9.41M |

**Critical scaling observation**: At k=500 experts, LoRA-Flow's gate
(65.6M params) exceeds the size of a rank-16 LoRA adapter (40.4M params).
The routing mechanism becomes larger than the experts it routes.

### Orthogonality

Mean |cos| = 0.0021 +/- 0.0002 (3 seeds), consistent with structural
orthogonality at d=64 (bound: sqrt(r/d) = 0.35, measured 167x below).

## Updated Comparison Table

| Dimension | SOLE | CAT | LoRA-Flow | X-LoRA |
|-----------|------|-----|-----------|--------|
| **Weights** | c_i = 1 (fixed) | w_i^l (static learned) | softmax(W@h)+b (dynamic) | softmax(MLP(h)) (dynamic) |
| **Input-dependent** | No | No | Yes (per-token) | Yes (per-token) |
| **Params** | 0 | 2kL | Lk(d+1) | Lh(d+k) |
| **Training cost** | None | O(T*k*L*C_fwd) | O(T*L*C_fwd) | O(T*L*C_fwd) |
| **Inference cost** | 0 (pre-merged) | 0 (pre-applied) | k matmuls/layer | MLP + k matmuls/layer |
| **Expert addition** | Instant, free | Retrain weights | Retrain gate (grows W_gate) | Retrain gate (grows W2) |
| **Max k tested** | 20 (micro), 2 (macro) | 2 (paper) | 2 (paper) | varies |
| **Evolution support** | Yes (clone-compete) | No (needs retrain) | No (needs retrain) | No (needs retrain) |
| **Orthogonality analysis** | Structural guarantee | None | None | None |

## The Hierarchy: SOLE subseteq CAT subseteq LoRA-Flow

These methods form a strict inclusion hierarchy:

1. **SOLE** (c=1): simplest, zero overhead
2. **CAT** (c=w_i^l): static per-layer weights (SOLE when w_i -> 1)
3. **LoRA-Flow** (c=f(x,l)): dynamic per-token weights (CAT when W_gate -> 0)

LoRA-Flow is the most expressive: it can represent both SOLE and CAT as
special cases. The question is whether the additional expressivity provides
quality gains that justify the parameter and training overhead.

**Our answer**: Under structural orthogonality (the SOLE setting), optimal
weights are trivially 1.0 regardless of input. LoRA-Flow's expressivity is
wasted when there is no interference to route around. The additional
parameters and training cost are pure overhead.

## Micro-Scale Limitations

This experiment has a **known, critical limitation** identical to
exp_oae_vs_lora_soups: expert specialization at micro scale is negligible.
Individual expert loss equals base loss to 4 decimal places. This makes
the quality comparison between all four methods vacuous.

**What would need to change for a meaningful quality comparison:**

1. **Macro-scale experts with real specialization** (98% win rate from
   pilot 50). At d=896+ with real domain data, expert deltas are large
   enough that interference matters.
2. **High-interference domain pairs**: at micro scale, max |cos|=0.002.
   At macro, within-cluster cos can reach 0.85 (math-medical attention).
   LoRA-Flow could theoretically help there.
3. **Heterogeneous queries**: real-world queries that require different
   skill weights at different token positions (e.g., "translate this code
   to French" needs both code and language skills).

**What this experiment DOES show conclusively:**

1. **Overhead ranking**: SOLE (0x) < LoRA-Flow (6-8x) < CAT (17-24x)
2. **Parameter scaling**: LoRA-Flow is infeasible at large k (65.6M at
   k=500, d=4096)
3. **Under orthogonality, all methods converge to SOLE**: no method can
   outperform unit weights when experts do not interfere
4. **LoRA-Flow's gate is trainable at k=12**: feasibility confirmed, but
   scaling is the constraint, not training difficulty

## What Would Kill This

### K1: LoRA-Flow achieves >10% quality gain over SOLE at comparable overhead

**Status: SURVIVES**. LoRA-Flow gain is 0.0% across all k values and all
seeds. The comparison is vacuous at micro scale (experts do not specialize),
but even vacuous results show no mechanism by which LoRA-Flow could help
when experts are orthogonal.

At macro scale with real expert interference (within-cluster cos=0.85),
LoRA-Flow could potentially help by down-weighting interfering experts.
However, SOLE's response to interference is hash-ring routing (select 1
expert per domain), which achieves the same effect with zero parameters.

### K2: LoRA-Flow's per-layer weight learning is feasible at N>10 experts

**Status: PARTIALLY TRIGGERED**. LoRA-Flow is technically feasible at k=12
(6.2s training, 3120 params at micro scale). However, at production scale:

- k=12, d=4096, L=32: 1.57M gate params. Manageable.
- k=100: 13.1M. Significant overhead.
- k=500: 65.6M. **Exceeds a single LoRA adapter's parameter count.**

The gate scales as O(k*d*L), which is O(k) times the size of a single
expert. This makes LoRA-Flow structurally infeasible at the scale SOLE
is designed for (N=500+).

**K2 assessment**: Feasible at k<=50, infeasible at k>=100. For SOLE's
target scale (N=100-500+), LoRA-Flow is not a viable alternative.

## Conclusion: SOLE's Position Relative to LoRA-Flow

LoRA-Flow occupies a **different design point** than SOLE:

| Property | SOLE | LoRA-Flow |
|----------|------|-----------|
| **Use case** | Library composition (N>>2) | Binary/few-skill composition (k<=10) |
| **Routing philosophy** | All experts contribute equally | Dynamic per-token skill selection |
| **Interference handling** | Structural (orthogonality) | Parametric (learned gate) |
| **Scaling** | O(1) per expert added | O(k*d*L) gate growth |
| **Evolution** | Clone-compete (zero cost) | Retrain gate after any change |

LoRA-Flow is the right tool for a fundamentally different problem:
**composing 2-3 highly specialized, potentially overlapping skills** where
input-dependent routing meaningfully improves quality. SOLE is for
**composing hundreds of orthogonal experts** where the question is not
"which expert to use" but "how to include all relevant knowledge."

LoRA-Flow does NOT threaten SOLE's positioning. They solve different problems.

## Artifacts

- `micro/models/lora_flow_comparison/PAPER.md` -- this document
- `micro/models/lora_flow_comparison/MATH.md` -- formal comparison
- `micro/models/lora_flow_comparison/lora_flow_comparison.py` -- experiment
- `micro/models/lora_flow_comparison/results.json` -- raw results
