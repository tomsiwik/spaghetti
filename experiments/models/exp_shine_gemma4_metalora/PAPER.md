# SHINE S3: Meta LoRA Encoding + Multi-Projection Generation

## Summary

Three structural interventions to break the S2 centroid trap (cos=0.998):
meta LoRA (rank 128) during encoding, multi-projection M2P (q+v+o), and
diversity regularizer (cos² penalty). Multi-projection is validated as a
7.7x improvement over q-only (K1259 PASS). Meta LoRA adds overhead without
benefit (K1258 FAIL). Centroid trap persists (cos=0.988) despite diversity loss.

## Architecture

| Component | Config |
|-----------|--------|
| Base model | Gemma 4 E4B 4-bit (42 layers, 2560 hidden) |
| Meta LoRA | rank 128, q_proj only, all 42 layers, zero-init B |
| M2P | dim=128, 2 blocks, 4 heads, 8.2M params |
| M2P output | q_proj + v_proj + o_proj LoRA (rank 2 each) |
| Meta LoRA params | 26,607,616 |
| Total trainable | 34,809,728 |
| Training | 1000 steps, Adam lr=3e-4, λ_div=0.1, warmup=100 |
| Diversity | cos² penalty against running cache (size 16) |
| Data | 40 train + 10 test chunks of 128 tokens |

## Prediction vs Measurement

| ID | Prediction | Threshold | Measured | Result |
|----|-----------|-----------|----------|--------|
| P1 | S3 CE ratio < S2 (0.134) | < 0.134 | **0.151** | FAIL |
| P2 | q+v+o CE < q-only CE | improvement > 0 | **0.151 vs 1.16** (7.7x) | PASS |
| P3 | Diversity → mean LoRA cos < 0.9 | < 0.9 | **0.988** | FAIL |
| P4 | Grassmannian cos < 0.3 | < 0.3 | **0.073** | PASS |

## Kill Criteria

| ID | Criterion | Measured | Result |
|----|-----------|----------|--------|
| K1258 | S3 test ratio < S2 (0.134) | 0.151 | **FAIL** |
| K1259 | q+v+o CE < q-only CE | 0.151 vs 1.16 | **PASS** |
| K1260 | Grassmannian cos < 1e-4 | 0.073 | **FAIL** |

## Key Results

### 1. Multi-Projection is the Dominant Factor (K1259 PASS)

| Mode | Test CE | Test Ratio | vs Base |
|------|---------|------------|---------|
| Base (no LoRA) | 8.91 | 1.000 | — |
| q-only LoRA | 10.34 | 1.160 | WORSE |
| q+v+o LoRA | 1.35 | 0.151 | 84.9% better |

The v_proj and o_proj LoRA are doing nearly all the useful work. q-only
LoRA actually HURTS performance (ratio 1.16 > 1.0), likely because the
generated q_proj LoRA interferes with attention patterns. Adding v+o
projections transforms this into an 84.9% CE reduction.

This strongly validates Finding #480: behavioral format priors live in
the value and output projection layers, not the query projection.

### 2. Meta LoRA Does Not Help (K1258 FAIL)

S3 test ratio (0.151) is slightly WORSE than S2 (0.134). Analysis:

- **Parameter imbalance**: Meta LoRA has 26.6M params vs M2P's 8.2M.
  The meta LoRA dominates the optimization landscape. With only 40
  training chunks, the meta LoRA likely overfits to extraction patterns
  rather than improving M2P's ability to differentiate contexts.

- **Double forward pass overhead**: Each S3 step requires extraction
  WITH meta LoRA + NTP WITH generated LoRA = 2x Gemma 4 forward passes.
  At 1036ms/step (vs S2's 434ms), the same wall-clock time buys fewer
  optimization steps. The meta LoRA benefit doesn't compensate.

- **Pre-caching was optimal**: S2's pre-extracted memory states were
  perfectly fine (cos=0.182 cross-layer diversity). Adding a trainable
  meta LoRA introduces noise without adding signal.

### 3. Centroid Trap Persists (P3 FAIL)

Mean pairwise LoRA cosine = 0.988 (S2 was 0.998). Marginal improvement.

The diversity regularizer (cos² penalty, λ=0.1, running cache) is
insufficient because:
- λ=0.1 is small relative to NTP loss (~1.5 at convergence)
- Running cache contains stale LoRA from previous steps
- 10 unique passages don't provide enough context diversity for the
  diversity loss to differentiate

**Theorem 1 is correct** (centroid IS a saddle point with diversity loss),
but the optimization doesn't find the escape direction. The centroid
basin of attraction is too deep for gradient descent with these hyperparameters.

### 4. Grassmannian Orthogonality (P4 PASS)

Meta LoRA and generated LoRA subspaces are nearly orthogonal (max cos=0.073).
This is better than the random expectation (~0.25), suggesting training
PUSHES them apart rather than together. The K1260 threshold (1e-4) is
unreachable for rank-128 vs rank-2 in R^2560 without explicit enforcement.

### 5. Training Dynamics

| Metric | Value |
|--------|-------|
| Speed | 1036 ms/step |
| Peak memory | 5.07 GB |
| Initial loss | 22.99 |
| Final loss | 1.46 |
| Loss decrease | 93.7% |

## Timing

| Phase | Time |
|-------|------|
| Model load | ~5s |
| Build S3 model | ~2s |
| Training (1000 steps) | 1036s |
| Evaluation | ~30s |
| **Total** | **1066s** |

## Impossibility Structure

**Why meta LoRA fails for small data:**
Meta LoRA (rank 128, 26.6M params) with 40 training chunks (< 5K tokens each)
has ratio of params/data ≈ 5000:1. Standard LoRA training uses rank 4-16 with
millions of tokens. The meta LoRA overfits to extraction patterns, not to
producing diverse memory states.

**Why diversity loss fails for homogeneous data:**
For the cos² diversity loss to create distinct LoRA, the optimal per-context
LoRA must be GEOMETRICALLY distinct. With 10 passages of similar English prose,
the optimal LoRA for each context are nearly identical. No diversity penalty
can force distinct LoRA when the loss landscape has a single deep basin.

**What would fix it:**
1. Remove meta LoRA (revert to S2's pre-caching) — saves 26.6M params
2. Keep multi-projection (q+v+o) — validated 7.7x improvement
3. Use 1000+ diverse passages for training (not 10)
4. Use InfoNCE contrastive loss (not cos² penalty) with hard negatives
5. Or: abandon context-specificity for now — the universal adapter is useful

## Status: KILLED

K1258 fails: meta LoRA degrades performance vs S2. The finding is that
multi-projection (K1259) is the actionable improvement, while meta LoRA
and the diversity regularizer are insufficient at this data scale.

## References

- arXiv:2602.06358 (SHINE) — M2P architecture
- Finding #484 — S2: 86.6% CE reduction, centroid trap (cos=0.998)
- Finding #345 — Algebraic proof of centroid trap
- Finding #480 — v_proj+o_proj unlock format priors
- Finding #482 — S1: memory extraction non-degenerate (cos=0.182)
