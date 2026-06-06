# Unified Routing Pipeline: Research Digest

## Hypothesis

Combining entropy gating (skip confident tokens) with per-adapter routing heads
(select top-2 adapters for uncertain tokens) into a single pipeline will beat
the best individual routing method (PPL 6.42) while saving >50% compute.

**Result: KILLED.** Both kill criteria fail.

## What This Model Is

A two-stage routing pipeline that processes each input sequence as follows:

1. **Stage 1 (Entropy Gate):** Run a base model forward pass, compute Shannon
   entropy of the output distribution per token. If all tokens have entropy
   below the Otsu threshold (tau=2.10), use base output and skip routing.

2. **Stage 2 (Routing Heads + Composition):** For sequences with any uncertain
   token, extract hidden states, run N=5 per-adapter routing heads (~82K params
   each), select top-2 adapters by score, score-weight compose them, and run a
   second forward pass with the composed adapter.

The pipeline reuses proven components from two prior experiments:
- Entropy gating: 63% token skip rate at 1.13% PPL cost (SUPPORTED)
- Routing heads: 100% accuracy, 2.32% overhead, PPL 6.42 (SUPPORTED)

## Key References

- exp_entropy_gated_experts (this project, 2026-03-26): entropy distribution analysis
- exp_tiny_routing_heads (this project, 2026-03-26): per-adapter binary classifiers
- exp_molora_per_token_routing + exp_mixed_domain_sequences: killed Gumbel-sigmoid routing

## Empirical Results

### Kill Criteria

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| K1: Unified PPL <= 6.42 | > 6.42 -> KILL | 6.87 (+7.0%) | **FAIL** |
| K2: Overhead < 10% of base | > 10% -> KILL | 222.7% | **FAIL** |

### Per-Domain Unified Pipeline PPL (Otsu, tau=2.10)

| Domain | Base | Uniform | Routed | Unified | Skip Rate | Delta vs Routed |
|--------|------|---------|--------|---------|-----------|-----------------|
| python | 2.74 | 2.51 | 2.22 | 2.36 | 89.0% | +6.3% |
| math | 5.54 | 4.95 | 3.61 | 4.22 | 75.7% | +16.9% |
| medical | 6.96 | 6.19 | 4.75 | 5.06 | 59.1% | +6.5% |
| legal | 21.87 | 20.39 | 16.59 | 17.36 | 37.4% | +4.6% |
| creative | 6.35 | 5.96 | 4.94 | 5.37 | 66.6% | +8.7% |
| **Average** | **8.69** | **8.00** | **6.42** | **6.87** | **63.1%** | **+7.0%** |

### Threshold Sensitivity

| Threshold | Skip Rate | Domain-Avg PPL | vs Routed 6.42 |
|-----------|-----------|----------------|----------------|
| p20 (0.22) | 20.0% | 6.47 | +0.8% |
| p30 (0.64) | 30.0% | 6.51 | +1.4% |
| p40 (1.07) | 39.9% | 6.57 | +2.3% |
| p60 (1.96) | 60.0% | 6.82 | +6.3% |
| **otsu (2.10)** | **63.1%** | **6.87** | **+7.0%** |
| p70 (2.45) | 69.8% | 7.00 | +9.0% |
| p80 (3.02) | 79.9% | 7.17 | +11.7% |

At p20 (20% skip), unified PPL is 6.47 -- still 0.8% worse than pure routing.
No threshold achieves unified PPL <= 6.42. This is a fundamental limitation,
not a threshold tuning issue.

### Timing

| Pipeline | Time/Seq | Overhead vs Base |
|----------|----------|------------------|
| Base-only | 51.6ms | -- |
| Always-compose (1/N) | 51.3ms | -0.6% |
| Unified | 166.5ms | +222.7% |

All 10 python test sequences had at least one uncertain token (max entropy >
2.10), so 100% of sequences triggered the full pipeline (base + routing +
composed forward pass = ~3x cost).

## Why It Failed

### K1 Failure: Entropy Gating Degrades Quality

The fundamental flaw is that entropy gating substitutes base-model output for
"confident" tokens, but the routed composition is ALSO better on confident
tokens. The routing heads already achieve near-oracle quality (6.42 vs 6.41
oracle). Entropy gating can only hurt by replacing routed output with base
output.

Quantitatively: even at p20 (only 20% of tokens use base output), the unified
PPL (6.47) is still worse than pure routing (6.42). This means there is NO
threshold at which entropy gating helps -- it always degrades quality relative
to always-routing.

**Root cause:** Entropy measures base model confidence, not routing necessity.
A token where the base model is confident may still benefit from the specialized
adapter (the adapter may push the already-correct prediction to be more correct,
reducing loss). Entropy < tau does NOT mean "adapter is unnecessary."

### K2 Failure: Two-Pass Architecture is 3x Cost

The unified pipeline requires TWO forward passes through the full model for
any sequence containing uncertain tokens: one for entropy computation, one
with the composed adapter. Since most sequences contain at least one uncertain
token, the effective cost is ~3x base.

This was already known from the entropy gating experiment (S3 FAIL: 2.1x slower)
but the overhead is even worse here because the routing heads and hidden state
extraction add additional cost on top of the second forward pass.

## What Was Learned

1. **Entropy gating and routing heads are complementary in theory but
   incompatible in practice.** The quality gain from routing heads is so large
   (6.42 vs 8.00 uniform) that skipping them on any tokens -- even "easy" ones
   -- degrades the overall result.

2. **The value proposition of entropy gating is WRONG for this use case.**
   Entropy gating saves compute when composition is expensive and routing is
   cheap. But pre-merge composition is already nearly free (0.80% overhead).
   The expensive part is the routing decision itself (hidden states + heads),
   and entropy gating does not eliminate that cost.

3. **Sequence-level gating is too coarse.** Even one uncertain token triggers
   the full pipeline. Per-token gating would require separate forward passes
   per token, which is even more expensive.

4. **The best routing strategy remains: always route with routing heads.**
   PPL 6.42 with 2.32% overhead is the current best. No combination with
   entropy gating improves on this.

## Limitations

- Single model (BitNet-2B-4T) and 5 trivially separable domains
- Routing heads tested on these 5 domains achieve 100% accuracy; harder
  domain mixtures might show different entropy/routing tradeoffs
- Per-token entropy gating (not tested) might perform differently than
  the sequence-level gating used for timing
- The Otsu threshold was derived from a single run and not cross-validated

## What Would Kill This

Already killed. The fundamental issue is that entropy gating and routing heads
are not complementary in this architecture -- routing heads already solve the
problem that entropy gating was designed for (avoiding unnecessary computation
on easy tokens), and they do it better because they also improve quality on
those tokens.

A revival would require:
- A routing cost that scales with N (making skip worthwhile for large N)
- Or a domain mixture where routing is harmful on confident tokens
- Or an architecture where the base pass is reusable (early-exit, not two-pass)
