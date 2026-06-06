# Huffman Macro Routing Skew: Research Digest

## Hypothesis

If production MoE expert utilization distributions are sufficiently non-uniform
(H < 0.95 * log2(L)), Huffman tree routing will reduce average routing depth
by at least 5% compared to balanced binary trees.

**Falsifiable**: If expert utilization follows near-uniform distribution
(H > 0.95 * log2(L)), or if Huffman reshaping provides <5% depth reduction.

---

## What This Experiment Is

An analytical and empirical investigation of whether real-world MoE routing
distributions are skewed enough for Huffman tree routing to help. This is a
**macro-scale hypothesis** tested through:

1. **Distribution modeling**: Simulate expert utilization for DeepSeek-V3
   (256 experts), Mixtral (8 experts), Qwen3-Coder-Next (512 experts), and
   Switch Transformer (128 experts) using Zipf+uniform mixture models
2. **Huffman analysis**: Compute expected depth reduction for each model
3. **Micro empirical**: Measure actual routing distributions from trained
   hierarchical trees at micro scale (L=8, character-level data)
4. **Gradient flow analysis**: Check whether deep Huffman paths (depth 12+)
   cause gradient vanishing
5. **Sensitivity sweep**: Find the critical (alpha, w) boundary where
   Huffman becomes useful

The key question is not "does Huffman work?" (the micro experiment proved it
does) but "does production-scale routing have enough skew for Huffman to matter?"

---

## Lineage in the Arena

```
gpt -> moe -> capsule_moe -> hierarchical_tree -> huffman_tree
                                                      |
                                                      v
                                              huffman_macro_skew
                                              (analytical + empirical)
```

---

## Key References

**Huffman micro experiment** (this project): Validated mechanism works, but
micro-scale uniform routing (H=2.999/3.0) produces 0% benefit. Synthetic
skew validates 12-26% depth reduction. Quality insensitive to tree shape.

**DeepSeek-V3** (Dec 2024): 256 experts, auxiliary-loss-free balancing via
per-expert bias. Explicitly motivated by auxiliary loss "imparing performance."
This is direct evidence that natural routing IS non-uniform -- they need bias
terms precisely because experts develop unequal utilization.

**Switch Transformers** (Fedus et al., 2022): 128 experts with strong balance
loss and capacity factor. Push hard toward uniform but don't fully achieve it.

**Qwen3-Coder-Next** (2025): 512 experts, top-10 routing. Extreme fine-grained
sparsity. At 512 experts, even small deviations from uniform produce large
absolute depth reductions.

---

## Empirical Results

### Part 1: Analytical -- Modeled Production Distributions

We model each production system at multiple skew levels using mixture_zipf
distributions: f_i = w/L + (1-w) * Zipf(alpha), where w controls balance
loss strength and alpha controls natural specialization.

| System | Scenario | L | H/Hmax | E[d] | D_bal | Reduction | Kill? |
|--------|----------|---|--------|------|-------|-----------|-------|
| DeepSeek-V3 | mild (a=0.3, w=0.5) | 256 | 0.997 | 8.00 | 8 | 0.1% | YES |
| DeepSeek-V3 | moderate (a=0.6, w=0.3) | 256 | 0.968 | 7.78 | 8 | 2.7% | YES |
| DeepSeek-V3 | heavy (a=1.0, w=0.1) | 256 | 0.813 | 6.53 | 8 | 18.3% | no |
| Mixtral | balanced (a=0.5, w=0.6) | 8 | 0.995 | 3.00 | 3 | 0.0% | YES |
| Mixtral | specialized (a=1.0, w=0.3) | 8 | 0.936 | 2.85 | 3 | 5.0% | no |
| Qwen3-Next | mild (a=0.3, w=0.5) | 512 | 0.997 | 8.99 | 9 | 0.1% | YES |
| Qwen3-Next | moderate (a=0.5, w=0.3) | 512 | 0.981 | 8.86 | 9 | 1.5% | YES |
| Qwen3-Next | heavy (a=0.8, w=0.1) | 512 | 0.891 | 8.05 | 9 | 10.6% | no |
| Switch | balanced (a=0.2, w=0.7) | 128 | 1.000 | 7.00 | 7 | 0.0% | YES |
| Switch | natural (a=0.5, w=0.4) | 128 | 0.985 | 6.93 | 7 | 1.0% | YES |

**Result**: 3/10 production scenarios survive both kill criteria.
The survivors are all "heavy skew" scenarios with weak balance loss.

### Part 2: Micro Empirical -- Actual Routing Distributions

| Metric | Value | Kill threshold | Verdict |
|--------|-------|----------------|---------|
| Mean H/Hmax (3 seeds, 4 layers) | 0.9972 | > 0.95 | KILLED |
| Mean depth reduction | 0.00% | < 5% | KILLED |

Per-layer frequencies are near-uniform (min 0.091, max 0.162), consistent
with the homogeneous character-level data. This reproduces the original
huffman_tree finding: micro data has no natural routing skew.

### Part 3: Gradient Flow Analysis

Sigmoid chain gradient attenuates as prod[p_i * (1-p_i)] per gate.

| Max Depth | Gradient (sharp gates p~0.9) | Viable? |
|-----------|------------------------------|---------|
| 4 (L=8)  | 6.6e-5 | YES |
| 7 (L=32) | 4.8e-8 | borderline |
| 9 (L=128) | 3.9e-10 | NO (rare experts) |
| 12 (L=512) | 2.8e-13 | NO (rare experts) |

Gradient vanishing is a real concern at L >= 128 but affects only the
DEEPEST paths (rarest experts). These experts handle few tokens, so the
training impact is naturally weighted by their low utilization.

### Part 4: Sensitivity Analysis -- Critical Boundary

For L >= 64, Huffman provides >5% reduction when:
- alpha >= 0.7 with w <= 0.1 (heavy natural skew, minimal balance correction)
- alpha >= 1.0 with w <= 0.5 (Zipf-law skew, moderate balance loss)

The critical alpha decreases slightly with L (larger trees need less skew),
stabilizing at alpha_critical approximately 0.6 for L >= 64 at w = 0.

---

## Parameter Comparison

No new model parameters. This experiment uses the existing huffman_tree
and hierarchical_tree models. The analysis is purely about measuring
distribution properties and computing Huffman constructions.

---

## Micro-Scale Limitations

1. **Distribution models are synthetic, not measured.** The Zipf+uniform
   mixture is a reasonable first-order model but not ground truth. The actual
   expert utilization distribution of a trained DeepSeek-V3 or Qwen3 model
   is an empirical question that requires profiling a real macro MoE.

2. **Top-k routing creates complex dependencies.** With top-k, expert
   utilization is not independently distributed. The mixture model treats
   each expert independently, ignoring competition effects.

3. **Balance loss strength is unknown.** The parameter w (balance loss
   weight) in our mixture model is a proxy. Different systems use different
   mechanisms (auxiliary loss, per-expert bias, capacity factor) with
   different effective strengths.

4. **Gradient analysis is worst-case.** The sigmoid chain gradient analysis
   assumes sharp gates (p~0.9). In practice, gates early in training are
   closer to 0.5 (uncertain), where gradients are stronger. The gradient
   concern is primarily relevant for inference-optimized Huffman trees
   that won't be further trained.

5. **The experiment cannot measure THE key quantity.** The actual H_norm
   of a production MoE model is the single measurement that would resolve
   the hypothesis definitively. Everything else is modeling.

---

## What Would Kill This

### At Macro Scale (the definitive test)

- **Measure H_norm of a real trained MoE.** If H_norm > 0.95 across all
  layers for DeepSeek-V3 or Qwen3-Coder-Next, Huffman is dead.
  The experiment to run: load a production MoE checkpoint, profile expert
  utilization over a diverse evaluation set, compute H_norm per layer.

- **Gradient vanishing at depth 10+.** If rare experts at deep Huffman
  paths fail to learn at L=256+, the tree needs depth capping, which
  reduces the Huffman advantage. Testable by training a Huffman-shaped
  MoE and comparing rare expert quality against a balanced tree.

### What Would Validate This

- **H_norm < 0.90 on a real MoE.** This would put us firmly in the
  "heavy skew" regime where Huffman provides 10-20% depth reduction.
  DeepSeek-V3's auxiliary-loss-free design is the most promising
  candidate -- their explicit motivation suggests natural routing IS
  significantly non-uniform.

---

## Summary

| Kill Criterion | Micro | Analytical (production models) |
|---------------|-------|-------------------------------|
| H > 0.95*log2(L) | KILLED (H/Hmax=0.997) | 7/10 killed, 3/10 survive |
| Reduction < 5% | KILLED (0.00%) | 7/10 killed, 3/10 survive |

**Verdict: CONDITIONAL PASS.**

The hypothesis survives if and only if production MoE routing distributions
have sufficient natural skew (alpha >= 0.7 with weak balance loss). Three
modeled scenarios survive, all representing heavy skew with minimal balance
correction.

Key findings:

1. **The boundary is sharp.** The transition from "Huffman useless" to
   "Huffman useful" happens in a narrow parameter band. Small changes in
   balance loss strength (w: 0.3 -> 0.1) can swing the reduction from
   2.7% to 18.3%.

2. **DeepSeek-V3 is the strongest candidate.** Their auxiliary-loss-free
   design specifically preserves natural routing skew. If any production
   system benefits from Huffman, it's DeepSeek-V3.

3. **Balance loss is the enemy of Huffman.** Strong balance losses (w >= 0.5)
   push routing toward uniform, eliminating Huffman's advantage. Systems
   designed for balanced expert utilization (Switch Transformer) are
   fundamentally incompatible with Huffman routing.

4. **The definitive test is profiling a real model.** Everything in this
   experiment is modeling. The single measurement that resolves the
   hypothesis is H_norm of a production MoE over diverse inputs.

5. **Gradient depth is a secondary constraint.** At L >= 128, rare experts
   in Huffman trees may receive vanishing gradients. This constrains the
   maximum useful tree size but does not kill the approach for L <= 64.
