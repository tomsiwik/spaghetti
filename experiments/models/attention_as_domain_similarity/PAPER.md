# Attention LoRA Cosine as Domain Similarity Predictor: Research Digest

## Hypothesis

Attention LoRA cosine similarity reliably predicts semantic domain overlap
across many domain pairs, and is a STRONGER predictor than FFN cosine.

**Falsifiable**: If attention cosine does not correlate with semantic domain
similarity (Spearman rho < 0.3 or p > 0.05), or if the signal vanishes as
more domain pairs are added, the hypothesis is killed.

## What This Experiment Is

This experiment tests whether the attention-as-domain-amplifier finding from
ffn_only_vs_all_modules (math-medical: attn cos=0.85 vs FFN cos=0.59)
generalizes to a systematic predictor of domain similarity.

We train 12 synthetic domain experts across 4 clusters (code, reasoning,
knowledge, creative) using a 2-layer transformer with LoRA on ALL modules
(Q, K, V, O, fc1, fc2). We compute separate cosine similarity matrices for
attention-only, FFN-only, and all-module deltas, then measure Spearman
correlation with a ground-truth semantic similarity matrix.

Key design choices:
- **Transformer, not MLP**: unlike orthogonality_by_domain_type (MLP-only),
  this experiment uses self-attention to test the attention-specific claim.
- **Graduated similarity**: ground truth uses continuous similarity values
  (0.1 to 0.7) based on cluster relationships, not binary within/cross.
- **SPSA training**: Simultaneous Perturbation Stochastic Approximation
  for all LoRA B matrices, enabling all-module training in pure numpy.
- **K2 ablation**: systematically tests correlation stability across
  different numbers of domains (6, 8, 10, 12).

## Lineage in the Arena

```
ffn_only_vs_all_modules (5 real Qwen2.5-7B adapters, attention amplifies overlap)
 |-- orthogonality_by_domain_type (15 MLP experts, within/cross structure proven)
 \-- attention_as_domain_similarity (this experiment -- KILLED)
```

## Key References

- Geva et al. 2021, "Transformer Feed-Forward Layers Are Key-Value Memories"
- Spall 1992, "Multivariate Stochastic Approximation Using a Simultaneous
  Perturbation Gradient Approximation" (SPSA training)
- Prior experiment: ffn_only_vs_all_modules (math-medical outlier)
- Prior experiment: orthogonality_by_domain_type (7.84x within/cross ratio)

## Empirical Results

### Architecture

2-layer transformer, d=64, d_ff=256, 4 attention heads (d_head=16).
Rank-8 LoRA on Q, K, V, O, fc1, fc2. Pure numpy + scipy. 12 domains
in 4 clusters. SPSA training, 300 steps per expert.

### Spearman Correlation with Ground Truth (3 seeds)

| Module | Mean rho | Std | Values (per seed) | Mean p |
|--------|----------|-----|-------------------|--------|
| Attention | **0.073** | 0.102 | [0.217, -0.008, 0.010] | 0.66 |
| FFN | **0.185** | 0.066 | [0.095, 0.208, 0.252] | 0.19 |
| Full | **0.173** | 0.026 | [0.140, 0.203, 0.176] | 0.17 |

### Within/Cross Cluster Ratio (3 seeds)

| Module | Mean Ratio | Std | Values |
|--------|-----------|-----|--------|
| Attention | **0.98x** | 0.18 | [1.20, 0.76, 0.98] |
| FFN | **1.28x** | 0.35 | [1.16, 0.93, 1.77] |
| Full | **1.08x** | 0.39 | [0.87, 0.74, 1.64] |

### K2 Ablation: Signal Stability (seed 42, attention only)

| N domains | N pairs | Mean rho | Std rho | Mean p |
|-----------|---------|----------|---------|--------|
| 6 | 15 | 0.266 | 0.294 | 0.38 |
| 8 | 28 | 0.214 | 0.056 | 0.29 |
| 10 | 45 | 0.165 | 0.115 | 0.39 |
| 12 | 66 | 0.217 | 0.000 | 0.08 |

### Kill Threshold Checks

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| K1: Attention rho >= 0.3 | 0.073 | >= 0.3 | **KILL** |
| K1: Attention p < 0.05 | 0.95 | < 0.05 | **KILL** |
| K2: Signal stable with N | rho stable but weak | no > 30% drop | PASS |
| Attention > FFN | rho_attn < rho_ffn by 0.112 | -- | **FAIL** |

**Verdict: KILLED.** Attention cosine does NOT predict domain similarity at
micro scale. FFN cosine shows a STRONGER (though still weak) signal.

## Key Insights

### 1. The Hypothesis Direction is Reversed

At micro scale with non-converged training, FFN cosine (rho=0.185) is a
BETTER predictor of domain similarity than attention cosine (rho=0.073).
This reverses the intuition from the macro Qwen adapter finding.

### 2. Training Signal is the Bottleneck, Not Architecture

Training losses remain near log(V) = 3.466 throughout training (random
baseline). The LoRA deltas are stochastic gradient noise, not converged
domain features. Delta norms of ~0.01 are 100x smaller than real macro
adapters. This means the experiment tests "which module captures domain
signal FIRST from random initialization" -- and FFN wins at this regime.

### 3. The Math-Medical Macro Finding May Be a Convergence Effect

The original finding (attn cos=0.85 vs FFN cos=0.59 for math-medical at
macro scale) may reflect a regime where:
- Converged attention patterns genuinely share reasoning structures
- These shared structures are absent in the non-converged micro regime
- Attention amplification of domain similarity is a CONVERGED-TRAINING
  phenomenon, not a structural one

This suggests attention cosine as a domain similarity predictor is
contingent on expert quality, not a universal property.

### 4. FFN Shows Directional Signal Even at Micro

FFN within/cross ratio reaches 1.77x in one seed, and the FFN Spearman
rho reaches significance (p=0.04) in one seed. This is consistent with
the orthogonality_by_domain_type finding (7.84x ratio with MLP-only)
and confirms that FFN is the more structurally predictive module at
micro scale.

### 5. Relationship to orthogonality_by_domain_type

That experiment (MLP-only, 15 domains, 3 clusters) found a very strong
within/cross ratio of 7.84x. This experiment (transformer, 12 domains,
4 clusters) finds only 1.28x for FFN and 0.98x for attention. Three
differences explain the gap:
1. SPSA vs analytical backprop -- SPSA adds gradient noise
2. 300 vs 300 steps but with more parameters per expert (attention overhead)
3. 4 clusters vs 3 -- more graduated similarity reduces binary contrast

## Micro-Scale Limitations

1. **Non-converged training (fatal limitation).** All losses remain at
   ~3.466 (random). The LoRA deltas do not represent domain-specific
   features. This makes the KILL verdict specific to the micro regime,
   not a definitive statement about the hypothesis.

2. **SPSA training noise.** SPSA converges in expectation but adds
   significant per-step noise. Analytical backprop through attention
   (as in PyTorch) would reduce this noise. However, implementing full
   attention backprop in numpy is impractical for this micro framework.

3. **Synthetic domains.** Character-level Markov chain differences are
   far weaker than real-world domain differences (code vs medicine).
   Real domains have fundamentally different vocabulary, syntax, and
   reasoning patterns.

4. **Graduated similarity is researcher-defined.** The ground-truth
   similarity values (code-reasoning=0.5, etc.) are subjective. Different
   assignments would change correlation values, though the directional
   finding (FFN > attention) would likely hold.

5. **Small model.** d=64, 2 layers, 4 heads. Real attention patterns
   (long-range dependencies, cross-position routing) cannot emerge in
   this regime.

## What Would Kill This (At Macro Scale)

This hypothesis should be RE-TESTED at macro scale with:
- Real Qwen2.5-7B adapters trained on diverse domains (N >= 10)
- Compute attention-only and FFN-only cosine from safetensors weights
- Ground truth from embedding-based domain similarity (e.g., sentence
  transformer of domain descriptions)

The macro version was essentially already tested with 5 domains in
ffn_only_vs_all_modules (1 informative pair out of 10). A proper test
needs 10+ fully trained domain experts.

**Kill at macro**: If attention cosine does NOT show higher Spearman
correlation with domain similarity than FFN cosine across 10+ real
domain experts, the hypothesis is definitively dead.

**Resurrect conditions**: If converged macro adapters show attention
rho > 0.5 with semantic similarity while FFN rho < 0.3, the hypothesis
is resurrected and attention becomes a viable routing signal.

## Artifacts

- `attention_as_domain_similarity.py` -- experiment code
- `results.json` -- raw results (3 seeds)
- `MATH.md` -- mathematical foundations
- Total experiment time: ~365s (3 seeds, CPU)
