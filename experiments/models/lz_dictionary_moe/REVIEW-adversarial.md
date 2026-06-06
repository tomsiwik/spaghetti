# Peer Review: LZ Dictionary MoE

## NotebookLM Findings

Skipped -- manual deep review was sufficient given the focused scope of this experiment.

## Mathematical Soundness

### What holds

1. **Expert decomposition is correctly formulated.** The decomposition
   `expert_i(x) = sum_j alpha_{i,j} * dict_j(x) + delta_i(x)` is a valid
   factorization of expert MLPs into shared + unique components. The softmax
   over alpha logits guarantees a proper convex combination of dictionary
   outputs.

2. **Parameter count derivation is correct.** The worked examples in MATH.md
   match the code. Dict MoE small at d=64, D=8, r=32, r_delta=16 yields
   40,992 MLP params per layer vs 131,072 for standard MoE. The 68.7% savings
   figure is arithmetically correct.

3. **Effective rank analysis is sound.** The upper bound of D*r + r_delta on
   per-expert effective rank is correct, though as MATH.md acknowledges, the
   actual rank is typically lower due to shared structure.

4. **Utilization metric is well-defined.** The threshold at 1/(2D) = 0.0625
   for D=8 is a reasonable "active vs inactive" boundary. Shannon entropy
   normalized by log(D) is the standard measure.

### What does not hold or is incomplete

5. **MATH.md Eq for W1_eff_i (line 96) is informal and imprecise.** The
   notation `[W^up_j; 0] @ [W^down_j; 0]` does not properly represent the
   effective weight matrix. Since each dictionary entry applies ReLU between
   down and up projections, the composition `sum_j alpha_j * up_j * relu(down_j * x)`
   is NOT equivalent to `(sum_j alpha_j * up_j * down_j) * relu(x)`. The
   nonlinearity prevents factoring out a single effective weight matrix per
   expert. The rank analysis that follows (line 98-106) is therefore an
   approximation that holds only in the linear regime (pre-ReLU activations
   all positive). This is not wrong per se, but the paper presents it as
   exact.

6. **Assumption 2 ("soft composition is sufficient") is not tested against
   any alternative.** The experiment only tests soft (full softmax) alpha.
   Without a hard/sparse alpha baseline, we cannot know if soft composition
   is sufficient vs simply the only thing tested.

## Novelty Assessment

### Prior art

The paper cites the correct prior work: StructMoE, L-MoE, AoE, DeepSeek-MoE.
The claimed delta -- composing from a SHARED codebook within each expert rather
than sharing entire experts or using per-expert secondary selection -- is a
genuine differentiator.

The closest published work is:

- **Union-of-Experts (UoE, 2024)**: Performs SVD analysis showing experts
  share low-rank structure, then proposes shared + unique decomposition. The
  LZ Dictionary MoE is essentially the same insight applied at training time
  rather than post-hoc. The paper should cite UoE more prominently as the
  motivating analysis.

- **Soft Merging of Experts with Adaptive Routing (SoMoE-A)**: Combines
  expert outputs with learned soft weights. The alpha coefficients in this
  experiment serve an analogous role but at the sub-expert level.

- **VQ-based weight quantization/compression**: The "codebook" framing is
  borrowed from vector quantization. The difference is that LZ Dictionary MoE
  operates on functional sub-networks rather than individual weight vectors.

### Novelty verdict

The idea is a valid and useful synthesis of existing techniques (low-rank
factorization + shared codebook + per-expert residual). It is not a radical
departure from prior art but represents a clean, testable combination. The
LZ77 analogy is a framing device, not a technical contribution -- no actual
LZ compression algorithm is used.

## Experimental Design

### Critical flaw: kill criterion not actually tested

The HYPOTHESES.yml kill criterion states:

> "dictionary experts >3% worse than independent experts **at same total params**"

The experiment does NOT test this. The configurations are:

| Model | Params | vs Standard MoE |
|-------|--------|-----------------|
| Standard MoE | 596,352 | baseline |
| Dict MoE small | 236,032 (40%) | -0.9% |
| Dict MoE large | 432,640 (73%) | -0.6% |

Neither Dict MoE configuration matches Standard MoE's parameter count. What
the experiment actually tests is: "can dictionary MoE achieve similar quality
with FEWER params?" This is a valid and interesting question, but it is not
the stated kill criterion.

To properly test the kill criterion, the experiment needs a Dict MoE
configuration with ~596K total params (e.g., D=16, r=64, r_delta=64 or
similar). If dictionary composition provides no benefit at matched params,
the mechanism may just be acting as regularization (fewer params = less
overfitting at micro scale).

### The regularization confound

This is the central weakness. At micro scale:

- Standard MoE (596K) is WORSE than Dense GPT (202K) by +0.6%
- Dict MoE small (236K) is close to Dense GPT (202K) size

Standard MoE is overfitting. Dict MoE small "wins" partly because it has
fewer parameters, not necessarily because dictionary composition captures
shared structure. The paper acknowledges this in Limitation #2 but still
claims the result as a PASS on the kill criteria.

**The honest interpretation:** Dict MoE at 236K params matches a 202K dense
model (within noise: -0.3%). This is consistent with "low-rank factorization
is a good regularizer" and does not require the LZ dictionary mechanism to
explain.

### Near-uniform alpha: feature or bug?

All alpha weights have normalized entropy 0.999 (essentially uniform 1/8 for
each entry). This means every expert computes:

    expert_i(x) ~ (1/8) * sum_j dict_j(x) + delta_i(x)

The dictionary entries are not being composed differently by different experts.
Every expert uses the same uniform average of dictionary outputs and relies
entirely on its delta residual for differentiation. This is functionally
equivalent to:

    expert_i(x) = shared_mean_mlp(x) + delta_i(x)

where `shared_mean_mlp` is a single averaged sub-network. This is a
**degenerate** form of dictionary composition -- it is simply a shared base
MLP (like DeepSeek's shared expert) plus per-expert residuals. The "100%
utilization" result is trivially achieved because all entries are weighted
equally; it tells us nothing about whether the codebook captures meaningful
sub-structures.

### Missing control: shared-base + residual without dictionary

The experiment lacks the most important ablation. Since alpha weights are
uniform, the effective architecture is "one shared low-rank MLP + per-expert
residual." A proper control would be:

    expert_i(x) = shared_mlp(x) + delta_i(x)

where `shared_mlp` is a single rank-(D*r) MLP with the same total params as
the D dictionary entries. If this control matches Dict MoE performance, the
dictionary mechanism provides no value -- it is just a roundabout way of
implementing a shared base.

### Gradient test does not verify alpha_logits

The gradient flow test (`test_gradient_flow`) checks dictionary entry weights
and delta weights but does NOT check that `alpha_logits` receives gradients.
In MLX, `mx.array` attributes on `nn.Module` are included in the parameter
tree, so this should work -- but the test should explicitly verify it. Given
that alpha remains near-uniform after 500 steps, it is worth confirming that
the gradient signal reaches alpha_logits and that the learning rate is
sufficient to move them.

### Statistical power

The differences between models are small (0.3-0.9%) relative to the
per-seed standard deviation (0.004-0.008 in absolute loss, which is ~1%).
With only 3 seeds, no statistical test (t-test, bootstrap CI) is reported.
The -0.9% advantage of Dict MoE small over Standard MoE may not be
statistically significant.

## Macro-Scale Risks (advisory)

1. **Alpha may remain near-uniform at scale.** If the softmax alpha logits
   have vanishing gradients (the alpha gradient scales as alpha_j * (1 -
   alpha_j), which is ~1/D * (1 - 1/D) when near-uniform), the dictionary
   may never specialize even with longer training. At D=8, this is 0.109;
   at D=64, it would be 0.015. Larger codebooks may need temperature
   annealing or alternative parameterizations.

2. **Computational cost at inference.** Every expert evaluates ALL D
   dictionary entries (soft composition). With D=8, this is 8x the forward
   passes per expert compared to a standard MLP. The parameter savings are
   real but the FLOP count may be worse. At macro scale, this needs to be
   benchmarked: latency matters, not just param count.

3. **The compression story only works if MoE > Dense.** At micro scale,
   MoE does not beat Dense, so "60% fewer MoE params" is less compelling.
   At macro scale, where MoE genuinely outperforms Dense, the Dictionary
   MoE must demonstrate that it captures the MoE quality advantage (not
   just regularize down to Dense-equivalent performance).

## Verdict

**REVISE**

The mechanism is sound in principle, the code is clean, and the experimental
infrastructure is good. But the experiment does not actually test its own kill
criterion, and the near-uniform alpha result undermines the core claim that
dictionary composition captures shared sub-structure. Specific fixes:

1. **Add a param-matched Dictionary MoE configuration.** Create a Dict MoE
   variant with ~596K params (matching Standard MoE) and compare. This is the
   actual kill criterion. If Dict MoE at matched params is >3% worse, KILL.

2. **Add the shared-base ablation.** Implement `expert_i(x) = shared_mlp(x) +
   delta_i(x)` with a single shared MLP of rank D*r = 256, same total params
   as the dictionary. If this matches Dict MoE performance, the dictionary
   mechanism is providing no value over a simpler shared-base architecture.

3. **Verify alpha_logits gradient flow.** Add an explicit assertion in
   `test_gradient_flow` that `grads["layers"][0]["moe"]["experts"][0]["alpha_logits"]`
   is non-zero. If it IS zero, the near-uniform alpha is a bug, not a finding.

4. **Report statistical significance.** With 3 seeds and differences of
   ~0.5-1%, compute paired t-test p-values or bootstrap confidence intervals.
   If the Dict MoE vs Standard MoE difference is not significant at p<0.05,
   report it as "no significant difference" rather than "-0.9% better."

5. **Revise the PAPER.md claim language.** Change "60% parameter reduction
   with no quality loss" to acknowledge the regularization confound: the
   Dict MoE small model is close in size to Dense GPT, and the advantage
   over Standard MoE may be due to reduced overfitting rather than structural
   sharing. The near-uniform alpha weights support this interpretation.

6. **Fix MATH.md effective rank derivation (line 96-106).** Clarify that the
   rank analysis assumes linearity and does not hold through the per-entry
   ReLU nonlinearity. The effective rank of `sum_j alpha_j * up_j * relu(down_j * x)`
   depends on the input distribution, not just the weight matrix ranks.

Items 1-3 are blocking. Items 4-6 are required for paper quality but not for
the micro-experiment verdict. If item 1 shows >3% degradation at matched
params, or item 2 shows the shared-base ablation matches, the experiment
should be re-evaluated as KILL.
