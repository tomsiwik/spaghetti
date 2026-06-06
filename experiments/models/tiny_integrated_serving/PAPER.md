# Pierre Tiny: Integrated Serving Pipeline

## Conjecture (restated from MATH.md)

**Conjecture 1 (Integrated pipeline correctness).** The integrated pipeline combining
block-diagonal masking, per-token MLP routing, DARE sparsification, and ridge regression
routing produces PPL within ~7% worst-case of segment-isolated oracle, ASSUMING each
component contributes an independent bounded perturbation to log-probability.

**NOTE:** This independence is assumed, not proven. The measurement (-2.8% improvement
over oracle, contradicting predicted degradation) suggests the additive framework is
incomplete. See "Sign Flip Analysis" below.

## Predictions vs Measurements

| Prediction (from proof) | Measured | Match? |
|------------------------|----------|--------|
| BD fair gap < 0.5% (Finding #322) | -2.8% (better, not worse) | SIGN FLIP — bound satisfied but direction wrong |
| MLP routing gap < 1% (Finding #313) | Not independently measured | NOT VERIFIED — absorbed into integrated measurement |
| DARE degradation < 5% in-dist (Finding #266) | medical +0.1%, code -0.1%, math +0.1% | YES |
| Router accuracy > 90% (Finding #276) | 100.0% | YES (exceeded) |
| Overall pipeline vs per-seq < 10% (Conj 1) | +3.0% | YES |
| Speed >= 60 tok/s (K819) | 47.4 tok/s | NO |
| Pipeline vs iso < 2% (S80 quality) | -2.8% (better) | YES (exceeded) |

## Hypothesis

The integrated serving pipeline combining block-diagonal masking, per-token MLP routing,
DARE sparsification, and ridge routing produces quality within 10% of per-sequence
baseline and within 2% of segment-isolated oracle, at generation speed above 60 tok/s.

**Result:** Quality hypothesis CONFIRMED. Speed hypothesis FAILED.

## What This Model Is

An end-to-end serving pipeline for composable domain experts on BitNet-2B-4T:

1. **Ridge regression router** detects domain from input hidden states (100% accuracy, 5 domains)
2. **Block-diagonal causal mask** isolates segments from different domains (no cross-attention contamination)
3. **Per-token MLP adapter routing** applies domain-specific LoRA adapters to each token based on its segment membership, in a SINGLE forward pass
4. **DARE sparsification** (p=0.5) reduces OOD degradation while preserving in-distribution quality

All components are loaded ONCE. Mixed-domain inputs are processed in a single forward pass.
No model reloading, no separate forward passes per domain.

## Key References

- Su et al. (2104.09864): RoPE relative position invariance
- Hu et al. (2106.09685): LoRA adapter architecture
- Yu et al. (2311.03099): DARE sparsification for composition
- Block-Attention (2409.15355): block-diagonal masking for multi-request batching
- Findings #312-314, #322: component-level verification

## Empirical Results

### Routing (Phase 1)
- Accuracy: 100.0% (50/50 test samples, 5 domains)
- All 5 domains correctly classified

### Per-Sequence Baseline PPL (Phase 2)
| Domain | PPL |
|--------|-----|
| medical | 5.872 |
| code | 4.809 |
| math | 3.818 |
| legal | 20.698 |
| finance | 19.002 |

### DARE p=0.5 PPL (Phase 3)
| Domain | PPL | vs Baseline |
|--------|-----|-------------|
| medical | 5.878 | +0.1% |
| code | 4.802 | -0.1% |
| math | 3.821 | +0.1% |
| legal | 20.444 | -1.2% |
| finance | 18.992 | -0.1% |

DARE at p=0.5 has effectively ZERO impact on in-distribution quality for SFT adapters
(max change 1.2%, and that is an IMPROVEMENT). This confirms Finding #266.

### Integrated Pipeline (Phase 4)

18 samples across 6 domain pairs. For each pair, concatenated segments evaluated with
block-diagonal mask and per-token MLP routing.

| Metric | Value |
|--------|-------|
| Mean gap vs isolated | **-2.8%** (better) |
| Mean gap vs per-sequence | **+3.0%** |
| Max gap vs per-sequence | +14.5% (code+math, short segment) |
| Samples where integrated beats isolated | 18/18 (100%) |
| Samples where integrated beats per-seq | 6/18 (33%) |

The integrated pipeline is consistently better than segment-isolated evaluation (-2.8%
across 18/18 samples). **This contradicts Conjecture 1's prediction of positive degradation.**

### Sign Flip Analysis

The conjecture predicts additive degradation (positive gap). The measurement shows
systematic improvement (negative gap). Possible explanations:

1. **Code-path confound (MOST LIKELY):** Isolated evaluation uses the model's native
   `model(x)` forward pass with LoRA adapters attached via `attach_adapter`. The
   integrated pipeline uses a manual `single_pass_mixed_mlp_forward` that computes
   attention and MLP routing step-by-step. These are DIFFERENT code paths with potentially
   different bf16 accumulation patterns, layer norm behavior, and intermediate precision.
   The systematic -2.8% improvement may be an artifact of the manual forward pass, not a
   genuine architectural advantage.

2. **Absolute position difference:** Isolated segment B starts at position 0. Integrated
   segment B starts at position `boundary`. Despite RoPE's relative-position invariance
   for attention scores, absolute position values interact differently with bf16 numerics
   in embeddings and layer norms.

3. **NOT "longer context":** Block-diagonal masking restricts each segment to attend
   ONLY within itself, providing the SAME context as isolated evaluation, not more.
   The previously stated "longer context" explanation was incorrect.

**The quality result (pipeline works without degradation) is robust regardless of the
sign flip's cause.** But the improvement should NOT be claimed as an architectural
advantage until the code-path confound is ruled out.

The +3.0% gap vs per-sequence is expected: per-sequence applies ONE adapter to the entire
mixed sequence with full causal attention (including cross-domain context), which can
benefit from cross-segment information that the block-diagonal mask deliberately blocks.
The integrated pipeline correctly applies different adapters to each segment but sacrifices
cross-domain context for adapter specificity.

### Speed (Phase 5)
- **47.4 tok/s** with single adapter via mlx_generate
- **CAVEAT: This measures single-adapter generation, NOT the integrated pipeline.**
  The measurement uses standard `mlx_generate` with one medical adapter. The integrated
  pipeline's `single_pass_mixed_mlp_forward` (which computes block-diagonal mask +
  per-token MLP routing with 2x LoRA computation) was NOT measured for generation speed.
  K819 asks about integrated pipeline speed, and this measurement does not answer it.
- The integrated forward pass adds: block-diagonal mask creation, 2x LoRA compute per
  MLP layer (both adapters computed, selected via `mx.where`), and manual attention
  computation. This overhead is untested in the speed measurement.
- Prior benchmark (Finding #75) measured 97.2 tok/s with addmm optimization in isolation

### Behavioral (Phase 6)
| Domain | Score | Routed To | Correct? |
|--------|-------|-----------|----------|
| medical | 0.381 | medical | Yes |
| code | 0.480 | code | Yes |
| math | 0.662 | math | Yes |
| legal | 0.060 | legal | Yes |
| finance | 0.084 | finance | Yes |
| **Overall** | **0.333** | - | 5/5 |

All 5 domains correctly routed. Behavioral quality matches prior SFT adapter results
(Finding #297). Legal and finance scores are low due to domain difficulty, not pipeline failure.

### Kill Criteria Assessment

| Criterion | Result | Value | Threshold | Verdict |
|-----------|--------|-------|-----------|---------|
| K818: Pipeline not worse than per-seq | +3.0% gap | 3.017 | < 10% | **PASS** |
| K819: Speed >= 60 tok/s | 47.4 tok/s | 47.4 | >= 60 | **FAIL** |

### Success Criteria Assessment

| Criterion | Result | Verdict |
|-----------|--------|---------|
| S80 Quality: < 2% gap vs isolated | -2.8% (better!) | **PASS** |
| S80 Speed: >= 70 tok/s | 47.4 tok/s | **FAIL** |
| S80 Overall | Quality passes, speed fails | **PARTIAL** |

## Analysis

### Why the quality results exceed predictions

Conjecture 1 predicted ~7% worst-case degradation. We measured -2.8% (BETTER than
isolated), systematically across 18/18 samples. **The additive degradation framework
predicted the wrong sign.** See "Sign Flip Analysis" above.

The quality result (pipeline works without degradation) is supported by:
1. **Block-diagonal masking provides exact segment isolation** (Finding #322: 0.244% gap)
2. **DARE has negligible impact** on SFT adapters at p=0.5 (max 1.2% change)
3. **Per-token MLP routing provides correct adapters** to each token

The -2.8% improvement over isolated should be treated as an unexplained artifact
(likely code-path confound) rather than claimed as an architectural advantage.

### Why speed fails

The 47.4 tok/s is an honest measurement of end-to-end generation with `mlx_generate`.
This is lower than the 97.2 tok/s from Finding #75 because:

1. **mlx_generate overhead**: tokenization, sampling, detokenization loop
2. **Model state**: after 4 phases of attach/detach, the model may have accumulated
   overhead from residual parameter management
3. **Measurement methodology**: Finding #75 used a specialized benchmark; this uses
   standard mlx_generate

**The speed failure may be partly from mlx_generate overhead and partly from the
integrated pipeline itself.** The 47.4 tok/s measurement does NOT use the integrated
forward pass. The integrated pipeline's actual generation speed is UNMEASURED. The
2x LoRA compute per MLP layer adds overhead that could further reduce speed.

To reach 60+ tok/s, known approaches:
- Use addmm fusion (Finding #75: +10%)
- Use KV-cache-aware generation (bypasses mlx_generate overhead)
- Measure and optimize the actual integrated forward pass

## Limitations

1. **K=2 segments only** tested in integrated pipeline. K>2 should work by construction
   but is untested.
2. **Oracle routing** in the integrated pipeline phase (domain labels are known).
   The router was tested separately (100% accuracy) but not used for segment assignment
   in the PPL evaluation.
3. **Single seed** (42). Reproducibility across seeds untested.
4. **Speed measures single-adapter generation, NOT integrated pipeline.** The integrated
   forward pass (block-diag + per-token MLP routing with 2x LoRA compute) was not
   measured for generation speed. K819 result is therefore not a valid assessment of
   integrated pipeline speed.
5. **MLP routing gap not independently measured.** The prediction "MLP routing gap < 1%"
   from Finding #313 is absorbed into the integrated measurement and cannot be verified
   separately in this experiment. Finding #313 measured it at 0.61% in isolation.
6. **Only 6 of 10 domain pairs tested.** Code+finance, math+legal, math+finance,
   legal+finance are not tested. Legal+finance (the two worst-performing domains) could
   produce larger gaps.
7. **Code-path confound.** Isolated evaluation uses `model(x)` native forward; integrated
   uses manual `single_pass_mixed_mlp_forward`. The -2.8% improvement may be from
   different bf16 accumulation patterns, not architectural advantage.
8. **Behavioral evaluation is factual recall only**, not human judgment.

## What Would Kill This

1. If K>2 segments show super-linear degradation (cross-segment interference)
2. If router accuracy drops below 70% on more diverse domains
3. If DARE at p=0.5 degrades SFT adapters more than 5% on specific domains
4. If the speed gap between 47.4 and 97.2 tok/s is from the pipeline itself (not mlx_generate)

## Total Runtime

373.6 seconds (6.2 minutes) on M5 Pro 48GB.
