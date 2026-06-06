# Peer Review: bitnet_galore_scaffold (Re-Review After 5 Fixes)

## NotebookLM Findings

Skipped -- the experiment scope and prior review history are well-understood. The re-review focuses on verifying the 5 applied fixes and checking for new issues.

## Fix Verification

### Fix 1: K1 Kill Criteria Corrected -- ADEQUATE

HYPOTHESES.yml now reads "GaLore scaffold ternary PPL > 2x standard Adam scaffold ternary PPL after equivalent compute." PAPER.md Kill Criteria section (lines 96-101) explicitly acknowledges the original pretrained-base comparison is untestable at micro scale and states what was tested. The HYPOTHESES.yml notes field still mentions "compare composition quality vs pretrained BitNet-2B" (line 3425) which is aspirational rather than tested, but this is in the notes (context), not kill_criteria (binding), so it is acceptable.

### Fix 2: Scaffold-Agnostic Claim Downgraded -- ADEQUATE

Key Insight #3 (PAPER.md lines 123-127) now says "consistent with but not proving scaffold-agnosticism" and lists caveats (N=2 scaffolds, same init, would need more diverse scaffolds). This is appropriately hedged.

### Fix 3: Adapter/Model Parameter Ratio -- ADEQUATE

Limitation #7 (PAPER.md lines 153-161) documents the 107% ratio, explains why it weakens scaffold claims, and notes the >1000x ratio difference at production scale. This is thorough.

### Fix 4: Multi-Seed K1 -- ADEQUATE BUT WITH RESIDUAL CONCERN

Three seeds run: [42, 123, 456] yielding K1 = [1.910, 1.493, 2.349], mean=1.918, std=0.349. Results verified against results.json. The paper correctly reports one seed exceeds 2.0, labels the pass as "marginal," and notes "high variance (CV=18.2%)" in multiple places (PAPER.md line 77, Limitation #5).

**Residual concern (non-blocking):** The decision rule "PASS on 3-seed mean" is ad hoc. The paper does not specify whether K1 should be evaluated on mean, median, or all-seeds-must-pass. With 1/3 seeds individually KILL, a more conservative protocol would label this INCONCLUSIVE. However, the paper's overall verdict of "SUPPORTED with significant caveats" is appropriately conservative, and the individual seed failure is prominently documented. This is honest reporting.

### Fix 5: Moment-Persistence Deviation -- ADEQUATE

MATH.md Assumption #6 (lines 146-155) documents the deviation, explains the mechanism (moments in old projected space not compatible with new), notes it biases against GaLore, and argues this makes the pre-quant result more impressive. The reasoning is correct.

## Mathematical Soundness

### GaLore Core: Sound

The GaLore algorithm description (MATH.md lines 19-58) is correct. Memory analysis checks out. The worked example for (256, 256) matrices is verified: 2*64*256 + 256*64 = 49,152 vs 2*256*256 = 131,072, savings 62.5%.

### K1 Metric: Correctly Scoped Now

The K1 metric now measures what it claims to measure: GaLore-ternary / standard-ternary PPL ratio. The 2.0x threshold is reasonable as a "within noise" bound for two training methods on the same architecture.

### Quantization Degradation Analysis: Sound

The hypothesis that GaLore produces higher effective rank weights (more distributed singular values) leading to worse ternary quantization is mechanistically plausible and well-supported by the data. Standard quant degradation is remarkably stable across seeds (1.091, 1.099, 1.126 -- CV=1.7%) while GaLore quant degradation is highly variable (2.499, 2.135, 3.173 -- CV=21%). This variance asymmetry itself is informative: it suggests GaLore weight distributions are more seed-sensitive with respect to quantization friendliness.

### Minor Numerical Note

PAPER.md line 55 reports the quant degradation gap as "2.0-2.9x" but the actual per-seed gaps (GaLore-degradation / standard-degradation) are 2.29x, 1.94x, 2.82x -- so the range is more accurately 1.9-2.8x. This is cosmetic.

## Novelty Assessment

The primary finding -- that GaLore weights degrade disproportionately under ternary quantization -- is the experiment's genuine contribution. While not surprising in hindsight (low-rank gradient projection -> higher effective weight rank -> worse quantization), it has not been documented in the GaLore literature. The GaLore papers (Zhao et al. 2024, GaLore 2) focus on FP16/BF16 performance and do not test post-training quantization to ternary.

The experiment is correctly differentiated from the killed exp_bitnet_basefree_exploration (adapter transfer vs scaffold growth).

## Experimental Design

### Strengths

1. Clean A/B comparison: same init, same data, same architecture, same adapter training, differing only in pretraining optimizer.
2. Multi-seed validation addresses the prior review's most actionable concern.
3. Primary finding (quantization gap) is robust across all seeds -- the directional result is clear even if K1 is marginal.
4. Limitations section is comprehensive and honest (9 items).

### Remaining Weakness: No Random Scaffold Control

The prior review noted this, and it remains unaddressed. A random (untrained) scaffold control would definitively test whether the 107% adapter-to-model ratio makes composition scaffold-independent. However, the paper already heavily caveats the composition claim (Fix 2, Fix 3), and exp_bitnet_basefree_exploration's PPL 319M on a random scaffold (different test: adapter transfer) suggests scaffolds do matter for some workflows. This is not blocking given the caveats.

### Adapter Convergence Issues

Across all seeds, legal adapters consistently fail to converge on both scaffolds (first_50_loss ~ last_50_loss). Creative adapters fail on standard (seed 123) and GaLore (seeds 42, 123). This means the "adapter quality" comparison is confounded by non-convergence for 2/5 domains. The paper notes this partially (Limitation #6: "Python/math adapters hurt PPL") but does not discuss the convergence failures explicitly. Non-blocking but worth noting.

## Hypothesis Graph Consistency

HYPOTHESES.yml kill_criteria now match what was tested. Status "supported" is appropriate given the marginal K1 and heavy caveats. The evidence entry (line 3406) accurately summarizes the multi-seed results.

The experiment correctly feeds into exp_bitnet_meta_scaffold (depends_on includes exp_bitnet_galore_scaffold).

## Macro-Scale Risks (advisory)

1. **GaLore + STE is the critical untested path.** The paper correctly identifies this. The micro experiment shows post-hoc ternary quantization is destructive for GaLore weights. STE-aware training is the proposed solution but is completely untested. At macro, this should be the first thing to validate.

2. **Adapter/model ratio inversion.** At micro: 107%. At production: 0.08%. The composition similarity finding may not survive this >1000x ratio change. Macro experiments must verify scaffold effects on composition when adapters are tiny relative to the model.

3. **K1 variance at scale.** The 18.2% CV on K1 at micro could narrow or widen at d=2560. The GaLore quant degradation variance (CV=21%) is the specific risk.

## Verdict

**PROCEED**

All 5 fixes from the prior review have been adequately addressed:

1. K1 kill criteria correctly scoped to standard-Adam comparison.
2. Scaffold-agnostic claim appropriately downgraded.
3. Adapter/model ratio limitation thoroughly documented.
4. Multi-seed validation provides real uncertainty quantification.
5. Moment-persistence deviation documented in MATH.md.

The experiment is honest about its limitations. The primary finding (GaLore ternary degradation gap) is robust and valuable -- it precisely identifies the bottleneck for the base-free path. The K1 marginal pass (mean 1.918, one seed KILL) is correctly labeled "SUPPORTED with significant caveats" rather than over-claimed. The paper does not hide the seed-456 failure.

The remaining weaknesses (no random scaffold control, adapter convergence issues, 3-seed sample size) are acknowledged in the paper and are reasonable trade-offs within the micro-experiment scope.

### Non-Blocking Observations for Future Work

1. The quant degradation variance asymmetry (GaLore CV=21% vs standard CV=1.7%) deserves investigation at macro. If GaLore weight distributions are fundamentally unstable with respect to quantization, even STE-aware training may not fully resolve the gap.

2. Running 5-10 seeds on K1 (total ~30 min) would give a proper confidence interval. Current 3-seed CI is too wide to be definitive. Consider this before building on K1's pass in downstream experiments.

3. Legal adapter non-convergence across all seeds/scaffolds suggests a data or learning-rate issue, not a scaffold issue. Worth investigating separately.
