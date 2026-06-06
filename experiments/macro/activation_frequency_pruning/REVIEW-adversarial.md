# Peer Review: Activation Frequency Pruning

## NotebookLM Findings

Skipped. The experiment is a clean negative result with straightforward math. A deep review is not warranted for a self-killed experiment whose findings are unambiguous.

## Mathematical Soundness

**Definitions: correct.** The firing frequency f_j(eps) = (1/M) sum_m 1[|h_j(x_m)| > eps] is well-defined and correctly implemented in `profile_frequency.py` (line 144: `(gp_abs > eps).astype(mx.float32)` followed by summation and division by total_positions).

**Spearman correlation: correct.** The `compute_spearman_correlation` function (lines 199-234) properly handles tied ranks via average rank assignment and computes Pearson on the ranks. The implementation is standard.

**Kill criterion 1 formalization: correct.** The ratio delta_rand / delta_freq is the right way to measure whether frequency pruning causes less damage than random. The threshold of 2x (frequency must be more than twice as good) is reasonable for a pruning signal to be considered useful.

**Kill criterion 2 formalization: correct.** Spearman |rho| > 0.8 as a redundancy threshold is standard. The correlation is computed across all 116,736 neurons globally (not per-layer), which is the right choice since pruning operates globally.

**One minor note on MATH.md Section 7 (worked example).** The example correctly identifies that frequency alone is insufficient -- a high-frequency, high-magnitude neuron (Neuron 3) would be catastrophic to prune. This is honest about the limitation and is consistent with the empirical findings. Good scientific practice.

**No mathematical errors found.**

## Novelty Assessment

**Prior art awareness is adequate.** The paper correctly cites PowerInfer (2023) as using activation frequency for GPU/CPU offloading decisions (not pruning), and Wanda (2023) as the weight-times-activation approach. The experiment's contribution is testing frequency as a *standalone structured pruning signal* at macro scale, which has not been directly published. However, this is a narrow incremental test of a known concept, not a novel mechanism.

**The negative result is itself the contribution.** The finding that rho(frequency, magnitude) > 0.83 at all epsilon values is a useful empirical fact for the research program. It closes off an entire branch of the hypothesis tree (any activation statistic as a zero-shot structured pruning signal for SwiGLU).

**No reinvention detected.** The experiment properly reuses the parent experiment's data loading, model loading, and perplexity computation (`from profile_gate_products import ...`).

## Experimental Design

**Controls are adequate.**
- Random pruning (3 seeds) provides the baseline.
- Mean magnitude (parent signal) provides comparison.
- Both high-first and low-first frequency directions are tested, preventing the objection that the wrong direction was chosen.
- Seven epsilon values span a reasonable range from well below the median gate product (0.078) to above it.

**Data provenance is clean.** Calibration on WikiText-2 test, evaluation on WikiText-2 validation. These are genuinely disjoint splits. The positions (16,384 cal, 8,192 eval) are modest but sufficient for perplexity estimation on a 0.5B model.

**One design critique: the attention mask is None.** In `profile_frequency.py` line 122, `layer.self_attn(x, mask=None, cache=None)` passes no causal mask. This means the profiling forward pass uses *bidirectional* attention rather than causal attention. This is a methodological error shared with the parent experiment. In practice, this means the gate product statistics are computed under a different attention pattern than the model would use during actual inference. However, this affects all signals equally (frequency, magnitude, random baseline), so it does not invalidate the *relative* comparisons. It does mean the absolute perplexity numbers (baseline 21.31) may differ from a correctly-masked evaluation. This should be noted but is not blocking.

**The kill criteria match HYPOTHESES.yml exactly.** KC1 ("not >2x better than random at tau=0.05") and KC2 ("correlates >0.8 with mean magnitude") are tested and triggered. The HYPOTHESES.yml entry correctly records both as killed.

## Macro-Scale Risks (advisory)

Not applicable -- this experiment is already at macro scale (Qwen2.5-0.5B, 116K neurons). The findings are directly relevant. The paper correctly notes limitations: single model, single dataset, SwiGLU-only, zero-shot only. These are legitimate caveats for generalization but do not undermine the core finding.

The broader implication for the research program is well-stated: zero-shot structured pruning using any activation statistic appears to be a dead end for production models without auxiliary loss. The next steps (Wanda-style combined signals, or pruning + recovery training) are correctly identified.

## Verdict

**PROCEED** (as a completed, self-killed experiment with valid negative results)

The experiment is methodologically sound, the kill criteria are clearly defined and unambiguously triggered, and the negative result is informative for the research program. The finding that activation frequency is redundant with mean magnitude (rho > 0.83 at all tested epsilons) and that both frequency pruning directions are catastrophically worse than random (66x and 53x at 5%) closes off this branch of the hypothesis tree.

One non-blocking note for future experiments in this lineage:

1. **Causal mask omission.** Future profiling experiments should pass a proper causal attention mask during the forward pass. While this does not affect relative signal comparisons, it means absolute perplexity numbers are not directly comparable to standard benchmarks. This is inherited from the parent experiment and should be fixed at the root.

No revisions needed. The experiment achieved its purpose: it tested a plausible hypothesis, found a clear negative, documented the evidence, and correctly identified next steps. This is what good micro/macro research looks like.
