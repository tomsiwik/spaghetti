# Peer Review: notebooklm_ternary_training_sota

## NotebookLM Findings

This is a literature survey experiment, not an empirical run. The review
therefore focuses on: (1) accuracy of mathematical claims, (2) fidelity of
paper citations, (3) quality and actionability of recommendations, and
(4) missing work.

## Mathematical Soundness

### The erf derivation contains a symbolic error

The central claim is that the dead-weight fraction is a fixed point determined
by the error function. The derivation in MATH.md Section 0 proceeds:

```
alpha = mean(|W|) = sqrt(2/pi) * sigma       (correct, half-normal mean)
P(|w| < alpha/2) = erf(alpha / (2*sqrt(2)*sigma))   (correct formula)
```

Substituting alpha = sqrt(2/pi) * sigma into the argument:

```
alpha / (2*sqrt(2)*sigma) = sqrt(2/pi) / (2*sqrt(2)) = 1 / (2*sqrt(pi))
```

However, MATH.md claims the result is erf(1/(2*sqrt(2))), not erf(1/(2*sqrt(pi))).
This is a symbolic simplification error:

- erf(1/(2*sqrt(pi))) = erf(0.2821) = 0.3108  (matches empirical 31.3%)
- erf(1/(2*sqrt(2))) = erf(0.3536) = 0.383    (does NOT match)

The numerical answer 0.3108 is correct for the right expression. The error is
in the intermediate symbolic step, not the conclusion. This should be fixed to
avoid confusing anyone who checks the algebra.

**Impact: Low.** The final number is right. The derivation logic is sound. Only
the symbolic simplification sqrt(2/pi)/(2*sqrt(2)) was incorrectly reduced to
1/(2*sqrt(2)) instead of 1/(2*sqrt(pi)).

### The self-reinforcing equilibrium argument is qualitatively sound

The claim that if dead weights escape, alpha increases, raising the threshold
and re-trapping others, is correct directionally for W ~ N(0, sigma) with
alpha = mean(|W|). However, MATH.md does not prove this is a *stable*
equilibrium in the dynamical-systems sense (i.e., that perturbations around
31% always decay). It is presented as intuitive, which is fine for a survey,
but it is not a rigorous stability proof. For a full treatment, one would need
to show d(zero_fraction)/d(alpha) > 0 at the fixed point, creating negative
feedback. This is straightforward but absent.

### Hestia convergence claim is correctly cited

The statement that Lemma 4.2 proves pointwise convergence as tau -> 0+ is a
standard result for Gibbs distributions. The gradient expression for the
surrogate is correctly derived. The claim that gradient is "nonzero everywhere"
at finite tau is correct: the softmax probabilities are strictly positive for
finite tau, so the expected value w_eff is a smooth function of w.

### FOGZO orthogonality argument is correct but incomplete

The claim that when STE gradient g is orthogonal to true gradient g*, the
biased component vanishes, is correct by construction. However, the important
case is not g perpendicular to g* (rare in practice) but g having a *biased*
component. FOGZO's actual benefit is variance reduction via the prior, not
just the orthogonal case. The survey simplifies this to the limiting case.
Acceptable for a survey but slightly misleading about the mechanism.

### TernaryLM threshold comparison has a subtle issue

MATH.md states: "New threshold: 0.5*std(W) = 0.5*sigma" and "Old threshold:
alpha/2 = 0.399*sigma." The claim is that 0.5*sigma > 0.399*sigma, so MORE
weights are zeroed initially. This is correct. But the PAPER.md and LEARNINGS.md
then claim the *dynamics* favor TernaryLM during training. No mathematical
argument or citation is provided for why std-based coupling has better dynamics
than mean-based coupling. The only argument is qualitative: "std(W) responds
differently." This is the weakest claim in the survey -- the core selling point
of TernaryLM threshold change has no rigorous backing for why the equilibrium
shifts favorably during training.

## Novelty Assessment

This is a survey, not a novel contribution. Novelty assessment is therefore
about whether the synthesis adds value:

1. **The three-class taxonomy (smooth quantizer / decouple threshold / fix
   gradient) is useful.** It provides a clear framework for evaluating future
   methods. This is original organizational work.

2. **The combined TernaryLM + Tequila recommendation is novel in the sense
   that no paper proposes this combination.** The orthogonality argument
   (one changes WHERE the boundary is, the other compensates for WHAT happens
   inside) is sensible but untested.

3. **The connection to adapter composition (Section 7 of MATH.md) is the
   survey's main contribution** -- linking general ternary training improvements
   to the specific problem of composable experts. No cited paper addresses this.

## Paper Citation Accuracy

### Verified claims:
- BitNet b1.58 (2402.17764): ternary {-1,0,+1} via STE + absmean -- correct
- MatMul-free LM (2406.02528): ternary + addition-only inference to 2.7B -- correct
- GaLore (2403.03507): low-rank gradient training -- correct

### Flagged issues:

1. **Tequila arxiv ID uncertainty.** The PAPER.md Limitations section (item 5)
   already flags this: "arXiv:2509.23800 ... resolves to a different paper."
   The LEARNINGS from exp_tequila_deadzone_fix uses 2509.23809. Both may be
   wrong. This is acknowledged but still a problem -- the survey cites a paper
   it cannot definitively link to an arxiv ID. The Tencent/AngelSlim codebase
   is the actual source.

2. **TernaryLM (2602.07374), Hestia (2601.20745), FOGZO (2510.23926),
   PT2-LLM (2510.03267):** These are Feb/Oct 2025-2026 papers. I cannot verify
   these arxiv IDs resolve correctly, but the method descriptions are internally
   consistent and detailed enough to have come from actual papers.

3. **1-Bit Wonder (2602.15563):** The claim "31B in 7.7 GB" and "1.25 bits/weight"
   is internally consistent (31B * 1.25 / 8 = 4.84 GB for weights, plus
   overhead for scales). Reasonable.

4. **Sparse-BitNet (2603.05168):** Cited for "42% natural sparsity." This
   conflicts with the 31% prediction from the erf calculation. The discrepancy
   is likely due to different threshold definitions (per-channel vs per-tensor,
   or different alpha computation). Not addressed in the survey.

## Experimental Design

As a survey, the "experiment" is the quality of literature synthesis and the
actionability of recommendations. Evaluating on those terms:

**Strengths:**
- Clear prioritization with effort/risk/gain tradeoffs
- Honest about what was NOT found (Sherry, Falcon-Edge training innovation)
- Builds on validated prior result (exp_tequila_deadzone_fix)
- Recommendations are concrete: code snippets, time estimates, kill criteria
- Three-phase plan (immediate / validation / scale) is well-structured

**Weaknesses:**
- No computational validation of any recommendation beyond Tequila
- The "combined TernaryLM + Tequila" recommendation is speculative (the
  interaction could be subadditive or even harmful if std-threshold changes
  which weights are dead in a way that reduces Tequila bias signal)
- Missing quantitative prediction: what zero fraction should TernaryLM produce?
  For W ~ N(0, sigma), threshold = 0.5*sigma gives P(|w| < 0.5*sigma) =
  erf(0.5/sqrt(2)) = erf(0.3536) = 0.383. So TernaryLM predicts 38.3% zeros
  initially (MORE than BitNet's 31%). The survey notes this but does not
  compute the number. This is a testable, falsifiable prediction that should
  be stated explicitly.

**Controls:**
- Kill criteria (K1) in PAPER.md are well-defined: "no method reduces zero
  fraction below 25% AND no method improves PPL beyond -6.7%"
- The prior experiment (exp_tequila_deadzone_fix) provides a concrete baseline

## Hypothesis Graph Consistency

The experiment is registered as a survey and marked SUPPORTED. For a survey
experiment, "supported" means "produced actionable recommendations backed by
literature." This is met:

- 5 methods identified with clear prioritization
- 3 are immediately implementable on MLX
- Concrete next experiments proposed with kill criteria
- Prior validated result (Tequila) correctly incorporated

The kill criterion ("no actionable recommendations") is clearly not triggered.
The status is appropriate.

## Macro-Scale Risks (advisory)

1. **TernaryLM threshold has no proven benefit at any scale for composition.**
   All TernaryLM results are for individual model quality. Different zero
   patterns could change adapter interference characteristics unpredictably.
   The micro-experiment should explicitly test composition, not just PPL.

2. **Hestia's Hutch++ at d=2560 with 192 layers.** Each Hutch++ call requires
   O(m) Hessian-vector products. At m=3, that is 576 extra HVPs per step.
   On M5 Pro, this could dominate training time. The "simplified Hestia"
   (uniform schedule) is the right first step.

3. **Tequila's bias fusion assumes dead weight set is stable after training.**
   If the dead set changes between training and deployment (e.g., due to
   weight quantization rounding), the fused bias is stale. At micro scale
   this is not an issue, but at production scale with different quantization
   implementations, it could matter.

4. **The 42% vs 31% sparsity discrepancy** (Sparse-BitNet vs this survey's
   prediction) suggests the Gaussian assumption may not hold at scale, or
   that different implementations produce different equilibria. Worth
   investigating before committing to a specific zero-fraction target.

## Missing Work

1. **GPTQ/AWQ-style post-training ternary quantization.** While PT2-LLM is
   covered, the broader family of activation-aware quantization methods is
   not discussed. Some may be relevant for the adapter quantization path.

2. **OneBit (Xu et al., 2024):** 1-bit weight quantization with value-aware
   knowledge distillation. Relevant because it directly addresses the capacity
   loss from aggressive quantization.

3. **No discussion of ternary training stability literature.** Several papers
   discuss training instability with low-bit quantization (gradient explosion,
   loss spikes). The survey focuses on the deadzone but ignores other failure
   modes of ternary training.

## Verdict

**PROCEED**

This is a well-executed literature survey that correctly identifies the core
mathematical mechanism (alpha-coupling equilibrium), provides a useful taxonomy
of solutions, and produces actionable recommendations. The one symbolic error
in the erf derivation does not affect conclusions.

**Minor revisions recommended (not blocking):**

1. Fix the symbolic expression: erf(1/(2*sqrt(2))) should be erf(1/(2*sqrt(pi))).
   The numerical value 0.3108 is correct for the right expression.

2. Add the quantitative prediction for TernaryLM initial zero fraction:
   P(|w| < 0.5*sigma) = erf(0.5/sqrt(2)) = 0.383. This is higher than
   BitNet's 0.31, making it a concrete falsifiable prediction for the
   follow-up experiment.

3. Note the Sparse-BitNet 42% vs predicted 31% discrepancy and hypothesize
   why (per-channel alpha, non-Gaussian tails, different threshold formula).

4. Strengthen the TernaryLM dynamics argument or downgrade from "RECOMMENDED
   Priority 1" to "RECOMMENDED Priority 2, needs theoretical justification."
   Currently the claim that std-coupling has better dynamics than mean-coupling
   is unsupported by any rigorous argument.
