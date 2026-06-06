# Peer Review: Energy-Gated Composition

## Experiment Type
Guided exploration (Type 2). MATH.md cites a proven framework (Neyman-Pearson lemma) and identifies three unknowns: optimal threshold, whether gating recovers prose quality, whether selective composition preserves structured wins.

## Hack Detector
- Fix count: 1 (single gating mechanism). No flags.
- Is MATH.md a proof or a description? **Description dressed in equations.** The Neyman-Pearson lemma is correctly stated, and the mapping of energy gap to log-likelihood ratio is valid algebra. But the critical step -- that the test has power (i.e., that the two hypothesis distributions are separable) -- is assumed, not proven. The framework proves that IF you have a good test statistic, the likelihood ratio is optimal. It does NOT prove that the energy gap produces separable distributions in this setting. This is a mechanism description using a theorem as decoration, not a proof that the mechanism works.
- Metric used as evidence: Domain-specific quality scores (keyword density proxies). Not proven to predict behavioral outcomes, but adequate for directional signal.
- Kill criteria source: K572 and K573 are derived from the predictions in MATH.md (gated > base on >= 4/5, gated > uniform on 5/5). K574 is a reasonable overhead bound. These are well-grounded.

## Self-Test Audit

1. **One-sentence impossibility property:** "The Neyman-Pearson energy gate excludes adapters that increase NLL, making it impossible for harmful adapters to participate." This is a circular assertion, not an impossibility property. The experiment itself proved that "increase NLL" and "harmful" are different things -- all adapters decrease NLL yet some are harmful to generation quality. The claimed impossibility was vacuous from the start.

2. **Cited theorems:** Neyman-Pearson lemma (1933) -- real theorem, correctly stated. However, the precondition that matters is that H0 and H1 must be simple hypotheses with known distributions. In this setting, neither P(x|adapter helps) nor P(x|adapter hurts) is known. The lemma's application is aspirational, not rigorous.

3. **Predicted numbers:** Specific and falsifiable. 7 quantitative predictions in a table. Good.

4. **Falsification condition:** "If energy gap on prompt tokens does NOT predict which adapters help during generation, the gate makes wrong decisions." This correctly identifies the key assumption. However, it should have been tested BEFORE committing to the full generation experiment -- a cheap pre-check measuring the correlation between energy gap sign and generation quality would have killed this in minutes.

5. **Hyperparameter count:** 1 (threshold tau). Acknowledged as a Type 2 unknown. Acceptable.

6. **Hack check:** Clean. Single mechanism replacing uniform composition. No layered fixes.

## Mathematical Soundness

The algebra mapping energy gap to Neyman-Pearson log-likelihood ratio (Step C) is correct:

    log Lambda(x) = log P(adapted) - log P(base) = -NLL_adapted + NLL_base = -Delta_E

This is valid. The issue is not the algebra but the missing theorem:

**Missing Theorem (the one that would have mattered):** "LoRA adapters of rank r applied to a pretrained model of dimension d reduce NLL on ANY input distribution, not just the training distribution." This is provable: adding rank-r parameters to any linear layer creates a strictly more expressive model in the NLL sense (the base model is a special case at A=B=0, so the optimizer can only do better or equal). This means Delta_E <= 0 almost surely, which means the Neyman-Pearson threshold at tau=0 has zero rejection rate.

If MATH.md had proven this negative result first, the experiment would never have been run. The framework correctly identifies the test statistic but fails to check whether the test has any power. A one-paragraph proof would have shown it cannot.

**Assumption 1 ("energy gap generalizes to generation quality") was falsifiable a priori.** The fact that adding parameters to a model can only reduce NLL (by the argument above) means the energy gap is always negative, making binary gating vacuous. This is not an empirical surprise -- it is a mathematical necessity that should have been caught in the proof stage.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table. Credit for this.

| Prediction | Measured | Match? |
|-----------|----------|--------|
| Gated > base on >= 4/5 | 1/5 | NO |
| Gated > uniform on 5/5 | 0/5 | NO |
| Code >= +10% | -13.5% | NO |
| Math >= +50% | +49.8% | MARGINAL |
| Prose >= -2% vs base | -5.2% to -6.8% | NO |
| Overhead < 20% | 4.7% | YES |
| Energy gap gates selectively | 100% inclusion, zero gating | NO |

6 of 7 predictions failed. The experiment was properly killed.

## NotebookLM Findings

Skipped -- the experiment is already killed with thorough root cause analysis in PAPER.md. NotebookLM review would not change the verdict.

## Novelty Assessment

The Neyman-Pearson framing of adapter selection is mildly novel as a presentation, but the underlying idea (use NLL ratio to gate adapters) is straightforward. The experiment's main contribution is the negative result: proving empirically that NLL-based gating is vacuous for LoRA adapters because they universally reduce NLL.

This negative result is genuinely valuable and should be recorded as a finding with proper impossibility structure.

## Macro-Scale Risks (advisory)

Not applicable -- the mechanism is killed. However, the impossibility result (LoRA always reduces NLL) scales to any model size and adapter configuration. This is a structural property of overparameterization, not a micro-scale artifact. Any future NLL-based gating proposal must address this.

## Strengths (Credit Where Due)

1. **Excellent post-mortem.** PAPER.md's root cause analysis (Section "What Went Wrong") is thorough and correctly identifies the structural impossibility. The distinction between ranking and gating (Finding #182 does ranking, this experiment needs gating) is precisely articulated.

2. **Clean experimental design.** Three seeds, five domains, threshold sweep, timing analysis. The experiment itself is well-executed.

3. **Self-killing.** The experiment correctly identified its own failure and produced actionable follow-up directions (relative ranking instead of absolute gating).

4. **MATH.md Assumption 1** explicitly flagged the risk that ultimately killed the experiment. The framework was honest about its vulnerabilities.

## Critical Weaknesses

1. **The proof has no theorem.** MATH.md contains a framework description with equations but no Theorem/Proof/QED block. The Neyman-Pearson lemma is cited but its preconditions are not verified for this setting. A proper proof would have required showing that the energy gap distributions under H0 and H1 are separable -- which would have immediately revealed they are not.

2. **The impossibility was provable a priori.** Adding rank-r parameters to a linear model creates a strictly larger hypothesis class. The minimum NLL over the larger class is <= the minimum over the smaller class. Therefore Delta_E <= 0 for any well-trained adapter on any input. This kills the gating mechanism before any experiment is run.

3. **Finding #182 was misinterpreted.** The AUC=0.851 measures ranking quality (which adapter is better), not gating quality (which adapters to include). MATH.md acknowledges this in Assumption 2 but does not prove that ranking implies gating. PAPER.md correctly identifies this disconnect post-hoc -- but it should have been caught pre-experiment.

## Verdict

**KILL**

The experiment correctly killed itself. The post-mortem analysis is strong and the impossibility structure is well-identified. The kill is confirmed for three reasons:

1. **Structural impossibility:** LoRA adapters universally reduce NLL (proven by overparameterization). Binary energy gating at any threshold <= 0 is vacuous.

2. **No theorem in MATH.md:** The framework describes a mechanism but does not prove the mechanism has discriminative power. The missing proof (that energy gap distributions are separable) would have revealed the impossibility before any compute was spent.

3. **6/7 quantitative predictions failed.** The measurement table is definitive.

**Impossibility structure to record:** For any LoRA adapter of rank r > 0 applied to a model of dimension d, the adapted model contains the base model as a special case (A=B=0). Therefore, any optimizer achieving near-optimal NLL will satisfy NLL_adapted <= NLL_base on all inputs, making Delta_E <= 0 universally. Binary gating on sign(Delta_E) has zero rejection rate. This is a property of overparameterized function approximation, not of the specific model or domain.

**What survives:** The energy gap as a RANKING signal (not gating signal) for top-k adapter selection. This is the correct interpretation of Finding #182 and should be tested separately.
