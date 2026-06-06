# Peer Review: Hierarchical Expert Composition

## NotebookLM Findings

Skipped -- experiment is a clean negative result with straightforward math. Deep review not warranted for a KILL confirmation.

## Mathematical Soundness

**Derivations are correct.** The setup is clean:

1. Rank budget equalization (r_flat = r_f + r_s = 8 = 4 + 4) is the right control. Without this, any comparison would be confounded by capacity differences.

2. The SVD-based foundation extraction (Section 2.3 of MATH.md, lines 57-66 / code lines 89-152) is mathematically sound. Stacking domain deltas, computing SVD, projecting onto top-r shared directions, then taking the residual -- this is standard subspace extraction.

3. Kill criterion K1 is correctly formulated: hierarchy must beat flat on within-cluster queries specifically (not overall), since that is where the hypothesis predicts improvement.

**One hidden assumption worth flagging (non-blocking):** The foundation extraction averages projections across cluster domains (code line 141: `foundation[key] = onp.mean(projections, axis=0)`), then re-truncates via SVD. This double truncation (project onto shared Vr, average, SVD-truncate again) loses information compared to a single optimal rank-r_f approximation. However, since both extraction and the hypothesis itself failed, this is moot.

**Statistical note:** The p-value of 0.381 on the t-test means the 2.99pp difference is not statistically significant. The PAPER correctly reports this. The kill is still justified because: (a) the hypothesis predicted hierarchy would be *better*, not just not-worse, and (b) the direction of the effect is consistently wrong across seeds. A non-significant result in the wrong direction is evidence against the hypothesis.

## Novelty Assessment

**This is a competent negative result, not a novelty claim.** The experiment correctly tests whether AdapterFusion-style two-level composition (cited) adds value over flat PPL-probe weighting. The answer is no.

**Prior art coverage is adequate.** The PAPER cites MoE-Adapters4CL, AdapterFusion, and X-LoRA as related hierarchical adapter approaches. None of these are in `references/REFERENCES.yml`, but since the experiment was killed, this is not blocking.

**One missing reference (non-blocking):** LoRAHub (Huang et al., 2023) explores dynamic LoRA composition with learned coefficients, which is closer to what PPL-probe weighting does. Worth noting that the winning strategy (flat + PPL-probe) is essentially a simplified LoRAHub.

## Experimental Design

**The experiment tests exactly what it claims.** Five strengths:

1. **Proper controls.** Four strategies tested (flat_equal, flat_ppl, hier_equal, hier_ppl) with the right comparisons: hier_ppl vs flat_ppl for the main kill criterion, plus equal-weight variants as secondary evidence.

2. **Cluster structure is grounded.** The "symbolic" vs "string" clustering is motivated by prior experimental evidence (exp_orthogonality_by_domain_type: 7.84x within-cluster cosine), not arbitrary.

3. **Within vs across-cluster decomposition.** The analysis correctly separates these cases and finds the nuanced result: hierarchy marginally helps across-cluster (+0.64pp) but hurts within-cluster (-2.99pp). The PAPER interprets this correctly.

4. **5 seeds.** Adequate for micro-scale. Per-type breakdown reveals the variance is driven by specific pair interactions (reverse_repeat -8.00pp, repeat_sort -7.05pp), not random noise.

5. **The explanation for failure is convincing.** Section "Why It Failed" in PAPER.md correctly identifies that PPL-probe already solves the dilution problem that hierarchy was designed to address, making the structural overhead pure cost with no benefit.

**One design concern (minor):** The PPL-probe for hierarchical experts (code lines 449-460) scores foundation+specialist combos, not specialists alone. This is correct -- it tests the full hierarchical pipeline. But it means the hierarchy has a slight disadvantage: its probe must evaluate a more complex composition, potentially misweighting when the foundation adds noise. This concern is minor because the equal-weight comparison (hier_equal vs flat_equal) shows the same pattern: -7.29% vs -7.25%, essentially identical.

**Could a simpler explanation account for the results?** Yes, and the PAPER identifies it: the rank-4/4 split is suboptimal for clusters with low sharing (symbolic: cos=0.05-0.13). The foundation wastes 4 ranks on noise for arithmetic+parity. An adaptive split (more foundation rank for high-sharing clusters, less for low-sharing) might help, but this adds complexity that PPL-probe avoids entirely. The PAPER acknowledges this in Limitations point 4.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry matches the experiment:
- Kill criterion "hierarchical is NOT better than flat for clustered domains" maps to K1.
- Kill criterion "hierarchy adds >30% complexity for <5% quality improvement" maps to K2.
- Status is correctly set to `killed`.
- Evidence summary is accurate.
- `blocks: []` -- correct, no downstream experiments depend on this.

## Macro-Scale Risks (advisory)

Not applicable -- experiment was killed. No macro follow-up is warranted.

However, for the record: the core insight (PPL-probe weighting at runtime beats structural hierarchy at training time) is an important architectural decision for SOLE. If someone revisits this at macro, the key question would be whether real domain clusters (e.g., medical subspecialties) have substantially higher shared subspace (cos > 0.5) than the synthetic clusters tested here. Even then, PPL-probe's adaptivity advantage likely dominates.

## Verdict

**PROCEED** (with the kill)

The experiment is well-designed, the math is sound, the result is a clean negative, and the kill is justified. The analysis correctly identifies why hierarchy fails (PPL-probe already solves dilution without structural constraints) and draws the right implication for SOLE (flat composition + runtime weighting > structural hierarchy).

No revisions needed. The PAPER and MATH documents are publication-ready as negative results. The finding reinforces a key SOLE design principle already documented in VISION.md: adaptive weighting at inference beats rigid structure at training time.
