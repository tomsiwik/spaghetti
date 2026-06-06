# Peer Review: bitnet_grassmannian_init

## NotebookLM Findings

Skipped -- the experiment is already self-killed with thorough analysis. The PAPER.md and MATH.md are unusually well-written for a killed experiment and contain their own adversarial decomposition. The review below validates whether the self-kill was correct and whether salvageable insights are properly scoped.

## Mathematical Soundness

**Welch bound application: correct.** At Nr=20 < d=64, mu_welch=0 is correctly computed. The paper correctly identifies that AP packing is trivially achievable in this regime. This is the single most important mathematical observation in the experiment and it invalidates the core hypothesis.

**STE gradient approximation: correctly stated.** dL/dW_latent = dL/dW_q is the standard STE formulation. No errors.

**Cosine metric: appropriate.** Flattened delta cosine over (N choose 2) = 10 pairs is consistent with prior SOLE experiments.

**Statistical test: adequate.** Wilcoxon signed-rank at p=0.020 over 30 cosine pairs (10 pairs x 3 seeds) is reasonable for a micro experiment. However, the per-seed ratios (0.739, 0.930, 0.762) show high variance -- seed 123 nearly shows no effect at all. With only 3 seeds, one outlier seed could flip the result.

**Scale confound correctly identified.** AP-init entries are ~1/sqrt(64) = 0.125 while random-init entries are ~0.01. This 12.5x magnitude difference means the two conditions enter QAT in fundamentally different quantization regimes. The paper acknowledges this in Limitation 3 but does not control for it. A random-orthonormal condition (same orthonormality, no AP geometry) would isolate the mechanism. This is a real confound but does not invalidate the self-kill verdict.

**Composition PPL analysis: sound reasoning.** The observation that composed/individual PPL ratio (0.927/0.859 ~ 1.08) suggests composition benefit tracks individual quality improvement, not reduced interference, is a well-reasoned decomposition.

## Novelty Assessment

**Not novel as stated.** The experiment intended to test whether AP packing geometry survives ternary QAT. It actually tested whether orthonormal-init helps ternary QAT -- a much simpler and less novel question. The "orthonormal init helps quantization" observation is directionally consistent with standard QAT literature (e.g., the known advantage of well-conditioned weight matrices for quantization).

**Prior art within the project:**
- grassmannian_expert_init: already proved AP packing works at FP16 (1.23-1.52x improvement at d=128-256). This experiment adds the ternary dimension but at a scale where AP is irrelevant.
- bitnet_ternary_adapter_composition: already showed ternary QAT decorrelates experts by 19.3%. This experiment adds AP-init but cannot distinguish AP from orthonormal-init.

**Salvageable insight (orthonormal-init for ternary QAT) is modestly novel** within the project but not publishable on its own. It is a practical recommendation, not a scientific finding.

## Experimental Design

**The experiment does not test what it claims.** The stated hypothesis is about Grassmannian AP packing surviving ternary QAT. At Nr=20 < d=64, AP packing is trivially achievable -- there is no packing pressure. The experiment actually tests orthonormal-init vs Gaussian-init for ternary QAT. The paper correctly identifies this confound.

**Missing control: random-orthonormal init.** This would isolate the AP packing effect from the orthonormality effect. The grassmannian_expert_init experiment had a three-condition design; this experiment regressed to a two-condition design, which is insufficient to attribute the improvement to AP specifically.

**Kill criteria were correctly pre-registered.** K1 requires ratio < 0.70 (>30% improvement). The observed ratio is 0.808. The kill is clean and honest. The kill criterion was arguably too strict for a micro experiment, but the researcher pre-registered it and stuck to it, which is commendable.

**K2 (quality) passes comfortably.** AP-init produces 14% better individual PPL. This is a genuine and surprising finding, though not the one the experiment was designed to test.

**Composition evaluation is concerning.** Both conditions show composed PPL 6-8x worse than individual PPL (30.5 vs 4.2). At this degradation level, composition is fundamentally broken regardless of initialization, and small differences in composed PPL are dominated by noise.

## Hypothesis Graph Consistency

**Node status: killed.** Correct. K1 fails, experiment is killed.

**Kill criteria match:** The HYPOTHESES.yml K1 says "mean |cos| > 0.7x random-init (no improvement)." The observed ratio is 0.808. The criterion as written in HYPOTHESES.yml is slightly ambiguous -- "0.7x" could mean "70% of random-init" (which 0.808 > 0.7 so it would be killed) or it could be read differently. Looking at PAPER.md, the threshold is "ratio < 0.70" for passing, which means the kill fires at ratio >= 0.70. The observed 0.808 >= 0.70, so: killed. Consistent.

**Dependencies:** Depends on exp_bitnet_ternary_adapter_composition (proven). No downstream blocks. Clean kill with no cascade impact.

**"5th consecutive BitNet-track kill"** noted in the evidence. This is a strong signal that the BitNet track (Track A in VISION.md) is accumulating negative evidence. The reviewer notes this is a pattern worth flagging -- not as a critique of this experiment, but as strategic advice.

## Macro-Scale Risks (advisory)

Not applicable -- experiment is killed. However, the salvageable insight (orthonormal-init for ternary QAT) has a specific scale risk: at d=4096, random Gaussian matrices are already near-orthonormal due to concentration of measure (cos ~ 1/sqrt(d) ~ 0.016). The 14% quality advantage seen at d=64 may vanish at production scale. The paper acknowledges this in Limitation 2.

## Verdict

**PROCEED** (as a self-kill confirmation)

The experiment was correctly killed at K1. The self-analysis in PAPER.md is thorough, honest, and identifies the right confounds. No mathematical errors found. The kill criteria were pre-registered and applied consistently. The salvageable insight (orthonormal-init helps ternary QAT by 14%) is properly scoped as a practical finding, not overclaimed.

No revisions needed. The experiment, its kill, and its documentation are all sound.

One strategic note: the 5-consecutive-kill pattern on the BitNet track should prompt a meta-level decision about whether to continue investing in Track A (BitNet-SOLE) or consolidate on Track B (FP16 + Smart Routing). This is not a critique of this experiment but of resource allocation.
