# Peer Review: BitNet Orthogonality Trained

## NotebookLM Findings

Skipped -- experiment is a clean negative result (KILLED) with straightforward design. The review below covers all critical dimensions without requiring external synthesis.

## Mathematical Soundness

**Correct:**
- The paired t-test implementation is sound: t(2) = 0.643, df=2, critical value 4.303 at alpha=0.05. Cannot reject H0. Verified against raw data in results.json.
- The 5.9% worse claim checks out: (0.2755 - 0.2600) / 0.2600 = 5.93%.
- Per-seed pass rates (1/3 K1 pass, 3/3 K2 pass) match the raw JSON.
- Cosine similarity computation via flattened weight deltas is standard and correctly implemented.

**Error in random baseline interpretation (non-blocking):**
The MATH.md presents two random baselines and conflates them:
1. sqrt(r/d) = sqrt(4/64) = 0.25 -- this is the expected overlap for rank-r subspaces in R^d (per Grassmannian theory).
2. sqrt(2/(pi*D)) where D=98,304 -- this is the expected |cos| for random unit vectors in R^D (the flattened dimension).

The text says "Both FP16 and ternary mean |cos| are close to the sqrt(r/d) = 0.25 random subspace bound, suggesting trained adapters are only marginally more aligned than random at this scale." This is misleading. The correct random baseline for flattened deltas is 0.0025, not 0.25. Observed values of ~0.26 are 100x above this baseline, meaning trained adapters are highly non-random in flat space. The sqrt(r/d) bound applies to structured rank-r subspace overlap, which is a different geometric object than cosine of flattened concatenated weight vectors.

This error does not affect the kill verdict (the comparison between FP16 and ternary is valid regardless of baseline interpretation), but it mischaracterizes the nature of what is being measured. If the adapters truly were "near random," the experiment would be uninformative.

**Hidden assumption (non-blocking):**
The gradient flow argument in "Why This Might Work" (Section: Prior Reasoning) is informal and post-hoc. The claim that "dead channels partition the feature space" was never formalized as a bound. The post-mortem explanation ("gradient averaging destroys channel structure") is equally informal. Both are reasonable intuitions but neither constitutes a mathematical proof. This is acceptable for a micro diagnostic experiment.

## Novelty Assessment

**Prior art:** The question "does a quantized base model produce more orthogonal adapters" is relatively novel. Most LoRA composition work (LoRAHub, TIES-Merging, DARE) studies merging strategies, not how base model properties affect adapter geometry. The BitNet b1.58 paper does not study LoRA composition at all.

**Delta over existing work:** This experiment provides a clean negative result that separates two mechanisms (base quantization vs. adapter quantization) in the BitNet-SOLE track. The finding that exp_bitnet_ternary_adapter_composition's -19.3% decorrelation comes from adapter quantization, not base quantization, is a genuine insight that prevents future confusion. No published work makes this distinction.

**References check:** No relevant prior art in references/ that already answers this question. The MoTE paper (2506.14435) uses ternary experts but does not measure adapter orthogonality as a function of base precision.

## Experimental Design

**Strengths:**
1. Clean A/B design: same architecture, same data, same seeds, same LoRA init. Only variable is base precision (FP16 vs post-quantized ternary). This is good experimental hygiene.
2. Three seeds with paired statistical test. Underpowered (acknowledged in Limitations) but correctly executed.
3. Full 10-pair cosine matrix provides granular per-pair analysis, revealing the important pattern that ternary worsens high-overlap pairs while marginally improving low-overlap pairs.
4. Kill criteria are pre-registered in HYPOTHESES.yml and correctly evaluated.

**Weaknesses:**
1. **K2 is not testing what it claims.** The HYPOTHESES.yml says K2 is "math-medical pair cos on BitNet >= 0.5 (high-overlap domains still collide)." The experiment uses arithmetic-sort as a proxy, but this pair has cos ~0.12 on both bases -- nowhere near the 0.5 threshold. The PAPER acknowledges this ("K2 PASS is trivially satisfied") but the kill criterion was poorly designed: the arithmetic-sort pair is not a valid proxy for math-medical overlap (cos=0.703 at macro). A better design would have chosen the highest-overlap pair (reverse-sort, cos ~0.77-0.83) as the K2 target. This is a design flaw, not a fatal one -- the experiment is already killed on K1.

2. **Post-quantization confound.** The ternary base is created by post-quantizing a trained FP16 model, not by training natively with ternary weights. The gradient dynamics during adapter training on a post-quantized base may differ from a natively-trained BitNet model. The PAPER acknowledges this (Limitation 2). This is acceptable for micro but should be noted for any macro follow-up.

3. **The experiment correctly identifies that the result is not statistically significant.** With n=3, power to detect Cohen's d=0.37 is ~6%. The paper is honest about this. The directional consistency (2/3 seeds) provides weak evidence. This is fine for a kill decision -- the burden of proof is on the hypothesis, and it failed to show even a trend in the right direction.

## Hypothesis Graph Consistency

- Kill criteria in HYPOTHESES.yml match those in the code and paper.
- K1 is correctly evaluated: ternary mean |cos| >= FP16 mean |cos| (0.276 >= 0.260). KILLED.
- K2 passes trivially (0.121 < 0.5) but is acknowledged as uninformative.
- Status "killed" in HYPOTHESES.yml is correct.
- `blocks: []` -- this experiment does not block anything, which is correct since it's a diagnostic.
- The evidence entry in HYPOTHESES.yml accurately summarizes the results.
- The connection to exp_bitnet_composition_stability and exp_bitnet_ternary_adapter_composition is correctly traced: this experiment disambiguates base vs. adapter quantization effects.

## Macro-Scale Risks (advisory)

1. **The kill may not transfer to scale.** At d=64, trained cosines (~0.26) are already 100x above random (0.0025), but at d=4096 the ratio could change. A natively-trained BitNet-2B might create genuinely different gradient dynamics than a post-quantized toy model. The paper's own Section "What Would Kill This (at Macro Scale)" correctly identifies this: if math-medical cos on BitNet-2B < 0.5, the hypothesis could be rescued.

2. **The "quantization recovery" explanation is untested.** MATH.md Section "Implication for SOLE Architecture" proposes that ternary composition advantage comes from magnitude bounding or quantization recovery. Neither mechanism has been directly measured. This is a separate experiment.

## Verdict

**PROCEED**

This is a well-executed negative result. The experiment cleanly tests its hypothesis (does ternary base improve adapter orthogonality?), gets a clear no, and correctly traces the implications for the broader BitNet-SOLE track. The kill is justified.

Specific notes:

1. The random baseline conflation (sqrt(r/d) vs sqrt(2/(pi*D))) in MATH.md should be corrected in any future reference to this work, as it understates how non-random the trained adapters actually are. This does not affect the verdict.

2. K2 was poorly designed (arithmetic-sort is not a valid math-medical proxy) but is moot given the K1 kill.

3. The disambiguation of base-quantization vs adapter-quantization effects is the real contribution of this experiment and is correctly highlighted in the paper's Connection to Prior Results section.
