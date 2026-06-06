# Peer Review: XSA Composition Quality

## NotebookLM Findings

Skipped -- the experiment is already self-killed with clean results. Deep review not warranted for a clear negative result; the effort is better spent verifying the kill is legitimate and the learnings are correctly extracted.

## Mathematical Soundness

### XSA Projection (correct)

The orthogonal projection z_i = y_i - (y_i^T v_i / ||v_i||^2) v_i is standard linear algebra. It correctly zeroes cos(z_i, v_i). The epsilon (1e-8) in the denominator prevents division by zero. The implementation in `run_experiment.py` lines 267-270 matches the math:

```python
dot_yv = mx.sum(y * v, axis=-1, keepdims=True)
norm_v_sq = mx.sum(v * v, axis=-1, keepdims=True) + 1e-8
y = y - (dot_yv / norm_v_sq) * v
```

This operates on (B, H, T, d_h) tensors correctly -- the projection is per-position, per-head. No shape errors.

### Interference Hypothesis (logically sound but empirically irrelevant)

The MATH.md argument in Section 3 is structurally correct: the self-value component a_{i,i} * v_i^{merged} does contain all adapter perturbations. Removing it does eliminate that interference channel. The error is not in the math but in the premise -- the Grassmannian A matrices already suppress inter-adapter interference to |cos| ~ 0.011, making the self-value interference channel negligible relative to the capacity cost of removing it.

### Composition Ratio Calculation (correct)

K2 computes per-domain, per-seed ratios (composed_PPL / single_PPL), then averages across all 15 data points (5 domains x 3 seeds). This is a reasonable aggregate. Verified against raw JSON: standard mean 1.9005, XSA mean 1.9206.

### K1 Calculation (correct but worth scrutinizing)

K1 takes the per-domain 3-seed mean, computes (XSA - standard) / standard * 100. The 11.85% figure is the repeat domain. Spot-checking from results.json:

- Standard repeat PPLs: 2.474, 3.064, 2.630 -> mean 2.723
- XSA repeat PPLs: 3.369, 2.811, 2.955 -> mean 3.045
- Degradation: (3.045 - 2.723) / 2.723 = 11.83%

This matches (rounding difference). The K1 kill is legitimate.

### Grassmannian A Generation (minor concern, non-blocking)

`make_grassmannian_As` (line 387) generates n independent QR decompositions. These are individually orthonormal but NOT jointly Grassmannian-optimized (no alternating projection for cross-adapter orthogonality). However, at d_in=128 and rank=8, random orthonormal frames are already nearly orthogonal by concentration of measure (|cos| ~ 1/sqrt(d_in*r) ~ 0.03), so this is adequate for micro scale. The low measured cosines (0.011) confirm this.

## Novelty Assessment

### Prior Art

XSA originates from arXiv 2603.09072 and the parameter-golf competition. The specific hypothesis -- that XSA reduces inter-adapter interference during composition -- appears novel. No prior work was found testing XSA in the context of LoRA merging.

### Reference: "Rethinking Inter-LoRA Orthogonality" (2510.03262)

Correctly cited in PAPER.md. This paper's finding that weight-space orthogonality does not equal semantic disentanglement is directly relevant and actually predicts the null result: even if XSA removes a weight-space interference channel, it does not follow that semantic interference decreases.

### HYPOTHESES.yml Missing

The experiment is listed in VISION.md under Track C but has no corresponding node in HYPOTHESES.yml. The kill criteria IDs (201, 202, 203) referenced in the code do not exist in HYPOTHESES.yml. This is a bookkeeping gap -- the experiment ran without formal hypothesis registration.

## Experimental Design

### Strengths

1. **Clean A/B comparison.** Standard vs XSA conditions are identical except for XSA in last 2 layers. Same seeds, same data, same training schedule.
2. **3-seed validation.** Adequate for distinguishing signal from noise at micro scale.
3. **Three independent kill criteria** testing different failure modes (capacity cost, composition benefit, per-domain wins).
4. **Diagnostic metrics.** Adapter cosine similarity provides mechanistic insight into why XSA fails (no interference to fix).

### Weaknesses

1. **Confounded base models.** The standard and XSA conditions train DIFFERENT base models (one with XSA, one without). This means the XSA condition starts from a different base quality level. Looking at the results, XSA base PPLs are substantially different from standard base PPLs (e.g., seed 42: standard parity base 10.32 vs XSA parity base 4.55). The adapters are then trained on top of these different bases. A cleaner design would train adapters on the SAME base and apply XSA only during composition/inference.

    The PAPER.md Limitations section (line 129) acknowledges this: "An alternative is to train without XSA and apply it only at inference during composition." This is an important confound but does not invalidate the kill -- it actually suggests a potentially cleaner experiment that was not run.

2. **No attention self-similarity diagnostic.** MATH.md Section 4 lists "Attention self-similarity cos(y_i, v_i) before/after XSA (diagnostic)" as metric #5, but this was never measured. This would have directly validated whether XSA actually reduces self-value bias at this scale. Without it, we cannot distinguish "XSA works but the capacity cost outweighs it" from "XSA does not reduce self-value bias at d_h=32."

3. **No significance testing.** With only 3 seeds, statistical power is limited, but at minimum a paired t-test on composition ratios would quantify whether the 1.06% difference is within noise. Given the small N, this is understandable but noted.

### Could a Simpler Explanation Account for the Result?

Yes. The most parsimonious explanation is: at d_h=32, removing 1 dimension (3.1% of capacity) per head is too costly. The PAPER.md correctly identifies this. The adapter cosine diagnostic (0.011 for both conditions) confirms that there is no interference to reduce. This is a clean negative result.

## Hypothesis Graph Consistency

The experiment is referenced in VISION.md (Track C: "XSA for composition quality improvement (exp_xsa_composition_quality)") but has no HYPOTHESES.yml node. This should be added as a killed node with the evidence from this experiment.

The kill criteria in the code (K1/K2/K3) are well-defined and were correctly evaluated. All three failed. The verdict of KILLED is appropriate.

## Macro-Scale Risks (advisory)

Even if this were to be revisited at larger scale:

1. **The fundamental insight holds:** Grassmannian A matrices already solve inter-adapter interference. XSA addresses a problem that does not exist in this architecture. At d_h=64 or d_h=128, the capacity cost would be lower (1.6% or 0.8%), but the interference is also already near-zero, so the benefit remains negligible.

2. **Inference-only XSA:** The confounded-base issue (weakness #1 above) means there is a variant not tested: train normally, apply XSA only during composition. This might avoid the capacity cost during training while still removing the self-value interference channel at merge time. However, given that adapter cosines are already ~0.011, the expected benefit is minimal.

3. **XSA may have value for non-Grassmannian systems.** If adapters were trained with standard random A matrices (higher interference), XSA might help. But that is a different architecture than the one being developed.

## Verdict

**PROCEED** (as a completed, killed experiment)

The experiment is correctly killed. All three kill criteria failed clearly:
- K1: 11.85% degradation (threshold 3%) -- not borderline
- K2: XSA ratio 1.9206 > standard 1.9005 -- wrong direction
- K3: XSA wins 2/5 domains -- majority loss

The experimental design is adequate despite the confounded-base issue, because the negative result is so clear that no reasonable correction would flip it. The key learning is correctly extracted: Grassmannian orthogonality already handles inter-adapter interference, making XSA's capacity cost unjustified.

**Recommended actions (non-blocking):**
1. Add a killed node to HYPOTHESES.yml with the evidence from this experiment.
2. Record in FINDINGS.md: "XSA for composition KILLED -- zero-parameter modifications are not free; Grassmannian A matrices already suppress interference to |cos| ~ 0.011."
3. Note the untested variant (inference-only XSA) as a low-priority future direction, not as an active research track.
