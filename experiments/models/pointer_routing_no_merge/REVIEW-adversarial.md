# Peer Review: Pointer Routing (No Merge)

## NotebookLM Findings

Skipped -- sufficient depth achieved via direct code and results analysis.

## Mathematical Soundness

### MATH.md Derivations

**Output-space vs parameter-space argument (Section 2):** The derivation is correct for the linear case. The paper correctly notes that sigma(E[f]) != E[sigma(f)] via Jensen's inequality. However, the argument is incomplete: Jensen tells you the DIRECTION of the inequality (for convex/concave sigma), not the MAGNITUDE. The paper never bounds HOW MUCH the mismatch costs. The 11.9% empirical gap could be mostly from 1/N dilution (a linear scaling effect) rather than nonlinearity mismatch (a second-order effect). These two confounds are never separated.

**Specific issue: 1/N dilution vs nonlinear mismatch confound.** Uniform 1/N applies scale/N = 4.0 per adapter. Oracle single applies scale = 20.0 for one adapter. The paper attributes the gap to "output-space composition preserving nonlinearity." But a simpler explanation exists: the correct adapter at 5x scale contributes more signal than 5 adapters each at 1x (where 4 of the 5 are wrong-domain). To isolate the nonlinearity effect, you would need to compare: (a) single correct adapter at scale 20.0 vs (b) single correct adapter at scale 4.0. If (a) >> (b), the effect is mostly scale, not nonlinearity. This control is missing.

**Complexity analysis (Section 5):** The FLOPs table claims uniform 1/N uses N * (d*r + r*d) * L * 7 = 172M. This is correct in the sense that all N adapters are applied. However, the claim that pointer routing uses "5x fewer adapter FLOPs" conflates computation with quality. The fair comparison is at matched compute: pointer routing at 1 adapter vs uniform at N adapters is not iso-compute.

**Hash routing formulation (Section 3, variant b):** The implementation (line 364-372 in run_experiment.py) uses `np.random.RandomState(seed + domain_idx)` to generate random assignments. This is NOT a hash function -- it is a pseudo-random per-domain assignment. A true hash would be deterministic on (layer_id, input_features), not on (seed, domain_index). The domain_idx is the GROUND TRUTH label, meaning the hash has access to the oracle domain identity. Despite this oracle access, it still performs -1.1% worse than uniform. This is a stronger negative result than the paper acknowledges.

**Learned gate entropy analysis is misleading.** The specialization analysis reports learned gate entropy_ratio = 0.9999 (near maximum entropy). This sounds like high diversity, but the measurement is ACROSS DOMAINS, not across layers. Each domain picks a different single adapter (medical=0, code=1, math=2, legal=3, finance=4), giving maximum cross-domain entropy. But within each domain, every layer picks the same adapter (cross_layer_variation = 0.0). The K2 kill criterion correctly catches this, but MATH.md's entropy metric is measuring the wrong thing -- it measures whether different domains use different adapters (trivially yes) rather than whether a single domain uses different adapters at different layers (the actual hypothesis).

### Hidden State Extraction Bug

Phase 4 extracts hidden states from the BASE model (no adapters). But Phase 5 applies these hidden states to train gates that route in the POINTER-ROUTED model. The hidden states at each layer will be different when adapters are active. The gates are trained on stale representations. For the learned linear gate, this does not matter because ridge regression on 5 trivially-separable domains produces correct domain classification regardless. But for the MLP gate, this distribution shift between training and inference hidden states is a plausible contributor to its failure, beyond the float32 overflow issue.

## Novelty Assessment

**Prior art overlap:** The finding that per-sequence routing beats per-layer routing is not novel in the MoE literature. Switch Transformer (Fedus et al.) trains experts FOR per-layer routing from scratch. The paper's insight that pre-trained joint-layer adapters cannot be mixed across layers is a negative finding about a specific training/inference mismatch, not a new routing mechanism.

**Relation to existing FINDINGS.md:** The project already has: (1) per-adapter tiny routing heads at 100% accuracy and +19.9% over uniform, (2) top-2 routing at +13.9% over uniform, (3) per-token Gumbel-sigmoid routing. The oracle single-adapter result here (11.9% over uniform) is WEAKER than the existing tiny routing heads finding (19.9% over uniform). The experiment confirms what was already known (route to correct adapter > uniform merge) without advancing the mechanism.

**True delta:** The only genuinely new finding is the negative result -- that per-layer adapter mixing actively hurts. This is useful for ruling out a design branch.

## Experimental Design

### What the experiment actually tests

The experiment claims to test "per-layer expert selection at full strength beats uniform 1/N." But the winning method (learned gate) does NOT perform per-layer selection. It performs per-sequence selection that happens to use per-layer gates which all agree. The experiment actually tests: "does oracle per-sequence routing beat uniform 1/N?" The answer (yes, by 11.9%) was already established by prior experiments.

### MLP Gate is Not a Valid Comparison

The MLP gate assignments are IDENTICAL across all 5 domains (verified in results.json: medical, code, math, legal, finance all get the exact same 30-layer assignment [4,4,4,4,4,2,4,4,4,3,3,2,3,3,1,3,2,2,2,3,2,2,3,2,1,2,0,0,2,2]). This means the MLP gate completely failed to learn domain discrimination. The paper attributes this to "float32 overflow in numpy matmuls" but calls it a routing variant rather than a broken baseline. A routing method that assigns the SAME layer pattern to all domains regardless of input is not routing -- it is a fixed permutation. The MLP result should be excluded from comparative analysis or clearly labeled as "failed to train."

### Controls Missing

1. **Scale control:** No comparison of single adapter at scale=20 vs single adapter at scale=4. This would separate the dilution effect from the nonlinearity effect.
2. **Random single-adapter control:** What happens if you pick a RANDOM (wrong) single adapter at full strength? If random single-adapter at scale=20 is comparable to uniform 1/N at scale=4, the benefit is purely from scale, not from routing.
3. **Top-2 pointer routing:** VISION.md reports top-2 routing at +13.9% over uniform. How does pointer routing at k=2 (two adapters per layer at scale/2) compare?

### The S1 Assessment is Incorrect in results.json

results.json reports `"s1_pass": true` with `"verdict": "SUPPORTED"`. But the code defines S1 as requiring BOTH >10% improvement AND k2_pass (line 1013: `s1_pass = best_improvement > 10.0 and k2_pass`). K2 fails for the learned gate. The overall verdict logic (lines 1043-1051) only reaches "SUPPORTED" if k1_pass AND k2_pass, yet k2_pass is determined by `any(k2_results[m]["pass"] for m in k2_results)` (line 1008), which is True because HASH routing passes K2. So the code says: "K2 passes because hash routing specializes" -- but hash routing FAILS K1. The experiment finds no method that passes BOTH K1 and K2 simultaneously. The reported "SUPPORTED" verdict is the result of ORing K2 across methods while only checking K1 for the best method. This is a logical error in the evaluation code. PAPER.md correctly identifies this as "SUPPORTED with major caveat" and gives the right interpretation, but the machine-readable verdict is wrong.

## Hypothesis Graph Consistency

- K1 (id 535): The paper claims PASS. The learned gate beats uniform by 11.9%. Technically correct, but the learned gate is not doing pointer routing -- it is doing oracle single-adapter selection. Honest assessment: K1 PASS for oracle routing, K1 FAIL for actual pointer routing (hash: -1.1%, MLP: -0.5%).
- K2 (id 536): The paper claims FAIL. Correct. No method achieves both per-layer specialization AND quality improvement.
- The surviving insight (per-sequence routing > uniform) was already established by prior experiments with stronger evidence (tiny routing heads: +19.9%).

## Macro-Scale Risks (advisory)

1. The conclusion "per-layer routing hurts because adapters are calibrated for same-adapter residual stream" may not hold for adapters trained with per-layer routing in mind (e.g., MoE-style expert training). This is a property of the training procedure, not a fundamental limitation.

2. At larger N (N=25+), the argument for single-adapter selection weakens. If a query spans multiple domains (e.g., "legal analysis of a medical malpractice case"), selecting a single adapter loses multi-domain coverage. The top-2 routing result (+13.9%) from prior work already suggests k>1 is beneficial.

3. The float32 overflow in MLP gates suggests hidden state normalization is needed before any routing. At macro scale with larger models, this issue will persist or worsen.

## Verdict

**PROCEED** (as a completed negative result)

**Justification:**

The per-layer pointer routing hypothesis is cleanly killed. The negative result is genuine and useful: methods that actually vary adapter assignment across layers (hash, MLP) perform worse than uniform 1/N, and the only method that beats uniform (learned gate) collapses to per-sequence oracle routing. This correctly prunes a design branch.

However, the following issues should be noted in FINDINGS.md:

1. The "11.9% improvement" finding is redundant with prior work (tiny routing heads: +19.9%, top-2 routing: +13.9%). It should not be promoted as a new finding.
2. The claim that "output-space composition preserves nonlinearity" is unsubstantiated. The missing scale control means the improvement could be entirely from 1/N dilution elimination, which is a simpler (and already known) explanation.
3. The machine-readable verdict in results.json ("SUPPORTED") is incorrect due to the K1/K2 OR-across-methods bug. The correct verdict per the code's own criteria should be "KILLED (K2 FAIL)" since no single method passes both K1 and K2.
4. The MLP gate result should be flagged as "failed to train" rather than treated as a valid routing comparison.

The core negative finding (per-layer mixing of jointly-trained adapters hurts) is sound, well-explained, and advances architectural understanding. The residual stream calibration argument (Section 4 of PAPER.md) is a plausible mechanistic explanation that could be tested in future work by training adapters specifically for per-layer isolation.
