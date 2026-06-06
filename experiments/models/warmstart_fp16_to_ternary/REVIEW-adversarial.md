# Peer Review: Warm-Start FP16 to Ternary QAT (Post-Revision)

## Review Context

This is a post-revision review. The previous review (REVISE verdict) identified 4 issues:

1. **Weight decay confound** -- cold-start used wd=0.01, warm-start used wd=0.0 in ternary phase
2. **Recovery metric artifact** -- "recovered in ~51 steps" misrepresented as a finding when it was the first measurement point
3. **Docstring mismatch** -- docstring said 5000 steps / 500+1000 switch, code used 3000 / 300+600
4. **FP32 baseline note** -- should acknowledge Extra RMSNorm architectural difference in Limitations

## Fix Verification

### Fix 1: Weight Decay Confound -- FIXED

The researcher added `phase_cold_start_ternary_no_wd()` (run_experiment.py lines 499-567) with `weight_decay=0.0` (line 521). Results are recorded in `results.json` under `cold_start_ternary_no_wd` and `weight_decay_ablation`. The ablation is thoroughly discussed in both MATH.md (lines 188-219) and PAPER.md (Finding 1, line 62).

Key numbers:
- Cold-start wd=0.01: PPL 416.80 (1.211x)
- Cold-start wd=0.0: PPL 411.16 (1.195x)
- Warm-start 10%: PPL 360.06 (1.046x)

Weight decay removal accounts for 1.4% of the improvement. Warm-start still improves 12.4% over the no-wd control. The confound is real but minor, and now properly quantified. This is exactly the right fix.

### Fix 2: Recovery Metric Language -- FIXED

The code now prints "recovered within at most {step} steps (first measurement point; ...)" (line 681-682). PAPER.md uses "within at most 51 steps (first measurement point; actual recovery may be faster)" (lines 48, 68). MATH.md uses "recovered within at most 51 QAT steps -- first measurement point" (lines 171, 177). The language is now accurate throughout.

### Fix 3: HYPOTHESES.yml Node -- NOT FIXED

No node for `warmstart_fp16_to_ternary` exists in HYPOTHESES.yml. The kill criteria IDs 266 and 267 referenced in the script docstring are not traceable to any hypothesis graph node. A grep for "warmstart_fp16" in HYPOTHESES.yml returns zero matches.

This was listed as required fix 3 in the previous review. It remains unaddressed.

### Fix 4: Docstring Mismatch -- FIXED

The script docstring (lines 15-19) now correctly lists 5 conditions with 3000 steps and the actual switch points (300/600 steps).

### Non-Blocking Observation: FP32 Baseline Note -- FIXED

PAPER.md Limitation 1 (line 88) explicitly acknowledges the architectural difference and explains why the FP32 baseline is appropriate as a reference point rather than a strict ablation.

## Mathematical Soundness

No changes from the previous review. The math remains sound:

- STE formulation is correct
- Quantization error bounds are directionally correct
- LR schedule math is standard
- Optimizer state transfer rationale is plausible (though the experiment still does not ablate optimizer state transfer vs weight initialization separately -- this is acceptable for a micro experiment)

The weight decay ablation math is correctly computed and the interpretation is conservative and fair.

## Novelty Assessment

Unchanged. Low novelty, correctly acknowledged. This validates known production recipes (BitNet b1.58, Falcon-Edge, Continual QAT) at micro scale. No reinvention.

## Experimental Design

With the weight decay ablation added, the experimental design is now adequate for its stated purpose. The five conditions form a coherent set:

1. FP32 baseline (reference)
2. Cold-start ternary wd=0.01 (original control)
3. Cold-start ternary wd=0.0 (ablation control for weight decay confound)
4. Warm-start 10% (main treatment)
5. Warm-start 20% (switch-fraction comparison)

Remaining known limitation: no ablation of optimizer state transfer vs weight initialization. The warm-start advantage (12.4% over no-wd cold-start) could come from (a) weight initialization from FP16, (b) Adam state transfer, or (c) the combination. Separating these would require a sixth condition (warm-start with optimizer state reset). This is noted as acceptable for micro scale -- the project cares about the recipe as a whole.

Single-seed caveat remains. The 10% vs 20% finding (360 vs 382 PPL) is within plausible seed variance and should not be treated as definitive. The paper acknowledges this.

## Macro-Scale Risks (advisory)

Unchanged from the previous review:

1. Optimizer state transfer stability at d=4096 with 32+ layers
2. The 10% switch fraction may not generalize to billions-of-tokens training
3. Extra RMSNorm alone accounts for most of the improvement (2.78x to 1.211x); warm-start adds a smaller increment (1.211x to 1.046x)
4. All conditions are far from convergence (PPL > 300); the warm-start advantage could diminish with longer training

## Verdict

**PROCEED**

Three of four required fixes from the previous review have been applied correctly. The weight decay confound (the most important issue) is resolved with a clean ablation. The recovery metric language is accurate. The docstring is corrected. The FP32 baseline architectural difference is acknowledged in Limitations.

The missing HYPOTHESES.yml node (fix 3) is a traceability issue, not a scientific one. It does not affect the validity of the experimental results or the soundness of the conclusions. It should be added as part of routine housekeeping but does not warrant blocking the experiment.

The core finding stands: warm-start ternary QAT achieves 1.046x FP32 PPL at d=512, a 12.4% improvement over cold-start ternary (controlling for weight decay). The mechanism is well-implemented, the math is sound, the confounds are properly addressed, and the results clearly pass all kill criteria (K1, K2) and success criteria (S1).

### Housekeeping (non-blocking)

1. Add this experiment to HYPOTHESES.yml with kill criteria K1 (id=266), K2 (id=267), and success criterion S1 (id=31).
2. Consider a future micro experiment with optimizer state reset to separate initialization vs state transfer contributions.
