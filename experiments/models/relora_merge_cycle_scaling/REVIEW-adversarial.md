# Peer Review: ReLoRA Merge Cycle Scaling

## Mathematical Soundness

**What holds:**

1. The rank accumulation bound (rank(W_final - W_0) <= K*r) is correct. At K>=8 with r=8, d=64, the accumulated perturbation is full-rank. This is properly noted.

2. The waste_fraction analysis (10/T_c = 10K/T) is a reasonable first-order model for Adam momentum warmup cost. At K=200, T_c=10, this predicts severe inefficiency, consistent with the observed base_ratio=1.187.

3. The decomposition of loss_ratio into base_ratio and composition_penalty is sound and is the strongest analytical contribution. The finding that composition penalty is stable at ~1-2% across K while base_ratio drives loss_ratio growth is well-supported by the data.

4. The log-linear trend fitting is correctly implemented (power-law fit in log-log space).

**What does not hold or is questionable:**

1. **The K1 kill criterion threshold is too generous.** MATH.md derives the 5x threshold by noting 200/5 = 40x more cycles and allowing "sub-linear scaling" at 2.8x growth over the K=5 baseline of 1.77x. But the K=5 result in THIS experiment is 2.42x (not 1.77x from the original experiment). Using the experiment's own K=5 baseline: 2.42 * 2.8 = 6.8, which would make 5x a TIGHTER threshold than intended. The threshold derivation references a different experiment's baseline, creating an inconsistency.

2. **The "no systematic trend" claim for cos_ratio is statistically unsupported.** With n=5 data points (K values) and variance that exceeds the signal range, claiming "no trend" is as unwarranted as claiming "strong trend." The correct statement is "insufficient power to detect a trend." The paper acknowledges this but then proceeds to interpret the absence of a detected trend as evidence of no trend (classic absence-of-evidence-as-evidence-of-absence fallacy).

3. **The log-linear slope of 0.055 is NOT "effectively zero."** The paper claims this on line 71 of PAPER.md. In log-log space, a slope of 0.055 means cos_ratio ~ K^0.055. Over the range K=5 to K=200 (a 40x increase), this predicts a factor of 40^0.055 = 1.23x growth. Given that the fit is dominated by the K=25 outlier (11.32x), the slope is essentially meaningless -- but calling it "effectively zero" implies the fit is informative when it is not.

## Novelty Assessment

This experiment is a straightforward parameter sweep of an existing mechanism (ReLoRA composition test from `relora_composition_test`). It is not novel research -- it is scaling validation, which is exactly what it claims to be. No prior art issues.

The code cleanly reuses the parent experiment's infrastructure (`train_relora`, `train_conventional`, `train_lora_expert`, `compute_pairwise_cosine`). This is good engineering practice.

## Experimental Design

**Critical issue: the code's own verdict disagrees with the paper.**

The code (line 308-311) applies this logic:
- KILLED if K1 or K2 exceeds threshold
- SURVIVES if cos_ratio < 3.0 AND loss_ratio < 1.30
- Otherwise INCONCLUSIVE

The code outputs `"verdict": "INCONCLUSIVE"` in results.json (cos_ratio=4.58 > 3.0). But PAPER.md declares "VERDICT: SURVIVES (both kill criteria pass)." The paper overrides the code's own assessment without acknowledging the discrepancy. The code's verdict logic is more conservative and arguably more appropriate given the data.

**Design confound: K and T_c are perfectly anti-correlated.**

MATH.md acknowledges this (Section 4.1, limitation 6 in PAPER.md) but the implications are underexplored. The key finding -- "composition penalty stable at 1-2%" -- could mean either:
(a) Merge cycles do not affect composition geometry (the claimed interpretation), OR
(b) Shorter cycles produce weaker per-cycle perturbations that individually have less geometric impact, offsetting any cumulative effect.

Interpretation (b) is equally consistent with the data. At K=200 with T_c=10, each cycle contributes a tiny delta (most steps wasted on warmup), so each merge barely changes the base. The TOTAL perturbation may be similar across K values, not because merges are harmless, but because high-K merges are individually impotent.

To distinguish (a) from (b), one would need a fixed-T_c design where total training budget grows with K. The paper acknowledges this but dismisses it because "fixed budget matches production usage." This is a practical argument, not a mechanistic one. The mechanistic claim ("merge mechanism is lossless") is not fully supported.

**FFN-only LoRA is a known-killed configuration.**

The experiment uses FFN-only LoRA (M=2 modules per layer: fc1, fc2). But the project has decisively killed FFN-only at macro scale (PPL +66.7%, ortho 424% worse). Production SOLE uses all-modules LoRA (q/k/v/o/gate/up/down). This is acknowledged in Limitation 4 but deserves more weight: the cos_ratio behavior could differ substantially with attention-module LoRA, where attention amplifies domain overlap (cos=0.85 vs FFN cos=0.59 for math-medical).

**Two seeds is marginal but acceptable for a micro sweep.**

The paper is honest about this. The cos_ratio standard deviation at K=25 (8.2) exceeds the mean (11.3), meaning the measurement is dominated by noise. The loss_ratio measurements are much more stable (std 0.001-0.020), making the loss_ratio trend the only reliable finding.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_relora_merge_cycle_scaling` has status "active" with kill criteria matching the experiment. The evidence claim accurately summarizes the results. However, the status should be updated to reflect the experiment's completion -- either "supported" or "inconclusive."

The node has `blocks: []`, meaning nothing depends on this result. This is appropriate -- the result is informational, not gate-keeping.

## Macro-Scale Risks (advisory)

1. **All-modules LoRA may show different K-scaling.** Attention modules create correlated gradients across domains. Whether K merge cycles amplify this correlation is untested.

2. **The composition penalty decomposition is the key result to replicate.** If the finding that "composition penalty is stable at 1-2% while base quality degrades" holds at d=3584, it is a strong architectural result. If composition penalty grows at macro scale (where cosine baselines are orders of magnitude lower), the safety margin may be tighter than micro suggests.

3. **Production T_c >> 10.** The extreme stress test (T_c=10) is not representative of production. At T_c=1000-5000, the waste_fraction drops to 0.2-1%, and base_ratio should be much closer to 1.0. The composition penalty finding should transfer cleanly.

## Verdict

**PROCEED** (with caveats noted below)

The experiment achieves its primary goal: demonstrating that ReLoRA merge cycles up to K=200 do not catastrophically degrade composition quality. The composition penalty decomposition (stable 1-2% across K) is the strongest finding and is well-supported by the data.

The cos_ratio metric is too noisy at 2 seeds to draw any conclusions (positive or negative). The loss_ratio trend is clean and informative. The code's INCONCLUSIVE verdict is technically more honest than the paper's SURVIVES, but the underlying data supports the directional claim that K-scaling is safe.

Specific caveats to carry forward:

1. The paper should acknowledge that the code's own verdict was INCONCLUSIVE, not override it silently. The paper's re-interpretation as SURVIVES should be explicitly justified (e.g., "the kill criteria -- the primary decision mechanism -- both pass; the code's stricter SURVIVES threshold of cos_ratio < 3.0 was not a pre-registered criterion").

2. The mechanistic claim "merge mechanism is lossless" (PAPER.md line 117) is too strong. The evidence shows composition penalty is small, not zero, and the confound between K and T_c prevents isolating the merge mechanism from training efficiency.

3. The FFN-only limitation should be flagged as higher-risk given the macro kill of FFN-only. Any macro replication must use all-modules LoRA.

4. Status in HYPOTHESES.yml should be updated from "active" to "supported" with the noted caveats.
