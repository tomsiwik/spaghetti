# Peer Review: gap_practical_regime

## NotebookLM Findings

Skipped -- the experiment is a self-kill with clean data and the analysis is straightforward enough to verify manually. The math involves basic statistics (Cohen's d, r-squared, SNR), not novel derivations requiring deep review.

## Mathematical Soundness

**Cohen's d computation: correct.** The pooled standard deviation formula matches the standard definition. The implementation in `cohens_d()` (lines 71-82 of `test_gap_practical_regime.py`) is correct.

**SNR definition: reasonable but nonstandard.** The paper defines SNR = range(mean(Q)) / mean(std(Q)) across cosine levels. This is a defensible choice but note that range is the most outlier-sensitive summary statistic. Using std(mean(Q)) / mean(std(Q)) (i.e., between-group std vs within-group std) would be more robust. However, the F-ratio (between_var / within_var = 0.01) is also reported and tells the same story even more damning -- so this is not materially problematic.

**r-squared computation: inherited from parent, assumed correct.** The `compute_r_squared` function is imported from `gap_as_signal`. This is a simple Pearson correlation; no concerns.

**Projection method: correct.** Gram-Schmidt decomposition to construct a vector at a target cosine while preserving norm is mathematically sound. The parallel component is target_cos * ||b|| * a_hat and the perpendicular component fills the remaining norm via Pythagorean theorem.

**One hidden assumption worth noting:** The 0.5pp threshold in KC1 is arbitrary. The paper acknowledges this implicitly by also reporting Cohen's d = 0.24 (small). The threshold could have been set at 0.3pp (in which case the experiment would pass) or 1.0pp. The choice of 0.5pp is stated in MATH.md before seeing results, which is correct experimental practice. However, the actual measured value (0.47pp) is *just* below threshold. At d=0.24, you would need approximately 270 samples per group (not 5) for a two-sided t-test to reach significance at alpha=0.05. The paper's Assumption 3 acknowledges this -- the experiment is designed so that failure to detect = the effect is too small to matter. This is a valid framing.

**The joint_val_loss baseline concern.** On line 315-316 of the test file, the experiment passes `joint_on_joint` (joint model evaluated on joint_val) as the `joint_val_loss` parameter to `run_single_trial`. But on lines 244-247, `joint_val_loss` is computed as the average of domain-specific losses. These are two different numbers. Looking at the call: `run_single_trial(... joint_on_joint, V, seed=seed)` where the parameter name is `joint_val_loss`. Inside `run_single_trial`, this value is used for calibration convergence (line 161: `calibrate_router_tracked(..., joint_val_loss, ...)`). Meanwhile, the `vs_joint_pct` metric (line 192) uses `joint_val_loss` which here is `joint_on_joint`. This is consistent within the experiment (same baseline across all cosine levels and seeds), so it does not invalidate the kill criteria which are about *relative* differences between cosine levels. However, it means the absolute "vs joint" percentages in PAPER.md Table 1 use the joint-on-joint baseline, not the domain-averaged baseline. This is a minor documentation issue, not a flaw.

## Novelty Assessment

**This is not a novelty experiment.** It is a targeted falsification of a specific claim from the parent experiment (gap_as_signal). The adversarial review of gap_as_signal correctly identified the leverage effect, and this experiment directly tests it. No prior art search is needed -- this is internal validation work.

**Relationship to gap_causal_mechanism:** The sibling experiment (also dated 2026-03-07) found that the gap is a *symptom*, not a *cause* -- the real mechanism is expert discriminability driving router gradients. This is consistent with the practical-regime kill: in the low-cosine regime, discriminability is uniformly high (all experts produce different outputs), so neither the gap nor discriminability provides ranking information. The two experiments converge on the same conclusion from different angles.

## Experimental Design

**Strengths:**

1. Kill criteria were stated before seeing results (in MATH.md). The 0.5pp threshold is arbitrary but pre-registered. Good practice.

2. 5 seeds (up from 3 in parent) is appropriate given the expected small effect size. The paper honestly acknowledges this is still underpowered for detecting very small effects.

3. Fine-grained cosine sweep (0.05 increments) provides 7 data points in the practical regime vs 4 in the parent. This is the right design for the question.

4. Anchor points (0.50, 0.90) enable direct comparison with the parent experiment's full-range results. Smart design choice.

**Weaknesses:**

1. **The experiment tests the wrong question for the project's needs.** Given that VISION.md now states "orthogonality is free" and natural cosine is 0.0002, the practical question is not "does the gap discriminate between cos=0.0 and cos=0.3" but rather "is there any quality difference between cos=0.0000 and cos=0.0010?" The experiment's practical regime [0.0, 0.3] is itself 1000x wider than the actual operating regime of real LoRA adapters. This doesn't invalidate the result -- it actually *strengthens* the kill conclusion -- but it means the experiment is conservatively generous to the hypothesis. If there is no signal in [0.0, 0.3], there is certainly no signal in [0.0000, 0.0010].

2. **Projection creates synthetic experts, not natural ones.** This is acknowledged in both MATH.md and PAPER.md. A projected expert at cos=0.15 has the same norm as the original but a different orientation in weight space. A naturally-trained expert at cos=0.15 (from overlapping training data) would have different internal structure. This limitation is inherited from the parent experiment and is acceptable at micro scale.

3. **No multiple-testing correction.** The experiment runs 9 cosine levels x 5 seeds = 45 trials and then does pairwise comparisons and correlations. However, the kill criteria are pre-specified (compare cos=0.0 vs cos=0.3 only), so this is not a fishing expedition. The additional analyses (monotonicity, correlation) are descriptive, not hypothesis tests. Acceptable.

4. **The 5/6 monotonicity claim is fragile.** With means that differ by <0.5pp and per-seed std of ~2pp, the ordering of means is heavily influenced by sampling noise. Bootstrap the mean ordering 1000 times and the monotonicity fraction would likely be much lower. The paper correctly frames this as "suggests a real but extremely weak underlying signal" rather than claiming it as evidence. Fine.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry matches the experiment exactly:
- Kill criteria: "quality difference between cos=0.0 and cos=0.3 is < 0.5pp" and "gap magnitude variation in [0.0, 0.3] is within noise" -- both tested and both fail.
- Status: killed -- correct given both KC fail.
- The evidence lines accurately summarize the results.
- The experiment correctly does NOT change the parent (exp_gap_as_signal) status, which remains PROVEN for the full-range claim.

**Integration with exp_gap_causal_mechanism:** These two experiments together paint a coherent picture. The causal mechanism experiment shows the real driver is discriminability (cos -> discriminability -> gradient -> quality). The practical-regime experiment shows this chain provides no ranking within [0.0, 0.3] because discriminability is uniformly maximal in that regime. The two results are consistent and mutually reinforcing.

## Macro-Scale Risks (advisory)

1. **Already confirmed at macro.** VISION.md notes that macro gap-calibration r-squared at d=896 is 0.22 (all cosines approximately 0). This is the macro version of exactly the same finding: when everything is at cos ~ 0, there is no variance to correlate. The practical-regime kill is already validated at macro.

2. **The "binary classifier" framing may need refinement.** At macro, the question shifts from "does the gap predict quality within low cosine?" to "what cosine threshold triggers the pathological regime?" The micro experiment places this boundary somewhere around cos=0.5 (where quality starts degrading sharply). At macro with d=896, achieving cos=0.5 would require extreme overlap -- essentially training two adapters on the same data. The safety-check utility is real but the threshold needs macro calibration.

## Verdict

**PROCEED**

The experiment is well-designed, honestly reported, and reaches the correct conclusion. Both kill criteria fail cleanly (not borderline -- the SNR of 0.33 is definitively below 1.0, and the F-ratio of 0.01 is definitive). The kill is well-earned and the implications for the project are correctly stated: no gap measurement is needed in the contribution protocol because orthogonality is guaranteed by dimensionality.

Minor documentation fix (non-blocking): The PAPER.md should note which baseline the "vs joint" percentages use (joint_on_joint vs domain-averaged), since the code passes `joint_on_joint` to the trial function. This does not affect any kill criterion since all comparisons are relative.

The experiment advances the project by eliminating unnecessary complexity from the contribution protocol (no gap measurement needed) and by correctly scoping the gap-as-signal claim to a binary safety check for pathological cases.
