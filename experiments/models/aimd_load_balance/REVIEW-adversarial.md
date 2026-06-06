# Peer Review: AIMD Expert Load Balancing

## NotebookLM Findings

Skipped -- the experiment is a well-documented negative result with clear root cause analysis. The materials are self-contained enough for direct review.

## Mathematical Soundness

**MATH.md derivations are correct but the Chiu-Jain analogy is weaker than presented.**

1. The AIMD update rule (lines 47-52 of MATH.md) is correctly stated and the worked example checks out: excess = 0.15, alpha * (excess/target) = 0.05 * 0.6 = 0.03, so b_0 = 0.5 * 0 - 0.03 = -0.03. Correct.

2. The auxiliary loss formulation L_bal = alpha * G * sum(f_i * p_bar_i) is the standard Switch Transformer formulation. Correct.

3. The convergence analysis (lines 84-106) correctly identifies that the Chiu-Jain theorem requires monotonicity of load in bias, and that this assumption is violated in neural routing. The paper is honest about this. Good.

4. **Implementation diverges from MATH.md.** MATH.md specifies alpha = 0.05 but the class default in `aimd_load_balance.py` line 69 is alpha = 0.01. The run_experiment.py overrides back to alpha = 0.05, so the experiment used the MATH.md value. Not a bug, but the class default is inconsistent -- minor issue.

5. **The Chiu-Jain analogy has a deeper problem the paper partially identifies but underweights.** In TCP, each sender independently controls its own window. In MoE, the softmax creates a zero-sum constraint: increasing one expert's logit necessarily decreases others' probabilities. The AIMD updates treat each expert independently, but the softmax couples them. This means the "additive increase" on an underloaded expert implicitly acts as a "decrease" on all other experts. The asymmetry that makes TCP work (gentle increase, aggressive decrease) is partially neutralized by the softmax coupling. The paper mentions this indirectly in the root cause analysis but does not formalize it.

6. **Aux loss baseline has a subtle implementation issue.** In `AuxLossCapsulePool.balance_loss()` (line 220), the code computes `mx.sum(mean_probs * mean_probs)` -- this is sum(p_i^2), which is the Herfindahl index of routing probabilities. The Switch Transformer loss is sum(f_i * p_bar_i), where f_i is the dispatch fraction (from masked routing) and p_bar_i is the mean probability. Using mean_probs as a proxy for both f_i and p_bar_i (the code comment on line 219 acknowledges this) makes the balance loss a squared term rather than a cross-term. This is a weaker balance signal than the true Switch loss, which means the aux loss baseline is actually handicapped. Despite this, aux loss still wins decisively -- which makes the negative result for AIMD even more convincing.

## Novelty Assessment

**Low novelty, but the negative result has value.**

- DeepSeek-V3 already implements bias-based routing without auxiliary loss. The AIMD variant (multiplicative decrease instead of symmetric additive) is a minor twist.
- The Chiu-Jain connection is a nice framing but the analogy breaks down for exactly the reasons the paper identifies. The TCP literature does not apply cleanly because the softmax introduces coupling that TCP senders do not have.
- The paper correctly notes DeepSeek-V3 uses 256 experts where this uses 4. The G-dependence argument is reasonable.
- **Prior art check**: The `references/deepseek-v3/` folder documents the bias approach. The researcher correctly positioned this as a variant of the DeepSeek-V3 technique with AIMD asymmetry. No reinvention problem.

**The real value is the negative result**: feedback-based bias control conflicts with gradient-based router optimization at low G. This is a useful finding that explains why auxiliary losses persist as the standard approach.

## Experimental Design

**Adequate for the hypothesis, with two notable issues.**

1. **Three-way comparison is well-designed.** AIMD vs aux loss vs no-balance gives proper baselines and control. Three seeds provide minimal statistical coverage. The experiment tests exactly what it claims.

2. **The aux loss baseline is slightly weakened** (see point 6 above about using mean_probs^2 instead of f_i * p_bar_i). This actually strengthens the negative finding for AIMD, since AIMD lost to a handicapped baseline. But if someone wanted to use these results to calibrate aux loss performance, the numbers would be pessimistic.

3. **Load imbalance metric is coarse.** max(f_i) - min(f_i) with G=4 is dominated by the most and least loaded experts. The coefficient of variation (std/mean) or entropy of load distribution would be more informative. With G=4, a single hot expert can dominate the metric. This does not invalidate the conclusion but makes the 2.7x ratio less interpretable.

4. **AIMD bias updates happen inside the forward pass** (line 121 of `aimd_load_balance.py`), which means the bias is updated using the same batch it was computed on. In TCP, AIMD updates are based on feedback from previous round-trips, not the current one. This is a design choice that creates a one-step lag mismatch -- the bias used for routing at step t is updated based on load at step t, then applied at step t+1. The MATH.md correctly describes this as b(t+1) = F(b(t), f(t)), so the notation is consistent, but it means the load fractions used for the AIMD decision are always one step stale relative to the router weights that just got a gradient update.

5. **All groups run on every token** (lines 116-119). The routing weights determine the soft mixing, but every group's forward pass executes regardless of whether it has near-zero weight. This is standard for micro experiments but means load "balance" is about weight balance, not compute balance. The paper does not claim compute savings, so this is fine.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_tcp_congestion_load_balance` has two kill criteria:
- KC1: "AIMD balancing worse than auxiliary load-balancing loss" -- **triggered** (AIMD +0.41% worse quality, 2.7x worse load balance)
- KC2: "convergence to fair allocation takes >2x training steps vs aux loss" -- **cannot evaluate** (neither converged to 0.15 threshold within 500 steps)

The evidence entry correctly records the kill with appropriate nuance about the G-dependence. The node status should be updated to "killed" if not already. The kill criteria match what was tested.

## Macro-Scale Risks (advisory)

1. **G-dependence is the key macro question.** The paper's prediction that feedback-based bias works at G=256 but not G=4 is plausible and consistent with DeepSeek-V3's results. A macro experiment should test AIMD at G=16, G=64, G=256 to find the crossover point.

2. **Interaction with learning rate schedules.** The adversarial dynamic between AIMD and gradient descent may be ameliorated by lower learning rates (late in training) or by freezing the router and only using AIMD. Neither was tested.

3. **The symmetric (AI/AD) variant should be tested before AIMD at macro.** DeepSeek-V3 uses symmetric additive updates, not AIMD. The multiplicative decrease may add unnecessary instability. The paper correctly notes this.

## Verdict

**PROCEED** (as a closed negative result)

This is a well-executed negative result. The experiment tested a clear hypothesis with appropriate controls, found it fails at micro scale, provided a sound root cause analysis (feedback vs gradient conflict, softmax coupling, low-G coarseness), and correctly scoped the limitations. The kill criteria were honestly evaluated.

Minor issues that do not warrant revision:
- The aux loss implementation uses mean_probs^2 instead of the true Switch loss (f_i * p_bar_i), but this handicaps the baseline and makes the negative result for AIMD more robust, not less.
- The class default for alpha (0.01) differs from MATH.md (0.05), but the experiment used the correct value via run_experiment.py.
- The softmax coupling argument could be formalized more precisely, but the intuitive explanation in the root cause analysis is sufficient for a micro experiment.

No further work needed on this experiment. The finding should be recorded in FINDINGS.md. If macro-scale routing is pursued, test DeepSeek-V3's symmetric variant (AI/AD) at larger G before revisiting AIMD.
