# Peer Review: ternary_base_scale_d512

## Mathematical Soundness

### BitLinear formulation: CORRECT
The STE quantization (alpha = mean(|W|), round-and-clip, stop_gradient trick) is standard and correctly implemented. The implementation matches the MATH.md derivation exactly.

### Parameter count: CORRECT
76.7M total checks out: 25.7M (token embed) + 65K (pos embed) + 8 * 3.15M (layers) + 25.7M (lm_head) + norms. The embedding dominance (67% of params in embeddings + lm_head) is correctly identified in the paper as a structural issue.

### STE depth error accumulation formula: IMPRECISE but not wrong
MATH.md states epsilon_l = prod(I + delta_i), which is a first-order approximation. The actual error propagation through nonlinearities (GELU, attention softmax) is not multiplicative in this simple form. However, the paper uses this only as motivation for monitoring, not as a quantitative prediction, so this is acceptable.

### Memory budget: CORRECT
1.2GB parameter memory + 200MB activations is reasonable for the stated configuration. Well within platform constraints.

### PPL prediction (1.5-2.5x): REASONABLE but missed
The paper predicted 1.5-2.5x based on pQuant scaling trends. Actual result was 2.78x, slightly outside the upper bound. The prediction was directionally correct but the confidence interval was too tight. The paper acknowledges this honestly.

## Novelty Assessment

### Prior art
This is not a novelty experiment. It is a scale validation of the proven d=256 ternary-from-scratch mechanism (exp_ternary_base_from_scratch_mlx). The methodology is BitNet b1.58 STE training, which is well-established. The experiment's value is empirical -- testing whether the mechanism holds at larger scale with real data.

### Delta over prior work
The delta is the empirical data point: ternary STE at 77M params on real English text produces a 2.78x PPL gap. This confirms the pQuant prediction of sublinear scaling (gap widens before narrowing at 3B+) and adds a concrete data point in an under-explored regime (sub-100M ternary from scratch on real text).

## Experimental Design

### Critical flaw 1: Severe data scarcity confounds the PPL ratio

This is the most important methodological issue. The experiment trains a 76.7M parameter model on 2M tokens. This is a ratio of ~0.026 tokens per parameter. Standard language model training uses 20-300 tokens per parameter (Chinchilla scaling: ~20 tokens/param optimal). The FP32 baseline itself achieves PPL 420 -- a terrible result that indicates radical undertrain even for FP32.

The ternary model trains on 41M tokens (10K steps * 4096 tokens/step), giving ~0.53 tokens/param. Still deeply in the undertraining regime.

**Why this matters for the PPL ratio**: In the extreme data-scarcity regime, the quantization constraint is maximally punishing because the model cannot learn the weight structure that makes ternary approximation accurate. The 2.78x ratio measures "ternary penalty under extreme data starvation," not "ternary penalty for language modeling." The paper partially acknowledges this in Limitations (point 1) but does not recognize that the entire kill criterion evaluation is confounded by it.

**Counterpoint (why this is acceptable within micro constraints)**: The experiment is deliberately micro-scale. Getting 100M+ tokens would require substantially more training time. The data scarcity is a known constraint, and the paper correctly identifies "more training data" as the top recommendation. The kill criterion is still valid as a gate: at THIS scale with THIS data budget, ternary STE does not meet the bar.

### Critical flaw 2: Train-val divergence undermines PPL comparison

The paper correctly identifies the overfitting signal (train loss 2.48 < FP32's 3.17, but val PPL 2.78x worse). However, it does not draw the full implication: **the ternary model is severely overfitting while the FP32 model is not, making the val PPL comparison unfair.**

The ternary model trains for 2x the steps (10K vs 5K) and 2x the tokens (41M vs 20.5M) on the same 2M unique tokens. It sees each training token ~20 times vs FP32's ~10 times. This means:
- The ternary model is memorizing the training set (train loss well below FP32)
- The val PPL gap partially reflects overfitting, not quantization quality loss

**The fair comparison would be**: (a) train both for the same number of unique tokens (epochs), or (b) add dropout/weight decay to the ternary model to match the generalization regime, or (c) use a held-out data stream so each token is seen at most once.

The paper acknowledges the token asymmetry in Limitation 3 but dismisses it ("ternary model already overfits -- more steps would not help without more data"). This is backwards: the point is that the ternary model was GIVEN more steps, which caused more overfitting, which inflated the val PPL gap.

### Minor issue: K1 convergence metric is trivially satisfied

K1 checks whether loss drops below ln(50257) = 10.825. The ternary model hits this at step 1. This means K1 is vacuous -- any model that makes any progress at all will pass it. The convergence criterion should have been defined as "loss still decreasing at 10K steps" or "loss within X% of minimum achieved."

### Minor issue: No dropout or weight decay

The paper lists this as a limitation. It is standard in BitNet b1.58 training. The absence of regularization disproportionately affects the model with more training steps (ternary), compounding the overfitting issue above.

### Minor issue: Single random seed

Results are from one seed. The PPL ratio of 2.78x could shift meaningfully with different initialization, especially given the overfitting dynamics.

## Hypothesis Graph Consistency

The experiment is not explicitly tracked in HYPOTHESES.yml under its own node, but it advances Track A ("Own Our Ternary Base") in VISION.md. The kill criterion K2 (PPL within 2x FP32) is reasonable as a gate. The experiment correctly applied it and correctly killed itself.

The paper's conclusions align with FINDINGS.md's prior entry for the d=256 experiment, which warned about overcapacity artifacts.

## Positive Assessment

Despite the critiques above, this experiment does several things well:

1. **Honest self-kill.** The experiment set clear criteria, measured them, and killed itself when K2 failed. This is the system working as designed.

2. **Deadzone tracking is thorough.** Per-layer, per-1K-step monitoring of zero fractions with clean results (31.4%, stable, uniform across layers). This disproves the STE depth accumulation concern for 8 layers.

3. **Code quality is solid.** Function-scoped phases, proper memory management, mx.eval() at loop boundaries, correct use of MLX patterns.

4. **The overfitting finding is genuinely valuable.** The observation that ternary models memorize training data better but generalize worse at small data scales is a useful empirical contribution. It suggests that ternary training may require MORE data than FP32, not just equal data.

5. **The paper correctly identifies next steps.** Vocabulary reduction, regularization, and data scaling are all well-motivated.

## Macro-Scale Risks (advisory)

1. **Data scaling is the key unknown.** At 20+ tokens/param (Chinchilla-optimal), does the ternary PPL ratio drop below 2x? pQuant suggests it does at 3B+ but the curve at 77M-1B is uncharted.

2. **Vocabulary-to-param ratio.** The 67% embedding overhead is not just a micro problem. Any ternary model needs careful vocabulary sizing relative to core transformer capacity.

3. **Regularization interaction with STE.** Dropout + STE + ternary quantization creates three sources of noise. The interaction could be constructive (regularization helps) or destructive (too much noise prevents convergence). This needs empirical testing at macro scale.

## Verdict

**PROCEED** (the kill is valid; the experiment succeeded at its purpose)

The experiment correctly identified that ternary STE at d=512/8L on real text with 2M training tokens fails the 2x PPL bar, and killed itself. The self-kill is appropriate. The experimental methodology has confounds (data scarcity, overfitting from token asymmetry, no regularization), but these confounds are:

1. Acknowledged in the paper
2. Inherent to micro-scale constraints
3. The paper's conclusions are appropriately scoped ("the path forward is more data, regularization, and vocabulary optimization" -- not "ternary from scratch is dead")

The key finding -- ternary models overfit more aggressively than FP32 at small data scales -- is a genuine and useful empirical contribution that should inform future experiments.

**Recommended follow-ups (not blocking):**

1. A revised experiment with dropout=0.1 and weight_decay=0.01, training both models for the same number of unique-token epochs (not the same number of steps), to get a cleaner PPL ratio.
2. A vocabulary-reduced variant (4K-8K BPE) to rebalance embedding vs. transformer parameters.
3. Streaming 10-50M unique tokens to test whether the ternary PPL ratio drops below 2x with adequate data.
