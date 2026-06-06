# Peer Review: Entropy-Adaptive Router

## NotebookLM Findings

Skipped (tool authentication not configured). Review conducted via direct analysis of MATH.md, PAPER.md, code, and references.

## Mathematical Soundness

### What holds

1. **Entropy computation is correct.** The Shannon entropy formula H = -sum(p_g log p_g) is standard. The worked example in MATH.md (Section "Worked Example") computes correctly: exp(2.1) = 8.17, exp(0.5) = 1.65, exp(0.3) = 1.35, exp(-0.1) = 0.90, Z = 12.07, verified.

2. **Sigmoid parametrization of tau is sound.** tau = sigmoid(raw_tau) * H_max keeps the threshold in [0, log(G)]. The initialization via inverse-sigmoid is algebraically correct.

3. **Soft mask interpolation is differentiable.** M = (1-alpha) * M_1 + alpha * M_2 with alpha = sigmoid((H - tau) / T_temp) provides gradient flow through the k-selection decision. At T_temp = 0.1, this is sharp enough that the soft/hard gap is small.

4. **Balance loss is standard.** L_bal = G * sum(mean(p_g)^2) is the same as Switch Transformers / capsule_moe.

### What does not hold

5. **FLOP savings are phantom -- all experts are computed regardless of k.** The code at `run_composition_experiment.py:139-143` and `entropy_adaptive_router.py:157-161` runs ALL G expert groups for every token:

   ```python
   out = mx.zeros_like(x)
   for i, group in enumerate(self.groups):
       w = masked_probs[..., i:i+1]
       out = out + w * group(x)
   ```

   When alpha is near 0 (k=1 regime), the weight w for the 2nd-ranked expert is near-zero but NOT zero -- the sigmoid never reaches exactly 0. The expert forward pass still executes. The claimed "8.5% FLOP savings" in MATH.md Section "Compute Savings" is a theoretical projection that requires conditional execution (skip expert computation when w < epsilon). No such conditional execution exists in the code. The actual FLOP count is identical to fixed k=G for every token. This makes KC2 (average k reduction) an accounting metric with no real compute impact as implemented.

6. **Threshold (raw_tau) is likely frozen during calibration.** In `run_composition_experiment.py:263-306`, the calibration function calls `model.freeze()` at line 278 (after an earlier attempt at unfreezing raw_tau), then only unfreezes `layer.pool.router` (lines 280-281). The `raw_tau` parameter at `layer.pool.raw_tau` is NOT explicitly unfrozen after the second freeze. In MLX, `freeze()` marks all leaf arrays as non-trainable; only modules explicitly `unfreeze()`-d afterwards receive gradients. Since `raw_tau` is a bare `mx.array` (not inside an `nn.Module`), calling `layer.pool.router.unfreeze()` does not affect it. If `raw_tau` is frozen, it receives zero gradient from `nn.value_and_grad`, meaning the "learned threshold" per-layer variation reported in PAPER.md Table "Per-Layer Entropy Profile" would come from initialization or some other source, not from gradient-based learning during calibration. The reported tau values (0.428, 0.420, 0.361, 0.332) differ from the initial value of sigmoid(0) * log(8) = 1.04, which is puzzling. This needs investigation: either MLX handles bare array unfreezing differently than expected, or the tau values come from a different code path than what is committed.

7. **Overhead calculation underestimates.** MATH.md claims entropy overhead is "~26 MADs" at G=8. The actual overhead per token includes: entropy (8 multiplies + 8 logs + 8 multiplies + 7 adds = ~31 ops), sigmoid (1 exp + 1 div), TWO topk operations (one for top-1, one for top-2), mask interpolation (2*8 = 16 MADs), and renormalization. The topk operations are O(G log G) each. At G=8 this is minor, but the paper's claim of "5% increase in routing cost" undercounts.

## Novelty Assessment

### Prior art

1. **ReMoE (ICLR 2025)** achieves variable per-token expert count via ReLU routing. The paper correctly cites this. Delta: entropy-adaptive uses softmax + entropy threshold rather than ReLU activation magnitudes. This is a genuine architectural distinction -- entropy-based selection is an explicit information-theoretic criterion vs. ReLU's implicit magnitude-based one.

2. **Mixture-of-Depths (Raposo et al. 2024)** adapts compute at the layer level. Correctly cited as complementary.

3. **No exact prior art found** for using Shannon entropy of routing probabilities to select per-token k in MoE. The novelty claim appears valid. However, the contribution is incremental: it is a one-line decision rule on top of standard softmax routing. The mechanism is simple enough that the absence of prior art may reflect it being considered too obvious to publish rather than a blind spot in the field.

### Delta over closest work

ReMoE's ReLU routing gives variable k naturally without any threshold parameter. The entropy-adaptive approach adds a learned threshold, which provides an explicit knob for the quality/compute tradeoff but also adds a hyperparameter (sparsity_coeff) and a questionable learning procedure (see point 6 above). ReMoE's approach is arguably simpler and more principled for end-to-end training. The entropy approach may be better suited for post-hoc composition (where the router is calibrated separately), but this advantage is not demonstrated.

## Experimental Design

### What works well

1. **Composition-specific testing.** The experiment correctly identifies that k=1 is only catastrophic under composition, not single-domain. This is a useful clarifying finding.

2. **Multi-seed evaluation.** 3 seeds with mean reporting is appropriate for micro scale.

3. **Sparsity coefficient sweep.** Testing sc=0.0, 0.1, 0.3 shows the quality/sparsity tradeoff.

4. **Joint model baseline.** Including a jointly-trained reference is good practice.

### Issues

5. **The primary control is incomplete.** The experiment compares entropy-adaptive to fixed k=1 and fixed k=2, but does not compare to a random k selection baseline. If tokens were randomly assigned k=1 or k=2 with probability matching the entropy-adaptive distribution (e.g., 82% k=2, 18% k=1), would quality be the same? This would falsify the claim that entropy-based selection is meaningful vs. simple probabilistic budget allocation.

6. **KC2 threshold of avg_k < 1.8 may be too lenient.** At k=1 vs k=2, getting avg_k = 1.82 means only 18% of tokens use k=1. The FLOP savings (if conditional execution were implemented) would be 9%. Whether this justifies the architectural complexity (entropy computation, learned threshold, sparsity loss, soft masking) is questionable. The overhead of the entropy mechanism itself may approach the savings.

7. **Train/test mismatch.** Training uses soft sigmoid interpolation between masks. Inference should use hard threshold for actual compute savings. The paper acknowledges this in "What Would Kill This" but does not measure the soft-to-hard quality gap. If the model relies on the smooth interpolation during training, hard thresholding at inference could degrade quality.

8. **Per-layer analysis is n=1.** The "Per-Layer Entropy Profile" table is from seed 42 only, not averaged across seeds. This is disclosed but layer-dependent patterns from a single seed should not be treated as robust findings.

## Hypothesis Graph Consistency

The experiment maps to `exp_arithmetic_coding_router` in HYPOTHESES.yml. Kill criteria match:
- KC1 (variable-k worse than fixed k=2): correctly tested, passes at +0.01%
- KC2 (avg_k < 1.8): correctly tested, marginally fails at 1.82-1.85

The status is `partial_pass`, which is honest. The evidence list accurately reflects results.

## Macro-Scale Risks (advisory)

1. **Conditional execution is mandatory for real savings.** At G=256 with projected avg_k=6, skipping 2 experts per token saves nothing unless the forward pass is conditional. This requires either (a) sparse matrix operations, (b) expert parallelism with early termination, or (c) batching tokens by k value. None of these are trivial at scale.

2. **Entropy distribution at large G is unknown.** With 256 experts, the softmax distribution over experts may always be near-uniform (high entropy for all tokens), collapsing the mechanism to always k=8. Alternatively, with stronger specialization, entropy may always be very low, collapsing to always k=1. The useful operating regime requires a bimodal entropy distribution, which is not guaranteed.

3. **The projection of avg_k=6 at G=256 is unsupported.** The paper extrapolates from G=8 to G=256 with no theoretical justification. The entropy distribution's shape depends on the data distribution, expert specialization, and training dynamics, none of which scale linearly.

4. **Soft-to-hard gap at scale.** The sigmoid temperature of 0.1 makes training effectively "almost hard," but at scale with larger batch sizes the gradient landscape may differ.

## Verdict

**REVISE**

The mechanism is conceptually sound and the experimental design demonstrates that entropy-adaptive routing does not degrade quality under composition. However, there are implementation issues that undermine the key claims:

1. **Fix the FLOP savings claim.** Either implement conditional expert execution (skip groups with weight below epsilon) and measure actual wall-clock savings, OR honestly reframe the contribution as "identifying which tokens could use fewer experts" rather than "achieving 8.5% FLOP savings." The current code computes all experts for all tokens. The savings are zero as implemented.

2. **Verify and fix `raw_tau` unfreezing during calibration.** The code at `run_composition_experiment.py:263-306` appears to leave `raw_tau` frozen after the second `model.freeze()` call at line 278. Add explicit unfreezing: after `layer.pool.router.unfreeze()` at line 281, add code to unfreeze `raw_tau`. Then re-run the experiment. If the per-layer tau variation disappears, the "learned per-layer threshold" finding is an artifact. If it persists with corrected unfreezing, document why.

3. **Add a random-k baseline.** For each entropy-adaptive config, run a control where each token randomly gets k=1 with probability p (matched to the observed fraction) and k=2 with probability 1-p. If quality is the same, the entropy criterion adds no value over random budget allocation.

4. **Report per-layer entropy profile as 3-seed mean.** The current per-layer analysis (Table in PAPER.md) uses only seed 42. Average across all 3 seeds and report standard deviations to validate the layer-dependent pattern.

5. **Measure the soft-to-hard quality gap.** After calibration with soft sigmoid routing, evaluate with hard thresholding (k=1 if H < tau, k=2 otherwise, no interpolation). Report the quality delta. If it exceeds 1%, the mechanism cannot be deployed at inference without retraining.

None of these are fundamental mechanism failures. The core idea -- using routing entropy to identify tokens that need fewer experts under composition -- is validated by KC1 passing. The revisions address honest accounting (fixes 1, 4), correctness (fix 2), scientific rigor (fixes 3, 5).
