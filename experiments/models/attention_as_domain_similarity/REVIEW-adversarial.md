# Peer Review: attention_as_domain_similarity

## NotebookLM Findings

Skipped -- the experiment is already killed by the researcher. Deep review not warranted for a negative result that the PAPER.md documents honestly and thoroughly.

## Mathematical Soundness

**MATH.md is largely correct, with two minor errors:**

1. **Section 2.2 counting error.** The text states "5 distinct similarity levels" then lists 6 values: {0.1, 0.15, 0.2, 0.4, 0.5, 0.7}. This is cosmetic and does not affect any downstream calculation. The correlation analysis uses all 66 pairs regardless.

2. **SPSA gradient estimate (Section 5.1, Step 3).** The formula shows `g_i = (L+ - L-) / (2 * eps * Delta_i)`, which is correct for general perturbations. For Rademacher perturbations (Delta_i in {+1, -1}), dividing by Delta_i is equivalent to multiplying by Delta_i, which the code correctly implements (line 386: `grad_scalar * perturbations[key][l]`). The MATH.md does not make this simplification explicit, which could confuse readers, but the implementation is correct.

**Delta vector dimensions are correctly computed:**
- Attention: L * 4 * d * d = 2 * 4 * 64 * 64 = 32,768. Correct.
- FFN: L * (d * d_ff + d_ff * d) = 2 * (64*256 + 256*64) = 65,536. Correct.
- Full: 98,304. Correct.

**Random baseline expectations are correct:**
- E[|cos|] = sqrt(2/(pi*D)) for random unit vectors in D dimensions. Formula and values check out.

**Spearman correlation formula is standard.** The use of |cos| (absolute value) rather than raw cosine is the right choice -- domain similarity is non-negative and cosine can be negative due to random sign assignments in SPSA.

**The SPSA convergence claim is correct in principle** but practically problematic. SPSA converges in expectation, but the variance of the gradient estimate is O(1/eps^2) for each parameter, and with Rademacher perturbations all parameters share a single scalar loss difference. At 300 steps (later increased to 500 in code -- see below), the signal-to-noise ratio for any individual parameter is very low. The MATH.md acknowledges this in Section 6.

## Experimental Design

**The experiment tests the stated hypothesis correctly.** The design is sound: train domain-specific LoRA experts, compute module-specific cosine similarities, correlate with ground truth. The graduated similarity matrix is a good design choice over binary within/cross.

**Code-MATH.md discrepancy (non-critical).** MATH.md Section 5.2 says "300 steps * 2 forward passes" but the code uses `steps=500` (line 683). The PAPER.md reports results are from 3 seeds. This discrepancy does not invalidate results but should be noted. The extra 200 steps make the negative result stronger -- more training, still no signal.

**Controls are adequate for a negative result.** The experiment includes:
- 3 seeds (42, 123, 7) for variance estimation
- Module-type comparison (attention vs FFN vs full) as internal control
- K2 ablation over domain counts (6, 8, 10, 12)
- Random baseline expectations from theory

**The negative result is convincing at micro scale.** Attention rho=0.073 (p=0.66) across 3 seeds, with FFN consistently higher (rho=0.185). The variance across seeds is large (attn std=0.102), but none of the 3 seeds exceeds the 0.3 threshold. This is a clean kill on K1.

**One design weakness: shared base model across all experts.** All 12 experts share the same randomly initialized base model and the same frozen A matrices (Kaiming-initialized). Since only B matrices are trained via SPSA, and B starts at zero, the initial gradient direction is determined by the shared A matrices and the shared base weights. This means early-training delta directions are partially determined by shared initialization rather than domain content. However, this is the standard LoRA setup and reflects real practice, so it is not a flaw per se.

**The A-matrix initialization creates a subtle confound.** Each domain expert gets a fresh RNG for LoRA init (line 682: `np.random.RandomState(seed + hash(domain) % 1000)`), which means A matrices differ across domains. Since B starts at zero and the delta is A@B, the delta direction is initially determined by A (which is random and domain-independent). The domain signal must overcome this initialization noise. At 500 SPSA steps with non-converging loss, it does not. This is acknowledged honestly in the paper.

## Novelty Assessment

**This is not a novel mechanism -- it is a diagnostic experiment.** It tests whether a previously observed macro-scale pattern (attention cosine amplifies domain overlap) generalizes. The negative result at micro scale is informative precisely because it bounds the mechanism: attention-as-domain-predictor requires converged training, not just architecture.

**Prior art alignment is correct.** The experiment correctly cites:
- ffn_only_vs_all_modules (the original observation)
- orthogonality_by_domain_type (the MLP-only comparison)
- Geva et al. 2021 (FFN as key-value memories)

**No prior work was missed that would change the conclusion.** The finding that FFN captures domain structure better than attention at micro scale is consistent with Geva et al.'s thesis and with the orthogonality_by_domain_type experiment.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_attention_as_domain_similarity` has:
- Kill criteria: "attention cosine does not correlate with semantic domain similarity" and "signal vanishes with more data"
- Status: killed
- Evidence correctly documents K1 kill with specific numbers

**K1 kill is properly applied.** Mean rho=0.073 < 0.3, p=0.66 > 0.05. Both conditions fail simultaneously.

**K2 technically passes** (rho at N=12 is 0.217, rho at N=6 is 0.266, so only a ~18% drop, below the 30% threshold). This is correctly noted in the PAPER.md. The signal is stable but weak.

**The overall KILL verdict is correct** -- K1 alone is sufficient, and K2 passing does not resurrect a correlation that is not statistically significant.

## SPSA as Confound -- Does It Matter?

The most important question for this experiment is whether SPSA training (rather than analytical backprop) is responsible for the null result. Two arguments:

**Against SPSA as confound:** orthogonality_by_domain_type used analytical backprop on an MLP-only model and found a 7.84x within/cross ratio for FFN deltas. This experiment finds only 1.28x for FFN and 0.98x for attention using SPSA on a transformer. If SPSA were the only problem, both modules would be equally affected. The fact that FFN still shows a directional signal (rho=0.185, ratio=1.28x) while attention does not (rho=0.073, ratio=0.98x) suggests a real module-specific difference, not just training noise.

**For SPSA as confound:** SPSA adds noise equally to all parameters, but attention parameters interact through softmax (highly nonlinear), while FFN parameters interact through GELU (mildly nonlinear). SPSA gradient estimates may be less accurate for attention because the loss surface is less smooth in the attention parameter subspace. This would selectively degrade attention signal more than FFN signal, creating a false differential.

**Assessment:** The SPSA confound is real but does not invalidate the kill. Even if SPSA selectively harms attention, the paper correctly frames this as a micro-scale limitation and recommends macro retest with real backprop. The kill is specific to the micro regime, which the paper states explicitly.

## Macro-Scale Risks (advisory)

1. **The hypothesis may resurrect at macro.** The original math-medical observation (attn cos=0.85) was on converged Qwen2.5-7B adapters. This micro experiment cannot replicate that regime. A definitive test requires 10+ converged adapters at d >= 4096.

2. **If attention cosine does predict similarity at macro, it has routing implications.** Attention-module cosine could serve as a zero-shot domain similarity metric for routing or collision prediction, which would be valuable for the Compose phase.

3. **The FFN > attention finding at micro is consistent with prior work** (orthogonality_by_domain_type, Geva et al.). At macro scale, attention may close the gap or overtake FFN, but the project should not assume this.

## Verdict

**PROCEED** (as a completed, killed experiment)

The experiment is well-designed, honestly reported, and correctly killed. The negative result is informative: attention cosine as a domain similarity predictor is not a universal structural property but (at best) a convergence-dependent phenomenon. The kill criteria are applied correctly. The PAPER.md thoroughly documents limitations and identifies clear conditions for macro retest.

No revisions needed. The experiment has fulfilled its purpose in the hypothesis graph.
