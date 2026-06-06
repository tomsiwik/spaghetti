# Peer Review: m2p_qwen06b_gsm8k_v3 (RE-REVIEW)

## Re-Review Context

First review issued REVISE with 5 blocking fixes and 2 advisory fixes.
This re-review verifies all fixes were properly applied.

## Fix Verification

| # | Fix Required | Status | Evidence |
|---|-------------|--------|----------|
| 1 | quality_ratio formula wrong (showed M2P_acc/SFT_acc=83.3%) | FIXED | PAPER.md line 54 now shows correct formula: (M2P-base)/(SFT-base) = (0.25-0.20)/(0.26-0.20) = 83.3%. All 6 occurrences of quality_ratio in PAPER.md use the consistent improvement-based formula. |
| 2 | K914 prediction failure unacknowledged | FIXED | MATH.md line 216 explicitly marks prediction as wrong. MATH.md Self-Test #3 (line 375) says "WRONG PREDICTION." PAPER.md lines 75-87 devote a full section to explaining the failure: output_scale=0.032 not modeled, K914 "trivially satisfied from step 0." |
| 3 | K915 no statistical uncertainty | FIXED | PAPER.md lines 98-101 report binomial 95% CIs: M2P [19.0%, 31.6%], SFT [19.9%, 32.5%]. Notes overlap, declares difference "not statistically significant," and downgrades finding to "supported" (not conclusive). |
| 4 | M2P parameter overhead unacknowledged | FIXED | PAPER.md lines 137-140 state: "M2P (357M params, 1.4GB) is nearly as large as the base model... The hypernetwork overhead is ~4.6x... This undermines the 'huge-model quality at small-model cost' vision unless future work reduces M2P size." |
| 5 | Conflicting ratio definitions (96% vs 83.3%) | FIXED | No mention of "96%" anywhere in PAPER.md. The only ratio definition used is the improvement-based quality_ratio. |
| 6 (advisory) | Theorem 5 proof gap: product of non-zero Jacobians | FIXED | MATH.md lines 186-193 now include the analytic function argument: composed map is analytic, not identically zero, so vanishes only on measure-zero set (identity theorem). |
| 7 (advisory) | SHINE reference unverifiable | FIXED | MATH.md lines 97-101 add explicit disclaimer: "arXiv:2602.06358 is a February 2026 paper that cannot be independently verified at review time... If this reference does not exist, the hyperparameter grounding reduces to empirical choices; the functional autodiff invariant (Theorem 4) remains valid independently." |

All 5 blocking fixes and 2 advisory fixes verified as properly applied.

## Experiment Type
Verification (Type 1)

## Hack Detector
- Fix count: 1 core mechanism change (functional forward replaces mutation). Hyperparameter changes (d_M2P, output_scale, lr, warmup) are grounded in SHINE. Not hack-stacking.
- Is MATH.md a proof or a description? Theorem 4 is a genuine proof (functional autodiff semantics). Theorem 5 is a proof with a weak but valid guarantee (non-zero gradient at init with probability 1). Theorems 1-3 are citations. Theorem 6 is a retrospective confirmation.
- Metric used as evidence: grad_norm (direct verification of Theorem 5), quality_ratio (behavioral, improvement-based), NTP loss (convergence).
- Kill criteria source: K913 derived from Theorem 5 (proof). K914 derived from proof prediction (acknowledged as wrong). K915 threshold from SHINE benchmark.

## Self-Test Audit

1. **One-sentence impossibility property:** PASS. "B as tensor argument makes zero gradients impossible via MLX graph tracing." Single property, not a list.
2. **Cited theorems -- real, conditions apply?** Ha et al. (real, correct application), Hu et al. (real, correct application), Finding #375 (internal, consistent with v2 data). SHINE (arXiv:2602.06358): cannot verify independently; MATH.md acknowledges this with explicit disclaimer. PASS with caveat.
3. **Predicted numbers -- specific and falsifiable?** Four predictions: (a) grad_norm in [0.1, 100] -- wide but falsifiable, (b) loss start ~11.93 -- explicitly marked as WRONG, (c) loss < 2.0 in 200 steps -- trivially satisfied, (d) quality_ratio 70-90% -- measured 83.3%. Honest accounting of wrong predictions. PASS.
4. **Falsification condition:** grad_norm=0 at step 1 falsifies Theorem 5. Correctly targets the proof. PASS.
5. **Hyperparameter count:** 4 (d_M2P=1024, output_scale=0.032, lr=5e-5, warmup=100). All grounded in SHINE or standard practice. PASS.
6. **Hack check:** Clean replacement, not stacking fixes. PASS.

## Mathematical Soundness

**Theorem 4 (Functional Autodiff Invariant):** Sound. This is a statement about MLX/JAX-style functional graph tracing semantics. Python attribute assignment is invisible to the graph tracer. The "proof" is definitional (it follows from how MLX works by design), but it is correct and the formalization is useful.

**Theorem 5 (Gradient Flow):** Sound after the fix. The proof now properly addresses the Jacobian composition gap using the analytic function identity theorem. The argument chain:
1. Each layer function (GELU, linear) is analytic -- correct.
2. Composition of analytic functions is analytic -- correct.
3. The map theta -> dL/dtheta is analytic -- follows from 1-2.
4. This map is not identically zero -- justified by the existence of specific theta values where the gradient is non-trivial (random B produces non-zero delta, x @ A_q generically non-zero).
5. An analytic function not identically zero vanishes on a measure-zero set -- identity theorem. Correct.

One remaining weakness: "produces non-trivial gradients by construction" (line 192) is less formal than ideal. A cleaner statement would exhibit a specific theta value where the gradient is provably non-zero. But this is a minor style issue, not a soundness issue -- the existence of such a theta is clear from the construction.

**Theorem 6:** Correctly identifies v2's flat loss at ln(vocab_size) as confirmation of Theorem 4's corollary. Sound.

## Prediction vs Measurement

PAPER.md contains a prediction-vs-measurement table (lines 12-17).

| Criterion | Prediction | Measured | Match? | Assessment |
|-----------|-----------|----------|--------|------------|
| K913: grad_norm > 0 | [0.1, 100] w.p. 1 | 6.301 | YES | Core theorem verified. Prediction wide but correct. |
| K914: loss < 2.0 in 200 steps | Start ~11.93, descend | Start 1.945, end 1.076 | PARTIAL | Starting point wrong by 6x (acknowledged). K914 trivially satisfied from step 0 (acknowledged). Loss does descend, confirming gradient flow works. |
| K915: quality_ratio >= 70% | 70-90% | 83.3% | YES | Within predicted range. Not stat. sig. at n=200 (acknowledged). |

PAPER.md honestly flags both the K914 prediction failure and K915 statistical limitations.

## Novelty Assessment

Limited novelty, correctly scoped. The core contribution is:
1. Identifying that MLX module attribute mutation severs hypernetwork gradients (known property of functional autodiff, but valuable to document in the M2P context).
2. Demonstrating that the functional forward pattern works for M2P on a real LLM (Qwen3-0.6B) and real benchmark (GSM8K).
3. Achieving 83.3% quality ratio in a single experiment, suggesting the M2P architecture scales beyond toy models.

This is engineering verification of a known principle, not a novel mathematical contribution. The paper correctly positions it as such.

## Macro-Scale Risks (advisory, not blocking)

1. **M2P parameter ratio:** 357M/600M = 59% at this scale. The B-heads grow as O(n_layers * d_M2P * rank * proj_out). For Qwen3-4B (d_model=3584, 40 layers), naive scaling gives ~10B M2P params -- larger than the base model. Shared heads or bottleneck architectures will be essential.
2. **Statistical power at n=200:** The 83.3% quality ratio is directional. Macro experiments need n >= 1000 to resolve 1-2pp differences.
3. **Training budget:** 200 steps on 2000 examples is extremely limited. The training curve shows loss still declining at step 200 (1.076). More steps may improve quality_ratio.

## Residual Issues (non-blocking)

1. **Stale docstring in run_experiment.py line 7:** Says "quality_ratio = M2P_acc / SFT_acc >= 70%" but the actual code (line 910-914) correctly computes (m2p-base)/(sft-base). Cosmetic only; does not affect results. Should be cleaned up in future.

## Finding Status Assessment

- **Recommended status: supported**
- K913 (gradient flow) is conclusively verified -- this is the core theorem.
- K915 (quality ratio) passes the threshold but is not statistically significant -- honest uncertainty.
- K914 had a wrong quantitative prediction -- acknowledged openly.
- The functional LoRA forward pattern is proven correct for MLX hypernetworks.
- Quality parity with SFT is directionally supported but needs larger evaluation to confirm.

## Verdict

**PROCEED**

All 5 blocking fixes and 2 advisory fixes have been properly applied. The paper is now honest about:
- The K914 prediction failure (output_scale not modeled, wrong starting loss by 6x)
- The K915 statistical limitations (overlapping CIs, "not statistically significant")
- The M2P parameter overhead (357M for 600M base, ~4.6x ratio)
- The SHINE reference unverifiability (explicit disclaimer)
- The Theorem 5 proof gap (now closed with analytic function argument)

The core finding -- functional LoRA forward fixes hypernetwork gradient flow in MLX -- is conclusively verified by K913 (grad_norm=6.301). The quality finding (83.3% quality ratio) is directionally supported but acknowledged as noisy. Finding status "supported" is appropriate.

Record as: **supported finding** with caveats on K914 prediction failure and K915 statistical significance.
