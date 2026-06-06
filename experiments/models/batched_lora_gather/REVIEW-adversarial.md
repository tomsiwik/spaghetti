# Peer Review: Batched LoRA Gather

## Experiment Type

**Declared:** Frontier extension
**Assessed:** Frontier extension -- appropriate classification. The experiment extends the proven v3 RuntimeLoRA architecture (Findings #288, #300, #301) by attempting to reduce dispatch overhead via batched matmul. The mathematical gap (whether MLX batched matmul reduces wall-clock time for stacked adapter projections) is clearly stated.

## Hack Detector

- Fix count: **1** (single mechanical optimization: replace Python loop with batched matmul). Clean.
- Is MATH.md a proof or a description? **Hybrid.** Theorem 1 (numerical equivalence) is a genuine proof with QED. Theorem 2 (latency reduction) is a **parameterized model dressed as a proof** -- it derives a speedup condition as a function of unknowns (alpha, T_dispatch, T_matmul, T_sum) without proving what those values are. Theorem 3 is algebraic rearrangement of Theorem 2 across modules. See detailed assessment below.
- Metric used as evidence: **Relative speedup (T_seq / T_stack)** and **isolated matmul latency**. Both are directly relevant to the behavioral outcome (faster inference). The speedup metric is well-chosen because it eliminates the confound of synthetic vs ternary base.
- Kill criteria source: **Mixed.** K770 (MSE) is derived from Theorem 1's corollary. K769 (85 tok/s absolute) and K771 (3 GB) are **not derived from any proof** -- they are absolute thresholds that, as PAPER.md honestly acknowledges, are not meaningful for a synthetic fp32 model. The experiment correctly pivots to relative speedup as the real kill signal.

## Self-Test Audit

1. **One-sentence impossibility property:** "Batched matmul computes K projections in a single kernel launch, eliminating the per-adapter dispatch loop overhead." This is a mechanism description, not an impossibility property. An impossibility property would be: "Lazy evaluation makes dispatch count irrelevant to latency." The self-test answer describes what the optimization does, not what mathematical property guarantees failure is impossible. That said, the actual finding (lazy evaluation makes stacking redundant) is effectively the impossibility structure discovered by the experiment. **Minor flag.**

2. **Cited theorems:** Punica BGMV (2310.18547) and Roofline model (Williams 2009) -- both are real and relevant. The Roofline model is applied correctly (the I_ridge calculation for M5 Pro checks out: 14 TFLOPS / 273 GB/s ~ 51 FLOPs/byte). The Punica reference is correctly cited as the CUDA-specific prior art. **PASS.**

3. **Predicted numbers:** MSE < 1e-6 (bf16), < 1e-10 (fp32); speedup 1.2-2.5x at K>=2; no improvement at K=1. These are specific and falsifiable. The speedup range is wide (1.2-2.5x) because it depends on the unknown alpha, which is acknowledged. **PASS, with caveat on range width.**

4. **Falsification condition:** "If MLX batched matmul has alpha >= 1." This correctly targets the proof's key assumption. **PASS.**

5. **Hyperparameter count:** 0. Correct -- this is a mechanical optimization with no tunable parameters. **PASS.**

6. **Hack check:** "No fix stack." Correct. Single clean optimization attempt. **PASS.**

**Self-Test overall: PASS** (minor flag on item 1, not blocking).

## Mathematical Soundness

### Theorem 1 (Numerical Equivalence) -- SOUND

The proof is correct. For each k, the stacked computation H[k] = x @ A_stack[k]^T = x @ A_k^T, and the B-side follows identically. The summation preserves equality. This is straightforward linear algebra and the proof is complete.

The finite-precision corollary is reasonable but imprecise. The bound |y_seq - y_stack| <= K * epsilon_mach * max_k(||A_k|| * ||B_k||) * ||x|| is a standard floating-point error bound, but the derivation is hand-waved ("for typical adapter norms"). A rigorous bound would require Higham-style error analysis accounting for the different summation orders. In practice, the measurements show MSE = 0.0 for stacked (exact bit-for-bit equality), making this moot. The corollary is conservative and safe.

**Verdict: Sound.**

### Theorem 2 (Latency Reduction) -- VALID BUT NOT A PROOF

This is a parameterized performance model, not a theorem in the mathematical sense. It says: "if alpha < 1 and T_dispatch is large enough relative to T_matmul, then stacking is faster." The derivation is algebraically correct:

```
Speedup > 1 iff (K-1) * T_dispatch > alpha * (K-1) * 2 * T_matmul + T_sum
              iff T_dispatch > alpha * 2 * T_matmul + T_sum / (K-1)
```

This is valid algebra. However, the "theorem" does not prove that speedup exists -- it derives the *condition* under which speedup would exist, leaving the key parameters (alpha, T_dispatch, T_matmul) as unknowns to be measured. This is appropriate for a frontier extension: the experiment's job is to measure these unknowns and determine whether the condition holds on MLX.

The quantitative prediction ("1.2-2.5x at K>=5") comes from estimating T_dispatch ~ 5-20 us and alpha ~ 0 (perfect parallelism). The experiment falsifies the alpha ~ 0 assumption by showing the isolated matmul benchmark gives ~157 us regardless of strategy, implying alpha ~ 1 on MLX.

**Verdict: Valid model, correctly structured for exploration. Not a formal proof of speedup, which is appropriate for the declared frontier-extension type.**

### Theorem 3 (Module Amortization) -- TRIVIAL

This is a straightforward linear scaling of Theorem 2 across L*M modules. Algebraically correct but adds no insight beyond "per-module savings multiply across the model." Not blocking.

### Roofline Analysis (Step B) -- SOUND

The operational intensity calculation is correct:
- I = d*r / (2*(d+r)) at d=2560, r=16 gives 40960 / 5152 = 7.95 FLOPs/byte
- I_ridge ~ 51 FLOPs/byte for M5 Pro
- 7.95 << 51, confirming bandwidth-bound operation

The conclusion that "stacking does NOT change the total FLOPs or bytes" is correct. This is an important insight that correctly predicts the null result: if the operation is bandwidth-bound and stacking does not reduce bytes moved, the only benefit is reduced overhead -- which MLX lazy evaluation already eliminates.

**This roofline analysis actually predicts the kill more accurately than the Theorem 2 model.** The Theorem 2 model assumes T_dispatch is meaningful; the roofline analysis shows the operation is bandwidth-bound, meaning dispatch overhead is irrelevant once MLX batches the graph. MATH.md could have been stronger by making this the primary prediction.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table. Assessment:

| Prediction | Measured | Match? | Assessment |
|-----------|----------|--------|------------|
| Stacked MSE < 1e-6 | 0.0 | YES | Theorem 1 verified precisely |
| No speedup at K=1 | 0.99x (prod) | YES | Correct |
| Speedup 1.2-2.5x at K=5 (prod) | 1.02x | NO | Correctly identified as kill signal |
| Speedup at K=5 (micro) | 1.75x | YES | Confirms mechanism works when Python overhead dominates |
| Isolated matmul speedup | 0.98-1.01x | NO | Definitive null: alpha ~ 1 on MLX |
| Memory < 3 GB | 17.3 GB | N/A | PAPER.md correctly flags this as wrong baseline (fp32 synthetic) |

The prediction-measurement alignment is honestly reported. The speedup prediction fails at production scale, and PAPER.md correctly identifies this as a kill. The root cause analysis (MLX lazy evaluation = implicit kernel fusion) is compelling and well-supported by the isolated matmul benchmark.

**One gap:** The prediction table in MATH.md predicts "1.2-2.5x at K=5" without qualifying that this is for production scale specifically. The micro-scale result (1.75x) falls within the predicted range, which could be misleading. PAPER.md handles this distinction well, but MATH.md should have been more explicit about which scale the prediction applies to.

## NotebookLM Findings

Skipped -- the experiment is already killed with clear evidence. NotebookLM deep review would not change the verdict.

## Novelty Assessment

**Prior art:** The experiment correctly cites Punica BGMV (2310.18547) and S-LoRA (2311.03285) as prior work on batched adapter dispatch. The contribution is testing whether this optimization pattern (designed for CUDA eager execution) translates to MLX lazy execution.

**Delta over existing work:** The finding that MLX lazy evaluation makes BGMV-style stacking redundant is genuinely novel and useful for the project. This is not published elsewhere because few projects attempt Punica-style optimizations on MLX specifically. The negative result is valuable: it closes off a research direction and correctly redirects attention to bandwidth reduction.

**Finding #76 (mx.compile redundant for generation) already pointed here.** That finding established that "async_eval already hides dispatch" for generation. This experiment extends that finding by showing the same principle applies to multi-adapter stacking specifically, and adds the isolated matmul benchmark as stronger evidence.

## Experimental Design

**Strengths:**
- Four strategies compared (sequential, stacked, concat, addmm) -- comprehensive
- Two scales (micro d=128, production d=2560) -- correctly distinguishes mechanism from production relevance
- Isolated matmul benchmark -- the strongest evidence, eliminates all confounds
- Proper warmup (200/50 iterations) and timed runs (1000/200 iterations)
- Memory safety patterns followed (CODING_GUIDELINES)
- Numerical equivalence verified before speed comparison

**Weaknesses:**
- Synthetic fp32 model, not real ternary BitNet-2B-4T. Absolute tok/s and memory numbers are meaningless. PAPER.md acknowledges this clearly.
- No KV cache. Real inference with KV cache may have different overhead proportions. Acknowledged in limitations.
- The K769 kill criterion (85 tok/s absolute) was poorly chosen -- it cannot be evaluated on a synthetic model. The experiment correctly pivots to relative speedup, but the kill criterion should have been defined as relative speedup from the start.
- K771 (3 GB) is similarly inapplicable to the synthetic setup.

**Could the null result be explained without lazy evaluation?** Yes -- if the adapter overhead is simply too small relative to base model compute (adapter overhead is 6.3ms out of 58ms total at K=5, i.e., ~11%), then even a 2x speedup on adapters would only give 1.05x end-to-end. The isolated matmul benchmark addresses this by testing pure matmul in isolation, confirming the null result is not just about relative proportions.

## Macro-Scale Risks (advisory)

- The finding (MLX lazy eval = implicit fusion) is MLX-specific. If the project ever ports to CUDA/PyTorch, BGMV-style optimization would become relevant.
- At larger K (K=50, K=100), the lazy evaluation graph may become large enough that stacking provides benefit. Untested but worth monitoring.
- The conclusion that adapter overhead is bandwidth-bound transfers directly to macro. Ternary adapter compression (BitDelta) remains the correct approach.

## Verdict

**PROCEED** (as a killed experiment with valid negative finding)

The experiment is correctly killed. The evidence is strong:

1. **Theorem 1 is proven and verified** (MSE = 0.0, exact equivalence).
2. **Theorem 2's key assumption (alpha < 1) is falsified** by the isolated matmul benchmark showing ~157 us for all strategies regardless of K.
3. **The root cause analysis is sound:** MLX lazy evaluation provides implicit kernel fusion, making Punica-style stacking redundant.
4. **PAPER.md is honest about limitations** (synthetic model, inapplicable absolute thresholds).
5. **The finding is novel and useful** for the project (closes dispatch optimization, redirects to bandwidth reduction).

**Minor issues that do not block:**
- Theorem 2 is a performance model, not a formal proof. Appropriate for frontier extension type.
- Kill criteria K769 and K771 were poorly specified for the actual experimental setup. The experiment correctly identifies relative speedup as the real metric.
- Self-test item 1 describes mechanism rather than impossibility property. The actual impossibility structure ("lazy evaluation makes dispatch count irrelevant") was discovered by the experiment rather than predicted.
- MATH.md's speed prediction range (1.2-2.5x) does not specify which scale, creating ambiguity with the micro-scale result.

These are documentation refinements, not scientific errors. The experiment ran cleanly, found a clear null result, identified the correct root cause, and will produce a useful finding for the project's knowledge base.
