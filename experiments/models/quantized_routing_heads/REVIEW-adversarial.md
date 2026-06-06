# Peer Review: Quantized Routing Heads

## NotebookLM Findings

Skipped -- the experiment is straightforward enough that a NotebookLM deep review would not surface additional concerns beyond what manual analysis reveals.

## Mathematical Soundness

### A. Proof Completeness (BLOCKING)

**FAIL.** MATH.md does not contain a formal Theorem/Proof/QED block. It contains:
- A description of symmetric uniform quantization (textbook material)
- A per-weight error bound (correct but standard)
- A first-order output perturbation bound (line 27-29, correct in form)
- An informal argument that "logit margins are large so sign flips are unlikely"

This is a **mechanism description with equations**, not a proof. To qualify as a proof, MATH.md would need:

> **Theorem:** For a 2-layer MLP routing head with ReLU activation, weights W1, W2 and input x, if the fp32 logit satisfies |logit(x)| > delta, then b-bit symmetric quantization preserves sign(logit) whenever delta > [explicit bound in terms of ||W1||, ||W2||, ||x||, b].
>
> **Proof:** [Derivation using submultiplicativity of operator norms, triangle inequality, etc.] QED.

Instead, the document jumps from "per-weight error is ~7.1% relative" to "int4 maintains >95%" without connecting them through the perturbation bound on line 27-29. That bound is stated but never evaluated to produce the >95% prediction.

### B. Proof Correctness

The perturbation bound on line 27-29 is structurally correct (first-order Taylor expansion of the composed function) but:

1. **Missing second-order terms.** The bound is linear in delta_W1 and delta_W2 separately but omits the cross-term delta_W2 * ReLU(delta_W1 * x). For int4 with ~7% relative error, this cross-term could be ~0.5% -- small but not proven to be negligible.

2. **Wrong d_model throughout.** MATH.md uses d_model=2048 for all calculations. The actual model (BitNet-2B-4T) has d_model=2560. This makes the absolute memory predictions in the table wrong:
   - Predicted: 256 KB/head. Actual: 320 KB/head.
   - Predicted N=100: 25.6 MB. Actual: 32.8 MB.
   - The percentage reductions (75%, 87.5%) are correct since they depend only on bit-width ratios.

3. **The key quantitative prediction ("int4 maintains >95%") is not derived from the bound.** The argument is qualitative: "logit magnitudes >> 0, so 7.1% error won't flip signs." A proper derivation would compute the bound using measured weight norms and input norms, yielding a minimum margin requirement. The experiment shows mean logit magnitude ~4.0 and max int4 error ~0.045, so the margin-to-error ratio is ~89:1. This is trivially safe, but the proof should have predicted this ratio.

### C. Prediction Verification

The prediction-vs-measurement table in PAPER.md exists and is well-structured. However:

| Prediction | Issue |
|-----------|-------|
| "Int8 max logit diff < 1% of logit" | The prediction says "< 1% of logit" but no specific logit magnitude was predicted. Measured: max diff 0.016, mean logit ~4.5, so actual ratio ~0.35%. Passes, but the prediction was vague. |
| "Int4 max logit diff < 7% of logit" | Same issue. Measured: 0.045 / ~4.5 = ~1.0%. Passes by 7x margin. |
| "Int4 accuracy > 95%" | Measured 100%. The prediction was too conservative -- a proper bound would have predicted 100% given the 89:1 margin-to-error ratio. |

**Kill criteria are not derived from the proof.** K1 (int4 acc >= 90%) is arbitrary, not linked to the perturbation bound. A proof-derived kill criterion would be: "if min |logit| / max_quantization_error < 2, KILL" (sign flip possible).

## Novelty Assessment

**Low novelty.** Post-training quantization of small classifiers is well-established. The specific contribution is:
1. Empirical confirmation that routing heads in this architecture survive int4 quantization.
2. Memory projection at scale (N=100 to N=853).

This is useful engineering characterization, not a research finding. No new mechanism is proposed. The result is entirely expected given the measured logit margins.

**Relevant prior finding:** The "more_adapters_is_better" experiment (KILLED) showed that binary routing heads collapse at N>=10 with 46% of domains falling back to base-only. This means the 5-domain, 100% accuracy baseline is the easy case. The experiment acknowledges this in "What This Doesn't Test" (item 2: "Routing at N>5 domains"), but this is the critical gap. Quantizing already-broken routing heads would be meaningless; the experiment should have tested N=24 where margins are tight and quantization might actually cause failures.

## Experimental Design

### Strengths
- Clean separation of phases (extract hidden states, then quantize/evaluate)
- Both positive and negative samples evaluated per domain
- Latency measured with proper warmup (50 iterations) and averaging (1000 runs)
- Memory calculation tracks actual quantized storage, not dequantized runtime buffers
- Logit difference measured directly (not just accuracy)

### Weaknesses

1. **The "int4" is not actually int4 in memory.** Line 112: `w_q = mx.clip(w_q, -8, 7).astype(mx.int8)` -- the quantized values are stored as int8. The `memory_bytes()` method on line 156 divides by 2 to simulate packed int4, but the actual runtime memory is int8. The claimed "87.5% memory reduction" only holds if you implement bit-packing, which this experiment does not do. The dequantized fp32 weights are also kept in memory (line 136-137), meaning runtime memory is actually LARGER than fp32.

2. **Pre-dequantization defeats the purpose.** Line 135-137: weights are dequantized to fp32 at construction time and stored alongside the quantized weights. Inference uses the fp32 dequantized weights (line 143-150). This means:
   - The "latency overhead" is measuring fp32 matmul vs fp32 matmul (identical compute), not int4 dequantization overhead
   - The ~5% overhead is likely just noise from the extra object indirection, not dequantization cost
   - PAPER.md claims "The ~5% latency increase is from the dequantization step" -- this is factually wrong

3. **Only 25 validation samples per domain.** With 25 positive and 40 negative samples per head (65 total), a single misclassification changes accuracy by 1.5%. The "100% accuracy" claim means zero errors on 65 samples. This cannot distinguish 100% from 98.5% true accuracy.

4. **No adversarial/borderline analysis.** The experiment does not identify or test samples near the decision boundary. With mean logit magnitude ~4.0 and max quantization error ~0.045, the minimum margin is ~89x. But this only considers the validation set. The proof should have characterized the minimum margin over the training distribution.

## Macro-Scale Risks (advisory)

1. **N-scaling is the real question.** At N=5, routing heads have wide margins (mean logit ~4.0). The killed "more_adapters_is_better" finding shows margins collapse at N>=10. Quantization safety at N=24+ is untested and likely the actual failure mode.

2. **Per-token routing heads (mentioned in VISION.md) have different characteristics.** Per-sequence mean-pooled hidden states are smooth; per-token hidden states are noisier with potentially tighter margins.

3. **The memory savings are marginal in context.** At N=100, routing heads save 28.7 MB (32.8 to 4.1 MB). But N=100 adapters at ~1.9 KB each is only ~190 KB for the adapters themselves. The routing heads dominate adapter memory by 170x. This suggests the architecture should use cheaper routing (e.g., learned embeddings + cosine similarity) rather than optimizing the current routing heads.

## Verdict

**REVISE**

The experiment is competently executed engineering characterization, but fails the proof-first standard required by this research process.

### Required Revisions

1. **Write a formal proof (BLOCKING).** MATH.md must contain a Theorem/Proof/QED block that:
   - States the perturbation bound for 2-layer ReLU MLP under symmetric quantization
   - Evaluates the bound using measured weight norms (from the saved head.npz files) and input norms (from validation hidden states)
   - Derives a minimum margin requirement for sign preservation
   - Predicts the margin-to-error ratio per domain (should be ~89:1 based on results)
   - Derives kill criteria from the bound (e.g., "KILL if predicted margin-to-error < 2")

2. **Fix d_model error.** MATH.md uses d_model=2048; actual is 2560. Update all absolute memory calculations.

3. **Fix the int4 memory claim.** Either implement actual int4 bit-packing (so memory savings are real, not simulated) or clearly state that int4 memory is a projection assuming packed storage, not measured.

4. **Fix the latency claim.** PAPER.md states "The ~5% latency increase is from the dequantization step" but the code pre-dequantizes at construction time. Either implement real dequantize-on-the-fly inference and measure it, or correct the claim to "latency is essentially identical since both paths use fp32 matmul."

5. **Acknowledge the N=5 limitation more prominently.** The "more_adapters_is_better" finding showed routing heads break at N>=10. Testing quantization robustness at N=5 (where margins are enormous) does not validate the claim for production use at N=100+. Either test at N>=10 or downgrade the recommendation from "Use int4 in production" to "Int4 is safe at N=5; N>=10 requires validation."
