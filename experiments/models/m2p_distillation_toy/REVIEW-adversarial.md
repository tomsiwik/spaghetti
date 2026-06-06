# Peer Review: M2P Distillation Toy (Revision 1)

## Experiment Type
Guided exploration -- proven framework (Grassmannian orthogonality, Theorem 1), unknown parameter (achievable M2P quality ratio c on synthetic domains).

## Hack Detector
- Fix count: 1 (single M2P mechanism, no stacked fixes). CLEAN.
- Is MATH.md a proof or a description? **Mixed but honest.** Theorem 1 (Section C) is a genuine proof with QED. Section D ("Theorem 2") is correctly labeled a "Claim (guided exploration)" with an unknown constant to discover.
- Metric used as evidence: Quality ratio = (PPL_base - PPL_M2P) / (PPL_base - PPL_SFT). Median used for K847, which is an improvement over the prior run's mean. Appropriate for robustness to single outlier.
- Kill criteria source: K847 threshold (25%) derived from prior finding #339 (66.6%) with safety margin. K848 derived directly from Theorem 1. Both sound.

## Assessment of 4 Prior Blocking Fixes

### Fix 1: Self-Test section added to MATH.md -- ADEQUATE
Section I now contains all 6 self-test questions with substantive answers. The impossibility property (Q1) identifies round-robin training with heterogeneous loss scales as the structural failure mode. The hyperparameter count (Q5) correctly lists 6 hyperparameters and acknowledges none are swept. All items are filled with real content. PASS.

### Fix 2: Parameter-space vs activation-space scope clarified -- ADEQUATE
Section A now contains a clear scope caveat: "Theorem 1 proves zero Frobenius inner product between weight perturbations -- this is parameter-space orthogonality. It does not guarantee zero activation-space interference." The caveat correctly identifies the condition under which the cross-term `trace(B_j^T A_j^T x^T x A_i B_i)` vanishes (when `A_j^T A_i = 0`, which holds) and notes the practical argument (low-rank subspaces in high-dimensional space). The only mild concern: the caveat says the cross-term IS zero by Theorem 1 "only if A_j^T A_i = 0 (which holds)", and then says activation-space interference is "not formally zero for arbitrary data." This creates a slight contradiction. Let me parse more carefully. The cross-term in the activation output norm is `trace(B_j^T A_j^T x^T x A_i B_i)`. If A_j^T A_i = 0, then A_j^T (x^T x) A_i is NOT necessarily zero -- the x^T x matrix sits between the A matrices and prevents cancellation. The caveat correctly notes this distinction in the final sentence, but the parenthetical "(which holds)" applied to "only if A_j^T A_i = 0" is misleading because A_j^T A_i = 0 is necessary but not sufficient when x^T x is interposed. This is a minor imprecision, not a blocking issue. PASS with note.

### Fix 3: LoRA bug fixed -- VERIFIED CORRECT
The prior review's most serious finding was that the M2P forward path applied LoRA corrections additively to attention output rather than inside the attention mechanism. The revision now:
- Applies LoRA to q, k, v individually before softmax: `q = attn.wq(x_norm) + scale * (x_norm @ A) @ B`
- Applies LoRA to wo on the attention context: `wo(attn_ctx) + scale * (attn_ctx @ A_wo) @ B_wo`
- Applies LoRA to fc1 inside the MLP before relu: `fc1(x) + scale * (x @ A_fc1) @ B_fc1`, then `fc2(relu(...))`

This exactly matches the SFT `GrassmannianLoRA.__call__` which computes `base(x) + (x @ A) @ B * scale` per module. The computational graphs are now equivalent. PASS.

Additionally, the B-matrix dimensions were corrected: `GrassmannianLoRA` uses `base.weight.shape[0]` for the output dimension of B. For fc1 (`nn.Linear(64, 256, bias=False)`), `weight.shape[0] = 256`, so B is (4, 256). The M2P generates the same via `module_out_dims = [64, 64, 64, 64, 256]`. N_MEMORY was increased from 16 to 32 to accommodate the larger fc1 B-matrices (total per layer: 4*64*4 + 4*256 = 2048 = 32*64). Capacity is tight but exactly sufficient. PASS.

### Fix 4: Median quality used for K847 -- ADEQUATE
K847 now uses median instead of mean. The results.json confirms `median_quality_ratio: 0.219` and the kill threshold is 0.25. PASS.

## Mathematical Soundness

### Theorem 1 (Grassmannian A-Slot Orthogonality) -- CORRECT
Unchanged from prior review. Short, standard, correct. QR decomposition of (64, 20) matrix, column slices form orthonormal A-matrices, off-diagonal blocks of Q^T Q = I are zero. Substitution into Frobenius inner product yields zero for any B. Sound.

### Section D (Quality Lower Bound) -- CORRECTLY FRAMED AS EXPLORATION
Prediction: c in [0.30, 0.70]. Measured: median 0.219 (FAIL), per-domain range [-3.294, 0.556]. The prediction was falsified. Section D correctly identifies this as a guided exploration with an unknown constant, not a theorem.

### Capacity Analysis (Section E) -- CORRECT, UPDATED
The updated analysis correctly accounts for the increased B-matrix sizes and N_MEMORY=32. Total capacity per layer = 32 * 64 = 2048 values. Required = 4*(4*64) + 4*256 = 2048. Exact match. The analysis notes the compression ratio is 1:1, which is tight -- there is zero slack for the M2P to encode auxiliary information.

## Prediction vs Measurement

PAPER.md Section B contains a well-structured prediction-vs-measurement table with both revision 1 and prior run (Finding #341) columns. This is a clear improvement.

| Metric | Predicted | Measured (Rev 1) | Status |
|--------|-----------|-------------------|--------|
| Grassmannian cos | <= 1e-5 | 0.000000 | PASS |
| Mean quality | 0.30-0.70 | -41.2% | FAIL |
| Median quality | 0.30-0.70 | 21.9% | FAIL |
| K847 (median >= 25%) | PASS | 21.9% < 25% | FAIL |
| K848 (structural) | PASS | PASS | PASS |

Per-domain breakdown is provided in Section C. This is thorough.

## Detailed Analysis of Revision 1 Results

### The LoRA Fix Made Performance Worse

This is the most important finding. After correctly fixing the forward path to match SFT:
- Median quality dropped from 53.7% to 21.9%
- Mean quality dropped from 11.5% to -41.2%
- `reverse` domain collapsed from 53.7% to 10.2%
- `repeat` domain worsened from -146.8% to -329.4%

PAPER.md Section E provides a reasonable analysis of why: the correctly-shaped fc1 B-matrices (4, 256) add 1024 new parameters per layer that the M2P must generate, increasing the optimization difficulty. The prior "passing" result was based on a buggy forward path that happened to produce better numbers -- a classic case where fixing a bug reveals the true difficulty.

### B-Matrix Mode Collapse Persists

M2P B-matrix cosine = 0.9945 (prior: 0.9956). Nearly identical B-matrices across all 5 domains. This is the core failure mode. The M2P is not learning domain-specific adaptations.

### Impossibility Structure (Section F) -- WELL-SUPPORTED

The analysis that round-robin training with heterogeneous loss scales (5.44x ratio between parity and repeat) causes B-matrix centroid convergence is plausible and consistent with the data. The formal argument is:
- M2P gradient is the sum of per-domain gradients
- Domains with high base loss contribute larger gradient magnitude
- Without domain conditioning, the M2P cannot distinguish which domain it is currently serving
- Result: B-matrices converge to a single point that minimizes average loss, not per-domain loss

This is not a formal proof but a reasonable mechanistic explanation supported by the evidence (cos=0.9945 between domain B-matrices, catastrophic failure on the lowest-loss domain `repeat`).

One important nuance PAPER.md gets right: the `repeat` domain has the smallest base-SFT gap (1.098 - 0.500 = 0.598) while other domains have gaps of 2-4x larger. An adapter tuned for the average gradient will overshoot this domain, producing worse-than-baseline loss. This explains the -329% quality.

## Issues Found in Revision 1

### Issue 1: Phase 5 Evaluation Context Mismatch (NON-BLOCKING, but worth noting)

In Phase 5 (lines 510-526), the code generates B-matrices from `domain_data[name]["train"][0]` (line 512) and evaluates them (line 515), but then calls `m2p_loss(m2p, base, tokens, di)` for each validation token (line 520). Inside `m2p_loss`, the function calls `m2p_model(hidden_states)` where `hidden_states = base_model.get_hidden_states(tokens[None, :])` -- i.e., it generates B-matrices from the validation token being evaluated, NOT from the training context token.

This means:
- The B-matrices generated in lines 514-515 are thrown away and never used for quality evaluation
- Each validation token generates its own B-matrices from its own hidden states
- The M2P is adapting to each individual sequence, not to a domain context

This is arguably correct for the M2P paradigm (adapt at runtime from any context), but it differs from how the quality comparison is framed. The SFT adapters were trained on 300 steps of domain data. The M2P is adapting per-sequence. These are different settings. This does not invalidate the result (the M2P is still trying to produce useful B-matrices), but it is an asymmetry worth noting.

### Issue 2: Grassmannian Verification Measures A-Matrix Orthogonality, Not Delta Orthogonality (NON-BLOCKING)

The Phase 2 verification (lines 369-378) computes the cosine between A-matrix columns, which is guaranteed to be zero by construction (it's measuring Q^T Q off-diagonal blocks of an orthonormal matrix). This is tautological -- it confirms the QR implementation works but does not test anything about the composed system.

A more informative test would compute the Frobenius inner product between full weight deltas: `<scale * B_i @ A_i^T, scale * B_j @ A_j^T>_F`. This is what Theorem 1 actually guarantees. The current measurement skips the B-matrices entirely. However, since Theorem 1 proves the result for ANY B, and the A orthogonality is the sufficient condition, the measurement is logically sound -- just not as informative as it could be.

### Issue 3: MATH.md Self-Test Q1 Is Strong but Not Formal (NON-BLOCKING)

The self-test Q1 answer identifies gradient imbalance as the structural failure mode and gives a reasonable mechanistic argument: "without domain conditioning or loss normalization, the B-matrix gradient is sum of g_d, not per-domain." This is a good identification of the failure mode but falls short of a formal impossibility result. A true impossibility would need to show that the M2P's optimal B under the summed gradient necessarily fails K847, which would require bounding the quality ratio as a function of loss heterogeneity. This is acceptable for a guided exploration but should not be promoted to "conclusive."

### Issue 4: Prediction Range Was Too Optimistic (NON-BLOCKING)

MATH.md Section D predicted c in [0.30, 0.70] based on Finding #339 (66.6% on Qwen3-4B medical) and the assumption that "synthetic patterns are more compressible." The measured median is 0.219 and the mean is -0.412. The prediction range was substantially wrong. The reasoning that "synthetic domains have sharp pattern structure so M2P should learn faster" did not account for:
1. The M2P capacity is barely sufficient (2048/2048 values, 1:1 compression)
2. Round-robin training across 5 domains (vs. single domain in Finding #339)
3. Heterogeneous loss scales (5.44x ratio)

The prediction was based on single-domain results applied to a multi-domain setting -- a classic extrapolation error.

## Novelty Assessment

Unchanged from prior review. The decoupled architecture (frozen Grassmannian A for composition guarantee, M2P-generated B for domain knowledge) is a natural and sound combination of known components. The novel insight from this revision is that correctly implementing the forward path reveals the true difficulty of M2P training in the multi-domain setting.

## Macro-Scale Risks (advisory)

1. **B-matrix generation by activation reshaping has no scaling path.** At d=2048 with rank 16, each layer needs 5 modules with B-matrices totaling 16*(2048*4 + 2048) = 163,840 values per layer. N_MEMORY would need to be ~2560 tokens. Projection heads are mandatory at scale.

2. **Domain conditioning (the proposed fix) is necessary but changes the M2P's operating regime.** With domain embeddings, the M2P is told which domain to generate for. This moves from "adapt from context" to "generate known adapter from label." The vision's promise (add domain N+1 via one forward pass) requires the M2P to work from context alone. Domain conditioning is a valid training scaffold but the evaluation should also test the context-only case.

3. **The 1:1 capacity ratio (2048 needed / 2048 available) means zero headroom.** Any increase in model size, rank, or module count will require architectural changes (projection heads, factored B generation, etc.).

## Verdict

**KILL -- with clean exit and well-characterized impossibility structure.**

### Justification

1. **K847 FAIL is legitimate.** Median quality 21.9% < 25% threshold. The fix from revision 1 (correct LoRA forward path) actually worsened performance, revealing that the prior "near-pass" result was an artifact of a buggy evaluation. The true M2P quality in this setting is substantially below threshold.

2. **The impossibility structure is well-characterized.** B-matrix mode collapse (cos=0.9945) from round-robin training with heterogeneous loss scales is a clear, reproducible failure mode. The analysis in PAPER.md Section F correctly identifies the root cause and proposes three concrete fixes (domain conditioning, loss normalization, sequential training).

3. **Theorem 1 (Grassmannian orthogonality) is verified and sound.** K848 PASS is structural and guaranteed by construction. This result carries forward to future experiments.

4. **The revision process was well-executed.** All 4 blocking fixes from the prior review were correctly applied. The LoRA forward path now correctly matches SFT. The fact that fixing the bug made results worse is itself a valuable finding -- it reveals the true difficulty.

5. **No additional investigation needed at this scale.** The remaining unknowns (domain conditioning, loss normalization) are new experiments, not revisions of this one. Continuing to iterate within this experiment would add fixes on fixes without testing a distinct hypothesis.

### Recommendations for the Next Experiment

1. Test domain conditioning as a separate experiment with its own MATH.md, kill criteria, and predictions. The prediction should be derived from the impossibility structure found here: domain conditioning eliminates the gradient-averaging failure mode, so quality should rise above 25% with high confidence.

2. The Grassmannian A-slot architecture is proven correct and should be carried forward unchanged.

3. Consider whether the M2P memory capacity (1:1 ratio) is an independent bottleneck. Domain conditioning might fix mode collapse but still produce weak B-matrices if the M2P has no headroom to encode auxiliary features.

### Finding Status Recommendation

**Status: killed** (K847 FAIL, impossibility structure well-characterized, next steps identified). The Grassmannian orthogonality guarantee (K848 PASS) should be recorded as a separate supported/conclusive finding if not already captured elsewhere.
