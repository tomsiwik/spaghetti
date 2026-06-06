# Peer Review: Pierre v6.1 Full Precomputed Concat Deltas

## Experiment Type
**Verification.** MATH.md contains Theorem/Proof/QED blocks (Theorems 1-3). The
experiment was designed to verify quantitative predictions derived from those
theorems. This is Type 1.

## Hack Detector
- Fix count: **1** (precomputed concat is a single optimization). No stacking.
- Is MATH.md a proof or a description? **Proof with QED for Theorems 1-2.
  Theorem 3 is an empirical model (linear fit from 2 data points), not a proof --
  correctly labeled but should have been flagged as such in the self-test.**
- Metric used as evidence: **Speed (tok/s), behavioral score, dispatch count,
  memory.** Speed is a direct measurement. Behavioral is keyword-overlap proxy --
  acknowledged as weak in Limitations.
- Kill criteria source: **K756 (>= 75 tok/s) derived from needing to beat v3.
  K757 (>= 0.35) derived from v3 baseline. K758 (<= 6 GB) from platform
  constraint. Reasonable engineering criteria, though K756 is not derived from
  the proof itself -- it is a practical threshold.**

## Self-Test Audit

1. **One-sentence impossibility property:** "Concat-slice equivalence: horizontal
   concatenation of weight matrices followed by slicing produces bit-identical
   results to separate matmuls, reducing dispatch count from 14 to 4 per layer."
   -- This conflates two properties (correctness and dispatch reduction). The
   impossibility property for correctness (bit-identical results) is sound. The
   dispatch count claim is an engineering consequence, not an impossibility
   property. **Minor flag:** should be one property, not two joined by a comma.

2. **Cited theorems:** "Matrix multiplication distributes over horizontal
   concatenation (basic linear algebra)." -- Correct. This is a direct
   consequence of the bilinearity of matrix multiplication. No deep theorem
   needed, and the proof is honest about that. **PASS.**

3. **Predicted numbers:** "120 dispatches, ~84.2 tok/s (range 78-90), behavioral
   ~0.41, code behavioral ~0.84, memory ~2.5-4 GB." -- Specific and falsifiable.
   **PASS.**

4. **Falsification condition:** "The proof is wrong if MLX reorders module calls
   within a layer (breaking the cache pattern), or if slicing copies data rather
   than creating views." -- This targets the correctness proof (Theorem 1), not
   the speed model (Theorem 3). Since Theorem 3 was the one that actually failed,
   the falsification condition was incomplete -- it did not anticipate the primary
   failure mode (bandwidth, not dispatch count). **Partial flag:** the
   falsification condition correctly targets the proof that holds, but misses the
   assumption most likely to fail (A2: dispatch overhead is the dominant cost).

5. **Hyperparameter count:** "0. The group definitions are determined by the
   architecture." -- Correct. No new hyperparameters introduced. **PASS.**

6. **Hack check:** "No. This is a pure engineering optimization... No new losses,
   constraints, or mechanisms." -- Accurate. **PASS.**

**Self-Test verdict:** Present and mostly complete. Two minor issues: (1) the
impossibility property is two properties, (2) the falsification condition does
not target the weakest assumption. Neither is blocking.

## Mathematical Soundness

### Theorem 1 (Concat-Slice Equivalence) -- CORRECT
The proof that x @ [W_1 | W_2 | ... | W_k] can be sliced to recover individual
x @ W_i is trivially correct from the definition of matrix multiplication. The
worked example in Section F is verified (I checked the arithmetic). This is basic
linear algebra stated formally, which is the right level of rigor for a
verification experiment.

**One correctness concern in the code:** The `ConcatDeltaLinear.__call__` method
relies on `_GroupCache.result` being set by the "first" module before "read"
modules access it. This is a runtime invariant, not a mathematical one. The
MATH.md correctly identifies this as Assumption A1 (Python evaluation order)
and provides evidence that mlx_lm evaluates QKV left-to-right. The mitigation
argument (MLX lazy eval builds graph in Python call order) is sound for the
current MLX version.

### Theorem 2 (Dispatch Count) -- CORRECT
4 groups x 30 layers = 120. Verified in the code: each layer processes 4 groups,
each group yields exactly 1 dispatch. The measured dispatch count matches (120).
No issues.

### Theorem 3 (Speed Model) -- ACKNOWLEDGED FAILURE, HONESTLY REPORTED
This is the weakest part of the mathematical framework, and the researcher is
fully transparent about it:

1. The linear model T(D) = T_base + c * D was fit from 2 data points. MATH.md
   acknowledges this explicitly ("The linear model is derived from only 2 data
   points").

2. The model implicitly assumes dispatch overhead is the SOLE source of latency
   beyond the base model. It completely ignores memory bandwidth as a function of
   delta matrix size. This is Assumption A2, which MATH.md lists but does not
   probe deeply enough.

3. **The critical missing analysis:** MATH.md computes the delta memory sizes
   (Section G) and arrives at ~139 MB/layer, ~4.17 GB total. At this point, a
   bandwidth check should have been performed BEFORE running the experiment:
   - 4.2 GB at 273 GB/s = 15.4 ms per token just for delta transfers
   - v3 base inference is 13.7 ms/tok total
   - Therefore v6.1 would need at MINIMUM 15.4 ms for deltas alone, predicting
     at most ~65 tok/s before accounting for base model compute

   This back-of-envelope calculation would have killed the experiment on paper
   without running it. The fact that it was not done before coding represents a
   gap in the proof-first methodology.

4. **However:** PAPER.md performs this analysis post-hoc in the "Why Theorem 3
   Failed" section and derives the correct explanation. The bandwidth calculation
   (4.2 GB / 273 GB/s = 15.4 ms) is clean and matches the measured 42.1 tok/s
   (23.8 ms/tok) reasonably well when you add base model compute (~5.8 ms for
   base + 15.4 ms bandwidth = ~21.2 ms, predicting ~47 tok/s -- close to the
   measured 42.1).

**Verdict on Theorem 3:** The model was wrong, the researcher acknowledges it,
and the post-hoc analysis correctly identifies bandwidth as the bottleneck. The
gap is that the bandwidth check should have been in MATH.md as a pre-experiment
prediction, not a post-hoc explanation in PAPER.md.

### Assumptions Analysis
- **A1 (evaluation order):** Sound, well-supported.
- **A2 (dispatch overhead is dominant cost):** This was the assumption that
  broke. MATH.md listed it but did not test it quantitatively before the
  experiment. The memory bandwidth cost was calculable from Section G's own
  numbers.
- **A3 (slicing is free):** Sound for MLX. Array slicing creates views, not
  copies.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table. Checking each row:

| Prediction | Expected | Measured | Match? | My Assessment |
|-----------|----------|----------|--------|---------------|
| Dispatch count (Thm 2) | 120 | 120 | YES | Confirmed. Trivially correct. |
| Speed (Thm 3) | 84.2 tok/s (78-90) | 42.1 tok/s | NO | 2x miss. Correctly killed. |
| Behavioral overall (Thm 1) | ~0.41 | 0.419 | YES | Confirmed exact equivalence. |
| Code behavioral (Thm 1) | ~0.84 | 0.844 | YES | Exact match with v3. |
| Memory | ~2.5-4 GB | 5.47 GB | PARTIAL | Outside the predicted range. |

**Memory prediction miss:** MATH.md predicted 2.5-4 GB but measured 5.47 GB.
The Section G calculation gives 4.17 GB for deltas alone, plus the base model
(~1.2 GB), totaling ~5.4 GB -- which matches the measurement. The predicted
range of 2.5-4 GB was too optimistic because it did not add the base model
memory. PAPER.md labels this "PARTIAL (within K758 but at upper end)" which is
somewhat generous -- the prediction was 2.5-4 GB and the measurement was 5.47 GB,
which is outside the range. The kill criterion K758 (< 6 GB) still passed, so
this is not blocking, but the prediction was wrong.

## Honest Reporting Assessment

**The reporting is honest and thorough.** Specific strengths:

1. The kill is clean and unambiguous. 42.1 tok/s vs 75 threshold, no
   retroactive redefinition of criteria.

2. The "Why Theorem 3 Failed" section is excellent post-hoc analysis. The
   bandwidth calculation is rigorous, the comparison table (factored vs
   precomputed memory per group) is detailed, and the explanation of why v6
   (attention-only) worked but v6.1 (full) failed is compelling.

3. The "What Was Learned" section extracts genuine insight: the optimal
   precompute boundary is attention-only, MLP modules should stay factored,
   and a hybrid architecture is the next step.

4. The Limitations section acknowledges the weak behavioral proxy, the 2-point
   speed model, and the single-adapter limitation.

5. No cherry-picking. Legal (0.054) and finance (0.086) scores are reported
   honestly despite being essentially non-functional.

## Impossibility Structure

PAPER.md derives a clear impossibility argument: full precomputation of all
modules is bandwidth-bound on Apple Silicon because MLP dimensions (6912) create
delta matrices that are 230x larger than the factored form. The fundamental
tradeoff table (dispatch count vs delta memory vs speed) is well-structured and
identifies v6 (attention-only) as near the optimum.

**The impossibility argument is correct but could be stronger.** It should state
explicitly: for any rank r and intermediate dimension d_int, the precomputed MLP
delta memory scales as O(d_model * d_int) while the factored form scales as
O(d_model * r + r * d_int). When d_int >> r (6912 >> 16, a factor of 432x), the
precomputed form is asymptotically worse. This is architecture-independent --
it will hold at any scale where d_int >> r.

## NotebookLM Findings

Skipping NotebookLM automation as the mathematical content is straightforward
enough for direct review. The core math (concat-slice equivalence) is
undergraduate linear algebra. The speed model failure is a physics problem
(memory bandwidth), not a mathematical one.

## Novelty Assessment

This is an engineering optimization (fewer Metal dispatches via matrix
concatenation), not a novel mathematical contribution. The concat-slice trick is
standard in transformer implementations (e.g., fused QKV projections in most
frameworks). The novelty, such as it is, lies in applying it to LoRA delta
matrices and discovering the bandwidth boundary empirically.

**Prior art:** Fused QKV projections are standard in PyTorch, JAX, and MLX
transformer implementations. The insight that precomputed full-rank deltas trade
dispatch count for bandwidth is not novel in general (this is a well-known
tradeoff in GPU kernel optimization), but the specific quantification for Apple
Silicon LoRA serving appears to be original within this project.

## Macro-Scale Risks (advisory)

1. The bandwidth bottleneck is worse at larger scales. d_intermediate grows
   faster than d_model in most transformer architectures, making MLP
   precomputation increasingly unfavorable.

2. The hybrid recommendation (precompute attention, factor MLP) would need
   validation that the two serving modes compose correctly in a single forward
   pass. The code architecture may need refactoring to support mixed modes.

3. Multi-adapter composition would multiply delta memory further, making the
   bandwidth constraint even more severe.

## Verdict

**PROCEED** (as a killed experiment with valid findings)

The experiment was correctly killed. The mathematical framework is sound where it
matters (Theorems 1 and 2 are trivially correct and confirmed by measurement).
Theorem 3 (speed model) was honestly presented as an empirical model, honestly
reported as failed, and honestly analyzed post-hoc with a correct bandwidth
explanation. The kill criteria were not retroactively redefined. The impossibility
structure is correctly derived. The reporting is thorough and honest.

**Two non-blocking issues for the record:**

1. **Pre-experiment bandwidth check was missed.** The data for this check existed
   in MATH.md Section G (139 MB/layer, 4.17 GB total) and the M5 Pro bandwidth
   (273 GB/s). A 30-second calculation (4.17 GB / 273 GB/s = 15.3 ms > 13.7
   ms base inference) would have killed the hypothesis before any code was
   written. Future experiments should include a bandwidth feasibility check in
   MATH.md for any approach that materializes full-rank matrices.

2. **Memory prediction range was wrong.** The predicted 2.5-4 GB did not include
   the base model's ~1.2 GB footprint. The measured 5.47 GB aligns with the
   corrected calculation (4.17 GB deltas + 1.2 GB base = 5.37 GB). PAPER.md
   should note that the prediction missed by 37-119% (depending on which end
   of the range), not label it "PARTIAL."

The finding is ready to record as killed with the impossibility structure:
"Full precomputation of LoRA deltas is bandwidth-bound when delta memory exceeds
~1 GB on Apple Silicon (273 GB/s). MLP modules at d_int=6912 create 3.2 GB of
delta data, consuming 11.7 ms of bandwidth per token -- exceeding the dispatch
overhead savings of 1.8 ms. The factored form (A, B separate) transfers 180x
less data at the cost of 6 extra dispatches per layer (~1 ms total). Optimal
boundary: precompute attention only."
