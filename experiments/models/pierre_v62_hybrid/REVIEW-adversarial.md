# Peer Review: Pierre v6.2 Hybrid Precomputed Attention + Factored MLP

## Experiment Type
**Verification** (Type 1). MATH.md contains four Theorem/Proof/QED blocks (Theorems 1-4).
The experiment was designed to verify quantitative predictions from these theorems.

## Hack Detector
- Fix count: **1** (hybrid partition of existing injection strategies). No stacking.
- Is MATH.md a proof or a description? **Mixed.** Theorems 1-3 are genuine proofs with QED.
  Theorem 4 is an empirical model dressed in theorem notation -- it relies on calibrated
  constants from prior measurements, not derivations from first principles. This distinction
  matters because Theorem 4 is the one that failed.
- Metric used as evidence: Speed (tok/s, direct measurement), behavioral score (keyword
  recall proxy -- weak but consistent across v3/v6/v6.1/v6.2).
- Kill criteria source: K759 (>=75 tok/s) is an engineering threshold derived from needing
  to beat v3 (73 tok/s). K760-K762 derived from prior configurations. Kill criteria are
  reasonable engineering targets, not directly from the proof's predictions. The proof
  predicted 80-90 tok/s, but the kill threshold is 75 -- this gives a 5 tok/s cushion.
  Honest choice: if the proof predicted 80-90, the kill should have been >=80, not >=75.

## Self-Test Audit

1. **One-sentence impossibility property:** "The bandwidth budget partition: attention
   deltas (983 MB) stay within the ~1 GB bandwidth-efficient regime while MLP adapters
   in factored form add negligible bandwidth load (~27 MB)." -- This is one property
   (bandwidth partition) but it is stated in two clauses. The property is that the hybrid
   stays below the bandwidth phase transition. Acceptable. **PASS (minor: wordy).**

2. **Cited theorems:** Concat-Slice Equivalence (proven in v6.1), matrix multiplication
   associativity, two-regime speed model (empirical). The first two are legitimate. The
   "two-regime speed model" is labeled "empirical, calibrated from v3/v6/v6.1" -- this is
   honest. However, the two-regime model was FALSIFIED by this very experiment, meaning the
   proof's foundation was shaky from the start. The self-test should have flagged the speed
   model as the weakest link. **PASS with caveat: empirical model correctly labeled but
   its fragility underappreciated.**

3. **Predicted numbers:** 240 dispatches, ~1010 MB materialized, 80-90 tok/s (range 75-105),
   code behavioral ~0.84, overall behavioral ~0.41, memory ~2.5-3.5 GB. All specific and
   falsifiable. **PASS.**

4. **Falsification condition:** "The proof is wrong if (a) mixing ConcatDeltaLinear and
   RuntimeLoRA causes framework conflicts, (b) the two-regime speed model is fundamentally
   miscalibrated, or (c) factored MLP dispatches have much higher overhead than measured in
   v3." -- Condition (b) is precisely what happened. The falsification condition was correct
   and comprehensive. **PASS.**

5. **Hyperparameter count:** 0. The partition is derived from bandwidth analysis. **PASS.**

6. **Hack check:** "No. This is an optimal partition of the same adapter math across two
   proven injection strategies." -- Accurate. **PASS.**

**Self-Test verdict:** Complete and honest. The weakest item is #2 (the empirical speed
model is the foundation most likely to break, and the self-test acknowledges this in #4
but not in #2). Not blocking.

## Mathematical Soundness

### Theorem 1 (Hybrid Exact Equivalence) -- CORRECT
The proof that precomputed (x @ DeltaW) and factored (alpha * (x @ A) @ B) produce
the same correction is trivially correct from associativity of matrix multiplication.
The code implements this correctly:
- Precomputed path: `delta_W = (scale * (B.T @ A.T)).T` then `x @ delta_W` (line 170)
- Factored path: `(x @ A) @ B * alpha` (line 68 in pierre.py)

These are the same computation up to floating-point ordering. The claim of
"bit-identical" output is strictly false (floating-point reordering changes LSBs),
but this is a standard abuse of terminology and does not affect the behavioral
equivalence claim. Measurements confirm: behavioral scores match across all
configurations (0.844 code, 0.425 overall).

**Verified: sound.**

### Theorem 2 (Dispatch Count = 240) -- CORRECT
Per layer: 1 QKV dispatch + 1 O dispatch + 3 MLP modules x 2 dispatches = 8.
8 x 30 layers = 240. Measured: 240. Trivially confirmed.

**One subtlety worth noting:** The ConcatDeltaLinear "first" module does the matmul
and caches the result. The "read" modules access the cache. This means the QKV group
does have 3 Python-level __call__ invocations, but only 1 Metal dispatch (the matmul).
The other 2 are slice operations. Whether MLX treats slicing as a separate dispatch or
a view is implementation-dependent, but the measurement (240 dispatches reported by the
code's counting logic) is consistent with the theorem's definition of "dispatch" as a
matmul kernel launch, not a Python call. The counting in inject_hybrid (line 206:
`attn_dispatches += 1` for the whole QKV group) is correct.

**Verified: sound.**

### Theorem 3 (Memory = ~1010 MB) -- CORRECT
The arithmetic checks out:
- QKV: 2560 x (2560 + 640 + 640) x 2 = 2560 x 3840 x 2 = 19,660,800 bytes = 19.66 MB
- O: 2560 x 2560 x 2 = 13,107,200 bytes = 13.11 MB
- Per layer attention: 32.77 MB. 30 layers: 983 MB.
- MLP factored: 6 matrices per layer, ~0.91 MB per layer, 27.3 MB total.
- Total: 1010 MB.

Measured: 2.30 GB peak (which includes the 1.18 GB base model). Delta memory:
2.30 - 1.18 = 1.12 GB. The predicted 1010 MB = 1.01 GB is 10% below the measured
1.12 GB. This gap is within reason (MLX graph overhead, alignment padding, cache
allocations). PAPER.md reports 2.30 GB vs predicted 2.5-3.5 GB and calls this "YES
(below range, better)." This is slightly misleading -- the predicted range was for
TOTAL memory (base + deltas), and 2.30 GB is below the 2.5 lower bound. The actual
explanation is that the base model memory prediction was too high, or that MLX's
memory reporting captures a subset of allocations. Either way, the delta memory
prediction (1010 MB) is reasonably close to the back-calculated measurement (1120 MB),
and the discrepancy is not concerning.

**Verified: sound, minor prediction range mismatch acknowledged.**

### Theorem 4 (Speed Prediction) -- FALSIFIED, HONESTLY REPORTED

This is where the experiment's value lies: the falsification.

**The two-regime (max) model:**
```
T = T_base + max(c_factored * D_factored + c_precomp * D_precomp, M / BW)
```

Dispatch term: 0.0188 * 180 + 0.00606 * 60 = 3.38 + 0.36 = 3.74 ms
Bandwidth term: 1010 / 273000 * 1000 = 3.70 ms
T_predicted = 5.81 + max(3.74, 3.70) = 5.81 + 3.74 = 9.55 ms = 104.7 tok/s

The "more conservative additive model" gives:
T_additive = 5.81 + 3.74 + 3.70 = 13.25 ms = 75.5 tok/s

Measured: T_actual = 1000/67.4 = 14.84 ms

**Critical analysis of the speed model failure:**

The MATH.md presents a range of 75-105 tok/s with a "central estimate" of ~85 tok/s.
This range is honest about the uncertainty, but the lower bound (75 tok/s) already
requires the fully additive model PLUS some pipelining. The actual measurement (67.4)
falls BELOW even the fully additive prediction (75.5). The gap is:

14.84 - 13.25 = 1.59 ms unexplained overhead.

PAPER.md attributes this to "module type switching, framework overhead" (~1.7 ms).
This is plausible but untestable without Metal profiling. It could also reflect:
- Memory allocation overhead from having two different module types in the graph
- Python-level overhead from the ConcatDeltaLinear cache pattern
- MLX graph compilation inefficiency with mixed module types

The key finding -- that bandwidth and dispatch are ADDITIVE, not max -- is physically
sound. Metal dispatches execute on the GPU, and memory transfers happen on the memory
bus. These CAN pipeline in theory (GPU computes while bus loads next buffer), but in
practice, each precomputed dispatch reads its delta from memory (bandwidth cost) then
executes the matmul (dispatch cost). These are serial within a single module. Cross-module
pipelining would require Metal's async compute, which mlx_lm's sequential Python dispatch
does not exploit.

**The impossibility structure is correctly derived:** any hybrid configuration that
has BOTH factored dispatches AND materialized deltas pays both penalties because the
operations are serialized within a layer's forward pass.

**One concern about the "no intermediate optimum" claim:** PAPER.md states "The Pareto
frontier is NOT between v6 and v3 -- it IS v6 and v3." This is a strong claim that
rests on the additive cost model being correct for ALL possible partitions. The
experiment tested exactly one partition (attention precomputed, MLP factored). Other
partitions (e.g., precompute only QKV, not O, factor everything else) were not tested.
The additive model predicts they would also be suboptimal, but this is extrapolation
from one data point of the additive model. The claim is likely correct but formally
undertested.

**Verified: correctly falsified, honestly reported, post-hoc analysis sound.**

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table (lines 12-18):

| Prediction | Expected | Measured | PAPER says | My Assessment |
|-----------|----------|----------|------------|---------------|
| Dispatches (Thm 2) | 240 | 240 | YES | Confirmed. Trivial. |
| Code behavioral (Thm 1) | ~0.84 | 0.844 | YES | Confirmed. Exact. |
| Overall behavioral (Thm 1) | ~0.41 | 0.425 | YES | Confirmed. +3.7% above. |
| Peak memory (Thm 3) | 2.5-3.5 GB | 2.30 GB | YES (below) | Prediction missed. 2.30 < 2.5 lower bound. See Thm 3 analysis. |
| Speed (Thm 4) | 80-90 tok/s | 67.4 | NO (-16%) | Correctly killed. Even additive lower bound (75.5) missed. |

The table is honest. The memory "YES (below range, better)" is slightly generous --
the measurement is outside the predicted range (below it), which is a miss in the
favorable direction. This is acknowledged but not flagged as a prediction error.

**Assessment: Reporting is honest. Speed failure correctly identified and correctly
killed. No retroactive redefinition of kill criteria.**

## Kill Criteria Assessment

| Criterion | Threshold | Value | Status | My Assessment |
|-----------|-----------|-------|--------|---------------|
| K759 | >= 75 tok/s | 67.4 | FAIL | Correctly killed. Clean miss. |
| K760 | >= 0.80 code | 0.844 | PASS | Confirmed. |
| K761 | >= 0.35 overall | 0.425 | PASS | Confirmed. |
| K762 | <= 6.0 GB | 2.30 | PASS | Confirmed with large margin. |

**Were kill criteria retroactively redefined?** No. The 75 tok/s threshold was set
before the experiment ran (in run_experiment.py line 371: `k759 = r3["hybrid_tps"] >= 75.0`).
The kill is clean and unambiguous.

**Should the threshold have been higher?** MATH.md predicted 80-90 tok/s. Setting the
kill at 75 gives a 5-15 tok/s cushion. This is standard practice (set threshold below
prediction to account for noise), but it means the experiment was designed to pass even
if the max()/additive distinction did not matter much. The measurement at 67.4 is far
enough below 75 that this is not a borderline case.

## Impossibility Structure

PAPER.md derives the impossibility structure clearly (lines 99-114):

1. Costs are additive: T = T_base + c_factored * D_factored + c_precomp * D_precomp + M/BW
2. Any hybrid adds both dispatch overhead AND bandwidth cost
3. v6 (pure precomputed attention) minimizes dispatches at 60, bandwidth is 983 MB
4. v3 (pure factored) minimizes bandwidth at 27 MB, dispatches are 420
5. v6.2 hybrid: 240 dispatches + 1010 MB = pays both penalties

**The Pareto frontier claim:** "There is NO intermediate optimum because the costs are
additive." This is mathematically sound under the additive model. If T = T_base + f(D) + g(M),
and moving modules from factored to precomputed reduces D but increases M, and both
f and g are monotonically increasing, then any intermediate configuration is worse than
the convex hull of the two pure endpoints IF the tradeoff curve is convex (which it is
when f and g are linear). The formal argument should be:

For any split alpha in [0,1] (fraction of modules precomputed):
- D(alpha) = D_v3 - alpha * (D_v3 - D_v6) [linear decrease in dispatches]
- M(alpha) = M_v3 + alpha * (M_v6 - M_v3) [linear increase in memory]
- T(alpha) = T_base + c * D(alpha) + M(alpha) / BW
- dT/dalpha = -c * (D_v3 - D_v6) + (M_v6 - M_v3) / BW

If dT/dalpha > 0 (bandwidth cost of precomputing exceeds dispatch savings), then
T is minimized at alpha=0 (all factored = v3). If dT/dalpha < 0, minimized at alpha=1
(all precomputed = v6). Since dT/dalpha is constant (independent of alpha), there is
no interior minimum. The only question is the sign.

This linear argument is not explicitly in PAPER.md but is implied by the data. The
argument holds only if c_factored and c_precomp are constant per dispatch and M/BW is
linear in M -- both reasonable assumptions for the current architecture.

**One exception the paper does not consider:** If different module types have different
dispatch costs (they do: c_factored=0.0188 vs c_precomp=0.00606), then the optimal
strategy is to precompute the modules with the highest dispatch-to-bandwidth ratio first.
Attention modules have small deltas (32.77 MB/layer) and high factored cost (7 dispatches
at 0.0188 ms each = 0.132 ms/layer), so their dispatch-to-bandwidth ratio is
0.132 / (32.77/273000*1000) = 0.132 / 0.12 = 1.1. MLP modules have large deltas
(~139 MB/layer in full precomp) and 6 factored dispatches (0.113 ms/layer), so their
ratio is 0.113 / (139/273000*1000) = 0.113 / 0.509 = 0.22. Since attention has a
higher ratio (1.1 > 1.0), precomputing attention saves more per unit of bandwidth
cost. Since MLP has ratio 0.22 < 1.0, precomputing MLP costs MORE bandwidth than it
saves in dispatch. This confirms v6 (attention-only precomp) as the speed-optimal
configuration. The paper arrives at this conclusion empirically but does not present
this ratio analysis.

**Verdict on impossibility structure: correct and insightful. The "no intermediate
optimum" claim is formally sound under the additive model.**

## Novelty Assessment

The experiment is not novel in isolation -- it tests a standard engineering partition
(some modules precomputed, others factored). The novelty lies in:

1. **Falsifying the max() speed model.** The prior v6.1 review assumed
   T = T_base + max(c*D, M/BW), which the v6.1 LEARNINGS.md explicitly recommended
   (line 104-106). This experiment proved the max model wrong and established the
   additive model. This is a genuine empirical contribution to the project's speed
   modeling.

2. **Proving the Pareto frontier is discrete, not continuous.** The finding that any
   hybrid pays both penalties is actionable: it eliminates an entire design space from
   future exploration.

**Prior art within this project:** The v6.1 LEARNINGS.md (line 79-82) explicitly
recommended v6.2 as "the direct next step" and predicted "~80 tok/s." The v6.1
review (line 236-237) recommended a bandwidth feasibility check. The v6.2 MATH.md
does include a bandwidth feasibility check (Section C.1) but uses the max() model,
which was the prevailing model at the time. The progression from v6.1's recommendation
to v6.2's execution to the falsification is clean scientific workflow.

## Honest Reporting Assessment

**Strengths:**
1. Kill is unambiguous and not retroactively redefined
2. Speed failure clearly analyzed with revised cost model
3. Comparison table (PAPER.md lines 50-56) includes all four configurations
4. Quality predictions confirmed; only speed failed
5. Impossibility structure correctly prevents future hybrid attempts
6. Limitations section (lines 118-121) acknowledges weak behavioral proxy,
   speed variance, post-hoc nature of additive model, and unexplained overhead

**Weaknesses:**
1. Memory prediction labeled "YES (below range, better)" when 2.30 < 2.5 lower bound
   is technically a miss, not a pass. Minor.
2. The ~1.7 ms unexplained overhead is hand-waved as "module type switching,
   framework overhead." This could be measured with Metal profiling but is not.
3. The additive model is derived from ONE new data point (v6.2). With four configurations
   (v3, v6, v6.1, v6.2), the additive model should be validated against all four.

Let me check the additive model against all four configurations:

For v3 (all factored, D=420, M=27 MB):
T = 5.81 + 0.0188*420 + 27/273000*1000 = 5.81 + 7.90 + 0.10 = 13.81 ms = 72.4 tok/s
Measured: 73.0 tok/s. Error: -0.8%. EXCELLENT.

For v6 (attn precomp, D=60 precomp, M=983 MB):
T = 5.81 + 0.00606*60 + 983/273000*1000 = 5.81 + 0.36 + 3.60 = 9.77 ms = 102.4 tok/s
Measured: 86.8 tok/s. Error: +18%. POOR.

For v6.1 (all precomp, D=120 precomp, M=4200 MB):
T = 5.81 + 0.00606*120 + 4200/273000*1000 = 5.81 + 0.73 + 15.38 = 21.92 ms = 45.6 tok/s
Measured: 42.1 tok/s. Error: +8%. MODERATE.

For v6.2 (hybrid, D_f=180, D_p=60, M=1010 MB):
T = 5.81 + 0.0188*180 + 0.00606*60 + 1010/273000*1000 = 5.81 + 3.38 + 0.36 + 3.70 = 13.25 ms = 75.5 tok/s
Measured: 67.4 tok/s. Error: +12%. MODERATE-POOR.

**The additive model overestimates speed (underestimates time) for v6 by 18%.** This
suggests an additional overhead that scales with precomputed delta size beyond pure
bandwidth transfer -- possibly MLX eval() calls on the materialized deltas, or
memory allocation overhead for the ConcatDeltaLinear modules. The additive model is
BETTER than the max model but is NOT a complete explanation.

PAPER.md presents the additive model as the correct replacement for the max model
(line 77-87) but does not validate it against v6 or v6.1. This is a gap in the
analysis. The additive model fits v3 and v6.1 reasonably but misses v6 by 18%
and v6.2 by 12%, suggesting a THIRD cost component that scales with delta size in
a nonlinear way (possibly sublinear or logarithmic).

## NotebookLM Findings

Skipping NotebookLM automation. The mathematical content is straightforward enough
for direct review -- the core proofs are basic linear algebra (Theorems 1-3) and the
speed model (Theorem 4) is an empirical fit. The interesting analysis is physical
(memory bandwidth vs dispatch overhead), not mathematical.

## Macro-Scale Risks (advisory)

1. **The additive cost model will get worse at scale.** Larger models have larger
   d_int (MLP intermediate dimensions), making the bandwidth penalty for any
   precomputation steeper. The dispatch overhead may decrease (more work per
   dispatch at larger dimensions), shifting the Pareto frontier. The specific
   numerical predictions (c_factored, c_precomp) will not transfer.

2. **Multi-adapter composition multiplies the problem.** With N active adapters,
   precomputed delta memory is N * 983 MB. At N=24, this is 23.6 GB for attention
   alone. The room model (W_combined = sum of delta_W_i) collapses this to O(1) but
   requires a different composition strategy.

3. **The 73 tok/s "quality-optimal" point is hardware-specific.** On hardware with
   higher memory bandwidth (e.g., future Apple Silicon or GPUs), the bandwidth
   penalty shrinks and full precomputation may become viable. On hardware with
   lower bandwidth, even attention-only precomputation may be too expensive.

## Verdict

**PROCEED** (as a killed experiment with valid findings)

The experiment was correctly killed. The kill is honest, the criteria were not
retroactively redefined, and the speed failure is correctly attributed to the
falsification of the max() speed model. The quality predictions (Theorems 1-3) are
confirmed precisely, demonstrating that the hybrid approach is mathematically correct
but operationally suboptimal.

The key finding -- bandwidth and dispatch costs are additive, not max -- is a genuine
and useful contribution that eliminates an entire design space (hybrid partitions) from
future exploration. The Pareto frontier analysis is sound: the optimal serving strategy
is either all-factored (v3, quality) or attention-only-precomputed (v6, speed).

**Three non-blocking issues for the record:**

1. **Additive model not validated against all configurations.** The additive model
   overestimates v6 speed by 18% (predicts 102.4, measured 86.8), suggesting there is
   additional overhead beyond simple bandwidth + dispatch summation. PAPER.md should
   present the additive model's fit against all four data points, not just v6.2. The
   model is still better than max(), but it is not the complete picture -- a third
   cost component (possibly nonlinear in delta size) is missing.

2. **Memory prediction labeled as PASS when outside predicted range.** 2.30 GB < 2.5 GB
   lower bound. This is a favorable miss but still a prediction error. The table should
   say "PARTIAL (below range)" not "YES (below range, better)."

3. **"No intermediate optimum" claim is stronger than the evidence.** Only one hybrid
   partition was tested. The additive model predicts no intermediate optimum, but the
   additive model itself has 12-18% error on some configurations. A partition that
   precomputes only QKV (not O) with smaller deltas might theoretically outperform v6
   on the speed axis. This is unlikely to change the qualitative conclusion but should
   be acknowledged as untested.

The finding is ready to record as killed with the impossibility structure:
"Bandwidth and dispatch costs are additive on Apple Silicon, not pipelined. Any hybrid
configuration that combines factored dispatches with materialized precomputed deltas pays
both penalties. The per-token time is T = T_base + c_f * D_f + c_p * D_p + M / BW (plus
~1.5 ms of unexplained overhead). The Pareto frontier has two discrete points: all-factored
(v3: 73 tok/s, full quality) and attention-only-precomputed (v6: 86.8 tok/s, reduced quality).
No intermediate partition improves on both simultaneously."
