# Peer Review: Cuckoo Collision-Free Routing

## NotebookLM Findings

Skipped -- the experiment is straightforward enough to review from source.

## Mathematical Soundness

### Derivations

The math in MATH.md is correct at the level presented. The dual-hash blending formula is sound: `p_blend = (1 - alpha) * p1 + alpha * p2` is a valid convex combination of probability distributions and preserves the simplex. The sigmoid-based soft eviction is differentiable and gradient-friendly. FLOPs accounting is reasonable.

### Hidden Assumptions

1. **The cuckoo hashing analogy is structurally misleading.** In real cuckoo hashing, eviction is a *sequential* process: key A displaces key B from slot 1, key B then moves to its alternative slot 2, potentially displacing key C, etc. The chain depth is the length of this cascade. In this implementation, there is no cascade. There is a single soft blend of two distributions. Calling this "eviction" and measuring "chain depth" imports terminology that overstates the mechanism's sophistication. What is actually implemented is: **a learned mixture of two softmax routers with an input-dependent mixing coefficient**. This is a well-understood construction.

2. **"Max chain depth 0.24" is a misleading metric.** The chain depth is defined as `evicted + double_evicted` where both are binary indicators, so per-token depth is always in {0, 1, 2}. The reported "max chain depth 0.24" is actually the *mean* chain depth across the batch with the highest mean -- not the true maximum depth of any individual token. The true max depth observed is either 0, 1, or 2. The kill criterion (depth > 3) can never be triggered because the mechanism is architecturally bounded at 2. This makes KC2 unfalsifiable as stated.

3. **The eviction chain depth formula in MATH.md has a redundancy:**
   ```
   depth(x) = I[conf_h1 < tau] + I[conf_h1 < tau] * I[conf_h2 < tau]
   ```
   This computes `I[h1 low] + I[h1 low AND h2 low]`, which equals 1 when only h1 is low, and 2 when both are low. This is correct but trivially bounded by 2. The "well below the kill threshold of 3" claim is vacuously true -- the mechanism cannot produce depth > 2 by construction.

### Tau Not Training

The paper acknowledges that `_raw_tau` was stored as a raw `mx.array` rather than a registered `nn.Module` parameter. Examining the code confirms this: line 70 of the implementation uses `self._raw_tau = mx.array([-0.85])`. In MLX, parameters must go through `nn.Module` registration to be included in `model.trainable_parameters()`. This means the experiment tested a **fixed-threshold mixture of two routers**, not the intended adaptive-threshold mechanism. The claimed "learned collision threshold" was never learned.

## Novelty Assessment

### Prior Art

The closest published work is **not** Pagh & Rodler (2004) -- that is a data structure paper with no connection to neural routing. The actual prior art is:

1. **Multi-head routing / auxiliary routers in MoE.** Using multiple routing functions and combining them is explored in several MoE papers. The "expert choice" line of work (Zhou et al., 2022) uses multiple perspectives on routing. Hash-based routing appears in THOR (Roller et al., 2021).

2. **Mixture of softmax (MoS)** (Yang et al., 2018): Uses multiple softmax functions with learned mixing, which is exactly what this mechanism implements -- two softmax distributions blended with an input-dependent coefficient.

3. **Input-dependent gating of router outputs** is standard in attention mechanisms (multi-head attention blends multiple projections with input-dependent weights).

### Delta Over Existing Work

The mechanism is a standard **input-dependent mixture of two linear routers** dressed in cuckoo hashing terminology. The "collision detection" is just thresholding the max softmax probability. The "eviction" is just increasing the weight of the second router. The framing is creative but the mechanism has no structural property unique to cuckoo hashing (no actual eviction chains, no O(1) guarantees, no load factor constraints).

The 57.4% collision rate finding (softmax score ties) is genuinely interesting and well-measured. However, this is an observation about softmax routing, not a contribution of the cuckoo mechanism.

## Experimental Design

### Does It Test the Hypothesis?

The hypothesis is that cuckoo hashing provides collision-free routing. The experiment tests whether a mixture-of-two-routers matches softmax quality. These are different claims. The mechanism provides "collision resolution" only in the sense that a second opinion is available -- but this is true of any ensemble of routers.

### Controls

The softmax baseline is appropriate. The use of 3 seeds is minimal but adequate for a micro experiment. The per-seed breakdown shows the result is within noise (seed 123: -0.67%, seed 42: +0.77%), which the paper honestly reports.

### Collision Rate Measurement

The collision rate measurement in `measure_softmax_collision_rate` is correctly implemented: it accesses `layer.capsule_pool.router(h_normed)` on the softmax baseline model, where `router` is an `nn.Linear` returning raw logits, then applies softmax. The gap < 0.05 threshold is reasonable.

However, there is a subtle issue: the function runs the forward pass through the model layer-by-layer (line 82: `x = layer(x)`), meaning deeper layers see transformed representations. The collision rate is measured per-layer but reported as a single aggregate. The per-layer breakdown would be more informative -- early layers likely have higher collision rates than later layers where representations are more specialized.

### The Throughput Problem

The -63.6% throughput hit is dismissed as an "implementation artifact" but this is significant. The theoretical overhead is 2x routing FLOPs, which should be ~6.5% of total compute (as correctly computed in MATH.md). A -63.6% throughput hit suggests the implementation has serious inefficiency beyond the 2x routing cost -- likely the `mx.eval(alpha)` call on line 140 of the router, which forces synchronous evaluation mid-forward-pass and breaks MLX's lazy evaluation graph. This is an implementation bug, not just an artifact.

## Hypothesis Graph Consistency

The experiment matches HYPOTHESES.yml node `exp_cuckoo_collision_free_routing` with kill criteria:
- KC1: >2% worse than softmax -- testable and tested
- KC2: eviction chain length >3 -- **unfalsifiable** (bounded at 2 by construction)

KC2 should be revised to a meaningful threshold, e.g., "eviction rate > 50%" or "h1 and h2 converge (cosine > 0.9)".

## Macro-Scale Risks (advisory)

1. **h1/h2 convergence during extended training.** Two linear projections from the same input, trained with the same loss signal, will tend to align. The paper acknowledges this but proposes no mitigation. At macro scale with thousands of steps, orthogonality regularization is likely required.

2. **The throughput regression is real.** Even with the `mx.eval` bug fixed, two full router forward passes plus blending plus sigmoid is non-trivial overhead. At macro scale where router latency matters for expert parallelism, this needs careful benchmarking.

3. **The mechanism adds complexity without demonstrated benefit.** At +0.15% (within noise), there is no evidence this helps. The honest conclusion is that the mechanism is neutral, which means the added complexity (second router, threshold, blending) is unjustified.

## Verdict

**PROCEED** (with caveats)

The experiment is honest, well-documented, and correctly identifies its own limitations. Both kill criteria technically pass. The 57.4% collision rate finding is genuinely useful. The mechanism is mathematically sound even if the cuckoo analogy is overstated.

However, the following should be noted in FINDINGS.md and future work:

1. **KC2 is vacuously satisfied** -- the chain depth is bounded at 2 by construction and can never exceed the kill threshold of 3. This kill criterion should be replaced with something falsifiable for any follow-up.

2. **The mechanism is a learned mixture of two softmax routers**, not cuckoo hashing. The terminology should be adjusted to avoid overclaiming novelty from a data-structure analogy that does not structurally transfer.

3. **Tau never trained** due to an implementation bug (`mx.array` vs registered parameter). The experiment tested a fixed-threshold variant, not the intended adaptive mechanism. Any follow-up must fix this.

4. **The `mx.eval(alpha)` call in the router forward pass** (line 140) likely causes the -63.6% throughput regression by breaking lazy evaluation. Remove it and re-benchmark.

5. **The +0.15% result is noise.** The honest conclusion is that dual-router blending provides no quality benefit at micro scale, consistent with all routing being equivalent at G=8. This is not a negative result -- it confirms the mechanism is at least not harmful.

The experiment advances understanding (collision rate measurement, dual-router feasibility) without wasting resources on a dead end. Proceed to FINDINGS.md notation, but do not prioritize macro follow-up unless a compelling theoretical argument emerges for why collision resolution matters more at scale.
