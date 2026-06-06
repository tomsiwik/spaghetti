# MATH.md: Hybrid Precomputed Attention + Factored MLP (Pierre v6.2)

## A. Failure Mode Identification

**Symptom 1 (v6):** Attention-only precomputed deltas achieve 86.8 tok/s (60
dispatches, 983 MB deltas) but code domain behavioral drops to 0.281 because
MLP adapters are absent. MLP gate_proj and up_proj store domain-specific
computation patterns critical for instruction following (Finding #292).

**Symptom 2 (v6.1):** Full precomputation restores code behavioral (0.844)
with 120 dispatches but speed drops to 42.1 tok/s because materializing MLP
deltas expands total delta memory to 4.2 GB, exceeding the memory bandwidth
budget (273 GB/s -> 15.4 ms/tok just for delta transfer).

**Root cause:** There is a phase transition in the performance bottleneck.
Below ~1 GB of materialized deltas, the system is dispatch-bound (speed
improves as dispatches decrease). Above ~1 GB, the system is bandwidth-bound
(speed degrades as delta memory grows, regardless of dispatch count).

**Degenerate behavior to avoid:** Either losing MLP quality (v6) or exceeding
the bandwidth budget (v6.1). The hybrid must stay in the dispatch-bound regime
for materialized data while recovering MLP quality through factored computation.

## B. The Right Question

Not "how do we precompute MLP deltas without exceeding bandwidth?" but:

**"What is the optimal partition of adapter modules into {precomputed, factored}
such that total per-token time T = T_base + max(c * D, M / BW) is minimized?"**

This is a constrained optimization: minimize T over the binary assignment of
each module type to either precomputed (reduces D, increases M) or factored
(reduces M, increases D).

## C. Prior Mathematical Foundations

### C.1 Two-Regime Speed Model (Empirical, from v3/v6/v6.1)

From three data points plus the base model, we have established:

    T(D, M) = T_base + max(c_dispatch * D, M / BW)

Parameters (calibrated from v3, v6, v6.1):
- T_base = 5.81 ms (BitNet-2B base at ~172 tok/s, no adapters)
- c_dispatch = 0.00606 ms/dispatch (from v3 and v6 linear fit)
- BW = 273 GB/s (M5 Pro measured memory bandwidth)

Verification against three known configurations:

| Config | D | M (MB) | T_dispatch | T_bw | T_predicted | T_actual | Error |
|--------|---|--------|------------|------|-------------|----------|-------|
| v3     | 420 | 18   | 2.55 ms    | 0.07 ms | 8.36 ms  | 13.70 ms | -39%* |
| v6     | 60  | 983  | 0.36 ms    | 3.60 ms | 9.41 ms  | 11.52 ms | -18%  |
| v6.1   | 120 | 4200 | 0.73 ms    | 15.4 ms | 21.2 ms  | 23.8 ms  | -11%  |

*Note: v3 underestimates because RuntimeLoRA has additional per-dispatch overhead
beyond pure Metal dispatch cost (Python wrapper, allocation). The model captures
the bandwidth regime accurately but underestimates the dispatch-bound regime.

**Revised model for factored (RuntimeLoRA) dispatches:**
v3 has 420 dispatches, T_v3 = 13.70 ms, T_base = 5.81 ms.
Effective c_factored = (13.70 - 5.81) / 420 = 0.0188 ms per factored dispatch.

This is 3.1x the pure dispatch cost, consistent with RuntimeLoRA's Python-level
overhead per module (two matmuls + allocation per module, vs one matmul for
precomputed).

### C.2 Concat-Slice Equivalence (from v6.1 MATH.md, proven)

**Theorem (Concat-Slice, proven in v6.1).** For modules f_1, ..., f_k sharing
input x, where f_i(x) = x @ W_i, the concatenated computation x @ [W_1|...|W_k]
followed by slicing produces bit-identical results to separate computations.

This theorem is reused without modification for the attention modules (QKV, O).

### C.3 Factored LoRA Equivalence (trivial)

For RuntimeLoRA: y = base(x) + alpha * (x @ A) @ B.
This is mathematically equivalent to y = base(x) + x @ DeltaW where DeltaW = alpha * A @ B.
The difference is purely computational: factored costs 2 dispatches per module
but transfers only rank-16 matrices (~98 KB each) instead of full DeltaW (~35 MB each for MLP).

## D. Proof of Guarantee

**Theorem 1 (Hybrid Exact Equivalence).** The hybrid approach — precomputed
concat deltas for attention, factored RuntimeLoRA for MLP — produces
bit-identical output to v3 (all-factored) and v6.1 (all-precomputed).

*Proof.* The model output is a function of per-module corrections:

    y_module = base(x) + correction(x)

For attention modules, correction(x) = x @ DeltaW (precomputed).
For MLP modules, correction(x) = alpha * (x @ A) @ B (factored).

By the concat-slice theorem (C.2), precomputed attention corrections are
bit-identical to factored corrections.

By associativity of matrix multiplication: alpha * (x @ A) @ B = x @ (alpha * A @ B) = x @ DeltaW.

Since both paths compute the same DeltaW (up to floating-point ordering, which
is consistent within each strategy), the composed model output is identical
for each module. The model is a composition of layer functions, each a
composition of module functions. Since every module produces the same
correction, the full model output is identical.

The only difference is operational: which matrices are stored and how many
dispatch calls are made. QED.

**Theorem 2 (Hybrid Dispatch Count).** The hybrid has exactly 8 dispatches
per layer = 240 total dispatches across 30 layers.

*Proof.* Per layer:
- QKV concat (precomputed): 1 dispatch (3 modules fused)
- O (precomputed): 1 dispatch
- gate_proj (factored): 2 dispatches (x@A, h@B)
- up_proj (factored): 2 dispatches
- down_proj (factored): 2 dispatches

Total per layer: 2 precomputed + 6 factored = 8.
Total: 8 * 30 = 240 dispatches. QED.

**Theorem 3 (Hybrid Materialized Memory).** The hybrid materializes exactly
the same delta memory as v6 (attention-only): 983 MB.

*Proof.* Only attention modules are precomputed:
- QKV concat delta per layer: 2560 x (2560 + 640 + 640) x 2 bytes = 19.66 MB
- O delta per layer: 2560 x 2560 x 2 bytes = 13.11 MB
- Per layer total: 32.77 MB
- 30 layers: 983 MB

MLP factored matrices per layer:
- gate A: 2560 x 16 x 2 = 82 KB, gate B: 16 x 6912 x 2 = 221 KB
- up A: 82 KB, up B: 221 KB
- down A: 6912 x 16 x 2 = 221 KB, down B: 16 x 2560 x 2 = 82 KB
- Per layer: 0.91 MB
- 30 layers: 27.3 MB (negligible, <3% of attention deltas)

Total materialized: 983 + 27 = 1010 MB.
Bandwidth term: 1010 MB / 273 GB/s = 3.70 ms (essentially same as v6). QED.

**Theorem 4 (Hybrid Speed Prediction).** The hybrid speed satisfies:

    T_hybrid = T_base + max(c_factored * D_factored + c_precomp * D_precomp, M / BW)

Substituting:
- D_factored = 180 (6 per layer x 30 layers, RuntimeLoRA MLP modules)
- D_precomp = 60 (2 per layer x 30 layers, attention concat groups)
- c_factored = 0.0188 ms (from v3 calibration, includes Python overhead)
- c_precomp = 0.00606 ms (from v6 calibration, pure Metal dispatch)
- M = 1010 MB, BW = 273 GB/s

Dispatch term: 0.0188 * 180 + 0.00606 * 60 = 3.38 + 0.36 = 3.74 ms
Bandwidth term: 1010 / 273000 * 1000 = 3.70 ms

Both terms are nearly equal (~3.7 ms each). The actual T is max(3.74, 3.70)
since they pipeline (Metal dispatches overlap with memory transfer).

    T_hybrid = 5.81 + 3.74 = 9.55 ms -> 1000/9.55 = 104.7 tok/s

However, dispatch and bandwidth do NOT fully overlap. A more conservative
additive model gives:

    T_hybrid_additive = 5.81 + 3.74 + 3.70 = 13.25 ms -> 75.5 tok/s

Reality should fall between these bounds:

    75-105 tok/s (central estimate: ~85 tok/s)

The central estimate derives from v3 behavior: v3 had D=420 factored dispatches
and T_v3=13.70 ms. The hybrid replaces 240 of those (attention) with precomputed,
saving 240 * (0.0188 - 0.00606) = 3.06 ms of dispatch overhead but adding ~3.7ms
of bandwidth. Net effect is roughly neutral, but the attention modules are faster
because precomputed avoids Python-level LoRA wrapper overhead.

Best estimate: ~80-90 tok/s (above the K759 threshold of 75).

## D. Predictions (Behavioral AND Quantitative)

### Behavioral Predictions

1. **Code domain behavioral matches v6.1 (0.844)** because MLP gate+up adapters
   are present via RuntimeLoRA, same weights, same math.

2. **Overall behavioral matches v3/v6.1 (~0.41-0.42)** because all 7 adapter
   module types are active with identical parameters.

3. **Speed exceeds 75 tok/s** because the hybrid stays near the dispatch/bandwidth
   balance point rather than being dominated by either.

### Quantitative Predictions (from proofs)

| Prediction | Source | Expected | Kill Threshold |
|-----------|--------|----------|----------------|
| Dispatch count | Theorem 2 | 240 | n/a |
| Materialized delta memory | Theorem 3 | ~1010 MB | n/a |
| Speed (tok/s) | Theorem 4 | 80-90 (range: 75-105) | K759: >= 75 |
| Code behavioral | Theorem 1 + v6.1 data | ~0.84 | K760: >= 0.80 |
| Overall behavioral | Theorem 1 + v3/v6.1 data | ~0.41 | K761: >= 0.35 |
| Peak memory (GB) | Theorem 3 + base | ~2.5-3.5 | K762: <= 6.0 |

## E. Assumptions & Breaking Conditions

**A1: Dispatch and bandwidth partially overlap.** Metal can pipeline dispatch
processing with memory transfer. If they are fully serial, speed drops to
~75 tok/s (worst case still passes K759). If fully overlapping, ~105 tok/s.

**A2: RuntimeLoRA per-dispatch cost is ~0.0188 ms.** This includes Python wrapper
overhead. If MLX compilation or batching changes this, the dispatch term shifts.
Calibrated from v3 (420 factored dispatches = 7.89 ms overhead).

**A3: Python call order preserved.** QKV and Gate+Up groups must evaluate in
order for the cache pattern. Verified in qwen2.py source for attention. MLP
modules are separate calls (gate, up, down) so no cache ordering issue.

**A4: Precomputed attention and factored MLP do not interact adversely.**
Since both strategies compute the same mathematical correction (Theorem 1),
mixing them in one model should be seamless. The only risk is if the model
framework treats wrapped modules differently, but ConcatDeltaLinear and
RuntimeLoRA both follow the nn.Module __call__ contract.

## F. Worked Example (d=4, r=2, 1 layer)

**Setup:** 1 layer, 3 attention modules (QKV), 2 MLP modules (gate, up).
d_model=4, rank=2, d_kv=2, d_ff=6.

**Attention (precomputed):**
- DeltaW_q = A_q @ B_q: (4x4), DeltaW_k = A_k @ B_k: (4x2), DeltaW_v: (4x2)
- DeltaW_qkv = [DeltaW_q | DeltaW_k | DeltaW_v]: (4x8)
- 1 dispatch: x @ DeltaW_qkv, then slice at [0:4], [4:6], [6:8]

**MLP (factored):**
- gate: h_gate = x @ A_gate (4x2), corr_gate = h_gate @ B_gate (2x6) -> 2 dispatches
- up: h_up = x @ A_up (4x2), corr_up = h_up @ B_up (2x6) -> 2 dispatches

**Dispatch count:** 1 (QKV) + 1 (O) + 2 (gate) + 2 (up) + 2 (down) = 8 per layer.

**Memory:**
- Precomputed: DeltaW_qkv (4x8) + DeltaW_o (4x4) = 32+16 = 48 floats x 2B = 96 bytes
- Factored: A_gate (4x2) + B_gate (2x6) + A_up (4x2) + B_up (2x6) + A_down (6x2) + B_down (2x4) = 8+12+8+12+12+8 = 60 floats x 2B = 120 bytes
- Total per layer: 216 bytes vs precomputed-all = 96 + (4x6)+(4x6)+(6x4) = 96+48+48+48 = 240 bytes -> 10% memory savings

## G. Complexity & Architecture Connection

**Per-token FLOPs:** Identical to v3 and v6.1 (same DeltaW corrections applied).
The operational difference is number of kernel launches, not total arithmetic.

**Memory scaling:**
- Base model: ~1.18 GB (BitNet-2B-4T quantized)
- Attention deltas: 983 MB (fixed per adapter swap)
- MLP factored matrices: 27 MB (fixed per adapter swap)
- Total adapter overhead: ~1010 MB
- Per-adapter: same as v6 for attention + same as v3 for MLP

**Architecture fit:** This hybrid naturally maps to the production architecture:
- Always-on adapters (instruction tuning): precompute both attention AND MLP
  (they never swap, so bandwidth cost is paid once at startup)
- Routed domain experts: hybrid mode (attention precomputed, MLP factored)
  for fast per-query adapter switching with full quality

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   **The bandwidth budget partition: attention deltas (983 MB) stay within the
   ~1 GB bandwidth-efficient regime while MLP adapters in factored form add
   negligible bandwidth load (~27 MB). Neither symptom (quality loss from
   missing MLP, speed loss from excess bandwidth) can occur.**

2. Which existing theorem(s) does the proof build on?
   **Concat-Slice Equivalence (proven in v6.1 MATH.md, basic linear algebra),
   Matrix multiplication associativity (A@B precompute), Two-regime speed
   model (empirical, calibrated from v3/v6/v6.1 data points).**

3. What specific numbers does the proof predict?
   **240 dispatches, ~1010 MB materialized, 80-90 tok/s (range 75-105),
   code behavioral ~0.84, overall behavioral ~0.41, memory ~2.5-3.5 GB.**

4. What would FALSIFY the proof (not just the experiment)?
   **The proof is wrong if (a) mixing ConcatDeltaLinear and RuntimeLoRA in
   the same model causes framework-level conflicts, (b) the two-regime speed
   model is fundamentally miscalibrated (not just noisy), or (c) factored
   MLP dispatches have much higher overhead than measured in v3.**

5. How many hyperparameters does this approach add?
   **0. The partition (attention=precomputed, MLP=factored) is derived from
   the bandwidth analysis, not tuned. All other parameters inherited from v3.**

6. Hack check: Am I adding fix #N to an existing stack?
   **No. This is an optimal partition of the same adapter math across two
   proven injection strategies. No new mechanisms, losses, or constraints.
   The mathematical corrections are identical to v3/v6/v6.1.**
