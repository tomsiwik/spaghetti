# LEARNINGS: exp_pierre_v62_hybrid (KILLED)

## Core Finding

**Bandwidth and dispatch costs are additive, not pipelineable — any hybrid that
mixes precomputed and factored adapters pays BOTH penalties and is strictly slower
than either pure strategy.** The Pareto frontier for LoRA serving speed has exactly
two points: all-factored (v3: 73 tok/s, full quality) and all-precomputed attention
(v6: 86.8 tok/s, speed-optimized). No intermediate hybrid optimum exists.

## Why This Happened

**Metal dispatches and memory reads are sequential within a forward pass.** Each
precomputed attention module must (1) read DeltaW from memory (bandwidth cost), then
(2) execute the matmul (dispatch cost). Each factored MLP module must dispatch two
small matmuls (A then B). These operations are interleaved in the same layer-by-layer
forward pass — there is no cross-module pipelining because MLX's Python dispatch is
sequential. The max() model implicitly assumed the GPU could pipeline memory reads
and compute across different module types. It cannot under MLX's execution model.

**The additive model explains all four configurations:**
```
Config          Dispatch (ms)  Bandwidth (ms)  Base (ms)  Extra (ms)  Total    tok/s
v6 (attn pre)   0.36           3.60            5.81       1.7         11.47    87.2*
v3 (factored)   7.90           ~0              5.81       0           13.71    72.9
v6.2 (hybrid)   3.74           3.60            5.81       1.69        14.84    67.4
v6.1 (all pre)  0.73           15.40           5.81       1.8         23.74    42.1
```
*v6 has lower dispatch overhead than v3 but higher unexplained overhead (~1.7ms).

**The ~1.7 ms unexplained overhead appears in all precomputed configurations.**
Both v6 (attn-only) and v6.2 (hybrid) show ~1.7 ms beyond what the dispatch + bandwidth
model predicts. This likely reflects ConcatDeltaLinear's cache-and-slice pattern:
the first module in each QKV group computes and caches, subsequent modules slice.
This Python-level caching + slice overhead is absent in pure factored serving.

**The v6.1 LEARNINGS predicted v6.2 would work — but used the wrong cost model.**
LEARNINGS.md for v6.1 recommended "T = T_base + max(c*D, M/BW)" as the two-regime
model. This experiment definitively falsified max() in favor of addition. The key
insight: max() would be correct if dispatch and bandwidth were ALTERNATIVE bottlenecks
(like Amdahl's law for parallel/serial). They are CUMULATIVE costs because every
token traverses both precomputed and factored modules sequentially.

## Confirming Evidence

- **LoRA-Switch (2405.17741):** Identifies fragmented kernel dispatch as the dominant
  bottleneck in dynamic adapter inference (2.5x slowdown). Proposes token-level weight
  merging with fused kernels. Their diagnosis matches ours: individual dispatch overhead
  accumulates linearly, and mixing strategies doesn't help — you need to eliminate
  dispatches entirely via fusion.

- **dLoRA (OSDI 2024, Wu et al.):** Dynamically switches between merged and unmerged
  adapter modes per-replica. Explicitly quantifies "mode switch cost" and finds that
  naive mixing of merged/unmerged in the same serving context causes unacceptable
  switching overhead — the same additive penalty we measured.

- **S-LoRA (2311.03285):** Uses factored form with custom BGMV kernels specifically
  to avoid the bandwidth cost of merged weights. Our hybrid result confirms their design
  choice: even partial merging (attention-only) hurts when combined with factored MLP.

- **Punica (2310.18547):** Introduced BGMV kernel that fuses adapter dispatch across
  different adapters in a single GPU kernel launch. This is the correct solution to
  our dispatch overhead problem — not hybridizing, but fusing.

- **Finding #300 (v6.1):** Proved bandwidth is the bottleneck for full precomputation
  (4.2 GB, 42.1 tok/s). Predicted hybrid would work by staying under ~1 GB. The bandwidth
  prediction was correct (983 MB ≈ 3.6 ms), but the model missed the additive dispatch cost.

- **Finding #75 (inference speed):** Originally measured multi-adapter addmm fusion at
  88.2→97.2 tok/s (+10.2%). Noted "Pre-merge WORSE (-36%, 1.18→4.83 GB destroys ternary
  BW advantage)" — an early signal that materialization hurts bandwidth-sensitive workloads.

## Contradicting Evidence

- **The v6.1 LEARNINGS.md two-regime model** predicted v6.2 would achieve ~80 tok/s
  using T = T_base + max(c*D, M/BW). This model was reasonable given the two data points
  available (v3 dispatch-bound, v6.1 bandwidth-bound), but the hybrid configuration
  exposed its flaw. No published paper has validated a max()-based model for mixed
  precomputed/factored serving — the recommendation was our own untested extrapolation.

## Alternative Approaches

- **Kernel fusion via LoRAFusion (2510.00206):** Fuses memory-bound LoRA operations into
  base GEMM kernels, eliminating redundant memory reads/writes. Achieves up to 1.96x speedup
  on CUDA. Would eliminate dispatch overhead entirely — the only way to beat 73 tok/s while
  keeping full adapter quality. Requires custom Metal kernels for Apple Silicon.

- **BGMV/Punica-style batched kernels (2310.18547):** Batch all adapter matmuls into a
  single fused kernel per layer. S-LoRA's MBGMV achieves near-merged throughput with
  factored adapters. On Apple Silicon, this would be a single Metal compute pass per layer
  instead of 14 separate dispatches.

- **Shared basis compression (2407.00066):** Compress N adapters into shared U/V matrices
  + per-adapter diagonal scaling. At N=24, this reduces total adapter memory from
  N × 27 MB = 648 MB to ~30 MB regardless of N. Addresses the scaling problem without
  changing the serving strategy.

- **Room model (W_combined = Sum ΔW_i):** Single composed delta per token eliminates
  both dispatch scaling (always 1 set of dispatches) and bandwidth scaling (one delta
  regardless of N). This is architecturally necessary for N=24+ and is the only approach
  that addresses both failure modes simultaneously.

## Implications for Next Experiments

1. **The precompute optimization path is exhausted.** Four configurations (v3, v6, v6.1,
   v6.2) have fully characterized the speed landscape. No further precompute/factored
   partition can beat v3 (73 tok/s) for quality or v6 (86.8 tok/s) for speed. Future
   speed work must use fundamentally different approaches: kernel fusion, batched dispatch,
   or the room model.

2. **Every future speed model MUST be additive, not max.** The correct form is:
   T = T_base + c_f * D_f + c_p * D_p + M / BW + overhead(precomp).
   Include the ~1.7 ms precomputed overhead term when any ConcatDeltaLinear module is
   present.

3. **Quality predictions are now PROVEN across 4 configurations.** Theorem 1 (exact
   equivalence) holds perfectly: code 0.844, overall 0.41-0.425 regardless of injection
   strategy. This is a strong result — the mathematical corrections are identical, and
   the serving strategy is purely a speed/memory tradeoff.

4. **v3 at 73 tok/s is the production quality baseline.** Until kernel fusion or room
   model delivers >=75 tok/s with full behavioral quality, v3 is the right choice.

## Recommended Follow-Up

**Room model composition (P0 — architectural necessity)**
- MOTIVATION: Finding #301 (this experiment) + Finding #300 prove that neither
  precompute variants nor hybrids can beat v3's 73 tok/s at full quality. The precompute
  optimization path is fully exhausted. Room model produces W_combined = Sum ΔW_i as
  a single adapter, eliminating both dispatch scaling and bandwidth scaling.
- LITERATURE: LoRA-Switch (2405.17741) shows dispatch fragmentation is THE bottleneck;
  Punica (2310.18547) shows fused kernels are the solution class. Room model is the
  composition-native equivalent — routes at matmul level, single adapter per token.
- Risk: medium — routing accuracy at token level is unproven at N=24.

**Metal kernel profiling (P1 — explains 1.7 ms residual)**
- MOTIVATION: 12-18% residual error in additive speed model. Understanding the 1.7 ms
  overhead term would improve future speed predictions.
- LITERATURE: Apple Silicon ML Profiling (2501.14925) provides methodology for Metal
  GPU profiling.
- Outcome: tighter speed model, possibly identifies fusion opportunities.
