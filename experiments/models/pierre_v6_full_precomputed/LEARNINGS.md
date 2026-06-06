# LEARNINGS: exp_v6_full_precomputed (KILLED)

## Core Finding

**Memory bandwidth, not dispatch count, determines LoRA serving speed when delta
matrices exceed ~1 GB.** Full precomputation of all modules (QKV+MLP) materializes
4.2 GB of dense deltas — 230x more than factored form — making inference bandwidth-bound
at 42.1 tok/s (worse than v3's 73 tok/s at 3.5x more dispatches). The optimal boundary
is attention-only precomputation (~1 GB), keeping MLP in factored form.

## Why This Happened

**The bottleneck transition is a phase change, not a gradual shift.** v6 (attention-only,
983 MB deltas) achieved 86.8 tok/s because delta bandwidth (3.6 ms/tok) was below base
model compute (~5.8 ms/tok). v6.1 (full, 4.2 GB) crossed the threshold where delta
bandwidth (15.4 ms/tok) dominates. This is not a smooth degradation — the system flips
from compute-bound to bandwidth-bound once delta memory exceeds the per-token bandwidth
budget (~1.2 GB at 273 GB/s for 75 tok/s target).

**MLP dimensions drive the blowup.** The intermediate dimension d_ff=6912 is 2.7x
larger than d_model=2560. Gate+up precomputed delta is 70.8 MB/layer vs QKV's 19.7 MB
— 3.6x larger per group. MLP contributes 3.2 GB of the 4.2 GB total. This is structural:
transformer FFN dimensions scale faster than attention dimensions in most architectures.

**The speed model failed because it modeled the wrong resource.** Theorem 3 fit a
linear model T(D) = T_base + c*D from 2 data points, treating dispatches as the sole
cost variable. It predicted 84.2 tok/s but measured 42.1 tok/s. The model was valid in
the dispatch-bound regime (v3→v6) but breaks when the system crosses into bandwidth-bound
territory. A correct model needs at minimum: T(D, M) = T_base + c_dispatch*D + M/BW
where M is total delta memory and BW is memory bandwidth.

**Positive result confirmed: MLP adapters are essential for code.** Behavioral scores
match v3 exactly (0.419 overall, 0.844 code), confirming that gate/up/down adapters
carry code-domain instruction-following capability. v6's 0.281 code score was not
routing or composition failure — it was missing MLP corrections.

## Confirming Evidence

- **S-LoRA (2311.03285):** Foundational multi-LoRA serving system. Uses factored A/B
  form with custom CUDA kernels specifically because merged weights consume too much GPU
  memory. Their Unified Paging system manages adapter memory precisely because full
  materialization is impractical at scale. Our finding is the Apple Silicon equivalent.

- **Compress then Serve (2407.00066):** Joint compression of 1000+ LoRA adapters via
  shared basis + per-adapter scaling matrices. Achieves 1.6x throughput improvement by
  reducing adapter memory footprint. Core insight aligns with ours: adapter memory, not
  compute, is the bottleneck for multi-adapter serving.

- **EdgeLoRA (2507.01438):** Multi-tenant LoRA serving on edge devices (Jetson, RPi).
  Uses heterogeneous memory management with intelligent adapter caching because edge
  devices face the same bandwidth constraint we hit. Achieves 4x throughput via batch
  LoRA inference — keeping adapters in factored form and batching the small matmuls.

- **CaraServe (2401.11240):** Rank-aware scheduling for LoRA serving. Uses CPU-assisted
  prefill to overlap adapter loading with compute. The rank-awareness directly addresses
  our finding: higher-rank adapters (or equivalently, precomputed full-rank deltas)
  consume disproportionate bandwidth.

- **Orion (2603.06728):** ANE-native LoRA serving on Apple Silicon. LoRA matrices passed
  as IOSurface inputs (factored form), enabling hot-swap without recompilation. Even
  Apple's own approach keeps adapters factored — they do NOT precompute merged deltas.

- **Finding #292 (Pierre v6):** Attention-only precomputation at 983 MB achieved 86.8
  tok/s. Confirms the ~1 GB boundary empirically.

- **Finding #288 (Pierre v3):** Factored serving at 18 MB achieved 73 tok/s. The
  dispatch-bound baseline.

## Contradicting Evidence

- **LoRA original paper (2106.09685):** Recommends merging W + DeltaW for zero-overhead
  inference. This works for single-adapter serving where the merged weight replaces the
  base weight (no additional memory). Our case is different: we need multiple adapters
  simultaneously, so deltas must be stored separately from the base. The merge strategy
  does not apply to multi-adapter composition.

## Alternative Approaches

- **Hybrid precompute (our v6.2 plan):** Precompute attention (2 dispatches/layer,
  ~1 GB), keep MLP factored (6 dispatches/layer, ~12 MB). Predicted 8 dispatches/layer
  = 240 total, ~80 tok/s. This is the direct next step and requires no new research —
  just combining what v6 and v3 already proved.

- **Kernel fusion (AdaFuse, 2603.11873):** Fuse adapter computation with base model
  GEMM to eliminate separate dispatch overhead entirely. Would make the dispatch count
  irrelevant but requires custom Metal kernels on Apple Silicon.

- **Shared basis compression (2407.00066):** Compress N adapters into shared U/V
  matrices + per-adapter diagonal scaling. Reduces per-adapter footprint from O(d*r) to
  O(r). Applicable if we scale to 24+ simultaneous adapters where even factored form
  becomes bandwidth-constrained.

- **Profiling Apple Silicon LLM inference (2508.08531):** Quantitative characterization
  of memory bandwidth utilization for quantized LLM inference on Apple Silicon. Could
  provide tighter bandwidth models for our speed predictions.

## Implications for Next Experiments

1. **Every future MATH.md that materializes full-rank matrices MUST include a bandwidth
   feasibility check.** The calculation is trivial: total_bytes / BW_GB_s = ms_per_token.
   If this exceeds the per-token budget (1000/target_tok_s - base_compute_ms), kill the
   hypothesis on paper. This experiment could have been killed before writing code.

2. **The two-regime speed model should replace the linear dispatch model.** Speed is
   min(dispatch_limited, bandwidth_limited): T = T_base + max(c*D, M/BW). This predicts
   v6.1's failure correctly: M/BW = 15.4 ms >> c*120 = 0.7 ms.

3. **Multi-adapter composition multiplies the bandwidth problem.** If N=24 adapters are
   active simultaneously, even attention-only precomputation is 24 × 983 MB = 23.6 GB —
   impossible on 48 GB. The room model (W_combined = Σ ΔW_i) produces a single set of
   deltas, keeping memory at O(1) regardless of N. This is not a nice-to-have; it is
   architecturally necessary.

## Recommended Follow-Up

**Hybrid v6.2: precompute attention, factor MLP** (P0 — direct next step)
- MOTIVATION: Finding #300 (this experiment) proved optimal boundary is attention-only
- LITERATURE: S-LoRA (2311.03285) and Orion (2603.06728) both keep adapters factored;
  our v6 (Finding #292) proved attention precomputation works at 86.8 tok/s
- Predicted outcome: ~80 tok/s, full behavioral quality, 8 dispatches/layer
- Risk: low — combines two proven configurations, no new mechanisms

**Bandwidth-aware speed model** (P1 — improves future predictions)
- MOTIVATION: Theorem 3's 2x miss; reviewer noted bandwidth check should be pre-experiment
- LITERATURE: Apple Silicon LLM profiling (2508.08531) provides bandwidth characterization
- Outcome: closed-form speed prediction for any precompute configuration
