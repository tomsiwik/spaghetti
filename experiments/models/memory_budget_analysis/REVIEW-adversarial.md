# Peer Review: Memory Budget Analysis

## Verdict: PROCEED

This is a well-executed capacity planning experiment. The math is correct, the
measurements are consistent with theory (4.4% overhead is plausible for Metal
allocator alignment), and the kill criterion is passed with 8.5x margin. The
findings are useful for the SOLE architecture. However, several measurement
confounds and framing issues should be noted for intellectual honesty.

---

## Findings

### Finding 1: Residual memory between Phase 2 iterations (minor confound)

In Phase 2 (adapter scaling), `mem_before_mb` is 0.0 for N=10 but 45.1-45.4 MB
for N=50, N=100, N=500. This means ~45 MB of the previous iteration's allocations
survive cleanup (`gc.collect()` + `mx.clear_cache()`). The per-adapter cost IS
computed as `(mem_after - mem_before) / N`, so the residual is correctly subtracted
out and does not corrupt the per-adapter figure. However, it demonstrates that MLX
memory cleanup is imperfect, which is relevant to the claimed N_max -- if cleanup
between adapter swaps leaks 45 MB each time, production serving could accumulate
leaks.

**Impact:** Low. Per-adapter cost is correctly measured. But the 45 MB residual
per cleanup cycle is a production concern worth noting.

### Finding 2: N_max=853 is extrapolated, not measured (moderate concern)

The experiment measures up to N=500 (22.7 GB actual). N_max=853 is computed by
linear extrapolation: `available_budget / per_adapter_cost`. The experiment never
actually allocates 853 adapters to verify that MLX can handle it. Key risks at
high N that extrapolation cannot capture:

- Metal allocator behavior at >35 GB active memory (near system limits)
- Per-allocation metadata scaling (853 adapters x 210 targets x 2 matrices = 358K
  separate tensor allocations)
- System memory pressure triggering OS-level memory management

The 100 MB "safety margin" in `phase_practical_max` (line 453) is ad hoc and
unvalidated. At 45.2 MB per adapter, 100 MB is only 2.2 adapters of margin.

**Recommendation:** Run one test at N=800 (or as close to N_max as safe) to
validate that linear extrapolation holds near the boundary. Alternatively,
acknowledge N_max=853 as a theoretical upper bound, not a measured capacity.

### Finding 3: Forward pass test is incomplete (moderate concern)

Phase 3 tests LoRA computation on a SINGLE projection (q_proj, layer 0) for each
of k active adapters. In production, runtime LoRA applies to all 210 target layers.
The paper claims "Forward pass peak memory adds only ~25 MB above steady state"
but this is measured with 1/210th of the actual LoRA computation.

The ~25 MB peak overhead comes from `peak_mb - active_mb` during a forward pass
that includes the full base model forward (all 30 layers) plus a trivial LoRA
overlay on one layer. The LoRA intermediate tensors for a single projection are:
`(1, 9, 2560) @ (2560, 16) = (1, 9, 16)` -- 288 bytes. At full scale with 210
targets, intermediates would be 210x larger but still tiny (<100 KB). So the
conclusion is likely correct, but the measurement does not support it.

Additionally, the `lora_overhead_mb` values in results.json are **negative**
(-0.3, -1.2, -0.4 MB), meaning MLX freed cache during the LoRA computation. The
"25 MB" figure comes from peak-vs-active, not from the lora overhead metric. The
paper does not mention the negative values.

**Recommendation:** Either run the LoRA computation across all 210 targets to
validate the overhead claim, or reframe as "theoretical overhead is negligible
because intermediate tensors are O(batch * seq * rank) which is << adapter storage."

### Finding 4: Routing heads allocated with mx.zeros (minor concern)

Adapter matrices are allocated with `mx.random.normal` to prevent zero-page
deduplication (line 272-273, correctly noted in code comments). Routing heads are
allocated with `mx.zeros` (lines 288-291). If MLX or Metal optimizes zero-filled
buffers (e.g., copy-on-write, lazy allocation), the measured per-head cost of
82 KB could be an undercount.

However, the measured 82 KB matches the theoretical value exactly (41,009 params *
2 bytes = 82,018 bytes), suggesting MLX does not zero-page-dedup at this scale.
Risk is low but the inconsistency in methodology is worth noting.

### Finding 5: Platform is 52 GB, not 48 GB (framing issue)

Results.json shows `total_memory_gb: 51.5`. The experiment frames everything around
"48 GB" (hypothesis, kill criteria, section titles). The 40 GB usable budget
assumes 8 GB system reservation from a 48 GB machine, but the actual machine has
51.5 GB, meaning the true usable budget is ~43.5 GB (3.5 GB more than assumed).
This makes the results conservative.

The paper notes this in Limitation 4 ("Platform reports 52 GB") but the headline
framing throughout MATH.md and PAPER.md says "48 GB." This is honest (conservative)
but could confuse readers.

### Finding 6: Single measurements, no variance (acceptable for this type)

Every data point is a single measurement. There are no repeated runs, no confidence
intervals. For a memory profiling experiment (not a stochastic training experiment),
this is acceptable -- memory allocation is largely deterministic on MLX. The
consistency of per-adapter cost across N=10 (45.19), N=50 (44.38), N=100 (44.87),
N=500 (45.24) provides implicit replication showing stability (range: 44.38-45.24,
CV = 0.8%).

### Finding 7: Grassmannian A-matrix sharing ambiguity (MATH.md internal inconsistency)

MATH.md section (c) has a visible "Wait -- this is critical" moment where the
author self-corrects from "A is shared" to "each adapter has its own A_i." This
is correctly resolved: the final per-adapter cost of 43.3 MB includes both A and B.
However, the table in section "Total Memory Formula" still lists "bf16 B only
(shared A)" as 21.9 MB, and the N_max table includes a "bf16 B (shared A)" row
showing N_max=1,763. This format is not actually available in the current
architecture (Grassmannian gives each adapter a unique A). Including it in the
tables could mislead about actual capacity.

The experiment correctly uses the bf16 A+B format (43.3 MB theoretical, 45.2 MB
measured) for all empirical work.

### Finding 8: The math is correct

I verified the following calculations:

- Packed ternary: `ceil(out/4) * in` bytes per layer -- correct for 2-bit packing
- Per-adapter bf16: sum over 30 layers x 7 targets of `(in*16 + 16*out) * 2` bytes
  = 43.25 MB -- matches results.json
- KV cache: `2 * 30 * 8 * 80 * seq * 2` = 76,800 * seq bytes -- correct
- N_max formula: `(40000 - 1178.6 - 1.32 - KV - 10) / (43.25 + 0.082)` = 895
  at seq=256 -- matches
- Measured N_max: `(40000 - 1178.6 - 1.32 - 19.7 - fwd_overhead - 100) / (45.24 + 0.082)`
  = 853 -- matches

The 100 MB safety margin and the fwd_overhead term (which defaults to the
lora_overhead_mb from k=2, but that is -1.2 MB so it becomes 0) are the only
soft assumptions. The core arithmetic is sound.

---

## Mathematical Soundness

The derivations are straightforward arithmetic with correct tensor shape accounting.
No hidden distributional assumptions. The only mathematical concern is the
extrapolation beyond N=500 (Finding 2), which assumes perfect linear scaling
of memory with adapter count -- a reasonable but unverified assumption at high N.

## Novelty Assessment

This is a capacity planning analysis, not a novel mechanism. It provides useful
engineering data for the SOLE architecture. The S-LoRA reference is appropriate --
S-LoRA demonstrated concurrent serving of thousands of LoRA adapters on GPU; this
experiment validates the memory arithmetic for the same concept on Apple Silicon
with a ternary base model. The delta over S-LoRA is the specific platform (MLX +
M5 Pro) and base model (BitNet-2B-4T).

## Experimental Design

The four-phase design (theoretical, base measurement, scaling measurement, forward
pass peak) is logical and each phase builds on the previous. The use of
`mx.random.normal` to prevent zero-page optimization is a good methodological
choice. The cleanup between Phase 2 iterations is imperfect but the per-adapter
cost calculation correctly accounts for residuals.

The experiment does test what it claims: it measures how many adapters fit in
memory. The kill criterion (K1: <100 adapters) is honestly evaluated from measured
data, not hardcoded. The verdict flows from measured per-adapter cost through
a simple division.

## Macro-Scale Risks (advisory)

1. **Multi-request serving:** KV cache per concurrent request. At batch=8,
   seq=2048: 1.26 GB additional KV cache, reducing N_max by ~28 adapters.
   Manageable but worth measuring.

2. **Adapter loading latency:** 7.2s to allocate 500 adapters. Production
   serving needs lazy loading with an LRU cache, not all-at-once allocation.

3. **Memory fragmentation over time:** The 45 MB residual between cleanup
   cycles (Finding 1) could accumulate during long-running serving. Needs
   stress testing.

4. **358K tensor allocations:** At N=853, there are ~358,000 separate MLX
   tensors (853 x 210 x 2). MLX graph tracking overhead at this scale is
   unmeasured.

---

## Kill Criteria Assessment

**K1 (261): "< 100 adapters fit in 48GB" -- PASS (honestly evaluated)**

N_max=853 is derived from measured per-adapter cost of 45.24 MB (not hardcoded).
The measurement was taken at N=500 (the largest test) and extrapolated. Even using
the worst measured per-adapter cost (45.24 MB from N=500) with a generous safety
margin, N_max >> 100. The 8.5x margin above the kill threshold is robust.

**S1 (27): ">500 adapters fit in 48GB" -- PASS (honestly evaluated)**

N=500 was directly measured at 22.7 GB (59.6% of budget). This is an actual
measurement, not an extrapolation.
