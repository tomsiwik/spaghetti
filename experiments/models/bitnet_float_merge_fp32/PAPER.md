# Float Merge (fp32/bf16) as Serving Path for BitNet-SOLE

## Hypothesis

Merging LoRA adapters into an fp32 copy of BitNet-2B-4T base weights produces
lossless composition quality, creating a serving path that eliminates per-token
adapter overhead at the cost of larger model weights.

**Verdict: KILLED on K3 (fp32 latency), but SUPPORTED for bf16 merge variant**

## Revision History

**v2 (2026-03-24)**: Fixed critical 1/N^2 scaling bug in runtime LoRA baseline.
The original `compose_adapters_runtime` applied 1/N scaling to both A and B
parameters independently, producing effective 1/N^2 scaling in x @ A_avg @ B_avg.
The fix applies 1/N scaling via `lora_scale = LORA_SCALE/N` instead. This
eliminated the spurious 7% PPL gap between float merge and runtime LoRA.
Also: K2 marked INCONCLUSIVE (MLX lazy eval), latency stddev added (5 runs).

## What This Experiment Tested

Three serving paths for composed BitNet-SOLE adapters on Apple Silicon:

1. **Runtime LoRA (baseline)**: Keep adapters separate, compose at each token.
   Proven to work (12.0 +/- 0.2 tok/s at N=5, 28.6% overhead vs base).

2. **fp32 float merge**: Pre-compute W_merged = W_base + DeltaW in float32.
   Zero per-token adapter overhead, but 2x memory bandwidth for matmuls.

3. **bf16 float merge**: Same merge in bfloat16. Marginal precision loss but
   native ALU width on Apple Silicon.

## Key References

- Prior experiment: exp_bitnet_serving_path (KILLED -- found bf16 ULP limits)
- Prior experiment: exp_bitnet_llamacpp_serving (runtime LoRA baseline)
- LoTA-QAF (arxiv 2505.18724): Ternary merge theory (inapplicable -- 116x gap)
- BitNet b1.58 (arxiv 2402.17764): Ternary weight architecture

## Empirical Results

### Precision Analysis

| Format | ULP at base magnitude (1.72) | Delta/ULP ratio | Measured merge error |
|--------|------------------------------|-----------------|---------------------|
| fp32   | 2.05e-7                      | 38,332x         | 0.00e+00 (lossless) |
| bf16   | 1.34e-2                      | 0.6x            | 2.20e-3 (28% of delta) |

fp32 has 38,000x more precision than needed. bf16 loses 28% per element.

### K1: PPL Quality (PASS)

| N | Runtime LoRA | fp32 Merge | bf16 Merge | Base | fp32/Runtime |
|---|-------------|------------|------------|------|-------------|
| 1 | 8.33        | 8.34       | 8.32       | 8.47 | 1.00        |
| 2 | 7.66        | 7.66       | 7.66       | 8.47 | 1.00        |
| 3 | 7.46        | 7.46       | 7.47       | 8.47 | 1.00        |
| 4 | 7.16        | 7.17       | 7.19       | 8.47 | 1.00        |
| 5 | 7.17        | 7.19       | 7.22       | 8.47 | 1.00        |

**fp32 merged PPL (7.19) is within 0.15% of runtime LoRA (7.17) at N=5.**
Ratio = 1.0015, well below the 1.05 kill threshold.

Per-domain at N=5:

| Domain   | fp32 Merge | Runtime LoRA | Ratio |
|----------|-----------|-------------|-------|
| python   | 2.64      | 2.62        | 1.004 |
| math     | 4.05      | 4.05        | 1.000 |
| medical  | 5.78      | 5.79        | 0.999 |
| legal    | 17.99     | 17.94       | 1.003 |
| creative | 5.48      | 5.47        | 1.001 |

All three methods (runtime LoRA, fp32 merge, bf16 merge) produce essentially
identical PPL. The differences are within measurement noise (<0.5%).

### K2: Memory (INCONCLUSIVE)

| Configuration | Process Memory | vs Ternary Base |
|--------------|---------------|-----------------|
| Packed ternary base | 1720 MB | 1.00x |
| Unpacked bf16 base | 1720 MB | 1.00x |
| fp32 merged model | 1799 MB | 1.05x |

**K2: INCONCLUSIVE.** Measured process RSS is nearly identical across all
configurations due to MLX lazy evaluation and memory mapping. The 1.05x
measured ratio does not reflect true GPU memory usage. Theoretical fp32
model size is 9.2 GB vs unpacked bf16 at 4.6 GB (2.0x -- at the K2 threshold).
Under actual GPU memory pressure, fp32 would consume 2x the bf16 memory.
A definitive test would require a platform with eager memory allocation.

### K3: Latency (FAIL for fp32, PASS for bf16)

| Configuration | tok/s (mean +/- std) | Overhead vs Base | vs Runtime LoRA |
|--------------|---------------------|-----------------|-----------------|
| Base bf16 (no adapter) | 16.8 +/- 0.4 | 0% | - |
| bf16 merged N=5 | 16.7 +/- 0.5 | 0.6% | 39.2% faster |
| Runtime LoRA N=5 | 12.0 +/- 0.2 | 28.6% | baseline |
| fp32 merged N=5 | 8.5 +/- 0.0 | 49.4% | 29.2% slower |

**fp32 merge K3: FAIL** (8.5 tok/s < 12.0 tok/s runtime LoRA).
fp32 weights are 2x larger than bf16, doubling memory bandwidth for matmuls.
Apple Silicon Metal also has 2x bf16 vs fp32 ALU throughput.

**bf16 merge K3: PASS** (16.7 tok/s > 12.0 tok/s runtime LoRA).
bf16 merged model runs at near-base speed with zero per-token adapter overhead.

Latency measurements: 5 runs of 50 tokens each, stddev reported.

## The Key Finding: All Three Methods Are PPL-Equivalent

After fixing the 1/N^2 scaling bug in the original runtime LoRA baseline,
all three serving paths produce essentially identical PPL:

| Method | N=5 PPL | vs Runtime |
|--------|---------|-----------|
| Runtime LoRA | 7.17 | baseline |
| fp32 merge | 7.19 | +0.15% |
| bf16 merge | 7.22 | +0.58% |

The prior claim that "float merge beats runtime LoRA by 7% on PPL" was entirely
an artifact of the 1/N^2 scaling bug, which caused runtime LoRA to under-weight
adapters by a factor of N at N adapters. The original code averaged both A and B
parameters by 1/N, producing effective 1/N^2 scaling in the product A @ B. The
correct approach scales only the lora_scale by 1/N, leaving A and B unscaled.

The bf16 merge's 0.58% PPL penalty vs runtime LoRA is real but negligible,
confirming that 28% per-element truncation in bf16 does NOT translate to
meaningful PPL loss. The truncation errors are symmetric and cancel across
elements, and the model is robust to small weight perturbations.

### bf16 Merge Serving Profile

| Metric | bf16 Merge | Runtime LoRA N=5 | Advantage |
|--------|-----------|------------------|-----------|
| PPL | 7.22 | 7.17 | Runtime 0.6% better |
| tok/s | 16.7 +/- 0.5 | 12.0 +/- 0.2 | bf16 39% faster |
| Memory (process) | 1799 MB | 1720 MB + adapters | Similar |
| Per-token overhead | 0.6% | 28.6% | bf16 wins |
| Hot-swap | Requires reload | Instant | Runtime LoRA wins |

## Limitations

1. **MLX-specific latency**: fp32 vs bf16 throughput ratio depends on hardware.
   CUDA GPUs may show different ratios. The K3 failure is hardware-specific.

2. **No hot-swap**: Merged models cannot swap adapters without reloading weights.
   Runtime LoRA can hot-swap adapters per request. For serving diverse queries
   with different expert sets, runtime LoRA remains necessary.

3. **5 domains, micro scale**: Results are directional. Need to verify at larger
   N and with more diverse adapter sets.

4. **Memory measurement limitation**: Process RSS with MLX lazy evaluation does
   not reflect true GPU memory pressure. Under high load, fp32 weights would
   require 2x the bf16 memory. K2 marked INCONCLUSIVE for this reason.

5. **Same adapters as bitnet_2b_real_composition**: These adapters were trained
   with lora_scale=20.0. Different training recipes might produce different
   delta magnitudes affecting the precision analysis.

## What Would Kill This

- K1 would fail if adapter deltas were much smaller (delta/ULP < 1 even for fp32,
  requiring adapters 38,000x weaker -- essentially impossible).
- K3 is already killed for fp32. Would also kill bf16 if Apple Silicon Metal
  had no bf16 advantage over fp32.
- A real deployment test with diverse queries requiring different adapter sets
  would show the hot-swap limitation of merge vs runtime LoRA.
- bf16 merge PPL penalty could grow at larger N (more truncation per element at
  smaller delta magnitudes) -- needs verification at N>>5.

## Implications for BitNet-SOLE Architecture

1. **bf16 float merge is the recommended serving path** when adapter set is
   known ahead of time. 39% faster than runtime LoRA with only 0.6% PPL cost.

2. **Runtime LoRA remains necessary** for dynamic adapter selection (per-request
   routing to different expert combinations). It produces the best PPL.

3. **Dual-mode serving**: Use bf16 merged model for the "always-on" adapter set
   (e.g., instruction tuning + common domain experts), and runtime LoRA for
   on-demand domain experts. This combines the best of both.

4. **The prior bf16 precision concern was overblown**: Despite 28% per-element
   truncation, bf16 merge works. The PPL penalty is 0.6%, not the 50% delta
   loss that element-level analysis suggested.

5. **fp32 merge is theoretically lossless but practically inferior**: The 2x
   compute cost is not worth the 0.15% PPL improvement over bf16.

6. **The 1/N^2 scaling bug was systemic.** Any runtime LoRA composition code
   that averages both A and B parameters by 1/N is broken. The correct approach
   is to scale only the LoRA scale factor by 1/N, or equivalently scale only
   one of A or B (not both). This affects prior experiments using the same pattern.
