# Peer Review: bitnet_float_merge_fp32 (v2 Revision)

## Revision Verification

The prior review required 4 specific fixes. Verification of each:

### Fix 1: 1/N^2 scaling bug in compose_adapters_runtime -- VERIFIED

The `compose_adapters_runtime` function (line 336-347) now sums adapter parameters
without any per-parameter scaling. The 1/N composition scaling is applied via
`lora_scale = LORA_SCALE / N` (line 504) when constructing LoRALinear layers. The
forward pass computes `x @ A_sum @ B_sum * (s/N)`, which has correct 1/N effective
scaling on the product.

The results confirm the fix: runtime PPL at N=5 = 7.1745, fp32 merge = 7.1852,
ratio = 1.0015. The prior 7% gap has disappeared completely, consistent with the
gap being entirely caused by the 1/N^2 bug.

One subtlety correctly identified in MATH.md (lines 108-123): the runtime path
computes `(sum A_i) @ (sum B_i)` (product of sums, includes cross-terms) while
the merge path computes `sum_i (A_i @ B_i)` (sum of products, no cross-terms).
These are mathematically different for N>1. The empirical difference is 0.15% at
N=5, which is honestly reported and negligible.

### Fix 2: Retract wrong PPL explanation -- VERIFIED

The prior "Why Float Merge Beats Runtime LoRA on PPL" section (which attributed
the gap to bf16 truncation in the forward pass) has been removed. PAPER.md lines
110-128 now correctly attribute the prior 7% gap to the 1/N^2 scaling bug, with
the explicit statement: "The prior claim that 'float merge beats runtime LoRA by
7% on PPL' was entirely an artifact of the 1/N^2 scaling bug."

### Fix 3: K2 marked INCONCLUSIVE -- VERIFIED

PAPER.md lines 77-89 clearly marks K2 as INCONCLUSIVE with proper justification
(MLX lazy evaluation and memory mapping make RSS measurements meaningless). MATH.md
lines 129-145 provides the same analysis. HYPOTHESES.yml evidence entry (line 3124)
also states "K2 INCONCLUSIVE."

Minor note: `results.json` still has `k2_pass: true` because the measured ratio
(1.05x) satisfies the 2.0x threshold numerically. The PAPER correctly overrides
this with INCONCLUSIVE, acknowledging the measurement itself is unreliable. This
code-vs-paper distinction is acceptable -- the paper is the authoritative
interpretation.

### Fix 4: Stddev added to latency measurements -- VERIFIED

`BENCH_RUNS` increased from 3 to 5 (line 61). The `benchmark_generation` function
(lines 207-247) computes and reports both mean and stddev for latency and tok/s.
Results.json confirms: runtime LoRA 12.0 +/- 0.2, bf16 16.7 +/- 0.5, fp32 8.5
+/- 0.0, base 16.8 +/- 0.4 tok/s. Both PAPER.md and MATH.md report stddev
alongside all latency figures.

## Mathematical Soundness

### ULP Analysis: Correct

Spot-checked against results.json:
- fp32 ULP at 1.72: `1.72 * 2^{-23} = 2.05e-7` -- correct
- bf16 ULP at 1.72: `1.72 * 2^{-7} = 0.0134` -- correct
- delta/ULP: 0.007854 / 2.05e-7 = 38,312 (paper reports 38,332; difference from
  rounding of alpha between measured 1.71875 and reported 1.72) -- acceptable

### Merge Formula: Correct

The weight-space delta `B^T @ A^T * lora_scale * scale` (line 139) correctly
transposes to match `nn.Linear.weight` convention where forward is `y = x @ W^T`.
Wait -- actually, `nn.Linear` in MLX stores weight as `(out_features, in_features)`
and computes `x @ W.T`. The LoRA forward is `x @ A @ B * scale` where A is
`(in, r)` and B is `(r, out)`. The weight-space equivalent delta is `(B^T @ A^T)`
giving shape `(out, in)` to add to `W`. This is correct.

### Cross-Term Transparency: Good

The paper is honest that runtime LoRA composition (product of sums) differs from
merge (sum of products) by cross-terms. At N=1 they are identical (confirmed:
8.328 vs 8.341, 0.15% gap likely from bf16 vs fp32 precision in the base weight
representation). The 0.15% gap at N=5 is plausibly from cross-terms. This is a
known approximation, not a bug.

### PPL Numbers vs Results.json: Consistent

| Metric | PAPER | results.json | Match |
|--------|-------|-------------|-------|
| Runtime N=5 | 7.17 | 7.1745 | Yes (rounding) |
| fp32 N=5 | 7.19 | 7.1852 | Yes |
| bf16 N=5 | 7.22 | 7.2159 | Yes |
| Base | 8.47 | 8.4689 | Yes |
| fp32 tok/s | 8.5 +/- 0.0 | 8.5 +/- 0.0 | Yes |
| bf16 tok/s | 16.7 +/- 0.5 | 16.7 +/- 0.5 | Yes |
| Runtime tok/s | 12.0 +/- 0.2 | 12.0 +/- 0.2 | Yes |

## Novelty Assessment

This is an engineering measurement, not a novel technique. LoRA weight merging is
standard practice (Hu et al., 2022). The contribution is empirical validation on
the specific BitNet-SOLE stack with MLX. This is appropriate for a micro-experiment
in service of the architecture -- it does not claim novelty.

## Experimental Design

The design is sound for a micro-experiment:
- Three serving paths compared on the same adapters, data, and hardware
- PPL measured across N=1..5 with 5 domains and 50 validation batches
- Latency benchmarked with 5 runs of 50 tokens each, stddev reported
- Kill criteria clearly defined and honestly applied
- K3 failure for fp32 is legitimate and clearly explained

The cross-term difference between runtime and merge composition is a minor
confound: the methods are not computing exactly the same function. However, the
paper is transparent about this, and the 0.15% difference is negligible for the
practical conclusions drawn.

## Hypothesis Graph Consistency

HYPOTHESES.yml node (line 3107-3135) correctly shows:
- Status: killed
- Evidence: v2 revision with all four fixes noted
- Kill criteria match the code's K1/K2/K3 checks

The bf16 merge finding (SUPPORTED) does not have its own hypothesis node. The
prior review flagged this as non-blocking and it remains non-blocking -- it is a
secondary finding within the fp32 experiment.

## Macro-Scale Risks (advisory)

1. **Cross-term divergence at large N.** The product-of-sums vs sum-of-products
   gap is 0.15% at N=5. At N=50 or N=100, cross-terms may dominate. If runtime
   LoRA is used at scale, the macro experiment should verify PPL equivalence at
   the target N.

2. **bf16 truncation at large N.** At N=100, individual deltas are 1/100th the
   N=1 magnitude, pushing more elements below the bf16 ULP. The paper's claim
   that truncation errors cancel needs verification at N>>5.

3. **Hardware specificity.** The bf16 latency advantage is Apple Silicon specific
   (2x bf16 vs fp32 ALU throughput). On NVIDIA GPUs, the fp32 penalty may differ.

4. **Systemic 1/N^2 bug.** The paper correctly flags (line 192-195) that the same
   pattern may exist in other experiments. Any prior runtime LoRA composition
   results using the old averaging pattern should be audited.

## Verdict

**PROCEED**

All four required fixes have been properly implemented and verified:

1. The 1/N^2 scaling bug is fixed -- lora_scale carries the 1/N factor, not
   per-parameter averaging. The 7% PPL gap disappeared entirely, confirming
   the bug was the sole cause.

2. The wrong bf16 truncation explanation has been retracted and replaced with
   the correct attribution to the scaling bug.

3. K2 is marked INCONCLUSIVE with clear justification about MLX lazy eval.

4. Latency measurements now include stddev from 5 runs.

The conclusions are sound and appropriately scoped: fp32 merge is KILLED on
latency (K3), bf16 merge is SUPPORTED as the recommended serving path (39%
faster than runtime LoRA, 0.6% PPL cost), and all three methods are shown to
be PPL-equivalent after the bug fix. The paper is honest about limitations
(MLX-specific latency, no hot-swap, micro scale, cross-term approximation).
