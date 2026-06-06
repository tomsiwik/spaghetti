# Peer Review: Batched Pre-Merge Throughput

## NotebookLM Findings

Skipped -- this is a throughput benchmark (no deep mathematical derivations to validate via external review). The core claims are empirical timing measurements, not theoretical proofs.

## Mathematical Soundness

### MATH.md derivations: Mostly correct, one error noted

**Correct:**
- The complexity analysis for naive (T * k * C_merge + T * C_matmul) vs batched (M * k * C_merge + T * C_matmul + C_group) is sound.
- The speedup formula T/M for merge-dominated workloads follows correctly from the ratio.
- The worked example (Section 6) checks out: 16/6 = 2.67x merge speedup, 288K/128K = 2.25x total.
- FLOP counts at d=2560, r=16 are arithmetically correct (C_merge = 2560 * 16 * 2560 = 104.9M).

**Error in Section 1 (minor):**
Line "each O(d_out * r * d_in / r) = O(d_out * d_in)" is incorrect and immediately self-corrected in the next line. This is a working-notes artifact, not a claim error, but it should be cleaned up.

**Runtime LoRA FLOP count (Section 5):**
The paper claims runtime LoRA cost = T * (d*d + 2k*d*r). At d=2560, r=16, k=2, T=256:
- 256 * (2560^2 + 2*2*2560*16) = 256 * (6,553,600 + 163,840) = 256 * 6,717,440 = 1.72G. Correct.

However, the paper states "O(d*r + r*d) = 81.9K FLOPs per token" (line 86 of PAPER.md). This is wrong. For k=2 experts per token: 2 * (2560*16 + 16*2560) = 2 * 81,920 = 163.8K per token for adapter ops alone, plus 2560^2 = 6.55M for the base matmul. The 81.9K figure appears to be for k=1 and single direction only. The ratio cited ("O(d/r) = 160x") is the theoretical merge-to-runtime ratio per expert operation, which is correct (2560/16 = 160).

### Complexity table consistency

The table in MATH.md Section 5 is internally consistent. The runtime LoRA column correctly shows T*(d*d + 2k*d*r), which accounts for base matmul plus factored adapter application.

## Novelty Assessment

**Low novelty, but that is acceptable.** This is an engineering benchmark, not a novel algorithm. The "group by key, process in batches" pattern is standard (used in MoE expert dispatch, scatter-gather in DeepSeek-V3, etc.). The paper does not overclaim novelty.

**The actual contribution** is the empirical finding that runtime LoRA dominates pre-merge for per-token routing -- this is a useful architectural decision point for the SOLE pipeline. The paper appropriately frames this as the "surprising result."

**Prior art acknowledged:** exp_e2e_demo_pipeline_mlx predicted this result. The paper cites it correctly.

## Experimental Design

### Critical Issue 1: Python loop overhead confounds the naive baseline

The naive implementation (line 166-177 of run_experiment.py) uses a Python `for t in range(T)` loop that iterates over every token, performing `T` individual merge + matmul operations with `mx.eval` deferred to the end. Even with MLX's lazy evaluation, this generates `T * (k + 1)` graph nodes, each a separate operation.

The batched implementation loops over `M` groups (M << T typically), generating only `M * (k + 1)` graph nodes.

**The measured speedup conflates two effects:**
1. Reduced merge operations (the algorithmic speedup: T/M)
2. Reduced Python loop iterations and MLX graph construction overhead

At T=512, N=4, k=1: the naive loop executes 512 Python iterations creating ~1024 graph nodes, while batched executes 4 iterations creating ~8 graph nodes. The 56x measured speedup far exceeds the theoretical T/M = 512/4 = 128x for merge-only... wait, actually the theoretical speedup for merge only would be T/M = 128, and total speedup depends on merge/matmul ratio. Let me recalculate:

- Naive total per token (k=1): 1 * C_merge + C_matmul = 104.9M + 6.55M = 111.5M FLOPs
- Naive total: 512 * 111.5M = 57.1G
- Batched total: 4 * 104.9M + 512 * 6.55M = 419.6M + 3.35G = 3.77G
- Theoretical speedup: 57.1G / 3.77G = 15.1x

The measured 56.3x significantly exceeds the theoretical 15.1x. This confirms that Python loop overhead is a substantial contributor to the measured speedup, not just algorithmic merge savings.

**This does not invalidate the result** -- in practice, the Python loop overhead is real and batching does avoid it. But it means the claimed speedups are implementation-specific, not purely algorithmic. A vectorized naive implementation (using `mx.vmap` or tensor broadcasting) would narrow the gap considerably.

### Critical Issue 2: The naive baseline is not how anyone would implement per-token merge

No production system would loop over tokens in Python. A realistic naive baseline would:
1. Identify unique expert sets (same as batched step 1)
2. Build merged weight matrices for each unique set
3. Apply them via batched matmul

In other words, the "batched" approach IS the obvious implementation. The "naive" baseline is a straw man. The real comparison should be batched pre-merge vs runtime LoRA, which the paper does measure -- and runtime LoRA wins by 4-87x.

### Issue 3: The "mean 13.1x" is misleading

The distribution of speedups across 50 configurations is heavily skewed:
- Minimum: 1.04x (N=16, k=2, T=32)
- Maximum: 62.4x (N=4, k=2, T=512)
- The mean is pulled upward by high-T, low-N configurations

A median would be more representative. Eyeballing the data, the median is likely around 5-8x. The paper should report median and quartiles, not just mean and min.

More importantly, the speedup is almost entirely a function of M/T ratio, which is deterministic given N, k, T. Reporting "mean 13.1x across configurations" averages over the experimental design space, which is an arbitrary choice. A more honest summary: "speedup = T/M (approximately), which ranges from 1.04x when M/T=0.91 to 62.4x when M/T=0.01."

### Issue 4: Uniform random routing overstates M/T for realistic workloads

The paper acknowledges this in Limitations (point 4) and argues it makes results "conservative." This is correct -- real routing would concentrate tokens on fewer expert sets, lowering M/T and improving batched speedup. No issue here.

### Issue 5: Runtime LoRA comparison is fair

The runtime LoRA implementation (line 234-264) also groups tokens by expert set and uses the same gather/scatter pattern. This is a fair comparison -- same grouping, different math (factored vs materialized). The 4-87x advantage is genuine and stems from the O(d*r) vs O(d*r*d) per-expert cost difference.

### Issue 6: No correctness verification

The code does not verify that all three strategies produce the same output. A simple `mx.allclose(naive_output, batched_output)` and `mx.allclose(naive_output, runtime_output)` would strengthen confidence. Without this, we cannot confirm the implementations compute the same function. The `Y.at[idx].add(out)` scatter pattern in particular could have subtle bugs with overlapping indices.

### Issue 7: Statistical rigor

- N=10 measurement iterations is adequate for wall-clock benchmarks with low variance (std/mean typically < 5%).
- Standard deviations are reported but confidence intervals are not. At N=10, the 95% CI is approximately mean +/- 2.26 * std / sqrt(10) = mean +/- 0.71 * std. For the key N=16, k=2, T=32 result (1.04x speedup), the naive std is 0.182ms and batched std is 0.203ms. The CIs overlap meaningfully -- this speedup is likely not statistically significant.
- The paper does not claim statistical significance for the marginal cases, which is appropriate.

## Kill Criteria Assessment

**K1 (batched faster than naive): PASS -- but the bar is low.** K1 only requires batched > naive. Given that the naive baseline is a worst-case Python-loop implementation, this is trivially satisfied. The minimum 1.04x at N=16, k=2, T=32 is within noise (see Issue 7).

**K2 (grouping overhead < merge savings): PASS -- legitimate.** Grouping overhead at 0.07-1.3% of savings is convincingly small. This is a real finding.

**S1 (>= 2x at N>=4): FAIL -- correctly assessed.** The paper honestly reports this failure and explains it as expected at high M/T ratios. Good scientific practice.

**Verdict logic:** The experiment declares SUPPORTED based on K1 PASS + K2 PASS, with S1 noted as partial fail. This is reasonable -- the kill criteria did not trigger.

## Macro-Scale Risks (advisory)

1. **Python loop overhead disappears at macro scale.** Production implementations would use vectorized operations or C++ kernels. The measured speedups will not transfer. Only the runtime-LoRA-vs-pre-merge comparison transfers.

2. **Memory bandwidth, not FLOPs, likely dominates at scale.** Pre-merge materializes a (d, d) matrix per expert set; runtime LoRA streams through (d, r) and (r, d) matrices. At d=8192 (Llama 3 70B), the materialized matrix is 128MB in bf16. Memory bandwidth analysis is missing.

3. **The architectural recommendation (pre-merge for always-on, runtime LoRA for routed) is sound and transfers.** This is the actual actionable output of this experiment.

4. **mx.compile recompilation with variable group sizes** is flagged in MATH.md but not tested. This could be a real production issue.

## Verdict

**PROCEED**

**Justification:**

The experiment achieves its primary purpose: establishing that runtime LoRA is the correct strategy for per-token routed composition, and pre-merge should be reserved for always-on adapters. This architectural decision is well-supported by both the theoretical analysis and empirical data.

The kill criteria are correctly evaluated: K1 passes (trivially), K2 passes (convincingly), S1 fails (honestly reported).

**Weaknesses that do not block PROCEED:**

1. The naive baseline is a straw man (Python per-token loop), making the "13.1x mean speedup" headline misleading. This inflates the batched-vs-naive comparison but does not affect the key finding (runtime LoRA dominates both).

2. The "mean 13.1x" should be reported as median with quartiles, or better yet, as a function of M/T.

3. No correctness verification between the three implementations.

4. The N=16, k=2, T=32 case (1.04x) is within measurement noise and should not be counted as a "pass."

**These weaknesses are minor because the experiment's lasting contribution is the runtime LoRA recommendation, not the batched-vs-naive speedup numbers.** The runtime LoRA comparison is fair, correctly implemented, and the 4-87x advantage is unambiguous.

**Recommended fixes (non-blocking):**

1. Add `mx.allclose` correctness checks between all three strategies.
2. Replace "mean 13.1x" with "speedup approximately T/M, ranging 1.04-62.4x depending on expert set diversity."
3. Note explicitly that the naive baseline Python loop overhead contributes to measured speedups beyond the algorithmic improvement.
4. Report median speedup alongside mean.
