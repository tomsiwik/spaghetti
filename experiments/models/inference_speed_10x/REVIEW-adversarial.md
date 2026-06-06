# Peer Review: Inference Speed 10x

## NotebookLM Findings
Skipped (manual deep review performed instead -- all source material is local markdown/JSON).

## Mathematical Soundness

### CRITICAL: Bandwidth constant is wrong (153 GB/s vs 273 GB/s)

The entire "132.5% bandwidth utilization" narrative is built on `bw_available = 153e9` (run_experiment.py line 156, MATH.md Section 1). This is wrong. Other experiments in the same project use **273 GB/s** for M5 Pro:

- `benchmark_composition_latency_sweep/MATH.md` line 51: "M5 Pro memory bandwidth: ~273 GB/s"
- `memory_optimized_serving/LEARNINGS.md` line 19: "consistent with M5 Pro's 273 GB/s memory bandwidth"
- `benchmark_composition_latency_sweep/LEARNINGS.md` line 23: "273 GB/s"

153 GB/s is the M3 Pro's bandwidth. The M5 Pro has approximately 273 GB/s.

**Corrected calculation:**
```
T_max = 273 GB/s / 1.178 GB = 231.7 tok/s
Utilization = 172 / 231.7 = 74.2%
```

At 74.2%, this is a perfectly normal, unremarkable bandwidth utilization. The entire Section 1 of MATH.md -- the "effective read cost" derivation dividing packed ternary bytes by 4 to get an "equivalent 788 MB" -- is a post-hoc rationalization of an impossible number. With the correct bandwidth, there is nothing anomalous to explain.

This error propagates to:
- MATH.md: "132.5% BW utilization" claim, "effective bandwidth" derivation
- PAPER.md: "32% above the naive memory bandwidth bound"
- LEARNINGS.md: "132.5% of the naive bandwidth bound"
- results.json: `bandwidth_utilization_pct: 132.5`, `theoretical_max_tps: 129.8`
- PAPER.md Section "Architectural Insight for SOLE" point 3

The "effective read cost" analysis (521 MB / 4 + 657.6 MB = 788 MB) is not mathematically justified. Ternary multiply-accumulate being cheaper than bf16 FMA does not reduce the bytes read from memory. Memory bandwidth is about data transfer, not compute cost. The analysis conflates compute-bound and memory-bound regimes.

### LoRA FLOPs analysis is correct

The per-layer LoRA overhead calculation (r * (d_in + d_out) FLOPs per layer) is standard and correct. The distinction between attention (d_out = 2560) and MLP (d_out = 6912) projections explaining the 1.39x cost ratio is sound arithmetic.

### addmm analysis is directionally correct but overclaims

The claim that `mx.addmm` saves "one kernel launch per layer (210 launches saved per token)" is plausible but unverified. No profiling data (Metal Performance Shaders traces, kernel launch counts) is presented. The 10% speedup (88.2 -> 97.4 tok/s) could be from:
1. Kernel fusion (as claimed)
2. Reduced intermediate memory allocation (avoiding the temporary `lora_out` tensor)
3. Better instruction scheduling in the fused path

The mechanism attribution is speculative without profiling. The speedup itself is real and reproducible.

### Multi-adapter scaling model

The linear regression `tok/s ~ 172 - 23*N` (R^2 ~ 0.98) is claimed from 3 data points (N=1, N=2, N=5). With 3 points, R^2 ~ 0.98 is expected for nearly any monotonic relationship. The fit has 1 degree of freedom. This is not evidence of linearity vs. any other functional form.

### mx.compile analysis is accurate

The explanation that KVCache objects violate `mx.compile`'s tree-of-arrays constraint is correct. The note about BitLinear already being a fused custom kernel is relevant.

### Speculative decoding analysis is reasonable

The derivation of required draft model speed (T_draft < 1.5 ms/tok) and the conclusion that no suitable ternary draft model exists is a sound negative result.

## Novelty Assessment

**Low novelty, high practical value.** The findings are:

1. **Correcting a prior measurement artifact** (82 -> 172 tok/s): Important for the project but not a research contribution. The prior measurement simply included Python overhead.

2. **addmm for LoRA serving**: `mx.addmm` is a standard BLAS operation. Using it for LoRA is obvious once you notice the y + h @ B pattern. Not novel, but a useful engineering observation.

3. **Pre-merge is worse for ternary**: This is the most valuable finding. The insight that unpacking ternary to bf16 destroys the bandwidth advantage (1.18 GB -> 4.83 GB) is specific to ternary architectures and worth documenting. However, it is straightforward once stated.

4. **Attention-only adapters faster than full**: Well-known in the LoRA literature (QLoRA, LoRA-FA, etc. all discuss selective layer adaptation). The speed measurement is new data, but the strategy is not.

**Prior art:** The bitnet.cpp reference (2502.11880) is appropriate. The vllm-mlx reference is of a different class. No references to standard LoRA serving optimization literature (e.g., S-LoRA, Punica) which discuss batched LoRA, kernel fusion, and similar optimizations at scale.

## Experimental Design

### Strengths

1. **Multiple measurement methods**: Internal timing (stream_generate perf_counter) vs wall-clock (time.perf_counter wrapping mlx_generate). The distinction and the 10.8% overhead quantification are valuable.

2. **Comprehensive negative results**: Testing precomputed A@B, pre-merge, and KV quantization and honestly reporting they are worse is good experimental practice.

3. **Multiple prompts**: 4 prompts for baseline, showing low variance (172.1, 172.1, 172.0, 171.9 tok/s) demonstrates measurement stability.

4. **Code quality**: Follows CODING_GUIDELINES for memory limits, proper cleanup, systematic phase structure.

### Weaknesses

1. **Single measurement per configuration**: The addmm result (97.4 tok/s) and attention-only result (126.3 tok/s) each come from a single `stream_generate` call on one prompt. No variance reported. Compare with baseline which uses 4 prompts.

2. **No statistical uncertainty**: No confidence intervals, no repeated runs for the optimization strategies. The difference between addmm (97.4) and naive (88.2) is presented as definitive but could include measurement noise.

3. **Attention-only quality impact unacknowledged in criteria**: The S1 criterion is ">100 tok/s with adapter composition." The attention-only result (126.3 tok/s) achieves this, but by removing 3/7 of the adapter's capacity. The paper acknowledges this in Limitations but the CONDITIONAL PASS verdict underplays the severity -- if attention-only adapters lose quality, S1 genuinely fails at 97.4 tok/s.

4. **N=5 multi-adapter test uses naive LoRA, not addmm**: Phase 5 wraps with `BitLinearWithMultiLoRA` using separate matmul + add (line 406: `y = y + (x @ a) @ b * self.lora_scale`), not addmm. The addmm optimization is only tested at N=1. The multi-adapter scaling model does not reflect the best-known optimization.

5. **Hardcoded bandwidth constant**: As detailed above, 153 GB/s is wrong.

## Hypothesis Graph Consistency

- **K1 (>50 tok/s): PASS** -- Unambiguous at 172 tok/s base. Sound.
- **S1 (>100 tok/s with adapter): CONDITIONAL PASS** -- Full adapter: 97.4 tok/s (FAIL). Attention-only: 126.3 tok/s (PASS). The "conditional" framing is honest but should be clearer: the original S1 likely intended full adapter composition.

The experiment tests what it claims to test. The kill criterion is appropriate and clearly passed.

## Macro-Scale Risks (advisory)

1. **Attention-only adapters need quality validation**: The 126.3 tok/s number is meaningless if attention-only adapters degrade domain quality by more than a few percent. This should be the immediate next experiment.

2. **Multi-adapter addmm composition untested**: The scaling model (172 - 23*N) uses naive LoRA, not the best-known addmm variant. At N=5 with addmm, the actual throughput is unknown.

3. **The hybrid strategy (attention routed, MLP pre-merged) is untested**: LEARNINGS.md recommends this but no experiment validates it. Pre-merging MLP adapters into the ternary base was shown to be slow (55.2 tok/s for full pre-merge), so the hybrid would need to selectively pre-merge only MLP layers while keeping the base ternary for attention layers.

## Verdict

**REVISE**

The experiment produces useful engineering data, but the central mathematical narrative (132.5% bandwidth utilization, "effective read cost" derivation) is wrong due to an incorrect hardware constant. The fixes are:

1. **Correct the bandwidth constant from 153 GB/s to 273 GB/s** (or whatever the verified M5 Pro spec is). Update MATH.md Section 1, PAPER.md, LEARNINGS.md, and re-run `run_experiment.py` line 156. The "exceeds naive bandwidth bound" narrative must be removed entirely.

2. **Remove the "effective read cost" derivation** (MATH.md lines 47-53). Dividing memory bytes by a compute efficiency factor is not a valid bandwidth calculation. With the correct bandwidth, 172 tok/s = ~74% utilization, which needs no exotic explanation.

3. **Add variance measurements for optimization strategies**: Run addmm and attention-only tests across 4 prompts (like baseline) and report standard deviation. The single-prompt measurements are insufficient for the precision of the claims.

4. **Test multi-adapter with addmm**: Phase 5 should use BitLinearWithLoRAAddmm, not BitLinearWithMultiLoRA with naive matmul. The scaling model should reflect the best-known optimization.

5. **Clarify S1 verdict**: State explicitly that S1 FAILS for full adapter (97.4 tok/s) and PASSES only for attention-only adapter (126.3 tok/s, quality unvalidated). Remove "CONDITIONAL PASS" ambiguity.

None of these are blocking kills -- the data itself is valuable and the negative results (pre-merge worse, KV quant hurts) are genuinely useful for the SOLE architecture. But the bandwidth narrative is wrong and must be corrected before these numbers propagate to other experiments or VISION.md.

---

## Re-Review (Post-REVISE)

### Fix Verification

All 5 mandated fixes applied and verified:

1. **BW 153→273 GB/s: FIXED.** `run_experiment.py` line 156: `bw_available = 273e9`. MATH.md Section 1: "273 GB/s for M5 Pro." results.json: `bandwidth_utilization_pct: 74.2`, `theoretical_max_tps: 231.6`. PAPER.md: "74.2% of the M5 Pro memory bandwidth bound (231.7 tok/s)." LEARNINGS.md: "273 GB/s memory bandwidth (not 153 GB/s as previously stated)." The "132.5% BW utilization" narrative is entirely removed. ✓

2. **"Effective read cost" derivation removed: FIXED.** MATH.md Section 1 now states: "74.2% bandwidth utilization is unremarkable and consistent with production serving benchmarks on Apple Silicon." No exotic mechanism invoked. ✓

3. **Variance added: FIXED.** PAPER.md and MATH.md report addmm = 97.2 ± 0.0 tok/s, attn-only = 126.7 ± 0.2 tok/s, both measured across all 4 test prompts. Variance is negligible — throughput is highly deterministic at this scale. ✓

4. **Phase 5 multi-adapter uses addmm: FIXED.** `run_experiment.py` line 396: BitLinearWithMultiLoRA docstring says "addmm". Line 407: `y = mx.addmm(y, h, b, alpha=self.lora_scale)`. results.json Phase 5: N=2 = 87.6 tok/s, N=5 = 39.6 tok/s. MATH.md Section 5 updated: "Phase 5 now uses addmm for all adapters." Scaling model updated to ~26 tok/s/adapter (was ~23 with naive). ✓

5. **S1 verdict clarified: FIXED.** results.json: `s1_full_adapter_pass: "False"`, `s1_full_adapter_verdict: "FAIL (full adapter: 88.2 tok/s < 100)"`, `s1_attn_only_verdict: "PASS (attn-only addmm, quality not validated)"`. PAPER.md: "S1 FAIL (full adapter: 97.2 tok/s < 100)". No "CONDITIONAL PASS" ambiguity. ✓

### Remaining Minor Issues (Non-Blocking)

1. **S1 discrepancy between results.json and PAPER.md:** results.json says full adapter 88.2 tok/s (naive, from Phase 4), PAPER.md says 97.2 tok/s (addmm, from optimize_final.py). Both are correct — they measure different configurations. The PAPER.md number (addmm) is the relevant one for S1. Confusing but not wrong.

2. **Multi-adapter scaling from 3 points:** Linear fit R^2 ~ 0.97 from N=0,1,2,5 (actually 4 points now including base). Still minimal degrees of freedom, but adequate for an engineering estimate.

3. **Attention-only quality unvalidated:** Correctly flagged as limitation in PAPER.md. Not a review issue — this is an acknowledged next step.

### Verdict: **PROCEED**

All 5 mandated fixes applied correctly. The bandwidth narrative is fixed. The data is sound engineering measurement — 172 tok/s base, 97.2 tok/s full adapter (addmm), 126.7 tok/s attention-only. Key negative results (pre-merge worse, precomputed A@B worse, KV quant hurts at short context) are valuable for SOLE architecture decisions. The experiment is complete.
