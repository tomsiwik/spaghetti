# Peer Review: adapter_inference_speed_mlx

## NotebookLM Findings

Skipped -- this is a benchmarking experiment with straightforward methodology. The math and claims can be verified directly from code and results.json.

## Mathematical Soundness

### FLOPs Analysis: Correct

The MATH.md derivation is clean and verifiable:
- Base FLOPs per block: 24*B*T*d^2. At micro values: 12,582,912 per block. Correct.
- LoRA FLOPs per block: 36*B*T*r*d. At micro values: 1,179,648 per block. Correct.
- Overhead ratio rho = 3r/(2d) = 9.375%. Correct.

The theoretical overhead only counts FLOPs. The paper honestly reports and explains the 3-4x gap between theory and measurement (kernel dispatch, Python overhead, memory access). This is appropriate for a benchmarking paper.

### Power-Law Fit: Methodologically Sound but Fragile

The log-log fit for beta uses 4 data points (k=1,2,4,8). With only 4 points, a single outlier changes the exponent significantly. The reported beta=0.955 is plausible but the confidence interval is not reported. This is acceptable for a directional result at micro scale.

### Pre-Merge Analysis: Trivially Correct

Pre-merge overhead is 0% by construction (same weight shapes, same forward pass). The slightly negative measured values (-0.67% to -0.80%) are within noise and likely reflect run-to-run variance in Metal scheduler behavior. The paper correctly interprets these as "within measurement noise."

### One Concern: K2 Tests the Wrong Thing

**K2 as stated in MATH.md:** "N-adapter overhead is sub-linear or linear in N. For runtime apply at fixed k: overhead should be independent of N (library size)."

**K2 as actually tested:** The experiment varies k (adapters applied at runtime) not N (library size). The code at line 374 sets `k=N` -- it applies all N adapters, so k and N are confounded. The experiment never tests whether overhead is independent of library size at fixed k (e.g., k=2 with N=8 library vs k=2 with N=2 library).

This is a minor issue because at micro scale with random adapters, library size should not affect forward pass latency (unused adapters are never touched). But the paper should be precise about what K2 actually measures: **runtime overhead scales sub-linearly with k (active adapters), not N (library size).**

## Novelty Assessment

### Prior Art in This Project

This experiment fills a gap in the cross-platform serving matrix:
- CPU (llama.cpp): `exp_bitnet_llamacpp_serving` -- 9.5% + 7.5%*N overhead
- CPU (PyTorch): Referenced in current_direction.md -- pre-merge 0%, dynamic ~260%
- GPU (RTX 4090): Referenced -- pre-merge max +3.3% at N=50
- **Apple Silicon MLX: THIS EXPERIMENT**

The experiment is clearly motivated and non-redundant. It provides a new data point for an architecturally relevant platform.

### External Prior Art

MLX LoRA benchmarking is not novel in isolation -- the MLX community has published adapter fine-tuning benchmarks. However, measuring *inference* overhead of *runtime adapter composition* (activation path, not fine-tuning) on MLX is specific enough to this project's needs that external benchmarks do not directly apply.

### Delta

The contribution is an engineering data point, not a mechanism. This is appropriate -- the experiment exists to inform a serving architecture decision, not to propose a new method.

## Experimental Design

### Strengths

1. **Proper MLX synchronization.** The benchmark correctly calls `mx.eval(out)` inside the timing loop (line 276), ensuring Metal execution completes before the timer stops. This is the most common mistake in MLX benchmarks.

2. **Adequate warmup.** 100 warmup iterations for Metal shader compilation. Sufficient.

3. **Fresh model per pre-merge N.** Lines 355-356 rebuild the model for each N value, preventing weight accumulation artifacts.

4. **500 timed iterations.** Provides reasonable statistical power for sub-millisecond measurements.

### Issues

**Issue 1: k=4 runtime measurement is extremely noisy.**
The results.json shows k=4 std_ms=1.121 against mean_ms=1.859 -- a coefficient of variation of 60.3%. Compare to k=1 (6.7%), k=2 (14.4%), k=8 (3.8%). The k=4 p95 (2.168ms) is close to the k=8 mean (2.330ms), suggesting some k=4 iterations experienced interference. This outlier is not discussed in the paper. The k=4 data point may be pulling the power-law fit downward (making beta appear more sub-linear than it truly is).

**Recommendation:** Report median-based overhead as a robustness check. k=4 median (1.624ms) gives 130.1% overhead vs mean-based 163.5%. This would yield a different beta.

**Issue 2: The RuntimeLoRATransformer re-implements the full forward pass in Python.**
Lines 221-259 rewrite the entire transformer forward pass in Python, replacing the base model's `__call__`. This means the runtime overhead includes not just LoRA matmuls but also the cost of reimplementing attention, softmax, reshape, etc. in explicit Python. The base model's forward pass goes through `nn.Module.__call__` which may have different dispatch characteristics.

This is acknowledged in the discussion ("replaces MLX's optimized nn.Linear.__call__ with explicit Python-level matrix operations") but understated. The 33.2% overhead is an upper bound on LoRA overhead -- part of it is the cost of the Python reimplementation itself. A fairer test would apply LoRA only to `nn.Linear.__call__` while keeping the rest of the forward pass identical.

**Impact on K1 verdict:** This means the KILL on K1 may be overly pessimistic. The true LoRA overhead could be significantly less than 33.2%. However, the KILL verdict is still likely correct -- even at half the measured overhead (~16.5%), it would still exceed 15%.

**Issue 3: No statistical significance test on K1.**
K1 threshold is 15%. Measured is 33.2%. While the gap is large enough that significance is obvious (33.2 >> 15 with std 6.3%), best practice would report a confidence interval: approximately 33.2% +/- 0.55% (95% CI using std/sqrt(500)).

**Issue 4: Pre-merge rebuilds with different random weights.**
Lines 355-356 create a fresh model for each pre-merge N, meaning the base weights differ from the "base" benchmark's weights. This is fine for latency measurement (weight values don't affect computation time) but is worth noting.

## Hypothesis Graph Consistency

The experiment does not have a formal HYPOTHESES.yml entry. The closest nodes are:
- `exp_bitnet_serving_path` (killed) -- tested ternary merge, runtime LoRA on real BitNet-2B
- `exp_bitnet_llamacpp_serving` (supported) -- CPU serving path
- `exp_sole_inference_throughput` (killed) -- macro-scale throughput

This experiment should be registered as a new node or linked to `exp_bitnet_serving_path` as supplementary evidence. The lack of a HYPOTHESES.yml entry means the kill criteria are self-defined rather than externally validated.

The K1 threshold of 15% appears reasonable given the llama.cpp baseline (7.5% per adapter on CPU). However, its provenance is not documented -- why 15% specifically? The MATH.md theoretical prediction of 9.375% would suggest a threshold around 10-12% with some headroom, making 15% generous. The experiment fails even this generous threshold.

## Macro-Scale Risks (advisory)

1. **The 3-4x theory-measurement gap will likely shrink at production scale.** At d=4096, rho = 3*16/(2*4096) = 0.59%. Even at 4x measured/theoretical, that is only ~2.4% per adapter. The KILL on K1 at micro scale does not imply a kill at macro scale.

2. **MLX's lazy evaluation may fuse operations at larger graph sizes.** The micro model's computation graph is small enough that dispatch overhead dominates. At production scale, the graph is much larger and dispatch overhead becomes proportionally smaller.

3. **The recommendation "always pre-merge" may not hold for per-token dynamic routing.** Pre-merge requires knowing the expert set before each forward pass. If the router output changes per token, merge cost must be amortized. The paper acknowledges this but does not measure merge latency.

## Verdict

**PROCEED**

This is a clean, well-executed benchmarking experiment that fills a real gap in the cross-platform serving matrix. The core findings are sound:

- Pre-merge is free on MLX (trivially correct, correctly verified)
- Runtime LoRA is expensive on MLX at micro scale (33.2% for k=1)
- Scaling is sub-linear (beta=0.96, though fragile with 4 points)

The K1 KILL is the right call -- even accounting for the Python reimplementation overhead (Issue 2), single-adapter runtime overhead clearly exceeds 15%. The architectural recommendation (always pre-merge on Apple Silicon) follows logically and is consistent with findings on other platforms.

**Minor fixes to address before merging (not blocking):**

1. Note the k=4 noise anomaly (std/mean=60%) and report median-based overhead as robustness check.
2. Clarify that K2 measures overhead vs k (active adapters), not N (library size), since the experiment confounds the two.
3. Register the experiment in HYPOTHESES.yml with proper node linkage to `exp_bitnet_serving_path`.
4. Acknowledge that RuntimeLoRATransformer reimplements the full forward pass, making the 33.2% an upper bound on pure LoRA overhead.
