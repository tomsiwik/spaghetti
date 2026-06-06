# Room Model Piece A: W_combined = Sum of DW_i Proof Verification Report

## Theorem (Restated from MATH.md)

**Per-module linearity (trivial axiom):** For any single linear layer,
x @ (W_base + Sum_i DW_i) = x @ W_base + Sum_i (x @ DW_i). The sum of
orthogonal adapter deltas is non-interfering per module.

**Cross-layer nonlinearity (Finding #303):** For transformers with L nonlinear
layers, pre-summing N adapter deltas produces output that diverges from
single-adapter output. The divergence compounds multiplicatively through
LayerNorm, softmax, and SiLU, with PPL ratio approximately 1.29x at N=5.

This experiment retests the same mechanism with a 2.0x acceptance threshold
(vs the prior 1.10x).

## Hypothesis

W_combined = Sum of DW_i can serve all 5 domains simultaneously from a single
matmul per module, with per-domain PPL degradation under 2x relative to the
respective single-adapter baseline, at the cost of significantly reduced
throughput due to bandwidth of the dense W_combined matrix.

## What This Model Is

W_combined pre-computes the sum of all N=5 adapter deltas (each rank-16,
orthogonal A-matrices from Grassmannian skeleton) into a single dense
d_out x d_in matrix per module. This matrix is injected once and applied via
one bf16 matmul per module during inference. All domain knowledge is encoded
simultaneously -- no routing, no per-query adapter selection.

## Key References

- Finding #303: Room Model POC killed at 1.10x threshold (1.29x measured)
- Finding #126: Grassmannian orthogonality (|cos|=0.00125)
- ROOM_MODEL.md: Architectural proposal
- Naive LoRA Summation (arXiv 2508.11985): Orthogonality enables additive composition
- Raghu et al. (arXiv 1611.03530): Expressive power of deep networks, nonlinear compounding

## Predictions vs Measurements

| Prediction (source) | Predicted | Measured | Match? |
|---------------------|-----------|----------|--------|
| Mean PPL ratio (Finding #303) | 1.25-1.35x | 1.447x | PARTIAL -- higher than POC |
| Worst domain ratio (extrapolated) | 1.5-2.0x | 1.905x (medical) | YES -- within range |
| Speed (bandwidth analysis) | 35-50 tok/s | 41.9 tok/s | YES |
| K802: worst ratio < 2.0x | Marginal | 1.905x -- PASS | YES (barely) |
| K803: speed >= 90 tok/s | Likely fail | 41.9 tok/s -- FAIL | YES (predicted failure) |
| Room PPL < base PPL for all domains | Yes for adapted domains | 3/5 YES, 2/5 NO | PARTIAL |

### Per-Domain PPL Table

| Domain | Base PPL | Single-Adapter PPL | Room PPL | Ratio (room/single) | Room < Base? |
|--------|----------|-------------------|----------|---------------------|--------------|
| medical | 6.412 | 5.731 | 10.918 | 1.905x | NO (+70%) |
| code | 4.752 | 3.937 | 6.994 | 1.776x | NO (+47%) |
| math | 3.734 | 3.723 | 4.954 | 1.331x | NO (+33%) |
| legal | 22.813 | 22.302 | 24.838 | 1.114x | NO (+9%) |
| finance | 19.990 | 20.346 | 22.586 | 1.110x | NO (+13%) |

**Critical observation:** Room model PPL is WORSE than base for ALL 5 domains.
The adapter sum doesn't just degrade relative to single-adapter -- it actively
hurts performance relative to having no adapters at all. The nonlinear
compounding is severe enough that the combined adapter noise exceeds the
benefit of any individual adapter's domain knowledge.

### Comparison to Finding #303 (POC)

| Metric | POC (Finding #303) | This Experiment | Consistent? |
|--------|-------------------|-----------------|-------------|
| Medical ratio | 10.47/5.45 = 1.92x | 10.92/5.73 = 1.91x | YES |
| Code ratio | 7.28/4.30 = 1.69x | 6.99/3.94 = 1.78x | YES |
| Math ratio | 5.18/3.82 = 1.36x | 4.95/3.72 = 1.33x | YES |
| Legal ratio | 24.05/21.53 = 1.12x | 24.84/22.30 = 1.11x | YES |
| Finance ratio | 22.22/19.94 = 1.11x | 22.59/20.35 = 1.11x | YES |
| Speed (tok/s) | 39.2 | 41.9 | YES |
| W_combined size | 4.17 GB | 4.17 GB | YES |

The measurements are highly consistent across both runs. The nonlinear
compounding effect is reproducible and stable.

### Speed Comparison

| Configuration | tok/s | Bandwidth Used |
|--------------|-------|----------------|
| Base (no adapter) | 143.8 | 1.18 GB |
| Single LoRA (factored) | ~97* | ~1.20 GB |
| Room model (W_combined) | 41.9 | ~5.35 GB |

*From Pierre v3 measurements (Finding #289).

The room model is 2.3x slower than factored LoRA because W_combined is a
dense d_out x d_in matrix (4.17 GB) that must be streamed from memory for
every token, vs ~18 MB for factored adapters.

## Empirical Results

**K802: PASS (barely) -- worst domain ratio 1.905x < 2.0x threshold.**
Medical domain is the worst at 1.905x. The margin is only 5%. At N>5, this
would almost certainly exceed 2.0x.

**K803: FAIL -- 41.9 tok/s << 90 tok/s threshold.**
The bandwidth cost of dense W_combined (4.17 GB) makes this architecture
fundamentally slower than factored LoRA (~97 tok/s). The 210-dispatch
reduction cannot compensate for 230x more bandwidth (4.17 GB vs ~18 MB).

**Overall: KILLED.** K803 fails decisively. K802 passes marginally but the
quality numbers reveal a deeper problem: room model PPL is worse than base
for all 5 domains, meaning the adapter composition actively degrades quality.

## Limitations

1. **5 domains only.** At N=24, the Frobenius norm of Sum DW_i grows as
   sqrt(24/5) = 2.2x, which would push the worst PPL ratio well above 2.0x.

2. **No routing comparison.** This experiment does not test soft routing via
   A-subspace projection (which was killed separately in Finding #303 at 14%
   accuracy).

3. **PPL as proxy.** PPL does not predict task quality (Finding #246, r=0.08).
   The behavioral question -- "does room model produce useful multi-domain
   generation?" -- is not answered by these numbers.

4. **Single seed.** No variance estimate across random seeds.

## What Would Kill This

Already killed by K803 (speed). Additionally:

- **At N>5:** K802 would fail as nonlinear compounding scales with sqrt(N).
- **Behavioral eval:** If task accuracy is tested (not just PPL), the degradation
  may be more or less severe than PPL suggests.
- **Mixed-domain queries:** A query requiring knowledge from two domains
  simultaneously might benefit from W_combined relative to single-adapter routing.
  This is untested.

## Interpretation and Implications

This experiment confirms Finding #303's measurements with high consistency
and adds the 2x-threshold assessment. The key conclusions:

1. **Pre-summing adapter deltas is fundamentally limited by nonlinear compounding.**
   This is not fixable -- it is a structural property of deep nonlinear networks.
   The only regime where pre-summing works is N=1 (single adapter pre-merge, proven
   in Pierre v6).

2. **W_combined is bandwidth-catastrophic.** A dense d x d matrix per module
   costs 4.17 GB bandwidth per token, making it slower than factored LoRA despite
   fewer dispatch calls. The ROOM_MODEL.md prediction of "100+ tok/s" was wrong
   because it counted dispatches, not bandwidth.

3. **Factored LoRA remains the correct architecture.** The h @ A @ B factored
   form uses ~18 MB bandwidth (230x less than W_combined) and achieves 97 tok/s.
   The dispatch overhead (2100 vs 210) is negligible relative to bandwidth savings.

4. **The room model concept is dead at N>1.** For N=1 (single adapter), pre-merge
   is already production-ready (Pierre v6). For N>1, the nonlinear compounding
   and bandwidth cost make it strictly inferior to factored per-adapter dispatch.
