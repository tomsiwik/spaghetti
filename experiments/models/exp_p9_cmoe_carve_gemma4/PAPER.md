# CMoE Carving: Dense FFN to Shared + Routed Experts on Gemma 4 E4B

## Experiment

Port CMoE (arXiv:2502.04416) training-free expert extraction to Gemma 4 E4B
4-bit on MLX. Carve each dense FFN layer into 8 experts (1 shared + 7 routed),
activate 4/8 (50%) at inference for theoretical 2x speedup.

**Configuration:**
- Model: gemma-4-e4b-it-4bit (Gemma 4 E4B, 4-bit quantized)
- N_EXPERTS=8, N_SHARED=1, N_ACTIVATED=3 (routed), total active=4/8
- K_ACT=128 (top-128 activation markers, ~1.25% of D=10240)
- Calibration: 4 samples x 256 tokens (hardcoded diverse texts)
- Eval: 4 passages, seq_len=256 (raw text, not chat-formatted)

## Prediction vs Measurement

| Kill Criterion | Prediction (MATH.md) | Measurement | Status |
|---|---|---|---|
| K1342: PPL degradation <=5% | 1-3% | +219.5% (36,879 -> 117,821) | **FAIL** |
| K1343: Carving time <10min | 3-5 min | 13s | **PASS** |
| K1344: Speedup >=1.3x at 50% | 1.4-1.8x | 0.42x (29 vs 69 tok/s) | **FAIL** |

## Verification

Carving decomposition is mathematically exact (Theorem 1 confirmed):
- Layer 0 max diff: 2.86e-6 (all-experts sum vs dense, within FP16 tolerance)
- Layer 20 max diff: 1.79e-6

This confirms the neuron partitioning correctly decomposes the dense FFN.
Error comes entirely from routing (activating k < K groups).

## Root Cause Analysis

### Why PPL exploded (+219%)

Three compounding factors:

1. **Base PPL anomaly (37K):** Gemma 4 E4B is instruction-tuned. Evaluating on
   raw text (not chat-formatted) produces meaningless PPL. A well-calibrated
   model on matching data would have PPL 5-30. At PPL 37K, the model is already
   producing near-random predictions; carving error amplifies catastrophically
   in this regime.

2. **Dequantization confound:** Base PPL uses 4-bit quantized inference; carved
   model uses dequantized float16 weights. No control isolating dequantization
   error from carving error. The dequantization alone could shift PPL
   substantially.

3. **Tiny calibration:** 4 samples x 256 tokens = 1024 tokens total. The CMoE
   paper uses substantially more calibration data. With so few tokens, the
   activation statistics may not capture meaningful expert specialization
   patterns, leading to poor neuron groupings.

### Why speed decreased (0.42x)

The CMoELayer implementation runs ALL N_ROUTED=7 expert forward passes then
applies a binary mask — O(N) compute, not O(k). Combined with:
- 7 small matmuls replacing 1 large fused matmul (worse hardware utilization)
- Router overhead (2 matmuls + softmax + argpartition per layer)
- Additional memory traffic from expert parameter scattering

For genuine speedup, only activated experts should be computed. This requires
conditional computation (gather/scatter) which MLX does not efficiently support
for small expert sizes.

### Argpartition bug (fixed in REVISE)

Original code used `N_ACTIVATED=7` (diagnostic mode) with `mx.argpartition(kth=7)`
on axis of size 7 — out-of-bounds. Fixed: bypass routing when n_activated >= n_routed;
set N_ACTIVATED=3 for intended 50% activation. Results unchanged because the
fundamental issues above dominate.

## Impossibility Structure

CMoE carving on MLX faces a structural speed barrier: MLX's GPU dispatch model
favors large fused operations. Splitting one (D_inter x D_hidden) matmul into
N smaller (D_inter/N x D_hidden) matmuls loses fusion efficiency. Even with
perfect O(k) routing, the speedup ceiling on MLX for N=8, k=4 experts is likely
<1.0x due to kernel launch overhead dominating the 50% compute reduction.

The quality barrier requires: (a) matching eval data distribution to model type
(chat-formatted for IT models), (b) sufficient calibration data (32+ samples),
and (c) controlling for dequantization error. Without these controls, the PPL
measurement is not meaningful.

## Conclusion

**Status: KILLED.** 2 of 3 kill criteria fail. The carving decomposition is
mathematically exact (verified), but the experiment has three confounds that
make the PPL measurement uninterpretable, and a structural speed barrier on MLX
that makes >1.0x speedup unlikely with the current approach.

**For resurrection:** Would need (a) O(k) expert computation via MLX-native
conditional execution, (b) chat-formatted eval data matching IT model distribution,
(c) dequantization control experiment, (d) 32+ calibration samples. The
CMoE quality recovery experiment (exp_p9_cmoe_quality_recovery) may address (b-d)
but (a) remains a platform limitation.
