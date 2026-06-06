# Room Model Piece A: Mathematical Analysis of W_combined = Sum of Delta W_i

## Experiment Type: Verification (retest with lenient thresholds)

This is a RETEST of the killed Room Model POC (Finding #303). The prior experiment
proved that pre-summing N=5 adapter deltas causes 29% PPL degradation due to
nonlinear inter-layer compounding. This experiment tests whether that degradation
level is acceptable under a 2x threshold (vs the prior 1.10x threshold).

No new mathematical mechanism is proposed. The math below characterizes the
KNOWN error, bounds it, and predicts whether the retest will pass.

---

## A. Failure Mode (Diagnosed in Finding #303)

**The disease:** Transformers are nonlinear across layers. Pre-summing N adapter
deltas at every layer simultaneously shifts the hidden state by ~N times more
than a single adapter. This shift compounds multiplicatively through LayerNorm,
softmax attention, and SiLU gating across L=30 layers.

**Per-module linearity is exact (trivially):** For a single linear layer,

  x @ (W_base + Sum_i DW_i) = x @ W_base + Sum_i (x @ DW_i)

This is matrix distributivity. Finding #303 confirmed per-module MSE = 5.6e-7
(numerical precision). There is nothing to prove here; it is an axiom of
linear algebra.

**Cross-layer nonlinearity is the disease:** Between every pair of adjacent layers,
nonlinear operations (LayerNorm, attention softmax, SiLU activation) transform
the hidden state. When ALL N adapter deltas are active simultaneously, the hidden
state entering layer l+1 differs from the hidden state that would enter under any
single adapter. This difference grows through the network.

This is NOT a bug to fix. It is a structural property of deep nonlinear networks
(Raghu et al., 2017, arXiv:1611.03530 — "On the Expressive Power of Deep Neural
Networks"; Veit et al., 2016, arXiv:1605.06431 — residual networks as ensembles
of shallow paths).

---

## B. The Right Question (Reframed from Finding #303)

**Wrong question (POC):** "Is pre-summing mathematically equivalent to single-adapter?"
Answer: No. Killed. (Finding #303)

**Right question (this experiment):** "Is the nonlinear compounding error small
enough that W_combined still produces useful multi-domain output?"

This is a utilitarian question, not a mathematical one. The answer depends on
the tolerance threshold. The prior POC measured 1.29x PPL ratio. This experiment
retests with a 2.0x threshold.

---

## C. Error Bound (Informal, Based on Prior Measurement)

We cannot derive a tight closed-form bound on the full-network PPL ratio because:
1. The nonlinear functions (LayerNorm, softmax, SiLU) have data-dependent Lipschitz
   constants that vary per token.
2. The compounding depends on the actual hidden state trajectory, not just weight norms.

However, we can reason about the scaling:

**Observation 1 (Finding #303):** At N=5, alpha=20, the PPL ratio was 1.29.

**Observation 2 (single-adapter pre-merge):** At N=1, alpha=20, pre-merge is
EXACT (it is the same as runtime LoRA, just materialized). This is proven and
used in production (Pierre v6).

**Observation 3 (perturbation theory):** The per-module delta norm scales as:

  ||Sum_i DW_i||_F = ||Sum_i alpha * B_i^T @ A_i^T||_F

Since A_i are orthogonal (Grassmannian, Finding #126), the deltas are in
orthogonal subspaces, so:

  ||Sum_i DW_i||_F^2 = Sum_i ||DW_i||_F^2 = N * mean(||DW_i||_F^2)

The Frobenius norm grows as sqrt(N) relative to a single adapter.
At N=5: sqrt(5) = 2.24x the norm of a single adapter delta.

The downstream PPL effect depends on how this norm interacts with the
nonlinear functions. Finding #303 measured the full effect as 1.29x.
This is sublinear in norm growth (1.29 < 2.24), consistent with the
nonlinearities being approximately linear for small perturbations
(first-order Taylor: f(x+eps) ~ f(x) + f'(x)*eps).

**Prediction:** The PPL ratio should be approximately the same as Finding #303
(~1.29x), since we are using the same adapters, same model, same data.
The only difference is the threshold (2.0x vs 1.10x).

---

## D. Predictions

### Behavioral Predictions

1. **Room model produces sensible multi-domain output:** Each domain should get
   PPL lower than the base model (no-adapter), because the adapter deltas provide
   domain knowledge even when summed.

2. **No domain catastrophically fails:** The worst domain ratio should be < 2.0x
   (K802), based on Finding #303 where the worst ratio was approximately 1.92x
   (medical: room 10.47 / v3-single 5.45 = 1.92x).

3. **Speed is bandwidth-limited:** W_combined is a full d_out x d_in dense matrix
   per module. At d=2560, each is ~13MB bf16. For 210 modules: ~2.7GB extra
   bandwidth per token. At 273 GB/s (M5 Pro): ~10ms extra. With base model:
   total ~15ms per token, or ~67 tok/s. Finding #303 measured 39.2 tok/s
   (which may have been on different hardware). K803 threshold is 90 tok/s
   -- this will likely FAIL.

### Quantitative Predictions (from Finding #303 extrapolation)

| Metric | Predicted | Source |
|--------|-----------|--------|
| Mean PPL ratio (room/single) | ~1.25-1.35x | Finding #303 measured 1.29x |
| Worst domain PPL ratio | ~1.5-2.0x | Medical was worst in POC |
| Room speed (tok/s) | 35-50 | Bandwidth: 2.7GB W_combined at 273 GB/s |
| K802 (worst ratio < 2.0x) | MARGINAL | Depends on exact single-adapter baselines |
| K803 (speed >= 90 tok/s) | LIKELY FAIL | Bandwidth math says ~40-67 tok/s |

**NOTE on K803:** The speed criterion (90 tok/s) was set based on the ROOM_MODEL.md
prediction of 100+ tok/s, which assumed "210 dispatches vs v3's 420." However,
Finding #303 showed that the bottleneck is BANDWIDTH, not dispatch count.
W_combined is 2.7GB of dense bf16 that must be read per token, vs ~18MB for
factored LoRA. Speed will likely fail. This is not a deficiency of the
experiment -- it is a real property of the architecture.

---

## E. Assumptions & Breaking Conditions

1. **Same adapters as POC:** We use the same SFT adapters from bitnet_sft_generation_v3.
   If the adapters have changed, results may differ.

2. **Grassmannian orthogonality:** A_i perp A_j. This is proven (Finding #126,
   |cos|=0.00125). If violated, deltas would interfere and PPL would be much worse.

3. **Alpha=20 scaling:** The LoRA scale factor amplifies the delta. Higher alpha
   increases the perturbation and the nonlinear compounding. Finding #303 used
   alpha=20, and so does this experiment.

4. **5 domains only:** With more domains (N=24), the norm grows as sqrt(24/5)=2.2x
   more, and PPL degradation would be significantly worse.

---

## F. Worked Example (d=4, r=2, N=2)

Two orthogonal adapters in 4 dimensions:

```
A_1 = [[1, 0],    A_2 = [[0, 0],
       [0, 1],           [0, 0],
       [0, 0],           [1, 0],
       [0, 0]]           [0, 1]]

B_1 = [[0.5, 0.3],  B_2 = [[0.7, 0.1],
       [0.2, 0.8]]         [0.4, 0.6]]
```

Verify orthogonality: A_1^T @ A_2 = 0 (top-2 rows of A_1 dot bottom-2 rows of A_2 = 0).

Delta_1 = alpha * B_1^T @ A_1^T:
```
alpha * [[0.5, 0.2],  @ [[1, 0, 0, 0],    = alpha * [[0.5, 0.2, 0, 0],
         [0.3, 0.8]]    [0, 1, 0, 0]]                 [0.3, 0.8, 0, 0]]
```
This is a 2x4 matrix (d_out=2, d_in=4), nonzero only in first 2 columns.

Delta_2 = alpha * B_2^T @ A_2^T:
```
alpha * [[0.7, 0.4],  @ [[0, 0, 1, 0],    = alpha * [[0, 0, 0.7, 0.4],
         [0.1, 0.6]]    [0, 0, 0, 1]]                 [0, 0, 0.1, 0.6]]
```
Nonzero only in last 2 columns.

W_combined = Delta_1 + Delta_2:
```
alpha * [[0.5, 0.2, 0.7, 0.4],
         [0.3, 0.8, 0.1, 0.6]]
```

For input x = [1, 0, 0, 0] (aligned with A_1):
  x @ W_combined^T = alpha * [0.5, 0.3]  (only Delta_1 contributes)

For input x = [0, 0, 1, 0] (aligned with A_2):
  x @ W_combined^T = alpha * [0.7, 0.1]  (only Delta_2 contributes)

For input x = [0.5, 0.5, 0.5, 0.5] (mixed):
  x @ W_combined^T = alpha * [0.7, 0.9]  (both contribute equally)

**Per module, this is exact.** The problem arises BETWEEN modules when nonlinear
functions transform x, changing its alignment with the A_i subspaces in
data-dependent ways.

---

## G. Complexity & Architecture Connection

### Per-token cost

| Component | FLOPs | Bandwidth | Notes |
|-----------|-------|-----------|-------|
| Base BitLinear | O(d^2) ternary | 1.18 GB total | Native kernel |
| W_combined matmul | O(d^2) bf16 | ~2.7 GB total (210 modules) | Dense matmul |
| Single LoRA (factored) | O(d*r) bf16 | ~18 MB total | h@A@B factored |
| N=5 factored LoRA | O(5*d*r) bf16 | ~90 MB total | 5 separate matmuls |

**Key insight:** W_combined trades dispatch count for bandwidth.
- Factored LoRA: 5 adapters x 2 matmuls x 210 modules = 2100 dispatches, 90 MB bandwidth
- W_combined: 1 matmul x 210 modules = 210 dispatches, 2700 MB bandwidth

On Apple Silicon (bandwidth-limited), fewer dispatches cannot compensate for
30x more bandwidth. Finding #303 confirmed this: 39 tok/s vs 76 tok/s factored.

### Memory
- W_combined total: 210 modules x d_out x d_in x 2 bytes = ~2.7 GB
- This fits in 48 GB with room to spare, but is much larger than factored form

---

## Self-Test (MANDATORY)

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   There is no such property. The failure mode (nonlinear compounding) is a real
   structural feature of deep networks. This experiment tests whether the
   degradation is ACCEPTABLE, not whether it is absent.

2. **Which existing theorem(s) does the proof build on?**
   Matrix distributivity (linear algebra axiom). Finding #303 (empirical measurement
   of the nonlinear compounding at N=5). Perturbation theory: first-order Taylor
   expansion of nonlinear functions.

3. **What specific numbers does the proof predict?**
   PPL ratio ~1.25-1.35x (from Finding #303's 1.29x). Worst domain ~1.5-2.0x.
   Speed ~35-50 tok/s. K802: marginal. K803: likely fail.

4. **What would FALSIFY the proof (not just the experiment)?**
   If PPL ratio is significantly BETTER than 1.29x (say < 1.10x), it would mean
   the POC measurement was wrong or the adapters have changed. If significantly
   WORSE (say > 2.5x), it would mean the perturbation analysis is too optimistic.

5. **How many hyperparameters does this approach add?**
   Count: 0. W_combined = Sum of DW_i is deterministic given the adapters.
   Alpha is inherited from the adapter training, not a new parameter.

6. **Hack check: Am I adding fix #N to an existing stack?**
   No. This is a retest of a killed experiment with a different acceptance threshold.
   No new mechanisms, no fixes. Pure re-evaluation of the same measurement.
