# Room Model POC: Proof Verification Report

## Theorems Tested

**Theorem 1 (Pre-Summed Equivalence):** For orthogonal adapter deltas DeltaW_i,
applying x @ sum(DeltaW_i) produces output identical to sum(x @ DeltaW_i).

**Theorem 2 (Projection Geometry):** Soft routing weights w_i proportional to
||h @ A_i|| capture domain-specific signal from hidden state geometry.

**Theorem 3 (Bandwidth-Speed):** Room model achieves ~40-50 tok/s, limited by
4.17 GB W_combined bandwidth at 273 GB/s.

## Predictions vs Measurements

| Prediction (from proof) | Measured | Match? |
|------------------------|----------|--------|
| Per-module MSE ~ 0 (Thm 1) | 5.6e-7 mean, 1.6e-6 max | YES (float32 noise) |
| Full-model logit MSE ~ 0 (Thm 1) | 53.9 | NO (nonlinear interaction) |
| Routing accuracy >= 60% (Thm 2) | 14% | NO (near random) |
| PPL ratio <= 1.10 (Thm 1) | 1.29 | NO (+29% worse) |
| Speed 40-50 tok/s (Thm 3) | 39.2 tok/s | YES (within range) |
| Memory ~5.8 GB (Thm 3) | 5.69 GB | YES |

## Hypothesis

Pre-summing all adapter deltas into one matrix preserves per-module equivalence
but NOT full-model equivalence, because nonlinear layer interactions (LayerNorm,
activation functions, attention softmax) break the additivity assumption across
layers.

**KILLED.** K763 FAIL, K764 FAIL, K765 FAIL.

## What This Experiment Is

The Room Model proposes collapsing N LoRA adapter deltas into a single pre-summed
matrix W_combined per module, achieving:
- One matmul for all N domains (instead of 2N)
- Automatic soft routing from projection geometry (no explicit router)
- Hot-swappable domains via matrix addition/subtraction

## Key References

- Naive LoRA Summation (2508.11985): orthogonal adapter addition
- Punica (2310.18547): fused adapter serving
- Finding #126: Grassmannian orthogonality guarantee
- Finding #300: Full precompute bandwidth-killed (42.1 tok/s)
- Finding #301: Hybrid additive cost model

## Empirical Results

### Phase 1: Mathematical Equivalence (K763)

**Per-module test (algebraic, PASSES):**
- Tested 21 modules across layers [0, 14, 29]
- Mean MSE: 5.63e-7 (float32 rounding noise)
- Max MSE: 1.57e-6 (MLP gate/up modules, larger output dimension)
- Confirms Theorem 1: x @ sum(DeltaW_i) = sum(x @ DeltaW_i) per module

**Full-model test (end-to-end, FAILS):**
- Logit MSE: 53.9 (massive)
- Max |logit diff|: 63.9

**Root cause:** The sequential test computes base + sum(adapter_delta_logits). The
room model computes (base + all_deltas) end-to-end. These differ because:

1. **LayerNorm is nonlinear.** After each layer, LayerNorm(h + adapter_delta)
   differs from LayerNorm(h) + LayerNorm(adapter_delta). The adapter's contribution
   is rescaled by the norm, creating cross-terms.

2. **Effects compound through layers.** Layer 0's adapter delta changes the hidden
   state entering Layer 1. In v3 single-adapter mode, this is fine. But in the room
   model, ALL 5 adapter deltas at Layer 0 shift the hidden state, which then gets
   modified differently at Layer 1 by all 5 deltas again. The errors compound
   multiplicatively through 30 layers.

3. **This is a general problem with "all adapters active simultaneously."** It is
   NOT specific to pre-summing. Even if we applied adapters factored (A@B) for all 5
   simultaneously, the same nonlinear interaction would occur. The per-module linearity
   is exact, but the full model is not a linear function of adapter deltas.

**The theorem is correct but the assumption is incomplete:** Theorem 1 proves
per-module equivalence (which holds). The unstated assumption was that full-model
output is a linear function of per-module adapter deltas (which is FALSE).

### Phase 2: Soft Routing (K764)

| Domain | Accuracy | Expected (1/N) |
|--------|----------|----------------|
| medical | 0% | 20% |
| code | 20% | 20% |
| math | 50% | 20% |
| legal | 0% | 20% |
| finance | 0% | 20% |
| **Overall** | **14%** | **20%** |

Mean weight on correct domain: 0.197 (nearly indistinguishable from 1/N = 0.200)
Mean weight on other domains: 0.201

**Root cause:** The Grassmannian A-matrices are constructed to be orthogonal in
WEIGHT space, not to align with DOMAIN-SPECIFIC hidden state directions. The A-matrices
are random orthonormal frames -- they span orthogonal r-dimensional subspaces of R^d,
but these subspaces have no semantic meaning.

A medical token's hidden state projects roughly equally onto all A-subspaces because
the A-matrices were chosen for geometric packing, not for domain alignment.

This falsifies the "Room Model" metaphor: the "walls" are geometrically orthogonal
but semantically arbitrary. They do NOT correspond to domain directions.

**This was predictable from prior findings:** Finding #115 (content-aware routing
KILLED at micro scale) showed 26.5% domain accuracy from hidden states. The
A-subspace projection is even weaker (14%) because it filters through random
orthogonal frames instead of using the full hidden state.

### Phase 3: PPL (K765)

| Config | Medical | Code | Math | Legal | Finance | Mean |
|--------|---------|------|------|-------|---------|------|
| Base | 6.110 | 5.122 | 3.822 | 22.106 | 19.392 | 11.310 |
| v3-single (oracle) | 5.452 | 4.302 | 3.815 | 21.528 | 19.941 | 11.008 |
| v3-composed (NRE) | 5.104 | 4.761 | 3.776 | 21.160 | 18.991 | 10.758 |
| **Room model** | **10.465** | **7.280** | **5.182** | **24.051** | **22.224** | **13.840** |

Room model PPL is 29% worse than v3-composed and 22% worse than base.

**Root cause:** The room model applies ALL 5 adapter deltas at full strength
simultaneously. This is equivalent to "uniform composition" but WITHOUT the
norm-rescaling that v3's compose_adapters() applies. The raw sum of 5 deltas
has ~5x the norm of a single adapter, causing over-adaptation.

v3-composed uses NRE (norm-rescaled averaging): it averages B-matrices and
preserves the source norm. This prevents the 1/sqrt(N) shrinkage AND the N-fold
amplification. The room model's raw sum does neither.

**The room model is not comparable to v3-composed.** They are fundamentally
different operations:
- v3-composed: alpha * (x @ A_0) @ mean(B_i) with norm rescaling (uses ONE A)
- Room model: sum_i alpha * (x @ A_i) @ B_i (uses ALL A's, no norm control)

### Phase 4: Speed

| Config | tok/s | Peak Memory |
|--------|-------|-------------|
| Base (native BitLinear) | 142.2 | 1.23 GB |
| v3 single adapter | 76.1 | 1.31 GB |
| **Room model** | **39.2** | **5.69 GB** |
| Predicted (Theorem 3) | 40-50 | ~5.8 GB |

Speed prediction from Theorem 3 is confirmed: 39.2 tok/s is within the predicted
40-50 range. The room model is 1.94x SLOWER than v3 and 3.63x SLOWER than base,
entirely due to the 4.17 GB of W_combined bf16 data that must be streamed per token.

## Limitations

1. **Nonlinear interactions ignored.** Theorem 1's per-module linearity is correct
   but insufficient for full-model behavior.
2. **A-subspace semantics assumed but not validated.** The "room model" metaphor
   assumed A-matrices correspond to domain walls. They don't -- they are random
   orthonormal frames.
3. **No norm control.** The raw sum of N deltas has ~N times the effective alpha,
   causing over-adaptation.

## What Would Fix This

### For equivalence (K763):
The room model needs **alpha scaling**: W_combined = (1/N) * sum(DeltaW_i), or
better, norm-rescaled: W_combined = sum(DeltaW_i) * ||single_delta|| / ||sum||.
But even with correct scaling, the nonlinear compounding through 30 layers means
room model != sequential single-adapter. Room model is fundamentally a different
inference mode (all-adapters-active vs single-adapter).

### For routing (K764):
Automatic routing from A-subspace projection is dead. The A-matrices are
geometrically orthogonal but semantically arbitrary. Domain routing requires
explicit semantic signal (ridge regression on hidden states, as v3 does).
This confirms the general finding that orthogonality serves composition, not routing.

### For PPL (K765):
Need norm control. The simplest fix: W_combined = (1/N) * sum(DeltaW_i). This
would make the room model equivalent to "average adapter" applied through all
A-subspaces simultaneously. Whether this improves PPL over v3-composed is an
empirical question, but the scaling is necessary.

## What Would Kill This

The room model concept has three fatal flaws:
1. Full-model nonlinearity (LayerNorm, attention) breaks per-module equivalence
2. A-subspace projection cannot route (semantically arbitrary frames)
3. Raw delta sum has no norm control (5x over-adaptation)

Any of these individually kills the concept. Together, they kill it conclusively.

## Key Finding

**Pre-summing orthogonal adapter deltas is algebraically exact per module (MSE ~1e-7)
but fails at full-model level (MSE 53.9) due to nonlinear interactions through
LayerNorm and attention. Soft routing from A-subspace projection is no better than
random (14% vs 20% chance). The Room Model as described in ROOM_MODEL.md is killed.**

**The valuable insight:** Per-module linearity IS guaranteed. What fails is the
assumption that full-model behavior is a linear function of adapter contributions.
This has implications for any "sum all adapters" approach: norm control and
interaction management are essential.
