# PAPER — exp_sigreg_composition_monitor

**Verdict:** PROVISIONAL — design-locked, empirical deferred.
**Is_smoke:** false. **all_pass:** false. No KCs measured; no falsifier fired.

## Prediction vs measurement

| # | Prediction (from MATH.md §5) | Measurement | Status |
|---|---|---|---|
| P1 | Spearman r(S(N), −A(N)) > 0.5 across N ∈ {1,2,4,8,16,24} | not_measured | K1779 not_measured |
| P2 | N* (SIGReg crosses τ) < N** (A drops 5pp); lead ≥ 10% training steps | not_measured | K1780 not_measured |
| P3 | S(Z_1) rejection rate < 10% at N=1 (healthy ground-truth) | not_measured | K1781 not_measured |

## Why PROVISIONAL (not KILLED, not SUPPORTED)

Three preconditions for empirical verification are unmet, and remediating them
exceeds the researcher-hat 30-min cap:

1. **Adapter inventory.** KC K1779 requires N ∈ {1,2,4,8,16,24}, i.e. ≥24
   trained Gemma 4 E4B v_proj+o_proj r=6 adapters. F#627 trained **q_proj** r=6
   on 3 domains — wrong target module, wrong count. Zero matching adapters on
   disk today. Training 24 new adapters = ~10h sequential (M5 Pro 48GB cannot
   parallelise Gemma 4 E4B training).
2. **Composition harness.** No reusable W_comp(S) = W + Σ_{i∈S} Δ_i harness
   exists in the repo that emits an activation hook at layer 21 with MLX
   memory discipline. F#571 K1690 proved bf16 additive exactness for N=1 hot-
   merge only; N>1 was killed 4×, so the research surface has not invested in
   reusable N>1 composition plumbing.
3. **SIGReg activation capture.** Epps-Pulley on M=1024 projections with K=32
   Gauss-Hermite quadrature is ~150-line MLX code not yet in the repo;
   F#691/F#682 have it design-locked but not implemented.

Total remediation: ~14-20h; exceeds cap by 28-40×. PROVISIONAL is the honest
verdict per researcher guardrail (step 6.4) and mirrors the F#682/F#691
precedent for novel-mechanism designs awaiting an `_impl` follow-up.

## Novelty and fit with F#682 / F#691

SIGReg as a collapse detector now has three pre-registered surfaces:

| Finding | Failure axis | Collapse mode |
|---|---|---|
| F#682 | layer-wise (single adapter, layer L) | predictor→constant |
| F#691 | cross-depth (RDT iterates {h_d}) | h_d = h_{d+1} (idempotent loop) |
| **this exp** | **N-composition (N=1..24 adapters)** | **cross-term compound (Room-Model-style)** |

Three geometrically distinct surfaces. The N-axis is novel within this repo:
no prior experiment measures SIGReg on composed-adapter activations.

## Assumptions (for future `_impl`)

- "Healthy" composition configs (K1781) are defined as configs where task
  accuracy is within 2pp of the single-adapter ceiling (F#627 baseline).
  Assumption: N=1 is the primary healthy case; N>1 with orthogonal domains
  (if any exist) may also qualify — will be enumerated at `_impl` claim.
- Spearman r (K1779) computed over 6 grid points {1,2,4,8,16,24}; n=6 is
  minimal but yields p<0.05 at r=0.829 (one-tailed). KC r > 0.5 is
  intentionally permissive to avoid statistical underpowering.
- LORA_SCALE=6 (F#328/F#330 safe); ADAPTER_RANK=6 (matches F#627); MLX memory
  discipline per F#673 (mx.eval + mx.clear_cache between compositions).
- Base model: mlx-community/gemma-4-e4b-it-4bit (matches F#627).

## Unblock path (recommended order for `_impl`)

1. Train 6 domain adapters first (the bare minimum for meaningful Spearman on
   the N grid we can afford: N ∈ {1,2,3,4,5,6}); this also delivers reusable
   v_proj+o_proj inventory for other downstream experiments.
2. Rescope KC K1779 as an addendum: Spearman r on N ∈ {1,2,3,4,5,6} (grid
   adjustment is a **KC modification** and therefore requires pre-registering
   in a v2 experiment — do not silently edit K1779).
3. Build composition harness (1 day).
4. Implement SIGReg capture (~half day, lift from F#691 design).
5. Run + correlate.

## Status

`experiment complete exp_sigreg_composition_monitor --status provisional`
issued; Finding filed as design-locked novel-mechanism entry (6th in drain
window).
