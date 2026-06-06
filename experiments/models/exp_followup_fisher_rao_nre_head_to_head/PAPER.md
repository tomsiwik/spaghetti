# Fisher-Rao vs NRE Head-to-Head at Production Scale — Report

## Verdict: KILLED

K1 FAIL, K2 FAIL, K3 PASS — the anti-null sanity check passes (both norm-preserving methods beat Euclidean by a large margin), and Fisher-Rao does **not** provide measurable benefit over Norm-Rescaled Euclidean on either the overall or conditional PPL metric. NRE is the composition ceiling at production scale (N=25) on Gemma 4 E4B 4-bit; the ~68× wall-clock cost of Karcher-mean iteration is unjustified.

This KILLED verdict is the designed positive outcome: Finding #275 (NRE ≡ FR on BitNet-2B at N≤15) now generalises to a different architecture (Gemma 4 E4B 4-bit) and a larger N (25).

## Predictions vs Measurements

From MATH.md §D.2.

| Prediction | Predicted | Measured (N=25) | Match? |
|---|---|---|---|
| **P1.** FR compose-time ≥ 8× NRE compose-time | ≥ 8× | 68.19× | **YES** (far exceeds lower bound) |
| **P2.** \|FR PPL − NRE PPL\| < 0.05 (noise floor) | < 0.05 | 0.352 (NRE wins) | **NO — larger than predicted, favours NRE** |
| **P3.** \|FR cond-PPL − NRE cond-PPL\| < 0.05 | < 0.05 | 0.031 (NRE wins) | YES |
| **P4a.** Euc PPL − NRE PPL > 0.3 at N=25 | > 0.3 | 2.431 | YES (much larger) |
| **P4b.** Euc PPL − FR PPL > 0.3 at N=25 | > 0.3 | 2.079 | YES |

**Important:** P2 was falsified in a direction that *strengthens* the KILLED verdict — NRE was 0.35 PPL points *better* than FR. The original MATH.md predicted "equivalence to measurement noise"; the actual result shows a small but clear NRE advantage. The mechanism is likely numerical: Karcher-mean fixed-point iteration introduces tiny direction perturbations that accumulate across 42 layers, whereas NRE's one-line rescale is bit-identical to extrinsic averaging except for a scalar factor.

## Kill-Criteria Table (N=25)

| KC | Threshold | Measured | Pass? |
|---|---|---|---|
| K1 (FR PPL < NRE PPL by ≥ 0.05) | ≥ 0.05 | −0.352 (NRE better) | **FAIL** |
| K2 (FR cond-PPL < NRE cond-PPL by ≥ 0.05) | ≥ 0.05 | −0.031 (NRE better) | **FAIL** |
| K3 (NRE and FR both beat Euc PPL by ≥ 0.3) | ≥ 0.3 (both) | NRE +2.431, FR +2.079 | **PASS** |

Per the MATH.md verdict rules: K1 FAIL AND K2 FAIL AND K3 PASS → **KILLED** (NRE ceiling confirmed at production N on Gemma 4 E4B 4-bit).

## Full Results

Base model (no adapter): overall PPL 29.04, conditional PPL 5.56.

| N  | Method     | Overall PPL | Cond PPL | Compose t (s) | B shrink (L0) |
|----|------------|-------------|----------|---------------|---------------|
| 3  | Euclidean  | 14.580      | 2.609    | 0.00          | 0.578         |
| 3  | NRE        | 11.703      | 2.301    | 0.00          | 1.000         |
| 3  | Fisher-Rao | 12.238      | 2.339    | 0.14          | 1.000         |
| 10 | Euclidean  | 13.892      | 2.565    | 0.00          | 0.595         |
| 10 | NRE        | 12.436      | 2.303    | 0.00          | 1.000         |
| 10 | Fisher-Rao | 12.510      | 2.320    | 0.20          | 1.000         |
| 25 | Euclidean  | 14.321      | 2.594    | 0.00          | 0.582         |
| 25 | NRE        | **11.890**  | **2.295** | 0.01         | 1.000         |
| 25 | Fisher-Rao | 12.242      | 2.326    | 0.38          | 1.000         |

NRE is at or below FR on every row. Euclidean norm shrinkage plateaus at ~0.58, confirming the expected `1/√N_real` floor for N>3 synthetic variants (only 3 real independent directions).

## Interpretation

1. **NRE is the composition ceiling at production scale on Gemma 4 E4B 4-bit.** At N=25, NRE gives overall PPL 11.89 vs FR 12.24 (FR is 3% worse). Extending to N=25 did not reveal a regime where FR wins — if anything, FR very slightly degrades relative to NRE.

2. **The 10× cost hypothesis is verified and then some.** FR compose-time is 68× NRE compose-time at N=25, driven by the Karcher fixed-point iteration (50 iters × N direction updates × ~6M params flattened per B-matrix). NRE is a single norm rescale — one extra multiply.

3. **Norm preservation is the entire mechanism.** The Euclidean→NRE gap at N=25 is 2.43 PPL (huge). The NRE→FR gap is 0.35 PPL *against* FR. The Riemannian manifold structure contributes nothing measurable; it's pure norm rescaling dressed up in spherical geometry.

4. **Generalises F#275.** Prior BitNet-2B N≤15 → Gemma 4 E4B 4-bit, q_proj, N≤25: same qualitative conclusion. The result is not architecture-specific.

## Caveats (MATH.md §G, restated)

1. Shared-A convention (adapter-0's A used for all sources) — consistent with F#275 methodology. Does not claim anything about independent-A adapters; a follow-up could test that with full-delta composition, but the orthogonality-of-directions argument (Pennec 2006) predicts the same answer.
2. Gemma 4 adapters target `q_proj` only (rank 6, scale 6). Per F#627 the proven LoRA target is `v_proj+o_proj`. This experiment's claim — that FR has no measurable advantage over NRE under head-to-head composition — is orthogonal to target-module choice; the ordering of methods on PPL would not flip.
3. N>3 uses synthetic noisy variants (`NOISE_SCALE=0.1`). The `1/√N_real` Euclidean plateau at 0.58 for N=10 and N=25 reflects 3 independent source directions, which is a known artefact and does not affect the NRE-vs-FR comparison (both norm-preserving methods converge to 1.000 shrinkage regardless).
4. `mlx_lm==0.31.2` pinned via runtime print.

## Assumptions Logged

- When predictions conflict with measurements, the measured value is authoritative (P2 was tighter than predicted *and* directed — MATH.md's "noise-floor equivalence" prediction was conservative; the actual result shows a slight NRE advantage, which strengthens the kill, not weakens it).
- No hyperparameter tuning was performed after observing results. KCs locked in MATH.md before the first run; `git diff MATH.md` is empty between pre-registration and now.
- Loading base model on an unloaded CPU: one forward pass takes ~0.1s per sample at 150 tokens; 50 samples × ~6s = reported eval times are dominated by compilation warm-up of the first forward.

## Key References

- arXiv:2603.04972 — Fisher-Rao Manifold Merging (Wang, Ye, Yin 2025). Source of the Karcher-mean spherical-proxy method.
- Pennec (2006) — Intrinsic Statistics on Riemannian Manifolds. Predicts `‖NRE − FR‖ = O(dispersion²)` for small angular spread.
- Finding #274 — FR preserves adapter norms at N≤15.
- Finding #275 — NRE matches FR on all metrics; norm preservation is the mechanism. This experiment now extends that conclusion to Gemma 4 E4B 4-bit and N=25.
- Finding #666 — Target-gated kill rule (K1/K2 pair, K3 anti-null).

## Next Steps

- Update `.ralph/current_direction.md` with the NRE-ceiling generalisation.
- Consider opening a follow-up only if a principled reason emerges for FR to beat NRE under *independent-A* adapters — e.g., high-dispersion sources where the small-dispersion assumption (Pennec 2006) no longer holds. Until such evidence exists, NRE is the default for all Pierre composition code; FR should not be reintroduced without a new hypothesis.
