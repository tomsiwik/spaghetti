# MATH.md — Pico calibration + TIES merge (combinatorial test)

## Hypothesis

Pico (B-space SVD calibration) and TIES (full-delta sign-aware merge)
attack interference at different geometric levels. **Are they orthogonal
(compose to gain on top of each other) or redundant (one absorbs the
other)?**

> **Does pico_then_ties beat both Pico+Fisher-Rao and TIES-alone?**

## Pipeline

1. Compute Pico calibration matrix `S` from stacked B-matrices (per layer).
2. Apply `B_t_calib = B_t @ S^T` per adapter.
3. Materialize `ΔW_t = scale · A · B_t_calib` per adapter.
4. Run TIES three-step (Trim → Sign-Elect → Disjoint mean) on materialized deltas.
5. Install via `_FusedDeltaLinear`.

## Pre-registered Kill Criteria

- **K1 (DECISION)** Pico+TIES avg ≥ Fisher-Rao avg + 3pp.
- **K2 (ARCH GAP)** Full-delta DARE avg − Pico+TIES avg ≤ 4pp.
- **K3 (BUDGET)** Preprocessing ≤ 35s (Pico ~5s + TIES ~30s).
- **K4 (ORTHOGONALITY)** Pico+TIES avg compared to max(Pico+Fisher-Rao, TIES-alone). PASS if Pico+TIES exceeds the better parent by ≥1pp (operations compose). FAIL = redundant.

## Verdict logic

| K1 | K4 | Outcome |
|----|----|---------|
| ✓ | ✓ | **SUPPORTED** — Pico and TIES are orthogonal; Pierre adopts the combination. |
| ✓ | ✗ | **SUPPORTED** with caveat — adopt the better parent alone; combination is redundant. |
| ✗ | * | **KILLED** — neither operation rescues the merge for Pierre's adapter set. |

## Honest gaps

- K4 depends on Pico+Fisher-Rao and TIES-alone results being available. Both will run before this experiment under the current queue order. K4 evaluation pulls from those `results.json` files at completion time.
- Code duplicates Pico's calibration step inline (rather than importing) to access per-adapter calibrated B's; the public API of Pico exposes only the merged result. Flagged as port choice.

## References

- Pico (arxiv 2604.16826) + TIES (arxiv 2306.01708)
- Sibling experiments: `exp_pierre_pico_calibration`, `exp_pierre_ties_full_delta`
