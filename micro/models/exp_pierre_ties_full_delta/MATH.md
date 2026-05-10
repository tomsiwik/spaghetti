# MATH.md — TIES-Merging on shared-A materialized deltas

## Hypothesis

TIES-Merging (arxiv 2306.01708) is the established structured composition
method that handles cross-task interference via three operators:
**T**rim (TopK by magnitude), **I**nterference-resolution (Elect Sign),
**E**lection-aware **S**core merging (Disjoint mean of sign-agreed entries).

Where DARE drops randomly, TIES drops principled (smallest magnitude).
Where naive averaging cancels conflicting signs, TIES votes per-cell.

> **Does TIES on shared-A materialized deltas beat Fisher-Rao, and how
> does it compare to ACE (covariance-weighted) and DARE (random drop)?**

## Adaptation to Pierre's shared-A

Same pattern as ACE/OrthoMerge: materialize `ΔW_t = scale · A · B_t` at
compose time, run TIES on full deltas, install via `_FusedDeltaLinear`.

## Algorithm (verbatim from paper)

For each (layer, module) key:

1. Materialize `ΔW_t` per adapter, stack flat: `T = stack([flatten(ΔW_t)])` → `(K, D)`.
2. **Trim**: per-row TopK-by-magnitude, keeping `keep_frac · D` entries (rest zeroed).
3. **Elect Sign**: `γ = sign(Σ_t T_trim_t)` — per-cell majority vote.
4. **Disjoint Merge**: for each cell `i`, average only `t` with `sign(T_trim_t[i]) == γ[i]`.
5. Reshape back to `(d_in, d_out)`, install via fused-delta wrapper.

## Pre-registered Kill Criteria

- **K1 (DECISION)** TIES avg ≥ Fisher-Rao avg + 3pp
- **K2 (ARCH GAP)** Full-delta DARE avg − TIES avg ≤ 4pp
- **K3 (BUDGET)** Preprocessing ≤ 30s
- **K4 (SANITY)** With `keep_frac=1.0` and post-trim sign agreement, TIES degenerates to weighted mean (within 2pp of Fisher-Rao under uniform weights). *Verified at run-time as a smoke step before main eval.*

## Verdict logic

| K1 | K2 | Outcome |
|----|----|---------|
| ✓ | ✓ | **SUPPORTED** — TIES is the right merge for Pierre. |
| ✓ | ✗ | **SUPPORTED** with caveat — adopt; gap not fully closed. |
| ✗ | * | **KILLED** — TIES doesn't transfer; revisit Pico/ACE. |

## References

- TIES-Merging paper (arxiv 2306.01708)
- Reference impl: prateeky2806/ties-merging
- Implementation spec: research agent verified
