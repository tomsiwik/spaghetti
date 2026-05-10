# MATH.md — TIES applied directly to B-matrices (B-only ablation)

## Hypothesis

The research agent's note on TIES claimed: *"TopK-by-magnitude on B alone
wouldn't carry semantic meaning the way it does on ΔW entries — the
magnitudes of B's entries don't directly correspond to output influence
(A scales them)."*

This is a falsifiable claim. **Test it.** If TIES-on-B works anyway, we
have a B-only TIES variant compatible with Pierre's existing architecture.
If it fails (likely), we have empirical evidence that the full-delta
materialization step is what makes TIES meaningful.

> **Does TIES applied directly to B-matrices (no full-delta materialization)
> beat Fisher-Rao?**

## Algorithm

Same TIES 3-step but operating on B-matrices (`r × d_out`) instead of
full deltas (`d_in × d_out`):

1. Stack: `T = stack([flatten(B_t)])` shape `(K, r·d_out)`.
2. Trim: TopK-by-magnitude per row, keep `keep_frac · r·d_out`.
3. Elect sign + disjoint merge as in TIES.
4. Reshape back to `(r, d_out)` and (optionally) rescale to mean source norm.

Returns a B-dict — plugs into Pierre's existing `PoLARLinear` storage.

## Pre-registered Kill Criteria

- **K1 (DECISION)** TIES-B avg ≥ Fisher-Rao avg + 3pp.
- **K2 (CONSISTENCY)** TIES-B compared to TIES-full-delta — within 5pp expected if claim is wrong; >5pp gap confirms claim.
- **K3 (BUDGET)** Preprocessing ≤ 5s (cheap — operates only on B).
- **K4 (BIPOLAR)** Two informative outcomes:
  - PASS K1 → research agent's claim is wrong, novel B-only TIES variant.
  - FAIL K1 → claim confirmed, full-delta is necessary for TIES.

## Verdict logic

| K1 | Outcome | Interpretation |
|----|---------|----------------|
| ✓ | **SUPPORTED** | Architectural-honest TIES variant for Pierre. |
| ✗ | **KILLED, informative** | TIES requires full-delta semantics; B-space TIES is meaningless. Confirms a structural property of LoRA factoring. |

## References

- TIES-Merging paper (arxiv 2306.01708)
- Sibling experiment: `exp_pierre_ties_full_delta` (full-delta TIES — for K2 comparison)
- Prior: `exp_pierre_dare_b_vs_fisher_rao` (showed B-space DARE fails for similar reasons)
