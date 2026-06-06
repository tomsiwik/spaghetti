# PAPER.md — exp_g4_per_layer_cos_baseline

## Verdict: KILLED (preempt, F#666-pure KC-structural)

This experiment was preempt-killed before any MLX code was written. The kill is **structural**: the pre-registered kill-criterion set consists of a single proxy KC (K1856: per-layer cos-sim variance) with no paired target-metric KC. Under F#666 (guardrail 1007), proxy-alone verdicts are tautological — KILL requires proxy+target BOTH to fail, SUPPORTED requires BOTH to pass. Therefore no empirical outcome can produce a valid verdict.

**Distinguishing feature:** no parent dependency (`depends_on: []`). This is NOT an F#669-family preempt (parent-target-unverified). It is a standalone F#666-pure structural violation — the first of its sub-case in drain window.

## Prediction vs measurement

| KC    | Prediction                                                                           | Kind  | Measurement  | Verdict   |
| ----- | ------------------------------------------------------------------------------------ | ----- | ------------ | --------- |
| K1856 | Per-layer cos-sim variance across 100 diverse prompts < 0.02 (routing uniform) | proxy | not measured | untested  |

Row is "not measured" because measurement would yield a proxy-only verdict, which F#666 explicitly forbids. Running the measurement and marking K1856 PASS/FAIL without a paired target KC would constitute antipattern-t (producing a structurally invalid verdict) or antipattern-u (post-claim KC mutation if a target were added after the fact).

## Secondary structural defects

Per `experiment get exp_g4_per_layer_cos_baseline`:

1. `success_criteria: []` — no SUPPORTED-condition declared; independent defect that also blocks SUPPORTED verdict.
2. `references: []` — violates guardrail 1002 (cite arxiv or prior finding).
3. `platform: null` — unset; MATH.md §0 discipline violated.

Each of these independently warrants a pre-registration fix.

## Assumptions

- No attempt was made to substitute a runnable proxy-only verdict. Measuring cos-sim variance and marking K1856 pass/fail in isolation would produce a tautological outcome per F#666.
- No attempt was made to invent a target-metric KC post-claim (antipattern-u).
- The Hedgehog-axis experiments (F#683/F#684/F#696/F#697) that this baseline was meant to "establish" have already reached PROVISIONAL without it, so the downstream gating value is minimal. Re-scoping as a Hedgehog-family sibling may be preferable to resurrecting the malformed pre-reg.

## Pattern taxonomy (drain-window context)

| Sub-case                                      | Parent status       | KC-structure        | Finding |
| --------------------------------------------- | ------------------- | ------------------- | ------- |
| F#669 classic (parent-unverified, F#666-ok)   | PROVISIONAL         | target-gated        | F#669 / F#687 / F#699 |
| F#669 + F#666 compound                        | PROVISIONAL         | proxy-only          | F#698   |
| **F#666-pure (this)**                         | **none / ok**       | **proxy-only**      | **this finding** |
| (runnable, F#666-compliant)                   | none / SUPPORTED    | target-gated        | regular KILL/SUPPORT |

Drain-wide pattern count after this iteration:
- 5 novel-mechanism PROVISIONALs (F#682, F#683, F#684, F#696, F#697)
- 6 F#669-family preempt-KILLs (F#669, F#671, F#672, F#687, F#698, F#699)
- **1 F#666-pure standalone preempt-KILL (this)**
- 3 SUPPORTED (budget_forcing, semantic_router, cayley_riemannian)
- 1 regular KILL (kv_cache_reuse_honest)

## Unblock path

Re-claim requires **KC-augmentation** (pre-registration modification before re-claim):

1. Add a target-metric KC pairing K1856 to a behavioral outcome. Candidate:
   - "On Gemma 4 E4B distilled via Hedgehog cos-sim (per F#683/F#684/F#696/F#697 recipes), downstream task accuracy gain correlates with per-layer cos-sim variance at r ≥ 0.4 (Spearman across N=24 layers)."
2. Add an arxiv reference (Hedgehog method paper, if any, or prior finding).
3. Set `platform=mlx`.
4. Populate `success_criteria` (mirror of KC pass condition).

Alternative (recommended): re-scope as sibling of Hedgehog family rather than resurrecting the malformed pre-reg.

## Related

- **F#666** (guardrail 1007) — target-gated KILL discipline; the rule this experiment's KC set violates.
- **F#698** — prior F#666 compound sub-case (parent-unverified + proxy-only). Orthogonal to this case (no parent dep here).
- **F#669 / F#687 / F#699** — preempt-KILL family on parent-unverified. Not applicable (no parent).
- **F#683 / F#684 / F#696 / F#697** — Hedgehog-axis PROVISIONAL set. Per `notes`, this experiment's purpose was to baseline Hedgehog claims.
- **Guardrail 1002** — every experiment cites a paper/finding; empty `references` is a secondary violation.

## Follow-up filed

None — preempt-structural KILL does not spawn an `_impl` companion (per F#687/F#698/F#699 + reviewer.md §5). Unblock is pre-registration-external (edit DB entry to add target KC), not implementation-external.
