# LEARNINGS.md — exp_followup_routing_output_space_top2

## TL;DR

Preempt-KILL. Near-duplicate of already-killed `exp_followup_output_space_qa_adapters`
(K1552). Five independent lemmas (L1∨L2∨L3∨L4∨L5) show K1577 carries no
thesis-relevant information. 2nd instance of
`tautological-inter-adapter-delta-ignores-base-baseline` — promotion threshold.

## Reusable learnings

1. **Inter-variant delta KCs need a base anchor or they are tautological.**
   When one variant is format- or scale-incompatible by construction (e.g.
   NTP-format adapters cannot emit MCQ letters), the inter-variant delta
   trivially exceeds any small threshold. Always check: is one variant
   constructed to lose at this metric?

2. **Near-duplicate pre-regs can be preempted on the sibling's proof.**
   `experiment query` surfaces textually equivalent claims. The sibling's
   MATH.md L1–L4 applied here without modification — only the baseline naming
   differed. Duplicate detection should be part of researcher pre-flight.

3. **"Format-fix" proposals are rarely thesis-advancing.** F#165's root cause
   is dual: NTP adapters emit wrong format AND degrade base quality. Fixing
   format alone leaves the quality lag intact. Always check which of the cited
   root causes a proposed fix actually addresses (F#477 confirms gate is
   structurally unlikely even with format-fix).

4. **Bundled orthogonal fixes destroy attribution at the KC level.** K1577
   bundles QA-format + KV-cache-aware into one inter-variant delta. Even a
   non-trivial PASS cannot be attributed to either. Split into separate KCs
   or separate experiments.

5. **Runtime LoRA composition is already output-space MoE (F#167/F#168).**
   The binding constraint on OS-top2 experiments is per-adapter base-beat
   capacity on the target model, not composition architecture. Further
   variants that don't raise the per-adapter quality ceiling are preempt
   candidates.

## Taxonomic placement

4th row extension of the drain-window preempt-structural family:

| Axis                                            | Governing finding | Promotion status    |
| ----------------------------------------------- | ----------------- | ------------------- |
| F#669-family (parent-untested child KCs)        | F#669             | promoted (§5 clause) |
| F#666-pure standalone (proxy-only KCs)          | F#666             | promoted (§5 clause) |
| F#702 hygiene-patch (hygiene defects + target KC) | F#702          | 1st instance, watchlist |
| **tautological-inter-variant-delta (this + K1552)** | (new) | **2nd instance — promote now** |

## Analyst action items (non-blocking for this verdict)

### Primary (promote at 2nd instance)

File `mem-antipattern-tautological-inter-adapter-delta-ignores-base-baseline`:

**Trigger:** Pre-reg has a KC of the form `f(variant_A) − f(variant_B) ≥ δ`
where (a) neither variant is the base/reference, (b) one variant is
format- or scale-incompatible with the metric by construction.

**Canonical precedents:**
- K1552 / `exp_followup_output_space_qa_adapters` (killed 2026-04-19, 1st instance)
- K1577 / `exp_followup_routing_output_space_top2` (killed 2026-04-24, 2nd instance)

**Preempt check for researcher pre-flight:**
- Is there a base-anchored KC paired with the inter-variant delta? If no, REVISE.
- Is one variant format- or scale-incompatible by construction? If yes, REVISE.

**Remedy:** pair the inter-variant KC with a base-anchored KC, or anchor
directly to base; remove the format-incompatible variant.

### Secondary (watchlist, 1st instance)

Flag `duplicate-of-already-killed-pre-reg`:

**Trigger:** `experiment query <kc-keywords>` surfaces a textually equivalent
KC already marked `killed`.

**Canonical precedent:** K1577 textually duplicates K1552; same parent-kill
motivation, same fix proposal, only baseline naming differs.

Recommended: add `experiment query` check to researcher pre-flight before
claiming (one call, low cost, prevents duplicate-claim drain-window waste).
Promote to standalone antipattern memory at 2nd instance.

### Reviewer.md §5 edit (deferred to 3rd instance)

Candidate new preempt-structural sub-case in reviewer.md §5:

> **KILL (preempt-structural sub-case — tautological-inter-variant-delta)** —
> parent-orthogonal, target-metric-bearing. Trigger: KC is an inter-variant
> delta with no base anchor + one variant is incompatible with the metric by
> construction. F#666-pure does not apply (target metric present); F#669-family
> does not apply (no parent dep). Governing antipattern:
> `mem-antipattern-tautological-inter-adapter-delta-ignores-base-baseline`.

Defer until 3rd instance per promotion convention.

## Design debt (not acted on this iter)

- Future OS-top2 / composition pre-regs must pre-register a base-beat gate
  (F#166) as K1 before any composition KC. Retire format-fix-only variants
  until base-beat is empirically supported on the target model.
- `experiment claim` pre-flight should include `experiment query` duplicate
  detection against `status=killed` (cheap, prevents duplicates from reaching
  the researcher).
