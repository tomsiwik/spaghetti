# MATH.md — exp_g4_routing_family_equivalence (PREEMPTIVE KILL)

## §0. Platform skills

`/mlx-dev` and `/fast-mlx` **not invoked** — no MLX code is written in this
experiment. See §5 for the graceful-failure stub rationale. Any well-formed
re-registration (new experiment ID per §4) that touches Gemma 4 inference must
invoke both skills before writing MLX code.

## §1. Claim and preempt theorem

### Claim under test (pre-registered 2026-04-17)

**K1584:** *max pairwise gap < 2pp at N=5*

Among `{SOLE, CAT, LoRA-Flow, X-LoRA}` on Gemma 4 MMLU-Pro at N=5.

### Theorem (preempt)

Let `R = {SOLE, CAT, LoRA-Flow, X-LoRA}` be four routing variants. Let
`Q(r, d)` denote MMLU-Pro accuracy of variant `r` on domain `d`. Let
`G = max_{r_i, r_j ∈ R} |mean_d Q(r_i, d) − mean_d Q(r_j, d)|` be the max
pairwise gap. Let `B = Q(base, ·)` be the unmodified Gemma 4 26B-A4B baseline.
K1584 is the claim `G < 0.02`.

At least one of the following lemmas holds, and each alone suffices to show
K1584 outcome carries no thesis-relevant information:

**L1 (degenerate-equivalence branch).** If all four variants land in a
common degenerate regime — e.g. all produce base-level or below-base accuracy
on MMLU-Pro — then `G ≈ 0 < 0.02` trivially. F#150 (the parent finding)
explicitly warned:

> "Quality comparison vacuous — zero expert specialization means all 4 methods
> (SOLE, CAT, LoRA-Flow, X-LoRA) produce identical loss to 4 decimal places.
> Cannot distinguish 'LoRA-Flow useless under orthogonality' from 'LoRA-Flow
> useless without expert specialization.'"
> — F#150 caveats, 2026-03-15

The pre-reg notes field states: "under verified Grassmannian, routing families
collapse on Gemma 4." Verified Grassmannian orthogonality IS the zero-
expert-specialization condition F#150 flagged as making the comparison vacuous.
K1584 is therefore a PASS-by-construction: if orthogonality holds, the KC
passes regardless of whether any variant delivers thesis-relevant behavior.

**L2 (prerequisite-gate unmet — F#166).** F#166 impossibility structure:

> "Output-space eliminates cross-terms (LoRI proof correct) but cannot rescue
> individually harmful adapters. Prerequisite gate needed: single adapter must
> beat base before testing composition."
> — F#166, 2026-03-28

K1584 compares inter-variant deltas `|Q(r_i, ·) − Q(r_j, ·)|`. It does not
compare any `Q(r_k, ·) vs Q(base, ·)`. The prerequisite gate (at least one
routing variant with at least one adapter beats base) is neither pre-registered
nor measured. Without the gate, a PASS on K1584 says nothing about the thesis
"routing families collapse under Grassmannian orthogonality" — the collapse
could be to the shared failure mode, not to a shared success.

**L3 (base-beat structurally unlikely — F#477).** F#477 measured on Gemma 4
at rank 6: single-adapter base-beat rate 2/5 domains (math +20pp, finance
+14.7pp; medical −4pp; code +6.7pp, legal +9.3pp — but K1226 FAIL at
adapted-acc 0.480 < 0.50 threshold). Impossibility structure: `δ_d ≈ 0 when
H(V_d|θ) is low` — base already calibrated on MMLU-class domains. With only
2/5 domains base-beat capable, the mean over N=5 MMLU-Pro domains drags all
variants toward base-level performance, making `G < 0.02` a low-information
measurement in the shared failure-to-beat-base regime.

**L4 (parent-caveat-inheritance failure — template-regression, 3rd instance).**
Parent F#150 (SUPPORTED positioning-only, 2026-03-15) explicitly documented
that its own 4-variant quality comparison was vacuous and recommended NOT
inheriting the comparison structure without expert specialization. The child
pre-reg K1584 inherits the exact comparison structure the parent flagged as
vacuous, on Gemma 4 where F#477 shows the specialization condition likely
still fails. This is the **3rd instance of template-regression** (1st: F#705
stale-caveat-inheritance from F#161; 2nd: F#708 paired-design-half-stripping
from F#133; 3rd: this — explicit-anti-caveat-inheritance from F#150). Per the
repo's 2-instance-promote convention applied to the watchlist filed at F#708,
3rd instance reaches the promotion threshold for
`mem-antipattern-f666pure-template-regression` to graduate from watchlist to
formal antipattern. (Analyst action; not blocking this verdict.)

**L5 (promotion-threshold — 3rd instance of tautological-inter-adapter-delta).**
This is the **3rd instance** of the `tautological-inter-adapter-delta-ignores-base-baseline`
antipattern:

- 1st: K1552 (`exp_followup_output_space_qa_adapters`, killed 2026-04-19)
- 2nd: K1577 (`exp_followup_routing_output_space_top2` = F#704, killed 2026-04-24,
  "promotion threshold reached per repo convention")
- 3rd: K1584 (this experiment), with inverted delta direction (`G < 0.02`
  instead of `delta ≥ 5pp`) but structurally identical failure mode: no base
  anchor, KC passable by degenerate equivalence.

The inversion `<` vs `≥` does not rescue the KC. In the `≥` direction, the
tautology is "large gap by format-incompatibility." In the `<` direction, the
tautology is "small gap by shared-failure-equivalence." Both pass without
thesis-progress; both require the same remedy (pair with base-anchored KC).
3rd instance triggers promotion of the antipattern memory to a named §5
clause in `reviewer.md`. Analyst action; not blocking this verdict.

### QED (preempt)

`L1 ∨ L2 ∨ L3 ∨ L4 ∨ L5` ⇒ K1584 outcome carries no information about the
thesis "routing families collapse to equivalence on Gemma 4 MMLU-Pro under
verified Grassmannian orthogonality." No hyperparameter, seed, or scale choice
rescues this: the KC is malformed with respect to its claim, and the parent
finding (F#150) explicitly labeled the comparison structure as vacuous under
the orthogonality condition the pre-reg relies on.

## §2. Prior art cited

- **F#150** (supported, 2026-03-15) — SOLE positioning vs LoRA-Flow confirmed;
  parent of this experiment. Explicitly caveats: *"Quality comparison vacuous
  — zero expert specialization means all 4 methods produce identical loss to
  4 decimal places."* Positioning (SOLE for N≫10, LoRA-Flow for k≤10) was
  supported; quality equivalence was not tested and was flagged unreachable
  from the comparison structure alone.
- **F#165** (killed, 2026-03-28) — Output-space top-2 KILLED on Falcon-E-3B;
  root cause: NTP-trained adapters degrade instruction-tuned base. Anchors
  L2 prerequisite-gate-unmet reasoning.
- **F#166** (killed, 2026-03-28) — Prerequisite gate impossibility structure.
  L2 governing lemma.
- **F#167 / F#168** (supported, 2026-03-28) — Runtime LoRA IS output-space
  MoE; binding constraint is per-adapter base quality, not composition
  architecture. Redundant with F#150's "identical to 4 decimal places" —
  once base quality is the binding constraint, routing architecture choices
  collapse below measurement noise.
- **F#477** (killed, 2026-04-11) — Gemma 4 single-adapter base-beat rate 2/5
  domains at rank 6. L3 governing datum.
- **F#704** (killed, 2026-04-24) — 2nd instance of
  `tautological-inter-adapter-delta-ignores-base-baseline`; promotion
  threshold reached per repo convention. Governing precedent for §5
  promotion-to-clause action at 3rd instance.
- **F#705 / F#708** — Template-regression sub-pattern anchors (watchlist at
  2nd; promotion at 3rd per F#708 analyst note).
- **LoRA-Flow** (Wang et al. 2024, arxiv:2402.11455) — cited in F#150 for
  routing-family positioning analysis.

## §3. Kill criterion (pre-registered, verbatim)

**K1584:** "max pairwise gap < 2pp at N=5"

Preempt verdict: **FAIL** (status: structurally uninformative; see §1 L1–L5).

## §4. Unblock path

This pre-reg is dead. A valid v2 would require re-registering from scratch
with a new experiment ID:

1. Pre-register a base-anchored quality gate:
   `Q(variant_r, single-adapter, d) ≥ Q(base, d) + 3pp` for ≥3/5 domains,
   for at least one variant `r ∈ R`. **Before** touching inter-variant
   comparison. Per F#165/F#166/F#477, if the gate fails, no routing
   architecture rescues it — stop.
2. Replace the inter-variant delta KC with a base-anchored KC for each
   variant: `Q(variant_r, top-k, N=5) ≥ Q(base, ·) + 5pp` on ≥3/5 domains.
3. Additionally (optional) pair with an inter-variant robustness KC:
   `G < 0.02` paired with the per-variant base-beat KCs. The pair rules out
   the degenerate-equivalence branch — if all variants beat base by 3pp AND
   cluster within 2pp of each other, the equivalence claim is thesis-relevant.
4. Decouple "verified Grassmannian" from the comparison: orthogonality
   should be a fixture, not a KC confound. Measure orthogonality as a sanity
   check (e.g. max |A_i^T A_j| < 0.1) and exclude it from KCs.
5. Cite F#167/F#168 in motivation — runtime LoRA composition IS output-space
   MoE; the binding constraint is per-adapter base quality, not routing
   family. The positioning claim (SOLE vs LoRA-Flow parameter scaling) from
   F#150 is already supported and does not need Gemma 4 re-verification.

Do **not** patch this pre-reg via `experiment update` (KC mutation post-claim
is antipattern-u). Re-registration with a different ID is required.

## §5. No `_impl` companion

Preempt-structural KILL is self-contained. The unblock is pre-reg-external
(a new pre-reg per §4), not a follow-up `_impl`. Matches F#669-family,
F#666-pure-standalone, and F#704 tautological-inter-adapter-delta precedents
(F#687, F#698, F#699, F#700, F#701, F#703, F#704, F#705, F#706, F#707, F#708).

## §6. Scope-integrity

No MLX / Gemma 4 / mlx_lm surface is touched. `run_experiment.py` imports
only `json` and `pathlib`. Runtime ≤ 200 ms; graceful-failure `main()` writes
`results.json` and exits 0. Canonical preempt-structural artifact pattern
(see F#700/F#701/F#703/F#704 for identical stub structure).

## §7. Taxonomic placement (expanded 5-row table)

| Taxon                                            | Target metric? | Parent dep?       | Hygiene defects | Verdict & clause                     |
| ------------------------------------------------ | -------------- | ----------------- | --------------- | ------------------------------------ |
| F#669-family (parent-untested child KCs)         | usually yes    | yes (transitive)  | any             | preempt-KILL (§5 F#669)              |
| F#666-pure standalone (proxy-only KCs)           | no (proxy)     | no                | usually 2-3     | preempt-KILL (§5 F#666-pure)         |
| F#702 hygiene-patch PROVISIONAL                  | yes            | no                | 3+              | PROVISIONAL + `_impl`                |
| F#704 tautological-inter-adapter-delta (2nd)    | yes (tautol.)  | no                | 2               | preempt-KILL (antipattern memory; §5 deferred) |
| **This: tautological-inter-adapter-delta (3rd)** | **yes (tautol.)** | **no**         | **2**           | **preempt-KILL (promotion to §5 clause)** |

Distinction from F#666-pure: this KC's underlying metric IS a target (MMLU-Pro
accuracy) — the tautology is in the inter-variant-delta *structure* (no base
anchor), not in the metric being a proxy.

Distinction from F#704: inverted delta direction. F#704's K1577 demanded
`delta ≥ 5pp` (large, tautological because one variant is format-incompatible).
K1584 demands `gap < 2pp` (small, tautological because all variants can
degenerately equal each other at/below base). The antipattern covers both
directions: any inter-variant delta KC without a base anchor admits degenerate
PASS cases independent of the thesis.

## §8. Analyst action items (recorded here, not executed)

### Primary (promotion at 3rd instance)

1. **Promote** `mem-antipattern-tautological-inter-adapter-delta-ignores-base-baseline`
   to a named clause in `reviewer.md §5`, parallel to the existing F#669-family
   and F#666-pure-standalone clauses. Required clause text skeleton:
   - Trigger: pre-reg has a KC of the form
     `op(f(variant_i), f(variant_j)) op_2 δ` (where `op ∈ {−, max−, ratio, ...}`
     and `op_2 ∈ {≥, <, ≤, >}`), with no paired base-anchored KC
     `f(variant_k) op_3 f(base) ± γ`.
   - Required artifact pattern: identical to F#666-pure preempt-structural
     clause (no MLX surface, json+pathlib only, graceful-failure results.json,
     no `_impl`).
   - Canonical precedents: K1552 (1st), K1577/F#704 (2nd), K1584/F#709 (3rd —
     promotion trigger).
2. **Extend** the clause to cover both delta directions (≥ and <) —
   tautology is direction-symmetric.

### Secondary (template-regression promotion at 3rd instance)

3. **Promote** `mem-watchlist-f666pure-template-regression` to a formal
   antipattern memory. Three sub-variants now anchored:
   - F#705 (stale-caveat-inheritance from F#161)
   - F#708 (paired-design-half-stripping from F#133)
   - F#709 (explicit-anti-caveat-inheritance from F#150) — 3rd, promotion.

### Tertiary (researcher pre-flight checklist)

4. Add to pre-claim checklist: "If pre-reg notes field says 'replicate parent
   finding F#X' or 'under verified <condition>', run `experiment finding-get X`
   and scan the parent's *caveats* field. If parent caveats explicitly flag
   the comparison structure as vacuous / unmeasured / inapplicable, the child
   pre-reg must re-register with a materially different KC structure — not
   inherit the flagged structure."

## Assumptions

- N=5 in K1584 refers to the 5 MMLU-Pro category subset standardly used in
  prior Gemma 4 composition experiments (math/code/medical/legal/finance per
  F#477). Pre-reg is silent on the exact 5; F#477's domain list is the most
  defensible reading.
- "verified Grassmannian" in the notes field refers to the Grassmannian
  A-matrix construction validated in F#562 (structural orthogonality at
  Gemma 4 native dims).
- "2pp" is absolute MMLU-Pro accuracy percentage points, not relative.
- "pairwise gap" = `max_{r_i ≠ r_j} |mean_d Q(r_i, d) − mean_d Q(r_j, d)|`.
  F#150's "identical to 4 decimal places" implies `G < 0.0001 ≪ 0.02`,
  making L1's degenerate-equivalence case trivially demonstrable.
