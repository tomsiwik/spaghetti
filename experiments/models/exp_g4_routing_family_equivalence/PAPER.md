# PAPER.md — exp_g4_routing_family_equivalence

**Status:** KILLED — preemptive, structurally uninformative KC.
**Verdict:** KILLED (no code executed; five-lemma proof, see MATH.md).
**is_smoke:** false. **all_pass:** false. **preemptive:** true.

## One-line conclusion

K1584 ("max pairwise gap < 2pp at N=5" among SOLE/CAT/LoRA-Flow/X-LoRA on
Gemma 4 MMLU-Pro) is either degenerate-equivalence (L1 — parent F#150
explicitly measured the comparison as "identical to 4 decimal places" under
the orthogonality condition the pre-reg cites), prerequisite-gate unmet (L2
— F#166), base-beat structurally unlikely on Gemma 4 (L3 — F#477 2/5
domains), parent-caveat-inheritance failure (L4 — 3rd instance of
template-regression sub-pattern), or 3rd instance of
`tautological-inter-adapter-delta-ignores-base-baseline` triggering §5
reviewer.md promotion (L5). The KC does not test the thesis.

## Prediction-vs-measurement

| Quantity                                          | Predicted (MATH.md §1)                                       | Measured     | Verdict       |
| ------------------------------------------------- | ------------------------------------------------------------ | ------------ | ------------- |
| `Q(SOLE, ·)` Gemma 4 MMLU-Pro mean                | ≈ base (F#150: identical to 4 decimal places under orthog.) | not measured | preempt       |
| `Q(CAT, ·)` Gemma 4 MMLU-Pro mean                 | ≈ base                                                       | not measured | preempt       |
| `Q(LoRA-Flow, ·)` Gemma 4 MMLU-Pro mean           | ≈ base                                                       | not measured | preempt       |
| `Q(X-LoRA, ·)` Gemma 4 MMLU-Pro mean              | ≈ base                                                       | not measured | preempt       |
| K1584 pairwise gap `G`                            | < 0.0001 (F#150 → vacuous < 0.02, L1)                       | not measured | preempt FAIL  |
| Single-adapter base-beat rate (prerequisite gate) | 2/5 domains (F#477)                                          | not measured | preempt FAIL  |
| `mean_variant(Q(variant, ·)) − Q(base, ·)`        | [−0.03, +0.02] (F#477 shared shallow regime)                 | not measured | preempt (L3)  |
| Thesis-relevance of any PASS                      | 0 (PASS compatible with all-variants-fail-to-beat-base)      | not measured | preempt (L1+L2)|

## Why preempt instead of run

Five independent lemmas show K1584 carries no thesis-information (full proofs
in MATH.md §1):

- **L1 — degenerate-equivalence.** Parent F#150 (2026-03-15) explicitly
  measured the same 4-variant quality comparison as "identical to 4 decimal
  places" under the zero-specialization / orthogonality condition the pre-reg
  cites. The PASS `G < 0.02` is already demonstrated by the parent as vacuous
  under verified Grassmannian — the structural precondition the pre-reg
  inherits.
- **L2 — prerequisite-gate unmet.** F#166: single adapter must beat base
  before composition is testable. K1584 compares inter-variant deltas with
  no base-anchored KC. PASS says nothing about thesis.
- **L3 — base-beat unlikely on Gemma 4.** F#477 measured base-beat 2/5
  domains at rank 6 on the target model. With majority of domains in
  below-base regime, all variants cluster toward base-level accuracy,
  making `G < 0.02` a low-information noise-regime measurement.
- **L4 — parent-caveat-inheritance failure.** 3rd instance of
  template-regression sub-pattern. Child K1584 inherits parent F#150's
  explicit anti-caveat ("Quality comparison vacuous") as its KC structure.
  Strictly worse than prior template-regression instances (F#705 inherited
  stale secondary advice; F#708 half-stripped a paired design).
- **L5 — 3rd-instance promotion.** Antipattern
  `tautological-inter-adapter-delta-ignores-base-baseline` reaches 3rd
  instance (K1552, K1577/F#704, K1584). Triggers promotion to named §5
  clause in `reviewer.md` per repo's 3-instance-promote convention.

## Findings reused

F#150 (parent, anchors L1, L4), F#165, F#166, F#167, F#168, F#477, F#704
(2nd-instance precedent), F#705, F#708 (template-regression anchors).

**Sibling precedents:**
- `exp_followup_output_space_qa_adapters` K1552 (killed 2026-04-19, 1st
  instance, delta ≥ 5pp direction).
- `exp_followup_routing_output_space_top2` K1577 / F#704 (killed 2026-04-24,
  2nd instance, delta ≥ 5pp direction, promotion threshold reached).
- This: K1584 (3rd instance, **inverted delta direction `< 2pp`**, promotion
  triggered).

## Antipattern flags

- `tautological-inter-adapter-delta-ignores-base-baseline` — **3rd instance**;
  triggers promotion to named §5 clause in `reviewer.md`. Both delta
  directions (≥ and <) covered by the same structural failure.
- `template-regression-explicit-anti-caveat-inheritance` — **3rd sub-variant
  instance** (F#705 stale-caveat, F#708 paired-design-half-strip, this
  explicit-anti-caveat); triggers watchlist → formal antipattern promotion.
- `prerequisite-gate-unmet-routing-composition` — L2.
- `degenerate-equivalence-branch-gap-by-shared-failure` — L1.
- `parent-caveat-inheritance-failure-from-F#150` — L4 anchor.

## Taxonomic comparison (drain-window anchors, 5-row table)

| Experiment                                        | Parent dep | Target KC         | Hygiene defects | Clause                                 | Finding |
| ------------------------------------------------- | ---------- | ----------------- | --------------- | -------------------------------------- | ------- |
| F#700 exp_g4_per_layer_cos_baseline              | no         | proxy-only        | 3               | F#666-pure                             | F#700   |
| F#701 exp_adapter_orthogonality_audit            | no         | proxy-only        | 3               | F#666-pure                             | F#701   |
| F#703 exp_followup_tfidf_medical_unaliased       | no         | proxy-only        | 2               | F#666-pure                             | F#703   |
| F#702 exp_pierre_adapter_hotswap_latency         | no         | target+target     | 3               | hygiene-patch PROVISIONAL              | F#702   |
| F#704 exp_followup_routing_output_space_top2     | no         | target (tautol.)  | 2               | tautological-inter-adapter-delta (2nd) | F#704   |
| F#705 exp_followup_budget_forcing_baseline_fix   | no         | target            | —               | template-regression watchlist (1st)    | F#705   |
| F#708 exp_g4_hash_ring_remove_n25                | no         | proxy-only (PPL)  | 2               | F#666-pure (7th) + template-reg (2nd)  | F#708   |
| **this: exp_g4_routing_family_equivalence**      | **no**     | **target (tautol.)** | **2**        | **tautological-inter-adapter-delta (3rd) + template-reg (3rd)** | **(new)** |

## Recommended v2 (not filed)

Name: `exp_g4_routing_family_base_anchored_v2`.

K-set:
- **K1 (prerequisite gate):** at least one variant `r ∈ R` achieves
  `Q(r, single-adapter, d) ≥ Q(base, d) + 3pp` for ≥3/5 domains on Gemma 4
  MMLU-Pro at rank ≤ 6.
- **K2 (base-anchored composition, conditional on K1):** for any variant
  `r` passing K1, `Q(r, top-k, N=5) ≥ Q(base, ·) + 5pp` on ≥3/5 domains.
- **K3 (inter-variant robustness, conditional on K1 & K2):** `G < 2pp`
  paired with K1/K2 — this then ruling out degenerate-equivalence.
- **K4 (orthogonality fixture, not KC):** `max_{i≠j} |A_i^T A_j| < 0.1`
  as sanity check, not pass/fail condition.

References: F#150, F#165, F#166, F#167, F#168, F#477, F#704, K1552.

## Assumptions

- "verified Grassmannian" in the pre-reg notes = the Grassmannian A-matrix
  construction validated in F#562 (structural orthogonality at Gemma 4
  native dims).
- "N=5" = the 5 MMLU-Pro categories used in F#477
  (math/code/medical/legal/finance); pre-reg is silent on exact subset.
- "pairwise gap" = max pairwise absolute difference of per-variant mean
  MMLU-Pro accuracy across the N=5 domains.
- "2pp" = 0.02 absolute accuracy difference (not relative).
- F#150's "identical to 4 decimal places" (|gap| < 0.0001) applies verbatim
  under orthogonality, making K1584 PASS by construction in the orthogonal
  regime the pre-reg cites.
