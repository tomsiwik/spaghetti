# PAPER.md — exp_followup_routing_output_space_top2

**Status:** KILLED — preemptive, structurally uninformative KC.
**Verdict:** KILLED (no code executed; five-lemma proof, see MATH.md).
**is_smoke:** false. **all_pass:** false. **preemptive:** true.

## One-line conclusion

K1577 ("QA-format + cache-aware top-2 beats NTP+swap-per-token baseline on QA
by ≥5pp") is either tautological (L1), prerequisite-gate unmet (L2),
base-beat-impossible (L3), bundled-fixes unidentifiable (L4), or a textual
duplicate of already-killed `exp_followup_output_space_qa_adapters` K1552 (L5).
The KC does not test the thesis it claims to test.

## Prediction-vs-measurement

| Quantity                                    | Predicted (MATH.md §1)       | Measured       | Verdict            |
| ------------------------------------------- | ---------------------------- | -------------- | ------------------ |
| `Q(A_NTP)` Falcon-E-3B MMLU                 | ≤ 0.42 (F#165 reproduces)    | not measured   | preempt            |
| `Q(A_QA)` Falcon-E-3B MMLU                  | [0.47, 0.53] (L1 regime)     | not measured   | preempt            |
| K1577 delta `Q(A_QA) − Q(A_NTP)`            | ≥ 5pp (tautological, L1)     | not measured   | preempt FAIL       |
| Prerequisite gate `Q(A_QA,single) − Q(B)`   | [−0.08, −0.01] (F#477)       | not measured   | preempt FAIL       |
| Cache-aware speed on M5 Pro 48GB            | bundled with K1577 → unidentifiable | not measured | preempt (L4) |

## Why preempt instead of run

Five independent lemmas show K1577 carries no thesis-information (full proofs
in MATH.md §1):

- **L1 — tautological KC.** NTP-format adapters emit continuation prose on MCQ
  items (F#165 measured −24pp vs base on Falcon-E-3B). QA-format adapters emit
  letter answers by construction. Any positive `Q(A_QA)` trivially crosses 5pp
  over a prose-emitter — format-alignment is the measured effect, not composition.
- **L2 — prerequisite-gate unmet.** F#166 requires single adapter to beat base
  before composition is testable. K1577 compares inter-variant delta, not
  adapter-vs-base. A PASS says nothing about the thesis.
- **L3 — base-beat unlikely on Falcon-E-3B.** F#477 measured base-beat rate
  2/5 on the stronger Gemma 4; Falcon-E-3B is at least as well calibrated, so
  the F#166 gate is structurally unlikely to hold.
- **L4 — bundled fixes.** K1577 mixes two orthogonal remedies (QA-format +
  KV-cache-aware) into one inter-variant delta. Even a PASS cannot be
  attributed to either remedy.
- **L5 — near-duplicate.** `exp_followup_output_space_qa_adapters` (K1552) was
  killed 2026-04-19 with the same four-lemma proof on a textually equivalent
  KC and the same parent-kill motivation. This pre-reg differs only in
  baseline naming ("NTP+swap-per-token baseline" vs "NTP-format adapters"); the
  preempt applies byte-for-byte.

## Findings reused

F#165, F#166, F#167, F#168, F#477.

**Sibling precedent:** `exp_followup_output_space_qa_adapters` K1552 (killed
2026-04-19, preempt-structural).

## Antipattern flags

- `tautological-inter-adapter-delta-ignores-base-baseline` (**2nd instance** — sibling K1552 was 1st)
- `prerequisite-gate-unmet-output-space-composition`
- `bundled-orthogonal-fixes-format-plus-speed-one-kc`
- `format-alignment-symptom-fix-not-disease`
- `duplicate-of-already-killed-pre-reg` (new sub-axis flag — 1st instance)

## Taxonomic comparison (drain-window anchors)

| Experiment                                      | Parent dep | Target KC | Hygiene defects | Clause      | Finding |
| ----------------------------------------------- | ---------- | --------- | --------------- | ----------- | ------- |
| F#700 exp_g4_per_layer_cos_baseline            | no         | proxy-only | 3               | F#666-pure | F#700   |
| F#701 exp_adapter_orthogonality_audit          | no         | proxy-only | 3               | F#666-pure | F#701   |
| F#703 exp_followup_tfidf_medical_unaliased     | no         | proxy-only | 2               | F#666-pure | F#703   |
| F#702 exp_pierre_adapter_hotswap_latency       | no         | target+target | 3           | hygiene-patch PROVISIONAL | F#702 |
| **this: exp_followup_routing_output_space_top2** | **no**   | **target (tautological)** | **2** | **new: tautological-inter-variant-delta** | (new) |

Distinction from F#666-pure: this KC *has* a target metric (QA accuracy) — it
just happens to be a tautological target. Distinction from F#702: not patchable
via hygiene-fix + `_impl` because the tautology is in the KC structure, not in
the surrounding metadata.

## Recommended v2 (not filed)

A well-formed re-registration would be named `exp_followup_os_top2_base_anchored_v2`
with:

1. **K1 — base-beat gate:** `Q(A_QA, single, d) ≥ Q(B) + 3pp` on ≥3/5 domains at rank ≤6.
2. **K2 — anchored composition (conditional on K1):** `Q(A_QA, top2) ≥ Q(B) + 5pp` on ≥3/5 domains.
3. **K3 — speed:** ≥30 tok/s on M5 Pro 48GB with KV-cache-aware impl (separate KC for the implementation fix).
4. **References:** F#165, F#166, F#477, F#167, F#168, sibling K1552.
5. **Motivation:** cite F#165/F#166 measured values directly; drop the "killed_00.md" markdown pointer.

## Assumptions

- "NTP+swap-per-token baseline" = F#166's A_NTP configuration (NTP training
  format + naïve adapter-swap per token). Pre-reg text is ambiguous; this is
  the most defensible reading aligned with the cited parent-kill.
- Falcon-E-3B MMLU priors are ≥ Gemma 4's at the relevant rank (sibling L3
  assumption, unchanged).
- Ralph-autonomy: no user clarification requested for the ambiguous "NTP+swap"
  phrasing — preempt applies under either reading (format, impl, or both).

## Unblock path

This pre-reg is **not** re-claimable by patching. A new pre-reg (per §
"Recommended v2") is required. Do not patch via `experiment update` — KC
mutation post-claim is antipattern-u.
