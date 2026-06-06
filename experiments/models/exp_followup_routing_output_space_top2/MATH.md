# MATH.md — exp_followup_routing_output_space_top2 (PREEMPTIVE KILL)

## §0. Platform skills

`/mlx-dev` and `/fast-mlx` **not invoked** — no MLX code is written in this
experiment. See §5 for the graceful-failure stub rationale. Any `_v2` re-design
(not filed here; see §6) must invoke both skills before writing Falcon-E-3B
inference code.

## §1. Claim and preempt theorem

### Claim under test (pre-registered 2026-04-17)

**K1577:** *QA-format + cache-aware top-2 beats NTP+swap-per-token baseline on
QA by ≥5pp.*

### Theorem (preempt)

Let `A_NTP` = output-space top-2 over adapters trained on next-token-prediction
(prose) on Falcon-E-3B, executed with naive swap-per-token. Let `A_QA` = the
same but over adapters trained on QA-format supervision, executed with
KV-cache-aware top-2. Let `B` = unmodified instruction-tuned base. Let `Q(·)` =
QA accuracy on MCQ items. K1577 is the claim `Q(A_QA) − Q(A_NTP) ≥ 0.05`.

At least one of the following lemmas holds, and each alone suffices to show
K1577 carries no thesis-relevant information:

**L1 (tautological-KC branch).** NTP-format adapters on MCQ items systematically
emit continuation-style prose, not letter answers. F#165 measured `Q(A_NTP)` at
0.410 on Falcon-E-3B MMLU, −24pp vs base. QA-format adapters emit letter
answers by construction of their training distribution. Therefore any adapter
that lands in the MCQ-letter-answer regime trivially crosses the 5pp delta over
a prose-emitter — format-alignment is the measured effect, not composition
quality. K1577 measures output-format alignment, not the thesis.

**L2 (prerequisite-gate unmet).** F#166 (verbatim):
> Output-space eliminates cross-terms (LoRI proof correct) but cannot rescue
> individually harmful adapters. Prerequisite gate needed: single adapter must
> beat base before testing composition.
> — F#166 impossibility structure

F#165 (sharper):
> If `E[quality(adapter_i)] < E[quality(base)]`, then
> `E[avg(adapter_i, adapter_j)] < E[quality(base)]` — no aggregation fixes
> negative individual contributions.

K1577 compares inter-variant delta `Q(A_QA) − Q(A_NTP)`. It does not compare
`Q(A_QA) vs Q(B)`. The prerequisite gate (single QA-adapter on Falcon-E-3B
beats base) is neither pre-registered nor measured. Without the gate, a PASS on
K1577 says nothing about the thesis "OS-top2 composition works."

**L3 (base-beat structurally unlikely on Falcon-E-3B).** F#477 (Gemma 4, a
stronger base than Falcon-E-3B) measured single-adapter base-beat rate 2/5
domains, with K1226 FAIL (adapted acc 0.480 < 0.50). Dominant failure mode:
`δ_d ≈ 0 when H(V_d|θ) is low` (base already calibrated). Falcon-E-3B is
ternary-quantized and instruction-tuned; its MMLU calibration is not weaker
than Gemma 4's at rank 6. Therefore the prerequisite gate from L2 is
structurally unlikely to hold, even with QA-format supervision — format-fix
addresses the symptom (wrong output vocabulary) but not the disease (low
`H(V_d|θ)` on MMLU-class domains).

**L4 (bundled orthogonal fixes destroy attribution).** K1577 bundles *two*
independent remedies (QA-format supervision + KV-cache-aware implementation)
into a single inter-variant delta. Even if K1577 PASSED non-trivially, the 5pp
could not be attributed to format, to cache-awareness, or to composition. The
KC is unidentifiable at the mechanism level.

**L5 (near-duplicate of already-killed pre-reg).** `exp_followup_output_space_qa_adapters`
(K1552, killed 2026-04-19, `micro/models/exp_followup_output_space_qa_adapters/`)
tested the textually equivalent claim:

| Field        | K1552 (sibling, killed)                                      | K1577 (this experiment)                                  |
| ------------ | ------------------------------------------------------------ | -------------------------------------------------------- |
| Variant A    | QA-format adapters with KV-cache-aware top-2                 | QA-format + cache-aware top-2                            |
| Variant B    | NTP-format adapters                                          | NTP + swap-per-token baseline                            |
| Metric       | QA accuracy                                                  | QA                                                       |
| Threshold    | ≥5pp                                                         | ≥5pp                                                     |
| Parent-kill  | exp_top2_output_space_falcon (F#166)                         | exp_top2_output_space_falcon (F#166)                     |
| Motivation   | "killed_00.md NTP/QA format mismatch"                        | "killed_00.md exp_top2_output_space_falcon ceiling"      |

"NTP + swap-per-token baseline" IS NTP-format adapters: swap-per-token is the
implementation of the naïve OS-top2 from F#166; NTP is the training distribution.
The two KCs differ only in wording. The sibling's L1–L4 apply byte-for-byte.

### QED (preempt)

`L1 ∨ L2 ∨ L3 ∨ L4 ∨ L5` ⇒ K1577 outcome carries no information about the
thesis. No hyperparameter, seed, or scale choice changes this: the KC itself
is malformed with respect to the claim it purports to test, AND a structurally
equivalent pre-reg has already been preempt-killed for the same reasons.

## §2. Prior art cited

- **F#165** (killed, 2026-03-28) — Output-space top-2 KILLED on Falcon-E-3B: adapters degrade instruction-tuned base, -24pp vs base, naïve swap 17× overhead.
- **F#166** (killed, 2026-03-28) — Prerequisite gate: single adapter must beat base before composition is testable.
- **F#167 / F#168** (supported, 2026-03-28) — Runtime LoRA IS output-space MoE; binding constraint is base quality per domain, not composition architecture.
- **F#477** (killed, 2026-04-11) — Gemma 4 single-adapter base-beat rate 2/5 domains, gate structurally unlikely even with stronger base.
- **LoRI** (arxiv:2504.07448) — mathematical proof that output-space composition eliminates cross-terms (cited in F#165/F#166).
- **Sibling precedent** — `exp_followup_output_space_qa_adapters` (killed 2026-04-19, preempt with L1 ∧ L2 ∧ L3 ∧ L4). Textually equivalent K1552; same parent-kill motivation. This experiment is a pure duplicate with a different numeric KC id.

## §3. Kill criterion (pre-registered, verbatim)

**K1577:** "QA-format + cache-aware top-2 beats NTP+swap-per-token baseline on
QA by >=5pp"

Preempt verdict: **FAIL** (status: structurally uninformative; see §1 L1–L5).

## §4. Unblock path

This pre-reg is dead. A valid `_v2` would require re-registering from scratch:

1. Pre-register a base-beat gate: `Q(A_QA, single, d) ≥ Q(B) + 3pp` for ≥3/5
   domains, **before** touching composition. If the gate fails, stop — per
   F#165/F#166/F#477 no composition architecture can rescue it.
2. Replace the inter-variant KC with a base-anchored KC:
   `Q(A_QA, top2) ≥ Q(B) + 5pp` (composition beats base, not just a straw
   adapter).
3. Decompose the bundled fixes: one KC for the format change, one KC for the
   cache-aware implementation. Otherwise attribution is impossible even at
   PASS.
4. Cite F#167/F#168 in the motivation — the binding constraint is single-
   adapter base-beat capacity on Falcon-E-3B / Gemma 4, not composition
   architecture.
5. Update motivation: current notes cite "killed_00.md exp_top2_output_space_falcon
   ceiling"; the ceiling is documented in F#165/F#166 with measured values —
   cite those directly instead of a markdown-file pointer.

Do **not** patch this pre-reg via `experiment update` (KC mutation post-claim
is antipattern-u). Re-registration with a different ID is required.

## §5. No `_impl` companion

Preempt-structural KILL is self-contained. The unblock is pre-reg-external (a
new pre-reg per §4), not a follow-up `_impl`. Matches the F#669-family and
F#666-pure-standalone precedent (F#687, F#698, F#699, F#700, F#701, F#703).

## §6. Scope-integrity

No MLX / Falcon-E-3B / mlx_lm surface is touched. `run_experiment.py` imports
only `json` and `pathlib`. Runtime ≤ 200 ms; graceful-failure `main()` writes
`results.json` and exits 0. This is the canonical preempt-structural artifact
pattern (see F#700/F#701/F#703 for identical stub structure).

## §7. Taxonomic placement (expanded 4-row table)

| Taxon                                           | Target metric? | Parent dep?       | Hygiene defects    | Verdict & clause            |
| ----------------------------------------------- | -------------- | ----------------- | ------------------ | --------------------------- |
| F#669-family (parent-untested child KCs)        | usually yes    | yes (transitive)  | any                | preempt-KILL (§5 F#669)     |
| F#700/F#701/F#703 F#666-pure standalone        | no (proxy)     | no                | usually 2-3        | preempt-KILL (§5 F#666-pure)|
| F#696/F#697 novel-mechanism design-only         | usually yes    | no                | usually clean      | PROVISIONAL + `_impl`       |
| F#702 hygiene-patch PROVISIONAL                 | yes            | no                | 3+                 | PROVISIONAL + `_impl`       |
| **This: tautological-inter-adapter-delta (2nd)**| **yes**        | **no**            | **2**              | **preempt-KILL (new sub-axis)** |

The new sub-axis does not fit the existing reviewer.md §5 clauses. The KC has
a target metric (QA accuracy) and no parent dep, so it is neither F#666-pure
(which requires proxy-only) nor F#669-family (which requires parent dep). It
is instead an **inter-variant delta KC where one variant is format-incompatible
by construction**, making the delta tautologically large.

The sibling (K1552, killed 2026-04-19) registered this as a tripwire candidate
in its LEARNINGS.md §Tripwire section but the tripwire was never filed as an
analyst-owned antipattern memory. This is the **2nd instance** — if the 2nd
instance triggers promotion per the repo's 2-instance-promote convention,
analyst should file `mem-antipattern-tautological-inter-adapter-delta-ignores-base-baseline`.

## §8. Follow-up (well-formed, not filed here)

For the analyst: a valid v2 pre-reg would be named
`exp_followup_os_top2_base_anchored_v2` with:

- K1: Base-beat gate at rank ≤ 6: `Q(A_QA, single, d) ≥ Q(B) + 3pp` for ≥3/5 domains.
- K2 (conditional on K1): `Q(A_QA, top2) ≥ Q(B) + 5pp` on ≥3/5 domains.
- K3 (speed): ≥30 tok/s on M5 Pro 48GB with KV-cache-aware impl (disentangles the implementation variable).
- References: F#165, F#166, F#477, F#167, F#168, sibling K1552.

## Assumptions

- "NTP+swap-per-token baseline" refers to the `A_NTP` configuration from F#166
  (NTP-format adapters executed with naïve per-token adapter swap). K1577 text
  is silent on whether "NTP" refers to the training format or just the
  implementation pattern; the most defensible reading is both, matching the
  parent-killed baseline.
- Falcon-E-3B MMLU priors are at least as strong as Gemma 4's on relevant
  domains (F#477 inheritance from sibling MATH.md L3; same assumption,
  unchanged).
- The parent-kill language "killed_00.md" in the pre-reg's notes field is a
  reference to the same experiment-index markdown the sibling cited; no
  semantic divergence.
