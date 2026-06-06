# PAPER.md — exp_hedgehog_adapter_sql_domain_impl

**Verdict:** PROVISIONAL — BLOCKED-on-resource, model-cache sub-form (F#768
super-family). 0 measurements; KCs K1957/K1958 untested.

**Iteration scope:** design-only IMPL filing under F#768 BLOCKED-on-resource pattern.
Parent F#718 PROVISIONAL design-locked at `micro/models/exp_hedgehog_adapter_sql_domain/`.
KCs inherit verbatim from DB K#1957 (proxy PPL) + K#1958 (target PostgreSQL EXPLAIN
dual-ground-truth judge), F#666-paired.

---

## 1. Prediction-vs-measurement table

| KC | Prediction (parent §5) | Measurement | Status |
|---|---|---|---|
| K1957 | `PPL(Hedgehog) ≈ 0.95–1.02 × PPL(base+generic LoRA)` — matched-or-better; PASS expected | not measured | untested |
| K1958 | query-correctness Δ ∈ [+4, +9] pp vs base; mean prediction +6 pp | not measured | untested |

All-untested by design — refusal scaffold runs in <2s, never loads teacher, never
runs Phase A/B/Baseline/C/D.

## 2. KC resolution under F#666

- **K1957 (PPL proxy):** untested. Cannot evaluate without Hedgehog adapter (Phase B
  output) and generic-LoRA baseline (Phase Baseline output).
- **K1958 (PostgreSQL EXPLAIN dual-ground-truth target):** untested. Cannot evaluate
  without judge run on adapter-vs-base generations.
- **F#666 mapping (parent §4):** SUPPORTED requires both PASS, KILLED requires both
  FAIL. With both untested, no verdict beyond PROVISIONAL is admissible.

## 3. Measurement blockers

### Primary — model cache absent (matches F#768)

The 26B Gemma 4 teacher (`mlx-community/gemma-4-26b-a4b-it-4bit`, ~14 GB) is not
present in `~/.cache/huggingface/hub/`. Hedgehog per-layer cos-sim distillation
(Wang 2024 arxiv:2604.14191) requires the teacher to be capacity-strictly-greater
than the student so that the per-layer attention target carries information beyond
the student's representational ceiling. Silent substitution to E4B-as-teacher would:

1. Collapse the cos-sim objective into a noisy identity map (teacher attention
   bounded by student capacity → distillation signal degenerate).
2. Violate researcher antipattern (m) "proxy-model-substituted-for-target".
3. Mismatch parent §0 explicit teacher pin.

**Refused.** This refusal is the correct action under F#768 precedent.

### Secondary — compute budget exceeds drain iteration

Even with cache present, parent §6 pipeline (Phase 0 dataset curation +
Phase A teacher attention capture across 42 E4B layers + Phase B 800-step LoRA
training + Phase Baseline generic LoRA + Phase C PPL head-to-head 3-config + Phase D
blind-paired PostgreSQL EXPLAIN-dual-ground-truth judge 50 prompts × 2 conds) totals
4–6 h. Researcher single-iteration cap is 30 min per guardrail 1009. This advisory
parallels F#769 compute-budget sub-form but is here strictly secondary to the
model-cache blocker.

## 4. Decision rationale — why PROVISIONAL not KILLED, not RELEASED

**Why not KILLED (F#666 + F#702 reasoning):**
- KCs are F#666-paired with non-trivial target gating: K1958's PostgreSQL EXPLAIN
  dual syntactic+semantic hard-floor is the strictest behavioral ground truth across
  the 4-axis hedgehog domain sub-family (per F#718 caveats). Strict KILL on
  monotonic prior alone discards this novel-structure signal.
- F#702 hygiene-patch path remains AVAILABLE (parent A9): K1958 is target, not
  F#666-pure-saturated. `mem-impossibility-f666pure-saturation-implies-f702-unavailable`
  does NOT fire here.

**Why not RELEASED-to-OPEN:**
- Doom-loop avoidance: prior 2 researcher iterations (knowledge_gap_26b on F#768,
  long_context on F#769) demonstrated PROVISIONAL escalation as the structurally
  successful escape from RELEASE-to-OPEN repeats. Releasing this _impl back to
  OPEN with the same blocker would not advance the cohort — it would leave 4
  sibling _impls (JS/Python/Rust/SQL) all pinned on the same 26B cache absence.

**Why PROVISIONAL is the right action:**
- Refusal scaffold + locked KCs + reclaim path documented + no silent substitution
  = the F#768 canonical filing pattern. Reviewer can route to PROVISIONAL via
  two-step (`experiment update --status provisional` + `experiment evidence
  --verdict inconclusive`).
- Per F#769 closing-note "further BLOCKED-on-resource P=3 macros should reuse
  F#768/F#769, not register new finding numbers (ledger-explosion antipattern)" —
  no new finding number is requested. Reviewer's filing options:
  (a) append super-family evidence to F#768 (recommended), OR
  (b) skip finding registration entirely (filing pattern is now canonical and
      4-axis cohort-level recurrence is the expected default, not novel).

## 5. Verdict-consistency pre-flight (PLAN.md §1)

1. `results.json["verdict"] = "PROVISIONAL"` ✓ (not KILLED, not silently SUPPORTED).
2. `results.json["all_pass"] = false` ✓ (correct for PROVISIONAL with 0 measurements).
3. PAPER.md verdict line is `PROVISIONAL` ✓.
4. `is_smoke = false` ✓ (this is design-only filing, not a smoke run).
5. No KC mutation (K1957/K1958 byte-for-byte from DB) ✓.
6. Antipattern scan (MATH.md §5): all rows OK or N/A ✓.

## 6. Cohort observation (advisory, not a finding)

This is the 1st of 4 sibling _impl experiments to be processed in the drain cycle:

- `exp_hedgehog_domain_adapter_js_impl` (still OPEN P=3)
- `exp_hedgehog_adapter_python_domain_impl` (still OPEN P=3)
- `exp_hedgehog_adapter_rust_domain_impl` (still OPEN P=3)
- `exp_hedgehog_adapter_sql_domain_impl` (this filing)

All 4 share the identical 26B teacher cache blocker. A single `huggingface-cli
download` unblocks the entire cohort plus `exp_model_knowledge_gap_26b_base`
(F#768 macro-scope sibling). Reclaim path filed as runtime task `cache:26b-gemma4-teacher`
to make the cohort tractable without per-experiment escalation.

## 7. Assumptions (delta from parent — see parent §8 for A1–A10)

- **D1.** PROVISIONAL is the most defensible verdict for this iteration; KILL
  on monotonic prior alone would discard the K1958 PostgreSQL EXPLAIN structural-
  novelty signal.
- **D2.** Reviewer should NOT register a new finding (per F#769 closing-note).
- **D3.** Cohort-level reclaim task filed at runtime-task layer (`ralph tools task
  ensure --key cache:26b-gemma4-teacher`).

## 8. Suggested follow-ups

1. Once cache lands, single dedicated session executes parent §6 verbatim — no
   redesign needed.
2. If K1957 PASS but K1958 FAIL after `_impl` runs, sibling
   `exp_hedgehog_loss_variant_kl_div_impl` (still open P=3) is the natural next
   step (KL+cos-sim combined loss).
3. Triple-composition child `exp_hedgehog_triple_composition_3domain` (still OPEN
   P=2) gates on JS+Python+SQL _impls landing — entire cascade unblocks together.

## 9. Antipattern scan summary

See MATH.md §5. All rows OK or N/A; no antipattern triggered for this design-only
PROVISIONAL filing under F#768 model-cache sub-form precedent.

## 10. Doom-loop note

Prior 2 researcher iterations:
- N-2: `exp_model_knowledge_gap_26b_base` PROVISIONAL F#768 (model-cache)
- N-1: `exp_model_long_context_adapter_stability` PROVISIONAL F#769 (compute-budget)
- N (this iter): `exp_hedgehog_adapter_sql_domain_impl` PROVISIONAL [reuse F#768]

3rd consecutive PROVISIONAL escalation but on 3 different mechanisms (scaling-law /
range-extrapolation / per-layer cos-sim distillation). Not a literal A→B→A→B
doom-loop. Structurally orthogonal content axes preserve learning despite shared
"resource-blocked design-only" filing pattern. The `experiment claim researcher`
queue is returning P=3 work because P≤2 are claimed-or-blocked elsewhere — this is
the cohort drain shape, not a researcher loop pathology.
