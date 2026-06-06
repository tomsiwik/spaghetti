# MATH.md — exp_hedgehog_adapter_sql_domain_impl

**IMPL follow-up to** `exp_hedgehog_adapter_sql_domain` (parent F#718 PROVISIONAL,
2026-04-24). Design fully inherited from parent MATH.md byte-for-byte; the present
file documents only the IMPL-iteration delta and the resource-blocker handling under
the F#768/F#769 BLOCKED-on-resource super-family.

---

## 0. Inheritance pointer

The parent design is locked at `micro/models/exp_hedgehog_adapter_sql_domain/MATH.md`
(255 lines). All sections — claim, failure modes, prior-math citations, derivation
sketch, KC map, eval set, judge rubric, protocol, assumptions A1–A10 — apply verbatim
to this `_impl` and must NOT be re-edited here per KC-lock discipline (verdict-
consistency check #5). KC text is also locked in the DB at K#1957 (proxy) and K#1958
(target).

The IMPL iteration's job is exclusively to execute the locked protocol once the
prerequisite resources are present.

## 1. KC inheritance (locked text — no edits)

| KC | Text (DB-canonical) | Type |
|---|---|---|
| K1957 | "SQL adapter PPL on SQL-specific eval > base + generic LoRA" (KILL if PPL strictly greater) | proxy (PPL) |
| K1958 | "SQL query correctness auto-judge < +5pp vs base" (KILL if Δ < +5pp) | target (PostgreSQL EXPLAIN dual syntactic+semantic hard-floor) |

**F#666 target-gating preserved** — K1957/K1958 carry the same proxy/target pairing
as parent K#1868/K#1869. Verdict mapping per parent §4 §F#666 unchanged.

## 2. Platform skill citation (PLAN.md §1011/1012)

- `/mlx-dev` and `/fast-mlx` are the gates the `_impl` execution iteration MUST clear
  before any MLX training-loop code is written. **Not invoked in this iteration**:
  this iteration files PROVISIONAL because the prerequisite 26B teacher cache is
  absent (§3 below). No platform code is written; therefore skill invocation is
  correctly deferred to the unblocked iteration that downloads the teacher and runs
  the protocol. Cited explicitly so the reviewer can verify `/mlx-dev` / `/fast-mlx`
  are not silently bypassed when actual training code lands.

## 3. Resource blocker — F#768/F#769 super-family classification

**Sub-form:** model-cache sub-form (matches F#768 byte-for-byte, distinct from F#769
compute-budget sub-form).

**Detail:**
- Required teacher: `mlx-community/gemma-4-26b-a4b-it-4bit` (~14 GB).
- HF cache state at iteration start: confirmed absent via `ls
  ~/.cache/huggingface/hub/`. Available Gemma variants: `gemma-2-2b-it-4bit`,
  `gemma-4-e2b-it-4bit`, `gemma-4-e4b-it-4bit`, `gemma-4-e4b-it-8bit`,
  `google/gemma-4-e4b-it`. The 26B-A4B variant is NOT present.
- Substitution refused: silently swapping teacher to E4B violates researcher
  antipattern (m) "proxy-model-substituted-for-target" — Hedgehog distillation
  specifically requires the teacher to embed the docs-in-context signal at higher
  capacity than the student, otherwise the per-layer cos-sim target is
  capacity-degenerate (teacher's attention is bounded by the student's own
  representational ceiling and the distillation objective collapses to a noisy
  identity map).
- Compute footprint of the prerequisite + the run combined exceeds the
  single-iteration researcher budget per guardrail 1009 (download ≈ 14 GB plus the
  4–6 h Phase 0/A/B/Baseline/C/D pipeline already documented in parent §6).

**Decision:** PROVISIONAL escalation reusing F#768 (model-cache sub-form). NO new
finding number is registered for this iteration — per F#769 closing-note "further
such filings should reuse F#768/F#769, not register new finding numbers
(ledger-explosion antipattern)". This pattern explicitly anticipates the JS/Python/
Rust/SQL `_impl` cohort all sharing the same teacher-cache blocker; filing 4 separate
findings for one resource gap would saturate the ledger without adding learning.

**Doom-loop check:** prior 2 iterations escalated PROVISIONAL on
`exp_model_knowledge_gap_26b_base` (F#768) and `exp_model_long_context_adapter_stability`
(F#769). This is the 3rd consecutive PROVISIONAL escalation of the cycle but on a
different experiment with a different mechanism (Hedgehog distillation, not raw
scaling-law extension). Not a literal A→B→A→B doom-loop. Structurally different
content axis (hedgehog domain-knowledge attention-routing vs scaling-law knowledge
gap vs long-context adapter stability) preserves orthogonal learning despite same
blocker shape.

## 4. Reclaim path (documented in `run_experiment.py` results)

Cohort-level unblock (single download unblocks 4 sibling _impls):

1. Acquire 26B Gemma 4 teacher: `huggingface-cli download
   mlx-community/gemma-4-26b-a4b-it-4bit` — ~14 GB to `~/.cache/huggingface/hub/`.
2. Verify cache: `ls ~/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/`.
3. Schedule ≥ 6 h dedicated session per parent §6.
4. Bump experiment priority P=3 → P=2 to enter drain scope.
5. Invoke `/mlx-dev` + `/fast-mlx` skills.
6. Execute parent §6 Phase 0/A/B/Baseline/C/D verbatim.
7. Apply parent §6 KC measurements at K1957/K1958.

This same reclaim path applies to siblings:
`exp_hedgehog_domain_adapter_js_impl`, `exp_hedgehog_adapter_python_domain_impl`,
`exp_hedgehog_adapter_rust_domain_impl`. Single download serves all 4.

## 5. Antipattern scan (this iteration)

| Antipattern | Status | Reason |
|---|---|---|
| (a) composition math bug | N/A | No composition math written; design only. |
| (b) tautological routing | N/A | No router code written. |
| (c) unsafe LORA_SCALE | N/A | No LoRA training run; parent locks 6.0 (≤ 8). |
| (d) KC-swap-after-failure | OK | KCs locked at K1957/K1958 verbatim from DB. |
| (e) verdict-DB mismatch | will-verify | scaffold writes `verdict=PROVISIONAL` and `experiment update --status provisional` is reviewer's call. |
| (f) smoke-as-full | OK | `is_smoke=false`; this is a refusal scaffold not a smoke run. |
| (g) tautological KC | OK | F#666 paired proxy + behavioral target (PostgreSQL EXPLAIN hard-floor). |
| (h) thinking-mode truncation | N/A | No generation in this iteration. |
| (i) wrong-model proxy | OK | Refusal scaffold explicitly rejects substitution; `BASE_MODEL_ID = mlx-community/gemma-4-e4b-it-4bit` and `TEACHER_MODEL_ID = mlx-community/gemma-4-26b-a4b-it-4bit` both verified against parent §0. |
| (j) synthetic padding | N/A | No data generated. |
| (k) `shutil.copy` as new adapter | N/A | No adapter writes. |
| (l) hardcoded `"pass": True` | OK | Both KCs set to `untested` in results.json. |
| (m) eval-template truncation | N/A | No eval run. |
| (m2) skill invocation evidence | OK | §2 above documents deferral with rationale. |

## 6. Assumptions (delta from parent)

- **D1.** This iteration's PROVISIONAL verdict is conditional on the F#768 super-
  family pattern remaining the most defensible action. If the reviewer judges that
  KILL-on-prior is appropriate (e.g., a new cohort-level KILL precedent emerges),
  the verdict can be downgraded — but K1957/K1958 are F#666-paired with non-trivial
  target gating, so a strict KILL would discard the cohort's structural-novelty
  signal (PostgreSQL EXPLAIN hard-floor is novel across the 4-domain-axis sub-family
  per F#718).
- **D2.** Reviewer's choice on finding registration: per F#769 closing-note this
  filing should NOT register a new finding. If the reviewer judges that the cohort
  scale (4 _impls × 1 shared blocker) warrants a single super-family note (NOT 4
  individual filings), that note should be appended to F#768 evidence, not a new F#.
- **D3.** A runtime task `task: cache:26b-gemma4-teacher` (cohort-level unblocker)
  should be filed via `ralph tools task ensure` so the JS/Python/Rust/SQL _impl
  cohort and `exp_model_knowledge_gap_26b_base` reclaim path become tractable in a
  future scheduled session. Researcher iteration files this task before emitting
  `experiment.done`.
