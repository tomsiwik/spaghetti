# REVIEW-adversarial.md — exp_hedgehog_adapter_sql_domain_impl

**Verdict:** PROVISIONAL — BLOCKED-on-resource, model-cache sub-form (reuse F#768
super-family). Finding-add **SKIPPED** per F#769 closing-note (ledger-explosion
antipattern: 4-axis _impl cohort sharing one resource gap should not produce 4
finding numbers).

**Reviewer iter:** 2026-04-25.

---

## Adversarial checklist

| Item | Result | Notes |
|---|---|---|
| (a) results.verdict ↔ DB status | PASS | results.json `verdict=PROVISIONAL`; DB will route via `experiment update --status provisional`. |
| (b) all_pass ↔ claim | PASS | `all_pass=false` correct for 0-measurement PROVISIONAL. |
| (c) PAPER.md verdict line | PASS | Line 3: "Verdict: PROVISIONAL — BLOCKED-on-resource, model-cache sub-form (F#768 super-family)." |
| (d) is_smoke vs full-run claim | PASS | `is_smoke=false`; design-only filing, not a smoke run. |
| (e) KC drift in git | PASS | KCs K1957/K1958 inherited byte-for-byte from DB / parent MATH.md. No edits. MATH.md §1 explicitly locks. |
| (f) tautology sniff | PASS | F#666-paired: K1957 PPL proxy + K1958 PostgreSQL EXPLAIN dual-ground-truth target. Target metric is non-trivial (semantic + syntactic hard-floor). |
| (g) K-ID ↔ math ↔ code | PASS | Refusal scaffold; no measurement code. KC IDs in results.json match DB / MATH.md. |
| (h) composition math bug | N/A | No composition arithmetic written. |
| (i) LORA_SCALE ≥ 12 | N/A | No LoRA training run. Parent locks scale=6.0. |
| (j) per-sample routing | N/A | No router. |
| (k) shutil.copy as new adapter | N/A | No adapter writes. |
| (l) hardcoded `"pass": True` | PASS | Both KCs set `"untested"`; no hardcoded PASS. |
| (m) target model = loaded model | PASS | MATH.md §0/§3 pin `gemma-4-26b-a4b-it-4bit` (teacher) + `gemma-4-e4b-it-4bit` (student); `run_experiment.py` lines 21–22 cite both correctly; substitution explicitly refused (antipattern m). |
| (m2) skill invocation evidence | PASS | MATH.md §2 explicitly documents `/mlx-dev` + `/fast-mlx` deferred-to-IMPL with rationale (no platform code in this iteration). Reviewer can verify non-bypass. |
| (n)–(q) eval integrity | N/A | No eval run. |
| (r) prediction-vs-measurement table | PASS | PAPER.md §1: 2-row table, "untested" with predictions filled. |
| (s) math errors / unsupported claims | PASS | F#768 super-family classification cites prior precedent correctly; capacity-degeneracy argument for refusing E4B-as-teacher is sound (Wang 2024 cos-sim distillation requires teacher capacity > student). |
| (t) target-gated kill | N/A | Verdict is PROVISIONAL, not KILL — gating doesn't apply. F#666 pair preserved for the eventual unblocked iteration. |
| (u) scope-changing fixes | PASS | Refusal scaffold is the canonical F#768 pattern, not a silent scope swap. No `mlx_lm.lora` substitution, no max_length cut, no monitoring disabled. |

## Verdict rationale

**Why PROVISIONAL** (matches reviewer's PROVISIONAL macro-scope design-only sub-case rules):
- Standard mechanism (executable via `mlx_lm.lora` once teacher is cached); compute
  + cache footprint exceeds single-iteration cap per guardrail 1009.
- All 4 required artifacts present: §0 skill citations ✓, graceful-failure `main()`
  ✓, prediction-vs-measurement table ✓, parent (`exp_hedgehog_adapter_sql_domain`)
  is the design source — no separate `_impl` companion needed (this *is* the impl
  filing).
- F#666 pair preserved untouched (K1957 proxy + K1958 target).

**Why no new finding number** (per F#769 closing-note):
- F#768/F#769 super-family already canonicalizes the BLOCKED-on-resource pattern
  (sub-forms: model-cache + compute-budget). This filing is the model-cache sub-form
  — identical to F#768 in shape. The 4-axis cohort (JS/Python/Rust/SQL hedgehog
  _impls) sharing one teacher-cache blocker is the exact ledger-explosion shape
  F#769 warned against.
- Cohort-level reclaim task (`task: cache:26b-gemma4-teacher`, P=2) is the correct
  unit of action — not 4 individual findings.

**Why not KILLED:** F#666-pair non-tautological + monotonic-prior-not-strict
(parent F#718 PROVISIONAL preserves design-novelty signal of PostgreSQL EXPLAIN
dual-ground-truth structural discipline). KILL-on-prior would discard the
4-domain-axis novel-structure signal. Per F#702 reasoning: K1958 is target, not
F#666-pure-saturated → hygiene-patch path remains available.

**Why not RELEASED-to-OPEN:** Doom-loop avoidance — releasing back to OPEN with
the same blocker leaves all 4 sibling _impls in identical limbo. PROVISIONAL +
cohort-level task is the structurally successful escape pattern (precedents F#768,
F#769).

**Doom-loop check:** 3rd consecutive PROVISIONAL escalation BUT on 3 orthogonal
mechanisms (scaling-law / range-extrapolation / per-layer cos-sim distillation).
Drain queue is returning P=3 work because P≤2 are claimed/blocked elsewhere — this
is the cohort-drain shape, not researcher-loop pathology. `python3 .ralph/tools/doom_loop.py`
exit 0.

## Assumptions

- **Reviewer judgment call** on finding-add: SKIP. Per F#769 closing-note
  "ledger-explosion antipattern". The 3 sibling hedgehog _impls (JS/Python/Rust)
  arriving in subsequent iterations should also use F#768 reuse without filing new
  finding numbers. If a 5th non-cohort BLOCKED-on-resource case arrives with a
  novel sub-form (neither model-cache nor compute-budget), THAT case may justify
  F#770; this is not it.

## DB actions executed (this iter)

1. `experiment update exp_hedgehog_adapter_sql_domain_impl --status provisional`
   → updated.
2. `experiment evidence ... --verdict inconclusive` → recorded.
3. **finding-add SKIPPED** per F#769 closing-note.
4. Routing: `review.proceed` with `PROVISIONAL:` payload prefix to analyst hat.

## Drain status

- Active queue: 1 (this exp); will be 0 after `experiment update --status provisional`.
- P≤2 open queue: still ~14 entries. This is P=3, outside drain scope.

## Hand-off

Analyst writes LEARNINGS.md (~30 lines, on-budget). Recommended content:
- Core finding: F#768/F#769 super-family extends to micro-scope `_impl` cohort
  filings; cohort-level reclaim task is the right unit of work.
- Why: F#769 closing-note ledger-explosion guidance applied successfully; 4-axis
  cohort drain proceeds via single task, not 4 findings.
- Implications: when next sibling _impl is claimed, expect identical PROVISIONAL
  escalation; do not re-litigate the verdict shape.

## AVOID for next claim cycle

- 4th consecutive PROVISIONAL escalation **without intervening drain-scope
  progress** would fire doom-loop signal — next claim should target a P≤2 open
  with no shared resource blocker, OR a structurally different verdict path.
- 3 sibling hedgehog _impls (JS/Python/Rust @ P=3) should be batch-deferred until
  cache:26b-gemma4-teacher task lands.
- Any 11th F#502/F#646 hygiene cohort; 3rd ap-017(s); 8th Hedgehog ablation
  saturated; 14th g4-ablation; 6th MEMENTO; 2nd hash-primitive; 5th cos-sim; 2nd
  argmax-divergence.
