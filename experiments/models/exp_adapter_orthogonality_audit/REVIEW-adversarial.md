# REVIEW-adversarial.md — exp_adapter_orthogonality_audit (reviewer independent pass)

## Verdict: KILL (preempt-structural, F#666-pure standalone, 2nd drain-window instance, F#701)

Independent pass overwrote researcher's self-review per precedent (F#687, F#698, F#699, F#700). All (a)–(u) PASS. Already-DB-killed, F#701 already filed.

## Consistency checks

- **(a)** `results.json.verdict="KILLED"` ↔ DB `status=killed` (verified via `experiment get`) ↔ PAPER.md "Verdict: KILLED" ↔ MATH.md §1 theorem. Four-way consistent.
- **(b)** `all_pass=false`; both KCs `result="untested"`. No contradictory PASS/FAIL claim.
- **(c)** PAPER.md verdict line = "KILLED (preempt, F#666-pure KC-structural — 2nd drain-window instance)". No PROVISIONAL / NOT SUPPORTED / INCONCLUSIVE drift.
- **(d)** `is_smoke=false`. Preempt is a structural verdict, not a smoke substitution.

## KC integrity

- **(e)** Dir `micro/models/exp_adapter_orthogonality_audit/` freshly created this iteration (untracked on entry, confirmed in git status). KC text byte-for-byte matches DB via `experiment get`:
  - K1857: "Pairwise cosine between any two adapter weight matrices > 0.15 (not orthogonal)" — matches MATH.md §1, results.json, PAPER.md.
  - K1858: "Effective rank of N-adapter stack < N * rank/2 (subspace overlap > 50%)" — matches MATH.md §1, results.json, PAPER.md.
- **(f)** No tautology — neither KC was evaluated. Preempt occurs *before* any measurement precisely to avoid F#666-tautological verdict (running + mark PASS/FAIL would be antipattern-t per MATH.md §6).
- **(g)** N/A — no prior run exists to back out of.

## Code ↔ math

- **(h)–(l)** Vacuously satisfied. `run_experiment.py` imports only `json` + `pathlib`. Zero MLX, zero LoRA, zero safetensor ops, no `add_weighted_adapter`, no `shutil.copy`, no `LORA_SCALE`, no routing, no adapter loading, no hardcoded KC-pass dicts. `main()` writes `results.json` and exits. Honest preempt form.
- **(m)** Base model `mlx-community/gemma-4-e4b-it-4bit` pinned per F#627 in MATH.md §0 with "Not loaded" disclosure. Adapters that WOULD have been audited listed explicitly (`exp_p1_t2_single_domain_training/adapters/{code,math,medical}/adapters.safetensors` + ~10 others) with "Not loaded" qualifier. No proxy substitution.
- **(m2)** MATH.md §0 cites `/mlx-dev` + `/fast-mlx` (per PLAN.md Part 2) with canonical "Not invoked — no MLX code written" disclosure. Matches F#700 precedent form. No unidiomatic MLX code to flag (nothing was written).

## Eval integrity

- **(n)–(q)** Vacuously satisfied — no eval pipeline, no sample count, no baseline drift, no synthetic padding.
- **(r)** PAPER.md contains prediction-vs-measurement table with K1857 + K1858 rows, "not measured", verdict "untested". ✓
- **(s)** MATH.md §1 theorem contains explicit 2×2 truth table over {K1857, K1858} ∈ {PASS, FAIL}:
  - (P, P): proxy-SUPPORT with no target pair → tautological
  - (P, F), (F, P): mixed-proxy, no target → ambiguous, cannot KILL
  - (F, F): per F#666, "Proxy-FAIL ... = finding about proxy, not kill"
  
  All four cells unidentifiable. QED is step-by-step; no unsupported algebraic claims.

## F#666 / F#669 routing check

- **(t)** **F#666 applies as the structural reason for preempt, NOT as a kill-block.** Regular (t) blocks proxy-FAIL→KILL when no target KC is paired. Here the kill is "KC set is structurally malformed; no verdict derivable," not "proxy failed." This matches reviewer.md §5 preempt-structural exclusion clause: *"F#666 gates kills on proxy-FAIL; preempt-KILL is a structural verdict where NO KC was measured."* F#700's reviewer pass explicitly established this routing for the 1st F#666-pure-standalone instance; same reasoning extends.
- F#669 arm absent: `depends_on=[]` (verified via `experiment get`). Not an F#669-family preempt.
- Not F#698-compound: no parent exists to pair with F#666.
- **Row 3 in sub-case taxonomy reaches 2 instances** (F#700 + F#701). Promotion trigger reached — analyst to file AP memories.

## Scope integrity (u)

PASS. MATH.md §6 explicitly rejects all four scope-swap shortcuts:
1. Running the measurement for a proxy-only verdict (antipattern-t: structurally invalid).
2. Inventing a target-metric KC post-claim (antipattern-u: post-claim KC mutation).
3. Swapping to a simpler proxy (e.g. cosine over 2 of 12 adapters) — preserves F#666 violation.
4. Substituting an alternate audit target (e.g. base-model weight cosine) — misrepresents pre-reg.

KC text preserved verbatim from DB. No silent simplification.

## Secondary pre-reg defects (non-blocking for KILL)

Reviewer confirms via `experiment get`:
1. `success_criteria: []` — empty; CLI emits `⚠ INCOMPLETE: success_criteria`.
2. `references: []` — violates guardrail 1002. Notes reference F#562 informally; F#42, F#571 also relevant but not formally registered.
3. `platform: null` — MATH.md §0 discipline violated; CLI emits `⚠ INCOMPLETE: platform`.

**Exact same 4-defect structural shape as F#700** (F#666-violating KC + empty success_criteria + empty references + null platform). 2nd independent pre-reg with identical structural shape → promotion trigger for `AP-prereg-hygiene-multi-defect` antipattern memory.

## Follow-up disposition

- `_impl` companion: **NOT filed.** Verified `experiment get exp_adapter_orthogonality_audit_impl` → 404. Correct per F#687/F#698/F#699/F#700 + reviewer.md §5 preempt-structural exclusion. Unblock is pre-registration-external (edit DB to add target KC or re-register fresh), not implementation-external.
- No `mem-antipattern-impl-follow-up-delegation` obligation — that antipattern applies to novel-mechanism PROVISIONAL only.

## Comparison to F#700 (1st instance)

| Dimension                    | F#700 (`exp_g4_per_layer_cos_baseline`)    | F#701 (this exp.)                              |
| ---------------------------- | ------------------------------------------ | ---------------------------------------------- |
| Parent dependency            | none                                       | none                                           |
| KC count                     | 1 (K1856 cos-sim variance)                 | 2 (K1857 cosine + K1858 effective rank)        |
| KC kinds                     | proxy (variance)                           | proxy (cosine) + proxy (rank)                  |
| F#666 violation              | yes                                        | yes                                            |
| success_criteria             | []                                         | []                                             |
| references                   | []                                         | []                                             |
| platform                     | null                                       | null                                           |
| `_impl` filed                | no                                         | no                                             |
| Antipattern promotion        | 1st — watchlist                            | **2nd — promote**                              |

Structural identity across two independent pre-regs confirms the pattern is stable — not coincidence.

## Non-blocking notes for analyst

1. **Promotion trigger reached.** File two antipattern memories this iteration:
   - `AP-F666-pure-standalone`: `depends_on=[]` + proxy-only KC set → preempt-KILL at claim.
   - `AP-prereg-hygiene-multi-defect`: ≥3 empty/null hygiene fields + F#666 violation → structurally unsalvageable; preempt-KILL + recommend re-register.
2. **Do NOT edit reviewer.md §5 yet.** F#669-family analogy (structural-impossibility principle) covers preempt-structural KILL at the rule level. Consider explicit F#666-pure clause only at 3rd instance.
3. **Drain tally after this iteration:**
   - 5 novel-mechanism PROVISIONALs (F#682, F#683, F#684, F#696, F#697)
   - 6 F#669-family preempt-KILLs (F#669, F#671, F#672, F#687, F#698, F#699)
   - **2 F#666-pure standalone preempt-KILLs (F#700, F#701)** ← promotion threshold reached
   - 3 SUPPORTED (budget_forcing, semantic_router, cayley_riemannian)
   - 1 regular KILL (kv_cache_reuse_honest)
   - Total drained this window: **17**.
4. **Taxonomy refactor watchlist**: row 3 at 2 instances. A 3rd instance would warrant consolidating F#666-pure-standalone + F#669-family preempt-KILLs into a unified "KC-structural preempt-KILL" super-category.
5. **Doc-drift micro-observation**: `notes` field "Empirical audit of F#562 claim..." (per `experiment get`) does not flag the F#666 violation — consistent with pre-reg design process not engaging F#666 prior art (fits AP-prereg-hygiene-multi-defect pattern).

## Assumptions

- `experiment complete --status killed` + `experiment finding-add --status killed` already executed by researcher before handoff (verified via `experiment get` showing `status: killed` and `experiment finding-get 701` returning full record). Reviewer does not re-run.
- Both antipattern memories to be filed by analyst during LEARNINGS pass (researcher documented the promotion trigger; analyst executes `ralph tools memory add`).

## Verdict

**KILL (preempt-structural, F#666-pure standalone, 2nd drain-window instance, F#701).** All checks PASS. Route `review.killed` → analyst.
