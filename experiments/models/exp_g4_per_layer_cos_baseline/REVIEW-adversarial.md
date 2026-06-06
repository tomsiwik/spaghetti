# REVIEW-adversarial.md — exp_g4_per_layer_cos_baseline (reviewer independent pass)

_Overwrites researcher self-review per precedent. Reviewer's independent adversarial checklist._

## Verdict: KILL (preempt-structural, F#666-pure standalone)

1st instance of row-3 sub-case (F#666-pure, no parent dep). Orthogonal to F#669 family (parent-unverified). All (a)–(u) PASS.

## Consistency & integrity

- (a) results.json `verdict=KILLED` ↔ DB `status=killed` ↔ PAPER.md "Verdict: KILLED" ↔ MATH.md §1 theorem. Four-way consistent.
- (b) `all_pass=false`; single KC `result=untested`. No PASS/FAIL claim to contradict.
- (c) PAPER.md verdict line matches DB. No "PROVISIONAL"/"NOT SUPPORTED"/"INCONCLUSIVE" drift.
- (d) `is_smoke=false`. Preempt is NOT a smoke substitution.
- (e) Dir `micro/models/exp_g4_per_layer_cos_baseline/` is fresh (untracked on entry). MATH.md §3 KC1856 text matches DB pre-reg byte-for-byte.
- (f) No tautology — K1856 never evaluated; preempt occurs before measurement precisely to avoid F#666-tautological verdict.
- (g) N/A — no prior run to back out of.

## Code ↔ math

- (h)-(l) **Vacuously satisfied.** `run_experiment.py` imports only `json` + `pathlib`. No MLX, no LoRA, no safetensor ops, no `add_weighted_adapter`, no `shutil.copy`, no hardcoded KC dicts, no `LORA_SCALE`, no routing. `main()` writes results.json directly. Honest preempt form.
- (m) Base model `mlx-community/gemma-4-e4b-it-4bit` pinned per F#627 in MATH.md §0 with "Not loaded" disclosure. Canonical.
- (m2) MATH.md §0 cites `/mlx-dev` + `/fast-mlx` per PLAN.md Part 2 with "Not invoked — no MLX code written" disclosure. Canonical preempt-form.

## Eval integrity

- (n)-(q) Vacuously satisfied — no eval pipeline, no sample count, no baseline drift.
- (r) PAPER.md contains prediction-vs-measurement table (K1856 row, "not measured", untested verdict). ✓
- (s) §1 theorem: classification of K1856 as proxy → F#666 gating → corollary → QED. Structure sound; no unsupported algebraic claims.

## F#666 / F#669 routing check

- (t) **F#666 routing applies at the structural level, NOT as a kill-block.** Regular (t) blocks proxy-FAIL→KILL when no target KC is paired; here, the kill is NOT "proxy failed," it is "KC set is structurally malformed, no verdict derivable." This matches the reviewer.md §5 preempt-structural exclusion: "F#666 gates kills on proxy-FAIL; preempt-KILL is a structural verdict where NO KC was measured." F#669 named precedent extends by analogy — governing discipline is the same (structural unidentifiability ⇒ preempt-KILL, no `_impl`).
- F#669 arm absent: `depends_on=[]`. Confirmed via `experiment get`.
- Not F#698-compound: no parent to pair with F#666.
- **New row in sub-case taxonomy** (row 3: F#666-pure standalone). 1st instance. Promotion threshold per drain precedent: 2nd instance → standalone antipattern memory. Do not promote yet.

## Scope-changing fixes (u)

PASS. MATH.md §6 explicitly rejects all three known scope-swap shortcuts:
1. Running the measurement for a proxy-only verdict (would be antipattern-t: structurally invalid verdict).
2. Inventing a target-metric KC post-claim (antipattern-u: post-claim KC mutation).
3. Substituting a simpler proxy (preserves F#666 violation).

KC text preserved verbatim from DB. No silent simplification.

## Secondary pre-reg defects (non-blocking for KILL, flagged for analyst)

Reviewer confirms via `experiment get`:
1. `success_criteria: []` — empty; independently blocks SUPPORTED.
2. `references: []` — guardrail 1002 violation (no arxiv citation, no prior-finding anchor).
3. `platform: null` — MATH.md §0 discipline hole.

Also visible in the CLI's own `⚠ INCOMPLETE: success_criteria, references, platform` warning line. The pre-reg was broadly malformed; F#666 is only the primary blocker. Re-claim via edit-and-augment is mechanically possible but the DB pre-reg should be re-scoped as a Hedgehog-family sibling (per MATH.md §4 recommendation) rather than patched. Analyst may promote "pre-reg hygiene" to a watch-list item if it recurs.

## Follow-up disposition

- `_impl` companion filed? **NO.** Verified via `experiment get exp_g4_per_layer_cos_baseline_impl` → not found. Correct per F#687/F#698/F#699 + reviewer.md §5 preempt-structural exclusion. Unblock is pre-registration-external (edit DB to add target KC), not implementation-external.
- Honoring `mem-antipattern-impl-follow-up-delegation`? Scope-check: antipattern applies to novel-mechanism PROVISIONAL, not preempt-structural KILL. Not triggered here.

## DB actions verified

- `experiment complete --status killed --k 1856:inconclusive` executed by researcher (DB `status=killed` confirmed).
- `experiment finding-add --status killed` → F#700 filed and verified via `experiment finding-get 700`. Summary correctly names "F#666-pure KC-structural preempt-KILL: standalone proxy-only KC set (exp_g4_per_layer_cos_baseline)."
- No reviewer-side DB actions needed.

## Non-blocking notes for analyst

1. **New sub-case observed** — F#666-pure standalone preempt-KILL. 1st instance. Do NOT promote to antipattern memory yet (threshold=2nd). Document in LEARNINGS.md taxonomy (already done by researcher).
2. **Pre-reg hygiene pattern** — this experiment had FOUR simultaneous pre-reg defects (F#666 violation + 3 empty/null fields). If another such broadly malformed pre-reg surfaces, promote "pre-reg hygiene" (empty success_criteria + empty references + unset platform at claim time) to a separate memory. Currently 1st instance; watch.
3. **Taxonomy is now 2×2-complete plus orphan** — F#669 classic / F#669+F#666 compound / F#666-pure standalone / runnable-F#666-compliant. If a 4th novel sub-case appears, the taxonomy itself may warrant refactor.
4. **Hedgehog baseline role obsolete** — per `notes` field, this experiment was meant to baseline Hedgehog distillation claims. Hedgehog family (F#683/F#684/F#696/F#697) has proceeded to PROVISIONAL without it. Re-scoping recommendation in MATH.md §4 stands.

## Drain-wide pattern count (after this iteration)

- 5 novel-mechanism PROVISIONALs (F#682, F#683, F#684, F#696, F#697)
- 6 F#669-family preempt-KILLs (F#669, F#671, F#672, F#687, F#698, F#699)
- **1 F#666-pure standalone preempt-KILL (F#700, this)**
- 3 SUPPORTED (budget_forcing, semantic_router, cayley_riemannian)
- 1 regular KILL (kv_cache_reuse_honest)

## Assumptions

- Preempt-KILL routing generalizes from F#669 (parent-unverified) to F#666-pure (KC malformed) by the structural-impossibility principle: when no measurement outcome yields a valid verdict, preempt. Reviewer.md §5 names only F#669 explicitly; 1st-instance extension by analogy is defensible without promoting the §5 text.
- No reviewer.md §5 edit required at 1st instance — only at threshold (2nd/3rd reuse).

## Result

Emitting `review.killed` → analyst.
