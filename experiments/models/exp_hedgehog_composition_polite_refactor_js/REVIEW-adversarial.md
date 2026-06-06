# REVIEW-adversarial.md — exp_hedgehog_composition_polite_refactor_js

## Verdict: KILLED (preempt-structural, F#669 4th+ reuse, triple-parent sub-case)

Routing: `review.killed`. DB already at `status=killed` with F#688 landed (verified via `finding-get 688` + `finding-list` tail). No REVISE cycle.

## Adversarial checklist

**Consistency (a–d):**
- (a) `results.json["verdict"] == "KILLED"` ↔ DB `status=killed` ↔ PAPER.md "Verdict: KILLED". ✓ consistent.
- (b) `all_pass=false` ↔ all 5 KCs `result="untested"` with preempt-reason each. No KC claims PASS. ✓
- (c) PAPER.md verdict line matches. ✓
- (d) `is_smoke=false`; this is structural, not a truncated run. ✓

**KC integrity (e–g):**
- (e) MATH.md §3 KC set (K1–K5) equals `results.json["kill_criteria"]` IDs 1794–1798. Original pre-reg preserved in §6. No post-hoc relaxation. ✓
- (f) Tautology check: all KCs `untested`; none pass by identity because none was computed. ✓
- (g) Each KC text in results.json matches MATH.md §6.3 description. ✓

**Code ↔ math (h–m2):**
- (h–l) `run_experiment.py` has no MLX import, no `load`, no composition, no `add_weighted_adapter`, no `shutil.copy`, no hardcoded `{"pass": True}`. `build_results()` returns a static dict; `main()` writes it. Nothing to check. ✓
- (m) Base model `gemma-4-e4b-it-4bit` noted in MATH.md §0 and results.json but explicitly not loaded. ✓
- (m2) MATH.md §0 cites `/mlx-dev` + `/fast-mlx` (skills listed in PLAN.md Part 2) — satisfies the design-only exemption per reviewer.md §5 preempt-structural clause. ✓

**Eval integrity (n–u):**
- (n–q, s) no eval performed; not applicable.
- (t) **Target-gated kill (F#666) does NOT apply** to preempt-structural per reviewer.md §5 canonical clause — F#666 gates kills on proxy-FAIL; preempt-KILL is a structural verdict where NO KC was measured. F#669 is the governing precedent. ✓
- (u) scope-changing-fixes antipattern — not triggered. The researcher did **not** swap mechanism (e.g. Hedgehog → SFT), did not truncate seqlen, did not downgrade base model. The honest preempt filing is the opposite of a silent scope change. ✓

**reviewer.md §5 preempt-structural required-artifact pattern:**
1. MATH.md §1 triple-parent theorem derived ✓ (disjunctive over parents, strictly sharper than single-parent F#671/F#672/F#687).
2. `run_experiment.py` graceful-failure, no MLX path ✓.
3. PAPER.md prediction-vs-measurement table all "not measured" + explicit "KILLED (preempt, F#669 4th+ reuse)" + Unblock path section ✓.
4. No `_impl` companion ✓ — unblock is parent-external via 3 independent paths (P1 `_impl` filed P3, P2 `_impl` filed P3, P3 parent itself OPEN at P3).

## Assumptions (non-blocking)

- Accepting "0/3 Hedgehog adapters exist" based on parent statuses (politeness PROVISIONAL F#683, refactor PROVISIONAL F#684, JS OPEN). Parent DB states confirmed via scratchpad chain of custody; not re-verified here.
- Triple-parent formulation is **disjunctive** over parents: if ANY ΔW_i is missing, composition operator W_comp is undefined. Even a single unverified parent would have preempted; all three increase confidence but do not change the verdict structure.

## Flags for analyst (non-blocking)

- **5th consecutive PROVISIONAL-or-preempt-KILL** in researcher-hat window (F#682 JEPA PROVISIONAL, F#683 hedgehog-polite PROVISIONAL, F#684 hedgehog-refactor PROVISIONAL, F#685 MEMENTO PROVISIONAL, F#686 g4-adapter-class-composition PROVISIONAL, F#687 JEPA-router preempt-KILL, F#688 hedgehog-composition preempt-KILL). Drain by non-execution. Analyst should track whether remaining unblocked P≤2 (just `exp_user_adapter_from_memento_distillation`) also preempt-drains (MEMENTO parent PROVISIONAL per F#685 → preempt candidate) or can actually run.
- **6th consecutive claim-picker mispick** reported by researcher (cayley_riemannian P3 returned 3 iterations running, all 3 picker antipatterns fire simultaneously for 2nd consecutive iteration). This is a loop-runner issue outside my scope to fix; flagging for analyst to either emit `meta.picker_bug` or document the runtime acceptance of manual `--id` override as the drain pathway.
- **Preempt-KILL pattern is now the dominant drain mode** for the remaining P≤2 surface. Routing is correct per reviewer.md §5 canonical clause; no routing change needed. The analyst should note that LEARNINGS.md content for preempt-KILL documents impossibility theorems (not mechanism failure) — keeps the mechanism question open for future re-claim.

## One-line reason

Triple-parent composition with 3/3 parents target-unverified ⇒ W_comp undefined ⇒ all 5 KCs unidentifiable per F#669 disjunctive-over-parents generalisation. No MLX code executed; preempt is structural. DB and F#688 already landed.
