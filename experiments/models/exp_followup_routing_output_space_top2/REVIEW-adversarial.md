# REVIEW-adversarial.md — exp_followup_routing_output_space_top2

**Verdict:** KILL (preempt-structural, tautological-inter-variant-delta)
**Reviewer:** independent pass (overwrites researcher self-review per F#700/F#701/F#703 precedent).
**Finding:** F#704 filed and verified via `experiment finding-get 704`.

## Adversarial checklist (a)–(u)

- **(a) results.json verdict vs DB.** `results.json["verdict"] = "KILLED"` ↔ DB `status=killed`. Byte-match. PASS.
- **(b) all_pass.** `false`. Correct for preempt-KILL (no KC measured = no KC can pass). PASS.
- **(c) PAPER verdict line.** `**Status:** KILLED — preemptive, structurally uninformative KC.` No inconclusive/provisional/partial language. Consistent with DB `status=killed`. PASS.
- **(d) is_smoke.** `false`. Preempt is structural; no subsampled measurement. PASS.
- **(e) KC stability (git diff).** DB text: `QA-format + cache-aware top-2 beats NTP+swap-per-token baseline on QA by >=5pp`. MATH.md §3: `"QA-format + cache-aware top-2 beats NTP+swap-per-token baseline on QA by >=5pp"`. Byte-equivalent; no post-claim KC mutation. PASS.
- **(f) Tautology sniff test.** K1577 IS tautological (L1: NTP adapters cannot emit MCQ letters by training-distribution construction). This is the *reason* for preempt — MATH.md §1 L1 and PAPER.md §Why preempt flag it explicitly. Preempt-disclosure valid. PASS.
- **(g) K-ID in code vs DB.** N/A — no measurement code executed. `run_experiment.py` is a graceful-failure stub (json+pathlib only). Stub writes K1577 with `result="untested"`. PASS.
- **(h)–(m) Composition / LORA_SCALE / per-sample routing / shutil / hardcoded PASS / proxy-substitution.** N/A — no MLX surface, no composition code, no adapter load. PASS.
- **(m2) Platform skill invocation.** MATH.md §0: `/mlx-dev` and `/fast-mlx` "not invoked — no MLX code is written in this experiment." Canonical preempt disclosure matching F#700/F#701/F#703. PASS.
- **(n)–(q) Eval integrity.** N/A — no measurement. PASS.
- **(r) Prediction-vs-measurement table.** Present in PAPER.md (5 rows, all "not measured / preempt"). PASS.
- **(s) Mathematical soundness.** Five-lemma preempt verified independently:
  - **L1** (tautological-KC): F#165 measured `Q(A_NTP) = 0.410` on Falcon-E-3B MMLU, −24pp vs base. NTP adapters emit continuation prose; QA-format adapters emit letter answers by construction. Delta ≥5pp trivial. Sound.
  - **L2** (prerequisite-gate): F#166 verbatim quote reads correctly; prerequisite-gate is canonical per the experiment's own cited governing finding. Sound.
  - **L3** (base-beat unlikely): F#477 independently verified (2/5 base-beat rate on Gemma 4, K1226 FAIL adapted acc 0.480 < 0.50). Falcon-E-3B-weaker-prior assumption flagged in PAPER.md §Assumptions. Inverting assumption flips only L3, not L1/L2/L4/L5; ∨-disjunction still holds. Sound.
  - **L4** (bundled fixes): QA-format + KV-cache-aware are independent remedies, bundled into one delta. Standard attribution hygiene. Sound.
  - **L5** (near-duplicate): Verified independently via `experiment get exp_followup_output_space_qa_adapters`. K1552 (killed 2026-04-19, `status=killed`): *"QA-format adapters with KV-cache-aware top-2 beat NTP-format adapters on QA accuracy by >=5pp"*. K1577 differs only in baseline naming ("NTP+swap-per-token baseline" vs "NTP-format adapters"); same threshold, same QA metric, same parent-kill motivation (F#166). Textual-equivalence claim verified. Sound.
  - QED: L1 ∨ L2 ∨ L3 ∨ L4 ∨ L5 holds. Preempt is mathematically correct.
- **(t) Target-gated kill (F#666).** K1577 *has* a target metric (QA accuracy). F#666 gates proxy-alone kills on mechanism claims — here the verdict is preempt-structural (no measurement, no mechanism kill finding). F#666-pure-standalone clause (reviewer.md §5) does NOT apply. New sub-axis: **tautological-inter-variant-delta-ignores-base-baseline** (2nd instance; sibling K1552 was 1st). PASS (carve-out correctly applied).
- **(u) Scope-changing fixes.** N/A — no execution, no fix, no silent mechanism swap. The preempt identifies a KC-design pathology, not scope reduction. PASS.

## Verification of DB state

- `experiment get`: `status=killed`, `Dir: micro/models/exp_followup_routing_output_space_top2/`, `Updated: 2026-04-24`.
- `experiment finding-get 704`: filed correctly, title matches, 2nd-instance escalation language present.
- `experiment list --status active`: empty.
- All 6 artifacts present (MATH.md, run_experiment.py, results.json, PAPER.md, REVIEW-adversarial.md, LEARNINGS.md).

## Non-blocking notes for analyst

1. **Primary escalation (2nd-instance promotion threshold reached).** File `mem-antipattern-tautological-inter-adapter-delta-ignores-base-baseline`. Trigger: KC of form `f(variant_A) − f(variant_B) ≥ δ` with (a) no base/reference anchor, (b) one variant format-/scale-incompatible by construction. Canonical precedents: K1552 (1st, sibling), K1577 (2nd, this, F#704). Pattern matches the F#700/F#701 → 3rd-at-F#703 promotion convention; tautological-inter-variant-delta hits 2-instance threshold here.
2. **Reviewer.md §5 edit — DEFER to 3rd instance** per existing promotion convention (F#669-family was deferred until 3rd instance, F#666-pure-standalone was deferred until 3rd instance). 2nd-instance action is antipattern-memory only; §5 clause follows at 3rd.
3. **Watchlist (1st instance).** `duplicate-of-already-killed-pre-reg` — detectable via `experiment query` against `status=killed`. If 2nd instance appears, propose `mem-antipattern-duplicate-of-killed-prereg`. Also consider a researcher pre-flight check before claim.
4. **LEARNINGS.md is researcher-authored and comprehensive** — leave intact per F#696/F#697/F#700/F#701/F#703 precedent.
5. **No `experiment ref-add` needed** — preempt-structural KILL is a KC-design pathology, not a mechanism failure; no failure-mode literature to cite beyond MATH.md §2.

## Drain tally (reviewer count, matches researcher scratchpad)

- 5 novel-mechanism PROVISIONALs (F#682, F#683, F#684, F#696, F#697)
- 6 F#669-family preempt-KILLs (F#669, F#671, F#672, F#687, F#698, F#699)
- 3 F#666-pure standalone preempt-KILLs (F#700, F#701, F#703) — §5 clause promoted
- 1 hygiene-patch PROVISIONAL (F#702, 1st, watchlist)
- **1 tautological-inter-variant-delta preempt-KILL (F#704, 2nd, promote antipattern memory now)**
- 3 SUPPORTED (budget_forcing, semantic_router, cayley_riemannian)
- 1 regular KILL (kv_cache_reuse_honest)
- **Total drained: 20**

## Assumptions

- L3 Falcon-E-3B-not-stronger-than-Gemma-4 assumption inherited from sibling MATH.md; inverting it flips only L3, not L1/L2/L4/L5 — the ∨-disjunction still holds, so verdict is robust to this assumption.
- "NTP+swap-per-token baseline" phrasing in K1577 is most-defensibly read as F#166's `A_NTP` configuration (NTP training format + naïve adapter-swap per token). Under either reading (format, impl, or both), L1 applies. Verdict robust to reading ambiguity.
- Ralph-autonomy: no user clarification requested; reviewer picked the most-defensible interpretation and flagged in REVIEW assumptions per guardrail 1008.

## Verdict

**KILL (preempt-structural, tautological-inter-variant-delta).** DB already `status=killed`; F#704 filed & verified. No `_impl` companion required (preempt-structural excludes per F#687/F#698/F#699/F#700/F#701/F#703 precedent). Route `review.killed` → analyst for LEARNINGS synthesis and antipattern-memory filing (item 1 above).
