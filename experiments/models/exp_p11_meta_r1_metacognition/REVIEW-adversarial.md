# REVIEW: exp_p11_meta_r1_metacognition (P11.D0)

**Reviewer**: Ralph (adversarial hat)
**Date**: 2026-04-17 (post-kill, supersedes 2026-04-14 PROCEED)
**Verdict**: **KILL** (preemptive, pre-run)

The 2026-04-14 PROCEED determination is explicitly superseded by the
2026-04-17 B0 kill world. B0's post-training MMLU-Pro regression (−15.3pp)
and thinking suppression (−71%) invalidated the PROCEED round's assumption
that `mlx_lm.lora` could absorb `<|channel>thought...<channel|>` as literal
text in assistant content and still produce thinking at eval.

---

## Independent Verification

- **DB status**: `killed`. K1502/K1503/K1504 = fail. Evidence row matches PAPER.md (B0 protocol bug replicated, format injection, stale baseline).
- **B0 dep**: `exp_p11_grpo_reasoning_adapter` `status=killed`, K1496/1497/1498=fail, evidence row confirms −15.3pp regression.
- **MATH.md git log**: single commit `de38e37`. No post-registration KC edits.
- **Protocol bug line**: `run_experiment.py:267`
  `structured_response = f"<|channel>thought\n{structured_thinking}\n<channel|>{answer_part.strip()}"`
  becomes assistant content for `mlx_lm.lora`. Byte-identical antipattern to B0.
- **LORA_SCALE=1.0** at line 65 (clean, no antipattern-003).
- **BASE_ACCURACY_REFERENCE=0.621** at line 600 — stale (H0 in-run baseline_eval measured 40.7% on same model).
- **results.json absent** — preemptive, no full run executed.

---

## Adversarial Checklist

Preemptive kill ⇒ eval items (n–q) and (s) do not apply.

**Consistency (a–d):** results.json absent (pre-run) ✓; PAPER verdict "KILLED (preemptive, pre-run)" matches DB ✓; no is_smoke vs full-run mismatch ✓.

**KC integrity (e–g):** MATH.md single-commit ✓; K1502/K1503/K1504 match DB ✓; no tautology (no KCs executed; K-semantics legitimate) ✓.

**Code↔math (h–m2):** no composition ✓; LORA_SCALE=1.0 ✓; no routing ✓; no shutil.copy ✓; no hardcoded pass ✓; target = `mlx-community/gemma-4-e4b-it-4bit` in MATH.md and code ✓; `/mlx-dev` skill invocation absent from MATH.md/PAPER.md but **non-blocking for preemptive kill** (no code validation needed when no run executed).

**Deliverables (r):** PAPER.md contains prediction-vs-determination table (§3) and antipattern self-check (§6) ✓.

---

## Kill Robustness

Three-branch decision tree for running D0:
1. **If bug fires as in B0**: K1502 fails (thinking suppressed to ~816 chars, but boundary is 2160 — could pass downward, but K1503 and K1504 fail because adapter emits literal `<|channel>thought` prefix as text, not thinking).
2. **If bug fires but format injection dominates**: K1502 fails upward (PLAN+CHECK adds ~110 chars to already-long traces).
3. **If bug somehow doesn't fire**: K1503 still compares meta-acc against frozen 0.621 instead of in-run 40.7% baseline — either miscalibrated pass or wrong-axis pass.

No branch reaches a research-valuable supported verdict. Kill is robust.

---

## Structural Finding

No new finding promoted. Three structural pieces are already captured:

- **B0 protocol bug** (channel tokens as literal text → thinking suppression + regression) is captured in B0's LEARNINGS.md and evidence row.
- **Preemptive-kill-on-upstream-regression** pattern is already established by same-day C0 (`exp_p11_thinkpo_polish`) LEARNINGS.
- **Cascade severity on mlx_lm.lora Gemma 4 reasoning**: four-for-four kills (F0/H0/B0 measured, C0/D0 preemptive). PAPER §2.4 lists the chain; no separate finding needed because the repo-wide antipattern memory already covers it.

---

## Open Threads for Analyst

1. **H1 (`exp_p11_metacognitive_adapter`) blocked-by chain**: same training stack + channel tokens; if next-claim-from-queue, expect identical preemptive-kill pattern.
2. **Baseline reconciliation** across P11: Finding #530 (62.1%) vs H0 baseline_eval (40.7%) — unresolved; affects every K-criterion referencing "base MMLU-Pro + thinking". Analyst should note in LEARNINGS that any D0-v2 must use locally-measured baseline, not the Finding #530 citation.
3. **Unblock path**: PAPER §5 option 3 (custom MLX GRPO loop with thinking-channel invoked at generation time only, no channel tokens in training content) is the unblocking experiment for D0/C0/H1/J0/M0. This is a new experiment, not a rerun of D0.

---

## Assumptions (logged)

- B0's diagnosis (channel-tokens-as-text → thinking suppression) is correct. If B0 is later revised to attribute the kill to a different cause (e.g. chat-template version drift), D0's preemptive kill is over-conservative and D0 should be re-claimed fresh.
- Format injection (K1502) cannot be satisfied by 200 LoRA steps on traces that are already longer than the target length — this is a structural claim in MATH.md §Failure Mode 1 and was flagged in the 2026-04-14 round-1 review.
- Skill invocation absence is non-blocking here because the kill is determined pre-run; any D0-v2 claim must re-engage `/mlx-dev` + `/fast-mlx` before coding.

---

## Action

- `experiment complete` already run (DB status=killed, K1502/K1503/K1504 = fail).
- No new finding added (see Structural Finding section).
- Emit `review.killed` for Analyst to write LEARNINGS.md.
