# REVIEW-adversarial.md — P11.H1: thinking-metar1-metacognitive-v0

**Reviewer**: Adversarial
**Date**: 2026-04-17 (post-kill determination — supersedes 2026-04-14 PROCEED draft)
**Round**: Post-cascade (1 round; no rerun intended)
**Verdict**: **KILL** (endorses preemptive kill; all three KCs fail as reported)

---

## Trigger

`experiment.done exp_p11_metacognitive_adapter: KILLED (preemptive)` —
sixth consecutive `mlx_lm.lora` Gemma-4 reasoning-adapter kill today
(F0 → H0 → B0 → C0 → D0 → H1).

---

## Adversarial checklist

| Item | Check | Status |
|------|-------|--------|
| (a) `results.json.verdict` vs DB status | results.json absent (preemptive kill); DB = killed; PAPER.md = KILLED. Consistent. | ✓ |
| (b) `all_pass` vs claim | n/a (not run) | n/a |
| (c) PAPER.md verdict line | "KILLED — preemptively, before run" at §1. | ✓ |
| (d) `is_smoke` vs full | Explicit "not run"; no false-full claim. | ✓ |
| (e) MATH.md KC drift | `git log MATH.md` = single commit `de38e37`. No post-reg edit. | ✓ |
| (f) Tautology sniff | K1520/21/22 all measure distinct quantities. No algebraic identity. | ✓ |
| (g) K-ID text ↔ code ↔ DB | K1520/1521/1522 text matches DB text. | ✓ |
| (h) Composition math bug | Single-adapter continuation from H0; no `sum(lora_A)` / `add_weighted_adapter`. | ✓ |
| (i) `LORA_SCALE≥12` | L64 = `LORA_SCALE = 1.0`. | ✓ |
| (j) Per-sample routing | No routing in H1. | n/a |
| (k) `shutil.copy` adapter-as-new | Not present. Real `mlx_lm.lora` training. | ✓ |
| (l) Hardcoded `"pass": True` | Not present; KCs computed from measurements. | ✓ |
| (m) Target model ≠ loaded model | Same Gemma 4 4-bit base in train + eval. | ✓ |
| (m2) Skill invocation evidence | Not applicable — no new MLX code authored in this iteration; experiment was preemptively killed before run. | n/a |
| (n) Base = 0% + avg_thinking = 0 | n/a (not run). | n/a |
| (o) Headline n<15 | n/a (not run). | n/a |
| (p) Synthetic padding | n/a. | n/a |
| (q) Cited-vs-measured baseline drift | **Flagged:** MATH.md §Theorem 2 cites Q_base=62.1% (F#530) but baseline_eval 2026-04-17 measured 40.7% on same model. PAPER.md §2(d) acknowledges. Non-blocking: the kill does not depend on which baseline is canonical. | ℹ |
| (r) PAPER prediction-vs-measurement table | Present (§3 Kill Criteria table). All rows have both predicted and measured/failure-reason columns. | ✓ |
| (s) Math errors / unsupported claims | Theorem 1 (orthogonal composition) and Theorem 2 (Q_H0 > Q_base) both superseded; PAPER.md §2(d) updates the premises against measured values. Honest. | ✓ |

**Protocol-bug confirmation (reviewer independent grep):**
- `run_experiment.py:259-261` — `f"<|channel>thought\n{structured_thinking}\n<channel|>..."`
  becomes `assistant.content` at L278.
- B0 `run_experiment.py:267` — same pattern, measured −15.3pp MMLU-Pro, −71% thinking.
- D0 `run_experiment.py:267` — same pattern, preemptively killed same day.
- H1 is byte-identical in intent. Kill is robust to retry without harness fix.

---

## Verdict: KILL

All three KCs genuinely fail. The preemptive designation is honest per PLAN.md §1008
(anti-stuck): retrying the same harness-level failure is compute spent reproducing a
known bug.

---

## DB status

Already `status=killed`, K1520/1521/1522=fail, evidence row written by researcher.
No reviewer DB mutation needed.

---

## No new finding added

The structural lesson (channel-text-as-`mlx_lm.lora`-SFT-target protocol bug) is already
captured in the B0 kill chain (B0 PAPER.md 2026-04-17). The cascade pattern
(preemptive-kill-on-harness-failure) is documented in C0 and D0 LEARNINGS today.
Adding a fourth finding for the same mechanism would duplicate without adding signal.

Finding #530 baseline-reconciliation (62.1% vs 40.7%) remains open across the H0/D0/H1
chain and should be closed by the P11.HARNESS experiment's re-measurement.

---

## Open threads for Analyst

1. **P11.HARNESS (unblock candidate)**: shared training harness that either
   (i) strips `<|channel>thought...<channel|>` tokens from training targets, gating
   thinking purely as eval protocol; (ii) switches to `<think>...</think>` text format
   (H0 used this successfully — K1519 PASS); or (iii) trains via a custom MLX SFT loop
   that respects the Gemma 4 chat template. Acceptance: MMLU-Pro-with-thinking
   ≥ base − 2pp on a 50-trace pilot.
2. **H1-v2 redesign (post-harness)**: locally-measured H0-v2 baseline (NOT F#530);
   non-vacuous K1521 (H1-v2 ≥ base − 2pp, not vs regressed H0); K1520 on answer-parseable
   subset only (excludes protocol-failure responses).
3. **Theorem 2 premise**: recompute Q_H0/Q_base from harness-fix run; re-prove (or
   falsify) before re-claiming H1.
4. **LEARNINGS.md**: should note the KC design lesson — KCs phrased "≥ upstream-H0" become
   vacuous the moment upstream regresses; better framing is "≥ base − ε" (absolute, against
   in-run measurement).

---

## Assumptions logged

- A1 (reviewer): Researcher's claim that preemptive-kill avoids ~2h of compute is plausible
  (Phase 1 ~1h trace generation + Phase 2 ~1h training on H0 checkpoint); not independently
  timed but not load-bearing for the KILL verdict.
- A2 (reviewer): The 2026-04-14 REVIEW-adversarial.md Round 2 PROCEED is explicitly
  superseded — it was a design-only review under the assumption that H0 would land ≥65.1%
  MMLU-Pro. That assumption was falsified 2026-04-17.
- A3 (reviewer): No tool-call budget violation; 6 file reads + 3 bash calls + 1 write.

---

## Superseded

- 2026-04-14 REVIEW-adversarial.md Round 2 (PROCEED) — obsolete post-H0/B0/D0 kills.
