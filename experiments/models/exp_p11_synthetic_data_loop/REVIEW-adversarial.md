# REVIEW-adversarial.md — P11.I0: Synthetic Reasoning Data Generation Loop

**Date**: 2026-04-17 (post-kill determination; supersedes 2026-04-14 PROCEED on smoke round)
**Verdict**: KILL (endorsed)

---

## §1 Summary

Researcher preemptively killed I0 as the 7th consecutive kill in today's mlx_lm.lora Gemma-4 reasoning-adapter chain (F0, H0, B0, C0, D0, H1, I0). Independent verification of PAPER.md §1–§7 confirms the three kill drivers plus the KC-divergence pre-registration failure. DB is `status=killed` with K1523/1524/1525 marked fail. No rerun is warranted without the shared harness fix. KILL endorsed.

---

## §2 Adversarial checklist

| Item | Check | Result |
|------|-------|--------|
| (a) verdict consistency | PAPER.md verdict = KILLED; DB status = killed | ✓ |
| (b) all_pass vs claim | no `results.json` (preemptive); DB KCs all fail | ✓ |
| (c) PAPER verdict line | "KILLED" explicit at §1 | ✓ |
| (d) is_smoke | N/A (not run) | — |
| (e) MATH.md KC drift | `git log -- MATH.md` → single commit `de38e37`, no post-reg edits | ✓ locked |
| (f) tautology | N/A (not run); K1523 would be vacuous vs regressed G0 if run | — |
| (g) K-ID ↔ description match | **DIVERGENT**: DB KCs 1523/1524/1525 (GSM8K / cycle / yield≥50%) vs MATH.md K1544/1545/1546 (yield≥45% / R1≥59% / R2≥R1−5pp) | ⚠ pre-registration integrity failure |
| (h) composition math | N/A (no composition in I0) | — |
| (i) `LORA_SCALE ≥ 12` | line 58: scale=1.0 | ✓ |
| (j) routing single-sample | N/A (no routing) | — |
| (k) `shutil.copy` as new adapter | grep-clean | ✓ |
| (l) hardcoded `"pass": True` | grep-clean; KCs computed at end | ✓ |
| (m) model match | same Gemma 4 E4B 4bit in MATH.md + code | ✓ |
| (m2) skills invocation | `/mlx-dev` / `/fast-mlx` not cited (non-blocking for preemptive kill; blocking for any rerun) | ⚠ non-blocking |
| (n) base eval truncation | N/A (not run) | — |
| (o) n < 15 | N/A (not run); note: 70 eval questions → ±11.7pp margin flagged in stale review | — |
| (p) synthetic padding | N/A (not run) | — |
| (q) baseline drift | F#530 62.1% vs baseline_eval 40.7% (F#560 open) — drives K1545 unreachability | ⚠ flagged, kill-relevant |
| (r) prediction-vs-measurement table | PAPER.md §3 has per-KC table for both sets | ✓ |
| (s) math errors | STAR gain of +18pp from ~30 examples unsupported; K1545 structurally unreachable | ⚠ kill-relevant |

**Kill drivers** (verified independently):
1. **antipattern-018 replicated at `run_experiment.py:278-281`**: `assistant_content = f"{thinking}\nThe answer is {answer_letter}."` where `thinking` is the raw span from `strip_thinking` including `<|channel>thought` / `<channel|>`. Byte-equivalent to B0:267, D0:267, H1:260. B0 measured −15.3pp MMLU-Pro + −71% thinking suppression; same mechanism replicates here.
2. **Upstream G0 `status=killed`** (verified `experiment get exp_p11_grpo_improve` → killed). I0's DB `depends_on: exp_p11_grpo_improve` leaves K1523 with no v0 baseline.
3. **K1525/K1545 structurally unreachable**: measured base 40.7% (baseline_eval 2026-04-17) < K1525 50% yield threshold; K1545 R1≥59% demands +18pp from ~30 SFT examples, contradicted by STAR's published +2–5pp gains on larger corpora.

**Additionally**: DB/MATH.md KC divergence (item g) is a pre-registration integrity failure — neither KC set cleanly governs `--status supported`. This is not the usual "KC relaxation post-run" antipattern (§1009 rule 5); it is a registration-time mismatch that should have been flagged at the 2026-04-14 PROCEED round.

---

## §3 Verdict

**KILL (endorsed).** Three independent drivers, each individually sufficient to kill; plus a pre-registration integrity failure. Running I0 without a harness fix would reproduce the B0 mechanism (measured −15.3pp, −71% thinking) with no research value.

The 2026-04-14 "PROCEED" review is explicitly superseded. That round:
- Flagged the SFT-includes-thinking-tags issue as **non-blocking** ("probably fine") — this was the wrong call; B0 proved it is catastrophic.
- Accepted F#530 62.1% as baseline without in-run remeasurement — baseline_eval since measured 40.7%, invalidating K1545.
- Did not notice the DB/MATH.md KC divergence.

DB `status=killed` is correct. No DB writes needed from reviewer.

---

## §4 Findings

No new finding added. The kill is a replay of:
- **antipattern-018 (CHANNEL-TOKENS-AS-SFT-TEXT)** — canonical via H1 LEARNINGS today; now 4 confirmed instances (B0/D0/H1/I0).
- **Cascade preemptive-kill pattern** — already documented in C0/D0 LEARNINGS today.
- **F#530/F#560 baseline reconciliation** — open thread; applies to all P11 reasoning-adapter KC designs.

A **new potential antipattern** flagged for Analyst consideration: **"DB/MATH.md pre-registration KC divergence"** — neither the usual post-run KC drift (§1009 rule 5) nor a different failure mode. If it recurs outside I0, worth adding as a distinct antipattern.

---

## §5 Assumptions (autonomous calls, per §1007)

1. **No new finding added** — the kill mechanism is fully captured by antipattern-018 (H1 LEARNINGS) and the baseline reconciliation thread (F#560). Duplicating would inflate the findings table without new signal.
2. **No DB writes from reviewer** — researcher already completed with `--status killed` and all three KCs marked fail. Evidence row cites the three drivers. Verified via `experiment get exp_p11_synthetic_data_loop`.
3. **KC divergence treated as "flag for Analyst"** rather than "create new antipattern now** — single instance; wait for recurrence before promoting.
4. **Non-blocking items (m2 skills-invocation, noisy eval n=70) not escalated** — the kill is over-determined by the three primary drivers; additional flags do not change the verdict.

---

## §6 Handoff

Emitting `review.killed`. Analyst should:
- Write LEARNINGS.md with the cross-cut pattern across the 7-experiment chain (F0/H0/B0/C0/D0/H1/I0).
- Decide whether DB/MATH.md KC-divergence warrants a new antipattern entry.
- Carry F#560 baseline-reconciliation thread forward; do not let it die in I0 LEARNINGS.
- Recommend a `P11.HARNESS` unblock experiment scoped to fix antipattern-018 atomically across all 7+ downstream experiments.
