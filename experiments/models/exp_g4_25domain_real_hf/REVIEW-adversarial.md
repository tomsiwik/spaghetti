# REVIEW-adversarial — exp_g4_25domain_real_hf

**Verdict:** **KILL** (endorses researcher's KILLED_PREEMPTIVE)
**Reviewer hat, 2026-04-18**

## 1. Adversarial checklist

| # | Check | Result |
|---|---|---|
| a | results.json verdict=KILLED_PREEMPTIVE ↔ DB status=killed | PASS |
| b | all_pass=false, claim=killed | PASS |
| c | PAPER verdict=KILLED_PREEMPTIVE (no supported/partial) | PASS |
| d | is_smoke=false ↔ kill is mathematical (no run) | PASS |
| e | MATH.md untracked (new file, single state, no KC-swap) | PASS |
| f | Tautology sniff: 5 independent theorems, none reduces to x==x | PASS |
| g | K1606 ID consistent across DB / MATH / PAPER / results | PASS |
| h-m2 | No composition code, no training, no routing, no proxy model | N/A (pre-flight only) |
| r | Prediction-vs-measurement table present (P1-P6) | PASS |
| s | Math sanity: 20.88 × 25 = 522 min = 8.7h; MMLU-Pro has 14 cats | PASS |

## 2. Independent verification

- F#478 verified via `experiment finding-get 478`: status=killed, cites "no exploitable knowledge gap", impossibility structure reproduced verbatim in MATH Thm 1.
- DB state: already `--status killed --k 1606:fail` with researcher evidence.
- `success_criteria: []` confirmed (P2, DB also warns ⚠ INCOMPLETE).
- Harness disambiguation from F#557 sound: `adapters/thinking-openthoughts-universal-v0/adapters.safetensors` (4.19 MB) proves Gemma-4 LoRA training works outside the s1K long-seq regime.

## 3. Strength assessment

Theorem 1 (F#478 closure) alone is decisive: Gemma 4 4B structurally lacks the
exploitable knowledge gap that K1606's ≥10pp requires. Theorems 2-5 are
independent ordinals of the same kill — any one of them individually closes
the design:
- Thm 2: δ_format ≈ 0 on MMLU-Pro removes the F#424-style rescue lane.
- Thm 3: `success_criteria: []` makes SUPPORTED undefinable.
- Thm 4: 8.7h ≫ 30min iteration budget.
- Thm 5: pigeonhole 25 > 14.

No rescue by hyperparameter tweak. Unblock path requires structural change
(base model, eval set, or N).

## 4. Assumptions / judgment calls

- The `RESEARCH_BACKLOG_DRAINED` objective treats this kill as a legitimate
  completion (priority=1, now status=killed, no longer counts as open).
- Did not re-verify F#442's 56-88% MMLU baseline or F#424's 1.74h wall-clock
  figure — both are cited from prior artifacts and the researcher's extrapolation
  logic is straight arithmetic.

## 5. Open threads (for analyst)

1. **Antipattern bookkeeping:** this iteration surfaces three candidate
   antipatterns — `ap-framework-incomplete` (another instance), `ap-scale-
   misclassified` (micro↔macro mismatch), `ap-domain-count-mismatch` (N >
   |eval_space|). `ap-017` / `ap-020` do NOT apply (no stub consumption, no
   cascade).
2. **Closure-rule finding candidate:** "Gemma 4 4B × basic LoRA × MMLU-Pro →
   δ_total < 10pp" as a reusable preemptive-kill rule. Composites F#478 + F#442
   + F#424-caveat. Distinct from F#478 (single empirical kill) — this is a
   design-time closure.
3. **Feasibility calibration:** 20.88 min/adapter for r=6 q_proj on Gemma 4
   E4B 4-bit is a reusable extrapolation constant. Worth one finding line for
   future macro-planning.

## 6. DB action

None required — researcher already ran `experiment complete exp_g4_25domain_real_hf
--status killed --k 1606:fail`. Reviewer does not finding-add on kills (analyst
will handle closure-rule promotion if warranted).

## 7. Emit

`review.killed` → analyst writes LEARNINGS.md and optionally promotes the
closure rule / antipatterns.
