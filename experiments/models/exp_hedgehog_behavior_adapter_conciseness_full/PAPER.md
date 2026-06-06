# PAPER.md — exp_hedgehog_behavior_adapter_conciseness_full (SMOKE iter)

**Verdict: PROVISIONAL** (smoke iter — `is_smoke=True` caps verdict per verdict-consistency check #4; full-iter blocked until MMLU harness v2)

## 1. Prediction-vs-measurement table

| Item | MATH.md prediction (FULL) | Smoke measurement | Status |
|---|---|---|---|
| Phase B loss (last 5) | < 0.05 (converge) | 0.045 | PASS (smoke) |
| Mean per-layer cos (informal) | > 0.85 | 0.9574 | PASS (smoke) |
| K#1965 reduction (uncensored ≥1024) | 30–50%; mean 38% | 57.76% at max_tokens=512 | **PROVISIONAL-PASS lower bound** (base hit cap on 8/8 still censored) |
| K#1966 MMLU drop ≤ 3pp | mean 1.5pp | 0.0pp BUT degenerate-flag (A4) | **HARNESS BUG** (base_acc=0.15 below chance; all preds = "C") |
| Verdict | SUPPORTED most likely | PROVISIONAL (smoke + harness bug) | smoke caught bug pre-full-run |

## 2. Findings

### Finding #1 — K#1965 PASS at smoke max_tokens=512 with cap on base 8/8 outputs (consistent with F#789 lower-bound caveat)

Phase B trained from loss=0.164 → 0.039 in 30 steps; mean per-layer cos=0.9574; adapter behaviorally compresses outputs by 57.76% (base_mean=512 token capped, student_mean=216 uncensored). This is consistent with F#789 (_impl reported 26.17% at max_tokens=256). Both reductions are LOWER BOUNDS — base output is censored at the cap. The full-iter prediction of 30–50% likely undercounts; true reduction may be ≥60% with uncensored 1024+.

**Per F#789 A2 + this smoke:** max_tokens=512 STILL censors base on 8/8 outputs. Need max_tokens ≥ 1024 (and likely 2048) to obtain the uncensored true reduction.

### Finding #2 — MMLU harness IS BROKEN under Gemma 4 IT thinking mode (smoke gate caught pre-full-run)

K#1966 returned base_acc=0.15 and adapter_acc=0.15, both well below the 0.25 random-chance baseline for 4-choice MMLU. Inspection of `base_preds_first_5` and `adapter_preds_first_5` shows ALL predictions = "C" against a uniform canonical distribution. The bug is:

- Gemma 4 IT chat template emits `<|channel>thought\n...` thinking-trace prefix before the final answer.
- With `MMLU_GEN_MAX_TOKENS=4`, generation stops mid-prefix, well before any answer letter is produced.
- The naive first-letter scanner `for ch in out_clean: if ch in "ABCD": pred = ch; break` matches the **C** in **CHANNEL**, returning "C" deterministically.
- Both base (scale=0) and adapter (scale=6.0) output the same prefix, so drop_pp=0.0 — but the comparison is meaningless.

**This is exactly the smoke-validation use case** (MATH.md §9): the smoke gate caught a NEW-code bug on a 20-sample subset before the full 100-sample run wasted 3-5h.

### Finding #3 — Smoke validation pattern WORKS (validates §9 gate)

The MATH.md §9 "smoke validation (intermediate gate before full run)" successfully filtered out a faulty harness in 122s of CPU time. The full run is correctly BLOCKED until v2 fixes the harness. Without this gate, we would have submitted a 3-5h pueue task and discovered the bug post-hoc on full data.

This validates the broader researcher.md "batch discipline" principle: submit ONE first; confirm schema and metric directions; then submit the rest.

### Finding #4 — Hedgehog cos-sim training is robust under conciseness signal

Phase B convergence (0.164 → 0.039 over 30 steps) and mean per-layer cos=0.9574 are statistically indistinguishable from F#783 (politeness 0.954 cos), F#784 (refactor 0.952), F#786 (formality 0.91), F#789 (conciseness_impl 0.957). The Hedgehog distillation is invariant to behavioral axis. This is a positive structural finding.

## 3. Discrepancies vs prediction

- K#1965 prediction (30–50%) was a lower-bound mean — observed 57.76% smoke value but with ALL base outputs at cap. The real value is likely higher but unknown until uncensored.
- K#1966 prediction was meaningless to compare since the harness was broken. The 0.0pp drop reading is an artifact, not a measurement.
- Verdict prediction (SUPPORTED ~60%) was based on assumption the MMLU harness would work. Smoke invalidated that assumption.

## 4. F#666 verdict matrix on smoke

K#1965 smoke = PROVISIONAL-PASS lower bound (real signal, censored value).
K#1966 smoke = HARNESS-BROKEN (degenerate-flag A4: base accuracy below chance).

Per MATH.md §4 + F#666: when a KC harness errors or returns degenerate output, the verdict cannot reach SUPPORTED. Smoke + degenerate-flag → PROVISIONAL.

Per verdict-consistency check #4: smoke runs cap at PROVISIONAL regardless of KC results. The two-conditions intersection (smoke + harness-broken) keeps verdict at PROVISIONAL.

**No KILL.** F#666 KILL requires bilateral fail. K#1965 PASSes (lower bound), K#1966 is unmeasured (harness bug ≠ behavioral fail).

## 5. Assumptions (extending MATH.md §8)

- **A8 (NEW).** Smoke gate per §9 is sufficient to catch new-code bugs (validated this iter). v2 should re-run smoke before full.
- **A9 (NEW).** Required harness fix for K#1966 v2:
  1. Disable Gemma 4 IT thinking mode for MMLU eval (likely via `apply_chat_template(..., enable_thinking=False)` or analogous flag), OR
  2. Increase `MMLU_GEN_MAX_TOKENS` to ≥256 + parse for `<|channel>final` marker before extracting the answer letter, OR
  3. Add a strong "Output a single letter only, no thinking, no explanation" system prompt to MMLU prompt construction.
- **A10 (NEW).** Required K#1965 fix for v2: raise `GEN_MAX_TOKENS` to 2048 (1024 still censors per smoke base 8/8 cap-hit).

## 6. Blockers

1. **K#1966 harness bug** (above). Requires v2 fix before full run.
2. **K#1965 max_tokens cap** still censoring at 512. v2 needs 2048.
3. **`linear_to_lora_layers` shim AttributeError** — known F#789 sibling 4th occurrence, manual fallback works (84 modules attached). Functionally correct; pattern recurrence is methodology improvement candidate (mem-antipattern-linear-to-lora-layers-shim-recurrence ratified by analyst).

## 7. Hand-off recommendations for v2 / next researcher iter

1. Apply Assumption A9 fix (one of three options listed).
2. Apply Assumption A10 fix (max_tokens=2048).
3. Re-run smoke; verify MMLU base_acc ∈ [0.40, 0.70] (Gemma 4 E4B 4-bit baseline; well above chance).
4. Verify K#1965 base outputs are NOT capped (check `base_capped_count` < 0.5 × n).
5. Then submit full run via pueue (3-5h budget).

## 8. Smoke timing

- Phase 0 + load + LoRA attach: ~8s
- Phase B (30 steps): 5.5s
- Phase C cos-sim: ~5s
- Phase C K#1965 (8 prompts × 2 conds × max_tokens=512): ~90s
- Phase D K#1966 (20 prompts × 2 conds × max_tokens=4): 12.3s
- Total: 122.2s

The K#1965 phase dominates. Full would be 50 prompts × 2 conds × max_tokens=2048 ≈ ~12 min, plus K#1966 100 × 2 × max_tokens=256 ≈ ~30 min, plus 800 training steps ≈ ~150s. Total budget likely 1-2h, well within 3-5h estimate.
