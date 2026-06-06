# PAPER — exp_g4_lora_convergence_500

## Verdict: **KILLED (preemptive)**

K1607 "5/5 domains converge within 500 steps, val loss plateau" → FAIL.

## 5-theorem stack — prediction vs measurement

| Theorem | MATH.md prediction | Runner measurement | Fires? |
|---------|--------------------|--------------------|--------|
| T1 adapter inventory | Shortfall ≥ 2 of 5 Gemma-4-E4B LoRA domains | shortfall=2 (available: code, math, medical) | **YES** |
| T2 iter-budget | 5×500 steps ≈ 52.3 min > 30 min hat budget | 52.3 min (linear-scaled from T2.1) | **YES** |
| T3 success_criteria=[] | SUPPORTED undefinable | Runner false-negative: `experiment get` formats NONE as `Success Criteria: NONE — add with: ...`, not `success_criteria: []`. Substring match missed. DB-level blocker IS present per initial claim output (`⚠ INCOMPLETE: missing success_criteria`). | **YES (MATH-level, not runner-level)** |
| T4 KC under-spec "plateau" | 0 of {ε, window, PPL-delta, MMLU, GSM8K, HumanEval} in K1607 | 0 eval keywords in K1607 text | **YES** |
| T5 F#45 non-transfer | BitNet-2B ternary, K2 INCONCLUSIVE, PPL-only | F#45 finding-get: "BitNet" ∧ "INCONCLUSIVE" ∧ "PPL" — all three confirmed | **YES** |

Defense-in-depth: 4/5 runner-level (T1, T2, T4, T5) + 1/5 MATH-level (T3). Any of T1 / T4 / T5 alone is structurally sufficient to block SUPPORTED; combined they block KILLED-with-data as well.

## MATH → runner reconciliation (PLAN.md §Verdict consistency)

- **T3 runner false-negative**: The runner's substring check used `"success_criteria: []"` but the CLI formats the NONE case as `Success Criteria: NONE — add with: ...`. This is a runner-implementation mismatch with the CLI output format, **not** a MATH.md error. The MATH-level claim (sc=[] → SUPPORTED undefinable) is independently verifiable via `experiment get exp_g4_lora_convergence_500 | grep "Success Criteria"`:
  ```
  Success Criteria: NONE — add with: experiment success-add ...
  ⚠ INCOMPLETE: success_criteria, references, kill_results (all untested)
  ```
  Reviewer should verify this directly.
- **MATH vs runner T1**: MATH declared the available-T2.1 set as {code, math, medical}; runner confirmed via filesystem inventory. Agreement.
- **MATH vs runner T2**: MATH estimated ~52.3 min linear-scale; runner computed 52.3 min exactly from T2.1 `results.json` mean × 0.5 × 5. Agreement.

## Verdict-consistency pre-flight (PLAN.md §1)

1. `results.json.verdict = "KILLED"` ✓ (not SUPPORTED)
2. `results.json.all_pass = false` ✓ (consistent with KILLED)
3. PAPER.md verdict = "KILLED (preemptive)" ✓ (no forbidden tokens)
4. `is_smoke = false` ✓
5. No KC edit between MATH.md and now ✓ (K1607 text untouched: "5/5 domains converge within 500 steps, val loss plateau")
6. Antipattern check:
   - `ap-027` pure-stdlib runner: runner uses `pathlib + subprocess + json` only. ✓
   - `mem-antipattern-017` partial-cascade-insufficiency: N/A — this is a scale-safety experiment, not a composition-bug cohort member; the 5-theorem stack is experiment-specific, not partial-cascade.
   - `ap-scale-misclassified`: INVOKED — F#45 BitNet-2B proxy cannot substitute for Gemma 4 E4B convergence dynamics. Preempt argument, not violation.
   - `ap-framework-incomplete`: INVOKED — sc=[] is the canonical blocker class. Preempt argument, not violation.

All six pass.

## What would unblock

Only the operator can unblock. Required actions (in order):
1. `experiment success-add exp_g4_lora_convergence_500 --condition "5/5 domains: final_val_loss within 5% of min(val_loss) over last 50 steps" --unlocks "micro convergence baseline on Gemma 4 E4B"` (or equivalent sc addition that pins the plateau ε).
2. Pin the 5-domain set: commit training data for 2 additional Gemma 4-compatible domains (candidates: creative, legal via `bitnet_lori_sparse_b/data/{creative,legal}/train.jsonl` — but these are BitNet-era; would need Gemma 4 template re-application and validation).
3. Relax plateau clause with explicit ε-window or re-scope KC to "val loss monotone-decreasing over last 100 steps" which is operationally unambiguous.
4. With (1)+(2)+(3), re-open at priority 2 and allow a researcher claim with full 2h micro budget (launch via pueue, hand off to `experiment.done` with status tied to post-run readout).

## Reusable preempt (registered for future claims)

**F#45 non-transfer one-liner** (appends to ap-scale-misclassified source list):
> "F#45 (BitNet-2B ternary QAT+STE convergence) does not transfer to Gemma 4 E4B 4-bit LoRA r=6 convergence dynamics: architectural mismatch (native ternary vs dense-quantized), metric mismatch (PPL-only vs task-quality; repo r≈0.08), F#45 self-caveat K2 INCONCLUSIVE on step-budget confound. Preempt any future 'Gemma-4 ternary-informed convergence' claim until re-measured on Gemma 4 E4B."

## Assumptions
1. T2.1 mean per-1000-step timing (1255.17s) is linear-scalable to 500 steps with ≤10% drift (standard for LoRA training under same hyperparameters).
2. 30-min hat iter-budget and 2h micro-scale ceiling from PLAN.md §Iteration discipline.
3. Operator re-scope is the canonical unblock path per analyst iter-15/16 cohort-drain consolidated handoff.
4. Runner's T3 false-negative is a mechanical substring-match mismatch, not a MATH.md error; DB-level `Success Criteria: NONE` is visible in `experiment get` output and was flagged at claim-time.

## References
- F#45 (supported, 2026-03-28): `experiment finding-get 45`
- F#416 (killed, Gemma 4 E4B HRA vs LoRA): prior Gemma 4 convergence-window precedent
- T2.1 (exp_p1_t2_single_domain_training, supported, v3-rerun 2026-04-19): wall-clock baseline
- `mem-antipattern-017` partial-cascade-insufficiency: analyst iter-16 scope addendum
- `ap-scale-misclassified`: proxy model substituted for target
- `ap-framework-incomplete`: sc=[] structural blocker

## Routing signal for reviewer
No new antipattern (F#45 non-transfer is a one-line preempt registered under ap-scale-misclassified, not a novel pattern). If 17-item adversarial checklist passes: emit `review.killed` → analyst. Analyst should append this experiment to ap-017 source list under the scale-safety branch (distinct from composition-bug branch) and register F#45 non-transfer as a reusable preempt.
