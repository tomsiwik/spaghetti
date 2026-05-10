# Adversarial Review — TIES-Merging (exp_pierre_ties_full_delta)

## Verdict: KILL (confirmed)

## Checklist

| # | Check | Result |
|---|-------|--------|
| a | results.json verdict matches DB | PASS — both KILLED |
| b | KC pass/fail consistent | PASS — K1 FAIL, K2 FAIL, K3 PASS |
| c | PAPER.md verdict matches | PASS — "KILLED" |
| e | KC not modified post-run | PASS |
| f | No tautological KC | PASS — target-metric based |
| g | Code measures what MATH.md describes | PASS |
| h | No independent A/B summation | PASS — materializes A_t @ B_t per adapter |
| i | LORA_SCALE < 12 | PASS — scale=6.0 |
| k | No shutil.copy of sibling | PASS |
| l | No hardcoded pass | PASS |
| m | Model matches across artifacts | PASS — gemma-4-e4b-it-4bit |
| o | n >= 15 | PASS — n=50 |
| p | Target-metric KC present | PASS |

## Challenges

### 1. Could keep_frac tuning save TIES?

keep_frac=0.3 (paper default). Higher values reduce trim aggression, but
at keep_frac=1.0 TIES degenerates to sign-elected average ~ Fisher-Rao.
Tuning might close the 1pp K1 gap but won't fix the 4.7pp K2 gap.

### 2. Is N=50 sufficient?

At N=50, +2pp vs +3pp is within noise. But K2 (4.7pp gap to DARE) is the
binding kill, and DARE reference is stable across all three experiments.

### 3. Shared-A magnitude correlation claim

Mechanistic reasoning, not measured. Plausible but unverified. Kill stands
on KC numbers alone regardless of mechanism.

## Final Assessment

Kill is justified. Third structured merge killed for shared-A (after ACE,
OrthoMerge). LEARNINGS.md correctly closes the class.
