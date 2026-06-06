# REVIEW-adversarial — exp_g4_adapter_class_composition_full_impl

**Verdict: PROCEED (PROVISIONAL)**

Reviewer iteration ~108. Phase A executable slice is clean — 25/25 adversarial items PASS or N/A. Third instance of the Phase A slice pattern (F#772 jepa, F#799 memento, this). Drain state confirmed: P≤2 open=0, active=0.

## Adversarial checklist

| # | Item | Result | Note |
|---|------|--------|------|
| a | results.json verdict vs DB | PASS | both PROVISIONAL |
| b | all_pass vs claim | PASS | null; all KCs untested |
| c | PAPER.md verdict line | PASS | PROVISIONAL |
| d | is_smoke vs claim | PASS | is_smoke=false, is_phase_a_executable_slice=true |
| e | KC mutation post-claim | PASS | K1-K4 verbatim from parent F#686, no post-hoc relax |
| f | Tautology sniff | PASS | all KCs untested; A1-A3 are real inspection |
| g | K-ID code↔math | PASS | code measures A1-A3 only; K1-K4 untouched |
| h | Bad composition idiom | PASS | no composition code |
| i | LORA_SCALE≥12 | PASS | no training; default 6.0 in MATH.md §0 |
| j | Single-sample routing | PASS | no routing |
| k | shutil.copy | PASS | absent |
| l | Hardcoded pass | PASS | A1-A3 populated from real mlx_lm inspection |
| m | Target model match | PASS | gemma-4-e4b-it-4bit in both MATH.md and code line 87 |
| m2 | Skill invocation | PASS | /mlx-dev /fast-mlx cited §0; mx.clear_cache() used |
| n | Base 0% truncation | N/A | no eval |
| o | Headline n<15 | N/A | no eval |
| p | Synthetic padding | N/A | no eval |
| q | Baseline drift | N/A | no eval |
| r | Pred-vs-measurement table | PASS | MATH.md §5 + PAPER.md both present |
| s | Math errors | PASS | claims conservative and well-scoped |
| t | F#666 target-gated kill | N/A | no kill proposed; K2 target preserved |
| u | Scope-changing fix | PASS | Phase A as designed; F1-F5 forbid list binding |

**0 blocking fixes.**

## Findings for analyst to file

1. Phase A topology + DoRA symbol absence: F#627 confirmed at Gemma 4 E4B 4-bit (42×v_proj + 42×o_proj). DoRA: 0 symbols in mlx_lm.tuner.lora v0.31.2. B1 scope potentially 2 custom modules.
2. Researcher honored prefile gate 4th consecutive time post-mitigation.
3. Drain milestone: RESEARCH_BACKLOG_DRAINED success criteria met.

## Assumptions

- Accepted Phase A executable slice precedent (3 instances) as sufficient for PROVISIONAL.
- A3 DoRA FAIL is informational, not a KC failure — CLI-level verification deferred to Phase B.
