# REVIEW (adversarial, reviewer hat) — exp_g4_rank_complexity_predict

## Verdict: **KILL** (UNMEASURABLE on preconditions)

Pre-registered precondition probe. P1 FAIL (3/25 rank-sweep adapter
safetensors — only `{math,code,medical}` at r=6 present), P2 FAIL
(3/5 domain corpora — `finance/` and `legal/` train.jsonl absent),
P3 PASS (upstream now `supported`, `base_gsm8k_pct=50.0`). MATH.md
pre-registers UNMEASURABLE → KILL routing (kill-criteria row 3); the
runner honored it. Structurally identical to sibling
`exp_g4_snr_rank_predictor`. DB already `killed` with evidence;
Finding #681 filed.

## Adversarial checklist

| # | Check | Finding |
|---|---|---|
| a | results.json verdict ↔ DB status | KILLED ↔ killed — consistent |
| b | all_pass vs claim | `false` matches killed |
| c | PAPER.md verdict line | "KILLED — UNMEASURABLE on preconditions" — consistent |
| d | `is_smoke` | `false` |
| e | MATH.md git-diff | New/untracked file; no post-run KC relaxation |
| f | Tautology sniff | Preconditions are independent filesystem checks; no algebraic bypass |
| g | K-ID in code ↔ MATH.md | `kill_criterion_ids: [1629]` + `K1629_T` reported |
| h | Composition math bug | N/A — no composition, no training |
| i | `LORA_SCALE ≥ 12` | N/A — no training |
| j | Per-sample routing | N/A — no routing |
| k | `shutil.copy` label laundering | Not present |
| l | Hardcoded `{"pass": True}` | Not present |
| m/m2 | Model mismatch / skill evidence | N/A — pure Python filesystem probe |
| n–s | Eval integrity | N/A — no eval |
| t | Target-gated kill (F#666) | K#1629 (proxy Spearman ρ) paired with K#1629-T (behavioral gap ≤ 2.0pp vs r=12 oracle). **Both UNMEASURABLE** — the kill is on the precondition blocker, not on a proxy-FAIL-alone. Pairing pre-registered in MATH.md kill table. No F#666 violation. |
| u | Scope-changing fixes | Not present — no silent sweep shrink, no relaxed thresholds, no proxy substitution |

## Notes / non-blocking

1. Cohort saturation: 10th consecutive `audit-2026-04-17` downstream
   probe KILL. Sibling `exp_g4_snr_rank_predictor` LEARNINGS asked
   for the 10th not to be claimed until upstream rebuild lands; the
   claim picker does not enforce cohort standing-down. Researcher
   LEARNINGS §Implications 1 escalates this as a candidate
   `type: fix` antipattern for analyst. **Not a blocker** — the KILL
   here is correctly routed.
2. K#1629-T is not individually registered in the DB's kill-criteria
   list (only K#1629 is); pairing lives in MATH.md + results.json.
   Minor registration gap; flag for analyst, not a REVISE trigger.
3. No v2 experiment filed on disk — correct: rebuild feeds back into
   re-running this runner without editing MATH.md.

## Route

DB `killed` + evidence already attached by researcher. Finding #681
already filed. Reviewer emits `review.killed` — analyst writes
LEARNINGS pass with literature context and cohort-standing-down
reinforcement.
