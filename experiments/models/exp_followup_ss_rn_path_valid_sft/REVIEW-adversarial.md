# REVIEW-adversarial.md — exp_followup_ss_rn_path_valid_sft

**Verdict: KILL (confirm)** | Reviewer, 2026-04-19

## 1-line summary
Pre-registered precondition-probe KILL. 3/3 probes FAIL (adapters.safetensors missing,
no persona corpus, upstream T2.1 KILLED). K1572 routes FAIL-unmeasurable per MATH.md
§Preconditions. Verdict consistent across results.json / PAPER.md / DB.

## Adversarial checklist

| Item | Check | Result |
|---|---|---|
| (a) verdict vs DB | `results.json.verdict="KILLED"` ↔ DB `status=killed` | ✓ consistent |
| (b) all_pass vs claim | `all_pass=false`, no "supported" proposed | ✓ |
| (c) PAPER.md verdict line | Line 3 `Status: KILLED`; no hidden PROVISIONAL/SUPPORTED | ✓ |
| (d) is_smoke | `is_smoke=false` honestly reported | ✓ |
| (e) KC diff in git | Dir is untracked (new) — K1572 cannot have been edited post-hoc | ✓ |
| (f) tautology sniff | K1572 measures ‖acc_final − 74.4‖ vs ≤5pp; not identity, not tautological | ✓ |
| (g) K-ID semantic match | K1572 in MATH.md ↔ results.json ↔ run_experiment.py all measure the same object (or route FAIL-unmeasurable) | ✓ |
| (h) composition math | No composition executed (probe-only); no `sum(lora_A` / `add_weighted_adapter(linear)` patterns | ✓ |
| (i) LORA_SCALE hardcoding | No training; no LORA_SCALE≥12 in code | ✓ |
| (j) per-sample routing | No routing executed (probe-only) | ✓ |
| (k) shutil.copy as adapter | None | ✓ |
| (l) hardcoded `{"pass": True}` | `status:"FAIL"` honest across probes and K1572 | ✓ |
| (m) proxy-model substitution | Gemma 4 referenced but never loaded — probe runs in pure Python, no model ops | ✓ |
| (m2) skill-invocation | No MLX code executed, so `/mlx-dev` not required for this probe | N/A |
| (n)-(q) eval integrity | `ran=false`; no eval executed, nothing to flag | N/A |
| (r) PAPER prediction table | Present (PAPER.md §"Prediction vs Measurement", 4 rows incl. K1572) | ✓ |
| (s) math/claim integrity | Theorem 1 (A+B+C) chain unchanged; unmeasurable ≠ supported is the correct routing | ✓ |

## Assumptions (judgment calls, per reviewer autonomy)
- Sibling Finding #600 (`exp_followup_sft_behavioral_lora_scale_5`) established the
  precondition-probe routing template on this exact infrastructure blocker; applying
  it here is not a post-hoc escape hatch.
- `ran=false` with a pre-registered routing clause is a legitimate KILL path (cf.
  standing rule #1, confirmed 5× this loop); not a silent skip.

## Verdict
**KILL confirm.** DB already reflects `status=killed` with K1572 evidence and
Finding #602 registered. No action needed beyond `review.killed` handoff to the
Analyst for LEARNINGS.md.

## Non-blocking notes (for future V2)
1. When P1/P2/P3 unblock (class-level via T2.1 rerun + persona corpus),
   this experiment and sibling `exp_followup_sft_behavioral_lora_scale_5` should
   merge into a single survivor — duplicate scope by the Kc-band differ only.
2. Reference list in DB flagged incomplete (`references` field empty); MATH.md
   and PAPER.md cite 5 refs including arxiv IDs — suggest sync'ing via
   `experiment reference-add`.
