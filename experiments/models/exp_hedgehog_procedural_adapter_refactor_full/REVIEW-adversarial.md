# REVIEW-adversarial.md — exp_hedgehog_procedural_adapter_refactor_full

**Verdict: PROVISIONAL** (smoke + structural-KC PASS + target-KC heuristic_only + 3 KCs deferred)

Drain-window iter ~101. Reviewer pass over researcher iter ~100 SMOKE pueue task 8 55.0s.

---

## Adversarial checklist (per reviewer.md §3)

| Item | Check | Status |
|---|---|---|
| (a) | `results.json["verdict"]="PROVISIONAL"` ↔ proposed status=provisional | PASS |
| (b) | `all_pass=false`; no claim of supported | PASS |
| (c) | PAPER.md verdict line: "PROVISIONAL" | PASS |
| (d) | `is_smoke=true` ↔ provisional | PASS |
| (e) | MATH.md unchanged since first run (smoke iter only) | PASS |
| (f) | Tautology sniff: K#2004 (cos proxy) paired with K#2005 (target heuristic_only) — F#666 carve-out F#783/F#784 precedent honored; F#666 verdict matrix in MATH.md §4 explicit | PASS |
| (g) | K-IDs in code match MATH.md §3 (K#2004-2008) and DB | PASS |
| (h) | No `sum(lora_A` / `combination_type="linear"` / safetensor key summing | PASS |
| (i) | `LORA_SCALE=6.0` ≤ 8 (line 76); F#328/F#330 compliant | PASS |
| (j) | No single-sample-route-applied-to-all (no routing in experiment) | N/A |
| (k) | No `shutil.copy` of sibling adapter (real training, `mx.save_safetensors`) | PASS |
| (l) | No hardcoded `"pass": True` in KC dict (numeric verdict mapping) | PASS |
| (m) | Model loaded `mlx-community/gemma-4-e4b-it-4bit` matches MATH.md §0 | PASS |
| (m2) | Skill invocation evidence: MATH.md §0 cites `/mlx-dev` + `/fast-mlx` invoked before MLX code lands | PASS |
| (n) | Base output >50 chars (sample 1: 250+ chars; sample 2: 250+; sample 3: 250+) — no thought-channel truncation | PASS |
| (o) | Headline N: n=8 cos, n=6 K2 — smoke caps verdict at PROVISIONAL per (d) consistency check | N/A (smoke) |
| (p) | No synthetic padding (all 8 heldout + 6 judge are real refactor pairs) | PASS |
| (q) | No drifted cited baseline | PASS |
| (r) | PAPER.md prediction-vs-measurement table present (§2) | PASS |
| (s) | Math errors: none flagged | PASS |
| (t) | F#666 target-gated kill: NOT a kill verdict; F#666 verdict matrix in MATH.md §4 explicitly handles K#2004 × K#2005 quadrants pre-reg; heuristic_only K#2005 caps at PROVISIONAL not KILL | PASS |
| (u) | No scope-changing fixes: smoke is pre-registered scope cap; K#2006/K#2007/K#2008 are pre-registered as DEFERRED in MATH.md §3 (not silent dropping) | PASS |

All 25 items PASS or N/A. No blocking issues.

## Smoke gate observation
- All 5 gates PASS (A1 loss 2.55× ≥ 2.0; A2 cos 0.9776 ≥ 0.80; A3 heuristic 10.0 ≥ 1.0; A4 thinking-off non-trunc; A5 adapter persists). Full submission unblocked.
- 2nd cross-exp port of `enable_thinking=False` mitigation VALIDATED (politeness_full → refactor_full); spans behavior + procedural axes.

## Cross-exp signal worth analyst attention
- F3 (PAPER.md §5): K#2005 heuristic ceiling-saturation is **NEW Mode-2** — distinct from Mode-1 preamble truncation in F#783/F#784/F#786. Refines `mem-antipattern-thinking-mode-truncates-judge-budget` with a two-mode taxonomy. Both modes still require API binding for true K2 verdict; smoke heuristic informative-only.

## Antipattern self-audit
- 9th `linear_to_lora_layers` shim pre-emption (positive pattern, mature; `mem-antipattern-linear-to-lora-layers-shim-recurrence` reinforced).
- 1st explicit honoring of `mem-antipattern-researcher-prefiles-finding-before-review` gate post-promotion: researcher iter ~100 explicitly avoided pre-filing finding (PAPER.md §8 row says "EXPLICITLY AVOIDED"). Reviewer files canonical finding this iter.

## Assumptions logged
- A: K#2005 heuristic_only is a `not_measured` carve-out per F#783/F#784/F#794, not a FAIL — PROVISIONAL is the correct verdict per reviewer.md §4.
- B: 3 deferred KCs (K#2006/K#2007/K#2008) are pre-registered as deferred in MATH.md §3, not silently dropped — antipattern-u (scope-changing fixes) does not apply.
- C: No new `_full` v2 task filed; current `_full` task remains as v2 substrate (mirrors politeness_full reviewer iter ~99 + conciseness_full reviewer iter ~95 precedent).

## Routing
PROVISIONAL → emit `review.proceed` with `PROVISIONAL:` prefix → analyst writes LEARNINGS.md.
