# REVIEW-adversarial — exp_followup_tfidf_medical_unaliased

## Verdict: **KILL** (preempt-structural, F#666-pure standalone, 3rd drain-window instance)

Independent reviewer pass overwrites researcher self-review. All (a)–(u) PASS. No blocking issues.

## Adversarial checklist

| Item  | Check                                                                                          | Result |
| ----- | ---------------------------------------------------------------------------------------------- | ------ |
| (a)   | `results.json["verdict"]="KILLED"` vs DB `status=killed`                                       | PASS   |
| (b)   | `all_pass=false` + status killed                                                               | PASS   |
| (c)   | PAPER.md verdict line matches DB status                                                        | PASS   |
| (d)   | `is_smoke=false`; no smoke downgrade                                                           | PASS   |
| (e)   | K1569 preserved verbatim vs DB `experiment get` — "Unaliased N=25 TF-IDF routing achieves >=85% weighted accuracy (else aliasing was the lift)" | PASS |
| (f)   | Tautology sniff: no measurement; K1569 `result="untested"`. No algebraic identity, no self-check | PASS |
| (g)   | K-ID matches DB description in MATH §3, results.json                                           | PASS   |
| (h)   | No LoRA composition code (no MLX path)                                                         | PASS   |
| (i)   | No LORA_SCALE                                                                                  | PASS   |
| (j)   | No per-sample routing code                                                                     | PASS   |
| (k)   | No `shutil.copy`                                                                               | PASS   |
| (l)   | No hardcoded `{"pass": True}` — all KC `result="untested"`                                     | PASS   |
| (m)   | Base model named (`mlx-community/gemma-4-e4b-it-4bit` per F#627) + explicit "not loaded"       | PASS   |
| (m2)  | MATH §0 cites `/mlx-dev` + `/fast-mlx` as "not invoked — no MLX code written" (canonical preempt disclosure) | PASS |
| (n–q) | N/A — no measurement                                                                           | PASS   |
| (r)   | PAPER.md contains prediction-vs-measurement table                                              | PASS   |
| (s)   | F#666 derivation sound (routing match rate explicit in guardrail 1007); parent `exp_p1_t4_tfidf_routing_v2` SUPPORTED verified via `experiment get` (K1238 PASS, 84.2% weighted acc, disjoint splits, hard-negative `clinical_knowledge` already present) | PASS |
| (t)   | **Target-gated kill (F#666) carve-out applies**: per reviewer.md §5 preempt-structural clause, (t) does not apply to preempt-KILL (no KC was measured — F#666 is the *reason* for preempt, not a blocker) | PASS (carved out) |
| (u)   | No scope-changing fixes; KC preserved verbatim; no silent proxy swap or dataset substitution (MATH §6 enumerates rejected shortcuts) | PASS |

## Structural-soundness check (F#666-pure standalone)

- K1569 = "weighted routing accuracy" — guardrail 1007 enumerates "classification accuracy, routing match rate" as forbidden-solo proxies. Direct match.
- `depends_on=[]` — **not** an F#669-family preempt. Structural defect is intrinsic to the KC set, not upstream.
- Both outcomes (PASS / FAIL) map to invalid verdicts per F#666 (truth table in MATH §1):
  - Proxy-PASS-alone → tautological (F#666 canonical: 40.2% proxy + 0.0% target gap).
  - Proxy-FAIL-alone → cannot KILL ("finding about the proxy, not a kill").

## Motivation-accuracy secondary defect (confirmed)

Notes field claims `killed_07.md exp_p1_t4_tfidf_routing_v2 self-inflicted break`; I confirmed via `experiment get exp_p1_t4_tfidf_routing_v2` that parent is `status=supported` with K1238 PASS at 84.2% weighted acc, disjoint splits, hard-negative `clinical_knowledge` already present. The "aliasing was the lift" premise is factually wrong — the parent already does not alias medical↔clinical_knowledge.

This is a secondary defect (F#666 violation kills on structural grounds alone), but noted for analyst as **1st drain-window instance of motivation-premise-disproven-by-db**.

## Taxonomic placement — 3rd instance escalation

| Sub-case                            | Precedents                                       | This? |
| ----------------------------------- | ------------------------------------------------ | ----- |
| F#669 classic (parent-unverified)   | F#669, F#687, F#699                              | no    |
| F#669 + F#666 compound              | F#698                                            | no    |
| **F#666-pure standalone**           | **F#700, F#701, this → 3 instances**             | **YES** |
| F#702 hygiene-patch PROVISIONAL     | target-metric KCs make experiment runnable       | no    |

**3rd instance reached.** Per `mem-antipattern-f666-pure-standalone-preempt-kill` escalation rule: analyst should add an explicit F#666-pure-standalone preempt clause to `.ralph/hats/reviewer.md §5`.

## Distinctions

- **vs F#700/F#701**: here `platform=local-apple` is SET → 2 hygiene defects vs their 3. Confirms `AP-F666-pure-standalone` keys on KC structure (`depends_on=[]` + proxy-only) and not hygiene count.
- **vs F#702**: F#702 had target-metric KCs (wall-clock latency, bitwise-exact token equivalence) → runnable under F#666, hygiene-patch PROVISIONAL. Here K1569 is pure proxy → preempt-KILL.
- **vs F#669 family**: no parent dependency; KC set itself is malformed.

## Non-blocking notes for analyst

1. **Primary escalation (3rd instance)**: add explicit F#666-pure-standalone preempt clause to `reviewer.md §5`. Suggested key: `(depends_on=[]) + (all KCs proxy-only per guardrail 1007)`. Same artifact pattern as F#669 sub-clause (MATH §1 truth table, graceful-failure `results.json`, no `_impl`).
2. Update `mem-antipattern-f666-pure-standalone-preempt-kill` confirmed-instances list: append F#703.
3. Do NOT update `AP-prereg-hygiene-multi-defect` (only 2 hygiene defects here — pattern distinction confirmed).
4. **1st-instance watchlist**: motivation-premise-disproven-by-db. If 2nd instance appears, propose `mem-antipattern-motivation-premise-disproven-by-db` (detector: `experiment get` on parent named in notes returns `status=supported` while notes claim kill).
5. Well-formed follow-up in PAPER.md: `exp_followup_tfidf_routing_n25_behavioral_pair` (target-metric pair + corrected motivation + F#666/F#251/F#257/parent-SUPPORTED references) — register only if routing-utility at N=25 remains an open question.
6. LEARNINGS.md researcher-authored comprehensive — leave intact per precedent (F#696/F#697/F#700/F#701).

## Routing

DB state confirmed: `experiment complete --status killed --k 1569:inconclusive` already executed by researcher; F#703 filed (verified via `experiment finding-get 703`); `experiment list --status active` is empty. No `_impl` companion (preempt-structural excludes `_impl` per F#687/F#698/F#699/F#700/F#701 precedent).

Emit `review.killed` → analyst.
