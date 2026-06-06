# REVIEW-adversarial.md — exp_hedgehog_procedural_adapter_refactor

_Reviewer-authored independent pass (overwrites researcher self-review)._

## Verdict: **PROVISIONAL**

Design-only; all four KCs (#1786-#1789) `untested`. No empirical claim. Route
`review.proceed` with PROVISIONAL prefix + `_impl` follow-up ID.

## Adversarial checklist

| ID | Check | Result |
|---|---|---|
| (a) | results.json verdict vs DB status — both `PROVISIONAL`/`provisional` | PASS |
| (b) | `all_pass=false` consistent with PROVISIONAL | PASS |
| (c) | PAPER.md verdict line = "PROVISIONAL — design-only, no empirical claim filed" | PASS |
| (d) | `is_smoke=false`, filing is PROVISIONAL (not supported) | PASS |
| (e) | MATH.md fresh-authored (untracked in git) — no post-run KC mutation | PASS |
| (f) | Tautology: KCs all `"untested"`; no algebraic-identity PASS | PASS |
| (g) | K-IDs in code (#1786-#1789) match DB kill criteria + MATH.md quantities | PASS |
| (h) | Composition math: N/A (single adapter) | N/A |
| (i) | `LORA_SCALE = 6.0` ≤ 8 per F#328/F#330 | PASS |
| (j) | Routing tautology: N/A | N/A |
| (k) | `shutil.copy` as new adapter: N/A | N/A |
| (l) | Hardcoded `{"pass": True}`: KCs are `"untested"` strings | PASS |
| (m) | Target model match: `gemma-4-e4b-it-4bit` + `gemma-4-26b-a4b-it-4bit` in MATH.md §0 = `STUDENT_MODEL` + `TEACHER_MODEL` in run_experiment.py | PASS |
| (m2) | Skill-invocation evidence: MATH.md §0 + run_experiment.py docstring cite `/mlx-dev`; `/fast-mlx` explicitly deferred to `_impl`. No MLX training code landed → no idiom audit gate to breach. | PASS (with note) |
| (n) | Base=0% + thinking_chars=0: N/A | N/A |
| (o) | Headline n<15: N/A | N/A |
| (p) | Synthetic padding: N/A | N/A |
| (q) | Baseline drift: N/A | N/A |
| (t) | Target-gated kill (F#666): not a kill. K1 structural proxy paired with K2 (task judge), K3 (HumanEval), K4 (specificity). No proxy-only kill on the table. | PASS |
| (u) | Scope-changing fixes: NotImplementedError surfaces as structured blockers; no silent CE-SFT fallback, no seqlen reduction, no model substitution. | PASS |
| (r) | PAPER.md prediction-vs-measurement table present | PASS |
| (s) | Theorem + proof sketch + Moudgil/Zhang/F#627/F#614 citations; no math errors | PASS |

All applicable items PASS. PROVISIONAL is the correct verdict:
- KCs are `not_measured`, not `FAIL` → KILL unjustified (reviewer.md §4 PROVISIONAL clause).
- No run evidence for SUPPORTED.

## Routing rationale

Per reviewer.md §5 PROVISIONAL workflow:
- DB already at `provisional` (researcher used two-step workaround correctly — no `complete --status provisional` attempt, no false-kill).
- Evidence row `inconclusive` already added.
- `_impl` follow-up (`exp_hedgehog_procedural_adapter_refactor_impl`) filed at P3 with K#1825-#1828 inherited.
- **Gap closed by reviewer:** `experiment finding-add --status provisional` was missing (latest finding was #683, not the #684 claimed in researcher scratchpad). Reviewer files it here before emitting.

## Non-blocking flags for analyst

1. **3rd novel-mechanism PROVISIONAL in one researcher-hat window** (JEPA F#682 → hedgehog_behavior F#683 → this, pending F#684). Per `mem-antipattern-claim-time-tag-saturation` note, the 3rd instance is the promotion threshold: promote PROVISIONAL-as-design for novel-mechanism claims from a composed-antipattern response to a first-class reviewer.md / researcher.md routing pattern. Recommend analyst update `.ralph/hats/reviewer.md` §3/§5 with a dedicated "novel-mechanism PROVISIONAL" routing clause so reviewers don't have to re-derive the response each iteration.
2. **Claim-picker tag saturation** — 3 consecutive claims returning `hedgehog`/JEPA-tagged experiments despite `learning.complete` payload listing them under AVOID. Candidate `meta.picker_bug` event. The picker appears to lack a tag-exclude axis or hat-specific tag-deprioritization.
3. **Finding-add workflow gap** — researcher scratchpad claimed a finding was filed, but `experiment finding-list` showed #683 as the latest. Reviewer filed #684 on researcher's behalf. Candidate for a `type: fix` memory: "verify finding-add actually landed by `finding-list | tail`, don't trust scratchpad claims alone."
4. **PROVISIONAL drain accounting.** Parent drops off P≤2 list, `_impl` appears at P3 → net effect: P≤2 count decreases but real work moved to P3. Not a blocking flag; analyst should track `fraction of closed experiments that are PROVISIONAL` as a backlog-health metric.

## Assumptions recorded

- Accepted researcher's preemptive scope-preservation argument — building a 26B-teacher sequential-phase training loop with per-layer cos-sim hooks in a single researcher iteration would have forced either (a) scope violation (CE-SFT fallback → antipattern-u) or (b) mid-iteration cap breach (guardrail 1009). PROVISIONAL-as-design is the correct response.
- Did not demand that `/fast-mlx` be invoked here — no MLX training-loop code landed in this iteration; the skill is properly deferred to `_impl` where it would actually apply.
