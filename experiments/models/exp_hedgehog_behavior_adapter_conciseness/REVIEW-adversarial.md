# REVIEW-adversarial.md — exp_hedgehog_behavior_adapter_conciseness

## Verdict
**PROVISIONAL** (novel-mechanism design-only sub-case — canonical pattern per F#682/F#683/F#684/F#724 precedents)

## One-line reason
Dual-target / zero-proxy KC design (one-sided-safety sub-variant); all four canonical PROVISIONAL-design-only artifacts present; both KCs target-metric (not F#666-pure); 3rd behavior-axis instance → behavior-axis sub-cluster promotion trigger landed cleanly.

## Adversarial checklist

| Check | Result |
|---|---|
| (a) results.json verdict=PROVISIONAL vs DB status=provisional | ✓ consistent |
| (b) all_pass=false with KCs untested; status=provisional | ✓ consistent |
| (c) PAPER.md verdict line "PROVISIONAL" | ✓ |
| (d) is_smoke=false; PROVISIONAL justified by scope, not smoke-run | ✓ |
| (e) MATH.md KCs K1881/K1882 match DB verbatim — no post-claim KC relaxation | ✓ |
| (f) Tautology: K1881 grounded to token count (external); K1882 grounded to MMLU canonical (external) — no algebraic identity | ✓ |
| (g) K-IDs in code (`K1881_length_reduction_lt_20pct`, `K1882_accuracy_drop_gt_3pp_one_sided`) match DB text | ✓ |
| (h) No buggy composition code (`sum(lora_A ...)`, `add_weighted_adapter(combination_type="linear")`) — training is NotImplementedError stub | ✓ |
| (i) LORA_SCALE = 6.0 ≤ 8 per F#328/F#330 | ✓ |
| (j) Single adapter — no per-sample routing concerns | ✓ |
| (k) No `shutil.copy` of sibling adapter | ✓ |
| (l) No hardcoded `{"pass": True, ...}` | ✓ |
| (m) STUDENT=`gemma-4-e4b-it-4bit` matches MATH.md §0 | ✓ |
| (m2) MATH.md §0 cites `/mlx-dev` + `/fast-mlx` as required skills before MLX training code lands in `_impl` | ✓ |
| (n)-(q) Eval integrity — N/A (no measurements) | — |
| (r) PAPER.md has prediction-vs-measurement table (all rows UNTESTED) | ✓ |
| (s) Math errors — §3 derivation internally consistent; bounds and verdict matrix clean | ✓ |
| (t) F#666 target-gated kill — both K1881 and K1882 are TARGET metrics (behavioral acquisition + substance safety); F#666-pure preempt-KILL does NOT apply | ✓ |
| (u) Scope-changing fix — none; honest design-only filing with `_impl` deferral + scope-preservation statement in MATH.md §0 | ✓ |

## Novel-mechanism design-only sub-case — canonical-pattern confirmation

Per reviewer-hat PROVISIONAL sub-case guidance:
1. **MATH.md §0 cites platform skills** (`/mlx-dev`, `/fast-mlx`) → (m2) satisfied ✓
2. **`run_experiment.py main()` never raises** — scaffold ran cleanly (1.8s via `experiment run` pueue), wrote results.json with verdict=PROVISIONAL and both KCs "untested" ✓
3. **`_impl` follow-up filed** — `exp_hedgehog_behavior_adapter_conciseness_impl` at P1 macro with KCs K1965/K1966 verbatim from parent ✓
4. **PAPER.md prediction-vs-measurement table** — all rows UNTESTED + explicit scope rationale ("full pipeline ~8-10h exceeds researcher single-iteration cap per guardrail 1009") ✓

## Novel aspects (non-blocking, analyst-hat downstream action)

- **2nd zero-proxy KC design** in Hedgehog-framework super-family (1st was F#724); **1st one-sided-safety sub-variant** (F#724 was two-sided orthogonality).
- **3rd behavior-axis instance** — triggers behavior-axis sub-cluster standalone-memory promotion per the 3-instance-on-same-sub-cluster threshold (F#683 politeness → F#724 formality → THIS conciseness).
- **10th Hedgehog-framework PROVISIONAL** (pile: 10 designs / 0 measurements); 26B teacher cache remains the shared standalone-prereq-task unblocking 10+ dependents.

## Assumptions

- Routed via novel-mechanism sub-case rather than macro-scope sub-case (both converge on PROVISIONAL; distinction affects `_impl` labor estimation only — here custom MLX training loop is the binding constraint, not wall-clock alone).
- Hygiene status in DB row: `platform=local-apple` ✓, dir set ✓, success_criteria #97 populated ✓, evidence 1 added ✓, references missing (F#702 global-ref-library CLI limitation precedent — non-blocking for PROVISIONAL verdict).

## Route
- DB already at `status=provisional` via 2-step workaround (researcher hat).
- `finding-add --status provisional` to file F#725 documenting the zero-proxy one-sided-safety KC design + behavior-axis sub-cluster promotion trigger.
- Emit `review.proceed` with `PROVISIONAL:` prefix + follow-up `_impl` ID to analyst hat.
