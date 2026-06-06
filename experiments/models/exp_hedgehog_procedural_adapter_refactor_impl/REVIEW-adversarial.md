# REVIEW-adversarial — exp_hedgehog_procedural_adapter_refactor_impl

**Verdict: PROVISIONAL** (smoke + structural-KC PASS + target-KC `not_measured`)

Canonical pattern per reviewer.md line 62: `is_smoke=true` AND structural KC PASS with target KC `not_measured` (heuristic_only ≠ FAIL). F#783 (politeness_impl) is the immediate precedent for this cluster sub-form. K1 PASS (mean cos = 0.9706, n=8 held-out, all 42 layers > 0.92, threshold > 0.80, predicted 0.83). K2 ran in `heuristic_only` mode (no ANTHROPIC_API_KEY in pueue env, max_tokens=192 truncated thinking-mode preamble, both arms scored 10.0 length-floor) — `not_measured` per F#666. K3 (HumanEval) and K4 (non-refactor) deferred to `_full`, blockers logged.

## Adversarial checklist (18 items)

| # | Check | Result |
|---|---|---|
| (a) | results.json verdict=PROVISIONAL == DB status=provisional | OK |
| (b) | all_pass=false consistent with provisional | OK |
| (c) | PAPER.md verdict line "PROVISIONAL" == DB | OK |
| (d) | is_smoke=true → provisional (not supported) | OK |
| (e) | MATH.md inherited verbatim from parent (untracked, no KC drift) | OK |
| (f) | Tautology sniff: K1 measures cos(scale=0 teacher attn, scale=6.0 student attn) on **same model** — student attn = base + LoRA·scale, teacher attn = base. Not algebraic identity; LoRA must approximate the catalog-prompt routing perturbation. | OK |
| (g) | K-IDs in code (K#1825-K#1828) match MATH.md §4 and DB | OK |
| (h) | No `sum(lora_A`, no `add_weighted_adapter`, no safetensor key summing | OK |
| (i) | LORA_SCALE=6.0 ≤ 8 (F#328/F#330) | OK |
| (j) | No routing in this experiment | N/A |
| (k) | No `shutil.copy` of sibling adapters | OK |
| (l) | KCs marked `pass` / `heuristic_only` / `untested` — not hardcoded `True` | OK |
| (m) | Target model in MATH.md §0 (`mlx-community/gemma-4-e4b-it-4bit`) == loaded in run_experiment.py L62 (smoke; 26B teacher deferred to `_full`) | OK |
| (m2) | Skill attestation: MATH.md §0 cites /mlx-dev + /fast-mlx; PAPER.md "Pre-flight" cites /mlx-dev patterns inherited from politeness_impl. Code uses `mx.eval(model.parameters(), optimizer.state, loss)` per step (L313), `nn.value_and_grad` (L293), `mx.clear_cache` between phases (L169, L320, L389), `mx.set_memory_limit` with 8GB headroom (L55-57), `mlx.optimizers.AdamW` (L279). Idiomatic. | OK |
| (n) | Base accuracy not 0% / thinking truncation N/A — K2 measures Δ between two arms on same generation budget, not accuracy from base | OK |
| (o) | Smoke headline n=8 (K1) / n=6 (K2) — both <15. Non-blocking under PROVISIONAL; flagged as smoke caveat. `_full` raises N=50/50. | NON-BLOCKING |
| (p) | No synthetic padding — embedded smoke set has 24 real Fowler-catalog `c_pre` snippets | OK |
| (q) | No baselines cited from prior findings — token-space LoRA baseline deferred to `_full` | OK |
| (r) | PAPER.md prediction-vs-measurement table present (L30-35) | OK |
| (s) | Math: per-layer cos formula in code (L231-244) matches MATH.md §3 (eq. 6, Moudgil arxiv:2604.14191) | OK |
| (t) | Target-gated kill (F#666): verdict is PROVISIONAL not KILLED — `not_measured` ≠ FAIL. K1 paired with K2; K3+K4 deferred-not-dropped. Carve-out applies. | OK |
| (u) | Scope-changing fix: researcher hit a sizing bug (N_TRAIN=24+N_HELDOUT=8 = 32 > len(SMOKE_REFACTOR_PRES)=24 → silent slice empty heldout, pueue task 1). Fix was N_TRAIN=16, N_HELDOUT=8 — **arithmetic correction**, not mechanism swap. Pre-reg KCs unchanged. Scope preserved. Operational, not antipattern-(u). | OK |

18/18 PASS or non-blocking under PROVISIONAL.

## Why PROVISIONAL not SUPPORTED, not KILLED

- **Not SUPPORTED**: K2 is `heuristic_only`, K3+K4 deferred. PLAN.md §1010 #4: smoke completes as provisional, never supported.
- **Not KILLED**: K1 PASS; K2/K3/K4 are `not_measured` (not FAIL). F#666 carve-out: proxy-PASS-alone is tautological-SUPPORT, but here paired-target K2 is documented-deferred under blockers, not silently absent. F#666 KILL requires both proxy AND target to FAIL; neither happened.
- **Not preempt-KILL (F#669-family)**: parent `exp_hedgehog_procedural_adapter_refactor` is itself PROVISIONAL design-only, but children depending on it are themselves smoke implementations of the same mechanism — this experiment IS the parent's `_impl`. No transitivity-blocker.

## K2 heuristic-collapse (signal worth carrying forward)

Both base and student scored 10.0 (length-floor in `heuristic_refactor_score` L416: `post_len < pre_len*0.3 → 10.0`). Cause: `max_tokens=192` (L435) truncated *all* generations mid-`<|channel>thought` preamble before any code emerged. Smoke samples confirm: every `base_post`/`student_post` is mid-thought. Two consequences for `_full`:
1. Raise `max_tokens` to ~512, OR set `enable_thinking=False` for K2-judge generations only (preserve thinking for K1 which captures attn during full forward).
2. The same trap caught politeness_impl K2 (F#783); document in `mem-antipattern-thinking-mode-truncates-judge-budget` if it recurs once more (3-instance promotion threshold).

## Operational: K1=0.97 vs predicted 0.83

PAPER.md §"Why PROVISIONAL" point 6 explicitly flags this as the smoke shortcut: same-arch teacher/student with scale-toggle (teacher = E4B + catalog system prompt + scale=0; student = E4B + neutral prompt + scale=6.0) inflates the metric vs the canonical 26B-teacher comparison from MATH.md §0. The 0.83 prediction was for the 26B teacher; smoke uses E4B-teacher (an easier regime where the student is "closer" to teacher because both share the same base). `_full` runs the 26B teacher as written. Not a finding gain; an honest caveat.

## Assumptions logged

- F#783-cluster extension: refactor_impl PROVISIONAL same template as politeness_impl PROVISIONAL. New finding F#NEW per cluster-extension pattern.
- F#666 carve-out applies to PROVISIONAL — checklist (t) does NOT block.
- Sizing-bug fix (N_TRAIN=24→16) is operational arithmetic, not antipattern-(u) scope reduction. Same logic as the politeness-impl `linear_to_lora_layers` shim fallback at iter ~58.
