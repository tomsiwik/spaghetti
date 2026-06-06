# REVIEW-adversarial — exp_hedgehog_loss_variant_kl_div

**Verdict: PROVISIONAL** (novel-mechanism design-only sub-case — reviewer.md clause)

## Sub-case routing

Canonical PROVISIONAL (novel-mechanism design-only). Loss-variant ablation (KL-div
attention-map distillation) is a framework-level mechanism not executable via
`mlx_lm.lora` CLI; custom MLX training loops required for both arms. Structurally
equivalent to F#682/F#683/F#684/F#696/F#697/F#717/F#718 novel-mechanism precedents
for verdict purposes. 1st **loss-variant-ablation** sub-type — NOT an axis-extension
(reuses F#683 politeness axis).

## Required-artifact pattern (all 4 present)

1. ✓ MATH.md §0 cites `/mlx-dev` + `/fast-mlx` — satisfies (m2) carve-out.
2. ✓ `run_experiment.py main()` wraps all 6 phases in try/except; never raises; writes
   `results.json` with `verdict="PROVISIONAL"`, both KCs `"untested"`, 6 structured
   blockers. 1.6s runtime.
3. ✓ `_impl` follow-up `exp_hedgehog_loss_variant_kl_div_impl` filed at P=3 (KCs
   1959/1960 text-inheriting 1870/1871, grounded-by 2604.14191).
4. ✓ PAPER.md prediction-vs-measurement table: both rows "not measured" + 7 explicit
   scope-preservation rejections (axis mismatch / hyperparameter mismatch / KL direction
   drift / teacher proxy / rubric drift / step asymmetry / ε relaxation).

## Adversarial checklist

**Consistency (a–d):** all pass.
- (a) `results.json["verdict"]="PROVISIONAL"` ↔ DB `provisional` ↔ PAPER.md line 3 ↔ handoff consistent.
- (b) `all_pass=false` ↔ status=provisional (both KCs `untested`, not failed).
- (c) PAPER.md line 3 explicitly "PROVISIONAL (design-only)".
- (d) `is_smoke=false` but verdict is design-only PROVISIONAL — matches novel-mechanism sub-case; `is_smoke` orthogonal.

**KC integrity (e–g):** all pass.
- (e) K1870/K1871 are the original pre-reg KCs; no post-hoc relaxation.
- (f) **Tautology sniff — disclosed, does NOT preempt.** K1870 is tautological-for-cos-loss by construction (cos-loss optimizes cos-sim directly); PAPER.md §"K1870 tautology" explicitly discloses this. K1870 is retained as diagnostic ("did KL arm converge?"), paired with K1871 per F#666. Crucially, **K1871 is NOT tautological**: behavioral-quality judge is a downstream task test not optimized by either loss function. §5 tautological-inter-variant-delta preempt-KILL requires *all* inter-variant KCs to be tautological or lacking a base-anchored pair; K1871 is an independent behavioral target on the Hedgehog axis, not an algebraic identity. The F#666-compliant proxy+target pair rescues K1870's disclosed tautology.
- (g) K1870/K1871 text matches MATH.md §4 table and DB row.

**Code ↔ math (h–m2):** all pass or N/A.
- (h–l) N/A — design-only scaffold with `NotImplementedError` stubs. No composition sum, no unsafe LORA_SCALE, no per-sample routing hack, no `shutil.copy` spoof, no hardcoded `{"pass": True}`.
- (i) LORA_SCALE = 6.0 ≤ 8 ✓ (F#328/F#330).
- (m) Student E4B and Teacher 26B consistent between MATH.md §0 and `run_experiment.py` (lines 51/52).
- (m2) **Skill invocation evidence present.** MATH.md §0 explicitly names both skills and flags them as hard-gate per F#673/2026-04-17 audit. Code-level MLX idioms (`mx.set_memory_limit`, `mx.set_cache_limit`, `mx.eval + mx.clear_cache between batches` documented as the _impl requirement, `nn.value_and_grad` functional gradients) are named in docstrings. `/fast-mlx` deferred to _impl — acceptable: no training loop runs this iteration.

**Eval integrity (n–u):** all pass or inapplicable.
- (n) N/A — no eval runs.
- (o) N/A — no headline n.
- (p) N/A — no synthetic padding.
- (q) N/A — no drifted cited baseline.
- (t) **Target-gated kill — K1871 target present.** Not a KILL verdict; PROVISIONAL (design-only, target `untested` not `failed`). Per reviewer.md: "PROVISIONAL applies when structural-KC PASS with target-KC `not_measured` — `not_measured` is NOT `FAIL`, so KILL is unjustified". Same logic applies when both are unmeasured.
- (u) **Scope-preservation — honest design-only, NOT a silent scope swap.** PAPER.md lists 7 explicit scope-preservation rejections (axis mismatch, hyperparameter mismatch, KL direction drift, teacher proxy, rubric drift, step asymmetry, ε relaxation). Forward-KL direction locked at `KL(teacher || student)` in §1 tertiary-failure + A7; ε=1e-6 locked; reverse-KL/JS explicitly deferred. Honest filing — antipattern-u does not fire.

**Deliverables (r–s):** all pass.
- (r) PAPER.md prediction-vs-measurement table present; both rows "not measured".
- (s) Math is sound: §3 derivation shows K1870's tautology (cos-loss optimizes cos-sim directly), §3.4 weights-mismatch-differently argument supports null-favored prediction for K1871. §8 A1–A10 assumptions explicit. Forward-KL choice motivated (mode-seeking, analogous to teacher-forcing).

## F#702 hygiene-patch

3rd F#702 instance. Platform + success_criteria #94 patched before `experiment complete`. `references` field INCOMPLETE per F#702 precedent (global ref library CLI-linking limitation) — non-blocking for verdict. A8 discloses.

**Pairing classification** (analyst-deferred per MATH.md §A10): this filing could be either (a) 3rd same-pairing instance `novel-mechanism-primary + hygiene-patch-secondary` (triggering sub-classification promotion per F#718 analyst pre-commit), or (b) 1st-instance-of-new-sub-type (loss-variant-ablation vs axis-extension). Reviewer leaves classification to analyst; both routings converge on PROVISIONAL.

## Non-blocking observations

- **Transitive blocker on F#683 `_impl`.** Phase 0 corpus reuse depends on F#683 `_impl` landing. If F#683 `_impl` stalls, re-scope ablation to whichever Hedgehog axis `_impl` lands first (A1). Documented in PAPER.md Handoff — non-blocking for this verdict.
- **26B teacher cache** now blocks 8+ dependents (6 Hedgehog axis `_impl` + this ablation `_impl` + knowledge-gap-26B). Standalone prereq task candidate — non-blocking.
- **7 Hedgehog-framework PROVISIONALs, 0 `_impl` measured.** Loss-variant sub-type is structurally distinct from axis-extension (hard-defer-axis rule does NOT fire), but the broader 0-measurement concern persists. Analyst to weigh whether loss-variant sub-type opening is a separate concern or consolidates under the Hedgehog-framework pile. Researcher A9 correctly notes either K1870+K1871 outcome is forward-actionable.
- **K1870 tautology disclosure pattern.** Novel within Hedgehog-framework: 1st KC explicitly flagged tautological-for-one-variant + paired with independent target per F#666. If a 2nd Hedgehog experiment pre-registers a similarly-disclosed tautological proxy, consider memory-promoting as a standalone pattern.

## Assumptions logged (PLAN.md 1008)

- A8 classification routing (analyst-deferred) does not affect this PROVISIONAL verdict — both routings accept as canonical PROVISIONAL-novel-mechanism.
- Forward-KL direction lock at `KL(teacher || student)` is a single-choice lock, not a silent scope reduction — reverse-KL/JS explicitly enumerated as sibling-follow-up candidates if the null holds at _impl time.

## Verdict: PROVISIONAL

Route: two-step workaround (DB already set to provisional by researcher; F#719 landed and verified via `experiment finding-list --status provisional`). _impl follow-up at P=3 filed. Emit `review.proceed PROVISIONAL: exp_hedgehog_loss_variant_kl_div → _impl exp_hedgehog_loss_variant_kl_div_impl (F#719, 7th Hedgehog-framework, 1st loss-variant sub-type, 3rd F#702)`.
