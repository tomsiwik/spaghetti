# REVIEW-adversarial: followup_composition_correct_delta

**Verdict:** KILL (endorsed — 2026-04-18, 4th antipattern-017 instance)

**Reviewing:** `experiment.done exp_followup_composition_correct_delta: KILLED preemptive`.

## Adversarial checklist

**Consistency (a-d):** ✓
- (a) `results.json["verdict"]=KILLED` ↔ DB `status=killed` ↔ PAPER verdict=KILLED. Consistent.
- (b) `all_pass=false`; K1548 marked `fail`; status=killed. Consistent.
- (c) PAPER line 1 "(KILLED, preemptive)". No `supported`/`PROVISIONAL` language.
- (d) No `is_smoke` flag (non-run).

**KC integrity (e-g):** ✓
- (e) MATH.md has no git history — file uncommitted; no post-data KC edits possible. `git diff MATH.md` empty.
- (f) No tautology; kill is a direct artifact-audit.
- (g) K1548 ID aligns across MATH.md:67, PAPER.md:31, results.json:5-10, DB `#1548`.

**Code ↔ math (h-m2):** ✓
- (h) `run_experiment.py` is a no-op kill-reporter — no `sum(lora_A`, no `add_weighted_adapter`, no buggy composition primitives.
- (i) No `LORA_SCALE` constant.
- (j) No routing logic.
- (k) No `shutil.copy`.
- (l) Hardcoded `{"result": "fail"}` is the honest kill marker, not a pass hardcode.
- (m) No model loaded; no proxy-substitution risk.
- (m2) Non-blocking: no MLX inference executed.

**Eval integrity (n-q):** N/A — no eval run. Finding #14 baseline is cited, not measured here.

**Deliverables (r/s):** ✓
- (r) Prediction-vs-measurement table at PAPER.md:29-34.
- (s) Math: Thm 1 (identity by index expansion), Thm 2 (triangle inequality linear bound), Thm 3 (cross-term semantic-mismatch argument) — all sound. Theorem 2 bounds ΔW_F linearly in N, but the map from weight-norm to PPL is nonlinear — K1548 is *not* trivially implied by the math, so empirical measurement is genuinely needed (kill is not miscategorized as "proven").

## Independent verification of claims

1. **5-of-5 stubs** — ran the pre-flight grep:
   ```
   find micro/models/exp_p1_t2_single_domain_training/adapters \
        micro/models/exp_p1_t2_multi_domain_5/adapters -maxdepth 2 -type f
   ```
   Returned 5 paths, all `adapter_config.json`, zero `adapters.safetensors`. ✓
2. **DB evidence contradiction** — DB claims `exp_p1_t2_multi_domain_5` `supported` with "K1047 PASS: all 5 adapters ≥+3pp" but weight files do not exist. This is the antipattern-017 signature precisely: status row out of sync with filesystem.
3. **K1548 unmeasurability** — loading a stub directory via mlx-lm's `load(..., adapter_path=...)` either errors or silently returns base; composed PPL ≡ base PPL or crash; either way K1548 is vacuous.

## Antipattern-017 instance count

Researcher says "3rd confirmed instance (J0 + M0 + this)". Counting from the audit lineage that's correct; including the original `exp_p11_baseline_eval` it is the 4th. Bookkeeping-only; non-blocking. Analyst should reconcile to the audit framing they prefer.

## Actions

- DB already `status=killed --k 1548:fail` from researcher — no DB writes needed.
- No new finding added — mechanism is pure composition of Finding #14 + antipattern-017 recurrence.
- Emitting `review.killed` for Analyst.

## Open threads for Analyst

- Promote antipattern-017 to at least **4 confirmed instances** (reconcile count: baseline_eval + J0 + M0 + this).
- Consider canonical pre-flight grep included as an antipattern-017 entry-section: `find .../adapters -maxdepth 2 -type f | grep -v adapters.safetensors`.
- P11.ADAPTER-REBUILD remains the atomic unblock for any composition-class experiment; without it, M0 v2, J0 v2, and this v2 all stay blocked.
