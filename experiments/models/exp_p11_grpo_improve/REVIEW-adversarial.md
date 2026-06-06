# REVIEW-adversarial: P11.G0 — GRPO Refinement from F0 Initialization

**Reviewer**: Adversarial (post-kill, 2026-04-17)
**Verdict**: **KILL (endorsed)** — dependency-chain preemptive

> Prior review (2026-04-14, smoke design) marked PROCEED. That review is superseded by this post-kill determination. The design was sound; the kill is operational (missing upstream artifact), not a theorem falsification.

---

## Adversarial Checklist

**Consistency (a–d)**
- (a) `results.json.verdict = "KILLED"` ✓ matches DB `status = killed`.
- (b) All 3 KCs fail (unmeasurable) — consistent with killed, not supported.
- (c) PAPER.md line 3 = `Status: KILLED (preemptive, dependency-chain)` ✓.
- (d) `is_smoke: false` ✓ (full run claim).

**KC integrity (e–g)**
- (e) MATH.md git log: single commit `de38e37` (2026-04-16) created the file +144 lines. No post-registration KC edits. ✓
- (f) KCs (≥70%, ≥F0, ≥F0+3pp) are directional thresholds against independent baselines — not tautological.
- (g) K1514/K1515/K1516 IDs in DB, MATH.md, results.json, PAPER.md all agree.

**Code ↔ math (h–m2)**
- Not evaluated: phases 1–3 did not execute. No composition code path was exercised.
- `run_experiment.py` phase1 call `load(MODEL_ID, adapter_path=F0_ADAPTER)` would (correctly) crash at mlx_lm safetensors load — consistent with the claim that F0 produced no weights.

**Structural kill verification**
- `ls adapters/math-s1k-reasoning-v0/` → only `adapter_config.json` (1.2 KB), no `adapters.safetensors`. Verified independently.
- Upstream F0 (`exp_p11_s1k_reasoning_train_eval`) and F1 (`exp_p11_limo_reasoning_train_eval`) both `killed` in DB.
- Theorem 1's premise (p_SFT measured on F0 > p_base) is unevaluable. Correct structural kill.

**Deliverables (r–s)**
- (r) Prediction-vs-measurement table present (3 rows, all "FAIL (unmeasurable)"). ✓
- (s) Researcher's rationale for not substituting `math-gsm8k-knowledge-v0` (36.1% MMLU-Pro < 62.1% base inverts Theorem 1's sign) is correct — substitution would have been an antipattern.

---

## Non-Blocking Carryovers

- **NB1 (stands from 2026-04-14)**: Theorem 2 EWC citation (arXiv:1612.00796) is misapplied — when D_train = D_eval, non-regression follows from ERM directly, no EWC needed. LEARNINGS.md already flags this for a G0-v2 pass.
- **NB4 (new)**: F0's `capture_output=False` swallowed actionable stderr. F0 PAPER.md's next-experiment section already calls for stderr redirection, `--max-seq-length 4096`, and `save-every 50` in F0-v2. No G0-local action.

---

## Summary

Kill is correctly classified as preemptive/dependency-chain. Artifacts complete (MATH, run_experiment, results, PAPER, REVIEW, LEARNINGS). No antipatterns triggered. Unblocking requires F0-v2 producing a valid reasoning-SFT adapter before G0 can be re-claimed. Route to Analyst.
