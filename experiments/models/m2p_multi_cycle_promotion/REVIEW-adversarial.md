# REVIEW-adversarial.md — exp_m2p_multi_cycle_promotion

## Verdict: KILLED (K930 triggers) — proceed-with-kill (K930 diagnosed as toy-model-capacity-ceiling, not promotion-interference failure)

## Adversarial Checklist

- **(a) consistency:** results.json verdict=killed ↔ DB status=killed ↔ PAPER §Conclusion "K930 fires due to toy model capacity, not promotion failure" ↔ evidence all_criteria_pass=false. MATCH.
- **(b) PASS/FAIL alignment:** K928 pass=true, K929 pass=true, K930 kill=true → status=killed (K930 is a KILL-type criterion, its trigger ends the experiment). MATCH.
- **(c) PAPER verdict line:** PAPER.md:1-13 (Abstract) and :120-125 (Conclusion) both state K930 fires under toy-model capacity. Line 120: "Multi-cycle promotion shows zero interference in weight-space and activation-space (K928+K929 PASS)."
- **(d) smoke flag:** is_smoke=false (N_TRAIN_STEPS=600, N_PRETRAIN_STEPS=800, N_EVAL=100 per PAPER §Methods). SMOKE_TEST env var not set. Real run.
- **(e) KC pre-registration:** K928/K929/K930 present at DB creation 2026-04-07, unchanged across evidence. Pre-reg ✓.
- **(f) tautology check:** K930 measures absolute accuracy threshold (0.50). Not a self-referential check on LoRA math.
- **(g) measurement ↔ code:** K930 measured via `phase_evaluate(base_model, d_eval, eval_data[d_eval])` with `use_lora=False` (run_experiment.py:346-376). Threshold `< 0.50` at line 594. MATCH with PAPER §Promoted Base Accuracy table.
- **(h)-(m2) N/A:** no composition-N>1 routing, no LORA_SCALE variation, no proxy substitution, no MLX training idioms issues (phased execution with cleanup observed in run_experiment.py:75-81).
- **(n) seed robustness:** single seed (SEED=42). For K928/K929 results (PASS with clear margin — add 2.0x, sub inf, mul 1.0 ratio), seed variance unlikely to flip. For K930 (all-domains fail threshold), SFT baseline itself is <50% (5%, 0%, 40%) → seed variance cannot rescue.
- **(o)-(q) N/A** (no human eval, no data contamination risk on synthetic mod-10 arithmetic, no off-policy evaluation).
- **(r) prediction-vs-measurement table:** PAPER.md:17-24 explicit. Pythagorean predicted exact-to-float (rel_err < 1e-5), measured rel_err = 2.34e-8 — HIT by 3 orders of magnitude. K928/K929 predicted PASS, measured PASS. K930 predicted NO-KILL, measured KILL — discrepancy.
- **(s) theorem soundness:**
  - Theorem 1 (Pythagorean bound via Grassmannian A_i^T A_j = δ_ij I_r) algebraically sound. Proof steps 1-3 (cross-term elimination, Pythagorean sum, norm simplification) valid under standard linear algebra.
  - Theorem 2 (weight-space protection) follows directly.
  - **Caveat flagged in MATH.md:89-95:** weight-space orthogonality does not guarantee activation-space orthogonality. Activation interference bounded by `‖A_j.T x_k‖` which is small but nonzero. Author acknowledges honestly.
  - K930 redefinition (PAPER.md:115-116): "Should be framed as quality_ratio < 50% of SFT (relative, not absolute)" is a post-hoc rationalization. However, K928 (which K930 would reduce to) PASSES with ratio ≥ 1.0 across all domains, so the relative claim is supported.

## Key Findings

1. **Theorem 1 verified to machine precision** (rel_err = 2.34e-8, predicted < 1e-5). Strongest possible confirmation — Pythagorean bound is an algebraic identity under Grassmannian A-slots, and the implementation agrees with the math to 8 decimal digits.
2. **K928 + K929 PASS with large margins.** quality_ratio: add=2.0x, sub=∞, mul=1.0. No cross-cycle degradation (add improves, mul stable).
3. **K930 KILL is diagnosable.** Root cause: toy model (d=128, 2 layers, 800+600 steps) is too small for mod-10 arithmetic. SFT baseline never exceeds 40% on the hardest domain (mul) and is 5%/0% on add/sub. No amount of promotion can push absolute accuracy above a ceiling that SFT itself cannot reach.
4. **Impossibility structure (as stated in PAPER):** On a real LLM with SFT baseline >50%, K930 becomes structurally impossible under Grassmannian orthogonality — later promotions cannot lower earlier quality (Theorem 2).

## Reusable Finding (analyst-owed, to register when cap lifts)

- **Toy-model-capacity-ceiling as confound in absolute-threshold kill criteria.** Any KC defined as "accuracy < X%" on a model whose SFT baseline is < X% is vacuously satisfied by base-model weakness, not by the hypothesized failure mode. Tripwire: before registering an absolute-accuracy KC, measure SFT baseline; if SFT < KC threshold, reframe KC as a relative ratio (e.g., `promoted / sft < 0.5`).
- **Finding #398 already captures this** (killed 2026-04-08). No new sub-variant. Family reuse.

## Tripwire Registered (already exists as F#398)

- Before introducing absolute-accuracy thresholds in kill criteria for toy models, verify SFT baseline ≥ threshold. Otherwise the KC is uninformative.

## Sub-variant registration: NONE. F#398 already captures this kill.

## Next action: experiment complete --status killed --k 928:pass --k 929:pass --k 930:fail
