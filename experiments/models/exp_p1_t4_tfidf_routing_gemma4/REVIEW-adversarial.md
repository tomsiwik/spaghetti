# Adversarial Review (V2 audit-rerun, 2026-04-18)

**Reviewer:** Red-team pass (hat)
**Verdict:** KILL

Overwrites V1 `PROCEED` (2026-04-10). V1 preserved in git blame.

---

## Adversarial Checklist

| Item | Check | Result |
|------|-------|--------|
| (a) | results.json verdict=KILLED vs DB status=killed | ✅ matches |
| (b) | all_pass=false and verdict=KILLED consistent with K1074 fail | ✅ |
| (c) | PAPER.md "Status: KILLED" — no PROVISIONAL/INCONCLUSIVE leakage | ✅ |
| (d) | is_smoke=false (full run, 1533 N=25 + 497 N=5 samples) | ✅ |
| (e) | MATH.md git-diff: V2 Audit Section prepended above V1 Problem Statement; V1 thresholds K1073≥95%, K1074≥85%, K1075<1ms, K1076=0 byte-preserved in Quantitative Predictions (lines 201-206) | ✅ |
| (f) | Tautology sniff: K1073 honest on disjoint MBPP/GSM8K/PubMedQA/MMLU splits; K1074 fails honestly at 72.21%; K1075 measured on router25 (the router the KC is actually about); K1076 trivially 0 because no neural weights exist | ✅ |
| (g) | K-IDs 1073/1074/1075/1076 in code match DB kill-criteria | ✅ |
| (h) | No sum(lora_A) / add_weighted_adapter (N/A — TF-IDF, not LoRA) | ✅ |
| (i) | No LORA_SCALE (N/A) | ✅ |
| (j) | Per-sample routing: `predict()` computes `argmax` per row of `X_norm @ centroids.T` | ✅ |
| (k) | No `shutil.copy` (N/A) | ✅ |
| (l) | KC dict computed from measured booleans — no hardcoded `{"pass": True}` | ✅ |
| (m) | No model loaded (TF-IDF over text) — no proxy-substitution risk | ✅ |
| (m2) | Pure sklearn/numpy; `/mlx-dev`, `/fast-mlx` N/A | ✅ |
| (n) | No thinking channel (not a generation experiment) | ✅ |
| (o) | n=497 (N=5) and n=1533 (N=25), both ≫ 15 | ✅ |
| (p) | All 25 subjects are real MMLU/GSM8K/MBPP/PubMedQA — no synthetic padding, no B=0 | ✅ |
| (q) | V1 baseline (86.1%) explicitly retracted; V2 measured fresh | ✅ |
| (r) | PAPER.md V2 prediction-vs-measurement table present (lines 14-19) | ✅ |
| (s) | Math errors — Theorem 1 acceptable as heuristic; Theorem 3 predicted 0.1ms vs measured 0.656ms (off by ~6×) but PAPER.md attributes correctly to sparse-vocab-lookup dominance over inner product | ✅ |

All 19 items clean. No blocking issue.

---

## Independent Re-verify

1. **V2 code fixes real (not cosmetic).** `git diff run_experiment.py` shows:
   - `load_code_prompts` swapped from `openai_humaneval` (test-split-duplicated) to `mbpp` (`full` config, train/test upstream-disjoint, raises if pool too small — refuses to duplicate).
   - `load_mmlu_prompts` + `load_mmlu_test_prompts` collapsed into `load_mmlu_split`, which calls `_split_disjoint` (dedupe by text → shuffle once → index-slice `[0:n_train]` / `[n_train:n_train+n_test]` → `assert not overlap`).
   - `MMLU_EXTRA_SUBJECTS`: 6 hard-negatives (`clinical_knowledge`, `virology`, `high_school_biology`, `nutrition`, `human_sexuality`, `high_school_psychology`) restored; 6 easy subjects (`prehistory`, `high_school_european_history`, `high_school_us_history`, `astronomy`, `sociology`, `global_facts`) dropped; N_extra=20 preserved; comment explicitly names the hard-negatives.
   - K1075 measured on `router25` (final phase, after N=25 fit), not `router5`.
2. **V1 leakage signature matches V2 drops.** Finance domain: 91% V1 → 67% V2 (MMLU high_school_macroeconomics V1 overlap dominant; V2 disjoint leaves n=97 test). Code: V1 100% on 164-problem duplicated HumanEval → V2 100% on disjoint MBPP (MBPP text vocabulary is still distinctive — code passes honestly). Medical: 98% V1 → 91% V2 (PubMedQA test set was disjoint even in V1; the 7pp drop comes from the N=25 pool now including biology/nutrition/virology competitors). These drop patterns are the ones predicted by the audit mechanism.
3. **K1074 hard-negative cluster confirmed.** Six restored subjects: clinical_knowledge 36.4%, virology 42.9%, biology 59.0%, nutrition 48.0%, human_sexuality 27.3%, psychology 51.0% — mean 44.1%, vs the other 19 non-medical domains averaging ~74%. Medical ↔ life-science vocabulary cluster is the dominant failure mode, exactly as MATH.md V2 prediction.
4. **DB state.** `status=killed`, evidence entry 2026-04-18 `fail` added, K1073 [✓] K1074 [✗] K1075 [✓] K1076 [✓] — matches results.json. V1 evidence entry 2026-04-10 [pass] preserved alongside 2026-04-11 [fail] LOOPHOLE_AUDIT and 2026-04-18 [fail] V2 rerun. Audit trail intact.

---

## Kill classification

**NOT a precondition-probe kill.** This is a genuine empirical KILL on the original pre-registered K1074 threshold (85%) against fixed, leakage-free code. Neither upstream T2.1 (Gemma 4 training) nor any adapter artefact blocks execution — TF-IDF is pure text→domain classification, no model load.

**Mechanism of V1-invalidity**: mem-antipattern-007 (tautological KC via test-set curation) + data-leakage-via-padding + single-load-two-shuffles MMLU pattern. All four documented in `LOOPHOLE_*.md`.

**What V2 establishes honestly:**
- N=5 TF-IDF routing (coarse NLP task families): PASS at 96.78% on fully-disjoint splits. Finding #389 mechanism replicates.
- N=25 TF-IDF routing with semantic-neighbor hard-negatives: FAIL at 72.21% (13.79pp below 85% threshold). This retroactively narrows Findings #389 and #354 to "TF-IDF works when domains are not semantic neighbors."

---

## Kill-cause taxonomy (non-blocking notes)

1. **TF-IDF centroid geometry cannot separate semantic neighbors at the 85% bar.** Medical/clinical/biology/virology/nutrition share too much raw-vocabulary mass; their domain centroids overlap; a nearest-centroid rule in TF-IDF space has no escape.
2. **The V1→V2 swap is audit-motivated, not result-motivated.** Comment in code names the restored hard-negatives and frames the swap as "must separate medical from closest semantic neighbors." Swap preserves N_extra=20.
3. **Follow-up `exp_p1_p1_ridge_routing_n25` is already queued** with new KCs K1081/K1082/K1083 (leakage-free N=25, native MLX latency, format robustness). That is the right next step; this experiment is correctly closed.

---

## Assumptions (judgment calls)

- Treated MBPP as an adequate stand-in for the "code domain vocabulary" role. Both are English problem descriptions over Python-syntax exemplars; TF-IDF cares about the English text. Supported by PAPER.md's explicit assumption.
- Treated K1075 as specified on isolated sklearn/numpy CPU latency (MATH.md Theorem 3), not end-to-end CPU→GPU sync. LOOPHOLE_METHODOLOGY's CPU-GPU-sync concern belongs to follow-up K1082, not this KC.
- No round-2 revise cycle; the kill is clear and mathematically grounded.

---

## Routing

`review.killed`. Finding captured. V2 code + artefacts + MATH.md V2 audit section are the permanent record of the retraction.
