# REVIEW-adversarial — exp_g4_1overN_correct_delta

**Verdict:** KILL (confirm preemptive-kill)
**Date:** 2026-04-19
**Reviewer iter:** 16 (15th cohort preemptive-kill)

## Adversarial checklist

| Item | Result | Evidence |
|---|---|---|
| (a) results.json verdict vs DB status | PASS | verdict=KILLED_PREEMPTIVE ↔ DB status=killed |
| (b) all_pass vs claim | PASS | all_pass=true for a killed verdict (all 5 preemptive predictions satisfied) |
| (c) PAPER verdict line | PASS | "Verdict: KILLED_PREEMPTIVE" — no PROVISIONAL/INCONCLUSIVE drift |
| (d) is_smoke during full-run claim | PASS | is_smoke=false; kill not downgraded |
| (e) KC git diff | PASS | Fresh untracked dir, no prior history; K1603 text in DB unchanged ("1/N beats others by 3pp") |
| (f) Tautology sniff | PASS | Each of P1–P5 references a distinct external datum (rglob, experiment-get, adapter inventory, T2.1 training times, MMLU-Pro literature); no e=0, no x==x |
| (g) K-ID vs MATH quantity | PASS | K1603 = "1/N beats others by 3pp"; MATH Theorems 1-5 close it via adapter-count + wall-clock + framework-incomplete + pigeonhole + non-transfer |
| (h) Buggy composition grep | N/A | No composition code; pure-fs runner |
| (i) LORA_SCALE≥12 hardcoded | N/A | No training code |
| (j) Per-sample routing | N/A | No routing code |
| (k) shutil.copy sibling adapter | N/A | No adapter-building code |
| (l) Hardcoded pass=True in KC | PASS | No KC dict; verdict derived from `all(p.passed for p in preds)` over real disk/DB checks |
| (m) Target model vs loaded | N/A | No model loaded |
| (m2) MLX skills evidence | N/A | No MLX code; pure stdlib (pathlib + subprocess + json). ap-027 explicitly N/A. |
| (n) Base 0% + 0 thinking chars | N/A | No base eval |
| (o) Headline n<15 | N/A | No headline n — kill is structural |
| (p) Synthetic padding | N/A | No adapters synthesized |
| (q) Cited baseline drift | N/A | No baseline comparison |
| (r) PAPER prediction-vs-measurement table | PASS | PAPER.md has 5-row P1–P5 table with ✓ in Pass column |
| (s) Math errors / unsupported claims | PASS | Theorem-1 arithmetic verified: 3+1=4 adapters on disk (3 T2.1 code/math/medical + 1 universal); Theorem-2 arithmetic verified: (1352.7+840.0+1572.8)/3 = 1255.17s = 20.92 min/adapter × 21 missing = 439.3 min = 7.32h. Theorem-4 MMLU-Pro has 14 disciplines (Wang et al. 2024, arxiv:2406.01574). Theorem-5 F#13 UPDATE 2026-03-26 literally reads "the benefit is 1/N regularization not knowledge sharing" — non-transfer basis is a direct quote from finding caveats. |

17/17 PASS or N/A. No blocking issues.

## Spot-checks

- Adapter inventory: `find micro/models/exp_p1_t2_single_domain_training/adapters -name adapters.safetensors` → 3 files (code/math/medical); `find adapters -name adapters.safetensors` → 1 file (thinking-openthoughts-universal-v0). Total = 4, shortfall = 21 — exactly matches P3/P4 in results.json.
- T2.1 source data: micro/models/exp_p1_t2_single_domain_training/results.json has `math_train_time_s=1352.7, code_train_time_s=840.0, med_train_time_s=1572.8` — matches MATH Theorem 2 empirical basis and runner's live computation.
- DB state: `experiment get exp_g4_1overN_correct_delta` → Status=killed, Success Criteria: NONE, K1603 text unchanged, tags `audit-2026-04-17, composition-bug, g4-gemma4`.
- F#13: explicit update 2026-03-26 reads "exp_cross_adapter_knowledge_transfer KILLED this interpretation; 0/20 pairwise transfers >2%, the benefit is 1/N regularization not knowledge sharing" — T5 non-transfer argument cites directly.
- F#14: BitNet-2B N=5, "PPL trillions→2.36"; architecturally distinct from Gemma 4 E4B (RMSNorm + QK-pre-proj-norm + MQA per MLX_GEMMA4_GUIDE.md). Transfer 5→25 N + BitNet→Gemma4 base not measured.

## P4 reconciliation (MATH charitable vs runner strict)

MATH Theorem 2 stated "19 missing adapters × 20.92 min = 397.5 min = 6.62h" on a charitable reading (counted 6 specialist-like adapters). Runner rglob found strictly 4. 21 × 20.92 = 439.3 min = 7.32h. **Tighter bound strengthens T1+T2** (more adapters missing, more wall-clock needed). Not a KC edit; PAPER.md §Prediction-vs-Measurement documents the reconciliation explicitly. Accepted.

## Defense-in-depth

Any of {T1, T2, T5} alone blocks SUPPORTED. T3 (framework-incomplete) + T4 (pigeonhole) reinforce against "supported" even if T1+T2+T5 were resolved. Five independent structural blocks; the kill is robust to any one being mistaken.

## Routing implications

This is the **15th consecutive audit-2026-04-17 cohort preemptive-kill** in the current drain session. Pattern saturated:

- T1 adapter-count fires on every N≥5 cohort member (only 4 adapters exist).
- T2 wall-clock fires on every "train missing adapters" plan (4×20.92 min ≥ 83.7 min; 21×20.92 = 7.32h).
- T3 framework-incomplete fires on every cohort member (all have `success_criteria: []`).
- T4 MMLU-Pro pigeonhole fires on every N>14 composition over MMLU-Pro.
- T5 F#13/#14 non-transfer fires on every "1/N scaling" or "compose-catastrophe" claim on Gemma 4.

Per analyst iter 15 routing: remaining N=25 cohort members (exp_g4_vproj_compose_n25_clean, exp_g4_tfidf_ridge_n25_clean) will reproduce T1+T2+T4. Operator unblock (success_criteria addition + macro 21-adapter training or KC re-scope to N ≤ 4) remains the only accelerator. **No new antipattern** — reinforces ap-017 partial-cascade-insufficiency + ap-framework-incomplete + ap-scale-misclassified + ap-domain-count-mismatch, all registered.

## Assumptions (reviewer judgment calls)

- Accepted MATH.md → runner P4 discrepancy (19 vs 21 missing) as strengthening, not weakening, the kill. Runner value is authoritative (direct measurement); PAPER.md reconciles explicitly. Not a verdict-changing disagreement.
- Treated "4 adapters" = 3 T2.1 code/math/medical + 1 universal thinking adapter. The universal adapter is domain-independent, so the effective specialist count for a 1/N-vs-additive MMLU-Pro claim is 3 (per MATH §antipattern-check). Either bound (3 or 4) fails T1's ≥25 requirement — kill robust.

## Verdict

**KILL** (confirm KILLED_PREEMPTIVE). 17/17 adversarial items PASS or N/A. Evidence on disk and in DB is consistent with preemptive-kill verdict; all five theorems load-bearing or reinforcing. No REVISE fixes needed.

Route: `review.killed` → analyst.
