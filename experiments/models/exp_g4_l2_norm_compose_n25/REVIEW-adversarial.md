# REVIEW-adversarial — exp_g4_l2_norm_compose_n25

**Reviewer hat.** Iteration 13 (post-cascade-drain).
**Date:** 2026-04-19
**Verdict:** **KILL (confirm preemptive)**

---

## Adversarial checklist (17 items, per reviewer hat a–s)

| # | Check | Result |
|---|---|---|
| (a) | `results.verdict = KILLED_PREEMPTIVE` vs DB `status=killed` | ✓ consistent |
| (b) | `results.all_pass=true`: all 5 predictions pass ⇒ kill confirmed | ✓ |
| (c) | PAPER verdict line "KILLED_PREEMPTIVE" matches | ✓ |
| (d) | `is_smoke=false` in results.json | ✓ |
| (e) | KC integrity: DB K1600 text unchanged, no git history (fresh dir, no pre-existing KC edits) | ✓ |
| (f) | Tautology sniff: 5 predictions are independent (fs inventory, DB state, wall-clock extrapolation from T2.1 empirical, MMLU-Pro dataset spec). No `x==x`. | ✓ |
| (g) | K-ID ↔ measurement: P1–P5 don't measure K1600 directly — they measure *preconditions* that make K1600 unreachable. Four distinct impossibility vectors. Preemptive-kill framing is valid. | ✓ |
| (h) | Composition code check: N/A — pure fs/JSON verification script, no LoRA composition | ✓ |
| (i) | `LORA_SCALE` ≥ 12: N/A — no training | ✓ |
| (j) | Per-sample routing: N/A | ✓ |
| (k) | `shutil.copy` of sibling adapter: N/A (grep clean) | ✓ |
| (l) | Hardcoded `{"pass": True}`: N/A — passed computed from `len(hits)==0`, `total < required`, `T_total_min >= threshold`, etc. | ✓ |
| (m) | Target model proxy substitution: N/A — no model load | ✓ |
| (m2) | Skill invocation: N/A — pure-fs Python script, no MLX API surface. ap-027 (venv-vs-system-python3) also N/A: runner is 100% stdlib + `subprocess("experiment get")`; `#!/usr/bin/env python3` shebang is safe because no MLX/datasets imports. | ✓ |
| (n) | Base eval truncation: N/A — no eval | ✓ |
| (o) | Headline n < 15: N/A — preemptive-kill, no inferential statistics | ✓ |
| (p) | Synthetic padding: N/A | ✓ |
| (q) | Baseline drift: N/A | ✓ |
| (r) | PAPER.md prediction-vs-measurement table (§4): present, 5 rows, all ✓ | ✓ |
| (s) | Math errors: see §"Theorem spot-checks" below | ✓ (all sound) |

**All 17 items pass or N/A.** No blocking issues.

---

## Theorem spot-checks

| # | Theorem | Spot-check |
|---|---|---|
| 1 | Adapter-count shortfall: 25 required, 4 exist (3 T2.1 + 1 universal thinking) | Verified `ls micro/models/exp_p1_t2_single_domain_training/adapters/` → `code/ math/ medical/` (3 dirs). T2.1 V3 `results.json.all_pass=true` confirms domain safetensors exist. Theorem 1 holds. |
| 2 | Wall-clock bound: 21 × 20.9 min = 7.32 h ≥ 2 h macro ceiling | Arithmetic checks: (1352.7 + 840.0 + 1572.8)/3 = 1255.2 s = 20.92 min; 21 × 20.92 = 439.3 min = 7.32 h. Matches results.json P4. |
| 3 | Framework-incomplete: `success_criteria=[]` ⇒ only KILLED valid | Verified `experiment get` output: "Success Criteria: NONE". SUPPORTED undefinable. |
| 4 | MMLU-Pro 14 disciplines ⇒ N=25 pigeonhole violation | Wang et al. 2024 enumerated 14 categories; `results.json.predictions[4].n_mmlu_pro_categories=14`. Valid. |
| 5 | Finding #8 non-transfer (Qwen2.5 QK-L2 ≠ Gemma 4 RMSNorm + QK-pre-projection-norm) | Architectural difference is real per `MLX_GEMMA4_GUIDE.md`; Finding #8 mechanism does not lift. Valid. |

Theorem stack is **defense-in-depth**: even if any single theorem were waived, the remaining four still close K1600 independently.

---

## Precedent alignment

Matches the preemptive-kill pattern established by `exp_g4_25domain_real_hf` (KILLED_PREEMPTIVE, 2026-04-18) for the same cohort. This is the **first post-T2.1-V3 cascade-drain instance** of the pattern — confirms cascade unblock (3 adapters delivered) is *necessary but not sufficient* for any cohort member with KC fixing N≥4 on Gemma 4 domain specialists.

---

## Routing implications for analyst / downstream drain

1. **Cohort-filter forecast:** the following cohort members have "N=25" or similar N>4 lock-in and will preemptive-kill under the same Theorem-1 structure:
   - `exp_g4_1overN_correct_delta`
   - `exp_g4_relevance_weighted_n25`
   - `exp_g4_vproj_compose_n25_clean`
   - `exp_g4_tfidf_ridge_n25_clean`
   - Any cohort title containing `n25` / `n14` (if > 3 Gemma 4 specialists demanded).
2. **LEARNINGS axis:** partial cascade unblock satisfies ap-017 for N≤3 but not for cohort N≥4; downstream preemptive-kill is the correct drain path, not retry.
3. **Non-cohort drain candidates:** any open P≤2 experiment that does NOT demand N>3 Gemma 4 domain specialists is now unblocked and should be preferred for actual runs.

---

## Assumptions / judgment calls

- **T2.1 adapter-inventory as ground truth.** Reviewer verified 3 adapter dirs present; did not validate safetensor integrity (would require loading). Researcher iter-15 did set T2.1 V3 `all_pass=true` and the cascade-drain precedent is explicit.
- **"Universal thinking adapter" (`adapters/thinking-openthoughts-universal-v0/adapters.safetensors`) not counted as a K1600 domain specialist.** K1600's "0/25 drop on MMLU-Pro drift" reading requires per-domain specialists; a universal thinking adapter is a valid floor but not a subject-domain specialist. This is the researcher's reading and the reviewer concurs.

---

## Verdict

**KILL (confirm preemptive).** Experiment already marked `status=killed` in DB with K1600=fail and evidence logged. No REVISE pass needed — the preemptive-kill analysis is mathematically sound, the results artifacts are consistent, and the KC text is unmodified. Emit `review.killed` → analyst.
