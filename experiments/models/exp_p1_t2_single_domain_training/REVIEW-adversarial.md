# REVIEW-adversarial.md — T2.1: Single Domain Adapter Training on Gemma 4 E4B

## V3 Rerun Review (2026-04-19) — Verdict: **PROCEED**

Adversarial review of the V3 RERUN end-to-end pass (pueue task 0, 04:09:51 → 05:29:33,
~80 min wall on `.venv` Python 3.12 / datasets 4.3.0 / dill 0.4.0). Both audit-2026-04-17
fixes landed in code (KC text in DB unchanged): K1030 metric-swap MedMCQA→MedQA-USMLE-4-opt,
K1028 format-artifact `max_tokens` 256→1024.

### Adversarial checklist (a)–(s)

- (a) `results.json["verdict"]="supported"` matches DB `status=supported` ✓
- (b) `all_pass=true`; all 5 K-cells in `results.json` = `PASS` ✓
- (c) PAPER.md verdict line = "**SUPPORTED**" (V3 section); no PROVISIONAL/PARTIALLY etc. ✓
- (d) `is_smoke=false`, `n_eval=50`, `n_train=2000`, `n_steps=1000` — full run ✓
- (e) **KC integrity:** DB KC text for K1028–K1032 unchanged. MATH.md §"Audit-2026-04-17 Reconciliation" documents the divergence as a documentation/code bug aligned to canonical DB text — not a relaxation. K1030 threshold (≥3pp) and K1028 threshold (≥5pp) both unchanged ✓
- (f) No tautology — accuracy deltas measured by independent generate→regex extraction; bytes/wall measured at file-system / wall-clock layer ✓
- (g) Code↔DB reconciled: `run_experiment.py:121` trains on `GBaker/MedQA-USMLE-4-options`, `eval_medqa` evaluates the matching test split, K1030 dict key renamed `K1030_med_medqa` ✓
- (h) No `sum(lora_A)` / weighted-add composition — single-domain ✓
- (i) `lora_config.scale=6.0` — safe (< 12) per Findings #328/#330 ✓
- (j) No routing ✓
- (k) No `shutil.copy` of sibling adapters; each domain has its own 1000-step training ✓
- (l) No hardcoded `{"pass": True}` — all five K cells derived from measured deltas ✓
- (m) MATH.md target = Gemma 4 E4B; `MODEL_ID="mlx-community/gemma-4-e4b-it-4bit"` ✓
- (m2) MLX-idiomatic: `mx.set_memory_limit`, `mx.set_cache_limit`, `mx.clear_cache`, `mx.reset_peak_memory`; LoRA training delegated to `mlx_lm.lora` subprocess (canonical mlx-lm path per `reference_mlx_gemma4.md`) ✓
- (n) `base_gsm8k=50.0%` non-zero with `max_tokens=1024` — format-artifact cured. Adapter delta now measures capability, not format-adaptation ✓
- (o) n_eval=50 per domain — borderline-acceptable; deltas (+22 to +62pp) exceed binomial 95% CI (~±14pp at 50%) ✓
- (p) No synthetic padding — three real domain datasets ✓
- (q) Base numbers measured in-run by the same script that trains/evaluates adapters ✓
- (r) PAPER.md prediction-vs-measurement table present (V3 RERUN section) ✓
- (s) Theorem 1 (≤50MB) verified at 10 MB/domain (5× margin); Theorem 2 (<1 GPU-hr) holds at 14–26 min/domain even though point-prediction was 8–15× off; Theorem 3 (≥+5pp) verified at +22/+48/+62pp ✓

### Independent on-disk verification

- `adapters/{math,code,medical}/adapters.safetensors` — 4,999,229 bytes each, mtime within the pueue run window ✓
- `adapters/{math,code,medical}/0001000_adapters.safetensors` 1000-step checkpoint present ✓
- `adapter_config.json` per domain ✓
- DB `experiment get` confirms `Status: supported`; latest evidence line dated 2026-04-19 [pass] ✓

### Caveat noted (non-blocking — robust under conservative discount)

- **K1030 MedQA base = 6.0% ≪ random (25%)** is a refusal/format artifact (PAPER.md Finding 3). Even discounting 19pp to format (= 25% − 6%), residual capability gain is +43pp ≫ +3pp threshold (14× margin). PAPER.md instructs downstream cites to use 25% base-floor — sufficient disclosure.

### Routing signal for analyst

- **First end-to-end SUPPORTED in this drain session.** V3 unblocks the 13 entries in `Blocks:` plus the 17-member `audit-2026-04-17` cohort §P probes plus `exp_p9_benchmark_showdown` — re-claim should now auto-PASS the precondition probes.
- **Two cluster-level wins worth memorizing:** (i) the .venv-vs-system-python3 misdiagnosis (HALT iter-13 wrongly attributed blocker to Py3.14 incompat — actual cause was `#!/usr/bin/env python3` shebang vs `.venv/bin/python`); (ii) the format-artifact remediation (max_tokens=256 → 1024) doubles the GSM8K base measurement and changes how all downstream "math capability" claims should be computed. Both candidates for new `mem-antipattern` entries.

**Verdict: PROCEED.** DB already at `status=supported`. Finding-add for V3 will be issued on emit.

---

## V2 Audit Review (2026-04-18) — Verdict: **KILL (confirm)** (superseded by V3 PROCEED above)

Adversarial re-review of the audit-2026-04-17-rerun finalize-only pass. Researcher reconstructed `results.json` from the 2026-04-09 PAPER.md numbers, prepended V2 audit section to PAPER.md, and rewrote LEARNINGS.md flipping Finding #421 from Supported → KILLED. No rerun (datasets/dill Py3.14 incompat + missing adapter safetensors — unverifiable). DB already reflects `status=killed` with 2026-04-18 fail evidence line.

### Adversarial checklist (a)–(s)

- (a) results.json `verdict=KILLED` matches DB `status=killed` ✓
- (b) `all_pass=false`, K1030 FAIL recorded in KC dict ✓
- (c) PAPER.md verdict line = KILLED ✓
- (d) `is_smoke=false`, N_eval=50 / N_train=2000 / steps=1000 full-run ✓
- (e) `git diff HEAD -- MATH.md` is empty — no KC relaxation after pre-reg ✓
- (f) KCs are accuracy deltas and bytes/minutes bounds; no tautology ✓
- (g) **K1030 code↔DB divergence IS the kill reason.** DB K1030 text = "MedQA"; `run_experiment.py:351-412` measures `openlifescienceai/medmcqa`. Metric-swap acknowledged and drives the FAIL — properly handled, not hidden ✓
- (h) No `sum(lora_A)` / buggy composition — single-domain experiment ✓
- (i) `run_experiment.py:172` `scale=6.0` — safe (< 12) ✓
- (j) No routing ✓
- (k) No `shutil.copy` of sibling adapters ✓
- (l) No hardcoded `{"pass": True}` in KC dict — all derived from measured deltas ✓
- (m) MATH.md target = Gemma 4 E4B; `run_experiment.py:32` `MODEL_ID="mlx-community/gemma-4-e4b-it-4bit"` — match ✓
- (m2) MLX skill evidence: uses `mlx_lm.load` + `mlx_lm.lora` per `reference_mlx_gemma4.md`; idiomatic for the HF LoRA trainer path ✓
- (n) base_gsm8k=0% format-artifact acknowledged in PAPER.md §"⚠️ Note on base GSM8K" and V2 audit block. K1028 conservatively kept PASS since 30-50pp true gain >> 5pp threshold — honest disclosure ✓
- (o) n_eval=50 per domain — borderline but consistent with prior micro policy ✓ (non-blocking)
- (p) No synthetic padding ✓
- (q) Base numbers are measured in-run, not cited ✓
- (r) Prediction-vs-measurement table present ✓
- (s) No unsupported math claims — Theorem 1 verified (1,247,232 vs 1,290,240 predicted, 3.3% off), Theorem 2 acknowledged 7.8× off on quantitative prediction but claim (<1 GPU-hr) survives, Theorem 3 conservative ✓

### Assumptions / judgment calls

1. **Missing adapter safetensors accepted for KILL path.** The researcher could not rerun (Py3.14 datasets/dill incompat). For a `supported` claim this would block, but for a `killed` claim the reconstruction is fine — PAPER.md evidence string on DB already matched the 2026-04-09 measurement. Reconstruction provenance is explicit in `results.json._reconstruction_note`.
2. **K1028 kept as PASS despite format-artifact.** Conservative: 30-50pp estimated true gain > 5pp threshold by >6×. Analyst may want to propagate this as a conditional — if future audit tightens the estimator to ≤ 5pp, K1028 flips.
3. **LEARNINGS.md already written by researcher.** Minor hat-order deviation; content is correct and covers literature context. Analyst pass can refine cluster-level propagation rather than re-author.

### Downstream propagation (routing signal for analyst)

- **6th metric-swap instance this loop** (cluster with `m2p_moe_routing` V2, `m2p_hard_moe`, `m2p_composition_n5` V2, `exp_p1_t1_householder` V2 sentinel bug, `exp_bench_*` today). At 6 instances, a new `mem-antipattern-db-kc-text-divergence` entry is warranted — analyst should add.
- **T2.1 cited by**: T2.5 SFT-residual, T2.6 5-domain, T4.3 vLLM, `exp_model_peer_comparison_*`, `exp_model_mtbench_composed`. Status revised: proven for Math/Code, **unproven for Medical**. Siblings load-bearing on medical specialization must either re-run medical on actual MedQA or drop medical dimension.
- **Finding #421 status**: Supported (2026-04-09) → KILLED (2026-04-18). LEARNINGS.md already reflects this.

### Non-blocking observations

- Evaluation pipeline (`eval_gsm8k`, `eval_humaneval`, `eval_medmcqa`) should be refactored into a shared module with a unified answer-extraction protocol to prevent future format-artifact bugs. Out of scope for this audit.
- `adapter_config.json` stub preservation (without safetensors) is a half-state that confuses reproducibility audits. Future policy: either commit full artifact or delete the stub.

**Verdict: KILL confirmed.** DB already reflects this. Route to analyst for cluster-level mem-antipattern synthesis and sibling propagation. No finding-add needed — already issued on 2026-04-09 (Supported→KILLED flip) and 2026-04-18 (metric-swap) via evidence lines.
