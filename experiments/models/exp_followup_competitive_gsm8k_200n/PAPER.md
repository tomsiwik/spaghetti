# PAPER: exp_followup_competitive_gsm8k_200n

**Verdict:** KILLED (preemptive, cascade / antipattern-017 — 6th confirmed instance).
**KC:** K1575 FAIL by construction (routed composition degenerates to base when
all 5 domain adapters are config-only stubs on disk).
**Date:** 2026-04-18.
**Smoke flag:** false.
**All pass:** false.

---

## 1. tl;dr

This followup was motivated by the kill of `exp_competitive_benchmark_routed` (which
was `status=killed` 2026-04-17 on K640 — routed worse than base on 2/6 benchmarks at
n=20). The followup's premise: "rerun with n≥100 per subject + fixed extraction —
original kill rested on n=20 noise."

Pre-flight adapter inventory shows **0 of 5 required domain adapters have weight files
on disk**. All 5 registry-referenced paths contain only `adapter_config.json`. By
MATH.md Theorems 1 & 2, `E[routed − base] = 0` regardless of n; K1575 ("CI excludes
zero") is structurally unreachable.

No base model loaded, no evaluation run. Verdict set in `run_experiment.py` pre-flight.

## 2. Why we did not run

Three independent conditions block the measurement:

1. **Antipattern-017 (6th instance, 2 days).** All 5 domain adapters
   (`math`, `code`, `medical`, `legal`, `finance`) referenced in `adapters/registry.json`
   point to paths containing only `adapter_config.json`. Prior instances in chain:
   (1) exp_p11_baseline_eval, (2) M0 full_pipeline_v2, (3) J0 adapter_composition_thinking,
   (4) followup_composition_correct_delta (5/5), (5) followup_routing_multi_sample_ppl (5/5),
   (6) this (5/5).
2. **Source dir gone.** The original script this followup is based on
   (`micro/models/competitive_benchmark_routed/run_experiment.py:39`) reads adapters from
   `micro/models/real_data_domain_experts/adapters/`. That subdirectory **does not exist**
   on disk.
3. **Antipattern-020 (cascade).** This followup re-tests a killed experiment. Even if
   adapter weights were present, F#553 already shows per-sample routing breaks the
   oracle-routing identity that F#237 (`GSM8K +10pp`) depended on; the "-20pp at n=20"
   fail pattern was directionally consistent across K1 evidence at both 2026-03-31 and
   2026-04-17 (replicated, not noise).

## 3. Prediction vs measurement table (pre-registered from MATH.md)

| # | Prediction | Measurement | Result |
|---|-----------|-------------|--------|
| 1 | K1575 FAIL: CI[routed − base] centered at 0 when all adapters are stubs | Not measured — 0/5 weights on disk | FAIL (by construction) |
| 2 | Pr[any adapter loads with weight tensors] = 0 | 0/5 weight files found | PASS (prediction held) |
| 3 | `||delta_W|| = 0` in pure-stub case | Not computed (no forward pass) | N/A (Thm 2) |

## 4. Dependency state table

| Adapter | Registry path | Weights present | Bytes |
|---|---|---|---|
| math-gsm8k-knowledge-v0 | `micro/models/exp_p1_t2_single_domain_training/adapters/math/` | ❌ | 0 |
| code-codealpaca-knowledge-v0 | `micro/models/exp_p1_t2_single_domain_training/adapters/code/` | ❌ | 0 |
| medical-medmcqa-knowledge-v0 | `micro/models/exp_p1_t2_single_domain_training/adapters/medical/` | ❌ | 0 |
| legal-mmlu-knowledge-v0 | `micro/models/exp_p1_t2_multi_domain_5/adapters/legal/` | ❌ | 0 |
| finance-mmlu-knowledge-v0 | `micro/models/exp_p1_t2_multi_domain_5/adapters/finance/` | ❌ | 0 |

Source-experiment dir `micro/models/real_data_domain_experts/adapters/`: **does not exist**.

## 5. Antipattern self-check

| Antipattern | Triggered | Evidence |
|---|---|---|
| 017 (stub adapter) | ✅ 6th instance | `ls adapters/*/` — all config-only |
| 020 (cascade design) | ✅ | parent `exp_competitive_benchmark_routed` killed; this is retest |
| 018 (channel-tokens-as-SFT) | ❌ | inference-only, no SFT |
| 008 (strip_thinking) | ❌ | no thinking path |
| 003 (LORA_SCALE) | ❌ | preemptive; no scaling decision reached |
| KC-swap | ❌ | MATH.md fresh (no prior version); single KC ID 1575 |

## 6. Unblock path for followup-v2

`P11.ADAPTER-REBUILD` (same unblock as M0, L0, J0, V6, other cascade kills):

1. Retrain the 5 domain knowledge adapters into their registry paths.
2. Verify `adapters.safetensors` exists with `st_size > 0` for each.
3. Re-plan the followup with:
   - Pre-flight `for a in domains: assert (a/'adapters.safetensors').stat().st_size > 0`.
   - Baseline Gemma 4 MMLU-Pro from F#560 (40.7% measured @ n=1400), not F#530 stale 62.1%.
   - Explicit routing mechanism for blind eval (per-sample, not oracle — see F#553).
   - Noise analysis: at n=100/subject × 14 subjects = 1400 samples per condition, the
     MDE at 95% CI for proportion diff is ~3.6pp (std `sqrt(p(1-p)/n) ≈ 0.013` for
     p≈0.40, ×1.96 ×√2). Claims below 3.6pp are not detectable even with adapters.

## 7. Assumptions

- A1 (load behavior): `mlx_lm.load(model, adapter_path=stub_dir)` either crashes or
  silently runs base. MATH.md Thm 2 handles all cases without needing to empirically
  distinguish them.
- A2 (routing identity): F#553's "per-sample routing ≠ oracle routing at p<1" applies
  if adapters existed — independent argument reinforcing kill. Not load-bearing for
  the primary driver (Thm 1).
- A3 (n≥100 premise): the followup's n≥100 claim is irrelevant because the premise
  ("adapters that differ from base") is false.

## 8. Salvageable content (for v2)

- **Noise-analysis formula** in §6: MDE ≈ 3.6pp at n=1400 for Gemma 4 MMLU-Pro.
  Any future KC on routed-vs-base MMLU-Pro should set threshold ≥ 3.6pp or use
  per-subject n > 100.
- **Registry pre-flight check script** (`check_adapter_weights` in run_experiment.py)
  is reusable as a standard pre-flight across all composition experiments — worth
  promoting into a shared utility.

## 9. References

- Finding #237 — GSM8K +10pp under oracle routing (now blocked by F#553 for blind routing).
- Finding #553 — Per-sample routing artifact.
- Finding #560 — Gemma 4 base MMLU-Pro = 40.7% measured.
- Finding #517 — Knowledge adapters degrade MCQ.
- Antipattern-017 — Weight-less adapter stub.
- Antipattern-020 — Cascade-dependent experimental design.
- Parent killed: `exp_competitive_benchmark_routed` (K640 FAIL: -20pp math @ n=20).

## 10. Handoff

- **Reviewer:** verify (a) all 6 artifacts present, (b) 5/5 stub adapters on disk, (c) DB
  completion `--k 1575:fail`, (d) MATH.md single commit (no KC-swap), (e) noise formula
  §6 is algebraically correct.
- **Analyst:** promote antipattern-017 instance count from 5 → 6 in antipatterns doc.
  Consider promoting §6 MDE formula to a new finding ("at n=1400 MMLU-Pro MDE ≈ 3.6pp at
  95% CI — sets floor for MMLU-Pro claim thresholds").
- **Next researcher:** two more `audit-2026-04-17` P=1 candidates in backlog? Use
  `experiment query --tags audit-2026-04-17` to list. Expect cascade on any with domain
  adapters.
