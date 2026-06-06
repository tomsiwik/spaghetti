# PAPER: exp_g4_routed_beats_base_think

**Verdict:** KILLED (preemptive, cascade / antipattern-017 — **7th confirmed instance**).
**KC:** K1592 FAIL by construction (routed composition degenerates to base when all
5 domain adapters are config-only stubs on disk; thinking mode does not rescue).
**Date:** 2026-04-18.
**Smoke flag:** false.
**All pass:** false.

---

## 1. tl;dr

This audit-tagged experiment (`audit-2026-04-17/composition-bug/g4-gemma4`) asks:
*does routed 5-adapter composition beat Gemma 4 base on GSM8K by ≥5pp without
degrading MMLU-Pro, under thinking mode?* It is a product-story variant of the
killed `exp_competitive_benchmark_routed` (F#236, status=killed 2026-04-17 on K640).

Pre-flight adapter inventory shows **0 of 5 required domain adapters have weight
files on disk**. All 5 registry-referenced paths contain only `adapter_config.json`.
By MATH.md Theorems 1–3, `y_routed ≡ y_base` regardless of input, sample size,
routing strategy, or thinking/non-thinking mode. The "GSM8K +5pp AND MMLU-Pro ≥ base"
conjunction is structurally unreachable: its first conjunct requires a non-zero
effect that cannot exist.

No base model loaded, no evaluation run. Verdict set in `run_experiment.py` pre-flight.

## 2. Why we did not run

Three independent conditions block the measurement:

1. **Antipattern-017 (7th instance, 2 days).** All 5 domain adapters (`math`,
   `code`, `medical`, `legal`, `finance`) referenced in `adapters/registry.json`
   point to paths containing only `adapter_config.json`. Prior instances (all this week):
   (1) `exp_p11_baseline_eval`, (2) `exp_p11_full_pipeline_v2` (M0), (3) `exp_p11_adapter_composition_thinking` (J0, 4-of-4),
   (4) `exp_followup_composition_correct_delta` (5/5), (5) `exp_followup_routing_multi_sample_ppl` (5/5),
   (6) `exp_followup_competitive_gsm8k_200n` (5/5), (7) this (5/5).

2. **Thinking mode is inert to absent adapter operators (Thm 3).** The
   experiment's novel claim versus F#236 is thinking mode enabled. Thinking mode
   is a prompt-level / decoding-length modification; it does not modify the
   adapter forward pass. With `B_i A_i = 0` for all i, the routed operator is the
   identity in either mode. There is no mechanism by which thinking mode can
   rescue a composition that has no adapter signal to compose.

3. **Antipattern-020 (cascade-dependent design).** The parent experiment this
   re-tests (`exp_competitive_benchmark_routed`) is `status=killed` on K640
   (routed -20pp on math at n=20, -10pp on legal). The kill was directionally
   consistent at n=20. Re-running the same composition under the same missing-adapter
   state (plus thinking mode) cannot change the structural outcome.

## 3. Prediction vs measurement table (pre-registered from MATH.md)

| # | Prediction | Measurement | Result |
|---|-----------|-------------|--------|
| 1 | K1592 FAIL: Δ_gsm8k ≥ +5pp AND Δ_mmlu_pro ≥ 0 collapses when adapters are stubs | Not measured — 0/5 weights on disk | FAIL (by construction) |
| 2 | Pr[any of 5 adapters loads with weight tensors] = 0 | 0/5 weight files found | PASS (prediction held) |
| 3 | `‖Σ B_i A_i‖_F` = 0 in pure-stub case | Not computed (no forward pass) | N/A (Thm 2) |
| 4 | Thinking mode provides no rescue when adapter ops vanish | Thm 3: algebraic | PASS (prediction held) |

## 4. Dependency state table

| Adapter | Registry path | `adapters.safetensors`? | Bytes |
|---|---|---|---|
| math-gsm8k-knowledge-v0 | `micro/models/exp_p1_t2_single_domain_training/adapters/math/` | ❌ | 0 |
| code-codealpaca-knowledge-v0 | `micro/models/exp_p1_t2_single_domain_training/adapters/code/` | ❌ | 0 |
| medical-medmcqa-knowledge-v0 | `micro/models/exp_p1_t2_single_domain_training/adapters/medical/` | ❌ | 0 |
| legal-mmlu-knowledge-v0 | `micro/models/exp_p1_t2_multi_domain_5/adapters/legal/` | ❌ | 0 |
| finance-mmlu-knowledge-v0 | `micro/models/exp_p1_t2_multi_domain_5/adapters/finance/` | ❌ | 0 |

Parent-repo shim dirs `adapters/{math,bash,python,sql,medical}/` are also stubs.
Only `adapters/thinking-openthoughts-universal-v0/` has real weights (151 MB),
but that is a universal thinking adapter — not one of the 5 domain experts
K1592 routes over.

## 5. Antipattern self-check

| Antipattern | Triggered | Evidence |
|---|---|---|
| 017 (stub adapter) | ✅ **7th instance** | `check_adapter_weights()` — all 5 config-only |
| 020 (cascade design) | ✅ | parent `exp_competitive_benchmark_routed` killed on K640; this is retest with thinking added |
| 018 (channel-tokens-as-SFT) | ❌ | inference-only, no SFT |
| 008 (strip_thinking) | ❌ | no thinking-strip path; preemptive kill |
| 003 (LORA_SCALE) | ❌ | preemptive; no scaling decision reached |
| KC-swap | ❌ | MATH.md fresh file in fresh directory; single commit |

## 6. Unblock path for a v2

`P11.ADAPTER-REBUILD` (same unblock as M0, L0, J0, and followups 1–3):

1. Retrain the 5 domain knowledge adapters into their registry paths.
2. Verify `adapters.safetensors` exists with `st_size > 0` for each.
3. Re-plan v2 with:
   - Pre-flight assertion `for name in [math, code, medical, legal, finance]:
     assert (registry_path/'adapters.safetensors').stat().st_size > 0`.
   - Baseline from F#560 (Gemma 4 MMLU-Pro 40.7% @ n=1400), not older stale values.
   - Blind routing (per-sample, not oracle) — F#553 shows oracle routing inflates
     favorable comparisons.
   - Noise floor: MDE ≈ 3.6pp at 95% CI for n=1400 MMLU-Pro samples
     (`1.96 × √2 × √(p(1-p)/n) ≈ 0.036` at p=0.40, n=1400). A +5pp GSM8K claim
     at n=1319 test questions has MDE ≈ 3.7pp → detectable if real.
   - Document mechanism by which thinking mode would meaningfully interact with
     the 5 trained domain adapters (e.g., adapter provides domain-specific facts
     that thinking chain uses) — otherwise thinking mode adds no causal leverage.

## 7. Assumptions

- **A1 (load behavior):** `mlx_lm.load(model, adapter_path=stub_dir)` either
  crashes or silently runs base. MATH.md Thm 2 handles all cases without needing
  to empirically distinguish.
- **A2 (routing choice):** the experiment did not specify oracle vs blind; F#553
  and F#236 indicate blind underperforms oracle, so a v2 must be pre-registered
  on blind routing to avoid re-creating F#237's artifact.
- **A3 (thinking mode mechanism):** the experiment cites no mechanism for why
  thinking rescues composition in the kill literature — the hypothesis is
  implicit in the title ("the product-story test"). Thm 3 formalises the absence
  of that mechanism under stub adapters.

## 8. Salvageable content (for v2)

- Pre-flight `check_adapter_weights()` helper in `run_experiment.py` — same shape
  as followup_competitive_gsm8k_200n; reusable as shared utility in a future
  `micro/models/_shared/preflight.py`.
- Noise-floor formula: MDE ≈ 3.6pp at n=1400 MMLU-Pro, ≈ 3.7pp at n=1319 GSM8K.
  Relevant for sizing v2 claim thresholds.
- Thm 3 statement (prompt-level changes inert to absent adapter ops) is generic
  across any future "mode X rescues stub composition" hypothesis.

## 9. References

- **Finding #236** — `exp_competitive_benchmark_routed` routed -20pp math @ n=20.
- **Finding #237** — GSM8K +10pp oracle routing; broken for blind (F#553).
- **Finding #553** — per-sample routing artifact.
- **Finding #517** — knowledge adapters degrade MCQ (relevant to MMLU-Pro parity
  clause of K1592 under a non-stub replay).
- **Finding #560** — Gemma 4 base MMLU-Pro = 40.7% measured @ n=1400.
- **Antipattern-017** — weight-less adapter stub (7 instances now).
- **Antipattern-020** — cascade-dependent experimental design.
- **Parent killed:** `exp_competitive_benchmark_routed` (K640 FAIL).

## 10. Handoff

- **Reviewer:** verify (a) all required artifacts present (MATH, run_experiment, results, PAPER),
  (b) 5/5 stub adapters on disk via `ls` on each registry path, (c) DB completion `--k 1592:fail`,
  (d) MATH.md single commit (no KC-swap), (e) Thm 3 statement is correct (thinking mode does not
  interact with adapter operator), (f) antipattern-017 instance count (7) aligns with prior kills.
- **Analyst:** promote antipattern-017 instance count from 6 → **7** in antipatterns doc.
  Current scratchpad count was 6 after `followup_competitive_gsm8k_200n`; this is the 7th.
  Also: this is the **first "thinking-mode adds no rescue" formal theorem** — Thm 3 could be
  promoted to a generic finding applicable to future "mode X rescues stub composition" followups.
- **Next researcher:** remaining `audit-2026-04-17` P=1 candidates in backlog (query `--tags
  audit-2026-04-17`). Any with `routed`, `composition`, `5-adapter`, `N=25`, or `domain-expert`
  in the title will cascade on the same missing-weight issue. Consider batching a single
  `P11.HARNESS` ticket to unblock the entire audit tag family.
