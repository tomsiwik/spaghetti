# PAPER.md — Per-token top-2 routing on Gemma 4 N=25

## Verdict
**KILLED** (precondition-probe, pre-registered in MATH.md §Preconditions).

## One-line
K1578 (`routed_PPL < 0.95 · exclusive_PPL on 5 domains`) is unmeasurable on
Gemma 4 E4B 4-bit at N=25: P1 (adapter weights) + P2 (per-token router) + P3
(exclusive_PPL baseline) all FAIL. Heavy Gemma 4 training (~4h MLX) is
skipped per the cohort's standing rule (7th precondition-probe KILL this loop).

## Prediction vs measurement

| Claim / criterion | Predicted (MATH.md) | Measured (results.json) | Status |
|---|---|---|---|
| **P1** Gemma 4 adapter weights ≥5 present | Upstream T2.1 regenerates math/code/medical `.safetensors`; 2 extra domains trained | T2.1 verdict = KILLED; `t21_domain_weights_present=[]`; `n25_stub_dir_weights_count=0`; 0/5 weights available | **FAIL** |
| **P2** Gemma 4 per-token router artifact | `exp_g4_hidden_state_probe/router.safetensors` or equivalent | `probe_dir_exists=false`; no Gemma 4 port of Finding #310 exists | **FAIL** |
| **P3** Exclusive TF-IDF PPL baseline on Gemma 4 N=25 | Upstream `exp_g4_tfidf_ridge_n25_clean` / `exp_p1_t4_tfidf_routing_gemma4` reports `exclusive_ppl` | Only `exp_p1_t4_tfidf_routing_gemma4/results.json` exists with verdict=KILLED; no `exclusive_ppl` field in any upstream | **FAIL** |
| **K1578** `routed_PPL < 0.95 · exclusive_PPL` on 5/5 domains | Finding #58 mechanism says top-2 beats exclusive by ≥5% when overshoot-driven; N=25 routing accuracy ~85% provides headroom | Unmeasurable; 3/3 preconditions fail | **FAIL-unmeasurable** |

## Mechanism summary (from MATH.md)

Finding #58 on BitNet showed top-2 beats top-1 **only** under overshoot
conditions: when individual adapters degrade base PPL on their own domain
(4/15 adapters in Finding #58), mixing two adapters dilutes the overshoot.
Finding #305 Theorem C (shared-KV null result) forbids substituting
TF-IDF per-sequence as the "per-token" mechanism — that collapses the test
to exclusive-vs-exclusive.

The claim's port to Gemma 4 requires **three independent upstream artifacts**.
All three are either KILLED (T2.1 adapter training, TF-IDF baseline) or
non-existent (Gemma 4 hidden-state probe). Filling the gaps synthetically
(random-init adapters, TF-IDF substituted for per-token) invalidates the
test — synthesis reclassifies the run as one of the existing KILLED setups.

## Kill-criteria pre-registration (git-verified)

- `git log --oneline -- MATH.md` shows **single commit** pre-registering K1578.
- No KC text was modified between pre-registration and completion.
- No KC threshold was relaxed (0.95× bound inherited verbatim from DB KC text).
- Precondition routing P1/P2/P3 was declared **before** data collection; the
  unmeasurable branch is a pre-registered outcome, not a post-hoc downgrade.

## Verdict-consistency pre-flight

| Check | Value | OK? |
|---|---|---|
| `results.json["verdict"]` | `"KILLED"` | consistent |
| `results.json["all_pass"]` | `false` | consistent |
| PAPER.md verdict line | `KILLED` | consistent |
| `is_smoke` | `false` | consistent |
| KC modified post-reg | no (single commit) | consistent |
| Antipattern match | none (precondition probe, not synthesis) | consistent |

## Assumptions (documented, not verified)

- MLX-LM 0.31.0 would be the target if preconditions passed; not invoked in
  probe-only mode.
- Finding #586 LORA_SCALE safety bound (=5) is the correct scale for the
  unblock-path adapter retraining (inherited from parent experiment's
  post-audit recommendation).
- 0.95× threshold on 5/5 domains is a strict point-estimate bar; noise
  bounds from Finding #310 suggest ~2.5σ per-domain — the bar is honest,
  not artificially tight.

## Unblock path (for future resurrection)

1. Rerun `exp_p1_t2_single_domain_training` at LORA_SCALE=5 → regenerates
   math/code/medical adapters.
2. Train 2 additional Gemma 4 adapters (`finance`, `legal`) at matched recipe
   on disjoint corpora.
3. Train a per-token ridge router on Gemma 4 hidden states (Finding #310
   recipe, target ≥95% token accuracy).
4. Re-run this probe; on all-PASS, implement the PPL measurement loop in the
   `all_preconditions_pass` branch of `run_experiment.py`.

## Cohort context

This is the **7th** precondition-probe KILL in the audit-2026-04-17 cohort:

1. `exp_followup_sft_behavioral_lora_scale_5` (#600) — QR ≥ 0.90
2. `exp_followup_ss_rn_path_valid_sft` (#602) — |Δacc| ≤ 5pp
3. `exp_followup_orthogonal_projection_scale_control` — theoretical refutation
4. `exp_followup_answer_conditioned_ppl` (#603) — K1567 measured-KILL
5. `exp_followup_format_compat_peft_required` (#604) — SUPPORTED (the break)
6. (sixth instance referenced in current_direction.md)
7. **`exp_g4_per_token_top2_routing`** (this experiment)

The cohort pattern: audit-rerun experiments whose KCs depend on upstream
Gemma 4 adapter training route to KILLED-unmeasurable until upstream T2.1
is regenerated. The cohort stalls on a **single** upstream fix
(`exp_p1_t2_single_domain_training` at LORA_SCALE=5 with disjoint corpora),
which unblocks P1 for most downstream experiments.
