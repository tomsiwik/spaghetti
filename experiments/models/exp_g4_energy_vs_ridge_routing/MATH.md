# MATH — exp_g4_energy_vs_ridge_routing (precondition probe)

## Type
Precondition probe — verifies upstream artifacts exist before running the target comparison.

## Target claim (would-be KC)
K1588: On Gemma 4 E4B with N=25 trained adapters, ridge-routing accuracy exceeds
energy-gap-routing accuracy by ≥ 10 pp on a held-out MMLU-Pro subset.

## Why a probe (not the full experiment)
Audit-2026-04-17 cohort has produced 9 consecutive precondition-probe KILLs, all
blocked on the same upstream artifact: `exp_p1_t2_single_domain_training` at
LORA_SCALE=5 with ≥5 disjoint domains (math/code/medical/finance/legal), all at
`max_tokens ≥ 512`. Running the full N=25 energy-vs-ridge compare without those
weights produces either synthetic results or zero-delta artifacts (see
Findings #605, #606, #608, #610, #611, #612, #613, #615, #616).

Cohort standing rule (from analyst learning.complete after the 9th KILL): do not
run heavy MLX work until the upstream retrain lands. Probe preconditions instead
and record a KILL on the pre-registered P1/P2/P3 structure.

## Preconditions (pre-registered)
- **P1**: `micro/models/exp_p1_t2_single_domain_training/adapters/<domain>/adapters.safetensors`
  exists for ≥ 25 domains (or equivalent N=25 Gemma 4 adapter set on disk).
- **P2**: a baseline ridge-routing head trained on those adapters' embeddings
  exists (artifact or evidence in a supported experiment).
- **P3**: an energy-gap scoring function over those same embeddings with a
  reference AUC measured on Gemma 4 (not a different base) exists.

## Kill rule (pre-registered, no post-hoc editing)
K1588 is UNMEASURABLE → KILLED if any of P1/P2/P3 fail. Probe output determines
the verdict; no scalar comparison is attempted if preconditions fail.

## Cited prior math / findings
- Finding #605/#606/#608/#610/#611/#612/#613/#615/#616 — identical blocking
  structure on this cohort's upstream.
- Finding #182 (`exp_self_embedding_quality_discriminator`) — the original
  energy-gap collapse motivation; does not by itself measure the Gemma 4 gap.
- PLAN.md Part 2, anti-pattern catalog entry "file-existence cache" and
  "KC measures wrong object" — both trigger when a probe is skipped and the
  code fabricates a routing comparison over non-existent adapters.

## Assumptions (Ralph autonomy, rule 1007)
- The claim system picked this cohort member despite the analyst's explicit
  learning.complete guidance ("pick OUT-OF-COHORT priority-≤2, no dep on
  exp_p1_t2_single_domain_training"). Since `experiment claim` is
  auto-ordered and there is no unclaim flag, the most defensible call is
  to probe + KILL with cited blocking and move on. Logged here for audit.

## Predicted probe result
P1 FAIL (only `adapter_config.json` stubs on disk at
`exp_p1_t2_single_domain_training/adapters/{math,code,medical}/` — 3 stubs, 0
weights, 0/25 coverage). P2 FAIL (no ridge head on Gemma 4 in DB). P3 FAIL (no
Gemma 4 energy-gap AUC measured). Verdict = KILLED per pre-registered rule.
