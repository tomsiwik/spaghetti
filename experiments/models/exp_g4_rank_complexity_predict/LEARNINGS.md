# LEARNINGS — exp_g4_rank_complexity_predict

## Core Finding
KC #1629 (Spearman ρ ≥ 0.85 between domain complexity and optimal rank r*) and
paired target KC #1629-T (behavioral gap ≤ 2.0pp vs r=12 oracle) are
**UNMEASURABLE** on current platform state. Probe: P1 FAIL (3/25 rank-sweep
safetensors — only math/code/medical at r=6), P2 FAIL (3/5 corpora — finance
and legal train.jsonl absent), P3 PASS (upstream `supported`,
`base_gsm8k_pct=50.0`). DB `killed`, Finding #681 filed. 10th consecutive
`audit-2026-04-17` downstream probe-KILL.

## Why
Computing Spearman on 3 domains × 1 rank is vacuous (n=3, no rank dimension);
KC discipline (PLAN.md §1) forbids retroactive sweep relaxation. The same
upstream (`exp_p1_t2_single_domain_training` lacking 22 of 25 adapters +
finance/legal corpora) has blocked every sibling in this cohort. F#666 target
pairing held — no proxy-only kill; both KCs UNMEASURABLE together on
preconditions, so KILL is on the blocker, not on Spearman-FAIL alone.

## Implications for Next Experiment
1. **Cohort standing-down is now hard precedent.** Memory
   `mem-antipattern-claim-time-cohort-saturation` already captures this
   (sources now include both rank_complexity_predict and snr_rank_predictor).
   Next researcher MUST `experiment query --tag audit-2026-04-17` before
   claiming; if ≥5 siblings killed on UNMEASURABLE with an un-landed upstream,
   skip the cohort entirely.
2. **Out-of-cohort candidates at P≤2:** `exp_hedgehog_*` (distillation),
   `exp_jepa_*` (latent-error routing), `exp_memento_*` (SFT replication),
   `exp_g4_adapter_class_composition_full` (reuses existing r=6 adapters),
   `exp_model_knowledge_gap_26b_base` (requires 26B download, orthogonal).
3. **Corpus-prep is separable** from 22-adapter retraining. A cheap data-prep
   task (harvest finance from FinQA-dev, legal from CaseHOLD-dev) could
   pre-compute `c(D)` across 5 domains independently — reducing v2 scope from
   ~12h to corpus-prep + rank-sweep training.
4. **MATH.md is canonical.** Do NOT file v2 on this hypothesis — same MATH.md
   re-runs once upstream rebuild supplies the 25-adapter endpoints. Editing
   MATH.md to relax the sweep would violate the locked-KC contract.
5. **Minor registration gap:** K#1629-T lives in MATH.md + results.json but
   not in DB's kill_criteria list. Non-blocking; flag for DB hygiene pass.
