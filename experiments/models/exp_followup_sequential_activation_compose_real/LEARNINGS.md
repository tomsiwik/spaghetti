# LEARNINGS — exp_followup_sequential_activation_compose_real

**Status:** PROVISIONAL (K_vacate; Thm 1 structural PASS, behavioral KCs vacated)

## Core Finding

Weight-space sequential LoRA composition on Gemma 4 E4B `q_proj` is
architecturally infeasible: GQA forces `d_in=2560 != d_out=2048`, so
`ΔW_P @ ΔW_D` is non-computable as a weight-space cross-term. **Thm 1
PASS** (verified against the loaded personal adapter shape). Behavioral
KCs K1563a (pipeline style ≥24%) and K1563b (pipeline MCQ ≥15%) were
**vacated** honestly because the parent math adapter
(`exp_p1_t2_single_domain_training/adapters/math/adapters.safetensors`)
is not on disk — gitignored, not recoverable without retraining. Second
independent occurrence of the same infra blocker (sibling:
`exp_followup_hypernetwork_residual`).

## Why

- Thm 1's output is a *design rule*, not a one-off gotcha: any
  weight-space `(I + ΔW_P)(W + ΔW_D)` requires `d_in = d_out`. On
  Gemma 4 E4B, that cuts the candidate-projection set to `o_proj` (and
  hidden-matched MLP projections), not `q_proj`/`k_proj`/`v_proj`.
- The K_vacate was the correct call: smoke run at N=5, `is_smoke=true`,
  math adapter stub-only on disk — running the model pipeline would
  have produced noise, not signal. Reporting PROVISIONAL (mapped to DB
  `killed`) preserves honesty of the record.
- mem-antipattern-017 (adapter-weight preflight) is the governing
  systemic issue. This is instance N+1: the fix is upstream
  (training-side post-save assertion + un-gitignore trained
  artefacts), not in this experiment.

## Implications for Next Experiment

1. **Do not re-run this experiment until the parent math adapter is
   regenerated.** One rerun of `exp_p1_t2_single_domain_training` (at
   `LORA_SCALE ≤ 8`, antipattern-003 compliance) unblocks both this
   experiment *and* `exp_followup_hypernetwork_residual` — batch the
   infra fix; do not cascade-kill.
2. **Sibling `o_proj` experiment is the higher-value next step.**
   Pre-register `exp_sequential_activation_compose_oproj`: Thm 1' (on
   `o_proj`, `d_in = d_out = 2048`, cross-term IS computable) + paired
   math/personal adapters on `o_proj` only + KC of sequential beats
   additive by ≥5pp on style. This promotes the current result from
   "`q_proj` gotcha" to "design rule: sequential composition requires
   `d_in = d_out`".
3. **Model-level pipeline (cascade generation) is orthogonal.** Once
   the math adapter is back on disk, this run_experiment.py stands up
   as-is at full scale (`SMOKE_TEST=0`) — no code changes. Treat it as
   a separate question from weight-space composition.
