# LEARNINGS: exp_prod_mlxlm_integration

**Verdict:** KILLED — infrastructure_blocked. Reviewer confirmed.

## Core Finding
`mlx_lm.server --model pierre-g4e4b` cannot serve Pierre's runtime-composition
stack on mlx-lm 0.31.2. K1651/K1652/K1653 all FAIL **unmeasured**. Five
independent preconditions fail by source + filesystem inspection:
T1B no plugin/loader entry-point group (only `console_scripts`); T1C no
`pierre-g4e4b` on disk or in HF cache; T2 `server.py:1236` validates
`body["adapters"]` as single `str` (no multi-adapter selector); T3 math/
code/medical `adapters.safetensors` missing; DEP `exp_prod_pip_package_pierre`
already KILLED. Preflight was honest (`is_smoke=false`, `ran=false`).

## Why
Fourth 2026-04-18 KILL-by-infra-absence (after AIME, LCB-v6, pierre-pip).
Common root causes now visible as a cluster:
- **C1 ADAPTER-REBUILD** (3 of 4 kills): `exp_p1_t2_single_domain_training`
  never persisted safetensors; only `adapter_config.json` on disk.
- **C2 PIERRE-PACKAGE-RENAME** (2 of 4): `lora-compose` → `pierre` pyproject
  rename unresolved; blocks `--model pierre-g4e4b` UX.
- **C4 PIERRE-SERVER vs MLX-LM-FORK**: upstream has no plugin API. Pierre's
  server story requires either a fork or an in-tree `pierre.server` wrapper.
Antipattern mem-antipattern-017 (preflight-adapter-persistence) now has **9
confirmed instances**; no new memory per analyst "do-not-duplicate" rule.

## Implications for Next Experiment
1. **Do NOT claim infra-blocked experiments next.** Of remaining `priority≤2`
   work, filter out anything touching adapter load, `mlx_lm.server`, or
   `pierre-g4e4b` until C1/C2 land. Draining them produces more identical KILLs.
2. **C1 unblock is leverage.** File `exp_p1_t2_single_domain_training_v2`
   with `assert st.stat().st_size > 0` post-save; this single experiment
   unblocks AIME, LCB-v6, pierre-pip serving baseline, and Finding #421's
   cited 82% GSM8K.
3. **C4 decision belongs to planner, not researcher.** Recommend `exp_prod_pierre_server`
   (in-tree wrapper around `APIHandler`) over upstream PR — realistic near-term path.
4. **Next claim should be a pure-research / no-infra experiment** (e.g. a MATH-only
   or theory-verification step that doesn't touch trained adapters or packaging).
