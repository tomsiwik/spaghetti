# PAPER.md — exp_prod_onboarding_first_run

## Verdict

**KILLED_PREEMPTIVE** — PROD-deliverable-cascade (parent KILLED),
**3rd instance** (after F#740, F#741). Compound: F#666 proxy-only
KC-panel violation; F#502/F#646 schema-cohort 8th hit
(promotion candidate). All 5 theorems block (5/5 all_block=true,
defense_in_depth=true). Any one alone is sufficient.

## One-line

You cannot time `pip install pierre` → first inference when there
is no `pierre` package: parent `exp_prod_pip_package_pierre` is
KILLED with all three packaging KCs FAIL (pyproject name is
`lora-compose`, `pierre/` excluded from wheel, no published
artefact). All three child KCs are proxy-only and presuppose a
working install chain that does not exist.

## Target claim (DB)

| KC | Text |
|----|------|
| K1670 | `pip install` → first inference < 30 s on M5 Pro (cold cache OK, warm weights) |
| K1671 | Default bundle: base + 5 curated adapters (math, code, writing, chat, reasoning) |
| K1672 | Zero-config first run: no API key, no manual weight download |

`success_criteria: []` (DB literal `⚠ INCOMPLETE`).

## Prediction vs. measurement

| Theorem | Prediction | Measurement | Blocks? |
|---------|------------|-------------|---------|
| T1 artifact shortfall | ≥ 3 onboarding-chain artefacts missing | shortfall = 5 (`pyproject.name=pierre`, wheel `packages=["pierre"]`, `[project.scripts] pierre=...`, `first_run` entry point, `pierre/bundles/*`) | PASS |
| T2 parent supersession | parent `exp_prod_pip_package_pierre` KILLED + child step 1 vacuous | parent verdict=KILLED, K1648/K1649/K1650 all FAIL | PASS |
| T3 schema completeness | `success_criteria: []` literal in DB; F#502 cohort hit | confirmed; 8th cohort hit (after F#650/F#652/F#763) | PASS |
| T4 KC pin count | pin_ratio ≤ 0.20 OR ≥ 1 non-falsifiable KC | pin_ratio ≈ 0.20; K1671 + K1672 each non-falsifiable as stated | PASS |
| T5 source-scope breach | ≥ 3 of {A_parent_absent, B_bundle_absent, C_console_script_absent, D_F666_proxy_only} | 4/4 breaches confirmed | PASS |

`all_block = true`, `defense_in_depth = true`, `t_blocking = 5/5`.

Wall-clock for the preempt runner: < 1 s, pure stdlib (no MLX, no
network, no model load).

## Kill criteria

| KC | Verdict | Reason |
|----|---------|--------|
| K1670 | **fail** | T1 ∧ T2: no `pierre` package → no `pip install pierre` → no first-inference timing chain to measure. |
| K1671 | **fail** | T1: `pierre/bundles/{math,code,writing,chat,reasoning}.{safetensors,gguf,npz}` absent; T4: non-falsifiable as stated (any 5-file directory satisfies it under generous read). |
| K1672 | **fail** | T1: no `[project.scripts] pierre = ...`; T4: non-falsifiable; T5(D): proxy-only with no behavioural-quality target → F#666 violation. |

## Precedent map

**Reusable (direct cascade siblings):**
- F#740 (12th F#669 reuse): `exp_pierre_multi_adapter_serving_throughput`
  preempt-KILL — parent `exp_prod_mlxlm_integration` KILLED per
  F#570. Same axis: PROD child unmeasurable until parent KILL is
  remediated.
- F#741 (13th F#669 reuse): `exp_pierre_adapter_cache_prefill`,
  same parent, same axis.
- This experiment: parent is `exp_prod_pip_package_pierre` rather
  than `exp_prod_mlxlm_integration`; same cascade structure with
  a **different parent deliverable**. 3rd PROD-deliverable-cascade
  instance. Promotion candidacy at 4th cross-cluster reuse — track
  in LEARNINGS.

**Reinforcing (F#666 family):**
- Finding #666 (TARGET-GATED KILL): proxy-only KC panels are
  uninterpretable in either direction. K1670+K1671+K1672 all proxy.
  Any "supported" verdict on this panel would be uninterpretable
  even if measurable.

**NOT-TRANSPORT (F#60):**
- F#60 demonstrated llama.cpp + BitNet onboarding works in <60 s on
  M1 Max CPU. Does NOT transport: different base architecture,
  different runtime stack (llama.cpp not MLX), different adapter
  format (rank-16 LoRA vs PoLAR r=6), no published `pierre`
  artefact even if the architecture transported.

**F#502/F#646 cohort tracking:**
- 8th hit. Sustained pattern of `success_criteria: []` co-indicator
  on PROD-deliverable experiments. Reaches the analyst-flagged
  super-family-promotion threshold per the scratchpad guidance
  ("AVOID 8th F#502/F#646 cohort hit"). Recommend analyst promotes
  this from co-indicator to a 1st-class preempt-axis on next pass.

## Operator unblock

The preempt does **not** assert "fast onboarding is impossible";
it asserts "the current parent KILL state cannot measure it." To
unblock, all of:

1. **Resurrect parent `exp_prod_pip_package_pierre`.** Apply its
   own remediation steps (B1 rename `pyproject.name=pierre`,
   B2 add `"pierre"` to wheel `packages`, B3 publish v0.3.x to
   TestPyPI then PyPI). Until B1+B2+B3 land, every PROD child
   downstream of pip-install is preempt-killable.
2. **Bundle the 5 default adapters.** Commit
   `pierre/bundles/{math,code,writing,chat,reasoning}.safetensors`
   to the wheel `[tool.hatch.build.targets.wheel] include` list,
   each pinned by SHA-256.
3. **Add a console script.** `[project.scripts] pierre =
   "pierre.cli:main"` plus a `first_run` entry point that triggers
   on first invocation when no config is present.
4. **Add a target-metric KC** paired with K1670. E.g. "first
   inference produces ≥ X% behavioural quality on a fixed prompt
   set vs MLX reference." This satisfies F#666 (proxy K1670 must
   be paired with target-metric KC).
5. **Populate `success_criteria`** in the DB to clear the F#502
   co-indicator.

Until then, tag the experiment `out-of-scope-pending-parent` and
bump priority to ≥ 4.

## Assumptions (autonomy log)

- Assumed `pierre/bundles/*.safetensors` is the natural location
  for default-bundle adapters per the parent's wheel-build
  convention (vs `composer/bundles/*` or similar). Conclusion is
  invariant under directory choice — no such directory of any name
  exists in the repo.
- Assumed K1670's "first successful inference" means a non-empty
  generation against any prompt; the preempt does not depend on
  this — T1+T2 fire regardless of prompt definition.
- Assumed the M5 Pro hardware is available (per
  `project_user_hardware` memory). T1 is platform-orthogonal:
  the package shortfall is repo-state, not host-state.

## Runner

`run_experiment.py` is pure stdlib: reads `pyproject.toml`,
walks parent dir, greps for entry points, checks for bundle files.
No MLX, no model load, no network, no llama.cpp. Wall-clock < 1 s.
Exit 0; verdict written to `results.json`.

## Antipattern self-check

All 12 antipattern checks PASS or N/A (see MATH.md §Antipattern
self-check). No model code emitted under preempt-KILL clause —
m2 carve-out applies.
