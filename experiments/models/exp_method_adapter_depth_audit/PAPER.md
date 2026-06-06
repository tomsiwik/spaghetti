# PAPER — exp_method_adapter_depth_audit

**Verdict: KILLED (dependency structural incompatibility).**
`is_smoke: false` — this is a full measurability audit, not a smoke run.

## Summary

This experiment pre-registered three kill criteria (K1727, K1728, K1729) that test whether the Pierre "method" LoRA adapter sits in Todd-style procedural layer bands (arxiv:2310.15213). The measurability audit shows that **all three KCs are structurally unmeasurable** against the available dependency artifacts:

| KC     | Prediction | Measurement | Verdict |
|--------|-----------|-------------|---------|
| K1727  | Ablation at layers 0-8 causes ≥2× drop vs layers 24-32 on procedural bench | Adapter support = layers [26,41]; layers 0-8 have **zero LoRA weight** → ablation is a degenerate no-op | **FAIL** (unmeasurable) |
| K1728  | Domain adapter shows opposite depth pattern | No validated domain adapter exists: `exp_knowledge_disentanglement_control` KILLED (ΔMMLU=−30pp), `exp_prompt_erasure_gemma4` KILLED (Finding #588), `exp_method_vs_domain_adapter` shipped no domain sibling | **FAIL** (prerequisite missing) |
| K1729  | Residual-stream intervention reproduces method effect within 1pp | No intervention harness exists in repo; building it is multi-day work (per Todd §3) | **FAIL** (out of scope) |

`all_pass = false`, `verdict = KILLED`.

## What was measured

The audit loaded the base model (`mlx-community/gemma-4-e4b-it-4bit`) and enumerated:
1. **Base transformer layer count**: `L = 42` (from `model.model.layers`).
2. **Adapter support**: the safetensors at `exp_method_vs_domain_adapter/adapters/method_multi/adapters.safetensors` contain LoRA weights for layers `[26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]` on modules `{o_proj, v_proj}`.
3. **KC-band overlap**: band `[0, 8]` ∩ adapter support = `∅` (cardinality 0); band `[24, 41]` ∩ adapter support = `[26…41]` (cardinality 16).

The low-band ablation arm of K1727 is therefore a zero-times-zero identity: masking LoRA weights in layers 0-8 cannot change model output because there are no LoRA weights in those layers to begin with. A ≥2× differential cannot arise mechanically. K1727 is unmeasurable with this adapter.

## Why this is KILLED, not RE-SPEC'd

PLAN.md §1 (kill-criteria discipline) and the `kc_swap_after_failure` antipattern explicitly forbid post-hoc KC reformulation. The pre-registered K1727 named specific layer bands `[0,8]` and `[24,32]`; silently changing them to `[26,33]` vs `[34,41]` (the adapter's actual support) would be exactly the behavior the antipattern catalog flags.

Separately, K1728 and K1729 require prerequisites (a validated domain adapter; an intervention harness) that do not exist in this repo and cannot be manufactured in a single hat iteration. They are "prerequisite missing" failures, not "evaluated and falsified".

The dependency chain is: `exp_method_vs_domain_adapter` (smoke, all KCs inconclusive) → this experiment. When the dependency ships only smoke-level artifacts, a downstream experiment that presumes a validated artifact is structurally blocked.

## v2 design (for the follow-up researcher)

Record the requirements pre-flight for a viable v2:
- **A1.** Train a method adapter with `num_layers=-1` so support covers all 42 layers.
- **A2.** Train a paired domain adapter with the same span; validate it retains knowledge (ΔMMLU within ±2pp) before using it as a K1728 arm.
- **A3.** Pre-register KCs over **quartile bands of L=42**: `[0,10]`, `[10,21]`, `[21,31]`, `[31,41]` — all four guaranteed to overlap any full-span adapter.
- **A4.** For K1729, build the residual-stream harness as its own experiment (`exp_residual_intervention_harness`) and mark this experiment blocked on that artifact. Do not entangle harness construction with the depth test.

## Assumptions (autonomous decisions)

- **Ass1.** Interpreted `num_layers=16` in `adapter_config.json` as "LoRA on the last 16 layers of the base", matching `mlx-lm`'s `linear_to_lora_layers` semantics. Runtime confirmation: the safetensors contain layer indices 26-41, exactly the last 16 of L=42.
- **Ass2.** Treated K1727's `[0,8]` and `[24,32]` as closed integer ranges in a 0-indexed layer numbering.
- **Ass3.** Did not attempt to substitute `method_single_math` for `method_multi`; both have the same `num_layers=16` config so the structural issue is identical.
- **Ass4.** Did not train new adapters or build the residual harness. Both are out of scope for a single hat iteration and would constitute entirely new experiments.

## Antipattern pre-flight

| Antipattern | Status |
|-------------|--------|
| composition math bug | N/A (no composition in this audit) |
| unsafe adapter scale | N/A (no training) |
| tautological routing | N/A (no routing) |
| `kc_swap_after_failure` | **avoided** — KCs are failed as unmeasurable, not relaxed |
| `hardcoded "pass": True` | absent (`all_pass=false`, `verdict=KILLED` both explicit) |
| eval-template truncation | N/A (no eval) |
| smoke-as-full | `is_smoke=false` correctly, because this is an audit not a smoke run |
| proxy-model-substituted | N/A (we used the real Gemma-4-E4B) |

All antipatterns N/A or avoided. No KC modifications between MATH.md (git) and now.

## Files produced

- `MATH.md` — theorem, measurability analysis, pre-registered KCs, assumptions
- `run_experiment.py` — loads base, enumerates adapter layers, writes measurability verdict
- `results.json` — verdict KILLED, full per-KC measurability data, adapter layer enumeration
- `PAPER.md` — this file
- `REVIEW-adversarial.md` — pending reviewer
- `LEARNINGS.md` — pending analyst
