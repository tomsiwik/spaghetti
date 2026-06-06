# Learnings — exp_pierre_v5_ternary_lora (KILLED)

## Core Finding

The v5 composition hypothesis ("ternary LoRA composed across domains preserves behavior") is **untested**, not falsified. The pre-audit `results.json` appeared ALL PASS, but the composition column (`ternary_pierre`) was byte-identical to the single-adapter column (`ternary_single`) across all 5 domains — a tautology produced by `route(val[d][0])` at `run_experiment.py:159,:183` plus `LORA_SCALE=20.0` at :44. Reviewer hat (2026-04-17) independently verified (f)(i)(j) in the checklist and 3/3 missing dependencies (`pierre/v5` module, `grassmannian_skeleton.npz`, `bitnet_sft_generation_v3/sft_adapters/`).

## Why this is KILL, not PROVISIONAL

- Numerically K727/728/729 pass, so pre-audit this would have been `supported`. Under `mem-framework-004` rule 6, any `type: fix` antipattern match forces reclassification. Two hit (002, 003). `supported` not available.
- `provisional` implies "rerun at larger N fixes it". The problem is logic (tautology) and scale (20), not N — a rerun with same code gets the same tautology. So `provisional` is wrong routing.
- Code is additionally not rerunnable (3 missing deps). `killed` is the only honest status.

## Salvageable (untainted by tautology/scale)

Routing accuracy 0.996 at N=5 · single-adapter PPL drops 8–14% · decode 77.2 tok/s (45% overhead). **None** support the composition claim.

## Implications for Next Experiment

Pierre v8 (per `PLAN.md` Part 2) is the forward path, not a v5.x retry. A clean rerun requires four separate upstream experiments:

1. **Per-sample routing** inside the eval loop (`route(sample)` not `route(val[d][0])`). At acc 0.996 `pierre_ppl` will still approach `single_ppl`, but not by construction.
2. **`LORA_SCALE ≤ 8`**. Findings #328/#330. Scale-20 claims are scale-specific, not general.
3. **Rebuild 5 ternary SFT adapters** (`exp_pierre_v5_rebuild_adapters`) on BitNet-2B.
4. **Re-derive `grassmannian_skeleton.npz`** (pure function of `d, r, N_domains`; Finding #3 cos≈0.0002 at N=5).
5. **Restore `pierre/v5` module** (BitLinear side-path wrappers, `calibrate_router`, `inject_ternary_lora`, `strip_lora`, `load_adapter`, `load_skeleton`).

None are in scope of this rerun. Each belongs in its own experiment.

## Cross-references

Finding #553 (tautological routing, formal) · Finding #289 (v4 premerge kill) · Finding #3 (Grassmannian orthogonality) · `mem-antipattern-002`, `mem-antipattern-003` · `PLAN.md` Part 2 §"Next version plan (v8)".

## Meta

Second kill this iteration with shape "claim → pre-flight → preemptive kill" (also exp_p11_grpo_improve). If recurring, DB could use `rerun-prereq-missing` status distinct from `killed`, and pueue-gated rerun protocol could add a pre-claim rerunnability check.
