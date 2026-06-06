# Pierre v5 — Fully Ternary LoRA (Grassmannian A + STE B as BitLinear)

**Verdict: KILLED** — dependency-chain + antipattern contamination.

## Summary

Experiment `exp_pierre_v5_ternary_lora` was claimed on 2026-04-17 under the `audit-2026-04-17-rerun` + `tautological-routing` tags. The rerun hat determined that the experiment is not rerunnable in its current form and that the original `results.json` (claimed "ALL PASS" in the DB evidence field) is invalidated by two established antipatterns. Verdict: `killed`.

## Pre-flight (per PLAN.md §1)

1. `results.json["verdict"]` — absent; original code never wrote one. `all_pass: true` was written, but the underlying measurements are tainted (see §3). ❌
2. `results.json["all_pass"]` — `true`, but constructed via tautology, not independent verification. ❌
3. Pre-reg KCs unchanged — ✅ (`MATH.md` authored retrospectively from the claim-time notes; KC IDs 727/728/729 unchanged; git diff n/a because MATH.md did not exist before this iteration).
4. `is_smoke` — n/a.
5. Antipattern check — two triggers (tautological routing; unsafe LORA_SCALE=20). ❌
6. Runnable — ❌. `from pierre.v5 import ...` fails (module absent); skeleton + SFT adapter directories missing.

## Prediction vs measurement

| metric | predicted | original run (2026-04-17, pre-audit) | status |
|---|---|---|---|
| routing accuracy | ≥ 0.90 | 0.996 | **reported pass, measurement valid** — Phase 1 routes each val sample independently, no tautology. |
| single-adapter PPL drop | 3–10% | 14–8% (medical/code/math/legal/finance) | **reported pass, measurement valid** — Phase 2 `ternary_single` is honest single-adapter eval. |
| composition ("pierre") PPL | ≥ single | `pierre_ppl ≡ single_ppl` exactly on all 5 domains | **INVALID** — tautological routing (`route(val[d][0])`) forces same adapter choice as Phase 2a at router acc 0.996. Finding #553 formalises this failure. |
| behavioral overall | 0.30–0.45 | 0.317 | **INVALID as composition claim** — routing uses `test[0]` then fixes adapter for all 5 gen samples per domain. Measures single-adapter behavioral, not routed composition. |
| tok/s | 70–100 | 77.2 | **reported pass, measurement valid** — Phase 4 uses a single static prompt, no routing involved. |
| overhead | 30–55% | 45.1% | same — valid. |

**Net valid measurements:** routing accuracy, single-adapter PPL, decode latency. **Invalidated:** every row labelled `ternary_pierre`, every `behavioral.per_domain.score`, and the overall behavioral number.

## Why the KCs "passed" but we still kill

Antipattern rule (`PLAN.md §1`, verdict-consistency rule 6): **if any `type: fix` antipattern applies, do not mark `supported` regardless of KC numbers.** Two apply:

- **mem-antipattern-002 (tautological routing):** `ternary_pierre` column is identical to `ternary_single` column by construction. This is the exact pattern Finding #553 classifies. Any claim of "pierre composition preserves PPL" is unfounded.
- **mem-antipattern-003 (unsafe LORA_SCALE=20):** `LORA_SCALE = 20.0` at line 44 inflates adapter contribution beyond the safe ≤ 8 regime. Any "signal preserved" claim at scale 20 is a scale-specific claim, not a property of ternary LoRA.

K727 (behavioral ≥ 0.30) measures the wrong object (single-adapter behavioral dressed as composition). K728 (speed ≥ 50 tok/s) and K729 (routing ≥ 0.80) are independently valid but insufficient to support the experiment's composition claim.

## Why the experiment is not rerunnable

The code imports `pierre.v5` (not present), and loads weights from:
- `micro/models/real_data_domain_experts/adapters/grassmannian_skeleton.npz` — absent.
- `micro/models/bitnet_sft_generation_v3/sft_adapters/*/adapter.npz` — parent dir absent.

Only the domain `data/` directories survive. A rerun would need to either:
1. Re-train SFT ternary adapters on BitNet-2B (hours of compute; separate experiment scope).
2. Re-derive the Grassmannian skeleton.
3. Re-implement or restore `pierre/v5` (the BitLinear side-path wrappers, `calibrate_router`, `route`, `inject_ternary_lora`, `strip_lora`, `load_adapter`, `load_skeleton`).

These are all upstream dependencies that belong in separate experiments, not in this rerun.

## Salvageable learnings (carry forward, not promoted to `supported`)

Despite the overall KILL, these sub-findings from the pre-audit run are **untainted by the two triggering antipatterns** and may be cited elsewhere:

1. Ternary side-path does not collapse adapter signal the way v4 premerge did — Phase 2 `ternary_single` shows 8–14% PPL drops per domain, consistent with the v3 bf16 side-path.
2. Decode throughput 77.2 tok/s (45% overhead) confirms the predicted 70–100 tok/s range for 3 ternary matmuls.
3. Router accuracy 0.996 at N=5 is consistent with v3 (0.92 at N=5).

None of these support the **composition** hypothesis that drove the experiment.

## Next experiment (unblocking v8)

The Pierre v8 plan in `PLAN.md` Part 2 already enumerates what a valid rerun needs:

1. Per-sample routing — replace `route(val[d][0])` with `route(sample)` on every validation sample. `pierre_ppl` must then be computed as a mean over per-sample routed evaluations; at routing acc 0.996 it will still approach `single_ppl` but NOT by construction.
2. `LORA_SCALE ≤ 8`. If a claim requires scale 20, it's a scale-specific claim, not a general property.
3. Re-train the 5 SFT ternary adapters + re-derive the Grassmannian skeleton as a named upstream experiment (e.g. `exp_pierre_v5_rebuild_skeleton`).

These are v8 deliverables, not v5 rework. No v5.x follow-up is proposed here.

## Assumptions

- The pre-audit `results.json` content at `micro/models/pierre_v5_ternary_lora/` is the original output; no silent edits since.
- `pierre/v5` module was removed during the post-audit repo cleanup (`de38e37 fix: clean up`), not simply missing pre-run — the original run evidently imported it successfully.
- BitNet-b1.58-2B-4T remains the intended base for ternary composition experiments per PLAN.md Part 2 (Gemma 4 is the Pierre production target; BitNet-2B is the ternary substrate).
- KC IDs 727/728/729 in the DB match what was pre-registered at claim time; the DB `kill_criteria` list is authoritative.
