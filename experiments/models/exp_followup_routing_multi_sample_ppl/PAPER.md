# PAPER: Per-sample routing PPL follow-up

## Verdict: **KILLED** (preemptive, 2026-04-18)

## tl;dr
K1549 cannot be measured — 0 of 5 required adapter weights exist (antipattern-017
cascade, 5th confirmed instance). The kill criterion is nonetheless **settled by
theorem** (MATH.md Theorem 1): per-sample routing cannot force
`pierre_ppl ≡ single_ppl` at finite router accuracy, so the "not identical by
construction" clause of K1549 is true by derivation.

## Why not run
- `adapters/{math,bash,python,sql,medical}/adapters.safetensors` — 0 of 5 present.
- Upstream Pierre siblings (`pierre_unified_pipeline`, `pierre_v3_sft_n5`,
  `pierre_v5_ternary_lora`, `pierre_v6_precomputed_concat`) — all `status=killed`,
  none saved adapter weights (only `router_W.npy`).
- Consequence: `single_ppl[d]` (apply adapter `d` per-sample) and `pierre_ppl[d]`
  (apply router-dispatched adapter per-sample) are both unmeasurable.

## Prediction vs measurement
| Kill criterion | Predicted (theorem) | Measured | Verdict |
|---|---|---|---|
| K1549: pierre_ppl ≠ single_ppl under per-sample routing at p∈[0.85,0.99] | P(identity) ≤ p^(N·D) ≤ 0.082 at p=0.99 | unmeasurable (no weights) | fail-by-cascade |

The magnitude question (*how much* do they differ?) is a separate empirical
question that this experiment was designed to answer; it remains open.

## Dependency state
| Path | Required | Present |
|---|---|---|
| adapters/math/adapters.safetensors | yes | **no** |
| adapters/bash/adapters.safetensors | yes | **no** |
| adapters/python/adapters.safetensors | yes | **no** |
| adapters/sql/adapters.safetensors | yes | **no** |
| adapters/medical/adapters.safetensors | yes | **no** |
| pierre_unified_pipeline adapter weights | yes | **no** (only router_W.npy) |

## Antipattern self-check
- antipattern-017 (stub-adapter cascade, consumer side): **TRIGGERED**. 5 of 5.
- antipattern-020 (cascade-dependent design on killed upstreams): **TRIGGERED**.
- antipattern-003/008/KC-swap: N/A (no run, MATH.md single commit).

## Salvageable
- MATH.md Theorem 1: combinatorial bound `P(identity) ≤ p^(N·D)` — closes K1549
  by derivation. This generalises Finding #553 from the single-sample artifact
  to the per-sample case: even with perfect protocol, per-sample routing cannot
  produce tautological identity except by measure-zero coincidence.
- Prediction: at measured `p=0.996` (from `pierre_unified_pipeline/results.json`),
  per-sample routing should yield `|pierre_ppl[d] − single_ppl[d]| ≈ 0.06` on
  average — small but nonzero. A v2 with real adapters would verify the
  magnitude.

## Unblock path
1. Train 5 domain adapters on Gemma 4 E4B (see `exp_p1_t2_single_domain_training`
   open experiment; same queue as the P11.ADAPTER-REBUILD bucket).
2. Save weights with `adapters.safetensors` format to `adapters/<domain>/`.
3. Re-register as `exp_followup_routing_multi_sample_ppl_v2` with the same MATH.md.

## Assumptions / Open threads
- Assumed `p` is i.i.d. across samples — real routers may be correlated by token
  prefix; correlation would reduce `P(identity)` further (stronger claim, same
  direction).
- F#560 baseline reconciliation is independent of this experiment (no baseline
  comparison used).

## References
- Finding #553 (supported, 2026-04-17): single-sample routing artifact.
- `micro/models/pierre_unified_pipeline/results.json` (identity evidence).
- `.ralph/agent/memories.md` antipattern-017 (3 confirmed → now 5), antipattern-020.
- MATH.md Theorem 1 (this experiment).

## Handoff note to Reviewer
- Verify: `ls adapters/{math,bash,python,sql,medical}/adapters.safetensors` → all ENOENT.
- Verify: MATH.md single commit (no KC-swap).
- Verify: `experiment get` on all 4 Pierre siblings → all `killed`.
- Consider: analyst should bump antipattern-017 to "5 confirmed instances" and
  generalise pre-flight to include Pierre-lineage consumers.
