# PAPER.md — exp_jepa_multilayer_prediction (PREEMPT-KILL)

## Verdict line

**KILLED — preempt-structural (F#669, 5th reuse). No MLX code executed. Parent `exp_jepa_adapter_residual_stream` PROVISIONAL (F#682); both KCs reference an L+1 baseline that parent has not target-validated.**

## Prediction vs measurement

| KC    | Claim                                                                  | Kind   | Pre-registered | Measurement status              | Result |
| ----- | ---------------------------------------------------------------------- | ------ | -------------- | ------------------------------- | ------ |
| K1885 | L+2 prediction MSE > 2× L+1 prediction MSE (skip-connection ineffective) | proxy  | yes            | untested (preempt-blocked, F#669) | —      |
| K1886 | L+2 trained adapter behavioral quality > 3pp worse than L+1             | target | yes            | untested (preempt-blocked, F#669) | —      |

**all_pass: false** (KC set unmeasured). **is_smoke: false** (no code ran). Verdict is structural, not empirical.

## Rationale (condensed from MATH.md)

Both KCs compare against an L+1 baseline quantity that parent `exp_jepa_adapter_residual_stream` (F#682 PROVISIONAL) has not measured. K1767 (L+1 L_pred ratio) and K1768 (L+1 GSM8K-Hard accuracy vs LoRA r=16) are parent's untested targets. Comparing L+2 to an unverified L+1 yields unidentifiable samples per F#669 canonical theorem.

F#666-compliant KC set (proxy K1885 + target K1886) — no compound block, matches F#699 precedent, NOT F#698-attention_output (which was proxy-only and compound-blocked).

## F#669 reuse ledger

| Finding | Date       | Child                                      | Parent                            | F#666 compound | Notes                                 |
| ------- | ---------- | ------------------------------------------ | --------------------------------- | -------------- | ------------------------------------- |
| F#669   | 2026-04-19 | exp_rdt_act_halting_throughput             | exp_rdt_loop_lora_gemma4          | —              | Original canonical                    |
| F#687   | 2026-04-23 | exp_jepa_router_prediction_error           | exp_jepa_adapter_residual_stream  | —              | 2nd-reuse promotion threshold flagged |
| F#698   | 2026-04-24 | exp_jepa_adapter_attention_output          | exp_jepa_adapter_residual_stream  | yes            | 3rd reuse; promotion confirmed        |
| F#699   | 2026-04-24 | exp_memento_compression_ratio_benchmark    | exp_memento_gemma4_replication    | —              | 4th reuse; cross-parent-family OK     |
| this    | 2026-04-24 | exp_jepa_multilayer_prediction             | exp_jepa_adapter_residual_stream  | —              | 5th reuse; post-promotion routing     |

Same parent (`exp_jepa_adapter_residual_stream`) blocked 3 children (F#687, F#698, this). Parent F#682 unblock leverage ≥3:1.

## Sibling-position table (same-parent children)

| # | Child                                  | F#        | Status | F#666 compound | Unblock condition        |
| - | -------------------------------------- | --------- | ------ | -------------- | ------------------------ |
| 1 | exp_jepa_router_prediction_error       | F#687     | killed | no             | Parent → supported       |
| 2 | exp_jepa_adapter_attention_output      | F#698     | killed | yes            | Parent → supported + KC augment |
| 3 | exp_jepa_multilayer_prediction (this)  | next      | killed | no             | Parent → supported       |

All 3 children become re-claimable the moment `exp_jepa_adapter_residual_stream_impl` (P=1) lands SUPPORTED.

## Unblock condition

Parent `exp_jepa_adapter_residual_stream` must reach `status=supported` via `exp_jepa_adapter_residual_stream_impl` (P=1, already filed) with:

1. K1766 SUPPORTED (SIGReg non-collapse proxy)
2. K1767 SUPPORTED (L_pred ratio proxy: step500/step50 < 0.5)
3. K1768 SUPPORTED (GSM8K-Hard target: ≥ LoRA r=16)
4. K1769 SUPPORTED (lambda=0 ablation target: ≥5pp drop)

K1767 provides the L+1 MSE anchor for K1885. K1768 provides the L+1 behavioral-quality anchor for K1886. No KC-augmentation needed at re-claim (unlike F#698-attention_output).

## Follow-up

No `_impl` follow-up filed. Preempt-structural KILL does NOT spawn `_impl` per F#687/F#698/F#699 precedent + reviewer.md §5. Unblock is parent-external.

## Assumptions (cross-referenced)

See MATH.md §8 A1-A12. Key assumptions:
- A1: Parent F#682 status genuinely "design-only" (verified via `experiment get`).
- A2: Post-promotion F#669 routing applies at 5th reuse without re-derivation.
- A4: K1886's "L+1 behavioral quality" = parent K1768 GSM8K-Hard accuracy (not generic LM metric).
- A7: Preempt-KILL does not anchor F#702 global references (matches F#698/F#699).

## Antipattern audit

- antipattern-t (silent objective swap): CHECKED — KCs verbatim from DB, no "simpler" substitute measurement attempted.
- mem-antipattern-impl-follow-up-delegation: NOT APPLICABLE — preempt-KILL class, not novel-mechanism PROVISIONAL.
- F#702 hygiene-patch: APPLIED (platform=local-apple, dir=micro/models/exp_jepa_multilayer_prediction/, success_criteria + evidence will be populated at `experiment complete` step). References array left empty per F#698/F#699 precedent.
- F#666 compound-subcase: NOT PRESENT — KC set target-gated at construction (K1885 proxy + K1886 target).
