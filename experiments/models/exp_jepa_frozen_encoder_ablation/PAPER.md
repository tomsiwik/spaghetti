# PAPER.md — exp_jepa_frozen_encoder_ablation (PREEMPT-KILL, TRIPLE-FIRE, 5th SAME-PARENT-F#682, 1st POST-PROMOTION)

## Verdict line

**KILLED — preempt-structural triple-fire (F#666-pure 17th + §5-tautological 11th + F#669 7th reuse). No MLX code executed. 5th child of parent `exp_jepa_adapter_residual_stream` (F#682 PROVISIONAL) to hit preempt-KILL → first post-promotion instance of `mem-promotion-same-parent-repeat-blocker` (promoted at F#728). Routing stable; N+=1 census update only. |K|=1 — most degenerate F#666-pure case to date.**

## Prediction vs measurement

| KC    | Claim                                                                | Kind          | Pre-registered | Measurement status                      | Result |
| ----- | -------------------------------------------------------------------- | ------------- | -------------- | --------------------------------------- | ------ |
| K1889 | Frozen-encoder JEPA MSE > 1.5× fine-tuned encoder JEPA MSE           | proxy         | yes            | untested (preempt-blocked, triple-fire) | —      |

**|K|=1**. Single-KC experiment. **all_pass: false** (KC set unmeasured). **is_smoke: false** (no code ran). Verdict is structural, not empirical.

## Rationale (condensed from MATH.md)

Three independent preempts fire on this claim:

1. **F#666-pure standalone (17th reuse):** Sole KC K1889 is a prediction-loss ratio (proxy). No target metric per guardrail 1007. With |K|=1, F#666-pure is maximally degenerate — there is no companion KC to re-pair at re-claim; a genuine NEW target must be added from scratch.

2. **§5 tautological-inter-variant-delta (11th reuse):** K1889 compares frozen-encoder JEPA MSE to fine-tuned-encoder JEPA MSE with no external anchor. Both variants are untested realizations of parent F#682's JEPA mechanism; the 1.5× threshold has no external calibration (why 1.5× rather than 1.2× or 2.0×?).

3. **F#669 parent-target-unverified (7th reuse):** Parent `exp_jepa_adapter_residual_stream` (F#682) is PROVISIONAL with K1767/K1768/K1769 untested. K1889's fine-tuned-encoder RHS is exactly parent's canonical training trajectory (parent K1767 measures `L_pred` on this trajectory).

Each block is independently sufficient. Triple co-occurrence catalogued per `mem-promotion-triple-fire-mode` (anchored F#721).

## F#669 reuse ledger (updated)

| Finding | Date       | Child                                      | Parent                            | F#666 state         | Notes                                 |
| ------- | ---------- | ------------------------------------------ | --------------------------------- | ------------------- | ------------------------------------- |
| F#669   | 2026-04-19 | exp_rdt_act_halting_throughput             | exp_rdt_loop_lora_gemma4          | —                   | Original canonical                    |
| F#687   | 2026-04-23 | exp_jepa_router_prediction_error           | exp_jepa_adapter_residual_stream  | —                   | 2nd reuse; same-parent-F#682 child 1  |
| F#698   | 2026-04-24 | exp_jepa_adapter_attention_output          | exp_jepa_adapter_residual_stream  | compound            | 3rd reuse; same-parent-F#682 child 2  |
| F#699   | 2026-04-24 | exp_memento_compression_ratio_benchmark    | exp_memento_gemma4_replication    | compliant           | 4th reuse; cross-parent-family OK     |
| F#727   | 2026-04-24 | exp_jepa_multilayer_prediction             | exp_jepa_adapter_residual_stream  | compliant           | 5th reuse; same-parent-F#682 child 3  |
| F#728   | 2026-04-24 | exp_jepa_contrastive_variant               | exp_jepa_adapter_residual_stream  | pure (|K|=2)        | 6th reuse; same-parent-F#682 child 4 — PROMOTION |
| **this**| 2026-04-24 | **exp_jepa_frozen_encoder_ablation**       | **exp_jepa_adapter_residual_stream** | **pure (|K|=1)** | **7th reuse; same-parent-F#682 child 5 — 1st post-promotion** |

## Same-parent-repeat-blocker (post-promotion) ledger

`mem-promotion-same-parent-repeat-blocker` was promoted at F#728 (4-child threshold). This is the first post-promotion instance.

| # | Child                                     | Finding | KC structure                    | F#666 state | Pre/Post promotion |
| - | ----------------------------------------- | ------- | ------------------------------- | ----------- | ------------------ |
| 1 | exp_jepa_router_prediction_error          | F#687   | proxy+proxy                     | compound    | pre                |
| 2 | exp_jepa_adapter_attention_output         | F#698   | proxy-only (|K|=2)              | compound    | pre                |
| 3 | exp_jepa_multilayer_prediction            | F#727   | proxy+target                    | compliant   | pre                |
| 4 | exp_jepa_contrastive_variant              | F#728   | proxy+safety (|K|=2)            | pure        | promotion trigger  |
| 5 | **exp_jepa_frozen_encoder_ablation (this)** | **TBD** | **proxy-only (|K|=1)**        | **pure**    | **1st post-promotion** |

**Post-promotion routing (per promoted memory):** 5th-and-later instances contribute N+=1 census only; no new memory theorem. Route as preempt-KILL per standard triple-fire template. No new memory write here.

Parent F#682 unblock leverage now **5:1**. Still the highest-leverage single JEPA unblock action remains `exp_jepa_adapter_residual_stream_impl` (P=1). All 5 children become re-claimable the moment parent `_impl` lands SUPPORTED — **modulo** each child's independent KC-augmentation needs. This 5th child requires adding a target KC from scratch (|K|=1→2), making it the most costly child to unblock.

## Triple-fire ledger

| # | Experiment                                 | Finding | Memories fired                                    |
| - | ------------------------------------------ | ------- | ------------------------------------------------- |
| 1 | exp_sigreg_hedgehog_combined               | F#714   | multi-bucket                                      |
| 2 | exp_g4_adapter_svd_denoise                 | F#716   | F#666-pure + PPL-bucket + F#702-unavailability    |
| 3 | exp_hedgehog_loss_variant_mse              | F#720   | F#666-pure + ...                                  |
| 4 | exp_hedgehog_layer_selection_top6          | F#721   | F#666-pure + ... (PROMOTION of triple-fire-mode)  |
| 5 | exp_hedgehog_teacher_temperature_sweep     | F#722   | F#666-pure + ... (post-promotion)                 |
| 6 | exp_jepa_contrastive_variant               | F#728   | F#666-pure + §5 + F#669 (novel sub-composition)   |
| 7 | **exp_jepa_frozen_encoder_ablation (this)** | **TBD** | **F#666-pure + §5 + F#669** (same as F#728)    |

This is the 7th triple-fire, 6th post-promotion. The memory composition (F#666-pure + §5 + F#669) matches F#728 exactly — confirming the structural/parent-dependent sub-composition first observed at F#728. Two-instance observation of the same triple-composition within same-parent cohort.

## Distinct severity vs F#728

Both F#728 and this experiment fire the same three memories. Key structural distinction:

- **F#728** (contrastive_variant): |K|=2 (K1887 proxy + K1888 safety-guard). KC augmentation at re-claim can potentially re-pair K1887 with a new target (K1888 may remain as safety-guard).
- **This** (frozen_encoder_ablation): |K|=1 (K1889 proxy only). KC augmentation requires adding a genuine NEW target KC from scratch — no existing KC to re-pair.

This makes F#666-pure maximally degenerate here. First such instance in the drain window.

## Unblock condition

Three gates must clear:

1. **F#666-pure + §5 blocks** cleared by one action: augment KC set with a genuine new target metric (e.g. "Frozen-encoder JEPA adapter GSM8K-Hard accuracy ≥ (fine-tuned encoder - 3pp) at matched param budget on Gemma 4 E4B"). This is more work than F#728's re-pair — here it's a de-novo addition.

2. **F#669 block** clears only when parent F#682 reaches SUPPORTED via its `_impl` (P=1, filed) with K1767 + K1768 + K1769 all passing.

Re-claim at that point is F#666-compliant (with |K|≥2) and inter-variant-delta becomes meaningful (both sides anchored to external baseline).

## Follow-up

No `_impl` follow-up filed. Preempt-structural KILL does NOT spawn `_impl` per F#687/F#698/F#699/F#727/F#728 precedent + reviewer.md §5. Unblock actions are:

- Researcher-owned at re-claim: KC augmentation (here: de-novo target KC addition).
- External: parent's `exp_jepa_adapter_residual_stream_impl` (P=1, already filed).

**No analyst memory write pending.** `mem-promotion-same-parent-repeat-blocker` already promoted at F#728; this is first post-promotion instance, routed per promoted memory rules (N+=1 census only).

## Sibling-position table

Same-family JEPA experiments (all blocked by F#682 PROVISIONAL):

| # | Child                                     | Priority | Status  | Route                       |
| - | ----------------------------------------- | -------- | ------- | --------------------------- |
| 1 | exp_jepa_adapter_residual_stream          | —        | PROVISIONAL | parent F#682            |
| 2 | exp_jepa_router_prediction_error          | 2        | killed  | F#687 preempt-KILL          |
| 3 | exp_jepa_adapter_attention_output         | 2        | killed  | F#698 preempt-KILL          |
| 4 | exp_jepa_multilayer_prediction            | 2        | killed  | F#727 preempt-KILL          |
| 5 | exp_jepa_contrastive_variant              | 2        | killed  | F#728 triple-fire + promotion |
| 6 | **exp_jepa_frozen_encoder_ablation (this)** | **2**  | **killed** | **triple-fire, 1st post-promotion** |
| 7 | exp_jepa_scale_sweep_5m_15m_50m           | 2        | open    | likely next post-promotion instance |

## Assumptions (cross-referenced)

See MATH.md §8 A1-A12. Key:

- A1: Parent F#682 remains PROVISIONAL at claim time (verified via `experiment get`).
- A4: DB `depends_on=[]` does not preclude structural dependence (F#687/F#698/F#727/F#728 precedent).
- A5-A7: Post-promotion routing stable for F#669, triple-fire-mode, and same-parent-repeat-blocker; no re-derivation. N+=1 census only.
- A11: F#702 hygiene-patch APPLIED (platform=local-apple, dir set, success_criteria + evidence populated at `experiment complete`). References array left empty per preempt-KILL precedent.

## Antipattern audit

- **antipattern-t (silent objective swap):** CHECKED — KC verbatim from DB, no "simpler" substitute measurement attempted.
- **mem-antipattern-impl-follow-up-delegation:** NOT APPLICABLE — preempt-KILL class, not novel-mechanism PROVISIONAL.
- **F#702 hygiene-patch:** APPLIED — platform=local-apple, dir=`micro/models/exp_jepa_frozen_encoder_ablation/`, success_criteria populated at complete, evidence populated at complete. References array left empty (matches F#698/F#699/F#727/F#728 preempt-KILL precedent).
- **F#666 classification:** PURE (sole KC K1889 is proxy; |K|=1) — distinct from F#728 (|K|=2 pure), F#687/F#698 compound, F#727 compliant.
- **Mutation at re-claim audit:** KC text preserved verbatim. MATH.md §3 KC table matches DB `experiment get` output exactly.
