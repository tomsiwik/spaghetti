# PAPER.md — exp_jepa_contrastive_variant (PREEMPT-KILL, TRIPLE-FIRE + SAME-PARENT-REPEAT-BLOCKER PROMOTION)

## Verdict line

**KILLED — preempt-structural triple-fire (F#666-pure + §5-tautological-inter-variant-delta + F#669 6th reuse). No MLX code executed. 4th child of parent `exp_jepa_adapter_residual_stream` (F#682 PROVISIONAL) to hit preempt-KILL → same-parent-repeat-blocker watchlist promotion triggered (analyst-owned memory write).**

## Prediction vs measurement

| KC    | Claim                                                                | Kind          | Pre-registered | Measurement status                      | Result |
| ----- | -------------------------------------------------------------------- | ------------- | -------------- | --------------------------------------- | ------ |
| K1887 | InfoNCE variant next-embedding accuracy < MSE variant                | proxy         | yes            | untested (preempt-blocked, triple-fire) | —      |
| K1888 | InfoNCE training unstable (loss NaN or divergence > 3 epochs)        | safety-guard  | yes            | untested (preempt-blocked, triple-fire) | —      |

**all_pass: false** (KC set unmeasured). **is_smoke: false** (no code ran). Verdict is structural, not empirical.

## Rationale (condensed from MATH.md)

Three independent preempts fire on this claim:

1. **F#666-pure standalone (16th reuse):** Neither KC is a target metric per guardrail 1007. K1887 is a structural prediction-quality proxy; K1888 is a training-dynamics safety guard. No target KC exists — the experiment is structurally unsupportable (cannot reach SUPPORTED even in principle).

2. **§5 tautological-inter-variant-delta (10th reuse):** K1887 compares InfoNCE-variant accuracy to MSE-variant accuracy with no external anchor. Both variants are untested realizations of parent F#682's JEPA mechanism; "A beats B" on a proxy between two unvalidated designs is non-informative.

3. **F#669 parent-target-unverified (6th reuse):** Parent `exp_jepa_adapter_residual_stream` (F#682) is PROVISIONAL with K1767/K1768/K1769 untested. K1887's MSE-variant RHS is exactly parent's untested mechanism; K1888's stability check on a loss variant of an unvalidated design is behaviorally uninformative.

Each block is independently sufficient. The triple co-occurrence is catalogued per `mem-promotion-triple-fire-mode` (anchored F#721).

## F#669 reuse ledger (updated)

| Finding | Date       | Child                                      | Parent                            | F#666 state         | Notes                                 |
| ------- | ---------- | ------------------------------------------ | --------------------------------- | ------------------- | ------------------------------------- |
| F#669   | 2026-04-19 | exp_rdt_act_halting_throughput             | exp_rdt_loop_lora_gemma4          | —                   | Original canonical                    |
| F#687   | 2026-04-23 | exp_jepa_router_prediction_error           | exp_jepa_adapter_residual_stream  | —                   | 2nd reuse; same-parent-F#682 child 1  |
| F#698   | 2026-04-24 | exp_jepa_adapter_attention_output          | exp_jepa_adapter_residual_stream  | compound            | 3rd reuse; same-parent-F#682 child 2  |
| F#699   | 2026-04-24 | exp_memento_compression_ratio_benchmark    | exp_memento_gemma4_replication    | compliant           | 4th reuse; cross-parent-family OK     |
| F#727   | 2026-04-24 | exp_jepa_multilayer_prediction             | exp_jepa_adapter_residual_stream  | compliant           | 5th reuse; same-parent-F#682 child 3  |
| **this**| 2026-04-24 | **exp_jepa_contrastive_variant**           | **exp_jepa_adapter_residual_stream** | **pure**         | **6th reuse; same-parent-F#682 child 4 — PROMOTION TRIGGER** |

## Same-parent-repeat-blocker promotion trigger

Watchlist anchored at F#727 stated: *"If 4th same-parent child of F#682 hits preempt-KILL, promote to standalone memory."* This is the 4th. Trigger fires.

| # | Child                                     | Finding | KC structure                  | F#666 state   | Preempt memories fired        |
| - | ----------------------------------------- | ------- | ----------------------------- | ------------- | ----------------------------- |
| 1 | exp_jepa_router_prediction_error          | F#687   | proxy+proxy                   | compound      | F#669                         |
| 2 | exp_jepa_adapter_attention_output         | F#698   | proxy-only                    | compound      | F#669                         |
| 3 | exp_jepa_multilayer_prediction            | F#727   | proxy+target                  | compliant     | F#669                         |
| 4 | **exp_jepa_contrastive_variant (this)**   | **TBD** | **proxy+safety**              | **pure**      | **F#669 + §5 + F#666-pure**   |

All 4 children become re-claimable the moment `exp_jepa_adapter_residual_stream_impl` (P=1) lands SUPPORTED — **modulo** each child's independent KC-augmentation needs (F#698 compound and this F#666-pure variant both need target KC added at re-claim).

Parent F#682 unblock leverage: **≥4:1**. Highest-leverage single JEPA unblock action remains `exp_jepa_adapter_residual_stream_impl`.

## Triple-fire ledger

| # | Experiment                                 | Finding | Memories fired                                    |
| - | ------------------------------------------ | ------- | ------------------------------------------------- |
| 1 | exp_sigreg_hedgehog_combined               | F#714   | multi-bucket                                      |
| 2 | exp_g4_adapter_svd_denoise                 | F#716   | F#666-pure + PPL-bucket + F#702-unavailability    |
| 3 | exp_hedgehog_loss_variant_mse              | F#720   | F#666-pure + ...                                  |
| 4 | exp_hedgehog_layer_selection_top6          | F#721   | F#666-pure + ... (PROMOTION)                      |
| 5 | exp_hedgehog_teacher_temperature_sweep     | F#722   | F#666-pure + ... (post-promotion)                 |
| 6 | **exp_jepa_contrastive_variant (this)**    | **TBD** | **F#666-pure + §5 + F#669**                       |

This is the 6th triple-fire, 5th post-promotion-of-triple-fire-mode. Distinction: this is the **first** triple-fire that combines F#666-pure with the *structural / parent-dependent* memories (§5 + F#669), whereas prior triple-fires combined F#666-pure with metric-bucket memories. Novel composition pattern for the triple-fire ledger; noted in LEARNINGS analyst pass.

## Unblock condition

Three gates must clear:

1. **F#666-pure block + §5 block** cleared by one action: augment KC set with one target metric (e.g. "InfoNCE-variant adapter GSM8K-Hard accuracy ≥ LoRA r=16 baseline"). This is identical to F#698-attention_output's unblock action and breaks both patterns simultaneously.

2. **F#669 block** clears only when parent F#682 reaches SUPPORTED via its `_impl` (P=1, filed) with K1767 + K1768 + K1769 all passing.

Re-claim at that point is F#666-compliant and inter-variant-delta becomes meaningful (both sides anchored to external baseline).

## Follow-up

No `_impl` follow-up filed. Preempt-structural KILL does NOT spawn `_impl` per F#687/F#698/F#699/F#727 precedent + reviewer.md §5. Unblock actions are:

- Researcher-owned at re-claim: KC augmentation.
- External: parent's `exp_jepa_adapter_residual_stream_impl` (P=1, already filed).

**Analyst-owned memory write pending:** `mem-promotion-same-parent-repeat-blocker` documenting: (a) 4-instance threshold reached, (b) same parent F#682, (c) canonical unblock-leverage ratio "N same-parent children : 1 parent `_impl`", (d) future 5th+ same-parent-same-PROVISIONAL-parent should claim-time-gate against parent status.

## Sibling-position table

Same-family JEPA experiments (all blocked by F#682 PROVISIONAL):

| # | Child                                     | Priority | Status  | Route                       |
| - | ----------------------------------------- | -------- | ------- | --------------------------- |
| 1 | exp_jepa_adapter_residual_stream          | —        | PROVISIONAL | parent F#682            |
| 2 | exp_jepa_router_prediction_error          | 2        | killed  | F#687 preempt-KILL          |
| 3 | exp_jepa_adapter_attention_output         | 2        | killed  | F#698 preempt-KILL          |
| 4 | exp_jepa_multilayer_prediction            | 2        | killed  | F#727 preempt-KILL          |
| 5 | **exp_jepa_contrastive_variant (this)**   | **2**    | **killed** | **triple-fire preempt-KILL** |
| 6 | exp_jepa_scale_sweep_5m_15m_50m           | 2        | open    | likely 5th same-parent if claimed |
| 7 | exp_jepa_frozen_encoder_ablation          | 2        | open    | likely same-parent if claimed |

## Assumptions (cross-referenced)

See MATH.md §8 A1-A12. Key:

- A1: Parent F#682 remains PROVISIONAL at claim time (verified via `experiment get`).
- A4: DB `depends_on=[]` does not preclude structural dependence (F#687/F#698/F#727 precedent).
- A5-A6: Post-promotion routing stable for F#669 and triple-fire-mode; no re-derivation.
- A7: Same-parent-repeat-blocker promotion threshold = 4 (this is the 4th).
- A11: F#702 hygiene-patch APPLIED (platform=local-apple, dir set, success_criteria + evidence populated at `experiment complete`). References array left empty per preempt-KILL precedent.

## Antipattern audit

- **antipattern-t (silent objective swap):** CHECKED — KCs verbatim from DB, no "simpler" substitute measurement attempted.
- **mem-antipattern-impl-follow-up-delegation:** NOT APPLICABLE — preempt-KILL class, not novel-mechanism PROVISIONAL.
- **F#702 hygiene-patch:** APPLIED — platform=local-apple, dir=`micro/models/exp_jepa_contrastive_variant/`, success_criteria populated at complete, evidence populated at complete. References array left empty (matches F#698/F#699/F#727 preempt-KILL precedent).
- **F#666 classification:** PURE (no target KC paired with K1887/K1888 proxies) — distinct from F#687/F#698 compound (multiple proxies) and F#727 compliant (proxy+target).
- **Mutation at re-claim audit:** KC text preserved verbatim. MATH.md §3 KC table matches DB `experiment get` output exactly.
