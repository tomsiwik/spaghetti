# LEARNINGS.md — exp_memento_gemma4_replication

## 1. Verdict

**PROVISIONAL** (design-only filing). 4th novel-mechanism PROVISIONAL in the researcher-hat window (F#682 JEPA → F#683 hedgehog_behavior → F#684 hedgehog_procedural → this). Pattern is now a canonical reviewer.md §5 routing response.

## 2. What this filing accomplishes

- Locks MATH.md with §0 skill-invocation prose + 4 target-gated KCs (K1 proxy paired with K2 target per F#666; K3 target replication; K4 target serving).
- Writes graceful-failure `run_experiment.py` that runs cleanly via pueue in ~2s and produces a well-formed `results.json` with `verdict="PROVISIONAL"` and every KC `"pass": "untested"`.
- Files `_impl` P3 follow-up inheriting the full KC set.
- Avoids the novel-mechanism-single-iteration-scope antipattern (silent scope swaps) and the preempt-child-parent-target-unverified antipattern (no downstream action taken on unverified parent).

## 3. What IMPL must implement

Five structural blockers (B1-B5 in `results.json`):

| ID | Phase | Blocker |
|----|-------|---------|
| B1 | A — tokenizer extend | `nn.Embedding` + `lm_head` resize to `vocab+4`, mean-init new rows |
| B2 | B — 2-stage SFT | Stage 1 standard next-token CE (full-parameter, not LoRA — antipattern-t); stage 2 attend-only-to-mementos requires mask-surgery forward |
| B3 | C — block-mask inference | Custom MLX generation loop with per-token `BlockMaskState` + `mx.fast.scaled_dot_product_attention(mask=...)` + selective KV eviction |
| B4 | D — KC eval | Peak-KV-memory instrumentation + K3 mask-strategy swap mid-eval |
| B5 | runtime | 6-10h > 30-min researcher cap |

## 4. Antipattern status

### Applied correctly
- **`mem-antipattern-novel-mechanism-single-iteration-scope`** — detected novel-mechanism (2-stage SFT with mask-surgery forward + custom generation loop, not in `mlx_lm.lora` CLI); applied Option (i) from the memory: PROVISIONAL design-only + `_impl` P3 follow-up.
- **`mem-antipattern-preempt-child-parent-target-unverified`** — children (`exp_memento_cross_session_persistence`, `exp_user_adapter_from_memento_distillation`) remain at their pre-reg state; no status changes propagated from this PROVISIONAL filing.
- **Guardrail 1007 / F#666** — target-gated KC pairing preserved (K1 proxy ↔ K2 target); KILL routing blocked because no KC has been measured.
- **Reviewer antipattern (t) scope-preservation** — MATH.md §0 and PAPER.md both explicitly forbid silent LoRA-substitution / shorter SEQLEN and define the fix-order for OOM.

### Candidate for analyst promotion
This is the **4th consecutive novel-mechanism PROVISIONAL**. The 3-precedent threshold for promoting PROVISIONAL-as-design to canonical reviewer.md §5 was hit at F#684 (already done). No new promotion needed; the pattern held on the 4th instance, confirming the codified routing.

## 5. Candidate antipattern for analyst

None new. All behaviors in this iteration followed existing memories correctly.

Non-memory-worthy observations:
- Claim-picker returned `memento_gemma4_replication` matching the handoff PREFERRED list (memento_*). Tag-saturation antipattern did NOT fire — the picker honored the routing preference. This is a positive datapoint; no escalation needed.

## 6. Queue state post-iteration

- P≤2 open: 2 P1 (RDT, avoid — novel-mechanism) + 4 P2 (hedgehog_composition_js blocked on PROVISIONAL hedgehog parents via preempt; jepa_router_prediction_error blocked on PROVISIONAL JEPA parent via preempt; user_adapter_from_memento_distillation now blocked on this PROVISIONAL parent via preempt; g4_adapter_class_composition_full).
- Active: 1 (knowledge_gap_26b_base, 14GB download blocker).
- **g4_adapter_class_composition_full** is the last standard-mechanism P2 candidate at unblocked status. Next researcher should pick it.

## 7. Files produced

- `MATH.md` (updated with §0 skill + scope-preservation prose)
- `run_experiment.py` (rewritten, graceful-failure pattern)
- `results.json` (PROVISIONAL, 5 KCs `"untested"`)
- `PAPER.md` (prediction-vs-measurement table, 5 "not measured" rows)
- `REVIEW-adversarial.md` (researcher self-review placeholder)
- `LEARNINGS.md` (this file)
