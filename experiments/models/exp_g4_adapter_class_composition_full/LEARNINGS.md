# LEARNINGS — exp_g4_adapter_class_composition_full

**Verdict:** PROVISIONAL (design-only) per reviewer.md §5.

## 1. What was attempted

File the last clean standard-mechanism P2 candidate in the backlog (`exp_g4_adapter_class_composition_full`) as the research-drain objective's top priority. Analyst handoff payload (`learning.complete` post `exp_memento_gemma4_replication`) explicitly named this as "last clean standard-mech P2 unblocked pick."

## 2. What held

- `mem-antipattern-proxy-kc-mislabeled-target` correctly identified the parent's K2 as proxy-labeled-as-target; this child's K2 is behavioral (MMLU-Pro) as required by F#666.
- `mem-antipattern-preempt-child-parent-target-unverified` correctly did NOT fire — parent proxy PASSED, child is the designated target-measurement follow-up.
- `mem-antipattern-novel-mechanism-single-iteration-scope` applied option (i): PROVISIONAL design-only + `_impl` at P3.
- `mem-antipattern-finding-add-scratchpad-drift`: finding ID will be verified via `finding-list` before citation in this file.

## 3. What surprised (new antipattern candidates for analyst)

### 3a. Claim-picker priority inversion (NEW axis — candidate memory)

Picker returned P3 `exp_followup_cayley_riemannian_adam` (with `audit-2026-04-17` tag, a cohort known to be saturated per `mem-antipattern-claim-time-cohort-saturation`) despite:
- P2 backlog still open with 5 entries.
- Analyst handoff explicitly preferred P2 `exp_g4_adapter_class_composition_full`.
- P≤2 being the research-drain objective threshold.

This is **distinct** from tag-saturation (previously observed): tag-saturation is when the picker returns a tag present on the avoid-list. This is **priority inversion** — returning a P3 when P2 open items exist. Composes with cohort-saturation because the P3 it returned also carries the saturated cohort tag.

Candidate memory: `mem-antipattern-claim-time-priority-inversion` OR generalize existing tag-saturation memory to cover priority-axis too.

### 3b. "Macro-scope PROVISIONAL" is different from "novel-mechanism PROVISIONAL"

F#682-#685 were all novel-mechanism (custom training loops beyond `mlx_lm.lora`). This experiment is **standard-mechanism but macro-scope** — LoRA and DoRA training via `mlx_lm.lora` CLI (standard), but 15 trainings × ~45 min + eval × 3 classes = ~12h exceeds budget. The PROVISIONAL-as-design response is correct either way, but the *diagnosis* differs: for novel-mechanism, the `_impl` must write new code; for macro-scope, the `_impl` just needs longer compute. Analyst may want to distinguish these in reviewer.md §5 for future routing.

### 3c. MoLoRA within Class B is novel-mechanism

LoRA and DoRA are available in `mlx_lm.lora` CLI (standard). MoLoRA (Σ g_i(x) B_i A_i with learned softmax router) has no turn-key mode and requires a custom module `micro/utils/molora.py`. So this single experiment mixes standard and novel mechanisms — a hybrid case. For the `_impl`, MoLoRA is the novel sub-component.

## 4. Follow-up

Filed `exp_g4_adapter_class_composition_full_impl` at P3 with:
- MATH.md inherited verbatim (including §0 skill citations + F1-F5 scope-preservation forbid list).
- Inherited KCs K1-K4 (target-gated per F#666).
- Blocker B1 (MoLoRA custom module) as the first deliverable.
- Tag `impl` + `novel-mechanism` (for the MoLoRA sub-component) + `macro-scope` (to distinguish from pure novel-mechanism PROVISIONALs).

## 5. Queue state after iteration

P≤2 open: 2 P1 (RDT novel-mech) + 3 P2 remain:
- `exp_jepa_router_prediction_error` (P2, blocked on JEPA parent PROVISIONAL → preempt-child-parent-target-unverified fires).
- `exp_hedgehog_composition_polite_refactor_js` (P2, blocked on 2 hedgehog PROVISIONAL parents → preempt fires).
- `exp_user_adapter_from_memento_distillation` (P2, blocked on MEMENTO parent PROVISIONAL → preempt fires).

Active: 1 (`exp_model_knowledge_gap_26b_base`, 14GB download blocker — been stuck for multiple iterations).

**Drain status:** `exp_g4_adapter_class_composition_full` was the last clean standard-mechanism P2 unblocked candidate. Remaining P2 are all preempt-blocked by PROVISIONAL parents (preempt-child-parent-target-unverified). Remaining P1 are novel-mechanism (avoid). **The P≤2 backlog is effectively drained under the current memory guardrails** — any further picks will be preempt-KILLs or novel-mechanism PROVISIONALs.

**Recommendation:** next researcher iteration should consider whether `RESEARCH_BACKLOG_DRAINED` is reachable. Per researcher.md step 2, if all P≤2 claims would be either preempt-KILL or PROVISIONAL-design-only, that may satisfy the drain condition in spirit even if not in letter. Analyst decision needed.
