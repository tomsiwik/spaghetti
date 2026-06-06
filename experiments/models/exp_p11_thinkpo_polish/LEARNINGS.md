# LEARNINGS: P11.C0 — ThinkPO Polish

**Status**: KILLED (preemptive / upstream-dependency, 2026-04-17)
**Supersedes**: 2026-04-14 design-era learnings ("Design approved PROCEED") — the design
was sound in vacuum but the reference policy π_ref became structurally broken when B0
was killed on 2026-04-17.

## Core Finding
C0 cannot falsify anything meaningful because its reference policy is the GRPO adapter
from B0 (`exp_p11_grpo_reasoning_adapter`), which was killed 2026-04-17 with −15.3pp
MMLU-Pro regression (57.1%→41.8%) and −71% thinking suppression (2819→816 chars). All
three of C0's kill criteria (K1499/1500/1501) are stated *relative to GRPO*, so even a
three-for-three pass would leave the stack ~13pp below Gemma 4 E4B base — metric-pass /
behavioral-fail by MEMORY.md behavioral-outcomes rule.

## Why
Three independent arguments (Review §Kill Robustness):
1. **Protocol bug, not hyperparameter miss.** B0's root cause is `mlx_lm.lora` receiving
   `<|channel>thought` tokens as literal cross-entropy text targets. C0 is blind to this
   at the preference-pair sampling stage — no ThinkPO loop fixes a regressed π_ref.
2. **Fallback path inverted vs measurement.** `run_experiment.py:544-571` branches on
   `if grpo_len > base_len`; B0 produced 816 chars < base's 2819, so the fallback
   collapses to zero pairs. If it collected pairs with the inverted assumption, DPO
   would train toward base — *undoing* GRPO, still no ThinkPO signal.
3. **LoRA-on-LoRA on regressed subspace.** `:361-372` applies r=4 LoRA on top of the
   already-hooked GRPO LoRA (`:527`). Trainable params become
   `LoRA_thinkpo(LoRA_grpo(W_base))`; even a perfect DPO update polishes a subspace that
   starts below base.

## Implications for Next Experiment
- **D0 (`exp_p11_meta_r1_metacognition`) and M0 (`exp_p11_full_pipeline_v2`) will inherit
  the same kill pattern** if claimed before B0-v2 lands. D0 depends on B0 directly; M0's
  pipeline embeds ThinkPO/GRPO. Precedent: P11.G0 (same day) was preemptively killed on
  identical upstream-regression reasoning. Researcher should preemptive-kill D0/M0 the
  same way unless B0-v2 is already `supported`.
- **B0-v2 is the single unblocking experiment** for the entire P11 reasoning-adapter
  chain (C0, D0, M0). Design requirements in B0 PAPER.md §Unblock Path: replace
  `mlx_lm.lora` thinking-channel handling with custom GRPO loop, or plain-prompt SFT, or
  chat-template fork.
- **C0 redesign (post-B0-v2)**: fix the `mlx_lm save` antipattern (save LoRA-only via
  `mx.savez(tree_flatten(model.trainable_parameters()))`, not `from mlx_lm import save`);
  delete the fallback path (if phase 1 yields <2 pairs, kill — don't invert directions);
  consider Gemma 4 base as π_ref to isolate DPO's contribution from stacking effects.
- **Cascade audit rule**: before claiming any experiment whose `blocked-by` dependency
  is killed, check whether the kill is a protocol bug (cascades) or a hyperparameter
  miss (may not cascade). B0 is protocol — cascades to every stack that uses its adapter
  as π_ref or as a warmstart.

## No New Finding / No Paper Ref
- Preemptive-kill-on-upstream-regression pattern already established by P11.G0
  (2026-04-17); no new structural finding to promote.
- No paper ref added — the failure is upstream (B0 protocol bug), not explainable by a
  new literature citation; B0 LEARNINGS already cites DPO (2305.18290) and ThinkPO
  (2502.13173), which remain the correct references once π_ref is healthy.
