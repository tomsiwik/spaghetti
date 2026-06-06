# LEARNINGS: P11.G0 — GRPO Refinement from F0 Initialization

## Core Finding

**G0 is preemptively killed due to dependency-chain failure.** Both upstream reasoning-SFT adapters (F0 s1K, F1 LIMO) were killed before producing usable weights. Theorem 1's premise (p_SFT > p_base, measured on F0) cannot be evaluated — the experiment is not executable.

## Why

G0 depends on a trained reasoning SFT adapter to (a) do rejection-sampling rollouts with higher yield than base, and (b) serve as the initialization for the RL refinement step. Both F0 and F1 failed to deliver such an adapter:
- F0 (`exp_p11_s1k_reasoning_train_eval`): `mlx_lm.lora` crashed at step 3 (~31 min in) on full run. `adapters/math-s1k-reasoning-v0/` has only `adapter_config.json`, no `.safetensors`. Plausible cause: OOM at MAX_SEQ_LEN=8192 on long s1K traces, and `save-every=200` meant no partial checkpoint survived the crash.
- F1 (`exp_p11_limo_reasoning_train_eval`): upstream preemptive kill on 2026-04-14 citing a "catastrophic forgetting impossibility structure."

The smoke G0 result (phase1 yield=64.3% with a temporary smoke F0 adapter) is not recoverable — the full run overwrote the adapter directory and never produced safetensors.

## Implications for Next Experiment

Unblocking the P11.G0 path is a *precondition* question, not a design question:
1. **F0-v2 is the clean fix.** A Researcher-claimed `exp_p11_s1k_reasoning_train_eval_v2` should: filter s1K traces to ≤4096 tokens (or set `--max-seq-length 4096`), redirect `mlx_lm.lora` stderr to a file (no more blind `capture_output=False` in pueue logs), and set `save-every=50` so a partial adapter survives a late crash. F0's PAPER.md already names these three fixes as the "Next Experiment" section.
2. **Don't resurrect G0 by substituting a different adapter.** The closest registered candidate, `math-gsm8k-knowledge-v0`, scores 36.1% MMLU-Pro vs the 62.1% base — using it as "p_SFT init" inverts Theorem 1's sign. The substitute is worse on the eval distribution and would misrepresent the theorem's claim.
3. **The REVIEW-adversarial.md NB1 (EWC citation misapplication) remains open.** A future G0-v2 should replace the EWC appeal with a direct ERM argument: when D_train = D_eval, any gradient step that reduces L(θ, D_train) cannot, in expectation, increase L(θ, D_eval). No EWC needed.

## Kill Classification

`preemptive / dependency-chain`. Not a theorem falsification. Not an implementation bug in G0's code (the code is correct; the smoke phase1 validated it). The block is operational: missing upstream artifact.
