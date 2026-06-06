# PAPER.md — exp_memento_cross_domain_transfer

## Verdict: KILLED (preempt, F#669 — ≥10 reuses; 3rd MEMENTO-cluster child)

This experiment was preempt-killed before any MLX code was written. The kill is structural, not empirical: K1906 requires Gemma-4-MEMENTO checkpoints trained on two distinct single-corpus training mixtures (GSM8K-only, MMLU-only) that do not exist. Parent `exp_memento_gemma4_replication` is `provisional` (F#685 — design-only, MEMENTO 2-stage SFT not executable via `mlx_lm.lora` CLI). The MEMENTO paper (Kontonis et al., arxiv:2604.09852) released checkpoints only for Qwen3 / Phi-4 / Olmo 3 on pooled OpenMementos 228K traces — no per-corpus separation, no Gemma 4 at any mixture.

A cross-training-domain KC additionally requires N=2 independent parent `_impl` training runs at single-corpus mixtures; parent's `_impl` validates a single pooled mixture. The cross-domain KC is strictly stronger than single-mixture measurement and remains preempt-blocked even under `P.status = supported` at one training mixture.

## Prediction vs measurement

| KC    | Prediction                                                                                   | Kind   | Measurement  | Verdict   |
| ----- | -------------------------------------------------------------------------------------------- | ------ | ------------ | --------- |
| K1906 | acc(MEMENTO-GSM8K, MMLU) / acc(MEMENTO-MMLU, MMLU) < 0.85 ⇒ no cross-training-domain transfer | target | not measured | untested  |

**K1906 is "not measured" because the accuracy ratio is 0/0 — both arms require per-corpus Gemma-4-MEMENTO checkpoints that do not exist.** Substituting a pooled-trained checkpoint, a base Gemma 4 cross-benchmark baseline, or prompt-level domain shift for training-corpus separation would be antipattern-t (silent objective swap from cross-TRAINING-domain transfer to cross-BENCHMARK eval or prompting-under-pooled-model). Substituting a non-Gemma-4 MEMENTO checkpoint (Qwen3/Phi-4/Olmo 3) would additionally be antipattern-m (proxy-model-substitution).

## Assumptions

- `exp_memento_gemma4_replication` will eventually be re-run to full scale via `exp_memento_gemma4_replication_impl` (P=3). That alone is necessary but **not sufficient** — the cross-domain KC requires N=2 additional `_impl` runs at single-corpus mixtures (GSM8K-only and MMLU-only) beyond the paper-default pooled OpenMementos.
- No redesign attempted this iteration to avoid the two-checkpoint requirement. Collapsing to single-checkpoint cross-benchmark evaluation would duplicate parent's K1800 (task accuracy drop < 5pp) and substitute the mechanism (pooled-MEMENTO benchmark performance ≠ cross-training-domain transfer) — antipattern-t risk.
- F#666 gating: K1906 is a target (task accuracy on MMLU in ratio form), so the KC set is target-gated trivially. No proxy to pair. KC-augmentation not needed at re-claim.

## Pattern continuation

F#669 reached its promotion threshold at F#698 (3rd reuse) and has since been reused extensively. This is the **3rd MEMENTO-cluster child preempt-KILL** after `exp_memento_compression_ratio_benchmark` (F#699, 1st cluster child, single-config) and `exp_memento_block_size_ablation` (2nd cluster child, scalar-sweep). All three share the same parent `exp_memento_gemma4_replication` (PROVISIONAL per F#685) and all three are F#666-compliant.

## Multi-parent-run sub-axis (2nd observation — watchlist)

| Observation | Experiment                              | Variant                           | N parent `_impl` required |
| ----------- | --------------------------------------- | --------------------------------- | ------------------------- |
| 1st         | `exp_memento_block_size_ablation`       | scalar-hyperparameter-sweep       | N=4 ({128,256,512,1024})  |
| **2nd**     | **`exp_memento_cross_domain_transfer`** | **categorical cross-corpus**      | **N=2 ({GSM8K, MMLU})**   |
| 3rd (cand)  | `exp_hedgehog_rank_ablation_r4_r8_r16`  | scalar-hyperparameter-sweep       | N=3 ({4,8,16})            |
| 3rd (cand)  | `exp_jepa_scale_sweep_5m_15m_50m`       | scalar-hyperparameter-sweep       | N=3 ({5M,15M,50M})        |

The underlying structural property: child KC requires M=N parent `_impl` checkpoints when parent's `_impl` validates only a single configuration. The specific axis (scalar hyperparameter vs categorical training corpus) is a sub-variant. Canonical promotion at 3rd observation per mem-pattern-triple-fire.

## Related

- **F#669** — defining precedent for preempt-KILL on target-unverified parent.
- **F#687 / F#698** — 2nd / 3rd F#669 reuse; promotion confirmed at F#698.
- **F#699** — 4th F#669 reuse; 1st MEMENTO-cluster child preempt-KILL.
- **(≥9th F#669 reuse)** — 2nd MEMENTO-cluster child preempt-KILL, 1st multi-parent-run sub-axis obs.
- **F#685** — parent PROVISIONAL.
- **F#666** — target-gated KC discipline (this experiment compliant: K1906 target-only).
- `exp_memento_gemma4_replication_impl` — P=3 impl-companion to parent; necessary-but-not-sufficient gate.
- Kontonis et al. arxiv:2604.09852 — MEMENTO paper.

## Unblock path

Re-claim this experiment when ALL of:

1. Parent `exp_memento_gemma4_replication` reaches `status=supported` via its `_impl` companion at the paper-default pooled OpenMementos training corpus.
2. Parent K1799 (KV reduction proxy) AND K1800 (task accuracy target) AND K1801 (KV-channel ablation target) AND K1802 (throughput target) SUPPORTED at full scale.
3. **N=2 additional parent `_impl` runs** at single-corpus training mixtures (GSM8K-only, MMLU-only) SUPPORTED — specifically the task-accuracy target K1800 at each per-corpus `_impl`.

No KC-augmentation needed. K1906 already target per F#666 trivially.

## Follow-up filed

None — preempt-structural kill does not spawn an `_impl` companion (per F#687/F#698/F#699 precedent + reviewer.md §5). The unblock is parent-external and requires 2 additional per-corpus `_impl` runs outside this experiment's scope.
