# PAPER.md — exp_memento_block_size_ablation

## Verdict: KILLED (preempt, F#669 — ≥9 reuses; 2nd MEMENTO-cluster child)

This experiment was preempt-killed before any MLX code was written. The kill is structural, not empirical: both kill criteria require Gemma-4-MEMENTO checkpoints at swept block sizes {128, 256, 512, 1024} that do not exist. Parent `exp_memento_gemma4_replication` is `provisional` (F#685 — design-only, MEMENTO 2-stage SFT not executable via `mlx_lm.lora` CLI). The MEMENTO paper (Kontonis et al., arxiv:2604.09852) released checkpoints only for Qwen3 / Phi-4 / Olmo 3 at a fixed block size (paper default 512) — no Gemma 4 at any block size, and no existing model at any other block size.

A block-size sweep additionally requires N=4 independent parent `_impl` training runs; parent's `_impl` validates a single configuration. The sweep KC is strictly stronger than single-config measurement and remains preempt-blocked even under `P.status = supported` at one block size.

## Prediction vs measurement

| KC    | Prediction                                                                   | Kind         | Measurement  | Verdict   |
| ----- | ---------------------------------------------------------------------------- | ------------ | ------------ | --------- |
| K1904 | Block size < 256 ⇒ MEMENTO compression ratio < 2x (too fine-grained)         | proxy        | not measured | untested  |
| K1905 | Block size > 512 ⇒ compressed-context accuracy < 80% of full-context (coarse)| quasi-target | not measured | untested  |

**Both rows are "not measured" because no Gemma-4-MEMENTO checkpoint exists at any block size.** Measuring against base Gemma 4 would yield a vacuous 1.0x ratio (uncompressed/uncompressed) and an undefined "compressed-context" arm; substituting shorter context windows or text-level chunking for block-mask attention would be antipattern-t (silent objective swap).

## Assumptions

- `exp_memento_gemma4_replication` will eventually be re-run to full scale via `exp_memento_gemma4_replication_impl` (P=3). That alone is necessary but **not sufficient** — the sweep requires N-1 additional `_impl` runs at block sizes {128, 256, 1024} beyond the paper default 512.
- No redesign attempted this iteration to avoid the sweep requirement. Collapsing to single-config (block size 512) measurement would duplicate parent's K1800 (GSM8K drop < 5pp) and yield no new information — antipattern-t risk.
- F#666 gating: K1905 is a quasi-target (task accuracy on GSM8K/MMLU), so the KC set is target-gated in form. No KC-augmentation needed at re-claim (analogous to F#699, not F#698).

## Pattern continuation

F#669 reached its promotion threshold at F#698 (3rd reuse) and has since been reused in F#699, F#727-730, etc. This is the **2nd MEMENTO-cluster child preempt-KILL** after F#699 (`exp_memento_compression_ratio_benchmark`), same parent, F#666-compliant KC set. Distinguishing factor: this experiment is a hyperparameter **sweep**, not a single-config measurement — the sweep KC is strictly stronger and remains preempt-blocked even at parent-SUPPORTED-at-one-config.

## New sub-axis observation (1st instance, not yet promotion-eligible)

Hyperparameter-sweep on PROVISIONAL parent: a sweep KC over N values requires N independent parent `_impl` training runs. Parent's `_impl` validates a single configuration. Sweep KC is therefore strictly stronger than single-config measurement. This is a formal tightening of the F#669 precondition: unblock condition reads "N parent `_impl` runs at each sweep value SUPPORTED", not just "parent SUPPORTED".

1st observation in drain window; promotion per canonical threshold at 3rd instance.

## Related

- **F#669** — defining precedent for preempt-KILL on target-unverified parent.
- **F#687 / F#698** — 2nd / 3rd F#669 reuse; promotion confirmed at F#698.
- **F#699** — 4th F#669 reuse; **1st MEMENTO-cluster child preempt-KILL** (sibling).
- **F#685** — parent PROVISIONAL.
- **F#666** — target-gated KC discipline (this experiment is compliant: K1905 quasi-target).
- `exp_memento_gemma4_replication_impl` — P=3 impl-companion to parent; necessary-but-not-sufficient gate.
- Kontonis et al. arxiv:2604.09852 — MEMENTO paper.

## Unblock path

Re-claim this experiment when ALL of:

1. Parent `exp_memento_gemma4_replication` reaches `status=supported` via its `_impl` companion.
2. Parent K1799 (KV reduction proxy) AND K1800 (GSM8K target) AND K1801 (KV-channel ablation target) AND K1802 (throughput target) SUPPORTED at full scale.
3. **N=3 additional parent `_impl` runs** at block sizes {128, 256, 1024} SUPPORTED (beyond paper-default 512 validated in parent's `_impl`).

No KC-augmentation needed. K1905 already provides a quasi-target gate per F#666.

## Follow-up filed

None — preempt-structural kill does not spawn an `_impl` companion (per F#687/F#698/F#699 precedent + reviewer.md §5). The unblock is parent-external and requires 3 additional per-block-size `_impl` runs outside this experiment's scope.
