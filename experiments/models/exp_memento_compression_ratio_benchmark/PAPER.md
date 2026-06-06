# PAPER.md — exp_memento_compression_ratio_benchmark

## Verdict: KILLED (preempt, F#669 — 4th reuse)

This experiment was preempt-killed before any MLX code was written. The kill is structural, not empirical: both kill criteria require a Gemma-4-MEMENTO checkpoint that does not exist. Parent `exp_memento_gemma4_replication` is currently `provisional` (F#685 — design-only, MEMENTO 2-stage SFT not executable via `mlx_lm.lora` CLI, all 4 target-gated KCs untested). The MEMENTO paper (Kontonis et al., arxiv:2604.09852) released checkpoints only for Qwen3 / Phi-4 / Olmo 3 — Gemma 4 is not among them, so no public alternative exists either.

The experiment notes claimed "No dependency on full replication" but this is materially false: there is no Gemma-4-MEMENTO checkpoint to load.

## Prediction vs measurement

| KC    | Prediction                                                       | Kind   | Measurement  | Verdict   |
| ----- | ---------------------------------------------------------------- | ------ | ------------ | --------- |
| K1850 | MEMENTO compression ratio < 3x (not worth the SFT cost)          | proxy  | not measured | untested  |
| K1851 | Compressed-context accuracy < 85% of full-context on GSM8K       | target | not measured | untested  |

**Both rows are "not measured" because no Gemma-4-MEMENTO checkpoint exists.** Measuring against base Gemma 4 would yield a vacuous 1.0x ratio (uncompressed/uncompressed) and an undefined "compressed-context" arm — antipattern-t (silent objective swap) per reviewer checklist (t).

## Assumptions

- `exp_memento_gemma4_replication` will eventually be re-run to full scale via its `_impl` companion (already filed P=3 as `exp_memento_gemma4_replication_impl`). If that reaches `supported` with K1799+K1800+K1801+K1802 SUPPORTED, this experiment becomes immediately re-claimable.
- No redesign attempted this iteration to avoid the parent dependency (e.g. train MEMENTO + measure ratio in one experiment). Out of scope per drain objective; that would essentially duplicate parent's `_impl`.
- F#666 gating: K1851 is a target metric (GSM8K accuracy), so the KC set is properly target-gated out of the box. **No KC-augmentation needed at re-claim time** (unlike F#698 which required adding a target metric).

## Pattern continuation

F#698 (3rd reuse of F#669) confirmed the promotion of the preempt-child-parent-target-unverified sub-axis to standalone canonical finding-family. This is the **4th reuse** — the pattern is now established as a routine drain operation. Distinguishing factor from F#698: this child has a properly target-gated KC set (no F#666 compound block), so the unblock path is simpler.

## Related

- **F#669** — defining precedent for preempt-KILL on target-unverified parent.
- **F#687** — 2nd reuse.
- **F#698** — 3rd reuse + first F#666 compound sub-case.
- **F#685** — parent PROVISIONAL.
- **F#666** — target-gated KC discipline (this experiment is compliant: K1851 target).
- `exp_memento_gemma4_replication_impl` — P=3 impl-companion to parent; blocks re-claim of this child.
- Kontonis et al. arxiv:2604.09852 — MEMENTO paper.

## Unblock path

Re-claim this experiment when ALL of:

1. Parent `exp_memento_gemma4_replication` reaches `status=supported`.
2. Parent K1799 (KV reduction proxy) AND K1800 (GSM8K target) AND K1801 (KV-channel ablation target) AND K1802 (throughput target) SUPPORTED at full scale.

No KC-augmentation needed. K1851 already provides the target gate per F#666.

## Follow-up filed

None — preempt-structural kill does not spawn an `_impl` companion (per F#687/F#698 precedent + reviewer.md §5). The unblock is parent-external.
