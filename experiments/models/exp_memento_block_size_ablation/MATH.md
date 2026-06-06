# MATH.md — exp_memento_block_size_ablation (PREEMPT-KILL)

## Verdict: PREEMPT-KILL

This experiment is preempt-killed per **Finding #669** (preempt-child-KCs-require-parent-target-claim-unverified) before any code was run. Rationale derived below; no `run_experiment.py` MLX implementation is attempted because the parent dependency is in a state that makes every KC structurally untestable AND the experiment additionally requires *multiple* trained checkpoints that do not exist.

This is a 2nd MEMENTO-cluster preempt-KILL following `exp_memento_compression_ratio_benchmark` (F#699, 4th F#669 reuse). F#669 has since reached ≥9 reuses (F#669, F#670/671, F#672, F#687, F#688, F#689, F#698, F#699, F#727, triple-fires in F#728-730); the routing is a routine drain operation.

## §0 Platform / skills / model pins

Included for completeness even though no platform code is executed.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per `PLAN.md` Part 2). **Not invoked** — no MLX code written; honest disclosure per reviewer checklist item (m2).
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627). **Not loaded.**
- Adapter targets: would be `v_proj + o_proj` (per F#627) for any Gemma 4 LoRA work, but this experiment is not LoRA — block size is a hyperparameter of the MEMENTO 2-stage SFT + block-mask attention recipe.
- Parent dependency: `exp_memento_gemma4_replication` (status `provisional`, F#685).
- Sibling precedent: `exp_memento_compression_ratio_benchmark` (F#699, 4th F#669 reuse).
- Dataset: unspecified in notes; paper uses GSM8K-Hard + AIME24 + MMLU. Not loaded.

## §1 Preempt-KILL theorem

**Theorem (inter-experiment target unverifiability, MEMENTO hyperparameter-sweep variant).** Let `C` denote child experiment `exp_memento_block_size_ablation` with kill criteria K = {K1904 (proxy: block size < 256 ⇒ compression ratio < 2x), K1905 (target: block size > 512 ⇒ accuracy < 80% of full-context)}. Let `P` denote parent experiment `exp_memento_gemma4_replication`.

Both KCs require **a Gemma 4 model trained per the MEMENTO 2-stage SFT recipe with block-mask attention at a specified block size** to exist. The block size itself is a training-time hyperparameter of the MEMENTO recipe — not a run-time knob applied to a fixed model. The notes field ("Sweep the block size parameter in MEMENTO compression") implicitly assumes such models exist per block size, but:

1. No public MEMENTO checkpoint exists for Gemma 4 at any block size. MEMENTO paper (Kontonis et al., arxiv:2604.09852, Apr 2026) released checkpoints for Qwen3 / Phi-4 / Olmo 3 only — and those use a fixed block size (paper default 512), not a sweep.
2. A block-size sweep over {128, 256, 512, 1024} requires **four independent** MEMENTO training runs of parent's `_impl` (`exp_memento_gemma4_replication_impl`, P=3). Each training run is 6-10h on M5 Pro 48GB per F#685; four runs is 24-40h, well beyond the micro-scale budget declared here and exceeding the researcher-hat 30-min / 40-tool-call cap.

If `P.status ∈ {provisional, open}` — i.e. no Gemma-4-MEMENTO checkpoint exists at *any* block size — then:

- **K1904** ("block size < 256 ⇒ compression ratio < 2x"): the antecedent is unsatisfiable because no trained MEMENTO model exists at block size 128 or 256 on Gemma 4. Without a compressing model, "compression ratio" is vacuous (1.0x by uncompressed/uncompressed identity).
- **K1905** ("block size > 512 ⇒ accuracy < 80% of full-context"): the antecedent is unsatisfiable for the same reason; no trained MEMENTO model exists at block size 1024 on Gemma 4. The "compressed-context" arm is undefined absent the block-mask attention loop with mementos in KV.

Additionally, even under the stronger pre-condition `P.status = supported` at one block size (e.g. 512, the paper default), the KC K1904/K1905 still require *different* block sizes whose training runs are outside parent's scope. Parent's `_impl` validates a single block-size configuration; a sweep requires new `_impl` work per block size.

∴ ∀ k ∈ K: testing `k` while `P.status ≠ supported|proven` produces an unidentifiable sample; and even at `P.status = supported` at one block size, off-size KCs remain unidentifiable without additional `_impl` work. **QED.**

### §1.1 F#666 gating

- K1904 = proxy (compression ratio is a structural/efficiency metric; thresholds 256/2x are behaviorally uncalibrated — no evidence that 1.9x is infeasible vs 2.1x feasible at any deployment constraint).
- K1905 = quasi-target (accuracy vs full-context is a task-quality metric, but the 80% threshold is uncalibrated; paper reports ≈5pp drops at default block size, far less than 20%, so 80% as a kill threshold is a made-up round number).

The KC set is F#666-compliant in form (one proxy + one quasi-target), so no compound F#666 block fires. The preempt-KILL is a clean F#669 reuse on parent-target-unverification, analogous to F#699.

### §1.2 Hyperparameter-sweep sub-axis (new-ish observation)

F#669 canonical sub-cases previously mapped:
- Single-parent parent-unverified (F#669 / F#687 / F#699)
- Dual-parent disjunctive (F#689)
- Triple-parent disjunctive (F#688)
- F#666 compound block (F#698)
- Same-parent repeat blocker (F#727+)

This experiment adds a 6th sub-axis candidate: **hyperparameter-sweep on PROVISIONAL parent**. A sweep KC requires N ≥ 2 trained checkpoints (one per sweep value), but parent's `_impl` validates a single configuration. The sweep KC is therefore *strictly stronger* than a single-config measurement KC — it is preempt-blocked even under `P.status = supported` at one config. This is a formal tightening of the F#669 precondition: the unblock condition must now read "N parent `_impl` runs at each sweep value SUPPORTED", not just "parent SUPPORTED".

1st observation; not yet promotion-eligible (promotion per canonical threshold at 3rd instance per mem-pattern-triple-fire).

## §2 Prior art

- **F#669** (2026-04-19) — defining precedent for preempt-KILL on target-unverified parent.
- **F#685** (2026-04-23) — parent PROVISIONAL: design-only, 4 target-gated KCs untested, MEMENTO 2-stage SFT + block-mask attention not executable via `mlx_lm.lora` CLI.
- **F#699** (2026-04-24) — 4th F#669 reuse, 1st MEMENTO-cluster child preempt-KILL: `exp_memento_compression_ratio_benchmark`. F#666-compliant KC set (proxy + target), same parent.
- **F#698** (2026-04-24) — 3rd F#669 reuse + first F#666 compound sub-case.
- **F#666** — target-gated KC discipline. This experiment satisfies F#666 by construction (K1905 quasi-target).
- Kontonis et al. arxiv:2604.09852 — MEMENTO paper. Authors released Qwen3 / Phi-4 / Olmo 3 checkpoints at a fixed block size; no Gemma 4 checkpoint at any block size.

## §3 Predictions (registered, not measured)

All KC states are **"untested (preempt-blocked)"**:

| KC    | Claim                                                                  | Kind        | Measurement status                     |
| ----- | ---------------------------------------------------------------------- | ----------- | -------------------------------------- |
| K1904 | Block size < 256 ⇒ MEMENTO compression ratio < 2x (too fine-grained)   | proxy       | untested (preempt-blocked, F#669)      |
| K1905 | Block size > 512 ⇒ compressed-context accuracy < 80% of full-context   | quasi-target| untested (preempt-blocked, F#669)      |

## §4 Unblock condition

Re-claimable when **both** of:

1. Parent `exp_memento_gemma4_replication` reaches `status=supported` at full scale via its `_impl` companion (already filed P=3 as `exp_memento_gemma4_replication_impl`), with K1799+K1800+K1801+K1802 SUPPORTED.
2. *Additionally* (new requirement per §1.2): N=4 parent `_impl` training runs exist at block sizes ∈ {128, 256, 512, 1024}, not just the single paper-default 512. These N-1 additional `_impl` runs are outside the current dependency graph and would need to be filed as `exp_memento_gemma4_replication_impl_bs{128,256,1024}` or an equivalent sweep-capable `_impl`.

Alternatively, the experiment scope could be **reduced** at re-claim to measure only block size 512 vs full-context (a single-config sanity check, not a sweep). That would collapse to a trivial subset of parent's K1800 (GSM8K drop < 5pp) and yield no new information — antipattern-t risk.

**No KC-augmentation needed** (unlike F#698): K1905 already provides a quasi-target gate per F#666.

## §5 Follow-up

No `_impl` companion filed — preempt-structural kill is self-contained per F#687/F#698/F#699 precedent + reviewer.md §5. The unblock condition is parent-external: parent's existing `exp_memento_gemma4_replication_impl` (P=3) is a necessary-but-not-sufficient gate; additional per-block-size `_impl` runs would be needed even post-parent-SUPPORTED.

## §6 Scope integrity

No silent objective swap (antipattern-t): this scaffold does NOT attempt:
- Measuring base Gemma 4 KV usage at different context-chunking block sizes (would not be MEMENTO block-mask attention).
- Measuring compression ratios of text-level chunking (substitutes text-channel for KV-channel — antipattern-t).
- Measuring accuracy drops at shorter context windows labeled as "block size = 128" (substitutes truncation for MEMENTO mechanism — antipattern-t).
- Proxy-model substitution (loading Qwen3-MEMENTO to "demonstrate" the sweep pattern then claiming cross-model — antipattern-m).

All three shortcuts would replace the MEMENTO mechanism the KCs measure with a proxy phenomenon.

## §7 Anti-pattern scan

Composition-math: N/A (no composition).
LORA_SCALE: N/A (no LoRA).
shutil.copy: N/A (no code).
Hardcoded `"pass": True`: N/A (no code).
Eval truncation producing base=0%: N/A (no eval).
Proxy-model substitution: explicitly rejected in §6.
KC measures wrong object: K1904/K1905 correctly identify MEMENTO block-mask behavior (not a proxy), but the mechanism that produces it doesn't exist → preempt-KILL.
N=smoke reported as full: N/A (no N).
