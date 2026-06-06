# MATH.md — exp_memento_compression_ratio_benchmark (PREEMPT-KILL)

## Verdict: PREEMPT-KILL

This experiment is preempt-killed per **Finding #669** (preempt-child-KCs-require-parent-target-claim-unverified) before any code was run. Rationale derived below; no `run_experiment.py` MLX implementation is attempted because the parent dependency is in a state that makes every KC structurally untestable.

This is the **4th reuse** of the F#669 pattern (F#669 → F#687 → F#698 → this). F#698 confirmed promotion of the sub-axis at the 3rd reuse; 4th reuse re-confirms it as canonical routing.

## §0 Platform / skills / model pins

Included for completeness even though no platform code is executed.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per `PLAN.md` Part 2). **Not invoked** — no MLX code written; honest disclosure per reviewer checklist item (m2).
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627). **Not loaded.**
- Adapter targets: would be `v_proj + o_proj` (per F#627) for any Gemma 4 LoRA work, but this experiment is not LoRA — it's a measurement on a 2-stage SFT MEMENTO-trained model that does not exist.
- Parent dependency: `exp_memento_gemma4_replication` (status `provisional`, F#685).

## §1 Preempt-KILL theorem

**Theorem (inter-experiment target unverifiability, MEMENTO-measurement variant).** Let `C` denote child experiment `exp_memento_compression_ratio_benchmark` with kill criteria K = {K1850 (proxy: MEMENTO compression ratio < 3x), K1851 (target: compressed-context GSM8K accuracy < 85% of full-context)}. Let `P` denote parent experiment `exp_memento_gemma4_replication`.

Both KCs require **a Gemma 4 model trained per the MEMENTO 2-stage SFT recipe with block-mask attention** to exist. The notes field claims "No dependency on full replication" but this is materially false: there is no public MEMENTO checkpoint for Gemma 4. The MEMENTO paper (Kontonis et al., arxiv:2604.09852, Apr 2026) released checkpoints for Qwen3 / Phi-4 / Olmo 3 only; Gemma 4 is not among them. The only path to a Gemma-4-MEMENTO model is via parent `P` reaching `supported` (or its `_impl` companion `exp_memento_gemma4_replication_impl` at P=3 landing).

If `P.status ∈ {provisional, open}` — i.e. no Gemma 4 MEMENTO checkpoint exists — then:

- **K1850** ("compression ratio < 3x"): the metric is undefined absent a model that performs compression. No model = no compression operation = no ratio. Loading the base Gemma 4 E4B and running it on GSM8K-Hard measures only base inference KV usage; there is no "compressed" arm to ratio against. Vacuous: 1.0x by trivial identity (uncompressed/uncompressed).

- **K1851** ("compressed-context accuracy < 85% of full-context"): the "compressed-context" arm requires the MEMENTO block-mask attention loop with mementos in the KV channel — this is precisely the mechanism parent `P` is supposed to validate but has not. Without a trained model that produces mementos, there is no "compressed-context" answer to evaluate. Substituting "shorter context window" for "MEMENTO compression" would be antipattern-t (silent objective swap); see §6.

∴ ∀ k ∈ K: testing `k` while `P.status ≠ supported|proven` produces an unidentifiable sample. **QED.**

### §1.1 No F#666 compound block

Unlike `exp_jepa_adapter_attention_output` (F#698) where the KC set was proxy-only and triggered a F#666 compound block, **this experiment's KC set is properly target-gated** per F#666:

- K1850 = proxy (compression ratio is a structural/efficiency metric)
- K1851 = target (GSM8K accuracy is a task quality target)

KILL would require BOTH to fail; SUPPORTED requires BOTH to pass. This is F#666-compliant out of the box. No KC-augmentation is needed at re-claim time — the only blocker is parent target-verification.

## §2 Prior art

- **F#669** (2026-04-19) established the preempt pattern for `exp_rdt_act_halting_throughput` over `exp_rdt_loop_lora_gemma4`.
- **F#687** (2026-04-23) 2nd reuse: `exp_jepa_router_prediction_error` over `exp_jepa_adapter_residual_stream`.
- **F#698** (2026-04-24) 3rd reuse: `exp_jepa_adapter_attention_output` over `exp_jepa_adapter_residual_stream` + first F#666 compound sub-case. Promotion threshold confirmed.
- **F#685** (2026-04-23) parent PROVISIONAL: design-only, 4 target-gated KCs untested, MEMENTO 2-stage SFT + block-mask attention not executable via `mlx_lm.lora` CLI.
- **F#666** — target-gated kill discipline. This experiment satisfies F#666 by construction (K1851 is target).
- Kontonis et al. arxiv:2604.09852 — MEMENTO paper. Authors released Qwen3 / Phi-4 / Olmo 3 checkpoints; not Gemma 4.

## §3 Predictions (registered, not measured)

All KC states are **"untested (preempt-blocked)"**:

| KC    | Claim                                                              | Kind   | Measurement status                      |
| ----- | ------------------------------------------------------------------ | ------ | --------------------------------------- |
| K1850 | MEMENTO compression ratio < 3x (not worth the SFT cost)            | proxy  | untested (preempt-blocked, F#669)       |
| K1851 | Compressed-context accuracy < 85% of full-context on GSM8K         | target | untested (preempt-blocked, F#669)       |

## §4 Unblock condition

Re-claimable when parent `exp_memento_gemma4_replication` reaches `status=supported` at full scale via its `_impl` companion `exp_memento_gemma4_replication_impl` (already filed P=3). Specifically:

1. Parent K1799 (KV reduction ≥ 2x proxy) and K1800 (GSM8K-Hard drop < 5pp target) BOTH SUPPORTED.
2. Parent K1801 (KV-channel ablation target) SUPPORTED — establishes mementos are doing real work, not text-channel artifacts.
3. Parent K1802 (throughput ≥ 1.3x target) SUPPORTED — establishes the model runs at usable speed for benchmarking.

At that point, a Gemma 4 MEMENTO checkpoint exists and both K1850 (ratio measurable) and K1851 (compressed-context accuracy measurable) become well-defined.

**No KC-augmentation needed** (unlike F#698 which required adding a target metric): K1851 already provides the target gate per F#666.

**Alternative unblock:** redesign child to perform measurement *jointly* with training (i.e. train MEMENTO + measure ratio in one experiment). Out of scope for drain window; this is `_impl`-class work and would essentially duplicate the parent's `_impl`.

## §5 Follow-up

No `_impl` companion filed — preempt-structural kill is self-contained per F#687/F#698 + reviewer.md §5. The unblock condition is external: parent's existing `exp_memento_gemma4_replication_impl` (P=3) is the gate. If that lands SUPPORTED, this child becomes immediately re-claimable without further KC modification.

This distinction between novel-mechanism PROVISIONAL (which mandates `_impl`) and preempt-structural KILL (which does NOT spawn `_impl`) is canonical per F#687 / F#698 / reviewer.md §5.

## §6 Scope integrity

No silent objective swap (antipattern-t): this scaffold does NOT attempt a "simpler" measurement (e.g. measuring base Gemma 4 KV usage and calling it a 1.0x compression baseline; or shortening the GSM8K context window and calling it "compressed-context"). Both shortcuts would substitute proxy phenomena for the MEMENTO mechanism the KCs measure.

Pre-registered KCs are preserved verbatim and marked `untested (preempt-blocked)`. KC text in DB (`experiment get`) matches MATH.md §3 verbatim — no post-claim KC mutation.

No `_impl` inline-file obligation (`mem-antipattern-impl-follow-up-delegation`): that antipattern applies to novel-mechanism PROVISIONAL only. Preempt-structural KILL is structurally distinct and explicitly does **not** spawn `_impl` per F#687/F#698 precedent.
