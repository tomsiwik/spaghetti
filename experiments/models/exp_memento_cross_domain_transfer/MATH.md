# MATH.md — exp_memento_cross_domain_transfer (PREEMPT-KILL)

## Verdict: PREEMPT-KILL

This experiment is preempt-killed per **Finding #669** (preempt-child-KCs-require-parent-target-claim-unverified) before any code was run. No `run_experiment.py` MLX implementation is attempted because the parent dependency is in a state that makes the KC structurally untestable AND the experiment additionally requires *two* distinct trained checkpoints (one per training domain) that do not exist.

This is the **3rd MEMENTO-cluster preempt-KILL** in the drain window, following:
1. `exp_memento_compression_ratio_benchmark` (F#699, 4th F#669 reuse — canonical single-config).
2. `exp_memento_block_size_ablation` (≥9th F#669 reuse — 1st observation of the multi-parent-run sub-axis, scalar-hyperparameter-sweep variant).
3. (this) `exp_memento_cross_domain_transfer` — 2nd observation of the multi-parent-run sub-axis, **cross-training-domain variant**.

## §0 Platform / skills / model pins

Included for completeness even though no platform code is executed.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per `PLAN.md` Part 2). **Not invoked** — no MLX code written; honest disclosure per reviewer checklist item (m2).
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627). **Not loaded.**
- Adapter targets: N/A — MEMENTO is 2-stage SFT + block-mask attention, not LoRA. Block size default 512 per paper.
- Parent dependency: `exp_memento_gemma4_replication` (status `provisional`, F#685).
- Sibling precedents: `exp_memento_compression_ratio_benchmark` (F#699), `exp_memento_block_size_ablation` (≥9th F#669 reuse).
- Datasets: GSM8K (training + eval) and MMLU (training + eval). Neither is loaded. Paper tested MEMENTO on GSM8K-Hard, MMLU, AIME24 — but at fixed training mixture, not cross-domain transfer.

## §1 Preempt-KILL theorem

**Theorem (inter-experiment target unverifiability, cross-training-domain variant).** Let `C` denote child experiment `exp_memento_cross_domain_transfer` with kill criterion K = {K1906 (target: MEMENTO-GSM8K accuracy on MMLU < 85% of MEMENTO-MMLU accuracy on MMLU)}. Let `P` denote parent experiment `exp_memento_gemma4_replication`.

K1906 requires **two** Gemma 4 models, each trained per the MEMENTO 2-stage SFT recipe with block-mask attention, on **different training mixtures**:
- `M_GSM8K`: MEMENTO-trained on GSM8K (segment-and-summarize traces over math/reasoning content).
- `M_MMLU`: MEMENTO-trained on MMLU (segment-and-summarize traces over encyclopedic/knowledge content).

Evaluation is the ratio `acc(M_GSM8K, MMLU) / acc(M_MMLU, MMLU)` compared to the 0.85 threshold. Both arms of the ratio require a trained MEMENTO checkpoint on its respective corpus:

1. **No public MEMENTO checkpoint exists for Gemma 4 at any training mixture.** MEMENTO paper (Kontonis et al., arxiv:2604.09852, Apr 2026) released checkpoints for Qwen3 / Phi-4 / Olmo 3 only, on the paper's OpenMementos dataset (a fixed 228K-trace mixture) — not separated per-target-benchmark.
2. **A cross-domain-transfer KC requires N=2 independent MEMENTO training runs of parent's `_impl`** (`exp_memento_gemma4_replication_impl`, P=3): one on GSM8K, one on MMLU. Each training run is 6-10h on M5 Pro 48GB per F#685 scale estimate; two runs is 12-20h, well beyond the micro-scale budget declared here and exceeding the researcher-hat 30-min / 40-tool-call cap.
3. Parent's `_impl` validates a **single** training mixture (the paper's OpenMementos default). It does not bind `M_GSM8K` or `M_MMLU` separately. The cross-domain KC is therefore *strictly stronger* than parent's `_impl` — it needs two distinct runs at two distinct training corpora.

If `P.status ∈ {provisional, open}` — i.e. no Gemma-4-MEMENTO checkpoint exists at *any* training mixture — then:

- **K1906** ("MEMENTO-GSM8K accuracy on MMLU < 85% of MEMENTO-MMLU accuracy"): both terms of the ratio are undefined. `acc(M_GSM8K, MMLU)` requires a GSM8K-trained MEMENTO that does not exist; `acc(M_MMLU, MMLU)` requires an MMLU-trained MEMENTO that does not exist. The ratio is `0/0` — unidentifiable.

Additionally, even under the stronger pre-condition `P.status = supported` at the paper-default mixed training corpus (OpenMementos), K1906 remains untestable without two *new* `_impl` training runs at two distinct single-corpus mixtures. Parent's `_impl` validates a pooled mixture; cross-domain transfer requires corpus-separated training runs outside parent's scope.

∴ Testing K1906 while `P.status ≠ supported|proven` produces an unidentifiable sample; and even at `P.status = supported` at one training mixture, cross-domain KCs remain unidentifiable without N=2 additional per-corpus `_impl` runs. **QED.**

### §1.1 F#666 gating

- K1906 = **target** (task accuracy on MMLU is a behavioral quality metric; ratio form against a peer checkpoint is a calibrated comparison to a trained model's own domain performance, not an uncalibrated round number).
- No proxy KC present. The KC set is therefore F#666-compliant **trivially** — F#666 requires every *proxy* KC to be paired with a target KC; a target-only KC set satisfies this by vacuous quantification.

However, the KC set is sparse (N=1). A target-only single-KC design is defensible here because the claim is strictly behavioral (does cross-domain transfer survive?) and a compression-ratio or routing-accuracy proxy would add no information the target doesn't already measure. Not an F#666 compound block; not a tautological proxy block.

### §1.2 Multi-parent-run sub-axis (2nd observation)

The sub-axis **multi-parent-run-KC-strictly-stronger-than-single-config** was first observed in `exp_memento_block_size_ablation` (scalar-hyperparameter-sweep variant, 1st instance). This experiment is the **2nd instance**, in a distinct variant:

| Dimension                        | 1st obs (block_size_ablation)                | 2nd obs (this, cross_domain_transfer)       |
| -------------------------------- | -------------------------------------------- | ------------------------------------------- |
| Sweep kind                       | scalar hyperparameter (block size)           | categorical training-corpus (GSM8K / MMLU)  |
| # parent `_impl` runs required   | N=4 ({128, 256, 512, 1024})                  | N=2 ({GSM8K-only, MMLU-only})               |
| Inner axis                       | training-time knob                           | training-data choice                        |
| Paper default coverage           | 512 only                                     | pooled OpenMementos (neither pure corpus)   |
| Unblock tightening               | N=4 parent `_impl` runs SUPPORTED            | N=2 parent `_impl` runs SUPPORTED           |

The underlying structural property is the same: the child KC requires access to M = N parent `_impl` checkpoints, where parent's `_impl` validates only a single configuration. The specific axis (scalar vs categorical) is a sub-variant; the preempt logic is identical.

**Sub-axis state:** 2nd observation, watchlist. Canonical promotion at 3rd observation per mem-pattern-triple-fire. Candidate 3rd instances listed in §4 of block_size_ablation LEARNINGS.md (`exp_hedgehog_rank_ablation_r4_r8_r16`, `exp_jepa_scale_sweep_5m_15m_50m`), both eligible if claimed under same parent-state conditions.

## §2 Prior art

- **F#669** (2026-04-19) — defining precedent for preempt-KILL on target-unverified parent.
- **F#685** (2026-04-23) — parent PROVISIONAL: design-only, 4 target-gated KCs untested, MEMENTO 2-stage SFT + block-mask attention not executable via `mlx_lm.lora` CLI.
- **F#699** (2026-04-24) — 4th F#669 reuse, **1st MEMENTO-cluster** child preempt-KILL: `exp_memento_compression_ratio_benchmark`. F#666-compliant KC set (proxy + target), same parent.
- **≥9th F#669 reuse** (2026-04-24) — **2nd MEMENTO-cluster** child preempt-KILL: `exp_memento_block_size_ablation`. F#666-compliant. 1st observation of multi-parent-run sub-axis (scalar-sweep variant).
- **F#698** (2026-04-24) — 3rd F#669 reuse + first F#666 compound sub-case.
- **F#666** — target-gated KC discipline. This experiment satisfies F#666 trivially (K1906 target-only).
- Kontonis et al. arxiv:2604.09852 — MEMENTO paper. Authors released Qwen3 / Phi-4 / Olmo 3 checkpoints on pooled OpenMementos (228K traces); no per-corpus (GSM8K-only, MMLU-only) separation; no Gemma 4 at any mixture.

## §3 Predictions (registered, not measured)

All KC states are **"untested (preempt-blocked)"**:

| KC    | Claim                                                                               | Kind   | Measurement status                     |
| ----- | ----------------------------------------------------------------------------------- | ------ | -------------------------------------- |
| K1906 | acc(MEMENTO-GSM8K, MMLU) / acc(MEMENTO-MMLU, MMLU) < 0.85 ⇒ no cross-domain transfer | target | untested (preempt-blocked, F#669)      |

## §4 Unblock condition

Re-claimable when **both** of:

1. Parent `exp_memento_gemma4_replication` reaches `status=supported` at full scale via its `_impl` companion (already filed P=3 as `exp_memento_gemma4_replication_impl`), with K1799+K1800+K1801+K1802 SUPPORTED at the pooled OpenMementos training corpus.
2. *Additionally* (per §1.2): N=2 parent `_impl` training runs exist at **single-corpus** training mixtures — one on GSM8K-only traces, one on MMLU-only traces — with K1800 (task accuracy drop < 5pp vs base) SUPPORTED on each evaluation set. These N=2 additional `_impl` runs are outside the current dependency graph and would need to be filed as `exp_memento_gemma4_replication_impl_gsm8k` and `exp_memento_gemma4_replication_impl_mmlu` (or an equivalent corpus-separated `_impl`).

**No KC-augmentation needed** at re-claim: K1906 is already a target per F#666. A compression-ratio or routing-accuracy proxy would add no information about the behavioral cross-domain transfer claim.

Alternatively, the experiment scope could be **reduced** at re-claim to evaluate the paper-default pooled MEMENTO on both GSM8K and MMLU (a single-checkpoint cross-benchmark evaluation, not a cross-training-domain transfer test). That would collapse to a subset of parent's K1800 (evaluated on two benchmarks) and yield no information about domain-specificity of the block-masking mechanism — antipattern-t risk (silent objective swap from "cross-training-domain transfer" to "single-model cross-benchmark eval").

## §5 Follow-up

No `_impl` companion filed — preempt-structural kill is self-contained per F#687/F#698/F#699 precedent + reviewer.md §5. The unblock condition is parent-external: parent's existing `exp_memento_gemma4_replication_impl` (P=3) is a necessary-but-not-sufficient gate; additional per-corpus `_impl` runs would be needed even post-parent-SUPPORTED.

## §6 Scope integrity

No silent objective swap (antipattern-t): this scaffold does NOT attempt:
- Evaluating base Gemma 4 (no MEMENTO compression) on GSM8K and MMLU and calling the ratio a "cross-domain transfer" result (would measure base model cross-benchmark stability, not MEMENTO mechanism transfer — antipattern-t).
- Loading the paper's pooled OpenMementos checkpoint (Qwen3/Phi-4/Olmo 3) and re-labeling it a "GSM8K-trained" or "MMLU-trained" checkpoint because the mixture contains math/knowledge traces (substitutes pooled for per-corpus — antipattern-t AND antipattern-m proxy-model if Qwen3/Phi-4/Olmo 3 used instead of Gemma 4).
- Prompt-level "domain shift" (few-shotting GSM8K prompts on a pooled checkpoint then evaluating on MMLU) as a substitute for training-corpus separation (substitutes prompting for training — antipattern-t).
- Using text-level summarization of GSM8K/MMLU as a proxy for MEMENTO block-mask attention (substitutes mechanism — antipattern-t).

All four shortcuts would replace the cross-training-domain mechanism the KC measures with a proxy phenomenon.

## §7 Anti-pattern scan

Composition-math: N/A (no composition).
LORA_SCALE: N/A (no LoRA).
shutil.copy: N/A (no code).
Hardcoded `"pass": True`: N/A (no code, `all_pass: false` written).
Eval truncation producing base=0%: N/A (no eval).
Proxy-model substitution: explicitly rejected in §6.
KC measures wrong object: K1906 correctly identifies cross-training-domain MEMENTO behavior (not a proxy), but the mechanism that produces it doesn't exist → preempt-KILL.
N=smoke reported as full: N/A (no N; `is_smoke: false`).
Tautological routing: N/A (no routing in this experiment).
Thinking-mode truncation: N/A (no eval).
File-existence cache: N/A (no code).
Copy-paste scaffolding: Scaffold derived from `exp_memento_block_size_ablation` (1st multi-parent-run observation) but variant-specific sections (cross-training-domain vs scalar-hyperparameter-sweep) are rewritten, not copy-pasted. KCs and parent-specific fields distinct.
