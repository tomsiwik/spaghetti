# MATH: exp_g4_routed_beats_base_think (PREEMPTIVE KILL)

**Verdict:** KILLED — preemptive (antipattern-017 cascade, 7th confirmed instance in 2 days).
**KC:** K1592 — "GSM8K +5pp AND MMLU-Pro ≥ base (with thinking mode enabled)."
**Status:** unmeasurable. The domain adapters required for routed composition do not exist on disk.

---

## Theorem 1 (Measurability of K1592 under missing adapter weights)

**Statement.** Let `A = {a_math, a_code, a_medical, a_legal, a_finance}` be the five
domain adapters referenced by `adapters/registry.json` for the routed Gemma 4 composition.
Let `W(a_i) = (A_i, B_i) ∈ R^(r × d_in) × R^(d_out × r)` be the LoRA weight pair for
adapter `a_i`. Define the routed forward pass

  `y = x + Σ_i α_i · (B_i A_i x)`   (eq. 1)

where `α_i` is the per-sample routing weight over adapter `i`. Then K1592 is measurable
iff `∀ i ∈ A: ||A_i||_F > 0 ∧ ||B_i||_F > 0`.

**Proof.** If `A_i = 0 ∨ B_i = 0` for some `i`, then `B_i A_i x = 0` and adapter `i`
contributes no signal regardless of `α_i`. With all 5 adapter weight tensors absent
(only `adapter_config.json` on disk, no `adapters.safetensors`), `y = x` for every
input. Under either branch of thinking mode, the routed generator and the base
generator consume the same prompt, pass it through the same base transformer, and
emit the same token distribution. Hence

  `Pr[acc_routed(GSM8K) ≠ acc_base(GSM8K)] = 0`
  `Pr[acc_routed(MMLU-Pro) ≠ acc_base(MMLU-Pro)] = 0`

K1592 is a conjunction "GSM8K +5pp AND MMLU-Pro ≥ base". Under routed ≡ base, the
first conjunct requires `Δ_gsm8k ≥ +0.05` but `Δ_gsm8k = 0`. The conjunction is
structurally false. ∎

## Theorem 2 (Observational equivalence of config-only stubs)

**Statement.** An adapter directory containing only `adapter_config.json` (no
`adapters.safetensors`) is observationally equivalent under `mlx_lm.load(model,
adapter_path=dir)` to not loading any adapter — whether the loader (a) crashes,
(b) warns and random-inits, or (c) silently runs the base model.

**Proof.** `adapter_config.json` contains only metadata (`lora_alpha`, `r`,
`target_modules`, `base_model_name_or_path`). It contains no weight tensors. The
loader must obtain `(A_i, B_i)` from (a) safetensors on disk, (b) random init,
or (c) noop. Case (a) → FileNotFoundError, no inference possible. Case (b) →
random noise, `E[B_i A_i x] = 0` (by independence of Gaussian init), variance
nonzero — but this is not "learned adapter semantics" (which is `Δ ≠ 0` in a
specific data-learned subspace); marginal accuracy = base accuracy up to noise.
Case (c) → `y = x` by direct construction. In all three cases,
`E[acc_routed − acc_base] = 0`. ∎

## Theorem 3 (Thinking mode is inert to absent adapters)

**Statement.** Thinking mode modifies the prompt with a "<think>…</think>" scaffold
and extends generation length; it does not modify the adapter forward pass (eq. 1).
Therefore, conditioning on thinking mode does not rescue the kill.

**Proof.** Thinking mode is prompt-level: the base transformer sees `[thinking_prefix]
∘ prompt` instead of `prompt`, and the decoder runs for more steps. The adapter
operator `x → x + Σ α_i B_i A_i x` is layer-internal and depends only on the hidden
state `x` flowing through attention/MLP at each layer. With `B_i A_i = 0` for all i
(Theorems 1 + 2), the operator reduces to the identity whether the input hidden state
came from a thinking-mode prompt or a direct prompt. Therefore, under 5/5 stub
adapters:

  `acc_routed(GSM8K, think) = acc_base(GSM8K, think)`
  `acc_routed(MMLU-Pro, think) = acc_base(MMLU-Pro, think)`

and K1592 remains unreachable. ∎

## Corollary (K1592 FAIL)

From Theorems 1, 2, 3: with 5/5 domain adapters stub-only on disk, the routed
composition is indistinguishable from the base model under thinking mode, so
`Δ_gsm8k_routed_vs_base = 0 < +0.05`. The "AND" clause collapses. K1592 = FAIL
by construction, independent of sample size or routing strategy (oracle vs blind).

---

## Dependency state (pre-flight verification)

Registry (`adapters/registry.json`) references these paths for the 5 domain
knowledge adapters:

| adapter | registry path | state |
|---------|---------------|-------|
| math-gsm8k-knowledge-v0 | `micro/models/exp_p1_t2_single_domain_training/adapters/math/` | **config-only stub** |
| code-codealpaca-knowledge-v0 | `micro/models/exp_p1_t2_single_domain_training/adapters/code/` | **config-only stub** |
| medical-medmcqa-knowledge-v0 | `micro/models/exp_p1_t2_single_domain_training/adapters/medical/` | **config-only stub** |
| legal-mmlu-knowledge-v0 | `micro/models/exp_p1_t2_multi_domain_5/adapters/legal/` | **config-only stub** |
| finance-mmlu-knowledge-v0 | `micro/models/exp_p1_t2_multi_domain_5/adapters/finance/` | **config-only stub** |

Parent-repo shim dirs `adapters/{math,bash,python,sql,medical}/` are also stubs
(config + chat_template + readme + tokenizer_config, **no `adapters.safetensors`**).

Only `adapters/thinking-openthoughts-universal-v0/0000050_adapters.safetensors`
(151 MB) exists with real weights — unrelated to this experiment (universal
thinking adapter, not a domain expert).

## Antipattern self-check

- **antipattern-017** (weight-less adapter stub): TRIGGERED — 5/5 registry
  paths are stub directories. **7th confirmed instance** across 2 days.
  Prior 6: baseline_eval, M0 full_pipeline_v2, J0 adapter_composition_thinking
  (4-of-4 stubs), followup_composition_correct_delta (5/5), followup_routing_multi_sample_ppl
  (5/5), followup_competitive_gsm8k_200n (5/5).
- **antipattern-020** (cascade-dependent design): TRIGGERED — registry
  adapter state is a known-killed cascade; re-running the same routed composition
  without first rebuilding adapters reproduces the same failure mode.
- **antipattern-018** (channel-tokens-as-SFT): N/A (inference only, no SFT).
- **antipattern-003** (`LORA_SCALE` without forward-pass scaling): N/A (preemptive,
  no scaling decision reached).
- **KC-swap**: CLEAN — MATH.md is a new file in a new directory; single commit.

## Pre-registered predictions

| Metric | Predicted | Rationale |
|--------|-----------|-----------|
| K1592: Δ_gsm8k ≥ +5pp AND Δ_mmlu_pro ≥ 0 | **FAIL** | Thm 1+2+3: `y_routed ≡ y_base` ⇒ both deltas = 0, AND-clause collapses |
| Pr[any of 5 adapters loads with weights] | 0 | All 5 paths are config-only |
| `‖Σ B_i A_i‖_F` (routed delta norm) | 0 | Thm 2 case (a)/(c); negligible under case (b) |
| Thinking-mode rescue | None | Thm 3: prompt-level change, inert to absent adapter ops |

## KC numeric ID alignment

- Kill Criterion ID in DB: **1592** (`experiment get exp_g4_routed_beats_base_think`).
- No MATH-vs-DB drift.
- Success criteria: **None** (DB flagged `⚠ INCOMPLETE`) — metadata defect, does
  not affect kill (kill rests on K1592 alone).

## References

- **Finding #236**: `exp_competitive_benchmark_routed` — "routed -20pp on math at n=20"
  (now status=killed on K640).
- **Finding #237**: "GSM8K +10pp under oracle routing" (doesn't transfer to blind
  routing per F#553).
- **Finding #517**: knowledge adapters degrade MCQ — relevant to why MMLU-Pro
  conjunct of K1592 would also be at risk even if GSM8K clause were met.
- **Finding #553**: single-sample routing artifact — per-sample routing breaks
  oracle identity at p<1. Independent argument reinforcing kill if adapters existed.
- **Finding #560**: Gemma 4 base MMLU-Pro = 40.7% measured at n=1400 (relative
  KC only — baseline parity clause of K1592 relies on match, not an absolute threshold).
- **Antipattern-017**: weight-less adapter stub (7 instances now).
- **Antipattern-020**: cascade-dependent experimental design.
- **Unblock path**: `P11.ADAPTER-REBUILD` (same as M0, L0, J0, followups 1+2+3).
