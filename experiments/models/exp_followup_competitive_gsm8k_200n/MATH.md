# MATH: exp_followup_competitive_gsm8k_200n (PREEMPTIVE KILL)

**Verdict:** KILLED — preemptive (antipattern-017 cascade, 6th confirmed instance).
**KC:** K1575 — "at n≥100/subject with fixed extraction, routed beats base with CI not overlapping zero."
**Status:** unmeasurable. The domain adapters required for `routed` composition do not exist on disk.

---

## Theorem 1 (Measurability of K1575)

**Statement.** Let `A = {a_math, a_code, a_medical, a_legal, a_finance}` be the five domain
adapters referenced by the routing composition in `competitive_benchmark_routed/run_experiment.py`
(L39-41: `SOURCE_DIR/"adapters"/{domain}`). Let `W(a_i) ∈ R^(r×d_in) × R^(d_out×r)` be the
LoRA weight pair for adapter `a_i`. Define the routed forward pass

  `y = x + Σ_i α_i · (B_i A_i x)`   (eq. 1)

where `α_i` is the per-sample routing weight over adapter `i`. Then K1575 is measurable
iff `∀ i ∈ A: ||A_i|| > 0 ∧ ||B_i|| > 0`.

**Proof.** If `A_i = 0 ∨ B_i = 0` for some `i`, then `B_i A_i x = 0`, so adapter `i`
contributes no signal regardless of `α_i`. The routed forward pass degenerates to the
identity over the missing-adapter subspace. At the K=5 case with ALL adapters missing,
`y = x` → routed ≡ base for every input, so `Pr[y_routed ≠ y_base] = 0` and the
claim "routed beats base with CI not overlapping zero" is **structurally false**:
no measurement can return a difference. ∎

## Theorem 2 (Observational Equivalence of Config-Only Stubs)

**Statement.** An adapter directory containing only `adapter_config.json` (no
`adapters.safetensors`) is observationally equivalent under `load_adapter(dir)`
to **not loading any adapter** — whether the loader (a) crashes, (b) warns, or
(c) silently returns the base model.

**Proof.** `adapter_config.json` is a JSON metadata file with fields like
`lora_alpha`, `r`, `target_modules`. It contains **no weight tensors**. Any runtime
that honors the config must initialize `(A_i, B_i)` from (a) safetensors on disk,
(b) random init, or (c) noop. Case (a) → fails (file missing). Case (b) →
equivalent to random-noise "adapter" with E[output] = base output, variance > 0
(not adapter semantics, which is `Δ ≠ 0` in a specific subspace learned from
data). Case (c) → y = x (base). In all cases, `E[routed − base] = 0`. ∎

## Corollary (K1575 unreachable)

From Theorems 1 & 2: with 5/5 domain adapters stub-only on disk, `E[acc_routed − acc_base] = 0`
and CI[acc_routed − acc_base] is centered at 0. K1575 requires "CI not overlapping zero" —
impossible. K1575 = FAIL by construction, independent of n.

---

## Dependency state (pre-flight verification)

Registry (`adapters/registry.json`) references these paths:

| adapter | path | state |
|--------|------|-------|
| math-gsm8k-knowledge-v0 | `micro/models/exp_p1_t2_single_domain_training/adapters/math/` | **config-only stub** |
| code-codealpaca-knowledge-v0 | `micro/models/exp_p1_t2_single_domain_training/adapters/code/` | **config-only stub** |
| medical-medmcqa-knowledge-v0 | `micro/models/exp_p1_t2_single_domain_training/adapters/medical/` | **config-only stub** |
| legal-mmlu-knowledge-v0 | `micro/models/exp_p1_t2_multi_domain_5/adapters/legal/` | **config-only stub** |
| finance-mmlu-knowledge-v0 | `micro/models/exp_p1_t2_multi_domain_5/adapters/finance/` | **config-only stub** |

Parent-repo adapter dirs `adapters/{math,bash,python,sql,medical}/` (shim-style) are
also stubs (config + chat_template + readme + tokenizer_config, no `.safetensors`).

Source experiment dir `micro/models/real_data_domain_experts/` — referenced by the prior
killed `competitive_benchmark_routed/run_experiment.py` L39 — has **no `adapters/`
subdirectory at all**. Adapter weights are gone everywhere.

## Antipattern self-check

- **antipattern-017** (weight-less adapter stub): TRIGGERED — 5/5 registry-referenced
  adapters are stubs. This is the **6th confirmed instance** across 2 days
  (baseline_eval, M0/J0 [stub pairs], followup_composition_correct_delta,
  followup_routing_multi_sample_ppl, this).
- **antipattern-020** (cascade-dependent design): TRIGGERED secondarily — the prior
  experiment this followup retests (`exp_competitive_benchmark_routed`) is itself
  `status=killed` on K640 (routed worse than base on 2/6 benchmarks, math -20pp @ n=20).
- **antipattern-018** (channel-tokens-as-SFT): N/A (inference only, no SFT).
- **antipattern-003** (`LORA_SCALE` without scaling the forward pass): N/A preemptive.

## Pre-registered predictions

| Metric | Predicted | Rationale |
|--------|-----------|-----------|
| K1575: CI excludes 0 | **FAIL** | Thm 1 + Thm 2: `E[routed − base] = 0` ⇒ CI centered at 0 |
| `Pr[any adapter loads]` | 0 | All 5 paths config-only |
| `||delta_W|| = ||Σ B_i A_i||_F` | 0 | Thm 2 case (c) or NaN (case a) |

## KC numeric ID alignment

- Kill Criterion ID in DB: **1575** (`experiment get exp_followup_competitive_gsm8k_200n`).
- No MATH-vs-DB drift.
- Success criteria: `None` (DB flagged `⚠ INCOMPLETE`) — does not affect kill (the
  kill rests on K1575 alone). Note for analyst/harness: SC missing is a separate
  metadata defect.

## References

- Finding #237: "GSM8K +10pp is the only consistent competitive advantage" (oracle routing;
  doesn't transfer to blind routing per F#553).
- Finding #517: "Knowledge adapters degrade MCQ" (registry note on math-gsm8k-v0).
- Finding #553: single-sample routing artifact (argued per-sample routing forbids
  tautological identity at p<1 — now moot since adapters absent).
- Finding #560: Gemma 4 base MMLU-Pro = 40.7% measured @ n=1400 (blocks absolute KCs
  but not relative ones; this KC is relative).
- Prior kill: `exp_competitive_benchmark_routed` killed 2026-04-17 on K640 (routed WORSE
  than base on math/legal at n=20). Current followup attempted to argue "noise at n=20,
  rerun at n≥100"; premise requires adapters that no longer exist.
- Antipattern-017: weight-less adapter stubs. 6 instances now.
- Antipattern-020: cascade-dependent experimental design.
