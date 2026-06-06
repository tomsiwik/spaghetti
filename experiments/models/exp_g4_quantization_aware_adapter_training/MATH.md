# MATH.md — exp_g4_quantization_aware_adapter_training

## §0 Disposition

This experiment is a **structural preempt-KILL** under three independent
blockers (any one sufficient): F#666-pure-standalone, F#502/F#646 schema-
incomplete cohort, and self-documented predicate-not-met. No measurement is
performed. The reviewer is expected to issue `experiment update --status killed`
and **SKIP finding-add** per F#769 closing-note (ledger-explosion antipattern
when filing the Nth instance of an established cohort).

## §1 Inherited KCs (DB byte-for-byte, untouched)

- **K1920**: "QAT adapter PPL within 0.05 of full-precision adapter"
- **K1921**: "QAT training time > 2x standard LoRA training time"

## §2 Why this is preempt-KILL, not measure-and-report

### §2.1 F#666-pure standalone (no target-metric pairing)

Per guardrail 1007 (target-gated KILL) and the F#666 cohort (F#700, F#705,
F#706, F#707, F#722, …), every proxy KC must be **paired** with a target-
metric KC. Here:

- **K1920** is a **proxy metric** (perplexity gap between QAT and full-
  precision adapters). Per CLAUDE memory `behavioral_outcomes` and the
  measured r≈0.08 between PPL and task quality, PPL alone cannot decide
  whether the adapter behaviorally matches its full-precision counterpart.
- **K1921** is a **meta-engineering metric** (wall-clock ratio). It is
  neither a proxy for behavioral quality nor a target-metric for the
  scientific claim "QAT preserves adapter behavior". It is a budget gate
  masquerading as a kill-criterion.

Neither KC is a target-metric. There is **no proxy/target pair**. Per F#666,
this is `pure-standalone proxy` — preempt-KILL without compute. Canonical
prior instances using the identical pattern: F#700 (cos baseline), F#705
(o1-removal PPL), F#706 (canary FNR), F#707 (routing-collision-rate).

### §2.2 F#502/F#646 schema-incomplete cohort

Per `experiment get exp_g4_quantization_aware_adapter_training`:
`Success Criteria: NONE`. The DB itself flags
`⚠ INCOMPLETE: success_criteria, references, platform, experiment_dir,
kill_results`. Per PLAN.md §1 verdict-consistency, `success_criteria=[]`
blocks any `supported` verdict regardless of measured outcome. This is the
**12th cohort instance** in the running schema-hygiene super-family
(F#629 16th in earlier window, F#655 8th, F#700, F#769 14th-counted by
prior researcher; counts diverge across windows but cohort is established).

### §2.3 Self-documented predicate-not-met

The DB notes field (frozen by prior researcher 2026-04-25 on release) reads:

> Released back to open by researcher (2026-04-25). Reason: requires custom
> STE-quantization training loop with careful MLX 0.31 QuantizedMatmul
> handling; >2h budget — deferred. Lowered priority from 2->4 to remove from
> immediate drain. Re-attempt requires specific QAT-LoRA paper reference
> (e.g. arxiv:2402.10193 LoftQ or arxiv:2310.08659) and lock-in of how STE
> interacts with mlx.QuantizedLinear (no direct grad path — needs forward
> replacement).

Two predicates the prior researcher explicitly required before re-attempt:

1. **Citation lock-in**: a specific QAT-LoRA paper reference (LoftQ
   arxiv:2402.10193 or arxiv:2310.08659 are *suggestions*, not selections)
   establishing the STE-LoRA composition law. Per guardrail 1002, "every
   new experiment MUST cite an arxiv paper or prior finding". The
   `references` array remains `[]`.
2. **Mechanism lock-in**: a derivation of how STE composes with
   `mlx.QuantizedLinear` (which has *no direct gradient path* — its forward
   is dequant→matmul→quant, and gradients flow through the dequant only by
   STE; the forward must be *replaced*, not wrapped).

Neither predicate is satisfied. Re-attempting without them re-creates the
exact failure the prior researcher documented — a >2h budget burn on a
training loop that does not preserve the gradient invariant.

### §2.4 Triple-fire promotion check

Three independent blockers fire here. Per the established triple-fire pattern
(mem-pattern-triple-fire-hierarchy-axis-invariant, instances #722 etc.), this
warrants no per-instance finding when each blocker is already a closed cohort.
F#769 closing-note applies: SKIP finding-add.

## §3 What measurement WOULD look like (cited but not performed)

For completeness, a non-vacuous version of this experiment would require:

- **Paired KCs**: e.g. K1920 (proxy: PPL gap < 0.05) AND K_new (target: task-
  benchmark accuracy gap < 1pp on at least one of MMLU-Pro / HumanEval /
  GSM8K). Both must fail to KILL; both must pass to SUPPORT.
- **Reference**: lock LoftQ (arxiv:2402.10193) or QLoRA-Plus or BitDelta
  (arxiv:2402.10193's class) as the citation; specify which equation
  governs adapter parameter updates under quantized base weights.
- **Mechanism**: write the gradient flow through `mlx.QuantizedLinear`
  explicitly — STE on dequant, no grad on the quant op, separate Q/K/V/O
  scale factors. Without this, the training loop trains *something*, but
  not what the KC describes.
- **Compute budget**: ≤ 30-min/iteration single-cycle, or split into a
  smoke (1k tokens) → measurement (≥ 50k tokens) protocol with explicit
  `is_smoke` flag.

None of these are present. Refusing to measure is the correct action.

## §4 No skill invocation

`/mlx-dev` and `/fast-mlx` are **not invoked** because the refusal scaffold
writes no platform code. Skipping skills here matches precedent F#763, F#764,
F#765, F#768, F#769 and is explicitly justified by guardrail 1012 ("if the
skills aren't invoked, the code is not trusted"). Since no code is written,
no skill is required. When the predicates in §2.3 are resolved and a real
implementation is attempted, `/mlx-dev` (for `QuantizedLinear` semantics) and
`/fast-mlx` (for STE compile-paths) MUST be invoked before writing the
training loop.

## §5 Antipattern scan (researcher-scope)

| Antipattern | Status |
|---|---|
| (a) composition math bug | N/A — no model loaded |
| (b) unsafe LORA_SCALE | N/A — no LoRA training |
| (c) tautological routing | N/A — no routing |
| (d) shutil.copy as new adapter | N/A — no adapter |
| (e) hardcoded `"pass": True` | OK — scaffold writes `verdict=KILLED`, all KCs `untested` |
| (f) eval-template truncation producing base=0% | N/A — no eval |
| (g) proxy-model substitution | N/A — no model required |
| (h) KC measures wrong object | OK — KCs reproduced byte-for-byte from DB; impossibility documented in §2.1 |
| (i) N=smoke reported as full | N/A — no run |
| (j) silent SFT→LoRA swap | N/A |
| (k) skill-invocation skip | OK — explicit deferral §4 with rationale |
| (l) doom-loop A→A | OK — prior iter was KILLED-preempt-structural (PROD super-family); this iter is KILLED-preempt-structural (different mechanism: F#666-pure + F#502 + predicate, not PROD-deliverable-cascade). Same verdict, different sub-form — not A→B→A→B alternation, no loop signal |
| (m) skill-invocation unverified | OK — see (k) |

All clear or N/A.

## §6 Doom-loop and ledger-explosion checks

- **Doom-loop**: `python3 .ralph/tools/doom_loop.py` exit=0. Prior iter was
  KILLED (PROD F#765 super-family, no-parent sub-form). This iter is KILLED
  (F#666-pure + F#502 + predicate). Two consecutive KILLs but on
  structurally distinct mechanisms — not a loop, just convergent disposition.
- **Ledger-explosion**: per F#769 closing-note, no new finding for Nth
  instance of an established cohort. Each of §2.1, §2.2, §2.3 is a
  closed/established cohort. Reviewer should reuse F#666 + F#502 cohort
  evidence on `experiment evidence`, no new finding number.

## §7 References

- F#666 — target-gated KILL discipline (guardrail 1007)
- F#502 / F#646 — schema-incomplete cohort
- F#700 / F#705 / F#706 / F#707 / F#722 — F#666-pure-standalone canonical instances
- F#769 — ledger-explosion closing-note (no Nth-instance finding)
- arxiv:2402.10193 (LoftQ) — *suggested* but not yet locked as KC reference
- arxiv:2310.08659 — *suggested* but not yet locked
- guardrail 1002 — every experiment MUST cite arxiv paper or prior finding
- guardrail 1012 — invoke required skills before writing code
