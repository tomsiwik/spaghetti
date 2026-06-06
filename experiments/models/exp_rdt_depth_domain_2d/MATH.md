# exp_rdt_depth_domain_2d — MATH.md

## Preemptive-kill structural argument

**TYPE**: verification (structural preempt)

**FAILURE MODE**: All 4 kill criteria transitively require two parent experiments to have
produced SUPPORTED target claims with trained artifacts. Neither has.

**PRIOR FINDING**: F#669 — preempt-child-KCs-require-parent-target-claim-unverified.
Precedent: exp_rdt_act_halting_throughput was preempt-killed under this rule because all
KCs required exp_rdt_loop_lora_gemma4 target KCs K1740/K1741/K1742 SUPPORTED.
Precedent: F#513 (MemoryLLM dep-killed), F#558 (GRPO warm-start dep-block).

## Dependency state (as of 2026-04-19)

1. **exp_rdt_loop_lora_gemma4** — status=killed (smoke-PROVISIONAL per CLI rule #4).
   - K1740 (loop quality rises with T): not_measured (is_smoke=true)
   - K1741 (saturating-exp fit R²≥0.7): not_measured
   - K1742 (ΔW_loop(t) distinct): not_measured
   - Scaffolding KCs K1743/K1744 PASS at init only; no trained loop-LoRAs exist.
   - F#668 registered (provisional, static scaffolding only).

2. **exp_method_composition_k_saturation** — status=killed.
   - 2026-04-18 SMOKE ABORTED at Phase-1 teacher gate. 0/5 methods reached
     70% teacher-signature rate. **No trained method adapters exist.**

Both parents are in non-trained-artifact states. Our experiment requires trained
artifacts from BOTH axes. This is a **double-parent-unfulfilled preempt**.

## Theorem 1 — K1749 dep-unfulfilled

**Claim**: K1749 ("2D composition beats domain-only by ≥+3pp") requires both
(a) N=5 trained domain adapters and (b) T=4 trained loop adapters at matched
trainable-param budget.

**Proof**: The quality delta Δquality = Q(domain × loop) − Q(domain only) is
an empirical measurement on trained composition outputs. At init, all LoRA B=0
⇒ both sides equal base-model quality ⇒ Δ = 0. Therefore K1749 cannot be
measured without trained artifacts from both parents. Parent loop-LoRA is
smoke-only (F#668); parent method-composition is aborted at Phase-1.
∎

## Theorem 2 — K1750 dep-unfulfilled

**Claim**: K1750 ("loop axis does not saturate at T=6 with N=5 domains active")
requires T={1,2,3,4,5,6} trained loop-adapter composition measured on task
quality. This IS parent target K1740/K1741 content (saturating curve fit).

**Proof**: Saturation is a property of the Q(T) function measured on trained
outputs. Parent K1741 explicitly pre-registers "saturating-exp fit R²≥0.7".
Without parent supported, we have no Q(T) curve. K1750 is a stricter variant
(evaluated at N=5 domains active) of the parent target — cannot exceed parent.
∎

## Theorem 3 — K1751 requires trained deltas, not init

**Claim**: K1751 ("avg |cos(ΔW_domain_i, ΔW_loop_j)| < 0.1") requires trained
ΔW, not partition-QR init geometry.

**Proof**: ΔW = A·B where B is initialized to 0 in standard LoRA. At init:
ΔW = A·0 = 0 ⇒ cos(ΔW_i, ΔW_j) = undefined (0/0). Measurement must occur
after B gradient-updates away from 0 via task loss, i.e. after training.
Parent loop-LoRA has no trained B; parent method-composition has no trained
adapters at all. F#562 (partition-QR A-init orthogonality) is a DIFFERENT
claim about A-geometry at init — not about ΔW orthogonality after training.
∎

## Theorem 4 — K1752 Room Model requires all trained artifacts

**Claim**: K1752 ("W_combined = Σ α·ΔW_domain + Σ β·ΔW_loop reproduces
per-(domain,loop) routing within cos > 0.999") requires N=5 trained domain
ΔW and T=4 trained loop ΔW and a working explicit-routing baseline.

**Proof**: The Room Model cos identity is a property of specific trained
ΔW matrices. The identity is only testable against a reference routing
output, which itself requires trained adapters on both axes.
Additionally, F#571 already demonstrated Room Model superseded for N>1.
∎

## Antipattern self-audit

- F#138 (copy-paste scaffolding): N/A, no code being written for trained run.
- F#452/453/1564 (proxy with empirical refutation): N/A.
- F#498/F#666 (tautology / target-gated KC): N/A (preempt, not measuring).
- F#667 (LTI primitive Float32 caveat): N/A.
- F#571 (Room Model N>1 superseded): K1752 would fall under this anyway.
- F#669 (child KCs require parent target unverified): **EXACT MATCH** — this
  is a second-reuse. Promotion candidate on next occurrence.

## Predicted outcome

PREEMPTIVE-KILL. All 4 KCs fail as not_measured (dependency not fulfilled).
Unblock path: finish exp_rdt_loop_lora_gemma4_full (macro follow-up ticket
logged in parent LEARNINGS) AND resurrect exp_method_composition_k_saturation
with a structural fix to the Phase-1 teacher gate.

## KC disposition

- K1749 → fail (not_measured, dep-unfulfilled via T1)
- K1750 → fail (not_measured, dep-unfulfilled via T2)
- K1751 → fail (not_measured, dep-unfulfilled via T3)
- K1752 → fail (not_measured, dep-unfulfilled via T4)

No code executes. executed=false, is_smoke=false, preemptive=true.
