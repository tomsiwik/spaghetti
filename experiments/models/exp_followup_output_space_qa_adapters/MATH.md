# MATH.md — exp_followup_output_space_qa_adapters (PREEMPTIVE KILL)

## Claim
The hypothesis "QA-format adapters with KV-cache-aware top-2 output-space composition
will beat NTP-format adapters on QA accuracy by ≥5pp (K1552)" is either **trivially
tautological** or **fails the prerequisite gate that F#166 declared a structural
requirement**. In both branches, running the experiment cannot advance the thesis.

## Theorem (preempt — logical AND of three independent lemmas)

Let `A_NTP` = top-2 output-space composition over adapters trained on next-token
prediction (prose) on Falcon-E-3B. Let `A_QA` = the same but over adapters trained
on QA-format supervision. Let `B` = the unmodified instruction-tuned base. Let
`Q(·)` = QA accuracy on multiple-choice items (e.g. MMLU).

Then at least one of the following holds, and each alone suffices to invalidate the
experiment as a thesis-advancing test:

**L1 (tautological-KC branch):** `Q(A_QA) − Q(A_NTP) ≥ 5pp` is trivially satisfied.
NTP-format adapters on MCQ tasks systematically emit continuation-style prose, not
letter answers; on Falcon-E-3B they measured −24% vs base on MMLU (F#165, 0.410 vs
0.540). Any adapter that answers in the expected MCQ letter format — which QA-format
supervision enforces by construction — trivially clears the 5pp gap over prose-
emitting adapters. K1552 measures output-format alignment, not composition quality.

**L2 (prerequisite-gate branch):** Even if K1552 PASSES non-trivially, the thesis is
not advanced. F#166 impossibility structure (verbatim):

> Output-space eliminates cross-terms (LoRI proof correct) but cannot rescue
> individually harmful adapters. Prerequisite gate needed: single adapter must beat
> base before testing composition.

F#165 impossibility structure (sharper):

> If E[quality(adapter_i)] < E[quality(base)], then
> E[avg(adapter_i, adapter_j)] < E[quality(base)] — no aggregation fixes negative
> individual contributions.

K1552 compares inter-adapter ΔA_QA − A_NTP; it does not measure Q(A_QA) vs Q(B).
The prerequisite gate (single QA-adapter on Falcon-E-3B must beat base) is neither
pre-registered nor measured. Without it, K1552 PASS reveals nothing about whether
the mechanism works.

**L3 (base-beating impossibility on Falcon-E-3B):** F#477 (Gemma 4, a stronger base
than Falcon-E-3B) measured single-adapter base-beat rate 2/5 domains, with K1226
"adapted acc ≥ 0.50" FAILED. The dominant failure mode: "δ_d ≈ 0 when H(V_d|θ) is low
— base already calibrated." Falcon-E-3B is ternary-quantized and already instruction-
tuned; its calibration is at least as strong as Gemma 4's on MMLU-class tasks.
Therefore the prerequisite gate is structurally unlikely to hold even with QA-format
adapters: format-fix addresses the symptom (wrong output vocabulary) but not the
disease (low `H(V_d|θ)` on MMLU-class domains).

**L4 (implementation-cost confounder):** F#166 further measured 2.7 tok/s vs required
30 — a 17× overhead from naive adapter-swap. KV-cache-aware impl is the proposed
remedy, but even if speed is rescued, accuracy is not. The experiment bundles two
orthogonal fixes (format + speed) into one KC, so a PASS on K1552 cannot be
attributed to either.

### QED (preempt)
`L1 ∨ L2 ∨ L3` ⇒ K1552 outcome carries no information about the thesis. No
hyperparameter, scale, or seed choice changes this: the KC itself is malformed with
respect to the claim it purports to test.

## Behavioral prediction (if the code had been run)

- P1: Q(A_NTP) ≤ 0.42 (reproducing F#165 on Falcon-E-3B)
- P2: Q(A_QA) ∈ [0.47, 0.53] (format-fix lifts NTP adapters into the MCQ-answer
  regime, but does not solve domain quality)
- P3: Q(A_QA) − Q(A_NTP) ≥ 5pp — K1552 would PASS (tautologically, per L1)
- P4: Q(A_QA) − Q(B) ∈ [−0.08, −0.01] (unchanged F#477 pattern — adapters still lag
  base on MMLU-class)

## Kill criteria (pre-registered 2026-04-17)

**K1552** (numeric id 1552): "QA-format adapters with KV-cache-aware top-2 beat
NTP-format adapters on QA accuracy by ≥5pp."

Preempt-verdict: K1552 **FAIL** (status: structurally uninformative; see L1/L2/L3).

## Why no structural repair
The experiment is well-intentioned but mis-scoped. A correct v2 would:
1. Pre-register a base-beat gate: `Q(A_QA,single,d) ≥ Q(B) + 3pp` for ≥3/5 domains.
2. Replace K1552 with `Q(A_QA,top2) ≥ Q(B) + 5pp` (i.e. composition beats base,
   not just beats a straw adapter).
3. Cite F#167/F#168 that runtime LoRA IS output-space MoE — so the binding
   constraint is base model quality, not composition architecture.

Rather than run a malformed KC, we preempt-kill and recommend the v2 design above.
No code is executed.

## References
- F#166 (killed, 2026-03-28): OS top-2 ceiling, prerequisite gate statement
- F#165 (killed, 2026-03-28): adapters −24% on Falcon-E-3B; impossibility structure
- F#477 (killed, 2026-04-11): Gemma 4 single-adapter base-beat rate 2/5
- F#167/F#168 (supported, 2026-03-28): Runtime LoRA IS output-space MoE
- LoRI paper (arxiv:2504.07448): cited in F#165 impossibility structure
