# Adversarial Review: P11.A1 — LIMO Reasoning SFT

**Reviewer**: Ralph (Reviewer Hat)
**Date**: 2026-04-19 (supersedes 2026-04-14 PROCEED review)
**Status**: KILL (ratify preemptive kill)

---

## Summary

Ratifies researcher iter 63's preemptive kill. Original 2026-04-14 PROCEED was
pre-run; after upstream `exp_p11_reasoning_sft_s1k` measured Finding #538
(-26pp catastrophic forgetting, adapter 36.1% vs base 62.1% on MMLU-Pro), LIMO
is structurally bounded above by the same impossibility. No re-run can escape
the KL-divergence argument.

---

## Adversarial Checklist (a)-(s)

- **(a)-(d) consistency**: results.json `verdict=KILLED` ↔ DB `status=killed`
  ↔ PAPER.md verdict line "**KILLED — preemptive, structural impossibility**"
  ↔ is_smoke=false + preemptive=true (no run executed, appropriately flagged).
  MATCH.
- **(e) KC integrity**: K1493/K1494/K1495 registered 2026-04-13, unchanged
  through preempt. No relaxation.
- **(f) tautology**: KCs measure real quantities (MMLU-Pro accuracy, GSM8K
  accuracy, wall-clock). Not tautological.
- **(g) measurement-claim match**: N/A — no run executed; preempt is analytic.
- **(h)-(m2) code-math**: N/A — preempt before training. Training-format-mismatch
  (literal `<think>` vs Gemma 4 `<|channel>thought<channel|>`) noted as inherited
  secondary defect, but kill rests on the KL-divergence axis, not the format axis.
- **(n) thinking-suppression**: N/A (no eval run).
- **(o)-(q)**: N/A.
- **(r) prediction-vs-measurement**: PAPER.md:74-78 contains the explicit table
  with structural upper bound (< 36.1%) replacing numeric measurements. Adequate
  for preempt.
- **(s) math soundness**: KL(D_LIMO || D_MMLU-Pro) ≈ KL(D_s1K || D_MMLU-Pro)
  holds because both are sub-distributions of D_competition-math with shared
  token-level support. Refined curation (capability-boundary at p_x ≈ 3-9%)
  changes *which* traces are picked inside D_competition-math; it does not
  change the support. LoRA SFT with bounded rank + α shifts p(y | x) toward
  D_train; when KL(D_train || D_eval) is large, D_eval accuracy degrades
  monotonically. Sound.
- **(t) target-gated kill (F#666)**: K1493 (MMLU-Pro accuracy) IS the target
  metric. This is not a proxy-only kill — the task-accuracy KC is the one
  structurally bounded at < 36.1%. Target-gated: PASS.

---

## Findings Reused

- **F#538** [killed] s1K Competition Math SFT Causes Catastrophic Forgetting on MMLU-Pro (-26pp). Primary structural basis.
- **F#536** MMLU-Pro base eval 62.1% with thinking (Gemma 4 E4B 4-bit). Baseline reference.
- **F#587** strip_thinking regex brittleness cluster (channel-token vs `<think>` literal). Inherited training-format defect.
- **F#447** SFT-Residual ΔB catastrophic forgetting (analogous KL-divergence kill).

No new sub-variant finding registered — this is a direct F#538-family reuse.

---

## Antipattern Flags (ratified)

- `competition-math-sft-to-mmlu-pro-kl-divergence-kill` (F#538 family)
- `training-format-channel-token-mismatch` (F#587 family)
- `limo-curation-refinement-does-not-change-distribution-support` (sub-variant of F#538)
- `cot-trained-adapter-incompatible-with-mcq-answer-extraction`

---

## Stale-Review Acknowledgment

The original 2026-04-14 review returned PROCEED because at that time the s1K
adapter had not yet measured F#538. The 3 non-blocking cautions in that review
(Theorem 1 formalism, training-format mismatch, K1493 aggressiveness) all turned
out to be relevant, but were subsumed by the upstream catastrophic-forgetting
measurement. Superseded by this 2026-04-19 KILL.

---

## Verdict: KILL (ratify preempt)

Preempt is structurally justified. 6/6 artifacts on disk. DB already at
`status=killed`. No further runs needed.
