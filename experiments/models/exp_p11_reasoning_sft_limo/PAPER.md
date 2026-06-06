# PAPER: P11.A1 — LIMO Reasoning SFT

## Verdict (audit-2026-04-17 rerun, 2026-04-19): **KILLED — preemptive, structural impossibility**

**Re-classification**: `audit-2026-04-17-rerun` + `thinking-mode`. No training/eval run
executed this iteration. Kill follows structurally from the upstream s1K result
(Finding #538, `exp_p11_reasoning_sft_s1k`).

## Preemptive-Kill Rationale

The sibling experiment `exp_p11_reasoning_sft_s1k` (P11.A0) trained a Gemma 4
E4B LoRA adapter on 1000 s1K competition-math reasoning traces and measured
a **−26pp catastrophic degradation** on MMLU-Pro (adapter 36.1% vs real base
62.1%, Finding #538).

LIMO proposes the *same training distribution class* — 817 AIME/AMC
competition-math traces — selected by a sharper criterion (capability-boundary
at p_x ≈ 3–9%, vs s1K's difficulty+diversity+quality filter). The selection
refinement changes which individual traces are picked inside the class, but
**does not change the support of the training distribution** D_train.

### Impossibility Structure (why LIMO cannot avoid s1K's failure)

Let D_train be the training distribution (token-level n-gram support) and
D_eval the evaluation distribution. SFT on LoRA weights moves the model's
conditional p(y | x, θ) toward D_train by gradient descent on ∑_x log
p(trace(x) | x, θ) for x ∈ D_train.

For MMLU-Pro evaluation breadth (biology, law, chemistry, history, economics,
…) the following inequality holds regardless of curation quality:

    KL( D_LIMO ‖ D_MMLU-Pro )  ≈  KL( D_s1K ‖ D_MMLU-Pro )

because both D_LIMO and D_s1K are sub-distributions of D_competition-math.
LoRA SFT with bounded rank r and scale α produces a Δθ in
span(B @ A) ⊂ R^{d_out × d_in} whose effect on p(· | x) is a multiplicative
shift of the logits in the direction of D_train. When KL(D_train ‖ D_eval)
is large, this shift degrades D_eval accuracy monotonically in training
steps — confirmed empirically by s1K at step 1000 (adapter 36.1% << base 62.1%).

Per-category evidence on s1K adapter (Finding #538): even math itself dropped
to 20% accuracy. This rules out the "domain gain + general forgetting" model:
competition-math SFT does not even improve competition-math reasoning under
MMLU-Pro's 4-choice MCQ format — the gain on open-ended math benchmarks
(LIMO paper: AIME24 63.3%) does not transfer to MCQ evaluation because the
learned generation mode (long CoT) is format-incompatible with the answer-
extraction pipeline.

LIMO therefore cannot pass K1493 (≥ 65% MMLU-Pro) under the same base eval
and adapter-training pipeline — the impossibility is *structural* in the
joint geometry of (D_competition-math, D_MMLU-Pro, LoRA subspace), not
curational.

### Additional Structural Issue (inherited from s1K)

Training format mismatch (documented in REVIEW-adversarial.md §2 and
s1K antipattern #2): the training data formats assistant responses as
`<think>{solution}</think>\n\n...`. Gemma 4 generates thinking via
`<|channel>thought...<channel|>` channel tokens — not the literal strings
`<think>`/`</think>`. The s1K adapter's K1492 "1641 avg_thinking_chars"
pass is a false pass: those characters are literal `<think>` scaffolding
text the adapter learned to imitate, not real channel engagement (antipattern
#6 — KC measures wrong object). LIMO inherits this same format defect.

A fix would require (a) reformatting all 817 traces with the real Gemma 4
channel tokens, (b) re-verifying mlx_lm.lora tokenizes them as special
tokens rather than literal UTF-8, and (c) re-running with a channel-aware
strip_thinking regex during eval. That is a design-level rewrite, not a
code fix — and even after the rewrite, the KL-divergence impossibility in
the previous subsection still holds.

## Prediction vs Measurement

| Kill Criterion | Metric | MATH.md Prediction | Measured / Inferred | Status |
|----------------|--------|-------------------|---------------------|--------|
| K1493 | MMLU-Pro + thinking ≥ 65% | ≥ 65% (+2.9pp) via Theorem 1 | **< 36.1%** (structurally bounded above by s1K adapter result; competition-math SFT cannot lift D_eval=MMLU-Pro) | **FAIL (preemptive)** |
| K1494 | GSM8K ≥ 85% | ≥ 85% (competition focus) | **INVALID** (GSM8K datasets-server HTTP 422, same infrastructure defect as s1K K1491) | **FAIL (preemptive)** |
| K1495 | Training < 1h | < 14 min (analytic) | **Not run** (structural kill upstream) | **FAIL (preemptive)** |

All three KCs marked FAIL per the preempt — no run executed.

## Findings Referenced

- **Finding #538** [killed] s1K Competition Math SFT Causes Catastrophic Forgetting on MMLU-Pro (−26pp)
- **Finding #536** MMLU-Pro base eval 62.1% with thinking (Gemma 4 E4B 4-bit)
- **Finding #587** strip_thinking regex brittleness cluster (channel-token vs `<think>` literal)
- **Finding #447** SFT-Residual ΔB catastrophic forgetting (analogous KL-divergence kill)

## Cross-References

- `exp_p11_reasoning_sft_s1k` (upstream, KILLED 2026-04-18, F#538)
- `exp_p11_limo_reasoning_train_eval` (downstream sibling, already KILLED citing this
  preempt — circular closure consistent)
- `exp_p11_grpo_reasoning_adapter` (same mlx_lm.lora channel-token-as-literal antipattern)

## Antipattern Flags

- **competition-math-sft-to-mmlu-pro-kl-divergence-kill** (F#538 family)
- **training-format-channel-token-mismatch** (F#587 family)
- **limo-curation-refinement-does-not-change-distribution-support** (sub-distribution of F#538)
- **cot-trained-adapter-incompatible-with-mcq-answer-extraction**

## Assumptions (per G1007)

1. K1493 is measured against the 62.1% base eval (F#536), not the corrupted
   12.5% Phase 4a eval in s1K. This is the only defensible reference given
   F#536.
2. The strip_thinking fix in s1K's `run_experiment.py` would apply symmetrically
   here; retained as a non-blocking fix to `run_experiment.py` if ever re-run.
3. LIMO paper's 63.3% AIME24 result is not an upper bound on what E4B 4-bit
   can achieve — the LIMO curation model is a substantially larger base.
   Capability-ceiling gap (REVIEW §Failure Mode 2) is a *second* independent
   kill axis; we rely only on the KL-divergence axis for this preempt.

## Conclusion

**KILLED** preemptively. LIMO shares D_competition-math support with s1K; the
−26pp MMLU-Pro degradation measured on the s1K adapter is a structural
property of (D_competition-math, D_MMLU-Pro, LoRA SFT), not a curation-specific
defect. No re-run can escape this unless the training distribution support
changes — which would no longer be "LIMO".

Overall status: **killed**
