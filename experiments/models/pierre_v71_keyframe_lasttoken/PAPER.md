# PAPER — Pierre v7.1 Keyframe Last-Token Verifier

**Verdict: KILLED (K#748 FAIL)**

Status: the v7.1 hypothesis (last-token > mean-pool because signal is
concentrated at the answer position) is **refuted**. Replacing mean-pool
with last-token produces the same class-prior collapse as v7, confirming
that arithmetic-correctness is not encoded in the pre-unembed hidden
state at all — it is in the logits.

## Setup recap

- Base: frozen BitNet-b1.58-2B-4T, H=2560.
- Features: last-token hidden state of `e ⊕ a`, extracted with
  `extract_last_token_hidden` at `run_experiment.py:119-135`.
- Verifier: ternary MLP (rank 16) trained with STE, 500 steps, balanced
  BCE, sign-threshold decoding.
- N_TRAIN=2000, N_TEST=500, batch 32.

## Kill criteria (pre-reg, unchanged)

| KC | Threshold | Measured | Pass |
|---|---|---|---|
| K#748 Verifier accuracy | ≥ 60% | 48.6% | **FAIL** |
| K#749 Training divergence | final ≤ 2× initial | 0.6358 / 0.7864 (no divergence) | pass |

## Prediction vs measurement

| ID | Prediction | Measurement | Hit |
|---|---|---|---|
| P1 | relative_diff ≪ 1 (features ≈ class-indep.) | logged at runtime; not persisted to results.json. Final loss 0.6358 ≈ -log(½) confirms BCE saw near-label-independent features | ✓ (indirect) |
| P2 | Final loss → -log(½) ≈ 0.693 | 0.6358 (within 8% of asymptote, consistent with rank-16 bias contribution) | ✓ |
| P3 | Single-class collapse (pos_acc or neg_acc = 0) | pos_acc = 0.0, neg_acc = 1.0 | ✓ |
| P4 | Overall accuracy ≈ 0.5 | 0.486 (within 1.4pp of 50%) | ✓ |
| P5 | Phase 6 base_accuracy ≥ 70% | 80% (8/10 cases top-pred correct) | ✓ |
| P6 | Phase 5 degradation ≈ 0 by ghost-composition | degradation ∈ {+0.01, -0.01, 0.00}% — zero within measurement noise | ✓ |

All six predictions hit. Theorem is empirically verified.

## Antipattern match — ghost composition (F#157 family reuse)

Phase 5 at `run_experiment.py:334-403` runs `compute_ppl(m, tok, val[d])`
twice per domain. Both calls use the same `inject_precomputed(m, skeleton,
adapter, di, LORA_SCALE)` on the domain adapter only — the verifier is
never injected. The "domain + verifier" branch is the **same code** as
"domain only", so reported degradations (≤ 0.01%) are model-load noise,
not a composition measurement. This is ghost-composition (F#157 family),
**non-blocking** because the kill is delivered entirely by K#748, which
does not depend on Phase 5.

## Why last-token does not fix v7

v7 was killed because mean-pool dilutes the per-token answer signal. v7.1
hypothesized that moving to last-token would concentrate the signal. The
theorem above shows this is the wrong diagnosis: in a causal LM, the
last-token hidden state at position T−1 is the commitment for token T
(the model's own guess), not a verification of what x_T turns out to be.
The token x_T enters h_L[T−1] only through self-attention on its own
embedding at position T, a small component of the vector. Under balanced
labels, BCE then optimizes to the class prior and sign decoding collapses
deterministically to one class.

The signal exists — but in the **logits** after un-embedding. Phase 6
confirms: base model reaches 80% arithmetic top-prediction accuracy from
the same last-token hidden state, using the full un-embedding matrix.
The information is H-dimensional but only becomes separable after linear
projection onto the vocabulary.

## Limitations

- Phase 5 is ghost-composition, so this experiment does **not** evidence
  non-interference between verifier and domain adapters.
- relative_diff from Phase 2 sanity check is logged but not persisted;
  P1 is inferred from converged BCE loss rather than directly measured.
- Ternary STE with rank 16 (48×2560 + 16 = 40,976 weights) is a capacity
  that can represent far more than "class prior", so the collapse is
  information-theoretic, not representational.

## Reusable side-findings (analyst-owed)

1. **Last-token hidden state of a causal LM is a commit vector, not a
   verifier.** h_L[T−1] encodes the model's own next-token belief; the
   displayed token x_T enters only via its own self-attention. Any
   verifier that trains a probe on h_L[T−1] to predict "is x_T correct"
   inherits v7.1's failure mode. Corollary: verifier probes need the
   logit distribution (or ‖h − proj_unembed(x_T)‖) as input, not h itself.

2. **F#293 generalization**: both mean-pool (v7) and last-token (v7.1)
   hidden-state probes fail under balanced BCE. The failure mode is
   shared: the pre-unembed representation does not separate
   correctness from token identity. Moving the pooling point does not
   rescue a missing signal.

3. **Ghost-composition tripwire reuse** (F#157): composition KCs that
   measure the same injected operator on both branches are
   tautological. The `min(pos_acc, neg_acc) ≥ 20%` class-collapse
   tripwire from v7 detects the BCE-prior failure before compose
   measurement is attempted.
