# LEARNINGS: Zero-shot precision transfer 4→8 bit

## Verdict: PROVISIONAL (downgraded from KILLED by reviewer)

## Core Finding

Adapter benefit transfers with 90–98% PPL-gain retention from 4-bit to 8-bit base (median R=0.9459, code 0.90 / math 0.95 / medical 0.98). Marginal miss on the 0.95 threshold. **BUT:** K2 is a PPL-derived ratio with target metric `not_measured`, so per rule (t) / F#666 the kill is unsafe. Real behavioral consequence on downstream tasks remains unknown.

## Why the downgrade

1. **K2 is a proxy, not a target.** MATH.md labeled R as "target/behavioral" but R = gain₈/gain₄ is still PPL. Per codebase r≈0.08 PPL↔task, a 5–10% PPL-gain loss need not translate to task degradation.
2. **Marginal kill (0.0041 below threshold) on N=3 with one saturated domain.** Medical adapter PPL=1.0 means that R inherits noise, not signal.
3. **Structural PASS (K1) + target `not_measured` = PROVISIONAL**, not KILLED.

## Mechanism (if real)

Ceiling effect: 8-bit base already fixes part of what the 4-bit-trained adapter corrects (code base PPL 3.82→3.24, 15% gap). Block quantization is *not* spectrally equivalent to F#97's SVD perturbation — it shifts the weight manifold in a structurally correlated way. Transfer loss therefore domain-dependent: worst on easy domains (code), saturated on hard domains (medical).

## Implications for Pierre

- **Do not close the "train on 4-bit, deploy on 8-bit" question yet.** PPL-ratio evidence is suggestive but insufficient. The follow-up `exp_g4_zs_base_transfer_4bit_fp16_full` (P3, task-accuracy KC on HumanEval / GSM8K / MedQA) is the binding test.
- **Process lesson:** any KC defined on PPL (ratio, delta, dispersion, per-sample variance) is a proxy. Adding per-sample stats does not upgrade it to behavioral. Pair with a downstream task metric or mark PROVISIONAL at registration.
- **F#97's guarantee is rank-perturbation only.** Do not cite it for quantization-perturbation claims without an explicit bridge lemma.

## What to explore next

- Run the `_full` follow-up with task-accuracy KCs to close the behavioral question.
- Reverse direction (8-bit-trained adapter on 4-bit base) as a symmetry check for the ceiling-effect explanation — but only after behavioral KC is established.
- Higher-rank adapters (r>6): secondary, gated on the follow-up resolving.
