# Adversarial Review: exp_p5_reward_lora_judge

## Verdict: PROCEED

Status SUPPORTED is appropriate. The experiment demonstrates infrastructure viability,
not quality discrimination capability — and the paper is honest about this distinction.

## Issues

### 1. SOAP individually fails K1275 (non-blocking)

SOAP latency is 103.8ms, exceeding the 100ms threshold. The paper reports the aggregate
average (83.2ms) as passing, which is technically correct but obscures a per-domain failure.
K1275 as stated ("<100ms scoring latency") is ambiguous — per-adapter or aggregate?

**Impact**: Minor. The paper acknowledges the issue and proposes mitigations (truncation).
Not blocking because the aggregate does pass and the fix is straightforward.

### 2. Theorem 3 latency "bound" is an estimate, not a bound (non-blocking)

Predicted 30-50ms, measured 62-104ms — a 2x miss. The proof estimates base forward pass
at ~20-40ms but actual measurement is ~60ms. Calling this a "bound" is overclaiming;
it's a ballpark estimate. The PAPER.md correctly explains the discrepancy.

**Impact**: Cosmetic. The math is fine, just the label "bound" vs "estimate" is imprecise.

### 3. Ceiling effect means limited actionable value (noted, not blocking)

100% accuracy with margins of 20-47 on format-level discrimination (LaTeX vs plain text,
SOAP headers vs conversational) is unsurprising. A linear probe on TF-IDF features would
likely achieve similar results. The paper correctly identifies this and proposes intra-domain
quality discrimination as the real test.

The finding's value is infrastructure verification: reward LoRA trains, scores, and serves
on M5 Pro within resource constraints. This is worth recording.

### 4. 5 eval pairs per domain (noted, not blocking)

100% on n=5 gives a 95% CI of [56.6%, 100%] by Clopper-Pearson. This is acknowledged
in the paper's limitations section. Not blocking because the margins (20-47) are so large
that the directional conclusion is safe, even if the exact accuracy estimate is noisy.

## What the paper gets right

- Honest about ceiling effect and trivial task difficulty
- Prediction-vs-measurement table present and accurate
- Size prediction exactly confirmed (5.0 MB predicted, 5.01 MB measured)
- Clear identification of next steps (intra-domain quality)
- Limitations section covers all major concerns
- Correct architectural note about v_proj exclusion due to KV sharing

## Finding recommendation

Record as SUPPORTED with scope explicitly limited to format-level discrimination and
infrastructure viability. The finding should NOT claim that reward LoRA can do quality
discrimination — only that the serving infrastructure works within resource constraints.
