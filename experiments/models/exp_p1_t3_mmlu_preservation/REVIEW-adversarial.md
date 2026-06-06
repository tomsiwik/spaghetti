# REVIEW-adversarial.md: T3.2 — MMLU Preserved Under N=5 Composition

**Verdict: PROCEED (with caveats; findings recorded)**

---

## Blocking Issues

**None.** The core finding (K1054 FAIL) is well-supported and actionable.

---

## Critical Observations

### 1. Theorem 1 Corollary Error (Theoretical)

MATH.md correctly proves `||Q_norm(s)||_RMS = sqrt(h_q)` for all s > 0. But the
corollary ("scale invariance of attention") is wrong. Magnitude clamping ≠ directional
invariance. The query direction rotates toward BAx as scale increases, changing attention
patterns even with constant magnitude. Data confirms this:

```
scale=6:  70.7%
scale=12: 64.0% (-6.7pp)
scale=18: 46.7% (-24.0pp)
```

This is a clean refutation. The PAPER.md correctly identifies the impossibility structure.
Finding: **Theorem 1 proves necessary but not sufficient condition for scale invariance.**

### 2. Base MMLU = 4.0% (Diagnostic Issue)

The base model gets 4.0% on 5 neutral MMLU subjects. This is far below random (25%).
Root cause: the base Gemma 4 E4B model likely fails to output parseable letter responses
without adapter priming. This means K1053/K1055 "PASS" is trivially satisfied and doesn't
measure what the kill criteria intended.

**Impact on findings:** K1054 is unaffected (measured against scale=6 baseline, not base).
K1053/K1055 are NOT meaningful as stated — they test format compliance transfer, not
knowledge preservation. This is a **non-blocking caveat** for the killed experiment.

**Future fix:** Measure base MMLU with greedy decoding + verbose output inspection to
verify whether the model generates parseable letters. If not, add a system prompt or
use a different MCQ evaluation format.

### 3. Scale Sensitivity Is Steep (Finding Confirmed)

The scale degradation is not just "missing the 3pp threshold by a little" — it's
6.7pp at scale=12 and 24.0pp at scale=18. This is a strong, monotonic effect.
The prediction (scale invariant within 2pp noise) was significantly off.

**Implication:** For P1, scale must be fixed at training value (6). Any future
experiment that loads adapters at higher scale MUST re-evaluate MMLU preservation.

---

## Non-Blocking Caveats

1. **n=75 for scale test** (3 subjects × 25): sufficient to detect 6.7pp and 24pp
   effects, but CI at 75 questions is ±11pp (Wilson). The K1054 failure is so large
   (Δ=24pp at scale=18) that it's solid. Scale=12 at 6.7pp is borderline given n=75.
   Future work should confirm scale=12 with n=125.

2. **Scale test uses math adapter only**: The scale sensitivity might differ for
   MCQ-trained adapters (legal/finance) since their B matrices are more MMLU-aligned.
   Non-blocking since scale=6 is already confirmed safe.

3. **PAPER.md prediction table is correct**: All three rows accurately represent
   prediction vs. measurement. The K1054 FAIL is noted with actual numbers.

---

## Verdict: PROCEED

The experiment is correctly classified as KILLED (K1054 fails; main theorem corollary
refuted). PAPER.md is complete and accurate. The findings are:

1. QK-norm clamps magnitude, not direction → scale is NOT invariant
2. Use scale=6 for all P1 adapters
3. Base MMLU = 4% reveals format compliance artifact (need corrected baseline)

Emit `review.proceed` → Analyst writes LEARNINGS.md.
