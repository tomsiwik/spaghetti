# Review: exp_p0_n25_vproj_composition

## Verdict: PROCEED (kill justified, findings actionable)

## Evidence Consistency: PASS

results.json values match PAPER.md claims exactly:
- K1324: 0.55 = mean of 25 vocab_retention values (verified: sum=13.75/25)
- K1325: -2.0 from chemistry/biology/neuroscience (solo_vocab < base_vocab)
- K1326: -8.07% PPL improvement (13.11 -> 12.05)
- K1327: 1.032x latency ratio (5.65s -> 5.83s)

No fabricated evidence.

## Prediction-vs-Measurement Table: PRESENT

7 predictions, 5 match or exceed, 2 miss (mean retention and min domain).
Misses correctly attributed to adapter quality floor, not theory failure.

## Root Cause Analysis: STRONG

The paper correctly separates three distinct effects:
1. **Adapter quality** (~70%): 12/20 new adapters are dead (zero solo improvement)
2. **SIR degradation** (~30%): P8 domains 113% -> 67%, ratio 0.59 vs predicted 0.408
3. **Composition mechanism**: NOT a cause (PPL improves, ensemble creates signal)

The impossibility structure is mathematically sound: with f=0.44 dead adapters,
K1324 requires R_active > 125% which is unreachable for most domains.

## Issues (non-blocking)

1. **Infinity handling in rate_retention**: results.json contains `Infinity` for 5
   zero-solo domains. The mean retention uses vocab_retention (avoids Infinity) but
   this is not explicitly stated in PAPER.md. Minor documentation gap.

2. **v_proj null finding undersold**: ALL adapters have v_proj.lora_b = 0. This
   means the dual-target (v_proj+o_proj) framing is misleading — it's effectively
   single-target (o_proj only). This deserves its own finding or at minimum a
   prominent callout. Currently buried in a subsection.

3. **n_eval = 5**: Each domain evaluated on only 5 queries. The rate_retention
   values (0%, 50%, 100%, 150%, 200%) are quantized to 1/5 increments, which
   limits statistical power. Noted but acceptable for a scaling stress test.

## Kill Status: CORRECT

K1324 and K1325 fail. The experiment type is frontier-extension. "Killed" is the
right status — the predictions from MATH.md did not hold at N=25 with this adapter
quality level. The findings (composition works, adapters are the bottleneck) are
supported and actionable.

## What the Analyst Should Capture

1. Composition mechanism works at N=25 (PPL improves, latency bounded)
2. SIR degradation 0.59x (better than 0.408x predicted) — ensemble partially compensates
3. Adapter quality is the binding constraint, not composition
4. vocab_retention metric is broken for domains where solo < base
5. v_proj effectively untrained across all adapters (lora_b = 0)
6. 6/11 zero-solo domains gain signal under composition (constructive interference)
