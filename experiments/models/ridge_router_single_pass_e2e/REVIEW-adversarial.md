# Adversarial Review: exp_ridge_router_single_pass_e2e

## Verdict: KILL (confirmed) — but root cause analysis in PAPER.md is WRONG

The kill is correct — all 3 criteria failed. But PAPER.md misdiagnoses every failure.
The "Impossibility Theorem" in PAPER.md is false. Finding #340's impossibility structure
must be revised.

---

## 1. CRITICAL BUG: Oracle PPL == Ridge PPL (everywhere)

**Smoking gun:** `oracle_single_pass_ppl == ridge_single_pass_ppl` for ALL 10 domain
pairs. This is mathematically impossible if routing matters.

**Root cause:** The code (lines 1031-1043) uses **segment-level majority vote**, not
per-token routing:

```python
seg_a_majority = int(Counter(seg_a_preds.tolist()).most_common(1)[0][0])
seg_b_majority = int(Counter(seg_b_preds.tolist()).most_common(1)[0][0])
ridge_mixed = MixedAdapterMLP(model, mlp_adapters[pred_domain_a],
                               mlp_adapters[pred_domain_b], boundary, LORA_SCALE)
```

With 90%+ per-token accuracy on 128-token segments, majority vote recovers the correct
domain for virtually every segment. The `MixedAdapterMLP` then receives the SAME two
domains and the SAME boundary as oracle → identical PPL.

**Consequence:** K800 (per-token accuracy = 89.67%) is **completely decoupled from
K799 (PPL)**. The experiment measures per-token routing accuracy but the PPL pipeline
uses segment-level routing. K799 cannot test what it claims to test.

**What PAPER.md claims:** "10.33% of tokens are misrouted... expected NLL penalty =
0.1033 · 1.16 ≈ 0.12 nats per token, resulting in ≈ 10-12% PPL inflation."

**Reality:** Zero PPL inflation from routing. The majority vote masks all per-token
errors. The 62% PPL inflation is entirely from using the wrong baseline (4.684 vs 7.598).

---

## 2. Scale=20 Adapters Provide ZERO Benefit Over Base

Per-pair data from results.json:

| Pair | Base PPL | Oracle PPL | Adapter Effect |
|------|----------|------------|----------------|
| medical+code | 4.352 | 4.378 | **+0.6% WORSE** |
| medical+math | 4.369 | 4.406 | **+0.8% WORSE** |
| medical+legal | 8.509 | 8.560 | **+0.6% WORSE** |
| medical+finance | 8.712 | 8.781 | **+0.8% WORSE** |
| code+math | 4.162 | 4.171 | **+0.2% WORSE** |
| code+legal | 7.800 | 7.803 | **+0.04% WORSE** |
| code+finance | 9.438 | 9.410 | -0.3% better |
| math+legal | 8.036 | 8.042 | **+0.08% WORSE** |
| math+finance | 10.434 | 10.529 | **+0.9% WORSE** |
| legal+finance | 17.318 | 17.056 | -1.5% better |

**8 of 10 pairs:** adapters make PPL WORSE than no adapters at all.

This is consistent with Findings #328, #330, #337, #338: scale=20 adapters are
catastrophic. The real_data_domain_experts at scale=20 are harmful. The entire routing
question is moot — you cannot route to beneficial adapters when the adapters aren't
beneficial.

**What PAPER.md claims:** "The failure is NOT in individual components, but in composition."

**Reality:** The failure IS in individual components (scale=20 adapters), not composition.
Routing works perfectly (majority vote recovers correct domains). The adapters are harmful.

---

## 3. K799 Threshold Fundamentally Wrong (Acknowledged But Not Fixed)

MATH.md Section E correctly warns: "Finding #310 used adapters from
real_data_domain_experts/adapters/. Finding #313 used adapters from
tiny_routing_heads/adapters/."

Despite this warning, the K799 threshold is still 4.778 (based on Finding #313's 4.684
oracle). The correct oracle with these adapters is 7.598. The K799 threshold should have
been 7.598 * 1.02 = 7.750.

**Under corrected threshold:** K799 would PASS (ridge_ppl=7.598 <= 7.750). This is
trivially true because oracle==ridge (Bug #1 above).

---

## 4. K801: Theorem Proves 1-Pass, Code Measures 2-Pass

Theorem 3 predicts ~1.01x for a 1-pass pipeline. The code measures a 2-pass pipeline
(base forward + single-pass forward = 2.326x). The `theoretical_ratio_1pass` in
results.json is 1.013 — well within K801 threshold.

This is not a theorem failure. It's a measurement methodology mismatch. PAPER.md
should acknowledge this clearly.

---

## 5. PAPER.md "Impossibility Theorem" Is False

PAPER.md states:
> "A ridge regression router trained on IID single-domain hidden states will exhibit
> LOWER accuracy on mixed-domain concatenated sequences due to context-induced
> distribution shift."

**The per-token accuracy drop (98.3% → 89.67%) is REAL.** But it's irrelevant to the
architecture as implemented, because:
1. The PPL pipeline uses majority vote, not per-token routing
2. Majority vote is ~100% correct at 90% per-token accuracy
3. The architecture SHOULD use per-token routing (as designed in MATH.md) but doesn't

The "impossibility" is an artifact of measuring one thing (per-token) while using
another (per-segment majority). If the code implemented true per-token routing, the
accuracy drop would matter — but we don't know by how much because the adapters provide
no benefit anyway.

---

## 6. Finding #340 Must Be Revised

**Current impossibility structure (WRONG):**
> "A linear classifier trained on IID single-domain data cannot generalize to
> mixed-domain sequences without retraining."

**Revised impossibility structure:**
Three independent failure modes, each sufficient to invalidate:

1. **Experimental design error:** K799 threshold (4.684 + 2%) from wrong adapter set.
   Correct oracle is 7.598. Threshold was never achievable as designed.

2. **Implementation error:** PPL measurement uses segment-level majority vote, not
   per-token routing as designed in MATH.md. K800 (per-token) is decoupled from K799
   (PPL). The experiment cannot test its own hypothesis.

3. **Adapter quality:** scale=20 adapters provide no benefit (8/10 pairs worse than
   base). Routing to harmful adapters cannot improve PPL regardless of routing accuracy.

**What IS salvageable:**
- The per-token accuracy drop on mixed-domain sequences IS real (89.67% vs 98.3%)
- This WOULD matter if (a) per-token routing were implemented and (b) adapters helped
- Ridge router on IID data → mixed-domain context: genuine 8.6pp accuracy degradation
- Fix: retrain router on mixed-domain calibration data (as PAPER.md recommends)

---

## Required Actions

1. **Update Finding #340** — Replace "context-induced distribution shift is impossible
   to overcome" with the 3-way decomposition above. The distribution shift is real but
   was not tested (majority vote masks it).

2. **Do NOT re-run.** The fundamental issue is scale=20 adapter quality, which is
   already proven catastrophic (Findings #328, #330, #337). A re-run with scale=5
   adapters would be a different experiment.

3. **LEARNINGS.md must capture:** (a) majority vote vs per-token routing distinction,
   (b) scale=20 adapter harm confirmed again, (c) correct oracle baseline for future
   E2E experiments.

---

## Meta: What PAPER.md Got Right

- The observation that per-token accuracy drops on mixed-domain is genuine
- The recommendation to retrain on mixed-domain data is sound
- The recommendation to match adapter provenance is critical
- The measurement of the 2-pass latency overhead is accurate

What it got wrong: attributing PPL failure to routing errors when (a) routing errors
don't affect PPL (majority vote), (b) adapters don't help (scale=20), and (c) the
threshold was wrong.
