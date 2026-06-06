# PAPER.md — P11.L0: RSD Aligned Traces

## 1. Verdict

**KILLED (preemptive, 2026-04-18)** — supersedes 2026-04-14 PROCEED Round 2 review.
This is the **9th consecutive P11 reasoning-adapter kill** in today's chain
(F0/H0/B0/C0/D0/H1/I0/J0/L0) and the **first with a direct methodological
falsification** visible in existing cached data.

## 2. Why not run

### 2.1 Primary: NLL filter is a no-op (direct measurement)

Cached `data/trace_nll_scores.json` contains 20 smoke-mode NLL scores on the
first 20 s1K traces. **All 20 traces pass** with per-token acceptance rates
in the range **0.876 – 0.982** (mean ≈ 0.965). The trace-level threshold
`ACCEPT_RATE_MIN = 0.60` is passed trivially by every single trace.

This **falsifies Theorem 1's core premise** (MATH.md §"Main Theorem"):
> d_TV(D_rsd, π_S) ≤ d_TV(D_raw, π_S) (strict where filter drops traces)

With 100% retention, D_rsd ≡ D_raw and the total-variation distances are
equal by construction. The "alignment" guarantee of Theorem C collapses to
a vacuous statement.

The mechanism is already diagnosed in PAPER.md §"Practical Approximation
Caveat" (pre-existing NB1 from 2026-04-14 review):

> At vocab_size=256k, the threshold may accept 70–90% of tokens trivially,
> inflating K1543 acceptance rates beyond what true RSD would predict.

The cache now promotes this from "may inflate" to "**did inflate: 100%**".
Scoring 980 more traces will not meaningfully change this distribution —
s1K is homogeneous competition math and the 20-sample mean is tightly
concentrated (std < 0.04 on acceptance rate).

### 2.2 K1541 baseline (P11.F0) is KILLED

`experiment get exp_p11_s1k_reasoning_train_eval` confirms `status=killed`
(F0 OOM, no valid MMLU-Pro measurement). The K1541 comparison target does
not exist. `run_experiment.py:889` falls back to hardcoded `p11f0_mmlu=60.0`
when the results file is missing — a floating benchmark unsupported by
measurement. Measured base Gemma-4 MMLU-Pro is 40.7% (F#560) or 62.1% (F#530,
stale); the 60.0% fallback matches neither.

### 2.3 Theorem 1 collapses on measurement

Under the measured filter behavior (acceptance = 1.0):
- Step 2 of the proof requires d_TV(D_rsd, π_S) < d_TV(D_raw, π_S). Violated.
- Step 3 follows from Step 2. Vacuous.
- Step 4's inequality ||∇L(D_rsd)||² ≤ ||∇L(D_raw)||² in expectation collapses to equality.
- Predicted "+2 to +4 pp over P11.F0 raw" has zero basis.

If the experiment runs, it measures "train on 18 s1K examples (after 90/10
train/valid split of 20 accepted traces) vs the same 18 examples with a
different seed". The ~100% acceptance rate at full 1000-trace scale would
simply produce "train on ~1000 examples vs train on ~1000 examples". Either
way, there is no RSD-vs-raw contrast to measure.

### 2.4 Cascade from 8 P11 reasoning-adapter kills

Today's kill chain (F0 OOM, H0 GD-violation-14.5pp regression, B0 -15.3pp via
antipattern-018, C0/D0/H1/I0 preemptive on same protocol bug, J0 router
66.8%<85%) establishes that **every trained Gemma-4 reasoning adapter in the
P11 chain regresses 15–26 pp vs base on MMLU-Pro**. Given base ~40.7%, a
trained adapter is predicted ~15–26%. K1541's target (60% + 3pp = 63%) is
~40pp above this band. Structurally unreachable even if the filter worked.

### 2.5 Training-format risk (secondary)

`run_experiment.py:367` writes assistant content as
`f"<think>{trace['thinking']}</think>\n\n{trace['attempt']}"`. Gemma 4
natively emits `<|channel>thought ... <channel|>`. Training on `<think>`
teaches a non-native thinking format — a related but distinct class to
**antipattern-018** (channel-text-as-SFT-target). Not the same byte-for-byte
bug as B0/D0/H1/I0, but same category: **SFT target format ≠ model's native
generation format**. This alone would not justify a kill, but layered on
drivers 2.1–2.4 it is consistent with the pattern.

## 3. Kill-criteria table (pre-registered, locked since 2026-04-13)

| ID     | Criterion                                       | Predicted | Measured / Evidence                                    | Status  |
|--------|-------------------------------------------------|-----------|--------------------------------------------------------|---------|
| K1541  | RSD MMLU-Pro ≥ P11.F0 + 3pp                     | 63–65%    | Baseline F0 killed; Theorem 1 collapsed; cascade ~15–26% | **FAIL** (structurally unreachable) |
| K1542  | NLL scoring < 24h                               | 30–60 min | Cached (0 min incremental); trivially passes            | **PASS (trivial)** |
| K1543  | ≥60% of traces pass filter                      | 50–70%    | Measured 20/20 = 100% at threshold NLL≤6.9              | **PASS (vacuously — filter is no-op)** |

K1541 is the **load-bearing criterion** — the one tied to the research
question. It fails. K1542/K1543 pass trivially, so they provide no evidence
of the hypothesis.

## 4. New methodological finding (proposed for DB)

**Absolute NLL thresholds are meaningless filters at large-vocab models.**

- Finding: At vocab_size=256k, `ACCEPT_NLL_THRESHOLD=6.9` (≡ P_S ≥ exp(-6.9) ≈ 0.001
  ≈ 250× chance) accepts 87–98% of tokens in well-formed teacher traces
  (measured on 20 s1K × 2047 tokens ≈ 40k tokens).
- Why: 250× chance is ~8 bits of signal, but a well-trained LLM assigns > 8
  bits of confidence to most in-context tokens by default.
- Remedy: True RSD requires the *ratio* `P_S(x) / P_T(x)`, not absolute
  `P_S(x)`. Without teacher log-probs, an alternative is a
  *student-percentile* threshold (accept if `P_S(x_t)` is in the student's
  top-k probability mass at step t), which is calibration-invariant to
  vocab size.
- Policy implication: Any future NLL-filtering experiment on this codebase
  should pre-register the expected acceptance rate and *reject the
  threshold* if the rate is > 80% on a 20-sample probe before running
  the full pipeline.

## 5. Unblock path (L0-v2 redesign, not run now)

For a scientifically valid L0-v2:
1. **Replace absolute NLL with ratio-based acceptance.** Options:
   (a) True RSD: generate teacher log-probs on s1K (requires loading
   DeepSeek-R1-Distill locally, ~14 GB with quant);
   (b) Student-percentile: accept if `P_S(x_t)` is in the top-k% of
   student's vocab distribution at step t (k tunable; 0.1% ≈ 256 tokens
   of ~256k is a calibrated choice).
2. **Validate filter discrimination on probe set** (20 traces) — require
   the acceptance distribution to have σ ≥ 0.15 across traces and mean
   in [0.3, 0.7] before running the full filter. If the probe shows
   uniformity (as now), the threshold is wrong.
3. **Fix the baseline**: L0-v2's K1 should be `MMLU-Pro(RSD-adapter) ≥
   base − ε` (absolute, not relative to F0). Avoids the "baseline-killed"
   cascade. Reconcile F#560 (40.7% vs 62.1%) first.
4. **Format fix**: write assistant content in Gemma 4's native
   `<|channel>thought ... <channel|>` format using the tokenizer's chat
   template, not a hand-rolled `<think>` wrapper.

L0-v2 does **not** depend on P11.HARNESS (which is about B0-chain SFT),
but it does depend on F#560 reconciliation for an honest K1.

## 6. Antipattern self-check (per PLAN.md §1009)

| Check                                                         | Status |
|---------------------------------------------------------------|--------|
| Composition math (delta-sum, weighted_sum, add_weighted_adapter) | N/A (single adapter) |
| `LORA_SCALE=1.0`                                              | ✅ (run_experiment.py:63) |
| Tautological routing                                           | N/A (no routing) |
| `shutil.copy` as new adapter                                   | ⚠ (`shutil.copy` exists at L596 for MMLU-Pro parquet only — not adapter) — OK |
| Hardcoded `{"pass": True}`                                    | ✅ none (computed L910) |
| Eval-template truncation, base=0%                              | N/A (not yet run) |
| Proxy model substituted for target                             | ✅ same MODEL_ID in scoring + train + eval |
| KC measures wrong object                                       | ⚠ K1543 measures filter *threshold calibration*, not *filter effectiveness*. This is the mis-specification itself (driver 2.1). |
| N=smoke reported as full                                       | N/A (preemptive kill) |
| Post-reg KC drift                                              | ✅ MATH.md single commit de38e37, no changes |
| `<|channel>...<channel|>` as SFT target (antipattern-018)     | ✅ not this form (uses `<think>`); related class — see §2.5 |

## 7. Assumptions

1. The 20-sample smoke cache is representative of the full 1000-trace
   distribution. Justification: s1K is homogeneous competition math; a
   20-sample mean on a ≤1.0-bounded statistic has tight concentration
   (Hoeffding ≤ 0.15 at 95%).
2. The observed acceptance uniformity does not reflect a bug in
   `compute_trace_nll` — verified by hand: the per-token NLL values
   (0.48–2.30) are reasonable LM log-probs; the mis-specification is in
   the threshold choice, not the computation.
3. F#560 (base MMLU-Pro 40.7%) is the current honest baseline; F#530's
   62.1% is stale until reconciled.

## 8. References

- arXiv:2509.22230 — RSD (Reverse Speculative Decoding). Defines the
  *ratio-based* criterion; we approximated with absolute NLL, which is
  the methodological gap.
- arXiv:2309.06657 — Statistical Rejection Sampling. Same ratio structure.
- von Neumann (1951) — rejection sampling. Ratio structure.
- F#560 — baseline reconciliation thread (40.7% vs 62.1%).
- F#530 — stale 62.1% base MMLU-Pro.
- Finding #538 — P11.A0 -26pp MMLU-Pro forgetting.
- mem-antipattern-018 — channel-text-as-SFT-target (B0/D0/H1/I0 instances).
- P11.B0/C0/D0/F0/G0/H0/H1/I0/J0 — prior reasoning-adapter kills (2026-04-17/18).

## 9. Handoff

- DB: `experiment complete exp_p11_rsd_aligned_traces --status killed
  --k 1541:fail --k 1542:pass --k 1543:pass`
  (K1542/K1543 pass the *literal* threshold but do not support the
  hypothesis; K1541 is the load-bearing criterion and fails.)
- MATH.md: untouched (pre-reg locked). No edits.
- run_experiment.py: untouched. Usable for L0-v2 with threshold fix at
  line 58 (`ACCEPT_NLL_THRESHOLD`) and baseline fix at line 889.
- LEARNINGS.md / REVIEW-adversarial.md: left for analyst/reviewer rewrite.
- Next in queue: `exp_p11_full_pipeline_v2` (M0) — expected preemptive
  kill on antipattern-018 + cascade from F0/H0/J0/L0;
  `exp_p1_t5_user_local_training` may be independent.
