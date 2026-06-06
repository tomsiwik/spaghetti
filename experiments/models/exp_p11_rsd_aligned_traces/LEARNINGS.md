# LEARNINGS.md — P11.L0: RSD Aligned Traces

## Core Finding

KILLED (preemptive, 2026-04-18) — 9th consecutive P11 reasoning-adapter kill and the **first by cached-data falsification** (not protocol-bug recurrence or measurement-run failure). All 3 KCs evaluate adversely via 5 independent drivers: K1541 fails structurally (baseline KILLED, cascade band ~15–26% vs target ≥63%); K1542/K1543 pass trivially but vacuously — the NLL filter accepts 20/20 traces at per-token rate 0.876–0.982 (mean 0.965), falsifying Theorem 1 Step 2 by measurement.

## Why

- **Driver 1 (load-bearing, novel):** Absolute NLL threshold 6.9 at vocab=256k is an 8-bit-of-signal filter on a ≥8-bit-confident LM. The cached `data/trace_nll_scores.json` (20 traces) shows ≥87% acceptance per trace with std <0.04 — structurally a no-op. True RSD needs ratio `P_S(x)/P_T(x)`, not absolute `P_S(x)`.
- **Driver 2:** K1541 anchor experiment F0 is KILLED; `run_experiment.py:889` falls back to hardcoded `p11f0_mmlu=60.0` that matches neither measured (40.7%, F#560) nor cited (62.1%, F#530) baseline — floating benchmark.
- **Driver 3:** Theorem 1 collapses algebraically once Step 2 fails — Steps 3–4 become vacuous/equality; predicted +2–4 pp has zero mechanistic basis.
- **Driver 4:** 8-kill cascade establishes P11 trained adapters regress 15–26pp; K1541 target ~40pp above this band.
- **Driver 5 (secondary):** `<think>...</think>` SFT format is same *category* as antipattern-018 (SFT format ≠ native generation), distinct instance — non-load-bearing for kill.

## Implications for Next Experiment

1. **New antipattern promoted:** ABSOLUTE-LOG-PROB-THRESHOLD-AT-LARGE-VOCAB (mem-antipattern-019). Any filtering experiment at vocab ≥100k must pre-register expected acceptance rate and reject threshold if probe-set (20 traces) shows σ <0.15 or mean outside [0.3, 0.7]. Generalizes beyond P11.
2. **L0-v2 unblock path:** Replace absolute NLL with (a) true RSD using teacher log-probs (requires loading DeepSeek-R1-Distill, ~14 GB), or (b) student-percentile threshold (top-k% at each step, calibration-invariant to vocab). Change KC from relative-to-F0 to `base − ε` absolute (avoids baseline-killed cascade). Fix SFT format to Gemma 4 native channel tokens via tokenizer chat template.
3. **Protocol lesson for reviewers:** 2026-04-14 Round 2 PROCEED missed already-cached `data/trace_nll_scores.json` that falsified Theorem 1 premise. Future "design-only" reviews must grep `data/*.json` for cached results before endorsing.
4. **F#560 baseline reconciliation (40.7% vs 62.1%)** remains open and blocks honest absolute K1 design across the remaining P11 chain.
5. **M0 (next in queue):** expected preemptive kill on antipattern-018 + cascade; reviewer should expect 10th P11 kill.
