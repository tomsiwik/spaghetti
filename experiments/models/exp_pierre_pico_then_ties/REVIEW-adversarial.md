# REVIEW-adversarial.md — exp_pierre_pico_then_ties

## Verdict: PROCEED

## Adversarial Checklist

| Check | Result |
|-------|--------|
| (a) results.json verdict matches DB status | ✓ SUPPORTED |
| (b) all_pass consistent | ✓ All 4 KCs pass |
| (c) PAPER.md verdict matches | ✓ |
| (d) is_smoke → provisional | N/A (not smoke) |
| (e) KC not modified post-run | ✓ Thresholds match MATH.md |
| (f) No tautological KC | ✓ K4 tests real orthogonality |
| (g) Code measures what MATH.md describes | ✓ Task accuracy on 3 benchmarks |
| (h) Composition math correct | ✓ `dW = s_t * (A_t @ B_calib)` per adapter, then TIES on stacked deltas |
| (i) LORA_SCALE safe | ✓ scale=6.0 |
| (j) No single-sample routing applied globally | ✓ Merge method, not routing |
| (k) No shutil.copy fake adapters | ✓ |
| (l) No hardcoded pass | ✓ |
| (m) Model consistent | ✓ gemma-4-e4b-it-4bit throughout |

## Non-blocking Flags

- (o) N=50 — margins tight (K1: +0.33pp over threshold, K4: +0.3pp over threshold). PAPER.md honestly flags this and recommends N=200 confirmation. Acceptable for a combinatorial test.

## Notes

- Composition code is correct: Pico calibration applied per-adapter in B-space via SVD, then full deltas materialized, then TIES (trim/sign-elect/disjoint-mean) on the flattened delta stack. No shortcuts.
- The inline Pico reimplementation (lines 82–115) is justified: the public Pico API returns only the merged result, not per-adapter calibrated B's.
- GSM8K regression (−4pp vs Fisher-Rao) is noted but does not violate any KC. The avg gain is real (+3.3pp).
