# PAPER — Pierre v5.2 Bankai Row Flip

## Verdict: **KILLED** (preempt; impossibility-structure F#291)

## Abstract

Pierre v5.2 proposed Bankai-style row flips (github.com/nikshepsvn/bankai)
adapted to ternary BitLinear: for each target row, increment or decrement
every entry by 1 and clip to {-1, 0, +1}. Greedy search over rows was to
find a sparse (≤100-flip) patch per domain. The approach is preempt-killed:
the row-level flip operation is mathematically identical to the per-entry
LoTA-QAF merge that v5.1 ran and v5.1's Finding #291 proved impossible on
ternary bases.

## Kill Criteria Assessment

| ID   | Text                                             | Result  | Evidence                                                                                      |
|------|--------------------------------------------------|---------|-----------------------------------------------------------------------------------------------|
| K733 | Zero domain signal (PPL same as base)            | **FAIL**| `clip(r + s, -1, +1)` saturates ⅓ of entries per flip by construction (MATH.md Theorem).       |
| K734 | Search > 30 min per domain                       | n/a     | Not run — irrelevant once K733 fails by reduction to F#291.                                   |
| K735 | Speed < 120 tok/s after apply                    | n/a     | Not run — v5.1 measured 138.3 tok/s with full merge; flip-only would exceed this.             |

## Prediction vs Measurement

| Prediction (MATH.md)                    | Source                     | Measurement                                      |
|-----------------------------------------|----------------------------|--------------------------------------------------|
| Behavioral ≤ 0.05                       | P1                         | v5.1 parent measured 0.003 (worst ω=4).          |
| PPL ratio ≥ 10× base                    | P2                         | v5.1 parent measured 80 543 923× at ω=4.         |
| Greedy cannot discover beneficial flip  | P3                         | Row-level op = v5.1 per-entry op; no escape.     |
| Fixed-point fraction = 1/3              | Theorem, sign-symmetric    | Confirmed by exhaustive case analysis (MATH.md). |

## Antipattern Match

**composition-bug / ternary-saturation**. The proposal believed row-level
granularity would avoid the v5.1 failure because fewer flips → fewer clips.
This is wrong: the boundary-clip rate is per-entry (1/3 expected), not
per-flip. Row-level granularity reduces total clips only by reducing total
flips — it does not improve the signal-to-destruction ratio.

## Why Impossibility, Not Hyperparameter Tweak

K733 cannot be lifted by any parameter change accessible to the proposed
runner:
  - MAX_FLIPS tuning: reduces destruction linearly with signal; ratio unchanged.
  - MAX_SEARCH_ITERS: explores same space, same per-step loss.
  - Scale-guided targeting: high-scale rows have more signal AND more destruction.
  - N_CAL_PPL: measurement granularity, not target.

The structural fix (F#291): move to K ≥ 5 (2-bit+sign base). Out of scope
for v5.x ternary.

## References

- Finding #291 (parent): `KILLED: LoTA-QAF lossless merge impossible on ternary base — 3 levels leaves no room for integer adjustment`.
- `micro/models/pierre_v51_lota_merge/results.json` — v5.1 empirical measurements of the identical operation.
- Bankai (upstream): github.com/nikshepsvn/bankai (1-bit XOR, not ternary).

## Limitations

Preempt verdict relies on reduction argument, not an empirical v5.2 run.
The code path is bit-identical (see run_experiment.py:180 `mx.clip(row,
-1, 1)`) to v5.1's merge at a per-row granularity; v5.1 measurements
(behavioral=0.003) therefore transfer as an upper bound on v5.2 outcomes.
