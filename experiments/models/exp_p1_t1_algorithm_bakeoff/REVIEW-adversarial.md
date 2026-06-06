# REVIEW-adversarial.md — T1.6 Algorithm Bake-off

**Verdict: PROCEED**

## Checklist

### 1. Prediction-vs-Measurement Table
Present in PAPER.md. All 5 predictions mapped. ✓

### 2. Kill Criteria vs results.json

| K | PAPER.md | results.json | Match |
|---|---------|-------------|-------|
| K1024 | LoRA_r6 wins (5.69e-06) | winner=LoRA_r6, composite=5.686e-06 | ✓ |
| K1025 | \|cos\|=0.00078 | abs_cos=0.000778 | ✓ |
| K1026 | sr=3.76 | stable_rank=3.756 | ✓ |
| K1027 | 106s max | max_train_time=105.86s | ✓ |

No fabricated numbers. All match exactly.

### 3. Finding Status

"supported" is correct. The bake-off goal (identify best adapter format) was achieved with
all K passing. P1/P2 refuted (HRA underperforms LoRA at equal params), which is itself a
finding: T1.2's impossibility extended to equal-params regime. The architecture decision
(LoRA r=6) is well-evidenced.

### 4. Math Errors

None found. Spot-checks:
- Composite score formula: (0.07-0.03+0.001)/(39936×0.18054) = 0.041/7206 = 5.69e-06 ✓
- Equal-params derivation: r_HRA = 6 × (2560+4096)/2560 = 6×2.6 = 15.6 → r=16 ✓
- Step time ratio: 0.353/0.181 = 1.95× as stated ✓

## Non-Blocking Notes (informational only)

1. **Noise floor:** n=100 GSM8K eval means 7% vs 5% = 2 answers. Winner is LoRA_r6 but
   with high uncertainty. The composite metric (integrating params+time) is more robust
   than accuracy alone — this is the right call.

2. **Gemma4 proxy:** All results on Qwen3-4B-4bit (Gemma4 still not loadable). T2.1
   on actual Gemma4 will be the true test of LoRA r=6.

3. **LoRA_r16 converged at step 247** (all others DNF at step 301). Higher-budget LoRA
   converges faster — worth noting when planning T2.1 step count.

4. **Impossibility structures** (Cayley/PoLAR/Givens excluded) are clearly documented
   in MATH.md with mathematical derivations referencing killed T1.x experiments. ✓

## Summary

Clean experiment. Results honest (refuted predictions labeled ❌, confirmed labeled ✓).
Composite metric correctly handles the speed/quality/params tradeoff. Decision to use
LoRA r=6 is well-motivated. Ready for T2.1.
