# Adversarial Review: exp_p9_ffn_memory_map_gemma4

## Verdict: PROCEED (KILLED, Finding #514)

## Verification

All 3 kill criteria verified against results.json:
- K1372: pattern_rate = 0.2372 vs threshold 0.5, pass:false ✓
- K1373: clustering_ratio = 1.66 vs threshold 2.0, pass:false ✓
- K1374: agreement_rate = 0.001009 vs threshold 0.01, pass:false ✓

Prediction-vs-measurement table present and complete in PAPER.md.
KILLED status appropriate — all criteria fail, impossibility structure derived.

## Non-Blocking Issues

1. **Predictions contradict risk analysis.** MATH.md "What Could Kill This" correctly identifies GeGLU spreading activation (K1) and 4-bit degrading values (K3) as risks, yet predictions assume success (55-70% pattern rate, 1.5-4% agreement). The proof should have bounded the expected degradation from these mechanisms rather than predicting Geva-like numbers with a caveat.

2. **N=100 vs paper's 100K+ makes statistical claims fragile.** The impossibility structure lists "insufficient data" as a compounding factor, which undermines the other two factors — we can't cleanly attribute the failure to GeGLU or quantization when sample size is 1000x smaller. Acknowledged in PAPER.md but doesn't invalidate KILLED status since margins are large (23.7% vs 50%).

3. **MATH.md assumed double-wide layers (d_ff=20480 in layers 22-41) that don't exist in E4B.** Experiment discovered all 42 layers have d_ff=10240. PAPER.md correctly reports this. Minor — architecture prediction was wrong, but doesn't affect kill criteria.

4. **Impossibility structure compounds 3 factors without quantifying individual contributions.** Which factor dominates? The 265x-above-random K3 result suggests neurons DO store value information (just degraded by quantization). Running on full-precision model would isolate quantization vs GeGLU. Not actionable for our project (we use 4-bit), but worth noting.

## Downstream Impact
- exp_p9_ffn_targeted_edit: should be KILLED by dependency (neuron editing not viable)
- Confirms LoRA subspace approach is the correct abstraction level for adaptation
