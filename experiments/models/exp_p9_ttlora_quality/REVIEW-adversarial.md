# Adversarial Review: exp_p9_ttlora_quality

## Verdict: PROCEED

## What's Solid

1. **Math is clean.** Theorem 1 (param count) and Theorem 2 (rank preservation) have proper proofs with correct citations (Oseledets 2011). Predictions match measurements: 1,518 params/layer exact, total 64,260 vs predicted 63,756 (+0.8% explained by heterogeneous layers).

2. **Kill criteria are honest.** All three pass with clear margins. K1357 quality ratio 0.844 well above 0.60 threshold. K1358 size 154 KB well under 200 KB. K1359 convergence verified by loss trajectory in results.json.

3. **Behavioral claim is grounded.** PAPER.md correctly identifies the quality gap as "reasoning shortcuts" not degenerate output, consistent with the Kronecker submanifold constraint from the Corollary.

## Non-Blocking Notes

1. **Mixed-precision size comparison.** The "20x adapter size compression" compares float16 TT-LoRA (154 KB) to float32 LoRA (3.1 MB). At equal precision, compression would be ~10x. The **param-count compression (12.4x)** is precision-independent and should be the primary metric. The "20x" number is technically correct but potentially misleading. Not blocking because the param compression is the real result.

2. **Training speed cost under-discussed.** TT-LoRA trains 4x slower (5258s vs 1295s). For the "$2, 10 min" adapter goal, this matters. At 1000 steps on M5 Pro, TT-LoRA takes ~87 min vs LoRA ~22 min. Future work should address whether fewer steps suffice (TT-LoRA reached lower final loss 0.37 vs LoRA 0.40, suggesting possible over-training).

3. **results.json has two LoRA size fields.** `lora_train.adapter_size_bytes=9,579,720` (~9.1 MB) vs `lora_adapter_bytes=3,192,832` (~3.0 MB). Likely the former includes optimizer state or non-weight data. PAPER.md uses the latter, which is reasonable.

## Status Assessment

"Supported" is appropriate. All predictions match within stated ranges. The quality ratio (84.4%) falls within the predicted 60-90% band. For "conclusive", we'd want a tighter prediction (e.g., 80-90%) confirmed. "Supported" is the right call for a verification experiment with a wide predicted range.
