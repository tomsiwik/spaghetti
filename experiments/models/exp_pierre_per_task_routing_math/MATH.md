# MATH.md — Per-Task Routing for Math (GSM8K Regression Fix)

## Problem
Finding #831 fix uncovered: DARE/uniform/TIES all regress GSM8K -6.7pp consistently while preserving HumanEval/MedQA. Math reasoning is sensitive to fused-delta perturbation in a way other tasks aren't.

## Hypothesis
Hybrid serving: route math-shaped queries to single best math-domain adapter; route everything else through DARE composition.

## Architecture
- Binary classifier (microGPT-scale, talos-style C/NEON) decides math-vs-not
- Math → single domain_math adapter via PoLAR injection (no fused delta)
- Non-math → 7-adapter DARE composition via `_FusedDeltaLinear`

## Predictions
- K2143: GSM8K within 2pp of best single (70.0%) — closes regression
- K2144: HumanEval/MedQA preserved within 2pp of DARE result
- K2145: Math classifier ≥85% accuracy on held-out
- K2146: Routing overhead ≤5ms

## References
- Finding #831 (composition fix)
- Finding #58 (top-2 routing wins)
- DSv4 §2.3.1 (Lightning Indexer architecture pattern)
