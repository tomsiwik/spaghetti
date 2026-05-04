# Learnings: exp_pierre_composition_method_ablation

## Core Finding

Uniform 1/N averaging is the optimal composition method. Gated routing (M2P) adds complexity without benefit because the gate has zero calibration — it classifies domains perfectly (99.6%) but confidence has no correlation with per-prompt correctness (Spearman ρ=0.009, p=0.93).

## Why

The gate converges to near-deterministic top-1 selection (weight 0.993), making gated ≈ hard top-1 in practice. Hard top-1 loses cross-domain regularization that uniform averaging provides: MedQA drops from 60.0 (uniform) to 50.0 (top-1/gated). The gate solves classification, not calibration — knowing *which* domain tells you nothing about *whether* the answer will be correct.

## Key Numbers

| Method | GSM8K | HumanEval | MedQA | Avg |
|--------|-------|-----------|-------|-----|
| M1: Uniform 1/N | 63.3 | 70.0 | 60.0 | **64.4** |
| M2: Hard top-1 | 63.3 | 73.3 | 50.0 | 62.2 |
| M3: M2P-gated | 56.7 | 76.7 | 50.0 | 61.1 |

23% of prompts (21/90) fail all methods — these are hard queries, not a routing problem.

## Implication for Pierre v1

Use uniform 1/N via `_FusedDeltaLinear`. Repurpose the gate for **sparse adapter selection** (which adapters to load) rather than continuous weighting (how much weight each gets). This simplifies the runtime and removes a component with zero demonstrated value.
