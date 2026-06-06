# LEARNINGS.md — T3.1: Pairwise Interference (KILLED)

## Core Finding

Weight-space cosine is NOT the correct interference predictor. Math/code adapters (low cosine 0.01-0.02) collapsed catastrophically under N=5 simultaneous composition (82→8%, 66→8%), while MMLU adapters (high cosine 0.10-0.17) degraded only moderately (83-87% retention). The actual failure mechanism is O(N-1) additive activation-space noise, not weight-space overlap.

## Why

Simultaneous block-diagonal composition activates all N adapters: `output = Σ A_i B_i x`. This creates N-1 noise terms per forward pass. Precision tasks (arithmetic chains, code execution) require exact sequential token generation — even small per-step noise compounds catastrophically over L=10-20 token chains. MCQ tasks (select A/B/C/D) are robust to logit shifts that don't flip the argmax. The sensitivity to composition noise is determined by task precision requirements, not adapter weight-space proximity.

## Impossibility Structure

Simultaneous N-adapter activation makes interference O(N-1) by construction. No weight-space regularization (Grassmannian, null-space, cosine minimization) can fix this — those address weight-space overlap, not activation-space SNR. The structural fix is routing: activating only the matched adapter makes interference exactly zero (not approximately).

## Implications for Next Experiment

T3.2 must implement PLE-M2P routing over the 5 existing adapters. Target: composed accuracy ≥ 90% of single adapter (the threshold K1051 failed here). Use n≥50 per domain (n=25 was too small for MMLU cluster claims). Verify that routing + existing adapters achieves what block-diagonal merging structurally cannot.

## Reviewer Caveats (non-blocking)

- n=25 too small for MMLU cluster ratios; treat 83-87% retention as provisional
- The exponential bound `acc ≤ acc_single × (1-(N-1)ε)^L` is a heuristic, not a theorem
- Verify Finding #225 composition method before inheriting its routing claim
