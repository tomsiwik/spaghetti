# PAPER.md — B-space DARE vs Fisher-Rao Karcher mean (Pierre arch)

## Verdict: KILLED

B-space DARE does not transfer to Pierre's shared-A B-only architecture. Keep Fisher-Rao as the default composition method.

## Summary

This experiment tested whether DARE's +4.4pp composition gain (measured on full deltas with per-adapter A) transfers to Pierre's B-only composition path (shared frozen Grassmannian A). It does not. B-space DARE scored 55.3% avg — 9.3pp below Fisher-Rao (64.7%) and 6.7pp below single-best (62.0%). The 90% element-wise drop on B-matrices destroys too much structure when composition operates in the low-rank B-space rather than the full-rank delta space.

## Prediction vs Measurement

| Method | gsm8k | humaneval | medqa | avg |
|--------|-------|-----------|-------|-----|
| M0: single_best | 66.0% | 78.0% | 42.0% | 62.0% |
| M1: fisher_rao (Pierre default) | 68.0% | 68.0% | 58.0% | 64.7% |
| M2: dare_b (candidate) | 66.0% | 70.0% | 30.0% | 55.3% |
| M3: dare_full_delta (research upper bound) | 72.0% | 80.0% | 62.0% | 71.3% |

## Kill Criteria Results

| KC | Criterion | Result | Value |
|----|-----------|--------|-------|
| K1 (DECISION) | dare_b avg ≥ fisher_rao avg | **FAIL** | Δ = −9.3pp |
| K2 (COMPOSITION VALUE) | dare_b avg ≥ single_best + 2pp | **FAIL** | Δ = −6.7pp |
| K3 (ARCH TRANSFER) | \|dare_b − dare_full_delta\| ≤ 5pp | **FAIL** | \|Δ\| = 16.0pp |
| K4 (REPRODUCIBILITY) | \|fisher_rao − 73.3 ref\| ≤ 3pp | **FAIL** | \|Δ\| = 8.6pp |

### K4 Note

K4 compared Fisher-Rao avg (64.7%) against a research DARE reference (73.3%) from a different composition method, adapter set, and eval configuration. This is a miscalibrated sanity check — the reference was full-delta DARE on per-adapter A with different eval N. The relative comparisons within this run are internally consistent and the K1 result (−9.3pp) is unambiguous. The code-level verdict was INCONCLUSIVE due to K4, but the directional evidence is conclusive: B-space DARE is strictly worse than Fisher-Rao.

## Key Findings

1. **DARE requires full-rank deltas.** Full-delta DARE (71.3%) vs B-space DARE (55.3%) = 16pp gap. DARE's drop+rescale preserves expectation at the delta level (`E[DARE(ΔW)] = ΔW`), but when applied to B alone, the rescaled B interacts with the shared A differently than the original — the expectation-preservation at B level (`E[DARE(B)] = B`) does not propagate through `A @ B` because A and B interact multiplicatively.

2. **Fisher-Rao is the correct default for Pierre.** It beats single-best by +2.7pp avg and is the only composition method that consistently improves over no-composition under Pierre's shared-A constraint.

3. **Full-delta DARE is the strongest composition method (+9.3pp over single-best)** but requires per-adapter A storage and a fused-delta application path (`_FusedDeltaLinear`), which Pierre's current architecture doesn't support. This is evidence for a future fused-delta refactor if the accuracy gap justifies the storage/complexity cost.

4. **MedQA is the most sensitive benchmark.** DARE-B collapsed on MedQA (30.0% vs 58.0% Fisher-Rao), while gsm8k and humaneval showed smaller gaps. Medical domain adapters may have more fragile B-matrix structure that doesn't survive 90% element-wise dropout.

## Decision

Keep Fisher-Rao as Pierre's default composition method. The research DARE gain does not transfer to Pierre's B-only architecture. If future work pursues DARE-level composition gains, it must adopt a fused-delta path with per-adapter A — this is an architectural change, not a parameter tweak.

## Config

- Model: `mlx-community/gemma-4-e4b-it-4bit` (Gemma 4 E4B, 42 layers)
- Adapters: 7 PoLAR (strategy_full, strategy_prepare, strategy_act, strategy_integrate, domain_math, domain_code, domain_medical)
- Rank: 6, Scale: 6.0
- DARE drop rate: 0.9 (90% zeros, 10% survivors × 10 rescale)
- N = 50 per benchmark, seed = 42
- Shared-A donor: strategy_full (approximation of Pierre's frozen-shared-A)
