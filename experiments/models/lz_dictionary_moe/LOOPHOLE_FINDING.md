# Loophole Finding: LZ Dictionary MoE

## 1. Vacuous Composition (MoE Bypass)
The paper claims success because "All dictionary entries are used (100% utilization, entropy ~1.0)." However, a normalized entropy of ~0.999 means that the `alpha` weights are almost perfectly uniform. If every expert uses an identical, uniform weighting of the dictionary entries, the dictionary component `sum_j alpha_{i,j} * dict_j(x)` is mathematically identical for all experts. This means the MoE router is physically bypassed for the dictionary component; it acts purely as a shared dense trunk. The "utilization" is just an artifact of a uniform average, not dynamic routing or composition. 

## 2. Weak Baseline
The paper claims a victory because Dict MoE (small) with 236K params beats Standard MoE with 596K params. However, the results show that Dense GPT (202K params) achieves ~0.5118 loss, which is *better* than the Standard MoE (0.5148). Beating a baseline that is actively worse than a simpler, smaller dense model is a meaningless comparison. Dict MoE (small) is effectively just acting as a slightly larger dense model (since its MoE aspect is collapsed), explaining why it performs similarly to Dense GPT.

## Verdict
**Invalid.** The experiment relies on a mathematically vacuous composition (uniform averaging) that bypasses the MoE structure, and justifies its performance against a broken baseline that is worse than a simple dense network.
