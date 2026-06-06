# PAPER.md — M2P on Qwen3-4B + GSM8K

## Objective
To scale the M2P (Map-to-Parameters) mechanism from Qwen3-0.6B to the larger Qwen3-4B model and verify if the hidden-state-to-adapter representation mapping continues to function mathematically and empirically at a 4B parameter scale.

## Predictions vs. Measurements

| Criterion | Predicted | Measured | Outcome |
|-----------|-----------|----------|---------|
| K937: M2P quality ≥ 60% SFT | PASS | quality_ratio = -0.125 (M2P worse than base) | **FAIL** |
| K938: Adapter generation < 100ms | PASS | 37.2ms (5-run avg) | **PASS** |
| K939: M2P quality < 20% (KILL) | NO KILL | quality_ratio = -0.125 | **KILL** |
| Theorem 2: grad_norm > 0 at step 0 | PASS | 15.6491 | **PASS** |

## Primary Finding: M2P Architecture Collapse at 4B Scale

The experiment resulted in a **catastrophic failure** of the M2P mapping on the Qwen3-4B model. While the model gradients flowed properly at step 0 (grad_norm: 15.65) and adapter generation was well within real-time bounds (37.2ms), the mapping mechanism entirely failed to learn useful B-matrices.

- **Base Qwen3-4B accuracy:** 65.0%
- **SFT LoRA accuracy:** 73.0%
- **M2P accuracy:** 64.0%

M2P accuracy was strictly worse than the zero-shot base model accuracy. The `quality_ratio` was -0.125. 

### Falsification of Theorem 1
Theorem 1 stated that `d_M2P = 1024` would satisfy the bottleneck sufficiency condition, allowing the adapter manifold to be representable. The failure implies one of the following:
1. **Intrinsic Dimension Miscalculation:** The `d_intrinsic` for the 4B adapter manifold may be significantly larger than for the 0.6B scale for the same task, invalidating our assumption.
2. **Trainability Degradation:** While the bottleneck capacity `d_M2P = 1024` might theoretically be sufficient, the M2P parameter size (760M params) coupled with standard SGD/Adam gradients from the LoRA head is severely ill-conditioned. Learning fails to converge to the low-dimensional adapter manifold.

## Conclusion
The current M2P architecture fails to scale. Qwen3-4B requires an architectural rethink of how B-matrices are generated, perhaps moving away from pure dense bottleneck encoders towards factored representations (e.g., VeRA-inspired scalar bottleneck routing).
