# SiLU vs ReLU Capsule Activation

## Question

Does SiLU produce better capsules than ReLU at micro scale?

Macro capsule training on Qwen2.5-Coder-0.5B produced 0% dead capsules — contradicting micro experiments showing 20-47% death under ReLU. Two explanations: (1) scale effect (d=896 vs d=64), (2) activation mismatch (Qwen uses SwiGLU but capsules use ReLU).

## Method

Minimal change: replace `nn.relu` with `nn.silu` in CapsulePool forward pass.

- **Architecture**: y = B · σ(A · x), σ ∈ {ReLU, SiLU}
- **Everything else identical**: same router (none), same composition protocol, same hyperparameters
- **Metrics adapted**: "dead" → "near-dead" (mean |activation| < 0.01), "sparsity" → "effective sparsity" (|a| < 0.01)

Composition protocol: pretrain base → fine-tune per domain → compose by weight concatenation → evaluate zero-shot, scalar calibration, full calibration, weight averaging.

3 seeds × 5 methods × 2 activations = 30 runs.

## Results

### Quality (Val Loss, 3-seed mean)

| Method | ReLU | SiLU | Delta |
|--------|------|------|-------|
| Joint training | 0.5232 | 0.5301 | +1.3% |
| Zero-shot composition | 0.5518 | 0.6017 | +9.0% |
| Scalar calibration | 0.5489 | 0.5892 | +7.3% |
| Full calibration | 0.5191 | 0.5215 | +0.5% |
| Weight averaging | 0.5318 | 0.5350 | +0.6% |

ReLU wins every comparison. The gap is small for joint/full-cal/weight-avg (<1.3%) but large for zero-shot composition (+9.0%).

### Sparsity & Death

| Metric | ReLU | SiLU |
|--------|------|------|
| Sparsity (exact/effective) | 88-92% | 3.1-7.6% |
| Dead/near-dead capsules | 138-195/512 | 0/512 |

ReLU at micro scale shows extreme sparsity (~90%) and massive death (27-38% of capsules). SiLU has near-zero effective sparsity (~3%) and zero near-dead capsules.

### Composition Degradation (vs own joint baseline)

| Method | ReLU | SiLU |
|--------|------|------|
| Zero-shot vs joint | +5.5% | +13.5% |
| Scalar cal vs joint | +4.9% | +11.1% |
| Full cal vs joint | -0.8% | -1.6% |
| Weight avg vs joint | +1.6% | +0.9% |

SiLU composition degrades 2.4x more than ReLU in zero-shot (+13.5% vs +5.5%).

## Kill Criteria

- [x] SiLU val loss > 5% worse than ReLU → **NO** (joint: +1.3%, OK)
- [x] SiLU effective sparsity < 10% → **YES** (~3%, too dense)
- [x] SiLU composition degradation > 5% worse than ReLU → **YES** (+8.0% worse)

**KILLED on criteria 2 and 3.**

## Conclusions

1. **SiLU eliminates capsule death** (0 near-dead vs 138-195 dead for ReLU) but at the cost of losing all sparsity. SiLU capsules are dense — only ~3% of activations are near-zero vs ReLU's ~90%.

2. **SiLU composition is significantly worse** than ReLU. Zero-shot composition degrades +13.5% vs joint (ReLU: +5.5%). This makes sense: ReLU's hard sparsity means each capsule operates independently (its activation depends only on its own detector vector). SiLU's smooth activations create cross-capsule dependencies that break when pools are concatenated.

3. **The 0% macro death is a scale effect, not an activation mismatch.** SiLU doesn't improve things — it just makes death unmeasurable by eliminating exact zeros. At micro scale, SiLU is strictly worse than ReLU for composition.

4. **Weight averaging is competitive for both activations** (ReLU: +1.6%, SiLU: +0.9% vs joint). This remains the most practical composition method at micro scale.

5. **Recommendation**: Keep ReLU for capsules. The macro 0% death finding is due to higher dimensionality (d=896), not activation mismatch.
