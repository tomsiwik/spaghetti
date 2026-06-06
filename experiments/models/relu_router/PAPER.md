# ReLU Router: Research Digest

## Hypothesis

Independently-trained ReLU MLPs can be composed by weight concatenation
(A vertically, B horizontally), producing mathematically exact sum-of-pools
output, and this composition is practically useful for multi-domain LM.

**Falsifiable**: If zero-shot composition degrades val loss > 5% vs joint
training, the composition protocol does not work without calibration.

---

## What This Model Is

`ReLURouterGPT` IS a standard two-layer ReLU MLP (y = B @ ReLU(A @ x)).
The architecture has zero novelty -- it is parameter-identical to the
dense GPT baseline.

The contribution is the COMPOSITION PROTOCOL: domain-specific MLPs
trained from a shared base can be composed by concatenating weight
matrices. The composed output is mathematically the exact sum of
individual pool outputs: Pool_composed(x) = Pool_A(x) + Pool_B(x).

This is NOT "routerless self-routing." Calling ReLU activations "routing
decisions" is a change of terminology applicable to any ReLU MLP.

---

## Lineage in the Arena

```
gpt  ->  moe  ->  capsule_moe  ->  relu_router
                                    (same MLP architecture,
                                     composition protocol tested)
```

---

## Key References

**ReMoE: Fully Differentiable MoE with ReLU Routing** (ICLR 2025)
ReLU applied to router logits outperforms softmax+top-k. Uses adaptive L1
for sparsity control. Differs from our work: ReMoE keeps separate experts.

**Union-of-Experts: Experts in MoE are Secretly Routers** (2024)
Internal activation magnitudes capture routing information. External
routers are redundant. Routing by activation norm outperforms external
routing on DeepSeek-MoE 16B.

**MoRAM: Mixture of Rank-1 Associative Memory Experts** (2025)
Rank-1 experts self-route via intrinsic keys. Closest prior art for the
composition concept, but operates in LoRA adapter context.

**The Lazy Neuron Phenomenon** (Li et al., 2023)
~50% of ReLU neurons are inactive for any given input in trained
transformers. This natural sparsity is structural, not a training artifact.

---

## Empirical Results

### Single-Domain (names dataset, 500 steps, 3 seeds)

| Model | Params | Val Loss (mean) | Val Loss (std) |
|-------|--------|-----------------|----------------|
| gpt | 202,112 | 0.5159 | 0.0045 |
| capsule_moe | 203,136 | 0.5121 | 0.0059 |
| **relu_router** | **202,112** | **0.5137** | **0.0052** |

This result is trivially expected: the ReLU Router IS the dense GPT
baseline under a different name. The 0.3% gap vs capsule_moe is noise.

### Composition Experiment (3 seeds, all methods under identical conditions)

| Method | Avg Val Loss | Std | vs relu_joint | vs cap_joint |
|--------|-------------|-----|---------------|--------------|
| relu_joint (baseline) | 0.5247 | 0.0078 | -- | -0.0% |
| relu_zero_shot | 0.5512 | 0.0076 | +5.0% | +5.0% |
| relu_scalar_cal | 0.5480 | 0.0082 | +4.4% | +4.4% |
| relu_full_cal | 0.5182 | 0.0084 | -1.2% | -1.3% |
| relu_weight_avg | 0.5350 | 0.0077 | +2.0% | +1.9% |
| cap_joint (baseline) | 0.5248 | 0.0065 | +0.0% | -- |
| cap_zero_shot | 0.6133 | 0.0835 | +16.9% | +16.9% |
| cap_calibrated | 0.5243 | 0.0150 | -0.1% | -0.1% |

Protocol: pretrain base 300 steps on all data, fine-tune MLP/capsule
weights per domain 200 steps (attention frozen), compose, evaluate.
128 capsules per domain (256 composed). Calibration: 100 steps.

### Key Findings

**1. Zero-shot composition is borderline KILLED (+5.0%).**
The +5.0% degradation is right at the 5% kill threshold across 3 seeds.
The previous experiment showed +6.6% with different random seeds.
Zero-shot composition does not reliably work.

**2. Scalar calibration barely helps (+4.4% vs +5.0%).**
Training only 1 scaling factor per pool per layer (8 total parameters)
reduces degradation by only 0.6 percentage points. This disproves the
hypothesis that the "loudness problem" is the main barrier. The issue
is deeper: the sum of two MLPs produces outputs in a distribution that
downstream layers were not trained to handle.

**3. Full capsule calibration works (-1.2%) but is unfair.**
Training all capsule weights on joint data for 100 steps at 0.1x LR is
effectively continued joint training. The -1.2% result is unsurprising:
any continued training on the correct distribution improves loss.

**4. Weight averaging outperforms concatenation for zero-shot (+2.0% vs +5.0%).**
Simple weight averaging (A_avg = (A_1 + A_2) / 2) is BETTER than the
proposed concatenation for zero-shot composition. This is a significant
challenge: the mathematically exact concatenation identity does not
translate to practical superiority when the downstream layers are not
calibrated for the sum of two pools. Weight averaging keeps the output
in the same magnitude range as a single pool.

**5. Capsule_moe zero-shot is much worse (+16.9%) but calibration recovers (-0.1%).**
Without router calibration, capsule_moe composition fails badly (uniform
routing over 8 groups with a random router). With 100-step router
calibration (training only router weights), it recovers to -0.1% vs
joint. This is the strongest composition result, but it requires a
router -- exactly what relu_router tried to eliminate.

**6. Both joint baselines are equivalent (0.5247 vs 0.5248).**
relu_router and capsule_moe produce identical joint training loss,
confirming that the architectures are equivalent for single-model training.

---

## Activation Statistics

| Layer | Sparsity | Mean Freq | Min Freq | Max Freq | Dead |
|-------|----------|-----------|----------|----------|------|
| 0 | 50.0% | 0.502 | 0.027 | 0.967 | 0 |
| 1 | 49.3% | 0.507 | 0.030 | 0.960 | 0 |
| 2 | 51.9% | 0.481 | 0.016 | 0.984 | 0 |
| 3 | 47.6% | 0.524 | 0.013 | 0.979 | 0 |

Natural ReLU sparsity is ~50%. The adaptive L1 mechanism does not push
this further at micro scale. The sparsity control is effectively inert
during the 500-step training used in all experiments.

---

## Micro-Scale Limitations

1. **The model IS a dense MLP.** At micro scale with P=4d, the ReLU Router
   is parameter-identical to the dense GPT baseline. "Self-routing" is a
   relabeling, not an architectural innovation.

2. **Sparsity control is untested.** The adaptive L1 mechanism never pushes
   sparsity beyond ~50% natural ReLU sparsity. Whether it would work at
   larger scale with more training steps is unknown.

3. **Same domains.** a-m vs n-z names share character distributions. True
   domain diversity (code vs prose) would test composition more meaningfully.

4. **Scalar calibration test may need more steps or different LR.**
   The 100-step scalar calibration uses manual SGD on 8 parameters. The
   learned scales are all close to 1.0 (0.95-1.01), suggesting either the
   loudness problem is small OR the calibration needs more optimization.

5. **Weight averaging baseline uses same-size model.** The averaged model
   has P capsules (same as individual), while concatenation doubles to 2P.
   At 2P, the model has more capacity which should help, making the +5.0%
   degradation even more concerning.

---

## What Would Kill This

### At Micro Scale (tested)

- **Zero-shot composition > 5%.** BORDERLINE KILLED. At +5.0%, zero-shot
  composition is at the kill threshold. With different seeds the result
  varies from +5% to +7%. Zero-shot concatenation does not reliably work.

- **Weight averaging beats concatenation.** CONFIRMED at micro scale.
  Weight averaging (+2.0%) outperforms concatenation (+5.0%) for
  zero-shot composition. The concatenation protocol does not justify
  its 2x parameter increase.

### At Macro Scale (untested)

- **Weight averaging scales better than concatenation.** If task
  arithmetic / TIES-merging techniques consistently outperform
  concatenation at scale, there is no reason to use concatenation.

- **Sparsity control fails at scale.** Without achieving >50% sparsity,
  there are no compute savings. The model is a dense MLP.

- **Concatenation does not scale.** N domains = N*P parameters per layer.
  This is linear scaling in domain count, worse than capsule_moe's top-k
  routing which limits compute to k groups regardless of N.

---

## Honest Assessment

The hypothesis -- that concatenation of independently-trained ReLU MLPs
produces useful multi-domain models without calibration -- is not
validated at micro scale. The mathematical identity (Pool_composed =
Pool_A + Pool_B) is correct and numerically verified, but mathematical
correctness does not imply practical utility.

The key lesson: **the composition bottleneck is not routing but distribution
mismatch.** Downstream layers expect outputs from a single pool. The sum
of two pools, even if mathematically exact, produces out-of-distribution
inputs for those layers.

Calibrated capsule_moe composition (-0.1% with 100-step router calibration)
remains the strongest composition result in the project. The router,
far from being overhead, provides the mechanism for rebalancing composed
groups.

### What This Experiment Contributed

1. **Numerically verified the composition identity.** Pool_composed(x) =
   Pool_A(x) + Pool_B(x) with < 3e-7 error.

2. **Showed that loudness is NOT the main problem.** Scalar calibration
   (+4.4%) barely helps vs zero-shot (+5.0%). The issue is deeper.

3. **Established weight averaging as a strong baseline.** Any future
   composition method must beat simple weight averaging (+2.0%).

4. **Confirmed capsule_moe router calibration works.** -0.1% vs joint
   with only router parameter training. The router earns its 0.5%
   parameter overhead.
