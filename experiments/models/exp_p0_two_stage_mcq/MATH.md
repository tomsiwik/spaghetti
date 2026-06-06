# Two-Stage Training: Sequential Spectral Reorganization Under TT Compression

## Type
Guided exploration — proven framework (TT-LoRA SVD, gradient concentration), unknown parameter (achievable accuracy under sequential optimization).

## Motivation

Finding #522 established that MCQ classification loss recovers +14.5pp under TT-LoRA r6
(20.0% → 34.5%), but MCQ loss converged only to 1.261 (vs 0 optimal). The ceiling at ~35%
suggests rank-6 capacity is saturated under *simultaneous* NTP+MCQ optimization.

**Question:** Can sequential optimization (NTP first, then MCQ-only) exceed this ceiling
by eliminating gradient competition for the same rank-6 capacity?

## Theorem 1: Gradient Competition Under Joint Optimization

**Statement.** In joint training with loss L = L_NTP + λL_MCQ, the gradient update
ΔW at each step is:

$$\Delta W = -\eta(\nabla_W L_{NTP} + \lambda \nabla_W L_{MCQ})$$

For TT-LoRA with rank r, the effective update is projected onto the top-r singular
directions of the TT reconstruction. When both gradients are active simultaneously,
they compete for the same r directions: NTP gradient pushes toward language-modeling
singular values, MCQ gradient pushes toward discriminative singular values.

**Proof.** Let $\Sigma = \text{diag}(\sigma_1, ..., \sigma_r)$ be the effective singular
values of the TT reconstruction $\Delta W$. The NTP gradient $g_{NTP}$ has high effective
rank (spreading across many directions — 256K vocabulary). The MCQ gradient $g_{MCQ}$ has
low effective rank (concentrating on 4-class separation — rank ≤ 3).

Under joint optimization, at equilibrium:
- The top singular directions balance NTP and MCQ contributions
- Neither objective gets optimal allocation of the rank-r budget
- MCQ loss converges to $L^*_{MCQ} > 0$ because the discriminative subspace
  cannot be fully represented in the top-r directions

Finding #522 measured: $L^*_{MCQ} = 1.261$ (vs $\log(4) = 1.386$ random, vs 0 optimal).
This implies only partial discriminative representation. **QED.**

## Theorem 2: Sequential Optimization Allows Full Spectral Reorganization

**Statement.** In two-stage training:
- Stage 1: minimize $L_{NTP}$ for $T_1$ steps → singular spectrum $\Sigma_1$ dominated by NTP directions
- Stage 2: minimize $L_{MCQ}$ only for $T_2$ steps → gradient rotates spectrum toward discrimination

During Stage 2, with no NTP gradient competing, the MCQ-only gradient can reorganize
the full rank-r budget toward discriminative features.

**Proof sketch.** After Stage 1, the TT cores encode medical knowledge (NTP loss ~0.2).
Stage 2 MCQ-only gradient:

$$\Delta W_{stage2} = -\eta \nabla_W L_{MCQ}$$

With rank $r = 6$ and 4-class MCQ needing rank-3 discriminative subspace:
- Stage 2 can allocate 3 of 6 directions to discrimination (50% budget)
- vs joint training where NTP gradient continuously pulls these directions back
- The remaining 3 directions retain NTP knowledge (catastrophic forgetting bounded
  by the orthogonal complement)

**Key prediction:** Stage 2 MCQ loss should converge below 1.261 (the joint ceiling)
because the MCQ gradient faces no competition.

**NTP knowledge retention:** Some NTP degradation is expected (Stage 2 overwrites
some NTP-optimal directions). This is the desired tradeoff — the question is whether
the MCQ gain exceeds the NTP loss in behavioral terms.

## Theorem 3: MCQ-Only From Scratch Lacks Knowledge Foundation

**Statement.** MCQ classification loss alone (without NTP pretraining) cannot build
medical knowledge representations — it only learns answer letter mapping.

**Proof sketch.** MCQ loss $L_{MCQ} = -\log p(y_{correct} | x)$ over 4 classes provides
gradient only at the answer position token. The gradient:
- Has no signal about medical reasoning (why A vs B)
- Only learns statistical correlations between input features and answer position
- With 4 classes and random initialization, converges to memorization of
  surface patterns (keyword → answer mapping)

**Prediction:** MCQ-only from scratch achieves 25-33% (near random, slightly above
through surface pattern memorization), significantly below two-stage (38%+).

## Quantitative Predictions

| Condition | Predicted MedMCQA | Reasoning |
|---|---|---|
| Base (no adapter) | 29-33% | Prior: 30.5% (Finding #521, #522) |
| TT-LoRA r6 NTP-only | 18-22% | Prior: 18.5-20.0% (Finding #521, #522) |
| TT-LoRA r6 Two-Stage (NTP→MCQ) | 38-45% | Stage 2 reorganizes spectrum without competition |
| TT-LoRA r6 MCQ-only from scratch | 25-33% | No NTP knowledge foundation |
| Stage 2 MCQ loss | < 1.20 | Better convergence without NTP gradient interference |
| NTP→MCQ improvement over mixed | ≥ 3.5pp | Sequential > simultaneous (Theorem 2) |

## Kill Criteria (derived from predictions)

- **K1440:** Two-stage MedMCQA ≥ 38% (exceeds mixed ceiling 34.5% by ≥ 3.5pp)
- **K1441:** Stage 2 MCQ loss < 1.20 (better convergence than joint 1.261)
- **K1442:** MCQ-only from scratch < two-stage by ≥ 5pp (NTP knowledge is load-bearing)

## What Would Kill This

If two-stage does NOT exceed 34.5%:
- The ceiling is not from gradient competition but from TT rank-6 information capacity
- Rank-6 can encode at most ~35% MedMCQA regardless of training procedure
- Implication: need higher rank, not better training

If MCQ-only from scratch ≈ two-stage:
- NTP pretraining is not load-bearing for MCQ performance
- MCQ loss alone learns sufficient representations
- Implication: NTP stage is wasted computation

## References

- Finding #521 — Compression diagnosis (34pp gap)
- Finding #522 — MCQ recovery (+14.5pp) and ceiling (34.5%)
- arXiv:2504.21190 — TT-LoRA: SVD truncation preserves top-r singular directions
- arXiv:2410.21228 — Sequential LoRA: intruder dimensions from sequential fine-tuning
