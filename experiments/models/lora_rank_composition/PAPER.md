# LoRA Rank Sensitivity for Composition Quality: Research Digest

## Hypothesis

LoRA rank constrains composition quality through a rate-distortion tradeoff:
there exists a critical rank below which composition degrades and above which
additional capacity is wasted.

**Falsifiable**: If all ranks produce composition quality within 1% of each
other, rank does not constrain composition. If orthogonality does not correlate
with rank (r-squared < 0.2), the subspace-dimensionality argument fails.

---

## What This Experiment Is

A systematic sweep of LoRA rank r in {2, 4, 8, 16, 32, 64} on the same
2-domain composition pipeline validated in exp_lora_procrustes_linear. For each
rank, the experiment:

1. Trains a joint baseline (standard GPT on both domains simultaneously)
2. Pretrains a shared base model
3. Fine-tunes rank-r LoRA adapters per domain (freezing base)
4. Computes task arithmetic composition (base + mean of deltas)
5. Computes concatenated + calibrated router composition
6. Measures orthogonality, shared fraction, effective rank, dead neuron rate

This is the first controlled rank sweep in the project. All prior experiments
used either full-rank capsule pools or fixed rank-8 LoRA.

---

## Lineage in the Arena

```
gpt
 `-- lora_gpt (LoRA adapters on MLP, rank parameterized)
      `-- lora_procrustes (shared/unique decomposition, fixed r=8)
      `-- lora_rank_composition (rank sweep: r in {2,4,8,16,32,64})
```

---

## Key References

- **LoRA** (Hu et al., ICLR 2022): foundational low-rank adaptation.
- **LoRA Learns Less and Forgets Less** (Biderman et al., 2024): full
  fine-tuning uses rank 10-100x greater than typical LoRA configurations;
  LoRA's rank constraint acts as implicit regularization.
- **InfLoRA** (Liang & Li, 2024): orthogonality constraints on LoRA for
  continual learning; enforces what we observe naturally.
- **exp_lora_procrustes_linear**: validated that LoRA decomposition is
  mathematically exact at r=8; this experiment extends across ranks.

---

## Empirical Results

### Core Metrics (3-seed aggregate, 2 domains)

| Rank | LoRA Params | TA vs Joint | CC vs Joint | Shared Frac | Cos Sim | Eff Rank | Dead Rate |
|------|-------------|-------------|-------------|-------------|---------|----------|-----------|
| 2    | 5,120       | +1.48%      | +1.09%      | 0.509       | 0.035   | 1.82     | 32.0%     |
| 4    | 10,240      | +1.74%      | +1.49%      | 0.504       | 0.017   | 2.97     | 29.4%     |
| 8    | 20,480      | +1.09%      | +0.66%      | 0.504       | 0.018   | 4.54     | 32.5%     |
| 16   | 40,960      | +1.12%      | +0.76%      | 0.509       | 0.039   | 6.02     | 26.2%     |
| 32   | 81,920      | +1.14%      | +0.40%      | 0.508       | 0.034   | 7.79     | 28.4%     |
| 64   | 163,840     | +1.04%      | +0.84%      | 0.499       | -0.003  | 8.11     | 34.0%     |

### Kill Threshold Checks

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Quality range across ranks (TA) | 0.70pp | >1pp to PASS | **KILL** |
| Orthogonality-rank correlation (r-squared) | 0.156 | >=0.2 to PASS | **KILL** |

**Both kill criteria are triggered. The hypothesis that rank constrains
composition quality is rejected at micro scale.**

---

## Analysis

### 1. Composition Quality is Rank-Invariant at Micro Scale

Task arithmetic composition quality ranges from +1.04% to +1.74% across a 32x
range in rank (r=2 to r=64). The total spread is only 0.70 percentage points.
This is well within the 1pp kill threshold.

Concat+calibrated routing shows slightly more variation (1.09pp range) but the
pattern is not monotonic: r=4 is the worst (+1.49%) and r=32 is the best
(+0.40%). There is no clear "distortion knee" where composition degrades at
low rank.

**Root cause**: The character-level names task has inherent dimensionality much
lower than even the smallest rank tested (r=2). The effective rank measurements
confirm this -- at r=64, only 8.11 effective dimensions are used.

### 2. Effective Rank Saturates Around 6-9

This is the most informative finding. The effective rank (Shannon entropy of
normalized singular values) grows sublinearly with nominal rank and saturates:

| Nominal r | Effective r | Utilization |
|-----------|-------------|-------------|
| 2         | 1.82        | 91%         |
| 4         | 2.97        | 74%         |
| 8         | 4.54        | 57%         |
| 16        | 6.02        | 38%         |
| 32        | 7.79        | 24%         |
| 64        | 8.11        | 13%         |

The ratio effective/nominal decreases monotonically. At r=64, 87% of the
rank capacity is wasted. The task's inherent dimensionality is approximately
8 -- the effective rank at r=64 converges to this ceiling.

This has a clear information-theoretic interpretation: the task has about 8
bits of structure that LoRA adapters can learn. More rank cannot extract more
signal because there is no more signal to extract.

### 3. Orthogonality is Rank-Independent

Cosine similarity between domain deltas ranges from -0.039 to +0.035 with no
systematic trend (r = -0.395, r-squared = 0.156). All values are near zero.

This confirms that orthogonality is a property of the task geometry and
optimization dynamics, not of the adapter rank. The deltas occupy nearly
orthogonal subspaces regardless of how large those subspaces are. This is
consistent with the InfLoRA finding that independent training naturally
produces near-orthogonal adapters.

### 4. Shared Fraction is Locked at ~50%

The shared fraction (||dW_shared|| / (||dW_shared|| + ||dW_unique||)) is
remarkably stable at 0.499-0.509 across all ranks. At N=2 with near-orthogonal
deltas, this is mathematically predicted (see MATH.md Section 6): when the
cross term is near zero, the shared and unique norms are approximately equal.

### 5. Dead Neuron Rate is Rank-Independent

Dead neuron rate ranges from 26.2% to 34.0% with no systematic trend. The
composition operation's effect on neuron liveness is determined by the
magnitude and direction of the delta relative to the ReLU threshold, not by
the rank of the delta.

### 6. Delta Norm Decreases with Rank (Counterintuitive)

Average delta norm across seeds:

| Rank | Avg Delta Norm |
|------|----------------|
| 2    | 15.62          |
| 4    | 14.04          |
| 8    | 12.28          |
| 16   | 12.09          |
| 32   | 11.80          |
| 64   | 10.40          |

Higher rank adapters produce smaller deltas. This is because the alpha/r
scaling factor decreases with rank (we use alpha=1.0 throughout). At r=2,
each parameter has 5x the gradient multiplier as at r=64. The net effect
is that lower-rank adapters make larger, coarser corrections while
higher-rank adapters make smaller, more distributed corrections.

This also explains why composition quality is similar across ranks: the
"distortion" from merging deltas depends on both the rank structure AND the
magnitude. Lower-rank deltas are larger (more distortion risk) but simpler
(less interference). These effects approximately cancel.

---

## What We Learned

### The Task Has Inherent Dimensionality ~8

The most valuable finding is not about composition quality (which is boring
-- it is rank-invariant) but about effective rank saturation. The character-
level names domain pair has approximately 8 dimensions of learnable structure.
This is a fundamental property of the task, not of the architecture.

**Implication for macro**: Real tasks (code vs prose, different languages)
likely have much higher inherent dimensionality. The rank sensitivity that
is absent at micro scale may appear at macro scale when the task demands more
dimensions than the adapter provides. The experiment should be repeated with
diverse, high-dimensional domains.

### Rate-Distortion Tradeoff is Degenerate at Micro Scale

The predicted rate-distortion curve (composition quality vs rank) is flat
because even the smallest rank (r=2) exceeds the task's information content.
This is analogous to compressing a 100-byte file: whether you use a 200-byte
or 10,000-byte container makes no difference.

### Composition Quality is Dominated by Base Training, Not Adapter Rank

The 1-2% gap between composed and joint training is stable across all ranks.
This gap comes from the composition procedure itself (task arithmetic or
routing), not from adapter capacity. The bottleneck is the composition
mechanism, not the adapter information budget.

---

## Micro-Scale Limitations

1. **Task dimensionality too low.** The character-level names task has inherent
   dimensionality ~8. Real NLP tasks likely require much higher effective rank,
   which means the rank sensitivity absent here may emerge at macro scale.

2. **Only N=2 domains.** With more domains, shared/unique decomposition
   becomes non-trivial, and rank may constrain the decomposition quality.

3. **Fixed alpha=1.0.** The alpha/r scaling interacts with rank. A sweep with
   alpha=r (constant effective scale) would isolate the pure rank effect.

4. **Short training (300 steps).** Longer training may fill higher-rank
   subspaces more completely, revealing rank sensitivity.

5. **Similar domains.** a-m vs n-z names share most structure. With truly
   distinct domains, adapters may need more dimensions to capture
   domain-specific knowledge.

---

## What Would Kill This

### At Micro Scale (already triggered)
- **Rank has no effect on composition quality**: CONFIRMED. All ranks within
  0.70pp. The hypothesis is killed at this scale.
- **Orthogonality does not correlate with rank**: CONFIRMED. r-squared = 0.156.

### Why This Kill is Informative, Not Terminal

The kill tells us that micro-scale experiments cannot test rank sensitivity
because the task is too low-dimensional. This is itself a finding: it sets
a lower bound on the complexity needed to observe rank effects.

**Prediction for macro scale**: With BPE tokens, d=512+, and diverse domains
(code vs prose), the effective rank of adapters will be much higher, and rank
sensitivity will emerge. Specifically:
- Tasks with effective rank >> 8 will show a composition quality "knee"
- The knee position will correlate with domain dissimilarity
- Below the knee, composition quality will degrade sharply

### At Macro Scale
- If rank sensitivity is also absent at d=512 with diverse domains, then LoRA
  rank truly does not constrain composition, and the rate-distortion framing is
  wrong.
- If effective rank at macro scale also saturates at ~8, the task dimensionality
  hypothesis would need revision.

---

## Artifacts

- `micro/models/lora_rank_composition/` -- experiment code, MATH.md, PAPER.md
- Parent model: `lora_gpt` (LoRA-augmented GPT)
- Reuses all infrastructure from `lora_procrustes`
- Ranks tested: {2, 4, 8, 16, 32, 64}
- Seeds: {42, 123, 7}
- Total experiment time: ~20 minutes (18 runs)
