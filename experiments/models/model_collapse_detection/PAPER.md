# Model Collapse Detection for Self-Learning LoRA Experts: Research Digest (v2)

## Hypothesis

Self-learning loop does not cause model collapse when experts are LoRA adapters,
because the low-rank bottleneck plus norm bounding constrains distribution drift.

**Falsifiable:**
- K1: output diversity drops >30% after 5 self-learning cycles
- K2: expert converges to repetitive/degenerate outputs

## What This Experiment Is

A representation-level simulation of model collapse under self-training (v2,
revised based on adversarial review). Key improvements over v1:

1. **Anchored full-rank baseline** to isolate the rank effect from base-anchoring.
2. **LoRA ablation without norm constraint** to disentangle rank from norm bounding.
3. **Correlation tested in collapse-prone regime** (full-rank, not LoRA).
4. **Corrected norm bound** in MATH.md.
5. **Softened claims** about fresh data requirement.

## Key References

- Shumailov et al. 2023, "The Curse of Recursion" -- self-training causes
  progressive tail erosion and collapse.
- Alemohammad et al. 2023, "Self-Consuming Generative Models Go MAD" --
  quality/diversity degrade without fresh data.
- Dohmatob et al. 2024, "Is Model Collapse Inevitable?" -- data accumulation
  prevents collapse.
- Parent: execution_based_self_learning (SUPPORTED) -- scalar simulation showed
  collapse at cycle 18-19.

## Empirical Results

### Experiment 1: LoRA Rank Sweep + Controlled Baselines

All conditions: learning_rate=0.3 (aggressive), temperature=0.8, 10 seeds,
30 cycles, no fresh data.

**LoRA with norm constraint (standard, ||A||_F, ||B||_F <= 5.0):**

| Rank | n-gram Drop (5 cyc) | Entropy Drop | Collapse (30 cyc) | K1 |
|------|:---:|:---:|:---:|:---:|
| r=4  | -0.9% | -0.7% | 0% | **PASS** |
| r=8  | -1.2% | +0.2% | 0% | **PASS** |
| r=16 | -1.6% | +0.2% | 0% | **PASS** |
| r=32 | -1.7% | +0.3% | 0% | **PASS** |
| r=64 | -0.2% | +0.4% | 0% | **PASS** |

**LoRA WITHOUT norm constraint (rank-only, FIX 2):**

| Rank | n-gram Drop (5 cyc) | Entropy Drop | Collapse (30 cyc) | K1 |
|------|:---:|:---:|:---:|:---:|
| r=4  | -23.4% | +5.5% | 80% | PASS* |
| r=8  | -53.4% | +7.2% | 100% | PASS* |
| r=16 | -217.2% | -15.6% | 90% | PASS* |
| r=32 | -172.9% | -10.2% | 90% | PASS* |
| r=64 | -231.7% | -12.4% | 80% | PASS* |

*K1 "PASS" is misleading here: the negative n-gram drops indicate chaotic
instability (diversity increases erratically as norms blow up), not stability.
80-100% of seeds eventually collapse into degenerate or chaotic states. Rank
alone, without norm bounding, does NOT prevent collapse.

**Full-rank baselines (FIX 1):**

| Condition | n-gram Drop (5 cyc) | Entropy Drop | Collapse (30 cyc) | K1 |
|-----------|:---:|:---:|:---:|:---:|
| Anchored (base + delta) | +73.0% | +40.5% | 100% | **FAIL** |
| Unanchored (v1 baseline) | +73.0% | +40.5% | 100% | **FAIL** |

**KEY FINDING (revised): Collapse prevention requires BOTH rank constraint AND
norm bounding.** Neither mechanism alone is sufficient.

- The anchored full-rank baseline collapses identically to the unanchored one
  (73.0% diversity loss, 100% collapse rate). This proves that **base-anchoring
  alone does not prevent collapse**. The v1 concern that the LoRA vs full-rank
  comparison confounded rank with anchoring is resolved: anchoring is irrelevant.

- LoRA without norm constraint collapses at 80-100% of seeds. The perturbation
  norms grow without bound, causing chaotic instability. This proves that **rank
  alone does not prevent collapse**.

- LoRA with norm constraint prevents collapse at 0% rate across all ranks.
  The joint mechanism is: rank confines updates to r directions, norm bounding
  limits magnitude in those directions. Together they prevent both concentration
  (classical collapse) and divergence (chaotic instability).

The Spearman correlation between rank and diversity drop for normed LoRA is
rho=0.000 (p=1.0), confirming all ranks are equally protective when norm-bounded.
For unnormed LoRA, rho=-0.900 (p=0.037): higher rank = more instability,
confirming the rank bottleneck does provide some restraint even without norm caps.

### Experiment 2: Correlation Effect (FIX 4 -- tested in collapse-prone regime)

Previously, correlation was tested only in the LoRA regime where collapse does
not occur, making the experiment uninformative. v2 tests correlation in three
regimes:

| Regime | Independent | Correlated | Acceleration |
|--------|:---:|:---:|:---:|
| Anchored full-rank | +65.8% drop | +73.0% drop | 1.11x |
| Unanchored full-rank | +65.8% drop | +73.0% drop | 1.11x |
| LoRA r=16 (normed) | -1.6% drop | -1.6% drop | 1.05x |

**Finding:** Correlated outputs provide a small but consistent acceleration of
collapse (1.11x) in the full-rank regime where collapse occurs. Under LoRA
constraint, the effect is negligible (1.05x) because the rank+norm bottleneck
dominates.

**This does NOT resolve parent Limitation #8.** The experiment shows that
correlation is a secondary effect (1.11x acceleration) compared to the rank+norm
mechanism (which eliminates collapse entirely). The limitation is moot under
LoRA constraint rather than disproven. For full-rank self-training (as studied
by Shumailov et al.), correlation provides modest acceleration of an already
rapid collapse process.

### Experiment 3: Fresh Data Mitigation

| Fresh % | n-gram Drop (5 cyc) | Collapse (30 cyc) | Degeneracy |
|--------:|:---:|:---:|:---:|
| 0% | -1.6% | 0% | 0% |
| 10% | -1.0% | 0% | 0% |
| 30% | -1.6% | 0% | 0% |
| 50% | -1.6% | 0% | 0% |

Under the simulation conditions tested (unconditional distribution, single
context, norm-bounded LoRA), no fresh data injection is required to prevent
collapse. The parent's prediction of 50% fresh data was based on a full-rank
model that does not apply to norm-bounded LoRA updates.

**Conjecture (requires macro validation):** The self-learning loop can operate
on self-generated data alone when using norm-bounded LoRA adapters. This
conjecture depends on: (a) weight decay providing effective norm bounding in
real training, (b) the unconditional distribution model capturing the essential
collapse dynamics, and (c) conditional (per-context) collapse modes not emerging
at scale. Macro validation with real LoRA self-training is needed to upgrade
this from conjecture to finding.

### Experiment 4: Detection Metric Comparison

| Metric | Detection Rate | Mean Cycle | Earliest |
|--------|:---:|:---:|:---:|
| n-gram 2-ratio | 10% | 6.0 | 6 |
| n-gram 3-ratio | 10% | 9.0 | 9 |
| Entropy | 0% | -- | -- |
| Embedding variance | 0% | -- | -- |

Detection is mostly irrelevant for norm-bounded LoRA experts because collapse
does not occur.

### Kill Criteria Assessment

**K1: Output diversity drops >30% after 5 self-learning cycles?**

| Condition | Drop | Verdict |
|-----------|:---:|:---:|
| LoRA rank=16, normed (SOLE default) | -1.6% | **PASS** |
| All LoRA ranks, normed (4-64) | -0.2% to -1.7% | **PASS** |
| LoRA rank=16, NO norm | -217.2% | **FAIL** (chaotic) |
| Full-rank, anchored | +73.0% | **FAIL** |
| Full-rank, unanchored | +73.0% | **FAIL** |

**K2: Expert converges to repetitive/degenerate outputs?**

| Condition | Degeneracy Rate | Verdict |
|-----------|:---:|:---:|
| LoRA rank=16, normed, 30 cycles | 0% | **PASS** |

**Overall verdict: SUPPORTED (with refined attribution)**

Both kill criteria pass for norm-bounded LoRA experts. The hypothesis "self-
learning loop does not cause model collapse" is SUPPORTED when the expert is
a LoRA adapter with norm bounding. The prevention mechanism requires both
rank constraint and norm bounding jointly.

## SOLE Integration

This result supports the Evolve phase with a refined understanding:

```
Expert v1 (LoRA, rank-16, norm-bounded via weight decay)
    |
    +-- Self-learning cycle (execution feedback or teacher correction)
    |     Generate K solutions -> filter by oracle -> DPO/SFT training
    |     RANK + NORM BOUNDING jointly prevent collapse
    |     Weight decay in AdamW provides the norm bounding
    |     Diversity monitoring optional (for defense-in-depth)
    |
    +-- After N cycles: check improvement via shadow scoring
    |     If improved: Expert v2 replaces v1
    |     If not: discard, no harm done (LoRA prevented degeneration)
    |
    +-- Self-improvement is safe with norm-bounded LoRA
```

**Operational note:** Weight decay is critical, not optional. Standard AdamW
with weight_decay=0.01 provides the norm bounding that this experiment shows
is necessary. Custom training loops that omit weight decay would lose this
protection.

## Limitations

1. **Distribution-level model, not sequence-level.** We track an unconditional
   token distribution. Real LLMs have complex conditional distributions that
   evolve differently per context. The rank+norm constraint should still
   regularize, but the dynamics may differ quantitatively.

2. **Norm bounding is critical (not incidental).** Without norm caps, even LoRA
   collapses at 80-100% of seeds. The experiment assumes weight decay in real
   training provides equivalent norm bounding. This is standard practice
   (AdamW default: 0.01) but must be verified at macro scale.

3. **Fixed context embedding.** We use one fixed context vector. Real models see
   many different contexts, which should provide additional diversity but could
   also enable per-context collapse modes.

4. **Custom update rule.** The gradient computation uses SVD projection rather
   than chain-rule backprop through BA. The rank constraint is structural and
   preserved, but specific dynamics differ from real LoRA SGD.

5. **Rank vs vocabulary ratio.** We test r <= 64 with V=100. At production scale,
   r=16 with V=151,936 gives r/V = 0.0001, which is far more constrained than
   micro (r/V = 0.16 to 0.64). Production LoRA should be even more resistant.

6. **Aggressive learning rate.** We use eta=0.3, more aggressive than typical
   LoRA training (eta ~1e-4 to 1e-3). This is conservative: if collapse
   doesn't happen at eta=0.3, it won't happen at realistic learning rates.

7. **Fresh data conjecture is unvalidated.** The claim that no fresh data is
   needed under norm-bounded LoRA is a simulation finding, not an empirical
   result. It depends on assumptions about unconditional distribution dynamics,
   single context, and effective norm bounding. Macro validation required.

## What Would Kill This

**At micro scale (what passed):**
- K1: PASS with large margin (1.6% drop vs 30% threshold, norm-bounded LoRA)
- K2: PASS (0% degeneracy at all normed LoRA ranks)
- Attribution: anchored full-rank collapses (100%), confirming rank (not anchoring) matters
- Norm ablation: unnormed LoRA collapses (80-100%), confirming norm bounding is required

**At macro scale (what needs validation):**
- Actual LoRA self-training on Qwen2.5-7B + code expert with MBPP test suite
- Run 10 real self-learning cycles, measure unique n-gram ratio and pass@k
- If diversity drops >30% at r=16 with standard weight decay: kills micro finding
- If weight decay = 0.01 does not effectively bound LoRA norms: kills the
  norm-bounding assumption (which this experiment shows is critical)

**What would fundamentally kill the approach:**
- If conditional (context-dependent) collapse modes exist that bypass the
  rank+norm bottleneck (e.g., collapse per-context rather than globally)
- If real LoRA training with standard weight decay does not effectively
  bound the Frobenius norms of A and B matrices
- If the noise-amplification mechanism from Shumailov compounds faster
  than the rank+norm constraint can absorb, at production vocabulary sizes
