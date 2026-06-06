# Neural Death Trajectories (Gurbuzbalaban et al., 2024)

**Source:** https://arxiv.org/abs/2402.00722 (2024)
"Maxwell's Demon at Work: Efficient Pruning in Deep Networks"

**Key Insight:** >90% of revived neurons eventually die again. Cosine decay
learning rate promotes revival vs constant LR. Death trajectory is non-monotonic —
neurons can revive mid-training then die permanently. This is the foundational
reference for our capsule death/revival experiments.

**Relevance to our work:**
- Directly informed Exp 17 (training duration), Exp 18 (capsule revival),
  LR schedule death, and warmup sweep experiments
- Our capsule revival experiment confirmed their prediction: 28.1% revival
  at S=3200, Jaccard=0.669 (death identity is fluid)
- Their cosine decay finding directly applies to macro training schedules
- Relevant to `exp17_training_duration_death` and all pruning strategies

**What to use:**
- Their non-monotonic death model (cumulative-LR-integral predictor)
- Cosine decay vs constant LR comparison methodology
- Their "re-death" tracking protocol (which we adapted for capsule revival)
