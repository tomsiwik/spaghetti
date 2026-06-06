# InfLoRA: LoRA-Based Continual Learning

**Source:** https://arxiv.org/abs/2404.00228 (2024)

**Key Insight:** Interference-free LoRA for continual learning. Each task gets
its own LoRA adapter with orthogonality constraints to prevent interference.
Achieves ~86% final accuracy / ~3% forgetting on Split CIFAR-100.

**Relevance to our work:**
- Their orthogonality constraints during training address our function-space
  gap problem (`exp11_training_time_compat`)
- Our LoRA deltas are naturally orthogonal (cos ~ 0.000 at N=2), but InfLoRA
  enforces it explicitly — could improve scaling to many domains
- Their per-task adapter accumulation is similar to our contribution protocol
- Relevant to `exp11_training_time_compat` and `exp_ortho_diagnostic`

**What to use:**
- Their orthogonality constraint formulation
- Their continual learning evaluation protocol
- Their interference measurement methodology
