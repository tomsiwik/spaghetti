# MoRAM: Mixture-of-Experts with Rank-1 Associative Memory

**Source:** https://arxiv.org/abs/2501.13662 (2025)

**Key Insight:** LoRA adapters treated as key-value pairs in associative memory.
Self-routing via intrinsic keys eliminates router degradation in continual
learning. Each LoRA adapter is a rank-1 "memory" that self-selects based on
input similarity to its key.

**Relevance to our work:**
- Closest published work to our capsule composition protocol
- Their "intrinsic keys" solve the routing problem we face: how to route
  to independently-trained adapters without joint calibration
- If MoRAM's self-routing works for composition, it could eliminate our
  100-200 step calibration requirement
- Directly relevant to `exp11_training_time_compat` (training-time composition)

**What to use:**
- The associative memory routing mechanism
- Their evaluation of composition without calibration
- Their continual learning protocol (adapter addition without forgetting)
