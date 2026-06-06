# ReDo: Reinitializing Dead Neurons

**Source:** https://arxiv.org/abs/2405.14540 (2024)

**Key Insight:** Activation-based profiling identifies dead neurons, then
reinitializes their weights to recover capacity. Includes optimizer state
reset for reinitialized neurons. Closest published analog to our dead
capsule pruning — but they reinitialize instead of pruning.

**Relevance to our work:**
- Our dead capsule pruning (Exp 9) takes the opposite approach: prune dead
  neurons for compression instead of reinitializing for recovery
- ReDo's profiling methodology validated our activation-frequency approach
- Their optimizer state reset insight is relevant: our capsule revival
  experiment (Exp 18) found 28.1% revival rate, suggesting reinitialization
  could recover even more capacity
- Relevant to pruning strategy for `exp5_macro_match`

**What to use:**
- Their activation profiling implementation (hooks-based, not forward-pass duplication)
- The optimizer state management for reinitialized neurons
- Their analysis of when to reinitialize vs when dead is permanent
