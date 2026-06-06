# DARE: Drop and Rescale for Expert Merging

**Source:** https://arxiv.org/abs/2311.03099 (2023)

**Key Insight:** Randomly drop a fraction p of delta parameters, then rescale
remaining by 1/(1-p). Similar to dropout but applied to weight deltas before
merging. Surprisingly effective — works because most delta parameters are
redundant.

**Relevance to our work:**
- Complementary to TIES for weight averaging composition
- Our dead capsule pruning (57% dead) is conceptually similar — both exploit
  parameter redundancy in fine-tuned models
- DARE's random drop could be tested against our targeted pruning
  (Exp 10 showed random pruning is competitive: -2.9% better without cal)
- Relevant to `exp5_macro_match` merging strategy

**What to use:**
- The drop-and-rescale algorithm
- Their analysis of why random dropping works (parameter redundancy)
- Combination with TIES: DARE+TIES is a common pipeline
