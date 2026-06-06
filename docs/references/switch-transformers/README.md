# Switch Transformers: Scaling to Trillion Parameters

**Source:** https://arxiv.org/abs/2101.03961 (Fedus et al., 2022)

**Key Insight:** Simplified MoE with k=1 routing (each token goes to exactly
one expert). The Switch balance loss L = N * sum(f_i * p_i) prevents routing
collapse. Showed that k=1 works at scale with large enough experts.

**Relevance to our work:**
- Our sparse routing experiment (Exp 2) killed k=1 at micro scale (+200%
  degradation), but Switch uses k=1 successfully at scale
- The phase transition between k=1 and k=2 at micro scale is likely
  capacity-bound, not mechanism-bound — Switch confirms this
- Their balance loss is what we use for router calibration
- Relevant to `exp5_macro_match` routing decisions

**What to use:**
- The balance loss formulation
- Their capacity factor and expert buffer size analysis
- Their scaling analysis (what expert size is needed for k=1 to work)
