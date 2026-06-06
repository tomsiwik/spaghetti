# Union-of-Experts: "Experts in MoE are Secretly Routers"

**Source:** https://arxiv.org/abs/2410.02285 (2024)

**Key Insight:** Internal routing neurons in MoE experts align with principal
singular vectors of expert weight matrices. This means routing can bypass the
external router entirely — experts contain their own routing information.
The "A-matrix self-routing" hypothesis (killed at micro scale) was testing
a version of this idea.

**Relevance to our work:**
- Our A-matrix self-routing experiment was killed (50% = coin flip), but UoE
  operates at full scale where expert specialization is real
- Suggests that at macro scale, capsule A-matrices might contain routing signal
  after sufficient training — worth revisiting for `exp5_macro_match`
- The SVD-based analysis of expert weights could be applied to our capsule
  weight deltas as a diagnostic

**What to use:**
- Their SVD analysis methodology for detecting routing signal in expert weights
- Their comparison of self-routing vs external routing at scale
