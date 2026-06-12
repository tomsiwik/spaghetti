# LEARNINGS — exp_spark_layer_self_gate

**Core finding:** Aggressively restricting the off-domain (math) adapter to only 6 of 42 q_proj layers
recovers HumanEval pass@1 from a collapsed C=0.02 back to the code-solo ceiling B=0.34 (+32pp), but
best_D=B exactly and the k-sweep is monotone-in-fewer, making the result indistinguishable from "less
math adapter is simply better."

**Why:** No random-k=6 or anti-γ/bottom-k=6 control was run, so γ-based layer selection is not
separated from pure amount reduction; γ values are near-zero (mean≈+0.011) and ~25/42 layers are
positive per prompt while k=6 wins — there is no shown evidence the γ ranking carries signal past the
top few. Additionally, the code adapter itself degrades its own task (B=0.34 < A=0.44), so "ceiling
recovery" lands 10pp below base.

**Implication for the next experiment:** Before claiming layer-localization, pre-register and run a
random-k=6 (seeded) and bottom-k=6 (anti-γ) control in the same run; SUPPORTED requires top-k-by-γ
to beat both controls by a meaningful margin, and recovery must be reported against base A, not the
degraded code-solo ceiling B.
