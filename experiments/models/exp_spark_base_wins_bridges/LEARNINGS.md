# LEARNINGS — exp_spark_base_wins_bridges

**Core finding.** Per-token entropy-argmin routing over {frozen base, math adapter, code adapter} achieves EM 0.6167 vs best-single 0.4333 (base-alone), a +18.33pp lift that clears the kill-2311 +3pp bar — the arc's first composition-beats-best-single result with a base-alone control in place.

**Why.** The frozen base is lower-entropy than scale-6 perturbed adapters on most tokens, so it wins 76.6% of routing decisions; adapters fire on their confident domain spans (23.4% of tokens). The original bridge-token mechanism hunch is refuted — bridges are only 17.4% of base wins; the base dominates generic scaffold/prose globally, not at domain-crossing pivots.

**Implication for the next experiment.** Before treating this as a durable arc result, two follow-on runs are required: (1) replicate across ≥2 additional seeds and ≥1 genuinely two-domain task (seed variance and task generality are uncharacterized); (2) run a compute-matched comparison — three single-arm forward passes vs one router pass — to determine whether the +18.33pp win survives or collapses to a 3x-compute artifact.
