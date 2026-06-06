# LEARNINGS — exp_p1_t4_tfidf_routing_v2

## Core Finding
TF-IDF + Ridge regression routing is production-viable at N=25 domains: 96.0% accuracy at N=5, 84.2% at N=25 with hard negatives, 0.388ms p99 latency — all pass kill criteria with margin.

## Why
TF-IDF encodes domain vocabulary distributions as sparse high-dimensional vectors with sufficient linear separability for Ridge classification. Genuine domain confusions (legal↔jurisprudence) top out at 5%, confirming that vocabulary divergence is strong enough to route correctly across 25 domains without embedding-level representations.

## Impossibility Constraint Discovered
When two domain labels map to the same source data (same MMLU subject under different names), no linear router can separate them — this is ill-posed by construction (X_i = X_j but y_i ≠ y_j). The medical/clinical_knowledge alias (78.8% confusion) is a labeling design bug, not a router failure. **Rule: each domain label must correspond to a genuinely distinct dataset.**

## Implications for Next Experiment
- Routing is solved for P1 T4. Move to `exp_p1_t4_serving_v2` (adapter serving with real router + graph recompilation loophole fix).
- For future domain expansion beyond N=25: track domain-label uniqueness at registration time to prevent aliasing; the router cannot compensate for label design errors.
- Ridge bound at small K is vacuous (concentration inequality doesn't tighten until n >> K²). Empirical measurement is the right approach; don't rely on this bound for guarantees.
