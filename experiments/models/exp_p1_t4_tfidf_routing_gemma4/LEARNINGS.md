# LEARNINGS: exp_p1_t4_tfidf_routing_gemma4

**Status:** SUPPORTED (Finding #431)

## Core Finding

TF-IDF nearest-centroid routing achieves 96.6% accuracy at N=5 and 86.1% at N=25 with zero
neural parameters and ~0.3ms median CPU latency — suitable for P1 production routing.

## Why It Works

TF-IDF bigrams separate domain-specific vocabulary clusters (medical: "patient/diagnosis",
code: "function/return", math: "equation/integer") via nearest-centroid classification in
sparse feature space. This is the same principle as Finding #354/389 but now validated at
production NLP scale (N_TRAIN=300, N_TEST=100 per domain).

## Key Numbers

- N=5: 96.6% (weakest: finance 91% — economics↔quantitative vocabulary overlap)
- N=25: 86.1% — margin is thin (1.08pp above threshold)
- Confusion floor ~74%: finance/prehistory/astronomy/world_religions share MCQ vocabulary
- p50=0.30ms, p99=1.11ms (Python GIL jitter; ~1/20th of first-token latency at 6ms/tok)

## Implications for T4.2 (LSH Routing)

1. **Report confidence intervals** — K1074 margin was only 1.08pp; need CIs to trust the comparison
2. **Low-data ablation required** — test N_TRAIN=50 (niche domains may have sparse training data)
3. **Finance confusion is structural** — TF-IDF won't fix finance↔statistics; LSH must be
   evaluated against this floor to determine if it improves topically adjacent domains
4. **Scale=6 is the safe adapter operating point** (Finding #426) — routing accuracy figures
   assume scale=6 adapters, do not increase scale in T4.2 experiments
